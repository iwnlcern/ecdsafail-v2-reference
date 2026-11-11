"""Exact reversible integer and modular arithmetic. Registers are wire lists,
least significant bit first. Every routine returns its ancillas clean on every
input inside its stated domain."""


def _maj(C, c, b, a):
    C.cx(a, b)
    C.cx(a, c)
    C.ccx(c, b, a)


def _uma(C, c, b, a):
    C.ccx(c, b, a)
    C.cx(a, c)
    C.cx(c, b)


def _cuma(C, ctrl, c, b, a):
    C.ccx(c, b, a)
    C.ccx(ctrl, c, b)
    C.cx(a, c)
    # Left ungated because it restores b when ctrl is 0, so _cuma must not become a gated _uma.
    C.cx(a, b)


def _cx(C, ctrl, src, dst):
    if ctrl is None:
        C.cx(src, dst)
    else:
        C.ccx(ctrl, src, dst)


def add(C, a, b, cout=None, ctrl=None, cin=None):
    """b += ctrl*a mod 2^w (CDKM). cout ^= ctrl*carry out. A wire holding 1 as cin adds 1."""
    w = len(a)
    assert len(b) == w
    assert ctrl is None or ctrl not in (*a, *b, cout)
    if w == 0:
        return
    own = cin is None
    if own:
        (cin,) = C.alloc(1)
    carries = [cin] + a[:-1]
    # With no carry out and no control the top _maj and _uma cancel to b ^= a ^ c.
    top = w - 1 if cout is None and ctrl is None else w
    for i in range(top):
        _maj(C, carries[i], b[i], a[i])
    if cout is not None:
        _cx(C, ctrl, a[-1], cout)
    if top < w:
        C.cx(a[-1], b[-1])
        C.cx(carries[-1], b[-1])
    for i in reversed(range(top)):
        if ctrl is None:
            _uma(C, carries[i], b[i], a[i])
        else:
            _cuma(C, ctrl, carries[i], b[i], a[i])
    if own:
        C.free([cin])


def lookahead_sizes(w):
    """Block sizes for a w-bit lookahead chain, from the bottom: 1, 1, 2, 2, ... up to 8, and the
    same from the top down, so the blocks at both ends are short. A long block's precompute holds
    up the chain at the bottom, and its fixes and clears hold up the end of the sweep at the top."""
    bottom, top, left, b, d = [], [], w, 0, 0
    while left > 0:
        nb, nt = min(8, 1 + b // 2), min(8, 1 + d // 2)
        if nb <= nt:
            r = min(nb, left)
            bottom.append(r)
            b += 1
        else:
            r = min(nt, left)
            top.append(r)
            d += 1
        left -= r
    return bottom + top[::-1]


def lookahead_chain(C, U, V, temps, sizes):
    """f_j = x_j ^ p_j f_(j-1) in place on U for j = 1 .. len(U) - 1, from the chain input U[0],
    with U[j] holding x_j and V[j] holding p_j (a carry chain in CDKM's up-sweep form).
    For a block of r bits after U[i], first X_(i+j) = x_(i+j) ^ p_(i+j) X_(i+j-1) goes onto U[i+j]
    and the products T_j = p_(i+1) ... p_(i+j) onto temps, with no carry; then one CCX in series,
    U[i+r] ^= T_r U[i]; then U[i+j] ^= T_j U[i] for j < r, each temp cleared after its last use.
    One CCX a block is in series instead of one a bit. Temps are taken in turn and left clean; a
    block never needs more than len(temps) of them at once."""
    n = len(U)
    q = 0
    i = 0
    blk = 0
    while i < n - 1:
        r = min(sizes[min(blk, len(sizes) - 1)], len(temps) + 1, n - 1 - i)
        blk += 1
        if r == 1:
            C.ccx(U[i], V[i + 1], U[i + 1])
            i += 1
            continue
        for j in range(2, r + 1):
            C.ccx(U[i + j - 1], V[i + j], U[i + j])
        T = {1: V[i + 1]}
        for j in range(2, r + 1):
            T[j] = temps[q % len(temps)]
            q += 1
            C.ccx(T[j - 1], V[i + j], T[j])
        C.ccx(U[i], T[r], U[i + r])
        for j in range(r, 1, -1):
            C.ccx(T[j - 1], V[i + j], T[j])
            C.ccx(U[i], T[j - 1], U[i + j - 1])
        i += r


class _Gates:
    """Records cx and ccx calls, so a sweep can be emitted forward and then in reverse."""

    def __init__(self):
        self.ops = []

    def cx(self, a, b):
        self.ops.append((a, b))

    def ccx(self, a, b, c):
        self.ops.append((a, b, c))


def _emit(C, ops):
    for o in ops:
        (C.cx if len(o) == 2 else C.ccx)(*o)


def adjacent_xor(C, a):
    """a[i] ^= a[i + 1] for i < len(a) - 1, from the old values, in about 2 log2 len(a) CX layers:
    the inverse of an in-place suffix XOR done as a Brent-Kung prefix network (Brent and Kung,
    1982), so that network's CXs in reverse order. The ripple it replaces is len(a) - 1 CXs in
    series, since each shares a wire with the next."""
    n = len(a)
    ops = []
    d = 1
    while d < n:
        for p in range(2 * d - 1, n, 2 * d):
            ops.append((p - d, p))
        d *= 2
    while d > 1:
        d //= 2
        for p in range(3 * d - 1, n, 2 * d):
            ops.append((p - d, p))
    # prefix position p is suffix index n - 1 - p: a[n - 1 - p] ^= a[n - 1 - p + d]
    for s, t in reversed(ops):
        C.cx(a[n - 1 - s], a[n - 1 - t])


def sub(C, a, b, borrow=None, ctrl=None):
    """b -= ctrl*a mod 2^w. borrow ^= ctrl*[a > b]."""
    for w in b:
        C.x(w)
    add(C, a, b, borrow, ctrl)
    for w in b:
        C.x(w)


def gt_into(C, a, b, t, ctrl=None):
    """t ^= ctrl*[a > b], a and b restored. Carry out of (~b + a) is [a > b]."""
    assert ctrl is None or ctrl not in (*a, *b, t)
    w = len(a)
    if w == 0:
        return
    (cin,) = C.alloc(1)
    for x in b:
        C.x(x)
    carries = [cin] + a[:-1]
    for i in range(w):
        _maj(C, carries[i], b[i], a[i])
    _cx(C, ctrl, a[-1], t)
    for i in reversed(range(w)):
        C.ccx(carries[i], b[i], a[i])
        C.cx(a[i], carries[i])
        C.cx(a[i], b[i])
    for x in b:
        C.x(x)
    C.free([cin])


def load_const(C, reg, value, ctrl=None):
    for i, w in enumerate(reg):
        if value >> i & 1:
            if ctrl is None:
                C.x(w)
            else:
                C.cx(ctrl, w)


def _carry_into(C, b, bit, c, t, ctrl=None, low=None):
    """t ^= MAJ(b, ctrl*bit, c), where c is a wire or None for 0. With ctrl, c
    must imply ctrl, which holds for every carry of reg + ctrl*value. low(t), if
    given, XORs ctrl*b into t, for the one b that meets the first set bit."""
    if not bit:
        if c is not None:
            C.ccx(b, c, t)
    elif c is None:
        if low is None:
            _cx(C, ctrl, b, t)
        else:
            low(t)
    elif ctrl is None:
        C.ccx(b, c, t)
        C.cx(b, t)
        C.cx(c, t)
    else:
        # MAJ(b, ctrl, c) = c ^ b*(ctrl ^ c) when c implies ctrl.
        C.cx(c, ctrl)
        C.ccx(b, ctrl, t)
        C.cx(c, ctrl)
        C.cx(c, t)


def _carries(C, reg, value, ctrl=None, low=None):
    """cs[i] is the carry into reg[i] of reg + ctrl*value. Each entry is None
    when the carry is always 0, reg[i - 1] itself when it equals that bit, or a
    fresh wire. Release the fresh wires with _uncarry."""
    cs = [None]
    for i in range(len(reg) - 1):
        bit, c = value >> i & 1, cs[i]
        if c is None and not bit:
            cs.append(None)
        elif c is None and ctrl is None:
            cs.append(reg[i])
        else:
            (t,) = C.alloc(1)
            _carry_into(C, reg[i], bit, c, t, ctrl, low)
            cs.append(t)
    return cs


def _uncarry(C, reg, value, cs, i, ctrl=None, low=None):
    """Clear and free cs[i]. reg[i - 1] and cs[i - 1] must still hold what they held in _carries."""
    if i and cs[i] is not None and cs[i] != reg[i - 1]:
        _carry_into(C, reg[i - 1], value >> (i - 1) & 1, cs[i - 1], cs[i], ctrl, low)
        C.free([cs[i]])


def add_const(C, reg, value, ctrl=None, cout=None, low=None):
    """reg += ctrl*value mod 2^w, cout ^= ctrl*carry out. The carry chain reads
    the constant's bits, so 0 bits below the first 1 cost nothing. low(t), if
    given, XORs ctrl*reg[j] into t for the lowest set bit j of value."""
    w = len(reg)
    value %= 1 << w
    assert ctrl is None or ctrl not in (*reg, cout)
    cs = _carries(C, reg, value, ctrl, low)
    if cout is not None:
        _carry_into(C, reg[-1], value >> (w - 1) & 1, cs[-1], cout, ctrl, low)
    for i in reversed(range(w)):
        if value >> i & 1:
            if ctrl is None:
                C.x(reg[i])
            else:
                C.cx(ctrl, reg[i])
        if cs[i] is not None:
            C.cx(cs[i], reg[i])
        _uncarry(C, reg, value, cs, i, ctrl, low)


def gt_const_into(C, value, b, t, ctrl=None):
    """t ^= ctrl*[value > b], b restored. Carry out of (~b + value) is [value > b]."""
    assert ctrl is None or ctrl not in (*b, t)
    assert t not in b
    assert value < 1 << len(b)
    for x in b:
        C.x(x)
    cs = _carries(C, b, value, ctrl)
    _carry_into(C, b[-1], value >> (len(b) - 1) & 1, cs[-1], t, ctrl)
    for i in reversed(range(len(b))):
        _uncarry(C, b, value, cs, i, ctrl)
    for x in b:
        C.x(x)


def _reduce(C, r, value):
    """r holds s on m wires, value has bit m - 1 set, and s's top bit and the
    carry into it are never both 1. Leaves (s + flag*value) mod 2^(m-1) in
    r[:-1] and flag = [s + value >= 2^m] in r[-1]. One carry chain serves both
    the comparison and the conditional add."""
    assert value >> (len(r) - 1) & 1
    cs = _carries(C, r, value)
    flag = r[-1]
    if cs[-1] is not None:
        C.cx(cs[-1], flag)
    for i in reversed(range(len(r) - 1)):
        _uncarry(C, r, value, cs, i + 1)
        if cs[i] is not None:
            C.ccx(flag, cs[i], r[i])
        if value >> i & 1:
            C.cx(flag, r[i])


def cswap(C, ctrl, a, b):
    for ai, bi in zip(a, b):
        C.cx(bi, ai)
        C.ccx(ctrl, ai, bi)
        C.cx(bi, ai)


def _selected_carry(C, bit, selectors, masks, i, incoming, target):
    """XOR a majority carry into target, restoring bit and the selectors.

    The selected addend bit is an XOR of constant masks. Callers use exclusive
    selectors, so that XOR is also the intended selected integer's bit.
    """
    active = [s for s, mask in zip(selectors, masks) if mask >> i & 1]
    if not active:
        if incoming is not None:
            C.ccx(bit, incoming, target)
        return
    b = active[0]
    for other in active[1:]:
        C.cx(other, b)
    if incoming is None:
        C.ccx(bit, b, target)
    else:
        # MAJ(x,b,c) = c XOR ((x XOR c) AND (b XOR c)).
        C.cx(incoming, bit)
        C.cx(incoming, b)
        C.ccx(bit, b, target)
        C.cx(incoming, target)
        C.cx(incoming, b)
        C.cx(incoming, bit)
    for other in reversed(active[1:]):
        C.cx(other, b)


def _selected_chain(C, low, selectors, masks):
    carries = [None]
    for i, bit in enumerate(low):
        (carry,) = C.alloc(1)
        _selected_carry(C, bit, selectors, masks, i, carries[-1], carry)
        carries.append(carry)
    return carries


def _add_selected_const(C, b, selectors, masks):
    """Add one of the constant masks under exclusive selectors, inline."""
    n = len(b)
    masks = [mask % (1 << n) for mask in masks]
    start = min(((mask & -mask).bit_length() - 1 for mask in masks if mask), default=n)
    low = b[start:]
    shifted = [mask >> start for mask in masks]
    chain = _selected_chain(C, low[:-1], selectors, shifted)
    for i in reversed(range(len(low))):
        if i + 1 < len(low):
            # Clear the carry against the old bit, before writing the sum.
            _selected_carry(C, low[i], selectors, shifted, i, chain[i], chain[i + 1])
            C.free([chain[i + 1]])
        for selector, mask in zip(selectors, shifted):
            if mask >> i & 1:
                C.cx(selector, low[i])
        if chain[i] is not None:
            C.cx(chain[i], low[i])


class Mod:
    """Arithmetic modulo an odd q < 2^n on canonical values in [0, q)."""

    def __init__(self, q):
        self.q = q
        self.n = q.bit_length()
        self.low_bits = (2 * ((1 << self.n) - q)).bit_length()

    def _wrap(self):
        """2^(n+1) - q, so s + _wrap() carries out of n + 1 bits exactly when s >= q."""
        return (1 << (self.n + 1)) - self.q

    def modadd(self, C, a, b, ctrl=None):
        """b = (b + ctrl*a) mod q, for b < q, and a < q when ctrl is 1."""
        (top,) = C.alloc(1)
        add(C, a, b, top, ctrl)
        # s = b + ctrl*a < 2q, so s's top bit and the carry into it are never both 1.
        _reduce(C, b + [top], self._wrap())
        # top = [q was subtracted] = ctrl*[a > b]; ctrl = 0 means s = b < q and top = 0.
        gt_into(C, a, b, top, ctrl)
        C.free([top])

    def moddbl(self, C, s):
        """s = 2s mod q. 2s is a relabel with a zero low bit, so the reduction
        runs on s against _wrap() >> 1, and the low bit of the result is the flag."""
        _reduce(C, s, self._wrap() >> 1)
        s[:] = s[-1:] + s[:-1]

    def dbl_add(self, C, ctrl, a, s, m=None):
        """s = (2s + ctrl*a) mod q, canonical inputs, with clean scratch.

        Like moddbl, rotate the output wire list right by one. The two quotient
        bits are cleared by parity and a controlled comparison, respectively.
        See DESIGN.md for the all-input reduction and cleanup argument.
        """
        n, q = self.n, self.q
        m = self.low_bits if m is None else m
        assert q >= 3 and q & 1 and self.low_bits <= m <= n
        assert len(a) == len(s) == n and len(set([ctrl] + a + s)) == 2 * n + 1
        f = (1 << n) - q
        l0, top = C.alloc(2)
        add(C, a[1:] + [l0], s, top, ctrl=ctrl)
        C.ccx(ctrl, a[0], l0)
        low = [l0] + s[:-1]
        L, H = low[:m], low[m:]
        # A[1] aliases H[0]; only A[2:] are owned scratch.
        prefix = [None] + (H[:1] if H else [])
        for bit in H[1:]:
            (p,) = C.alloc(1)
            C.ccx(prefix[-1], bit, p)
            prefix.append(p)
        k0, k1 = s[-1], top
        C.x(k1)
        C.cx(k0, k1)
        selectors, masks = [k1, k0], [f, 2 * f]
        chain = _selected_chain(C, L, selectors, masks)
        (e,) = C.alloc(1)
        if H:
            C.ccx(prefix[-1], chain[-1], e)
        else:
            C.cx(chain[-1], e)
        for i in reversed(range(m)):
            _selected_carry(C, L[i], selectors, masks, i, chain[i], chain[i + 1])
            C.free([chain[i + 1]])
        C.cx(k0, k1)
        C.x(k1)
        C.ccx(k0, e, k1)
        C.cx(e, k0)
        selectors = [k0, k1]
        chain = _selected_chain(C, L, selectors, masks)
        if H:
            C.ccx(prefix[-1], chain[-1], e)
        else:
            C.cx(chain[-1], e)
        C.free([e])
        for i in reversed(range(1, len(H))):
            # Clear the prefix before overwriting the bit that computed it.
            C.ccx(prefix[i], H[i], prefix[i + 1])
            C.ccx(chain[-1], prefix[i], H[i])
            C.free([prefix[i + 1]])
        if H:
            C.cx(chain[-1], H[0])
        for i in reversed(range(m)):
            _selected_carry(C, L[i], selectors, masks, i, chain[i], chain[i + 1])
            C.free([chain[i + 1]])
            if i:
                for selector, mask in zip(selectors, masks):
                    if mask >> i & 1:
                        C.cx(selector, L[i])
                C.cx(chain[i], L[i])
        (z,) = C.alloc(1)
        C.x(k0)
        C.ccx(ctrl, k0, z)
        C.x(k0)
        gt_into(C, a[1:], low[1:], k1, ctrl=z)
        C.x(k0)
        C.ccx(ctrl, k0, z)
        C.x(k0)
        C.cx(l0, k0)
        C.ccx(ctrl, a[0], l0)
        C.free([l0, top, z])
        s[:] = s[-1:] + s[:-1]

    def cmodadd_const(self, C, ctrl, value, b):
        """b = (b + ctrl*value) mod q on canonical b, with ctrl restored."""
        value %= self.q
        if value == 0:
            return
        (k,) = C.alloc(1)
        gt_const_into(C, self.q - value, b, k, ctrl)
        C.cx(ctrl, k)
        C.cx(k, ctrl)
        _add_selected_const(C, b, [ctrl, k], [value, value + (1 << self.n) - self.q])
        C.cx(k, ctrl)
        gt_const_into(C, value, b, k, ctrl)
        C.free([k])

    def creflect(self, C, ctrl, a, b):
        """b = (a - b) mod q when ctrl, valid on all canonical b."""
        a %= self.q
        (k,) = C.alloc(1)
        gt_const_into(C, a + 1, b, k, ctrl)
        C.cx(ctrl, k)
        for wire in b:
            C.cx(ctrl, wire)
        C.cx(k, ctrl)
        _add_selected_const(C, b, [ctrl, k], [a + 1, (a + 1 + self.q) % (1 << self.n)])
        C.cx(k, ctrl)
        C.cx(ctrl, k)
        gt_const_into(C, a + 1, b, k, ctrl)
        C.free([k])

    def cneg(self, C, ctrl, b):
        """b = q - b when ctrl, valid for b in [1, q). A bijection on all of
        {0,1}^n either way, since it is ~(b + 2^n - q - 1) mod 2^n."""
        add_const(C, b, (1 << self.n) - self.q - 1, ctrl=ctrl)
        for w in b:
            C.cx(ctrl, w)

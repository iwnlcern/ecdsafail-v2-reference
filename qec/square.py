"""Y^2 mod q on a fold of 2^n = f, for q = 2^n - f with f*f <= q.

Write ~Y = 2^n - 1 - Y. Each pair i < j gives y_i*~y_j + y_j*~y_i = y_i ^ y_j, and the
diagonal gives 0, so Y*~Y = sum over i < j of (y_i ^ y_j)*2^(i + j), a sum of rows with no
control on any of them. Q = Y + Y*~Y = Y*(2^n - Y), so Y^2 = 2^n*(Y - hi) - lo for
Q = lo + 2^n*hi, and Y - hi = D is at least 0. The rest folds 2^n to f:

    Z = (2^n - lo) + f*D = lo1 + 2^n*E, E <= f
    V = lo1 + f*E, V = Y^2 + f mod q, 0 <= V - f, V < 2^n + f*f

If V >= 2^n, dropping that bit subtracts q and f together and leaves V - 2^n <= f*f - 1 < q.
Otherwise V - f < q, so subtracting f is enough. So no comparison with q is needed.
"""
from .arith import (_carries, _carry_into, _uncarry, add, add_const, add_lookahead, cadd_lookahead, fan_into,
                    gt_into, gt_lookahead, prefix_xor, sub)
from .builder import Block, Circuit
from .two_bit import Prefix

# Add the square's rows as lookahead trees, on ROW_TEMPS clean wires beside the fold's own.
ROW_TREE = True
ROW_TEMPS = 120
# Fan each row's Gray-code step out through at most this many copies on clean temps, and undo the walk
# by a prefix network; 0 runs both as CX chains.
ROW_FAN_COPIES = 63
# Run modadd's controlled add and comparison as lookahead trees, on MODADD_TEMPS clean wires and
# MODADD_COPIES copies of the control.
MODADD_TREE = True
MODADD_TEMPS = 95
MODADD_COPIES = 15
# Run the fold's adds as add_lookahead trees, on FOLD_TEMPS clean wires and the fold's clean junk.
FOLD_TREE = True
FOLD_TEMPS = 86
# The Karatsuba half() rows without a carry in: PR #12's tree _rows when True, else lean's ripple rows.
KARATSUBA_ROW_TREE = True
# The Karatsuba modadd per group: None is PR #12's modadd as it is, 0 its ripple form, (temps, copies) its trees.
KARATSUBA_AC_MODADD = (44,7)
KARATSUBA_B_MODADD = None
# Clean temps for each of the AC group's halves: 0 keeps the class's ripple rows, t > 0 runs PR #12's tree rows
# on them.
KARATSUBA_HALF_TEMPS = 110
# Clean temps for the folds' _multiple adds (AC group, B group): 0 keeps lean's add, t > 0 runs trees on them.
KARATSUBA_FOLD_TEMPS = (59,123)


def _add(C, a, b, cin=None, temps=()):
    """b += a mod 2^len(b), plus 1 if cin holds 1: add_lookahead's tree on temps when FOLD_TREE."""
    if FOLD_TREE:
        add_lookahead(C, a, b, None, list(temps), tree=True, cin=cin)
    else:
        add(C, a, b, cin=cin)


def _fan_copies(targets, cap):
    """The number of copies, 2^j - 1 <= cap, that fans src into targets in the fewest CX layers:
    j to make them and ceil(targets / 2^j) to fan out (unmaking them overlaps the add)."""
    best, j = (targets, 0), 1
    while (1 << j) - 1 <= cap:
        best = min(best, (j + -(-targets // (1 << j)), (1 << j) - 1))
        j += 1
    return best[1]


def signed_digits(v):
    """Non-adjacent form of v, as (position, sign)."""
    out, s = [], 0
    while v:
        if v & 1:
            d = 2 - (v & 3)
            out.append((s, d))
            v -= d
        v >>= 1
        s += 1
    return out


class SquareFold:
    def __init__(self, mod):
        self.mod, self.n = mod, mod.n
        self.f = f = (1 << mod.n) - mod.q
        self.b, self.m = f.bit_length(), (f * f).bit_length()
        # The low digit is +1 whatever f is, so its carry in can hold Z's + 1.
        self.digits = [(0, 1)] + signed_digits(f - 1)
        self.junk = 2 * mod.n + self.b + self.m + 3

    @staticmethod
    def applies(q):
        f = (1 << q.bit_length()) - q
        return f * f <= q

    def modadd(self, C, a, b, ctrl, shape=None):
        """Mod.modadd with _reduce's carries above f's window held by a Prefix, so it runs
        beside the wires compute leaves live. The same flag and the same write; with MODADD_TREE the
        add and the comparison run as lookahead trees."""
        f, w = self.f, self.b
        # shape: None for the module's MODADD_* constants, 0 for the ripple form, or (temps, copies).
        tree, ntemps, ncopies = ((MODADD_TREE, MODADD_TEMPS, MODADD_COPIES) if shape is None
                                 else (bool(shape), *(shape or (0, 0))))
        top, t = C.alloc(2)
        if tree:
            temps, copies = C.alloc(ntemps), C.alloc(ncopies)
            cadd_lookahead(C, a, b, top, ctrl, temps, copies)
            C.free(copies)
            C.free(temps)
        else:
            add(C, a, b, top, ctrl)
        cs = _carries(C, b[:w + 1], f)
        carry = Prefix(C, b[w:])
        # s's top bit and the carry into it are never both 1, so their XOR is the flag.
        C.ccx(cs[w], carry.top, top)
        C.ccx(top, cs[w], t)
        carry.write(t)
        C.ccx(top, cs[w], t)
        for i in reversed(range(w)):
            _uncarry(C, b, f, cs, i + 1)
            if cs[i] is not None:
                C.ccx(top, cs[i], b[i])
            if f >> i & 1:
                C.cx(top, b[i])
        if tree:
            temps = C.alloc(ntemps)
            gt_lookahead(C, a, b, top, ctrl, temps)
            C.free(temps)
        else:
            gt_into(C, a, b, top, ctrl)
        C.free([top, t])

    def _rows(self, C, Y, Q, temps):
        """Q = Y*(2^n - Y) on clean wires. After row i, Q < 2^(n + i + 1), so each carry
        lands on a wire no earlier row reached. Y's wires walk through their Gray code.
        The rows add in series, so each adds with add_lookahead, on clean temps, as a tree
        with ROW_TEMPS more clean wires when ROW_TREE."""
        n = self.n
        extra = C.alloc(ROW_TEMPS) if ROW_TREE else []
        temps = list(temps) + extra
        for y, q in zip(Y, Q):
            C.cx(y, q)
        for i in range(n - 1):
            if ROW_FAN_COPIES:
                # The tree takes temps from the front, so the copies sit at the back.
                k = _fan_copies(n - 1 - i, min(ROW_FAN_COPIES, len(temps)))
                fan_into(C, Y[i], Y[i + 1:], temps[len(temps) - k:])
            else:
                for w in Y[i + 1:]:
                    C.cx(Y[i], w)
            add_lookahead(C, Y[i + 1:], Q[2 * i + 1:n + i], Q[n + i], temps, tree=ROW_TREE)
        if ROW_FAN_COPIES:
            prefix_xor(C, Y)
        else:
            for i in range(1, n):
                C.cx(Y[i - 1], Y[i])
        C.free(extra)

    def _multiple(self, C, a, T, cin=None, temps=()):
        """T += f*a mod 2^len(T), plus 1 if cin holds 1. temps: clean wires for the adds' trees."""
        for s, sign in self.digits:
            target = T[s:]
            if not target:
                continue
            pad = C.alloc(max(0, len(target) - len(a)))
            b = a[:len(target)] + pad
            if sign > 0:
                _add(C, b, target, cin, temps)
            else:
                for w in target:
                    C.x(w)
                _add(C, b, target, None, temps)
                for w in target:
                    C.x(w)
            C.free(pad)
            cin = None

    def compute(self, C, Y, J):
        """J clean on entry. On exit J[:n] holds Y^2 mod q, canonical, for any n-bit Y.
        The other wires of J hold values fixed by Y, for the reverse of this block."""
        n, f, b, m = self.n, self.f, self.b, self.m
        lo, hi, E, T = J[:n], J[n:2 * n], J[2 * n:2 * n + b], J[2 * n + b:2 * n + b + m]
        t, c, k = J[2 * n + b + m:]
        # E, T, t, c and k are clean until the rows are done
        self._rows(C, Y, lo + hi, J[2 * n:])
        fold = C.alloc(FOLD_TEMPS) if FOLD_TREE else []
        # hi = Y - hi = ~(hi + ~Y), and lo = ~lo.
        for w in Y:
            C.x(w)
        _add(C, Y, hi, None, J[2 * n:] + fold)
        for w in Y:
            C.x(w)
        for w in hi + lo:
            C.x(w)
        (one,) = C.alloc(1)
        C.x(one)
        self._multiple(C, hi, lo + E, one, T + [t, c, k] + fold)
        C.x(one)
        C.free([one])
        self._multiple(C, E, T, None, [t, c, k] + fold)
        high = lo[m:]
        if FOLD_TREE:
            add_lookahead(C, T, lo[:m], t, [c, k] + fold, tree=True)
        else:
            add(C, T, lo[:m], cout=t)
        C.free(fold)
        if high:
            carry = Prefix(C, high)
            C.ccx(t, carry.top, c)
            carry.write(t)
        else:
            C.cx(t, c)
        # With c = 0, lo >= f. Take f off the low bits, and the borrow off the rest.
        C.x(c)
        add_const(C, lo[:b], (1 << b) - f, c, k)
        C.cx(c, k)
        high = lo[b:]
        for w in high:
            C.x(w)
        Prefix(C, high).write(k)
        for w in high:
            C.x(w)
        C.x(c)


class _Rows:
    def __init__(self, k):
        self.n = k


class KaratsubaMulsub:
    """One-level Karatsuba square-subtract, with canonical build/apply/unbuild groups."""

    @staticmethod
    def fits(q):
        """The construction's own bound: n even, and AC's c1 and c2 exclusive, which needs
        f*G1 well under 2^n, so h + bitlen(f) + 3 < n."""
        n = q.bit_length()
        f = (1 << n) - q
        return n % 2 == 0 and 0 < f and n // 2 + f.bit_length() + 3 < n

    @staticmethod
    def applies(q):
        """Where the generator takes it: at 128 bits and up, which keeps every exhaustive row and I2.mutation on
        SquareFold; scripts/karatsuba_check.py checks the class itself at forced small primes."""
        return q.bit_length() >= 128 and KaratsubaMulsub.fits(q)

    @staticmethod
    def _record_build(build, size):
        """Expose every retained wire as an explicit junk register of the recorded build.

        Freed temporaries are clean and remain the builder's local scratch. Retained wires
        may be dirty until the matching inverse, so apply must not recycle them in between.
        The local allocator is inspected only; record/apply own all state restoration.
        """
        local = Circuit()
        inputs = local.alloc(size)
        state = {}

        def capture(y):
            state['result'] = build(local, y)
            state['junk'] = sorted(set(range(size, local.num_wires)) - set(local.free_stack))

        block = local.record(capture, [inputs])
        junk = state['junk']
        layout = inputs + junk
        block = Block(block.ops, layout, layout, block.calls)
        result = [layout.index(w) for w in state['result']]
        return block, len(junk), result

    def __init__(self, mod):
        self.mod, self.q, self.n = mod, mod.q, mod.n
        n = self.n
        assert self.fits(self.q), "AC's c1 and c2 are exclusive only if f*G1 stays well under 2^n"
        self.h = h = n // 2
        self.f = f = (1 << n) - self.q
        self.fb = fb = f.bit_length()
        assert 3 * f < 1 << (fb + 2)
        self.sq = SquareFold(mod)
        self.digits = self.sq.digits
        beta = 1 << h
        K = beta * ((1 << (h + 1)) - 1) - (beta - 1) * ((1 << h) - 1)
        K -= (beta - f) * ((1 << h) - 1) + f * ((1 << n) - 1)
        self.K = K % self.q

    # lean's bounded SquareFold._multiple (c4380ee), with its adds as add_lookahead trees when temps are given:
    # PR #12's has no bounds
    def _multiple(self, C, a, T, a_max, t_max=0, cin=None, temps=()):
        """T += f*a mod 2^len(T), plus 1 if cin holds 1, for a <= a_max and T <= t_max on entry.

        Around each digit T holds t + d*a, plus 1 once cin is in, with d the digits so far.
        Those values at the corners of the box bound every input. A subtract adds onto ~T, so
        it bounds -v - 1 instead. Where every bound lies in [-2^i, 2^i) for a bit i at or above
        a's top, the digit's bits from i up all hold the sign. A sign that is always 0 never
        moves, so the digit adds below bit i only. Otherwise bit i takes the carry out and the
        bits above copy it. A digit onto a T that is 0, with no carry in, copies a."""
        inc, done, d = int(cin is not None), 0, 0
        for s, sign in self.digits:
            box = [(t, x) for t in (0, t_max) for x in (0, a_max)]
            before = [t + done + x * d for t, x in box]
            d += sign << s
            after = [t + inc + x * d for t, x in box]
            done = inc
            target = T[s:]
            if not target:
                continue
            if not any(before) and cin is None and sign > 0:
                for x, w in zip(a, target):
                    C.cx(x, w)
                continue
            ends = [v >> s for v in before + after]
            lo, hi = (min(ends), max(ends)) if sign > 0 else (-max(ends) - 1, -min(ends) - 1)
            top = next((i for i in range(len(a), len(target)) if -(1 << i) <= lo and hi < 1 << i),
                       len(target))
            carry = lo < 0 and top < len(target)
            above = target[top + 1:] if carry else []
            flips = [] if sign > 0 else target if carry else target[:top]
            for w in flips:
                C.x(w)
            for w in above:
                C.cx(target[top], w)
            pad = C.alloc(max(0, top - len(a)))
            if temps:
                add_lookahead(C, a[:top] + pad, target[:top], target[top] if carry else None, temps,
                              tree=True, cin=cin)
            else:
                add(C, a[:top] + pad, target[:top], cout=target[top] if carry else None, cin=cin)
            C.free(pad)
            for w in above:
                C.cx(target[top], w)
            for w in flips:
                C.x(w)
            cin = None

    # pieces; each leaves junk that the group's run back clears

    def half(self, C, Z, W, cin=None, temps=()):
        """W (2k clean) <- Z^2 + 2^k - 1: SquareFold's rows, then hi <- Z - hi and lo <- ~lo, as compute does.
        A clean cin wire, if given, is every add's carry in, so two halves on their own cins share no wire."""
        k = len(Z)
        if cin is None and KARATSUBA_ROW_TREE:
            SquareFold._rows(_Rows(k), C, Z, W, [])
        elif temps:
            for y, w in zip(Z, W):              # PR #12's SquareFold._rows on these temps, with the carry in
                C.cx(y, w)
            for i in range(k - 1):
                j = _fan_copies(k - 1 - i, min(ROW_FAN_COPIES, len(temps)))
                fan_into(C, Z[i], Z[i + 1:], temps[len(temps) - j:])
                add_lookahead(C, Z[i + 1:], W[2 * i + 1:k + i], W[k + i], temps, tree=True, cin=cin)
            prefix_xor(C, Z)
        else:
            for y, w in zip(Z, W):              # SquareFold._rows, with the carry in passed explicitly
                C.cx(y, w)
            for i in range(k - 1):
                for w in Z[i + 1:]:
                    C.cx(Z[i], w)
                add(C, Z[i + 1:], W[2 * i + 1:k + i], cout=W[k + i], cin=cin)
            for i in range(1, k):
                C.cx(Z[i - 1], Z[i])
        lo, hi = W[:k], W[k:]
        for w in Z:
            C.x(w)
        if temps:
            add_lookahead(C, Z, hi, None, temps, tree=True, cin=cin)
        else:
            add(C, Z, hi, cin=cin)
        for w in Z:
            C.x(w)
        for w in hi + lo:
            C.x(w)

    @staticmethod
    def add_low(C, a, b):
        """b += a mod 2^len(b) for len(a) < len(b); the carry above b's top bit goes to a fresh wire."""
        k = len(a)
        (t, c) = C.alloc(2)
        add(C, a, b[:k], cout=t)
        high = b[k:]
        if not high:
            C.cx(t, c)
            return c
        carry = Prefix(C, high)
        C.ccx(t, carry.top, c)
        carry.write(t)
        return c

    def canon(self, C, R):
        """R < 2^n becomes R mod q. k = [R >= q] = [R + f >= 2^n]."""
        f, fb = self.f, self.fb
        low, high = R[:fb], R[fb:]
        (t, k, t2) = C.alloc(3)
        cs = _carries(C, low, f)
        _carry_into(C, low[-1], f >> (fb - 1) & 1, cs[-1], t)
        for i in reversed(range(fb)):
            _uncarry(C, low, f, cs, i)
        block, count, result = self._record_build(lambda c, r: [Prefix(c, r).top], len(high))
        junk = C.alloc(count)
        C.apply(block, [high, junk])
        top = (high + junk)[result[0]]
        C.ccx(t, top, k)
        C.apply(block, [high, junk], inverse=True)
        C.free(junk)
        add_const(C, low, f, ctrl=k, cout=t2)
        for w in high:
            C.cx(k, w)
        C.cx(k, t2)
        C.free([t2])

    def finish(self, C, R, c1, c2=None):
        """R + 2^n*c1 + 2^(n+1)*c2, c1 and c2 never both 1, becomes its canonical residue on R."""
        f, fb = self.f, self.fb
        F = C.alloc(fb + 2)
        for i in range(fb):
            if f >> i & 1:
                C.cx(c1, F[i])
                if c2 is not None:
                    C.cx(c2, F[i + 1])
        c3 = self.add_low(C, F, R)
        for i in range(fb):                      # F back to 0: c1 and c2 still hold what built it
            if f >> i & 1:
                C.cx(c1, F[i])
                if c2 is not None:
                    C.cx(c2, F[i + 1])
        C.free(F)
        add_const(C, R[:fb + 2], f, ctrl=c3)
        self.canon(C, R)

    def fold_ac(self, C, RA, RC):
        """RA <- (beta*(C' - A') + A') mod q, canonical. RC <- C' - A'."""
        h, fb = self.h, self.fb
        pad = C.alloc(2)
        sub(C, RA + pad, RC)
        C.free(pad)
        G0, G1 = RC[:h], RC[h:]
        T = C.alloc(len(G1) + fb + 1)
        tm = C.alloc(KARATSUBA_FOLD_TEMPS[0])
        self._multiple(C, G1, T, (1 << len(G1)) - 1, temps=tm)
        C.free(tm)
        c1 = self.add_low(C, T, RA)
        (padg, c2) = C.alloc(2)
        add(C, G0 + [padg], RA[h:] + [c1], cout=c2)
        C.free([padg])
        self.finish(C, RA, c1, c2)
        return RA

    def fold_b(self, C, RB):
        """Leaves (2^h*B0 + f*~P) mod q, canonical, on fresh wires. RB <- ~P."""
        n, h, fb = self.n, self.h, self.fb
        T = C.alloc(n + fb + 1)
        for x, w in zip(RB[:h], T[h:]):
            C.cx(x, w)
        (b,) = C.alloc(1)
        sub(C, RB[h:], RB[:h], borrow=b)          # RB0 <- B0 - B1 mod 2^h, b = [B1 > B0]
        for w in RB[h:]:                         # RB1 <- RB1 - b
            C.x(w)
        Prefix(C, RB[h:]).write(b)
        for w in RB[h:]:
            C.x(w)
        for w in RB:                             # RB = ~P
            C.x(w)
        tm = C.alloc(KARATSUBA_FOLD_TEMPS[1])
        self._multiple(C, RB, T, (1 << n) - 1, (1 << n) - 1, temps=tm)
        lo, hi = T[:n], T[n:]
        E2 = C.alloc(len(hi) + fb + 1)
        self._multiple(C, hi, E2, (1 << len(hi)) - 1, temps=tm)
        C.free(tm)
        c1 = self.add_low(C, E2, lo)
        self.finish(C, lo, c1)
        return lo

    def build_ac(self, C, Y):
        n, h = self.n, self.h
        L, H = Y[:h], Y[h:]
        (s,) = C.alloc(1)
        add(C, L, H, cout=s)
        RA, RC = C.alloc(n), C.alloc(n + 2)
        cins = C.alloc(2)
        temps = [C.alloc(KARATSUBA_HALF_TEMPS) for _ in range(2)]
        self.half(C, L, RA, cins[0], temps[0])
        self.half(C, H + [s], RC, cins[1], temps[1])
        C.free(temps[1])
        C.free(temps[0])
        C.free(cins)
        return self.fold_ac(C, RA, RC)

    def build_b(self, C, Y):
        RB = C.alloc(self.n)
        self.half(C, Y[self.h:], RB)
        return self.fold_b(C, RB)

    def mulsub(self, C, gen, Y, X, add_k=True):
        """Subtract gen*Y^2; omit K only when the caller pays it in its preceding add."""
        n = self.n
        modadds = {}
        for build, inverse, shape in ((self.build_ac, True, KARATSUBA_AC_MODADD),
                                      (self.build_b, False, KARATSUBA_B_MODADD)):
            if shape not in modadds:
                modadds[shape] = C.record_shape(lambda k, a, b: self.sq.modadd(C, a, b, k[0], shape), [1, n, n])
            modadd = modadds[shape]
            block, count, result = self._record_build(build, n)
            junk = C.alloc(count)
            C.apply(block, [Y, junk])
            value = [(Y + junk)[i] for i in result]
            C.apply(modadd, [[gen], value, X], inverse=inverse)
            C.apply(block, [Y, junk], inverse=True)
            C.free(junk)
        if add_k:
            self.mod.cmodadd_const(C, gen, self.K, X)

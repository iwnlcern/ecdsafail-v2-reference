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
from .arith import _carries, _carry_into, _uncarry, add, add_const, gt_into, sub
from .builder import Block, Circuit
from .two_bit import Prefix


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

    def modadd(self, C, a, b, ctrl):
        """Mod.modadd with _reduce's carries above f's window held by a Prefix, so it runs
        beside the wires compute leaves live. The same flag, the same write, the same CCX."""
        f, w = self.f, self.b
        top, t = C.alloc(2)
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
        gt_into(C, a, b, top, ctrl)
        C.free([top, t])

    def _rows(self, C, Y, Q):
        """Q = Y*(2^n - Y) on clean wires. After row i, Q < 2^(n + i + 1), so each carry
        lands on a wire no earlier row reached. Y's wires walk through their Gray code."""
        n = self.n
        for y, q in zip(Y, Q):
            C.cx(y, q)
        for i in range(n - 1):
            for w in Y[i + 1:]:
                C.cx(Y[i], w)
            add(C, Y[i + 1:], Q[2 * i + 1:n + i], cout=Q[n + i])
        for i in range(1, n):
            C.cx(Y[i - 1], Y[i])

    def _multiple(self, C, a, T, a_max, t_max=0, cin=None):
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
            add(C, a[:top] + pad, target[:top], cout=target[top] if carry else None, cin=cin)
            C.free(pad)
            for w in above:
                C.cx(target[top], w)
            for w in flips:
                C.x(w)
            cin = None

    def compute(self, C, Y, J):
        """J clean on entry. On exit J[:n] holds Y^2 mod q, canonical, for any n-bit Y.
        The other wires of J hold values fixed by Y, for the reverse of this block."""
        n, f, b, m = self.n, self.f, self.b, self.m
        lo, hi, E, T = J[:n], J[n:2 * n], J[2 * n:2 * n + b], J[2 * n + b:2 * n + b + m]
        t, c, k = J[2 * n + b + m:]
        self._rows(C, Y, lo + hi)
        # hi = Y - hi = ~(hi + ~Y), and lo = ~lo.
        for w in Y:
            C.x(w)
        add(C, Y, hi)
        for w in Y:
            C.x(w)
        for w in hi + lo:
            C.x(w)
        (one,) = C.alloc(1)
        C.x(one)
        # hi is any n-bit D, and lo + E holds ~lo, below 2^n.
        self._multiple(C, hi, lo + E, (1 << n) - 1, (1 << n) - 1, one)
        C.x(one)
        C.free([one])
        self._multiple(C, E, T, f)
        high = lo[m:]
        add(C, T, lo[:m], cout=t)
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
        """Where the generator takes it. Standalone, with K left to the caller, it takes more CCX than SquareFold
        at 2^64 - 59 (11,802 against 10,635) and fewer at 2^128 - 159 (35,838 against 37,661), at more wires
        (613 against 554)."""
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
        beta = 1 << h
        K = beta * ((1 << (h + 1)) - 1) - (beta - 1) * ((1 << h) - 1)
        K -= (beta - f) * ((1 << h) - 1) + f * ((1 << n) - 1)
        self.K = K % self.q

    # pieces; each leaves junk that the group's run back clears

    def half(self, C, Z, W, cin=None):
        """W (2k clean) <- Z^2 + 2^k - 1: SquareFold's rows, then hi <- Z - hi and lo <- ~lo, as compute does.
        A clean cin wire, if given, is every add's carry in, so two halves on their own cins share no wire."""
        k = len(Z)
        if cin is None:
            SquareFold._rows(_Rows(k), C, Z, W)
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
        self.sq._multiple(C, G1, T, (1 << len(G1)) - 1)
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
        self.sq._multiple(C, RB, T, (1 << n) - 1, (1 << n) - 1)
        lo, hi = T[:n], T[n:]
        E2 = C.alloc(len(hi) + fb + 1)
        self.sq._multiple(C, hi, E2, (1 << len(hi)) - 1)
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
        self.half(C, L, RA, cins[0])
        self.half(C, H + [s], RC, cins[1])
        C.free(cins)
        return self.fold_ac(C, RA, RC)

    def build_b(self, C, Y):
        RB = C.alloc(self.n)
        self.half(C, Y[self.h:], RB)
        return self.fold_b(C, RB)

    def mulsub(self, C, gen, Y, X, add_k=True):
        """Subtract gen*Y^2; omit K only when the caller pays it in its preceding add."""
        n = self.n
        modadd = C.record_shape(lambda k, a, b: self.sq.modadd(C, a, b, k[0]), [1, n, n])
        for build, inverse in ((self.build_ac, True), (self.build_b, False)):
            block, count, result = self._record_build(build, n)
            junk = C.alloc(count)
            C.apply(block, [Y, junk])
            value = [(Y + junk)[i] for i in result]
            C.apply(modadd, [[gen], value, X], inverse=inverse)
            C.apply(block, [Y, junk], inverse=True)
            C.free(junk)
        if add_k:
            self.mod.cmodadd_const(C, gen, self.K, X)

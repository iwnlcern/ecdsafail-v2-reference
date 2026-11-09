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
from .arith import _carries, _uncarry, add, add_const, add_lookahead, gt_into, sub_lookahead
from .two_bit import Prefix


# Clean temps for the fold's lookahead adds, taken after the rows and freed clean before the Prefix steps.
FOLD_TEMPS = 72


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

    def _rows(self, C, Y, Q, temps):
        """Q = Y*(2^n - Y) on clean wires. After row i, Q < 2^(n + i + 1), so each carry
        lands on a wire no earlier row reached. Y's wires walk through their Gray code.
        The rows add in series, so each adds with add_lookahead, on clean temps."""
        n = self.n
        for y, q in zip(Y, Q):
            C.cx(y, q)
        for i in range(n - 1):
            for w in Y[i + 1:]:
                C.cx(Y[i], w)
            add_lookahead(C, Y[i + 1:], Q[2 * i + 1:n + i], Q[n + i], temps)
        for i in range(1, n):
            C.cx(Y[i - 1], Y[i])

    def _multiple(self, C, a, T, temps, cin=None):
        """T += f*a mod 2^len(T), plus 1 if cin holds 1. Each digit adds in lookahead form on temps."""
        for s, sign in self.digits:
            target = T[s:]
            if not target:
                continue
            pad = C.alloc(max(0, len(target) - len(a)))
            b = a[:len(target)] + pad
            if sign > 0:
                add_lookahead(C, b, target, None, temps, cin=cin)
            else:
                sub_lookahead(C, b, target, temps)
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
        # The fold's four long adds run in lookahead form on clean temps taken after the rows.
        temps = C.alloc(FOLD_TEMPS)
        # hi = Y - hi = ~(hi + ~Y), and lo = ~lo. No carry out: hi + ~Y = 2^n - 1 - (Y - hi) < 2^n.
        for w in Y:
            C.x(w)
        add_lookahead(C, Y, hi, None, temps)
        for w in Y:
            C.x(w)
        for w in hi + lo:
            C.x(w)
        (one,) = C.alloc(1)
        C.x(one)
        self._multiple(C, hi, lo + E, temps, one)
        C.x(one)
        C.free([one])
        self._multiple(C, E, T, temps)
        high = lo[m:]
        add_lookahead(C, T, lo[:m], t, temps)
        C.free(temps)
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

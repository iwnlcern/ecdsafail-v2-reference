"""Classical integer model of the exact dialog in-place multiplier.

The circuit M requires canonical 1 <= x < q.
The integer model retains its zero convention.
The reference Euclid dialog is the oracle for the iteration bound and Bezout identity.
The held-offset recurrence and sweep describe the alternative physical
representation used by the shared-chain circuit.
"""


# Explicit startup selection, inherited by each fresh child process.
import os
_jump4_mode = os.environ.get('QEC_JUMP4','0')
if _jump4_mode not in ('0','1'):
    raise ValueError('QEC_JUMP4 must be 0 or 1')
JUMP4 = _jump4_mode=='1'
SECP256K1_Q = 2**256 - 2**32 - 977


def jump4_encode_cell(g: int, b: int, d_next: int) -> int:
    """Archive swap for g1, incoming offset for g0, and NEXT strip D1..4."""
    if g not in (0, 1) or b not in (0, 1) or d_next not in (1, 2, 3, 4):
        raise ValueError("invalid Jump-4 cell")
    return d_next - 1 + 4 * g + 8 * b


def jump4_decode_cell(code: int) -> tuple[int, int, int]:
    """All sixteen symbols are legal; state reachability is a separate check."""
    if code not in range(16):
        raise ValueError("invalid Jump-4 code")
    return (code >> 2) & 1, code >> 3, (code & 3) + 1


def jump4_physical_tape(q: int, x: int, ticks: int) -> tuple[int, list[int]]:
    """True-value reference for the one-live-offset physical transcript.

    At boundaries stored V=true v-E; E1 means positive even true v.
    Initial stripping precedes all cells; each cell archives the NEXT strip.
    """
    if q <= 2 or q % 2 == 0 or not 0 < x < q or ticks < 0:
        raise ValueError("Jump-4 needs odd q>2, nonzero canonical x, ticks>=0")
    d0 = min(4, (x & -x).bit_length() - 1)
    u, v, e, codes = q, x >> d0, 0, []
    for _ in range(ticks):
        g = v & 1
        swap = int(bool(g) and u > v)
        b = swap if g else e
        if swap:
            u, v = v, u
        v -= g * u
        d = min(4, (v & -v).bit_length() - 1) if v else 4
        v >>= d
        e = int(bool((swap or e) and d == 4 and v and v & 1 == 0))
        codes.append(jump4_encode_cell(g, b, d))
    return d0, codes


def _jump4_step(u, v):
    """Strip, swap, subtract; zero records freeze terminal padding."""
    if v == 0:
        return u, v, (0, 0, 0)
    d = min(4, (v & -v).bit_length() - 1)
    v >>= d
    g = v & 1
    swap = int(bool(g) and u > v)
    if swap:
        u, v = v, u
    if g:
        v -= u
    return u, v, (d, g, swap)


def iteration_bound(q):
    """Proven cap on active Euclid iterations: u*v at least halves per step."""
    return (q * (q - 1)).bit_length()


def euclid(q, x, iters):
    u, v, dialog = q, x, []
    for _ in range(iters):
        b0 = v & 1
        sw = b0 & (u > v)
        dialog.append((b0, sw))
        if sw:
            u, v = v, u
        if b0:
            v -= u
        v >>= 1
    return u, v, dialog


def offset_pass(q: int, x: int, iters: int) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
    """States before each step and at the end, and the unswept tape cells."""
    U, V, e = q, x, 0
    states = [(U, V, e)]
    cells = []
    for _ in range(iters):
        V ^= e
        odd = V & 1
        if odd:
            if U > V:
                U, V, e = V, U - V - 2, 1
            else:
                V, e = V - U, 0
        V //= 2
        cells.append((odd, e))
        states.append((U, V, e))
    return states, cells


def sweep_tape(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the reference (parity, swap) tape without changing the input."""
    out = list(cells)
    for i in range(len(out) - 1, 0, -1):
        odd, held = out[i]
        out[i] = (odd, held ^ ((1 - odd) * out[i - 1][1]))
    return out


def bezout(q, y, dialog):
    r, s = y, 0
    for b0, sw in reversed(dialog):
        s = 2 * s % q
        if b0:
            s = (s + r) % q
        if sw:
            r, s = s, r
    return r, s


def inplace_mul(q, x, y, iters=None):
    iters = iters or iteration_bound(q)
    u, v, dialog = euclid(q, x or 1, iters)
    assert u == 1 and v == 0, "euclid did not reach (1, 0) within the bound"
    r, s = bezout(q, y, dialog)
    assert r == 0
    return s


def width_schedule(q, iters):
    """Bits needed before iteration i while v != 0.

    For nonzero canonical x the terminal state (1, 0) also fits every
    emitted step. The multiplier remaps x = 0 to this nonzero domain.
    """
    n = q.bit_length()
    return [min(n, (q * (q - 1) >> i).bit_length()) for i in range(iters)]


def _ticks_from(q, m):
    u, v, t = q, m, 0
    if JUMP4:
        while v:
            u, v, _ = _jump4_step(u, v)
            t += 1
        return t
    while v:
        if v & 1:
            u, v = min(u, v), (max(u, v) - min(u, v)) >> 1
        else:
            v >>= 2 if v & 3 == 0 else 1
        t += 1
    return t


def _no_slow_odd_state(q):
    """No odd m with 2m < q takes more than log4(q^2 m) + 1/2 ticks from (q, m).

    Any m that did would follow one of these run patterns, each giving
    q = P c + Q 2^L and m = M c + N 2^L for even L; see the two-bit tick proof.
    """
    candidates = {1}
    for P, Q, M, N in ((1, 1, 1, 0), (5, 4, 1, 0), (5, 1, 1, 1), (9, 5, 5, 1), (29, 9, 9, 5), (65, 29, 29, 9)):
        for L in range(2, q.bit_length() + 2, 2):
            c, r = divmod(q - (Q << L), P)
            if c > 0 and r == 0:
                candidates.add(M * c + (N << L))
    return not any(
        2 * m < q and 1 << (2 * _ticks_from(q, m) - 1) > q * q * m for m in candidates if m & 1
    )


def tick_bound(q):
    """Cap from grouped-run potential decay, allowing one initial short shift.

    At even bit length the short shift can only cost a tick from a slow odd state.
    """
    if JUMP4:
        return (5*q.bit_length()+3)//4
    n, bound = q.bit_length(), ((2 * q * q * (q - 1)).bit_length() + 1) // 2
    return bound - 1 if bound > (3 * n + 1) // 2 and _no_slow_odd_state(q) else bound


def tick_widths(q, ticks):
    """Bits needed before each tick, from max^2 <= 2 * q^2 * (q - 1) / 4^t."""
    if JUMP4:
        n=q.bit_length();cap=(5*n+3)//4
        return [min(n,max(1,cap-t)) for t in range(ticks)]
    n, w = q.bit_length(), 2 * q * q * (q - 1)
    return [min(n, max(1, ((w >> 2 * t).bit_length() + 1) // 2)) for t in range(ticks)]


def tick_joint(q, ticks):
    """Per tick, the least K with 8^K > 2 * q^2 * (q - 1) / 4^t. The potential u * v * max(u, v)
    is at most that bound before tick t, so u and v are never both 2^K or more."""
    if JUMP4:
        W=tick_widths(q,ticks);B=q**5-1
        return [min(w,max(1,(((B>>(4*t)).bit_length())+4)//5)) for t,w in enumerate(W)]
    w = 2 * q * q * (q - 1)
    return [((w >> 2 * t).bit_length() + 2) // 3 for t in range(ticks)]


def tick_tape(q, x, ticks):
    """Semantic (add, bit) tape: bit is the swap, or the second strip when add is 0."""
    u, v, tape = q, x, []
    if JUMP4:
        for _ in range(ticks):
            u, v, cell = _jump4_step(u, v)
            tape.append(cell)
        return u, v, tape
    for _ in range(ticks):
        if v & 1:
            sw = int(u > v)
            if sw:
                u, v = v, u
            v -= u
            tape.append((1, sw))
            v >>= 1
        else:
            two = int(v % 4 == 0)
            tape.append((0, two))
            v >>= 1 + two
    return u, v, tape


def tick_pass(q, x, ticks):
    """Physical states (U, V, E) and unswept cells (add, b) of the held-offset pass."""
    U, V, E = q, x, 0
    states, cells = [(U, V, E)], []
    for _ in range(ticks):
        V ^= E
        if V & 1:
            if U > V:
                U, V, swap = V, U - V - 2, 1
            else:
                V, swap = V - U, 0
            b, E = E, swap
            cell = (1, b)
            V //= 2
        else:
            V //= 2
            two = int((V ^ E) & 1 == 0)
            cell = (0, two)
            V >>= two
        cells.append(cell)
        states.append((U, V, E))
    return states, cells


def tick_sweep(cells, offset):
    """Give each subtract cell its own swap bit; returns (cells, final offset)."""
    out = list(cells)
    for i in range(len(out) - 1, -1, -1):
        if out[i][0]:
            offset, out[i] = out[i][1], (1, offset)
    return out, offset


def tick_bezout(q, y, tape):
    if JUMP4:
        r, s = y, 0
        for d, g, swap in reversed(tape):
            s = (s + g * r) % q
            if swap:
                r, s = s, r
            s = (s << d) % q
        return r, s
    r, s = y, 0
    for add, b in reversed(tape):
        if add:
            s = (2 * s + r) % q
            if b:
                r, s = s, r
        else:
            s = (s << 1 + b) % q
    return r, s


def tick_mul(q, x, y):
    ticks = tick_bound(q)
    u, v, tape = tick_tape(q, x or 1, ticks)
    assert u == 1 and v == 0, "pass did not reach (1, 0) within the bound"
    r, s = tick_bezout(q, y, tape)
    assert r == 0
    return s


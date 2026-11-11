"""Exact, complete, controlled fixed-addend point addition:
|c>|R>|0> -> |c>|R + c*A>|0> for every R in E(F_q) encoded as little-endian
affine (x, y) with infinity = (0, 0)."""
from .arith import Mod, _maj, add_const, cswap, load_const
from .builder import Circuit
from .curve import Curve
from .square import SquareFold
from .two_bit import TwoBitMod
from .model import tick_bound, tick_widths
from . import block0_code, tail_code, tape_block
from .tape_block import FREE, TICKS
from array import array
from .builder import CCX, CX
from .curve import SECP256K1, SECP256K1_G

# K1: in cmodadd_const(ctrl, 3a, b) at secp256k1's G, the CCX pair (b[254], ctrl) at ops 2929 and 5373 of the
# call, the selected add's carry clear and the second comparison's carry, equals CX from ctrl on every b < q, the
# call's contract, and either ctrl (scripts/k1_check.py, a BDD over (ctrl, b)). Keyed by (q, value mod q), so
# every other curve and addend emits as before.
_K1_CUTS = {(SECP256K1.q, 3 * SECP256K1_G[0] % SECP256K1.q): (2929, 5373)}


def _cut_k1(C, start, q, value, ctrl, b):
    """Turn the proven CCX pair of the cmodadd_const call that began at op `start` into CX from ctrl."""
    for pos in _K1_CUTS.get((q, value % q), ()):
        j = start + 4 * pos
        op, x, y, t = C.prims[j:j + 4]
        assert (op, x, y) == (CCX, b[254], ctrl), (pos, op, x, y)
        C.prims[j:j + 4] = array("i", (CX, ctrl, t, t))


def _fanout(C, src, copies):
    have, todo = [src], list(copies)
    while todo:
        nxt = []
        for h in list(have):
            if not todo:
                break
            t = todo.pop(0)
            C.cx(h, t)
            nxt.append(t)
        have += nxt

def _unfanout(C, src, copies):
    # The doubling tree's CX in reverse order.
    rec = []
    have, todo = [src], list(copies)
    while todo:
        nxt = []
        for h in list(have):
            if not todo:
                break
            t = todo.pop(0)
            rec.append((h, t))
            nxt.append(t)
        have += nxt
    for h, t in reversed(rec):
        C.cx(h, t)

def _par_rotate(C, two, V, cp):
    """Rotate V left by one under `two`: reverse V[0..L-1], then reverse V[0..L-2]."""
    L = len(V)
    # With fewer clean wires than L // 2 - 1, the copies are shared round-robin; the rotation is
    # the same permutation, only less parallel.
    ctl = [two] + cp
    _fanout(C, two, cp)
    for p in range(L // 2):
        cswap(C, ctl[p % len(ctl)], [V[p]], [V[L - 1 - p]])
    for p in range((L - 1) // 2):
        cswap(C, ctl[p % len(ctl)], [V[p]], [V[L - 2 - p]])
    _unfanout(C, two, cp)


class DialogMultiplier:
    """M(x, y) = (x, x*y mod q) for canonical 1 <= x < q; see model.py."""

    # The tape stays decoded: no block, block-zero or tail code, and so no pebbled ticks. Every
    # code call below is gated on blocks or tail, and each encode was paired with a decode that
    # nothing between reads past, so the passes are the same maps on the same states.
    RAW_TAPE = True

    def __init__(self, C, mod, junk=0):
        """junk is the number of wires the forward Euclid pass may leave set for its inverse to
        clear. Each one it uses saves a CCX in the pass and another in the inverse."""
        self.C, self.mod, self.junk = C, mod, junk
        q, n = mod.q, mod.n
        self.iters = tick_bound(q)
        self.widths = tick_widths(q, self.iters)
        self._euclid_steps = {}
        self._euclid_tick0_block = None
        self._encode_block = None
        self._encode_block0 = None
        self._bezout_step = None
        self._bezout_ticks = {}
        self.narrow = TwoBitMod(q)
        self.block = None

    @property
    def pass_tail_width(self):
        return self.widths[self.iters - 1]

    @property
    def tape_external(self):
        return 2 * self.iters - (self.mod.n - self.pass_tail_width) - self.u_lent

    @property
    def blocks(self):
        return 0 if self.RAW_TAPE else self.iters // TICKS

    @property
    def u_lent(self):
        """U's wires that late tape cells take once they are zero, one per zero wire the pass
        ends with: V's tail and each coded block's free wires, which U takes back."""
        coded_free = sum(
            len(self._block_free(j, self.swept)) for j in range(self.blocks))
        return min(self.pass_tail_width + coded_free, self.mod.n - self.pass_tail_width)

    @property
    def lent_after_pass(self):
        coded_free = sum(
            len(self._block_free(j, self.swept)) for j in range(self.blocks))
        return self.pass_tail_width + coded_free - self.u_lent

    @property
    def swept(self):
        """Whether the tape at rest is the B1-swept one, which the code reads last cell first."""
        return self.narrow.merged

    def with_free_slots(self, wires):
        """A coded tape from its wires, with None where each block's free wires were."""
        G = list(wires)
        slots = sorted(slot for j in range(self.blocks) for slot in self._free_slots(j, self.swept))
        for slot in slots:
            G.insert(slot, None)
        return G

    def coded_tape(self, pool):
        """The tape after a pass."""
        return self.with_free_slots(pool[self.lent_after_pass:])

    def _block_free(self, j, swept):
        return block0_code.FREE if swept and j == 0 else FREE

    def _free_slots(self, j, swept):
        return [2 * ((j + 1) * TICKS - 1 - f // 2 if swept else j * TICKS + f // 2)
                + f % 2 for f in self._block_free(j, swept)]

    def _block_wires(self, G, j, swept):
        cells = range(j * TICKS, (j + 1) * TICKS)
        return [G[2 * k + h] for k in (reversed(cells) if swept else cells) for h in (0, 1)]

    def _encoder(self, j, swept):
        if swept and j == 0:
            if self._encode_block0 is None:
                self._encode_block0 = self.C.record_shape(
                    lambda W: block0_code.encode(self.C, W), [2 * TICKS])
            return self._encode_block0
        if self._encode_block is None:
            self._encode_block = self.C.record_shape(lambda W: tape_block.encode(self.C, W), [2 * TICKS])
        return self._encode_block

    def _encode(self, G, j, swept):
        """Code block j. Its free wires leave G, which keeps None in those places."""
        W = self._block_wires(G, j, swept)
        self.C.apply(self._encoder(j, swept), [W])
        wires = [W[f] for f in self._block_free(j, swept)]
        for slot, wire in zip(self._free_slots(j, swept), wires):
            assert G[slot] == wire
            G[slot] = None
        return wires

    def _block0_places(self, U, pool):
        """Where a merged pass leaves block 0's free wires: the last of its zero words, which fill
        U's top u_lent wires and then the pool's first lent_after_pass."""
        n = self.mod.n
        places = [(U, i) for i in range(n - self.u_lent, n)] + [(pool, i) for i in range(self.lent_after_pass)]
        return places[len(places) - len(self._block_free(0, self.swept)):]

    def _code_block0(self, U, pool):
        """Code block 0 of a pass that left it raw, with its free wires at _block0_places."""
        G = self.coded_tape(pool)
        for slot, (reg, i) in zip(self._free_slots(0, self.swept), self._block0_places(U, pool)):
            G[slot] = reg[i]
        self._encode(G, 0, self.swept)

    def _decode(self, G, j, wires, swept):
        assert len(wires) == len(self._block_free(j, swept))
        for slot, wire in zip(self._free_slots(j, swept), wires):
            G[slot] = wire
        self.C.apply(self._encoder(j, swept),
                     [self._block_wires(G, j, swept)], inverse=True)


    def _shared_difference(self, U, V, g0, g1, top=None, junk=()):
        """Compare and form the even difference using one majority chain. With top set, bit w - 1
        is done by _top_bit on the domain that mode names.

        junk is a list of clean wires. The carries out of bits 1..len(junk) are built there and
        left there, so those bits keep a_i in place and need no restore: 3 CCX a bit instead
        of 4. The junk is a function of this step's input, so the step's inverse clears it."""
        C = self.C
        (cin,) = C.alloc(1)
        chain = len(U) if top is None else len(U) - 1
        kept = len(junk)
        assert kept <= max(0, chain - 1)
        # carries[i] is the wire that holds the carry into bit i while the chain is up.
        carries = [cin, U[0]] + list(junk) + U[kept + 1:]
        for wire in V:
            C.x(wire)
        for i in range(chain):
            if i == 0:
                C.cx(U[i], V[i])
                C.cx(U[i], carries[i])
                C.cx(V[i], U[i])
            elif i <= kept:
                # MAJ(a, b, c) = c ^ (a ^ c)(b ^ c) onto the junk wire, a and b restored.
                c, t = carries[i], carries[i + 1]
                C.cx(c, U[i])
                C.cx(c, V[i])
                C.ccx(U[i], V[i], t)
                C.cx(c, t)
                C.cx(c, V[i])
                C.cx(c, U[i])
            else:
                _maj(C, carries[i], V[i], U[i])
        if top is None:
            C.ccx(g0, carries[chain], g1)
        else:
            self._top_bit(C, top, g0, U[-1], V[-1], carries[chain], g1)
        # Each in-place bit's selects go in before the chain reaches it. With S = carries[i] and B = V[i]:
        # S ^= g1*B puts g1*p on the restore's control, B ^= g0*not(S) keeps the restore's
        # product at S*p ^ g1*p (S*not(S) = 0), and the chain's CX cancels g1*p from the carry.
        for i in reversed(range(kept + 1, chain)):
            C.ccx(g1, V[i], carries[i])
            C.x(carries[i])
            C.ccx(g0, carries[i], V[i])
            C.x(carries[i])
        for i in reversed(range(chain)):
            # Restore a_i (a_i ^ g1*p_i on an in-place bit) while unwinding the carry chain.
            if i == 0:
                C.cx(V[i], U[i])
                C.cx(g0, V[i])
                C.cx(carries[i], V[i])
                C.cx(U[i], carries[i])
                C.cx(carries[i], V[i])
            elif i <= kept:
                C.cx(carries[i], U[i])
                C.ccx(g0, U[i], V[i])
                C.cx(carries[i], U[i])
                C.cx(carries[i], V[i])
                C.ccx(g1, V[i], U[i])
                C.cx(g1, U[i])
                C.cx(carries[i], V[i])
                C.cx(g1, V[i])
            else:
                C.ccx(carries[i], V[i], U[i])
                C.cx(U[i], carries[i])
        for i in range(kept + 1, chain):
            C.cx(U[i], V[i])
            C.cx(g1, U[i])
            C.cx(g0, V[i])
            C.cx(g1, V[i])
        C.free([cin])
        for wire in V:
            C.x(wire)
        if chain < len(V):
            C.cx(g1, V[-1])

    @staticmethod
    def _top_bit(C, mode, g0, A, B, Cw, H):
        """Bit w - 1 of the shared difference in place of its MAJ, g1 and unwind (5 CCX). On entry
        A = U[w-1] holds a, B = V[w-1] holds b = NOT vhat_top, Cw = U[w-2] holds the carry c into
        bit w - 1, H = g1 is 0. On exit A = U'_top, B = b ^ g0*(a ^ c), Cw = c, H = g0*MAJ(a, b, c),
        as the five CCX leave them, on the domain the mode names:
        top_clear: b = 1 (V's top wire is 0 on entry), so g1 = g0*(a | c) and U'_top = a & ~g0.
        narrow: the step leaves U's top wire 0, so a implies g0 & b, g1 = a ^ g0*c*(a ^ b) and
            B = a ^ b ^ g0*c.
        both: both hold."""
        if mode == "top_clear":
            C.cx(Cw, A)
            C.ccx(g0, A, B)
            C.cx(Cw, A)
            C.ccx(g0, A, H)
            C.cx(H, A)
            C.x(B)
            C.ccx(Cw, B, H)
            C.x(B)
        elif mode == "narrow":
            C.ccx(g0, Cw, H)
            C.cx(A, B)
            C.ccx(H, B, A)
            C.cx(H, B)
            C.ccx(g0, Cw, H)
            C.cx(A, H)
            C.cx(H, A)
        elif mode == "both":
            # P1 = ~a & ~c on B, P2 = ~g0 & ~c on A; then g1 = g0 ^ P1 ^ P2, B = a ^ g0 ^ c ^ P2.
            C.x(B)
            C.x(A)
            C.x(Cw)
            C.ccx(A, Cw, B)
            C.x(A)
            C.x(Cw)
            C.cx(A, B)
            C.x(g0)
            C.x(Cw)
            C.ccx(g0, Cw, A)
            C.x(g0)
            C.x(Cw)
            C.cx(A, H)
            C.cx(B, H)
            C.cx(g0, H)
            C.cx(H, B)
            C.cx(Cw, B)
            C.cx(B, A)
            C.cx(g0, A)
            C.cx(Cw, A)
        else:
            raise ValueError(mode)

    def _euclid_step(self, w: int, top_clear: bool = False, narrow: bool = False, last: bool = False,
                     kept: int = 0, keep_two: bool = False, k: int = 0):
        """top_clear: the tick before ran at the same width, so V is below 2^(w - 1) on entry
        and the second strip's last swap, of two zero bits, never fires.
        narrow: the next width is w - 1, so by E7 this step leaves U below 2^(w - 1).
        last: the pass's last tick, whose input reaches (U, V, E) = (1, 0, 0) by E3."""
        if last:
            top_clear = narrow = False
            assert not kept and not keep_two
        k = 0 if last else k
        key = (w, top_clear, narrow, last, kept, keep_two, k)
        if key not in self._euclid_steps:
            C = self.C

            def step(U, V, E, cell, J, cp=()):
                g0, b = cell
                offset = E[0]
                if last:
                    # The input is (1, 0, 0), (1, 1, 0) or (1, 0, 1), and the cell becomes
                    # (V0 ^ E, NOT V0): only V[0], E and the cell change, all by CX.
                    C.cx(V[0], g0)
                    C.cx(offset, g0)
                    C.x(b)
                    C.cx(V[0], b)
                    C.cx(b, V[0])
                    C.x(V[0])
                    C.cx(g0, offset)
                    C.cx(b, offset)
                    C.x(offset)
                    V[:] = V[1:] + V[:1]
                    return
                top = None
                if w >= 2 and (top_clear or narrow):
                    top = "both" if top_clear and narrow else "top_clear" if top_clear else "narrow"
                C.cx(offset, V[0])
                C.cx(V[0], g0)
                self._shared_difference(U[:w], V[:w], g0, b, top, J[:kept])
                V[:] = V[1:] + V[:1]
                # A subtract keeps its swap on E and moves the old offset to the cell.
                cswap(C, g0, [offset], [b])
                # A no-subtract tick strips a second bit when V + E is even. With keep_two the
                # flag stays on a junk wire for the step's inverse to clear.
                (two,) = J[kept:] if keep_two else C.alloc(1)
                C.cx(offset, V[0])
                C.x(g0)
                C.x(V[0])
                C.ccx(g0, V[0], two)
                C.x(V[0])
                C.x(g0)
                C.cx(offset, V[0])
                C.cx(two, b)
                C.ccx(two, offset, V[0])
                if cp:
                    _par_rotate(C, two, V[:w - 2 if top_clear else w - 1], list(cp))
                else:
                    for i in range(w - 3 if top_clear else w - 2):
                        cswap(C, two, [V[i]], [V[i + 1]])
                if not keep_two:
                    C.x(g0)
                    C.ccx(g0, b, two)
                    C.x(g0)
                    C.free([two])

            self._euclid_steps[key] = C.record_shape(step, [w, w, 1, 2, kept + keep_two] + ([k] if k else []))
        return self._euclid_steps[key]

    @staticmethod
    def step_carries(w, top_clear, narrow):
        """Carries a step of width w can keep: those out of bits 1 to chain - 1."""
        chain = w - 1 if w >= 2 and (top_clear or narrow) else w
        return max(0, chain - 1)

    @property
    def junk_capacity(self):
        """Junk wires a pass can use: every step but the first and the last, its carries and its
        second-strip flag. The first step may be _euclid_tick0, which keeps none."""
        total = 0
        for i in range(1, self.iters - 1):
            w = self.widths[i]
            top_clear = w >= 3 and self.widths[i - 1] == w
            narrow = self.widths[i + 1] == w - 1
            total += self.step_carries(w, top_clear, narrow) + 1
        return total

    def _euclid_tick0(self):
        """First Euclid step on U=q, E=0, using unwritten tape for carries."""
        if self._euclid_tick0_block is None:
            C, q, n = self.C, self.mod.q, self.mod.n
            f1 = (1 << n) - q + 1
            k = n - 1 - ((f1 & -f1).bit_length() - 1)

            def step(U, V, E, cell, scr):
                C.free(scr)
                g0, b = cell
                C.cx(V[0], g0)
                for i in range(1, n):
                    if q >> i & 1:
                        C.x(V[i])
                    C.ccx(g0, V[i], U[i])
                    if q >> i & 1:
                        C.x(V[i])
                # The select left U[j] = q_j ^ g0*(v_j ^ q_j), so the first carry g0*v_j,
                # j the lowest set bit of f1, is U[j], or U[j] ^ g0 ^ 1 when q_j = 1.
                j = (f1 & -f1).bit_length() - 1

                def low(t):
                    C.cx(U[j], t)
                    if q >> j & 1:
                        C.cx(g0, t)
                        C.x(t)

                add_const(C, V, f1, ctrl=g0, low=low)
                for wire in V:
                    C.cx(g0, wire)
                C.cx(g0, E[0])
                V[:] = V[1:] + V[:1]
                C.x(g0)
                C.x(V[0])
                C.ccx(g0, V[0], b)
                C.x(V[0])
                C.x(g0)
                for i in range(n - 2):
                    cswap(C, b, [V[i]], [V[i + 1]])

            self._euclid_tick0_block = C.record_shape(step, [n, n, 1, 2, k])
        return self._euclid_tick0_block

    def _euclid_pass(self, U: list[int], pool: list[int], block0_raw: bool = False, junk=()) -> None:
        # The pool starts as V plus clean external tape. It ends as the zero wires U did not
        # take back, then tape; U ends as its low wires and those zero wires. The boundary is a
        # permutation of U and the pool together.
        n = self.mod.n
        V, available, G = pool[:n], pool[n:], []
        junk = list(junk)
        # Carries spread evenly: the least level L with sum(min(carries, L)) >= len(junk) caps each step's kept
        # carries; junk left once every carry is kept goes to the steps' flags, in step order.
        caps = [0] * self.iters
        for j in range(1, self.iters - 1):
            wj = self.widths[j]
            caps[j] = self.step_carries(wj, wj >= 3 and self.widths[j - 1] == wj, self.widths[j + 1] == wj - 1)
        level = 0
        while level < max(caps) and sum(min(c, level) for c in caps) < len(junk):
            level += 1
        flags = max(0, len(junk) - sum(min(c, level) for c in caps))
        lend = U[n - self.u_lent:]
        E = self.C.alloc(1)
        f1 = (1 << n) - self.mod.q + 1
        k = n - 1 - ((f1 & -f1).bit_length() - 1)
        for i in range(self.iters):
            w = self.widths[i]
            assert len(V) == w and len(available) >= 2
            cell, available = available[:2], available[2:]
            tick0 = self._euclid_tick0() if i == 0 and w == n and len(available) >= k else None
            if tick0 is not None and tick0.wires <= set(tick0.flat_in) and set(tick0.flat_out) == set(tick0.flat_in):
                self.C.apply(tick0, [U, V, E, cell, available[:k]])
            else:
                # A tick leaves V below 2^(w' - 1), w' its own width.
                top_clear = i >= 1 and w >= 3 and self.widths[i - 1] == w
                # E7: the next tick's U fits its width. E3: the last tick ends at (1, 0, 0).
                last = i == self.iters - 1
                narrow = not last and self.widths[i + 1] == w - 1
                # The steps take junk wires in order: a step's carries first, then its flag.
                plain = last or i == 0
                kept = 0 if plain else min(len(junk), caps[i], level)
                keep_two = not plain and flags > 0 and len(junk) > kept
                flags -= keep_two
                J, junk = junk[:kept + keep_two], junk[kept + keep_two:]
                # Alternate clean scratch so adjacent carry and strip sweeps can overlap.
                hold = self.C.alloc(1) if i % 2 else []
                # Clean tape wires carry copies of the strip's control (none on the last step).
                k = 0 if last else min(max(0, (w - 1) // 2 - 1), len(available))
                self.C.apply(self._euclid_step(w, top_clear, narrow, last, kept, keep_two, k),
                             [U[:w], V, E, cell, J] + ([available[:k]] if k else []))
                self.C.free(hold)
            G.extend(cell)
            next_width = self.widths[i + 1] if i + 1 < self.iters else self.pass_tail_width
            assert w - 1 <= next_width <= w
            available.extend(V[next_width:])
            V[:] = V[:next_width]
            # By L6, u < 2**next_width from here on, and later steps touch only U[:w] for
            # w <= next_width, so U's wires from next_width up are zero for the rest of the pass.
            available.extend(reversed([u for u in U[next_width:w] if u in lend]))
        assert len(V) == self.pass_tail_width and not available and not junk
        # Sweep down, coding each block as it finishes. The offset sweep leaves b on a
        # no-subtract tick alone. On the merged path the B1 sweep follows one cell behind it,
        # so a block is coded only after the B1 step that reads its bottom cell.
        T, swept, clean = self.iters, self.swept, []
        both = None

        def mark(k):
            self.C.x(G[2 * k])
            self.C.x(G[2 * k - 2])
            self.C.ccx(G[2 * k], G[2 * k - 2], both)
            self.C.x(G[2 * k - 2])
            self.C.x(G[2 * k])

        # The pass reaches (1, 0) within T ticks: its last subtract leaves E = 0, and each
        # no-subtract cell after it has b = 1. So at cell T - 1, g0 = 0 implies E ^ b = 1, and
        # when cells T - 1 and T - 2 are both no-subtract their b are equal: the B1 step between
        # them never swaps, and mark(T - 2) is made at once. Cell 0 is tick 0 on U = q > x, where
        # a subtract always swaps, so there g0 = 1 implies E ^ b = 1.
        late = swept and T >= 3
        if swept and T >= 2:
            (both,) = self.C.alloc(1)
            mark(T - 2 if late else T - 1)
        for i in range(T - 1, -1, -1):
            g0, b = G[2 * i], G[2 * i + 1]
            if i == 0 or i == T - 1:
                self.C.cx(b, E[0])
                self.C.cx(g0, b)
                if i:
                    self.C.cx(E[0], b)
                    self.C.x(b)
                self.C.cx(b, E[0])
            else:
                cswap(self.C, g0, E, [b])
            if swept and i + 1 < T and not (late and i == T - 2):
                k = i + 1
                self._b1_step(G, k, both)
                if k >= 2:
                    # Turn mark(k,k-1) into mark(k-1,k-2) before any encode.
                    self.C.cx(G[2 * k - 4], G[2 * k])
                    self.C.x(G[2 * k - 2])
                    self.C.ccx(G[2 * k - 2], G[2 * k], both)
                    self.C.x(G[2 * k - 2])
                    self.C.cx(G[2 * k - 4], G[2 * k])
                else:
                    mark(1)
            done = i + 1 if swept else i
            if done % TICKS == 0 and done // TICKS < self.blocks:
                clean.extend(self._encode(G, done // TICKS, swept))
        if both is not None:
            self.C.free([both])
        self.C.free(E)
        # Block 0's free wires take their places among the zero words raw, and block 0 is coded
        # last, unless block0_raw leaves that to the caller.
        raw = []
        if swept and self.blocks:
            raw = [G[slot] for slot in self._free_slots(0, swept)]
            for slot in self._free_slots(0, swept):
                G[slot] = None
        zeros = V + clean + raw
        U[:] = U[:n - self.u_lent] + zeros[:self.u_lent]
        pool[:] = zeros[self.u_lent:] + [w for w in G if w is not None]
        if raw and not block0_raw:
            self._code_block0(U, pool)

    def _bezout(self):
        if self._bezout_step is None:
            C, mod = self.C, self.mod

            def step(g, R, S):
                self.narrow.step(C, g, R, S)

            self._bezout_step = C.record_shape(step, [2, mod.n, mod.n])
        return self._bezout_step

    def _bezout_tick(self, first, last, pebble=False):
        if (first, last, pebble) not in self._bezout_ticks:
            C, n = self.C, self.mod.n

            def tick(prev, g, nxt, R, S, flag):
                # flag comes in holding [previous cell is B1], or 0 on a first tick, which has no
                # previous cell, and goes out holding [this cell is B1], or empty after the last.
                out = self.narrow.tick(C, prev or None, g, nxt[0] if nxt else None, R, S,
                                       pebble=pebble, held=None if first else flag[0],
                                       cell0=last)
                flag[:] = [] if out is None else [out]

            sizes = [0 if first else 2, 2, 0 if last else 1, n, n, 1]
            self._bezout_ticks[first, last, pebble] = C.record_shape(tick, sizes)
        return self._bezout_ticks[first, last, pebble]

    def _b1_step(self, G, i, both):
        """Use the pass-wide mark while both cells are still raw."""
        cswap(self.C, both, [G[2 * i + 1]], [G[2 * i - 1]])

    def _bezout_pass(self, G, Yr, S, block0_raw=False):
        """G is the coded tape. Each block is decoded on a clean wire before a tick reads it and
        coded again once no tick reads it, except block 0 on the merged path with block0_raw."""
        C, T, swept = self.C, self.iters, self.swept
        if not self.narrow.merged:
            bz = self._bezout()
            for i in reversed(range(T)):
                j = i // TICKS
                if j < self.blocks and i % TICKS == TICKS - 1:
                    self._decode(G, j, C.alloc(len(self._block_free(j, swept))), swept)
                C.apply(bz, [G[2 * i:2 * i + 2], Yr, S])
                if j < self.blocks and i % TICKS == 0:
                    C.free(self._encode(G, j, swept))
            return
        # A tick reads the previous cell and writes the next cell's g0, so a block is decoded
        # one tick before its first tick and coded one tick after its last. Across each edge
        # two blocks are decoded for two ticks.
        if self.blocks * TICKS == T:
            self._decode(G, self.blocks - 1,
                         C.alloc(len(self._block_free(self.blocks - 1, swept))), swept)
            if self.blocks == 1:
                block0_code.hold(C, self._block_wires(G, 0, swept))
                C.free([G[1]])
                G[1] = None
        # The final swept cell is (1, 0) or (0, 1): the terminal subtract
        # is followed only by no-subtract cells. Lend its b only when uncoded.
        lend_tail = T > 1 and self.blocks * TICKS < T
        tail = (not self.RAW_TAPE and lend_tail and T >= tail_code.CELLS + 1
                and T - self.blocks * TICKS >= tail_code.CELLS
                and tick_widths(self.mod.q, T)[T - tail_code.CELLS] <= 6)
        lo = 2 * (T - tail_code.CELLS)
        tail_encoded = False
        flag = []
        for i in reversed(range(T)):
            nxt_block = i // TICKS - 1
            if i % TICKS == 0 and 0 <= nxt_block < self.blocks:
                self._decode(G, nxt_block,
                             C.alloc(len(self._block_free(nxt_block, swept))), swept)
                if nxt_block == 0:
                    block0_code.hold(C, self._block_wires(G, 0, swept))
                    C.free([G[1]])
                    G[1] = None
            if i == 1 and self.blocks:
                G[1] = C.alloc(1)[0]
                block0_code.hold(C, self._block_wires(G, 0, swept), inverse=True)
            no_prev = lend_tail and i == T - 2
            if no_prev:
                # The terminal cell is never B1; release b before its last reader.
                C.cx(G[-2], G[-1])
                C.x(G[-1])
                C.free([G[-1]])
                G[-1] = None
            first, last = i == T - 1 or no_prev, i == 0
            edge = any(i in (j * TICKS, j * TICKS - 1) for j in range(1, self.blocks))
            pebble = edge or (tail and self.blocks > 0 and i == self.blocks * TICKS)
            pebble = pebble or (self.blocks > 0 and i in (0, 1))
            prev = [] if first else G[2 * i + 2:2 * i + 4]
            nxt = [] if last else G[2 * i - 2:2 * i - 1]
            if i == T - 1:
                # S starts at zero and the terminal cell is A0, A1 or J, never B1.
                for r, s in zip(Yr, S):
                    C.ccx(G[2 * i], r, s)
            else:
                if not flag:
                    # The pass's first tick call: no tick before it handed a flag over.
                    flag = C.alloc(1)
                    if not first:
                        self.narrow._mark_b1(C, prev, flag[0])
                C.apply(self._bezout_tick(first, last, pebble),
                        [prev, G[2 * i:2 * i + 2], nxt, Yr, S, flag])
            prev_block = (i + 1) // TICKS
            if (i + 1) % TICKS == 0 and i + 1 < T and prev_block < self.blocks:
                C.free(self._encode(G, prev_block, swept))
            if tail and i == T - tail_code.CELLS - 1:
                W = G[lo:lo + tail_code.LIVE]
                tail_code.apply(C, W)
                tail_encoded = True
                C.free([W[tail_code.FREE[0]]])
                G[lo + tail_code.FREE[0]] = None
        assert not flag
        if self.blocks and not block0_raw:
            C.free(self._encode(G, 0, swept))
        if tail:
            assert tail_encoded, "tail inverse requires its forward apply"
            G[lo + tail_code.FREE[0]] = C.alloc(1)[0]
            tail_code.apply(C, G[lo:lo + tail_code.LIVE], inverse=True)
        if lend_tail:
            G[-1] = C.alloc(1)[0]
            C.x(G[-1])
            C.cx(G[-2], G[-1])

    def _body(self, Xr, Yr):
        C, q, n, T = self.C, self.mod.q, self.mod.n, self.iters
        U = C.alloc(n)
        load_const(C, U, q)
        pool = list(Xr) + C.alloc(self.tape_external)

        # On the merged path block 0 stays decoded from Bezout's end into the inverse pass, so
        # the pass is recorded without its last encode, which runs here instead.
        raw0 = self.swept and self.blocks > 0
        # Junk wires the forward pass fills and the inverse pass clears. Nothing between reads them.
        junk = C.alloc(min(self.junk, self.junk_capacity))
        E = C.run(lambda U, pool, J: self._euclid_pass(U, pool, raw0, J), [U, pool, junk])
        if raw0:
            self._code_block0(U, pool)
        lent = self.lent_after_pass
        G = self.coded_tape(pool)
        # Terminal V and the wires the code freed are zero; lend them during Bezout.
        C.free(pool[:lent])
        # U is (1,0,...,0). Reuse its entire word for the reverse coefficient;
        # the surviving V wire stays zero for the inverse Euclid boundary.
        C.x(U[0])
        S = U
        self._bezout_pass(G, Yr, S, raw0)
        if raw0:
            raw = [G[slot] for slot in self._free_slots(0, True)]
            for slot in self._free_slots(0, True):
                G[slot] = None
        # Restore the tape boundary using the newly allocated tail wire.
        pool[lent:] = [w for w in G if w is not None]
        C.x(Yr[0])
        pool[:lent] = C.alloc(lent)
        if raw0:
            # Block 0's free wires take the places of the last zero words, which are released.
            places = self._block0_places(Yr, pool)
            C.free([reg[i] for reg, i in places])
            for (reg, i), w in zip(places, raw):
                reg[i] = w
        C.apply(E, [Yr, pool, junk], inverse=True)
        C.free(junk)
        load_const(C, Yr, q)
        Xr[:] = pool[:n]
        C.free(Yr + pool[n:])
        Yr[:] = S

    def apply(self, Xr, Yr, inverse=False):
        if self.block is None:
            self.block = self.C.record(self._body, [Xr, Yr])
        self.C.apply(self.block, [Xr, Yr], inverse)


def eq_const_into(C, ctrl, bits, value, target):
    """XOR equality into target, optionally controlled by ctrl."""
    flips = [w for i, w in enumerate(bits) if not value >> i & 1]
    for w in flips:
        C.x(w)
    chain = []
    acc = ctrl if ctrl is not None else bits[0]
    factors = bits[:-1] if ctrl is not None else bits[1:-1]
    for w in factors:
        (a,) = C.alloc(1)
        C.ccx(acc, w, a)
        chain.append((acc, w, a))
        acc = a
    C.ccx(acc, bits[-1], target)
    for acc, w, a in reversed(chain):
        C.ccx(acc, w, a)
    C.free([a for _, _, a in chain])
    for w in flips:
        C.x(w)


def encode(P, n):
    x, y = (0, 0) if P is None else P
    return x | y << n


class ControlledPointAdd:
    # The wire budget, 4,096 at n = 256. Wires under it that the plain circuit leaves unused
    # hold junk: Euclid carries across the Bezout pass, and square row carries across the modadd.
    WIRES_PER_BIT = 16

    def __init__(self, curve: Curve, A):
        assert A is not None and curve.on_curve(A)
        self.curve, self.A = curve, A
        self.mod = Mod(curve.q)
        self.n = self.mod.n
        self.junk, self.keep = self._junk_budgets()
        neg = curve.neg
        twoA = curve.add(A, A)
        # (exceptional input, its sum with A); the generic kernel divides by zero on these.
        self.exceptions = [(None, A), (A, twoA), (neg(A), None), (neg(twoA), neg(A))]
        enc = [encode(e, self.n) for e, _ in self.exceptions]
        assert len(set(enc)) == 4, "addend order must exceed 3"
        self._square_blk = None

    def _junk_budgets(self):
        """(Euclid junk wires per pass, square row-carry wires) that fit under the wire budget.

        Each stage is built alone with no junk to find its own peak. Around M the control and
        two flag wires are also live; around the square-subtract the control and two flags are,
        beside the gen wire the stage itself is given."""
        n, mod = self.n, self.mod
        cap, around = self.WIRES_PER_BIT * n, 3
        C = Circuit()
        M = DialogMultiplier(C, mod)
        M.apply(C.alloc(n), C.alloc(n))
        junk = max(0, min(cap - around - C.num_wires, M.junk_capacity))
        if not SquareFold.applies(mod.q):
            return junk, 0
        keep = 0
        for attempt in range(2):
            # First with no keep, for the peak; then with the room left, for the wires rows use.
            saved = self.__dict__.copy()
            self.C, self.keep, self._square_blk, self._modadd_blk = Circuit(), keep, None, None
            (gen,) = self.C.alloc(1)
            self._square_mulsub(gen, self.C.alloc(n), self.C.alloc(n))
            wires, used = self.C.num_wires, self._square_fold.kept
            self.__dict__.clear()
            self.__dict__.update(saved)
            if attempt == 0:
                keep = max(0, cap - around - wires)
            else:
                keep = used
        return junk, keep

    def _mulsub(self, gen, Yr, Xr):
        """X = X - gen * Y^2 mod q, Y untouched."""
        C, mod, n = self.C, self.mod, self.n
        if SquareFold.applies(mod.q):
            self._square_mulsub(gen, Yr, Xr)
            return
        T = C.alloc(n)
        for y, t in zip(Yr, T):
            C.cx(y, t)
        if self._modadd_blk is None:
            self._modadd_blk = C.record_shape(lambda k, a, b: mod.modadd(C, a, b, ctrl=k[0]), [1, n, n])
            self._dbl = C.record_shape(lambda s: mod.moddbl(C, s), [n])
        for i in range(n):
            (k,) = C.alloc(1)
            C.ccx(gen, Yr[i], k)
            C.apply(self._modadd_blk, [[k], T, Xr], inverse=True)
            C.ccx(gen, Yr[i], k)
            C.free([k])
            C.apply(self._dbl, [T])
        for _ in range(n):
            C.apply(self._dbl, [T], inverse=True)
        for y, t in zip(Yr, T):
            C.cx(y, t)
        C.free(T)

    def _square_mulsub(self, gen, Yr, Xr):
        C, mod, n = self.C, self.mod, self.n
        if self._square_blk is None:
            fold = SquareFold(mod, self.keep)
            self._square_blk = C.record_shape(lambda y, j: fold.compute(C, y, j), [n, fold.junk])
            self._modadd_blk = C.record_shape(lambda k, a, b: fold.modadd(C, a, b, k[0]), [1, n, n])
            self._square_junk = fold.junk
            self._square_fold = fold
        J = C.alloc(self._square_junk)
        C.apply(self._square_blk, [Yr, J])
        C.apply(self._modadd_blk, [[gen], J[:n], Xr], inverse=True)
        C.apply(self._square_blk, [Yr, J], inverse=True)
        C.free(J)

    def _kernel(self, c, gen, Xr, Yr, f0, f1):
        C, mod, q = self.C, self.mod, self.curve.q
        a, b = self.A

        def recode(inverse=False):
            if not inverse:
                C.cx(gen, c)
                C.cx(gen, f0)
            # gen ^= NOT c AND (f0 XOR f1): no c = 0 word has f0 = 0 and f1 = 1.
            C.cx(f1, f0)
            C.x(c)
            C.ccx(c, f0, gen)
            C.x(c)
            C.cx(f1, f0)
            if inverse:
                C.cx(gen, f0)
                C.cx(gen, c)

        mod.cmodadd_const(C, gen, q - a, Xr)
        mod.cmodadd_const(C, gen, q - b, Yr)
        C.ccx(f0, f1, Xr[0])
        recode()
        C.free([gen])
        self.M.apply(Xr, Yr, inverse=True)
        (gen,) = C.alloc(1)
        recode(inverse=True)
        start = len(C.prims) // 4
        mod.cmodadd_const(C, gen, 3 * a, Xr)
        _cut_k1(C, 4 * start, q, 3 * a, gen, Xr)
        self._mulsub(gen, Yr, Xr)
        recode()
        C.free([gen])
        self.M.apply(Xr, Yr)
        (gen,) = C.alloc(1)
        recode(inverse=True)
        C.ccx(f0, f1, Xr[0])
        mod.creflect(C, gen, a, Xr)
        mod.cmodadd_const(C, gen, q - b, Yr)
        return gen

    def _eq_flags(self, c, Xr, Yr, flags, points):
        C, n = self.C, self.n
        groups = {}
        for f, p in zip(flags, points):
            x, y = (0, 0) if p is None else p
            groups.setdefault(x, []).append((f, y))
        assert len(groups) == 3 and len(groups.get(0, [])) == 1, "0/a-pair/2A groups"
        pair_x = next(x for x, leaves in groups.items() if len(leaves) == 2)
        single_x = next(x for x in groups if x not in (0, pair_x))
        shared_zero = [i for i in range(n) if not ((pair_x | single_x) >> i & 1)]
        shared_one = [i for i in range(n) if (pair_x & single_x) >> i & 1]
        differing = [i for i in range(n) if (pair_x ^ single_x) >> i & 1]

        def chain(acc, wires):
            history = []
            for wire in wires:
                if acc is None:
                    acc = wire
                else:
                    (target,) = C.alloc(1)
                    C.ccx(acc, wire, target)
                    history.append((acc, wire, target))
                    acc = target
            return acc, history

        def clear(history):
            for left, right, target in reversed(history):
                C.ccx(left, right, target)
            C.free([target for _, _, target in history])

        # The zero leaf is uncontrolled; c joins only on the nonzero branches.
        for i in shared_zero:
            C.x(Xr[i])
        root, root_chain = chain(None, [Xr[i] for i in shared_zero])
        zero_bits = [Xr[i] for i in shared_one + differing]
        eq_const_into(C, root, zero_bits, 0, groups[0][0][0])
        common, common_chain = chain(root, [c] + [Xr[i] for i in shared_one])
        flips = [Xr[i] for i in differing if not (pair_x >> i & 1)]
        for wire in flips:
            C.x(wire)
        leaf, leaf_chain = chain(common, [Xr[i] for i in differing])
        for f, y in groups[pair_x]:
            eq_const_into(C, leaf, Yr[:1], y & 1, f)
        clear(leaf_chain)
        for wire in flips:
            C.x(wire)
        f, y = groups[single_x][0]
        bits = [Xr[i] for i in differing] + Yr[:1]
        value = sum(((single_x >> i) & 1) << j for j, i in enumerate(differing))
        value |= (y & 1) << len(differing)
        eq_const_into(C, common, bits, value, f)
        clear(common_chain)
        clear(root_chain)
        for i in shared_zero:
            C.x(Xr[i])

    def build(self, hierarchical=False):
        curve, A = self.curve, self.A
        twoA = curve.add(A, A)
        if A[0] == 0 or twoA is None or twoA[0] == 0:
            raise ValueError("kernel zero flag requires nonzero x(A) and x(2A)")
        if pow(curve.b, (curve.q - 1) // 2, curve.q) != curve.q - 1:
            raise ValueError("kernel zero flag refuses curves with a point at x = 0")
        self.C = C = Circuit(hierarchical)
        n = self.n
        (c,) = C.alloc(1)
        Xr, Yr = C.alloc(n), C.alloc(n)
        declared = Xr + Yr
        self.M = DialogMultiplier(C, self.mod, self.junk)
        self._modadd_blk = self._square_blk = None

        flags = C.alloc(4)
        self._eq_flags(c, Xr, Yr, flags, [e for e, _ in self.exceptions])
        f0, f1, f2, gen = flags
        for f, (e_in, e_out) in zip(flags[1:], self.exceptions[1:]):
            load_const(C, Xr + Yr, encode(e_in, n) ^ encode(e_out, n), f)
        # Move the uncontrolled x = 0 marker to f2 when c = 0.
        C.x(c)
        C.ccx(c, f0, f2)
        C.ccx(c, f2, f0)
        C.x(c)
        e_in, e_out = self.exceptions[0]
        load_const(C, Xr + Yr, encode(e_in, n) ^ encode(e_out, n), f0)
        # Only the c-controlled part of f2 contributes to the generic flag.
        for w in (c, f0, f1):
            C.cx(w, gen)
        C.ccx(c, f2, gen)
        # Two wires name the one set flag among three; the wire left over is zero.
        C.cx(f2, f0)
        C.cx(f2, f1)
        C.ccx(f0, f1, f2)
        C.free([f2])

        gen = self._kernel(c, gen, Xr, Yr, f0, f1)

        (f2,) = C.alloc(1)
        C.ccx(f0, f1, f2)
        C.cx(f2, f1)
        C.cx(f2, f0)
        for w in (c, f0, f1):
            C.cx(w, gen)
        C.ccx(c, f2, gen)
        flags = [f0, f1, f2, gen]
        self._eq_flags(c, Xr, Yr, flags, [e for _, e in self.exceptions])
        C.free(flags)

        _restore_layout(C, Xr + Yr, declared)
        self.layout = {"control": [c], "x": declared[:n], "y": declared[n:]}
        return C


def _restore_layout(C, cur, declared):
    """SWAP logical bit i from wire cur[i] onto declared[i]. A displaced wire
    holds either another live bit, which is tracked, or a zero ancilla."""
    cur = list(cur)
    where = {w: i for i, w in enumerate(cur)}
    for i, d in enumerate(declared):
        s = cur[i]
        if s == d:
            continue
        C.swap(s, d)
        j = where.pop(d, None)
        del where[s]
        if j is not None:
            cur[j] = s
            where[s] = j
        cur[i] = d
        where[d] = i

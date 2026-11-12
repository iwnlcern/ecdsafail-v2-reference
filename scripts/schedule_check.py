"""The high-word schedule against the joint bound, with no use of qec.model.

A flagged tick's kappa must be min(w, K_t), K_t the least K with 2^(3K + 2t) > 2 q^2 (q - 1), and a
high tick's must be K_t itself. On the semantic Euclid recurrence, with the held offset tracked
beside it, no state before a flagged tick has both registers at 2^kappa or more, and a subtract
there leaves U below 2^kappa.

The lazy ticks must be the high ticks with more than WINDOW bits above kappa, in one run that starts
at WINDOW + 1 bits after a drop at the same width and ends at WINDOW + 1 bits before a narrower tick
at the same threshold; raw must say that the tick before kept its threshold; the counter must be
wide enough. On the same recurrence, with the pending borrow p followed through the lazy rules, the
window is all ones under p once the tick's crossing bits have met it, p is never set beside U's
word, physical V fits w + 1 bits, and the last tick's top bit and extra wire are what its exit
assumes.

The lag blocks must be runs of c lazy ticks, none the lazy run's first or last, each tick marked
(F, c, e, size) with one front F and size the block's first width plus one. F must keep the holes clear of
the cells a lazy tick reads, F - 2 e >= cut_e + 2 (cut_e + 1 at the block's last tick, whose holes are
closed before the next tick reads), be at most the width after the block and at least 2 c. A block whose
realign leaves X's top cell out must have its lazy word below 2^w by the potential bound. A block whose flag-off
bits are counted as a thermometer must keep V's word under the front less that count wherever they are not 0s
then 1s. On the recurrence, with lag the count so far of ticks that left the flag off and of strips counted
(two AND NOT flag): F - lag >= cut + 2 at every tick but the last, cut + 1 there; under the flag U is below
2^(F - lag) when lag > 0 and V is below 2^kappa; and on the lazy word's physical V, a word X0 that the far
cells hold when the block starts is, after every tick, either U's bits from F up, or V's physical bits
from F - lag up with U below 2^F; with no such word V stays below 2^(F - lag) and U below 2^F.

    python3 -I -B scripts/schedule_check.py     # toy primes, every x; secp256k1, sampled x
"""
import json
import random
import sys
from math import isqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qec.arith import Mod
from qec.builder import Circuit
from qec.point_add import DialogMultiplier

SECP = 2**256 - 2**32 - 977
TOY = [3, 5, 7, 11, 13, 31, 43, 61, 127, 151, 251, 509, 1021, 4093, 8117, 65521]
WINDOW = 2


from qec import model as _model
if _model.JUMP4:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import jump_gate_reference as _reference


def joint_threshold(q, t):
    if _model.JUMP4:
        k = 1
        while 1 << (5*k + 4*t) < q**5:
            k += 1
        return k
    k = 0
    while 1 << (3 * k + 2 * t) <= 2 * q * q * (q - 1):
        k += 1
    return k


def plan_of(q, lag=None):
    """The plan, or with lag = c the plan that has a lag block of c ticks wherever one fits."""
    if _model.JUMP4:
        if lag is not None:
            raise ValueError("explicit lag is unsupported by the eager schedule")
        return DialogMultiplier(Circuit(), Mod(q))._jump4_eager_plan()
    saved = DialogMultiplier.LAG_BLOCK, DialogMultiplier.LAG_MIN_SAVING
    if lag:
        DialogMultiplier.LAG_BLOCK, DialogMultiplier.LAG_MIN_SAVING = (lag,), None
    try:
        return DialogMultiplier(Circuit(), Mod(q))._tick_plan()
    finally:
        DialogMultiplier.LAG_BLOCK, DialogMultiplier.LAG_MIN_SAVING = saved


def thresholds_ok(q, plan):
    if _model.JUMP4:
        return plan == _reference.schedule_reference(q, (5*q.bit_length()+3)//4)
    for t, key in enumerate(plan):
        if key is None or key[5] is None:
            if key is not None and key[6]:
                return False
            continue
        w, kappa, high = key[0], key[5], key[6]
        K = joint_threshold(q, t)
        if kappa != min(w, K) or (high and not 1 <= kappa == K < w):
            return False
    return True


def states_ok(q, plan, xs):
    if _model.JUMP4:
        if not thresholds_ok(q, plan):
            return False
        T = len(plan)
        for x in xs:
            ref = _reference.semantic_trace(q, x, T)
            if not (ref['proof_ok'] and ref['terminal'] and ref['reconstructed'] == (q, x)
                    and ref['active'] == _reference.oracle_count(q, x)):
                return False
            phys = _reference.physical_trace(q, x, T, plan)
            if not phys['terminal'] or phys['reconstructed'] != (q, x):
                return False
            for t, (u, stored, e) in enumerate(phys['states'][:-1]):
                if max(u, stored).bit_length() > plan[t]['w']:
                    return False
                if e and (stored+e <= 0 or (stored+e) % 2):
                    return False
                if plan[t]['high'] and u >= 2**plan[t]['k'] and stored >= 2**plan[t]['k']:
                    return False
        return True
    for x in xs:
        u, v, e = q, x, 0
        for key in plan:
            lim = None if key is None or key[5] is None else 1 << key[5]
            V = v - e
            if V < 0 or (lim and u >= lim and (V ^ e) >= lim):
                return False
            if v & 1:
                e = int(u > v)
                u, v = min(u, v), abs(u - v) // 2
                if lim and u >= lim:
                    return False
            else:
                v //= 4 if v % 4 == 0 else 2
        if (u, v) != (1, 0):
            return False
    return True


def lazy_fields_ok(q, plan):
    lazy = [t for t, key in enumerate(plan) if key and key[9]]
    long_high = [t for t, key in enumerate(plan) if key and key[6] and key[0] - key[5] > WINDOW]
    if not lazy:
        return True
    first, final = lazy[0], lazy[-1]
    # The run is the long high ticks from the first, up to the last whose word is long enough to end at
    # (all of them, or a later end at a longer word: the ticks after it run their word whole).
    if lazy != long_high[:len(lazy)] or lazy != list(range(first, final + 1)) or final + 1 >= len(plan) or first < 1:
        return False
    K = [None if key is None else min(key[0], joint_threshold(q, t)) for t, key in enumerate(plan)]
    w = [None if key is None else key[0] for key in plan]
    if w[first - 1] != w[first] or K[first - 1] != K[first] + 1 or w[first] - K[first] != WINDOW + 1:
        return False
    if lazy == long_high:
        if w[final + 1] != w[final] - 1 or K[final + 1] != K[final] or w[final] - K[final] != WINDOW + 1:
            return False
    elif w[final] - K[final] <= WINDOW + 1 or plan[final + 1][9] or not plan[final + 1][6]:
        return False
    width = 0
    for t in lazy:
        raw, m_in, m_out, leave = plan[t][9]
        if raw != (t > first and K[t - 1] == K[t]) or (t > first and not raw and K[t - 1] != K[t] + 1):
            return False
        if m_in != width or leave != (t == final) or plan[t][5] != K[t]:
            return False
        if w[t] - K[t] - WINDOW + 1 > 1 << (m_out - 1) or m_out < 1:
            return False
        width = m_out
    return True


def lazy_states_ok(q, plan, xs):
    lazy = {t for t, key in enumerate(plan) if key and key[9]}
    if not lazy:
        return True
    first, final = min(lazy), max(lazy)
    mask = (1 << WINDOW) - 1
    for x in xs:
        u, v, e = q, x, 0
        p, phys = 0, None
        for t, key in enumerate(plan):
            V = v - e
            odd, two = v & 1, int(v % 4 == 0)
            if odd:
                un, vn, en = min(u, v), abs(u - v) // 2, int(u > v)
            else:
                un, vn, en = u, v // (4 if two else 2), e
            Vn = vn - en
            if t in lazy:
                w, k, (raw, _, _, leave) = key[0], key[5], key[9]
                cut = k + WINDOW
                if t == first:
                    phys, p = V, 0
                    if V >> (w - 1):
                        return False
                if raw:
                    r = phys >> (cut - 1) & 1
                    phys ^= p << (cut - 1)
                    p &= 1 - r
                if phys - (p << cut) != V or phys >> (w + 1):
                    return False
                z, hi = u >> k != 0, phys >> k
                if not odd and two:
                    hi >>= 1
                    r = hi >> (WINDOW - 1) & 1
                    hi ^= p << (WINDOW - 1)
                    p &= 1 - r
                if (z and (hi or p)) or (p and hi & mask != mask):
                    return False
                if odd:
                    word = u >> k if z else hi
                    borrow = en if word == 0 else (word if z else word - (p << WINDOW)) - (Vn >> (k - 1))
                    if borrow not in (0, 1):
                        return False
                    if borrow and word & mask == 0 and (hi >> WINDOW or u >> cut):
                        p = 1
                phys = Vn + (p << (cut - 1))
                if leave and w - k > WINDOW + 1:
                    if phys >> (w + 1) or phys - (p << (cut - 1)) != Vn:
                        return False
                elif leave:
                    r = phys >> (cut - 1) & 1
                    if phys >> (w - 1) != p & (1 - r) or (phys ^ (p << (cut - 1)) ^ (p & (1 - r)) << (w - 1)) != Vn:
                        return False
            u, v, e = un, vn, en
    return True


def lag_blocks(plan):
    """[(first tick, c, front)] or None when the far fields are not whole blocks."""
    blocks, t = [], 0
    while t < len(plan):
        far = plan[t] and plan[t][10]
        if not far:
            t += 1
            continue
        front, c, _, size = far
        for e in range(c):
            key = plan[t + e] if t + e < len(plan) else None
            if not (key and key[10] == (front, c, e, plan[t][0] + 1)):
                return None
        blocks.append((t, c, front))
        t += c
    return blocks


def lag_fields_ok(q, plan):
    blocks = lag_blocks(plan)
    if blocks is None:
        return False
    lazy = [t for t, key in enumerate(plan) if key and key[9]]
    for t, c, front in blocks:
        ticks = range(t, t + c)
        if not all(lazy[0] < s < lazy[-1] and plan[s][9] and plan[s][5] == joint_threshold(q, s) for s in ticks):
            return False
        K = [joint_threshold(q, s) for s in ticks]
        if any(front - 2 * e < K[e] + WINDOW + 2 - (e == c - 1) for e in range(c)):
            return False
        if plan[t + c] is None or front > plan[t + c][0] or front < 2 * c:
            return False
    return True


def top_free_ok(q, plan):
    """A block whose realign leaves X's top cell out must have the lazy word below 2^w on entry, the
    word's top cell being w. Under the flag U < 2^w by the width promise. Otherwise P = Y + p 2^cut with
    Y <= v: on a top_clear tick Y < 2^(w - 1); else u < 2^kappa, and v < u or v^2 <= u v^2 <= Psi <= B_t."""
    m = DialogMultiplier(Circuit(), Mod(q))
    for t, c, front in lag_blocks(plan) or []:
        if not m._top_free(plan, t):
            continue
        w, k = plan[t][0], plan[t][5]
        y = (1 << (w - 1)) - 1 if plan[t][1] else max((1 << k) - 1, isqrt((2 * q * q * (q - 1)) >> 2 * t))
        if y + (1 << (k + WINDOW)) >> w:
            return False
    return True


def thermo_ok(q, plan):
    """A block whose flag-off bits are counted as a thermometer (digits j = XOR of left_(c-m) over the m where
    m - 1 and m differ in bit j) must have, for every pattern of those bits that is not 0s then 1s, V's word
    under front - k'. Such a pattern has X = 0. With j the last tick whose flag comes back on, V is below
    2^K_c if the flag is still on at the end, else below 2^max(K_es, K_j + 1 - (c - 1 - es)), es the tick
    after j that turns it off; k' is the thermometer value plus every strip that can count."""
    m = DialogMultiplier(Circuit(), Mod(q))
    for t, c, front in lag_blocks(plan) or []:
        K = [plan[t + e][5] for e in range(c + 1)]
        if not m._thermo(K, front, c):
            continue
        if c < 3 or (c + 1) & c:
            return False
        for bits in range(1 << c):
            left = [bits >> e & 1 for e in range(c)]
            if left == sorted(left):
                continue
            j = max(e for e in range(1, c) if left[e - 1] > left[e])
            ones = [e for e in range(j, c) if left[e]]
            bound = max(K[ones[0]], K[j] + 1 - (c - 1 - ones[0])) if ones else K[c]
            value = 0
            for d in range(c.bit_length()):
                value |= (sum(left[c - n] for n in range(1, c + 1) if (n >> d) % 2 != ((n - 1) >> d) % 2) % 2) << d
            if front - value - 1 - sum(left[:-1]) < bound:
                return False
    return True


def lag_states_ok(q, plan, xs):
    blocks = lag_blocks(plan)
    if not blocks:
        return blocks is not None
    start = {t: (c, front) for t, c, front in blocks}
    lazy = [t for t, key in enumerate(plan) if key and key[9]]
    mask = (1 << WINDOW) - 1
    for x in xs:
        u, v, e = q, x, 0
        left, p, phys = 0, 0, None
        for t, key in enumerate(plan):
            odd, two = v & 1, int(v % 4 == 0)
            if odd:
                un, vn, en = min(u, v), abs(u - v) // 2, int(u > v)
            else:
                un, vn, en = u, v // (4 if two else 2), e
            if key and key[9]:
                # the lazy word's physical V, as lazy_states_ok follows it
                k, raw = key[5], key[9][0]
                cut = k + WINDOW
                if t == lazy[0]:
                    phys, p = v - e, 0
                z = u >> k != 0
                if t in start:
                    left, front = start[t]
                    lag, word = 0, (u if z else phys) >> front
                if left and (front - lag < cut + 2 - (left == 1) or (z and lag and u >> (front - lag)) or (z and (v - e) >> k)):
                    return False
                if raw:
                    r = phys >> (cut - 1) & 1
                    phys ^= p << (cut - 1)
                    p &= 1 - r
                hi = phys >> k
                if not odd and two:
                    hi >>= 1
                    r = hi >> (WINDOW - 1) & 1
                    hi ^= p << (WINDOW - 1)
                    p &= 1 - r
                if odd:
                    top = u >> k if z else hi
                    borrow = en if top == 0 else (top if z else top - (p << WINDOW)) - ((vn - en) >> (k - 1))
                    if borrow and top & mask == 0 and (hi >> WINDOW or u >> cut):
                        p = 1
                phys = (vn - en) + (p << (cut - 1))
                if left:
                    lag += (1 - int(un >> plan[t + 1][5] != 0)) + ((1 - odd) & two & (1 - z))
                    left -= 1
                    if word:
                        if not (un >> front == word or (phys >> (front - lag) == word and un >> front == 0)):
                            return False
                    elif phys >> (front - lag) or un >> front:
                        return False
            u, v, e = un, vn, en
    return True


def secp_inputs():
    q, rng = SECP, random.Random(2026)
    xs = {x for k in range(256) for d in (-1, 0, 1) for x in ((1 << k) + d, q - (1 << k) + d, (q >> k) + d)}
    a, b = 1, 1
    while b < 4 * q:
        xs |= {q * a // b + d for d in (-1, 0, 1)}
        a, b = b, a + b
    xs |= {rng.randrange(1, q) for _ in range(2000)}
    return sorted(x for x in xs if 0 < x < q)


def schedule_ok(q, xs, lag=None):
    plan = plan_of(q, lag)
    if _model.JUMP4:
        return (all(set(p) == {'w','k','high','carry','fmode','init','scratch'} for p in plan)
                and thresholds_ok(q, plan) and states_ok(q, plan, xs))
    return (thresholds_ok(q, plan) and states_ok(q, plan, xs)
            and lazy_fields_ok(q, plan) and lazy_states_ok(q, plan, xs)
            and lag_fields_ok(q, plan) and top_free_ok(q, plan) and thermo_ok(q, plan) and lag_states_ok(q, plan, xs))


def main():
    if _model.JUMP4:
        rows = []
        for q in TOY + [15361, 16381, 32749]:
            plan = plan_of(q)
            rows.append({"q": q, "high_ticks": sum(p['high'] for p in plan),
                         "lazy_ticks": 0, "lag_ticks": 0, "eager_only": True,
                         "inputs": q-1, "schedule_ok": schedule_ok(q, range(1, q))})
        xs = secp_inputs()
        plan = plan_of(SECP)
        rows.append({"q": "secp256k1", "high_ticks": sum(p['high'] for p in plan),
                     "lazy_ticks": 0, "lag_ticks": 0, "eager_only": True,
                     "inputs": len(xs), "schedule_ok": schedule_ok(SECP, xs)})
        # Unsupported lag is a rejected request, not claimed forced-lag coverage.
        try:
            schedule_ok(31, range(1, 31), lag=1)
        except ValueError as exc:
            assert "explicit lag" in str(exc)
        else:
            raise AssertionError("eager schedule accepted explicit lag")
        for row in rows:
            print(json.dumps(row), flush=True)
        return 0 if all(row["schedule_ok"] for row in rows) else 1
    rows = [{"q": q, "high_ticks": sum(1 for key in plan_of(q) if key and key[6]),
             "lazy_ticks": sum(1 for key in plan_of(q) if key and key[9]),
             "schedule_ok": schedule_ok(q, range(1, q))} for q in TOY]
    for lag in (1, 2, 3, 7):
        for q in TOY + [15361, 16381, 32749]:
            ticks = sum(1 for key in plan_of(q, lag) if key and key[10])
            if ticks:
                rows.append({"q": q, "lag_block": lag, "lag_ticks": ticks, "schedule_ok": schedule_ok(q, range(1, q), lag)})
    saved = DialogMultiplier.LAZY_END
    for end in (4, 5):
        DialogMultiplier.LAZY_END = end
        for q in TOY:
            plan = plan_of(q)
            ticks = sum(1 for key in plan if key and key[9])
            if any(key and key[6] and not key[9] and key[0] - key[5] > WINDOW for key in plan):
                rows.append({"q": q, "lazy_end": end, "lazy_ticks": ticks, "schedule_ok": schedule_ok(q, range(1, q))})
    DialogMultiplier.LAZY_END = saved
    xs = secp_inputs()
    for lag in (1, 3, 7, 15):
        rows.append({"q": "secp256k1", "lag_block": lag, "lag_ticks": sum(1 for key in plan_of(SECP, lag) if key and key[10]),
                     "inputs": len(xs), "schedule_ok": schedule_ok(SECP, xs, lag)})
    rows.append({"q": "secp256k1", "high_ticks": sum(1 for key in plan_of(SECP) if key and key[6]),
                 "lazy_ticks": sum(1 for key in plan_of(SECP) if key and key[9]),
                 "lag_ticks": sum(1 for key in plan_of(SECP) if key and key[10]), "inputs": len(xs), "schedule_ok": schedule_ok(SECP, xs)})
    for row in rows:
        print(json.dumps(row))
    return 0 if all(row["schedule_ok"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

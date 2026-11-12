"""Classical and bit-sliced gate checks of the dialog multiplier and pass.
Small primes cover every x; y is exhaustive below 300 and sampled above."""
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check import expand, structural
import schedule_check
from qec import flatten, sim
from qec.arith import Mod, load_const
from qec.builder import Circuit
from qec.emit import write_circuit
from qec import model as _model
if _model.JUMP4:
    import jump_gate_reference as _reference
from qec.model import tick_bound, tick_joint, tick_mul, tick_pass, tick_sweep, tick_tape, tick_widths
from qec.point_add import DialogMultiplier


def gate_mul(q: int, ys, inverse: bool = False, iters: int | None = None) -> int:
    """Check every x against the supplied y values."""
    return gate_pairs(q, [(x, y) for x in range(1, q) for y in ys], inverse, iters)


def gate_pairs(q: int, pairs, inverse: bool = False, iters: int | None = None) -> int:
    """Count bad output or dirty-scratch lanes; malformed builds raise."""
    pairs = list(pairs)
    C = Circuit(hierarchical=True)
    mod = Mod(q)
    X, Y = C.alloc(mod.n), C.alloc(mod.n)
    inputs = (list(X), list(Y))
    m = DialogMultiplier(C, mod)
    if iters is not None:
        m.iters = iters
    m.apply(X, Y, inverse=inverse)
    assert len(set(X + Y)) == 2 * mod.n, "aliased output register"
    buf = io.StringIO()
    write_circuit(buf, C, {"control": [], "x": X, "y": Y})
    ops = expand(flatten.load(io.StringIO(buf.getvalue())))
    structural(ops, C.num_wires)
    xs, values = zip(*pairs)
    init = sim.pack(xs, inputs[0]) | sim.pack(values, inputs[1])
    state = sim.run(ops, C.num_wires, init, len(pairs))
    expected = [y if x == 0 else y * (pow(x, -1, q) if inverse else x) % q
                for x, y in pairs]
    want = sim.pack(xs, X) | sim.pack(expected, Y)
    bad = 0
    for w, actual in enumerate(state):
        bad |= actual ^ want.get(w, 0)
    return bad.bit_count()


def b1_swept(cells):
    """In Bezout order, the reverse of the tape, rotate b left by one in each run of no-subtract
    cells, so each B1 moves to its run's end."""
    out = list(reversed(cells))
    t = 0
    while t < len(out):
        end = t
        while end < len(out) and not out[end][0]:
            end += 1
        if end > t:
            bs = [b for _, b in out[t:end]]
            bs = bs[1:] + bs[:1]
            out[t:end] = [(0, b) for b in bs]
        t = end + 1
    return list(reversed(out))


def _lazy_pending(m, plan, states, cells):
    """Per lazy tick, the pending borrow p as the tick finds it, after the raw bit took it, and as the
    tick leaves it. V's physical word is the model's plus p at cut = kappa + LAZY_WINDOW. A bit that
    comes to the window's top takes p and clears it if the bit was 1; a subtract's borrow e that finds
    the window 0 sets p when U or V has a 1 from cut up, and is the swap otherwise."""
    R = m.LAZY_WINDOW
    out, p, V = {}, 0, None
    for t, key in enumerate(plan):
        if not (key and key[9]):
            continue
        k = key[5]
        cut = k + R
        U, Y, _ = states[t]
        Yn = states[t + 1][1]
        add, bit = cells[t]
        if key[9][1] == 0:
            V, p = Y, 0
        found = p
        if key[9][0]:
            r = V >> (cut - 1) & 1
            V ^= p << (cut - 1)
            p &= 1 - r
        mid = p
        z, hi = U >> k != 0, V >> k
        if not add and bit:
            hi >>= 1
            r = hi >> (R - 1) & 1
            hi ^= p << (R - 1)
            p &= 1 - r
        if add:
            X = U >> k if z else hi
            e = bit if X == 0 else (X if z else X - (p << R)) - (Yn >> (k - 1))
            if e and X & ((1 << R) - 1) == 0 and (hi >> R or U >> cut):
                p ^= 1
        V = (Yn >> (k - 1)) + (p << R) << (k - 1) | Yn & ((1 << (k - 1)) - 1)
        out[t] = (found, mid, p)
    return out


def _jump4_pass_detail(q, xs, iters=None):
    """Check the recorded producer on independent endpoints and fresh maps."""
    xs = list(xs)
    maker = Circuit()
    m = DialogMultiplier(maker, Mod(q))
    if iters is not None:
        m.iters = iters
    plan = m._jump4_eager_plan()
    expected_plan = _reference.schedule_reference(q, m.iters)
    if plan != expected_plan:
        raise ValueError("schedule metadata differs from independent reference")
    n = q.bit_length()
    cap = (5*n+3)//4
    widths = [min(n, max(1, cap-t)) for t in range(m.iters)]
    tail = widths[-1]
    lent = min(tail, n-tail)
    remaining = tail-lent
    external = 4*m.iters-(n-tail)-lent
    history_size = 1+sum(p['carry']+int(p['init'])+int(p['fmode'] in ('same','drop')) for p in expected_plan)
    assert m._jump4_pool_sizes() == (lent, remaining, external)
    assert m._jump4_junk_size() == history_size
    sizes = [n, n+external, history_size, 3]
    block = maker.record_shape(m._jump4_euclid_pass, sizes)
    assert len(set(block.flat_in)) == len(block.flat_in)
    assert len(set(block.flat_out)) == len(block.flat_out)
    refs = [None if x == 0 else _reference.physical_trace(q, x, m.iters, expected_plan) for x in xs]
    endpoint_u = [q if r is None else r['endpoint'][0] for r in refs]
    endpoint_pool = [0 if r is None else r['endpoint'][1] | (r['packed_tape'] << remaining) for r in refs]
    endpoint_junk = [0 if r is None else r['packed_history'] for r in refs]
    endpoint_header = [0 if r is None else r['header'] for r in refs]
    results = {}
    for inverse in (False, True):
        C = Circuit()
        pad, spectators = C.alloc(5 if inverse else 2), C.alloc(3)
        regs = [C.alloc(size) for size in sizes]
        entry = [list(reg) for reg in regs]
        if inverse:
            values = [endpoint_u, endpoint_pool, endpoint_junk, endpoint_header]
        else:
            values = [[q]*len(xs), xs, [0]*len(xs), [0]*len(xs)]
        init = sim.pack([5]*len(xs), spectators)
        for values_for_reg, reg in zip(values, entry):
            init.update(sim.pack(values_for_reg, reg))
        C.apply(block, regs, inverse=inverse)
        assert len(set(sum(regs, []))) == sum(sizes), "aliased pass register"
        structural(C.prims, C.num_wires)
        state = sim.run(C.prims, C.num_wires, init, len(xs))
        values = ([[q]*len(xs), xs, [0]*len(xs), [0]*len(xs)] if inverse else
                  [endpoint_u, endpoint_pool, endpoint_junk, endpoint_header])
        want = sim.pack([5]*len(xs), spectators)
        for values_for_reg, reg in zip(values, regs):
            want.update(sim.pack(values_for_reg, reg))
        bad = 0
        for w, actual in enumerate(state):
            bad |= actual ^ want.get(w, 0)
        results['inverse_bad' if inverse else 'forward_bad'] = bad.bit_count()
        results['inverse_wires' if inverse else 'forward_wires'] = C.num_wires
    return results


def _jump4_controls():
    """Inject faults into actual model steps, gates and schedule metadata."""
    from unittest.mock import patch
    results = {}
    for name in ('strip3', 'missing_swap', 'mandatory_half'):
        reached = [0]
        def wrong_step(u, v, fault=name):
            reached[0] += 1
            if v == 0:
                return u, v, (0, 0, 0)
            d = 0
            while d < (3 if fault == 'strip3' else 4) and v % (2**(d+1)) == 0:
                d += 1
            v //= 2**d
            g = v % 2
            swap = int(g and u > v and fault != 'missing_swap')
            if swap:
                u, v = v, u
            v -= g*u
            if fault == 'mandatory_half':
                v //= 2
            return u, v, (d, g, swap)
        with patch.object(_model, '_jump4_step', wrong_step):
            detected = not all(offset_checks(17, range(1, 17)))
        results[name] = dict(detected=detected, reached=reached[0])
    original = DialogMultiplier._jump4_euclid_pass
    for name in ('header', 'code', 'offset', 'carry', 'flag'):
        reached = [0]
        q = 65521 if name == 'flag' else 31
        def wrong_gate(self, U, pool, junk, H, fault=name):
            original(self, U, pool, junk, H)
            reached[0] += 1
            if fault == 'header':
                target = H[0]
            elif fault == 'code':
                target = self._jump4_coded_tape(pool)[0]
            elif fault == 'offset':
                target = junk[0]
            else:
                used = 1
                target = None
                for p in self._jump4_eager_plan():
                    if p['init']:
                        if fault == 'flag':
                            target = junk[used]
                            break
                        used += 1
                    if p['carry'] and fault == 'carry':
                        target = junk[used]
                        break
                    used += p['carry'] + int(p['fmode'] in ('same','drop'))
                assert target is not None, "fault target not reached"
            self.C.x(target)
        with patch.object(DialogMultiplier, '_jump4_euclid_pass', wrong_gate):
            detail = _jump4_pass_detail(q, [1, 3, q-1, q//2])
        results[name] = dict(detail, reached=reached[0],
                             detected=detail['forward_bad'] > 0 and detail['inverse_bad'] > 0)
    original_plan = DialogMultiplier._jump4_eager_plan
    for name, field in [('wrong_k', 'k'), ('wrong_width', 'w')]:
        reached = [0]
        def wrong_plan(self, key=field):
            reached[0] += 1
            plan = original_plan(self)
            plan[0][key] += 1
            return plan
        with patch.object(DialogMultiplier, '_jump4_eager_plan', wrong_plan):
            detected = not schedule_check.schedule_ok(31, range(1, 31))
        results[name] = dict(detected=detected, reached=reached[0])
    assert all(r['detected'] and r['reached'] > 0 for r in results.values()), results
    return results


def _junk_value(m, q, x):
    """The pass's junk register: E as the last tick found it, then per tick its chain's lowest carries
    c_2 .. c_(carry+1) of U + NOT(V') on its width, V' being V ^ E at bit 0 with a no-subtract
    tick's second strip already done on V[1:]; then the flag wire the tick makes, and the wire its
    flag update leaves; then a lazy tick's wires; then add AND E, the offset a subtract retires. The
    flag before a tick is [U >= 2^kappa]. A wire that carries it holds the flag of the tick that last
    wrote it. The first lazy tick brings V's extra wire and the counter, which end 0, and the wire for
    p. A raw tick moves p to a new wire, the old one keeping p as the tick found it, a tick of a lag
    block leaves two AND NOT flag, and every lazy tick leaves two AND p."""
    if _model.JUMP4:
        return _reference.physical_trace(q, x, m.iters, m._jump4_eager_plan())['packed_history']
    states = tick_pass(q, x, m.iters)[0]
    cells = tick_tape(q, x, m.iters)[2]
    plan = m._tick_plan()
    flags = [None if key is None or key[5] is None else int(states[t][0] >> key[5] != 0)
             for t, key in enumerate(plan)]
    bits = [states[m.iters - 1][2]]
    holder = p_wire = count_wire = None
    pending = _lazy_pending(m, plan, states, cells)
    counter = m._lazy_width(plan)
    for t, (key, (U, V, E), (add, bit)) in enumerate(zip(plan, states, cells)):
        if key is None or key[3]:
            continue
        w, top_clear, narrow, _, carry, kappa, high, fmode, init = key[:9]
        if carry:
            vv = V ^ E
            if not add and bit and w >= 2:
                hi = w - 2 if top_clear else w - 1
                vb = [vv >> i & 1 for i in range(w)]
                vb[1:hi + 1] = vb[2:hi + 1] + [0]
                vv = sum(b << i for i, b in enumerate(vb))
            nb = ~vv & ((1 << w) - 1)
            c = nb & 1
            for i in range(1, carry + 1):
                a, b = U >> i & 1, nb >> i & 1
                c = a & b | a & c | b & c
                bits.append(c)
        if init:
            holder = len(bits)
            bits.append(flags[t])
        if fmode == "same":
            bits[holder] = flags[t + 1]
            bits.append(add & flags[t])
        elif fmode == "drop":
            holder = len(bits)
            bits.append(flags[t + 1])
        if key[9]:
            found, mid, left = pending[t]
            if key[9][1] == 0:
                p_wire = len(bits) + 1
                count_wire = p_wire + 1
                bits.extend([0] * (2 + counter))
            if key[9][3] and key[0] - key[5] > m.LAZY_WINDOW + 1:
                # A run that ends at a longer word leaves the counter as it stands: rho = pi - 1 on
                # m_out bits, pi the ones of U and of V's physical word from cut up.
                cut, Un, Yn = key[5] + m.LAZY_WINDOW, states[t + 1][0], states[t + 1][1]
                rho = bin(Un >> cut).count("1") + bin((Yn + (left << (cut - 1))) >> (cut - 1)).count("1") - 1
                rho &= (1 << key[9][2]) - 1
                for i in range(counter):
                    bits[count_wire + i] = rho >> i & 1
            if key[9][0]:
                bits[p_wire] = found
                p_wire = len(bits)
                bits.append(0)
            if key[10]:
                bits.append((1 - add) & bit & (1 - flags[t]))
            bits.append((1 - add) & bit & mid)
            bits[p_wire] = left
        bits.append(add & E)
    return sum(b << k for k, b in enumerate(bits))


def joint_domain_ok(q):
    """Before tick t, u * v * max(u, v) <= 2 * q^2 * (q - 1) / 4^t for every x, so U and V are
    never both at or above 2^K_t, and a subtract leaves U below it."""
    if _model.JUMP4:
        T = tick_bound(q)
        expected = _reference.schedule_reference(q, T)
        return (tick_joint(q, T) == [p['k'] for p in expected]
                and all(_reference.semantic_trace(q, x, T)['proof_ok']
                        for x in range(1, q)))
    T = tick_bound(q)
    K, bound = tick_joint(q, T), 2 * q * q * (q - 1)
    for x in range(1, q):
        states, cells = tick_pass(q, x, T)
        for t, (U, V, E) in enumerate(states[:T]):
            v = V + E
            if U * v * max(U, v) > bound >> 2 * t or (U >> K[t] and (V ^ E) >> K[t]):
                return False
            if cells[t][0] and states[t + 1][0] >> K[t]:
                return False
    return True


def _pass_bad(q, xs, iters):
    if _model.JUMP4:
        result = _jump4_pass_detail(q, xs, iters)
        return result['forward_bad'] + result['inverse_bad']
    C = Circuit()
    mod = Mod(q)
    m = DialogMultiplier(C, mod)
    if iters is not None:
        m.iters = iters
    U, V = C.alloc(mod.n), C.alloc(mod.n)
    inputs = list(V)
    pool = V + C.alloc(m.tape_external)
    load_const(C, U, q)
    junk = C.alloc(m.junk_width)
    m._euclid_pass(U, pool, junk)
    G = m.coded_tape(pool)
    for j in range(m.blocks):
        m._decode(G, j, C.alloc(len(m._block_free(j, m.swept))), m.swept)
    freed = sum(len(m._block_free(j, m.swept)) for j in range(m.blocks))
    assert len(set(U + pool + G)) == len(set(U + pool)) + freed, "aliased pass register"
    structural(C.prims, C.num_wires)
    state = sim.run(C.prims, C.num_wires, sim.pack(xs, inputs), len(xs))
    # Keep the reference model independent of tick_pass and tick_sweep.
    cells = [tick_tape(q, x, m.iters)[2] for x in xs]
    if m.swept:
        cells = [b1_swept(c) for c in cells]
    tapes = [sum((add + 2 * bit) << (2 * i) for i, (add, bit) in enumerate(c)) for c in cells]
    junks = [_junk_value(m, q, x) for x in xs]
    want = sim.pack([q if x == 0 else 1 for x in xs], U) | sim.pack(tapes, G) | sim.pack(junks, junk)
    bad = 0
    for w, actual in enumerate(state):
        bad |= actual ^ want.get(w, 0)
    return bad.bit_count()


def gate_pass(q: int, iters: int | None = None) -> int:
    """Check terminal words, exact tape and all scratch on remapped inputs."""
    return _pass_bad(q, [x or 1 for x in range(q)], iters)


def gate_pass_zero(q: int) -> int:
    """Mismatched bits for a V = 0 caller against the old generic result: q kept, idle tape,
    clean scratch. Late tape takes U's vacated wires, so that caller is outside the pass's domain
    and this must be nonzero; the multiplier never makes the call, by the remap control."""
    return _pass_bad(q, [0], None)


def width_edge_inputs(q: int) -> list[tuple[int, int]]:
    """Schedule-edge powers and neighbours, boundaries, and seed-2026 samples."""
    rng = random.Random(2026)
    xs = {x for k in range(q.bit_length())
          for x in (2**k, 2**k - 1, 2**k + 1) if 0 <= x < q}
    xs |= {0, 1, 3, q - 1, q - 2, (q - 1) // 2, (q + 1) // 2}
    xs |= {rng.randrange(q) for _ in range(64)}
    return [(x, rng.choice([0, 1, q - 1, rng.randrange(q)])) for x in sorted(xs)]


def offset_checks(q, xs):
    """Compare held states and swept cells with the independent semantic tape."""
    T = tick_bound(q)
    if _model.JUMP4:
        plan = _reference.schedule_reference(q, T)
        offset_ok = tape_ok = True
        for x in xs:
            ref = _reference.semantic_trace(q, x, T)
            u, v, cells = tick_tape(q, x, T)
            tape_ok &= ((u, v) == ref['endpoint'] and cells == ref['tape']
                        and ref['reconstructed'] == (q, x)
                        and ref['active'] == _reference.oracle_count(q, x))
            phys = _reference.physical_trace(q, x, T, plan)
            header, codes = _model.jump4_physical_tape(q, x, T)
            offset_ok &= (header == phys['header'] and codes == phys['codes']
                          and phys['terminal'] and phys['reconstructed'] == (q, x)
                          and all(U >= 0 and V >= 0 and E in (0, 1)
                                  and (not E or V+E > 0 and (V+E) % 2 == 0)
                                  for U, V, E in phys['states']))
        return bool(offset_ok), bool(tape_ok)
    offset_ok = tape_ok = True
    for x in xs:
        states, cells = tick_pass(q, x, T)
        if len(states) != T + 1 or len(cells) != T:
            return False, False
        u, v = q, x
        for i, (U, V, E) in enumerate(states):
            offset_ok &= U == u and V + E == v and V >= 0
            if i < T:
                if v & 1:
                    if u > v:
                        u, v = v, u
                    v = (v - u) // 2
                else:
                    v >>= 2 if v % 4 == 0 else 1
        swept, final = tick_sweep(cells, states[-1][2])
        tape_ok &= swept == tick_tape(q, x, T)[2] and states[-1][1:] == (0, 0) and final == 0
    return offset_ok, tape_ok


def trace(q, x):
    if _model.JUMP4:
        ref = _reference.semantic_trace(q, x, tick_bound(q))
        return [max(u, v).bit_length() for u, v in ref['post_strip'][:ref['active']]]
    u, v, sizes = q, x, []
    while v:
        sizes.append(max(u, v).bit_length())
        if v & 1:
            if u > v:
                u, v = v, u
            v = (v - u) // 2
        else:
            v >>= 2 if v % 4 == 0 else 1
    return sizes


rows = []
_enabled_controls = _jump4_controls() if _model.JUMP4 else None
for q in [3, 5, 7, 11, 13, 31, 43, 61, 127, 251, 509, 1021, 4093]:
    B, W = tick_bound(q), tick_widths(q, tick_bound(q))
    worst = 0
    width_ok = True
    if _model.JUMP4:
        expected_plan = _reference.schedule_reference(q, B)
        width_ok = (B == (5*q.bit_length()+3)//4
                    and W == [p['w'] for p in expected_plan])
    for x in range(q):
        t = trace(q, x)
        worst = max(worst, len(t))
        width_ok &= all(s <= W[i] for i, s in enumerate(t))
    ys = range(q) if q < 300 else random.Random(q).sample(range(q), 8)
    mul_ok = all(tick_mul(q, x, y) == (x * y % q if x else y) for x in range(q) for y in ys)
    pass_end = all(tick_tape(q, x or 1, B)[:2] == (1, 0) for x in range(q))
    remap_control = [x for x in range(q) if tick_tape(q, x, B)[:2] != (1, 0)] == [0]
    short_rejected = any(tick_tape(q, x or 1, worst - 1)[1] != 0 for x in range(q))
    if _model.JUMP4:
        short_rejected &= any(not _reference.semantic_trace(q, x, worst-1)['terminal']
                              for x in range(1, q))
    short_rejected &= gate_mul(q, ys, iters=worst - 1) > 0
    pass_gates = gate_pass(q) == 0
    pass_zero_outside = gate_pass_zero(q) > 0
    forward_bad = gate_mul(q, ys)
    inverse_bad = gate_mul(q, ys, inverse=True)
    pass_offset, pass_tape = offset_checks(q, range(1, q))
    joint_ok = joint_domain_ok(q)
    high_ticks = (sum(p['high'] for p in DialogMultiplier(Circuit(), Mod(q))._jump4_eager_plan())
                  if _model.JUMP4 else sum(1 for key in DialogMultiplier(Circuit(), Mod(q))._tick_plan() if key and key[6]))
    rows.append({"q": q, "worst_iterations": worst, "bound": B,
                 "width_schedule_ok": width_ok, "multiplier_ok": mul_ok,
                 "pass_end_ok": pass_end, "remap_control_ok": remap_control,
                 "pass_short_rejected": short_rejected, "pass_gates_ok": pass_gates,
                 "multiplier_gates_ok": forward_bad == inverse_bad == 0,
                 "forward_bad": forward_bad, "inverse_bad": inverse_bad,
                 "pass_offset_ok": pass_offset, "pass_tape_ok": pass_tape, "pass_zero_outside_domain": pass_zero_outside,
                 "joint_domain_ok": joint_ok, "schedule_ok": schedule_check.schedule_ok(q, range(1, q)),
                 "high_ticks": high_ticks})
    if _model.JUMP4 and q == 3:
        rows[-1]['controls'] = _enabled_controls
    print(json.dumps(rows[-1]), flush=True)
q = 2**256 - 2**32 - 977
rng = random.Random(1)
pairs = [(rng.randrange(q), rng.randrange(q)) for _ in range(2000)]
ok = all(tick_mul(q, x, y) == (x * y % q if x else y) for x, y in pairs)
pass_end = all(tick_tape(q, x or 1, tick_bound(q))[:2] == (1, 0)
               for x in [0, 1, q - 1] + [x for x, _ in pairs])
offset_ok = all(offset_checks(q, [1, q - 1] + [x for x, _ in pairs]))
edge_pairs = [(x, y) for x, y in width_edge_inputs(q) if x != 0]
width_edge_ok = gate_pairs(q, edge_pairs) == 0 and gate_pairs(q, edge_pairs, inverse=True) == 0
schedule_ok = (all(schedule_check.schedule_ok(p, range(1, p)) for p in schedule_check.TOY)
               and schedule_check.schedule_ok(q, schedule_check.secp_inputs()))
print(json.dumps({"q": "secp256k1", "bound": tick_bound(q), "random_pairs": 2000,
                  "multiplier_ok": ok, "pass_end_ok": pass_end, "offset_ok": offset_ok,
                  "width_edge_ok": width_edge_ok, "schedule_ok": schedule_ok,
                  "schedule_primes": schedule_check.TOY, "width_edge_lanes": len(edge_pairs)}))
keys = ("width_schedule_ok", "multiplier_ok", "pass_end_ok", "remap_control_ok",
        "pass_short_rejected", "pass_gates_ok", "multiplier_gates_ok",
        "pass_offset_ok", "pass_tape_ok", "pass_zero_outside_domain", "joint_domain_ok", "schedule_ok")
sys.exit(0 if ok and pass_end and offset_ok and width_edge_ok and schedule_ok and all(r["worst_iterations"] <= r["bound"]
         and all(r[k] for k in keys) for r in rows) else 1)

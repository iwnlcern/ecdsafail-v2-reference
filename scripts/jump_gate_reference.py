"""Independent integer expectations for capped-strip transcripts and schedules.

No circuit, model, or simulator imports. The pinned integer oracle contributes
only its isolated iteration-count function; its sampling driver never runs.
"""
import ast
from functools import lru_cache
import hashlib
from pathlib import Path

ORACLE_SHA256 = '8bbddcf9b8823ade5efca20d5354e885871d2724145b57f6b0dc2b7f469289a6'


def _strip(value):
    if not value:
        return 4
    shift = 0
    while shift < 4 and value % (2 ** (shift + 1)) == 0:
        shift += 1
    return shift


@lru_cache(maxsize=1)
def _count_function():
    path = Path(__file__).resolve().parent.parent / 'receipts/jump4/oracle.py.txt'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ORACLE_SHA256:
        raise ValueError('oracle source pin changed')
    function = next(n for n in ast.parse(raw).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'jump')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['jump'](4)


def oracle_count(q, x):
    return _count_function()(q, x)


@lru_cache(maxsize=64)
def _schedule(q, ticks):
    if q <= 2 or q % 2 == 0:
        raise ValueError('odd modulus greater than two required')
    n = q.bit_length()
    cap = (5 * n + 3) // 4
    if not 0 <= ticks <= cap:
        raise ValueError('schedule length outside cap')
    rows = []
    for t in range(ticks):
        width = min(n, max(1, cap - t))
        # Least positive fifth-power threshold at the post-strip phase.
        k = 1
        while (1 << (5 * k + 4 * t)) < q ** 5:
            k += 1
        k = min(k, width)
        rows.append(dict(w=width, k=k, high=width-k >= 3,
                         carry=0, fmode=None, init=False, scratch=0))
    high = [i for i, row in enumerate(rows) if row['high']]
    if high:
        first, final = high[0], high[-1]
        assert high == list(range(first, final + 1))
        rows[first]['init'] = True
        for i in high:
            if i == final:
                rows[i]['fmode'] = 'final'
            else:
                delta = rows[i]['k'] - rows[i+1]['k']
                assert delta in (0, 1)
                rows[i]['fmode'] = 'drop' if delta else 'same'
    reserved = 1 + sum(int(row['init']) + int(row['fmode'] in ('same', 'drop'))
                       for row in rows)
    remaining = 2236 - reserved
    assert remaining >= 0
    for row in reversed(rows):
        chain = row['k'] if row['high'] else row['w']
        row['carry'] = min(remaining, max(chain-1, 0))
        remaining -= row['carry']
    for row in rows:
        high_words = row['w'] - row['k'] + 1 if row['high'] else 0
        flag_update = int(bool(row['fmode']) and row['fmode'] != 'same')
        row['scratch'] = high_words + flag_update
        if row['init']:
            row['scratch'] = max(row['scratch'], max(0, row['w']-row['k']-2))
    return tuple(tuple(row.items()) for row in rows)


def schedule_reference(q, ticks):
    # Return fresh metadata: mutation of a consumer's copy cannot poison a cache.
    return [dict(row) for row in _schedule(q, ticks)]


def semantic_trace(q, x, ticks):
    if not 0 <= x < q:
        raise ValueError('canonical input required')
    plan = schedule_reference(q, ticks)
    u, v, tape, states, boundaries, active = q, x, [], [(q, x)], [], 0
    proof_ok = True
    for t in range(ticks):
        if v == 0:
            tape.append((0, 0, 0)); states.append((u, v))
            boundaries.append((u, v))
            continue
        d = _strip(v)
        v //= 2 ** d
        boundaries.append((u, v))
        w, k = plan[t]['w'], plan[t]['k']
        proof_ok &= (max(u, v).bit_length() <= w
                     and not (u >= 2**k and v >= 2**k)
                     and (u * v * max(u, v)**3) * 2**(4*t) < q**5)
        g = v % 2
        swap = int(g == 1 and u > v)
        tape.append((d, g, swap))
        if swap:
            u, v = v, u
        v -= g * u
        states.append((u, v)); active += 1
    a, b = u, v
    for d, g, swap in reversed(tape):
        b += g * a
        if swap:
            a, b = b, a
        b *= 2**d
    return dict(tape=tape, active=active, endpoint=(u, v), states=states,
                post_strip=boundaries, proof_ok=bool(proof_ok),
                reconstructed=(a, b), cap=(5*q.bit_length()+3)//4,
                terminal=(u, v) == (1, 0), widths=[p['w'] for p in plan],
                joints=[p['k'] for p in plan])


def physical_trace(q, x, ticks, plan):
    expected = schedule_reference(q, ticks)
    if plan != expected:
        raise ValueError('schedule metadata differs from independent reference')
    if not 0 < x < q:
        raise ValueError('physical domain requires nonzero canonical input')
    header = _strip(x)
    u, true_v, e = q, x // 2**header, 0
    states, codes = [(u, true_v, 0)], []
    # Index zero is the final live offset; every other slot is retained history.
    history = [0] * (1 + sum(p['carry'] + int(p['init'])
                            + int(p['fmode'] in ('same', 'drop')) for p in plan))
    used, flag_index, active = 1, None, 0
    for p in plan:
        stored = true_v - e
        assert stored >= 0 and (not e or true_v > 0 and true_v % 2 == 0)
        active += int(true_v != 0)
        if p['init']:
            flag_index = used; used += 1
            history[flag_index] = int(u >= 2**p['k'])
        # Carry into bit j is the overflow of the two low j-bit addends.
        # The live offset is toggled before the complemented subtraction input.
        difference_input = stored ^ e
        for j in range(2, p['carry']+2):
            modulus = 2**j
            history[used] = ((u % modulus) + (modulus-1-difference_input % modulus)) // modulus
            used += 1
        previous_flag = history[flag_index] if p['fmode'] else None
        g = true_v % 2
        swap = int(g == 1 and u > true_v)
        b = swap if g else e
        if swap:
            u, true_v = true_v, u
        true_v -= g * u
        next_shift = _strip(true_v)
        true_v //= 2**next_shift
        e = int((swap or e) and next_shift == 4 and true_v > 0 and true_v % 2 == 0)
        codes.append(next_shift-1 + 4*g + 8*b)
        if p['fmode'] == 'same':
            history[used] = previous_flag * g; used += 1
            history[flag_index] = int(u >= 2**p['k'])
        elif p['fmode'] == 'drop':
            history[used] = previous_flag; used += 1
            history[flag_index] = int(u >= 2**(p['k']-1))
        states.append((u, true_v-e, e))
    history[0] = e
    assert used == len(history)
    a, b = u, true_v
    for code in reversed(codes):
        g, sw, shift = (code >> 2) & 1, code >> 3, (code & 3)+1
        b = b * 2**shift + g*a
        if g and sw:
            a, b = b, a
    reconstructed = a, b * 2**header
    return dict(header=header, codes=codes, states=states, active=active,
                endpoint=states[-1], history=history,
                packed_history=sum(bit << i for i, bit in enumerate(history)),
                packed_tape=sum(code << (4*i) for i, code in enumerate(codes)),
                reconstructed=reconstructed, terminal=(u,true_v,e)==(1,0,0))

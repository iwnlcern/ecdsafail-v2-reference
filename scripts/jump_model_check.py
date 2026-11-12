"""Independent semantic checks: wrong strip, swap, or post-subtract half must fail.

The oracle is the isolated AST of jump(m), never its sampling driver.
No physical Jump circuit or tape encoding is exercised here.
"""
import argparse
import ast
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qec import model

SOURCE_SHA256 = '8bbddcf9b8823ade5efca20d5354e885871d2724145b57f6b0dc2b7f469289a6'


def oracle(path):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256, 'oracle source pin changed'
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'jump')
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns['jump'](4)


def primes(limit):
    return ([2] if limit >= 2 else []) + [q for q in range(3, limit + 1, 2)
            if all(q % d for d in range(2, int(q**0.5) + 1))]


def check_case(q, x, reference, ys=None):
    n = q.bit_length()
    bound = model.tick_bound(q)
    assert bound <= (5 * n + 3) // 4, ('bound', q, bound)
    widths = model.tick_widths(q, bound)
    joints = model.tick_joint(q,bound)
    # Refined Psi boundary width; check physical post-strip boundaries below.
    expected_widths = [min(n, max(1, (5 * n + 3) // 4 - t)) for t in range(bound)]
    assert widths == expected_widths, ('schedule', q, widths)
    end_u, end_v, tape = model.tick_tape(q, x, bound)
    assert len(tape) == bound and (end_u, end_v) == (1, 0), ('terminal', q, x)
    if q>2:
        _,codes=model.jump4_physical_tape(q,x,bound)
        assert codes[-1] in (3,7),('first reverse domain',q,x,codes[-1])
        terminal=max(i for i,c in enumerate(codes) if c&4)
        assert codes[terminal]==7 and all(c==3 for c in codes[terminal+1:]),('terminal/padding b0',q,x)

    u, v, active, saw_padding = q, x, 0, False
    for t, cell in enumerate(tape):
        assert len(cell) == 3, ('cell arity', q, x, cell)
        d, g, swap = cell
        if not v:
            assert cell == (0, 0, 0), ('padding', q, x, t, cell)
            saw_padding = True
            continue
        assert not saw_padding
        # Compute the expected record from arithmetic, independently of the model helper.
        want_d = min(4, (v & -v).bit_length() - 1)
        v //= 2**want_d
        assert max(u,v).bit_length()<=widths[t],('post-strip width',q,x,t,u,v)
        K=joints[t]
        assert not(u>=1<<K and v>=1<<K),('joint',q,x,t,K)
        assert u*v*max(u,v)**3*(1<<(4*t))<q**5,('Psi',q,x,t)
        want_g = v % 2
        want_swap = int(bool(want_g) and u > v)
        assert cell == (want_d, want_g, want_swap), ('transition', q, x, t, cell)
        if swap:
            u, v = v, u
        if g:
            v -= u
        active += 1
    assert (u, v) == (end_u, end_v)
    assert active == reference(q, x), ('oracle count', q, x, active, reference(q, x))
    # Independent integer inverse reconstruction checks the entire record order.
    a, b = 1, 0
    for d, g, swap in reversed(tape):
        if g:
            b += a
        if swap:
            a, b = b, a
        b *= 2**d
    assert (a, b) == (q, x), ('reverse reconstruction', q, x, a, b)
    for y in ((1, 2, q - 1) if ys is None else ys):
        assert model.tick_bezout(q, y, tape) == (0, x * y % q), ('multiplier map', q, x, y)
        assert model.tick_mul(q, x, y) == x * y % q
    return active


def run_small(limit, reference):
    cases = 0
    worst = 0
    for q in primes(limit):
        for x in range(1, q):
            worst = max(worst, check_case(q, x, reference))
            cases += 1
    return dict(prime_limit=limit, primes=len(primes(limit)), cases=cases, worst=worst)


def fixture():
    # Hand-derived 7,3: (7,3)->(3,4)->(1,2)->(1,0), then frozen padding.
    got = model.tick_tape(7, 3, 4)
    assert got == (1, 0, [(0, 1, 1), (2, 1, 1), (1, 1, 0), (0, 0, 0)]), ('fixture', got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oracle', type=Path, default=Path(__file__).resolve().parent.parent/'receipts/jump4/oracle.py.txt')
    ap.add_argument('--small-limit', type=int, default=1021)
    ap.add_argument('--secp-cases', type=int, default=20000)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    reference = oracle(args.oracle)
    model.JUMP4 = True
    fixture()
    if args.smoke:
        print(json.dumps(dict(smoke=check_case(7, 3, reference))))
        return
    small = run_small(args.small_limit, reference)
    print(json.dumps(dict(small=small)), flush=True)
    q = model.SECP256K1_Q
    rng = random.Random(23)
    xs = [1, q - 1] + [2**k for k in range(256)] + [rng.randrange(1, q) for _ in range(args.secp_cases)]
    worst = max(check_case(q, x, reference, [rng.randrange(q)]) for x in xs)
    print(json.dumps(dict(secp=dict(seed=23, random=args.secp_cases, cases=len(xs), worst=worst))), flush=True)
    # Mutate the actual production transition used by tick_tape, one defect at a time.
    original = model._jump4_step
    controls = {}
    for name in ('strip3', 'missing_swap', 'mandatory_half'):
        def broken(u, v, name=name):
            if not v:
                return u, v, (0, 0, 0)
            d = min(3 if name == 'strip3' else 4, (v & -v).bit_length() - 1)
            v >>= d
            g = v & 1
            swap = int(bool(g) and u > v and name != 'missing_swap')
            if swap:
                u, v = v, u
            if g:
                v -= u
                if name == 'mandatory_half':
                    v >>= 1
            return u, v, (d, g, swap)
        model._jump4_step = broken
        try:
            run_small(args.small_limit, reference)
        except AssertionError as exc:
            controls[name] = str(exc)
        else:
            raise AssertionError(('control escaped', name))
        finally:
            model._jump4_step = original
    correct = model.tick_bezout
    def old_order(q, y, tape):
        r, s = y, 0
        for d, g, swap in reversed(tape):
            s = ((s << d) + g * r) % q
            if swap:
                r, s = s, r
        return r, s
    witness = model.tick_tape(13, 2, model.tick_bound(13))[2]
    assert old_order(13, 1, witness) == (11, 1)
    assert correct(13, 1, witness) == (0, 2)
    model.tick_bezout = old_order
    try:
        run_small(args.small_limit, reference)
    except AssertionError as exc:
        controls['old_reverse_order'] = str(exc)
    else:
        raise AssertionError('old-order control escaped')
    finally:
        model.tick_bezout = correct
    root=Path(__file__).resolve().parent.parent
    print(json.dumps(dict(controls_detected=controls,status='pass',argv=sys.argv,
                         head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
                         sources={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in
                                  ('qec/model.py','qec/point_add.py','qec/two_bit.py','scripts/jump_model_check.py')})),flush=True)


if __name__ == '__main__':
    main()

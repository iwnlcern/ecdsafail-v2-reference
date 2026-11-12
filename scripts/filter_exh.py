"""Exhaustive small-prime check of the kernel's constant blocks, for the XOR-sum filter in qec/arith.py, with its control.

    python3 scripts/filter_exh.py [--bound 1021] [--control-bound 251] [--out FILE]

At every odd prime q from 5 to the bound, `cmodadd_const(ctrl, c, b)` at every constant c in [1, q) and
`creflect(ctrl, a, b)` at every a in [0, q) are recorded alone and run bit-sliced on every b in [0, q) with ctrl 0
and 1 and all scratch at 0. Each must give its map, (b + ctrl c) mod q and ctrl ? (a - b) mod q : b, return ctrl, and
leave every other wire at 0.

Control. A copy of this tree's `qec` whose `gt_const_tree` is also told that its lowest leaf is 0, though that leaf
holds ~b at the constant's lowest 1, runs the same check up to the control bound and must fail blocks.
The last line printed is a JSON summary.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTROL = ("    zeros = [where[i] for i in idx if not value >> i & 1] + list(temps)\n",
           "    zeros = [where[i] for i in idx if not value >> i & 1] + list(temps) + [where[ones[0]]]\n")

WORKER = r'''
import json, sys
from pathlib import Path
tree, bound = Path(sys.argv[1]).resolve(), int(sys.argv[2])
sys.path.insert(0, str(tree))
import qec.arith as ar
from qec.arith import Mod
from qec.builder import CCX, CX, SWAP, X, Circuit
from qec.curve import is_prime
assert Path(ar.__file__).resolve().is_relative_to(tree), ar.__file__


def run(blk, q):
    lanes = [(c, x) for c in (0, 1) for x in range(q)]
    full = (1 << len(lanes)) - 1
    st = {}
    ctrl, b = blk.flat_in[0], blk.flat_in[1:]
    st[ctrl] = sum(1 << i for i, (c, _) in enumerate(lanes) if c)
    for j, w in enumerate(b):
        st[w] = sum(1 << i for i, (_, x) in enumerate(lanes) if x >> j & 1)
    ops = blk.ops
    for p in range(0, len(ops), 4):
        k, x, y, t = ops[p:p + 4]
        if k == CCX:
            st[t] = st.get(t, 0) ^ (st.get(x, 0) & st.get(y, 0))
        elif k == CX:
            st[y] = st.get(y, 0) ^ st.get(x, 0)
        elif k == X:
            st[x] = st.get(x, 0) ^ full
        elif k == SWAP:
            st[x], st[y] = st.get(y, 0), st.get(x, 0)
        else:
            raise ValueError(k)
    out = blk.flat_out[1:]
    res = [sum(((st.get(w, 0) >> i) & 1) << j for j, w in enumerate(out)) for i in range(len(lanes))]
    others = set(st) - set(blk.flat_out)
    return lanes, res, all(st[w] == 0 for w in others) and st.get(blk.flat_out[0], 0) == st[ctrl]


tot = {"primes": 0, "blocks": 0, "failing blocks": 0, "ccx": 0}
fails = []
for q in range(5, bound + 1, 2):
    if not is_prime(q):
        continue
    tot["primes"] += 1
    mod = Mod(q)
    n = mod.n
    for op, consts in (("add", range(1, q)), ("reflect", range(q))):
        for const in consts:
            C = Circuit()
            if op == "add":
                f = lambda k, r: mod.cmodadd_const(C, k[0], const, r)
            else:
                f = lambda k, r: mod.creflect(C, k[0], const, r)
            blk = C.record_shape(f, [1, n])
            tot["blocks"] += 1
            tot["ccx"] += list(blk.ops[0::4]).count(CCX)
            lanes, res, clean = run(blk, q)
            want = [(x + c * const) % q if op == "add" else ((const - x) % q if c else x) for c, x in lanes]
            if res != want or not clean:
                tot["failing blocks"] += 1
                if len(fails) < 20:
                    fails.append([q, op, const, sum(r != w for r, w in zip(res, want)), clean])
print(json.dumps({**tot, "first failures": fails}))
'''


def check(tree, bound):
    p = subprocess.run([sys.executable, "-I", "-B", "-c", WORKER, str(tree), str(bound)], capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"worker on {tree} exited {p.returncode}: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bound", type=int, default=1021)
    ap.add_argument("--control-bound", type=int, default=251)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    tree = check(ROOT, args.bound)
    with tempfile.TemporaryDirectory() as td:
        ctl = Path(td) / "control"
        shutil.copytree(ROOT / "qec", ctl / "qec", ignore=shutil.ignore_patterns("__pycache__"))
        p = ctl / "qec/arith.py"
        text = p.read_text()
        if text.count(CONTROL[0]) != 1:
            raise SystemExit(f"control anchor matches {text.count(CONTROL[0])} times")
        p.write_text(text.replace(*CONTROL))
        control = check(ctl, args.control_bound)
    problems = [f"{tree['failing blocks']} failing blocks"] * bool(tree["failing blocks"])
    problems += ["control not rejected"] * (not control["failing blocks"])
    summary = {"summary": True, "bound": args.bound, **{k: tree[k] for k in ("primes", "blocks", "failing blocks", "ccx")},
               "first failures": tree["first failures"][:5],
               "control": {"bound": args.control_bound, **{k: control[k] for k in ("primes", "blocks", "failing blocks")},
                           "rejected": bool(control["failing blocks"])},
               "problems": problems, "ok": not problems}
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()

"""The tick's 7-block carry-select compare, checked against the comparison it replaces, with its control.

    python3 scripts/compare_check.py [--out FILE]

The tick compares a[1:] with low[1:] by carry-select blocks on its borrowed clean scratch, in place of one
`gt_into(C, a[1:], low[1:], k1, ctrl=z)`. This row builds three copies of `qec`:
  tree       this tree;
  reference  this tree with the compare block replaced by that one `gt_into` call;
  control    this tree with the forward combine's first gate dropped.
1. Equivalence. For q = 7, 31 and 127 it builds `DialogMultiplier._body` and runs it on every (x, y) with x in [1, q)
   and y in [0, q), bit-sliced: the tree's x and y must equal the reference's on every lane, every other wire must end
   at 0, and the reversed circuit must restore the inputs. The control must differ from the reference.
2. Components. `scripts/component_check.py --bezout`, the Bezout tick's exhaustive and boundary rows, must pass on the
   tree and fail on the control.
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
PRIMES = (7, 31, 127)
BLOCK = ("        # Carry-select blocks borrow", "            C.free([cin])\n")
GT = "        gt_into(C, a[1:], low[1:], k1, ctrl=z)\n"
CONTROL = ("            for gate, args in combine:\n", "            for gate, args in combine[:1] + combine[2:]:\n")

WORKER = r'''
import hashlib, json, sys
from pathlib import Path
tree = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(tree))
from qec import builder as b
from qec.arith import Mod
from qec.point_add import DialogMultiplier
import qec.point_add as pa
assert Path(pa.__file__).resolve().is_relative_to(tree)


def run(ops, s, mask):
    for k, x, y, t in ops:
        if k == b.X:
            s[x] ^= mask
        elif k == b.CX:
            s[y] ^= s[x]
        elif k == b.CCX:
            s[t] ^= s[x] & s[y]
        else:
            s[x], s[y] = s[y], s[x]


rows = []
for q in json.loads(sys.argv[2]):
    C = b.Circuit()
    M = DialogMultiplier(C, Mod(q))
    n = M.mod.n
    x, y = C.alloc(n), C.alloc(n)
    xin, yin = list(x), list(y)
    M._body(x, y)
    cases = [(u, v) for u in range(1, q) for v in range(q)]
    L = len(cases)
    mask = (1 << L) - 1
    s = [0] * C.num_wires
    for j, w in enumerate(xin):
        s[w] = sum(((u >> j) & 1) << i for i, (u, v) in enumerate(cases))
    for j, w in enumerate(yin):
        s[w] = sum(((v >> j) & 1) << i for i, (u, v) in enumerate(cases))
    initial = list(s)
    ops = [tuple(C.ops[i:i + 4]) for i in range(0, len(C.ops), 4)]
    run(ops, s, mask)
    live = set(x) | set(y)
    clean = all(val == 0 for w, val in enumerate(s) if w not in live)
    result = hashlib.sha256(b"".join(s[w].to_bytes((L + 7) // 8, "little") for w in x + y)).hexdigest()
    run(list(reversed(ops)), s, mask)
    rows.append({"q": q, "lanes": L, "wires": C.num_wires, "ccx": sum(o[0] == b.CCX for o in ops), "result": result,
                 "clean": clean, "reverses": s == initial})
print(json.dumps(rows))
'''


def copy_tree(td, name, edit):
    dst = Path(td) / name
    shutil.copytree(ROOT / "qec", dst / "qec", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "scripts", dst / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    p = dst / "qec" / "two_bit.py"
    s = p.read_text()
    p.write_text(edit(s))
    return dst


def to_reference(s):
    i = s.index(BLOCK[0])
    j = s.index(BLOCK[1], i) + len(BLOCK[1])
    assert s.count(BLOCK[0]) == 1 and GT not in s
    return s[:i] + GT + s[j:]


def to_control(s):
    assert s.count(CONTROL[0]) == 1
    return s.replace(*CONTROL)


def worker(tree):
    p = subprocess.run([sys.executable, "-I", "-B", "-c", WORKER, str(tree), json.dumps(PRIMES)], capture_output=True, text=True)
    if p.returncode:
        return {"error": f"exit {p.returncode}: {p.stderr.strip()[-600:]}"}
    return json.loads(p.stdout.strip().splitlines()[-1])


def components(tree):
    p = subprocess.run([sys.executable, "-I", "-B", str(tree / "scripts" / "component_check.py"), "--bezout"],
                       cwd=tree, capture_output=True, text=True)
    try:
        out = json.loads(p.stdout)
    except ValueError:
        out = {}
    rows = {k: v for k, v in out.items() if isinstance(v, dict)}
    return {"exit": p.returncode, "rows": len(rows), "failing rows": sorted(k for k, v in rows.items() if v.get("failures")),
            "lanes": sum(v.get("lanes", 0) for v in rows.values())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    problems = []
    with tempfile.TemporaryDirectory() as td:
        ref = copy_tree(td, "reference", to_reference)
        ctl = copy_tree(td, "control", to_control)
        got, want, bad = worker(ROOT), worker(ref), worker(ctl)
        comp, comp_ctl = components(ROOT), components(ctl)
    for name, r in (("tree", got), ("reference", want), ("control", bad)):
        if isinstance(r, dict):
            problems.append(f"{name}: {r['error']}")
    if not problems:
        for g, w in zip(got, want):
            if g["result"] != w["result"]:
                problems.append(f"q {g['q']}: the compare differs from gt_into")
            if not (g["clean"] and g["reverses"]):
                problems.append(f"q {g['q']}: scratch not clean or inverse not exact")
        if all(c["result"] == w["result"] and c["clean"] for c, w in zip(bad, want)):
            problems.append("control not rejected by the equivalence")
    if comp["exit"] or comp["failing rows"] or not comp["rows"]:
        problems.append(f"component_check --bezout: exit {comp['exit']}, failing {comp['failing rows']}")
    if not comp_ctl["exit"]:
        problems.append("control not rejected by component_check --bezout")
    summary = {"summary": True, "primes": list(PRIMES),
               "lanes": sum(r["lanes"] for r in got) if isinstance(got, list) else None,
               "ccx tree vs reference": [[g["ccx"], w["ccx"]] for g, w in zip(got, want)] if isinstance(got, list) and isinstance(want, list) else None,
               "control differs at": [c["q"] for c, w in zip(bad, want) if c["result"] != w["result"] or not c["clean"]]
               if isinstance(bad, list) and isinstance(want, list) else None,
               "components": comp, "components control": comp_ctl, "problems": problems, "ok": not problems}
    if args.out:
        args.out.write_text(json.dumps({**summary, "tree": got, "reference": want, "control": bad}, indent=1) + "\n")
    print(json.dumps(summary))
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()

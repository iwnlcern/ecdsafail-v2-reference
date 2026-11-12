"""Proof that the XOR-sum filter (`_live` in qec/arith.py) changes no block's map, with its control.

    python3 scripts/filter_check.py [--out FILE]

`gt_const_tree` and `add_lookahead` (when its caller passes `known`) emit a recorded gate list through `_live`, which
drops every CX or CCX one of whose controls is an empty XOR sum of the values the function is told its wires hold.
This check records each block twice, as this tree emits it and with `_live` replaced by the identity, which is the
code without the filter:
- the kernel's four constant blocks, `cmodadd_const` at q - a, q - b and 3a + K (K the Karatsuba square's offset,
  0 off that path) and `creflect` at a, for A = G to 8G and 8 random multiples of G drawn from seed 7;
- tick 0's selected add, `_add_selected_tree(V, [g0], [f1])`, which does not depend on the addend.
Each pair runs on one binary decision diagram from ctrl and b as variables and every other wire at 0, the blocks'
own convention, and must leave every wire the same function of (ctrl, b) over every b in [0, 2^n), with no domain.
The filtered list must also be the unfiltered one with ops removed in order.

Control. A copy of this tree's `qec` whose `gt_const_tree` is also told that its lowest leaf is 0, though that leaf
holds ~b at the constant's lowest 1, must fail: some block must then differ. A tree with no `_live` is reported as
having no filter, and fails.
The last line printed is a JSON summary.
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED, RANDOM_ADDENDS, G_MULTIPLES = 7, 8, range(1, 9)
CONTROL = ("    zeros = [where[i] for i in idx if not value >> i & 1] + list(temps)\n",
           "    zeros = [where[i] for i in idx if not value >> i & 1] + list(temps) + [where[ones[0]]]\n")

WORKER = r'''
import json, sys
from array import array
from pathlib import Path
tree = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(tree))
sys.setrecursionlimit(100000)
import qec.arith as ar
import qec.point_add as pa
from qec.arith import Mod, _add_selected_tree
from qec.builder import CCX, CX, SWAP, X, Circuit
from qec.curve import SECP256K1
assert Path(ar.__file__).resolve().is_relative_to(tree), ar.__file__
if not hasattr(ar, "_live"):
    print(json.dumps({"filter": False}))
    sys.exit(0)
exec(sys.argv[3])
q = SECP256K1.q
mod = Mod(q)
n = mod.n
K = pa.KaratsubaMulsub(mod).K if hasattr(pa, "KaratsubaMulsub") and pa.KaratsubaMulsub.applies(q) else 0
live = ar._live


def record(f, filtered):
    ar._live = live if filtered else (lambda ops, known: list(ops))
    try:
        C = Circuit()
        blk = C.record_shape(lambda k, r: f(C, k[0], r), [1, n])
    finally:
        ar._live = live
    ops = blk.ops
    return [tuple(ops[i:i + 4]) for i in range(0, len(ops), 4)], list(blk.flat_in), list(blk.flat_out)


def removed(base, filt):
    out, j = [], 0
    for i, o in enumerate(base):
        if j < len(filt) and filt[j] == o:
            j += 1
        else:
            out.append(i)
    return out if j == len(filt) else None


def run(bdd, ops, st):
    for k, x, y, t in ops:
        if k == CCX:
            st[t] = bdd.XOR(st.get(t, 0), bdd.AND(st.get(x, 0), st.get(y, 0)))
        elif k == CX:
            st[y] = bdd.XOR(st.get(y, 0), st.get(x, 0))
        elif k == X:
            st[x] = bdd.NOT(st.get(x, 0))
        elif k == SWAP:
            st[x], st[y] = st.get(y, 0), st.get(x, 0)
        else:
            raise ValueError(k)
    return st


blocks = []
for a, b in json.loads(sys.argv[2]):
    for name, op, const in (("add q-a", "add", q - a), ("add q-b", "add", q - b), ("add 3a+K", "add", 3 * a + K),
                            ("reflect a", "reflect", a)):
        if op == "add":
            f = (lambda c: lambda C, k, r: mod.cmodadd_const(C, k, c, r))(const)
        else:
            f = (lambda c: lambda C, k, r: mod.creflect(C, k, c, r))(const)
        blocks.append((name, hex(const), hex(a), f))
f1 = (1 << n) - q + 1
mt = pa.TICK0_SCRATCH - n - pa.TICK0_COPIES - 1
blocks.append(("tick0 add", hex(f1), None, lambda C, k, r: _add_selected_tree(C, r, [k], [f1], max_temps=mt)))
for name, const, a, f in blocks:
    bo, bin_, bout = record(f, False)
    fo, fin, fout = record(f, True)
    assert (bin_, bout) == (fin, fout), name
    rm = removed(bo, fo)
    bdd = BDD()
    init = {w: bdd.var(i) for i, w in enumerate(bin_)}
    sb, sf = run(bdd, bo, dict(init)), run(bdd, fo, dict(init))
    bad = sorted(w for w in set(sb) | set(sf) if sb.get(w, 0) != sf.get(w, 0))
    print(json.dumps({"name": name, "const": const, "a": a, "ops": len(bo), "ccx": sum(o[0] == CCX for o in bo),
                      "filtered ccx": sum(o[0] == CCX for o in fo), "removed in order": rm is not None,
                      "removed ccx": None if rm is None else sum(bo[i][0] == CCX for i in rm),
                      "removed cx": None if rm is None else sum(bo[i][0] == CX for i in rm),
                      "wires differing": len(bad), "equal on every input": not bad, "bdd nodes": len(bdd.nodes)}), flush=True)
'''

BDD_SOURCE = r'''
class BDD:
    """Reduced ordered BDD; node 0 is false, 1 is true; variable order is the index."""

    def __init__(self):
        self.nodes = [None, None]
        self.unique = {}
        self.memo_and, self.memo_xor = {}, {}

    def mk(self, v, lo, hi):
        if lo == hi:
            return lo
        key = (v, lo, hi)
        r = self.unique.get(key)
        if r is None:
            r = len(self.nodes)
            self.nodes.append(key)
            self.unique[key] = r
        return r

    def var(self, v):
        return self.mk(v, 0, 1)

    def top(self, u):
        return self.nodes[u][0] if u > 1 else 1 << 30

    def cofactors(self, u, v):
        return (self.nodes[u][1], self.nodes[u][2]) if self.top(u) == v else (u, u)

    def AND(self, u, w):
        if u == 0 or w == 0:
            return 0
        if u == 1:
            return w
        if w == 1 or u == w:
            return u
        if u > w:
            u, w = w, u
        r = self.memo_and.get((u, w))
        if r is None:
            v = min(self.top(u), self.top(w))
            (ul, uh), (wl, wh) = self.cofactors(u, v), self.cofactors(w, v)
            r = self.memo_and[(u, w)] = self.mk(v, self.AND(ul, wl), self.AND(uh, wh))
        return r

    def XOR(self, u, w):
        if u == 0:
            return w
        if w == 0:
            return u
        if u == w:
            return 0
        if u > w:
            u, w = w, u
        r = self.memo_xor.get((u, w))
        if r is None:
            if u == 1:
                r = self.NOT(w)
            else:
                v = min(self.top(u), self.top(w))
                (ul, uh), (wl, wh) = self.cofactors(u, v), self.cofactors(w, v)
                r = self.mk(v, self.XOR(ul, wl), self.XOR(uh, wh))
            self.memo_xor[(u, w)] = r
        return r

    def NOT(self, u):
        if u < 2:
            return 1 - u
        r = self.memo_xor.get(("not", u))
        if r is None:
            v, lo, hi = self.nodes[u]
            r = self.memo_xor[("not", u)] = self.mk(v, self.NOT(lo), self.NOT(hi))
        return r
'''


def addends():
    sys.path.insert(0, str(ROOT))
    from qec.curve import SECP256K1, SECP256K1_G
    rng = random.Random(SEED)
    scalars = [*G_MULTIPLES, *(rng.randrange(2, 1 << 255) for _ in range(RANDOM_ADDENDS))]
    return scalars, [SECP256K1_G if s == 1 else SECP256K1.mul(s, SECP256K1_G) for s in scalars]


def check(tree, points):
    p = subprocess.run([sys.executable, "-I", "-B", "-c", WORKER, str(tree), json.dumps(points), BDD_SOURCE],
                       capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"worker on {tree} exited {p.returncode}: {p.stderr[-2000:]}")
    rows = [json.loads(line) for line in p.stdout.splitlines() if line.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    scalars, points = addends()
    rows = check(ROOT, points)
    problems = []
    if rows and rows[0].get("filter") is False:
        summary = {"summary": True, "filter": False, "problems": ["no filter in qec/arith.py"], "ok": False}
        print(json.dumps(summary))
        sys.exit(1)
    if len(rows) != 4 * len(points) + 1:
        problems.append(f"{len(rows)} blocks, expected {4 * len(points) + 1}")
    problems += [f"{r['name']} {r['const'][:12]}: differs on {r['wires differing']} wires" for r in rows if not r["equal on every input"]]
    problems += [f"{r['name']} {r['const'][:12]}: not the unfiltered list with ops removed" for r in rows if not r["removed in order"]]
    with tempfile.TemporaryDirectory() as td:
        ctl = Path(td) / "control"
        shutil.copytree(ROOT / "qec", ctl / "qec", ignore=shutil.ignore_patterns("__pycache__"))
        p = ctl / "qec/arith.py"
        text = p.read_text()
        if text.count(CONTROL[0]) != 1:
            raise SystemExit(f"control anchor matches {text.count(CONTROL[0])} times")
        p.write_text(text.replace(*CONTROL))
        ctl_rows = check(ctl, points[:1])
    ctl_bad = [r["name"] for r in ctl_rows if not r["equal on every input"]]
    if not ctl_bad:
        problems.append("control not rejected")
    removed = [r["removed ccx"] for r in rows if r["removed ccx"] is not None]
    summary = {"summary": True, "filter": True, "seed": SEED, "addends": [hex(s) for s in scalars], "blocks": len(rows),
               "all equal on every input": all(r["equal on every input"] for r in rows),
               "all removed in order": all(r["removed in order"] for r in rows),
               "removed ccx": sum(removed), "removed ccx per kernel block": [min(removed[:-1] or [0]), max(removed[:-1] or [0])],
               "tick0 removed ccx": rows[-1]["removed ccx"] if rows else None,
               "control": {"addend": "G", "blocks": len(ctl_rows), "blocks differing": ctl_bad, "rejected": bool(ctl_bad)},
               "problems": problems, "ok": not problems}
    if args.out:
        args.out.write_text(json.dumps({**summary, "rows": rows, "control rows": ctl_rows}, indent=1) + "\n")
    print(json.dumps(summary))
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()

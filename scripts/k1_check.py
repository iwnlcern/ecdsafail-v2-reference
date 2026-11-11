"""Certificate for the kernel's K1 cut (`_K1_CUTS` and `_cut_k1` in qec/point_add.py).

    python3 scripts/k1_check.py [--seed S] [--sample N] [--out FILE]

K1 replaces two CCX of `Mod.cmodadd_const(gen, 3a, X)` at secp256k1's G with CX from gen. The table is keyed by
(q, 3a mod q), so it applies at A = G only. The cut is exact on every b < q, the call's contract: X is canonical
there, as the multiplier's contract already requires, and the cut adds no condition the circuit did not rest on.
This script checks, on this tree:

1. The table has one entry, at secp256k1 and 3 * G.x mod q.
2. In the block `cmodadd_const(ctrl, 3a, b)` recorded alone, each listed op is CCX(b[254], ctrl, t), and the
   block's CCX on those two controls are exactly the listed pair and the control pair, ops 2920 and 5378.
3. A proof over every input. The block is a carry chain over one register, so every wire is a small binary decision
   diagram over (ctrl, b_0 .. b_255). The block runs symbolically once; each candidate runs again from its first
   changed op, and holds only if every wire ends as the same function of (ctrl, b) on the whole domain b < q,
   for both values of ctrl. The cut pair must hold. Each half of it alone must fail. So must the control, the one
   other pair of the same form, CCX(b[254], ctrl) at ops 2920 and 5378 cut the same way.
4. The emitted circuit. G is built hierarchically with the table and with it empty. The definitions must be
   identical, and the top level must differ at exactly the listed ops, each CCX(X[254], gen, t) become CX(gen, t).
   Every other addend in accept.py's sample, [1]G to [8]G and N random multiples drawn from seed S, has no key in
   the table, so `_cut_k1` leaves its circuit as it was without the cut.
The last line printed is a JSON summary.
"""
import argparse
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import qec.point_add as pa  # noqa: E402
from qec.arith import Mod  # noqa: E402
from qec.builder import CCX, CX, SWAP, X, Circuit  # noqa: E402
from qec.curve import SECP256K1, SECP256K1_G, SECP256K1_R  # noqa: E402
from qec.emit import write_circuit  # noqa: E402

sys.setrecursionlimit(100000)
G_MULTIPLES = range(1, 9)
CONTROL = (2920, 5378)


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

    def less_than(self, bits, c):
        """[the number on `bits` (LSB first) < c]."""
        f = 0
        for i, var in enumerate(bits):
            bi = self.var(var)
            eq = bi if c >> i & 1 else self.NOT(bi)
            gt = self.NOT(bi) if c >> i & 1 else 0
            f = self.XOR(gt, self.AND(eq, f))
        return f


def run(bdd, ops, st, lo, hi, mutate):
    for p in range(lo, hi):
        for op, a, b, t in mutate.get(p, (tuple(ops[4 * p:4 * p + 4]),)):
            if op == CCX:
                st[t] = bdd.XOR(st[t], bdd.AND(st[a], st[b]))
            elif op == CX:
                st[b] = bdd.XOR(st[b], st[a])
            elif op == X:
                st[a] = bdd.NOT(st[a])
            elif op == SWAP:
                st[a], st[b] = st[b], st[a]
            else:
                raise ValueError(op)
    return st


def block_proof(q, value, cut):
    """The block cmodadd_const(ctrl, value, b) at q, recorded alone, and each candidate's verdict."""
    mod = Mod(q)
    n = mod.n
    C = Circuit(True)
    blk = C.record_shape(lambda k, b: mod.cmodadd_const(C, k[0], value, b), [1, n])
    assert not blk.calls, "the block makes calls"
    ops = blk.ops
    size = len(ops) // 4
    ctrl, b = blk.flat_in[0], list(blk.flat_in[1:])
    sites = [{"op": p, "prim": list(ops[4 * p:4 * p + 4])} for p in cut]
    shape_ok = all(s["prim"][:3] == [CCX, b[254], ctrl] for s in sites)
    ccx_at = [p for p in range(size) if ops[4 * p] == CCX]
    same_form = [p for p in ccx_at if sorted(ops[4 * p + 1:4 * p + 3]) == sorted([b[254], ctrl])]

    def to_cx(p):
        return {p: ((CX, ctrl, ops[4 * p + 3], ops[4 * p + 3]),)}

    candidates = {"cut": list(cut), **{f"half {p}": [p] for p in cut}, "control": list(CONTROL)}
    bdd = BDD()
    wires = sorted(blk.wires)
    st0 = {w: 0 for w in wires}
    st0[ctrl] = bdd.var(0)
    for j, w in enumerate(b):
        st0[w] = bdd.var(j + 1)
    dom = bdd.less_than(list(range(1, n + 1)), q)
    starts = {min(ps) for ps in candidates.values()}
    snaps, st = {}, dict(st0)
    for p in range(size):
        if p in starts:
            snaps[p] = dict(st)
        run(bdd, ops, st, p, p + 1, {})
    ref = st
    verdicts = {}
    for name, ps in candidates.items():
        mut = {}
        for p in ps:
            mut.update(to_cx(p))
        first = min(ps)
        got = run(bdd, ops, dict(snaps[first]), first, size, mut)
        holds = all(bdd.AND(bdd.XOR(ref[w], got[w]), dom) == 0 for w in wires)
        verdicts[name] = {"ops": ps, "prims": [list(ops[4 * p:4 * p + 4]) for p in ps], "holds on every b < q": holds}
    return {"block ops": size, "block ccx": len(ccx_at), "sites": sites, "sites are CCX(b[254], ctrl)": shape_ok,
            "CCX on (b[254], ctrl)": same_form, "only the cut and the control": sorted(same_form) == sorted([*cut, *CONTROL]),
            "bdd nodes": len(bdd.nodes), "candidates": verdicts}


def build(A, table):
    saved = pa._K1_CUTS
    pa._K1_CUTS = table
    try:
        gen = pa.ControlledPointAdd(SECP256K1, A)
        C = gen.build(True)
        buf = io.StringIO()
        write_circuit(buf, C, gen.layout)
        return list(C.prims), json.loads(buf.getvalue())["definitions"]
    finally:
        pa._K1_CUTS = saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sample", type=int, default=16)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    q = SECP256K1.q
    key = (q, 3 * SECP256K1_G[0] % q)
    table = dict(pa._K1_CUTS)
    problems = []
    if list(table) != [key]:
        problems.append(f"the table's keys are {list(table)}, expected only secp256k1 at 3 G.x")
    cut = table.get(key, ())
    proof = block_proof(q, 3 * SECP256K1_G[0], cut) if cut else {}
    v = proof.get("candidates", {})
    if not proof.get("sites are CCX(b[254], ctrl)"):
        problems.append("a listed op is not CCX(b[254], ctrl)")
    if not proof.get("only the cut and the control"):
        problems.append(f"the CCX on (b[254], ctrl) are {proof.get('CCX on (b[254], ctrl)')}, expected the cut and {list(CONTROL)}")
    if not v.get("cut", {}).get("holds on every b < q"):
        problems.append("the cut does not hold on every b < q")
    problems += [f"{name} holds, so the proof does not bite" for name, r in v.items()
                 if name != "cut" and r["holds on every b < q"]]
    prims_cut, defs_cut = build(SECP256K1_G, table)
    prims_plain, defs_plain = build(SECP256K1_G, {})
    diff = [j // 4 for j in range(0, max(len(prims_cut), len(prims_plain)), 4)
            if prims_cut[j:j + 4] != prims_plain[j:j + 4]]
    changed = [{"top op": i, "without": prims_plain[4 * i:4 * i + 4], "with": prims_cut[4 * i:4 * i + 4]} for i in diff]
    if defs_cut != defs_plain:
        problems.append("the cut changes a definition")
    if len(prims_cut) != len(prims_plain) or len(diff) != len(cut):
        problems.append(f"the top level differs at {len(diff)} ops, expected {len(cut)}")
    for c in changed:
        w, p = c["with"], c["without"]
        if not (p[0] == CCX and w == [CX, p[2], p[3], p[3]]):
            problems.append(f"top op {c['top op']}: {p} -> {w} is not CCX(x, gen, t) -> CX(gen, t)")
    rng = random.Random(args.seed)
    addends = [*G_MULTIPLES, *(rng.randrange(G_MULTIPLES[-1] + 1, SECP256K1_R) for _ in range(args.sample))]
    keyed = []
    for k in addends:
        A = SECP256K1_G if k == 1 else SECP256K1.mul(k, SECP256K1_G)
        if (q, 3 * A[0] % q) in table:
            keyed.append(k)
    if keyed != [1]:
        problems.append(f"addends with a table entry: {keyed}, expected [1]")
    summary = {"table": [[hex(k[0]), hex(k[1]), list(c)] for k, c in table.items()], "entries": len(table),
               "block": {k: proof.get(k) for k in ("block ops", "block ccx", "bdd nodes", "sites are CCX(b[254], ctrl)",
                                                   "CCX on (b[254], ctrl)", "only the cut and the control")},
               "verdicts": {name: r["holds on every b < q"] for name, r in v.items()},
               "top level changes": changed, "definitions identical": defs_cut == defs_plain, "definitions": len(defs_cut),
               "seed": args.seed, "addends": len(addends), "addends with a table entry": keyed,
               "problems": problems, "ok": not problems}
    if args.out:
        args.out.write_text(json.dumps({**summary, "proof": proof}, indent=1) + "\n")
    print(json.dumps(summary))
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()

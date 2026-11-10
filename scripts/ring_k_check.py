"""The Euclid step at every ring size, exhaustively on the step domain, as built and after the scored post-pass.

    python3 scripts/ring_k_check.py [--control] [--out FILE]

It first requires point_add.U6_RING == 217, the cap this branch sets. Then for w = 9..12 and every ring size
k = w - 2 .. 2w (54 circuits), _euclid_step(w, k) is run on every input of the step domain (U odd, every U[1:], V,
E), as scripts/sweep_check.py item 2 does for k = w - 2 and 2w, in two forms: the flat circuit, and the step built
hierarchically, written by qec/emit.py with the depth pass on, run through qec/gate_pass.py's main() (the post-pass
the board scores), parsed by qec/flatten.py and expanded by scripts/check.py's expand(). Each must equal the step at
k = w - 2 on U, V, E and the cell, and leave the ring and scratch 0. U6_RING enters only through _ring_size, which
picks k, so this covers what a change of U6_RING changes in the step; at 256 bits the ring sizes it changes are reached
only by the sampled checks. With --control the ring's last wire starts at 1, and the check must fail at every k in
both forms. One JSON line per width; the last line is the summary. Exits 0 when the check holds (or, with
--control, fails everywhere)."""
import io
import json
import os
import runpy
import sys
import tempfile
import argparse
from pathlib import Path

TREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TREE))
from qec import flatten, point_add, sim  # noqa: E402
import qec.gate_pass as gp  # noqa: E402
from qec.arith import Mod  # noqa: E402
from qec.builder import Circuit  # noqa: E402
from qec.emit import write_circuit  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--control", action="store_true", help="the ring's last wire starts at 1; the check must fail")
ap.add_argument("--out", type=Path)
args = ap.parse_args()
assert point_add.U6_RING == 217, f"U6_RING is {point_add.U6_RING}, not 217"
WMIN, WMAX, DIRTY, POST = 9, 12, args.control, True
EXPAND = runpy.run_path(str(TREE / "scripts" / "check.py"))["expand"]


def lane_mask(i, N):
    L = 1 << N
    period = 1 << (i + 1)
    block = ((1 << (1 << i)) - 1) << (1 << i)
    return ((1 << L) - 1) // ((1 << period) - 1) * block


def build(w, k, hierarchical):
    point_add.RIPPLE_MAX_W = 0
    q = next(p for p in (251, 509, 1021, 2039, 4093, 8191, 16381, 32749) if p.bit_length() >= w)
    C = Circuit(hierarchical=hierarchical)
    m = point_add.DialogMultiplier(C, Mod(q))
    U, V, E, cell, ring = C.alloc(w), C.alloc(w), C.alloc(1), C.alloc(2), C.alloc(k)
    C.apply(m._euclid_step(w, k), [U, V, E, cell, ring])
    return C, U, V, E, cell, ring


def post_ops(C, io_wires, tmp):
    """The step as the board scores it: emitted, post-passed, parsed and expanded."""
    buf = io.StringIO()
    write_circuit(buf, C, {"control": [], "x": io_wires, "y": []})
    src, dst = os.path.join(tmp, "in.json"), os.path.join(tmp, "out.json")
    with open(src, "w") as fh:
        fh.write(buf.getvalue())
    out, sys.stdout = sys.stdout, io.StringIO()
    try:
        gp.main(src, dst)
        log = json.loads(sys.stdout.getvalue())
    finally:
        sys.stdout = out
    with open(dst) as fh:
        doc = flatten.load(fh)
    return EXPAND(doc), doc["num_wires"], log


def step_run(w, k, dirty=False, tmp=None):
    C, U, V, E, cell, ring = build(w, k, False)
    forms = [("flat", C.ops, C.num_wires, None)]
    if tmp is not None:
        H, *_ = build(w, k, True)
        forms.append(("post", *post_ops(H, U + V + E + cell, tmp)))
    free = U[1:] + V + E
    N = len(free)
    full = (1 << (1 << N)) - 1
    init = {x: lane_mask(i, N) for i, x in enumerate(free)}
    init[U[0]] = full
    if dirty:
        init[ring[-1]] = full
    io_wires = U + V + E + cell
    res = {}
    for name, ops, nw, log in forms:
        st = sim.run(ops, nw, init, 1 << N)
        rest = [x for x in range(nw) if x not in set(io_wires)]
        res[name] = ([st[x] for x in io_wires], not any(st[x] for x in rest), log)
    return res, C.ops[0::4].count(2), N, tuple(C.ops)


ok, everywhere, rows, cancelled = True, True, [], 0
with tempfile.TemporaryDirectory() as tmp:
    for w in range(WMIN, WMAX + 1):
        res, ref_ccx, N, _ = step_run(w, w - 2)
        ref, ref_clean, _ = res["flat"]
        ok &= ref_clean
        bad, wrong, circuits, ccxs = {}, {}, set(), set()
        ks = list(range(w - 2, 2 * w + 1))
        for k in ks:
            res, ccx, _, ops = step_run(w, k, DIRTY, tmp)
            circuits.add(ops)
            ccxs.add(ccx)
            for form, (out, clean, log) in res.items():
                good = out == ref and clean
                ok &= good
                if not good:
                    bad.setdefault(form, []).append(k)
                if out != ref:
                    wrong.setdefault(form, []).append(k)
                if log:
                    cancelled += log["records_cancelled"]
        everywhere &= all(bad.get(form) == ks for form in ("flat", "post"))
        row = {"w": w, "lanes": 1 << N, "k": [ks[0], ks[-1]], "sizes": len(ks), "distinct_circuits": len(circuits),
               "ccx": sorted(ccxs), "failing_k": bad, "wrong_on_outputs": wrong}
        print(json.dumps(row), flush=True)
        rows.append(row)
summary = {"summary": True, "control": DIRTY, "u6_ring": point_add.U6_RING, "widths": len(rows),
           "circuits": sum(r["sizes"] for r in rows), "forms": ["flat", "post"], "exact": ok,
           "control_fails_everywhere": everywhere if DIRTY else None,
           "post_pass_records_cancelled": cancelled}
if args.out:
    args.out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1) + "\n")
print(json.dumps(summary))
sys.exit(0 if (everywhere if DIRTY else ok) else 1)

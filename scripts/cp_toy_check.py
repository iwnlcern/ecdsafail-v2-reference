"""Exhaustive check of scripts/cp_search.py on toy point adds: as emitted, wrapped, and searched.

    python3 -I scripts/cp_toy_check.py [--bits 4 6 7 8 9 10] [--per-size 2] [--out FILE]

For each toy curve and addend, as scripts/check.py enumerates them, it emits the hierarchical JSON (depth pass on, as
scripts/emit.py writes it), wraps it as the board guide does, and runs scripts/cp_search.py's search on the definitions
its step searches by default, at 40,000 evaluations. Each form (emitted, wrapped, searched) is expanded
by scripts/check.py's rules and checked on every lane (c, R): every R on the curve, c in {0, 1}, the accumulator
R + c*A, the control unchanged and every other wire 0. At the default sizes that is q = 13, 43, 61, 67, 79, 163, 211,
349, 397, 547 and 577, two addends each: 22 instances.

Control, which must fail: in one searched definition the search changed, swap one adjacent pair of gates that do
not commute (share a wire in different roles), CCX pairs first, up to 64 tries an instance, until the lanes catch
one. Prints one JSON line per instance and a summary; exits 0 when every form is exact and every control is caught."""
import argparse
import io
import json
import os
import runpy
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 7, 8, 9, 10])
ap.add_argument("--per-size", type=int, default=2)
ap.add_argument("--out")
args = ap.parse_args()
TREE = os.path.dirname(HERE)
sys.path.insert(0, TREE)
sys.path.insert(0, HERE)
import cp_search as cpsearch  # noqa: E402
from qec import flatten  # noqa: E402
from qec.curve import toy_secp_curves  # noqa: E402
from qec.emit import write_circuit  # noqa: E402
from qec.point_add import ControlledPointAdd  # noqa: E402
from cp_cert import wrap, circuit_of  # noqa: E402

CK = runpy.run_path(os.path.join(TREE, "scripts", "check.py"), run_name="check")
TRIES = 64
MAX_EVALS = 40000                       # the scored pipeline passes --max-evals 40000


def lanes_fail(gen, text, cases):
    doc = flatten.load(io.StringIO(json.dumps(circuit_of(json.loads(text)))))
    return len(CK["check"](gen, CK["expand"](doc), cases, doc["num_wires"]))


def searched(text):
    """The step's search on its default definitions; returns (text, changed definitions, evals)."""
    upload = json.loads(text)
    doc = circuit_of(upload)
    changed, evals = [], 0
    for d in cpsearch.flat_defs(doc["definitions"]):
        defn = doc["definitions"][d]
        ops = defn["operations"]
        n = defn["arity"]
        body = [cpsearch.enc(o) for o in ops]
        new, c0, c1, ev, _ = cpsearch.search(body, n, MAX_EVALS)
        evals += ev
        assert cpsearch.runs(new) == cpsearch.runs(body), f"run sequences changed on definition {d}"
        if new != body:
            defn["operations"] = [cpsearch.dec(t) for t in new]
            changed.append((d, c0, c1))
    return json.dumps(upload, separators=(",", ":")), changed, evals


def control(gen, text, changed, cases):
    doc = circuit_of(json.loads(text))
    cands = []
    for d, _, _ in changed:
        body = [cpsearch.enc(o) for o in doc["definitions"][d]["operations"]]
        for i in range(len(body) - 1):
            if not cpsearch.commutes(body[i], body[i + 1]):
                cands.append((0 if body[i][0] == 3 and body[i + 1][0] == 3 else 1, d, i))
    cands.sort()
    tried = 0
    for _, d, i in cands[:TRIES]:
        bad = circuit_of(json.loads(text))
        ops = bad["definitions"][d]["operations"]
        ops[i], ops[i + 1] = ops[i + 1], ops[i]
        tried += 1
        if lanes_fail(gen, json.dumps(bad), cases):
            return {"caught": True, "tries": tried, "def": d, "pos": i, "candidates": len(cands)}
    return {"caught": False, "tries": tried, "candidates": len(cands)}


rows = []
with tempfile.TemporaryDirectory() as tmp:
    for bits in args.bits:
        for curve in toy_secp_curves(bits)[: args.per_size]:
            pts = curve.points()
            for A in [P for P in pts[1:] if P[1] != 0][:2]:
                t0 = time.time()
                gen = ControlledPointAdd(curve, A)
                C = gen.build(hierarchical=True)
                buf = io.StringIO()
                write_circuit(buf, C, gen.layout)
                emitted = buf.getvalue()
                wrapped = json.dumps(wrap(json.loads(emitted)), separators=(",", ":"))
                srch, changed, evals = searched(wrapped)
                cases = [(c, R) for c in (0, 1) for R in pts]
                forms = {name: lanes_fail(gen, t, cases) for name, t in (("emitted", emitted), ("wrapped", wrapped), ("searched", srch))}
                ctl = control(gen, srch, changed, cases) if changed else {"caught": None, "tries": 0}
                row = {"q": curve.q, "addend": list(A), "n": gen.n, "lanes": len(cases), "failures": forms,
                       "changed_defs": len(changed), "standalone_ticks_saved": sum(c0 - c1 for _, c0, c1 in changed),
                       "evals": evals, "control": ctl, "seconds": round(time.time() - t0, 1)}
                print(json.dumps(row), flush=True)
                rows.append(row)
summary = {"summary": True, "instances": len(rows), "lanes": sum(r["lanes"] for r in rows),
           "all_exact": all(not any(r["failures"].values()) for r in rows),
           "instances_changed": sum(1 for r in rows if r["changed_defs"]),
           "controls_caught": sum(1 for r in rows if r["control"]["caught"]),
           "controls_run": sum(1 for r in rows if r["control"]["caught"] is not None)}
print(json.dumps(summary), flush=True)
if args.out:
    with open(args.out, "w") as fh:
        json.dump({"rows": rows, "summary": summary}, fh, indent=1)
sys.exit(0 if summary["all_exact"] and summary["controls_caught"] == summary["controls_run"] else 1)

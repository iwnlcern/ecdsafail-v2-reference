"""The Euclid unwind as the up-sweep run backwards (`shared_difference_u6`), checked exhaustively, with its reach and controls.

    python3 scripts/u6_check.py [--width 8] [--out FILE]

1. Exhaustive. `shared_difference_u6` against the tree's unwind without a ring (`DialogMultiplier._shared_difference`),
   on the step's domain (U odd, g0 = V[0], g1 = 0, ring clean): every other U and V bit, at widths 2 to --width and
   ring sizes 7 to 25, 33, 49, 97 and 193; every wire, the ring and the carry wire included, must be equal.
2. Reach. Which of the toy row's instances (`scripts/check.py`, as `scripts/accept.py` runs it) call
   `shared_difference_u6`, counted by wrapping it and building each instance.
3. Controls. Two copies of this tree's `qec`, each with one line of `shared_difference_u6` dropped:
     m1      the g1 half of the middle step, A_(i-1) ^= g1 B_i (the swap branch's carries)
     select  the final U ^= g1 (the swap branch's output)
   Both read g1, which the exhaustive domain holds at 0, so each runs the toy row instead, and must fail on exactly
   the instances that reach U6.
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
RINGS = list(range(7, 26)) + [33, 49, 97, 193]
TOY_ARGS = ["--toy-bits", "4", "6", "7", "8", "9", "10", "--toy-per-size", "6"]  # scripts/accept.py's TOY_ARGS
CONTROLS = {"m1": "        C.ccx(G1[i % r], V[i], k)\n", "select": "        C.cx(G1[i % r], U[i])\n"}


def exhaustive(width):
    sys.path.insert(0, str(ROOT))
    from qec import builder
    import qec.point_add as pa
    assert Path(pa.__file__).resolve().is_relative_to(ROOT) and pa.U6_RING == 193

    class Fake:
        pass

    def run(build, w, n):
        C = builder.Circuit()
        U, V, (g0, g1) = C.alloc(w), C.alloc(w), C.alloc(2)
        ring = C.alloc(n)
        build(C, U, V, g0, g1, ring)
        free = U[1:] + V
        N = 1 << len(free)
        lanes = (1 << N) - 1
        st = [0] * C.num_wires
        for bit, wire in enumerate(free):
            st[wire] = sum(1 << L for L in range(N) if L >> bit & 1)
        st[U[0]] = lanes
        st[g0] = st[V[0]]
        ccx = 0
        for j in range(0, len(C.prims), 4):
            op, x, y, z = C.prims[j:j + 4]
            if op == builder.X:
                st[x] = ~st[x]
            elif op == builder.CX:
                st[y] ^= st[x]
            elif op == builder.CCX:
                st[z] ^= st[x] & st[y]
                ccx += 1
            else:
                raise ValueError(op)
        return [v & lanes for v in st], ccx

    def tree_unwind(C, U, V, g0, g1, ring):
        f = Fake()
        f.C = C
        pa.DialogMultiplier._shared_difference(f, U, V, g0, g1)

    rows = []
    for w in range(2, width + 1):
        for n in RINGS:
            ref, rc = run(tree_unwind, w, n)
            got, gc = run(pa.shared_difference_u6, w, n)
            rows.append({"w": w, "ring": n, "equal": got == ref, "ccx": gc, "unwind ccx": rc})
    return rows


def reach():
    sys.path.insert(0, str(ROOT))
    import qec.point_add as pa
    from qec.curve import toy_secp_curves
    calls = []
    inner = pa.shared_difference_u6

    def counted(C, U, V, g0, g1, ring):
        calls.append((len(U), len(ring)))
        return inner(C, U, V, g0, g1, ring)

    pa.shared_difference_u6 = counted
    rows = []
    try:
        for bits in (4, 6, 7, 8, 9, 10):
            for curve in toy_secp_curves(bits)[:6]:
                pts = curve.points()
                for A in [P for P in pts[1:] if P[1] != 0][:2]:
                    calls.clear()
                    pa.ControlledPointAdd(curve, A).build()
                    rows.append({"q": curve.q, "addend": list(A), "calls": len(calls),
                                 "widths": sorted({w for w, _ in calls}), "rings": sorted({n for _, n in calls})})
    finally:
        pa.shared_difference_u6 = inner
    return rows


def control(name, line, td):
    dst = Path(td) / name
    shutil.copytree(ROOT / "qec", dst / "qec", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "scripts", dst / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    p = dst / "qec" / "point_add.py"
    s = p.read_text()
    if s.count(line) != 1:
        return {"error": f"anchor matches {s.count(line)} times"}
    p.write_text(s.replace(line, ""))
    out = Path(td) / f"{name}.json"
    r = subprocess.run([sys.executable, "-I", "-B", str(dst / "scripts" / "check.py"), *TOY_ARGS, "--out", str(out)],
                       cwd=dst, capture_output=True, text=True)
    rows = json.loads(out.read_text()) if out.exists() else []
    return {"exit": r.returncode, "instances": len(rows), "failing": sorted((x["q"], tuple(x["addend"])) for x in rows if x["failures"]),
            "failing lanes": sum(x["failures"] for x in rows), "lanes": sum(x["lanes"] for x in rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    ex = exhaustive(args.width)
    rch = reach()
    reached = sorted((r["q"], tuple(r["addend"])) for r in rch if r["calls"])
    problems = [f"w {r['w']} ring {r['ring']} differs" for r in ex if not r["equal"]]
    if not reached:
        problems.append("no toy instance reaches U6")
    controls = {}
    with tempfile.TemporaryDirectory() as td:
        for name, line in CONTROLS.items():
            c = controls[name] = control(name, line, td)
            if "error" in c:
                problems.append(f"control {name}: {c['error']}")
            elif not c["exit"] or c["failing"] != reached:
                problems.append(f"control {name}: exit {c['exit']}, fails {len(c['failing'])} instances, "
                                f"{len(reached)} reach U6")
    for c in controls.values():
        if "failing" in c:
            c["failing"] = len(c["failing"])
    summary = {"summary": True, "width": args.width, "exhaustive rows": len(ex), "exhaustive equal": all(r["equal"] for r in ex),
               "toy instances": len(rch), "reach U6": len(reached),
               "reach widths": sorted({w for r in rch for w in r["widths"]}), "reach rings": sorted({n for r in rch for n in r["rings"]}),
               "controls": controls, "problems": problems, "ok": not problems}
    if args.out:
        args.out.write_text(json.dumps({**summary, "exhaustive": ex, "reach": rch}, indent=1) + "\n")
    print(json.dumps(summary))
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()

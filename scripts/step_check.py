"""Exhaustive check of the Euclid step against the base's, with controls.

    python3 scripts/step_check.py [--out FILE] [--wmax 7]

The base is commit 171738f (abdowns's `lowt-junk`), whose `_euclid_step` this branch rewrites: the parallel strip
rotation and the unwind identities. Its `qec` package is read from this repository's history with `git archive`, so
the check needs a clone that holds that commit, and fails without one.

Each tree records `_euclid_step(w, top_clear, narrow, last, kept, keep_two[, k])` on a fresh Circuit in its own
process, since both name their package `qec`. This tree's step gets k clean copy wires when it takes them, with k at
the pass's own choice, (w - 1) // 2 - 1, and at 1, which shares the copies. Every input with a clean cell, clean junk
and clean copies runs as bit-sliced lanes: all U, V in [0, 2^w) and E in {0, 1}, on the lanes the step's modes
promise. Under top_clear, V's top wire is 0 on entry. Under narrow, the base's general step (both modes off) leaves
U's top wire 0. The outputs (U, V, E, cell, junk) must agree lane for lane, every other wire must end zero in both
trees, and the CCX counts must be equal. On those lanes the base's moded step must also equal its general step,
which checks the domain itself. Variants: w = 2 to wmax, top_clear and narrow each on and off (top_clear only for
w >= 3, as the pass sets it), last on and off, kept in {0, 1, half, full} up to the step's carries, keep_two on and
off.

Controls. Each copies this tree's `qec`, makes one wrong edit, and must fail the same check. A control whose anchor
is absent from this tree is not applied; an anchor that matches more than once is an error.
  sel-top    drop the top bit's g1 CX that the unwind moves to its end
  sel-kept   run the early selects over the kept bits too
  rot-short  rotate one wire fewer in the parallel strip
  rot-kept   give the kept bits the in-place bits' c ^= a in the unwind's tail
The last line printed is a JSON summary.
"""
import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_COMMIT = "171738fdc528276022da469151048843141f07a4"
PA = "qec/point_add.py"

CONTROLS = {
    "sel-top": ("""        if chain < len(V):
            C.cx(g1, V[-1])
""", ""),
    "sel-kept": ("""        for i in reversed(range(kept + 1, chain)):
            C.ccx(g1, V[i], carries[i])""", """        for i in reversed(range(1, chain)):
            C.ccx(g1, V[i], carries[i])"""),
    "rot-short": ("""                    _par_rotate(C, two, V[:w - 2 if top_clear else w - 1], list(cp))""",
                  """                    _par_rotate(C, two, V[:w - 3 if top_clear else w - 2], list(cp))"""),
    "rot-kept": ("""            if i == 0 or i > kept:
                C.cx(U[i], carries[i])""", """            C.cx(U[i], carries[i])"""),
}

HELPER = r'''
import inspect, json, sys
from pathlib import Path
tree = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(tree))
import qec.builder as qb
import qec.point_add as pa
assert Path(pa.__file__).resolve().is_relative_to(tree), pa.__file__
out = []
for spec in json.loads(sys.argv[2]):
    w, tc, nw, last, kept, k2, k = spec
    C = qb.Circuit()
    dm = pa.DialogMultiplier.__new__(pa.DialogMultiplier)
    dm.C, dm._euclid_steps = C, {}
    takes_k = "k" in inspect.signature(pa.DialogMultiplier._euclid_step).parameters
    blk = dm._euclid_step(w, tc, nw, last, kept, k2, k) if takes_k else dm._euclid_step(w, tc, nw, last, kept, k2)
    codes = {name: getattr(qb, name) for name in ("X", "CX", "CCX", "SWAP")}
    out.append({"ops": list(blk.ops), "flat_in": list(blk.flat_in), "flat_out": list(blk.flat_out), "codes": codes,
                "ccx": list(blk.ops[0::4]).count(codes["CCX"]), "takes_k": takes_k, "calls": len(blk.calls)})
print(json.dumps(out))
'''


def record(tree, specs):
    out = subprocess.run([sys.executable, "-I", "-B", "-c", HELPER, str(tree), json.dumps(specs)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def bitmask(k, lanes):
    block = 1 << k
    pattern = ((1 << block) - 1) << block
    m = 0
    for r in range(lanes // (2 * block)):
        m |= pattern << (r * 2 * block)
    return m


def simulate(blk, w, n_out, mask=None):
    """Inputs U (w), V (w) and E (1) take lane bits; cell, junk and copies start clean. With mask, only those
    lanes are compared and checked for dirty wires."""
    n_in = 2 * w + 1
    lanes = 1 << n_in
    full = (1 << lanes) - 1
    val = {}
    for k, wire in enumerate(blk["flat_in"][:n_in]):
        val[wire] = bitmask(k, lanes)
    c = blk["codes"]
    ops = blk["ops"]
    assert blk["calls"] == 0
    for j in range(0, len(ops), 4):
        op, a, b, t = ops[j:j + 4]
        if op == c["X"]:
            val[a] = val.get(a, 0) ^ full
        elif op == c["CX"]:
            val[b] = val.get(b, 0) ^ val.get(a, 0)
        elif op == c["CCX"]:
            val[t] = val.get(t, 0) ^ (val.get(a, 0) & val.get(b, 0))
        elif op == c["SWAP"]:
            val[a], val[b] = val.get(b, 0), val.get(a, 0)
        else:
            raise ValueError(op)
    m = full if mask is None else mask
    outs = [val.get(wire, 0) & m for wire in blk["flat_out"][:n_out]]
    live = set(blk["flat_out"][:n_out])
    dirty = sorted(wire for wire, v in val.items() if wire not in live and v & m)
    return outs, dirty


def domain(spec, general, w):
    """Lanes the step's modes promise, or None when neither mode is set."""
    _, tc, nw = spec[:3]
    if not (tc or nw):
        return None
    lanes = 1 << (2 * w + 1)
    m = (1 << lanes) - 1
    if tc:
        m &= ~bitmask(2 * w - 1, lanes)
    if nw:
        outs, _ = simulate(general, w, w)
        m &= ~outs[w - 1]
    return m & ((1 << lanes) - 1)


def carries(w, tc, nw):
    chain = w - 1 if w >= 2 and (tc or nw) else w
    return max(0, chain - 1)


def variants(wmax):
    specs = []
    for w in range(2, wmax + 1):
        for last in (False, True):
            for tc in (False, True):
                if tc and w < 3:
                    continue
                for nw in (False, True):
                    if last and (tc or nw):
                        continue
                    full = carries(w, tc, nw)
                    for kept in sorted({0, min(1, full), full // 2, full}):
                        for k2 in (False, True):
                            if last and (kept or k2):
                                continue
                            pick = 0 if last else max(0, (w - 1) // 2 - 1)
                            for k in sorted({pick, min(1, pick)}):
                                specs.append([w, tc, nw, last, kept, k2, k])
    return specs


def compare(base, tree, specs):
    refs, news = record(base, [s[:6] + [0] for s in specs]), record(tree, specs)
    gens = record(base, [[s[0], False, False, s[3], s[4], s[5], 0] for s in specs])
    rows = []
    for s, r, n, g in zip(specs, refs, news, gens):
        w, tc, nw, last, kept, k2, k = s
        n_out = 2 * w + 3 + kept + k2
        mask = domain(s, g, w)
        ro, rd = simulate(r, w, n_out, mask)
        no, nd = simulate(n, w, n_out, mask)
        go, gd = simulate(g, w, n_out, mask)
        row = {"w": w, "top_clear": tc, "narrow": nw, "last": last, "kept": kept, "keep_two": k2, "k": k if n["takes_k"] else 0,
               "lanes": 1 << (2 * w + 1), "domain_lanes": (1 << (2 * w + 1)) if mask is None else bin(mask).count("1"),
               "ccx_base": r["ccx"], "ccx_tree": n["ccx"], "outputs_equal": ro == no, "base_equals_general": ro == go,
               "dirty_base": len(rd), "dirty_tree": len(nd), "dirty_general": len(gd)}
        row["ok"] = ro == no and ro == go and not rd and not nd and not gd and r["ccx"] == n["ccx"]
        rows.append(row)
    return rows


def base_tree(tmp):
    data = subprocess.run(["git", "archive", BASE_COMMIT, "qec"], cwd=ROOT, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(tmp, filter="data")
    return tmp


def control_tree(name, tmp):
    old, new = CONTROLS[name]
    text = (ROOT / PA).read_text()
    n = text.count(old)
    if n > 1:
        raise SystemExit(f"control {name}: anchor matches {n} times")
    if n == 0:
        return None
    shutil.copytree(ROOT / "qec", tmp / "qec", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp / PA).write_text(text.replace(old, new))
    return tmp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--wmax", type=int, default=7)
    args = ap.parse_args()
    specs = variants(args.wmax)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base = base_tree(td / "base")
        rows = compare(base, ROOT, specs)
        failing = [r for r in rows if not r["ok"]]
        for r in failing:
            print("FAIL", json.dumps(r), flush=True)
        controls = {}
        for name in CONTROLS:
            tree = control_tree(name, td / f"control-{name}")
            if tree is None:
                controls[name] = {"applied": False}
                continue
            bad = sum(not r["ok"] for r in compare(base, tree, specs))
            controls[name] = {"applied": True, "failing": bad, "rejected": bad > 0}
            print(f"control {name}: {bad} of {len(specs)} variants fail", flush=True)
    ok = not failing
    applied = [n for n, c in controls.items() if c["applied"]]
    summary = {"base": BASE_COMMIT, "variants": len(rows), "failing": len(failing), "all_ok": ok, "controls": controls,
               "controls_applied": len(applied), "controls_rejected": sum(controls[n]["rejected"] for n in applied)}
    if args.out:
        args.out.write_text(json.dumps({**summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary))
    sys.exit(0 if ok and all(controls[n]["rejected"] for n in applied) else 1)


if __name__ == "__main__":
    main()

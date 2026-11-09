"""Exactness of the Bezout tick's lookahead add where its chain has blocks of 3 and reuses temps, with controls.

    python3 scripts/add_check.py [--out FILE]

The toy rows' primes sit just above 2^(bits-1) and leave the tick's merged path from 14 bits, so below that the
add's chain (add_or_shift_ours in qec/two_bit.py, built on arith.lookahead_chain) never has a block of 3, and with
19 temps it never takes a temp twice. At 256 bits it has both. This row runs scripts/check.py's own exhaustive check
(ControlledPointAdd, build_ops, check: every point, both control values) at the largest prime under 2^bits with the
merged path and a prime-order y^2 = x^3 + 7, for bits 14 to 17, and logs each chain's blocks of 3 and temp uses:
  as built       the tree's own code, 19 temps: blocks of 3 at 16 and 17 bits; 0 failures required;
  temps = 2      the tree's two_bit.py with the add's 19 temps made 2, so its temps wrap from 15 bits: 0 failures;
and three controls on the temps = 2 build, each compiled from this tree's source text with one fault:
  blk3           lookahead_chain's serial CCX in a block of 3 reads T_2 in place of T_3: must fail where a block of 3
                 exists and nowhere else;
  wrap           lookahead_chain clears the chain's first temp at the chain's end, not after its block: exact unless
                 a later block takes that temp again while it is set, so it must fail exactly where temps wrap;
  lactl          the add's middle loop drops the CX after each gated CCX: must fail;
and the lactl fault at 19 temps through scripts/check.py's exhaustive toy check (toy bits 4 and 6 to 10, 6 curves a
size), which must fail too, so the toy rows reach the add's middle.
Each variant runs in its own subprocess, with its two_bit.py installed as qec.two_bit before qec.point_add is imported.
One JSON line per variant and prime; the last line is the summary. Exits 0 only when every requirement holds."""
import argparse
import collections
import json
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TWO_BIT, ARITH = ROOT / "qec" / "two_bit.py", ROOT / "qec" / "arith.py"
DISPATCH = ("            return add_or_shift_ours(C, J, a, word, self.n, 19)\n",
            "            return add_or_shift_ours(C, J, a, word, self.n, 2)\n")
LACTL = ("        C.ccx(k, c, u)\n        C.cx(k, u)\n", "        C.ccx(k, c, u)\n")
BLK3 = ("        C.ccx(U[i], T[r], U[i + r])\n", "        C.ccx(U[i], T[2] if r == 3 else T[r], U[i + r])\n")
WRAP = [("    blk = 0\n", "    blk = 0\n    deferred = None\n"),
        ("""        for j in range(r, 1, -1):
            C.ccx(T[j - 1], V[i + j], T[j])
            C.ccx(U[i], T[j - 1], U[i + j - 1])
        i += r
""", """        for j in range(r, 1, -1):
            if j == 2 and deferred is None and T[2] == temps[0]:
                deferred = (T[1], V[i + 2], T[2])
            else:
                C.ccx(T[j - 1], V[i + j], T[j])
            C.ccx(U[i], T[j - 1], U[i + j - 1])
        i += r
    if deferred is not None:
        C.ccx(*deferred)
""")]
# variant: (temps 2?, two_bit fault, arith fault, bits)
VARIANTS = {"as built": (False, None, None, (14, 15, 16, 17)),
            "temps = 2": (True, None, None, (14, 15, 16, 17)),
            "control blk3": (True, None, "blk3", (14, 15, 16)),
            "control wrap": (True, None, "wrap", (14, 15)),
            "control lactl": (True, "lactl", None, (14,)),
            "control lactl, toy check": (False, "lactl", None, "toy")}
TOY_ARGS = ["--toy-bits", "4", "6", "7", "8", "9", "10", "--toy-per-size", "6"]


def edited(path, pairs):
    src = path.read_text()
    for old, new in pairs:
        assert src.count(old) == 1, f"{path.name}: anchor not found exactly once: {old[:60]!r}"
        src = src.replace(old, new)
    return src


def run_variant(name, out):
    """In this process: install the variant's two_bit.py as qec.two_bit, then check the forced primes."""
    two, fault2, fault_a, bits = VARIANTS[name]
    pairs = [DISPATCH] * two + [LACTL] * (fault2 == "lactl")
    import qec
    mod = types.ModuleType("qec.two_bit")
    mod.__file__, mod.__package__ = str(TWO_BIT), "qec"
    sys.modules["qec.two_bit"] = mod
    exec(compile(edited(TWO_BIT, pairs), str(TWO_BIT), "exec"), mod.__dict__)
    qec.two_bit = mod
    import qec.point_add
    assert qec.point_add.TwoBitMod is mod.TwoBitMod
    chain = mod.lookahead_chain
    if fault_a:
        amod = types.ModuleType(f"qec.arith_ctl_{fault_a}")
        amod.__package__ = "qec"
        exec(compile(edited(ARITH, [BLK3] if fault_a == "blk3" else WRAP), f"arith_ctl_{fault_a}.py", "exec"),
             amod.__dict__)
        chain = amod.lookahead_chain
    seen = collections.Counter()

    def logged(C, U, V, temps, sizes):
        n, i, blk, uses, threes = len(U), 0, 0, 0, 0
        while i < n - 1:
            r = min(sizes[min(blk, len(sizes) - 1)], len(temps) + 1, n - 1 - i)
            blk += 1
            threes += r == 3
            uses += max(0, r - 1)
            i += r
        seen["chains"] += 1
        seen["blocks_of_3"] += threes
        seen["wraps"] += uses > len(temps)
        seen["max_temp_uses"] = max(seen["max_temp_uses"], uses)
        seen["temps"] = len(temps)
        return chain(C, U, V, temps, sizes)
    mod.lookahead_chain = logged
    if bits == "toy":
        toy_out = str(out) + ".toy.json"
        sys.argv = [str(ROOT / "scripts" / "check.py"), *TOY_ARGS, "--out", toy_out]
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit:
            pass  # check.py exits 1 on a failing lane; the receipts below say how many
        receipts = json.loads(Path(toy_out).read_text())
        Path(out).write_text(json.dumps([{"bits": "toy", "instances": len(receipts), "lanes": sum(r["lanes"] for r in receipts),
                                          "failures": sum(r["failures"] for r in receipts),
                                          "failing_instances": sum(1 for r in receipts if r["failures"])}]))
        return
    from qec.curve import Curve, is_prime
    ck = runpy.run_path(str(ROOT / "scripts" / "check.py"), run_name="check")
    rows = []
    for b in bits:
        curve = next(Curve(q, 0, 7) for q in range((1 << b) - 1, 1 << (b - 1), -2)
                     if is_prime(q) and mod.TwoBitMod(q).merged and is_prime(len(Curve(q, 0, 7).points())))
        pts = curve.points()
        A = next(P for P in pts[1:] if P[1] != 0)
        seen.clear()
        gen = ck["ControlledPointAdd"](curve, A)
        C, ops, _ = ck["build_ops"](gen, False)
        cases = [(c, R) for c in (0, 1) for R in pts]
        fails = ck["check"](gen, ops, cases)
        rows.append({"bits": b, "q": curve.q, "lanes": len(cases), "failures": len(fails), **dict(seen)})
    Path(out).write_text(json.dumps(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--variant", help=argparse.SUPPRESS)
    ap.add_argument("--rows", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.variant:
        run_variant(args.variant, args.rows)
        return
    import tempfile
    rows, failed, lanes = [], [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for name in VARIANTS:
            out = os.path.join(tmp, "rows.json")
            p = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--variant", name, "--rows", out],
                               cwd=ROOT, capture_output=True, text=True)
            got = json.loads(Path(out).read_text()) if p.returncode == 0 and os.path.exists(out) else None
            if got is None:
                row = {"row": name, "exit": p.returncode, "stderr_tail": p.stderr[-2000:], "ok": False}
                rows.append(row)
                failed.append(name)
                print(json.dumps(row), flush=True)
                continue
            for r in got:
                if name == "as built":
                    ok = r["failures"] == 0 and r["wraps"] == 0 and (r["blocks_of_3"] > 0) == (r["bits"] >= 16)
                elif name == "temps = 2":
                    ok = r["failures"] == 0 and (r["wraps"] > 0) == (r["bits"] >= 15) and r["temps"] == 2 + 2  # carry, t
                elif name == "control blk3":
                    ok = (r["failures"] > 0) == (r["blocks_of_3"] > 0)
                elif name == "control wrap":
                    ok = (r["failures"] > 0) == (r["wraps"] > 0)
                else:
                    ok = r["failures"] > 0
                if name in ("as built", "temps = 2"):
                    lanes += r["lanes"]
                row = {"row": name, **r, "ok": ok}
                rows.append(row)
                if not ok:
                    failed.append(f"{name} at {r['bits']} bits")
                print(json.dumps(row), flush=True)
    controls = [r for r in rows if r["row"].startswith("control") and r.get("failures")]
    summary = {"summary": True, "rows": len(rows), "lanes": lanes, "failed": failed,
               "control_rows_failing": len(controls),
               "blocks_of_3_reached": any(r.get("blocks_of_3") for r in rows if r["row"] == "as built"),
               "wrap_reached": any(r.get("wraps") for r in rows if r["row"] == "temps = 2")}
    print(json.dumps(summary), flush=True)
    if args.out:
        args.out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

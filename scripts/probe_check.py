"""Exactness of the Bezout tick's probe (chain 1) as a carry-out tree, with controls that must fail.

    python3 scripts/probe_check.py [--out FILE]

1. The probe alone, exhaustively, at the 7 secp-shaped primes q = 2^n - (2^k + d), n = 12 to 18, that have the
   merged path, a quiet run of at least 2 bits and at least 2 H bits. For each prime it builds the probe's context as
   TwoBitMod.tick does: P over H (P_SIZES_221 at 221 H bits, sizes_for otherwise) and M over the quiet run (released),
   both taking their wires from the tick's idle allocator as the tick does, three = k0*k1, the four selectors and
   masks, and take()/drop() that borrow H's wires with an X, the one that comes free first. Built alone, the ops the
   tick runs before the probe are absent, so the idle choices are not the tick's own; the borrow argument does not
   depend on which H wire is borrowed, and part 2 runs the real tick. Every lane is simulated: every window value,
   every H value, and every value of one, k0 and k1. A variant passes when e = AND(H) * carry out of (L + the XOR of
   the active selectors' masks) on every lane, every other wire ends as it began, and every wire take() borrows reads
   0 right after its X wherever AND(H) = 1, the only lanes on which e can be 1.
   Variants: `ripple`, the chain the probe replaced (_chain_up, the CCX into e, _chain_down), which still serves the
   tick's second chain; `tree`, TwoBitMod._probe_tree; and two controls that must fail: `ctl-nox` (take() borrows
   without its X, so a borrowed wire reads 1 where it matters) and `ctl-merge` (the tree's P merge P = Pr*Pl
   becomes P = Pr, compiled from this tree's qec/two_bit.py text).
2. The whole point add with `ctl-merge` installed as qec.two_bit, through scripts/check.py's exhaustive toy check
   (toy bits 4 and 6 to 10, 6 curves a size): it must fail, so the toy rows reach the tree.
One JSON line per row; the last line is the summary. Exits 0 only when every check holds and every control fails."""
import argparse
import json
import os
import runpy
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TWO_BIT = ROOT / "qec" / "two_bit.py"
MERGE = ("                rec.ccx(pr, pl, p)\n", "                rec.cx(pr, p)\n")
TOY_ARGS = ["--toy-bits", "4", "6", "7", "8", "9", "10", "--toy-per-size", "6"]


def merge_source():
    src = TWO_BIT.read_text()
    assert src.count(MERGE[0]) == 1, "the tree's P merge line is not in qec/two_bit.py exactly once"
    return src.replace(*MERGE)


def toy_control(out):
    """Run scripts/check.py's toy check with ctl-merge installed as qec.two_bit; its receipts go to out."""
    import qec
    mod = types.ModuleType("qec.two_bit")
    mod.__file__, mod.__package__ = str(TWO_BIT), "qec"
    sys.modules["qec.two_bit"] = mod
    exec(compile(merge_source(), str(TWO_BIT), "exec"), mod.__dict__)
    qec.two_bit = mod
    import qec.point_add
    assert qec.point_add.TwoBitMod is mod.TwoBitMod
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.argv = [str(ROOT / "scripts" / "check.py"), *TOY_ARGS, "--out", out]
    runpy.run_path(sys.argv[0], run_name="__main__")  # check.py exits 1 on any failing lane


def is_prime(m):
    return m > 1 and all(m % d for d in range(2, int(m ** 0.5) + 1))


def secp_shaped(tb):
    found = []
    for n in range(12, 19):
        for k in range(6, n - 4):
            for d in range(1, 1 << 5, 2):
                q = (1 << n) - ((1 << k) + d)
                if not is_prime(q):
                    continue
                m = tb.TwoBitMod(q)
                a, b = m.run
                if m.merged and b - a >= 2 and n - m.window >= 2:
                    found.append(q)
                    break
            else:
                continue
            break
    return found


def pattern(t, total_bits):
    """Lane mask with lane j set iff bit t of j is set, over 2^total_bits lanes."""
    N = 1 << total_bits
    block = ((1 << (1 << t)) - 1) << (1 << t)
    return ((1 << N) - 1) // ((1 << (1 << (t + 1))) - 1) * block


def build(b, tb, merge_mutant, q, variant):
    mod = tb.TwoBitMod(q)
    n, w, f = mod.n, mod.window, mod.f
    C = b.Circuit(False)
    low = C.alloc(n)
    one, k0, k1 = C.alloc(3)
    L, H = low[:w], low[w:]
    ra, rb = mod.run
    idle = tb._Idle(C, 0) if tb.IDLE_WIRES else None
    if idle:
        C.alloc = idle.alloc
    if tb.PARALLEL_PREFIX and len(H) > 1:
        P = tb.PrefixR(C, H, sizes=list(tb.P_SIZES_221) if len(H) == 221 else tb.sizes_for(len(H)))
    else:
        P = tb.Prefix(C, H, cap=37)
    M = tb.Prefix(C, L[ra:rb])
    if idle:
        del C.alloc
    M.release()
    spare = H[:] if len(H) > 1 else []
    borrowed = []

    def take():
        if not spare:
            (c,) = C.alloc(1)
            return c
        c = spare.pop(idle.first(spare) if idle else -1)
        if variant != "ctl-nox":
            C.x(c)
        borrowed.append((c, len(C.prims) // 4))
        return c

    def drop(c):
        if c in H:
            if variant != "ctl-nox":
                C.x(c)
            spare.append(c)
        else:
            C.free([c])

    (three,) = C.alloc(1)
    C.ccx(k0, k1, three)
    selectors = [one, k0, k1, three]
    masks = [f, f ^ 2 * f, f ^ 3 * f, mod.corner ^ 4 * f]
    start = len(C.prims) // 4
    (e,) = C.alloc(1)
    if variant == "ripple":
        chain = mod._chain_up(C, L, selectors, masks, M, {}, take)
        if H:
            C.ccx(P.top, chain[w], e)
        else:
            C.cx(chain[w], e)
        mod._chain_down(C, L, selectors, masks, M, {}, chain, drop, False)
    else:
        tree = merge_mutant if variant == "ctl-merge" else tb.TwoBitMod._probe_tree
        tree(mod, C, L, selectors, masks, M, P.top if H else None, e, take, drop, bool(H))
    end = len(C.prims) // 4
    return dict(C=C, low=low, H=H, one=one, k0=k0, k1=k1, e=e, start=start, end=end, borrowed=borrowed,
                masks=masks, n=n, w=w)


def simulate(b, x):
    C, n, w = x["C"], x["n"], x["w"]
    total = n + 3
    N = 1 << total
    full = (1 << N) - 1
    val = [0] * C.num_wires
    for i, wire in enumerate(x["low"]):
        val[wire] = pattern(i, total)
    for i, wire in enumerate((x["one"], x["k0"], x["k1"])):
        val[wire] = pattern(n + i, total)
    andh = full
    for wire in x["H"]:
        andh &= val[wire]
    prims = C.prims
    snap, bad_borrow, when = None, 0, {}
    for wire, pos in x["borrowed"]:
        when.setdefault(pos, []).append(wire)
    for k in range(len(prims) // 4):
        if k == x["start"]:
            snap = list(val)
        for wire in when.get(k, ()):
            bad_borrow += bin(val[wire] & andh).count("1")
        code, p, r, s = prims[4 * k:4 * k + 4]
        if code == b.X:
            val[p] ^= full
        elif code == b.CX:
            val[r] ^= val[p]
        elif code == b.CCX:
            val[s] ^= val[p] & val[r]
        elif code == b.SWAP:
            val[p], val[r] = val[r], val[p]
        else:
            raise ValueError(code)
    for wire in when.get(x["end"], ()):
        bad_borrow += bin(val[wire] & andh).count("1")
    exp = 0
    for sel in range(8):
        one_, k0_, k1_ = sel & 1, sel >> 1 & 1, sel >> 2 & 1
        c = 0
        for on, mask in zip((one_, k0_, k1_, k0_ & k1_), x["masks"]):
            if on:
                c ^= mask
        for lane_low in range(1 << n):
            if ((lane_low & ((1 << w) - 1)) + c) >> w & 1:
                exp |= 1 << (sel << n | lane_low)
    if x["H"]:
        exp &= andh
    e_bad = bin(val[x["e"]] ^ exp).count("1")
    others = sum(bin(val[i] ^ snap[i]).count("1") for i in range(C.num_wires) if i != x["e"])
    return {"lanes": N, "e_wrong": e_bad, "other_wire_bits_changed": others,
            "borrow_reads_1_where_it_matters": bad_borrow, "borrowed": len(x["borrowed"]),
            "ccx": sum(1 for k in range(x["start"], x["end"]) if prims[4 * k] == b.CCX)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--toy-control", metavar="FILE", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.toy_control:
        toy_control(args.toy_control)
        return
    import qec.builder as b
    import qec.two_bit as tb
    assert Path(tb.__file__).resolve() == TWO_BIT
    mut = types.ModuleType("qec.two_bit_ctl_merge")
    mut.__package__ = "qec"
    exec(compile(merge_source(), "two_bit_ctl_merge.py", "exec"), mut.__dict__)
    rows, failed = [], []
    lanes = 0
    for q in secp_shaped(tb):
        m = tb.TwoBitMod(q)
        row = {"row": "probe", "q": q, "f": hex((1 << q.bit_length()) - q), "window": m.window, "run": list(m.run),
               "h": q.bit_length() - m.window}
        for variant in ("ripple", "tree", "ctl-nox", "ctl-merge"):
            r = simulate(b, build(b, tb, mut.TwoBitMod._probe_tree, q, variant))
            r["pass"] = r["e_wrong"] == 0 and r["other_wire_bits_changed"] == 0 and r["borrow_reads_1_where_it_matters"] == 0
            row[variant] = r
        lanes += row["tree"]["lanes"]
        row["ok"] = row["ripple"]["pass"] and row["tree"]["pass"] and not row["ctl-nox"]["pass"] and not row["ctl-merge"]["pass"]
        rows.append(row)
        if not row["ok"]:
            failed.append(q)
        print(json.dumps(row), flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "toy.json")
        p = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--toy-control", out], cwd=ROOT,
                           capture_output=True, text=True)
        receipts = json.loads(Path(out).read_text()) if os.path.exists(out) else []
    toy = {"row": "control: toy check with ctl-merge", "exit": p.returncode, "instances": len(receipts),
           "lanes": sum(r["lanes"] for r in receipts), "failing_instances": sum(1 for r in receipts if r["failures"]),
           "failures": sum(r["failures"] for r in receipts)}
    toy["ok"] = p.returncode == 1 and toy["failing_instances"] > 0
    if not toy["ok"]:
        failed.append(toy["row"])
        toy["stderr_tail"] = p.stderr[-2000:]
    rows.append(toy)
    print(json.dumps(toy), flush=True)
    summary = {"summary": True, "primes": len(rows) - 1, "lanes_per_variant": lanes, "failed": failed,
               "toy_control_instances": toy["instances"], "toy_control_failing_instances": toy["failing_instances"]}
    print(json.dumps(summary), flush=True)
    if args.out:
        args.out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

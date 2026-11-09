"""Exactness of the Euclid difference's tree sweep at every width the 256-bit pass uses, with a control that must fail.

    python3 scripts/sweep_check.py [--out FILE]

The toy and small-prime rows never build a 257-wire ring, and blocks of two need about n - 9 temps, so at 256 bits
the sweep's shapes are reached only by sampled points. This row checks the sweep on bits instead, at every length.
1. lookahead_tree at blocks of 2 and of 3, for every n from 2 to 260, on exactly tree_temps(n, r) temps: on 640 lanes
   (random, plus lanes with every x set, every x clear and alternating p), U[j] must become x_j ^ p_j U[j-1] for
   j >= 1 from U[0] (lookahead_chain's map), V must be unchanged and every temp clean.
   Control: lookahead_tree compiled from this tree's qec/arith.py with each block's fixes U[i+j] ^= T_j U[i] removed.
   It must fail at every n that has a fix to remove (n >= 3).
2. point_add.difference_sweep, the sweep the Euclid difference calls, for every n from 2 to 260, on tree_temps(n, 2)
   temps (it must take blocks of two) and on one temp fewer (it must fall back to lookahead_sweep's choice), with
   the same map, V unchanged and every temp clean.
One JSON line per block size and per sweep; the last line is the summary. Exits 0 only when every check holds and
the control fails wherever it must."""
import argparse
import json
import random
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qec import arith, point_add  # noqa: E402
from qec.builder import CCX, CX, SWAP, X, Circuit  # noqa: E402

N, LANES = 260, 640
MASK = (1 << LANES) - 1
FIX = "            C.ccx(U[i], T[j], U[i + j])\n"


def no_fix_tree():
    src = (ROOT / "qec" / "arith.py").read_text()
    assert src.count(FIX) == 1, "lookahead_tree's fix line is not in qec/arith.py exactly once"
    mod = types.ModuleType("qec.arith_ctl_nofix")
    mod.__package__ = "qec"
    exec(compile(src.replace(FIX, "            pass\n"), "arith_ctl_nofix.py", "exec"), mod.__dict__)
    return mod.lookahead_tree


def failing_lanes(n, build, ntemps, seed):
    """Lanes on which build(C, U, V, temps) misses the map, changes V or leaves a temp dirty; also the block size
    the sweep took, when it is lookahead_tree."""
    C = Circuit()
    U, V = C.alloc(n), C.alloc(n)
    temps = C.alloc(ntemps)
    taken = []
    orig = point_add.lookahead_tree

    def spy(C_, U_, V_, t_, r):
        taken.append(r)
        return orig(C_, U_, V_, t_, r)
    point_add.lookahead_tree = spy
    try:
        build(C, U, [None] + V[1:], temps)  # V[0] is unused, as in the sweep's [None] + V
    finally:
        point_add.lookahead_tree = orig
    rng = random.Random(seed)
    lanes = [rng.getrandbits(LANES) for _ in range(2 * n)]
    for k in range(2 * n):
        lanes[k] = (lanes[k] | 1) & ~2  # lane 0: every x and p set; lane 1: every one clear
        if k >= n:
            lanes[k] = lanes[k] & ~4 | ((k % 2) << 2)  # lane 2: p alternates
    st = [0] * C.num_wires
    for k, w in enumerate(U + V):
        st[w] = lanes[k]
    x, p = [st[w] for w in U], [st[w] for w in V]
    ops = C.ops
    for k in range(0, len(ops), 4):
        o, a, b, c = ops[k:k + 4]
        if o == X:
            st[a] ^= MASK
        elif o == CX:
            st[b] ^= st[a]
        elif o == CCX:
            st[c] ^= st[a] & st[b]
        elif o == SWAP:
            st[a], st[b] = st[b], st[a]
        else:
            raise ValueError(o)
    f = [x[0]]
    for j in range(1, n):
        f.append(x[j] ^ (p[j] & f[j - 1]))
    bad = 0
    for j in range(n):
        bad |= st[U[j]] ^ f[j]
        if j:
            bad |= st[V[j]] ^ p[j]
    for t in temps:
        bad |= st[t]
    return bin(bad).count("1"), taken


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    control = no_fix_tree()
    rows, failed = [], []
    cases = control_fails = control_must = 0
    for r in (2, 3):
        bad_n, ctl_missed = [], []
        for n in range(2, N + 1):
            k = arith.tree_temps(n, r)
            bad, _ = failing_lanes(n, lambda C, U, V, t: arith.lookahead_tree(C, U, V, t, r), k, 1000 * n + r)
            ctl, _ = failing_lanes(n, lambda C, U, V, t: control(C, U, V, t, r), k, 1000 * n + r)
            cases += 1
            bad_n += [n] * bool(bad)
            if n >= 3:
                control_must += 1
                control_fails += ctl > 0
                ctl_missed += [n] * (ctl == 0)
        row = {"row": f"lookahead_tree, blocks of {r}", "n": [2, N], "lanes": LANES, "failing_n": bad_n,
               "control_missed_n": ctl_missed}
        row["ok"] = not bad_n and not ctl_missed
        rows.append(row)
        print(json.dumps(row), flush=True)
    for fewer in (0, 1):
        bad_n, wrong_choice = [], []
        for n in range(2, N + 1):
            k = arith.tree_temps(n, 2) - fewer
            if k < 1:
                continue
            bad, taken = failing_lanes(n, point_add.difference_sweep, k, 7000 * n + fewer)
            cases += 1
            bad_n += [n] * bool(bad)
            if (2 in taken) == bool(fewer):
                wrong_choice.append(n)
        row = {"row": "difference_sweep, " + ("one temp short of blocks of 2" if fewer else "temps for blocks of 2"),
               "n": [2, N], "lanes": LANES, "failing_n": bad_n, "wrong_block_choice_n": wrong_choice}
        row["ok"] = not bad_n and not wrong_choice
        rows.append(row)
        print(json.dumps(row), flush=True)
    failed = [row["row"] for row in rows if not row["ok"]]
    summary = {"summary": True, "cases": cases, "lanes": LANES, "control_fails": control_fails,
               "control_must_fail": control_must, "failed": failed}
    print(json.dumps(summary), flush=True)
    if args.out:
        args.out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

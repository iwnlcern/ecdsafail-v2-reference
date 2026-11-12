"""SQ_PAR check: qec/square.py's SquareFold.compute with SQ_PAR on against the same block with SQ_PAR off.

Simulates both blocks bit-parallel (each wire an int, one bit per input) and checks that J[:n] holds Y^2 mod q,
that every wire of Y and J equals the SQ_PAR-off block's output, so the reverse block sees the same values, and that
every scratch wire ends 0. Toy primes with SquareFold.applies run every n-bit Y; secp256k1 runs edge values and
seeded random ones. --control turns the middle CCX of the SQ_PAR-on block into a CX, and the check must fail.

    python3 -I scripts/square_par_check.py [--seed S] [--random N] [--control]

Prints one JSON line per prime and a summary line; exits 0 only if every prime passes (or, with --control, only
if secp256k1 fails)."""
import argparse
import json
import random
import sys
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.dont_write_bytecode = True

from qec import builder, square  # noqa: E402
from qec.arith import Mod  # noqa: E402
from qec.square import SquareFold  # noqa: E402

SECP = 2 ** 256 - 2 ** 32 - 977
TOYS = [q for q in (13, 29, 61, 113, 127, 241, 251, 499, 509, 1021, 2039, 4093)
        if q.bit_length() >= 4 and SquareFold.applies(q)]


def build(q, par):
    saved = square.SQ_PAR
    square.SQ_PAR = par
    try:
        C = builder.Circuit()
        sq = SquareFold(Mod(q))
        Y, J = C.alloc(sq.n), C.alloc(sq.junk)
        sq.compute(C, Y, J)
    finally:
        square.SQ_PAR = saved
    return C, sq, Y, J


def run(C, Y, ys):
    st = [0] * C.num_wires
    for i, w in enumerate(Y):
        st[w] = sum(((y >> i) & 1) << k for k, y in enumerate(ys))
    ops, full = C.prims, (1 << len(ys)) - 1
    for g in range(0, len(ops), 4):
        cc, a, b, t = ops[g:g + 4]
        if cc == builder.CCX:
            st[t] ^= st[a] & st[b]
        elif cc == builder.CX:
            st[b] ^= st[a]
        elif cc == builder.X:
            st[a] ^= full
        else:
            st[a], st[b] = st[b], st[a]
    return st


def value(st, reg, k):
    return sum(((st[w] >> k) & 1) << i for i, w in enumerate(reg))


def check(q, ys, control):
    C1, sq, Y1, J1 = build(q, True)
    if control:
        at = [g for g in range(0, len(C1.prims), 4) if C1.prims[g] == builder.CCX]
        g = at[len(at) // 2]
        C1.prims[g:g + 4] = array("i", (builder.CX, C1.prims[g + 1], C1.prims[g + 3], C1.prims[g + 3]))
    C0, _, Y0, J0 = build(q, False)
    st1, st0 = run(C1, Y1, ys), run(C0, Y0, ys)
    wrong = sum(1 for k, y in enumerate(ys) if value(st1, J1[:sq.n], k) != y * y % q)
    differ = sum(1 for a, b in zip(Y1 + J1, Y0 + J0) if st1[a] != st0[b])
    io = set(Y1 + J1)
    dirty = sum(1 for w in range(C1.num_wires) if w not in io and st1[w])
    ccx = [sum(1 for c in X.prims[::4] if c == builder.CCX) for X in (C1, C0)]
    return {"q_bits": q.bit_length(), "q": q if q < 1 << 64 else "secp256k1", "inputs": len(ys), "square_wrong": wrong,
            "wires_differing": differ, "scratch_dirty": dirty, "ccx_on": ccx[0], "ccx_off": ccx[1],
            "pass": not (wrong or differ or dirty)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--random", type=int, default=240)
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for q in TOYS:
        rows.append(check(q, list(range(1 << q.bit_length())), a.control))
    edges = [0, 1, 2, SECP - 2, SECP - 1, SECP, SECP + 1, 2 ** 256 - 1, 2 ** 255, 2 ** 128 - 1, 2 ** 128,
             (2 ** 128 - 1) << 128, 2 ** 32 + 977, 2 ** 256 - 2 ** 128]
    rows.append(check(SECP, edges + [rng.getrandbits(256) for _ in range(a.random)], a.control))
    for r in rows:
        print(json.dumps(r))
    passed = sum(r["pass"] for r in rows)
    # A demoted CCX can be equivalent on a small toy, so the control must fail at least at secp256k1.
    ok = not rows[-1]["pass"] if a.control else passed == len(rows)
    print(json.dumps({"primes": len(rows), "passed": passed, "control": a.control, "ok": ok}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Exactness of the square fold's lookahead adds, with controls that must fail.

    python3 scripts/square_la_check.py [--out FILE]

1. add_lookahead on every (a, b) at widths 1 to 7, with no carry-in, a carry-in wire at 0 and at 1, and with and
   without a carry-out wire: b must become (a + b + cin) mod 2^w, cout the carry, and a, cin and the temps restored.
   Control: the carry-in wire is passed to the caller but not to the add, so the sum loses it.
2. X <- (X - g*Y^2) mod q through SquareFold.compute and modadd, the square-subtract the point add runs, by bit-sliced
   simulation, with Y and g restored and every other wire 0: every Y at 2 X at q = 65,521; 4,000 random Y at
   262,121; 3,000 at 2^64 - 59 and 2^128 - 159; 600 plus edge Y at secp256k1; and 50 lanes at g = 0 at each.
3. Controls on the same square-subtract at q = 65,521 (every Y) and 262,121 (4,000 Y): the hi-into-lo multiple's add
   drops its carry-in wire (the fold's + 1 is lost), and the last add drops its carry-out into t. Each must fail.
One JSON line per row; the last line is the summary. Exits 0 only when every check holds and every control fails."""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import qec.square as square  # noqa: E402
from qec.arith import Mod, add_lookahead  # noqa: E402
from qec.builder import CCX, Circuit  # noqa: E402
from qec.curve import SECP256K1  # noqa: E402
from qec.sim import run  # noqa: E402
from qec.square import SquareFold  # noqa: E402

SEED = 5114


def pack(values, wires):
    bits = [format(v, f"0{len(wires)}b")[::-1] for v in values]
    return {w: int("".join(b[i] for b in reversed(bits)), 2) for i, w in enumerate(wires)}


def unpack(state, wires, lanes):
    cols = [format(state[w], f"0{lanes}b")[::-1] for w in wires]
    return [sum(int(c[j]) << i for i, c in enumerate(cols)) for j in range(lanes)]


def add_cases(control=False):
    bad = total = 0
    for w in range(1, 8):
        for mode in ("none", "cin0", "cin1"):
            for with_cout in (False, True):
                C = Circuit()
                a, b, temps = C.alloc(w), C.alloc(w), C.alloc(3)
                cin = C.alloc(1)[0] if mode != "none" else None
                cout = C.alloc(1)[0] if with_cout else None
                add_lookahead(C, a, b, cout, temps, cin=None if control else cin)
                lanes = [(x, y) for x in range(1 << w) for y in range(1 << w)]
                c0 = 1 if mode == "cin1" else 0
                init = pack([x for x, _ in lanes], a)
                init.update(pack([y for _, y in lanes], b))
                if cin is not None:
                    init[cin] = (1 << len(lanes)) - 1 if c0 else 0
                s = run(C.ops, C.num_wires, init, len(lanes))
                oa, ob = unpack(s, a, len(lanes)), unpack(s, b, len(lanes))
                oc = unpack(s, [cout], len(lanes)) if cout is not None else [0] * len(lanes)
                oi = unpack(s, [cin], len(lanes)) if cin is not None else [0] * len(lanes)
                clean = all(s[t] == 0 for t in temps)
                for i, (x, y) in enumerate(lanes):
                    total += 1
                    v = x + y + c0
                    want_c = v >> w if cout is not None else 0
                    bad += not (ob[i] == v % (1 << w) and oa[i] == x and oc[i] == want_c and oi[i] == c0 and clean)
    return {"lanes": total, "bad": bad}


def square_subtract(C, mod, gen, Y, X):
    fold = SquareFold(mod)
    n = mod.n
    sq = C.record_shape(lambda y, j: fold.compute(C, y, j), [n, fold.junk])
    ma = C.record_shape(lambda k, a, b: fold.modadd(C, a, b, k[0]), [1, n, n])
    J = C.alloc(fold.junk)
    C.apply(sq, [Y, J])
    C.apply(ma, [[gen], J[:n], X], inverse=True)
    C.apply(sq, [Y, J], inverse=True)
    C.free(J)


def faulty(kind):
    """square.add_lookahead with one fault: `cin` drops the carry-in of the add that has one (the hi-into-lo
    multiple's low digit); `cout` drops the carry-out of the fold's add that has one (the last add, not the rows')."""
    rows = SquareFold._rows

    def in_rows(self, C, *a, **k):
        state["rows"] += 1
        try:
            return rows(self, C, *a, **k)
        finally:
            state["rows"] -= 1

    state = {"rows": 0}

    def add(C, a, b, cout, temps, cin=None):
        if kind == "cin" and cin is not None:
            cin = None
        if kind == "cout" and cout is not None and not state["rows"]:
            cout = None
        return add_lookahead(C, a, b, cout, temps, cin=cin)
    return add, in_rows


def check(q, lanes, fault=None):
    mod = Mod(q)
    C = Circuit()
    (gen,) = C.alloc(1)
    X, Y = C.alloc(mod.n), C.alloc(mod.n)
    saved = square.add_lookahead, SquareFold._rows
    if fault:
        square.add_lookahead, SquareFold._rows = faulty(fault)
    try:
        square_subtract(C, mod, gen, Y, X)
    finally:
        square.add_lookahead, SquareFold._rows = saved
    init = pack([g for g, _, _ in lanes], [gen])
    init.update(pack([x for _, x, _ in lanes], X))
    init.update(pack([y for _, _, y in lanes], Y))
    s = run(C.ops, C.num_wires, init, len(lanes))
    ox, oy, og = unpack(s, X, len(lanes)), unpack(s, Y, len(lanes)), unpack(s, [gen], len(lanes))
    live = {gen, *X, *Y}
    dirty = sum(1 for w in range(C.num_wires) if w not in live and s[w])
    bad = sum(ox[i] != (x - g * y * y) % q or oy[i] != y or og[i] != g for i, (g, x, y) in enumerate(lanes))
    return {"ccx": C.ops[0::4].count(CCX), "wires": C.num_wires, "lanes": len(lanes), "bad": bad, "dirty": dirty}


def edge_ys(q, n):
    h = n // 2
    lo = (1 << h) - 1
    ys = {0, 1, 2, q - 1, q - 2, lo, lo + 1, lo << h, (lo << h) | lo, (1 << (n - 1)) % q, ((1 << (n - 1)) - 1) % q}
    return sorted(y for y in ys if 0 <= y < q)


def lanes_for(rng, q, ys, xs_per):
    return [(1, rng.randrange(q), y) for y in ys for _ in range(xs_per)] + [(0, rng.randrange(q), y) for y in ys[:50]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows, failed = [], []

    def emit(row, ok):
        row["ok"] = ok
        rows.append(row)
        if not ok:
            failed.append(row["row"])
        print(json.dumps(row), flush=True)

    r = add_cases()
    emit({"row": "add_lookahead", **r}, r["bad"] == 0)
    r = add_cases(control=True)
    emit({"row": "control: add_lookahead without its carry-in", **r}, r["bad"] > 0)
    rng = random.Random(SEED)
    plan = [(65521, list(range(65521)), 2), (262121, [rng.randrange(262121) for _ in range(4000)], 2),
            ((1 << 64) - 59, [rng.randrange((1 << 64) - 59) for _ in range(3000)], 1),
            ((1 << 128) - 159, [rng.randrange((1 << 128) - 159) for _ in range(3000)], 1),
            (SECP256K1.q, [rng.randrange(SECP256K1.q) for _ in range(600)] + edge_ys(SECP256K1.q, 256), 1)]
    fold_lanes = {}
    for q, ys, xs_per in plan:
        r = check(q, lanes_for(rng, q, ys, xs_per))
        fold_lanes[str(q)] = r["lanes"]
        emit({"row": "square-subtract", "q": q, "n": q.bit_length(), **r}, r["bad"] == 0 and r["dirty"] == 0)
    crng = random.Random(SEED + 1)
    controls = 0
    for q, ys, xs_per in ((65521, list(range(65521)), 2), (262121, [crng.randrange(262121) for _ in range(4000)], 2)):
        lanes = lanes_for(crng, q, ys, xs_per)
        for kind in ("cin", "cout"):
            r = check(q, lanes, fault=kind)
            controls += 1
            emit({"row": f"control: square-subtract, {kind} dropped", "q": q, **r}, r["bad"] + r["dirty"] > 0)
    summary = {"summary": True, "rows": len(rows), "failed": failed, "add_lanes": rows[0]["lanes"],
               "fold_lanes": fold_lanes, "controls": controls + 1, "fold_temps": square.FOLD_TEMPS}
    print(json.dumps(summary), flush=True)
    if args.out:
        args.out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

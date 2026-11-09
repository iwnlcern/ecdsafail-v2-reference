"""Exactness of the one-level Karatsuba square-subtract, qec.square.KaratsubaMulsub.

The class applies only at even n >= 128, so no exhaustive gate row reaches it. This
check forces it at small primes and drives its pieces at full width:

- selection: which moduli take the class, and which path ControlledPointAdd._mulsub takes;
- the canonical tail by enumeration. At q = 65,521, 262,121 and 262,139, it simulates every
  Y up to the entry of each group's `canon` and selects every Y whose group value lies in
  [q, 2^n). For each selected Y it then checks every X with gen = 1. Those Y are the only
  inputs on which `canon` changes the value, so the pairs are the whole population on which
  the tail's subtraction fires. The test needs no seed, and dropping `canon` fails it;
- every Y with gen in {0, 1} and a seeded X at the same primes;
- `canon` and `finish` at secp256k1 and 2^128 - 159, on register values next to q and 2^n;
- seeded samples at 2^128 - 159 and secp256k1, with K paid by the class and with K merged
  into a preceding constant add, as the point add does;
- controls that must fail: no canonical tail, no high carry, K off by one, and B's top part
  not folded.

Each lane requires X = (X - gen * Y^2) mod q, Y and gen restored, and every other wire 0.
The simulator, qec.sim, shares no arithmetic with the generator. One JSON line is printed
per row and a summary line last. The exit status is 1 if any row fails.
"""
import argparse
import json
import random
import sys
import time
from array import array
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qec.arith import Mod, add, sub
from qec.builder import CCX, Circuit
from qec.curve import SECP256K1
from qec.point_add import ControlledPointAdd
from qec.sim import run
from qec.square import KaratsubaMulsub, SquareFold
from qec.two_bit import Prefix

TAIL_PRIMES = (65_521, 262_121, 262_139)
CONTROL_PRIME = 65_521
WIDE_PRIMES = ((1 << 128) - 159, SECP256K1.q)
SAMPLE_SEEDS = (2041, 7331, 6062)
SAMPLE_RANDOM_Y = 300
SECP_STANDALONE = {"wires": 1240, "ccx": 124_362}


class Harness(NamedTuple):
    """A standalone square-subtract on registers gen, X and Y, as flat ops."""
    q: int
    ops: array
    num_wires: int
    gen: int
    x: list
    y: list
    offset: int


def build(q, cls=KaratsubaMulsub, merged=False):
    """X -= gen * Y^2 by cls at q. With merged, K is paid by a preceding add of 3 + K, as the
    point add pays it, so the result is X + 3 * gen - gen * Y^2."""
    c, mod = Circuit(), Mod(q)
    (gen,) = c.alloc(1)
    x, y = c.alloc(mod.n), c.alloc(mod.n)
    square = cls(mod)
    if merged:
        mod.cmodadd_const(c, gen, (3 + square.K) % q, x)
    square.mulsub(c, gen, y, x, add_k=not merged)
    return Harness(q, c.ops, c.num_wires, gen, x, y, 3 if merged else 0)


def point_add_mulsub(q):
    """ControlledPointAdd._mulsub on standalone registers, without building a whole point add."""
    gen_obj = object.__new__(ControlledPointAdd)
    c, mod = Circuit(), Mod(q)
    gen_obj.C, gen_obj.mod, gen_obj.n = c, mod, mod.n
    gen_obj._square_blk = gen_obj._modadd_blk = None
    (gen,) = c.alloc(1)
    x, y = c.alloc(mod.n), c.alloc(mod.n)
    gen_obj._mulsub(gen, y, x)
    return gen_obj, Harness(q, c.ops, c.num_wires, gen, x, y, 0)


def planes(values, width):
    """Per-lane integers to per-bit lane masks."""
    data = [bytearray((len(values) + 7) // 8) for _ in range(width)]
    for lane, v in enumerate(values):
        for i in range(width):
            if v >> i & 1:
                data[i][lane // 8] |= 1 << (lane % 8)
    return [int.from_bytes(b, "little") for b in data]


def simulate(h, ops, gen_plane, x_planes, y_planes, expected_planes, lanes):
    """(wrong lanes, dirty lanes) of ops on these input planes against the expected X."""
    state = run(ops, h.num_wires, {h.gen: gen_plane, **dict(zip(h.x, x_planes)), **dict(zip(h.y, y_planes))}, lanes)
    wrong = state[h.gen] ^ gen_plane
    for wire, plane in zip(h.y, y_planes):
        wrong |= state[wire] ^ plane
    for wire, plane in zip(h.x, expected_planes):
        wrong |= state[wire] ^ plane
    dirty = 0
    for wire in set(range(h.num_wires)) - {h.gen, *h.x, *h.y}:
        dirty |= state[wire]
    return wrong.bit_count(), dirty.bit_count()


def prepare(h, lanes):
    """The input and expected planes of (gen, X, Y) lanes, for simulate."""
    q, n = h.q, len(h.x)
    expected = [(x - g * y * y + g * h.offset) % q for g, x, y in lanes]
    return (planes([g for g, _, _ in lanes], 1)[0], planes([x for _, x, _ in lanes], n),
            planes([y for _, _, y in lanes], n), planes(expected, n), len(lanes))


def check_lanes(h, lanes, ops=None):
    """Simulate (gen, X, Y) lanes, or lanes already prepared. ops defaults to the harness's own."""
    prepared = lanes if isinstance(lanes, tuple) else prepare(h, lanes)
    wrong, dirty = simulate(h, h.ops if ops is None else ops, *prepared)
    return {"lanes": prepared[-1], "wrong": wrong, "dirty": dirty}


def all_y_lanes(q, seed=2041):
    """Every Y with gen in {0, 1}, each with a seeded X."""
    rng = random.Random(seed)
    return [(g, rng.randrange(q), y) for y in range(q) for g in (0, 1)]


def tail_population(q):
    """The Y whose AC or B group value lies in [q, 2^n) at the entry of `canon`, and the count per group.

    Each group is recorded on a fresh local Circuit whose first n wires are Y and whose other
    wires start at 0, so the ops up to `canon` simulate on every Y at once."""
    snapshots = []

    class Trace(KaratsubaMulsub):
        def canon(self, C, R):
            snapshots.append((array("i", C.prims), C.num_wires, list(R)))
            super().canon(C, R)

    build(q, Trace)
    assert len(snapshots) == 2, "one canonical tail per group"
    n = q.bit_length()
    mask = (1 << q) - 1
    y_planes = planes(list(range(q)), n)
    hits, counts = 0, []
    for ops, num_wires, register in snapshots:
        state = run(ops, num_wires, dict(enumerate(y_planes)), q)
        equal, greater = mask, 0
        for i in reversed(range(n)):
            bit = state[register[i]]
            if q >> i & 1:
                equal &= bit
            else:
                greater |= equal & bit
                equal &= mask ^ bit
        selected = greater | equal
        counts.append(selected.bit_count())
        hits |= selected
    return [v for v in range(q) if hits >> v & 1], counts


def check_tail(h, ys, ops=None):
    """Every X with gen = 1 for each Y in ys. Lane j holds X = j."""
    q, n = h.q, len(h.x)
    mask = (1 << q) - 1
    width_mask = (1 << (1 << n)) - 1
    # Bit b of the lane index, as a mask over 2^n lanes, cut to q lanes.
    x_planes = [((width_mask // ((1 << (1 << b)) + 1)) << (1 << b)) & mask for b in range(n)]
    wrong = dirty = 0
    for v in ys:
        y_planes = [mask if v >> b & 1 else 0 for b in range(n)]
        shift = (v * v - h.offset) % q
        # Lane X expects (X - shift) mod q: rotate every X bit plane by shift lanes.
        expected = [((p << shift) | (p >> (q - shift))) & mask for p in x_planes]
        w, d = simulate(h, h.ops if ops is None else ops, mask, x_planes, y_planes, expected, q)
        wrong, dirty = wrong + w, dirty + d
    return {"ys": len(ys), "pairs": len(ys) * q, "wrong": wrong, "dirty": dirty}


# Controls. Each breaks one part of the construction and must fail.

class NoCanon(KaratsubaMulsub):
    def canon(self, C, R):
        pass


class NoHighCarry(KaratsubaMulsub):
    @staticmethod
    def add_low(C, a, b):
        (t,) = C.alloc(1)
        add(C, a, b[:len(a)], cout=t)
        return t


class KOff(KaratsubaMulsub):
    def __init__(self, mod):
        super().__init__(mod)
        self.K = (self.K + 1) % self.q


class NoBTop(KaratsubaMulsub):
    def fold_b(self, C, RB):
        n, h = self.n, self.h
        T = C.alloc(n + self.fb + 1)
        for a, b in zip(RB[:h], T[h:]):
            C.cx(a, b)
        (b,) = C.alloc(1)
        sub(C, RB[h:], RB[:h], borrow=b)
        for w in RB[h:]:
            C.x(w)
        Prefix(C, RB[h:]).write(b)
        for w in RB[h:]:
            C.x(w)
        for w in RB:
            C.x(w)
        self.sq._multiple(C, RB, T, (1 << n) - 1, (1 << n) - 1)
        lo = T[:n]
        (c1,) = C.alloc(1)
        self.finish(C, lo, c1)
        return lo


def piece_harness(q, piece, cls=KaratsubaMulsub):
    """Build `canon` or `finish` as a recorded block on (R, c1, c2), copy R out, and unbuild it.

    piece is "canon", "finish" (c1 and c2) or "finish1" (c1 only, as group B calls it)."""
    square = cls(Mod(q))
    n = square.n

    def body(C, w):
        R = w[:n]
        if piece == "canon":
            square.canon(C, R)
        elif piece == "finish":
            square.finish(C, R, w[n], w[n + 1])
        else:
            square.finish(C, R, w[n])
        return R

    c = Circuit()
    regs, out = c.alloc(n + 2), c.alloc(n)
    block, count, result = square._record_build(body, n + 2)
    junk = c.alloc(count)
    c.apply(block, [regs, junk])
    for wire, o in zip([(regs + junk)[i] for i in result], out):
        c.cx(wire, o)
    c.apply(block, [regs, junk], inverse=True)
    c.free(junk)
    return c, regs, out


def check_piece(q, piece, cls=KaratsubaMulsub):
    """Register values next to q and 2^n with every allowed (c1, c2)."""
    n = q.bit_length()
    values = [0, q - 1, q, q + 1, (1 << n) - 1]
    carries = {"canon": [(0, 0)], "finish": [(0, 0), (1, 0), (0, 1)], "finish1": [(0, 0), (1, 0)]}[piece]
    lanes = [(r, c1, c2) for r in values for c1, c2 in carries]
    c, regs, out = piece_harness(q, piece, cls)
    inputs = [r | c1 << n | c2 << (n + 1) for r, c1, c2 in lanes]
    expected = [(r + (c1 << n) + (c2 << (n + 1))) % q for r, c1, c2 in lanes]
    in_planes, out_planes = planes(inputs, n + 2), planes(expected, n)
    state = run(c.ops, c.num_wires, dict(zip(regs, in_planes)), len(lanes))
    wrong = 0
    for wire, plane in zip(regs + out, in_planes + out_planes):
        wrong |= state[wire] ^ plane
    dirty = 0
    for wire in set(range(c.num_wires)) - set(regs + out):
        dirty |= state[wire]
    return {"lanes": len(lanes), "wrong": wrong.bit_count(), "dirty": dirty.bit_count()}


def sample_lanes(q, seed):
    rng = random.Random(seed)
    n = q.bit_length()
    edge = [0, 1, q - 1, q - 2, (1 << (n // 2)) - 1, 1 << (n // 2)]
    return [(g, rng.randrange(q), y) for y in edge + [rng.randrange(q) for _ in range(SAMPLE_RANDOM_Y)] for g in (0, 1)]


class Rows:
    def __init__(self, out):
        self.out, self.rows = out, []

    def add(self, name, ok, **result):
        row = {"row": name, "ok": bool(ok), **result}
        self.rows.append(row)
        print(json.dumps(row), flush=True)
        if self.out:
            self.out.write_text(json.dumps(self.rows, indent=1) + "\n")

    def timed(self, name, fn, ok):
        start = time.time()
        result = fn()
        self.add(name, ok(result), seconds=round(time.time() - start, 1), **result)
        return result


def clean(r):
    return r["wrong"] == 0 and r["dirty"] == 0


def failed(r):
    return r["wrong"] > 0 or r["dirty"] > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="also write the rows here as a JSON list")
    args = ap.parse_args()
    rows, start = Rows(args.out), time.time()

    # Selection. fits is the class's own bound; applies adds the size threshold.
    fits = {q: KaratsubaMulsub.fits(q) for q in (*TAIL_PRIMES, 65_519, (1 << 64) - 59, *WIDE_PRIMES)}
    applies = {q: KaratsubaMulsub.applies(q) for q in fits}
    want_fits = {q: q != 65_519 for q in fits}
    want_applies = {q: q in WIDE_PRIMES for q in fits}
    try:
        KaratsubaMulsub(Mod(65_519))
        refused = False
    except AssertionError:
        refused = True
    rows.add("selection predicates", fits == want_fits and applies == want_applies and refused,
             fits={str(q): v for q, v in fits.items()}, applies={str(q): v for q, v in applies.items()},
             refuses_65519=refused)

    for q in WIDE_PRIMES:
        gen_obj, h = point_add_mulsub(q)
        c = Circuit()
        (gen,) = c.alloc(1)
        x, y = c.alloc(len(h.x)), c.alloc(len(h.y))
        KaratsubaMulsub(Mod(q)).mulsub(c, gen, y, x)
        same = h.ops == c.ops and (gen_obj._square_blk, gen_obj._modadd_blk) == (None, None)
        rows.add(f"point add takes the class at {q.bit_length()} bits", same, ops=len(h.ops) // 4)
    for q in ((1 << 64) - 59, 65_519):
        gen_obj, h = point_add_mulsub(q)
        rng = random.Random(1)
        r = check_lanes(h, [(g, rng.randrange(q), rng.randrange(q)) for _ in range(128) for g in (0, 1)])
        rows.add(f"point add keeps SquareFold at {q}", clean(r) and gen_obj._square_blk is not None, **r)

    # The canonical tail, by enumeration, then every Y, at each tail prime.
    for q in TAIL_PRIMES:
        ys, counts = tail_population(q)
        rows.add(f"tail population {q}", bool(ys), group_ys={"AC": counts[0], "B": counts[1]}, union_ys=len(ys),
                 pairs=len(ys) * q)
        h = build(q)
        rows.timed(f"tail {q}", lambda: check_tail(h, ys), clean)
        rows.timed(f"tail {q} NoCanon control", lambda: check_tail(build(q, NoCanon), ys), failed)
        lanes = all_y_lanes(q)
        rows.timed(f"every Y {q}", lambda: check_lanes(h, lanes), clean)
        if q == CONTROL_PRIME:
            for cls in (NoHighCarry, KOff, NoBTop):
                rows.timed(f"every Y {q} {cls.__name__} control", lambda: check_lanes(build(q, cls), lanes), failed)

    # canon and finish at full width.
    for q in WIDE_PRIMES:
        for piece in ("canon", "finish", "finish1"):
            rows.timed(f"{piece} {q.bit_length()} bits", lambda: check_piece(q, piece), clean)
            rows.timed(f"{piece} {q.bit_length()} bits NoCanon control", lambda: check_piece(q, piece, NoCanon), failed)

    # Seeded samples at full width, with K paid and K merged.
    for q in WIDE_PRIMES:
        n = q.bit_length()
        for merged in (False, True):
            h = build(q, merged=merged)
            ccx = h.ops[0::4].count(CCX)
            for seed in SAMPLE_SEEDS:
                r = check_lanes(h, sample_lanes(q, seed))
                ok = clean(r)
                if n == 256 and not merged:
                    ok = ok and (h.num_wires, ccx) == (SECP_STANDALONE["wires"], SECP_STANDALONE["ccx"])
                rows.add(f"{n} bits seed {seed}{' K merged' if merged else ''}", ok, wires=h.num_wires, ccx=ccx, **r)

    bad = [r["row"] for r in rows.rows if not r["ok"]]
    tail = {r["row"].split()[-1]: r["pairs"] for r in rows.rows if r["row"].startswith("tail population")}
    pieces = sum(r["lanes"] for r in rows.rows if r["row"].split()[0] in ("canon", "finish", "finish1") and "control" not in r["row"])
    every_y = sum(r["lanes"] for r in rows.rows if r["row"].startswith("every Y") and "control" not in r["row"])
    controls = sum("control" in r["row"] for r in rows.rows)
    print(json.dumps({"summary": True, "rows": len(rows.rows), "failed": bad, "tail_pairs": tail, "every_y_lanes": every_y,
                      "piece_lanes": pieces, "controls": controls, "seconds": round(time.time() - start, 1)}), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

"""The Bezout tick's add with its temps reused: every block layout at n <= 7, with the add's own temps cut to one.

    python3 scripts/add_reuse_check.py [--control] [--out FILE]

The add (add_or_shift_la) takes its temps in turn, and with LA_TOP_TEMPS its carry and t serve as two more. At this
branch's 256-bit layout a sweep takes far more temps than the pool holds, so later blocks take temps that earlier blocks
returned; the toy rows and scripts/add_or_shift_check.py, at 5 temps or more, never reuse one. This runs
add_or_shift_check.py's wide and wideb modes (every composition of n - 1 into F, or B, blocks of 1 to 4 bits,
n = 4..7: 112 layouts) with ntemps=1, a pool of three, and requires the same specification on all 2^(2n+3) inputs:
J and R unchanged, every scratch wire 0, word = word + (R >> 1) mod 2^(n+2) when J = 0, and word[:n + 1] rotated left
by one with word[n + 1] unchanged when J = 1, computed bit-sliced with no generator code.
It also records, without simulating, the layouts in which one of the add's own temps (not carry or t) is taken again
after it was returned. With --control, the add's own temps go back to the pool flipped and are flipped back after the
add, so a temp no block takes again ends clean, and the check must fail exactly on the layouts that reuse one.
One JSON line per failing layout and per mode; the last line is the summary. Exits 0 when every layout is exact (or,
with --control, when the failing layouts are exactly the reusing ones, and there are some)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qec import sim, two_bit  # noqa: E402
from qec.builder import Circuit  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--control", action="store_true", help="return the add's own temps dirty; must fail where reused")
ap.add_argument("--out", type=Path)
args = ap.parse_args()
NTEMPS, NMAX = 1, 7
STATE = {"circuit": None, "flips": {}, "reused": False}
_Base = two_bit._LaTemps


class _Watched(_Base):
    """The add's temp pool, noting a reuse of one of its own temps and, with --control, returning them flipped."""

    def __init__(self, wires, idle=None):
        super().__init__(wires, idle)
        self.top = set(wires[:2]) if two_bit.LA_TOP_TEMPS else set()
        self.given = set()

    def take(self):
        w = super().take()
        if w in self.given and w not in self.top:
            STATE["reused"] = True
        return w

    def give(self, w):
        self.given.add(w)
        if args.control and w not in self.top:
            STATE["circuit"].x(w)
            STATE["flips"][w] = STATE["flips"].get(w, 0) ^ 1
        super().give(w)


two_bit._LaTemps = _Watched


def lane_mask(i, N):
    L = 1 << N
    period = 1 << (i + 1)
    block = ((1 << (1 << i)) - 1) << (1 << i)
    return ((1 << L) - 1) // ((1 << period) - 1) * block


def compositions(m, top=4):
    if m == 0:
        yield []
        return
    for r in range(1, min(top, m) + 1):
        for rest in compositions(m - r, top):
            yield [r] + rest


rows, layouts = [], 0
for mode in ("wide", "wideb"):
    for n in range(4, NMAX + 1):
        for sizes in compositions(n - 1):
            layouts += 1
            C = Circuit()
            (J,) = C.alloc(1)
            R, word = C.alloc(n), C.alloc(n + 2)
            ins = [J] + R + word
            STATE.update(circuit=C, flips={}, reused=False)
            kw = {"fsizes": sizes} if mode == "wide" else {"bsizes": sizes}
            two_bit.add_or_shift_la(C, J, R, word, n, ntemps=NTEMPS, **kw)
            for w_, f_ in STATE["flips"].items():
                if f_:
                    C.x(w_)
            N = len(ins)
            full = (1 << (1 << N)) - 1
            init = {w: lane_mask(i, N) for i, w in enumerate(ins)}
            st = sim.run(C.ops, C.num_wires, init, 1 << N)
            j, r, w = init[J], [init[x] for x in R], [init[x] for x in ins[n + 1:]]
            addend = r[1:] + [0, 0, 0]
            total, c = [], 0
            for x, y in zip(w, addend):
                total.append(x ^ y ^ c)
                c = (x & y) | (c & (x ^ y))
            rot = [w[n]] + w[:n] + [w[n + 1]]
            want = [(j & u) | (~j & full & v) for u, v in zip(rot, total)]
            scratch = [x for x in range(C.num_wires) if x not in set(ins)]
            good = (st[J] == j and [st[x] for x in R] == r and [st[x] for x in word] == want
                    and all(st[x] == 0 for x in scratch))
            row = {"mode": mode, "n": n, "sizes": sizes, "exact": good, "reuses_a_temp": STATE["reused"]}
            rows.append(row)
            if not good:
                print(json.dumps(row), flush=True)
reusing = [r for r in rows if r["reuses_a_temp"]]
failing = [r for r in rows if not r["exact"]]
summary = {"summary": True, "control": args.control, "ntemps": NTEMPS, "layouts": layouts,
           "exact_layouts": layouts - len(failing), "reusing_layouts": len(reusing), "failing_layouts": len(failing),
           "failing_are_the_reusing": failing == reusing}
for mode in ("wide", "wideb"):
    print(json.dumps({"mode": mode, "layouts": sum(r["mode"] == mode for r in rows),
                      "reusing": sum(r["mode"] == mode and r["reuses_a_temp"] for r in rows),
                      "failing": sum(r["mode"] == mode and not r["exact"] for r in rows)}))
if args.out:
    args.out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1) + "\n")
print(json.dumps(summary))
ok = (failing == reusing and bool(reusing)) if args.control else not failing
sys.exit(0 if ok else 1)

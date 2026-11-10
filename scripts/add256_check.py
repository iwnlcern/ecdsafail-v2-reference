"""The Bezout tick's add alone at n = 256 with this branch's layout, on every carry run, as built and after the post-pass.

    python3 scripts/add256_check.py [--control] [--out FILE]

It first requires LA_F_WIDE == (3, 63), LA_B_WIDE == (3, 62) and LA_NTEMPS == 14, the layout this branch sets, and that
TwoBitMod._add_or_shift calls add_or_shift_la at secp256k1's n = 256. It builds that call on J, R (256 wires)
and word (258) with fresh scratch, and checks it against scripts/add_or_shift_check.py's specification, computed with
Python integers and no generator code: J and R unchanged, every scratch wire 0, and word = word + (R >> 1) mod 2^258
when J = 0, word's low 257 bits rotated left by one with its top bit unchanged when J = 1.
Lanes at J = 0, with A = R >> 1 (bits 0 to 254): for every i <= j with i < 255 and j < 258, a word and R whose sum
generates a carry at i (word_i = A_i = 1), propagates it through j (word_k ^ A_k = 1, a random split where A has a bit,
word_k = 1 above A's top) and kills it above j (word_k = A_k = 0), with 0 below i and R's bit 0 random: 33,405 runs.
Then word and R all 0 or all 1 (4 lanes) and 2^16 seeded random lanes. At J = 1, the same words with seeded random R.
Two forms: the circuit as built, and the add recorded as a block, written by qec/emit.py with the depth pass on, run
through qec/gate_pass.py's main() (the post-pass the board scores), parsed by qec/flatten.py and expanded by
scripts/check.py's expand(). With --control, one inner fix above a block's first bit (the F sweep's fix of the top bit
of its first block of 3) is dropped, compiled from qec/two_bit.py's own text, and both forms must fail.
One JSON line per form; the last line is the summary. Exits 0 when both forms are exact (or, with --control, both fail)."""
import argparse
import io
import json
import os
import random
import runpy
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import qec  # noqa: E402
from qec import flatten, sim  # noqa: E402
import qec.gate_pass as gp  # noqa: E402
import qec.two_bit as tb  # noqa: E402
from qec.builder import Circuit  # noqa: E402
from qec.curve import SECP256K1  # noqa: E402
from qec.emit import write_circuit  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--control", action="store_true", help="drop one inner fix in a block of 3; both forms must fail")
ap.add_argument("--out", type=Path)
args = ap.parse_args()

assert tb.LA_F_WIDE == (3, 63) and tb.LA_B_WIDE == (3, 62) and tb.LA_NTEMPS == 14, \
    (tb.LA_F_WIDE, tb.LA_B_WIDE, tb.LA_NTEMPS)
assert tb.LOOKAHEAD_ADD and not tb.CDKM_ADD

FIX = "                C.ccx(U[s], T[i - 1], U[s + i])         # fix the inner bits, top down\n"
FIX_CTL = ("                if not (r == 3 and i == 2 and _drop_once()):\n"
           "                    C.ccx(U[s], T[i - 1], U[s + i])     # fix the inner bits, top down\n")
DROP_ONCE = '''

_DROPPED = []


def _drop_once():
    if _DROPPED:
        return False
    _DROPPED.append(True)
    return True
'''


def module():
    if not args.control:
        return tb
    src = (ROOT / "qec" / "two_bit.py").read_text()
    assert src.count(FIX) == 1, "the add's inner-fix line is not in qec/two_bit.py exactly once"
    mod = types.ModuleType("qec.two_bit_add256_control")
    mod.__file__, mod.__package__ = str(ROOT / "qec" / "two_bit.py"), "qec"
    exec(compile(src.replace(FIX, FIX_CTL) + DROP_ONCE, "two_bit_add256_control.py", "exec"), mod.__dict__)
    return mod


N = 256
W = N + 2
MOD = module()
GEN = MOD.TwoBitMod(SECP256K1.q)
assert GEN.n == N, GEN.n


def build(hierarchical):
    if args.control:
        MOD._DROPPED.clear()
    C = Circuit(hierarchical=hierarchical)
    (J,) = C.alloc(1)
    R, word = C.alloc(N), C.alloc(W)
    regs = [[J], R, word]
    if hierarchical:
        block = C.record(lambda j, r, w: GEN._add_or_shift(C, j[0], r, w), regs)
        C.apply(block, regs)
    else:
        GEN._add_or_shift(C, J, R, word)
    if args.control:
        assert MOD._DROPPED, "the control dropped no fix: no block of 3 in F"
    return C, J, R, word


def post_ops(C, io_wires, tmp):
    buf = io.StringIO()
    write_circuit(buf, C, {"control": [], "x": io_wires, "y": []})
    src, dst = os.path.join(tmp, "in.json"), os.path.join(tmp, "out.json")
    with open(src, "w") as fh:
        fh.write(buf.getvalue())
    old, sys.stdout = sys.stdout, io.StringIO()
    try:
        gp.main(src, dst)
        log = json.loads(sys.stdout.getvalue())
    finally:
        sys.stdout = old
    with open(dst) as fh:
        doc = flatten.load(fh)
    expand = runpy.run_path(str(ROOT / "scripts" / "check.py"))["expand"]
    return expand(doc), doc["num_wires"], log


def lanes():
    """(j, R, word) per lane, as Python integers."""
    rng = random.Random(256)
    top_a = N - 1                                   # A = R >> 1 has bits 0 .. 254
    words, rs = [], []
    for i in range(top_a):
        for j in range(i, W):
            wv = av = 0
            wv |= 1 << i
            av |= 1 << i
            for k in range(i + 1, j + 1):
                if k < top_a:
                    b = rng.getrandbits(1)
                    wv |= b << k
                    av |= (1 - b) << k
                else:
                    wv |= 1 << k
            words.append(wv)
            rs.append((av << 1) | rng.getrandbits(1))
    runs = len(words)
    for wv in (0, (1 << W) - 1):
        for rv in (0, (1 << N) - 1):
            words.append(wv)
            rs.append(rv)
    for _ in range(1 << 16):
        words.append(rng.getrandbits(W))
        rs.append(rng.getrandbits(N))
    out = [(0, r, w) for r, w in zip(rs, words)] + [(1, rng.getrandbits(N), w) for w in words]
    return out, runs


def spec(j, r, w):
    if j:
        low = w & ((1 << (N + 1)) - 1)
        return (w >> (N + 1) << (N + 1)) | ((low << 1) & ((1 << (N + 1)) - 1)) | (low >> N)
    return (w + (r >> 1)) % (1 << W)


def columns(vals, width):
    """Per bit b < width, an int whose bit l is bit b of vals[l]."""
    rows = [format(v, f"0{width}b") for v in vals]
    cols = list(zip(*rows))
    return [int("".join(cols[width - 1 - b])[::-1], 2) for b in range(width)]


CHUNK = 1 << 14


def check_form(ops, num_wires, J, R, word, cases):
    io_set = {J, *R, *word}
    scratch = [x for x in range(num_wires) if x not in io_set]
    failing = 0
    for c0 in range(0, len(cases), CHUNK):
        chunk = cases[c0:c0 + CHUNK]
        L = len(chunk)
        js = columns([c[0] for c in chunk], 1)[0]
        rc = columns([c[1] for c in chunk], N)
        wc = columns([c[2] for c in chunk], W)
        init = {J: js, **dict(zip(R, rc)), **dict(zip(word, wc))}
        st = sim.run(ops, num_wires, init, L)
        want = columns([spec(*c) for c in chunk], W)
        bad = st[J] ^ js
        for x, v in zip(R, rc):
            bad |= st[x] ^ v
        for x, v in zip(word, want):
            bad |= st[x] ^ v
        for x in scratch:
            bad |= st[x]
        failing += bin(bad).count("1")
    return failing


t0 = time.time()
cases, runs = lanes()
rows = []
C, J, R, word = build(False)
ccx = C.ops[0::4].count(2)
rows.append({"form": "built", "lanes": len(cases), "carry_runs": runs, "ccx": ccx, "wires": C.num_wires,
             "failing_lanes": check_form(C.ops, C.num_wires, J, R, word, cases)})
print(json.dumps(rows[-1]), flush=True)
H, J, R, word = build(True)
with tempfile.TemporaryDirectory() as tmp:
    ops, nw, log = post_ops(H, [J, *R, *word], tmp)
rows.append({"form": "post", "lanes": len(cases), "carry_runs": runs, "pass": log,
             "pairs_cancelled": log["records_cancelled"] // 2,
             "failing_lanes": check_form(ops, nw, J, R, word, cases)})
print(json.dumps(rows[-1]), flush=True)
exact = all(r["failing_lanes"] == 0 for r in rows)
fails_both = all(r["failing_lanes"] > 0 for r in rows)
summary = {"summary": True, "control": args.control, "n": N, "layout": [list(tb.LA_F_WIDE), list(tb.LA_B_WIDE), tb.LA_NTEMPS],
           "lanes": len(cases), "carry_runs": runs, "forms": [r["form"] for r in rows], "exact": exact,
           "control_fails_both": fails_both if args.control else None, "pairs_cancelled": rows[1]["pairs_cancelled"],
           "seconds": round(time.time() - t0, 1)}
if args.out:
    args.out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1) + "\n")
print(json.dumps(summary))
sys.exit(0 if (fails_both if args.control else exact) else 1)

"""Prefix with a block-length cap, checked against its contract on structured inputs at the tick's sizes, with its control.

    python3 scripts/prefix_check.py [--out FILE]

The cap only takes effect when the doubling would build a block longer than it, so at small moduli it never fires
and the exhaustive small-q rows do not reach it. This builds Prefix(bits, cap) standalone at lengths 221, 190, 163,
100, 64, 37, 5 and 2 and at caps none, 63, 48, 37, 30, 26, 23, 20, 16 and 13, with and without release/restore
around a read of top (as M uses it), and, when Prefix takes `nor`, with the write's per-block clear (nor off) and
its complemented-prefix clear (nor on), and checks bit-sliced over lanes:
  after construction  top = AND(bits) (the bits hold chain values under X until write, by design);
  after write(t)      bits = bits + t mod 2^h, t unchanged, every other wire 0.
Lanes: all ones, all ones with one 0 at each position, all ones below each block start and 0 at it, 0, and 64
random words, each with t = 0 and t = 1.
Controls, each of which must fail:
  cap    a capped block takes its top from the first block's top instead of the last's;
  early  `write`'s complemented-prefix clears read that prefix straight after building it, before P's higher
         blocks unwind (a copy of this tree's `qec` with that one region of `Prefix.write` replaced).
The last line printed is a JSON summary.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTROLS = ("cap", "early")
EARLY = ("        # Each block's carry a*t is still on its top, and a*t = t*AND(~new bits below the block). Build a prefix of\n        # the complemented bits and clear each carry from it as that prefix unwinds top down: a block's top is read\n        # only once every block above it is unwound, as in write above.\n        for w in bits:\n            C.x(w)\n        N = Prefix(C, bits, self.cap)\n        for (_, p, _, _, _), (na, np_, x, u, v) in zip(reversed(self.blocks), reversed(N.blocks)):\n            C.ccx(t, na, p)\n            C.ccx(na, v, np_)\n            for i in reversed(range(1, len(x))):\n                C.ccx(x[0] if i == 1 else u[i - 2], x[i], u[i - 1])\n            for w in reversed(u):\n                C.x(w)\n            C.free([np_])\n        C.ccx(bits[0], bits[1], N.first)\n        C.free([N.first])\n",
         "        # Each block's carry a*t is still on its top: a*t = t*AND(~new bits below the block), so clear the carries\n        # from a prefix of the complemented bits, built and unbuilt here, instead of one block at a time above.\n        from .builder import CX as _CX, X as _X\n        for w in bits:\n            C.x(w)\n        start = len(C.prims)\n        N = Prefix(C, bits, self.cap)\n        built = C.prims[start:]\n        for (_, p, _, _, _), (na, _, _, _, _) in zip(self.blocks, N.blocks):\n            C.ccx(t, na, p)\n        for q in range(len(built) - 4, -1, -4):\n            op, i, j, k = built[q:q + 4]\n            if op == _X:\n                C.x(i)\n            elif op == _CX:\n                C.cx(i, j)\n            else:\n                C.ccx(i, j, k)\n        C.free([N.first] + [np_ for _, np_, _, _, _ in N.blocks])\n")


def worker(control, tree):
    sys.path.insert(0, str(tree))
    import inspect
    import qec.two_bit as tb
    from qec.builder import CCX, CX, SWAP, X, Circuit
    assert tb.__file__.startswith(str(tree) + os.sep)
    nors = (False, True) if "nor" in inspect.signature(tb.Prefix.__init__).parameters else (None,)
    if control == "cap":
        src = inspect.getsource(tb.Prefix.__init__)
        assert src.count("C.ccx(tops[-1], v, p)") == 1
        ns = {}
        body = src.replace("C.ccx(tops[-1], v, p)", "C.ccx(tops[0] if m == cap else tops[-1], v, p)")
        body = "\n".join(line[4:] for line in body.split("\n"))
        exec(body, tb.__dict__, ns)
        tb.Prefix.__init__ = ns["__init__"]
    mask = [0]

    def run(prims, state):
        for p in range(0, len(prims), 4):
            op, a, b, c = prims[p:p + 4]
            if op == X:
                state[a] ^= mask[0]
            elif op == CX:
                state[b] ^= state[a]
            elif op == CCX:
                state[c] ^= state[a] & state[b]
            elif op == SWAP:
                state[a], state[b] = state[b], state[a]
            else:
                raise ValueError(op)

    def check(h, cap, release, nor):
        words = [(1 << h) - 1, 0]
        words += [((1 << h) - 1) ^ (1 << i) for i in range(h)]
        words += [(1 << i) - 1 for i in range(h + 1)]
        rnd = random.Random(h * 1000 + (cap or 0))
        words += [rnd.getrandbits(h) for _ in range(64)]
        lanes = [(w, t) for w in words for t in (0, 1)]
        mask[0] = (1 << len(lanes)) - 1
        C = Circuit()
        bits = C.alloc(h)
        (t,) = C.alloc(1)
        kw = {} if nor is None else {"nor": nor}
        P = tb.Prefix(C, bits, cap, **kw) if cap else tb.Prefix(C, bits, **kw)
        if release:
            P.release()
        built = len(C.prims)
        if release:
            P.restore()
        top = P.top
        P.write(t)
        state = [0] * C.num_wires
        for k, (w, tv) in enumerate(lanes):
            for i in range(h):
                if w >> i & 1:
                    state[bits[i]] |= 1 << k
            if tv:
                state[t] |= 1 << k
        run(C.prims[:built], state)
        bad = 0
        if not release and top is not None:
            for k, (w, tv) in enumerate(lanes):
                want = int(w == (1 << h) - 1)
                bad += (state[top] >> k & 1) != want
        run(C.prims[built:], state)
        for k, (w, tv) in enumerate(lanes):
            out = sum((state[bits[i]] >> k & 1) << i for i in range(h))
            bad += out != (w + tv) % (1 << h)
            bad += (state[t] >> k & 1) != tv
        live = set(bits) | {t}
        bad += sum(1 for wire in range(C.num_wires) if wire not in live and state[wire])
        return bad, len(lanes)

    total = cases = lanes = 0
    for h in (221, 190, 163, 100, 64, 37, 5, 2):
        for cap in (None, 63, 48, 37, 30, 26, 23, 20, 16, 13):
            for release in (False, True):
                for nor in nors:
                    bad, n = check(h, cap, release, nor)
                    total += bad
                    cases += 1
                    lanes += n
    print(json.dumps({"control": control, "nor": list(nors), "cases": cases, "lanes": lanes, "failures": total}))


def spawn(control, tree=ROOT):
    p = subprocess.run([sys.executable, "-I", "-B", __file__, "--worker", control or "none", "--tree", str(tree)],
                       capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode or not lines:
        return {"control": control, "error": f"exit {p.returncode}: {p.stderr.strip()[-600:]}"}
    return json.loads(lines[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker")
    ap.add_argument("--tree", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.worker:
        worker(None if args.worker == "none" else args.worker, args.tree.resolve())
        return
    check = spawn(None)
    controls = {"cap": spawn("cap")}
    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(ROOT / "qec", Path(td) / "qec", ignore=shutil.ignore_patterns("__pycache__"))
        p = Path(td) / "qec" / "two_bit.py"
        text = p.read_text()
        if text.count(EARLY[0]) != 1:
            controls["early"] = {"control": "early", "error": f"anchor matches {text.count(EARLY[0])} times"}
        else:
            p.write_text(text.replace(*EARLY))
            controls["early"] = spawn(None, Path(td).resolve())
            controls["early"]["control"] = "early"
    problems = []
    if "error" in check:
        problems.append(f"check: {check['error']}")
    elif check["failures"]:
        problems.append(f"check: {check['failures']} failures")
    for c, r in controls.items():
        if "error" in r:
            problems.append(f"control {c}: {r['error']}")
        elif not r["failures"]:
            problems.append(f"control {c} not rejected")
    summary = {"summary": True, "check": check, "controls": controls, "problems": problems, "ok": not problems}
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()

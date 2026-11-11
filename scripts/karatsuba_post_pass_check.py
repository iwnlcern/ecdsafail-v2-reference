"""Exactness of the Karatsuba square-subtract under the scored post-pass (qec/gate_pass.py), with a control that must fail.

    python3 scripts/karatsuba_post_pass_check.py [--out FILE]

The point add takes qec.square.KaratsubaMulsub only at 128 bits and up, so no toy row and no instance of
scripts/post_pass_check.py reaches it. This check builds the class alone, on registers gen, X and Y with K paid by
the class, at the forced small primes of scripts/karatsuba_check.py (65,521, 262,121 and 262,139) and at its widths
(2^128 - 159 and secp256k1). Each instance is checked in three forms on the same lanes:
  built    the flat build;
  raw      the hierarchical build written by qec/emit.py (depth pass on, as scripts/emit.py writes it), expanded by
           scripts/check.py's rules;
  post     that JSON after qec/gate_pass.py's main(), expanded the same way.
At the small primes the lanes are karatsuba_check.py's exhaustive ones: every Y with gen in {0, 1} and a seeded X,
and every X with gen = 1 for each Y whose group value reaches the canonical tail. At the widths they are its seeded
samples at seeds 2041, 7331 and 6062. Each lane requires X = (X - gen * Y^2) mod q, Y and gen restored, and every
other wire 0.
control  removes one pair the pass may not cancel (a gate and its next copy, with a gate between them on a shared
         wire that does not commute with them) from one body of the raw JSON, and then runs the same pass. Every
         gate of the standalone circuit is the class's, so the candidates come from every definition, in order,
         and then the top level, CCX pairs first within each body. It tries one candidate at a time, up to 64 an
         instance, until the lanes catch one, and counts the tries that left the function unchanged on the lanes.
pairs    the pass log's records_cancelled, halved: the pairs the pass cancels in the class.
One JSON line per instance; the last line is the summary. Exits 0 when every form is exact on every instance and
the control catches on at least one instance."""
import argparse
import bisect
import io
import json
import os
import runpy
import sys
import tempfile
import time
from pathlib import Path

TREE = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "scripts"))

from karatsuba_check import (SAMPLE_SEEDS, TAIL_PRIMES, WIDE_PRIMES, Harness, all_y_lanes, build, check_lanes,  # noqa: E402
                             check_tail, prepare, sample_lanes, tail_population)
import qec.gate_pass as gp  # noqa: E402
from qec import flatten  # noqa: E402
from qec.arith import Mod  # noqa: E402
from qec.builder import Circuit  # noqa: E402
from qec.emit import write_circuit  # noqa: E402
from qec.square import KaratsubaMulsub  # noqa: E402

CK = runpy.run_path(os.path.join(TREE, "scripts", "check.py"), run_name="check")
TRIES = 64


def emitted(q):
    """The class's hierarchical build at q, as the JSON text qec/emit.py writes, and its registers."""
    c, mod = Circuit(hierarchical=True), Mod(q)
    (gen,) = c.alloc(1)
    x, y = c.alloc(mod.n), c.alloc(mod.n)
    KaratsubaMulsub(mod).mulsub(c, gen, y, x)
    buf = io.StringIO()
    write_circuit(buf, c, {"control": [gen], "x": x, "y": y})
    return buf.getvalue(), gen, x, y


def harness(q, text, gen, x, y):
    doc = flatten.load(io.StringIO(text))
    return Harness(q, CK["expand"](doc), doc["num_wires"], gen, x, y, 0)


def run_pass(text, tmp, name):
    src, dst = os.path.join(tmp, name + ".in.json"), os.path.join(tmp, name + ".out.json")
    with open(src, "w") as fh:
        fh.write(text)
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        gp.main(src, dst)
    finally:
        sys.stdout = old
    with open(dst) as fh:
        return fh.read(), json.loads(buf.getvalue())


def wrong_pairs(ops):
    """Every (j, i) of identical records, i the next copy after j, where some record between them shares a wire
    and does not commute with them, so the pass may not cancel them: CCX pairs first, then CX, then X. The same
    candidates as scripts/post_pass_check.py's, found through per-wire indexes."""
    keys = [None if "call" in r else gp.key_of(r) for r in ops]
    on_wire = {}
    for i, r in enumerate(ops):
        for w in gp.wires_of(r):
            on_wire.setdefault(w, []).append(i)
    nxt, last = {}, {}
    for i in reversed(range(len(ops))):
        k = keys[i]
        if k is not None:
            if k in last:
                nxt[i] = last[k]
            last[k] = i
    found = []
    for j, i in nxt.items():
        r = ops[j]
        blocked = False
        for w in set(gp.wires_of(r)):
            idx = on_wire[w]
            for t in idx[bisect.bisect_right(idx, j):bisect.bisect_left(idx, i)]:
                if not gp.commute(ops[t], r):
                    blocked = True
                    break
            if blocked:
                break
        if blocked:
            found.append((j, i))
    rank = {"CCX": 0, "CX": 1, "X": 2}
    return sorted(found, key=lambda p: (rank.get(ops[p[0]].get("op"), 3), p[0]))


class Lanes:
    """An instance's lanes, prepared once, checked on any form's flat ops."""

    def __init__(self, q, h):
        self.q = q
        if q in TAIL_PRIMES:
            self.every_y = prepare(h, all_y_lanes(q))
            self.ys, _ = tail_population(q)
            self.samples = []
        else:
            self.every_y, self.ys = None, []
            self.samples = [prepare(h, sample_lanes(q, seed)) for seed in SAMPLE_SEEDS]

    def count(self):
        if self.every_y is not None:
            return self.every_y[-1] + len(self.ys) * self.q
        return sum(s[-1] for s in self.samples)

    def check(self, h, stop=False):
        """{"lanes", "wrong", "dirty"} summed over the lanes. With stop, return at the first failing population."""
        total = {"lanes": 0, "wrong": 0, "dirty": 0}
        parts = [lambda: check_lanes(h, self.every_y)] if self.every_y is not None else []
        if self.ys:
            parts.append(lambda: check_tail(h, self.ys))
        parts += [lambda s=s: check_lanes(h, s) for s in self.samples]
        for part in parts:
            r = part()
            total["lanes"] += r.get("lanes", r.get("pairs", 0))
            total["wrong"] += r["wrong"]
            total["dirty"] += r["dirty"]
            if stop and (r["wrong"] or r["dirty"]):
                break
        return total


def instance(q, tmp):
    t0 = time.time()
    built = build(q)
    lanes = Lanes(q, built)
    text, gen, x, y = emitted(q)
    raw = harness(q, text, gen, x, y)
    post_text, log = run_pass(text, tmp, "post")
    post = harness(q, post_text, gen, x, y)
    forms = {name: lanes.check(h) for name, h in (("built", built), ("raw", raw), ("post", post))}
    doc = json.loads(text)
    bodies = [d["operations"] for d in doc["definitions"]] + [doc["operations"]]
    cands = [(b, p) for b, ops in enumerate(bodies) for p in wrong_pairs(ops)]
    ctl = {"candidates": len(cands), "tried": 0, "benign": 0, "caught": None}
    for b, (j, i) in cands[:TRIES]:
        bad = json.loads(text)
        ops = bad["definitions"][b]["operations"] if b < len(doc["definitions"]) else bad["operations"]
        op = ops[j].get("op")
        del ops[i], ops[j]
        ctl_text, _ = run_pass(json.dumps(bad, separators=(",", ":")), tmp, "ctl")
        r = lanes.check(harness(q, ctl_text, gen, x, y), stop=True)
        ctl["tried"] += 1
        if r["wrong"] or r["dirty"]:
            where = "top level" if b == len(doc["definitions"]) else f"definition {b}"
            ctl["caught"] = {"body": where, "pair": [j, i], "op": op, "wrong": r["wrong"], "dirty": r["dirty"]}
            break
        ctl["benign"] += 1
    name = f"q = {q}" if q in TAIL_PRIMES else f"{q.bit_length()} bits"
    return {"instance": name, "q": str(q), "lanes": lanes.count(), "wires": raw.num_wires,
            "definitions": len(doc["definitions"]), "forms": forms, "pass": log,
            "pairs_cancelled": log["records_cancelled"] // 2, "control": ctl, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="also write the rows and summary here as JSON")
    args = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for q in (*TAIL_PRIMES, *WIDE_PRIMES):
            row = instance(q, tmp)
            print(json.dumps(row), flush=True)
            rows.append(row)
    exact = all(f["wrong"] == 0 and f["dirty"] == 0 for r in rows for f in r["forms"].values())
    caught = [r["instance"] for r in rows if r["control"]["caught"]]
    summary = {"summary": True, "instances": len(rows), "lanes": sum(r["lanes"] for r in rows),
               "built_raw_post_exact": exact, "control_caught_rows": len(caught), "control_caught": caught,
               "control_benign_tries": sum(r["control"]["benign"] for r in rows),
               "pairs_cancelled": sum(r["pairs_cancelled"] for r in rows)}
    if args.out:
        args.out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1) + "\n")
    print(json.dumps(summary), flush=True)
    sys.exit(0 if exact and caught else 1)


if __name__ == "__main__":
    main()

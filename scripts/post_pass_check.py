"""Exactness of the scored post-pass (qec/gate_pass.py) on the Bezout tick's probe tree, with a control that must fail.

    python3 scripts/post_pass_check.py [--out FILE]

The board scores the method after qec/gate_pass.py, which cancels pairs and reorders CCX controls in every definition,
so this checks the pass on the probe's carry-out tree (TwoBitMod._probe_tree), exhaustively. The instances are
scripts/check.py's toy rows (bits 4 and 6 to 10, 6 curves a size, 2 addends a curve) and, for each of 14 to 17 bits,
the largest prime under 2^bits with the tick's merged path and a prime-order y^2 = x^3 + 7, with one addend: 40 in all.
For each instance it builds the hierarchical circuit, writes it with qec/emit.py (depth pass on, as scripts/emit.py
does), and:
  raw      expands the JSON by check.py's rules and runs check.py's exhaustive check (every point, both controls);
  post     runs qec/gate_pass.py's main() on that JSON and checks the result the same way;
  control  removes one pair the pass may not cancel (one of the tree's gates and its next copy, with a non-commuting
           gate between them on a shared wire) and then runs the same pass, one candidate at a time, CCX pairs first,
           up to 20 an instance, until the check catches one; the row counts the tries that left the function
           unchanged (such a removal is wrong for the pass but harmless on that instance).
The tree's gates are found by wrapping _probe_tree: each call's op range is kept with the block (the definition) it is
recorded in, and its gates, in that definition's local wires, are the tree's keys. pairs_cancelled_with_tree_keys counts
a cancelled pair when its key is one of those keys in that definition: an upper bound on the pairs that include a tree
gate, since other code in the definition can emit the same gate. The pass's cancel step is re-run with the pairs
recorded, and must keep exactly the records gate_pass.cancel_pairs keeps.
One JSON line per instance; the last line is the summary. Exits 0 when raw and post have 0 failures on every instance
and the control fails on some instance."""
import argparse, collections, copy, io, json, os, runpy, sys, tempfile, time
from pathlib import Path

TREE = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, TREE)
import qec.builder as qb  # noqa: E402
import qec.two_bit as tb  # noqa: E402
from qec import flatten  # noqa: E402
from qec.curve import Curve, is_prime, toy_secp_curves  # noqa: E402
from qec.emit import definitions, write_circuit  # noqa: E402
import qec.gate_pass as gp  # noqa: E402

CK = runpy.run_path(os.path.join(TREE, "scripts", "check.py"), run_name="check")
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path)
out_path = ap.parse_args().out
spec = ["toy", "14", "15", "16", "17"]

# The tree's op ranges: (circuit, first op, end op) while _probe_tree runs; Circuit.record moves the ranges recorded
# inside it onto the block it returns, relative to the block's ops.
RANGES = []


def tracked(fn):
    def run(*args, **kw):
        C = next(a for a in args if isinstance(a, qb.Circuit))
        p0 = len(C.prims) // 4
        out = fn(*args, **kw)
        RANGES.append((C, p0, len(C.prims) // 4))
        return out
    return run


assert hasattr(tb.TwoBitMod, "_probe_tree"), "qec/two_bit.py has no TwoBitMod._probe_tree"
tb.TwoBitMod._probe_tree = tracked(tb.TwoBitMod._probe_tree)
_record = qb.Circuit.record


def record(self, fn, regs):
    start, mark = len(self.prims) // 4, len(RANGES)
    block = _record(self, fn, regs)
    block.tree_ranges = [(a - start, b - start) for C, a, b in RANGES[mark:] if C is self and a >= start]
    del RANGES[mark:]
    return block


qb.Circuit.record = record


def key(op, a, b, c, local):
    if op == qb.X:
        return ("X", local[a])
    if op == qb.CX:
        return ("CX", local[a], local[b])
    if op == qb.CCX:
        return ("CCX", frozenset((local[a], local[b])), local[c])
    return ("SWAP", frozenset((local[a], local[b])))


def our_keys(order):
    """Per definition index: a Counter of the tree's gates' keys in that definition's local wires."""
    res = {}
    for i, block in enumerate(order):
        rs = getattr(block, "tree_ranges", [])
        if not rs:
            continue
        local = {w: k for k, w in enumerate(sorted(block.wires))}
        cnt = collections.Counter()
        for a, b in rs:
            for p in range(a, b):
                cnt[key(*block.ops[4 * p:4 * p + 4], local)] += 1
        res[i] = cnt
    return res


def cancel_recorded(ops):
    """gate_pass.cancel_pairs, also returning the cancelled pairs' keys."""
    alive, on_wire, pairs = [True] * len(ops), {}, []
    for i, r in enumerate(ops):
        ws = gp.wires_of(r)
        k = None if "call" in r else gp.key_of(r)
        partner = None
        if k is not None:
            for w in ws:
                found = None
                for j in reversed(on_wire.get(w, ())):
                    if not alive[j]:
                        continue
                    h = ops[j]
                    if "call" not in h and gp.key_of(h) == k:
                        found = j
                        break
                    if not gp.commute(h, r):
                        break
                if found is None or (partner is not None and partner != found):
                    partner = None
                    break
                partner = found
        if partner is not None:
            alive[partner] = alive[i] = False
            pairs.append(k)
            continue
        for w in ws:
            on_wire.setdefault(w, []).append(i)
    return [r for r, a in zip(ops, alive) if a], pairs


def wrong_pairs(ops, keys):
    """Every (j, i) of identical records with one of the tree's keys, i the next copy after j, where some record
    between them shares a wire and does not commute with them, so the pass may not cancel them: CCX pairs first, then
    CX, then X."""
    found = []
    for j, r in enumerate(ops):
        k = None if "call" in r else gp.key_of(r)
        if k is None or k not in keys:
            continue
        ws = set(gp.wires_of(r))
        blocked = False
        for i in range(j + 1, len(ops)):
            h = ops[i]
            if "call" not in h and gp.key_of(h) == k:
                if blocked:
                    found.append((j, i))
                break
            if ws & set(gp.wires_of(h)) and not gp.commute(h, r):
                blocked = True
    rank = {"CCX": 0, "CX": 1, "X": 2}
    return sorted(found, key=lambda p: rank.get(ops[p[0]].get("op"), 3))


TRIES = 20


def secp_shaped(bits):
    for q in range((1 << bits) - 1, 1 << (bits - 1), -2):
        if is_prime(q) and tb.TwoBitMod(q).merged:
            c = Curve(q, 0, 7)
            if is_prime(len(c.points())):
                return c
    return None


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


def check_text(gen, text, cases):
    doc = flatten.load(io.StringIO(text))
    ops = CK["expand"](doc)
    return len(CK["check"](gen, ops, cases, num_wires=doc["num_wires"]))


instances = []
for s in spec:
    if s == "toy":
        for bits in (4, 6, 7, 8, 9, 10):
            for curve in toy_secp_curves(bits)[:6]:
                pts = curve.points()
                instances += [(f"toy{bits}", curve, A) for A in [P for P in pts[1:] if P[1] != 0][:2]]
    else:
        curve = secp_shaped(int(s))
        pts = curve.points()
        instances.append((f"forced{s}", curve, next(P for P in pts[1:] if P[1] != 0)))

rows = []
with tempfile.TemporaryDirectory() as tmp:
    for name, curve, A in instances:
        t0 = time.time()
        RANGES.clear()
        gen = CK["ControlledPointAdd"](curve, A)
        C = gen.build(hierarchical=True)
        order = definitions(C)
        keys = our_keys(order)
        buf = io.StringIO()
        write_circuit(buf, C, gen.layout)
        text = buf.getvalue()
        raw_doc = json.loads(text)
        circuit = raw_doc.get("circuit", raw_doc)
        cancelled, with_keys, in_defs = 0, 0, 0
        for i, d in enumerate(circuit.get("definitions", [])):
            kept, pairs = cancel_recorded(copy.deepcopy(d["operations"]))
            assert kept == gp.cancel_pairs(copy.deepcopy(d["operations"]))[0]
            cancelled += len(pairs)
            if i in keys:
                in_defs += len(pairs)
                with_keys += sum(1 for k in pairs if k in keys[i])
        cases = [(c, R) for c in (0, 1) for R in curve.points()]
        raw_fails = check_text(gen, text, cases)
        post_text, log = run_pass(text, tmp, "post")
        post_fails = check_text(gen, post_text, cases)
        # The control: one wrong cancel at a time, from the candidates in the tree's definitions (CCX first), until
        # the check catches one; the row reports how many it tried and how many of those left the function unchanged.
        cands = [(i, p) for i in sorted(keys) for p in wrong_pairs(circuit["definitions"][i]["operations"], keys[i])]
        ctl = {"candidates": len(cands), "tried": 0, "benign": 0, "caught": None} if cands else None
        for i, wp in cands[:TRIES]:
            bad = json.loads(text)
            ops = bad.get("circuit", bad)["definitions"][i]["operations"]
            r = ops[wp[0]]
            del ops[wp[1]], ops[wp[0]]
            ctl_text, _ = run_pass(json.dumps(bad, separators=(",", ":")), tmp, "ctl")
            fails = check_text(gen, ctl_text, cases)
            ctl["tried"] += 1
            if fails:
                ctl["caught"] = {"definition": i, "pair": list(wp), "op": r.get("op"), "failures": fails}
                break
            ctl["benign"] += 1
        row = {"instance": name, "q": curve.q, "addend": list(A), "lanes": len(cases), "tree_definitions": len(keys),
               "tree_gates": sum(sum(c.values()) for c in keys.values()), "pairs_cancelled": cancelled,
               "pairs_cancelled_in_tree_definitions": in_defs, "pairs_cancelled_with_tree_keys": with_keys,
               "pass": log, "raw_failures": raw_fails, "post_failures": post_fails, "control": ctl,
               "seconds": round(time.time() - t0, 1)}
        print(json.dumps(row), flush=True)
        rows.append(row)
ok = all(r["raw_failures"] == 0 and r["post_failures"] == 0 for r in rows)
caught = [r for r in rows if r["control"] and r["control"]["caught"]]
ctl_ok = bool(caught)
summary = {"summary": True, "instances": len(rows), "lanes": sum(r["lanes"] for r in rows),
           "raw_and_post_exact": ok, "control_rows": sum(1 for r in rows if r["control"]),
           "control_caught_rows": len(caught),
           "control_benign_tries": sum(r["control"]["benign"] for r in rows if r["control"]),
           "pairs_cancelled_with_tree_keys": sum(r["pairs_cancelled_with_tree_keys"] for r in rows)}
if out_path:
    out_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1) + "\n")
print(json.dumps(summary))
sys.exit(0 if ok and ctl_ok else 1)

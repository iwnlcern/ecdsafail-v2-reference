"""Certificate for qec/depth_pass.py, sharing no code with it.

For each instance the circuit is built once and written twice, without and with the pass. Every
definition must keep, on every wire, the same sequence of role runs, each run holding the same ops with
CCX controls sorted. A run is a maximal stretch of ops that read the wire ('c') or only XOR into it
('t'); SWAP wires and call wires a callee both reads and writes ('e') are runs of one. Ops that share
only c/c or t/t wires commute, so equal run sequences mean the same permutation. The top level may
differ only in CCX control order. The control swaps the first adjacent non-commuting pair of one passed
definition and must be rejected."""
import argparse
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qec.curve import SECP256K1, SECP256K1_G, toy_secp_curves
from qec.emit import write_circuit
from qec.point_add import ControlledPointAdd


def norm(ins):
    """(kind, sorted controls or wires, target) for a primitive; the call record itself for a call."""
    if "call" in ins:
        return ("call", ins["call"], tuple(ins["wires"]), ins["inverse"])
    op = ins["op"]
    if op == "X":
        return ("X", (), ins["target"])
    if op == "CX":
        return ("CX", (ins["control"],), ins["target"])
    if op == "CCX":
        return ("CCX", tuple(sorted((ins["control1"], ins["control2"]))), ins["target"])
    return ("SWAP", tuple(sorted((ins["wire1"], ins["wire2"]))), None)


def roles(ins, summaries):
    n = norm(ins)
    if n[0] == "call":
        rs = summaries[n[1]]
        return [(n[2][w], r) for w, r in sorted(rs.items())]
    if n[0] == "SWAP":
        return [(w, "e") for w in n[1]]
    return [(w, "c") for w in n[1]] + [(n[2], "t")]


def summarize(defs):
    out = []
    for d in defs:
        acc = {}
        for ins in d["operations"]:
            for w, r in roles(ins, out):
                acc[w] = r if acc.get(w, r) == r else "e"
        out.append(acc)
    return out


def runs(body, summaries):
    """wire -> list of (role, Counter of normalised ops) runs."""
    per = {}
    for ins in body:
        n = norm(ins)
        for w, r in roles(ins, summaries):
            lst = per.setdefault(w, [])
            if lst and r != "e" and lst[-1][0] == r:
                lst[-1][1][n] += 1
            else:
                lst.append((r, Counter({n: 1})))
    return per


def certify(raw, passed, summaries):
    return len(raw) == len(passed) and runs(raw, summaries) == runs(passed, summaries)


def conflict(a, b, summaries):
    ra, rb = dict(roles(a, summaries)), dict(roles(b, summaries))
    return any(w in rb and not (r == rb[w] != "e") for w, r in ra.items())


def instance(name, curve, A):
    gen = ControlledPointAdd(curve, A)
    C = gen.build(hierarchical=True)
    docs, texts = [], []
    for flag in (False, True):
        buf = io.StringIO()
        write_circuit(buf, C, gen.layout, depth_pass=flag)
        texts.append(buf.getvalue())
        docs.append(json.loads(texts[-1]))
    raw, passed = docs
    summaries = summarize(raw["definitions"])
    failures = []
    if raw["num_wires"] != passed["num_wires"] or raw["registers"] != passed["registers"]:
        failures.append("interface")
    if len(raw["definitions"]) != len(passed["definitions"]):
        failures.append("definition count")
    for i, (d0, d1) in enumerate(zip(raw["definitions"], passed["definitions"])):
        if d0["arity"] != d1["arity"] or not certify(d0["operations"], d1["operations"], summaries):
            failures.append(f"definition {i}")
    if [norm(x) for x in raw["operations"]] != [norm(x) for x in passed["operations"]]:
        failures.append("top level")
    # Control: swap the first adjacent non-commuting pair in the largest passed definition.
    big = max(range(len(passed["definitions"])), key=lambda i: len(passed["definitions"][i]["operations"]))
    body = list(passed["definitions"][big]["operations"])
    pos = next(i for i in range(len(body) - 1) if conflict(body[i], body[i + 1], summaries))
    body[pos], body[pos + 1] = body[pos + 1], body[pos]
    rejected = not certify(raw["definitions"][big]["operations"], body, summaries)
    ops = sum(len(d["operations"]) for d in passed["definitions"])
    return {"name": name, "sha256": hashlib.sha256(texts[1].encode()).hexdigest(),
            "definitions": len(passed["definitions"]), "definition_ops": ops,
            "reordered_definitions": sum(d0 != d1 for d0, d1 in zip(raw["definitions"], passed["definitions"])),
            "failures": failures, "control": {"definition": big, "position": pos}, "control_rejected": rejected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--toy-only", action="store_true")
    args = ap.parse_args()
    toy = next(c for c in toy_secp_curves(7) if c.q == 127)
    cases = [("toy127", toy, toy.points()[5])]
    if not args.toy_only:
        cases.append(("secp256k1 G", SECP256K1, SECP256K1_G))
    rows = [instance(*c) for c in cases]
    out = {"instances": rows}
    if args.out:
        args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    sys.exit(1 if any(r["failures"] or not r["control_rejected"] for r in rows) else 0)


if __name__ == "__main__":
    main()

"""Certificate for scripts/cp_search.py, sharing no code with it.

    python3 -I scripts/cp_cert.py BEFORE AFTER [--out FILE]
    python3 -I scripts/cp_cert.py --secp [--work DIR] [--out FILE]

BEFORE and AFTER are circuits, or uploads whose circuit sits under "circuit". AFTER is certified as BEFORE with
only the order of commuting gates and of CCX controls changed, inside definitions without calls:
- everything outside the definitions' bodies is identical: the rest of the upload, num_wires, registers, the top
  level, the number of definitions and each one's arity;
- a definition with a call in its body is identical;
- in a definition without calls, every wire has the same sequence of role runs before and after, each run holding
  the same gates. A run is a maximal stretch of gates that read the wire as a control ('c') or only XOR into it
  (X, or the target of CX or CCX: 't'); each SWAP on the wire is a run of its own. A gate is X t, CX c t, CCX with
  its two controls sorted and its target, or SWAP w1 w2. Two gates that share only c/c or t/t wires commute, and two
  orders whose run sequences agree on every wire differ only by swaps of such gates, so they are the same
  permutation; a CCX with its controls exchanged is the same gate.
Control: in every changed definition, the first adjacent pair of AFTER's gates that do not commute (they share a
wire on which their roles differ, or a SWAP wire) is swapped, and the certificate must refuse that definition.

With --secp it builds the 256-bit circuit at G as the board's flow does, with scripts/emit.py --hier, then runs
qec/gate_pass.py and scripts/cp_search.py on it, each as its own process, and certifies the searched circuit against
the post-passed one. Prints one JSON line; exits 0 when the certificate holds and every control is refused."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def gate(op):
    """(key, [(wire, role)]) of one primitive."""
    kind = op["op"]
    if kind == "X":
        return ("X", op["target"]), [(op["target"], "t")]
    if kind == "CX":
        return ("CX", op["control"], op["target"]), [(op["control"], "c"), (op["target"], "t")]
    if kind == "CCX":
        a, b = sorted((op["control1"], op["control2"]))
        return ("CCX", a, b, op["target"]), [(op["control1"], "c"), (op["control2"], "c"), (op["target"], "t")]
    if kind == "SWAP":
        return ("SWAP", op["wire1"], op["wire2"]), [(op["wire1"], "s"), (op["wire2"], "s")]
    raise ValueError(f"unexpected op {kind}")


def role_runs(body):
    """wire -> [(role, Counter of gate keys)], each entry one maximal run."""
    per = {}
    for op in body:
        key, wires = gate(op)
        for w, role in wires:
            seq = per.setdefault(w, [])
            if seq and role != "s" and seq[-1][0] == role:
                seq[-1][1][key] += 1
            else:
                seq.append((role, Counter({key: 1})))
    return per


def same_order_class(before, after):
    return len(before) == len(after) and role_runs(before) == role_runs(after)


def conflict(x, y):
    wx, wy = dict(gate(x)[1]), dict(gate(y)[1])
    return any(w in wy and (r != wy[w] or r == "s") for w, r in wx.items())


def circuit_of(doc):
    return doc["circuit"] if "circuit" in doc else doc


def certify(before_doc, after_doc):
    failures, changed, refused, tried = [], [], 0, 0
    b, a = circuit_of(before_doc), circuit_of(after_doc)
    if ("circuit" in before_doc) != ("circuit" in after_doc):
        failures.append("one is an upload and the other a bare circuit")
    elif "circuit" in before_doc:
        outer_b = {k: v for k, v in before_doc.items() if k != "circuit"}
        outer_a = {k: v for k, v in after_doc.items() if k != "circuit"}
        if outer_b != outer_a:
            failures.append("upload fields outside the circuit")
    for k in sorted(set(b) | set(a)):
        if k != "definitions" and b.get(k) != a.get(k):
            failures.append(f"circuit field {k}")
    db, da = b["definitions"], a["definitions"]
    if len(db) != len(da):
        failures.append("definition count")
    for i, (x, y) in enumerate(zip(db, da)):
        if {k: v for k, v in x.items() if k != "operations"} != {k: v for k, v in y.items() if k != "operations"}:
            failures.append(f"definition {i} fields")
            continue
        if x["operations"] == y["operations"]:
            continue
        changed.append(i)
        if any("call" in op for op in x["operations"]) or any("call" in op for op in y["operations"]):
            failures.append(f"definition {i} has calls and changed")
            continue
        if not same_order_class(x["operations"], y["operations"]):
            failures.append(f"definition {i} run sequences")
            continue
        body = list(y["operations"])
        pos = next((j for j in range(len(body) - 1) if conflict(body[j], body[j + 1])), None)
        if pos is not None:
            tried += 1
            body[pos], body[pos + 1] = body[pos + 1], body[pos]
            refused += not same_order_class(x["operations"], body)
    gates = Counter()
    for d in da:
        for op in d["operations"]:
            if "op" in op:
                gates[op["op"]] += 1
    return {"definitions": len(da), "changed_definitions": len(changed), "failures": failures,
            "certified": not failures, "control_tried": tried, "control_refused": refused,
            "definition_gates": dict(sorted(gates.items()))}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(argv, log):
    p = subprocess.run([sys.executable, "-I", *map(str, argv)], cwd=ROOT, capture_output=True, text=True)
    log.write_text(p.stdout + p.stderr)
    if p.returncode:
        raise SystemExit(f"{argv[0]} exited {p.returncode}: {p.stderr[-1000:]}")
    return p.stdout


def secp(work):
    t0 = time.time()
    emitted, post, searched = work / "secp256k1_G.hier.json", work / "post.json", work / "searched.json"
    run(["scripts/emit.py", "--hier", "--out", emitted], work / "emit.log")
    pass_log = json.loads(run(["qec/gate_pass.py", emitted, post], work / "gate_pass.log").splitlines()[-1])
    lines = [json.loads(x) for x in run(["scripts/cp_search.py", post, searched], work / "cp_search.log").splitlines()]
    with open(post) as fh:
        before = json.load(fh)
    with open(searched) as fh:
        after = json.load(fh)
    res = certify(before, after)
    res.update({"instance": "secp256k1 G", "emitted_sha256": sha(emitted), "post_sha256": sha(post),
                "searched_sha256": sha(searched), "gate_pass": pass_log, "search": lines[-1],
                "seconds": round(time.time() - t0, 1)})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", nargs="?")
    ap.add_argument("after", nargs="?")
    ap.add_argument("--secp", action="store_true")
    ap.add_argument("--work", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.secp:
        if args.work:
            args.work.mkdir(parents=True, exist_ok=True)
            res = secp(args.work)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                res = secp(Path(tmp))
    else:
        with open(args.before) as fh:
            before = json.load(fh)
        with open(args.after) as fh:
            after = json.load(fh)
        res = certify(before, after)
        res.update({"before_sha256": sha(args.before), "after_sha256": sha(args.after)})
    res["summary"] = True
    if args.out:
        args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    ok = res["certified"] and res["control_refused"] == res["control_tried"] == res["changed_definitions"]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

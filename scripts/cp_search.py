"""Critical-hop search: a post-pass step after qec/gate_pass.py that reorders commuting gates inside definitions.

    python3 -I scripts/cp_search.py IN.method.json OUT.method.json [--defs D,D,...] [--max-evals N]

IN is a circuit, or an upload whose circuit sits under "circuit", as qec/gate_pass.py writes it; OUT keeps that shape
and is written compact. By default it searches every definition with no calls in its body and at most MAX_OPS ops.
The one larger definition, the square-subtract's, is left as it is: each evaluation re-times the whole body.

For each searched body it times the body standalone forward and inverse under the board's model (X and CX 1 tick,
SWAP 3, CCX target busy 1-10, control2 2-11, control1 4-11, every wire ready at 0), walks each run's critical path,
and finds every hop where a gate waited only for its turn on a wire it shares with the gate before it in the same
role: both read it as a control, or both only XOR into it. Each such pair commutes. Four moves are tried per hop, in
this order: the waiting gate goes just before the other, the other goes just after it, or either CCX takes its
controls the other way round. A move is kept only when the body's forward plus inverse ticks fall, and the walk
restarts on the new path; the search ends when no hop's move helps, or after --max-evals evaluations.

The search reads only the body it changes and has no randomness, so equal bodies give equal results. It keeps every
gate and every gate count; only the order of gates that commute and the order of CCX controls change.
scripts/cp_cert.py checks that, independently of this file. Prints one JSON line per definition and a summary."""
import argparse
import json
import sys
import time
from collections import Counter

CODES = {"X": 0, "CX": 1, "SWAP": 2, "CCX": 3}


def enc(o):
    k = CODES[o["op"]]
    if k == 0:
        return (0, o["target"], -1, -1)
    if k == 1:
        return (1, o["control"], o["target"], -1)
    if k == 2:
        return (2, o["wire1"], o["wire2"], -1)
    return (3, o["control1"], o["control2"], o["target"])


def dec(t):
    k, a, b, c = t
    if k == 0:
        return {"op": "X", "target": a}
    if k == 1:
        return {"op": "CX", "control": a, "target": b}
    if k == 2:
        return {"op": "SWAP", "wire1": a, "wire2": b}
    return {"op": "CCX", "control1": a, "control2": b, "target": c}


def roles(t):
    k, a, b, c = t
    if k == 0:
        return ((a, "t"),)
    if k == 1:
        return ((a, "c"), (b, "t"))
    if k == 2:
        return ((a, "e"), (b, "e"))
    return ((a, "c"), (b, "c"), (c, "t"))


def timed(body, n):
    """(makespan, pred, crit) of body run in order: pred[i] the position of the gate that last set the wire
    deciding gate i's start, crit[i] that wire; and the position of the gate ending last."""
    r = [0] * n
    last = [-1] * n
    L = len(body)
    pred = [-1] * L
    crit = [-1] * L
    endp = [0] * L
    for i, (k, a, b, c) in enumerate(body):
        if k == 3:
            ra, rb, rt = r[a], r[b], r[c]
            t2 = (rb if rb > rt + 1 else rt + 1) + 1
            t4 = (ra if ra > t2 + 1 else t2 + 1) + 1
            w = a if ra + 1 >= t2 + 2 else (b if rb >= rt + 1 else c)
            pred[i], crit[i] = last[w], w
            r[c] = t4 + 6
            r[a] = r[b] = t4 + 7
            endp[i] = t4 + 7
            last[a] = last[b] = last[c] = i
        elif k == 0:
            pred[i], crit[i] = last[a], a
            r[a] += 1
            endp[i] = r[a]
            last[a] = i
        else:
            w = a if r[a] >= r[b] else b
            pred[i], crit[i] = last[w], w
            v = (r[a] if r[a] > r[b] else r[b]) + (1 if k == 1 else 3)
            r[a] = r[b] = v
            endp[i] = v
            last[a] = last[b] = i
    D = max(r) if r else 0
    tail = max(range(L), key=lambda i: endp[i]) if L else -1
    return D, pred, crit, tail


def cost(body, n):
    return timed(body, n)[0] + timed(body[::-1], n)[0]


def role_on(t, w):
    for x, ro in roles(t):
        if x == w:
            return ro
    return None


def commutes(s, t):
    rs = dict(roles(s))
    for w, ro in roles(t):
        q = rs.get(w)
        if q is not None and (q != ro or q == "e"):
            return False
    return True


def hops(body, n):
    """(p, o, w) along the critical path of body run in order, where o waited on wire w for p in the same role."""
    _, pred, crit, o = timed(body, n)
    out = []
    while o >= 0:
        p, w = pred[o], crit[o]
        if p >= 0:
            ro, rp = role_on(body[o], w), role_on(body[p], w)
            if ro == rp and ro in ("c", "t"):
                out.append((p, o, w))
        o = p
    return out


def move_before(body, p, o):
    """o to just before p, if o commutes with every gate from p up to it; else None."""
    t = body[o]
    if not all(commutes(body[j], t) for j in range(p, o)):
        return None
    return body[:p] + [t] + body[p:o] + body[o + 1:]


def move_after(body, p, o):
    """p to just after o, if p commutes with every gate after it up to o; else None."""
    t = body[p]
    if not all(commutes(body[j], t) for j in range(p + 1, o + 1)):
        return None
    return body[:p] + body[p + 1:o + 1] + [t] + body[o + 1:]


def flip(body, i):
    k, a, b, c = body[i]
    if k != 3:
        return None
    return body[:i] + [(3, b, a, c)] + body[i + 1:]


def runs(body):
    """Per wire, its sequence of (role, multiset of gates with CCX controls sorted)."""
    seq = {}
    for t in body:
        k, a, b, c = t
        key = (3, min(a, b), max(a, b), c) if k == 3 else t
        for w, ro in roles(t):
            lst = seq.setdefault(w, [])
            if lst and ro != "e" and lst[-1][0] == ro:
                lst[-1][1][key] += 1
            else:
                lst.append((ro, Counter({key: 1})))
    return seq


def search(body, n, max_evals):
    best = cost(body, n)
    start, evals, kept = best, 0, Counter()
    improved = True
    while improved and evals < max_evals:
        improved = False
        for inv in (False, True):
            view = body[::-1] if inv else body
            for p, o, _ in hops(view, n):
                for kind in ("before", "after", "flip_o", "flip_p"):
                    if kind == "before":
                        new = move_before(view, p, o)
                    elif kind == "after":
                        new = move_after(view, p, o)
                    else:
                        new = flip(view, o if kind == "flip_o" else p)
                    if new is None:
                        continue
                    cand = new[::-1] if inv else new
                    evals += 1
                    c = cost(cand, n)
                    if c < best:
                        best, body, improved = c, cand, True
                        kept[kind + ("/inv" if inv else "/fwd")] += 1
                        break
                if improved or evals >= max_evals:
                    break
            if improved or evals >= max_evals:
                break
    return body, start, best, evals, kept


MAX_OPS = 50_000


def flat_defs(defs):
    return [d for d, x in enumerate(defs)
            if x["operations"] and not any("call" in o for o in x["operations"]) and len(x["operations"]) <= MAX_OPS]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--defs", help="comma-separated definitions to search instead of the default set")
    ap.add_argument("--max-evals", type=int, default=20000)
    args = ap.parse_args()
    t_all = time.time()
    with open(args.src) as fh:
        doc = json.load(fh)
    circ = doc.get("circuit", doc)
    defs = circ["definitions"]
    chosen = [int(x) for x in args.defs.split(",")] if args.defs else flat_defs(defs)
    total, changed, before, after = Counter(), 0, 0, 0
    for d in chosen:
        t0 = time.time()
        ops = defs[d]["operations"]
        assert not any("call" in o for o in ops), f"definition {d} has calls"
        n = defs[d]["arity"]
        body = [enc(o) for o in ops]
        new, c0, c1, evals, kept = search(body, n, args.max_evals)
        assert runs(new) == runs(body), f"run sequences changed on definition {d}"
        f0, f1 = timed(body, n)[0], timed(new, n)[0]
        i0, i1 = timed(body[::-1], n)[0], timed(new[::-1], n)[0]
        if new != body:
            defs[d]["operations"] = [dec(t) for t in new]
            changed += 1
        before, after = before + c0, after + c1
        total.update(kept)
        print(json.dumps({"def": d, "ops": len(body), "fwd": [f0, f1], "inv": [i0, i1], "sum": [c0, c1],
                          "evals": evals, "kept": dict(kept), "seconds": round(time.time() - t0, 1)}), flush=True)
    with open(args.dst, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    print(json.dumps({"summary": True, "searched": len(chosen), "changed": changed, "standalone": [before, after],
                      "kept": dict(total), "seconds": round(time.time() - t_all, 1)}), flush=True)


if __name__ == "__main__":
    main()

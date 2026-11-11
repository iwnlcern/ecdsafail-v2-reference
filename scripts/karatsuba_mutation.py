"""Negative control for the Karatsuba square-subtract: its checks must reject single-gate mutations.

At q = 65,521 with the size threshold forced, the class's standalone circuit is built flat.
Each trial demotes one CCX to a CX, as scripts/mutation_control.py does for the point add,
and judges the mutant on the two exhaustive-in-Y populations of scripts/karatsuba_check.py:
every Y with gen in {0, 1}, and every X for each Y that reaches the canonical tail.

A demoted CCX(a, b, t) differs from the original only on states where a = 1 and b = 0 at that
gate. A gate inside a group's build or unbuild sees the state of that group's recorded block,
whose inputs are Y and zeros, and the unbuild retraces the build, so its controls take the
values they take at the matching build gate. Simulating the block on every Y therefore
decides whether the mutant equals the circuit on every input at this prime. The constant add
of K that ends the circuit sees gen, X in [0, q) and zeros, and touches no Y wire, so simulating
it from its first op on every (gen, X) decides a gate there the same way. A survivor proved
equal either way is classified as equivalent. Any other survivor is reported as unclassified.

Prints one JSON line; exits 1 only if the unmutated circuit fails.
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from karatsuba_check import CONTROL_PRIME, Harness, all_y_lanes, check_lanes, check_tail, planes, prepare, tail_population
from qec.arith import Mod
from qec.builder import CCX, CX, Circuit
from qec.sim import run
from qec.square import KaratsubaMulsub

SEED = 7
TRIALS = 200


class Marking(Circuit):
    """Records where each top-level block application lands in the flat op stream."""

    def __init__(self):
        super().__init__()
        self.marks = []

    def apply(self, block, regs, inverse=False):
        start = len(self.prims) // 4
        super().apply(block, regs, inverse)
        self.marks.append((start, len(self.prims) // 4, len(block.ops) // 4, inverse))


class Located(KaratsubaMulsub):
    """Records each group's canonical tail as an op range inside that group's build block."""

    tails = []

    def canon(self, C, R):
        start = len(C.prims) // 4
        super().canon(C, R)
        Located.tails.append((start, len(C.prims) // 4))


def harness(q):
    c, mod = Marking(), Mod(q)
    (gen,) = c.alloc(1)
    x, y = c.alloc(mod.n), c.alloc(mod.n)
    Located.tails = []
    Located(mod).mulsub(c, gen, y, x)
    return Harness(q, c.ops, c.num_wires, gen, x, y, 0), c.marks, list(Located.tails)


def regions(marks, tails, total):
    """(start, end, label) for every op range of the flat stream."""
    assert len(marks) == 6 and len(tails) == 2, "build, modadd and unbuild for each of AC and B"
    out = []
    for g, group in enumerate(("AC", "B")):
        (b0, b1, length, _), (m0, m1, _, _), (u0, u1, _, _) = marks[3 * g:3 * g + 3]
        t0, t1 = tails[g]
        out += [(b0, b0 + t0, f"{group} build"), (b0 + t0, b0 + t1, f"{group} build tail"), (b0 + t1, b1, f"{group} build"),
                (m0, m1, f"{group} modadd"),
                (u0, u0 + length - t1, f"{group} unbuild"), (u0 + length - t1, u0 + length - t0, f"{group} unbuild tail"),
                (u0 + length - t0, u1, f"{group} unbuild")]
    out.append((marks[-1][1], total, "K add"))
    return out


def group_blocks(q):
    """Each group's recorded build block, on local wires whose first n are Y."""
    square = KaratsubaMulsub(Mod(q))
    return [square._record_build(build, square.n)[0] for build in (square.build_ac, square.build_b)]


def classify(h, i, marks, blocks):
    """Why a demotion of op i survives, or None if this test cannot say."""
    k0 = marks[-1][1]
    if i >= k0:
        q, n = h.q, len(h.x)
        it = iter(h.ops[4 * k0:])
        assert not {a for op, *ws in zip(it, it, it, it) for a in ws} & set(h.y), "the K add touches Y"
        lanes = [(g, x) for g in (0, 1) for x in range(q)]
        init = {h.gen: planes([g for g, _ in lanes], 1)[0], **dict(zip(h.x, planes([x for _, x in lanes], n)))}
        _, a, b, _ = h.ops[4 * i:4 * i + 4]
        state = run(h.ops[4 * k0:4 * i], h.num_wires, init, len(lanes))
        if state[a] & ~state[b]:
            return None
        return "equivalent: the controls are never (1, 0) on any (gen, X) in the K add"
    for g, block in enumerate(blocks):
        (b0, b1, length, _), _, (u0, u1, _, _) = marks[3 * g:3 * g + 3]
        if b0 <= i < b1:
            k = i - b0
        elif u0 <= i < u1:
            k = length - 1 - (i - u0)
        else:
            continue
        op, a, b, _ = block.ops[4 * k:4 * k + 4]
        assert op == CCX == h.ops[4 * i]
        q, n = h.q, len(h.y)
        state = run(block.ops[:4 * k], max(block.wires) + 1, dict(enumerate(planes(list(range(q)), n))), q)
        if state[a] & ~state[b]:
            return None
        return "equivalent: the controls are never (1, 0) on any Y in the group's block"
    return None


def main():
    start = time.time()
    q = CONTROL_PRIME
    h, marks, tails = harness(q)
    spans = regions(marks, tails, len(h.ops) // 4)
    blocks = group_blocks(q)
    lanes = prepare(h, all_y_lanes(q))
    ys, _ = tail_population(q)
    head = {"q": q, "ops": len(h.ops) // 4, "lanes": lanes[-1], "tail_pairs": len(ys) * q}
    base = [check_lanes(h, lanes), check_tail(h, ys)]
    if any(r["wrong"] or r["dirty"] for r in base):
        print(json.dumps({**head, "base": base}))
        sys.exit(1)
    rng = random.Random(SEED)
    ccx = [i for i in range(len(h.ops) // 4) if h.ops[4 * i] == CCX]
    caught, survivors, refused, refused_k = 0, [], [], []
    for i in rng.sample(ccx, TRIALS):
        ops = h.ops[:]
        ops[4 * i] = CX
        ops[4 * i + 2] = ops[4 * i + 3]
        r = check_lanes(h, lanes, ops)
        if not (r["wrong"] or r["dirty"]):
            r = check_tail(h, ys, ops)
        if r["wrong"] or r["dirty"]:
            caught += 1
            if len(refused) < 3 and any(lo <= i < hi for lo, hi, _, _ in marks[0::3] + marks[2::3]):
                # Control: the classifier must not call a caught build or unbuild mutant equivalent.
                refused.append(classify(h, i, marks, blocks) is None)
            elif len(refused_k) < 3 and i >= marks[-1][1]:
                # The same control for the K add's branch.
                refused_k.append(classify(h, i, marks, blocks) is None)
            continue
        region = next(label for s, e, label in spans if s <= i < e)
        survivors.append({"op": i, "region": region, "class": classify(h, i, marks, blocks)})
    unclassified = sum(s["class"] is None for s in survivors)
    print(json.dumps({**head, "mutation": "one CCX demoted to CX", "seed": SEED, "ccx": len(ccx), "trials": TRIALS,
                      "caught": caught, "survivors": survivors, "unclassified": unclassified,
                      "classifier_control": len(refused) == 3 and all(refused) and len(refused_k) == 3 and all(refused_k),
                      "seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()

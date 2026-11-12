# Exact controlled point addition for ecdsafail-v2

## Opt-in Jump-4 multiplier

`QEC_JUMP4=1` selects the raw cap-four multiplier; the unset/default `0`
keeps the original PR #32 circuit and submitted-method bytes. The flag is
strict and captured when a multiplier is constructed. Enabled `split()`
returns false, so the unchanged point-add kernel uses its unsplit branch;
non-None legacy cut inputs are rejected.

The refined 320-cell secp schedule (sum W 49,280) uses raw four-bit
cells, a three-bit initial-strip header and live offset E. Eager high
words apply at gaps of at least three bits, with explicit same/drop flag history,
and retained 1,933 majority carries allocated from the last step backward.
The nominal 2,236-wire extra-junk envelope includes E and 302 flag/history
wires. All history survives the full coefficient reader and is cleared only
by the same recorded producer inverse. Only unused clean tape or unconsumed
junk is high-word scratch. Coefficients remain full canonical words.

The coefficient reader derives quiet spans from both existing selector masks and conservatively
requires zero addends for every K=0..16 in both chains, including the header.
At secp the shared span is bits 15..31. One Prefix (ours, from PR #2) survives the
read-only chain and is written/restored only in the second chain. Only the
first reverse cell uses S=0 with terminal code 7 or padding code 3: R stays
unchanged and S XOR=g*R. Later cells retain the full canonical map.

Measured at secp G: 2,945,864 raw CCX, 20,621,048 T, 4,096 construction wires.
The partition is four Euclid traversals at 265,282 each, two integrated
coefficient traversals at 897,366 each, square 83,828 and glue 6,176.
Before eager high words the circuit measured 2,996,796 CCX; with them it
measured 2,961,240. Quiet-run and first-cell specialization remove 15,376
and bring it 15,144 below PR #32's 2,961,008. The 2G kernel costs 2,945,884 CCX.
The unsplit fallback's glue is measured; no lost cut saving is inherited.
Independent immutable-source QA remains separate from builder checks.
No full gates, scored artifact or qualification are implied.

The same-source eager producer with zero retained carries costs 267,215 CCX.
Budgets 1, 967 and 1,933 cost 267,214, 266,248 and 265,282 respectively:
the measured marginal saving is one CCX per carry per traversal. Independently,
the eager high word itself saves 6,956 CCX per traversal.

The measured reader inventory is raw tape 1,280 + canonical R/S 512 + H 3
+ E/flag/carry history 2,236 + three outside live wires = 4,034, with 49
internal coefficient wires. The actual whole construction peak is 4,096;
the inventory alone is not substituted for that measurement. The producer
block has 3,775 input wires and four internal wires. High-word scratch peaks
at 53 clean loans, sourced only from future tape or unconsumed junk.

QA-only forced HIGH_MIN1 exercises 13/31/127 and larger joint shapes; the
extra q=3 forced shape explicitly refuses its three-wire scratch request
when only two future wires exist. Default production q=3 remains supported.

Credit jtaroreh's PR #32 (https://github.com/jtaroreh/lowt-climb) for the base this builds on, including its eager high word, with the credits PR #32 gives; ab-godder's arith.add with a one-CCX top bit, from PR #9 and PR #28; and our PR #2 (source c39f210) for the shrinking tape pool, the held offset E and Prefix. The Jump-4 step extends the Jump-2 Euclidean algorithm in §5.3.1 of the ECDSA.Fail paper (arXiv:2609.09582) to strip up to four bits; its cap, coefficient cell, selectors and quiet-run specialization are ours.

## Start here

This repository holds the reference circuit for the ecdsafail-v2 controlled point addition, an exact and complete |c>|R>|0> -> |c>|R + c*A>|0> on secp256k1. Fork it to make the circuit cheaper. The benchmark that will judge your change is not published yet, so `scripts/accept.py` checks the invariants a change must keep.

```sh
python3 scripts/accept.py --quick   # about 80 s of CPU, stdlib Python only
python3 scripts/accept.py           # about 600 s of CPU, run it before you hand a change back
```

The checkers run in parallel, so wall time depends on your cores and load. On an Apple M4 with 10 cores at load average 6 to 15, quick mode took 17 s and full mode 103 s.

- `INVARIANTS.md` says what each check protects, what no script can check, where the CCX are, and how to hand a change back.
- `.claude/skills/verify-reference-circuit/` is a Claude Code skill that drives the generator, the checks, the cost report and the limits, with a feature map under `features/`.

## What this circuit is

This directory holds a generator for the arithmetic kernel that the `secp256k1-controlled-exact-v1` profile asks for: |c>|R>|0> -> |c>|R + c*A>|0> for a fixed public point A. It covers every R on secp256k1, including infinity and R in {A, -A, -2A}. It is built from the exact core of Schrottenloher's dialog Euclid and Bezout multiplier (arXiv 2606.02235). It is emitted in the challenge's own `ArithmeticCircuit` JSON and checked with the challenge's own executor.

**It does not pass the challenge at 256 bits, as the benchmark is published today.** The regenerated circuit is 1.34x the published operation cap. The retained reference SubmittedMethod was 7.84x the upload cap; its executor and service receipts were not rerun for this change. No route to qualification exists in the repository today in any case. See "Unresolved gates" below.

`DESIGN.md` records the contract reading, the three designs compared, the data shape, and the completeness argument.

## Run

```sh
scripts/run_all.sh                                # about 2 minutes; rewrites everything in receipts/ and artifacts/
ECDSAFAIL_V2=../ecdsafail-v2 scripts/run_all.sh   # also builds and runs the harness against the benchmark checkout
```

Without `ECDSAFAIL_V2`, `run_all.sh` skips the harness steps and says so, and the harness receipts keep their committed contents.

Individual steps:

```sh
PY=python3
$PY -I scripts/model_check.py                                  # classical dialog model: Euclid trace exhaustive at small q, multiplier sampled
$PY -I scripts/check.py --toy-bits 4 6 7 8 9 10 --toy-per-size 6   # gate-level exhaustive check on toy curves
$PY -I scripts/mutation_control.py                             # negative control for the checker
$PY -I scripts/component_check.py                              # 256-bit components on boundary inputs
$PY -I scripts/check.py --secp --secp-random 120               # 256-bit full circuit, sampled
$PY -I scripts/emit.py --out artifacts/secp256k1_G_circuit.json.gz
gzip -dc artifacts/secp256k1_G_circuit.json.gz | harness/target/release/qec-harness --circuit - --cases artifacts/secp256k1_cases.json
```

Python is stdlib only. The harness is a Rust crate whose dependencies are path references into a sibling checkout of the benchmark at `../ecdsafail-v2`, next to this repository. That checkout is not public until the benchmark is published. It builds offline into `harness/target` and writes nothing into the sibling repo. Receipts were produced against sibling `HEAD 9152f4d` with its existing uncommitted diff (sha256 prefix of `git diff HEAD` is `3eaa988f0901d0d1`).

## Layout

- `qec/model.py` is the classical integer model of the multiplier, the proven iteration bound, and the Euclid width schedule.
- `qec/builder.py` holds the circuit, allocator, and recordable blocks that replay forward or reversed under wire remapping.
- `qec/arith.py` holds exact CDKM add, subtract and compare, each optionally controlled, plus modular add, double and negate.
- `qec/point_add.py` holds the dialog multiplier, the gen-controlled kernel, and the four-case completeness wrapper.
- `qec/sim.py` is an independent bit-sliced simulator. `qec/curve.py` is a reference curve law. `qec/emit.py` streams challenge-format JSON.
- `scripts/accept.py` is the acceptance gate. `INVARIANTS.md` explains each of its checks.
- `harness/` runs an emitted circuit through `v2_ir` parsing and validation, `v2_runtime::BitSlicedExecutor`, a `SubmittedMethod` round trip, `QualificationGraph::evaluate`, and `v2_lowering`.
- `artifacts/` holds the emitted circuits, test lanes, and the toy `SubmittedMethod`. `receipts/` holds every measured result quoted below.

## Results

All numbers are measured unless marked as estimates.

**This branch.**
At G the circuit is 1,246 wires, 3,299,923 CCX, 23,099,461 T and 12,138,599 ops, measured from the unchanged emitter and baseline script.
Wires × T is 28,781,928,406, which is 75.32% below the reference's 2,311 × 50,465,933.
The hierarchical JSON for G is 39,150,695 bytes, under the 100 MiB upload cap; the flat circuit is still over the 10,000,000 operation cap.
Two named limits apply.
`I2.model`'s zero-caller row is narrowed from the reference's gate.
The standalone multiplier is restricted to x in [1, q), and nonzero x at its entries rests on the kernel's flag code and the non-full build refusal.
`DESIGN.md` and `INVARIANTS.md` give the constructions and their arguments.
The table and paragraphs below still give the earlier shared-chain build's figures, and so do the files in `receipts/` and `artifacts/` other than `receipts/baseline/`; they were not regenerated for this branch.

| | toy q=127 | secp256k1, A = G | cap |
|---|---|---|---|
| wires | 60 | 1,803 | 4,096 |
| ops | 11,971 | 13,378,277 | 10,000,000 |
| CCX | 4,197 | 4,908,983 | |
| T + T-dagger (7 x regenerated CCX) | 29,379 | 34,362,881 | |
| measurements, R or Hmr | 0 | 0 | 0 |
| SubmittedMethod JSON bytes | 583,085 | 822,274,918 | 104,857,600 |
| `v2_ir` validator, service caps | PASS | REJECT, ops | |
| challenge executor lanes | 254 of 254 correct (exhaustive) | 510 of 510 correct (sampled) | |
| dirty scratch or phase lanes | 0 | 0 | |
| `is_qualified` | false | false | |

The SubmittedMethod byte counts, validator, executor lanes, dirty/phase lanes and qualification rows above retain the harness receipts measured on the reference at `e3bf0c0`; the optional harness steps were skipped. Ratios derived from those harness values, including the 7.84x reference upload ratio, are reference-only. The regenerated flat circuit JSON is 467,063 bytes for toy q=127 and 568,986,741 bytes for G; these are circuit files, not new SubmittedMethod measurements.

This branch builds on the stack’s three changes. One controlled comparison per Euclid step removes 787,456 CCX and 5,512,192 T at G. The exact merged Bézout reduction removes another 731,136 CCX and 5,117,952 T, with peak wires rising from 2,311 to 2,313. Reusing zero Euclid words and reversibly remapping x = 0 adds 1,524 CCX and 10,668 T while reducing the combined peak to 1,803 wires. The shared-chain Euclid step now holds the subtraction offset in the preceding tape cell and clears even-cell copies with a descending sweep. Relative to that stack it removes another 783,368 CCX and 5,483,576 T, with the same 1,803-wire peak. These CCX and T deltas are additive; wire counts are peaks. The bound and width schedule are unchanged, and `DESIGN.md` gives the held-offset, sweep, merged reduction and zero-remap arguments.

Full `scripts/accept.py` on this source reports 13 pass, 0 fail and 1 pending (`I7.charge`), including R1. With seed 2026, [1]G through [8]G plus 16 random addends, the sample maximum is 34,362,937 T (16,103,052 below the reference sample maximum 50,465,989); the sample minimum is 34,362,573 T. This sample maximum is not a bound over every addend. All 263 definitions remain identical across addends. EuclidStep changes to a held-offset relation with first/later variants, and the tape sweep needs a new contract. EuclidPass retains its external relation but needs a new composition proof; BezoutStep is unchanged. DialogMul still needs the previously identified composition support. The standalone leaf probe establishes E1 for the 257 actual step definitions under its preconditions; it does not establish checker acceptance, sweep composition or DialogMul. Passing this repository's gate does not establish checker acceptance or qualification.

Where the 256-bit CCX go (`receipts/secp256k1_breakdown.json`). One dialog multiplier is 1,990,646 CCX: two Euclid passes of 395,262 each, one Bézout pass of 1,199,104, and 1,018 for the remaining zero-remap and cleanup operations. The old controlled word swap is removed; the word swap is now unconditional. The kernel uses the multiplier twice. The lambda-squared subtraction is 910,080 CCX. Flags and constant adds make up the remaining 17,611.

The controlled subtract and the controlled modular add run a controlled CDKM ripple instead of copying `ctrl*a` into an AND register. A controlled subtract costs 3w CCX instead of 4w, and a controlled modular add went from 10n + 2 to 9n + 4. That took the 256-bit point addition from 8,686,084 to 7,967,236 CCX, 8.3% fewer.

Classical constants are no longer copied into a register for a full adder. The carry chain reads the constant's bits, so it skips carries that are always 0 and aliases a carry that equals an input bit. Each modular reduction runs one carry chain of s + 2^(n+1) - q. Its carry out is [s >= q]. It lands in the spare top bit, because the top bit and the carry into it are never both 1 when s < 2q. The same carries then drive the conditional subtraction. The reduction costs 3n - 3 CCX instead of 4n + 2, so a controlled modular add now costs 8n - 1, which is 2,047 at n = 256. A modular double costs 3n - 15 instead of 4n + 4 at secp256k1, because 2s has a zero low bit and the next three bits of 2^(n+1) - q are 0, so the first carries are always 0. Its flag is the low bit of the result, so the flag wire becomes that bit with no gate. Negation adds the sparse 2^n - q - 1 and then complements. That took the point addition from 7,967,236 to 7,209,419 CCX, 9.5% fewer, and from 26,931,713 to 18,995,461 ops. The carry chain still ripples the full width, because exactness needs it.

Exactness evidence, strongest first:

1. **Exhaustive gate-level check.** `receipts/toy_exhaustive.json` covers 36 instances: 18 primes of 4, 6, 7, 8, 9 and 10 bits, 2 addends each, every curve point plus infinity, and both control values. It finds 0 failures, with all scratch returned to zero. The same generator code emits the 256-bit circuit. Only the constants and the bound-derived width schedule differ.
2. **Challenge executor (retained reference evidence).** `receipts/harness_toy127.json` and `receipts/harness_secp256k1.json` replay the emitted JSON on `v2_runtime::BitSlicedExecutor`. Expected values come from `qec/curve.py`, which shares no code with the generator.
3. **256-bit boundary components.** `receipts/secp256k1_components.json` covers modadd with and without ctrl on edge-by-edge pairs around q and 2^255 and at q - f and 2^256 - 1 mod q, where f = 2^256 - q, plus moddbl on the same values and cneg. It runs cmodadd_const and gt_const_into with ctrl 0 and 1, gt_const_into without ctrl, and add_const under ctrl with a carry out, each on the kernel's constants for G (q - a, q - b, 3a and a) and on sparse and dense edge constants (1, f, 2^255, q - 1, q - 2, 2^255 - 1 and 2^256 - 1 mod q), with inputs on both sides of each constant. It also runs the multiplier on x = 0 and on x values whose Euclid run takes 511 of the proven 512 iterations. All pass.
4. **Proven iteration bound.** u*v at least halves per active Euclid step, so bitlen(q(q-1)) iterations always suffice. That is 512 at secp256k1. `receipts/model_check.jsonl` shows the observed worst case never exceeds the bound at any of the 13 tested q. It equals the bound at q = 3, 5, 11 and 43 and is one below it elsewhere.
5. **Checker sensitivity.** A must-fail control (an extra X on the output) fails every lane. Exhaustively, 176 of 200 random CCX-to-CX demotions are caught, and the survivors are correct on the whole promise domain, so they are equivalent circuits. Most CCX in constant adders and idle Euclid tails fire only on rare carries, so random 256-bit sampling cannot separate an exact circuit from an approximate one, which is why items 1, 3 and 4 carry the exactness claim.

The circuit is pure X, CX, CCX and SWAP, so it is a phase-free permutation. Basis exactness with clean scratch on the promise domain therefore gives the exact unitary on superpositions and entangled references (QNT-04). Outside the promise it is still a permutation, which satisfies QNT-06.

## Unresolved gates

This section describes the benchmark as published. Its next release accepts a hierarchical upload of definitions and calls. The hierarchical JSON for G is 25,891,033 bytes, which `scripts/accept.py` checks against the 100 MiB upload cap. How that release charges a hierarchical upload is pending its publication.

These are listed in the order they would block a 256-bit submission on the published benchmark.

1. **Operation cap.** 13,378,277 ops against a cap of 10,000,000 (`MAX_DECLARED_RECORDS`). One multiplier alone is 5,876,708 ops. The paper reports about 2^24 total gates per addition even for its approximate circuits, so no published point addition fits this cap.
2. **Upload cap.** The service parses plain JSON, at about 43 bytes per op. 100 MiB therefore holds about 2.4M ops, roughly 5.5x fewer than this circuit needs. This also binds before the operation cap does.
3. **Whole-attack T budget (estimate).** DEF-02's binary wrapper makes 512 controlled additions, which comes to about 1.76e10 T against the 3.25e9 cap. A w = 16 windowed interface (QNT-02) would need about 32 additions plus exact lookups, roughly 1.49e9 T (32 × 34,362,881 plus about 0.39e9 for lookups). That would fit, but it means a coherent-addend kernel, which is not built here.
4. **Qualification route.** `QualificationGraph` requires a universal secp256k1 certificate checked by a `SoundTheoremChecker`, a Shor wrapper certificate, a schedule, observations, and a retry model. The production registry holds no sound checker, so every artifact ends `is_qualified: false` with `No sound theorem checker exists`. This candidate supplies none of those artifacts.
5. **Formal proof.** Exactness rests on exhaustive toy verification of a size-parametric generator, full-size boundary tests, and the proven iteration bound. That is strong evidence, not a machine-checked proof.

## Optimization headroom (estimates, not built)

These estimates are against the published benchmark's caps.

- The dialog garbage compression from the paper's Fig. 1 saves qubits, not ops.
- The exact equivalent of the paper's pseudo-Mersenne reduction still needs a full-width carry, so that saving is smaller here than in the paper.
- A further 2x reduction would bring the ops under the operation cap, but the flat file would stay about 3.1x over the upload cap.

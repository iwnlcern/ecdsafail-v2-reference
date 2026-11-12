# Exact controlled point addition for ecdsafail-v2

## Start here

This repository holds the reference circuit for the ecdsafail-v2 controlled point addition, an exact and complete |c>|R>|0> -> |c>|R + c*A>|0> on secp256k1. Fork it to make the circuit cheaper. A live board tests uploads, and its benchmark source is not public yet, so `scripts/accept.py` checks the invariants a change must keep.

```sh
python3 scripts/accept.py --quick   # about 80 s of CPU, stdlib Python only
python3 scripts/accept.py           # about 600 s of CPU, run it before you hand a change back
```

The checkers run in parallel, so wall time depends on your cores and load. On an Apple M4 with 10 cores at load average 6 to 15, quick mode took 17 s and full mode 103 s.

- `INVARIANTS.md` says what each check protects, what no script can check, where the CCX are, and how to hand a change back.
- `.claude/skills/verify-reference-circuit/` is a Claude Code skill that drives the generator, the checks, the cost report and the limits, with a feature map under `features/`.

## Result on the live board

The board tested this circuit on 2026-11-06 as `reference-c4`, evaluation `eval-6d612fd4c6156cdb`, in round 2. It ended `TESTED_UNQUALIFIED`, with 0 of 10 boundary cases and 0 of 10,000 seeded draws failing. That is a sampled pass and not a qualification. Its raw costs are the line to beat.

| | Reference | Board limit |
|---|---|---|
| `lowered_t` | 50,465,933 | |
| `qubit_time` | 164,135,627,418 | |
| `toffoli` | 7,209,419 | |
| `allocated_wires` | 2,311 | 4,096 |
| `scheduled_ticks` | 71,023,638 | |
| `expanded_ops` charge | 20,960,620 | 21,696,718 |
| encoded records | 974,699 | 10,000,000 |
| listed call wires | 1,055,748 | 10,000,000 |
| upload bytes | 43,423,435 | 104,857,600 |

The charge leaves 736,098 of room, about 3.4% of the limit, and a call costs about 64 of it.

To start from the circuit on the board, emit the hierarchical file. The flat file is 822,274,918 bytes, which is over the upload limit.

```sh
python3 -I -B scripts/emit.py --hier --out my-circuit.json
```

The SHA-256 of `my-circuit.json` is `01eab537d5cd4c81680c17d94f0e664f5b616fc9e4818dc15937c839cdbfee00`. The upload wraps those exact bytes in a `SubmittedMethod` with `public_key` set to G. The top-level `artifact_hash` in the board's receipt is the SHA-256 of that upload, `666d5595e59e0755f91acf327cac07884e0fc58226d6abe8ee9b0845c7018457`. Another wrapper, such as one with a different `method_id`, changes that hash and leaves the circuit unchanged, so compare `my-circuit.json` to check your starting point.

That hash is the reference generator's. This branch's generator emits a different circuit, described next.

## This branch

This branch starts from ab-godder's PR #29 source at `61e0c50` and uses its source, passes and circuit-level post-pass. It adds our constant-tree XOR-sum filter, parallel carry tree for the coefficient update (`KPAR`), backward-wave carry network (`WAVE_BACK_ONE`) and parallel folded squaring (`SQ_PAR`). The secp256k1 width cut is ab-godder's PR #28 (`e1b32bb`, merged as `5ca6779`). `DESIGN.md` describes these changes and their controls.

| | #29 entered | This branch emitted | This branch post-passed | Board limit |
|---|---:|---:|---:|---:|
| wires | 1,546 | 1,546 | 1,546 | 4,096 |
| scheduled ticks | 1,920,902 | 1,928,261 | 1,856,756 | |
| qubit-time | 2,969,714,492 | 2,981,091,506 | 2,870,544,776 | |
| CCX | 7,208,765 | 7,188,435 | 7,187,137 | |
| T | 50,461,355 | 50,319,045 | 50,309,959 | 50,465,933 |
| expanded-ops charge | 19,201,248 | 19,673,384 | 19,144,695 | 21,696,718 |
| JSON bytes | 76,683,894 | 69,643,749 | 75,309,234 (wrapped method) | 104,857,600 |

These are local measurements. Post-passed qubit-time is 3.34% below #29's entered figure. T is 155,974 below the line. The gate pass takes 1,928,261 ticks to 1,910,922 and the reorder pass takes that to 1,856,756.

Every figure carries two limits: I2.model's zero-caller row is narrower than the reference gate; standalone M requires x in [1,q), and nonzero inputs at its kernel entries rely on flag code. The component check explicitly excludes the six x=0 lanes that fail in ab-godder's PR #29 source at `61e0c50`; it covers 49 components and 8,945 lanes instead of 8,951. This does not establish a physical zero-input convention or a universal nonzero-entry proof.

A further inherited source limit is that `scripts/check_hier.py` is red at `61e0c50`: the hierarchical and flat builds order and label gates differently, so its comparison against flat receipts fails. Receipts stay unchanged. Both builds are checked separately against the specification on every promised toy input; the full acceptance gates additionally check the hierarchical upload and I2.secp-flat. `I7.charge` remains a pending gate row; the measured release charge is separate evidence. These local checks do not establish universal qualification.

The scored circuit is the hierarchical emission after #29's deterministic post-pass:

```sh
python3.12 -I -B scripts/emit.py --hier --out x.circuit.json
python3.12 -I -B scripts/post_pass.py x.circuit.json x-pp.json --jobs 2 --rounds 300
python3 wrap.py x-pp.json x.method.json
```

`wrap.py` is the reference's SubmittedMethod wrapper from release `qcb-8376098` of jtaroreh/ecdsafail-v2-reference. The locally measured SHA-256 values are:

```text
1feca2c1ea0c67e274bd5320c702a0d91373daf9c9824e6de25b5682ee13a609  x.circuit.json
5649f4139a389cf7ae03b53aff0877e7c81592993d74c32686d299850e9e4d02  x-pp.json
6ad4f26ff1233d48f5b4acebbefc32789d955f5410509a28b1401feeb75f93e3  x.method.json
```

The method hash depends on the wrapper; the circuit pins are the first two hashes. Independent finite tests and final runtime checks are distinct from a board entry or qualification.

## What this circuit is

This directory holds a generator for the arithmetic kernel that the `secp256k1-controlled-exact-v1` profile asks for: |c>|R>|0> -> |c>|R + c*A>|0> for a fixed public point A. It covers every R on secp256k1, including infinity and R in {A, -A, -2A}. It is built from the exact core of Schrottenloher's dialog Euclid and Bezout multiplier (arXiv 2606.02235). It is emitted in the challenge's own `ArithmeticCircuit` JSON and checked with the challenge's own executor.

**It does not qualify at 256 bits.** It is exact and complete, and its hierarchical upload passes the live board's sampled test. No route to qualification exists in the repository today. See "Unresolved gates" below.

`DESIGN.md` records the contract reading, the three designs compared, the data shape, the completeness argument, and this branch's qubit-time changes.

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
$PY -I -B scripts/check_passes.py                               # the same toy check through split_dirs, gate_pass and reorder_pass
$PY -I -B scripts/tick_equiv.py                                 # every Bezout tick variant against PR #18's flags, random states
$PY -I -B scripts/prefix_pebble_check.py                        # the pebbled prefix write, every mode, with mutants
$PY -I scripts/mutation_control.py                             # negative control for the checker
$PY -I scripts/component_check.py                              # 256-bit components on boundary inputs
$PY -I scripts/check.py --secp --secp-random 120               # 256-bit full circuit, sampled
$PY -I scripts/emit.py --out artifacts/secp256k1_G_circuit.json.gz
gzip -dc artifacts/secp256k1_G_circuit.json.gz | harness/target/release/qec-harness --circuit - --cases artifacts/secp256k1_cases.json
$PY -I -B scripts/emit.py --hier --out x.circuit.json
$PY -I -B scripts/post_pass.py x.circuit.json x-pp.json --jobs 8 --rounds 300   # post-passes on the hierarchical circuit
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

All numbers are measured unless marked as estimates. This section describes the reference circuit. This branch's figures are in "This branch" above.

| | toy q=127 | secp256k1, A = G | cap |
|---|---|---|---|
| wires | 70 | 2,311 | 4,096 |
| ops | 14,803 | 18,995,461 | 10,000,000 |
| CCX | 5,099 | 7,209,419 | |
| T + T-dagger from repo lowering | 35,693 (7.0 per CCX) | 50,465,933 (7 x CCX) | |
| measurements, R or Hmr | 0 | 0 | 0 |
| SubmittedMethod JSON bytes | 583,085 | 822,274,918 | 104,857,600 |
| `v2_ir` validator, service caps | PASS | REJECT, ops | |
| challenge executor lanes | 254 of 254 correct (exhaustive) | 510 of 510 correct (sampled) | |
| dirty scratch or phase lanes | 0 | 0 | |
| `is_qualified` | false | false | |

Where the 256-bit CCX go (`receipts/secp256k1_breakdown.json`). One dialog multiplier is 3,140,864 CCX: two Euclid passes of 787,968 each, one Bezout pass of 1,564,672, and 256 for the n-bit controlled swap between them. The kernel uses it twice. The lambda-squared subtraction is 910,080 CCX. Flags and constant adds make up the remaining 17,611.

The controlled subtract and the controlled modular add run a controlled CDKM ripple instead of copying `ctrl*a` into an AND register. A controlled subtract costs 3w CCX instead of 4w, and a controlled modular add went from 10n + 2 to 9n + 4. That took the 256-bit point addition from 8,686,084 to 7,967,236 CCX, 8.3% fewer.

Classical constants are no longer copied into a register for a full adder. The carry chain reads the constant's bits, so it skips carries that are always 0 and aliases a carry that equals an input bit. Each modular reduction runs one carry chain of s + 2^(n+1) - q. Its carry out is [s >= q]. It lands in the spare top bit, because the top bit and the carry into it are never both 1 when s < 2q. The same carries then drive the conditional subtraction. The reduction costs 3n - 3 CCX instead of 4n + 2, so a controlled modular add now costs 8n - 1, which is 2,047 at n = 256. A modular double costs 3n - 15 instead of 4n + 4 at secp256k1, because 2s has a zero low bit and the next three bits of 2^(n+1) - q are 0, so the first carries are always 0. Its flag is the low bit of the result, so the flag wire becomes that bit with no gate. Negation adds the sparse 2^n - q - 1 and then complements. That took the point addition from 7,967,236 to 7,209,419 CCX, 9.5% fewer, and from 26,931,713 to 18,995,461 ops. The carry chain still ripples the full width, because exactness needs it.

Exactness evidence, strongest first:

1. **Exhaustive gate-level check.** `receipts/toy_exhaustive.json` covers 36 instances: 18 primes of 4, 6, 7, 8, 9 and 10 bits, 2 addends each, every curve point plus infinity, and both control values. It finds 0 failures, with all scratch returned to zero. The same generator code emits the 256-bit circuit. Only the constants and the bound-derived width schedule differ.
2. **Challenge executor.** `receipts/harness_toy127.json` and `receipts/harness_secp256k1.json` replay the emitted JSON on `v2_runtime::BitSlicedExecutor`. Expected values come from `qec/curve.py`, which shares no code with the generator.
3. **256-bit boundary components.** `receipts/secp256k1_components.json` covers modadd with and without ctrl on edge-by-edge pairs around q and 2^255 and at q - f and 2^256 - 1 mod q, where f = 2^256 - q, plus moddbl on the same values and cneg. It runs cmodadd_const and gt_const_into with ctrl 0 and 1, gt_const_into without ctrl, and add_const under ctrl with a carry out, each on the kernel's constants for G (q - a, q - b, 3a and a) and on sparse and dense edge constants (1, f, 2^255, q - 1, q - 2, 2^255 - 1 and 2^256 - 1 mod q), with inputs on both sides of each constant. It also runs the multiplier on x = 0 and on x values whose Euclid run takes 511 of the proven 512 iterations. All pass.
4. **Proven iteration bound.** u*v at least halves per active Euclid step, so bitlen(q(q-1)) iterations always suffice. That is 512 at secp256k1. `receipts/model_check.jsonl` shows the observed worst case never exceeds the bound at any of the 13 tested q. It equals the bound at q = 3, 5, 11 and 43 and is one below it elsewhere.
5. **Checker sensitivity.** A must-fail control (an extra X on the output) fails every lane. Exhaustively, 163 of 200 random CCX-to-CX demotions are caught, and the survivors are correct on the whole promise domain, so they are equivalent circuits. Most CCX in constant adders and idle Euclid tails fire only on rare carries, so random 256-bit sampling cannot separate an exact circuit from an approximate one, which is why items 1, 3 and 4 carry the exactness claim.

The circuit is pure X, CX, CCX and SWAP, so it is a phase-free permutation. Basis exactness with clean scratch on the promise domain therefore gives the exact unitary on superpositions and entangled references (QNT-04). Outside the promise it is still a permutation, which satisfies QNT-06.

## Unresolved gates

These gates are listed in the order they blocked the flat upload on the earlier benchmark. Gates 1 and 2 do not block the hierarchical upload on the live board, which has the limits in "Result on the live board" above. Gates 3 to 5 remain open.

1. **Operation cap.** 18,995,461 ops against a cap of 10,000,000 (`MAX_DECLARED_RECORDS`). One multiplier alone is 8,685,300 ops. The paper reports about 2^24 total gates per addition even for its approximate circuits, so no published point addition fits this cap.
2. **Upload cap.** The service parses plain JSON, at about 43 bytes per op. 100 MiB therefore holds about 2.4M ops, roughly 8x fewer than this circuit needs. This also binds before the operation cap does.
3. **Whole-attack T budget (estimate).** DEF-02's binary wrapper makes 512 controlled additions, which comes to about 2.6e10 T against the 3.25e9 cap. A w = 16 windowed interface (QNT-02) would need about 32 additions plus exact lookups, roughly 2.0e9 T. That would fit, but it means a coherent-addend kernel, which is not built here.
4. **Qualification route.** `QualificationGraph` requires a universal secp256k1 certificate checked by a `SoundTheoremChecker`, a Shor wrapper certificate, a schedule, observations, and a retry model. The production registry holds no sound checker, so every artifact ends `is_qualified: false` with `No sound theorem checker exists`. This candidate supplies none of those artifacts.
5. **Formal proof.** Exactness rests on exhaustive toy verification of a size-parametric generator, full-size boundary tests, and the proven iteration bound. That is strong evidence, not a machine-checked proof.

## Optimization headroom (estimates, not built)

These estimates are against the earlier benchmark's caps, not the live board's limits.

- The dialog garbage compression from the paper's Fig. 1 saves qubits, not ops.
- The exact equivalent of the paper's pseudo-Mersenne reduction still needs a full-width carry, so that saving is smaller here than in the paper.
- A further 2x reduction would bring the ops under the operation cap, but the flat file would stay about 3.9x over the upload cap.

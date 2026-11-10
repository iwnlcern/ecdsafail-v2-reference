# Exact controlled point addition for `secp256k1-controlled-exact-v1`

## Current mux contract

The merged path now uses the paired-tick contract D1-D4 in [the mux proof](docs/mux-proof.md), superseding the older unrestricted canonical per-symbol Bézout-cell claims below. Logical T and R are canonical and scratch starts clean, but after a non-final B1 the physical word stores H = 2T + R in S4's encoded form, borrowing the next cell's g0 for its high bit. A semantic subtract follows B1; next.g0 must be 1 before holding; a final B1 requires R = 0. The held tick restores the borrowed symbol. Fallback primes retain the old step.

The additional sweep rotates each no-subtract run so B1 is last in Bézout order.
It runs in Euclid's tail, one cell behind the offset sweep, with which it commutes: the offset sweep's swap acts only on b of g0 = 1 cells, and each B1 step only on b of two g0 = 0 cells.
So Euclid hands Bézout the B1-swept tape, and inverse Euclid unsweeps it before undoing its steps; this supersedes the placement in S2's last sentence and S17's last sentence of the mux proof, not the maps they state.
The tape at rest is block-coded, two wires freed per 15 ticks by `qec/tape_block.py`; on the swept tape every B1 follows a subtract cell in tape order, so the code reads each block's cells last first and relies on the same clause.
Cells read previous-symbol wires and can write the next g0 across a block edge, so Bézout decodes a block one tick before its first tick and codes it again one tick after its last, and two blocks are decoded across each edge for two ticks.
The gate's toy primes have at most one block, so the edge path is covered by `scripts/block_check.py` at 20 bits and above and by the secp256k1 rows, not by a toy instance.
The four-symbol alphabet, 384-tick bound and width schedule are unchanged. M and its inverse use clean states in their respective input/output layouts; top-level R remains on-curve or infinity with canonical encoding and zero scratch.

The corrected S13 distinguishes logical carries from physical borrowed first-probe wires: only on P = 1 do those wires equal the carries; on P = 0 their junk is unobserved except through P and reversed. Exported e and the second chain remain zero on the inert lane. Credit arXiv:2609.05625 section VI for the offset representation.

The exact-source independent review covers this pair contract; it does not grant upstream checker acceptance. The local held secp BDD probe stopped at the production 4,194,304-node budget. The earlier cell's cap-transfer proof does not automatically transfer. I7.charge, qualification, and the flat-upload operation cap remain open limits.

## Clean wires in the mux tick

Before these changes the mux tick held the circuit's peak at 1,275 wires, with 26 clean, while Euclid, the code calls and the B1 step peaked at 1,249.
So each clean wire the tick gives up lowers Q by one, until another stage binds.
A wire is worth T / (768 × 7 × Q) CCX per tick, counting 768 tick uses and 7 T per CCX, which is 3.40 on this tree.

Two changes lower the tick to 19 clean wires at secp256k1, for 1,268 wires × 23,151,240 T on this tree, against 1,275 × 23,075,976 before them.
Their statements are C1–C4 and R1–R4 in `INVARIANTS.md`.

- **Kept low carries.** The second chain no longer keeps every low carry through its top.
  It keeps c2, c6, c9 and c11 on wires and c12 on bit 11's selector, clears each other one once the next kept carry is built, and rebuilds them on the way down, at 2 CCX per dropped carry.
  The mux cell already drops c1 to pay for k2, so this saves 6 more wires, at +12 CCX per tick.
  At secp256k1 a pebble search over these carries, whose up pass may clear and rebuild any carry many times, finds this peak at the same cost, one wire lower at +6 CCX more, which is above a wire's worth, and nothing two wires lower.
- **The quiet run's prefix top.** M's top below its last is read only by `M.write`, so it is cleared right after M is built and recomputed on a new wire just before the write.
  That is 1 wire at +2 CCX per tick.

Three other wire trades were priced and set aside:

- `l0` cannot be recomputed across the second chain from g0 and R₀.
  On a non-last B1 it holds H's bit n, parked there by the hide, and that bit is on no other wire.
- Moving `Prefix` tops onto lower wires costs about +6 CCX per wire.
  Each intermediate top is also the next block's `a`, read once when the prefix is built and twice in `write`.
  A conditionally clean copy needs a true control beside it at every read, as in arXiv:2407.17966, so every read pays a gate.
- Releasing P's top below its last saves nothing, because `P.write` runs at the second chain's top, which is the tick's peak.


## What the challenge actually requires

Read from a checkout of the ecdsafail-v2 benchmark repository (read-only), which is not public until the benchmark is published.

- Profile `profiles/secp256k1-controlled-exact-v1.json` sets `arithmetic_contract: EXACT_UNITARY`, `approximate_arithmetic: DISABLED`, `measurement_cleanup: DISABLED`, target `CLIFFORD_PLUS_T`.
- SPEC (`ecdsa specs/synthesized-v2/SPEC.md`) DEF-01 requires complete controlled addition of fixed public points. QNT-03 requires identity on control 0 and exact addition on control 1. QNT-05 requires the default exact profile to handle every reachable case, including infinity and R in {A, -A, -2A}. QNT-04 requires correctness on superpositions and with entangled references. SEM-03 rejects `R` and `Hmr`.
- Encoding (QNT-01) is little-endian affine coordinates with `(0,0)` as infinity.
- A submission is a `SubmittedMethod` JSON (`assurance/qualification/src/bundle.rs`) whose `circuit` is an `ArithmeticCircuit` of `X`, `CX`, `CCX`, `SWAP`, `Z`, `CZ`, `CCZ`, `NEG`, `CONTROLLED_NEG` (`contracts/ir/src/arithmetic.rs`).
- Service caps (`service/src/controller.rs`, `capacity/`, `contracts/api/src/quota.rs`) are 4096 wires, 10,000,000 operations, and a 100 MiB upload. The upload is parsed as plain JSON with no compression.
- `limits.json` caps compiled T at 3.25e9 for the whole attack. `compiler/lowering` lowers each CCX to the 7-T Toffoli.
- Qualification (`assurance/qualification/src/graph.rs`) needs a universal certificate checked by a `SoundTheoremChecker`. The production registry holds none, and `qcb diagnostic check` executes nothing. No artifact can be `is_qualified` in this repository today.

So Schrottenloher's published circuits (arXiv 2606.02235) cannot pass as published. They use MSB-only comparisons, pseudo-Mersenne shortcuts that skip rare carries, a fixed iteration count tuned for random inputs (failure up to 2^-13.3), measurement-based uncomputation, and they ignore doubling and infinity. Each of these violates one of the clauses above. What carries over is the exact structure underneath: the dialog Euclid algorithm and the out-of-band Bezout reconstruction that turns an inverse into an in-place multiplication.

## Designs compared

Costs are Toffoli estimates at n = 256 before implementation. Every design here is exact and uses only X, CX, CCX, SWAP.

**A. Exact dialog multiplier (chosen).** Point addition follows the paper's Algorithm 1. It uses an in-place multiplier M(x, y) = (x, xy) and its reverse, plus one out-of-place lambda-squared subtraction. M runs Euclid forward to turn x into a dialog bit-vector, then runs Bezout on (y, 0), then runs Euclid backward. Making it exact means three changes. The iteration count becomes the proven bound bitlen(q(q-1)) = 2n. Comparisons and modular reductions become full width. Uncomputation uses reverse circuits instead of measurement. Euclid widths still shrink, because u*v at least halves each iteration, so the width can be min(n, bitlen(q(q-1) >> i)). Estimated cost is about 58n^2 per M and about 136n^2, roughly 9M Toffolis, per addition. Qubits are about 9n.

**B. Kaliski or binary-EEA inversion plus schoolbook multiplications (Roetteler et al. 2017, Haner et al. 2020).** This design computes lambda with an explicit inverse, then uncomputes the inverse. That costs two inversion passes plus at least three multiply-and-uncompute pairs. Published exact counts are about 2.5e8 Toffolis per addition for Roetteler. Haner's improvements remain several times above design A. Rejected because it costs more on every axis that binds.

**C. Projective complete formulas (Renes, Costello, Batina).** These need no inversion and are complete. They are rejected because projective coordinates are not canonical. An in-place R -> R + A on the declared affine encoding still needs a normalizing inversion to uncompute the old point. That adds design A's cost on top of the formulas.

## Completeness on the fixed addend

For a fixed public A = (a, b), the generic kernel fails only on R in {inf, A, -A, -2A}. On those inputs x - a = 0 or a - x' = 0, and M is not a bijection there. All four inputs and their sums {A, 2A, inf, -A} are public constants. The circuit therefore:

1. computes flags f_k = c AND [R == E_k] for the four exceptional inputs, and gen = c AND NOT(f1 xor f2 xor f3 xor f4);
2. XORs E_k xor O_k into the accumulator under f_k;
3. runs the generic kernel controlled only through gen, using M^-1 ... M. With gen = 0, every gen-controlled step is the identity, and M^-1 followed by M cancels exactly at the circuit level when M's entries are in [1, q), which the kernel's flag code ensures;
4. uncomputes gen, then uncomputes each f_k from the output as c AND [R' == O_k]. This works because R -> R + A is a bijection.

M's standalone contract is narrowed to canonical x in [1, q); x != 0 at M's entries rests on the kernel's flag code.
The build refuses curves with a finite point at x = 0 and addends with x(A) = 0 or x(2A) = 0.
The kernel codes both the -A exception and the c = 0, R = inf marker as f0 = f1 = 1, while keeping gen zero.
It flips X's low bit under f0 AND f1 before M inverse and restores it after M, holding no extra wire during M.
The direct evidence is the instrumented observation that the indicator equals [X = 0] before the remap and X != 0 at both M entries, on all 25,800 toy lanes and the secp256k1 special points.
The guarantee's control of record removes both remaps with every flag intact; the existing point gates then fail on 72 lanes, c = 0 with R = inf and c = 1 with R = -A, across all 36 toy instances.
The marker-removal, one-remap and old-gen-fold mutants are sensitivity controls; they do not test the nonzero-X guarantee.
This named limit accompanies I2.model's narrowed zero-caller row.

## Reusing the Euclid pass's zero words

These arguments apply to every odd prime q of n >= 2 bits and every canonical x and y in [0, q).
Let P = q(q - 1), T = bitlen(P), and x' = x XOR [x = 0].
Then 1 <= x' < q, including x = 0 and x = 1, whose distinct original values are retained by z.

**F1: the terminal state is (1, 0).**
Initially u = q is positive and odd, v = x' is nonnegative, and gcd(u, v) = 1 by primality.
If v is even, halving v halves the product and preserves the gcd because u is odd.
If v is odd, the conditional swap makes the subtracted operand no larger than v, and (v - u)/2 is nonnegative.
That step preserves the gcd and makes the product at most half its preceding value, with u still positive and odd.
Thus u_i * v_i <= floor(P / 2^i), and at i = T the product is zero.
Since u_T is positive, v_T = 0; the preserved gcd then forces u_T = 1.
This is an argument using the existing iteration bound, not a sampled reduction of that bound.

**F2: the zero remap is reversible.**
The flag z stays allocated while x' is used, and the inverse pass restores x' before the remap is undone.
The resulting x is the original input, so the repeated zero test clears z exactly.
The x' = 1 path multiplies by one and therefore implements the specified identity on y for original x = 0.

**F3: V supplies S.**
V is zero after the pass, even without remapping zero, so its wires can hold the initially zero Bezout word S.
The existing exact Bezout recurrence on canonical R = y and S = 0 ends at R = 0 and S = x' * y mod q.
No arithmetic cell, modular reduction or transcript alphabet changes.

**F4: only U's upper wires are lent.**
With the remap, U = 1, so U[1:] are zero and U[0] must remain allocated.
The allocator can lend those n - 1 zero wires to Bezout scratch; the unchanged exact cells restore that scratch to zero.
Reclaiming n - 1 zero wires afterwards need not return the original wires or their order.
The reclaimed word consists of the still-live U[0] followed by these zero wires, disjoint from the live x and y words.

**F5: restore the terminal state on a new layout.**
X on U[0] zeros the reclaimed word, and swapping it with S moves the product onto the word that becomes out while zeroing V again.
R on y's wires is zero, so X on its low wire recreates one there.
The triple (y wires, V wires, tape) now contains (1, 0, tape), the terminal state required by the inverse Euclid block.
Block replay on that matching layout restores (q, x', zero tape), and loading q again clears y's wires.
The cleared y word and tape are freed before the final equality test, allowing its AND chain to reuse their wires without increasing the peak.
The remap is then undone, z is cleared and freed, and the product word becomes the logical y output.
Every other wire is returned clean; reversing this same block gives M inverse, including its identity at x = 0.

This change relies on the existing proven bound and width schedule and changes neither, nor any modular reduction.
For nonzero remapped inputs, active operands fit the existing schedule, and the idle state (1, 0) also fits every emitted step.
The gate-level small-prime checks and 256-bit boundary components exercise the implementation; they do not replace the all-input argument or constitute acceptance by the separate benchmark proof checker.

## Shared-chain Euclid pass with a held offset

For every odd prime q, put n = bitlen(q), P = q(q - 1), T = bitlen(P), and use the unchanged width_schedule.
Let (u_i, v_i) be the reference Euclid state and (b_i, s_i) its parity and conditional-swap bits.
The following describes the step and sweep introduced with this proof; the existing F1-F5 and C1-C10 arguments remain its external boundaries.

**E1: the physical step, before wire rotation.**
For w >= 1, take a w-bit odd word a, a w-bit word B, prior offset bit e (zero for the first step), two zero tape bits and a zero carry-in.
Let vhat = B XOR e on the low bit, S = a + NOT(vhat) modulo 2^w, and c = [a > vhat], its carry out.
The physical output before interpreting the rotation as halving is:

| Condition | a output | B output | g0 | g1 |
|---|---|---|---|---|
| vhat even | a | vhat | 0 | e |
| vhat odd, c = 0 | a | vhat - a | 1 | 0 |
| vhat odd, c = 1 | vhat | a - vhat - 2 | 1 | 1 |

The upward majority chain stores c_(i+1) on a[i], NOT(vhat_i) XOR a_i on B[i], and c_i XOR a_i on the preceding carry wire.
On descent, one CCX restores a_i and a second adds g0*(c_i XOR a_i) to B[i].
Two CX gates restore the preceding carry and remove a_i from B[i], leaving S_i when g0=1 and NOT(vhat_i) otherwise.
The identity S_i XOR c_i XOR 1 = a_i XOR vhat_i lets an X-conjugated CCX controlled by g1 exchange the selected a bit without a second ripple.
Complementing B gives NOT(S) when g0=1 and vhat otherwise.
A CX from g1 to each upper B bit selects S on those bits in the swap case, while retaining NOT(S_0) on the low bit.
Odd a and odd vhat imply S_0=1, so B's low bit is zero in both odd branches; it is also zero in the even branch.
The final controlled copy adds (NOT(g0))*e to g1; it preserves e, and the carry-in returns to zero.
Thus the low-bit rotation of the full V word is exact halving, and E1's table must not be confused with the logical post-rotation word.
The first variant costs 4w+1 CCX and later variants cost 4w+2, including the offset copy.

**E2: held-state induction.**
For 1 <= x < q, before each step U=u_i and V+e_i=v_i with V>=0, e_0=0, and e_i the preceding cell's second bit for i>=1.
If e=0 the absorb leaves V=v; if e=1 and v is odd it changes the even V=v-1 to v; if e=1 and v is even it changes the odd V=v-1 to v-2.
Therefore vhat has v's parity and is at most v.
For odd v with u>v, the gate outputs after rotation are (U,V,e')=(v,(u-v)/2-1,1); u-v>=2 makes V nonnegative.
For odd v with u<=v they are (u,(v-u)/2,0).
For even v they are (u,v/2,0) if e=0, and (u,v/2-1,1) if e=1.
Each case preserves V+e'=v_(i+1) and U=u_(i+1), with e' the current second tape bit.
During active steps u and v fit the existing window, so vhat<=v and V<=v make this chain exact in that window.
This uses the existing product-halving bound proved in F1; it is not Bernstein-Yang's different divsteps bound.

**E3: zero tail and terminal offset.**
When v_i=0, nonnegative V and e force V=e=0.
Then g0=g1=0 and the upward/downward chain restores U and V for every window, including one narrower than U.
F1 gives u_T=1 and v_T=0 for nonzero canonical x over prime q, hence U=1,V=e=0 at the end.
Primality is used here through F1's gcd argument.
The last odd step has u=v=1, since reaching zero requires equal operands and their gcd is 1, so that step leaves no offset.

**E4: descending tape sweep.**
Write z_i for the unswept second bit: z_i=s_i on odd cells, z_i=z_(i-1) on later even cells, and z_0=s_0.
For i=T-1 down to 1, XOR (NOT(b_i))*z_(i-1) into z_i.
Descending order reads the predecessor before it is changed, so even cells become zero and odd cells keep s_i.
The swept tape is exactly the reference dialog, allowing the unchanged Bezout cell, reduction proof and F1-F5 cleanup to apply.
The sweep costs T-1 CCX and runs inline in the pass body, so that blocks of the tape code can be decoded and coded between its steps.

**E5: zero inputs and generic callers.**
The multiplier remaps original x=0 to x'=1, covered by E2; the retained flag distinguishes it from original x=1.
The pass's domain is therefore V != 0, which E7 needs.
A generic caller with V=0 would keep U=q, whose high wires are not zero, so E7's lent wires would not start clean; the pass no longer supports that caller.

**E7: late tape on U's vacated wires.**
For V != 0, the proven bound gives max(u_t, v_t) < 2^(w_t) at every boundary t, or (u_t, v_t) = (1, 0), so U < 2^(w_t).
A step at width w touches only U[:w], and widths never grow, so U's wires from w_t up are zero from boundary t to the end of the pass.
When the width drops, the pass adds those wires to the tape's supply, the highest first, as it already does with V's.
It lends as many as it ends with zero wires: V's tail and two free wires per coded block, 52 at secp256k1.
At the end U takes those zero wires back as its upper wires, so U is again (1, 0, ..., 0) and S reuses the whole word.
The boundary is a permutation of U and the pool together, so replaying the pass block in reverse gives inverse Euclid the mirror layout with no extra gate.
Euclid's pool is smaller by the lent wires, which puts Euclid's peak below Bezout's, so no block is coded during the pass any more.

**E6: reversal and unchanged surfaces.**
Reversing the pass reverses its sweep and then its step calls, including their layout permutations.
F5 restores precisely the terminal words and swept tape needed by that inverse, so no additional garbage or cleanup assumption is introduced.
The external pass relation, iteration bound, width schedule and modular reduction are unchanged; EuclidStep's physical relation and internal second-bit meaning change as E1 states.

At G, sum(W)=98432 and T=512, so each pass costs 4*98432+2*512-1+511=395262 CCX instead of 591104.
Four passes save 783368 CCX, or 5483576 T; these are formulas to check against emitted counts.
This change leaves _bezout and qec/arith.py unchanged.
Finite gate checks and the separate leaf lift probe do not establish acceptance by the benchmark checker or compose this proof automatically.

## Exact merged Bézout reduction

For every odd q >= 3 and canonical R, S, `Mod.dbl_add` computes (2S + g0 R) mod q with clean scratch. The Bézout step then applies the existing g1-controlled swap; its contract and [2, n, n] layout are unchanged.

q is odd with n = bitlen(q), and f = 2^n − q, so f is odd and 2^n < 2q.
m is any integer with bitlen(2f) ≤ m ≤ n, so 2f < 2^m, and the generator takes the smallest.
That range is never empty, because 2f < 2^n.
At secp256k1, m = 34 and h = n − m = 222.

R, S < q and g0 is a bit.
s = 2S + g0·R, so s ≤ 3q − 3.
t = s >> n, low = s mod 2^n, L = low mod 2^m, H = low >> m, and A = [H = 2^h − 1], with A = 1 when h = 0.
For j in {0, 1, 2}, c_j is the carry out of L + j·f on m bits, so c_0 = 0 and c_1 ≤ c_2 ≤ 1.
e = A·c_(t+1) for t ≤ 1, e = 0 for t = 2, and k = t + e.

### Gate order

Every wire below is a named position in `s`'s bits: bit 0 is a fresh wire `l0`, bits 1 to n are S's wires, and bit n + 1 is a fresh wire `top`.
So `low` is `[l0] + S[:n-1]`, t0 is `S[n-1]`, and t1 is `top`.

| # | Action | CCX at n = 256 |
|---:|---|---:|
| 1 | `add(R[1:] + [l0], S, top, ctrl=g0)`, with `l0` still zero as a's top bit | 3n + 1 = 769 |
| 2 | l0 ^= g0·R[0] | 1 |
| 3 | Prefix-AND chain over H, the carries of H + 1, giving A_i and A_h = A | h − 1 = 221 |
| 4 | Set selectors in place: X t1, CX t0 → t1, so t1's wire is [t = 0] and t0's wire is [t = 1] | 0 |
| 5 | Carry chain up on L + [t = 0]·f + [t = 1]·2f, giving c at the top | m = 34 |
| 6 | Fresh `e` ^= A·c, a CX when h = 0 | 1 |
| 7 | The same chain down, then undo step 4 | m = 34 |
| 8 | t += e in place: CCX(t0, e, t1), CX e → t0. The wires now hold k0 and k1 | 1 |
| 9 | Carry chain up on L + k0·f + k1·2f, giving c_k at the top | m = 34 |
| 10 | e ^= A·c_k, which clears `e`, a CX when h = 0 | 1 |
| 11 | For i from h − 1 down to 1: clear A_(i+1) from A_i and the old H_i, then H_i ^= c_k·A_i. Then H_0 ^= c_k, a CX | 2(h − 1) = 442 |
| 12 | For i from m − 1 down to 1: clear c_(i+1) from the old L_i and c_i, then L_i ^= addend_i xor c_i. Then clear c_1 from `l0` and the bit-0 addend, leaving `l0` alone | m = 34 |
| 13 | Fresh z = g0·¬k0 | 1 |
| 14 | `gt_into(R[1:], low[1:], t1, ctrl=z)`, which clears k1 | 2(n − 1) + 1 = 511 |
| 15 | Clear z | 1 |
| 16 | CX l0 → t0, so t0's wire holds S'[0], then l0 ^= g0·R[0], which clears `l0` | 1 |
| 17 | Free `l0`, `top`, `z`, then rotate `S` by one wire, as `moddbl` does | 0 |
| 18 | `cswap(g1, R, S)`, in `_bezout` | n = 256 |

A low chain has one AND per position, with the addend bit at position i being one of 0, the first selector, the second selector, or their XOR.
That is MAJ(L_i, addend, carry), and CX conjugation gives it with one CCX.
`_carries(C, H, 1)` already builds step 3, up to A_(h−1), and A_h is one more carry on top of it, so reuse what fits.
`_reduce`'s down loop has step 11's order: it uncarries position i + 1 before it writes position i.

**Cleanup order in steps 11 and 12.** A stored wire reads the old bit below it, so each one clears before that bit is written.
A_h goes first, right after step 10, because it is used only by steps 6 and 10 and writing H_(h−1) first would leave it dirty. With h = 2, H = 11 and c_k = 1, A_2 would stay 1.
A_1 is an alias of H_0 and has no wire to clear.
With h = 1, A is H_0, step 3 is empty, steps 6 and 10 read H_0, and step 11 is the single CX.
With h = 0, A is the constant 1, step 3 is empty, steps 6 and 10 are CX from c and c_k, and step 11 is empty.
Step 7 runs step 5 in reverse with nothing written, so it has no such order to get wrong.

The cell is inline, with no `C.apply` or `C.record` inside, because `lift` refuses a leaf with calls.

### Correctness and cleanup

**C1.** t is 0, 1 or 2, because s < 3q < 3·2^n.

**C2.** For j ≤ 2, low + j·f ≥ 2^n exactly when A·c_j = 1.
This holds because low + j·f = (H + c_j)·2^m + ((L + j·f) mod 2^m), and H + c_j reaches 2^h only from H = 2^h − 1 with c_j = 1.
The carry is one bit because 2f < 2^m.

**C3.** k = floor(s/q), and S' = s − kq = low + k·f − e·2^n.
With e = 0 and t ≤ 1, C2 gives low + (t + 1)f < 2^n, so 0 ≤ low + t·f < q.
With e = 1, C2 gives low + (t + 1)f ≥ 2^n, so S' ≥ 0, and S' < q is the same as s < (t + 2)q, which holds for t = 0 because s < 2^n < 2q, and for t = 1 because s < 3q.
With t = 2, S' = low + 2f < q is the same as s < 3q.

**C4.** Step 5's addend is (t + 1)·f below 2^m for t ≤ 1 and 0 for t = 2, so step 6 computes the e above.
Step 7 finds the selectors as step 5 left them, because step 6 touches only `e`.

**C5.** e = A·c_k, on every reachable state.
If e = 1 then c_k = c_(t+1) and A = 1.
If e = 0 and t ≤ 1 then c_k = c_t ≤ c_(t+1), so A·c_k ≤ A·c_(t+1) = 0.
If t = 2 then low + 2f < q < 2^n, so A·c_2 = 0 by C2.
So step 10 clears `e` while A and c_k are still live.
This is the sketch's trap: `e` is the overflow out of H as well as k − t, so it needs no clear of its own from the final state.

**C6.** S' = ((H + c_k) mod 2^h)·2^m + ((L + k·f) mod 2^m), which follows from C3, and S' < 2^n.
Steps 11 and 12 write exactly that, and k0, k1 are exclusive because k ≤ 2.
Every stored A_(i+1) and c_(i+1) reads 0 after its clear, because each clear runs on the same inputs that set the wire: the old bit, and a lower chain wire that has not been cleared yet.

**C7.** k1 = 1 needs g0 = 1, because g0 = 0 gives s = 2S < 2q.

**C8.** Step 14 clears k1.
If z = 1 then k is even, so S' and R have the same parity and [R > S'] = [R >> 1 > S' >> 1].
With k = 0, S' = 2S + R ≥ R.
With k = 2, S' = R − 2(q − S) ≤ R − 2.
So the compare equals k1.
If z = 0 then k1 = 0, by C7 or because k0 = 1.
k = 1 can have R > S', so the compare needs z and not g0 alone.

**C9.** S'[0] = k0 xor g0·R[0], because 2S is even and q is odd.
So step 16 leaves S' on the rotated wires and clears `l0`.

**C10.** For each g, S → 2S + g0·R mod q is a bijection on [0, q), because q is odd.
The circuit is a permutation that takes canonical states with zero scratch onto canonical states with zero scratch, so the reversed cell that M⁻¹ runs is exact on canonical inputs too.

Nothing here uses primality, n = 256, or a run of ones.
The run of ones only makes m small.

## Data shape

- **Wire.** An integer index. **Register.** A Python list of wires, least significant bit first. Shifts are relabels, implemented as list rotations with no gates.
- **Circuit.** A flat `array('i')` of (opcode, a, b, c), a free-list wire allocator that requires ancillas to be returned clean, and a high-water mark that becomes `num_wires`.
- **Block.** A recorded op range with input layout, output layout and internal wires. It can be instantiated forward or reversed onto any matching layout. That is how M^-1, Euclid-backward and every compute/uncompute pair are emitted without hand-written inverse code.
- **Simulator.** A bit-sliced simulator in which each wire is a Python int and bit j is test lane j. It shares no code with the generator's arithmetic.

## Why the permutation is enough for QNT-04

The circuit contains only X, CX, CCX and SWAP. It is therefore a permutation matrix with no phases. The point-addition promise is R on the curve or infinity, with canonical encoding and zero scratch. On this domain, x and y parity identify each finite point; the x=0 guard retains a full y comparison when required to distinguish infinity. Both flag computations use this same domain before and after the controlled translation. If every promise basis state |c, R, 0> maps to |c, R + cA, 0>, the same matrix acts exactly on any superposition of promise states and on any extension with a reference system. Basis exactness plus clean scratch is the whole unitary contract. Outside the promise, the circuit is still a permutation, so QNT-06's valid-channel requirement holds by construction.

## Last-cell lend

The merged pass lends the uncoded final tape cell's b after its last read, using its proved complement relation with g0.
Restoration allocates fresh zero storage and updates the tape boundary before inverse Euclid.
The codec and tick schedule do not change.
See `docs/tail-lend-proof.md`.

## Seven-tick whole-block capacity codec

The seven-cell block codes W[0..12], whose six inner windows (g_i, b_i, g_(i+1)) exclude 000, while W[13] remains raw.
The existing block reading order reads cells last first on the B1-swept tape.
There are 4,021 coded words and 8,042 inputs including both raw-b values.
The primitive list is pinned by canonical-JSON SHA256 8a43178163d443fa37f9f9e8da3431e876f476d8f1d4c87b0710760106813ed2.
Its 326 primitives include 36 CCX: the five-cell part costs 15, the re-tag costs 11, and the U1 hand pack costs 10.
Two clean ancillas at indices 14 and 15 return zero, and FREE = (10,) names the block wire returned zero.
W[13] is absent from every primitive operand.

The generic encode/decode plumbing and sweep order are unchanged.
Each decode supplies one fresh zero wire and each encode returns one wire to the allocator.
At secp256k1, 54 blocks code 378 cells and leave six uncoded, preserving the last-cell lend.
Euclid lends 56 wires: its two tail zeros plus one zero per coded block.
Two decoded blocks still overlap across each edge.
Item 4's borrowed tape storage must be checked on this layout at its four borrow boundaries; later physical reuse is distinct from a live FREE or codec-ancilla role.

This section supersedes the earlier fifteen-tick codec summaries and their 25-block, two-free-wire and 52-wire lending figures.
The whole-block domain requires all six inner windows, including the boundaries the previous three-part codec did not code.
Checks cover the emitted list exhaustively, both sweep states at every gate prime with coded blocks, M and its inverse with every wire compared, the merged block edges, and directed secp256k1 lanes.
The ordinary toy mutation row is finite-domain evidence; separate codec deletion controls test codec sensitivity.
Every figure retains the limit: **`I2.model`'s zero-caller row narrowed from the reference's gate**.
No checker admission or qualification is implied.

### Euclid tick 0 on its initial boundary

The initial U is q, V is the nonzero-remapped x, E is zero, and the cell and remaining tape are clean.
The first step alone uses `_euclid_tick0` when the width is n and k = n - 1 - lsb(2^n - q + 1) unwritten tape wires are available.
For odd v its output is U=v, V=(q-v)/2-1, E=1, cell=(1,0).
For v=2 mod4 it leaves U=q, E=0 and cell=(0,0), with V=v/2; for v=0 mod4, V=v/4 and cell=(0,1).
The V list rotates exactly as in the general step.
Controlled addition of 2^n-q+1 uses the borrowed wires as carries; it uncomputes them before returning.
The recorded block's input/output wire sets agree and contain every used wire.
If width, supply or that shape contract fails, the first step falls back to `_euclid_step`; later steps always use the general block.
The borrowed wires are zero at tick 0's entry/exit and the mirror points in the inverse pass.
They may hold nonzero tape at E's exit, so they are not generally clean scratch there.

### Inline kernel constants and reflection

For cmodadd_const, k=c*[b>=q-v].
The control temporarily holds c XOR k, so selectors (c XOR k,k) exclusively select v or v+2^n-q.
The selected carry chain starts at the lowest set bit of either mask and uncomputes each carry before overwriting its old input bit.
After restoring c, [b'<v] clears k.
For creflect, k=c*[b>a].
Complement b under c and add a+1 or (a+1+q) mod2^n under the same exclusive selectors.
The output remains canonical and lies above a exactly when k, so CX(c,k) followed by comparison a+1>b' clears the flag.
This reflection is valid at b=0 as well.
It replaces kernel cneg and the final constant addition; the intervening subtraction acts on Yr and commutes.
All gates are inline: addend-dependent constants introduce no new recorded definitions.
Existing standalone cneg behavior and its tests remain.

### Kernel flag recode across M

At either M boundary, (c, f0, f1, gen) is one of 0000, 0110, 1000, 1100, 1010, 1110 and 1001.
With c = 0 the word is 0000, or 0110 with the zero marker; with c = 1 it is 1001 on the generic path and 1000, 1100, 1010 or 1110 on an exception.
CX(gen, c) and CX(gen, f0) move 1001 to 0101, then gen XOR= NOT c AND f0 AND NOT f1 clears gen using three CCX and one temporary wire.
The other six words are unchanged, and the seven remaining (c, f0, f1) words are distinct.
The kernel releases zero gen before each M apply and allocates a zero wire before reversing the recode, restoring c, f0, f1 and gen before any subsequent reader.
The released physical wire may become M scratch or an output register; the reallocated wire carries the restored logical gen and is returned to build.
Four recodes cost twelve CCX in total and free one live wire throughout both M applies.
Direct observation checks both boundaries on all 25,800 toy lanes and twelve secp256k1 special-point lanes, including zero at release and reallocation, word restoration and final data/control/scratch.
Removing gen's clearing CCX is the freed-wire control of record; removing CX(gen, f0) is a sensitivity control, each checked over the full toy population.
The inherited narrowed zero-caller row and standalone M domain x in [1, q) remain unchanged.

## Five-cell tail code

The merged Bezout pass codes five raw tail cells when the tail lend applies, at least five cells are uncoded, and the width at tick T - 5 is at most six.
The code applies after tick T - 6 and its block encode check, and reverses after block zero is encoded but before the lend restore.
Its 168 primitives cost 21 CCX per apply, return FREE = (2,) zero, and restore three clean ancillas.
The domain is all 230 L6 words on nine live wires; the exhaustive tail_code component row checks distinct images, zero FREE and ancillas, and exact reverse restoration.
A real-tape corpus reaches 84 of those words; the remaining words are covered by the exhaustive component row, not claimed as reached tape states.
The new row adds one component and 230 lanes, changing the census from 50 / 9,615 to 51 / 9,845 without changing any other row.
Both existing named limits remain: I2.model's narrowed zero-caller row, and standalone M restricted to x in [1, q) with nonzero entries resting on kernel flags and the non-full build refusal.
The tail code additionally requires T >= CELLS + 1 so its forward tick exists; the inverse asserts that the forward apply ran.

## Block-zero code

For canonical x < q, strips before the first subtract leave U = q > V, so that subtract swaps and the swept block-zero word lies in the 4,022-word D1 subset of the L6 domain.
On the merged path with a block, a 21-CCX pre-map followed by the existing codec frees W10 and W13, with seven pre-map ancillas returned before the codec allocates its two.
The extra free wire joins the clean pool that U borrows at the pass boundary, reducing external tape by one; with_free_slots restores both positions for later passes.
A one-CCX HOLD frees b0 while ticks 7 through 2 run and restores it before tick 1, without touching the g1 next-cell lend.
Nonmerged paths keep the original codec, and no new per-prime domain refusal is needed.
The exhaustive D1 component row checks encode and HOLD both ways, clean FREE and ancillas, and distinct images, increasing the census only to 52 rows and 13,867 lanes.

## Pooled tick schedule

Selected merged ticks use a cap-four reversible pebble search on the second chain's unperched prefix, derived from that shape's length and quiet-run top wire.
The construction applies when the sole low perch is at a-1 and a cap-four schedule exists; otherwise that shape keeps the original kept-carry schedule.
At secp256k1 the 106 block-edge ticks, tail-code edge 378, and ticks 1 and 0 use the schedule, adding six CCX each.
At the other measured merged primes below 20,000 where this schedule runs, CCX changes by zero to minus two per use, with no wire change or scratch increase.
The uncoded terminal cell is never B1, so when the existing tail lend applies, tick T - 2 uses the no-prev shape and the terminal b release moves before that tick.
That released b is unread until the existing end-of-pass X and CX restore it.
These changes leave the nonmerged path and no-tail-lend instances unchanged.

The codec's primitives 4, 128, 145, 162, 188 and 197 use the six checked CX substitutions, giving a 30-CCX reversible code.
Its contract holds on all 8,042 window words and all 4,022 pre-map images; 6,254 window words have changed codewords, and each block uses the same shape for encoding and decoding.
Each of the six wrong-control substitutions fails the domain contract.

The merged Euclid sweep carries one mark equal to NOT g_k AND NOT g_(k-1) across the pass, updating it before the next block encode and clearing it before block zero is encoded.
Each update toggles the mark by NOT g_(k-1) AND (g_k XOR g_(k-2)), so every g read is still raw.
At 384 ticks this saves 382 CCX per Euclid pass, 1,528 per point addition, while the nonmerged sweep is unchanged.

## Euclid scratch scheduling

On odd Euclid ticks, the pass holds the most recently freed wire around the general step call, then returns it.
Each step takes clean scratch and restores it to zero, so selecting another clean wire preserves the step function and its inverse on the same reachable domain.
Alternating scratch lets the next carry sweep overlap the previous strip sweep without changing the arithmetic or allocator.
At secp256k1 this keeps 1,246 wires, 3,301,711 CCX and 12,144,387 expanded operations, while scheduled depth falls from 37,219,511 to 34,257,903 ticks.

### Parallel final coefficient swap

The final coefficient swap copies its control through a breadth-first CX tree onto scratch wires already touched and freed by that tick.
The free-wire contract makes those copies equal to the control, and the reverse tree clears them after the swap while restoring the allocator stack exactly.
Using only the tick's own wires preserves its arity and peak space, while distributing the swap controls reduces scheduled depth without adding CCX gates.
At secp256k1 the circuit uses 1,246 wires and 32,866,267 scheduled ticks, with 3,301,711 CCX and 12,171,527 expanded primitive operations.

## Euclid unwind and parallel rotation

The unwind replaces `c ^= a; v ^= a; v ^= c` with `v ^= c; c ^= a`, an identity on every bit state.
The complemented-control select `X(v); CCX(g1, v, a); X(v)` becomes `CCX(g1, v, a); CX(g1, a)`, while the empty X pair at bit zero disappears.
Each final `CX(g1, V[i])` moves into its unwind bit, across operations that neither write g1 nor change V[i] except by X.
The conditional left rotation reverses the whole word and then all but its last bit, giving two layers of disjoint swaps with the same number of CCX gates.
A reversible CX tree copies the rotation control onto available clean tape wires and uncomputes those copies after both layers.
The pass supplies at most `(w - 1) // 2 - 1` copies from its unused, vacated or lent clean wires; sharing fewer copies round-robin preserves the same permutation.
The zero-copy case retains the original step shape and serial rotation, including the smallest widths.
Together these changes retain 1,246 wires and 3,301,711 CCX while reducing expanded operations to 11,901,731 and scheduled depth to 31,583,277 ticks on the current circuit before emission reordering.
### Block carry-select coefficient comparison

The coefficient tick splits its comparison into at most seven blocks, borrowing two clean touched scratch wires per additional block.
Each block computes carries for zero and one input carry; their XOR is propagation, so combining the copied zero-carry result with the incoming carry produces the full comparison carry.
Reversing the combine and block computations restores the inputs, clears borrowed scratch and restores the allocator stack exactly.
The 255-bit comparison adds 888 CCX per tick call without increasing arity, trading T for shorter depth.
On the coefficient-swap base this gives 1,246 wires, 29,830,617 scheduled ticks, 3,981,919 CCX and T 27,873,433.

### Split coefficient add and shift

The coefficient add-or-shift separates the add on not-J from the rotation on J.
The forward gate modifies a sum bit while its carry wire is idle; x*x = x makes the added term cancel in the ordinary carry undo.
This leaves one CCX per bit in the back pass, followed by two layers of reversal swaps as soon as their bits are final.
Four clean copies of J supply the gates and swaps, and the reverse copy tree clears them before the original allocator order is restored.
On the block-compare base the circuit has 1,246 wires, 26,720,097 scheduled ticks and T 29,235,381.

The three extra copies increase small-prime M allocation by one wire at q=7 and two at q=31 and q=127, and toy127 point-add by two wires, while secp256k1 stays at 1,246 wires with unchanged tick arities.
On the integrated unwind-and-depth-pass head, the split add gives 1,246 wires, 17,970,862 scheduled ticks and T 32,869,445.

### Compare carry-pair cancellation

A block chain equals its CX gates followed by its CCX ladder.
The CX gates only write the carry-in and commute with its X, so their undo-redo pairs cancel around each carry flip.
Only the ladder is undone and redone at the two compare carry transitions; all other chain calls remain complete.
On the U1 head the circuit has 1,246 wires, 17,824,402 scheduled ticks and T 32,869,445.

### Increment prefix before uncompute

Prefix.write increments each bit on x XOR p times not-below before uncomputing its AND chain.
The uncompute is unchanged because below times (x XOR p times not-below) equals below times x.
The remaining CX from p to each bit waits until the block's last bit, so the carry chain contains only the uncompute CCX.
The first two bits take the same form with t, and every Prefix.write caller uses this identity, including the square.
CCX, T and wires are unchanged; the extra X and CX gates trade operation count for shorter depth.
On the U1 head the circuit has 1,246 wires, 16,313,762 scheduled ticks and T 32,869,445.

### Cap the tick's prefix blocks

Prefix accepts an optional block-length cap, preserving its AND-top and increment contracts for every block partition.
Only the tick's high-word P uses cap 37; other Prefix callers retain the uncapped default.
The shorter P blocks add three top wires without changing CCX or T, and the compare keeps its existing weights.
At the tick's 221-bit high word the blocks are 3, 7, 15, 31, 37, 37, 37, 37 and 15.

| Cap | Wires | Ticks | Qubit-time |
|---|---:|---:|---:|
| none | 1246 | 10236496 | 12754674016 |
| 62 | 1247 | 9849259 | 12282025973 |
| 47 | 1248 | 9743813 | 12160278624 |
| 37 | 1249 | 9710374 | 12128257126 |
| 30 | 1250 | 9708694 | 12135867500 |
| 26 | 1251 | 9693851 | 12127007601 |
| 23 | 1252 | 9705667 | 12151495084 |
| 21, 19, 17, 16 | 1253–1256 | 9778105 or more | not tabulated |

The 1,253–1,256-wire variants reduce qubit-time by at most 3.941 percent versus the uncapped head.

Cap 37 uses two fewer wires than cap 26 for only 0.010 percent higher qubit-time.

On the U7 head the circuit has 1,249 wires, 9,710,374 scheduled ticks and T 43,753,913.

### Weighted compare block bounds

Block zero runs its carry chain once each way, while the later blocks also undo and redo their ladders around two carry flips.
Weighting block zero by 400 and block j by 200 + 9 times (2j minus the block count) balances these unequal costs.
Cuts round each cumulative weight's share of the width half up, then clamp to retain nonempty blocks and both endpoints.
At 255 bits and seven blocks the cuts are 0, 64, 88, 116, 146, 180, 216 and 255.
The comparison contract requires clean borrowed scratch at entry; moving a cut preserves that contract, not arbitrary dirty-wire states.
On the U1 head the circuit has 1,246 wires, 16,075,064 scheduled ticks and T 32,268,901.

On the U7 head the circuit has 1,246 wires, 10,236,496 scheduled ticks and T 43,753,913.

## Depth pass

`write_circuit` runs `qec/depth_pass.py` on the hierarchical branch after the definition ids are fixed.
Per definition, in emission order, it sorts each CCX's controls, list-schedules the definition's commutation DAG forward and backward under the board's lowering, and keeps the input unless a schedule has fewer standalone forward plus inverse ticks.
It then gives each CCX the control order its own standalone forward and inverse runs prefer.
It reads definition bodies only, so definitions stay identical over addends, and a second application returns its input.
The flat branch never changes, so `check_hier.py` checks the builder with `depth_pass=False`, and `receipts/baseline/flat_ops.sha256` does not move.
At G the pass takes the board's ticks from 28,547,627 to 26,332,479 with the same wires, CCX, ops, call records and bytes.

## Out-of-place Euclid unwind

When at least three clean tape wires are available, the unwind copies carries onto a ring of at most seventeen wires; otherwise it retains the existing unwind.
The restore chain undoes each MAJ through U[i-1], and a following CX from restored U[i+1] fixes the temporary a_i XOR a_(i+1).
Each bit selects its outputs using the saved carry, leaving that copy as c_i.
The uncopy clears K_(i+1) with MAJ(A_i, B_i, c_i), where A_i is U_out XOR g1 and B_i is V_out XOR g0.
This majority equals c_(i+1) on each reachable branch: g0=0, g0=1 with g1=0, and g1=1.
The final carry copy is cleared from U[0] while it holds c1, and the control copies and parity wire are uncomputed.
Two carry slots suffice for the pipelined descending chains; the remaining ring wires hold the parity and repeated controls to reduce serialization.
The ring uses only the pass's available clean wires and restores them before the rotation reuses them.
On the rotation baseline, U1 keeps 1,246 wires and gives 25,670,489 ticks, 3,820,863 CCX and 26,746,041 T, with 13,759,163 expanded operations.

On the integrated compare-and-depth-pass head, U1 gives 1,246 wires, 20,776,137 scheduled ticks and 31,507,497 T.

## Top bit without a carry output

For an uncontrolled add without a carry output, the top majority and unmajority cancel to b XOR a XOR the incoming carry.
The lower bits retain their majority chain and the top bit uses two CX gates.
Controlled adds and adds with a carry output retain their existing chain.
This removes 44 CCX and 88 operations from the full circuit, leaving 14,808,479 operations and T 32,869,137.

## Lookahead rows in the square

The carry recurrence f_j = x_j XOR p_j f_(j-1) uses blocks of increasing size from each end, capped at eight.
Each block computes its local prefix and product before receiving the incoming carry, then clears every temporary.
The square rows borrow the clean workspace after the two product registers.
Their add computes an up-sweep, selects the sum, and reverses the same sweep using the equal carries of a + b and a + complement(sum).
The square retains the chain implementation and leaves its borrowed workspace clean.
The full circuit has 15,306,887 operations, 4,998,159 CCX and T 34,987,113.

## Tree sweeps in the Euclid difference

The shared difference uses a ring of at most 193 clean tape wires for its carry sweeps and repeated controls.
The adjacent XOR network exposes the same inputs as the serial ripple with logarithmic depth.
A Brent–Kung tree links block prefixes and clears its segment products before fixing the lower bits of each block.
The sweep selects the smallest fitting block size from three, four, five, six and eight, falling back to the carry chain when necessary.
On the step domain U is odd, g0 equals the low bit of V, and g1 and the ring start clean.
The middle selection and reversed sweep produce the original unwind map and restore the ring before reuse.
Smaller rings retain the existing unwind implementations; the square keeps its lookahead chain.
The full circuit has 1,246 wires, 10,475,194 ticks, 6,336,351 CCX, T 44,354,457 and 17,125,135 operations.

## Narrow-width ripple in the Euclid difference

At widths up to 12 the shared difference runs as one in-place majority chain instead of a ring variant, as in 2elliot depth/v2 (9d34aef).
Each MAJ's two CXs run ahead of every CCX, so the carry reaches each CCX as its first control.
The selection moves into (c, v) before the carry arrives, by maps that keep c·¬v, which is all that the restore and the hand-down read.
At w = 12 with the ring of 10 that the pass hands it, the step costs 61 CCX against 133.
`scripts/euclid_step_equiv.py` compares it with the chain fallback on every input for w = 2..10, and with this tree's previous step on the step domain for w = 2..12.

## CDKM ripple for the tick's add

With `CDKM_ADD`, the tick's add is `add_or_shift_cdkm`, also from 2elliot depth/v2.
It is CDKM's ripple on every lane, with its sum writes gated on ¬J and moved off the carry chain; on J the shift is two reversals whose swaps follow the down pass.
It costs 3(n − 1) + 5 + n CCX a call, and docs/mux-proof.md S9, S10 and S19 give the argument.
`scripts/add_or_shift_check.py` checks it, and the lookahead add, on all 2^(2n+3) inputs for n = 4..10.
It spends fewer CCX and more ticks than the blocked-lookahead add, which stays behind `LOOKAHEAD_ADD`.
The full circuit has 1,249 wires, 6,620,707 CCX, T 46,344,949 and 7,866,220 ticks.
After `qec/gate_pass.py` it has 7,806,303 ticks, against 6,272,299 for this tree's previous head after the same pass: 580,120 fewer CCX for 1,534,004 more ticks.

## Idle wires for the tick's prefixes and borrowed carriers

The builder hands out the most recently freed clean wire, so P's tops came from the add's J copies, which the add holds until its last CX, and the first chain borrowed the H wires P's last block reads last.
With `IDLE_WIRES`, P and M allocate, among the clean wires the tick has touched, the one the tick's ops so far free first under the board's tick model, and `take` borrows the H wire that comes free first.
Which clean wire a top gets is free, and so is which H wire the first chain borrows: every H wire reads 1 when P's top is 1, and the chains reverse their own XORs on P = 0.
A fresh wire is taken only when no touched clean wire is free, as the stack would, so the tick touches no more wires.
`scripts/check.py --toy-bits 4 6 7 8 9 10 --toy-per-size 6` passes on all 36 toy circuits, 6 of which the flag changes; handing out a live wire fails 6 of them.
The full circuit has 1,249 wires, 7,200,859 CCX and 6,221,117 ticks, 109,569 fewer than before, and the same CCX.
After `qec/gate_pass.py` it has 7,199,803 CCX and 6,161,800 ticks, against 6,270,603: qubit-time 7,696,088,200, 1.74% lower.

## Ten P blocks

P's write starts with its top block's ladder, so on the critical path its length counts once in full, and every other block's tree and its reverse cost 2 CCX a bit.
At 221 bits P now has ten blocks, against nine ending in 29.
They were (3, 7, 14, 30, 31, 30, 29, 27, 25, 23) with a 40-bit compare block 0; after the next section's carriers the tick's ticks stop tracking the top block's size (several shapes give the same total), so (3, 7, 14, 30, 29, 29, 28, 27, 26, 26) costs 4,596 fewer CCX at fewer ticks, and block 0 goes back to 38 bits.
A top block of 28 or more bits costs about 340,000 more raw ticks in a screen of the Bezout pass alone (both directions summed), not in the full circuit.
The tenth top is live at the tick's peak, which costs one wire.
The add runs before P is built, so it takes 6 temps and 8 J copies inside the same peak instead of 5 and 6.
`scripts/prefix_check.py` checks `PrefixR` on all 2^(h+1) inputs for every size list it accepts at h = 2..12, and at 221 bits with these sizes on all ones and every lowest-zero position with 24 random fillings above, for t = 0 and 1.
`scripts/compare_check.py` checks `compare_blocks` on all inputs and borrowed-wire values for every bound list with up to 4 blocks at widths 2..6, and at width 255 with these cuts on equal words and on every first-differing bit, both orders, with 16 random fillings below.
Dropping one deferred CX from P's write fails the first check; dropping the top block's dirty carry-in fix fails the second.
The full circuit has 1,250 wires, 7,203,157 CCX and 6,174,044 ticks.
After `qec/gate_pass.py` it has 7,202,101 CCX and 6,110,534 ticks: qubit-time 7,638,167,500, 0.75% below the idle-wires head at 2,298 more CCX.

## P's chains on the bits below each block

`PrefixR` built every block's chain on the lowest wires below it, so the lowest H bits carried every block and came free only after the whole build.
With `HIGH_CARRIERS`, a block whose start is at least its length less one puts its chain on the bits just below it instead.
Every wire below the block reads 1 when the block's lower top is 1, so the argument in `Prefix` holds for any such choice.
The lowest bits then come free once the first blocks are built, the first chain borrows them, and in M's inverse P's unbuild starts before they come back.
`scripts/prefix_check.py` passes on all 873 size lists for h = 2..12 and at 221 bits; putting one carrier above the block fails it.
The full circuit has 1,250 wires, 7,203,157 CCX and 6,164,742 ticks, 9,302 fewer.
After `qec/gate_pass.py` it has 7,202,101 CCX and 6,106,977 ticks: qubit-time 7,633,721,250, 0.06% lower.
With P at (3, 7, 14, 30, 29, 29, 28, 27, 26, 26) and compare block 0 at 38 bits, the full circuit has 1,250 wires, 7,204,689 CCX and 6,148,277 ticks.
After `qec/gate_pass.py` it has 7,203,633 CCX and 6,091,268 ticks: qubit-time 7,614,085,000, 0.26% lower, at 1,532 more CCX.

## Codec calls off the tick chain

`_aside` runs each of the Bezout pass's block codec calls on the free wires freed longest ago, and leaves the wires it frees under the rest of the free stack.
Before, a decode or encode between two ticks took the scratch the tick before it had just freed, and the tick after it took the codec's wires, so each codec call sat on the tick chain: about 360 ticks for each of 108 calls a pass.
The allocator hands out only clean wires in any order, so the change permutes which clean wire each fresh slot gets and nothing else; wires and CCX are unchanged.
After `qec/gate_pass.py` the circuit has 1,249 wires and 6,204,501 ticks, against 6,270,603.

## Rotating encoder wires in the Euclid sweep

The Euclid pass's down-sweep encodes its blocks back to back, and every encode took the same two clean wires from the top of the free stack, so each one waited for the one before it to return them.
Block j's encode now holds `ANCILLAS * (j % ENCODE_SETS)` free wires around itself, so with `ENCODE_SETS = 3` three consecutive encodes use three different pairs.
The held wires are clean and freed unchanged, and the extra wires sit below the Bezout pass's high-water mark, so wires and CCX are unchanged.
After `qec/gate_pass.py` the circuit has 1,249 wires and 6,169,869 ticks, against 6,204,501; two sets give 6,174,029.

## Bit-level tree in the Euclid difference

`lookahead_sweep` now tries blocks of 1 and 2 before 3, through `SWEEP_BLOCKS`.
Blocks of 1 make `lookahead_tree` a plain Brent–Kung prefix over the bits, with each p read from its V wire, so the block-local prefixes, products and fixes disappear.
It needs about len(U) temps; every narrowed step's ring of w - 2 wires has them, and the 193-wire ring at w = 256 does not, so those steps keep blocks of 3.
In isolation the step at w = 197 goes from 2,382 CCX and 599 + 609 standalone ticks to 2,110 and 541 + 516, and every measured width from 14 to 197 gains on both.
`scripts/sweep_check.py` checks the tree for r = 1, 2, 3 on all inputs up to n = 11, and the tree step against the old block sizes on the step domain for w = 9..12; dropping the tree's last CCX fails it.
After `qec/gate_pass.py` the circuit has 1,249 wires, 6,147,009 ticks and 7,080,363 CCX, against 6,169,869 and 7,199,803.

## Wide ring for the full-width steps

`_ring_size` gives a step the bit-level tree's `tree_temps(w + 1, 1)` ring wires when that is more than the usual `U6_RING` cap and the pass has that many clean tape wires; `WIDE_RING` turns it off.
At w = 256 that is 247 wires, which the pass has for about the first hundred steps, so those steps run the bit-level tree instead of blocks of 3: 2,765 CCX against 3,099, and 629 + 629 standalone ticks against 672 + 678.
Narrower steps keep their rings, so only one more definition appears; `_shared_difference` now takes the ring it is given, since the pass alone caps it.
After `qec/gate_pass.py` the circuit has 1,249 wires, 6,128,027 ticks and 6,939,075 CCX, against 6,147,009 and 7,080,363.

## Tree rows in the square

With `ROW_TREE`, each of the square's rows runs `add_lookahead` with `tree=True`: the up-sweep's adjacent XORs are `adjacent_xor`'s network instead of a serial CX chain, and its carries are `lookahead_sweep`'s tree instead of the blocked chain.
The rows take `ROW_TEMPS = 80` clean wires beside the fold's own 101; the square runs while the multiplier's wires are free, so the wire count stays 1,249 (120 would make it 1,250).
The rows cost more CCX this way: 272,798 a square against 221,818, so 101,960 more for the circuit.
`scripts/row_add_check.py` checks both forms on all inputs for w = 1..8 and temps 1, 2, w and 2w; running the tree form's adjacent XORs in reverse order fails it.
After `qec/gate_pass.py` the circuit has 1,249 wires, 6,047,310 ticks and 7,041,035 CCX, against 6,128,027 and 6,939,075: about 0.79 ticks for each CCX spent.

## Both sets of changes together

The sections from "Idle wires for the tick's prefixes and borrowed carriers" to "P's chains on the bits below each block" were measured on one branch from the lookahead-add head, and the sections from "Codec calls off the tick chain" to "Tree rows in the square" on another, so each one's numbers are against its own branch.
Together, the full circuit has 1,250 wires, 7,044,865 CCX (T 49,314,055) and 5,916,318 ticks.
After `qec/gate_pass.py`, which cancels no CCX here, it has 5,867,922 ticks: qubit-time 7,334,902,500, against 7,831,983,147 for the lookahead-add head and 7,909,085,166 for PR8.
The two sets touch different functions, and their tick savings add up to within 53 ticks.

## Blocks of 3 in the add's forward sweep

The add's forward sweep F advances its carry one block per CCX, and the backward sweep B undoes it one block per CCX, so each block is one step on the critical path: 7 ticks where a step's target is the next step's first control, 11 where a step's control is the next step's target.
In the tick, F's steps cost 7 and B's 11; in M's inverse they swap, so a step saved in either sweep saves 18 ticks over the pass.
An F block of r bits runs the local carries up the block, the products p_s ... p_(s+i) on r - 1 temps, the chain step, and then fixes the inner bits top down from those products: 4r - 3 CCX.
Each chain step saved therefore costs 3 CCX a call for any r, against 18 ticks.
With `LA_F_WIDE = (3, 68)`, F has 68 blocks of 3 among 24 blocks of 2 and three ripple steps (93 steps, against 127).
A block of 3 holds two temps for about 90 ticks around its step, against one for about 30, so F alone ran short of temps.
The add's `carry` and `t` are clean until its top and every temp is returned clean, so `LA_TOP_TEMPS` lends them to the sweeps, and `LA_IDLE_TEMPS` hands out the temp that comes free first under the tick model instead of the oldest returned.
With 8 temps and 6 J copies (against 6 and 8) the add still fits the tick's peak.
`scripts/add_or_shift_check.py <tree> 10 wide` runs every composition of n - 1 into F blocks of 1 to 4 bits, 428 layouts, on all inputs; dropping the inner fixes above a block's first bit fails 27 of the 56 layouts at n <= 7.
The full circuit has 1,250 wires, 7,122,997 CCX (78,132 more, T 49,860,979) and 5,716,356 ticks, 199,962 fewer.
After `qec/gate_pass.py` it has 5,666,078 ticks against 5,867,922: qubit-time 7,082,597,500, 3.44% lower.

## Blocks of 3 split between the sweeps

B runs F's blocks backwards, which undoes the carries because f·V = f·p on every lane: V differs from p only where ¬J·¬f is 1.
Each block's first inner unfix needs no temp and still goes ahead of the chain; a block of 3 unfixes its second inner bit from the product it builds for its chain step.
F's steps cost 7 ticks in the tick and B's cost 11, and M's inverse swaps them, so the sweep that runs at 7 a step, and starves for temps, is F in one direction and B in the other.
`LA_B_WIDE = (3, 34)` with `LA_F_WIDE = (3, 34)` puts half the blocks of 3 in each, at the same CCX as 68 in F.
In a screen of the Bezout pass, 68 in B alone reaches its computed ticks in the tick's direction and starves in the inverse, and clustering the blocks at the top or bottom of either sweep costs 43,000 to 53,000 ticks.
`scripts/add_or_shift_check.py <tree> 10 wideb` runs all 428 B layouts on all inputs; dropping the second inner unfix fails 27 of the 56 layouts at n <= 7.
The full circuit has 1,250 wires, 7,122,997 CCX and 5,705,368 ticks, 10,988 fewer.
After `qec/gate_pass.py` it has 5,656,726 ticks: qubit-time 7,070,907,500, 0.17% lower.

## M's restored top on an idle wire

The second chain drops chain[b] just before `M.restore()` allocates the top it rebuilds, so the stack handed the restore that carrier, and the restore waited for the carrier's uncompute.
With `RESTORE_IDLE` the restore takes the touched clean wire that comes free first under the tick model, as P's and M's tops do; the allocator still hands out only clean wires.
The toy circuits never release M's top, so the check is M itself at secp256k1 (`block_check.run` on 64 random x and 0, 1, 2, q - 1, q - 2, forward and backward): putting the restore on a live wire fails all 69 lanes, and the tree fails only x = 0, as e18f122 does.
After `qec/gate_pass.py` the circuit has 1,250 wires, 7,122,997 CCX and 5,649,382 ticks, 7,344 fewer: qubit-time 7,061,727,500.
In a screen, giving each value in the second chain a wire of its own would save up to 107,000 raw ticks, but each such wire is one more circuit wire; choosing among wires already touched, by the tick model in one or both directions, did not find it.

## P's blocks after the faster add

With the add's blocks of 3, P's ten blocks retune to (3, 7, 14, 28, 28, 28, 28, 28, 28, 27), against (3, 7, 14, 30, 29, 29, 28, 27, 26, 26): the fourth block shrinks to 28 and the rest even out.
A block is hosted on the bits of the block below only when that block has at least its size less one, so a fourth block of 26 or 27 under blocks of 29 loses hosting and costs about 330,000 ticks in a screen of the Bezout pass.
`scripts/prefix_check.py` passes at h = 2..9 and at 221 bits with these sizes; dropping one deferred CX from the write fails 5 lanes at 221 bits.
After `qec/gate_pass.py` the circuit has 1,250 wires, 7,121,465 CCX (1,532 fewer) and 5,628,639 ticks, 20,743 fewer: qubit-time 7,035,798,750.
The CCX that P's retune saves pay for a 35th block of 3 in B (`LA_B_WIDE = (3, 35)`, 2,298 CCX): B then has 111 steps and F 112.
After `qec/gate_pass.py` the circuit has 1,250 wires, 7,123,763 CCX (78,898 more than e18f122, T 49,866,341) and 5,621,253 ticks, 7,386 fewer: qubit-time 7,026,566,250, 4.20% below e18f122's 7,334,902,500.

## Trees for the top-level constant adds

`Mod.cmodadd_const` and `Mod.creflect` run three 256-bit carry chains each: a comparison, the selected-constant add, and the comparison that clears the flag.
The kernel runs five of them on the critical path, about 80,000 ticks in all.
With `CONST_TREE`, `gt_const_tree` joins the carry out of ~b + value by a balanced tree of segment joins (`_carry_tree`) on fresh wires, and `_add_selected_tree` copies the selected constant onto clean wires by doubling fans and adds it with `add_lookahead`'s tree; creflect's complement of b fans the control out through `REFLECT_COPIES` copies.
Each call takes about 513 scratch wires, and the top level has more than 730 free, so the wire count stays 1,250.
In isolation one call goes from about 18,000 ticks and 1,530 CCX to about 1,110 ticks and 4,000 CCX.
`scripts/const_tree_check.py` checks `gt_const_tree` on every input and constant up to 8 bits, and both kernel calls on every odd q below 256, every constant and every canonical b; dropping the tree's last join fails it.
After `qec/gate_pass.py` the circuit has 1,250 wires, 5,794,237 ticks and 7,057,205 CCX, against 5,867,922 and 7,044,865: about 6 ticks for each CCX spent.

## Trees for the exception flags

`_eq_flags` ANDs X's bits into the four exception flags by chains of CCX, one bit after another, at the start and at the end of the circuit: about 21,000 ticks on the critical path.
With `EQ_TREE`, `and_tree` builds each AND as a balanced tree on fresh wires instead, with the same CCX, and `eq_const_into` joins its control last, since each control is itself a tree's root and is ready last.
`scripts/eq_tree_check.py` checks `eq_const_into` on every input for every value up to 9 bits, with and without a control, and `and_tree` up to 9 wires; dropping the tree's last CCX fails it.
After `qec/gate_pass.py` the circuit has 1,250 wires, 5,774,637 ticks and 7,057,205 CCX, against 5,794,237.

## Copies for the square's row fans

Each of the square's rows starts with a Gray-code step that XORs Y[i] into every wire above it, n - 1 - i CXs from one wire, and the rows end with a chain of n - 1 CXs that undoes the walk: about 28,000 ticks a square, all in series.
With `ROW_FAN_COPIES`, `fan_into` copies Y[i] onto at most that many clean row temps by doubling, XORs the copies into the targets in parallel and clears them, with the copy count `_fan_copies` picks for the fewest layers; the temps it uses sit at the back of the queue the row's tree takes from.
`prefix_xor` undoes the walk as a Brent-Kung prefix network.
Wires and CCX are unchanged.
`scripts/square_check.py` checks `SquareFold.compute` on every n-bit Y for every odd q below 2^11 the fold applies to, and `prefix_xor` and `fan_into` on every input; dropping the prefix network's last CX fails it.
After `qec/gate_pass.py` the circuit has 1,250 wires and 5,730,583 ticks, against 5,774,637.

## More row temps

`ROW_TEMPS` goes from 80 to 120. The square runs while the multiplier's wires are free, 121 below the 1,250-wire peak that the Bezout tick now sets, so the extra temps cost no wire.
With 221 temps the narrower rows run the bit-level tree instead of blocks of 2 or 3, which saves CCX as well as ticks.
After `qec/gate_pass.py` the circuit has 1,250 wires, 5,725,277 ticks and 7,032,405 CCX, against 5,730,583 and 7,057,205.

## Trees for the fold's adds

After the rows, the square folds 2^n to f with a 256-bit add, ten adds of f's signed digits times hi and E (five of about 290 bits, five of about 65), and one 65-bit add, all CDKM ripples: about 35,000 ticks a square on the critical path.
With `FOLD_TREE` each runs `add_lookahead`'s tree, on `FOLD_TEMPS = 86` clean wires plus whichever of the fold's junk wires are still clean at that point (E, T, t, c and k for the first add, T, t, c and k for hi's digits).
`add_lookahead` now takes a carry-in wire, so the first digit still adds Z's + 1. The square's block peaks at the same 121 scratch wires as the rows, so the wire count stays 1,250; 88 temps would make it 1,251.
The adds cost more CCX: 287,300 a square against 272,798 in isolation.
`scripts/square_check.py` covers the fold's trees; dropping the first add's carry in fails it.
After `qec/gate_pass.py` the circuit has 1,250 wires, 5,652,708 ticks and 7,061,409 CCX, against 5,725,277 and 7,032,405: about 2.5 ticks for each CCX spent.

## Borrowed ring wires for the wide Euclid steps

From w = 255 down to about 207 the Euclid pass has 197 clean tape wires a step, and the last 25 steps at w = 256 have 197 to 245, so those steps ran blocks of 3 on a 193-wire ring while the bit-level tree needs 198 to 247.
The pass runs below the Bezout tick's peak, so with `RING_BORROW = 18` a step whose tree needs at most 18 wires more than the tape has allocates them for itself and frees them after; 20 would make the circuit 1,251 wires.
About 24 steps take the bit-level tree this way, which saves CCX as well as ticks.
`scripts/ring_borrow_check.py` runs the multiplier on every x below q for q = 1021, 4093 and 16381, forward and inverse, with the borrow on (and with `U6_RING = 0` so that most steps borrow); flipping the borrowed wires to 1 fails it.
After `qec/gate_pass.py` the circuit has 1,250 wires, 5,646,628 ticks and 7,025,145 CCX, against 5,652,708 and 7,061,409.

## Trees for the first Euclid step

`_euclid_tick0` runs the first Euclid step on U = q: 255 CCXs that all read g0, a 256-bit constant add under g0 as a carry chain, 256 CXs from g0, and a rotation of V as 254 cswaps in series; about 11,000 ticks each of the four times it runs.
With `TICK0_TREE` it takes `TICK0_SCRATCH = 450` wires of the unwritten tape instead of 254 (the pass has 455 at that step): the CCXs read `TICK0_COPIES` copies of g0, the add is `_add_selected_tree` with its temps capped to fit, the CXs go through `fan_into`, and the rotation is `_par_rotate`, as in the later steps.
In isolation the step goes from 10,981 to 693 ticks and from 1,018 to 2,836 CCX.
`scripts/tick0_check.py` checks the tree form against the chain form for every odd q from 9 to 1023 on every V, with 1, 3 and 31 copies, and that the secp256k1 block uses no wire outside its registers, without which the pass would skip it; rotating one pair too few fails it.
After `qec/gate_pass.py` the circuit has 1,250 wires, 5,606,298 ticks and 7,032,417 CCX, against 5,646,628 and 7,025,145.

## Trees for the square's modadd

`SquareFold.modadd` adds the square into X under the generic flag with a controlled 256-bit CDKM ripple and clears its flag with a 256-bit `gt_into` ripple: about 16,000 ticks on the critical path.
With `MODADD_TREE`, `cadd_lookahead` runs `add_lookahead`'s tree up-sweep and its reverse, and puts only the middle (the sum's CXs and both X layers) under the control, through `MODADD_COPIES` copies of it; with the control at 0 the up-sweep meets its own reverse and leaves both registers as they were.
`gt_lookahead` reads the comparison from the carry out of the same up-sweep on (a, ~b).
Both take `MODADD_TEMPS = 95` clean wires, which with the flag, its scratch bit and the copies stays within the 121 the square leaves free.
`scripts/square_check.py` now also checks `SquareFold.modadd` on every canonical a, b and control for every q below 2^8 the fold applies to, and both helpers on every input up to 6 bits; leaving the second X layer uncontrolled fails it.
After `qec/gate_pass.py` the circuit has 1,250 wires, 5,596,014 ticks and 7,035,319 CCX, against 5,606,298 and 7,032,417.

## Round 2's arm D together

The sections from "Trees for the top-level constant adds" to "Trees for the square's modadd" were measured in order on one branch from `depth/v4` (e18f122), each against the section before it.
Together, after `qec/gate_pass.py`, the circuit has 1,250 wires, 5,596,014 ticks and 7,035,319 CCX (T 49,247,233): qubit-time 6,995,017,500, 4.63% below `depth/v4` at 7,334,902,500, with 9,546 fewer CCX.
Two attempts were measured and dropped: lookahead chains for rows of 32 bits or fewer (2,426 more ticks in the circuit, though fewer in isolation), and B1 marks made per cell on their own wires in the Euclid sweep (11,552 more ticks, because consecutive marks share a g wire).

## Round 2 together

The sections from "Blocks of 3 in the add's forward sweep" to "P's blocks after the faster add" were measured on one branch from the combined head above, and the sections from "Trees for the top-level constant adds" to "Round 2's arm D together" on another, so each one's numbers are against its own branch.
Together, the full circuit has 1,250 wires, 7,114,217 CCX (T 49,799,519) and 5,387,028 ticks.
After `qec/gate_pass.py` it has 5,349,345 ticks: qubit-time 6,686,681,250, against 7,334,902,500 for the combined head above and 7,909,085,166 for PR8.
The two branches touch different functions, and their tick savings add up exactly.

## Changes from 2elliot's and ab-godder's entries

This branch builds on 2elliot's `cold-trees` entry (`elliot/comp-2elliot-pr12-b2284da7c963`), which built on ab-godder's `tick-parallel` entry (`mike/comp-ab-godder-pr8-c4994f3b4b23`), which built on the `v2-depth` entry.
It keeps every change of both entries, and each is its author's work.
From ab-godder's entry it keeps the five commits that change the Bézout tick:
- the compare runs in up to 11 blocks on the tick's freed scratch, with cuts tuned to when wires come free (`0db514b`);
- the tick's add is a blocked-lookahead chain (`510ac23`);
- p is un-copied after the add in the order the rotation finishes the word's bits (`8cc7563`);
- the tick's P is `PrefixR`, whose write runs its blocks in parallel (`7163a75`);
- each Bézout tick hands its B1 mark to the next tick, and the final swap's wires come late (`6692a5e`).
From 2elliot's entry it keeps all 28 commits from `116a377` to `435d287`, which the sections from "Narrow-width ripple in the Euclid difference" to "Round 2 together" describe.
They include the post-pass `qec/gate_pass.py` (`116a377`), which every figure the board scores runs through, and the blocks of 3 in the tick add's sweeps (`66504b0`, `fcc7a30` and `241d751`).
At G that entry is 1,250 wires × 5,349,345 ticks after `qec/gate_pass.py`, 7,114,217 CCX and T 49,799,519.
The section after this one describes the change this branch makes on top of it.

## The tick's probe as a carry-out tree

The Bézout tick's first chain is a probe: it computes e = P.top * (the carry out of L plus the selected constant), and leaves every other wire as it was.
At G it rippled 15 `_selected_carry` steps up, 12 below the quiet run and 3 above it, with one CCX across the quiet run through M, took one CCX into e, and rippled the same steps down.
`_probe_tree` in `qec/two_bit.py` computes the same carry out by a tree.
For each bit it puts the XOR of the active selectors on a temp k, forms g = L * k, and sets L ^= k, so L holds p; g and p are never both 1.
The low bits below the quiet run merge in a balanced tree, with G = Gr ^ Pr * Gl and P = Pr * Pl formed only where a later merge reads it.
M's top carries the result across the quiet run, the top bits ripple, e takes its CCX, and the recorded gates run backwards.
Its temps come from the same `take()` the ripple used, which borrows the H wire that comes free first with an X and returns it with an X.
That is sound because e can be 1 only where every H wire reads 1, so a borrowed wire is clean exactly where the result matters, whichever H wire it is.
P's and M's tops come from the tick's idle allocator, which hands out only clean wires, so the tree never borrows a top.
`I2.probe` (`scripts/probe_check.py`) checks the tree and the ripple on every lane at seven secp-shaped primes, in the tick's probe context, with controls for a borrow without its X and for a broken merge, and checks that the broken merge fails the exhaustive toy check.
`I2.post-pass` (`scripts/post_pass_check.py`) checks the tree under the post-pass the board scores: 40 circuits, the toy rows and four 14- to 17-bit merged-path primes, are exact before and after `qec/gate_pass.py`, and one wrong cancel among the tree's gates is caught on 16 of the 22 that build the tree.
That is a claim for the tree's code only; the pass and the rest of the circuit it acts on are 2elliot's.
At G the pass cancels 21,344 pairs, against 21,312 on 2elliot's entry: 8 of them pair two of the tree's own gates, and 288 more pair an X of `take()` or `drop()`, which borrow and return H's wires while the tree runs, with another gate.
The tick's second chain writes L, so it needs every carry and keeps the ripple.
At G, after `qec/gate_pass.py`, the circuit is 1,250 wires × 5,246,467 ticks, 7,144,857 CCX and T 50,013,999: 102,878 fewer ticks for 30,640 more CCX, and qubit-time 6,558,083,750, 1.92% below 2elliot's entry.

## A 217-wire cap on the Euclid difference's ring

`U6_RING` in `qec/point_add.py` caps the ring of clean tape wires that the Euclid difference's sweeps use; it is this branch's constant, which came in at 193 with `d9ecbbe`.
`WIDE_RING` and `RING_BORROW`, which decide when a step takes the bit-level tree's wider ring instead, are 2elliot's.
With the cap at 217, a step that the cap held to 193 wires takes up to 217 of the clean wires the pass has.
217 is the best of the caps we ran from 193 to 289, not a proven optimum.
`I2.ring` (`scripts/ring_k_check.py`) runs the Euclid step at widths 9 to 12 at every ring size from w - 2 to 2w, 54 circuits, on every input of the step domain, as built and after `qec/gate_pass.py`; each must equal the step at w - 2 and leave the ring and scratch clean.
`I2.ring-control` starts the ring's last wire at 1, and the check must fail at every size in both forms.
That exhaustive check reaches the step's code at every ring size only at widths 9 to 12; at 256 bits, the ring sizes the cap changes are reached only by the sampled checks.
At G the pass cancels 22,340 pairs, against 21,344 with the cap at 193, and 4,357 of them are in the 53 definitions the cap changes.
At G, after `qec/gate_pass.py`, the circuit is 1,250 wires × 5,244,059 ticks, 7,144,857 CCX and T 50,013,999: 2,408 fewer ticks at the same wires and CCX, and qubit-time 6,555,073,750, 1.97% below 2elliot's entry.

## More blocks of 3 in the tick's add

The tick's add, its blocks of 3 in both sweeps, `_la_wide`, which spreads them, and the form of its counts and temps (`LA_F_WIDE`, `LA_B_WIDE`, `LA_NTEMPS`) are 2elliot's; the sections from "Blocks of 3 in the add's forward sweep" to "P's blocks after the faster add" describe them.
This branch raises the counts from 34 in F and 35 in B to 63 and 62, and the temps from 8 to 14.
At 256 bits F then has 97 chain steps and B 98, against 112 and 111, and each step saved costs 2,298 CCX, 3 for each of the add's 766 calls.
That spends the T the probe tree leaves: the scored T is 50,464,407, 1,526 under the reference's 50,465,933, so one more step saved would not fit.
Each temp past 8 adds a wire, and 14, at 1,256 wires, was the best of the counts we ran from 8 to 18, not a proven optimum; the effect is not monotone in the count.
For every n below 108 these layouts equal 2elliot's.
So the exhaustive checks reach the add's block types and adjacencies only at n <= 7: `scripts/add_or_shift_check.py` runs every layout there, and `I2.add-reuse` (`scripts/add_reuse_check.py`) runs those layouts with the add's temps cut to one, so that blocks take temps that others returned, with a control that returns them dirty and must fail exactly the layouts that reuse one.
The 256-bit layout itself is reached by `I2.add256` (`scripts/add256_check.py`), which checks the add alone at n = 256 on every carry run, all-0 and all-1 inputs and 2^16 random lanes, at J = 0 and J = 1, as built and after `qec/gate_pass.py`, in which the pass cancels no pair, with a control that drops one inner fix; and by the sampled 256-bit rows.
In the scored circuit the add is inlined in the tick's definitions, where the pass can cancel across its boundary, and at 256 bits those cancellations are reached only by `qcb` and the fresh roots on the post-passed method.
At G the raw emission has 7,210,499 CCX and T 50,473,493, 7,560 over the reference; `qec/gate_pass.py`, which is 2elliot's, cancels 1,298 of those CCX, so the scored method has 7,209,201 CCX and T 50,464,407.
At G, after `qec/gate_pass.py`, the circuit is 1,256 wires × 5,055,456 ticks: 188,603 fewer ticks for 64,344 more CCX and 6 more wires, and qubit-time 6,349,652,736, 5.04% below 2elliot's entry.

## A wider search for the post-pass control

`I2.post-pass`'s control removes one pair of the probe tree's gates that the pass may not cancel, runs the pass, and stops at the first removal the exhaustive check catches; the row requires a catch on at least 16 of the 22 circuits that build the tree, which is what the probe tree's commit measured at 20 tries.
Commit 3's 14 temps change the toy and forced circuits' adds and so reorder each circuit's candidates, and at 20 tries the control then caught on 15; only forced16 changed, where the first 20 of its 640 candidates are harmless.
Lowering the floor to 15 would let the next reordering lower it again, so the search was widened instead: `scripts/post_pass_check.py` now tries up to 64 candidates a circuit, and the floor stays 16.
At 64 tries the control catches on all 22, the latest at its 48th try (forced16), with 337 tries that left the function unchanged, and all 40 circuits stay exact before and after the pass; its rows take 1,619 s in all, against 952 s at 20 tries.
This changes no circuit: at G the emission, the post-passed method and the figures are those of the section above.

## The Karatsuba square-subtract on the tree adders

`KaratsubaMulsub` in `qec/square.py`, the one-level Karatsuba square-subtract, is our construction, from our `lean` entry (pull request #11).
That is the split Y = L + beta H with beta = 2^h, h = n/2 and beta^2 = 2^n = f mod q, and its two groups: AC builds A = L^2 and C = (L + H)^2 and folds beta G + A' with G = C' - A', and B builds B = H^2 and folds 2^h B0 + f ~P with P = B' - B1.
It also takes in each group's fold to a canonical residue through `finish` and `canon`, and the bounded `_multiple`, which the class carries as its own method, since this branch's `SquareFold._multiple` has another signature and no bounds.
Building, applying and unbuilding half squares into X with beta^2 = f mod q follows [arXiv:2609.09582, section 5.3.4](https://arxiv.org/abs/2609.09582), and the Karatsuba modular square is [arXiv:2609.05625, section VII.2](https://arxiv.org/abs/2609.05625); the uncontrolled rows follow section VII.A of the latter, after Litinski.
Each group is a recorded block whose retained wires form an explicit junk register; one controlled modadd subtracts the group's residue from X or adds it, and the block's inverse then restores Y and clears the junk.
The half-square offsets and the complement's -f (2^n - 1) collect into one constant K, which joins `_kernel`'s constant add of 3a under gen, as in that entry.

On this branch the class's adds run on 2elliot's adders, each with the same carry in and carry out as the add it replaces.
B's half square takes `SquareFold._rows`, and each AC half runs the same loop, `fan_into`, `add_lookahead`'s tree and `prefix_xor`, on its own 110 temps and its own carry in, so the two halves still share no wire.
The folds' f-multiples add by `add_lookahead` trees on 59 temps in AC and 123 in B, and each group's modadd is 2elliot's tree modadd, B's at the square's own shape and AC's at 44 temps and 7 copies, which `SquareFold.modadd` now takes as an optional shape.
`add_low`, `finish`, `canon` and the last add of B's half keep the ripple.
Five constants in `qec/square.py`, `KARATSUBA_ROW_TREE`, `KARATSUBA_AC_MODADD`, `KARATSUBA_B_MODADD`, `KARATSUBA_HALF_TEMPS` and `KARATSUBA_FOLD_TEMPS`, set these choices, and `I2.karatsuba` pins them.
Taken as it was, the class peaks at 1,315 wires here, because the AC group's junk stays live through the tree modadd's scratch; at 44 temps and 7 copies the AC modadd brings the circuit back to 1,256, and the AC build, the AC modadd and the B build each then peak at 1,256.
The point add takes the class at even n >= 128 with h + bitlen(f) + 3 < n, so at secp256k1, and keeps `SquareFold` below that.
At secp256k1 the class alone, with K paid, takes 1,253 wires and 431,620 CCX.

No toy row reaches the class, so it is checked at forced small primes: `I2.karatsuba` (`scripts/karatsuba_check.py` and `scripts/karatsuba_mutation.py`) runs every Y and every X at each canonical tail at q = 65,521, 262,121 and 262,139, its pieces at full width, twelve controls and 200 mutants, and `I2.karatsuba-post-pass` (`scripts/karatsuba_post_pass_check.py`) checks the class as built, as `qec/emit.py` writes it and after `qec/gate_pass.py` at those primes and at both full widths, with a wrong cancel that must be caught.
At 256 bits the class is reached only by those checks' seeded samples and by the sampled 256-bit rows.
The exactness claim is for the class's code on those inputs; the adders it now calls and the post-pass are 2elliot's.
At G the raw emission has 7,082,064 CCX and T 49,574,448, and the pass cancels 23,370 pairs, against 22,341 before; in the class alone at secp256k1 it cancels 3,389.
At G, after `qec/gate_pass.py`, the circuit is 1,256 wires × 5,112,574 ticks, 7,080,538 CCX and T 49,563,766: 128,663 fewer CCX for 57,118 more ticks at the same wires, and qubit-time 6,421,392,944, 1.13% above the previous commit and 3.97% below 2elliot's entry, with T 1.79% under the reference's 50,465,933.

## The tick's add at its counts' cap

The tick's add, its blocks of 3, `_la_wide`, which spreads them, and the form of its counts and temps are 2elliot's, as "More blocks of 3 in the tick's add" says.
With the T the Karatsuba square-subtract frees, this branch raises the counts from 63 in F and 62 in B to 84 in each, which is `_la_wide`'s cap at n = 256: each sweep is 84 blocks of 3 and 3 ripple steps, 87 chain steps against 97 and 98.
That is 21 steps, at 2,298 CCX each, 48,258 CCX; the temps stay 14, at 1,256 wires.
After them 564,361 T is still unused, so the layout's cap, not T, now stops the counts.
For every n below 108 these layouts still equal 2elliot's, so the exhaustive checks reach the add's block types and adjacencies only at n <= 7, as before.
The 256-bit layout is reached by `I2.add256` (`scripts/add256_check.py`), which now requires 84 and 84 and checks the add alone at n = 256 on every carry run, as built and after `qec/gate_pass.py`, with its control, and by the sampled 256-bit rows.
At G the raw emission has 7,130,322 CCX and T 49,912,254, 553,679 under the reference.
At G, after `qec/gate_pass.py`, the circuit is 1,256 wires × 5,000,829 ticks, 7,128,796 CCX and T 49,901,572: 111,745 fewer ticks for 48,258 more CCX, and qubit-time 6,281,041,224, 1.08% below the commit before the square-subtract and 6.07% below 2elliot's entry.

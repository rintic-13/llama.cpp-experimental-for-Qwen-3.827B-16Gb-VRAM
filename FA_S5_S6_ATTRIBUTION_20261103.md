# S5–S8 verdict + decode-kernel latency attribution — 2026-11-03 (branch closed)

Reference baseline: `FA_TURBO_BASELINE_20261103.md` (S2d, 3 blocks/SM, 168 regs, 4864 B smem).
This file records the prefetch experiments that followed it, the attribution probe that explains why
they all came out negative/neutral, and the table-based decode work (S7/S8) that finally moved the
kernel — including the production flip and its gates (§9).

Probe: `tools/fa_turbo_ab_probe.cu` (nq=1, nkv=32768, heads 24/4, D=256, med of 11 reps).
Raw logs: `s5_gate_20261103.log`, `s6_gate_20261103.log`, `s8_discrim_20261103.log`,
`s8_pick_20261103.log`, `ppl_s78_20261103.log`.

Tree: `ee7dcf0` at the time of the S5/S6 runs; at close (2026-11-03) HEAD is `dfcd625` with the S7/S8
work uncommitted — see §9 for the exact file list.
GPU: RTX 5060 Ti, `nvidia-smi` checked before every run below; all rows in §1–§4 were taken at
0 MiB used / 0 %.

## TL;DR

| experiment | idea | verdict |
|---|---|---|
| **S5** tile-level `cp.async` staging of the K tile into smem (1 tile ahead) | hide K load latency in shared memory | **REJECTED**: −34 % if allowed to keep its smem, −3.5 % even at equal occupancy |
| **S6** register-level fragment prefetch in the fused turbo4 KQ walk (depth 1) | hide K load latency in registers | **NEUTRAL** (1.061 → 1.066 ms), bitwise identical, no reg/stack change; kept behind an opt-in flag |
| **S6b** tile-level `prefetch.global.L1/L2` (zero regs, zero smem, dist 1 / 2 tiles) | hide K load latency without touching occupancy | **REJECTED**: +1…2 % on every distance/level combination |
| **S6c** self-row `prefetch.global.L1` at tile top (dist 0) | same, closest possible distance | **REJECTED**: +1.5 % |
| **S6d** one-turbo-block lead prefetch inside the fragment walk (dist 8/16 frag) | right distance, zero cost | **NEUTRAL** (1.064 / 1.065), bitwise; last point on the axis - K prefetch CLOSED |
| **TLP ceiling** `MIN_BLOCKS=4` (128 regs, 4 blocks/SM) | more warps instead of ILP | **WORSE** (+11 %): 3 blocks/SM is the optimum |
| **attribution** collapse K / V row strides (L1-hot) | find where the time actually is | K walk = **22 %**, PV walk = **0.0 %**, remaining 78 % = compute |
| **S7** byte-indexed turbo4 centroid table in smem (2 KB) | kill the 7-`FSEL` K decode mux | **ACCEPTED**: −21.7 % instructions, T6 −27.4 %, bitwise |
| **S8** byte-indexed turbo3 V centroid table (one 8 KB table, global) | kill the V decode mux + per-coord extraction | **ACCEPTED**: −17.9 % instructions → **−17.8 %** time (1:1), bitwise; T1 −16.2 %, T8 −15.7 %, C2 −11.4 % |
| **S7+S8** both tables | | **T6 −29.7 %** (0.9985 → 0.7025 ms), C2 −24.6 %, T7 −24.2 %; combination sub-linear (interaction, see §2f) |
| **§2g/§2h discriminators** dead-smem pad vs gather-collapse mask vs 32 B table | find the 10 missing points | **ROOT CAUSE: the value-indexed table read is a gather worth ≈13 % of the kernel.** smem footprint ruled out; `S7+S8` with the gather collapsed = **−37.1 %**; the bitwise-valid 32 B broadcast table gets **−34.4 %** |
| **§2i/§2j S8e: 512 B signed float2 table (`VTAB=4`)** | the sign bit belongs in the index (the magnitude tree reflects when the sign bit is clear) | **PICK: `CEN_TAB=1` + `VTAB=4` — T6 −40.8 %, C2 −34.4 %, best form on every row, bitwise. The per-instantiation auto rules (3/5) are rejected: they cost 9.9 % on the S7 rows** |
| **§2k flip + production gate** | ship the pick | **PASSED and shipped as defaults: every frozen hash identical, T6 0.591 ms (−40.7 %), PPL 7.1719 vs 7.1751 (Δ −0.0032 ≤ 0.05). BRANCH CLOSED** |

## 1. S5 — tile-level `cp.async` K staging

`GGML_CUDA_FATTN_VEC_CPASYNC=1` stages the 128-row K tile one loop iteration ahead into a static
double buffer (2 × 128 × 132 B = 33 792 B) and reads the A fragments from smem.

| row | S2d baseline 3 blk / 168 reg | S5 forced 3 blk → **occ 2** / 168 reg | S5 production / 255 reg, occ 2 |
|---|---|---|---|
| T1 t4–t3 dot | 1.091 | 1.098 | 1.105 |
| **T6 t4–t3 fused** | **1.060** | **1.448** (+37 %) | **1.417** (+34 %) |
| T7 t4–t4 fused | 1.058 | 1.476 | 1.455 |
| T8 t5–t3 fused (not staged) | 1.195 | 1.181 | 1.188 |
| C1 / C2 D128 | 0.503 / 0.491 | 0.507 / 0.481 | 0.503 / 0.491 |

The occupancy is set by smem, not by registers: 3 × (33 792 + 4 864) = 116 KB > the 100 KB/SM
budget, so `__launch_bounds__(…,3)` cannot be honoured and `cudaOccupancy…` also reports 2, which
feeds `launch_fattn`'s `parallel_blocks` (pb 9 → 3). Both register budgets give the same time, i.e.
the loss is purely the third block.

### 1b. Equal-occupancy A/B (the decisive one)

`-DGGML_CUDA_FATTN_VEC_MIN_BLOCKS=2` (no cp.async): 255 regs, occ 2, pb 3 — the pre-S2d policy —
versus the S5 build at the same occupancy:

| row | 255 reg, occ 2, **no staging** | 255 reg, occ 2, **staging** | delta |
|---|---|---|---|
| T6 t4–t3 fused | 1.369 | 1.417 | **+3.5 %** |
| T7 t4–t4 fused | 1.416 | 1.455 | +2.8 % |

So the staging does not pay even when the occupancy cost is removed: it adds a smem hop plus
33 `cp.async` issues per lane per tile and buys nothing, because the loads it prefetches were
already being overlapped. Other rows of that build reproduce the frozen pre-policy baseline exactly
(T1 1.413 vs 1.412, T3 1.445, T4 1.697, T8 1.502), which validates the A/B.

### 1c. Is the staging bitwise? Yes - the raw-fnv drift was a harness artifact

`s5/s5_pad_log.txt` (`fa_turbo_ab_pad38k.exe`: `SMEM_PAD=33792`, NO cp.async -> same occ 2 / pb 3 /
168 regs) versus `s5/s5_cpasync_mb3c_log.txt` (cp.async build rebuilt with the cfnv probe):

| row | pad (no staging) cfnv | cp.async (staging) cfnv |
|---|---|---|
| T6 | 640ed377327151de | **640ed377327151de** |
| T7 | df35af1a77583973 | **df35af1a77583973** |

Identical merged-output hashes at matched geometry: the cp.async transport is bitwise-exact. The
earlier raw-`fnv` disagreement (302f5921 vs 73d4a5f0) is fully explained by the pb-dependent
partial-slot layout of the raw dst/meta hash (§6) - at matched pb=3 the two staging builds even
share 302f5921. The `SMEM_PAD` asm-barrier detail: a plain dead read/write of a `__shared__` array
gets deleted by ptxas (first pad run showed 4864 B smem); leaking its address through
`asm volatile("" :: "r"(cvta_shared))` keeps the allocation.

## 2. S6 — register-level fragment prefetch (fused turbo4 K, D=256, ncols=1)

The fused walk is `LDG Kb0/Kb1 → decode+dither → mma` per k-fragment, 16 fragments × 2 halves.
`ggml_cuda_fattn_turbo_kq_afrag` was split into `…_afrag_load` (raw bytes + norms) and
`…_afrag_build` (decode + dither + fragment pack), so the loop can hold fragment f+1 in registers
while f decodes and multiplies; both row bases and both dither-seed bases are hoisted out of the
f walk. `GGML_CUDA_FATTN_VEC_TURBO_KQ_PF=1`.

| | med ms T6 | T7 | regs | stack | instr | LDG | HMMA | LDL/STL |
|---|---|---|---|---|---|---|---|---|
| PF=0 | 1.066 | 1.064 | 168 | 0 | 4096 | 403 | 32 | 0 |
| PF=1 | 1.061 | 1.064 | 168 | 0 | 4096 | 403 | 32 | 0 |

SASS is genuinely different (`sha256` of the T6 instruction stream `327ba5ad…` vs `4562bc13…`), so
the flag is live and ptxas re-scheduled — but the timing difference is inside the noise, and
`fnv` / `cfnv` are identical, i.e. bitwise.

Interpretation: at the 168-register budget needed for 3 blocks/SM (hard cap 170), ptxas has no
room for a prefetch that is actually *early*; it can only reorder. See §4 for the arithmetic.

## 2b. S6b/S6c — tile-level and self-row `prefetch.global`

Before touching `Bfr` I closed the "how early" axis with prefetches that cost nothing at all: no
destination register, no smem, no occupancy change, only a `CCTL.E.PF1/PF2` in the loop. Four
distances and both cache levels, all bitwise identical to the reference (`fnv`/`cfnv` equal), all
measured in one window at 0 MiB:

| mode | what it does | T6 ms | T7 ms | vs ref |
|---|---|---|---|---|
| 0 | reference | 1.062 | 1.065 | — |
| 2 | next tile, `prefetch.global.L2`, 2 lines/row | 1.085 | 1.080 | +2.2 % |
| 4 | next tile, `prefetch.global.L1` | 1.072 | 1.074 | +0.9 % |
| 8 | **2** tiles ahead, `prefetch.global.L2` | 1.075 | 1.079 | +1.2 % |
| 3 | next tile L2 **+** register pipeline | 1.084 | 1.076 | +2.1 % |
| 16 | this warp's own rows at tile top (dist 0), L1 | 1.082 | 1.071 | +1.5 % |
| 17 | mode 16 + register pipeline | 1.074 | 1.081 | +1.5 % |

A tile is `cp_stride = gridDim.y*nthreads = 1152` rows = 152 KB per block, i.e. one loop iteration
≈ 38 µs — far beyond any L1 capacity, which is why the L1 variant cannot help; the L2 variant has
the right lifetime but still loses, so those bytes were already on their way. Combined with §4
(the `attr_k` −22 % also constant-folds the strided address arithmetic away, so it over-attributes
to memory) the honest reading is: **the K walk is not latency-exposed at any granularity reachable
by a prefetch.** Six independent points — smem staging (§1), register pipeline (§2), L2 dist 1,
L2 dist 2, L1 dist 1, L1 dist 0 — and none of them recovers anything. K prefetch is closed.

## 2c. Instruction inventory — why the compute side is the remaining lever

Static SASS count of the T6 kernel (`cuobjdump -sass -fun <mangled>`): **12 176 instructions**
per warp-tile. (An earlier count of "4096" in this file was a regex artifact: `cuobjdump` switches
to 5-hex-digit addresses past `0xffff` and the 4-digit pattern silently dropped everything after.)

Mixing that with the measured 1.06 ms (36 SMs, 216 blocks, 28.4 iterations/block, 3 blocks/SM)
gives 8.3 M warp-instructions in 2.76 M cycles, i.e. **IPC ≈ 3.0 of 4** — the kernel is ~75 %
issue-bound, not latency-bound. That is the opposite of what §4's attribution suggested, and it is
the single most important correction in this file: it means *removing instructions does convert to
time*, and it explains why every prefetch variant failed (there is no idle issue slot to hide
anything in).

Top of the histogram: `LOP3` 2095 (17.2 %), `FSEL` 1798 (14.8 %), `IMAD` 1414 (11.6 %), `FFMA` 1390
(11.4 %), `SHF` 1105 (9.1 %), `IADD` 734, `FMUL` 564, `SEL` 519 + `R2P` 512, `LDG` 403 (only 3.3 %).

Five diagnostic builds (`GGML_CUDA_FATTN_VEC_COST`, numerics meaningless) attribute it exactly:

| build | what is removed | instructions | delta | share |
|---|---|---|---|---|
| reference | — | 12 176 | — | — |
| bit 1 | K-side turbo4 dither → 0 | 10 072 | −2 104 | 17.3 % |
| bit 2 | K-side turbo4 centroid tree → one I2F+FMUL | 9 992 | −2 184 | 17.9 % |
| bits 1+2 | both (K side) | 7 880 | −4 296 | 35.2 % |
| bit 4 | V-side turbo3 extraction + centroid | 9 152 | −3 024 | **24.8 %** |
| bit 8 | V-side turbo3 dither group hash | 10 608 | −1 568 | 12.9 % |
| bits 4+8 | both (V side, whole walk) | 7 376 | −4 800 | **39.4 %** |

So 65 per fragment goes into the four `turbo_dither4_pair` calls (= 16 per pair: 2 IMAD for
`seed*C1 + w*C2 + C0`, 5 for the avalanche, 1 `IMAD.WIDE` for the `mulhi`, then 2×{SHF/LOP3, I2F,
FFMA, FMUL}) and 8.5 per decoded element into `turbo4_centroid_mag` — which is a 3-level binary
tree over 8 magnitudes, i.e. **7 `FSEL` plus index work, and 7 is the provable minimum for an
8-to-1 mux**. Neither is sloppy code; both are at their floor *as code*.

The surprise is that the **V walk (39.4 %) is bigger than the whole K-side decode+dither (35.2 %)**,
even though §4 showed it costs nothing to *fetch*. It is pure decode work: 256 coords per lane per
tile (32 rows × 256 coords / 32 lanes), each needing `qs[j/4]`/`signs[j/8]` byte loads with runtime
`j`-derived addresses, extraction, the 8-entry mux and the sign XOR — 3 024 instructions, versus
8192 coords per warp-tile for the K side that share their address math across the fragment walk.

**Trap found here (worth remembering):** `V_DOT2_F32_F16_AVAILABLE` is HIP-only (`common.cuh:764`),
so on CUDA the vec kernel takes the `float` branch of `dequantize_V_*`, not the `half2` one. The
first version of the bit-4 ablation stubbed the `half` branch and measured **exactly zero** delta —
which looks like "the V decode is free" unless you check which branch instantiates.

## 2e. S7 / S8 — replacing the mux trees with byte-indexed tables

§2c's conclusion "both are at their floor" is true for the *code*, but the *function* has a small
domain, and a domain that fits in memory beats a mux:

**S7 — K side.** `turbo4_centroid(two nibbles)` has 256 inputs and 512 float outputs, so one
`__shared__ float2 KCEN[256]` (2 KB, filled once per launch from the same `turbo4_centroid()`, so
bit-identical) turns extraction + 7-`FSEL` tree + sign XOR into one `LDS.64`.
`GGML_CUDA_FATTN_VEC_CEN_TAB=1`: 12 176 → **9 528 instructions (−21.7 %)**, REG 168, STACK 0,
smem 4 864 → 7 936 B (still 3 blocks/SM), and in the one clean window measured so far
T6 **1.065 → 0.770 ms**, T7 1.068 → 0.798 ms (−25.3 %), T1/T3/T8 untouched, `fnv 73d4a5f05d924efc`
/ `cfnv 4664e8eae073af4c` unchanged (`s7_centab_20261103.log`). Note the *time* gain (−27.6 %) is
larger than the instruction gain (−21.7 %): removing the mux also removes its dependency chain.

**S8 — V side.** `dequantize_V_turbo3_0` is only ever instantiated with `ne == 4` and every caller
passes `i0 % 4 == 0` (fattn-vec: `i0 = 2*i_VKQ_0 + lane*4`; lightning-indexer: `lane*4`,
`i_embd*4`), so the four coords of a call share **one** `qs` byte (four 2-bit fields) and **one**
`signs` byte (four 1-bit fields at offset `j0 & 4`) — i.e. one byte indexes all four centroids.
The 3-bit codebook is antisymmetric as `TURBO_C3_REF[7-k] == -TURBO_C3_REF[k]`, so the sign bit also
reflects the magnitude index and one lookup cannot serve both halves: one `__device__ constexpr`
table of 512 `float4` (8 KB, built from `turbo3_centroid_mag()` in a constexpr loop) holds both
halves at `[0..255]` / `[256..511]`, and the sign nibble becomes 4 `SEL`.  `GGML_CUDA_FATTN_VEC_VTAB=1`:
12 176 → **10 000 (−17.9 %)**; opcodes moved as expected (`LOP3 −771`, `SHF −385`, `IADD −266`,
`SEL −257`, `FSEL −256`, `R2P −192`, `IMAD −135`, `LDG +128`), REG 168 / STACK 0 unchanged.
The single-table form is deliberate: two separate 256-entry arrays needed two 64-bit bases and
measurably spilled; one base + an immediate `0x1000` offset is 64 instructions cheaper.
S7+S8 compose almost exactly: **7 344 instructions (−39.7 %)**.
Side effect on the non-fused rows (same trick, no extra work): the t5–t3 *dot* kernel goes
14 448 → 12 288 (−15.0 %).  It is also the only instantiation where S8 changes `STACK` (0 → 8 B),
and that shows up as **2 `LDL` + 1 `STL` in 12 288 instructions**, i.e. prologue/epilogue only —
`LOCAL:0`, no hot-path spill, so it is noted and not chased.
Timing: measured in §2f — ACCEPTED.
Cost that did materialise: S8 reads 128 × 16 B = 64 KB of table per warp-tile through L1 (≈ 24 % of
L1/LDS bandwidth at the measured rate). Standalone it is free (1:1 conversion), on top of S7 it eats
most of the win — see §2f/§2g.
Bitwise status before timing: proven *by construction* and gated at compile time — two
`static_assert`s inside the `VTAB` guard check `turbo3_centroid_c(i) == TURBO_C3_REF[i]` for all 8
codes and the whole 512 × 4 table against `TURBO_C3_REF` (checked on a host copy of the object,
since a `__device__ constexpr` object is not host-readable). Negative control run: corrupting one
`want` index in the checker makes the build fail with
`"S8 turbo3 V centroid table differs from TURBO_C3_REF"`, so the assert is live, not vacuous.
Cost to watch: S8 reads 128 × 16 B = 64 KB of table per warp-tile through L1 (≈ 24 % of L1/LDS
bandwidth at the measured rate), which is the one mechanism by which it could underperform its
static count.  The production build with both flags at 0 is clean (`ggml-cuda.dll` relinked, 64/65).

## 2d. S6d — lead prefetch with the right distance (MEASURED 2026-11-03, neutral)

§2b's 2-tile prefetch was 38 µs of lead, far past L1; §2c's mode 16 was 0 fragments of lead. The
remaining point on that axis is one turbo block: at `f == 0` prefetch the row's block 1 (read at
`f == 8`, ≈ 1000 cycles later at the measured 127 cycles/fragment) and at `mt == 1, f == 0` the
block 0 of the *next* iteration (≈ 16 fragments), one line per lane with lane→row `32*ty + lane`,
so all 32 rows are covered once with no redundancy and no extra registers. Mode 32 (L1), 64 (L2).

Measured (`s5/s6_lead_pf_log.txt`, 0 MiB window, T6 reference 1.062 / T7 1.065):

| mode | T6 med ms | T7 med ms | fnv/cfnv |
|---|---|---|---|
| 32 | 1.064 | 1.070 | identical |
| 64 | 1.066 | 1.066 | identical |
| 33 | 1.065 | 1.060 | identical |

Neutral (inside noise, no +1…2 % penalty of the tile-level variants — the lead time matches the
actual consumption window) but buys nothing: consistent with §2c/§5, the bytes were already in
flight. **K prefetch axis CLOSED** (8 independent points: smem staging, register pipeline, L2 d1,
L2 d2, L1 d1, L1 d0, L1 block-lead, L2 block-lead).

## 2f. S8 measured — 2026-11-03, clean window, ACCEPTED (0 MiB before every row)

`s8_vtab_20261103.log`, each build run twice in one window (0 MiB used throughout), med of 11 reps.
**Bitwise: every `fnv` and `cfnv` in all four builds is identical to the frozen S2d baseline** — T6
`73d4a5f05d924efc`/`4664e8eae073af4c`, T7 `78934835d49f0066`, T8 `a066508a45916751`, T4
`8381bd7573e3d88b`, T2 `14650fb0739d0383`, C1 `209c4022a4eefab3`, C2 `f3a87d8d9fac3a51`, plus the
`cfnv` values of T1/T3. So S8 is bitwise exact in practice, not only by construction.

| row | ref | S7 `CEN_TAB` | S8 `VTAB` | **S7+S8** |
|---|---|---|---|---|
| T1 t4–t3 dot | 1.029 | 1.032 (0) | **0.862 (−16.2 %)** | **0.862 (−16.2 %)** |
| T2 q5_1–t3 | 0.631 | 0.633 (0) | **0.575 (−8.9 %)** | **0.576 (−8.7 %)** |
| T3 t4–t4 dot | 1.020 | 1.023 (0) | 1.024 (0) | 1.018 (0) |
| T4 t5–t3 dot | 1.356 | 1.359 (0) | **1.194 (−11.9 %)** | **1.201 (−11.4 %)** |
| **T6 fused t4–t3** | **0.9985** | 0.725 (−27.4 %) | 0.8205 (−17.8 %) | **0.7025 (−29.7 %)** |
| T7 fused t4–t4 | 1.000 | **0.753 (−24.7 %)** | 1.003 (0) | **0.758 (−24.2 %)** |
| T8 fused t5–t3 | 1.117 | 1.120 (0) | **0.942 (−15.7 %)** | **0.944 (−15.5 %)** |
| C1 d128 t4–t3 | 0.472 | 0.472 (0) | **0.4215 (−10.7 %)** | **0.4205 (−10.9 %)** |
| C2 d128 fused | 0.4575 | **0.358 (−21.7 %)** | 0.4055 (−11.4 %) | **0.345 (−24.6 %)** |

Zero on exactly the rows it should be zero on (T3/T7 are turbo4 V; T7 only moves with S7, T6/T1/T8/
T4/T2/C1 only with S8) — that is the A/B control working.

Two things the numbers say:

1. **S8 converts 1:1.** Static −17.9 % instructions → **−17.8 %** time on T6. The static proxy is
   calibrated now (S7 converted super-linearly: −21.7 % instr → −27.4 % time).
2. **The two tables interact strongly, and the interaction is one-sided.** S8 alone −17.8 %, but on
   top of S7 it is worth only 0.725 → 0.7025 = **−3.1 %**; S7 on top of S8 still gives −14.3 %. So
   once the K-side ALU work is gone, the V table stops paying — the most plausible mechanism is that
   the 128 extra `LDG.128` per tile (64 KB/warp-tile through L1) were previously hidden behind the
   K decode and are now partly exposed. That is the open lead: put the V table in **smem** instead
   (S8b, 8 KB more smem: 6 912 → ~15 KB/block, still 3 blocks/SM at 228 KB) and see whether the
   combination moves from −29.7 % toward the ~−38 % the instruction counts would predict.

**Window calibration warning:** this window is ~6 % faster than the frozen S2d window on *every* row
(ref T6 0.9985 vs 1.062-1.068, ref T1 1.029 vs 1.09, ref T8 1.117 vs 1.195-1.205, ref T4 1.356 vs
1.445). Absolute ms therefore are not comparable to `FA_TURBO_BASELINE_20261103.md`; only ratios
inside the same window are. S7 also measured 0.770 in the earlier `s7_centab_20261103.log` window
vs 0.725 here — same ratio, different clock.

Production reading (using the S2d closeout's ~12 % attention share of decode): −29.7 % on the fused
t4–t3 kernel ≈ **−3.6 % decode time**, i.e. tg ≈ 28.97 → ~30.0 before any PPL check.

## 2g. Why is the S7+S8 combination sub-linear?  Three probes, one clean window

§2f leaves a 8‑point gap between what the instruction counts predict (−39.7 %) and what the same
window measures (−29.7 %).  Two candidate mechanisms, and they are separable:

**(a) smem footprint / L1 carveout.** S8 itself adds no smem, but it is read through the unified
L1/LDS; S7 already moved 2 KB of the same budget into smem.
Probe: `fa_turbo_vtab_pad2k` = `VTAB=1 -DSMEM_PAD=2048`, i.e. S8 plus 2 KB of *dead* static smem
whose address is leaked through an asm barrier (the first `pad38k` attempt in §2 showed ptxas deletes
an unreferenced pad, so the barrier is required).  It must be **bitwise identical to plain S8** and it
raises the T6 static smem 5 376 → 7 424 B, exactly S7's footprint, REG unchanged.
Weakness to keep in mind: occupancy here is register-limited (168 reg × 128 thr = 21 504 of 65 536 →
exactly 3 blocks/SM at either footprint, pb stays 9), so if this probe shows a cost it is the
carveout, not the occupancy cascade; if it shows nothing the mechanism is ruled out.

**(b) the table read is a gather.** The `qs`/`signs` byte loads are coalesced (lanes read consecutive
bytes), but `TAB.v[qb]` is indexed by a *value*: 32 lanes hit up to 32 different 128 B lines per
`LDG.128`.  Standalone that hides behind the ALU work (hence S8's clean 1:1 conversion); once S7
removes the K-side ALU work the L1 line-fetches are co-critical.
Probe: `fa_turbo_vtab_g2[_ce]` = `VTAB=1 -DCOST=16` masks the table index to two values 128 B apart,
so the two `LDG.128`, the byte loads, the `SEL` chain and the FMA walk all stay and only the spread
collapses (2 lines instead of up to 32).  Numerics meaningless, timing only.  Static cost of the mask:
10 000 → **10 184** and 7 344 → **7 536**, i.e. +2.5 %, cheap enough to read the result directly.

**S8c, the candidate fix that follows from (b):** the 3-bit codebook has only **8 distinct values**,
so the table can be 8 floats = 32 B = **one cache line**, and every lane's read becomes a broadcast
instead of a gather — `GGML_CUDA_FATTN_VEC_VTAB=2`, four `LDG.E.CONSTANT.32` with a 3-bit index
(`((qb >> 2l) & 3) | ((sb >> l) & 1) << 2`) instead of two `LDG.128` plus 4 `SEL`.  Same values, same
bitwise contract.  The gate is a *cross*-check rather than a third copy of the same comparison: the
two paths derive the codebook index differently, so `turbo3_c8tab_ok(scalar, float4)` requires the
scalar table to equal `TURBO3_C4_TAB.v[(hi1 << 8) | b].x` for all 8 codes — a typo in either index
derivation fails the build.  Negative control: reading `.y` instead of `.x` in that checker fails the
compile with `"S8c scalar V table disagrees with the S8 float4 table"`.
It deliberately costs *more* instructions, which makes it a direct test of the mechanism: if S8c ≈ S8
in time despite the extra ALU, and S7+S8c ≫ S7+S8, the gather was the limiter.

Measured static counts for all of the above (T6, instr per warp-tile, REG 168 / STACK 0 in every row):

| build | instr | LDG | LDG.128 | LDG.U8 | note |
|---|---|---|---|---|---|
| ref | 12 176 | 403 | 0 | 256 | |
| S7 | 9 528 | 403 | 0 | 256 | |
| S8 | 10 000 | 531 | 128 | 256 | |
| S7+S8 | 7 344 | 531 | 128 | 256 | −39.7 % |
| S8 `pad2k` | 10 000 | 531 | 128 | 256 | code identical to S8, smem 5 376 → 7 424 B |
| S8 `g2` | 10 184 | 531 | 128 | 256 | +2.5 %, spread collapsed to 2 lines |
| S7+S8 `g2` | 7 536 | 531 | 128 | 256 | |
| **S8c** | 10 832 | 659 | 0 | 256 | +832 vs S8 = 13 per decode call, as predicted |
| **S7+S8c** | 8 176 | 659 | 0 | 256 | −32.9 % |

So S8c gives back 11 % of the instruction win by construction.  It only pays if the mechanism is
real — the ratio between 7 344 (S7+S8) and 8 176 (S7+S8c) in *time* is exactly the quantity that
decides whether the FA decode path should keep an 8 KB gather table or an 8-value broadcast table.

Two secondary differences, in case the timing comes out close:

* **The non-fused rows pay more for S8c**: the t5–t3 *dot* kernel is 14 448 (ref) → 12 288 (S8) →
  **13 112** (S8c), i.e. S8c gives back 6.7 % of the gain on that instantiation.
* **S8c removes the only S8 side effect**: `D256 K47 V45 f160` had `STACK:8` with the float4 table
  (two 64-bit bases) and is back to `STACK:0` with the single 32 B table.  REG 168 / SHARED 5 376 and
  every occupancy figure are identical between S8 and S8c, so neither variant changes the schedule
  the launcher picks.

**S8b, the other fix (not started):** put the 8 KB float4 table in smem, which is where S7's
experience suggests gathers are much cheaper (S7 converted *super*-linearly). Plumbing: `dequantize_V`
is a `constexpr` function-pointer variable, so the call is fully inlined and the cost of adding a
table parameter is mechanical but wide — the `dequantize_V_t` typedef, every `dequantize_V_*`
variant, 3 sites in `fattn-vec.cuh`, 2 in `lightning-indexer.cu`.  Do it only if S8c underperforms.
Prefetching the table line instead is *not* promising: §2b/§2d show prefetches at this granularity
never paid here.

Gate script: `run_s8_discrim.cmd` → `s8_discrim_20261103.log` (9 builds, two interleaved passes).

## 2h. §2g measured — the gather IS the mechanism (clean window, 0 MiB throughout)

`s8_discrim_20261103.log`: 9 builds, two interleaved passes (forward then reverse), med of 11 reps,
`nvidia-smi` 0 MiB before every run.  The two passes agree to ±0,5 %, so all deltas below are real.

| build | T6 med ms (pass 1 / pass 2) | vs ref | vs S7+S8 | bitwise |
|---|---|---|---|---|
| ref `pfm0` | 0.997 / 0.998 | — | | ✓ |
| S7 `centab` | 0.726 / 0.722 | −27.4 % | | ✓ |
| S8 `vtab` | 0.817 / 0.822 | −17.9 % | | ✓ |
| **S7+S8 `vtab_ce`** | **0.703 / 0.702** | **−29.6 %** | — | ✓ |
| S8 + 2 KB dead smem `vtab_pad2k` | 0.822 / 0.822 | −17.6 % | | ✓ |
| S8 gather→2 lines `vtab_g2` | 0.816 / 0.814 | −18.3 % | | n/a (timing only) |
| **S7+S8 gather→2 lines `vtab_g2_ce`** | **0.628 / 0.627** | **−37.1 %** | **−10.7 %** | n/a (timing only) |
| S8c 32 B table `vtab8` | 0.890 / 0.891 | −10.7 % | | ✓ |
| **S7+S8c `vtab8_ce`** | **0.654 / 0.655** | **−34.4 %** | **−6.8 %** | ✓ |

**(a) smem footprint / carveout: ruled out.** `vtab_pad2k` is `vtab` to within noise (0.822 vs
0.817/0.822) at double the smem footprint (5 376 → 7 424 B, i.e. S7's footprint), identical code and
identical `fnv`.  As predicted, occupancy is register-limited so nothing else moved either — but
there is no footprint penalty to chase.

**(b) the gather: confirmed, and it is bigger than the probe's own overhead.** Collapsing the table
index to two lines changes nothing standalone (0.816 vs 0.817 — the gather hides behind the ALU, which
is exactly why S8 converted 1:1 in §2f) but changes everything on top of S7: **0.7025 → 0.6275,
−10.7 %, while the build carries +2.5 % MORE instructions** (7 536 vs 7 344).  Netting out the mask
cost, the 8 KB value-indexed gather costs ≈ 13 % of the kernel in the S7 configuration.  That is the
whole missing combination win: −29.6 % measured vs −39.7 % predicted ≈ 10 points, and the gather is
worth ≈ 13.

**S8c is bitwise-correct and already captures most of it.** The 32 B broadcast table is *worse*
standalone (0.8905 vs 0.8195, which is 1:1 with its +8.3 % instructions — the static proxy is still
calibrated) but better in the combination: **+11.3 % instructions over S7+S8 and still 6.8 % faster**.
So the mechanism pays more than the ALU it costs, and `vtab8_ce` = 0.6545 is the best *numerically
valid* T6 yet: −34.4 % vs ref, against −37.1 % for the impossible `g2_ce` ideal.

**The right V-table form depends on whether the K side was table-ised too.** Across the other rows the
two forms split cleanly (med of both passes):

| row | ref | S7 | S8 (gather) | S8c (broadcast) | S7+S8 | S7+S8c | better V form |
|---|---|---|---|---|---|---|---|
| T1 t4–t3 dot | 1.021/1.028 | 1.023 | **0.861/0.860** | 0.921/0.919 | 0.859 | 0.923 | gather |
| T2 q5_1–t3 | 0.632 | 0.632 | **0.575** | 0.613/0.614 | 0.575/0.576 | 0.612/0.613 | gather |
| T4 t5–t3 dot | 1.353/1.349 | 1.343/1.353 | **1.195** | 1.256/1.262 | 1.193/1.195 | 1.270/1.261 | gather |
| T8 t5–t3 fused | 1.122/1.121 | 1.127/1.112 | **0.946/0.937** | 1.007/1.013 | 0.939/0.937 | 1.008/1.011 | gather |
| C1 d128 t4–t3 | 0.472 | 0.472 | **0.422** | 0.439 | 0.423 | 0.439 | gather |
| **T6 fused t4–t3** | 0.997/0.998 | 0.726/0.722 | 0.817/0.822 | 0.890/0.891 | 0.703/0.702 | **0.654/0.655** | **broadcast** |
| **C2 d128 fused** | 0.458/0.457 | 0.359/0.358 | 0.405 | 0.433 | 0.345 | **0.333** | **broadcast** |

The split is not random: the broadcast table wins exactly on the rows where S7 is active (T6, C2 —
turbo4 K through the fused path), and loses on every row where the K side still carries its full ALU
load (dot kernels, turbo5 K, q5_1 K) and therefore absorbs the gather for free.  So the production
rule should be per-instantiation: **use the broadcast table where the K-side table is used, the gather
table otherwise** — expressible as a template parameter on `get_dequantize_V` fed by the same
`cen_tab` condition, no runtime cost, no ABI change.

Production reading with the S2d closeout's ~12 % attention share of decode: the fused t4–t3 kernel
0.9975 → 0.6545 = **−34.4 %**, i.e. ≈ **−4.1 % decode time**, tg ≈ 28.97 → ~30.2 before any PPL check.

## 2i. S8e, implemented: signed `float2` pairs (512 B table) — and the per-instantiation switch

The float4 form cannot be made small: 4 fields × 4 magnitudes = 256 distinct combinations, so its table
is ≥ 4 KB and the gather is intrinsic to it.  The scalar form reaches one cache line but needs 4 loads.

**S8e (`GGML_CUDA_FATTN_VEC_VTAB=4`)** sits between them, and one detail from the codebook makes it
nearly free: the sign is **not** a post-process — `turbo3_centroid_mag()` itself reflects the magnitude
index when the sign bit is clear (`t = (idx&3) ^ (idx&4 ? 0 : 3)`), so the sign bit belongs *in the
index*.  Two coords can therefore share one 6-bit index (4 magnitude bits + 2 sign bits): a table of
**64 `float2` = 512 B = four cache lines**, indexed by `qb`'s low/high nibble OR-ed with two sign bits:

```c
p01 = ldg(&TAB.v[(qb & 0xF) | ((sb & 3) << 4)]);      // coords 0,1
p23 = ldg(&TAB.v[((qb >> 4) & 0xF) | (((sb >> 2) & 3) << 4)]);  // coords 2,3
```

Two `LDG.64` and one `LOP3` per pair: roughly the **same ALU as the float4 form** (which pays 2 address
computations + 4 `SEL` instead) at **~4 touched lines per gather instead of ~20**, and unlike S8c it
needs no sign `SEL` at all.  Gated the same way: `turbo3_c2tab_ok` cross-checks all 64 × 2 entries
against the float4 table, which is an independent index derivation.

**The per-instantiation switch (implemented, `VTAB=3` and `VTAB=5`).** §2h shows the choice is not
global: the wide gather table is better exactly where the K side still decodes with mux trees, and the
narrow broadcast table is better where the K side was table-ised (S7).  `fattn-vec.cuh` now computes

```c
constexpr int vtab_mode = VTAB == 3 ? (cen_tab ? 2 : 1)      // broadcast where S7, gather otherwise
                  : VTAB == 5 ? (cen_tab ? 2 : 4) : VTAB;    // same rule, float2 instead of float4
```

and passes it as a template argument to `get_dequantize_V` → `dequantize_V_turbo3_0<T, ne, vtab>`.  The
`dequantize_V_t` function-pointer typedef and every other `dequantize_V_*` variant are untouched, and
`lightning-indexer.cu` keeps the macro default, so the change is compile-time only.

Not measured yet: `fa_turbo_vtabe[_ce]` (form 4 everywhere), `fa_turbo_vtab3_ce`, `fa_turbo_vtab5_ce`
(auto rules with S7 on).  The question the next window answers: does form 4 beat form 1 on the non-S7
rows (T1/T2/T4/T8/C1) — if yes, `VTAB=5` is the production setting; if not, `VTAB=3` is.

**Static verification that the auto rule really is per-instantiation** (`s8_static_counts.ps1`, instr
per warp-tile; the two auto builds are `CEN_TAB=1 + VTAB=3/5`):

| instantiation | form 1 | form 2 | form 4 | `VTAB=3` | `VTAB=5` |
|---|---|---|---|---|---|
| D128 fused (S7 on) | 5 320 | 5 736 | 5 384 | **5 736 → 4 440** = form 2 | **4 440** = form 2 |
| D256 t5–t3 fused (S7 off) | 11 048 | 11 872 | 11 176 | **11 048** = form 1 | **11 176** = form 4 |
| D256 T6 fused (S7 on) | 10 000 | 10 832 | **10 128** | **8 176** = form 2 | **8 176** = form 2 |
| D256 t4–t4 dot (V = turbo4) | 12 736 | 12 736 | 12 736 | 12 736 | 12 736 |
| D256 T1 dot (S7 off) | 10 304 | 11 136 | **10 432** | **10 304** = form 1 | **10 432** = form 4 |

Form 4 costs only **+128 instructions** over the float4 form (≈2 per decode call) against form 2's
+832, which is why it is the candidate for the rows that keep the gather: same ALU budget, ~4 L1 lines
per gather instead of ~20.  Registers/smem are unaffected (all D256 instantiations REG 168 / SHARED
5 376 or 7 936 as before), and form 4 also keeps the `D256 K47 V45 f160` row at `STACK:0` — the float4
form's 8-byte spill (§2e) is back in `VTAB=3` and absent in `VTAB=5`.

## 2j. The pick — `CEN_TAB=1 + VTAB=4`: **T6 −40.8 %**, and the auto rule turned out unnecessary

`s8_pick_20261103.log` (clean window, 0 MiB before every run, 10 builds × 2 interleaved passes, med of
11 reps; parsed with `parse_fa_log.ps1`).  **All 9 rows produced exactly one distinct `fnv`/`cfnv`
across all 10 builds** — including the new float2 form — so all three table forms are bitwise exact in
the running kernel, not only by construction.  Max spread between the two passes: < 0.02 ms everywhere.
`ref` T6 in this window is 0.9975, identical to §2h's window, so these numbers are directly comparable
to §2h/§2f.

med ms (mean over the two passes):

| row | ref | S7 | S7+form 1 (float4) | S7+form 2 (scalar) | **S7+form 4 (float2)** | auto 3 | auto 5 |
|---|---|---|---|---|---|---|---|
| **T6 fused t4–t3** | 0.9975 | 0.7265 | 0.703 | 0.648 | **0.591** | 0.6485 | 0.6495 |
| **C2 d128 fused** | 0.457 | 0.358 | 0.345 | 0.3355 | **0.300** | 0.3345 | 0.334 |
| T1 dot t4–t3 | 1.0275 | 1.027 | 0.861 | 0.9185 | **0.8325** | — | — |
| T2 q5_1–t3 | 0.633 | 0.6335 | 0.575 | 0.609 | **0.559** | — | — |
| T4 t5–t3 dot | 1.352 | 1.353 | 1.1965 | 1.2795 | **1.195** | — | — |
| T8 fused t5–t3 | 1.1245 | 1.1225 | 0.940 | 1.0045 | **0.920** | — | — |
| C1 d128 dot | 0.4725 | 0.472 | 0.4205 | 0.448 | **0.406** | — | — |
| T7 fused t4–t4 (V=turbo4) | 1.0015 | 0.7555 | 1.0015 | 1.0065 | 0.7575 | 0.7585 | 0.7575 |
| T3 dot t4–t4 (V=turbo4) | 1.0185 | 1.0225 | 1.0185 | 1.016 | 1.024 | 1.023 | 1.017 |

**Form 4 (512 B signed float2 pairs) is the best form on every single row**, not just on the S7 rows:

* S7-active rows, where §2h said the narrow table must win: T6 **0.591** vs 0.648 for the 32 B scalar
  table (−8.8 %) and 0.703 for the 8 KB float4 (−16 %); C2 **0.300** vs 0.3355 / 0.345.
* Non-S7 rows, where §2h said the gather form must win: form 4 beats form 1 on all of them — T1 0.8325
  vs 0.861 (−3.3 %), C1 0.406 vs 0.4205 (−3.4 %), T2 0.559 vs 0.575 (−2.8 %), T8 0.920 vs 0.940
  (−2.1 %), T4 1.195 vs 1.1965 (tie).

So the §2h split was real but *not* a reason to keep two forms: form 4 dominates both corners, because
it removes most of the line traffic at ~2 instructions of extra ALU instead of ~13.  **The auto rules
(`VTAB=3`, `VTAB=5`) are therefore rejected** — they pick form 2 for the S7 rows and cost 9.9 % there
(0.6485/0.6495 vs 0.591).  They stay in the tree as forensics of that conclusion.

**The gather penalty is gone and the static proxy is calibrated again.** S7+form 4 measures −40.8 %
against the instruction-count prediction of −39.7 % (7 472 / 12 176).  Compare §2f where S8 alone
converted 1:1 and the combination lost 10 points, and §2h where the best valid combination was −34.4 %
with an unreachable ideal of −37.1 %.  Form 4 is *below* that "ideal", which says the `g2` mask probe
was not an ideal at all (it kept 2 gathers of 2 lines plus the mask ALU) — the lesson stands, but the
bound was drawn too pessimistically.

Production candidate: **`GGML_CUDA_FATTN_VEC_CEN_TAB=1` + `GGML_CUDA_FATTN_VEC_VTAB=4`**, everything
else at defaults.  Against ref in this window: T6 −40.8 %, C2 −34.4 %, T1 −18.7 %, T2 −11.5 %,
T4 −11.4 %, T8 −17.9 %, C1 −14.1 %, and exactly zero on the two rows whose V type is turbo4 (T3, T7
without S7).  With the S2d closeout's ~12 % attention share of decode: ≈ **−4.9 % decode time**,
tg ≈ 28.97 → ~30.4, pending PPL.

Residual risk to gate before shipping: the probe covers `flash_attn_ext_vec` only.  The
`lightning-indexer.cu` kernels call `get_dequantize_V` with the macro default, so a global `VTAB=4`
also changes them (bitwise-safe by the same asserts, but untimed).  Gate: diff `cuobjdump -res-usage`
for the indexer kernels between the current DLL and the candidate DLL; if REG/SHARED/STACK are
unchanged, accept, otherwise pass `vtab_mode = 0` explicitly from that call site.

Optional beyond that: S8b (float4 table in smem, 8 KB more smem — occupancy is register-limited so it
would not cost blocks/SM) converts the L1 line cost into smem bank conflicts instead; plumbing is the
wider change (a table parameter through the typedef and every variant), so it stays behind S8e.

## 2k. Flip: both defaults ON — production gate **PASSED**

Per the §2j pick and explicit user approval (2026-11-03):

- `fattn-vec.cuh`: `GGML_CUDA_FATTN_VEC_CEN_TAB` default **0 → 1** (comment marked ACCEPTED).
- `turbo-quant.cuh`: `GGML_CUDA_FATTN_VEC_VTAB` default **0 → 4** (comment records the pick and that
  3/5 are forensics of the rejected hypothesis).
- No CMake option or `-D` in the tree overrides either macro (grep over `*.txt` / `*.cmake` = 0
  matches), so the header defaults are what production compiles with.
- Indexer gate passed **before** the flip: `lightning-indexer.cu` compiled standalone with and
  without `CEN_TAB=1 + VTAB=4` gives **identical `REG/STACK/SHARED` for all 36
  `lightning_indexer_kernel_{vec,wmma}` instantiations** (64×8, 69, 70×7, 40×6, 71×8, 40×6) — the
  change is numerically bitwise-safe there by the same asserts and costs nothing in resources.

Production gate script: `run_s78_gate.cmd` → `ppl_s78_YYYYMMDD.log`.  It (0) prints `nvidia-smi`,
(1) rebuilds the probe from the current headers **with no `-D` flags** — i.e. the shipping
configuration — and requires every frozen hash to match, (2) runs the S2d-protocol PPL row
(`-c4096 -b1 -ub1 -s42`, turbo4/turbo3, fused routing) against fused-t4 reference **7.1751**, gate
|Δ| ≤ 0.05 — and since the kernels are bitwise-identical, it must land *on* the reference, not
merely near it.

### 2k measured (`ppl_s78_20261103.log`, clean window: 0 MiB before, 0 MiB after)

Probe rebuilt from the headers with **no `-D` flags** (`nvcc exit: 0`) = the shipping configuration.

| row | regs / smem | med ms | fnv / cfnv | frozen? |
|---|---|---|---|---|
| T1 t4–t3 dot | 168 / 4 352 | 0.833 | `33b6fa5ef169e4e6` / `782e697aec966835` | ✓ |
| T2 q5_1–t3 | 254 / 4 352 | 0.558 | `14650fb0739d0383` / – | ✓ |
| T3 t4–t4 dot | 168 / 4 352 | 1.015 | `44464b17ae8d1478` / `5a6e7d955d4d7722` | ✓ |
| T4 t5–t3 dot | 168 / 4 352 | 1.197 | `8381bd7573e3d88b` / `ecbe126da140afe0` | ✓ |
| **T6 fused t4–t3** | 168 / **6 912** | **0.591** | `73d4a5f05d924efc` / `4664e8eae073af4c` | ✓ |
| T7 fused t4–t4 | 168 / 6 912 | 0.759 | `78934835d49f0066` ✓ | ✓ |
| T8 fused t5–t3 | 168 / 4 864 | 0.919 | `a066508a45916751` ✓ | ✓ |
| C1 d128 dot | 244 / 2 304 | 0.407 | `209c4022a4eefab3` ✓ | ✓ |
| C2 d128 fused | 254 / 4 864 | 0.300 | `f3a87d8d9fac3a51` ✓ | ✓ |

Every hash matches the frozen S2d baseline.  What did move: `smem` on the fused turbo4 rows (6 912 B =
the S7 table) and the T2/C2 register count (254 instead of 255, `STACK:0` — one register freed, no
spills).  Timings land on the §2j `vtabe` numbers (T6 0.591, C2 0.300, T1 0.833, C1 0.407, T4 1.197,
T8 0.919, T7 0.759), i.e. the shipping default reproduces the measured pick exactly, and vs the §2j
`ref` row that is T6 **−40.7 %**, C2 −34.4 %, T1 −18.9 %, T2 −11.8 %, T4 −11.5 %, T8 −18.3 %,
C1 −13.9 %.

**PPL gate: PPL = 7.1719 ± 0.37892** vs the fused-t4 reference **7.1751 ± 0.380** → Δ = **−0.0032**,
gate |Δ| ≤ 0.05 **PASSED**.  Protocol identical to the S2d gate row (same corpus file, `-c 4096
--chunks 1 -b 1 -ub 1 -ngl 99 -fa on -t 7 -s 42`, `-ctk turbo4 -ctv turbo3`, fused routing).  Because
the hash gate proves the attention math is bitwise unchanged, this residual is the harness's known
pass-to-pass drift (the dot row itself drifted 7.1802 → 7.1720 across S2d for the same reason), not a
numerics change.

## 3. TLP ceiling — `MIN_BLOCKS=4` (128 regs, occ 4, pb 6)

| row | 3 blk / 168 reg | 4 blk / 128 reg |
|---|---|---|
| T1 t4–t3 dot | 1.091 | 1.156 |
| T6 t4–t3 fused | 1.060 | 1.175 |
| T7 t4–t4 fused | 1.058 | 1.153 |
| T8 t5–t3 fused | 1.195 | 1.257 |
| C1 / C2 D128 | 0.503 / 0.491 | 0.513 / 0.493 |

More concurrency does not help past 12 warps/SM; the extra warps cost more than they hide.

## 4. Attribution — where the 1.06 ms actually goes

Diagnostic knob `GGML_CUDA_FATTN_VEC_ATTR` (probe builds only, numerics meaningless): bit 1
collapses the K row stride, bit 2 the V row stride, so every row of a tile resolves to one cache
line. T6 (fused turbo4 K + turbo3 V):

| variant | med ms | delta |
|---|---|---|
| reference | 1.062 | — |
| K rows L1-hot | 0.827 | **−22.1 %** |
| V rows L1-hot | 1.069 | +0.7 % (nothing) |
| both hot | 0.823 | −22.5 % |

- The K byte stream (128 rows × 132 B per tile) is 22 % of the kernel. That is the *ceiling* of any
  K-prefetch work, and both S5 and S6 captured 0 of it.
- The V side is free: 100 B/row × 128 rows per tile costs no measurable time, which is consistent
  with the byte-sensitivity test (t3 V 100 B vs t4 V 132 B: 1.060 vs 1.058 ms).
- 78 % of the kernel is compute: mux decode + dither + 32 mma + softmax/expf + KQ/VKQ register
  traffic + the PV FMAs themselves.

## 5. Why the prefetches fail (and what would be needed)

§2c changes the diagnosis. At IPC 3.0 of 4 the kernel has almost no idle issue slot, so a prefetch
can only compete with real work for the same slot — which is exactly what the ±1.5 % measurements
show. The 22 % that §4 attributes to the K byte stream is real, but it is *issue occupancy* of the
96 `LDG.U8` + address IMADs per tile, not a stall that could be filled.

## 5b. Where the remaining time actually is

35 % of the instruction stream is dither + centroid, and both are at their minimum *as code*
(§2c). The rest is 32 HMMA + the PV decode/FMA walk + softmax + address math. So the way forward is
fewer *operations*, not better scheduling.

The part of this section that said "micro-optimising this kernel is a dead end" was **wrong** and is
kept only as a marker: a mux tree is minimal as code, but its *domain* is tiny, and a table beats a
minimal mux. S7/S8 (§2e) took 39 % of the instructions out of the kernel with no change to the
numerics, the register budget or the occupancy.

What is left after both, and what can still be done about it:

| block | instr/tile | share of 12 176 | addressable? |
|---|---|---|---|
| K dither hash | 2 104 | 17.3 % | no — bitwise contract forbids changing the hash |
| V dither hash | 1 568 | 12.9 % | no — same |
| K centroid mux | 2 184 | 17.9 % | **gone (S7)** |
| V decode + extraction | 3 024 | 24.8 % | **72 % gone (S8)**, floor ≈ 848 |
| everything else (mma, softmax, PV FMA, addressing) | 3 296 | 27.1 % | unexplored |

The two dithers together are 30 % of the kernel and are pure murmur-style hashing of
`(block_idx, coordinate-group)` — they are a property of the cache tensor, not of the query, which
is what makes them look precomputable. They are not: `u` is a float per coordinate (512 B per row,
vs 132 B for the K row itself). Closing that 30 % needs a different *contract* (e.g. moving the
dither into the quantization step so the cache stores dithered centroids), not an optimization.

## 6. Harness fix that came out of this

`run_s5_gate.cmd` compared raw `fnv` across builds whose occupancy differs. That hash covers the
`dst`/`meta` partials, whose slot layout is `(head*parallel_blocks + t)*D`, so it changes whenever
pb changes — it cannot be compared across builds. The probe now also prints `cfnv`, the hash of the
**merged, normalised** per-head output, which is pb-invariant and is the value to gate on
(`fnv` stays for same-pb A/B).

## 7. Reproduce

```
:: S5 (staging) — already rejected; numbers in s5_gate_20261103.log
build-x64-windows-msvc-release\bin\fa_turbo_ab_cpasync.exe        :: stage ON, 255 reg, occ 2
build-x64-windows-msvc-release\bin\fa_turbo_ab_cpasync_mb3.exe    :: stage ON, 168 reg, occ 2
build-x64-windows-msvc-release\bin\fa_turbo_ab_occ2.exe           :: stage OFF, 255 reg, occ 2
build-x64-windows-msvc-release\bin\fa_turbo_ab_occ4.exe           :: stage OFF, 128 reg, occ 4
build-x64-windows-msvc-release\bin\fa_turbo_ab_pad38k.exe         :: stage OFF, 168 reg, occ 2 (SMEM_PAD, §1c)
build-x64-windows-msvc-release\bin\fa_turbo_ab_cpasync_mb3c.exe  :: stage ON,  168 reg, occ 2, cfnv probe (§1c)
:: S6 (register prefetch) — s6_gate_20261103.log
build-x64-windows-msvc-release\bin\fa_turbo_s6_ref.exe            :: PF=0, production policy
build-x64-windows-msvc-release\bin\fa_turbo_s6_pf.exe             :: PF=1
:: S6b/S6c (tile / self-row prefetch, till-level CCTL) — s6_pf_modes_20261103.log
build-x64-windows-msvc-release\bin\fa_turbo_pfm0.exe              :: mode 0 (reference)
build-x64-windows-msvc-release\bin\fa_turbo_pfm2.exe              :: mode 2 (next tile, L2)
build-x64-windows-msvc-release\bin\fa_turbo_pfm4.exe              :: mode 4 (next tile, L1)
build-x64-windows-msvc-release\bin\fa_turbo_pfm8.exe              :: mode 8 (2 tiles, L2)
build-x64-windows-msvc-release\bin\fa_turbo_pfm3.exe              :: mode 3 (L2 + registers)
build-x64-windows-msvc-release\bin\fa_turbo_pfm16.exe             :: mode 16 (own rows, L1, dist 0)
build-x64-windows-msvc-release\bin\fa_turbo_pfm17.exe             :: mode 17 (16 + registers)
:: S6d (lead prefetch, one turbo block ahead) — run_s6_lead_pf.cmd -> s6_lead_pf_20261103.log
build-x64-windows-msvc-release\bin\fa_turbo_pfm32.exe             :: mode 32 (lead 8 frags, L1)
build-x64-windows-msvc-release\bin\fa_turbo_pfm64.exe             :: mode 64 (lead 8 frags, L2)
build-x64-windows-msvc-release\bin\fa_turbo_pfm33.exe             :: mode 33 (lead + registers)
:: instruction-cost attribution — NUMERICS MEANINGLESS
build-x64-windows-msvc-release\bin\fa_turbo_cost1.exe             :: K dither forced to 0
build-x64-windows-msvc-release\bin\fa_turbo_cost2.exe             :: K centroid tree -> I2F
build-x64-windows-msvc-release\bin\fa_turbo_cost3.exe             :: both
build-x64-windows-msvc-release\bin\fa_turbo_cost4.exe             :: V extraction + centroid removed
build-x64-windows-msvc-release\bin\fa_turbo_cost8.exe             :: V dither group hash removed
build-x64-windows-msvc-release\bin\fa_turbo_cost12.exe            :: both V blocks
:: S7 (K centroid table in smem) — s7_centab_20261103.log, measured ACCEPTED
build-x64-windows-msvc-release\bin\fa_turbo_centab.exe           :: CEN_TAB=1
:: S8 (V centroid table, global) — s8_vtab_20261103.log, measured ACCEPTED
build-x64-windows-msvc-release\bin\fa_turbo_vtab.exe             :: VTAB=1
build-x64-windows-msvc-release\bin\fa_turbo_vtab_ce.exe          :: VTAB=1 + CEN_TAB=1
build-x64-windows-msvc-release\bin\fa_turbo_vtab_stub.exe        :: VTAB=1 + COST=4 (static check only)
build-x64-windows-msvc-release\bin\fa_turbo_s8_cost8.exe         :: VTAB=1 + COST=8 (static check only)
:: §2g discriminators for the sub-linear S7+S8 combination — run_s8_discrim.cmd -> s8_discrim_20261103.log
build-x64-windows-msvc-release\bin\fa_turbo_vtab_pad2k.exe       :: VTAB=1 + SMEM_PAD=2048 (bitwise-valid, footprint control)
build-x64-windows-msvc-release\bin\fa_turbo_vtab_g2.exe          :: VTAB=1 + COST=16 (TIMING ONLY, gather collapsed)
build-x64-windows-msvc-release\bin\fa_turbo_vtab_g2_ce.exe       :: VTAB=1 + CEN_TAB=1 + COST=16 (TIMING ONLY)
build-x64-windows-msvc-release\bin\fa_turbo_vtab8.exe            :: VTAB=2 (S8c, bitwise-valid candidate fix)
build-x64-windows-msvc-release\bin\fa_turbo_vtab8_ce.exe         :: VTAB=2 + CEN_TAB=1 (S7+S8c candidate)
build-x64-windows-msvc-release\bin\fa_turbo_vtabe.exe           :: VTAB=4 (S8e, signed float2 pairs)
build-x64-windows-msvc-release\bin\fa_turbo_vtabe_ce.exe        :: VTAB=4 + CEN_TAB=1
build-x64-windows-msvc-release\bin\fa_turbo_vtab3_ce.exe        :: VTAB=3 + CEN_TAB=1 (auto: 2 where S7, else 1)
build-x64-windows-msvc-release\bin\fa_turbo_vtab5_ce.exe        :: VTAB=5 + CEN_TAB=1 (auto: 2 where S7, else 4)
:: the pick run: run_s8_pick.cmd -> s8_pick_20261103.log; static helpers: s8_static_counts.ps1, s8_res_usage.ps1
:: attribution — timing only
build-x64-windows-msvc-release\bin\fa_turbo_attr1.exe             :: K rows L1-hot
build-x64-windows-msvc-release\bin\fa_turbo_attr2.exe             :: V rows L1-hot
build-x64-windows-msvc-release\bin\fa_turbo_attr3.exe             :: both
```

Probe builds are plain nvcc:

```
nvcc -O3 -std=c++17 -arch=sm_120a -DGGML_TURBO_DITHER \
     -I ggml/include -I ggml/src -I ggml/src/ggml-cuda \
     tools\fa_turbo_ab_probe.cu -o build-x64-windows-msvc-release\bin\<name>.exe
```

plus one of `-DGGML_CUDA_FATTN_VEC_TURBO_KQ_PF={1,2,3,4,8,16,17,32,33,64}`,
`-DGGML_CUDA_FATTN_VEC_CPASYNC=1`, `-DGGML_CUDA_FATTN_VEC_MIN_BLOCKS={2,4}`,
`-DGGML_CUDA_FATTN_VEC_ATTR={1,2,3}`, `-DGGML_CUDA_FATTN_VEC_CEN_TAB=1`,
`-DGGML_CUDA_FATTN_VEC_VTAB={1,2}`, `-DGGML_CUDA_FATTN_VEC_SMEM_PAD=2048`,
`-DGGML_CUDA_FATTN_VEC_COST={1,2,3,4,8,12,16}` as needed.

`GGML_CUDA_FATTN_VEC_TURBO_KQ_PF` is a bitmask: `1` register fragment pipeline, `2` next-tile
`prefetch.global.L2`, `4` next-tile `prefetch.global.L1`, `8` two tiles ahead, `16` this warp's own
rows at tile top, `32`/`64` rolling one-turbo-block lead (L1/L2). Bits compose (`3` = pipeline +
L2 prefetch), except that `4`/`2` select the cache level, so with `4` set the prefetch goes to L1.

## 8. State of the flags

| flag | default | status |
|---|---|---|
| `GGML_CUDA_FATTN_VEC_CPASYNC` | 0 | S5, rejected — do not enable |
| `GGML_CUDA_FATTN_VEC_TURBO_KQ_PF` | 0 | S6/S6b/S6c bitmask, all bitwise but all slower or neutral — leave at 0 |
| `GGML_CUDA_FATTN_VEC_MIN_BLOCKS` | 1 (policy) | production policy = 3 blocks for turbo4/5 K, D256, ncols 1 |
| `GGML_CUDA_FATTN_VEC_SMEM_PAD` | 0 | diagnostic: dead smem to reproduce an occupancy cascade |
| `GGML_CUDA_FATTN_VEC_ATTR` | 0 | diagnostic: L1-hot K / V rows, timing only |
| `GGML_CUDA_FATTN_VEC_COST` | 0 | diagnostic: instruction / L1 attribution by ablation — bit 1 K dither, bit 2 K centroid, bit 4 V extraction+centroid, bit 8 V dither hash, bit 16 collapse the S8 table gather; numerics meaningless with bits 1/2/4/8/16 set |
| `GGML_CUDA_FATTN_VEC_CEN_TAB` | **1** | **S7, ACCEPTED + ON by default 2026-11-03 (§2f/§2h/§2j) — 2 KB smem byte→float2 turbo4 centroid table for the fused KQ walk; −21.7 % instructions, T6 −27.4 %, bitwise. `=0` keeps the mux-tree path for A/B** |
| `GGML_CUDA_FATTN_VEC_VTAB` | **4** | **S8, ACCEPTED + ON by default 2026-11-03 (§2f/§2h/§2j) — byte-indexed turbo3 V centroid table, form 4 = 512 B signed float2 pairs: best form on every probe row, T6 −40.8 % with S7, bitwise. `1` = 8 KB float4 (fewest instr, ~20 L1 lines/gather), `2` = 8-value scalar (32 B, no gather, +11 % instr in combo), `3`/`5` = per-instantiation auto rules — **measured and rejected** (they cost 9.9 % on the S7 rows). `=0` keeps the mux decode for A/B** |

## 9. Branch closed (2026-11-03)

Axis status after 11 measured experiments on the decode kernel
(`flash_attn_ext_vec<256,1,TURBO4_0,TURBO3_0,softcap=false,KQ_mma=false,KQ_f16dq=true>`):

| axis | points | verdict |
|---|---|---|
| memory side (prefetch / staging / TLP) | S5, S6, S6b, S6c, S6d, `MIN_BLOCKS=4` | **CLOSED** — 8 negative or neutral points, 3 blocks/SM is the optimum, the loads are already covered |
| instruction side (mux trees → byte-indexed tables) | S7, S8 forms 1/2/4, `VTAB` auto 3/5 | **CLOSED with the win**: `CEN_TAB=1` + `VTAB=4`, T6 0.9975 → **0.591 ms (−40.7 %)**, every frozen hash unchanged |
| root cause of the earlier sub-linearity | `pad2k`, `g2`, `g2_ce`, `vtab8` | **RESOLVED** — the value-indexed 8 KB table read is a gather worth ≈13 % of the kernel; form 4 removes almost all of it for ≈2 extra instructions per decode call |

Production state: `CEN_TAB=1`, `VTAB=4` compile by default in `fattn-vec.cuh` / `turbo-quant.cuh`;
`ggml-cuda.dll` rebuilt with them; gates green — indexer resources identical (36 kernels), probe hashes
identical to the frozen S2d baseline (9/9 rows), PPL 7.1719 vs 7.1751 (Δ −0.0032, tolerance 0.05).

Expected production effect (S2d's measured ~12 % attention share of decode): ≈ **−4.9 % decode time**,
tg ≈ 28.97 → ~30.4.  Not benchmarked in production — the user closed the branch here; `llama-bench`
d8192 A/B remains available if that number is ever wanted.

Deliberately NOT done: no commit (the user closed the branch without approving one).  Uncommitted at
close: `fattn-vec.cuh`, `turbo-quant.cuh`, `fattn-common.cuh` (the S8 table plumbing and the two
defaults), this file, `run_s8_*.cmd` / `run_s78_gate.cmd`, `parse_fa_log.ps1`, `s8_*.ps1`, and the raw
logs `s8_discrim_20261103.log`, `s8_pick_20261103.log`, `ppl_s78_20261103.log`.  Nothing else in the
tree was touched; `lightning-indexer.cu`, the `dequantize_V_t` typedef and all other quant variants are
byte-for-byte unchanged.

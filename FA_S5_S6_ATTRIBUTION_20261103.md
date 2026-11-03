# S5 / S6 verdict + decode-kernel latency attribution — 2026-11-03

Reference baseline: `FA_TURBO_BASELINE_20261103.md` (S2d, 3 blocks/SM, 168 regs, 4864 B smem).
This file records the two prefetch experiments that followed it and the attribution probe that
explains why both came out negative/neutral.

Tree: `ee7dcf0` + the uncommitted S5/S6 work in `ggml/src/ggml-cuda/fattn-vec.cuh`.
GPU: RTX 5060 Ti, `nvidia-smi` checked before every run below; all rows in §1–§4 were taken at
0 MiB used / 0 %.

Probe: `tools/fa_turbo_ab_probe.cu` (nq=1, nkv=32768, heads 24/4, D=256, med of 11 reps).
Raw logs: `s5_gate_20261103.log`, `s6_gate_20261103.log`.

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

Two diagnostic builds (`GGML_CUDA_FATTN_VEC_COST`, numerics meaningless) attribute it exactly:

| build | instructions | delta | share |
|---|---|---|---|
| reference (mode 0) | 12 176 | — | — |
| bit 1: dither forced to 0 | 10 072 | −2 104 | **17.3 %** |
| bit 2: centroid tree → one I2F+FMUL | 9 992 | −2 184 | **17.9 %** |
| both | 7 880 | −4 296 | 35.2 % |

So 65 per fragment goes into the four `turbo_dither4_pair` calls (= 16 per pair: 2 IMAD for
`seed*C1 + w*C2 + C0`, 5 for the avalanche, 1 `IMAD.WIDE` for the `mulhi`, then 2×{SHF/LOP3, I2F,
FFMA, FMUL}) and 8.5 per decoded element into `turbo4_centroid_mag` — which is a 3-level binary
tree over 8 magnitudes, i.e. **7 `FSEL` plus index work, and 7 is the provable minimum for an
8-to-1 mux**. Neither is sloppy code; both are at their floor.

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

## 3. TLP ceiling — `MIN_BLOCKS=4` (128 regs, occ 4, pb 6)| row | 3 blk / 168 reg | 4 blk / 128 reg |
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

35 % of the instruction stream is dither + centroid, and both are at their minimum (§2c). The rest
is 32 HMMA + the PV dequant/FMA walk + softmax + address math. Nothing in that list is
obviously wasteful, so further gains need fewer *operations*, not better scheduling: e.g. a wider
MMA shape (m16n8k16 is already the largest for f16 with 16 coords/fragment), or avoiding the
per-element decode altogether by keeping K in a wider pre-decoded form (which the project rejected
for cache-size reasons). Micro-optimising this kernel further is a dead end — the measured ceiling
is the few percent that the 16-instruction dither pair and the 7-`FSEL` centroid mux can still be
squeezed by, and neither has slack.

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
build-x64-windows-msvc-release\bin\fa_turbo_cost1.exe             :: dither forced to 0
build-x64-windows-msvc-release\bin\fa_turbo_cost2.exe             :: centroid tree -> I2F
build-x64-windows-msvc-release\bin\fa_turbo_cost3.exe             :: both
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

plus one of `-DGGML_CUDA_FATTN_VEC_TURBO_KQ_PF={1,2,3,4,8,16,17}`, `-DGGML_CUDA_FATTN_VEC_CPASYNC=1`,
`-DGGML_CUDA_FATTN_VEC_MIN_BLOCKS={2,4}`, `-DGGML_CUDA_FATTN_VEC_ATTR={1,2,3}` as needed.

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
| `GGML_CUDA_FATTN_VEC_COST` | 0 | diagnostic: dither / centroid instruction attribution, numerics meaningless |

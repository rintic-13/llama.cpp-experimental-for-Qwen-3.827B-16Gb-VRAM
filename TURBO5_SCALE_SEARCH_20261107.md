# TURBO5 SCALE SEARCH + UNBIASED SCALE — ported to production, 2026-11-07

Question that started it (user): "в теории turbo5 будет лучше q5_1? если так — ввести в него
оптимизации точности как в turbo4-3". The theory was checked on real rows first
(KV_OUTLIERS_20261107.md), then the port was made and gated.

## Why turbo5 was behind until now

`quantize_row_turbo5_0_ref` (and `k_set_rows_turbo5`) carried a SEPARATE encoder:
5-bit PolarQuant, dither-free by design, and the OLD norm correction `grp_norm/||rec||`.
The llama-tq encoder upgrade (32-trial scale search + unbiased scale `grp_norm/inner`)
was only in the TURBO3_0/TURBO4_0 paths. Real-row effect before the port:
t5 aggregate relL2 0.0509 vs q5_1 0.0422 (K) - t5 lost to q5_1 both median and tail.

## Port (encode-side only, same rules as turbo4)

Files: `ggml/src/ggml-turbo-quant.c` (`quantize_row_turbo5_0_ref`, `#ifdef
GGML_TURBO_SCALE_SEARCH` branch mirroring the TURBO4_0 path: 32 trial scales from
`TURBO_TRIAL_SCALE_INV`, sqrt-free winner `inner*inner*best_rsq > best_in*best_in*rsq`,
nominal k=16 wins ties, stored scale `grp_norm/inner`, fp16); `ggml/src/ggml-cuda/set-rows.cu`
(`k_set_rows_turbo5`: warp-tree per-trial sums, thread-0 scan with the same ascending/strict>
rule, re-quantize at winner, unbiased scale; `#else` keeps the old path bit-identically).
Block layout, packing (qs nibbles + signs high bit) and every reader kernel unchanged.

Shared sign-mask (the llama-tq "one shared sign pattern" design): the fork's rotation is
`D(s2)*(H/sqrt(128))*D(s1)` with GLOBAL constants (`ggml_turbo_wht_signs`, seed 42; CUDA
`TURBO_WHT_SIGNS1/2` in __constant__ memory) - identical for every 128-block on both
engines, and Q is pre-rotated in the graph from the very same signs (one rotation per query,
which is exactly the point of the shared mask in llama-tq). The port did not touch the
rotation: the search and packing sit strictly after the FWHT stage in both engines.

## Gate 1 - CPU parity (real rows, no GPU)

`kv_outliers.exe` against the rebuilt production ggml-base.lib: the production t5 encoder
and the standalone prototype agree DIGIT-FOR-DIGIT on 25 600 K + 25 600 V real rows
(aggregate K 0.04403 = 0.04403, V 0.04341 = 0.04341; worst-row, p99.9, elemAbsMax identical
including the @rec/r/blk addresses). Post-port production numbers vs q5_1 (same data):

| side | metric | q5_1 | turbo5 prod (post-port) |
|---|---|---|---|
| K | relL2 med | 0.04150 | 0.04396 |
| K | relL2 max row | 0.05825 | **0.05148** |
| V | relL2 med | 0.04211 | 0.04389 |
| V | relL2 max row | 0.06070 | **0.05151** |
| K/V | blocks with >10% RMS err | 0% | 0% |

The profile matches the 4-bit gate exactly: median ~5% behind q5_1, worst rows 12-15% better.

## Gate 2 - CPU<->GPU bit-exactness of the t5 mirror (the shared-sign-mask check)

`verify_t5mirror.exe` over a fresh KVD1/KVG1 dump pair (`ppl_results/t5mirror_gate_20261107`,
lease run 20261107-231801, `-ctk turbo5 -ctv turbo5`, 6 dumped calls, 24 576 blocks): CPU
re-quantization of the raw rows vs the GPU-written bytes, per block on qs[64]/signs[16]/norm:

- qs mismatches 20, signs mismatches 6 (total differing SIGN BITS: 6 out of ~3.1 M),
  norm mismatches 30 - i.e. 32 differing blocks out of 24 576 (0.13%);
- every sampled block: `k_cpu == k_gpu` (identical winning scale, overlap 128/128 indices);
  differences are fp16 norm last-ulp and single centroid-boundary straddles;
- max relative angle gap between any diverging choices: **1.146e-07**; classification
  near-tie/ULP: 32, with a real angle gap: **0**. Verdict: PASS - documented CPU serial-sum
  vs CUDA warp-tree summation difference (the same rule the turbo3/turbo4 mirrors state:
  near-tie winners may differ per engine, each written cache is self-describing and
  self-consistent). The sign-mask packing itself is bit-identical: 6 straddled elements
  out of 3 145 728.

Tooling bug found by this gate: the `GGML_KV_GPU_DUMP` (KVG1) companion ran inside
`ggml_cuda_op_set_rows` BEFORE the kernel dispatch, so it captured the PRE-write state
(fresh rows = zeros) and any mirror compare failed 24 576/24 576. The hook is now called
after the dispatch (src0/src1 are read-only, capture semantics unchanged). First-round dump
(`runs/20261107-224849-*`) is superseded; use the 23:18 pair as the frozen evidence.

## Gate 3 - shared sign-mask: improvement or systematic distortion for t5? (CPU-only, real K rows)

`kv_mask_corr.exe` (built by `build_maskcorr.bat`, source `kv_mask_corr.cpp`) over the 10-31 KVD1
dump: 200 real K records x 128 tokens x 8 KV heads = 204 800 128-blocks, one identical
scale-search+unbiased t5 encoder under two mask policies, every reconstruction inverse-rotated
back to the raw domain, plus REAL pre-rotated head-0 queries from `ppl_eval/Q_real.f32`
(un-rotated with the same `D(s1)HD(s2)/sqrt(128)` the graph uses as stage-3, so
`<q_rot, e_rot> == <q_raw, e_raw>` exactly):

| variant | relL2 | channel-bias ratio &#124;&#124;mu&#124;&#124;/indep-expectation | score coherence mean (max) |
|---|---|---|---|
| A shared global masks (production today) | 0.04403 | **1.26x** | **1.14** (4.00) |
| B per-block Rademacher masks (splitmix64) | 0.04405 | 1.02x | 0.68 (2.77) |

- MSE accuracy: statistically IDENTICAL (0.04403 vs 0.04405; A reproduces the production
  aggregate 0.04403 exactly, confirming the tool's calibration). The shared mask buys no
  local accuracy.
- Systematic channel bias: A shows 26% excess over the independent-error expectation on 1024
  channels - with 1024-channel statistics (null sd ~2%) that is ~12 sigma, i.e. real. B sits
  at the independent level (1.02x, noise).
- Attention-level effect on real queries: coherent score distortion in A is ~1.7x the
  per-block design (1.14 vs 0.68 mean; per-query mean|mean|/std of `<q, e_k>` over 4096 keys,
  2048 queries; a fully independent design gives ~0.8). Worst queries: 4.00 vs 2.77.

Verdict for the turbo4-3 concern applied to turbo5: the shared sign-mask gives NO improvement
(equal MSE) and DOES add a measurable but modest systematic component - channel bias +26%,
coherent score error ~1.7x larger than per-block masks, heavy-tailed on some queries. It is
real structure, not noise - exactly the mechanism that motivated rejecting the mask dependency
in turbo4-3 - but its magnitude is a factor, not an order. Whether that factor helps or hurts
PPL is decided by the GPU rows, not by this test; per-block masks in production would also
require per-block Q rotations (cost out of scope here).

## Not yet measured (next)

- GPU PPL rows with the new t5 encoder: `t5/t3` at c=8192 (old-encoder anchor 7.2863,
  same-day `t4/t3` 7.2483) and `q5_1/t3` at c=8192 (the head-to-head reference row).
  Until then the PPL-level "t5p vs q5_1" claim rests on the real-row tail/median profile.
- Encoder cost: 32-trial search per 128-block, same structure as t4's (measured there at
  ~1% pp/tg at set_rows time); t5 decode path unchanged.

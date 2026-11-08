# KV QUANT MAX-OUTLIERS TEST — 2026-11-07 (CPU-only, real rows)

Question: where does the quantization error of `q4_1-q4_1` vs `turbo4-turbo4` concentrate?
Max-outlier behaviour on REAL K/V rows of this model (Qwen3.8-27B, post-rope K / raw V).

## Data and method

- Input: `ppl_results/kv_dump_20261031_204410/kv_dump.bin` — GGML_KV_DUMP capture from a real
  perplexity run (400 records, 200 x K + 200 x V, 128 rows x ne0=1024 each, f32 raw
  pre-rotation pre-quantization rows) => **25 600 real K rows and 25 600 real V rows**.
- Codecs (CPU reference, production semantics): `q4_1` and `q5_1` quantize the raw row
  (error measured in raw domain); `turbo3`/`turbo4` via `quantize_row_*_0_ref` from the
  production `ggml-base.lib` (DITHER ON + SCALE_SEARCH ON, rotated internally, error
  measured in the WHT-rotated domain, which is the dot-product domain — rel-per-row
  metrics are domain-invariant, single-element ABS values are only comparable inside
  one codec's domain).
- Tool: `kv_outliers.cpp` (Temp\opencode, built against production ggml-base.lib).
  Does not call the GPU; no lease needed.

## Results — K rows (25 600)

| codec (domain) | relL2 med | p99 | max | blk>10% RMS err | row-max err med | row-max p99 |
|---|---|---|---|---|---|---|
| q4_1 (raw)      | 0.0858 | 0.1007 | **0.1268** | 13.8% | 0.363 | 0.566 |
| q5_1 (raw)      | 0.0415 | 0.0488 | 0.0583 | 0% | 0.176 | 0.275 |
| turbo3 (rot)    | 0.1817 | 0.1975 | 0.2175 | 100% | 1.498 | 2.574 |
| **turbo4 (rot)**| 0.0883 | 0.0951 | **0.1066** | 5.4% | 0.582 | 0.982 |

## Results — V rows (25 600)

| codec (domain) | relL2 med | p99 | max | blk>10% RMS err | row-max err med | row-max p99 |
|---|---|---|---|---|---|---|
| q4_1 (raw)      | 0.0870 | 0.1062 | **0.1359** | 14.7% | 0.249 | 2.016 |
| q5_1 (raw)      | 0.0421 | 0.0516 | 0.0607 | 0% | 0.120 | 0.978 |
| turbo3 (rot)    | 0.1814 | 0.1975 | 0.2163 | 100% | 1.036 | 6.272 |
| **turbo4 (rot)**| 0.0881 | 0.0951 | **0.1032** | 5.3% | 0.405 | 2.562 |

## Interpretation

1. **Worst-case row is better for turbo4, median is a wash.** The worst K row:
   q4_1 0.1268 vs turbo4 0.1066 (-16%); the worst V row: 0.1359 vs 0.1032 (-24%).
   Median relL2 is within 0.003 on both sides (q4_1 marginally better).
2. **q4_1's tail is outlier-driven and heavier.** V rows: q4_1 single-element error
   median-of-row-max 0.25 -> p99 2.0 (x8 blowout across rows, one recorded elem error
   4.41); 14-15% of its 32-blocks have RMS err > 10% of the block signal (one outlier
   poisons a whole 32-block amax scale). Turbo4 spreads outliers with the WHT rotation:
   only 5.3-5.4% of its (128-wide) blocks exceed the same relative threshold, and its
   row-max distribution is tight (V: med 0.41 -> p99 2.6).
3. **Rows that hurt the codecs are different rows.** Correlation of per-row relL2:
   q4_1 vs turbo4 = -0.16 (K) / -0.38 (V) — no shared weakness; turbo3 vs turbo4 = 0.79 / 0.83
   (same rotation mechanism, different bits).
4. Consistency with the same-day PPL rows (this date, c=8192 frozen protocol):
   q4_1/q4_1 = 7.2573, turbo4/turbo4 = 7.2465, turbo4/turbo3 anchor = 7.2483.
   The small median disadvantage of turbo4 is outweighed by its better tail — matching
   "tail errors matter more than median errors" for attention quality.
5. q5_1 (already a closed line) confirms its known large accuracy margin as reference.

## Verdict

- On max-outliers the real-data test agrees with the PPL verdict: **turbo4 does not lose
  accuracy to q4_1 at the same 4-bit width — it wins the tail (the outliers) while using
  993 MiB vs 1370 MiB of KV at c=8192.**
- (Done in passing, noted q4_1 V-side tail is much worse than its K-side tail.)

## Measurement hygiene

- The test is CPU-only and runs without a lease. Dump file is from 2026-10-31;
  no re-capture needed: raw rows are codec-input ground truth, independent of later
  encoder changes (scale search affects only the encoder, which we call explicitly).

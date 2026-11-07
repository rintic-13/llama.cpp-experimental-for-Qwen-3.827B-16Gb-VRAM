# Engine-level KV A/B: turbo4-turbo3 vs q4_1-q4_1 — 2026-11-04

Real-generation engine comparison (llama-server, NOT PPL — the engine is the deciding
counter, same lesson as the Strata work). Run: `run_engine_kv_ab.ps1`,
artifacts in `engine_ab_20261104_215538\` (summary.json + per-request answers).

## Protocol

| item | value |
|------|-------|
| build | `build-x64-windows-msvc-release\bin\llama-server.exe` (GGML_CUDA_FA_QUANTS incl. both pairs) |
| model | `G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` (qwen35, 16 KV layers, reasoning=deepseek) |
| server | `-fa on -ngl 99 -t 7 -c 70000 --port 8081 --no-webui`, ONE session per arm |
| sampling | greedy (temperature 0, top_p 1, seed 42), `cache_prompt=false` |
| requests | chat0/1/2 (300 tok), code (600 tok), long doc ~7.5k tok + needle, xl doc ~44.2k tok + needle |
| needle | mechanically scored: MK-5033 / 7.42 (long), YT-1187 / 3.15 / 9.68 (xl), present in prompt by assert |
| GPU | RTX 5060 Ti 16 GB, idle gate passed (16050 MiB free), no leftover processes |
| log | timings from slot print_timing lines in `.err`; usage from HTTP; KV sizes from startup log |

## Results

### Timings (from slot print_timing)

| req | prompt tok | prefill t/s T | prefill t/s Q | gen tok | decode t/s T | decode t/s Q |
|-----|-----------:|--------------:|--------------:|--------:|-------------:|-------------:|
| chat0 |     187 | 473.9 | 406.5 | 300 | 30.5 | 30.6 |
| chat1 |     180 | 546.2 | 547.0 | 300 | 30.5 | 30.7 |
| chat2 |     182 | 552.1 | 552.2 | 300 | 30.5 | 30.7 |
| code  |     515 | 798.4 | 799.6 | 600 | 30.2 | 30.6 |
| long  |    7551 | 942.3 | 945.7 | 158 | 27.3 | 29.4 |
| xl    |   44163 | 802.0 | 805.7 | 208/173 | **18.26** | **24.51** |

- **Prefill: TIE.** 802 vs 805.7 t/s at 44k (−0.5%), equal at short ctx. raw ms/tok at 44k: T 1.25 vs Q 1.24.
- **Decode: q4_1 WINS as context grows.** Equal at small ctx (~30.5 t/s both), −7% for turbo at
  7.5k ctx, **−25.5% for turbo at 44k ctx (54.76 ms/tok vs 40.80 ms/tok → +34% t/s for q4_1)**.
- **Root cause (runtime isolation, turbo_kq_20261104_2253*, 44.2k ctx, one xl req each):**
  the cost is the turbo4 **K-side KQ walk**, NOT a missing fused kernel and NOT the turbo3 V side:

  | KV pair | decode t/s @44k | ms/tok |
  |---------|---------------:|-------:|
  | turbo4-turbo3 (A/B ref)          | 18.26 | 54.76 |
  | turbo4-turbo3, env TURBO_KQ_MMA=0 (dot) | 18.30 | 54.65 |
  | turbo4-turbo4                    | 18.37 | 54.43 |
  | q5_1-turbo3                      | 22.03 | 45.40 |
  | q4_1-q4_1                        | 24.51 | 40.80 |

  - fused (default ON, `GGML_CUDA_FATTN_TURBO_KQ_MMA_F16=ON`) vs forced q8_1-dot: **identical**
    on the engine (18.26 vs 18.30) - the selector is active for turbo4 K post-S2d, and both
    KQ paths cost the same here (kernel bench said fused only ~3% faster anyway).
  - V side: turbo3 -> turbo4 V changes nothing (18.26 -> 18.37), so turbo3 V is NOT the
    bottleneck; the spread q5_1-turbo3 (22.03) vs q4_1-q4_1 (24.51) budgets the remaining
    ~2.5 t/s to the turbo3 V decode side.
  - K side: turbo4 K -> plain q5_1 K recovers +3.8 t/s (18.3 -> 22.0): centroid mux + dither +
    norm folding + f16 fragments on every KQ read is what q4_1/q5_1 K do not pay.

### KV memory (startup log, 70144 cells / 16 layers / -c 70000)

| arm | KV size | K | V | GPU total after all reqs |
|-----|--------:|-:|-:|-------------------------:|
| turbo4-turbo3 | **993.25 MiB** | turbo4 565.12 | turbo3 428.12 | 13286 MiB |
| q4_1-q4_1     | 1370 MiB      | q4_1         | q4_1         | 13662 MiB |

turbo saves **377 MiB** VRAM (KV buffer) and stays ~14 GB total on the 16 GB card.

### Quality (mechanical needle scoring)

| req | turbo4 | q4_1 |
|-----|:------:|:----:|
| long | 2/2 | 2/2 |
| xl   | 3/3 | 3/3 |

Both arms extract the needles exactly; the needle answers are literally identical
between arms ("codified **MK-5033** … **7.42 megapascals**" / the full YT-1187 triplet).
On the two long-context retrieval probes the KV quantization noise does NOT cost quality.

### Greedy divergence (same prompt, both arms)

| req | common prefix words | jaccard |
|-----|--------------------:|--------:|
| chat0 | 20 | 0.85 |
| chat1 |  2 | 0.30 |
| chat2 |  1 | 0.23 |
| code  | 16 | 0.42 |
| long  | 14 | 1.00 |
| xl    | 21 | 1.00 |

Open-ended chat continuation diverges (dequant noise amplified by sampling chains),
deterministic extraction tasks converge to identical answers. Divergence here is a
noise meter, not a quality meter.

## Accuracy data (short)

### PPL anchors in this fork (model Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S, eval `ppl_eval\wiki.test.quick.txt`)

Protocol A (production-style, `-c 16384 -b 8192 -ub 512 -fa on -ngl 99 -t 7`, baseline 2026-10-31):

| pair | PPL | vs f16 ref |
|------|----:|-----------:|
| f16-f16 (ref) | 5.0414 | — |
| q5_1-turbo5 | 5.0346 | −0.0068 |
| q5_1-turbo4 | 5.0560 | +0.0146 |
| q5_1-turbo3 | 5.1040 | +0.0626 |
| turbo4-turbo3 | 5.1391 (run 20261031_025751) / 5.0719 (run 20261031_214045, FA-fallback build) | +0.098 / +0.031 |

Protocol B (strict-token, `-c 4096 --chunks 1 -b 1 -ub 1 -s 42`, FA_TURBO_BASELINE_20261103):

| row | pair | PPL | delta vs gold |
|-----|------|----:|--------------:|
| T4 | f16-f16 (gold) | **7.1378** | — |
| T1 | turbo4-turbo3 (prod) | 7.1719 | +0.0341 (+0.48%) |
| T3 | q5_1-turbo3 | 7.1743 | +0.0365 |
| T2 | turbo4-turbo3 native dot | 7.1802 | +0.0424 |
| T7 | turbo5-turbo3 fused | 7.1868 | +0.0490 |
| —  | q4_1-q4_1 | NOT MEASURED in the repo | pending |

- PPL accuracy cost of the turbo pair vs gold f16: ~+0.03–0.10 PPL (protocol-dependent),
  i.e. under +0.5–2%; turbo4-turbo3 sits mid-pack, better than q5_1-turbo3 and turbo5-turbo3,
  worse than q5_1-K pairs (q5_1-turbo5 is even better than gold).
- The q4_1-q4_1 PPL row never existed in this fork. Prepared: `ppl_q41_quick.ps1`
  (same protocol as T1/T4; gate-ready). As of 2026-11-05 it could not run: GPU busy
  15772 MiB / 287 free at 99 % util (foreign process `strata` PID 10524) - rerun when GPU frees.

### Engine-level accuracy (this A/B, the deciding signal per user rule)

- mechanical needle: turbo4 **2/2 long, 3/3 xl** == q4_1 **2/2, 3/3** - 100% both, and the
  needle answers are literally identical between arms (same code, same thresholds, same format);
- retrieval-style tasks (long/xl): arms converge (jaccard 1.00) - KV quantization noise did not
  propagate into wrong facts;
- open-ended chat continuation diverges (jaccard 0.23-0.42, common prefix 1-20 words) - that is
  amplified dequant noise in sampling chains, not measurable accuracy loss (both arms' answers
  were coherent and on-topic);
- so on the engine the two pairs are ACCURACY-EQUAL at 7.5k-44k context; they differ in speed
  (q4_1 +34% decode @44k) and memory (turbo −377 MiB).


## Verdict

**turbo4-turbo3: keep as the fork's memory/production pair.**
- It delivers exactly its raison d'être: −377 MiB KV (993 vs 1370 MiB at 70k cells) and
  equal prefill, with zero measurable needle-quality loss (both 100%).
- BUT the encode-quality story hides a real engine cost: at 44k context decode is
  ~18.3 vs 24.5 t/s (turbo −25.5%). If the workload is long-context multi-turn
  (decode-dominated at big ctx), **q4_1-q4_1 is the faster pair** at +377 MiB cost.
- The cost is measured to sit in the turbo4 K-side KQ walk (fused and dot cost the same
  on the engine; V-side turbo3 is a minor contributor). A cheaper fused turbo4-K decode
  (the KQ walk itself, e.g. smem centroid folding of the fused builder or fewer per-byte
  ops) is the kernel lever - NOT "wire up a missing fused path", it is already wired and
  default-ON for turbo4 K (S2d).
- if VRAM headroom exists and the workload is decode-heavy at 40k+ ctx,
  `-ctk q4_1 -ctv q4_1` gains up to +34% decode t/s at no needle-quality cost.

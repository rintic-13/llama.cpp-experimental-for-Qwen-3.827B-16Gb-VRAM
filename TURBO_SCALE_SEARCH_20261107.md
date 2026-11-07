# TurboQuant encoder upgrade: scale search + unbiased scale - PASSED, new frozen anchors (2026-11-07)

Port of the llama-tq encoder upgrade (docs/kv-cache, "scale search over the codebook" + "unbiased
scale", 2026-10-11) onto the production turbo4-turbo3 KV pair, behind CMake option
`GGML_TURBO_SCALE_SEARCH` (default ON; OFF compiles the bit-identical pre-upgrade codec via `#else`).

## Change summary

Encode-side only. Block layout, bpw, dither rule and every reader (FA vec kernels, convert,
GGUF compat) are untouched.

- `ggml/src/ggml-common.h`: `TURBO_TRIAL_SCALE_INV[32]` - codebook multipliers 0.36x..1.60x,
  step 0.04, index 16 = exact nominal scale; `static __constant__` under `__CUDACC__`.
- `ggml/src/ggml-turbo-quant.c` (turbo3/turbo4 refs) + `ggml/src/ggml-cuda/set-rows.cu`
  (`k_set_rows_turbo3/4`): assignment retried at all 32 trial scales, winner = max angle cosine
  tracked sqrt-free as inner^2/rsq (k ascending, strict >, nominal wins ties); the stored fp16
  scalar becomes the unbiased scale `grp_norm / inner` instead of the norm correction
  `grp_norm / ||rec||`. The CUDA warp-tree sum vs the CPU serial sum differ at last-ulp, so
  near-tie winners may differ per engine; each written cache is self-describing (idx + norm).

## CPU numeric check (real tree codec compiled with prod macros, 8192 blocks)

| metric | anchor (SCALE_SEARCH=OFF) | new (ON) |
|---|---|---|
| turbo4 proj_ratio = <x,xhat>/\|\|x\|\|^2 | 0.9954 (= cos, systematic shrink) | **1.0000** |
| turbo4 cos | 0.9954 | 0.9961 |
| turbo3 cos | 0.9827 | 0.9836 |

## GPU gates (RTX 5060 Ti 16 GB, model Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S, tree f95e932 + 4 modified files)

All GPU rows below measured 2026-11-07 with the card idle (16050 MiB free) and CUDA graphs active
(`graphs reused = 8159`).

### PPL, frozen 8K protocol (FA_PPL_8K_GATE_20261103), `-c 8192 --chunks 1 -b 1 -ub 1 -s 42`

| row | K/V | PPL | delta vs gold 7.2415 |
|---|---|---|---|
| P8-1 old anchor (norm correction) | turbo4 / turbo3 | 7.2900 +/- 0.27601 | +0.0485 |
| **P8-1 new (scale search + unbiased)** | turbo4 / turbo3 | **7.2483 +/- 0.27678** | **+0.0068** |
| P8-3 gold | f16 / f16 | 7.2415 +/- 0.27601 | - |

The turbo4-turbo3 quality gap to f16 shrinks from 0.0485 to 0.0068 (-86%).

### Speed, llama-bench `-ngl 99 -fa on -t 7 -ub 512 -p 512 -n 128 -r 2` (ON vs OFF same day)

| row | test | ON | OFF control | delta ON-OFF |
|---|---|---|---|---|
| C1 | tg128 @ d512 | 32.33 +/- 0.30 | 32.77 +/- 0.20 | -1.3% |
| C2 | tg128 @ d8192 | 28.80 +/- 0.12 | 29.11 +/- 0.14 | -1.1% |
| C3 | pp512 @ d8192 | 877.41 +/- 6.65 | 886.57 +/- 5.66 | -1.0% |
| C4 | pp512 @ d512 | 937.48 +/- 6.36 | 947.42 +/- 7.52 | -1.1% |
| - | tg128 @ d8192 f16-f16 control | 32.32 +/- 0.08 | - | - |

Decode cost is the per-token set_rows encode (~0.3 ms/token, 32 trial assignments per 128-value
block); prefill cost -1%. ptxas (sm_120a): turbo kernels 40 registers, 0 spills, 1564 B smem
(anchor 32/36 registers, 536 B) - no occupancy cliff.

Frozen pre-upgrade anchors (30.04 +/- 0.12 tg, build d89cfff) do NOT transfer to today's machine
state: the same-day OFF control measures 29.11; environment/tree drift is -3%.

## Incident note (measurement hygiene)

A first d8192 bench run today showed tg 11.55 - NOT REPRODUCIBLE (ON: 28.80, OFF: 29.11). Cause:
run against the broken tooling state after the failed 2026-11-06 CUDA compile - the per-tool
impl DLLs (llama-perplexity-impl.dll, llama-cli-impl.dll, ...) were left inconsistent and the
thin clients crashed with 0xC0000005 at startup before any output (this is why three manual
llama-perplexity runs produced zero output and a 0-byte log). A full incremental rebuild of all
targets fixed the clients. GPU tests must start from an idle card (>= 14000 MiB free) and from a
build where `--help` on every used tool prints usage.

## Verdict / new frozen production anchors (this tree)

- PPL 8K turbo4-turbo3: **7.2483** (gold 7.2415)
- tg128 @ d8192: **28.80 +/- 0.12**; pp512 @ d8192: **877.41 +/- 6.65** (same-day OFF control:
  29.11 / 886.57 - the codec upgrade costs ~1%)
- `GGML_TURBO_SCALE_SEARCH` stays ON in the production default.

## Reproduce

```
gate_scale_search_20261107.cmd    (PPL 8K + speed, logs ppl8k_ss_turbo.log bench8k_ss_turbo.log)
ctrl_off_bench_20261107.cmd       (ON/OFF speed control, run after cmake -DGGML_TURBO_SCALE_SEARCH=OFF)
diag_scale_search_20261107.cmd    (bottleneck discriminators S1-S3 + live PPL S4)
```

CPU harness: `%TEMP%\opencode\build_ss_test.bat` (compiles ggml-turbo-quant.c with prod macros,
ON and OFF variants side by side).

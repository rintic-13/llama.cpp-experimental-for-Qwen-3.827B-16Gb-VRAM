# FA turbo baseline — 2026-11-03 (post S1+S2, pre S3)

Immutable reference numbers for the turbo flash-attention work. Everything below was measured on THIS
machine with the build identified in §Identity, GPU idle (0 MiB used / 16050 free / 0 % before each run).
Any future FA change must be compared against these rows (same machine, idle GPU, same model/args).

## Identity

| item | value |
|------|-------|
| tree the numbers were taken on | `476e390` (S2 closed: fused KQ default ON for turbo5 K) |
| baseline marker | this file, tag `fa-turbo-baseline-20261103` |
| ggml-cuda.dll | 65 078 272 B, 2026-11-03 00:46, `-DGGML_CUDA_FATTN_TURBO_KQ_MMA_F16_DEFAULT=1` in 161 rules |
| llama.dll / llama-perplexity.exe / llama-bench.exe | 31.10.2026 (Oct-31 thin clients + old type table) |
| GPU | NVIDIA GeForce RTX 5060 Ti, cc 12.0, 36 SMs, 16 283 MiB |
| model | `G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` (qwen35 27B IQ3_S, 11.28 GiB, 27.32 B) |
| quant flags | `-DGGML_TURBO_DITHER` (production) + `sm_120a` |

## 1. Kernel-level decode time — `fa_turbo_ab_new.exe` (nq=1, nkv=32768, heads 24/4, med ms)

| row | K/V | D | med ms | min–max | t/s-equiv | regs | smem |
|-----|-----|---|--------|---------|-----------|------|------|
| T1 | turbo4–turbo3 (dot) | 256 | **1.412** | 1.411–1.414 | 708.0 | 255 | 4352 |
| T2 | q5_1–turbo3 (dot) | 256 | **0.674** | 0.667–0.678 | 1484.4 | 255 | 4352 |
| T3 | turbo4–turbo4 (dot) | 256 | 1.445 | 1.445–1.447 | 691.8 | 254 | 4352 |
| T4 | turbo5–turbo3 (dot) | 256 | **1.697** | 1.697–1.704 | 589.1 | 254 | 4352 |
| T6 | turbo4–turbo3 (FUSED) | 256 | 1.925 | 1.923–1.934 | 519.5 | 255 | 4864 |
| T7 | turbo4–turbo4 (FUSED) | 256 | 1.964 | 1.959–1.970 | 509.2 | 253 | 4864 |
| T8 | turbo5–turbo3 (FUSED) | 256 | **1.503** | 1.502–1.505 | 665.4 | 255 | 4864 |
| C1 | turbo4–turbo3 (dot) | 128 | **0.503** | 0.503–0.504 | 1988.3 | 254 | 2304 |
| C2 | turbo4–turbo3 (FUSED) | 128 | 0.855 | 0.850–0.861 | 1169.1 | 254 | 2816 |

NB the `regs`/`smem` columns come from `cudaFuncGetAttributes` inside the probe; the `cuobjdump` SHARED
figures in §1b are 1024 B higher in every row (static/align overhead) — the relative dot-vs-fused delta
is identical.

Head-to-head (the S2 speed gate):

| K | dot ms | fused ms | verdict |
|---|--------|----------|---------|
| turbo4 D256 | 1.412 | 1.925 | fused **+36.3 %** (rejected) |
| turbo4 D128 | 0.503 | 0.855 | fused **+70.0 %** (rejected) |
| turbo5 D256 | 1.697 | 1.503 | fused **−11.4 %** (accepted, default) |

### 1b. Resource usage per instantiation (`cuobjdump -res-usage` on `fa_turbo_ab_new.exe`)

Mangled form is `flash_attn_ext_vec<D, ncols, type_K, type_V, softcap, KQ_mma, KQ_f16dq>`; enum values in
this fork: 46 = turbo4, 45 = turbo3, 47 = turbo5, 7 = q5_1. `KQ_f16dq=1` means the fused f16-MMA path.

| row | kernel | REG | STACK | SHARED | LOCAL |
|-----|--------|-----|-------|--------|-------|
| T1 | t4–t3 D256 dot | 255 | 0 | 5376 | 0 |
| T6 | t4–t3 D256 FUSED | 255 | **64** | 5888 | 0 |
| T7 | t4–t4 D256 FUSED | 253 | **64** | 5888 | 0 |
| T3 | t4–t4 D256 dot | 254 | 0 | 5376 | 0 |
| T4 | t5–t3 D256 dot | 254 | 0 | 5376 | 0 |
| T8 | t5–t3 D256 FUSED | 255 | 0 | 5888 | 0 |
| T2 | q5_1–t3 D256 dot | 255 | 0 | 5376 | 0 |
| C1 | t4–t3 D128 dot | 254 | 0 | 3328 | 0 |
| C2 | t4–t3 D128 FUSED | 254 | **64** | 3840 | 0 |

SASS instruction mix (`cuobjdump -sass -fun <mangled>`, static counts):

| row | total instr | LDL+STL (spill traffic) | MMA | PRMT | IMAD | LOP3 | SHFL | FFMA |
|-----|-------------|-------------------------|-----|------|------|------|------|------|
| T1 t4–t3 dot | 12 488 | **0** | 0 | 0 | 1490 | 1842 | 199 | 1741 |
| T6 t4–t3 FUSED | 13 688 | **256** | 32 | 0 | 1694 | 1969 | 44 | 1390 |
| T4 t5–t3 dot | 14 448 | 0 | 0 | 0 | 1098 | 2609 | 199 | 1165 |
| T8 t5–t3 FUSED | 13 208 | **0** | 32 | 128 | 1040 | 2609 | 44 | 878 |

Reading: the fused kernel is strictly cheaper in the KQ block (MMA 32 instead of the dp4a/PRMT decode
chain, FFMA −20 %, SHFL 199 → 44) and its static instruction count is close to the dot kernel, yet only
the variant that carries a 64 B stack frame + 256 LDL/STL is slower. REG is 253-255 in EVERY variant,
i.e. the SM register file (65536/SM, 255/thread cap) is already fully allocated and occupancy is pinned
at 2 blocks/SM by registers in both paths (smem 5376 vs 5888 B is nowhere near the limit) - so extra
live state cannot cost occupancy, it can only spill to local memory. That is exactly what happens, and
it correlates 1:1 with the speed loss. Root cause pinned in §12 of `NVFP4_FA_RESEARCH.md`.

## 2. Model-level numerics — `ppl_quick_turbo.cmd`
(`-c 4096 --chunks 1 -b 1 -ub 1 -ngl 99 -fa on -t 7 -s 42`, file `ppl_eval\wiki.test.quick.txt`)

| row | K/V | routing | PPL | stderr |
|-----|-----|---------|-----|--------|
| T1 | turbo4–turbo3 | default | 7.1719 | 0.37892 |
| T2 | turbo4–turbo3 | NATIVE vec dot | 7.1802 | 0.37961 |
| T3 | q5_1–turbo3 | NATIVE vec dot | 7.1743 | 0.37943 |
| T4 | f16–f16 | gold | **7.1378** | 0.37703 |
| T5 | turbo4–turbo3 | NATIVE + FUSED | 7.1802 | 0.37961 |
| T6 | turbo5–turbo3 | NATIVE vec dot | 7.1920 | 0.38042 |
| T7 | turbo5–turbo3 | NATIVE + FUSED | **7.1868** | 0.38010 |

Anchors: T4 gold = 7.1378 (unchanged since the Oct reference runs, so the protocol is stable);
T5 − T2 = 0.0000 exactly (selector excludes turbo4 from fused); T7 − T6 = −0.0052 (fused numerics gate,
|Δ| ≤ 0.05 → PASS).

## 3. Production PPL with the flipped default (`ppl_prod_check.cmd`, NO env vars)

| row | K/V | PPL | must equal |
|-----|-----|-----|-----------|
| P1 | turbo4–turbo3 | **7.1719** | T1 (pre-flip production, bit-identical behaviour) |
| P2 | turbo5–turbo3 | **7.1868** | T7 (fused KQ now on by default) |

## 4. Routing (prefill path only) — `run_fa_turbo_routing.ps1`
(`-ngl 99 -fa on -t 7 -ub 512 -pg 512,128 -pg 4096,128 -r 2`, GGML_FATTN_DEBUG=1)

| row | K/V | routing | pp512 | pp4096+tg | tg128 | pp512+tg |
|-----|-----|---------|-------|-----------|-------|----------|
| R1 | turbo4–turbo3 | default (gate ON) | 954.02 ± 18.47 | **500.95** | 32.82 | 143.81 |
| R2 | turbo4–turbo3 | NATIVE vec | 955.37 ± 21.05 | 493.74 | 32.38 | 142.71 |
| R3 | q5_1–turbo3 | default (vec) | 953.72 ± 25.35 | **508.43** | 32.92 | 144.39 |
| R4 | q5_1–turbo3 | FALLBACK | 951.48 ± 28.60 | 502.82 | 33.09 | 144.48 |

Structural fact to remember: at nq=1 every row logs `kernel=vec`, only nq=512 logs `kernel=mma_f16` —
the routing env switches move PREFILL only, decode is always the native vec kernel.

## 5. End-to-end effect of the S2 flip — `bench_prod_s2.cmd`
(`-ngl 99 -fa on -t 7 -ub 512 -d 8192 -p 512 -n 128 -r 3`, turbo5–turbo3)

| mode | pp512 @ d8192 | tg128 @ d8192 |
|------|---------------|---------------|
| fused KQ ON (default) | 890.32 ± 10.98 | **27.28 ± 0.12** |
| forced dot (env `...TURBO_KQ_MMA_F16=0`) | 890.44 ± 11.27 | 26.79 ± 0.09 |

Decode +1.8 %, prefill unchanged. (The 11.4 % kernel-level win becomes 1.8 % end-to-end: at d8192 the
FA kernel is only part of the step.)

## How to re-measure

```
:: standalone kernel probes (no llama needed, ~0.1 GB VRAM)
build-x64-windows-msvc-release\bin\fa_turbo_frag_probe.exe     # expect 0 0 0
build-x64-windows-msvc-release\bin\fa_turbo_ab_new.exe         # §1 table
:: model level (needs idle GPU, ~11.3 GB, ~2.3 min per row)
ppl_quick_turbo.cmd
run_fa_turbo_routing.ps1
:: resource forensics for the register-pressure question
cuobjdump -res-usage build-x64-windows-msvc-release\bin\fa_turbo_ab_new.exe
```

CLI names carry NO `_0` suffix (`-ctk turbo4 -ctv turbo3`); only cmake `GGML_CUDA_FA_QUANTS` combo
strings use `turbo4_0-turbo3_0`. `-pg` takes one `pp,tg` pair per flag.

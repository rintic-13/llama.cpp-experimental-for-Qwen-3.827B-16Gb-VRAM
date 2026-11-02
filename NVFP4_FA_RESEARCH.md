# Hardware FP4 (NVFP4) tensor-core path for Flash Attention on SM120 — Stage 1 + 2 report

**Device:** RTX 5060 Ti (cc 12.0, 36 SMs, 2647 MHz).
**Probe:** `tools/fa_fp4_mma_probe.cu` — standalone, llama.cpp build untouched.
**Baseline compared against:** the fork's current KV-attention inner loop,
`vec_dot_fattn_vec_KQ_nvfp4` + `quantize_q8_1_to_shared` (`ggml/src/ggml-cuda/fattn-common.cuh`,
`fattn-vec.cuh`), replicated instruction-faithfully in the probe ("PathVEC", replica fidelity
2e-6 vs host double simulation of the same formula).
**Build:** nvcc with `--generate-code=arch=compute_120a,code=[sm_120a]` (plain `-arch=sm_120a`
silently emits `compute_120` and the FP4 mma fails to assemble).

---

## TL;DR — Stage 2 gate: PASS for QKᵀ, structural NO-GO for PV

> **Stage 3 update (2026-11-01, final): CLOSED for production.** The mma-KQ vec kernel was built
> and gated: PPL 8.2475 (single-pass FP4 Q) / 8.3840 (hi+lo residual split) vs 7.1289 q8_1 -
> both fail by >100x the ±0.01 tolerance despite 10.7x better isolated Q-error for hi+lo; decode
> gain only +2.7%/+9.3%. Kept in-tree behind `GGML_CUDA_FATTN_VEC_KQ_MMA` (default off), not to
> be extended. Full record: §9.

> **Next-direction result (2026-11-01, §10): the fused f16-MMA + in-register NVFP4-K-decode path is
> a GO.** Probed standalone (`k_path_fused` / `--mode bench` PathFUSED). Answers both posed
> questions: (1) KQ via Tensor Cores with **lossless Q (f32→f16 only)** runs **~8–9x faster than
> PathVEC** (11.7 vs 1.35 TF/s) and the mma feed adds *nothing* to the score error (rms 3.2e-6 vs
> the f16-reference); total score error is then just the K-side NVFP4 quantization (rms 2.27), vs
> 4.8 for the q8_1 vec path. (2) The NVFP4→f16 decode is **cheap and DRAM-bound** once PRMT-based
> (0.013 ms at 16K tokens; hidden behind the mma). No FP16×FP4 mixed MMA exists on SM120 (mixtest:
> ptxas rejects f16×e2m1/e4m3) - decode-to-f16 + dense m16n8k16 is the correct design. Integration
> into fattn-mma-f16 (replace the f16 K tile load with the same decode feed) is the next step.

| Metric (D=256, ntok=16384, nq=64) | PathVEC (current) | PathMMA (hw FP4) | Ratio |
|---|---|---|---|
| QKᵀ tile throughput | 1.35 TFLOP/s | **20.0–22.4 TFLOP/s** | **14.8–16.8×** |
| KV bytes consumed | 5.9 GB/s | 88 GB/s | — |
| Score rms err vs true fp32 QKᵀ | 4.82 (incl. repo formula approx.) | **3.22** | MMA more accurate |
| Score rms err vs codec-exact nvfp4 K | 4.75 (pure formula error) | **0** | exact |

* QKᵀ via `mma.sync...kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` is **feasible, ~11–17×
  faster than the vec path on the same tiles, and numerically better** than the current q8_1 vec path.
* The full GPU chain — **fp32 Q → device nvfp4 Q-prep → unmodified GGML KV bytes → in-kernel
  nibble repack → hardware MMA — is bit-exact end-to-end** against the host reference (all probes).
* PV with block-scaled MMA is **structurally impossible without changing V's storage format**:
  nvfp4 UE4M3 scales for V vary per *token* while PV's reduction dimension is the token axis —
  `block_scale` requires scales invariant along the M×K / K×N fragment reduction. (Detailed §6.)
* Decode is **not attention-bound** at these depths (measured earlier: f16-KV decode is *faster*
  than nvfp4 decode: 32.4 vs 30.1 t/s @ 8K), so the win applies mainly to **long-context prefill**
  (which today uses `mma_f16` at ~6.7 TF/s with f16 dequant scratch) — PathMMA at 20 TF/s with
  zero f16 scratch beats it too.
* **Status 2026-11-01:** the vec-kernel wiring (step 3) is **implemented and compiled** behind
  `GGML_CUDA_FATTN_VEC_KQ_MMA=1` (default off). First A/B: decode tg **30.27→31.35 t/s @8K
  (+3.6%), 23.63→26.35 @32K (+11.5%)** — the mma variant runs and wins. But the numerics gate
  **FAILED**: ub=1 PPL on the same window 7.1289 → 8.2475 (**Δ = +1.12**, gate was 0.01) — nvfp4
  Q requant clips Qwen3.5's outlier Q coords (E2M1 range ±6). Flag stays OFF; a fix would be
  residual-split Q (two mma chains, §8 record). PPL/prefill routing: quantized K/V only reach the
  vec kernel for `Q->ne[1] ≤ 2`, so numerics must be measured with PPL `-b 1 -ub 1`.

---

## 1. Confirmed SM120 capability matrix (assembled + run on this device)

Confirmed **supported** (compiled `sm_120a`, executed, numerically verified):

| PTX instruction | Notes |
|---|---|
| `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.ue4m3.f32.e2m1.e2m1.f32` | the nvfp4 path; **403.6 TFLOP/s** peak issue rate |
| `mma.sync...kind::mxf4.block_scale.scale_vec::2X...ue8m0` (mxfp4) | supported (not further used) |
| `mma.sync...kind::mxf8f6f4...e4m3 × e2m1` (k=32, ue8m0 scales) | supported, 201.8 TF/s — but numerically disqualified, §5 |
| fp16 `m16n8k16` 50.5 TF/s, fp8 dense `m16n8k32` 101.1 TF/s | sanity anchors (fp16 matches spec; fp4 = 2× fp8 = 4× fp16 issue rate) |

Confirmed **not supported** on `sm_120a` (assembler or runtime rejection):

* `tcgen05.mma` (Blackwell datacenter instruction — absent on consumer SM120);
* f16 A-operand × e2m1 B-operand mixed mma;
* FP4 mma with anything but f32 accumulator;
* nvf4 `m16n8k32`, `scale_vec::1X/2X` with ue4m3, `.col.col` / `.row.row` variants.

Implication: the **only** hardware FP4 shape usable for this KV layout is
`mxf4nvf4 · scale_vec::4X · ue4m3 · m16n8k64`, i.e. exactly one GGML `block_nvfp4` row (64 coords)
per mma K-depth, with the 4 UE4M3 sub-block scales mapping directly onto the 4 scale_vec slots.

## 2. Fragment + scale-register layout (PTX ISA §9.7.16.5.11/§9.7.16.3, confirmed empirically)

For `g = lane>>2, t = lane&3`:

* **A** (Q, m16×k64, e2m1): `a0` = (row g, k 8t..8t+7), `a1` = (row g+8, same k),
  `a2`/`a3` = same rows at k 8t+32..39 / +32.
* **B** (K, n8×k64): `b0` = (col g, k 8t..8t+7), `b1` = (col g, k 8t+32..).
* **C** f32: standard m16n8 pattern.
* **SF_A** (b32 = 4 UE4M3 bytes, LE): only lanes `4g+0` (→ row g) and `4g+1` (→ row g+8) are read;
  bytes = sub-vectors 0..3 in increasing k-slot order.
* **SF_B**: only lane `4g` is read (→ col g); bytes = 4 sub-vectors.

This matches MMQ's existing scale indexing (`lane/4 + (lane%2)*8`, `lane/4`) in `mmq-vec-dot.cuh`.
Verification (probe `--mode verify`, `--mode map`, `--mode scale`):
one-hot sweep ok=32768 / wrong=0; uniform scales give C=64 exactly (k64 depth real);
sub-vector pairing constant test hits (1360) exactly. **`ggml`'s nvfp4 sub-block ↔ PTX scale_vec
pairing is 1:1 and in-order.**

## 3. Packing finding: GGML nibble order ≠ hardware nibble order

Hardware expects byte *i* = FP4 coords (2i, 2i+1) (consecutive pairs). GGML `block_nvfp4` stores
byte *i* = coords (c+i, c+i+8) (halves of the 16-coord sub-block). Scale bytes map identically;
the gap is a **fixed nibble permutation inside each 64-coord block** (equivalently a prmt/lop3
shuffle over the 32 payload bytes). The prototype repacked tiles host-side once
(`ggml_to_native_words`) and stored them fragment-native, so **measured MMA speed carries zero
repack cost**. Integration must add either

* (a) load-time repack in the FA kernel: ≈ 2–4 `prmt`/`lop3` per 16 B = ~8–16 cheap int ops per
  token-row — small vs 8 mma k64 per 64 coords; or
* (b) fragment-native KV encoder order (touches cache format — rejected per requirements).

The sweep numbers below are therefore an *upper bound* on integration speed; a repack-aware
loader can be prototyped cheaply before wiring into FA.

### RESOLVED (repack-aware loader, prototyped + measured)

`k_path_mma<D, REPACK=true>` stages **unmodified GGML `block_nvfp4` words** straight out of the
KV buffer and permutes nibbles in registers during the smem copy — 14 int ops per 8-byte group
(`lop3` masks + two `__byte_perm` gathers + shift/or; `ggml_pair_to_native` in the probe).
Verified **bit-exact vs the host-repacked path at every sweep point** (0 mismatches / 262144
scores), and it is **not slower — usually faster**: pair-wise staging (2 words read, 2 words
written per thread) coalesces better than the native path's single-word copy, which more than
pays for the permutation:

| ntok | nq | host-repacked | **ggml bytes + in-kernel repack** | bit-exact |
|---|---|---|---|---|
| 4 096 | 16 | 4 306 | 4 286 | ✓ |
| 4 096 | 64 | 13 465 | 13 855 | ✓ |
| 16 384 | 16 | 12 861 | 12 829 | ✓ |
| 16 384 | 64 | 19 790 | **22 425** | ✓ |
| 65 536 | 16 | 18 312 | 19 809 | ✓ |
| 65 536 | 64 | 20 506 | **24 051** | ✓ |

**KV storage format needs no change; the packing gap costs zero.** Option (b) is off the table;
integration reads `block_nvfp4` bytes as-is.

## 4. Prototype QKᵀ kernel + measured sweep

`k_path_mma<256>`: one warp owns one 64-token × 8-col slab via 8×(m16n8k64) mmas over D=256
(=4 k64 steps × 2 col groups… fully unrolled); K tile + Q tile + both scale sets staged in smem
(fragment-native), online-softmax-compatible f32 accumulators, C-fragment → (row, col) store.
Grid = `ntok/64` blocks × 8 warps — deliberately naive (prototype), no async copy, no double
buffering, K re-read per block. **PathVEC replica uses the identical tile grid and the repo's exact
word window + ds=(d, Σx) formula.**

Raw issue rate (no memory, DCE-proof store):

```
fp4  nvf4 4X m16n8k64       403.56 TFLOP/s  (24.63 G-mma/s)
fp8xfp4 mxf8f6f4 k32        201.82 TFLOP/s
fp8  dense m16n8k32         101.08 TFLOP/s
fp16 m16n8k16                50.54 TFLOP/s
```

Tile bench (D=256, reps=30):

| ntok | nq | PathVEC GF/s | PathMMA GF/s | speedup | PathMMA % of peak issue |
|---|---|---|---|---|---|
| 4 096 | 16 | 954 | 4 110 | 4.3× | 1.0% |
| 4 096 | 64 | 1 227 | 13 756 | 11.2× | 3.4% |
| 16 384 | 16 | 1 189 | 12 854 | 10.8× | 3.2% |
| 16 384 | 64 | 1 350 | **20 012** | 14.8× | 5.0% |
| 65 536 | 16 | 1 278 | 18 484 | 14.5× | 4.6% |
| 65 536 | 64 | 1 435 | **20 495** | 14.3× | 5.1% |

At ntok=65536 the MMA path consumes KV at **325 GB/s (nq=16)** — above the 5060 Ti's effective
memory bandwidth regime, i.e. at small nq the tile bench is already bandwidth-limited, which is
exactly where a production FA kernel (bigger nq tiles, async copies) would sit.

With the in-kernel GGML repack (§3) the best case rises to **24.1 TFLOP/s (ntok=65536, nq=64,
16.8× over PathVEC)** and 348 GB/s KV at nq=16.

## 5. Numerics (probe `--mode numerics`, synthetic Gaussian, D=256, double-accumulated reference)

Score error vs true fp32 QKᵀ (mean|score| ≈ 19.0):

| Q codec × K codec | rms | max | note |
|---|---|---|---|
| fp32 Q × nvfp4 K (K-only floor) | 2.27 | 11.7 | floor of the current K storage |
| q8_1 Q × nvfp4 K, exact product (current path, idealized) | 2.27 | 11.7 | q8_1 Q adds ~nothing on this data |
| **nvfp4 Q × nvfp4 K (= PathMMA)** | **3.22** | 13.1 | requantizing Q to FP4 costs +~1 rms |
| e4m3 Q × **mxfp4** K (the mxf8f6f4 route) | 16.6 | 76.2 | **disqualified**: mxfp4 needs ue8m0 (pow-2) K scales per 32 — would force K re-encode and destroys K accuracy |

Two views of the same fact, both relevant:

* **vs what the repo actually computes today** (`vec_dot_fattn_vec_KQ_nvfp4`): PathMMA is *more*
  accurate. The fork's q8_1 KQ formula itself injects ~4.7 rms approximation error — the
  `(Σx/32)·Σc` mean-shift term approximates the quantization-residual mean — and PathMMA scores
  0 against the codec-exact reference (mma is exact on the stored bytes).
* **vs an idealized exact q8_1 product**: nvfp4-Q adds a modest rms 2.27→3.22 (+41% on synthetic
  data). Whether that matters for logits is a **PPL question (Stage 3 gate)**, not a probe question.

## 6. PV feasibility — structural no-go for block_scale MMA

QKᵀ reduces over `head_dim`: nvfp4 scales are per (token, 16 coords) — invariant along the
reduction **only for the K operand of QKᵀ**, where they vary along head_dim = the mma's K axis,
which `block_scale` natively handles (scale_vec::4X per k64). ✓

PV reduces over **tokens** (mma K axis = token index). V's UE4M3 scales vary with the token
(every 16 tokens the scale changes along the mma K axis) — `block_scale` scales may only vary
along M/N/K *within* a fragment's fixed scale vector, and `scale_vec::4X` on k64 means one scale
per 16 *K-slots*: for PV those 16 K-slots are 16 tokens × one coordinate, but a V row's scale for
coord *d* depends on *which 16-token sub-block the token falls in*, i.e. differs per lane and
per column simultaneously. No fragment mapping, lane binding or repacking resolves this —
it would require transposing V to coord-major blocks (changing the KV cache format, forbidden)
or per-token requantization to ue8m0 pow-2 scales (numerically disqualified above, plus a per-token
reduction pass). **PV must stay on dequant-to-f16 `mma_f16` (today's path) or vec, with V's UE4M3
scale folded at dequant time.** No hardware-FP4 path for PV exists on SM120 without format change.

## 7. System context from earlier measurements (unchanged)

* PPL (nvfp4 KV): 5.0519 (nvfp4-nvfp4) vs 5.0322 (q5_1 K / nvfp4 V).
* Decode t/s at 8K: f16-KV 32.39 > nvfp4-KV 30.13 → decode is **not** attention-throughput-bound;
  FP4-FA will not move decode on this card. Report it, don't promise it.
* Prefill at d=512: FA runs `mma_f16` on f16 dequant scratch (≈21 ms of the profile, ~6.7 TF/s).
  PathMMA replaces that stage's KQ half at 20 TF/s and removes the f16 K scratch entirely
  (K never leaves e2m1+UE4M3 until accumulation).

## 8. Stage-3 integration plan (gated proposal, not yet done)

1. ~~**Repack-aware loader first**~~ — **done** (see §3 RESOLVED): unmodified GGML bytes,
   in-kernel permutation, bit-exact, zero performance loss. GGML KV format stays as-is.
2. ~~**Q→nvfp4 prep in FA**~~ — **device kernel prototyped and bit-exact** (`k_quant_nvfp4` in
   the probe): fp32 rows → unmodified GGML `block_nvfp4` words entirely on-GPU. Bit-exact vs
   `nvfp4_quant_row` ref (all code words + scale words), and the full chain
   **fp32 Q → GPU prep → in-kernel repack → hw MMA is bit-exact vs the host reference**.
   Nearest-E2M1 is a closed-form chain of `a < mid*dd` comparisons: every midpoint × a 4-bit
   UE4M3 mantissa is exact in fp32, so it reproduces the ref's scan tie-breaks (ties → smaller
   magnitude; `-a` at a 0.25 tie → code `0`) exactly. Standalone launch cost ≈ 8 µs
   (launch-latency-bound); in FA it folds into the existing Q-prep stage. Along the way a probe
   UB was fixed: word-stride D/4 arrays were being filled past the D/8 real data words with
   next-row bytes (padding now zeroed; MMA never reads it).
3. ~~**Wire PathMMA as the KQ stage of the vec FA kernel**~~ — **done** (2026-11-01, compiled
   in the repo build + smoke-routed on-device; gate measurements pending, see record below):
   `GGML_TYPE_NVFP4` K only, FP32 accumulation and online softmax kept, PV stage untouched.
4. **Stage-3 gate** (harness `run_fa_mma_ab.ps1`): speed measured (B: +3.6/+11.5% decode);
   numerics measured and **FAILED on 2026-11-01** (ub=1 PPL 7.1289→8.2475, Δ = +1.1186,
   gate 0.01) — see record below. The routing facts: quantized K/V reach the vec kernel
   **only for `Q->ne[1] ≤ 2`** and `GGML_CUDA_FATTN_VEC_NATIVE` gates vec-case *availability*,
   not kernel selection — so a ub=512 PPL A/B is a no-op; the numerics A/B runs PPL with
   `-c 4096 --chunks 1 -b 1 -ub 1` (every forward nq=1 → vec; NB `-c` = ctx-size, `--chunks`
   = chunk limit) plus the greedy-generation diff (§8 D).
5. Risks to watch: real Q outliers vs synthetic Gaussian (PPL is the arbiter); smem pressure at
   D=256 with double-buffered tiles; correctness of the in-kernel repack (guard with the existing
   `--mode verify` vectors ported into a kernel self-test).

### Step-3 implementation record (2026-11-01)

**Feature** (`ggml/src/ggml-cuda/fattn-vec.cuh`, template param `KQ_mma`): per K-tile, each warp
stages its 32 K-token rows once into fragment-native words (`ggml_cuda_fp4_repack_pair`, §3) +
per-64-coord scale dwords read verbatim from the cache, runs `mma...kind::mxf4nvf4.block_scale.
scale_vec::4X.m16n8k64` via `ggml_cuda_mma::mma_block_scaled_fp4<GGML_TYPE_NVFP4>`
(2 m-tiles × D/64 k-frags), Q rows are requantized to nvfp4 in smem once per block with the
bit-exact closed-form codec from §8.2 (B-side, up to 8 query rows), raw fp32 scores land in a new
`F4_raw` smem tile (the `KQ` tile may be half) with the attention `scale` applied at write, then
softcap/mask/max/exp/PV continue through the original code. SF lanes: A scale dword from lanes
4g+0 (row g) / 4g+1 (row g+8), B from lane 4g (§2). Gate: compile-time `type_K == NVFP4 && D >=
128`, runtime `blackwell_mma_available(cc) && getenv("GGML_CUDA_FATTN_VEC_KQ_MMA")`. **Default
off**; q8_1-KQ vec path stays selectable and untouched.

**Mandatory pre-build fix** (`fattn-mma-f16.cuh`): the fork contained a half-finished "Stage B1"
nvfp4-direct K loader that had never compiled (its source was newer than the newest matching .obj,
broken since 2026-10-31 22:59). Three defects: `flash_attn_ext_f16_process_tile` got an 11th
template param `kv_nvfp4` that its 3 call sites never passed (deduction failure); the loader body
referenced an undeclared `k`; and once unreferenced (constexpr-dead branch), cudafe copies such a
`__device__` function verbatim to the MSVC host pass and normalizes `__restrict__`→`restrict`,
which cl rejects in C++ (C3646/C2059 at the signature). Fixed by plumbing `kv_nvfp4` through
`flash_attn_ext_f16_iter` and passing `false` at the call sites, and removing the unfinished
loader + its branch. Semantics identical to the last known-good binary (nvfp4 KV reaches the
mma_f16 kernel dequantized to f16 scratch anyway).

**Measured so far** (full ggml-cuda + tools build OK, CUDA 13.2 sm_120a; first A/B in
`fa_mma_ab_20261101_032449`, quick file, 2 reps):
- `GGML_FATTN_DEBUG=1` routing: decode (nq=1) nvfp4-nvfp4 D=256 → `kernel=vec`; with the flag on
  the mma variant launches and runs clean (tg8@d1024 = 27.33 t/s, 1 rep). Prefill (nq=8) →
  `kernel=mma_f16`.
- Routing fact (corrects an earlier note in this doc): `get_best_fattn_kernel` picks vec for
  quantized K/V **only when `Q->ne[1] ≤ 2`** (Ada+); `GGML_CUDA_FATTN_VEC_NATIVE` only gates vec
  *case availability* (the turbo4 speed gate), it does NOT change kernel selection. Therefore a
  ub=512 PPL/prefill run never enters the vec kernel and the flag is inert there: the identical
  quick-PPL pair (6.1025 vs 6.1025) and the flat pp2048 trio (950.04 / 948.47 / 948.63 t/s) are
  routing artifacts, not flag failures.
- Decode tg A/B (default routing = vec, flag off → on): **d8192 30.27 → 31.35 t/s (+3.6%),
  d32768 23.63 → 26.35 t/s (+11.5%)**. The mma variant demonstrably runs and its gain grows with
  context length, as expected when KQ share of decode time grows.
- Numerics gate: **FAIL**. ub=1 PPL A/B, identical window (ctx 4096, 1 chunk, quick file):
  q8_1-KQ **7.1289** vs mma-KQ **8.2475**, Δ = **+1.1186** ≫ 0.01. The different numbers prove
  the mma path ran. Interpretation: §5 predicted nvfp4-Q adds ~+1 score-rms on Gaussian data
  (2.27→3.22, mean|score|≈19), and §8-5 flagged real-Q outliers vs synthetic Gaussian as the
  risk — PPL is the arbiter and it says the E2M1 dynamic range (±6 per UE4M3 block) clips
  Qwen3.5's outlier coordinates hard. Speed (+3.6/+11.5% decode) is real but only worth
  keeping with Q-accuracy compensation: **residual-split Q** (`Q_hi=nvfp4(Q)`, `Q_lo=nvfp4(Q−
  dequant(Q_hi))`, two mma chains, same K bytes) reduces the nvfp4-Q error to second order;
  ~2× KQ mma work, decode gain roughly halved. Undecided (2026-11-01).

**Gate A/B commands** (from repo root; unset vars afterwards):

```powershell
# A) numerics: ub=1 => every forward is nq=1 => vec kernel (NATIVE is irrelevant for routing).
# NB: llama-perplexity -c is --ctx-size; the chunk limit is --chunks. -c 1 would be ctx=1!
$env:GGML_CUDA_FATTN_VEC_KQ_MMA = ""    # then "1" for the second run; swap order too
.\build-x64-windows-msvc-release\bin\llama-perplexity.exe -m G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf `
  -f ppl_eval\wiki.test.quick.txt -c 4096 --chunks 1 -b 1 -ub 1 -ngl 99 -fa on -ctk nvfp4 -ctv nvfp4 -t 7 -s 42
#   4096 single-token forwards per run (~2-4 min). Final gate number: --chunks 4 over wiki.test.raw.

# B) decode speed (default routing already picks vec):
.\build-x64-windows-msvc-release\bin\llama-bench.exe -m G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf `
  -fa on -ngl 99 -p 8 -n 128 -d 8192 -r 2 -ctk nvfp4 -ctv nvfp4      # flag off, then flag on

# C) prefill speed: with ub=512 ALL combos route to mma_f16 (vec is never picked for prefill),
#    so C is a stability reference only. FP4-mma prefill would need a nvfp4-direct K loader
#    inside fattn-mma-f16 (the fork's abandoned Stage B1), not a routing change.

# D) greedy generation diff: llama-completion -p <fixed prompt> -n 256 -s 42 --temp 0, flag off/on;
#    outputs must DIFFER (Q requant changes scores) and stay plausible.

Remove-Item Env:GGML_CUDA_FATTN_VEC_NATIVE, Env:GGML_CUDA_FATTN_VEC_KQ_MMA -ErrorAction SilentlyContinue
```

Expectation after the first A/B: §B already shows the win (decode is weights-bound, so +3.6%
@8K is consistent with KQ being a minority of decode time; +11.5% @32K shows it grows with
context). §C cannot show anything until `fattn-mma-f16` gains a nvfp4-direct K loader — the §4
probe numbers (15-17× on large batches) apply to that loader, not to vec routing. The remaining
gate question is numerics: §A (ub=1 PPL) + §D (greedy diff).

### Residual-split Q study (2026-11-01, probe `--mode residual`)

Gate A failed (ΔPPL +1.1186) while D's greedy outputs stayed fully fluent on both sides
(divergence at char 151 = the reasoning trace drifting, no corruption at ctx ~300) — consistent
with "Q requant cost", not kernel breakage. To isolate the Q side (K bytes and K scales are
identical in every row, so the K floor cancels; reference = fp32 Q × dequant(nvfp4 K), fp64
accumulation):

```
Q_hi = nvfp4(Q);  Q_lo = nvfp4(Q − Q_hi);  scores = Q_hi·K + Q_lo·K   (same K fragments)
```

Synthetic results (probe, D=256, 256 Q rows × 2048 K rows; K rms 1.5):

| Q distribution | max-E2M1-bin coords | Q err pass1 → pass1+2 | score rms pass1 → pass1+2 | improvement |
|---|---|---|---|---|
| gaussian | 11.3% | 9.45% → 0.86% rms\|Q\| | 2.25 → 0.20 (mean\|score\| 19.4) | **11.0×** |
| 10% ×6 outliers | 8.6% | 9.18% → 0.86% | 4.57 → 0.43 (mean\|score\| 39.8) | 10.7× |
| 5% ×20 outliers | 8.7% | 9.02% → 0.87% | 10.1 → 0.98 (mean\|score\| 86.8) | 10.3× |

The ~10× error reduction is stable across outlier strength — the residual pass also absorbs
saturation clipping (its block scales adapt to the clipped residual). Pass1+2 Q error (~0.86%)
is at q8_1-Q level; score error then ≈ the K floor alone, i.e. strictly better than the current
q8_1-KQ vec path (§5: that path itself injects ~4.7 rms). Prediction if integrated: PPL Δ vs
q8_1-KQ within gate, decode KQ cost doubles (gain roughly halves).

Real-activation check prepared: `fattn.cu` gained an opt-in, numerics-neutral Q dump
(`GGML_CUDA_FATTN_QDUMP=<file>` — first head, ≤8 FA calls × ≤1024 token rows of D fp32), and the
probe consumes it via `--qfile`. Commands:

```powershell
# dump Q from layers 0..7 (head 0) during one prefill pass (--no-warmup matters: warmup
# tokens otherwise consume the dump slots with empty-run activations):
$env:GGML_CUDA_FATTN_QDUMP = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main\ppl_eval\Q_real.f32"
Remove-Item "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main\ppl_eval\Q_real.f32" -ErrorAction SilentlyContinue  # file appends, delete old first
.\build-x64-windows-msvc-release\bin\llama-perplexity.exe -m G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf `
  -f ppl_eval\wiki.test.quick.txt -c 4096 --chunks 1 -b 512 -ub 512 --no-warmup -ngl 99 -fa on -ctk nvfp4 -ctv nvfp4
Remove-Item Env:GGML_CUDA_FATTN_QDUMP
# then:
.\build-x64-windows-msvc-release\bin\fa_fp4_mma_probe.exe --mode residual --qfile ppl_eval\Q_real.f32
```

Decision point: if real-Q shows the same ~10× (and the max-bin share is high, confirming
saturation as the dominant mechanism), implement the two-chain variant in `fattn-vec.cuh`
(second A fragment set + second SF dwords per k64; K-side loads/scales unchanged) and re-run the
full A/B+E gate. Otherwise close stage 3 negative.

First real-Q sample (layers 0-3, head 0, but only 8 rows - the slots were consumed by llama's
warmup run; re-dump with `--no-warmup` and 8-call cap pending):
`max-bin 10.11%, Q err 9.06% -> 0.86% (10.5x), score rms 1.28 -> 0.125` — the synthetic picture
holds on real activations (saturation is real, residual split fixes ~10x).

Full real-Q dump (2026-11-01, 4096 rows: layers 0-7 x 512 real-prefill tokens, head 0,
`--no-warmup`):
`max-bin 10.50%, Q err 9.51% -> 0.888% (10.7x), score rms 1.142 -> 0.107, max 16.87 -> 1.43`.
Verdict: mechanism confirmed (saturation + E2M1 grid error, ~10% of coords in the max bin),
residual split recovers ~10x on real activations across all layers → **decision: implement
two-chain Q_hi+Q_lo in the vec kernel**. Done same day (`fattn-vec.cuh`): Q prep now codes hi
into a 16-code array per sub-block lane, derives `rv = v - dequant(codes)` and quantizes it with
the same closed-form codec under its own UE4M3 scales (new smem `F4_Qnc2/F4_Qsc2`, ~1.2 KB at
D=256); the KQ loop runs `mma_block_scaled_fp4` twice per (mt, f) - same A/K fragments and SF_A,
second B set + SF_B dword, both chains accumulate into the same zero-initialized f32 `Dt`.
`GGML_CUDA_FATTN_VEC_KQ_MMA=1` now selects the hi+lo variant (single-pass variant removed - it
failed the gate). K-side loads, KV format, softmax, PV untouched. Gate re-run pending: A) ub=1
PPL (expect ~7.13±0.01 vs 7.1289 q8_1 baseline; 8.25 single-pass must not come back),
B) decode tg (expect +~2%/+~6%, halved from single-pass), D) greedy diff.

## 9. Stage-3 gate result + branch closure (2026-11-01, final)

hi+lo gate run (`fa_mma_ab_20261101_145814`):

| metric | q8_1-KQ vec | FP4-mma KQ (hi+lo) | verdict |
|---|---|---|---|
| PPL (ub=1, 4096-tok window) | 7.1289 | **8.3840** (dPPL **+1.2551**) | FAIL (also worse than single-pass 8.2475 / +1.1186) |
| decode tg d8192 | 30.27 t/s | 31.10 t/s (+2.7%) | real but small |
| decode tg d32768 | 23.62 t/s | 25.81 t/s (+9.3%) | 2nd mma chain costs ~half the gain |
| greedy 256 tok | fluent | fluent, diverges @char 151 | no corruption, numerics-only |

Fact pattern: the host probe shows hi+lo cuts real-Q representation error **10.7x** (score rms
1.14 -> 0.107), yet in-kernel PPL got *worse* than single-pass. The extra Q_lo mma chain therefore
injects comparable-magnitude noise in the kernel - i.e. the defect is in the kernel port, not in
the codec. A bit-exact probe<->kernel comparison (`--mode kernelcheck`, replica of the FA prep +
two-chain mma loop) was started: the device replica diverged from the host reference and the run
hit a host-side heap crash before the score-comparison section completed. The replica itself was
unproven, so kernelcheck is **inconclusive as evidence**; debugging it further is explicitly
dropped by decision below.

**Decision (2026-11-01): the FP4-Q MMA-KQ branch is CLOSED for production integration.**
Rationale: both Q codecs fail the PPL gate by >100x the tolerance (dPPL +1.12 single-pass,
+1.26 hi+lo despite 10.7x better isolated QK error); the speed envelope is modest (+2.7%/+9.3%);
accuracy work on Q raised kernel complexity without reaching acceptable quality, and remaining
fixes would need open-ended bit-level debugging of the kernel port.
The implementation stays in the tree behind `GGML_CUDA_FATTN_VEC_KQ_MMA` (env, default off) as
documented experimental code - not routed, not gated, not to be extended.

**Preserved artifacts** (all reusable for the next attempt):
- `mma_block_scaled_fp4` + verified m16n8k64 fragment/scale-lane map (docs SS2, probe `verify`/`map`/`scale`);
- in-kernel GGML->hardware nibble repack on the K side - works with **no KV storage format change** (SS3, probe `bench` rp check);
- probe modes `verify/map/scale/numerics/residual/bench` + `--mode kernelcheck` (skeleton kept, abandoned mid-debug);
- Q-dump hook `GGML_CUDA_FATTN_QDUMP` and the real-activation stats (4096 rows, layers 0-7:
  max-E2M1-bin share 10.5%, nvfp4-Q err 9.5% rms - the quantizer-quality ceiling of this approach);
- measured speed envelope of the fp4 tensor-core path on this device (up to +11.5% decode @32K single-pass).

**Next research direction (separate, agreed):** quantize Q only to a much higher-precision
operand (or keep it lossless), dequantize nvfp4 K *inside* the fused kernel, and use tensor-core
MMA f16/bf16 x f16/bf16 (f32 accumulation) for KQ. Constraint to design around: the vec path's
win comes from reading the nvfp4 cache bytes directly; staging K as f16 in smem (32 tok x D=256 =
16 KB at D=256) is feasible in smem but forfeits the f16-cache-read comparison on traffic, so the
architecture must fuse dequant into the mma feed (and/or use the mixed-kind mxf8f6f4 paths whose
fp4 issue rate - 2x fp8, 4x fp16 - is already measured in `--mode bench`).

## 10. Fused f16-MMA path with in-register NVFP4 K decode (2026-11-01, RESULT: GO)

Answers the two questions posed for the next direction: (1) can KQ be sped up via tensor cores
without Q precision loss - **yes, ~8-9x over PathVEC**; (2) how expensive is NVFP4 decode inside
attention - **cheap when PRMT-based; DRAM-bound, essentially hidden**. Probe: `fa_fp4_mma_probe.cu`
`--mode bench` (PathFUSED block), kernels `k_path_fused<MODE,TPR,QGMEM>`.

### 10a. Mixed-precision capability check (`tools/fa_mma_mixtest.cu`)

`mma.sync kind::mxf8f6f4 block_scale scale_vec::1X m16n8k32` operand pairs, ptxas sm_120a verdict:

| A x B | compiles |
|---|---|
| e2m1 x e2m1 | YES |
| e4m3 x e2m1 | YES (also in probe bench mode 3) |
| e4m3 x e5m2, e5m2 x e2m1 | YES |
| **f16 x e2m1, f16 x e4m3, e2m1 x f16** | **NO - "Unexpected instruction types"** |

There is **no FP16xFP4 mixed MMA** on SM120; the "ideal candidate" row in the planning table is
hardware-impossible. Dense f16 m16n8k16 + fused decode (10b) is the surviving design.
Side findings (k32 kind): ue8m0 scale byte `0x81` measured = 1.0 for uniform fills (+1 vs the
PTX-spec 2^(b-127) convention, scale_vec::1X); the V3 e4m3/e2m1 uniform-fill sweep shows a
persistent -4 sign/factor vs operand expectations, and the mixtest one-hot fragment-map sweeps
disagreed for the k32/k16 slot assumptions - **mixtest numeric tables are UNVERIFIED harness-side
and parked**; the capability verdict (which instructions exist at all) is independent of that and
firm. The production f16 k16 fragment map was instead verified end-to-end through the fused
kernel itself (below).

### 10b. Fused kernel (probe PathFUSED): f16 Q x NVFP4-K-decoded-to-f16-fragments

Pipeline per warp: coalesced load of UNMODIFIED ggml block_nvfp4 words -> decode E2M1 nibbles to
f16 half2 + multiply by UE4M3 scale -> store fragment-native half2 words in smem -> dense
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (A = 16 Q rows, B = 8 token columns) ->
scores. **No f16 KV ever exists in global memory**; Q f32->f16 conversion is the only Q rounding.

Key implementation facts (all confirmed on device):
- PTX m16n8k16 f16 fragment map: a0=(row g, k 2t,2t+1) a1=(row g+8, same) a2/a3 = k 8+2t; b0=(col g,
  k 2t,2t+1) b1 = k 8+2t; C same map as the verified mxf4nvf4 k64 C map. Fragment smem word index
  = `8f + t` / `8f + t + 4` (word = half2 of coords 2w,2w+1), i.e. the same addressing idiom as the
  verified k64 nvfp4 loader with word = 2 coords instead of 8 nibbles.
- **ggml nvfp4 subblock byte pairing gotcha:** byte b of a 16-coord subblock holds coords (b, b+8),
  NOT consecutive pairs (first attempt assumed 2b,2b+1 -> rms err 31.9; fixed pairing -> exact).
- E2M1->f16 via **PRMT with table regs** `0x3E3C3800`/`0x46444240` (magnitude high bytes, low byte
  always 0, sign OR at bit 15): constant-LUT indexing serialized on divergent nibble indices;
  PRMT took decode-only from 0.061 ms to 0.012-0.035 ms (2-5x). Decode = PRMT + OR + HMUL2 per coord-pair.
- Scale = `ue4m3_hw(d)` raw UE4M3 as half (exact; max 448*6 < 65504) applied per coord-pair with HMUL2.

### 10c. Measured (D=256, RTX 5060 Ti)

| tile bench | ms | GF/s | note |
|---|---|---|---|
| ntok=16384 nq=64: PathVEC (q8_1) | 0.397 | 1 351 | baseline |
| PathMMA fp4 (closed branch) | 0.023 | 22 849 | for reference only |
| FUSED TPR=32 Q-in-smem | 0.052 | 10 373 | |
| **FUSED TPR=16 Q-from-gmem-f16** | **0.045** | **11 733** | best; ~= the dense-f16 issue envelope (bench mode 1), decode hidden |
| FUSED decode-only (no mma) | 0.013 | - | = DRAM-roofline read of the nvfp4 K bytes |
| ntok=65536 nq=16: PathVEC | 0.421 | 1 277 | |
| FUSED TPR=16 Q-gmem | 0.056 | 9 650 | decode part 0.035 ms = 269 GB/s, DRAM-bound |

Numerics (fused vs double references, n up to 32768 scores):
- vs fp64(f16-rounded Q x exact-decoded NVFP4 K): **rms 3.2e-6, max 2.7e-5** -> pure f32 accumulation
  noise; the mma path itself adds nothing (products of f16 pairs are exact in f32).
- vs UNQUANTIZED fp32 Q x raw fp32 K: rms 2.27 = **the K-side NVFP4 error only** (identical to the
  production vec path's K error); Q f16-rounding contribution rms **0.005** (max 0.021) vs the q8_1
  Q's ~4.3 -> ~900x smaller. Total KQ score error: fused 2.27 vs PathVEC 4.82 (~2.1x better) while
  being ~8.7x faster.

### 10d. Architecture notes for FA integration (not yet done)

- Q side: the vec kernel has ncols<=8 queries; f16 A fragments are m16 - pad half-wasted or batch
  2 heads/CTA; the fattn-mma-f16 kernel tiles queries properly (its K/Q f16 smem tiles + ldmatrix
  pattern is the natural host for the decode-staging change: replace the cp_async f16 K tile load
  with nvfp4 decode into the same tile, K stays nvfp4 in VRAM; V-side PV needs the same decode).
- Best fused config in the probe: 4 warps/CTA, 16 tokens/warp, Q staged in gmem as f16 (query
  matrix only - not KV), 3 CTAs/SM (33 KB smem). For FA-vec integration, Q smem is tiny (8 rows)
  and the same decode+mma feed applies.
- Decode ALU cost per coord: ~0.5 op/coord-class after PRMT; never the bottleneck at nq>=16 tiles;
  at nq=8 expect decode ~ DRAM rate.

### 10e. Production integration into the FA vec kernel (2026-11-01)

Integrated the fused path into `ggml/src/ggml-cuda/fattn-vec.cuh` behind a new env flag
`GGML_CUDA_FATTN_VEC_KQ_MMA_F16` (default off; takes precedence over the closed
`GGML_CUDA_FATTN_VEC_KQ_MMA`). Flag-off dispatch and the generic kernel are untouched -
the added code is `if constexpr`-gated on a new template param `KQ_f16dq` (default false),
so every pre-existing instance compiles to the same code path as before integration.

What the fused instance does (`flash_attn_ext_vec<D, ncols, K, V, softcap, false, true>`):
- Q f32 -> f16 half2 words staged once into dynamic smem (8 rows x FHSTR = D/2+1 words,
  inactive rows zero). Only Q is f16; **no f16 KV ever touches global memory**.
- Each warp decodes its 32 K rows straight from the `block_nvfp4` cache bytes into
  fragment-native half2 words in its own dynamic-smem slice (PRMT E2M1->f16 table pairs x
  UE4M3 scale half2; the byte(b)=coords(b,b+8) ggml packing gotcha of §10b applies).
- Dense `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`: A = 16 token rows x k16 (2 m-tiles
  per warp), B = 8 Q cols (ncols<=2 used, rest zero - wasted n-slots are harmless). Scores
  written to `F4_raw` with the kernel `scale`; the existing closed-branch consumer
  (softcap/alibi/mask/KQ_max) is shared unchanged.
- Dynamic smem: `(8 + nthreads)*(D/2+1)*4` B = 70 KB @ D=256, nthreads=128; set via
  `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` in the launcher before `launch_fattn`.
  (`__launch_bounds__(128,1)` already forces 1 CTA/SM; SM120 max opt-in dyn smem is 227 KB.)
- Build gate in `ggml_cuda_flash_attn_ext_vec_case`: `kq_f16_build = (type_K == GGML_TYPE_NVFP4)
  && (D == 128 || D == 256)` (whole 16-coord k-fragments); runtime gate `turing_mma_available(cc)`.

Verified: `ggml-cuda` compiles clean for sm_120a (only benign unused-static warnings in
non-nvfp4 instances). Greedy smoke (D=256 model, nvfp4 KV, `-fa on`, temp 0, seed 42, 64 tok):
flag on -> plausible fluent output, no crash; flag off -> plausible; outputs diverge at char
~210 (KQ method really switches). Note the first fused-build attempt failed only because
`fattn_kernel_ptr_t` is a local alias in `fattn-mma-f16.cuh`; the vec launcher now defines its
own HIP-guarded alias.

Gate harness `run_fa_mma_ab.ps1` `Set-Flags` now toggles `GGML_CUDA_FATTN_VEC_KQ_MMA_F16`
and force-clears the closed flag. Expectation vs the q8_1-KQ baseline (PPL 7.1289 @ ub=1/4096):
KQ score rms error drops 4.82 -> 2.27 (K-nvfp4 only), so PPL should move slightly FA VORTEIL;
|dPPL| <= 0.01 still holds as hard gate, decode tg +2.5-4x on the KQ part expected at d8192/d32768
(§10c: 1.35 -> 11.8 TF/s on the KQ GEMM alone). A/B run by user: `-Sections ABD`.

**v1 gate result (2026-11-01, smem-staging build): FAIL both ways.**
PPL fused 7.1417 vs q8_1 7.1289 (delta +0.0128, just over the 0.01 bar); decode tg 30.20 ->
26.82 t/s @d8192 (-11.2%) and 23.64 -> 17.05 @d32768 (-27.9%). The tg loss scaled with context
(+4.2 ms -> +16.3 ms per token), i.e. it was the attention loop per-K-byte cost, not a fixed
launch cost. Primary cause: 70 KB dynamic smem collapsed occupancy to 1 CTA/SM (the generic
vec path uses only a few KB and hosts several CTAs/SM) -> lost DRAM latency hiding; secondary:
per-iteration smem round-trip for the K f16 tiles that are per-warp private anyway.

**v2 = register-resident fused KQ (same day).** The m16n8k16 fragment map is lane-aligned:
lane (g,tg) owns token rows g/g+8 at coords {2tg,2tg+1,8+2tg,9+2tg} of each 16-coord nvfp4
subblock - and one subblock IS one k-fragment. So A fragments are decoded straight from the
two uint32 words of that subblock (`__byte_perm(W0,W1, 2tg|(2tg+1)<<4)` picks bytes 2tg/2tg+1;
byte b = coords (b,b+8) -> low nibbles = a0/a1 coords, high nibbles = a2/a3; PRMT E2M1->f16 +
__hmul2 with the row's UE4M3 scale half2), and the B (Q) fragments are CTA-constant: built once
into per-lane regs `Bfr[D/16][2]` (32 regs @D=256) by f32->f16 cvt of Q rows. Dynamic smem,
FHSTR, Q smem staging, cudaFuncSetAttribute and __syncwarp all REMOVED; only the 1 KB static
`F4_raw` score buffer remains. Decode ALU ~0.5 op/coord unchanged. Rebuild clean; smoke fluent.
Numerics math is bit-identical to v1 (same decode/scale/MMA order), so PPL must reproduce
7.1417; only speed changes. Check harness: `run_fa_f16_v2_check.ps1` (-Ppl for the PPL triple
incl. an f16-KV gold reference; the honest numerics judge is |dPPL| vs gold, since the q8_1
path's Q error partly cancels nvfp4-V error and can luck into a lower raw PPL).
If v2 is still slower at nq=1, the remaining suspect is issue-slot waste (m16n8k16 with 7/8
n-columns and ~half the m-rows idle at ncols<=2; KQ at nq=1 is DRAM/issue bound, not tensor
FLOP bound - probe's 11.8 TF/s was measured at nq=64).

**v2 gate result (2026-11-01): SPEED PASS.**
tg128: d8192 29.90 -> 30.41 (+1.7%) and 23.35 -> 24.42 (+4.6%) t/s, reproducible across two
runs. v1's -11.2%/-27.9% regression is fully explained by the 70 KB dynamic-smem occupancy
collapse; with zero dynamic smem the fused path is also slightly *faster* than q8_1-KQ, and
the gain grows with context (KQ share of decode time grows with depth, consistent with a
DRAM-roofline kernel that needs fewer ALU ops per K byte than the q8_1 requant path).
PPL (quick eval, seed 42): gold(f16 KV) 7.1378, q8_1-KQ 7.1289 (-0.0089 vs gold),
fused-v2 7.1607 (+0.0229 vs gold). So at the model level fused is +0.0318 vs the q8 baseline
(0.44%) despite the probe's fp64 score-level ground truth saying fused KQ error rms is 2.1x
SMALLER (2.27 vs 4.82). Two caveats: (a) a 4096-token window is near noise for ranking two
perturbation patterns at 0.03 PPL - use `run_fa_f16_v2_check.ps1 -Ppl -Full` (full wiki.test.raw)
for the decision-grade triple; (b) v1 fused measured 7.1417, v2 7.1607 - PPL is deterministic,
so some ulp-level difference exists between the smem and register staging paths even though the
intended math is identical; v2's register map is the one verified bit-for-bit against fp64 in
the probe (rms 3.2e-6), so v1 is the suspect. Open question whether score-rms ordering can
invert at PPL level (softmax/softcap nonuniformity + nvfp4-V error identical in both paths,
so it cannot explain the ordering either) - pending -Full triple before any default flip.
Status: fused flag stays default-off; opt-in wins ~2-5% decode tg today.

**v1-vs-v2 PPL shift resolved (tools/fa_fused_v12_diff.cu): NOT a math difference.**
Bitwise A/B of the two decode maps (v1 smem-shift extraction vs v2 byte_perm register
extraction), same random nvfp4 K + f32 Q, 16 tokens x D=256 x 8 queries:
0/4096 decode-array mismatches, 0/128 score mismatches, and both agree with an independent
scalar (fragment-free) reference at rms 0.0035 against |score| ~ 2e4 - pure f32 accumulation
noise. The v2 register map is confirmed correct.
So the 7.1417 -> 7.1607 PPL move comes from the LAUNCHER, not the kernel: v1 passed
70 KB nbytes_shared to launch_fattn, which feeds cudaOccupancyMaxActiveBlocksPerMultiprocessor
-> fewer blocks/SM -> different blocks_num (grid.y = KV parallel split) -> different tile
boundaries and partial-softmax combine ORDER -> ulp-level score differences summed differently
across a 4096-token chain. v2 (zero dynamic smem) shares the exact split of the q8_1 baseline,
which is why the q8 number reproduced (7.1289) in both builds while fused moved. Consequence:
the current fused-vs-q8 PPL comparison (7.1607 vs 7.1289, same split, same combine order) is
apples-to-apples - the +0.0318 is genuinely attributable to the KQ method at this eval size.
**Project criteria re-based 2026-11-01 evening.** Primary goal = NVFP4 KV + fused decode/MMA
(max memory saving); secondary = speed at acceptable PPL deviation FROM THE FP16 FALLBACK
(the only gold reference), not from the q8_1 vec path. Exact model facts (from run logs):
qwen35 27B, 64 layers, n_head=24, n_head_kv=4, D=256 -> KV bytes/token: f16 = 256 KiB,
nvfp4 = 72 KiB (ratio 3.56x; nvfp4 = 9 B per 16 coords: 8 qs + 1 scale). Model = 11.28 GiB,
so at d32768 KV is 8.0 GiB f16 vs 2.25 GiB nvfp4 - **f16 KV at 32K does not fit 16 GB VRAM
at all**; f16 gold is only measurable at c<=8192. nvfp4 is not an optimization, it is what
makes the target context exist. New gate for the fused flag (run_fa_f16_v2_check.ps1):
tg gain > 0 AND |PPL_fused - PPL_gold| <= GoldTol (param, default 0.05) at equal windows;
q8-vs-fused raw comparison demoted to informational (q8's extra requant noise = attention
smoothing that flatters its raw PPL). Also fixed: --chunks 1 caps the sample at 4096 tokens
regardless of file (earlier -Full intent did NOT extend the sample; use -Chunks N).
Quick-set reference (4096 tok): gold 7.1378, q8 -0.0089, fused +0.0229 (+0.32%) - inside a
0.05 tol but sample-SE(diff) ~0.01 at that size; decision-grade = -Chunks 4 (16K tokens,
~27 min) or -Full -Chunks 68 (278K tokens, overnight).

**Quick triple reproduced 20:13 (deterministic, same numbers).** Decomposition of the triple:
fused - gold = +0.0229 with near-exact KQ, so it is ~ the nvfp4-V penalty; q8 - gold = -0.0089,
so the q8 path's EXTRA KQ requant noise is worth about -0.032 NLL on this eval - i.e. extra
score noise acts as attention smoothing/regularization and happens to help PPL here. The fused
path is not "worse math" (per-score it is 2.1x closer to exact); it just doesn't get the free
regularization. Caveats: run-to-run noise is zero (deterministic) but the 4K-token quick set
has SE(diff) ~0.01 PPL, so +0.032 is probably but not certainly the average gap - the -Full
triple (70x data) is the decision-grade measurement. Working verdict while pending: fused is an
OPT-IN perf feature trading ~+0.4% PPL for ~+2-5% decode tg; default stays OFF.

**Competitor set re-based (user criteria, 2026-11-01): memory-comparable storage only.**
q8_1-KQ raw PPL and f16-KV are removed as competitors (different memory class); f16-KV stays
only as PPL gold anchor (and cannot run at 32K on 16 GB at all: 8 GiB KV + 11.28 GiB model).
Rows now: nvfp4-dot (production vec = integer q8_1-style dot over nvfp4 bytes - NOT a
decode->f16->MMA path, no f16 intermediate exists there), nvfp4-fused (target), q4_1-dot
(0.625 B/coord vs nvfp4 0.5625 B/coord; vec instance exists), turbo4_0-dot (0.5156 B/coord -
the vec template pair turbo4_0-turbo4_0 is NOT in this build's FA_QUANTS set, the instance
must be added before the row runs), fp16-gold anchor (2.0 B/coord). Harness:
run_fa_nvfp4_competitors.ps1. Requirement "fused must not add error vs unfused
decode->f16->MMA" is SATISFIED and proven by tools/fa_fused_v12_diff.cu: fused feeds the MMA
bitwise-identical f16 fragments to the unfused smem-staging kernel (0/4096 decode, 0/128 score
mismatches) and both equal exact scalar dequantize within f32 accumulation noise. The +0.023
vs gold is nvfp4 STORAGE codec error, identical for fused and unfused; the only fused/unfused
model-level delta (7.1417 vs 7.1607) is accumulator-order + grid-split confound noise.

**turbo4_0-turbo4_0 vec instance added + routing finding (2026-11-01 late).** To make the turbo4
PPL-rival row runnable, the missing vec instance was created (template-instances/fattn-vec-instance-
turbo4_0-turbo4_0.cu, 128/256 only - QK_TURBO4=128 never fits a 64-D row), extern-declared in
fattn-vec.cuh, dispatch cases added in fattn.cu, and the build cache GGML_CUDA_FA_QUANTS extended
with turbo4_0-turbo4_0 (regenerated defines in build.ninja: GGML_CUDA_FA_TURBO4_0_TURBO4_0=1).
ROUTING FINDING: this fork gates turbo4 K/V AWAY from the vec kernel by default (fattn.cu
GGML_TURBO_DITHER speed gate: vec scalar-dot for turbo4 lost 6-15% vs the MMA path at 8K,
measured 2026-10-31), so default routing converts K/V per FA node to f16 scratch + MMA_F16.
The scratch is per-node: 32K ctx -> 67 MB per tensor per layer x 2 x 64 layers ~ 8.5 GiB in the
CUDA-graph pool -> turbo4 via MMA OOMs at d32768 on 16 GB (fits at 8K: 33 MB/layer ~ 2.1 GiB).
So at 32K turbo4 only runs at all through the NEW native vec instance (NATIVE=1 bypasses the gate).
nvfp4 never had this problem: its vec kernel is native since Stage A, zero scratch.
Memory nuance for the rival comparison: turbo4_0 storage (0.5156 B/coord) is actually SMALLER
than nvfp4 (0.5625) - nvfp4 does not win on bytes vs turbo4; the win must come from accuracy
vs turbo4, speed, and the zero-scratch 32K path. Harness rows: turbo4-mma (default routing, bench
capped d<=8192) and turbo4-vec (NATIVE=1, same kernel class as nvfp4/q4_1, PPL-rival row).

**turbo4 CLOSED as a work item (2026-11-01, user decision):** all turbo4 speed/accuracy data already
exists from prior tests; harness rows turbo4-mma/turbo4-vec removed (the instance itself stays in the
build, harmless). Remaining comparison axis = nvfp4-fused vs q4_1 (SPEED ETALON) + gold PPL anchor.

### 10f. V-side PRMT register decode (2026-11-01, RESULT: etalon gap closed to ~0/-1.8%)

Diagnosis: after the K-side MMA fusion the nvfp4-vs-q4_1 decode gap lived on the **V/PV side** -
`dequantize_V_nvfp4` decoded per-coordinate with single-byte loads + divergent `kvalues_fp4`
constant-cache LUT reads + f32 multiplies, while q4_1 uses a 4-byte load + `__vsub4` SIMD. The
fix applies the same PRMT register decoder already proven on the fused K side:
* one `ne`-byte word load replaces `ne` byte loads (`ne`=4 -> u32, `ne`=2 -> u16; window never
  crosses the 32-byte qs[] end because coord starts are 4-aligned and o%8 ranges stay inside);
* high-nibble coords (o>=8) via one `w >>= 4` (garbage entering upper nibbles is never read:
  `ggml_cuda_e2m1_pair_f16` consumes bits 0..3 per byte operand, bit 3 = sign);
* decode = PRMT pair-lookup (tl=0x3E3C3800/th=0x46444240, sign OR) + one `__hmul2` with
  `__float2half_rn(2*ue4m3_to_fp32(d))` - E2M1 x E4M3 products have <=6 significant mantissa
  bits so every product is exact in f16: the f16-domain result equals the old f32 LUT product
  rounded to f16, bit-exact by construction;
* `ggml_cuda_e2m1_pair_f16` moved fattn-vec.cuh -> fattn-common.cuh (single definition, both K
  and V use it); scalar `kv_fp4_val` deleted; K-side `vec_dot_fattn_vec_KQ_nvfp4` (production
  dot path) untouched - still `kvalues_fp4` integer dot.

Bit-identity probe `tools/fa_vnvfp4_decode_diff.cu` (random blocks + exhaustive scale-byte x
nibble-pair sweep): half ne=4 exact 583671 / signed-zero 71689 / **0 mismatches**; float ne=4
491566/32722/**0**; half ne=2 230346/31798/**0**. The ONLY raw-bit difference: E2M1 code 8 (-0)
decodes to -0 via PRMT while the LUT table stores +0 at index 8. Value-equal and provably
harmless in the VKQ chain: products with +/-0 give +/-0, accumulator init is +0, and under RN
(+0)+(-0)=+0; downstream exp(-0)=exp(+0) - logits bit-identical. (First probe iteration showed
56k "mismatches" - that was a comparator bug: it required BOTH f16 lanes to be zero; fixed to
per-lane comparison.)

Measured (smoke, llama-bench -p 8 -n 128, 2 reps, same build; q4_1 row from fa_comp_20261101_204956):

| row            | tg128 @ d8192 | tg128 @ d32768 |
|----------------|--------------:|---------------:|
| nvfp4-dot      |    31.01      |     25.70      |
| nvfp4-fused    |    31.95      |     27.50      |
| q4_1 (etalon)  |    31.97      |     28.01      |

fused vs dot: +3.0% / +7.0%; fused vs q4_1 etalon: **-0.1% @8K (parity), -1.8% @32K** - was
-3.8% / -11.6% before this fix. The V decode is shared by dot and fused, so BOTH rows gained
(dot gained even more in absolute terms: 30.33->31.01, 23.66->25.70). No bit-identity risk:
KQ side untouched, V decode value-identical per probe.

Residual @32K candidates if pushed further: fused PV via mma, or `V_rows_per_thread=8` for
nvfp4 (8-byte coalesced V loads). Decision-grade numbers to be re-collected by
run_fa_nvfp4_competitors.ps1 (turbo4 rows now removed) + `-Ppl -Chunks 4` vs gold.

Full-harness confirmation (fa_comp_20261101_225351, user-run): fused 32.36 vs dot 31.69 (+2.1%)
@8192, 27.51 vs 25.79 (+6.7%) @32768; vs q4_1 etalon 32.63 (-0.8%, within 2%) and 28.76 (-4.3%).
PPL note: llama-perplexity at -Chunks 4 takes ~8 min/row (ub=1 -> all-decode, 16K tokens) and
spams "CUDA Graph reused" debug lines - it looked hung but completed: nvfp4-dot final-chunk PPL
= 5.5052 (chunks [7.1289, 6.9060, 5.4756, 5.5052]; quick 4K value reproduced exactly in chunk 1).

### 10g. HW cvt.e2m1x2 decoder (external-implementation review, 2026-11-01)

Reviewed github.com/hikarioyama/vllm-nvfp4-kv-sm120 (NVFP4 KV for vLLM on SM120 via FlashInfer FA2):
same fundamental architecture as our fused vec path - SM120 has no FP4 tensor-core cubins/paths for
this workload either; they dequantize e2m1->bf16 IN REGISTERS using the HW instruction
`cvt.rn.f16x2.e2m1x2` and compute with standard Ampere mma.sync; measured nvfp4 decode = 91-100% of
fp8 decode (single-stream 97.6%), nvfp4 KV quality +0.01-0.04 nats PPL. Their unique contribution is
storage wiring (explicit SF strides + in-kernel V-SF de-swizzle, zero SF scratch), which our ggml
interleaved block_nvfp4 layout already provides structurally. Conclusion: our fused approach is the
right one for SM120; the portable idea is the decoder instruction.

Action taken: PTX check (nvcc -ptx, sm_120a) confirms __nv_cvt_fp4x2_to_halfraw2 emits
`cvt.rn.f16x2.e2m1x2` verbatim (1 instr). Probe tools/fa_nvfp4_cvt_probe.cu: HW cvt is
BITWISE-IDENTICAL to the PRMT decoder for all 256 pair encodings (incl. -0 for code 8), and equals
the old LUT up to the same +0/-0 caveat (225 exact / 31 sign-zero / 0 real). Production swap:
new helper ggml_cuda_e2m1x2_f16 (fattn-common.cuh, PRMT fallback under non-CUDA/CUDA<12.8) now
drives both (a) dequantize_V_nvfp4 (pack (s&0xF)|((s>>4)&0xF0) + 1 cvt per half2, was sel+PRMT+2
sign-ORs) and (b) the fused nq=1 K-fragment decode (q0..q3 pack + 4 cvt per f/mt, was 4x e2m1_pair_f16).
Bit-exact by the probe => no numeric risk, no PPL movement expected; ~4 fewer SASS ops per pair on
both hot sides. ggml-cuda.dll rebuilt clean. Awaiting user bench re-run vs q4_1 (gap before this:
-0.8% @8K, -4.3% @32K). CORRECTION (see 10i): the production swap never actually took effect - the
macro guard used GGML_USE_CUDA, which is not defined inside this fork's CUDA-file compiles.

### 10h. Production-kernel A/B probe + bit-identical dot/scale fixes (2026-11-02)

New instrument: tools/fa_vec_ab_probe.cu times the PRODUCTION flash_attn_ext_vec kernel directly
(synthetic benign KV caches, launch config replicated exactly from launch_fattn: occupancy query +
parallel_blocks loop) for the nq=1 / D=256 / nkv=32768 decode shape (24 q heads, 4 kv heads, 36 SMs).
First run (before the fixes below, ms medians, 11 reps):

| variant | ms | vs etalon |
|---|---:|---:|
| V1 q4_1/q4_1 dot (SPEED ETALON)        | 0.381 | -      |
| V2 nvfp4/nvfp4 dot                     | 0.612 | +61%   |
| V3 nvfp4/nvfp4 f16dq (fused)           | 0.449 | +17.8% |
| V4 nvfp4 K(fused) + q4_1 V             | 0.439 | +15.2% |
| V5 q4_1 K + nvfp4 V (dot)              | 0.371 | -2.6%  |

Attribution: the nvfp4 V-side is already FASTER than q4_1 V (V5 < V1), all remaining fused gap is the
KQ side (fused mma KQ vs q4_1 integer dot) plus small scheduling interaction. occ=2, pb=3 identical
for every variant, so occupancy/waves are not the issue (regs 198-255). Probe ms runs ~2.4x hotter
than the production kernel share - use for RELATIVE attribution only.

Fix 1 (bit-identical), fused KQ scales: new helper ggml_cuda_ue4m3_dword_h2 loads the 4 scale bytes
of a block as ONE u32 and converts with the HW e4m3->half2 cvt. e4m3 -> f16 is exact (no rounding),
and 2*ue4m3_to_fp32(b) == the raw e4m3 value, so this is bitwise-identical to the old
__float2half_rn(2*ue4m3_to_fp32(...)) scalar path; the NaN->0 flush of ue4m3_to_fp32 is preserved
explicitly. Applied in the fused KQ block and in dequantize_V_nvfp4.

Fix 2 (bit-identical), production nvfp4 KQ dot: vec_dot_fattn_vec_KQ_nvfp4 rewritten from the
divergent kvalues_fp4 constant-LUT gather (the original reason for the nvfp4 dot penalty) to a SIMD
integer decode: one PRMT (0x03020100/0x0C080604 = magnitudes 0,1,2,3|4,6,8,12) + __vcmpgts4 sign mask
+ __vsub4 negate + __dp4a dot + __vadd4 tree for sumc. Same exact integers => same float tail =>
bitwise-identical dot (kvalues_fp4[8] = +0 maps to the -0 code, value-equal). Host proof:
tools/fa_nvfp4_dot_intdiff.cu PASS (all 16 nibbles x lane positions + 2M random words: values,
dp4a doti, sumc all equal). Note: __vcmpgt4 does not exist, __vcmpgts4 does.

After both fixes (probe rerun): V2 0.612 -> 0.517 (-16%), V3 0.449 -> 0.425 (-5%),
V4 0.402, V5 0.361, V1 unchanged 0.380. Residual fused gap vs etalon is now ~+0.022 ms of KQ-side
mma/decode + scheduling. ggml-cuda.dll rebuilt clean (build_dotfix.log, [61/61] Linking).

PPL clarification: the earlier "nvfp4-dot Chunks-4 hang" was NOT a hang - the row completed,
final-chunk PPL = 5.5052 (chunk sequence [7.1289, 6.9060, 5.4756, 5.5052]), ~8 min/row, log
fa_comp_20261101_225351\ppl_nvfp4-dot.log. No numeric movement from any fix in this section (all
bit-identical changes): quick PPL spot checks stayed dot 7.1289 / fused 7.1607.

### 10i. SASS forensics: the HW-cvt path was dead code in production (2026-11-02)

Harness after 10h (fa_comp_20261102_001828): fused 32.48 @8K (-0.9% vs etalon, PASS), 27.92 @32K
(-2.5%, gate miss); nvfp4-dot 32K 25.79 -> 26.82 (+4%, the PRMT+dp4a dot rewrite confirmed).
To find the remaining fused gap we disassembled the ACTUAL production DLL kernels
(cuobjdump -sass -fun <mangled flash_attn_ext_vec...>) instead of theorizing:

1. No register spills anywhere: LDL/STL = 0 in both fused and dot production kernels. The
   254/255-register pressure costs MOV/SEL churn (508 vs 110 static), not local memory.
2. Load width is not the limiter (reference q4_1 kernel also has zero LDG.64/128).
3. Production fused carried 5576 static instrs vs 4096 for the fast dot variant, and the whole
   excess sat in the e2m1 decode: IMAD.SHL/LOP3 cascades where the SASS should have had the
   native E2M1 convert opcode. E2M1 opcode count in the production fused kernel: ZERO.

Root cause: the GGML_CUDA_E2M1_HW_CVT macro of 10g was guarded on defined(GGML_USE_CUDA), but
this fork never defines GGML_USE_CUDA for the CUDA-file compiles of the ggml-cuda target (it only
exists host-side in ggml-backend-reg.cpp). The 10g "production swap" was a NO-OP: both old
harness runs and all fa_vec_ab_probe builds ran the PRMT decoder. ptxas on sm_120a DOES emit the
native instruction when the intrinsic actually reaches PTX (256x E2M1 in fused / 128x in dot for
the corrected probe build, PTX e2m1x2 x640); the earlier "cvt is lower quality" reading was an
artifact of comparing a PRMT-only SASS against a PRMT C-source sequence.

Fix: guard switched to defined(__CUDACC__) (nvcc is what compiles these TUs; GGML_USE_HIP and
GGML_CUDA_FORCE_E2M1_PRMT remain as escape hatches). Rebuilt DLL verified: fused f16dq D=256 now
4096 instrs (was 5576), E2M1 x256, LDL/STL 0; nvfp4 dot kernel E2M1 x128 + F2FP.E4M3 x128.
Decoder paths are bitwise-identical (fa_nvfp4_cvt_probe, fa_vnvfp4_decode_diff), so no PPL change.

Probe A/B of the two decoder codepaths (probe now compiled per production; med ms, 11 reps):

| variant | PRMT decoder (old production) | HW cvt (new production) |
|---|---:|---:|
| V3 fused nvfp4/nvfp4 | 0.426 | 0.363 (-15%) |
| V2 nvfp4 dot         | 0.516 | 0.495 (-4%)  |
| V5 q4_1K + nvfp4V dot| 0.363 | 0.329        |
| V1 q4_1 etalon       | 0.381 | 0.381        |

Expected harness effect: fused 32K roughly -10..-15% kernel time => the -2.5% gate miss should
flip to comfortably inside (fused KQ now cheaper than the q4_1 dp4a dot at equal bytes); nvfp4-dot
a few % more. Probe caveat unchanged: use probe numbers for RELATIVE attribution only.

RESULT (fa_comp_20261102_010329, same session, same etalon run): fused 32.83 t/s @8K = **+0.2% vs
q4_1** (32.77), 28.73 t/s @32K = **+0.1% vs q4_1** (28.69). BOTH context gates PASS with fused
slightly AHEAD of the speed etalon; fused vs dot +1.8% @8K / +5.6% @32K; nvfp4-dot 27.21 @32K
(25.79 -> 26.82 -> 27.21 across the three evening builds). Journey: -4.3% -> -2.5% -> +0.1%.
The remaining KQ-side fused deficit is gone - dense f16 MMA with the native cvt.rn.f16x2.e2m1x2
decoder beats the integer dp4a dot at equal 144 B/token. Speed objective CLOSED; what remains is
the PPL parity table (fused must stay within 0.05 of the unfused nvfp4 PPL - expected bit-stable,
decoder swaps were probe-proven bitwise) and an optional decision to default the fused flag on.

Follow-up (same day): GGML_CUDA_FATTN_VEC_KQ_MMA_F16 switched to DEFAULT ON (opt-out: set it to
"0"); the harness therefore sets 0/1 explicitly per row (PowerShell's "" deletes the variable and
would otherwise silently pick up the new default). Repository placed under git (initial commit
d13246b: fork baseline + all nvfp4 fused work; build/model/log dirs gitignored).

Follow-up 2 (same day, build integration): the fused KQ default became a real build switch -
cmake option GGML_CUDA_FATTN_VEC_KQ_MMA_F16 (ggml/CMakeLists.txt, default ON) ->
GGML_CUDA_FATTN_VEC_KQ_MMA_F16_DEFAULT compile definition (ggml-cuda/CMakeLists.txt), read by
ggml_cuda_fattn_vec_kq_mma_f16_enabled(); the env var stays as a pure runtime override for A/B.
FA_QUANTS list carries the two production nvfp4 combos: nvfp4-nvfp4 (memory, 0.5625 B/c) and
q5_1-nvfp4 (quality: integer q5_1 K dot + nvfp4 V), plus f16-nvfp4 for the bf16/f16-K path.

Follow-up 3 (same day, FREEZE): nvfp4 CUDA kernels are frozen - no further code changes.
Build verified complete: FA vec instances (nvfp4-nvfp4 x6 incl fused KQ_f16dq variant,
q5_1-nvfp4, f16-nvfp4) + mmq-instance-nvfp4 compiled; FORCE_MMQ/FORCE_CUBLAS OFF (MMQ used
where cuBLAS does not apply). First measurement of the quality combo q5_1-nvfp4 (user run,
quick set, Chunks-1): PPL 7.1439, decode 32.84 t/s @4K - between nvfp4-fused 7.1607 and
q4_1 7.1306 (gold 7.1378): q5_1-K recovers most of the score error while V stays at 0.5625 B/c.

### 10j. The 10h PRMT+dp4a dot rewrite was numerically WRONG (2026-11-02)

First PPL after the 10h dot rewrite (fa_comp_20261102_015717, Chunks-4, quick set):
nvfp4-DOT PPL = 9.3682 (chunk sequence [10.7803, 10.4235, 9.0482, 9.3682]) vs 5.5052 for the old
LUT dot (fa_comp_20261101_225351) at the same settings. Fused +0.0209, q4_1 +0.0108, gold 5.5128,
fused gate PASS - the anomaly was dot-only, and dot's KQ side is the only code fused does not use.

The 10h int8 SIMD decoder had a nibble-lane mispairing. A PRMT instruction consumes only SELECTOR
NIBBLES 0..3 (bits 0..15 of the selector). The rewrite set `sel = bs & 0x07070707` on the shifted
4-byte window bs, i.e. nibbles 0..3 of bs = [coord c, coord c+8, coord c+1, coord c+9] (the high
nibble of each nvfp4 storage byte is coord +8, not the next word coord). u4/v4 byte lanes therefore
held K coords [c, c+8, c+1, c+9] while the q8_1 word lanes (per-thread interleaved staging,
Q_i32[j][k_KQ_0/nthreads] = Qq word k_KQ) hold Q coords [c, c+1, c+2, c+3]: dp4a multiplied 3 of
4 lanes against the WRONG K coordinate, and sumc summed the wrong coordinate set. The old scalar
LUT body extracted one byte at a time (`byte = (b >> 8t) & 0xFF; lut[byte >> sh & 0xF]`), which is
immune to this - hence the regression appeared exactly with the rewrite.
(The earlier §10g suspicion of the dequantize_V_nvfp4 pack line `(s & 0x0F) | ((s >> 4) & 0xF0)`
was a FALSE lead: (s >> 4) & 0xF0 selects bits 8-11 of s = the low nibble of the NEXT storage byte
= coord c+1. V packing is consecutive-coord and correct - consistent with fused PPL being healthy.)

Host intdiff (fa_nvfp4_dot_intdiff, §10h) missed this because it emulated the decoder sequence
including the same wrong sel indexing instead of the coordinate semantics.

Fix: pack the 4 word coords (the EVEN nibbles of bs, i.e. low nibble of each byte) into selector
nibbles 0..3 before the PRMT: `pk = (bs & 0x000F000F) | ((bs >> 4) & 0x00F000F0); sel = (pk | pk
>> 8) & 0x00007777;` (+~6 int ops/word, still far cheaper than the old LUT gather). Sign mask sn4
and the vadd4 sumc tree are unchanged and now lane-align with u4.

Verification tool: tools/fa_nvfp4_devdiff.cu device probe (built twice: fa_nvfp4_devdiff_buggy.exe
against the pre-fix header, fa_nvfp4_devdiff.exe against the fixed one; user-run):
(A) old-LUT dot vs current dot on random KV rows, bitwise float compare; (B) dequantize_V_nvfp4
vs consecutive-coord and (c,c+8) host references + the historical 10f PRMT pair path;
(C) old-LUT dot vs correct host KQ reference. Expected: buggy build FAILS (A)/(C), fixed build
passes all; (B) should PASS against the consecutive reference in both builds.

PPL verification after the fix (fa_comp_pplq_20261102, quick set, Chunks-1, user run):
nvfp4-dot = 7.1289 - IDENTICAL to the pre-10h LUT-dot value (deterministic PPL => bitwise
regression closed); nvfp4 default-env == explicit-1 == 7.1607 (default-ON confirmed); q4_1 7.1306;
fp16-gold 7.1378; fused +0.0229 vs gold (quick-set spot check). Full Chunks-4 sequence
[7.1289, 6.9060, 5.4756, 5.5052]-class expected.

Probe note: the first devdiff builds exited silently after the header line - the probe itself
passed HOST pointers as kernel output buffers (illegal access). Fixed (device out buffers +
unbuffered stdout); fa_nvfp4_devdiff_buggy.exe / fa_nvfp4_devdiff.exe rebuilt from the pre-10j
(fed3fae) and fixed headers respectively.

## §11 turbo4/turbo3 FA kernels: LUT-gather bottleneck + S1 register-mux decode (2026-11-02)

Scope note: nvfp4 kernels are FROZEN (§10i follow-up 3); this section is the turbo KV path, the
slowest + most economical user configs (turbo4-turbo3 / q5_1-turbo3 at -fa on).

FA status first: FA is NOT the problem. All 13 turbo combos are compiled into vec-kernel
instances (cuobjdump symbol check), and the runtime support gate accepts them. turbo3_0/turbo3q_0
exist only as V types by design (no KQ integer dot implemented) - K=turbo3 fails FA support on
purpose, q5_1-turbo3_0 / turbo4_0-turbo3_0 are the legal economical combos.

Bottleneck (cuobjdump SASS forensics, flash_attn_ext_vec ILi256 D=256 vec kernels, pre-S1):

| kernel | instr | LDC (const gather) | IDP (dp4a) | notes |
|---|---:|---:|---:|---|
| turbo4_0-turbo3_0 | 4096 | **537** | 0 | one __constant__ float read PER COORDINATE; divergent lanes replay each LDC up to 16x |
| q5_1-turbo3_0 | - | high | 64 | K side integer dot OK, V side LUT |
| nvfp4-dot (ref) | 4096 | 19 | 64 | PRMT decode (S1 idea applied to nvfp4 in §10f-h) |
| q4_1-q4_1 (etalon) | 3872 | 20 | 64 | integer codebook |

dp4a is impossible for turbo (Lloyd-Max codebook is nonlinear), so S1 = same trick as §10f-h:
replace the gather with an SEL/FSEL mux tree over f32 immediates. All turbo codebooks are
sign-symmetric; structure discovered during implementation (and now asserted at compile time):
codebook entries ASCEND with idx, therefore |value| DESCENDS over the negative half:
- 4-bit: t = (idx&7) ^ (idx&8 ? 0 : 7), ascending 8-magnitude tree, sign = XOR of bit3 into the
  f32 sign bit (mag nonzero -> no -0 issues);
- 3-bit: t = (idx&3) ^ (idx&4 ? 0 : 3), 4 magnitudes;
- 2-bit:  t = (idx&1) ^ (idx&2 ? 0 : 1);
- 5-bit: m = (idx&16) ? 16-(idx&15) : (idx&15), 16 descending magnitudes, explicit +0 at idx 16.
IEEE negation is an exact sign-bit flip, the tree leaves are the exact table literals, so the
mux returns BITWISE-identical floats - PPL cannot move (and the dither/QJL logic is untouched).

Correctness chain (all executed; GPU gate PASSED):
1. static_asserts in turbo-quant.cuh check every index of all 4 tables at compile time (this
   caught two real bugs pre-build: sign-parity inversion in the assert loops, and the 4-bit
   reflect polarity);
2. host exhaustive check (temp turbo_mux_check.cu): 4+8+16+32 = 60/60 entries bitwise identical;
3. literal diff: __constant__ tables vs the constexpr REF mirrors used by the asserts - all 4
   arrays identical (PowerShell Compare-Object);
4. GPU dump A/B (fa_turbo_ab_probe, first run): caught a THIRD real bug class - the static
   asserts and the host mirror verified magnitude+sign-parity, but the __device__ wrapper XORed
   the sign for idx sign bit SET, while in these ascending tables the NEGATIVE half has the
   sign bit CLEAR. 2/3/4-bit wrappers inverted (+0 entry kept turbo5 correct); probe symptom:
   turbo4/turbo4q K meta and turbo3 V dst dumps differ wildly, turbo5/q5_1 K meta identical -
   exactly the predicted split. Fixed by routing every wrapper AND the host mirror through one
   constexpr turbo_sign_mask_ helper (single source of truth for sign polarity). Rerun after the
   fix: **GATE PASSED** - all 5 variant dump hashes identical OLD vs NEW, and the in-app fnv
   checksums of the NEW build equal the OLD build's to the hex digit on every variant
   (T1 16967c5c..., T2 18050523..., T3 fc4168b0..., T4 be2c62d9..., T5 34838b75...).
   PPL/outputs therefore cannot move.

GPU A/B timing result (first run, RTX 5060 Ti, nq=1/D=256/nkv=32768, med):

| variant | OLD LUT | NEW mux | speedup |
|---|---:|---:|---:|
| T1 turbo4-turbo3 | 2.891 ms (346 t/s-eq) | 1.418 ms (705) | **2.04x** |
| T2 q5_1-turbo3 | 1.140 | 0.679 | 1.68x |
| T3 turbo4-turbo4 | 3.642 | 1.447 | 2.52x |
| T4 turbo5-turbo3 | 3.778 | 1.693 | 2.23x |
| T5 turbo4q-turbo4 | 2.888 | 1.496 | 1.93x |

(Sign-inverted decode is timing-representative: same instruction mix, codebook is sign-symmetric,
so the OLD/NEW timings above carry over; the post-fix rerun reproduced them within noise and the
bitwise dump gate passed 5/5.)

Remaining gap to the q4_1 integer-dot etalon (~2.3k t/s-equiv at this shape) is instruction-mix
bound (FFMA float dot + dither hash); the next lever would be S2 = porting the fused decode +
f16 m16n8k16 MMA KQ recipe (§10e-10i) to turbo4/5/4q. Not scheduled; S1 closed the worst 2.5x.

SASS after S1 (same kernels): turbo4_0-turbo3_0 LDC 537 -> **25**, FSEL=1823 SEL=517 (nvcc folds
both magnitude immediates into FSEL operands), IMAD 1738, LOP3 1822, LDL/STL=0 (no spills),
static instr count unchanged. q5_1-turbo3 LDC=25. turbo5 KQ is the heaviest mux (16 magnitudes).

Touched sites (hot paths only): turbo-quant.cuh element dequants (turbo2/3/4/5 + _u dither
variants + 3q/4q) and the 3 direct LUT reads in fattn-common.cuh vec_dot KQ turbo4/turbo5/turbo4q
bodies; dequantize_V_* and dequantize_turbo*_0 entries pick the helpers up automatically (so the
MMA_F16 prefill path benefits too). set-rows.cu / TURBO_MID_* quantizer tables intentionally
untouched (write path, not hot).

## §12 S2: fused decode + f16 m16n8k16 MMA KQ for turbo K (2026-11-02, status: implemented, gates PENDING)

Port of the §10e-10i fused recipe to the turbo K types (TURBO4_0/TURBO5_0/TURBO4Q_0), inside the
same KQ_f16dq block of flash_attn_ext_vec. Numerics deliberately change (like nvfp4 fused): K is
f16-rounded once at fragment build and Q is exact f32->f16 (replacing q8_1 Q quantization), so the
gate is PPL, not bitwise dump.

Fragment map (validated against §10j's mispairing lesson):
- m16n8k16, m = 16 K-token rows, n = up to 8 Q columns, k = 16 coords. Lane (g, tg) owns rows
  g, g+8 at coords {2tg,2tg+1} (e0 frag) and {8+2tg,9+2tg} (e1 frag) of 16-coord chunk cc = f & 7
  of turbo block kb = f >> 3 (QK_TURBO = 128 = 8 chunks; DIFFERENT from nvfp4's f >> 2 = 32-coord
  subblocks - the single most dangerous constant in this port).
- turbo4/5 nibble packing = CONSECUTIVE coords: byte b holds (2b, 2b+1) as (low, high) nibbles,
  so the lane's e0/e1 fragments are just qs bytes 8cc+tg and 8cc+4+tg (nvfp4's (b, b+8) byte_perm
  map does NOT apply). turbo5 hi bit: signs byte 2cc(+1), bits 2tg(+1). turbo4q: 2-bit lo at qs
  byte 4cc+(tg>>1) bits 4*(tg&1), hi1/signs like turbo5; value = (C3[idx] +/- ALPHA*rnorm)*norm.
- Dither (turbo4 only; turbo5/4q have no u term - mirrors vec_dot bodies): group w = j/4 exactly
  as the dot path: w = 4cc + (tg>>1) (e0), +2 (e1); lane = 2*(tg&1). Seeds = linear block index
  (k_tile_ibase + row*k_blk_per_row + kb), same contract as set_rows/convert. 4x fewer hashes
  than the element path and ~16x fewer than the S1 vec_dot path per lane-tile.
- Values are computed in f32 mirroring turbo{4,5,4q}_dequant_element*() op-for-op, then rounded
  ONCE to f16 (__floats2half2_rn); the whole builder ggml_cuda_fattn_turbo_kq_afrag<type_K>() is
  the single source of truth used by the kernel - the probe compares it end-to-end against the
  dot path (fa_turbo_ab_probe rows T6-T9 @D=256, C2/C4 @D=128 max-rel report, expectation ~1e-3
  not ~1e+0; a mapping bug would blow up to order-1 scores).
- scale stays applied on the f32 score writeback (like nvfp4), mask/softcap downstream unchanged.

Dispatch: kq_f16_build compile gate extended to the 3 turbo K types (D=128/256 only); runtime
selector split: nvfp4 -> GGML_CUDA_FATTN_VEC_KQ_MMA_F16 (default ON, frozen phase), turbo ->
GGML_CUDA_FATTN_VEC_TURBO_KQ_MMA_F16 with cmake default OFF (GGML_CUDA_FATTN_TURBO_KQ_MMA_F16)
until the gate below passes; 0 always falls back to the S1 q8_1-dot vec kernel.

Gate plan (user runs, GPU idle per policy):
1. fa_turbo_ab_probe (NEW exe): T1-T5 bitwise dump rows must stay identical to dump_old (dot path
   untouched); T6-T9 med timings vs T1/T3/T4/T5 = S2 speed; C1..C4 D=128 rows print fused-vs-dot
   max-rel (expect ~1e-3, NOT order 1 - that is the mapping-bug alarm).
2. ppl_quick_turbo.cmd: T5 vs T2 (turbo4-t3 NATIVE vec, fused on/off) and T7 vs T6 (turbo4q) -
   |dPPL| <= 0.05 like the nvfp4 fused gate (nvfp4 moved +0.0318 = 7.1289->7.1607 precedent);
   then llama-bench tg/pp A/B at the production routing (see §11 gate note: turbo4 combos are
   currently routed to f16-scratch, routing re-measure R1-R6 runs first).
3. If both pass: flip GGML_CUDA_FATTN_TURBO_KQ_MMA_F16 default ON + record numbers here.

### §12 gate round 1 (2026-11-02): the "mapping bug" was a HARNESS ARTIFACT - fused numerics look fine

- First fused run printed max-rel 7e1..4e3 + "every valid output differs" -> mapped as the §10j mispairing
  alarm. It was NOT: (a) cmp_path compared the RAW gridDim.y partial outputs - the vec kernel writes
  UNNORMALIZED per-tile partials (`dst[((head*gridDim.y)+blockIdx.y)*D+i]`, finish-kernel normalizes later),
  so partial sums with cancellation produce garbage max-rel for ANY numerics change; (b) three of six pairs
  even used the wrong K/V reference (T7/T8/T9 vs snapT1).
- tools/fa_turbo_frag_probe.cu (new, unit-tests kq_afrag against the element reference coord-by-coord,
  value->coord locator on mismatch): 0/192 mismatches for turbo4 (2 norm sets), turbo5, turbo4q - the
  fragment builder is bitwise-correct.
- Fixed cmp harness: host-side merge of the pb partials (mirrors flash_attn_ext_finish) + correct pairs
  (T6vsT1, T7vsT3, T8vsT4, T9vsT5, C2vsC1, C4vsC3). Result: absolute output deltas are ~1e-6..8e-6, i.e.
  exactly the predicted f16-rounding noise scale ~1e-3*sigmaV/sqrt(N) - fused numerics CONFIRMED sane at
  probe level. max-rel is meaningless for near-zero outputs; the production gate is PPL (step 2 above).
- Speed (idle-GPU numbers, both runs agree): D256 t4 fused 1.91ms vs dot 1.41 (SLOWER, dither), t5 1.50 vs
  1.70, t4q 1.29 vs 1.49; D128 t4 0.84 vs 0.50 (SLOWER), t4q 0.47 vs 0.53. turbo4 fused pays the 4-lane
  dither redundancy (each lane recomputes a 4-word its pair-lane already hashed); t5/t4q (no dither) win
  ~1.13x. Option S2b-lite: shuffle the dither words between pair lanes (tg0<->tg1, tg2<->tg3) to halve
  dither cost, or fuse only t5/t4q K. PENDING: PPL gate + routing decision.
- Note: fa_turbo_ab_new.exe T2(q5_1) fnv differs from the 14:38 build's run (14650fb0 vs 18050523) but is
  stable across runs of the current exe; T2 has no snapshot and its bitwise gate was via DUMP comparison
  (identical=True in the dump_old/dump_new run), so this does not affect any gate. Unexplained, low prio.

### §12 speed gate DECISION (2026-11-02): per-type selector - fused for turbo5 K only (turbo4q retired)

Idle-GPU timings reproduced across 3 runs (fa_turbo_ab_new med ms, D=256 | D=128):
| K   | dot    | fused  | result |
|-----|--------|--------|--------|
| t4  | 1.413  | 1.921  | +36% slower (D128: 0.504 -> 0.863, +71%) -> fused REJECTED for turbo4 K |
| t5  | 1.697  | 1.501  | -12% |
| t4q | 1.487  | 1.293  | -13% (D128: 0.534 -> 0.469, -12%) - rows since removed, see retirement note below |
Break-even analysis: t4q fused (dither-free) = 1.29 is the floor for a hypothetical ZERO-dither t4 fused;
the dither (~0.5ms at D256) cannot be made free (v2 generator is already 1 avalanche + 2 wide IMADs per
4-coord word; per-lane half-hash or pair-lane shuffles save at most ~half), and the dp4a dot +
turbo_dither_range path stays cheaper for t4.  **SUPERSEDED 2026-11-03:** that ~0.5 ms was NOT hash ALU -
it is a 64 B/thread spill of the dither arrays; see the spill-forensics section at the end of §12.
S2b-lite NOT implemented; decision implemented infattn-vec.cuh instead: kq_f16_sel excludes TURBO4_0 unless env GGML_CUDA_FATTN_VEC_TURBO_KQ_MMA_F16_FORCE_T4=1
(experiment override; probe T6/C2 rows instantiate the template directly and still measure the t4 fused
path). ppl_quick_turbo.cmd updated: T7-vs-T6 (t5) is the real numerics gate, T5-vs-T2 must be exactly
equal (selector-exclusion proof). PENDING: PPL run + routing R1-R4 (user, idle GPU).

### §12 turbo4q_0 retirement from the FA path (2026-11-02, user decision)

turbo4q_0 (Q_prod: MSE + 1-bit QJL residual sign) has a worse PPL class than the turbo4/turbo3 combos -
the branch was buried, no point keeping an FA kernel for it. Removed from ALL FA-path code: fattn-vec.cuh
(afrag 4q decode branch + static_asserts, kq_f16_build type list, extern DECL turbo4q combos), fattn.cu
(FATTN_VEC_CASE 4q combos, ggml_cuda_fattn_kv_type_supported), probes (ab rows T5/T9/C3/C4, frag probe
4q unit row), routing script (R5/R6). The QUANT CODEC
itself (ggml-common.h block_turbo4q_0, turbo-quant.cuh, convert.cu, ggml type enum) stays for GGUF format
compatibility - -ctk/-ctv turbo4q on new builds just falls back to the f16-scratch FA path (correct, slow).
Side benefit: ~10 fewer fattn template instantiations in ggml-cuda.dll.
Build-system note: the fork compiles each FA combo from a tracked template-instances file, so the three
fattn-vec-instance-(*turbo4q*) .cu files were git-rm'd too, and GGML_CUDA_FA_QUANTS in the build-dir
CMakeCache.txt was updated (turbo4q combos removed) before reconfigure - cmake FATAL_ERRORs on a listed
combo whose instance file is missing, and just editing the header would otherwise leave stale cached
combos in ninja's list.

### §12 naming trap (2026-11-03): CLI type names have NO `_0` suffix

Two different spellings coexist and MUST NOT be mixed up:
- cmake `GGML_CUDA_FA_QUANTS` combo strings use the enum form: `turbo4_0-turbo3_0`, `turbo5_0-turbo3_0`
  (reconfigure.cmd / configure.cmd / build_turboq.ps1).
- the CLI `-ctk/-ctv` values come from the ggml type_name table: `turbo3`, `turbo4`, `turbo5`,
  `turbo3q`, `turbo4q` (ggml.c:855-879; mapping in llama-bench.cpp:525-542). `-ctk turbo4_0` is
  REJECTED: "Unsupported cache type" and the row never runs.
First real S2 PPL run (2026-11-03) failed exactly this way - T1/T2/T3/T5/T6/T7 all bailed at arg
parsing and only the f16 row (T4) produced PPL. An earlier note in this section claiming the Oct-31 exes
"cannot parse turbo4q_0 but DO parse turbo5_0/turbo3_0" was wrong (it conflated the cmake combo names with
CLI names). ppl_quick_turbo.cmd and run_fa_turbo_routing.ps1 fixed to the suffix-less spellings; the older
scripts (run_kv_dump.ps1, run_check_dither_turbo3.ps1, run_check_nvfp4_stageA.ps1) always used them, which
is why they worked.

### §12 PPL gate round 2 (2026-11-03): PASS

Idle GPU, `ppl_quick_turbo.cmd` (-c 4096 --chunks 1 -b 1 -ub 1 -ngl 99 -fa on -s 42), all 7 rows parsed
after the naming fix (the same binary/DLL as the probes: Oct-31 llama.dll + ggml-cuda.dll @ 00:13, i.e.
turbo4q removed, selector per type):

| row | K-V | routing | PPL | note |
|-----|-----|---------|-----|------|
| T1 | turbo4-turbo3 | default (f16-scratch) | 7.1719 | |
| T2 | turbo4-turbo3 | NATIVE dot (S1 vec) | 7.1802 | |
| T3 | q5_1-turbo3 | NATIVE dot (S1 vec) | 7.1743 | |
| T4 | f16-f16 | gold | 7.1378 | == documented gold ref exactly |
| T5 | turbo4-turbo3 | NATIVE + FUSED KQ | 7.1802 | EXACTLY == T2 (+-0.37961 both) |
| T6 | turbo5-turbo3 | NATIVE dot (S1 vec) | 7.1920 | |
| T7 | turbo5-turbo3 | NATIVE + FUSED KQ | 7.1868 | |

- Gate 1 (selector-exclusion proof): T5 == T2 to the last printed digit -> the fused KQ path is NOT taken
  for TURBO4_0 K; the per-type selector works in the real binary.
- Gate 2 (numerics): dPPL(T7-T6) = -0.0052, |dPPL| <= 0.05 -> PASS (nvfp4 fused precedent was +0.0318,
  i.e. 6x larger). The fused f16-MMA KQ for turbo5 K is numerically equivalent at gate resolution
  (stderr +-0.38 makes 0.005 invisible).
- Sanity: T4 gold 7.1378 reproduces the old ref bit-for-bit, so the whole protocol (chunks/sampler/seed) is
  unchanged; T1 (default f16-scratch numerics) vs T2 (native vec) differ by only 0.0083, consistent with
  the S1 numerics-noise class.

### §12 routing gate R1-R4 (2026-11-03): defaults already optimal, no flip

llama-bench (-ngl 99 -fa on -t 7 -ub 512, -pg 512,128 -pg 4096,128 -r 2, GGML_FATTN_DEBUG=1). NOTE:
`-pg` takes ONE `pp,tg` pair; several pairs must be repeated flags - the `-pg "512,128;4096,128"`
form dies with "invalid parameter for argument: -pg" (string_split on ',' yields 3 fields).

| row | K-V | routing | pp512 t/s | pp4096+tg t/s | tg128 t/s | pp512+tg t/s |
|-----|-----|---------|-----------|---------------|-----------|--------------|
| R1 | turbo4-turbo3 | default (gate ON) | 954.02 | 500.95 | 32.82 | 143.81 |
| R2 | turbo4-turbo3 | NATIVE vec | 955.37 | 493.74 | 32.38 | 142.71 |
| R3 | q5_1-turbo3 | default (vec) | 953.72 | 508.43 | 32.92 | 144.39 |
| R4 | q5_1-turbo3 | FALLBACK f16-scratch | 951.48 | 502.82 | 33.09 | 144.48 |

- KEY FINDING vs the old expectation in the script header: at decode (nq=1) the log prints
  `kernel=vec` for ALL four rows - the routing gate only changes the PREFILL path (nq=512 prints
  `kernel=mma_f16` in R1/R3/R4 and `vec` only in R2). So the env switches cannot move tg128, and the
  old "tg128 R2 ~2x R1" expectation is dead: S1 mux decode is already the production decode path for
  every one of these combos. Consequence for S2: the fused KQ -12% (turbo5 K) lands directly in
  PRODUCTION decode, no routing change needed to collect it.
- R1 >= R2 at pp4096+tg (500.95 vs 493.74, +1.5%) -> keep the turbo4 prefill gate ON (f16-scratch),
  it is still the faster prefill for the dither-heavy type.
- R3 > R4 at pp4096+tg (508.43 vs 502.82, +1.1%) -> keep the native vec prefill for q5_1/turbo3.
- pg512 columns are all within +-0.4% (noise). Verdict: routing defaults unchanged; only the S2 compile
  default flips (below).

### §12 S2 CLOSED (2026-11-03): GGML_CUDA_FATTN_TURBO_KQ_MMA_F16 default flips to ON

Both gates passed, so the cmake option default went OFF -> ON (ggml/CMakeLists.txt:210 + the build-dir
CMakeCache entry, which otherwise keeps overriding the new default) and the comments in
ggml/src/ggml-cuda/CMakeLists.txt + fattn-vec.cuh were updated. The runtime selector still confines the
fused path to TURBO5_0 K; TURBO4_0 K keeps q8_1 dot + range dither (env
GGML_CUDA_FATTN_VEC_TURBO_KQ_MMA_F16_FORCE_T4=1 to experiment), turbo4q is gone.
Production expectation at -b1 -ub1 (no prefill batching, so the turbo4 prefill gate is irrelevant):
turbo4-t3 == 7.1719 (= T1, the pre-flip production row - i.e. turbo4 is bit-identical before/after the
flip, selector exclusion holds) and turbo5-t3 == 7.1868 (= T7, fused really is the default now).
VERIFIED 2026-11-03 on the rebuilt DLL (65 078 272 B @ 00:46, 161 ninja rules carry
-DGGML_CUDA_FATTN_TURBO_KQ_MMA_F16_DEFAULT=1): P1 turbo4-t3 = 7.1719, P2 turbo5-t3 = 7.1868, no env
vars set. Note this also corrects the naive expectation "turbo4 production == T2": the production
turbo4 routing keeps the f16-scratch numerics class (T1) even at -b1 -ub1, so the right equality anchor
for turbo4 is T1, not T2 (T2 needs GGML_CUDA_FATTN_VEC_NATIVE=1).

Production A/B on the flipped DLL (llama-bench -ngl 99 -fa on -t 7 -ub 512 -d 8192 -p 512 -n 128 -r 3,
turbo5-turbo3, nothing else set):

| mode | pp512 @ d8192 | tg128 @ d8192 |
|------|---------------|---------------|
| fused KQ ON (default) | 890.32 t/s | 27.28 t/s |
| forced dot KQ (env =0) | 890.44 t/s | 26.79 t/s |

+1.8% end-to-end decode, prefill untouched (the probe's -12% is the KQ block only; at d8192 the rest of
the graph dominates, so the realized win is 1.8%). Small but free and numerically gated - and it was
already being collected in production because decode always runs the vec kernel (see R1-R4).

### §12 WHY fused was slower on turbo4: it is SPILLING, not the algorithm (2026-11-03 forensics)

User hypothesis (Triton analogy: BLOCK_M/N/K must keep the working fragments inside the SM register file,
otherwise 1.5-2x slowdown from (a) register spilling to local memory, (b) occupancy loss). Measured on
`fa_turbo_ab_new.exe` with `cuobjdump -res-usage` / `-sass -fun` (full tables in
FA_TURBO_BASELINE_20261103.md §1b):

| variant | REG | STACK/thread | LDL+STL | result |
|---------|-----|--------------|---------|--------|
| t4–t3 dot D256 | 255 | 0 | 0 | 1.412 ms |
| t4–t3 FUSED D256 | 255 | **64 B** | **256** | 1.925 ms (+36.3 %) |
| t4–t4 FUSED D256 | 253 | **64 B** | - | 1.964 ms |
| t4–t3 FUSED D128 | 254 | **64 B** | - | 0.855 ms (+70.0 %) |
| t5–t3 dot D256 | 254 | 0 | 0 | 1.697 ms |
| t5–t3 FUSED D256 | 255 | **0** | **0** | 1.503 ms (-11.4 %) |

The hypothesis is confirmed, with one important refinement about WHICH mechanism is live here:
- (b) occupancy is NOT the differentiator: REG is 253-255 in EVERY variant, i.e. the 65536 reg/SM file is
  already saturated by the 255/thread cap and occupancy is register-pinned at 2 blocks/SM in both paths
  (smem 5376 vs 5888 B is far from the limit). Extra live state therefore CANNOT cost blocks/SM - the
  compiler has nowhere to put it.
- (a) spilling is exactly what happens, and it correlates 1:1 with the loss: the only slow variants are
  the only ones with a stack frame (64 B/thread) and LDL/STL traffic in SASS (256 accesses). The fused
  KQ block itself is cheaper than the dot path (32 HMMA instead of the dp4a+decode chain, FFMA 1741 ->
  1390, SHFL 199 -> 44, static total 12488 -> 13688) - the algorithm is fine.

ROOT CAUSE: fattn-vec.cuh:103 in the TURBO4_0 branch of ggml_cuda_fattn_turbo_kq_afrag -
  `float du0[4], du1[4], dv0[4], dv1[4];` passed BY ADDRESS to turbo_dither4(...) (out-param array) and
  then indexed by `kl = 2*(tg & 1)`, a RUNTIME index derived from the lane id. 4 arrays x 4 floats =
  64 B/thread = exactly the reported STACK:64. nvcc cannot scalar-replace an address-taken array indexed
  by a runtime value, so the whole dither group lives in local memory and is re-read per fragment.
  turbo5 (no dither -> no arrays) shows STACK:0 / LDL:0 and IS FASTER than its dot path: it was the
  dither's storage class, not the MMA, that cost 36-70 %.

FIX (S2c, register-only dither; NOT implemented yet): keep turbo_dither4's exact value sequence but
return the group as 4 named scalars (struct/float4) instead of `float * u`, and pick the pair with a bool
from `tg & 1` (predicated register select - kl is only ever 0 or 2). Expect STACK:0 and LDL/STL:0 on the
t4 fused instantiations. Ceiling to aim at = the zero-dither floor already measured with turbo4q fused
(1.293 ms D256 / 0.469 ms D128) against 1.412 / 0.503 for the dot path, i.e. turbo4 fused could land ~8 %
FASTER than dot and the per-type selector exclusion would go away. Order of verification: cuobjdump must
show STACK:0 first, then fa_turbo_ab_new T6/C2 rows, then the PPL rows - all compared against
FA_TURBO_BASELINE_20261103.md.

### §12 S2c implemented + register-budget probe built (2026-11-03, GPU measurement PENDING)

S2c, as planned: `turbo_dither4_pair(block_idx, w, hi, step)` added in ggml-common.h returns the two
needed samples as a `float2` (v1 branch maps to j=4w+2hi+0/1, so the contract `u(block_idx,j)` is kept);
the TURBO4_0 branch of the afrag builder now holds them in scalars and selects the half with `hi = tg & 1`.
Since the pair always comes from ONE of the two wide products, the builder now pays a single 64x32
multiply per group instead of two (`hi ? 668265263u : 3266489917u`).

Register-budget knob added at the same time: `GGML_CUDA_FATTN_VEC_MIN_BLOCKS` (default 1 = today's
behavior) feeds `__launch_bounds__(128, MIN_BLOCKS)`.

Compile-level verification (done; no GPU needed):
- `fa_turbo_frag_probe.exe`: `turbo4(a)=0 turbo4(b-norms)=0 turbo5=0` -> S2c is bitwise identical to the
  pre-S2c fragment builder (which was itself 0/192 against the element reference), so dither VALUES are
  unchanged and no PPL re-baseline is implied.
- `cuobjdump -res-usage` on fa_turbo_ab_new.exe: STACK is 0 in ALL 9 instantiations (was 64 B in the three
  TURBO4_0 fused ones). LOCAL:0 everywhere.
- `cuobjdump -sass -fun` static counts (before -> after S2c):
  * T1 t4-t3 dot: 12488 -> 12488, LDL/STL 0 -> 0, IMAD.WIDE 406 -> 406 (dot path untouched, as intended)
  * T6 t4-t3 FUSED: 13688 -> **12152** (-11.2%), LDL/STL **256 -> 0**, IMAD.WIDE **477 -> 349** (-27%)
  * T4 t5-t3 dot: 14448 -> 14448 (untouched); T8 t5-t3 FUSED: 13208 -> 13208 (no dither, no change)
- MIN_BLOCKS=3 probe (fa_turbo_ab_mb3.exe): REG 255 -> **168** in every instantiation, STACK still **0**,
  i.e. the kernel genuinely fits in 168 registers (cap for 3 blocks/SM is 170) and ptxas needed no spills
  to do it. Occupancy would become 3 blocks/SM = 12 warps instead of 8 (+50 % latency hiding). Since
  neither config spills, this is now a pure occupancy-vs-ILP question answerable only by timing.

BLOCKED ON GPU: `strata.exe` (PID 4776) holds 15 638 MiB at 91 % util, 413 MiB free - no run started
(VRAM policy: never touch foreign processes). Run `run_s2c_gate.cmd` when the GPU is idle: it prints the
bitwise gate, the MIN_BLOCKS=1 timings (compare with baseline §1: T1 1.412 / T3 1.445 / T4 1.697 | T6 1.925
/ T7 1.964 / T8 1.503 | C1 0.503 / C2 0.855) and the MIN_BLOCKS=3 timings.
Decision tree: if t4 fused <= dot after S2c (target ~1.29/0.47), drop the TURBO4_0 exclusion in
kq_f16_sel + rebuild the DLL + re-run the PPL rows; for MIN_BLOCKS, adopt 3 only if it wins on BOTH the
dot and fused rows (it changes every vec instantiation in production, dot included).

### §12 S2d: gate ran, turbo4 fused ENABLED, conditional MIN_BLOCKS policy (2026-11-03)

Gate run: run_s2c_gate.cmd (fixed: the echo line's `|` needed `^|`), log s2c_gate_log.txt.
CAVEAT recorded per VRAM policy: the script's own nvidia-smi header showed 15 762 MiB / 97 % at start
(strata.exe had restarted after the idle snapshot) - the run should have been aborted first. Mitigating
evidence: every byte-identical-SASS dot row reproduced the baseline within 0.2 % across BOTH reps
(T1 1.412/1.414 vs 1.412, T3 1.447 vs 1.445, T4 1.697 vs 1.697, C1 0.503, T8 1.500 vs 1.503), which
time-slicing contention makes impossible - so there was no contention during the kernel timings and the
in-run A/Bs are valid. Absolute numbers of NEW configs still get re-confirmed on a provably idle GPU
(run_s2d_gate.cmd).

Timings (med ms; run1/run2 of the MIN_BLOCKS=1 probe = S2c code):
| row | baseline | S2c r1 | S2c r2 | vs own dot |
|---|---|---|---|---|
| T1 dot t4t3 D256 | 1.412 | 1.412 | 1.414 | - |
| T3 dot t4t4 | 1.445 | 1.447 | 1.447 | - |
| T4 dot t5t3 | 1.697 | 1.697 | 1.697 | - |
| T6 FUSED t4t3 | 1.925 | **1.369** | **1.375** | **-3.1 % vs T1** |
| T7 FUSED t4t4 | 1.964 | **1.416** | **1.414** | **-2.2 % vs T3** |
| T8 FUSED t5t3 | 1.503 | 1.502 | 1.500 | -11.5 % vs T4 (unchanged) |
| C1 dot d128 | 0.503 | 0.503 | 0.503 | - |
| C2 FUSED d128 | 0.855 | **0.491** | **0.489** | **-2.4 % vs C1** |

S2c flipped the speed gate: fused t4 now WINS every row (targets were 1.29/0.47, got 1.369/0.489).
The TURBO4_0 exclusion in kq_f16_sel is removed; the FORCE_T4 override helper is gone with it
(GGML_CUDA_FATTN_VEC_TURBO_KQ_MMA_F16=0 remains the runtime A/B lever).

MIN_BLOCKS=3 (global probe define): -18..-25 % on EVERY D256 turbo row on both paths (T1 1.100/1.087,
T3 1.084, T4 1.398, T6 1.065, T7 1.060, T8 1.194), but +1.0 % on q5_1-K D256 (T2 0.676 vs 0.669) and
+1.4 % on D128 dot (C1 0.510 vs 0.503). So no global adoption: new constexpr ggml_cuda_fattn_vec_min_blocks
raises the cap to 3 blocks/SM (REG 168, STACK 0) only for turbo4/turbo5 K at D=256, everything else keeps
today's codegen. Verified per instantiation with cuobjdump: 6x D256-turbo rows REG=168, q5_1/D128 rows
254/255, STACK 0 everywhere. The win on dot rows whose SOURCE IS BYTE-IDENTICAL is pure occupancy
(8 -> 12 warps/SM): the vec kernel is latency-bound, and 255 regs were ptxas greed, not need.
POLICY REFINEMENT (from cuobjdump -res-usage of the full DLL, 352 vec instantiations): min_blocks
additionally requires ncols==1 - the decode shape the probe actually measured. (The ncols=2 turbo D256
rows spill 80-96 B/thread at BOTH register budgets, 255 and 168: the spill is intrinsic to the shape,
not caused by the cap - but that shape was never timed, so it stays at min_blocks=1.) Result: the 36
ncol=1 turbo-D256 rows sit at REG=168 (STACK 0; the two softcap t5t4 dot variants spill 8 B, a shape
this model never calls), the 36 ncol=2 turbo-D256 rows and every non-turbo instantiation unchanged.

NOT YET MEASURED: fused-t4 PPL (at S2 close T5==T2==7.1802 EXACTLY, i.e. that run proved the exclusion,
not fused-t4 quality). run_s2d_gate.cmd step 3 measures T8 (NATIVE, fused default) vs T9 (NATIVE, forced
dot); gate |delta| <= 0.05 against the 7.1802 dot reference. Step 4 is the production A/B at d8192.
Bitwise dither contract is untouched since S2c (frag probe 0/0/0, re-verified in the same run).

## Reproduce

```
tools\build (vcvars64) && nvcc -O3 --generate-code=arch=compute_120a,code=[sm_120a] tools\fa_fp4_mma_probe.cu -o build...\bin\fa_fp4_mma_probe.exe
fa_fp4_mma_probe.exe --mode verify     # T1/T2/T3 layout proofs
fa_fp4_mma_probe.exe --mode map        # fragment map sweep (32768 one-hots)
fa_fp4_mma_probe.exe --mode scale      # scale_vec lane binding
fa_fp4_mma_probe.exe --mode numerics   # codec score-error table (§5)
fa_fp4_mma_probe.exe --mode residual   # Q residual-split study (§8 residual study); --qfile for real Q
fa_fp4_mma_probe.exe --mode kernelcheck # FA prep/mma replica check (abandoned mid-debug, §9 - results inconclusive)
fa_fp4_mma_probe.exe --mode bench --ntok 16384 --nq 64 --reps 30   # §4 incl. in-kernel repack check
nvcc -O3 --generate-code=arch=compute_120a,code=[sm_120a] tools\fa_fused_v12_diff.cu -o build...\bin\fa_fused_v12_diff.exe
fa_fused_v12_diff.exe   # v1-vs-v2 fused decode bitwise A/B + scalar ref (§10e resolution)
nvcc -O3 --generate-code=arch=compute_120a,code=[sm_120a] tools\fa_vnvfp4_decode_diff.cu -o build...\bin\fa_vnvfp4_decode_diff.exe
fa_vnvfp4_decode_diff.exe   # old-LUT vs new-PRMT nvfp4 V-decode bitwise A/B (§10f, expect PASS + signed-zero counts)
nvcc -O3 --generate-code=arch=compute_120a,code=[sm_120a] tools\fa_nvfp4_cvt_probe.cu -o build...\bin\fa_nvfp4_cvt_probe.exe
fa_nvfp4_cvt_probe.exe      # HW cvt.rn.f16x2.e2m1x2 vs PRMT vs LUT: all-256 bitwise A/B + decode throughput (§10g)
cl /O2 /std:c++17 tools\fa_nvfp4_dot_intdiff.cu /Fe:build...\bin\fa_nvfp4_dot_intdiff.exe && fa_nvfp4_dot_intdiff.exe
                            # PRMT+dp4a KQ-dot decode == kvalues_fp4 LUT, exhaustive host proof (§10h)
nvcc -O3 --generate-code=arch=compute_120a,code=[sm_120a] -I ggml\include -I ggml\src tools\fa_vec_ab_probe.cu -o build...\bin\fa_vec_ab_probe.exe
fa_vec_ab_probe.exe         # production flash_attn_ext_vec timing A/B/C/D/E at nq=1/D=256/nkv=32768 (§10h)
                            # decoder codepath A/B: add -DGGML_CUDA_FORCE_E2M1_PRMT for the PRMT variant (§10i)
tools\fa_nvfp4_devdiff.cu   # device diff: old-LUT dot vs current dot + V-dequant reference matrix (§10j);
                            # build both fa_nvfp4_devdiff_buggy.exe (pre-fix) and fa_nvfp4_devdiff.exe (fixed)
                            # via build_devdiff.cmd [outname], run standalone (~100 MB VRAM)
cuobjdump.exe -sass -fun <mangled kernel name> build...\bin\ggml-cuda.dll   # per-kernel SASS forensics (§10i);
                            # mangled names via: cuobjdump -sass ggml-cuda.dll | findstr "Function : .*flash_attn_ext_vec"
tools\fa_turbo_ab_probe.cu  # turbo vec-kernel A/B: production flash_attn_ext_vec timing + bitwise output dump
                            # for T1..T5 turbo combos at nq=1/D=256/nkv=32768 (§11). Build BOTH sides with
                            # rebuild_turbo_ab_both.cmd (header swap via git show HEAD) -> fa_turbo_ab_old/new.exe,
                            # run each with a dump dir arg, then per-row: Get-FileHash must match, timing should drop.
                            # nvcc flags include -DGGML_TURBO_DITHER (production dither) + sm_120a.
tools\fa_turbo_frag_probe.cu # S2 fragment unit probe: ggml_cuda_fattn_turbo_kq_afrag vs the element reference
                            # for every (f,tg,slot,half); on mismatch prints which coord the fragment word really
                            # holds (§12). Tiny standalone run; same nvcc flags as the ab probe.
```

# S3 — fused turbo dequant in fattn-mma-f16 (replacing the f16 scratch): CLOSED before coding, 2026-11-03

Candidate S3 was: replace the per-FA-op `to_fp16` f16-scratch conversion (`fattn-common.cuh:1556-1618`)
with an in-kernel tile loader that dequantizes turbo4/turbo3/turbo5 blocks straight into the swizzled
K/V smem tiles of `flash_attn_ext_f16` (design: `kind_K/kind_V` template params replacing the dead
`kv_nvfp4` param, D=256 t4/t5×t3 combo, nstages=1). It was never implemented. Both of its premises
are now disproved with evidence - one by code reading, one by direct nsys measurement - so the axis is
closed without writing a line of kernel code.

## TL;DR

| premise | verdict | evidence |
|---|---|---|
| A. "MMA prefill needs a multi-GB f16 copy; turbo KV cannot run long context on 16 GB without native vec" | **FALSE** - scratch is one-op and pool-reused: 134 MiB @32K, 492 MiB @120K peak | §1 |
| B. "the repeated O(n_ctx) dequant dominates deep-context prefill (pp512 945→206 t/s)" | **FALSE** - the whole `to_fp16` turbo pass is **28.0 ms = 0.36 %** of ~9.8 s GPU kernel time in the d8192 trace | §2 |

**Verdict: S3 REJECTED/CLOSED.** A perfect fusion buys ~0.36 % total GPU time and would *re-decode
every KV row once per query-tile CTA* (ntiles_x = 512/16 = 32 at ub512), almost certainly making the
kernel slower than the one bulk bandwidth-optimal pass it removes. The kernels stay untouched.

## 1. Premise A (VRAM) - wrong mechanism, corrected in-tree

The f16 scratch is not a model-level multi-layer copy. `ggml_cuda_flash_attn_ext_get_f16_extra_data`
(`fattn-common.cuh:93-125`) places K_f16/V_f16 *inside the FA op's own scratch extension*
(`dst->data + nbytes + extra`), sized `ggml_nelements(K_view) * 2` where the cache view's `ne[1]` is
the CURRENT `n_kv`, and `ggml_backend_cuda_buffer_type_get_alloc_size` (`ggml-cuda.cu:1274-1278`)
feeds that size to the shared scratch pool, which is reused by every op in the graph.

Peak scratch = ONE op's K+V at current depth: 256 x n_kv x 4 kv-heads x 2 B x 2 =
**134 MiB at 32K, ~0.48 GiB at 120K**. Production runs at 120K context on the 16 GB card with
`-ctk turbo4 -ctv turbo3` without any OOM (user-confirmed, consistent with the code).
The "MMA needs 8.0 GiB of f16 scratch at 32K" comment in
`template-instances/fattn-vec-instance-turbo4_0-turbo4_0.cu` assumed all 64 layers holding scratch
simultaneously - impossible with pool reuse - and has been corrected in this commit.

## 2. Premise B (time tax) - measured with nsys A/B, refuted

Protocol: `run_pp_nsys.cmd` (nsys 2025.6.3, `target-windows-x64\nsys.exe`), two profiles of
llama-bench, identical token flow (`-ub 512 -p 512 -r 1 -ctk turbo4 -ctv turbo3`), one at
`-d 8192` and one at `-d 512`, clean GPU window (0 MiB used / 0 %, header of `s5/pp_nsys_log.txt`).
Reports kept in `s5/pp_nsys_8k_kern.txt`, `s5/pp_nsys_512_kern.txt`, `s5/pp_nsys_8k_memcpy.txt`,
`s5/pp_nsys_8k_api.txt`.

d8192 trace (whole bench: pp512@8192 + tg128@8192 incl. context fill), GPU kernel total ~9.8 s:

| kernel | inst | total ms | avg us | avg us in d512 trace |
|---|---|---|---|---|
| mul_mat_q t21/t18/t23 (weight GEMMs) | 5724 | 5168 | ~880-1070 | identical per-instance |
| `gated_delta_net_cuda<128>` (48 recurrent layers) | 960 | 1293 | 1347 | 751 |
| `flash_attn_ext_f16<256,256,8,8>` (16 FA layers, prefill) | 288 | 377 | 1309 (min 193, max 2553) | 269 |
| `dequantize_block<128,1,turbo4_0/turbo3_0,half>` = **the entire to_fp16 scratch pass S3 would fuse away** | 320+320 | **28.0** | ~43 | - |

The dequant kernels run at ~490 GB/s effective (read 0.52 B + write 2 B per coord) - already
bandwidth-optimal. Their share: **28.0 / 9800 = 0.36 %** of GPU time, ~7 % of the FA kernel's own
time. Even zeroing them completely (impossible; the fused loader re-decodes every KV row once per
query-tile CTA, 32x at ub512, in ALU instead of once in a bulk HBM pass) moves end-to-end pp by far
less than the run-to-run noise floor. There is no S3 ceiling to chase.

## 3. Where the d512→d8192 pp slowdown actually is (recorded, axis NOT opened)

- Ordinary causal attention growth: FA kernel per-call avg 269 us → 1309 us (max 2553); 16 layers x
  ~2.4 ms = ~40 ms at the deepest 512-chunk. Expected quadratic term, tensor cores, nothing anomalous.
- The recurrent side grows with span too: `gated_delta_net` avg 751 → 1347 us and in total costs
  1293 ms vs the FA kernel's 377 ms across the trace - bigger than FA itself. Any future pp work at
  deep context targets GDN, not FA.
- Deep pp is largely launch/sync-gap-bound: measured pp512@8192 wall = 2.48 s/chunk vs roughly
  0.55-0.6 s of GPU kernels per chunk in that regime; trace-wide API totals: cudaStreamSynchronize
  5.2 s, cudaLaunchKernel(+ExC) 5.8 s, H2D memcpy 2.55 s (model load + masks). The 945→206 t/s
  curve is mostly wall-clock gaps plus attention/GDN growth - none of it is the f16 scratch.

## 4. Consequences

- S5/S6 (vec decode) CLOSED earlier; S3 (mma prefill dequant) CLOSED here. The flash-attention axis
  is done for the turbo-KV work in this tree: kernels untouched, production DLL unchanged.
- Remaining unexplored axes from the same trace, if the PPL gates stay green and appetite exists:
  (i) deep-pp launch/sync gap (graph capture / fewer syncs / bigger ub), (ii) gated_delta_net
  long-span behaviour. Neither is scheduled.
- S8 (`turbo-quant.cuh` VTAB decode tables) is unrelated in-flight work by another owner, still
  uncommitted, still default-off.

## How to re-measure

```
run_pp_nsys.cmd > s5\pp_nsys_log.txt 2>&1
"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe" stats ^
    --report cuda_gpu_kern_sum  %TEMP%\opencode\pp8k.nsys-rep
    --report cuda_gpu_mem_time_sum / cuda_api_sum the same way
```

Check `nvidia-smi` for foreign VRAM holders before the profile passes (project VRAM policy); the
stats passes are offline CPU-only re-reads of the saved `.nsys-rep`.

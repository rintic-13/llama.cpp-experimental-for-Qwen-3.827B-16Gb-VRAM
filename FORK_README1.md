# Fork README — trimmed llama.cpp build for a single model

Fork of llama.cpp tuned for **one model**: `Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`
(dense Qwen3.8-27B, IQ3_S quantization, 64 repeating layers + one MTP draft block `blk.64`)
running on an **NVIDIA RTX 5060 Ti 16 GB** (CUDA arch `120a`), Windows / MSVC / Ninja.

Everything below matches the actual source tree and the working build cache
(`build-x64-windows-msvc-release\CMakeCache.txt`). The Russian-language build/build-troubleshooting
guide is in `BUILD.md`; the full experiment protocol is in `tmp_sched_test\ffn_cpu_split_notes.md`.

---

## 1. Build recipe

### 1.1 Configure (once)

Open an *x64 Native Tools Command Prompt for VS 2022* (or call `vcvars64.bat` first), then:

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d <root of llama.cpp-master>

cmake -B build-x64-windows-msvc-release ^
  -G Ninja ^
  -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
  -DGGML_CUDA=ON ^
  -DGGML_NATIVE=ON ^
  -DGGML_OPENMP=ON ^
  -DGGML_LLAMAFILE=ON ^
  -DGGML_CUDA_FA=ON ^
  -DGGML_CUDA_GRAPHS=ON ^
  -DGGML_CUDA_FORCE_MMQ=OFF ^
  -DGGML_CUDA_FORCE_CUBLAS=OFF ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;q8_0-q4_0;q5_1-q5_1;q5_1-q4_1;q5_1-q4_0;q4_1-q4_1;f16-f16;q5_1-turbo4_0;q5_1-turbo3_0;turbo4_0-turbo3_0"
```

`reconfigure_trim.cmd` in the tree root re-runs this with `LLAMA_BUILD_TESTS/EXAMPLES=OFF` without
changing anything else.

Rules that must not be broken:

- **`RelWithDebInfo`, not `Release`** — the working cache is `/O2 /Ob1 /DNDEBUG -Zi`.
- **`GGML_CUDA_FORCE_CUBLAS` must stay `OFF`** — forcing cuBLAS prefill makes IQ3_S dense prefill
  measurably slower; MMQ kernels (including `mul_mat_q<IQ3_S, J=112>`) are compiled by default and
  selected at runtime. MMQ kernel tuning is closed: baseline dense prefill is **832.6–838 t/s @ `-ub 512`**.
- **`GGML_CUDA_FA_QUANTS` must include the turbo combos** (`q5_1-turbo4_0`, `q5_1-turbo3_0`,
  `turbo4_0-turbo3_0`), otherwise `-ctv turbo4/turbo3` falls back and prints
  `no FlashAttention vector kernel compiled for K/V types ...`.
- CUDA arch is auto-detected as `120a-real`; for another Blackwell card add
  `-DCMAKE_CUDA_ARCHITECTURES=120a` explicitly.
- OpenCL / Vulkan / SYCL are OFF in this build; the server target is `llama-server` only.

### 1.2 Build

```bat
cmake --build build-x64-windows-msvc-release --target llama-server
```

or use `rebuild.cmd` in the tree root, which calls `vcvars64.bat`, builds, and logs to
`build-x64-windows-msvc-release\build.log`:

```bat
rebuild.cmd          :: normal incremental build
rebuild.cmd cuda     :: after editing CUDA sources   (touches the CUDA TUs)
rebuild.cmd fa       :: after editing fa-template headers (touches the FA instance TUs)
rebuild.cmd all      :: after editing shared headers  (touches everything)
```

**Incremental-build trap:** CMake generates *unscanned* Ninja rules, so edits to headers are
**not** seen by Ninja — dependent translation units must be touched. `rebuild.cmd <mode>` does that
for you. Full CUDA rebuild ≈ 15–20 min, a single-TU rebuild ≈ 3.5 min.
To read build errors: `tmp_sched_test\show_build_errors.ps1` (the log is not greppable by ripgrep
due to encoding).

### 1.3 Verify the required kernels exist

- `build.ninja` contains `-DGGML_CUDA_FA_Q5_1_TURBO4_0=1`, `-DGGML_CUDA_FA_Q5_1_TURBO3_0=1`,
  `-DGGML_CUDA_FA_TURBO4_0_TURBO3_0=1` etc.
- Objects exist under `ggml\src\ggml-cuda\CMakeFiles\ggml-cuda.dir\template-instances\`
  (`fattn-vec-instance-q5_1-turbo4_0.cu.obj`, `...turbo4_0-turbo3_0.cu.obj`, ...).
- At runtime the log line `system_info: ... CUDA : ARCHS = 1200 | FA_QUANTS = ...turbo4_0,turbo3_0`
  is present and there is **no** "no FlashAttention vector kernel compiled" warning.

---

## 2. What in this fork reduces VRAM consumption

All fork changes are tagged in the source; the main tags are `TAG_FFN_CPU_SPLIT`,
`GGML_SCHED_*`, turbo KV quants, `TAG_ALLOC_SIZE_EXPAND`, `IQ3S_SIGN_SWAR`.

### 2.1 CPU-donor FFN weights (`src\llama-model-loader.*`) — the main VRAM knob

`select_cpu_donors()` lets N dense FFN weights of the model stay **host-resident** instead of
on the GPU. The loader picks which layers get a donor (even spread, or an explicit stride via
`--ffn-cpu-gap`, choosing the weight kind via `--ffn-cpu-op`). Donor tensors are placed on a host
buffer type in `buft_for_tensor`, and each one is uploaded H2D from host RAM inside the split
boundary of its own layer, every token. So `--ffn-cpu-count N` frees ≈ N × weight-size of VRAM
(a `down` weight of this model is 39.18 MiB; 64 donors ≈ 2.45 GB freed, 2.45 GB of PCIe H2D per token).
An explicit `-ot` match always wins over the auto choice; `blk.64` (MTP block) is never a donor.

### 2.2 Scheduler weight-prefetch ring (`ggml\src\ggml-backend.cpp`)

Donor weights arrive through a prefetch ring in the backend scheduler:
slot ring (`GGML_SCHED_N_COPIES`), disjoint H2D destination cells (`GGML_SCHED_H2D_CELLS`, default 1),
prefetch depth resolved **per graph** from the real copy list — `[PREFETCH_WINDOW]` / `[PREFETCH_AUTO]`
(`GGML_SCHED_PREFETCH_DEPTH`, auto by default, clamped to destination-reuse-distance − 1),
pipeline-fill burst (`GGML_SCHED_PREFETCH_BURST`), optional H2D chunking
(`GGML_SCHED_H2D_CHUNK_MIB`, off by default — auto chunking is known-bad with MTP, do not enable),
and event parking so the copy stream never overtakes compute. Host donor buffers are
`cudaHostRegister`-pinned on demand (`[PIN]` log lines), including single tensor ranges inside the
mmap view (`GGML_CUDA_PIN_RANGE`), so donor uploads are real pinned transfers.

### 2.3 Turbo KV-cache quants (`ggml\src\ggml-turbo-quant.c`)

`turbo2` / `turbo3` / `turbo4` KV types (WHT group + PolarQuant, types 44–46) shrink the KV cache
several times versus f16, and dedicated FlashAttention template instances are compiled for the
K/V combos listed in `GGML_CUDA_FA_QUANTS`. Requires FlashAttention (CUDA only).

### 2.4 IQ3_S MMQ dequant patch — `IQ3S_SIGN_SWAR` (`ggml\src\ggml-cuda\mmq-load-tiles.cuh` ~1406)

SWAR sign-extension trick in the IQ3_S MMQ tile loader; part of the 832–838 t/s dense-prefill baseline.

### 2.5 Generic fused CPU dot kernels (`ggml\src\ggml-cpu\quants.c`)

`ggml_vec_dot_q4_K_q8_K_x2/x3` and `ggml_vec_dot_iq3_s_q8_K_x2/x3` — two- and three-output fused
dot products used when host-resident donor weights and the MTP verification run on CPU.
Disable with `GGML_CPU_Q4K_FUSED_DOTS=0`.

### 2.6 Small supporting patches

- `TAG_ALLOC_SIZE_EXPAND` (`ggml-backend.cpp`, `ggml-backend-impl.h`, `ggml-cuda.cu`) — allows the
  CUDA buffer type to expand alloc size for row-padded quantized tensors (needed for IQ3_S tensors).
- `moe_copies` slot-aliasing guard in the scheduler (kept from the MoE work).

### 2.7 What was trimmed out (nothing deleted, sources archived in `tmp_sched_test\archive\`)

- **Phase 2A** — MoE expert host-residency machinery (`ggml-backend.cpp`, −2163 lines). This model
  is dense; upstream `MUL_MAT_ID` partial-expert copy was kept. `GGML_MOE_*` env vars are no longer read.
- **Phase 2B** — FFN host-row splitting (`--ffn-cpu-rows` / `--ffn-host-rows` / `--ffn-host-type`,
  the `build_ffn` split path, split-FFN loader hooks) and the whole `IQ3_S_FLAT3` repacked quant
  stack (enum value 43 is now a hole; turbo ids 44–46 unchanged), −1322 lines over 29 files.
- Those three CLI options are now rejected: `error: invalid argument: --ffn-host-rows` (likewise
  `--ffn-host-type`, `--ffn-cpu-rows`). The CPU-split worker skeleton exists but is dormant.

Validated on-model: speed unchanged, no regressions (user run, 26.10; protocol §13.72–13.73).

---

## 3. Command-line controls

### 3.1 Fork-specific options (`common\arg.cpp`)

| Option | Meaning |
|---|---|
| `--ffn-cpu-count N` | keep N dense FFN weights host-resident (donors); 0 = off. Main VRAM knob. Env: `LLAMA_ARG_FFN_CPU_COUNT` |
| `--ffn-cpu-op OP` | which weight is a donor: `down` (default) \| `up` \| `gate` \| `largest`. Env: `LLAMA_ARG_FFN_CPU_OP` |
| `--ffn-cpu-gap N` | layer stride to place donors on (2 → layers 2,7,12,...) instead of even spread; 0 = auto. Env: `LLAMA_ARG_FFN_CPU_GAP` |
| `-ot "<regex>=CPU0"` | manual alternative: name donor tensors one by one; always wins over the auto choice |

Working ladder on this model is `--ffn-cpu-op down` from 32 to 64 donors
(`gate_up` pairing existed and was reverted — it breaks the prefetch window; §13.70).

### 3.2 Reference run (the line the baseline was measured with)

```bat
build-x64-windows-msvc-release\bin\llama-server.exe -m "G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" ^
  -ngl 99 -c 50000 -b 2048 -ub 512 -np 1 -t 7 -ctk q5_1 -ctv turbo4 -lm mmap -ctxcp 0 -cram 0 -lv 3 ^
  --host 127.0.0.1 --port 8080 ^
  --reasoning-budget 12200 --reasoning-budget-message "Лимит рассуждений исчерпан, дай ответ" ^
  --spec-type draft-mtp --spec-draft-n-max 3 ^
  --ffn-cpu-count 32
```

For benchmark comparisons use a `-c` that does **not** print
`failed to fit params to free device memory` (with `-ctk q5_1`, `-c 150000` does not fit in 16 GB;
the fit checker is donor-blind, so a fit-abort warning at large `-c` is expected, not a bug).

### 3.3 Scheduler environment knobs (read by `ggml-backend.cpp`)

| Variable | Default | Meaning |
|---|---|---|
| `GGML_SCHED_N_COPIES` | auto | prefetch ring slots |
| `GGML_SCHED_PREFETCH` | on | enable/disable weight prefetch |
| `GGML_SCHED_H2D_CELLS` | 1 | H2D destination cells per slot (>1 widens the ring, costs device staging memory) |
| `GGML_SCHED_PREFETCH_DEPTH` | auto | splits ahead the copies are issued; auto is resolved per graph |
| `GGML_SCHED_PREFETCH_BURST` | = depth | leading splits prefetched before the first compute |
| `GGML_SCHED_H2D_CHUNK_MIB` | off | split donor uploads into N-MiB chunks; `auto` is currently known-bad with MTP drafts — avoid |
| `GGML_SCHED_INPUT_SINGLE_COPY` | off | single-copy input handling |
| `GGML_SCHED_CPU_SPLIT_WORKER` | on | split worker (dormant after trim; `0` = serialized behavior for A/B) |
| `GGML_SCHED_DEBUG`, `GGML_SCHED_DEBUG_REALLOC` | off | scheduler debug output |
| `GGML_SCHED_CPU_SPLIT_STATS` | off | donor/split statistics |
| `GGML_SCHED_LOG_VERBOSE` | off | 1 = ring summaries, 2/3 = per-node lines (expensive: `-lv 5` + verbose≥2 costs ~44 % t/s) |
| `GGML_CPU_Q4K_FUSED_DOTS` | on | fused x2/x3 CPU dot kernels; `0` disables |

### 3.4 Diagnostics

The fork's INFO lines (`select_cpu_donors: chose ...`, `[ENV]`, `[PREFETCH_WINDOW]`,
`[PREFETCH_AUTO]`, `[PIN]`) need **`-lv 4`** (`common\log.cpp` maps INFO→TRACE), and
`GGML_SCHED_LOG_VERBOSE=1` for the ring summaries. Keep `-lv 3` for speed runs.

---

## 4. Regression checks (after any source change)

```bat
rebuild.cmd all
powershell -NoProfile -ExecutionPolicy Bypass -File tmp_sched_test\smoke_revert.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File tmp_sched_test\smoke_donors.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File tmp_sched_test\smoke_cut_opts.ps1
```

- `smoke_revert.ps1` — donor layer selection per mode (D5 `[1 3 6 8 10]` must match the saved
  `prerevert\smoke_D5.err`; `up 3`, `largest 4`, `gate 2`; `gate_up` must be rejected).
- `smoke_donors.ps1` — donor A/B/C behaviour unchanged.
- `smoke_cut_opts.ps1` — the trimmed options must be rejected
  (`error: invalid argument: --ffn-host-rows / --ffn-host-type / --ffn-cpu-rows`).

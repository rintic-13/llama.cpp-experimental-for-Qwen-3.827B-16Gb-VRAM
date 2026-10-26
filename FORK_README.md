# llama.cpp Fork — Build and VRAM Optimizations

Experimental fork focused on running Qwen 3.8 27B IQ3_S with reduced VRAM usage and asynchronous CPU-to-GPU weight transfers.

## 1. Build recipe

### Working configuration

- Windows x64, Visual Studio 2022 MSVC, CUDA Toolkit 13.2
- CMake + Ninja
- `CMAKE_BUILD_TYPE=RelWithDebInfo`
- CUDA enabled
- `GGML_CUDA_FORCE_CUBLAS=OFF`
- `GGML_CUDA_FORCE_MMQ=OFF` — leave backend selection to the normal runtime logic; MMQ is the baseline path.

Run from an x64 Visual Studio developer environment with `vcvars64.bat` initialized.

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

cmake -S . -B build-x64-windows-msvc-release -G Ninja ^
  -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
  -DGGML_CUDA=ON ^
  -DGGML_CUDA_FORCE_CUBLAS=OFF ^
  -DGGML_CUDA_FORCE_MMQ=OFF ^
  "-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;q8_0-q4_0;q5_1-q5_1;q5_1-q4_1;q5_1-q4_0;q4_1-q4_1;f16-f16;q5_1-turbo4_0;q5_1-turbo3_0;turbo4_0-turbo3_0"

cmake --build build-x64-windows-msvc-release --parallel
```

The FA quantization list above is taken from the working CMake cache. Keep the custom Turbo combinations when reconfiguring.

### Build scripts and incremental-build caveat

The repository also provides:

```bat
rebuild.cmd cuda
rebuild.cmd fa
rebuild.cmd all
```

Use `cmake --build` for a normal incremental build. Use the relevant `rebuild.cmd` mode when its specialized rebuild behavior is required.

**Ninja caveat:** changes to headers that are not correctly tracked by generated dependency rules can leave stale objects in an incremental build. After changing shared headers or CUDA kernel definitions, verify the rebuild output; if necessary, perform a clean rebuild.

### Verify Flash Attention kernels

Check the effective configuration:

```powershell
Select-String build-x64-windows-msvc-release\CMakeCache.txt `
  -Pattern "GGML_CUDA_FA_QUANTS|GGML_CUDA_FORCE_CUBLAS|GGML_CUDA_FORCE_MMQ"
```

This confirms the configured options, **not** that every requested kernel was compiled. Inspect the actual build output for the expected FA kernel compilation and any unsupported quantization-pair warnings. Test the resulting binary with the intended KV types as well.

### Baseline

- **832–838 tokens/s prefill** at `-ub 512` in the working configuration.
- MMQ is the baseline backend.
- Treat this as a configuration-specific measurement, not a universal llama.cpp benchmark.

## 2. VRAM reduction and performance features

### CPU FFN donors — `select_cpu_donors`

Selected FFN weights are kept in host buffers rather than permanently occupying VRAM. The required data is transferred to the GPU over H2D as inference progresses.

This is the main VRAM-capacity lever. Its trade-off is recurring transfer traffic during generation.

### Scheduler prefetch ring — `GGML_SCHED_*`

Experimental scheduler changes coordinate prefetch slots and H2D cells, including:

- `[PREFETCH_WINDOW]` — automatic prefetch depth.
- Burst scheduling and transfer chunking.
- `[PIN]` — host-memory pinning path.

**Warning:** automatic prefetch depth is known to perform poorly with MTP in some configurations. Validate it with the actual draft model and workload instead of assuming automatic selection is optimal.

### Turbo KV cache — `turbo2`, `turbo3`, `turbo4`

Custom KV-cache quantization paths reduce cache storage and potentially attention data traffic. FA quantization combinations must match the intended K/V types and the kernels actually compiled into the binary.

### IQ3_S MMQ optimization — `IQ3S_SIGN_SWAR`

Custom sign-processing optimization in the IQ3_S MMQ path.

### Fused CPU dot products — `GGML_CPU_Q4K_FUSED_DOTS`

Experimental fused x2/x3 CPU dot-product paths for supported Q4_K workloads.

### Allocation-size expansion — `TAG_ALLOC_SIZE_EXPAND`

Custom allocation-size handling intended to support the fork's modified memory-management paths.

## 3. Retired experiments

The following experimental approaches are no longer exposed as supported CLI features:

- **2A — MoE residency**
- **2B — host rows / `IQ3_S_FLAT3`**

The CLI rejects these options. The experimental material has been retained under `tmp_sched_test\archive\`; it has not been deleted.

These are archived experiments, not supported runtime modes.

## 4. Benchmark matrix

Current baseline:

| Test | Result |
|---|---:|
| Prefill, `-ub 512` | 938 tokens/s |
| 8k-context generation | 58 |
| 64k-context generation | 42 |
| Original llama.cpp comparison | 12 |

For meaningful comparisons, use the same model file, prompt, context length, GPU-layer configuration, KV-cache types, MTP settings and measurement interval. Record prefill and generation separately, and include peak VRAM usage.


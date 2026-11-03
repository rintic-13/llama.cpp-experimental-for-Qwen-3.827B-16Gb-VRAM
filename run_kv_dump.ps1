$ErrorActionPreference = 'Stop'
# run_kv_dump.ps1 - short KV dump run for the NVFP4 feasibility probe.
#
# REQUIRES a build carrying the GGML_KV_DUMP hook in set-rows.cu (any build
# with the hook; numerics unaffected for all non-dump runs).
#
# Produces ppl_results\kv_dump\kv_dump.bin with raw (pre-rotation,
# pre-quantization) K and V rows; analyse afterwards with:
#   probe_nvfp4.exe ppl_results\kv_dump\kv_dump.bin
$root = 'G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main'
$bin  = Join-Path $root 'build-x64-windows-msvc-release\bin\llama-perplexity.exe'
$model= 'G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf'
$eval = Join-Path $root 'ppl_eval\wiki.ab.mixed_800k.txt'
$stamp= Get-Date -Format 'yyyyMMdd_HHmmss'
$out  = Join-Path $root "ppl_results\kv_dump_$stamp"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$env:GGML_KV_DUMP            = Join-Path $out 'kv_dump.bin'
$env:GGML_KV_DUMP_ROWS       = '128'
$env:GGML_KV_DUMP_MAX_CALLS  = '400'
Remove-Item Env:GGML_KV_DUMP_TYPES -ErrorAction SilentlyContinue

Write-Host "[dump] binary: $bin"
Write-Host "[dump] file:   $env:GGML_KV_DUMP"
Write-Host "[dump] note:   current build must carry the set_rows dump hook"
Write-Host "[dump] run:    short perplexity (turbo4 K, turbo5 V)"

$extraArgs = '--chunks', '16'
& cmd /c "`"$bin`" -m `"$model`" -f `"$eval`" --chunks 16 -c 2048 -b 2048 -ub 512 -fa on -ngl 99 -t 7 -ctk turbo4 -ctv turbo5 --no-warmup > `"$out\dump.log`" 2>&1"

Remove-Item Env:GGML_KV_DUMP -ErrorAction SilentlyContinue
$sz = (Get-Item "$out\kv_dump.bin" -ErrorAction SilentlyContinue).Length
Write-Host "[dump] done: $out\kv_dump.bin ($sz bytes); log: $out\dump.log"
Write-Host "[dump] analyse: .\build-x64-windows-msvc-release\bin\probe_nvfp4.exe `"$out\kv_dump.bin`" --k turbo4_0 --v turbo5_0"

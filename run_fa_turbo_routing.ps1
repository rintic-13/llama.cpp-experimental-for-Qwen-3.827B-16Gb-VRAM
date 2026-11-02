# run_fa_turbo_routing.ps1 - re-evaluate the 2026-10-31 turbo routing gate AFTER S1 (post-mux vec kernel)
# Rows: same build, same cache bits; env switches are pure routing A/B (see fattn.cu:435-447).
#   default      = current gate: any turbo4 combo -> f16-scratch path (full-cache convert per FA call)
#   NATIVE=1     = force native vec kernel (S1-mux decode) for turbo4 combos
#   FALLBACK=1   = force f16-scratch for turbo3/nvfp4 combos too (reference rows)
# GGML_FATTN_DEBUG=1 prints the routing proof line once per signature (kernel=vec / mma_f16, vec_compiled=...).
# Memory note: the f16-scratch path allocates ~full-cache-size f16 per FA node; at deep contexts prefer the
# NATIVE rows (the docs record scratch OOM at d32768 on 16 GB).
$root = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main"
$bench = "$root\build-x64-windows-msvc-release\bin\llama-bench.exe"
$model = "G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
Set-Location $root

function Clear-Env {
    Remove-Item Env:GGML_CUDA_FATTN_VEC_NATIVE -ErrorAction SilentlyContinue
    Remove-Item Env:GGML_CUDA_FATTN_VEC_FALLBACK -ErrorAction SilentlyContinue
    Remove-Item Env:GGML_FATTN_DEBUG -ErrorAction SilentlyContinue
}
function Run-Row($name, $kv, $envvar) {
    Clear-Env
    if ($envvar -eq "NATIVE")   { $env:GGML_CUDA_FATTN_VEC_NATIVE = "1" }
    if ($envvar -eq "FALLBACK") { $env:GGML_CUDA_FATTN_VEC_FALLBACK = "1" }
    $env:GGML_FATTN_DEBUG = "1"
    Write-Host "==================== $name ($kv, routing=$envvar) ====================" -ForegroundColor Cyan
    & $bench -m $model -ngl 99 -fa on -t 7 -ub 512 $kv -pg "512,128;4096,128" -r 2
    Clear-Env
}

Write-Host "GPU state (check first; abort if foreign VRAM users):" -ForegroundColor Yellow
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv

Run-Row "R1 turbo4-turbo3  GATE-ON (current default)" "-ctk turbo4_0 -ctv turbo3_0" ""
Run-Row "R2 turbo4-turbo3  NATIVE vec (S1 kernel)"     "-ctk turbo4_0 -ctv turbo3_0" "NATIVE"
Run-Row "R3 q5_1-turbo3    NATIVE (default, S1)"       "-ctk q5_1     -ctv turbo3_0" ""
Run-Row "R4 q5_1-turbo3    FALLBACK f16-scratch"       "-ctk q5_1     -ctv turbo3_0" "FALLBACK"
Run-Row "R5 turbo4q-turbo4 GATE-ON (default)"          "-ctk turbo4q_0 -ctv turbo4_0" ""
Run-Row "R6 turbo4q-turbo4 NATIVE vec"                 "-ctk turbo4q_0 -ctv turbo4_0" "NATIVE"
Write-Host "Compare pp512/pp4096/tg128 columns across pairs (R1vsR2, R3vsR4, R5vsR6)."
Write-Host "Expectation post-S1: tg128 R2 ~2x R1 (probe: vec 2.89->1.42ms); pp roughly tie (both do one convert per prefill call, not per token)."

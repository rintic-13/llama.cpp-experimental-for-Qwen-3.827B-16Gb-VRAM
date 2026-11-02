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
    Write-Host "==================== $name ($($kv -join ' '), routing=$envvar) ====================" -ForegroundColor Cyan
    # NB: $kv must be a STRING ARRAY - a single quoted string ("-ctk x -ctv y") becomes ONE argv entry
    # and llama-bench dies with "invalid parameter for argument".  Array elements splat as separate args.
    & $bench -m $model -ngl 99 -fa on -t 7 -ub 512 @kv -pg "512,128" -pg "4096,128" -r 2
    Clear-Env
}

Write-Host "GPU state (check first; abort if foreign VRAM users):" -ForegroundColor Yellow
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv

Run-Row "R1 turbo4-turbo3  GATE-ON (current default)" @("-ctk","turbo4","-ctv","turbo3") ""
Run-Row "R2 turbo4-turbo3  NATIVE vec (S1 kernel)"     @("-ctk","turbo4","-ctv","turbo3") "NATIVE"
Run-Row "R3 q5_1-turbo3    NATIVE (default, S1)"       @("-ctk","q5_1","-ctv","turbo3") ""
Run-Row "R4 q5_1-turbo3    FALLBACK f16-scratch"       @("-ctk","q5_1","-ctv","turbo3") "FALLBACK"
# R5/R6 (turbo4q_0) removed 2026-11-02: type retired from the FA path (worse PPL class, branch buried -
# SS12).
Write-Host "Compare pp512/pp4096/tg128 columns across pairs (R1vsR2, R3vsR4)."
# Measured 2026-11-03 (DLL with the S2 fused KQ default ON): the routing env switches only move the
# PREFILL path (nq=512 logs kernel=mma_f16 vs vec); at nq=1 ALL rows log kernel=vec, so tg128 cannot
# change and the old "tg128 R2 ~2x R1" expectation is obsolete - S1 mux decode is already production
# decode. Kept gate ON for turbo4 (R1 500.95 > R2 493.74) and vec for q5_1/turbo3 (R3 508.43 > R4 502.82).
# NOTE: -ctk/-ctv take the ggml type_name spelling WITHOUT the _0 suffix (turbo4/turbo5/turbo3); the
# cmake GGML_CUDA_FA_QUANTS combo strings are the ones that carry _0 (turbo4_0-turbo3_0).

# =============================================================================
# run_ab_dither.ps1 - A/B comparison: subtractive dither (arXiv 2605.13810) for
# the MSE turbo types (turbo2..5) vs the deterministic 2026-10-31 baselines.
#
# PREREQUISITE: the binary must be built with the flag ON:
#   powershell -ExecutionPolicy Bypass -File .\build_turboq.ps1 -Dither
# (a plain rebuild without -Dither reproduces the baseline codec bit-for-bit).
#
# Stage 1 (smoke, ~2 min, confirms the dithered build runs end-to-end):
#   powershell -ExecutionPolicy Bypass -File .\run_ab_dither.ps1 -Smoke
# Stage 2 (A/B PPL, ~10 min, quick subset at ctx 16384):
#   powershell -ExecutionPolicy Bypass -File .\run_ab_dither.ps1
# Survive session close (detached):
#   Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main\run_ab_dither.ps1' -WindowStyle Hidden
#
# Baselines to compare against (build WITHOUT dither, ppl_results\20261031_025751\summary.txt):
#   f16-f16          5.0414   (reference, dither does not change it)
#   q5_1-turbo5      5.0346   <- dither should improve or match
#   q5_1-turbo4      5.0560   <- dither should improve or match
#   q5_1-turbo3      5.1040   <- dither most likely improves (coarse cells)
#   turbo4-turbo3    5.1391   (K and V both dithered)
#
# Notes:
#   - With GGML_TURBO_DITHER the FA vector kernel refuses turbo K/V combos; decode
#     runs on the f16-scratch fallback (correct but slightly slower decode). Prefill
#     already used the same convert.cu dequant path in every previous run, so the
#     dither round-trip (set_rows seed == convert.cu ib) is what PPL exercises here.
#   - turbo3q/turbo4q are intentionally NOT dithered; they are not in this A/B.
# =============================================================================
param(
    [switch]$Smoke,
    [switch]$DryRun,
    [string]$RepoRoot = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main"
)
$ErrorActionPreference = "Stop"

if ($Smoke) {
    $pairs = "q5_1-turbo5,q5_1-turbo3"
    $ctx   = 4096
} else {
    $pairs = "f16-f16,q5_1-turbo5,q5_1-turbo4,q5_1-turbo3,turbo4-turbo3"
    $ctx   = 16384
}

$driver = Join-Path $RepoRoot "run_ppl_kv.ps1"
$drvArgs = @("-ExecutionPolicy","Bypass","-File",$driver,"-Quick","-Ctx",$ctx,"-Batch","8192","-UBatch","512","-Pair",$pairs)
if ($DryRun) { $drvArgs += "-DryRun" }

& powershell @drvArgs

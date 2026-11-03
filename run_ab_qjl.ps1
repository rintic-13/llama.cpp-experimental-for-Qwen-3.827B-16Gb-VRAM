# =============================================================================
# run_ab_qjl.ps1 - A/B comparison: TurboQuant Q_prod (turbo3q/turbo4q, MSE +
# 1-bit QJL residual sign) vs the MSE-only turbo types, via run_ppl_kv.ps1.
#
# Stage 1 (smoke, ~2 min, checks new FA kernels really run - watch for status):
#   powershell -ExecutionPolicy Bypass -File .\run_ab_qjl.ps1 -Smoke
# Stage 2 (A/B PPL, ~10 min, quick subset at ctx 16384):
#   powershell -ExecutionPolicy Bypass -File .\run_ab_qjl.ps1
# Survive session close (detached):
#   Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main\run_ab_qjl.ps1' -WindowStyle Hidden
#
# Pairs (K-V) and their KV budget per 128 values:
#   f16-f16          reference
#   q5_1-turbo4      4.125 bpw V  (MSE 4-bit)
#   q5_1-turbo4q     4.25 bpw V   (Q_prod: 3-bit + QJL)
#   q5_1-turbo3      3.125 bpw V  (MSE 3-bit)
#   q5_1-turbo3q     3.25 bpw V   (Q_prod: 2-bit + QJL)
#   q5_1-turbo5      5.125 bpw V  (MSE 5-bit, post-fix reference)
#   turbo4-turbo3    4.125/3.125 bpw K/V  (previous best cheap pair)
#   turbo4-turbo3q   4.125/3.25 bpw K/V   (V = Q_prod 2-bit+QJL)
#   turbo4q-turbo4q  4.25 bpw K/V         (both Q_prod, native K vec_dot)
# Status OK means the compiled FA vector kernel was used; FA_FALLBACK_F16_INVALID
# would mean the pair silently fell back to f16 (should not happen for these pairs).
# =============================================================================
param(
    [switch]$Smoke,
    [switch]$DryRun,
    [string]$RepoRoot = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main"
)
$ErrorActionPreference = "Stop"

if ($Smoke) {
    # covers: K vec_dot turbo4q + V turbo4q + V turbo3q (two FA combos)
    $pairs = "turbo4q-turbo4q,q5_1-turbo4q,q5_1-turbo3q,turbo4-turbo3q"
    $ctx   = 4096
} else {
    $pairs = "f16-f16,q5_1-turbo4,q5_1-turbo4q,q5_1-turbo3,q5_1-turbo3q,q5_1-turbo5,turbo4-turbo3,turbo4-turbo3q,turbo4q-turbo4q"
    $ctx   = 16384
}

$driver = Join-Path $RepoRoot "run_ppl_kv.ps1"
$drvArgs = @("-ExecutionPolicy","Bypass","-File",$driver,"-Quick","-Ctx",$ctx,"-Batch","8192","-UBatch","512","-Pair",$pairs)
if ($DryRun) { $drvArgs += "-DryRun" }

& powershell @drvArgs

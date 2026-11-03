# =============================================================================
# run_check_nvfp4_stageA.ps1 - проверка Stage A (NVFP4-like KV cache, single-level)
#
# Схема: per-16 UE4M3 scale (amax/6, max policy) + E2M1, БЕЗ вращения и дизера
# (пробник 2026-10-31: relMSE raw K 0.8715% / V 0.8739%, sat/zero scale = 0).
# Энкодер - k_set_rows_nvfp4 (set-rows.cu), KQ dot - vec_dot_fattn_vec_KQ_nvfp4,
# V dequant - dequantize_V_nvfp4 (fattn-common.cuh), маршрутизация - fattn.cu.
#
# Предусловие: бинарь собран (-ftk не нужен, nvfp4 в FA_TYPES по умолчанию):
#   powershell -ExecutionPolicy Bypass -File .\build_turboq.ps1 -Dither
#
# Запуск:
#   powershell -ExecutionPolicy Bypass -File .\run_check_nvfp4_stageA.ps1          # ~4-5 мин, 4 пары, ctx 16384
#   powershell -ExecutionPolicy Bypass -File .\run_check_nvfp4_stageA.ps1 -Smoke   # ~2 мин, ctx 4096
#
# Ожидания:
#   q5_1-turbo5   = 5.0346 строго, статус OK            (контроль: v2-дизер не затронут)
#   turbo4-turbo3 = 5.0719,       статус FA_FALLBACK    (контроль: гейт turbo4 не тронут)
#   q5_1-nvfp4    число не сверяем, статус OK          (K=q5_1 4.5 bpw, V=nvfp4 4.0 bpw; basis для big-eval)
#   nvfp4-nvfp4   число не сверяем, статус OK          (K=V=nvfp4, 9.0 bpw; главное - bit-exactness encoder)
# Внимание: nvfp4 KV НЕ вращается (llama-kv-cache исключает из attn_rot) - если в
# статусе/логе появится вращение или статус FA_FALLBACK для nvfp4-пары, это баг маршрутизации.
# Быстрая sanity: см. new пара nvfp4-nvfp4 должна дать PPL строго лучше пары q5_1-turbo3
# (5.1136) при сопоставимом... (не проверяем автоматически, глаза).
#
# Выход: 0 = все проверки прошли, 1 = есть расхождения.
# =============================================================================
param(
    [switch]$Smoke,
    [string]$RepoRoot = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main"
)
$ErrorActionPreference = "Stop"

if ($Smoke) {
    $pairs = "q5_1-turbo5,nvfp4-nvfp4"
    $ctx   = 4096
} else {
    $pairs = "q5_1-turbo5,turbo4-turbo3,q5_1-nvfp4,nvfp4-nvfp4"
    $ctx   = 16384
}

$drvArgs = @("-ExecutionPolicy","Bypass","-File",(Join-Path $RepoRoot "run_ppl_kv.ps1"),
             "-Quick","-Ctx",$ctx,"-Batch","8192","-UBatch","512","-Pair",$pairs)
& powershell @drvArgs

$latest = Get-ChildItem (Join-Path $RepoRoot "ppl_results") -Directory |
          Where-Object { $_.Name -match '^\d{8}_\d{6}$' } |
          Sort-Object Name | Select-Object -Last 1
$csvPath = Join-Path $latest.FullName "ppl_kv_results.csv"
if (-not (Test-Path $csvPath)) { Write-Host "[CHECK] CSV не найден: $csvPath"; exit 1 }
$rows = Import-Csv $csvPath
Write-Host ""
Write-Host "[CHECK] Результаты: $csvPath"

$checks = if ($Smoke) {
    @(
        @{ K="q5_1";  V="nvfp4";    PPL=$null;   Tol=0;     Status="OK" },          # nvfp4-nvfp4 контролируем только статус
        @{ K="q5_1";  V="turbo5";   PPL=$null;   Tol=0;     Status="OK" },
        @{ K="nvfp4"; V="nvfp4";    PPL=$null;   Tol=0;     Status=$null }
    )
} else {
    @(
        @{ K="q5_1";  V="turbo5";   PPL=5.0346;  Tol=0.0005; Status="OK" },          # контроль dither v2, не менялся
        @{ K="turbo4";V="turbo3";   PPL=5.0719;  Tol=0.0005; Status="FA_FALLBACK_F16_INVALID" }, # контроль гейта turbo4, не менялся
        @{ K="q5_1";  V="nvfp4";    PPL=$null;   Tol=0;      Status="OK" },
        @{ K="nvfp4"; V="nvfp4";    PPL=$null;   Tol=0;      Status="OK" }
    )
}

$fail = 0
Write-Host ("{0,-14} {1,9} {2,9} {3,9}  {4,-14} {5}" -f "pair","PPL","expect","delta","status","verdict")
Write-Host ("-" * 78)
foreach ($c in $checks) {
    $row = $rows | Where-Object { $_.K -eq $c.K -and $_.V -eq $c.V }
    if (-not $row) {
        Write-Host ("[CHECK] пара {0}-{1} НЕ НАЙДЕНА в CSV" -f $c.K, $c.V)
        $fail++
        continue
    }
    $ppl      = [double]$row.PPL
    $status   = $row.Status
    $verdict  = "OK"
    $expect   = if ($c.PPL) { $c.PPL } else { "" }
    if ($null -ne $c.PPL) {
        $delta = $ppl - $c.PPL
        if ([math]::Abs($delta) -gt $c.Tol) { $verdict = "FAIL(PPL)"; $fail++ }
    } else {
        $delta = ""
    }
    if ($null -ne $c.Status -and $status -ne $c.Status) {
        if ($verdict -eq "OK") { $verdict = "FAIL(status)"; $fail++ } else { $verdict += "+FAIL(status)"; $fail++ }
    }
    Write-Host ("{0,-14} {1,9:F4} {2,9} {3,9}  {4,-14} {5}" -f "$($c.K)-$($c.V)", $ppl, $expect, $delta, $status, $verdict)
}

if ($fail -eq 0) {
    Write-Host ""
    Write-Host "[CHECK] ALL STAGE A CHECKS PASS"
    exit 0
} else {
    Write-Host ""
    Write-Host "[CHECK] $fail проверок не пройдено"
    exit 1
}

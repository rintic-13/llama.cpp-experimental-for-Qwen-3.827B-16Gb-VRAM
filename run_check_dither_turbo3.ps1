# =============================================================================
# run_check_dither_turbo3.ps1 - контрольный прогон сборки дизера turbo3+turbo4
# (GGML_TURBO_DITHER применяется к turbo3_0 И turbo4_0, turbo5 без дизера).
#
# Предусловие: бинарь собран с гейтом:
#   powershell -ExecutionPolicy Bypass -File .\build_turboq.ps1 -Dither
#
# Запуск:
#   powershell -ExecutionPolicy Bypass -File .\run_check_dither_turbo3.ps1          # ~4 мин, 4 пары, ctx 16384
#   powershell -ExecutionPolicy Bypass -File .\run_check_dither_turbo3.ps1 -Smoke   # ~2 мин, ctx 4096, только статусы FA
#
# Ожидаемые результаты (эталон без дизера: ppl_results\20261031_025751,
# дизер v1 (покоординатный murmur): ppl_results\20261031_142641):
# ДИЗЕР v2 (turbo_dither4: один хэш на группу из 4 смежных компонент; 4 сэмпла из старших
# 32 бит двух широких умножений) - эталоны 20261031_170300 ниже строгие (Tol=0.0005).
# Селективный гейт (20261031_170836 bench, -d 8192): K=turbo4/V=turbo4 -> f16-скратч MMA
# (статус FA_FALLBACK, на 6-17% быстрее vec на 8k KV); V=turbo3 остаётся на нативном vec
# (+0.2%). A/B-переключатели: GGML_CUDA_FATTN_VEC_NATIVE=1 / GGML_CUDA_FATTN_VEC_FALLBACK=1;
# генератор дизера: build_turboq.ps1 -DitherV1 (cmake GGML_TURBO_DITHER_V1).
# Вопрос v1-vs-v2-vs-none по PPL решается прогоном run_ab_dither_big.ps1 (крупный eval).
#   q5_1-turbo5    = 5.0346 строго                                        статус OK
#   q5_1-turbo4    = 5.0625                                               статус FA_FALLBACK
#   q5_1-turbo3    = 5.1136                                               статус OK
#   turbo4-turbo3  = 5.0719                                               статус FA_FALLBACK
# PPL не меняется оттого, что vec-ядро теперь само вычитает дизер: префил и так шёл через
# MMA_F16 + convert.cu. Векторное ядро влияет только на декод - для его проверки есть
# run_check_decode_vec.ps1 (llama-bench tg + сверка генерации vec vs фолбэк).
#
# Скрипт после прогонов сам сверяет CSV и печатает PASS/FAIL по каждому пункту.
# Выход: 0 = все проверки прошли, 1 = есть расхождения.
# =============================================================================
param(
    [switch]$Smoke,
    [string]$RepoRoot = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main"
)
$ErrorActionPreference = "Stop"

if ($Smoke) {
    # sanity за 2 минуты: turbo5 (без дизера) и turbo3 (с дизером) оба на нативном vec-ядре => статус OK
    $pairs = "q5_1-turbo5,q5_1-turbo3"
    $ctx   = 4096
} else {
    $pairs = "q5_1-turbo5,q5_1-turbo4,q5_1-turbo3,turbo4-turbo3"
    $ctx   = 16384
}

$drvArgs = @("-ExecutionPolicy","Bypass","-File",(Join-Path $RepoRoot "run_ppl_kv.ps1"),
             "-Quick","-Ctx",$ctx,"-Batch","8192","-UBatch","512","-Pair",$pairs)
& powershell @drvArgs

# --- найти самый свежий каталог результатов и сверить CSV -------------------
$latest = Get-ChildItem (Join-Path $RepoRoot "ppl_results") -Directory |
          Where-Object { $_.Name -match '^\d{8}_\d{6}$' } |
          Sort-Object Name | Select-Object -Last 1
$csvPath = Join-Path $latest.FullName "ppl_kv_results.csv"
if (-not (Test-Path $csvPath)) { Write-Host "[CHECK] CSV не найден: $csvPath"; exit 1 }
$rows = Import-Csv $csvPath
Write-Host ""
Write-Host "[CHECK] Результаты: $csvPath"

# Ожидания: K,V, ожидаемый PPL ($null = не сверять число), допуск, ожидаемый статус
$checks = if ($Smoke) {
    @(
        @{ K="q5_1";  V="turbo5"; PPL=$null;     Tol=0.0; Status="OK" },
        @{ K="q5_1";  V="turbo3"; PPL=$null;     Tol=0.0; Status="OK" },
        @{ K="turbo4";V="turbo3"; PPL=$null;     Tol=0.0; Status=$null } # в smoke нет этой пары
    )
} else {
    @(
        @{ K="q5_1";  V="turbo5"; PPL=5.0346; Tol=0.0005; Status="OK" },          # turbo5 без дизера = эталон (не менялся)
        @{ K="q5_1";  V="turbo4"; PPL=5.0625; Tol=0.0005; Status="FA_FALLBACK" }, # v2-дизер; K/V=turbo4 -> гейт на MMA (скорость)
        @{ K="q5_1";  V="turbo3"; PPL=5.1136; Tol=0.0005; Status="OK" },          # v2-дизер; V=turbo3 на нативном vec (+0.2% tg)
        @{ K="turbo4";V="turbo3"; PPL=5.0719; Tol=0.0005; Status="FA_FALLBACK" }  # v2-дизер; K=turbo4 -> гейт на MMA
    )
}

$fail = 0
Write-Host ("{0,-14} {1,9} {2,9} {3,9}  {4,-22} {5}" -f "пара","PPL","ожидаем","дельта","статус","итог")
Write-Host ("-" * 78)
foreach ($c in $checks) {
    if ($null -eq $c.PPL -and $Smoke -and $c.V -eq "turbo3" -and $c.K -eq "turbo4") { continue }
    $row = $rows | Where-Object { $_.K -eq $c.K -and $_.V -eq $c.V } | Select-Object -First 1
    if (-not $row) { Write-Host ("{0,-14} {1,9} {2,9} {3,9}  {4,-22} {5}" -f "$($c.K)-$($c.V)","-","-","-","-","FAIL (нет строки в CSV)"); $fail++; continue }

    $ppl  = if ($row.PPL)  { [double]$row.PPL }  else { $null }
    $ok   = $true
    $dsup = "-"
    if ($null -ne $c.PPL) {
        if ($null -eq $ppl) { $ok = $false; $dsup = "PPL не получен" }
        else {
            $d = $ppl - $c.PPL
            $dsup = ("{0:+0.0000;-0.0000}" -f $d)
            if ([math]::Abs($d) -gt $c.Tol) { $ok = $false; $dsup += " вне допуска" }
        }
    }
    if ($null -ne $c.Status) {
        if ($row.Status -notmatch $c.Status) { $ok = $false }
    }
    if (-not $ok) { $fail++ }
    Write-Host ("{0,-14} {1,9} {2,9} {3,9}  {4,-22} {5}" -f "$($c.K)-$($c.V)",
                $(if ($null -ne $ppl)  { "{0:F4}" -f $ppl }  else { "-" }),
                $(if ($null -ne $c.PPL){ "{0:F4}" -f $c.PPL } else { "-" }),
                $dsup, $row.Status, $(if ($ok) { "PASS" } else { "FAIL" }))
}
Write-Host ("-" * 78)
if ($fail -eq 0) { Write-Host "[CHECK] ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ($($checks.Count) шт.)"; exit 0 }
else             { Write-Host "[CHECK] ПРОВАЛОВ: $fail"; exit 1 }

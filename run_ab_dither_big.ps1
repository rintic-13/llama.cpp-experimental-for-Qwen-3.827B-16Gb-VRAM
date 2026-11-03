# =============================================================================
# run_ab_dither_big.ps1 - крупный PPL A/B генератора вычитающего дизера (v2 / OFF / v1)
#
# Зачем: на quick-подмножестве (200 KB) разница генераторов 0.03-0.05 по PPL, а
# собственная оценка +- (~0.09) её перекрывает. Здесь eval ~2 МБ (весь wiki.test.raw
# 1.29 МБ + голова wiki.valid.raw) -> собственных +- ~0.03. ГЛАВНОЕ: сравниваются
# ОДНИ И ТЕ ЖЕ тексты в разных сборках, дисперсия текста в разность сокращается,
# поэтому разность различима лучше, чем обещают колонки +/-.
# valid вместо train: train, скорее всего, в преобучении модели (запоминание),
# valid - стандартный held-out, методологически чище.
#
# Конфигурации (пересборку выполняет ассистент; между прогонами обязательна ПЕРЕССБОРКА):
#   -Label V2  : build_turboq.ps1 -Dither            (дефолт: групповой хэш v2, гейт MMA для turbo4)
#   -Label OFF : build_turboq.ps1                   (без дизера, бит-в-бит baseline; гейт выключен -> статусы OK)
#   -Label V1  : build_turboq.ps1 -Dither -DitherV1 (покоординатный murmur v1)
#
# Запуск (каждая пара ~8-9 мин pp, три пары ~25 мин на конфигурацию):
#   powershell -ExecutionPolicy Bypass -File .\run_ab_dither_big.ps1 -Label V2
#   ... пересборка OFF ...
#   powershell -ExecutionPolicy Bypass -File .\run_ab_dither_big.ps1 -Label OFF
#   ... пересборка V1 ...
#   powershell -ExecutionPolicy Bypass -File .\run_ab_dither_big.ps1 -Label V1
#
# Контроль q5_1-turbo5 (-SkipControl для отключения): turbo5 НИКОГДА не дизерится,
# его PPL на big-eval обязан совпасть во всех трёх конфигурациях до цифр - чистая
# проверка, что пересборка не уронила численность ядер.
#
# Итог: таблица на экране + дописывание в один сводный файл
#   ppl_results\ab_dither_big_summary.csv  (ASCII: label,when,pair,ppl,unc,tokens,pp_tps,status)
# Ориентиры quick-набора (200 KB, не сравнимы напрямую с big-eval):
#   OFF: turbo5 5.0346 / turbo4pair(V=t4) 5.0560 / turbo3pair 5.1040 / t4-t3 5.1391
#   v1 : turbo4pair 5.0598 / turbo3pair 5.0799 / t4-t3 5.1182
#   v2 : turbo4pair 5.0625 / turbo3pair 5.1136 / t4-t3 5.0719
# =============================================================================
param(
    [string]$Label = "V2",              # тег строки сводки: V2 | OFF | V1
    [int]$ExtraKB = 800,                # сколько КБ головы wiki.valid.raw добавить к wiki.test.raw
    [switch]$SkipControl,               # не гонять q5_1-turbo5 (экономия ~8 мин, но теряется контроль сборок)
    [switch]$RebuildEval,               # пересоздать смешанный eval (если меняется ExtraKB)
    [string]$RepoRoot = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main"
)
$ErrorActionPreference = "Stop"

# --- 1. смешанный eval: побайтовая склейка без перекодирования -----------------
$testRaw  = Join-Path $RepoRoot "ppl_eval\wikitext-2-raw\wiki.test.raw"
$validRaw = Join-Path $RepoRoot "ppl_eval\wikitext-2-raw\wiki.valid.raw"
foreach ($f in @($testRaw, $validRaw)) { if (-not (Test-Path $f)) { Write-Host "нет файла: $f"; exit 1 } }
$mixed = Join-Path $RepoRoot ("ppl_eval\wiki.ab.mixed_{0}k.txt" -f $ExtraKB)
if ($RebuildEval -and (Test-Path $mixed)) { Remove-Item $mixed -Force }
if (-not (Test-Path $mixed)) {
    $tb = [IO.File]::ReadAllBytes($testRaw)
    $vb = [IO.File]::ReadAllBytes($validRaw)
    $take = [math]::Min($ExtraKB * 1024, $vb.Length)
    # обрезаем по последней границе абзацев: два идущих подряд LF (пустая строка МОЖЕТ СОДЕРЖАТЬ
    # ПРОБЕЛ - в wikitext-2-raw абзацы разделены "\n \n", соседних LF нет!). Резка по LF безопасна для UTF-8.
    $cut = $take
    for ($i = $take - 1; $i -ge 2; $i--) {
        if ($vb[$i] -ne 10) { continue }
        $j = $i - 1
        while ($j -ge 0 -and ($vb[$j] -eq 13 -or $vb[$j] -eq 32)) { $j-- }
        if ($j -ge 0 -and $vb[$j] -eq 10) { $cut = $i + 1; break }
    }
    $out = New-Object byte[] ($tb.Length + 2 + $cut)
    [Array]::Copy($tb, 0,                             $out, 0,           $tb.Length)
    [Array]::Copy([byte[]](10,10), 0,                 $out, $tb.Length,  2)
    [Array]::Copy($vb, 0,                             $out, $tb.Length+2, $cut)
    [IO.File]::WriteAllBytes($mixed, $out)
    Write-Host ("[AB] смешанный eval создан: {0}  ({1:N0} байт = test {2:N0} + valid-голова {3})" -f `
                $mixed, $out.Length, $tb.Length, $cut)
} else {
    Write-Host "[AB] смешанный eval уже есть: $mixed ($((Get-Item $mixed).Length) байт)"
}

# --- 2. прогон драйвера --------------------------------------------------------
$pairs = if ($SkipControl) { "q5_1-turbo3,turbo4-turbo3" } else { "q5_1-turbo5,q5_1-turbo3,turbo4-turbo3" }
$drvArgs = @("-ExecutionPolicy","Bypass","-File",(Join-Path $RepoRoot "run_ppl_kv.ps1"),
             "-EvalFile",$mixed,"-Ctx","16384","-Batch","8192","-UBatch","512","-Pair",$pairs)
& powershell @drvArgs
if ($LASTEXITCODE -ne 0) { Write-Host "[AB] драйвер завершился кодом $LASTEXITCODE"; exit $LASTEXITCODE }

# --- 3. сводка: последний каталог ppl_results -> экран + ab_dither_big_summary.csv
$latest = Get-ChildItem (Join-Path $RepoRoot "ppl_results") -Directory |
          Where-Object { $_.Name -match '^\d{8}_\d{6}$' } |
          Sort-Object Name | Select-Object -Last 1
$csvPath = Join-Path $latest.FullName "ppl_kv_results.csv"
if (-not (Test-Path $csvPath)) { Write-Host "[AB] CSV не найден: $csvPath"; exit 1 }
$rows = Import-Csv $csvPath

$sum = Join-Path $RepoRoot "ppl_results\ab_dither_big_summary.csv"
if (-not (Test-Path $sum)) {
    Set-Content -Path $sum -Value "label,when,pair,ppl,unc,tokens,pp_tps,status" -Encoding ASCII
}
$stamp = (Get-Date -Format "yyyy-MM-dd HH:mm")
Write-Host ""
Write-Host "[AB] ИТОГ label=$Label  eval=$(Split-Path $mixed -Leaf)  каталог=$($latest.Name)"
Write-Host ("{0,-14} {1,9} {2,8} {3,9} {4,9}  {5}" -f "пара","PPL","+-","токенов","pp t/s","статус")
Write-Host ("-" * 68)
foreach ($r in $rows) {
    Write-Host ("{0,-14} {1,9} {2,8} {3,9} {4,9}  {5}" -f "$($r.K)-$($r.V)", $r.PPL, $r.PPLUnc, $r.Tokens, $r.PpTps, $r.Status)
    Add-Content -Path $sum -Value ("{0},{1},{2},{3},{4},{5},{6},{7}" -f `
        $Label, $stamp, "$($r.K)-$($r.V)", $r.PPL, $r.PPLUnc, $r.Tokens, $r.PpTps, $r.Status) -Encoding ASCII
}
Write-Host ("-" * 68)
Write-Host "[AB] сводка (все прогоны со всех сборок): $sum"
Write-Host "[AB] следующие шаги: пересобрать нужную конфигурацию (build_turboq.ps1) и повторить этот скрипт с другим -Label"

# =============================================================================
# run_ppl_comparison.ps1
# Сравнение перплексии KV-кэша между разными типами квантования
# Запуск: .\run_ppl_comparison.ps1 -ModelPath "G:\Models\...\gguf" -EvalFile "wikitext-2-valid.txt"
# =============================================================================
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$ModelPath,

    [Parameter(Mandatory=$true)]
    [string]$EvalFile,

    [string]$BenchExe = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main\build-x64-windows-msvc-release\bin\llama-bench.exe",

    [int]$CtxSize = 4096,
    [int]$BatchSize = 512,
    [int]$NPredict = 128,
    [int]$NThreads = 7,

    [string]$OutCsv = "ppl_comparison_results.csv"
)

# Todas las combinaciones de KV cache quantization a probar
$combos = @(
    @{ K="q5_1"; V="q4_1"; Label="q5_1 / q4_1 (база)" },
    @{ K="turbo4"; V="turbo3"; Label="turbo4 / turbo3" },
    @{ K="turbo5"; V="turbo3"; Label="turbo5 / turbo3" },
    @{ K="turbo5"; V="turbo4"; Label="turbo5 / turbo4" },
    @{ K="turbo5"; V="turbo5"; Label="turbo5 / turbo5" }
)

if (-not (Test-Path $BenchExe)) {
    Write-Error "llama-bench.exe не найден: $BenchExe"
    Write-Host "Соберите его из upstream llama.cpp (examples/llama-bench) и укажите путь через -BenchExe"
    exit 1
}

if (-not (Test-Path $ModelPath)) {
    Write-Error "Модель не найдена: $ModelPath"
    exit 1
}

if (-not (Test-Path $EvalFile)) {
    Write-Error "Eval файл не найден: $EvalFile"
    exit 1
}

$results = @()
$header = "K_cache,V_cache,Label,PPL,Tokens_per_sec,Notes"

Write-Host "=== Перплексия KV-кэша: $ModelPath ===" -ForegroundColor Cyan
Write-Host "Контекст: $CtxSize, Батч: $BatchSize, Токенов: $NPredict, Потоков: $NThreads"
Write-Host ""

foreach ($combo in $combos) {
    $label = $combo.Label
    $k = $combo.K
    $v = $combo.V

    Write-Host "Тестирую: $label (-ctk $k -ctv $v) ..." -ForegroundColor Yellow

    $args = @(
        "-m", $ModelPath,
        "-c", $CtxSize,
        "-b", $BatchSize,
        "-np", "1",
        "-t", $NThreads,
        "-ctk", $k,
        "-ctv", $v,
        "-lm", "mmap",
        "-ctxcp", "0",
        "-cram", "0",
        "-ppl", $EvalFile,
        "-n", $NPredict,
        "-ngl", "99"
    )

    $start = Get-Date
    $output = & $BenchExe $args 2>&1
    $end = Get-Date
    $elapsed = ($end - $start).TotalSeconds

    # Парсим PPL из вывода llama-bench (формат может быть разным)
    $ppl = $null
    $tps = $null

    foreach ($line in $output) {
        if ($line -match "ppl\s*=\s*([\d.]+)") {
            $ppl = [double]$Matches[1]
        }
        if ($line -match "t/s\s*=\s*([\d.]+)|([\d.]+)\s*t/s|perplexity.*?([\d.]+)\s*t/s") {
            $tps = [double]$Matches[1]
        }
    }

    $result = [PSCustomObject]@{
        K       = $k
        V       = $v
        Label   = $label
        PPL     = $ppl
        TPS     = $tps
        Seconds = [math]::Round($elapsed, 1)
        Notes   = if ($ppl -eq $null) { "PARSE_ERROR" } else { "" }
    }
    $results += $result

    Write-Host "  PPL = $ppl, TPS = $tps, Время = $([math]::Round($elapsed,1)) с" -ForegroundColor Green
}

# Заголовок CSV
"K_cache,V_cache,Label,PPL,TPS,Seconds,Notes" | Out-File -FilePath $OutCsv -Encoding UTF8

foreach ($r in $results) {
    "$($r.K),$($r.V),$($r.Label),$($r.PPL),$($r.TPS),$($r.Seconds),$($r.Notes)" | Out-File -FilePath $OutCsv -Encoding UTF8 -Append
}

Write-Host ""
Write-Host "=== ИТОГОВАЯ ТАБЛИЦА ===" -ForegroundColor Cyan
Write-Host ("{0,-20} {1,-10} {2,-10} {3,-10} {4,-10}" -f "Комбинация", "PPL", "TPS", "Время", "Статус")
foreach ($r in $results) {
    $status = if ($r.PPL -eq $null) { "ОШИБКА" } else { "OK" }
    Write-Host ("{0,-20} {1,-10} {2,-10} {3,-10} {4,-10}" -f $r.Label, $r.PPL, $r.TPS, $r.Seconds, $status)
}

Write-Host ""
Write-Host "Результаты сохранены в: $OutCsv" -ForegroundColor Cyan

# Анализ: какой вариант лучше
$valid = $results | Where-Object { $_.PPL -ne $null }
if ($valid.Count -gt 1) {
    $best = $valid | Sort-Object PPL | Select-Object -First 1
    Write-Host "Лучшая перплексия: $($best.Label) (PPL = $($best.PPL))" -ForegroundColor Green
    $worst = $valid | Sort-Object PPL -Descending | Select-Object -First 1
    Write-Host "Худшая перплексия: $($worst.Label) (PPL = $($worst.PPL))" -ForegroundColor Red
}

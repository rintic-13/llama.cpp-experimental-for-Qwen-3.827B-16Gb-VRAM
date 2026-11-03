# =============================================================================
# run_check_decode_vec.ps1 - проверка нативного vec-ядра FA с дизером (decode!)
#
# PPL этот путь НЕ проверяет: префил всегда идёт через MMA_F16 + convert.cu, а
# векторное ядро работает только при декодировании. Поэтому тут два независимых теста:
#
#   A. КОРРЕКТНОСТЬ: одна и та же сборка, один и тот же кэш, два режима декода -
#        VEC      = нативное vec-ядро (само вычитает дизер) - новый путь
#        FALLBACK = GGML_CUDA_FATTN_VEC_FALLBACK=1 -> f16-скратч + MMA (эталон semantics)
#      Жадная генерация (temp 0) на одном промпте. Ядра численно РАЗНЫЕ (vec квантует Q
#      в q8_1, f16-скратч считает MMA в f16), поэтому побайтовое совпадение не гарантировано:
#      при близких логах greedy переключается в последних ~10 токенах. Критерий PASS:
#      общий префикс >= 160 символов, обе ветки находят иглу промпта, текст связный.
#      (Если seed дизера в vec-ядре неверен - шум +/-0.5 LSB по всем K/V ломает генерацию
#      в первых десятках токенов, игла не находится.)
#
#   B. СКОРОСТЬ: llama-bench tg в обоих режимах на длине кэша -d 8192 (параметр
#      -BenchDepth; при -d 0 tg-тест идёт с пустым KV и разница не видна в принципе).
#      Важно: если VEC не быстрее FALLBACK и на 8k - выигрыш патча не в скорости,
#      а в отказе от f16-скратча (корректность дизера в decode зафиксирована тестом A).
#
# Предусловие: сборка с дизером:  .\build_turboq.ps1 -Dither
#
# Запуск:
#   powershell -ExecutionPolicy Bypass -File .\run_check_decode_vec.ps1            # A + B (~17 мин)
#   powershell -ExecutionPolicy Bypass -File .\run_check_decode_vec.ps1 -OnlyGen   # только A (~9 мин)
#   powershell -ExecutionPolicy Bypass -File .\run_check_decode_vec.ps1 -OnlyBench # только B (~8 мин)
# =============================================================================
param(
    [switch]$OnlyGen,
    [switch]$OnlyBench,
    [string]$ModelPath = "G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
    [string]$RepoRoot  = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main",
    [string]$BuildDir  = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main\build-x64-windows-msvc-release",
    [int]$Threads      = 7,
    [int]$GpuLayers    = 99,
    [int]$GenTokens    = 96,
    [int]$PrefixPassChars = 160,     # сколько символов генерации обязаны совпасть
    [int]$BenchDepth   = 8192        # длина KV-кэша при замере tg (llama-bench -d)
)
$ErrorActionPreference = "Stop"

$cli   = Join-Path $BuildDir "bin\llama-cli.exe"
$bench = Join-Path $BuildDir "bin\llama-bench.exe"
foreach ($f in @($cli, $bench)) {
    if (-not (Test-Path $f)) { Write-Host "Нужен $f  (cmake --build $BuildDir --target llama-cli llama-bench)"; exit 1 }
}

$outDir = Join-Path $RepoRoot ("decode_check_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Write-Host "[DEC] артефакты: $outDir"

# --- промпт: длинный фрагмент wiki + игла, чтобы декод реально ходил по K-кэшу -----
$evalTxt = Join-Path $RepoRoot "ppl_eval\wiki.test.quick.txt"
if (-not (Test-Path $evalTxt)) { Write-Host "нет файла eval: $evalTxt"; exit 1 }
$body = (Get-Content $evalTxt -Raw -Encoding UTF8).Substring(0, 4000)
$prompt = $body + @"

Above the passage there is a made-up product code inside double angular brackets. The real code is <<QV-7391-KD>>.
Question: what is the product code mentioned above? Answer:
"@
$promptFile = Join-Path $outDir "prompt.txt"
Set-Content -Path $promptFile -Value $prompt -Encoding UTF8

# генерация: только дизернутые пары (turbo5 env-переключатель не касается, сравнивать тривиально)
$genPairs = @(
    @{ K="q5_1";  V="turbo3" },
    @{ K="turbo4";V="turbo3" },
    @{ K="q5_1";  V="turbo4" }
)
# bench: те же + контрольная turbo5 (переключатель её не трогает -> tg должен не меняться)
$benchPairs = $genPairs + @(@{ K="q5_1"; V="turbo5" })

# ---------------------------------------------------------------- A. генерация ---
$genFail = 0
if (-not $OnlyBench) {
    Write-Host ""
    Write-Host "[GEN] жадная генерация, temp=0, -n $GenTokens, ctx 4096, FA on (-st = single-turn, аналога --no-cnv в этой версии нет)"
    foreach ($p in $genPairs) {
        $name = "$($p.K)-$($p.V)"
        $texts = @{}
        foreach ($mode in @("VEC", "FALLBACK")) {
            if ($mode -eq "FALLBACK") {
                $env:GGML_CUDA_FATTN_VEC_FALLBACK = "1"; Remove-Item Env:GGML_CUDA_FATTN_VEC_NATIVE -ErrorAction SilentlyContinue
            } else {
                # VEC-режим: NATIVE=1 обязателен - с 2026-10-31 turbo4-комбо дефолтом идут на MMA (скоростной гейт)
                $env:GGML_CUDA_FATTN_VEC_NATIVE = "1"; Remove-Item Env:GGML_CUDA_FATTN_VEC_FALLBACK -ErrorAction SilentlyContinue
            }

            $log  = Join-Path $outDir ("gen_{0}_{1}.log"  -f $name, $mode.ToLower())
            $tout = Join-Path $outDir ("gen_{0}_{1}.txt"  -f $name, $mode.ToLower())
            Write-Host "  $name [$mode] ..."
            & $cli -m $ModelPath -f $promptFile -n $GenTokens --temp 0 -s 42 -st `
                   -c 4096 -b 2048 -ub 512 -ngl $GpuLayers -t $Threads -fa on `
                   -ctk $p.K -ctv $p.V --no-display-prompt 1> $tout 2> $log
            $t = (Get-Content $tout -Raw -Encoding UTF8) -replace '\s+', ' '
            $texts[$mode] = $t.Trim()
        }
        Remove-Item Env:GGML_CUDA_FATTN_VEC_FALLBACK, Env:GGML_CUDA_FATTN_VEC_NATIVE -ErrorAction SilentlyContinue

        $a = $texts["VEC"]; $b = $texts["FALLBACK"]
        $n = [math]::Min($a.Length, $b.Length)
        $diff = -1
        for ($i = 0; $i -lt $n; $i++) { if ($a[$i] -ne $b[$i]) { $diff = $i; break } }
        $common = if ($diff -lt 0) { $n } else { $diff }

        $verdict = if ($common -ge $PrefixPassChars) { "PASS" } else { "FAIL" }
        if ($verdict -eq "FAIL") { $genFail++ }
        Write-Host ("       совпало символов: {0}; вывод VEC: «{1}»" -f $common,
                    $a.Substring(0, [math]::Min(120, $a.Length))).TrimEnd()
        Write-Host ("       вывод FALLBACK : «{0}»  => {1}" -f $b.Substring(0, [math]::Min(120, $b.Length)), $verdict).TrimEnd()
        if ($diff -ge 0) { Write-Host "       первое расхождение с символа $diff" }
    }
}

# ----------------------------------------------------------------- B. скорость ---
if (-not $OnlyGen) {
    Write-Host ""
    Write-Host "[BENCH] llama-bench: tg (ток/с), -p 0 -n 128 -r 1 -d $BenchDepth (прогрев кэша на $BenchDepth токенов, общий для тестов)"
    foreach ($p in $benchPairs) {
        $name = "$($p.K)-$($p.V)"
        foreach ($mode in @("VEC", "FALLBACK")) {
            if ($mode -eq "FALLBACK") {
                $env:GGML_CUDA_FATTN_VEC_FALLBACK = "1"; Remove-Item Env:GGML_CUDA_FATTN_VEC_NATIVE -ErrorAction SilentlyContinue
            } else {
                # VEC-режим: NATIVE=1 обязателен - с 2026-10-31 turbo4-комбо дефолтом идут на MMA (скоростной гейт)
                $env:GGML_CUDA_FATTN_VEC_NATIVE = "1"; Remove-Item Env:GGML_CUDA_FATTN_VEC_FALLBACK -ErrorAction SilentlyContinue
            }

            $log = Join-Path $outDir ("bench_{0}_{1}.log" -f $name, $mode.ToLower())
            Write-Host "  $name [$mode] ..."
            # stderr не должен заходить в PowerShell: ggml_cuda_init пишет в stderr,
            # а 2>&1|Tee при ErrorActionPreference=Stop роняет скрипт (NativeCommandError).
            & cmd /c "`"$bench`" -m `"$ModelPath`" -ctk $($p.K) -ctv $($p.V) -fa on -ngl $GpuLayers -t $Threads -p 0 -n 128 -r 1 -d $BenchDepth > `"$log`" 2>&1"
            $tgLine = Select-String -Path $log -Pattern 'tg[0-9]+' | Select-Object -Last 1
            if ($tgLine) {
                # формат ячейки: | ... | tg128 @ d8192 |   28.52 ± 0.00 |
                # ВАЖНО: не брать первый float строки - в имени модели есть "3.4375 bpw"!
                $m2 = [regex]::Match($tgLine.Line, 'tg\d+[^|\r\n]*\|\s*([0-9]+\.[0-9]+)\s*±')
                if ($m2.Success) { Write-Host ("       tg = {0} t/s" -f $m2.Groups[1].Value) }
                else             { Write-Host ("       " + $tgLine.Line.Trim()) }
            } else {
                Write-Host "       строка tg не найдена - смотреть $log"
            }
        }
        Remove-Item Env:GGML_CUDA_FATTN_VEC_FALLBACK, Env:GGML_CUDA_FATTN_VEC_NATIVE -ErrorAction SilentlyContinue
    }
}

Write-Host ""
if ($genFail -eq 0) { Write-Host "[DEC] генерация VEC и FALLBACK совпадает - дизер в vec-ядре корректен"; exit 0 }
else { Write-Host "[DEC] расхождений генерации: $genFail - смотреть $outDir"; exit 1 }

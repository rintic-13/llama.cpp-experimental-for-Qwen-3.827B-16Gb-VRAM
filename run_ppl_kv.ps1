# =============================================================================
# ARCHIVED 2026-11-02 (partial): turbo4q_0 retired from the FA path (SS12); its rows below
# now fall back to f16-scratch on current builds - historical quality record only.
# run_ppl_kv.ps1
# Перплексия KV-кэша (llama-perplexity, форк с TurboQuant).
#
# Проверяет качество квантованных типов K/V кэша на Wikitext-2:
#   f16 (эталон), q5_1, q4_1, turbo5, turbo4, turbo3
#
# Примеры запуска:
#   powershell -ExecutionPolicy Bypass -File .\run_ppl_kv.ps1                # полный прогон (все пары)
#   powershell -ExecutionPolicy Bypass -File .\run_ppl_kv.ps1 -Quick         # быстрый прогон (200 KB текста)
#   powershell -ExecutionPolicy Bypass -File .\run_ppl_kv.ps1 -Pair q5_1-q5_1,turbo5-turbo5
#   powershell -ExecutionPolicy Bypass -File .\run_ppl_kv.ps1 -Ctx 8192      # длинный контекст (ошибки KV сильнее)
#
# ВАЖНО:
#  * turbo-типы требуют FlashAttention и CUDA. Ядро FA должно быть скомпилировано
#    под конкретную пару K-V (см. GGML_CUDA_FA_QUANTS в CMakeCache.txt).
#    Собраны: f16-f16, q4_0-q4_0, q8_0-q8_0, q5_1-q5_1, q5_1-q4_1, q5_1-q4_0,
#             q4_1-q4_1, q5_1-turbo3, q5_1-turbo4, q5_1-turbo5,
#             turbo5-turbo5, turbo5-turbo4, turbo5-turbo3, turbo4-turbo3,
#             q5_1-turbo3q, q5_1-turbo4q, turbo4q-turbo4, turbo4q-turbo4q, turbo4-turbo3q.
#  * turbo4-turbo4 и turbo3-turbo3 НЕ собраны: без пересборки ggml-cuda
#    (cmake -DGGML_CUDA_FA_QUANTS="...;turbo4_0-turbo4_0;turbo3_0-turbo3_0" + полная
#    пересборка, ~15-20 мин) такая пара молча откатится на f16-f16 (замер будет
#    недостоверным) - скрипт такие пары пропускает с сообщением.
# =============================================================================
[CmdletBinding()]
param(
    [string]$ModelPath = "G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",

    [string]$RepoRoot   = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main",
    [string]$BuildDir   = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main\build-x64-windows-msvc-release",
    [string]$Exe,                       # по умолчанию: $BuildDir\bin\llama-perplexity.exe

    [string]$EvalFile,                  # по умолчанию: ppl_eval\wikitext-2-raw\wiki.test.raw; -Quick -> quick-подмножество
    [switch]$Quick,                     # быстрая проверка механизма: 200 KB wiki.test.quick.txt (~1 мин на пару)

    [int]$Ctx       = 4096,
    [int]$Batch     = 4096,
    [int]$UBatch    = 512,
    [int]$Threads   = 7,
    [int]$GpuLayers = 99,
    [string]$FaMode = "on",             # 'on' обязательно для turbo; для честности сравнения FA включен и для обычных пар

    # список пар "K-V" (типы из: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1, turbo3, turbo4, turbo5)
    [string[]]$Pair = @(
        "f16-f16",       # эталон: без квантования кэша
        "q5_1-q5_1",
        "q5_1-q4_1",
        "q4_1-q4_1",
        "q5_1-turbo5",
        "q5_1-turbo4",
        "q5_1-turbo3",
        "turbo5-turbo5",
        "turbo5-turbo4",
        "turbo5-turbo3",
        "turbo4-turbo3",
        "q5_1-turbo3q",   # Q_prod V 3.25 bpw (2-bit + QJL)
        "q5_1-turbo4q",   # Q_prod V 4.25 bpw (3-bit + QJL)
        "turbo4q-turbo4", # Q_prod K 4.25 bpw + MSE V
        "turbo4q-turbo4q",
        "turbo4-turbo3q"
    ),
    [string]$Baseline = "f16-f16",      # от какой пары считать разницу PPL

    [string[]]$ExtraArg = @(),          # доп. флаги форка, например -ExtraArg -lm,mmap,-ctxcp,0,-cram,0
    [int]$TimeoutMin    = 90,           # таймаут на одну пару, мин
    [switch]$RunMissingFa,              # запускать пары без скомпилированного FA-ядра (результат = откат на f16!)
    [string]$OutRoot    = "",           # по умолчанию: ppl_results\<ГГГГММДД_ЧЧММСС>
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# ---------- базовые пути -----------------------------------------------
if (-not $Exe) { $Exe = Join-Path $BuildDir "bin\llama-perplexity.exe" }
if (-not $EvalFile) {
    if ($Quick) { $EvalFile = Join-Path $RepoRoot "ppl_eval\wiki.test.quick.txt" }
    else        { $EvalFile = Join-Path $RepoRoot "ppl_eval\wikitext-2-raw\wiki.test.raw" }
}

if (-not (Test-Path $Exe))      { Write-Error "Нет llama-perplexity.exe: $Exe (соберите: cmake --build `"$BuildDir`" --target llama-perplexity)" }
if (-not (Test-Path $ModelPath)){ Write-Error "Нет модели: $ModelPath" }
if (-not (Test-Path $EvalFile)) { Write-Error "Нет датасета: $EvalFile (см. tools/perplexity/README.md: скачайте wikitext-2-raw-v1.zip и распакуйте)" }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not $OutRoot) { $OutRoot = Join-Path $RepoRoot "ppl_results\$stamp" }
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$OutCsv      = Join-Path $OutRoot "ppl_kv_results.csv"
$summaryFile = Join-Path $OutRoot "summary.txt"

# powershell -File не разбивает "a,b" на массив - разрезаем и по запятой
$Pair = @($Pair | ForEach-Object { $_ -split "," } | Where-Object { $_.Trim() })
$ExtraArg = @($ExtraArg | ForEach-Object { if ($_ -match ",") { $_ -split "," } else { $_ } } | Where-Object { -not ($_ -eq "") })

function Info ([string]$msg, [string]$color = "Gray") { Write-Host ("[PPL] {0}" -f $msg) -ForegroundColor $color }

# формат чисел без локали (в ru-RU запятая-десятичный разделитель ломает CSV)
function FmtNum ([double]$v, [string]$fmt, [string]$empty = "") {
    if ([double]::IsNaN($v)) { return $empty }
    return $v.ToString($fmt, [Globalization.CultureInfo]::InvariantCulture)
}

Info ("Модель: {0}" -f (Split-Path $ModelPath -Leaf)) "Cyan"
Info ("Eval:   {0}   ctx={1} batch={2} ub={3} FA={4}" -f $EvalFile, $Ctx, $Batch, $UBatch, $FaMode) "Cyan"
Info ("Результаты: {0}" -f $OutRoot) "Cyan"

# ---------- какие FA-ядра собраны (из CMakeCache) ------------------------
$faSet = @{}
$cmakeCache = Join-Path $BuildDir "CMakeCache.txt"
if (Test-Path $cmakeCache) {
    $line = (Select-String -Path $cmakeCache -Pattern "^GGML_CUDA_FA_QUANTS:STRING=" | Select-Object -First 1).Line
    if ($line) {
        foreach ($combo in (($line -replace "^GGML_CUDA_FA_QUANTS:STRING=", "") -split "[;,]")) {
            $combo = $combo.Trim()
            if (-not $combo) { continue }
            $ab = $combo -split "-"
            if ($ab.Count -ne 2) { continue }
            foreach ($i in 0, 1) {
                if ($ab[$i] -match "^turbo([0-9]+q?)_0$") { $ab[$i] = "turbo$($Matches[1])" }
            }
            $faSet[($ab[0] + "-" + $ab[1])] = $true   # ядро собрано именно под этот порядок K-V
        }
    }
}
if ($faSet.Count -gt 0) {
    Info ("FA-ядра: {0}" -f (($faSet.Keys | Sort-Object) -join ", ")) "DarkCyan"
}

# ---------- парсинг лога ------------------------------------------------
function New-EmptyResult ([string]$K, [string]$V, [string]$LogFile, [double]$Seconds, [int]$ExitCode) {
    [PSCustomObject]@{
        K=$K; V=$V; PPL=[double]::NaN; PPLUnc=[double]::NaN
        Tokens=0; PpTps=[double]::NaN; CtxKvMiB=[double]::NaN; ComputeMiB=[double]::NaN; ModelMiB=[double]::NaN
        Seconds=[math]::Round($Seconds, 1); ExitCode=$ExitCode
        Status="UNKNOWN"; Notes=""; LogFile=$LogFile
    }
}

function ConvertFrom-PplLog {
    param([string[]]$Log, [string]$K, [string]$V, [string]$LogFile, [double]$Seconds, [int]$ExitCode)

    $r = New-EmptyResult $K $V $LogFile $Seconds $ExitCode
    $warns = New-Object System.Collections.Generic.List[string]

    foreach ($line in $Log) {
        if ($line -match "Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)") {
            if ([double]::IsNaN($r.PPL)) { $r.PPL = [double]$Matches[1]; $r.PPLUnc = [double]$Matches[2] }
        }
        elseif ($line -match "prompt eval time\s*=\s*([0-9.]+) ms /\s*(\d+) tokens \(\s*[0-9.]+ ms per token,\s*([0-9.]+) tokens per second") {
            $r.Tokens = [int]$Matches[2]; $r.PpTps = [double]$Matches[3]
        }
        elseif ($line -match "no FlashAttention vector kernel compiled") { $warns.Add("FA_FALLBACK_f16") }
        elseif ($line -match "failed to fit params to free device memory") { $warns.Add("VRAM_FIT") }
        elseif ($line -match "out of memory|OutOfMemory|cudaMalloc failed|CUDA error|failed to allocate") { $warns.Add("OOM_CUDA") }
        elseif ($line -match "failed to load model|unable to load model|unknown .*architecture|Unsupported cache type|invalid argument") { $warns.Add("LOAD_ARG_ERROR") }
        elseif ($line -match "common_memory_breakdown_print: \|\s*- CUDA0.*\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*\(\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*\)\s*\+\s*(\d+)\s*\|") {
            $r.ModelMiB = [double]$Matches[4]; $r.CtxKvMiB = [double]$Matches[5]; $r.ComputeMiB = [double]$Matches[6]
        }
    }

    if ($warns.Count -gt 0) { $r.Notes = (@($warns) | Select-Object -Unique) -join "," }
    if ($r.ExitCode -ne 0)               { $r.Status = "CRASH" }
    elseif ($r.Notes -match "OOM_CUDA|LOAD_ARG_ERROR|VRAM_FIT") { $r.Status = "ERROR" }
    elseif ([double]::IsNaN($r.PPL))     { $r.Status = "NO_PPL" }
    elseif ($r.Notes -match "FA_FALLBACK_f16") { $r.Status = "FA_FALLBACK_F16_INVALID" }
    else                                 { $r.Status = "OK" }
    return $r
}

# ---------- прогон одной пары ------------------------------------------
function Invoke-PplCombo {
    param([string]$K, [string]$V, [string]$LogOut, [string]$LogErr)

    Info ("Запуск: -ctk {0} -ctv {1} ..." -f $K, $V) "Yellow"

    $argList = @(
        "-m", $ModelPath,
        "-f", $EvalFile,
        "-c", "$Ctx",
        "-b", "$Batch",
        "-ub", "$UBatch",
        "-ngl", "$GpuLayers",
        "-fa", $FaMode,
        "-ctk", $K,
        "-ctv", $V,
        "-t", "$Threads",
        "-s", "42"
    )
    if ($ExtraArg.Count -gt 0) { $argList += $ExtraArg }

    if ($DryRun) {
        Info ("DRY-RUN: llama-perplexity.exe " + ((($argList | ForEach-Object { '"' + $_ + '"' })) -join " ")) "Cyan"
        $r = New-EmptyResult $K $V "" 0 0; $r.Status = "DRY_RUN"; return $r
    }

    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $Exe -ArgumentList $argList `
             -RedirectStandardOutput $LogOut -RedirectStandardError $LogErr `
             -WorkingDirectory $RepoRoot -NoNewWindow -PassThru
    if (-not $p.WaitForExit($TimeoutMin * 60000)) {
        try { $p.Kill() } catch {}
        Info ("Таймаут {0} мин, процесс остановлен" -f $TimeoutMin) "Red"
        $r = New-EmptyResult $K $V (Split-Path $LogOut -Leaf) $sw.Elapsed.TotalSeconds -9
        $r.Status = "TIMEOUT"; return $r
    }
    $sw.Stop()

    $log = @()
    foreach ($f in @($LogOut, $LogErr)) {
        if ((Test-Path $f) -and ((Get-Item $f).Length -gt 0)) { $log += @(Get-Content $f) }
    }
    $res = ConvertFrom-PplLog -Log $log -K $K -V $V -LogFile (Split-Path $LogOut -Leaf) -Seconds $sw.Elapsed.TotalSeconds -ExitCode $p.ExitCode

    $pplStr = if ([double]::IsNaN($res.PPL)) { "-" } else { $res.PPL }
    $col    = if ($res.Status -eq "OK") { "Green" } else { "Red" }
    Info ("  PPL={0}  pp={1} t/s  KV={2} MiB  время={3} c  статус={4} {5}" -f `
          $pplStr, $res.PpTps, $res.CtxKvMiB, $res.Seconds, $res.Status, $res.Notes) $col
    return $res
}

# ---------- основной цикл ----------------------------------------------
$results = @()
foreach ($pairRaw in $Pair) {
    $pair = $pairRaw.Trim()
    $ab = $pair -split "-"
    if ($ab.Count -ne 2) { Info ("Неверный формат пары '$pair' (нужно K-V), пропущена") "Red"; continue }
    $k = $ab[0].Trim(); $v = $ab[1].Trim()
    $label = "$k-$v"

    if ($faSet.Count -gt 0 -and -not $faSet.ContainsKey($label)) {
        if (-not $RunMissingFa) {
            Info ("ПРОПУСК $label : FA-ядро не собрано (GGML_CUDA_FA_QUANTS). Без пересборки прогон откатится на f16-f16.") "Red"
            $r = New-EmptyResult $k $v "" 0 0
            $r.Status = "SKIPPED_NO_FA_KERNEL"; $r.Notes = "add <$($k)_0-$($v)_0> to GGML_CUDA_FA_QUANTS"
            $results += $r
            continue
        }
        Info ("ВНИМАНИЕ: FA-ядро для $label не собрано, замер откатится на f16-f16!") "Red"
    }

    $logOut = Join-Path $OutRoot ("run_{0}_stdout.log" -f $label)
    $logErr = Join-Path $OutRoot ("run_{0}_stderr.log" -f $label)
    $res = Invoke-PplCombo -K $k -V $v -LogOut $logOut -LogErr $logErr
    $results += $res

    # сохраняем после каждой пары - можно прервать и не потерять
    $rows = @()
    $rows += "K,V,PPL,PPLUnc,Tokens,PpTps,CtxKvMiB,ComputeMiB,ModelMiB,Seconds,ExitCode,Status,Notes,LogFile"
    foreach ($r in $results) {
        $rows += ('"{0}","{1}",{2},{3},{4},{5},{6},{7},{8},{9},{10},"{11}","{12}","{13}"' -f `
            $r.K, $r.V,
            (FmtNum $r.PPL "0.000000"), (FmtNum $r.PPLUnc "0.000000"),
            (FmtNum ([double]$r.Tokens) "0"), (FmtNum $r.PpTps "0.00"),
            (FmtNum $r.CtxKvMiB "0.0"), (FmtNum $r.ComputeMiB "0.0"), (FmtNum $r.ModelMiB "0.0"),
            (FmtNum ([double]$r.Seconds) "0.0"),
            $r.ExitCode, $r.Status, ($r.Notes -replace '"', "'"), $r.LogFile)
    }
    $rows | Out-File -FilePath $OutCsv -Encoding UTF8
}

# ---------- сводка ------------------------------------------------------
$measured = @($results | Where-Object { $_.Status -eq "OK" })
$basePPL  = [double]::NaN
if ($measured.Count -gt 0) {
    $b = $measured | Where-Object { "$($_.K)-$($_.V)" -eq $Baseline } | Select-Object -First 1
    if ($b) { $basePPL = $b.PPL }
}

$summary = @()
$summary += "Перплексия KV-кэша"
$summary += ("  модель:  {0}" -f (Split-Path $ModelPath -Leaf))
$summary += ("  eval:    {0}" -f $EvalFile)
$summary += ("  режим:   -c {0}  -b {1}  -ub {2}  -fa {3}  -ngl {4}  -t {5}   ({6})" -f $Ctx, $Batch, $UBatch, $FaMode, $GpuLayers, $Threads, (Get-Date -Format s))
$summary += ""
$hdr = "пара-KV             |    PPL  |     +/- | dPPL,%   | pp t/s | KV MiB | сек  | статус"
$summary += $hdr
$summary += ("=" * $hdr.Length)

$okResults = @($results | Where-Object { $_.Status -ne "SKIPPED_NO_FA_KERNEL" -and $_.Status -ne "DRY_RUN" })
$best = $null; $worst = $null
foreach ($r in ($okResults | Sort-Object { if ([double]::IsNaN($_.PPL)) { 9e9 } else { $_.PPL } })) {
    $inv    = [Globalization.CultureInfo]::InvariantCulture
    $pplStr = FmtNum $r.PPL "0.0000"
    $uncStr = FmtNum $r.PPLUnc "0.0000"
    $dStr   = "-"
    if (-not [double]::IsNaN($r.PPL) -and -not [double]::IsNaN($basePPL)) {
        $dStr = (100.0 * ($r.PPL / $basePPL - 1.0)).ToString("+0.00;-0.00", $inv)
    }
    $tpsStr = FmtNum $r.PpTps "0"
    $kvStr  = FmtNum $r.CtxKvMiB "0"
    $summary += ("{0,-18} | {1,7} | {2,7} | {3,7}  | {4,6} | {5,6} | {6,4} | {7} {8}" -f `
        "$($r.K)-$($r.V)", $pplStr, $uncStr, $dStr, $tpsStr, $kvStr, $r.Seconds, $r.Status, $r.Notes)
    if ($r.Status -eq "OK") {
        if (-not $best -or $r.PPL -lt $best.PPL)   { $best  = $r }
        if (-not $worst -or $r.PPL -gt $worst.PPL) { $worst = $r }
    }
}
$skipped = @($results | Where-Object { $_.Status -eq "SKIPPED_NO_FA_KERNEL" })
if ($skipped.Count -gt 0) {
    $summary += ""
    $summary += "Пропущены (FA-ядро не собрано, нужна пересборка ggml-cuda):"
    foreach ($s in $skipped) { $summary += ("  {0}-{1}: {2}" -f $s.K, $s.V, $s.Notes) }
}
$summary += ""
if ($best)  { $summary += ("Лучшая:  {0}-{1}  PPL = {2}"  -f $best.K,  $best.V,  (FmtNum $best.PPL "0.0000")) }
if ($worst) { $summary += ("Худшая:  {0}-{1}  PPL = {2}"  -f $worst.K, $worst.V, (FmtNum $worst.PPL "0.0000")) }
if (-not [double]::IsNaN($basePPL)) { $summary += ("Эталон {0}: PPL = {1}" -f ($Baseline -replace "-", " / "), (FmtNum $basePPL "0.0000")) }
$summary += ""
$summary += ("CSV:     {0}" -f $OutCsv)
$summary += ("Логи:    {0}" -f $OutRoot)

$summary | Out-File -FilePath $summaryFile -Encoding UTF8
Info "Итоговая таблица:" "Cyan"
$summary | Where-Object { $_ } | ForEach-Object { Write-Host $_ }
Info ("Готово. Результаты: {0}" -f $OutRoot) "Green"

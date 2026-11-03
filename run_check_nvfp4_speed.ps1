# =============================================================================
# run_check_nvfp4_speed.ps1 - proof harness for the nvfp4 speed question
#
# Answers three separate things in one build (all runs are llama-bench, cheap):
#
#  A) ROUTING PROOF (GGML_FATTN_DEBUG=1 one-shot log in fattn.cu):
#     which FA kernel each pair really runs, and whether the vec template
#     instance exists (vec_compiled=1) or the pair falls onto the f16 scratch
#     MMA path (kernel=vec + "no FlashAttention vector kernel" warning =>
#     K/V dequantized to f16 scratch, NOT fused).
#
#  B) REGIME PROOF (depth sweep):
#     decode tg for f16-f16 (16 bpw KV) vs nvfp4-nvfp4 (9 bpw KV) at several KV
#     depths. If attention bandwidth were the bottleneck, nvfp4 would pull
#     further ahead as depth grows; if the run is weights-streaming bound,
#     both curves stay flat and nearly equal.
#
#  C) GATE A/B IN THE SAME BINARY (dither build has both switches):
#     GGML_CUDA_FATTN_VEC_NATIVE=1   -> force native vec kernels for turbo combos
#     GGML_CUDA_FATTN_VEC_FALLBACK=1 -> force f16 scratch MMA for turbo combos
#     (nvfp4 is deliberately NOT gated - both runs must stay vec for nvfp4.)
#
# Usage (user runs; ~6-8 min total):
#   powershell -ExecutionPolicy Bypass -File .\run_check_nvfp4_speed.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_check_nvfp4_speed.ps1 -Sections A,B
#   powershell -ExecutionPolicy Bypass -File .\run_check_nvfp4_speed.ps1 -Model "G:\Models\..." -Depths 2048,8192,32768
# =============================================================================
param(
    [string]$Model = "G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
    [string]$RepoRoot = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main",
    [string]$Sections = "A,B,C",
    [int[]]$Depths = @(2048, 8192, 32768),
    [int]$Ngl = 99
)
$ErrorActionPreference = "Stop"

$bench = Join-Path $RepoRoot "build-x64-windows-msvc-release\bin\llama-bench.exe"
if (-not (Test-Path $bench)) { Write-Error "no llama-bench.exe: $bench" }
if (-not (Test-Path $Model)) { Write-Error "no model: $Model" }

$sec = $Sections -split ","
$rawlog = Join-Path $RepoRoot ("bench_check_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

# run one llama-bench invocation, return [pscustomobject] with tg + captured stderr-ish lines
function Invoke-Bench {
    param([string]$KvK, [string]$KvV, [int]$Depth, [int]$NPre = 0, [int]$NNeg = 128,
          [string]$EnvName = "", [int]$Reps = 1)
    $env:GGML_FATTN_DEBUG = "1"   # one-shot routing log (each bench call is a fresh process)
    $argsList = @("-m", $Model, "-fa", "on", "-ngl", $Ngl,
                  "-p", $NPre, "-n", $NNeg, "-r", $Reps, "-d", $Depth,
                  "-ctk", $KvK, "-ctv", $KvV)
    if ($EnvName -eq "native")   { $env:GGML_CUDA_FATTN_VEC_NATIVE = "1";   $env:GGML_CUDA_FATTN_VEC_FALLBACK = "" }
    if ($EnvName -eq "fallback") { $env:GGML_CUDA_FATTN_VEC_FALLBACK = "1";  $env:GGML_CUDA_FATTN_VEC_NATIVE = "" }
    if ($EnvName -eq "")         { $env:GGML_CUDA_FATTN_VEC_NATIVE = "";     $env:GGML_CUDA_FATTN_VEC_FALLBACK = "" }

    # llama-bench writes info/warnings/FATTN_DEBUG lines to stderr; under EAP="Stop" the first
    # redirected stderr line aborts with NativeCommandError. Relax it just for the capture.
    $__eap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = & $bench @argsList 2>&1 | ForEach-Object { $_.ToString() }
    } finally {
        $ErrorActionPreference = $__eap
    }
    if ($EnvName -eq "native" -or $EnvName -eq "fallback") {
        $env:GGML_CUDA_FATTN_VEC_NATIVE = ""; $env:GGML_CUDA_FATTN_VEC_FALLBACK = ""
    }
    $env:GGML_FATTN_DEBUG = ""
    $tg = $null
    foreach ($l in $out) {
        if ($l -match 'tg\d+(?:\s*@\s*d\d+)?\s*\|\s*([0-9]+(?:\.[0-9]+)?)') { $tg = [double]$Matches[1] }
    }
    $dbg = @($out | Where-Object { $_ -match "FATTN_DEBUG|no FlashAttention vector kernel" })
    # append the raw output to a per-run log file so failures are debuggable offline
    Add-Content -LiteralPath $rawlog -Value "== ctk=$KvK ctv=$KvV d=$Depth p=$NPre n=$NNeg reps=$Reps env=$EnvName exit=$LASTEXITCODE" -Encoding UTF8
    Add-Content -LiteralPath $rawlog -Value $out -Encoding UTF8
    if ($null -eq $tg) {
        # parse failure => show the actual bench output so the cause is visible
        Write-Host "    [!] tg not parsed; bench exit=$LASTEXITCODE, raw output:" -ForegroundColor Yellow
        $out | Select-Object -First 60 | ForEach-Object { Write-Host ("      | " + $_) -ForegroundColor DarkGray }
    }
    return [pscustomobject]@{ tg = $tg; dbg = $dbg }
}

Write-Host "nvfp4 speed proof harness  ($bench)"
Write-Host "model: $Model"
Write-Host "raw bench log: $rawlog"
Write-Host ""

# ---------------------------------------------------------------- A) routing
if ($sec -contains "A") {
    Write-Host "== A) routing proof (GGML_FATTN_DEBUG=1, tiny runs) ==" -ForegroundColor Cyan
    $pairsA = @(
        @("nvfp4", "nvfp4"),
        @("q5_1",  "nvfp4"),
        @("turbo4","turbo3"),
        @("q5_1",  "turbo5"),
        @("f16",   "f16")
    )
    $env:GGML_FATTN_DEBUG = "1"
    foreach ($p in $pairsA) {
        $r = Invoke-Bench -KvK $p[0] -KvV $p[1] -Depth 1024 -NPre 8 -NNeg 8
        Write-Host ("--- {0}-{1}  (tg @d1024 = {2})" -f $p[0], $p[1], $(if ($r.tg) { $r.tg } else { "n/a" }))
        foreach ($l in $r.dbg) { Write-Host ("    " + ($l -replace '^\s*\d+\.\d+\.\d+\.\d+\s+\w\s+', '')) }
        if (-not $r.dbg -or $r.dbg.Count -eq 0) { Write-Host "    (no FATTN_DEBUG lines? check GGML_FATTN_DEBUG)" -ForegroundColor Yellow }
    }
    $env:GGML_FATTN_DEBUG = ""
    Write-Host ""
}

# ---------------------------------------------------------------- B) depth
if ($sec -contains "B") {
    Write-Host "== B) regime proof: tg vs KV depth (weights-bound => flat, attention-bound => grows) ==" -ForegroundColor Cyan
    $hdr = ("{0,-14}" -f "pair") + (($Depths | ForEach-Object { "{0,10}" -f "d$_" }) -join "")
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    Write-Host $hdr
    foreach ($p in @(@("nvfp4","nvfp4"), @("f16","f16"), @("q5_1","nvfp4"))) {
        $line = @()
        foreach ($d in $Depths) {
            $r = Invoke-Bench -KvK $p[0] -KvV $p[1] -Depth $d
            $line += $(if ($r.tg) { "{0,10}" -f $r.tg.ToString("F2", $inv) } else { "       OOM" })
        }
        Write-Host ("{0,-14} {1}" -f "$($p[0])-$($p[1])", ($line -join " "))
    }
    Write-Host ""
    Write-Host "expected traffic/token: weights 11.28 GiB = 12.11 GB; KV nvfp4-nvfp4 ~0.23 GB @8k;"
    Write-Host "f16-f16 KV ~4x larger. If nvfp4's lead widens strongly with depth, KV bandwidth matters."
    Write-Host ""
}

# ---------------------------------------------------------------- C) gate A/B
if ($sec -contains "C") {
    Write-Host "== C) same-binary A/B of the vec/MMA gate (turbo4-turbo3) @d8192 ==" -ForegroundColor Cyan
    foreach ($mode in @("", "fallback", "native")) {
        $r = Invoke-Bench -KvK "turbo4" -KvV "turbo3" -Depth 8192 -EnvName $mode
        $tag = if ($mode -eq "") { "default (gate)" } elseif ($mode -eq "native") { "forced native vec" } else { "forced f16 scratch MMA" }
        $inv = [System.Globalization.CultureInfo]::InvariantCulture
        Write-Host ("  turbo4-turbo3 {0,-22} tg = {1}" -f $tag, $(if ($r.tg) { $r.tg.ToString("F2", $inv) } else { "n/a" }))
        foreach ($l in $r.dbg) { Write-Host ("      " + ($l -replace '^\s*\d+\.\d+\.\d+\.\d+\s+\w\s+', '')) }
    }
    $r = Invoke-Bench -KvK "nvfp4" -KvV "nvfp4" -Depth 8192 -EnvName "fallback"
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    Write-Host ("  nvfp4-nvfp4     forced f16 scratch MMA (FALLBACK now covers nvfp4) tg = {0}" -f $(if ($r.tg) { $r.tg.ToString("F2", $inv) } else { "n/a" }))
    foreach ($l in $r.dbg) { Write-Host ("      " + ($l -replace '^\s*\d+\.\d+\.\d+\.\d+\s+\w\s+', '')) }
    Write-Host ""
}

Write-Host "done."

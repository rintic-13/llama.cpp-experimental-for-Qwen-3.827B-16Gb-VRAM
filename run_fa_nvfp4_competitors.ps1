# =============================================================================
# run_fa_nvfp4_competitors.ps1 - v3 gate: memory-comparable KV competitors only.
# Replaces run_fa_f16_v2_check.ps1 rows: q8_1/f16 KV are NOT competitors at equal
# memory and are removed from the competitor table (f16 stays ONLY as the PPL
# gold anchor at c4096 - at d32768 f16 KV = 8 GiB and cannot load on 16 GB anyway).
#
# Competitors (stored KV bytes/coord):
#   nvfp4-dot    GGML_TYPE_NVFP4 + production vec path (integer q8_1-style dot;
#                NOT "decode->f16->MMA" - no f16 intermediate exists there)  0.5625 B/c
#   nvfp4-fused  GGML_TYPE_NVFP4 + GGML_CUDA_FATTN_VEC_KQ_MMA_F16=1 (fused f16 MMA) 0.5625 B/c
#   q4_1-dot     GGML_TYPE_Q4_1  + production vec path                        0.6250 B/c  = SPEED ETALON
#   fp16-gold    reference anchor for PPL only                                2.0 B/c
#   (turbo4 rows removed per project decision: its speed/accuracy data already measured in prior tests)
#
# Numerics requirement, ALREADY PROVEN (tools/fa_fused_v12_diff.cu): the fused kernel feeds
# the MMA bitwise-identical f16 values to the unfused decode->f16->MMA kernel (0/128 score
# mismatches; both == scalar exact-dequantize reference at f32 rounding). Any fused-vs-unfused
# PPL delta is accumulator/GPU-split ordering noise only. Model-level unfused reference PPL
# was 7.1417 (v1 smem build, with a grid-split confound) vs fused 7.1607 at 4096 tokens.
#
# Gates:
#   speed:   nvfp4-fused tg must beat nvfp4-dot (post V-PRMT-fix smoke: +3.0% @8K, +7.0% @32K);
#            vs q4_1 etalon: 31.95 vs 31.97 (-0.1%) @8K, 27.50 vs 28.01 (-1.8%) @32K
#   quality: |PPL(nvfp4-fused) - PPL(fp16-gold)| <= -GoldTol  (default 0.05)
#   record:  q4_1 row vs gold = same-memory-class quality/speed reference
#
#   powershell -ExecutionPolicy Bypass -File .\run_fa_nvfp4_competitors.ps1            # T bench only
#   powershell -ExecutionPolicy Bypass -File .\run_fa_nvfp4_competitors.ps1 -Ppl       # + PPL quad
#   powershell -ExecutionPolicy Bypass -File .\run_fa_nvfp4_competitors.ps1 -Ppl -Chunks 4 -Full
# =============================================================================
param(
    [string]$ModelPath = "G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
    [string]$RepoRoot  = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main",
    [string]$BuildDir  = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main\build-x64-windows-msvc-release",
    [switch]$Ppl,
    [switch]$Full,
    [int]$Chunks   = 1,
    [double]$GoldTol = 0.05,
    [int[]]$Depths = @(8192, 32768),
    [int]$Reps     = 2
)
$ErrorActionPreference = "Continue"
$inv  = [System.Globalization.CultureInfo]::InvariantCulture
$pplExe = Join-Path $BuildDir "bin\llama-perplexity.exe"
$bench= Join-Path $BuildDir "bin\llama-bench.exe"
$eval = if ($Full) { Join-Path $RepoRoot "ppl_eval\wikitext-2-raw\wiki.test.raw" }
                 else { Join-Path $RepoRoot "ppl_eval\wiki.test.quick.txt" }
$outDir = Join-Path $RepoRoot ("fa_comp_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function Set-Flags { param($f16, $native)
    # NOTE: the fused f16-MMA KQ path is DEFAULT-ON in the DLL since 2026-11-02 (opt-out with "0"),
    # so the dot rows must set "0" explicitly, not "" (PowerShell deletes the variable on "").
    if ($f16) { $env:GGML_CUDA_FATTN_VEC_KQ_MMA_F16 = "1" } else { $env:GGML_CUDA_FATTN_VEC_KQ_MMA_F16 = "0" }
    $env:GGML_CUDA_FATTN_VEC_KQ_MMA = ""
    if ($native) { $env:GGML_CUDA_FATTN_VEC_NATIVE = "1" } else { $env:GGML_CUDA_FATTN_VEC_NATIVE = "" }
}
function Run-Capture { param($exe, [string[]]$argv, [string]$log)
    Write-Host ("  >>> running {0}  (log: {1})" -f (Split-Path $exe -Leaf), $log) -ForegroundColor DarkGray
    $o = & $exe @argv 2>&1 | Tee-Object -LiteralPath $log
    return $o
}

# tag, ctk, ctv, fused-flag, native-flag, max depth for bench
# turbo4 excluded per project decision (its speed/accuracy data already measured elsewhere);
# q4_1 is the speed etalon for nvfp4-fused.
$tRuns = @(
    @{ tag="nvfp4-dot "; ctk="nvfp4"; ctv="nvfp4"; f16=$false },
    @{ tag="nvfp4-fused"; ctk="nvfp4"; ctv="nvfp4"; f16=$true  },
    @{ tag="q4_1-dot  "; ctk="q4_1";  ctv="q4_1";  f16=$false }
)

Write-Host "nvfp4 KV competitor gate  |  model: $ModelPath"
Write-Host "memory per KV coord: nvfp4 0.5625B  q4_1 0.6250B  |  f16 fallback 2.0B"
Write-Host "KV per token (64 layers, 4 kv heads, D256): nvfp4 72KiB  q4_1 80KiB  f16 256KiB -> @32768: 2.25/2.5/8.0 GiB"
Write-Host "logs -> $outDir"
Write-Host ""

# ---------------------------------------------------------------- T) decode tg
Write-Host "== T) decode tg A/B ==" -ForegroundColor Cyan
$tg = @{}
foreach ($d in $Depths) {
    foreach ($r in $tRuns) {
        if ($r.maxD -and $d -gt $r.maxD) {
            Write-Host ("  (skipping {0} @ d{1}: MMA f16-scratch = 8 GiB at 32K -> OOM on 16 GB)" -f $r.tag.Trim(), $d) -ForegroundColor DarkGray
            continue
        }
        Set-Flags $r.f16 $r.native
        $logf = Join-Path $outDir ("bench_tg_d${d}_" + $r.tag.Trim() + ".log")
        $o = Run-Capture $bench @("-m",$ModelPath,"-fa","on","-ngl","99","-p","8","-n","128",
                                  "-d",$d,"-r",$Reps,"-ctk",$r.ctk,"-ctv",$r.ctv) $logf
        $m = ($o | Select-String 'tg128 @ d\d+\s*\|\s*([0-9.]+)').Matches
        $tg["$($r.tag.Trim())@d$d"] = $(if ($m) { [double]$m[0].Groups[1].Value } else { $null })
    }
    Set-Flags $false
    $base = $tg["nvfp4-dot@d$d"]; $fuse = $tg["nvfp4-fused@d$d"]; $q41 = $tg["q4_1-dot@d$d"]
    if ($base -and $fuse) {
        $g = ($fuse/$base - 1) * 100
        Write-Host ("  d{0,-6} nvfp4 fused {1,8:F2} t/s vs dot {2,8:F2} t/s  ({3:+0.0;-0.0}%)" -f $d, $fuse, $base, $g) `
                   -ForegroundColor $(if ($g -gt 0) {"Green"} else {"Red"})
    }
    if ($q41 -and $fuse) {
        $s = ($fuse/$q41 - 1) * 100
        Write-Host ("         vs q4_1 SPEED-ETALON {0,8:F2} t/s : {1:+0.0;-0.0}%  {2}" -f $q41, $s, `
                    $(if ($s -ge -2.0) {"WITHIN 2% of etalon"} else {"BELOW etalon by >2% - speed work needed"})) `
                   -ForegroundColor $(if ($s -ge 0) {"Green"} else {"Yellow"})
    }
}
Write-Host ""

# ---------------------------------------------------------------- P) PPL
if ($Ppl) {
    Write-Host "== P) PPL: nvfp4-fused / q4_1 vs fp16-gold anchor (ub=1 -> vec kernel) ==" -ForegroundColor Cyan
    $pRuns = $tRuns + @(@{ tag="fp16-gold"; ctk="f16"; ctv="f16"; f16=$false })
    $pplVals = [ordered]@{}
    foreach ($r in $pRuns) {
        Set-Flags $r.f16 $r.native
        $logf = Join-Path $outDir ("ppl_" + $r.tag.Trim() + ".log")
        $o = Run-Capture $pplExe @("-m",$ModelPath,"-f",$eval,"-c","4096","--chunks",("$Chunks"),"-b","1","-ub","1",
                               "-ngl","99","-fa","on","-ctk",$r.ctk,"-ctv",$r.ctv,"-t","7","-s","42") $logf
        $m = ($o | Select-String 'Final estimate: PPL = ([0-9.]+)').Matches
        if ($m) { $pplVals[$r.tag.Trim()] = [double]$m[0].Groups[1].Value
                  Write-Host ("  {0} PPL = {1}" -f $r.tag, $pplVals[$r.tag.Trim()].ToString("F4",$inv)) }
        else    { Write-Host ("  {0} PPL NOT PARSED - see {1}" -f $r.tag, $logf) -ForegroundColor Yellow }
    }
    Set-Flags $false
    if ($pplVals.Contains("fp16-gold")) {
        $gold = $pplVals["fp16-gold"]
        foreach ($k in @("nvfp4-fused","q4_1-dot")) {
            if (-not $pplVals.Contains($k)) { continue }
            $d = $pplVals[$k] - $gold
            $ok = ($k -ne "nvfp4-fused") -or ([Math]::Abs($d) -le $GoldTol)
            Write-Host ("  {0} vs gold: {1} ({2:+0.00;-0.00}%)  {3}" -f $k, $d.ToString("+0.0000;-0.0000",$inv),
                        (100*$d/$gold), $(if ($k -eq "nvfp4-fused") { if ($ok) {"PASS gate |d|<=$GoldTol"} else {"FAIL gate |d|>$GoldTol"} } else {"" })) `
                   -ForegroundColor $(if ($ok) {"Green"} else {"Red"})
        }
        if ($pplVals.Contains("nvfp4-dot")) {
            $dv = [Math]::Abs($pplVals["nvfp4-fused"] - $pplVals["nvfp4-dot"])
            Write-Host ("  fused vs dot decode-parity: |dPPL| {0}  (V-decode is provably value-identical, see fa_vnvfp4_decode_diff)" -f `
                        $dv.ToString("F4",$inv)) -ForegroundColor DarkGray
        }
    }
    Write-Host ""
}

Write-Host "done.  ($outDir)"

# =============================================================================
# run_fa_mma_ab.ps1 - Gate A/B for the fused-f16 KQ variant of the vec FA kernel
# (GGML_CUDA_FATTN_VEC_KQ_MMA_F16: NVFP4 K decoded in-register to f16 fragments + dense
#  m16n8k16 MMA; see NVFP4_FA_RESEARCH.md SS10e. The old GGML_CUDA_FATTN_VEC_KQ_MMA
#  FP4-Q path is closed and stays forced-off here.)
# Runs four sections; nothing is measured
# by the build itself. All runs use the SAME binary, only env vars change.
# Output streams live to the console (Tee) and into per-run log files.
#
#   powershell -ExecutionPolicy Bypass -File .\run_fa_mma_ab.ps1          # quick file, d8192
#   powershell -ExecutionPolicy Bypass -File .\run_fa_mma_ab.ps1 -Full    # wiki.test.raw for PPL
#   powershell -ExecutionPolicy Bypass -File .\run_fa_mma_ab.ps1 -Sections AB
#
# A) numerics: PPL with -c 4096 --chunks 1 -b 1 -ub 1, i.e. one 4096-token chunk processed as
#    single-token forwards (Q->ne[1]==1). NB: llama-perplexity -c means --ctx-size, the chunk
#    limit is --chunks (common/arg.cpp).
#    Routing fact (fattn.cu get_best_fattn_kernel): quantized K/V -> vec kernel ONLY for
#    Q->ne[1] <= 2; GGML_CUDA_FATTN_VEC_NATIVE does NOT change kernel selection (it only
#    gates vec-case availability), so a ub=512 PPL run is pure mma_f16 and the flag is inert.
#    With ub=1 both runs are guaranteed vec; only the KQ method differs (q8_1 dot vs FP4 mma).
#    Gate: |dPPL| <= ~0.01. Bit-identical numbers here would mean the flag did not take effect.
# B) decode tg: default routing already picks vec; flag off/on.
# C) prefill pp: with ub=512 all three runs route to mma_f16 (vec is never picked for prefill),
#    so C measures mma_f16-scratch only - kept as a stability reference. FP4-mma prefill would
#    need a nvfp4-direct loader inside fattn-mma-f16 (Stage B1), not a routing change.
# D) greedy generation A/B: fixed seed/prompt, temp 0, 256 decode tokens through the vec kernel;
#    outputs must DIFFER (Q requant changes scores) and stay plausible.
# =============================================================================
param(
    [string]$ModelPath = "G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
    [string]$RepoRoot  = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main",
    [string]$BuildDir  = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main\build-x64-windows-msvc-release",
    [string]$Sections  = "ABC",
    [switch]$Full,                                  # PPL over wikitext-2-raw\wiki.test.raw instead of 200KB quick
    [int[]]$Depths     = @(8192, 32768),            # B) tg depths
    [int]$PPTokens     = 2048,                      # C) prefill size / depth
    [int]$PPDepth      = 2048,
    [int]$Reps         = 2
)
$ErrorActionPreference = "Continue"
$inv  = [System.Globalization.CultureInfo]::InvariantCulture
$pp   = Join-Path $BuildDir "bin\llama-perplexity.exe"
$bench= Join-Path $BuildDir "bin\llama-bench.exe"
$eval = if ($Full) { Join-Path $RepoRoot "ppl_eval\wikitext-2-raw\wiki.test.raw" }
                 else { Join-Path $RepoRoot "ppl_eval\wiki.test.quick.txt" }
$outDir = Join-Path $RepoRoot ("fa_mma_ab_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function Set-Flags { param($native, $mma)
    if ($native) { $env:GGML_CUDA_FATTN_VEC_NATIVE = "1" } else { $env:GGML_CUDA_FATTN_VEC_NATIVE = "" }
    if ($mma) { $env:GGML_CUDA_FATTN_VEC_KQ_MMA_F16 = "1" } else { $env:GGML_CUDA_FATTN_VEC_KQ_MMA_F16 = "" }
    $env:GGML_CUDA_FATTN_VEC_KQ_MMA = ""  # closed stage-3 FP4-Q path, kept off for this gate
}
function Run-Capture { param($exe, [string[]]$argv, [string]$log)
    Write-Host ("  >>> running {0}  (log: {1})" -f (Split-Path $exe -Leaf), $log) -ForegroundColor DarkGray
    $o = & $exe @argv 2>&1 | Tee-Object -LiteralPath $log
    return $o
}

Write-Host "fa mma A/B  |  model: $ModelPath"
Write-Host "eval: $eval"
Write-Host "logs -> $outDir"
Write-Host ""

# ---------------------------------------------------------------- A) numerics
if ($Sections -match "A") {
    Write-Host "== A) PPL numerics (ub=1 => every forward is nq=1 => vec kernel) ==" -ForegroundColor Cyan
    Write-Host "  NB: in llama-perplexity -c is --ctx-size; the chunk LIMIT is --chunks." -ForegroundColor DarkGray
    Write-Host "  1 chunk x 4096 single-token forwards, ~2-4 min per run." -ForegroundColor DarkGray
    $ppl = @{}
    foreach ($mma in @($false, $true)) {
        Set-Flags $false $mma
        $tag  = if ($mma) { "mma-KQ " } else { "q8_1-KQ" }
        $logf = Join-Path $outDir ("ppl_vec_" + $(if ($mma) { "mma" } else { "q8" }) + ".log")
        $o = Run-Capture $pp @("-m",$ModelPath,"-f",$eval,"-c","4096","--chunks","1","-b","1","-ub","1",
                               "-ngl","99","-fa","on","-ctk","nvfp4","-ctv","nvfp4","-t","7","-s","42") $logf
        $m = ($o | Select-String 'Final estimate: PPL = ([0-9.]+)').Matches
        if ($m) { $ppl[$tag] = [double]$m[0].Groups[1].Value
                  Write-Host ("  vec {0}  PPL = {1}" -f $tag, $ppl[$tag].ToString("F4",$inv)) }
        else    { Write-Host ("  vec {0}  PPL NOT PARSED - see {1}" -f $tag, $logf) -ForegroundColor Yellow }
    }
    Set-Flags $false $false
    if ($ppl.Count -eq 2) {
        $d = $ppl["mma-KQ "] - $ppl["q8_1-KQ"]
        $verdict = if ($ppl["mma-KQ "] -eq $ppl["q8_1-KQ"]) { "IDENTICAL => mma variant did NOT run (check flag)" }
                  elseif ([Math]::Abs($d) -le 0.01) { "PASS (|dPPL| <= 0.01)" }
                  else { "FAIL (|dPPL| > 0.01)" }
        Write-Host ("  delta = {0}  ->  {1}" -f $d.ToString("+0.0000;-0.0000",$inv), $verdict) `
                   -ForegroundColor $(if ($verdict -like "PASS*") {"Green"} elseif ($verdict -like "FAIL*") {"Red"} else {"Yellow"})
    }
    Write-Host ""
}

# ---------------------------------------------------------------- B) decode
if ($Sections -match "B") {
    Write-Host "== B) decode tg A/B (default routing = vec for nvfp4 decode) ==" -ForegroundColor Cyan
    foreach ($d in $Depths) {
        $r = @()
        foreach ($mma in @($false, $true)) {
            Set-Flags $false $mma
            $logf = Join-Path $outDir ("bench_tg_d${d}_" + $(if ($mma) { "mma" } else { "off" }) + ".log")
            $o = Run-Capture $bench @("-m",$ModelPath,"-fa","on","-ngl","99","-p","8","-n","128",
                                      "-d",$d,"-r",$Reps,"-ctk","nvfp4","-ctv","nvfp4") $logf
            $m = ($o | Select-String 'tg128 @ d\d+\s*\|\s*([0-9.]+)').Matches
            $r += $(if ($m) { [double]$m[0].Groups[1].Value } else { $null })
        }
        Set-Flags $false $false
        if ($r[0] -and $r[1]) {
            $gain = ($r[1] / $r[0] - 1) * 100
            Write-Host ("  d{0,-6} q8_1-KQ {1,8} t/s   mma-KQ {2,8} t/s   ({3:+0.0;-0.0}%)" -f `
                        $d, $r[0].ToString("F2",$inv), $r[1].ToString("F2",$inv), $gain)
        } else {
            Write-Host ("  d{0} parse failed - see logs in $outDir" -f $d) -ForegroundColor Yellow
        }
    }
    Write-Host ""
}

# ---------------------------------------------------------------- C) prefill
if ($Sections -match "C") {
    Write-Host "== C) prefill pp A/B (all three route to mma_f16; reference only) ==" -ForegroundColor Cyan
    $combos = @(
        @{ native=$true;  mma=$false; tag="vec q8_1-KQ   " },
        @{ native=$true;  mma=$true;  tag="vec mma-KQ    " },
        @{ native=$false; mma=$false; tag="default gate  " }
    )
    foreach ($c in $combos) {
        Set-Flags $c.native $c.mma
        $logf = Join-Path $outDir ("bench_pp_" + ($c.tag.Trim() -replace '\s','_') + ".log")
        $o = Run-Capture $bench @("-m",$ModelPath,"-fa","on","-ngl","99","-p",$PPTokens,"-n","8",
                                  "-d",$PPDepth,"-r",$Reps,"-ctk","nvfp4","-ctv","nvfp4") $logf
        $m = ($o | Select-String ("pp{0} @ d{1}\s*\|\s*([0-9.]+)" -f $PPTokens, $PPDepth)).Matches
        if (-not $m) { $m = ($o | Select-String ("pp{0}\s*\|\s*([0-9.]+)" -f $PPTokens)).Matches }
        if ($m) { Write-Host ("  {0} pp = {1,9} t/s" -f $c.tag, $m[0].Groups[1].Value) }
        else    { Write-Host ("  {0} pp NOT PARSED - see {1}" -f $c.tag, $logf) -ForegroundColor Yellow }
    }
    Set-Flags $false $false
    Write-Host ""
}

# ------------------------------------------------- D) greedy generation numerics
if ($Sections -match "D") {
    Write-Host "== D) greedy generation A/B (decode forwards go through the vec kernel) ==" -ForegroundColor Cyan
    $comp  = Join-Path $BuildDir "bin\llama-completion.exe"
    $prompt = "The first man to walk on the surface of the Moon was Neil Armstrong. Write a short factual paragraph about the Apollo 11 mission."
    $outs = @{}
    foreach ($mma in @($false, $true)) {
        Set-Flags $false $mma
        $tag  = if ($mma) { "mma-KQ " } else { "q8_1-KQ" }
        $logf = Join-Path $outDir ("gen_" + $(if ($mma) { "mma" } else { "off" }))
        Write-Host ("  >>> {0}: generating 256 greedy tokens (text appears when the run finishes; tool log -> gen_*.log)" -f $tag.Trim()) -ForegroundColor DarkGray
        $o = & $comp @("-m",$ModelPath,"-p",$prompt,"-n","256","-s","42","--temp","0","-no-cnv","-st",
                       "-ngl","99","-fa","on","-ctk","nvfp4","-ctv","nvfp4") 2>($logf + ".log") | Tee-Object -Variable genlines
        $text = ($genlines | Out-String)
        Set-Content -LiteralPath ($logf + ".txt") -Value $text -Encoding UTF8
        $outs[$tag] = $text
    }
    Set-Flags $false $false
    if ($outs.Count -eq 2) {
        $a = $outs["q8_1-KQ"]; $b = $outs["mma-KQ "]
        if ($a -eq $b) {
            Write-Host "  outputs IDENTICAL - suspicious: Q requant must change scores (check the flag)" -ForegroundColor Yellow
        } else {
            $i = 0; $n = [Math]::Min($a.Length, $b.Length)
            while ($i -lt $n -and $a[$i] -eq $b[$i]) { $i++ }
            Write-Host ("  outputs DIFFER from char {0} ({1} vs {2} chars) - read gen_off.txt / gen_mma.txt" -f `
                        $i, $a.Length, $b.Length) -ForegroundColor Green
        }
    }
    Write-Host ""
}

Write-Host "done.  ($outDir)"

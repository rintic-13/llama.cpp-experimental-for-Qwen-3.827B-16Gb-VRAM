# build_fa_fp4_probe.ps1 - builds tools/fa_fp4_mma_probe.cu as a standalone CUDA tool (no llama.cpp link)
# Requires: Visual Studio 2022 (vcvars64) + CUDA 13.x. Output: build-.../bin/fa_fp4_mma_probe.exe
param(
    [string]$RepoRoot = "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main",
    [string]$Arch = "sm_120a"
)
$ErrorActionPreference = "Continue"
$prev_eap = $ErrorActionPreference
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
$src = Join-Path $RepoRoot "tools\fa_fp4_mma_probe.cu"
$exe = Join-Path $RepoRoot "build-x64-windows-msvc-release\bin\fa_fp4_mma_probe.exe"
if (-not (Test-Path $src)) { Write-Error "no source: $src" }

cmd /c "call `"$vcvars`" >nul 2>&1 && nvcc -O3 -std=c++17 `"-arch=$Arch`" `"$src`" -o `"$exe`" 2>&1"
if ($LASTEXITCODE -ne 0) { Write-Error "build failed ($LASTEXITCODE)" }
Write-Host "built: $exe"

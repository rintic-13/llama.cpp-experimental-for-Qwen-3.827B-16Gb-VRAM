# Reconfigure + rebuild llama.cpp TurboQuant fork with turbo3q/turbo4q FA kernels.
# Switches (experiment A/B toggles):
#   -QjlMse  : TURBO_QJL_MSE_OPTIMAL  (MSE-optimal QJL alpha; CLOSED experiment, never enable)
#   -Dither  : GGML_TURBO_DITHER      (subtractive dither turbo3+turbo4; VALIDATED default -
#                                      cmake default is now ON, so a plain build already has it;
#                                      use this script WITHOUT -Dither only to rebuild the
#                                      bit-identical 2026-10-31 baseline codec;
#                                      with dither ON, decode FA for turbo4 K/V combos goes to
#                                      the f16-scratch MMA path (speed gate), V=turbo3 stays vec)
#   -DitherV1: GGML_TURBO_DITHER_V1   (A/B: restore the v1 per-coordinate murmur dither stream;
#                                      only meaningful together with -Dither)
param([switch]$QjlMse, [switch]$Dither, [switch]$DitherV1)
$ErrorActionPreference = "Stop"
Set-Location "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main"

$vcvars = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$fa = 'q4_0-q4_0;q8_0-q8_0;q8_0-q4_0;q5_1-q5_1;q5_1-q4_1;q5_1-q4_0;q4_1-q4_1;f16-f16;q5_1-turbo4_0;q5_1-turbo3_0;turbo4_0-turbo3_0;q5_1-turbo5_0;turbo5_0-turbo5_0;turbo5_0-turbo4_0;turbo5_0-turbo3_0;q5_1-turbo4q_0;q5_1-turbo3q_0;turbo4q_0-turbo4_0;turbo4q_0-turbo4q_0;turbo4_0-turbo3q_0;nvfp4-nvfp4;q5_1-nvfp4;f16-nvfp4'

# Always pass ALL experiment options explicitly (ON or OFF) - otherwise the previous
# configure's value lingers in CMakeCache.txt and silently leaks into builds.
$defs = " -DGGML_TURBO_QJL_MSE=$(if ($QjlMse) {'ON'} else {'OFF'}) -DGGML_TURBO_DITHER=$(if ($Dither) {'ON'} else {'OFF'}) -DGGML_TURBO_DITHER_V1=$(if ($DitherV1 -and $Dither) {'ON'} else {'OFF'})"
Write-Host "configure flags:$defs"

cmd /c "call `"$vcvars`" && cmake -S . -B build-x64-windows-msvc-release -DGGML_CUDA_FA_QUANTS=`"$fa`"$defs"
if ($LASTEXITCODE -ne 0) { Write-Host "CONFIGURE FAILED ($LASTEXITCODE)"; exit 1 }

cmd /c "call `"$vcvars`" && cmake --build build-x64-windows-msvc-release --target llama-perplexity"
if ($LASTEXITCODE -ne 0) { Write-Host "BUILD FAILED ($LASTEXITCODE)"; exit 1 }

Write-Host "BUILD OK"

@echo off
rem run_s78_gate.cmd - production gate for S7 (CEN_TAB=1) + S8 (VTAB=4) now that both defaults are ON.
rem Run on an IDLE GPU.  Step 0 prints nvidia-smi - abort manually if a foreign process holds VRAM
rem (never kill foreign processes).  See FA_S5_S6_ATTRIBUTION_20261103.md SS2f-SS2j.
setlocal EnableDelayedExpansion
cd /d "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main"
set B=build-x64-windows-msvc-release\bin
set LOG=ppl_s78_%DATE:~6,4%%DATE:~3,2%%DATE:~0,2%.log

echo == 0) nvidia-smi before (must be ~0 MiB used / no foreign processes) == > %LOG%
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader >> %LOG%
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> %LOG%

echo == 1) probe built with the DEFAULT headers (S7+S8 now on): every hash must equal the frozen == >> %LOG%
echo ==    baseline: T6 fnv 73d4a5f05d924efc / cfnv 4664e8eae073af4c, T7 78934835d49f0066,       == >> %LOG%
echo ==    T8 a066508a45916751, T1 33b6fa5ef169e4e6/782e697aec966835, T3 44464b17ae8d1478/       == >> %LOG%
echo ==    5a6e7d955d4d7722, T4 8381bd7573e3d88b, T2 14650fb0739d0383, C1 209c4022a4eefab3,      == >> %LOG%
echo ==    C2 f3a87d8d9fac3a51.  Any move = stop, do not read the PPL below.                    == >> %LOG%
echo -- rebuild the default probe from the current headers (flags NOT passed on the command line) -- >> %LOG%
set SRC=%CD%
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
nvcc -O3 -std=c++17 -arch=sm_120a -DGGML_TURBO_DITHER -I "%SRC%\ggml\include" -I "%SRC%\ggml\src" -I "%SRC%\ggml\src\ggml-cuda" "%SRC%\tools\fa_turbo_ab_probe.cu" -o "%B%\fa_turbo_default.exe" >> %LOG% 2>&1
echo nvcc exit: %ERRORLEVEL% >> %LOG%
if errorlevel 1 (
    echo ERROR: probe build failed - see %LOG%
    exit /b 1
)
%B%\fa_turbo_default.exe >> %LOG% 2>&1

echo. >> %LOG%
echo == 2) PPL production default (fused t4-t3 routing), protocol identical to the S2d gate row T8 == >> %LOG%
echo ==    reference fused-t4 PPL 7.1751 +/- 0.380 (NVFP4_FA_RESEARCH.md line 1311);               == >> %LOG%
echo ==    the S7/S8 kernels are bitwise-identical so this must land ON 7.1751, gate ^|delta^| ^<= 0.05 == >> %LOG%
set M=G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf
set COMMON=-c 4096 --chunks 1 -b 1 -ub 1 -ngl 99 -fa on -t 7 -s 42
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> %LOG%
%B%\llama-perplexity.exe -m %M% -f ppl_eval\wiki.test.quick.txt %COMMON% -ctk turbo4 -ctv turbo3 >> %LOG% 2>&1

echo. >> %LOG%
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> %LOG%
echo. >> %LOG%
echo Full log: %LOG%
echo.
echo HOW TO READ: step 1 hashes must match the frozen list exactly (the probe is compiled from the
echo current headers with NO -D flags, so it now measures the shipping configuration).  Step 2 must
echo land within 0.05 of 7.1751; since the kernels are bitwise-identical to the pre-flip default, the
echo number should be identical, not merely close.
endlocal

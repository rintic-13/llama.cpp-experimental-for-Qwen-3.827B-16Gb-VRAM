@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set LOG=s8_vtab_20261103.log
set BIN=build-x64-windows-msvc-release\bin

echo == S8: byte-indexed turbo3 V centroid tables (bitwise) + S7 combination == > %LOG%
echo -- nvidia-smi before: must show ~0 MiB used, no foreign process (never kill foreign procs) -- >> %LOG%
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader >> %LOG%
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> %LOG%

for %%E in (fa_turbo_pfm0 fa_turbo_centab fa_turbo_vtab fa_turbo_vtab_ce) do (
    if not exist "%BIN%\%%E.exe" (
        echo ERROR: %BIN%\%%E.exe missing - rebuild the probes, see FA_S5_S6_ATTRIBUTION_20261103.md SS7
        exit /b 1
    )
)

for %%M in (0 1 2 3 0 1 2 3) do (
    if %%M==0 set "EXE=fa_turbo_pfm0"
    if %%M==1 set "EXE=fa_turbo_centab"
    if %%M==2 set "EXE=fa_turbo_vtab"
    if %%M==3 set "EXE=fa_turbo_vtab_ce"
    echo --- !EXE! --- >> %LOG%
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> %LOG%
    "%BIN%\!EXE!.exe" >> %LOG% 2>&1
)

echo.
echo Full log: %LOG%
echo Pass criteria (frozen S2d baseline, FA_TURBO_BASELINE_20261103.md):
echo   fnv/cfnv of every row must be unchanged: T6 fnv 73d4a5f05d924efc cfnv 4664e8eae073af4c,
echo   T7 fnv 78934835d49f0066, T8 fnv a066508a45916751, T1 cfnv 782e697aec966835, T3 cfnv 5a6e7d955d4d7722.
echo Static expectation (T6 instr/tile): ref 12176, S7 9528, S8 10000, S7+S8 7344.
echo Timing reference (this window, ref T6 0.9985): S7 0.725, S8 0.8205, S7+S8 0.7025.
echo NOTE absolute ms drift ~6% between windows; compare ratios inside one window, not vs FA_TURBO_BASELINE.
endlocal

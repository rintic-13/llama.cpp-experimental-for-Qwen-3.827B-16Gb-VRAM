@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set LOG=s8_pick_20261103.log
set BIN=build-x64-windows-msvc-release\bin

echo == Picking the production V-decode form: float4 gather vs float2 pairs vs scalar, per-instantiation == > %LOG%
echo -- nvidia-smi before: must show ~0 MiB used, no foreign process (never kill foreign procs) -- >> %LOG%
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader >> %LOG%
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> %LOG%

for %%E in (fa_turbo_pfm0 fa_turbo_centab fa_turbo_vtab fa_turbo_vtab8 fa_turbo_vtabe fa_turbo_vtab_ce fa_turbo_vtab8_ce fa_turbo_vtabe_ce fa_turbo_vtab3_ce fa_turbo_vtab5_ce) do (
    if not exist "%BIN%\%%E.exe" (
        echo ERROR: %BIN%\%%E.exe missing - rebuild, see FA_S5_S6_ATTRIBUTION_20261103.md SS2i
        exit /b 1
    )
)

:: 0 ref | 1 S7 alone | 2 +V float4(1) | 3 +V scalar(2) | 4 +V float2(4)
:: 5..9 the same three V forms with S7 on, plus the two per-instantiation auto rules (3 = 2/1, 5 = 2/4)
for %%M in (0 1 2 3 4 5 6 7 8 9 9 8 7 6 5 4 3 2 1 0) do (
    if %%M==0 set "EXE=fa_turbo_pfm0"
    if %%M==1 set "EXE=fa_turbo_centab"
    if %%M==2 set "EXE=fa_turbo_vtab"
    if %%M==3 set "EXE=fa_turbo_vtab8"
    if %%M==4 set "EXE=fa_turbo_vtabe"
    if %%M==5 set "EXE=fa_turbo_vtab_ce"
    if %%M==6 set "EXE=fa_turbo_vtab8_ce"
    if %%M==7 set "EXE=fa_turbo_vtabe_ce"
    if %%M==8 set "EXE=fa_turbo_vtab3_ce"
    if %%M==9 set "EXE=fa_turbo_vtab5_ce"
    echo --- !EXE! --- >> %LOG%
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> %LOG%
    "%BIN%\!EXE!.exe" >> %LOG% 2>&1
)

echo.
echo Full log: %LOG%
echo.
echo HOW TO READ THIS (FA_S5_S6_ATTRIBUTION_20261103.md SS2h-SS2i)
echo   EVERY build here is bitwise-valid: T6 fnv 73d4a5f05d924efc / cfnv 4664e8eae073af4c, T7
echo   78934835d49f0066, T8 a066508a45916751, T1 cfnv 782e697aec966835, T3 cfnv 5a6e7d955d4d7722,
echo   T4 8381bd7573e3d88b, T2 14650fb0739d0383, C1 209c4022a4eefab3, C2 f3a87d8d9fac3a51.
echo   Any row that moves = the build is wrong, stop and investigate before reading timings.
echo.
echo   Rows where S7 is active (fused turbo4-K: T6, C2) prefer the narrow broadcast table; rows where
echo   the K side keeps its mux trees (T1, T2, T4, T8, C1) prefer the wide gather table - SS2h measured
echo   that split.  The decision this run has to make:
echo     does vtabe (float2, form 4) beat vtab (float4, form 1) on the NON-S7 rows?
echo       yes  -^> production = CEN_TAB=1 + VTAB=5   (2 where S7, 4 elsewhere)
echo       no   -^> production = CEN_TAB=1 + VTAB=3   (2 where S7, 1 elsewhere)
echo   Also check vtab3_ce / vtab5_ce against the best single-form combo row-by-row: auto is only worth
echo   having if it is at least as good as the best fixed form on EVERY row.
echo.
echo Last window med ms (T6): ref 0.9975, S7 0.724, +float4 0.7025, +scalar 0.6545, ideal-no-gather 0.6275.
echo NOTE absolute ms drift ~6%% between windows; compare ratios inside one window, not vs FA_TURBO_BASELINE.
endlocal

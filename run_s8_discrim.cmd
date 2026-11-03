@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set LOG=s8_discrim_20261103.log
set BIN=build-x64-windows-msvc-release\bin

echo == Why is the S7+S8 win sub-linear (predict -39.7%%, measured -29.7%%)?  smem footprint vs table gather == > %LOG%
echo -- nvidia-smi before: must show ~0 MiB used, no foreign process (never kill foreign procs) -- >> %LOG%
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader >> %LOG%
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> %LOG%

for %%E in (fa_turbo_pfm0 fa_turbo_centab fa_turbo_vtab fa_turbo_vtab_ce fa_turbo_vtab_pad2k fa_turbo_vtab_g2 fa_turbo_vtab_g2_ce fa_turbo_vtab8 fa_turbo_vtab8_ce) do (
    if not exist "%BIN%\%%E.exe" (
        echo ERROR: %BIN%\%%E.exe missing - rebuild the probes, see FA_S5_S6_ATTRIBUTION_20261103.md SS2g
        exit /b 1
    )
)

:: 0 ref | 1 S7 | 2 S8 | 3 S7+S8 | 4 S8 +2KB dead smem | 5 S8 gather-collapsed | 6 S7+S8 gather-collapsed
:: 7 S8c scalar 32B table | 8 S7+S8c
for %%M in (0 1 2 3 4 5 6 7 8 8 7 6 5 4 3 2 1 0) do (
    if %%M==0 set "EXE=fa_turbo_pfm0"
    if %%M==1 set "EXE=fa_turbo_centab"
    if %%M==2 set "EXE=fa_turbo_vtab"
    if %%M==3 set "EXE=fa_turbo_vtab_ce"
    if %%M==4 set "EXE=fa_turbo_vtab_pad2k"
    if %%M==5 set "EXE=fa_turbo_vtab_g2"
    if %%M==6 set "EXE=fa_turbo_vtab_g2_ce"
    if %%M==7 set "EXE=fa_turbo_vtab8"
    if %%M==8 set "EXE=fa_turbo_vtab8_ce"
    echo --- !EXE! --- >> %LOG%
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> %LOG%
    "%BIN%\!EXE!.exe" >> %LOG% 2>&1
)

echo.
echo Full log: %LOG%
echo.
echo HOW TO READ THIS (see FA_S5_S6_ATTRIBUTION_20261103.md SS2g)
echo   Bitwise-valid builds (fnv 73d4a5f05d924efc / cfnv 4664e8eae073af4c on T6 must not move):
echo                    pfm0, centab, vtab, vtab_ce, vtab_pad2k (dead smem), vtab8, vtab8_ce.
echo   TIMING ONLY, NUMERICS MEANINGLESS - ignore their fnv/cfnv: vtab_g2, vtab_g2_ce.
echo.
echo   vtab_pad2k  S8 + 2048 B dead static smem (address leaked through an asm barrier), T6 static
echo                    smem 5376 -^> 7424 B = S7's footprint, REG unchanged.  If pad2k == vtab in time,
echo                    the footprint / L1 carveout is NOT the reason the combination is sub-linear.
echo                    (Occupancy is register-limited - 168 reg x 128 thr = 21504, so 3 blocks/SM at
echo                    both footprints - which is why this control can only be weak evidence.)
echo   vtab_g2     keeps both LDG.128, the byte loads, the SEL chain and the FMA walk, but forces the
echo                    table index to two values 128 B apart: 2 L1 lines per gather instead of up to
echo                    32.  +2.5%% instructions (10184 / 7536).  If g2_ce beats vtab_ce (0.7025 last
echo                    window) the missing win is L1 wavefront cost of the gather.
echo   vtab8       S8c, the candidate fix: 8-value 32 B scalar table (one cache line, always a
echo                    broadcast), 4 LDG.32 + 3-bit index instead of 2 LDG.128 + 4 SEL.  More
echo                    instructions, minimal L1.  If vtab8 ~= vtab in time and vtab8_ce clearly beats
echo                    vtab_ce, the gather was the limiter and S8c replaces S8.
echo.
echo Static counts (T6 instr/tile): ref 12176, S7 9528, S8 10000, S7+S8 7344, pad2k 10000,
echo                    g2 10184, S7+S8g2 7536, S8c 10832, S7+S8c 8176.
echo Previous window med ms (0 MiB): ref 0.9985, S7 0.725, S8 0.8205, S7+S8 0.7025.
echo NOTE absolute ms drift ~6%% between windows; compare ratios inside one window, not vs FA_TURBO_BASELINE.
endlocal

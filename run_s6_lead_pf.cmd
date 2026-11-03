@echo off
setlocal
cd /d "%~dp0"
set LOG=s6_lead_pf_20261103.log

echo == S6: lead prefetch + instruction-cost attribution == > %LOG%
echo -- nvidia-smi before: must show ~0 MiB used / no foreign process -- >> %LOG%
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader >> %LOG%
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> %LOG%

if not exist "build-x64-windows-msvc-release\bin\fa_turbo_pfm0.exe" (
    echo ERROR: probes missing, rebuild them first ^(see FA_S5_S6_ATTRIBUTION_20261103.md SS7^)
    exit /b 1
)

echo.
echo == A) lead prefetch: 8-fragment (one turbo block) lead, bitwise ==
for %%M in (0 32 64 33) do (
    echo --- mode %%M --- >> %LOG%
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> %LOG%
    "build-x64-windows-msvc-release\bin\fa_turbo_pfm%%M.exe" >> %LOG% 2>&1
    findstr /b /c:"T6_fused" /c:"T7_fused" %LOG%
)

echo.
echo == B) instruction-cost attribution: NUMERICS MEANINGLESS, timing only ==
for %%M in (1 2 3) do (
    echo --- cost %%M --- >> %LOG%
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> %LOG%
    "build-x64-windows-msvc-release\bin\fa_turbo_cost%%M.exe" >> %LOG% 2>&1
    findstr /b /c:"T6_fused" %LOG%
)

echo.
echo Full log: %LOG%
echo A) correct = T6 below 1.062 ms and fnv 73d4a5f05d924efc.
echo B) cost1 = dither removed (17.3%% of instructions), cost2 = centroid removed (17.9%%),
echo    cost3 = both (35.2%%). These bound how much instruction work can buy.
endlocal

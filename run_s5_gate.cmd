@echo off
rem run_s5_gate.cmd - gate for S5 (cp.async smem staging of K tiles in the fused turbo vec decode
rem kernel). Probe-only stage: no DLL involved. Requires both cp.async probe exes built from the
rem current tree:
rem   fa_turbo_ab_cpasync.exe     production shape: cp.async ON, turbo4-D256-min1 now 2 blocks/SM
rem                               (REG 255 / STACK 0 / SHARED 38656 - verified in build log)
rem   fa_turbo_ab_cpasync_mb3.exe ablation: cp.async ON but MIN_BLOCKS=3 forced -> identical reg
rem                               budget to the S2d baseline -> ISOLATES the transport effect.
rem Run on an IDLE GPU (step 0 prints nvidia-smi; abort manually if a foreign process holds VRAM -
rem never kill foreign processes). PPL/bench numbers are contention-immune, probe med timings are NOT.
setlocal
cd /d "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main"
set B=build-x64-windows-msvc-release\bin

echo == 0) nvidia-smi before (must be ~0 MiB used / no foreign processes) ==
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo == 1) BITWISE GATE, clean-budget: fa_turbo_ab_cpasync_mb3 ==
rem With MIN_BLOCKS=3 forced the cp.async turbo4-D256 kernels get the SAME reg budget as the S2d
rem baseline (168 regs). cp.async is pure transport, so every fnv MUST equal the frozen S2d values:
rem   T1 33b6fa5e  T2 14650fb0  T3 44464b17  T4 8381bd75
rem   T6 73d4a5f05d924efc  T7 78934835d49f0066  T8 a066508a45916751  C1 209c4022  C2 f3a87d8d
rem If T6 differs here, the cp.async path is NOT bitwise and S5 is REJECTED - stop and investigate.
%B%\fa_turbo_ab_cpasync_mb3.exe

echo == 2) production cpasync variant (2 blocks/SM, full 255-reg budget) ==
rem T6 may show budget-drift in the fnv here (reg budget 168->255 re-contracts FMAs; S2d documented
rem the same drift class). Judge this exe on TIMING only; bitwise is settled by step 1.
echo    expected med ms vs baseline: T6 below 1.062, T8 below 1.205 (T7 unchanged - turbo5 not staged)
%B%\fa_turbo_ab_cpasync.exe

echo == 3) S2d baseline probe for same-run reference ==
%B%\fa_turbo_ab_new.exe

echo == 4) repeat step 2 for a second reproduction (probe med timings are noisy) ==
%B%\fa_turbo_ab_cpasync.exe

echo == 5) nvidia-smi after ==
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
endlocal

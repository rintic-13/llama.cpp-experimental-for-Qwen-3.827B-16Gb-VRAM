@echo off
rem run_s2c_gate.cmd - S2c (register-only dither) + MIN_BLOCKS gate: standalone probes, ~seconds, no model.
rem Run on an IDLE GPU (nvidia-smi: no foreign VRAM users). Baseline to compare against:
rem FA_TURBO_BASELINE_20261103.md section 1 (med ms per row).
setlocal
cd /d "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main"
set B=build-x64-windows-msvc-release\bin

echo == nvidia-smi before (abort if a foreign process holds VRAM) ==
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo == 1) afrag bitwise gate (expect: turbo4(a)=0 turbo4(b-norms)=0 turbo5=0) ==
%B%\fa_turbo_frag_probe.exe

echo == 2) S2c timings, MIN_BLOCKS=1 (baseline default) ==
echo    compare vs baseline med ms: T1 1.412  T3 1.445  T4 1.697  ^|  T6 1.925  T7 1.964  T8 1.503  ^|  C1 0.503  C2 0.855
%B%\fa_turbo_ab_new.exe

echo == 3) MIN_BLOCKS=3 probe (registers capped to fit 3 blocks/SM) ==
%B%\fa_turbo_ab_mb3.exe

echo == 4) again 2) for a second reproduction ==
%B%\fa_turbo_ab_new.exe
endlocal

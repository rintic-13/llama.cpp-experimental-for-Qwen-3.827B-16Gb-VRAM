@echo off
rem run_s2d_gate.cmd - final gate for S2d (S2c dither + turbo4 fused enabled + conditional MIN_BLOCKS=3
rem for turbo-K D256). Requires: ggml-cuda.dll rebuilt from this tree. Run on an IDLE GPU (step 0 prints
rem nvidia-smi; abort manually if a foreign process holds VRAM - never kill foreign processes).
setlocal
cd /d "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main"
set B=build-x64-windows-msvc-release\bin

echo == 0) nvidia-smi before (must be ~0 MiB used / no foreign processes) ==
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo == 1) afrag bitwise gate (expect: turbo4(a)=0 turbo4(b-norms)=0 turbo5=0) ==
%B%\fa_turbo_frag_probe.exe

echo == 2) vec probe with FINAL policy (turbo-K D256 rows regs=168, others 255) ==
echo    expected med ms: T1 ~1.10  T2 ~0.67  T3 ~1.08  T4 ~1.40  T6 ~1.06  T7 ~1.06  T8 ~1.19  C1 ~0.50  C2 ~0.49
echo    baseline MB1 pre-S2c: T1 1.412 T3 1.445 T4 1.697 T6 1.925 T7 1.964 T8 1.503 C1 0.503 C2 0.855
%B%\fa_turbo_ab_new.exe

echo == 3) PPL: fused-t4 numerics gate, ^|delta^| ^<= 0.05 (T8 fused vs T9 dot, same -ub) ==
rem T8 = turbo4 NATIVE vec default (fused now selected), T9 = same with the fused switch forced OFF.
rem Reference: dot-t4 PPL 7.1802 (S2 close, ppl_results log from 2026-11-03 T2 row).
set M=G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf
set COMMON=-c 4096 --chunks 1 -b 1 -ub 1 -ngl 99 -fa on -t 7 -s 42
set GGML_CUDA_FATTN_VEC_NATIVE=1
echo -- T8 turbo4-turbo3 NATIVE default routing (fused KQ after flip) --
%B%\llama-perplexity.exe -m %M% -f ppl_eval\wiki.test.quick.txt %COMMON% -ctk turbo4 -ctv turbo3
echo -- T9 turbo4-turbo3 NATIVE forced dot (TURBO_KQ_MMA_F16=0) --
set GGML_CUDA_FATTN_VEC_TURBO_KQ_MMA_F16=0
%B%\llama-perplexity.exe -m %M% -f ppl_eval\wiki.test.quick.txt %COMMON% -ctk turbo4 -ctv turbo3
set GGML_CUDA_FATTN_VEC_TURBO_KQ_MMA_F16=
set GGML_CUDA_FATTN_VEC_NATIVE=

echo == 4) production A/B at d8192: fused+MB3 default vs everything OFF (old S2 behavior) ==
rem Pinned expectations from the S2 close: fused ON was 27.28 t/s, dot 26.79 t/s; MB3 should widen the gap.
echo -- P1 production default (fused ON, turbo-K D256 at 3 blocks/SM) --
%B%\llama-bench.exe -m %M% -ngl 99 -fa on -t 7 -ub 512 -d 8192 -p 512 -n 128 -r 3 -ctk turbo4 -ctv turbo3
echo -- P2 old behavior (dot + MB1 via forced fused OFF) --
set GGML_CUDA_FATTN_VEC_TURBO_KQ_MMA_F16=0
%B%\llama-bench.exe -m %M% -ngl 99 -fa on -t 7 -ub 512 -d 8192 -p 512 -n 128 -r 3 -ctk turbo4 -ctv turbo3
set GGML_CUDA_FATTN_VEC_TURBO_KQ_MMA_F16=

echo == 5) again step 2 for a second reproduction ==
%B%\fa_turbo_ab_new.exe
endlocal

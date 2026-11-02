@echo off
rem ppl_quick_turbo.cmd - turbo PPL baselines for the S2 gate (same -b1 -ub1 protocol as ppl_quick_20261102)
rem Rows: production default (turbo4 combos = f16-scratch numerics) vs forced NATIVE vec (q8_1 dot, S1 kernel).
rem S2 gate (updated 2026-11-02 after the speed gate + turbo4q retirement, NVFP4_FA_RESEARCH.md SS12): fused
rem f16-MMA KQ is selected for turbo5 K only (turbo4 K keeps dot+range-dither, fused is 35-71%% slower;
rem turbo4q_0 was retired from the FA path - worse PPL class, and the Oct-31 exes cannot parse its name).
rem  - T7 vs T6 (turbo5 K): the real numerics gate, |delta PPL| <= ~0.05 (nvfp4 fused precedent +0.03).
rem  - T5 vs T2 (turbo4 K): must be EXACTLY EQUAL - the selector excludes fused for turbo4 (proves it;
rem    set GGML_CUDA_FATTN_VEC_TURBO_KQ_MMA_F16_FORCE_T4=1 additionally to force fused for experiments).
rem T1/T4 are the f16-scratch and f16-gold references.
setlocal
cd /d "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main"
set B=build-x64-windows-msvc-release\bin\llama-perplexity.exe
set M=G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf
set F=ppl_eval\wiki.test.quick.txt
set COMMON=-c 4096 --chunks 1 -b 1 -ub 1 -ngl 99 -fa on -t 7 -s 42

echo == nvidia-smi before ==
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

echo == T1 turbo4-turbo3 DEFAULT routing (f16-scratch numerics) ==
%B% -m %M% -f %F% %COMMON% -ctk turbo4 -ctv turbo3

echo == T2 turbo4-turbo3 NATIVE vec (q8_1 dot + S1 mux) ==
set GGML_CUDA_FATTN_VEC_NATIVE=1
%B% -m %M% -f %F% %COMMON% -ctk turbo4 -ctv turbo3
set GGML_CUDA_FATTN_VEC_NATIVE=

echo == T3 q5_1-turbo3 NATIVE default (q8_1 dot + S1 mux) ==
%B% -m %M% -f %F% %COMMON% -ctk q5_1 -ctv turbo3

echo == T4 f16 gold reference ==
%B% -m %M% -f %F% %COMMON% -ctk f16 -ctv f16

echo == T5 turbo4-turbo3 NATIVE vec + FUSED f16-MMA KQ (S2 gate row) ==
set GGML_CUDA_FATTN_VEC_NATIVE=1
set GGML_CUDA_FATTN_VEC_TURBO_KQ_MMA_F16=1
%B% -m %M% -f %F% %COMMON% -ctk turbo4 -ctv turbo3
set GGML_CUDA_FATTN_VEC_TURBO_KQ_MMA_F16=
set GGML_CUDA_FATTN_VEC_NATIVE=

echo == T6 turbo5-turbo3 NATIVE vec (q8_1 dot + S1 mux) ==
set GGML_CUDA_FATTN_VEC_NATIVE=1
%B% -m %M% -f %F% %COMMON% -ctk turbo5 -ctv turbo3
set GGML_CUDA_FATTN_VEC_NATIVE=

echo == T7 turbo5-turbo3 NATIVE vec + FUSED f16-MMA KQ (S2 gate row) ==
set GGML_CUDA_FATTN_VEC_NATIVE=1
set GGML_CUDA_FATTN_VEC_TURBO_KQ_MMA_F16=1
%B% -m %M% -f %F% %COMMON% -ctk turbo5 -ctv turbo3
set GGML_CUDA_FATTN_VEC_TURBO_KQ_MMA_F16=
set GGML_CUDA_FATTN_VEC_NATIVE=
endlocal

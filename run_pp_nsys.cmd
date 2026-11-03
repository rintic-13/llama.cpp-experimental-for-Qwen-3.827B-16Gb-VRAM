@echo off
setlocal
cd /d "G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main"
set "NSYS=C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe"
set "OUT=C:\Users\rinat\AppData\Local\Temp\opencode"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

REM A) prefill at deep context: attention growth + per-op to_fp16 tax both present
"%NSYS%" profile -o "%OUT%\pp8k" --force-overwrite true .\build-x64-windows-msvc-release\bin\llama-bench.exe -m G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf -ngl 99 -fa on -t 7 -ub 512 -d 8192 -p 512 -r 1 -ctk turbo4 -ctv turbo3
if errorlevel 1 exit /b 1

REM B) same tokens, shallow context: baseline without the O(n_ctx) growth (neither attention nor dequant)
"%NSYS%" profile -o "%OUT%\pp512" --force-overwrite true .\build-x64-windows-msvc-release\bin\llama-bench.exe -m G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf -ngl 99 -fa on -t 7 -ub 512 -d 512 -p 512 -r 1 -ctk turbo4 -ctv turbo3
if errorlevel 1 exit /b 1

"%NSYS%" stats --report cuda_gpu_kern_sum "%OUT%\pp8k.nsys-rep" > s5\pp_nsys_8k_kern.txt 2>&1
"%NSYS%" stats --report cuda_gpu_kern_sum "%OUT%\pp512.nsys-rep" > s5\pp_nsys_512_kern.txt 2>&1
echo NSYS_DONE=%ERRORLEVEL%

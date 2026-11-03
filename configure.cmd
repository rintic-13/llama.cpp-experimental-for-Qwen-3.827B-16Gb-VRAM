@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main
cmake -B build-x64-windows-msvc-release -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DGGML_LLAMAFILE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FORCE_MMQ=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;q8_0-q4_0;q5_1-q5_1;q5_1-q4_1;q5_1-q4_0;q4_1-q4_1;f16-f16;q5_1-turbo4_0;q5_1-turbo3_0;turbo4_0-turbo3_0;q5_1-turbo5_0;turbo5_0-turbo5_0;turbo5_0-turbo4_0;turbo5_0-turbo3_0" > build-x64-windows-msvc-release\reconfigure.log 2>&1
echo CONFIGURE_EXIT=%ERRORLEVEL%
exit /b %ERRORLEVEL%

@echo off
rem reconfigure_trim.cmd - phase 1 of the fork trim: keep only CUDA+CPU targets, drop tests/examples.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d C:\Users\rinat\Downloads\1111111\llama.cpp-master
cmake -B build-x64-windows-msvc-release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > build-x64-windows-msvc-release\reconfigure_trim.log 2>&1
echo EXITCODE=%ERRORLEVEL%
exit /b %ERRORLEVEL%

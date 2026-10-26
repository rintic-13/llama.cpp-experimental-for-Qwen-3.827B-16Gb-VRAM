@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d C:\Users\rinat\Downloads\1111111\llama.cpp-master
rem CMake emits "unscanned" ninja rules for this project: header dependencies are NOT tracked, so an
rem edit that only touches ggml-moe-predict.hpp is silently not compiled and the next measurement
rem measures the previous binary. Touching the translation units that include it is what makes a
rem header change real. Pass "cuda" as the first argument to also rebuild ggml-cuda.cu, "fa" to also
rem rebuild fattn.cu (the FA_QUANTS set lives there), or "all" for both.
powershell -NoProfile -Command "(Get-Item 'ggml\src\ggml-backend.cpp').LastWriteTime = Get-Date; if ('%1' -eq 'cuda' -or '%1' -eq 'all') { (Get-Item 'ggml\src\ggml-cuda\ggml-cuda.cu').LastWriteTime = Get-Date }; if ('%1' -eq 'fa' -or '%1' -eq 'all') { (Get-Item 'ggml\src\ggml-cuda\fattn.cu').LastWriteTime = Get-Date }"
cmake --build build-x64-windows-msvc-release --target llama-server > build-x64-windows-msvc-release\build.log 2>&1
echo EXITCODE=%ERRORLEVEL%
rem without this the batch's own exit code is the echo's 0 and a failed build looks like a passed one
exit /b %ERRORLEVEL%

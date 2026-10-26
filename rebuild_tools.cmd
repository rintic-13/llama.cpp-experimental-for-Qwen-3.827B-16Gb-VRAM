@echo off
rem Rebuilds the tools this fork actually uses, against the current common.h / llama.h layouts.
rem rebuild.cmd only ever builds llama-server; every other binary in bin\ stays on the headers it was
rem compiled against, and a struct change in common_params or llama_context_params silently shifts their
rem field offsets. Pass the target names as arguments to build a different set.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d C:\Users\rinat\Downloads\1111111\llama.cpp-master
if "%1"=="" (set TARGETS=llama-fit-params llama-gguf llama-bench llama-cli llama-completion) else (set TARGETS=%*)
powershell -NoProfile -Command "Get-ChildItem 'tools' -Filter *.cpp -Recurse | ForEach-Object { (Get-Item $_.FullName).LastWriteTime = Get-Date }"
cmake --build build-x64-windows-msvc-release --target %TARGETS% > build-x64-windows-msvc-release\build_tools.log 2>&1
echo EXITCODE=%ERRORLEVEL%

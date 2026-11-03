$ErrorActionPreference = 'Stop'
# build_verify.ps1 - builds tools/verify_nvfp4_gpu.cpp (OFFLINE CPU-only bit-exactness verifier)
# Links against ggml-base.lib from the existing release build tree; touches no build artifact,
# safe to run while a GPU evaluation is in flight (never run cmake concurrently, though).
$root = 'G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main'
$vcvars = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "call `"$vcvars`" && cl /nologo /O2 /MT /EHsc /W0 `"$root\tools\verify_nvfp4_gpu.cpp`" /Fe:`"$root\build-x64-windows-msvc-release\bin\verify_nvfp4_gpu.exe`" /Fo:`"$root\build-x64-windows-msvc-release\bin\verify_nvfp4_gpu.obj`" /I `"$root\ggml\include`" /I `"$root\ggml\src`" /link `"$root\build-x64-windows-msvc-release\ggml\src\ggml-base.lib`" /LIBPATH:`"$root\build-x64-windows-msvc-release\ggml\src`""
if ($LASTEXITCODE -ne 0) { Write-Host "VERIFY BUILD FAILED ($LASTEXITCODE)"; exit 1 }
Remove-Item "$root\build-x64-windows-msvc-release\bin\verify_nvfp4_gpu.obj" -ErrorAction SilentlyContinue
Write-Host 'VERIFY BUILD OK'

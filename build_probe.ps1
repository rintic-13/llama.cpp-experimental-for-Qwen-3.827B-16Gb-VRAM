$ErrorActionPreference = 'Stop'
# build_probe.ps1 - builds tools/probe_nvfp4.cpp (OFFLINE CPU-only analyzer)
# Links against ggml-base.lib (import lib of ggml-base.dll) from the existing
# release build tree; does NOT touch any build artifact, safe to run during
# a GPU evaluation.
$root = 'G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main'
$vcvars = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$fields = New-Object System.Collections.Generic.List[string]
cmd /c "call `"$vcvars`" && cl /nologo /O2 /MT /EHsc /W0 `"$root\tools\probe_nvfp4.cpp`" /Fe:`"$root\build-x64-windows-msvc-release\bin\probe_nvfp4.exe`" /Fo:`"$root\build-x64-windows-msvc-release\bin\probe_nvfp4.obj`" /I `"$root\ggml\include`" /I `"$root\ggml\src`" /link `"$root\build-x64-windows-msvc-release\ggml\src\ggml-base.lib`" /LIBPATH:`"$root\build-x64-windows-msvc-release\ggml\src`""
if ($LASTEXITCODE -ne 0) { Write-Host "PROBE BUILD FAILED ($LASTEXITCODE)"; exit 1 }
Remove-Item "$root\build-x64-windows-msvc-release\bin\probe_nvfp4.obj" -ErrorAction SilentlyContinue
Write-Host 'PROBE BUILD OK'

@echo off
rem run_fa_turbo_ab.cmd - S1 turbo decode A/B: bitwise output match + timing (run when GPU is free)
rem ARCHIVED 2026-11-02: T5_turbo4q row retired with turbo4q from the FA path (SS12); the current
rem fa_turbo_ab_new.exe has no T5 snapshot - the T5 hash line below only matches the old S1 dumps.
setlocal
set ROOT=G:\llama.cpp-experimental-for-Qwen-3.827B-16Gb-VRAM-main
set TMPD=C:\Users\rinat\AppData\Local\Temp\opencode
cd /d "%ROOT%"

echo ================= nvidia-smi (check BEFORE running) =================
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo (if a foreign process occupies substantial VRAM - stop and report; do not kill anything)
echo.

set OLD=%ROOT%\build-x64-windows-msvc-release\bin\fa_turbo_ab_old.exe
set NEW=%ROOT%\build-x64-windows-msvc-release\bin\fa_turbo_ab_new.exe
if not exist "%OLD%" (echo MISSING fa_turbo_ab_old.exe - run rebuild_turbo_ab_both.cmd first & exit /b 1)
if not exist "%NEW%" (echo MISSING fa_turbo_ab_new.exe - run rebuild_turbo_ab_both.cmd first & exit /b 1)

rmdir /s /q "%TMPD%\turbo_ab_old" 2>nul
rmdir /s /q "%TMPD%\turbo_ab_new" 2>nul
mkdir "%TMPD%\turbo_ab_old"
mkdir "%TMPD%\turbo_ab_new"

echo ================= OLD build (LUT gathers, pre-S1) =================
"%OLD%" "%TMPD%\turbo_ab_old"
echo.
echo ================= NEW build (register-mux, S1) =================
"%NEW%" "%TMPD%\turbo_ab_new"
echo.
echo ================= bitwise dump comparison (all must be TRUE) =================
powershell -NoProfile -Command "foreach($n in 'T1_turbo4-t3','T2_q5_1-t3','T3_turbo4-t4','T4_turbo5-t3','T5_turbo4q-t4'){ $a=Join-Path '%TMPD%\turbo_ab_old' ($n+'.bin'); $b=Join-Path '%TMPD%\turbo_ab_new' ($n+'.bin'); $ha=(Get-FileHash $a -Algorithm SHA256).Hash; $hb=(Get-FileHash $b -Algorithm SHA256).Hash; '{0,-18} identical={1}  {2}' -f $n, ($ha -eq $hb), $ha }"
echo.
echo If identical=TRUE everywhere: PPL cannot move (bitwise proof). Compare med/t/s-equiv rows OLD vs NEW.
endlocal

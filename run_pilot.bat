@echo off
REM Phase 4 pilot: development data only (partition 0, outer fold 0, inner models). Never scores the outer test.
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe

echo [1/3] Benchmark: one run
%PY% scripts\run_train.py --mode pilot --bench 1 || goto :fail

echo [2/3] Full pilot: 45 runs (5 inner x 3 archs x 3 seeds), 2 workers on the GPU
%PY% scripts\run_train.py --mode pilot --workers 2 || goto :fail

echo [3/3] Pilot report
%PY% scripts\pilot_report.py > runs_pilot\pilot_report.txt 2>&1
type runs_pilot\pilot_report.txt
echo.
echo DONE. The pilot report is shown above and saved in runs_pilot\pilot_report.txt.
goto :eof

:fail
echo PILOT FAILED. See the error above and runs_pilot\logs\*.log.
exit /b 1

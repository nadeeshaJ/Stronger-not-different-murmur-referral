@echo off
REM Phase 5-7: freeze the protocol, full training, Study B corruption scoring, frozen analysis.
REM Resumable: double-click again after any interruption and it continues where it stopped.
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist logs mkdir logs

if not exist protocol\FREEZE.json (
  echo [1/4] Freezing protocol: tests + hashes
  %PY% scripts\make_freeze.py --reason "initial freeze" > logs\freeze.txt 2>&1 || goto :fail
  type logs\freeze.txt
) else (
  echo [1/4] Protocol already frozen
)

echo [2/4] Full training: 1,125 models (resumable)
%PY% scripts\run_train.py --mode full --workers 2 || goto :fail

echo [3/4] Study B: corrupted-audio scoring
%PY% scripts\run_corrupt.py > logs\corrupt.txt 2>&1 || goto :fail

echo [4/4] Frozen analysis (outer-test labels joined here only)
%PY% scripts\run_analysis.py > logs\analysis.txt 2>&1 || goto :fail
type logs\analysis.txt
echo.
echo ALL DONE. Results are in the results folder.
echo done> logs\ALL_DONE.txt
goto :eof

:fail
echo FAILED - see the logs folder for details.
echo failed> logs\FAILED.txt
exit /b 1

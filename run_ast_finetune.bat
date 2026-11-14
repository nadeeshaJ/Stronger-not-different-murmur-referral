@echo off
REM POST-HOC, EXPLORATORY (deviation 9): fine-tuned AST, partition 0, seed 0 (25 models).
REM Plan: protocol\deviation9_ast_finetune.md. Status in logs\FT_STATUS.txt. Safe to re-run: finished models are skipped.
cd /d "%~dp0"
if not exist logs mkdir logs
set PY=.venv\Scripts\python.exe
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
echo running > logs\FT_STATUS.txt
if not exist data\cache\ast400_segments.csv (
  echo [1/3] Computing input features
  %PY% post_hoc\ast_finetune.py features >> logs\ft.log 2>&1 || goto :fail
)
echo [2/3] Fine-tuning 25 models (several hours; keep the laptop awake and plugged in)
%PY% post_hoc\ast_finetune.py train >> logs\ft.log 2>&1 || goto :fail
echo [3/3] Analysis
%PY% post_hoc\ast_finetune.py analyse >> logs\ft.log 2>&1 || goto :fail
echo done > logs\FT_STATUS.txt
echo Finished. You can close this window.
pause
goto :eof
:fail
echo failed > logs\FT_STATUS.txt
echo FAILED - see logs\ft.log
pause
exit /b 1

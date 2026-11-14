@echo off
REM POST-HOC, EXPLORATORY pilot (deviation 8): frozen AST embeddings + small heads + analysis.
REM Plan: protocol\deviation8_fm_pilot.md. Writes logs\FM_STATUS.txt as running/done/failed.
cd /d "%~dp0"
if not exist logs mkdir logs
set PY=.venv\Scripts\python.exe
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
echo running > logs\FM_STATUS.txt
echo [1/4] Installing transformers (one time)
%PY% -m pip install "transformers>=4.48,<6" >> logs\fm_pilot.log 2>&1 || goto :fail
echo [2/4] Extracting AST embeddings (downloads the model once, about 350 MB; roughly 10-20 minutes)
%PY% post_hoc\fm_extract.py >> logs\fm_pilot.log 2>&1 || goto :fail
echo [3/4] Training 375 small heads (a few minutes)
%PY% post_hoc\fm_heads.py >> logs\fm_pilot.log 2>&1 || goto :fail
echo [4/4] Analysis
%PY% post_hoc\fm_pilot_analysis.py >> logs\fm_pilot.log 2>&1 || goto :fail
echo done > logs\FM_STATUS.txt
echo.
echo Pilot finished. Tell Claude it is done. You can close this window.
pause
goto :eof
:fail
echo failed > logs\FM_STATUS.txt
echo.
echo PILOT FAILED - see logs\fm_pilot.log. Tell Claude.
pause
exit /b 1

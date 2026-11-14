@echo off
REM External transportability scoring (addendum E1). Scores every frozen model on each external manifest.
REM Labels are NOT read here. Writes logs\EXTERNAL_STATUS.txt as running/done/failed.
cd /d "%~dp0"
if not exist logs mkdir logs
echo running > logs\EXTERNAL_STATUS.txt
for %%N in (zchsound pascal_b) do (
  if exist external\%%N_manifest.csv (
    echo Scoring %%N ...
    .venv\Scripts\python external\run_external.py score --manifest external\%%N_manifest.csv --name %%N >> logs\external_%%N.log 2>&1
    if errorlevel 1 ( echo failed %%N > logs\EXTERNAL_STATUS.txt & echo FAILED on %%N - see logs\external_%%N.log & pause & exit /b 1 )
  )
)
echo done > logs\EXTERNAL_STATUS.txt
echo All external scoring finished. You can close this window.
pause

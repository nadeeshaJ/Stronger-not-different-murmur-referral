@echo off
REM One-time setup on the training laptop. Run from the murmur-referral folder:  setup_windows.bat
setlocal
cd /d "%~dp0"

echo [1/6] Creating virtual environment .venv
if not exist .venv\Scripts\python.exe (
    py -3 -m venv .venv || python -m venv .venv || goto :fail
)
set PY=.venv\Scripts\python.exe

echo [2/6] Installing Python packages
%PY% -m pip install --upgrade pip || goto :fail
%PY% -m pip install -r requirements.txt psutil || goto :fail

echo [3/6] Installing PyTorch with CUDA (cu128; needs NVIDIA driver 570 or newer)
%PY% -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
    %PY% -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 || goto :fail
)

echo [4/6] Extracting CirCor (overwrites any partial extraction)
if not exist data\raw mkdir data\raw
tar -xf "..\circor-heart-sound-1.0.3.zip" -C data\raw || goto :fail

echo [5/6] Phase 1 build: checksums, cohort, folds, waveform cache
%PY% scripts\phase1_build.py || goto :fail
%PY% -m pytest -q -p no:cacheprovider tests || goto :fail

echo [6/6] GPU check
%PY% scripts\env_check.py > data\derived\env_check.txt 2>&1
type data\derived\env_check.txt

echo.
echo DONE. Environment summary saved in data\derived\env_check.txt.
goto :eof

:fail
echo.
echo SETUP FAILED at the step above. See the error message for details.
exit /b 1

@echo off
setlocal

cd /d "%~dp0"

echo.
echo ============================================================
echo      MARBYCORE - XTTS-v2 AUDIO GENERATOR (BLACKWELL)
echo ============================================================
echo.

:: 1. SEARCH FOR SYSTEM PYTHON (MUST BE 3.10 FOR COQUI)
py -3.10 --version >nul 2>&1
if not errorlevel 1 goto FOUND_PY

python --version 2>&1 | findstr " 3.10." >nul
if not errorlevel 1 goto FOUND_PYTHON

echo [ERROR] Python 3.10 is required for Coqui XTTS-v2!
echo We found a different Python version on your system.
echo Please install Python 3.10 from python.org and ensure it is in PATH.
pause
exit /b 1

:FOUND_PY
set "PYTHON_EXE=py -3.10"
goto CHECK_VENV

:FOUND_PYTHON
set "PYTHON_EXE=python"
goto CHECK_VENV

:CHECK_VENV
:: 2. AUTOMATIC ENVIRONMENT SETUP
if not exist ".venv\Scripts\activate.bat" goto CREATE_VENV
if not exist ".venv\.installed_v1" goto INSTALL_DEPS
goto ACTIVATE_VENV

:CREATE_VENV
echo [SETUP] Creating isolated environment using %PYTHON_EXE%...
%PYTHON_EXE% -m venv .venv
if errorlevel 1 goto ERROR_VENV
goto INSTALL_DEPS

:INSTALL_DEPS
echo [SETUP] Installing technical dependencies (this may take a few minutes)...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo [SETUP] Installing Blackwell (sm_120) PyTorch Nightly Core...
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
if errorlevel 1 goto ERROR_DEPS

echo [SETUP] Installing Coqui XTTS and remaining dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto ERROR_DEPS

:: Create a flag file to signal successful installation
echo ALREADY_INSTALLED > ".venv\.installed_v1"
echo [SETUP] Installation complete!
goto LAUNCH

:ERROR_VENV
echo [ERROR] Failed to create virtual environment.
pause
exit /b 1

:ERROR_DEPS
echo [ERROR] Failed to install dependencies.
pause
exit /b 1

:ACTIVATE_VENV
echo [1/3] Activating environment...
call .venv\Scripts\activate.bat

:LAUNCH
:: 3. VERIFY SCRIPT
if not exist "app.py" (
    echo [ERROR] Critical file missing: app.py
    pause
    exit /b 1
)

:: 4. START ENGINE
echo [2/3] Verifying environment...
echo [3/3] Launching Engine...
echo.
echo ------------------------------------------------------------
echo The dashboard will open automatically in your browser.
echo Keep this window open.
echo ------------------------------------------------------------
echo.

python app.py

if errorlevel 1 (
    echo.
    echo [ERROR] The engine stopped unexpectedly.
    pause
)

endlocal
exit /b 0
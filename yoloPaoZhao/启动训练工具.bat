@echo off
chcp 65001 >nul
setlocal

:: Change to the directory where this bat file lives (the yoloPaoZhao project dir)
cd /d "%~dp0"

:: Verify the entry script exists
if not exist "train_ui.py" (
    echo [ERROR] train_ui.py not found. Place this bat in the yoloPaoZhao folder.
    pause
    exit /b 1
)

:: ===== Auto-detect Python interpreter =====
set "PYTHON_EXE="

:: 1) Prefer project-local venv
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"

:: 2) Known working Anaconda env
if not defined PYTHON_EXE if exist "D:\AI\Anaconda3\python.exe" set "PYTHON_EXE=D:\AI\Anaconda3\python.exe"

:: 3) Python38 (used by LabelImg)
if not defined PYTHON_EXE if exist "C:\Program Files\Python38\python.exe" set "PYTHON_EXE=C:\Program Files\Python38\python.exe"

:: 4) Fallback to python on PATH
if not defined PYTHON_EXE (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        set "PYTHON_EXE=%%i"
        goto :found_python
    )
)
:found_python

if not defined PYTHON_EXE (
    echo [ERROR] Python not found. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

:: ===== Check streamlit dependency =====
"%PYTHON_EXE%" -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [INFO] streamlit not found, installing dependencies - first run may take a while...
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies. Run manually: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo.
echo ============================================
echo   YOLO11 Training Tool - Starting...
echo   Python: %PYTHON_EXE%
echo   Browser will open: http://localhost:8501
echo   Close this window to stop the tool.
echo ============================================
echo.

:: ===== Kill any leftover streamlit on port 8501 =====
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501" ^| findstr LISTENING 2^>nul') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: ===== Launch a background watcher that opens the browser when server is ready =====
start "YOLO11-Browser-Watcher" /MIN powershell -NoProfile -Command ^
    "$u='http://localhost:8501';" ^
    "for($i=0;$i -lt 30;$i++){" ^
    "  try{ Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 2 | Out-Null; Start-Process $u; exit 0 } catch {}" ^
    "  Start-Sleep -Seconds 1" ^
    "}" ^
    "Write-Host 'Streamlit not ready - open browser manually:' $u"

:: ===== Launch Streamlit (blocking; closing this window stops the tool) =====
"%PYTHON_EXE%" -m streamlit run train_ui.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

echo.
echo [INFO] Streamlit process ended.
pause

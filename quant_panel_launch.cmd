@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHON_EXE=C:\Python311\python.exe"
set "APP_FILE=%~dp0app.py"
set "PORT=8501"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "STREAMLIT_SERVER_HEADLESS=true"

if not exist "%APP_FILE%" (
    echo [ERROR] app.py not found: %APP_FILE%
    pause
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python 3.11 not found: %PYTHON_EXE%
    pause
    exit /b 1
)

echo Checking old process on port %PORT%...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    if not "%%P"=="0" (
        taskkill /PID %%P /F >nul 2>&1
    )
)

timeout /t 1 /nobreak >nul
echo Starting quant panel...
start "" "http://localhost:%PORT%"
"%PYTHON_EXE%" -m streamlit run "%APP_FILE%" --server.port %PORT% --server.headless true

if errorlevel 1 (
    echo.
    echo [ERROR] Panel failed to start.
    echo Try:
    echo   "%PYTHON_EXE%" -m pip install -r requirements.txt
    pause
    exit /b 1
)

endlocal

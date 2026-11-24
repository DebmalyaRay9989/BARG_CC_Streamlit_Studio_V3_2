@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    where python >nul 2>nul || (echo Install Python 3.11 or 3.12 first. & pause & exit /b 1)
    python -m venv .venv
    if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" setup_check.py
if errorlevel 1 (
    echo Installing or repairing required dependencies...
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 goto fail
    ".venv\Scripts\python.exe" setup_check.py --mark
    if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --runner.magicEnabled=false --server.address 127.0.0.1 --server.port 8501
if errorlevel 1 goto fail
exit /b 0
:fail
echo Setup or application failed. Review the error above.
pause
exit /b 1

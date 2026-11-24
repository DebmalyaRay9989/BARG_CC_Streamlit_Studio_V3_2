@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run start_windows.bat first to install the environment.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m unittest discover -s tests -v
if errorlevel 1 goto fail
".venv\Scripts\python.exe" smoke_ui.py
if errorlevel 1 goto fail
echo All local automated checks passed.
pause
exit /b 0
:fail
echo A check failed. Review the terminal output above.
pause
exit /b 1

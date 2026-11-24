@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Run start_windows.bat first to install dependencies. & pause & exit /b 1)
".venv\Scripts\python.exe" run_batch.py data/demo_mixed_documents.zip --profile Quick --dropout 0.1 --output "demo_output_%RANDOM%_%RANDOM%"
pause

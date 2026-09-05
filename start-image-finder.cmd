@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Image Finder's project environment was not found.
  echo Expected: %~dp0.venv\Scripts\pythonw.exe
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "run_image_finder.py"
endlocal

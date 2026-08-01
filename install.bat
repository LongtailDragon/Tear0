@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if errorlevel 1 (
  echo.
  echo Tear0 install failed.
  pause
  exit /b 1
)
echo.
echo Tear0 install finished.
pause

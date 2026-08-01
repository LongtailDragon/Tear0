@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m tear0.cli %*
) else (
  uv run tear0 %*
)
echo.
echo Tear0 closed. Press any key to exit.
pause >nul

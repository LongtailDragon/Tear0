param(
  [switch]$SkipModels,
  [switch]$BuildExe
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "=== Tear0 installer ==="
Write-Host "Project: $ProjectRoot"

function Test-Command($Name) {
  $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

if (-not (Test-Command uv)) {
  Write-Host "uv was not found. Installing uv for the current user..."
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  $env:Path = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
}

if (-not (Test-Command uv)) {
  throw "uv installation did not put uv on PATH. Restart this terminal and run install.ps1 again."
}

if (-not (Test-Command hermes)) {
  Write-Warning "Hermes Agent was not found on PATH. Tear0 can install, but voice commands cannot be sent until 'hermes' works from a terminal."
}

if (-not (Test-Command ffmpeg)) {
  Write-Warning "ffmpeg was not found on PATH. faster-whisper may need it for some audio formats. Tear0 records WAV directly, so this is not usually fatal."
}

Write-Host "Creating/updating Python environment..."
uv sync --extra test --extra build

$InstallerArgs = @('-m', 'tear0.installer', '--project-root', $ProjectRoot)
if ($SkipModels) { $InstallerArgs += '--skip-models' }
Write-Host "Inspecting hardware and downloading local Kokoro assets..."
uv run python @InstallerArgs

if ($BuildExe) {
  Write-Host "Building optional Tear0.exe with PyInstaller..."
  uv run pyinstaller --paths src --noconfirm --onefile --name Tear0 --console tear0_launcher.py
}

Write-Host ""
Write-Host "Install complete."
Write-Host "Start Tear0 with: $ProjectRoot\Tear0.bat"
if (Test-Path "$ProjectRoot\dist\Tear0.exe") {
  Write-Host "Or run executable: $ProjectRoot\dist\Tear0.exe"
}

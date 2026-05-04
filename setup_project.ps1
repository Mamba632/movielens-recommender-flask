$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python is not installed or not on PATH. Install Python 3.11+, then run this script again."
}

if (-not (Test-Path ".\ml-32m\movies.csv")) {
    $zipPath = Join-Path $PSScriptRoot "ml-32m.zip"
    Write-Host "[INFO] MovieLens data not found. Downloading ml-32m dataset..."
    Invoke-WebRequest "https://files.grouplens.org/datasets/movielens/ml-32m.zip" -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $PSScriptRoot -Force
}

$venvPython = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[INFO] Creating virtual environment..."
    python -m venv .venv
}

try {
    & $venvPython --version | Out-Host
}
catch {
    $backup = ".venv_broken_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Write-Host "[INFO] Existing .venv is broken. Moving it to $backup..."
    Move-Item -LiteralPath ".\.venv" -Destination $backup
    python -m venv .venv
}

Write-Host "[INFO] Installing Python dependencies..."
& $venvPython -m pip install -r requirements.txt

Write-Host "[INFO] Checking installed packages..."
& $venvPython -m pip check

Write-Host "[INFO] Running tests..."
& $venvPython -m pytest -q

Write-Host ""
Write-Host "[OK] Setup complete. Start the app with:"
Write-Host "     .\run_app.ps1"

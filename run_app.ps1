$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe") -or -not (Test-Path ".\ml-32m\movies.csv")) {
    Write-Host "[INFO] Environment or dataset missing. Running setup first..."
    & ".\setup_project.ps1"
}

Write-Host "[INFO] Starting Flask app..."
Write-Host "[INFO] Open http://localhost:5000 in your browser."
Write-Host "[INFO] Loading the movie model can take 1-2 minutes on first startup."
& ".\.venv\Scripts\python.exe" app.py

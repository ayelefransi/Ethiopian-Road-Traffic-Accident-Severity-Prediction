param()

Write-Host ""
Write-Host "  ██████╗ ████████╗ █████╗     ██████╗ ██████╗ ███████╗██████╗ "
Write-Host "  ██╔══██╗╚══██╔══╝██╔══██╗    ██╔══██╗██╔══██╗██╔════╝██╔══██╗"
Write-Host "  ██████╔╝   ██║   ███████║    ██████╔╝██████╔╝█████╗  ██║  ██║"
Write-Host "  ██╔══██╗   ██║   ██╔══██║    ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║"
Write-Host "  ██║  ██║   ██║   ██║  ██║    ██║     ██║  ██║███████╗██████╔╝"
Write-Host "  ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ "
Write-Host ""
Write-Host "  Ethiopian Road Traffic Accident Severity Predictor"
Write-Host "  Stacking Ensemble · 90.82% Accuracy · ROC-AUC 0.9822"
Write-Host "────────────────────────────────────────────────────────"

# Check Python
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "ERROR: Python not found. Please install Python 3.9+."
    exit 1
}

# Check Node
if (!(Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Error "ERROR: npm not found. Please install Node.js 18+."
    exit 1
}

# Backend setup
Write-Host ""
Write-Host "[1/4] Installing Python dependencies..." -ForegroundColor Cyan
Set-Location "$PSScriptRoot\backend"
python -m pip install -r requirements.txt -q
Write-Host "      ✓ Backend dependencies installed" -ForegroundColor Green

# Start backend
Write-Host "[2/4] Starting FastAPI backend on http://localhost:8000 ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit","-Command","cd `"$PSScriptRoot\backend`"; python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload" -WindowStyle Normal
Write-Host "      ✓ Backend running in a new window" -ForegroundColor Green
Start-Sleep -Seconds 3

# Frontend setup
Write-Host "[3/4] Installing Node dependencies..." -ForegroundColor Cyan
Set-Location "$PSScriptRoot\frontend"
npm install --silent
Write-Host "      ✓ Frontend dependencies installed" -ForegroundColor Green

# Start frontend
Write-Host "[4/4] Starting React frontend on http://localhost:3000 ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit","-Command","cd `"$PSScriptRoot\frontend`"; npm start" -WindowStyle Normal
Write-Host "      ✓ Frontend running in a new window" -ForegroundColor Green

Write-Host ""
Write-Host "────────────────────────────────────────────────────────"
Write-Host "  App is running!"
Write-Host "  Frontend : http://localhost:3000"
Write-Host "  Backend  : http://localhost:8000"
Write-Host "  API Docs : http://localhost:8000/docs"
Write-Host "────────────────────────────────────────────────────────"
Write-Host "  Close the two newly opened PowerShell windows to stop."
Write-Host ""

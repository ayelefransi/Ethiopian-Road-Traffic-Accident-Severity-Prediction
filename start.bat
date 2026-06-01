@echo off
echo.
echo   Ethiopian RTA Severity Predictor
echo   Stacking Ensemble - 90.82%% Accuracy
echo ================================================

:: Backend
echo [1/4] Installing Python dependencies...
cd backend
python -m pip install -r requirements.txt -q
echo       Backend dependencies installed

echo [2/4] Starting FastAPI backend on http://localhost:8000 ...
start "RTA Backend" cmd /k "python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 >nul
echo       Backend started

:: Frontend
echo [3/4] Installing Node dependencies...
cd ..\frontend
call npm install --silent
echo       Frontend dependencies installed

echo [4/4] Starting React app on http://localhost:3000 ...
start "RTA Frontend" cmd /k "npm start"

echo.
echo ================================================
echo   Frontend : http://localhost:3000
echo   Backend  : http://localhost:8000
echo   API Docs : http://localhost:8000/docs
echo ================================================
echo   Close both terminal windows to stop.
pause

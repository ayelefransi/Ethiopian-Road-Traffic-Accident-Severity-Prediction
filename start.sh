#!/bin/bash
# ── Ethiopian RTA Severity Predictor — Start Script ──────────────────────────
set -e

echo ""
echo "  ██████╗ ████████╗ █████╗     ██████╗ ██████╗ ███████╗██████╗ "
echo "  ██╔══██╗╚══██╔══╝██╔══██╗    ██╔══██╗██╔══██╗██╔════╝██╔══██╗"
echo "  ██████╔╝   ██║   ███████║    ██████╔╝██████╔╝█████╗  ██║  ██║"
echo "  ██╔══██╗   ██║   ██╔══██║    ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║"
echo "  ██║  ██║   ██║   ██║  ██║    ██║     ██║  ██║███████╗██████╔╝"
echo "  ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ "
echo ""
echo "  Ethiopian Road Traffic Accident Severity Predictor"
echo "  Stacking Ensemble · 90.82% Accuracy · ROC-AUC 0.9822"
echo "────────────────────────────────────────────────────────"

# Check Python
if ! command -v python3 &> /dev/null; then
  echo "ERROR: Python 3 not found. Please install Python 3.9+."
  exit 1
fi

# Check Node
if ! command -v node &> /dev/null; then
  echo "ERROR: Node.js not found. Please install Node.js 18+."
  exit 1
fi

# Backend setup
echo ""
echo "[1/4] Installing Python dependencies..."
cd backend
python3 -m pip install -r requirements.txt -q
echo "      ✓ Backend dependencies installed"

# Start backend
echo "[2/4] Starting FastAPI backend on http://localhost:8000 ..."
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "      ✓ Backend running (PID $BACKEND_PID)"
sleep 2

# Frontend setup
echo "[3/4] Installing Node dependencies..."
cd ../frontend
npm install --silent
echo "      ✓ Frontend dependencies installed"

# Start frontend
echo "[4/4] Starting React frontend on http://localhost:3000 ..."
npm start &
FRONTEND_PID=$!
echo "      ✓ Frontend running (PID $FRONTEND_PID)"

echo ""
echo "────────────────────────────────────────────────────────"
echo "  App is running!"
echo "  Frontend : http://localhost:3000"
echo "  Backend  : http://localhost:8000"
echo "  API Docs : http://localhost:8000/docs"
echo "────────────────────────────────────────────────────────"
echo "  Press Ctrl+C to stop both servers"
echo ""

# Wait and handle stop
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Done.'; exit 0" INT
wait

# AI-Powered Fraud Detection System Launcher
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Starting AI Fraud Detection & Anomaly Prevention System " -ForegroundColor Green
Write-Host " Ensemble ML: Isolation Forest + Random Forest + Heuristics" -ForegroundColor Yellow
Write-Host " UI Design System: heydamir/unslop specifications" -ForegroundColor Magenta
Write-Host "==========================================================" -ForegroundColor Cyan

Start-Process "http://127.0.0.1:8000"
python backend/server.py

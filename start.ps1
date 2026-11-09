# AI-Powered Fraud Detection System Launcher (Indian Banking Edition)
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " Starting AI Fraud Detection & Anomaly Prevention System       " -ForegroundColor Green
Write-Host " Indian Banking Edition: RuPay, UPI, IMPS, Cards (INR ₹)        " -ForegroundColor Yellow
Write-Host " Ensemble ML: Isolation Forest + Random Forest + Haversine      " -ForegroundColor Magenta
Write-Host " UI Design Standard: heydamir/unslop specifications             " -ForegroundColor White
Write-Host "================================================================" -ForegroundColor Cyan

Start-Process "http://127.0.0.1:8000"
python backend/server.py

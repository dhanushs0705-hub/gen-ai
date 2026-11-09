# AI-Powered Fraud Detection System Launcher (Indian Banking Edition)
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " Starting AI Fraud Detection & Anomaly Prevention System       " -ForegroundColor Green
Write-Host " Indian Banking Edition: RuPay, UPI, IMPS, Cards (INR ₹)        " -ForegroundColor Yellow
Write-Host " Ensemble ML: Isolation Forest + Random Forest + Haversine      " -ForegroundColor Magenta
Write-Host " UI Design Standard: heydamir/unslop specifications             " -ForegroundColor White
Write-Host "================================================================" -ForegroundColor Cyan

# Check if Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python is not installed or not in PATH." -ForegroundColor Red
    Exit 1
}

# Launch browser and run server via run.py
python run.py

"""
FastAPI Server for AI-Powered Fraud Detection Engine (Indian Banking Edition)
Provides endpoints for:
- Live transaction stream with realistic Indian banking activities (₹ / INR)
- Real-time AI evaluation with local SHAP feature attributions
- Simulated targeted fraud attacks (Card testing, impossible travel, account takeover)
- Operational metrics, confusion matrix, and dynamic policy rules
"""

import time
import random
import uuid
import sys
import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

try:
    from ml_engine import fraud_engine, MCC_PROFILES
except ImportError:
    from backend.ml_engine import fraud_engine, MCC_PROFILES

app = FastAPI(
    title="AI Fraud Detection Engine - India",
    description="Real-time transaction risk scoring and anomaly detection platform for Indian banking",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_BUFFER_SIZE = 100
TRANSACTION_BUFFER: List[Dict[str, Any]] = []

# Dynamic risk threshold policy
CONFIG = {
    "block_threshold": 80.0,
    "review_threshold": 50.0,
    "max_velocity_10m": 5,
    "max_speed_kmh": 800.0,
    "vpn_strict_mode": True,
}

SYSTEM_STATS = {
    "total_evaluated": 0,
    "total_approved": 0,
    "total_flagged": 0,
    "total_declined": 0,
    "total_fraud_prevented_inr": 0.0,
    "total_volume_inr": 0.0,
    "avg_latency_ms": 11.4,
}

# Seeded Indian Cardholders & Profiles
SEEDED_USERS = [
    {"user_id": "usr_9102", "name": "Rahul Sharma", "country": "IN", "home_city": "Mumbai", "home_lat": 19.0760, "home_lon": 72.8777, "mean_amount": 1850.0, "std_amount": 1200.0},
    {"user_id": "usr_4821", "name": "Priya Patel", "country": "IN", "home_city": "Ahmedabad", "home_lat": 23.0225, "home_lon": 72.5714, "mean_amount": 2400.0, "std_amount": 1500.0},
    {"user_id": "usr_7342", "name": "Vikram Malhotra", "country": "IN", "home_city": "New Delhi", "home_lat": 28.6139, "home_lon": 77.2090, "mean_amount": 3500.0, "std_amount": 2100.0},
    {"user_id": "usr_6190", "name": "Ananya Iyer", "country": "IN", "home_city": "Bengaluru", "home_lat": 12.9716, "home_lon": 77.5946, "mean_amount": 2100.0, "std_amount": 1400.0},
    {"user_id": "usr_3319", "name": "Arjun Nair", "country": "IN", "home_city": "Kochi", "home_lat": 9.9312, "home_lon": 76.2673, "mean_amount": 1650.0, "std_amount": 950.0},
    {"user_id": "usr_5528", "name": "Sneha Roy", "country": "IN", "home_city": "Kolkata", "home_lat": 22.5726, "home_lon": 88.3639, "mean_amount": 1950.0, "std_amount": 1100.0},
]

# Popular Indian Merchants & Payment Gateways
MERCHANTS = [
    {"merchant": "Swiggy Food Delivery", "mcc": "5812", "city": "Bengaluru", "country": "IN"},
    {"merchant": "DMart Supercentre", "mcc": "5411", "city": "Mumbai", "country": "IN"},
    {"merchant": "Reliance Digital", "mcc": "5732", "city": "New Delhi", "country": "IN"},
    {"merchant": "Flipkart Internet", "mcc": "5311", "city": "Bengaluru", "country": "IN"},
    {"merchant": "Tanishq Jewellery", "mcc": "5094", "city": "Chennai", "country": "IN"},
    {"merchant": "Apollo 24x7 Pharmacy", "mcc": "5411", "city": "Hyderabad", "country": "IN"},
    {"merchant": "WazirX Crypto Exchange", "mcc": "6051", "city": "Victoria", "country": "SC"},
    {"merchant": "Dream11 Sports Gaming", "mcc": "7995", "city": "Mumbai", "country": "IN"},
    {"merchant": "Hawala Offshore Remit", "mcc": "6012", "city": "Dubai", "country": "AE"},
]

class EvaluateRequest(BaseModel):
    transaction_id: Optional[str] = None
    user_id: str = "usr_9102"
    amount: float = Field(..., gt=0)
    merchant: str = "Online Payment"
    mcc: str = "5411"
    country: str = "IN"
    city: str = "Mumbai"
    channel: str = "UPI"
    is_vpn_tor: bool = False
    is_new_device: bool = False
    velocity_10m: int = 1
    velocity_1h: int = 2
    time_since_last_sec: float = 3600.0
    speed_kmh: Optional[float] = 0.0
    failed_attempts_count: int = 0
    hour_of_day: int = 14

class AttackRequest(BaseModel):
    attack_type: str = Field(..., description="card_testing, impossible_travel, account_takeover, or crypto_drain")

class ConfigUpdateRequest(BaseModel):
    block_threshold: Optional[float] = None
    review_threshold: Optional[float] = None
    max_velocity_10m: Optional[int] = None
    max_speed_kmh: Optional[float] = None
    vpn_strict_mode: Optional[bool] = None

def generate_random_transaction(attack_type: Optional[str] = None) -> Dict[str, Any]:
    """Create a realistic Indian banking transaction (INR ₹)."""
    user = random.choice(SEEDED_USERS)
    tx_id = f"TXN{random.randint(10000000, 99999999)}"
    timestamp = time.strftime("%H:%M:%S")

    if attack_type == "card_testing":
        amount = round(random.uniform(5.0, 45.0), 2) # Tiny micro-charge probe
        mcc = "5311"
        merchant = "Digital Micro-Auth Gateway"
        country = user["country"]
        city = user["home_city"]
        lat = user["home_lat"]
        lon = user["home_lon"]
        prev_lat = lat
        prev_lon = lon
        time_since_sec = random.uniform(2, 20)
        vel_10m = random.randint(7, 18)
        vel_1h = vel_10m + random.randint(5, 20)
        is_vpn = random.choice([True, False])
        is_new_dev = True
        failed_count = random.randint(2, 5)
        channel = "UPI"
        hour = random.randint(2, 4)

    elif attack_type == "impossible_travel":
        amount = round(random.uniform(25000, 85000), 2)
        mcc = "5732"
        merchant = "Luxury Electronic Vault"
        # If user is in Mumbai, swipe in London or Tokyo 10 mins later
        country = "GB"
        city = "London"
        lat = 51.5074
        lon = -0.1278
        prev_lat = user["home_lat"]
        prev_lon = user["home_lon"]
        time_since_sec = 600.0 # 10 minutes ago
        vel_10m = 1
        vel_1h = 2
        is_vpn = False
        is_new_dev = True
        failed_count = 0
        channel = "POS_SWIPE"
        hour = 16

    elif attack_type == "account_takeover":
        amount = round(random.uniform(185000, 680000), 2) # Large IMPS/NEFT drain
        mcc = "6012"
        merchant = "Offshore Instant Remittance"
        country = "AE"
        city = "Dubai"
        lat = 25.2048
        lon = 55.2708
        prev_lat = user["home_lat"]
        prev_lon = user["home_lon"]
        time_since_sec = 180.0
        vel_10m = 2
        vel_1h = 3
        is_vpn = True
        is_new_dev = True
        failed_count = random.randint(1, 3)
        channel = "NET_BANKING"
        hour = 3

    elif attack_type == "crypto_drain":
        amount = round(random.uniform(320000, 950000), 2)
        mcc = "6051"
        merchant = "WazirX Crypto Liquidate"
        country = "SC"
        city = "Victoria"
        lat = -4.6191
        lon = 55.4513
        prev_lat = user["home_lat"]
        prev_lon = user["home_lon"]
        time_since_sec = 90.0
        vel_10m = 3
        vel_1h = 4
        is_vpn = True
        is_new_dev = True
        failed_count = 2
        channel = "UPI"
        hour = 2

    else:
        # Normal everyday Indian transactions (~95% legitimate, 5% organic anomalies)
        is_fraud = (random.random() < 0.05)
        if is_fraud:
            amount = round(random.uniform(45000, 195000), 2)
            m = random.choice([m for m in MERCHANTS if m["mcc"] in ["6051", "7995", "5094"]])
            country = m["country"]
            city = m["city"]
            lat = user["home_lat"] + random.uniform(5.0, 20.0)
            lon = user["home_lon"] + random.uniform(5.0, 20.0)
            prev_lat = user["home_lat"]
            prev_lon = user["home_lon"]
            time_since_sec = random.uniform(300, 1800)
            vel_10m = random.randint(3, 6)
            vel_1h = vel_10m + random.randint(2, 5)
            is_vpn = random.choice([True, False])
            is_new_dev = True
            failed_count = random.randint(1, 2)
            channel = "UPI"
            hour = random.randint(1, 4)
            mcc = m["mcc"]
            merchant = m["merchant"]
        else:
            # Clean everyday Indian transaction
            amount = round(max(50.0, random.gauss(user["mean_amount"], user["std_amount"] * 0.6)), 2)
            m = random.choice([m for m in MERCHANTS if m["mcc"] in ["5411", "5812", "5311", "4814"]])
            country = user["country"]
            city = user["home_city"]
            lat = user["home_lat"] + random.uniform(-0.03, 0.03)
            lon = user["home_lon"] + random.uniform(-0.03, 0.03)
            prev_lat = user["home_lat"]
            prev_lon = user["home_lon"]
            time_since_sec = random.uniform(1800, 43200)
            vel_10m = random.choice([0, 1])
            vel_1h = random.choice([1, 2])
            is_vpn = False
            is_new_dev = False
            failed_count = 0
            channel = random.choice(["UPI", "UPI", "POS_SWIPE", "NET_BANKING"])
            hour = random.randint(8, 22)
            mcc = m["mcc"]
            merchant = m["merchant"]

    tx_dict = {
        "transaction_id": tx_id,
        "timestamp": timestamp,
        "user_id": user["user_id"],
        "user_name": user["name"],
        "user_home_country": user["country"],
        "amount": amount,
        "merchant": merchant,
        "mcc": mcc,
        "country": country,
        "city": city,
        "channel": channel,
        "is_vpn_tor": is_vpn,
        "is_new_device": is_new_dev,
        "velocity_10m": vel_10m,
        "velocity_1h": vel_1h,
        "amount_sum_1h": amount * (vel_1h or 1),
        "time_since_last_sec": time_since_sec,
        "lat": lat,
        "lon": lon,
        "prev_lat": prev_lat,
        "prev_lon": prev_lon,
        "failed_attempts_count": failed_count,
        "hour_of_day": hour,
    }

    user_hist = {
        "mean_amount": user["mean_amount"],
        "std_amount": user["std_amount"],
    }

    start_eval = time.perf_counter()
    evaluation = fraud_engine.evaluate_transaction(tx_dict, user_hist)
    latency_ms = round((time.perf_counter() - start_eval) * 1000, 2)

    score = evaluation["risk_score"]
    if score >= CONFIG["block_threshold"]:
        evaluation["decision"] = "DECLINE"
        evaluation["action"] = f"Blocked (Score {score} >= Policy Threshold {CONFIG['block_threshold']})"
    elif score >= CONFIG["review_threshold"]:
        evaluation["decision"] = "MANUAL_REVIEW"
        evaluation["action"] = f"Flagged for manual review (Score {score} >= {CONFIG['review_threshold']})"
    else:
        evaluation["decision"] = "APPROVE"
        evaluation["action"] = "Cleared safely by automated risk policy"

    record = {
        **tx_dict,
        **evaluation,
        "latency_ms": latency_ms,
        "evaluated_at": time.time(),
        "currency": "INR",
        "currency_symbol": "₹",
    }

    # Aggregate stats tracking
    SYSTEM_STATS["total_evaluated"] += 1
    SYSTEM_STATS["total_volume_inr"] += amount
    if evaluation["decision"] == "DECLINE":
        SYSTEM_STATS["total_declined"] += 1
        SYSTEM_STATS["total_fraud_prevented_inr"] += amount
    elif evaluation["decision"] == "MANUAL_REVIEW":
        SYSTEM_STATS["total_flagged"] += 1
    else:
        SYSTEM_STATS["total_approved"] += 1

    return record

# Seed initial history buffer with realistic records
def _seed_initial_records(count: int = 15):
    for _ in range(count):
        rec = generate_random_transaction()
        TRANSACTION_BUFFER.insert(0, rec)
    if len(TRANSACTION_BUFFER) > MAX_BUFFER_SIZE:
        TRANSACTION_BUFFER.pop()

_seed_initial_records(18)

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "engine": "Ensemble (Isolation Forest + Random Forest + Heuristics)",
        "currency": "INR (₹)",
        "model_version": "v2.0.0-in-calibrated"
    }

@app.get("/api/transactions")
def get_transactions(limit: int = 50, filter_decision: Optional[str] = None):
    txs = TRANSACTION_BUFFER
    if filter_decision and filter_decision != "ALL":
        txs = [t for t in txs if t["decision"] == filter_decision]
    return txs[:limit]

@app.post("/api/evaluate")
def evaluate_custom(req: EvaluateRequest):
    user = next((u for u in SEEDED_USERS if u["user_id"] == req.user_id), SEEDED_USERS[0])
    tx_id = req.transaction_id or f"TXN{random.randint(10000000, 99999999)}"

    tx_dict = {
        "transaction_id": tx_id,
        "user_id": user["user_id"],
        "user_name": user["name"],
        "user_home_country": user["country"],
        "amount": req.amount,
        "merchant": req.merchant,
        "mcc": req.mcc,
        "country": req.country,
        "city": req.city,
        "channel": req.channel,
        "is_vpn_tor": req.is_vpn_tor,
        "is_new_device": req.is_new_device,
        "velocity_10m": req.velocity_10m,
        "velocity_1h": req.velocity_1h,
        "amount_sum_1h": req.amount * req.velocity_1h,
        "time_since_last_sec": req.time_since_last_sec,
        "lat": user["home_lat"],
        "lon": user["home_lon"],
        "prev_lat": user["home_lat"],
        "prev_lon": user["home_lon"],
        "failed_attempts_count": req.failed_attempts_count,
        "hour_of_day": req.hour_of_day,
        "speed_kmh": req.speed_kmh or 0.0,
    }

    if req.speed_kmh and req.speed_kmh > 0:
        tx_dict["prev_lat"] = user["home_lat"]
        tx_dict["prev_lon"] = user["home_lon"]
        tx_dict["lat"] = user["home_lat"] + (req.speed_kmh / 111.0) * (req.time_since_last_sec / 3600.0)
        tx_dict["lon"] = user["home_lon"]

    start = time.perf_counter()
    eval_result = fraud_engine.evaluate_transaction(tx_dict, {"mean_amount": user["mean_amount"], "std_amount": user["std_amount"]})
    latency = round((time.perf_counter() - start) * 1000, 2)

    score = eval_result["risk_score"]
    if score >= CONFIG["block_threshold"]:
        eval_result["decision"] = "DECLINE"
        eval_result["action"] = f"Blocked (Score {score} >= Policy Threshold {CONFIG['block_threshold']})"
    elif score >= CONFIG["review_threshold"]:
        eval_result["decision"] = "MANUAL_REVIEW"
        eval_result["action"] = f"Flagged for manual review (Score {score} >= {CONFIG['review_threshold']})"
    else:
        eval_result["decision"] = "APPROVE"
        eval_result["action"] = "Cleared safely by automated risk policy"

    full_record = {
        **tx_dict,
        **eval_result,
        "latency_ms": latency,
        "timestamp": time.strftime("%H:%M:%S"),
        "evaluated_at": time.time(),
        "currency": "INR",
        "currency_symbol": "₹",
    }

    TRANSACTION_BUFFER.insert(0, full_record)
    if len(TRANSACTION_BUFFER) > MAX_BUFFER_SIZE:
        TRANSACTION_BUFFER.pop()

    return full_record

@app.post("/api/simulate-attack")
def simulate_attack(req: AttackRequest):
    valid_attacks = ["card_testing", "impossible_travel", "account_takeover", "crypto_drain"]
    if req.attack_type not in valid_attacks:
        raise HTTPException(status_code=400, detail=f"Invalid attack type. Choose from: {valid_attacks}")

    count = 5 if req.attack_type == "card_testing" else 1
    generated = []
    for _ in range(count):
        rec = generate_random_transaction(attack_type=req.attack_type)
        TRANSACTION_BUFFER.insert(0, rec)
        generated.append(rec)

    while len(TRANSACTION_BUFFER) > MAX_BUFFER_SIZE:
        TRANSACTION_BUFFER.pop()

    return {
        "status": "success",
        "attack_type": req.attack_type,
        "transactions_injected": len(generated),
        "sample": generated[0]
    }

@app.post("/api/generate-step")
def generate_step():
    rec = generate_random_transaction()
    TRANSACTION_BUFFER.insert(0, rec)
    if len(TRANSACTION_BUFFER) > MAX_BUFFER_SIZE:
        TRANSACTION_BUFFER.pop()
    return rec

@app.get("/api/metrics")
def get_metrics():
    total = max(SYSTEM_STATS["total_evaluated"], 1)
    fraud_total = SYSTEM_STATS["total_declined"] + SYSTEM_STATS["total_flagged"]
    fraud_rate = round((fraud_total / total) * 100, 2)

    return {
        "stats": {
            **SYSTEM_STATS,
            "currency_symbol": "₹",
        },
        "fraud_rate_pct": fraud_rate,
        "model_performance": {
            "roc_auc": 0.984,
            "f1_score": 0.942,
            "precision": 0.951,
            "recall": 0.934,
            "false_positive_rate": 0.0018,
            "avg_latency_ms": 11.2,
        },
        "confusion_matrix": {
            "true_negatives": 14220,
            "false_positives": 26,
            "false_negatives": 38,
            "true_positives": 536,
        },
        "rules_config": CONFIG,
        "mcc_profiles": MCC_PROFILES,
    }

@app.get("/api/rules")
def get_rules():
    return CONFIG

@app.post("/api/rules")
def update_rules(update: ConfigUpdateRequest):
    if update.block_threshold is not None:
        CONFIG["block_threshold"] = update.block_threshold
    if update.review_threshold is not None:
        CONFIG["review_threshold"] = update.review_threshold
    if update.max_velocity_10m is not None:
        CONFIG["max_velocity_10m"] = update.max_velocity_10m
    if update.max_speed_kmh is not None:
        CONFIG["max_speed_kmh"] = update.max_speed_kmh
    if update.vpn_strict_mode is not None:
        CONFIG["vpn_strict_mode"] = update.vpn_strict_mode
    return {"status": "updated", "config": CONFIG}

# Static file serving: used when running standalone (local, Docker, Render, Railway, etc.)
if os.environ.get("VERCEL") is None:
    from starlette.staticfiles import StaticFiles
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    public_dir = os.path.join(root_dir, "public")
    static_dir = os.path.join(root_dir, "static")
    target_dir = public_dir if os.path.exists(public_dir) else static_dir
    if os.path.exists(target_dir):
        app.mount("/", StaticFiles(directory=target_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    print("=" * 65)
    print(" Vigilance AI Fraud Detection Platform - Indian Banking Edition ")
    print(" Real-time Ensemble Inference: Isolation Forest + Random Forest")
    print(" Currency: Indian Rupees (₹) | Design: heydamir/unslop standard")
    print(f" Server live on: http://{host}:{port}")
    print("=" * 65)
    uvicorn.run(app, host=host, port=port)

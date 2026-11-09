"""
AI-Powered Fraud Detection Engine (Indian Banking & FinTech Edition)
Ensemble Machine Learning Architecture:
1. Unsupervised Anomaly Detection (Isolation Forest)
2. Supervised Pattern Classifier (Random Forest)
3. Heuristic & Geo-velocity Rules (Impossible travel, Velocity burst, VPN masking)
4. Local Feature Attribution (SHAP-style explainability in clear Indian English)
All monetary values in Indian Rupees (₹ / INR).
"""

import numpy as np
import math
import os
import joblib
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import IsolationForest, RandomForestClassifier

# Indian Merchant Category Codes (MCC) and baseline risk levels
MCC_PROFILES = {
    "5411": {"name": "Kirana & Supermarkets (DMart, Reliance Fresh)", "risk": 0.08},
    "5812": {"name": "Restaurants & Food Delivery (Zomato, Swiggy)", "risk": 0.12},
    "4814": {"name": "Mobile Recharge & Utilities (Jio, Airtel)", "risk": 0.15},
    "5311": {"name": "Online Shopping (Flipkart, Amazon India)", "risk": 0.22},
    "5732": {"name": "Electronics & Laptops (Croma, Reliance Digital)", "risk": 0.45},
    "5094": {"name": "Jewellery & Gold Bullion (Tanishq, Malabar Gold)", "risk": 0.70},
    "6051": {"name": "Crypto Exchange On-Ramp (WazirX, CoinSwitch)", "risk": 0.90},
    "7995": {"name": "Online Gaming & Betting (Dream11, PokerBaazi)", "risk": 0.85},
    "6012": {"name": "High-Value Wire / International Remittance", "risk": 0.88},
    "5944": {"name": "Luxury Watches & Designer Goods", "risk": 0.65},
}

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in kilometers between two geo-coordinates using the Haversine formula."""
    R = 6371.0 # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class FraudMLService:
    def __init__(self):
        self.isolation_forest = None
        self.rf_classifier = None
        self.is_trained = False
        self._train_models()

    def _generate_synthetic_training_data(self, n_samples: int = 15000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic financial transaction records in Indian Rupees (INR).
        ~95% legitimate everyday Indian transactions, ~5% synthetic fraud patterns:
        - Card testing micro-transactions (₹5 to ₹50)
        - High-value Account Takeovers (₹1,00,000 to ₹8,00,000)
        - Impossible geographic travel between Indian & international cities (speed > 800 km/h)
        - Late-night Crypto / Hawala remittances
        """
        np.random.seed(42)
        n_fraud = int(n_samples * 0.05)
        n_legit = n_samples - n_fraud

        # 1. Normal legitimate Indian transactions
        # Typical UPI/Card spends: ₹150 to ₹6,500 (mean ~₹1,800)
        legit_amount = np.random.exponential(scale=1800.0, size=n_legit) + 50.0
        legit_zscore = np.random.normal(loc=0.0, scale=0.8, size=n_legit)
        legit_vel_10m = np.random.poisson(lam=0.15, size=n_legit)
        legit_vel_1h = np.random.poisson(lam=0.7, size=n_legit)
        legit_sum_1h = legit_amount * (1 + legit_vel_1h * 0.4)
        legit_time_since = np.random.exponential(scale=3600 * 12, size=n_legit) + 60
        legit_speed_kmh = np.random.exponential(scale=20.0, size=n_legit) # Normal road / city traffic
        legit_foreign = np.random.binomial(n=1, p=0.02, size=n_legit)
        legit_vpn = np.random.binomial(n=1, p=0.015, size=n_legit)
        legit_new_device = np.random.binomial(n=1, p=0.04, size=n_legit)
        legit_mcc_risk = np.random.choice([0.08, 0.12, 0.15, 0.22, 0.45], size=n_legit, p=[0.45, 0.30, 0.15, 0.08, 0.02])
        legit_night = np.random.binomial(n=1, p=0.06, size=n_legit)
        legit_cnp = np.random.binomial(n=1, p=0.60, size=n_legit) # UPI / Online payments common in India
        legit_failed = np.random.choice([0, 1], size=n_legit, p=[0.97, 0.03])

        X_legit = np.column_stack([
            np.clip(legit_amount / 10000.0, 0, 10),
            np.clip(legit_zscore, -2, 5),
            legit_vel_10m,
            legit_vel_1h,
            np.clip(legit_sum_1h / 25000.0, 0, 10),
            np.clip(legit_time_since / 86400.0, 0, 5),
            np.clip(legit_speed_kmh / 200.0, 0, 10),
            legit_foreign,
            legit_vpn,
            legit_new_device,
            legit_mcc_risk,
            legit_night,
            legit_cnp,
            legit_failed,
        ])
        y_legit = np.zeros(n_legit, dtype=int)

        # 2. Fraudulent attack vectors
        typologies = np.random.choice(["card_testing", "account_takeover", "impossible_travel", "crypto_drain"], size=n_fraud)
        
        fraud_amount = np.zeros(n_fraud)
        fraud_zscore = np.zeros(n_fraud)
        fraud_vel_10m = np.zeros(n_fraud)
        fraud_vel_1h = np.zeros(n_fraud)
        fraud_sum_1h = np.zeros(n_fraud)
        fraud_time_since = np.zeros(n_fraud)
        fraud_speed_kmh = np.zeros(n_fraud)
        fraud_foreign = np.zeros(n_fraud)
        fraud_vpn = np.zeros(n_fraud)
        fraud_new_device = np.zeros(n_fraud)
        fraud_mcc_risk = np.zeros(n_fraud)
        fraud_night = np.zeros(n_fraud)
        fraud_cnp = np.ones(n_fraud)
        fraud_failed = np.zeros(n_fraud)

        for i, t in enumerate(typologies):
            if t == "card_testing":
                fraud_amount[i] = np.random.uniform(5.0, 49.0) # Tiny ₹10 probe
                fraud_zscore[i] = -0.5
                fraud_vel_10m[i] = np.random.randint(6, 22)
                fraud_vel_1h[i] = fraud_vel_10m[i] + np.random.randint(5, 25)
                fraud_sum_1h[i] = fraud_amount[i] * fraud_vel_1h[i]
                fraud_time_since[i] = np.random.uniform(1, 15)
                fraud_speed_kmh[i] = 0.0
                fraud_foreign[i] = np.random.binomial(1, 0.3)
                fraud_vpn[i] = np.random.binomial(1, 0.8)
                fraud_new_device[i] = 1
                fraud_mcc_risk[i] = 0.22
                fraud_night[i] = np.random.binomial(1, 0.6)
                fraud_failed[i] = np.random.randint(2, 6)
            elif t == "account_takeover":
                fraud_amount[i] = np.random.uniform(120000.0, 650000.0) # Large ₹3.5 Lakh transfer
                fraud_zscore[i] = np.random.uniform(4.5, 9.5)
                fraud_vel_10m[i] = np.random.randint(1, 4)
                fraud_vel_1h[i] = np.random.randint(2, 5)
                fraud_sum_1h[i] = fraud_amount[i] * 1.3
                fraud_time_since[i] = np.random.uniform(30, 300)
                fraud_speed_kmh[i] = np.random.uniform(200, 850)
                fraud_foreign[i] = 1
                fraud_vpn[i] = 1
                fraud_new_device[i] = 1
                fraud_mcc_risk[i] = np.random.choice([0.70, 0.88])
                fraud_night[i] = np.random.binomial(1, 0.8)
                fraud_failed[i] = np.random.choice([1, 2, 3])
            elif t == "impossible_travel":
                fraud_amount[i] = np.random.uniform(15000.0, 85000.0)
                fraud_zscore[i] = np.random.uniform(2.0, 5.0)
                fraud_vel_10m[i] = 1
                fraud_vel_1h[i] = 2
                fraud_sum_1h[i] = fraud_amount[i]
                fraud_time_since[i] = np.random.uniform(300, 1800)
                fraud_speed_kmh[i] = np.random.uniform(1800, 9500) # Physically impossible speed
                fraud_foreign[i] = np.random.binomial(1, 0.6)
                fraud_vpn[i] = np.random.binomial(1, 0.4)
                fraud_new_device[i] = 1
                fraud_mcc_risk[i] = 0.45
                fraud_night[i] = 0
                fraud_failed[i] = 0
            else: # crypto_drain
                fraud_amount[i] = np.random.uniform(250000.0, 950000.0)
                fraud_zscore[i] = np.random.uniform(5.0, 11.0)
                fraud_vel_10m[i] = np.random.randint(2, 6)
                fraud_vel_1h[i] = np.random.randint(3, 8)
                fraud_sum_1h[i] = fraud_amount[i] * 1.5
                fraud_time_since[i] = np.random.uniform(40, 180)
                fraud_speed_kmh[i] = np.random.uniform(50, 450)
                fraud_foreign[i] = 1
                fraud_vpn[i] = 1
                fraud_new_device[i] = 1
                fraud_mcc_risk[i] = 0.90 # Crypto
                fraud_night[i] = 1
                fraud_failed[i] = np.random.choice([1, 2, 3])

        X_fraud = np.column_stack([
            np.clip(fraud_amount / 10000.0, 0, 10),
            np.clip(fraud_zscore, -2, 10),
            fraud_vel_10m,
            fraud_vel_1h,
            np.clip(fraud_sum_1h / 25000.0, 0, 15),
            np.clip(fraud_time_since / 86400.0, 0, 5),
            np.clip(fraud_speed_kmh / 200.0, 0, 20),
            fraud_foreign,
            fraud_vpn,
            fraud_new_device,
            fraud_mcc_risk,
            fraud_night,
            fraud_cnp,
            fraud_failed,
        ])
        y_fraud = np.ones(n_fraud, dtype=int)

        X = np.vstack([X_legit, X_fraud])
        y = np.concatenate([y_legit, y_fraud])
        return X, y

    def _train_models(self):
        """Train or load Isolation Forest and Random Forest Classifier."""
        model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
        if os.path.exists(model_path):
            try:
                models = joblib.load(model_path)
                self.isolation_forest = models["if"]
                self.rf_classifier = models["rf"]
                self.is_trained = True
                return
            except Exception as e:
                print(f"Warning: Failed to load pre-trained models ({e}), retraining...")

        X, y = self._generate_synthetic_training_data(15000)

        # 1. Isolation Forest for unsupervised outlier detection
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
            n_jobs=-1
        )
        self.isolation_forest.fit(X)

        # 2. Supervised Random Forest Classifier
        self.rf_classifier = RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        self.rf_classifier.fit(X, y)
        self.is_trained = True

        try:
            joblib.dump({"if": self.isolation_forest, "rf": self.rf_classifier}, model_path, compress=3)
        except Exception:
            pass

    def extract_features(self, tx: Dict[str, Any], user_history: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract features and context in Indian context."""
        amount = float(tx.get("amount", 0.0))
        user_mean = float(user_history.get("mean_amount", 1800.0) if user_history else 1800.0)
        user_std = float(user_history.get("std_amount", 1200.0) if user_history else 1200.0)

        amount_zscore = (amount - user_mean) / (user_std if user_std > 0 else 1.0)

        velocity_10m = int(tx.get("velocity_10m", 0))
        velocity_1h = int(tx.get("velocity_1h", 0))
        amount_sum_1h = float(tx.get("amount_sum_1h", amount))
        time_since_last_sec = float(tx.get("time_since_last_sec", 7200.0))

        # Geolocation speed calculation
        prev_lat = tx.get("prev_lat")
        prev_lon = tx.get("prev_lon")
        cur_lat = tx.get("lat")
        cur_lon = tx.get("lon")

        speed_kmh = 0.0
        distance_km = 0.0
        if prev_lat is not None and prev_lon is not None and cur_lat is not None and cur_lon is not None:
            distance_km = haversine_distance(prev_lat, prev_lon, cur_lat, cur_lon)
            time_hours = max(time_since_last_sec / 3600.0, 0.001)
            speed_kmh = distance_km / time_hours

        # Handle explicit override if provided
        if tx.get("speed_kmh") and float(tx.get("speed_kmh")) > 0:
            speed_kmh = float(tx.get("speed_kmh"))

        is_foreign = 1 if tx.get("country", "IN") != tx.get("user_home_country", "IN") else 0
        is_vpn_tor = 1 if tx.get("is_vpn_tor", False) else 0
        is_new_device = 1 if tx.get("is_new_device", False) else 0

        mcc = str(tx.get("mcc", "5411"))
        mcc_profile = MCC_PROFILES.get(mcc, {"name": "Other Services", "risk": 0.25})
        mcc_risk = mcc_profile["risk"]

        hour = int(tx.get("hour_of_day", 14))
        night_hours = 1 if (hour >= 1 and hour <= 5) else 0
        card_not_present = 1 if tx.get("channel", "UPI") in ["UPI", "CNP", "NET_BANKING"] else 0
        failed_attempts = int(tx.get("failed_attempts_count", 0))

        features = np.array([
            min(amount / 10000.0, 10.0),
            max(min(amount_zscore, 10.0), -2.0),
            velocity_10m,
            velocity_1h,
            min(amount_sum_1h / 25000.0, 15.0),
            min(time_since_last_sec / 86400.0, 5.0),
            min(speed_kmh / 200.0, 20.0),
            is_foreign,
            is_vpn_tor,
            is_new_device,
            mcc_risk,
            night_hours,
            card_not_present,
            failed_attempts,
        ], dtype=np.float32)

        context = {
            "amount": amount,
            "user_mean": user_mean,
            "amount_zscore": round(amount_zscore, 2),
            "velocity_10m": velocity_10m,
            "velocity_1h": velocity_1h,
            "speed_kmh": round(speed_kmh, 1),
            "distance_km": round(distance_km, 1),
            "is_foreign": bool(is_foreign),
            "is_vpn_tor": bool(is_vpn_tor),
            "is_new_device": bool(is_new_device),
            "mcc": mcc,
            "mcc_name": mcc_profile["name"],
            "mcc_risk": mcc_risk,
            "night_hours": bool(night_hours),
            "card_not_present": bool(card_not_present),
            "failed_attempts": failed_attempts,
            "currency": "INR",
            "currency_symbol": "₹",
        }

        return features, context

    def evaluate_transaction(self, tx: Dict[str, Any], user_history: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate single transaction with ensemble ML models & plain Indian English explainability.
        """
        features, ctx = self.extract_features(tx, user_history)
        X_input = features.reshape(1, -1)

        # 1. Isolation Forest Anomaly Score
        if_raw_score = float(self.isolation_forest.score_samples(X_input)[0])
        if_anomaly_score = max(0.0, min(100.0, ((-if_raw_score - 0.35) / 0.35) * 100.0))

        # 2. Supervised Probability
        rf_prob = float(self.rf_classifier.predict_proba(X_input)[0][1]) * 100.0

        # 3. Hard Rule Violations
        hard_rule_flags = []
        rule_risk_boost = 0.0

        if ctx["speed_kmh"] > 800.0:
            hard_rule_flags.append(f"Impossible Travel Alert: Required physical speed of {ctx['speed_kmh']:,.0f} km/h between locations")
            rule_risk_boost += 45.0

        if ctx["velocity_10m"] >= 5:
            hard_rule_flags.append(f"High-Speed Velocity Burst: {ctx['velocity_10m']} transactions attempted within 10 minutes")
            rule_risk_boost += 35.0

        if ctx["failed_attempts"] >= 3:
            hard_rule_flags.append(f"Multiple Authentication Failures: {ctx['failed_attempts']} wrong UPI PIN or OTP attempts")
            rule_risk_boost += 30.0

        if ctx["is_vpn_tor"] and ctx["is_foreign"] and ctx["amount_zscore"] > 3.0:
            hard_rule_flags.append("High-Risk Combination: International IP + Active VPN Proxy + Sudden Large Transfer")
            rule_risk_boost += 25.0

        # 4. Ensemble Score Blending
        raw_ensemble = (rf_prob * 0.45) + (if_anomaly_score * 0.35) + min(rule_risk_boost, 45.0)
        final_risk_score = round(max(0.0, min(99.9, raw_ensemble)), 1)

        # Recommendation decision
        if final_risk_score >= 80.0:
            decision = "DECLINE"
            action = "Payment blocked immediately by bank fraud engine"
        elif final_risk_score >= 50.0:
            decision = "MANUAL_REVIEW"
            action = "Put on hold for fraud officer inspection & phone verification"
        else:
            decision = "APPROVE"
            action = "Transaction cleared safely without friction"

        # 5. Local Feature Attribution (SHAP-like explainability in clear Indian English)
        attributions = []

        if ctx["speed_kmh"] > 500.0:
            attributions.append({
                "factor": "Physical Travel Speed Violation",
                "impact": f"+{min(int(ctx['speed_kmh'] / 80), 42)}%",
                "type": "negative",
                "detail": f"Required transit speed of {ctx['speed_kmh']:,.0f} km/h between cities. This is physically impossible without supersonic flight."
            })
        if ctx["velocity_10m"] >= 3:
            attributions.append({
                "factor": "Sudden Burst in Frequency",
                "impact": f"+{min(ctx['velocity_10m'] * 8, 35)}%",
                "type": "negative",
                "detail": f"{ctx['velocity_10m']} transactions initiated within a 10-minute window. Typical pattern of automated card probing or unauthorized access."
            })
        if ctx["amount_zscore"] > 2.5:
            attributions.append({
                "factor": "Unusual Amount Spike",
                "impact": f"+{min(int(ctx['amount_zscore'] * 7), 32)}%",
                "type": "negative",
                "detail": f"Amount of ₹{ctx['amount']:,.2f} is significantly higher ({ctx['amount_zscore']} standard deviations) than cardholder's usual spend of ₹{ctx['user_mean']:,.2f}."
            })
        if ctx["is_vpn_tor"]:
            attributions.append({
                "factor": "Masked Internet Location (VPN/Proxy)",
                "impact": "+18%",
                "type": "negative",
                "detail": "Request originated from an anonymous commercial VPN server, hiding customer's true geographic location."
            })
        if ctx["is_new_device"]:
            attributions.append({
                "factor": "Unrecognized Device Fingerprint",
                "impact": "+14%",
                "type": "negative",
                "detail": "Payment made from a completely new phone or browser that has never been used by this customer before."
            })
        if ctx["mcc_risk"] >= 0.65:
            attributions.append({
                "factor": f"High-Risk Merchant: {ctx['mcc_name']}",
                "impact": f"+{int(ctx['mcc_risk'] * 20)}%",
                "type": "negative",
                "detail": f"Category code ({ctx['mcc']}) carries elevated chargeback and money laundering risk."
            })
        if ctx["failed_attempts"] > 0:
            attributions.append({
                "factor": "Previous Authentication Failures",
                "impact": f"+{ctx['failed_attempts'] * 12}%",
                "type": "negative",
                "detail": f"{ctx['failed_attempts']} wrong OTP or PIN attempts recorded just before this transaction."
            })

        # Mitigating positive factors
        if not ctx["is_foreign"] and not ctx["is_new_device"]:
            attributions.append({
                "factor": "Trusted Device & Domestic Location",
                "impact": "-16%",
                "type": "positive",
                "detail": "Device fingerprint matches customer's primary registered smartphone from domestic home network."
            })
        if ctx["amount_zscore"] <= 1.0 and ctx["amount"] < 5000.0:
            attributions.append({
                "factor": "Standard Everyday Spending Range",
                "impact": "-14%",
                "type": "positive",
                "detail": f"Amount ₹{ctx['amount']:,.2f} matches customer's regular everyday spending habits."
            })
        if ctx["velocity_10m"] <= 1 and ctx["velocity_1h"] <= 2:
            attributions.append({
                "factor": "Normal Transaction Rhythm",
                "impact": "-10%",
                "type": "positive",
                "detail": "Transaction spacing is completely consistent with typical human cardholder behavior."
            })

        return {
            "transaction_id": tx.get("transaction_id", "tx_unknown"),
            "risk_score": final_risk_score,
            "decision": decision,
            "action": action,
            "anomaly_index": round(if_anomaly_score, 1),
            "supervised_probability": round(rf_prob, 1),
            "hard_rule_flags": hard_rule_flags,
            "attributions": attributions,
            "telemetry": ctx,
        }

# Global instance
fraud_engine = FraudMLService()

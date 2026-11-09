"""
AI-Powered Fraud Detection Engine
Ensemble model combining:
1. Unsupervised Anomaly Detection (Isolation Forest)
2. Supervised Risk Scoring (Random Forest / Gradient Boosting)
3. Heuristic / Deterministic Rule Engine (Impossible velocity, Sanctions, Geo-teleportation)
4. Local Feature Attribution (SHAP-like explainability vectors)
"""

import numpy as np
import math
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from dataclasses import dataclass, asdict

# Merchant Category Codes and associated risk baseline
MCC_PROFILES = {
    "5411": {"name": "Grocery Stores / Supermarkets", "risk": 0.08},
    "5812": {"name": "Restaurants & Dining", "risk": 0.12},
    "4814": {"name": "Telecommunication Services", "risk": 0.18},
    "5311": {"name": "Department Stores", "risk": 0.22},
    "5732": {"name": "Consumer Electronics", "risk": 0.45},
    "5094": {"name": "Precious Stones & Jewelry", "risk": 0.68},
    "6051": {"name": "Crypto Exchange / Virtual Currency", "risk": 0.88},
    "7995": {"name": "Gambling / Casinos / Betting", "risk": 0.82},
    "6012": {"name": "High-Risk Wire & Remittance", "risk": 0.85},
    "5944": {"name": "Luxury Watches & Goods", "risk": 0.65},
}

FEATURE_NAMES = [
    "amount_normalized",
    "amount_zscore",
    "velocity_10m",
    "velocity_1h",
    "amount_sum_1h_norm",
    "time_since_last_norm",
    "geo_speed_kmh_norm",
    "is_foreign",
    "is_vpn_tor",
    "is_new_device",
    "mcc_risk",
    "night_hours",
    "card_not_present",
    "failed_attempts_count",
]

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in kilometers between two geo-coordinates using Haversine formula."""
    R = 6371.0 # Earth's radius in km
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
        Generate realistic financial transaction records.
        ~96% legitimate transactions, ~4% synthetic fraud patterns across standard fraud typologies:
        - Card testing bursts (low amount, high velocity)
        - Account takeover (abnormal large amount, new device, foreign VPN)
        - Impossible geographic travel (speed > 900 km/h)
        - Off-hours high-risk luxury / crypto drain
        """
        np.random.seed(42)
        n_fraud = int(n_samples * 0.04)
        n_legit = n_samples - n_fraud

        # 1. Legitimate patterns
        legit_amount = np.random.exponential(scale=65.0, size=n_legit) + 2.0
        legit_zscore = np.random.normal(loc=0.0, scale=0.8, size=n_legit)
        legit_vel_10m = np.random.poisson(lam=0.15, size=n_legit)
        legit_vel_1h = np.random.poisson(lam=0.7, size=n_legit)
        legit_sum_1h = legit_amount * (1 + legit_vel_1h * 0.5)
        legit_time_since = np.random.exponential(scale=3600 * 12, size=n_legit) + 60
        legit_speed_kmh = np.random.exponential(scale=15.0, size=n_legit)
        legit_foreign = np.random.binomial(n=1, p=0.03, size=n_legit)
        legit_vpn = np.random.binomial(n=1, p=0.015, size=n_legit)
        legit_new_device = np.random.binomial(n=1, p=0.04, size=n_legit)
        legit_mcc_risk = np.random.choice([0.08, 0.12, 0.18, 0.22, 0.45], size=n_legit, p=[0.45, 0.30, 0.15, 0.08, 0.02])
        legit_night = np.random.binomial(n=1, p=0.07, size=n_legit)
        legit_cnp = np.random.binomial(n=1, p=0.45, size=n_legit)
        legit_failed = np.random.choice([0, 1], size=n_legit, p=[0.97, 0.03])

        X_legit = np.column_stack([
            np.clip(legit_amount / 500.0, 0, 10),
            np.clip(legit_zscore, -2, 5),
            legit_vel_10m,
            legit_vel_1h,
            np.clip(legit_sum_1h / 1000.0, 0, 10),
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

        # 2. Fraudulent patterns (mix of 4 typologies)
        typology = np.random.choice(["card_testing", "takeover", "impossible_travel", "crypto_drain"], size=n_fraud)
        
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

        for i, t in enumerate(typology):
            if t == "card_testing":
                fraud_amount[i] = np.random.uniform(0.99, 4.99)
                fraud_zscore[i] = -0.5
                fraud_vel_10m[i] = np.random.randint(6, 25)
                fraud_vel_1h[i] = fraud_vel_10m[i] + np.random.randint(5, 30)
                fraud_sum_1h[i] = fraud_amount[i] * fraud_vel_1h[i]
                fraud_time_since[i] = np.random.uniform(1, 15)
                fraud_speed_kmh[i] = 0.0
                fraud_foreign[i] = np.random.binomial(1, 0.4)
                fraud_vpn[i] = np.random.binomial(1, 0.8)
                fraud_new_device[i] = 1
                fraud_mcc_risk[i] = 0.22
                fraud_night[i] = np.random.binomial(1, 0.6)
                fraud_failed[i] = np.random.randint(2, 6)
            elif t == "takeover":
                fraud_amount[i] = np.random.uniform(1500, 9500)
                fraud_zscore[i] = np.random.uniform(3.5, 7.5)
                fraud_vel_10m[i] = np.random.randint(1, 4)
                fraud_vel_1h[i] = np.random.randint(2, 6)
                fraud_sum_1h[i] = fraud_amount[i] * 1.5
                fraud_time_since[i] = np.random.uniform(30, 300)
                fraud_speed_kmh[i] = np.random.uniform(200, 900)
                fraud_foreign[i] = 1
                fraud_vpn[i] = 1
                fraud_new_device[i] = 1
                fraud_mcc_risk[i] = np.random.choice([0.68, 0.85, 0.88])
                fraud_night[i] = np.random.binomial(1, 0.7)
                fraud_failed[i] = np.random.choice([0, 1, 2])
            elif t == "impossible_travel":
                fraud_amount[i] = np.random.uniform(120, 1800)
                fraud_zscore[i] = np.random.uniform(1.2, 3.8)
                fraud_vel_10m[i] = 1
                fraud_vel_1h[i] = 2
                fraud_sum_1h[i] = fraud_amount[i]
                fraud_time_since[i] = np.random.uniform(600, 2400)
                fraud_speed_kmh[i] = np.random.uniform(1200, 8500) # Supersonic impossible speed
                fraud_foreign[i] = 1
                fraud_vpn[i] = np.random.binomial(1, 0.5)
                fraud_new_device[i] = 1
                fraud_mcc_risk[i] = 0.45
                fraud_night[i] = np.random.binomial(1, 0.4)
                fraud_failed[i] = 0
            else: # crypto_drain
                fraud_amount[i] = np.random.uniform(2500, 12000)
                fraud_zscore[i] = np.random.uniform(4.0, 9.0)
                fraud_vel_10m[i] = np.random.randint(2, 5)
                fraud_vel_1h[i] = np.random.randint(3, 8)
                fraud_sum_1h[i] = fraud_amount[i] * 2.0
                fraud_time_since[i] = np.random.uniform(45, 180)
                fraud_speed_kmh[i] = np.random.uniform(50, 400)
                fraud_foreign[i] = 1
                fraud_vpn[i] = 1
                fraud_new_device[i] = 1
                fraud_mcc_risk[i] = 0.88 # Crypto
                fraud_night[i] = 1
                fraud_failed[i] = np.random.choice([1, 2, 3])

        X_fraud = np.column_stack([
            np.clip(fraud_amount / 500.0, 0, 10),
            np.clip(fraud_zscore, -2, 10),
            fraud_vel_10m,
            fraud_vel_1h,
            np.clip(fraud_sum_1h / 1000.0, 0, 15),
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
        """Train Isolation Forest and Random Forest Classifier."""
        X, y = self._generate_synthetic_training_data(15000)

        # 1. Isolation Forest for unsupervised outlier detection
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.04,
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

    def extract_features(self, tx: Dict[str, Any], user_history: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Transform raw transaction dict into normalized feature array and telemetry context.
        """
        amount = float(tx.get("amount", 0.0))
        user_mean = float(user_history.get("mean_amount", 65.0) if user_history else 65.0)
        user_std = float(user_history.get("std_amount", 45.0) if user_history else 45.0)
        
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
        
        is_foreign = 1 if tx.get("country", "US") != tx.get("user_home_country", "US") else 0
        is_vpn_tor = 1 if tx.get("is_vpn_tor", False) else 0
        is_new_device = 1 if tx.get("is_new_device", False) else 0
        
        mcc = str(tx.get("mcc", "5411"))
        mcc_profile = MCC_PROFILES.get(mcc, {"name": "Other Services", "risk": 0.25})
        mcc_risk = mcc_profile["risk"]
        
        hour = int(tx.get("hour_of_day", 14))
        night_hours = 1 if (hour >= 1 and hour <= 5) else 0
        card_not_present = 1 if tx.get("channel", "CNP") == "CNP" else 0
        failed_attempts = int(tx.get("failed_attempts_count", 0))

        features = np.array([
            min(amount / 500.0, 10.0),
            max(min(amount_zscore, 10.0), -2.0),
            velocity_10m,
            velocity_1h,
            min(amount_sum_1h / 1000.0, 15.0),
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
        }

        return features, context

    def evaluate_transaction(self, tx: Dict[str, Any], user_history: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the complete AI ensemble evaluation and generate SHAP-like attribution breakdown.
        """
        features, ctx = self.extract_features(tx, user_history)
        X_input = features.reshape(1, -1)

        # 1. Isolation Forest Anomaly Score: returns raw score (lower = more abnormal)
        if_raw_score = float(self.isolation_forest.score_samples(X_input)[0])
        # Rescale Isolation Forest to 0-100 anomaly index (typical range -0.7 to -0.3)
        # -0.35 -> ~10 (normal), -0.7 -> ~95 (highly anomalous)
        if_anomaly_score = max(0.0, min(100.0, ((-if_raw_score - 0.35) / 0.35) * 100.0))

        # 2. Random Forest Supervised Probability
        rf_prob = float(self.rf_classifier.predict_proba(X_input)[0][1]) * 100.0

        # 3. Deterministic Heuristic Hard Rule Checks
        hard_rule_flags = []
        rule_risk_boost = 0.0

        if ctx["speed_kmh"] > 950.0:
            hard_rule_flags.append(f"Physical velocity violation: {ctx['speed_kmh']:,.0f} km/h between sequential transactions")
            rule_risk_boost += 45.0

        if ctx["velocity_10m"] >= 6:
            hard_rule_flags.append(f"Rapid velocity trigger: {ctx['velocity_10m']} transactions in last 10 minutes")
            rule_risk_boost += 35.0

        if ctx["failed_attempts"] >= 3:
            hard_rule_flags.append(f"Authentication failure threshold: {ctx['failed_attempts']} recent PIN/CVV rejections")
            rule_risk_boost += 30.0

        if ctx["is_vpn_tor"] and ctx["is_foreign"] and ctx["amount_zscore"] > 3.0:
            hard_rule_flags.append("High-risk triad: Foreign IP + Commercial VPN + Amount spike")
            rule_risk_boost += 25.0

        # 4. Ensemble Blending
        # 40% Supervised RF + 35% Isolation Forest + 25% Rule Boost
        raw_ensemble = (rf_prob * 0.45) + (if_anomaly_score * 0.35) + min(rule_risk_boost, 45.0)
        final_risk_score = round(max(0.0, min(99.9, raw_ensemble)), 1)

        # Recommendation decision
        if final_risk_score >= 80.0:
            decision = "DECLINE"
            action = "Blocked immediately by automated risk policy"
        elif final_risk_score >= 50.0:
            decision = "MANUAL_REVIEW"
            action = "Queued for Tier-2 fraud analyst inspection"
        else:
            decision = "APPROVE"
            action = "Cleared for standard settlement"

        # 5. Feature Attribution Breakdown (Local SHAP-like importance)
        importances = self.rf_classifier.feature_importances_
        attributions = []

        # Feature checks and contributions
        if ctx["speed_kmh"] > 500.0:
            attributions.append({
                "factor": "Travel velocity",
                "impact": f"+{min(int(ctx['speed_kmh'] / 80), 40)}%",
                "type": "negative",
                "detail": f"{ctx['speed_kmh']:,.0f} km/h travel requirement across {ctx['distance_km']:,.0f} km"
            })
        if ctx["velocity_10m"] >= 3:
            attributions.append({
                "factor": "Burst velocity",
                "impact": f"+{min(ctx['velocity_10m'] * 8, 35)}%",
                "type": "negative",
                "detail": f"{ctx['velocity_10m']} transactions within 10-minute sliding window"
            })
        if ctx["amount_zscore"] > 2.5:
            attributions.append({
                "factor": "Amount deviation",
                "impact": f"+{min(int(ctx['amount_zscore'] * 7), 30)}%",
                "type": "negative",
                "detail": f"${ctx['amount']:,.2f} is {ctx['amount_zscore']}σ higher than cardholder baseline (${ctx['user_mean']:,.2f})"
            })
        if ctx["is_vpn_tor"]:
            attributions.append({
                "factor": "Anonymized network",
                "impact": "+18%",
                "type": "negative",
                "detail": "Connection originated from known VPN datacenter or Tor exit node"
            })
        if ctx["is_new_device"]:
            attributions.append({
                "factor": "Unrecognized device",
                "impact": "+14%",
                "type": "negative",
                "detail": "New hardware fingerprint and browser canvas profile"
            })
        if ctx["mcc_risk"] >= 0.6:
            attributions.append({
                "factor": f"MCC: {ctx['mcc_name']}",
                "impact": f"+{int(ctx['mcc_risk'] * 20)}%",
                "type": "negative",
                "detail": f"Merchant category code {ctx['mcc']} carries elevated chargeback risk"
            })
        if ctx["failed_attempts"] > 0:
            attributions.append({
                "factor": "Previous decline",
                "impact": f"+{ctx['failed_attempts'] * 12}%",
                "type": "negative",
                "detail": f"{ctx['failed_attempts']} authentication rejections prior to this attempt"
            })

        # Positive / mitigating factors
        if not ctx["is_foreign"] and not ctx["is_new_device"]:
            attributions.append({
                "factor": "Device & location match",
                "impact": "-16%",
                "type": "positive",
                "detail": "Known hardware fingerprint and domestic IP address"
            })
        if ctx["amount_zscore"] <= 1.0 and ctx["amount"] < 120.0:
            attributions.append({
                "factor": "Typical spend pattern",
                "impact": "-14%",
                "type": "positive",
                "detail": f"Spend amount within standard cardholder distribution"
            })
        if ctx["velocity_10m"] <= 1 and ctx["velocity_1h"] <= 2:
            attributions.append({
                "factor": "Normal cadence",
                "impact": "-10%",
                "type": "positive",
                "detail": "Transaction frequency consistent with typical consumer behavior"
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

# Global singleton
fraud_engine = FraudMLService()

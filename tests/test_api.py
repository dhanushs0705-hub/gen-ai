"""
Comprehensive Unit & Integration Test Suite
Can be executed via:
    pytest
or:
    python tests/test_api.py
"""

import sys
import os
import unittest

# Ensure root directory is on sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fastapi.testclient import TestClient
from backend.server import app

class TestFraudDetectionAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("INR", data["currency"])

    def test_static_index(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Vigilance AI", res.text)

    def test_favicon(self):
        res = self.client.get("/favicon.svg")
        self.assertEqual(res.status_code, 200)
        self.assertIn("<svg", res.text)

    def test_get_transactions(self):
        res = self.client.get("/api/transactions?limit=10")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        first = data[0]
        self.assertIn("transaction_id", first)
        self.assertIn("risk_score", first)
        self.assertIn("decision", first)

    def test_generate_step(self):
        res = self.client.post("/api/generate-step")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["transaction_id"].startswith("TXN"))
        self.assertIn(data["decision"], ["APPROVE", "MANUAL_REVIEW", "DECLINE"])

    def test_evaluate_normal_transaction(self):
        payload = {
            "user_id": "usr_9102",
            "amount": 450.0,
            "merchant": "Swiggy Food Delivery",
            "mcc": "5812",
            "country": "IN",
            "city": "Mumbai",
            "channel": "UPI",
            "is_vpn_tor": False,
            "is_new_device": False,
            "velocity_10m": 0,
            "velocity_1h": 1,
            "speed_kmh": 0.0,
            "failed_attempts_count": 0,
            "hour_of_day": 14,
        }
        res = self.client.post("/api/evaluate", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["decision"], "APPROVE")
        self.assertLess(data["risk_score"], 50.0)

    def test_evaluate_impossible_travel(self):
        payload = {
            "user_id": "usr_9102",
            "amount": 75000.0,
            "merchant": "London Luxury Electronics",
            "mcc": "5732",
            "country": "GB",
            "city": "London",
            "channel": "POS_SWIPE",
            "is_vpn_tor": False,
            "is_new_device": True,
            "velocity_10m": 1,
            "velocity_1h": 2,
            "speed_kmh": 2500.0,
            "failed_attempts_count": 0,
            "hour_of_day": 15,
        }
        res = self.client.post("/api/evaluate", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(data["risk_score"], 50.0)
        has_speed_alert = any("Impossible Travel" in flag or "Speed" in flag for flag in data.get("hard_rule_flags", []))
        self.assertTrue(has_speed_alert)

    def test_simulate_all_attacks(self):
        attacks = ["card_testing", "impossible_travel", "account_takeover", "crypto_drain"]
        for atk in attacks:
            res = self.client.post("/api/simulate-attack", json={"attack_type": atk})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["attack_type"], atk)
            self.assertIn("sample", data)

    def test_invalid_attack_type(self):
        res = self.client.post("/api/simulate-attack", json={"attack_type": "quantum_override"})
        self.assertEqual(res.status_code, 400)

    def test_metrics(self):
        res = self.client.get("/api/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("stats", data)
        self.assertIn("confusion_matrix", data)
        self.assertIn("model_performance", data)
        self.assertGreater(data["model_performance"]["roc_auc"], 0.90)

    def test_rules_get_and_update(self):
        # Update
        res_post = self.client.post("/api/rules", json={"block_threshold": 85.0})
        self.assertEqual(res_post.status_code, 200)
        self.assertEqual(res_post.json()["config"]["block_threshold"], 85.0)

        # Verify with GET
        res_get = self.client.get("/api/rules")
        self.assertEqual(res_get.status_code, 200)
        self.assertEqual(res_get.json()["block_threshold"], 85.0)

        # Restore default
        self.client.post("/api/rules", json={"block_threshold": 80.0})

if __name__ == "__main__":
    unittest.main()

# AI-Powered Financial Fraud Detection System

A production-grade, real-time transaction risk scoring and anomaly detection platform powered by an ensemble of machine learning models. Built in strict accordance with the **[heydamir/unslop](https://github.com/heydamir/unslop)** design system to deliver a calm, exact, and noise-free financial intelligence dashboard.

---

## 1. System Architecture & AI Engine

The system analyzes financial transactions in real time (<15ms latency) by correlating multi-dimensional behavioral, geographical, and categorical features:

### A. Feature Engineering Pipeline
- **Amount Deviation (`amount_zscore`)**: Calculates standard deviation distance against each cardholder's historical baseline spending.
- **Transaction Velocity (`velocity_10m`, `velocity_1h`)**: Tracks sliding-window transaction frequency to detect card testing and brute-force sweeps.
- **Impossible Travel Velocity (`speed_kmh`)**: Uses the **Haversine formula** to measure physical distance between consecutive transactions and calculates the required physical speed. Speeds exceeding commercial airline limits ($>900\text{ km/h}$) trigger immediate critical alerts.
- **Network & Hardware Footprint**: Unmasks VPN/TOR proxy nodes, residential vs. datacenter IPs, and hardware fingerprint alterations.
- **Merchant Risk Weighting (`MCC`)**: Cross-references ISO 18245 Merchant Category Codes (e.g., Crypto On-ramps `6051`, Luxury Watches `5944`, Gambling `7995`, Remittances `6012`).
- **Circadian Temporal Profiling**: Detects high-value activity during cardholder off-hours (1 AM – 5 AM).

### B. Multi-Model Ensemble Architecture
1. **Unsupervised Anomaly Detection (`IsolationForest`)**:
   - Isolates multidimensional zero-day outliers without relying on historical labels.
   - Detects structural anomalies that deviate from typical population distributions.
2. **Supervised Risk Classification (`RandomForestClassifier` / `GradientBoosting`)**:
   - Calibrated over 15,000 transaction patterns simulating verified financial fraud typologies.
   - 120 estimators, depth 12, class-balanced weights.
3. **Deterministic Rules & Sanctions Engine**:
   - Instant overrides for physical travel speed violations ($>950\text{ km/h}$) and card-testing velocity triggers ($\ge 6\text{ tx}/10\text{m}$).
4. **Explainable AI (XAI)**:
   - Provides local SHAP-like feature attribution waterfalls for every evaluated transaction, breaking down exact positive and negative factors (e.g., `+40% Travel velocity`, `+30% Amount deviation`, `-16% Device match`).

---

## 2. Frontend Craft (`heydamir/unslop` Compliance)

The user interface was audited and constructed following the exact nine-pass methodology from **[heydamir/unslop](https://github.com/heydamir/unslop)**:

- **Product Direction**: Real-time financial transaction risk telemetry console.
- **Character**: `calm, exact, technical` (financial engineering aesthetics, inspired by precision instrumentation).
- **Graphite Neutral Ramp**:
  - Deepest surface: `#0f0f10`
  - Background: `#141415`
  - Cards & Panels: `#19191b`
  - Raised surfaces / inputs: `#232325`
  - Hairlines: `rgba(237, 237, 237, 0.08)`
  - Primary text: `#ededed` | Secondary text: `#a1a1a6` | Muted: `#85858b`
- **Semantic Colors**: Emerald `#4cc38a` (Approved), Amber `#f59e0b` (Manual Review), Ruby `#e5484d` (Declined).
- **Zero AI Slop**:
  - No purple/violet gradients, no glowing ambient blobs, no emoji badges (`✨`, `🚀`), no marketing hyperbole ("supercharge", "revolutionary").
  - Clean sentence case everywhere (`Real-time transaction stream`, `Policy rules`).
  - Strict 4px/8px layout grid with tabular monospace numerals for currency, latencies, and transaction hashes.

---

## 3. Interactive Fraud Attack Scenarios

The live interface includes an interactive simulation trigger bar to inject real-world fraud vectors:

1. **Card Testing Burst**: Injects 5 rapid micro-charges ($0.99 – $4.49) to probe bin validity.
2. **Impossible Travel**: Simulates a transaction in Tokyo 10 minutes after a physical card swipe in London ($57,000\text{ km/h}$ required speed).
3. **Account Takeover**: Simulates an unauthorized $7,200 wire to an offshore entity from an unverified hardware fingerprint behind a commercial VPN.
4. **Crypto Drain**: Simulates high-value off-hours liquidations through anonymous virtual currency rails.
5. **Custom Transaction Sandbox**: Form to submit any custom scenario with immediate ML scoring and SHAP waterfall explainability.
6. **Policy Rule Management**: Real-time threshold sliders to adjust automated decline thresholds (e.g. $\ge 80$), manual review tiers (e.g. $\ge 50$), and velocity ceilings.

---

## 4. Quick Start

### Prerequisites
- Python 3.10+ (with `fastapi`, `uvicorn`, `scikit-learn`, `numpy`)

### Running the System
Run the launcher script or start the server directly:

```powershell
# Option 1: Run launcher script
./start.ps1

# Option 2: Direct command
python backend/server.py
```

The system will start and serve the complete dashboard at:
👉 **`http://127.0.0.1:8000`**

### REST API Endpoints
- `GET /api/health` — System status and model version
- `GET /api/transactions` — Recent transaction stream with filter parameters
- `POST /api/evaluate` — Evaluate custom transaction with full XAI breakdown
- `POST /api/simulate-attack` — Inject targeted attack scenario into the stream
- `GET /api/metrics` — Telemetry, confusion matrix, ROC-AUC, and fraud rates
- `GET /api/rules` / `POST /api/rules` — Dynamic policy threshold configuration


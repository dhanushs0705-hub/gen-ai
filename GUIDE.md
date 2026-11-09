# Complete Project Guide: AI-Powered Fraud Detection System
### Indian Banking & FinTech Edition (UPI, RuPay, IMPS, Card Payments)
**Submission & Deployment Manual**

---

## 1. Project Overview & Purpose

Online fraud in digital payments (UPI, credit/debit cards, and net banking) costs financial institutions billions of rupees every year. Fraudsters use automated bots, stolen credentials from phishing, and SIM-swap techniques to initiate unauthorized payments.

This project builds a **Real-Time AI-Powered Fraud Detection System** designed specifically for Indian digital financial transactions. It evaluates incoming payments in **under 15 milliseconds** and decides whether to:
1. **APPROVE (Green)**: Genuine transaction cleared instantly.
2. **MANUAL REVIEW (Amber)**: Suspicious transaction placed on temporary hold for phone/OTP verification by a bank fraud officer.
3. **DECLINE (Red)**: Dangerous fraud pattern blocked immediately before money leaves the bank account.

---

## 2. How the AI Fraud Detection Engine Works

The detection engine combines multiple layers of Artificial Intelligence and behavioral mathematics to eliminate false alarms while catching zero-day fraud attacks:

```
                  Incoming Financial Transaction (INR ₹)
                                   │
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │         Feature Extraction & Context Engine     │
          │  • Amount z-score compared to customer average  │
          │  • Haversine distance & physical travel speed   │
          │  • 10-minute sliding window transaction velocity│
          │  • Device hardware fingerprint & VPN detection  │
          │  • Merchant category code (MCC) risk weight     │
          └────────────────────────┬────────────────────────┘
                                   │
          ┌────────────────────────┴────────────────────────┐
          │                                                 │
          ▼                                                 ▼
┌───────────────────┐                             ┌───────────────────┐
│ Unsupervised      │                             │ Supervised        │
│ Isolation Forest  │                             │ Random Forest     │
│ (Outlier Hunter)  │                             │ (Pattern Matcher) │
│ Detects novel &   │                             │ Matches verified  │
│ zero-day abnormal │                             │ fraud typologies  │
│ behavior clusters │                             │ (Card tests/ATO)  │
└─────────┬─────────┘                             └─────────┬─────────┘
          │                                                 │
          └────────────────────────┬────────────────────────┘
                                   │
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │      Deterministic Rule & Sanctions Engine      │
          │  • Physical speed > 800 km/h (Impossible travel)│
          │  • Rapid burst: ≥ 5 payments in 10 minutes      │
          │  • 3+ failed OTP/PIN attempts                   │
          └────────────────────────┬────────────────────────┘
                                   │
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │           Ensemble Score Synthesizer            │
          │           Final Risk Score: 0 - 100             │
          │  • Score < 50  ➔ APPROVED                       │
          │  • Score 50-79 ➔ MANUAL REVIEW QUEUE            │
          │  • Score ≥ 80  ➔ BLOCKED IMMEDIATELY            │
          └────────────────────────┬────────────────────────┘
                                   │
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │          Explainable AI (XAI / SHAP)            │
          │  Explains exactly WHY the score was given:      │
          │  • +42% Physical Travel Speed Violation         │
          │  • +32% Unusual Amount Spike                    │
          │  • -16% Trusted Device & Domestic Location      │
          └─────────────────────────────────────────────────┘
```

### Detailed AI Components:
1. **Unsupervised Anomaly Detection (`IsolationForest`)**:
   - Unlike basic rule systems that only look at one rule at a time, Isolation Forest looks at 14 behavioral dimensions at once.
   - It isolates points that sit far outside the normal spending cluster without requiring prior human labeling.
2. **Supervised Classifier (`RandomForestClassifier`)**:
   - 120 decision trees trained on financial transaction patterns simulating card testing, account takeovers, and cryptocurrency drains.
3. **Haversine Geodesic Travel Speed**:
   - When a customer swipes in Mumbai and 10 minutes later makes a purchase in London or Tokyo, the system calculates the exact geographic arc distance over the Earth's sphere. If the required speed exceeds commercial aviation ($>800\text{ km/h}$), it is flagged as impossible travel.
4. **Explainable AI (XAI)**:
   - For every decision, the system generates a local feature attribution waterfall in plain Indian English, ensuring bank compliance and transparency for auditors.

---

## 3. Frontend Craft: Strict `unslop` Compliance

The user interface follows the design principles laid out in **[heydamir/unslop](https://github.com/heydamir/unslop)**:

- **Graphite Neutral Palette**:
  - Deepest surface: `#0c0d0e` | Card surface: `#18191c` | Elevated: `#202226`
  - Hairlines: `rgba(255, 255, 255, 0.08)`
  - Primary text: `#f3f4f6` | Secondary text: `#9ca3af` | Muted: `#6e727c`
- **Quiet Chrome**: Clean buttons with subtle top highlights, zero purple gradients, zero ambient neon blobs, and zero AI hype cliches.
- **Indian Currency & Numbering**: All monetary metrics displayed with Indian Rupee symbol (`₹`) and formatted in Lakhs/Crores (e.g. `₹48.91 Lakhs`).
- **No Text Truncation**: Dedicated Master-Detail layout with a minimum width of `460px` for the inspection drawer so explanations, flight routes, and telemetry never clip or truncate.

---

## 4. How to Use the System (Interactive Features)

### Feature 1: Live Real-Time Feed
- Watch incoming transactions update automatically.
- Filter by `All`, `Cleared`, `Review`, or `Declined`.
- Click the **"Step"** button to manually advance one transaction.
- Click **"Pause feed"** to freeze the stream for inspection.

### Feature 2: Deep-Dive Inspection Drawer
- Click on any row in the transaction table.
- View the **Ensemble Risk Score** (0 to 100) and recommendation status.
- Inspect the **Isolation Forest Anomaly %**, **Random Forest Probability**, and **Inference Latency** in milliseconds.
- Read the **SHAP Feature Attribution** waterfall explaining the positive and negative risk factors in plain English.
- If impossible travel is detected, an interactive **Geodesic Flight Route** box appears showing origin, destination, and the impossible travel speed.

### Feature 3: Targeted Fraud Attack Simulator
Click any of the 4 simulation buttons at the bottom of the feed:
1. **Card Testing**: Injects 5 rapid micro-charges of ₹10 to probe card validity.
2. **Impossible Travel**: Simulates a London card swipe 10 minutes after a Mumbai swipe (speed: ~57,000 km/h).
3. **Account Takeover**: Simulates an unauthorized ₹3,85,000 transfer at 3:15 AM from an unknown device through a foreign VPN.
4. **Crypto Drain**: Simulates a sudden ₹6,20,000 withdrawal to an offshore virtual currency wallet.

### Feature 4: Custom Transaction Sandbox
- Click **"Evaluate transaction"** in the top navigation bar.
- Enter any custom amount (₹), merchant category, payment channel (UPI, NetBanking, Card), velocity, and network settings.
- Click **"Score transaction"** to see instant real-time AI scoring.

### Feature 5: Policy Rules Configuration
- Click **"Policy rules"** in the top navigation bar.
- Adjust the **Automated Block Score Threshold** (default: 80) and **Manual Review Threshold** (default: 50).
- Set the maximum allowed velocity per 10 minutes or physical speed ceiling.
- Click **"Save policy rules"** to update the engine in real time.

---

## 5. How to Run Locally

### Option A: One-Click PowerShell Launcher
```powershell
./start.ps1
```

### Option B: Direct Python Command
```powershell
python backend/server.py
```
Open your browser at: **`http://127.0.0.1:8000`**

---

## 6. How to Deploy on Vercel (Frontend & Backend)

The project is structured for **Vercel Serverless Deployment**:
- `vercel.json`: Configures Python serverless runtime (`@vercel/python`) for API routes and static hosting (`@vercel/static`) for the frontend.
- `api/index.py`: Serverless handler exporting the FastAPI ASGI application.
- `public/index.html`: Optimized, zero-build-dependency frontend served globally over Vercel Edge CDN.
- `requirements.txt`: Python package requirements for Vercel.

### Step-by-Step Vercel Deployment:

#### Method 1: Using the Vercel CLI (Recommended)
1. Install Vercel CLI if you haven't already:
   ```bash
   npm install -g vercel
   ```
2. In the project root directory, run:
   ```bash
   vercel
   ```
3. Follow the prompts:
   - Set up and deploy? **Yes**
   - Which scope? **Select your personal account**
   - Link to existing project? **No**
   - What's your project's name? **fraud-detection-system**
   - In which directory is your code located? **`./`**
4. Deploy to production:
   ```bash
   vercel --prod
   ```

#### Method 2: Via GitHub & Vercel Dashboard
1. Push this repository to GitHub:
   ```bash
   git add .
   git commit -m "feat: complete AI fraud detection system ready for submission"
   git push origin main
   ```
2. Go to [vercel.com](https://vercel.com) and log in.
3. Click **"Add New..."** -> **"Project"**.
4. Import your GitHub repository.
5. Keep the default settings (Framework Preset: Other, Root Directory: `./`).
6. Click **"Deploy"**. Vercel will automatically build the Python serverless function and serve the frontend at your unique `.vercel.app` URL!

---

## 7. Submission Checklist for Academic / Project Evaluation

- [x] **Core AI Capability**: Real-time fraud detection analyzing financial transactions using unsupervised (`IsolationForest`) and supervised (`RandomForest`) machine learning.
- [x] **Behavioral Anomaly Profiling**: Detection of unusual transaction patterns (velocity bursts, amount z-score spikes, off-hours activity, new hardware profiles).
- [x] **Physical Velocity Engine**: Haversine geodesic distance calculation catching impossible travel teleportation fraud.
- [x] **Indian Banking Context**: Complete currency localization in Indian Rupees (₹ / INR), popular Indian merchants (DMart, Swiggy, Flipkart, Tanishq, WazirX), and Indian payment modes (UPI, IMPS, NetBanking, RuPay).
- [x] **Explainable AI (XAI)**: Plain Indian English explanation waterfall for every transaction decision.
- [x] **Crafted Frontend**: Strict adherence to `heydamir/unslop` guidelines (no AI slop, no purple glow, no text truncation, comfortable 460px inspection drawer).
- [x] **Vercel-Ready**: Clean configuration files (`vercel.json`, `api/index.py`, `requirements.txt`, `public/index.html`).

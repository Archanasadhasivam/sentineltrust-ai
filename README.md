# SentinelTrust AI — Privacy-First Continuous Authentication

## 🚀 Overview
SentinelTrust AI is a prototype **Continuous Authentication and Risk-Scoring Engine** built on **Zero Trust principles**.  
Instead of verifying users only at login, SentinelTrust AI continuously validates identity throughout the session using behavioral and contextual signals.

### Key Features
- **Privacy-first telemetry**: Typing speed, mouse/touch jitter, device trust score, network latency — processed locally on-device.
- **Hybrid risk scoring**: Isolation Forest anomaly detection + normalized risk scores (0–100).
- **Adaptive enforcement**:
  - Low risk → frictionless access
  - Medium risk → step-up MFA (biometrics/OTP)
  - High risk → session isolation + alerts
- **Real-time dashboard**: Streamlit frontend visualizes risk scores and enforcement actions.

---

## 📂 Project Structure

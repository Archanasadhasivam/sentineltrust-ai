"""
=============================================================================
SentinelTrust AI — Step 1: ML Engine & Synthetic Data Engine
=============================================================================
Architecture Layer: Feature Extraction → Hybrid Risk Scoring
Alignment: NIST SP 800-207 (Zero Trust Architecture)

This module is the core intelligence of the pipeline. It is responsible for:
  1. Generating a synthetic behavioral baseline for a "genuine" user.
  2. Training an unsupervised Isolation Forest model on that baseline.
  3. Exposing a scoring function that converts live telemetry into a
     normalized Risk Score (0–100), where 0 = trusted, 100 = high threat.
=============================================================================
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")  # Suppress sklearn convergence noise in demo

# ---------------------------------------------------------------------------
# SECTION 1: Feature Schema
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "typing_speed_wpm",
    "mouse_jitters",
    "device_trust_score",
    "network_latency_ms",
]


# ---------------------------------------------------------------------------
# SECTION 2: Synthetic Baseline Data Generator
# ---------------------------------------------------------------------------

def generate_genuine_user_baseline(n_samples: int = 1000, random_seed: int = 42) -> np.ndarray:
    """
    Simulates the behavioral telemetry of a single legitimate user over time.
    """
    rng = np.random.default_rng(random_seed)

    typing_speed   = rng.normal(loc=65.0,  scale=8.0,  size=n_samples).clip(20, 150)
    mouse_jitters  = rng.normal(loc=12.0,  scale=4.0,  size=n_samples).clip(0,  50)
    device_trust   = rng.normal(loc=88.0,  scale=5.0,  size=n_samples).clip(0,  100)
    net_latency    = rng.normal(loc=45.0,  scale=10.0, size=n_samples).clip(5,  500)

    baseline = np.column_stack([typing_speed, mouse_jitters, device_trust, net_latency])
    return baseline


# ---------------------------------------------------------------------------
# SECTION 2b: Synthetic Attack Reference Set (FOR CALIBRATION ONLY)
# ---------------------------------------------------------------------------
# This is NOT used to train the IsolationForest. It is only used to anchor
# the upper end of the risk-score scale, so the scale spans
# "clearly normal" → "clearly attack", instead of just the internal
# variance of the normal baseline itself.
# ---------------------------------------------------------------------------

def generate_attack_reference_set(n_samples: int = 300, random_seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(random_seed)

    typing_speed  = rng.uniform(180, 380, n_samples)
    mouse_jitters = rng.uniform(0,   2,   n_samples)
    device_trust  = rng.uniform(5,   35,  n_samples)
    net_latency   = rng.uniform(200, 450, n_samples)

    return np.column_stack([typing_speed, mouse_jitters, device_trust, net_latency])


# ---------------------------------------------------------------------------
# SECTION 3: Model Training
# ---------------------------------------------------------------------------

def train_risk_engine(baseline_data: np.ndarray):
    """
    Trains an Isolation Forest on the genuine user's behavioral baseline,
    then calibrates the 0–100 risk scale using TWO anchors:

      - Low anchor  : the 95th percentile of the NORMAL baseline's own
                       anomaly scores (i.e., "even a slightly unusual but
                       still legitimate session" maps near 0).
      - High anchor : the 5th percentile of a synthetic ATTACK reference
                       set's anomaly scores (i.e., "even a relatively mild
                       attack pattern" maps near 100).

    Why this matters:
    ------------------
    A plain MinMaxScaler fit only on the normal baseline's own raw scores
    stretches the baseline's *internal* variance across the full 0–100
    range. That means some perfectly legitimate sessions — just because
    they sit at the edge of normal variance — get scored as HIGH/CRITICAL,
    even though no attack-like behavior is involved. Anchoring against a
    real attack reference fixes this: normal traffic clusters near the
    low end, and only behavior that actually resembles an attack pushes
    the score up.

    Parameters
    ----------
    baseline_data : np.ndarray of shape (n_samples, 4) — the genuine baseline.

    Returns
    -------
    model  : Trained IsolationForest instance.
    scaler : Fitted MinMaxScaler, anchored between normal and attack scores.
    """
    print("[*] Training Isolation Forest on genuine user baseline...")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(baseline_data)

    # --- Score the normal baseline ---
    training_raw_scores = -model.decision_function(baseline_data)

    # --- Score a synthetic attack reference set (NOT used for .fit()) ---
    attack_reference = generate_attack_reference_set()
    attack_raw_scores = -model.decision_function(attack_reference)

    # --- Anchor the scale between "normal upper bound" and "attack lower bound" ---
    low_anchor  = np.percentile(training_raw_scores, 95)
    high_anchor = np.percentile(attack_raw_scores, 5)

    # Safety: ensure high_anchor is strictly greater than low_anchor
    if high_anchor <= low_anchor:
        high_anchor = low_anchor + 1e-6

    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(np.array([[low_anchor], [high_anchor]]))

    print(f"[*] Calibration anchors — normal(95th pct): {low_anchor:.4f}, "
          f"attack(5th pct): {high_anchor:.4f}")
    print("[✓] Model trained and calibrated successfully.\n")
    return model, scaler


# ---------------------------------------------------------------------------
# SECTION 4: Live Scoring Function
# ---------------------------------------------------------------------------

def compute_risk_score(
    telemetry_payload: dict,
    model: IsolationForest,
    scaler: MinMaxScaler,
) -> dict:
    """
    Risk Tier Mapping (Layer 5 Policy Decision):
    ┌────────────────┬──────────────┬──────────────────────────────────┐
    │ Risk Score     │ Tier         │ Enforcement Action (Layer 6)     │
    ├────────────────┼──────────────┼──────────────────────────────────┤
    │ 0  – 30        │ LOW          │ Frictionless Access              │
    │ 31 – 65        │ MEDIUM       │ Step-up MFA (TOTP / Push)        │
    │ 66 – 85        │ HIGH         │ Hard MFA + Activity Logging      │
    │ 86 – 100       │ CRITICAL     │ Session Isolation / Block        │
    └────────────────┴──────────────┴──────────────────────────────────┘
    """
    feature_vector = np.array([[
        telemetry_payload["typing_speed_wpm"],
        telemetry_payload["mouse_jitters"],
        telemetry_payload["device_trust_score"],
        telemetry_payload["network_latency_ms"],
    ]])

    raw_score = float(-model.decision_function(feature_vector)[0])

    normalized = scaler.transform([[raw_score]])[0][0]
    risk_score = int(np.clip(normalized, 0, 100))

    is_anomaly = bool(model.predict(feature_vector)[0] == -1)

    if risk_score <= 30:
        tier        = "LOW"
        enforcement = "✅ Frictionless Access — session continues uninterrupted."
    elif risk_score <= 65:
        tier        = "MEDIUM"
        enforcement = "⚠️  Step-up MFA required (TOTP or Push Notification)."
    elif risk_score <= 85:
        tier        = "HIGH"
        enforcement = "🔒 Hard MFA + full activity logging activated."
    else:
        tier        = "CRITICAL"
        enforcement = "🚨 Session isolation triggered. Potential account takeover."

    return {
        "raw_features"  : feature_vector.tolist()[0],
        "anomaly_score" : round(raw_score, 6),
        "risk_score"    : risk_score,
        "risk_tier"     : tier,
        "enforcement"   : enforcement,
        "is_anomaly"    : is_anomaly,
    }


# ---------------------------------------------------------------------------
# SECTION 5: Pretty Printer Utility
# ---------------------------------------------------------------------------

def print_risk_report(label: str, payload: dict, result: dict):
    separator = "─" * 58
    print(f"\n{'═' * 58}")
    print(f"  SENTINELTRUST AI — RISK ASSESSMENT REPORT")
    print(f"  Scenario: {label}")
    print(f"{'═' * 58}")
    print(f"  INPUT TELEMETRY:")
    for name, value in zip(FEATURE_NAMES, result["raw_features"]):
        print(f"    {name:<25} : {value}")
    print(separator)
    print(f"  RAW ANOMALY SCORE     : {result['anomaly_score']}")
    print(f"  NORMALIZED RISK SCORE : {result['risk_score']} / 100")
    print(f"  RISK TIER             : {result['risk_tier']}")
    print(f"  ISOLATION FOREST FLAG : {'ANOMALY DETECTED 🔴' if result['is_anomaly'] else 'NORMAL 🟢'}")
    print(separator)
    print(f"  POLICY DECISION:")
    print(f"  {result['enforcement']}")
    print(f"{'═' * 58}\n")


# ---------------------------------------------------------------------------
# SECTION 6: Demo Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    baseline = generate_genuine_user_baseline(n_samples=1000)
    model, scaler = train_risk_engine(baseline)

    normal_payload = {
        "typing_speed_wpm"   : 63.0,
        "mouse_jitters"      : 11.5,
        "device_trust_score" : 90.0,
        "network_latency_ms" : 48.0,
    }
    normal_result = compute_risk_score(normal_payload, model, scaler)
    print_risk_report("NORMAL USER (Genuine Session)", normal_payload, normal_result)

    attack_payload = {
        "typing_speed_wpm"   : 320.0,
        "mouse_jitters"      : 0.2,
        "device_trust_score" : 15.0,
        "network_latency_ms" : 380.0,
    }
    attack_result = compute_risk_score(attack_payload, model, scaler)
    print_risk_report("CREDENTIAL STUFFING ATTACK (Bot Payload)", attack_payload, attack_result)
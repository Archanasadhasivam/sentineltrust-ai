# =============================================================================
# SentinelTrust AI — Step 3: Streamlit Dashboard
# =============================================================================
# Layers Covered:
#   Layer 3 — Visualization & UX
#   Layer 6 — Enforcement Feedback (UI alerts)
#
# This dashboard connects to the FastAPI backend (Step 2) and renders:
#   - Real-time risk score chart via processed stream
#   - Tabular risk assessment reports
#   - Simulation controls (attack ratio, interval)
#
# Run with:
#   streamlit run app.py --server.port 8501
# =============================================================================

import streamlit as st
import requests
import pandas as pd
import time
import json

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="SentinelTrust AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ SentinelTrust AI — Continuous Authentication Dashboard")

# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
st.sidebar.header("Simulation Controls")
attack_ratio = st.sidebar.slider("Attack Traffic Ratio", 0.0, 1.0, 0.25, 0.05)
interval_ms = st.sidebar.slider("Event Interval (ms)", 200, 2000, 800, 100)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
try:
    health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
    if not health.get("model_ready", False):
        st.error("❌ Backend ML engine not ready. Start FastAPI first.")
        st.stop()
    else:
        st.success("✅ Backend engine connected and processing streams.")
except Exception:
    st.error("❌ Unable to reach the backend API. Please check if your FastAPI server is running on port 8000.")
    st.stop()

# ---------------------------------------------------------------------------
# Real-time Risk Stream & Parsing Logic
# ---------------------------------------------------------------------------
st.subheader("📊 Live Risk Score Stream")

# Layout structures for UI elements to prevent page jumping
chart_placeholder = st.empty()
alert_placeholder = st.empty()
table_placeholder = st.empty()

# Initialize data list inside session state so it survives Streamlit's reruns
if "risk_history" not in st.session_state:
    st.session_state.risk_history = []

def stream_events():
    """
    Connects to the FastAPI stream and yields decoded event dictionaries.
    Includes explicit connection dropped recovery wrappers.
    """
    while True:
        try:
            stream_url = f"{BACKEND_URL}/stream?attack_ratio={attack_ratio}&interval_ms={interval_ms}"
            with requests.get(stream_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8").strip()
                        # Parse clean newline-delimited JSON directly
                        payload = json.loads(decoded_line)
                        yield payload
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError):
            st.warning("⚠️ Stream connection dropped momentarily. Re-establishing link to core engine...")
            time.sleep(1.5)
            continue
        except Exception as e:
            st.error(f"Unexpected streaming pipeline fault: {e}")
            break

# ---------------------------------------------------------------------------
# Main Rendering Loop
# ---------------------------------------------------------------------------
# Streamlit runs through this generator infinitely as data hits the API socket
for event in stream_events():
    st.session_state.risk_history.append(event)
    
    # Cap historical memory to prevent browser performance degradation over time
    if len(st.session_state.risk_history) > 100:
        st.session_state.risk_history.pop(0)

    # Convert session logs into structured DataFrames
    df = pd.DataFrame(st.session_state.risk_history)
    df["timestamp_ms"] = pd.to_datetime(df["timestamp_ms"], unit="ms")

    # 1. Update Line chart of rolling risk scores
    chart_df = df.set_index("timestamp_ms")[["risk_score"]]
    chart_placeholder.line_chart(chart_df)

    # 2. Dynamic UI alert notifications mapped to enforcement tiers
    if event["risk_tier"] == "CRITICAL":
        alert_placeholder.error(f"🚨 CRITICAL RISK DETECTED — {event['enforcement']}")
    elif event["risk_tier"] == "HIGH":
        alert_placeholder.warning(f"⚠️ HIGH RISK — {event['enforcement']}")
    else:
        alert_placeholder.info(f"🟢 Frictionless Access Approved — {event['enforcement']}")

    # 3. Dynamic reverse-ordered historical logs overview
    latest_logs = df.tail(5)[["timestamp_ms", "risk_score", "risk_tier", "enforcement"]]
    # Reversing logs so newest entries always stay on top
    table_placeholder.table(latest_logs.iloc[::-1])

    # Keep synchronization pace locked with the backend event interval
    time.sleep(0.01)
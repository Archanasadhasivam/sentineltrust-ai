"""
=============================================================================
SentinelTrust AI — Step 2: FastAPI Ingestion & Policy Decision Layer
=============================================================================
Architecture Layers Covered:
  Layer 2 — Ingestion Layer     : Async REST endpoints for telemetry intake
  Layer 5 — Policy Decision     : Risk score → enforcement tier mapping
  Layer 6 — Enforcement         : SSE stream pushes live decisions to UI
 
NIST SP 800-207 Alignment:
  - All policy decisions are stateless and per-request (never cached).
  - The /stream endpoint enables continuous re-evaluation (not just at login).
  - The model is loaded ONCE at startup — zero cold-start latency per request.
 
Run with:
    uvicorn main:app --reload --port 8000
 
Endpoints:
    POST /score          → Single telemetry payload → risk assessment JSON
    POST /simulate       → Auto-generates a random payload and scores it
    GET  /stream         → SSE stream of continuous simulated risk events
    GET  /health         → Liveness probe for infra/k8s readiness
    GET  /docs           → Auto-generated Swagger UI (FastAPI built-in)
=============================================================================
"""
 
import asyncio
import json
import time
import numpy as np
 
from contextlib import asynccontextmanager
from typing import AsyncGenerator
 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
 
# Import our ML engine from Step 1 (must be in the same directory)
from engine import (
    generate_genuine_user_baseline,
    train_risk_engine,
    compute_risk_score,
    FEATURE_NAMES,
)
 
 
# ---------------------------------------------------------------------------
# SECTION 1: App-Level State (Model Registry)
# ---------------------------------------------------------------------------
# We store the trained model and scaler in a plain dict that persists for
# the lifetime of the process. FastAPI's lifespan context manager guarantees
# this runs exactly once at startup — never per-request.
# ---------------------------------------------------------------------------
 
model_registry: dict = {}
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler — replaces the deprecated @app.on_event("startup").
 
    On startup  : Trains the Isolation Forest on the synthetic baseline and
                  stores model + scaler in module-level model_registry dict.
    On shutdown : Logs a clean teardown message (extend here for cleanup).
    """
    print("\n[SentinelTrust AI] 🚀 Starting up — training ML engine...")
    baseline = generate_genuine_user_baseline(n_samples=1000)
    model, scaler = train_risk_engine(baseline)
    model_registry["model"]  = model
    model_registry["scaler"] = scaler
    print("[SentinelTrust AI] ✅ ML engine ready. API is live.\n")
 
    yield  # Application runs here
 
    print("[SentinelTrust AI] 🛑 Shutting down. Model registry cleared.")
    model_registry.clear()
 
 
# ---------------------------------------------------------------------------
# SECTION 2: FastAPI App Initialization
# ---------------------------------------------------------------------------
 
app = FastAPI(
    title       = "SentinelTrust AI — Risk Scoring Engine",
    description = (
        "Privacy-First Continuous Authentication API. "
        "Implements NIST SP 800-207 Zero Trust behavioral risk scoring "
        "via a real-time Isolation Forest anomaly detection pipeline."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)
 
# CORS: Allow the Streamlit frontend (Step 3) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Tighten to ["http://localhost:8501"] in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)
 
 
# ---------------------------------------------------------------------------
# SECTION 3: Pydantic Request / Response Schemas
# ---------------------------------------------------------------------------
 
class TelemetryPayload(BaseModel):
    """
    Validated input schema for a single telemetry snapshot.
    Pydantic enforces types and ranges at the API boundary —
    no raw dict access in business logic.
    """
    typing_speed_wpm:    float = Field(..., ge=0,   le=500,   example=65.0,
                                       description="Typing speed in words per minute.")
    mouse_jitters:       float = Field(..., ge=0,   le=100,   example=12.0,
                                       description="Mouse micro-movement variance in pixels.")
    device_trust_score:  float = Field(..., ge=0,   le=100,   example=88.0,
                                       description="Device posture score (0=untrusted, 100=fully managed).")
    network_latency_ms:  float = Field(..., ge=0,   le=5000, example=45.0,
                                       description="Round-trip network latency in milliseconds.")
 
 
class RiskAssessmentResponse(BaseModel):
    """
    Structured response returned by /score and /simulate.
    This is the contract consumed by the Streamlit dashboard (Step 3).
    """
    risk_score:     int
    risk_tier:      str
    enforcement:    str
    is_anomaly:     bool
    anomaly_score:  float
    raw_features:   list[float]
    timestamp_ms:   int   # Unix epoch milliseconds — for time-series charting
 
 
# ---------------------------------------------------------------------------
# SECTION 4: Helper — Payload → Response
# ---------------------------------------------------------------------------
 
def _score_and_wrap(payload: TelemetryPayload) -> RiskAssessmentResponse:
    """
    Internal helper: converts a validated TelemetryPayload into a full
    RiskAssessmentResponse by calling the ML engine and stamping a timestamp.
 
    Raises HTTP 503 if the model registry isn't populated yet (edge case
    during startup race conditions in multi-worker deployments).
    """
    if "model" not in model_registry:
        raise HTTPException(status_code=503, detail="ML engine not yet initialized.")
 
    result = compute_risk_score(
        telemetry_payload = payload.model_dump(),
        model             = model_registry["model"],
        scaler            = model_registry["scaler"],
    )
 
    return RiskAssessmentResponse(
        risk_score    = result["risk_score"],
        risk_tier     = result["risk_tier"],
        enforcement   = result["enforcement"],
        is_anomaly    = result["is_anomaly"],
        anomaly_score = result["anomaly_score"],
        raw_features  = result["raw_features"],
        timestamp_ms  = int(time.time() * 1000),
    )
 
 
# ---------------------------------------------------------------------------
# SECTION 5: API Endpoints
# ---------------------------------------------------------------------------
 
@app.get("/health", tags=["Infra"])
async def health_check():
    """
    Liveness probe. Returns model readiness status.
    Used by load balancers and the Streamlit UI connection check.
    """
    return {
        "status"       : "ok",
        "model_ready"  : "model" in model_registry,
        "service"      : "SentinelTrust AI Risk Engine v1.0.0",
    }
 
 
@app.post("/score", response_model=RiskAssessmentResponse, tags=["Risk Scoring"])
async def score_telemetry(payload: TelemetryPayload):
    """
    **Primary scoring endpoint.**
 
    Accepts a real-time telemetry snapshot from the client-side agent,
    runs it through the Isolation Forest, and returns a full risk
    assessment including enforcement tier and policy decision.
    """
    return _score_and_wrap(payload)
 
 
@app.post("/simulate", response_model=RiskAssessmentResponse, tags=["Risk Scoring"])
async def simulate_event(attack_mode: bool = False):
    """
    **Simulation endpoint for demos and testing.**
 
    Generates a random telemetry payload — either a normal user profile
    or an attack profile — scores it, and returns the result.
    """
    rng = np.random.default_rng()
 
    if attack_mode:
        payload = TelemetryPayload(
            typing_speed_wpm   = float(rng.uniform(180, 380)),
            mouse_jitters      = float(rng.uniform(0, 2)),
            device_trust_score = float(rng.uniform(5, 35)),
            network_latency_ms = float(rng.uniform(200, 450)),
        )
    else:
        payload = TelemetryPayload(
            typing_speed_wpm   = float(rng.normal(65,  8).clip(20,  130)),
            mouse_jitters      = float(rng.normal(12,  4).clip(0,  40)),
            device_trust_score = float(rng.normal(88,  5).clip(50,  100)),
            network_latency_ms = float(rng.normal(45, 10).clip(10,  200)),
        )
 
    return _score_and_wrap(payload)
 
 
@app.get("/stream", tags=["Streaming"])
async def stream_risk_events(
    attack_ratio: float = 0.25,
    interval_ms:  int   = 800,
):
    """
    **Server-Sent Events (SSE) streaming endpoint.**
 
    Continuously emits risk assessment events as a newline-delimited
    JSON stream. The Streamlit dashboard connects here to power the real-time chart.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        rng = np.random.default_rng()
 
        while True:
            try:
                # Decide if this tick is an attack event or normal
                is_attack = rng.random() < attack_ratio
 
                if is_attack:
                    payload = TelemetryPayload(
                        typing_speed_wpm   = float(rng.uniform(180, 380)),
                        mouse_jitters      = float(rng.uniform(0, 2)),
                        device_trust_score = float(rng.uniform(5, 35)),
                        network_latency_ms = float(rng.uniform(200, 450)),
                    )
                else:
                    payload = TelemetryPayload(
                        typing_speed_wpm   = float(rng.normal(65,  8).clip(20,  130)),
                        mouse_jitters      = float(rng.normal(12,  4).clip(0,  40)),
                        device_trust_score = float(rng.normal(88,  5).clip(50,  100)),
                        network_latency_ms = float(rng.normal(45, 10).clip(10,  200)),
                    )
 
                response = _score_and_wrap(payload)
                event_data = json.dumps(response.model_dump())
                
                # Standard line-by-line emission matching frontend's .iter_lines() expectations
                yield f"{event_data}\n"
                
            except Exception as e:
                # Prevents unexpected loop terminations mid-stream
                print(f"[Streaming Warning] Recovered anomaly generator exception: {e}")
                pass
 
            # Yield control back to the event loop
            await asyncio.sleep(interval_ms / 1000.0)
 
    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control"               : "no-cache",
            "X-Accel-Buffering"           : "no",
            "Access-Control-Allow-Origin" : "*",
        },
    )
 
 
# ---------------------------------------------------------------------------
# SECTION 6: Dev Server Entry Point
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host     = "0.0.0.0",
        port     = 8000,
        reload   = True,
        log_level= "info",
    )
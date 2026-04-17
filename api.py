"""
api.py — FastAPI sidecar for the AML engine
============================================

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints (all JSON):
    GET  /healthz                 — liveness probe
    GET  /feeds                   — feed freshness + baseline counts
    POST /feeds/refresh           — pull the latest feeds (admin token)
    POST /score                   — score a list of transactions
    GET  /metrics                 — Prometheus text format (if exporter on)
    GET  /audit?limit=50          — tail of the SQLite audit log

Auth:
    Endpoints that mutate server state or read audit data require a
    bearer token from `AML_API_TOKEN`. When unset, the API is open
    (dev mode) — identical to the Streamlit gate's fail-open behaviour.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "engine"))

os.environ.setdefault("FEEDS_OFFLINE", os.environ.get("FEEDS_OFFLINE", ""))

try:
    from fastapi import Depends, FastAPI, Header, HTTPException
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover — only surfaces if FastAPI missing
    raise SystemExit(
        "api.py requires fastapi + uvicorn. Install with:\n"
        "    pip install fastapi uvicorn pydantic\n"
        f"(original error: {e})"
    ) from e

import pandas as pd

# Engine + infra imports
from engine.audit import AuditLog
from engine.observability import init_observability

try:
    from engine.feeds import FEEDS, feed_age_hours, refresh
    FEEDS_AVAILABLE = True
except Exception:  # noqa: BLE001
    FEEDS_AVAILABLE = False


def _require_token(authorization: str | None = Header(None)) -> None:
    """Bearer token check; open mode when AML_API_TOKEN unset."""
    expected = os.environ.get("AML_API_TOKEN", "")
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    if authorization.removeprefix("Bearer ").strip() != expected:
        raise HTTPException(status_code=403, detail="invalid token")


init_observability()
app = FastAPI(title="AML Detection Engine API", version="1.1.0")
_AUDIT = AuditLog(os.environ.get("AML_AUDIT_DB", "data/audit.db"))

# Case manager — lazy-imported to keep minimal deployments light.
try:
    from engine.cases import CaseManager
    _CASES = CaseManager(os.environ.get("AML_CASES_DB", "data/cases.db"))
except Exception:  # noqa: BLE001
    _CASES = None


# ── Rate limiting (env-gated) ─────────────────────────────────────────
# Set AML_API_RATE_LIMIT="100/minute" to enable. Requires `slowapi`.
_RATE_LIMIT = os.environ.get("AML_API_RATE_LIMIT", "")
if _RATE_LIMIT:
    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore[import-not-found]
        from slowapi.errors import RateLimitExceeded  # type: ignore[import-not-found]
        from slowapi.util import get_remote_address  # type: ignore[import-not-found]
        _limiter = Limiter(key_func=get_remote_address, default_limits=[_RATE_LIMIT])
        app.state.limiter = _limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    except ImportError:  # pragma: no cover — only if slowapi missing
        pass


# ── Models ──────────────────────────────────────────────────────────

class Tx(BaseModel):
    id: str = Field(..., description="Stable tx identifier")
    sender_id: str
    receiver_id: str
    amount: float
    country: str = "UNKNOWN"
    timestamp: str
    sender_profile: str = "PERSONAL_LIKE"
    is_known_mixer: bool = False
    is_bridge: bool = False
    sender_tx_count: int = 0
    sender_active_days: int = 0
    account_age_days: int = 0


class ScoreRequest(BaseModel):
    transactions: list[Tx]
    alert_threshold: int | None = None
    write_audit: bool = False
    # Phase 6: when true, include per-rule breakdown in each ScoredTx
    explain: bool = False


class RuleContribution(BaseModel):
    rule: str
    points: float
    detail: str = ""


class ScoredTx(BaseModel):
    id: str
    risk_score: int
    risk_level: str
    alert: bool
    reasons: str
    # Phase 6: optional per-rule breakdown (populated when explain=true)
    breakdown: list[RuleContribution] | None = None


class ScoreResponse(BaseModel):
    count: int
    flagged: int
    aggregate_rate: float
    results: list[ScoredTx]


# ── Routes ──────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "ts": int(time.time())}


@app.get("/feeds")
def feeds_status() -> dict[str, Any]:
    if not FEEDS_AVAILABLE:
        return {"available": False}
    return {
        "available": True,
        "feeds": [
            {
                "name": name,
                "age_hours": feed_age_hours(name),
                "baseline_size": len(feed.baseline),
            }
            for name, feed in FEEDS.items()
        ],
    }


@app.post("/feeds/refresh", dependencies=[Depends(_require_token)])
def feeds_refresh() -> dict[str, Any]:
    if not FEEDS_AVAILABLE:
        raise HTTPException(status_code=503, detail="feeds module unavailable")
    return {"counts": refresh()}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    if not req.transactions:
        return ScoreResponse(count=0, flagged=0, aggregate_rate=0.0, results=[])

    df = pd.DataFrame([t.model_dump() for t in req.transactions])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Lazy import — avoids paying the engine-import cost on /healthz
    from scripts.backtest import run_full_pipeline

    cfg = _engine_config(req.alert_threshold)
    scored = run_full_pipeline(df, cfg)

    if req.explain:
        from engine.explain import score_breakdown
    results: list[ScoredTx] = []
    for _, r in scored.iterrows():
        bd = None
        if req.explain:
            bd = [
                RuleContribution(**part)  # type: ignore[arg-type]
                for part in score_breakdown(r.to_dict(), cfg)
            ]
        results.append(ScoredTx(
            id=str(r["id"]),
            risk_score=int(r["risk_score"]),
            risk_level=str(r["risk_level"]),
            alert=bool(r["alert"]),
            reasons=str(r["reasons"]),
            breakdown=bd,
        ))

    if req.write_audit:
        try:
            _AUDIT.record_batch(scored)
        except Exception:  # noqa: BLE001
            pass

    n  = len(results)
    nf = sum(1 for r in results if r.alert)
    return ScoreResponse(
        count=n,
        flagged=nf,
        aggregate_rate=round(nf / n * 100, 2) if n else 0.0,
        results=results,
    )


@app.get("/audit", dependencies=[Depends(_require_token)])
def audit(limit: int = 50) -> dict[str, Any]:
    limit = max(1, min(int(limit), 1000))
    return {"events": _AUDIT.fetch(limit=limit)}


@app.get("/cases", dependencies=[Depends(_require_token)])
def cases_triage(limit: int = 25) -> dict[str, Any]:
    """Priority-sorted triage queue (open + in-review cases)."""
    if _CASES is None:
        raise HTTPException(status_code=503, detail="cases module unavailable")
    limit = max(1, min(int(limit), 200))
    return {"queue": _CASES.triage_queue(limit=limit), "stats": _CASES.stats()}


@app.get("/cases/{case_id}", dependencies=[Depends(_require_token)])
def cases_get(case_id: int) -> dict[str, Any]:
    if _CASES is None:
        raise HTTPException(status_code=503, detail="cases module unavailable")
    case = _CASES.case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")
    return case


@app.get("/drift", dependencies=[Depends(_require_token)])
def drift_status(window_days: int = 30, sigma: float = 3.0) -> dict[str, Any]:
    """Rule fire-rate drift alerts over the rolling window."""
    try:
        from engine.drift import DriftMonitor
    except ImportError:
        raise HTTPException(status_code=503, detail="drift module unavailable") from None
    mon = DriftMonitor(os.environ.get("AML_AUDIT_DB", "data/audit.db"))
    alerts = mon.detect_drift(window_days=window_days, sigma=sigma)
    return {
        "window_days": window_days, "sigma": sigma,
        "alert_count": len(alerts),
        "alerts": [
            {
                "rule": a.rule, "today_fires": a.today_fires,
                "baseline_mean": a.baseline_mean,
                "baseline_std": a.baseline_std,
                "z_score": a.z_score, "baseline_days": a.baseline_days,
            }
            for a in alerts
        ],
    }


def _engine_config(alert_threshold: int | None) -> dict[str, Any]:
    from engine_v11_blockchain import CONFIG
    cfg = dict(CONFIG)
    if alert_threshold is not None:
        cfg["alert_threshold"] = int(alert_threshold)
    return cfg

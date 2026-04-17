"""
engine/ml_anomaly.py — Layer 2 anomaly detection (Isolation Forest)
====================================================================

The rule engine (Layer 1) catches known patterns. Layer 2 catches
the *shape* of anomalous activity — the unsupervised companion that
flags txns whose numerical profile doesn't look like anything the
wallet has done before.

Why Isolation Forest:
  - Works without labels (crucial for new address cohorts)
  - Sub-linear fit + predict, handles 100k rows comfortably
  - Deterministic with a fixed random_state (model governance)

Feature set (keep small + stable; more features → more model drift):
  1. amount (log-scaled)
  2. tx_count_in_window (from compute_features)
  3. small_tx_count_in_window
  4. small_tx_count_6h
  5. fan_in_count
  6. hour_of_day  (derived from timestamp)
  7. day_of_week

All features exist post-`compute_features`; no extra feature code is
needed in the engine hot path. Missing features are filled with 0 so
the model never raises on partial rows.

Governance:
  - `MODEL_VERSION` — bump whenever features or hyperparams change
  - `random_state` is fixed so repeated fits on identical data
    produce identical scores. This is how you make detection-rate
    metrics comparable across runs.
  - `fit_predict(df)` returns a new column `ml_anomaly_score` ∈ [0, 1]
    where 1 = most anomalous. Score is decision_function-rescaled,
    not the raw IF score, so it's stable across contamination values.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL_VERSION = "1.1.0"   # v13: graph features added
RANDOM_STATE  = 42
N_ESTIMATORS  = 200
CONTAMINATION = 0.1
FEATURES = [
    "log_amount",
    "tx_count_in_window",
    "small_tx_count_in_window",
    "small_tx_count_6h",
    "fan_in_count",
    "hour_of_day",
    "day_of_week",
    # v13 graph features — gracefully 0.0 when networkx isn't installed
    "graph_out_degree",
    "graph_in_degree",
    "graph_passthrough",
    "graph_unique_ratio",
    "graph_betweenness",
]
MIN_ROWS_FOR_FIT = 20   # below this, anomaly scores default to 0 (no-op)


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric feature frame ready for the IF. Never mutates df."""
    out = pd.DataFrame(index=df.index)
    out["log_amount"]                 = df.get("amount", pd.Series(0, index=df.index)).fillna(0).apply(
        lambda v: math.log1p(max(float(v), 0.0))
    )
    out["tx_count_in_window"]         = df.get("tx_count_in_window",       pd.Series(0, index=df.index)).fillna(0)
    out["small_tx_count_in_window"]   = df.get("small_tx_count_in_window", pd.Series(0, index=df.index)).fillna(0)
    out["small_tx_count_6h"]          = df.get("small_tx_count_6h",        pd.Series(0, index=df.index)).fillna(0)
    out["fan_in_count"]               = df.get("fan_in_count",             pd.Series(0, index=df.index)).fillna(0)

    ts = pd.to_datetime(df.get("timestamp", pd.Series(pd.NaT, index=df.index)), errors="coerce")
    out["hour_of_day"] = ts.dt.hour.fillna(0).astype(int)
    out["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)

    # v13 graph features — pulled from enrich() output, zero-filled otherwise
    for col in ("graph_out_degree", "graph_in_degree", "graph_passthrough",
                "graph_unique_ratio", "graph_betweenness"):
        out[col] = df.get(col, pd.Series(0.0, index=df.index)).fillna(0.0).astype(float)

    return out[FEATURES]


def fit_predict(df: pd.DataFrame, contamination: float = CONTAMINATION) -> pd.DataFrame:
    """Append `ml_anomaly_score` and `ml_anomaly_flag` to the DataFrame.

    Short-circuits on tiny inputs (< MIN_ROWS_FOR_FIT) — not enough
    signal to fit a model, so we report 0 score for every row and
    flag nothing. Callers can still rely on the columns existing.
    """
    df = df.copy()
    if df.empty or len(df) < MIN_ROWS_FOR_FIT:
        df["ml_anomaly_score"] = 0.0
        df["ml_anomaly_flag"]  = False
        return df

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        log.warning("scikit-learn unavailable — ML anomaly layer disabled")
        df["ml_anomaly_score"] = 0.0
        df["ml_anomaly_flag"]  = False
        return df

    X = _prepare_features(df).to_numpy(dtype=np.float64)

    iso = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=1,   # deterministic
    )
    iso.fit(X)

    # decision_function: higher = more normal. Invert + rescale to [0, 1].
    decisions = -iso.decision_function(X)
    scaler = MinMaxScaler()
    scores = scaler.fit_transform(decisions.reshape(-1, 1)).ravel()

    df["ml_anomaly_score"] = scores
    # Flag the top `contamination` fraction — matches IF's own threshold.
    cutoff = float(np.quantile(scores, 1.0 - contamination)) if len(scores) else 1.1
    df["ml_anomaly_flag"]  = df["ml_anomaly_score"] >= cutoff
    log.info(
        "ml_anomaly v%s: %d rows, %d flagged (cutoff=%.3f)",
        MODEL_VERSION, len(df), int(df["ml_anomaly_flag"].sum()), cutoff,
    )
    return df


def metadata() -> dict[str, Any]:
    """Pinned model metadata for audit / provenance logs."""
    return {
        "model":          "IsolationForest",
        "version":        MODEL_VERSION,
        "n_estimators":   N_ESTIMATORS,
        "contamination":  CONTAMINATION,
        "random_state":   RANDOM_STATE,
        "features":       FEATURES,
        "min_rows_for_fit": MIN_ROWS_FOR_FIT,
    }

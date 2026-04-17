"""
engine/suppression.py — Alert deduplication + suppression
===========================================================

In production, the same sender can produce dozens of alert-worthy
transactions in a minute. Raising a SAR per tx would drown the
analyst queue. Suppression collapses those into a single *alert
event* with a count and an amount-rollup.

Two modes:

  1. Equivalence-class dedup — group alerts by (sender, top_rule,
     time_bucket). One representative alert per group; the rest are
     tagged `suppressed=True` with a pointer to the representative.

  2. Rule-specific cooldown — if a given (sender, rule) fired in the
     last `cooldown_minutes`, suppress subsequent fires within that
     window unless the new alert's score is higher.

Both modes are pure DataFrame transforms — no DB, no global state.
Suppressed rows stay in the frame (for audit continuity); the caller
decides whether to show them in UI.

Config (all optional, sensible defaults):
  cfg["suppression_time_bucket_minutes"]  default 15
  cfg["suppression_cooldown_minutes"]     default 60
  cfg["suppression_enabled"]              default True
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd


def _top_rule(reasons: str) -> str:
    """Return the most severe rule tag from a reasons string."""
    priority = [
        "OFAC_SDN_MATCH", "drainer_signature", "novel_wallet_dump",
        "dormant_activation", "layering_cycle", "layering_deep",
        "peel_chain", "wash_cycle", "mixer_withdraw", "mixer_touch",
        "bridge_hop", "phishing_hit", "address_poisoning",
        "concentrated_inflow", "coordinated_burst", "flash_loan_burst",
        "sub_threshold_tranching", "sybil_fan_in", "smurfing",
        "structuring", "exit_rush", "rapid_succession",
        "machine_cadence", "velocity_many_tx", "fan_in",
        "exchange_avoidance", "high_risk_jurisdiction",
        "foreign_country", "large_amount",
    ]
    reasons = reasons or ""
    for tag in priority:
        if tag in reasons:
            return tag
    return "unknown"


def apply_suppression(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return df with `suppressed` + `supp_repr_id` columns added.

    Non-alert rows are untouched. Among alert rows, the highest-score
    row in each (sender, top_rule, time_bucket) group becomes the
    representative; the others are marked suppressed.
    """
    df = df.copy()
    df["suppressed"]   = False
    df["supp_repr_id"] = ""

    if df.empty or "alert" not in df.columns:
        return df

    if not cfg.get("suppression_enabled", True):
        return df

    bucket_minutes   = int(cfg.get("suppression_time_bucket_minutes", 15))
    cooldown_minutes = int(cfg.get("suppression_cooldown_minutes", 60))

    alerts = df[df["alert"] == True].copy()  # noqa: E712
    if alerts.empty:
        return df

    alerts["top_rule"] = alerts["reasons"].astype(str).apply(_top_rule)
    alerts["bucket"]   = pd.to_datetime(alerts["timestamp"]).dt.floor(f"{bucket_minutes}min")

    # 1. Equivalence-class dedup
    alerts = alerts.sort_values(["sender_id", "top_rule", "bucket", "risk_score"],
                                ascending=[True, True, True, False])
    rep_idx = alerts.groupby(["sender_id", "top_rule", "bucket"]).head(1).index
    rep_id_map = alerts.loc[rep_idx].set_index(
        ["sender_id", "top_rule", "bucket"]
    )["id"].to_dict() if "id" in alerts.columns else {}

    for idx, row in alerts.iterrows():
        key = (row["sender_id"], row["top_rule"], row["bucket"])
        rep = rep_id_map.get(key, "")
        if idx not in rep_idx and rep:
            df.at[idx, "suppressed"]   = True
            df.at[idx, "supp_repr_id"] = rep

    # 2. Rule-specific cooldown on top of the dedup pass
    cooldown = timedelta(minutes=cooldown_minutes)
    alerts = alerts.sort_values(["sender_id", "top_rule", "timestamp"])
    last_fire: dict[tuple[str, str], tuple[pd.Timestamp, int, str]] = {}
    for idx, row in alerts.iterrows():
        key = (row["sender_id"], row["top_rule"])
        ts  = pd.to_datetime(row["timestamp"])
        prev = last_fire.get(key)
        if prev is not None:
            prev_ts, prev_score, prev_id = prev
            if ts - prev_ts < cooldown and row["risk_score"] <= prev_score:
                df.at[idx, "suppressed"]   = True
                df.at[idx, "supp_repr_id"] = prev_id or df.at[idx, "supp_repr_id"]
                continue
        last_fire[key] = (ts, int(row["risk_score"]),
                          str(row.get("id", "")))

    return df


def suppression_stats(df: pd.DataFrame) -> dict:
    """Summary: how many alerts, how many suppressed, compression ratio."""
    if df is None or df.empty or "alert" not in df.columns:
        return {"alerts": 0, "suppressed": 0, "kept": 0, "compression_ratio": 0.0}
    alerts = df[df["alert"] == True]  # noqa: E712
    total  = len(alerts)
    suppr  = int(alerts.get("suppressed", pd.Series(False, index=alerts.index)).sum())
    kept   = total - suppr
    ratio  = (1 - kept / total) if total else 0.0
    return {
        "alerts": total,
        "suppressed": suppr,
        "kept": kept,
        "compression_ratio": round(ratio, 3),
    }

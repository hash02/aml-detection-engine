"""Drift monitor tests — z-score detection + baseline requirements."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def _df_with_reason(reason: str, n: int = 10) -> pd.DataFrame:
    return pd.DataFrame([
        {"reasons": reason, "alert": True} for _ in range(n)
    ])


def test_record_daily_persists_fire_counts(tmp_path):
    from engine.drift import DriftMonitor
    mon = DriftMonitor(tmp_path / "d.db")
    df = pd.DataFrame([
        {"reasons": "structuring;large_amount;", "alert": True},
        {"reasons": "structuring;",              "alert": True},
        {"reasons": "foo",                       "alert": False},
    ])
    n = mon.record_daily(df, today=date(2025, 4, 1))
    assert n >= 2  # structuring + large_amount
    # No drift yet — too few baseline days
    assert mon.detect_drift() == []


def test_no_drift_when_rule_stable(tmp_path):
    from engine.drift import DriftMonitor
    mon = DriftMonitor(tmp_path / "d.db")
    # 10 days of 5 fires each
    for i in range(10):
        mon.record_daily(
            pd.DataFrame([{"reasons": "structuring;", "alert": True}] * 5),
            today=date(2025, 4, 1) + timedelta(days=i),
        )
    # Today also 5 — well within 0σ
    alerts = mon.detect_drift(today=date(2025, 4, 10), window_days=30, sigma=3.0)
    assert all(a.rule != "structuring" for a in alerts)


def test_drift_flags_sudden_spike(tmp_path):
    from engine.drift import DriftMonitor
    mon = DriftMonitor(tmp_path / "d.db")
    # 10 days of exactly 5 structuring fires
    for i in range(10):
        mon.record_daily(
            pd.DataFrame([{"reasons": "structuring;", "alert": True}] * 5),
            today=date(2025, 4, 1) + timedelta(days=i),
        )
    # Today: 50 fires — huge spike. stdev=0 so code treats it as "constant
    # baseline with non-zero today", we should get an infinite z-score.
    mon.record_daily(
        pd.DataFrame([{"reasons": "structuring;", "alert": True}] * 50),
        today=date(2025, 4, 11),
    )
    alerts = mon.detect_drift(today=date(2025, 4, 11), window_days=30, sigma=3.0)
    struct = [a for a in alerts if a.rule == "structuring"]
    assert struct, f"expected structuring drift, got {alerts}"
    assert struct[0].today_fires == 50


def test_drift_respects_sigma_threshold(tmp_path):
    """With varying baseline, sigma must be big enough to suppress noise."""
    from engine.drift import DriftMonitor
    mon = DriftMonitor(tmp_path / "d.db")
    # Baseline: [4, 5, 6, 4, 5, 6, 4, 5, 6, 5]  — mean ≈ 5, std ≈ 0.8
    for i, fires in enumerate([4, 5, 6, 4, 5, 6, 4, 5, 6, 5]):
        mon.record_daily(
            _df_with_reason("structuring;", n=fires),
            today=date(2025, 4, 1) + timedelta(days=i),
        )
    # Today: 5 (identical to mean) — no drift
    mon.record_daily(_df_with_reason("structuring;", n=5),
                     today=date(2025, 4, 11))
    assert [a for a in mon.detect_drift(today=date(2025, 4, 11)) if a.rule == "structuring"] == []


def test_drift_alert_stringifies():
    from engine.drift import DriftAlert
    a = DriftAlert(rule="foo", today_fires=10, baseline_mean=2.0,
                   baseline_std=1.0, z_score=8.0, baseline_days=10)
    s = str(a)
    assert "DriftAlert[foo]" in s
    assert "z=+8.0" in s or "z=8.0" in s

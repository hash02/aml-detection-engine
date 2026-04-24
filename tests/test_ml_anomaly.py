"""ML anomaly layer tests — determinism + short-circuit + metadata."""

from __future__ import annotations

from datetime import timedelta


def _bulk_rows(row_fn, t0, n=30, outlier_idx=None):
    """n normal-ish rows plus an optional obvious outlier."""
    rows = []
    for i in range(n):
        rows.append(row_fn(f"0x{i:040x}", f"0x{i+1:040x}", 100,
                            t0 + timedelta(minutes=i)))
    if outlier_idx is not None and 0 <= outlier_idx < n:
        rows[outlier_idx]["amount"] = 10_000_000  # clear outlier
    return rows


def test_ml_anomaly_short_circuits_small_inputs(row, t0, build_df):
    from engine.ml_anomaly import fit_predict
    rows = _bulk_rows(row, t0, n=5)
    df = build_df(rows)
    # Rule engine populates these; here we fabricate them
    for col in ("tx_count_in_window", "small_tx_count_in_window",
                "small_tx_count_6h", "fan_in_count"):
        df[col] = 0
    out = fit_predict(df)
    assert (out["ml_anomaly_score"] == 0.0).all()
    assert (out["ml_anomaly_flag"] == False).all()  # noqa: E712


def test_ml_anomaly_is_deterministic(row, t0, build_df):
    """Same input twice → identical anomaly scores (random_state pinned)."""
    from engine.ml_anomaly import fit_predict
    rows = _bulk_rows(row, t0, n=30, outlier_idx=17)
    df1 = build_df(rows)
    df2 = build_df(rows)
    for col in ("tx_count_in_window", "small_tx_count_in_window",
                "small_tx_count_6h", "fan_in_count"):
        df1[col] = 0
        df2[col] = 0

    out1 = fit_predict(df1.copy())
    out2 = fit_predict(df2.copy())
    assert list(out1["ml_anomaly_score"].round(6)) == list(out2["ml_anomaly_score"].round(6))


def test_ml_anomaly_flags_the_outlier(row, t0, build_df):
    from engine.ml_anomaly import fit_predict
    rows = _bulk_rows(row, t0, n=30, outlier_idx=10)
    df = build_df(rows)
    for col in ("tx_count_in_window", "small_tx_count_in_window",
                "small_tx_count_6h", "fan_in_count"):
        df[col] = 0
    out = fit_predict(df)
    # Find the outlier row (highest amount) → it should have the highest score
    outlier_row = out.loc[out["amount"].idxmax()]
    max_score   = out["ml_anomaly_score"].max()
    assert outlier_row["ml_anomaly_score"] == max_score
    assert outlier_row["ml_anomaly_flag"] is True or bool(outlier_row["ml_anomaly_flag"]) is True


def test_ml_anomaly_metadata_shape():
    from engine.ml_anomaly import metadata
    md = metadata()
    assert md["model"] == "IsolationForest"
    assert "version" in md
    assert isinstance(md["features"], list)

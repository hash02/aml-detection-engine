"""Graph-feature tests — enrichment columns + degree/passthrough values."""

from __future__ import annotations

from datetime import timedelta


def test_enrich_adds_feature_columns(row, t0, build_df):
    from engine.graph_features import FEATURE_COLUMNS, enrich
    df = build_df([
        row("0xaaa", "0xbbb", 100, t0),
        row("0xaaa", "0xccc", 100, t0 + timedelta(minutes=1)),
    ])
    out = enrich(df)
    for col in FEATURE_COLUMNS:
        assert col in out.columns


def test_out_degree_counts_unique_receivers(row, t0, build_df):
    from engine.graph_features import enrich
    df = build_df([
        row("0xrelay", "0xrx1", 100, t0),
        row("0xrelay", "0xrx2", 100, t0 + timedelta(minutes=1)),
        row("0xrelay", "0xrx3", 100, t0 + timedelta(minutes=2)),
        row("0xother", "0xrx4", 100, t0 + timedelta(minutes=3)),
    ])
    out = enrich(df)
    relay_rows = out[out["sender_id"] == "0xrelay"]
    # NetworkX is dependency; if unavailable, feature is 0.0
    try:
        import networkx  # noqa: F401
        assert (relay_rows["graph_out_degree"] == 3).all()
    except ImportError:
        assert (relay_rows["graph_out_degree"] == 0.0).all()


def test_passthrough_detects_relay_wallets(row, t0, build_df):
    """A wallet that both receives from and sends to many is a relay."""
    try:
        import networkx  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("networkx not installed")

    from engine.graph_features import enrich
    rows = []
    # 3 senders → 0xrelay
    for i in range(3):
        rows.append(row(f"0xsender{i}", "0xrelay", 100, t0 + timedelta(minutes=i)))
    # 0xrelay → 3 receivers
    for i in range(3):
        rows.append(row("0xrelay", f"0xrecv{i}", 100, t0 + timedelta(minutes=10 + i)))
    df = build_df(rows)
    out = enrich(df)
    relay_rows = out[out["sender_id"] == "0xrelay"]
    assert (relay_rows["graph_passthrough"] >= 3).all()


def test_enrich_empty_df_returns_zero_filled():
    import pandas as pd

    from engine.graph_features import FEATURE_COLUMNS, enrich
    out = enrich(pd.DataFrame())
    for col in FEATURE_COLUMNS:
        assert col in out.columns


def test_enrich_missing_columns_safe():
    """Partial frame (no sender_id) shouldn't raise."""
    import pandas as pd

    from engine.graph_features import enrich
    df = pd.DataFrame([{"amount": 100, "timestamp": "2025-04-01"}])
    out = enrich(df)
    assert "graph_out_degree" in out.columns

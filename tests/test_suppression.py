"""Alert-suppression tests — dedup + cooldown behaviour."""

from __future__ import annotations

from datetime import timedelta


def _alert_rows(row_fn, t0, sender, count, base_score=85, reason="structuring;"):
    """Build `count` alert rows for one sender, spaced 2 min apart."""
    rows = []
    for i in range(count):
        r = row_fn(sender, f"0xdest{i:04x}", 9_500, t0 + timedelta(minutes=2 * i))
        r.update({
            "id":         f"tx-{sender[-4:]}-{i}",
            "alert":      True,
            "risk_score": base_score - i,   # decreasing so the first is top
            "risk_level": "CRITICAL",
            "reasons":    reason,
        })
        rows.append(r)
    return rows


def test_suppression_collapses_burst(row, t0, build_df, cfg):
    from engine.suppression import apply_suppression, suppression_stats

    # 6 alerts from the same sender within 10 min, same top rule
    rows = _alert_rows(row, t0, "0xbadbadbadbad", count=6)
    df = build_df(rows)
    out = apply_suppression(df, cfg)
    stats = suppression_stats(out)

    assert stats["alerts"] == 6
    # Representative is the top-scored row; others should be suppressed
    assert stats["suppressed"] >= 5
    assert stats["kept"] == stats["alerts"] - stats["suppressed"]
    assert stats["compression_ratio"] > 0.7


def test_suppression_respects_different_rules(row, t0, build_df, cfg):
    """Same sender, different top rules → different equivalence classes."""
    from engine.suppression import apply_suppression

    rows  = _alert_rows(row, t0, "0xone", count=2, reason="structuring;")
    rows += _alert_rows(row, t0, "0xone", count=2, reason="mixer_touch;")
    df = build_df(rows)
    out = apply_suppression(df, cfg)
    # Two representatives (one per rule), two suppressed
    assert int(out["suppressed"].sum()) == 2


def test_suppression_disabled_is_passthrough(row, t0, build_df, cfg):
    from engine.suppression import apply_suppression
    rows = _alert_rows(row, t0, "0xany", count=5)
    df = build_df(rows)
    cfg2 = {**cfg, "suppression_enabled": False}
    out = apply_suppression(df, cfg2)
    assert int(out["suppressed"].sum()) == 0


def test_suppression_handles_empty_df(cfg):
    import pandas as pd

    from engine.suppression import apply_suppression
    empty = pd.DataFrame(columns=["sender_id", "timestamp", "alert", "reasons",
                                   "risk_score", "id"])
    out = apply_suppression(empty, cfg)
    assert out.empty

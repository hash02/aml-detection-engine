"""UNKNOWN country is missing data, not a high-risk foreign jurisdiction.

Live fetches (etherscan_fetcher.infer_country) label almost every address
UNKNOWN. That used to add high-risk points and, with home_country CA, foreign
points. A real FATF code such as KP must still score.
"""

from __future__ import annotations


def _score(rows, cfg):
    from engine_v11_blockchain import (
        compute_features,
        detect_high_risk_country,
        score_transactions,
    )

    df = compute_features(rows, cfg)
    df = detect_high_risk_country(df, cfg)
    return score_transactions(df, cfg)


def test_unknown_is_not_in_the_high_risk_list(cfg):
    assert "UNKNOWN" not in cfg["high_risk_countries"]
    assert "KP" in cfg["high_risk_countries"]
    assert "XX" in cfg["high_risk_countries"]


def test_ordinary_unknown_country_scores_amount_only(row, t0, build_df, cfg):
    """~$12k ETH to an unlisted address: large_amount only, no country points."""
    from engine_v11_blockchain import score_row

    from engine.explain import score_breakdown

    df = build_df([
        row("0xa1", "0xb1", 12_000, t0, profile="UNKNOWN", country="UNKNOWN"),
        row("0xa2", "0xb2", 500, t0, profile="UNKNOWN", country="UNKNOWN"),
        row("0xa3", "0xb3", 25_000, t0, profile="UNKNOWN", country="UNKNOWN"),
    ])
    out = _score(df, cfg)

    by_amount = {int(r.amount): r for _, r in out.iterrows()}

    twelve = by_amount[12_000]
    assert int(twelve.risk_score) == 30
    assert "large_amount" in twelve.reasons
    assert "foreign_country" not in twelve.reasons
    assert "high_risk_jurisdiction" not in twelve.reasons
    assert bool(twelve.high_risk_country_flag) is False
    rules = {p["rule"] for p in score_breakdown(twelve.to_dict(), cfg)}
    assert rules == {"large_amount"}
    scored, _ = score_row(twelve, cfg)
    assert scored == 30

    assert int(by_amount[500].risk_score) == 0
    assert "foreign_country" not in by_amount[500].reasons
    assert int(by_amount[25_000].risk_score) == 30


def test_known_countries_keep_their_country_points(row, t0, build_df, cfg):
    df = build_df([
        row("0xc1", "0xd1", 12_000, t0, profile="UNKNOWN", country="US"),
        row("0xc2", "0xd2", 12_000, t0, profile="UNKNOWN", country="CA"),
        row("0xc3", "0xd3", 12_000, t0, profile="UNKNOWN", country="KP"),
    ])
    out = _score(df, cfg)
    by_country = {r.country: r for _, r in out.iterrows()}

    us = by_country["US"]
    assert int(us.risk_score) == 50  # large 30 + foreign 20
    assert "foreign_country" in us.reasons
    assert "high_risk_jurisdiction" not in us.reasons

    ca = by_country["CA"]
    assert int(ca.risk_score) == 30  # home country, large only
    assert "foreign_country" not in ca.reasons

    kp = by_country["KP"]
    assert int(kp.risk_score) == 65  # large 30 + foreign 20 + high-risk half 15
    assert "foreign_country" in kp.reasons
    assert "high_risk_jurisdiction" in kp.reasons
    assert bool(kp.high_risk_country_flag) is True

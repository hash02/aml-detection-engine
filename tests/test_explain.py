"""Score-breakdown tests — breakdown sum matches score_row()."""

from __future__ import annotations


def _score_row(row: dict, cfg: dict) -> int:
    from engine_v11_blockchain import score_row
    score, _ = score_row(row, cfg)
    return score


def _breakdown_total(row: dict, cfg: dict) -> int:
    """Emulate score_row's final rounding: sum × profile_multiplier."""
    from engine_v11_blockchain import DEFAULT_PROFILE, PROFILE_CONFIG

    from engine.explain import score_breakdown
    pcfg = PROFILE_CONFIG.get(row.get("sender_profile"), PROFILE_CONFIG[DEFAULT_PROFILE])
    raw  = sum(part["points"] for part in score_breakdown(row, cfg))
    return round(raw * pcfg["score_multiplier"])


def test_breakdown_empty_for_clean_row(row, t0, cfg):
    from engine.explain import score_breakdown
    r = row("0xaa", "0xbb", 50, t0)
    # Engine features that score_row reads
    r.update({
        "tx_count_in_window": 1, "small_tx_count_in_window": 0,
        "small_tx_count_6h": 0, "fan_in_count": 0,
    })
    parts = score_breakdown(r, cfg)
    assert parts == []


def test_breakdown_matches_score_for_large_amount_row(row, t0, cfg):
    r = row("0xaa", "0xbb", 250_000, t0)  # triggers large_amount
    r.update({
        "tx_count_in_window": 1, "small_tx_count_in_window": 0,
        "small_tx_count_6h": 0, "fan_in_count": 0,
    })
    assert _breakdown_total(r, cfg) == _score_row(r, cfg)


def test_breakdown_matches_score_for_multi_rule_row(row, t0, cfg):
    """Multiple flags simultaneously — breakdown total equals score_row."""
    r = row("0xaa", "0xbb", 250_000, t0, country="KP")
    r.update({
        "tx_count_in_window": 20, "small_tx_count_in_window": 5,
        "small_tx_count_6h": 8, "fan_in_count": 6,
        "ofac_flag": True, "ofac_address": "0xsanctioned",
        "phish_flag": True, "phish_source": "metamask_phish_receiver",
    })
    assert _breakdown_total(r, cfg) == _score_row(r, cfg)


def test_breakdown_surfaces_rule_names(row, t0, cfg):
    from engine.explain import score_breakdown
    r = row("0xaa", "0xbb", 250_000, t0)
    r.update({
        "tx_count_in_window": 1, "small_tx_count_in_window": 0,
        "small_tx_count_6h": 0, "fan_in_count": 0,
        "ofac_flag": True, "ofac_address": "0xsanc",
    })
    rules = {p["rule"] for p in score_breakdown(r, cfg)}
    assert "large_amount" in rules
    assert "OFAC_SDN_MATCH" in rules

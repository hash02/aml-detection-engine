"""Golden-case tests for the v13 rules — drainer_signature + address_poisoning."""

from __future__ import annotations

from datetime import timedelta


# ─────────────────────────────────────────────────────────────────────
# Drainer signature  (v13)
# ─────────────────────────────────────────────────────────────────────
def test_drainer_signature_fires_on_multi_asset_drain(row, t0, build_df):
    """4 distinct tokens moved from victim → drainer inside 90s → flag."""
    from engine_v11_blockchain import CONFIG, detect_drainer_signature

    victim  = "0x00000000000000000000000000000000000victim"
    drainer = "0x0000000000000000000000000000000000drainer"
    tokens  = ["0xUSDC", "0xUSDT", "0xWETH", "0xDAI"]
    rows = []
    for i, token in enumerate(tokens):
        base = row(victim, drainer, 10_000, t0 + timedelta(seconds=10 * i))
        base["asset_type"]     = "erc20"
        base["token_contract"] = token
        base["token_symbol"]   = token.lstrip("0x")
        rows.append(base)
    df = build_df(rows)
    out = detect_drainer_signature(df, CONFIG)
    # At least the final row (sees all 4 assets in-window) should fire
    assert out["drainer_flag"].sum() >= CONFIG["drainer_min_assets"]
    assert (out.loc[out["drainer_flag"], "drainer_assets"] >= CONFIG["drainer_min_assets"]).all()


def test_drainer_signature_no_ops_on_legacy_schema(row, t0, build_df):
    """Without v13 columns, the detector is a no-op."""
    from engine_v11_blockchain import CONFIG, detect_drainer_signature

    df = build_df([
        row("0xaaa", "0xbbb", 100, t0),
        row("0xaaa", "0xbbb", 100, t0 + timedelta(seconds=30)),
    ])
    out = detect_drainer_signature(df, CONFIG)
    assert out["drainer_flag"].sum() == 0


# ─────────────────────────────────────────────────────────────────────
# Address poisoning  (v13)
# ─────────────────────────────────────────────────────────────────────
def test_address_poisoning_fires_on_lookalike_dust(row, t0, build_df):
    """Dust from a first-4/last-4 lookalike sender → flag."""
    from engine_v11_blockchain import CONFIG, detect_address_poisoning

    victim = "0x1111111111111111111111111111111111111111"
    legit  = "0xabcd22222222222222222222222222222222dcba"
    # Lookalike: same first 4 (abcd) + last 4 (dcba), different middle
    poison = "0xabcd99999999999999999999999999999999dcba"

    df = build_df([
        # Legit large-value prior transaction so poison has a target to mimic
        row(legit,  victim, 5_000, t0),
        # The poison dust: arrives 3 days later
        row(poison, victim, 0.01,  t0 + timedelta(days=3)),
    ])
    out = detect_address_poisoning(df, CONFIG)
    poison_row = out[out["amount"] < 1.0]
    assert poison_row["poison_flag"].iloc[0] is True or bool(poison_row["poison_flag"].iloc[0]) is True
    assert poison_row["poison_target"].iloc[0] == legit


def test_address_poisoning_does_not_fire_on_matching_legit_counterparty(row, t0, build_df):
    from engine_v11_blockchain import CONFIG, detect_address_poisoning

    victim = "0x1111111111111111111111111111111111111111"
    legit  = "0xabcd22222222222222222222222222222222dcba"

    df = build_df([
        row(legit, victim, 5_000, t0),
        # Same legit sender dusting (not lookalike) — no poisoning
        row(legit, victim, 0.01,  t0 + timedelta(days=1)),
    ])
    out = detect_address_poisoning(df, CONFIG)
    assert out["poison_flag"].sum() == 0


def test_address_poisoning_ignores_non_dust(row, t0, build_df):
    from engine_v11_blockchain import CONFIG, detect_address_poisoning

    victim = "0x1111111111111111111111111111111111111111"
    legit  = "0xabcd22222222222222222222222222222222dcba"
    lookalike = "0xabcd99999999999999999999999999999999dcba"

    df = build_df([
        row(legit,     victim, 5_000, t0),
        row(lookalike, victim, 2_500, t0 + timedelta(days=1)),  # real money — not poison
    ])
    out = detect_address_poisoning(df, CONFIG)
    assert out["poison_flag"].sum() == 0

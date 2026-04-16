"""
Golden-case tests — each rule is asserted to fire on a hand-crafted
positive case and NOT fire on a benign control. Small, fast, no network.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# OFAC SDN hit
# ─────────────────────────────────────────────────────────────────────
def test_ofac_hit_fires_on_sanctioned_sender(row, t0, build_df):
    from engine_v11_blockchain import CONFIG, detect_ofac_hit

    # Ronin Bridge exploiter — bundled in the OFAC baseline
    ronin = "0x098b716b8aaf21512996dc57eb0615e2383e2f96"
    df = build_df([
        row(ronin, "0x0000000000000000000000000000000000000001", 1_000_000, t0),
        row("0x0000000000000000000000000000000000000002",
            "0x0000000000000000000000000000000000000003", 50, t0),
    ])
    out = detect_ofac_hit(df, CONFIG)
    assert out.loc[0, "ofac_flag"] is True or out.loc[0, "ofac_flag"] == True  # noqa: E712
    assert bool(out.loc[1, "ofac_flag"]) is False


def test_ofac_hit_empty_df(cfg):
    from engine_v11_blockchain import detect_ofac_hit

    empty = pd.DataFrame(columns=["sender_id", "receiver_id", "amount", "timestamp"])
    out = detect_ofac_hit(empty, cfg)
    assert out.empty
    assert "ofac_flag" in out.columns


# ─────────────────────────────────────────────────────────────────────
# Phishing / no-KYC off-ramp hit  (v12)
# ─────────────────────────────────────────────────────────────────────
def test_phish_hit_fires_on_offramp_receiver(row, t0, build_df):
    from engine_v11_blockchain import CONFIG, detect_phish_hit

    # FixedFloat — bundled in BASELINE_LRT_OFFRAMPS
    offramp = "0x4e5b2e1dc63f6b91cb6cd759936495434c7e972f"
    df = build_df([
        row("0x00000000000000000000000000000000000000aa", offramp, 5_000, t0),
        row("0x00000000000000000000000000000000000000bb",
            "0x00000000000000000000000000000000000000cc", 5_000, t0),
    ])
    out = detect_phish_hit(df, CONFIG)
    assert bool(out.loc[0, "phish_flag"]) is True
    assert bool(out.loc[1, "phish_flag"]) is False
    assert out.loc[0, "phish_source"] == "no_kyc_offramp_receiver"


# ─────────────────────────────────────────────────────────────────────
# Sub-threshold tranching  (v12)
# ─────────────────────────────────────────────────────────────────────
def test_sub_threshold_tranching_fires_on_three_tranches(row, t0, build_df):
    from engine_v11_blockchain import CONFIG, detect_sub_threshold_tranching

    sender = "0x00000000000000000000000000000000000000d1"
    rx     = "0x00000000000000000000000000000000000000d2"
    # Three $9k txns over 6 hours — classic human tranching
    df = build_df([
        row(sender, rx, 9_100, t0),
        row(sender, rx, 9_500, t0 + timedelta(hours=2)),
        row(sender, rx, 8_800, t0 + timedelta(hours=4)),
        # Control row: small amount outside the band
        row(sender, rx,   200, t0 + timedelta(hours=5)),
    ])
    out = detect_sub_threshold_tranching(df, CONFIG)
    # At least the last-of-three rows should flip on
    assert out["tranching_flag"].sum() >= 3
    # Control (amount=200) must NOT be flagged
    assert bool(out.loc[out["amount"] == 200, "tranching_flag"].iloc[0]) is False


def test_sub_threshold_tranching_does_not_fire_on_single_tranche(row, t0, build_df):
    from engine_v11_blockchain import CONFIG, detect_sub_threshold_tranching

    sender = "0x00000000000000000000000000000000000000e1"
    rx     = "0x00000000000000000000000000000000000000e2"
    df = build_df([
        row(sender, rx, 9_500, t0),
        row(sender, rx,   500, t0 + timedelta(minutes=10)),
    ])
    out = detect_sub_threshold_tranching(df, CONFIG)
    assert out["tranching_flag"].sum() == 0


# ─────────────────────────────────────────────────────────────────────
# Machine cadence  (v12)
# ─────────────────────────────────────────────────────────────────────
def test_machine_cadence_fires_on_uniform_spacing(row, t0, build_df):
    from engine_v11_blockchain import CONFIG, detect_machine_cadence

    sender = "0x00000000000000000000000000000000000000f1"
    # 8 txns each exactly 60s apart — inter-arrival CV ≈ 0
    rows = [
        row(sender,
            f"0x{i:040x}",
            1_000,
            t0 + timedelta(seconds=60 * i))
        for i in range(1, 9)
    ]
    df = build_df(rows)
    out = detect_machine_cadence(df, CONFIG)
    assert out["cadence_flag"].all()
    # CV must be tiny
    assert (out["cadence_cv"] < 0.01).all()


def test_machine_cadence_does_not_fire_on_irregular_spacing(row, t0, build_df):
    from engine_v11_blockchain import CONFIG, detect_machine_cadence

    sender = "0x0000000000000000000000000000000000000aa1"
    # Irregular inter-arrival — 30s, 2min, 45s, 10min, 1min, 5min
    offsets = [0, 30, 150, 195, 795, 855, 1155]
    rows = [
        row(sender, f"0x{i:040x}", 1_000,
            t0 + timedelta(seconds=s))
        for i, s in enumerate(offsets)
    ]
    df = build_df(rows)
    out = detect_machine_cadence(df, CONFIG)
    assert out["cadence_flag"].sum() == 0


# ─────────────────────────────────────────────────────────────────────
# Sybil fan-in  (v12)
# ─────────────────────────────────────────────────────────────────────
def test_sybil_fan_in_fires_on_six_similar_senders(row, t0, build_df):
    from engine_v11_blockchain import CONFIG, detect_sybil_fan_in

    receiver = "0x0000000000000000000000000000000000000777"
    # 6 distinct senders, each sending $1000 ± <5%, within 10 min
    amounts = [1000, 1020, 995, 1010, 985, 1005]
    rows = [
        row(f"0x{i:040x}", receiver, amt,
            t0 + timedelta(minutes=i))
        for i, amt in enumerate(amounts, start=1)
    ]
    df = build_df(rows)
    out = detect_sybil_fan_in(df, CONFIG)
    assert out["sybil_flag"].sum() >= 6
    assert (out.loc[out["sybil_flag"], "sybil_senders"] >= 6).all()


def test_sybil_fan_in_does_not_fire_on_wide_amount_spread(row, t0, build_df):
    from engine_v11_blockchain import CONFIG, detect_sybil_fan_in

    receiver = "0x0000000000000000000000000000000000000888"
    # 6 senders but amounts span 300 – 5000 (spread >>5%)
    amounts = [300, 800, 1500, 2200, 3500, 5000]
    rows = [
        row(f"0x{i:040x}", receiver, amt,
            t0 + timedelta(minutes=i))
        for i, amt in enumerate(amounts, start=1)
    ]
    df = build_df(rows)
    out = detect_sybil_fan_in(df, CONFIG)
    assert out["sybil_flag"].sum() == 0


# ─────────────────────────────────────────────────────────────────────
# Smoke test — the full pipeline on a tiny frame runs without raising
# ─────────────────────────────────────────────────────────────────────
def test_full_pipeline_runs_clean(row, t0, build_df, cfg):
    """Exercises score_row + every detector on a 6-row mixed input."""
    from engine_v11_blockchain import (
        compute_features,
        detect_bridge_hops,
        detect_concentrated_inflow,
        detect_coordinated_burst,
        detect_dormant_activation,
        detect_exchange_avoidance,
        detect_exit_rush,
        detect_flash_loan_burst,
        detect_high_risk_country,
        detect_layering,
        detect_layering_deep,
        detect_machine_cadence,
        detect_mixer_touch,
        detect_novel_wallet_dump,
        detect_ofac_hit,
        detect_peel_chain,
        detect_phish_hit,
        detect_rapid_succession,
        detect_smurfing,
        detect_sub_threshold_tranching,
        detect_sybil_fan_in,
        detect_wash_cycle,
        score_transactions,
    )

    df = build_df([
        row("0xaa", "0xbb", 500, t0),
        row("0xaa", "0xbb", 500, t0 + timedelta(minutes=5)),
        row("0xcc", "0xdd", 9_500, t0 + timedelta(minutes=10)),
        row("0xcc", "0xdd", 9_200, t0 + timedelta(hours=3)),
        row("0xee", "0xff", 200_000, t0 + timedelta(hours=4)),
        row("0xgg", "0xhh", 10, t0 + timedelta(hours=5)),
    ])
    df = compute_features(df, cfg)
    df, _ = detect_layering(df, cfg)
    for fn in (
        detect_mixer_touch, detect_bridge_hops, detect_peel_chain,
        detect_novel_wallet_dump, detect_concentrated_inflow, detect_ofac_hit,
        detect_flash_loan_burst, detect_coordinated_burst, detect_dormant_activation,
        detect_wash_cycle, detect_smurfing, detect_exit_rush,
        detect_rapid_succession, detect_high_risk_country,
        detect_exchange_avoidance, detect_layering_deep, detect_phish_hit,
        detect_sub_threshold_tranching, detect_machine_cadence, detect_sybil_fan_in,
    ):
        df = fn(df, cfg)
    out = score_transactions(df, cfg)
    assert "risk_score" in out.columns
    assert len(out) == 6

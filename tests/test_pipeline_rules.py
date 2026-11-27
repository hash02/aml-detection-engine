"""pipeline.apply_rules runs every scoring detector, including v12 and v13."""

from __future__ import annotations

import inspect
from datetime import timedelta


def test_apply_rules_calls_v12_and_v13_in_score_order():
    from pipeline import apply_rules

    source = inspect.getsource(apply_rules).split('"""', 2)[-1]
    names = [
        "detect_layering",
        "detect_mixer_touch",
        "detect_bridge_hops",
        "detect_peel_chain",
        "detect_novel_wallet_dump",
        "detect_concentrated_inflow",
        "detect_ofac_hit",
        "detect_flash_loan_burst",
        "detect_coordinated_burst",
        "detect_dormant_activation",
        "detect_wash_cycle",
        "detect_smurfing",
        "detect_exit_rush",
        "detect_rapid_succession",
        "detect_high_risk_country",
        "detect_exchange_avoidance",
        "detect_layering_deep",
        "detect_phish_hit",
        "detect_sub_threshold_tranching",
        "detect_machine_cadence",
        "detect_sybil_fan_in",
        "detect_drainer_signature",
        "detect_address_poisoning",
        "score_transactions",
    ]
    positions = [source.index(name) for name in names]
    assert positions == sorted(positions)


def test_apply_rules_fires_drainer_and_runs_address_poisoning(row, t0, build_df):
    from engine_v11_blockchain import CONFIG

    from pipeline import apply_rules

    victim = "0x" + "ab" * 20
    drainer = "0x" + "cd" * 20
    rows = []
    for i, token in enumerate(("0xusdc", "0xusdt")):
        base = row(victim, drainer, 5_000, t0 + timedelta(seconds=20 * i))
        base["asset_type"] = "erc20"
        base["token_contract"] = token
        base["token_symbol"] = token
        rows.append(base)
    out = apply_rules(build_df(rows), CONFIG)
    assert int(out["drainer_flag"].sum()) == 2
    assert "poison_flag" in out.columns
    assert int(out["poison_flag"].sum()) == 0
    assert "phish_flag" in out.columns

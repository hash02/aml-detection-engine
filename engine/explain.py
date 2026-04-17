"""
engine/explain.py — Per-rule score contribution breakdown
==========================================================

`score_row()` produces a total `risk_score` and a `reasons` string.
That string tells analysts WHICH rules fired but not HOW MUCH each
one contributed. For compliance review and model governance, that's
the missing piece.

This module re-runs the same arithmetic in a breakdown-aware form:
for each rule that fired, we record (rule_name, points_added).
Output is small, purely additive, and suitable for:
  - an analyst UI showing score components per alert
  - SAR narratives explaining the score
  - model-governance review (does any one rule dominate?)
  - regression tests on weight changes

This is deliberately *mechanical*. It mirrors score_row() exactly so
the numbers always match. When score_row() gains a new rule, extend
both places or this goes stale.
"""

from __future__ import annotations

from typing import Any


def _pcfg(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    from engine_v11_blockchain import DEFAULT_PROFILE, PROFILE_CONFIG
    return PROFILE_CONFIG.get(row.get("sender_profile"), PROFILE_CONFIG[DEFAULT_PROFILE])


def score_breakdown(row: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return an ordered list of rule contributions for a single row.

    Each entry is ``{"rule": str, "points": float, "detail": str}``.
    The sum of ``points`` — times the profile multiplier — equals the
    integer risk_score that score_row() produced.
    """
    p = _pcfg(row, cfg)
    parts: list[dict[str, Any]] = []

    # ── v6 rules ────────────────────────────────
    if row.get("amount", 0) >= p["large_amount_threshold"]:
        parts.append({"rule": "large_amount", "points": cfg["w_large"],
                      "detail": f"amount >= {p['large_amount_threshold']:,}"})
    if row.get("tx_count_in_window", 0) >= p["velocity_tx_count_threshold"]:
        parts.append({"rule": "velocity_many_tx", "points": cfg["w_velocity"],
                      "detail": f"tx_count_in_window >= {p['velocity_tx_count_threshold']}"})
    struct_60m = row.get("small_tx_count_in_window", 0) >= p["struct_min_count"]
    struct_6h  = row.get("small_tx_count_6h", 0) >= p.get("struct_min_count_6h", p["struct_min_count"] * 2)
    if struct_60m or struct_6h:
        parts.append({"rule": "structuring", "points": cfg["w_structuring"],
                      "detail": "small-amount bunches"})
    if row.get("fan_in_count", 0) >= cfg["fan_in_threshold"]:
        parts.append({"rule": "fan_in", "points": cfg["w_fan_in"],
                      "detail": f"{int(row['fan_in_count'])} unique senders"})

    is_foreign = str(row.get("country", "")).upper() != cfg["home_country"].upper()
    other_hit  = any((
        row.get("amount", 0) >= p["large_amount_threshold"],
        row.get("tx_count_in_window", 0) >= p["velocity_tx_count_threshold"],
        struct_60m or struct_6h,
        row.get("fan_in_count", 0) >= cfg["fan_in_threshold"],
    ))
    if is_foreign and other_hit:
        parts.append({"rule": "foreign_country", "points": cfg["w_foreign"],
                      "detail": f"country={row.get('country')}"})

    # ── v7 rules ────────────────────────────────
    if row.get("layering_flag"):
        parts.append({"rule": "layering_cycle", "points": cfg["w_layering"],
                      "detail": str(row.get("layering_chain", ""))[:120]})
    if row.get("mixer_flag"):
        mtype = row.get("mixer_type", "mixer_deposit")
        mult  = 1.3 if mtype == "mixer_withdraw" else 1.0
        parts.append({"rule": mtype, "points": cfg["w_mixer_touch"] * mult,
                      "detail": "known mixer counterparty"})
    if row.get("bridge_flag"):
        hops = row.get("bridge_hop_count", 2)
        pts  = cfg["w_bridge_hop"] * min(hops / cfg["bridge_hop_threshold"], 2.0)
        parts.append({"rule": "bridge_hop", "points": pts, "detail": f"{int(hops)} bridges"})
    if row.get("peel_flag"):
        parts.append({"rule": "peel_chain", "points": cfg["w_peel_chain"],
                      "detail": str(row.get("peel_chain", ""))[:120]})

    # ── v8 rules ────────────────────────────────
    if row.get("ofac_flag"):
        parts.append({"rule": "OFAC_SDN_MATCH", "points": cfg["w_ofac_hit"],
                      "detail": f"match: {row.get('ofac_address', '')[:20]}..."})
    if row.get("flash_flag"):
        fc = row.get("flash_count", cfg["flash_min_tx_count"])
        parts.append({"rule": "flash_loan_burst",
                      "points": cfg["w_flash_loan"] * min(fc / cfg["flash_min_tx_count"], 2.0),
                      "detail": f"{int(fc)} txns in {cfg['flash_window_seconds']}s"})
    if row.get("burst_flag"):
        bc = row.get("burst_count", cfg["burst_min_senders"])
        parts.append({"rule": "coordinated_burst",
                      "points": cfg["w_coordinated_burst"] * min(bc / cfg["burst_min_senders"], 2.0),
                      "detail": f"{int(bc)} senders converging"})
    if row.get("novel_dump_flag"):
        af = max(0.5, 1.0 - (row.get("sender_active_days", 0) / cfg["novel_dump_max_active_days"]))
        parts.append({"rule": "novel_wallet_dump", "points": cfg["w_novel_dump"] * af,
                      "detail": f"active_days={row.get('sender_active_days', 0)}"})
    if row.get("conc_inflow_flag"):
        sc = row.get("conc_inflow_count", cfg["conc_inflow_min_senders"])
        parts.append({"rule": "concentrated_inflow",
                      "points": cfg["w_concentrated_inflow"] * min(sc / cfg["conc_inflow_min_senders"], 2.0),
                      "detail": f"{int(sc)} unique senders"})

    # ── v10 / v11 rules ─────────────────────────
    if row.get("dormant_flag"):
        years = row.get("dormant_years", 1.0)
        scale = cfg.get("dormant_scale_years", 5)
        factor = 1.0 + min((years - 1) / (scale - 1), 1.0)
        parts.append({"rule": "dormant_activation",
                      "points": cfg["w_dormant_activation"] * factor,
                      "detail": f"{years:.1f}yr dormant"})
    if row.get("wash_flag"):
        parts.append({"rule": "wash_cycle", "points": cfg["w_wash_cycle"],
                      "detail": "circular flow between wallets"})
    if row.get("smurf_flag"):
        sc = row.get("smurf_count", cfg["smurf_min_wallets"])
        parts.append({"rule": "smurfing",
                      "points": cfg["w_smurfing"] * min(sc / cfg["smurf_min_wallets"], 2.0),
                      "detail": f"{int(sc)} coordinating wallets"})
    if row.get("exit_rush_flag"):
        parts.append({"rule": "exit_rush", "points": cfg["w_exit_rush"],
                      "detail": "novel wallet → rapid bridge exit"})
    if row.get("rapid_flag"):
        rc = row.get("rapid_receivers", cfg["rapid_min_receivers"])
        parts.append({"rule": "rapid_succession",
                      "points": cfg["w_rapid_succession"] * min(rc / cfg["rapid_min_receivers"], 2.0),
                      "detail": f"{int(rc)} unique receivers in {cfg['rapid_window_minutes']}m"})
    if row.get("high_risk_country_flag"):
        other_signals = any([
            row.get("wash_flag"),     row.get("smurf_flag"),
            row.get("exit_rush_flag"),row.get("rapid_flag"),
            row.get("ofac_flag"),     row.get("mixer_flag"),
            row.get("novel_dump_flag"), row.get("layering_flag"),
        ])
        pts = cfg["w_high_risk_country"] * (1.0 if other_signals else 0.5)
        parts.append({"rule": "high_risk_country", "points": pts,
                      "detail": "FATF blacklist / grey list"})
    if row.get("ex_avoid_flag"):
        hops = row.get("ex_avoid_hops", cfg["exchange_avoidance_min_hops"])
        parts.append({"rule": "exchange_avoidance",
                      "points": cfg["w_exchange_avoidance"] * min(hops / cfg["exchange_avoidance_min_hops"], 2.0),
                      "detail": f"{int(hops)} non-exchange hops"})
    if row.get("deep_peel_flag"):
        depth = row.get("deep_peel_depth", cfg["deep_peel_min_hops"])
        parts.append({"rule": "layering_deep",
                      "points": cfg["w_layering_deep"] * min(depth / cfg["deep_peel_min_hops"], 2.0),
                      "detail": f"{int(depth)} hops"})

    # ── v12 rules ───────────────────────────────
    if row.get("phish_flag"):
        parts.append({"rule": "phishing_hit", "points": cfg["w_phishing_hit"],
                      "detail": str(row.get("phish_source", ""))})
    if row.get("tranching_flag"):
        tc = row.get("tranching_count", cfg["tranching_min_count"])
        parts.append({"rule": "sub_threshold_tranching",
                      "points": cfg["w_sub_threshold_tranching"] * min(tc / cfg["tranching_min_count"], 2.0),
                      "detail": f"{int(tc)} sub-$10k tranches"})
    if row.get("cadence_flag"):
        parts.append({"rule": "machine_cadence", "points": cfg["w_machine_cadence"],
                      "detail": f"CV={row.get('cadence_cv', 0):.3f}"})
    if row.get("sybil_flag"):
        sc = row.get("sybil_senders", cfg["sybil_min_senders"])
        parts.append({"rule": "sybil_fan_in",
                      "points": cfg["w_sybil_fan_in"] * min(sc / cfg["sybil_min_senders"], 2.0),
                      "detail": f"{int(sc)} similar-amount senders"})

    # ── v13 rules ───────────────────────────────
    if row.get("drainer_flag"):
        na = row.get("drainer_assets", cfg["drainer_min_assets"])
        parts.append({"rule": "drainer_signature",
                      "points": cfg["w_drainer_signature"] * min(na / cfg["drainer_min_assets"], 2.0),
                      "detail": f"{int(na)} distinct assets"})
    if row.get("poison_flag"):
        parts.append({"rule": "address_poisoning", "points": cfg["w_address_poisoning"],
                      "detail": f"lookalike of {row.get('poison_target', '')[:16]}..."})

    # Round each and return (score_row() applies the profile multiplier
    # to the sum at the end, so we report pre-multiplier points here).
    return [
        {**entry, "points": round(float(entry["points"]), 1)}
        for entry in parts
    ]

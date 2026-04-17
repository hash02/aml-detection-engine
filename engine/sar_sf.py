"""
engine/sar_sf.py — SAR-SF-compatible alert export
==================================================

FinCEN's Suspicious Activity Report (SAR-SF) is canonically filed as
XML via the BSA E-Filing system. Most institutions maintain an internal
JSON representation that their SAR-SF submission pipeline consumes.

This module produces that internal JSON — pragmatic, field-compatible
with the SAR-SF sections most analysts fill by hand, and readable by a
compliance officer without an XML parser.

Mapping (SAR-SF section → JSON key):
  Part I   Subject Information       → `subjects` (list)
  Part II  Suspicious Activity Info  → `suspicious_activity`
  Part III Information about the     → `financial_institution`
           Financial Institution Where Activity Occurred
  Part IV  Filer Contact Information → `filer`  (optional, operator-set)
  Part V   Narrative                 → `narrative`

Typology codes use the FinCEN-published SAR-SF activity code set where
a clean mapping exists (e.g. 906 Structuring, 809 Money laundering,
501 Terrorist Financing). Unknown typologies fall back to the
conservative "901 Other suspicious activities".

Usage:
    from engine.sar_sf import build_sar_sf_report
    report = build_sar_sf_report(alert_row, narrative_dict)
    json.dumps(report, indent=2)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from typing import Any

# Mapping from engine rule-reason tags → SAR-SF activity codes.
# Codes are the categorical ones FinCEN accepts in Part II.
RULE_TO_ACTIVITY_CODE: dict[str, str] = {
    "structuring":              "906",  # Structuring
    "sub_threshold_tranching":  "906",
    "smurfing":                 "906",
    "layering_cycle":           "809",  # Money laundering
    "layering_deep":            "809",
    "peel_chain":               "809",
    "wash_cycle":               "809",
    "mixer_touch":              "809",
    "mixer_withdraw":           "809",
    "bridge_hop":               "809",
    "OFAC_SDN_MATCH":           "501",  # Terrorist financing / sanctions
    "phishing_hit":             "912",  # Cyber event
    "flash_loan_burst":         "912",
    "novel_wallet_dump":        "901",  # Other
    "concentrated_inflow":      "906",
    "sybil_fan_in":             "906",
    "machine_cadence":          "912",
    "exit_rush":                "809",
    "rapid_succession":         "906",
    "exchange_avoidance":       "809",
    "high_risk_jurisdiction":   "901",
    "dormant_activation":       "901",
    "coordinated_burst":        "912",
    "large_amount":             "901",
    "velocity_many_tx":         "901",
    "fan_in":                   "906",
    "foreign_country":          "901",
}


def _parse_reasons(reasons: str) -> list[str]:
    """Turn 'structuring;layering_cycle;...' into a clean list."""
    parts: list[str] = []
    for p in (reasons or "").split(";"):
        p = p.strip()
        if not p or p.startswith("profile_") or p.startswith("foreign_context"):
            continue
        # Drop numeric/hex suffixes (e.g. bridge_hop_3 → bridge_hop)
        base = p.split("_cv_")[0]
        for tag in RULE_TO_ACTIVITY_CODE:
            if base.startswith(tag):
                parts.append(tag)
                break
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _activity_codes(rules: list[str]) -> list[str]:
    codes = [RULE_TO_ACTIVITY_CODE.get(r, "901") for r in rules]
    # Unique, stable order
    seen: set[str] = set()
    out: list[str] = []
    for c in codes:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def build_sar_sf_report(
    alert_row: dict[str, Any],
    narrative: dict[str, Any] | None = None,
    filer: dict[str, Any] | None = None,
    institution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a SAR-SF-compatible JSON dict for a single alert.

    `alert_row` must include: sender_id, receiver_id, amount, timestamp,
    risk_score, risk_level, reasons (the reason string produced by
    score_row()).

    `narrative`, `filer`, and `institution` are optional — pass them in
    from the running Streamlit session / CLI config.
    """
    rules  = _parse_reasons(str(alert_row.get("reasons", "")))
    codes  = _activity_codes(rules)
    ts     = alert_row.get("timestamp")
    if hasattr(ts, "isoformat"):
        ts_iso = ts.isoformat()
    else:
        ts_iso = str(ts)

    # SAR-SF control number must be a stable 14-digit numeric.
    control_seed = f"{alert_row.get('id', '')}|{alert_row.get('sender_id', '')}|{ts_iso}"
    control_number = int(hashlib.sha256(control_seed.encode()).hexdigest()[:14], 16) % (10**14)

    subjects = [
        {
            "role": "subject",
            "identification_type": "on_chain_wallet",
            "identifier": alert_row.get("sender_id"),
            "notes": "Sender wallet on the alerting transaction",
        },
        {
            "role": "counterparty",
            "identification_type": "on_chain_wallet",
            "identifier": alert_row.get("receiver_id"),
            "notes": "Receiver wallet on the alerting transaction",
        },
    ]

    report: dict[str, Any] = {
        "form_type":         "SAR-SF",
        "form_version":      "2024-07",
        "filing_type":       "initial",
        "control_number":    f"{control_number:014d}",
        "generated_at_utc":  _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "subjects":          subjects,
        "suspicious_activity": {
            "activity_date":          ts_iso,
            "amount_involved_usd":    float(alert_row.get("amount", 0) or 0),
            "activity_codes":         codes,
            "rules_fired":            rules,
            "risk_score":             int(alert_row.get("risk_score", 0) or 0),
            "risk_level":             alert_row.get("risk_level"),
            "cumulative_amount_usd":  float(alert_row.get("amount", 0) or 0),
        },
        "financial_institution": institution or {
            "type":            "crypto_asset_service_provider",
            "jurisdiction":    "US",
            "legal_name":      None,
            "ein":             None,
        },
        "filer":     filer or {
            "name":            None,
            "title":           None,
            "contact_phone":   None,
            "contact_email":   None,
        },
        "narrative": {
            "summary":      (narrative or {}).get("summary", ""),
            "signals":      (narrative or {}).get("signals", []),
            "recommended_actions": (narrative or {}).get("rec_actions", []),
        },
    }
    return report

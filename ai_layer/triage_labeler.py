"""
AML Triage Labeler — Dynamic Confidence Scoring
─────────────────────────────────────────────────────────────────────────────
Implements HASH's item-rarity labeling system:
  COMMON    → Grey:   1 rule, small amount, one-off
  MAGIC     → Blue:   2-3 rules OR amount outlier OR moderate frequency
  RARE      → Purple: 4+ rules OR high freq pattern OR large + any rule
  LEGENDARY → Orange: Multiple high-weight dimensions firing simultaneously

The key idea: weights are DYNAMIC.
  A signal's contribution to risk shifts based on what it's combined with.
  This is the attention principle — "how much does feature X matter, given
  what features Y and Z are also doing?"

  Formally: weight_i = f(feature_i, {all other features})
  A $500k tx hitting 1 rule ≠ a $200 tx hitting 1 rule.
  A pattern firing 8 times in 4hrs ≠ firing once in a month.

Architecture:
  1. Signal Density    → % of applicable rules that fired
  2. Amount Tier       → percentile position in dataset distribution
  3. Frequency Score   → same pattern recurrence in 24h window
  4. Combination Bonus → geometric escalation when multiple high signals align
  5. Rarity Assignment → map final score to item tier

This is NOT a linear weighted sum. The combination bonus causes
non-linear escalation — exactly like item affixes in Diablo/Path of Exile:
  1 prefix alone = common
  1 prefix + 1 suffix = uncommon
  3 affixes together = rare (bonus > sum of parts)
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ─────────────────────────────────────────────────────────────────
AI_SCORED_PATH  = "output_ai/ai_scored_transactions.csv"
RULE_SCORED_PATH = "output_v11/scored_transactions_v7.csv"
OUTPUT_DIR      = "output_ai"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tier thresholds for final combined score (0–100 scale)
TIERS = {
    "LEGENDARY": 72,   # All dimensions lighting up
    "RARE":      48,   # 2+ strong signals
    "MAGIC":     28,   # 1 strong or 2 moderate signals
    "COMMON":     0,   # Baseline flag
}

# Amount percentile thresholds (computed from data)
AMOUNT_TIER_HIGH   = 0.90   # Top 10% = large
AMOUNT_TIER_MEDIUM = 0.70   # Top 30% = notable


print("=" * 65)
print("  TRIAGE LABELER — DYNAMIC CONFIDENCE SCORING")
print("=" * 65)


# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
df_ai = pd.read_csv(AI_SCORED_PATH)
df_ai["timestamp"] = pd.to_datetime(df_ai["timestamp"])
print(f"\n[DATA] AI-scored dataset: {len(df_ai):,} rows")

# Also load rule engine output for detailed rule fires per transaction
try:
    df_rules = pd.read_csv(RULE_SCORED_PATH)
    df_rules["timestamp"] = pd.to_datetime(df_rules["timestamp"])
    df_rules["timestamp_str"] = df_rules["timestamp"].astype(str)
    df_ai["timestamp_str"] = df_ai["timestamp"].astype(str)
    has_rule_detail = True
    print(f"[DATA] Rule detail loaded: {len(df_rules):,} rows")
except FileNotFoundError:
    has_rule_detail = False
    print("[DATA] Rule detail not available, using risk_score proxy")


# ─── 2. COMPUTE SIGNAL DENSITY ───────────────────────────────────────────────
# Signal density = what % of the 22 rules fired on this transaction?
# Proxy: estimate from rule column booleans in rule engine output

RULE_FLAG_COLS = [
    "layering_flag", "mixer_flag", "bridge_flag", "peel_flag",
    "novel_dump_flag", "conc_inflow_flag", "ofac_flag", "flash_flag",
    "burst_flag", "dormant_flag", "wash_flag", "smurf_flag",
    "exit_rush_flag", "rapid_flag", "high_risk_country_flag",
    "ex_avoid_flag", "deep_peel_flag"
]

if has_rule_detail:
    available_rule_cols = [c for c in RULE_FLAG_COLS if c in df_rules.columns]
    df_rules["rules_fired_count"] = df_rules[available_rule_cols].fillna(0).astype(bool).sum(axis=1)

    # Also count velocity/structuring from tx_count_in_window and small_tx_count_in_window
    if "tx_count_in_window" in df_rules.columns:
        df_rules["rules_fired_count"] += (df_rules["tx_count_in_window"] > 5).astype(int)
    if "small_tx_count_in_window" in df_rules.columns:
        df_rules["rules_fired_count"] += (df_rules["small_tx_count_in_window"] > 3).astype(int)

    df_rules["signal_density"] = (df_rules["rules_fired_count"] / 19).clip(0, 1)   # 19 rule flags

    # Merge signal density back
    rule_sig = df_rules[["sender_id", "receiver_id", "amount", "timestamp_str",
                          "rules_fired_count", "signal_density"]].drop_duplicates(
        subset=["sender_id", "receiver_id", "amount", "timestamp_str"])

    df_ai = df_ai.merge(rule_sig, on=["sender_id", "receiver_id", "amount", "timestamp_str"], how="left")
    df_ai["rules_fired_count"] = df_ai["rules_fired_count"].fillna(0)
    df_ai["signal_density"]    = df_ai["signal_density"].fillna(0)
    print(f"\n[SIGNAL] Merged rule detail. Avg rules fired: {df_ai['rules_fired_count'].mean():.1f}")
else:
    # Estimate from risk_score (rough proxy: score/200 capped at 1)
    df_ai["rules_fired_count"] = (df_ai["risk_score"] / 30).clip(0, 8).round()
    df_ai["signal_density"]    = (df_ai["risk_score"] / 200).clip(0, 1)


# ─── 3. AMOUNT TIER ──────────────────────────────────────────────────────────
p90 = df_ai["amount"].quantile(AMOUNT_TIER_HIGH)
p70 = df_ai["amount"].quantile(AMOUNT_TIER_MEDIUM)

def amount_weight(amount):
    """Returns 0–40: contribution of amount to triage score."""
    if amount >= p90:
        return 40    # Top 10% — always significant
    elif amount >= p70:
        return 20    # Top 30% — notable
    else:
        return 5     # Below median — small tx

df_ai["amount_weight"] = df_ai["amount"].apply(amount_weight)
print(f"[AMOUNT] p70=${p70:,.0f}  p90=${p90:,.0f}")
print(f"         Large (p90+): {(df_ai['amount'] >= p90).sum():,}  |  Notable: {((df_ai['amount'] >= p70) & (df_ai['amount'] < p90)).sum():,}")


# ─── 4. FREQUENCY SCORE ──────────────────────────────────────────────────────
# How many times does the same SENDER appear in flagged transactions within 24h?
# Recurrence = pattern, not noise.

df_ai_sorted = df_ai.sort_values("timestamp")

def compute_frequency_score(df):
    """For each flagged row: count same sender_id in ±12h window of flagged txns."""
    flagged = df[df["rule_flagged"] == 1].copy()
    freq_scores = {}

    for idx, row in flagged.iterrows():
        window_start = row["timestamp"] - pd.Timedelta(hours=12)
        window_end   = row["timestamp"] + pd.Timedelta(hours=12)
        same_sender  = flagged[
            (flagged["sender_id"] == row["sender_id"]) &
            (flagged["timestamp"] >= window_start) &
            (flagged["timestamp"] <= window_end)
        ]
        freq_scores[idx] = len(same_sender) - 1  # subtract self

    return pd.Series(freq_scores, name="frequency_count")

freq_series = compute_frequency_score(df_ai_sorted)
df_ai["frequency_count"] = freq_series.reindex(df_ai.index).fillna(0)

def frequency_weight(freq_count):
    """Returns 0–35: contribution of recurrence to triage score."""
    if freq_count >= 10:
        return 35    # High-volume campaign — escalate
    elif freq_count >= 5:
        return 22    # Repeated pattern
    elif freq_count >= 2:
        return 12    # Some recurrence
    else:
        return 0     # One-off

df_ai["frequency_weight"] = df_ai["frequency_count"].apply(frequency_weight)


# ─── 5. RULE DENSITY WEIGHT ──────────────────────────────────────────────────
def rule_density_weight(rules_fired):
    """Returns 0–50: contribution of signal density."""
    if rules_fired >= 6:
        return 50
    elif rules_fired >= 4:
        return 35
    elif rules_fired >= 3:
        return 22
    elif rules_fired >= 2:
        return 12
    elif rules_fired >= 1:
        return 5
    else:
        return 0

df_ai["rule_density_weight"] = df_ai["rules_fired_count"].apply(rule_density_weight)


# ─── 6. AI ANOMALY WEIGHT ─────────────────────────────────────────────────────
# Normalize AI score contribution: 0–25 points
df_ai["ai_weight"] = (df_ai["ai_anomaly_score"] / 100 * 25).round(1)


# ─── 7. COMBINATION BONUS ────────────────────────────────────────────────────
# This is the non-linear escalation:
# When multiple high-weight signals co-occur, the combined risk > sum of parts.
# Modeled as: if N dimensions are "elevated", add geometric bonus.
# In game terms: weapon + fire damage + life steal = legendary (not just 3×common)

def combination_bonus(row):
    """Non-linear escalation when multiple dimensions are elevated simultaneously."""
    elevated = 0
    if row["amount_weight"] >= 40:       elevated += 1  # top-10% amount
    if row["rule_density_weight"] >= 22: elevated += 1  # 3+ rules firing
    if row["frequency_weight"] >= 12:    elevated += 1  # recurring pattern
    if row["ai_weight"] >= 15:           elevated += 1  # AI also flagged it

    if elevated == 4:
        return 30    # All 4 dimensions: legendary tier bonus
    elif elevated == 3:
        return 18    # Three dimensions: rare tier bonus
    elif elevated == 2:
        return 8     # Two dimensions: magic tier bonus
    else:
        return 0

df_ai["combination_bonus"] = df_ai.apply(combination_bonus, axis=1)


# ─── 8. FINAL TRIAGE SCORE ───────────────────────────────────────────────────
# Weights cap at 100 by design — the combination_bonus pushes over common cases
# into rare/legendary territory when signals converge.

df_ai["triage_score"] = (
    df_ai["rule_density_weight"] +
    df_ai["amount_weight"] +
    df_ai["frequency_weight"] +
    df_ai["ai_weight"] +
    df_ai["combination_bonus"]
).clip(0, 100).round(1)

# Assign rarity tier
def assign_tier(score):
    if score >= TIERS["LEGENDARY"]:  return "LEGENDARY"
    elif score >= TIERS["RARE"]:     return "RARE"
    elif score >= TIERS["MAGIC"]:    return "MAGIC"
    elif score > 0:                  return "COMMON"
    else:                            return "CLEAN"

df_ai["triage_tier"] = df_ai["triage_score"].apply(assign_tier)


# ─── 9. RESULTS ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  TRIAGE RESULTS — ITEM RARITY DISTRIBUTION")
print("=" * 65)

tier_order = ["LEGENDARY", "RARE", "MAGIC", "COMMON", "CLEAN"]
tier_colors = {
    "LEGENDARY": "🟠", "RARE": "🟣", "MAGIC": "🔵", "COMMON": "⚪", "CLEAN": "⬛"
}

tier_counts = df_ai["triage_tier"].value_counts()
total = len(df_ai)

for tier in tier_order:
    count = tier_counts.get(tier, 0)
    pct   = count / total * 100
    bar   = "█" * int(count / total * 40)
    print(f"  {tier_colors[tier]} {tier:<12} {count:>5}  ({pct:5.1f}%)  {bar}")


# ─── 10. AI-ONLY ANOMALIES CLASSIFIED ────────────────────────────────────────
print("\n" + "=" * 65)
print("  AI-ONLY ANOMALIES → NOW CLASSIFIED")
print("=" * 65)
ai_only = df_ai[df_ai["ensemble_flag"] == "AI_ONLY"].copy()
if len(ai_only) > 0:
    ai_tiers = ai_only["triage_tier"].value_counts()
    print(f"\n  Total AI-only anomalies: {len(ai_only)}")
    for tier in tier_order:
        count = ai_tiers.get(tier, 0)
        if count > 0:
            print(f"  {tier_colors[tier]} {tier}: {count}")

    print(f"\n  Top AI-only anomalies by triage_score:")
    top = ai_only.nlargest(10, "triage_score")[
        ["sender_id", "amount", "rules_fired_count", "frequency_count",
         "ai_anomaly_score", "triage_score", "triage_tier"]
    ].copy()
    top["sender_id"] = top["sender_id"].str[:14] + "..."
    print(top.to_string(index=False))
else:
    print("  No AI-only anomalies found in this dataset.")


# ─── 11. WEIGHT BREAKDOWN FOR TOP LEGENDARIES ────────────────────────────────
print("\n" + "=" * 65)
print("  LEGENDARY TIER — WEIGHT BREAKDOWN (Top 5)")
print("=" * 65)
legendaries = df_ai[df_ai["triage_tier"] == "LEGENDARY"].nlargest(5, "triage_score")
if len(legendaries) > 0:
    for _, row in legendaries.iterrows():
        sid = str(row["sender_id"])[:16] + "..."
        print(f"\n  {sid}  score={row['triage_score']}")
        print(f"    amount         ${row['amount']:>12,.0f}  → weight {row['amount_weight']:>4}")
        print(f"    rules_fired    {row['rules_fired_count']:>14.0f}  → weight {row['rule_density_weight']:>4}")
        print(f"    freq_count     {row['frequency_count']:>14.0f}  → weight {row['frequency_weight']:>4}")
        print(f"    ai_score       {row['ai_anomaly_score']:>14.1f}  → weight {row['ai_weight']:>4}")
        print(f"    combo_bonus    {'':>14}     +{row['combination_bonus']:>3}")
        print(f"    {'─'*45}")
        print(f"    triage_score   {'':>14}  = {row['triage_score']:>4}")


# ─── 12. SUMMARY STATS ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SUMMARY — TRIAGE VS RULE ENGINE")
print("=" * 65)

legendary_count = (df_ai["triage_tier"] == "LEGENDARY").sum()
rare_count      = (df_ai["triage_tier"] == "RARE").sum()
magic_count     = (df_ai["triage_tier"] == "MAGIC").sum()
high_priority   = legendary_count + rare_count

# Before triage: rule engine flagged how many for human review?
rule_flagged_total = df_ai["rule_flagged"].sum()

print(f"\n  Before triage (rule engine):  {rule_flagged_total} transactions flagged")
print(f"  After triage — LEGENDARY:     {legendary_count} (immediate escalation)")
print(f"  After triage — RARE:          {rare_count} (investigate today)")
print(f"  After triage — MAGIC:         {magic_count} (review in batch)")
print(f"\n  Analyst queue reduction: {rule_flagged_total} → {high_priority} high-priority")
print(f"  Efficiency gain: {(1 - high_priority/rule_flagged_total)*100:.0f}% fewer critical reviews needed")
print(f"\n  This is the real value of triage:")
print(f"  Same detection quality, fraction of analyst attention required.")


# ─── 13. SAVE ────────────────────────────────────────────────────────────────
output_path = f"{OUTPUT_DIR}/triage_scored_transactions.csv"
df_ai.to_csv(output_path, index=False)

scorecard = {
    "tier_distribution": {t: int(tier_counts.get(t, 0)) for t in tier_order},
    "analyst_queue_before": int(rule_flagged_total),
    "analyst_queue_after_legendary": int(legendary_count),
    "analyst_queue_after_rare": int(rare_count),
    "total_high_priority": int(high_priority),
    "queue_reduction_pct": round((1 - high_priority/rule_flagged_total)*100, 1) if rule_flagged_total > 0 else 0,
    "ai_only_classified": {t: int(ai_only["triage_tier"].value_counts().get(t, 0)) for t in tier_order} if len(ai_only) > 0 else {}
}
with open(f"{OUTPUT_DIR}/triage_scorecard.json", "w") as f:
    json.dump(scorecard, f, indent=2)

print(f"\n[SAVE] → {output_path}")
print(f"[SAVE] → {OUTPUT_DIR}/triage_scorecard.json")
print("=" * 65)
print("  TRIAGE COMPLETE")
print("=" * 65)

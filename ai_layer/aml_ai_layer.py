"""
AML AI Layer — v1
─────────────────────────────────────────────────────────────────────────────
Layer 2 of the detection stack: Graph features + Isolation Forest anomaly
detection on top of the existing rule engine output.

Architecture:
  Layer 1 ─ Rule Engine v11       → known patterns, high precision
  Layer 2 ─ Graph Features        → wallet relationship topology
  Layer 3 ─ Isolation Forest      → statistical outlier (zero-day immune)
  Layer 4 ─ Ensemble Decision     → union with triage priority

Why Isolation Forest first (not GNN)?
  GNN needs ~10k+ labelled graph-structured examples and a training loop.
  Isolation Forest is unsupervised — it learns what "normal" looks like from
  the data itself, then flags deviations. No labels needed. Exactly what you
  have in real-world AML: a sea of normal txns with rare anomalies buried inside.

The Feynman test: imagine sorting apples. Normal apples sit in clusters.
  Weird apples (wrong size, wrong colour, wrong weight) are easy to isolate —
  you just need fewer cuts to separate them from the pile. Isolation Forest
  does exactly that: it measures how few random cuts it takes to isolate a point.
  The fewer cuts needed → the more anomalous the point.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import json
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ─────────────────────────────────────────────────────────────────
DATA_PATH       = "data/transactions_v10.csv"
RULE_SCORES_PATH = "output_v11/scored_transactions_v7.csv"   # v11 engine output
OUTPUT_DIR      = "output_ai"
CONTAMINATION   = 0.18   # expected anomaly fraction (~fraud ratio in dataset)

# Known ground-truth labels from case taxonomy
FRAUD_CASES = {
    "lazarus_stake_hack", "ronin_exploiter", "tornado_0.1eth",
    "tornado_100eth", "tornado_10eth", "tornado_gov",
    "wormhole_exploiter", "nomad_crowd_exploiter",
    "euler_attacker", "bybit_drainer_impl"
}
CONTROL_CASES = {"eth_foundation_ctrl", "vitalik_eth_control"}

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
print("=" * 65)
print("  AML AI LAYER — GRAPH + ISOLATION FOREST DETECTION")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"\n[DATA] Loaded {len(df):,} transactions | {df['label'].value_counts().to_dict()}")


# ─── 2. GRAPH FEATURE ENGINEERING ───────────────────────────────────────────
# This is the "Layer 2 / GNN proxy" — we compute structural graph features
# without PyTorch Geometric. Full GNN would learn these + higher-order patterns.
print("\n[GRAPH] Engineering network topology features...")

# Out-degree: how many unique wallets does this sender fan out to?
out_degree = df.groupby("sender_id")["receiver_id"].nunique().rename("out_degree")

# In-degree: how many unique senders funnel INTO this receiver?
in_degree = df.groupby("receiver_id")["sender_id"].nunique().rename("in_degree")

# Sender velocity: transactions per hour within a 6-hour sliding window proxy
sender_tx_count = df.groupby("sender_id")["amount"].count().rename("sender_node_tx_count")

# Amount concentration: sender sends >80% of value to single receiver?
def concentration_ratio(group):
    total = group["amount"].sum()
    if total == 0:
        return 0.0
    max_to_one_receiver = group.groupby("receiver_id")["amount"].sum().max()
    return max_to_one_receiver / total

conc = df.groupby("sender_id").apply(concentration_ratio).rename("amount_concentration")

# Receiver re-sends: does a receiver immediately become a sender?
all_senders = set(df["sender_id"].unique())
all_receivers = set(df["receiver_id"].unique())
pass_through_nodes = all_senders & all_receivers  # wallets that both receive AND send

df["sender_is_passthrough"] = df["sender_id"].isin(pass_through_nodes).astype(int)
df["receiver_is_passthrough"] = df["receiver_id"].isin(pass_through_nodes).astype(int)

# Time clustering: are transactions from this sender bunched in short windows?
def time_burstiness(group):
    if len(group) < 2:
        return 0.0
    sorted_ts = group["timestamp"].sort_values()
    gaps = sorted_ts.diff().dt.total_seconds().dropna()
    if gaps.mean() == 0:
        return 0.0
    # Coefficient of variation of inter-arrival times — high CV = bursty
    return gaps.std() / (gaps.mean() + 1e-9)

burstiness = df.groupby("sender_id").apply(time_burstiness).rename("time_burstiness")

# Cross-border flag
HIGH_RISK_COUNTRIES = {"IR", "KP", "SY", "CU", "RU", "BY", "MM", "MX_DARK"}
df["is_high_risk_country"] = df["country"].isin(HIGH_RISK_COUNTRIES).astype(int)

# Amount relative to sender's own history (z-score per sender)
df["sender_amount_zscore"] = df.groupby("sender_id")["amount"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-9)
)

# Attach graph features back to df
df = df.join(out_degree, on="sender_id")
df = df.join(in_degree, on="receiver_id")
df = df.join(sender_tx_count, on="sender_id")
df = df.join(conc, on="sender_id")
df = df.join(burstiness, on="sender_id")

# Fill NaN for wallets appearing only once
df["out_degree"]          = df["out_degree"].fillna(1)
df["in_degree"]           = df["in_degree"].fillna(1)
df["sender_node_tx_count"] = df["sender_node_tx_count"].fillna(1)
df["amount_concentration"] = df["amount_concentration"].fillna(1.0)
df["time_burstiness"]     = df["time_burstiness"].fillna(0.0)

print(f"    → Computed 9 graph/statistical features per transaction")


# ─── 3. LOAD RULE ENGINE OUTPUT (before feature matrix — keeps rows aligned) ──
print("\n[RULES] Loading rule engine v11 scores...")
try:
    rule_df = pd.read_csv(RULE_SCORES_PATH)
    rule_df["timestamp"] = pd.to_datetime(rule_df["timestamp"])
    rule_df["timestamp_str"] = rule_df["timestamp"].astype(str)
    df["timestamp_str"] = df["timestamp"].astype(str)

    if "risk_score" not in rule_df.columns:
        raise KeyError("risk_score column missing from rule output")

    rule_sub = rule_df[["sender_id", "receiver_id", "amount", "timestamp_str", "risk_score"]].copy()
    rule_sub = rule_sub.drop_duplicates(subset=["sender_id", "receiver_id", "amount", "timestamp_str"])
    df = df.merge(rule_sub, on=["sender_id", "receiver_id", "amount", "timestamp_str"], how="left")
    df = df.drop_duplicates(subset=["sender_id", "receiver_id", "amount", "timestamp_str"]).reset_index(drop=True)
    df["risk_score"]   = df["risk_score"].fillna(0)
    df["rule_flagged"] = (df["risk_score"] >= 50).astype(int)
    print(f"    → Rows after merge: {len(df)} | Rule-flagged: {df['rule_flagged'].sum():,}")
except Exception as e:
    print(f"    ⚠ Rule output error: {e}. Continuing without rule scores.")
    df["risk_score"]   = 0
    df["rule_flagged"] = 0


# ─── 4. BUILD FEATURE MATRIX ─────────────────────────────────────────────────
FEATURES = [
    "amount",
    "out_degree",
    "in_degree",
    "sender_node_tx_count",
    "amount_concentration",
    "time_burstiness",
    "sender_is_passthrough",
    "receiver_is_passthrough",
    "is_high_risk_country",
    "sender_amount_zscore",
    "sender_tx_count",      # from profiling
    "sender_avg_amount",    # from profiling
    "sender_active_days",   # from profiling
]

X = df[FEATURES].fillna(0).astype(float)

# Log-scale amount (heavy-tailed distribution — log makes it learnable)
X = X.copy()
X["amount"]            = np.log1p(X["amount"])
X["sender_avg_amount"] = np.log1p(X["sender_avg_amount"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n[FEATURES] Matrix shape: {X_scaled.shape}  ({len(FEATURES)} features × {len(df)} rows)")


# ─── 5. ISOLATION FOREST ─────────────────────────────────────────────────────
print("\n[MODEL] Training Isolation Forest...")
print(f"    contamination={CONTAMINATION} | n_estimators=200 | random_state=42")

iso = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=42,
    n_jobs=-1
)
iso.fit(X_scaled)

# -1 = anomaly, 1 = normal → convert to binary
raw_preds = iso.predict(X_scaled)
df["ai_anomaly"]    = (raw_preds == -1).astype(int)
df["ai_score_raw"]  = iso.score_samples(X_scaled)   # more negative = more anomalous

# Normalize anomaly score to 0–100 for readability
ai_min, ai_max = df["ai_score_raw"].min(), df["ai_score_raw"].max()
df["ai_anomaly_score"] = 100 * (1 - (df["ai_score_raw"] - ai_min) / (ai_max - ai_min + 1e-9))

flagged = df["ai_anomaly"].sum()
print(f"    → Flagged {flagged:,} transactions as anomalous ({flagged/len(df)*100:.1f}%)")


# ─── 6. ENSEMBLE DECISION ────────────────────────────────────────────────────
# Three tiers of evidence:
#   CONFIRMED  → rules AND AI both flag (highest confidence)
#   AI_ONLY    → AI flags, rules missed (novel / data-gap cases)
#   RULES_ONLY → rules flag, AI missed (pattern-matched but statistically normal)
#   CLEAN      → neither flags

df["ensemble_flag"] = "CLEAN"
df.loc[(df["rule_flagged"] == 1) & (df["ai_anomaly"] == 1), "ensemble_flag"] = "CONFIRMED"
df.loc[(df["rule_flagged"] == 0) & (df["ai_anomaly"] == 1), "ensemble_flag"] = "AI_ONLY"
df.loc[(df["rule_flagged"] == 1) & (df["ai_anomaly"] == 0), "ensemble_flag"] = "RULES_ONLY"

# Ensemble score: weighted combination
df["ensemble_score"] = (0.6 * df["risk_score"] + 0.4 * df["ai_anomaly_score"]).round(1)

print("\n[ENSEMBLE] Decision breakdown:")
for tier, count in df["ensemble_flag"].value_counts().items():
    pct = count / len(df) * 100
    print(f"    {tier:<15} {count:>4}  ({pct:.1f}%)")


# ─── 7. GROUND-TRUTH EVALUATION ──────────────────────────────────────────────
print("\n" + "=" * 65)
print("  GROUND TRUTH EVALUATION")
print("=" * 65)

# Assign true labels based on case_id and sender_profile/label
# SYNTH_FORENSIC cases are all known fraud
# REAL_DATA cases: sender_profile != "BUSINESS_LIKE"+"clean" needed

# Build true fraud label
def is_true_fraud(row):
    if row.get("label") == "SYNTH_FORENSIC":
        return 1
    # REAL_DATA — check if from a fraud case wallet
    case = str(row.get("case_id", "")).lower()
    for fraud_case in FRAUD_CASES:
        if fraud_case in case:
            return 1
    return 0

df["true_fraud"] = df.apply(is_true_fraud, axis=1)

# If case_id merge didn't work well, use label as proxy
synth_count = (df["label"] == "SYNTH_FORENSIC").sum()
if df["true_fraud"].sum() < synth_count:
    # Fallback: use label directly
    df["true_fraud"] = (df["label"] == "SYNTH_FORENSIC").astype(int)
    print(f"    [note] Using label-based ground truth ({synth_count} SYNTH_FORENSIC = fraud)")

true_fraud_n    = df["true_fraud"].sum()
true_benign_n   = len(df) - true_fraud_n
print(f"\n    True fraud transactions:  {true_fraud_n:>4}")
print(f"    True benign transactions: {true_benign_n:>4}")

# Rule engine performance
rule_tp = ((df["rule_flagged"] == 1) & (df["true_fraud"] == 1)).sum()
rule_fp = ((df["rule_flagged"] == 1) & (df["true_fraud"] == 0)).sum()
rule_fn = ((df["rule_flagged"] == 0) & (df["true_fraud"] == 1)).sum()
rule_detection = rule_tp / true_fraud_n * 100 if true_fraud_n > 0 else 0
rule_fp_rate   = rule_fp / (rule_fp + ((df["rule_flagged"] == 0) & (df["true_fraud"] == 0)).sum()) * 100

# AI performance
ai_tp = ((df["ai_anomaly"] == 1) & (df["true_fraud"] == 1)).sum()
ai_fp = ((df["ai_anomaly"] == 1) & (df["true_fraud"] == 0)).sum()
ai_fn = ((df["ai_anomaly"] == 0) & (df["true_fraud"] == 1)).sum()
ai_detection = ai_tp / true_fraud_n * 100 if true_fraud_n > 0 else 0
ai_fp_rate   = ai_fp / (ai_fp + ((df["ai_anomaly"] == 0) & (df["true_fraud"] == 0)).sum()) * 100

# Ensemble performance
ens_flagged = (df["ensemble_flag"] != "CLEAN").astype(int)
ens_tp = ((ens_flagged == 1) & (df["true_fraud"] == 1)).sum()
ens_fp = ((ens_flagged == 1) & (df["true_fraud"] == 0)).sum()
ens_detection = ens_tp / true_fraud_n * 100 if true_fraud_n > 0 else 0
ens_fp_rate   = ens_fp / (ens_fp + ((ens_flagged == 0) & (df["true_fraud"] == 0)).sum()) * 100

print(f"\n{'Layer':<20} {'Detection':>10} {'FP Rate':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
print("-" * 65)
print(f"{'Rule Engine v11':<20} {rule_detection:>9.1f}% {rule_fp_rate:>9.1f}% {rule_tp:>6} {rule_fp:>6} {rule_fn:>6}")
print(f"{'AI (Iso Forest)':<20} {ai_detection:>9.1f}% {ai_fp_rate:>9.1f}% {ai_tp:>6} {ai_fp:>6} {ai_fn:>6}")
print(f"{'Ensemble (Union)':<20} {ens_detection:>9.1f}% {ens_fp_rate:>9.1f}% {ens_tp:>6} {ens_fp:>6} {(true_fraud_n - ens_tp):>6}")
print("-" * 65)


# ─── 8. WHAT AI CATCHES THAT RULES MISS ─────────────────────────────────────
print("\n[INSIGHT] AI-ONLY catches (fraud rules missed):")
ai_only_fraud = df[(df["ensemble_flag"] == "AI_ONLY") & (df["true_fraud"] == 1)]
if len(ai_only_fraud) > 0:
    print(f"    {len(ai_only_fraud)} fraud transactions caught by AI alone")
    print(f"    Top anomaly scores: {sorted(ai_only_fraud['ai_anomaly_score'].tolist(), reverse=True)[:10]}")
    print(f"    Avg amount: ${ai_only_fraud['amount'].mean():,.0f}")
    print(f"    Sample sender_profiles: {ai_only_fraud['sender_profile'].value_counts().head(3).to_dict()}")
else:
    print("    None — (AI missed same cases as rules)")

print("\n[INSIGHT] RULES-ONLY catches (AI missed):")
rules_only_fraud = df[(df["ensemble_flag"] == "RULES_ONLY") & (df["true_fraud"] == 1)]
if len(rules_only_fraud) > 0:
    print(f"    {len(rules_only_fraud)} fraud transactions caught by rules alone")
    print(f"    Sample sender_profiles: {rules_only_fraud['sender_profile'].value_counts().head(3).to_dict()}")

print("\n[INSIGHT] Still missed by BOTH (data ceiling):")
both_miss_fraud = df[(df["ensemble_flag"] == "CLEAN") & (df["true_fraud"] == 1)]
print(f"    {len(both_miss_fraud)} transactions missed by both engine and AI")
if len(both_miss_fraud) > 0:
    print(f"    Avg amount: ${both_miss_fraud['amount'].mean():,.0f}")
    print(f"    → These require: block-level timestamps / cross-chain oracles / real OFAC matching")


# ─── 9. FEATURE IMPORTANCE (Isolation Forest proxy) ─────────────────────────
print("\n[FEATURES] Feature contribution to anomaly detection:")
# Compute correlation between each feature and the anomaly score (use df directly)
feature_corr = {}
ai_scores = df["ai_anomaly_score"].values
for feat in FEATURES:
    if feat in df.columns:
        vals = df[feat].fillna(0).values
        if len(vals) == len(ai_scores):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr_mat = np.corrcoef(vals.astype(float), ai_scores.astype(float))
                corr = corr_mat[0, 1] if not np.isnan(corr_mat[0, 1]) else 0.0
            feature_corr[feat] = abs(corr)

sorted_features = sorted(feature_corr.items(), key=lambda x: x[1], reverse=True)
for feat, corr in sorted_features:
    bar = "█" * int(corr * 30)
    print(f"    {feat:<28} {corr:.3f}  {bar}")


# ─── 10. SAVE OUTPUTS ────────────────────────────────────────────────────────
output_csv = f"{OUTPUT_DIR}/ai_scored_transactions.csv"
df.to_csv(output_csv, index=False)
print(f"\n[SAVE] Full scored dataset → {output_csv}")

# Save scorecard
scorecard = {
    "rule_engine_v11": {
        "detection_pct": round(rule_detection, 1),
        "fp_rate_pct":   round(rule_fp_rate, 1),
        "tp": int(rule_tp), "fp": int(rule_fp), "fn": int(rule_fn)
    },
    "ai_isolation_forest": {
        "detection_pct": round(ai_detection, 1),
        "fp_rate_pct":   round(ai_fp_rate, 1),
        "tp": int(ai_tp), "fp": int(ai_fp), "fn": int(ai_fn)
    },
    "ensemble_union": {
        "detection_pct": round(ens_detection, 1),
        "fp_rate_pct":   round(ens_fp_rate, 1),
        "tp": int(ens_tp), "fp": int(ens_fp)
    },
    "ensemble_breakdown": df["ensemble_flag"].value_counts().to_dict(),
    "feature_importance": {k: round(v, 3) for k, v in sorted_features},
    "ai_only_fraud_caught": int(len(ai_only_fraud)),
    "data_ceiling_misses": int(len(both_miss_fraud))
}

scorecard_path = f"{OUTPUT_DIR}/ai_scorecard.json"
with open(scorecard_path, "w") as f:
    json.dump(scorecard, f, indent=2)
print(f"[SAVE] Scorecard → {scorecard_path}")


# ─── 11. GNN ARCHITECTURE SKETCH ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  GNN ARCHITECTURE (Layer 3 — requires PyTorch Geometric)")
print("=" * 65)
print("""
  Graph construction:
    Nodes = unique wallet addresses
    Edges = transactions (directed, weighted by amount)
    Node features = [out_degree, in_degree, total_volume, tx_count,
                     avg_gap_seconds, is_bridge, is_exchange, country_risk]
    Edge features = [amount, log_amount, time_gap_to_prev_tx, is_cross_chain]

  GraphSAGE layer stack:
    Input → GraphSAGE(64) → ReLU → Dropout(0.3)
          → GraphSAGE(32) → ReLU → Dropout(0.3)
          → Linear(16)    → ReLU
          → Linear(1)     → Sigmoid  → anomaly_probability

  Why GraphSAGE over GCN:
    GCN aggregates ALL neighbors equally — bad for hub wallets (exchanges)
    that touch thousands of legit txns. GraphSAGE samples a fixed-size
    neighborhood → scalable, exchange-aware, attack-pattern-aware.

  What GNN catches that Isolation Forest misses:
    → Multi-hop money laundering (structurally identical to known attack
      graphs even with different wallet addresses)
    → New bridge exploits following Wormhole/Nomad topology
    → Ronin-style intermediary chains (non-OFAC but structurally identical)

  Training data needed:
    ~5,000 labelled graph snapshots minimum
    Source: Chainalysis datasets / Elliptic Bitcoin dataset (public)
          / Dune Analytics with wallet labels
""")

print("=" * 65)
print("  RUN COMPLETE")
print("=" * 65)

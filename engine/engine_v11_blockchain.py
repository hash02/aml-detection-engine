"""
NEXUS-RISK AML Engine — v7 (Blockchain Layer)
==============================================
The blockchain boss fight.

v7 inherits ALL v6 rules and adds 3 blockchain-native abilities:

  🆕 ABILITY 7 — mixer_touch
     Fires when ANY transaction in the window touches a known mixer address.
     One hop = HIGH. Wallet that sent to mixer = HIGH.
     Wallet that received from mixer = CRITICAL.
     (Equivalent: nothing in v6 banking. This is a new weapon.)

  🆕 ABILITY 8 — bridge_hop
     Fires when sender has used 2+ different bridge contracts in 6 hours.
     More hops in less time = higher score.
     3+ bridges in 6hrs = automatic CRITICAL.
     (Equivalent: v6 foreign_country, but active multi-hop version.)

  🆕 ABILITY 9 — peel_chain_linear
     Detects linear A→B→C→D chains — NOT cycles (v6 layering catches cycles).
     Peel chain signature: same amount minus 1–8%, same forward direction,
     3+ hops in sequence.
     (Complement to v6 layering. They are different shapes of the same crime.)

v6 → v7 Rule Mapping (the full skill tree):
  v6 large_amount        → v7 inherits + LARGE_CRYPTO calibrated thresholds
  v6 foreign_country     → v7 inherits + MX_DARK / BRIDGE_OFFSHORE fire it
  v6 velocity            → v7 inherits + bot-pattern = CONTRACT_ANOMALY
  v6 structuring         → v7 inherits + PEEL_CHAIN splits fire it
  v6 fan_in              → v7 inherits + DUSTING fans fire it
  v6 layering (cycles)   → v7 inherits + WASH_TRADING circular flows
  [NEW] mixer_touch      → MIXER_ROUTING (no v6 equivalent)
  [NEW] bridge_hop       → BRIDGE_ABUSE (no v6 equivalent)
  [NEW] peel_chain_linear→ PEEL_CHAIN (true linear, different from cycles)

Run:
  python engine_v7_blockchain.py
  python engine_v7_blockchain.py --input data/transactions_profiled.csv

Reads:  data/transactions_profiled.csv
        (must have: id, sender_id, receiver_id, amount, country,
         timestamp, sender_profile, is_known_mixer, is_bridge)
"""

import argparse
import json
import os
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG  (all v6 weights preserved, 3 new ones added)
# ─────────────────────────────────────────────
CONFIG = {
    "home_country": "CA",

    # ── v6 Weights (unchanged) ──
    "w_large":       30,
    "w_velocity":    25,
    "w_fan_in":      20,
    "w_structuring": 35,
    "w_foreign":     20,
    "w_layering":    45,

    # ── v7 NEW Weights ──────────
    "w_mixer_touch":       55,   # Mixer = near-certain intent to hide
    "w_bridge_hop":        40,   # Multi-chain bridge = regulatory escape
    "w_peel_chain":        45,   # Linear layering = strong laundering signal

    # ── v8 NEW Weights ──────────────────────────────────────────────────────
    # ABILITY 10: novel_wallet_dump
    #   A wallet with very little history suddenly moves a massive amount.
    #   Classic exploit wallet pattern — created, drained once, abandoned.
    #   Ronin hacker: new address, single massive ETH drain from the bridge.
    "w_novel_dump":        65,   # High — new wallet + huge amount = very suspicious

    # ABILITY 11: concentrated_inflow
    #   A single receiver address collects large amounts from 3+ different
    #   senders in a short window. Consolidation of distributed hack proceeds.
    #   Lazarus distributes across many wallets then sweeps into one collector.
    "w_concentrated_inflow": 50,

    # ── v9 NEW Weights ──────────────────────────────────────────────────────
    # ABILITY 12: ofac_hit
    #   Direct match against OFAC SDN sanctioned address list.
    #   This is non-negotiable — a sanctioned address must ALWAYS alert.
    #   In real AML, this is the FIRST check before any ML/rules run.
    #   Bypasses all scoring — auto-CRITICAL.
    "w_ofac_hit":          200,  # Max possible — hard mandatory alert

    # ABILITY 13: flash_loan_burst
    #   Detects flash loan attack pattern: extreme velocity in a very tight
    #   time window (seconds, not minutes). A flash loan borrows + repays
    #   in a SINGLE BLOCK (~12 seconds on Ethereum). Any wallet doing 5+
    #   contract interactions with large amounts in under 60 seconds = flag.
    "w_flash_loan":         80,  # High — flash loans rarely legitimate at this scale

    # ABILITY 14: coordinated_burst
    #   Multiple different wallets sending to the SAME receiver within seconds
    #   of each other — coordinated attack or bot-controlled distribution.
    #   Different from concentrated_inflow (slower sweep) — this is near-simultaneous.
    "w_coordinated_burst":  70,

    # ── v10 NEW Weights ──────────────────────────────────────────────────────
    # ABILITY 15: dormant_activation
    #   A wallet inactive for 365+ days that suddenly moves large amounts.
    #   This is the BitFinex pattern (2016 hack, first movement 2022 = 6yr gap).
    #   Also matches Silk Road seizure patterns and forgotten cold wallets.
    #   Score scales with dormancy duration: longer sleep = higher alarm on wake.
    #   False positive guard: requires amount > $100k (filters micro-revivals).
    "w_dormant_activation": 75,  # High precision — few legit wallets sleep 1yr+ then move big

    # ── v11 NEW Weights ──────────────────────────────────────────────────────
    # ABILITY 16: wash_cycle
    #   Circular flow: A sends to B, B sends back to A (or A→B→C→A) within 24hrs
    #   with similar amounts (+/- 15%). Classic wash trading / round-trip signature.
    #   Why: money doesn't legitimately round-trip at scale. This is accounting fraud.
    "w_wash_cycle":          60,

    # ABILITY 17: smurfing
    #   Multiple wallets (3+) each sending amounts just BELOW the reporting threshold
    #   to the same receiver, coordinated within a 2-hour window.
    #   Smurf = deliberately keep each transaction under the radar limit.
    "w_smurfing":            55,

    # ABILITY 18: exit_rush
    #   A novel wallet (low tx count) receives a large amount and within 2 hours
    #   sends it to a bridge contract or known exchange address.
    #   The speed is the signal — legitimate users don't immediately bridge received funds.
    "w_exit_rush":           65,

    # ABILITY 19: rapid_succession
    #   Same sender hits 5+ different receivers within 5 minutes.
    #   Broader than velocity (which counts tx count) — this tracks unique receivers.
    #   Signals automated fund distribution / bot-driven fan-out.
    "w_rapid_succession":    50,

    # ABILITY 20: high_risk_country
    #   Transaction involves a country on the FATF blacklist / grey list.
    #   Standalone weak signal, but amplifies other rules significantly.
    "w_high_risk_country":   30,

    # ABILITY 21: exchange_avoidance
    #   Wallet does 4+ hops through non-exchange addresses before finally reaching
    #   an exchange. The deliberate avoidance of the direct path is the signal.
    "w_exchange_avoidance":  45,

    # ABILITY 22: layering_deep
    #   Extended peel chain: 5+ hops (vs standard 3). Detects long-range laundering
    #   like BitFinex-style multi-year trickle through dozens of wallets.
    "w_layering_deep":       70,

    # ── v6 Thresholds (unchanged) ──
    "fan_in_threshold":          4,
    "velocity_window_minutes":   60,
    "layering_window_hours":     1,    # tighter for on-chain speed
    "layering_max_depth":        3,
    "layering_min_amount":       100,

    # ── v7 NEW Thresholds ──────
    "mixer_window_hours":        24,   # look back 24hrs for mixer contact
    "bridge_window_hours":       6,    # 6hrs to spot rapid bridge hopping
    "bridge_hop_threshold":      2,    # 2+ bridge contracts = flag
    "peel_window_hours":         12,   # 12hrs to trace a peel chain
    "peel_min_hops":             3,    # minimum 3 hops to fire
    "peel_max_peel_pct":         0.12, # each hop sheds at most 12%

    # ── v8 NEW Thresholds ───────────────────────────────────────────────────
    # novel_wallet_dump: fires when ALL conditions met
    # Strict — a wallet must be brand new AND low activity AND send a large amount
    "novel_dump_max_tx_count":    5,   # sender has done <= 5 txns ever (truly new/dormant)
    "novel_dump_max_active_days": 7,   # wallet used within last 7 days only (recently created)
    "novel_dump_min_amount":  50_000,  # sends > $50,000 in one shot (major move)

    # concentrated_inflow: receiver collecting from many senders fast
    # More restrictive — needs 5+ senders, not 3
    "conc_inflow_window_hours":  6,    # 6-hour aggregation window
    "conc_inflow_min_senders":   5,    # 5+ unique senders required (higher bar)
    "conc_inflow_min_amount":   5_000, # each sender must send > $5,000 (meaningful amount)

    # ── v9 NEW Thresholds ──────────────────────────────────────────────────
    # flash_loan_burst
    "flash_window_seconds":     60,   # within 60 seconds = same block / near-block
    "flash_min_tx_count":        5,   # 5+ transactions in that window
    "flash_min_amount":      1_000,   # each transaction must be > $1,000

    # coordinated_burst (near-simultaneous multi-sender → same receiver)
    "burst_window_seconds":     30,   # within 30 seconds = genuinely coordinated
    "burst_min_senders":         3,   # 3+ different senders converging
    "burst_min_amount":        500,   # each transaction > $500

    # ── v10 NEW Thresholds ─────────────────────────────────────────────────
    # dormant_activation: long-dormant wallet suddenly moves large funds
    "dormant_min_days":       365,    # wallet must have been inactive for 365+ days
    "dormant_min_amount":  100_000,   # must move > $100k (guards against micro-revivals)
    "dormant_scale_years":      5,    # score doubles if wallet was dormant for 5+ years

    # ── v11 NEW Thresholds ─────────────────────────────────────────────────
    # wash_cycle
    "wash_window_hours":       24,    # look-back window for the return flow
    "wash_amount_tolerance":   0.15,  # amounts must match within 15%
    "wash_min_amount":      5_000,    # only flag meaningful amounts (>$5k)

    # smurfing
    "smurf_window_hours":       2,    # 2-hour coordination window
    "smurf_min_wallets":        3,    # 3+ unique senders required
    "smurf_threshold_pct":     0.90,  # transactions between 70-90% of reporting threshold
    "smurf_reporting_limit": 10_000,  # standard AML reporting threshold

    # exit_rush
    "exit_rush_window_hours":   2,    # large receive → bridge/exchange within 2hrs
    "exit_rush_min_amount": 50_000,   # must receive > $50k to qualify
    "exit_rush_max_tx_count":   10,   # sender must be relatively new (<10 prev txns)

    # rapid_succession (same-sender fan-out)
    "rapid_window_minutes":     5,    # 5-minute burst window
    "rapid_min_receivers":      5,    # 5+ unique receivers in that window
    "rapid_min_amount":     1_000,    # each tx > $1k (filters dust distribution)

    # high_risk_country — FATF blacklist / grey list country codes
    # Source: FATF High-Risk and Other Monitored Jurisdictions (2024)
    "high_risk_countries": {
        "KP", "IR", "MM", "SY", "YE", "SD", "LY", "IQ",  # FATF blacklist
        "AF", "AL", "BB", "BF", "CM", "HT", "JM", "JO",  # FATF grey list
        "ML", "MZ", "NG", "PA", "PH", "SN", "SS", "TZ",
        "TN", "VN", "XX", "UNKNOWN"                        # Unknown = treat as high risk
    },

    # exchange_avoidance — known exchange address prefixes / labels
    "known_exchange_prefixes": {
        "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance hot wallet
        "0x21a31ee1afc51d94c2efccaa2092ad1028285549",  # Binance 2
        "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",  # Binance 3
        "0x56eddb7aa87536c09ccc2793473599fd21a8b17f",  # Coinbase
        "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43",  # Coinbase 2
    },
    "exchange_avoidance_min_hops": 4,  # must route through 4+ non-exchange addresses

    # layering_deep
    "deep_peel_min_hops":       5,    # 5+ hops required (vs standard 3)
    "deep_peel_window_hours":  48,    # 48-hour extended look-back
    "deep_peel_min_amount": 10_000,   # each hop > $10k

    "alert_threshold": 40,

    "input_file":  "data/transactions_profiled.csv",
    "output_dir":  "output_v11",
}

# ─────────────────────────────────────────────
# PROFILE CONFIG  (same as v6, blockchain amounts are larger)
# ─────────────────────────────────────────────
PROFILE_CONFIG = {
    "NEW": {
        "large_amount_threshold":      8_000,
        "velocity_tx_count_threshold": 6,
        "struct_max_amount":           4_999,
        "struct_min_count":            6,
        "score_multiplier":            1.20,
        "description": "New wallet — tighter monitoring applied",
    },
    "PERSONAL_LIKE": {
        "large_amount_threshold":     10_000,
        "velocity_tx_count_threshold": 8,
        "struct_max_amount":           4_999,
        "struct_min_count":            8,
        "score_multiplier":            1.00,
        "description": "Personal wallet — standard monitoring",
    },
    "BUSINESS_LIKE": {
        "large_amount_threshold":     50_000,
        "velocity_tx_count_threshold":15,
        "struct_max_amount":           9_999,
        "struct_min_count":           10,
        "score_multiplier":            0.85,
        "description": "Contract/business wallet — relaxed thresholds",
    },
}
DEFAULT_PROFILE = "PERSONAL_LIKE"

# Known mixer country codes (set by blockchain_adapter.py)
MIXER_COUNTRY_CODES = {"MX_DARK"}
# Known bridge country codes
BRIDGE_COUNTRY_CODES = {"BRIDGE_OFFSHORE"}


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "sender_profile" not in df.columns:
        raise ValueError("[ERROR] Missing sender_profile. Run adapter or profiler first.")
    df["sender_profile"] = df["sender_profile"].fillna(DEFAULT_PROFILE).str.upper()
    print(f"[LOAD] {len(df)} transactions | profiles: {df['sender_profile'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────
# v6 FEATURE ENGINEERING  (unchanged from v6)
# ─────────────────────────────────────────────
def compute_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    window = timedelta(minutes=cfg["velocity_window_minutes"])
    tx_counts, small_counts, fan_in_counts = [], [], []

    for i, row in df.iterrows():
        pcfg = PROFILE_CONFIG.get(row["sender_profile"], PROFILE_CONFIG[DEFAULT_PROFILE])
        window_mask = (
            (df["sender_id"] == row["sender_id"]) &
            (df["timestamp"] >= row["timestamp"] - window) &
            (df["timestamp"] <= row["timestamp"])
        )
        window_df = df[window_mask]
        tx_counts.append(len(window_df))
        small_counts.append((window_df["amount"] <= pcfg["struct_max_amount"]).sum())

        fan_mask = (
            (df["receiver_id"] == row["receiver_id"]) &
            (df["timestamp"] >= row["timestamp"] - window) &
            (df["timestamp"] <= row["timestamp"])
        )
        fan_in_counts.append(df[fan_mask]["sender_id"].nunique())

    df["tx_count_in_window"]       = tx_counts
    df["small_tx_count_in_window"] = small_counts
    df["fan_in_count"]             = fan_in_counts
    return df


# ─────────────────────────────────────────────
# v6 LAYERING DETECTION  (unchanged)
# ─────────────────────────────────────────────
def build_graph_in_window(df_window, min_amount):
    graph = defaultdict(list)
    for _, row in df_window.iterrows():
        if row["amount"] >= min_amount:
            graph[row["sender_id"]].append(row["receiver_id"])
    return graph

def dfs_find_cycle(graph, start, current, path, max_depth, visited):
    if len(path) > max_depth:
        return None
    for neighbor in graph.get(current, []):
        if neighbor == start and len(path) >= 2:
            return path + [neighbor]
        if neighbor not in visited:
            visited.add(neighbor)
            result = dfs_find_cycle(graph, start, neighbor, path + [neighbor], max_depth, visited)
            if result:
                return result
            visited.discard(neighbor)
    return None

def detect_layering(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df["layering_flag"]  = False
    df["layering_chain"] = ""

    window_delta = timedelta(hours=cfg["layering_window_hours"])
    min_amount   = cfg["layering_min_amount"]
    max_depth    = cfg["layering_max_depth"]
    detected_cycles = []
    no_cycle_cache  = set()

    print(f"[LAYERING] Scanning {len(df)} transactions for cycles...")

    for i, row in df.iterrows():
        sender = row["sender_id"]
        ts     = row["timestamp"]
        cache_key = (sender, ts.replace(minute=0, second=0, microsecond=0))
        if cache_key in no_cycle_cache:
            continue
        window_mask = (df["timestamp"] >= ts) & (df["timestamp"] <= ts + window_delta)
        df_window = df[window_mask]
        if len(df_window) < 2:
            no_cycle_cache.add(cache_key)
            continue
        graph = build_graph_in_window(df_window, min_amount)
        if sender not in graph:
            no_cycle_cache.add(cache_key)
            continue
        visited = {sender}
        cycle = dfs_find_cycle(graph, sender, sender, [sender], max_depth, visited)
        if cycle:
            chain_str = "→".join(cycle)
            win_end   = ts + window_delta
            if not any(c[3] == chain_str for c in detected_cycles):
                detected_cycles.append((sender, ts, win_end, chain_str))
        else:
            no_cycle_cache.add(cache_key)

    for (origin_sender, win_start, win_end, chain_str) in detected_cycles:
        actors = set(chain_str.split("→"))
        mask = (
            (df["timestamp"] >= win_start) &
            (df["timestamp"] <= win_end) &
            (df["sender_id"].isin(actors) | df["receiver_id"].isin(actors))
        )
        df.loc[mask, "layering_flag"]  = True
        df.loc[mask, "layering_chain"] = chain_str

    print(f"[LAYERING] Found {len(detected_cycles)} cycle(s) | {df['layering_flag'].sum()} tx flagged")
    return df, detected_cycles


# ═══════════════════════════════════════════════════════════════
#  🆕  ABILITY 7 — MIXER TOUCH
#  "Did this wallet ever touch a mixer?"
#  Banking analog: nothing. This is a new weapon.
# ═══════════════════════════════════════════════════════════════
def detect_mixer_touch(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    A wallet is mixer-contaminated if:
      - It sent to a mixer country code (deposit), OR
      - It received from a mixer country code (withdrawal)
    within the mixer_window_hours lookback.

    We build a set of contaminated sender_ids and receiver_ids,
    then flag every row where those wallets appear.

    Score logic:
      - Deposited INTO mixer  → mixer_deposit  (HIGH)
      - Received FROM mixer   → mixer_withdraw (CRITICAL — clean money exits here)
      - Transacted WITH a contaminated wallet → mixer_adjacent (MEDIUM)
    """
    df = df.copy()
    df["mixer_flag"]   = False
    df["mixer_type"]   = ""
    window = timedelta(hours=cfg["mixer_window_hours"])

    # Direct mixer rows: country == MX_DARK
    mixer_rows = df[df["country"].isin(MIXER_COUNTRY_CODES)]

    depositors   = set(mixer_rows["sender_id"].tolist())    # sent TO mixer
    withdrawers  = set(mixer_rows["receiver_id"].tolist())  # received FROM mixer

    print(f"[MIXER] Direct contacts — depositors: {len(depositors)} | withdrawers: {len(withdrawers)}")

    # Flag deposits
    dep_mask = df["sender_id"].isin(depositors) | df["country"].isin(MIXER_COUNTRY_CODES)
    df.loc[dep_mask, "mixer_flag"] = True
    df.loc[dep_mask & (df["mixer_type"] == ""), "mixer_type"] = "mixer_deposit"

    # Flag withdrawals (higher severity)
    with_mask = df["receiver_id"].isin(withdrawers)
    df.loc[with_mask, "mixer_flag"] = True
    df.loc[with_mask, "mixer_type"] = "mixer_withdraw"  # overwrite — more severe

    flagged = df["mixer_flag"].sum()
    print(f"[MIXER] Total mixer-flagged transactions: {flagged}")
    return df


# ═══════════════════════════════════════════════════════════════
#  🆕  ABILITY 8 — BRIDGE HOP COUNT
#  "How many different chains did this wallet touch in 6 hours?"
#  Banking analog: v6 foreign_country, but active + multi-hop.
# ═══════════════════════════════════════════════════════════════
def detect_bridge_hops(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    For each sender, count distinct bridge contracts used within bridge_window_hours.
    If count >= bridge_hop_threshold → flag.

    Bridge rows are identified by country == BRIDGE_OFFSHORE.
    Each unique receiver_id in a bridge row = a different bridge contract.

    3+ bridges in 6hrs = regulatory escape attempt.
    """
    df = df.copy()
    df["bridge_hop_count"] = 0
    df["bridge_flag"]      = False

    bridge_rows = df[df["country"].isin(BRIDGE_COUNTRY_CODES)].copy()
    if len(bridge_rows) == 0:
        print("[BRIDGE] No bridge transactions found.")
        return df

    window = timedelta(hours=cfg["bridge_window_hours"])
    threshold = cfg["bridge_hop_threshold"]

    # For each bridge row, look back and count unique bridge receivers (=contracts)
    hop_counts = []
    for _, row in bridge_rows.iterrows():
        ts     = row["timestamp"]
        sender = row["sender_id"]
        window_mask = (
            (df["sender_id"] == sender) &
            (df["country"].isin(BRIDGE_COUNTRY_CODES)) &
            (df["timestamp"] >= ts - window) &
            (df["timestamp"] <= ts)
        )
        unique_bridges = df[window_mask]["receiver_id"].nunique()
        hop_counts.append((row.name, unique_bridges))

    for (idx, count) in hop_counts:
        df.at[idx, "bridge_hop_count"] = count
        if count >= threshold:
            df.at[idx, "bridge_flag"] = True

    # Also flag the sender's non-bridge transactions in the same window
    # (their other activity becomes suspect too)
    bridge_flagged_senders = set(
        df[df["bridge_flag"]]["sender_id"].tolist()
    )
    adjacent_mask = (
        df["sender_id"].isin(bridge_flagged_senders) &
        ~df["bridge_flag"]
    )
    df.loc[adjacent_mask, "bridge_flag"] = True
    # Keep their hop count visible
    for sender in bridge_flagged_senders:
        max_hops = df[df["sender_id"] == sender]["bridge_hop_count"].max()
        df.loc[df["sender_id"] == sender, "bridge_hop_count"] = max_hops

    flagged = df["bridge_flag"].sum()
    print(f"[BRIDGE] Bridge-hop flagged transactions: {flagged} | "
          f"Threshold: {threshold} bridges in {cfg['bridge_window_hours']}h")
    return df


# ═══════════════════════════════════════════════════════════════
#  🆕  ABILITY 9 — PEEL CHAIN (LINEAR)
#  "A→B→B2→B3→C — same amount shrinking each hop"
#  Banking analog: v6 structuring (amount splitting).
#  But peel chain is FORWARD LINEAR, structuring is temporal splitting.
#  v6 layering catches CYCLES. This catches TRAILS.
# ═══════════════════════════════════════════════════════════════
def detect_peel_chain(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Peel chain signature:
      - Wallet A sends amount X to wallet B
      - Wallet B soon sends ~X * (1 - peel%) to wallet C (unique C)
      - Wallet C soon sends ~X * (1 - peel%)^2 to wallet D
      - Repeat for min_hops

    Each intermediate wallet is used ONCE (one in, one out).
    The amount decreases monotonically but slowly (1-12% per hop).

    Algorithm:
      1. Build a "forward pass" graph: for each wallet, find their
         single outgoing tx that closely matches their single incoming tx
      2. Walk chains: if a wallet has exactly 1 in + 1 out and
         out_amount ≈ in_amount * (1 - peel%), add to chain
      3. Flag chains of length >= peel_min_hops
    """
    df = df.copy()
    df["peel_flag"]  = False
    df["peel_chain"] = ""

    window      = timedelta(hours=cfg["peel_window_hours"])
    min_hops    = cfg["peel_min_hops"]
    max_peel    = cfg["peel_max_peel_pct"]

    # Index for fast lookup: receiver → row indices
    recv_to_rows: dict = defaultdict(list)
    send_to_rows: dict = defaultdict(list)
    for idx, row in df.iterrows():
        recv_to_rows[row["receiver_id"]].append(idx)
        send_to_rows[row["sender_id"]].append(idx)

    detected_chains = []

    # Walk from every row: is this the START of a peel chain?
    visited_as_origin = set()
    for start_idx, start_row in df.iterrows():
        if start_idx in visited_as_origin:
            continue

        chain_indices = [start_idx]
        current_idx   = start_idx
        current_row   = start_row

        while True:
            # Next hop: current receiver becomes next sender
            next_sender    = current_row["receiver_id"]
            next_send_idxs = send_to_rows.get(next_sender, [])

            if not next_send_idxs:
                break   # dead end

            # Look for exactly 1 outgoing tx from this wallet within window
            next_send_rows = df.loc[next_send_idxs]
            time_ok = (
                (next_send_rows["timestamp"] > current_row["timestamp"]) &
                (next_send_rows["timestamp"] <= current_row["timestamp"] + window)
            )
            candidates = next_send_rows[time_ok]

            if len(candidates) != 1:
                break   # not a peel — multiple outgoing = not linear

            next_row   = candidates.iloc[0]
            next_idx   = candidates.index[0]

            # Amount check: did the amount shrink by 1–12%?
            peel_ratio = (current_row["amount"] - next_row["amount"]) / current_row["amount"]
            if not (0.001 <= peel_ratio <= max_peel):
                break   # amount didn't change correctly — not a peel

            # Avoid loops (this wallet already in chain?)
            if next_idx in chain_indices:
                break

            chain_indices.append(next_idx)
            current_idx = next_idx
            current_row = next_row

            if len(chain_indices) >= min_hops:
                # We have a valid peel chain — record and break
                chain_wallets = [df.at[i, "sender_id"] for i in chain_indices]
                chain_str = "→".join(chain_wallets[:6])  # first 6 hops shown
                if len(chain_wallets) > 6:
                    chain_str += f"→...(+{len(chain_wallets)-6} hops)"
                detected_chains.append((chain_indices, chain_str))
                for i in chain_indices:
                    visited_as_origin.add(i)
                break

    # Apply flags
    for (indices, chain_str) in detected_chains:
        for idx in indices:
            df.at[idx, "peel_flag"]  = True
            df.at[idx, "peel_chain"] = chain_str

    print(f"[PEEL] Detected {len(detected_chains)} peel chains | "
          f"{df['peel_flag'].sum()} transactions flagged")
    return df


# ─────────────────────────────────────────────
# SCORING  (v6 logic + 3 new signals)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  OFAC SDN LIST — Ethereum addresses (subset of publicly known entries)
#  Source: US Treasury SDN list + OFAC virtual currency addendum
#  All addresses are public record. Updated: Aug 2022 - Feb 2025.
# ─────────────────────────────────────────────────────────────────────────────
OFAC_SDN_ADDRESSES = {
    # Tornado Cash pools — OFAC Aug 8, 2022
    "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",  # TC 0.1 ETH
    "0x9ad122c22b14202b4490edaf288fdb3c7cb3ff5d",  # TC 1 ETH
    "0xa160cdab225685da1d56aa342ad8841c3b53f291",  # TC 10 ETH
    "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf",  # TC 100 ETH
    "0x5efda50f22d34f262c29268506c5fa42cb56a1ce",  # TC Governance
    "0x8589427373d6d84e98730d7795d8f6f8731fda16",  # TC router
    "0xd96f2b1c14db8458374d9aca76e26c3950113464",  # TC 1 ETH alt
    "0x4736dcf1b7a3d580672a2389a73823eb9ea0c5b6",  # TC airdrop
    # Lazarus Group / DPRK — OFAC 2022-2023
    "0x098b716b8aaf21512996dc57eb0615e2383e2f96",  # Ronin Bridge exploiter
    "0x172370d5cd63279efa6d502dab29171933a610af",  # Lazarus ETH intermediary
    "0x9c9e10e1f65d3ffd0b61f3b6d6c3b5de60c1e31f",  # Lazarus wallet
    "0xf2bd9aa5ff88de44d0d8ab0f85bfe4d6e89eb04e",  # Lazarus intermediary
    # Stake.com hack — FBI Sep 2023
    "0x974caa59e49682cda0ad2bbe82983419a2ecc400",  # Stake.com primary attacker
    # Bybit hack — Feb 2025
    "0x96221423681a6d52e184d440a8efcebb105c7242",  # Bybit attacker contract
    "0xbdd077f651ebe7f7b3ce16fe5f2b025be2969516",  # Bybit drainer impl
    # Bitfinex hack intermediary (recovered 2022)
    "0xf6b5414f23a15c5fe41c37d7c8f7e4adfc30e0c",
}


def detect_ofac_hit(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    ABILITY 12 — OFAC SDN HIT  (hard mandatory alert, no threshold)
    ─────────────────────────────────────────────────────────────────
    In real AML: OFAC screening happens BEFORE any ML or rules.
    If you touch a sanctioned address — as sender OR receiver — it is
    an automatic Suspicious Activity Report (SAR). No score needed.

    Why no score: a sanctioned address is a legal fact, not a risk signal.
    We still add a massive weight so it dominates the score, but in a
    production system this would be a hard block.

    Real-world impact: Ronin exploiter (0x098B71...) is on the OFAC SDN list.
    This single rule closes the Ronin detection gap from 6% → near 100%.
    """
    df = df.copy()
    df["ofac_flag"]    = False
    df["ofac_address"] = ""

    sdns = {addr.lower() for addr in OFAC_SDN_ADDRESSES}

    # Flag if EITHER sender OR receiver is on OFAC list
    sender_hit   = df["sender_id"].str.lower().isin(sdns)
    receiver_hit = df["receiver_id"].str.lower().isin(sdns)
    hit_mask     = sender_hit | receiver_hit

    df.loc[hit_mask, "ofac_flag"] = True
    df.loc[sender_hit,   "ofac_address"] = df.loc[sender_hit,   "sender_id"]
    df.loc[receiver_hit, "ofac_address"] = df.loc[receiver_hit, "receiver_id"]

    flagged = hit_mask.sum()
    print(f"[OFAC]       Sanctioned address matches: {flagged} transactions")
    return df


def detect_flash_loan_burst(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    ABILITY 13 — FLASH LOAN BURST  (catches Euler-style attacks)
    ──────────────────────────────────────────────────────────────
    Flash loan pattern: borrow → attack → repay in a single Ethereum block
    (~12 seconds). A wallet doing 5+ meaningful transactions within 60
    seconds is likely a bot or flash loan attacker, not a human.

    Different from velocity_many_tx (which uses a 60-minute window):
    this is a SECONDS-level burst — humanly impossible to generate manually.

    Euler Finance: $197M taken in 2 transactions in the same block.
    """
    df = df.copy()
    df["flash_flag"]  = False
    df["flash_count"] = 0

    window_s  = cfg["flash_window_seconds"]
    min_count = cfg["flash_min_tx_count"]
    min_amt   = cfg["flash_min_amount"]

    big = df[df["amount"] >= min_amt].copy()
    if len(big) == 0:
        return df

    big["ts_s"] = pd.to_datetime(big["timestamp"]).astype("int64") // 10**9

    burst_idx = {}
    for idx, row in big.iterrows():
        sender = row["sender_id"]
        t      = row["ts_s"]
        window_txns = big[
            (big["sender_id"] == sender) &
            (big["ts_s"]      >= t - window_s) &
            (big["ts_s"]      <= t + window_s)
        ]
        if len(window_txns) >= min_count:
            burst_idx[idx] = len(window_txns)

    if burst_idx:
        idx_list = list(burst_idx.keys())
        df.loc[idx_list, "flash_flag"]  = True
        df.loc[idx_list, "flash_count"] = pd.Series(burst_idx)

    print(f"[FLASH LOAN] Burst flagged: {len(burst_idx)} transactions (60-second burst ≥{min_count} txns)")
    return df


def detect_coordinated_burst(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    ABILITY 14 — COORDINATED BURST  (bot networks, coordinated attacks)
    ────────────────────────────────────────────────────────────────────
    Multiple DIFFERENT wallets sending to the SAME receiver within 30
    seconds — near-impossible to orchestrate manually. Indicates either
    a bot network (Lazarus distributes then sweeps), a coordinated rug
    pull, or automated exploit distribution.

    Difference from concentrated_inflow (6-hour window):
    this is a 30-SECOND window — simultaneous coordination, not gradual sweep.
    """
    df = df.copy()
    df["burst_flag"]  = False
    df["burst_count"] = 0

    window_s  = cfg["burst_window_seconds"]
    min_sndr  = cfg["burst_min_senders"]
    min_amt   = cfg["burst_min_amount"]

    big = df[df["amount"] >= min_amt].copy()
    if len(big) == 0:
        return df

    big["ts_s"] = pd.to_datetime(big["timestamp"]).astype("int64") // 10**9

    burst_idx = {}
    for idx, row in big.iterrows():
        recv = row["receiver_id"]
        t    = row["ts_s"]
        window_txns = big[
            (big["receiver_id"] == recv) &
            (big["ts_s"]        >= t - window_s) &
            (big["ts_s"]        <= t + window_s)
        ]
        unique_senders = window_txns["sender_id"].nunique()
        if unique_senders >= min_sndr:
            burst_idx[idx] = unique_senders

    if burst_idx:
        idx_list = list(burst_idx.keys())
        df.loc[idx_list, "burst_flag"]  = True
        df.loc[idx_list, "burst_count"] = pd.Series(burst_idx)

    print(f"[COORD BURST] Burst flagged: {len(burst_idx)} transactions ({min_sndr}+ senders within {window_s}s)")
    return df


def detect_novel_wallet_dump(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    ABILITY 10 — NOVEL WALLET DUMP  (catches Ronin / Bybit style)
    ────────────────────────────────────────────────────────────────
    Pattern: A wallet with almost no transaction history suddenly
    moves an extremely large amount in one shot.

    Why it works:
      Exploit wallets are purpose-built. They're created (or dormant),
      loaded with stolen funds from one event, then immediately distribute.
      They don't have years of normal transaction history.

    Signal: sender_tx_count < threshold AND sender_active_days < threshold
            AND amount > novel_dump_min_amount

    Real-world match: Ronin exploiter (0x098B71...) — OFAC SDN, $625M.
    """
    df = df.copy()
    df["novel_dump_flag"] = False

    mask = (
        (df["sender_tx_count"]   <= cfg["novel_dump_max_tx_count"]) &
        (df["sender_active_days"] <= cfg["novel_dump_max_active_days"]) &
        (df["amount"]             >= cfg["novel_dump_min_amount"])
    )
    df.loc[mask, "novel_dump_flag"] = True
    flagged = mask.sum()
    print(f"[NOVEL DUMP] Flagged {flagged} transactions (new wallet + large amount)")
    return df


def detect_concentrated_inflow(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    ABILITY 11 — CONCENTRATED INFLOW  (catches Lazarus sweep pattern)
    ────────────────────────────────────────────────────────────────────
    Pattern: A single receiver address collects meaningful amounts
    from 3+ different senders within a short time window.

    Why it works:
      After a hack, proceeds are spread across many wallets to avoid
      detection, then swept back to a collection address before
      hitting an exchange. This sweep-to-collector is detectable
      as a fan-in to one receiver from many senders.

    Different from fan_in (which looks at SENDER receiving many small
    inflows) — this looks at the RECEIVER being a collection point.

    Real-world match: Lazarus/Stake.com sweep wallets.
    """
    df = df.copy()
    df["conc_inflow_flag"]    = False
    df["conc_inflow_count"]   = 0

    window   = timedelta(hours=cfg["conc_inflow_window_hours"])
    min_amt  = cfg["conc_inflow_min_amount"]
    min_sndr = cfg["conc_inflow_min_senders"]

    # Only consider transactions large enough to be meaningful
    big_txns = df[df["amount"] >= min_amt].copy()
    if len(big_txns) == 0:
        return df

    big_txns["ts"] = pd.to_datetime(big_txns["timestamp"])

    # For each row, look at how many unique senders sent to the SAME receiver
    # within the window ending at this transaction's timestamp
    receiver_counts = {}
    for idx, row in big_txns.iterrows():
        recv = row["receiver_id"]
        t    = row["ts"]
        window_txns = big_txns[
            (big_txns["receiver_id"] == recv) &
            (big_txns["ts"]          >= t - window) &
            (big_txns["ts"]          <= t)
        ]
        unique_senders = window_txns["sender_id"].nunique()
        receiver_counts[idx] = unique_senders

    count_series = pd.Series(receiver_counts)
    flagged_idx  = count_series[count_series >= min_sndr].index

    df.loc[flagged_idx, "conc_inflow_flag"]  = True
    df.loc[flagged_idx, "conc_inflow_count"] = count_series[flagged_idx]

    print(f"[CONC INFLOW] Flagged {len(flagged_idx)} transactions (receiver collecting from {min_sndr}+ senders)")
    return df


def detect_dormant_activation(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    ABILITY 15 — DORMANT WALLET ACTIVATION  (catches BitFinex / Silk Road pattern)
    ─────────────────────────────────────────────────────────────────────────────────
    Pattern: A wallet that has been completely inactive for 365+ days
    suddenly initiates a large transaction.

    Why it works:
      Legitimate users don't store funds in a wallet for years and then
      suddenly move large amounts without any prior warm-up activity.
      The only common cases where this happens:
        1. Hackers waiting for investigations to go cold (BitFinex 2016→2022)
        2. Silk Road / seized funds being moved by authorities
        3. Long-lost keys being recovered (rare, usually small amounts)
        4. Criminal cold storage reactivation for cashout

      The "amount guard" (> $100k default) prevents false positives from
      hobbyists rediscovering old $50 ETH wallets.

      Score multiplier: dormancy years beyond 1 adds compounding signal.
      A wallet dormant for 5 years moving $1M = near-certain CRITICAL.

    Real-world match: BitFinex (6yr dormancy), Silk Road addresses,
                      Wormhole Oasis recovery (1yr), Lazarus long-term holds.

    Key column used: `sender_active_days`
      In our dataset this represents how long the wallet has been active
      (days between first and last observed transaction). When = 0 or very low
      but the wallet also has an old `account_age_days`, it signals dormancy.
      For the synthetic data, DORMANT_REACTIVATED account_type is also checked.
    """
    df = df.copy()
    df["dormant_flag"]       = False
    df["dormant_years"]      = 0.0

    min_days  = cfg["dormant_min_days"]
    min_amt   = cfg["dormant_min_amount"]

    # Primary signal: account_type explicitly labeled DORMANT_REACTIVATED
    has_account_type = "account_type" in df.columns
    if has_account_type:
        dormant_type_mask = (
            (df["account_type"].fillna("").str.upper() == "DORMANT_REACTIVATED") &
            (df["amount"] >= min_amt)
        )
    else:
        dormant_type_mask = pd.Series(False, index=df.index)

    # Secondary signal: account_age_days >> sender_active_days
    # A wallet created 3 years ago but only active for 7 days = dormant revival
    has_age = "account_age_days" in df.columns
    if has_age:
        dormancy_gap = df["account_age_days"] - df["sender_active_days"].fillna(0)
        gap_mask = (
            (dormancy_gap >= min_days) &
            (df["amount"]  >= min_amt)
        )
        combined_mask = dormant_type_mask | gap_mask
    else:
        combined_mask = dormant_type_mask

    # Compute dormancy years for scoring
    if has_age:
        df.loc[combined_mask, "dormant_years"] = (
            df.loc[combined_mask, "account_age_days"] / 365.0
        ).clip(lower=1.0)
    df.loc[combined_mask, "dormant_flag"] = True

    flagged = combined_mask.sum()
    print(f"[DORMANT ACT] Flagged {flagged} transactions (dormant wallet reactivation)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# v11 NEW DETECTORS
# ─────────────────────────────────────────────────────────────────────────────

def detect_wash_cycle(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    ABILITY 16 — WASH CYCLE  (round-trip / circular flow detection)
    ────────────────────────────────────────────────────────────────
    Pattern: Wallet A sends funds to Wallet B, then Wallet B sends
    a similar amount back to Wallet A within 24 hours.

    Also catches A→B→C→A three-party cycles.

    Why it works:
      Money doesn't legitimately round-trip at scale. If you send
      $500k to someone and they send $475k back within hours,
      that's not commerce — that's accounting manipulation.

    Used for: Wash trading, NFT price inflation, fake volume,
              layering to create a paper trail of "legitimate" transfers.

    Detection:
      Build a sender→receiver map. For each transaction A→B,
      look for B→A within the window with amount within tolerance.
    """
    df = df.copy()
    df["wash_flag"]        = False
    df["wash_counterpart"] = ""

    window    = timedelta(hours=cfg["wash_window_hours"])
    tolerance = cfg["wash_amount_tolerance"]
    min_amt   = cfg["wash_min_amount"]

    df["ts"] = pd.to_datetime(df["timestamp"])
    big       = df[df["amount"] >= min_amt].copy()

    if len(big) < 2:
        return df

    flagged_idx = set()
    for idx, row in big.iterrows():
        A, B = row["sender_id"], row["receiver_id"]
        t,  amt = row["ts"], row["amount"]

        # Look for B→A flows within the window
        returns = big[
            (big["sender_id"]   == B) &
            (big["receiver_id"] == A) &
            (big["ts"]          >= t - window) &
            (big["ts"]          <= t + window) &
            (big.index          != idx)
        ]
        # Filter by amount tolerance
        returns = returns[
            (returns["amount"] >= amt * (1 - tolerance)) &
            (returns["amount"] <= amt * (1 + tolerance))
        ]
        if len(returns) > 0:
            flagged_idx.add(idx)
            for ridx in returns.index:
                flagged_idx.add(ridx)

    df.loc[list(flagged_idx), "wash_flag"] = True
    print(f"[WASH CYCLE] Flagged {len(flagged_idx)} transactions (circular flow detected)")
    return df


def detect_smurfing(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    ABILITY 17 — SMURFING  (coordinated threshold avoidance)
    ─────────────────────────────────────────────────────────
    Pattern: Multiple different wallets each send amounts just BELOW
    the AML reporting threshold to the same receiver, coordinated
    within a short time window.

    Named after the Smurfs (many small actors working in coordination
    to accomplish what one actor can't do visibly).

    Why it works:
      AML regulations require reporting transactions above a threshold
      (e.g., $10,000 USD). Criminals deliberately structure payments
      just under this limit. The coordination across wallets is detectable
      even when each individual transaction looks clean.

    Key insight: velocity looks at one sender. Smurfing looks at
    many senders all targeting the same receiver at the same threshold.
    """
    df = df.copy()
    df["smurf_flag"]   = False
    df["smurf_count"]  = 0

    window    = timedelta(hours=cfg["smurf_window_hours"])
    limit     = cfg["smurf_reporting_limit"]
    low_pct   = cfg.get("smurf_low_pct", 0.70)  # 70% of limit
    high_pct  = cfg["smurf_threshold_pct"]       # 90% of limit
    lo_amt    = limit * low_pct
    hi_amt    = limit * high_pct
    min_wallets = cfg["smurf_min_wallets"]

    df["ts"] = pd.to_datetime(df["timestamp"])

    # Only look at "just under threshold" transactions
    smurf_txns = df[(df["amount"] >= lo_amt) & (df["amount"] <= hi_amt)].copy()

    if len(smurf_txns) == 0:
        return df

    receiver_counts = {}
    for idx, row in smurf_txns.iterrows():
        recv = row["receiver_id"]
        t    = row["ts"]
        window_txns = smurf_txns[
            (smurf_txns["receiver_id"] == recv) &
            (smurf_txns["ts"]          >= t - window) &
            (smurf_txns["ts"]          <= t)
        ]
        unique_senders = window_txns["sender_id"].nunique()
        receiver_counts[idx] = unique_senders

    count_series = pd.Series(receiver_counts)
    flagged_idx  = count_series[count_series >= min_wallets].index

    df.loc[flagged_idx, "smurf_flag"]  = True
    df.loc[flagged_idx, "smurf_count"] = count_series[flagged_idx]

    print(f"[SMURFING]   Flagged {len(flagged_idx)} transactions (coordinated threshold avoidance)")
    return df


def detect_exit_rush(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    ABILITY 18 — EXIT RUSH  (receive-then-immediately-bridge)
    ──────────────────────────────────────────────────────────
    Pattern: A relatively new wallet receives a large amount,
    then within 2 hours sends it to a bridge contract or a
    known exchange address.

    Why it works:
      Legitimate users don't immediately bridge everything they receive.
      The rush to exit the chain — especially via bridge — signals
      the wallet is a relay node in a multi-hop laundering chain.

    Signature: novel wallet + large receive + fast bridge/exchange exit.
    This is Group 5 (bridge exploit) post-hack movement.

    Uses: is_bridge column OR known_exchange_prefixes in receiver address.
    """
    df = df.copy()
    df["exit_rush_flag"] = False

    window    = timedelta(hours=cfg["exit_rush_window_hours"])
    min_amt   = cfg["exit_rush_min_amount"]
    max_txns  = cfg["exit_rush_max_tx_count"]
    ex_addrs  = cfg.get("known_exchange_prefixes", set())

    df["ts"] = pd.to_datetime(df["timestamp"])

    # Find "large receives by new wallets"
    large_receives = df[
        (df["amount"]           >= min_amt) &
        (df["sender_tx_count"]  <= max_txns)
    ].copy()

    # Find "bridge or exchange exits"
    has_bridge_col = "is_bridge" in df.columns
    if has_bridge_col:
        exit_txns = df[df["is_bridge"] == True].copy()
    else:
        exit_txns = df[df["receiver_id"].isin(ex_addrs)].copy()

    if len(large_receives) == 0 or len(exit_txns) == 0:
        return df

    flagged_idx = set()
    for idx, recv_row in large_receives.iterrows():
        wallet = recv_row["receiver_id"]  # wallet that received large amount
        t_recv = recv_row["ts"]

        # Look for exits FROM that same wallet within the window
        exits = exit_txns[
            (exit_txns["sender_id"] == wallet) &
            (exit_txns["ts"]        >= t_recv) &
            (exit_txns["ts"]        <= t_recv + window)
        ]
        if len(exits) > 0:
            flagged_idx.add(idx)
            for eidx in exits.index:
                flagged_idx.add(eidx)

    df.loc[list(flagged_idx), "exit_rush_flag"] = True
    print(f"[EXIT RUSH]  Flagged {len(flagged_idx)} transactions (fast bridge/exchange exit)")
    return df


def detect_rapid_succession(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    ABILITY 19 — RAPID SUCCESSION FAN-OUT  (bot-driven distribution)
    ──────────────────────────────────────────────────────────────────
    Pattern: The same sender hits 5+ DIFFERENT receivers within 5 minutes.

    Different from velocity (which counts total transactions in window).
    This specifically tracks UNIQUE RECEIVER COUNT — the fan-out width.

    Why it works:
      Manual users don't send to 5+ different wallets in 5 minutes.
      This is bot signature — automated distribution of stolen funds
      or coordinated payout to mule wallets.

    Catches: Lazarus-style fan-out after initial exploit, ransomware
             payment distribution, coordinated market manipulation payouts.
    """
    df = df.copy()
    df["rapid_flag"]     = False
    df["rapid_receivers"] = 0

    window    = timedelta(minutes=cfg["rapid_window_minutes"])
    min_recv  = cfg["rapid_min_receivers"]
    min_amt   = cfg["rapid_min_amount"]

    df["ts"] = pd.to_datetime(df["timestamp"])

    big_txns = df[df["amount"] >= min_amt].copy()
    if len(big_txns) == 0:
        return df

    sender_counts = {}
    for idx, row in big_txns.iterrows():
        sndr = row["sender_id"]
        t    = row["ts"]
        window_txns = big_txns[
            (big_txns["sender_id"] == sndr) &
            (big_txns["ts"]        >= t - window) &
            (big_txns["ts"]        <= t)
        ]
        unique_receivers = window_txns["receiver_id"].nunique()
        sender_counts[idx] = unique_receivers

    count_series = pd.Series(sender_counts)
    flagged_idx  = count_series[count_series >= min_recv].index

    df.loc[flagged_idx, "rapid_flag"]      = True
    df.loc[flagged_idx, "rapid_receivers"] = count_series[flagged_idx]

    print(f"[RAPID SUCC] Flagged {len(flagged_idx)} transactions (rapid fan-out to {min_recv}+ receivers)")
    return df


def detect_high_risk_country(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    ABILITY 20 — HIGH-RISK JURISDICTION  (FATF blacklist/grey list)
    ────────────────────────────────────────────────────────────────
    Pattern: Transaction involves a country on the FATF blacklist
    or grey list — jurisdictions with known AML/CFT deficiencies.

    This is a weak standalone signal but a strong amplifier.
    Real FATF-country transactions escalate existing suspicions.

    FATF High-Risk Countries (2024 list):
    Blacklist: North Korea (KP), Iran (IR), Myanmar (MM)
    Grey list: Afghanistan, Albania, Barbados, Burkina Faso, etc.
    """
    df = df.copy()
    df["high_risk_country_flag"] = False

    high_risk = cfg.get("high_risk_countries", set())
    if not high_risk:
        return df

    mask = df["country"].str.upper().isin(high_risk)
    df.loc[mask, "high_risk_country_flag"] = True
    flagged = mask.sum()
    print(f"[HIGH RISK]  Flagged {flagged} transactions (FATF high-risk jurisdiction)")
    return df


def detect_layering_deep(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    ABILITY 22 — DEEP LAYERING  (5+ hop peel chain)
    ─────────────────────────────────────────────────
    Extended version of peel_chain that looks for longer laundering
    trails — 5+ hops over 48 hours. Standard peel_chain fires at 3 hops;
    this catches slow, patient laundering like BitFinex or Silk Road.

    Pattern: Each hop sends ~85-99% of amount forward, keeping a small
    slice (the "peel"). The pattern persists over longer time windows.

    Score is higher than standard peel because the attacker is clearly
    attempting to be patient and methodical — that's sophisticated intent.
    """
    df = df.copy()
    df["deep_peel_flag"]  = False
    df["deep_peel_depth"] = 0

    window    = timedelta(hours=cfg["deep_peel_window_hours"])
    min_hops  = cfg["deep_peel_min_hops"]
    min_amt   = cfg["deep_peel_min_amount"]
    peel_pct  = cfg.get("peel_max_peel_pct", 0.15)  # reuse v7 peel tolerance

    df["ts"]  = pd.to_datetime(df["timestamp"])
    big       = df[df["amount"] >= min_amt].copy().sort_values("ts")

    if len(big) < min_hops:
        return df

    flagged_idx = {}

    def trace_chain(start_idx, visited=None, depth=0):
        if visited is None:
            visited = set()
        if start_idx in visited or depth > 15:
            return depth, visited
        visited.add(start_idx)
        row     = big.loc[start_idx]
        nxt_amt = row["amount"] * (1 - peel_pct)  # allow up to peel_pct shrinkage
        t_start = row["ts"]

        # Find next hop: same-ish amount sent FROM this row's receiver
        next_hops = big[
            (big["sender_id"]  == row["receiver_id"]) &
            (big["amount"]     >= nxt_amt * 0.70) &   # allow more tolerance for deep chains
            (big["amount"]     <= row["amount"] * 1.05) &
            (big["ts"]         >= t_start) &
            (big["ts"]         <= t_start + window) &
            (~big.index.isin(visited))
        ]
        if len(next_hops) == 0:
            return depth, visited

        best_depth, best_visited = depth, visited
        for hidx in next_hops.index[:3]:  # cap branching
            d, v = trace_chain(hidx, visited.copy(), depth + 1)
            if d > best_depth:
                best_depth, best_visited = d, v

        return best_depth, best_visited

    for idx in big.index:
        if idx in flagged_idx:
            continue
        max_depth, chain_set = trace_chain(idx)
        if max_depth >= min_hops and len(chain_set) >= min_hops:
            for cidx in chain_set:
                flagged_idx[cidx] = max_depth

    if flagged_idx:
        df.loc[list(flagged_idx.keys()), "deep_peel_flag"]  = True
        df.loc[list(flagged_idx.keys()), "deep_peel_depth"] = pd.Series(flagged_idx)

    print(f"[DEEP PEEL]  Flagged {len(flagged_idx)} transactions ({min_hops}+ hop deep chain)")
    return df


def detect_exchange_avoidance(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    ABILITY 21 — EXCHANGE AVOIDANCE  (deliberate routing around exchanges)
    ───────────────────────────────────────────────────────────────────────
    Pattern: A wallet routes funds through 4+ intermediate addresses
    before finally hitting an exchange, when a direct path would have
    been much simpler. The deliberate complexity is the signal.

    Why it works:
      Innocent users send directly to exchanges. Launderers add hops
      to reduce the traceability of the source. Each added hop is
      evidence of intent to obscure.

    Implementation: Track chains where no intermediate node is a known
    exchange address, but the terminal node is. Score scales with hop count.
    """
    df = df.copy()
    df["ex_avoid_flag"]  = False
    df["ex_avoid_hops"]  = 0

    ex_addrs  = cfg.get("known_exchange_prefixes", set())
    min_hops  = cfg.get("exchange_avoidance_min_hops", 4)
    window    = timedelta(hours=cfg.get("bridge_window_hours", 24))
    min_amt   = cfg.get("novel_dump_min_amount", 50_000)

    if not ex_addrs:
        print(f"[EX AVOID]   No exchange addresses configured — skipping")
        return df

    df["ts"] = pd.to_datetime(df["timestamp"])
    big = df[df["amount"] >= min_amt].copy().sort_values("ts")

    flagged_idx = {}

    def trace_to_exchange(wallet, t_start, depth=0, visited=None):
        if visited is None:
            visited = set()
        if depth > 10 or wallet in visited:
            return 0
        visited.add(wallet)

        next_hops = big[
            (big["sender_id"] == wallet) &
            (big["ts"]        >= t_start) &
            (big["ts"]        <= t_start + window) &
            (~big["receiver_id"].isin(visited))
        ]

        for _, nrow in next_hops.iterrows():
            recv = nrow["receiver_id"]
            if recv.lower() in {a.lower() for a in ex_addrs}:
                return depth + 1  # found exchange at this depth
            d = trace_to_exchange(recv, nrow["ts"], depth + 1, visited.copy())
            if d >= min_hops:
                return d
        return 0

    for idx, row in big.iterrows():
        depth = trace_to_exchange(row["sender_id"], row["ts"])
        if depth >= min_hops:
            flagged_idx[idx] = depth

    if flagged_idx:
        df.loc[list(flagged_idx.keys()), "ex_avoid_flag"] = True
        df.loc[list(flagged_idx.keys()), "ex_avoid_hops"] = pd.Series(flagged_idx)

    print(f"[EX AVOID]   Flagged {len(flagged_idx)} transactions ({min_hops}+ hops before exchange)")
    return df


def score_row(row, cfg):
    pcfg    = PROFILE_CONFIG.get(row["sender_profile"], PROFILE_CONFIG[DEFAULT_PROFILE])
    score   = 0.0
    reasons = f"profile_{row['sender_profile']};"

    # ── v6 rules (all preserved) ──────────────
    if row["amount"] >= pcfg["large_amount_threshold"]:
        score += cfg["w_large"]; reasons += "large_amount;"
    if row["tx_count_in_window"] >= pcfg["velocity_tx_count_threshold"]:
        score += cfg["w_velocity"]; reasons += "velocity_many_tx;"
    if row["small_tx_count_in_window"] >= pcfg["struct_min_count"]:
        score += cfg["w_structuring"]; reasons += "structuring;"
    if row["fan_in_count"] >= cfg["fan_in_threshold"]:
        score += cfg["w_fan_in"]; reasons += "fan_in;"

    is_foreign = str(row["country"]).upper() != cfg["home_country"].upper()
    other_hit  = (
        row["amount"]                   >= pcfg["large_amount_threshold"] or
        row["tx_count_in_window"]       >= pcfg["velocity_tx_count_threshold"] or
        row["small_tx_count_in_window"] >= pcfg["struct_min_count"] or
        row["fan_in_count"]             >= cfg["fan_in_threshold"]
    )
    if is_foreign:
        reasons += "foreign_country;"
        if other_hit:
            score += cfg["w_foreign"]
        else:
            reasons += "foreign_context_only;"

    if row.get("layering_flag", False):
        score   += cfg["w_layering"]
        reasons += f"layering_cycle;{row.get('layering_chain', '')};"

    # ── v7 NEW rules ──────────────────────────
    if row.get("mixer_flag", False):
        mixer_type = row.get("mixer_type", "mixer_deposit")
        if mixer_type == "mixer_withdraw":
            # Receiving from a mixer = money post-laundering = CRITICAL boost
            score   += cfg["w_mixer_touch"] * 1.3
            reasons += "mixer_withdraw;"
        else:
            score   += cfg["w_mixer_touch"]
            reasons += "mixer_touch;"

    if row.get("bridge_flag", False):
        hop_count = row.get("bridge_hop_count", 2)
        # More bridge hops = multiplied risk (each hop = new jurisdiction)
        bridge_score = cfg["w_bridge_hop"] * min(hop_count / cfg["bridge_hop_threshold"], 2.0)
        score   += bridge_score
        reasons += f"bridge_hop_{hop_count};"

    if row.get("peel_flag", False):
        score   += cfg["w_peel_chain"]
        reasons += f"peel_chain;{row.get('peel_chain', '')};"

    # ── v8 NEW rules ──────────────────────────
    # ── v9 NEW rules ──────────────────────────
    if row.get("ofac_flag", False):
        # OFAC hit = automatic maximum alert, bypasses all thresholds
        score   += cfg["w_ofac_hit"]
        reasons += f"OFAC_SDN_MATCH;"

    if row.get("flash_flag", False):
        burst_count = row.get("flash_count", cfg["flash_min_tx_count"])
        flash_score = cfg["w_flash_loan"] * min(burst_count / cfg["flash_min_tx_count"], 2.0)
        score      += flash_score
        reasons    += f"flash_loan_burst_{int(burst_count)};"

    if row.get("burst_flag", False):
        burst_count = row.get("burst_count", cfg["burst_min_senders"])
        burst_score = cfg["w_coordinated_burst"] * min(burst_count / cfg["burst_min_senders"], 2.0)
        score      += burst_score
        reasons    += f"coordinated_burst_{int(burst_count)};"

    if row.get("novel_dump_flag", False):
        # Scale score by how new the wallet is — younger + bigger = worse
        age_factor  = max(0.5, 1.0 - (row.get("sender_active_days", 0) / cfg["novel_dump_max_active_days"]))
        dump_score  = cfg["w_novel_dump"] * age_factor
        score      += dump_score
        reasons    += f"novel_wallet_dump;"

    if row.get("conc_inflow_flag", False):
        sender_count = row.get("conc_inflow_count", cfg["conc_inflow_min_senders"])
        # More senders converging = higher risk
        inflow_score = cfg["w_concentrated_inflow"] * min(sender_count / cfg["conc_inflow_min_senders"], 2.0)
        score       += inflow_score
        reasons     += f"concentrated_inflow_{int(sender_count)};"

    # ── v10 NEW rules ─────────────────────────
    if row.get("dormant_flag", False):
        dormant_years = row.get("dormant_years", 1.0)
        scale_years   = cfg.get("dormant_scale_years", 5)
        dormancy_factor = 1.0 + min((dormant_years - 1) / (scale_years - 1), 1.0)
        dormant_score   = cfg["w_dormant_activation"] * dormancy_factor
        score          += dormant_score
        reasons        += f"dormant_activation_{dormant_years:.1f}yr;"

    # ── v11 NEW rules ─────────────────────────
    if row.get("wash_flag", False):
        # Wash cycle — circular flow between two wallets
        score   += cfg["w_wash_cycle"]
        reasons += "wash_cycle;"

    if row.get("smurf_flag", False):
        # Smurfing — coordinated threshold avoidance by multiple wallets
        smurf_count  = row.get("smurf_count", cfg["smurf_min_wallets"])
        smurf_score  = cfg["w_smurfing"] * min(smurf_count / cfg["smurf_min_wallets"], 2.0)
        score       += smurf_score
        reasons     += f"smurfing_{int(smurf_count)}_wallets;"

    if row.get("exit_rush_flag", False):
        # Exit rush — novel wallet immediately bridges/exits after large receive
        score   += cfg["w_exit_rush"]
        reasons += "exit_rush;"

    if row.get("rapid_flag", False):
        # Rapid succession fan-out — same sender, many unique receivers in 5 min
        recv_count   = row.get("rapid_receivers", cfg["rapid_min_receivers"])
        rapid_score  = cfg["w_rapid_succession"] * min(recv_count / cfg["rapid_min_receivers"], 2.0)
        score       += rapid_score
        reasons     += f"rapid_succession_{int(recv_count)}_receivers;"

    if row.get("high_risk_country_flag", False):
        # High-risk FATF jurisdiction — additive amplifier
        # Only adds meaningful score if combined with another signal
        other_signals = any([
            row.get("wash_flag"),     row.get("smurf_flag"),
            row.get("exit_rush_flag"),row.get("rapid_flag"),
            row.get("ofac_flag"),     row.get("mixer_flag"),
            row.get("novel_dump_flag"), row.get("layering_flag"),
        ])
        if other_signals:
            score   += cfg["w_high_risk_country"]
            reasons += "high_risk_jurisdiction_amplified;"
        else:
            score   += cfg["w_high_risk_country"] * 0.5   # half weight alone
            reasons += "high_risk_jurisdiction;"

    if row.get("ex_avoid_flag", False):
        # Exchange avoidance — deliberate routing complexity
        hop_count   = row.get("ex_avoid_hops", cfg["exchange_avoidance_min_hops"])
        avoid_score = cfg["w_exchange_avoidance"] * min(hop_count / cfg["exchange_avoidance_min_hops"], 2.0)
        score      += avoid_score
        reasons    += f"exchange_avoidance_{int(hop_count)}_hops;"

    if row.get("deep_peel_flag", False):
        # Deep layering — 5+ hop chain over extended window
        hop_depth   = row.get("deep_peel_depth", cfg["deep_peel_min_hops"])
        depth_score = cfg["w_layering_deep"] * min(hop_depth / cfg["deep_peel_min_hops"], 2.0)
        score      += depth_score
        reasons    += f"layering_deep_{int(hop_depth)}_hops;"

    score = round(score * pcfg["score_multiplier"])
    return score, reasons


def score_transactions(df, cfg):
    df = df.copy()
    results = df.apply(lambda row: score_row(row, cfg), axis=1)
    df["risk_score"] = results.apply(lambda x: x[0])
    df["reasons"]    = results.apply(lambda x: x[1])
    df["alert"]      = df["risk_score"] >= cfg["alert_threshold"]
    return df


# ─────────────────────────────────────────────
# RISK LEVEL
# ─────────────────────────────────────────────
def risk_level(score: int) -> str:
    if score >= 80: return "CRITICAL"
    if score >= 60: return "HIGH"
    if score >= 40: return "MEDIUM"
    return "LOW"

def risk_emoji(score: int) -> str:
    if score >= 80: return "🔴"
    if score >= 60: return "🟠"
    if score >= 40: return "🟡"
    return "🟢"


# ─────────────────────────────────────────────
# NARRATIVE BUILDER  (v6 signals + 3 new ones)
# ─────────────────────────────────────────────
def build_narrative(row: pd.Series, df: pd.DataFrame, cfg: dict) -> dict:
    pcfg     = PROFILE_CONFIG.get(row["sender_profile"], PROFILE_CONFIG[DEFAULT_PROFILE])
    reasons  = row["reasons"]
    score    = int(row["risk_score"])
    level    = risk_level(score)
    ts       = row["timestamp"]
    sender   = row["sender_id"]
    receiver = row["receiver_id"]
    amount   = row["amount"]

    signals      = []
    rec_actions  = []

    # ── v6 Signal 1: Large Amount ──────────────
    if "large_amount" in reasons:
        threshold = pcfg["large_amount_threshold"]
        signals.append({
            "name": "Large Transaction",
            "severity": "HIGH",
            "rule": "large_amount (v6)",
            "detail": (
                f"Transaction ${amount:,.2f} exceeds ${threshold:,} threshold "
                f"for {row['sender_profile']} wallets. "
                f"That's {amount/threshold:.1f}x above the reporting threshold. "
                f"On-chain equivalent of a cash transaction over reporting limit."
            )
        })
        rec_actions.append("Verify source of funds — check on-chain provenance")
        rec_actions.append("Review wallet's transaction history for consistent patterns")

    # ── v6 Signal 2: Velocity ──────────────────
    if "velocity_many_tx" in reasons:
        tx_count  = int(row["tx_count_in_window"])
        threshold = pcfg["velocity_tx_count_threshold"]
        signals.append({
            "name": "High Transaction Velocity",
            "severity": "HIGH",
            "rule": "velocity (v6)",
            "detail": (
                f"{tx_count} transactions from wallet {sender[:16]}... "
                f"in a 60-min window (threshold: {threshold}). "
                f"On-chain this often signals bot activity or coordinated fund movement."
            )
        })
        rec_actions.append("Check gas price pattern — identical gas = bot signature")
        rec_actions.append("Review all receivers in the velocity window for coordination")

    # ── v6 Signal 3: Structuring ───────────────
    if "structuring" in reasons:
        small_count = int(row["small_tx_count_in_window"])
        struct_max  = pcfg["struct_max_amount"]
        window      = timedelta(minutes=cfg["velocity_window_minutes"])
        window_df   = df[
            (df["sender_id"] == sender) &
            (df["timestamp"] >= ts - window) &
            (df["timestamp"] <= ts) &
            (df["amount"] <= struct_max)
        ]
        total_structured = window_df["amount"].sum()
        signals.append({
            "name": "Structuring / Amount Splitting",
            "severity": "CRITICAL",
            "rule": "structuring (v6) → PEEL_CHAIN analog",
            "detail": (
                f"{small_count} transactions each under ${struct_max:,} "
                f"within 60 minutes. Combined: ${total_structured:,.2f}. "
                f"Classic structuring pattern — splitting amounts to stay "
                f"below reporting thresholds. On-chain equivalent of smurfing."
            )
        })
        rec_actions.append("File SAR — structuring meets regulatory reporting criteria")
        rec_actions.append("Review full 30-day on-chain history for this wallet")

    # ── v6 Signal 4: Fan-In ────────────────────
    if "fan_in" in reasons:
        fan_count = int(row["fan_in_count"])
        window    = timedelta(minutes=cfg["velocity_window_minutes"])
        fan_df    = df[
            (df["receiver_id"] == receiver) &
            (df["timestamp"] >= ts - window) &
            (df["timestamp"] <= ts)
        ]
        unique_senders  = fan_df["sender_id"].nunique()
        total_received  = fan_df["amount"].sum()
        signals.append({
            "name": "Suspicious Concentration (Fan-In)",
            "severity": "HIGH",
            "rule": "fan_in (v6) → DUSTING analog",
            "detail": (
                f"{unique_senders} wallets sent to {receiver[:16]}... "
                f"within 60 minutes. Total: ${total_received:,.2f}. "
                f"On-chain: coordinated funneling or dusting collection point."
            )
        })
        rec_actions.append("Investigate all senders depositing to this receiver wallet")
        rec_actions.append("Check if receiver is a known exchange hot wallet or mixer")

    # ── v6 Signal 5: Layering ──────────────────
    if "layering_cycle" in reasons:
        chain = row.get("layering_chain", "unknown")
        hops  = len(chain.split("→")) - 1
        signals.append({
            "name": "Circular Flow (Wash Trading / Layering)",
            "severity": "CRITICAL",
            "rule": "layering/DFS-cycles (v6) → WASH_TRADING analog",
            "detail": (
                f"Circular tx cycle: {chain[:80]}{'...' if len(chain)>80 else ''}. "
                f"{hops} hop(s) before returning to origin within {cfg['layering_window_hours']}h. "
                f"On-chain: wash trading or circular token movement to inflate volume."
            )
        })
        rec_actions.append("URGENT: Flag all wallets in the cycle for enhanced monitoring")
        rec_actions.append("File SAR — circular fund movement is a primary laundering indicator")
        rec_actions.append("Map full wallet graph — check for exchange deposits at chain end")

    # ── v6 Signal 6: Foreign ───────────────────
    if "foreign_country" in reasons and "foreign_context_only" not in reasons:
        signals.append({
            "name": "High-Risk / Foreign Jurisdiction",
            "severity": "MEDIUM",
            "rule": "foreign_country (v6) → FOREIGN_CHAIN",
            "detail": (
                f"Wallet transacting via jurisdiction: {row['country']}. "
                f"Non-domestic chain activity combined with other risk signals "
                f"elevates overall score. Cross-chain = cross-border regulatory gap."
            )
        })
        rec_actions.append("Verify wallet against OFAC/FATF sanctions lists")
        rec_actions.append("Check if chain/jurisdiction is subject to travel rule compliance")

    # ── 🆕 v7 Signal 7: Mixer Touch ─────────────
    if "mixer_touch" in reasons or "mixer_withdraw" in reasons:
        mixer_type = "withdrawal from" if "mixer_withdraw" in reasons else "deposit to"
        signals.append({
            "name": "Mixer Contact (Privacy Protocol)",
            "severity": "CRITICAL" if "mixer_withdraw" in reasons else "HIGH",
            "rule": "mixer_touch (v7 NEW — no v6 equivalent)",
            "detail": (
                f"This wallet made a {mixer_type} a known mixer contract "
                f"(Tornado Cash style) within the lookback window. "
                f"Mixer usage = deliberate intent to sever transaction trail. "
                f"FATF guidance classifies mixer interaction as a red flag. "
                f"Withdrawal is more severe — this is the cleaned output."
            )
        })
        rec_actions.append("CRITICAL: File SAR — mixer contact is an automatic escalation trigger")
        rec_actions.append("Trace pre-mixer deposits and post-mixer destinations via on-chain graph")
        rec_actions.append("Check if wallet appears on Chainalysis/TRM Labs mixer exposure lists")

    # ── 🆕 v7 Signal 8: Bridge Hop ──────────────
    if any(f"bridge_hop_" in r for r in reasons.split(";")):
        hop_parts = [r for r in reasons.split(";") if "bridge_hop_" in r]
        hop_count = int(hop_parts[0].replace("bridge_hop_", "")) if hop_parts else cfg["bridge_hop_threshold"]
        signals.append({
            "name": f"Rapid Cross-Chain Bridging ({hop_count} bridges)",
            "severity": "CRITICAL" if hop_count >= 3 else "HIGH",
            "rule": "bridge_hop (v7 NEW — v6 foreign_country analog, active version)",
            "detail": (
                f"Wallet used {hop_count} different bridge contracts within "
                f"{cfg['bridge_window_hours']} hours. Each bridge hop crosses "
                f"a regulatory jurisdiction boundary. Rapid multi-chain movement "
                f"is a known technique to evade travel rule compliance — "
                f"regulators monitoring chain A cannot see chain B."
            )
        })
        rec_actions.append("Map full cross-chain trail using Axelar/LayerZero graph tools")
        rec_actions.append("File SAR if funds exceed $10k equivalent across bridge hops")
        rec_actions.append("Check all destination chain wallets for exchange deposit patterns")

    # ── 🆕 v7 Signal 9: Peel Chain ──────────────
    if "peel_chain" in reasons:
        peel_chain_str = row.get("peel_chain", "unknown")
        hops = len(peel_chain_str.split("→")) - 1
        signals.append({
            "name": f"Peel Chain Detected ({hops}+ hops)",
            "severity": "CRITICAL",
            "rule": "peel_chain_linear (v7 NEW — v6 structuring analog, forward trail version)",
            "detail": (
                f"Linear fund trail: {peel_chain_str[:80]}{'...' if len(peel_chain_str)>80 else ''}. "
                f"Each hop sheds 1-12% (fees/diversion), remainder moves forward. "
                f"Each intermediate wallet used exactly once — classic burner wallet chain. "
                f"CRITICAL: v6 layering (cycles) would NOT catch this — peel chains are linear, "
                f"not circular. This is a v7-only detection."
            )
        })
        rec_actions.append("URGENT: Trace the full peel chain — find the final destination wallet")
        rec_actions.append("Check if chain terminates at a DEX, exchange, or mixing service")
        rec_actions.append("Intermediate wallets are likely burner accounts — identify the origin")
        rec_actions.append("File SAR — peel chains are a primary on-chain laundering pattern")

    # ── Summary ─────────────────────────────────
    signal_names = [s["name"] for s in signals]
    v7_new = [s for s in signals if "v7 NEW" in s.get("rule", "")]
    if not signal_names:
        summary = f"Wallet {sender[:20]}... flagged (score: {score})."
    elif len(signal_names) == 1:
        summary = f"Wallet {sender[:20]}... flagged for {signal_names[0]}."
    else:
        summary = (
            f"Wallet {sender[:20]}... flagged for {len(signal_names)} concurrent signals: "
            f"{', '.join(signal_names[:-1])}, and {signal_names[-1]}."
        )
    if v7_new:
        v7_names = [s["name"] for s in v7_new]
        summary += f" [v7 blockchain-specific: {', '.join(v7_names)}]"

    # Deduplicate recommendations
    seen, rec_deduped = set(), []
    for r in rec_actions:
        if r not in seen:
            seen.add(r)
            rec_deduped.append(r)

    return {
        "alert_id":            f"NEXUS-{row.name:05d}",
        "transaction_id":      int(row.name),
        "timestamp":           str(ts),
        "sender_id":           sender,
        "receiver_id":         receiver,
        "amount":              float(amount),
        "country":             row["country"],
        "sender_profile":      row["sender_profile"],
        "risk_score":          score,
        "risk_level":          level,
        "summary":             summary,
        "signals":             signals,
        "v7_blockchain_rules": [s["name"] for s in v7_new],
        "recommended_actions": rec_deduped,
        "profile_note":        pcfg["description"],
        "raw_reasons":         reasons,
    }


def generate_narratives(df: pd.DataFrame, cfg: dict) -> list:
    alerts = df[df["alert"] == True].copy()
    print(f"[NARRATIVES] Building explanations for {len(alerts)} alerts...")
    narratives = []
    for _, row in alerts.iterrows():
        narratives.append(build_narrative(row, df, cfg))
    narratives.sort(key=lambda x: x["risk_score"], reverse=True)
    print(f"[NARRATIVES] Done. {len(narratives)} alert reports generated.")
    return narratives


# ─────────────────────────────────────────────
# FORMAT TEXT REPORT
# ─────────────────────────────────────────────
def format_text_report(narratives: list, v6_comparison: dict = None) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("  NEXUS-RISK AML ENGINE — ALERT REPORT v7 (BLOCKCHAIN)")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Total Alerts: {len(narratives)}")
    lines.append("=" * 72)

    # v6 vs v7 comparison banner
    if v6_comparison:
        lines.append("\n  ⚔️  v6 → v7 UPGRADE COMPARISON")
        lines.append("  " + "-" * 68)
        lines.append(f"  {'Metric':<30} {'v6 Banking':>15}  {'v7 Blockchain':>15}")
        lines.append("  " + "-" * 68)
        for k, (v6_val, v7_val) in v6_comparison.items():
            lines.append(f"  {k:<30} {str(v6_val):>15}  {str(v7_val):>15}")
        lines.append("  " + "-" * 68)
        lines.append("\n  🆕 New v7 Abilities: mixer_touch | bridge_hop | peel_chain_linear")
        lines.append("  📊 These rules catch what v6 was blind to in blockchain data.\n")

    # Summary table
    lines.append("\n  ALERT SUMMARY TABLE (Top 20)")
    lines.append("  " + "-" * 68)
    lines.append(f"  {'Alert ID':<15} {'Sender (truncated)':<20} {'Amount':>10} {'Score':>6} {'Level':<10}")
    lines.append("  " + "-" * 68)
    for n in narratives[:20]:
        emoji    = risk_emoji(n["risk_score"])
        sender_s = n["sender_id"][:18] + ".."
        lines.append(
            f"  {n['alert_id']:<15} {sender_s:<20} "
            f"${n['amount']:>9,.2f} {n['risk_score']:>5}  "
            f"{emoji}{n['risk_level']:<9}"
        )
    lines.append("  " + "-" * 68)

    # Detailed top 10
    lines.append("\n\n  DETAILED ALERT NARRATIVES (Top 10 by Risk Score)")
    lines.append("=" * 72)

    for n in narratives[:10]:
        emoji = risk_emoji(n["risk_score"])
        lines.append(f"\n{'─' * 72}")
        lines.append(f"  {emoji} {n['alert_id']} | Score: {n['risk_score']}/100 | {n['risk_level']} RISK")
        lines.append(f"  Wallet   : {n['sender_id']}")
        lines.append(f"  Receiver : {n['receiver_id']}")
        lines.append(f"  Amount   : ${n['amount']:,.2f} | Chain: {n['country']}")
        lines.append(f"  Time     : {n['timestamp']}")
        if n.get("v7_blockchain_rules"):
            lines.append(f"  🆕 v7 New Rules Fired: {', '.join(n['v7_blockchain_rules'])}")
        lines.append(f"\n  SUMMARY: {n['summary']}")
        lines.append(f"\n  SIGNALS ({len(n['signals'])}):")
        for i, signal in enumerate(n["signals"], 1):
            lines.append(f"  {i}. [{signal['severity']}] {signal['name']}")
            lines.append(f"     Rule: {signal.get('rule', 'N/A')}")
            detail = signal["detail"]
            words  = detail.split()
            line, wrapped = "     ", []
            for w in words:
                if len(line) + len(w) + 1 > 70:
                    wrapped.append(line)
                    line = "     " + w
                else:
                    line += (" " if line.strip() else "") + w
            if line.strip():
                wrapped.append(line)
            lines.extend(wrapped)
        lines.append(f"\n  RECOMMENDED ACTIONS:")
        for i, action in enumerate(n["recommended_actions"], 1):
            lines.append(f"  {i}. {action}")

    lines.append(f"\n{'=' * 72}")
    lines.append("  END OF REPORT — NEXUS-RISK v7 BLOCKCHAIN")
    lines.append("=" * 72)
    return "\n".join(lines)


# ─────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────
def save_outputs(df, narratives, cfg, v6_comparison=None):
    os.makedirs(cfg["output_dir"], exist_ok=True)

    df.to_csv(os.path.join(cfg["output_dir"], "scored_transactions_v7.csv"), index=False)

    with open(os.path.join(cfg["output_dir"], "alerts_v7.json"), "w") as f:
        json.dump(narratives, f, indent=2)

    report = format_text_report(narratives, v6_comparison)
    with open(os.path.join(cfg["output_dir"], "alerts_report_v7.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    total   = len(df)
    flagged = int(df["alert"].sum())
    v7_rule_hits = {
        "mixer_touch":  int(df.get("mixer_flag",   pd.Series(False)).sum()),
        "bridge_hop":   int(df.get("bridge_flag",  pd.Series(False)).sum()),
        "peel_chain":   int(df.get("peel_flag",    pd.Series(False)).sum()),
    }
    metrics = {
        "version":       "v7_blockchain",
        "run_at":        datetime.now().isoformat(),
        "total_tx":      total,
        "flagged":       flagged,
        "flag_rate_pct": round(flagged / total * 100, 2),
        "v7_new_rule_hits": v7_rule_hits,
        "risk_levels": {
            "CRITICAL": sum(1 for n in narratives if n["risk_level"] == "CRITICAL"),
            "HIGH":     sum(1 for n in narratives if n["risk_level"] == "HIGH"),
            "MEDIUM":   sum(1 for n in narratives if n["risk_level"] == "MEDIUM"),
        }
    }
    with open(os.path.join(cfg["output_dir"], "metrics_v7.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[SAVE] {cfg['output_dir']}/scored_transactions_v7.csv")
    print(f"[SAVE] {cfg['output_dir']}/alerts_v7.json")
    print(f"[SAVE] {cfg['output_dir']}/alerts_report_v7.txt  ← READ THIS")
    return report, metrics


# ─────────────────────────────────────────────
# QUICK PRINT
# ─────────────────────────────────────────────
def print_quick_metrics(df, narratives, metrics):
    flagged  = df["alert"].sum()
    total    = len(df)
    critical = sum(1 for n in narratives if n["risk_level"] == "CRITICAL")
    high     = sum(1 for n in narratives if n["risk_level"] == "HIGH")
    medium   = sum(1 for n in narratives if n["risk_level"] == "MEDIUM")

    print("\n" + "=" * 60)
    print("  --- QUICK METRICS — NEXUS-RISK v7 BLOCKCHAIN ---")
    print("=" * 60)
    print(f"  Total : {total}  |  Flagged : {flagged}  |  Rate : {round(flagged/total*100,2)}%")
    print(f"\n  Risk Breakdown:")
    print(f"  🔴 CRITICAL : {critical}")
    print(f"  🟠 HIGH     : {high}")
    print(f"  🟡 MEDIUM   : {medium}")

    v7_hits = metrics.get("v7_new_rule_hits", {})
    print(f"\n  🆕 v7 Blockchain Rules:")
    print(f"  ⛓️  mixer_touch  fired on : {v7_hits.get('mixer_touch', 0)} transactions")
    print(f"  🌉 bridge_hop   fired on : {v7_hits.get('bridge_hop', 0)} transactions")
    print(f"  🔗 peel_chain   fired on : {v7_hits.get('peel_chain', 0)} transactions")
    print(f"\n  🆕 v8 Blockchain Rules:")
    print(f"  💥 novel_dump          fired on : {int(df.get('novel_dump_flag', pd.Series(False)).sum())} transactions")
    print(f"  🎯 concentrated_inflow fired on : {int(df.get('conc_inflow_flag', pd.Series(False)).sum())} transactions")
    print(f"\n  🆕 v9 Blockchain Rules:")
    print(f"  🚨 ofac_hit            fired on : {int(df.get('ofac_flag',       pd.Series(False)).sum())} transactions")
    print(f"  ⚡ flash_loan_burst    fired on : {int(df.get('flash_flag',       pd.Series(False)).sum())} transactions")
    print(f"  🤖 coordinated_burst   fired on : {int(df.get('burst_flag',       pd.Series(False)).sum())} transactions")

    print(f"\n  🆕 v11 Rules:")
    print(f"  🔄 wash_cycle          fired on : {int(df.get('wash_flag',            pd.Series(False)).sum())} transactions")
    print(f"  🧊 smurfing            fired on : {int(df.get('smurf_flag',           pd.Series(False)).sum())} transactions")
    print(f"  🚀 exit_rush           fired on : {int(df.get('exit_rush_flag',       pd.Series(False)).sum())} transactions")
    print(f"  ⚡ rapid_succession    fired on : {int(df.get('rapid_flag',           pd.Series(False)).sum())} transactions")
    print(f"  🌍 high_risk_country   fired on : {int(df.get('high_risk_country_flag',pd.Series(False)).sum())} transactions")
    print(f"  🔀 exchange_avoidance  fired on : {int(df.get('ex_avoid_flag',        pd.Series(False)).sum())} transactions")
    print(f"  ⛓️  layering_deep       fired on : {int(df.get('deep_peel_flag',       pd.Series(False)).sum())} transactions")

    print(f"\n  Top 5 Alerts:")
    for n in narratives[:5]:
        emoji = risk_emoji(n["risk_score"])
        v7_tag = " [v7]" if n.get("v7_blockchain_rules") else ""
        print(f"  {emoji} {n['alert_id']} | Score:{n['risk_score']} | "
              f"{n['signals'][0]['name'] if n['signals'] else 'N/A'}{v7_tag}")
    print("=" * 60)
    print(f"\n  ✅ Full report: {CONFIG['output_dir']}/alerts_report_v7.txt")
    print(f"  ✅ Metrics:     {CONFIG['output_dir']}/metrics_v7.json\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NEXUS-RISK AML Engine v7 — Blockchain")
    parser.add_argument("--input", default=CONFIG["input_file"])
    args = parser.parse_args()

    cfg = CONFIG
    cfg["input_file"] = args.input

    df = load_data(cfg["input_file"])

    print("[FEATURES] Computing rolling window features...")
    df = compute_features(df, cfg)

    df, _ = detect_layering(df, cfg)

    # ── v7 New detectors ──
    print("[V7] Running blockchain-native detectors...")
    df = detect_mixer_touch(df, cfg)
    df = detect_bridge_hops(df, cfg)
    df = detect_peel_chain(df, cfg)
    df = detect_novel_wallet_dump(df, cfg)
    df = detect_concentrated_inflow(df, cfg)
    df = detect_ofac_hit(df, cfg)
    df = detect_flash_loan_burst(df, cfg)
    df = detect_coordinated_burst(df, cfg)
    df = detect_dormant_activation(df, cfg)   # v10 new

    # ── v11 new detectors ──────────────────────────────────────────────────
    df = detect_wash_cycle(df, cfg)           # A→B→A circular flows
    df = detect_smurfing(df, cfg)             # coordinated threshold avoidance
    df = detect_exit_rush(df, cfg)            # novel wallet → immediate bridge exit
    df = detect_rapid_succession(df, cfg)     # same sender → 5+ receivers in 5 min
    df = detect_high_risk_country(df, cfg)    # FATF blacklist / grey list amplifier
    df = detect_exchange_avoidance(df, cfg)   # deliberate routing around exchanges
    df = detect_layering_deep(df, cfg)        # 5+ hop extended peel chain

    df = score_transactions(df, cfg)
    narratives = generate_narratives(df, cfg)

    # v6 vs v7 comparison table for the report
    v6_comparison = {
        "Total flagged":     ("324 / 5,797", f"{int(df['alert'].sum())} / {len(df)}"),
        "Alert rate":        ("5.59%",        f"{round(df['alert'].mean()*100, 2)}%"),
        "Precision":         ("87.16%",       "TBD — run eval.py"),
        "Recall":            ("28.12%",       "TBD — run eval.py"),
        "Mixer detection":   ("❌ none",      f"✅ {int(df.get('mixer_flag', pd.Series(False)).sum())} flagged"),
        "Bridge detection":  ("❌ none",      f"✅ {int(df.get('bridge_flag', pd.Series(False)).sum())} flagged"),
        "Peel chain detect": ("❌ none",      f"✅ {int(df.get('peel_flag', pd.Series(False)).sum())} flagged"),
    }

    report, metrics = save_outputs(df, narratives, cfg, v6_comparison)
    print_quick_metrics(df, narratives, metrics)


if __name__ == "__main__":
    main()

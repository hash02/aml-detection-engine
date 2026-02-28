"""
Generate synthetic but forensically accurate transactions for:
  - Case 8: Wormhole Bridge Exploit (Feb 2, 2022, ~$320M / 120,000 ETH)
  - Case 9: Nomad Bridge Crowd-Looting (Aug 1, 2022, ~$190M, 300+ wallets)

These are modeled on the REAL attack signatures from public post-mortems:
  Wormhole: Certik, Jump Crypto disclosure, Etherscan labels
  Nomad: Coinbase Security, Paradigm research, Immunefi post-mortem

Attack Pattern Encodings:
  Wormhole = Group 5 (Bridge Exploit): single large event, new wallet,
             cross-chain bridge flag, concentrated_inflow in seconds
  Nomad    = Group 4/5 hybrid (Crowd Looting): 300+ wallets draining
             same contract, fan_in extreme, coordinated_burst
"""

import csv
import hashlib
import random
import os
from datetime import datetime, timezone, timedelta

random.seed(42)

def fake_tx_hash(seed: str) -> str:
    return "0x" + hashlib.sha256(seed.encode()).hexdigest()[:62]

def fake_addr(seed: str) -> str:
    return "0x" + hashlib.sha256(("addr_" + seed).encode()).hexdigest()[:40]

ETH_PRICE = 3_200.0  # Feb/Aug 2022 approximate ETH price

rows = []

# ─────────────────────────────────────────────────────────────────────────────
# CASE 8: WORMHOLE BRIDGE EXPLOIT — February 2, 2022
# ─────────────────────────────────────────────────────────────────────────────
# Facts:
#   - Attacker exploited signature verification bug in Wormhole smart contract
#   - Minted 120,000 wETH on Solana with no backing collateral
#   - Claimed 120,000 real ETH from Wormhole's ETH escrow on Ethereum
#   - All ETH landed in a single newly-created wallet
#   - Funds partially moved through Oasis Protocol 1 year later (court-ordered)
#   - Jump Crypto replenished the missing ETH within 24 hours
#
# Signature: large_amount + novel_wallet_dump + concentrated_inflow
#            bridge interaction flag, contract interaction, new wallet

WORMHOLE_EXPLOITER = "0x629e7Da20197a5429d30da36E77d06CdF796b71A"
WORMHOLE_CONTRACT  = "0x3ee18B2214AFF97000D974cf647E7C347E8fa585"  # Wormhole ETH escrow

# Base timestamp: Feb 2, 2022, ~18:24 UTC (block 14173128)
t0 = datetime(2022, 2, 2, 18, 24, 0, tzinfo=timezone.utc)

# Primary exploit transaction: contract drains 120,000 ETH to exploiter
# This is the kill shot — one massive transaction
exploit_amount_eth = 120_000.0
exploit_amount_usd = exploit_amount_eth * ETH_PRICE
rows.append({
    "tx_hash":            fake_tx_hash("wormhole_main_exploit"),
    "block_number":       14173128,
    "timestamp":          t0.strftime("%Y-%m-%d %H:%M:%S"),
    "sender_id":          WORMHOLE_CONTRACT,
    "receiver_id":        WORMHOLE_EXPLOITER,
    "amount":             round(exploit_amount_usd, 2),
    "value_eth":          exploit_amount_eth,
    "country":            "UNKNOWN",
    "sender_profile":     "CONTRACT",
    "is_known_mixer":     False,
    "mixer_type":         "",
    "is_bridge":          True,
    "contract_interaction": True,
    "gas_price_gwei":     150.0,
    "gas_used":           210000,
    "token":              "ETH",
    "chain":              "ethereum",
    "target_label":       "wormhole_exploiter",
    "sender_tx_count":    1,      # new wallet, never seen before
    "sender_avg_amount":  exploit_amount_usd,
    "sender_active_days": 0,      # wallet created same block
    "account_age_days":   0,
    "account_type":       "NEW",
    "label":              "SYNTH_FORENSIC",
})

# Secondary: exploiter quickly splits into 6 sub-wallets (documented pattern)
# Funds fanned out within 2 hours of the exploit
for i in range(6):
    sub_wallet = fake_addr(f"wormhole_sub_{i}")
    sub_amount_eth = random.uniform(15_000, 25_000)
    sub_ts = t0 + timedelta(minutes=random.randint(30, 120))
    rows.append({
        "tx_hash":            fake_tx_hash(f"wormhole_fanout_{i}"),
        "block_number":       14173128 + random.randint(10, 200),
        "timestamp":          sub_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "sender_id":          WORMHOLE_EXPLOITER,
        "receiver_id":        sub_wallet,
        "amount":             round(sub_amount_eth * ETH_PRICE, 2),
        "value_eth":          round(sub_amount_eth, 2),
        "country":            "UNKNOWN",
        "sender_profile":     "UNKNOWN",
        "is_known_mixer":     False,
        "mixer_type":         "",
        "is_bridge":          False,
        "contract_interaction": False,
        "gas_price_gwei":     120.0,
        "gas_used":           21000,
        "token":              "ETH",
        "chain":              "ethereum",
        "target_label":       "wormhole_exploiter",
        "sender_tx_count":    i + 1,
        "sender_avg_amount":  round(sub_amount_eth * ETH_PRICE, 2),
        "sender_active_days": 0,
        "account_age_days":   0,
        "account_type":       "NEW",
        "label":              "SYNTH_FORENSIC",
    })

# Tertiary: Oasis move ~1 year later (documented: Jan 24, 2023, court order)
# This demonstrates the dormant-activation precursor pattern
oasis_ts = datetime(2023, 1, 24, 10, 0, 0, tzinfo=timezone.utc)
for i in range(3):
    rows.append({
        "tx_hash":            fake_tx_hash(f"wormhole_oasis_{i}"),
        "block_number":       16479100 + i,
        "timestamp":          (oasis_ts + timedelta(minutes=i*5)).strftime("%Y-%m-%d %H:%M:%S"),
        "sender_id":          WORMHOLE_EXPLOITER,
        "receiver_id":        fake_addr(f"oasis_protocol_{i}"),
        "amount":             round(random.uniform(10_000_000, 30_000_000), 2),
        "value_eth":          round(random.uniform(3000, 9000), 2),
        "country":            "UNKNOWN",
        "sender_profile":     "UNKNOWN",
        "is_known_mixer":     False,
        "mixer_type":         "",
        "is_bridge":          True,
        "contract_interaction": True,
        "gas_price_gwei":     18.0,
        "gas_used":           150000,
        "token":              "ETH",
        "chain":              "ethereum",
        "target_label":       "wormhole_exploiter",
        "sender_tx_count":    i + 8,   # 7 previous txns + new ones
        "sender_avg_amount":  round(exploit_amount_usd / 9, 2),
        "sender_active_days": 356,     # ~1 year dormant, then active
        "account_age_days":   356,
        "account_type":       "DORMANT_REACTIVATED",
        "label":              "SYNTH_FORENSIC",
    })

print(f"  Wormhole: {sum(1 for r in rows if r['target_label']=='wormhole_exploiter')} transactions")

wormhole_count = len(rows)

# ─────────────────────────────────────────────────────────────────────────────
# CASE 9: NOMAD BRIDGE CROWD-LOOTING — August 1, 2022
# ─────────────────────────────────────────────────────────────────────────────
# Facts:
#   - A single legitimate transaction revealed an exploit in Nomad's message
#     verification. Anyone could copy-paste the transaction and change the
#     recipient address to drain the bridge.
#   - Within 3 hours, 300+ unique wallets had exploited this.
#   - Total drained: ~$190M across ETH, USDC, WBTC, DAI
#   - Pattern: extreme fan_in (300+ senders to 1 contract), coordinated_burst
#     (all hitting the same contract within minutes of each other)
#   - Classic "opportunistic crowd looting" — not one attacker, but a mob
#
# Signature: fan_in_extreme + coordinated_burst + velocity (300 wallets, 3hrs)

NOMAD_BRIDGE_CONTRACT = "0x88A69B4E698A4B090DF6CF5Bd7B2D47325Ad30A3"  # Nomad bridge

# 3-hour window, Aug 1 2022 ~21:32 UTC (block 15259101)
t_nomad = datetime(2022, 8, 1, 21, 32, 0, tzinfo=timezone.utc)

# Generate 80 exploiter wallets (compressed from 300+ for dataset size)
# Each drains $100k–$2M in 1-3 transactions
nomad_wallets = [fake_addr(f"nomad_lootter_{i}") for i in range(80)]

for w_idx, wallet in enumerate(nomad_wallets):
    n_txns = random.randint(1, 3)
    # Wallets exploited within the 3-hour window — bot-like speed
    wallet_start = t_nomad + timedelta(minutes=random.randint(0, 180))
    
    for t_idx in range(n_txns):
        drain_amount_usd = random.uniform(100_000, 2_000_000)
        drain_amount_eth = drain_amount_usd / ETH_PRICE
        tx_ts = wallet_start + timedelta(seconds=random.randint(0, 120))
        
        rows.append({
            "tx_hash":            fake_tx_hash(f"nomad_{w_idx}_{t_idx}"),
            "block_number":       15259101 + random.randint(0, 800),
            "timestamp":          tx_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "sender_id":          wallet,
            "receiver_id":        NOMAD_BRIDGE_CONTRACT,
            "amount":             round(drain_amount_usd, 2),
            "value_eth":          round(drain_amount_eth, 4),
            "country":            random.choice(["UNKNOWN", "US", "RU", "CN", "XX"]),
            "sender_profile":     "UNKNOWN",
            "is_known_mixer":     False,
            "mixer_type":         "",
            "is_bridge":          True,
            "contract_interaction": True,
            "gas_price_gwei":     round(random.uniform(10, 50), 2),
            "gas_used":           random.randint(80000, 200000),
            "token":              random.choice(["ETH", "USDC", "WBTC", "DAI"]),
            "chain":              "ethereum",
            "target_label":       "nomad_crowd_exploiter",
            "sender_tx_count":    t_idx + 1,   # most are first-time users of this
            "sender_avg_amount":  round(drain_amount_usd, 2),
            "sender_active_days": random.randint(0, 5),  # many fresh wallets
            "account_age_days":   random.randint(0, 30),
            "account_type":       "NEW" if random.random() < 0.6 else "UNKNOWN",
            "label":              "SYNTH_FORENSIC",
        })

# A few "white-hat" rescuers also exploited and returned funds (documented)
# These show up as low-score transactions (should NOT be flagged as critical)
for i in range(5):
    rescue_wallet = fake_addr(f"nomad_whitehat_{i}")
    rescue_ts = t_nomad + timedelta(minutes=random.randint(30, 120))
    rows.append({
        "tx_hash":            fake_tx_hash(f"nomad_rescue_{i}"),
        "block_number":       15259101 + random.randint(100, 500),
        "timestamp":          rescue_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "sender_id":          rescue_wallet,
        "receiver_id":        NOMAD_BRIDGE_CONTRACT,
        "amount":             round(random.uniform(50_000, 500_000), 2),
        "value_eth":          round(random.uniform(15, 150), 4),
        "country":            "US",
        "sender_profile":     "BUSINESS_LIKE",
        "is_known_mixer":     False,
        "mixer_type":         "",
        "is_bridge":          True,
        "contract_interaction": True,
        "gas_price_gwei":     25.0,
        "gas_used":           150000,
        "token":              "ETH",
        "chain":              "ethereum",
        "target_label":       "nomad_crowd_exploiter",
        "sender_tx_count":    random.randint(50, 200),  # established wallets
        "sender_avg_amount":  round(random.uniform(1000, 10000), 2),
        "sender_active_days": random.randint(180, 700),
        "account_age_days":   random.randint(180, 700),
        "account_type":       "BUSINESS_LIKE",
        "label":              "SYNTH_FORENSIC",
    })

nomad_count = len(rows) - wormhole_count
print(f"  Nomad: {nomad_count} transactions")

# ─────────────────────────────────────────────────────────────────────────────
# WRITE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), "data", "new_cases.csv")
fieldnames = list(rows[0].keys())

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n  Total new case rows: {len(rows)}")
print(f"  Saved → {out_path}")

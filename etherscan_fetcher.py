"""
etherscan_fetcher.py — Live Blockchain Data Feed for AML Engine v11
====================================================================
Fetches real Ethereum transactions from Etherscan, enriches them,
and produces a CSV that feeds directly into engine_v11_blockchain.py.

The engine was built on synthetic data. This makes it real.

HOW IT WORKS
────────────
1. Start from seed addresses (known high-risk wallets, exchange hubs,
   or any address you want to investigate)
2. Fetch their transaction history via Etherscan API (free tier works)
3. Build the full transaction graph: for each tx, we have sender + receiver
4. Enrich each row:
   - Convert ETH (wei) → USD using live price from CoinGecko
   - Flag mixers and bridge contracts using known address lists
   - Infer sender_profile (EXCHANGE, CONTRACT, UNKNOWN, HIGH_RISK)
   - Compute per-sender stats (tx_count, avg_amount, active_days)
5. Output: data/live_transactions.csv — ready for the engine

USAGE
─────
  # Basic: uses default seed addresses (known exploit/hack wallets)
  python etherscan_fetcher.py

  # Custom seed address:
  python etherscan_fetcher.py --address 0xABC123... --limit 200

  # With your Etherscan API key (5 req/sec vs 1 req/5sec without):
  python etherscan_fetcher.py --api-key YOUR_KEY_HERE

  # Save to custom path:
  python etherscan_fetcher.py --output data/custom_run.csv

GET FREE API KEY
───────────────
  https://etherscan.io/register → free → My Profile → API Keys
  Free tier: 5 calls/sec, 100k calls/day — more than enough.

SEED ADDRESSES (default)
────────────────────────
  These are publicly documented high-risk addresses used in research,
  academic papers, and blockchain forensics. All are well-known in the
  AML/blockchain analytics community.

  - Ronin Bridge Exploit (Axie/Lazarus Group): 0x098B...
  - Bitfinex Hack 2016 wallets (DOJ-indicted, publicly released)
  - Known Tornado Cash interaction cluster
  - FTX exchange wallets (Chapter 11 filing, publicly documented)
"""

import argparse
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN ADDRESS LISTS
# All addresses below are publicly documented in academic papers, court filings,
# or blockchain analytics reports. None are private or sensitive information.
# ─────────────────────────────────────────────────────────────────────────────

# Tornado Cash core contracts (publicly documented, OFAC-sanctioned)
KNOWN_MIXERS = {
    "0x12d66f87a04a9e220c9d35925b72aca3ca8c78e2",  # TC 0.1 ETH
    "0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936",  # TC 1 ETH
    "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf",  # TC 10 ETH
    "0xa160cdab225685da1d56aa342ad8841c3b53f291",  # TC 100 ETH
    "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",  # TC Router
    "0x722122df12d4e14e13ac3b6895a86e84145b6967",  # TC 0.1 ETH (v2)
    "0x2717c5e28cf931547b621a5dddb772ab6a35b701",  # Blender.io (OFAC)
    "0x7f367cc41522ce07553e823bf3be79a889debe1b",  # Lazarus-linked mixer
}

# Bridge contracts: legitimate but high-risk for layering (Hop, Across, Stargate)
KNOWN_BRIDGES = {
    "0x3666f603cc164936c1b87e207f36beba4ac5f18d",  # Hop: USDC Bridge
    "0x3e4a3a4796d16c0cd582c382691998f7c06420b6",  # Hop: USDT Bridge
    "0xb8901acb165ed027e32754e0ffe830802919727f",  # Hop: ETH Bridge
    "0x4d9079bb4165aeb4084c526a32695dcfd2f77381",  # Across Protocol
    "0x8731d54e9d02c286767d56ac03e8037c07e01e98",  # Stargate Router
    "0x66a71dcef29a0ffbdbe3c6a460a3b5bc225cd675",  # LayerZero Endpoint
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f",  # SushiSwap (cross-chain)
    "0x40ec5b33f54e0e8a33a975908c5ba1c14e5bbbdf",  # Polygon Bridge
    "0x2796317b0ff8538f253012862c06787adfb8ceb6",  # Synapse Bridge
}

# Known exchange hot wallets (for profile classification)
KNOWN_EXCHANGES = {
    "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": "Binance",
    "0xd551234ae421e3bcba99a0da6d736074f22192ff": "Binance",
    "0x564286362092d8e7936f0549571a803b203aaced": "Binance",
    "0x0681d8db095565fe8a346fa0277bffde9c0edbbf": "Binance",
    "0xfe9e8709d3215310075d67e3ed32a380ccf451c8": "Binance",
    "0x28c6c06298d514db089934071355e5743bf21d60": "Binance",
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": "Binance",
    "0x3cfd6b39b4b27a14c8c0acf2b0f2e8cbb81bf67d": "Coinbase",
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": "Coinbase",
    "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43": "Coinbase",
    "0x1b3cb81e51011b549d78bf720b0d924ac763a7c2": "Kraken",
    "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": "Kraken",
    "0x2b5634c42055806a59e9107ed44d43c426e58258": "KuCoin",
}

# Seed addresses: publicly documented high-risk wallets used in research
# Each has a note explaining the public record
DEFAULT_SEEDS = [
    # Ronin Bridge Exploiter — Lazarus Group (North Korea)
    # Source: U.S. Treasury OFAC designation, Chainalysis public report
    "0x098b716b8aaf21512996dc57eb0615e2383e2f96",

    # FTX Exchange — Chapter 11 bankruptcy estate, publicly filed
    # Used as seed because of the massive fan-in pattern during collapse
    "0xa14c04dea16798aa8f25b1da583cd5fbbfba5c3d",

    # Bitfinex Hack 2016 — DOJ-indicted wallets (publicly released court docs)
    # The original stolen funds aggregation address
    "0x3dfd23a6c5e8bbcfc9581d2e864a68feb6a076d3",

    # Known structuring cluster (Chainalysis 2022 public report on DeFi laundering)
    # High velocity, multi-hop, bridges to BNB chain
    "0x4f6742badb049791cd9a37ea913f2bac38d01279",
]

# Country mapping for known exchange clusters
# Based on where exchanges are HQ'd or primarily regulated
EXCHANGE_COUNTRY = {
    "Binance": "KY",    # Cayman Islands entity (publicly known)
    "Coinbase": "US",
    "Kraken": "US",
    "KuCoin": "SC",     # Seychelles
}

# High-risk country codes for the engine's foreign_country rule
HIGH_RISK_COUNTRIES = {"KP", "IR", "SY", "MM", "CU", "SD", "BY"}


# ─────────────────────────────────────────────────────────────────────────────
# ETH PRICE FETCHER
# ─────────────────────────────────────────────────────────────────────────────

def get_eth_price_usd() -> float:
    """Get current ETH/USD price from CoinGecko (free, no API key needed)."""
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "ethereum", "vs_currencies": "usd"},
            timeout=10
        )
        price = resp.json()["ethereum"]["usd"]
        print(f"[PRICE] ETH/USD: ${price:,.2f}")
        return float(price)
    except Exception as e:
        print(f"[PRICE] CoinGecko failed ({e}), using fallback $2,500")
        return 2500.0


# ─────────────────────────────────────────────────────────────────────────────
# ETHERSCAN API
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# BLOCKCHAIR API  (primary — free, no key required)
# Falls back to Etherscan V2 if a key is provided.
# ─────────────────────────────────────────────────────────────────────────────

BLOCKCHAIR_URL = "https://api.blockchair.com/ethereum/transactions"
ETHERSCAN_V2_URL = "https://api.etherscan.io/v2/api"


def fetch_blockchair(address: str, limit: int = 300) -> list:
    """
    Fetch transactions for an address using Blockchair API.
    No API key needed. Free tier: ~30 req/min.
    Returns list of raw Blockchair transaction dicts.
    """
    rows = []
    for role in ["sender", "recipient"]:
        offset = 0
        while offset < limit:
            try:
                resp = requests.get(
                    BLOCKCHAIR_URL,
                    params={
                        "q": f"{role}({address})",
                        "limit": min(100, limit - offset),
                        "offset": offset,
                        "s": "time(asc)",
                    },
                    timeout=20,
                )
                data = resp.json()
                batch = data.get("data", [])
                if not batch:
                    break
                rows.extend(batch)
                offset += len(batch)
                if len(batch) < 100:
                    break
                time.sleep(0.35)   # polite to free tier
            except Exception as e:
                print(f"[FETCH] Blockchair error ({role}, {address[:10]}...): {e}")
                break
    print(f"[FETCH] {address[:10]}... → {len(rows)} transactions (Blockchair)")
    return rows


def fetch_etherscan_v2(address: str, api_key: str, limit: int = 500) -> list:
    """
    Fetch transactions using Etherscan V2 API (requires free API key).
    Register at etherscan.io/register — free, takes 30 seconds.
    """
    try:
        resp = requests.get(
            ETHERSCAN_V2_URL,
            params={
                "chainid": 1,
                "module": "account",
                "action": "txlist",
                "address": address,
                "page": 1,
                "offset": min(limit, 10000),
                "sort": "asc",
                "apikey": api_key,
            },
            timeout=15,
        )
        data = resp.json()
        if data["status"] == "1":
            print(f"[FETCH] {address[:10]}... → {len(data['result'])} transactions (Etherscan V2)")
            return data["result"]
        else:
            print(f"[FETCH] Etherscan V2 error: {data.get('message', 'unknown')} — falling back to Blockchair")
            return fetch_blockchair(address, limit)
    except Exception as e:
        print(f"[FETCH] Etherscan V2 failed: {e} — falling back to Blockchair")
        return fetch_blockchair(address, limit)


def fetch_transactions(address: str, api_key: str = "", limit: int = 300) -> list:
    """
    Main fetch function. Uses Blockchair by default (no key needed).
    If an Etherscan V2 API key is provided, uses that instead.
    """
    if api_key:
        return fetch_etherscan_v2(address, api_key, limit)
    return fetch_blockchair(address, limit)


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE CLASSIFIER
# Infers sender_profile from what we know about the address.
# This mirrors what the engine's PROFILE_CONFIG calibrates thresholds against.
# ─────────────────────────────────────────────────────────────────────────────

def classify_profile(address: str, is_contract: bool) -> str:
    addr_lower = address.lower()
    if addr_lower in KNOWN_MIXERS:
        return "HIGH_RISK"
    if addr_lower in KNOWN_EXCHANGES:
        return "EXCHANGE"
    if is_contract:
        return "CONTRACT"
    return "UNKNOWN"


def infer_country(address: str) -> str:
    addr_lower = address.lower()
    if addr_lower in KNOWN_MIXERS:
        return "XX"        # Unknown/high-risk — engine treats XX as suspicious
    if addr_lower in KNOWN_BRIDGES:
        return "BRIDGE"    # Not a real country but engine won't penalize
    if addr_lower in KNOWN_EXCHANGES:
        exchange_name = KNOWN_EXCHANGES[addr_lower]
        return EXCHANGE_COUNTRY.get(exchange_name, "UNKNOWN")
    return "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# TRANSACTION BUILDER
# Maps raw Etherscan API response → engine schema
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_tx(tx: dict) -> dict:
    """
    Normalize a raw transaction dict to a common internal format,
    handling both Blockchair and Etherscan V2 field names.

    Blockchair fields: sender, recipient, value (wei str), time (datetime str), hash
    Etherscan V2 fields: from, to, value (wei str), timeStamp (unix str), hash, isError, contractAddress
    """
    # Detect source by field presence
    if "sender" in tx:
        # Blockchair format
        return {
            "hash":            tx.get("hash", ""),
            "from":            tx.get("sender", "").lower(),
            "to":              tx.get("recipient", "").lower(),
            "value":           str(tx.get("value", "0")),
            "timestamp_str":   tx.get("time", ""),
            "timestamp_unix":  None,
            "is_error":        tx.get("failed", False),
            "is_contract":     tx.get("type", "") == "call" and tx.get("recipient", "") == "",
        }
    else:
        # Etherscan V2 format
        return {
            "hash":            tx.get("hash", ""),
            "from":            tx.get("from", "").lower(),
            "to":              tx.get("to", "").lower(),
            "value":           str(tx.get("value", "0")),
            "timestamp_str":   None,
            "timestamp_unix":  tx.get("timeStamp"),
            "is_error":        tx.get("isError") == "1",
            "is_contract":     bool(tx.get("contractAddress")),
        }


def build_transaction_rows(raw_txns: list, eth_price: float, source_address: str) -> list:
    """
    Convert raw transaction list (Blockchair or Etherscan V2) into engine-ready rows.
    Filters out failed transactions and zero-value txns (unless mixer-involved).
    """
    rows = []
    for tx in raw_txns:
        n = _normalize_tx(tx)

        if n["is_error"]:
            continue

        # Convert value (wei string) → USD
        try:
            value_wei = int(n["value"])
        except (ValueError, TypeError):
            value_wei = 0

        sender   = n["from"]
        receiver = n["to"]

        if value_wei == 0:
            # Keep zero-value only if mixer-involved
            if sender not in KNOWN_MIXERS and receiver not in KNOWN_MIXERS:
                continue

        if not sender or not receiver:
            continue

        amount_usd = (value_wei / 1e18) * eth_price

        # Parse timestamp
        try:
            if n["timestamp_str"]:
                ts = datetime.strptime(n["timestamp_str"], "%Y-%m-%d %H:%M:%S")
            elif n["timestamp_unix"]:
                ts = datetime.utcfromtimestamp(int(n["timestamp_unix"]))
            else:
                continue
        except (ValueError, TypeError):
            continue

        is_mixer  = (sender in KNOWN_MIXERS or receiver in KNOWN_MIXERS)
        is_bridge = (sender in KNOWN_BRIDGES or receiver in KNOWN_BRIDGES)

        rows.append({
            "id":             n["hash"],
            "sender_id":      sender,
            "receiver_id":    receiver,
            "amount":         round(amount_usd, 2),
            "country":        infer_country(sender),
            "timestamp":      ts,
            "label":          "LIVE",
            "sender_profile": classify_profile(sender, n["is_contract"]),
            "is_known_mixer": is_mixer,
            "is_bridge":      is_bridge,
            "sender_tx_count":    0,
            "sender_avg_amount":  0.0,
            "sender_active_days": 0.0,
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# SENDER STATS ENRICHMENT
# Computes per-sender stats across the FULL fetched dataset.
# This is what the engine uses for novel_wallet_dump, dormant_activation, etc.
# ─────────────────────────────────────────────────────────────────────────────

def enrich_sender_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each transaction, compute:
    - sender_tx_count: how many txns this sender has in the dataset
    - sender_avg_amount: average tx amount for this sender
    - sender_active_days: span of days from first to last tx for this sender
    """
    df = df.copy()
    stats = {}

    for sender, group in df.groupby("sender_id"):
        tx_count  = len(group)
        avg_amt   = group["amount"].mean()
        first_ts  = group["timestamp"].min()
        last_ts   = group["timestamp"].max()
        active_days = max(1, (last_ts - first_ts).days)
        stats[sender] = {
            "sender_tx_count":    tx_count,
            "sender_avg_amount":  round(avg_amt, 2),
            "sender_active_days": float(active_days),
        }

    df["sender_tx_count"]    = df["sender_id"].map(lambda s: stats.get(s, {}).get("sender_tx_count", 1))
    df["sender_avg_amount"]  = df["sender_id"].map(lambda s: stats.get(s, {}).get("sender_avg_amount", 0.0))
    df["sender_active_days"] = df["sender_id"].map(lambda s: stats.get(s, {}).get("sender_active_days", 1.0))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH EXPANSION
# Given seed transactions, collect the next-hop addresses so the engine
# can detect peel_chain and layering patterns that span multiple wallets.
# ─────────────────────────────────────────────────────────────────────────────

def expand_graph(seed_txns: list, api_key: str, hop_limit: int = 1,
                 per_address_limit: int = 100, delay: float = 0.25) -> list:
    """
    From seed transactions, collect unique receiver addresses and
    fetch THEIR transactions too (one hop deeper).
    This turns a single wallet's history into a transaction graph —
    which is what the layering/peel_chain detectors need.
    """
    all_txns = list(seed_txns)
    seen_addresses = set()

    # Collect unique receivers from seed txns
    receivers = set()
    for tx in seed_txns:
        receivers.add(tx.get("to", "").lower())
        seen_addresses.add(tx.get("from", "").lower())

    # Fetch one hop deeper
    for addr in list(receivers)[:20]:   # cap at 20 to avoid rate limits
        if addr in seen_addresses or not addr:
            continue
        seen_addresses.add(addr)
        time.sleep(delay)   # be polite to Etherscan free tier
        hop_txns = fetch_transactions(addr, api_key, limit=per_address_limit)
        all_txns.extend(hop_txns)

    print(f"[GRAPH] Total transactions after graph expansion: {len(all_txns)}")
    return all_txns


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate(rows: list) -> list:
    """Remove duplicate transactions by hash."""
    seen = set()
    unique = []
    for row in rows:
        tx_id = row.get("id", "")
        if tx_id and tx_id not in seen:
            seen.add(tx_id)
            unique.append(row)
        elif not tx_id:
            unique.append(row)   # keep if no hash (internal txns sometimes lack it)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def fetch_live_data(
    addresses: list,
    api_key: str = "",
    limit: int = 500,
    expand_hops: bool = True,
    output_path: str = "data/live_transactions.csv"
) -> pd.DataFrame:
    """
    Full pipeline: addresses → raw txns → enriched DataFrame → CSV.
    Returns the DataFrame so the Streamlit app can use it directly.
    """
    print(f"\n{'='*60}")
    print(f"  AML Engine v11 — Live Blockchain Feed")
    print(f"  Addresses: {len(addresses)} | Limit per address: {limit}")
    print(f"{'='*60}\n")

    # Step 1: Get current ETH price
    eth_price = get_eth_price_usd()

    # Step 2: Fetch raw transactions for all seed addresses
    all_raw = []
    for addr in addresses:
        raw = fetch_transactions(addr.lower(), api_key, limit=limit)
        all_raw.extend(raw)
        if not api_key:
            time.sleep(0.3)   # free tier rate limit buffer

    if not all_raw:
        print("[ERROR] No transactions fetched. Check addresses and API key.")
        return pd.DataFrame()

    # Step 3: Optionally expand one hop deeper for graph detection
    if expand_hops and len(all_raw) < 2000:
        print(f"\n[GRAPH] Expanding one hop from {len(all_raw)} seed transactions...")
        all_raw = expand_graph(all_raw, api_key)

    # Step 4: Deduplicate
    all_raw = deduplicate(all_raw)
    print(f"[DEDUP] {len(all_raw)} unique transactions")

    # Step 5: Convert to engine schema
    all_rows = []
    for addr in addresses:
        # Build rows using the specific seed address as context
        rows = build_transaction_rows(all_raw, eth_price, addr.lower())
        all_rows.extend(rows)

    # Re-deduplicate after building rows
    df = pd.DataFrame(deduplicate(all_rows))
    if df.empty:
        print("[ERROR] No valid transactions after filtering.")
        return df

    # Step 6: Enrich sender stats across the full graph
    df = enrich_sender_stats(df)

    # Step 7: Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Step 8: Save
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"  ✅ Live data saved → {output_path}")
    print(f"  Transactions: {len(df)}")
    print(f"  Unique senders: {df['sender_id'].nunique()}")
    print(f"  Unique receivers: {df['receiver_id'].nunique()}")
    print(f"  Mixer-involved: {df['is_known_mixer'].sum()}")
    print(f"  Bridge-involved: {df['is_bridge'].sum()}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Amount range: ${df['amount'].min():,.2f} → ${df['amount'].max():,.2f}")
    print(f"{'='*60}\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch live Ethereum transactions and prepare them for AML engine v11"
    )
    parser.add_argument(
        "--address",
        type=str,
        help="Single Ethereum address to investigate (overrides default seeds)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Etherscan API key (free at etherscan.io/register). Improves rate limits."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max transactions to fetch per address (default: 500)"
    )
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Skip graph expansion (only fetch seed addresses, no hop-following)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/live_transactions.csv",
        help="Output CSV path (default: data/live_transactions.csv)"
    )
    args = parser.parse_args()

    addresses = [args.address] if args.address else DEFAULT_SEEDS

    df = fetch_live_data(
        addresses=addresses,
        api_key=args.api_key,
        limit=args.limit,
        expand_hops=not args.no_expand,
        output_path=args.output
    )

    if df.empty:
        print("[ABORT] No data to run engine on.")
        sys.exit(1)

    # Auto-run the engine on the live data
    print("[ENGINE] Running AML engine v11 on live data...\n")
    try:
        import subprocess
        result = subprocess.run(
            ["python", "engine/engine_v11_blockchain.py", "--input", args.output],
            capture_output=False
        )
        if result.returncode != 0:
            print("[ENGINE] Engine run failed. Check output above.")
    except Exception as e:
        print(f"[ENGINE] Could not auto-run engine: {e}")
        print(f"[ENGINE] Run manually: python engine/engine_v11_blockchain.py --input {args.output}")


if __name__ == "__main__":
    main()

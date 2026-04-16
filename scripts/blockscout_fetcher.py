"""
blockscout_fetcher.py — Multi-Chain Blockchain Data Feed for AML Engine v11
============================================================================
Fetches real transactions from Blockscout API (free, no key needed).
Supports Ethereum, Polygon, Optimism, Base, Arbitrum, and more.
Outputs CSV compatible with engine_v11_blockchain.py.

Usage:
  python scripts/blockscout_fetcher.py --address 0xABC123 --chain 1
  python scripts/blockscout_fetcher.py --address 0xABC123 --chain 137 --limit 500
"""

import argparse
import io
import os
import sys
import time
from datetime import datetime

import pandas as pd
import requests

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etherscan_fetcher import (
    KNOWN_BRIDGES,
    KNOWN_MIXERS,
    classify_profile,
    deduplicate,
    enrich_sender_stats,
    get_eth_price_usd,
    infer_country,
)

# Blockscout API base URLs per chain
BLOCKSCOUT_URLS = {
    1:     "https://eth.blockscout.com/api/v2",
    137:   "https://polygon.blockscout.com/api/v2",
    10:    "https://optimism.blockscout.com/api/v2",
    8453:  "https://base.blockscout.com/api/v2",
    42161: "https://arbitrum.blockscout.com/api/v2",
    100:   "https://gnosis.blockscout.com/api/v2",
    324:   "https://zksync.blockscout.com/api/v2",
}

CHAIN_NAMES = {
    1: "Ethereum", 137: "Polygon", 10: "Optimism",
    8453: "Base", 42161: "Arbitrum", 100: "Gnosis", 324: "zkSync",
}


def get_blockscout_url(chain_id: int) -> str:
    if chain_id in BLOCKSCOUT_URLS:
        return BLOCKSCOUT_URLS[chain_id]
    # Fallback: try eth.blockscout.com
    print(f"[WARN] Chain {chain_id} not in known Blockscout URLs, using Ethereum")
    return BLOCKSCOUT_URLS[1]


def fetch_blockscout_transactions(address: str, chain_id: int = 1, limit: int = 300) -> list:
    """Fetch transactions for an address from Blockscout API."""
    base_url = get_blockscout_url(chain_id)
    chain_name = CHAIN_NAMES.get(chain_id, f"Chain {chain_id}")
    url = f"{base_url}/addresses/{address}/transactions"

    all_txns = []
    next_params = None
    page = 0

    while len(all_txns) < limit:
        try:
            params = next_params or {}
            resp = requests.get(url, params=params, timeout=20)

            if resp.status_code == 404:
                print(f"[FETCH] Address not found on {chain_name}: {address[:10]}...")
                break
            if resp.status_code != 200:
                print(f"[FETCH] Blockscout {chain_name} error {resp.status_code}")
                break

            data = resp.json()
            items = data.get("items", [])
            if not items:
                break

            all_txns.extend(items)
            page += 1

            # Check for pagination
            next_page = data.get("next_page_params")
            if next_page and len(all_txns) < limit:
                next_params = next_page
                time.sleep(0.25)
            else:
                break

        except Exception as e:
            print(f"[FETCH] Blockscout error ({chain_name}): {e}")
            break

    print(f"[FETCH] {address[:10]}... → {len(all_txns)} transactions (Blockscout {chain_name})")
    return all_txns[:limit]


def fetch_blockscout_token_transfers(address: str, chain_id: int = 1, limit: int = 100) -> list:
    """Fetch token transfers for an address from Blockscout API."""
    base_url = get_blockscout_url(chain_id)
    url = f"{base_url}/addresses/{address}/token-transfers"

    all_transfers = []
    next_params = None

    while len(all_transfers) < limit:
        try:
            params = next_params or {}
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code != 200:
                break

            data = resp.json()
            items = data.get("items", [])
            if not items:
                break

            all_transfers.extend(items)

            next_page = data.get("next_page_params")
            if next_page and len(all_transfers) < limit:
                next_params = next_page
                time.sleep(0.25)
            else:
                break

        except Exception as e:
            print(f"[FETCH] Token transfer error: {e}")
            break

    print(f"[FETCH] {address[:10]}... → {len(all_transfers)} token transfers")
    return all_transfers[:limit]


def build_rows_from_blockscout(txns: list, eth_price: float) -> list:
    """Convert Blockscout API transaction objects to engine-compatible rows."""
    rows = []
    for tx in txns:
        # Skip failed transactions
        if tx.get("status") == "error" or tx.get("result") == "error":
            continue

        sender = (tx.get("from", {}) or {}).get("hash", "").lower()
        receiver = (tx.get("to", {}) or {}).get("hash", "").lower()

        if not sender or not receiver:
            continue

        # Parse value (wei string → USD)
        try:
            value_wei = int(tx.get("value", "0"))
        except (ValueError, TypeError):
            value_wei = 0

        if value_wei == 0:
            if sender not in KNOWN_MIXERS and receiver not in KNOWN_MIXERS:
                continue

        amount_usd = (value_wei / 1e18) * eth_price

        # Parse timestamp
        try:
            ts_str = tx.get("timestamp", "")
            if ts_str:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
            else:
                continue
        except (ValueError, TypeError):
            continue

        is_contract = (tx.get("to", {}) or {}).get("is_contract", False)
        is_mixer = sender in KNOWN_MIXERS or receiver in KNOWN_MIXERS
        is_bridge = sender in KNOWN_BRIDGES or receiver in KNOWN_BRIDGES

        rows.append({
            "id": tx.get("hash", ""),
            "sender_id": sender,
            "receiver_id": receiver,
            "amount": round(amount_usd, 2),
            "country": infer_country(sender),
            "timestamp": ts,
            "label": "BLOCKSCOUT",
            "sender_profile": classify_profile(sender, is_contract),
            "is_known_mixer": is_mixer,
            "is_bridge": is_bridge,
            "sender_tx_count": 0,
            "sender_avg_amount": 0.0,
            "sender_active_days": 0.0,
        })

    return rows


def fetch_and_build(
    address: str,
    chain_id: int = 1,
    limit: int = 300,
    output_path: str = "data/blockscout_transactions.csv",
) -> pd.DataFrame:
    """Full pipeline: address → Blockscout API → enriched DataFrame → CSV."""
    chain_name = CHAIN_NAMES.get(chain_id, f"Chain {chain_id}")
    print(f"\n{'='*60}")
    print(f"  AML Engine v11 — Blockscout Feed ({chain_name})")
    print(f"  Address: {address}")
    print(f"{'='*60}\n")

    eth_price = get_eth_price_usd()

    raw_txns = fetch_blockscout_transactions(address, chain_id, limit)
    if not raw_txns:
        print("[ERROR] No transactions fetched.")
        return pd.DataFrame()

    rows = build_rows_from_blockscout(raw_txns, eth_price)
    rows = deduplicate(rows)

    if not rows:
        print("[ERROR] No valid transactions after processing.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = enrich_sender_stats(df)
    df = df.sort_values("timestamp").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"  Saved → {output_path}")
    print(f"  Transactions: {len(df)}")
    print(f"  Unique senders: {df['sender_id'].nunique()}")
    print(f"  Unique receivers: {df['receiver_id'].nunique()}")
    if len(df) > 0:
        print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
        print(f"  Amount range: ${df['amount'].min():,.2f} → ${df['amount'].max():,.2f}")
    print(f"{'='*60}\n")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Fetch transactions from Blockscout API for AML engine v11"
    )
    parser.add_argument("--address", type=str, required=True, help="Blockchain address to investigate")
    parser.add_argument("--chain", type=int, default=1, help=f"Chain ID (default: 1). Supported: {list(CHAIN_NAMES.keys())}")
    parser.add_argument("--limit", type=int, default=300, help="Max transactions to fetch (default: 300)")
    parser.add_argument("--output", type=str, default="data/blockscout_transactions.csv", help="Output CSV path")
    args = parser.parse_args()

    df = fetch_and_build(
        address=args.address,
        chain_id=args.chain,
        limit=args.limit,
        output_path=args.output,
    )

    if df.empty:
        sys.exit(1)

    # Auto-run engine
    print("[ENGINE] Running AML engine v11 on Blockscout data...\n")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "engine/engine_v11_blockchain.py", "--input", args.output],
            capture_output=False,
        )
    except Exception as e:
        print(f"[ENGINE] Could not auto-run: {e}")
        print(f"[ENGINE] Run manually: python engine/engine_v11_blockchain.py --input {args.output}")


if __name__ == "__main__":
    main()

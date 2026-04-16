#!/usr/bin/env python3
"""
AML Live Inference Pipeline
============================
Fetches live blockchain transactions → runs AML engine v11 →
generates risk report → sends summary via Telegram.

All free: Blockscout API (no key), Ollama (local), Telegram bot.

Usage:
  python3 pipeline.py --address 0xABC123
  python3 pipeline.py --address 0xABC123 --chain 137 --limit 100
  python3 pipeline.py --watchlist watchlist.txt
"""

import argparse
import json
import os
import sys
import subprocess
import requests
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.blockscout_fetcher import fetch_and_build
from engine.engine_v11_blockchain import (
    load_data, compute_features, detect_layering,
    detect_mixer_touch, detect_bridge_hops, detect_peel_chain,
    detect_novel_wallet_dump, detect_concentrated_inflow,
    detect_ofac_hit, detect_flash_loan_burst, detect_coordinated_burst,
    detect_dormant_activation, detect_wash_cycle, detect_smurfing,
    detect_exit_rush, detect_rapid_succession, detect_high_risk_country,
    detect_exchange_avoidance, detect_layering_deep,
    score_transactions, risk_level, generate_narratives,
    format_text_report, CONFIG
)

# Telegram config — opt-in. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
# in the environment to enable on-alert notifications. When either is
# unset, send_telegram() is a no-op.
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_telegram(message: str):
    """Send message via Telegram bot (no-op if not configured)."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    # Telegram limit is 4096 chars
    if len(message) > 4000:
        message = message[:3997] + "..."
    try:
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }, timeout=10)
    except Exception as e:
        print(f"[TELEGRAM] Failed: {e}")


def ollama_summarize(report_text: str) -> str:
    """Use local Ollama to generate a human-readable summary."""
    prompt = f"""You are an AML analyst. Summarize this risk report in 5-7 bullet points.
Focus on: highest risk addresses, which rules fired, and recommended actions.
Be concise and direct.

REPORT:
{report_text[:3000]}

SUMMARY:"""

    try:
        result = subprocess.run(
            ["ollama", "run", "nemotron-3-nano:4b", prompt],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout.strip() if result.stdout else "Summary generation failed."
    except Exception as e:
        return f"Ollama unavailable: {e}"


def run_pipeline(address: str, chain_id: int = 1, limit: int = 200, notify: bool = True):
    """Full pipeline: fetch → analyze → report → notify."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"pipeline_runs/{timestamp}_{address[:10]}"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Fetch live transactions and build enriched CSV
    print(f"\n[1/5] Fetching transactions for {address} on chain {chain_id}...")
    csv_path = f"{output_dir}/transactions.csv"

    try:
        df = fetch_and_build(
            address=address,
            chain_id=chain_id,
            limit=limit,
            output_path=csv_path
        )
    except Exception as e:
        error_msg = f"[PIPELINE FAILED] Fetch error for {address}: {e}"
        print(error_msg)
        if notify:
            send_telegram(f"🔴 {error_msg}")
        return None

    # Step 2: Run engine detectors
    print("[2/5] Running AML engine v11...")
    try:
        cfg = CONFIG.copy()
        cfg["input_file"] = csv_path
        cfg["output_dir"] = output_dir
        if df is None or df.empty or len(df) == 0:
            msg = f"[PIPELINE] No transactions found for {address}"
            print(msg)
            if notify:
                send_telegram(msg)
            return None

        df = compute_features(df, cfg)
        df, _cycles = detect_layering(df, cfg)
        df = detect_mixer_touch(df, cfg)
        df = detect_bridge_hops(df, cfg)
        df = detect_peel_chain(df, cfg)
        df = detect_novel_wallet_dump(df, cfg)
        df = detect_concentrated_inflow(df, cfg)
        df = detect_ofac_hit(df, cfg)
        df = detect_flash_loan_burst(df, cfg)
        df = detect_coordinated_burst(df, cfg)
        df = detect_dormant_activation(df, cfg)
        df = detect_wash_cycle(df, cfg)
        df = detect_smurfing(df, cfg)
        df = detect_exit_rush(df, cfg)
        df = detect_rapid_succession(df, cfg)
        df = detect_high_risk_country(df, cfg)
        df = detect_exchange_avoidance(df, cfg)
        df = detect_layering_deep(df, cfg)
        df = score_transactions(df, cfg)
    except Exception as e:
        error_msg = f"[PIPELINE FAILED] Engine error: {e}"
        print(error_msg)
        if notify:
            send_telegram(f"🔴 {error_msg}")
        return None

    # Step 3: Generate narratives and report
    print("[3/5] Generating risk narratives...")
    narratives = generate_narratives(df, cfg)
    report = format_text_report(narratives)

    report_path = f"{output_dir}/report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Step 4: Ollama summary
    print("[4/5] AI summary via Ollama...")
    summary = ollama_summarize(report)

    summary_path = f"{output_dir}/summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    # Step 5: Results
    critical = len(df[df["risk_score"] >= 80]) if "risk_score" in df.columns else 0
    high = len(df[(df["risk_score"] >= 50) & (df["risk_score"] < 80)]) if "risk_score" in df.columns else 0
    total = len(df)

    # Build Telegram message
    tg_msg = f"""🔍 *AML Scan Complete*
📍 Address: `{address[:10]}...{address[-6:]}`
⛓️ Chain: {chain_id}
📊 Transactions: {total}

🔴 CRITICAL: {critical}
🟠 HIGH: {high}

*AI Summary:*
{summary[:2000]}

_Report saved: {output_dir}_"""

    print(f"\n[5/5] Done. Report: {report_path}")
    print(f"  Critical: {critical} | High: {high} | Total: {total}")

    if notify:
        send_telegram(tg_msg)

    # Save scored CSV
    df.to_csv(f"{output_dir}/scored.csv", index=False)

    return {
        "address": address,
        "chain": chain_id,
        "total": total,
        "critical": critical,
        "high": high,
        "report_path": report_path,
        "summary": summary
    }


def run_watchlist(watchlist_path: str, chain_id: int = 1):
    """Scan all addresses in a watchlist file."""
    with open(watchlist_path) as f:
        addresses = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"[WATCHLIST] Scanning {len(addresses)} addresses...")
    results = []
    for addr in addresses:
        result = run_pipeline(addr, chain_id=chain_id, notify=False)
        if result:
            results.append(result)

    # Send combined summary
    if results:
        total_critical = sum(r["critical"] for r in results)
        msg = f"🔍 *Watchlist Scan Complete*\n"
        msg += f"Addresses: {len(results)}/{len(addresses)}\n"
        msg += f"Total CRITICAL: {total_critical}\n\n"
        for r in results:
            emoji = "🔴" if r["critical"] > 0 else "🟢"
            msg += f"{emoji} `{r['address'][:10]}...` — {r['critical']} critical, {r['high']} high\n"
        send_telegram(msg)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Live Inference Pipeline")
    parser.add_argument("--address", "-a", help="Ethereum address to scan")
    parser.add_argument("--chain", "-c", type=int, default=1, help="Chain ID (default: 1 = Ethereum)")
    parser.add_argument("--limit", "-l", type=int, default=200, help="Max transactions to fetch")
    parser.add_argument("--watchlist", "-w", help="Path to watchlist file (one address per line)")
    parser.add_argument("--no-telegram", action="store_true", help="Skip Telegram notification")

    args = parser.parse_args()

    if args.watchlist:
        run_watchlist(args.watchlist, chain_id=args.chain)
    elif args.address:
        run_pipeline(args.address, chain_id=args.chain, limit=args.limit, notify=not args.no_telegram)
    else:
        # Demo: scan Ronin Bridge exploiter (Lazarus Group)
        print("No address specified. Running demo on Ronin Bridge exploiter...")
        run_pipeline(
            "0x098B716B8Aaf21512996dC57EB0615e2383E2f96",
            chain_id=1,
            limit=50,
            notify=not args.no_telegram
        )

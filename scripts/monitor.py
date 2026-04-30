"""
monitor.py — Scheduled AML Monitoring Pipeline
================================================
Runs etherscan_fetcher + blockscout_fetcher on a watchlist of addresses,
feeds results to the engine, and generates alert reports.

Usage:
  python scripts/monitor.py                    # Run full watchlist
  python scripts/monitor.py --address 0xABC    # Single address check
  python scripts/monitor.py --status           # Show last run status
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd

# Fix Windows console encoding without replacing the stream object.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# Add parent dir for imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine"))

from etherscan_fetcher import fetch_live_data
from scripts.blockscout_fetcher import fetch_and_build as blockscout_fetch

NOW = datetime.now()
DATE_STR = NOW.strftime("%Y-%m-%d")
TIME_STR = NOW.strftime("%Y-%m-%d %H:%M:%S")

CONFIG_PATH = os.path.join(ROOT, "config", "watchlist.json")
MONITORING_DIR = os.path.join(ROOT, "data", "monitoring")


def load_watchlist() -> dict:
    if not os.path.exists(CONFIG_PATH):
        print(f"[ERROR] Watchlist not found: {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_watchlist(config: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def run_engine_on_df(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Run the 26-rule engine on a DataFrame and return scored results."""
    from engine_v11_blockchain import (
        CONFIG,
        compute_features,
        detect_bridge_hops,
        detect_concentrated_inflow,
        detect_coordinated_burst,
        detect_dormant_activation,
        detect_exchange_avoidance,
        detect_exit_rush,
        detect_flash_loan_burst,
        detect_high_risk_country,
        detect_layering,
        detect_layering_deep,
        detect_mixer_touch,
        detect_novel_wallet_dump,
        detect_ofac_hit,
        detect_peel_chain,
        detect_rapid_succession,
        detect_smurfing,
        detect_wash_cycle,
        risk_level,
        score_transactions,
    )

    cfg = {**CONFIG, "alert_threshold": int(threshold * 100)}

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in ["is_known_mixer", "is_bridge"]:
        if col not in df.columns:
            df[col] = False

    df = compute_features(df, cfg)
    df, _ = detect_layering(df, cfg)
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
    df["risk_level"] = df["risk_score"].apply(risk_level)

    return df


def generate_alert_report(results: list, threshold: float) -> str:
    """Generate human-readable alert report from monitoring results."""
    report = f"# AML Monitoring Alert Report — {DATE_STR}\n\n"
    report += f"**Run time:** {TIME_STR}\n"
    report += f"**Alert threshold:** {threshold}\n\n"
    report += "---\n\n"

    total_alerts = 0
    total_txns = 0

    for entry in results:
        label = entry["label"]
        address = entry["address"]
        source = entry["source"]
        df = entry.get("df")
        error = entry.get("error")

        report += f"## {label}\n"
        report += f"**Address:** `{address}`\n"
        report += f"**Source:** {source}\n"

        if error:
            report += f"**Status:** Error — {error}\n\n---\n\n"
            continue

        if df is None or df.empty:
            report += "**Status:** No transactions found\n\n---\n\n"
            continue

        n_total = len(df)
        total_txns += n_total

        # Filter alerts
        score_threshold = int(threshold * 100)
        alerts = df[df["risk_score"] >= score_threshold] if "risk_score" in df.columns else pd.DataFrame()
        n_alerts = len(alerts)
        total_alerts += n_alerts

        report += f"**Transactions scanned:** {n_total}\n"
        report += f"**Alerts triggered:** {n_alerts}\n\n"

        if n_alerts > 0:
            # Risk breakdown
            for level in ["CRITICAL", "HIGH", "MEDIUM"]:
                count = len(alerts[alerts["risk_level"] == level]) if "risk_level" in alerts.columns else 0
                if count > 0:
                    emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(level, "")
                    report += f"- {emoji} **{level}:** {count} transaction(s)\n"

            report += "\n### Top Alerts\n\n"

            # Show top 5 alerts
            top = alerts.nlargest(5, "risk_score")
            for _, row in top.iterrows():
                score = int(row.get("risk_score", 0))
                level = row.get("risk_level", "UNKNOWN")
                amount = row.get("amount", 0)
                reasons = row.get("reasons", "")
                sender = str(row.get("sender_id", ""))[:20]
                receiver = str(row.get("receiver_id", ""))[:20]

                report += f"- **Score {score} ({level})** — ${amount:,.0f}\n"
                report += f"  - Sender: `{sender}...`\n"
                report += f"  - Receiver: `{receiver}...`\n"
                if reasons:
                    signals = [s.strip() for s in str(reasons).split(";") if s.strip() and not s.strip().startswith("profile_")]
                    if signals:
                        report += f"  - Signals: {', '.join(signals[:5])}\n"
                report += "\n"

        report += "---\n\n"

    # Summary
    report += "## Summary\n\n"
    report += f"- **Addresses monitored:** {len(results)}\n"
    report += f"- **Total transactions scanned:** {total_txns}\n"
    report += f"- **Total alerts:** {total_alerts}\n"
    report += f"- **Report generated:** {TIME_STR}\n"

    return report


def monitor_watchlist(config: dict, api_key: str = "") -> list:
    """Run monitoring on all addresses in the watchlist."""
    addresses = config.get("addresses", [])
    threshold = config.get("alert_threshold", 0.7)
    output_dir = config.get("output_dir", "data/monitoring/")

    os.makedirs(os.path.join(ROOT, output_dir), exist_ok=True)

    results = []

    for entry in addresses:
        address = entry["address"]
        label = entry.get("label", address[:10])
        chain = entry.get("chain", 1)
        source = entry.get("source", "etherscan")

        print(f"\n{'─'*50}")
        print(f"  Monitoring: {label}")
        print(f"  Address: {address}")
        print(f"  Source: {source} (chain {chain})")
        print(f"{'─'*50}")

        try:
            if source == "blockscout":
                csv_path = os.path.join(ROOT, output_dir, f"{label.replace(' ', '_')}_{DATE_STR}.csv")
                df = blockscout_fetch(
                    address=address,
                    chain_id=chain,
                    limit=200,
                    output_path=csv_path,
                )
            else:
                csv_path = os.path.join(ROOT, output_dir, f"{label.replace(' ', '_')}_{DATE_STR}.csv")
                df = fetch_live_data(
                    addresses=[address],
                    api_key=api_key,
                    limit=200,
                    expand_hops=False,
                    output_path=csv_path,
                )

            if df is not None and not df.empty:
                scored_df = run_engine_on_df(df.copy(), threshold)
                results.append({
                    "label": label,
                    "address": address,
                    "source": source,
                    "df": scored_df,
                })
            else:
                results.append({
                    "label": label,
                    "address": address,
                    "source": source,
                    "df": None,
                })

        except Exception as e:
            print(f"[ERROR] {label}: {e}")
            results.append({
                "label": label,
                "address": address,
                "source": source,
                "error": str(e),
            })

        # Update last_checked
        if "last_checked" not in config:
            config["last_checked"] = {}
        config["last_checked"][address] = TIME_STR

        time.sleep(1)  # Rate limit buffer

    # Save updated timestamps
    save_watchlist(config)

    # Generate report
    report = generate_alert_report(results, threshold)
    report_path = os.path.join(ROOT, output_dir, f"alert-{DATE_STR}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"  Alert report saved → {report_path}")
    print(f"  Addresses monitored: {len(results)}")
    print(f"{'='*60}\n")

    return results


def print_status(config: dict):
    """Print last monitoring run status."""
    print(f"\n{'='*50}")
    print("AML MONITOR — Status")
    print(f"{'='*50}")

    addresses = config.get("addresses", [])
    last_checked = config.get("last_checked", {})

    for entry in addresses:
        addr = entry["address"]
        label = entry.get("label", addr[:10])
        source = entry.get("source", "etherscan")
        last = last_checked.get(addr, "never")
        print(f"\n  {label}")
        print(f"    Address: {addr[:20]}...")
        print(f"    Source: {source}")
        print(f"    Last checked: {last}")

    print(f"\n{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="AML Monitoring Pipeline")
    parser.add_argument("--address", type=str, help="Single address to check (bypasses watchlist)")
    parser.add_argument("--api-key", type=str, default=os.environ.get("ETHERSCAN_API_KEY", ""), help="Etherscan API key")
    parser.add_argument("--status", action="store_true", help="Show last run status")
    args = parser.parse_args()

    config = load_watchlist()

    if args.status:
        print_status(config)
        return

    if args.address:
        # Single address mode
        config["addresses"] = [{
            "address": args.address,
            "label": f"Manual_{args.address[:10]}",
            "chain": 1,
            "source": "etherscan",
        }]

    monitor_watchlist(config, api_key=args.api_key)


if __name__ == "__main__":
    main()

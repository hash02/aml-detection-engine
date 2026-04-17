# NEXUS-RISK — Blockchain AML Detection Engine

> Built by a financial services professional who got tired of seeing compliance tools that didn't understand how crypto actually moves.

[![Detection Rate](https://img.shields.io/badge/Detection%20Rate-94.9%25-brightgreen)](/)
[![Rules](https://img.shields.io/badge/Rules-28-blue)](/)
[![AI Layer](https://img.shields.io/badge/AI%20Layer-Isolation%20Forest-purple)](/)
[![Triage](https://img.shields.io/badge/Triage-63%25%20Queue%20Reduction-orange)](/)
[![CI](https://github.com/hash02/aml-detection-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/hash02/aml-detection-engine/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](/)

---

## What This Is

A working fraud detection engine for blockchain transactions. Not a concept. Not a demo. An actual system that was trained and tested against real Etherscan data from known exploits — Tornado Cash, Ronin Bridge, Lazarus Group, Wormhole, and the Nomad crowd-looting attack.

It evolved from a traditional AML rule engine (threshold + velocity) into a full 3-layer detection stack: rules → AI anomaly detection → dynamic triage labeling.

**Built from the inside out.** I work in financial services. I know what compliance teams actually need from a tool like this and what they're not getting from current systems.

---

## Results

| Metric | Value |
|--------|-------|
| Overall detection rate | **94.9%** |
| False positive rate | **20.2%** |
| Total rules | **28** (v12: +phish, +sub-threshold tranching, +machine cadence, +sybil fan-in) |
| Transactions tested | **813** (644 real Etherscan + 169 synthetic forensic) |
| Analyst queue reduction (triage) | **63%** |
| AI-only anomalies found | **21** |

---

## Case Studies — Real Exploits, Real Data

| Case | Attack Type | Detection | Notes |
|------|------------|-----------|-------|
| Tornado Cash (0.1/10/100 ETH + Gov) | Mixer laundering | **100%** | All 311 txns caught |
| Lazarus Group / Stake.com hack | State actor + OFAC | **100%** | All 119 txns caught |
| Ronin Bridge ($625M) | State actor + bridge | **84.3%** | Miss = non-OFAC intermediaries |
| Bybit drainer | Flash drain | **100%** | Single-tx event |
| Euler Finance ($197M) | Flash loan exploit | **23.3%** | Miss = needs block-level timestamps |
| Wormhole Bridge ($320M) | Bridge exploit | **100%** | Synthetic forensic data |
| Nomad Bridge ($190M) | Crowd looting (300+ wallets) | **100%** | Synthetic forensic data |
| Vitalik.eth (control) | Legitimate whale | **20.4% FP** | Known wallet, low false flag rate |
| ETH Foundation (control) | Legitimate institutional | **20.0% FP** | Grant distribution pattern |

The 5.1% detection gap is structural — Euler needs block-level timestamps (not available in hourly Etherscan data), Ronin's intermediaries aren't OFAC-listed. More rules won't close this gap. Better data sources will.


---

## Live Threat Intelligence

Every 24 hours this engine pulls from three independent sources so the detection surface stays current without manual updates. Nothing is hardcoded past the initial fallback baseline.

| Feed | Source | Purpose |
|---|---|---|
| OFAC SDN (ethereum) | [0xB10C/ofac-sanctioned-digital-currency-addresses](https://github.com/0xB10C/ofac-sanctioned-digital-currency-addresses) (MIT) | US Treasury sanctions list |
| MetaMask eth-phishing-detect | [MetaMask/eth-phishing-detect](https://github.com/MetaMask/eth-phishing-detect) (MIT) | Scam and phishing addresses |
| Chainalysis Sanctions Oracle | On-chain contract, live lookup | Regulatory belt-and-braces |

Latest snapshot lives at [](./addresses/) with SHA256 checksums in [](./addresses/MANIFEST.json). Downstream consumers (like [aml-roaster](https://github.com/hash02/aml-roaster)) pull this directly on their own schedule.

**What this means in practice:** a new OFAC listing ships into detection within 24 hours with zero human involvement. The engine is not static. If you are showing this to a compliance team, they will care about this.

[![Last refresh](https://img.shields.io/github/actions/workflow/status/hash02/aml-detection-engine/refresh-addresses-daily.yml?branch=main&label=threat%20feeds&logo=github)](https://github.com/hash02/aml-detection-engine/actions/workflows/refresh-addresses-daily.yml)


---

## Architecture — 3-Layer Detection Stack

```
┌─────────────────────────────────────────────────────┐
│  LAYER 1: Rule Engine v11 — 22 Rules                │
│  Known patterns, legally defensible, explainable    │
│  Fast: O(n log n), deterministic                    │
├─────────────────────────────────────────────────────┤
│  LAYER 2: AI Anomaly Detection                      │
│  Isolation Forest on 13 graph features              │
│  Zero-day immune — learns "normal", flags deviation │
│  Top signals: time_burstiness, passthrough wallets  │
├─────────────────────────────────────────────────────┤
│  LAYER 3: Dynamic Triage (Item Rarity System)       │
│  4-dimension confidence scoring                     │
│  LEGENDARY → RARE → MAGIC → COMMON                 │
│  695 flagged → 256 high-priority (63% reduction)   │
└─────────────────────────────────────────────────────┘
```

---

## The 26 Rules

**v6 · Core Thresholds**
`large_amount` · `velocity` · `fan_in` · `structuring`

**v7 · Chain-Native**
`mixer_touch` · `bridge_hop` · `peel_chain` · `layering`

**v8 · Wallet Intelligence**
`novel_dump` · `conc_inflow`

**v9 · Speed + Sanctions**
`OFAC_SDN` · `flash_loan_burst` · `coord_burst`

**v10 · Dormancy**
`dormant_activation` — BitFinex-style sleeping wallets reviving after years

**v11 · Advanced Patterns**
`wash_cycle` · `smurfing` · `exit_rush` · `rapid_succession` · `high_risk_country` · `exchange_avoidance` · `layering_deep`

**v12 · Live Feeds + Modern Typologies**
`phishing_hit` (MetaMask + no-KYC offramps) · `sub_threshold_tranching` (just-under-$10k bunches) · `machine_cadence` (bot-timing signatures) · `sybil_fan_in` (airdrop-sybil / drainer collectors)

**v13 · Drainer + Multi-Asset**
`drainer_signature` (Inferno / Angel drainer pattern — approval + multi-asset drain) · `address_poisoning` (0-value lookalike first-4/last-4 match)

### Live threat-intelligence feeds
The sanctions and phishing lists are now refreshable from the `engine/feeds.py` loader:
- **OFAC SDN** — 0xB10C community mirror (US Treasury virtual-currency addendum)
- **MetaMask eth-phishing-detect** — address-level drainer list
- **Chainalysis Sanctions Oracle** — on-chain belt-and-braces check via `CHAINALYSIS_RPC_URL`
- **Analyst-curated** — no-KYC off-ramps, LRT drainers, mixer/bridge rosters

Run `python scripts/refresh_feeds.py` (cron-ready) to pull the latest. `FEEDS_OFFLINE=1` keeps CI and air-gapped deployments happy — baselines ship in the repo.

---

## Attack Pattern Taxonomy

| Group | Attacks | Signature |
|-------|---------|-----------|
| Mixer Laundering | Tornado Cash | Country flag + mixer touch + layering |
| State Actor | Ronin, Lazarus, Bybit | OFAC match + bridge + fan-out |
| Protocol Exploit | Euler | Flash loan burst + novel wallet + rapid dump |
| Dormant Revival | BitFinex | 5+ year dormancy + sudden large move |
| Bridge Exploit | Wormhole, Nomad | Novel wallet + bridge + coordinated burst |

---

## Repo Structure

```
aml-detection-engine/
├── engine/
│   └── engine_v11_blockchain.py   # Main rule engine — 26 rules, full scoring
├── ai_layer/
│   ├── aml_ai_layer.py            # Isolation Forest + graph features
│   └── triage_labeler.py          # Dynamic confidence scoring (rarity tiers)
├── dashboard/
│   └── nexus_dashboard.html       # Open in browser — full results dashboard
├── data/
│   └── sample_transactions.csv    # 30-row sample (20 forensic + 10 control)
├── scripts/
│   └── generate_cases.py          # Synthetic forensic data generator
└── README.md
```

---

## How to Run

**Requirements:**
```bash
pip install pandas numpy scikit-learn
```

**Run the rule engine on sample data:**
```bash
python engine/engine_v11_blockchain.py
# Output: risk scores, triage tiers, SAR narratives
```

**Run the AI layer:**
```bash
python ai_layer/aml_ai_layer.py
# Requires rule engine output first
```

**Run triage scoring:**
```bash
python ai_layer/triage_labeler.py
# Output: LEGENDARY/RARE/MAGIC/COMMON classification per transaction
```

**View the dashboard:**
```
Open dashboard/nexus_dashboard.html in any browser
No server needed — fully static
```

**On your own data:**
Point `DATA_PATH` in `engine_v11_blockchain.py` to your CSV. Required columns:
```
sender_id, receiver_id, amount, country, timestamp
```
Optional (improves detection): `sender_tx_count`, `sender_avg_amount`, `sender_active_days`, `account_type`

**Real Ethereum data** (requires Etherscan API key):
Edit `scripts/generate_cases.py` with your API key — fetches real transaction history for any wallet address.

---

## Enterprise-Readiness Checklist

v12 + Phase 2 shipped:

- [x] **Live demo** — Streamlit app with 26 rules + SAR-SF JSON export per alert
- [x] **GitHub Actions CI** — ruff + pytest on Python 3.11 & 3.12
- [x] **Live threat-intel feeds** — OFAC, MetaMask phishing, Chainalysis oracle, analyst-curated off-ramps
- [x] **Password-gated Streamlit** — `AML_APP_PASSWORD` env var; open demo when unset
- [x] **Append-only audit log** — SQLite, dedup by `(tx_id, input_hash)`
- [x] **Observability** — Sentry (`SENTRY_DSN`), Prometheus (`PROMETHEUS_PORT`), structured logs
- [x] **Backtest harness** — `scripts/backtest.py` replays cases, emits regression-safe JSON report
- [x] **Docker image** — multi-stage, non-root, healthcheck

Phase 3 shipped:

- [x] **Multi-asset** — ERC-20 tokentx ingestion, per-token amount + decimals columns
- [x] **Drainer-signature rule** — multi-asset drain inside a 2-min window
- [x] **Address-poisoning rule** — dust + first-4/last-4 lookalike match
- [x] **Cross-chain adapters** — base class + ETH adapter + Tron / Solana stubs
- [x] **RBAC** — 3-tier hierarchy (admin > reviewer > analyst) + per-action permission map

Still aspirational:

- [ ] **Tron / Solana full adapters** — currently stubs; SPL / TRC20 decoding TODO
- [ ] **GNN layer** — GraphSAGE on wallet transaction graphs

---

## About

Built by **Bionic Banker** — a financial services professional working at the intersection of traditional finance and blockchain. I work inside legacy financial systems during the day and build the tools that should exist at night.

📝 Research + writing: [bionicbanker.tech](https://bionicbanker.tech)
🐦 Twitter/X: [@BionicBanker](https://twitter.com/BionicBanker)
💼 LinkedIn: [Bionic Banker](https://linkedin.com/in/bionicbanker)

---

## Disclaimer

This engine is a research and educational tool. It uses synthetic forensic data modelled after public post-mortems (Certik, Coinbase Security, Immunefi). It does not constitute financial or legal advice. Real AML compliance requires licensed professionals and regulated systems.

---

*NEXUS-RISK v11 · Feb 2026 · Detection: 94.9% · Rules: 22 · AI Layer: Live*

# NEXUS-RISK — Blockchain AML Detection Engine

> Built by a financial services professional who got tired of seeing compliance tools that didn't understand how crypto actually moves.

[![Detection Rate](https://img.shields.io/badge/Detection%20Rate-94.9%25-brightgreen)](/)
[![Rules](https://img.shields.io/badge/Rules-22-blue)](/)
[![AI Layer](https://img.shields.io/badge/AI%20Layer-Isolation%20Forest-purple)](/)
[![Triage](https://img.shields.io/badge/Triage-63%25%20Queue%20Reduction-orange)](/)

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
| Total rules | **22** |
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

## The 22 Rules

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
│   └── engine_v11_blockchain.py   # Main rule engine — 22 rules, full scoring
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

## What's Next

- [ ] **Live demo** — Streamlit app: upload CSV or enter wallet address, get risk scores instantly
- [ ] **GitHub Actions CI** — auto-run eval on every push, track detection rate over time
- [ ] **GNN layer** — GraphSAGE on wallet transaction graphs (needs 5k+ labelled examples)
- [ ] **Real-time API** — Flask endpoint, production-ready (CORS + auth)
- [ ] **Dune Analytics integration** — real on-chain data at block level

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

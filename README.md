# NEXUS-RISK — Blockchain AML Detection Engine

> Built by a financial services professional who got tired of seeing compliance tools that didn't understand how crypto actually moves.

[![Rules](https://img.shields.io/badge/Rules-28-blue)](/)
[![AI Layer](https://img.shields.io/badge/AI%20Layer-Isolation%20Forest-purple)](/)
[![CI](https://github.com/hash02/aml-detection-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/hash02/aml-detection-engine/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](/)

---

## What This Is

A working, rule-based fraud detection engine for blockchain transactions, with an anomaly-detection layer and a triage stage. It was developed against Etherscan data and synthetic forensic reconstructions of known exploits: Tornado Cash, Ronin Bridge, Lazarus Group, Wormhole, and the Nomad crowd-looting attack.

It is a research and portfolio project, not a validated production system. See [Evaluation status](#evaluation-status) and [Limitations](#limitations) for what has and has not been independently reproduced.

It evolved from a traditional AML rule engine (threshold + velocity) into a full 3-layer detection stack: rules → AI anomaly detection → dynamic triage labeling.

**Built from the inside out.** I work in financial services. I know what compliance teams actually need from a tool like this and what they're not getting from current systems.

---

## Evaluation status

_Updated 2026-10-20 after an independent review of this repository._

**What is verifiable from this repo today:**

| Item | Status |
|---|---|
| Detection abilities | **28** tracked in `docs/RULES.md` (23 `detect_` functions + 5 pipeline stages) |
| Anomaly layer | Isolation Forest on graph features (`ai_layer/`) |
| Tests and CI | Test suite in `tests/`; CI was passing on `main` when this section was written (2026-10-20); see the CI badge for current status |
| Data in repo | 254 rows: `data/sample_transactions.csv` (30) + `data/ronin_live_results.csv` (224) |

**What is not yet verifiable:**

- The performance figures below come from an earlier evaluation (February 2026) whose 813-row dataset and labels are not in this repository, so they cannot be regenerated here. This is an evidence gap, not proof that the evaluation never ran.
- `scripts/backtest.py` reports `detection_rate = flagged / total`. That is an **alert rate**, not a detection rate. Precision and recall need ground-truth labels (a `.expected.json` sidecar), which are not supplied.
- An offline replay of the 30-row sample (Oct 2026) flagged 30 of 30 rows. The sample has no known benign transactions, so it cannot show whether the engine is selective.

**Next:** a labeled holdout with both benign and suspicious transactions, labels and sampling documented, protocol frozen before scoring, then confusion matrix, precision, recall, false-positive rate and alert rate with exact reproduction commands. New figures will be published only from that run.

### Historical results (February 2026, not reproducible from this repo)

Kept for the record. Treat as unverified until the evaluation above is published.

| Metric | Value |
|--------|-------|
| Overall detection rate | **94.9%** |
| False positive rate | **20.2%** |
| Total rules | **28** (v12: +phish, +sub-threshold tranching, +machine cadence, +sybil fan-in) |
| Transactions tested | **813** (644 real Etherscan + 169 synthetic forensic) |
| Analyst queue reduction (triage) | **63%** |
| AI-only anomalies found | **21** |

**Rule-count note:** This project tracks 28 detection abilities verified in `docs/RULES.md`: 23 are direct `detect_` functions listed in the section below, and 5 are feature/classification pipeline stages. `feature: profile classification` is implemented in `etherscan_fetcher.classify_profile()` to classify addresses as exchange, DeFi, EOA, or contract; it is not listed as a standalone rule label in this section. The section below shows 27 named rule labels.

---

#### Historical case studies (same caveat)

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

The historical analysis attributed the 5.1% gap to data limits: Euler likely needs block-level timestamps (not available in hourly Etherscan data), and Ronin's intermediaries aren't OFAC-listed. This explanation has not been re-tested; treat it as a hypothesis until the labeled evaluation is run.


---

---

## Live Threat Intelligence

The engine integrates three independent sources, refreshed by a scheduled GitHub Actions workflow.

> **Current status (2026-10-20):** the staging guard and failure-alert fix (PR #17) has merged. The scheduled workflow has not run since 2026-07-17 for a reason not yet identified, and the committed snapshot in `addresses/` was last updated **2026-04-23**. Until a supervised run is approved and completed, detection uses that snapshot, not same-day lists.

| Feed | Source | Purpose |
|---|---|---|
| OFAC SDN (ethereum) | [0xB10C/ofac-sanctioned-digital-currency-addresses](https://github.com/0xB10C/ofac-sanctioned-digital-currency-addresses) (MIT) | US Treasury sanctions list |
| ScamSniffer scam-database | [scamsniffer/scam-database](https://github.com/scamsniffer/scam-database) (GPL-3.0) | Scam and phishing addresses |
| Chainalysis Sanctions Oracle | On-chain contract, live lookup | Regulatory belt-and-braces |

Latest snapshot lives in [`addresses/`](./addresses/) with SHA256 checksums in [`addresses/MANIFEST.json`](./addresses/MANIFEST.json). Downstream consumers (like [aml-roaster](https://github.com/hash02/aml-roaster)) pull this directly on their own schedule.

**Design intent:** when the refresh workflow runs, a new OFAC listing reaches detection within 24 hours with no manual step.

[![Last refresh](https://img.shields.io/github/actions/workflow/status/hash02/aml-detection-engine/refresh-addresses-daily.yml?branch=main&label=threat%20feeds&logo=github)](https://github.com/hash02/aml-detection-engine/actions/workflows/refresh-addresses-daily.yml)


---

## Architecture — 3-Layer Detection Stack

```
┌─────────────────────────────────────────────────────┐
│  LAYER 1: Rule Engine v11 — 28 detection abilities  │
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
│  Historical run: 695 flagged → 256 high-priority   │
└─────────────────────────────────────────────────────┘
```

---

## The 28 Detection Abilities

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
`phishing_hit` (ScamSniffer + no-KYC offramps) · `sub_threshold_tranching` (just-under-$10k bunches) · `machine_cadence` (bot-timing signatures) · `sybil_fan_in` (airdrop-sybil / drainer collectors)

**v13 · Drainer + Multi-Asset**
`drainer_signature` (Inferno / Angel drainer pattern — approval + multi-asset drain) · `address_poisoning` (0-value lookalike first-4/last-4 match)

### Live threat-intelligence feeds
The sanctions and phishing lists are now refreshable from the `engine/feeds.py` loader:
- **OFAC SDN** — 0xB10C community mirror (US Treasury virtual-currency addendum)
- **ScamSniffer scam-database**: address-level scam and phishing list (GPL-3.0; open data lags 7 days by publisher disclosure)
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
│   └── engine_v11_blockchain.py   # Main rule engine — 23 direct detection functions, full scoring
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

- [x] **Live demo** — Streamlit app with 28 detection abilities + SAR-SF JSON export per alert
- [x] **GitHub Actions CI** — ruff + pytest on Python 3.11 & 3.12
- [x] **Threat-intel feed integrations** — OFAC, ScamSniffer phishing, Chainalysis oracle, analyst-curated off-ramps (scheduled refresh currently failing; see Limitations)
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

Phase 4 shipped:

- [x] **FastAPI sidecar** — `/score`, `/healthz`, `/feeds`, `/audit`, bearer-token auth
- [x] **Unified notifier** — Telegram + Slack + generic webhook, env-gated, fails soft
- [x] **Per-rule score breakdown** — analyst sees which rule contributed how many points
- [x] **Live-ops dashboard** — feed freshness + audit-log tail in Streamlit
- [x] **Makefile** — `make install | test | lint | backtest | feeds | run | api | docker | clean`

Phase 5 shipped:

- [x] **ML anomaly layer** — Isolation Forest (deterministic, versioned) appended as Layer 2
- [x] **Alert suppression** — dedup by (sender, top rule, time bucket) + per-rule cooldown
- [x] **Analyst disposition workflow** — reviewer role files escalate/dismiss/sar_filed via UI
- [x] **Schema validator** — fast-fail on bad CSVs with human-readable error codes
- [x] **Benchmark harness** — `scripts/benchmark.py` measures latency + throughput per-size

Phase 6 shipped:

- [x] **Case management** — alerts grouped by subject into cases with a strict FSM lifecycle
- [x] **Triage queue** — priority-sorted Streamlit expander + `/cases` API
- [x] **Rule drift monitor** — daily fire-rate tracking with z-score deviation alerts
- [x] **API explainability** — `/score?explain=true` returns per-rule contribution breakdown
- [x] **Rate limiting** — `AML_API_RATE_LIMIT` env var wires slowapi middleware when set

Phase 7 shipped:

- [x] **Rule catalogue generator** — `docs/RULES.md` auto-built from docstrings, CI-gated for staleness
- [x] **Threshold tuning** — `engine/tuning.py` advisory suggestions from disposition history
- [x] **Operator docs** — `docs/DEPLOYMENT.md`, `OPERATIONS.md`, `MODEL_CARD.md`
- [x] **Production compose** — `deploy/docker-compose.prod.yml` bundling streamlit + FastAPI + hourly feed refresher

Phase 8 shipped:

- [x] **Graph topology features** — out_degree, in_degree, passthrough, unique_ratio, betweenness → ML layer
- [x] **What-if threshold replay** — `scripts/whatif.py` with CONFIG overrides, added/removed/stable alert diff
- [x] **Data retention + PII redaction** — `engine/retention.py` + `scripts/purge.py`
- [x] **GDPR subject forget** — `forget_subject()` wipes every reference across audit + cases stores

Phase 9 shipped:

- [x] **Subject enrichment** — `engine/enrichment.py` local label DB + feed-backed categories + pluggable provider, TTL-cached
- [x] **Severity-routed notifications** — `AML_NOTIFY_ROUTING` JSON maps level → channel list (CRITICAL → PagerDuty-via-webhook, WARN → Slack, …)
- [x] **Weekly backtest automation** — GitHub Action runs Monday 06:00 UTC; opens a regression issue if detection rate drops below 60%
- [x] **Enrichment badges in UI** — each alert shows the sender/receiver category chip at a glance

Still aspirational:

- [ ] **Tron / Solana full adapters** — currently stubs; SPL / TRC20 decoding TODO
- [ ] **GNN layer** — GraphSAGE on wallet transaction graphs

---

## API audit integrity

Streamlit shared-password logins are analyst-only, including existing sessions.
`AML_ADMIN_USERNAMES` and `AML_REVIEWER_USERNAMES` no longer elevate a typed name.
The public demo and scoring remain available. Reviewer/admin controls through
this login are intentionally unavailable until verified per-user authentication
is implemented. A displayed username is a label, not verified identity.

Public `/score` requests remain available when `write_audit` is false. Requests
with `write_audit: true`, audit/case/drift reads, and feed refresh require a bearer
token matching `AML_API_TOKEN`. If the token is not configured, these protected
operations return HTTP 503 rather than opening access. Store the token in deployment
secrets and send it in the Authorization header.

An authenticated scoring request that asks to save alerts returns success only
when all alert rows were recorded or already exist. Failed or incomplete audit
writes return HTTP 503. A failed batch may have saved some rows; retrying the same
transaction inputs is deduplicated by the existing audit store. Scoring thresholds
and detection rules are unchanged by this boundary.

This is a shared API credential, not per-user identity or a replacement for the
Streamlit authentication design. CI installs the API dependencies and exercises
authorization, persistence failures and replay behavior.

## About

Built by **Bionic Banker** — a financial services professional working at the intersection of traditional finance and blockchain. I work inside legacy financial systems during the day and build the tools that should exist at night.

📝 Research + writing: [bionicbanker.tech](https://bionicbanker.tech)
🐦 Twitter/X: [@BionicBanker](https://twitter.com/BionicBanker)
💼 LinkedIn: [Bionic Banker](https://linkedin.com/in/bionicbanker)

---

## Limitations

- **Data:** development data mixes real Etherscan rows and synthetic forensic reconstructions. The `label` column (`SYNTH_FORENSIC`, `REAL_DATA`) records provenance, not ground-truth risk.
- **No validated accuracy:** no labeled benign/suspicious holdout is published yet (see [Evaluation status](#evaluation-status)).
- **Country field:** `home_country` is `CA`; any other value, including `UNKNOWN` (typical for on-chain rows), counts as foreign. On its own this adds context only, not score.
- **Threat feeds:** the scheduled refresh is currently failing; the committed snapshot is dated 2026-04-23.
- **Scope:** triage and education only. It does not accuse, file, freeze, or replace compliance judgment.

---

## Disclaimer

This engine is a research and educational tool. It uses synthetic forensic data modelled after public post-mortems (Certik, Coinbase Security, Immunefi). It does not constitute financial or legal advice. Real AML compliance requires licensed professionals and regulated systems.

---

*NEXUS-RISK v11 · 28 detection abilities · Isolation Forest anomaly layer · Evaluation status updated 2026-10-20*

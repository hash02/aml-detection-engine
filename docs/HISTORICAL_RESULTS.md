# Historical results (February 2026)

These are historical figures from earlier runs of this project, computed on data that is not in this repository, and they cannot be reproduced from this repo. They are retained for provenance only and are not current results. The README's Evaluation status section states the project's standing position.

Kept for the record. Treat as unverified until the evaluation above is published.

| Metric | Value |
|--------|-------|
| Overall detection rate | **94.9%** |
| False positive rate | **20.2%** |
| Total rules | **28** (v12: +phish, +sub-threshold tranching, +machine cadence, +sybil fan-in) |
| Transactions tested | **813** (644 real Etherscan + 169 synthetic forensic) |
| Analyst queue reduction (triage) | **63%** |
| AI-only anomalies found | **21** |

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

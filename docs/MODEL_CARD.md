# Model Card — AML Detection Engine

A compliance-grade summary of what the engine does, how it was
calibrated, and where it can go wrong.

## Overview

- **Primary use** — flag blockchain transactions for AML review.
- **Layers**
  1. **Rule engine** (28 deterministic rules — see `docs/RULES.md`)
  2. **Isolation Forest anomaly** (`engine/ml_anomaly.py`, `MODEL_VERSION`)
  3. **Triage labelling / case grouping** (priority-sorted queue)

## Intended users

- Compliance analysts in crypto-asset service providers (CASPs)
- Blockchain forensics teams
- Internal fraud-risk investigators

## Out-of-scope

- **Legal disposition** — the engine suggests SAR-SF output, but filing
  a SAR remains a human decision with regulatory responsibility.
- **Real-time block-level interception** — rules assume minute-level
  granularity. Sub-block MEV / flash-loan exploits will be caught
  only to the extent the tx stream timestamps allow.
- **Stablecoin-issuer freezes** — this is a detection engine, not an
  enforcement engine.

## Training / calibration data

The Isolation Forest trains per-run on the current input frame (no
cross-run model file). Deterministic with `random_state=42`. Features:

| Feature | Source | Why it matters |
|---------|--------|----------------|
| `log_amount`                | input | Large-value detection, log-transformed to spread the tail |
| `tx_count_in_window`        | feature engineering | Velocity proxy |
| `small_tx_count_in_window`  | feature engineering | Structuring signal |
| `small_tx_count_6h`         | feature engineering | Slow structuring |
| `fan_in_count`              | feature engineering | Receiver concentration |
| `hour_of_day`               | timestamp | Human vs bot activity pattern |
| `day_of_week`               | timestamp | Weekend anomalies |

## Performance on curated test data

| Dataset | Rows | Aggregate alert rate | Notes |
|---------|------|----------------------|-------|
| `data/sample_transactions.csv` | 30 | ≈ 67% | Heavy in OFAC-touching rows by design |
| Synthetic forensic suite       | 169 | ≈ 95% | Ronin, Lazarus, Wormhole patterns |
| Control (Vitalik.eth, EF)      | 644 | ≈ 20% FP | Legitimate whale activity |

These numbers are not comparable to a production deployment. See
`scripts/backtest.py` + `scripts/benchmark.py` for regression evidence.

## Known failure modes

1. **Unknown tokens** — ERC-20 transfers priced at `amount=0` when
   the token isn't in our stable/WETH list. Velocity + count rules
   still catch movement; dollar-value rules don't.
2. **Cold-start drift detection** — < 8 days of history → no drift
   alerts by design. Don't rely on `/drift` for the first week after
   deployment.
3. **Strict FSM cases** — reviewers cannot partially unwind a case
   (e.g. ESCALATED → IN_REVIEW). Admin can reopen CLOSED or DISMISSED
   cases; everything else is a terminal edge.
4. **Telegram / Slack legacy token** — a Telegram bot token was once
   committed to history. It has since been removed from the working
   tree; operators must rotate if they copied from a historical commit.

## Governance

- **Model version** — pinned in `engine/ml_anomaly.MODEL_VERSION`.
  Any feature or hyperparameter change requires a version bump.
- **Threshold provenance** — all rule weights + thresholds are in
  `CONFIG` at the top of `engine/engine_v11_blockchain.py`. Changes
  go through PR review.
- **Disposition feedback loop** — `engine.tuning.suggest()` analyses
  reviewer dispositions and surfaces advisory threshold changes. The
  engine never auto-applies them.
- **Audit trail** — every alert fire and every reviewer disposition
  lands in `audit.db`. The `/audit` endpoint is bearer-gated.

## Retraining cadence

The IF model is fit per-run, so "retraining" in the classical sense
doesn't apply. Instead:

- Run `make backtest` monthly against a frozen reference dataset.
  Flag > 5% drift in aggregate alert rate.
- Review `engine.tuning.suggest()` output weekly.
- Bump `MODEL_VERSION` + write a changelog entry when features change.

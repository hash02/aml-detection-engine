# Operations Runbook

This is the on-call guide. If something is on fire, start here.

## Daily

- **Check feed freshness** — `/feeds` API or the Streamlit Live-Ops
  panel. If any feed is > 24h old, run `make feeds` (or `/feeds/refresh`).
- **Check drift alerts** — `/drift` API. Any rule with |z| ≥ 3 means
  either the world changed (investigate) or the rule is broken (fix).
- **Glance at triage queue** — `/cases`. Anything in `IN_REVIEW` for
  more than 48 hours is likely stuck and needs reassignment.

## Weekly

- **Threshold tuning review** — run `python -c "from engine.tuning
  import suggest; [print(s) for s in suggest()]"`. Review suggestions
  with the lead reviewer; accepted changes go into `CONFIG` via a PR.
- **Backtest diff** — `make backtest`, diff against the previous
  week's report. Any rule whose fire count changed > 50% is worth a
  look.

## Monthly

- **Rotate secrets** — see `docs/DEPLOYMENT.md` → "Secrets to rotate".
- **Model governance review** — verify `engine/ml_anomaly.metadata()`
  still reflects the deployed model. Bump `MODEL_VERSION` if you
  change features.
- **Audit log export** — `/audit?limit=1000` → JSON → compliance
  archive.

## Incident runbook

### CI red on `main`

1. Check the failed job logs.
2. Most common cause: a detection rule changed and its test needs
   updating. Fix on a branch; never disable the test.
3. If the failure is a transient feed fetch, verify `FEEDS_OFFLINE=1`
   is set in the CI workflow.

### Streamlit 500

1. Check Sentry if wired; look for an unhandled exception in
   `run_engine()`.
2. Verify the input CSV passes `engine.schema.validate_dataframe()`.
3. Check the audit db isn't locked (another process writing).

### API rate-limited by Etherscan

1. `ETHERSCAN_API_KEY` unset → register at etherscan.io/register (free).
2. Already set → you're over 5 rps; tune `--limit` on the fetcher.
3. Long-term: switch to Blockchair (no key) or a paid Alchemy plan.

### OFAC feed mirror 404

The 0xB10C mirror occasionally moves. `engine.feeds._fetch_ofac_sdn()`
swallows the failure and we keep using the bundled baseline. If the
baseline is stale, update `BASELINE_OFAC` manually from the Treasury
website.

### Drift alert on a core rule

If `OFAC_SDN_MATCH` or `drainer_signature` drift-alerts:
1. Do not silence — these are high-precision rules. A spike means
   real activity.
2. Check the triage queue — are there corresponding open cases?
3. If the spike is legitimate (mass exploit event), file an ops
   note and keep moving.

## Feed refresh cadence

| Feed                | Recommended cadence | Source risk |
|---------------------|---------------------|-------------|
| `ofac_sdn`          | Daily               | Government mirror — low |
| `metamask_phish`    | Hourly              | Community list — rotate often |
| `chainalysis`       | Per-query (on-chain)| Free oracle |
| `mixers` / `bridges`| Weekly              | Analyst-maintained |

`scripts/refresh_feeds.py` in the `feeds` compose service defaults to
hourly. Increase to 15-min if you're dealing with active incident
response.

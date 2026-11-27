# Monitor fixes, November 2026

Two scoring-path fixes. Weights and thresholds are unchanged. `detect_drainer_signature` and `detect_address_poisoning` are still not called by the daily monitor.

## Four detectors were never called by the monitor

`scripts/monitor.py` `run_engine_on_df` is what `.github/workflows/aml-monitor.yml` runs. Commit `3b206e8` (PR 1, merged as `c750ea7` on 2026-04-16) added four detectors to `engine/engine_v11_blockchain.py`:

- `detect_phish_hit`
- `detect_sub_threshold_tranching`
- `detect_machine_cadence`
- `detect_sybil_fan_in`

That commit also edited `scripts/monitor.py`. It changed the docstring on `run_engine_on_df` from "22-rule engine" to "26-rule engine" and reformatted the import list. It did not add calls. `streamlit_app.py` and `scripts/backtest.py` do call the four functions, in that order, after `detect_layering_deep` and before `score_transactions`. `detect_phish_hit` loads the phishing feed and the no-KYC offramp feed itself through `get_feed`, so the monitor does not pass a separate list.

The only later commit that touches `scripts/monitor.py` is `1ecfaff` (`ci: fix scheduled AML engine workflows`), which changes stdout encoding. A search of commits, issues, and pull requests in this repo found no recorded reason for leaving the four calls out. The comment was updated and the calls were not. That is consistent with the calls being forgotten.

The monitor now calls them the same way it calls the other `detect_*` functions, and the same way Streamlit does. The docstring no longer says "26-rule engine". It states what this function runs: 21 of the 23 `detect_*` functions, then `score_transactions`. `detect_phish_hit` still loads its own feeds.

`detect_drainer_signature` and `detect_address_poisoning` stay unwired. The drainer detector returns immediately when `asset_type` is absent, and the monitor's fetched rows do not carry token fields. Address poisoning was left out of this change on purpose.

### Offline run

`FEEDS_OFFLINE=1`. No API keys. `run_engine_on_df` on `data/ronin_live_results.csv` (224 rows) completed with no error. Log lines:

- `[PHISH] Phishing / drainer matches: 0 transactions`
- `[TRANCH] Sub-threshold tranching: 0 transactions`
- `[CADENCE] Machine cadence: 0 senders flagged`
- `[SYBIL] Sybil fan-in: 0 transactions`

None of the four flags fired. This file is the Ronin exploiter set. Every row matches OFAC and stays `CRITICAL`. It does not contain a phishing hit, a sub-$10k tranche cluster, a uniform cadence, or a tight sybil fan-in.

A synthetic frame through the same function, shaped like the golden tests (FixedFloat offramp, three $8.8k to $9.5k tranches, eight transfers 60 seconds apart, six similar amounts into one receiver), also completed with no error and fired all four: phishing 1 row, tranching 3 rows, cadence 1 sender (8 rows), sybil 6 rows.

## Unknown country was scored as high-risk and foreign

`etherscan_fetcher.infer_country` returns `UNKNOWN` for an address that is not a known mixer, bridge, or exchange. That is the usual value on a live fetched row.

Two engine rules then treated that default as risk:

- `CONFIG["high_risk_countries"]` included `UNKNOWN`, so `detect_high_risk_country` flagged the row. Alone, that adds half of `w_high_risk_country` (15). With another listed signal (including OFAC), it adds the full 30.
- `home_country` is `CA`. `score_row` treated every other country code, including `UNKNOWN`, as foreign. Foreign points (`w_foreign`, 20) are added only when another v6 signal is already present. Otherwise the reason is `foreign_context_only` and the points are 0.

A normal fetched transfer therefore received high-risk points, and a large one also received foreign points.

The change is limited to that default. `UNKNOWN` is removed from `high_risk_countries`. `is_scored_foreign` returns false for `UNKNOWN`, and both `score_row` and `engine/explain.py` use it, so the score and the point breakdown stay aligned. `infer_country` still writes `UNKNOWN`. The label means the jurisdiction is not known. It is no longer a risk signal.

`XX` stays on the high-risk list. `infer_country` returns `XX` for known mixers so the engine treats that tag as suspicious. Known country codes other than `CA`, including FATF codes such as `KP`, are unchanged.

### Point breakdown, before and after

Each ordinary sample is one row, scored by `run_engine_on_df`. `sender_profile` is `UNKNOWN`, which uses the `PERSONAL_LIKE` multiplier of 1.0. The large-amount threshold for that profile is $10,000 (30 points). Counterparties are unlisted addresses.

| Sample | Before | After |
| --- | --- | --- |
| $12,000, country `UNKNOWN` | 65 HIGH. large_amount 30, foreign_country 20, high_risk_country 15 | 30 LOW. large_amount 30 |
| $2,500, country `UNKNOWN` | 15 LOW. high_risk_country 15. foreign recorded as context only, 0 points | 0 LOW. profile label only |
| $8,750, country `UNKNOWN` | 15 LOW. high_risk_country 15. foreign context only, 0 points | 0 LOW. profile label only |
| $12,000, country `CA` | 30 LOW. large_amount 30 | 30 LOW. unchanged |
| $12,000, country `KP` | 65 HIGH. large_amount 30, foreign_country 20, high_risk_country 15 | 65 HIGH. unchanged |
| $12,000, country `US` | 50 MEDIUM. large_amount 30, foreign_country 20 | 50 MEDIUM. unchanged |

`KP` and `US` are controls. They are not part of the bug. `KP` is still high-risk and foreign. `US` is still foreign and is not on the high-risk list.

`data/ronin_live_results.csv`, re-scored by the same function. 224 rows, country `UNKNOWN` on every row, OFAC match on every row.

- Before: all 224 `CRITICAL`. Scores 230 to 381, median 270. High-risk reason on all 224. Foreign points on 176. Foreign context only on 48.
- After: all 224 still `CRITICAL`. `OFAC_SDN_MATCH` is 200 points on every row. Scores 200 to 331, median 220. High-risk flags 0. Foreign reasons 0.

Examples (same transaction ids):

- `0xe0669bbaaa12cf5ecc682848ddc373a9b86e1351bccc01092b744099bf52a87d`, $2,068.39. Before 230 (OFAC 200 + high-risk amplified 30). After 200 (OFAC 200).
- `0x9f6a19c9fed374450dcb1657d5e404ec8a60be752b07c1db1f21049d7bf8d61a`, $65,693,880.51. Before 280 (large_amount 30 + foreign 20 + OFAC 200 + high-risk amplified 30). After 230 (large_amount 30 + OFAC 200).
- `0x431136dd361557abe34fe4685a278654e9e1bc7547a40719b348c096c5092d2b`, $1,957.03, layering cycle. Before 275 (layering 45 + OFAC 200 + high-risk amplified 30). After 245 (layering 45 + OFAC 200).

## Tests

Same commands as CI: `ruff check engine scripts tests` and `pytest tests/`.

- Before, on `main` at `f6ef659`: ruff passed. 166 passed, 1 skipped.
- After the two code commits: ruff passed. 166 passed, 1 skipped.

## Still unwired in the monitor

`detect_drainer_signature` and `detect_address_poisoning` are not called from `scripts/monitor.py`. The Streamlit app does call them. The drainer rule needs token fields the monitor data does not have.

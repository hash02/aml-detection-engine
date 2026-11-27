# Monitor fixes, November 2026

Scoring-path fixes for the daily monitor. Weights and thresholds are unchanged. `detect_address_poisoning` is still not called. `detect_drainer_signature` is called once the fetchers attach token fields. See the later sections.

## Four detectors were never called by the monitor

`scripts/monitor.py` `run_engine_on_df` is what `.github/workflows/aml-monitor.yml` runs. Commit `3b206e8` (PR 1, merged as `c750ea7` on 2026-04-16) added four detectors to `engine/engine_v11_blockchain.py`:

- `detect_phish_hit`
- `detect_sub_threshold_tranching`
- `detect_machine_cadence`
- `detect_sybil_fan_in`

That commit also edited `scripts/monitor.py`. It changed the docstring on `run_engine_on_df` from "22-rule engine" to "26-rule engine" and reformatted the import list. It did not add calls. `streamlit_app.py` and `scripts/backtest.py` do call the four functions, in that order, after `detect_layering_deep` and before `score_transactions`. `detect_phish_hit` loads the phishing feed and the no-KYC offramp feed itself through `get_feed`, so the monitor does not pass a separate list.

At the time of that investigation, the only later commit that touched `scripts/monitor.py` was `1ecfaff` (`ci: fix scheduled AML engine workflows`), which changes stdout encoding. A search of commits, issues, and pull requests found no recorded reason for leaving the four calls out. The comment was updated and the calls were not. That is consistent with the calls being forgotten.

The monitor now calls them the same way it calls the other `detect_*` functions, and the same way Streamlit does. The docstring no longer says "26-rule engine". It states what this function runs. `detect_phish_hit` still loads its own feeds.

`detect_drainer_signature` and `detect_address_poisoning` were left unwired in that change. The drainer detector returns immediately when `asset_type` is absent, and the fetched rows did not carry token fields. Address poisoning was out of scope. The drainer follow-up is below. Address poisoning is still unwired.

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

## Tests for the first two fixes

Same commands as CI: `ruff check engine scripts tests` and `pytest tests/`.

- Before, on `main` at `f6ef659`: ruff passed. 166 passed, 1 skipped.
- After wiring the four detectors and the unknown-country change: ruff passed. 166 passed, 1 skipped.

## Drainer rule can see token transfers

`detect_drainer_signature` no-ops when `asset_type` is missing. It then looks for two or more distinct non-empty `token_contract` values from the same sender to the same receiver inside `drainer_window_seconds` (120). `etherscan_fetcher.build_transaction_rows` and `scripts/blockscout_fetcher.build_rows_from_blockscout` only emitted native ETH rows, and they did not set `asset_type`. The daily monitor calls `fetch_live_data` (Etherscan, two watchlist addresses, `expand_hops=False`, limit 200) and `fetch_and_build` (Blockscout, one address). Both paths dropped the fields the rule needs.

`engine/adapters/ethereum.py` already calls Etherscan `tokentx`, and `fetch_blockscout_token_transfers` already exists, but neither is on the path `scripts/monitor.py` uses.

What changed, without rewriting the fetchers:

- Native rows from both fetchers now include `asset_type=native`, an empty `token_contract`, `token_symbol=ETH`, and `token_amount_decimal`. The column is present, so the rule runs. Empty contracts are ignored by the rule, so plain ETH transfers do not look like a multi-asset drain.
- With an Etherscan API key, `fetch_live_data` makes one extra `account/tokentx` call per seed address and appends ERC-20 rows. The row id is `hash:token_contract`, so two tokens that share a transaction hash both survive `deduplicate`, and they do not collide with the native row for that hash. Stablecoins in {USDC, USDT, DAI, FRAX, LUSD, TUSD, USDP} are priced at $1. WETH, STETH, and RETH use the ETH spot price. Other symbols are stored at $0. The rule does not look at the dollar amount.
- No API key: tokentx is skipped and native rows are unchanged apart from the new columns.
- Graph expansion does not call tokentx. The CLI default `expand_hops=True` can fetch up to 20 extra addresses. Adding tokentx there would be up to 20 more Etherscan calls. The monitor passes `expand_hops=False`.
- Blockscout `fetch_and_build` now also calls the existing token-transfer endpoint and keeps ERC-20 rows. That request is not an Etherscan call. ERC-721 rows are skipped. The id includes the contract and log index.

`scripts/monitor.py` calls `detect_drainer_signature` after the four v12 detectors and before `score_transactions`. The docstring now says 22 of the 23 `detect_*` functions.

### Call cost

Etherscan free tier is 5 calls per second and 100,000 calls per day. The monitor already does one `txlist` per Etherscan address. tokentx adds one call per seed address. The current watchlist has two Etherscan addresses, so the daily job adds two calls. `fetch_etherscan_tokentx` waits 0.25 seconds first, because `txlist` for that address just returned. The monitor also sleeps 1 second between watchlist addresses. Blockscout adds one paginated token-transfer fetch for the single Blockscout address, on Blockscout's own host.

A failed or empty tokentx response is logged and ignored. Native history still scores.

### Offline behavior

`tests/test_monitor_token_drain.py` uses fixtures. No network and no API key value.

- Two ERC-20 rows, same sender and receiver, 30 seconds apart, different contracts: `run_engine_on_df` sets `drainer_flag` on both rows and the reasons include `drainer_signature`.
- The same token twice: 0 flags.
- Two native ETH rows inside the window: 0 flags.
- Two different tokens 10 minutes apart: 0 flags.
- A mocked `fetch_live_data` merges a native row with two tokentx rows that share a hash. Both token contracts remain.
- With `api_key=""`, tokentx is not called.
- A Blockscout token-transfer fixture keeps the two ERC-20 rows and drops an ERC-721 row.

### Address poisoning stays unwired

`detect_address_poisoning` does not read `asset_type` or `token_contract`. It looks for a dust or zero-value transfer from a lookalike address. Both fetchers still drop zero-value transactions unless a known mixer is involved. Adding token columns does not make that rule correct, so it is not called.

## Per-run CSVs are kept as artifacts

`scripts/monitor.py` writes `data/monitoring/{label}_{DATE}.csv` for each watchlist address before scoring. `.gitignore` ignores `data/monitoring/*` and un-ignores only `alert-*.md`. The workflow commits `data/monitoring/` and `config/watchlist.json`, so git keeps the alert markdown and drops the CSVs when the runner is destroyed. That ignore was intentional: the comment says to publish the markdown alerts and not the transaction exports.

Size, from the committed alert for 2026-11-26 and from row width:

- That run scanned 320 transactions (Ronin 170, Bitfinex 0, structuring cluster 150).
- An input-shaped row is about 230 bytes. With the token columns it is about 300 bytes. 320 rows is on the order of 100 KB. If tokentx roughly doubles the two active addresses, a day is still a few hundred KB, not megabytes.

Committing the CSVs would fit in a commit. It would also add a raw transaction export to git every day, with no retention, and it would undo the ignore that keeps those exports out of the repo. A year of that is on the order of 100 MB of history.

The workflow uploads `data/monitoring/*.csv` with `actions/upload-artifact@v4`, name `monitoring-csvs`, `retention-days: 14`, `if-no-files-found: warn`, and `if: always()` so a later commit failure still keeps the files. The job's token permissions include `actions: write`, which upload needs when `permissions` is set explicitly. The CSVs stay gitignored. Alert markdown is still committed.

Tradeoff: the files are available for 14 days from the Actions run, then GitHub deletes them. They are not in the git history. Fourteen days covers looking back at a recent alert. It does not build a permanent archive.

## Tests after the drainer and artifact changes

Same CI commands.

- After the token-field and drainer commit: ruff passed. 172 passed, 1 skipped. The six new tests are in `tests/test_monitor_token_drain.py`.
- The artifact change is workflow YAML and a `.gitignore` comment. `tests/test_workflow_feed_paths.py` still passes. The YAML parses. The full suite is unchanged by that commit: 172 passed, 1 skipped.

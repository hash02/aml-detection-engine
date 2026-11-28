# Mixer touch fix, November 2026

Two label bugs kept known mixer transfers off `mixer_touch`. Weights and thresholds are unchanged. `MX_DARK` was not added to `high_risk_countries`.

## Root cause

Fetchers and the engine used different labels for the same fact.

- `etherscan_fetcher.infer_country` returned `XX` when the sender address was in `KNOWN_MIXERS`. It also set `is_known_mixer` when either the sender or the receiver was in that set.
- `scripts/blockscout_fetcher.py` does not define `infer_country`. It imports the etherscan function (line 38) and calls it for the sender. One function serves both fetchers.
- `detect_mixer_touch` selected mixer rows only with `df["country"].isin(MIXER_COUNTRY_CODES)`. That set is `{"MX_DARK"}`. It never read `is_known_mixer`.
- Country is inferred from the sender only. A transfer into a mixer (user → mixer) therefore kept the user's country, usually `UNKNOWN`, plus `is_known_mixer=True`. A transfer out of a mixer (mixer → user) was the only shape that received `XX`.

`XX` is in `CONFIG["high_risk_countries"]`. `MX_DARK` is not. `is_scored_foreign` returns false only for `UNKNOWN`. Every other code, including `XX` and `MX_DARK`, is foreign when `home_country` is `CA`. Foreign points (`w_foreign`, 20) are added only when a v6 signal is already present (large amount, velocity, structuring, or fan-in). Otherwise the reason is `foreign_context_only` and the points are 0.

So a fetched mixer withdrawal scored as a high-risk jurisdiction (`XX`, half of `w_high_risk_country` = 15 when alone) and as foreign context. It did not score as a mixer. A fetched mixer deposit scored as neither.

## Fix 1 — read `is_known_mixer`

`detect_mixer_touch` still treats `MX_DARK` as a mixer row. It also treats a truthy `is_known_mixer` as a mixer row.

The coercion defaults to false. A missing column, `None`, `NaN`, blank, `false` / `0` / `no`, and any unrecognized string stay false. `true` / `yes` / `1` (any case) and non-zero numbers are true. Bool columns use `fillna(False)` so a nullable missing value does not become true.

The first version of this fix still built one depositor set and one withdrawer set from every mixer row. That mislabeled deposits. Direction is now decided per row. See [Label fix](#label-fix-deposit-vs-withdrawal).

## Fix 2 — tag known mixers `MX_DARK`

`infer_country` returns `MX_DARK` for `KNOWN_MIXERS`. Bridges still return `BRIDGE`. An unrecognized address still returns `UNKNOWN`.

`XX` is still a member of `high_risk_countries`. The trailing comment on that entry now says the fetchers tag known mixers `MX_DARK`, and that `MX_DARK` is not in the set. The old comment called `XX` the mixer jurisdiction tag.

## Scores, measured

Both passes used `scripts.monitor.run_engine_on_df` (the daily monitor path) with `FEEDS_OFFLINE=1` and no API key. `engine.explain.score_breakdown` supplies the unrounded parts. `score_row` then multiplies by the profile multiplier and rounds with Python's half-to-even rule.

The known mixer sender is `0x12d66f87a04a9e220c9d35925b72aca3ca8c78e2` (in `KNOWN_MIXERS`, not in the bundled offline OFAC set, so `ofac_flag` stays false). Amount is $2,500, under the personal large-amount line of $10,000. `sender_tx_count` is 50 and `sender_active_days` / `account_age_days` are 365, so novel-dump and dormancy stay off. `classify_profile` returns `HIGH_RISK` for that mixer address. `HIGH_RISK` is not in `PROFILE_CONFIG`, so scoring uses the `PERSONAL_LIKE` multiplier of 1.0.

`w_mixer_touch` is 55. Withdraw is `55 * 1.3 = 71.5`, and `round(71.5)` is 72.

Before is `main` at `243a98d`. The After column is this branch after the flag and the `MX_DARK` tag, and before the direction fix below. The deposit row in that column is the mislabel.

| Sample | Before | After |
| --- | --- | --- |
| Mixer sends $2,500 to a receiver. Before: country `XX`, `is_known_mixer` true. After: country `MX_DARK` (what `infer_country` now returns), flag true. | 15 LOW. High-risk alone, 15. Foreign context, 0. `mixer_flag` false. Reason: `profile_HIGH_RISK;foreign_country;foreign_context_only;high_risk_jurisdiction;` | 72 HIGH. `mixer_withdraw` 71.5, rounded to 72. High-risk flag false (`MX_DARK` is not in the list). Foreign context, 0. Reason: `profile_HIGH_RISK;foreign_country;foreign_context_only;mixer_withdraw;` |
| Thirty minutes later, an unrelated wallet sends $2,500 to that same receiver. Country `US`, flag false. | 0 LOW. Foreign context, 0. The receiver was never collected as a withdrawer. Reason: `profile_PERSONAL_LIKE;foreign_country;foreign_context_only;` | 72 HIGH. `mixer_withdraw` 71.5, rounded to 72. The receiver of the mixer row is a withdrawer, so this later inbound payment is overwritten to withdraw. Foreign context, 0. |

That pair is the mixer-sender row and a later payment to its receiver. On an exit, the sender address is the mixer, so `mixer_withdraw` is the right label. The second row stays `mixer_withdraw` because that customer is a withdrawal recipient.

A deposit the fetcher actually emits (country comes from the non-mixer sender):

| Sample | Before | After |
| --- | --- | --- |
| User sends $2,500 to the mixer. Country `UNKNOWN`, `is_known_mixer` true, profile `PERSONAL_LIKE`. | 0 LOW. `UNKNOWN` is not foreign and not high-risk. Flag ignored. Reason: `profile_PERSONAL_LIKE;` | 72 HIGH. `mixer_withdraw` (overwrite), 72. Country stays `UNKNOWN` because the sender is not the mixer. Reason: `profile_PERSONAL_LIKE;mixer_withdraw;` |

`MX_DARK` means the sender is the mixer only when the receiver address is not itself a known mixer. A row whose receiver is a known mixer is a deposit even if the country column says `MX_DARK`. See the label fix.

### Already-fetched `XX` rows

A stored row that still says country `XX` with `is_known_mixer` true is re-scored by the new engine at 102 CRITICAL: `mixer_withdraw` 71.5 plus the amplified high-risk weight 30 (`mixer_flag` is an amplifying signal), `round(101.5) = 102`. Reason: `profile_HIGH_RISK;foreign_country;foreign_context_only;mixer_withdraw;high_risk_jurisdiction_amplified;`. New fetches no longer write `XX` for mixers, so a newly fetched withdrawal is the 72 row above, not this 102 row.

`XX` with the flag false still scores 15 LOW. That is the old high-risk-only path, and it still works.

## Label fix: deposit vs withdrawal

Sending into a mixer is placement. Receiving from a mixer is the layering exit. Both are red flags. The points stay separate: deposit adds `w_mixer_touch` (55). Withdrawal adds `55 * 1.3 = 71.5`, which rounds to 72 at a profile multiplier of 1.0. They are not merged into one score.

### Root cause

The first fix treated every mixer row as both a deposit and a withdrawal. Depositors were every sender on a mixer row. Withdrawers were every receiver on a mixer row. The withdrawal label was applied second, so it overwrote the row.

On a payment into a mixer the receiver is the mixer contract, so that contract landed in the withdrawer set. The deposit was relabeled `mixer_withdraw`. The same set membership made a later payment to the mixer contract look like a withdrawal. Mixer addresses are not customer wallets.

### How direction is decided

A known mixer address is an entry in the engine mixers feed (`get_feed("mixers")`), unioned with `etherscan_fetcher.KNOWN_MIXERS`. Those two sets are the same eight addresses.

1. The receiver address is a known mixer. The row is `mixer_deposit`. The sender is a depositor when that sender is not a mixer address.
2. The sender address is a known mixer. The row is `mixer_withdraw`. The receiver is a withdrawal recipient when that receiver is not a mixer address.
3. Country is `MX_DARK` and the receiver is not a known mixer. Country is inferred from the sender, so the sender is the mixer. The row is `mixer_withdraw`. That sender is not stored as a customer wallet.
4. `is_known_mixer` is truthy, neither address is a known mixer, and country is not `MX_DARK`. The row is `mixer_deposit`. Neither wallet is stored, because the side is unknown.

When both addresses are known mixers, the row is `mixer_withdraw` (funds left a mixer) and neither address is stored.

Later sends by a depositor stay `mixer_deposit`. Later receipts by a withdrawal recipient stay `mixer_withdraw`. If a row that is not itself a mixer contact is both, the withdrawal label wins. A direct send into a mixer stays `mixer_deposit`. Mixer addresses are never depositor wallets and never withdrawal-recipient wallets.

`cfg["mixer_window_hours"]` is still read and still unused.

### The three $2,500 cases

Same engine path and the same rows as the table above. Before in this table is the After column of that table (flag and `MX_DARK` tag on, direction still wrong). After is direction per row. `run_engine_on_df`, `FEEDS_OFFLINE=1`. The exit and the follow-up payment are one frame. The deposit is its own frame. Scoring the deposit by that same customer in the exit frame would also be a round trip, and layering would add points that are not the mixer term.

| Sample | Before (overwrite) | After (direction) |
| --- | --- | --- |
| Mixer sends $2,500. Country `MX_DARK`, flag true, profile `HIGH_RISK` (multiplier 1.0). | 72 HIGH. `mixer_withdraw`. Reason: `profile_HIGH_RISK;foreign_country;foreign_context_only;mixer_withdraw;` | 72 HIGH. `mixer_withdraw`. Same reason. The sender address is the mixer, so the label was already the exit. |
| Thirty minutes later, an unrelated wallet sends $2,500 to that receiver. Country `US`, flag false, profile `PERSONAL_LIKE`. | 72 HIGH. `mixer_withdraw`. Reason: `profile_PERSONAL_LIKE;foreign_country;foreign_context_only;mixer_withdraw;` | 72 HIGH. `mixer_withdraw`. Same reason. The customer is a withdrawal recipient. The mixer address is not. |
| User sends $2,500 to the mixer. Country `UNKNOWN`, flag true, profile `PERSONAL_LIKE`. | 72 HIGH. `mixer_withdraw` (the mixer contract was in the withdrawer set). Reason: `profile_PERSONAL_LIKE;mixer_withdraw;` | 55 MEDIUM. `mixer_deposit`. Reason: `profile_PERSONAL_LIKE;mixer_touch;` |

Foreign context on the first two rows is still 0. `UNKNOWN` on the deposit is still not foreign and not high-risk. A stored exit that still has country `XX`, flag true, and a mixer sender still scores 102 CRITICAL (`mixer_withdraw` 71.5 plus amplified high-risk 30). A deposit into the mixer no longer takes that withdrawal term.

An isolated check with profile `PERSONAL_LIKE`, country `CA` (home, so foreign adds 0), and $2,500: send-to-mixer scores 55, mixer-send scores 72. That is `test_deposit_scores_55_and_withdrawal_scores_72`.

## Recommendation, not done

Do not add `MX_DARK` to `high_risk_countries` unless a separate review wants the stack. On the $2,500 mixer-sender row, membership would turn the high-risk term on. `mixer_flag` would amplify it from 15 to 30. The row would score 102 instead of 72, the same total as a legacy `XX` row. That adds a jurisdiction weight on top of the mixer rule. This change leaves the set membership as it was.

`ai_layer/aml_ai_layer.py` has its own `HIGH_RISK_COUNTRIES` set, and that set already contains `MX_DARK`. It only fills `is_high_risk_country` on the feature frame. `score_row` does not read it. That file was not edited.

`scripts/generate_cases.py` can still draw country `XX` at random for a synthetic row. Those rows still take the high-risk path. That generator was not edited.

## `mixer_adjacent` removed from the docs

The detector docstring and `docs/RULES.md` (generated from that docstring) said a third tier existed: transacting with a contaminated wallet → `mixer_adjacent` (MEDIUM). The module header also said the rule fires inside a lookback window, and that "one hop" was its own HIGH tier. None of that was implemented.

The code flags a depositor's later sends and a withdrawal recipient's later receipts. It does not flag a wallet merely because it transacted with a contaminated wallet and is neither of those. Adding that tier would be a new detection claim and needs fixtures. It was removed from the header, the docstring, and `docs/RULES.md`. It was not implemented. The README ability list never named it.

## `mixer_window_hours` is still unused

`CONFIG["mixer_window_hours"]` is 24. `detect_mixer_touch` still does `window = timedelta(hours=cfg["mixer_window_hours"])` and never reads `window`. A depositor send 48 hours after the mixer row is still `mixer_deposit` (`test_mixer_window_hours_does_not_drop_older_rows`).

The lookback was not turned on. Applying it would drop older contacts, and the anchor (each row, the end of the frame, or the clock) is not defined. Behavior is the same as `243a98d`.

## Tests

`tests/test_mixer_touch.py` covers:

- a send to a known mixer is `mixer_deposit`, including string `true` / `1` / `yes` on a row whose addresses are not in the roster; `false`, `0`, blank, `nan`, and `None` do not flag that row
- a send from a known mixer is `mixer_withdraw`, and a later receipt by that customer stays `mixer_withdraw`
- the mixer address is not a withdrawal recipient: another payment into it stays `mixer_deposit`
- `MX_DARK` with a known-mixer receiver is a deposit; `MX_DARK` with neither address in the roster treats the sender as the mixer and does not store that sender as a customer
- a flag with an unknown side is `mixer_deposit` and does not contaminate either wallet
- profile `PERSONAL_LIKE`, country `CA`, $2,500: deposit scores 55 and withdrawal scores 72
- the 24 hour setting does not drop a 48 hour-later depositor send
- `infer_country` returns `MX_DARK` for every `KNOWN_MIXERS` address, `UNKNOWN` for an unrecognized address, `BRIDGE` for a known bridge, and the blockscout import is the same function object

Full suite, `FEEDS_OFFLINE=1`, `pytest tests/` and `ruff check engine scripts tests` (the CI lint paths).

- Before the flag and tag fixes, on `main` at `243a98d`: ruff passed. 172 passed, 1 skipped, 0 failed.
- Before the direction fix, on this branch at `36a950a`: ruff passed. 182 passed, 1 skipped, 0 failed.
- After the direction fix: ruff passed. 184 passed, 1 skipped, 0 failed.

The skip is `tests/test_graph_features.py`, which skips when `networkx` is not installed. Both runs skipped that one test.

# AML scoring rules

28 rules, in `score_row` order. Twenty-three are `detect_*` functions. Five are applied inside `score_row` from window features and have no `detect_*` function: `large_amount`, `velocity`, `structuring`, `fan_in`, and `foreign_country`.

Profile multipliers apply to the total after the rule points are summed, then the result is rounded:

- `NEW`: 1.20 (`engine/engine_v11_blockchain.py:351`)
- `PERSONAL_LIKE` (default): 1.00 (`:360`)
- `BUSINESS_LIKE`: 0.85 (`:369`)

Some rules scale their base weight before that multiplier. The base `CONFIG` value is listed below. Code lines are in `engine/engine_v11_blockchain.py` unless another file is named.

Who runs each rule:

- Daily monitor: `scripts/monitor.py` `run_engine_on_df`. 27 of 28. It does not call `detect_address_poisoning`.
- Streamlit: `streamlit_app.py` `run_engine`. All 28.
- Pipeline: `pipeline.py` `apply_rules`. All 28.
- Backtest: `scripts/backtest.py` `run_full_pipeline`. 26 of 28. It does not call `detect_drainer_signature` or `detect_address_poisoning`.

The engine CLI (`main`) also calls all 28.

### 1. large_amount

Checks whether `amount` is at least the sender profile's `large_amount_threshold`.

Weight: `w_large` = 30 (`:84`). Thresholds: NEW $8,000 (`:346`), PERSONAL_LIKE $10,000 (`:355`), BUSINESS_LIKE $50,000 (`:364`).

Scored at `:1844`. No `detect_*` function.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 2. velocity

Checks whether `tx_count_in_window` is at least the profile's `velocity_tx_count_threshold` (60 minute window).

Weight: `w_velocity` = 25 (`:85`).

Scored at `:1846`. No `detect_*` function. The count is built in `compute_features`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 3. structuring

Checks for enough small transactions in a 60 minute window or a 6 hour window, using the profile's `struct_min_count` and `struct_min_count_6h`.

Weight: `w_structuring` = 35 (`:87`).

Scored at `:1851`. No `detect_*` function.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 4. fan_in

Checks whether `fan_in_count` is at least `fan_in_threshold`.

Weight: `w_fan_in` = 20 (`:86`).

Scored at `:1853`. No `detect_*` function.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 5. foreign_country

Checks whether the country code is a known jurisdiction other than `home_country` (`CA`, `:81`). `UNKNOWN` is not foreign (`is_scored_foreign`, `:1825`). Points are added only when another v6 signal already fired. Otherwise the reason is `foreign_context_only` and the points are 0.

Weight: `w_foreign` = 20 (`:88`), added at `:1865`.

Scored at `:1862`. No `detect_*` function.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 6. layering

Checks for a circular flow (A back to A) inside the layering window. Reason tag: `layering_cycle`.

Weight: `w_layering` = 45 (`:89`).

Detector: `detect_layering` `:465`. Scored at `:1869`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 7. mixer_touch / mixer_withdraw

Checks whether the row touches a known mixer. A withdrawal (received from a mixer) uses a higher multiplier than a deposit.

Weight: `w_mixer_touch` = 55 (`:92`). Withdrawal is 1.3 times that (`:1878`). Deposit uses the base (`:1881`).

Detector: `detect_mixer_touch` `:522`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 8. bridge_hop

Checks for several distinct bridge contracts used by one sender inside the bridge window.

Weight: `w_bridge_hop` = 40 (`:93`), scaled by hop count up to 2 times at `:1887`.

Detector: `detect_bridge_hops` `:570`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 9. peel_chain

Checks for a linear peel (each hop forwards most of the amount) of at least the configured hop count.

Weight: `w_peel_chain` = 45 (`:94`).

Detector: `detect_peel_chain` `:639`. Scored at `:1892`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 10. OFAC_SDN_MATCH

Checks whether the sender or receiver is on the OFAC SDN feed or the bundled sanctioned-address set.

Weight: `w_ofac_hit` = 200 (`:115`). Added to the score. It is not a cap.

Detector: `detect_ofac_hit` `:772`. Scored at `:1899`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 11. flash_loan_burst

Checks for many large transactions from one sender inside a short second-scale window.

Weight: `w_flash_loan` = 80 (`:122`), scaled by burst count up to 2 times. Scored at `:1906`.

Detector: `detect_flash_loan_burst` `:1146`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 12. coordinated_burst

Checks for several senders paying the same receiver inside a few seconds.

Weight: `w_coordinated_burst` = 70 (`:128`), scaled by sender count up to 2 times. Scored at `:1912`.

Detector: `detect_coordinated_burst` `:1194`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 13. novel_wallet_dump

Checks for a young, low-history wallet moving at least `novel_dump_min_amount`.

Weight: `w_novel_dump` = 65 (`:101`), scaled by how new the wallet is (down to 0.5). Scored at `:1919`.

Detector: `detect_novel_wallet_dump` `:1242`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 14. concentrated_inflow

Checks for many senders paying one receiver with meaningful amounts inside a longer window than `coordinated_burst`.

Weight: `w_concentrated_inflow` = 50 (`:107`), scaled by sender count up to 2 times. Scored at `:1926`.

Detector: `detect_concentrated_inflow` `:1273`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 15. dormant_activation

Checks for a wallet inactive for at least `dormant_min_days` that then moves more than `dormant_min_amount`.

Weight: `w_dormant_activation` = 75 (`:137`), scaled by years dormant. Scored at `:1935`.

Detector: `detect_dormant_activation` `:1330`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 16. wash_cycle

Checks for a round trip between two wallets with similar amounts inside the wash window.

Weight: `w_wash_cycle` = 60 (`:144`). Scored at `:1941`.

Detector: `detect_wash_cycle` `:1407`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 17. smurfing

Checks for several wallets each sending just under the reporting limit to the same receiver inside the smurf window.

Weight: `w_smurfing` = 55 (`:150`), scaled by wallet count up to 2 times. Scored at `:1948`.

Detector: `detect_smurfing` `:1470`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 18. exit_rush

Checks for a new wallet that receives a large amount and soon sends it to a bridge or exchange.

Weight: `w_exit_rush` = 65 (`:156`). Scored at `:1953`.

Detector: `detect_exit_rush` `:1532`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 19. rapid_succession

Checks for one sender paying many distinct receivers inside a few minutes, each above `rapid_min_amount`.

Weight: `w_rapid_succession` = 50 (`:162`), scaled by receiver count up to 2 times. Scored at `:1960`.

Detector: `detect_rapid_succession` `:1597`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 20. high_risk_country

Checks whether `country` is in `high_risk_countries` (weight at `:167`, set at `:249`). `UNKNOWN` is not in that set. `XX` (the mixer tag) is.

Weight: `w_high_risk_country` = 30 (`:167`). Full weight when another listed signal is present (`:1973`). Half weight (15) alone (`:1976`).

Detector: `detect_high_risk_country` `:1650`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 21. exchange_avoidance

Checks for a path of at least `exchange_avoidance_min_hops` non-exchange hops before an exchange.

Weight: `w_exchange_avoidance` = 45 (`:172`), scaled by hop count up to 2 times. Scored at `:1983`.

Detector: `detect_exchange_avoidance` `:1755`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 22. layering_deep

Checks for a longer peel than `peel_chain` (at least `deep_peel_min_hops`, over a longer window).

Weight: `w_layering_deep` = 70 (`:177`), scaled by depth up to 2 times. Scored at `:1990`.

Detector: `detect_layering_deep` `:1678`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 23. phishing_hit

Checks whether the sender or receiver is on the phishing feed or the no-KYC offramp feed.

Weight: `w_phishing_hit` = 50 (`:276`). Scored at `:1995`.

Detector: `detect_phish_hit` `:826`. The detector loads both feeds itself.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 24. sub_threshold_tranching

Checks for at least `tranching_min_count` payments from one sender in the 80 to 100 percent band under the reporting limit, inside `tranching_window_hours`.

Weight: `w_sub_threshold_tranching` = 60 (`:284`), scaled by tranche count up to 2 times. Scored at `:2001`.

Detector: `detect_sub_threshold_tranching` `:865`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 25. machine_cadence

Checks whether a sender's recent inter-arrival times have a coefficient of variation under `cadence_max_cv`, with at least `cadence_min_tx` transactions.

Weight: `w_machine_cadence` = 45 (`:289`). Scored at `:2005`.

Detector: `detect_machine_cadence` `:911`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 26. sybil_fan_in

Checks for at least `sybil_min_senders` distinct senders paying one receiver inside `sybil_window_minutes`, with amounts inside a tight band.

Weight: `w_sybil_fan_in` = 55 (`:295`), scaled by sender count up to 2 times. Scored at `:2011`.

Detector: `detect_sybil_fan_in` `:957`.

Runs in: daily monitor, Streamlit, pipeline, backtest.

### 27. drainer_signature

Checks for at least `drainer_min_assets` distinct non-empty `token_contract` values from the same sender to the same receiver inside `drainer_window_seconds` (120). If `asset_type` is missing, the detector flags nothing.

Weight: `w_drainer_signature` = 70 (`:318`), scaled by asset count up to 2 times. Scored at `:2018`.

Detector: `detect_drainer_signature` `:1009`.

Runs in: daily monitor, Streamlit, pipeline. Not in backtest.

### 28. address_poisoning

Checks for a dust or zero-value transfer whose sender matches the first 4 and last 4 hex characters of a recent real counterparty, while the full address differs.

Weight: `w_address_poisoning` = 50 (`:324`). Scored at `:2023`.

Detector: `detect_address_poisoning` `:1066`.

Runs in: Streamlit, pipeline. Not in the daily monitor. Not in backtest.

## Known gaps

`address_poisoning` is defined and scored when its flag is set, and Streamlit and `pipeline.py` call the detector. The daily monitor does not call it. That is a known gap, not a hidden one. The fetchers still drop zero-value transfers unless a known mixer is involved, so the dust rows this rule needs are not in the monitor's data even if the call were added.

`scripts/backtest.py` also does not call `detect_drainer_signature` or `detect_address_poisoning`.

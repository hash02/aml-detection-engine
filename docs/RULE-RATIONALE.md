# Rule rationale

One stub per scoring rule, in `score_row` order. Behavior is the current check. `Rationale / origin` is empty except for `large_amount`, which is the example. Weights and line numbers are in [docs/RULES.md](RULES.md).

### large_amount

Name: `large_amount`

Current behavior: Adds `w_large` (30) when the amount is at least the sender profile threshold. Thresholds are $8,000 for `NEW`, $10,000 for `PERSONAL_LIKE` (the default), and $50,000 for `BUSINESS_LIKE`. The profile multiplier (1.20, 1.00, or 0.85) applies to the whole score, not to this rule alone.

Rationale / origin: Present since the first commit, `f5b3456` (2026-02-27). That commit's sample file had 30 rows: 20 labeled `SYNTH_FORENSIC` and 10 labeled `REAL_DATA`. The same commit already had `w_large` 30 and those three thresholds. Hash recalls that the rule started in early synthetic transaction work. The $10,000 default mirrors the large cash transaction reporting threshold he works with in banking. That last sentence is his recollection, not something this file derived from a named policy document.

### velocity

Name: `velocity`

Current behavior: Adds `w_velocity` (25) when the sender's transaction count in the 60 minute window meets the profile threshold.

Rationale / origin:

### structuring

Name: `structuring`

Current behavior: Adds `w_structuring` (35) when enough small payments land in the 60 minute window or the 6 hour window.

Rationale / origin:

### fan_in

Name: `fan_in`

Current behavior: Adds `w_fan_in` (20) when enough distinct senders pay the same receiver inside the velocity window.

Rationale / origin:

### foreign_country

Name: `foreign_country`

Current behavior: Adds `w_foreign` (20) when the country is a known code other than `CA` and another v6 signal already fired. `UNKNOWN` does not count as foreign. Alone, the reason is context only and the points are 0.

Rationale / origin:

### layering

Name: `layering`

Current behavior: `detect_layering` flags a circular flow. `score_row` then adds `w_layering` (45) and tags `layering_cycle`.

Rationale / origin:

### mixer_touch

Name: `mixer_touch` / `mixer_withdraw`

Current behavior: `detect_mixer_touch` flags a known mixer counterparty. A withdrawal scores 1.3 times `w_mixer_touch` (55). A deposit scores the base weight.

Rationale / origin:

### bridge_hop

Name: `bridge_hop`

Current behavior: `detect_bridge_hops` flags several distinct bridges in the window. The score starts from `w_bridge_hop` (40) and scales with hop count, up to double.

Rationale / origin:

### peel_chain

Name: `peel_chain`

Current behavior: `detect_peel_chain` flags a linear peel. `score_row` adds `w_peel_chain` (45).

Rationale / origin:

### OFAC_SDN_MATCH

Name: `OFAC_SDN_MATCH`

Current behavior: `detect_ofac_hit` flags a sender or receiver on the sanctions set. `score_row` adds `w_ofac_hit` (200). The points are added. They do not replace the rest of the score.

Rationale / origin:

### flash_loan_burst

Name: `flash_loan_burst`

Current behavior: `detect_flash_loan_burst` flags a tight burst of large payments. The score starts from `w_flash_loan` (80) and scales with the burst count, up to double.

Rationale / origin:

### coordinated_burst

Name: `coordinated_burst`

Current behavior: `detect_coordinated_burst` flags several senders paying one receiver within seconds. The score starts from `w_coordinated_burst` (70) and scales with the sender count, up to double.

Rationale / origin:

### novel_wallet_dump

Name: `novel_wallet_dump`

Current behavior: `detect_novel_wallet_dump` flags a young, low history wallet moving a large amount. The score starts from `w_novel_dump` (65) and scales down with wallet age, to half at the oldest edge of the window.

Rationale / origin:

### concentrated_inflow

Name: `concentrated_inflow`

Current behavior: `detect_concentrated_inflow` flags many senders paying one receiver. The score starts from `w_concentrated_inflow` (50) and scales with the sender count, up to double.

Rationale / origin:

### dormant_activation

Name: `dormant_activation`

Current behavior: `detect_dormant_activation` flags a long inactive wallet that then moves a large amount. The score starts from `w_dormant_activation` (75) and scales with years dormant.

Rationale / origin:

### wash_cycle

Name: `wash_cycle`

Current behavior: `detect_wash_cycle` flags a round trip of similar amounts. `score_row` adds `w_wash_cycle` (60).

Rationale / origin:

### smurfing

Name: `smurfing`

Current behavior: `detect_smurfing` flags several wallets each sending just under the reporting limit to one receiver. The score starts from `w_smurfing` (55) and scales with the wallet count, up to double.

Rationale / origin:

### exit_rush

Name: `exit_rush`

Current behavior: `detect_exit_rush` flags a new wallet that receives a large amount and soon sends it onward to a bridge or exchange. `score_row` adds `w_exit_rush` (65).

Rationale / origin:

### rapid_succession

Name: `rapid_succession`

Current behavior: `detect_rapid_succession` flags one sender paying many receivers in a few minutes. The score starts from `w_rapid_succession` (50) and scales with the receiver count, up to double.

Rationale / origin:

### high_risk_country

Name: `high_risk_country`

Current behavior: `detect_high_risk_country` flags a country in the FATF high risk set. `UNKNOWN` is not in that set. Full `w_high_risk_country` (30) when another listed signal is present, otherwise half (15).

Rationale / origin:

### exchange_avoidance

Name: `exchange_avoidance`

Current behavior: `detect_exchange_avoidance` flags a long path of non exchange hops before an exchange. The score starts from `w_exchange_avoidance` (45) and scales with hop count, up to double.

Rationale / origin:

### layering_deep

Name: `layering_deep`

Current behavior: `detect_layering_deep` flags a longer peel than `peel_chain`. The score starts from `w_layering_deep` (70) and scales with depth, up to double.

Rationale / origin:

### phishing_hit

Name: `phishing_hit`

Current behavior: `detect_phish_hit` flags a sender or receiver on the phishing feed or the no KYC offramp feed. `score_row` adds `w_phishing_hit` (50).

Rationale / origin:

### sub_threshold_tranching

Name: `sub_threshold_tranching`

Current behavior: `detect_sub_threshold_tranching` flags repeated payments just under the reporting limit from one sender. The score starts from `w_sub_threshold_tranching` (60) and scales with the tranche count, up to double.

Rationale / origin:

### machine_cadence

Name: `machine_cadence`

Current behavior: `detect_machine_cadence` flags inter arrival times that are too regular for a person. `score_row` adds `w_machine_cadence` (45).

Rationale / origin:

### sybil_fan_in

Name: `sybil_fan_in`

Current behavior: `detect_sybil_fan_in` flags many senders paying one receiver similar amounts in a short window. The score starts from `w_sybil_fan_in` (55) and scales with the sender count, up to double.

Rationale / origin:

### drainer_signature

Name: `drainer_signature`

Current behavior: `detect_drainer_signature` flags two or more distinct token contracts moving from one sender to one receiver inside two minutes. It flags nothing when `asset_type` is missing. The score starts from `w_drainer_signature` (70) and scales with the asset count, up to double.

Rationale / origin:

### address_poisoning

Name: `address_poisoning`

Current behavior: `detect_address_poisoning` flags dust from a lookalike of a recent real counterparty. `score_row` adds `w_address_poisoning` (50). The daily monitor does not call this detector.

Rationale / origin:

# AML Rule Catalogue

> Auto-generated from docstrings in `engine/engine_v11_blockchain.py`.
> Do not edit by hand — run `python scripts/gen_rule_catalog.py` to refresh.

**23 detectors** are wired into the engine.

---

### `detect_address_poisoning` — Address Poisoning

```text
ABILITY 28 — ADDRESS POISONING  (v13)
─────────────────────────────────────────────────────────────────
A 0-value / dust transaction appears from a lookalike address —
one whose first-4 and last-4 hex characters match a known recent
counterparty of the target. This is the prelude to a clipboard-swap
attack: the victim later copy-pastes the lookalike from their tx
history and sends real funds to the attacker.

Detection logic:
  1. Build the set of "legit" counterparties per wallet from any
     non-dust, non-poison transaction in the past `poison_window_hours`.
  2. Flag any incoming tx whose amount ≤ `poison_dust_max_amount`
     AND whose sender's first-4 + last-4 chars match any legit
     counterparty's — while the full address is different.
```

### `detect_bridge_hops` — Bridge Hops

```text
For each sender, count distinct bridge contracts used within bridge_window_hours.
If count >= bridge_hop_threshold → flag.

Bridge rows are identified by country == BRIDGE_OFFSHORE.
Each unique receiver_id in a bridge row = a different bridge contract.

3+ bridges in 6hrs = regulatory escape attempt.
```

### `detect_concentrated_inflow` — Concentrated Inflow

```text
ABILITY 11 — CONCENTRATED INFLOW  (catches Lazarus sweep pattern)
────────────────────────────────────────────────────────────────────
Pattern: A single receiver address collects meaningful amounts
from 3+ different senders within a short time window.

Why it works:
  After a hack, proceeds are spread across many wallets to avoid
  detection, then swept back to a collection address before
  hitting an exchange. This sweep-to-collector is detectable
  as a fan-in to one receiver from many senders.

Different from fan_in (which looks at SENDER receiving many small
inflows) — this looks at the RECEIVER being a collection point.

Real-world match: Lazarus/Stake.com sweep wallets.
```

### `detect_coordinated_burst` — Coordinated Burst

```text
ABILITY 14 — COORDINATED BURST  (bot networks, coordinated attacks)
────────────────────────────────────────────────────────────────────
Multiple DIFFERENT wallets sending to the SAME receiver within 30
seconds — near-impossible to orchestrate manually. Indicates either
a bot network (Lazarus distributes then sweeps), a coordinated rug
pull, or automated exploit distribution.

Difference from concentrated_inflow (6-hour window):
this is a 30-SECOND window — simultaneous coordination, not gradual sweep.
```

### `detect_dormant_activation` — Dormant Activation

```text
ABILITY 15 — DORMANT WALLET ACTIVATION  (catches BitFinex / Silk Road pattern)
─────────────────────────────────────────────────────────────────────────────────
Pattern: A wallet that has been completely inactive for 365+ days
suddenly initiates a large transaction.

Why it works:
  Legitimate users don't store funds in a wallet for years and then
  suddenly move large amounts without any prior warm-up activity.
  The only common cases where this happens:
    1. Hackers waiting for investigations to go cold (BitFinex 2016→2022)
    2. Silk Road / seized funds being moved by authorities
    3. Long-lost keys being recovered (rare, usually small amounts)
    4. Criminal cold storage reactivation for cashout

  The "amount guard" (> $100k default) prevents false positives from
  hobbyists rediscovering old $50 ETH wallets.

  Score multiplier: dormancy years beyond 1 adds compounding signal.
  A wallet dormant for 5 years moving $1M = near-certain CRITICAL.

Real-world match: BitFinex (6yr dormancy), Silk Road addresses,
                  Wormhole Oasis recovery (1yr), Lazarus long-term holds.

Key column used: `sender_active_days`
  In our dataset this represents how long the wallet has been active
  (days between first and last observed transaction). When = 0 or very low
  but the wallet also has an old `account_age_days`, it signals dormancy.
  For the synthetic data, DORMANT_REACTIVATED account_type is also checked.
```

### `detect_drainer_signature` — Drainer Signature

```text
ABILITY 27 — DRAINER SIGNATURE FLOW  (v13)
─────────────────────────────────────────────────────────────────
Canonical Inferno / Angel / Pink Drainer pattern:
  1. Victim signs an approval (or `permit`) giving a drainer
     contract allowance over their assets.
  2. Within seconds-to-minutes, a distinct receiver drains several
     assets from the victim to an aggregator wallet.

We can't see signatures in the canonical schema, but the observable
post-approval fingerprint is unmistakable: the sender starts bleeding
multiple assets to the same receiver inside a 2-minute window.
Requires the v13 `asset_type` / `token_contract` columns; gracefully
no-ops when they're absent (older inputs).
```

### `detect_exchange_avoidance` — Exchange Avoidance

```text
ABILITY 21 — EXCHANGE AVOIDANCE  (deliberate routing around exchanges)
───────────────────────────────────────────────────────────────────────
Pattern: A wallet routes funds through 4+ intermediate addresses
before finally hitting an exchange, when a direct path would have
been much simpler. The deliberate complexity is the signal.

Why it works:
  Innocent users send directly to exchanges. Launderers add hops
  to reduce the traceability of the source. Each added hop is
  evidence of intent to obscure.

Implementation: Track chains where no intermediate node is a known
exchange address, but the terminal node is. Score scales with hop count.
```

### `detect_exit_rush` — Exit Rush

```text
ABILITY 18 — EXIT RUSH  (receive-then-immediately-bridge)
──────────────────────────────────────────────────────────
Pattern: A relatively new wallet receives a large amount,
then within 2 hours sends it to a bridge contract or a
known exchange address.

Why it works:
  Legitimate users don't immediately bridge everything they receive.
  The rush to exit the chain — especially via bridge — signals
  the wallet is a relay node in a multi-hop laundering chain.

Signature: novel wallet + large receive + fast bridge/exchange exit.
This is Group 5 (bridge exploit) post-hack movement.

Uses: is_bridge column OR known_exchange_prefixes in receiver address.
```

### `detect_flash_loan_burst` — Flash Loan Burst

```text
ABILITY 13 — FLASH LOAN BURST  (catches Euler-style attacks)
──────────────────────────────────────────────────────────────
Flash loan pattern: borrow → attack → repay in a single Ethereum block
(~12 seconds). A wallet doing 5+ meaningful transactions within 60
seconds is likely a bot or flash loan attacker, not a human.

Different from velocity_many_tx (which uses a 60-minute window):
this is a SECONDS-level burst — humanly impossible to generate manually.

Euler Finance: $197M taken in 2 transactions in the same block.
```

### `detect_high_risk_country` — High Risk Country

```text
ABILITY 20 — HIGH-RISK JURISDICTION  (FATF blacklist/grey list)
────────────────────────────────────────────────────────────────
Pattern: Transaction involves a country on the FATF blacklist
or grey list — jurisdictions with known AML/CFT deficiencies.

This is a weak standalone signal but a strong amplifier.
Real FATF-country transactions escalate existing suspicions.

FATF High-Risk Countries (2024 list):
Blacklist: North Korea (KP), Iran (IR), Myanmar (MM)
Grey list: Afghanistan, Albania, Barbados, Burkina Faso, etc.
```

### `detect_layering` — Layering

```text
(no docstring)
```

### `detect_layering_deep` — Layering Deep

```text
ABILITY 22 — DEEP LAYERING  (5+ hop peel chain)
─────────────────────────────────────────────────
Extended version of peel_chain that looks for longer laundering
trails — 5+ hops over 48 hours. Standard peel_chain fires at 3 hops;
this catches slow, patient laundering like BitFinex or Silk Road.

Pattern: Each hop sends ~85-99% of amount forward, keeping a small
slice (the "peel"). The pattern persists over longer time windows.

Score is higher than standard peel because the attacker is clearly
attempting to be patient and methodical — that's sophisticated intent.
```

### `detect_machine_cadence` — Machine Cadence

```text
ABILITY 25 — MACHINE CADENCE
─────────────────────────────────────────────────────────────────
Flags senders whose inter-arrival-time coefficient of variation is
below `cadence_max_cv` (default 5%). Humans don't maintain sub-5%
precision across many transactions; bots / scheduled jobs do.
```

### `detect_mixer_touch` — Mixer Touch

```text
A wallet is mixer-contaminated if:
  - It sent to a mixer country code (deposit), OR
  - It received from a mixer country code (withdrawal)
within the mixer_window_hours lookback.

We build a set of contaminated sender_ids and receiver_ids,
then flag every row where those wallets appear.

Score logic:
  - Deposited INTO mixer  → mixer_deposit  (HIGH)
  - Received FROM mixer   → mixer_withdraw (CRITICAL — clean money exits here)
  - Transacted WITH a contaminated wallet → mixer_adjacent (MEDIUM)
```

### `detect_novel_wallet_dump` — Novel Wallet Dump

```text
ABILITY 10 — NOVEL WALLET DUMP  (catches Ronin / Bybit style)
────────────────────────────────────────────────────────────────
Pattern: A wallet with almost no transaction history suddenly
moves an extremely large amount in one shot.

Why it works:
  Exploit wallets are purpose-built. They're created (or dormant),
  loaded with stolen funds from one event, then immediately distribute.
  They don't have years of normal transaction history.

Signal: sender_tx_count < threshold AND sender_active_days < threshold
        AND amount > novel_dump_min_amount

Real-world match: Ronin exploiter (0x098B71...) — OFAC SDN, $625M.
```

### `detect_ofac_hit` — Ofac Hit

```text
ABILITY 12 — OFAC SDN HIT  (hard mandatory alert, no threshold)
─────────────────────────────────────────────────────────────────
In real AML: OFAC screening happens BEFORE any ML or rules.
If you touch a sanctioned address — as sender OR receiver — it is
an automatic Suspicious Activity Report (SAR). No score needed.

v12: the sanctions set is loaded from engine.feeds.get_feed("ofac_sdn")
which merges the bundled baseline with the last refreshed copy of the
0xB10C OFAC mirror. If an ETH RPC URL is configured
(`CHAINALYSIS_RPC_URL` env var), non-hits are also verified against
the on-chain Chainalysis Sanctions Oracle for belt-and-braces.
```

### `detect_peel_chain` — Peel Chain

```text
Peel chain signature:
  - Wallet A sends amount X to wallet B
  - Wallet B soon sends ~X * (1 - peel%) to wallet C (unique C)
  - Wallet C soon sends ~X * (1 - peel%)^2 to wallet D
  - Repeat for min_hops

Each intermediate wallet is used ONCE (one in, one out).
The amount decreases monotonically but slowly (1-12% per hop).

Algorithm:
  1. Build a "forward pass" graph: for each wallet, find their
     single outgoing tx that closely matches their single incoming tx
  2. Walk chains: if a wallet has exactly 1 in + 1 out and
     out_amount ≈ in_amount * (1 - peel%), add to chain
  3. Flag chains of length >= peel_min_hops
```

### `detect_phish_hit` — Phish Hit

```text
ABILITY 23 — PHISHING / DRAINER / NO-KYC OFF-RAMP HIT
─────────────────────────────────────────────────────────────────
Sender or receiver appears in:
  - MetaMask eth-phishing-detect addresses feed
  - Analyst-curated drainer / no-KYC offramp list (feeds.lrt_offramps)

Unlike OFAC, these are not legal facts — they're crowd-sourced threat
intel. Weighted below OFAC but enough to push a tx into HIGH alone.
```

### `detect_rapid_succession` — Rapid Succession

```text
ABILITY 19 — RAPID SUCCESSION FAN-OUT  (bot-driven distribution)
──────────────────────────────────────────────────────────────────
Pattern: The same sender hits 5+ DIFFERENT receivers within 5 minutes.

Different from velocity (which counts total transactions in window).
This specifically tracks UNIQUE RECEIVER COUNT — the fan-out width.

Why it works:
  Manual users don't send to 5+ different wallets in 5 minutes.
  This is bot signature — automated distribution of stolen funds
  or coordinated payout to mule wallets.

Catches: Lazarus-style fan-out after initial exploit, ransomware
         payment distribution, coordinated market manipulation payouts.
```

### `detect_smurfing` — Smurfing

```text
ABILITY 17 — SMURFING  (coordinated threshold avoidance)
─────────────────────────────────────────────────────────
Pattern: Multiple different wallets each send amounts just BELOW
the AML reporting threshold to the same receiver, coordinated
within a short time window.

Named after the Smurfs (many small actors working in coordination
to accomplish what one actor can't do visibly).

Why it works:
  AML regulations require reporting transactions above a threshold
  (e.g., $10,000 USD). Criminals deliberately structure payments
  just under this limit. The coordination across wallets is detectable
  even when each individual transaction looks clean.

Key insight: velocity looks at one sender. Smurfing looks at
many senders all targeting the same receiver at the same threshold.
```

### `detect_sub_threshold_tranching` — Sub Threshold Tranching

```text
ABILITY 24 — SUB-THRESHOLD TRANCHING
─────────────────────────────────────────────────────────────────
Catches the classic human-operated structuring pattern: 3+ txns
bunched just below the $10k reporting threshold (80%–100% band)
from the same sender within 24 hours. Different from v6 structuring
(which uses a 60-min window and any-small amount) — this is the
slow, deliberate, reporting-aware variant.
```

### `detect_sybil_fan_in` — Sybil Fan In

```text
ABILITY 26 — SYBIL FAN-IN
─────────────────────────────────────────────────────────────────
6+ distinct senders each forwarding amounts within a 5% band to the
same receiver inside 30 minutes. Matches airdrop-farming sybils,
drainer collection wallets, and exchange-account-splitting schemes.
Distinct from fan_in (v6: any sender count with no amount band) and
concentrated_inflow (v9: slower, larger amounts, wider tolerance).
```

### `detect_wash_cycle` — Wash Cycle

```text
ABILITY 16 — WASH CYCLE  (round-trip / circular flow detection)
────────────────────────────────────────────────────────────────
Pattern: Wallet A sends funds to Wallet B, then Wallet B sends
a similar amount back to Wallet A within 24 hours.

Also catches A→B→C→A three-party cycles.

Why it works:
  Money doesn't legitimately round-trip at scale. If you send
  $500k to someone and they send $475k back within hours,
  that's not commerce — that's accounting manipulation.

Used for: Wash trading, NFT price inflation, fake volume,
          layering to create a paper trail of "legitimate" transfers.

Detection:
  Build a sender→receiver map. For each transaction A→B,
  look for B→A within the window with amount within tolerance.
```

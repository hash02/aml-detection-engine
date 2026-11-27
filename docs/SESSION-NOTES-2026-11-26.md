# Session notes, 2026-11-26

Factual notes for Hash. Numbers that this session did not recheck are marked.

## a) Merges

`hash02/aml-detection-engine` pull request 27 is merged. The merge commit on `main` is `243a98d` (`243a98dfca5694e5cd7a2eca92f0b9d5169e7d7c`). This session confirmed that SHA on `origin/main`.

`hash02/hash-2026` pull request 24 is recorded as merged into `codex/solana-alive-lab-v0` as `75105a3`. Hash's record says GitHub stored no CI check runs on that commit, and that the 25 passing tests were a local run. This session could not read `hash-2026` (the remote answered repository not found), so that SHA, the missing CI runs, and the local test count were not rechecked here.

## b) AML fixes

Pull request 27 is summarized in [docs/monitor-fixes-2026-11.md](monitor-fixes-2026-11.md).

The four v12 rules (`detect_phish_hit`, `detect_sub_threshold_tranching`, `detect_machine_cadence`, `detect_sybil_fan_in`) are wired into the daily monitor. They had been added in pull request 1 and called from Streamlit, but `scripts/monitor.py` only changed a comment.

`UNKNOWN` is no longer treated as a high risk country or as foreign. An ordinary $12,000 transfer with country `UNKNOWN` scored 65 before and 30 after. Ronin rows stay `CRITICAL`. Their scores moved from 230 to 381 down to 200 to 331. `OFAC_SDN_MATCH` stays +200 and is added to the score. It is not a cap.

The drainer rule now has token fields. With an Etherscan key the monitor makes one extra `tokentx` call per seed address, which is two extra Etherscan calls per day on the current watchlist. Per run CSVs are uploaded as Actions artifacts kept for 14 days. They are not committed.

Tests before that work: 166 passed, 1 skipped. After it: 172 passed, 1 skipped, including 6 new drainer tests. `address_poisoning` is still not called by the daily monitor.

This follow up pull request corrects the public labels and docs. The Streamlit sidebar and demo line said 22 rules. They now say 28. `pipeline.py` had stopped at the March 2026 v11 detector list. Nothing later said that stop was intentional, so it now runs all 28 rules in `score_row` order. `docs/RULES.md` lists all 28. The README named list now includes `foreign_country`. The old "23 detect functions plus 5 pipeline stages" line is replaced. Those five are scoring rules inside `score_row`, not feature stages, and profile classification is not one of them. The daily monitor runs 27 rules. It still does not call `address_poisoning`. The backtest runs 26. It does not call `detect_drainer_signature` or `detect_address_poisoning`.

## c) Agent Chess harness

From hash-2026 pull request 24, as Hash recorded it. This session did not rerun those tests.

That branch has 25 tests, not the 12 an October 19 note implied. Before the fix: 6 passed, 17 failed, 2 errors. After: 25 passed.

Root cause: `tools/agent_chess_surface.py` and `tools/verify_agent_chess_boards.py` were never committed. The only known original was uncommitted on wukong at `/home/hash/maya01-brain/tools/signal_board_publisher`. That copy has not yet been checked for restoration. Both files were rewritten to the test contract in `tools/agent_chess_mcp/FIX-NOTES.md`. They are not byte restores.

MCP integration is not ready. The server is an unregistered local stdio process. The harness checks transcript shape, not whether a claim is true. Nothing is deployed.

## d) Vault mirror archive

The figures in this section come from a read only search of wukong on 2026-11-26, as supplied for these notes. This session did not open wukong and did not recount the files.

Location: wukong `/home/hash/vault-mirror`. About 14 MB, 1332 files, 69 directories.

Agent Chess folder `70-ops-logs/agent-chess/`: 601 files, 2.9 MB (460 markdown, 140 json, 1 html). That includes 414 turn markdown files and 124 turn json files, 7 game summaries, `mcp-boards/` (13 files but only 3 distinct board ids: `cold-agent-board-001`, `demo-mcp-board-001`, `demo-mcp-board-002`), and `codex-templates/`.

Agent names in turn filenames: Codex 322, CloudCode 206. Turn numbers run about 2 to 205, with some numbers reused across boards. Turn filename dates run from May 15 to May 24, 2026. The newest files in that folder are May 31, 2026.

A second copy is at `/home/hash/vault-mirror-non-job` (about 11 MB, the same 601 Agent Chess files). The total file count of that second copy was not verified on that search.

The designed source is the Windows OneDrive path `C:\Users\himan\OneDrive\hash 2026`. That machine was not reachable, so whether it holds newer data is unknown.

The last mirror cron log is June 7, 2026, 2:00 AM PDT, and it reported "Mirror not updated in >48 hours".

Folders cited by turns but missing from the mirror: `70-ops-logs/capital-accounting/` and `70-ops-logs/edge-research/`. `30-infrastructure/secrets-local/` exists and was not opened.

Protocol file: `30-infrastructure/revenue-machine/AGENT_CHESS_PROTOCOL.md`. Codex and CloudCode alternate claim, evidence, and required pushback. Hash is the arbiter. The work is paper and research only.

Unverified figures from Hash's recollection, not from the file count above:

- Whole mirror date range March 7 to May 31.
- "537 turn files (321 Codex)".
- Turns 3 to 203.
- "383 turns resolved or accepted".
- "Only 1 contradiction".
- 910 files in the second copy.
- The waiting for Hash loop named in turn 195.

## e) Debate analysis

Hash's assessment, not a measurement from this session: the agents mostly agreed with each other rather than genuinely challenging, Hash was the bottleneck because debates parked waiting for him, and evidence sometimes pointed at folders that do not exist in the mirror. The missing folder point matches the verified absence of `70-ops-logs/capital-accounting/` and `70-ops-logs/edge-research/` from the mirror search above.

Samples recorded from that search:

- Fee gate, turns 22 to 23 (May 15), resolved as `proposal_candidate_review_only`.
- Pancake, turns 112 to 121 (May 17), resolved. The note in `20260517-1130-cloudcode-turn-115.md` says "There is no contradiction. Both results are correct."
- Learning curve and GARCH, turns 197 to 198 (May 22), partly resolved, `wait_for_full_evidence_surface`.
- Cold MCP board (May 31), validated with open risks.

This session did not open those turn files.

## f) Planned improvements

These are Hash's plans. They are not implemented.

- Give the agents opposing objectives so they actually challenge each other.
- Auto resolve and auto escalate rules so Hash is not the bottleneck.
- Make evidence machine openable, or the claim fails automatically.
- Require an explicit forward simulation, about 5 moves ahead, before a claim is committed.
- Harden the May 31 proposer, challenger, and reviewer structure, with a simple automated checker as the reviewer.

## g) Rule count phrases in bionic-banker

`hash02/bionic-banker` was cloned read only at HEAD `92dfd853b9832c43ad017d677e5b156611cbb981`. Nothing in that repo was edited. Published posts stay as they are until Hash approves exact text.

The engine now counts 28 scoring rules: 23 `detect_*` functions plus `large_amount`, `velocity`, `structuring`, `fan_in`, and `foreign_country` inside `score_row`. The daily monitor runs 27. Phrases that say 28 rules match that scoring total. Phrases that explain the 28 as 23 functions plus 5 feature stages do not. No source phrase saying 26 rules or 27 rules was found. "26 signals detected" in `ai-report-latest.json` is a signal headline, not a rule count.

Source mentions (built HTML under `blog/`, `apps/`, `labs/`, `investigate/`, `evidence/`, `wallet-risk/`, and `dashboard-data/` repeats the same sentences):

- `_astro-source/src/content/blog/aml-engine.md`, post date 2026-02-27, line 13: "28 detection rules that catch known patterns".
- Same file, line 53: "The current manifest maps 28 detection abilities."
- Same file, line 55: "That number includes 23 direct `detect_` functions and 5 feature-stage abilities that are computed before the rules fire." This is the inaccurate split. Those five are scoring rules, not a stage that runs before the rules.
- Same file, line 76: "The feature-stage abilities look at wallet behavior: velocity, amount distribution, time gaps, counterparty shape, and address profile." Velocity is one scoring rule. Address profile is not one of the 28. `foreign_country` is not named.
- `_astro-source/src/content/blog/aml-roaster.md`, post date 2026-03-08, line 12: "using 28 detection rules". Line 186: "The full AML Detection Engine (28 rules, 94.9% detection rate)". The 94.9 percent figure was not recomputed in this session.
- `_astro-source/src/content/blog/nexus-agent.md`, post date 2026-03-05, line 122: "my AML engine (28 detection rules, 94.9% accuracy)". Same note on 94.9 percent.
- `_astro-source/src/content/blog/wallet-risk-notes.md`, post date 2026-03-14, line 23: "28 mapped detection abilities". Line 24: "23 direct detection functions". 23 is the `detect_*` count, not the full scoring total.
- `_astro-source/src/pages/wallet-risk-notes.astro`, no post date in the file, lines 54 and 58: "28" mapped abilities and "23" direct functions.
- `_astro-source/src/pages/apps.astro`, line 20: "28 mapped detection abilities, 23 direct functions".
- `_astro-source/src/pages/labs/aml-engine.astro`, line 10: "28 rules".
- `_astro-source/src/pages/investigate.astro`, lines 101, 115, and 129: "AML Detection Engine v12, 28 rules". Line 136: "Running 28 AML detection rules...". Line 253 repeats the v12, 28 rules label. Line 302: "AML Detection Engine v12, 28 rules (Example)".
- `_astro-source/public/dashboard-data/engineering-report-catalog.json`, lines 60 to 68: "Direct detection functions" value 23, and "Mapped detection abilities" value 28.
- `_astro-source/public/dashboard-data/source-catalog.json` and `public-proof-catalog.json`, lines 36 and 37: "28 mapped detection abilities" and "23 direct detection functions".
- `_astro-source/public/blog-visuals/nexus-agent/02-split-brain.html`, line 61: "Applies 22 AML detection rules". Line 88: "22 Rules Applied". This is the stale count. The engine scores 28 rules, and the daily monitor scores 27.

Internal site agent notes, not published posts, also say "28 detection rules" (`site-agent/codex-website-care-turn-110-homepage-ux.md` line 21, and `site-agent/cloudcode-website-care-turn-015-homepage-ux.md` line 99). `site-agent/codex-website-care-turn-099.md` line 128 says to use a static value of 28 AML rules.

## h) Open items

- `address_poisoning` is still not called by the daily monitor. The fetchers still drop zero value transfers unless a known mixer is involved.
- `maya01-brain` `signal_board_publisher` on wukong has not been checked for restoration.
- The Himan OneDrive machine was not accessed. Whether it holds a newer vault than the June 2026 mirror is unknown.
- `30-infrastructure/secrets-local/` was not opened on the mirror search, and this session did not open it.
- The recollection figures in section d (date range, 537 turn files, turns 3 to 203, 383 resolved, one contradiction, 910 files, turn 195) are not verified.
- The hash-2026 merge SHA and its CI status were not rechecked here.
- bionic-banker text was not edited.

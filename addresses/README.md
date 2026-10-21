# addresses/

Authoritative threat-intel address lists emitted by the engine for downstream
consumers (roaster, dashboards, future partners). Flows one direction, engine
to downstream.

Schema + version in MANIFEST.json. Regenerated nightly via scripts/refresh_feeds.py
(future work, issue #11).

See https://github.com/hash02/hash-2026/blob/main/analysis/aml_consolidation.md
for the full library/application split plan.

## Source attribution: phishing.json

The phishing address list is derived from ScamSniffer's
scam-database (https://github.com/scamsniffer/scam-database,
file blacklist/address.json), published by Scam Sniffer
(scamsniffer.io). That data is licensed under GPL-3.0; this
attribution and license notice accompany the snapshot as the
license requires. The publisher discloses that the open data
lags its internal database by 7 days; the delay is accepted
and recorded here and in the feed definition.

Before 2026-10 the list came from MetaMask's eth-phishing-detect
repo, which no longer publishes addresses; its committed
snapshots from that period contain domain strings, not
addresses, and should not be relied on.

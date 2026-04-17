# Deployment Guide

This guide covers the three ways to deploy the AML detection engine,
ordered by operational weight.

## 1. Streamlit Cloud (demo)

Zero infrastructure. Good for sharing the UI with a reviewer; not
appropriate for production workloads.

1. Fork the repo to your GitHub account
2. Go to https://share.streamlit.io → **New app**
3. Select your fork + `streamlit_app.py`
4. Under **Advanced settings → Secrets**, paste the env vars you need:

   ```toml
   AML_APP_PASSWORD       = "change-me"
   AML_REVIEWER_USERNAMES = "alice,bob"
   AML_ADMIN_USERNAMES    = "admin"
   FEEDS_OFFLINE          = "1"         # omit in prod to pull live feeds
   ```

5. Deploy. The engine comes up with every v1–v6 capability; no external
   services required.

## 2. Single-node Docker (pilot)

For a single reviewer team that wants persistence for the audit log +
case store but doesn't yet need horizontal scale.

```bash
docker build -t aml-engine:latest .
docker run -d \
  --name aml-engine \
  -p 8501:8501 -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e AML_APP_PASSWORD=change-me \
  -e SENTRY_DSN=... \
  -e PROMETHEUS_PORT=9090 \
  aml-engine:latest
```

The SQLite databases (`audit.db`, `cases.db`) live on the mounted
volume. Back them up nightly — they are the only stateful component.

## 3. Production compose (team)

Use `deploy/docker-compose.prod.yml` — brings up:

- `streamlit` — the reviewer UI on :8501
- `api`       — FastAPI sidecar on :8000 for programmatic access
- `feeds`     — cron-style container that runs `refresh_feeds.py` hourly

```bash
cd deploy
cp env.example .env            # fill in secrets
docker compose -f docker-compose.prod.yml up -d
```

### Secrets to rotate

| Variable | What | Rotation cadence |
|----------|------|------------------|
| `AML_APP_PASSWORD`   | Streamlit sign-in         | 90 days |
| `AML_API_TOKEN`      | FastAPI bearer token      | 90 days |
| `AML_SESSION_SECRET` | Session cookie signing    | 90 days |
| `ETHERSCAN_API_KEY`  | Live fetcher             | Rotate if leaked |
| `TELEGRAM_BOT_TOKEN` | Alert notifier            | Rotate if leaked |
| `SENTRY_DSN`         | Error reporting           | Generally permanent |

### DR plan

1. **Back up** — `data/audit.db` + `data/cases.db` nightly to object
   storage with 7-day retention minimum.
2. **Restore** — stop the container, replace the files on the volume,
   start the container. No schema migration needed between minor
   releases (schema is additive).
3. **Rebuild** — the engine itself is stateless. `docker compose up`
   against a clean volume + restored DBs is the full recovery path.

# AML Detection Engine — developer DX
# Every target is idempotent and self-contained. `make help` lists them.

.PHONY: help install test test-fast lint format backtest feeds run api docker docker-run clean

PY       ?= python3
PYTEST   ?= $(PY) -m pytest
RUFF     ?= $(PY) -m ruff

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## Install runtime + dev deps
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install pytest ruff fastapi uvicorn pydantic

test:  ## Run full test suite
	FEEDS_OFFLINE=1 $(PYTEST) tests/ -q

test-fast:  ## Run tests, stop on first failure
	FEEDS_OFFLINE=1 $(PYTEST) tests/ -x -q

lint:  ## Ruff lint (read-only)
	$(RUFF) check engine scripts tests api.py

format:  ## Ruff format + autofix
	$(RUFF) check --fix engine scripts tests api.py

backtest:  ## Replay bundled sample data and emit JSON report
	FEEDS_OFFLINE=1 $(PY) scripts/backtest.py --dir data --out backtest_report.json

feeds:  ## Pull the latest threat-intel feeds
	$(PY) scripts/refresh_feeds.py

run:  ## Run Streamlit locally on :8501
	streamlit run streamlit_app.py --server.port=8501

api:  ## Run the FastAPI sidecar on :8000
	uvicorn api:app --host 0.0.0.0 --port 8000 --reload

docker:  ## Build the container image
	docker build -t aml-engine:local .

docker-run:  ## Run the container image on :8501
	docker run --rm -p 8501:8501 aml-engine:local

clean:  ## Remove caches + build artefacts
	rm -rf .pytest_cache .ruff_cache __pycache__ **/__pycache__ backtest_report.json

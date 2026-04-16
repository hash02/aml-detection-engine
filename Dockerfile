# syntax=docker/dockerfile:1.7
# Two-stage build so the final image doesn't carry build tooling.

FROM python:3.12-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --prefix=/install -r requirements.txt \
 && pip install --prefix=/install pytest ruff

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Non-root runtime user for Streamlit Cloud parity
RUN groupadd -r aml && useradd -r -g aml aml

COPY --from=builder /install /usr/local
COPY --chown=aml:aml . /app

USER aml

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health', timeout=3).raise_for_status()" || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

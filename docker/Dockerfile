# syntax=docker/dockerfile:1.7
# Both stages must share the Python build: the venv's interpreter is a symlink into it.
ARG PYTHON_VERSION=3.11

# --- Builder: venv, models and vector store ---------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.cache/huggingface
WORKDIR /app

# Dependencies: exact versions from uv.lock (CPU-only torch on Linux, see pyproject.toml).
# The project itself is imported via PYTHONPATH, not installed.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Models, named by config.py alone: this layer survives any change outside src/common/.
COPY src/common/ ./src/common/
RUN python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
from common.config import config; \
SentenceTransformer(config.embedding_model_name); \
CrossEncoder(config.reranker_model_name)"

# Vector store: rebuilt when the corpus or the ingestion code changes.
COPY src/ ./src/
COPY data/raw/ ./data/raw/
RUN python -m ingestion.build

# --- Runtime: no uv, no build cache ----------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime
# HF_HUB_OFFLINE: the models are baked in, so the Hub is never contacted at startup.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    ANONYMIZED_TELEMETRY=False
RUN useradd --create-home --uid 1000 docqa
WORKDIR /app

# Chroma writes to its SQLite file and the Hugging Face libraries expect a writable cache,
# so both belong to the runtime user; code and venv stay root-owned and read-only.
# `--chown` sets ownership without the duplicate layer a `chown -R` would create.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder --chown=docqa:docqa /app/.cache/huggingface /app/.cache/huggingface
COPY --from=builder --chown=docqa:docqa /app/data /app/data
COPY --from=builder /app/src /app/src
USER docqa

EXPOSE 8000
# Startup loads both models and indexes both collections for BM25 before /health answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Generation defaults to Ollama on the *host*; see the README (or compose.yaml) for how to
# reach it from the container.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

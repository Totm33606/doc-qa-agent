# syntax=docker/dockerfile:1.7
# --- Base: uv-managed Python, shared by all stages ---------------------------
FROM python:3.11-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /usr/local/bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# --- Dependencies layer (cached independently of source changes) ------------
FROM base AS deps
# Exact versions from uv.lock; the project itself is imported via PYTHONPATH below.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-dev --no-emit-project -o requirements.txt \
    && uv pip install --system -r requirements.txt

# --- Vector store and both models, baked in at image-build time --------------
# At runtime the container only needs network to reach the LLM. `ingestion.build`
# downloads the embedder; the re-ranker is fetched explicitly. HF_HOME sits under
# /app because a cache under /root is unreadable by the non-root runtime user,
# which would re-download both models at every container start.
FROM deps AS store
COPY src/ ./src/
COPY data/raw/ ./data/raw/
ENV PYTHONPATH=/app/src \
    HF_HOME=/app/.cache/huggingface
RUN python -m ingestion.build \
    && python -c "from retrieval.rerank import CrossEncoderReranker; CrossEncoderReranker()"

# --- Runtime image ------------------------------------------------------------
FROM store AS runtime

# Non-root user; the store and model cache were built as root, so hand them over.
RUN useradd --create-home --uid 1000 docqa \
    && chown -R docqa:docqa /app/data /app/.cache
USER docqa

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Generation defaults to Ollama on the *host* — from inside the container
# that's usually host.docker.internal, not localhost:
#   docker run -p 8000:8000 -e LOCAL_LLM_BASE_URL=http://host.docker.internal:11434/v1 doc-qa-agent
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

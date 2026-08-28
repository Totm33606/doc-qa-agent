# syntax=docker/dockerfile:1.7
# --- Base: uv-managed Python, shared by all stages ---------------------------
FROM python:3.11-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# --- Dependencies layer (cached independently of source changes) ------------
FROM base AS deps
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e .

# --- Build the vector store at image-build time, not at container startup ---
# Bakes in both Chroma collections (~16MB) so the container is self-contained
# at runtime — no BGE-model download, no re-embedding, no network dependency
# beyond reaching whichever LLM `/ask` is configured to call.
FROM deps AS store
COPY src/ ./src/
COPY data/raw/ ./data/raw/
ENV PYTHONPATH=/app/src
RUN python -m ingestion.build

# --- Runtime image ------------------------------------------------------------
FROM store AS runtime

# Non-root user: standard hardening for a production-facing image. The store
# was built as root above; chown so the runtime user can actually open it.
RUN useradd --create-home --uid 1000 docqa \
    && chown -R docqa:docqa /app/data
USER docqa

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Generation defaults to Ollama on the *host* — from inside the container
# that's usually host.docker.internal, not localhost:
#   docker run -p 8000:8000 -e LOCAL_LLM_BASE_URL=http://host.docker.internal:11434/v1 doc-qa-agent
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

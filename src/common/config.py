"""Centralized, typed configuration for ingestion, retrieval and generation.

Kept as a single Pydantic settings object so `fetch.py`, `build.py`,
`retrieval/retriever.py`, `retrieval/rerank.py` and `eval/run_eval.py` all
agree on the same paths/model names — the same anti-drift rationale as
`ml_pipeline/config.py` in the finrisk-agent sibling project.

It lives in `common/` (next to `schemas.py`) rather than in `ingestion/`
for the same reason `schemas.py` does: every stage of the pipeline reads
it, so it belongs to none of them in particular. Query-time code importing
its settings from the *ingestion* package would be a layering inversion —
`retriever.py` has no business depending on ingestion for the value of
`rrf_k`.

**One deliberate exception to "one source of truth".** The LLM settings
below are defaults, not the last word: `generation/llm.py` reads the
*unprefixed* environment variables (`LOCAL_LLM_MODEL`, `OPENAI_MODEL`,
`AZURE_OPENAI_*`) first and falls back to these, so the provider is
configured with the same variable names as in the finrisk-agent sibling
project and as the underlying SDKs themselves document. The `DOCQA_`-prefixed
spellings still work, but they lose to the unprefixed ones when both are set.
Everything else in this file is configured only via `DOCQA_*`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DocQAConfig(BaseSettings):
    """Environment-overridable settings. Prefix: DOCQA_."""

    model_config = SettingsConfigDict(env_prefix="DOCQA_", env_file=".env", extra="ignore")

    # --- Corpus ------------------------------------------------------
    raw_docs_dir: Path = PROJECT_ROOT / "data" / "raw"
    corpus_manifest_path: Path = PROJECT_ROOT / "data" / "raw" / "manifest.json"
    fastapi_repo_ref: str = "0.141.1"

    # --- Chunking ------------------------------------------------------
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50
    token_encoding: str = "cl100k_base"

    # --- Embeddings ------------------------------------------------------
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # --- Vector store ------------------------------------------------------
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"

    def collection_name(self, strategy: str) -> str:
        return f"fastapi_docs_{strategy}"

    # --- Retrieval ------------------------------------------------------
    default_top_k: int = 5
    # Hybrid (dense + BM25) fusion, see retrieval/retriever.py:
    rrf_k: int = 60  # the reciprocal-rank-fusion constant, 60 per Cormack et al. (2009)
    hybrid_candidates: int = 20  # candidates asked of *each* retriever before fusing

    # --- Re-ranking ------------------------------------------------------
    # Cross-encoder re-scoring of the fused candidates, see retrieval/rerank.py.
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20  # fused candidates re-scored before truncating to top_k
    # Relevance floor: a question whose best candidate scores below this is
    # answered with an abstention rather than from irrelevant passages. Unlike
    # `rrf_k`, this one *is* tuned — `run_eval.py --sweep` measures the whole
    # grid, and 0.2 is the highest value that refuses none of the 38 golden
    # questions (the lowest of which scores 0.2514) while still catching 6 of
    # the 10 out-of-domain ones. See the README's Evaluation section.
    rerank_min_score: float = 0.2

    # --- Generation (LLM) ------------------------------------------------------
    # Same priority order as finrisk-agent's `_build_llm`: Azure OpenAI > a
    # local OpenAI-API-compatible server (Ollama by default) > plain OpenAI.
    # Defaults only — see this module's docstring on the unprefixed env vars.
    local_llm_base_url: str = "http://localhost:11434/v1"
    local_llm_model: str = "qwen2.5:7b-instruct"
    local_llm_api_key: str = "not-needed"
    openai_model: str = "gpt-4.1"
    azure_openai_deployment: str = "gpt-5"
    azure_openai_api_version: str = "2024-12-01-preview"


config = DocQAConfig()

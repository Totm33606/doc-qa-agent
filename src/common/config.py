"""Typed settings shared by every stage, overridable via `DOCQA_*` environment variables.

The LLM settings are only defaults: `generation/llm.py` reads the unprefixed variables
(`LOCAL_LLM_*`, `OPENAI_MODEL`, `AZURE_OPENAI_*`) first.
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
    rrf_k: int = 60  # reciprocal rank fusion constant, per Cormack et al. (2009); not tuned
    hybrid_candidates: int = 20  # candidates asked of each retriever before fusion

    # --- Re-ranking ------------------------------------------------------
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20  # fused candidates re-scored before truncating to top_k
    # Relevance floor, from `run_eval.py --sweep`: the highest grid value that refuses
    # none of the 38 golden questions (lowest in-domain score: 0.2514).
    rerank_min_score: float = 0.2

    # --- Generation (LLM) ------------------------------------------------------
    # Defaults only: generation/llm.py reads the unprefixed env vars first.
    local_llm_base_url: str = "http://localhost:11434/v1"
    local_llm_model: str = "qwen2.5:7b-instruct"
    local_llm_api_key: str = "not-needed"
    openai_model: str = "gpt-4.1"
    azure_openai_deployment: str = "gpt-5"
    azure_openai_api_version: str = "2024-12-01-preview"


config = DocQAConfig()

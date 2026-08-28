"""Centralized, typed configuration for ingestion, retrieval and generation.

Kept as a single Pydantic settings object so `fetch.py`, `build.py`,
`retrieval/retriever.py` and `eval/run_eval.py` all agree on the same
paths/model names — the same anti-drift rationale as `ml_pipeline/config.py`
in the finrisk-agent sibling project.
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

    # --- Generation (LLM) ------------------------------------------------------
    # Same priority order as finrisk-agent's `_build_llm`: Azure OpenAI > a
    # local OpenAI-API-compatible server (Ollama by default) > plain OpenAI.
    local_llm_base_url: str = "http://localhost:11434/v1"
    local_llm_model: str = "qwen2.5:7b-instruct"
    local_llm_api_key: str = "not-needed"
    openai_model: str = "gpt-4.1"
    azure_openai_deployment: str = "gpt-5"
    azure_openai_api_version: str = "2024-12-01-preview"


config = DocQAConfig()

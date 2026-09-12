"""LLM selection: Azure OpenAI > local OpenAI-compatible server (default: Ollama) > OpenAI."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from common.config import PROJECT_ROOT, config

# The provider variables are unprefixed, so pydantic-settings never loads them from `.env`.
load_dotenv(PROJECT_ROOT / ".env")


def build_llm() -> ChatOpenAI | AzureChatOpenAI:
    """Build the generation model (any instruction-following model works, no tool calling)."""
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", config.azure_openai_deployment),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", config.azure_openai_api_version),
            temperature=0,
        )
    # The local server is the default because `local_llm_base_url` has one; set
    # `LOCAL_LLM_BASE_URL=` (empty) to fall through to plain OpenAI.
    local_base_url = os.getenv("LOCAL_LLM_BASE_URL", config.local_llm_base_url)
    if local_base_url:
        return ChatOpenAI(
            base_url=local_base_url,
            api_key=os.getenv("LOCAL_LLM_API_KEY", config.local_llm_api_key),
            model=os.getenv("LOCAL_LLM_MODEL", config.local_llm_model),
            temperature=0,
        )
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", config.openai_model), temperature=0)

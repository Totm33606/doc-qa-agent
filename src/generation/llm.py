"""Chat-model construction, mirroring `agent/agent.py::_build_llm` in the finrisk-agent
sibling project so both portfolio projects configure an LLM the same way.

Priority: Azure OpenAI > a local OpenAI-API-compatible server (Ollama by
default) > plain OpenAI. Ollama needs no API key and no cost, which is why
it's the default rather than an opt-in — cloning this repo and running the
demo end to end should never require paying for anything.
"""

from __future__ import annotations

import os

from langchain_openai import AzureChatOpenAI, ChatOpenAI

from ingestion.config import config


def build_llm() -> ChatOpenAI | AzureChatOpenAI:
    """Build the chat model used for answer generation.

    Setting `LOCAL_LLM_BASE_URL` switches to a local, OpenAI-API-compatible
    server instead — e.g. Ollama (`http://localhost:11434/v1`), LM Studio
    or vLLM's OpenAI-compatible endpoint. Pick a model that's actually
    reasonable at following instructions (e.g. `qwen2.5:7b-instruct`); this
    project doesn't need tool-calling support, just instruction-following.
    """
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", config.azure_openai_deployment),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", config.azure_openai_api_version),
            temperature=0,
        )
    # Unlike finrisk-agent (where a local server is opt-in and plain OpenAI
    # is the ultimate fallback), `config.local_llm_base_url` defaults to
    # Ollama's local endpoint rather than being unset — this project's
    # brief calls for a local LLM as the *default*, not an alternative, so
    # cloning the repo and running the demo never implies an API cost.
    local_base_url = os.getenv("LOCAL_LLM_BASE_URL", config.local_llm_base_url)
    if local_base_url:
        return ChatOpenAI(
            base_url=local_base_url,
            api_key=os.getenv("LOCAL_LLM_API_KEY", config.local_llm_api_key),
            model=os.getenv("LOCAL_LLM_MODEL", config.local_llm_model),
            temperature=0,
        )
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", config.openai_model), temperature=0)

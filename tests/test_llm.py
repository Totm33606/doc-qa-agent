"""Tests for build_llm()'s provider-selection branching.

Constructing `ChatOpenAI`/`AzureChatOpenAI` doesn't make a network call —
only `.invoke()` does — so these stay hermetic by checking the returned
client's type and configuration, never calling it. A dummy API key is
enough to get past `langchain_openai`'s own instantiation-time validation
(it insists a key be *present*, never checks it's *real*).
"""

from __future__ import annotations

import pytest
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from generation.llm import build_llm

_ENV_VARS = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
    "LOCAL_LLM_BASE_URL",
    "LOCAL_LLM_MODEL",
    "LOCAL_LLM_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate every test from whatever's actually in this machine's environment/.env."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_build_llm_defaults_to_local_ollama_endpoint() -> None:
    llm = build_llm()
    assert isinstance(llm, ChatOpenAI)
    assert not isinstance(llm, AzureChatOpenAI)
    assert llm.openai_api_base == "http://localhost:11434/v1"


def test_build_llm_respects_local_llm_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://example.local:9999/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "custom-model")

    llm = build_llm()

    assert llm.openai_api_base == "http://example.local:9999/v1"
    assert llm.model_name == "custom-model"


def test_build_llm_prefers_azure_when_both_azure_vars_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")

    llm = build_llm()

    assert isinstance(llm, AzureChatOpenAI)


def test_build_llm_ignores_azure_when_only_one_var_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy")
    # AZURE_OPENAI_ENDPOINT deliberately left unset — both are required to select Azure.

    llm = build_llm()

    assert not isinstance(llm, AzureChatOpenAI)


def test_build_llm_falls_back_to_plain_openai_when_local_url_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "")  # present but empty -> falsy, unlike unset
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")

    llm = build_llm()

    assert isinstance(llm, ChatOpenAI)
    assert not isinstance(llm, AzureChatOpenAI)
    assert llm.model_name == "gpt-4.1-mini"

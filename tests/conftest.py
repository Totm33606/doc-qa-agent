"""Shared, hermetic test doubles: a fake embedder and a fake chat model.

No test in this suite downloads the real `BAAI/bge-small-en-v1.5` model or
calls a real LLM (Ollama or otherwise) — `FakeEmbedder` and `FakeChatModel`
stand in for both, so the whole suite runs offline and in seconds. The one
exception is `tests/test_integration.py`, explicitly marked `integration`
and skipped unless `--run-integration` is passed — see that file's
docstring.
"""

from __future__ import annotations

import hashlib

from langchain_core.messages import AIMessage, BaseMessage

FAKE_EMBEDDING_DIM = 16


class FakeEmbedder:
    """Deterministic, hash-based embedder — same text always yields the same vector.

    Not semantically meaningful (unlike the real BGE model), but that's
    fine for tests that only need embeddings to be present and stable
    across the multiple lookups a Chroma round-trip performs.
    """

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:FAKE_EMBEDDING_DIM]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeChatModel:
    """A `ChatModel` (see `generation.generate.ChatModel`) that returns a canned answer."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.invocations: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        self.invocations.append(messages)
        return AIMessage(content=self.content)

"""Hermetic test doubles for the embedder, the re-ranker and the chat model.

Only `tests/test_integration.py` (marker `integration`) uses the real models.
"""

from __future__ import annotations

import hashlib

from langchain_core.messages import AIMessage, BaseMessage

from common.schemas import RetrievedPassage

FAKE_EMBEDDING_DIM = 16


class FakeEmbedder:
    """Deterministic, hash-based embedder: stable vectors, no semantics."""

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:FAKE_EMBEDDING_DIM]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeReranker:
    """A `Reranker` whose scores are dictated per chunk id (`default` for the rest).

    Lets tests state relevance as a premise instead of depending on a real model.
    """

    def __init__(self, scores: dict[str, float] | None = None, default: float = 0.5) -> None:
        self.scores = scores or {}
        self.default = default
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, question: str, passages: list[RetrievedPassage]) -> list[float]:
        self.calls.append((question, [p.chunk_id for p in passages]))
        return [self.scores.get(p.chunk_id, self.default) for p in passages]


class FakeChatModel:
    """A `ChatModel` (see `generation.generate.ChatModel`) that returns a canned answer."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.invocations: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        self.invocations.append(messages)
        return AIMessage(content=self.content)

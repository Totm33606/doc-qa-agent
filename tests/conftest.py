"""Shared, hermetic test doubles: a fake embedder, a fake re-ranker and a fake chat model.

No test in this suite downloads the real `BAAI/bge-small-en-v1.5` model or
the real `cross-encoder/ms-marco-MiniLM-L-6-v2`, or calls a real LLM (Ollama
or otherwise) — `FakeEmbedder`, `FakeReranker` and `FakeChatModel` stand in
for all three, so the whole suite runs offline and in seconds. The one
exception is `tests/test_integration.py`, explicitly marked `integration`
and skipped unless `--run-integration` is passed — see that file's
docstring.
"""

from __future__ import annotations

import hashlib

from langchain_core.messages import AIMessage, BaseMessage

from common.schemas import RetrievedPassage

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


class FakeReranker:
    """A `Reranker` (see `retrieval.rerank.Reranker`) with scores dictated per chunk id.

    The real cross-encoder is a judgement call about relevance, which is
    exactly what a test needs to be able to state outright: `scores` maps a
    chunk id to the relevance this fake should claim for it, and everything
    unlisted gets `default`. That makes both the reordering and the
    relevance floor testable without asserting anything about a real model's
    opinions.
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

"""Embedding backends; `Embedder` is a Protocol so tests can inject a fake model."""

from __future__ import annotations

from typing import Protocol

from common.config import config

# BGE's retrieval instruction, prepended to queries only — never to stored passages.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class QueryTooLongError(ValueError):
    """The question does not fit the embedding model's window, which would silently cut its end."""


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class BGEEmbedder:
    """`BAAI/bge-small-en-v1.5` via `sentence-transformers`, CPU by default, no API key."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name or config.embedding_model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        query = _QUERY_INSTRUCTION + text
        tokenizer = self._model.tokenizer
        n_tokens = len(tokenizer.tokenize(query)) + tokenizer.num_special_tokens_to_add()
        if n_tokens > self._model.max_seq_length:
            raise QueryTooLongError(
                f"The question is too long to embed: {n_tokens} tokens with the query "
                f"instruction, over the model's {self._model.max_seq_length}-token window."
            )
        vector = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        return list(vector[0].tolist())

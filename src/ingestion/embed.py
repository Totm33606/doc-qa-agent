"""Embedding backends.

`Embedder` is a small `Protocol` (not an ABC) so tests can inject a cheap,
deterministic fake instead of downloading and running the real 130MB
`BAAI/bge-small-en-v1.5` model — `BGEEmbedder` is the only implementation
that ever gets loaded outside of `ingestion.build`, `retrieval.retriever`
and the one real-model integration test.

**Why this lives in `ingestion/` even though query time uses it too.** A
query has to be embedded by the exact same model that embedded the stored
documents, or the two vectors aren't comparable — the embedder isn't a
shared utility both stages happen to want, it's the ingestion decision that
retrieval is obliged to match. Keeping it here says so. (Contrast
`common/config.py`, which every stage reads and none owns.)
"""

from __future__ import annotations

from typing import Protocol

from common.config import config

# BGE's model card recommends prefixing the *query* side only (never the
# stored passages) with this instruction for retrieval tasks — it's what
# the model was fine-tuned to expect, and measurably improves retrieval
# quality over embedding the bare query text.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


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
        vector = self._model.encode(
            [_QUERY_INSTRUCTION + text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return list(vector.tolist())

"""In-memory BM25 index over the same chunks as the dense index.

BM25 matches exact terms weighted by rarity, where embeddings are weakest: identifiers
like `response_model_exclude_unset` or `BackgroundTasks`. The index is built from
`ChunkStore.get_all()` when a hybrid `Retriever` is constructed, so there is no second
artifact to persist. Tokenization is deliberately plain (lowercase, no stemming, no stop
words) to keep API names intact.
"""

from __future__ import annotations

import re

from common.schemas import DocChunk

# Keeps Python identifiers whole: `response_model_exclude_unset` is one token.
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """An in-memory BM25 Okapi index over a list of chunks."""

    def __init__(self, chunks: list[DocChunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunk_ids = [c.chunk_id for c in chunks]
        # BM25Okapi raises on an empty corpus (it averages document lengths).
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, top_k: int) -> list[str]:
        """Chunk ids of the best-matching chunks, most relevant first."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            ((chunk_id, float(s)) for chunk_id, s in zip(self._chunk_ids, scores, strict=True)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        # Zero-score chunks share no discriminating term with the query; keeping them
        # would feed an arbitrary ranking into the fusion.
        return [chunk_id for chunk_id, score in ranked[:top_k] if score > 0.0]

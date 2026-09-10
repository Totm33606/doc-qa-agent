"""Lexical (BM25) retrieval over the same chunks the dense index holds.

The complement to embedding search, not a replacement for it: an embedding
generalizes ("how do I validate input?" finds a passage that never says
"validate"), but it blurs exactly the tokens a docs corpus is searched by —
`response_model_exclude_unset`, `BackgroundTasks`, `@app.on_event`. BM25
scores those by exact term match, weighted by how rare the term is across
the corpus, which is where dense retrieval is structurally weakest.

The index is built in memory from `ChunkStore.get_all()` when a hybrid
`Retriever` is constructed (347 / 693 chunks here — tens of milliseconds),
so there is no second artifact to build, persist or keep in sync with the
Chroma collection it mirrors.

Tokenization is deliberately plain: lowercase, no stemming, no stop-word
list. Stemming would fold `Response`/`responses` together at the cost of
mangling identifiers, and the tokens that actually carry signal in this
corpus are exact API names — the case BM25 is being added to handle.
"""

from __future__ import annotations

import re

from common.schemas import DocChunk

# Keeps Python identifiers whole (`response_model_exclude_unset` is one
# token, not four), and drops the punctuation-heavy syntax around them.
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """An in-memory BM25 Okapi index over a list of chunks."""

    def __init__(self, chunks: list[DocChunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunk_ids = [c.chunk_id for c in chunks]
        # BM25Okapi divides by the corpus size to get the average document
        # length, so an empty corpus raises rather than scoring nothing.
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, top_k: int) -> list[str]:
        """Chunk ids of the best-matching chunks, most relevant first."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        # A zero score means no term the query and the chunk share carries
        # any discriminating weight — in practice, on a corpus this size, no
        # shared term at all. `get_scores` returns one entry per chunk
        # regardless, so without this filter a query with no lexical match
        # would still contribute a full, arbitrarily-ordered ranking to the
        # fusion.
        ranked = sorted(
            ((chunk_id, float(s)) for chunk_id, s in zip(self._chunk_ids, scores, strict=True)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [chunk_id for chunk_id, score in ranked[:top_k] if score > 0.0]

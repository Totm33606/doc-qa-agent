"""Query-time retrieval over one strategy's Chroma collection, in three modes:

- `dense`: cosine similarity on the BGE embeddings.
- `hybrid`: dense search plus BM25 (`retrieval.bm25`), fused by reciprocal rank fusion.
- `hybrid_rerank`: the fused candidates re-scored by a cross-encoder (`retrieval.rerank`);
  candidates below the relevance floor are dropped, so only this mode can return nothing.

RRF fuses ranks rather than scores because cosine and BM25 scores aren't on comparable
scales: score(d) = Σ 1 / (rrf_k + rank(d)). `dense` and `hybrid` have no floor, since
neither score means the same thing from one question to the next.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.config import config
from common.schemas import ChunkingStrategy, DocChunk, RetrievalMode, RetrievedPassage
from ingestion.embed import Embedder
from ingestion.store import ChunkStore
from retrieval.bm25 import BM25Index
from retrieval.rerank import CrossEncoderReranker, Reranker

# Modes that search lexically as well as densely, and fuse the two rankings.
_LEXICAL_MODES = frozenset({RetrievalMode.HYBRID, RetrievalMode.HYBRID_RERANK})


@dataclass(frozen=True)
class _RerankStage:
    """A re-ranker and the floor applied to its scores: a mode has both or neither."""

    reranker: Reranker
    min_score: float


def _fuse_rrf(rankings: list[list[str]], rrf_k: int) -> list[tuple[str, float]]:
    """Reciprocal rank fusion of ranked chunk-id lists.

    Ties keep first-appearance order (stable sort), so the first ranking — the dense
    one — breaks them.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        strategy: ChunkingStrategy,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        store: ChunkStore | None = None,
        reranker: Reranker | None = None,
        min_rerank_score: float | None = None,
    ) -> None:
        """Build a retriever for one collection.

        `store` and `reranker` are built from `config` when omitted; pass them to inject
        fakes or to share one loaded re-ranker. `reranker` and `min_rerank_score` are
        ignored unless `mode` is `HYBRID_RERANK`, so callers can build every mode with
        the same call. `min_rerank_score` overrides `config.rerank_min_score` (`0.0`
        disables the floor).
        """
        self._embedder = embedder
        self._strategy = strategy
        self._mode = mode
        self._store = store or ChunkStore(
            persist_dir=config.chroma_dir, collection_name=config.collection_name(strategy.value)
        )
        self._chunks: dict[str, DocChunk] = {}
        self._bm25: BM25Index | None = None
        if mode in _LEXICAL_MODES:
            chunks = self._store.get_all()
            self._chunks = {c.chunk_id: c for c in chunks}
            self._bm25 = BM25Index(chunks)

        self._rerank: _RerankStage | None = None
        if mode is RetrievalMode.HYBRID_RERANK:
            self._rerank = _RerankStage(
                reranker=reranker or CrossEncoderReranker(),
                min_score=config.rerank_min_score if min_rerank_score is None else min_rerank_score,
            )

    @property
    def strategy(self) -> ChunkingStrategy:
        return self._strategy

    @property
    def mode(self) -> RetrievalMode:
        return self._mode

    @property
    def min_rerank_score(self) -> float | None:
        """The relevance floor in force, or None for modes that cannot abstain."""
        return self._rerank.min_score if self._rerank else None

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedPassage]:
        k = top_k or config.default_top_k
        query_embedding = self._embedder.embed_query(question)
        if self._bm25 is None:  # dense mode
            return self._store.query(query_embedding, top_k=k)

        # More than k candidates per side, so BM25 can surface passages dense search missed.
        n = max(k, config.hybrid_candidates)
        dense = self._store.query(query_embedding, top_k=n)
        lexical = self._bm25.search(question, top_k=n)

        fused = _fuse_rrf([[p.chunk_id for p in dense], lexical], config.rrf_k)
        candidates = [
            self._as_passage(chunk_id, score)
            for chunk_id, score in fused
            if chunk_id in self._chunks  # defensive: both rankings come from this store
        ]

        if self._rerank is None:
            return candidates[:k]
        return self._reranked(self._rerank, question, candidates[: config.rerank_candidates], k)

    def _reranked(
        self, stage: _RerankStage, question: str, candidates: list[RetrievedPassage], k: int
    ) -> list[RetrievedPassage]:
        """Re-score, drop candidates below the floor and keep the best k — possibly none.

        The fused `score` is kept; the cross-encoder's goes in `rerank_score`.
        """
        scores = stage.reranker.score(question, candidates)
        scored = [
            (passage.model_copy(update={"rerank_score": score}), score)
            for passage, score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)  # stable: ties keep fused order
        return [passage for passage, score in scored if score >= stage.min_score][:k]

    def _as_passage(self, chunk_id: str, score: float) -> RetrievedPassage:
        chunk = self._chunks[chunk_id]
        return RetrievedPassage(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source_file=chunk.source_file,
            section=chunk.section,
            score=score,
        )

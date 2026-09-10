"""Query-time retrieval: embed a question, search one strategy's Chroma collection.

Two modes, picked per `Retriever` (see `common.schemas.RetrievalMode`):

- **dense** — cosine similarity on the BGE embeddings alone.
- **hybrid** — the same dense search *plus* a BM25 lexical search over the
  same chunks (`retrieval.bm25`), the two rankings combined by reciprocal
  rank fusion.

**Why fuse on rank instead of on score.** A cosine similarity lives in
[0, 1]; a BM25 score is unbounded and scaled by corpus statistics. Adding or
averaging them means inventing a normalization and a weight, both of which
would need tuning per corpus. RRF throws the scores away and keeps only the
position each retriever put a chunk in:

    score(d) = Σ_retrievers 1 / (k + rank(d))

with `k = 60`, the constant from Cormack et al. (2009), where the method was
introduced — kept here by convention rather than tuned, since tuning it on
the same 38 golden questions it's then scored against would just be fitting
the eval set. The constant damps the top of each list: rank 1 scores 1/61
and rank 2 scores 1/62, so being first in *both* rankings beats being first
in one and absent from the other, but a single strong hit still surfaces.

Which mode retrieves better on this corpus is measured, not assumed — see
`eval/run_eval.py`, which scores every (strategy, mode) pair.
"""

from __future__ import annotations

from common.schemas import ChunkingStrategy, DocChunk, RetrievalMode, RetrievedPassage
from ingestion.config import config
from ingestion.embed import Embedder
from ingestion.store import ChunkStore
from retrieval.bm25 import BM25Index


def _fuse_rrf(rankings: list[list[str]], rrf_k: int) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: combine ranked chunk-id lists into one ranking.

    A chunk missing from a ranking simply contributes nothing for it. Ties
    are broken deterministically by first appearance — `sorted` is stable
    and the dict preserves insertion order, so the first ranking passed in
    (the dense one) decides between chunks with equal fused scores.
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
    ) -> None:
        self._embedder = embedder
        self._strategy = strategy
        self._mode = mode
        self._store = store or ChunkStore(
            persist_dir=config.chroma_dir, collection_name=config.collection_name(strategy.value)
        )
        # Only a hybrid retriever pays for the lexical index; a dense one
        # behaves exactly as it did before hybrid search existed.
        self._chunks: dict[str, DocChunk] = {}
        self._bm25: BM25Index | None = None
        if mode is RetrievalMode.HYBRID:
            chunks = self._store.get_all()
            self._chunks = {c.chunk_id: c for c in chunks}
            self._bm25 = BM25Index(chunks)

    @property
    def strategy(self) -> ChunkingStrategy:
        return self._strategy

    @property
    def mode(self) -> RetrievalMode:
        return self._mode

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedPassage]:
        k = top_k or config.default_top_k
        query_embedding = self._embedder.embed_query(question)
        if self._bm25 is None:  # dense mode — no lexical index was ever built
            return self._store.query(query_embedding, top_k=k)

        # Each side contributes more candidates than the caller asked for:
        # with only k per side, BM25 could never surface a passage the dense
        # search missed entirely, which is the whole point of fusing them.
        n = max(k, config.hybrid_candidates)
        dense = self._store.query(query_embedding, top_k=n)
        lexical = self._bm25.search(question, top_k=n)

        fused = _fuse_rrf([[p.chunk_id for p in dense], lexical], config.rrf_k)
        return [
            self._as_passage(chunk_id, score)
            for chunk_id, score in fused[:k]
            if chunk_id in self._chunks
        ]

    def _as_passage(self, chunk_id: str, score: float) -> RetrievedPassage:
        chunk = self._chunks[chunk_id]
        return RetrievedPassage(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source_file=chunk.source_file,
            section=chunk.section,
            score=score,
        )

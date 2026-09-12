"""Query-time retrieval: embed a question, search one strategy's Chroma collection.

Three modes, picked per `Retriever` (see `common.schemas.RetrievalMode`):

- **dense** — cosine similarity on the BGE embeddings alone.
- **hybrid** — the same dense search *plus* a BM25 lexical search over the
  same chunks (`retrieval.bm25`), the two rankings combined by reciprocal
  rank fusion.
- **hybrid_rerank** — the fused candidates re-scored by a cross-encoder
  (`retrieval.rerank`), which is also what lets this mode, and only this
  mode, decide that *nothing* is relevant enough (see below).

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

**The relevance floor.** Fusion ranks candidates against each other; it
never judges whether the best of them is any good. So a question the corpus
simply doesn't answer still comes back with a full, confidently-ordered
top-k, and the only thing standing between that and a fabricated answer is
a line in the system prompt asking the model to notice. `hybrid_rerank`
closes that hole in code: a cross-encoder score is a relevance probability
comparable across questions, so candidates below `config.rerank_min_score`
are dropped, and a question where *nothing* clears the floor retrieves
nothing at all — which `generation.generate` turns into an explicit
"I don't have anything relevant" answer instead of an LLM call.

`dense` and `hybrid` deliberately have no such floor. A cosine similarity
is only semi-calibrated, and an RRF score is a function of rank positions
with no absolute meaning whatsoever — a passage ranked first by both
retrievers scores 2/61 whether it answers the question perfectly or is
merely the least-bad chunk in a corpus that has nothing to say. A threshold
on either would be a number with no defensible origin. Knowing when to
abstain is a capability the re-ranker adds, not a property of retrieval in
general.

Which mode retrieves better on this corpus is measured, not assumed — see
`eval/run_eval.py`, which scores every (strategy, mode) pair, and scores the
abstention behaviour against a set of deliberately out-of-domain questions.
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
    """The second retrieval stage: a scorer, and the floor its scores are judged against.

    One object rather than two attributes on `Retriever`, because the two are
    meaningless apart. A floor is a threshold *on this model's* scores, so
    "a relevance floor but nothing to score with it" is not a state this class
    should be able to represent — and a mode either has both or neither.
    """

    reranker: Reranker
    min_score: float


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
        reranker: Reranker | None = None,
        min_rerank_score: float | None = None,
    ) -> None:
        """Build a retriever for one collection.

        `store` and `reranker` follow the same convention: injected for
        tests, built from `config` when omitted. Passing a `reranker` also
        matters in production — it is a model to load, so the API and the
        eval harness build one and share it across retrievers rather than
        letting each construct its own.

        **`reranker` and `min_rerank_score` are ignored unless `mode` is
        `HYBRID_RERANK`**, and callers are expected to pass them anyway.
        Which modes need a cross-encoder is this class's business: having the
        caller decide would copy that rule into `api/app.py` and
        `eval/run_eval.py`, where it would then have to be remembered every
        time a mode is added. Both build their retrievers with one uniform
        call for every (strategy, mode) pair, and adding `HYBRID_RERANK`
        needed no change to either.

        `min_rerank_score` overrides `config.rerank_min_score`; pass `0.0`
        to disable the relevance floor entirely, which is what the threshold
        sweep in `eval/run_eval.py` does to see the scores it would gate on.
        """
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
        if mode in _LEXICAL_MODES:
            chunks = self._store.get_all()
            self._chunks = {c.chunk_id: c for c in chunks}
            self._bm25 = BM25Index(chunks)

        # Likewise, only a re-ranking retriever loads the cross-encoder — and
        # only it has a relevance floor, which is why the two are one optional
        # attribute rather than two independent ones.
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
        """The relevance floor in force, or None in the modes that have none — and so can't abstain."""
        return self._rerank.min_score if self._rerank else None

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
        # The `in self._chunks` guard is defensive only: both rankings are
        # built from this same store, so every fused id resolves.
        candidates = [
            self._as_passage(chunk_id, score)
            for chunk_id, score in fused
            if chunk_id in self._chunks
        ]

        if self._rerank is None:
            return candidates[:k]
        return self._reranked(self._rerank, question, candidates[: config.rerank_candidates], k)

    def _reranked(
        self, stage: _RerankStage, question: str, candidates: list[RetrievedPassage], k: int
    ) -> list[RetrievedPassage]:
        """Re-score candidates with the cross-encoder, drop the irrelevant, keep the best k.

        The fused `score` is preserved on every passage rather than
        overwritten — the two numbers answer different questions ("how did
        the first stage rank this?" vs "how relevant is this, really?"), and
        keeping both is what makes a re-ranked response readable next to a
        plain hybrid one.

        Returning `[]` is a meaningful answer, not a failure: it means
        nothing in the corpus cleared the relevance floor.
        """
        scores = stage.reranker.score(question, candidates)
        scored = [
            (passage.model_copy(update={"rerank_score": score}), score)
            for passage, score in zip(candidates, scores, strict=True)
        ]
        # Stable, like `_fuse_rrf`'s sort: equal re-rank scores keep the fused
        # order, so the first stage still breaks the second stage's ties.
        scored.sort(key=lambda pair: pair[1], reverse=True)
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

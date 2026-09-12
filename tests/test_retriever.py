from __future__ import annotations

import pytest

from common.config import config
from common.schemas import ChunkingStrategy, DocChunk, RetrievalMode
from ingestion.store import ChunkStore
from retrieval.retriever import Retriever, _fuse_rrf
from tests.conftest import FakeEmbedder, FakeReranker

embedder = FakeEmbedder()


def _chunk(chunk_id: str, text: str) -> DocChunk:
    return DocChunk(
        chunk_id=chunk_id,
        text=text,
        source_file="tutorial/path-params.md",
        section="Order matters",
        strategy=ChunkingStrategy.MARKDOWN,
        token_count=len(text.split()),
        chunk_index=0,
    )


def _populated_store(name: str) -> ChunkStore:
    store = ChunkStore(persist_dir=None, collection_name=name)
    chunks = [_chunk("c1", "declare path parameters with types in fastapi")]
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return store


def test_retrieve_returns_passages_from_injected_store() -> None:
    store = _populated_store("retriever_injected")
    retriever = Retriever(embedder, ChunkingStrategy.MARKDOWN, RetrievalMode.DENSE, store=store)

    passages = retriever.retrieve("declare path parameters with types in fastapi", top_k=5)

    assert len(passages) == 1
    assert passages[0].chunk_id == "c1"


def test_retrieve_uses_default_top_k_when_not_specified() -> None:
    store = ChunkStore(persist_dir=None, collection_name="retriever_default_k")
    chunks = [_chunk(f"c{i}", f"chunk {i} content") for i in range(10)]
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    retriever = Retriever(embedder, ChunkingStrategy.FIXED, RetrievalMode.DENSE, store=store)

    passages = retriever.retrieve("chunk content")
    assert len(passages) == config.default_top_k


def test_retriever_exposes_its_strategy() -> None:
    store = ChunkStore(persist_dir=None, collection_name="retriever_strategy")
    retriever = Retriever(embedder, ChunkingStrategy.MARKDOWN, store=store)
    assert retriever.strategy is ChunkingStrategy.MARKDOWN


def test_retriever_exposes_its_mode_and_defaults_to_hybrid() -> None:
    store = ChunkStore(persist_dir=None, collection_name="retriever_mode")
    assert Retriever(embedder, ChunkingStrategy.FIXED, store=store).mode is RetrievalMode.HYBRID
    assert (
        Retriever(embedder, ChunkingStrategy.FIXED, RetrievalMode.DENSE, store=store).mode
        is RetrievalMode.DENSE
    )


@pytest.mark.parametrize("mode", list(RetrievalMode))
def test_retrieve_on_empty_store_returns_empty_list(mode: RetrievalMode) -> None:
    store = ChunkStore(persist_dir=None, collection_name=f"retriever_empty_{mode.value}")
    retriever = Retriever(
        embedder, ChunkingStrategy.FIXED, mode, store=store, reranker=FakeReranker()
    )
    assert retriever.retrieve("anything") == []


# --- Reciprocal rank fusion ------------------------------------------------
# Hand-computed against the formula in `retriever.py`: score = sum of
# 1 / (60 + rank) over the rankings a chunk appears in.


def test_fuse_rrf_sums_the_contributions_of_every_ranking() -> None:
    fused = _fuse_rrf([["a", "b"], ["a", "c"]], rrf_k=60)

    assert fused[0] == ("a", 2 / 61)  # rank 1 in both rankings
    assert dict(fused)["b"] == 1 / 62  # rank 2 in the first only
    assert dict(fused)["c"] == 1 / 62  # rank 2 in the second only


def test_fuse_rrf_ranks_agreement_above_a_single_first_place() -> None:
    """Second in both beats first in one and absent from the other — the point of fusing."""
    fused = _fuse_rrf([["solo", "shared"], ["other", "shared"]], rrf_k=60)

    assert fused[0][0] == "shared"
    assert fused[0][1] == pytest.approx(2 / 62)
    assert fused[0][1] > 1 / 61


def test_fuse_rrf_breaks_ties_by_the_first_ranking() -> None:
    """Both chunks score 1/61 + 1/62; the dense ranking (passed first) decides."""
    fused = _fuse_rrf([["a", "b"], ["b", "a"]], rrf_k=60)

    assert [chunk_id for chunk_id, _ in fused] == ["a", "b"]
    assert fused[0][1] == pytest.approx(fused[1][1])


def test_fuse_rrf_on_empty_rankings_returns_nothing() -> None:
    assert _fuse_rrf([[], []], rrf_k=60) == []


# --- Hybrid retrieval ------------------------------------------------------


def _lexical_store(name: str) -> ChunkStore:
    """A store where the answer is findable lexically but not by `FakeEmbedder`.

    `FakeEmbedder` hashes text, so its "similarity" is semantically blind —
    which makes it a clean way to show that a passage surfaced in hybrid
    mode came from the BM25 side, not from the dense side.
    """
    store = ChunkStore(persist_dir=None, collection_name=name)
    chunks = [
        _chunk(f"filler{i}", f"unrelated paragraph {i} about serving requests") for i in range(8)
    ]
    chunks.append(_chunk("lexical", "set response_model_exclude_unset to omit default values"))
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return store


def test_hybrid_promotes_the_lexically_matching_passage_to_rank_one() -> None:
    store = _lexical_store("retriever_hybrid_lexical")
    question = "response_model_exclude_unset"

    dense = Retriever(embedder, ChunkingStrategy.MARKDOWN, RetrievalMode.DENSE, store=store)
    hybrid = Retriever(embedder, ChunkingStrategy.MARKDOWN, RetrievalMode.HYBRID, store=store)

    dense_ids = [p.chunk_id for p in dense.retrieve(question, top_k=3)]
    hybrid_ids = [p.chunk_id for p in hybrid.retrieve(question, top_k=3)]

    # The hash embedder has no reason to lead with the one chunk that
    # actually contains the term; BM25 ranks it first, and fusion carries it.
    assert dense_ids[0] != "lexical"
    assert hybrid_ids[0] == "lexical"


def test_hybrid_passages_carry_the_fused_score_and_full_metadata() -> None:
    store = _lexical_store("retriever_hybrid_metadata")
    retriever = Retriever(embedder, ChunkingStrategy.MARKDOWN, RetrievalMode.HYBRID, store=store)

    top = retriever.retrieve("response_model_exclude_unset", top_k=3)[0]

    assert top.source_file == "tutorial/path-params.md"
    assert top.section == "Order matters"
    assert top.text.startswith("set response_model_exclude_unset")
    # An RRF score, not a cosine similarity: best possible is 1/61 + 1/61.
    assert 0.0 < top.score <= 2 / 61


def test_hybrid_respects_top_k() -> None:
    store = _lexical_store("retriever_hybrid_top_k")
    retriever = Retriever(embedder, ChunkingStrategy.MARKDOWN, RetrievalMode.HYBRID, store=store)
    assert len(retriever.retrieve("serving requests", top_k=2)) == 2


def test_hybrid_falls_back_to_the_dense_ranking_when_nothing_matches_lexically() -> None:
    store = _lexical_store("retriever_hybrid_no_lexical_match")
    retriever = Retriever(embedder, ChunkingStrategy.MARKDOWN, RetrievalMode.HYBRID, store=store)

    passages = retriever.retrieve("qqq zzz wwww", top_k=3)

    assert len(passages) == 3
    assert all(
        p.score == pytest.approx(1 / (60 + rank)) for rank, p in enumerate(passages, start=1)
    )


def test_hybrid_leaves_rerank_score_unset() -> None:
    """The field exists on every passage, but only a re-ranking mode fills it in."""
    store = _lexical_store("retriever_hybrid_no_rerank_score")
    retriever = Retriever(embedder, ChunkingStrategy.MARKDOWN, RetrievalMode.HYBRID, store=store)

    assert all(p.rerank_score is None for p in retriever.retrieve("serving requests", top_k=3))


# --- Cross-encoder re-ranking ----------------------------------------------


def _rerank_retriever(
    name: str, reranker: FakeReranker, min_rerank_score: float | None = None
) -> Retriever:
    return Retriever(
        embedder,
        ChunkingStrategy.MARKDOWN,
        RetrievalMode.HYBRID_RERANK,
        store=_lexical_store(name),
        reranker=reranker,
        min_rerank_score=min_rerank_score,
    )


def test_rerank_reorders_the_fused_ranking() -> None:
    """The whole point: the cross-encoder's opinion overrides the fused order."""
    hybrid = Retriever(
        embedder,
        ChunkingStrategy.MARKDOWN,
        RetrievalMode.HYBRID,
        store=_lexical_store("retriever_rerank_baseline"),
    )
    fused_order = [p.chunk_id for p in hybrid.retrieve("serving requests", top_k=3)]

    # Promote whatever fusion ranked last of the three, demote what it ranked first.
    reranker = FakeReranker({fused_order[-1]: 0.99, fused_order[0]: 0.01}, default=0.5)
    reranked = _rerank_retriever("retriever_rerank_reorders", reranker, min_rerank_score=0.0)

    reranked_order = [p.chunk_id for p in reranked.retrieve("serving requests", top_k=3)]

    assert reranked_order[0] == fused_order[-1]
    assert reranked_order[0] != fused_order[0]


def test_rerank_keeps_the_fused_score_and_adds_the_rerank_score() -> None:
    """Two scores, two meanings — the RRF score survives the second stage."""
    reranker = FakeReranker(default=0.75)
    retriever = _rerank_retriever("retriever_rerank_scores", reranker, min_rerank_score=0.0)

    passages = retriever.retrieve("response_model_exclude_unset", top_k=3)

    assert all(p.rerank_score == 0.75 for p in passages)
    # Still an RRF score, untouched: best possible is 1/61 + 1/61.
    assert all(0.0 < p.score <= 2 / 61 for p in passages)


def test_rerank_scores_more_candidates_than_it_returns() -> None:
    """Re-ranking only k passages could never promote anything fusion ranked below k."""
    reranker = FakeReranker()
    retriever = _rerank_retriever("retriever_rerank_candidates", reranker, min_rerank_score=0.0)

    retriever.retrieve("serving requests", top_k=2)

    _, scored_ids = reranker.calls[0]
    assert len(scored_ids) == 9  # every chunk in the store, not just the top 2


def test_rerank_drops_candidates_below_the_relevance_floor() -> None:
    reranker = FakeReranker({"lexical": 0.9}, default=0.01)
    retriever = _rerank_retriever("retriever_rerank_floor", reranker, min_rerank_score=0.5)

    passages = retriever.retrieve("response_model_exclude_unset", top_k=5)

    assert [p.chunk_id for p in passages] == ["lexical"]


def test_rerank_returns_nothing_when_no_candidate_clears_the_floor() -> None:
    """An empty result is the abstention signal — see `generation.generate`."""
    retriever = _rerank_retriever(
        "retriever_rerank_abstains", FakeReranker(default=0.02), min_rerank_score=0.5
    )

    assert retriever.retrieve("how do I train a random forest?", top_k=5) == []


def test_rerank_floor_of_zero_keeps_everything() -> None:
    """What the threshold sweep in eval/run_eval.py relies on."""
    retriever = _rerank_retriever(
        "retriever_rerank_no_floor", FakeReranker(default=0.0), min_rerank_score=0.0
    )

    assert len(retriever.retrieve("serving requests", top_k=4)) == 4


@pytest.mark.parametrize("mode", [RetrievalMode.DENSE, RetrievalMode.HYBRID])
def test_modes_without_a_reranker_have_no_relevance_floor(mode: RetrievalMode) -> None:
    """A floor is a threshold on cross-encoder scores, so a mode with no cross-encoder
    reports none — rather than carrying a number that means nothing."""
    store = ChunkStore(persist_dir=None, collection_name=f"retriever_no_floor_{mode.value}")
    retriever = Retriever(
        embedder, ChunkingStrategy.MARKDOWN, mode, store=store, reranker=FakeReranker()
    )

    assert retriever.min_rerank_score is None


def test_rerank_mode_reports_its_floor() -> None:
    retriever = _rerank_retriever("retriever_reports_floor", FakeReranker(), min_rerank_score=0.42)
    assert retriever.min_rerank_score == 0.42


def test_rerank_floor_defaults_to_the_configured_value() -> None:
    store = _lexical_store("retriever_rerank_config_floor")
    retriever = Retriever(
        embedder,
        ChunkingStrategy.MARKDOWN,
        RetrievalMode.HYBRID_RERANK,
        store=store,
        reranker=FakeReranker(default=config.rerank_min_score - 0.01),
    )

    assert retriever.retrieve("serving requests", top_k=3) == []


def test_rerank_mode_still_searches_lexically() -> None:
    """Re-ranking is a stage added to hybrid search, not a replacement for it."""
    reranker = FakeReranker(default=0.5)
    retriever = _rerank_retriever("retriever_rerank_lexical", reranker, min_rerank_score=0.0)

    passages = retriever.retrieve("response_model_exclude_unset", top_k=3)

    # `lexical` is findable only by BM25 (FakeEmbedder is semantically blind),
    # and the fake re-ranker scores everything alike, so its presence proves
    # the fused candidates still came from both retrievers.
    assert "lexical" in {p.chunk_id for p in passages}

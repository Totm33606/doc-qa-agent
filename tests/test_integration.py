"""The only tests using the real embedder and cross-encoder (downloaded on first run).

Marked `integration`: skip locally with `pytest -m "not integration"`; CI runs them.
"""

from __future__ import annotations

import logging

import pytest

from common.config import config
from common.schemas import ChunkingStrategy, DocChunk, RetrievedPassage
from eval.run_eval import load_golden_set
from ingestion.embed import BGEEmbedder, QueryTooLongError
from ingestion.store import ChunkStore
from retrieval.rerank import CrossEncoderReranker
from retrieval.retriever import Retriever

pytestmark = pytest.mark.integration


def _chunk(chunk_id: str, text: str, source_file: str) -> DocChunk:
    return DocChunk(
        chunk_id=chunk_id,
        text=text,
        source_file=source_file,
        section="root",
        strategy=ChunkingStrategy.FIXED,
        token_count=len(text.split()),
        chunk_index=0,
    )


def test_real_embedder_ranks_the_topically_relevant_passage_first() -> None:
    embedder = BGEEmbedder()
    store = ChunkStore(persist_dir=None, collection_name="integration_ranking")

    chunks = [
        _chunk(
            "relevant",
            "FastAPI path operations are matched in the order they are declared, so a fixed "
            "path like /users/me must come before a variable path like /users/{user_id}.",
            "tutorial/path-params.md",
        ),
        _chunk(
            "unrelated",
            "SQLModel is built on top of SQLAlchemy and Pydantic, and lets you declare a "
            "database table with table=True.",
            "tutorial/sql-databases.md",
        ),
    ]
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))

    results = store.query(
        embedder.embed_query("Why does the order of FastAPI path operations matter?"), top_k=2
    )

    assert results[0].chunk_id == "relevant"


def test_real_embedder_query_vector_has_configured_dimension() -> None:
    embedder = BGEEmbedder()
    vector = embedder.embed_query("How do I declare a path parameter?")
    assert len(vector) == config.embedding_dim


def test_real_embedder_refuses_a_question_one_token_past_its_window() -> None:
    """502 one-token words + the 8-token instruction + [CLS]/[SEP] is exactly 512."""
    embedder = BGEEmbedder()

    assert len(embedder.embed_query("word " * 502)) == config.embedding_dim
    with pytest.raises(QueryTooLongError, match="513 tokens"):
        embedder.embed_query("word " * 503)


def _passage(chunk_id: str, text: str) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        text=text,
        source_file="tutorial/path-params.md",
        section="root",
        score=0.5,
    )


def test_real_reranker_scores_are_probabilities_in_the_unit_interval() -> None:
    """The relevance floor is a threshold on these, so the [0, 1] range is load-bearing.

    The checkpoint returns raw logits; the range comes from `_sigmoid` in `rerank.py`.
    """
    scores = CrossEncoderReranker().score(
        "Why does the order of FastAPI path operations matter?",
        [_passage("a", "Path operations are matched in declaration order."), _passage("b", "x")],
    )

    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_real_reranker_on_no_passages_returns_nothing() -> None:
    """Guards the empty-batch case (e.g. an empty collection)."""
    assert CrossEncoderReranker().score("anything", []) == []


def test_real_reranker_warns_when_a_long_question_overflows_its_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    passage = _passage("long", "FastAPI path operations are matched in declaration order. " * 35)
    reranker = CrossEncoderReranker()

    with caplog.at_level(logging.WARNING, logger="retrieval.rerank"):
        reranker.score("Why does path order matter?", [passage])
        assert not caplog.records
        reranker.score("Why does path order matter? " * 40, [passage])

    assert "truncated before scoring" in caplog.text


def test_real_reranker_prefers_the_topically_relevant_passage() -> None:
    scores = CrossEncoderReranker().score(
        "Why does the order of FastAPI path operations matter?",
        [
            _passage(
                "relevant",
                "FastAPI path operations are matched in the order they are declared, so a "
                "fixed path like /users/me must come before a variable path like "
                "/users/{user_id}.",
            ),
            _passage(
                "unrelated",
                "SQLModel is built on top of SQLAlchemy and Pydantic, and lets you declare a "
                "database table with table=True.",
            ),
        ],
    )

    assert scores[0] > scores[1]


@pytest.mark.skipif(
    not (config.chroma_dir / "chroma.sqlite3").exists(),
    reason="data/chroma not built — run `uv run python -m ingestion.build` first",
)
def test_built_markdown_collection_retrieves_expected_source_for_a_golden_question() -> None:
    embedder = BGEEmbedder()
    retriever = Retriever(embedder, ChunkingStrategy.MARKDOWN)
    question = load_golden_set()[0]

    passages = retriever.retrieve(question.question, top_k=5)
    retrieved_files = {p.source_file for p in passages}

    assert retrieved_files & set(question.expected_sources)

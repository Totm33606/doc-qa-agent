"""The one place this suite touches the real models — everything else is hermetic.

Marked `integration` (registered in pyproject.toml) purely so a local dev
loop can skip it with `pytest -m "not integration"`; CI runs the full
suite, including this file — downloading `BAAI/bge-small-en-v1.5` (~130MB)
and `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB), both free and needing
no API key, is a one-time cost similar in kind to any other dependency
download, not the kind of external-service dependency the rest of the suite
avoids.
"""

from __future__ import annotations

import pytest

from common.config import config
from common.schemas import ChunkingStrategy, DocChunk, RetrievedPassage
from eval.run_eval import load_golden_set
from ingestion.embed import BGEEmbedder
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

    It comes from the model having a single output label (sentence-transformers
    then applies a sigmoid), which is a fact about the checkpoint rather than
    about our code — so it is checked here rather than assumed.
    """
    scores = CrossEncoderReranker().score(
        "Why does the order of FastAPI path operations matter?",
        [_passage("a", "Path operations are matched in declaration order."), _passage("b", "x")],
    )

    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_real_reranker_on_no_passages_returns_nothing() -> None:
    """Guards the empty-batch case: `CrossEncoder.predict([])` is not something to find out
    about in production, and an abstention upstream can legitimately produce it."""
    assert CrossEncoderReranker().score("anything", []) == []


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

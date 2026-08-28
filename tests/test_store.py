from __future__ import annotations

from common.schemas import ChunkingStrategy, DocChunk
from ingestion.store import ChunkStore
from tests.conftest import FakeEmbedder

embedder = FakeEmbedder()


def _chunk(chunk_id: str, text: str, source_file: str = "a.md", section: str = "Intro") -> DocChunk:
    return DocChunk(
        chunk_id=chunk_id,
        text=text,
        source_file=source_file,
        section=section,
        strategy=ChunkingStrategy.FIXED,
        token_count=len(text.split()),
        chunk_index=0,
    )


def test_add_and_count() -> None:
    store = ChunkStore(persist_dir=None, collection_name="test_add_and_count")
    chunks = [_chunk("c1", "alpha beta"), _chunk("c2", "gamma delta")]
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    assert store.count() == 2


def test_add_empty_list_is_a_noop() -> None:
    store = ChunkStore(persist_dir=None, collection_name="test_add_empty")
    store.add([], [])
    assert store.count() == 0


def test_query_returns_passages_with_metadata() -> None:
    store = ChunkStore(persist_dir=None, collection_name="test_query_metadata")
    chunk = _chunk(
        "c1",
        "path parameters in fastapi",
        source_file="tutorial/path-params.md",
        section="Order matters",
    )
    store.add([chunk], embedder.embed_documents([chunk.text]))

    results = store.query(embedder.embed_query("path parameters in fastapi"), top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].source_file == "tutorial/path-params.md"
    assert results[0].section == "Order matters"
    assert 0.0 <= results[0].score <= 1.0 + 1e-6


def test_query_on_empty_collection_returns_empty_list() -> None:
    store = ChunkStore(persist_dir=None, collection_name="test_query_empty")
    results = store.query(embedder.embed_query("anything"), top_k=5)
    assert results == []


def test_query_respects_top_k() -> None:
    store = ChunkStore(persist_dir=None, collection_name="test_query_top_k")
    chunks = [_chunk(f"c{i}", f"chunk number {i} about routing") for i in range(10)]
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))

    results = store.query(embedder.embed_query("routing"), top_k=3)
    assert len(results) == 3


def test_query_ranks_more_similar_text_higher() -> None:
    store = ChunkStore(persist_dir=None, collection_name="test_query_ranking")
    exact = _chunk("exact", "how to declare a path parameter with a type in fastapi")
    unrelated = _chunk("unrelated", "zzz completely different filler content qqq")
    store.add([exact, unrelated], embedder.embed_documents([exact.text, unrelated.text]))

    results = store.query(
        embedder.embed_query("how to declare a path parameter with a type in fastapi"), top_k=2
    )
    assert results[0].chunk_id == "exact"


def test_reset_clears_the_collection() -> None:
    store = ChunkStore(persist_dir=None, collection_name="test_reset")
    chunk = _chunk("c1", "some text")
    store.add([chunk], embedder.embed_documents([chunk.text]))
    assert store.count() == 1

    store.reset()
    assert store.count() == 0


def test_persistent_store_survives_reopening(tmp_path: object) -> None:
    from pathlib import Path

    persist_dir = Path(str(tmp_path)) / "chroma"
    collection_name = "test_persistent"

    store1 = ChunkStore(persist_dir=persist_dir, collection_name=collection_name)
    chunk = _chunk("c1", "persisted content")
    store1.add([chunk], embedder.embed_documents([chunk.text]))

    store2 = ChunkStore(persist_dir=persist_dir, collection_name=collection_name)
    assert store2.count() == 1

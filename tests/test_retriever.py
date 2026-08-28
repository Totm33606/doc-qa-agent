from __future__ import annotations

from common.schemas import ChunkingStrategy, DocChunk
from ingestion.store import ChunkStore
from retrieval.retriever import Retriever
from tests.conftest import FakeEmbedder

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
    retriever = Retriever(embedder, ChunkingStrategy.MARKDOWN, store=store)

    passages = retriever.retrieve("declare path parameters with types in fastapi", top_k=5)

    assert len(passages) == 1
    assert passages[0].chunk_id == "c1"


def test_retrieve_uses_default_top_k_when_not_specified() -> None:
    store = ChunkStore(persist_dir=None, collection_name="retriever_default_k")
    chunks = [_chunk(f"c{i}", f"chunk {i} content") for i in range(10)]
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    retriever = Retriever(embedder, ChunkingStrategy.FIXED, store=store)

    from ingestion.config import config

    passages = retriever.retrieve("chunk content")
    assert len(passages) == config.default_top_k


def test_retriever_exposes_its_strategy() -> None:
    store = ChunkStore(persist_dir=None, collection_name="retriever_strategy")
    retriever = Retriever(embedder, ChunkingStrategy.MARKDOWN, store=store)
    assert retriever.strategy is ChunkingStrategy.MARKDOWN


def test_retrieve_on_empty_store_returns_empty_list() -> None:
    store = ChunkStore(persist_dir=None, collection_name="retriever_empty")
    retriever = Retriever(embedder, ChunkingStrategy.FIXED, store=store)
    assert retriever.retrieve("anything") == []

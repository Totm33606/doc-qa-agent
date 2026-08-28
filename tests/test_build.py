from __future__ import annotations

from pathlib import Path

import pytest

from common.schemas import ChunkingStrategy, DocChunk
from ingestion.build import _batched, build_collection, run
from ingestion.config import config
from ingestion.store import ChunkStore
from tests.conftest import FakeEmbedder

embedder = FakeEmbedder()


def _make_corpus(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "a.md").write_text("# A\n\nSome content about topic A.\n", encoding="utf-8")
    (raw_dir / "b.md").write_text("# B\n\nSome content about topic B.\n", encoding="utf-8")
    return raw_dir


def test_batched_splits_into_expected_group_sizes() -> None:
    chunks = [
        DocChunk(
            chunk_id=str(i),
            text="x",
            source_file="a.md",
            section="s",
            strategy=ChunkingStrategy.FIXED,
            token_count=1,
            chunk_index=i,
        )
        for i in range(5)
    ]
    batches = list(_batched(chunks, batch_size=2))
    assert [len(b) for b in batches] == [2, 2, 1]


def test_batched_empty_list_yields_nothing() -> None:
    assert list(_batched([], batch_size=10)) == []


def test_build_collection_chunks_embeds_and_stores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "raw_docs_dir", _make_corpus(tmp_path))
    monkeypatch.setattr(config, "chroma_dir", tmp_path / "chroma")

    n_chunks = build_collection(ChunkingStrategy.FIXED, embedder)

    assert n_chunks == 2  # one chunk per tiny file, well under the token budget
    store = ChunkStore(
        persist_dir=config.chroma_dir, collection_name=config.collection_name("fixed")
    )
    assert store.count() == n_chunks


def test_build_collection_is_safe_to_rerun(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`build_collection` resets the collection each call — re-running must not duplicate."""
    monkeypatch.setattr(config, "raw_docs_dir", _make_corpus(tmp_path))
    monkeypatch.setattr(config, "chroma_dir", tmp_path / "chroma")

    first = build_collection(ChunkingStrategy.MARKDOWN, embedder)
    second = build_collection(ChunkingStrategy.MARKDOWN, embedder)

    assert first == second
    store = ChunkStore(
        persist_dir=config.chroma_dir, collection_name=config.collection_name("markdown")
    )
    assert store.count() == second


def test_build_collection_raises_when_corpus_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty_dir = tmp_path / "empty_raw"
    empty_dir.mkdir()
    monkeypatch.setattr(config, "raw_docs_dir", empty_dir)
    monkeypatch.setattr(config, "chroma_dir", tmp_path / "chroma")

    with pytest.raises(FileNotFoundError):
        build_collection(ChunkingStrategy.FIXED, embedder)


def test_run_builds_both_strategies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "raw_docs_dir", _make_corpus(tmp_path))
    monkeypatch.setattr(config, "chroma_dir", tmp_path / "chroma")
    monkeypatch.setattr("ingestion.build.BGEEmbedder", lambda: embedder)

    run(batch_size=64)

    for strategy in ChunkingStrategy:
        store = ChunkStore(
            persist_dir=config.chroma_dir, collection_name=config.collection_name(strategy.value)
        )
        assert store.count() > 0

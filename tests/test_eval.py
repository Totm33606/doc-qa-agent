from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import eval.run_eval as run_eval_module
from common.schemas import AskResponse, ChunkingStrategy, DocChunk, GoldenQuestion, RetrievedPassage
from eval.run_eval import (
    _reciprocal_rank,
    evaluate_generation,
    evaluate_retrieval,
    load_golden_set,
    run,
    run_all,
)
from ingestion.config import config
from ingestion.store import ChunkStore
from retrieval.retriever import Retriever
from tests.conftest import FakeChatModel, FakeEmbedder

embedder = FakeEmbedder()


def test_load_golden_set_parses_the_real_file() -> None:
    questions = load_golden_set()
    assert len(questions) >= 30
    assert all(q.expected_sources for q in questions)
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids))


def test_reciprocal_rank_first_position() -> None:
    assert _reciprocal_rank(["a.md", "b.md"], {"a.md"}) == 1.0


def test_reciprocal_rank_second_position() -> None:
    assert _reciprocal_rank(["b.md", "a.md"], {"a.md"}) == 0.5


def test_reciprocal_rank_no_match_is_zero() -> None:
    assert _reciprocal_rank(["b.md", "c.md"], {"a.md"}) == 0.0


def _chunk(chunk_id: str, source_file: str) -> DocChunk:
    return DocChunk(
        chunk_id=chunk_id,
        text=f"content for {source_file}",
        source_file=source_file,
        section="root",
        strategy=ChunkingStrategy.FIXED,
        token_count=5,
        chunk_index=0,
    )


def _retriever_with(source_files: list[str]) -> Retriever:
    store = ChunkStore(persist_dir=None, collection_name=f"eval_test_{'_'.join(source_files)}")
    chunks = [_chunk(f"c{i}", f) for i, f in enumerate(source_files)]
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return Retriever(embedder, ChunkingStrategy.FIXED, store=store)


def test_evaluate_retrieval_perfect_precision_and_recall() -> None:
    retriever = _retriever_with(["a.md", "b.md"])
    questions = [
        GoldenQuestion(
            id="q1",
            question="content for a.md",
            expected_answer="n/a",
            expected_sources=["a.md"],
            category="test",
        )
    ]
    metrics = evaluate_retrieval(questions, retriever, k=1)
    assert metrics.precision_at_k == 1.0
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0
    assert metrics.n_questions == 1


def test_evaluate_retrieval_zero_when_nothing_expected_matches() -> None:
    retriever = _retriever_with(["a.md"])
    questions = [
        GoldenQuestion(
            id="q1",
            question="content for a.md",
            expected_answer="n/a",
            expected_sources=["z.md"],
            category="test",
        )
    ]
    metrics = evaluate_retrieval(questions, retriever, k=1)
    assert metrics.precision_at_k == 0.0
    assert metrics.recall_at_k == 0.0
    assert metrics.mrr == 0.0


def test_evaluate_generation_uses_groundedness_from_generate_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _stub_generate_answer(question: str, passages: list[RetrievedPassage]) -> AskResponse:
        return AskResponse(
            question=question,
            answer="stub",
            citations=[],
            passages=passages,
            groundedness_score=0.75,
        )

    monkeypatch.setattr(run_eval_module, "generate_answer", _stub_generate_answer)
    retriever = _retriever_with(["a.md"])
    questions = [
        GoldenQuestion(
            id="q1",
            question="content for a.md",
            expected_answer="n/a",
            expected_sources=["a.md"],
            category="test",
        )
    ]
    metrics = evaluate_generation(questions, retriever, k=1)
    assert metrics.mean_groundedness == 0.75


def _seed_both_collections(chroma_dir: Path) -> None:
    for strategy in ChunkingStrategy:
        store = ChunkStore(
            persist_dir=chroma_dir, collection_name=config.collection_name(strategy.value)
        )
        chunk = _chunk(f"{strategy.value}-c1", "a.md")
        store.add([chunk], embedder.embed_documents([chunk.text]))


_FAKE_QUESTIONS = [
    GoldenQuestion(
        id="q1",
        question="content for a.md",
        expected_answer="n/a",
        expected_sources=["a.md"],
        category="test",
    )
]


def test_run_all_builds_a_report_entry_per_strategy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "chroma_dir", tmp_path / "chroma")
    _seed_both_collections(config.chroma_dir)
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: _FAKE_QUESTIONS)

    report = run_all(embedder, k=1, skip_generation=True)

    assert report == {
        "k": 1,
        "n_questions": 1,
        "strategies": {
            "fixed": {
                "retrieval": {
                    "strategy": "fixed",
                    "k": 1,
                    "precision_at_k": 1.0,
                    "recall_at_k": 1.0,
                    "mrr": 1.0,
                    "n_questions": 1,
                }
            },
            "markdown": {
                "retrieval": {
                    "strategy": "markdown",
                    "k": 1,
                    "precision_at_k": 1.0,
                    "recall_at_k": 1.0,
                    "mrr": 1.0,
                    "n_questions": 1,
                }
            },
        },
    }


def test_run_all_includes_generation_unless_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "chroma_dir", tmp_path / "chroma")
    _seed_both_collections(config.chroma_dir)
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: _FAKE_QUESTIONS)
    monkeypatch.setattr(
        "generation.generate.build_llm", lambda: FakeChatModel("An answer. [source: 1]")
    )

    report = run_all(embedder, k=1, skip_generation=False)

    strategies = cast("dict[str, dict[str, dict[str, object]]]", report["strategies"])
    for entry in strategies.values():
        assert entry["generation"]["mean_groundedness"] == 1.0


def test_run_command_writes_report_to_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "chroma_dir", tmp_path / "chroma")
    _seed_both_collections(config.chroma_dir)
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: _FAKE_QUESTIONS)
    monkeypatch.setattr(run_eval_module, "BGEEmbedder", lambda: embedder)
    report_path = tmp_path / "eval_report.json"
    monkeypatch.setattr(run_eval_module, "REPORT_PATH", report_path)

    run(k=1, skip_generation=True)

    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["n_questions"] == 1
    assert set(written["strategies"].keys()) == {"fixed", "markdown"}

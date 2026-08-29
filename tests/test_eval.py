from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import eval.run_eval as run_eval_module
from common.schemas import (
    AskResponse,
    ChunkingStrategy,
    Citation,
    DocChunk,
    GoldenQuestion,
    QuestionResult,
    RetrievalMetrics,
    RetrievedPassage,
)
from eval.run_eval import (
    _reciprocal_rank,
    evaluate_question,
    generation_metrics,
    load_golden_set,
    retrieval_metrics,
    run,
    run_all,
    write_details_markdown,
)
from ingestion.config import config
from ingestion.store import ChunkStore
from retrieval.retriever import Retriever
from tests.conftest import FakeChatModel, FakeEmbedder

embedder = FakeEmbedder()


def _retrieval_metrics_for(
    questions: list[GoldenQuestion], retriever: Retriever, k: int
) -> RetrievalMetrics:
    """Test-only composition mirroring what `run_all` does: score each question, aggregate."""
    results = [evaluate_question(q, retriever, k=k, skip_generation=True) for q in questions]
    return retrieval_metrics(results, retriever.strategy, k)


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


def test_retrieval_metrics_perfect_precision_and_recall() -> None:
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
    metrics = _retrieval_metrics_for(questions, retriever, k=1)
    assert metrics.precision_at_k == 1.0
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0
    assert metrics.n_questions == 1


def test_retrieval_metrics_zero_when_nothing_expected_matches() -> None:
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
    metrics = _retrieval_metrics_for(questions, retriever, k=1)
    assert metrics.precision_at_k == 0.0
    assert metrics.recall_at_k == 0.0
    assert metrics.mrr == 0.0


class _StubRetriever:
    """Returns a fixed, hand-chosen ranking regardless of the query — bypasses embedding
    entirely so precision@k/recall@k/MRR can be checked against a hand-computed value,
    not just the 0.0/1.0 extremes the tests above cover (which can't catch an off-by-one
    in the rank/denominator arithmetic)."""

    strategy = ChunkingStrategy.FIXED

    def __init__(self, passages_by_question: dict[str, list[RetrievedPassage]]) -> None:
        self._passages_by_question = passages_by_question

    def retrieve(self, question: str, top_k: int) -> list[RetrievedPassage]:
        return self._passages_by_question[question][:top_k]


def _passage(source_file: str) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=source_file, text="x", source_file=source_file, section="s", score=0.9
    )


def test_retrieval_metrics_partial_hit_matches_hand_computed_values() -> None:
    """One question, top-3, hit at rank 2, one of two expected files never retrieved:
    precision@3 = 1/3, recall@3 = 1/2, MRR = 1/2 (reciprocal of rank 2)."""
    retriever = _StubRetriever({"q1": [_passage("b.md"), _passage("a.md"), _passage("d.md")]})
    questions = [
        GoldenQuestion(
            id="q1",
            question="q1",
            expected_answer="n/a",
            expected_sources=["a.md", "c.md"],
            category="test",
        )
    ]

    metrics = _retrieval_metrics_for(questions, retriever, k=3)  # type: ignore[arg-type]

    assert metrics.precision_at_k == pytest.approx(1 / 3)
    assert metrics.recall_at_k == pytest.approx(1 / 2)
    assert metrics.mrr == pytest.approx(1 / 2)


def test_retrieval_metrics_averages_correctly_across_questions() -> None:
    """Two questions — one perfect hit at rank 1, one total miss — averaged, not summed."""
    retriever = _StubRetriever(
        {
            "q1": [_passage("a.md"), _passage("x.md"), _passage("y.md")],
            "q2": [_passage("b.md"), _passage("a.md"), _passage("d.md")],
        }
    )
    questions = [
        GoldenQuestion(
            id="q1",
            question="q1",
            expected_answer="n/a",
            expected_sources=["a.md"],
            category="test",
        ),
        GoldenQuestion(
            id="q2",
            question="q2",
            expected_answer="n/a",
            expected_sources=["z.md"],
            category="test",
        ),
    ]

    metrics = _retrieval_metrics_for(questions, retriever, k=3)  # type: ignore[arg-type]

    assert metrics.precision_at_k == pytest.approx((1 / 3 + 0) / 2)
    assert metrics.recall_at_k == pytest.approx((1.0 + 0) / 2)
    assert metrics.mrr == pytest.approx((1.0 + 0) / 2)


def test_generation_metrics_uses_groundedness_from_generate_answer(
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
    question = GoldenQuestion(
        id="q1",
        question="content for a.md",
        expected_answer="n/a",
        expected_sources=["a.md"],
        category="test",
    )
    result = evaluate_question(question, retriever, k=1, skip_generation=False)

    metrics = generation_metrics([result], retriever.strategy)

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


def test_evaluate_question_skip_generation_leaves_generation_fields_none() -> None:
    retriever = _retriever_with(["a.md"])
    question = _FAKE_QUESTIONS[0]

    result = evaluate_question(question, retriever, k=1, skip_generation=True)

    assert result.strategy is ChunkingStrategy.FIXED
    assert result.precision == 1.0
    assert result.generated_answer is None
    assert result.citations == []
    assert result.groundedness_score is None


def test_evaluate_question_includes_generation_when_not_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = _retriever_with(["a.md"])
    monkeypatch.setattr(
        "generation.generate.build_llm", lambda: FakeChatModel("An answer. [source: 1]")
    )

    result = evaluate_question(_FAKE_QUESTIONS[0], retriever, k=1, skip_generation=False)

    assert result.generated_answer == "An answer. [source: 1]"
    assert result.groundedness_score == 1.0
    assert len(result.citations) == 1


def test_run_all_builds_a_report_entry_per_strategy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "chroma_dir", tmp_path / "chroma")
    _seed_both_collections(config.chroma_dir)
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: _FAKE_QUESTIONS)

    report, results = run_all(embedder, k=1, skip_generation=True)

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
    assert len(results) == 2  # one row per strategy, for a single golden question
    assert {r.strategy for r in results} == {ChunkingStrategy.FIXED, ChunkingStrategy.MARKDOWN}


def test_run_all_includes_generation_unless_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "chroma_dir", tmp_path / "chroma")
    _seed_both_collections(config.chroma_dir)
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: _FAKE_QUESTIONS)
    monkeypatch.setattr(
        "generation.generate.build_llm", lambda: FakeChatModel("An answer. [source: 1]")
    )

    report, results = run_all(embedder, k=1, skip_generation=False)

    strategies = cast("dict[str, dict[str, dict[str, object]]]", report["strategies"])
    for entry in strategies.values():
        assert entry["generation"]["mean_groundedness"] == 1.0
    assert all(r.groundedness_score == 1.0 for r in results)


def test_write_details_markdown_includes_every_row(tmp_path: Path) -> None:
    retriever = _retriever_with(["a.md"])
    result = evaluate_question(_FAKE_QUESTIONS[0], retriever, k=1, skip_generation=True)
    out_path = tmp_path / "details.md"

    write_details_markdown([result], out_path, k=1)

    content = out_path.read_text(encoding="utf-8")
    assert "## Strategy: fixed" in content
    assert "q1" in content
    assert "content for a.md" in content
    assert "precision=1.00" in content


def test_write_details_markdown_shows_generated_answer_and_citations(tmp_path: Path) -> None:
    result = QuestionResult(
        strategy=ChunkingStrategy.FIXED,
        question_id="q1",
        question="q",
        category="test",
        expected_answer="ref",
        expected_sources=["a.md"],
        retrieved_sources=["a.md"],
        precision=1.0,
        recall=1.0,
        reciprocal_rank=1.0,
        generated_answer="Line one.\nLine two. [source: 1]",
        citations=[Citation(source_file="a.md", section="s", matched_passage=True)],
        groundedness_score=1.0,
    )
    out_path = tmp_path / "details.md"

    write_details_markdown([result], out_path, k=1)

    content = out_path.read_text(encoding="utf-8")
    assert "> Line one." in content
    assert "> Line two. [source: 1]" in content
    assert "a.md#s" in content


def test_write_details_markdown_shows_none_when_answer_has_no_citations(tmp_path: Path) -> None:
    result = QuestionResult(
        strategy=ChunkingStrategy.FIXED,
        question_id="q1",
        question="q",
        category="test",
        expected_answer="ref",
        expected_sources=["a.md"],
        retrieved_sources=["a.md"],
        precision=1.0,
        recall=1.0,
        reciprocal_rank=1.0,
        generated_answer="An answer with no citation markers at all.",
        citations=[],
        groundedness_score=0.0,
    )
    out_path = tmp_path / "details.md"

    write_details_markdown([result], out_path, k=1)

    assert "**Citations:** (none)" in out_path.read_text(encoding="utf-8")


def test_run_command_writes_report_and_details_to_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "chroma_dir", tmp_path / "chroma")
    _seed_both_collections(config.chroma_dir)
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: _FAKE_QUESTIONS)
    monkeypatch.setattr(run_eval_module, "BGEEmbedder", lambda: embedder)
    report_path = tmp_path / "eval_report.json"
    details_path = tmp_path / "eval_details.md"
    monkeypatch.setattr(run_eval_module, "REPORT_PATH", report_path)
    monkeypatch.setattr(run_eval_module, "DETAILS_PATH", details_path)

    run(k=1, skip_generation=True)

    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["n_questions"] == 1
    assert set(written["strategies"].keys()) == {"fixed", "markdown"}

    details = details_path.read_text(encoding="utf-8")
    assert "## Strategy: fixed" in details
    assert "## Strategy: markdown" in details

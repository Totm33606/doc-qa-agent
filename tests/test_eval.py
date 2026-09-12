from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import eval.run_eval as run_eval_module
from common.config import config
from common.schemas import (
    AskResponse,
    ChunkingStrategy,
    Citation,
    DocChunk,
    GoldenQuestion,
    OutOfDomainQuestion,
    QuestionResult,
    RetrievalMetrics,
    RetrievalMode,
    RetrievedPassage,
)
from eval.run_eval import (
    _best_rerank_scores,
    _reciprocal_rank,
    abstained,
    abstention_metrics,
    evaluate_question,
    generation_metrics,
    load_golden_set,
    load_out_of_domain,
    retrieval_metrics,
    run,
    run_all,
    sweep_threshold,
    write_details_markdown,
)
from ingestion.store import ChunkStore
from retrieval.retriever import Retriever
from tests.conftest import FakeChatModel, FakeEmbedder, FakeReranker

embedder = FakeEmbedder()
# Scores every candidate alike and well above any floor: these tests are about
# the harness's aggregation, not about the re-ranker's judgement.
reranker = FakeReranker(default=0.9)


def _retrieval_metrics_for(
    questions: list[GoldenQuestion], retriever: Retriever, k: int
) -> RetrievalMetrics:
    """Test-only composition mirroring what `run_all` does: score each question, aggregate."""
    results = [evaluate_question(q, retriever, k=k, skip_generation=True) for q in questions]
    return retrieval_metrics(results, retriever.strategy, retriever.mode, k)


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
    """Dense on purpose: the tests below check the metric arithmetic, not the fusion."""
    store = ChunkStore(persist_dir=None, collection_name=f"eval_test_{'_'.join(source_files)}")
    chunks = [_chunk(f"c{i}", f) for i, f in enumerate(source_files)]
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return Retriever(embedder, ChunkingStrategy.FIXED, RetrievalMode.DENSE, store=store)


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
    """Fixed ranking per question, so metrics can be checked against hand-computed values
    rather than only 0.0/1.0. An empty ranking stands in for an abstention."""

    strategy = ChunkingStrategy.FIXED
    mode = RetrievalMode.DENSE

    def __init__(self, passages_by_question: dict[str, list[RetrievedPassage]]) -> None:
        self._passages_by_question = passages_by_question

    def retrieve(self, question: str, top_k: int) -> list[RetrievedPassage]:
        return self._passages_by_question[question][:top_k]


def _passage(source_file: str, rerank_score: float | None = None) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=source_file,
        text="x",
        source_file=source_file,
        section="s",
        score=0.9,
        rerank_score=rerank_score,
    )


def _question_result(groundedness: float, abstained: bool) -> QuestionResult:
    """One eval row, reduced to the two fields the aggregation tests care about."""
    return QuestionResult(
        strategy=ChunkingStrategy.FIXED,
        mode=RetrievalMode.HYBRID_RERANK,
        question_id="q1",
        question="q",
        category="test",
        expected_answer="ref",
        expected_sources=["a.md"],
        retrieved_sources=[] if abstained else ["a.md"],
        precision=0.0 if abstained else 1.0,
        recall=0.0 if abstained else 1.0,
        reciprocal_rank=0.0 if abstained else 1.0,
        abstained=abstained,
        generated_answer="stub",
        groundedness_score=groundedness,
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

    metrics = generation_metrics([result], retriever.strategy, retriever.mode)

    assert metrics.mean_groundedness == 0.75


def test_generation_metrics_excludes_abstentions_from_the_mean() -> None:
    """A refusal has no citations and scores 0.0 — averaging it in would punish the
    mode for correctly declining, which is the behaviour the floor exists to produce."""
    answered = _question_result(groundedness=1.0, abstained=False)
    refused = _question_result(groundedness=0.0, abstained=True)

    metrics = generation_metrics(
        [answered, answered, refused], ChunkingStrategy.FIXED, RetrievalMode.HYBRID_RERANK
    )

    assert metrics.mean_groundedness == 1.0  # not 2/3
    assert metrics.n_questions == 2  # the denominator says so
    assert metrics.abstention_rate == pytest.approx(1 / 3)


def test_generation_metrics_when_everything_was_refused() -> None:
    """No answered rows means no mean to take — 0.0, not a ZeroDivisionError."""
    metrics = generation_metrics(
        [_question_result(groundedness=0.0, abstained=True)],
        ChunkingStrategy.FIXED,
        RetrievalMode.HYBRID_RERANK,
    )

    assert metrics.mean_groundedness == 0.0
    assert metrics.n_questions == 0
    assert metrics.abstention_rate == 1.0


def test_evaluate_question_marks_an_empty_retrieval_as_an_abstention() -> None:
    retriever = _StubRetriever({"q1": []})

    result = evaluate_question(
        _FAKE_QUESTIONS[0].model_copy(update={"question": "q1"}),
        retriever,  # type: ignore[arg-type]
        k=3,
        skip_generation=True,
    )

    assert result.abstained is True
    assert result.retrieved_sources == []


def test_evaluate_question_does_not_mark_a_normal_retrieval_as_an_abstention() -> None:
    retriever = _retriever_with(["a.md"])

    result = evaluate_question(_FAKE_QUESTIONS[0], retriever, k=1, skip_generation=True)

    assert result.abstained is False


def test_abstention_metrics_reduces_both_error_rates() -> None:
    """Both rates at once: 1 of 2 golden questions wrongly refused, 1 of 2 out-of-domain
    questions rightly refused. Hand-computed, and independent of any retriever."""
    metrics = abstention_metrics(
        ChunkingStrategy.FIXED,
        RetrievalMode.HYBRID_RERANK,
        0.2,
        [
            _question_result(groundedness=1.0, abstained=False),
            _question_result(groundedness=0.0, abstained=True),
        ],
        [True, False],
    )

    assert metrics.false_abstention_rate == 0.5
    assert metrics.correct_abstention_rate == 0.5
    assert metrics.n_in_domain == 2
    assert metrics.n_out_of_domain == 2
    assert metrics.threshold == 0.2


def test_abstention_metrics_with_no_questions_does_not_divide_by_zero() -> None:
    metrics = abstention_metrics(ChunkingStrategy.FIXED, RetrievalMode.DENSE, None, [], [])

    assert metrics.false_abstention_rate == 0.0
    assert metrics.correct_abstention_rate == 0.0
    assert metrics.threshold is None


def test_abstained_reads_an_empty_retrieval_as_a_refusal() -> None:
    """The signal the out-of-domain side is built on, named rather than inlined."""
    retriever = _StubRetriever({"refused": [], "answered": [_passage("a.md")]})

    assert abstained(retriever, "refused", k=3) is True  # type: ignore[arg-type]
    assert abstained(retriever, "answered", k=3) is False  # type: ignore[arg-type]


def test_load_out_of_domain_parses_the_real_file() -> None:
    questions = load_out_of_domain()
    assert len(questions) >= 5
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids))
    assert all(q.why for q in questions)


def test_best_rerank_scores_takes_the_top_candidate_per_question() -> None:
    """The gate keeps a question if *any* candidate clears the floor, so the best
    score is the only one a threshold sweep needs."""
    retriever = _StubRetriever(
        {
            "q1": [_passage("a.md", rerank_score=0.4), _passage("b.md", rerank_score=0.9)],
            "q2": [],
        }
    )

    scores = _best_rerank_scores(retriever, ["q1", "q2"], k=5)  # type: ignore[arg-type]

    assert scores == [0.9, 0.0]


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


_FAKE_OUT_OF_DOMAIN = [
    OutOfDomainQuestion(
        id="ood1", question="something the store has nothing about", category="test", why="n/a"
    )
]


def _expected_retrieval_entry(strategy: str, mode: str) -> dict[str, object]:
    return {
        "retrieval": {
            "strategy": strategy,
            "mode": mode,
            "k": 1,
            "precision_at_k": 1.0,
            "recall_at_k": 1.0,
            "mrr": 1.0,
            "n_questions": 1,
        },
        "abstention": {
            "strategy": strategy,
            "mode": mode,
            # Only the re-ranking mode has a floor at all; the other two report
            # null, which is what makes their zero rates readable as "cannot
            # abstain" rather than "did not happen to".
            "threshold": config.rerank_min_score if mode == "hybrid_rerank" else None,
            # The seeded store answers everything (one chunk, always retrieved)
            # and the fake re-ranker scores it well above the floor, so nothing
            # is refused either rightly or wrongly.
            "false_abstention_rate": 0.0,
            "correct_abstention_rate": 0.0,
            "n_in_domain": 1,
            "n_out_of_domain": 1,
        },
    }


def test_run_all_builds_a_report_entry_per_strategy_and_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "chroma_dir", tmp_path / "chroma")
    _seed_both_collections(config.chroma_dir)
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: _FAKE_QUESTIONS)
    monkeypatch.setattr(run_eval_module, "load_out_of_domain", lambda: _FAKE_OUT_OF_DOMAIN)

    report, results = run_all(embedder, reranker, k=1, skip_generation=True)

    assert report == {
        "k": 1,
        "n_questions": 1,
        "n_out_of_domain": 1,
        "strategies": {
            strategy.value: {
                mode.value: _expected_retrieval_entry(strategy.value, mode.value)
                for mode in RetrievalMode
            }
            for strategy in ChunkingStrategy
        },
    }
    # One row per (strategy, mode) pair, for a single golden question.
    assert len(results) == len(ChunkingStrategy) * len(RetrievalMode)
    assert {(r.strategy, r.mode) for r in results} == {
        (s, m) for s in ChunkingStrategy for m in RetrievalMode
    }


def test_run_all_includes_generation_unless_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "chroma_dir", tmp_path / "chroma")
    _seed_both_collections(config.chroma_dir)
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: _FAKE_QUESTIONS)
    monkeypatch.setattr(run_eval_module, "load_out_of_domain", lambda: _FAKE_OUT_OF_DOMAIN)
    monkeypatch.setattr(
        "generation.generate.build_llm", lambda: FakeChatModel("An answer. [source: 1]")
    )

    report, results = run_all(embedder, reranker, k=1, skip_generation=False)

    strategies = cast("dict[str, dict[str, dict[str, dict[str, object]]]]", report["strategies"])
    for by_mode in strategies.values():
        for entry in by_mode.values():
            assert entry["generation"]["mean_groundedness"] == 1.0
    assert all(r.groundedness_score == 1.0 for r in results)


def test_sweep_threshold_scores_the_whole_grid_from_one_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two thresholds bracketing the fake re-ranker's score must fall on either side of it:
    below it nothing is refused, above it everything is."""
    monkeypatch.setattr(config, "chroma_dir", tmp_path / "chroma")
    _seed_both_collections(config.chroma_dir)
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: _FAKE_QUESTIONS)
    monkeypatch.setattr(run_eval_module, "load_out_of_domain", lambda: _FAKE_OUT_OF_DOMAIN)
    monkeypatch.setattr(run_eval_module, "THRESHOLD_GRID", (0.1, 0.9))

    sweep = cast(
        "dict[str, dict[str, object]]",
        sweep_threshold(embedder, FakeReranker(default=0.5), k=1),
    )

    assert set(sweep.keys()) == {s.value for s in ChunkingStrategy}
    grid = cast("list[dict[str, float]]", sweep["fixed"]["grid"])
    assert grid[0] == {
        "threshold": 0.1,
        "false_abstention_rate": 0.0,
        "correct_abstention_rate": 0.0,
    }
    assert grid[1] == {
        "threshold": 0.9,
        "false_abstention_rate": 1.0,
        "correct_abstention_rate": 1.0,
    }
    assert sweep["fixed"]["in_domain_min"] == 0.5
    assert sweep["fixed"]["out_of_domain_max"] == 0.5


def test_sweep_threshold_ignores_the_configured_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The sweep has to see the scores it would gate on, so it must retrieve unfiltered —
    otherwise the floor being measured would silently pre-filter its own measurement."""
    monkeypatch.setattr(config, "chroma_dir", tmp_path / "chroma")
    _seed_both_collections(config.chroma_dir)
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: _FAKE_QUESTIONS)
    monkeypatch.setattr(run_eval_module, "load_out_of_domain", lambda: _FAKE_OUT_OF_DOMAIN)
    monkeypatch.setattr(config, "rerank_min_score", 0.99)

    sweep = cast(
        "dict[str, dict[str, object]]",
        sweep_threshold(embedder, FakeReranker(default=0.5), k=1),
    )

    # 0.5 is below the configured floor; had the floor applied, this would be 0.0.
    assert sweep["fixed"]["in_domain_min"] == 0.5


def test_run_command_can_append_the_threshold_sweep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "chroma_dir", tmp_path / "chroma")
    _seed_both_collections(config.chroma_dir)
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: _FAKE_QUESTIONS)
    monkeypatch.setattr(run_eval_module, "load_out_of_domain", lambda: _FAKE_OUT_OF_DOMAIN)
    monkeypatch.setattr(run_eval_module, "BGEEmbedder", lambda: embedder)
    monkeypatch.setattr(run_eval_module, "CrossEncoderReranker", lambda: reranker)
    report_path = tmp_path / "eval_report.json"
    monkeypatch.setattr(run_eval_module, "REPORT_PATH", report_path)
    monkeypatch.setattr(run_eval_module, "DETAILS_PATH", tmp_path / "eval_details.md")

    run(k=1, skip_generation=True, sweep=True)

    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(written["threshold_sweep"].keys()) == {s.value for s in ChunkingStrategy}


def test_write_details_markdown_includes_every_row(tmp_path: Path) -> None:
    retriever = _retriever_with(["a.md"])
    result = evaluate_question(_FAKE_QUESTIONS[0], retriever, k=1, skip_generation=True)
    out_path = tmp_path / "details.md"

    write_details_markdown([result], out_path, k=1)

    content = out_path.read_text(encoding="utf-8")
    assert "## Strategy: fixed · mode: dense" in content
    assert "q1" in content
    assert "content for a.md" in content
    assert "precision=1.00" in content


def test_write_details_markdown_shows_generated_answer_and_citations(tmp_path: Path) -> None:
    result = QuestionResult(
        strategy=ChunkingStrategy.FIXED,
        mode=RetrievalMode.DENSE,
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


def test_write_details_markdown_flags_an_abstention(tmp_path: Path) -> None:
    out_path = tmp_path / "details.md"

    write_details_markdown([_question_result(groundedness=0.0, abstained=True)], out_path, k=1)

    content = out_path.read_text(encoding="utf-8")
    assert "**Abstained:**" in content
    assert "**Retrieved sources (rank order):** (none)" in content


def test_write_details_markdown_shows_none_when_answer_has_no_citations(tmp_path: Path) -> None:
    result = QuestionResult(
        strategy=ChunkingStrategy.FIXED,
        mode=RetrievalMode.DENSE,
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
    monkeypatch.setattr(run_eval_module, "load_out_of_domain", lambda: _FAKE_OUT_OF_DOMAIN)
    monkeypatch.setattr(run_eval_module, "BGEEmbedder", lambda: embedder)
    monkeypatch.setattr(run_eval_module, "CrossEncoderReranker", lambda: reranker)
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

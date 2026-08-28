from __future__ import annotations

import pytest

from common.schemas import AskResponse, ChunkingStrategy, DocChunk, GoldenQuestion
from eval.run_eval import _reciprocal_rank, evaluate_generation, evaluate_retrieval, load_golden_set
from ingestion.store import ChunkStore
from retrieval.retriever import Retriever
from tests.conftest import FakeEmbedder

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
    import eval.run_eval as run_eval_module

    def _stub_generate_answer(question: str, passages: list[object]) -> AskResponse:
        return AskResponse(
            question=question,
            answer="stub",
            citations=[],
            passages=passages,  # type: ignore[arg-type]
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

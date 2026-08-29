"""Evaluate retrieval and generation against eval/golden_set.yaml, for both chunking strategies.

Mirrors `ml_pipeline/eval.py` in the finrisk-agent sibling project: load
artifacts (here, the two Chroma collections built by `ingestion.build`),
score against a held-out, hand-verified ground truth, and write a metrics
report — `eval_report.json` here, `models/metrics.json` there.

Retrieval metrics (per strategy, at a fixed k):
- **precision@k**: of the k passages retrieved, what fraction come from a
  file the golden question actually expects.
- **recall@k**: of the files the golden question expects, what fraction
  appear anywhere in the top-k retrieved passages.
- **MRR** (Mean Reciprocal Rank): 1/rank of the first retrieved passage
  that comes from an expected file, averaged over all questions — rewards
  putting a correct source *early*, which precision/recall at a fixed k
  don't directly capture.

Generation metric: mean groundedness (see `generation.generate`) over the
same question set, computed once per strategy using that strategy's own
retrieved passages — this is what lets the two strategies be compared
head-to-head on equal footing, not just on retrieval but on the answer
quality retrieval enables downstream.

`evaluate_question` is the single source of truth behind every aggregate
number: one question, one retrieval call, one (optional) generation call.
`run_all` calls it once per (question, strategy) pair, then
`retrieval_metrics`/`generation_metrics` reduce those rows into the
aggregate report — the same rows also feed `write_details_markdown`'s
full per-question dump, so a suspicious mean is never more than one file
away from the raw rows that produced it, without a second (and, for
generation, non-deterministic) pass re-calling the LLM.

Run: `uv run python -m eval.run_eval` (needs both collections already
built via `uv run python -m ingestion.build`). Generation scoring calls
the configured LLM (Ollama by default) once per question per strategy —
pass `--skip-generation` to score retrieval only, e.g. in CI, where no LLM
is available.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
import yaml

from common.schemas import (
    ChunkingStrategy,
    Citation,
    GenerationMetrics,
    GoldenQuestion,
    QuestionResult,
    RetrievalMetrics,
)
from generation.generate import generate_answer
from ingestion.embed import BGEEmbedder, Embedder
from retrieval.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.yaml"
REPORT_PATH = Path(__file__).parent / "eval_report.json"
DETAILS_PATH = Path(__file__).parent / "eval_details.md"

DEFAULT_K = 5


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[GoldenQuestion]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [GoldenQuestion.model_validate(item) for item in raw]


def _reciprocal_rank(retrieved_files: list[str], expected_files: set[str]) -> float:
    for rank, source_file in enumerate(retrieved_files, start=1):
        if source_file in expected_files:
            return 1.0 / rank
    return 0.0


def evaluate_question(
    question: GoldenQuestion,
    retriever: Retriever,
    k: int = DEFAULT_K,
    skip_generation: bool = False,
) -> QuestionResult:
    """Score one golden question against one strategy — the row everything else aggregates."""
    passages = retriever.retrieve(question.question, top_k=k)
    retrieved_files = [p.source_file for p in passages]
    expected_files = set(question.expected_sources)

    hits = sum(1 for f in retrieved_files if f in expected_files)
    precision = hits / k if k else 0.0
    recall = (
        len(expected_files & set(retrieved_files)) / len(expected_files) if expected_files else 0.0
    )
    reciprocal_rank = _reciprocal_rank(retrieved_files, expected_files)

    generated_answer: str | None = None
    citations: list[Citation] = []
    groundedness_score: float | None = None
    if not skip_generation:
        response = generate_answer(question.question, passages)
        generated_answer = response.answer
        citations = response.citations
        groundedness_score = response.groundedness_score

    return QuestionResult(
        strategy=retriever.strategy,
        question_id=question.id,
        question=question.question,
        category=question.category,
        expected_answer=question.expected_answer,
        expected_sources=question.expected_sources,
        retrieved_sources=retrieved_files,
        precision=precision,
        recall=recall,
        reciprocal_rank=reciprocal_rank,
        generated_answer=generated_answer,
        citations=citations,
        groundedness_score=groundedness_score,
    )


def retrieval_metrics(
    results: list[QuestionResult], strategy: ChunkingStrategy, k: int
) -> RetrievalMetrics:
    n = len(results)
    return RetrievalMetrics(
        strategy=strategy,
        k=k,
        precision_at_k=sum(r.precision for r in results) / n,
        recall_at_k=sum(r.recall for r in results) / n,
        mrr=sum(r.reciprocal_rank for r in results) / n,
        n_questions=n,
    )


def generation_metrics(
    results: list[QuestionResult], strategy: ChunkingStrategy
) -> GenerationMetrics:
    scores = [r.groundedness_score for r in results if r.groundedness_score is not None]
    n = len(scores)
    return GenerationMetrics(strategy=strategy, mean_groundedness=sum(scores) / n, n_questions=n)


def run_all(
    embedder: Embedder, k: int = DEFAULT_K, skip_generation: bool = False
) -> tuple[dict[str, object], list[QuestionResult]]:
    """Score both strategies. Returns the aggregate report plus every underlying row —
    the latter is what `write_details_markdown` dumps for manual review."""
    questions = load_golden_set()
    strategies: dict[str, object] = {}
    all_results: list[QuestionResult] = []

    for strategy in ChunkingStrategy:
        retriever = Retriever(embedder, strategy)
        results = [
            evaluate_question(q, retriever, k=k, skip_generation=skip_generation) for q in questions
        ]
        all_results.extend(results)

        r_metrics = retrieval_metrics(results, strategy, k)
        entry: dict[str, object] = {"retrieval": r_metrics.model_dump()}
        if not skip_generation:
            entry["generation"] = generation_metrics(results, strategy).model_dump()
        strategies[strategy.value] = entry

        logger.info(
            "strategy=%s precision@%d=%.3f recall@%d=%.3f mrr=%.3f",
            strategy.value,
            k,
            r_metrics.precision_at_k,
            k,
            r_metrics.recall_at_k,
            r_metrics.mrr,
        )

    report = {"k": k, "n_questions": len(questions), "strategies": strategies}
    return report, all_results


def _quote_block(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def write_details_markdown(results: list[QuestionResult], path: Path, k: int) -> None:
    """Dump every (question, strategy) row to a human-readable Markdown transcript.

    One section per strategy, one subsection per question, in golden-set
    order — meant to be read top to bottom, not queried, so a reviewer can
    check each generated answer against its expected answer and retrieved
    sources without re-running anything.
    """
    lines = ["# DocQA-Agent — Evaluation Details", ""]
    lines.append(
        f"Full per-question breakdown behind `eval_report.json`, k={k}. "
        "Regenerate via `uv run python -m eval.run_eval` — not meant to be hand-edited."
    )

    for strategy in ChunkingStrategy:
        strategy_results = [r for r in results if r.strategy is strategy]
        if not strategy_results:
            continue
        lines.append(f"\n## Strategy: {strategy.value}\n")

        for r in strategy_results:
            score_bits = (
                f"precision={r.precision:.2f}, recall={r.recall:.2f}, MRR={r.reciprocal_rank:.2f}"
            )
            if r.groundedness_score is not None:
                score_bits += f", groundedness={r.groundedness_score:.2f}"
            lines.append(f"### {r.question_id} — {r.category} ({score_bits})\n")
            lines.append(f"**Question:** {r.question}\n")
            lines.append(f"**Expected answer (reference):** {r.expected_answer}\n")
            lines.append(f"**Expected sources:** {', '.join(r.expected_sources)}\n")
            lines.append(
                f"**Retrieved sources (rank order):** {', '.join(r.retrieved_sources) or '(none)'}\n"
            )

            if r.generated_answer is not None:
                lines.append("**Generated answer:**\n")
                lines.append(_quote_block(r.generated_answer) + "\n")
                if r.citations:
                    cite_bits = "; ".join(
                        f"[{c.source_file}#{c.section}]"
                        + ("" if c.matched_passage else " — INVALID INDEX")
                        for c in r.citations
                    )
                    lines.append(f"**Citations:** {cite_bits}\n")
                else:
                    lines.append("**Citations:** (none)\n")

            lines.append("---\n")

    path.write_text("\n".join(lines), encoding="utf-8")


@app.command()
def run(k: int = DEFAULT_K, skip_generation: bool = False) -> None:
    embedder = BGEEmbedder()
    report, results = run_all(embedder, k=k, skip_generation=skip_generation)

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", REPORT_PATH)

    write_details_markdown(results, DETAILS_PATH, k=k)
    logger.info("Wrote %s", DETAILS_PATH)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

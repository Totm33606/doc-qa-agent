"""Evaluate every chunking strategy x retrieval mode against the golden and out-of-domain sets.

- Retrieval, at file level: precision@k, recall@k, MRR (1/rank of the first passage from
  an expected file).
- Generation: mean groundedness over answered questions (see `generation.generate`).
- Abstention: false rate on `golden_set.yaml`, correct rate on `out_of_domain.yaml`.
  `--sweep` adds the threshold grid `config.rerank_min_score` is chosen from.

`evaluate_question` produces one row per (question, strategy, mode); both the aggregates
in `eval_report.json` and the transcript in `eval_details.md` are built from those rows.

Run: `uv run python -m eval.run_eval` (needs the collections from `ingestion.build`).
Generation calls the LLM 6 x 38 times; `--skip-generation` scores retrieval and
abstention only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
import yaml

from common.schemas import (
    AbstentionMetrics,
    ChunkingStrategy,
    Citation,
    GenerationMetrics,
    GoldenQuestion,
    OutOfDomainQuestion,
    QuestionResult,
    RetrievalMetrics,
    RetrievalMode,
)
from generation.generate import generate_answer
from ingestion.embed import BGEEmbedder, Embedder
from retrieval.rerank import CrossEncoderReranker, Reranker
from retrieval.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.yaml"
OUT_OF_DOMAIN_PATH = Path(__file__).parent / "out_of_domain.yaml"
REPORT_PATH = Path(__file__).parent / "eval_report.json"
DETAILS_PATH = Path(__file__).parent / "eval_details.md"

DEFAULT_K = 5

# Relevance-floor candidates, dense near 0 where a cross-encoder scores irrelevant passages.
THRESHOLD_GRID = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[GoldenQuestion]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [GoldenQuestion.model_validate(item) for item in raw]


def load_out_of_domain(path: Path = OUT_OF_DOMAIN_PATH) -> list[OutOfDomainQuestion]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [OutOfDomainQuestion.model_validate(item) for item in raw]


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
    """Score one golden question against one retriever — the row everything else aggregates."""
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
        mode=retriever.mode,
        question_id=question.id,
        question=question.question,
        category=question.category,
        expected_answer=question.expected_answer,
        expected_sources=question.expected_sources,
        retrieved_sources=retrieved_files,
        precision=precision,
        recall=recall,
        reciprocal_rank=reciprocal_rank,
        abstained=not passages,  # empty retrieval is the abstention, even with --skip-generation
        generated_answer=generated_answer,
        citations=citations,
        groundedness_score=groundedness_score,
    )


def retrieval_metrics(
    results: list[QuestionResult], strategy: ChunkingStrategy, mode: RetrievalMode, k: int
) -> RetrievalMetrics:
    n = len(results)
    return RetrievalMetrics(
        strategy=strategy,
        mode=mode,
        k=k,
        precision_at_k=sum(r.precision for r in results) / n,
        recall_at_k=sum(r.recall for r in results) / n,
        mrr=sum(r.reciprocal_rank for r in results) / n,
        n_questions=n,
    )


def generation_metrics(
    results: list[QuestionResult], strategy: ChunkingStrategy, mode: RetrievalMode
) -> GenerationMetrics:
    """Mean groundedness over answered questions, plus the abstention rate.

    Abstentions are excluded: a refusal has no citations and would score 0.0, making a
    correct refusal look like a hallucination.
    """
    scores = [
        r.groundedness_score
        for r in results
        if r.groundedness_score is not None and not r.abstained
    ]
    n = len(scores)
    return GenerationMetrics(
        strategy=strategy,
        mode=mode,
        mean_groundedness=sum(scores) / n if n else 0.0,
        n_questions=n,
        abstention_rate=sum(r.abstained for r in results) / len(results) if results else 0.0,
    )


def abstained(retriever: Retriever, question: str, k: int = DEFAULT_K) -> bool:
    """Whether the retriever declined to answer — an empty retrieval is the abstention."""
    return not retriever.retrieve(question, top_k=k)


def abstention_metrics(
    strategy: ChunkingStrategy,
    mode: RetrievalMode,
    threshold: float | None,
    in_domain: list[QuestionResult],
    out_of_domain_abstained: list[bool],
) -> AbstentionMetrics:
    """Reduce both question sets to the false and correct abstention rates.

    Out-of-domain questions arrive as bare `abstained` flags: with no expected sources or
    answer, there is nothing else to score.
    """
    false_abstentions = sum(1 for r in in_domain if r.abstained)
    correct_abstentions = sum(out_of_domain_abstained)
    n_out = len(out_of_domain_abstained)
    return AbstentionMetrics(
        strategy=strategy,
        mode=mode,
        threshold=threshold,
        false_abstention_rate=false_abstentions / len(in_domain) if in_domain else 0.0,
        correct_abstention_rate=correct_abstentions / n_out if n_out else 0.0,
        n_in_domain=len(in_domain),
        n_out_of_domain=n_out,
    )


def _best_rerank_scores(retriever: Retriever, questions: list[str], k: int) -> list[float]:
    """Best re-rank score per question: the only score that decides whether the floor refuses it."""
    scores = []
    for question in questions:
        passages = retriever.retrieve(question, top_k=k)
        scores.append(max((p.rerank_score or 0.0) for p in passages) if passages else 0.0)
    return scores


def sweep_threshold(
    embedder: Embedder, reranker: Reranker, k: int = DEFAULT_K
) -> dict[str, object]:
    """Score every threshold in `THRESHOLD_GRID` on both question sets from one retrieval pass.

    The floor is disabled so every score is visible. If `out_of_domain_max < in_domain_min`,
    any threshold in between separates the two sets perfectly.
    """
    in_domain = load_golden_set()
    out_of_domain = load_out_of_domain()
    sweep: dict[str, object] = {}

    for strategy in ChunkingStrategy:
        retriever = Retriever(
            embedder,
            strategy,
            RetrievalMode.HYBRID_RERANK,
            reranker=reranker,
            min_rerank_score=0.0,
        )
        best_in = _best_rerank_scores(retriever, [q.question for q in in_domain], k)
        best_out = _best_rerank_scores(retriever, [q.question for q in out_of_domain], k)

        sweep[strategy.value] = {
            "n_in_domain": len(best_in),
            "n_out_of_domain": len(best_out),
            "in_domain_min": min(best_in),
            "out_of_domain_max": max(best_out),
            "grid": [
                {
                    "threshold": t,
                    "false_abstention_rate": sum(s < t for s in best_in) / len(best_in),
                    "correct_abstention_rate": sum(s < t for s in best_out) / len(best_out),
                }
                for t in THRESHOLD_GRID
            ],
        }
        logger.info(
            "strategy=%s sweep: lowest in-domain score=%.4f, highest out-of-domain score=%.4f",
            strategy.value,
            min(best_in),
            max(best_out),
        )

    return sweep


def run_all(
    embedder: Embedder,
    reranker: Reranker,
    k: int = DEFAULT_K,
    skip_generation: bool = False,
) -> tuple[dict[str, object], list[QuestionResult]]:
    """Score every strategy x mode; returns the aggregate report and every per-question row.

    Models are passed in so they load once and tests can substitute fakes.
    """
    questions = load_golden_set()
    out_of_domain = load_out_of_domain()
    strategies: dict[str, dict[str, object]] = {}
    all_results: list[QuestionResult] = []

    for strategy in ChunkingStrategy:
        for mode in RetrievalMode:
            retriever = Retriever(embedder, strategy, mode, reranker=reranker)
            results = [
                evaluate_question(q, retriever, k=k, skip_generation=skip_generation)
                for q in questions
            ]
            all_results.extend(results)

            r_metrics = retrieval_metrics(results, strategy, mode, k)
            a_metrics = abstention_metrics(
                strategy,
                mode,
                retriever.min_rerank_score,
                results,
                [abstained(retriever, q.question, k) for q in out_of_domain],
            )
            entry: dict[str, object] = {
                "retrieval": r_metrics.model_dump(),
                "abstention": a_metrics.model_dump(),
            }
            if not skip_generation:
                entry["generation"] = generation_metrics(results, strategy, mode).model_dump()
            strategies.setdefault(strategy.value, {})[mode.value] = entry

            logger.info(
                "strategy=%s mode=%s precision@%d=%.3f recall@%d=%.3f mrr=%.3f "
                "false-abstention=%.3f correct-abstention=%.3f",
                strategy.value,
                mode.value,
                k,
                r_metrics.precision_at_k,
                k,
                r_metrics.recall_at_k,
                r_metrics.mrr,
                a_metrics.false_abstention_rate,
                a_metrics.correct_abstention_rate,
            )

    report = {
        "k": k,
        "n_questions": len(questions),
        "n_out_of_domain": len(out_of_domain),
        "strategies": strategies,
    }
    return report, all_results


def _quote_block(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def write_details_markdown(results: list[QuestionResult], path: Path, k: int) -> None:
    """Dump every (question, strategy, mode) row to a Markdown transcript for manual review."""
    lines = ["# DocQA-Agent — Evaluation Details", ""]
    lines.append(
        f"Full per-question breakdown behind `eval_report.json`, k={k}. "
        "Regenerate via `uv run python -m eval.run_eval` — not meant to be hand-edited."
    )

    for strategy, mode in ((s, m) for s in ChunkingStrategy for m in RetrievalMode):
        section_results = [r for r in results if r.strategy is strategy and r.mode is mode]
        if not section_results:
            continue
        lines.append(f"\n## Strategy: {strategy.value} · mode: {mode.value}\n")

        for r in section_results:
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
            if r.abstained:
                lines.append(
                    "**Abstained:** nothing cleared the relevance floor, so no answer was "
                    "generated.\n"
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
def run(k: int = DEFAULT_K, skip_generation: bool = False, sweep: bool = False) -> None:
    """Score every strategy x mode; `--sweep` adds the relevance-floor grid.

    The sweep costs a second retrieval pass and is only needed to choose the floor.
    """
    embedder = BGEEmbedder()
    reranker = CrossEncoderReranker()
    report, results = run_all(embedder, reranker, k=k, skip_generation=skip_generation)
    if sweep:
        report["threshold_sweep"] = sweep_threshold(embedder, reranker, k=k)

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", REPORT_PATH)

    write_details_markdown(results, DETAILS_PATH, k=k)
    logger.info("Wrote %s", DETAILS_PATH)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

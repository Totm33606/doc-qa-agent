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

from common.schemas import ChunkingStrategy, GenerationMetrics, GoldenQuestion, RetrievalMetrics
from generation.generate import generate_answer
from ingestion.embed import BGEEmbedder, Embedder
from retrieval.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.yaml"
REPORT_PATH = Path(__file__).parent / "eval_report.json"

DEFAULT_K = 5


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[GoldenQuestion]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [GoldenQuestion.model_validate(item) for item in raw]


def _reciprocal_rank(retrieved_files: list[str], expected_files: set[str]) -> float:
    for rank, source_file in enumerate(retrieved_files, start=1):
        if source_file in expected_files:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    questions: list[GoldenQuestion], retriever: Retriever, k: int = DEFAULT_K
) -> RetrievalMetrics:
    precisions: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []

    for question in questions:
        passages = retriever.retrieve(question.question, top_k=k)
        retrieved_files = [p.source_file for p in passages]
        expected_files = set(question.expected_sources)

        hits = sum(1 for f in retrieved_files if f in expected_files)
        precisions.append(hits / k if k else 0.0)
        recalls.append(
            len(expected_files & set(retrieved_files)) / len(expected_files)
            if expected_files
            else 0.0
        )
        reciprocal_ranks.append(_reciprocal_rank(retrieved_files, expected_files))

    n = len(questions)
    return RetrievalMetrics(
        strategy=retriever.strategy,
        k=k,
        precision_at_k=sum(precisions) / n,
        recall_at_k=sum(recalls) / n,
        mrr=sum(reciprocal_ranks) / n,
        n_questions=n,
    )


def evaluate_generation(
    questions: list[GoldenQuestion], retriever: Retriever, k: int = DEFAULT_K
) -> GenerationMetrics:
    scores: list[float] = []
    for question in questions:
        passages = retriever.retrieve(question.question, top_k=k)
        response = generate_answer(question.question, passages)
        scores.append(response.groundedness_score)

    n = len(questions)
    return GenerationMetrics(
        strategy=retriever.strategy,
        mean_groundedness=sum(scores) / n,
        n_questions=n,
    )


def run_all(
    embedder: Embedder, k: int = DEFAULT_K, skip_generation: bool = False
) -> dict[str, object]:
    questions = load_golden_set()
    strategies: dict[str, object] = {}

    for strategy in ChunkingStrategy:
        retriever = Retriever(embedder, strategy)
        retrieval_metrics = evaluate_retrieval(questions, retriever, k=k)
        entry: dict[str, object] = {"retrieval": retrieval_metrics.model_dump()}

        if not skip_generation:
            generation_metrics = evaluate_generation(questions, retriever, k=k)
            entry["generation"] = generation_metrics.model_dump()

        strategies[strategy.value] = entry
        logger.info(
            "strategy=%s precision@%d=%.3f recall@%d=%.3f mrr=%.3f",
            strategy.value,
            k,
            retrieval_metrics.precision_at_k,
            k,
            retrieval_metrics.recall_at_k,
            retrieval_metrics.mrr,
        )

    return {"k": k, "n_questions": len(questions), "strategies": strategies}


@app.command()
def run(k: int = DEFAULT_K, skip_generation: bool = False) -> None:
    embedder = BGEEmbedder()
    report = run_all(embedder, k=k, skip_generation=skip_generation)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", REPORT_PATH)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

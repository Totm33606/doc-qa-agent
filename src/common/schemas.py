"""Shared data models used by ingestion, retrieval, generation, the API and the eval harness."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from common.config import config


class ChunkingStrategy(str, Enum):
    """The two chunking strategies compared by this project — see eval/run_eval.py."""

    FIXED = "fixed"
    MARKDOWN = "markdown"


class RetrievalMode(str, Enum):
    """How a query is matched against the store — see retrieval/retriever.py."""

    DENSE = "dense"  # cosine similarity on BGE embeddings only
    HYBRID = "hybrid"  # dense + BM25, combined by reciprocal rank fusion
    HYBRID_RERANK = "hybrid_rerank"  # hybrid, then re-scored by a cross-encoder


class DocChunk(BaseModel):
    """One chunk produced by ingestion, ready to be embedded and stored."""

    chunk_id: str
    text: str
    source_file: str = Field(
        ..., description="Corpus-relative path, e.g. 'tutorial/path-params.md'"
    )
    section: str = Field(
        ..., description="Header breadcrumb, e.g. 'Path Parameters > Order matters'"
    )
    strategy: ChunkingStrategy
    token_count: int
    chunk_index: int = Field(..., description="Position of this chunk within its source file")


class RetrievedPassage(BaseModel):
    """A chunk returned by the retriever, with its first-stage score and optional re-rank score.

    Only `rerank_score` is comparable across questions, so the relevance floor applies to it.
    """

    chunk_id: str
    text: str
    source_file: str
    section: str
    score: float = Field(
        ...,
        description="First-stage score, higher = better: cosine similarity in dense mode, fused RRF score in both hybrid modes",
    )
    rerank_score: float | None = Field(
        None,
        description="Cross-encoder relevance score in [0, 1] (sigmoid of the logit); passages are ordered by it when set. Null in modes that don't re-rank",
    )


class Citation(BaseModel):
    """A single `[source: N]` reference extracted from a generated answer, resolved to a passage."""

    source_file: str
    section: str
    matched_passage: bool = Field(
        ..., description="Whether N was a valid index into the passages actually retrieved"
    )


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(config.default_top_k, ge=1, le=20)
    strategy: ChunkingStrategy = ChunkingStrategy.MARKDOWN
    mode: RetrievalMode = RetrievalMode.HYBRID


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    passages: list[RetrievedPassage]
    groundedness_score: float = Field(
        ...,
        description="Fraction of the answer's citation-delimited claim segments backed by a valid citation",
    )
    abstained: bool = Field(
        False,
        description="True when no passage was relevant enough to answer from, so `answer` is a canned refusal and no LLM was called",
    )


class GoldenQuestion(BaseModel):
    """One handwritten entry in eval/golden_set.yaml."""

    id: str
    question: str
    expected_answer: str
    expected_sources: list[str] = Field(
        ..., description="Corpus-relative file paths that a correct retrieval should surface"
    )
    category: str


class OutOfDomainQuestion(BaseModel):
    """One entry in eval/out_of_domain.yaml — a question the corpus cannot answer.

    There is no `expected_answer`: the only correct response is a refusal.
    """

    id: str
    question: str
    category: str = Field(
        ..., description="What kind of wrongness this probes, e.g. 'adjacent-framework'"
    )
    why: str = Field(..., description="Why the FastAPI corpus cannot answer it")


class RetrievalMetrics(BaseModel):
    strategy: ChunkingStrategy
    mode: RetrievalMode
    k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    n_questions: int


class GenerationMetrics(BaseModel):
    strategy: ChunkingStrategy
    mode: RetrievalMode
    mean_groundedness: float = Field(
        ...,
        description="Mean groundedness over answered questions only; abstentions are excluded (a refusal has no citations and would score 0.0)",
    )
    n_questions: int = Field(
        ..., description="How many answered questions the mean is over (abstentions excluded)"
    )


class AbstentionMetrics(BaseModel):
    """How well a mode with a relevance floor tells answerable questions from unanswerable ones.

    Only reported for modes that can abstain. The two rates trade off as the floor moves.
    """

    strategy: ChunkingStrategy
    mode: RetrievalMode
    threshold: float = Field(..., description="Relevance floor these rates were measured at")
    false_abstention_rate: float = Field(
        ..., description="Fraction of in-domain golden questions wrongly refused. Lower is better"
    )
    correct_abstention_rate: float = Field(
        ...,
        description="Fraction of out-of-domain questions correctly refused. Higher is better",
    )
    n_in_domain: int
    n_out_of_domain: int


class QuestionResult(BaseModel):
    """One (question, strategy, mode) row — averaged into the metrics, dumped to eval_details.md."""

    strategy: ChunkingStrategy
    mode: RetrievalMode
    question_id: str
    question: str
    category: str
    expected_answer: str
    expected_sources: list[str]
    retrieved_sources: list[str] = Field(
        ..., description="Source files of the top-k retrieved passages, in rank order"
    )
    precision: float
    recall: float
    reciprocal_rank: float
    abstained: bool = Field(
        False,
        description="True when retrieval returned nothing — no passage cleared the relevance floor",
    )
    generated_answer: str | None = Field(
        None, description="None when generation was skipped for this run"
    )
    citations: list[Citation] = Field(default_factory=list)
    groundedness_score: float | None = None

"""Shared data models — the vocabulary used across ingestion, retrieval, generation and the API.

Kept in one module (mirroring `common.schemas` in the finrisk-agent sibling
project) so every stage of the pipeline agrees on the exact shape of a
"chunk", a "passage" and a "citation" instead of each script inventing its
own dict layout.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


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
    """A chunk returned by the retriever for a given query, with its relevance score(s).

    Two scores, because they come from different stages and mean different
    things. `score` is the first-stage score, and what it measures depends on
    the mode. `rerank_score` is the cross-encoder's, and is the only one that
    is comparable across questions — which is why it, and not `score`, is what
    the relevance floor in `retrieval/retriever.py` is applied to.
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
        description="Cross-encoder relevance score, squashed to [0, 1] by a sigmoid so it is comparable across questions — confident, not calibrated. None in modes that don't re-rank. When set, this is what the passages are ordered by",
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
    top_k: int = Field(5, ge=1, le=20)
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

    The golden set measures whether the right passages come back. This set
    measures the opposite obligation: recognizing that *no* passage is right,
    so the agent refuses instead of answering from whatever ranked highest.
    There is no `expected_answer` because the only correct answer is a refusal.
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
        description="Mean groundedness over the *answered* questions only — an abstention has no citations and would score 0.0, which would make a correct refusal indistinguishable from a hallucination",
    )
    n_questions: int = Field(..., description="How many questions the mean above is over")
    abstention_rate: float = Field(
        0.0,
        description="Fraction of golden questions the retriever refused, i.e. the rows excluded from mean_groundedness",
    )


class AbstentionMetrics(BaseModel):
    """How well one (strategy, mode) tells answerable questions from unanswerable ones.

    Two error rates, deliberately reported side by side rather than folded
    into one number: they trade off against each other as the relevance floor
    moves, and which trade is acceptable is a judgement call the report should
    show rather than make.
    """

    strategy: ChunkingStrategy
    mode: RetrievalMode
    threshold: float | None = Field(
        ...,
        description="The relevance floor these rates were measured at, or null for a mode that has no floor and so can never abstain",
    )
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
    """The full detail behind one (question, strategy, mode) data point in the aggregate metrics.

    `RetrievalMetrics`/`GenerationMetrics` are averages over exactly these
    rows — one `QuestionResult` per golden question per (strategy, mode)
    pair is what `eval/run_eval.py::write_details_markdown` dumps for manual
    review, so a suspicious aggregate number can always be traced back to
    the specific question(s) behind it instead of taken on faith.
    """

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

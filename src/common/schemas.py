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
    """A chunk returned by the retriever for a given query, with its similarity score."""

    chunk_id: str
    text: str
    source_file: str
    section: str
    score: float = Field(
        ..., description="Cosine similarity to the query embedding, higher = closer"
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


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    passages: list[RetrievedPassage]
    groundedness_score: float = Field(
        ...,
        description="Fraction of the answer's citation-delimited claim segments backed by a valid citation",
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


class RetrievalMetrics(BaseModel):
    strategy: ChunkingStrategy
    k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    n_questions: int


class GenerationMetrics(BaseModel):
    strategy: ChunkingStrategy
    mean_groundedness: float
    n_questions: int

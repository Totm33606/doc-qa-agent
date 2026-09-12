"""Cross-encoder re-ranking of the fused candidates — the second stage of retrieval.

**What this adds that fusion can't do.** Reciprocal rank fusion combines two
*orderings*; it never looks at the question and a passage together. It knows
that BM25 put a chunk second and the dense search put it fourth, and nothing
else. So it can tell you what the best of the candidates is, but not whether
that best is any good — a query with no answer anywhere in the corpus still
produces a confidently-ordered top 5.

A cross-encoder does the thing bi-encoders structurally can't: it reads the
question and the passage *in the same forward pass*, with full attention
between them. That's far more accurate than comparing two independently
computed vectors, and far too slow to run over a whole corpus — which is
exactly why it belongs here, re-scoring ~20 candidates that cheap retrieval
already narrowed down, rather than replacing either retriever.

The score it returns is bounded to [0, 1] and **comparable across
questions** — unlike a cosine similarity (only semi-calibrated) and
unlike an RRF score (a function of rank positions, with no absolute meaning
at all: a passage ranked first by both retrievers scores 2/61 whether it
answers the question or is merely the least-bad of a bad corpus). That
comparability is what makes the relevance floor in `retrieval/retriever.py`
possible; see that module and the README for the abstention behaviour built
on it.

`Reranker` is a `Protocol` for the same reason `ingestion.embed.Embedder`
is: tests inject a cheap deterministic fake rather than downloading the
real model.
"""

from __future__ import annotations

import math
from typing import Protocol

from common.config import config
from common.schemas import RetrievedPassage


def _sigmoid(x: float) -> float:
    """Logit -> [0, 1], written out to stay finite on large negative inputs.

    The naive `1 / (1 + exp(-x))` overflows for the very negative logits this
    model produces for irrelevant passages — which is precisely the range the
    relevance floor cares about, so it can't be the range that raises.
    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class Reranker(Protocol):
    def score(self, question: str, passages: list[RetrievedPassage]) -> list[float]: ...


class CrossEncoderReranker:
    """`cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`, CPU, no API key.

    ~22M parameters (~90MB) — small enough to re-score 20 candidates in
    well under a second on CPU, and it needs no dependency this project
    didn't already have: `sentence-transformers` is already here for the
    BGE embedder.

    **The sigmoid is applied here, deliberately.** The checkpoint is a
    regression model whose raw output is an unbounded logit — measured on
    this corpus, -11.3 to +8.6. Left as logits the scores still *rank*
    correctly, so re-ranking alone would work, but the relevance floor built
    on top of them would not: a threshold would have no natural zero, no
    bounded range, and no meaning the README could state in one sentence.
    `sentence-transformers` applies no activation by default for this model,
    so `_sigmoid` does it explicitly rather than depending on a library
    default that has changed across versions. The resulting [0, 1] range is
    asserted against the real model in `tests/test_integration.py`.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name or config.reranker_model_name)

    def score(self, question: str, passages: list[RetrievedPassage]) -> list[float]:
        if not passages:
            return []
        pairs = [(question, p.text) for p in passages]
        logits = self._model.predict(pairs, show_progress_bar=False)
        return [_sigmoid(float(logit)) for logit in logits]

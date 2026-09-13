"""Cross-encoder re-ranking of the fused candidates, the second stage of `hybrid_rerank`.

RRF only orders candidates; it can't tell whether the best one is any good. A
cross-encoder reads question and passage together, and its sigmoid-squashed score is
comparable across questions, which is what the relevance floor in `retriever.py` needs.
`Reranker` is a Protocol so tests can inject a fake.
"""

from __future__ import annotations

import logging
import math
from typing import Protocol

from common.config import config
from common.schemas import RetrievedPassage

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Logit -> [0, 1], written to avoid `math.exp` overflow on large negative inputs."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class Reranker(Protocol):
    def score(self, question: str, passages: list[RetrievedPassage]) -> list[float]: ...


class CrossEncoderReranker:
    """`cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` (~22M params, CPU).

    The checkpoint declares an identity output activation, so `predict` returns raw logits
    (-11.4 to +8.6 on this corpus); the sigmoid is applied here explicitly.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name or config.reranker_model_name)

    def score(self, question: str, passages: list[RetrievedPassage]) -> list[float]:
        if not passages:
            return []
        self._warn_if_truncated(question, passages)
        pairs = [(question, p.text) for p in passages]
        logits = self._model.predict(pairs, show_progress_bar=False)
        return [_sigmoid(float(logit)) for logit in logits]

    def _warn_if_truncated(self, question: str, passages: list[RetrievedPassage]) -> None:
        """Log (question, passage) pairs longer than the model's window, which it truncates silently.

        Chunks are sized to fit alongside every eval question, but a question sent to `/ask`
        has no length limit; the model then trims the longer side, usually the passage.
        """
        tokenizer = self._model.tokenizer
        budget = self._model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=True)
        question_tokens = len(tokenizer.tokenize(question))
        n_over = sum(question_tokens + len(tokenizer.tokenize(p.text)) > budget for p in passages)
        if n_over:
            logger.warning(
                "Question (%d tokens) plus passage exceeds the re-ranker's %d-token window for "
                "%d of %d passages; they are truncated before scoring.",
                question_tokens,
                self._model.max_seq_length,
                n_over,
                len(passages),
            )

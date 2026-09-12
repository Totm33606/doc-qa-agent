"""Unit tests for the logit -> probability conversion the relevance floor is built on.

The re-ranker's *ordering* would work on raw logits. The floor would not: it
needs a bounded scale with a fixed meaning, which is what `_sigmoid` provides
and what these tests pin down. The real model's behaviour is checked
separately in `tests/test_integration.py`.
"""

from __future__ import annotations

import pytest

from retrieval.rerank import _sigmoid


@pytest.mark.parametrize("logit", [-1000.0, -11.0, -1.0, 0.0, 1.0, 5.0, 1000.0])
def test_sigmoid_always_lands_in_the_unit_interval(logit: float) -> None:
    assert 0.0 <= _sigmoid(logit) <= 1.0


def test_sigmoid_is_centred_on_zero() -> None:
    assert _sigmoid(0.0) == 0.5


def test_sigmoid_is_monotonic() -> None:
    """Order-preserving, so adding it can't change which passage ranks first."""
    logits = [-11.0, -5.0, -1.0, 0.0, 2.0, 5.0]
    scores = [_sigmoid(x) for x in logits]
    assert scores == sorted(scores)


def test_sigmoid_survives_large_negative_logits() -> None:
    """The naive formula overflows here — and this is exactly the range the floor
    cares about, since it's what the model returns for irrelevant passages."""
    assert _sigmoid(-800.0) == pytest.approx(0.0)
    assert _sigmoid(800.0) == pytest.approx(1.0)

"""Generate an answer from retrieved passages, then score how grounded it is.

Accepted citation markers: `[source: N]` (what the prompt asks for), bare `[N]`, and
multi-index `[source: 2,5]`, grounded only if every index is valid. A marker only
counts when it trails claim text. This is a syntactic proxy: it catches citations to
passages never shown, not claims the cited passage doesn't support.

With no passages (nothing cleared the relevance floor), `generate_answer` returns a
fixed refusal without calling the LLM.
"""

from __future__ import annotations

import re
from typing import Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from common.schemas import AskResponse, Citation, RetrievedPassage
from generation.llm import build_llm
from generation.prompt import SYSTEM_PROMPT, build_user_message

_CITATION_RE = re.compile(r"\[(?:source:\s*)?(\d+(?:\s*,\s*\d+)*)\]")
_HAS_WORD_RE = re.compile(r"\w")


def _indices(match: re.Match[str]) -> list[int]:
    """`[source: 2,5]` -> [2, 5]."""
    return [int(n) for n in match.group(1).split(",")]


ABSTENTION_ANSWER = (
    "I couldn't find anything in the FastAPI documentation relevant to this question, "
    "so I won't try to answer it."
)


class ChatModel(Protocol):
    """Structural type satisfied by both `ChatOpenAI`/`AzureChatOpenAI` and test fakes."""

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage: ...


def extract_citations(answer: str, passages: list[RetrievedPassage]) -> list[Citation]:
    """Resolve every citation index in `answer` to a passage, one `Citation` per index.

    `matched_passage` is False for an index outside the passages shown to the model.
    """
    citations = []
    for match in _CITATION_RE.finditer(answer):
        for index in _indices(match):
            in_range = 1 <= index <= len(passages)
            passage = passages[index - 1] if in_range else None
            citations.append(
                Citation(
                    source_file=passage.source_file if passage else f"<invalid index {index}>",
                    section=passage.section if passage else "<invalid index>",
                    matched_passage=in_range,
                )
            )
    return citations


def compute_groundedness(answer: str, passages: list[RetrievedPassage]) -> float:
    """Fraction of claim segments that are grounded.

    Each marker closes the segment before it; text after the last marker is one more,
    uncited, segment. Segments without a word character (the "." in "claim [3].") are
    not claims.
    """
    text = answer.strip()
    if not text:
        return 0.0

    n_passages = len(passages)
    grounded_flags: list[bool] = []
    cursor = 0

    for match in _CITATION_RE.finditer(text):
        segment = text[cursor : match.start()].strip()
        if _HAS_WORD_RE.search(segment):
            grounded_flags.append(all(1 <= i <= n_passages for i in _indices(match)))
        cursor = match.end()

    trailing = text[cursor:].strip()
    if _HAS_WORD_RE.search(trailing):
        grounded_flags.append(False)

    if not grounded_flags:
        return 0.0
    return sum(grounded_flags) / len(grounded_flags)


def generate_answer(
    question: str, passages: list[RetrievedPassage], llm: ChatModel | None = None
) -> AskResponse:
    """Answer from the passages, or return a fixed refusal without an LLM call if there are none."""
    if not passages:
        return AskResponse(
            question=question,
            answer=ABSTENTION_ANSWER,
            citations=[],
            passages=[],
            groundedness_score=0.0,
            abstained=True,
        )

    model = llm or build_llm()
    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_message(question, passages)),
    ]
    response = model.invoke(messages)
    answer_text = response.content if isinstance(response.content, str) else str(response.content)

    citations = extract_citations(answer_text, passages)
    groundedness = compute_groundedness(answer_text, passages)

    return AskResponse(
        question=question,
        answer=answer_text,
        citations=citations,
        passages=passages,
        groundedness_score=groundedness,
    )

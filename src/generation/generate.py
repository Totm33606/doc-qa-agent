"""Generate a grounded answer from retrieved passages, then score how grounded it actually is.

`compute_groundedness` is the hallucination check: it doesn't ask a second
LLM to judge the answer (slower, another source of error) — it splits the
answer at each `[source: N]` marker and checks whether the claim text
immediately before that marker is backed by a valid, in-range citation.

An earlier version split the answer into grammatical sentences first and
then looked for a citation *inside* each sentence. That systematically
undercounted: a citation conventionally trails right after the sentence's
closing period ("Claim. [source: 1]"), which puts it in the *next*
sentence-split's text, not the one it actually supports — so a
fully-and-correctly-cited two-sentence answer scored 0.67, not 1.0. It
also broke down entirely on multi-paragraph answers ending in a code
block, since a citation after a fenced code block has no sentence-ending
punctuation to attach to at all. Splitting on the citation markers
themselves sidesteps both problems: whatever text sits between two
markers (or between the last marker and the end of the answer) is treated
as one claim segment, and that segment counts as grounded only if the
marker following it is a valid index. This is a syntactic proxy for "is
this traceable to a source", not a semantic entailment check — see the
README's evaluation section for what it does and doesn't catch.
"""

from __future__ import annotations

import re
from typing import Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from common.schemas import AskResponse, Citation, RetrievedPassage
from generation.llm import build_llm
from generation.prompt import SYSTEM_PROMPT, build_user_message

_CITATION_RE = re.compile(r"\[source:\s*(\d+)\]")

_NO_PASSAGES_ANSWER = (
    "I don't have any retrieved documentation passages to answer this question from."
)


class ChatModel(Protocol):
    """Structural type satisfied by both `ChatOpenAI`/`AzureChatOpenAI` and test fakes."""

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage: ...


def extract_citations(answer: str, passages: list[RetrievedPassage]) -> list[Citation]:
    """Pull every `[source: N]` marker out of `answer` and resolve N to a real passage.

    `matched_passage` is False when N is out of range (0, too large, or
    otherwise not one of the passages actually shown to the model) — a
    citation to a passage number that was never shown is exactly the
    hallucination case this is meant to catch.
    """
    citations = []
    for match in _CITATION_RE.finditer(answer):
        index = int(match.group(1))
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
    """Fraction of the answer's claim segments (text between citation markers) that are grounded.

    Each `[source: N]` marker closes out the claim segment before it; any
    leftover text after the last marker is one more, uncited, segment. A
    segment counts as grounded only if its marker's index actually falls
    within `passages` — an out-of-range index means the model cited a
    passage it was never shown, which is exactly the hallucination case
    this is meant to catch.
    """
    text = answer.strip()
    if not text:
        return 0.0

    n_passages = len(passages)
    grounded_flags: list[bool] = []
    cursor = 0
    for match in _CITATION_RE.finditer(text):
        segment = text[cursor : match.start()].strip()
        if segment:
            index = int(match.group(1))
            grounded_flags.append(1 <= index <= n_passages)
        cursor = match.end()

    trailing = text[cursor:].strip()
    if trailing:
        grounded_flags.append(False)

    if not grounded_flags:
        return 0.0
    return sum(grounded_flags) / len(grounded_flags)


def generate_answer(
    question: str, passages: list[RetrievedPassage], llm: ChatModel | None = None
) -> AskResponse:
    if not passages:
        return AskResponse(
            question=question,
            answer=_NO_PASSAGES_ANSWER,
            citations=[],
            passages=[],
            groundedness_score=0.0,
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

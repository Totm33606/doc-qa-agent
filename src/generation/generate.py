"""Generate a grounded answer from retrieved passages, then score how grounded it actually is.

`compute_groundedness` splits the answer at each citation marker and
checks whether the claim right before each marker cites valid, in-range
passages. A citation only counts if it trails the claim it supports — one
with nothing real before it (leading a claim instead of following one)
counts for nothing, matching what the prompt asks for. Three format
liberties are tolerated, all found by reading real generated output: bare
`[N]` as well as `[source: N]` (the model sometimes echoes the `[N]
file#section` notation shown in its own context instead of the requested
form), and one marker citing several passages at once, `[source: 2,5]`
(a segment behind one of these only counts as grounded if *every* index
in it is valid — one real citation bundled with one fabricated one is
still citing a passage that was never shown).

This is a syntactic proxy for "is this traceable to a source", not a
semantic entailment check — it can't verify the cited passage actually
*says* what the claim asserts. See the README's Evaluation section for
what this heuristic does and doesn't catch, and for the design history
(why leading citations, and sentence-based splitting, were tried and
dropped).

**Abstention.** `generate_answer` never calls the LLM with an empty passage
list. An empty list is not an error here, it's a decision made upstream:
`retrieval.retriever`'s relevance floor drops candidates a cross-encoder
judged irrelevant, so retrieving nothing means nothing in the corpus was
relevant enough to answer from. The response is then a fixed refusal with
`abstained=True`, which makes the prompt's "say so if the passages don't
contain enough information" rule enforced rather than merely requested —
a model that ignores it never gets the chance, because it is never asked.
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
    """Parse a citation marker's digits into indices — usually one, e.g. `2, 5` for `[source: 2,5]`."""
    return [int(n) for n in match.group(1).split(",")]


ABSTENTION_ANSWER = (
    "I couldn't find anything in the FastAPI documentation relevant to this question, "
    "so I won't try to answer it."
)


class ChatModel(Protocol):
    """Structural type satisfied by both `ChatOpenAI`/`AzureChatOpenAI` and test fakes."""

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage: ...


def extract_citations(answer: str, passages: list[RetrievedPassage]) -> list[Citation]:
    """Pull every citation marker out of `answer` and resolve each index to a real passage.

    A single marker citing several passages at once (`[source: 2,5]`)
    yields one `Citation` per index. `matched_passage` is False when the
    index is out of range (0, too large, or otherwise not one of the
    passages actually shown to the model) — a citation to a passage
    number that was never shown is exactly the hallucination case this is
    meant to catch.
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
    """Fraction of the answer's claim segments (text between citation markers) that are grounded.

    Each citation marker closes out the claim segment before it; leftover
    text after the last marker is one more, uncited, segment. A segment
    only counts as a real claim if it has at least one word character
    (`_HAS_WORD_RE`) — otherwise a citation styled "claim [3]." (period
    *after* the bracket) would leave a lone "." that isn't blank but also
    isn't a claim, and get counted as an extra uncited segment.
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
    """Answer from the retrieved passages, or abstain when there are none.

    Two paths, and the short one is not an error case. No passages means
    retrieval found nothing relevant enough to show (see the relevance floor
    in `retrieval.retriever`), so the honest response is a refusal — returned
    as a fixed string, with no LLM call, because there is nothing for a model
    to be grounded in.
    """
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

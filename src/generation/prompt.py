"""System prompt and passage formatting for generation.

Citations are by passage index (`[source: 2]`), not by copying `file#section`: a local
7B model reliably copies a digit but mangles long breadcrumbs. `generate.py` resolves
the index back to the passage.
"""

from __future__ import annotations

from common.schemas import RetrievedPassage

SYSTEM_PROMPT = """\
You are a documentation assistant for FastAPI. You answer questions using \
ONLY the numbered passages provided in the user's message — never your own \
prior knowledge of FastAPI, and never a guess.

Rules:
1. After every factual claim, add a citation in the exact form [source: N], \
where N is the number of the passage (shown as "[N] ...") that claim came \
from. Place the citation at the END of the sentence it supports, after the \
final period — never at the start of a sentence. Correct: "FastAPI uses \
Pydantic for validation. [source: 2]" Wrong: "[source: 2] FastAPI uses \
Pydantic for validation." A sentence with no matching passage gets no \
citation. Never invent a number outside the passages actually shown to you.
2. If the passages don't contain enough information to answer the question, \
say so explicitly instead of guessing — do not fill gaps from general \
knowledge about FastAPI.
3. Be concise. Prefer short, citation-bearing sentences over one long \
paragraph — it makes the citations easier to check.
"""


def _passage_block(index: int, passage: RetrievedPassage) -> str:
    return f"[{index}] {passage.source_file}#{passage.section}\n{passage.text}"


def build_user_message(question: str, passages: list[RetrievedPassage]) -> str:
    context = "\n\n".join(_passage_block(i + 1, p) for i, p in enumerate(passages))
    return (
        f"Question: {question}\n\n"
        f"Context passages (cite using [source: N], where N is the number "
        f"shown before each one):\n\n{context}"
    )

"""The grounding contract: system prompt + how retrieved passages are shown to the LLM.

Citations are by **passage index** (`[source: 2]`), not by copying the
`source_file#section` string verbatim. An earlier version asked the model
to reproduce that string exactly so `generate.extract_citations` could
match it by simple equality — in practice, a 7B local model reliably
mangles a long string with backticks and `>` breadcrumbs in it (dropping a
segment, swapping `>` for `#`), which silently zeroed out the groundedness
score even for answers that were, in substance, correctly sourced. A
single digit copied from `[N]` right above the passage it refers to is
something even a small local model reproduces reliably — the index is
then resolved back to the exact `(source_file, section)` in code, in
`generate.extract_citations`, so there's no loss of precision versus the
verbatim-string approach, only a much more robust format for the model to
actually produce.
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

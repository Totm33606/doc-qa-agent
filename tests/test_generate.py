from __future__ import annotations

import pytest

from common.schemas import RetrievedPassage
from generation.generate import compute_groundedness, extract_citations, generate_answer
from tests.conftest import FakeChatModel


def _passage(source_file: str, section: str, score: float = 0.9) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=f"{source_file}:{section}",
        text=f"Some passage text about {section}.",
        source_file=source_file,
        section=section,
        score=score,
    )


PASSAGES = [
    _passage("tutorial/path-params.md", "Order matters"),
    _passage("tutorial/query-params.md", "Defaults"),
]


def test_extract_citations_matches_valid_index() -> None:
    answer = "Declare a fixed path before a variable one. [source: 1]"
    citations = extract_citations(answer, PASSAGES)
    assert len(citations) == 1
    assert citations[0].source_file == "tutorial/path-params.md"
    assert citations[0].section == "Order matters"
    assert citations[0].matched_passage is True


def test_extract_citations_flags_out_of_range_index() -> None:
    answer = "This claim cites a passage that was never shown. [source: 99]"
    citations = extract_citations(answer, PASSAGES)
    assert len(citations) == 1
    assert citations[0].matched_passage is False


def test_extract_citations_flags_index_zero() -> None:
    answer = "Zero is not a valid 1-indexed passage. [source: 0]"
    citations = extract_citations(answer, PASSAGES)
    assert citations[0].matched_passage is False


def test_extract_citations_returns_empty_when_no_markers() -> None:
    assert extract_citations("No citations here at all.", PASSAGES) == []


def test_extract_citations_accepts_bare_index_form() -> None:
    """qwen2.5:7b-instruct sometimes echoes the "[N]" index notation shown in the passage
    context instead of the requested "[source: N]" — same meaning, different spelling."""
    answer = "Declare a fixed path before a variable one [1]."
    citations = extract_citations(answer, PASSAGES)
    assert len(citations) == 1
    assert citations[0].source_file == "tutorial/path-params.md"
    assert citations[0].matched_passage is True


def test_extract_citations_splits_a_multi_index_marker_into_separate_citations() -> None:
    """ "[source: 2,5]" bundles two references in one marker — one Citation per index."""
    answer = "Claim backed by two passages. [source: 1,2]"
    citations = extract_citations(answer, PASSAGES)
    assert [c.source_file for c in citations] == [
        "tutorial/path-params.md",
        "tutorial/query-params.md",
    ]
    assert all(c.matched_passage for c in citations)


def test_extract_citations_multi_index_marker_flags_the_invalid_one_only() -> None:
    answer = "Claim with one fabricated source. [source: 1,99]"
    citations = extract_citations(answer, PASSAGES)
    assert citations[0].matched_passage is True
    assert citations[1].matched_passage is False


def test_extract_citations_handles_multiple_markers() -> None:
    answer = "First claim. [source: 1] Second claim. [source: 2]"
    citations = extract_citations(answer, PASSAGES)
    assert [c.source_file for c in citations] == [
        "tutorial/path-params.md",
        "tutorial/query-params.md",
    ]


def test_compute_groundedness_all_sentences_cited() -> None:
    answer = "First claim. [source: 1] Second claim. [source: 2]"
    assert compute_groundedness(answer, PASSAGES) == 1.0


def test_compute_groundedness_partial_citation() -> None:
    answer = "Cited claim. [source: 1] Uncited claim with no marker at all."
    assert compute_groundedness(answer, PASSAGES) == 0.5


def test_compute_groundedness_out_of_range_citation_does_not_count() -> None:
    answer = "Claim with a fabricated source. [source: 99]"
    assert compute_groundedness(answer, PASSAGES) == 0.0


def test_compute_groundedness_empty_answer_is_zero() -> None:
    assert compute_groundedness("   ", PASSAGES) == 0.0


def test_compute_groundedness_punctuation_only_answer_is_zero() -> None:
    """Non-empty but wordless (no citations, no real claim text either) — distinct code
    path from the empty-string case: `grounded_flags` ends up empty after processing."""
    assert compute_groundedness("...!!!", PASSAGES) == 0.0


def test_compute_groundedness_trailing_period_after_last_citation_is_not_a_claim() -> None:
    """Regression: '[source: N].' (period *after* the bracket) must not add a phantom
    uncited segment — a lone '.' left over after the last marker isn't a real claim.
    """
    answer = "First claim [source: 1]. Second claim [source: 2]."
    assert compute_groundedness(answer, PASSAGES) == 1.0


def test_compute_groundedness_punctuation_only_segment_is_not_a_claim() -> None:
    """A citation with no real claim text before it (just leftover punctuation from the
    previous marker) shouldn't inflate the denominator either."""
    answer = "Real claim [source: 1]. [source: 2]"
    assert compute_groundedness(answer, PASSAGES) == 1.0


def test_compute_groundedness_accepts_bare_index_citations() -> None:
    answer = "First claim [1]. Second claim [2]."
    assert compute_groundedness(answer, PASSAGES) == 1.0


def test_compute_groundedness_multi_index_marker_grounded_when_all_valid() -> None:
    answer = "Claim backed by two passages. [source: 1,2]"
    assert compute_groundedness(answer, PASSAGES) == 1.0


def test_compute_groundedness_multi_index_marker_ungrounded_if_any_index_invalid() -> None:
    """One real citation bundled with one fabricated one still cites a passage that was
    never shown — no partial credit, matching the all-or-nothing rule everywhere else."""
    answer = "Claim with one fabricated source. [source: 1,99]"
    assert compute_groundedness(answer, PASSAGES) == 0.0


def test_compute_groundedness_leading_citation_counts_for_nothing() -> None:
    """A citation with nothing real before it ('[1] Claim.') doesn't grade anything — not
    the (empty) text before it, and not the claim that follows either. It's simply not a
    trailing citation for anyone. A deliberate design choice, not an oversight: see the
    module docstring for why leniency here was tried and then removed."""
    answer = "[source: 1] The only claim in this answer, cited up front."
    assert compute_groundedness(answer, PASSAGES) == 0.0


def test_compute_groundedness_leading_citation_does_not_rescue_the_next_claim() -> None:
    """ "[1] Claim A. [2] Claim B." — [1] leads with nothing before it, so it counts for
    nothing; "Claim A." is graded normally by the trailing [2] (valid); "Claim B" — the
    tail, with no marker after it — is genuinely uncited. 1 of 2 real claims grounded."""
    answer = "[source: 1] Claim A. [source: 2] Claim B."
    assert compute_groundedness(answer, PASSAGES) == 0.5


def test_compute_groundedness_leading_citation_real_world_example() -> None:
    """A real qwen2.5:7b-instruct answer (eval/eval_details.md, q06/fixed) that cites
    every claim *before* it instead of after. Each leading marker counts for nothing; each
    non-empty segment is graded by whichever marker trails it: "A FastAPI dependency is a
    function..." is graded by [2] (valid), "It can return values..." by [4] (valid),
    "Dependencies can have sub-dependencies..." by [5] (valid) — 3 grounded claims. The
    final "Therefore, ..." restatement has no marker after it at all, so it's the 4th,
    uncited, segment: 3/4 = 0.75."""
    answer = (
        "[1] A FastAPI dependency is a function that can take the same parameters as a "
        "path operation function. [2] It can return values or not, and can declare "
        "request requirements or other sub-dependencies. [4] Dependencies can have "
        "sub-dependencies, and FastAPI will take care of solving them. [5] What makes a "
        'dependency is that it should be a "callable". Therefore, the minimum requirement '
        "for something to be usable as a FastAPI dependency is that it must be a "
        '"callable".'
    )
    five_passages = [
        *PASSAGES,
        _passage("tutorial/dependencies/index.md", "root"),
        _passage("tutorial/dependencies/sub-dependencies.md", "root"),
        _passage("tutorial/dependencies/classes-as-dependencies.md", "root"),
    ]
    assert compute_groundedness(answer, five_passages) == pytest.approx(0.75)


def test_generate_answer_falls_back_to_build_llm_when_none_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers the `llm or build_llm()` branch — every other test injects an explicit llm."""
    fake = FakeChatModel("An answer. [source: 1]")
    monkeypatch.setattr("generation.generate.build_llm", lambda: fake)

    response = generate_answer("A question", PASSAGES, llm=None)

    assert len(fake.invocations) == 1
    assert response.groundedness_score == 1.0


def test_generate_answer_with_no_passages_short_circuits() -> None:
    llm = FakeChatModel("this should never be used")
    response = generate_answer("What is FastAPI?", [], llm=llm)
    assert response.passages == []
    assert response.citations == []
    assert response.groundedness_score == 0.0
    assert llm.invocations == []


def test_generate_answer_builds_response_from_llm_output() -> None:
    llm = FakeChatModel(
        "Declare fixed paths before variable ones. [source: 1] Query params can have defaults. [source: 2]"
    )
    response = generate_answer("How does path ordering work?", PASSAGES, llm=llm)

    assert response.passages == PASSAGES
    assert len(response.citations) == 2
    assert all(c.matched_passage for c in response.citations)
    assert response.groundedness_score == 1.0
    assert len(llm.invocations) == 1


def test_generate_answer_passes_system_and_human_messages() -> None:
    llm = FakeChatModel("An answer. [source: 1]")
    generate_answer("A question", PASSAGES, llm=llm)

    [messages] = llm.invocations
    assert len(messages) == 2
    content = messages[1].content
    assert isinstance(content, str)
    assert content.startswith("Question: A question")

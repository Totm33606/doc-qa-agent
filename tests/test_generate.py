from __future__ import annotations

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
    assert messages[1].content.startswith("Question: A question")

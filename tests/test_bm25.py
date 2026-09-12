from __future__ import annotations

from common.schemas import ChunkingStrategy, DocChunk
from retrieval.bm25 import BM25Index, tokenize


def _chunk(chunk_id: str, text: str) -> DocChunk:
    return DocChunk(
        chunk_id=chunk_id,
        text=text,
        source_file="tutorial/response-model.md",
        section="Response Model",
        strategy=ChunkingStrategy.MARKDOWN,
        token_count=len(text.split()),
        chunk_index=0,
    )


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Path Parameters, in FastAPI!") == ["path", "parameters", "in", "fastapi"]


def test_tokenize_keeps_python_identifiers_whole() -> None:
    """The exact case BM25 is here for — an identifier must not be split at its underscores."""
    assert tokenize("use `response_model_exclude_unset=True`") == [
        "use",
        "response_model_exclude_unset",
        "true",
    ]


def _filler(n: int) -> list[DocChunk]:
    """Unrelated filler chunks, so Okapi IDF can discriminate.

    IDF is `log(N - df + 0.5) - log(df + 0.5)`: 0 for a term in half the corpus, so a
    two-document fixture scores everything 0.
    """
    return [
        _chunk(f"filler{i}", f"unrelated paragraph {i} about serving requests") for i in range(n)
    ]


def test_search_ranks_the_chunk_containing_the_rare_term_first() -> None:
    index = BM25Index(
        [
            *_filler(8),
            _chunk("exact", "set response_model_exclude_unset to omit default values"),
        ]
    )
    assert index.search("response_model_exclude_unset", top_k=5)[0] == "exact"


def test_search_respects_top_k() -> None:
    index = BM25Index([_chunk(f"c{i}", f"chunk {i} about routing") for i in range(10)])
    assert len(index.search("routing", top_k=3)) == 3


def test_search_drops_chunks_sharing_no_term_with_the_query() -> None:
    """A zero BM25 score means no term overlap at all — it must not enter the fusion."""
    index = BM25Index(
        [*_filler(8), _chunk("match", "backgroundtasks runs work after the response")]
    )
    assert index.search("backgroundtasks", top_k=5) == ["match"]


def test_search_with_no_lexical_match_at_all_returns_empty() -> None:
    index = BM25Index([*_filler(8), _chunk("c1", "path parameters are declared in the path")])
    assert index.search("qqq zzz wwww", top_k=5) == []


def test_empty_corpus_builds_and_searches_without_raising() -> None:
    """`BM25Okapi([])` divides by the corpus size — the index must not build one."""
    index = BM25Index([])
    assert index.search("anything", top_k=5) == []

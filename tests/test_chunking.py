from __future__ import annotations

from pathlib import Path

import pytest

from common.config import config
from common.schemas import ChunkingStrategy
from eval.run_eval import load_golden_set, load_out_of_domain
from ingestion.chunking import (
    _tokenizer,
    chunk_corpus,
    chunk_fixed,
    chunk_markdown_aware,
    count_tokens,
)

SAMPLE_MARKDOWN = """\
# Title Section

Intro paragraph before any subsection.

## First Subsection

Some prose here, short enough to stay in one chunk.

```python
def handler(item_id: int):
    if item_id < 0:
        raise ValueError("negative")
    return item_id
```

## Second Subsection

### Nested Header

Deeply nested content under the second subsection.
"""


def test_chunk_fixed_respects_token_budget() -> None:
    long_text = "This is one sentence about FastAPI routing. " * 200
    chunks = chunk_fixed(long_text, "fake.md")
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= config.chunk_size_tokens
        assert chunk.strategy is ChunkingStrategy.FIXED
        assert chunk.source_file == "fake.md"


def test_chunk_fixed_indexes_are_sequential() -> None:
    long_text = "Paragraph text. " * 300
    chunks = chunk_fixed(long_text, "fake.md")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert [c.chunk_id for c in chunks] == [f"fixed:fake.md:{i}" for i in range(len(chunks))]


def test_chunk_markdown_aware_preserves_code_indentation() -> None:
    chunks = chunk_markdown_aware(SAMPLE_MARKDOWN, "fake.md")
    joined = "\n".join(c.text for c in chunks)
    assert '    if item_id < 0:\n        raise ValueError("negative")' in joined


def test_chunk_markdown_aware_breadcrumbs_nest_by_header_level() -> None:
    chunks = chunk_markdown_aware(SAMPLE_MARKDOWN, "fake.md")
    sections = {c.section for c in chunks}
    assert "Title Section" in sections
    assert "Title Section > First Subsection" in sections
    assert "Title Section > Second Subsection" in sections
    assert "Title Section > Second Subsection > Nested Header" in sections


def test_chunk_markdown_aware_resets_breadcrumb_on_shallower_header() -> None:
    text = "# A\n\n## A1\n\ncontent\n\n# B\n\ncontent under B\n"
    chunks = chunk_markdown_aware(text, "fake.md")
    sections = [c.section for c in chunks]
    assert "A > A1" in sections
    assert "B" in sections
    assert "A > B" not in sections


def test_chunk_markdown_aware_ignores_hash_inside_code_fence() -> None:
    text = "# Real Header\n\n```python\n# not a header, just a comment\nx = 1\n```\n"
    chunks = chunk_markdown_aware(text, "fake.md")
    assert len(chunks) == 1
    assert chunks[0].section == "Real Header"
    assert "# not a header, just a comment" in chunks[0].text


def test_chunk_markdown_aware_splits_oversized_section() -> None:
    text = "# Big Section\n\n" + ("Filler sentence about FastAPI. " * 400)
    chunks = chunk_markdown_aware(text, "fake.md")
    assert len(chunks) > 1
    assert all(c.section == "Big Section" for c in chunks)
    assert all(c.token_count <= config.chunk_size_tokens for c in chunks)


def test_count_tokens_uses_the_embedding_models_wordpiece_tokens() -> None:
    assert count_tokens("") == 0
    # WordPiece splits identifiers finely: 10 tokens here, against 5 for tiktoken's cl100k.
    assert count_tokens("response_model_exclude_unset=True") == 10


def test_chunks_and_the_longest_question_fit_the_512_token_window() -> None:
    """Regression: a budget counted in tiktoken tokens let most fixed-size chunks overflow 512."""
    questions = [q.question for q in load_golden_set()] + [q.question for q in load_out_of_domain()]
    longest_question = max(count_tokens(q) for q in questions)
    code_heavy = 'Set response_model_exclude_unset=True on @app.get("/users/{user_id}"). ' * 300

    chunks = chunk_fixed(code_heavy, "fake.md")

    assert len(chunks) > 1
    for chunk in chunks:
        # [CLS] question [SEP] passage [SEP]: the passage's own [CLS]/[SEP] plus one more.
        assert len(_tokenizer().encode(chunk.text).ids) + longest_question + 1 <= 512


def test_a_chunk_budget_larger_than_the_model_window_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized chunks would be silently truncated at embedding time, so refuse to build them."""
    monkeypatch.setattr(config, "chunk_size_tokens", 600)

    with pytest.raises(ValueError, match="chunk_size_tokens=600"):
        chunk_fixed("Some text.", "fake.md")


def test_chunk_corpus_over_real_data_dir_is_non_empty() -> None:
    """Sanity check against the committed corpus (data/raw/) — no network involved."""
    if not config.raw_docs_dir.exists():
        return  # corpus not fetched in this environment; other tests don't need it
    chunks = chunk_corpus(config.raw_docs_dir, ChunkingStrategy.MARKDOWN)
    assert len(chunks) > 0
    assert all(c.source_file.endswith(".md") for c in chunks)
    assert all(Path(c.source_file).as_posix() == c.source_file for c in chunks)

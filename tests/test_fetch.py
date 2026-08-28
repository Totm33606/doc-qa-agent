"""Regression tests for the pure (non-networked) parts of ingestion.fetch's preprocessing.

Two of these guard bugs found while building the real corpus: `\\s*$` in
both `_HEADER_ID` and `_ADMONITION` originally ate the blank line that
followed a header or a closing `///`, silently merging two paragraphs into
one — see the git history / module docstring in ingestion/fetch.py.
"""

from __future__ import annotations

from ingestion.fetch import (
    _flatten_admonitions,
    _snippet_url,
    _strip_header_ids,
    _strip_termy_divs,
)


def test_strip_header_ids_removes_the_anchor_only() -> None:
    text = "# Path Parameters { #path-parameters }\n"
    assert _strip_header_ids(text) == "# Path Parameters\n"


def test_strip_header_ids_preserves_the_blank_line_after_a_header() -> None:
    text = "# Path Parameters { #path-parameters }\n\nSome paragraph text.\n"
    result = _strip_header_ids(text)
    assert result == "# Path Parameters\n\nSome paragraph text.\n"


def test_strip_header_ids_ignores_headers_without_an_anchor() -> None:
    text = "## Plain Header\n\nBody.\n"
    assert _strip_header_ids(text) == text


def test_flatten_admonitions_converts_tip_block_to_blockquote() -> None:
    text = "/// tip\n\nThis is a tip.\n\n///\n"
    result = _flatten_admonitions(text)
    assert result == "> **TIP:**\n> This is a tip.\n\n"


def test_flatten_admonitions_preserves_blank_line_after_block() -> None:
    text = "/// note\n\nA note.\n\n///\n\nNext paragraph.\n"
    result = _flatten_admonitions(text)
    assert result.endswith("\nNext paragraph.\n")


def test_flatten_admonitions_uses_custom_title_when_present() -> None:
    text = "/// tip | Custom Title\n\nBody text.\n\n///\n"
    result = _flatten_admonitions(text)
    assert result.startswith("> **Custom Title:**\n")


def test_strip_termy_divs_removes_wrapper_but_keeps_code_block() -> None:
    text = '<div class="termy">\n\n```console\n$ uv add sqlmodel\n```\n\n</div>\n'
    result = _strip_termy_divs(text)
    assert "<div" not in result
    assert "</div>" not in result
    assert "```console\n$ uv add sqlmodel\n```" in result


def test_snippet_url_anchors_from_docs_src_regardless_of_leading_dots() -> None:
    url = _snippet_url("../../docs_src/path_params/tutorial001_py310.py", "0.141.1")
    assert url == (
        "https://raw.githubusercontent.com/fastapi/fastapi/0.141.1/"
        "docs_src/path_params/tutorial001_py310.py"
    )


def test_snippet_url_returns_none_when_docs_src_is_absent() -> None:
    assert _snippet_url("some/other/path.py", "0.141.1") is None

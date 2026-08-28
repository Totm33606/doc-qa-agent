"""Tests for fetch.py's networked functions (_inline_snippets, run), using
httpx.MockTransport — httpx's own offline-testing mechanism, no extra dependency,
no real network call, but exercising the real httpx.Client code path.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ingestion.config import config
from ingestion.fetch import _inline_snippets, run


def test_inline_snippets_fetches_and_inlines_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "docs_src/example/foo.py" in str(request.url)
        return httpx.Response(200, text="x = 1\ny = 2\n")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    markdown = "Before\n\n{* ../../docs_src/example/foo.py *}\n\nAfter"

    result = _inline_snippets(markdown, "0.1.0", client)

    assert result == "Before\n\n```python\nx = 1\ny = 2\n```\n\nAfter"


def test_inline_snippets_applies_line_range_selector() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="a\nb\nc\nd\ne\n")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    markdown = "{* ../../docs_src/example/foo.py ln[2:4] *}"

    result = _inline_snippets(markdown, "0.1.0", client)

    assert result == "```python\nb\nc\nd\n```"


def test_inline_snippets_handles_404_gracefully() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    markdown = "Before {* ../../docs_src/missing.py *} After"

    result = _inline_snippets(markdown, "0.1.0", client)

    assert "{*" not in result
    assert "Before" in result
    assert "After" in result


def test_inline_snippets_skips_non_docs_src_macros_without_a_network_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should never be called — path has no docs_src/ anchor")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    result = _inline_snippets("{* some/other/path.py *}", "0.1.0", client)

    assert result == ""


# Captured at import time, before any test monkeypatches `httpx.Client` — the
# replacement below constructs a real Client (just wired to a fake transport), so it
# must reference the *original* class, not the (about to be patched) module attribute,
# or calling it recurses into itself.
_RealClient = httpx.Client


def _mock_client(**kwargs: object) -> httpx.Client:
    return _RealClient(transport=httpx.MockTransport(_mock_handler))


def _mock_handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url.endswith("docs/en/docs/tutorial/fake-page.md"):
        return httpx.Response(200, text="# Fake Page\n\n{* ../../docs_src/fake/example.py *}\n")
    if url.endswith("docs_src/fake/example.py"):
        return httpx.Response(200, text="def hello():\n    return 'hi'\n")
    if url.endswith("docs/en/docs/tutorial/missing-page.md"):
        return httpx.Response(404)
    return httpx.Response(404)


def test_run_fetches_cleans_and_writes_pages_plus_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr("ingestion.fetch.CORPUS_PAGES", ["tutorial/fake-page.md"])
    monkeypatch.setattr(config, "raw_docs_dir", raw_dir)
    monkeypatch.setattr(config, "corpus_manifest_path", raw_dir / "manifest.json")
    monkeypatch.setattr("ingestion.fetch.httpx.Client", _mock_client)

    run(ref="test-ref")

    out_file = raw_dir / "tutorial" / "fake-page.md"
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "```python" in content
    assert "def hello():" in content

    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ref"] == "test-ref"
    assert manifest["pages"] == ["tutorial/fake-page.md"]
    assert manifest["n_pages_fetched"] == 1


def test_run_skips_pages_that_fail_to_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr("ingestion.fetch.CORPUS_PAGES", ["tutorial/missing-page.md"])
    monkeypatch.setattr(config, "raw_docs_dir", raw_dir)
    monkeypatch.setattr(config, "corpus_manifest_path", raw_dir / "manifest.json")
    monkeypatch.setattr("ingestion.fetch.httpx.Client", _mock_client)

    run(ref="test-ref")

    assert not (raw_dir / "tutorial" / "missing-page.md").exists()
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pages"] == []
    assert manifest["n_pages_fetched"] == 0

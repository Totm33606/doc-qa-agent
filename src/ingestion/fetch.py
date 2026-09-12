"""Fetch a curated snapshot of the FastAPI docs' Markdown source at a pinned tag.

Pages come from the `fastapi/fastapi` repository at `config.fastapi_repo_ref` and are
cleaned before being written to `data/raw/`: `{* path *}` snippet macros are inlined,
`/// tip` admonitions flattened, `{ #anchor }` header IDs and termy `<div>` wrappers
stripped. Only needed to refresh the corpus.

Run: `uv run python -m ingestion.fetch`
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime

import httpx
import typer

from common.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

RAW_BASE = "https://raw.githubusercontent.com/fastapi/fastapi"

# The core tutorial plus commonly looked-up advanced pages. `reference/`, `deployment/`,
# `about/` and meta pages are excluded (see the README's "Corpus" section).
TUTORIAL_PAGES = [
    "tutorial/first-steps.md",
    "tutorial/path-params.md",
    "tutorial/query-params.md",
    "tutorial/body.md",
    "tutorial/query-params-str-validations.md",
    "tutorial/path-params-numeric-validations.md",
    "tutorial/query-param-models.md",
    "tutorial/body-multiple-params.md",
    "tutorial/body-fields.md",
    "tutorial/body-nested-models.md",
    "tutorial/schema-extra-example.md",
    "tutorial/extra-data-types.md",
    "tutorial/cookie-params.md",
    "tutorial/cookie-param-models.md",
    "tutorial/header-params.md",
    "tutorial/header-param-models.md",
    "tutorial/response-model.md",
    "tutorial/extra-models.md",
    "tutorial/response-status-code.md",
    "tutorial/request-forms.md",
    "tutorial/request-forms-and-files.md",
    "tutorial/request-files.md",
    "tutorial/request-form-models.md",
    "tutorial/handling-errors.md",
    "tutorial/path-operation-configuration.md",
    "tutorial/encoder.md",
    "tutorial/body-updates.md",
    "tutorial/dependencies/index.md",
    "tutorial/dependencies/classes-as-dependencies.md",
    "tutorial/dependencies/sub-dependencies.md",
    "tutorial/dependencies/dependencies-in-path-operation-decorators.md",
    "tutorial/dependencies/global-dependencies.md",
    "tutorial/dependencies/dependencies-with-yield.md",
    "tutorial/security/index.md",
    "tutorial/security/first-steps.md",
    "tutorial/security/get-current-user.md",
    "tutorial/security/simple-oauth2.md",
    "tutorial/security/oauth2-jwt.md",
    "tutorial/middleware.md",
    "tutorial/cors.md",
    "tutorial/sql-databases.md",
    "tutorial/bigger-applications.md",
    "tutorial/background-tasks.md",
    "tutorial/metadata.md",
    "tutorial/static-files.md",
    "tutorial/testing.md",
    "tutorial/debugging.md",
]

ADVANCED_PAGES = [
    "advanced/additional-status-codes.md",
    "advanced/additional-responses.md",
    "advanced/response-directly.md",
    "advanced/custom-response.md",
    "advanced/response-cookies.md",
    "advanced/response-headers.md",
    "advanced/response-change-status-code.md",
    "advanced/advanced-dependencies.md",
    "advanced/security/oauth2-scopes.md",
    "advanced/security/http-basic-auth.md",
    "advanced/events.md",
    "advanced/middleware.md",
    "advanced/sub-applications.md",
    "advanced/websockets.md",
    "advanced/settings.md",
    "advanced/testing-dependencies.md",
    "advanced/testing-events.md",
    "advanced/testing-websockets.md",
    "advanced/path-operation-advanced-configuration.md",
    "advanced/async-tests.md",
]

CORPUS_PAGES = TUTORIAL_PAGES + ADVANCED_PAGES

_SNIPPET_MACRO = re.compile(r"\{\*\s*(?P<path>\S+?)(?P<selectors>(?:\s+\w+\[[^\]]*\])*)\s*\*\}")
_LINE_RANGE = re.compile(r"\bln\[(\d+):(\d+)\]")
_ADMONITION = re.compile(
    r"^///[ \t]*(?P<kind>[a-zA-Z-]+)(?:[ \t]*\|[ \t]*(?P<title>.+))?\n"
    r"(?P<body>(?:.*\n)*?)"
    r"^///[ \t]*$",
    re.MULTILINE,
)
_HEADER_ID = re.compile(r"^(#{1,6}[ \t]+.+?)[ \t]*\{[ \t]*#[\w-]+[ \t]*\}[ \t]*$", re.MULTILINE)
# The "termy" animated-terminal HTML wrapper; the code block inside is kept. Matched as a
# whole block so the `</div>` of other divs (e.g. screenshots) is left alone.
_TERMY_DIV = re.compile(
    r"^<div class=\"termy\">\n\n?(?P<body>.*?)\n?</div>\n", re.MULTILINE | re.DOTALL
)


def _snippet_url(macro_path: str, ref: str) -> str | None:
    """Resolve a `{* ../../docs_src/foo/bar.py *}` macro path to a raw-content URL.

    Macros always target the repo-root `docs_src/`, whatever the page depth, so the path
    is anchored at `docs_src/` rather than resolved relative to the page.
    """
    if "docs_src/" not in macro_path:
        return None
    anchored = macro_path[macro_path.index("docs_src/") :]
    return f"{RAW_BASE}/{ref}/{anchored}"


def _inline_snippets(markdown: str, ref: str, client: httpx.Client) -> str:
    """Resolve every `{* path [ln[a:b]] [hl[...]] [title[...]] *}` macro to a code block.

    The `ln[a:b]` selector (1-indexed, inclusive) is honored: pages like `sql-databases.md`
    show the same file many times, a few lines at a time.
    """

    def _replace(match: re.Match[str]) -> str:
        macro_path = match.group("path")
        url = _snippet_url(macro_path, ref)
        if url is None:
            return ""
        response = client.get(url, timeout=15.0)
        if response.status_code != 200:
            logger.warning("Snippet fetch failed (%s): %s", response.status_code, url)
            return ""
        code = response.text.rstrip("\n")
        line_range = _LINE_RANGE.search(match.group("selectors"))
        if line_range:
            start, end = int(line_range.group(1)), int(line_range.group(2))
            code = "\n".join(code.splitlines()[start - 1 : end])
        return f"```python\n{code}\n```"

    return _SNIPPET_MACRO.sub(_replace, markdown)


def _flatten_admonitions(markdown: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        kind = match.group("kind").strip().upper()
        title = match.group("title")
        body = match.group("body").strip()
        label = title.strip() if title else kind
        quoted = "\n".join(f"> {line}" if line else ">" for line in body.splitlines())
        return f"> **{label}:**\n{quoted}\n"

    return _ADMONITION.sub(_replace, markdown)


def _strip_header_ids(markdown: str) -> str:
    return _HEADER_ID.sub(r"\1", markdown)


def _strip_termy_divs(markdown: str) -> str:
    return _TERMY_DIV.sub(r"\g<body>", markdown)


@app.command()
def run(ref: str = typer.Option(None, help="FastAPI git ref/tag to pin the corpus to.")) -> None:
    """Fetch, clean and write every page in CORPUS_PAGES to data/raw/, plus a manifest."""
    resolved_ref = ref or config.fastapi_repo_ref
    config.raw_docs_dir.mkdir(parents=True, exist_ok=True)

    fetched: list[str] = []
    with httpx.Client(follow_redirects=True) as client:
        for page in CORPUS_PAGES:
            url = f"{RAW_BASE}/{resolved_ref}/docs/en/docs/{page}"
            response = client.get(url, timeout=15.0)
            if response.status_code != 200:
                logger.warning("Page fetch failed (%s): %s", response.status_code, url)
                continue

            cleaned = _inline_snippets(response.text, resolved_ref, client)
            cleaned = _flatten_admonitions(cleaned)
            cleaned = _strip_header_ids(cleaned)
            cleaned = _strip_termy_divs(cleaned)
            out_path = config.raw_docs_dir / page
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(cleaned, encoding="utf-8")
            fetched.append(page)
            logger.info("Wrote %s (%d bytes)", page, len(cleaned))

    manifest = {
        "source_repo": "fastapi/fastapi",
        "ref": resolved_ref,
        "fetched_at": datetime.now(UTC).isoformat(),
        "n_pages_requested": len(CORPUS_PAGES),
        "n_pages_fetched": len(fetched),
        "pages": fetched,
    }
    config.corpus_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "Fetched %d/%d pages at ref=%s -> %s",
        len(fetched),
        len(CORPUS_PAGES),
        resolved_ref,
        config.raw_docs_dir,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

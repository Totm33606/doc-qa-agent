"""Two chunking strategies, compared head-to-head by eval/run_eval.py.

**Fixed-size** (`chunk_fixed`): a `RecursiveCharacterTextSplitter` over a
paragraph -> line -> sentence -> word -> char separator hierarchy, sized in
real tokens (via `tiktoken`, not characters) so "~500 tokens" means what it
says regardless of how dense a page's prose is. This is the MVP default:
simple, fast, and blind to document structure.

**Markdown-aware** (`chunk_markdown_aware`): first split by header, so a
chunk boundary never falls in the middle of a section — each chunk carries
the header breadcrumb it came from (e.g. "Path Parameters > Order
matters") as citable metadata. A section that's still over the token
budget is then run back through the same recursive splitter, so both
strategies share one token budget and one splitting algorithm for the "too
big" case — the only difference is *where the first cut is made*.

The header split is a small hand-rolled, fence-aware line scanner
(`_split_by_headers`), not LangChain's own `MarkdownHeaderTextSplitter`:
that splitter round-trips content through the `markdown` HTML library,
which collapses code-block indentation (`    return {"item_id": item_id}`
becomes `return {"item_id": item_id}`) — silently corrupting every Python
example in a docs corpus that is mostly Python examples. The hand-rolled
version tracks fenced-code-block state line by line and never touches a
line's content, only decides where the cut points are.
"""

from __future__ import annotations

import re
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.schemas import ChunkingStrategy, DocChunk
from ingestion.config import config

_encoding = tiktoken.get_encoding(config.token_encoding)

# Paragraph, then line, then sentence, then word, then character — the same
# separator hierarchy LangChain's own default uses, tried in order until a
# piece fits inside the token budget.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_MAX_HEADER_LEVEL = 4
_FENCE_RE = re.compile(r"^(```|~~~)")
_HEADER_RE = re.compile(r"^(#{1," + str(_MAX_HEADER_LEVEL) + r"})\s+(.+?)\s*$")

_NO_SECTION = "(no section — fixed-size chunking)"
_DOCUMENT_ROOT = "(document root)"


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def _chunk_id(source_file: str, strategy: ChunkingStrategy, index: int) -> str:
    return f"{strategy.value}:{source_file}:{index}"


def _recursive_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size_tokens,
        chunk_overlap=config.chunk_overlap_tokens,
        length_function=count_tokens,
        separators=_SEPARATORS,
    )


def chunk_fixed(text: str, source_file: str) -> list[DocChunk]:
    """Fixed-size chunking: no awareness of headers or document structure."""
    pieces = _recursive_splitter().split_text(text)
    return [
        DocChunk(
            chunk_id=_chunk_id(source_file, ChunkingStrategy.FIXED, i),
            text=piece,
            source_file=source_file,
            section=_NO_SECTION,
            strategy=ChunkingStrategy.FIXED,
            token_count=count_tokens(piece),
            chunk_index=i,
        )
        for i, piece in enumerate(pieces)
    ]


def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Split `text` into (header breadcrumb, section content) pairs.

    Scans line by line, tracking fenced-code-block state so a `#` inside a
    Python comment or an f-string never gets mistaken for a Markdown
    header, and never rewrites a line's content — only decides where
    section boundaries fall. Deeper headers reset when a shallower one
    appears (an `##` under an `#` gets nested; a new `#` clears both).
    """
    breadcrumb_stack: list[str | None] = [None] * _MAX_HEADER_LEVEL
    sections: list[tuple[str, str]] = []
    current: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        content = "".join(current).strip("\n")
        if content.strip():
            crumbs = [c for c in breadcrumb_stack if c]
            breadcrumb = " > ".join(crumbs) if crumbs else _DOCUMENT_ROOT
            sections.append((breadcrumb, content))

    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        fence_match = _FENCE_RE.match(stripped)
        if fence_match:
            if not in_fence:
                in_fence, fence_marker = True, fence_match.group(1)
            elif stripped.startswith(fence_marker):
                in_fence = False
            current.append(line)
            continue

        header_match = None if in_fence else _HEADER_RE.match(stripped)
        if header_match:
            flush()
            current = [line]
            level = len(header_match.group(1))
            breadcrumb_stack[level - 1] = header_match.group(2)
            for i in range(level, _MAX_HEADER_LEVEL):
                breadcrumb_stack[i] = None
            continue

        current.append(line)

    flush()
    return sections


def chunk_markdown_aware(text: str, source_file: str) -> list[DocChunk]:
    """Header-aware chunking: split by section first, then by size only if needed."""
    sections = _split_by_headers(text)
    sub_splitter = _recursive_splitter()

    chunks: list[DocChunk] = []
    index = 0
    for breadcrumb, content in sections:
        if count_tokens(content) <= config.chunk_size_tokens:
            pieces = [content]
        else:
            pieces = sub_splitter.split_text(content)
        for piece in pieces:
            chunks.append(
                DocChunk(
                    chunk_id=_chunk_id(source_file, ChunkingStrategy.MARKDOWN, index),
                    text=piece,
                    source_file=source_file,
                    section=breadcrumb,
                    strategy=ChunkingStrategy.MARKDOWN,
                    token_count=count_tokens(piece),
                    chunk_index=index,
                )
            )
            index += 1
    return chunks


def chunk_corpus(raw_docs_dir: Path, strategy: ChunkingStrategy) -> list[DocChunk]:
    """Chunk every Markdown file under `raw_docs_dir`, in a stable (sorted) order."""
    chunk_one = chunk_fixed if strategy is ChunkingStrategy.FIXED else chunk_markdown_aware
    chunks: list[DocChunk] = []
    for path in sorted(raw_docs_dir.rglob("*.md")):
        source_file = path.relative_to(raw_docs_dir).as_posix()
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_one(text, source_file))
    return chunks

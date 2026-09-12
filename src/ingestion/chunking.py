"""Two chunking strategies sharing one token budget, sized in real tokens with `tiktoken`.

- `chunk_fixed`: `RecursiveCharacterTextSplitter` over paragraph -> line -> sentence ->
  word -> char separators, blind to document structure.
- `chunk_markdown_aware`: split by header first, so each chunk carries its breadcrumb
  (e.g. "Path Parameters > Order matters"); sections still over budget are then split
  by the same recursive splitter.

The header split is hand-rolled because LangChain's `MarkdownHeaderTextSplitter` strips
every line, code blocks included, which destroys Python indentation.
"""

from __future__ import annotations

import re
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.config import config
from common.schemas import ChunkingStrategy, DocChunk

_encoding = tiktoken.get_encoding(config.token_encoding)

# Paragraph, line, sentence, word, character: tried in order until a piece fits the budget.
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

    Fence-aware, so a `#` comment in a code block isn't a header, and never rewrites a
    line. A header clears the breadcrumb levels below it.
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

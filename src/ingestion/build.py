"""Orchestrate ingestion end to end: chunk the corpus (both strategies) -> embed -> store.

Run: `uv run python -m ingestion.build` (requires `data/raw/` to already be
populated by `ingestion.fetch`). Rebuilds both Chroma collections from
scratch each run, so it's safe to re-run after editing chunking parameters.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import typer

from common.schemas import ChunkingStrategy, DocChunk
from ingestion.chunking import chunk_corpus
from ingestion.config import config
from ingestion.embed import BGEEmbedder, Embedder
from ingestion.store import ChunkStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def _batched(chunks: list[DocChunk], batch_size: int) -> Iterator[list[DocChunk]]:
    for i in range(0, len(chunks), batch_size):
        yield chunks[i : i + batch_size]


def build_collection(strategy: ChunkingStrategy, embedder: Embedder, batch_size: int = 64) -> int:
    """Chunk, embed and store the whole corpus for one strategy. Returns chunk count."""
    chunks = chunk_corpus(config.raw_docs_dir, strategy)
    if not chunks:
        raise FileNotFoundError(
            f"No chunks produced from {config.raw_docs_dir} — run `uv run python -m ingestion.fetch` first."
        )

    store = ChunkStore(
        persist_dir=config.chroma_dir, collection_name=config.collection_name(strategy.value)
    )
    store.reset()

    for batch in _batched(chunks, batch_size):
        vectors = embedder.embed_documents([c.text for c in batch])
        store.add(batch, vectors)

    return len(chunks)


@app.command()
def run(batch_size: int = 64) -> None:
    """Build both the fixed-size and markdown-aware collections from data/raw/."""
    embedder = BGEEmbedder()
    for strategy in ChunkingStrategy:
        t0 = time.perf_counter()
        n_chunks = build_collection(strategy, embedder, batch_size=batch_size)
        elapsed = time.perf_counter() - t0
        logger.info(
            "strategy=%s: %d chunks embedded and stored in %.1fs -> %s",
            strategy.value,
            n_chunks,
            elapsed,
            config.chroma_dir / config.collection_name(strategy.value),
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

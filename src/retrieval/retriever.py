"""Query-time retrieval: embed a question, search one strategy's Chroma collection.

Deliberately thin — the interesting logic (chunking, the embedding
asymmetry between queries and documents) lives in `ingestion/`. This
module's only job is to be the one thing `generation` and the API import.
"""

from __future__ import annotations

from common.schemas import ChunkingStrategy, RetrievedPassage
from ingestion.config import config
from ingestion.embed import Embedder
from ingestion.store import ChunkStore


class Retriever:
    def __init__(
        self, embedder: Embedder, strategy: ChunkingStrategy, store: ChunkStore | None = None
    ) -> None:
        self._embedder = embedder
        self._strategy = strategy
        self._store = store or ChunkStore(
            persist_dir=config.chroma_dir, collection_name=config.collection_name(strategy.value)
        )

    @property
    def strategy(self) -> ChunkingStrategy:
        return self._strategy

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedPassage]:
        k = top_k or config.default_top_k
        query_embedding = self._embedder.embed_query(question)
        return self._store.query(query_embedding, top_k=k)

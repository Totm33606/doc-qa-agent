"""Thin wrapper around a Chroma collection: storing chunks and querying by embedding.

Embeddings are always computed by an `ingestion.embed.Embedder` and passed
in explicitly (`add(chunks, embeddings=...)`), rather than letting Chroma
call its own embedding function internally — that's what lets
`BGEEmbedder.embed_query` apply BGE's asymmetric query-instruction prefix
(see `ingestion/embed.py`) while `embed_documents` stays unprefixed, a
distinction Chroma's single-embedding-function model can't express.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api import ClientAPI

from common.schemas import DocChunk, RetrievedPassage


class ChunkStore:
    def __init__(self, persist_dir: Path | None, collection_name: str) -> None:
        self._client: ClientAPI = (
            chromadb.EphemeralClient()
            if persist_dir is None
            else chromadb.PersistentClient(path=str(persist_dir))
        )
        self._collection = self._client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[DocChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,  # type: ignore[arg-type]  # chromadb's stubs reject list[list[float]] due to List invariance; a plain nested list is valid at runtime
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source_file": c.source_file,
                    "section": c.section,
                    "strategy": c.strategy.value,
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                }
                for c in chunks
            ],
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedPassage]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]  # see add()'s ignore above
            n_results=top_k,
        )
        ids = result["ids"][0]
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []

        passages = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            passages.append(
                RetrievedPassage(
                    chunk_id=chunk_id,
                    text=text,
                    source_file=str(metadata["source_file"]),
                    section=str(metadata["section"]),
                    score=1.0 - distance,  # cosine distance -> cosine similarity
                )
            )
        return passages

    def count(self) -> int:
        return self._collection.count()

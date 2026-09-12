"""Thin wrapper around a Chroma collection: storing chunks and querying by embedding.

Embeddings are computed by an `Embedder` and passed in rather than delegated to a Chroma
embedding function, so queries and documents can be embedded differently (BGE's query
instruction).
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api import ClientAPI

from common.schemas import ChunkingStrategy, DocChunk, RetrievedPassage


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
            embeddings=embeddings,  # type: ignore[arg-type]  # stubs reject list[list[float]]
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
            query_embeddings=[query_embedding],  # type: ignore[arg-type]  # see add()
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

    def get_all(self) -> list[DocChunk]:
        """Every stored chunk, rebuilt from its documents + metadata (the BM25 index source)."""
        if self._collection.count() == 0:
            return []
        result = self._collection.get(include=["documents", "metadatas"])
        ids = result["ids"]
        documents = result["documents"] or []
        metadatas = result["metadatas"] or []

        return [
            DocChunk(
                chunk_id=chunk_id,
                text=text,
                source_file=str(metadata["source_file"]),
                section=str(metadata["section"]),
                strategy=ChunkingStrategy(metadata["strategy"]),
                chunk_index=int(metadata["chunk_index"]),  # type: ignore[arg-type]  # union-typed; add() writes ints
                token_count=int(metadata["token_count"]),  # type: ignore[arg-type]  # see above
            )
            for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=True)
        ]

    def count(self) -> int:
        return self._collection.count()

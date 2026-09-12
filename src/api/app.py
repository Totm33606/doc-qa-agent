"""The FastAPI serving layer: `POST /ask` wires retrieval and generation together per request.

Endpoints are plain `def`, not `async def` — embedding a query and calling
the LLM are both blocking calls, and FastAPI runs sync path functions in a
worker thread automatically, so this avoids stalling the event loop
without needing to wrap every call in `run_in_threadpool` by hand.

Run: `uv run uvicorn api.app:app --reload --port 8000`
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from common.schemas import AskRequest, AskResponse, ChunkingStrategy, RetrievalMode
from generation.generate import generate_answer
from ingestion.embed import BGEEmbedder, Embedder
from retrieval.rerank import CrossEncoderReranker, Reranker
from retrieval.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DocQARuntime:
    """Owns the models and one retriever per (chunking strategy, retrieval mode), built at startup.

    One retriever per combination, so a request never pays to construct one —
    and the expensive parts are per-mode: only the hybrid ones build a BM25
    index, only the re-ranking ones use the cross-encoder. The embedder and
    the re-ranker are each loaded **once** and shared by every retriever that
    needs them, rather than letting each construct its own copy of the same
    model.
    """

    def __init__(self) -> None:
        self.embedder: Embedder | None = None
        self.reranker: Reranker | None = None
        self.retrievers: dict[tuple[ChunkingStrategy, RetrievalMode], Retriever] = {}

    def start(self) -> None:
        self.embedder = BGEEmbedder()
        self.reranker = CrossEncoderReranker()
        self.retrievers = {
            (strategy, mode): Retriever(self.embedder, strategy, mode, reranker=self.reranker)
            for strategy in ChunkingStrategy
            for mode in RetrievalMode
        }

    def stop(self) -> None:
        self.retrievers = {}
        self.embedder = None
        self.reranker = None

    def ask(self, request: AskRequest) -> AskResponse:
        if self.embedder is None:
            raise RuntimeError("Runtime not started — call `runtime.start()` first.")
        retriever = self.retrievers[(request.strategy, request.mode)]
        passages = retriever.retrieve(request.question, top_k=request.top_k)
        return generate_answer(request.question, passages)


runtime = DocQARuntime()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    runtime.start()
    yield
    runtime.stop()


app = FastAPI(
    title="DocQA-Agent",
    description="Retrieval-augmented Q&A over the official FastAPI documentation, with sourced citations.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    try:
        return runtime.ask(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ask failed for question=%r", payload.question)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

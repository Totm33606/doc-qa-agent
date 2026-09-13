"""FastAPI serving layer: `POST /ask` runs retrieval, then generation.

Endpoints are plain `def`: the model and LLM calls block, and FastAPI runs sync
endpoints in a worker thread.

Run: `uv run uvicorn api.app:app --reload --port 8000`
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from common.schemas import AskRequest, AskResponse, ChunkingStrategy, RetrievalMode
from generation.generate import generate_answer
from ingestion.embed import BGEEmbedder, Embedder, QueryTooLongError
from retrieval.rerank import CrossEncoderReranker, Reranker
from retrieval.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class RuntimeNotStartedError(RuntimeError):
    """`/ask` was called before the lifespan loaded the models."""


class DocQARuntime:
    """Loads both models once at startup and builds one retriever per (strategy, mode)."""

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
            raise RuntimeNotStartedError("Runtime not started — call `runtime.start()` first.")
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
    except RuntimeNotStartedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QueryTooLongError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ask failed for question=%r", payload.question)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

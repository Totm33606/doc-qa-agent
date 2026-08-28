"""API tests. Fully hermetic: the real embedder and the real LLM are both monkeypatched out
before the app's `lifespan` runs — see advanced/testing-events.md in the corpus itself for why
`with TestClient(app) as client:` (not a bare `TestClient(app)`) is required to trigger it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from common.schemas import ChunkingStrategy, DocChunk
from ingestion.config import config
from ingestion.store import ChunkStore
from tests.conftest import FakeChatModel, FakeEmbedder

embedder = FakeEmbedder()


def _chunk(strategy: ChunkingStrategy) -> DocChunk:
    return DocChunk(
        chunk_id=f"{strategy.value}-c1",
        text="Declare a fixed path like /users/me before a variable path like /users/{user_id}.",
        source_file="tutorial/path-params.md",
        section="Order matters",
        strategy=strategy,
        token_count=12,
        chunk_index=0,
    )


def _seed_collections(chroma_dir: Path) -> None:
    for strategy in ChunkingStrategy:
        store = ChunkStore(
            persist_dir=chroma_dir, collection_name=config.collection_name(strategy.value)
        )
        chunk = _chunk(strategy)
        store.add([chunk], embedder.embed_documents([chunk.text]))


def test_health_endpoint_does_not_require_lifespan() -> None:
    import api.app as app_module

    client = TestClient(app_module.app)  # no `with` -> lifespan never runs
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.fixture
def hermetic_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> object:
    import api.app as app_module

    monkeypatch.setattr(config, "chroma_dir", tmp_path / "chroma")
    _seed_collections(config.chroma_dir)
    monkeypatch.setattr(app_module, "BGEEmbedder", lambda: embedder)
    monkeypatch.setattr(
        "generation.generate.build_llm",
        lambda: FakeChatModel("Fixed paths must come before variable ones. [source: 1]"),
    )
    return app_module.app


def test_ask_returns_grounded_answer(hermetic_app: object) -> None:
    with TestClient(hermetic_app) as client:  # type: ignore[arg-type]
        response = client.post(
            "/ask",
            json={"question": "Why does path order matter?", "top_k": 3, "strategy": "markdown"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Fixed paths must come before variable ones.")
    assert len(body["passages"]) == 1
    assert body["passages"][0]["source_file"] == "tutorial/path-params.md"
    assert body["groundedness_score"] == 1.0
    assert body["citations"][0]["matched_passage"] is True


def test_ask_defaults_top_k_and_strategy(hermetic_app: object) -> None:
    with TestClient(hermetic_app) as client:  # type: ignore[arg-type]
        response = client.post("/ask", json={"question": "Why does path order matter?"})

    assert response.status_code == 200


def test_ask_rejects_invalid_top_k(hermetic_app: object) -> None:
    with TestClient(hermetic_app) as client:  # type: ignore[arg-type]
        response = client.post("/ask", json={"question": "Why does path order matter?", "top_k": 0})
    assert response.status_code == 422


def test_ask_returns_502_on_unexpected_generation_failure(
    hermetic_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _RaisingChatModel:
        def invoke(self, messages: object) -> object:
            raise ValueError("simulated LLM failure")

    monkeypatch.setattr("generation.generate.build_llm", lambda: _RaisingChatModel())

    with TestClient(hermetic_app) as client:  # type: ignore[arg-type]
        response = client.post("/ask", json={"question": "Why does path order matter?"})

    assert response.status_code == 502


def test_ask_before_startup_returns_503() -> None:
    import api.app as app_module

    client = TestClient(app_module.app)  # lifespan never runs -> runtime not started
    response = client.post("/ask", json={"question": "Anything?"})
    assert response.status_code == 503

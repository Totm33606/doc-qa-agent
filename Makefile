.PHONY: install fetch build eval api test lint typecheck fmt clean

install:
	uv venv && uv pip install -e ".[dev]"

fetch:
	uv run python -m ingestion.fetch

build:
	uv run python -m ingestion.build

eval:
	uv run python -m eval.run_eval

eval-retrieval-only:
	uv run python -m eval.run_eval --skip-generation

api:
	uv run uvicorn api.app:app --reload --port 8000

test:
	uv run pytest

lint:
	uv run ruff check src tests eval

typecheck:
	uv run mypy src eval

fmt:
	uv run ruff format src tests eval

clean:
	rm -rf data/chroma eval/eval_report.json .pytest_cache .ruff_cache .mypy_cache **/__pycache__

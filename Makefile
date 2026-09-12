.PHONY: install fetch build eval eval-retrieval-only api test lint typecheck fmt clean

install:
	uv sync --extra dev

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
	uv run mypy src tests eval

fmt:
	uv run ruff format src tests eval

# Python rather than `rm` so it also works when GNU Make uses cmd.exe (Windows);
# one line because a trailing backslash means different things to sh and cmd.
clean:
	uv run python -c "import pathlib,shutil;rm=lambda p:shutil.rmtree(p,ignore_errors=True);[rm(d) for d in ('data/chroma','.pytest_cache','.ruff_cache','.mypy_cache')];[rm(d) for d in pathlib.Path('.').rglob('__pycache__')];[pathlib.Path(f).unlink(missing_ok=True) for f in ('eval/eval_report.json','eval/eval_details.md','.coverage')]"

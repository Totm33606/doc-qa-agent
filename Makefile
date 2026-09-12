.PHONY: install fetch build eval eval-retrieval-only api test lint typecheck fmt clean

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
	uv run mypy src tests eval

fmt:
	uv run ruff format src tests eval

# Written in Python, unlike every other target's shell one-liner, because
# GNU Make runs recipes through cmd.exe on Windows — which has no `rm`, so
# `make clean` simply failed there. `uv run python` is already how the rest
# of this file works, so it costs nothing. It also fixes a quieter bug in the
# old recipe: without `globstar`, sh collapses `**/__pycache__` to
# `*/__pycache__`, so nested caches were never actually removed.
#
# One physical line on purpose: a trailing backslash would be a line
# continuation for sh and a syntax error for cmd.exe.
clean:
	uv run python -c "import pathlib,shutil;rm=lambda p:shutil.rmtree(p,ignore_errors=True);[rm(d) for d in ('data/chroma','.pytest_cache','.ruff_cache','.mypy_cache')];[rm(d) for d in pathlib.Path('.').rglob('__pycache__')];[pathlib.Path(f).unlink(missing_ok=True) for f in ('eval/eval_report.json','eval/eval_details.md','.coverage')]"

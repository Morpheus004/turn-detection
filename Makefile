.PHONY: install format lint check test clean pre-commit-install

install:
	uv sync

install-dev:
	uv sync --group dev
	uv run pre-commit install

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run pyright src scripts app.py

check: format lint

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

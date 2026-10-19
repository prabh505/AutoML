# TabPilot developer commands (PRD §34.1, CLAUDE.md §8).
.DEFAULT_GOAL := check
UV ?= uv
RUN := $(UV) run --frozen

.PHONY: setup lint format format-check type contracts test test-all check schemas requirements requirements-check

setup:
	$(UV) sync --all-extras --frozen

lint:
	$(RUN) ruff check .

format:
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

format-check:
	$(RUN) ruff format --check .

type:
	$(RUN) mypy

contracts:
	$(RUN) lint-imports

# Default selection excludes perf, live and slow (pyproject addopts).
test:
	$(RUN) pytest --cov --cov-report=term-missing

# Everything except live provider calls.
test-all:
	$(RUN) pytest --cov --cov-report=term-missing -m "not live"

check: lint format-check type contracts test

# Implemented in P02 (scripts/export_schemas.py).
schemas:
	$(RUN) python scripts/export_schemas.py

requirements:
	scripts/export_requirements.sh

requirements-check: requirements
	git diff --exit-code requirements.txt

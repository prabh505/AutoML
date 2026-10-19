# TabPilot developer commands (PRD §34.1, CLAUDE.md §8).
.DEFAULT_GOAL := check
UV ?= uv
RUN := $(UV) run --frozen

.PHONY: setup lint format format-check type contracts test test-all check schemas schemas-check requirements requirements-check

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

schemas:
	$(RUN) python scripts/export_schemas.py

# Fails if docs/schemas is stale, or (with SCHEMA_BASE=<git ref>) if a contract changed
# without a schema_version bump.
schemas-check:
	$(RUN) python scripts/export_schemas.py --check $(if $(SCHEMA_BASE),--base $(SCHEMA_BASE))

requirements:
	scripts/export_requirements.sh

requirements-check: requirements
	git diff --exit-code requirements.txt

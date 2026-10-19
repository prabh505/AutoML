"""Shared pytest configuration. P03 adds the full harness (fixtures, spies, stubs)."""

from collections.abc import Iterator

import pytest

from tabpilot import logging as tp_logging


@pytest.fixture(autouse=True)
def _strict_logging() -> Iterator[None]:
    """In tests, passing dataset content to the logger raises (PRD §30.3, P02)."""
    tp_logging.set_strict(True)
    yield
    tp_logging.set_strict(False)

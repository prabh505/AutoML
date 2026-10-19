"""Shared pytest configuration and fixtures (P02 strict logging; P03 harness).

Network is blocked for every test by pytest-socket (`--disable-socket` in pyproject). Tests
marked `live` can request `allow_network` to re-enable it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pytest_socket import disable_socket, enable_socket
from tests.helpers.clock import FakeClock
from tests.helpers.fit_spy import reset_fit_log
from ulid import ULID

from tabpilot import logging as tp_logging

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(autouse=True)
def _strict_logging() -> Iterator[None]:
    """In tests, passing dataset content to the logger raises (PRD §30.3, P02)."""
    tp_logging.set_strict(True)
    yield
    tp_logging.set_strict(False)


@pytest.fixture(autouse=True)
def _fresh_fit_log() -> Iterator[None]:
    reset_fit_log()
    yield
    reset_fit_log()


@pytest.fixture
def allow_network(request: pytest.FixtureRequest) -> Iterator[None]:
    """Re-enable sockets. Only for tests marked `live`."""
    if request.node.get_closest_marker("live") is None:
        raise pytest.UsageError("allow_network is only for tests marked @pytest.mark.live")
    enable_socket()
    yield
    disable_socket(allow_unix_socket=True)


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """A fresh `runs/{run_id}/` directory with mode 0700 (PRD §24.3)."""
    run_dir = tmp_path / "runs" / str(ULID())
    run_dir.mkdir(parents=True, mode=0o700)
    os.chmod(run_dir, 0o700)
    return run_dir

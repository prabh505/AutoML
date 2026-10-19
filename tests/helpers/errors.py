"""Helpers for asserting user-facing errors (PRD §31.4 negative-test principle)."""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator
from typing import Any

import pytest

from tabpilot.schemas import ErrorCode, TabPilotError

_PATH = re.compile(r"(/home/|/tmp/|/var/|/Users/|[A-Za-z]:\\\\)")


@contextlib.contextmanager
def raises_error(
    code: ErrorCode, **expected_params: Any
) -> Iterator[pytest.ExceptionInfo[TabPilotError]]:
    """Assert that the block raises `TabPilotError(code)` with these params and a clean message.

    The message must not contain stack traces or file-system paths (§8.9).
    """
    with pytest.raises(TabPilotError) as info:
        yield info
    err = info.value
    assert err.code is code, f"expected {code.value}, got {err.code.value}"
    for key, value in expected_params.items():
        assert err.params.get(key) == value, f"param {key}: {err.params.get(key)!r} != {value!r}"
    assert "Traceback" not in err.user_message
    assert not _PATH.search(err.user_message), "user message contains a file-system path"

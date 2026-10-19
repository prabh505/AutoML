"""P02 error and warning catalog tests (PRD §8.9, §9.3; A-008; PI-09, PI-10)."""

import re

import pytest

from tabpilot.schemas.errors import (
    ERROR_TEMPLATES,
    ErrorCode,
    TabPilotError,
    WarningCode,
    render_message,
    template_params,
)

PRD_E_CODES = {
    "E-FILE-TOO-LARGE", "E-FILE-TYPE", "E-FILE-EMPTY", "E-FILE-CORRUPT", "E-ARCHIVE-UNSUPPORTED",
    "E-ENCODING", "E-DELIMITER-AMBIGUOUS", "E-PARSE-MALFORMED", "E-PARQUET-NESTED",
    "E-TOO-MANY-COLUMNS", "E-TOO-MANY-ROWS", "E-TARGET-CONSTANT", "E-TARGET-ALL-MISSING",
    "E-TARGET-TOO-FEW-ROWS", "E-TARGET-CLASS-TOO-SMALL", "E-TARGET-TOO-MANY-CLASSES",
    "E-TARGET-UNSUPPORTED-TYPE", "E-RESOURCE-INFEASIBLE", "E-NO-VALID-MODEL", "E-EXPORT-PARITY",
    "E-WORKER-LOST",
}  # fmt: skip
PRD_W_CODES = {
    "W-SMALL-DATA", "W-IMBALANCE", "W-HIGH-MISSING", "W-LEAK-STRONG",
    "W-LEAK-NEAR-DETERMINISTIC", "W-GROUP-SUSPECTED", "W-TEMPORAL", "W-DUPLICATES",
    "W-CONFLICTING-DUPLICATES", "W-WIDE", "W-ENCODING-FALLBACK", "W-COERCED", "W-INF",
    "W-TARGET-MISSING", "W-NO-HOLDOUT", "W-NEAR-CONSTANT", "W-HIGH-CARDINALITY",
}  # fmt: skip


@pytest.mark.unit
def test_every_prd_code_is_catalogued() -> None:
    assert {c.value for c in ErrorCode} == PRD_E_CODES
    assert {w.value for w in WarningCode} == PRD_W_CODES
    assert set(ERROR_TEMPLATES) == set(ErrorCode)


@pytest.mark.unit
@pytest.mark.parametrize("code", list(ErrorCode))
def test_every_message_renders_without_paths_or_traces(code: ErrorCode) -> None:
    params = {name: f"<{name}>" for name in template_params(code)}
    message = TabPilotError(code, **params).user_message
    assert message
    assert "Traceback" not in message
    assert not re.search(r"(/home/|/tmp/|[A-Z]:\\)", message)


@pytest.mark.unit
def test_prd_template_text_is_preserved() -> None:
    msg = render_message(ErrorCode.FILE_TOO_LARGE, {"size": 25, "limit": 20, "local_limit": 50})
    assert msg.startswith("The file is 25 MB; this demo accepts up to 20 MB.")
    assert "50 MB" in msg


@pytest.mark.unit
def test_resource_message_names_local_command() -> None:
    """US-F4: estimated memory, the limit, and the local command."""
    msg = render_message(ErrorCode.RESOURCE_INFEASIBLE, {"est": 900, "limit": 600})
    assert "900 MB" in msg
    assert "600 MB" in msg
    assert "TABPILOT_PROFILE=local" in msg


@pytest.mark.unit
def test_params_must_match_template() -> None:
    with pytest.raises(ValueError, match="missing"):
        TabPilotError(ErrorCode.FILE_TYPE)
    with pytest.raises(ValueError, match="unknown"):
        TabPilotError(ErrorCode.FILE_EMPTY, path="/etc/passwd")


@pytest.mark.unit
def test_error_carries_code_and_params() -> None:
    err = TabPilotError(ErrorCode.TARGET_CONSTANT, target="churn")
    assert err.code is ErrorCode.TARGET_CONSTANT
    assert err.params == {"target": "churn"}
    assert str(err).startswith("E-TARGET-CONSTANT: ")

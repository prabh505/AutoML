"""P02 logging tests: JSON format, key scrubbing, data refusal (PRD §30.2–§30.3; I-10, I-11)."""

import io
import json
import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from tabpilot import logging as tp_logging
from tabpilot.logging import (
    REDACTED_DATA,
    REDACTED_KEY,
    LoggingPolicyError,
    bind_run,
    configure_logging,
    data_value,
    get_logger,
    scrub,
)


@pytest.fixture
def stream() -> Iterator[io.StringIO]:
    buf = io.StringIO()
    handler = configure_logging("DEBUG", stream=buf)
    yield buf
    logging.getLogger("tabpilot").removeHandler(handler)


def records(buf: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in buf.getvalue().splitlines()]


@pytest.mark.unit
@pytest.mark.parametrize(
    "secret",
    ["sk-abcdefgh12345678", "sk-or-v1-abcdef0123456789", "gsk_ABCDEFGH12345678",
     "AIzaSyA1234567890abcdefghijklmn"],
)  # fmt: skip
def test_scrubber_redacts_key_patterns(secret: str) -> None:
    assert scrub(f"key={secret} end") == f"key={REDACTED_KEY} end"


@pytest.mark.unit
def test_json_record_with_stage_and_run(stream: io.StringIO) -> None:
    with bind_run("01JAB3K8X2V7Q9M4T6R1Z0YH5C"):
        get_logger("train_candidates").info("candidate_completed", mean=0.847, folds=5)
    get_logger("ingest").info("outside")
    first, second = records(stream)
    assert first["stage"] == "train_candidates"
    assert first["run_id"] == "01JAB3K8X2V7Q9M4T6R1Z0YH5C"
    assert first["event"] == "candidate_completed"
    assert (first["mean"], first["folds"]) == (0.847, 5)
    assert {"ts", "level", "logger"} <= first.keys()
    assert second["run_id"] is None


@pytest.mark.unit
def test_secrets_never_reach_output(stream: io.StringIO) -> None:
    log = get_logger("llm")
    log.warning("call failed gsk_ABCDEFGH12345678", detail={"auth": "Bearer sk-abcdefgh12345678"})
    try:
        raise RuntimeError("bad key AIzaSyA1234567890abcdefghijklmn")
    except RuntimeError:
        log.error("provider_error", exc_info=True)
    out = stream.getvalue()
    for secret in (
        "gsk_ABCDEFGH12345678",
        "sk-abcdefgh12345678",
        "AIzaSyA1234567890abcdefghijklmn",
    ):
        assert secret not in out
    err = records(stream)[1]
    assert err["error_class"] == "RuntimeError"
    assert "traceback" not in err


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [pd.DataFrame({"a": [1]}), pd.Series([1, 2]), pd.Index(["x"]), np.array([1.0]),
     data_value("customer 42")],
)  # fmt: skip
def test_dataset_content_raises_in_tests(value: object, stream: io.StringIO) -> None:
    with pytest.raises(LoggingPolicyError, match="dataset content"):
        get_logger("profile").info("stats", payload={"nested": [value]})
    assert stream.getvalue() == ""


@pytest.mark.unit
def test_dataset_content_redacted_outside_strict_mode(stream: io.StringIO) -> None:
    tp_logging.set_strict(False)
    get_logger("profile").info("stats", frame=pd.DataFrame({"secret_value": [123456]}))
    record = records(stream)[0]
    assert record["frame"] == REDACTED_DATA
    assert "123456" not in stream.getvalue()


@pytest.mark.unit
def test_data_value_never_formats_its_content() -> None:
    tagged = data_value("Jane Doe")
    assert "Jane" not in repr(tagged)
    assert "Jane" not in str(tagged)


@pytest.mark.unit
def test_unknown_objects_logged_by_type_only(stream: io.StringIO) -> None:
    class Opaque:
        def __repr__(self) -> str:
            return "SECRET-CONTENT"

    get_logger("export").info("obj", thing=Opaque())
    assert records(stream)[0]["thing"] == "<Opaque>"


@pytest.mark.unit
def test_tracebacks_only_in_debug_mode() -> None:
    buf = io.StringIO()
    handler = configure_logging("INFO", stream=buf, include_tracebacks=True)
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("run").error("failed", exc_info=True)
    logging.getLogger("tabpilot").removeHandler(handler)
    assert "Traceback" in records(buf)[0]["traceback"]


@pytest.mark.unit
def test_level_filtering(stream: io.StringIO) -> None:
    configure_logging("WARNING", stream=stream)
    get_logger("x").info("hidden")
    get_logger("x").debug("hidden")
    assert stream.getvalue() == ""

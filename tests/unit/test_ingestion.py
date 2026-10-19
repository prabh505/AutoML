"""P04 ingestion tests: T-ING-01…13, T-SEC-04 (filename part), PI-38 E-code tests."""

from __future__ import annotations

import io
import json
import logging
import re
import string
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.helpers.errors import raises_error

from tabpilot.config import Settings, load_settings
from tabpilot.ingestion import accept_upload, load_dataset, normalize_columns, normalize_name
from tabpilot.ingestion import loader as loader_module
from tabpilot.ingestion.inference import infer_column
from tabpilot.ingestion.upload import BYTES_PER_MB, sanitize_display_name
from tabpilot.schemas import ColumnRole, ErrorCode, SemanticType, TabPilotError

RID = "01JAB3K8X2V7Q9M4T6R1Z0YH5C"
FIX = Path(__file__).resolve().parents[2] / "fixtures"
MANIFEST: dict[str, Any] = yaml.safe_load((FIX / "MANIFEST.yaml").read_text(encoding="utf-8"))[
    "files"
]


@pytest.fixture
def hosted(tmp_path: Path) -> Settings:
    return load_settings({"TABPILOT_PROFILE": "hosted", "TABPILOT_DATA_DIR": str(tmp_path)})


@pytest.fixture
def local(tmp_path: Path) -> Settings:
    return load_settings({"TABPILOT_PROFILE": "local", "TABPILOT_DATA_DIR": str(tmp_path)})


def load(rel: str, s: Settings, **kw: Any) -> loader_module.LoadedDataset:
    return load_dataset(FIX / rel, Path(rel).name, s, RID, **kw)


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------- T-ING-01 / 02
@pytest.mark.unit
def test_T_ING_01_size_limit_checked_before_parsing(
    hosted: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 25 MB file on hosted is rejected before parsing; the parser is never invoked."""
    path = tmp_path / "big.csv"
    with path.open("wb") as fh:
        fh.truncate(25 * BYTES_PER_MB)
    calls: list[str] = []
    for name in ("_read_text", "_read_parquet"):
        monkeypatch.setattr(loader_module, name, lambda *a, n=name, **k: calls.append(n))
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: calls.append("read_csv"))
    with raises_error(ErrorCode.FILE_TOO_LARGE, size=25.0, limit=20, local_limit=50) as info:
        load_dataset(path, "big.csv", hosted, RID)
    assert calls == []
    assert "20 MB" in info.value.user_message

    class NoRead(io.BytesIO):
        def read(self, *a: Any) -> bytes:
            raise AssertionError("content read before the size check")

    stream = NoRead(b"\0" * (25 * BYTES_PER_MB))
    with raises_error(ErrorCode.FILE_TOO_LARGE):
        accept_upload(stream, "big.csv", hosted)


@pytest.mark.unit
@pytest.mark.parametrize(("rel", "code"), [
    ("malicious/zip_renamed.csv", ErrorCode.ARCHIVE_UNSUPPORTED),
    ("malicious/ole_renamed.csv", ErrorCode.FILE_CORRUPT),
])  # fmt: skip
def test_T_ING_02_extension_magic_mismatch(rel: str, code: ErrorCode, hosted: Settings) -> None:
    data = (FIX / rel).read_bytes()
    with raises_error(code):
        accept_upload(io.BytesIO(data), "data.csv", hosted)
    with raises_error(code):
        load(rel, hosted)
    assert not list((hosted.data_dir).rglob("raw.csv"))  # nothing stored


# ---------------------------------------------------------------- T-ING-03…07 (parsing)
@pytest.mark.unit
@pytest.mark.parametrize(("rel", "encoding", "fallback"), [
    ("parsing/bom_utf8.csv", "utf-8-sig", False), ("parsing/utf16.csv", "utf-16", False),
    ("parsing/cp1252.csv", None, None), ("parsing/latin1_fallback.csv", "latin-1", True),
    ("datasets/binary_small.csv", "utf-8", False),
])  # fmt: skip
def test_T_ING_03_encoding(
    rel: str, encoding: str | None, fallback: bool | None, local: Settings
) -> None:
    ld = load(rel, local)
    shape = MANIFEST[rel]["shape"]
    assert list(ld.frame.shape) == shape
    if encoding is not None:
        assert ld.parse_report.encoding == encoding
        assert ld.parse_report.encoding_fallback is fallback
    if rel.endswith("cp1252.csv"):  # PI-43: cp1252 or latin-1 fallback, text must be right
        assert ld.parse_report.encoding in {"cp1252", "latin-1"}
        assert ld.frame["city"].iloc[0] == "Zürich café 1"
    if ld.parse_report.encoding_fallback:
        assert any(w.startswith("W-ENCODING-FALLBACK:lines=") for w in ld.metadata.warnings)


@pytest.mark.unit
def test_encoding_failure_after_detection(local: Settings, tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_bytes(b"a,b\n1,\xff\xfe\n")
    with raises_error(ErrorCode.ENCODING):
        load_dataset(path, "bad.csv", local, RID, encoding_override="utf-8")


@pytest.mark.unit
@pytest.mark.parametrize(("rel", "delimiter"), [
    ("parsing/tab_separated.tsv", "\t"), ("parsing/pipe_delimited.txt", "|"),
    ("parsing/semicolon_decimal_comma.csv", ";"), ("datasets/wide.csv", ","),
])  # fmt: skip
def test_T_ING_04_delimiter(rel: str, delimiter: str, local: Settings) -> None:
    assert load(rel, local).parse_report.delimiter == delimiter


@pytest.mark.unit
def test_T_ING_04_ambiguous_delimiter_and_user_choice(local: Settings) -> None:
    with raises_error(ErrorCode.DELIMITER_AMBIGUOUS):
        load("parsing/ambiguous_delimiter.csv", local)
    chosen = load("parsing/ambiguous_delimiter.csv", local, delimiter_override=";")
    assert list(chosen.frame.columns) == ["a_b", "c"]


@pytest.mark.unit
def test_T_ING_05_decimal_comma(local: Settings) -> None:
    ld = load("parsing/semicolon_decimal_comma.csv", local)
    assert ld.parse_report.decimal_comma_columns == ["price"]
    np.testing.assert_allclose(
        ld.frame["price"].iloc[:3].astype(float), [1.15, 2.25, 3.35], rtol=1e-6
    )
    assert ld.metadata.column_map["price"] == "price"


@pytest.mark.unit
def test_T_ING_06_malformed_lines(local: Settings) -> None:
    ld = load("parsing/malformed_few.csv", local)
    assert ld.parse_report.malformed_lines == 2
    assert ld.parse_report.malformed_line_numbers == [51, 121]
    assert ld.metadata.source.malformed_lines_skipped == 2
    assert len(ld.frame) == 298
    assert 50 not in set(ld.frame["a"])
    assert 120 not in set(ld.frame["a"])
    with raises_error(ErrorCode.PARSE_MALFORMED, n=30, pct=10.0, line=11):
        load("parsing/malformed_many.csv", local)


@pytest.mark.unit
def test_T_ING_07_quoted_newlines(local: Settings) -> None:
    ld = load("parsing/quoted_newlines.csv", local)
    assert ld.frame.shape == (20, 3)
    assert ld.frame["comment"].iloc[0] == "line one 1\nline two, with comma"
    assert ld.parse_report.malformed_lines == 0


# ---------------------------------------------------------------- T-ING-08 normalization
@pytest.mark.unit
def test_T_ING_08_normalization_map() -> None:
    names = ["Customer ID", "TotalCharges", "  total charges ", "1st value", "class", "",
             "Target", "a.b-c", "Café ☕", "x" * 80, "customer_id"]  # fmt: skip
    assert normalize_columns(names) == [
        "customer_id", "totalcharges", "total_charges", "c_1st_value", "class_", "col_5",
        "target_", "a_b_c", "caf", "x" * 64, "customer_id__2",
    ]  # fmt: skip
    once = [normalize_name(n, i) for i, n in enumerate(names)]
    assert [normalize_name(n, i) for i, n in enumerate(once)] == once
    assert normalize_columns(["Amount", "Amount", "amount"]) == ["amount", "amount__2", "amount__3"]


@pytest.mark.unit
@settings(max_examples=300, deadline=None)
@given(st.lists(st.text(alphabet=string.printable + "éü☕ß \t", max_size=90), max_size=30))
def test_T_ING_08_normalization_properties(names: list[str]) -> None:
    out = normalize_columns(names)
    assert len(out) == len(names)
    assert len(set(out)) == len(out)
    for i, (raw, norm) in enumerate(zip(names, out, strict=True)):
        assert re.fullmatch(r"[a-z_][a-z0-9_]*", norm)
        assert not norm[0].isdigit()
        single = normalize_name(raw, i)
        assert normalize_name(single, i) == single  # idempotent
        assert len(single) <= 65


# ---------------------------------------------------------------- T-ING-09…13 typing
def _expected_type(text: str) -> tuple[str, bool]:
    return text.split(" ", maxsplit=1)[0], "possibly_categorical" in text


DATASETS = sorted(k for k, v in MANIFEST.items() if "expected_semantic_types" in v)


@pytest.mark.unit
@pytest.mark.parametrize("rel", DATASETS)
def test_T_ING_09_semantic_types_match_manifest(rel: str, local: Settings) -> None:
    columns = {c.name: c for c in load(rel, local).metadata.columns}
    for name, text in MANIFEST[rel]["expected_semantic_types"].items():
        semantic, hint = _expected_type(text)
        assert columns[name].semantic_type.value == semantic, name
        assert ("possibly_categorical" in columns[name].hints) == hint, name


@pytest.mark.unit
def test_T_ING_10_coercion_and_inf_counts(local: Settings) -> None:
    ld = load("datasets/quality_issues.csv", local)
    cols = {c.name: c for c in ld.metadata.columns}
    expected = MANIFEST["datasets/quality_issues.csv"]["expected_counts"]
    assert cols["mixed_type"].coerced_to_missing == expected["mixed_type_coerced"]
    assert cols["inf_values"].coerced_to_missing == expected["inf_values_converted"]
    assert "W-COERCED:mixed_type:20" in ld.metadata.warnings
    assert "W-INF:inf_values:5" in ld.metadata.warnings
    assert not np.isinf(ld.frame["inf_values"].astype(float)).any()
    assert f"W-DUPLICATES:{expected['duplicate_rows']}" in ld.metadata.warnings


@pytest.mark.unit
def test_T_ING_11_identifier_rules() -> None:
    n = 400
    rng = np.random.default_rng(0)
    # Name alone never excludes: an ID-named column without unique values stays a feature.
    assert (
        infer_column("store_id", pd.Series(rng.choice(["a", "b", "c"], n))).semantic_type
        == SemanticType.CATEGORICAL
    )
    # Unique key-like strings are identifiers even without an ID name.
    keys = pd.Series([f"K{i:06d}" for i in rng.permutation(n)])
    assert infer_column("token", keys).semantic_type == SemanticType.IDENTIFIER
    # Unique strings with spaces and an unrelated name are not identifiers.
    phrases = pd.Series([f"item {i} {'x' * (i % 7)}" for i in range(n)])
    assert infer_column("label", phrases).semantic_type == SemanticType.CATEGORICAL
    # PI-41: unique monotonic integers → identifier; unique shuffled integers without ID name → numeric.
    assert (
        infer_column("seq", pd.Series([str(i) for i in range(n)])).semantic_type
        == SemanticType.IDENTIFIER
    )
    shuffled = pd.Series([str(i) for i in rng.permutation(n)])
    assert infer_column("amount", shuffled).semantic_type == SemanticType.NUMERIC
    assert infer_column("row_no", shuffled).role == ColumnRole.EXCLUDED


@pytest.mark.unit
def test_T_ING_12_datetime_hints_and_downcasting(local: Settings, tmp_path: Path) -> None:
    single = pd.Series([f"2024-01-{i % 28 + 1:02d}" for i in range(100)])
    assert infer_column("d", single).semantic_type == SemanticType.DATETIME
    mixed = pd.Series(["2024-01-05"] * 50 + ["05/01/2024"] * 50)  # no single format ≥ 95%
    assert infer_column("d", mixed).semantic_type == SemanticType.CATEGORICAL
    assert infer_column("b", pd.Series(["Y", "N"] * 20)).semantic_type == SemanticType.BOOLEAN
    assert infer_column("z", pd.Series([str(i % 12) for i in range(100)])).hints == [
        "possibly_categorical"
    ]
    path = write(tmp_path, "d.csv", "small,big,ints,cat,y\n" + "\n".join(
        f"{i * 0.5},{12345678.5 + i},{(i * 7) % 40},c{i % 3},{i % 2}" for i in range(60)) + "\n")  # fmt: skip
    dtypes = {c.name: c.dtype for c in load_dataset(path, "d.csv", local, RID).metadata.columns}
    assert dtypes == {"small": "float32", "big": "float64", "ints": "int8", "cat": "category",
                      "y": "category"}  # fmt: skip


@pytest.mark.unit
def test_T_ING_13_no_silent_discard(local: Settings) -> None:
    ld = load("datasets/quality_issues.csv", local)
    meta = ld.metadata
    assert meta.n_columns == len(meta.columns) == ld.frame.shape[1] == 15
    excluded = {c.name: c.exclusion_reason for c in meta.columns if c.role == ColumnRole.EXCLUDED}
    expected = MANIFEST["datasets/quality_issues.csv"]["expected_exclusions"]
    assert set(excluded) == set(expected)
    assert all(excluded.values())
    assert meta.n_rows_raw == meta.n_rows_valid == len(ld.frame) == 1000
    for col in ("missing_60", "near_constant", "very_high_card"):
        assert any(w.endswith(col) for w in meta.warnings), col
    assert "high_cardinality" in {c.name: c for c in meta.columns}["high_card_cat"].hints


# ---------------------------------------------------------------- PI-38 E-code tests
@pytest.mark.unit
def test_E_FILE_TYPE(hosted: Settings) -> None:
    with raises_error(ErrorCode.FILE_TYPE, ext=".xlsx"):
        accept_upload(io.BytesIO(b"a,b\n"), "sheet.xlsx", hosted)
    with raises_error(ErrorCode.ARCHIVE_UNSUPPORTED):
        accept_upload(io.BytesIO(b"x"), "data.csv.gz", hosted)


@pytest.mark.unit
@pytest.mark.parametrize("rel", ["parsing/empty.csv", "parsing/header_only.csv"])
def test_E_FILE_EMPTY(rel: str, local: Settings) -> None:
    with raises_error(ErrorCode.FILE_EMPTY):
        load(rel, local)


@pytest.mark.unit
@pytest.mark.parametrize("rel", ["malicious/corrupt.parquet", "malicious/csv_named.parquet"])
def test_E_FILE_CORRUPT(rel: str, local: Settings) -> None:
    with raises_error(ErrorCode.FILE_CORRUPT, format="Parquet"):
        load(rel, local)


@pytest.mark.unit
def test_E_PARQUET_NESTED(local: Settings) -> None:
    with raises_error(ErrorCode.PARQUET_NESTED, cols="attrs, tags"):
        load("malicious/nested.parquet", local)
    flat = load("parsing/flat.parquet", local)
    assert flat.frame.shape == (180, 6)
    assert {c.name: c.semantic_type for c in flat.metadata.columns}[
        "level"
    ] == SemanticType.CATEGORICAL


@pytest.mark.unit
def test_E_TOO_MANY_COLUMNS(hosted: Settings, tmp_path: Path) -> None:
    path = write(
        tmp_path, "w.csv", ",".join(f"c{i}" for i in range(201)) + "\n" + ",".join("1" * 201) + "\n"
    )
    with raises_error(
        ErrorCode.TOO_MANY_COLUMNS, count=201, profile="hosted", limit=200, local_limit=500
    ):
        load_dataset(path, "w.csv", hosted, RID)


@pytest.mark.unit
def test_E_TOO_MANY_ROWS(hosted: Settings, tmp_path: Path) -> None:
    limits = hosted.limits.model_copy(update={"max_rows": 100})
    small = hosted.model_copy(update={"limits": limits})
    with raises_error(
        ErrorCode.TOO_MANY_ROWS, count=180, profile="hosted", limit=100, local_limit=500_000
    ):
        load("datasets/binary_small.csv", small)


@pytest.mark.unit
def test_single_column_file_is_ambiguous(local: Settings, tmp_path: Path) -> None:
    path = write(tmp_path, "one.csv", "value\n" + "\n".join(map(str, range(30))) + "\n")
    with raises_error(ErrorCode.DELIMITER_AMBIGUOUS):
        load_dataset(path, "one.csv", local, RID)


# ---------------------------------------------------------------- T-SEC-04 (filename part)
@pytest.mark.unit
def test_T_SEC_04_malicious_filenames(hosted: Settings) -> None:
    names = json.loads((FIX / "malicious" / "upload_names.json").read_text(encoding="utf-8"))
    for raw in names.values():
        shown = sanitize_display_name(raw)
        assert "/" not in shown
        assert "\\" not in shown
        assert all(ch.isprintable() for ch in shown)
        assert len(shown) <= 120
    stored = accept_upload(io.BytesIO(b"a,b\n1,2\n"), names["path_traversal"], hosted)
    assert stored.path == hosted.data_dir / "runs" / stored.run_id / "input" / "raw.csv"
    assert stored.display_name == "passwd.csv"
    long = accept_upload(io.BytesIO(b"a,b\n1,2\n"), names["long_300_chars"], hosted)
    assert len(long.display_name) == 120
    assert long.display_name.endswith(".csv")
    assert long.path.name == "raw.csv"


@pytest.mark.unit
def test_upload_stores_under_run_dir_with_private_mode(hosted: Settings) -> None:
    data = (FIX / "datasets" / "binary_small.csv").read_bytes()
    stored = accept_upload(io.BytesIO(data), "binary_small.csv", hosted)
    assert stored.path.read_bytes() == data
    assert (stored.path.parents[1].stat().st_mode & 0o777) == 0o700
    assert stored.format == "csv"
    assert stored.size_bytes == len(data)


@pytest.mark.unit
def test_non_seekable_upload_is_capped(hosted: Settings) -> None:
    class Stream(io.RawIOBase):
        def __init__(self, data: bytes) -> None:
            self._buf = io.BytesIO(data)

        def readable(self) -> bool:
            return True

        def seekable(self) -> bool:
            return False

        def readinto(self, b: Any) -> int:
            chunk = self._buf.read(len(b))
            b[: len(chunk)] = chunk
            return len(chunk)

    ok = accept_upload(io.BufferedReader(Stream(b"a,b\n1,2\n")), "x.csv", hosted)  # type: ignore[arg-type]
    assert ok.size_bytes == 8
    with raises_error(ErrorCode.FILE_TOO_LARGE):
        accept_upload(io.BufferedReader(Stream(b"0" * (21 * BYTES_PER_MB))), "x.csv", hosted)  # type: ignore[arg-type]


@pytest.mark.unit
def test_logs_contain_no_cell_values(local: Settings, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="tabpilot"):
        load("datasets/binary_basic.csv", local)
    text = " ".join(str(r.__dict__.get("tp_fields")) + r.getMessage() for r in caplog.records)
    for value in ("C1", "region_", "month_to_month"):
        assert value not in text


@pytest.mark.unit
def test_typed_fixture_values_never_raise_TabPilotError_on_valid_inputs(local: Settings) -> None:
    for rel in DATASETS:
        try:
            load(rel, local)
        except TabPilotError as exc:  # pragma: no cover - assertion message only
            pytest.fail(f"{rel}: {exc.code}")

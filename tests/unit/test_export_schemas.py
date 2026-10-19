"""P02 schema export tests: every contract exported; drift and missing version bumps caught."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tabpilot.schemas import CONTRACTS, UNION_CONTRACTS

ROOT = Path(__file__).resolve().parents[2]


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "export_schemas", ROOT / "scripts" / "export_schemas.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_every_contract_exported_with_version(tmp_path: Path) -> None:
    script = load_script()
    script.write(tmp_path)
    files = {p.stem for p in tmp_path.glob("*.json")}
    assert files == set(CONTRACTS) | set(UNION_CONTRACTS)
    plan = json.loads((tmp_path / "PlannerOutput.json").read_text(encoding="utf-8"))
    assert plan["x-schema-version"] == "plan/1.0"
    assert (
        json.loads((tmp_path / "FeatureSpec.json").read_text())["x-schema-version"]
        == "feature_spec/1.0"
    )


@pytest.mark.unit
def test_committed_schemas_are_current() -> None:
    assert load_script().check(ROOT / "docs" / "schemas", base=None) == []


@pytest.mark.unit
def test_stale_docs_detected(tmp_path: Path) -> None:
    script = load_script()
    script.write(tmp_path)
    (tmp_path / "RunEvent.json").write_text("{}\n", encoding="utf-8")
    assert script.check(tmp_path, base=None) == [
        "RunEvent.json: docs/schemas is out of date; run `make schemas`"
    ]


@pytest.mark.unit
def test_changed_contract_without_version_bump_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    generated = script.generate()
    old = json.loads(generated["RunEvent.json"])
    old["properties"]["seq"]["minimum"] = 5  # pretend the base revision differed
    monkeypatch.setattr(
        script, "_at_ref", lambda ref, name: json.dumps(old) if name == "RunEvent.json" else None
    )
    script.write(tmp_path)
    assert script.check(tmp_path, base="BASE") == [
        "RunEvent.json: contract changed but schema_version event/1.0 not bumped"
    ]
    old["x-schema-version"] = "event/0.9"
    assert script.check(tmp_path, base="BASE") == []


@pytest.mark.unit
def test_cli_check_exit_code() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/export_schemas.py", "--check"], cwd=ROOT, check=False,
        capture_output=True, text=True,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr

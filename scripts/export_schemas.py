"""Export a JSON Schema for every TabPilot contract to docs/schemas/ (PRD §26).

Usage:
    python scripts/export_schemas.py                    # (re)write docs/schemas/*.json
    python scripts/export_schemas.py --check [--base REF]

`--check` fails when:
  1. the committed files differ from freshly generated schemas (run `make schemas`), or
  2. with `--base REF`: a schema differs from its version at git REF but its schema_version
     did not change (a changed contract needs a version bump).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tabpilot.schemas import CONTRACTS, UNION_CONTRACTS

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "schemas"


def _version_of(schema: dict[str, Any]) -> str | None:
    """Return the schema_version constant of a model schema or of a union's first member."""
    props = schema.get("properties", {})
    if "schema_version" in props:
        return str(props["schema_version"].get("default"))
    for definition in schema.get("$defs", {}).values():
        version = definition.get("properties", {}).get("schema_version", {}).get("default")
        if version:
            return str(version)
    return None


def generate() -> dict[str, str]:
    """Return {file name: canonical JSON text} for every contract."""
    schemas: dict[str, dict[str, Any]] = {
        name: model.model_json_schema() for name, model in CONTRACTS.items()
    }
    schemas.update({name: adapter.json_schema() for name, adapter in UNION_CONTRACTS.items()})
    out = {}
    for name, schema in sorted(schemas.items()):
        schema["x-schema-version"] = _version_of(schema)
        out[f"{name}.json"] = (
            json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
    return out


def write(out_dir: Path) -> None:
    """Write all schemas to `out_dir`, removing stale files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = generate()
    for stale in {p.name for p in out_dir.glob("*.json")} - set(generated):
        (out_dir / stale).unlink()
    for name, text in generated.items():
        (out_dir / name).write_text(text, encoding="utf-8")


def _at_ref(ref: str, name: str) -> str | None:
    rel = (OUT_DIR / name).relative_to(ROOT).as_posix()
    result = subprocess.run(  # noqa: S603 - fixed git invocation, no shell
        ["git", "show", f"{ref}:{rel}"],  # noqa: S607
        cwd=ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    return result.stdout if result.returncode == 0 else None


def check(out_dir: Path, base: str | None) -> list[str]:
    """Return a list of problems; empty means the committed schemas are current and versioned."""
    problems = []
    generated = generate()
    committed = {p.name: p.read_text(encoding="utf-8") for p in out_dir.glob("*.json")}
    for name in sorted(set(generated) | set(committed)):
        if generated.get(name) != committed.get(name):
            problems.append(f"{name}: docs/schemas is out of date; run `make schemas`")
    if base:
        for name, text in generated.items():
            old = _at_ref(base, name)
            new_version = json.loads(text).get("x-schema-version")
            # Components without their own version are versioned through their parent contract.
            if old is not None and old != text and new_version is not None:
                old_version = json.loads(old).get("x-schema-version")
                if old_version == new_version:
                    problems.append(
                        f"{name}: contract changed but schema_version {old_version} not bumped"
                    )
    return problems


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--base", default=None, help="git ref to compare versions against")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    if not args.check:
        write(args.out)
        return 0
    problems = check(args.out, args.base)
    for problem in problems:
        sys.stderr.write(problem + "\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

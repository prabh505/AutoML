#!/usr/bin/env bash
# T-ART-03: unzip a bundle into an empty directory, install its pinned requirements into a fresh
# Python 3.11 virtual environment (no tabpilot), and run `pipeline.py verify` there.
# Usage: scripts/verify_bundle_clean_venv.sh <bundle.zip> [python3.11]
set -euo pipefail
bundle="$(realpath "$1")"
python="${2:-python3.11}"
work="$(mktemp -d)"
trap 'rm -rf -- "${work:?}"' EXIT
mkdir "$work/bundle"
(cd "$work/bundle" && unzip -q "$bundle")
"$python" -m venv "$work/venv"
"$work/venv/bin/python" -m pip install --quiet --upgrade pip
"$work/venv/bin/python" -m pip install --quiet -r "$work/bundle/requirements.txt"
if "$work/venv/bin/python" -c "import tabpilot" 2>/dev/null; then
  echo "tabpilot is importable in the clean venv" >&2
  exit 1
fi
cd "$work/bundle"
"$work/venv/bin/python" pipeline.py verify

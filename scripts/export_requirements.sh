#!/usr/bin/env bash
# Export the hosted requirement set from uv.lock (PRD §34.2).
# Core dependencies only: no extras (xgboost, catboost, mlflow, flaml) and no dev tools.
set -euo pipefail
cd "$(dirname "$0")/.."
uv export --frozen --no-hashes --no-dev --no-emit-project --no-header --format requirements-txt \
  > requirements.txt.tmp
{
  echo "# Hosted requirements, exported from uv.lock by scripts/export_requirements.sh. Do not edit."
  cat requirements.txt.tmp
} > requirements.txt
rm requirements.txt.tmp

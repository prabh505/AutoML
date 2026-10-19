"""P01 smoke tests: every package imports; config files carry the PRD values."""

import importlib
import pkgutil
import socket
from pathlib import Path
from typing import Any

import pytest
import yaml

import tabpilot
import tabpilot_runtime

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"

# PRD §25.1 packages (sandbox omitted: D-09 / PI-35) plus tabpilot_runtime (A-001).
EXPECTED_PACKAGES = {
    "tabpilot.schemas",
    "tabpilot.ingestion",
    "tabpilot.splitting",
    "tabpilot.profiling",
    "tabpilot.planning",
    "tabpilot.llm",
    "tabpilot.llm.adapters",
    "tabpilot.features",
    "tabpilot.features.dsl",
    "tabpilot.preprocessing",
    "tabpilot.models",
    "tabpilot.training",
    "tabpilot.tuning",
    "tabpilot.evaluation",
    "tabpilot.reflection",
    "tabpilot.ensembling",
    "tabpilot.explain",
    "tabpilot.reporting",
    "tabpilot.export",
    "tabpilot.resources",
    "tabpilot.tracking",
    "tabpilot.orchestration",
    "tabpilot.runtime",
    "tabpilot.ui",
    "tabpilot.ui.pages",
    "tabpilot.ui.components",
}


def _load(name: str) -> dict[str, Any]:
    data = yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


@pytest.mark.unit
def test_every_package_imports_and_documents_its_boundary() -> None:
    found = {m.name for m in pkgutil.walk_packages(tabpilot.__path__, "tabpilot.") if m.ispkg}
    assert found == EXPECTED_PACKAGES
    for name in [*sorted(found), "tabpilot_runtime"]:
        module = importlib.import_module(name)
        assert module.__doc__, f"{name} lacks a responsibility docstring"
    assert "tabpilot" in (tabpilot_runtime.__doc__ or "")


@pytest.mark.unit
def test_sandbox_package_is_not_built() -> None:
    """D-09 / PI-35: the P2 sandbox is not part of any v1 build (full check: T-SEC-08, P17)."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tabpilot.sandbox")


@pytest.mark.unit
def test_hosted_profile_matches_prd() -> None:
    """PRD §29.1 hosted column; time budgets per amendment A-004."""
    hosted = _load("profiles.yaml")["profiles"]["hosted"]
    assert hosted == {
        "max_file_mb": 20,
        "max_rows": 100_000,
        "max_columns": 200,
        "rss_soft_mb": 600,
        "rss_hard_mb": 750,
        "n_jobs": 2,
        "time_budget_default_s": 960,
        "time_budget_max_s": 1800,
        "max_concurrent_runs": 1,
        "max_trials_per_run": 60,
        "llm_calls_per_run": 12,
        "max_bundle_mb": 100,
        "models_enabled": ["linear", "random_forest", "lightgbm"],
    }


@pytest.mark.unit
def test_local_profile_matches_prd() -> None:
    """PRD §29.1 local column; time budgets per amendment A-004."""
    local = _load("profiles.yaml")["profiles"]["local"]
    assert local["max_file_mb"] == 50
    assert local["max_rows"] == 500_000
    assert local["max_columns"] == 500
    assert (local["rss_soft_fraction_of_ram"], local["rss_hard_fraction_of_ram"]) == (0.70, 0.85)
    assert (local["n_jobs"], local["n_jobs_cap"]) == ("auto", 8)
    assert (local["time_budget_default_s"], local["time_budget_max_s"]) == (1200, 7200)
    assert (local["max_trials_per_run"], local["max_bundle_mb"]) == (300, 500)
    assert (local["llm_calls_per_run"], local["llm_calls_max"]) == (12, 25)


@pytest.mark.unit
def test_model_availability_matches_prd() -> None:
    """PRD §16.1: hosted = linear, RF, LightGBM; XGBoost/CatBoost local; hist_gb fallback."""
    models = _load("models.yaml")["models"]
    hosted = sorted(
        n for n, m in models.items() if "hosted" in m["profiles"] and not m.get("fallback_only")
    )
    assert hosted == ["lightgbm", "linear", "random_forest"]
    assert models["xgboost"]["profiles"] == ["local"]
    assert models["catboost"]["profiles"] == ["local"]
    assert models["hist_gb"]["fallback_only"] is True
    assert models["random_forest"]["hosted_max_depth_cap"] == 20
    assert models["lightgbm"]["defaults"] == {"deterministic": True, "force_row_wise": True}


@pytest.mark.unit
def test_providers_match_prd() -> None:
    """PRD §23.3: provider order, endpoints, env-var placeholders; no hardcoded model ids."""
    cfg = _load("providers.yaml")
    assert cfg["order_hosted"] == ["groq", "gemini", "openrouter"]
    assert cfg["order_local"] == ["ollama", "groq", "gemini", "openrouter"]
    p = cfg["providers"]
    assert p["groq"]["base_url"] == "https://api.groq.com/openai/v1"
    assert p["groq"]["model_id"] == "${TABPILOT_GROQ_MODEL}"
    assert p["groq"]["api_key_env"] == "GROQ_API_KEY"
    assert p["gemini"]["api_key_env"] == "GEMINI_API_KEY"
    assert p["openrouter"]["model_ids"] == ["${TABPILOT_OR_MODEL_1}", "${TABPILOT_OR_MODEL_2}"]
    assert p["ollama"]["base_url"] == "http://localhost:11434/v1"
    for provider in p.values():
        ids = provider.get("model_ids", [provider.get("model_id")])
        assert all(str(i).startswith("${") for i in ids)


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:A test tried to use socket")
def test_network_is_blocked_by_default() -> None:
    """CLAUDE.md §5: tests make no real network calls (pytest-socket)."""
    with pytest.raises(Exception, match="socket"):
        socket.create_connection(("pypi.org", 443), timeout=1)

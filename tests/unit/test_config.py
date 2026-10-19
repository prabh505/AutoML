"""P02 configuration tests (PRD §16.1, §23.3, §29.1; A-004)."""

import shutil
from pathlib import Path

import pytest

from tabpilot.config import DEFAULT_CONFIG_DIR, ConfigError, load_settings
from tabpilot.schemas import ExecutionProfileName, ModelName, ProviderName


@pytest.fixture
def config_copy(tmp_path: Path) -> Path:
    target = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, target)
    return target


@pytest.mark.unit
def test_hosted_profile_values() -> None:
    s = load_settings({"TABPILOT_PROFILE": "hosted"})
    assert s.profile is ExecutionProfileName.HOSTED
    lim = s.limits
    assert (lim.max_file_mb, lim.max_rows, lim.max_columns) == (20, 100_000, 200)
    assert lim.rss_limits_mb(physical_ram_mb=1024) == (600.0, 750.0)
    assert lim.resolved_n_jobs(physical_cores=32) == 2
    assert (lim.time_budget_default_s, lim.time_budget_max_s) == (960, 1800)
    assert (lim.max_trials_per_run, lim.llm_calls_per_run, lim.max_bundle_mb) == (60, 12, 100)
    assert s.enabled_models() == [ModelName.LINEAR, ModelName.RANDOM_FOREST, ModelName.LIGHTGBM]
    assert s.provider_order == [ProviderName.GROQ, ProviderName.GEMINI, ProviderName.OPENROUTER]


@pytest.mark.unit
def test_local_profile_is_default() -> None:
    s = load_settings({})
    assert s.profile is ExecutionProfileName.LOCAL
    lim = s.limits
    assert (lim.max_file_mb, lim.max_rows, lim.max_columns) == (50, 500_000, 500)
    assert lim.rss_limits_mb(physical_ram_mb=16_000) == pytest.approx((11_200.0, 13_600.0))
    assert lim.resolved_n_jobs(physical_cores=4) == 4
    assert lim.resolved_n_jobs(physical_cores=64) == 8
    assert (lim.time_budget_default_s, lim.time_budget_max_s) == (1200, 7200)
    assert ModelName.XGBOOST in s.enabled_models()
    assert ModelName.HIST_GB not in s.enabled_models()
    assert s.provider_order[0] is ProviderName.OLLAMA
    assert s.estimator.safety == 1.5


@pytest.mark.unit
def test_env_placeholders_interpolate_and_unset_become_none() -> None:
    s = load_settings({"TABPILOT_GROQ_MODEL": "model-a", "TABPILOT_GROQ_CTX": "8192",
                       "TABPILOT_OR_MODEL_2": "model-c"})  # fmt: skip
    groq = s.providers[ProviderName.GROQ]
    assert groq.configured_model_ids == ["model-a"]
    assert groq.max_context_tokens == 8192
    assert s.providers[ProviderName.OPENROUTER].configured_model_ids == ["model-c"]
    assert s.providers[ProviderName.GEMINI].configured_model_ids == []
    assert groq.api_key_env == "GROQ_API_KEY"


@pytest.mark.unit
def test_settings_never_hold_keys() -> None:
    """I-10: only the env-var name is stored, never the key value."""
    s = load_settings({"GROQ_API_KEY": "gsk_secretvalue123456"})
    assert "gsk_secretvalue123456" not in s.model_dump_json()


@pytest.mark.unit
def test_data_dir_from_env(tmp_path: Path) -> None:
    assert load_settings({"TABPILOT_DATA_DIR": str(tmp_path)}).data_dir == tmp_path


@pytest.mark.unit
def test_invalid_profile_name_fails_fast() -> None:
    with pytest.raises(ConfigError, match="TABPILOT_PROFILE='cloud'"):
        load_settings({"TABPILOT_PROFILE": "cloud"})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("file", "old", "new", "message"),
    [("profiles.yaml", "max_rows: 100000", "max_rows: -5",
      r"profiles\.yaml: profiles\.hosted\.max_rows"),
     ("profiles.yaml", "rss_soft_mb: 600", "rss_soft_mb: 800", "rss_soft_mb must be below"),
     ("models.yaml", "library: lightgbm", "library: lightgbm\n    surprise: 1", "models.yaml"),
     ("providers.yaml", "order_hosted: [groq,", "order_hosted: [nobody,", "providers.yaml"),
     ("providers.yaml", '"${TABPILOT_GROQ_MODEL}"', '"prefix-${TABPILOT_GROQ_MODEL}"',
      "placeholders"),
     ("profiles.yaml", "models_enabled: [linear, random_forest, lightgbm]",
      "models_enabled: [linear]", "disagrees with models.yaml"),
     ("models.yaml", "models:", "models: [", "invalid YAML")],
)  # fmt: skip
def test_malformed_config_fails_with_clear_message(
    config_copy: Path, file: str, old: str, new: str, message: str
) -> None:
    path = config_copy / file
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_settings({"TABPILOT_PROFILE": "hosted"}, config_dir=config_copy)


@pytest.mark.unit
def test_missing_config_file(config_copy: Path) -> None:
    (config_copy / "models.yaml").unlink()
    with pytest.raises(ConfigError, match=r"models\.yaml: file not found"):
        load_settings({}, config_dir=config_copy)

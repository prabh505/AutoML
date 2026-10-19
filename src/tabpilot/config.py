"""Settings loading and validation (PRD §16.1, §23.3, §29.1; A-004).

Loads `config/profiles.yaml`, `config/models.yaml`, and `config/providers.yaml`, replaces quoted
`"${ENV_VAR}"` placeholders from the environment, validates everything with Pydantic, and selects
the execution profile from `TABPILOT_PROFILE` (default `local`). Missing optional provider
variables become `None`. Malformed configuration fails fast with a `ConfigError` that names the
file and the problem.

Secrets are never loaded here. Provider entries hold only the *name* of the environment variable
that contains the key (I-10).
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tabpilot.schemas import ExecutionProfileName, ModelName, ProviderName, Task

PLACEHOLDER = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_M = TypeVar("_M", bound=BaseModel)


class ConfigError(ValueError):
    """Configuration is missing or invalid. The message names the file and the problem."""


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class ProfileLimits(_Config):
    """One execution profile's limits (PRD §29.1)."""

    max_file_mb: int = Field(gt=0)
    max_rows: int = Field(gt=0)
    max_columns: int = Field(gt=0)
    rss_soft_mb: int | None = Field(default=None, gt=0)
    rss_hard_mb: int | None = Field(default=None, gt=0)
    rss_soft_fraction_of_ram: float | None = Field(default=None, gt=0, lt=1)
    rss_hard_fraction_of_ram: float | None = Field(default=None, gt=0, lt=1)
    n_jobs: int | Literal["auto"]
    n_jobs_cap: int | None = Field(default=None, gt=0)
    time_budget_default_s: int = Field(gt=0)
    time_budget_max_s: int = Field(gt=0)
    max_concurrent_runs: int = Field(ge=1)
    max_trials_per_run: int = Field(ge=1)
    llm_calls_per_run: int = Field(ge=0, le=25)
    llm_calls_max: int = Field(default=25, ge=0, le=25)
    max_bundle_mb: int = Field(gt=0)
    models_enabled: list[ModelName] | Literal["all_available"]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        absolute = self.rss_soft_mb is not None and self.rss_hard_mb is not None
        relative = (
            self.rss_soft_fraction_of_ram is not None and self.rss_hard_fraction_of_ram is not None
        )
        if absolute == relative:
            raise ValueError("set either rss_soft_mb/rss_hard_mb or the RAM fractions")
        if absolute and self.rss_soft_mb >= self.rss_hard_mb:  # type: ignore[operator]
            raise ValueError("rss_soft_mb must be below rss_hard_mb")
        if relative and self.rss_soft_fraction_of_ram >= self.rss_hard_fraction_of_ram:  # type: ignore[operator]
            raise ValueError("soft RAM fraction must be below hard RAM fraction")
        if (self.n_jobs == "auto") != (self.n_jobs_cap is not None):
            raise ValueError("n_jobs 'auto' requires n_jobs_cap (and only then)")
        if self.time_budget_default_s > self.time_budget_max_s:
            raise ValueError("time_budget_default_s exceeds time_budget_max_s")
        if self.llm_calls_per_run > self.llm_calls_max:
            raise ValueError("llm_calls_per_run exceeds llm_calls_max")
        return self

    def rss_limits_mb(self, physical_ram_mb: float) -> tuple[float, float]:
        """Soft and hard worker RSS limits in MB for a machine with `physical_ram_mb` (§29.1)."""
        if self.rss_soft_mb is not None and self.rss_hard_mb is not None:
            return float(self.rss_soft_mb), float(self.rss_hard_mb)
        assert self.rss_soft_fraction_of_ram is not None  # noqa: S101 - guaranteed by validator
        assert self.rss_hard_fraction_of_ram is not None  # noqa: S101
        return (
            physical_ram_mb * self.rss_soft_fraction_of_ram,
            physical_ram_mb * self.rss_hard_fraction_of_ram,
        )

    def resolved_n_jobs(self, physical_cores: int) -> int:
        """`n_jobs` for this machine: fixed, or min(physical cores, cap) (§29.1)."""
        if self.n_jobs == "auto":
            assert self.n_jobs_cap is not None  # noqa: S101 - guaranteed by validator
            return max(1, min(physical_cores, self.n_jobs_cap))
        return self.n_jobs


class EstimatorSettings(_Config):
    """Memory-estimator coefficients (PRD §29.4; calibrated in P18)."""

    safety: float = Field(ge=1.0)
    process_base_mb: float = Field(gt=0)


class ProfilesFile(_Config):
    schema_version: Literal["profiles/1.0"]
    profiles: dict[ExecutionProfileName, ProfileLimits]
    estimator: EstimatorSettings

    @model_validator(mode="after")
    def _both_profiles(self) -> Self:
        if set(self.profiles) != set(ExecutionProfileName):
            raise ValueError("profiles must define exactly 'hosted' and 'local'")
        return self


class ModelEntry(_Config):
    """One model-zoo entry (PRD §16.1)."""

    library: str
    estimators: dict[Literal["classification", "regression"], str]
    tasks: list[Task]
    profiles: list[ExecutionProfileName]
    defaults: dict[str, int | float | str | bool] = Field(default_factory=dict)
    hosted_max_depth_cap: int | None = Field(default=None, gt=0)
    fallback_only: bool = False


class ModelsFile(_Config):
    schema_version: Literal["models/1.0"]
    models: dict[ModelName, ModelEntry]


class ProviderEntry(_Config):
    """One LLM provider (PRD §23.3). Holds the key's env-var *name*, never the key."""

    base_url: str | None = None
    model_id: str | None = None
    model_ids: list[str | None] | None = None
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    json_mode: bool = False
    max_context_tokens: int | None = Field(default=None, gt=0)
    rate_limit_hint: dict[str, float] | None = None

    @property
    def configured_model_ids(self) -> list[str]:
        """Model ids that are actually set (empty when the env vars are unset)."""
        ids = self.model_ids if self.model_ids is not None else [self.model_id]
        return [i for i in ids if i]


class ProvidersFile(_Config):
    schema_version: Literal["providers/1.0"]
    order_hosted: list[ProviderName]
    order_local: list[ProviderName]
    providers: dict[ProviderName, ProviderEntry]

    @model_validator(mode="after")
    def _orders_reference_providers(self) -> Self:
        for order in (self.order_hosted, self.order_local):
            if not set(order) <= set(self.providers):
                raise ValueError("provider order lists unknown providers")
            if len(set(order)) != len(order):
                raise ValueError("provider order repeats a provider")
        return self


class Settings(_Config):
    """Validated, frozen settings for one process (selected profile + all config files)."""

    profile: ExecutionProfileName
    limits: ProfileLimits
    profiles: dict[ExecutionProfileName, ProfileLimits]
    estimator: EstimatorSettings
    models: dict[ModelName, ModelEntry]
    providers: dict[ProviderName, ProviderEntry]
    provider_order: list[ProviderName]
    data_dir: Path
    config_dir: Path

    def enabled_models(self) -> list[ModelName]:
        """Models configured for this profile, excluding fallback-only entries (§16.1).

        Library import checks happen in the model registry (P08).
        """
        return [
            name
            for name, entry in self.models.items()
            if self.profile in entry.profiles and not entry.fallback_only
        ]


def _interpolate(value: Any, env: Mapping[str, str], where: str) -> Any:
    """Replace full-string `${VAR}` placeholders; unset or empty variables become None."""
    if isinstance(value, dict):
        return {k: _interpolate(v, env, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, env, f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, str):
        match = PLACEHOLDER.match(value)
        if match:
            return env.get(match.group(1)) or None
        if "${" in value:
            raise ConfigError(f'{where}: placeholders must be the whole value, like "${{VAR}}"')
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"{path.name}: file not found in {path.parent}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: invalid YAML ({exc.__class__.__name__})") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: top level must be a mapping")
    return data


def _validate(model: type[_M], data: dict[str, Any], filename: str) -> _M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"{filename}: {problems}") from exc


def load_settings(env: Mapping[str, str] | None = None, config_dir: Path | None = None) -> Settings:
    """Load and validate settings. `env` defaults to `os.environ` (injectable for tests)."""
    env = os.environ if env is None else env
    config_dir = Path(env.get("TABPILOT_CONFIG_DIR") or config_dir or DEFAULT_CONFIG_DIR)
    profile_raw = env.get("TABPILOT_PROFILE") or ExecutionProfileName.LOCAL.value
    try:
        profile = ExecutionProfileName(profile_raw)
    except ValueError as exc:
        allowed = ", ".join(p.value for p in ExecutionProfileName)
        raise ConfigError(
            f"TABPILOT_PROFILE={profile_raw!r} is invalid; use one of: {allowed}"
        ) from exc

    files: dict[str, dict[str, Any]] = {}
    for name in ("profiles.yaml", "models.yaml", "providers.yaml"):
        files[name] = _interpolate(_read_yaml(config_dir / name), env, name)
    profiles = _validate(ProfilesFile, files["profiles.yaml"], "profiles.yaml")
    models = _validate(ModelsFile, files["models.yaml"], "models.yaml")
    providers = _validate(ProvidersFile, files["providers.yaml"], "providers.yaml")

    data_dir = Path(env.get("TABPILOT_DATA_DIR") or Path(tempfile.gettempdir()) / "tabpilot")
    order = (
        providers.order_hosted if profile == ExecutionProfileName.HOSTED else providers.order_local
    )
    settings = Settings(
        profile=profile,
        limits=profiles.profiles[profile],
        profiles=profiles.profiles,
        estimator=profiles.estimator,
        models=models.models,
        providers=providers.providers,
        provider_order=order,
        data_dir=data_dir,
        config_dir=config_dir,
    )
    for name, limits in profiles.profiles.items():
        configured = settings.model_copy(update={"profile": name}).enabled_models()
        if limits.models_enabled != "all_available" and set(limits.models_enabled) != set(
            configured
        ):
            raise ConfigError(
                f"profiles.yaml: models_enabled for '{name}' disagrees with models.yaml profiles"
            )
    return settings

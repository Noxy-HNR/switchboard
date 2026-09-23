"""Configuration loading.

Values come from four places, later ones winning: ``config/switchboard.toml``
(the commented defaults), ``config/local.toml`` (what the dashboard's settings
page saved; git-ignored), a ``.env`` file in the project root, and the process
environment. Environment overrides are named ``SWITCHBOARD_<SECTION>__<KEY>``.

Invalid configuration fails here, at load time, rather than as strange
behaviour later: a mistyped key, a port out of range, an unknown decision
backend or an environment value like ``flase`` all raise before the service
binds a socket or sends a request anywhere.

The only secret Switchboard reads is ``TYPESAFE_API_KEY``, and it is never
written to the database, the logs, the config file, or a ``repr``.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from switchboard.decisions.schema import DOMAINS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "switchboard.toml"
DEFAULT_MODELS_PATH = PROJECT_ROOT / "config" / "models.toml"

ENV_PREFIX = "SWITCHBOARD_"

# Written by the dashboard's settings page, next to the main config file.
LOCAL_SETTINGS_NAME = "local.toml"

# Every decision backend Switchboard ships. The factory builds from this list,
# so a name that is not here cannot be configured by accident.
KNOWN_BACKENDS: tuple[str, ...] = ("jev", "local", "logit", "rules")

# The store truncates previews to this regardless; the setting may only tighten it.
MAX_PREVIEW_CHARS = 200


class _Section(BaseModel):
    """A config table: unknown keys are errors, and so are NaN and infinity."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ServerConfig(_Section):
    host: str = "127.0.0.1"
    port: int = Field(default=8002, ge=1, le=65535)


class UpstreamConfig(_Section):
    base_url: str = "http://127.0.0.1:10000"
    connect_timeout_s: float = Field(default=5.0, gt=0)
    request_timeout_s: float = Field(default=900.0, gt=0)
    health_ttl_s: float = Field(default=2.0, ge=0)


class StackConfig(_Section):
    ai_root: Path = Path(r"C:\AI")
    models_dir: Path = Path(r"C:\AI\models")
    models_config: Path = Path(r"C:\AI\Config\models.json")
    dashboard_url: str = "http://127.0.0.1:9090"
    # Read-only: Lecture Notes' live status, for the recording guard.
    recording_status_file: Path = Path(r"C:\AI\Projects\lecture-notes\state\status.json")


class GuardConfig(_Section):
    hysteresis_seconds: float = Field(default=12.0, ge=0)
    min_quality_gain: float = Field(default=0.08, ge=0, le=1)
    min_dwell_seconds: float = Field(default=120.0, ge=0)
    never_switch_when_busy: bool = True
    max_switch_wait_s: float = Field(default=300.0, gt=0)
    utility_max_tokens: int = Field(default=160, ge=0)
    ram_guard: Literal["warn", "block"] = "warn"
    ram_headroom_gb: float = Field(default=1.0, ge=0)
    # While Lecture Notes records, no switch to a model file larger than this.
    recording_guard: bool = True
    recording_max_model_gb: float = Field(default=9.0, gt=0)


class RoutingConfig(_Section):
    mode: Literal["active", "shadow", "off"] = "active"
    allow_explicit_switch: bool = True
    default_policy: Literal["auto", "fast", "deep", "medical"] = "auto"
    guards: GuardConfig = Field(default_factory=GuardConfig)


class PolicyConfig(_Section):
    value_of_quality_s: float = Field(default=400.0, ge=0)
    capability_slope: float = Field(default=2.5, gt=0)
    generation_wait_weight: float = Field(default=0.35, ge=0, le=1)
    expected_output_tokens: list[int] = Field(default=[80, 300, 700, 1200], min_length=1)
    chars_per_token: float = Field(default=3.6, gt=0)
    default_load_s_per_gb: float = Field(default=1.7, ge=0)
    difficulty_ewma_alpha: float = Field(default=0.5, gt=0, le=1)

    @field_validator("expected_output_tokens")
    @classmethod
    def _positive_lengths(cls, value: list[int]) -> list[int]:
        if any(item <= 0 for item in value):
            raise ValueError("expected_output_tokens must all be positive")
        return value


class JevConfig(_Section):
    enabled: bool | Literal["auto"] = "auto"
    model: str = "jev-latest"
    timeout_s: float = Field(default=2.0, gt=0)
    max_retries: int = Field(default=1, ge=0, le=10)
    breaker_failures: int = Field(default=3, ge=1)
    breaker_cooldown_s: float = Field(default=60.0, ge=0)
    usd_per_input_mtok: float = Field(default=0.042, ge=0)


class LocalDecisionConfig(_Section):
    enabled: bool = True
    timeout_s: float = Field(default=20.0, gt=0)
    min_resident_gen_tps: float = Field(default=15.0, ge=0)
    skip_when_busy: bool = True
    answer_mode: Literal["discrete", "probabilities"] = "discrete"
    max_output_tokens: int = Field(default=512, ge=16)


class LogitDecisionConfig(_Section):
    enabled: bool = False
    timeout_s: float = Field(default=15.0, gt=0)
    top_logprobs: int = Field(default=20, ge=2, le=20)
    min_resident_gen_tps: float = Field(default=0.0, ge=0)
    skip_when_busy: bool = True


class RulesConfig(_Section):
    enabled: bool = True


class FastPathConfig(_Section):
    enabled: bool = True
    max_difficulty: float = Field(default=1.0, ge=0, le=3)
    min_confidence: float = Field(default=0.5, ge=0, le=1)
    always_confirm_domains: list[str] = ["medical"]

    @field_validator("always_confirm_domains")
    @classmethod
    def _known_domains(cls, value: list[str]) -> list[str]:
        unknown = [item for item in value if item not in DOMAINS]
        if unknown:
            raise ValueError(f"unknown domain(s) {unknown}; valid: {list(DOMAINS)}")
        return value


class DecisionsConfig(_Section):
    backends: list[str] = ["jev", "local", "rules"]
    budget_ms: int = Field(default=6000, ge=0)
    cache_ttl_s: float = Field(default=300.0, ge=0)
    cache_max_entries: int = Field(default=512, ge=1)
    jev: JevConfig = Field(default_factory=JevConfig)
    local: LocalDecisionConfig = Field(default_factory=LocalDecisionConfig)
    logit: LogitDecisionConfig = Field(default_factory=LogitDecisionConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    fast_path: FastPathConfig = Field(default_factory=FastPathConfig)

    @field_validator("backends")
    @classmethod
    def _known_backends(cls, value: list[str]) -> list[str]:
        unknown = [item for item in value if item not in KNOWN_BACKENDS]
        if unknown:
            raise ValueError(f"unknown decision backend(s) {unknown}; valid: {list(KNOWN_BACKENDS)}")
        if len(set(value)) != len(value):
            raise ValueError("a decision backend is listed twice")
        return value


class PrivacyConfig(_Section):
    redact: bool = True
    max_state_chars: int = Field(default=1500, ge=0)
    store_previews: bool = False
    preview_chars: int = Field(default=80, ge=0, le=MAX_PREVIEW_CHARS)
    allow_cloud: bool = True


class ObservabilityConfig(_Section):
    db_path: Path = Path("data/switchboard.sqlite3")
    retention_days: int = Field(default=30, ge=1)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    upstream: UpstreamConfig = Field(default_factory=UpstreamConfig)
    stack: StackConfig = Field(default_factory=StackConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    decisions: DecisionsConfig = Field(default_factory=DecisionsConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # Not part of the file: filled in from the environment.
    typesafe_api_key: str | None = Field(default=None, repr=False)
    config_path: Path = DEFAULT_CONFIG_PATH
    models_path: Path = DEFAULT_MODELS_PATH

    @property
    def base_url(self) -> str:
        return f"http://{self.server.host}:{self.server.port}"

    @property
    def db_file(self) -> Path:
        path = self.observability.db_path
        return path if path.is_absolute() else PROJECT_ROOT / path

    def jev_enabled(self) -> bool:
        """Whether the Jev backend should be attempted at all."""
        if not self.privacy.allow_cloud:
            return False
        if self.decisions.jev.enabled == "auto":
            return bool(self.typesafe_api_key)
        return bool(self.decisions.jev.enabled)


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse a .env file. Values already in the environment win."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            values[key.strip()] = value
    return values


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(reference: Any, raw: str, name: str) -> Any:
    """Coerce an environment string to the type of the setting it overrides.

    The reference is the setting's default, so the type comes from the schema
    rather than from whatever the TOML happened to contain. A boolean that is
    neither clearly true nor clearly false is an error, not a silent False.
    """
    text = raw.strip()
    if isinstance(reference, bool):
        lowered = text.lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ValueError(f"{name}={raw!r} is not a boolean (use true/false)")
    if isinstance(reference, int):
        return int(text)
    if isinstance(reference, float):
        return float(text)
    if isinstance(reference, list):
        return [item.strip() for item in text.split(",") if item.strip()]
    # Strings, paths, literals and "auto"-or-bool unions: pydantic validates them.
    return text


def _apply_env_overrides(data: dict[str, Any], environ: dict[str, str]) -> list[str]:
    """Apply SWITCHBOARD_SECTION__KEY overrides in place; return the keys applied.

    Paths are resolved against the schema, not against the TOML, so an override
    works even for a table the file omits. An override naming a setting that
    does not exist is an error: a typo in an environment variable should be as
    loud as a typo in the file.
    """
    reference_root = Config().model_dump(mode="python")
    applied: list[str] = []
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX) :].lower().split("__")
        reference: Any = reference_root
        for part in path:
            if not isinstance(reference, dict) or part not in reference:
                raise ValueError(f"{key} does not name a Switchboard setting")
            reference = reference[part]
        if isinstance(reference, dict):
            raise ValueError(f"{key} names a table, not a setting")

        target = data
        for part in path[:-1]:
            existing = target.get(part)
            if not isinstance(existing, dict):
                existing = {}
                target[part] = existing
            target = existing
        target[path[-1]] = _coerce(reference, raw, key)
        applied.append(key)
    return applied


def local_settings_path(config_path: Path) -> Path:
    return config_path.parent / LOCAL_SETTINGS_NAME


def dotenv_path(config_path: Path) -> Path:
    return config_path.parent.parent / ".env"


def read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8-sig"))


def merge_settings(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """`overlay` on top of `base`, table by table; neither input is modified."""
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            merged[key] = merge_settings(existing, value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None = None, environ: dict[str, str] | None = None) -> Config:
    """Load configuration from TOML, the dashboard's saved settings, .env and the environment."""
    config_path = path or DEFAULT_CONFIG_PATH
    data = merge_settings(read_toml(config_path), read_toml(local_settings_path(config_path)))

    env = dict(environ if environ is not None else os.environ)
    # A .env in the project root supplies anything the environment lacks.
    for key, value in load_dotenv(dotenv_path(config_path)).items():
        env.setdefault(key, value)

    _apply_env_overrides(data, env)

    config = Config.model_validate(data)
    config.config_path = config_path
    config.models_path = config_path.parent / "models.toml"
    key = (env.get("TYPESAFE_API_KEY") or "").strip()
    config.typesafe_api_key = key or None
    return config


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide configuration. Tests construct their own instead."""
    return load_config()

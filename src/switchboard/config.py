"""Configuration loading.

Values come from three places, later ones winning: ``config/switchboard.toml``,
a ``.env`` file next to it, and the process environment. Environment overrides
are named ``SWITCHBOARD_<SECTION>__<KEY>``.

The only secret Switchboard reads is ``TYPESAFE_API_KEY``, and it is never
written to the database, the logs, or the config file.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "switchboard.toml"
DEFAULT_MODELS_PATH = PROJECT_ROOT / "config" / "models.toml"

ENV_PREFIX = "SWITCHBOARD_"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8002


class UpstreamConfig(BaseModel):
    base_url: str = "http://127.0.0.1:10000"
    connect_timeout_s: float = 5.0
    request_timeout_s: float = 900.0
    health_ttl_s: float = 2.0


class StackConfig(BaseModel):
    ai_root: Path = Path(r"C:\AI")
    models_dir: Path = Path(r"C:\AI\models")
    models_config: Path = Path(r"C:\AI\Config\models.json")
    dashboard_url: str = "http://127.0.0.1:9090"


class GuardConfig(BaseModel):
    hysteresis_seconds: float = 12.0
    min_quality_gain: float = 0.08
    min_dwell_seconds: float = 120.0
    never_switch_when_busy: bool = True
    max_switch_wait_s: float = 300.0
    utility_max_tokens: int = 160
    ram_guard: Literal["warn", "block"] = "warn"
    ram_headroom_gb: float = 1.0


class RoutingConfig(BaseModel):
    mode: Literal["active", "shadow", "off"] = "active"
    allow_explicit_switch: bool = True
    default_policy: str = "auto"
    guards: GuardConfig = Field(default_factory=GuardConfig)


class PolicyConfig(BaseModel):
    value_of_quality_s: float = 400.0
    capability_slope: float = 2.5
    generation_wait_weight: float = 0.35
    expected_output_tokens: list[int] = [80, 300, 700, 1200]
    chars_per_token: float = 3.6
    default_load_s_per_gb: float = 1.7
    difficulty_ewma_alpha: float = 0.5


class JevConfig(BaseModel):
    enabled: bool | Literal["auto"] = "auto"
    model: str = "jev-latest"
    timeout_s: float = 4.0
    max_retries: int = 2
    breaker_failures: int = 3
    breaker_cooldown_s: float = 60.0
    usd_per_input_mtok: float = 0.042


class LocalDecisionConfig(BaseModel):
    enabled: bool = True
    timeout_s: float = 20.0
    min_resident_gen_tps: float = 15.0
    skip_when_busy: bool = True
    answer_mode: Literal["discrete", "probabilities"] = "discrete"
    max_output_tokens: int = 512


class LogitDecisionConfig(BaseModel):
    enabled: bool = False
    timeout_s: float = 15.0
    top_logprobs: int = 20
    min_resident_gen_tps: float = 0.0
    skip_when_busy: bool = True


class RulesConfig(BaseModel):
    enabled: bool = True


class DecisionsConfig(BaseModel):
    backends: list[str] = ["jev", "local", "rules"]
    budget_ms: int = 4000
    cache_ttl_s: float = 300.0
    cache_max_entries: int = 512
    jev: JevConfig = Field(default_factory=JevConfig)
    local: LocalDecisionConfig = Field(default_factory=LocalDecisionConfig)
    logit: LogitDecisionConfig = Field(default_factory=LogitDecisionConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)


class PrivacyConfig(BaseModel):
    redact: bool = True
    max_state_chars: int = 1500
    store_previews: bool = True
    preview_chars: int = 80
    allow_cloud: bool = True


class ObservabilityConfig(BaseModel):
    db_path: Path = Path("data/switchboard.sqlite3")
    retention_days: int = 30


class Config(BaseModel):
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


def _coerce(current: Any, raw: str) -> Any:
    """Coerce an environment string to the shape of the value it replaces."""
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return raw


def _apply_env_overrides(data: dict[str, Any], environ: dict[str, str]) -> list[str]:
    """Apply SWITCHBOARD_SECTION__KEY overrides in place; return the keys applied."""
    applied: list[str] = []
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX) :].lower().split("__")
        target: Any = data
        for part in path[:-1]:
            if not isinstance(target, dict) or part not in target:
                target = None
                break
            target = target[part]
        if not isinstance(target, dict):
            continue
        leaf = path[-1]
        target[leaf] = _coerce(target.get(leaf), raw)
        applied.append(key)
    return applied


def load_config(path: Path | None = None, environ: dict[str, str] | None = None) -> Config:
    """Load configuration from TOML plus .env plus the environment."""
    config_path = path or DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if config_path.is_file():
        data = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))

    env = dict(environ if environ is not None else os.environ)
    # A .env sitting next to the config supplies anything the environment lacks.
    for key, value in load_dotenv(config_path.parent.parent / ".env").items():
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

"""Settings and keys the dashboard can change, and where they are kept.

Two stores, both git-ignored and both in this project:

* ``config/local.toml`` holds settings saved from the dashboard. It is layered
  over ``config/switchboard.toml`` (see `config.load_config`), so the commented
  defaults file is never rewritten and deleting ``local.toml`` resets
  everything.
* ``.env`` in the project root holds keys. A key is written there and nowhere
  else: never to ``local.toml``, the database, a log, or an API response. The
  settings API reports only whether a key is set, where it came from and its
  last four characters.

Changes apply live. The service swaps in a new config, policy and decision
chain (`Router.reconfigure`), so there is no restart. Only the settings listed
in `SETTINGS` can be changed this way; addresses, ports and paths are read at
start-up and stay in ``switchboard.toml``.

A setting or key that the process environment sets (``SWITCHBOARD_...`` or
``TYPESAFE_API_KEY``) is shown as pinned and cannot be changed here: the
environment wins over both files, so a saved value would silently stop
applying at the next start.
"""

from __future__ import annotations

import json
import math
import os
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, ValidationError

from switchboard.config import (
    ENV_PREFIX,
    KNOWN_BACKENDS,
    Config,
    dotenv_path,
    load_dotenv,
    local_settings_path,
    read_toml,
)
from switchboard.decisions.base import BackendError, BackendFacts, TaskContext
from switchboard.decisions.factory import build_chain

Kind = Literal["choice", "bool", "int", "float", "text", "backends", "tristate"]


@dataclass(frozen=True, slots=True)
class Setting:
    """One setting the dashboard can change."""

    path: str
    section: str
    label: str
    help: str
    kind: Kind
    choices: tuple[str, ...] = ()
    unit: str = ""


SETTINGS: tuple[Setting, ...] = (
    # --- routing
    Setting(
        "routing.mode", "Routing", "Mode",
        "active: switchboard-auto may load a better model through the C:\\AI dashboard. "
        "shadow: never switches, and records what it would have done. "
        "off: plain pass-through to whatever is loaded.",
        "choice", ("active", "shadow", "off"),
    ),
    Setting(
        "routing.default_policy", "Routing", "Default policy",
        "Used when a request names no known model. auto weighs quality against waiting; "
        "fast stays on the loaded model; deep and medical prefer the strong models.",
        "choice", ("auto", "fast", "deep", "medical"),
    ),
    Setting(
        "routing.allow_explicit_switch", "Routing", "Load a model a request names",
        "When a request names a real model that is not loaded, load it. Off: refuse instead.",
        "bool",
    ),
    # --- guards
    Setting(
        "routing.guards.hysteresis_seconds", "Switch guards", "Hysteresis",
        "A switch must beat staying put by this much expected utility.", "float", unit="s",
    ),
    Setting(
        "routing.guards.min_quality_gain", "Switch guards", "Minimum quality gain",
        "A switch must raise the chance of a good answer by at least this much (0 to 1).",
        "float",
    ),
    Setting(
        "routing.guards.min_dwell_seconds", "Switch guards", "Minimum dwell",
        "After a conversation switches, keep its new model at least this long.", "float", unit="s",
    ),
    Setting(
        "routing.guards.never_switch_when_busy", "Switch guards", "Never switch while busy",
        "Never restart llama-server while it is generating for anyone.", "bool",
    ),
    Setting(
        "routing.guards.utility_max_tokens", "Switch guards", "Utility-call size",
        "Non-streaming calls asking for at most this many tokens (chat titles) never wait for a load.",
        "int", unit="tokens",
    ),
    Setting(
        "routing.guards.ram_guard", "Switch guards", "Host RAM guard",
        "warn: allow a model whose RAM share may not fit, and say so. block: exclude it.",
        "choice", ("warn", "block"),
    ),
    # --- policy
    Setting(
        "policy.value_of_quality_s", "Policy", "Value of a better answer",
        "Seconds of waiting a certain-good answer is worth over a certain-bad one. "
        "Higher switches more eagerly.",
        "float", unit="s",
    ),
    Setting(
        "policy.generation_wait_weight", "Policy", "Streaming generation weight",
        "How much a second of streamed generation counts against a second of loading (0 to 1).",
        "float",
    ),
    # --- decisions
    Setting(
        "decisions.backends", "Decisions", "Decision backends",
        "Tried in this order; the first real answer wins. The rules backend cannot fail and "
        "always runs last.",
        "backends",
    ),
    Setting(
        "decisions.jev.enabled", "Decisions", "Jev (cloud)",
        "auto: on when a TypeSafe key is saved. on/off force it.",
        "tristate", ("auto", "on", "off"),
    ),
    Setting(
        "decisions.jev.model", "Decisions", "Jev model",
        "jev-latest follows TypeSafe's current model; pin a version such as jev-1.13.0 to hold it.",
        "text",
    ),
    Setting(
        "decisions.logit.enabled", "Decisions", "Logit backend (experimental)",
        "First-token probabilities from the loaded model. Its numbers are not calibrated; it "
        "only runs if it is also listed among the backends.",
        "bool",
    ),
    Setting(
        "decisions.fast_path.enabled", "Decisions", "Fast path",
        "Let the rules settle requests they are confident are easy, without asking a model.",
        "bool",
    ),
    Setting(
        "decisions.budget_ms", "Decisions", "Decision budget",
        "Total time the semantic layer may take per request before the rules answer.",
        "int", unit="ms",
    ),
    Setting(
        "decisions.local.min_resident_gen_tps", "Decisions", "Local backend speed floor",
        "Skip the local backend when the loaded model generates slower than this.",
        "float", unit="tok/s",
    ),
    # --- privacy
    Setting(
        "privacy.allow_cloud", "Privacy", "Allow cloud decisions",
        "Off keeps everything on this machine, even with a key saved.", "bool",
    ),
    Setting(
        "privacy.redact", "Privacy", "Redact secrets before sending",
        "Strip credential-shaped text from anything sent to the cloud. Strongly advised.", "bool",
    ),
    Setting(
        "privacy.max_state_chars", "Privacy", "Cloud state size",
        "At most this many characters of a request are sent per decision.", "int", unit="chars",
    ),
    Setting(
        "privacy.store_previews", "Privacy", "Store request previews",
        "Keep a short redacted excerpt of each request in the database, for debugging.", "bool",
    ),
    Setting(
        "privacy.preview_chars", "Privacy", "Preview length",
        "Length of a stored preview (at most 200).", "int", unit="chars",
    ),
)

_BY_PATH = {setting.path: setting for setting in SETTINGS}


@dataclass(frozen=True, slots=True)
class KeySpec:
    """One secret the dashboard can store."""

    id: str
    env: str
    label: str
    used_by: str
    help: str


KEYS: tuple[KeySpec, ...] = (
    KeySpec(
        "typesafe",
        "TYPESAFE_API_KEY",
        "TypeSafe API key",
        "Jev, the cloud decision backend",
        "Turns on Jev when the Jev setting is auto and cloud decisions are allowed. "
        "Test sends one short synthetic sentence to TypeSafe, never your requests.",
    ),
)

_KEYS_BY_ID = {spec.id: spec for spec in KEYS}
_KEY_MIN_CHARS = 8
_KEY_MAX_CHARS = 512
_KEY_FORBIDDEN = set("\"'`\\")


class SettingsError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


# --- dotted paths ------------------------------------------------------------


def _get(data: dict[str, Any], path: str) -> Any:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _has(data: dict[str, Any], path: str) -> bool:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _set(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _delete(data: dict[str, Any], path: str) -> None:
    """Remove a setting, and any table it leaves empty."""
    parts = path.split(".")
    trail: list[tuple[dict[str, Any], str]] = []
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            return
        trail.append((node, part))
        node = child
    node.pop(parts[-1], None)
    for parent, part in reversed(trail):
        if parent[part]:
            break
        del parent[part]


def env_var_for(path: str) -> str:
    return ENV_PREFIX + path.upper().replace(".", "__")


def _environment(config_path: Path, environ: dict[str, str]) -> dict[str, str]:
    """What load_config would see: the process environment, then .env beneath it."""
    merged = dict(load_dotenv(dotenv_path(config_path)))
    merged.update(environ)
    return merged


# --- reading -------------------------------------------------------------------


def _file_value(config_path: Path, path: str) -> Any:
    """The value switchboard.toml gives a setting, or the built-in default."""
    base = read_toml(config_path)
    if _has(base, path):
        return _get(base, path)
    return _get(Config().model_dump(mode="json"), path)


def describe_settings(config: Config, environ: dict[str, str]) -> list[dict[str, Any]]:
    current = config.model_dump(mode="json")
    saved = read_toml(local_settings_path(config.config_path))
    env = _environment(config.config_path, environ)
    rows = []
    for setting in SETTINGS:
        pinned = env_var_for(setting.path)
        rows.append(
            {
                "path": setting.path,
                "section": setting.section,
                "label": setting.label,
                "help": setting.help,
                "kind": setting.kind,
                "choices": list(setting.choices),
                "unit": setting.unit,
                "value": _get(current, setting.path),
                "file_value": _file_value(config.config_path, setting.path),
                "saved": _has(saved, setting.path),
                "pinned_by": pinned if pinned in env else None,
            }
        )
    return rows


def describe_keys(config: Config, environ: dict[str, str]) -> list[dict[str, Any]]:
    stored = load_dotenv(dotenv_path(config.config_path))
    rows = []
    for spec in KEYS:
        from_process = (environ.get(spec.env) or "").strip()
        from_file = (stored.get(spec.env) or "").strip()
        value = from_process or from_file
        rows.append(
            {
                "id": spec.id,
                "env": spec.env,
                "label": spec.label,
                "used_by": spec.used_by,
                "help": spec.help,
                "set": bool(value),
                "source": "environment" if from_process else ".env" if from_file else None,
                # Enough to tell two keys apart, not enough to use one.
                "hint": f"…{value[-4:]}" if len(value) >= 12 else None,
                "pinned": bool(from_process),
            }
        )
    return rows


# --- writing -------------------------------------------------------------------


def _normalise(setting: Setting, value: Any) -> Any:
    if setting.kind == "tristate":
        mapping = {"auto": "auto", "on": True, "off": False, True: True, False: False}
        if value not in mapping:
            raise SettingsError(422, f"{setting.path} must be auto, on or off")
        return mapping[value]
    if setting.kind == "backends":
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise SettingsError(422, f"{setting.path} must be a list of backend names")
        unknown = [item for item in value if item not in KNOWN_BACKENDS]
        if unknown:
            raise SettingsError(422, f"unknown decision backend(s) {unknown}")
        # Canonical order, with the floor always last.
        chosen = [name for name in KNOWN_BACKENDS if name in value and name != "rules"]
        return [*chosen, "rules"]
    if setting.kind == "float" and isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise SettingsError(422, f"{setting.path} must be a finite number")
        return float(value)
    return value


def _validated(config: Config, changes: dict[str, Any]) -> Config:
    """The config with `changes` applied, or a 422 naming the first problem."""
    data = config.model_dump(mode="python")
    for path, value in changes.items():
        _set(data, path, value)
    try:
        updated = Config.model_validate(data)
    except ValidationError as error:
        first = error.errors()[0]
        where = ".".join(str(part) for part in first.get("loc", ()))
        raise SettingsError(422, f"{where}: {first.get('msg', 'invalid value')}") from None
    updated.typesafe_api_key = config.typesafe_api_key
    return updated


def save_settings(config: Config, values: dict[str, Any], environ: dict[str, str]) -> Config:
    """Validate, persist to local.toml and return the config to run with."""
    if not values:
        raise SettingsError(422, "no settings to save")
    env = _environment(config.config_path, environ)
    changes: dict[str, Any] = {}
    for path, raw in values.items():
        setting = _BY_PATH.get(path)
        if setting is None:
            raise SettingsError(404, f"{path} is not a setting the dashboard can change")
        if env_var_for(path) in env:
            raise SettingsError(
                409, f"{path} is set by the environment variable {env_var_for(path)}; change it there"
            )
        changes[path] = _normalise(setting, raw)

    updated = _validated(config, changes)

    local_path = local_settings_path(config.config_path)
    saved = read_toml(local_path)
    for path, value in changes.items():
        # A value equal to switchboard.toml's is not an override: keep local.toml minimal.
        if value == _file_value(config.config_path, path):
            _delete(saved, path)
        else:
            _set(saved, path, value)
    write_toml(local_path, saved)
    return updated


def reset_setting(config: Config, path: str, environ: dict[str, str]) -> Config:
    """Drop a saved override; the setting goes back to switchboard.toml's value."""
    if path not in _BY_PATH:
        raise SettingsError(404, f"{path} is not a setting the dashboard can change")
    env = _environment(config.config_path, environ)
    if env_var_for(path) in env:
        raise SettingsError(409, f"{path} is set by the environment variable {env_var_for(path)}")
    updated = _validated(config, {path: _file_value(config.config_path, path)})
    local_path = local_settings_path(config.config_path)
    saved = read_toml(local_path)
    _delete(saved, path)
    write_toml(local_path, saved)
    return updated


def _check_key(value: str) -> str:
    key = value.strip()
    if not key:
        raise SettingsError(422, "the key is empty")
    if not _KEY_MIN_CHARS <= len(key) <= _KEY_MAX_CHARS:
        raise SettingsError(422, f"a key is {_KEY_MIN_CHARS} to {_KEY_MAX_CHARS} characters")
    # The SDK's own rule (printable ASCII, no whitespace), plus the quote and
    # escape characters a .env line cannot carry unquoted.
    if not key.isascii() or not key.isprintable() or any(ch.isspace() for ch in key):
        raise SettingsError(422, "a key is printable ASCII with no spaces")
    if _KEY_FORBIDDEN & set(key):
        raise SettingsError(422, "a key cannot contain quotes or backslashes")
    return key


def _key_spec(key_id: str) -> KeySpec:
    spec = _KEYS_BY_ID.get(key_id)
    if spec is None:
        raise SettingsError(404, f"no key named {key_id!r}")
    return spec


def save_key(config: Config, key_id: str, value: str | None, environ: dict[str, str]) -> Config:
    """Write (or with None, remove) a key in .env and return the config to run with."""
    spec = _key_spec(key_id)
    if (environ.get(spec.env) or "").strip():
        raise SettingsError(
            409,
            f"{spec.env} is set in this process's environment, which wins over .env; "
            "change it there",
        )
    key = _check_key(value) if value is not None else None
    write_env_value(dotenv_path(config.config_path), spec.env, key)
    updated = config.model_copy(deep=True)
    if spec.env == "TYPESAFE_API_KEY":
        updated.typesafe_api_key = key
    return updated


# --- files -----------------------------------------------------------------------


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("TOML settings must be finite")
        return repr(value)
    if isinstance(value, str):
        # A JSON string with ASCII escapes is a valid TOML basic string.
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise ValueError(f"cannot write {type(value).__name__} to TOML")


def _toml_lines(data: dict[str, Any], prefix: str = "") -> list[str]:
    lines: list[str] = []
    scalars = {key: value for key, value in data.items() if not isinstance(value, dict)}
    tables = {key: value for key, value in data.items() if isinstance(value, dict)}
    if scalars and prefix:
        lines.append(f"[{prefix}]")
    for key, value in scalars.items():
        lines.append(f"{key} = {_toml_value(value)}")
    if scalars:
        lines.append("")
    for key, value in tables.items():
        lines.extend(_toml_lines(value, f"{prefix}.{key}" if prefix else key))
    return lines


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_toml(path: Path, data: dict[str, Any]) -> None:
    header = [
        "# Settings saved from the Switchboard dashboard. Layered over switchboard.toml;",
        "# delete this file (or use Reset in the dashboard) to go back to the defaults.",
        "",
    ]
    text = "\n".join(header + _toml_lines(data))
    if tomllib.loads(text) != data:  # pragma: no cover - a serialiser bug, not user input
        raise ValueError("local.toml would not read back as written")
    _atomic_write(path, text)


def write_env_value(path: Path, name: str, value: str | None) -> None:
    """Set or remove one ``NAME=value`` line, leaving every other line as it was."""
    if value is None and not path.is_file():
        return
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.is_file() else [
        "# Secrets for Switchboard. Git-ignored; written by the dashboard's settings page.",
    ]
    kept: list[str] = []
    written = False
    for line in lines:
        stripped = line.strip()
        name_here = stripped.partition("=")[0].strip() if "=" in stripped else None
        if not stripped.startswith("#") and name_here == name:
            if value is not None and not written:
                kept.append(f"{name}={value}")
                written = True
            continue
        kept.append(line)
    if value is not None and not written:
        kept.append(f"{name}={value}")
    _atomic_write(path, "\n".join(kept) + "\n")


# --- testing a key -------------------------------------------------------------------

_PROBE_MESSAGE = "What is two plus two?"


async def probe_typesafe_key(config: Config, transport: Any = None) -> dict[str, Any]:
    """One real Jev decision on a fixed synthetic sentence, to see whether the key works."""
    from switchboard.decisions.jev import JevBackend

    if not config.typesafe_api_key:
        return {"ok": False, "result": "not_set", "detail": "No TypeSafe key is saved."}
    if not config.privacy.allow_cloud:
        return {
            "ok": False,
            "result": "disabled",
            "detail": "Cloud decisions are off (Privacy: allow cloud decisions). Turn them on to test.",
        }
    probe = config.model_copy(deep=True)
    probe.decisions.jev.enabled = True
    probe.decisions.jev.max_retries = 0
    backend = JevBackend(probe, transport=transport)
    started = time.perf_counter()
    try:
        await backend.decide_task(TaskContext(message=_PROBE_MESSAGE), BackendFacts())
    except BackendError as error:
        outcome = {
            "auth": ("rejected", "TypeSafe refused the key."),
            "rate_limit": ("rate_limited", "The key works, but TypeSafe is rate-limiting it right now."),
            "timeout": ("unreachable", "TypeSafe did not answer in time."),
            "connection": ("unreachable", "Could not reach TypeSafe from this machine."),
        }.get(error.kind, ("error", "The test call failed."))
        return {
            "ok": outcome[0] == "rate_limited",
            "result": outcome[0],
            "detail": f"{outcome[1]} ({error.kind})",
        }
    finally:
        await backend.aclose()
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "ok": True,
        "result": "working",
        "detail": f"Jev answered in {elapsed:.0f} ms.",
        "latency_ms": round(elapsed, 1),
    }


# --- API -------------------------------------------------------------------------------


class _SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[str, Any]


class _KeyUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str


def build_settings_router(get_router: Callable[[Request], Any]) -> APIRouter:
    api = APIRouter()

    def payload(config: Config) -> dict[str, Any]:
        environ = dict(os.environ)
        return {
            "settings": describe_settings(config, environ),
            "keys": describe_keys(config, environ),
            "storage": {
                "settings_file": str(local_settings_path(config.config_path)),
                "keys_file": str(dotenv_path(config.config_path)),
            },
        }

    def apply(request: Request, updated: Config) -> dict[str, Any]:
        router = get_router(request)
        factory = getattr(request.app.state, "chain_factory", None) or build_chain
        router.reconfigure(updated, factory(updated, router.upstream))
        request.app.state.config = updated
        return payload(updated)

    def fail(error: SettingsError) -> HTTPException:
        return HTTPException(status_code=error.status, detail=str(error))

    @api.get("/settings", summary="Settings and keys the dashboard can change")
    async def get_settings(request: Request) -> dict[str, Any]:
        return payload(get_router(request).config)

    @api.put("/settings", summary="Save settings; they apply without a restart")
    async def put_settings(request: Request, update: _SettingsUpdate) -> dict[str, Any]:
        try:
            updated = save_settings(get_router(request).config, update.values, dict(os.environ))
        except SettingsError as error:
            raise fail(error) from None
        return apply(request, updated)

    @api.delete("/settings/{path}", summary="Reset one setting to switchboard.toml's value")
    async def delete_setting(request: Request, path: str) -> dict[str, Any]:
        try:
            updated = reset_setting(get_router(request).config, path, dict(os.environ))
        except SettingsError as error:
            raise fail(error) from None
        return apply(request, updated)

    @api.put("/settings/keys/{key_id}", summary="Save a key to .env (never returned)")
    async def put_key(request: Request, key_id: str, update: _KeyUpdate) -> dict[str, Any]:
        try:
            updated = save_key(get_router(request).config, key_id, update.value, dict(os.environ))
        except SettingsError as error:
            raise fail(error) from None
        return apply(request, updated)

    @api.delete("/settings/keys/{key_id}", summary="Remove a key from .env")
    async def delete_key(request: Request, key_id: str) -> dict[str, Any]:
        try:
            updated = save_key(get_router(request).config, key_id, None, dict(os.environ))
        except SettingsError as error:
            raise fail(error) from None
        return apply(request, updated)

    @api.post("/settings/keys/{key_id}/test", summary="Check that a saved key works")
    async def test_key(request: Request, key_id: str) -> dict[str, Any]:
        try:
            _key_spec(key_id)
        except SettingsError as error:
            raise fail(error) from None
        transport = getattr(request.app.state, "jev_transport", None)
        return await probe_typesafe_key(get_router(request).config, transport=transport)

    return api

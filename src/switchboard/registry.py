"""The model registry: what can actually run on this machine.

Built from three sources, never from a hardcoded list:

* ``C:\\AI\\Config\\models.json`` - the dashboard's own per-model llama-server
  parameters (context size, GPU layers, KV cache types). Read-only.
* the GGUF files themselves - native context, layer count, chat template.
* ``config/models.toml`` - capability priors and seed measurements, the only
  human judgments in the system.

Runtime measurements (load seconds, tokens/sec) come from the store and
override the seeds. Anything unknown stays ``None``: the policy is written to
handle missing numbers rather than to consume invented ones.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from switchboard.config import Config
from switchboard.decisions.schema import DOMAINS, Domain
from switchboard.gguf import GgufInfo, read_gguf_info

ToolCalling = Literal["good", "limited", "none", "unknown"]


@dataclass(slots=True)
class Measurement:
    """A number with its provenance, so nothing looks measured that was guessed."""

    value: float
    source: str
    measured: bool = True
    samples: int = 1

    def describe(self) -> dict[str, Any]:
        return {
            "value": round(self.value, 3),
            "source": self.source,
            "measured": self.measured,
            "samples": self.samples,
        }


@dataclass(slots=True)
class ModelEntry:
    """One logical model Switchboard can route to."""

    id: str
    filename: str
    path: Path
    exists: bool
    enabled: bool = True
    aliases: list[str] = field(default_factory=list)
    role: str = "general"
    summary: str = ""

    # From C:\AI\Config\models.json.
    configured_context: int | None = None
    gpu_layers: int | None = None
    configured: bool = False
    launch_args: dict[str, Any] = field(default_factory=dict)

    # From the GGUF header.
    size_bytes: int = 0
    native_context: int | None = None
    block_count: int | None = None
    architecture: str | None = None
    template_supports_tools: bool | None = None

    # From config/models.toml, overridden by measurements in the store.
    capability: dict[str, float] = field(default_factory=dict)
    tool_calling: ToolCalling = "unknown"
    gen_tps: Measurement | None = None
    prompt_tps: Measurement | None = None
    load_s: Measurement | None = None

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1024**3

    @property
    def context_limit(self) -> int | None:
        """Usable context: what the server is configured to start with."""
        if self.configured_context:
            return self.configured_context
        return self.native_context

    @property
    def supports_tools(self) -> bool:
        """Whether tool definitions survive a request to this model.

        The GGUF chat template is authoritative: llama.cpp renders tools through
        it, so a template that never mentions tools drops them silently. The
        overlay's `tool_calling` only breaks ties when the template is unreadable.
        """
        if self.template_supports_tools is not None:
            return self.template_supports_tools
        return self.tool_calling in {"good", "limited"}

    @property
    def fully_offloaded(self) -> bool:
        """Whether every layer is on the GPU, so host RAM is not on the hot path."""
        if self.gpu_layers is None or self.block_count is None:
            return False
        return self.gpu_layers >= self.block_count

    def host_ram_estimate_gb(self) -> float:
        """Roughly how much of this model must live in system RAM.

        Weights are split by layer, so the CPU-resident share tracks the fraction
        of layers not offloaded. This is an estimate used for a risk warning, not
        an allocation.
        """
        if self.gpu_layers is None or not self.block_count:
            return self.size_gb
        offloaded = min(self.gpu_layers, self.block_count)
        cpu_fraction = max(0.0, 1.0 - offloaded / self.block_count)
        return self.size_gb * cpu_fraction

    def capability_for(self, domain: Domain) -> float:
        """Capability ceiling for a domain, falling back to the general one."""
        if domain in self.capability:
            return self.capability[domain]
        return self.capability.get("general", 1.5)

    def names(self) -> set[str]:
        """Every string a client may use to name this model."""
        stem = self.filename.removesuffix(".gguf")
        return {self.id, self.filename, stem, *self.aliases, str(self.path)}

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "path": str(self.path),
            "exists": self.exists,
            "enabled": self.enabled,
            "role": self.role,
            "summary": self.summary,
            "aliases": self.aliases,
            "size_gb": round(self.size_gb, 2),
            "context_limit": self.context_limit,
            "native_context": self.native_context,
            "configured_context": self.configured_context,
            "gpu_layers": self.gpu_layers,
            "block_count": self.block_count,
            "architecture": self.architecture,
            "fully_offloaded": self.fully_offloaded,
            "supports_tools": self.supports_tools,
            "host_ram_estimate_gb": round(self.host_ram_estimate_gb(), 2),
            "capability": self.capability,
            "gen_tps": self.gen_tps.describe() if self.gen_tps else None,
            "prompt_tps": self.prompt_tps.describe() if self.prompt_tps else None,
            "load_s": self.load_s.describe() if self.load_s else None,
        }


@dataclass(slots=True)
class RegistrySnapshot:
    """The registry as of one build, plus whatever went wrong while building it."""

    models: list[ModelEntry] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    active_model_file: str | None = None
    source_config: Path | None = None

    def by_id(self, model_id: str) -> ModelEntry | None:
        for model in self.models:
            if model.id == model_id:
                return model
        return None

    def resolve(self, name: str | None) -> ModelEntry | None:
        """Find the model a client meant, however they spelled it."""
        if not name:
            return None
        needle = name.strip()
        if not needle:
            return None
        for model in self.models:
            if needle in model.names():
                return model
        lowered = needle.lower()
        for model in self.models:
            if lowered in {item.lower() for item in model.names()}:
                return model
        # llama-server reports its model as a full path or a bare file name.
        tail = Path(needle).name
        for model in self.models:
            if tail and tail.lower() == model.filename.lower():
                return model
        return None

    def routable(self) -> list[ModelEntry]:
        """Models automatic routing may choose."""
        return [model for model in self.models if model.enabled and model.exists]


def _load_models_json(path: Path) -> tuple[dict[str, Any], str | None, list[str]]:
    problems: list[str] = []
    if not path.is_file():
        return {}, None, [f"{path} not found: no per-model llama-server settings available"]
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        return {}, None, [f"{path} unreadable: {error}"]
    models = raw.get("models")
    if not isinstance(models, dict):
        problems.append(f"{path} has no 'models' object")
        models = {}
    active = raw.get("activeModel")
    return models, active if isinstance(active, str) else None, problems


def _load_overlay(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not path.is_file():
        return {}, [f"{path} not found: no capability priors, defaults will be used"]
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        return {}, [f"{path} unreadable: {error}"]
    overlay: dict[str, dict[str, Any]] = {}
    for entry in raw.get("model", []):
        filename = entry.get("file")
        if isinstance(filename, str):
            overlay[filename] = entry
    return overlay, []


def _seed_measurement(data: dict[str, Any], key: str) -> Measurement | None:
    value = data.get(key)
    if not isinstance(value, (int, float)):
        return None
    source = data.get(f"{key}_source", "config/models.toml")
    # A seed whose provenance says "estimate" must not claim to be measured.
    measured = "estimate" not in str(source).lower()
    return Measurement(value=float(value), source=str(source), measured=measured)


def _slugify(filename: str) -> str:
    return filename.removesuffix(".gguf").lower().replace("_", "-").replace(" ", "-")


def build_registry(
    config: Config,
    *,
    gguf_reader: Any = read_gguf_info,
    measurements: dict[str, dict[str, Measurement]] | None = None,
) -> RegistrySnapshot:
    """Build the registry from the live stack configuration and model files."""
    settings, active, problems = _load_models_json(config.stack.models_config)
    overlay, overlay_problems = _load_overlay(config.models_path)
    problems.extend(overlay_problems)

    models_dir = config.stack.models_dir
    on_disk: dict[str, Path] = {}
    if models_dir.is_dir():
        on_disk = {path.name: path for path in sorted(models_dir.glob("*.gguf"))}
    else:
        problems.append(f"{models_dir} is not a directory: no model files found")

    filenames = sorted(set(on_disk) | set(settings) | set(overlay))
    entries: list[ModelEntry] = []
    used_ids: set[str] = set()

    for filename in filenames:
        path = on_disk.get(filename, models_dir / filename)
        exists = filename in on_disk
        raw_settings = settings.get(filename)
        model_settings: dict[str, Any] = raw_settings if isinstance(raw_settings, dict) else {}
        model_overlay = overlay.get(filename, {})

        if not exists:
            # A stale entry in the dashboard's config, or an overlay for a model
            # that was removed. Surfaced rather than silently dropped.
            problems.append(
                f"{filename}: configured but the file is missing from {models_dir}"
            )

        info: GgufInfo = gguf_reader(path) if exists else GgufInfo(error="file not present")
        if info.error and exists:
            problems.append(f"{filename}: GGUF header unreadable ({info.error})")

        model_id = model_overlay.get("id") or _slugify(filename)
        if model_id in used_ids:
            model_id = f"{model_id}-{len(used_ids)}"
        used_ids.add(model_id)

        measured = model_overlay.get("measured", {})
        entry = ModelEntry(
            id=model_id,
            filename=filename,
            path=path,
            exists=exists,
            enabled=bool(model_overlay.get("enabled", True)) and exists,
            aliases=[str(item) for item in model_overlay.get("aliases", [])],
            role=str(model_overlay.get("role", "general")),
            summary=str(model_overlay.get("summary", "")),
            configured_context=model_settings.get("contextSize"),
            gpu_layers=model_settings.get("gpuLayers"),
            configured=bool(model_settings),
            launch_args=dict(model_settings),
            size_bytes=path.stat().st_size if exists else 0,
            native_context=info.context_length,
            block_count=info.block_count,
            architecture=info.architecture,
            template_supports_tools=info.supports_tools,
            capability={
                domain: float(value)
                for domain, value in (model_overlay.get("capability") or {}).items()
                if domain in DOMAINS and isinstance(value, (int, float))
            },
            tool_calling=model_overlay.get("tool_calling", "unknown"),
            gen_tps=_seed_measurement(measured, "gen_tps"),
            prompt_tps=_seed_measurement(measured, "prompt_tps"),
            load_s=_seed_measurement(measured, "load_s"),
        )

        # A configured gpuLayers of 999 means "all of them"; clamp so the host-RAM
        # estimate and the fully-offloaded check stay honest.
        if entry.gpu_layers is not None and entry.block_count:
            entry.gpu_layers = min(entry.gpu_layers, entry.block_count)

        if not entry.capability:
            problems.append(
                f"{filename}: no capability priors in {config.models_path.name}; "
                "routing will treat it as an average general model"
            )
            entry.capability = {domain: 1.5 for domain in DOMAINS}

        entries.append(entry)

    if measurements:
        for entry in entries:
            for field_name, measurement in measurements.get(entry.id, {}).items():
                if field_name in {"gen_tps", "prompt_tps", "load_s"}:
                    setattr(entry, field_name, measurement)

    return RegistrySnapshot(
        models=entries,
        problems=problems,
        active_model_file=active,
        source_config=config.stack.models_config,
    )

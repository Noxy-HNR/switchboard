"""Measuring the numbers the routing policy spends.

The policy trades in seconds: how long a model load costs, against how much
faster or better the next model would answer. Those seconds start life as
estimates - ``policy.default_load_s_per_gb`` and the seed values in
``config/models.toml`` - and an estimate nobody ever checks is a guess with a
decimal point on it. This module replaces them one sample at a time and writes
each sample to the store, so ``build_registry`` picks it up on its next build
and the estimate quietly stops being used.

Two things are measured, and they cost very different amounts:

* **generation speed** - one short non-streaming completion against whatever
  model is resident. llama.cpp reports its own ``timings`` for every request,
  so the number comes from the server's counters rather than from our
  stopwatch. Wall clock is the fallback and is labelled as such, because it
  also contains prompt processing and HTTP and therefore reads low.
* **switch cost** - asking the dashboard to load a model and timing how long
  until llama-server answers with it. On this machine a switch *is* a process
  restart, so it drops whatever llama-server was doing. It is opt-in, it
  refuses while the server is busy or while anything is in flight, and it is
  never part of the default run.

The default run is deliberately tiny: one 64-token completion against the
resident model, and no switches. This machine's GPU is left in a low-power
state on purpose and the user has asked not to hammer it; a benchmark that
takes minutes and heats the card is a benchmark that never gets run, and a
benchmark nobody runs measures nothing. ``--tokens`` and ``--models`` are there
for when more is actually wanted.

Provenance travels with every number - which model, how it was derived, how
many tokens, when, and whether llama-server was already busy. A sample taken
while the server was busy is recorded *with that flag* rather than dropped,
because a slow number with a known reason is evidence and a missing number is
not. Nothing here invents a number: when llama-server is down, or the model
whose speed was asked for is not the one loaded, the report says so and carries
no measurement at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from switchboard.config import Config, PROJECT_ROOT, load_config
from switchboard.decisions.chain import DecisionChain
from switchboard.executor import build_executor
from switchboard.registry import ModelEntry
from switchboard.router import Router
from switchboard.store import Store, iso_utc
from switchboard.upstream import LlamaClient, UpstreamError

logger = logging.getLogger("switchboard.benchmark")

DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

# Boring on purpose: no sampling luck, no reasoning, a predictable answer length,
# and nothing in it that could be mistaken for user data if it lands in the store.
BENCHMARK_PROMPT = "Count from 1 to 40, one number per line."
DEFAULT_TOKENS = 64

# What `metric` names the store accepts. Kept here so a typo fails at the call
# site rather than inside SQLite.
GENERATION_METRICS = ("gen_tps", "prompt_tps")
SWITCH_METRIC = "load_s"

_WALL_CLOCK_NOTE = (
    "llama-server reported no timings, so this is wall clock over the whole "
    "request: it includes prompt processing and HTTP and therefore reads low"
)
_NO_PROMPT_TPS_NOTE = (
    "prompt speed needs llama.cpp timings; wall clock cannot separate prompt "
    "processing from generation, so no number was derived"
)


# --- samples ----------------------------------------------------------------


@dataclass(slots=True)
class MetricValue:
    """One number and the arithmetic behind it."""

    metric: str
    value: float
    derived_from: str
    note: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": round(self.value, 3),
            "derived_from": self.derived_from,
            "note": self.note,
        }


@dataclass(slots=True)
class Sample:
    """One attempt to measure one model, successful or not.

    ``error`` and ``refused`` are kept apart on purpose: a refusal is the tool
    working (it declined to restart a busy server, or to attribute a speed to a
    model that was not loaded), an error is the tool or the stack failing. Only
    errors should make anyone reach for the logs.
    """

    kind: str
    model_id: str
    measured_at: str
    ok: bool = False
    metrics: list[MetricValue] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    server_busy: bool | None = None
    error: str | None = None
    refused: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "model_id": self.model_id,
            "measured_at": self.measured_at,
            "ok": self.ok,
            "metrics": [metric.describe() for metric in self.metrics],
            "context": self.context,
            "notes": self.notes,
            "server_busy": self.server_busy,
            "error": self.error,
            "refused": self.refused,
        }


def _number(value: Any) -> float | None:
    """A positive number, or None. Booleans are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _timing(timings: Any, key: str) -> float | None:
    if not isinstance(timings, dict):
        return None
    return _number(timings.get(key))


def _upstream_of(router_or_client: Any) -> LlamaClient:
    """Accept a Router or a bare LlamaClient.

    The CLI has a router; a test, or anything that only wants one completion,
    has a client. A Router exposes its client as ``.upstream``, a client does
    not, so this needs no isinstance check and works with a fake of either.
    """
    return getattr(router_or_client, "upstream", router_or_client)


# --- generation speed -------------------------------------------------------


async def measure_generation(
    config: Config,
    router_or_client: Any,
    model_id: str,
    *,
    tokens: int = DEFAULT_TOKENS,
) -> dict[str, Any]:
    """Time one short completion against the *resident* model.

    The caller is responsible for the fact that ``model_id`` is what is loaded:
    this function asks llama-server for a completion and llama-server answers
    with whatever it has, so attributing the result to the wrong model would be
    a fabricated number rather than a wrong one. ``run_benchmark`` checks first.
    """
    client = _upstream_of(router_or_client)
    sample = Sample(kind="generation", model_id=model_id, measured_at=iso_utc())
    sample.context.update(
        {
            "tokens_requested": int(tokens),
            "prompt": BENCHMARK_PROMPT,
            "upstream": client.base_url,
            "request_timeout_s": config.upstream.request_timeout_s,
        }
    )

    status = await client.status(force=True)
    sample.server_busy = status.busy
    sample.context["upstream_model"] = status.model_id
    if not status.healthy:
        sample.refused = (
            f"llama-server at {client.base_url} is not answering: "
            f"{status.error or 'no /health response'}"
        )
        return sample.describe()
    if status.busy:
        sample.notes.append(
            "llama-server was already generating when this sample started; the "
            "number is low and is flagged as busy in the store"
        )
    elif status.busy is None:
        sample.notes.append("/slots did not answer, so it is unknown whether the server was busy")

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": BENCHMARK_PROMPT}],
        "max_tokens": int(tokens),
        "temperature": 0.0,
        "stream": False,
    }

    started = time.perf_counter()
    try:
        response = await client.chat_completion(body)
    except UpstreamError as error:
        sample.context["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        sample.error = f"the completion failed: {error}"
        return sample.describe()
    elapsed_s = time.perf_counter() - started
    sample.context["elapsed_ms"] = round(elapsed_s * 1000, 1)

    if response.status_code >= 400:
        sample.error = (
            f"/v1/chat/completions returned {response.status_code}: {response.text[:200]}"
        )
        return sample.describe()

    try:
        payload = response.json()
    except ValueError:
        sample.error = "llama-server answered the completion with something that is not JSON"
        return sample.describe()
    if not isinstance(payload, dict):
        sample.error = f"llama-server answered with a {type(payload).__name__}, not an object"
        return sample.describe()

    raw_usage = payload.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    timings = payload.get("timings")
    prompt_tokens = _count(usage.get("prompt_tokens"))
    completion_tokens = _count(usage.get("completion_tokens"))
    # timings.predicted_n is what the server actually generated; usage is what it
    # reported to the OpenAI shape. They normally agree, and when they do not the
    # server's own counter is the one that produced predicted_per_second.
    predicted_n = _count(timings.get("predicted_n")) if isinstance(timings, dict) else None
    sample.context.update(
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "predicted_n": predicted_n,
            "has_timings": isinstance(timings, dict),
            "finish_reason": _finish_reason(payload),
        }
    )

    generated = predicted_n if predicted_n is not None else completion_tokens
    gen_tps = _timing(timings, "predicted_per_second")
    if gen_tps is not None:
        sample.metrics.append(MetricValue("gen_tps", gen_tps, "timings"))
    elif generated and elapsed_s > 0:
        sample.metrics.append(
            MetricValue("gen_tps", generated / elapsed_s, "wall_clock", note=_WALL_CLOCK_NOTE)
        )
    else:
        sample.notes.append(
            "no timings and no token count: generation speed could not be derived"
        )

    prompt_tps = _timing(timings, "prompt_per_second")
    if prompt_tps is not None:
        sample.metrics.append(MetricValue("prompt_tps", prompt_tps, "timings"))
    else:
        sample.notes.append(_NO_PROMPT_TPS_NOTE)

    sample.ok = bool(sample.metrics)
    if not sample.ok:
        sample.error = "the completion carried neither timings nor token counts"
    return sample.describe()


def _finish_reason(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        return reason if isinstance(reason, str) else None
    return None


# --- switch cost ------------------------------------------------------------


async def measure_switch(config: Config, router: Router, model_id: str) -> dict[str, Any]:
    """Load a model and time how long until llama-server answers with it.

    This restarts llama-server. Every reason not to is checked first and turned
    into a refusal, because the cost of being wrong here is somebody else's
    half-generated answer: the executor must be available, the server must be up
    and demonstrably idle, and nothing may be in flight through Switchboard.
    ``/slots`` answering "unknown" counts as a reason not to.
    """
    sample = Sample(kind="switch", model_id=model_id, measured_at=iso_utc())

    snapshot = router.registry()
    target = snapshot.by_id(model_id) or snapshot.resolve(model_id)
    if target is None:
        sample.refused = f"{model_id!r} is not a model in the registry"
        return sample.describe()

    sample.model_id = target.id
    sample.context.update(
        {
            "filename": target.filename,
            "size_gb": round(target.size_gb, 2),
            "executor": router.executor.name,
        }
    )
    if not target.exists:
        sample.refused = f"{target.filename} is not present in {target.path.parent}"
        return sample.describe()

    available, reason = await router.executor.available()
    if not available:
        sample.refused = f"the {router.executor.name} executor cannot switch: {reason}"
        return sample.describe()

    status = await router.upstream.status(force=True)
    sample.server_busy = status.busy
    if not status.healthy:
        sample.refused = (
            f"llama-server at {router.upstream.base_url} is not answering: "
            f"{status.error or 'no /health response'}"
        )
        return sample.describe()
    if status.busy is None:
        sample.refused = (
            "/slots did not answer, so it is unknown whether llama-server is "
            "generating; a switch restarts it and would drop that work"
        )
        return sample.describe()
    if status.busy:
        sample.refused = (
            "llama-server is generating for someone: a switch restarts it and "
            "would drop that work"
        )
        return sample.describe()
    if router.inflight_count:
        sample.refused = (
            f"{router.inflight_count} request(s) are in flight through Switchboard"
        )
        return sample.describe()

    resident = snapshot.resolve(status.model_path or status.model_id)
    from_model = resident.id if resident else status.model_id
    sample.context["from_model"] = from_model

    result = await router.executor.switch(target, from_model=from_model)
    sample.context["switch"] = result.describe()
    if not result.ok:
        sample.error = result.error or "the switch failed without saying why"
        return sample.describe()
    if result.noop:
        sample.refused = f"{target.id} was already resident; nothing was loaded to time"
        return sample.describe()
    if result.load_ms is None:
        sample.error = "the switch reported no load time, so there is nothing to record"
        return sample.describe()

    sample.metrics.append(
        MetricValue(
            SWITCH_METRIC,
            result.load_ms / 1000.0,
            "switch",
            note=(
                "wall clock from the load request until llama-server answered "
                "with this model; there is no server-side counter for a restart"
            ),
        )
    )
    sample.ok = True
    return sample.describe()


# --- orchestration ----------------------------------------------------------


def _build_router(config: Config) -> tuple[Router, Callable[[], Awaitable[None]]]:
    """A router with no decision backends: a benchmark measures, it never routes.

    An empty chain means no Jev client, no adapter and no cloud call is even
    constructed, so `switchboard benchmark` cannot accidentally spend money or
    warm up a backend it is not measuring.
    """
    store: Store | None
    try:
        store = Store(config.db_file, preview_chars=config.privacy.preview_chars)
    except Exception as error:  # noqa: BLE001 - a benchmark that cannot save is still worth running
        logger.warning("could not open the store at %s: %s", config.db_file, error)
        store = None

    upstream = LlamaClient(config)
    executor = build_executor(config, upstream)
    router = Router(
        config,
        upstream=upstream,
        chain=DecisionChain(config, []),
        executor=executor,
        store=store,
    )

    async def close() -> None:
        aclose = getattr(executor, "aclose", None)
        if aclose is not None:
            await aclose()
        await upstream.aclose()
        if store is not None:
            store.close()

    return router, close


def _refusal(kind: str, model_id: str, reason: str) -> dict[str, Any]:
    return Sample(kind=kind, model_id=model_id, measured_at=iso_utc(), refused=reason).describe()


def _record(router: Router, sample: dict[str, Any], report: dict[str, Any]) -> None:
    """Write a sample's numbers to the store, provenance and all.

    This is the only writer. ``measure_generation`` and ``measure_switch``
    deliberately do not touch the store, and the switch goes through the
    executor rather than ``Router.switch_to`` so that one measurement cannot
    land in the table twice and skew the EWMA behind it.
    """
    store = router.store
    metrics = sample.get("metrics") or []
    if store is None:
        if metrics:
            report["problems"].append(
                "no store is open, so these measurements were not saved and the "
                "registry will keep using its estimates"
            )
        return

    detail = (sample.get("context") or {}).get("switch")
    if sample.get("kind") == "switch" and isinstance(detail, dict):
        # The dashboard restarted llama-server; the switches table should say why.
        with contextlib.suppress(Exception):
            store.record_switch(
                from_model=detail.get("from_model"),
                to_model=detail.get("to_model"),
                executor=detail.get("executor"),
                trigger="manual",
                reason="benchmark: measuring load time",
                ok=bool(detail.get("ok")),
                load_ms=detail.get("load_ms"),
                error=detail.get("error"),
            )

    for metric in metrics:
        context = {
            "kind": sample.get("kind"),
            "derived_from": metric["derived_from"],
            "server_busy": sample.get("server_busy"),
            **(sample.get("context") or {}),
        }
        try:
            store.record_measurement(
                model_id=sample["model_id"],
                metric=metric["metric"],
                value=metric["value"],
                source=f"benchmark:{metric['derived_from']}",
                context=context,
                ts=sample.get("measured_at"),
            )
        except Exception as error:  # noqa: BLE001 - one bad row must not lose the rest
            report["problems"].append(
                f"could not record {metric['metric']} for {sample['model_id']}: {error}"
            )
            continue
        report["recorded"].append(
            {
                "model_id": sample["model_id"],
                "metric": metric["metric"],
                "value": metric["value"],
                "derived_from": metric["derived_from"],
                "server_busy": sample.get("server_busy"),
            }
        )
        # The registry caches its snapshot; a new measurement has to invalidate it
        # or this process would keep routing on the estimate it just replaced.
        router.invalidate_registry()


async def run_benchmark(
    config: Config | None = None,
    *,
    models: Sequence[str] | None = None,
    measure_switches: bool = False,
    tokens: int = DEFAULT_TOKENS,
    router: Router | None = None,
) -> dict[str, Any]:
    """Measure the requested models and record what was measured.

    By default that is the resident model and nothing else: one completion, no
    restart. ``measure_switches`` adds a real load per model, which restarts
    llama-server, and is the only way to get a generation number for a model
    that is not currently loaded.

    ``router`` is for callers that already have one (and for tests). When it is
    omitted a private one is built and closed here.
    """
    config = config or load_config()
    if router is not None:
        return await _run(
            config, router, models=models, measure_switches=measure_switches, tokens=tokens
        )

    built, close = _build_router(config)
    try:
        return await _run(
            config, built, models=models, measure_switches=measure_switches, tokens=tokens
        )
    finally:
        await close()


async def _run(
    config: Config,
    router: Router,
    *,
    models: Sequence[str] | None,
    measure_switches: bool,
    tokens: int,
) -> dict[str, Any]:
    status = await router.upstream.status(force=True)
    snapshot = router.registry(refresh=True)
    resident = snapshot.resolve(status.model_path or status.model_id) if status.healthy else None

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": {
            "upstream": status.describe(),
            "resident_model": resident.id if resident else None,
            "models": [model.id for model in snapshot.routable()],
        },
        "config": {
            "mode": config.routing.mode,
            "tokens": int(tokens),
            "prompt": BENCHMARK_PROMPT,
            "measure_switches": bool(measure_switches),
            "executor": router.executor.name,
            "store": str(config.db_file) if router.store is not None else None,
        },
        "refused": None,
        "samples": [],
        "recorded": [],
        "problems": list(snapshot.problems),
    }

    if not status.healthy:
        report["refused"] = (
            f"llama-server at {config.upstream.base_url} is not answering: "
            f"{status.error or 'no /health response'}. Nothing was measured, and "
            "nothing was written."
        )
        return report

    targets, unknown = _targets(snapshot, models, resident)
    for name in unknown:
        report["samples"].append(
            _refusal("generation", name, f"{name!r} is not a model in the registry")
        )
    if not targets:
        report["refused"] = (
            "none of the requested models are in the registry"
            if models
            else (
                f"llama-server reports {status.model_id!r}, which matches no model in "
                "the registry, so a measurement could not be attributed to anything"
            )
        )
        return report

    for target in targets:
        if measure_switches:
            switch = await measure_switch(config, router, target.id)
            report["samples"].append(switch)
            _record(router, switch, report)
            # The switch restarted llama-server: every fact above is now stale.
            status = await router.upstream.status(force=True)
            snapshot = router.registry(refresh=True)
            resident = (
                snapshot.resolve(status.model_path or status.model_id)
                if status.healthy
                else None
            )

        if resident is None or resident.id != target.id:
            report["samples"].append(
                _refusal("generation", target.id, _not_resident(target, status, measure_switches))
            )
            continue

        generation = await measure_generation(config, router, target.id, tokens=tokens)
        report["samples"].append(generation)
        _record(router, generation, report)

    return report


def _targets(
    snapshot: Any, models: Sequence[str] | None, resident: ModelEntry | None
) -> tuple[list[ModelEntry], list[str]]:
    """The models to measure, and the names that matched nothing."""
    if not models:
        return ([resident] if resident is not None else []), []

    targets: list[ModelEntry] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for name in models:
        entry = snapshot.by_id(str(name)) or snapshot.resolve(str(name))
        if entry is None:
            unknown.append(str(name))
            continue
        if entry.id not in seen:
            seen.add(entry.id)
            targets.append(entry)
    return targets, unknown


def _not_resident(target: ModelEntry, status: Any, measure_switches: bool) -> str:
    """Why a generation sample was skipped, in terms the reader can act on."""
    if not status.healthy:
        return (
            f"llama-server did not come back after loading {target.id}: "
            f"{status.error or 'no /health response'}"
        )
    if measure_switches:
        return (
            f"llama-server is running {status.model_id!r} rather than {target.id}, "
            "so a speed measured now would belong to the wrong model"
        )
    return (
        f"{target.id} is not the resident model, and a speed measured against "
        f"{status.model_id!r} would belong to the wrong model. Re-run with "
        "--switches to load it first."
    )


# --- reporting --------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    """The same numbers, for a human deciding whether to trust them."""
    machine = report["machine"]
    settings = report["config"]
    lines = [
        "# Switchboard benchmark",
        "",
        f"Generated {report['generated_at']} on this machine.",
        "",
        f"- resident model: `{machine['resident_model'] or 'unknown'}`",
        f"- models available: {', '.join(machine['models']) or 'none'}",
        f"- prompt: `{settings['prompt']}` at max_tokens={settings['tokens']}",
        f"- switch measurement: {'on (llama-server was restarted)' if settings['measure_switches'] else 'off'}",
        "",
    ]

    if report.get("refused"):
        lines += [
            "## Nothing was measured",
            "",
            report["refused"],
            "",
        ]
        return "\n".join(lines)

    lines += [
        "## Samples",
        "",
        "| model | kind | metric | value | derived from | busy | tokens |",
        "| --- | --- | --- | ---: | --- | --- | ---: |",
    ]
    for sample in report["samples"]:
        busy = {True: "yes", False: "no", None: "unknown"}[sample.get("server_busy")]
        context = sample.get("context") or {}
        tokens = context.get("predicted_n") or context.get("completion_tokens") or "-"
        if not sample["metrics"]:
            why = sample.get("refused") or sample.get("error") or "no number derived"
            lines.append(
                f"| {sample['model_id']} | {sample['kind']} | - | - | _{why}_ | {busy} | - |"
            )
            continue
        for metric in sample["metrics"]:
            unit = "s" if metric["metric"] == "load_s" else "tok/s"
            lines.append(
                f"| {sample['model_id']} | {sample['kind']} | {metric['metric']} | "
                f"{metric['value']} {unit} | {metric['derived_from']} | {busy} | {tokens} |"
            )

    recorded = report.get("recorded") or []
    lines += [
        "",
        "## Written to the store",
        "",
    ]
    if recorded:
        lines.append(
            f"{len(recorded)} measurement(s) recorded; the registry picks them up on its next build."
        )
        lines.append("")
        for row in recorded:
            flag = " **(server was busy - discount this one)**" if row["server_busy"] else ""
            lines.append(
                f"- `{row['model_id']}` {row['metric']} = {round(row['value'], 3)} "
                f"({row['derived_from']}){flag}"
            )
    else:
        lines.append("Nothing was recorded: no sample produced a number.")

    notes = [
        (sample["model_id"], note)
        for sample in report["samples"]
        for note in sample.get("notes") or []
    ]
    if notes:
        lines += ["", "## Notes", ""]
        lines += [f"- `{model_id}`: {note}" for model_id, note in notes]

    if report.get("problems"):
        lines += ["", "## Problems", ""]
        lines += [f"- {problem}" for problem in report["problems"]]

    lines += [
        "",
        "_A number derived from `timings` comes from llama.cpp's own counters. One "
        "derived from `wall_clock` includes prompt processing and HTTP and reads "
        "low. A sample taken while the server was busy is kept, and flagged, "
        "rather than dropped._",
        "",
    ]
    return "\n".join(lines)


def run_benchmark_cli(
    *,
    models: Sequence[str] | None = None,
    measure_switches: bool = False,
    tokens: int = DEFAULT_TOKENS,
    out: str | None = None,
) -> int:
    """Entry point for `switchboard benchmark`. Returns a process exit code."""
    config = load_config()
    report = asyncio.run(
        run_benchmark(
            config,
            models=models,
            measure_switches=measure_switches,
            tokens=tokens,
        )
    )

    directory = Path(out) if out else DEFAULT_REPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = directory / f"benchmark-{stamp}.json"
    md_path = directory / f"benchmark-{stamp}.md"
    markdown = render_markdown(report)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nWrote {json_path}\n      {md_path}")

    # A refusal is a failure of the stack, not of the benchmark, but the caller
    # asked for measurements and got none, so it should not look like success.
    if report.get("refused") or not report.get("recorded"):
        return 1
    return 0

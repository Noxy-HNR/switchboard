"""The dashboard's settings and keys: what is saved where, what applies live,
and what never leaves the process.

Every test points the config at a temporary project, so nothing here can write
the real config/local.toml or .env.
"""

from __future__ import annotations

import tomllib
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
from conftest import FakeExecutor, FakeLlama
from test_jev_backend import TASK_BODY
from test_proxy import Harness

from switchboard.config import Config, load_config
from switchboard.decisions.chain import DecisionChain
from switchboard.decisions.rules import RulesBackend
from switchboard.settings import write_env_value, write_toml

SECRET = "ts-live-SECRETVALUE-0123456789abcdef"
BASE_TOML = '[routing]\nmode = "active"\n\n[routing.guards]\nhysteresis_seconds = 12.0\n'


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "config" / "switchboard.toml").write_text(BASE_TOML, encoding="utf-8")
    return root


@pytest.fixture
async def settings_harness(
    config: Config, project: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Harness]:
    for name in ("TYPESAFE_API_KEY", "SWITCHBOARD_ROUTING__MODE"):
        monkeypatch.delenv(name, raising=False)
    config.config_path = project / "config" / "switchboard.toml"
    async with Harness(config, FakeLlama(), FakeExecutor()) as harness:
        built: list[Config] = []

        def chain_factory(new_config: Config, upstream: Any) -> DecisionChain:
            built.append(new_config)
            return DecisionChain(new_config, [RulesBackend(new_config)])

        harness.app.state.chain_factory = chain_factory
        harness.app.state.config = config
        harness.built_chains = built  # type: ignore[attr-defined]
        yield harness
        await harness.router.aclose_retired()


def local_toml(project: Path) -> dict[str, Any]:
    path = project / "config" / "local.toml"
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def setting(payload: dict[str, Any], path: str) -> dict[str, Any]:
    return next(row for row in payload["settings"] if row["path"] == path)


# --- settings --------------------------------------------------------------------


async def test_a_saved_setting_applies_live_and_persists(settings_harness: Harness, project: Path) -> None:
    response = await settings_harness.client.put("/settings", json={"values": {"routing.mode": "shadow"}})

    assert response.status_code == 200
    assert settings_harness.router.config.routing.mode == "shadow"
    assert (await settings_harness.client.get("/status")).json()["mode"] == "shadow"
    assert (await settings_harness.client.get("/")).json()["mode"] == "shadow"
    assert local_toml(project) == {"routing": {"mode": "shadow"}}
    row = setting(response.json(), "routing.mode")
    assert row["value"] == "shadow" and row["saved"] is True and row["file_value"] == "active"
    assert settings_harness.built_chains, "the decision chain is rebuilt with the new settings"


async def test_an_invalid_value_changes_nothing(settings_harness: Harness, project: Path) -> None:
    response = await settings_harness.client.put(
        "/settings", json={"values": {"routing.guards.min_quality_gain": 5}}
    )

    assert response.status_code == 422
    assert "min_quality_gain" in response.json()["detail"]
    assert settings_harness.router.config.routing.guards.min_quality_gain == pytest.approx(0.08)
    assert local_toml(project) == {}


@pytest.mark.parametrize(
    ("values", "status"),
    [
        ({"server.port": 9000}, 404),  # not changeable from the dashboard
        ({"decisions.backends": ["rules", "nope"]}, 422),
        ({"decisions.jev.enabled": "sometimes"}, 422),
        ({}, 422),
    ],
)
async def test_what_the_dashboard_cannot_save(settings_harness: Harness, values: dict[str, Any], status: int) -> None:
    response = await settings_harness.client.put("/settings", json={"values": values})
    assert response.status_code == status


async def test_a_setting_pinned_by_the_environment_is_refused(
    settings_harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SWITCHBOARD_ROUTING__MODE", "active")
    listed = setting((await settings_harness.client.get("/settings")).json(), "routing.mode")
    assert listed["pinned_by"] == "SWITCHBOARD_ROUTING__MODE"

    response = await settings_harness.client.put("/settings", json={"values": {"routing.mode": "off"}})
    assert response.status_code == 409
    assert "SWITCHBOARD_ROUTING__MODE" in response.json()["detail"]


async def test_backends_keep_their_order_and_the_floor(settings_harness: Harness, project: Path) -> None:
    await settings_harness.client.put("/settings", json={"values": {"decisions.backends": ["rules", "logit", "jev"]}})

    assert settings_harness.router.config.decisions.backends == ["jev", "logit", "rules"]
    assert local_toml(project)["decisions"]["backends"] == ["jev", "logit", "rules"]


async def test_jev_tristate_maps_on_and_off(settings_harness: Harness) -> None:
    await settings_harness.client.put("/settings", json={"values": {"decisions.jev.enabled": "off"}})
    assert settings_harness.router.config.decisions.jev.enabled is False
    await settings_harness.client.put("/settings", json={"values": {"decisions.jev.enabled": "auto"}})
    assert settings_harness.router.config.decisions.jev.enabled == "auto"


async def test_saving_the_file_value_removes_the_override(settings_harness: Harness, project: Path) -> None:
    await settings_harness.client.put("/settings", json={"values": {"routing.guards.hysteresis_seconds": 30}})
    assert local_toml(project) == {"routing": {"guards": {"hysteresis_seconds": 30.0}}}

    await settings_harness.client.put("/settings", json={"values": {"routing.guards.hysteresis_seconds": 12}})
    assert local_toml(project) == {}


async def test_reset_goes_back_to_the_file_value(settings_harness: Harness, project: Path) -> None:
    await settings_harness.client.put("/settings", json={"values": {"routing.mode": "off"}})
    response = await settings_harness.client.delete("/settings/routing.mode")

    assert response.status_code == 200
    assert settings_harness.router.config.routing.mode == "active"
    assert local_toml(project) == {}


def test_load_config_layers_local_settings_between_the_file_and_the_environment(project: Path) -> None:
    path = project / "config" / "switchboard.toml"
    write_toml(project / "config" / "local.toml", {"routing": {"mode": "shadow", "guards": {"min_dwell_seconds": 5.0}}})

    config = load_config(path, environ={})
    assert config.routing.mode == "shadow"
    assert config.routing.guards.min_dwell_seconds == 5.0
    assert config.routing.guards.hysteresis_seconds == 12.0  # from switchboard.toml

    assert load_config(path, environ={"SWITCHBOARD_ROUTING__MODE": "off"}).routing.mode == "off"


# --- keys ---------------------------------------------------------------------------


async def test_a_key_is_saved_to_dotenv_and_never_returned(settings_harness: Harness, project: Path) -> None:
    env_file = project / ".env"
    env_file.write_text("# mine\nOTHER_THING=keep-me\n", encoding="utf-8")

    response = await settings_harness.client.put("/settings/keys/typesafe", json={"value": SECRET})

    assert response.status_code == 200
    assert SECRET not in response.text
    key = next(row for row in response.json()["keys"] if row["id"] == "typesafe")
    assert key["set"] is True and key["source"] == ".env" and key["hint"] == "…cdef"
    assert env_file.read_text(encoding="utf-8") == f"# mine\nOTHER_THING=keep-me\nTYPESAFE_API_KEY={SECRET}\n"
    assert settings_harness.router.config.typesafe_api_key == SECRET
    assert SECRET not in (await settings_harness.client.get("/settings")).text
    assert not (project / "config" / "local.toml").exists(), "a key never goes into the settings file"


async def test_a_key_can_be_replaced_and_removed(settings_harness: Harness, project: Path) -> None:
    await settings_harness.client.put("/settings/keys/typesafe", json={"value": SECRET})
    await settings_harness.client.put("/settings/keys/typesafe", json={"value": "ts-second-key-000011112222"})
    assert (project / ".env").read_text(encoding="utf-8").count("TYPESAFE_API_KEY=") == 1

    response = await settings_harness.client.delete("/settings/keys/typesafe")
    assert response.status_code == 200
    assert "TYPESAFE_API_KEY" not in (project / ".env").read_text(encoding="utf-8")
    assert settings_harness.router.config.typesafe_api_key is None


@pytest.mark.parametrize("value", ["", "short", "has a space in it", 'quote"inside12', "tab\tinside1234", "é-not-ascii-12345"])
async def test_a_malformed_key_is_refused_and_nothing_is_written(
    settings_harness: Harness, project: Path, value: str
) -> None:
    response = await settings_harness.client.put("/settings/keys/typesafe", json={"value": value})

    assert response.status_code == 422
    assert not (project / ".env").exists()


async def test_bad_key_payload_is_never_reflected_in_validation_response(
    settings_harness: Harness,
) -> None:
    response = await settings_harness.client.put(
        "/settings/keys/typesafe", json={"value": {"pasted": SECRET}}
    )
    assert response.status_code == 422
    assert SECRET not in response.text


async def test_a_key_set_in_the_process_environment_is_pinned(
    settings_harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    listed = (await settings_harness.client.get("/settings")).json()["keys"][0]
    assert listed["pinned"] is True and listed["source"] == "environment"

    response = await settings_harness.client.put("/settings/keys/typesafe", json={"value": "ts-other-key-0000111122"})
    assert response.status_code == 409


async def test_an_unknown_key_is_404(settings_harness: Harness) -> None:
    assert (await settings_harness.client.put("/settings/keys/nope", json={"value": SECRET})).status_code == 404


async def test_a_cross_origin_page_cannot_write_settings(settings_harness: Harness) -> None:
    response = await settings_harness.client.put(
        "/settings/keys/typesafe", json={"value": SECRET}, headers={"Origin": "http://evil.example"}
    )
    assert response.status_code == 403


# --- testing a key -----------------------------------------------------------------


def jev_answering(status: int, body: dict[str, Any] | None = None) -> httpx2.MockTransport:
    return httpx2.MockTransport(lambda request: httpx2.Response(status, json=body or {"error": "no"}))


@pytest.mark.parametrize(
    ("status", "result", "ok"),
    [(200, "working", True), (401, "rejected", False), (429, "rate_limited", True)],
)
async def test_the_key_test_reports_what_typesafe_said(
    settings_harness: Harness, status: int, result: str, ok: bool
) -> None:
    await settings_harness.client.put("/settings/keys/typesafe", json={"value": SECRET})
    settings_harness.app.state.jev_transport = jev_answering(status, TASK_BODY if status == 200 else None)

    response = await settings_harness.client.post("/settings/keys/typesafe/test", json={})

    assert response.status_code == 200
    assert response.json()["result"] == result and response.json()["ok"] is ok
    assert SECRET not in response.text


async def test_the_key_test_needs_a_key_and_respects_the_cloud_switch(settings_harness: Harness) -> None:
    assert (await settings_harness.client.post("/settings/keys/typesafe/test", json={})).json()["result"] == "not_set"

    await settings_harness.client.put("/settings/keys/typesafe", json={"value": SECRET})
    await settings_harness.client.put("/settings", json={"values": {"privacy.allow_cloud": False}})
    settings_harness.app.state.jev_transport = jev_answering(200, TASK_BODY)
    assert (await settings_harness.client.post("/settings/keys/typesafe/test", json={})).json()["result"] == "disabled"


# --- files and lifecycle -------------------------------------------------------------


def test_env_writes_keep_every_other_line(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("# comment\nA=1\nTYPESAFE_API_KEY=old\nB=2\nTYPESAFE_API_KEY=dup\n", encoding="utf-8")

    write_env_value(path, "TYPESAFE_API_KEY", "new-value-123")
    assert path.read_text(encoding="utf-8") == "# comment\nA=1\nTYPESAFE_API_KEY=new-value-123\nB=2\n"

    write_env_value(path, "TYPESAFE_API_KEY", None)
    assert path.read_text(encoding="utf-8") == "# comment\nA=1\nB=2\n"

    write_env_value(tmp_path / "absent.env", "TYPESAFE_API_KEY", None)
    assert not (tmp_path / "absent.env").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ACL")
def test_saved_key_is_not_readable_by_other_local_users(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    write_env_value(path, "TYPESAFE_API_KEY", SECRET)

    acl = subprocess.run(["icacls", str(path)], capture_output=True, text=True, check=True).stdout
    assert "Authenticated Users" not in acl
    assert "BUILTIN\\Users:" not in acl
    assert "SYSTEM:" in acl
    assert path.read_text(encoding="utf-8").endswith(f"TYPESAFE_API_KEY={SECRET}\n")


def test_local_toml_round_trips(tmp_path: Path) -> None:
    data = {
        "routing": {"mode": "shadow", "guards": {"hysteresis_seconds": 1.5, "never_switch_when_busy": False}},
        "decisions": {"backends": ["jev", "rules"], "jev": {"model": 'jev-"1"', "enabled": "auto"}},
    }
    write_toml(tmp_path / "local.toml", data)
    assert tomllib.loads((tmp_path / "local.toml").read_text(encoding="utf-8")) == data


async def test_a_replaced_chain_is_closed_later_not_at_once(config: Config) -> None:
    class Tracked(DecisionChain):
        closed = False

        async def aclose(self) -> None:
            Tracked.closed = True

    async with Harness(config, FakeLlama(), FakeExecutor()) as harness:
        old = Tracked(config, [RulesBackend(config)])
        harness.router.reconfigure(config, old)
        harness.router.reconfigure(config, DecisionChain(config, [RulesBackend(config)]))
        assert Tracked.closed is False, "a decision may still be running on the old chain"
        await harness.router.aclose_retired()
        assert Tracked.closed is True


# --- models, usage and history ----------------------------------------------------


async def test_models_are_listed_and_exclusions_apply_live(settings_harness: Harness, project: Path) -> None:
    row = setting((await settings_harness.client.get("/settings")).json(), "routing.auto_exclude")
    assert {"fast-9b", "deep-27b", "med-27b"} <= set(row["choices"])
    assert "GB" in row["choice_details"]["deep-27b"]

    response = await settings_harness.client.put(
        "/settings", json={"values": {"routing.auto_exclude": ["deep-27b", "deep-27b", " "]}}
    )

    assert response.status_code == 200
    assert settings_harness.router.config.routing.auto_exclude == ["deep-27b"]
    assert local_toml(project) == {"routing": {"auto_exclude": ["deep-27b"]}}


async def test_the_key_row_reports_the_last_days_jev_usage(settings_harness: Harness) -> None:
    """Only calls that went out count: skips (no key yet, breaker open) never reached TypeSafe."""
    settings_harness.router.store.provider_stats = lambda hours=24: {  # type: ignore[method-assign]
        "jev": {
            "attempts": 15,
            "ok": 2,
            "error_kinds": {"unconfigured": 11, "breaker": 1, "auth": 1},
            "est_cost_usd": 0.00004,
        }
    }

    key = (await settings_harness.client.get("/settings")).json()["keys"][0]

    assert key["usage_24h"] == {"calls": 3, "answered": 2, "failures": {"auth": 1}, "est_cost_usd": 0.00004}


async def test_history_is_pruned_with_the_retention_the_settings_say(settings_harness: Harness) -> None:
    from switchboard.app import prune_history

    seen: list[int] = []
    settings_harness.router.store.prune = lambda days: seen.append(days) or {"routes": 0}  # type: ignore[method-assign]

    await settings_harness.client.put("/settings", json={"values": {"observability.retention_days": 7}})
    prune_history(settings_harness.router)

    assert seen == [7]

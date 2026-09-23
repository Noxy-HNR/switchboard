"""Tests for the dashboard page.

The dashboard is the one artefact in this project that cannot be checked by
running it: there is no llama-server in CI, no browser, and the page only says
anything interesting once real traffic has gone through. So the tests assert
the properties that make it *able* to work rather than what it looks like.

Four of them matter:

* it is one file, small enough to serve from memory without thinking about it;
* it references nothing external, because the machine it runs on may have no
  network and a dashboard that needs a CDN fails exactly when it is needed;
* the app really serves it, exercised through the real ASGI app rather than a
  hand-rolled stub of the route;
* every API path it polls is still mounted, which is the assertion that fails
  when someone renames an endpoint and the page silently goes blank.

The escaping rule is checked too, by banning ``innerHTML`` outright. Model ids
and decision reasons reach the page from request bodies and GGUF headers, so
the page treats them as data; a grep is a blunt instrument but it is the one
that keeps holding as the file is edited.
"""

from __future__ import annotations

import re

import httpx
import pytest

from switchboard.app import DASHBOARD_FILE, create_app
from switchboard.config import Config

# The page is served by reading the whole file into memory on every request,
# which is fine for something hand-written and not fine for something generated.
MAX_BYTES = 120 * 1024

# Paths the page polls. Spelled exactly as the page spells them, so a changed
# query string on either side shows up here.
POLLED_PATHS = (
    "/status",
    "/providers?probe=false",
    "/routes?limit=50",
    "/stats?hours=24",
    "/registry",
)

# src="http://...", href='https://...' and protocol-relative //cdn... forms.
EXTERNAL_REF = re.compile(r"""\b(?:src|href)\s*=\s*["']?\s*(?:https?:)?//""", re.IGNORECASE)


@pytest.fixture(scope="module")
def page() -> str:
    return DASHBOARD_FILE.read_text(encoding="utf-8")


def test_dashboard_file_exists_and_is_small() -> None:
    assert DASHBOARD_FILE.is_file(), f"{DASHBOARD_FILE} is missing"
    size = DASHBOARD_FILE.stat().st_size
    assert size < MAX_BYTES, f"dashboard.html is {size} bytes, over the {MAX_BYTES} budget"


def test_dashboard_has_no_external_references(page: str) -> None:
    """No CDN, no web font, no remote script: the page must work offline."""
    found = EXTERNAL_REF.findall(page)
    assert not found, f"dashboard.html references external resources: {found}"


def test_dashboard_is_self_contained(page: str) -> None:
    """Styles and behaviour are inline, so there is nothing else to install."""
    assert "<style>" in page
    assert "<script>" in page
    assert "<link" not in page.lower(), "a <link> element means a second file to serve"


def test_dashboard_never_assigns_innerhtml(page: str) -> None:
    """Everything from the API goes in as text, never as markup.

    The check is for an *assignment*, not the word: the file is allowed to
    explain in a comment why it never uses innerHTML.
    """
    assignments = re.findall(r"innerHTML\s*(?:\+)?=", page)
    assert not assignments, (
        "dashboard.html assigns innerHTML; API values are attacker-influenced in "
        "principle and must go through textContent"
    )


@pytest.mark.parametrize("path", POLLED_PATHS)
def test_dashboard_references_every_api_path_it_needs(page: str, path: str) -> None:
    assert path in page, f"dashboard.html no longer polls {path}"


@pytest.mark.parametrize("path", POLLED_PATHS)
def test_polled_paths_are_actually_mounted(path: str) -> None:
    """The other half of the previous test: the API still offers what the page asks for.

    Read from the OpenAPI schema rather than ``app.routes``: Starlette keeps an
    included router as one composite route instead of flattening its children,
    so walking ``app.routes`` would see none of these paths.
    """
    app = create_app(Config())
    documented = set(app.openapi()["paths"])
    assert path.split("?")[0] in documented


async def test_dashboard_is_served() -> None:
    """Through the real app. Lifespan never runs, so nothing contacts the stack."""
    app = create_app(Config())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://switchboard") as client:
        response = await client.get("/dashboard")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert 'id="dashboard"' in body
    assert 'id="mode-badge"' in body
    assert 'id="routes-body"' in body
    assert "no routes recorded yet" in body, "the empty state must ship with the page"


def test_dashboard_has_a_settings_panel_wired_to_a_mounted_api(page: str) -> None:
    """The settings panel reads and writes /settings; both halves must stay in step."""
    assert 'id="settings-btn"' in page and 'id="settings-wrap"' in page
    assert 'var SETTINGS_API = "/settings";' in page
    documented = set(create_app(Config()).openapi()["paths"])
    assert {"/settings", "/settings/{path}", "/settings/keys/{key_id}", "/settings/keys/{key_id}/test"} <= documented


def test_a_typed_key_goes_into_a_password_field_that_is_cleared_on_send(page: str) -> None:
    """The one secret the page handles is never echoed: masked while typed, gone once sent."""
    assert 'input.type = "password";' in page
    assert 'input.value = "";' in page


async def test_only_the_stack_dashboard_may_frame_the_page() -> None:
    """The :9090 dashboard shows the settings panel in a frame; no other site may,
    or it could overlay the page and steer clicks on key and settings buttons."""
    app = create_app(Config())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://switchboard") as client:
        response = await client.get("/dashboard")

    assert response.headers["content-security-policy"] == (
        "frame-ancestors 'self' http://127.0.0.1:9090 http://localhost:9090"
    )


def test_the_embedded_view_is_settings_only_and_polls_nothing(page: str) -> None:
    assert 'get("embed") === "settings"' in page
    assert "if (EMBEDDED || state.inflight)" in page, "embedded, refresh must not poll"
    assert "switchboard-settings-height" in page


def test_destructive_buttons_do_not_rely_on_confirm_dialogs(page: str) -> None:
    """A frame from another origin may have confirm() refused; Remove and Reset ask twice instead."""
    assert "secondClick(remove" in page and "secondClick(reset" in page


def test_setting_groups_flow_in_columns_without_row_gaps(page: str) -> None:
    """A two-column grid left a gap under every short group; columns stack them."""
    assert "#settings-groups { columns:" in page
    assert "break-inside: avoid" in page

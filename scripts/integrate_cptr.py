"""Add (or remove) Switchboard in cptr's configuration.

cptr keeps its app config in ``~/.cptr/config.toml`` and, in its own words,
"the file always wins - this ensures config survives DB loss and allows
hand-editing the TOML file". That is the supported path used here, because
cptr's admin API needs a browser session cookie that a script cannot get.

Two rules make this safe:

* **cptr must be stopped.** It rewrites the whole file after any settings
  change, so an edit made while it runs would be overwritten from its database.
* **Only two lines change.** The file is rewritten line by line, with the exact
  escaping cptr itself uses (backslash and double quote), so every other
  setting - including the keys stored in there - is preserved byte for byte.

Both changes are additive: the existing "Llama.cpp" connection and the existing
MCP tool servers are left exactly as they are.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path

CPTR_CONFIG = Path.home() / ".cptr" / "config.toml"

CONNECTION_NAME = "Switchboard"
TOOL_SERVER_ID = "switchboard"


def escape(value: str) -> str:
    """Exactly cptr's own escaping (see cptr/utils/config.py save_config)."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def load_section(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle).get("app_config", {}) or {}


def decode(raw: object) -> list:
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return value if isinstance(value, list) else []
    return raw if isinstance(raw, list) else []


def set_key(text: str, key: str, value: str) -> str:
    """Replace `"key" = "..."` in [app_config], or add it to that section."""
    encoded = f'"{key}" = "{escape(value)}"'
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f'"{key}"') or stripped.startswith(f"{key} ="):
            head = stripped.split("=", 1)[0].strip().strip('"')
            if head == key:
                lines[index] = encoded
                return "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    # Not present: append it to the end of the [app_config] section.
    try:
        start = next(
            index for index, line in enumerate(lines) if line.strip() == "[app_config]"
        )
    except StopIteration:
        lines += ["", "[app_config]", encoded]
        return "\n".join(lines) + "\n"

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].strip().startswith("["):
            end = index
            break
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    lines.insert(end, encoded)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def build_connection(base_url: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "name": CONNECTION_NAME,
        "provider": "openai",
        "api_type": "chat_completions",
        # Switchboard proxies llama.cpp, so cptr's llama.cpp handling (reasoning
        # replay, context estimation) applies to it too.
        "provider_type": "llama.cpp",
        "prefix_id": None,
        # No /v1: cptr appends "/chat/completions" to this string, and
        # Switchboard serves both path shapes for exactly that reason.
        "base_url": base_url,
        "api_key": None,
        "enabled": True,
        "data": {},
    }


def build_tool_server(base_url: str) -> dict:
    return {
        "id": TOOL_SERVER_ID,
        "type": "mcp",
        "url": f"{base_url}/mcp",
        "path": "openapi.json",
        "auth_type": "none",
        "key": "",
        "name": "Switchboard",
        "description": "Route a task to the right local model; evaluate agent steps.",
        "headers": None,
        "enabled": True,
        "command": "",
        "args": [],
        "env": None,
        "cwd": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--config", type=Path, default=CPTR_CONFIG)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--remove", action="store_true", help="Undo the integration.")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    path: Path = arguments.config
    if not path.is_file():
        print(f"cptr config not found at {path}; nothing to do.")
        return 0

    text = path.read_text(encoding="utf-8")
    section = load_section(path)
    connections = decode(section.get("chat.connections"))
    tool_servers = decode(section.get("tool_servers"))

    base_url = arguments.base_url.rstrip("/")
    changes: list[str] = []

    if arguments.remove:
        kept_connections = [
            item
            for item in connections
            if not (
                item.get("name") == CONNECTION_NAME
                or str(item.get("base_url", "")).rstrip("/") == base_url
            )
        ]
        if len(kept_connections) != len(connections):
            changes.append(f"removed the {CONNECTION_NAME} connection")
        kept_servers = [item for item in tool_servers if item.get("id") != TOOL_SERVER_ID]
        if len(kept_servers) != len(tool_servers):
            changes.append("removed the Switchboard MCP tool server")
        connections, tool_servers = kept_connections, kept_servers
    else:
        if any(
            item.get("name") == CONNECTION_NAME
            or str(item.get("base_url", "")).rstrip("/") == base_url
            for item in connections
        ):
            print(f"The {CONNECTION_NAME} connection is already configured.")
        else:
            connections.append(build_connection(base_url))
            changes.append(f"added the {CONNECTION_NAME} connection ({base_url})")

        if any(item.get("id") == TOOL_SERVER_ID for item in tool_servers):
            print("The Switchboard MCP tool server is already configured.")
        else:
            tool_servers.append(build_tool_server(base_url))
            changes.append(f"added the Switchboard MCP tool server ({base_url}/mcp)")

    if not changes:
        print("No changes needed.")
        return 0

    updated = set_key(text, "chat.connections", json.dumps(connections, ensure_ascii=False))
    updated = set_key(updated, "tool_servers", json.dumps(tool_servers, ensure_ascii=False))

    if arguments.dry_run:
        print("Would apply: " + "; ".join(changes))
        return 0

    arguments.backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = arguments.backup_dir / f"cptr-config.toml.{stamp}.bak"
    shutil.copy2(path, backup)

    path.write_text(updated, encoding="utf-8")

    # Re-read it: a config cptr cannot parse would lock the user out of their own
    # settings, so prove it still parses before declaring success.
    try:
        reloaded = load_section(path)
        assert isinstance(decode(reloaded.get("chat.connections")), list)
    except Exception as error:  # noqa: BLE001
        shutil.copy2(backup, path)
        print(f"Edit produced an unparseable file ({error}); restored from {backup}.")
        return 1

    print("Applied: " + "; ".join(changes))
    print(f"Backup:  {backup}")
    print("Start cptr to pick it up (it reads config.toml at startup).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

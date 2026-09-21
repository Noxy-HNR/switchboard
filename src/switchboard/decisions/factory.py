"""Building the decision chain from configuration.

Backends are found by the ``name`` they declare rather than by a hardcoded
class name, so adding one means dropping a module in this package and listing
its name in ``decisions.backends``. Imports are lazy: a machine with no
TypeSafe credentials should not pay to import the SDK, and a broken optional
backend must not stop Switchboard from starting.

The chain always ends with the rule engine, even if the configuration forgets
it, because "no decision at all" is not an outcome the router can use.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from switchboard.config import Config
from switchboard.decisions.chain import DecisionChain
from switchboard.upstream import LlamaClient

logger = logging.getLogger("switchboard.decisions")

# name -> module in this package.
BACKEND_MODULES: dict[str, str] = {
    "jev": "jev",
    "local": "local",
    "logit": "logit",
    "rules": "rules",
}

FLOOR_BACKEND = "rules"


def backend_names() -> list[str]:
    return list(BACKEND_MODULES)


def _find_backend_class(module_name: str, backend_name: str) -> type | None:
    """The class in `module_name` that declares `name = backend_name`."""
    module = importlib.import_module(f"switchboard.decisions.{module_name}")
    for value in vars(module).values():
        if (
            isinstance(value, type)
            and getattr(value, "name", None) == backend_name
            and hasattr(value, "decide_task")
            and hasattr(value, "decide_step")
        ):
            return value
    return None


def build_backend(name: str, config: Config, upstream: LlamaClient | None = None) -> Any | None:
    """Construct one backend, or return None if it cannot be built."""
    module_name = BACKEND_MODULES.get(name)
    if module_name is None:
        logger.warning("unknown decision backend %r; skipping", name)
        return None
    try:
        backend_class = _find_backend_class(module_name, name)
    except Exception as error:  # noqa: BLE001 - an optional dependency may be missing
        logger.warning("decision backend %r could not be imported: %s", name, error)
        return None
    if backend_class is None:
        logger.warning("module switchboard.decisions.%s declares no %r backend", module_name, name)
        return None
    try:
        return backend_class(config)
    except Exception as error:  # noqa: BLE001
        logger.warning("decision backend %r could not be constructed: %s", name, error)
        return None


def build_chain(config: Config, upstream: LlamaClient | None = None) -> DecisionChain:
    """Build the configured chain, guaranteeing a backend that cannot fail."""
    wanted = list(config.decisions.backends)
    if FLOOR_BACKEND not in wanted:
        logger.warning(
            "decisions.backends did not include %r; appending it so a decision is always possible",
            FLOOR_BACKEND,
        )
        wanted.append(FLOOR_BACKEND)
    elif wanted[-1] != FLOOR_BACKEND:
        # The floor only works as a floor if nothing follows it.
        wanted = [name for name in wanted if name != FLOOR_BACKEND] + [FLOOR_BACKEND]

    backends = []
    for name in wanted:
        backend = build_backend(name, config, upstream)
        if backend is not None:
            backends.append(backend)
    return DecisionChain(config, backends)

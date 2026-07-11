"""Agent intention package -- standing-want entities, collection, and status enum."""

from __future__ import annotations

# Version derived from pyproject.toml so the metadata is the single
# source of truth -- a future release that bumps pyproject without
# updating ``__init__.py`` can't drift the runtime ``__version__``.
# The except guard handles the rare case where the package isn't
# installed via importlib.metadata (e.g. running directly from a
# checked-out source tree without ``uv sync``); the fallback keeps
# imports working but reports ``unknown`` rather than crashing.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version
from typing import TYPE_CHECKING

try:
    __version__ = _version("3tears-agent-intention")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

# lazy public API (PEP 562). the package namespace no longer imports its
# implementation modules eagerly: importing this package (or any of its
# submodules) costs only this file, and each public attribute resolves
# its defining module on first access. the TYPE_CHECKING block carries
# the real imports so mypy and IDEs see the full statically-typed API;
# the _LAZY map is the runtime equivalent. the three-way agreement
# between __all__, _LAZY, and the TYPE_CHECKING block is pinned by the
# package's lazy-surface consistency test. mirrors agent/memory's
# hand-rolled PEP 562 (zero added runtime deps, no stub drift).
if TYPE_CHECKING:
    from threetears.agent.intention.collections import (
        IntentionsCollection,
        intentions_table,
    )
    from threetears.agent.intention.entities import IntentionEntity
    from threetears.agent.intention.types import (
        INTENTION_STATUS_VALUES,
        IntentionStatus,
    )

# public attribute -> (defining module, attribute name in that module)
_LAZY: dict[str, tuple[str, str]] = {
    "INTENTION_STATUS_VALUES": ("threetears.agent.intention.types", "INTENTION_STATUS_VALUES"),
    "IntentionEntity": ("threetears.agent.intention.entities", "IntentionEntity"),
    "IntentionStatus": ("threetears.agent.intention.types", "IntentionStatus"),
    "IntentionsCollection": ("threetears.agent.intention.collections", "IntentionsCollection"),
    "intentions_table": ("threetears.agent.intention.collections", "intentions_table"),
}

__all__ = [
    "INTENTION_STATUS_VALUES",
    "IntentionEntity",
    "IntentionStatus",
    "IntentionsCollection",
    "intentions_table",
]


def __getattr__(name: str) -> object:
    """resolve a public attribute from its defining module on first access.

    :param name: attribute name being resolved
    :ptype name: str
    :return: the resolved attribute (also cached in module globals so
        ``__getattr__`` fires at most once per name)
    :rtype: object
    :raises AttributeError: when ``name`` is not part of the public API
    """
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attr = entry
    value: object = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """include lazy attributes in ``dir()`` output.

    :return: sorted union of materialized globals and lazy names
    :rtype: list[str]
    """
    return sorted(set(globals()) | set(_LAZY))

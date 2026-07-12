"""Agent identity package -- versioned identity blocks for self-evolution.

Ships (T2.1a): the ``identity_versions`` entity + three-tier collection +
schema/table factory, and the block-key / status / consent-tier value
types. The lifecycle ops (propose/consent/reject/rollback) + events +
owner RBAC + tools land in T2.1b.
"""

from __future__ import annotations

# Version derived from pyproject.toml so the metadata is the single source
# of truth. The except guard handles running from a source tree without an
# installed distribution (reports ``unknown`` rather than crashing).
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version
from typing import TYPE_CHECKING

try:
    __version__ = _version("3tears-agent-identity")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

# lazy public API (PEP 562), mirroring agent/intention: importing this
# package costs only this file; each public attribute resolves its
# defining module on first access. The three-way agreement between
# __all__, _LAZY, and the TYPE_CHECKING block is pinned by the package's
# lazy-surface consistency test.
if TYPE_CHECKING:
    from threetears.agent.identity.collections import (
        IdentityVersionsCollection,
        identity_versions_table,
    )
    from threetears.agent.identity.entities import IdentityVersionEntity
    from threetears.agent.identity.types import (
        IDENTITY_BLOCK_KEY_VALUES,
        IDENTITY_BLOCK_TIERS,
        IDENTITY_STATUS_VALUES,
        IdentityBlockKey,
        IdentityTier,
        IdentityVersionStatus,
    )

# public attribute -> (defining module, attribute name in that module)
_LAZY: dict[str, tuple[str, str]] = {
    "IDENTITY_BLOCK_KEY_VALUES": ("threetears.agent.identity.types", "IDENTITY_BLOCK_KEY_VALUES"),
    "IDENTITY_BLOCK_TIERS": ("threetears.agent.identity.types", "IDENTITY_BLOCK_TIERS"),
    "IDENTITY_STATUS_VALUES": ("threetears.agent.identity.types", "IDENTITY_STATUS_VALUES"),
    "IdentityBlockKey": ("threetears.agent.identity.types", "IdentityBlockKey"),
    "IdentityTier": ("threetears.agent.identity.types", "IdentityTier"),
    "IdentityVersionEntity": ("threetears.agent.identity.entities", "IdentityVersionEntity"),
    "IdentityVersionStatus": ("threetears.agent.identity.types", "IdentityVersionStatus"),
    "IdentityVersionsCollection": (
        "threetears.agent.identity.collections",
        "IdentityVersionsCollection",
    ),
    "identity_versions_table": (
        "threetears.agent.identity.collections",
        "identity_versions_table",
    ),
}

__all__ = [
    "IDENTITY_BLOCK_KEY_VALUES",
    "IDENTITY_BLOCK_TIERS",
    "IDENTITY_STATUS_VALUES",
    "IdentityBlockKey",
    "IdentityTier",
    "IdentityVersionEntity",
    "IdentityVersionStatus",
    "IdentityVersionsCollection",
    "identity_versions_table",
]


def __getattr__(name: str) -> object:
    """resolve a public attribute from its defining module on first access.

    :param name: attribute name being resolved
    :ptype name: str
    :return: the resolved attribute (cached in module globals)
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

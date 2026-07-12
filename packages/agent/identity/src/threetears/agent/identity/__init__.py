"""Agent identity package -- versioned identity blocks for self-evolution.

The store + linear version chain (``identity_versions``), the
propose→consent/reject→apply + rollback lifecycle, the FrameworkEvents, the
owner RBAC, and the private ``identity_propose`` tool.
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
# package costs only this file; each public attribute resolves its defining
# module on first access. The three-way agreement between __all__, _LAZY,
# and the TYPE_CHECKING block is pinned by the package's lazy-surface test.
if TYPE_CHECKING:
    from threetears.agent.identity.authorize import (
        ACTION_IDENTITY_READ,
        ACTION_IDENTITY_WRITE,
        IDENTITY_NAMESPACE_TYPE,
        IdentityAccessDenied,
        IdentityAuthorizerDependencies,
        authorize_identity_access,
        identity_namespace_name,
    )
    from threetears.agent.identity.collections import (
        IdentityVersionsCollection,
        identity_versions_table,
    )
    from threetears.agent.identity.entities import IdentityVersionEntity
    from threetears.agent.identity.events import (
        IdentityAppliedEvent,
        IdentityConsentedEvent,
        IdentityProposedEvent,
        IdentityRestoredEvent,
    )
    from threetears.agent.identity.lifecycle import (
        consent,
        content_hash,
        propose,
        reject,
        rollback,
        seed_active,
    )
    from threetears.agent.identity.tools import (
        IdentityProposeInput,
        load_identity_propose_tool,
    )
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
    "ACTION_IDENTITY_READ": ("threetears.agent.identity.authorize", "ACTION_IDENTITY_READ"),
    "ACTION_IDENTITY_WRITE": ("threetears.agent.identity.authorize", "ACTION_IDENTITY_WRITE"),
    "IDENTITY_BLOCK_KEY_VALUES": ("threetears.agent.identity.types", "IDENTITY_BLOCK_KEY_VALUES"),
    "IDENTITY_BLOCK_TIERS": ("threetears.agent.identity.types", "IDENTITY_BLOCK_TIERS"),
    "IDENTITY_NAMESPACE_TYPE": ("threetears.agent.identity.authorize", "IDENTITY_NAMESPACE_TYPE"),
    "IDENTITY_STATUS_VALUES": ("threetears.agent.identity.types", "IDENTITY_STATUS_VALUES"),
    "IdentityAccessDenied": ("threetears.agent.identity.authorize", "IdentityAccessDenied"),
    "IdentityAppliedEvent": ("threetears.agent.identity.events", "IdentityAppliedEvent"),
    "IdentityAuthorizerDependencies": (
        "threetears.agent.identity.authorize",
        "IdentityAuthorizerDependencies",
    ),
    "IdentityBlockKey": ("threetears.agent.identity.types", "IdentityBlockKey"),
    "IdentityConsentedEvent": ("threetears.agent.identity.events", "IdentityConsentedEvent"),
    "IdentityProposeInput": ("threetears.agent.identity.tools", "IdentityProposeInput"),
    "IdentityProposedEvent": ("threetears.agent.identity.events", "IdentityProposedEvent"),
    "IdentityRestoredEvent": ("threetears.agent.identity.events", "IdentityRestoredEvent"),
    "IdentityTier": ("threetears.agent.identity.types", "IdentityTier"),
    "IdentityVersionEntity": ("threetears.agent.identity.entities", "IdentityVersionEntity"),
    "IdentityVersionStatus": ("threetears.agent.identity.types", "IdentityVersionStatus"),
    "IdentityVersionsCollection": (
        "threetears.agent.identity.collections",
        "IdentityVersionsCollection",
    ),
    "authorize_identity_access": (
        "threetears.agent.identity.authorize",
        "authorize_identity_access",
    ),
    "consent": ("threetears.agent.identity.lifecycle", "consent"),
    "content_hash": ("threetears.agent.identity.lifecycle", "content_hash"),
    "identity_namespace_name": ("threetears.agent.identity.authorize", "identity_namespace_name"),
    "identity_versions_table": (
        "threetears.agent.identity.collections",
        "identity_versions_table",
    ),
    "load_identity_propose_tool": (
        "threetears.agent.identity.tools",
        "load_identity_propose_tool",
    ),
    "propose": ("threetears.agent.identity.lifecycle", "propose"),
    "reject": ("threetears.agent.identity.lifecycle", "reject"),
    "rollback": ("threetears.agent.identity.lifecycle", "rollback"),
    "seed_active": ("threetears.agent.identity.lifecycle", "seed_active"),
}

__all__ = [
    "ACTION_IDENTITY_READ",
    "ACTION_IDENTITY_WRITE",
    "IDENTITY_BLOCK_KEY_VALUES",
    "IDENTITY_BLOCK_TIERS",
    "IDENTITY_NAMESPACE_TYPE",
    "IDENTITY_STATUS_VALUES",
    "IdentityAccessDenied",
    "IdentityAppliedEvent",
    "IdentityAuthorizerDependencies",
    "IdentityBlockKey",
    "IdentityConsentedEvent",
    "IdentityProposeInput",
    "IdentityProposedEvent",
    "IdentityRestoredEvent",
    "IdentityTier",
    "IdentityVersionEntity",
    "IdentityVersionStatus",
    "IdentityVersionsCollection",
    "authorize_identity_access",
    "consent",
    "content_hash",
    "identity_namespace_name",
    "identity_versions_table",
    "load_identity_propose_tool",
    "propose",
    "reject",
    "rollback",
    "seed_active",
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

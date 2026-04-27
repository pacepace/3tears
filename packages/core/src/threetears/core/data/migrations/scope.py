"""
migration scope enumeration.

PLATFORM-scope migrations apply once on hub startup against the shared
platform schema (customers, agents, namespaces, ...). AGENT-scope
migrations apply once per agent schema at provision time. the runner
tracks applied versions per-scope and per-package so the two domains
never collide even though they share the same ``_schema_migrations``
contract.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "MigrationScope",
]


class MigrationScope(str, Enum):
    """
    scope of a migration package registration.

    :cvar PLATFORM: migration applies once against the hub's shared
        platform schema. owned by the hub or by libraries that ship
        platform-level tables.
    :cvar AGENT: migration applies once per agent schema at provision
        time. composed across every registered agent-scoped package.
    """

    PLATFORM = "platform"
    AGENT = "agent"

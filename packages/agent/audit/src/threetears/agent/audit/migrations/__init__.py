"""
agent-audit package migrations.

:func:`register` wires the package's versioned migrations into a shared
:class:`~threetears.core.data.migrations.runner.MigrationRunner`. agent-audit
owns one platform-scoped table, ``audit_events``.

version history:

- v001 creates ``audit_events`` and its four indexes (3tears v0.59.0).
"""

from __future__ import annotations

from threetears.agent.audit.migrations.v001_create_audit_events import create_audit_events
from threetears.core.data.migrations import MigrationRunner, MigrationScope, PackageMigrations

PACKAGE_NAME = "agent_audit"


def register(runner: MigrationRunner) -> PackageMigrations:
    """
    register agent-audit migrations with the given runner.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(name=PACKAGE_NAME, scope=MigrationScope.PLATFORM)
    pkg.version(1)(create_audit_events)
    runner.register(pkg)
    return pkg


__all__ = ["PACKAGE_NAME", "create_audit_events", "register"]

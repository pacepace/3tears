"""
agent-acl package migrations.

:func:`register` wires the package's versioned migrations into a shared
:class:`~threetears.core.data.migrations.runner.MigrationRunner`. agent-acl
owns the five rbac tables the evaluator reads, platform-scoped:
``namespaces``, ``groups``, ``group_members``, ``roles`` and
``role_assignments``.

version history:

- v001 creates the five tables, their keys, and the uniqueness the
  evaluator and the idempotent writers rely on (3tears v0.59.0).

``agent_tools_platform`` ALTERs ``namespaces``; a deployment that runs
both registers it with ``depends_on=("agent_acl",)``.
"""

from __future__ import annotations

from threetears.agent.acl.migrations.v001_create_acl_tables import ACL_TABLES_DDL, create_acl_tables
from threetears.core.data.migrations import MigrationRunner, MigrationScope, PackageMigrations

PACKAGE_NAME = "agent_acl"


def register(runner: MigrationRunner) -> PackageMigrations:
    """
    register agent-acl migrations with the given runner.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(name=PACKAGE_NAME, scope=MigrationScope.PLATFORM)
    pkg.version(1)(create_acl_tables)
    runner.register(pkg)
    return pkg


__all__ = ["ACL_TABLES_DDL", "PACKAGE_NAME", "create_acl_tables", "register"]

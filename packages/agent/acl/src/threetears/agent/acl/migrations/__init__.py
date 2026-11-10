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
- v002 creates the indexes the evaluator's keyed reads use (3tears v0.90.0).
- v003 creates the indexes the namespace lookups by owner, customer, owner namespace and schema use
  (3tears v0.95.0).

``agent_tools_platform`` ALTERs ``namespaces``; a deployment that runs
both registers it with ``depends_on=("agent_acl",)``.
"""

from __future__ import annotations

from threetears.agent.acl.collections import (
    GroupCollection,
    GroupMemberCollection,
    NamespaceCollection,
    RoleAssignmentCollection,
    RoleCollection,
)
from threetears.agent.acl.migrations.v001_create_acl_tables import ACL_TABLES_DDL, create_acl_tables
from threetears.agent.acl.migrations.v002_reader_indexes import create_acl_reader_indexes
from threetears.agent.acl.migrations.v003_namespace_reader_indexes import create_acl_namespace_reader_indexes
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
    pkg.version(2)(create_acl_reader_indexes)
    pkg.version(3)(create_acl_namespace_reader_indexes)
    # the five access tables are switched on: each carries the write-mark trigger from this run
    pkg.mark_write_generation(
        GroupCollection.schema,
        GroupMemberCollection.schema,
        RoleCollection.schema,
        RoleAssignmentCollection.schema,
        NamespaceCollection.schema,
    )
    runner.register(pkg)
    return pkg


__all__ = [
    "ACL_TABLES_DDL",
    "PACKAGE_NAME",
    "create_acl_tables",
    "register",
]

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
- v002 creates the indexes the evaluator's keyed reads use (3tears v0.95.0).
- v003 creates the indexes the namespace lookups by owner, customer, owner namespace and schema use
  (3tears v0.95.0).
- v004 creates the indexes the access tables' other keyed methods use (3tears v0.95.0).
- v005 adds ``role_assignments.customer_id``, the grant's customer, nullable and unchecked (the expand step;
  ``RoleAssignmentCollection.backfill_customers`` fills the older grants) (3tears v0.95.0).
- v006 makes the five tables ``ROW_STAMPED``: each one's epoch (backfilled a page of keys a statement), gone flag, stored
  restore bucket, ``(epoch_bucket, epoch)`` index and declaration, so a restore of a lost change log reads them a
  bucket at a time and the change-note ring can be retired (3tears v0.95.0).
- v007 keys ``namespaces`` by ``namespace_id`` alone and makes ``row_scope`` a moveable column (its
  ``row_scope_moved_from`` / ``_moved_epoch``), so a rescope is an in-place update a restore hears; adds
  ``idx_namespaces_scope_id`` (row_scope, namespace_id) for the id walk; rebuilds the restore bucket for the new key
  (3tears v0.95.0).
- v008 makes the group and role name indexes hold live rows only (``... AND NOT gone``): a deleted (gone) group or role
  no longer holds its name, so a re-create by the same name makes a new row (3tears v0.95.0).

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
from threetears.agent.acl.migrations.v004_keyed_method_indexes import create_acl_keyed_method_indexes
from threetears.agent.acl.migrations.v005_assignment_customer import add_assignment_customer
from threetears.agent.acl.migrations.v006_access_tables_carry_their_epoch import access_tables_carry_their_epoch
from threetears.agent.acl.migrations.v007_namespaces_keyed_by_id import namespaces_keyed_by_id
from threetears.agent.acl.migrations.v008_names_hold_live_rows_only import names_hold_live_rows_only
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
    pkg.version(4)(create_acl_keyed_method_indexes)
    pkg.version(5)(add_assignment_customer)
    pkg.version(6)(access_tables_carry_their_epoch)
    pkg.version(7)(namespaces_keyed_by_id)
    pkg.version(8)(names_hold_live_rows_only)
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
    "access_tables_carry_their_epoch",
    "create_acl_tables",
    "names_hold_live_rows_only",
    "register",
]

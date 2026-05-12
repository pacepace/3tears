"""
agent-workspace package migrations.

single entry point :func:`register` wires the package's versioned
migration callables into a shared :class:`~threetears.core.data.
migrations.runner.MigrationRunner`. follow this same shape in every
new package that ships schema — one ``migrations/`` subpackage, one
migration file per version, one ``register`` function.

version history:

- v001: create workspaces, workspace_files, workspace_file_versions
- v002: add ``workspaces.date_deleted`` column for soft-delete
- v003: backfill ``platform.namespaces`` rows for existing workspaces
- v004: promote every datetime column from TIMESTAMP to TIMESTAMPTZ
  (collections-task-05 phase A3 -- eliminates DATETIME_TYPE in this
  package; the column declarations in
  :mod:`threetears.agent.workspace.collections` flip from
  DATETIME_TYPE to DATETIMETZ_TYPE in the same commit so
  ``tests/enforcement/test_column_type_alignment.py`` stays green).
"""

from __future__ import annotations

from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

from threetears.agent.workspace.migrations.v001_create_workspace_tables import (
    create_workspace_tables,
)
from threetears.agent.workspace.migrations.v002_add_date_deleted_column import (
    add_date_deleted_column,
)
from threetears.agent.workspace.migrations.v003_workspace_namespace_backfill import (
    workspace_namespace_backfill,
)
from threetears.agent.workspace.migrations.v004_datetime_to_datetimetz import (
    datetime_to_datetimetz,
)

PACKAGE_NAME = "agent_workspace"


def register(runner: MigrationRunner) -> PackageMigrations:
    """
    register agent-workspace migrations with the given runner.

    produces an agent-scoped :class:`PackageMigrations`, attaches every
    migration in version order, and calls ``runner.register``. returns
    the PackageMigrations object so callers can inspect registered
    versions in tests.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.AGENT,
    )
    pkg.version(1)(create_workspace_tables)
    pkg.version(2)(add_date_deleted_column)
    pkg.version(3)(workspace_namespace_backfill)
    pkg.version(4)(datetime_to_datetimetz)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "add_date_deleted_column",
    "create_workspace_tables",
    "datetime_to_datetimetz",
    "register",
    "workspace_namespace_backfill",
]

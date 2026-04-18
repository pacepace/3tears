"""
agent-workspace package migrations.

single entry point :func:`register` wires the package's versioned
migration callables into a shared :class:`~threetears.core.data.
migrations.runner.MigrationRunner`. follow this same shape in every
new package that ships schema — one ``migrations/`` subpackage, one
migration file per version, one ``register`` function.
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
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "add_date_deleted_column",
    "create_workspace_tables",
    "register",
]

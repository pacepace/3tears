"""3tears-epoch package migrations.

single entry point :func:`register` wires the package's versioned
migration callable into a shared :class:`~threetears.core.data.
migrations.runner.MigrationRunner`. the epoch package owns one
platform-scope table -- ``config_epochs`` -- shared by every consumer
that broadcasts cross-pod config-reload epochs.

PLATFORM scope: the table lives in the consumer application's
platform schema (whatever schema the caller binds via search_path
before calling :meth:`MigrationRunner.apply_for_platform_schema`).
3tears does not assume a literal ``platform`` schema name.

version history:

- v001 -- create the ``config_epochs`` table with subject_path PK,
  epoch counter, opaque payload, and update timestamp.
"""

from __future__ import annotations

from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)
from threetears.epoch.migrations.v001_create_config_epochs import (
    create_config_epochs_table,
)

PACKAGE_NAME = "epoch"


def register(runner: MigrationRunner) -> PackageMigrations:
    """register epoch migrations with the given runner.

    produces a platform-scoped :class:`PackageMigrations` with no
    ``depends_on`` edges (the table is standalone), attaches every
    migration in version order, and calls ``runner.register``.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.PLATFORM,
    )
    pkg.version(1)(create_config_epochs_table)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "create_config_epochs_table",
    "register",
]

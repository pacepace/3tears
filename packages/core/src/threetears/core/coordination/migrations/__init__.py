"""3tears core coordination migrations.

:func:`register` wires this package's versioned migrations into a shared
:class:`~threetears.core.data.migrations.runner.MigrationRunner`. The package owns the four tables
the durable coordination primitives keep their state in: ``coordination_counters``,
``coordination_claims``, ``coordination_revocations`` and ``coordination_redemptions``.

PLATFORM scope: the tables land in whatever schema the caller binds through ``search_path`` before
calling :meth:`MigrationRunner.apply_for_platform_schema`. A consumer whose primitives run in an
agent schema re-registers this package at AGENT scope, as scriob already does for
``scheduled_jobs`` and ``epoch``. A consumer that cannot run DDL at all -- an agent or tool pod,
whose broker refuses it -- declares the same schemas in its data section instead, from
:data:`~threetears.core.coordination.tables.COORDINATION_TABLE_SCHEMAS`.

version history:

- v001 -- create the four coordination tables, rendered from their collections' declared schemas.
"""

from __future__ import annotations

from threetears.core.coordination.migrations.v001_create_coordination_tables import (
    create_coordination_tables,
)
from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

PACKAGE_NAME = "coordination"


def register(runner: MigrationRunner, *, scope: MigrationScope = MigrationScope.PLATFORM) -> PackageMigrations:
    """register the coordination migrations with the given runner.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :param scope: which schema the tables belong in; PLATFORM by default, AGENT for a consumer
        whose primitives run in an agent schema
    :ptype scope: MigrationScope
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(name=PACKAGE_NAME, scope=scope)
    pkg.version(1)(create_coordination_tables)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "create_coordination_tables",
    "register",
]

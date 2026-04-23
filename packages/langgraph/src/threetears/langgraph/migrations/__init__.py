"""
3tears-langgraph package migrations.

single entry point :func:`register` wires the package's versioned
migration callables into a shared :class:`~threetears.core.data.
migrations.runner.MigrationRunner`. the langgraph package owns the
``checkpoints`` and ``checkpoint_writes`` tables used by
:class:`~threetears.langgraph.checkpoint.ThreeTierCheckpointSaver` and
:class:`~threetears.langgraph.proxy_checkpoint.ProxyCheckpointSaver`.
"""

from __future__ import annotations

from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

from threetears.langgraph.migrations.v001_create_checkpoint_tables import (
    create_checkpoint_tables,
)

PACKAGE_NAME = "langgraph"


def register(runner: MigrationRunner) -> PackageMigrations:
    """
    register 3tears-langgraph migrations with the given runner.

    produces an agent-scoped :class:`PackageMigrations` and attaches
    every migration in version order. langgraph checkpoint state is
    fully per-agent so this package is always AGENT scope.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.AGENT,
    )
    pkg.version(1)(create_checkpoint_tables)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "create_checkpoint_tables",
    "register",
]

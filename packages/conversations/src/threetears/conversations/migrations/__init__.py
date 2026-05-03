"""
3tears-conversations package migrations.

single entry point :func:`register` wires the package's versioned
migration callables into a shared :class:`~threetears.core.data.
migrations.runner.MigrationRunner`. the conversations package owns
exactly one table in every agent schema -- ``conversations`` -- plus
the three indexes that support scoped queries.

the package declares no ``depends_on`` edges: it is the root of the
agent-schema dependency graph, and every other package that keys off
``conversation_id`` declares ``depends_on=("conversations",)``.
"""

from __future__ import annotations

from threetears.conversations.migrations.v001_create_conversations_table import (
    create_conversations_table,
)
from threetears.conversations.migrations.v002_add_message_count import (
    add_message_count,
)
from threetears.conversations.migrations.v003_add_name_column import (
    add_name_column,
)
from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

PACKAGE_NAME = "conversations"


def register(runner: MigrationRunner) -> PackageMigrations:
    """
    register conversations migrations with the given runner.

    produces an agent-scoped :class:`PackageMigrations` with no
    ``depends_on`` edges, attaches every migration in version order,
    and calls ``runner.register``.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.AGENT,
    )
    pkg.version(1)(create_conversations_table)
    pkg.version(2)(add_message_count)
    pkg.version(3)(add_name_column)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "add_message_count",
    "add_name_column",
    "create_conversations_table",
    "register",
]

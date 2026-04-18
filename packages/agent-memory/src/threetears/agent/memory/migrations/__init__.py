"""
agent-memory package migrations.

single entry point :func:`register` wires the package's versioned
migration callables into a shared :class:`~threetears.core.data.
migrations.runner.MigrationRunner`. after the conversations / context
ownership reshape, agent-memory owns exactly two tables in every
agent schema:

- ``memories`` -- extracted long-term memories with pgvector embedding.
- ``conversation_memory_refs`` -- append-only ledger tracking which
  items have already been surfaced in a conversation.

the package declares ``depends_on=("conversations",)`` because the
ledger references ``conversations(id)`` even though no FK constraint
exists today (the constraint would force a cross-package teardown
order; ordering on apply is enough).
"""

from __future__ import annotations

from threetears.agent.memory.migrations.v001_create_memories_table import (
    create_memories_table,
)
from threetears.agent.memory.migrations.v002_create_conversation_memory_refs import (
    create_conversation_memory_refs,
)
from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

PACKAGE_NAME = "agent_memory"


def register(runner: MigrationRunner) -> PackageMigrations:
    """
    register agent-memory migrations with the given runner.

    produces an agent-scoped :class:`PackageMigrations` declaring
    ``depends_on=("conversations",)`` so the conversations table
    always exists before memory tables are created. attaches every
    migration in version order and calls ``runner.register``.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.AGENT,
        depends_on=("conversations",),
    )
    pkg.version(1)(create_memories_table)
    pkg.version(2)(create_conversation_memory_refs)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "create_conversation_memory_refs",
    "create_memories_table",
    "register",
]

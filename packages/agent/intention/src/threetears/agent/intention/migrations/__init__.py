"""
agent-intention package migrations.

Single entry point :func:`register` wires the package's versioned
migration callables into a shared :class:`~threetears.core.data.
migrations.runner.MigrationRunner`. agent-intention owns exactly one
table in an agent schema:

- ``intentions`` -- the standing-wants corpus (status lifecycle,
  salience decay substrate, embedding for dedup, cooldown anchor).

version history:

- v001 creates the intentions table, the ``intention_status`` PG enum,
  the pgvector extension reference, and the three indexes (partial
  salience-ranked, cooldown btree, HNSW dedup). Presence/aliveness
  program, 3tears v0.15.0.

The package declares ``depends_on=("conversations",)`` for apply-ordering
consistency with agent-memory: ``source_conversation_id`` is a soft-ref
provenance column (no FK -- so no cross-package teardown-order
constraint), but ordering the conversations package first keeps the two
agent-scoped corpora in a stable, predictable apply order.
"""

from __future__ import annotations

from threetears.agent.intention.migrations.v001_create_intentions_table import (
    create_intentions_table,
)
from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

PACKAGE_NAME = "agent_intention"


def register(runner: MigrationRunner) -> PackageMigrations:
    """
    register agent-intention migrations with the given runner.

    produces an agent-scoped :class:`PackageMigrations` declaring
    ``depends_on=("conversations",)`` so the conversations package
    always applies before the intention table (ordering consistency;
    ``source_conversation_id`` is a soft ref with no FK).

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
    pkg.version(1)(create_intentions_table)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "create_intentions_table",
    "register",
]

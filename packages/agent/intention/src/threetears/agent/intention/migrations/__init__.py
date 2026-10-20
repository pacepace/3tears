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

The ``embedding`` column is created at the width :func:`register` is given
(``embedding_dim``, default 1024, agent-memory's width too); see
:mod:`threetears.agent.intention.embedding_width` for verifying and changing
the width of a schema that already exists.

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
from threetears.core.data.embedding_width import DEFAULT_EMBEDDING_DIM, check_embedding_dim
from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)
from threetears.core.data.store import DataStore

PACKAGE_NAME = "agent_intention"


def register(runner: MigrationRunner, *, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> PackageMigrations:
    """
    register agent-intention migrations with the given runner.

    produces an agent-scoped :class:`PackageMigrations` declaring
    ``depends_on=("conversations",)`` so the conversations package
    always applies before the intention table (ordering consistency;
    ``source_conversation_id`` is a soft ref with no FK).

    ``embedding_dim`` is the width of ``intentions.embedding``: the
    deployment's embedding provider's, the width agent-memory is
    registered with too. v001 creates the column at it and is registered
    under its function's own name, so a schema migrated at any width keeps
    the ledger it has. an existing schema keeps the width it was built
    with, checked by :func:`~threetears.agent.intention.embedding_width.
    verify_embedding_width` and changed only by :func:`~threetears.agent.
    intention.embedding_width.resize_embedding_width`. PostgreSQL indexes
    at most 2,000 dimensions and YugabyteDB 16,000; a width above the
    database's is refused when v001 runs, before any statement.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :param embedding_dim: the embedding column's width; default 1024
    :ptype embedding_dim: int
    :return: populated package registration
    :rtype: PackageMigrations
    :raises ValueError: when ``embedding_dim`` is below 1 or above 16,000
    """
    check_embedding_dim(embedding_dim)
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.AGENT,
        depends_on=("conversations",),
    )

    async def create_at_width(store: DataStore) -> None:
        """create the intentions table with its embedding column ``embedding_dim`` wide.

        :param store: DataStore bound to the per-agent schema
        :ptype store: DataStore
        :return: nothing
        :rtype: None
        """
        await create_intentions_table(store, embedding_dim=embedding_dim)

    pkg.step(1, name=create_intentions_table.__name__)(create_at_width)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "create_intentions_table",
    "register",
]

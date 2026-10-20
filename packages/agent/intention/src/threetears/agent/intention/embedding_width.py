"""checking and changing the width of agent-intention's stored embeddings.

``intentions.embedding`` is the deployment's width, the same width as agent-memory's so one
embedding provider serves both: :func:`~threetears.agent.intention.migrations.register` creates it
at its ``embedding_dim`` (default 1024), and :class:`~threetears.agent.intention.collections.
IntentionsCollection` and :func:`~threetears.agent.intention.collections.intentions_table` take the
same keyword. A schema keeps the width it was built with, because its migration ran once:

- :func:`verify_embedding_width` -- for a host's start-up -- refuses a schema whose column is not
  the configured width with :class:`EmbeddingWidthMismatch`;
- :func:`resize_embedding_width` -- an explicit operator action, never run automatically, run with writes to the tables stopped --
  changes the column to the configured width. It clears every stored embedding, so each want has
  to be embedded again, and rebuilds ``ix_intentions_embedding_hnsw`` under its own name and
  options.

PostgreSQL's HNSW index takes at most 2,000 dimensions; YugabyteDB's 16,000.
:mod:`threetears.core.data.embedding_width` holds the mechanism, shared with agent-memory.
"""

from __future__ import annotations

from threetears.core.data.embedding_width import (
    EmbeddingWidthMismatch,
    EmbeddingWidthStore,
    embedding_columns,
    resize_embedding_columns,
    verify_embedding_columns,
)

from threetears.agent.intention.collections import IntentionsCollection

__all__ = [
    "EmbeddingWidthMismatch",
    "resize_embedding_width",
    "verify_embedding_width",
]

#: the package's embedding column, read from the collection's declaration
_EMBEDDING_COLUMNS = embedding_columns(IntentionsCollection.schema)


async def verify_embedding_width(store: EmbeddingWidthStore, embedding_dim: int) -> None:
    """refuse a schema whose ``intentions.embedding`` is not ``embedding_dim`` wide.

    :param store: a store bound to the agent schema, after its migrations have run
    :ptype store: EmbeddingWidthStore
    :param embedding_dim: the configured width
    :ptype embedding_dim: int
    :return: nothing
    :rtype: None
    :raises EmbeddingWidthMismatch: when the column is another width
    :raises LookupError: when the table is missing (the migration has not run)
    :raises ValueError: when the width is below 1 or above 16,000
    """
    await verify_embedding_columns(store, _EMBEDDING_COLUMNS, embedding_dim)


async def resize_embedding_width(store: EmbeddingWidthStore, embedding_dim: int) -> int:
    """change ``intentions.embedding`` to ``embedding_dim``, clearing what it holds.

    every want keeps its content and loses its embedding, so each has to be embedded again with a
    model of the new width. a column already at the width is left as it is.
    run it with writes to these tables stopped: on YugabyteDB the column change is a table rewrite
    that may not carry over writes made while it runs ("Concurrent DMLs may not be reflected in the
    new table", yugabyte-db issue 19860). a
    :class:`~threetears.core.data.store.DataStore` runs it on one connection holding the
    database-wide DDL lock.

    :param store: a store bound to the agent schema, after its migration has run
    :ptype store: EmbeddingWidthStore
    :param embedding_dim: the new width
    :ptype embedding_dim: int
    :return: how many stored embeddings were cleared
    :rtype: int
    :raises ValueError: when the width is out of range, or above what the database can index
    :raises LookupError: when the table is missing (the migration has not run)
    """
    cleared = await resize_embedding_columns(store, _EMBEDDING_COLUMNS, embedding_dim)
    return cleared

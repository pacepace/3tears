"""checking and changing the width of agent-memory's stored embeddings.

``memories``, ``media_content`` and ``memory_chunks`` each carry an ``embedding`` column of the
deployment's width: :func:`~threetears.agent.memory.migrations.register` creates them at its
``embedding_dim`` (default 1024), and the collections and table factories take the same keyword.
A schema keeps the width it was built with, because its migrations ran once:

- :func:`verify_embedding_width` -- for a host's start-up -- refuses a schema whose columns are not
  the configured width with :class:`EmbeddingWidthMismatch`, naming the table;
- :func:`resize_embedding_width` -- an explicit operator action, never run automatically, run with writes to the tables stopped --
  changes every column to the configured width. It clears every stored embedding (an embedding of
  one width means nothing at another, so each row has to be embedded again) and rebuilds each
  column's HNSW index under its own name and options.

PostgreSQL's HNSW index takes at most 2,000 dimensions; YugabyteDB's 16,000.
:mod:`threetears.core.data.embedding_width` holds the mechanism, shared with agent-intention.
"""

from __future__ import annotations

from threetears.core.data.embedding_width import (
    EmbeddingWidthMismatch,
    EmbeddingWidthStore,
    embedding_columns,
    resize_embedding_columns,
    verify_embedding_columns,
)

from threetears.agent.memory.collections import MediaContentCollection, MemoriesCollection, MemoryChunkCollection

__all__ = [
    "EmbeddingWidthMismatch",
    "resize_embedding_width",
    "verify_embedding_width",
]

#: the package's embedding columns, read from the collections' declarations
_EMBEDDING_COLUMNS = embedding_columns(
    MemoriesCollection.schema,
    MediaContentCollection.schema,
    MemoryChunkCollection.schema,
)


async def verify_embedding_width(store: EmbeddingWidthStore, embedding_dim: int) -> None:
    """refuse a schema whose memory embedding columns are not ``embedding_dim`` wide.

    :param store: a store bound to the agent schema, after its migrations have run
    :ptype store: EmbeddingWidthStore
    :param embedding_dim: the configured width
    :ptype embedding_dim: int
    :return: nothing
    :rtype: None
    :raises EmbeddingWidthMismatch: naming the first table whose column is another width
    :raises LookupError: when a table is missing (the migrations have not run)
    :raises ValueError: when the width is below 1 or above 16,000
    """
    await verify_embedding_columns(store, _EMBEDDING_COLUMNS, embedding_dim)


async def resize_embedding_width(store: EmbeddingWidthStore, embedding_dim: int) -> int:
    """change every memory embedding column to ``embedding_dim``, clearing what they hold.

    every memory, media content and chunk row keeps its content and loses its embedding, so each
    has to be embedded again with a model of the new width. a column already at the width is left
    as it is; running it again after a stop finishes the job.
    run it with writes to these tables stopped: on YugabyteDB the column change is a table rewrite
    that may not carry over writes made while it runs ("Concurrent DMLs may not be reflected in the
    new table", yugabyte-db issue 19860). a
    :class:`~threetears.core.data.store.DataStore` runs it on one connection holding the
    database-wide DDL lock.

    :param store: a store bound to the agent schema, after its migrations have run
    :ptype store: EmbeddingWidthStore
    :param embedding_dim: the new width
    :ptype embedding_dim: int
    :return: how many stored embeddings were cleared, across the three tables
    :rtype: int
    :raises ValueError: when the width is out of range, or above what the database can index
    :raises LookupError: when a table is missing (the migrations have not run)
    """
    cleared = await resize_embedding_columns(store, _EMBEDDING_COLUMNS, embedding_dim)
    return cleared

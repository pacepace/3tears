"""
agent-memory v015: add unified-memory backlink columns (additive only).

transcript-chunks-task-A. The agent-memory data model is being
unified so that every chunk parents to a memory (not media) and
every media attachment parents to a memory. This is the first
migration in the unification sequence — it only adds columns, with
no constraint or FK changes. The follow-up migrations (v016
backfill, v017 NOT NULL + new FKs, v018 drop old columns, v019
NOT NULL conversation_id) lock in the shape.

Columns added:

- ``memory_chunks.memory_id UUID NULL`` — the new parent pointer.
  After v017 this becomes NOT NULL with a composite FK + CASCADE
  to ``memories(agent_id, memory_id)``. The existing
  ``memory_chunks.media_id`` column stays in place through v018
  so the backfill in v016 can read it; v018 drops it.
- ``memory_chunks.message_id_start UUID NULL`` — backlink to the
  first message in a transcript-derived chunk's source range.
  Stays NULL on document chunks. No FK (the messages table lives
  in a sibling system and is hard-deleted; dangling refs are
  expected).
- ``memory_chunks.message_id_end UUID NULL`` — backlink to the last
  message. Same shape as ``message_id_start``.
- ``media.memory_id UUID NULL`` — the new parent pointer. After
  v017 this becomes NOT NULL with a composite FK + CASCADE to
  ``memories(agent_id, memory_id)``. The reverse-direction
  ``memories.media_id`` column stays in place through v018 so the
  backfill can read existing pairings; v018 drops it.

Indexes added:

- ``idx_chunks_memory_id`` partial index on ``memory_chunks
  (memory_id)`` WHERE ``memory_id IS NOT NULL``. Small until
  v016 backfill populates the column; needed for the new
  ``find_by_memory_id`` collection method.
- ``idx_chunks_memory_id_chunk_id`` composite index on
  ``memory_chunks (memory_id, chunk_id)`` for cursor paging
  (``ORDER BY chunk_id ASC`` within a memory).
- ``idx_chunks_message_id_start`` partial index on
  ``memory_chunks (message_id_start)`` WHERE ``message_id_start
  IS NOT NULL``. Supports the transcript-order query in
  ``find_by_conversation_id``.
- ``idx_chunks_message_id_end`` partial index on
  ``memory_chunks (message_id_end)`` WHERE ``message_id_end IS
  NOT NULL``. Symmetric to ``message_id_start``.
- ``idx_media_memory_id`` partial index on ``media (memory_id)``
  WHERE ``memory_id IS NOT NULL``. Backs the reverse lookup
  during v016 backfill (existing memories.media_id → media row)
  and any future reverse-traversal callers.

Idempotent via ``ADD COLUMN IF NOT EXISTS`` + ``CREATE INDEX IF
NOT EXISTS``.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_unified_memory_columns",
]

log = get_logger(__name__)


_ADD_CHUNK_MEMORY_ID_SQL = (
    "ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS memory_id UUID NULL"
)

_ADD_CHUNK_MESSAGE_ID_START_SQL = (
    "ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS message_id_start UUID NULL"
)

_ADD_CHUNK_MESSAGE_ID_END_SQL = (
    "ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS message_id_end UUID NULL"
)

_ADD_MEDIA_MEMORY_ID_SQL = (
    "ALTER TABLE media ADD COLUMN IF NOT EXISTS memory_id UUID NULL"
)

_CREATE_CHUNK_MEMORY_ID_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_memory_id "
    "ON memory_chunks (memory_id) WHERE memory_id IS NOT NULL"
)

_CREATE_CHUNK_MEMORY_ID_CHUNK_ID_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_memory_id_chunk_id "
    "ON memory_chunks (memory_id, chunk_id) WHERE memory_id IS NOT NULL"
)

_CREATE_CHUNK_MSG_START_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_message_id_start "
    "ON memory_chunks (message_id_start) WHERE message_id_start IS NOT NULL"
)

_CREATE_CHUNK_MSG_END_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_message_id_end "
    "ON memory_chunks (message_id_end) WHERE message_id_end IS NOT NULL"
)

_CREATE_MEDIA_MEMORY_ID_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_media_memory_id "
    "ON media (memory_id) WHERE memory_id IS NOT NULL"
)


async def add_unified_memory_columns(store: DataStore) -> None:
    """add backlink columns + indexes for the unified memory model.

    No data changes, no constraint changes — pure ADD COLUMN IF NOT
    EXISTS. The backfill (v016), NOT NULL + new FKs (v017), and the
    drop of the old columns + soft-delete columns (v018) follow in
    sequence.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding unified-memory backlink columns (v015)")
    await store.execute(_ADD_CHUNK_MEMORY_ID_SQL)
    await store.execute(_ADD_CHUNK_MESSAGE_ID_START_SQL)
    await store.execute(_ADD_CHUNK_MESSAGE_ID_END_SQL)
    await store.execute(_ADD_MEDIA_MEMORY_ID_SQL)
    await store.execute(_CREATE_CHUNK_MEMORY_ID_IDX_SQL)
    await store.execute(_CREATE_CHUNK_MEMORY_ID_CHUNK_ID_IDX_SQL)
    await store.execute(_CREATE_CHUNK_MSG_START_IDX_SQL)
    await store.execute(_CREATE_CHUNK_MSG_END_IDX_SQL)
    await store.execute(_CREATE_MEDIA_MEMORY_ID_IDX_SQL)

"""agent-memory v035: the indexes no read uses are dropped.

No keyed read in 3tears, the hub, the SDK, the agents or reports leads with these
indexes' columns (searched 2026-11-10); each was a write cost and a crowding of its
table for nothing. The declarations are gone from the collections; this drops them
from a schema an earlier step built them in:

- ``media``: ``ix_media_user_date`` (user_id, date_created), ``ix_media_mime_type``,
  ``idx_media_agent_user`` (agent_id, user_id), ``ix_media_extraction_pending``.
- ``media_content``: ``ix_media_content_user`` (user_id).
- ``memory_chunks``: ``ix_memory_chunks_user`` (user_id), ``idx_chunks_message_id_start``
  (agent_id, message_id_start), ``idx_chunks_message_id_end`` (message_id_end). A
  conversation's transcript chunks are read through ``idx_mem_conversation`` and the
  chunk's own (agent_id, memory_id) key.

Idempotent: ``IF EXISTS`` on every statement, one statement at a time (YugabyteDB runs
no DDL in a transaction).
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["DROP_UNREAD_INDEXES_SQL", "drop_unread_indexes"]

log = get_logger(__name__)


#: one literal statement per index, so the end state reads off this module
DROP_UNREAD_INDEXES_SQL: tuple[str, ...] = (
    "DROP INDEX IF EXISTS ix_media_user_date",
    "DROP INDEX IF EXISTS ix_media_mime_type",
    "DROP INDEX IF EXISTS idx_media_agent_user",
    "DROP INDEX IF EXISTS ix_media_extraction_pending",
    "DROP INDEX IF EXISTS ix_media_content_user",
    "DROP INDEX IF EXISTS ix_memory_chunks_user",
    "DROP INDEX IF EXISTS idx_chunks_message_id_start",
    "DROP INDEX IF EXISTS idx_chunks_message_id_end",
)


async def drop_unread_indexes(store: DataStore) -> None:
    """Drop each index no read uses.

    :param store: ``DataStore`` bound to the target schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("dropping the memory indexes no read uses (v035)")
    for sql in DROP_UNREAD_INDEXES_SQL:
        await store.execute(sql)

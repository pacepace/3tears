"""
agent-memory v004: add lifecycle + conversation-link columns to memories.

memory-task-01. the memories table produced by v001/v003 still lacks
columns the package's own code paths reference:

- ``conversation_id`` — tagged at extraction time and used by the ADD
  path in ``extraction.py``.
- ``message_id_source`` — source message UUID tagged alongside
  ``conversation_id`` by the extractor.
- ``is_deleted`` — soft-delete flag filtered on by ``collections.py``,
  ``retrieval.py``, ``extraction.py``, and ``tools.py``.
- ``media_id`` — optional link from a memory to an uploaded media
  artifact (present on :class:`MemoryEntity`).
- ``date_deleted`` — populated by ``soft_delete`` on collection /
  entity and by the DELETE action in the extractor.
- ``summary`` — optional short form of ``content`` selected by
  ``retrieval.py`` for display and FTS.

Indexes cover the query shapes we see in the code:

- ``(conversation_id)`` — loading live refs by conversation.
- ``(user_id) WHERE is_deleted = FALSE`` — the hot path in retrieval
  already filters on ``is_deleted = false`` with a user scope.

Idempotent: ``ADD COLUMN IF NOT EXISTS`` + ``CREATE INDEX IF NOT EXISTS``.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_lifecycle_columns",
]

log = get_logger(__name__)


_ADD_CONVERSATION_ID_SQL = "ALTER TABLE memories ADD COLUMN IF NOT EXISTS conversation_id UUID NULL"

_ADD_MESSAGE_ID_SOURCE_SQL = "ALTER TABLE memories ADD COLUMN IF NOT EXISTS message_id_source UUID NULL"

_ADD_IS_DELETED_SQL = "ALTER TABLE memories ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN NOT NULL DEFAULT FALSE"

_ADD_MEDIA_ID_SQL = "ALTER TABLE memories ADD COLUMN IF NOT EXISTS media_id UUID NULL"

_ADD_DATE_DELETED_SQL = "ALTER TABLE memories ADD COLUMN IF NOT EXISTS date_deleted TIMESTAMP NULL"

_ADD_SUMMARY_SQL = "ALTER TABLE memories ADD COLUMN IF NOT EXISTS summary TEXT NULL"

_CREATE_MEM_CONVERSATION_IDX_SQL = "CREATE INDEX IF NOT EXISTS idx_mem_conversation ON memories (conversation_id)"

_CREATE_MEM_USER_ACTIVE_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_mem_user_active ON memories (user_id) WHERE is_deleted = FALSE"
)


async def add_lifecycle_columns(store: DataStore) -> None:
    """
    add conversation / lifecycle / media / summary columns to memories.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    """
    log.info("adding memory lifecycle columns (v004)")
    await store.execute(_ADD_CONVERSATION_ID_SQL)
    await store.execute(_ADD_MESSAGE_ID_SOURCE_SQL)
    await store.execute(_ADD_IS_DELETED_SQL)
    await store.execute(_ADD_MEDIA_ID_SQL)
    await store.execute(_ADD_DATE_DELETED_SQL)
    await store.execute(_ADD_SUMMARY_SQL)
    await store.execute(_CREATE_MEM_CONVERSATION_IDX_SQL)
    await store.execute(_CREATE_MEM_USER_ACTIVE_IDX_SQL)

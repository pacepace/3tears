"""agent-memory v033: the legacy index names are gone, said in literal statements.

Every index below is already gone from a schema migrated past v028: v028 drops
each one by name from a tuple, or renames it to its declared twin, and v018's
``DROP COLUMN`` took ``idx_mem_user_active`` (``memories.is_deleted``) and
``idx_chunks_media`` (``memory_chunks.media_id``) with their columns. This step
says so in plain ``DROP INDEX IF EXISTS`` statements, so the end state of the
chain reads off the migrations themselves; on a migrated schema each statement
finds nothing and does nothing.

The exact duplicates among them (``idx_mem_user`` of
``idx_memories_agent_customer_user``, ``idx_mem_agent`` of the primary key,
``idx_media_memory_id`` of ``ix_media_memory_id``, ``idx_chunks_memory_id`` of
``ix_memory_chunks_memory``, ``idx_chunks_user`` of ``ix_memory_chunks_user``)
are the drops approved on 2026-11-10; each keeps its declared twin.

Idempotent: ``IF EXISTS`` on every statement.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["drop_legacy_index_names"]

log = get_logger(__name__)


#: one literal statement per legacy name, so the end state reads off this module
DROP_LEGACY_INDEXES_SQL: tuple[str, ...] = (
    "DROP INDEX IF EXISTS idx_mem_agent",
    "DROP INDEX IF EXISTS idx_mem_customer",
    "DROP INDEX IF EXISTS idx_mem_user",
    "DROP INDEX IF EXISTS idx_mem_embedding",
    "DROP INDEX IF EXISTS idx_mem_user_active",
    "DROP INDEX IF EXISTS idx_media_user",
    "DROP INDEX IF EXISTS idx_media_agent",
    "DROP INDEX IF EXISTS idx_media_memory_id",
    "DROP INDEX IF EXISTS idx_mc_user",
    "DROP INDEX IF EXISTS idx_mc_media",
    "DROP INDEX IF EXISTS idx_mc_embedding",
    "DROP INDEX IF EXISTS idx_chunks_user",
    "DROP INDEX IF EXISTS idx_chunks_media",
    "DROP INDEX IF EXISTS idx_chunks_embedding",
    "DROP INDEX IF EXISTS idx_chunks_memory_id",
    "DROP INDEX IF EXISTS ix_conversation_memory_refs_cid",
)


async def drop_legacy_index_names(store: DataStore) -> None:
    """Drop each legacy index name, which a migrated schema no longer has.

    :param store: ``DataStore`` bound to the target schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("dropping legacy memory index names (v033)")
    for sql in DROP_LEGACY_INDEXES_SQL:
        await store.execute(sql)

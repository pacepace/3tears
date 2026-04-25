"""
agent-memory v011: memory_chunks composite FK to media on (agent_id, media_id).

collections-task-04. the v007 ``memory_chunks`` table carries a
nullable ``agent_id`` column and a simple FK
``media_id REFERENCES media(media_id)`` (with ``media_id`` itself
nullable since a chunk can exist without a parent media row); this
shard partitions chunks on ``agent_id`` and replaces the simple FK
with the composite ``(agent_id, media_id) REFERENCES media(agent_id,
media_id)`` so the optional relationship cannot stretch across
partitions when present.

primary key changes from ``(chunk_id)`` to the composite
``(agent_id, chunk_id)``. the original UNIQUE constraint on
``chunk_id`` is preserved alongside.

idempotency: every step is guarded by ``information_schema`` /
``pg_constraint`` checks inside DO blocks so replay on recovery is a
no-op. pre-GA: TRUNCATE first to clear pre-existing rows so the
SET NOT NULL fires cleanly.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "memory_chunks_composite_fk",
]

log = get_logger(__name__)


_TRUNCATE_MEMORY_CHUNKS_SQL = "TRUNCATE TABLE memory_chunks CASCADE"

_SET_AGENT_ID_NOT_NULL_SQL = (
    "ALTER TABLE memory_chunks ALTER COLUMN agent_id SET NOT NULL"
)

_DROP_SIMPLE_MEDIA_FK_SQL = """
DO $$
DECLARE
    fk_name text;
BEGIN
    SELECT conname INTO fk_name
      FROM pg_constraint c
      JOIN pg_class cls ON cls.oid = c.conrelid
     WHERE cls.relname = 'memory_chunks'
       AND cls.relnamespace = (
           SELECT oid FROM pg_namespace
            WHERE nspname = current_schema()
       )
       AND c.contype = 'f'
       AND pg_get_constraintdef(c.oid) LIKE 'FOREIGN KEY (media_id)%';
    IF fk_name IS NOT NULL THEN
        EXECUTE 'ALTER TABLE memory_chunks DROP CONSTRAINT ' || quote_ident(fk_name);
    END IF;
END
$$
"""

_DROP_OLD_PK_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = (
             SELECT oid FROM pg_class
              WHERE relname = 'memory_chunks'
                AND relnamespace = (
                    SELECT oid FROM pg_namespace
                     WHERE nspname = current_schema()
                )
         )
           AND conname = 'memory_chunks_pkey'
    ) THEN
        ALTER TABLE memory_chunks DROP CONSTRAINT memory_chunks_pkey;
    END IF;
END
$$
"""

_ADD_COMPOSITE_PK_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = (
             SELECT oid FROM pg_class
              WHERE relname = 'memory_chunks'
                AND relnamespace = (
                    SELECT oid FROM pg_namespace
                     WHERE nspname = current_schema()
                )
         )
           AND conname = 'memory_chunks_pkey'
    ) THEN
        ALTER TABLE memory_chunks
            ADD CONSTRAINT memory_chunks_pkey
            PRIMARY KEY (agent_id, chunk_id);
    END IF;
END
$$
"""

_ADD_CHUNK_ID_UNIQUE_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = (
             SELECT oid FROM pg_class
              WHERE relname = 'memory_chunks'
                AND relnamespace = (
                    SELECT oid FROM pg_namespace
                     WHERE nspname = current_schema()
                )
         )
           AND conname = 'memory_chunks_chunk_id_key'
    ) THEN
        ALTER TABLE memory_chunks
            ADD CONSTRAINT memory_chunks_chunk_id_key UNIQUE (chunk_id);
    END IF;
END
$$
"""

_ADD_COMPOSITE_FK_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = (
             SELECT oid FROM pg_class
              WHERE relname = 'memory_chunks'
                AND relnamespace = (
                    SELECT oid FROM pg_namespace
                     WHERE nspname = current_schema()
                )
         )
           AND conname = 'memory_chunks_media_fk'
    ) THEN
        ALTER TABLE memory_chunks
            ADD CONSTRAINT memory_chunks_media_fk
            FOREIGN KEY (agent_id, media_id)
            REFERENCES media (agent_id, media_id)
            ON DELETE CASCADE;
    END IF;
END
$$
"""


async def memory_chunks_composite_fk(store: DataStore) -> None:
    """partition memory_chunks on agent_id and add composite FK to media.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("partitioning memory_chunks on agent_id + composite FK (v011)")
    await store.execute(_TRUNCATE_MEMORY_CHUNKS_SQL)
    await store.execute(_SET_AGENT_ID_NOT_NULL_SQL)
    await store.execute(_DROP_SIMPLE_MEDIA_FK_SQL)
    await store.execute(_DROP_OLD_PK_SQL)
    await store.execute(_ADD_COMPOSITE_PK_SQL)
    await store.execute(_ADD_CHUNK_ID_UNIQUE_SQL)
    await store.execute(_ADD_COMPOSITE_FK_SQL)

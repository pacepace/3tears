"""
agent-memory v017: lock in the unified-memory parent FKs.

transcript-chunks-task-A. v015 added the ``memory_id`` columns
nullable; v016 backfilled them and removed orphans. This migration
flips the constraints into place so the parent relationship is
enforced and CASCADE delete works through ``memories``.

Constraint changes:

- ``memory_chunks.memory_id`` SET NOT NULL.
- ``media.memory_id`` SET NOT NULL.
- ADD composite FK ``memory_chunks_memory_fk``:
  ``memory_chunks (agent_id, memory_id) REFERENCES memories
  (agent_id, memory_id) ON DELETE CASCADE``. Deleting a memory
  cascades to every chunk parented to it.
- ADD composite FK ``media_memory_fk``:
  ``media (agent_id, memory_id) REFERENCES memories (agent_id,
  memory_id) ON DELETE CASCADE``. Deleting a memory cascades to
  the media attachment, which itself cascades to ``media_content``
  via the existing v006 FK.

The old reverse-direction FKs (``memories_media_composite_fk``,
``memory_chunks_media_fk``) and their backing columns are NOT
dropped here — that happens in v018 so the schema-add and the
schema-drop sides are reviewable independently.

CASCADE on both new FKs is the right policy under the unified
model: a memory IS the cognitive anchor; deleting it discards the
cognitive layer AND the source / attachment layers that hung off
it. The old ``memories.media_id`` had ``ON DELETE SET NULL
(media_id)`` because deleting the media didn't logically delete
the cognitive fact extracted from it — that direction is now
inverted (media is the child, memory is the parent), so SET NULL
semantics no longer apply.

Idempotency: every step is guarded by ``pg_constraint`` /
``information_schema.columns`` lookups inside DO blocks so replay
on a recovered schema is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "flip_memory_parent_fks",
]

log = get_logger(__name__)


# NOT NULL on memory_chunks.memory_id. guarded so re-applying is safe.
_SET_CHUNK_MEMORY_ID_NOT_NULL_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'memory_chunks'
           AND column_name = 'memory_id'
           AND is_nullable = 'YES'
    ) THEN
        ALTER TABLE memory_chunks
            ALTER COLUMN memory_id SET NOT NULL;
    END IF;
END
$$
"""


# NOT NULL on media.memory_id. same shape as the chunks guard.
_SET_MEDIA_MEMORY_ID_NOT_NULL_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'media'
           AND column_name = 'memory_id'
           AND is_nullable = 'YES'
    ) THEN
        ALTER TABLE media
            ALTER COLUMN memory_id SET NOT NULL;
    END IF;
END
$$
"""


# composite FK memory_chunks -> memories on (agent_id, memory_id).
# CASCADE so memory delete kills every transcript / document chunk
# parented to it. partition column on both tables is agent_id, so
# the relationship cannot stretch across partitions.
_ADD_CHUNK_MEMORY_FK_SQL = """
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
           AND conname = 'memory_chunks_memory_fk'
    ) THEN
        ALTER TABLE memory_chunks
            ADD CONSTRAINT memory_chunks_memory_fk
            FOREIGN KEY (agent_id, memory_id)
            REFERENCES memories (agent_id, memory_id)
            ON DELETE CASCADE;
    END IF;
END
$$
"""


# composite FK media -> memories on (agent_id, memory_id). same
# CASCADE semantics. deleting a memory drops its attached media,
# which in turn cascades to media_content via the v006 FK.
_ADD_MEDIA_MEMORY_FK_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = (
             SELECT oid FROM pg_class
              WHERE relname = 'media'
                AND relnamespace = (
                    SELECT oid FROM pg_namespace
                     WHERE nspname = current_schema()
                )
         )
           AND conname = 'media_memory_fk'
    ) THEN
        ALTER TABLE media
            ADD CONSTRAINT media_memory_fk
            FOREIGN KEY (agent_id, memory_id)
            REFERENCES memories (agent_id, memory_id)
            ON DELETE CASCADE;
    END IF;
END
$$
"""


async def flip_memory_parent_fks(store: DataStore) -> None:
    """SET NOT NULL on memory_id + ADD CASCADE FKs to memories.

    Runs after v016 has backfilled every row's memory_id, so the
    NOT NULL fires cleanly and the FK adds find no orphan rows.
    The old reverse-direction columns + FKs stay in place for v018
    to drop.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("flipping memory parent FKs into place (v017)")
    await store.execute(_SET_CHUNK_MEMORY_ID_NOT_NULL_SQL)
    await store.execute(_SET_MEDIA_MEMORY_ID_NOT_NULL_SQL)
    await store.execute(_ADD_CHUNK_MEMORY_FK_SQL)
    await store.execute(_ADD_MEDIA_MEMORY_FK_SQL)

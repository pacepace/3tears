"""
agent-memory v018: hard-delete soft-deleted memories + drop legacy
columns / FKs.

transcript-chunks-task-A. The unified memory model is hard-delete
only — soft-delete via ``memories.is_deleted`` is removed. The
reverse-direction FK columns (``memories.media_id``,
``memory_chunks.media_id``) are removed too; their replacements
(``media.memory_id``, ``memory_chunks.memory_id``) were added in
v015, backfilled in v016, and locked in by v017's NOT NULL +
CASCADE FKs.

Order of operations:

Step 1 — hard-delete every memory with ``is_deleted = true``.
v017's new CASCADE FKs propagate the delete to the memory's
chunks and media attachments automatically. Count + log before
firing so operators see what got removed.

Step 2 — DROP CONSTRAINT for the old reverse-direction FKs:
``memories_media_composite_fk`` (v012) and
``memory_chunks_media_fk`` (v011). These must be dropped before
the columns they reference so the column drops succeed.

Step 3 — DROP COLUMN for the legacy parent / lifecycle columns:
``memories.media_id``, ``memories.is_deleted``,
``memories.date_deleted``, ``memory_chunks.media_id``. The first
three were on the ``memories`` table; the last was on
``memory_chunks``. ``media_content.media_id`` is preserved —
media_content's parent is still media (memory wraps media wraps
media_content).

The supporting indexes for the dropped columns (``idx_chunks_media``,
``idx_media_agent`` is unaffected because it's not on a dropped
column) are dropped implicitly with the column.

Idempotency: every DROP is wrapped in IF EXISTS guards or DO
blocks reading from the catalog so replay on a recovered schema
is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "drop_legacy_memory_columns",
]

log = get_logger(__name__)


# hard-delete soft-deleted memories. v017's CASCADE FKs propagate to
# chunks + media attachments. Counted + RAISE NOTICE'd inside the DO
# block so the migration runner only needs ``execute``. Guarded on
# the existence of ``is_deleted`` so the migration is idempotent —
# re-applying after step 3 has already dropped the column is a clean
# no-op rather than an UndefinedColumnError. The MigrationRunner's
# version tracking guards normal flow; this guard adds defense in
# depth for recovery-mode replays.
_HARD_DELETE_SOFT_DELETED_SQL = """
DO $$
DECLARE
    deleted_count int;
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'memories'
           AND column_name = 'is_deleted'
    ) THEN
        WITH deleted AS (
            DELETE FROM memories
             WHERE is_deleted = true
            RETURNING memory_id
        )
        SELECT COUNT(*) INTO deleted_count FROM deleted;
        IF deleted_count > 0 THEN
            RAISE NOTICE 'v018 hard-deleted % previously-soft-deleted memories (cascades to chunks + media)', deleted_count;
        END IF;
    END IF;
END
$$
"""


# drop memories -> media reverse FK (added by v012).
_DROP_MEMORIES_MEDIA_FK_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'memories_media_composite_fk'
           AND conrelid = (
               SELECT oid FROM pg_class
                WHERE relname = 'memories'
                  AND relnamespace = (
                      SELECT oid FROM pg_namespace
                       WHERE nspname = current_schema()
                  )
           )
    ) THEN
        ALTER TABLE memories DROP CONSTRAINT memories_media_composite_fk;
    END IF;
END
$$
"""


# drop memory_chunks -> media reverse FK (added by v011).
_DROP_CHUNKS_MEDIA_FK_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'memory_chunks_media_fk'
           AND conrelid = (
               SELECT oid FROM pg_class
                WHERE relname = 'memory_chunks'
                  AND relnamespace = (
                      SELECT oid FROM pg_namespace
                       WHERE nspname = current_schema()
                  )
           )
    ) THEN
        ALTER TABLE memory_chunks DROP CONSTRAINT memory_chunks_media_fk;
    END IF;
END
$$
"""


# drop the legacy parent / lifecycle columns from memories +
# memory_chunks. CASCADE on DROP COLUMN handles the implicit indexes
# (idx_chunks_media on memory_chunks (media_id)).
_DROP_MEMORIES_MEDIA_ID_COLUMN_SQL = (
    "ALTER TABLE memories DROP COLUMN IF EXISTS media_id"
)

_DROP_MEMORIES_IS_DELETED_COLUMN_SQL = (
    "ALTER TABLE memories DROP COLUMN IF EXISTS is_deleted"
)

_DROP_MEMORIES_DATE_DELETED_COLUMN_SQL = (
    "ALTER TABLE memories DROP COLUMN IF EXISTS date_deleted"
)

_DROP_CHUNKS_MEDIA_ID_COLUMN_SQL = (
    "ALTER TABLE memory_chunks DROP COLUMN IF EXISTS media_id"
)


async def drop_legacy_memory_columns(store: DataStore) -> None:
    """hard-delete soft-deleted rows then drop legacy columns + FKs.

    The CASCADE FKs added in v017 propagate the hard-delete to the
    memory's chunks + media attachments. The FK drops happen before
    the column drops so the column ALTERs find no dependent
    constraint.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("hard-deleting soft-deleted rows + dropping legacy columns (v018)")

    await store.execute(_HARD_DELETE_SOFT_DELETED_SQL)

    # drop reverse-direction FKs before dropping their columns.
    await store.execute(_DROP_MEMORIES_MEDIA_FK_SQL)
    await store.execute(_DROP_CHUNKS_MEDIA_FK_SQL)

    # drop the legacy columns. CASCADE handles dependent indexes.
    await store.execute(_DROP_MEMORIES_MEDIA_ID_COLUMN_SQL)
    await store.execute(_DROP_MEMORIES_IS_DELETED_COLUMN_SQL)
    await store.execute(_DROP_MEMORIES_DATE_DELETED_COLUMN_SQL)
    await store.execute(_DROP_CHUNKS_MEDIA_ID_COLUMN_SQL)

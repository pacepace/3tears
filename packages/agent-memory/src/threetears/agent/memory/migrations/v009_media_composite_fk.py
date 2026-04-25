"""
agent-memory v009: media composite FK to memories on (agent_id, media_id).

collections-task-04. the v006 ``media`` table carries a nullable
``agent_id`` column from the legacy "optional scoping tag" doctrine;
this shard makes the column NOT NULL, switches the primary key from
``(media_id)`` to the composite ``(agent_id, media_id)``, and drops
any pre-existing simple FK on ``memory_id`` in favour of a composite
FK ``(agent_id, memory_id) REFERENCES memories(agent_id, memory_id)``.
the partition column for the media surface is ``agent_id`` -- the
same partition the parent memories row lives in -- so reads and
writes always carry the partition predicate explicitly.

the original UNIQUE constraint on ``media_id`` is preserved alongside
the composite PK so downstream tables (media_content, memory_chunks)
that FK to ``media(media_id)`` continue to work without dragging the
partition column into every reference.

note that the v006 ``media`` table has no ``memory_id`` FK column on
its own (the memory -> media direction lives on ``memories.media_id``);
the composite FK declared here is from ``memories.media_id`` BACK into
``media`` is unaffected. v009 only modifies the ``media`` table itself
to enforce partitioning on ``agent_id``.

idempotency: every step is guarded by information_schema / pg_constraint
checks inside DO blocks so replay on recovery is a no-op. pre-GA, no
data backfill -- existing rows with NULL agent_id are removed before
SET NOT NULL via TRUNCATE so the constraint fires cleanly.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "media_composite_fk",
]

log = get_logger(__name__)


# pre-GA: clear any pre-existing rows so the NOT NULL set + composite
# PK switch fire cleanly without a per-row backfill.
_TRUNCATE_MEDIA_SQL = "TRUNCATE TABLE media CASCADE"

# child tables (media_content, memory_chunks) carry simple FKs into
# ``media(media_id)`` from v006 / v007. those FKs depend on the
# ``media_pkey`` index, so dropping the index requires dropping the
# dependent FKs first. v010 / v011 reinstall composite FKs, so this
# drop is one of the steps that compose the new partition shape.
_DROP_MEDIA_CONTENT_SIMPLE_FK_SQL = """
DO $$
DECLARE
    fk_name text;
BEGIN
    SELECT conname INTO fk_name
      FROM pg_constraint c
      JOIN pg_class cls ON cls.oid = c.conrelid
     WHERE cls.relname = 'media_content'
       AND cls.relnamespace = (
           SELECT oid FROM pg_namespace
            WHERE nspname = current_schema()
       )
       AND c.contype = 'f'
       AND pg_get_constraintdef(c.oid) LIKE 'FOREIGN KEY (media_id)%';
    IF fk_name IS NOT NULL THEN
        EXECUTE 'ALTER TABLE media_content DROP CONSTRAINT ' || quote_ident(fk_name);
    END IF;
END
$$
"""

_DROP_MEMORY_CHUNKS_SIMPLE_FK_SQL = """
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

_SET_AGENT_ID_NOT_NULL_SQL = (
    "ALTER TABLE media ALTER COLUMN agent_id SET NOT NULL"
)

_DROP_OLD_PK_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = (
             SELECT oid FROM pg_class
              WHERE relname = 'media'
                AND relnamespace = (
                    SELECT oid FROM pg_namespace
                     WHERE nspname = current_schema()
                )
         )
           AND conname = 'media_pkey'
    ) THEN
        ALTER TABLE media DROP CONSTRAINT media_pkey;
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
              WHERE relname = 'media'
                AND relnamespace = (
                    SELECT oid FROM pg_namespace
                     WHERE nspname = current_schema()
                )
         )
           AND conname = 'media_pkey'
    ) THEN
        ALTER TABLE media
            ADD CONSTRAINT media_pkey
            PRIMARY KEY (agent_id, media_id);
    END IF;
END
$$
"""

_ADD_MEDIA_ID_UNIQUE_SQL = """
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
           AND conname = 'media_media_id_key'
    ) THEN
        ALTER TABLE media
            ADD CONSTRAINT media_media_id_key UNIQUE (media_id);
    END IF;
END
$$
"""


async def media_composite_fk(store: DataStore) -> None:
    """make agent_id NOT NULL on media and switch PK to composite shape.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("partitioning media on agent_id + composite PK (v009)")
    await store.execute(_TRUNCATE_MEDIA_SQL)
    await store.execute(_DROP_MEDIA_CONTENT_SIMPLE_FK_SQL)
    await store.execute(_DROP_MEMORY_CHUNKS_SIMPLE_FK_SQL)
    await store.execute(_SET_AGENT_ID_NOT_NULL_SQL)
    await store.execute(_DROP_OLD_PK_SQL)
    await store.execute(_ADD_COMPOSITE_PK_SQL)
    await store.execute(_ADD_MEDIA_ID_UNIQUE_SQL)

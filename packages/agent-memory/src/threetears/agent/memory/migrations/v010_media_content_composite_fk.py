"""
agent-memory v010: media_content composite FK to media on (agent_id, media_id).

collections-task-04. the v006 ``media_content`` table carries a
nullable ``agent_id`` column and a simple FK
``media_id REFERENCES media(media_id)``; this shard partitions
content rows on ``agent_id`` (matching the parent media row's
partition) and replaces the simple FK with the composite
``(agent_id, media_id) REFERENCES media(agent_id, media_id)`` so the
relationship cannot stretch across partitions.

primary key changes from ``(content_id)`` to the composite
``(agent_id, content_id)``. the original UNIQUE constraint on
``content_id`` is preserved alongside so any downstream code that
references content rows by id alone keeps working.

idempotency: every step is guarded by ``information_schema`` /
``pg_constraint`` checks inside DO blocks so replay on recovery is a
no-op. pre-GA: TRUNCATE first to clear pre-existing rows so the
SET NOT NULL fires cleanly without per-row backfill.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "media_content_composite_fk",
]

log = get_logger(__name__)


_TRUNCATE_MEDIA_CONTENT_SQL = "TRUNCATE TABLE media_content CASCADE"

_SET_AGENT_ID_NOT_NULL_SQL = (
    "ALTER TABLE media_content ALTER COLUMN agent_id SET NOT NULL"
)

_DROP_SIMPLE_MEDIA_FK_SQL = """
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

_DROP_OLD_PK_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = (
             SELECT oid FROM pg_class
              WHERE relname = 'media_content'
                AND relnamespace = (
                    SELECT oid FROM pg_namespace
                     WHERE nspname = current_schema()
                )
         )
           AND conname = 'media_content_pkey'
    ) THEN
        ALTER TABLE media_content DROP CONSTRAINT media_content_pkey;
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
              WHERE relname = 'media_content'
                AND relnamespace = (
                    SELECT oid FROM pg_namespace
                     WHERE nspname = current_schema()
                )
         )
           AND conname = 'media_content_pkey'
    ) THEN
        ALTER TABLE media_content
            ADD CONSTRAINT media_content_pkey
            PRIMARY KEY (agent_id, content_id);
    END IF;
END
$$
"""

_ADD_CONTENT_ID_UNIQUE_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = (
             SELECT oid FROM pg_class
              WHERE relname = 'media_content'
                AND relnamespace = (
                    SELECT oid FROM pg_namespace
                     WHERE nspname = current_schema()
                )
         )
           AND conname = 'media_content_content_id_key'
    ) THEN
        ALTER TABLE media_content
            ADD CONSTRAINT media_content_content_id_key UNIQUE (content_id);
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
              WHERE relname = 'media_content'
                AND relnamespace = (
                    SELECT oid FROM pg_namespace
                     WHERE nspname = current_schema()
                )
         )
           AND conname = 'media_content_media_fk'
    ) THEN
        ALTER TABLE media_content
            ADD CONSTRAINT media_content_media_fk
            FOREIGN KEY (agent_id, media_id)
            REFERENCES media (agent_id, media_id)
            ON DELETE CASCADE;
    END IF;
END
$$
"""


async def media_content_composite_fk(store: DataStore) -> None:
    """partition media_content on agent_id and add composite FK to media.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("partitioning media_content on agent_id + composite FK (v010)")
    await store.execute(_TRUNCATE_MEDIA_CONTENT_SQL)
    await store.execute(_SET_AGENT_ID_NOT_NULL_SQL)
    await store.execute(_DROP_SIMPLE_MEDIA_FK_SQL)
    await store.execute(_DROP_OLD_PK_SQL)
    await store.execute(_ADD_COMPOSITE_PK_SQL)
    await store.execute(_ADD_CONTENT_ID_UNIQUE_SQL)
    await store.execute(_ADD_COMPOSITE_FK_SQL)

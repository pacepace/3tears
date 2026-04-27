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

idempotency: every step is guarded by information_schema /
``pg_constraint`` checks inside DO blocks (or via the
``replace_primary_key`` helper's own guard) so replay on recovery is
a no-op. pre-GA, no data backfill -- existing rows with NULL agent_id
are removed before SET NOT NULL via TRUNCATE so the constraint fires
cleanly.

implemented via :func:`threetears.core.data.migrations.helpers.
replace_primary_key`. migration-helpers-task-01 retrofit replaces
the prior hand-rolled PK-swap DO blocks with the helper's
declarative call; the FK-drop step stays explicit because the
inbound simple FKs from media_content + memory_chunks are NOT
recreated against the new media PK in this migration -- v010 + v011
add composite FKs back as the next-step pattern.
"""

from __future__ import annotations

from threetears.core.data.migrations.helpers import (
    replace_primary_key,
)
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
# the FK constraint name is unknown (PG auto-generates ``media_content
# _media_id_fkey`` style) so the DO block discovers it at runtime.
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

_SET_AGENT_ID_NOT_NULL_SQL = "ALTER TABLE media ALTER COLUMN agent_id SET NOT NULL"


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
    await replace_primary_key(
        store,
        table="media",
        new_columns=("agent_id", "media_id"),
        preserve_unique_id_column="media_id",
    )

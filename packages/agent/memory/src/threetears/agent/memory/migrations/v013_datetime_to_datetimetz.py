"""
agent-memory v013: TIMESTAMP -> TIMESTAMPTZ for every datetime column
in the agent partition.

collections-task-05: eliminate DATETIME_TYPE across 3tears + 14-eng-ai-bot.
This migration flips the agent-memory tables from naive TIMESTAMP to
aware TIMESTAMPTZ. The corresponding ``Column(...)`` declarations in
``collections.py`` flip from DATETIME_TYPE to DATETIMETZ_TYPE in the
same commit so ``tests/enforcement/test_column_type_alignment.py``
stays green.

Tables and columns:

- ``memories.date_created``, ``memories.date_deleted``,
  ``memories.date_updated``
- ``media.date_created``, ``media.date_updated``
- ``media_content.date_created``
- ``memory_chunks.date_created``
- ``conversation_memory_refs.date_added``

DDL pattern: ``ALTER TABLE <t> ALTER COLUMN <c> TYPE TIMESTAMPTZ
USING <c> AT TIME ZONE 'UTC'``. The ``AT TIME ZONE 'UTC'`` clause is
load-bearing: without it Postgres re-interprets the bare TIMESTAMP
value as the session timezone, silently shifting the wire instant on
non-UTC hosts. We assert here that every TIMESTAMP cell semantically
held UTC (per the project-wide aware-UTC convention) so the
conversion is a byte-stable no-op for already-deployed data.

Idempotent: each ALTER is wrapped in a DO block whose body is guarded
by an ``information_schema.columns`` lookup that runs the ALTER only
when the column is still ``timestamp without time zone``. Re-running
this migration on a schema where the columns are already TIMESTAMPTZ
is a no-op. Replay-safe under the per-agent migration runner.

The literal ``ALTER TABLE ... ALTER COLUMN ... TYPE TIMESTAMPTZ``
fragments are emitted as concrete per-column SQL strings (rather than
a templated DO block iterating a list) so
``tests/enforcement/test_column_type_alignment.py`` -- which AST-
walks every string constant in every migration file -- can match
each ``(table, column) -> TIMESTAMPTZ`` pair against its
``Column(..., DATETIMETZ_TYPE, ...)`` declaration in
``collections.py``.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "datetime_to_datetimetz",
]

log = get_logger(__name__)


# every datetime column in the agent-memory partition. each entry is
# ``(table_name, column_name)``.
_TARGETS: tuple[tuple[str, str], ...] = (
    ("memories", "date_created"),
    ("memories", "date_deleted"),
    ("memories", "date_updated"),
    ("media", "date_created"),
    ("media", "date_updated"),
    ("media_content", "date_created"),
    ("memory_chunks", "date_created"),
    ("conversation_memory_refs", "date_added"),
)


# concrete per-column DO blocks. each block guards the ALTER on an
# ``information_schema.columns`` lookup so replays on TIMESTAMPTZ
# columns are no-ops. ``current_schema()`` pins the lookup to the
# per-agent schema the migration runner sets via search_path. the
# ``AT TIME ZONE 'UTC'`` clause is load-bearing -- without it postgres
# re-interprets the bare TIMESTAMP value as the session timezone and
# silently shifts the wire instant on non-UTC hosts.
_PROMOTE_MEMORIES_DATE_CREATED_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'memories'
           AND column_name = 'date_created'
           AND data_type = 'timestamp without time zone'
    ) THEN
        ALTER TABLE memories
            ALTER COLUMN date_created TYPE TIMESTAMPTZ
                USING date_created AT TIME ZONE 'UTC';
    END IF;
END
$$
"""

_PROMOTE_MEMORIES_DATE_DELETED_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'memories'
           AND column_name = 'date_deleted'
           AND data_type = 'timestamp without time zone'
    ) THEN
        ALTER TABLE memories
            ALTER COLUMN date_deleted TYPE TIMESTAMPTZ
                USING date_deleted AT TIME ZONE 'UTC';
    END IF;
END
$$
"""

_PROMOTE_MEMORIES_DATE_UPDATED_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'memories'
           AND column_name = 'date_updated'
           AND data_type = 'timestamp without time zone'
    ) THEN
        ALTER TABLE memories
            ALTER COLUMN date_updated TYPE TIMESTAMPTZ
                USING date_updated AT TIME ZONE 'UTC';
    END IF;
END
$$
"""

_PROMOTE_MEDIA_DATE_CREATED_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'media'
           AND column_name = 'date_created'
           AND data_type = 'timestamp without time zone'
    ) THEN
        ALTER TABLE media
            ALTER COLUMN date_created TYPE TIMESTAMPTZ
                USING date_created AT TIME ZONE 'UTC';
    END IF;
END
$$
"""

_PROMOTE_MEDIA_DATE_UPDATED_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'media'
           AND column_name = 'date_updated'
           AND data_type = 'timestamp without time zone'
    ) THEN
        ALTER TABLE media
            ALTER COLUMN date_updated TYPE TIMESTAMPTZ
                USING date_updated AT TIME ZONE 'UTC';
    END IF;
END
$$
"""

_PROMOTE_MEDIA_CONTENT_DATE_CREATED_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'media_content'
           AND column_name = 'date_created'
           AND data_type = 'timestamp without time zone'
    ) THEN
        ALTER TABLE media_content
            ALTER COLUMN date_created TYPE TIMESTAMPTZ
                USING date_created AT TIME ZONE 'UTC';
    END IF;
END
$$
"""

_PROMOTE_MEMORY_CHUNKS_DATE_CREATED_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'memory_chunks'
           AND column_name = 'date_created'
           AND data_type = 'timestamp without time zone'
    ) THEN
        ALTER TABLE memory_chunks
            ALTER COLUMN date_created TYPE TIMESTAMPTZ
                USING date_created AT TIME ZONE 'UTC';
    END IF;
END
$$
"""

_PROMOTE_CONVERSATION_MEMORY_REFS_DATE_ADDED_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'conversation_memory_refs'
           AND column_name = 'date_added'
           AND data_type = 'timestamp without time zone'
    ) THEN
        ALTER TABLE conversation_memory_refs
            ALTER COLUMN date_added TYPE TIMESTAMPTZ
                USING date_added AT TIME ZONE 'UTC';
    END IF;
END
$$
"""


_ALL_PROMOTIONS: tuple[str, ...] = (
    _PROMOTE_MEMORIES_DATE_CREATED_SQL,
    _PROMOTE_MEMORIES_DATE_DELETED_SQL,
    _PROMOTE_MEMORIES_DATE_UPDATED_SQL,
    _PROMOTE_MEDIA_DATE_CREATED_SQL,
    _PROMOTE_MEDIA_DATE_UPDATED_SQL,
    _PROMOTE_MEDIA_CONTENT_DATE_CREATED_SQL,
    _PROMOTE_MEMORY_CHUNKS_DATE_CREATED_SQL,
    _PROMOTE_CONVERSATION_MEMORY_REFS_DATE_ADDED_SQL,
)


async def datetime_to_datetimetz(store: DataStore) -> None:
    """promote every naive TIMESTAMP column in agent-memory to TIMESTAMPTZ.

    runs one guarded ALTER per ``(table, column)`` pair listed in
    :data:`_TARGETS`. each ALTER is idempotent via an
    ``information_schema`` lookup so the migration is safe to replay.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info(
        "promoting agent-memory TIMESTAMP columns to TIMESTAMPTZ (v013)",
        extra={
            "extra_data": {
                "column_count": len(_TARGETS),
            },
        },
    )
    for sql in _ALL_PROMOTIONS:
        await store.execute(sql)

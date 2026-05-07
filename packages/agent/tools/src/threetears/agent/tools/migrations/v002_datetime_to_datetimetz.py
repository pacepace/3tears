"""
agent-tools v002: TIMESTAMP -> TIMESTAMPTZ for every datetime column
in the ``context_items`` table.

collections-task-05: eliminate DATETIME_TYPE across 3tears + 14-eng-ai-bot.
This migration flips the agent-tools ``context_items`` table from naive
TIMESTAMP to aware TIMESTAMPTZ. The corresponding ``Column(...)``
declarations in ``collections.py`` flip from DATETIME_TYPE to
DATETIMETZ_TYPE in the same commit so
``tests/enforcement/test_column_type_alignment.py`` stays green.

Tables and columns:

- ``context_items.date_accessed``
- ``context_items.date_created``
- ``context_items.date_updated``

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


# every datetime column in the agent-tools partition. each entry is
# ``(table_name, column_name)``.
_TARGETS: tuple[tuple[str, str], ...] = (
    ("context_items", "date_accessed"),
    ("context_items", "date_created"),
    ("context_items", "date_updated"),
)


# concrete per-column DO blocks. each block guards the ALTER on an
# ``information_schema.columns`` lookup so replays on TIMESTAMPTZ
# columns are no-ops. ``current_schema()`` pins the lookup to the
# per-agent schema the migration runner sets via search_path. the
# ``AT TIME ZONE 'UTC'`` clause is load-bearing -- without it postgres
# re-interprets the bare TIMESTAMP value as the session timezone and
# silently shifts the wire instant on non-UTC hosts.
_PROMOTE_CONTEXT_ITEMS_DATE_ACCESSED_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'context_items'
           AND column_name = 'date_accessed'
           AND data_type = 'timestamp without time zone'
    ) THEN
        ALTER TABLE context_items
            ALTER COLUMN date_accessed TYPE TIMESTAMPTZ
                USING date_accessed AT TIME ZONE 'UTC';
    END IF;
END
$$
"""

_PROMOTE_CONTEXT_ITEMS_DATE_CREATED_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'context_items'
           AND column_name = 'date_created'
           AND data_type = 'timestamp without time zone'
    ) THEN
        ALTER TABLE context_items
            ALTER COLUMN date_created TYPE TIMESTAMPTZ
                USING date_created AT TIME ZONE 'UTC';
    END IF;
END
$$
"""

_PROMOTE_CONTEXT_ITEMS_DATE_UPDATED_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'context_items'
           AND column_name = 'date_updated'
           AND data_type = 'timestamp without time zone'
    ) THEN
        ALTER TABLE context_items
            ALTER COLUMN date_updated TYPE TIMESTAMPTZ
                USING date_updated AT TIME ZONE 'UTC';
    END IF;
END
$$
"""


_ALL_PROMOTIONS: tuple[str, ...] = (
    _PROMOTE_CONTEXT_ITEMS_DATE_ACCESSED_SQL,
    _PROMOTE_CONTEXT_ITEMS_DATE_CREATED_SQL,
    _PROMOTE_CONTEXT_ITEMS_DATE_UPDATED_SQL,
)


async def datetime_to_datetimetz(store: DataStore) -> None:
    """promote every naive TIMESTAMP column in agent-tools to TIMESTAMPTZ.

    runs one guarded ALTER per ``(table, column)`` pair listed in
    :data:`_TARGETS`. each ALTER is idempotent via an
    ``information_schema`` lookup so the migration is safe to replay.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info(
        "promoting agent-tools TIMESTAMP columns to TIMESTAMPTZ (v002)",
        extra={
            "extra_data": {
                "column_count": len(_TARGETS),
            },
        },
    )
    for sql in _ALL_PROMOTIONS:
        await store.execute(sql)

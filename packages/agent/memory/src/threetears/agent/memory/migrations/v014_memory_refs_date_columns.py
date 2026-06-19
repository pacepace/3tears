"""agent-memory v014: rename ``conversation_memory_refs.date_added`` to
``date_created`` and add ``date_updated``.

Why: ``date_added`` was the original column name (migration v002,
inherited from the hub's earlier alembic ``003_memory_refs``). The
name reads as "this row was imported from elsewhere" -- semantics
that were never accurate. Every other 3tears table uses the standard
``(date_created, date_updated)`` convention. The ``BaseCollection.save``
write-path also unconditionally injects ``date_created`` and (on
update) ``date_updated`` into the row dict at the L1 boundary, which
caused every L1 SQLite mirror write for this table to fail with
``no such column: date_created`` until the schema lined up.

This migration:

- renames ``date_added`` -> ``date_created``;
- adds ``date_updated`` (NOT NULL, defaults to ``date_created`` for
  existing rows; future-update path mints ``now()``);
- renames the ``(conversation_id, date_added)`` index to
  ``(conversation_id, date_created)``.

Idempotent (defensive ``IF EXISTS`` / ``IF NOT EXISTS`` clauses so
re-applying against a schema that already shipped the rename is a
no-op).

The corresponding entity / collection / table-factory updates land
in the same commit so ``tests/enforcement/test_column_type_alignment.py``
and the canonical-schema enforcement stay green.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "rename_memory_refs_date_columns",
]

log = get_logger(__name__)


_RENAME_DATE_ADDED_SQL = """
ALTER TABLE conversation_memory_refs
RENAME COLUMN date_added TO date_created
"""

# add date_updated; backfill from date_created for existing rows so
# the NOT NULL constraint can land in one ALTER. future writes set
# both to ``now()`` via BaseCollection.save's standard path.
_ADD_DATE_UPDATED_SQL = """
ALTER TABLE conversation_memory_refs
ADD COLUMN IF NOT EXISTS date_updated TIMESTAMPTZ
"""

_BACKFILL_DATE_UPDATED_SQL = """
UPDATE conversation_memory_refs
SET date_updated = date_created
WHERE date_updated IS NULL
"""

_DATE_UPDATED_NOT_NULL_SQL = """
ALTER TABLE conversation_memory_refs
ALTER COLUMN date_updated SET NOT NULL
"""

# the lookup index from migration v002 was on (conversation_id,
# date_added). rename it alongside the column so the name and the
# columns stay coherent. ``ALTER INDEX ... RENAME TO`` is the
# canonical Postgres path.
_RENAME_INDEX_SQL = """
ALTER INDEX IF EXISTS idx_conv_mem_refs_conversation
RENAME TO idx_conv_mem_refs_conversation_date_created
"""


async def rename_memory_refs_date_columns(store: DataStore) -> None:
    """Rename date_added -> date_created and add date_updated.

    Idempotent via the existence-aware ALTERs above + an
    ``information_schema`` guard for the rename itself: if the
    column already named ``date_created`` is present (replay scenario)
    the rename is skipped. The ADD COLUMN IF NOT EXISTS,
    backfill, and SET NOT NULL clauses are themselves idempotent.

    :param store: DataStore bound to per-agent schema
    :ptype store: DataStore
    """
    log.info("renaming conversation_memory_refs.date_added -> date_created")
    # Guard the rename behind a schema check so re-applying against a
    # post-rename DB is a no-op.
    rename_guarded_sql = """
    DO $$
    BEGIN
        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'conversation_memory_refs'
              AND column_name = 'date_added'
        ) THEN
            ALTER TABLE conversation_memory_refs
            RENAME COLUMN date_added TO date_created;
        END IF;
    END
    $$
    """
    await store.execute(rename_guarded_sql)
    await store.execute(_ADD_DATE_UPDATED_SQL)
    await store.execute(_BACKFILL_DATE_UPDATED_SQL)
    # SET NOT NULL is not directly idempotent; guard via
    # information_schema.is_nullable.
    not_null_guarded_sql = """
    DO $$
    BEGIN
        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'conversation_memory_refs'
              AND column_name = 'date_updated'
              AND is_nullable = 'YES'
        ) THEN
            ALTER TABLE conversation_memory_refs
            ALTER COLUMN date_updated SET NOT NULL;
        END IF;
    END
    $$
    """
    await store.execute(not_null_guarded_sql)
    await store.execute(_RENAME_INDEX_SQL)

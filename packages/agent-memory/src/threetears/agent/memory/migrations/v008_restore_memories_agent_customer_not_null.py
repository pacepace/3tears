"""
agent-memory v008: restore agent_id + customer_id NOT NULL on memories.

collections-task-04. v003 loosened ``memories.agent_id`` and
``memories.customer_id`` to nullable on the theory that early code
paths treated them as optional scoping tags. this shard reverses
that decision: ``agent_id`` is the partition column for the memories
surface, and ``customer_id`` is a required sub-scope. nullable
agent_id breaks the cross-partition retrieval pattern (a row with
``agent_id IS NULL`` would slip past every ``agent_id = $N`` predicate
the new SQL emits) and breaks the AST walker that proves every SQL
literal touching ``memories`` filters by partition.

migration also rewrites the primary key from ``(memory_id)`` to the
composite ``(agent_id, memory_id)`` so the schema enforces row
uniqueness through the partition. the unique constraint on
``memory_id`` alone is preserved alongside the composite PK so child
tables (media, media_content, memory_chunks) can FK by ``memory_id``
without dragging the partition column into every reference; the
composite FK from those children carries the partition column
explicitly.

idempotency: every step is guarded by an information_schema check
inside a DO block so replay on recovery is a no-op. pre-GA there
are no live rows with NULL agent_id / customer_id; the ALTER COLUMN
... SET NOT NULL fires unconditionally because no backfill is
needed.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "restore_memories_agent_customer_not_null",
]

log = get_logger(__name__)


_SET_AGENT_ID_NOT_NULL_SQL = (
    "ALTER TABLE memories ALTER COLUMN agent_id SET NOT NULL"
)

_SET_CUSTOMER_ID_NOT_NULL_SQL = (
    "ALTER TABLE memories ALTER COLUMN customer_id SET NOT NULL"
)

_DROP_OLD_PK_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = (
             SELECT oid FROM pg_class
              WHERE relname = 'memories'
                AND relnamespace = (
                    SELECT oid FROM pg_namespace
                     WHERE nspname = current_schema()
                )
         )
           AND conname = 'memories_pkey'
    ) THEN
        ALTER TABLE memories DROP CONSTRAINT memories_pkey;
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
              WHERE relname = 'memories'
                AND relnamespace = (
                    SELECT oid FROM pg_namespace
                     WHERE nspname = current_schema()
                )
         )
           AND conname = 'memories_pkey'
    ) THEN
        ALTER TABLE memories
            ADD CONSTRAINT memories_pkey
            PRIMARY KEY (agent_id, memory_id);
    END IF;
END
$$
"""

_ADD_MEMORY_ID_UNIQUE_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = (
             SELECT oid FROM pg_class
              WHERE relname = 'memories'
                AND relnamespace = (
                    SELECT oid FROM pg_namespace
                     WHERE nspname = current_schema()
                )
         )
           AND conname = 'memories_memory_id_key'
    ) THEN
        ALTER TABLE memories
            ADD CONSTRAINT memories_memory_id_key UNIQUE (memory_id);
    END IF;
END
$$
"""


async def restore_memories_agent_customer_not_null(store: DataStore) -> None:
    """restore NOT NULL on agent_id / customer_id and switch to composite PK.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info(
        "restoring memories agent_id/customer_id NOT NULL + composite PK (v008)",
    )
    await store.execute(_SET_AGENT_ID_NOT_NULL_SQL)
    await store.execute(_SET_CUSTOMER_ID_NOT_NULL_SQL)
    await store.execute(_DROP_OLD_PK_SQL)
    await store.execute(_ADD_COMPOSITE_PK_SQL)
    await store.execute(_ADD_MEMORY_ID_UNIQUE_SQL)

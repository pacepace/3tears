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

idempotency: ``replace_primary_key`` consults ``pg_get_constraintdef``
and short-circuits when the table is already on the target composite
PK form, so replay on recovery is a no-op. pre-GA there are no live
rows with NULL agent_id / customer_id; the ALTER COLUMN SET NOT NULL
statements fire unconditionally because no backfill is needed.

implemented via :func:`threetears.core.data.migrations.helpers.
replace_primary_key`. migration-helpers-task-01 retrofit replaces
the prior hand-rolled PK-swap DO blocks with one declarative helper
call.
"""

from __future__ import annotations

from threetears.core.data.migrations.helpers import (
    replace_primary_key,
)
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "restore_memories_agent_customer_not_null",
]

log = get_logger(__name__)


_SET_AGENT_ID_NOT_NULL_SQL = "ALTER TABLE memories ALTER COLUMN agent_id SET NOT NULL"

_SET_CUSTOMER_ID_NOT_NULL_SQL = "ALTER TABLE memories ALTER COLUMN customer_id SET NOT NULL"


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
    await replace_primary_key(
        store,
        table="memories",
        new_columns=("agent_id", "memory_id"),
        preserve_unique_id_column="memory_id",
    )

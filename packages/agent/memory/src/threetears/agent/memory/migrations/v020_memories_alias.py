"""
agent-memory v020: add ``memories.alias`` column + per-user unique index.

v0.7.5 (named-anchor feature). Adds an optional TEXT column
``alias`` to ``memories`` plus a partial unique index
``ix_memories_user_alias ON memories(agent_id, user_id, alias)
WHERE alias IS NOT NULL`` so each user can reserve a short name
once per agent partition.

The alias lets the agent skip search entirely on familiar anchors:
``memory_recall(alias='cave-altar')`` resolves to a memory_id in
one query without going through ``memory_search``.

Idempotent: both steps guard on
``information_schema.columns`` / ``pg_indexes`` so replay is a
no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_memories_alias",
]

log = get_logger(__name__)


_ADD_ALIAS_COLUMN_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'memories'
           AND column_name = 'alias'
    ) THEN
        ALTER TABLE memories ADD COLUMN alias TEXT;
    END IF;
END
$$
"""


_CREATE_ALIAS_UNIQUE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS ix_memories_user_alias
    ON memories (agent_id, user_id, alias)
    WHERE alias IS NOT NULL
"""


async def add_memories_alias(store: DataStore) -> None:
    """add the ``alias`` column + per-user unique partial index.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding memories.alias column + unique index (v020)")
    await store.execute(_ADD_ALIAS_COLUMN_SQL)
    await store.execute(_CREATE_ALIAS_UNIQUE_INDEX_SQL)

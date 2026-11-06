"""agent-memory v032: a surfaced ref names the agent that surfaced it.

In a conversation several agents share, what one agent recalled is its own:
the "previously recalled" list an agent reads holds only its refs. The
table gains ``agent_id`` (nullable: a caller that names no agent writes
NULL, and a reader that names none reads every ref).

Backfill, for rows written before the column:

- a ``memory`` ref takes its memory's agent, and a ``chunk`` ref its
  chunk's: both tables are keyed by agent, so the owner is known exactly;
- anything still unset takes the conversation's own agent, which wrote
  every ref before a conversation could hold several agents.

The primary key stays ``(conversation_id, item_id)``: every item a ref
names (a memory, a chunk, a media row) is keyed by its agent, so two agents
never surface the same item.

Idempotent: ``ADD COLUMN IF NOT EXISTS``, and each backfill touches only
rows whose ``agent_id`` is still NULL.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "memory_refs_agent_id",
]

log = get_logger(__name__)


_ADD_AGENT_ID_SQL = "ALTER TABLE conversation_memory_refs ADD COLUMN IF NOT EXISTS agent_id UUID"

_BACKFILL_FROM_MEMORIES_SQL = """
UPDATE conversation_memory_refs r
SET agent_id = m.agent_id
FROM memories m
WHERE r.agent_id IS NULL AND r.item_type = 'memory' AND m.memory_id = r.item_id
"""

_BACKFILL_FROM_CHUNKS_SQL = """
UPDATE conversation_memory_refs r
SET agent_id = c.agent_id
FROM memory_chunks c
WHERE r.agent_id IS NULL AND r.item_type = 'chunk' AND c.chunk_id = r.item_id
"""

_BACKFILL_FROM_CONVERSATIONS_SQL = """
UPDATE conversation_memory_refs r
SET agent_id = cv.agent_id
FROM conversations cv
WHERE r.agent_id IS NULL AND cv.conversation_id = r.conversation_id
"""


async def memory_refs_agent_id(store: DataStore) -> None:
    """Add ``conversation_memory_refs.agent_id`` and backfill it.

    :param store: ``DataStore`` bound to the target schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding conversation_memory_refs.agent_id (v032)")
    await store.execute(_ADD_AGENT_ID_SQL)
    await store.execute(_BACKFILL_FROM_MEMORIES_SQL)
    await store.execute(_BACKFILL_FROM_CHUNKS_SQL)
    await store.execute(_BACKFILL_FROM_CONVERSATIONS_SQL)

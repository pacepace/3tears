"""agent-tools v005: a context item names the agent whose turn wrote it.

In a conversation several agents share, each agent's variables and tool
results are its own: an agent reads only the rows its turns wrote. The
table gains ``agent_id`` (nullable: a caller that names no agent writes
NULL, and a reader that names none reads every row).

Changes:

- add ``agent_id UUID`` IF NOT EXISTS
- backfill it from the conversation's own agent (``conversations.agent_id``):
  before a conversation could hold several agents, every row was written by
  its own agent's turns. Rows whose conversation is gone stay NULL.
- the two keyed-upsert indexes take the agent: a variable name and a tool
  result's dedup key are one row per agent. The new indexes are
  ``NULLS NOT DISTINCT`` (Postgres 15+), so a caller that names no agent
  still holds one row per key. They are built under new names and the old
  ones dropped, so a re-run finds the new shape and builds nothing.

Idempotent: ``ADD COLUMN IF NOT EXISTS``, the backfill touches only NULL
rows, ``CREATE UNIQUE INDEX IF NOT EXISTS`` and ``DROP INDEX IF EXISTS``.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "context_items_agent_id",
]

log = get_logger(__name__)


_ADD_AGENT_ID_SQL = "ALTER TABLE context_items ADD COLUMN IF NOT EXISTS agent_id UUID"

_BACKFILL_AGENT_ID_SQL = """
UPDATE context_items c
SET agent_id = cv.agent_id
FROM conversations cv
WHERE cv.conversation_id = c.conversation_id
  AND c.agent_id IS NULL
"""

_CREATE_AGENT_VAR_KEY_IDX_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_context_items_agent_var_key "
    "ON context_items (conversation_id, agent_id, key) NULLS NOT DISTINCT "
    "WHERE context_type = 'variable'"
)

_CREATE_AGENT_TOOL_RESULT_KEY_IDX_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_context_items_agent_tool_result_key "
    "ON context_items (conversation_id, agent_id, key) NULLS NOT DISTINCT "
    "WHERE context_type = 'tool_result'"
)

_DROP_VAR_KEY_IDX_SQL = "DROP INDEX IF EXISTS ix_context_items_var_key"

_DROP_TOOL_RESULT_KEY_IDX_SQL = "DROP INDEX IF EXISTS ix_context_items_tool_result_key"


async def context_items_agent_id(store: DataStore) -> None:
    """Add ``context_items.agent_id``, backfill it, and key the upsert indexes by agent.

    :param store: ``DataStore`` bound to the target schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding context_items.agent_id and keying its upsert indexes by agent (v005)")
    await store.execute(_ADD_AGENT_ID_SQL)
    await store.execute(_BACKFILL_AGENT_ID_SQL)
    # the new indexes first, so an upsert never runs without one
    await store.execute(_CREATE_AGENT_VAR_KEY_IDX_SQL)
    await store.execute(_CREATE_AGENT_TOOL_RESULT_KEY_IDX_SQL)
    await store.execute(_DROP_VAR_KEY_IDX_SQL)
    await store.execute(_DROP_TOOL_RESULT_KEY_IDX_SQL)

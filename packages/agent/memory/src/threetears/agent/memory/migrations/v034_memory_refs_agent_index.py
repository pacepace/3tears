"""agent-memory v034: the refs one agent surfaced are read by an index of their own.

:meth:`~threetears.agent.memory.collections.MemoryRefsCollection.find_by_conversation`
reads the newest refs one agent surfaced in a conversation, newest first, as
one page of ``idx_conv_mem_refs_conversation_agent_date``, which this step
creates with ``date_created DESC``: YugabyteDB will not read an ascending index
backwards, and sorts the conversation's refs instead. It replaces ``idx_conv_mem_refs_conversation_date_created``: the read
always names its agent now, so the conversation-wide index has no reader and
is dropped.

Guarded (``IF NOT EXISTS`` / ``IF EXISTS``), so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_memory_refs_agent_index"]

log = get_logger(__name__)

_AGENT_REFS_INDEX_SQL = (
    "-- purpose: packages/agent/memory/src/threetears/agent/memory/collections.py"
    "::MemoryRefsCollection.find_by_conversation\n"
    "-- plan_test: packages/agent/memory/tests/integration/test_index_plans.py"
    "::test_conv_mem_refs_conversation_agent_date\n"
    "CREATE INDEX IF NOT EXISTS idx_conv_mem_refs_conversation_agent_date "
    "ON conversation_memory_refs (conversation_id, agent_id, date_created DESC)"
)


_DROP_CONVERSATION_INDEX_SQL = "DROP INDEX IF EXISTS idx_conv_mem_refs_conversation_date_created"


async def add_memory_refs_agent_index(store: DataStore) -> None:
    """Create ``idx_conv_mem_refs_conversation_agent_date`` and drop the conversation-wide date index it replaces.

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("agent-memory v034: one agent's surfaced refs are indexed")
    await store.execute(_AGENT_REFS_INDEX_SQL)
    await store.execute(_DROP_CONVERSATION_INDEX_SQL)

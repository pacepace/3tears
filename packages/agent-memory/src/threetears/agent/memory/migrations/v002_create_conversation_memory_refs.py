"""
agent-memory v002: create conversation_memory_refs table.

translated from the hub's former alembic migration ``003_memory_refs``.
the ledger (``threetears.agent.memory.ledger.MemoryLedger``) tracks
per-conversation memory-surfacing to prevent the agent re-retrieving
items it has already shown the LLM in the same conversation.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

log = get_logger(__name__)


_CREATE_CONVERSATION_MEMORY_REFS_SQL = """
CREATE TABLE IF NOT EXISTS conversation_memory_refs (
    conversation_id UUID NOT NULL,
    item_id UUID NOT NULL,
    item_type VARCHAR(50) NOT NULL,
    short_desc VARCHAR(150) NOT NULL,
    date_added TIMESTAMP NOT NULL,
    CONSTRAINT pk_conversation_memory_refs PRIMARY KEY (conversation_id, item_id)
)
"""

_CREATE_CONV_MEM_REFS_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_conv_mem_refs_conversation "
    "ON conversation_memory_refs (conversation_id, date_added)"
)


async def create_conversation_memory_refs(store: DataStore) -> None:
    """
    create conversation_memory_refs table + lookup index.

    :param store: DataStore bound to per-agent schema
    :ptype store: DataStore
    """
    log.info("creating conversation_memory_refs table")
    await store.execute(_CREATE_CONVERSATION_MEMORY_REFS_SQL)
    await store.execute(_CREATE_CONV_MEM_REFS_IDX_SQL)

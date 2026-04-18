"""
agent-tools v001: create context_items table.

translated byte-equivalent from the former agent-memory v001 migration
where the table previously lived. agent-tools is the natural owner
because :class:`~threetears.agent.tools.context.ToolContextManager` is
the sole writer; memory just happened to ship the DDL in the original
implementation.

statements are unqualified so the L3 broker's ``search_path`` governs
which schema gets the table; every statement is idempotent so replay
on recovery is safe.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_context_items_table",
]

log = get_logger(__name__)


_CREATE_CONTEXT_ITEMS_SQL = """
CREATE TABLE IF NOT EXISTS context_items (
    context_id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL,
    context_type VARCHAR(50) NOT NULL,
    key VARCHAR(255) NOT NULL,
    short_desc VARCHAR(200),
    long_desc VARCHAR(1000),
    content TEXT,
    metadata JSONB,
    date_accessed TIMESTAMP NOT NULL,
    date_created TIMESTAMP NOT NULL,
    date_updated TIMESTAMP NOT NULL
)
"""

_CREATE_CTX_CONVERSATION_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_ctx_conversation "
    "ON context_items (conversation_id)"
)

_CREATE_CTX_CONVERSATION_TYPE_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_ctx_conversation_type "
    "ON context_items (conversation_id, context_type)"
)


async def create_context_items_table(store: DataStore) -> None:
    """
    create the context_items table and its two lookup indexes.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    """
    log.info("creating context_items table")
    await store.execute(_CREATE_CONTEXT_ITEMS_SQL)
    await store.execute(_CREATE_CTX_CONVERSATION_IDX_SQL)
    await store.execute(_CREATE_CTX_CONVERSATION_TYPE_IDX_SQL)

"""
agent-memory v001: create memories table.

translated from the hub's former alembic migration
``001_initial_agent_tables`` -- the conversations table moved to the
new :mod:`threetears.conversations` package and the context_items
table moved to :mod:`threetears.agent.tools`. agent-memory now owns
exactly one table in v001 (``memories``) plus its supporting indexes.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_memories_table",
]

log = get_logger(__name__)


_CREATE_VECTOR_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector"

_CREATE_MEMORIES_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY,
    agent_id UUID NOT NULL,
    customer_id UUID NOT NULL,
    user_id UUID NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    embedding_model VARCHAR(100),
    importance FLOAT,
    metadata JSONB,
    date_created TIMESTAMP NOT NULL,
    date_updated TIMESTAMP NOT NULL,
    date_accessed TIMESTAMP,
    embedding vector(1024)
)
"""

_CREATE_MEM_AGENT_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_mem_agent ON memories (agent_id)"
)

_CREATE_MEM_CUSTOMER_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_mem_customer "
    "ON memories (agent_id, customer_id)"
)

_CREATE_MEM_USER_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_mem_user "
    "ON memories (agent_id, customer_id, user_id)"
)

_CREATE_MEM_EMBEDDING_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_mem_embedding "
    "ON memories USING hnsw (embedding vector_cosine_ops)"
)


async def create_memories_table(store: DataStore) -> None:
    """
    create the memories table plus pgvector extension and indexes.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    """
    log.info("creating memory package tables")
    await store.execute(_CREATE_VECTOR_EXTENSION_SQL)
    await store.execute(_CREATE_MEMORIES_SQL)
    await store.execute(_CREATE_MEM_AGENT_IDX_SQL)
    await store.execute(_CREATE_MEM_CUSTOMER_IDX_SQL)
    await store.execute(_CREATE_MEM_USER_IDX_SQL)
    await store.execute(_CREATE_MEM_EMBEDDING_IDX_SQL)

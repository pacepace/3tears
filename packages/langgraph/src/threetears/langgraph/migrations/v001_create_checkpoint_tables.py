"""
3tears-langgraph v001: create checkpoints and checkpoint_writes tables.

translated from the hub's former alembic migration ``001_initial_agent_tables``.
LangGraph checkpoint persistence via
:class:`~threetears.langgraph.proxy_checkpoint.ProxyCheckpointSaver`
(agent-side, sandboxed via NATS L3 proxy) or
:class:`~threetears.langgraph.checkpoint.ThreeTierCheckpointSaver`
(trusted-service-side with direct asyncpg pool) uses two tables with
string IDs and BYTEA columns for serialized data. the NATS proxy
transmits hex-encoded bytes; the L3 worker on the hub side writes the
bytes as BYTEA.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_checkpoint_tables",
]

log = get_logger(__name__)


_CREATE_CHECKPOINTS_SQL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_ns VARCHAR(255) NOT NULL DEFAULT '',
    checkpoint_id VARCHAR(255) NOT NULL,
    parent_checkpoint_id VARCHAR(255),
    type VARCHAR(255),
    checkpoint BYTEA NOT NULL,
    metadata_ BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
)
"""

_CREATE_CHECKPOINT_WRITES_SQL = """
CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_ns VARCHAR(255) NOT NULL DEFAULT '',
    checkpoint_id VARCHAR(255) NOT NULL,
    task_id VARCHAR(255) NOT NULL,
    task_path VARCHAR(255) NOT NULL DEFAULT '',
    idx INTEGER NOT NULL,
    channel VARCHAR(255) NOT NULL,
    type VARCHAR(255),
    blob BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
)
"""


async def create_checkpoint_tables(store: DataStore) -> None:
    """
    create checkpoints and checkpoint_writes tables.

    :param store: DataStore bound to per-agent schema
    :ptype store: DataStore
    """
    log.info("creating langgraph checkpoint tables")
    await store.execute(_CREATE_CHECKPOINTS_SQL)
    await store.execute(_CREATE_CHECKPOINT_WRITES_SQL)

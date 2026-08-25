"""
3tears-langgraph v001: create checkpoints and checkpoint_writes tables.

translated from the hub's former alembic migration ``001_initial_agent_tables``.
LangGraph checkpoint persistence via
:class:`~threetears.langgraph.checkpoint.ThreeTierCheckpointSaver`
uses two tables with string IDs and BYTEA columns for serialized
data. trusted services back the saver with a direct asyncpg pool
wrapped in :class:`~threetears.langgraph.protocols.AsyncpgPoolAdapter`;
sandboxed agents back it with
:class:`~threetears.core.backends.nats_proxy.NatsProxyL3Backend`,
which transmits hex-encoded bytes over NATS and writes BYTEA
server-side.

**there is deliberately no ``customer_id`` column, and no migration
that adds one.** a multi-tenant direct-pool deployment scopes its
checkpoints by binding the saver to a customer, which folds that
customer into the ``thread_id`` value -- inside the existing primary
key, so a row is unique THROUGH its customer without altering the
key. the reasoning is on
:meth:`~threetears.langgraph.checkpoint.ThreeTierCheckpointSaver.storage_thread_id`;
the short version is that a column would have to JOIN
``(thread_id, checkpoint_ns, checkpoint_id)`` to give that
uniqueness, and altering the key of a table holding live rows in
several deployments is a far larger change than the isolation it
buys.

**migration path for a deployment that already has rows.** adopting
a customer binding needs no DDL, but existing rows were written
under un-prefixed thread ids and a bound saver will not find them:

- a deployment that stays unbound (``customer_id=None``, the
  default) is unaffected in every respect -- same statements, same
  keys, same rows.
- a deployment adopting a binding either re-keys its existing rows
  (``UPDATE checkpoints SET thread_id = $customer || '/' || thread_id``
  for that customer's threads, and the same on ``checkpoint_writes``),
  or accepts that in-flight conversations restart, which for
  checkpoint state means losing resumability of open turns rather
  than losing answered ones.
- re-keying is per-customer and needs the deployment's own mapping
  from thread to customer, which lives in the host's tables (the
  conversation or session row), not here. that mapping is why this
  package ships no re-key script: it cannot know it.
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

"""
conversations v002: add ``message_count`` column to ``conversations`` table.

data-layer-task-01 sub-task 3 lifecycle hook support: the
:meth:`Conversation.record_message` hook increments ``message_count``
on every observed message so admin queries can render conversation
lists without a per-row ``COUNT(*)`` against the (non-existent)
canonical messages table. backfilled to ``0`` for every existing row;
re-running the migration is a clean no-op via the
:func:`add_column_with_backfill` replay guard.

yugabyte-safe shape: the helper splits the ``ALTER TABLE`` and
``UPDATE`` into separate ``store.execute`` calls so the DDL and DML
never share a transaction. see ``project_yugabyte_ddl_dml_separation``.
"""

from __future__ import annotations

from threetears.core.data.migrations.helpers import add_column_with_backfill
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_message_count",
]

log = get_logger(__name__)


async def add_message_count(store: DataStore) -> None:
    """
    add ``message_count INTEGER NOT NULL DEFAULT 0`` to ``conversations``.

    yugabyte-safe via :func:`add_column_with_backfill`: ALTER and
    backfill UPDATE are emitted as two separate ``store.execute``
    calls. the backfill UPDATE is a no-op on a fresh schema (every
    row already carries the default); the column is added with
    ``ADD COLUMN IF NOT EXISTS`` for idempotency.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding message_count column to conversations")
    await add_column_with_backfill(
        store,
        table="conversations",
        column="message_count",
        column_type="INTEGER",
        default="0",
        not_null=True,
        backfill_value_sql="0",
        backfill_predicate="message_count IS NULL",
        backfill_replay_guard=False,
    )

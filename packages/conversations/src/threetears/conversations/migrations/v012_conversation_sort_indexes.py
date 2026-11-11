"""
conversations v012: the conversations listing's sorts each read an index.

An operator lists one agent's conversations (all of them, one user's, or one
status's) sorted by status, by creation date or by last update, a page at a
time. Pace's ruling B (2026-11-11):

- status and date_created are indexed fully, under each of the three pins:
  the agent; the agent and a user; the agent and a status. Five indexes (the
  agent+status pin sorted by status is the agent pin's status index), each
  ending in ``conversation_id``, the page's tie-break.
- "last updated" (date_updated) keeps ONE index, (agent_id, date_updated,
  conversation_id): with a user or status filter the page walks it and
  filters. date_updated moves on every message, so one index is all the
  per-message write pays for it.
- message_count gets none: that sort is deprecated (answered by date_updated,
  the reply's sorted_by says so; removed in 0.96.0).

Every index leads with ``agent_id``: each read binds it by equality (a
per-agent schema holds one agent's rows), so YugabyteDB's hash on the first
column is a point and the rest range.

Idempotent: ``CREATE INDEX IF NOT EXISTS``, one statement at a time
(YugabyteDB runs no DDL in a transaction). The purposes and plan tests are
declared beside each index on ``ConversationsCollection.schema``.

Forward-only: 3tears migrations do not declare downgrades.

Revision ID: 012
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["SORT_INDEX_NAMES", "add_conversation_sort_indexes"]

log = get_logger(__name__)

#: one literal statement per index, in the order they are built
_SORT_INDEXES_SQL: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_conv_agent_created ON conversations (agent_id, date_created, conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_conv_agent_status ON conversations (agent_id, status, conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_conv_agent_user_created"
    " ON conversations (agent_id, user_id, date_created, conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_conv_agent_user_status ON conversations (agent_id, user_id, status, conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_conv_agent_status_created"
    " ON conversations (agent_id, status, date_created, conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_conv_agent_updated ON conversations (agent_id, date_updated, conversation_id)",
)

#: the indexes v012 builds
SORT_INDEX_NAMES: tuple[str, ...] = (
    "idx_conv_agent_created",
    "idx_conv_agent_status",
    "idx_conv_agent_user_created",
    "idx_conv_agent_user_status",
    "idx_conv_agent_status_created",
    "idx_conv_agent_updated",
)


async def add_conversation_sort_indexes(store: DataStore) -> None:
    """
    build the listing's six sort indexes.

    runs in the per-agent schema set by the migration runner's
    ``search_path``; every statement is idempotent on its own.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("v012: building the conversations listing's sort indexes")
    for sql in _SORT_INDEXES_SQL:
        await store.execute(sql)
    log.info("v012: conversation sort indexes complete")

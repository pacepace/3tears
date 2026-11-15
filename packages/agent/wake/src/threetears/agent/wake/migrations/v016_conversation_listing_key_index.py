"""agent-wake v016: a conversation's schedules are indexed newest first, completed by the key.

:meth:`~threetears.agent.wake.collections.WakeScheduleCollection.list_for_conversation` (the REST listing and the
schedule manager) reads every schedule of a conversation through :func:`threetears.core.keyed.keyset_rows`, page by
page, ordered ``date_created DESC, schedule_id DESC``: ``schedule_id`` completes the primary key
``(conversation_id, schedule_id)`` under the pinned conversation, so the order is total, and it runs the way
``date_created`` does. Before this step the only index led by the conversation was the primary key (v001's
``(conversation_id, status)`` is held for the contract drop, v013's is partial on the active ones): each page read
every schedule of the conversation and sorted them, on YugabyteDB with no incremental sort.
``idx_wake_schedules_conv_created`` holds ``(conversation_id, date_created DESC, schedule_id DESC)``, so each page is
one walk of the index from where the last one ended (coordinator ruling 2026-11-14: the index and keyed pages, no
bounded-sort exception).

It covers neither v001's ``(conversation_id, status)`` (a different second column) nor v013's partial index, so it
drops nothing (v001's drop stays in v015, the held contract step). ``IF NOT EXISTS``, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_conversation_listing_index"]

log = get_logger(__name__)

_CONVERSATION_LISTING_INDEX_SQL = (
    "-- purpose: packages/agent/wake/src/threetears/agent/wake/collections.py::WakeScheduleCollection.list_for_conversation\n"
    "-- plan_test: packages/agent/wake/tests/integration/test_index_plans.py::test_wake_schedules_conv_created\n"
    "-- reason: agent_wake_schedules carries six indexes: one conversation's schedules listed newest first\n"
    "CREATE INDEX IF NOT EXISTS idx_wake_schedules_conv_created "
    "ON agent_wake_schedules (conversation_id, date_created DESC, schedule_id DESC)"
)


async def add_conversation_listing_index(store: DataStore) -> None:
    """Build ``idx_wake_schedules_conv_created``.

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_CONVERSATION_LISTING_INDEX_SQL)
    log.info("agent_wake v016: a conversation's schedules indexed newest first")

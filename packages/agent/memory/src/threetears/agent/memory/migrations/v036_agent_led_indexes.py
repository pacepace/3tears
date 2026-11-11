"""agent-memory v036: two reads' indexes lead with the agent, so YugabyteDB takes them.

YugabyteDB's planner (heuristic, as production runs it) takes the index binding the
most columns by equality. Two reads' indexes bound fewer than the owner index beside
them, so YugabyteDB read the owner's rows through that one instead:

- ``ix_memories_user_date`` (user_id, date_created) for hybrid_search's history pool
  (one user's memories older than the recent window). It and ``idx_memories_agent_user``
  (agent_id, user_id, every per-user read) are replaced by one index,
  ``ix_memories_agent_user_date`` (agent_id, user_id, date_created DESC), which YugabyteDB
  takes for both, the history's date bound read off it.
- ``ix_media_memory_id`` (memory_id) for a chunk's file, joined by its memory. YugabyteDB
  hash-joined every file of the agent (2,000 rows for 10) instead. Replaced by
  ``ix_media_agent_memory`` (agent_id, memory_id), each chunk's file one keyed lookup.

Idempotent: ``IF NOT EXISTS`` / ``IF EXISTS`` on every statement, one statement at a time
(YugabyteDB runs no DDL in a transaction); each replacement is built before the index it
replaces is dropped.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["AGENT_LED_INDEXES_SQL", "agent_led_indexes"]

log = get_logger(__name__)


#: one literal statement each, the replacement before the drop, so the end state reads off this module
AGENT_LED_INDEXES_SQL: tuple[str, ...] = (
    "-- purpose: "
    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoriesCollection.count_by_user; "
    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoriesCollection.find_similar_for_dedup; "
    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoriesCollection.find_same_event; "
    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoriesCollection.search_by_semantic; "
    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoriesCollection.search_by_fts; "
    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoriesCollection.list_for_user; "
    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoriesCollection.find_by_scope; "
    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoriesCollection.find_for_user_in_agents; "
    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoriesCollection.hybrid_search._nearest"
    "\n"
    "-- plan_test: packages/agent/memory/tests/integration/test_index_plans.py::test_memories_agent_user\n"
    "CREATE INDEX IF NOT EXISTS ix_memories_agent_user_date ON memories (agent_id, user_id, date_created DESC)",
    "DROP INDEX IF EXISTS ix_memories_user_date",
    "DROP INDEX IF EXISTS idx_memories_agent_user",
    "-- purpose: "
    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoryChunkCollection.search_by_ids; "
    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoryChunkCollection.find_for_recall; "
    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoryChunkCollection.hybrid_search; "
    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoryChunkCollection.search_by_semantic; "
    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoryChunkCollection.hybrid_search_within_memory"
    "\n"
    "-- plan_test: packages/agent/memory/tests/integration/test_index_plans.py::test_media_memory_id\n"
    "CREATE INDEX IF NOT EXISTS ix_media_agent_memory ON media (agent_id, memory_id)",
    "DROP INDEX IF EXISTS ix_media_memory_id",
)


async def agent_led_indexes(store: DataStore) -> None:
    """Build the agent-led replacements, then drop the indexes they replace.

    :param store: ``DataStore`` bound to the target schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("agent-memory v036: the history pool's and a chunk's file's indexes lead with the agent")
    for sql in AGENT_LED_INDEXES_SQL:
        await store.execute(sql)

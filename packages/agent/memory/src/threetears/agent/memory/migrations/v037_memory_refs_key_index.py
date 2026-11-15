"""agent-memory v037: one agent's refs in a conversation are indexed in their page's order, completed by the key.

:meth:`~threetears.agent.memory.collections.MemoryRefsCollection.find_by_conversation` reads one page newest first
through :func:`threetears.core.keyed.keyset_page`, ordered ``date_created DESC, item_id DESC``: ``item_id`` completes
the primary key ``(conversation_id, item_id)`` under the pinned conversation, so the order is total, and it runs the
way ``date_created`` does. v034's ``idx_conv_mem_refs_conversation_agent_date`` holds
``(conversation_id, agent_id, date_created DESC)`` alone: a page completed by the key read and sorted every ref the
agent surfaced in the conversation (YugabyteDB has no incremental sort). ``idx_conv_mem_refs_conversation_agent_key``
holds ``(conversation_id, agent_id, date_created DESC, item_id DESC)``.

It leads with v034's columns, in v034's directions, over the same rows, so it serves every read v034's did (a
previous release's ``ORDER BY date_created DESC`` page mid-rollout): v034's is dropped here, after the new one is built
(covering-drop ruling, 2026-11-14). Guarded (``IF NOT EXISTS`` / ``IF EXISTS``), so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_memory_refs_key_index"]

log = get_logger(__name__)

_KEY_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_conv_mem_refs_conversation_agent_key "
    "ON conversation_memory_refs (conversation_id, agent_id, date_created DESC, item_id DESC)"
)

_DROP_DATE_INDEX_SQL = "DROP INDEX IF EXISTS idx_conv_mem_refs_conversation_agent_date"


async def add_memory_refs_key_index(store: DataStore) -> None:
    """Build ``idx_conv_mem_refs_conversation_agent_key``, then drop v034's index it covers.

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_KEY_INDEX_SQL)
    await store.execute(_DROP_DATE_INDEX_SQL)
    log.info("agent-memory v037: one agent's refs are indexed in their page's order")

"""agent-tools v006 (CONTRACT, held): v003's ``ix_context_items_lru`` and ``ix_context_items_conv`` are dropped.

Neither is read by anything:

- ``ix_context_items_lru`` (``(conversation_id, date_accessed)``): the eviction
  (:meth:`~threetears.agent.tools.collections.ContextItemCollection.evict_lru`) pins the conversation and the
  ``tool_result`` type, which ``ix_context_items_type`` holds, and orders that one conversation's results.
- ``ix_context_items_conv`` (``(conversation_id)``): every read pinning the conversation alone
  (:meth:`~threetears.agent.tools.collections.ContextItemCollection.find_by_conversation`) reads the primary key
  ``(conversation_id, context_id)``, which leads with the same column over the same rows. YugabyteDB already planned
  it through the primary key with the index in place; PostgreSQL reads the key's lead the same way once it is gone
  (or ``ix_context_items_type``, led by it too:
  ``tests/integration/test_index_plans_keys.py::test_find_by_conversation_reads_the_primary_key``).

A pod of an older release may still plan through them mid-rollout, so the step is held:
:func:`~threetears.agent.tools.migrations.register` attaches it only when asked (``held_contract=True``: the test
schemas now, every deployment from the release after this one).

``DROP INDEX IF EXISTS``, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["drop_unread_context_indexes"]

log = get_logger(__name__)

_DROP_LRU_INDEX_SQL = "DROP INDEX IF EXISTS ix_context_items_lru"

_DROP_CONVERSATION_INDEX_SQL = "DROP INDEX IF EXISTS ix_context_items_conv"


async def drop_unread_context_indexes(store: DataStore) -> None:
    """Drop ``ix_context_items_lru`` and ``ix_context_items_conv`` (v006, contract).

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_DROP_LRU_INDEX_SQL)
    await store.execute(_DROP_CONVERSATION_INDEX_SQL)
    log.info("context_items v006 (contract): ix_context_items_lru and ix_context_items_conv dropped")

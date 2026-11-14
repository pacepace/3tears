"""agent-tools v006 (CONTRACT, held): v003's ``ix_context_items_lru`` is dropped.

``ix_context_items_lru`` (``(conversation_id, date_accessed)``) is read by nothing: the eviction
(:meth:`~threetears.agent.tools.collections.ContextItemCollection.evict_lru`) pins the conversation and the
``tool_result`` type, which ``ix_context_items_type`` holds, and orders that one conversation's results; every other
read pins the conversation alone. A pod of an older release may still plan through it mid-rollout, so the step is held:
:func:`~threetears.agent.tools.migrations.register` attaches it only when asked (``held_contract=True``: the test
schemas now, every deployment from the release after this one).

``DROP INDEX IF EXISTS``, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["drop_lru_index"]

log = get_logger(__name__)

_DROP_LRU_INDEX_SQL = "DROP INDEX IF EXISTS ix_context_items_lru"


async def drop_lru_index(store: DataStore) -> None:
    """Drop ``ix_context_items_lru`` (v006, contract).

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_DROP_LRU_INDEX_SQL)
    log.info("context_items v006 (contract): ix_context_items_lru dropped")

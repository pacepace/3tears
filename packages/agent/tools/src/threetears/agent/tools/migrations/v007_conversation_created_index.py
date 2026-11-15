"""agent-tools v007 (expand): ``ix_context_items_conv_created``, a conversation's context in creation order, by key.

:meth:`~threetears.agent.tools.collections.ContextItemCollection.find_by_conversation` reads one conversation's rows
oldest first. ``conversation_id`` is the table's partition column, so the read pins only the partition: a read the
no-whole-reads rule keys only through an index whose purpose names it. ``(conversation_id, date_created)`` holds the
conversation's rows in the order the read returns them, so neither engine sorts them after reading (PostgreSQL read
v003's ``ix_context_items_conv`` and sorted; YugabyteDB read the primary key and sorted).

It fully covers ``ix_context_items_conv`` (``(conversation_id)``: the same leading column over the same rows), so the
covering-drop ruling (Pace, 2026-11-14) drops that one here, in the same migration. ``IF NOT EXISTS`` / ``IF EXISTS``,
one statement at a time (YugabyteDB runs no DDL in a transaction), so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_conversation_created_index"]

log = get_logger(__name__)

_CONVERSATION_CREATED_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS ix_context_items_conv_created ON context_items (conversation_id, date_created)"
)

_DROP_COVERED_INDEX_SQL = "DROP INDEX IF EXISTS ix_context_items_conv"


async def add_conversation_created_index(store: DataStore) -> None:
    """Build ``ix_context_items_conv_created``, then drop the ``ix_context_items_conv`` it covers (v007, expand).

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_CONVERSATION_CREATED_INDEX_SQL)
    await store.execute(_DROP_COVERED_INDEX_SQL)
    log.info("context_items v007: a conversation's rows are read by ix_context_items_conv_created")

"""agent-memory v038: a writer's same-event lookup reads its source messages' memories by key.

:meth:`~threetears.agent.memory.collections.MemoriesCollection.find_same_event` asks, before a writer stores a
memory, for the owner's memories whose ``message_id_source`` is this turn's message or the one before (a handful of
rows), nearest first. No index named the message, so both engines read the whole owner: PostgreSQL scanned
``memories`` sequentially for a 20,000-memory owner and YugabyteDB walked the owner's primary-key range (K1's plan
probe, 2026-11-15; builder S2 re-measured it on the same world before this index: PG ``Seq Scan`` over every row,
YugabyteDB ``memories_pkey`` over the owner's 20,000). ``ix_memories_message_source``
``(message_id_source, agent_id, user_id)`` finds them by key on both. It binds all three of the read's equality pins,
so YugabyteDB's heuristic planner prefers it to the owner index (which binds two). Led by the message, it also serves
a delete of a message where the ``message_id_source`` foreign key is enforced (``ON DELETE SET NULL`` finds the
message's memories by it), which otherwise scans ``memories`` once a deleted message.

No index is dropped: none is covered by this one. Guarded (``IF NOT EXISTS``), so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_memories_message_source_index"]

log = get_logger(__name__)

_MESSAGE_SOURCE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS ix_memories_message_source ON memories (message_id_source, agent_id, user_id)"
)


async def add_memories_message_source_index(store: DataStore) -> None:
    """Build ``ix_memories_message_source``.

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_MESSAGE_SOURCE_INDEX_SQL)
    log.info("agent-memory v038: a writer's same-event lookup reads its source messages' memories by key")

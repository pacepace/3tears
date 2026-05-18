"""
agent-memory v023 (v0.8.2): re-shape ``idx_chunks_message_id_start``
to ``(agent_id, message_id_start)`` to match the schema declaration.

v0.8.1 shipped v022 with a mis-typed column list on this partial
index:

  Schema     (correct): (agent_id, message_id_start) WHERE message_id_start IS NOT NULL
  Migration  (wrong) : (message_id_start, chunk_index) WHERE message_id_start IS NOT NULL

The parity test passed because it compares the schema-derived
SQLAlchemy Table against the hand-written reference fixture (both
of which carry the correct shape); the migration SQL was never
consulted.

Effects of the v022-as-shipped index:

- Any agent pod that replayed v0.8.1's v022 against a fresh schema
  has the wrong-shape index (``(message_id_start, chunk_index)``).
  ``find_by_conversation_id`` queries filter on ``agent_id`` first
  and ``ORDER BY message_id_start, chunk_id`` -- the wrong-shape
  index doesn't lead with the partition column, so the planner
  cannot use it for index-driven ordering. Falls back to bitmap
  index scan + sort.
- Agent pods that never ran v022 (or ran it against a schema where
  the index already existed with the correct shape) are unaffected.

v023 does:

1. ``DROP INDEX IF EXISTS idx_chunks_message_id_start`` -- removes
   whichever shape was installed (correct OR wrong); the
   ``DROP IF EXISTS`` is a no-op on a fresh schema.
2. ``CREATE INDEX IF NOT EXISTS idx_chunks_message_id_start ON
   memory_chunks (agent_id, message_id_start) WHERE
   message_id_start IS NOT NULL`` -- the correct shape.

Idempotent on replay: the drop is unconditional but
``IF EXISTS``-guarded; the create is unconditional but
``IF NOT EXISTS``-guarded. On a schema where v023 already ran, the
drop is a no-op (no wrong-shape index left to drop) and the create
hits ``IF NOT EXISTS`` (already correct shape).

v022's correctness is restored in the v0.8.2 source -- fresh
deploys won't go through the broken path. v023 only matters for
already-deployed agent pods that ran the broken v022.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "fix_idx_chunks_message_id_start",
]

log = get_logger(__name__)


_DROP_OLD_SHAPE_SQL = "DROP INDEX IF EXISTS idx_chunks_message_id_start"

_CREATE_CORRECT_SHAPE_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_message_id_start "
    "ON memory_chunks (agent_id, message_id_start) "
    "WHERE message_id_start IS NOT NULL"
)


async def fix_idx_chunks_message_id_start(store: DataStore) -> None:
    """re-shape idx_chunks_message_id_start to the correct column order.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info(
        "v023: re-shaping idx_chunks_message_id_start to (agent_id, message_id_start) to match schema + prod metallm"
    )
    await store.execute(_DROP_OLD_SHAPE_SQL)
    await store.execute(_CREATE_CORRECT_SHAPE_SQL)

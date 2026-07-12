"""
agent-memory v026: create the ``memory_consolidations`` edge table.

Presence/aliveness program (3tears v0.15.0). The N:1 provenance edge
that Dream consolidation (A5) populates: one row per (gist, source)
pair records that ``consolidated_memory_id`` was synthesised from
``source_memory_id``. N source rows = N sources fanning into one gist.

Mirrors the ``conversation_memory_refs`` edge-table shape:

- partition column ``agent_id`` (matches ``memories``);
- composite PK ``(agent_id, consolidated_memory_id, source_memory_id)``;
- both memory refs are composite FKs to ``memories(agent_id,
  memory_id)`` ON DELETE CASCADE, so deleting either endpoint cleans up
  its edges (the source's own salience is preserved on the source row —
  only the edge is removed);
- ``rationale`` TEXT NULL — the audit trail (why these merged);
- a back-edge index on ``(agent_id, source_memory_id)`` serving the
  "what was this merged into?" lookup and the cycle-guard walk.

Fully non-destructive: sources are never mutated by this table; the edge
lives only in the join. Idempotent: ``CREATE TABLE IF NOT EXISTS`` +
``CREATE INDEX IF NOT EXISTS`` so replay is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_memory_consolidations",
]

log = get_logger(__name__)


_CREATE_MEMORY_CONSOLIDATIONS_SQL = """
CREATE TABLE IF NOT EXISTS memory_consolidations (
    agent_id UUID NOT NULL,
    consolidated_memory_id UUID NOT NULL,
    source_memory_id UUID NOT NULL,
    rationale TEXT NULL,
    date_created TIMESTAMPTZ NOT NULL DEFAULT now(),
    date_updated TIMESTAMPTZ NULL,
    CONSTRAINT pk_memory_consolidations
        PRIMARY KEY (agent_id, consolidated_memory_id, source_memory_id),
    CONSTRAINT fk_memory_consolidations_gist
        FOREIGN KEY (agent_id, consolidated_memory_id)
        REFERENCES memories (agent_id, memory_id) ON DELETE CASCADE,
    CONSTRAINT fk_memory_consolidations_source
        FOREIGN KEY (agent_id, source_memory_id)
        REFERENCES memories (agent_id, memory_id) ON DELETE CASCADE
)
"""

_CREATE_SOURCE_BACK_EDGE_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_memory_consolidations_source ON memory_consolidations (agent_id, source_memory_id)"
)


async def create_memory_consolidations(store: DataStore) -> None:
    """create the ``memory_consolidations`` edge table + back-edge index.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("creating memory_consolidations edge table + back-edge index (v026)")
    await store.execute(_CREATE_MEMORY_CONSOLIDATIONS_SQL)
    await store.execute(_CREATE_SOURCE_BACK_EDGE_IDX_SQL)

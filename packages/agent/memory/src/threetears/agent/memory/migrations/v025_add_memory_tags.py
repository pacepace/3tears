"""
agent-memory v025: nullable JSONB ``tags`` label set + GIN index.

Presence/aliveness program (3tears v0.15.0). One additive change to
``memories``: a ``tags`` JSONB column holding a JSON array of label
strings (e.g. ``["persona", "identity"]``), backed by a GIN index so
containment (``tags @> '["identity"]'``) and existence (``tags ?
'identity'``) queries are index-served rather than sequential scans.

JSONB (not ``text[]``): the schema DSL has no native array type, and
JSONB is forward-compatible with key-value tags later without a further
schema change.

Nullable + mutable + additive — the other five consumers never read it
and existing INSERTs that omit it leave it NULL. Idempotent: ``ADD
COLUMN IF NOT EXISTS`` and ``CREATE INDEX IF NOT EXISTS`` so replay is a
no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_memory_tags",
]

log = get_logger(__name__)


_ADD_TAGS_SQL = "ALTER TABLE memories ADD COLUMN IF NOT EXISTS tags JSONB NULL"

_ADD_TAGS_GIN_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING gin (tags)"


async def add_memory_tags(store: DataStore) -> None:
    """add the ``tags`` JSONB column and its GIN index.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding memory tags JSONB column + GIN index (v025)")
    await store.execute(_ADD_TAGS_SQL)
    await store.execute(_ADD_TAGS_GIN_INDEX_SQL)

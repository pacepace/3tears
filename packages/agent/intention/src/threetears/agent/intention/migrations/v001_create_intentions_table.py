"""
agent-intention v001: create the intentions table.

Presence/aliveness program (3tears v0.15.0). Creates the standing-wants
corpus in a per-agent schema:

- ``CREATE EXTENSION IF NOT EXISTS vector`` (shared, public schema) so
  the ``embedding`` pgvector column + HNSW index resolve.
- A fresh PG enum ``intention_status`` (``open`` / ``asked`` / ``granted``
  / ``dropped``). Postgres has no ``CREATE TYPE IF NOT EXISTS``, so the
  create is guarded by a ``pg_type`` probe scoped to ``current_schema()``
  -- on replay (already-migrated schema) the probe finds the type and the
  ``DO`` block is a no-op; the scope keeps a type in a *sibling* agent
  schema from masking the create in THIS one.
- ``CREATE TABLE IF NOT EXISTS intentions`` -- composite PK
  ``(agent_id, intention_id)`` partitioned on ``agent_id``; ``salience``
  / ``last_decayed_at`` reuse the memory decay substrate; ``status``
  defaults to ``open``.
- Three indexes: the partial salience-ranked deliberation hot path
  (``WHERE status = 'open'``), the cooldown-filter btree, and the HNSW
  dedup index on ``embedding``. All ``IF NOT EXISTS``.

Idempotent throughout: ``IF NOT EXISTS`` on the extension / table /
indexes and the ``current_schema()``-scoped enum guard make replay a
no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_intentions_table",
]

log = get_logger(__name__)


# pgvector installs once at the database level into the public schema;
# per-agent schemas reference the type via the schema-qualified name
# ``public.vector`` (and ``public.vector_cosine_ops`` on the index) so
# the agent-only search_path used during migration does not need to
# expand to find the type. Mirrors agent-memory v001.
_CREATE_VECTOR_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public"

# Postgres has no ``CREATE TYPE IF NOT EXISTS``; the ``pg_type`` probe is
# scoped to ``current_schema()`` so a replay no-ops and a sibling agent
# schema's copy of the type does not mask the create in this one.
_CREATE_INTENTION_STATUS_ENUM_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_type t
          JOIN pg_namespace n ON n.oid = t.typnamespace
         WHERE t.typname = 'intention_status'
           AND n.nspname = current_schema()
    ) THEN
        CREATE TYPE intention_status AS ENUM ('open', 'asked', 'granted', 'dropped');
    END IF;
END
$$;
"""

_CREATE_INTENTIONS_SQL = """
CREATE TABLE IF NOT EXISTS intentions (
    intention_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    customer_id UUID,
    user_id UUID,
    status intention_status NOT NULL DEFAULT 'open',
    content TEXT NOT NULL,
    embedding public.vector(1024),
    salience NUMERIC(5,4) NOT NULL DEFAULT 0.5,
    last_decayed_at TIMESTAMPTZ,
    last_surfaced_at TIMESTAMPTZ,
    source_memory_id UUID,
    source_conversation_id UUID,
    date_created TIMESTAMPTZ NOT NULL,
    date_updated TIMESTAMPTZ,
    PRIMARY KEY (agent_id, intention_id)
)
"""

# the deliberation hot path: a user's open wants ranked by salience.
# Partial on status='open' keeps the index small; salience DESC matches
# the ranked-list scan direction.
_CREATE_OPEN_RANKED_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_intentions_open_ranked "
    "ON intentions (agent_id, user_id, salience DESC) WHERE status = 'open'"
)

# the cooldown filter reads by last-surfaced recency.
_CREATE_LAST_SURFACED_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_intentions_last_surfaced ON intentions (agent_id, last_surfaced_at)"
)

# HNSW over the embedding for the log-time near-duplicate lookup.
_CREATE_EMBEDDING_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS ix_intentions_embedding_hnsw "
    "ON intentions USING hnsw (embedding public.vector_cosine_ops)"
)


async def create_intentions_table(store: DataStore) -> None:
    """create the intentions table plus its enum, extension, and indexes.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("creating agent-intention package tables (v001)")
    await store.execute(_CREATE_VECTOR_EXTENSION_SQL)
    await store.execute(_CREATE_INTENTION_STATUS_ENUM_SQL)
    await store.execute(_CREATE_INTENTIONS_SQL)
    await store.execute(_CREATE_OPEN_RANKED_IDX_SQL)
    await store.execute(_CREATE_LAST_SURFACED_IDX_SQL)
    await store.execute(_CREATE_EMBEDDING_IDX_SQL)

"""
agent-identity v001: create the identity_versions table.

Self-evolution (3tears v0.15.0). Creates the versioned identity-block
store in a per-agent schema:

- Two fresh PG enums: ``identity_block_key`` (``personality`` /
  ``reinforcement`` / ``anti_sycophant`` / ``self_improvement`` /
  ``presence``) and ``identity_version_status`` (``proposed`` / ``active``
  / ``superseded`` / ``rejected``). Postgres has no ``CREATE TYPE IF NOT
  EXISTS``, so each create is guarded by a ``pg_type`` probe scoped to
  ``current_schema()`` -- a replay no-ops and a sibling agent schema's copy
  of the type does not mask the create in this one.
- ``CREATE TABLE IF NOT EXISTS identity_versions`` -- composite PK
  ``(agent_id, version_id)`` partitioned on ``agent_id``; a linear
  parent-pointer version chain (``parent_version_id``); immutable snapshot
  columns (``content`` / ``rationale`` / ``content_hash`` / ``block_key``
  / ``proposer_agent_id``) + the mutable lifecycle (``status`` default
  ``proposed``, ``consenter_user_id``).
- Three indexes: the **partial UNIQUE** ``uq_identity_active_per_block``
  (``WHERE status='active'``) that enforces exactly one active version per
  ``(agent, customer, user, block)``; the block-history btree; and the
  partial pending-queue btree (``WHERE status='proposed'``). All ``IF NOT
  EXISTS``.

Idempotent throughout: ``IF NOT EXISTS`` on the table / indexes and the
``current_schema()``-scoped enum guards make replay a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_identity_versions_table",
]

log = get_logger(__name__)


_CREATE_BLOCK_KEY_ENUM_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_type t
          JOIN pg_namespace n ON n.oid = t.typnamespace
         WHERE t.typname = 'identity_block_key'
           AND n.nspname = current_schema()
    ) THEN
        CREATE TYPE identity_block_key AS ENUM (
            'personality', 'reinforcement', 'anti_sycophant',
            'self_improvement', 'presence'
        );
    END IF;
END
$$;
"""

_CREATE_STATUS_ENUM_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_type t
          JOIN pg_namespace n ON n.oid = t.typnamespace
         WHERE t.typname = 'identity_version_status'
           AND n.nspname = current_schema()
    ) THEN
        CREATE TYPE identity_version_status AS ENUM (
            'proposed', 'active', 'superseded', 'rejected'
        );
    END IF;
END
$$;
"""

_CREATE_IDENTITY_VERSIONS_SQL = """
CREATE TABLE IF NOT EXISTS identity_versions (
    version_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    customer_id UUID,
    user_id UUID,
    block_key identity_block_key NOT NULL,
    content TEXT NOT NULL,
    rationale TEXT,
    content_hash TEXT NOT NULL,
    parent_version_id UUID,
    status identity_version_status NOT NULL DEFAULT 'proposed',
    proposer_agent_id UUID,
    consenter_user_id UUID,
    date_created TIMESTAMPTZ NOT NULL,
    date_updated TIMESTAMPTZ,
    PRIMARY KEY (agent_id, version_id)
)
"""

# exactly one active version per (agent, customer, user, block) -- the
# linear-chain single-winner invariant, as a partial UNIQUE index.
_CREATE_ACTIVE_UNIQUE_IDX_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_identity_active_per_block "
    "ON identity_versions (agent_id, customer_id, user_id, block_key) "
    "WHERE status = 'active'"
)

# a block's version history, most-recent-first.
_CREATE_HISTORY_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_identity_block_history "
    "ON identity_versions (agent_id, user_id, block_key, date_created DESC)"
)

# the pending consent / veto queue.
_CREATE_PENDING_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_identity_pending ON identity_versions (agent_id, user_id) WHERE status = 'proposed'"
)


async def create_identity_versions_table(store: DataStore) -> None:
    """create the identity_versions table plus its enums and indexes.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("creating agent-identity package tables (v001)")
    await store.execute(_CREATE_BLOCK_KEY_ENUM_SQL)
    await store.execute(_CREATE_STATUS_ENUM_SQL)
    await store.execute(_CREATE_IDENTITY_VERSIONS_SQL)
    await store.execute(_CREATE_ACTIVE_UNIQUE_IDX_SQL)
    await store.execute(_CREATE_HISTORY_IDX_SQL)
    await store.execute(_CREATE_PENDING_IDX_SQL)

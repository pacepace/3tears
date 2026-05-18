"""
agent-tools v003: align ``context_items`` shape with prod parity.

shard 03 (v0.8.0) brings every agent-tools ``TableSchema`` to full
prod parity. The v001 migration created ``context_items`` with two
lookup indexes whose names diverge from prod (``idx_ctx_conversation``
/ ``idx_ctx_conversation_type`` vs prod's ``ix_context_items_conv``
/ ``ix_context_items_type``), without the partial-unique
``ix_context_items_var_key``, without the LRU index
``ix_context_items_lru``, without the FK on ``conversation_id``, and
with ``long_desc VARCHAR(1000) NULL`` instead of prod's
``TEXT NOT NULL DEFAULT ''``. v003 closes every gap so 3tears
integration tests carry the shape the v0.8.0
``ContextItemCollection.schema`` declares.

Changes:

- drop the two legacy indexes ``idx_ctx_conversation`` /
  ``idx_ctx_conversation_type`` IF EXISTS
- create the four v0.8.0 indexes (``ix_context_items_conv``,
  ``ix_context_items_type``, ``ix_context_items_lru``,
  ``ix_context_items_var_key`` partial-unique)
- backfill ``long_desc`` NULL -> '' and promote NOT NULL + SET
  DEFAULT '' to match prod
- add the FK ``conversation_id -> conversations(conversation_id) ON
  DELETE CASCADE`` (named ``fk_context_items_conversation``,
  matching prod) IF NOT EXISTS via a ``pg_constraint`` lookup

Idempotency: every CREATE / DROP uses IF EXISTS / IF NOT EXISTS, the
SET NOT NULL is guarded by an ``information_schema`` DO block
(Postgres has no SET NOT NULL IF NOT NULL), the FK add is guarded by
a ``pg_constraint`` DO block (Postgres has no ADD CONSTRAINT IF NOT
EXISTS). v0.8.0 schema parity is the target shape; subsequent test
runs should observe zero phantom migrations between the 3tears
migration output and the prod metallm Alembic output for
``context_items``.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "align_context_items_shape",
]

log = get_logger(__name__)


# Drop the two legacy v001 indexes (the names diverge from prod).
_DROP_LEGACY_CONV_IDX_SQL = "DROP INDEX IF EXISTS idx_ctx_conversation"

_DROP_LEGACY_CONV_TYPE_IDX_SQL = "DROP INDEX IF EXISTS idx_ctx_conversation_type"

# Create the four v0.8.0 indexes matching prod metallm + the v0.8.0
# ``ContextItemCollection.schema`` declaration.
_CREATE_CONV_IDX_SQL = "CREATE INDEX IF NOT EXISTS ix_context_items_conv ON context_items (conversation_id)"

_CREATE_TYPE_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS ix_context_items_type ON context_items (conversation_id, context_type)"
)

_CREATE_LRU_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS ix_context_items_lru ON context_items (conversation_id, date_accessed)"
)

_CREATE_VAR_KEY_IDX_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_context_items_var_key "
    "ON context_items (conversation_id, key) "
    "WHERE context_type = 'variable'"
)

# long_desc: v001 declared ``VARCHAR(1000) NULL`` with no default.
# Prod has ``TEXT NOT NULL DEFAULT ''``. Backfill must come first
# because SET NOT NULL fails on existing NULL rows.
_BACKFILL_LONG_DESC_SQL = "UPDATE context_items SET long_desc = '' WHERE long_desc IS NULL"

# Postgres has no ``ALTER COLUMN ... SET NOT NULL IF NOT NULL``;
# guard the promotion with a DO block.
_PROMOTE_LONG_DESC_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'context_items'
          AND column_name = 'long_desc'
          AND is_nullable = 'YES'
    ) THEN
        ALTER TABLE context_items ALTER COLUMN long_desc SET DEFAULT '';
        ALTER TABLE context_items ALTER COLUMN long_desc SET NOT NULL;
    END IF;
END
$$
"""

# Postgres has no ``ADD CONSTRAINT IF NOT EXISTS``; guard the FK add
# with a ``pg_constraint`` lookup keyed by the prod constraint name.
_ADD_FK_CONVERSATION_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'fk_context_items_conversation'
    ) THEN
        ALTER TABLE context_items
            ADD CONSTRAINT fk_context_items_conversation
            FOREIGN KEY (conversation_id)
            REFERENCES conversations(conversation_id)
            ON DELETE CASCADE;
    END IF;
END
$$
"""


async def align_context_items_shape(store: DataStore) -> None:
    """align 3tears context_items schema with prod (v0.8.0 shard 03).

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("aligning context_items schema with prod (v003)")
    # legacy indexes -> v0.8.0 names
    await store.execute(_DROP_LEGACY_CONV_IDX_SQL)
    await store.execute(_DROP_LEGACY_CONV_TYPE_IDX_SQL)
    await store.execute(_CREATE_CONV_IDX_SQL)
    await store.execute(_CREATE_TYPE_IDX_SQL)
    await store.execute(_CREATE_LRU_IDX_SQL)
    await store.execute(_CREATE_VAR_KEY_IDX_SQL)
    # long_desc shape
    await store.execute(_BACKFILL_LONG_DESC_SQL)
    await store.execute(_PROMOTE_LONG_DESC_SQL)
    # FK to conversations
    await store.execute(_ADD_FK_CONVERSATION_SQL)

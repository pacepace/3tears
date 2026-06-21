"""
agent-tools v003: align ``context_items`` shape with prod parity.

shard 03 (v0.8.0) brings every agent-tools ``TableSchema`` to full
prod parity. The v001 migration created ``context_items`` with two
lookup indexes whose names diverge from prod (``idx_ctx_conversation``
/ ``idx_ctx_conversation_type`` vs prod's ``ix_context_items_conv``
/ ``ix_context_items_type``), without the partial-unique
``ix_context_items_var_key``, without the LRU index
``ix_context_items_lru``, and with ``long_desc VARCHAR(1000) NULL``
instead of prod's ``TEXT NOT NULL DEFAULT ''``. v003 closes every
gap so 3tears integration tests carry the shape the v0.8.0
``ContextItemCollection.schema`` declares.

Changes:

- drop the two legacy indexes ``idx_ctx_conversation`` /
  ``idx_ctx_conversation_type`` IF EXISTS
- create the four v0.8.0 indexes (``ix_context_items_conv``,
  ``ix_context_items_type``, ``ix_context_items_lru``,
  ``ix_context_items_var_key`` partial-unique)
- backfill ``long_desc`` NULL -> '' and promote NOT NULL + SET
  DEFAULT '' to match prod

FK decision (v0.8.0 shard 04.6). Earlier drafts of this migration
added a single-column FK
``context_items.conversation_id -> conversations.conversation_id ON
DELETE CASCADE``. With the v0.8.0 shard 04.6 rename of
``conversations.id -> conversations.conversation_id``, that FK
target column now exists -- but the 3tears ``conversations`` table
has a COMPOSITE primary key ``(agent_id, conversation_id)``, and
Postgres requires every FK to reference a unique or primary-key
column set. A single-column FK against the
``conversation_id`` half of a composite PK is not legal.

The composite FK ``(agent_id, conversation_id)
-> conversations(agent_id, conversation_id)`` would be the correct
shape -- BUT ``context_items`` does NOT carry an ``agent_id``
column on the 3tears side (the table is partitioned by
``conversation_id`` alone; the prod metallm ``context_items``
table likewise has no ``agent_id`` column because metallm's
``conversations`` table has a non-composite PK on ``id`` only).

Therefore: v003 declares NO FK on ``conversation_id``. The 3tears
side relies on app-level cascade (
:class:`ConversationsCollection` is the sole writer to
``conversations`` and triggers context-item cleanup through the
:class:`ContextItemCollection` when a conversation is closed /
deleted). Prod metallm keeps the metallm-shaped single-column FK
because its ``conversations.id`` is a non-composite PK; that side
is unaffected by this divergence.

The ``ContextItemCollection.schema.foreign_keys`` declaration in
:mod:`threetears.agent.tools.collections` keeps the metallm-shaped
single-column FK so the ``to_sqlalchemy_table`` factory produces a
metadata table consumers can introspect. The 3tears integration
test fixtures override the metadata before applying it to a real
postgres so the FK does not surface as a DDL error.

Idempotency: every CREATE / DROP uses IF EXISTS / IF NOT EXISTS, the
SET NOT NULL is guarded by an ``information_schema`` DO block
(Postgres has no SET NOT NULL IF NOT NULL). v0.8.0 schema parity is
the target shape; subsequent test runs should observe zero phantom
migrations between the 3tears migration output and the prod metallm
Alembic output for the ``context_items`` index / column shape.
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
        WHERE table_schema = current_schema()
          AND table_name = 'context_items'
          AND column_name = 'long_desc'
          AND is_nullable = 'YES'
    ) THEN
        ALTER TABLE context_items ALTER COLUMN long_desc SET DEFAULT '';
        ALTER TABLE context_items ALTER COLUMN long_desc SET NOT NULL;
    END IF;
END
$$
"""

# FK on ``conversation_id`` was intentionally dropped from this
# migration. See module docstring "FK decision" section: 3tears
# ``conversations`` carries a composite PK ``(agent_id, conversation_id)``
# and ``context_items`` has no ``agent_id`` column, so no FK shape is
# legal. App-level cascade handles conversation -> context_items
# cleanup; the schema declaration in
# ``ContextItemCollection.schema.foreign_keys`` keeps the metallm-
# shaped single-column FK so ``to_sqlalchemy_table`` produces a
# metadata table consumers can introspect.
# A previously-installed FK from earlier v003 drafts would block
# further work on agent-aware partitioning, so drop it by name if
# present. The constraint name ``fk_context_items_conversation``
# matches the prod metallm declaration; on 3tears agent schemas
# that never carried it, the DROP IF EXISTS is a no-op.
_DROP_LEGACY_FK_CONVERSATION_SQL = "ALTER TABLE context_items DROP CONSTRAINT IF EXISTS fk_context_items_conversation"


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
    # FK on conversation_id: see module docstring; no FK is legal
    # against a composite-PK conversations table from a table that
    # lacks agent_id. Drop any stale FK from earlier v003 drafts.
    await store.execute(_DROP_LEGACY_FK_CONVERSATION_SQL)

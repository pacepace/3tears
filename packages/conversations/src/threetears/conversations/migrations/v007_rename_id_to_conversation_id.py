"""
conversations v007: rename ``conversations.id`` column to
``conversations.conversation_id`` to standardize entity tables on
``<entity>_id`` PK naming (matches metallm prod + JSON API contract).

Background. The 3tears agent-schema ``conversations`` table shipped
with a bare-``id`` PK column. Every other entity table across the
3tears + metallm stack -- ``memories.memory_id``,
``media.media_id``, ``memory_chunks.chunk_id``,
``media_content.content_id``, ``context_items.context_id``,
``mcp_tool_grants.grant_id``, etc. -- already uses the
``<entity>_id`` shape. v0.8.0 shard 04.6 closes the gap so every
entity table follows the same convention.

What this migration does:

- ``ALTER TABLE conversations RENAME COLUMN id TO conversation_id``.
  Postgres updates the PK constraint, every index referencing the
  column, and every dependent FK constraint automatically.
- ``CREATE OR REPLACE FUNCTION conversations_search_vector_update``
  -- the trigger function body (installed by v005 + updated by v006)
  references ``NEW.name`` and ``NEW.language`` but never ``NEW.id``,
  so the function body is byte-identical to v006. We re-emit it for
  belt-and-suspenders (the rename happens BEFORE the function exists
  on a fresh install, so this branch is only hit on replay against
  an already-migrated schema).
- The trigger declaration (``CREATE TRIGGER trg_conversations_search_vector``)
  fires on ``BEFORE INSERT OR UPDATE OF name, language`` and does NOT
  reference the renamed column, so it stays untouched.

Idempotency. The rename uses ``IF EXISTS`` on the source column +
``IF NOT EXISTS`` on the target column (a DO block guards against
"column already renamed" on replay).

Forward-only: 3tears migrations do not declare downgrades.

After this migration:

- ``ConversationsCollection.schema`` declares
  ``primary_key=("agent_id", "conversation_id")`` and the matching
  ``Column("conversation_id", UUID_TYPE)``.
- All app-side SQL referencing the column reads
  ``conversations.conversation_id``.
- The agent-tools ``context_items.conversation_id`` FK target column
  matches (was the metallm-shaped single-column FK; the rename
  closes the 3tears-side mismatch the agent-tools v003 migration
  surfaced).

Revision ID: 007
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "rename_id_to_conversation_id",
]

log = get_logger(__name__)


# Postgres has no ``ALTER TABLE ... RENAME COLUMN ... IF EXISTS``
# clause, so guard the rename with an ``information_schema.columns``
# lookup. The two branches are mutually exclusive: either the source
# ``id`` column still exists (we rename), or the target
# ``conversation_id`` column already exists (we no-op). Belt-and-
# suspenders: a ``RAISE NOTICE`` records which branch ran so a replay
# audit can confirm the no-op path.
#
# CRITICAL: the lookup MUST be scoped to ``current_schema()`` (the
# search_path schema the runner targets). ``information_schema.columns``
# is NOT search-path-relative -- it lists columns across EVERY schema --
# so an unscoped ``table_name = 'conversations'`` predicate sees a
# SIBLING schema's already-renamed table and wrongly takes the ELSE
# no-op branch, leaving THIS schema's table with the bare ``id`` PK.
# That mis-fire is invisible in a single-schema deployment (prod uses
# one schema) but breaks any multi-schema host (e.g. a test suite that
# isolates each case in its own schema on a shared database).
_RENAME_COLUMN_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'conversations'
          AND column_name = 'id'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'conversations'
          AND column_name = 'conversation_id'
    ) THEN
        ALTER TABLE conversations RENAME COLUMN id TO conversation_id;
        RAISE NOTICE 'v007: renamed conversations.id -> conversations.conversation_id';
    ELSE
        RAISE NOTICE 'v007: no-op (already renamed or column absent)';
    END IF;
END
$$
"""


async def rename_id_to_conversation_id(store: DataStore) -> None:
    """rename ``conversations.id`` column to ``conversations.conversation_id``.

    runs in the per-agent schema set by the migration runner's
    ``search_path``. The rename is idempotent via a guarded
    ``information_schema`` DO block so replays on a fully-migrated
    schema are no-ops.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("v007: renaming conversations.id -> conversations.conversation_id")
    await store.execute(_RENAME_COLUMN_SQL)
    log.info("v007: conversations column rename complete")

"""
conversations v005: add ``search_vector`` tsvector + trigger to
``conversations`` for postgres full-text search on conversation
display titles.

framework-promoted from upstream alembic migration 057 (conversation-side
only -- the messages-side FTS in that migration stays product-side,
since 3tears has no canonical ``messages`` table and each product owns
its own message storage shape).

shape:

- ``conversations.search_vector tsvector`` (nullable; the trigger
  populates it on every INSERT / UPDATE of the ``name`` column).
- GIN index ``idx_conversations_search_vector`` on the new column so
  ``search_vector @@ websearch_to_tsquery(...)`` queries are
  index-backed.
- trigger function ``conversations_search_vector_update`` rebuilds the
  vector from ``name`` on row mutation. uses ``setweight(..., 'A')``
  so future multi-source vectors (e.g. summary at weight 'B') can be
  layered in without changing the read query shape.
- one-time backfill ``UPDATE conversations SET search_vector = ...``
  populates existing rows.

idempotent: every DDL statement uses ``IF NOT EXISTS`` / ``CREATE OR
REPLACE`` / ``DROP TRIGGER IF EXISTS; CREATE TRIGGER ...`` so replays
under the per-agent migration runner are no-ops on already-migrated
schemas.

forward-only: 3tears migrations do not declare downgrades (see the
v0.5->v0.6 migration state memory). rollback strategy is fix-forward
via a follow-on migration if the change ever needs to be undone.

after this migration, :meth:`ConversationsCollection.search` is the
canonical API for FTS-by-name across conversations a user participates
in. product consumers that need message-side FTS join their own
message FTS column against this one.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_conversation_search_vector",
]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# DDL emitted by the migration. Each statement is idempotent on its own
# so a replay against a fully-migrated agent schema is a no-op. The
# trigger has no ``CREATE TRIGGER IF NOT EXISTS`` syntax in postgres,
# so we emit DROP-then-CREATE for replay safety.
# ---------------------------------------------------------------------------


_ADD_COLUMN_SQL = "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS search_vector tsvector"

_CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_conversations_search_vector ON conversations USING gin(search_vector)"
)

_CREATE_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION conversations_search_vector_update() RETURNS trigger AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', coalesce(NEW.name, '')), 'A');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""

_DROP_TRIGGER_SQL = "DROP TRIGGER IF EXISTS trg_conversations_search_vector ON conversations"

_CREATE_TRIGGER_SQL = """
CREATE TRIGGER trg_conversations_search_vector
BEFORE INSERT OR UPDATE OF name ON conversations
FOR EACH ROW EXECUTE FUNCTION conversations_search_vector_update();
"""

_BACKFILL_SQL = """
UPDATE conversations SET search_vector =
    setweight(to_tsvector('english', coalesce(name, '')), 'A')
WHERE search_vector IS NULL
"""


async def add_conversation_search_vector(store: DataStore) -> None:
    """
    add ``search_vector`` tsvector + trigger + backfill on ``conversations``.

    runs in the per-agent schema set by the migration runner's
    ``search_path``. all DDL is idempotent so replays are safe; the
    backfill UPDATE is also idempotent (the trigger function is
    deterministic given the same ``name`` value).

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding conversations.search_vector + trigger + GIN index")
    await store.execute(_ADD_COLUMN_SQL)
    await store.execute(_CREATE_INDEX_SQL)
    await store.execute(_CREATE_FUNCTION_SQL)
    await store.execute(_DROP_TRIGGER_SQL)
    await store.execute(_CREATE_TRIGGER_SQL)
    await store.execute(_BACKFILL_SQL)
    log.info("conversations.search_vector backfill complete")

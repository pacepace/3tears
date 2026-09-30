"""
conversations v011: a conversation records what started it.

adds an optional **parent** to ``conversations``: ``parent_type TEXT`` plus
``parent_id UUID``. the pair names the object that started the conversation
-- another conversation, a wake, a person, a webhook, an agent. the type
vocabulary is the consumer's; the platform stores any short type word and
reads none of them.

what this migration does, and nothing else:

- ``ADD COLUMN IF NOT EXISTS parent_type TEXT`` and ``parent_id UUID``, both
  nullable. no backfill: a conversation created before this migration has no
  recorded parent, which is what NULL says.
- a CHECK ``conversations_parent_set_together``: the two are both NULL or
  both NOT NULL. a type without an id, or an id without a type, names
  nothing. ``ADD CONSTRAINT`` has no ``IF NOT EXISTS`` form, so it is guarded
  by a ``pg_constraint`` probe scoped to ``current_schema()`` (the v009
  discipline): replay is a no-op, and a multi-schema host never sees a
  sibling schema's constraint.
- ``CREATE INDEX IF NOT EXISTS idx_conv_parent ON conversations (parent_type,
  parent_id)``, so "every conversation this thing started" is one index scan.

it must apply to a ``conversations`` table that is not this package's own
shape: a consumer may carry a single-column primary key, extra columns, and a
differently named owner column. every statement here names only the two new
columns, the check and the index, so the rest of the table's shape does not
matter.

Forward-only: 3tears migrations do not declare downgrades.

Revision ID: 011
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "PARENT_CHECK_NAME",
    "PARENT_INDEX_NAME",
    "add_conversation_parent",
]

log = get_logger(__name__)

#: the CHECK that keeps the parent pair whole.
PARENT_CHECK_NAME = "conversations_parent_set_together"

#: the index over the parent pair.
PARENT_INDEX_NAME = "idx_conv_parent"

_ADD_PARENT_TYPE_SQL = "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS parent_type TEXT"

_ADD_PARENT_ID_SQL = "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS parent_id UUID"

_ADD_PARENT_CHECK_SQL = f"""
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = '{PARENT_CHECK_NAME}'
          AND connamespace = current_schema()::regnamespace
    ) THEN
        ALTER TABLE conversations
            ADD CONSTRAINT {PARENT_CHECK_NAME}
            CHECK ((parent_type IS NULL) = (parent_id IS NULL));
    END IF;
END $$
"""

_CREATE_PARENT_INDEX_SQL = f"CREATE INDEX IF NOT EXISTS {PARENT_INDEX_NAME} ON conversations (parent_type, parent_id)"


async def add_conversation_parent(store: DataStore) -> None:
    """
    add the nullable parent pair, the both-or-neither check and the index.

    runs in the per-agent schema set by the migration runner's
    ``search_path``. every statement is idempotent on its own, so a replay on
    a migrated schema changes nothing.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("v011: adding conversations.parent_type / parent_id, their check and index")
    await store.execute(_ADD_PARENT_TYPE_SQL)
    await store.execute(_ADD_PARENT_ID_SQL)
    await store.execute(_ADD_PARENT_CHECK_SQL)
    await store.execute(_CREATE_PARENT_INDEX_SQL)
    log.info("v011: conversation parent complete")

"""
conversations v009: complete the folder <-> conversation referential
integrity that v008 left as a bare nullable column.

v008 created the ``folders`` table and added a bare nullable
``conversations.folder_id`` UUID column with NO foreign key, so every
consumer had to hand-roll the referential integrity (an orphaned
``folder_id`` pointing at a deleted folder stayed dangling) and the
unfile-on-delete fan-out. this migration moves both onto the platform.

what this migration does:

- ``CREATE UNIQUE INDEX IF NOT EXISTS uq_folders_folder_id ON folders
  (folder_id)`` -- a standalone single-column unique on ``folder_id``.
  ``folder_id`` is a globally-unique uuid7, so the single-column unique
  is a valid FK *target* for a single-column referencing column. the
  composite primary key ``(agent_id, folder_id)`` cannot be the target
  of a single-column FK, and a composite FK with ``ON DELETE SET NULL``
  would try to null BOTH referencing columns -- including the
  ``NOT NULL`` ``agent_id`` partition column -- which postgres rejects.
  the single-column unique is the clean way to express the FK.
- add the foreign key ``conversations.folder_id -> folders.folder_id
  ON DELETE SET NULL`` (named ``conversations_folder_id_fkey``).
  deleting a folder now auto-unfiles its conversations at the DB level
  (their ``folder_id`` is nulled) instead of leaving them pointing at a
  row that no longer exists. the FK is the DB-level backstop; the
  cache-coherent path consumers call when they hold L1/L2 caches is
  :meth:`ConversationsCollection.clear_folder`.

idempotency. ``CREATE UNIQUE INDEX IF NOT EXISTS`` is natively
idempotent and search-path-relative. ``ALTER TABLE ... ADD CONSTRAINT``
has no ``IF NOT EXISTS`` form, so the FK is guarded by a
``pg_constraint`` existence probe inside a ``DO`` block scoped to
``current_schema()`` -- on replay (already-migrated schema) the probe
finds the constraint and the block is a clean no-op, and a multi-schema
test host (each case isolated in its own schema on a shared database)
only ever sees / mutates the constraint in its own schema. matches the
``information_schema`` / ``current_schema()`` discipline v007 uses for
its un-guardable ``RENAME COLUMN``.

Forward-only: 3tears migrations do not declare downgrades.

Revision ID: 009
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_folder_referential_integrity",
]

log = get_logger(__name__)


# A standalone single-column unique on ``folder_id`` so a single-column
# FK can reference it. ``folder_id`` is a globally-unique uuid7, so this
# does not weaken the per-owner ``uq_folders_agent_user_name`` uniqueness
# v008 already enforces -- it simply makes ``folder_id`` itself a valid
# single-column FK target (the composite PK is not).
_CREATE_FOLDER_ID_UNIQUE_SQL = "CREATE UNIQUE INDEX IF NOT EXISTS uq_folders_folder_id ON folders (folder_id)"

# The FK from a conversation to its folder, with ``ON DELETE SET NULL``
# so deleting a folder auto-unfiles its conversations at the DB level.
# ``ADD CONSTRAINT`` has no ``IF NOT EXISTS`` form; the ``pg_constraint``
# probe is scoped to ``current_schema()`` so replays no-op and a
# multi-schema test host never touches a sibling schema's constraint.
_ADD_CONVERSATION_FOLDER_FK_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'conversations_folder_id_fkey'
          AND connamespace = current_schema()::regnamespace
    ) THEN
        ALTER TABLE conversations
            ADD CONSTRAINT conversations_folder_id_fkey
            FOREIGN KEY (folder_id)
            REFERENCES folders (folder_id)
            ON DELETE SET NULL;
    END IF;
END $$
"""


async def add_folder_referential_integrity(store: DataStore) -> None:
    """
    add the folder ``folder_id`` unique + the conversation->folder FK.

    runs in the per-agent schema set by the migration runner's
    ``search_path``. the unique index is natively idempotent
    (``IF NOT EXISTS``); the FK is added inside a ``pg_constraint``
    existence probe scoped to ``current_schema()`` so replays on a
    fully-migrated schema are no-ops and a multi-schema test host never
    crosses schema boundaries.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("v009: adding folders.folder_id unique + conversations.folder_id FK (ON DELETE SET NULL)")
    await store.execute(_CREATE_FOLDER_ID_UNIQUE_SQL)
    await store.execute(_ADD_CONVERSATION_FOLDER_FK_SQL)
    log.info("v009: folder referential integrity complete")

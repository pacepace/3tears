"""
conversations v008: create the ``folders`` table and add the mutable
``conversations.folder_id`` column.

a folder is an app-agnostic, mutable, per-owner named container that
groups conversations -- a product-side folder
primitive so any 3tears consumer that organises conversations under
named containers reuses one canonical entity instead of re-inventing a
per-product table. the canonical shape carries only scope + name +
timestamps; app-specific presentation (color, sort order, icon) lives
in the ``metadata`` JSONB blob so a new consumer never has to migrate
the table to carry its own bits.

what this migration does:

- ``CREATE TABLE IF NOT EXISTS folders`` with the composite primary
  key ``(agent_id, folder_id)`` (partition on ``agent_id``, matching
  the ``conversations`` table). ``date_created`` / ``date_updated``
  are declared ``TIMESTAMPTZ`` up front (fresh table -- no
  TIMESTAMP->TIMESTAMPTZ promotion needed) so the column-type
  alignment enforcement matches the ``DATETIMETZ_TYPE`` Column
  declarations in ``folder_collection.py``.
- ``CREATE UNIQUE INDEX IF NOT EXISTS uq_folders_agent_user_name`` on
  ``(agent_id, user_id, name)`` -- folders are scoped per user, so a
  folder name is unique within one owner (two users under the same
  agent can each have a ``"Work"`` folder). a unique index is the
  idempotent (``IF NOT EXISTS``) way to express the UNIQUE constraint;
  postgres' table-level ``UNIQUE (...)`` clause has no ``IF NOT
  EXISTS`` form.
- ``CREATE INDEX IF NOT EXISTS idx_folders_user`` on
  ``(agent_id, user_id)`` -- the lookup index backing
  :meth:`FolderCollection.find_by_user`.
- ``ALTER TABLE conversations ADD COLUMN IF NOT EXISTS folder_id UUID``
  -- the MUTABLE foreign key from a conversation to its folder.
  nullable (conversations are created unfiled) and intentionally NOT
  immutable (conversations move between folders over their lifetime).

idempotency. every statement uses a natively-idempotent,
search-path-relative DDL form: ``CREATE TABLE IF NOT EXISTS``,
``CREATE UNIQUE INDEX IF NOT EXISTS`` / ``CREATE INDEX IF NOT EXISTS``,
and ``ADD COLUMN IF NOT EXISTS``. these resolve against the
``current_schema()`` search_path the migration runner sets, so replay
on a fully-migrated schema is a clean no-op AND a multi-schema test
host (each case isolated in its own schema on a shared database) never
sees a sibling schema's table -- the same multi-schema isolation v007
guards with its explicit ``information_schema`` / ``current_schema()``
probe (only needed there because ``RENAME COLUMN`` has no ``IF EXISTS``
form). no ``information_schema`` probe is required here because none of
these statements interrogate cross-schema catalog tables.

Forward-only: 3tears migrations do not declare downgrades.

Revision ID: 008
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_folders_and_conversation_folder_id",
]

log = get_logger(__name__)


# DDL is unqualified so the L3 broker's ``search_path`` governs which
# schema gets the table. ``date_created`` / ``date_updated`` are
# TIMESTAMPTZ up front to align with the ``DATETIMETZ_TYPE`` Column
# declarations on ``FolderCollection`` (the column-type-alignment
# enforcement walker reads this literal).
_CREATE_FOLDERS_SQL = """
CREATE TABLE IF NOT EXISTS folders (
    agent_id UUID NOT NULL,
    folder_id UUID NOT NULL,
    customer_id UUID NOT NULL,
    user_id UUID NOT NULL,
    name TEXT NOT NULL,
    metadata JSONB,
    date_created TIMESTAMPTZ NOT NULL,
    date_updated TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (agent_id, folder_id)
)
"""

# folders are scoped per user: a folder name is unique within one
# ``(agent_id, user_id)`` owner. a unique index is the idempotent
# (``IF NOT EXISTS``) expression of the UNIQUE constraint -- postgres'
# table-level ``UNIQUE (...)`` clause has no ``IF NOT EXISTS`` form.
_CREATE_FOLDERS_UNIQUE_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_folders_agent_user_name ON folders (agent_id, user_id, name)"
)

# lookup index backing FolderCollection.find_by_user (scoped by the
# partition column + owner).
_CREATE_FOLDERS_USER_IDX_SQL = "CREATE INDEX IF NOT EXISTS idx_folders_user ON folders (agent_id, user_id)"

# the mutable FK from a conversation to its folder. nullable
# (conversations start unfiled) and NOT immutable (they move between
# folders over their lifetime).
_ADD_CONVERSATION_FOLDER_ID_SQL = "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS folder_id UUID"


async def create_folders_and_conversation_folder_id(store: DataStore) -> None:
    """
    create the ``folders`` table and add ``conversations.folder_id``.

    runs in the per-agent schema set by the migration runner's
    ``search_path``. every statement is natively idempotent
    (``IF NOT EXISTS`` / ``ADD COLUMN IF NOT EXISTS``) and
    search-path-relative, so replays on a fully-migrated schema are
    no-ops and a multi-schema test host never crosses schema
    boundaries.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("v008: creating folders table + conversations.folder_id column")
    await store.execute(_CREATE_FOLDERS_SQL)
    await store.execute(_CREATE_FOLDERS_UNIQUE_SQL)
    await store.execute(_CREATE_FOLDERS_USER_IDX_SQL)
    await store.execute(_ADD_CONVERSATION_FOLDER_ID_SQL)
    log.info("v008: folders table + conversations.folder_id complete")

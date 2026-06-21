"""
3tears-conversations package migrations.

single entry point :func:`register` wires the package's versioned
migration callables into a shared :class:`~threetears.core.data.
migrations.runner.MigrationRunner`. the conversations package owns
exactly one table in every agent schema -- ``conversations`` -- plus
the three indexes that support scoped queries.

the package declares no ``depends_on`` edges: it is the root of the
agent-schema dependency graph, and every other package that keys off
``conversation_id`` declares ``depends_on=("conversations",)``.

version history:

- v001 -- create the ``conversations`` table plus the three lookup
  indexes that support scoped queries (user, customer, status).
- v002 -- add the ``message_count INT`` counter column with a backfill
  that re-derives the counter from the messages table.
- v003 -- add the ``name TEXT`` nullable column for human-facing
  display labels (titles distinct from rolling summaries).
- v004 -- collections-task-05 phase A4: promote every TIMESTAMP column
  on ``conversations`` (``date_created``, ``date_updated``,
  ``date_last_message``) to TIMESTAMPTZ via
  ``ALTER ... USING ... AT TIME ZONE 'UTC'``. the corresponding
  ``Column(...)`` declarations in ``collection.py`` flip from
  DATETIME_TYPE to DATETIMETZ_TYPE in the same commit so the
  Column<->migration alignment enforcement test stays green.
- v005 -- v0.7.0 framework promotion: add ``search_vector`` tsvector +
  trigger + GIN index for postgres FTS on conversation display titles.
  lifted from metallm migration 057 (conversation-side only -- the
  messages-side FTS in that migration stays product-side because
  3tears has no canonical messages table).
- v006 -- v0.7.0 review item #7: add per-row ``language`` column +
  update the FTS trigger to read it instead of hard-coding ``english``.
  Future polyglot consumers set the per-conversation language without
  another migration; existing rows backfill to ``'english'`` via
  the column default.
- v007 -- v0.8.0 shard 04.6: rename the bare-``id`` PK column to
  ``conversation_id`` so the entity table follows the canonical
  ``<entity>_id`` naming used everywhere else in the stack (memory_id,
  media_id, context_id, etc.). Postgres updates the PK + every index
  + every dependent FK automatically; the rename is guarded by an
  ``information_schema`` DO block so replays are idempotent.
- v008 -- create the app-agnostic ``folders`` table (the Folder
  primitive lifted from metallm: a mutable, per-owner named container
  grouping conversations) plus the UNIQUE(agent_id, user_id, name)
  constraint and its lookup index, and add the mutable
  ``conversations.folder_id`` FK column. every statement is natively
  idempotent (``IF NOT EXISTS`` / ``ADD COLUMN IF NOT EXISTS``) and
  search-path-relative.
- v009 -- complete the folder<->conversation referential integrity v008
  left as a bare nullable column: add a standalone single-column UNIQUE
  on ``folders.folder_id`` (so a single-column FK can target it) and the
  ``conversations.folder_id -> folders.folder_id ON DELETE SET NULL`` FK
  (``conversations_folder_id_fkey``) so deleting a folder auto-unfiles
  its conversations at the DB level. the unique index is natively
  idempotent; the FK is guarded by a ``pg_constraint`` /
  ``current_schema()`` probe (``ADD CONSTRAINT`` has no ``IF NOT EXISTS``
  form), matching v007's discipline.
"""

from __future__ import annotations

from threetears.conversations.migrations.v001_create_conversations_table import (
    create_conversations_table,
)
from threetears.conversations.migrations.v002_add_message_count import (
    add_message_count,
)
from threetears.conversations.migrations.v003_add_name_column import (
    add_name_column,
)
from threetears.conversations.migrations.v004_datetime_to_datetimetz import (
    datetime_to_datetimetz,
)
from threetears.conversations.migrations.v005_conversation_search_vector import (
    add_conversation_search_vector,
)
from threetears.conversations.migrations.v006_conversation_language_column import (
    add_conversation_language_column,
)
from threetears.conversations.migrations.v007_rename_id_to_conversation_id import (
    rename_id_to_conversation_id,
)
from threetears.conversations.migrations.v008_create_folders_and_conversation_folder_id import (
    create_folders_and_conversation_folder_id,
)
from threetears.conversations.migrations.v009_folder_referential_integrity import (
    add_folder_referential_integrity,
)
from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

PACKAGE_NAME = "conversations"


def register(runner: MigrationRunner) -> PackageMigrations:
    """
    register conversations migrations with the given runner.

    produces an agent-scoped :class:`PackageMigrations` with no
    ``depends_on`` edges, attaches every migration in version order,
    and calls ``runner.register``.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.AGENT,
    )
    pkg.version(1)(create_conversations_table)
    pkg.version(2)(add_message_count)
    pkg.version(3)(add_name_column)
    pkg.version(4)(datetime_to_datetimetz)
    pkg.version(5)(add_conversation_search_vector)
    pkg.version(6)(add_conversation_language_column)
    pkg.version(7)(rename_id_to_conversation_id)
    pkg.version(8)(create_folders_and_conversation_folder_id)
    pkg.version(9)(add_folder_referential_integrity)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "add_conversation_language_column",
    "add_folder_referential_integrity",
    "add_conversation_search_vector",
    "add_message_count",
    "add_name_column",
    "create_conversations_table",
    "create_folders_and_conversation_folder_id",
    "datetime_to_datetimetz",
    "register",
    "rename_id_to_conversation_id",
]

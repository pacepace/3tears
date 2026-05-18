"""
agent-tools package migrations.

single entry point :func:`register` wires the package's versioned
migration callables into a shared :class:`~threetears.core.data.
migrations.runner.MigrationRunner`. agent-tools owns the
``context_items`` table consumed by
:class:`~threetears.agent.tools.context.ToolContextManager`. ownership
moved here from agent-memory because the table is conversation-scoped
context surfaced to the LLM at tool-call time, not extracted memory.

the package declares ``depends_on=("conversations",)`` because every
``context_items`` row references a ``conversations.id`` value; ordering
on apply guarantees the parent table exists before the child.

version history:

- v001 creates the ``context_items`` table with two lookup indexes.
- v002 (collections-task-05) promotes every naive TIMESTAMP column in
  the ``context_items`` table (date_accessed, date_created,
  date_updated) to TIMESTAMPTZ via ``ALTER COLUMN ... TYPE TIMESTAMPTZ
  USING ... AT TIME ZONE 'UTC'``. flips the package off the hybrid
  "naive-UTC at rest, aware everywhere else" convention so the
  database holds aware-UTC end to end. ships paired with the
  DATETIME_TYPE -> DATETIMETZ_TYPE Column-declaration flip in
  ``collections.py`` so the alignment enforcement test stays green.
- v003 (v0.8.0 shard 03) aligns the ``context_items`` shape with
  prod metallm: drops the v001 legacy indexes
  (``idx_ctx_conversation`` / ``idx_ctx_conversation_type``),
  creates the four v0.8.0 indexes (``ix_context_items_conv``,
  ``ix_context_items_type``, ``ix_context_items_lru``,
  ``ix_context_items_var_key`` partial-unique), backfills
  ``long_desc`` NULL -> '' and promotes it to NOT NULL DEFAULT '',
  and adds the FK ``conversation_id -> conversations(conversation_id)
  ON DELETE CASCADE`` so the parity gate stays clean.
"""

from __future__ import annotations

from threetears.agent.tools.migrations.v001_create_context_items_table import (
    create_context_items_table,
)
from threetears.agent.tools.migrations.v002_datetime_to_datetimetz import (
    datetime_to_datetimetz,
)
from threetears.agent.tools.migrations.v003_align_context_items_shape import (
    align_context_items_shape,
)
from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

PACKAGE_NAME = "agent_tools"


def register(runner: MigrationRunner) -> PackageMigrations:
    """
    register agent-tools migrations with the given runner.

    produces an agent-scoped :class:`PackageMigrations` declaring
    ``depends_on=("conversations",)``, attaches every migration in
    version order, and calls ``runner.register``.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.AGENT,
        depends_on=("conversations",),
    )
    pkg.version(1)(create_context_items_table)
    pkg.version(2)(datetime_to_datetimetz)
    pkg.version(3)(align_context_items_shape)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "align_context_items_shape",
    "create_context_items_table",
    "datetime_to_datetimetz",
    "register",
]

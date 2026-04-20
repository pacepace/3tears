"""
agent-memory package migrations.

single entry point :func:`register` wires the package's versioned
migration callables into a shared :class:`~threetears.core.data.
migrations.runner.MigrationRunner`. after the conversations / context
ownership reshape, agent-memory owns the following tables in every
agent schema:

- ``memories`` -- extracted long-term memories with pgvector embedding
  and FTS search vector.
- ``conversation_memory_refs`` -- append-only ledger tracking which
  items have already been surfaced in a conversation.
- ``media`` -- parent record for an uploaded document / image / audio
  artifact.
- ``media_content`` -- extracted text (OCR, caption, full-text) per
  media item with embedding and FTS.
- ``memory_chunks`` -- document-style sliceable chunks (heading,
  page number) for RAG-flavoured retrieval.

version history:

- v001 creates the memories table (pgvector + scoping indexes).
- v002 creates conversation_memory_refs.
- v003 reconciles memories column names with the package source
  (rename ``id`` -> ``memory_id``, ``memory_type`` -> ``type_memory``;
  drop unused columns).
- v004 adds lifecycle + conversation-link columns to memories
  (``conversation_id``, ``message_id_source``, ``is_deleted``,
  ``media_id``, ``date_deleted``, ``summary``).
- v005 adds ``search_vector`` + GIN index + maintenance trigger on
  memories for full-text search.
- v006 creates ``media`` + ``media_content`` tables with their own
  FTS triggers.
- v007 creates ``memory_chunks`` with its FTS trigger.

the package declares ``depends_on=("conversations",)`` because the
ledger references ``conversations(id)`` even though no FK constraint
exists today (the constraint would force a cross-package teardown
order; ordering on apply is enough).
"""

from __future__ import annotations

from threetears.agent.memory.migrations.v001_create_memories_table import (
    create_memories_table,
)
from threetears.agent.memory.migrations.v002_create_conversation_memory_refs import (
    create_conversation_memory_refs,
)
from threetears.agent.memory.migrations.v003_memory_column_reconciliation import (
    reconcile_memory_columns,
)
from threetears.agent.memory.migrations.v004_memory_lifecycle_columns import (
    add_lifecycle_columns,
)
from threetears.agent.memory.migrations.v005_memory_fts import (
    add_memory_fts,
)
from threetears.agent.memory.migrations.v006_memory_media_content import (
    create_media_tables,
)
from threetears.agent.memory.migrations.v007_memory_chunks import (
    create_memory_chunks,
)
from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

PACKAGE_NAME = "agent_memory"


def register(runner: MigrationRunner) -> PackageMigrations:
    """
    register agent-memory migrations with the given runner.

    produces an agent-scoped :class:`PackageMigrations` declaring
    ``depends_on=("conversations",)`` so the conversations table
    always exists before memory tables are created. attaches every
    migration in version order and calls ``runner.register``.

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
    pkg.version(1)(create_memories_table)
    pkg.version(2)(create_conversation_memory_refs)
    pkg.version(3)(reconcile_memory_columns)
    pkg.version(4)(add_lifecycle_columns)
    pkg.version(5)(add_memory_fts)
    pkg.version(6)(create_media_tables)
    pkg.version(7)(create_memory_chunks)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "add_lifecycle_columns",
    "add_memory_fts",
    "create_conversation_memory_refs",
    "create_media_tables",
    "create_memories_table",
    "create_memory_chunks",
    "reconcile_memory_columns",
    "register",
]

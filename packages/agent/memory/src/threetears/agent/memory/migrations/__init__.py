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
- v008 (collections-task-04) restores ``memories.agent_id`` /
  ``memories.customer_id`` to NOT NULL and rewrites the primary key
  to the composite ``(agent_id, memory_id)`` so the partition column
  is part of every uniqueness check; ``memory_id`` keeps its
  standalone UNIQUE constraint for child tables that reference it.
- v009 partitions ``media`` on ``agent_id`` (NOT NULL + composite PK
  on ``(agent_id, media_id)`` + UNIQUE on ``media_id``).
- v010 partitions ``media_content`` on ``agent_id`` and replaces the
  simple FK on ``media_id`` with the composite
  ``(agent_id, media_id) REFERENCES media(agent_id, media_id)``.
- v011 partitions ``memory_chunks`` on ``agent_id`` and replaces the
  simple FK on ``media_id`` with the composite
  ``(agent_id, media_id) REFERENCES media(agent_id, media_id)``.
- v012 (partition-hardening-task-01) adds the composite FK from
  ``memories.(agent_id, media_id)`` to ``media(agent_id, media_id)``
  with ``ON DELETE SET NULL`` semantics -- the v004 ``media_id``
  column landed without an FK constraint; this closes the gap. the
  AST walker enforces query-shape; the FK enforces data integrity.
- v013 (collections-task-05) promotes every naive TIMESTAMP column
  in the agent-memory partition (memories, media, media_content,
  memory_chunks, conversation_memory_refs) to TIMESTAMPTZ via
  ``ALTER COLUMN ... TYPE TIMESTAMPTZ USING ... AT TIME ZONE 'UTC'``.
  flips the package off the hybrid "naive-UTC at rest, aware
  everywhere else" convention so the database holds aware-UTC end
  to end. ships paired with the DATETIME_TYPE -> DATETIMETZ_TYPE
  Column-declaration flip in ``collections.py`` so the alignment
  enforcement test stays green.
- v024 (presence/aliveness, v0.15.0) adds the salience substrate
  (``salience``, ``last_decayed_at``, ``last_accessed``, ``evergreen``,
  ``superseded_by``) backing scheduled decay + reinforcement, and
  relaxes ``customer_id`` / ``user_id`` to nullable so the primitive
  supports agent / customer / user scope grains (metallm enforces
  NOT NULL at its own consumer layer). All additive.

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
from threetears.agent.memory.migrations.v008_restore_memories_agent_customer_not_null import (
    restore_memories_agent_customer_not_null,
)
from threetears.agent.memory.migrations.v009_media_composite_fk import (
    media_composite_fk,
)
from threetears.agent.memory.migrations.v010_media_content_composite_fk import (
    media_content_composite_fk,
)
from threetears.agent.memory.migrations.v011_memory_chunks_composite_fk import (
    memory_chunks_composite_fk,
)
from threetears.agent.memory.migrations.v012_memories_media_composite_fk import (
    memories_media_composite_fk,
)
from threetears.agent.memory.migrations.v013_datetime_to_datetimetz import (
    datetime_to_datetimetz,
)
from threetears.agent.memory.migrations.v014_memory_refs_date_columns import (
    rename_memory_refs_date_columns,
)
from threetears.agent.memory.migrations.v015_unified_memory_columns import (
    add_unified_memory_columns,
)
from threetears.agent.memory.migrations.v016_backfill_memory_ids import (
    backfill_memory_ids,
)
from threetears.agent.memory.migrations.v017_memory_fk_flip import (
    flip_memory_parent_fks,
)
from threetears.agent.memory.migrations.v018_drop_legacy_memory_columns import (
    drop_legacy_memory_columns,
)
from threetears.agent.memory.migrations.v019_conversation_id_not_null import (
    enforce_conversation_id_not_null,
)
from threetears.agent.memory.migrations.v020_memories_alias import (
    add_memories_alias,
)
from threetears.agent.memory.migrations.v021_memory_chunks_index_and_token import (
    add_chunk_index_and_token_count,
)
from threetears.agent.memory.migrations.v022_add_hnsw_gin_indexes import (
    add_hnsw_gin_indexes,
)
from threetears.agent.memory.migrations.v023_fix_idx_chunks_message_id_start import (
    fix_idx_chunks_message_id_start,
)
from threetears.agent.memory.migrations.v024_memory_salience_and_scope import (
    add_memory_salience_and_relax_scope,
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
    pkg.version(8)(restore_memories_agent_customer_not_null)
    pkg.version(9)(media_composite_fk)
    pkg.version(10)(media_content_composite_fk)
    pkg.version(11)(memory_chunks_composite_fk)
    pkg.version(12)(memories_media_composite_fk)
    pkg.version(13)(datetime_to_datetimetz)
    pkg.version(14)(rename_memory_refs_date_columns)
    pkg.version(15)(add_unified_memory_columns)
    pkg.version(16)(backfill_memory_ids)
    pkg.version(17)(flip_memory_parent_fks)
    pkg.version(18)(drop_legacy_memory_columns)
    pkg.version(19)(enforce_conversation_id_not_null)
    pkg.version(20)(add_memories_alias)
    pkg.version(21)(add_chunk_index_and_token_count)
    pkg.version(22)(add_hnsw_gin_indexes)
    pkg.version(23)(fix_idx_chunks_message_id_start)
    pkg.version(24)(add_memory_salience_and_relax_scope)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "add_chunk_index_and_token_count",
    "add_hnsw_gin_indexes",
    "add_memory_salience_and_relax_scope",
    "fix_idx_chunks_message_id_start",
    "add_lifecycle_columns",
    "add_memories_alias",
    "add_memory_fts",
    "add_unified_memory_columns",
    "backfill_memory_ids",
    "create_conversation_memory_refs",
    "create_media_tables",
    "create_memories_table",
    "create_memory_chunks",
    "datetime_to_datetimetz",
    "drop_legacy_memory_columns",
    "enforce_conversation_id_not_null",
    "flip_memory_parent_fks",
    "media_composite_fk",
    "media_content_composite_fk",
    "memories_media_composite_fk",
    "memory_chunks_composite_fk",
    "reconcile_memory_columns",
    "register",
    "rename_memory_refs_date_columns",
    "restore_memories_agent_customer_not_null",
]

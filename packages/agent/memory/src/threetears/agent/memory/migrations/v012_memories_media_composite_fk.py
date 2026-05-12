"""
agent-memory v012: memories composite FK to media on (agent_id, media_id).

partition-hardening-task-01 sub-task 3 + review-task-01 finding D-5.
the v004 ``memories`` table grew a nullable ``media_id UUID`` column
without any FK constraint (``ALTER TABLE memories ADD COLUMN IF NOT
EXISTS media_id UUID NULL``). v009 partitioned ``media`` on
``agent_id`` and pinned its primary key to ``(agent_id, media_id)``,
so a composite FK from ``memories`` is now structurally available.
adding it closes the remaining DB-layer integrity gap.

the AST walker in
``packages/core/tests/enforcement/test_partition_column_enforcement.py``
already enforces the query-shape rule (every read of ``memories``
must filter on ``agent_id``); the FK is the data-integrity
companion -- a row in ``memories`` with ``media_id`` set must
reference an actual ``(agent_id, media_id)`` pair in ``media``.

semantic relationship: a memory references a media artifact (the
memory was extracted from / is anchored to that media). deleting
the media row should NOT cascade-delete the memory (the extracted
fact still exists even after the source artifact is removed); it
should null the reference. ``ON DELETE SET NULL`` matches the data
semantics. ``ON DELETE CASCADE`` would lose information.

correctness gotcha: a plain ``ON DELETE SET NULL`` on a composite FK
nulls *every* FK column when the parent is deleted. on this table
that would null the partition column ``agent_id`` -- which is
NOT NULL per v008 -- so the cascade would fail at runtime with a
NotNullViolationError on the first parent delete. the PG 15+
column-list form ``ON DELETE SET NULL (media_id)`` restricts the
SET NULL action to the specific FK column the relationship is
about, leaving ``agent_id`` populated. test
:meth:`TestV012MemoriesMediaCompositeFK.test_media_delete_nulls_referencing_memory`
exercises the cascade and pins ``agent_id`` against unintended
nullification.

idempotency: every step is guarded by ``pg_constraint`` checks
inside DO blocks so replay on recovery is a no-op. pre-GA carve-out:
verify orphan ``memories.media_id`` count before adding the FK; on
a fresh test DB this is always zero, but on any pre-applied schema
the probe surfaces drift loudly rather than silently truncating.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "memories_media_composite_fk",
]

log = get_logger(__name__)


# null out any orphan ``memories.media_id`` references before the FK
# add. pre-GA there should be no rows to clean; the UPDATE is a
# defensive guard so the FK ADD does not surface as a constraint
# violation against drift data nobody can see. counting orphans first
# turns "silent truncate" into "loud probe + log".
_NULL_ORPHAN_MEDIA_REFS_SQL = """
DO $$
DECLARE
    orphan_count int;
BEGIN
    SELECT COUNT(*) INTO orphan_count
      FROM memories m
     WHERE m.media_id IS NOT NULL
       AND NOT EXISTS (
           SELECT 1 FROM media md
            WHERE md.agent_id = m.agent_id
              AND md.media_id = m.media_id
       );
    IF orphan_count > 0 THEN
        RAISE NOTICE 'v012 nulling % orphan memories.media_id refs', orphan_count;
        UPDATE memories m
           SET media_id = NULL
         WHERE m.media_id IS NOT NULL
           AND NOT EXISTS (
               SELECT 1 FROM media md
                WHERE md.agent_id = m.agent_id
                  AND md.media_id = m.media_id
           );
    END IF;
END
$$
"""


_ADD_COMPOSITE_FK_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = (
             SELECT oid FROM pg_class
              WHERE relname = 'memories'
                AND relnamespace = (
                    SELECT oid FROM pg_namespace
                     WHERE nspname = current_schema()
                )
         )
           AND conname = 'memories_media_composite_fk'
    ) THEN
        -- PG15+: ``ON DELETE SET NULL (column_list)`` restricts the
        -- SET NULL action to a specific FK column, leaving the
        -- partition column ``agent_id`` (which is NOT NULL on
        -- memories per v008) intact. plain ``SET NULL`` on a composite
        -- FK would null *every* FK column, hitting the agent_id
        -- NOT NULL constraint and failing the cascade.
        ALTER TABLE memories
            ADD CONSTRAINT memories_media_composite_fk
            FOREIGN KEY (agent_id, media_id)
            REFERENCES media (agent_id, media_id)
            ON DELETE SET NULL (media_id);
    END IF;
END
$$
"""


async def memories_media_composite_fk(store: DataStore) -> None:
    """add composite FK from memories.(agent_id, media_id) to media + null orphans.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding composite FK memories -> media on (agent_id, media_id) (v012)")
    await store.execute(_NULL_ORPHAN_MEDIA_REFS_SQL)
    await store.execute(_ADD_COMPOSITE_FK_SQL)

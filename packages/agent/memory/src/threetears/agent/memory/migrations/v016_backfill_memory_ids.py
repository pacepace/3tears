"""
agent-memory v016: backfill memory_id on media + memory_chunks.

transcript-chunks-task-A. v015 added the new ``memory_id`` columns
nullable; this migration populates them so v017 can lock them in
with NOT NULL + composite FKs.

Backfill strategy (idempotent — every step skips rows already
processed, every step's COUNT + decision live inside one DO block
so the migration runner only needs ``execute``):

Step 1 — ``media.memory_id`` from the existing reverse pairing.
Today the FK runs ``memories.media_id -> media.media_id`` (v012's
``memories_media_composite_fk``). For every media row whose
``memory_id`` is still NULL, look up a memory whose ``media_id``
matches and copy its ``memory_id`` over. This preserves the
existing media↔memory pair without creating a new row.

Step 2 — ``media.memory_id`` for orphan media with a ref. For
media rows still NULL after Step 1, look up a
``conversation_memory_refs`` entry of ``item_type='media'`` with
``item_id = media.media_id``; if found, create a new paired
memory row (``type_memory='topical_context'``, ``content``
synthesized from media metadata) under the ref's
``conversation_id``. Set ``media.memory_id`` to the new memory's
id. The synthetic memory carries the same
``(agent_id, customer_id, user_id)`` triple as its media child.

Step 3 — truly-orphan media (no existing pair, no ref) is
deleted. Its ``media_content`` and any chunks parented to it
cascade-die via the existing FK on ``media.media_id``. RAISE
NOTICE emits the count for operator audit.

Step 4 — ``memory_chunks.memory_id`` from the parent media
chain (chunks.media_id -> media.memory_id). Then drop any
chunks still NULL after the backfill (legacy paths that wrote
chunks without a parent media row).

After this migration:
- Every ``media`` row has ``memory_id`` set (orphans deleted).
- Every ``memory_chunks`` row has ``memory_id`` set (orphans
  deleted).
- v019 will later strip any synthetic memories whose ledger
  ref had no ``conversation_id`` — defensive cleanup.

Per-row UUIDs for synthetic memories use ``gen_random_uuid()``
(Postgres 13+ core, no extension required). The UUIDv7
enforcement test pins entity-creation-time IDs only; migration-
time inserts like this are an explicit carve-out.

Idempotent via WHERE-IS-NULL guards on every step.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "backfill_memory_ids",
]

log = get_logger(__name__)


# step 0 (probe): under the unified model media → memory is a 1:1
# relationship (one media row, one parent memory). The legacy reverse
# FK was many-memories → one-media — drift could produce a media row
# that multiple memories point at. Step 1's UPDATE would then pick
# one of them non-deterministically. RAISE NOTICE so operators see
# the divergence in the migration log; the resolution still picks
# the lowest-memory_id row (deterministic via the implicit asyncpg
# tuple ordering), but operators get to audit the choice rather than
# discover it post-fact.
_PROBE_MULTI_MEMORY_PER_MEDIA_SQL = """
DO $$
DECLARE
    divergent int;
BEGIN
    SELECT COUNT(*) INTO divergent
      FROM (
          SELECT agent_id, media_id
            FROM memories
           WHERE media_id IS NOT NULL
           GROUP BY agent_id, media_id
          HAVING COUNT(*) > 1
      ) AS dup;
    IF divergent > 0 THEN
        RAISE NOTICE 'v016 found % media rows with multiple referencing memories; step 1 will pick one deterministically — audit if unexpected', divergent;
    END IF;
END
$$
"""


# step 1: copy memory_id from existing memories.media_id reverse FK.
# WHERE media.memory_id IS NULL guards re-application. composite scope
# on (agent_id, media_id) keeps the lookup inside a single partition.
# When step 0 detected multi-memory→one-media drift, the DISTINCT ON
# clause picks the lowest memory_id deterministically so re-runs
# produce identical output.
_BACKFILL_MEDIA_MEMORY_ID_FROM_REVERSE_PAIR_SQL = """
UPDATE media md
   SET memory_id = picked.memory_id
  FROM (
      SELECT DISTINCT ON (agent_id, media_id)
             agent_id,
             media_id,
             memory_id
        FROM memories
       WHERE media_id IS NOT NULL
       ORDER BY agent_id, media_id, memory_id
  ) picked
 WHERE md.memory_id IS NULL
   AND picked.media_id = md.media_id
   AND picked.agent_id = md.agent_id
"""


# step 2: create a synthetic memory row for every media that still
# has memory_id NULL but has a conversation_memory_refs entry. Then
# point media.memory_id at the new memory. Wrapped in a DO block so
# the row count can be RAISE NOTICE'd in the same statement (the
# migration runner only exposes execute(), so any count we want to
# log has to happen inside SQL).
#
# When one media has multiple refs (same upload from two
# conversations) we arbitrarily pick the earliest ``date_created``
# ref via DISTINCT ON. ``message_id_source`` gets a fresh UUID since
# no actual source message exists — the memory describes a pre-
# existing media artifact.
_CREATE_SYNTHETIC_MEMORIES_AND_LINK_SQL = """
DO $$
DECLARE
    synthesized_count int;
BEGIN
    WITH refs AS (
        SELECT DISTINCT ON (r.item_id)
               r.item_id AS media_id,
               r.conversation_id
          FROM conversation_memory_refs r
          JOIN media md
            ON md.media_id = r.item_id
           AND md.memory_id IS NULL
         WHERE r.item_type = 'media'
         ORDER BY r.item_id, r.date_created ASC
    ),
    new_memories AS (
        INSERT INTO memories (
            memory_id,
            agent_id,
            customer_id,
            user_id,
            conversation_id,
            message_id_source,
            type_memory,
            content,
            is_deleted,
            media_id,
            date_created,
            date_updated
        )
        SELECT gen_random_uuid() AS memory_id,
               md.agent_id,
               md.customer_id,
               md.user_id,
               refs.conversation_id,
               gen_random_uuid() AS message_id_source,
               'topical_context' AS type_memory,
               COALESCE(
                   'Media artifact (' || md.media_category || ')',
                   'Media artifact'
               ) AS content,
               false AS is_deleted,
               md.media_id,
               now() AS date_created,
               now() AS date_updated
          FROM refs
          JOIN media md
            ON md.media_id = refs.media_id
           AND md.memory_id IS NULL
        RETURNING memory_id, media_id, agent_id
    ),
    linked AS (
        UPDATE media md
           SET memory_id = nm.memory_id
          FROM new_memories nm
         WHERE md.media_id = nm.media_id
           AND md.agent_id = nm.agent_id
           AND md.memory_id IS NULL
        RETURNING md.media_id
    )
    SELECT COUNT(*) INTO synthesized_count FROM linked;
    IF synthesized_count > 0 THEN
        RAISE NOTICE 'v016 synthesized % memories for ref-linked orphan media', synthesized_count;
    END IF;
END
$$
"""


# step 3: delete truly-orphan media (no memory pair, no ledger ref).
# Their media_content + chunks cascade-die via the existing
# (agent_id, media_id) FK from v011 / v010. Counted + RAISE NOTICE'd
# inside the DO block.
_DELETE_TRULY_ORPHAN_MEDIA_SQL = """
DO $$
DECLARE
    deleted_count int;
BEGIN
    WITH deleted AS (
        DELETE FROM media
         WHERE memory_id IS NULL
        RETURNING media_id
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;
    IF deleted_count > 0 THEN
        RAISE NOTICE 'v016 deleted % truly-orphan media rows (no memory, no ledger ref)', deleted_count;
    END IF;
END
$$
"""


# step 4 (backfill): copy memory_id from the parent media row for
# every chunk that still has memory_id NULL but has a media_id. Then
# drop any chunks still NULL — they had no discoverable parent.
_BACKFILL_CHUNK_MEMORY_ID_FROM_MEDIA_SQL = """
UPDATE memory_chunks mc
   SET memory_id = md.memory_id
  FROM media md
 WHERE mc.memory_id IS NULL
   AND mc.media_id = md.media_id
   AND mc.agent_id = md.agent_id
"""

_DELETE_TRULY_ORPHAN_CHUNKS_SQL = """
DO $$
DECLARE
    deleted_count int;
BEGIN
    WITH deleted AS (
        DELETE FROM memory_chunks
         WHERE memory_id IS NULL
        RETURNING chunk_id
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;
    IF deleted_count > 0 THEN
        RAISE NOTICE 'v016 deleted % truly-orphan chunks (no memory, no media)', deleted_count;
    END IF;
END
$$
"""


async def backfill_memory_ids(store: DataStore) -> None:
    """populate memory_id on media + memory_chunks; drop true orphans.

    See module docstring for the four-step strategy. Every step is
    idempotent (WHERE NULL guards) so replay on a partially-applied
    schema is safe. Counts are emitted via RAISE NOTICE inside DO
    blocks rather than fetched back to Python — the migration runner
    only exposes ``execute()``.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("backfilling memory_id on media + chunks (v016)")
    await store.execute(_PROBE_MULTI_MEMORY_PER_MEDIA_SQL)
    await store.execute(_BACKFILL_MEDIA_MEMORY_ID_FROM_REVERSE_PAIR_SQL)
    await store.execute(_CREATE_SYNTHETIC_MEMORIES_AND_LINK_SQL)
    await store.execute(_DELETE_TRULY_ORPHAN_MEDIA_SQL)
    await store.execute(_BACKFILL_CHUNK_MEMORY_ID_FROM_MEDIA_SQL)
    await store.execute(_DELETE_TRULY_ORPHAN_CHUNKS_SQL)

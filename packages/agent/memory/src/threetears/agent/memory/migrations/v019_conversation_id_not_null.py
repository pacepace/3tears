"""
agent-memory v019: enforce ``memories.conversation_id`` NOT NULL.

transcript-chunks-task-A. Every memory the agent creates is rooted
in a conversation — ``memory_add`` requires conversation_id at the
tool surface; auto-extracted memories carry the conversation that
generated them. v016 created synthetic memories for ledger-ref'd
media using the ref's conversation_id. The only remaining NULLs
come from pre-task-A data paths that didn't enforce this — those
rows are unreachable under the unified model (no conversation to
walk back to) and are deleted before the column is locked down.

Step 1 — count + log + delete memories with NULL conversation_id.
CASCADE on the v017 FKs propagates to chunks + media attachments
(media_content via the v006 FK).

Step 2 — ``ALTER COLUMN conversation_id SET NOT NULL`` guarded by
an information_schema check so replay is a no-op.

Idempotent.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "enforce_conversation_id_not_null",
]

log = get_logger(__name__)


_DELETE_NULL_CONV_ID_SQL = """
DO $$
DECLARE
    deleted_count int;
BEGIN
    WITH deleted AS (
        DELETE FROM memories
         WHERE conversation_id IS NULL
        RETURNING memory_id
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;
    IF deleted_count > 0 THEN
        RAISE NOTICE 'v019 deleted % memories with NULL conversation_id (cascades to chunks + media)', deleted_count;
    END IF;
END
$$
"""


_SET_CONV_ID_NOT_NULL_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'memories'
           AND column_name = 'conversation_id'
           AND is_nullable = 'YES'
    ) THEN
        ALTER TABLE memories
            ALTER COLUMN conversation_id SET NOT NULL;
    END IF;
END
$$
"""


async def enforce_conversation_id_not_null(store: DataStore) -> None:
    """delete null-conversation memories then lock the column NOT NULL.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("enforcing memories.conversation_id NOT NULL (v019)")
    await store.execute(_DELETE_NULL_CONV_ID_SQL)
    await store.execute(_SET_CONV_ID_NOT_NULL_SQL)

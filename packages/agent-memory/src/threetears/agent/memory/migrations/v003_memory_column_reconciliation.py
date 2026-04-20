"""
agent-memory v003: reconcile memories column names with code expectations.

memory-task-01. v001 shipped a schema whose column names diverged from
the names the package's own source (collections.py, retrieval.py,
extraction.py, tools.py, entities.py) has evolved to reference. v003
renames the PK (``id`` -> ``memory_id``) and the discriminator column
(``memory_type`` -> ``type_memory``), drops columns the source tree
does not reference anywhere (``embedding_model``, ``importance``,
``metadata``, ``date_accessed``), and loosens ``agent_id`` /
``customer_id`` to nullable (the code paths treat them as optional
scoping tags that can be omitted when the caller doesn't know them).

Idempotency: each alteration is guarded by an ``information_schema``
existence check inside a DO block so replay on recovery is a no-op.
``ALTER TABLE ... RENAME COLUMN IF EXISTS`` is not valid Postgres
syntax on the versions the project supports -- hence DO blocks. Column
drops use ``DROP COLUMN IF EXISTS`` which IS valid.

Data safety: renames preserve values; drops remove data that was never
read by the package. No back-compat aliases or translation views are
left behind per CLAUDE.md (NO BACKWARDS-COMPATIBILITY SHIMS).
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "reconcile_memory_columns",
]

log = get_logger(__name__)


_RENAME_ID_TO_MEMORY_ID_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'memories'
           AND column_name = 'id'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'memories'
           AND column_name = 'memory_id'
    ) THEN
        ALTER TABLE memories RENAME COLUMN id TO memory_id;
    END IF;
END
$$
"""

_RENAME_MEMORY_TYPE_TO_TYPE_MEMORY_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'memories'
           AND column_name = 'memory_type'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'memories'
           AND column_name = 'type_memory'
    ) THEN
        ALTER TABLE memories RENAME COLUMN memory_type TO type_memory;
    END IF;
END
$$
"""

_DROP_EMBEDDING_MODEL_SQL = (
    "ALTER TABLE memories DROP COLUMN IF EXISTS embedding_model"
)

_DROP_IMPORTANCE_SQL = "ALTER TABLE memories DROP COLUMN IF EXISTS importance"

_DROP_METADATA_SQL = "ALTER TABLE memories DROP COLUMN IF EXISTS metadata"

_DROP_DATE_ACCESSED_SQL = (
    "ALTER TABLE memories DROP COLUMN IF EXISTS date_accessed"
)

# agent_id / customer_id were declared NOT NULL in v001 but the code
# (``tools.load_add_memory_tool``, ``extraction.MemoryExtractor.extract``)
# treats them as optional scoping tags. Loosen the NOT NULL so the code
# paths that legitimately omit them do not fail. The code still scopes
# against them when present.
_ALTER_AGENT_ID_NULLABLE_SQL = (
    "ALTER TABLE memories ALTER COLUMN agent_id DROP NOT NULL"
)

_ALTER_CUSTOMER_ID_NULLABLE_SQL = (
    "ALTER TABLE memories ALTER COLUMN customer_id DROP NOT NULL"
)


async def reconcile_memory_columns(store: DataStore) -> None:
    """
    rename memories PK + type column and drop unused columns.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    """
    log.info("reconciling memories columns (v003)")
    await store.execute(_RENAME_ID_TO_MEMORY_ID_SQL)
    await store.execute(_RENAME_MEMORY_TYPE_TO_TYPE_MEMORY_SQL)
    await store.execute(_DROP_EMBEDDING_MODEL_SQL)
    await store.execute(_DROP_IMPORTANCE_SQL)
    await store.execute(_DROP_METADATA_SQL)
    await store.execute(_DROP_DATE_ACCESSED_SQL)
    await store.execute(_ALTER_AGENT_ID_NULLABLE_SQL)
    await store.execute(_ALTER_CUSTOMER_ID_NULLABLE_SQL)

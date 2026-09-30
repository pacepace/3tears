"""
agent-memory v028: every agent schema's indexes and unique constraints match the collections' declarations.

The chain built several objects twice under two names -- v001-v007 under
``idx_*`` / ``<table>_<id>_unique`` names, then v022 again under the names the
``TableSchema`` declarations carry -- and never built some objects the
declarations name. A schema built by the runner therefore differed from the
tables the collections declare, and from a consumer whose tables follow those
declarations. This migration closes the gap in both directions.

Exact duplicates of a declared index are renamed to the declared name when the
declared one is missing, and dropped otherwise, so no schema rebuilds an index
it already has:

- ``idx_mem_user`` -> ``idx_memories_agent_customer_user``
- ``idx_mc_user`` -> ``ix_media_content_user``
- ``idx_mc_embedding`` -> ``ix_media_content_embedding``
- ``idx_chunks_user`` -> ``ix_memory_chunks_user``
- ``idx_chunks_embedding`` -> ``ix_memory_chunks_embedding``

Declared indexes no migration built are created (``IF NOT EXISTS``):
``ix_memories_user_date``, ``ix_media_user_date``, ``ix_media_mime_type``,
``ix_media_memory_id``, the unique ``uq_media_cloud_connection_file``,
``ix_media_content_media_type`` and ``ix_memory_chunks_memory``. The four
migration-built indexes the declarations now also name (``idx_mem_conversation``,
``idx_chunks_memory_id_chunk_id``, ``idx_chunks_message_id_end``,
``idx_conv_mem_refs_conversation_date_created``) are created too, for a schema
that was built from the declarations rather than the chain.

Indexes another index already serves -- same leading columns, or a partial
index over a leading column a full index covers -- are dropped once that index
exists: ``idx_mem_agent``, ``idx_mem_customer``, ``idx_mem_embedding`` (the
unparameterised twin of ``ix_memories_embedding_hnsw``), ``idx_media_agent``,
``idx_media_user``, ``idx_media_memory_id``, ``idx_mc_media`` and
``idx_chunks_memory_id``. ``ix_conversation_memory_refs_cid`` goes as well: the
primary key and ``idx_conv_mem_refs_conversation_date_created`` both lead with
``conversation_id``, so a third index on it only costs writes.

The legacy unique constraints ``memories_memory_id_unique``,
``media_media_id_unique``, ``media_content_content_id_unique`` and
``memory_chunks_chunk_id_unique`` duplicate the declared ``uq_*`` constraints.
When the declared one is missing (a second agent schema in one database, where
v022's unscoped guard saw the first schema's constraint and skipped), the legacy
one is renamed to it. When both exist the legacy one is dropped, and a foreign
key bound to its index is dropped first and re-added with its own definition,
so it binds to the declared constraint instead.

The key from ``media`` to ``memories`` is the composite ``media_memory_fk`` that
v017 built. A single-column foreign key on ``media.memory_id`` alone, whatever
its name, is dropped.

Every catalog lookup is scoped to ``current_schema()``, and every drop names the
schema, so a database holding several agent schemas migrates each on its own.
Replay is a no-op: each step checks for the state it produces first.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "align_indexes_with_declarations",
]

log = get_logger(__name__)


# ----- exact duplicates: rename to the declared name, or drop ---------- #

#: ``(legacy, declared)`` index pairs over identical columns and method.
_EXACT_DUPLICATE_INDEXES: tuple[tuple[str, str], ...] = (
    ("idx_mem_user", "idx_memories_agent_customer_user"),
    ("idx_mc_user", "ix_media_content_user"),
    ("idx_mc_embedding", "ix_media_content_embedding"),
    ("idx_chunks_user", "ix_memory_chunks_user"),
    ("idx_chunks_embedding", "ix_memory_chunks_embedding"),
)

_RENAME_OR_DROP_INDEX_TEMPLATE = """
DO $$
DECLARE
    ns oid := (SELECT oid FROM pg_catalog.pg_namespace WHERE nspname = current_schema());
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_class
         WHERE relname = '{legacy}' AND relnamespace = ns AND relkind = 'i'
    ) THEN
        IF EXISTS (
            SELECT 1 FROM pg_catalog.pg_class
             WHERE relname = '{declared}' AND relnamespace = ns AND relkind = 'i'
        ) THEN
            EXECUTE format('DROP INDEX IF EXISTS %I.%I', current_schema(), '{legacy}');
        ELSE
            EXECUTE format('ALTER INDEX %I.%I RENAME TO %I', current_schema(), '{legacy}', '{declared}');
        END IF;
    END IF;
END
$$
"""


# ----- declared indexes: create when absent ----------------------------- #

_CREATE_DECLARED_INDEXES_SQL: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS ix_memories_user_date ON memories (user_id, date_created)",
    "CREATE INDEX IF NOT EXISTS idx_mem_conversation ON memories (conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_memories_agent_customer_user ON memories (agent_id, customer_id, user_id)",
    "CREATE INDEX IF NOT EXISTS ix_media_user_date ON media (user_id, date_created)",
    "CREATE INDEX IF NOT EXISTS ix_media_mime_type ON media (mime_type)",
    "CREATE INDEX IF NOT EXISTS ix_media_memory_id ON media (memory_id)",
    ("CREATE UNIQUE INDEX IF NOT EXISTS uq_media_cloud_connection_file ON media (cloud_connection_id, cloud_file_id)"),
    "CREATE INDEX IF NOT EXISTS ix_media_content_media_type ON media_content (media_id, content_type)",
    "CREATE INDEX IF NOT EXISTS ix_media_content_user ON media_content (user_id)",
    (
        "CREATE INDEX IF NOT EXISTS ix_media_content_embedding "
        "ON media_content USING hnsw (embedding public.vector_cosine_ops)"
    ),
    "CREATE INDEX IF NOT EXISTS ix_memory_chunks_memory ON memory_chunks (memory_id, chunk_index)",
    "CREATE INDEX IF NOT EXISTS ix_memory_chunks_user ON memory_chunks (user_id)",
    (
        "CREATE INDEX IF NOT EXISTS ix_memory_chunks_embedding "
        "ON memory_chunks USING hnsw (embedding public.vector_cosine_ops)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_chunks_memory_id_chunk_id "
        "ON memory_chunks (memory_id, chunk_id) WHERE memory_id IS NOT NULL"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_chunks_message_id_end "
        "ON memory_chunks (message_id_end) WHERE message_id_end IS NOT NULL"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_conv_mem_refs_conversation_date_created "
        "ON conversation_memory_refs (conversation_id, date_created)"
    ),
)


# ----- indexes another index already serves: drop ------------------------ #

#: each is served by a declared index leading with the same column(s),
#: created above when missing, or (``ix_conversation_memory_refs_cid``) by the
#: primary key.
_REDUNDANT_INDEXES: tuple[str, ...] = (
    "idx_mem_agent",
    "idx_mem_customer",
    "idx_mem_embedding",
    "idx_media_agent",
    "idx_media_user",
    "idx_media_memory_id",
    "idx_mc_media",
    "idx_chunks_memory_id",
    "ix_conversation_memory_refs_cid",
)

_DROP_INDEX_TEMPLATE = """
DO $$
BEGIN
    EXECUTE format('DROP INDEX IF EXISTS %I.%I', current_schema(), '{name}');
END
$$
"""


# ----- unique constraints: one per id column, under the declared name ---- #

#: ``(table, column, legacy constraint, declared constraint)``.
_UNIQUE_CONSTRAINTS: tuple[tuple[str, str, str, str], ...] = (
    ("memories", "memory_id", "memories_memory_id_unique", "uq_memories_memory_id"),
    ("media", "media_id", "media_media_id_unique", "uq_media_media_id"),
    ("media_content", "content_id", "media_content_content_id_unique", "uq_media_content_content_id"),
    ("memory_chunks", "chunk_id", "memory_chunks_chunk_id_unique", "uq_memory_chunks_chunk_id"),
)

# a foreign key is bound to one unique index. dropping the legacy constraint
# under a key bound to it fails, so each such key is dropped, the legacy
# constraint goes, and the key is added back from its own definition -- which
# then binds to the declared constraint over the same column.
_ALIGN_UNIQUE_CONSTRAINT_TEMPLATE = """
DO $$
DECLARE
    ns oid := (SELECT oid FROM pg_catalog.pg_namespace WHERE nspname = current_schema());
    tbl oid;
    legacy_index oid;
    fk_tables text[];
    fk_names text[];
    fk_defs text[];
BEGIN
    SELECT oid INTO tbl FROM pg_catalog.pg_class
     WHERE relname = '{table}' AND relnamespace = ns AND relkind IN ('r', 'p');
    IF tbl IS NULL THEN
        RETURN;
    END IF;
    SELECT conindid INTO legacy_index FROM pg_catalog.pg_constraint
     WHERE conrelid = tbl AND conname = '{legacy}' AND contype = 'u';
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint WHERE conrelid = tbl AND conname = '{declared}'
    ) THEN
        IF legacy_index IS NOT NULL THEN
            EXECUTE format(
                'ALTER TABLE %I.%I RENAME CONSTRAINT %I TO %I',
                current_schema(), '{table}', '{legacy}', '{declared}'
            );
        ELSE
            EXECUTE format(
                'ALTER TABLE %I.%I ADD CONSTRAINT %I UNIQUE (%I)',
                current_schema(), '{table}', '{declared}', '{column}'
            );
        END IF;
        RETURN;
    END IF;
    IF legacy_index IS NULL THEN
        RETURN;
    END IF;
    SELECT array_agg(conrelid::regclass::text), array_agg(conname::text), array_agg(pg_get_constraintdef(oid))
      INTO fk_tables, fk_names, fk_defs
      FROM pg_catalog.pg_constraint
     WHERE contype = 'f' AND conindid = legacy_index;
    FOR i IN 1 .. coalesce(array_length(fk_names, 1), 0) LOOP
        EXECUTE format('ALTER TABLE %s DROP CONSTRAINT IF EXISTS %I', fk_tables[i], fk_names[i]);
    END LOOP;
    EXECUTE format(
        'ALTER TABLE %I.%I DROP CONSTRAINT IF EXISTS %I', current_schema(), '{table}', '{legacy}'
    );
    FOR i IN 1 .. coalesce(array_length(fk_names, 1), 0) LOOP
        EXECUTE format('ALTER TABLE %s ADD CONSTRAINT %I %s', fk_tables[i], fk_names[i], fk_defs[i]);
    END LOOP;
END
$$
"""


# ----- media -> memories: the composite key only ------------------------- #

_DROP_SINGLE_COLUMN_MEDIA_MEMORY_FK_SQL = """
DO $$
DECLARE
    ns oid := (SELECT oid FROM pg_catalog.pg_namespace WHERE nspname = current_schema());
    tbl oid;
    fk record;
BEGIN
    SELECT oid INTO tbl FROM pg_catalog.pg_class
     WHERE relname = 'media' AND relnamespace = ns AND relkind IN ('r', 'p');
    IF tbl IS NULL THEN
        RETURN;
    END IF;
    FOR fk IN
        SELECT con.conname
          FROM pg_catalog.pg_constraint con
          JOIN pg_catalog.pg_attribute att
            ON att.attrelid = con.conrelid AND att.attnum = con.conkey[1]
         WHERE con.conrelid = tbl
           AND con.contype = 'f'
           AND array_length(con.conkey, 1) = 1
           AND att.attname = 'memory_id'
    LOOP
        EXECUTE format('ALTER TABLE %I.media DROP CONSTRAINT IF EXISTS %I', current_schema(), fk.conname);
    END LOOP;
END
$$
"""


def _rename_or_drop_index_sql(legacy: str, declared: str) -> str:
    """render the rename-or-drop block for one exact-duplicate index pair.

    :param legacy: the migration-built duplicate's name
    :ptype legacy: str
    :param declared: the name the declaration carries
    :ptype declared: str
    :return: one ``DO`` block
    :rtype: str
    """
    return _RENAME_OR_DROP_INDEX_TEMPLATE.format(legacy=legacy, declared=declared)


def _drop_index_sql(name: str) -> str:
    """render the schema-qualified drop of one redundant index.

    :param name: index name
    :ptype name: str
    :return: one ``DO`` block
    :rtype: str
    """
    return _DROP_INDEX_TEMPLATE.format(name=name)


def _align_unique_constraint_sql(table: str, column: str, legacy: str, declared: str) -> str:
    """render the block that leaves exactly the declared unique constraint on ``column``.

    :param table: table carrying the constraint
    :ptype table: str
    :param column: the id column the constraint covers
    :ptype column: str
    :param legacy: the migration-built constraint's name
    :ptype legacy: str
    :param declared: the name the declaration carries
    :ptype declared: str
    :return: one ``DO`` block
    :rtype: str
    """
    return _ALIGN_UNIQUE_CONSTRAINT_TEMPLATE.format(table=table, column=column, legacy=legacy, declared=declared)


async def align_indexes_with_declarations(store: DataStore) -> None:
    """bring the agent schema's memory indexes, unique constraints and media key to the declarations.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("aligning memory indexes and unique constraints with the declarations (v028)")
    for legacy, declared in _EXACT_DUPLICATE_INDEXES:
        await store.execute(_rename_or_drop_index_sql(legacy, declared))
    for statement in _CREATE_DECLARED_INDEXES_SQL:
        await store.execute(statement)
    for name in _REDUNDANT_INDEXES:
        await store.execute(_drop_index_sql(name))
    for table, column, legacy, declared in _UNIQUE_CONSTRAINTS:
        await store.execute(_align_unique_constraint_sql(table, column, legacy, declared))
    await store.execute(_DROP_SINGLE_COLUMN_MEDIA_MEMORY_FK_SQL)

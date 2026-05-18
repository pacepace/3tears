"""
agent-memory v021: align 3tears memory + media schemas with prod parity.

shard 03 (v0.8.0) brings every memory-package ``TableSchema`` to full
prod parity. Several columns the v0.7.5 factories declare exist on
metallm prod (via metallm Alembic) but the 3tears migrations never
added them. This migration closes the gap so 3tears integration
tests (which use the 3tears migration runner to set up the test DB)
carry the same shape the v0.8.0 schemas now declare.

Columns added:

- ``memory_chunks.chunk_index`` (INTEGER NOT NULL DEFAULT 0) -- chunk
  order within parent memory; required by every chunk hybrid-search
  ``ORDER BY`` clause.
- ``memory_chunks.token_count`` (INTEGER NOT NULL DEFAULT 0) -- per-
  chunk token tally; required by ``find_by_memory_id`` projections.
- ``media.mime_type`` (TEXT NOT NULL DEFAULT '') -- file MIME type.
- ``media.size_bytes`` (INTEGER NOT NULL DEFAULT 0) -- byte count.
- ``media.source`` (TEXT NOT NULL DEFAULT '') -- ``upload`` /
  ``generation`` / ``cloud_sync`` discriminator.
- ``media.s3_key`` (TEXT NULL) -- when present, the artifact lives
  in S3/MinIO.
- ``media.generation_prompt`` (TEXT NULL) -- when ``source =
  'generation'``, the prompt used.
- ``media.thumbnail_s3_key`` (TEXT NULL) -- thumbnail location.
- ``media.cloud_connection_id`` (UUID NULL) -- when synced from a
  cloud provider, the connection identifier.
- ``media.cloud_file_id`` (TEXT NULL) -- provider-side file id.
- ``media.cloud_file_url`` (TEXT NULL) -- provider-side URL.
- ``media.extraction_status`` (TEXT NOT NULL DEFAULT 'none') --
  ``'none'`` / ``'pending'`` / ``'complete'`` / ``'failed'``.
- ``media_content.search_vector`` already exists via v006; v021 also
  re-asserts ``media_content`` columns from the v0.7.5 factory that
  metallm Alembic landed but 3tears migrations omitted: ``model_id``,
  ``provider_id``, ``model_name``, ``provider_name``,
  ``token_count_prompt``, ``token_count_completion``, ``cost``,
  ``metadata_json``.

Columns promoted to NOT NULL + DEFAULT to match prod parity:

- ``media.metadata_json``: v006 declared this as ``JSONB NULL``; prod
  metallm has it ``JSONB NOT NULL DEFAULT '{}'::jsonb``. v021 closes
  the gap: backfill NULL → ``'{}'::jsonb`` first, then set the
  default + NOT NULL. (``media.media_category`` and
  ``media.extraction_status`` already carry the right shape: v006
  declared ``media_category VARCHAR(64) NOT NULL`` without a server
  default and the v0.8.0 ``MediaCollection.schema`` declares the
  default — but since the column is already NOT NULL, Postgres
  accepts whatever value the Python writer supplies. Adding the
  server default below for symmetry with prod and to let the
  SchemaBackedCollection INSERT generator omit the column when the
  caller omits it.)
- ``media.date_updated``: v006 declared the column ``TIMESTAMPTZ
  NULL`` with no default. v0.8.0 promotes it to ``NOT NULL DEFAULT
  now()`` AND installs a ``BEFORE UPDATE`` trigger that re-sets the
  column on every row update — the column carries meaningful
  modification-timestamp data automatically, no app-code coupling.
  The schema declares the column ``immutable=True`` so the
  Collection's UPDATE generator excludes it from SET clauses; the
  trigger owns the write path. Backfill any pre-existing NULL rows
  to ``date_created`` before SET NOT NULL.

Idempotency: every ALTER uses ``ADD COLUMN IF NOT EXISTS`` so replay
on recovery is safe. The NOT NULL promotion + SET DEFAULT statements
on ``media.metadata_json`` are guarded by a DO block that inspects
``information_schema`` first (Postgres has no
``ALTER COLUMN ... SET NOT NULL IF NOT NULL``). v0.7.5 factory parity
is the target shape; subsequent test runs should observe zero
phantom migrations between the 3tears migration output and the prod
metallm Alembic output for these tables.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_chunk_index_and_token_count",
]

log = get_logger(__name__)


# memory_chunks: chunk_index + token_count are required NOT NULL on
# prod; defaulted server-side so existing rows back-fill cleanly.
_ADD_CHUNK_INDEX_SQL = "ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS chunk_index INTEGER NOT NULL DEFAULT 0"

_ADD_TOKEN_COUNT_SQL = "ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS token_count INTEGER NOT NULL DEFAULT 0"

# media: prod adds many columns over v006 that the 3tears migrations
# never followed up on. Required (NOT NULL) columns carry server-side
# defaults so the constraint lands cleanly against existing rows.
# mime_type / size_bytes / source defaults match prod intent; nullable
# columns get NULL by default.
_ADD_MEDIA_MIME_TYPE_SQL = "ALTER TABLE media ADD COLUMN IF NOT EXISTS mime_type TEXT NOT NULL DEFAULT ''"

_ADD_MEDIA_SIZE_BYTES_SQL = "ALTER TABLE media ADD COLUMN IF NOT EXISTS size_bytes INTEGER NOT NULL DEFAULT 0"

_ADD_MEDIA_SOURCE_SQL = "ALTER TABLE media ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT ''"

_ADD_MEDIA_S3_KEY_SQL = "ALTER TABLE media ADD COLUMN IF NOT EXISTS s3_key TEXT NULL"

_ADD_MEDIA_GENERATION_PROMPT_SQL = "ALTER TABLE media ADD COLUMN IF NOT EXISTS generation_prompt TEXT NULL"

_ADD_MEDIA_THUMBNAIL_S3_KEY_SQL = "ALTER TABLE media ADD COLUMN IF NOT EXISTS thumbnail_s3_key TEXT NULL"

_ADD_MEDIA_CLOUD_CONNECTION_ID_SQL = "ALTER TABLE media ADD COLUMN IF NOT EXISTS cloud_connection_id UUID NULL"

_ADD_MEDIA_CLOUD_FILE_ID_SQL = "ALTER TABLE media ADD COLUMN IF NOT EXISTS cloud_file_id TEXT NULL"

_ADD_MEDIA_CLOUD_FILE_URL_SQL = "ALTER TABLE media ADD COLUMN IF NOT EXISTS cloud_file_url TEXT NULL"

_ADD_MEDIA_EXTRACTION_STATUS_SQL = (
    "ALTER TABLE media ADD COLUMN IF NOT EXISTS extraction_status TEXT NOT NULL DEFAULT 'none'"
)

# media.date_updated: 3tears v006 added the column with no default and
# nullable. v0.8.0 promotes it to NOT NULL DEFAULT now() and installs a
# BEFORE UPDATE trigger that auto-sets the column on every row update,
# so the column carries meaningful data without app-code coupling. The
# schema declares ``immutable=True`` so the Collection's UPDATE
# generator excludes the column from SET clauses; the trigger owns the
# write path.
#
# Migration walks:
#   1. ADD COLUMN IF NOT EXISTS (defensive for fresh DBs that never ran
#      v006; idempotent for v006 consumers who already have the column).
#   2. Backfill any legacy NULL rows to date_created.
#   3. DO-block SET DEFAULT + SET NOT NULL (guarded so replay is a
#      no-op).
#   4. CREATE OR REPLACE the trigger function (idempotent).
#   5. DROP TRIGGER IF EXISTS + CREATE TRIGGER (idempotent guard).
#
# Metallm prod does NOT (yet) have the column; the parallel Alembic
# migration in shard 05 adds it + the trigger on the metallm side so
# both consumers converge on the richer shape.
_ADD_MEDIA_DATE_UPDATED_SQL = "ALTER TABLE media ADD COLUMN IF NOT EXISTS date_updated TIMESTAMPTZ DEFAULT now()"

_BACKFILL_MEDIA_DATE_UPDATED_SQL = "UPDATE media SET date_updated = date_created WHERE date_updated IS NULL"

_PROMOTE_MEDIA_DATE_UPDATED_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'media'
          AND column_name = 'date_updated'
          AND is_nullable = 'YES'
    ) THEN
        ALTER TABLE media ALTER COLUMN date_updated SET DEFAULT now();
        ALTER TABLE media ALTER COLUMN date_updated SET NOT NULL;
    END IF;
END
$$
"""

_MEDIA_DATE_UPDATED_TRIGGER_FN_SQL = """
CREATE OR REPLACE FUNCTION media_set_date_updated()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.date_updated := now();
    RETURN NEW;
END;
$$
"""

_DROP_MEDIA_DATE_UPDATED_TRIGGER_SQL = "DROP TRIGGER IF EXISTS media_date_updated_trigger ON media"

_CREATE_MEDIA_DATE_UPDATED_TRIGGER_SQL = """
CREATE TRIGGER media_date_updated_trigger
    BEFORE UPDATE ON media
    FOR EACH ROW
    EXECUTE FUNCTION media_set_date_updated()
"""

# media_content: prod adds provider/model/cost/metadata columns over
# v006 that the 3tears migrations never followed up on. All nullable.
_ADD_MC_MODEL_ID_SQL = "ALTER TABLE media_content ADD COLUMN IF NOT EXISTS model_id UUID NULL"

_ADD_MC_PROVIDER_ID_SQL = "ALTER TABLE media_content ADD COLUMN IF NOT EXISTS provider_id UUID NULL"

_ADD_MC_MODEL_NAME_SQL = "ALTER TABLE media_content ADD COLUMN IF NOT EXISTS model_name TEXT NULL"

_ADD_MC_PROVIDER_NAME_SQL = "ALTER TABLE media_content ADD COLUMN IF NOT EXISTS provider_name TEXT NULL"

_ADD_MC_TOKEN_COUNT_PROMPT_SQL = "ALTER TABLE media_content ADD COLUMN IF NOT EXISTS token_count_prompt INTEGER NULL"

_ADD_MC_TOKEN_COUNT_COMPLETION_SQL = (
    "ALTER TABLE media_content ADD COLUMN IF NOT EXISTS token_count_completion INTEGER NULL"
)

_ADD_MC_COST_SQL = "ALTER TABLE media_content ADD COLUMN IF NOT EXISTS cost NUMERIC(12, 8) NULL"

_ADD_MC_METADATA_JSON_SQL = "ALTER TABLE media_content ADD COLUMN IF NOT EXISTS metadata_json JSONB NULL"


# media.metadata_json: v006 declared this as ``JSONB NULL`` (no default).
# Prod metallm has ``JSONB NOT NULL DEFAULT '{}'::jsonb``. Backfill +
# promote so v0.8.0 SchemaBackedCollection writes that omit
# metadata_json land cleanly on the server-side default. Backfill must
# come first because the SET NOT NULL fails on existing NULL rows.
_BACKFILL_MEDIA_METADATA_JSON_SQL = "UPDATE media SET metadata_json = '{}'::jsonb WHERE metadata_json IS NULL"

# Postgres has no ``ALTER COLUMN ... SET NOT NULL IF NOT NULL`` — guard
# the promotion with a DO block that inspects ``information_schema``
# first so v021 replays cleanly. The SET DEFAULT is idempotent on its
# own (re-setting the same default is a no-op) but is kept inside the
# guard for parity.
_PROMOTE_MEDIA_METADATA_JSON_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'media'
          AND column_name = 'metadata_json'
          AND is_nullable = 'YES'
    ) THEN
        ALTER TABLE media ALTER COLUMN metadata_json SET DEFAULT '{}'::jsonb;
        ALTER TABLE media ALTER COLUMN metadata_json SET NOT NULL;
    END IF;
END
$$
"""

# media.media_category: v006 declared ``VARCHAR(64) NOT NULL`` with no
# server default. v0.7.5 factory + prod carry ``DEFAULT 'image'::text``;
# the v0.8.0 ``MediaCollection.schema`` declares the matching
# ``server_default``. Idempotent: re-issuing SET DEFAULT with the same
# value is a no-op in Postgres.
_SET_MEDIA_CATEGORY_DEFAULT_SQL = "ALTER TABLE media ALTER COLUMN media_category SET DEFAULT 'image'::text"

# media.extraction_status: v021 declared the column with the right
# default above (idempotent ADD COLUMN IF NOT EXISTS ... DEFAULT
# 'none'); the SET DEFAULT here covers the corner case where a prior
# v021 run added the column WITHOUT the default (defensive replay).
_SET_MEDIA_EXTRACTION_STATUS_DEFAULT_SQL = "ALTER TABLE media ALTER COLUMN extraction_status SET DEFAULT 'none'::text"

# conversation_memory_refs.date_created: v002 declared without a
# default; v014 renamed date_added -> date_created but did not add
# the default. Prod has ``TIMESTAMPTZ NOT NULL DEFAULT now()``. SET
# DEFAULT is idempotent in Postgres so no DO-block guard needed; the
# v0.8.0 ``MemoryRefsCollection.schema`` declares the matching
# ``server_default="now()"``.
_SET_CMR_DATE_CREATED_DEFAULT_SQL = "ALTER TABLE conversation_memory_refs ALTER COLUMN date_created SET DEFAULT now()"


async def add_chunk_index_and_token_count(store: DataStore) -> None:
    """align 3tears memory schemas with prod (v0.8.0 shard 03).

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("aligning memory + media + media_content schemas with prod (v021)")
    # memory_chunks
    await store.execute(_ADD_CHUNK_INDEX_SQL)
    await store.execute(_ADD_TOKEN_COUNT_SQL)
    # media
    await store.execute(_ADD_MEDIA_MIME_TYPE_SQL)
    await store.execute(_ADD_MEDIA_SIZE_BYTES_SQL)
    await store.execute(_ADD_MEDIA_SOURCE_SQL)
    await store.execute(_ADD_MEDIA_S3_KEY_SQL)
    await store.execute(_ADD_MEDIA_GENERATION_PROMPT_SQL)
    await store.execute(_ADD_MEDIA_THUMBNAIL_S3_KEY_SQL)
    await store.execute(_ADD_MEDIA_CLOUD_CONNECTION_ID_SQL)
    await store.execute(_ADD_MEDIA_CLOUD_FILE_ID_SQL)
    await store.execute(_ADD_MEDIA_CLOUD_FILE_URL_SQL)
    await store.execute(_ADD_MEDIA_EXTRACTION_STATUS_SQL)
    # media.date_updated: ADD (defensive) + backfill NULLs to
    # date_created + promote NOT NULL + DEFAULT now() via DO-block
    # guard. Then CREATE OR REPLACE the trigger function and
    # (idempotently re-)install the BEFORE UPDATE trigger.
    await store.execute(_ADD_MEDIA_DATE_UPDATED_SQL)
    await store.execute(_BACKFILL_MEDIA_DATE_UPDATED_SQL)
    await store.execute(_PROMOTE_MEDIA_DATE_UPDATED_SQL)
    await store.execute(_MEDIA_DATE_UPDATED_TRIGGER_FN_SQL)
    await store.execute(_DROP_MEDIA_DATE_UPDATED_TRIGGER_SQL)
    await store.execute(_CREATE_MEDIA_DATE_UPDATED_TRIGGER_SQL)
    # media: align metadata_json with prod (NOT NULL + DEFAULT) and
    # pin server defaults on the two columns that v006 declared
    # NOT NULL without a default.
    await store.execute(_BACKFILL_MEDIA_METADATA_JSON_SQL)
    await store.execute(_PROMOTE_MEDIA_METADATA_JSON_SQL)
    await store.execute(_SET_MEDIA_CATEGORY_DEFAULT_SQL)
    await store.execute(_SET_MEDIA_EXTRACTION_STATUS_DEFAULT_SQL)
    # conversation_memory_refs: pin date_created server default to
    # match prod (now()).
    await store.execute(_SET_CMR_DATE_CREATED_DEFAULT_SQL)
    # media_content
    await store.execute(_ADD_MC_MODEL_ID_SQL)
    await store.execute(_ADD_MC_PROVIDER_ID_SQL)
    await store.execute(_ADD_MC_MODEL_NAME_SQL)
    await store.execute(_ADD_MC_PROVIDER_NAME_SQL)
    await store.execute(_ADD_MC_TOKEN_COUNT_PROMPT_SQL)
    await store.execute(_ADD_MC_TOKEN_COUNT_COMPLETION_SQL)
    await store.execute(_ADD_MC_COST_SQL)
    await store.execute(_ADD_MC_METADATA_JSON_SQL)

"""
agent-memory v006: create media + media_content tables.

memory-task-01. ``retrieval.py`` and ``tools.py`` query two tables that
no earlier migration creates:

- ``media`` — the parent record for an uploaded artifact (document,
  image, audio). Carries ``media_category`` + ``metadata_json``.
- ``media_content`` — one or more content rows per media item (OCR /
  extracted text / caption), each with its own embedding and FTS
  vector.

Retrieval joins ``media_content`` to ``media`` on ``media_id`` and
pulls ``med.media_category`` + ``med.metadata_json`` alongside the
content row's columns.

Tools' batch lookup path runs the same JOIN and also selects
``med.date_created``, so ``media`` carries a ``date_created`` column
too.

Idempotent: ``CREATE TABLE IF NOT EXISTS`` + ``CREATE INDEX IF NOT
EXISTS`` + trigger CREATE OR REPLACE pattern.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_media_tables",
]

log = get_logger(__name__)


_CREATE_MEDIA_SQL = """
CREATE TABLE IF NOT EXISTS media (
    media_id UUID PRIMARY KEY,
    agent_id UUID NULL,
    customer_id UUID NULL,
    user_id UUID NOT NULL,
    media_category VARCHAR(64) NOT NULL,
    metadata_json JSONB NULL,
    date_created TIMESTAMP NOT NULL,
    date_updated TIMESTAMP NOT NULL
)
"""

_CREATE_MEDIA_USER_IDX_SQL = "CREATE INDEX IF NOT EXISTS idx_media_user ON media (user_id)"

_CREATE_MEDIA_AGENT_IDX_SQL = "CREATE INDEX IF NOT EXISTS idx_media_agent ON media (agent_id)"

_CREATE_MEDIA_CONTENT_SQL = """
CREATE TABLE IF NOT EXISTS media_content (
    content_id UUID PRIMARY KEY,
    media_id UUID NOT NULL REFERENCES media(media_id) ON DELETE CASCADE,
    agent_id UUID NULL,
    customer_id UUID NULL,
    user_id UUID NOT NULL,
    content_type VARCHAR(64) NOT NULL,
    content TEXT NOT NULL,
    summary TEXT NULL,
    embedding public.vector(1024) NULL,
    search_vector TSVECTOR NULL,
    date_created TIMESTAMP NOT NULL
)
"""

_CREATE_MC_USER_IDX_SQL = "CREATE INDEX IF NOT EXISTS idx_mc_user ON media_content (user_id)"

_CREATE_MC_MEDIA_IDX_SQL = "CREATE INDEX IF NOT EXISTS idx_mc_media ON media_content (media_id)"

_CREATE_MC_EMBEDDING_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_mc_embedding "
    "ON media_content USING hnsw (embedding public.vector_cosine_ops) "
    "WHERE embedding IS NOT NULL"
)

_CREATE_MC_SEARCH_VECTOR_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_mc_search_vector ON media_content USING GIN (search_vector)"
)

_CREATE_MC_TRIGGER_FUNC_SQL = """
CREATE OR REPLACE FUNCTION media_content_search_vector_update()
RETURNS trigger AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', coalesce(NEW.content, '')), 'A')
        || setweight(to_tsvector('english', coalesce(NEW.summary, '')), 'B');
    RETURN NEW;
END
$$ LANGUAGE plpgsql
"""

_DROP_MC_TRIGGER_SQL = "DROP TRIGGER IF EXISTS media_content_search_vector_trigger ON media_content"

_CREATE_MC_TRIGGER_SQL = """
CREATE TRIGGER media_content_search_vector_trigger
BEFORE INSERT OR UPDATE OF content, summary ON media_content
FOR EACH ROW EXECUTE FUNCTION media_content_search_vector_update()
"""


async def create_media_tables(store: DataStore) -> None:
    """
    create media + media_content tables with indexes and FTS trigger.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    """
    log.info("creating media + media_content tables (v006)")
    await store.execute(_CREATE_MEDIA_SQL)
    await store.execute(_CREATE_MEDIA_USER_IDX_SQL)
    await store.execute(_CREATE_MEDIA_AGENT_IDX_SQL)
    await store.execute(_CREATE_MEDIA_CONTENT_SQL)
    await store.execute(_CREATE_MC_USER_IDX_SQL)
    await store.execute(_CREATE_MC_MEDIA_IDX_SQL)
    await store.execute(_CREATE_MC_EMBEDDING_IDX_SQL)
    await store.execute(_CREATE_MC_SEARCH_VECTOR_IDX_SQL)
    await store.execute(_CREATE_MC_TRIGGER_FUNC_SQL)
    await store.execute(_DROP_MC_TRIGGER_SQL)
    await store.execute(_CREATE_MC_TRIGGER_SQL)

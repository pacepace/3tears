"""Parity tests: production ``<Collection>.schema`` for every
agent-memory table must produce a SQLAlchemy :class:`Table`
structurally equal to the hand-written v0.8.0 reference fixture.

The reference fixtures live alongside the schemas they guard (this
file) so any future column add to ``MemoriesCollection.schema`` /
``MediaCollection.schema`` / etc. has its reference fixture in the
same package -- no cross-package layering inversion.

Structural-comparison helpers (``assert_tables_equivalent`` +
``column_signature`` / ``index_signature`` / ``fk_constraint_signature``)
live in :mod:`threetears.core.testing.sqla_parity` so generic
framework-mechanics tests stay in core and per-table fixtures stay
with their owning package.
"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column as SAColumn,
    DateTime,
    Enum as SAEnum,
    ForeignKey as SAForeignKey,
    ForeignKeyConstraint as SAForeignKeyConstraint,
    Index as SAIndex,
    Integer,
    Numeric,
    Text,
    UniqueConstraint as SAUniqueConstraint,
)
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from threetears.agent.memory.collections import (
    MediaCollection,
    MediaContentCollection,
    MemoriesCollection,
    MemoryChunkCollection,
    MemoryRefsCollection,
)
from threetears.core.testing.sqla_parity import assert_tables_equivalent

# Embedding dimension carried by ``memories`` / ``media_content`` /
# ``memory_chunks`` reference fixtures. Mirrors
# ``threetears.agent.memory.collections._MEMORY_VECTOR_DIM``; duplicated
# here so the reference fixtures remain a self-contained, frozen
# regression net independent of the source-under-test module.
_REFERENCE_MEMORY_VECTOR_DIM = 1024


# ---------------------------------------------------------------------------
# reference fixtures (FROZEN v0.8.0 canonical shapes)
# ---------------------------------------------------------------------------


def _reference_memories_table(metadata: sa.MetaData) -> sa.Table:
    """Hand-written reference Table for parity testing.

    Frozen v0.8.0 canonical shape. If prod schema legitimately
    changes, add a NEW reference function (e.g.
    ``_reference_memories_table_v090``) and run parity against the
    new one in addition to the old.
    """
    if "memories" in metadata.tables:
        return metadata.tables["memories"]
    return sa.Table(
        "memories",
        metadata,
        SAColumn("memory_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("agent_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("customer_id", PgUUID(as_uuid=True), nullable=False),
        SAColumn(
            "user_id",
            PgUUID(as_uuid=True),
            SAForeignKey("users.user_id"),
            nullable=False,
        ),
        SAColumn("conversation_id", PgUUID(as_uuid=True), nullable=False),
        # message_id_source FK is table-level (below) with
        # ON DELETE SET NULL to match prod metallm.
        SAColumn("message_id_source", PgUUID(as_uuid=True), nullable=True),
        SAColumn(
            "type_memory",
            SAEnum(
                "preference",
                "fact",
                "decision",
                "topical_context",
                "relational_context",
                name="memory_type",
                create_constraint=True,
            ),
            nullable=False,
        ),
        SAColumn("content", Text(), nullable=False),
        SAColumn("summary", Text(), nullable=True),
        SAColumn(
            "embedding",
            Vector(_REFERENCE_MEMORY_VECTOR_DIM),
            nullable=True,
        ),
        SAColumn("search_vector", TSVECTOR(), nullable=True),
        SAColumn("alias", Text(), nullable=True),
        SAColumn("date_created", DateTime(timezone=True), nullable=False),
        SAColumn("date_updated", DateTime(timezone=True), nullable=True),
        SAForeignKeyConstraint(
            ["message_id_source"],
            ["messages.message_id"],
            ondelete="SET NULL",
        ),
        SAIndex("ix_memories_user_date", "user_id", "date_created"),
        SAIndex(
            "ix_memories_user_alias",
            "agent_id",
            "user_id",
            "alias",
            unique=True,
            postgresql_where=sa_text("alias IS NOT NULL"),
        ),
        # v0.8.1 enrichments mirror prod metallm alembic.
        SAIndex("idx_memories_agent_user", "agent_id", "user_id"),
        SAIndex(
            "idx_memories_agent_customer_user",
            "agent_id",
            "customer_id",
            "user_id",
        ),
        SAIndex(
            "idx_memories_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
        SAIndex(
            "ix_memories_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": "16", "ef_construction": "64"},
        ),
        SAUniqueConstraint("memory_id", name="uq_memories_memory_id"),
    )


def _reference_media_table(metadata: sa.MetaData) -> sa.Table:
    """Hand-written reference Table for parity testing.

    Frozen v0.8.0 canonical shape. If prod schema legitimately
    changes, add a NEW reference function (e.g.
    ``_reference_media_table_v090``) and run parity against the new
    one in addition to the old.
    """
    if "media" in metadata.tables:
        return metadata.tables["media"]
    return sa.Table(
        "media",
        metadata,
        SAColumn("agent_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("media_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("customer_id", PgUUID(as_uuid=True), nullable=False),
        SAColumn(
            "user_id",
            PgUUID(as_uuid=True),
            SAForeignKey("users.user_id"),
            nullable=False,
        ),
        SAColumn("s3_key", Text(), nullable=True),
        SAColumn("mime_type", Text(), nullable=False),
        SAColumn("size_bytes", Integer(), nullable=False),
        SAColumn("source", Text(), nullable=False),
        SAColumn(
            "metadata_json",
            JSONB(),
            nullable=False,
            server_default="'{}'::jsonb",
        ),
        SAColumn("generation_prompt", Text(), nullable=True),
        SAColumn(
            "media_category",
            Text(),
            nullable=False,
            server_default="'image'::text",
        ),
        SAColumn(
            "extraction_status",
            Text(),
            nullable=False,
            server_default="'none'::text",
        ),
        SAColumn("thumbnail_s3_key", Text(), nullable=True),
        SAColumn("cloud_connection_id", PgUUID(as_uuid=True), nullable=True),
        SAColumn("cloud_file_id", Text(), nullable=True),
        SAColumn("cloud_file_url", Text(), nullable=True),
        SAColumn("memory_id", PgUUID(as_uuid=True), nullable=False),
        SAColumn("date_created", DateTime(timezone=True), nullable=False),
        SAColumn(
            "date_updated",
            DateTime(timezone=True),
            nullable=False,
            server_default="now()",
        ),
        SAForeignKeyConstraint(
            ["cloud_connection_id"],
            ["cloud_connections.cloud_connection_id"],
            ondelete="SET NULL",
        ),
        SAForeignKeyConstraint(
            ["memory_id"],
            ["memories.memory_id"],
            ondelete="CASCADE",
        ),
        SAIndex("ix_media_user_date", "user_id", "date_created"),
        SAIndex("ix_media_mime_type", "mime_type"),
        SAIndex("ix_media_memory_id", "memory_id"),
        SAIndex(
            "uq_media_cloud_connection_file",
            "cloud_connection_id",
            "cloud_file_id",
            unique=True,
        ),
        # v0.8.1 enrichments mirror prod metallm alembic.
        SAIndex("idx_media_agent_user", "agent_id", "user_id"),
        SAIndex(
            "ix_media_extraction_pending",
            "extraction_status",
            postgresql_where=sa_text("extraction_status = 'pending'"),
        ),
        SAUniqueConstraint("media_id", name="uq_media_media_id"),
    )


def _reference_media_content_table(metadata: sa.MetaData) -> sa.Table:
    """Hand-written reference Table for parity testing.

    Frozen v0.8.0 canonical shape.
    """
    if "media_content" in metadata.tables:
        return metadata.tables["media_content"]
    return sa.Table(
        "media_content",
        metadata,
        SAColumn("agent_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("content_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("customer_id", PgUUID(as_uuid=True), nullable=False),
        # media_id FK is composite at table level (below)
        SAColumn("media_id", PgUUID(as_uuid=True), nullable=False),
        SAColumn(
            "user_id",
            PgUUID(as_uuid=True),
            SAForeignKey("users.user_id"),
            nullable=False,
        ),
        SAColumn("content_type", Text(), nullable=False),
        SAColumn("content", Text(), nullable=False),
        SAColumn("summary", Text(), nullable=True),
        SAColumn(
            "embedding",
            Vector(_REFERENCE_MEMORY_VECTOR_DIM),
            nullable=True,
        ),
        SAColumn("search_vector", TSVECTOR(), nullable=True),
        SAColumn(
            "model_id",
            PgUUID(as_uuid=True),
            SAForeignKey("models.model_id"),
            nullable=True,
        ),
        SAColumn(
            "provider_id",
            PgUUID(as_uuid=True),
            SAForeignKey("providers.provider_id"),
            nullable=True,
        ),
        SAColumn("model_name", Text(), nullable=True),
        SAColumn("provider_name", Text(), nullable=True),
        SAColumn("token_count_prompt", Integer(), nullable=True),
        SAColumn("token_count_completion", Integer(), nullable=True),
        SAColumn("cost", Numeric(12, 8), nullable=True),
        SAColumn("metadata_json", JSONB(), nullable=True),
        SAColumn("date_created", DateTime(timezone=True), nullable=False),
        SAForeignKeyConstraint(
            ["agent_id", "media_id"],
            ["media.agent_id", "media.media_id"],
            ondelete="CASCADE",
        ),
        SAIndex("ix_media_content_media_type", "media_id", "content_type"),
        SAIndex("ix_media_content_user", "user_id"),
        # v0.8.1 enrichments mirror prod metallm alembic.
        SAIndex(
            "idx_media_content_agent_user",
            "agent_id",
            "user_id",
        ),
        SAIndex(
            "idx_media_content_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
        # prod does NOT carry a WITH clause for this HNSW index.
        SAIndex(
            "ix_media_content_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        SAUniqueConstraint("content_id", name="uq_media_content_content_id"),
    )


def _reference_memory_chunks_table(metadata: sa.MetaData) -> sa.Table:
    """Hand-written reference Table for parity testing.

    Frozen v0.8.0 canonical shape.
    """
    if "memory_chunks" in metadata.tables:
        return metadata.tables["memory_chunks"]
    return sa.Table(
        "memory_chunks",
        metadata,
        SAColumn("agent_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("chunk_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("customer_id", PgUUID(as_uuid=True), nullable=False),
        # memory_id FK is composite at table level (below)
        SAColumn("memory_id", PgUUID(as_uuid=True), nullable=False),
        SAColumn(
            "user_id",
            PgUUID(as_uuid=True),
            SAForeignKey("users.user_id"),
            nullable=False,
        ),
        SAColumn("chunk_index", Integer(), nullable=False),
        SAColumn("content", Text(), nullable=False),
        SAColumn("summary", Text(), nullable=True),
        SAColumn("heading_context", Text(), nullable=True),
        SAColumn("page_number", Integer(), nullable=True),
        SAColumn("token_count", Integer(), nullable=False),
        SAColumn(
            "embedding",
            Vector(_REFERENCE_MEMORY_VECTOR_DIM),
            nullable=True,
        ),
        SAColumn("search_vector", TSVECTOR(), nullable=True),
        SAColumn("message_id_start", PgUUID(as_uuid=True), nullable=True),
        SAColumn("message_id_end", PgUUID(as_uuid=True), nullable=True),
        SAColumn("date_created", DateTime(timezone=True), nullable=False),
        SAForeignKeyConstraint(
            ["agent_id", "memory_id"],
            ["memories.agent_id", "memories.memory_id"],
            ondelete="CASCADE",
        ),
        SAIndex("ix_memory_chunks_memory", "memory_id", "chunk_index"),
        SAIndex("ix_memory_chunks_user", "user_id"),
        # v0.8.1 enrichments mirror prod metallm alembic.
        SAIndex(
            "idx_chunks_message_id_start",
            "agent_id",
            "message_id_start",
            postgresql_where=sa_text("message_id_start IS NOT NULL"),
        ),
        SAIndex(
            "idx_memory_chunks_agent_user",
            "agent_id",
            "user_id",
        ),
        SAIndex(
            "idx_memory_chunks_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
        # prod does NOT carry a WITH clause for this HNSW index.
        SAIndex(
            "ix_memory_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        SAUniqueConstraint("chunk_id", name="uq_memory_chunks_chunk_id"),
    )


def _reference_conversation_memory_refs_table(metadata: sa.MetaData) -> sa.Table:
    """Hand-written reference Table for parity testing.

    Frozen v0.8.0 canonical shape.
    """
    if "conversation_memory_refs" in metadata.tables:
        return metadata.tables["conversation_memory_refs"]
    return sa.Table(
        "conversation_memory_refs",
        metadata,
        SAColumn("conversation_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("item_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("item_type", Text(), nullable=False),
        SAColumn("short_desc", Text(), nullable=False),
        SAColumn(
            "date_created",
            DateTime(timezone=True),
            nullable=False,
            server_default=sa_text("now()"),
        ),
        SAColumn("date_updated", DateTime(timezone=True), nullable=False),
        SAForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.conversation_id"],
            ondelete="CASCADE",
        ),
        SAIndex("ix_conversation_memory_refs_cid", "conversation_id"),
    )


# ---------------------------------------------------------------------------
# parity tests — Collection-schema vs reference
# ---------------------------------------------------------------------------


def test_parity_memories_collection_schema() -> None:
    """Production :attr:`MemoriesCollection.schema` produces a Table
    structurally equal to ``_reference_memories_table``.
    """
    via_reference = _reference_memories_table(sa.MetaData())
    via_collection = MemoriesCollection.schema.to_sqlalchemy_table(sa.MetaData())
    assert_tables_equivalent(via_reference, via_collection)


def test_parity_media_collection_schema() -> None:
    """Production :attr:`MediaCollection.schema` produces a Table
    structurally equal to ``_reference_media_table``.
    """
    via_reference = _reference_media_table(sa.MetaData())
    via_collection = MediaCollection.schema.to_sqlalchemy_table(sa.MetaData())
    assert_tables_equivalent(via_reference, via_collection)


def test_parity_media_content_collection_schema() -> None:
    """Production :attr:`MediaContentCollection.schema` produces a
    Table structurally equal to ``_reference_media_content_table`` AND
    carries the composite FK ``(agent_id, media_id) → media``.
    """
    via_reference = _reference_media_content_table(sa.MetaData())
    via_collection = MediaContentCollection.schema.to_sqlalchemy_table(sa.MetaData())
    assert_tables_equivalent(via_reference, via_collection)

    # ground-truth augmentation: confirm the composite FK is emitted
    # on the Collection-schema side.
    composite_fks = [
        c for c in via_collection.constraints if isinstance(c, sa.ForeignKeyConstraint) and len(c.column_keys) > 1
    ]
    assert any(
        tuple(c.column_keys) == ("agent_id", "media_id")
        and c.elements[0].target_fullname.split(".", 1)[0] == "media"
        and c.ondelete == "CASCADE"
        for c in composite_fks
    ), "composite FK (agent_id, media_id) → media missing from media_content"


def test_parity_memory_chunks_collection_schema() -> None:
    """Production :attr:`MemoryChunkCollection.schema` produces a
    Table structurally equal to ``_reference_memory_chunks_table`` AND
    carries the composite FK ``(agent_id, memory_id) → memories``.
    """
    via_reference = _reference_memory_chunks_table(sa.MetaData())
    via_collection = MemoryChunkCollection.schema.to_sqlalchemy_table(sa.MetaData())
    assert_tables_equivalent(via_reference, via_collection)

    composite_fks = [
        c for c in via_collection.constraints if isinstance(c, sa.ForeignKeyConstraint) and len(c.column_keys) > 1
    ]
    assert any(
        tuple(c.column_keys) == ("agent_id", "memory_id")
        and c.elements[0].target_fullname.split(".", 1)[0] == "memories"
        and c.ondelete == "CASCADE"
        for c in composite_fks
    ), "composite FK (agent_id, memory_id) → memories missing from memory_chunks"


def test_parity_conversation_memory_refs_collection_schema() -> None:
    """Production :attr:`MemoryRefsCollection.schema` produces a Table
    structurally equal to ``_reference_conversation_memory_refs_table``.
    """
    via_reference = _reference_conversation_memory_refs_table(sa.MetaData())
    via_collection = MemoryRefsCollection.schema.to_sqlalchemy_table(sa.MetaData())
    assert_tables_equivalent(via_reference, via_collection)

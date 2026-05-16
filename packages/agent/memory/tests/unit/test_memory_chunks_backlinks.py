"""Unit tests for the unified-memory backlink primitives on chunks.

Covers the additive surface from transcript-chunks-task-A:

- :class:`MemoryChunkEntity` exposes ``memory_id``, ``message_id_start``,
  ``message_id_end`` properties with the correct nullability semantics.
- :class:`MemoryChunkCollection` schema declares the new columns.
- The shared ``_chunk_row_to_dict`` helper round-trips the new fields.

End-to-end SQL coverage (the four new collection methods +
hybrid_search cursor paging) lives in the integration suite at
``tests/integration/test_chunk_collection_methods.py``.
"""

from __future__ import annotations

import uuid

import pytest

from threetears.agent.memory.collections import (
    MemoryChunkCollection,
    _chunk_row_to_dict,
)
from threetears.agent.memory.entities import MemoryChunkEntity


# -- Entity property semantics ------------------------------------------------


class TestMemoryChunkEntityBacklinks:
    def test_memory_id_required_present(self) -> None:
        """``memory_id`` is the canonical parent pointer; reading returns UUID."""
        memory_id = uuid.uuid7()
        entity = MemoryChunkEntity(
            {
                "agent_id": uuid.uuid7(),
                "chunk_id": uuid.uuid7(),
                "memory_id": memory_id,
                "content": "chunk text",
            }
        )
        assert entity.memory_id == memory_id

    def test_memory_id_setter_accepts_uuid(self) -> None:
        entity = MemoryChunkEntity(
            {
                "agent_id": uuid.uuid7(),
                "chunk_id": uuid.uuid7(),
                "memory_id": uuid.uuid7(),
                "content": "x",
            }
        )
        new_id = uuid.uuid7()
        entity.memory_id = new_id
        assert entity.memory_id == new_id

    def test_memory_id_accepts_string_input(self) -> None:
        memory_id = uuid.uuid7()
        entity = MemoryChunkEntity(
            {
                "agent_id": uuid.uuid7(),
                "chunk_id": uuid.uuid7(),
                "memory_id": str(memory_id),
                "content": "x",
            }
        )
        assert entity.memory_id == memory_id

    def test_message_id_start_defaults_none(self) -> None:
        """Document chunks have no transcript backlink — NULL is the contract."""
        entity = MemoryChunkEntity(
            {
                "agent_id": uuid.uuid7(),
                "chunk_id": uuid.uuid7(),
                "memory_id": uuid.uuid7(),
                "content": "doc chunk",
                "message_id_start": None,
                "message_id_end": None,
            }
        )
        assert entity.message_id_start is None
        assert entity.message_id_end is None

    def test_message_id_start_and_end_round_trip(self) -> None:
        """Transcript chunks carry a (start, end) message-range backlink."""
        start = uuid.uuid7()
        end = uuid.uuid7()
        entity = MemoryChunkEntity(
            {
                "agent_id": uuid.uuid7(),
                "chunk_id": uuid.uuid7(),
                "memory_id": uuid.uuid7(),
                "content": "transcript chunk",
                "message_id_start": start,
                "message_id_end": end,
            }
        )
        assert entity.message_id_start == start
        assert entity.message_id_end == end

    def test_message_id_start_setter_supports_none(self) -> None:
        entity = MemoryChunkEntity(
            {
                "agent_id": uuid.uuid7(),
                "chunk_id": uuid.uuid7(),
                "memory_id": uuid.uuid7(),
                "content": "x",
                "message_id_start": uuid.uuid7(),
            }
        )
        assert entity.message_id_start is not None
        entity.message_id_start = None
        assert entity.message_id_start is None

    def test_message_id_end_setter_round_trips_uuid(self) -> None:
        entity = MemoryChunkEntity(
            {
                "agent_id": uuid.uuid7(),
                "chunk_id": uuid.uuid7(),
                "memory_id": uuid.uuid7(),
                "content": "x",
                "message_id_end": None,
            }
        )
        new_end = uuid.uuid7()
        entity.message_id_end = new_end
        assert entity.message_id_end == new_end

    def test_media_id_property_removed(self) -> None:
        """``media_id`` was dropped from the chunk in v018; reverse lookups
        flow through ``memory_id`` -> media now."""
        entity = MemoryChunkEntity(
            {
                "agent_id": uuid.uuid7(),
                "chunk_id": uuid.uuid7(),
                "memory_id": uuid.uuid7(),
                "content": "x",
            }
        )
        assert not hasattr(entity, "media_id") or "media_id" not in dir(type(entity))


# -- Schema declaration -------------------------------------------------------


class TestMemoryChunkCollectionSchema:
    def test_schema_declares_memory_id_column(self) -> None:
        names = {col.name for col in MemoryChunkCollection.schema.columns}
        assert "memory_id" in names

    def test_schema_declares_message_id_start_column(self) -> None:
        names = {col.name for col in MemoryChunkCollection.schema.columns}
        assert "message_id_start" in names

    def test_schema_declares_message_id_end_column(self) -> None:
        names = {col.name for col in MemoryChunkCollection.schema.columns}
        assert "message_id_end" in names

    def test_schema_no_longer_declares_media_id(self) -> None:
        """``media_id`` was dropped in v018 — chunks parent to memory now."""
        names = {col.name for col in MemoryChunkCollection.schema.columns}
        assert "media_id" not in names

    def test_message_id_columns_are_nullable(self) -> None:
        """Document chunks leave these NULL; only transcript chunks fill them."""
        by_name = {col.name: col for col in MemoryChunkCollection.schema.columns}
        assert by_name["message_id_start"].nullable is True
        assert by_name["message_id_end"].nullable is True


# -- Helper round-trip --------------------------------------------------------


# parity-with: asyncpg.Record
class _FakeRow:
    """Minimal asyncpg.Record stand-in: dict subscript only.

    ``_chunk_row_to_dict`` only ever subscripts the row by column name;
    this fake re-implements the one operation we need. ``asyncpg.Record``
    is a C-extension class we can't subclass directly from Python.
    """

    def __init__(self, **kwargs: object) -> None:
        self._d = dict(kwargs)

    def __getitem__(self, key: str) -> object:
        return self._d[key]


class TestChunkRowToDict:
    @pytest.fixture
    def base_row(self) -> dict[str, object]:
        return {
            "chunk_id": uuid.uuid7(),
            "content": "verbatim text",
            "summary": "one-liner",
            "heading_context": None,
            "page_number": None,
            "memory_id": uuid.uuid7(),
            "media_id": None,
            "message_id_start": None,
            "message_id_end": None,
            "metadata_json": None,
            "embedding": None,
        }

    def test_transcript_chunk_fields_round_trip(self, base_row: dict[str, object]) -> None:
        start = uuid.uuid7()
        end = uuid.uuid7()
        base_row["message_id_start"] = start
        base_row["message_id_end"] = end
        result = _chunk_row_to_dict(_FakeRow(**base_row), score_key="similarity", score_value=0.5)
        assert result["message_id_start"] == str(start)
        assert result["message_id_end"] == str(end)
        assert result["memory_id"] == str(base_row["memory_id"])

    def test_document_chunk_has_null_message_backlinks(self, base_row: dict[str, object]) -> None:
        result = _chunk_row_to_dict(_FakeRow(**base_row), score_key="similarity", score_value=0.5)
        assert result["message_id_start"] is None
        assert result["message_id_end"] is None

    def test_score_key_semantic_seeds_similarity(self, base_row: dict[str, object]) -> None:
        result = _chunk_row_to_dict(_FakeRow(**base_row), score_key="similarity", score_value=0.42)
        assert result["similarity"] == 0.42
        assert result["fts_rank"] == 0.0

    def test_score_key_fts_rank_seeds_fts_rank(self, base_row: dict[str, object]) -> None:
        result = _chunk_row_to_dict(_FakeRow(**base_row), score_key="fts_rank", score_value=0.71)
        assert result["fts_rank"] == 0.71
        assert result["similarity"] == 0.0

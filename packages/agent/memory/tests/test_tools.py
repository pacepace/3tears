"""Tests for memory tools schemas and helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from threetears.agent.memory.tools import (
    ChunkRecallInput,
    ChunkSearchInput,
    MemoryRecallInput,
    MemorySearchInput,
    _fmt_dt,
    _tool_error,
)


# -- Input schema validation --------------------------------------------------


class TestMemorySearchInput:
    def test_query_only(self):
        inp = MemorySearchInput(query="what is my name")
        assert inp.query == "what is my name"
        assert inp.ids is None
        assert inp.type_filter is None

    def test_ids_only(self):
        inp = MemorySearchInput(ids=["abc-123"])
        assert inp.ids == ["abc-123"]
        assert inp.query == ""

    def test_both_query_and_ids(self):
        inp = MemorySearchInput(query="test", ids=["id-1"])
        assert inp.query == "test"
        assert inp.ids == ["id-1"]

    def test_neither_raises(self):
        with pytest.raises(ValidationError, match="query.*ids"):
            MemorySearchInput()

    def test_empty_query_no_ids_raises(self):
        with pytest.raises(ValidationError):
            MemorySearchInput(query="")

    def test_type_filter(self):
        inp = MemorySearchInput(query="test", type_filter="preference")
        assert inp.type_filter == "preference"


class TestMemoryRecallInput:
    def test_valid(self):
        inp = MemoryRecallInput(id="some-uuid", type="memory")
        assert inp.id == "some-uuid"
        assert inp.type == "memory"


# -- Helper functions ---------------------------------------------------------


class TestToolError:
    def test_format(self):
        result = _tool_error("memory_search", "embed", "connection timeout")
        assert result == "[TOOL ERROR] memory_search: embed failed — connection timeout"

    def test_format_consistency(self):
        result = _tool_error("memory_recall", "fetch", "not found")
        assert result.startswith("[TOOL ERROR]")
        assert "memory_recall" in result
        assert "fetch failed" in result


class TestFmtDt:
    def test_none(self):
        assert _fmt_dt(None) == ""

    def test_datetime(self):
        dt = datetime(2026, 3, 12, 14, 30, tzinfo=timezone.utc)
        result = _fmt_dt(dt)
        assert "Mar" in result
        assert "2026" in result

    def test_non_datetime_falls_through_to_str(self):
        assert _fmt_dt("some string") == "some string"


# -- Shard C input schemas (v0.7.0 transcript-chunks tools) ------------------


class TestChunkRecallInput:
    """``chunk_recall(chunk_id)`` -- single-chunk lookup by ID. The
    schema is minimal because the LLM only needs to supply the chunk_id;
    auth + parent-memory lookup happen inside the tool."""

    def test_valid(self):
        inp = ChunkRecallInput(chunk_id="some-uuid")
        assert inp.chunk_id == "some-uuid"

    def test_missing_chunk_id_raises(self):
        with pytest.raises(ValidationError):
            ChunkRecallInput()  # type: ignore[call-arg]


class TestChunkSearchInput:
    """``chunk_search(query, limit=5)`` -- cross-memory chunk hybrid
    search. The schema pins ``limit`` between 1 and 20 so the LLM
    can't request a runaway result set, and defaults to 5."""

    def test_valid_with_default_limit(self):
        inp = ChunkSearchInput(query="kerning argument")
        assert inp.query == "kerning argument"
        assert inp.limit == 5

    def test_custom_limit(self):
        inp = ChunkSearchInput(query="anything", limit=10)
        assert inp.limit == 10

    def test_limit_zero_raises(self):
        with pytest.raises(ValidationError):
            ChunkSearchInput(query="anything", limit=0)

    def test_limit_above_max_raises(self):
        with pytest.raises(ValidationError):
            ChunkSearchInput(query="anything", limit=21)

    def test_missing_query_raises(self):
        with pytest.raises(ValidationError):
            ChunkSearchInput()  # type: ignore[call-arg]
        assert _fmt_dt(42) == "42"

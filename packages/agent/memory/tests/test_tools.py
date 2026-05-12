"""Tests for memory tools schemas and helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from threetears.agent.memory.tools import (
    MemorySearchInput,
    RecallMemoryInput,
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


class TestRecallMemoryInput:
    def test_valid(self):
        inp = RecallMemoryInput(id="some-uuid", type="memory")
        assert inp.id == "some-uuid"
        assert inp.type == "memory"


# -- Helper functions ---------------------------------------------------------


class TestToolError:
    def test_format(self):
        result = _tool_error("memory_search", "embed", "connection timeout")
        assert result == "[TOOL ERROR] memory_search: embed failed — connection timeout"

    def test_format_consistency(self):
        result = _tool_error("recall_memory", "fetch", "not found")
        assert result.startswith("[TOOL ERROR]")
        assert "recall_memory" in result
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
        assert _fmt_dt(42) == "42"

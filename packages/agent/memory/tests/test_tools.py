"""Tests for memory tools schemas, and the error and timestamp shapes a tool returns."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.tools import (
    ChunkRecallInput,
    ChunkSearchInput,
    MemoryRecallInput,
    MemorySearchInput,
    load_memory_search_tool,
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
    """v0.7.0 (shard C) reshape: ``(memory_id, chunk_mode...)`` with
    four mutually-exclusive chunk-selection modes."""

    def test_minimum_fields(self) -> None:
        inp = MemoryRecallInput(memory_id="some-uuid")
        assert inp.memory_id == "some-uuid"
        assert inp.chunk_query is None
        assert inp.chunk_indexes is None
        assert inp.chunk_id_after is None
        assert inp.chunk_id_before is None
        assert inp.limit == 5

    def test_chunk_query_mode(self) -> None:
        inp = MemoryRecallInput(
            memory_id="some-uuid",
            chunk_query="needle",
            limit=10,
        )
        assert inp.chunk_query == "needle"
        assert inp.limit == 10

    def test_chunk_indexes_mode(self) -> None:
        inp = MemoryRecallInput(
            memory_id="some-uuid",
            chunk_indexes=[0, 2, 5],
        )
        assert inp.chunk_indexes == [0, 2, 5]

    def test_chunk_id_after_mode(self) -> None:
        inp = MemoryRecallInput(
            memory_id="some-uuid",
            chunk_id_after="cursor-uuid",
        )
        assert inp.chunk_id_after == "cursor-uuid"

    def test_chunk_id_before_mode(self) -> None:
        inp = MemoryRecallInput(
            memory_id="some-uuid",
            chunk_id_before="cursor-uuid",
        )
        assert inp.chunk_id_before == "cursor-uuid"

    def test_modes_are_mutually_exclusive(self) -> None:
        # Two modes set at once → ValidationError.
        with pytest.raises(ValidationError, match="at most one"):
            MemoryRecallInput(
                memory_id="any",
                chunk_query="q",
                chunk_indexes=[1],
            )
        with pytest.raises(ValidationError, match="at most one"):
            MemoryRecallInput(
                memory_id="any",
                chunk_id_after="a",
                chunk_id_before="b",
            )

    def test_missing_memory_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            MemoryRecallInput()  # type: ignore[call-arg]

    def test_limit_clamping(self) -> None:
        with pytest.raises(ValidationError):
            MemoryRecallInput(memory_id="any", limit=0)
        with pytest.raises(ValidationError):
            MemoryRecallInput(memory_id="any", limit=51)


# -- What a tool returns: the error line and the timestamp ----------------------


# parity-exempt: answers only find_by_alias, the one call memory_search's alias lookup makes; the SQL behind it is covered by the collection tests
class _FakeAliasMemories:
    """a memories collection whose alias lookup returns one fixed row, or raises."""

    def __init__(self, row: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.row = row
        self.error = error

    async def find_by_alias(self, *, user_id: UUID, agent_id: UUID, alias: str) -> dict[str, Any] | None:
        _ = user_id, agent_id, alias
        if self.error is not None:
            raise self.error
        return self.row


async def _search_by_alias(authorizer: MemoryAuthorizerDependencies, memories: _FakeAliasMemories) -> str:
    (tool,) = await load_memory_search_tool(
        uuid4(),
        None,  # type: ignore[arg-type]  # the alias lookup embeds nothing
        uuid4(),
        uuid4(),
        authorizer,
        memories,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
    )
    result = await tool.ainvoke({"alias": "home", "query": "home"})
    assert isinstance(result, str)
    return result


def _alias_row(date_created: Any) -> dict[str, Any]:
    return {"memory_id": uuid4(), "type_memory": "fact", "content": "lives in Leeds", "date_created": date_created}


class TestToolError:
    async def test_format(self, permissive_memory_authorizer: MemoryAuthorizerDependencies) -> None:
        result = await _search_by_alias(
            permissive_memory_authorizer, _FakeAliasMemories(error=RuntimeError("connection timeout"))
        )
        assert result == "[TOOL ERROR] memory_search: alias failed — connection timeout"

    async def test_format_consistency(self, permissive_memory_authorizer: MemoryAuthorizerDependencies) -> None:
        result = await _search_by_alias(
            permissive_memory_authorizer, _FakeAliasMemories(error=LookupError("not found"))
        )
        assert result.startswith("[TOOL ERROR]")
        assert "memory_search" in result
        assert "alias failed" in result
        assert result.endswith("not found")


class TestFmtDt:
    """the memory line carries ``[<timestamp>]`` when the row has one."""

    async def test_none(self, permissive_memory_authorizer: MemoryAuthorizerDependencies) -> None:
        result = await _search_by_alias(permissive_memory_authorizer, _FakeAliasMemories(_alias_row(None)))
        assert "[fact] lives in Leeds" in result

    async def test_datetime(self, permissive_memory_authorizer: MemoryAuthorizerDependencies) -> None:
        dt = datetime(2026, 3, 12, 14, 30, tzinfo=timezone.utc)
        result = await _search_by_alias(permissive_memory_authorizer, _FakeAliasMemories(_alias_row(dt)))
        assert "[fact] [Mar 12, 2026 at 2:30 PM] lives in Leeds" in result

    async def test_non_datetime_falls_through_to_str(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        result = await _search_by_alias(permissive_memory_authorizer, _FakeAliasMemories(_alias_row("some string")))
        assert "[fact] [some string] lives in Leeds" in result
        result = await _search_by_alias(permissive_memory_authorizer, _FakeAliasMemories(_alias_row(42)))
        assert "[fact] [42] lives in Leeds" in result


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

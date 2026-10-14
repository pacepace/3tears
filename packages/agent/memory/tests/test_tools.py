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


async def _search_by_alias(
    authorizer: MemoryAuthorizerDependencies, memories: _FakeAliasMemories, **loader_kwargs: Any
) -> str:
    (tool,) = await load_memory_search_tool(
        uuid4(),
        None,  # type: ignore[arg-type]  # the alias lookup embeds nothing
        uuid4(),
        uuid4(),
        authorizer,
        memories,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        **loader_kwargs,
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


class TestDatesAreInTheZoneTheCallerNames:
    """a date is read to a person who lives somewhere; an id is read by the next call."""

    async def test_an_alias_hit_is_dated_in_the_zone_and_names_it(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        row = _alias_row(datetime(2026, 3, 12, 14, 30, tzinfo=timezone.utc))
        result = await _search_by_alias(permissive_memory_authorizer, _FakeAliasMemories(row), tz="America/Los_Angeles")
        assert f"[memory:{row['memory_id']}] [fact] [Mar 12, 2026 at 7:30 AM PDT] lives in Leeds" in result

    async def test_a_search_dates_memories_and_files_in_the_zone_and_leaves_ids_as_stored(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        late = datetime(2026, 10, 10, 4, 11, tzinfo=timezone.utc)
        memory, media = _memory_row("lives in Leeds", late), _media_row("a photo of the house", late)
        result = await _search(
            permissive_memory_authorizer,
            memories=_FakeSearchRows(semantic=[memory]),
            media=_FakeSearchRows(semantic=[media]),
            chunks=_FakeSearchRows(),
            tz="America/Los_Angeles",
        )
        assert (
            f"- [memory:{memory['memory_id']}] [fact] [Oct 9, 2026 at 9:11 PM PDT] [match 0.90] lives in Leeds"
            in result
        )
        # the file by its media id, the id the media tools take, as stored
        assert f"- [media:{media['media_id']}] a photo of the house (created: Oct 9, 2026 at 9:11 PM PDT)" in result
        assert str(media["content_id"]) not in result
        assert "Oct 10" not in result, "no date is left in UTC beside the zoned ones"

    async def test_a_file_found_twice_is_listed_once_by_its_media_id(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        """a file's description and its text are two rows; the agent is handed one id for the file."""
        description = _media_row("a lease, two pages")
        text = {**_media_row("The tenant shall pay rent monthly."), "media_id": description["media_id"]}
        other = _media_row("a photo of the house")
        result = await _search(
            permissive_memory_authorizer,
            memories=_FakeSearchRows(),
            media=_FakeSearchRows(semantic=[description, text, other]),
            chunks=_FakeSearchRows(),
        )
        assert "Found 2 files:" in result
        assert result.count(f"[media:{description['media_id']}]") == 1
        assert f"[media:{other['media_id']}]" in result

    async def test_with_no_zone_named_a_date_reads_as_it_did(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        late = datetime(2026, 10, 10, 4, 11, tzinfo=timezone.utc)
        result = await _search(
            permissive_memory_authorizer,
            memories=_FakeSearchRows(semantic=[_memory_row("lives in Leeds", late)]),
            media=_FakeSearchRows(),
            chunks=_FakeSearchRows(),
        )
        assert "[fact] [Oct 10, 2026 at 4:11 AM] [match 0.90] lives in Leeds" in result

    async def test_a_zone_that_does_not_exist_is_refused_when_the_tool_is_built(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        with pytest.raises(ValueError, match="is not a time zone"):
            await _search_by_alias(permissive_memory_authorizer, _FakeAliasMemories(), tz="Mars/Olympus")


# -- A search that lost a branch says so before anything it found ---------------


# parity-exempt: answers only the two searches memory_search's query path makes on each collection; the SQL behind them is covered by the collection tests
class _FakeSearchRows:
    """a collection whose semantic and full-text searches return fixed rows, or raise."""

    def __init__(
        self, semantic: list[dict[str, Any]] | None = None, fts: list[dict[str, Any]] | Exception | None = None
    ) -> None:
        self.semantic = semantic or []
        self.fts = fts if fts is not None else []

    async def search_by_semantic(self, **kwargs: Any) -> list[dict[str, Any]]:
        _ = kwargs
        return self.semantic

    async def search_by_fts(self, **kwargs: Any) -> list[dict[str, Any]]:
        _ = kwargs
        if isinstance(self.fts, Exception):
            raise self.fts
        return self.fts


async def _search(
    authorizer: MemoryAuthorizerDependencies,
    *,
    memories: _FakeSearchRows,
    media: _FakeSearchRows,
    chunks: _FakeSearchRows,
    **loader_kwargs: Any,
) -> str:
    from langchain_core.embeddings import DeterministicFakeEmbedding

    (tool,) = await load_memory_search_tool(
        uuid4(),
        DeterministicFakeEmbedding(size=8),
        uuid4(),
        uuid4(),
        authorizer,
        memories,  # type: ignore[arg-type]
        media,  # type: ignore[arg-type]
        chunks,  # type: ignore[arg-type]
        **loader_kwargs,
    )
    result = await tool.ainvoke({"query": "where the person lives"})
    assert isinstance(result, str)
    return result


def _memory_row(content: str, date_created: Any = None) -> dict[str, Any]:
    return {"memory_id": uuid4(), "type": "fact", "content": content, "date_created": date_created, "similarity": 0.9}


def _media_row(content: str, date_created: Any = None) -> dict[str, Any]:
    return {
        "content_id": uuid4(),
        "content": content,
        "media_id": uuid4(),
        "media_category": "image",
        "metadata_json": None,
        "date_created": date_created,
        "similarity": 0.8,
    }


class TestPartialResultsAreSaidFirst:
    async def test_the_note_is_the_first_line_and_everything_found_follows_it(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        result = await _search(
            permissive_memory_authorizer,
            memories=_FakeSearchRows(semantic=[_memory_row("lives in Leeds")], fts=RuntimeError("fts down")),
            media=_FakeSearchRows(semantic=[_media_row("a photo of the house")]),
            chunks=_FakeSearchRows(),
        )
        lines = result.splitlines()
        assert lines[0] == "(Partial results: memory full-text search failed, so more may exist.)"
        assert lines[1] == "" and lines[2] == "Found 1 relevant memories:"
        assert "lives in Leeds" in result and "a photo of the house" in result
        assert result.count("Partial results") == 1 and not result.rstrip().endswith(")"), "no note after the body"

    async def test_a_search_that_lost_nothing_starts_with_what_it_found(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        result = await _search(
            permissive_memory_authorizer,
            memories=_FakeSearchRows(),
            media=_FakeSearchRows(semantic=[_media_row("a photo of the house")]),
            chunks=_FakeSearchRows(),
        )
        assert result.startswith("Found 1 files:\n- [media:"), "no blank line before the first section"
        assert "Partial results" not in result

    async def test_sections_are_parted_by_one_blank_line(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        result = await _search(
            permissive_memory_authorizer,
            memories=_FakeSearchRows(semantic=[_memory_row("lives in Leeds")]),
            media=_FakeSearchRows(semantic=[_media_row("a photo of the house")]),
            chunks=_FakeSearchRows(),
        )
        first, second = result.split("\n\n")
        assert first.startswith("Found 1 relevant memories:") and second.startswith("Found 1 files:")


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

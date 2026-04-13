"""tests for streaming chunk merging and tool call recovery utilities."""

from __future__ import annotations

from threetears.models.messages import ToolCallRequest
from threetears.models.results import ChatChunk, ChatResult
from threetears.models.streaming import (
    merge_chunks,
    recover_invalid_tool_calls,
    recover_split_tool_calls,
)


class TestMergeChunks:
    """tests for merge_chunks function."""

    def test_empty_chunks_returns_empty_result(self) -> None:
        """empty list returns ChatResult with empty content."""
        result = merge_chunks([])
        assert isinstance(result, ChatResult)
        assert result.content == ""

    def test_single_chunk(self) -> None:
        """single chunk produces result with that content."""
        chunks = [ChatChunk(content="hello")]
        result = merge_chunks(chunks)
        assert result.content == "hello"

    def test_multiple_chunks_concatenated(self) -> None:
        """multiple chunks have content concatenated."""
        chunks = [
            ChatChunk(content="hel"),
            ChatChunk(content="lo "),
            ChatChunk(content="world"),
        ]
        result = merge_chunks(chunks)
        assert result.content == "hello world"

    def test_tool_calls_merged(self) -> None:
        """chunks with tool_calls have all tool_calls in result."""
        tc1 = ToolCallRequest(id="tc-1", name="search", args={"q": "a"})
        tc2 = ToolCallRequest(id="tc-2", name="fetch", args={"url": "b"})
        chunks = [
            ChatChunk(content="", tool_calls=[tc1]),
            ChatChunk(content="", tool_calls=[tc2]),
        ]
        result = merge_chunks(chunks)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[1].name == "fetch"

    def test_tool_calls_none_ignored(self) -> None:
        """chunks with None tool_calls produce None in result."""
        chunks = [
            ChatChunk(content="hello"),
            ChatChunk(content=" world"),
        ]
        result = merge_chunks(chunks)
        assert result.tool_calls is None

    def test_mixed_tool_calls(self) -> None:
        """mix of chunks with and without tool_calls merges only non-None."""
        tc = ToolCallRequest(id="tc-1", name="search")
        chunks = [
            ChatChunk(content="thinking"),
            ChatChunk(content="", tool_calls=[tc]),
            ChatChunk(content=" done"),
        ]
        result = merge_chunks(chunks)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"

    def test_content_accumulation_with_finish_reason_chunks(self) -> None:
        """chunks with finish_reason still have content accumulated correctly."""
        chunks = [
            ChatChunk(content="a"),
            ChatChunk(content="b", finish_reason="length"),
            ChatChunk(content="c", finish_reason="stop"),
        ]
        result = merge_chunks(chunks)
        assert result.content == "abc"

    def test_model_is_empty_string(self) -> None:
        """result model field is empty string."""
        result = merge_chunks([ChatChunk(content="hi")])
        assert result.model == ""

    def test_usage_is_none(self) -> None:
        """result usage field is None."""
        result = merge_chunks([ChatChunk(content="hi")])
        assert result.usage is None


class TestRecoverSplitToolCalls:
    """tests for recover_split_tool_calls function."""

    def test_empty_list(self) -> None:
        """empty list returns empty list."""
        result = recover_split_tool_calls([])
        assert result == []

    def test_single_call_unchanged(self) -> None:
        """single normal call is returned as-is."""
        tc = ToolCallRequest(id="tc-1", name="search", args={"q": "test"})
        result = recover_split_tool_calls([tc])
        assert len(result) == 1
        assert result[0].id == "tc-1"
        assert result[0].name == "search"
        assert result[0].args == {"q": "test"}

    def test_normal_calls_unchanged(self) -> None:
        """two independent calls are both preserved."""
        calls = [
            ToolCallRequest(id="tc-1", name="search", args={"q": "a"}),
            ToolCallRequest(id="tc-2", name="fetch", args={"url": "b"}),
        ]
        result = recover_split_tool_calls(calls)
        assert len(result) == 2
        assert result[0].name == "search"
        assert result[1].name == "fetch"

    def test_split_call_merged(self) -> None:
        """name+empty_args followed by empty_name+args merges into one."""
        calls = [
            ToolCallRequest(id="tc-1", name="search", args={}),
            ToolCallRequest(id="tc-1b", name="", args={"q": "test"}),
        ]
        result = recover_split_tool_calls(calls)
        assert len(result) == 1
        assert result[0].id == "tc-1"
        assert result[0].name == "search"
        assert result[0].args == {"q": "test"}

    def test_multiple_split_calls(self) -> None:
        """two split pairs produce two merged calls."""
        calls = [
            ToolCallRequest(id="tc-1", name="search", args={}),
            ToolCallRequest(id="tc-1b", name="", args={"q": "a"}),
            ToolCallRequest(id="tc-2", name="fetch", args={}),
            ToolCallRequest(id="tc-2b", name="", args={"url": "b"}),
        ]
        result = recover_split_tool_calls(calls)
        assert len(result) == 2
        assert result[0].name == "search"
        assert result[0].args == {"q": "a"}
        assert result[1].name == "fetch"
        assert result[1].args == {"url": "b"}

    def test_split_followed_by_normal(self) -> None:
        """split pair followed by normal call produces merged + normal."""
        calls = [
            ToolCallRequest(id="tc-1", name="search", args={}),
            ToolCallRequest(id="tc-1b", name="", args={"q": "a"}),
            ToolCallRequest(id="tc-3", name="log", args={"msg": "done"}),
        ]
        result = recover_split_tool_calls(calls)
        assert len(result) == 2
        assert result[0].name == "search"
        assert result[0].args == {"q": "a"}
        assert result[1].name == "log"
        assert result[1].args == {"msg": "done"}

    def test_no_merge_when_first_has_args(self) -> None:
        """both entries having args prevents merging."""
        calls = [
            ToolCallRequest(id="tc-1", name="search", args={"q": "a"}),
            ToolCallRequest(id="tc-2", name="", args={"url": "b"}),
        ]
        result = recover_split_tool_calls(calls)
        assert len(result) == 2

    def test_no_merge_when_second_has_name(self) -> None:
        """both entries having names prevents merging."""
        calls = [
            ToolCallRequest(id="tc-1", name="search", args={}),
            ToolCallRequest(id="tc-2", name="fetch", args={"url": "b"}),
        ]
        result = recover_split_tool_calls(calls)
        assert len(result) == 2

    def test_input_not_mutated(self) -> None:
        """original input list is not mutated."""
        calls = [
            ToolCallRequest(id="tc-1", name="search", args={}),
            ToolCallRequest(id="tc-1b", name="", args={"q": "test"}),
        ]
        original_len = len(calls)
        recover_split_tool_calls(calls)
        assert len(calls) == original_len
        assert calls[0].name == "search"
        assert calls[1].name == ""

    def test_preserves_id_from_first(self) -> None:
        """merged call uses id from first entry."""
        calls = [
            ToolCallRequest(id="first-id", name="tool", args={}),
            ToolCallRequest(id="second-id", name="", args={"k": "v"}),
        ]
        result = recover_split_tool_calls(calls)
        assert len(result) == 1
        assert result[0].id == "first-id"


class TestRecoverInvalidToolCalls:
    """tests for recover_invalid_tool_calls function."""

    def test_empty_list(self) -> None:
        """empty list returns empty tuple."""
        recovered, invalid = recover_invalid_tool_calls([])
        assert recovered == []
        assert invalid == []

    def test_valid_json_recovered(self) -> None:
        """parseable args creates ToolCallRequest."""
        calls = [{"id": "tc-1", "name": "search", "args": '{"q": "test"}'}]
        recovered, invalid = recover_invalid_tool_calls(calls)
        assert len(recovered) == 1
        assert len(invalid) == 0
        assert recovered[0].id == "tc-1"
        assert recovered[0].name == "search"
        assert recovered[0].args == {"q": "test"}

    def test_invalid_json_stays_invalid(self) -> None:
        """unparseable args stays in invalid list."""
        calls = [{"id": "tc-1", "name": "search", "args": "{bad json"}]
        recovered, invalid = recover_invalid_tool_calls(calls)
        assert len(recovered) == 0
        assert len(invalid) == 1
        assert invalid[0]["id"] == "tc-1"

    def test_mixed_recovery(self) -> None:
        """some valid and some invalid are correctly split."""
        calls = [
            {"id": "tc-1", "name": "search", "args": '{"q": "test"}'},
            {"id": "tc-2", "name": "fetch", "args": "not json at all"},
            {"id": "tc-3", "name": "log", "args": '{"msg": "ok"}'},
        ]
        recovered, invalid = recover_invalid_tool_calls(calls)
        assert len(recovered) == 2
        assert len(invalid) == 1
        assert recovered[0].name == "search"
        assert recovered[1].name == "log"
        assert invalid[0]["id"] == "tc-2"

    def test_non_dict_json_stays_invalid(self) -> None:
        """args that parse to list or string stay in invalid list."""
        calls = [
            {"id": "tc-1", "name": "search", "args": '["not", "a", "dict"]'},
            {"id": "tc-2", "name": "fetch", "args": '"just a string"'},
        ]
        recovered, invalid = recover_invalid_tool_calls(calls)
        assert len(recovered) == 0
        assert len(invalid) == 2

    def test_missing_keys_handled_gracefully(self) -> None:
        """dicts missing expected keys do not crash."""
        calls = [
            {"id": "tc-1", "name": "search"},  # missing "args"
            {"name": "fetch", "args": '{"url": "b"}'},  # missing "id"
            {"args": '{"q": "test"}'},  # missing "id" and "name"
        ]
        recovered, invalid = recover_invalid_tool_calls(calls)
        # missing args → empty string → JSONDecodeError → invalid
        # missing id → defaults to "" → should recover if args valid
        assert len(recovered) + len(invalid) == 3

    def test_empty_string_args_stays_invalid(self) -> None:
        """empty string args stays in invalid list."""
        calls = [{"id": "tc-1", "name": "search", "args": ""}]
        recovered, invalid = recover_invalid_tool_calls(calls)
        assert len(recovered) == 0
        assert len(invalid) == 1

    def test_nested_json_args_recovered(self) -> None:
        """complex nested args are properly recovered."""
        nested_args = '{"filters": {"status": "active", "tags": ["a", "b"]}, "limit": 10}'
        calls = [{"id": "tc-1", "name": "query", "args": nested_args}]
        recovered, invalid = recover_invalid_tool_calls(calls)
        assert len(recovered) == 1
        assert len(invalid) == 0
        assert recovered[0].args["filters"]["status"] == "active"
        assert recovered[0].args["filters"]["tags"] == ["a", "b"]
        assert recovered[0].args["limit"] == 10

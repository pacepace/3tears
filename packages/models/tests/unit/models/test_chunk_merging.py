"""tests for :func:`threetears.models.chunk_merging.merge_chunks`."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.ai import UsageMetadata

from threetears.models.chunk_merging import merge_chunks


class TestMergeChunks:
    """tests for the streamed-chunk merge."""

    def test_empty_list_raises(self) -> None:
        """an empty chunk list is a caller bug; merge raises ``ValueError``.

        Every observed call site streams at least one chunk before
        invoking the merge. An empty list signals a control-flow bug
        upstream (e.g. an aborted stream that the caller treated as a
        normal completion). Surface it as a typed error rather than
        returning an empty ``AIMessage`` that would be persisted as a
        silent failure.
        """
        with pytest.raises(ValueError, match="empty chunk list"):
            merge_chunks([])

    def test_single_chunk_text_only(self) -> None:
        """a single text-only chunk merges into the equivalent ``AIMessage``."""
        chunk = AIMessageChunk(content="hello world")
        result = merge_chunks([chunk])
        assert isinstance(result, AIMessage)
        assert not isinstance(result, AIMessageChunk)
        assert result.content == "hello world"
        assert result.tool_calls == []
        assert result.invalid_tool_calls == []

    def test_multi_chunk_text_concatenates(self) -> None:
        """multiple text-only chunks concatenate content in order."""
        chunks = [
            AIMessageChunk(content="hello "),
            AIMessageChunk(content="world"),
            AIMessageChunk(content="!"),
        ]
        result = merge_chunks(chunks)
        assert result.content == "hello world!"

    def test_tool_call_chunks_accumulate_into_tool_calls(self) -> None:
        """``tool_call_chunks`` across chunks accumulate by index into one tool call.

        LangChain's ``AIMessageChunk.__add__`` merges per-index
        ``tool_call_chunks``: the first chunk carries ``name`` + the
        opening of ``args``, subsequent chunks carry the rest of
        ``args``. Once the JSON closes, the merged chunk parses into a
        single entry in ``tool_calls`` on the finalized message.
        """
        chunks = [
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "calculator",
                        "args": '{"expression":',
                        "id": "call_1",
                        "index": 0,
                    }
                ],
            ),
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": None,
                        "args": ' "2+2"}',
                        "id": None,
                        "index": 0,
                    }
                ],
            ),
        ]
        result = merge_chunks(chunks)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "calculator"
        assert result.tool_calls[0]["args"] == {"expression": "2+2"}
        assert result.tool_calls[0]["id"] == "call_1"

    def test_reasoning_content_preserved_in_additional_kwargs(self) -> None:
        """per-chunk ``additional_kwargs["reasoning_content"]`` concatenates across chunks.

        OpenRouter-routed reasoning models (deepseek-r1 and friends)
        surface their reasoning trace as ``reasoning_content`` in
        ``additional_kwargs`` per chunk. LangChain's chunk-merge
        concatenates the string values across chunks, so consumers
        that inspect ``additional_kwargs["reasoning_content"]`` on the
        finalized message see the full reasoning trace.
        """
        chunks = [
            AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": "Let me think... "},
            ),
            AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": "the answer is 4."},
            ),
            AIMessageChunk(content="4"),
        ]
        result = merge_chunks(chunks)
        assert result.content == "4"
        assert result.additional_kwargs.get("reasoning_content") == "Let me think... the answer is 4."

    def test_usage_metadata_sums_across_chunks(self) -> None:
        """``usage_metadata`` sums per-field across chunks (LangChain merge contract).

        Streaming providers commonly emit per-chunk token counts that
        the framework sums into a final ``usage_metadata`` on the
        merged chunk. ``merge_chunks`` preserves that sum on the
        finalized message so :class:`UsageTracker` callbacks see the
        complete count.
        """
        chunks = [
            AIMessageChunk(
                content="hi ",
                usage_metadata=UsageMetadata(
                    input_tokens=10,
                    output_tokens=2,
                    total_tokens=12,
                ),
            ),
            AIMessageChunk(
                content="there",
                usage_metadata=UsageMetadata(
                    input_tokens=0,
                    output_tokens=3,
                    total_tokens=3,
                ),
            ),
        ]
        result = merge_chunks(chunks)
        assert result.content == "hi there"
        assert result.usage_metadata is not None
        assert result.usage_metadata["output_tokens"] == 5
        assert result.usage_metadata["total_tokens"] == 15

    def test_response_metadata_from_chunks_preserved(self) -> None:
        """``response_metadata`` from the chunks is preserved on the merged message.

        Some providers populate ``response_metadata`` mid-stream
        (model id, finish reason, stop sequence). LangChain's
        chunk-merge deep-merges that dict; ``merge_chunks`` keeps the
        merged dict on the finalized message so consumers can inspect
        e.g. ``finish_reason`` after the stream closes.
        """
        chunks = [
            AIMessageChunk(content="hi", response_metadata={"model_name": "gpt-test"}),
            AIMessageChunk(content="", response_metadata={"finish_reason": "stop"}),
        ]
        result = merge_chunks(chunks)
        assert result.response_metadata.get("model_name") == "gpt-test"
        assert result.response_metadata.get("finish_reason") == "stop"

    def test_invalid_tool_calls_preserved(self) -> None:
        """``invalid_tool_calls`` survives the chunk -> message finalization.

        Consumers (the consumer-side tool router, 3tears-agents' dispatch)
        inspect ``invalid_tool_calls`` to attempt JSON-repair on
        malformed streaming tool calls. The merge must keep that
        field so the recovery code can run.
        """
        chunks = [
            AIMessageChunk(
                content="",
                invalid_tool_calls=[
                    {
                        "name": "calculator",
                        "args": "{partial",
                        "id": "call_1",
                        "error": "JSONDecodeError",
                    }
                ],
            ),
        ]
        result = merge_chunks(chunks)
        assert len(result.invalid_tool_calls) == 1
        assert result.invalid_tool_calls[0]["name"] == "calculator"
        assert result.invalid_tool_calls[0]["error"] == "JSONDecodeError"

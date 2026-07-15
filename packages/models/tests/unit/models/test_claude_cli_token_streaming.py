"""Tests for real token-level streaming on the subscription backend (streaming follow-up to
claude-max-convergence Chunk 9; see
``.prawduct/artifacts/3tears-change-claude-max-token-streaming.md`` in metallm for the sign-off).

``ClaudeCodeChatModel._astream`` sets ``include_partial_messages=True`` but never reads the
``StreamEvent`` deltas the SDK subprocess then emits -- a turn arrives as one or two large lumps
instead of a real token stream. ``_SubscriptionChatModel._astream`` (overridden in
``_claude_cli.py``) consumes those deltas and yields each one immediately.

These tests mock ``claude_agent_sdk.ClaudeSDKClient`` (the SDK subprocess boundary) and construct
REAL ``StreamEvent``/``AssistantMessage``/``ResultMessage`` instances so the method's own
``isinstance`` branching is exercised exactly as production code sees it -- no LLM call, no real
subprocess.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import AssistantMessage, ResultMessage, StreamEvent, TextBlock, ToolUseBlock

from threetears.models import DEFAULT_CHAT_MODEL
from threetears.models.providers._claude_cli import create_subscription_chat


def _result_message(**overrides: Any) -> ResultMessage:
    defaults: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 100,
        "duration_api_ms": 90,
        "is_error": False,
        "num_turns": 1,
        "session_id": "sess-1",
    }
    defaults.update(overrides)
    return ResultMessage(**defaults)


def _text_delta_event(index: int, text: str) -> StreamEvent:
    return StreamEvent(
        uuid="evt-1",
        session_id="sess-1",
        event={
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        },
    )


# parity-exempt: deliberately narrow -- only __aenter__/__aexit__/query/receive_response, the sole surface _astream calls; ClaudeSDKClient's other public methods (interrupt, rewind_files, set_model, MCP server control, ...) are unrelated to the text-delta streaming behavior under test
class _FakeSDKClient:
    """Async-context-manager stand-in for ``ClaudeSDKClient``: replays a fixed message sequence."""

    def __init__(self, messages: list[Any], *, options: Any = None) -> None:
        self._messages = messages
        self.queried_prompt: str | None = None

    async def __aenter__(self) -> "_FakeSDKClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def query(self, prompt: str) -> None:
        self.queried_prompt = prompt

    async def receive_response(self):
        for msg in self._messages:
            yield msg


def _model_streaming(messages: list[Any]):
    """Build a subscription chat model wired to replay ``messages`` from the fake SDK client."""

    def _client_factory(*, options: Any = None) -> _FakeSDKClient:
        return _FakeSDKClient(messages, options=options)

    return patch("claude_agent_sdk.ClaudeSDKClient", side_effect=_client_factory)


async def _collect_chunks(messages: list[Any]) -> list[Any]:
    # ``create_subscription_chat`` MUST run inside the patch context: it calls
    # ``_subscription_model_cls()``, which does ``from claude_agent_sdk import
    # ClaudeSDKClient`` -- a module-level import whose result is captured as a
    # CLOSURE variable in ``_astream`` at THAT moment. Patching
    # ``claude_agent_sdk.ClaudeSDKClient`` after that import already ran has no
    # effect on the already-bound closure -- the real class stays in use, and
    # ``_astream`` would launch a REAL Claude Code CLI subprocess call.
    with _model_streaming(messages):
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        chunks = [
            chunk
            async for chunk in model._astream(  # noqa: SLF001 -- the method under test
                [HumanMessage(content="hi")], run_manager=None,
            )
        ]
    return chunks


class TestTokenLevelStreaming:
    async def test_stream_event_deltas_yield_incrementally(self) -> None:
        """Each StreamEvent text delta arrives as its own chunk, in order."""
        messages = [
            _text_delta_event(0, "Hello "),
            _text_delta_event(0, "world"),
            AssistantMessage(content=[TextBlock(text="Hello world")], model=DEFAULT_CHAT_MODEL),
            _result_message(),
        ]
        chunks = await _collect_chunks(messages)

        text_chunks = [c for c in chunks if c.message.content]
        assert [c.message.content for c in text_chunks] == ["Hello ", "world"]

    async def test_assistant_message_text_not_doubled_after_streaming(self) -> None:
        """The terminal AssistantMessage does not re-yield text already streamed via deltas."""
        messages = [
            _text_delta_event(0, "Hello "),
            _text_delta_event(0, "world"),
            AssistantMessage(content=[TextBlock(text="Hello world")], model=DEFAULT_CHAT_MODEL),
            _result_message(),
        ]
        chunks = await _collect_chunks(messages)

        combined = "".join(c.message.content for c in chunks)
        assert combined == "Hello world"

    async def test_no_stream_events_falls_back_to_whole_message(self) -> None:
        """A block with zero StreamEvent deltas still gets its text emitted (defensive fallback)."""
        messages = [
            AssistantMessage(content=[TextBlock(text="No deltas here")], model=DEFAULT_CHAT_MODEL),
            _result_message(),
        ]
        chunks = await _collect_chunks(messages)

        text_chunks = [c for c in chunks if c.message.content]
        assert len(text_chunks) == 1
        assert text_chunks[0].message.content == "No deltas here"

    async def test_streamed_flag_resets_between_assistant_messages(self) -> None:
        """A second AssistantMessage in the same turn (post-tool-call) gets its own
        fallback check -- streaming the first message's text must not suppress the
        second message's whole-text fallback if IT has no deltas of its own."""
        messages = [
            _text_delta_event(0, "First, streamed."),
            AssistantMessage(
                content=[TextBlock(text="First, streamed.")], model=DEFAULT_CHAT_MODEL,
            ),
            AssistantMessage(
                content=[TextBlock(text="Second, not streamed.")], model=DEFAULT_CHAT_MODEL,
            ),
            _result_message(),
        ]
        chunks = await _collect_chunks(messages)

        combined = "".join(c.message.content for c in chunks)
        assert combined == "First, streamed.Second, not streamed."

    async def test_non_text_delta_events_are_ignored(self) -> None:
        """StreamEvents that aren't text deltas (e.g. input_json_delta for a tool call
        in progress) are skipped, not misread as text."""
        messages = [
            StreamEvent(
                uuid="evt-2",
                session_id="sess-1",
                event={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"x":'},
                },
            ),
            AssistantMessage(
                content=[ToolUseBlock(id="tu-1", name="echo", input={"x": 1})],
                model=DEFAULT_CHAT_MODEL,
            ),
            _result_message(),
        ]
        chunks = await _collect_chunks(messages)

        assert all(c.message.content == "" for c in chunks)

    async def test_tool_calls_and_results_still_flow_into_generation_info(self) -> None:
        """Tool-call extraction is unaffected by the streaming change."""
        messages = [
            AssistantMessage(
                content=[TextBlock(text="calling a tool"), ToolUseBlock(id="tu-1", name="echo", input={"x": 1})],
                model=DEFAULT_CHAT_MODEL,
            ),
            _result_message(),
        ]
        chunks = await _collect_chunks(messages)

        final = chunks[-1]
        assert final.generation_info["internal_tool_calls"] == [
            {"id": "tu-1", "name": "echo", "args": {"x": 1}},
        ]

    async def test_result_message_still_yields_the_terminal_chunk(self) -> None:
        """The ResultMessage -> final chunk_position='last' chunk is unaffected."""
        messages = [
            AssistantMessage(content=[TextBlock(text="hi")], model=DEFAULT_CHAT_MODEL),
            _result_message(total_cost_usd=0.01, session_id="sess-42"),
        ]
        chunks = await _collect_chunks(messages)

        final = chunks[-1]
        assert final.generation_info["session_id"] == "sess-42"
        assert final.generation_info["total_cost_usd"] == 0.01
        assert final.generation_info["finish_reason"] == "stop"

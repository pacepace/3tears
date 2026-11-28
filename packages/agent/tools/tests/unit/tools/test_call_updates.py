"""the pod's publisher of a mesh call's updates: in order, bounded, and never in the tool's way.

the publisher's only collaborator is a publish callable, so each test hands it one that records, blocks or fails.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from threetears.agent.tools.call_updates import CallUpdatePublisher, mesh_call_context
from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.core.tool_wire import CallUpdates
from threetears.mcp import CallContext
from threetears.nats import MAX_UPDATE_BYTES, PublishError, Subject, UpdateBudget

_SUBJECT = "test.tools.progress.pod-1.call-1"


class _Recorder:
    """a publish that records what it is given, and can be made to block or fail."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.gate: asyncio.Event | None = None
        self.fail_seqs: set[int] = set()

    async def __call__(self, subject: Subject, payload: bytes) -> None:
        assert subject.path == _SUBJECT
        update = json.loads(payload)
        if self.gate is not None:
            await self.gate.wait()
        if update["seq"] in self.fail_seqs:
            raise PublishError("the bus said no")
        self.sent.append(update)


def _publisher(publish: _Recorder, *, budget: UpdateBudget | None = None) -> CallUpdatePublisher:
    return CallUpdatePublisher(publish, CallUpdates(subject=_SUBJECT, progress_token=1), tool_key="t@1", budget=budget)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TestPublishing:
    @pytest.mark.asyncio
    async def test_updates_leave_in_order_numbered_from_one(self) -> None:
        publish = _Recorder()
        publisher = _publisher(publish)

        await publisher.progress(1, 2, "one")
        await asyncio.sleep(0)
        await publisher.log("info", {"a": 1}, "builder")
        await asyncio.sleep(0)
        await publisher.progress(2, 2, "two")
        await publisher.close()

        assert [(u["seq"], u["kind"]) for u in publish.sent] == [(1, "progress"), (2, "log"), (3, "progress")]
        assert publish.sent[1]["data"] == {"a": 1} and publish.sent[1]["logger"] == "builder"

    @pytest.mark.asyncio
    async def test_a_report_returns_at_once_while_the_link_is_blocked(self) -> None:
        publish = _Recorder()
        publish.gate = asyncio.Event()
        publisher = _publisher(publish)

        async with asyncio.timeout(1):
            for step in range(1, 200):
                await publisher.progress(step, None, None)
                await publisher.log("info", step, None)
        assert publish.sent == []

        publish.gate.set()
        await publisher.close()
        assert publish.sent  # what was pending went once the link moved

    @pytest.mark.asyncio
    async def test_progress_waiting_for_budget_coalesces_to_the_latest(self) -> None:
        publish = _Recorder()
        clock = _Clock()
        publisher = _publisher(publish, budget=UpdateBudget(rate=1.0, burst=1, clock=clock))

        await publisher.progress(1, None, None)
        await asyncio.sleep(0)  # the budget's one token: 1 leaves
        for step in range(2, 10):
            await publisher.progress(step, None, None)
        clock.now = 5.0  # the budget refills
        await publisher.close()

        assert [u["progress"] for u in publish.sent] == [1.0, 9.0]

    @pytest.mark.asyncio
    async def test_past_its_bound_the_oldest_pending_entry_is_dropped(self) -> None:
        publish = _Recorder()
        publish.gate = asyncio.Event()
        # a budget that never binds, so only the pending bound decides what is kept
        publisher = _publisher(publish, budget=UpdateBudget(rate=1e6, burst=1000))

        for entry in range(100):
            await publisher.log("info", entry, None)
        publish.gate.set()
        await publisher.close()

        # all 100 queued before the sender first ran: the newest 64 were kept, in order
        assert [u["data"] for u in publish.sent] == list(range(36, 100))

    @pytest.mark.asyncio
    async def test_a_failed_publish_loses_that_update_only(self) -> None:
        publish = _Recorder()
        publish.fail_seqs = {2}
        publisher = _publisher(publish)

        for entry in range(3):
            await publisher.log("info", entry, None)
            await asyncio.sleep(0)
        await publisher.close()

        assert [u["seq"] for u in publish.sent] == [1, 3]

    @pytest.mark.asyncio
    async def test_an_oversized_log_entry_is_sent_as_a_note_of_its_size(self) -> None:
        publish = _Recorder()
        publisher = _publisher(publish)

        await publisher.log("warning", "x" * (MAX_UPDATE_BYTES + 1), None)
        await publisher.close()

        [update] = publish.sent
        assert update["level"] == "warning"
        assert "bytes" in update["data"]["not_sent"]

    @pytest.mark.asyncio
    async def test_data_that_is_no_json_value_is_sent_as_a_note(self) -> None:
        publish = _Recorder()
        publisher = _publisher(publish)

        await publisher.log("info", object(), None)
        await publisher.close()

        [update] = publish.sent
        assert "no JSON value" in update["data"]["not_sent"]

    @pytest.mark.asyncio
    async def test_closing_waits_a_bounded_time_on_a_dead_link(self) -> None:
        publish = _Recorder()
        publish.gate = asyncio.Event()  # never set
        publisher = _publisher(publish)
        await publisher.progress(1, None, None)

        async with asyncio.timeout(2):
            await publisher.close(timeout=0.05)

        assert publish.sent == []
        # an update after the answer is not sent
        await publisher.progress(2, None, None)
        assert publish.sent == []


class _WithContext(TearsTool):
    async def execute(self, *, ctx: CallContext | None = None, **kwargs: Any) -> ToolResult:
        del ctx, kwargs
        return ToolResult(success=True, content="")

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(name="t.x", version="1", description="x", input_schema={"type": "object"})

    def mcp_name(self) -> str:
        return "t.x"

    def mcp_version(self) -> str:
        return "1"


class _Refused(_WithContext):
    async def execute(self, *, ctx: list[CallContext] | None = None, **kwargs: Any) -> ToolResult:  # type: ignore[override]
        del ctx, kwargs
        return ToolResult(success=True, content="")


class _Without(_WithContext):
    async def execute(self, **kwargs: Any) -> ToolResult:  # type: ignore[override]
        del kwargs
        return ToolResult(success=True, content="")


def _passed(tool: TearsTool, updates: CallUpdates) -> tuple[str, Any] | None:
    return mesh_call_context(tool, updates, _publisher(_Recorder()), asyncio.Event())


class TestTheContextPassedToExecute:
    @pytest.mark.asyncio
    async def test_an_execute_that_declares_one_gets_the_callers_token_and_levels(self) -> None:
        passed = _passed(_WithContext(), CallUpdates(subject=_SUBJECT, progress_token="tok", log_level="error"))

        assert passed is not None
        parameter, ctx = passed
        assert parameter == "ctx"
        assert isinstance(ctx, CallContext)
        assert ctx.progress_token == "tok"
        assert ctx.log_levels == frozenset(("error", "critical", "alert", "emergency"))

    @pytest.mark.asyncio
    async def test_a_caller_that_takes_no_log_entries_gets_a_context_that_sends_none(self) -> None:
        passed = _passed(_WithContext(), CallUpdates(subject=_SUBJECT, progress_token=1))

        assert passed is not None
        assert passed[1].log_levels == frozenset()

    @pytest.mark.asyncio
    async def test_an_execute_without_one_gets_none(self) -> None:
        assert _passed(_Without(), CallUpdates(subject=_SUBJECT, progress_token=1)) is None

    @pytest.mark.asyncio
    async def test_a_form_the_mcp_face_refuses_gets_none_and_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        assert _passed(_Refused(), CallUpdates(subject=_SUBJECT, progress_token=1)) is None
        assert any("form the MCP face refuses" in record.getMessage() for record in caplog.records)

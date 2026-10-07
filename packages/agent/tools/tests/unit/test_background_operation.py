"""A long operation started by one tool call and reported by another: one at a time, its outcome kept."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from threetears.agent.tools.background_operation import (
    BackgroundOperation,
    OperationStatusTool,
    StartOperationTool,
)
from threetears.agent.tools.base_tool import CONFLICT


@dataclass
class _Gate:
    """an operation body the test releases by hand."""

    release: asyncio.Event
    calls: int = 0
    fail_with: Exception | None = None

    async def __call__(self) -> dict[str, int]:
        self.calls += 1
        await self.release.wait()
        if self.fail_with is not None:
            raise self.fail_with
        return {"rows": 3}


def _gate() -> _Gate:
    return _Gate(release=asyncio.Event())


async def test_a_started_operation_runs_in_the_background_and_keeps_its_result() -> None:
    body = _gate()
    operation = BackgroundOperation("load", body)
    assert operation.state == "idle"
    assert operation.start()
    await asyncio.sleep(0)
    assert operation.running
    assert operation.state == "running"
    body.release.set()
    await operation.wait()
    assert not operation.running
    assert operation.state == "succeeded"
    assert operation.last is not None
    assert operation.last.result == {"rows": 3}
    assert operation.last.error is None
    assert operation.last.finished_at >= operation.last.started_at


async def test_a_second_start_while_one_runs_is_refused_and_runs_nothing() -> None:
    body = _gate()
    operation = BackgroundOperation("load", body)
    assert operation.start()
    assert not operation.start()
    body.release.set()
    await operation.wait()
    assert body.calls == 1


async def test_an_operation_that_raises_is_kept_as_failed_with_its_reason() -> None:
    body = _gate()
    body.fail_with = RuntimeError("table geos: 120 of 16320 rows written")
    operation = BackgroundOperation("load", body)
    operation.start()
    body.release.set()
    await operation.wait()
    assert operation.state == "failed"
    assert operation.last is not None
    assert operation.last.result is None
    assert operation.last.error == "RuntimeError: table geos: 120 of 16320 rows written"


async def test_a_failed_operation_can_be_started_again() -> None:
    body = _gate()
    body.fail_with = RuntimeError("boom")
    operation = BackgroundOperation("load", body)
    operation.start()
    body.release.set()
    await operation.wait()
    body.fail_with = None
    assert operation.start()
    await operation.wait()
    assert operation.state == "succeeded"


async def test_start_when_needed_retries_only_the_named_errors_then_starts_once() -> None:
    body = _gate()
    body.release.set()
    operation = BackgroundOperation("load", body)
    answers: list[bool | Exception] = [ConnectionError("no grant yet"), ConnectionError("no grant yet"), True]

    async def needed() -> bool:
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    operation.start_when_needed(needed, retry_on=(ConnectionError,), retry_seconds=0)
    await operation.wait_until_settled()
    assert answers == []
    assert body.calls == 1
    assert operation.state == "succeeded"


async def test_start_when_needed_starts_nothing_when_nothing_is_needed() -> None:
    body = _gate()
    operation = BackgroundOperation("load", body)

    async def needed() -> bool:
        return False

    operation.start_when_needed(needed, retry_on=(ConnectionError,), retry_seconds=0)
    await operation.wait_until_settled()
    assert body.calls == 0
    assert operation.state == "idle"


async def test_start_when_needed_does_not_retry_an_unnamed_error() -> None:
    body = _gate()
    operation = BackgroundOperation("load", body)

    async def needed() -> bool:
        raise ValueError("the check itself is broken")

    operation.start_when_needed(needed, retry_on=(ConnectionError,), retry_seconds=0)
    with pytest.raises(ValueError):
        await operation.wait_until_settled()
    assert body.calls == 0


async def test_stop_cancels_the_running_operation() -> None:
    body = _gate()
    operation = BackgroundOperation("load", body)
    operation.start()
    await asyncio.sleep(0)
    await operation.stop()
    assert not operation.running


def _tools(operation: BackgroundOperation[dict[str, int]] | None) -> tuple[StartOperationTool, OperationStatusTool]:
    start = StartOperationTool(
        name="enr.reload",
        description="Start a load.",
        operation=lambda: operation,
        status_tool="enr.load_status",
    )
    status = OperationStatusTool(name="enr.load_status", description="Report the load.", operation=lambda: operation)
    return start, status


async def test_the_start_tool_answers_at_once_and_the_status_tool_reports_running() -> None:
    body = _gate()
    operation = BackgroundOperation("load", body)
    start, status = _tools(operation)
    started = await start.run()
    assert started.success
    assert "enr.load_status" in started.content
    running = await status.run()
    assert running.success
    assert running.metadata is not None
    assert running.metadata["state"] == "running"
    assert running.metadata["running"] is True
    body.release.set()
    await operation.wait()
    done = await status.run()
    assert done.metadata is not None
    assert done.metadata["state"] == "succeeded"
    assert done.metadata["last"]["result"] == {"rows": 3}
    assert done.metadata["last"]["error"] is None


async def test_the_start_tool_refuses_a_second_start_as_a_conflict() -> None:
    body = _gate()
    operation = BackgroundOperation("load", body)
    start, _ = _tools(operation)
    await start.run()
    refused = await start.run()
    assert not refused.success
    assert refused.error_code == CONFLICT
    body.release.set()
    await operation.wait()
    assert body.calls == 1


async def test_the_status_tool_reports_a_failure_with_its_reason() -> None:
    body = _gate()
    body.fail_with = RuntimeError("part way")
    operation = BackgroundOperation("load", body)
    _, status = _tools(operation)
    operation.start()
    body.release.set()
    await operation.wait()
    failed = await status.run()
    assert failed.success
    assert failed.metadata is not None
    assert failed.metadata["state"] == "failed"
    assert failed.metadata["last"]["error"] == "RuntimeError: part way"
    assert "part way" in failed.content


async def test_both_tools_refuse_while_the_operation_is_not_built_yet() -> None:
    start, status = _tools(None)
    for tool in (start, status):
        result = await tool.run()
        assert not result.success
        assert result.error is not None
        assert "not ready" in result.error


def test_the_tools_are_named_and_versioned_as_given() -> None:
    start, status = _tools(None)
    assert start.mcp_name() == "enr.reload"
    assert status.mcp_name() == "enr.load_status"
    assert start.mcp_version() == status.mcp_version() == "1.0"
    assert start.mcp_schema().description == "Start a load."
    assert start.face_api and status.face_api

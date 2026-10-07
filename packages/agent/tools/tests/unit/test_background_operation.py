"""A long operation started by one tool call and reported by another: one at a time, its outcome kept."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from threetears.agent.tools.background_operation import (
    BackgroundOperation,
    OperationStatusTool,
    RequestOperationTool,
    StartOperationTool,
)
from threetears.agent.tools.base_tool import CONFLICT, TOOL_NOT_READY


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


async def test_an_unnamed_error_ending_the_wait_is_logged_even_when_nobody_waits_for_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    body = _gate()
    operation = BackgroundOperation("load", body)

    async def needed() -> bool:
        raise ValueError("the check itself is broken")

    operation.start_when_needed(needed, retry_on=(ConnectionError,), retry_seconds=0)
    await asyncio.sleep(0.01)
    # a later wait replaces the finished one, so its error can only be seen in the log
    operation.start_when_needed(needed, retry_on=(ConnectionError,), retry_seconds=0)
    await asyncio.sleep(0.01)
    failures = [
        record
        for record in caplog.records
        if record.levelname == "ERROR" and "the check itself is broken" in str(getattr(record, "extra_data", ""))
    ]
    assert len(failures) == 2
    assert body.calls == 0


async def test_a_second_start_when_needed_while_one_waits_is_refused_and_stop_still_ends_the_first() -> None:
    body = _gate()
    operation = BackgroundOperation("load", body)
    answer = asyncio.Event()
    cancelled: list[str] = []

    async def first() -> bool:
        try:
            await answer.wait()
        except asyncio.CancelledError:
            cancelled.append("first")
            raise
        return True

    async def second() -> bool:
        return True

    assert operation.start_when_needed(first, retry_on=(ConnectionError,), retry_seconds=0)
    assert not operation.start_when_needed(second, retry_on=(ConnectionError,), retry_seconds=0)
    await asyncio.sleep(0)
    await operation.stop()
    assert cancelled == ["first"]
    assert body.calls == 0


async def test_wait_until_settled_waits_for_the_first_start_when_needed() -> None:
    body = _gate()
    body.release.set()
    operation = BackgroundOperation("load", body)
    answer = asyncio.Event()

    async def first() -> bool:
        await answer.wait()
        return True

    async def never() -> bool:
        raise AssertionError("a second wait must not replace the first")

    operation.start_when_needed(first, retry_on=(ConnectionError,), retry_seconds=0)
    operation.start_when_needed(never, retry_on=(ConnectionError,), retry_seconds=0)
    settled = asyncio.create_task(operation.wait_until_settled())
    await asyncio.sleep(0)
    assert not settled.done()
    answer.set()
    await settled
    assert body.calls == 1
    assert operation.state == "succeeded"


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


async def test_the_status_tool_reports_the_progress_of_a_run_in_progress() -> None:
    body = _gate()
    operation = BackgroundOperation("load", body)
    done = {"states": 0}
    status = OperationStatusTool(
        name="enr.load_status",
        description="Report the load.",
        operation=lambda: operation,
        progress=lambda: {"summary": f"Loading · {done['states']} of 51 states", "states_done": done["states"]},
    )
    operation.start()
    done["states"] = 37

    running = await status.run()

    assert running.metadata is not None
    assert running.metadata["progress"] == {"summary": "Loading · 37 of 51 states", "states_done": 37}
    assert "Loading · 37 of 51 states" in running.content
    body.release.set()
    await operation.wait()
    assert (await status.run()).metadata["progress"]["states_done"] == 37  # type: ignore[index]


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
        assert result.error_code == TOOL_NOT_READY


def test_the_tools_are_named_and_versioned_as_given() -> None:
    start, status = _tools(None)
    assert start.mcp_name() == "enr.reload"
    assert status.mcp_name() == "enr.load_status"
    assert start.mcp_version() == status.mcp_version() == "1.0"
    assert start.mcp_schema().description == "Start a load."
    assert start.face_api and status.face_api


async def test_a_wait_that_fails_with_an_error_it_does_not_retry_is_reported_failed_with_its_reason() -> None:
    operation = BackgroundOperation("load", _gate())

    async def needed() -> bool:
        raise ValueError("the check itself is broken")

    _, status = _tools(operation)
    operation.start_when_needed(needed, retry_on=(ConnectionError,), retry_seconds=0)
    with pytest.raises(ValueError):
        await operation.wait_until_settled()
    assert operation.state == "failed"
    answer = await status.run()
    assert answer.metadata is not None
    assert answer.metadata["state"] == "failed"
    assert answer.metadata["last"]["error"] == "ValueError: the check itself is broken"


async def test_a_failure_is_logged_with_its_traceback(caplog: pytest.LogCaptureFixture) -> None:
    body = _gate()
    body.fail_with = RuntimeError("part way")
    body.release.set()
    operation = BackgroundOperation("load", body)

    async def needed() -> bool:
        raise ValueError("the check itself is broken")

    with caplog.at_level("WARNING"):
        operation.start()
        await operation.wait()
        operation.start_when_needed(needed, retry_on=(ConnectionError,), retry_seconds=0)
        with pytest.raises(ValueError):
            await operation.wait_until_settled()
    failures = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(failures) == 2
    assert all(r.exc_info is not None for r in failures)


async def test_a_wait_failing_after_a_run_finished_keeps_that_run_as_the_last_outcome() -> None:
    """the status reports the newest thing that happened: a run that ended after the wait began
    is newer than the wait's failure, so the failure does not replace it."""
    body = _gate()
    operation = BackgroundOperation("load", body)
    deciding = asyncio.Event()
    decide = asyncio.Event()

    async def needed() -> bool:
        deciding.set()
        await decide.wait()
        raise ValueError("the check itself is broken")

    operation.start_when_needed(needed, retry_on=(ConnectionError,), retry_seconds=0)
    await deciding.wait()
    assert operation.start()
    body.release.set()
    await operation.wait()
    decide.set()
    with pytest.raises(ValueError):
        await operation.wait_until_settled()

    assert operation.state == "succeeded"
    assert operation.last is not None and operation.last.result == {"rows": 3}


async def test_the_request_tool_records_a_request_and_starts_the_operation() -> None:
    body = _gate()
    requests: list[str] = []

    async def request() -> None:
        requests.append("asked")

    operation = BackgroundOperation("refresh", body, request=request)
    tool = RequestOperationTool(
        name="enr.reload",
        description="refresh",
        operation=lambda: operation,
        status_tool="enr.load_status",
    )

    first = await tool.run()
    second = await tool.run()

    assert requests == ["asked", "asked"], "a request was not recorded"
    assert first.success and first.metadata == {"started": True}
    # a request while the operation runs is not refused: the run in progress takes it
    assert second.success and second.metadata == {"started": False}
    assert "after the run in progress" in second.content
    body.release.set()
    await operation.wait()
    # the run in progress is followed by exactly one more, which takes the request it may have missed
    assert body.calls == 2


async def test_the_request_tool_refuses_while_the_operation_is_not_built_yet() -> None:
    tool = RequestOperationTool(name="enr.reload", description="refresh", operation=lambda: None, status_tool="s")

    refused = await tool.run()

    assert (refused.success, refused.error_code) == (False, TOOL_NOT_READY)


async def test_a_request_made_after_the_drains_last_check_still_runs() -> None:
    """the lost wake-up: the run has looked for requests for the last time but has not ended yet."""
    wanted = 0
    taken: list[int] = []
    past_last_check = asyncio.Event()
    finish = asyncio.Event()

    async def request() -> None:
        nonlocal wanted
        wanted += 1

    async def drain() -> int:
        # as CoalescedRun.drain: run while a request is waiting, then end
        nonlocal wanted
        runs = 0
        while wanted:
            wanted -= 1
            runs += 1
        taken.append(runs)
        past_last_check.set()
        await finish.wait()  # still running, its last look for requests already made
        return runs

    operation = BackgroundOperation("refresh", drain, request=request)
    tool = RequestOperationTool(name="enr.reload", description="refresh", operation=lambda: operation, status_tool="s")

    await tool.run()
    await past_last_check.wait()
    late = await tool.run()  # recorded now, after the drain's last check, while it still runs
    finish.set()
    await operation.wait()

    assert late.success and late.metadata == {"started": False}
    assert wanted == 0, "a request was recorded and no run took it"
    assert taken == [1, 1]


async def test_an_operation_built_without_a_request_step_refuses_a_request() -> None:
    operation = BackgroundOperation("refresh", _gate())

    with pytest.raises(TypeError, match="request="):
        await operation.request()
    assert not operation.running

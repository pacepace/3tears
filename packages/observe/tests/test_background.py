"""unit tests for ``threetears.observe.spawn_background``.

verifies the three outcome branches:

- normal completion logs INFO with ``outcome=ok``
- exceptional exit logs WARNING with ``exc_info`` and ``outcome=error``
- cancelled exit logs INFO with ``outcome=cancelled``
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from threetears.observe import spawn_background


class _RecordingHandler(logging.Handler):
    """log handler that captures records for assertion."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _make_logger() -> tuple[logging.Logger, _RecordingHandler]:
    """build fresh logger with recording handler attached.

    :return: pair of logger and handler
    :rtype: tuple[logging.Logger, _RecordingHandler]
    """
    name = f"test.spawn_background.{id(object())}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = _RecordingHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger, handler


@pytest.mark.asyncio
async def test_spawn_background_normal_completion_logs_info() -> None:
    """normal coroutine completion logs INFO with outcome=ok."""
    logger, handler = _make_logger()

    async def ok_coro() -> int:
        return 42

    task = spawn_background(ok_coro(), name="unit-ok", logger=logger)
    result = await task
    await asyncio.sleep(0)
    assert result == 42
    info_records = [r for r in handler.records if r.levelno == logging.INFO]
    assert len(info_records) == 1
    record = info_records[0]
    extra = record.__dict__.get("extra_data")
    assert extra is not None
    assert extra["task_name"] == "unit-ok"
    assert extra["outcome"] == "ok"


@pytest.mark.asyncio
async def test_spawn_background_exception_logs_warning_with_exc_info() -> None:
    """coroutine raising exception logs WARNING with exc_info attached."""
    logger, handler = _make_logger()

    async def bad_coro() -> None:
        raise RuntimeError("boom")

    task = spawn_background(bad_coro(), name="unit-bad", logger=logger)
    with pytest.raises(RuntimeError, match="boom"):
        await task
    await asyncio.sleep(0)
    warning_records = [r for r in handler.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    record = warning_records[0]
    assert record.exc_info is not None
    exc_type, exc_value, _tb = record.exc_info
    assert exc_type is RuntimeError
    assert str(exc_value) == "boom"
    extra = record.__dict__.get("extra_data")
    assert extra is not None
    assert extra["task_name"] == "unit-bad"
    assert extra["outcome"] == "error"
    assert extra["exc_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_spawn_background_cancelled_logs_info_cancelled() -> None:
    """cancelled task logs INFO with outcome=cancelled."""
    logger, handler = _make_logger()

    async def slow_coro() -> None:
        await asyncio.sleep(10)

    task = spawn_background(slow_coro(), name="unit-cancel", logger=logger)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    info_records = [
        r for r in handler.records
        if r.levelno == logging.INFO
        and r.__dict__.get("extra_data", {}).get("outcome") == "cancelled"
    ]
    assert len(info_records) == 1
    record = info_records[0]
    extra = record.__dict__["extra_data"]
    assert extra["task_name"] == "unit-cancel"


@pytest.mark.asyncio
async def test_spawn_background_returns_task_for_lifecycle_control() -> None:
    """returned Task supports standard cancel and await lifecycle."""
    logger, _ = _make_logger()

    async def coro() -> None:
        await asyncio.sleep(0.01)

    task = spawn_background(coro(), name="unit-lifecycle", logger=logger)
    assert isinstance(task, asyncio.Task)
    assert task.get_name() == "unit-lifecycle"
    await task

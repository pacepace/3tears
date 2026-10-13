"""CopyWriteHealth: whether a persisted copy's writes are landing, counted per write operation.

a single write is one operation; a pass of many writes is one operation, failed if any write in it
failed. one sync over many entries against a dead bucket must count once, not once per entry, or a
single bad pass takes a service out of rotation.
"""

from __future__ import annotations

import logging

import pytest

from threetears.nats import WRITE_FAILURE_THRESHOLD, CopyWriteHealth


class _Down(Exception):
    pass


async def _ok() -> str:
    return "ok"


async def _fail() -> None:
    raise _Down("connection closed")


def _health() -> CopyWriteHealth:
    return CopyWriteHealth(copy="test catalog", consequence="the service reports itself not ready")


async def _failed_write(health: CopyWriteHealth) -> None:
    with pytest.raises(_Down):
        await health.write(_fail(), key="k")


@pytest.mark.asyncio
async def test_a_write_returns_what_it_returned_and_a_new_copy_is_persisting() -> None:
    health = _health()
    assert health.persisting is True
    assert await health.write(_ok(), key="k") == "ok"
    assert health.persisting is True


@pytest.mark.asyncio
async def test_failed_writes_in_a_row_stop_it_persisting_with_one_error_and_a_landed_write_ends_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    health = _health()
    with caplog.at_level(logging.INFO, logger="threetears.nats.persisted_copy"):
        for _ in range(WRITE_FAILURE_THRESHOLD - 1):
            await _failed_write(health)
        assert health.persisting is True
        for _ in range(3):
            await _failed_write(health)
        assert health.persisting is False
        await health.write(_ok(), key="k")
    assert health.persisting is True
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert f"test catalog writes to its bucket have failed {WRITE_FAILURE_THRESHOLD} times in a row" in errors[0]
    assert "connection closed" in errors[0] and "not ready" in errors[0]
    assert any("landing again" in r.getMessage() for r in caplog.records if r.levelno == logging.INFO)


@pytest.mark.asyncio
async def test_a_pass_over_many_entries_that_fails_counts_once() -> None:
    health = _health()
    async with health.operation():
        for _ in range(WRITE_FAILURE_THRESHOLD + 2):
            with pytest.raises(_Down):
                await health.write(_fail(), key="k")
    assert health.persisting is True, "one failed sync pass is one failed operation"


@pytest.mark.asyncio
async def test_failed_passes_in_a_row_stop_it_persisting() -> None:
    health = _health()
    for _ in range(WRITE_FAILURE_THRESHOLD):
        async with health.operation():
            with pytest.raises(_Down):
                await health.write(_fail(), key="k")
    assert health.persisting is False


@pytest.mark.asyncio
async def test_a_pass_with_any_failed_write_fails_and_one_where_all_landed_ends_the_streak() -> None:
    health = _health()
    for _ in range(WRITE_FAILURE_THRESHOLD - 1):
        async with health.operation():
            await health.write(_ok(), key="a")
            with pytest.raises(_Down):
                await health.write(_fail(), key="b")
    async with health.operation():
        await health.write(_ok(), key="a")
    for _ in range(WRITE_FAILURE_THRESHOLD - 1):
        await _failed_write(health)
    assert health.persisting is True, "the clean pass ended the earlier streak"


@pytest.mark.asyncio
async def test_a_pass_that_wrote_nothing_counts_as_nothing() -> None:
    health = _health()
    for _ in range(WRITE_FAILURE_THRESHOLD - 1):
        await _failed_write(health)
    async with health.operation():
        pass
    await _failed_write(health)
    assert health.persisting is False, "an empty pass must not end a streak of failures"


@pytest.mark.asyncio
async def test_a_pass_that_raises_still_counts_its_failure() -> None:
    health = _health()
    for _ in range(WRITE_FAILURE_THRESHOLD):
        with pytest.raises(_Down):
            async with health.operation():
                await health.write(_fail(), key="k")
    assert health.persisting is False


@pytest.mark.asyncio
async def test_a_single_write_from_another_task_during_an_open_pass_counts_on_its_own() -> None:
    import asyncio

    health = _health()
    gate = asyncio.Event()

    async def sweep() -> None:
        async with health.operation():
            with pytest.raises(_Down):
                await health.write(_fail(), key="sweep")
            await gate.wait()

    async def registrations() -> None:
        # failing writes from a concurrent task are their own operations, not the sweep's
        for _ in range(WRITE_FAILURE_THRESHOLD - 1):
            await _failed_write(health)
        gate.set()

    await asyncio.gather(sweep(), registrations())
    assert health.persisting is False, "two failed registrations and one failed sweep are three operations"


@pytest.mark.asyncio
async def test_overlapping_passes_in_two_tasks_each_count_once() -> None:
    import asyncio

    health = _health()
    both_open = asyncio.Barrier(2)

    async def failing_pass() -> None:
        async with health.operation():
            await both_open.wait()
            for _ in range(WRITE_FAILURE_THRESHOLD + 1):
                with pytest.raises(_Down):
                    await health.write(_fail(), key="k")

    await asyncio.gather(failing_pass(), failing_pass())
    assert health.persisting is True, "two failed passes are two operations, not one per write"
    async with health.operation():
        with pytest.raises(_Down):
            await health.write(_fail(), key="k")
    assert health.persisting is False

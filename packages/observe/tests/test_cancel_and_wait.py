"""unit tests for ``threetears.observe.cancel_and_wait``.

the helper replaces the hand-written ``task.cancel(); await task`` under ``except
CancelledError: pass``, which swallowed the CALLER's cancellation along with the one it asked
for. these pin that the caller's cancellation now propagates, and that nothing else is lost.
"""

from __future__ import annotations

import asyncio

import pytest

from threetears.observe import cancel_and_wait


async def _forever() -> None:
    await asyncio.Event().wait()


async def test_a_running_task_is_cancelled_and_waited_for() -> None:
    task = asyncio.create_task(_forever())
    await asyncio.sleep(0)
    await cancel_and_wait(task)
    assert task.cancelled()


async def test_a_task_that_returns_on_its_cancellation_ends_without_error() -> None:
    async def _returns_on_cancel() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return

    task = asyncio.create_task(_returns_on_cancel())
    await asyncio.sleep(0)
    await cancel_and_wait(task)
    assert task.done() and not task.cancelled()


async def test_a_task_that_fails_while_unwinding_raises_its_failure() -> None:
    async def _fails_on_cancel() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise ValueError("teardown broke") from None

    task = asyncio.create_task(_fails_on_cancel())
    await asyncio.sleep(0)
    with pytest.raises(ValueError, match="teardown broke"):
        await cancel_and_wait(task)


async def test_the_callers_own_cancellation_is_not_swallowed() -> None:
    release = asyncio.Event()

    async def _slow_to_stop() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
            raise

    inner = asyncio.create_task(_slow_to_stop())
    await asyncio.sleep(0)
    reached_after = False

    async def _stopper() -> None:
        nonlocal reached_after
        await cancel_and_wait(inner)
        reached_after = True

    outer = asyncio.create_task(_stopper())
    await asyncio.sleep(0)
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    assert not reached_after
    release.set()
    await asyncio.wait([inner])
    assert inner.cancelled()


async def test_a_task_cannot_wait_for_its_own_cancellation() -> None:
    async def _self_stop() -> None:
        current = asyncio.current_task()
        assert current is not None
        await cancel_and_wait(current)

    with pytest.raises(RuntimeError, match="its own cancellation"):
        await asyncio.create_task(_self_stop())

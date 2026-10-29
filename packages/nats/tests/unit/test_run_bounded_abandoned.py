"""A JetStream call abandoned at its deadline, or by a caller that was cancelled, never leaves "Task exception was
never retrieved": its outcome is taken when it ends (cobalt dev, 2026-10-29: get_last_msg and add_consumer tasks
abandoned through a slow-consumer disconnect each logged one)."""

from __future__ import annotations

import asyncio
import gc
from typing import Any

import pytest

from threetears.nats._publish import run_bounded
from threetears.nats.errors import PublishTimeoutError


def _unretrieved(loop: asyncio.AbstractEventLoop) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, context: seen.append(context))
    return seen


async def _fails_later(started: asyncio.Event, release: asyncio.Event) -> None:
    started.set()
    try:
        await release.wait()
    except asyncio.CancelledError:
        # as nats-py's flush path does: the cancellation is swallowed, and the call fails on its own later
        await release.wait()
    raise TimeoutError("nats: timeout")


async def test_an_abandoned_call_that_fails_later_leaves_no_unretrieved_exception() -> None:
    seen = _unretrieved(asyncio.get_running_loop())
    started, release = asyncio.Event(), asyncio.Event()
    with pytest.raises(PublishTimeoutError):
        await run_bounded(lambda: _fails_later(started, release), timeout=0.05, what="a metadata read")
    release.set()
    for _ in range(5):
        await asyncio.sleep(0)
    gc.collect()
    await asyncio.sleep(0)
    assert not [context for context in seen if "never retrieved" in str(context.get("message", ""))]


async def test_a_call_whose_caller_is_cancelled_leaves_no_unretrieved_exception() -> None:
    seen = _unretrieved(asyncio.get_running_loop())
    started, release = asyncio.Event(), asyncio.Event()
    caller = asyncio.create_task(run_bounded(lambda: _fails_later(started, release), timeout=30, what="a read"))
    await started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    release.set()
    for _ in range(5):
        await asyncio.sleep(0)
    gc.collect()
    await asyncio.sleep(0)
    assert not [context for context in seen if "never retrieved" in str(context.get("message", ""))]

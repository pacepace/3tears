"""A JetStream call abandoned at its deadline, or by a caller that was cancelled, never leaves "Task exception was
never retrieved": its outcome is taken when it ends (cobalt dev, 2026-10-29: get_last_msg and add_consumer tasks
abandoned through a slow-consumer disconnect each logged one). Driven through an object's metadata read, one of the
bounded calls the store makes."""

from __future__ import annotations

import asyncio
import gc
from datetime import timedelta
from typing import Any

import pytest

from threetears.nats import ObjectStoreError
from threetears.nats.object_store import NatsObjectStore


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


def _store(started: asyncio.Event, release: asyncio.Event, *, deadline: float) -> NatsObjectStore:
    """an object store whose metadata read swallows its cancellation and fails on its own once released."""

    class _JetStream:
        async def get_last_msg(self, stream: str, subject: str, *, direct: bool = False) -> Any:
            await _fails_later(started, release)

    class _Client:
        def jetstream_context(self) -> _JetStream:
            return _JetStream()

    return NatsObjectStore(
        client=_Client(),  # type: ignore[arg-type]
        full_name="pod-objects",
        op_timeout=timedelta(seconds=deadline),
    )


async def test_an_abandoned_call_that_fails_later_leaves_no_unretrieved_exception() -> None:
    seen = _unretrieved(asyncio.get_running_loop())
    started, release = asyncio.Event(), asyncio.Event()
    with pytest.raises(ObjectStoreError, match="metadata"):
        await _store(started, release, deadline=0.05).info("a/b")
    release.set()
    for _ in range(5):
        await asyncio.sleep(0)
    gc.collect()
    await asyncio.sleep(0)
    assert not [context for context in seen if "never retrieved" in str(context.get("message", ""))]


async def test_a_call_whose_caller_is_cancelled_leaves_no_unretrieved_exception() -> None:
    seen = _unretrieved(asyncio.get_running_loop())
    started, release = asyncio.Event(), asyncio.Event()
    caller = asyncio.create_task(_store(started, release, deadline=30).info("a/b"))
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

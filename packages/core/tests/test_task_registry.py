"""unit tests for :class:`threetears.core.KeyedTaskRegistry`."""

from __future__ import annotations

import asyncio
import contextlib
from uuid import uuid4

from threetears.core import KeyedTaskRegistry


async def _forever() -> None:
    await asyncio.sleep(3600)


async def _drain(task: asyncio.Task[object]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_register_get_pop_len() -> None:
    reg = KeyedTaskRegistry()
    key = uuid4()
    task = asyncio.create_task(_forever())
    try:
        assert len(reg) == 0
        reg.register(key, task)
        assert reg.get(key) is task
        assert len(reg) == 1

        popped = reg.pop(key)
        assert popped is task
        assert reg.get(key) is None
        assert len(reg) == 0
        # popping a missing key is a clean None
        assert reg.pop(key) is None
    finally:
        await _drain(task)


async def test_discard_is_identity_guarded() -> None:
    reg = KeyedTaskRegistry()
    key = uuid4()
    first = asyncio.create_task(_forever())
    second = asyncio.create_task(_forever())
    try:
        reg.register(key, first)
        # a newer task reuses the same key
        reg.register(key, second)
        # a stale done-callback for the FIRST task must NOT evict the newer one
        reg.discard(key, first)
        assert reg.get(key) is second
        # the matching discard does remove it
        reg.discard(key, second)
        assert reg.get(key) is None
    finally:
        await _drain(first)
        await _drain(second)


async def test_pop_before_cancel_is_idempotent() -> None:
    reg = KeyedTaskRegistry()
    key = uuid4()
    task = asyncio.create_task(_forever())
    try:
        reg.register(key, task)
        assert reg.pop(key) is task
        # a redelivered / duplicate cancel finds nothing — no double-pop
        assert reg.pop(key) is None
    finally:
        await _drain(task)

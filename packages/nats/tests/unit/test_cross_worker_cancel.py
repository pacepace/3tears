"""unit tests for :class:`threetears.nats.CrossWorkerCanceller`.

covers local cancel (pop-before-cancel + on_cancel), cross-worker routing
(publish when not locally owned), cross-worker receipt (only the owning
worker cancels; unknown keys no-op), and fail-open on subscribe/publish.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any
from uuid import UUID, uuid4

from threetears.core import KeyedTaskRegistry
from threetears.nats import CrossWorkerCanceller, TaskCancelEnvelope
from threetears.nats.errors import PublishError, SubscribeError
from threetears.nats.subjects import Subject

_SUBJECT = Subject(path="test.task.cancel", kind="point")
_LOG = logging.getLogger("test.cross_worker_cancel")


# parity-exempt: intentional 2-method NatsClient stub (subscribe_typed + publish only) for cross-worker-cancel tests; full NatsClient parity is out of scope here
class _FakeNats:
    """minimal NatsClient stand-in capturing subscribe cb + published messages."""

    def __init__(self) -> None:
        self.published: list[TaskCancelEnvelope] = []
        self._cb: Any = None
        self.subscribe_fails = False
        self.publish_fails = False

    async def subscribe_typed(self, *, subject: Subject, cb: Any, message_type: Any) -> object:
        if self.subscribe_fails:
            raise SubscribeError("subscribe boom")
        self._cb = cb
        return object()

    async def publish(self, *, subject: Subject, message: TaskCancelEnvelope) -> None:
        if self.publish_fails:
            raise PublishError("publish boom")
        self.published.append(message)

    async def deliver(self, envelope: TaskCancelEnvelope) -> None:
        assert self._cb is not None, "bind() must run before deliver()"
        await self._cb(envelope)


async def _forever() -> None:
    await asyncio.sleep(3600)


async def _drain(task: asyncio.Task[object]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _make() -> tuple[CrossWorkerCanceller, list[tuple[UUID, dict[str, Any]]]]:
    calls: list[tuple[UUID, dict[str, Any]]] = []

    async def _on_cancel(key: UUID, payload: dict[str, Any]) -> None:
        calls.append((key, payload))

    canceller = CrossWorkerCanceller(
        subject=_SUBJECT,
        on_cancel=_on_cancel,
        logger=_LOG,
        registry=KeyedTaskRegistry(),
    )
    return canceller, calls


async def test_local_cancel_pops_cancels_and_runs_callback() -> None:
    canceller, calls = _make()
    key = uuid4()
    task = asyncio.create_task(_forever())
    canceller.registry.register(key, task)

    result = await canceller.request_cancel(key, {"who": "alice"})

    assert result is True
    assert task.cancelled() or task.cancelling() > 0
    assert calls == [(key, {"who": "alice"})]
    # pop-before-cancel: the task is gone from the registry
    assert canceller.registry.get(key) is None
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_request_cancel_unowned_without_nats_returns_false() -> None:
    canceller, calls = _make()
    # never bound -> no NATS; an unowned key is simply unreachable
    result = await canceller.request_cancel(uuid4(), {"who": "bob"})
    assert result is False
    assert calls == []


async def test_request_cancel_unowned_publishes_cross_worker() -> None:
    canceller, calls = _make()
    fake = _FakeNats()
    await canceller.bind(fake)  # type: ignore[arg-type]

    key = uuid4()
    result = await canceller.request_cancel(key, {"who": "carol"})

    assert result is False  # not cancelled locally
    assert len(fake.published) == 1
    env = fake.published[0]
    assert env.key == str(key)
    assert env.payload == {"who": "carol"}
    assert calls == []  # on_cancel runs on the OWNING worker, not here


async def test_cross_worker_receipt_cancels_locally_owned_task() -> None:
    canceller, calls = _make()
    fake = _FakeNats()
    await canceller.bind(fake)  # type: ignore[arg-type]

    key = uuid4()
    task = asyncio.create_task(_forever())
    canceller.registry.register(key, task)

    # a cancel published by another worker arrives here (we own the task)
    await fake.deliver(TaskCancelEnvelope(key=str(key), payload={"who": "dave"}))

    assert task.cancelled() or task.cancelling() > 0
    assert calls == [(key, {"who": "dave"})]
    assert canceller.registry.get(key) is None
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_cross_worker_receipt_unknown_key_is_noop() -> None:
    canceller, calls = _make()
    fake = _FakeNats()
    await canceller.bind(fake)  # type: ignore[arg-type]

    # this worker owns nothing for that key -> clean no-op
    await fake.deliver(TaskCancelEnvelope(key=str(uuid4()), payload={}))
    assert calls == []


async def test_cross_worker_receipt_bad_key_does_not_raise() -> None:
    canceller, calls = _make()
    fake = _FakeNats()
    await canceller.bind(fake)  # type: ignore[arg-type]

    await fake.deliver(TaskCancelEnvelope(key="not-a-uuid", payload={}))
    assert calls == []


async def test_redelivered_cancel_is_idempotent() -> None:
    canceller, calls = _make()
    fake = _FakeNats()
    await canceller.bind(fake)  # type: ignore[arg-type]

    key = uuid4()
    task = asyncio.create_task(_forever())
    canceller.registry.register(key, task)

    env = TaskCancelEnvelope(key=str(key), payload={"who": "erin"})
    await fake.deliver(env)
    await fake.deliver(env)  # duplicate delivery

    # cancelled + callback fired exactly ONCE (pop-before-cancel)
    assert calls == [(key, {"who": "erin"})]
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_bind_failopen_on_subscribe_error() -> None:
    canceller, _ = _make()
    fake = _FakeNats()
    fake.subscribe_fails = True
    # must not raise — a failed subscription only disables cross-worker receipt
    await canceller.bind(fake)  # type: ignore[arg-type]

    # local cancels still work despite the failed subscription
    key = uuid4()
    task = asyncio.create_task(_forever())
    canceller.registry.register(key, task)
    assert await canceller.request_cancel(key) is True
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_request_cancel_failopen_on_publish_error() -> None:
    canceller, calls = _make()
    fake = _FakeNats()
    fake.publish_fails = True
    await canceller.bind(fake)  # type: ignore[arg-type]

    # unowned key -> tries to publish -> publish raises -> swallowed, returns False
    result = await canceller.request_cancel(uuid4())
    assert result is False
    assert calls == []

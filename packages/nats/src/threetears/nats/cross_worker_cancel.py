"""cancel a keyed asyncio task on this worker or another, over NATS.

a cancel-by-key request (e.g. a user "Stop" on a streaming turn) can arrive
on a worker that does NOT hold the running task — a second tab/device pins
a different worker in a multi-worker deployment. remotely releasing another
worker's resources (locks, streams) is unsafe, so the cancel is ROUTED to
the owning worker, which cancels its own task and runs its own cleanup.

:class:`CrossWorkerCanceller` wraps a
:class:`threetears.core.KeyedTaskRegistry`. :meth:`request_cancel` cancels
locally when the task is owned here; otherwise it publishes a
:class:`TaskCancelEnvelope` on a consumer-supplied
:class:`~threetears.nats.Subject` that every worker subscribes to
(:meth:`bind`). on the owning worker the task is popped-before-cancel,
cancelled, and a consumer ``on_cancel(key, payload)`` callback is awaited.

the canceller knows NOTHING about what a cancel MEANS to the consumer —
releasing a lock, emitting a "cancelled" frame, aborting a checkpoint all
live in the consumer's ``on_cancel`` callback and in the cancelled
coroutine's own ``finally``. the cross-worker payload is opaque and
threaded through verbatim. this is the platform's ``RoomFanout``
publish-one/act-on-receive-per-pod pattern, specialised to cancellation.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel

if TYPE_CHECKING:
    from threetears.core.task_registry import KeyedTaskRegistry
    from threetears.nats.client import NatsClient
    from threetears.nats.subjects import Subject

__all__ = ["CrossWorkerCanceller", "TaskCancelEnvelope"]


class TaskCancelEnvelope(BaseModel):
    """cross-worker request to cancel a keyed task.

    :ivar key: the task key (a ``UUID`` rendered as a string).
    :vartype key: str
    :ivar payload: opaque consumer payload threaded verbatim to the owning
        worker's ``on_cancel`` callback (e.g. who to notify). the canceller
        never interprets it.
    :vartype payload: dict[str, Any]
    """

    key: str
    payload: dict[str, Any] = {}


class CrossWorkerCanceller:
    """cancel a keyed task locally, or route it to the owning worker.

    :param subject: broadcast subject every worker subscribes to for cancel
        requests. supply a plain ``Subject`` literal for a consumer-private
        signal (no ``Subjects`` builder needed); ``kind`` should be a
        broadcast point (no queue group — every worker must receive it).
    :ptype subject: Subject
    :param on_cancel: awaited on the owning worker AFTER the task is
        cancelled, with ``(key, payload)``. does the consumer's post-cancel
        work (notify, etc.); the released lock/aborted stream belong to the
        cancelled coroutine's own ``finally``, not here.
    :ptype on_cancel: Callable[[UUID, dict[str, Any]], Awaitable[None]]
    :param logger: logger for fail-open diagnostics.
    :ptype logger: logging.Logger
    :param registry: the task registry to own. the consumer supplies it (and
        registers spawned tasks through the SAME reference) so ``threetears.nats``
        keeps ``threetears.core`` a type-only dependency rather than a runtime
        one — construct a :class:`threetears.core.KeyedTaskRegistry` and pass it.
    :ptype registry: KeyedTaskRegistry
    """

    def __init__(
        self,
        *,
        subject: Subject,
        on_cancel: Callable[[UUID, dict[str, Any]], Awaitable[None]],
        logger: logging.Logger,
        registry: KeyedTaskRegistry,
    ) -> None:
        self._subject = subject
        self._on_cancel = on_cancel
        self._logger = logger
        self._registry = registry
        self._nats: NatsClient | None = None

    @property
    def registry(self) -> KeyedTaskRegistry:
        """the underlying task registry (register/discard spawned tasks here).

        :return: the owned registry
        :rtype: KeyedTaskRegistry
        """
        return self._registry

    async def bind(self, nats_client: NatsClient | None) -> None:
        """subscribe this worker to the cross-worker cancel subject.

        call once per worker after NATS connects. fail-open: a failed
        subscription leaves this worker unable to honour cross-worker
        cancels (LOCAL cancels still work), and ``None`` (degraded /
        single-worker) skips the subscription entirely.

        :param nats_client: connected client, or ``None`` for local-only
        :ptype nats_client: NatsClient | None
        :rtype: None
        """
        self._nats = nats_client
        if nats_client is None:
            return

        async def _on_msg(envelope: TaskCancelEnvelope) -> None:
            try:
                key = UUID(envelope.key)
            except (ValueError, TypeError):
                self._logger.warning(
                    "cross-worker cancel: bad key",
                    extra={"extra_data": {"key": envelope.key}},
                )
                return
            # idempotent: only the worker owning the task cancels; others no-op.
            await self._cancel_local(key, envelope.payload)

        try:
            await nats_client.subscribe_typed(
                subject=self._subject,
                cb=_on_msg,
                message_type=TaskCancelEnvelope,
            )
        except Exception as exc:  # noqa: BLE001 — startup fail-open boundary: a cancel-subscription failure of ANY kind must not break worker startup; local cancels still work
            self._logger.warning(
                "cross-worker cancel subscription failed; this worker "
                "cannot honour cross-worker cancels",
                extra={"extra_data": {"error": str(exc)}},
            )

    async def _cancel_local(self, key: UUID, payload: dict[str, Any]) -> bool:
        """cancel a locally-owned task (pop-before-cancel), then ``on_cancel``.

        :param key: task key to cancel
        :ptype key: UUID
        :param payload: opaque payload forwarded to ``on_cancel``
        :ptype payload: dict[str, Any]
        :return: ``True`` when a local task was found and cancelled
        :rtype: bool
        """
        task = self._registry.pop(key)
        if task is None:
            return False
        task.cancel()
        await self._on_cancel(key, payload)
        return True

    async def request_cancel(
        self, key: UUID, payload: dict[str, Any] | None = None
    ) -> bool:
        """cancel ``key``'s task on this worker, or route to its owner.

        tries the local registry first (the common case — the task usually
        runs on the worker that received the request). if not owned locally,
        publishes a cross-worker cancel so the owning worker acts. NATS-down
        / single-worker: a not-locally-owned task is simply unreachable.

        :param key: task key to cancel
        :ptype key: UUID
        :param payload: opaque payload forwarded to the owner's ``on_cancel``
        :ptype payload: dict[str, Any] | None
        :return: ``True`` if cancelled locally; ``False`` if routed
            cross-worker or unreachable
        :rtype: bool
        """
        payload = payload or {}
        if await self._cancel_local(key, payload):
            return True
        if self._nats is None:
            return False
        try:
            await self._nats.publish(
                subject=self._subject,
                message=TaskCancelEnvelope(key=str(key), payload=payload),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort fail-open: a cancel publish failure must not raise into the caller; the consumer's own TTL/backstop covers a missed cross-worker cancel
            self._logger.warning(
                "cross-worker cancel publish failed; a task on another "
                "worker cannot be cancelled",
                extra={"extra_data": {"key": str(key), "error": str(exc)}},
            )
        return False

"""per-worker registry of cancellable asyncio tasks, keyed by ``UUID``.

a fire-and-forget ``asyncio.Task`` (e.g. a streaming agent turn) drops its
handle after ``create_task``, so nothing can reach the running task to
abort it. :class:`KeyedTaskRegistry` keeps the handle reachable: the spawn
site :meth:`register`\\ s on start and self-cleans via :meth:`discard` in
the task's done-callback; a canceller :meth:`pop`\\ s the task BEFORE
cancelling it (pop-before-cancel avoids the double-cancel-aborts-\\
``CancelledError``-cleanup pitfall).

per-worker in-memory and single-event-loop (all access on one loop, so no
lock). for cross-worker cancellation pair with
:class:`threetears.nats.CrossWorkerCanceller`, which owns one of these and
routes a cancel to whichever worker holds the task.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

__all__ = ["KeyedTaskRegistry"]


class KeyedTaskRegistry:
    """in-process registry of cancellable ``asyncio.Task``\\ s keyed by ``UUID``.

    at most one task is registered per key at a time; registering a key
    that is already present overwrites it. not thread-safe — intended for
    single-event-loop use.
    """

    def __init__(self) -> None:
        self._tasks: dict[UUID, asyncio.Task[object]] = {}

    def register(self, key: UUID, task: asyncio.Task[object]) -> None:
        """register a running task so it can later be cancelled by key.

        :param key: identity the task is tracked under
        :ptype key: UUID
        :param task: the fire-and-forget task
        :ptype task: asyncio.Task[object]
        :rtype: None
        """
        self._tasks[key] = task

    def pop(self, key: UUID) -> asyncio.Task[object] | None:
        """remove and return a key's task, if any (pop-before-cancel).

        a canceller removes the task here BEFORE cancelling it, so a
        redelivered / duplicate cancel finds nothing and is a clean no-op
        rather than a double-cancel.

        :param key: identity whose task to claim
        :ptype key: UUID
        :return: the registered task, or ``None`` if none is active
        :rtype: asyncio.Task[object] | None
        """
        return self._tasks.pop(key, None)

    def get(self, key: UUID) -> asyncio.Task[object] | None:
        """return a key's task without removing it.

        :param key: identity to look up
        :ptype key: UUID
        :return: the registered task, or ``None``
        :rtype: asyncio.Task[object] | None
        """
        return self._tasks.get(key)

    def discard(self, key: UUID, task: asyncio.Task[object]) -> None:
        """drop a key's entry when its task finishes (identity-guarded).

        the identity guard ensures a stale done-callback cannot evict a
        newer task that reused the same key.

        :param key: the finished task's key
        :ptype key: UUID
        :param task: the task whose completion triggered this cleanup
        :ptype task: asyncio.Task[object]
        :rtype: None
        """
        if self._tasks.get(key) is task:
            del self._tasks[key]

    def __len__(self) -> int:
        """number of tasks currently registered.

        :return: registered task count
        :rtype: int
        """
        return len(self._tasks)

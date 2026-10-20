"""fire-and-forget asyncio task helper with logged done-callbacks.

an ``asyncio.create_task(...)`` call that nobody awaits is a silent
fire: if the coroutine raises, the exception is logged by the default
event loop handler on task destruction -- which may or may not reach
our structured logger depending on how the loop is configured. worse,
cancelled tasks (normal shutdown) look identical to crashed tasks in
the default log.

``spawn_background`` wraps ``asyncio.create_task`` with a
done-callback that routes outcomes through ``threetears.observe``:

- normal completion -> INFO (background task stop)
- ``CancelledError`` -> INFO (shutdown protocol, not failure)
- any other exception -> WARNING with ``exc_info=True``

use for any ``create_task`` where the creator does not ``await`` the
task and does not attach a custom ``add_done_callback``. for tasks
that are stored on ``self._foo_task`` and cancelled + awaited during
shutdown, ``spawn_background`` is still the right call -- the task is
returned unchanged so callers can cancel or await.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

__all__ = ["cancel_and_wait", "spawn_background"]


def spawn_background(
    coro: Coroutine[Any, Any, Any],
    *,
    name: str,
    logger: logging.Logger,
) -> asyncio.Task[Any]:
    """schedule ``coro`` as background task with logged done-callback.

    wraps ``asyncio.create_task`` and attaches done-callback that
    logs task outcome through ``logger``. INFO on normal completion
    and cancellation, WARNING with ``exc_info`` on any other exception.
    returned ``Task`` is suitable for storing on ``self._foo_task``
    and cancelling + awaiting during shutdown; callers may still
    ``await`` it directly if desired.

    accepts any ``logging.Logger`` -- including ``ThreeTearsLogger`` --
    so callers in any repo can pass their existing logger without
    importing ``threetears.observe`` types.

    :param coro: coroutine to run as background task
    :ptype coro: Coroutine[Any, Any, Any]
    :param name: short human-readable task name for log messages
    :ptype name: str
    :param logger: logger used to emit done-callback outcomes
    :ptype logger: logging.Logger
    :return: scheduled asyncio task
    :rtype: asyncio.Task[Any]
    """
    task = asyncio.create_task(coro, name=name)

    def _on_done(t: asyncio.Task[Any]) -> None:
        """done-callback that routes task outcomes to ``logger``."""
        if t.cancelled():
            logger.info(
                f"background task cancelled: {name}",
                extra={"extra_data": {"task_name": name, "outcome": "cancelled"}},
            )
        else:
            exc = t.exception()
            if exc is None:
                logger.info(
                    f"background task stop: {name}",
                    extra={"extra_data": {"task_name": name, "outcome": "ok"}},
                )
            else:
                logger.warning(
                    f"background task failed: {name}",
                    extra={
                        "extra_data": {
                            "task_name": name,
                            "outcome": "error",
                            "exc_type": type(exc).__name__,
                        },
                    },
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

    task.add_done_callback(_on_done)
    return task


async def cancel_and_wait(task: asyncio.Task[Any]) -> None:
    """cancel ``task`` and wait for it to end, swallowing nothing but the cancellation asked for.

    the hand-written form -- ``task.cancel()``, then ``await task`` under ``except
    CancelledError: pass`` -- swallows two cancellations that look the same: the one it asked
    for, and the caller's own, when the caller is cancelled while it waits. the second is a
    real swallow: the caller carries on as though nobody had asked it to stop. this waits with
    :func:`asyncio.wait`, which never raises the task's outcome, so a ``CancelledError`` raised
    here is always the caller's own, and propagates.

    the task's own failure -- anything but its cancellation -- is raised, as ``await task``
    would raise it. a task that returns rather than re-raising its cancellation ends normally,
    which is fine too: either way it has ended.

    :param task: the task to stop; not the current task, which could never see itself end
    :ptype task: asyncio.Task[Any]
    :return: nothing, once the task has ended
    :rtype: None
    :raises RuntimeError: ``task`` is the current task
    :raises asyncio.CancelledError: the CALLER was cancelled while waiting
    :raises BaseException: whatever the task itself raised other than its cancellation
    """
    if task is asyncio.current_task():
        raise RuntimeError("a task cannot wait for its own cancellation to finish")
    task.cancel()
    await asyncio.wait([task])
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        raise error

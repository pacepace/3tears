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

__all__ = ["spawn_background"]


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

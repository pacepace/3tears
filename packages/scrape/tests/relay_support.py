"""The two callables ``relay_stream`` takes besides its sender, as the relay tests supply them.

``relay_stream`` reads the operator's side through ``receive`` (bytes) and ends when ``until``
completes (nothing). The tests' operator never sends and their claim is lost by setting an
event, so both are an event wait -- returned here in the shape the relay declares, rather than
as ``Event.wait``, whose ``True`` is neither bytes nor nothing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


async def silent_operator() -> bytes:
    """An operator who never sends: parks until the relay cancels it.

    :return: never returns
    :rtype: bytes
    """
    await asyncio.Event().wait()
    raise AssertionError("an event nobody can set was set")


def when_set(event: asyncio.Event) -> Callable[[], Awaitable[None]]:
    """A stop signal that completes once *event* is set.

    :param event: the event that stands for the claim being lost
    :ptype event: asyncio.Event
    :return: the ``until`` callable for ``relay_stream``
    :rtype: Callable[[], Awaitable[None]]
    """

    async def until() -> None:
        await event.wait()

    return until

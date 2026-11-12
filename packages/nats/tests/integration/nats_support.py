"""a nats-server started from a configuration given as text, for tests that need one configured their way.

:func:`threetears.core.testing.services.running_nats` starts it: a child process on the testament pool
(``THREETEARS_TEST_SERVICES=process``), from the image's binary of the tag's version, else a testcontainer of the
tag. In a container the configuration reaches the server in an environment variable, never a bind mount, which is
why the image must carry a shell (:func:`threetears.core.testing.fixtures.nats_server`). In process mode the
configuration's own ``port`` and JetStream ``store_dir`` give way to a free port and a store of the server's own.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from typing import TYPE_CHECKING

from threetears.core.testing.fixtures import NATS_TEST_IMAGE
from threetears.core.testing.services import running_nats as running_server

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

#: how often :func:`until` looks again
UNTIL_TICK_SECONDS = 0.02


@contextlib.contextmanager
def running_nats(config_text: str, *, image: str = NATS_TEST_IMAGE) -> Iterator[str]:
    """start a nats-server serving ``config_text``; yield its URI and stop it on exit.

    :param config_text: the server configuration
    :ptype config_text: str
    :param image: the ``nats:<version>-alpine`` image it runs, or stands for in process mode
    :ptype image: str
    :return: the server's URI, yielded
    :rtype: Iterator[str]
    """
    with running_server(config_text, image=image) as server:
        yield server.uri


async def until(condition: Callable[[], bool | Awaitable[bool]], *, within: float, what: str) -> None:
    """return as soon as ``condition`` holds; fail when it still does not after ``within`` seconds.

    The bound only matters on a failure: a test waits for the event it needs, not for a fixed time.

    :param condition: what to wait for, looked at every :data:`UNTIL_TICK_SECONDS`; it may ask the server
    :ptype condition: Callable[[], bool | Awaitable[bool]]
    :param within: the failure-only bound, in seconds
    :ptype within: float
    :param what: what the condition is, for the failure
    :ptype what: str
    :return: nothing
    :rtype: None
    :raises AssertionError: when the bound passes first
    """
    deadline = time.monotonic() + within
    while True:
        held = condition()
        if inspect.isawaitable(held):
            held = await held
        if held:
            return
        assert time.monotonic() < deadline, f"{what}: not within {within}s"
        await asyncio.sleep(UNTIL_TICK_SECONDS)

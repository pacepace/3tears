"""Prove a NATS grant REFUSES an operation by the server's own word, not by waiting out a deadline.

nats-server drops a refused publish without replying, so the classic check -- send a request and
expect a timeout -- costs its whole timeout per refusal and cannot tell a refusal from a dead
broker, a lost message or a bucket nobody created. The server ALSO sends the connection an
asynchronous ``-ERR 'Permissions Violation for Publish to "<subject>"'`` (or ``for Subscription
to``) the instant it refuses. nats-py hands that to the connection's ``error_cb``, and
:func:`refused` ends on its arrival: it races the operation against a violation NAMING the refused
subject, so the check returns the moment the server says no, fails if the operation is answered,
and fails if neither happens within a bound.

nats-py lowercases the whole ``-ERR`` payload before dispatching it, so subjects are compared
case-insensitively.

Two ways to feed a :class:`Violations` collector:

- a raw nats-py connection: pass :meth:`Violations.record` as ``error_cb`` (or use :func:`connect`);
- a :class:`threetears.nats.NatsClient`, which owns its ``error_cb`` and reports violations as
  ERROR log records: wrap the test body in :func:`watching_client_log`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Iterator

import nats
import nats.aio.client
import nats.errors

#: the phrase every nats-server permissions refusal carries, matched lowercased
PERMISSIONS_VIOLATION = "permissions violation"

#: how long :func:`refused` waits for the server's word before failing the check
DEFAULT_BOUND_SECONDS = 5.0

#: the logger :class:`threetears.nats.NatsClient` reports connection errors on
CLIENT_LOGGER = "threetears.nats"

#: what a refused operation may itself raise once the server has dropped it; the violation is the
#: evidence, so these are the refusal arriving by a second route and must not leak out of the check
REFUSAL_ERRORS: tuple[type[BaseException], ...] = (
    nats.errors.TimeoutError,
    nats.errors.NoRespondersError,
    TimeoutError,
)


class Violations:
    """the permissions-violation errors one or more connections have reported, in arrival order."""

    def __init__(self) -> None:
        """start with nothing reported.

        :return: nothing
        :rtype: None
        """
        self.errors: list[str] = []
        self._arrived = asyncio.Event()

    async def record(self, exc: Exception) -> None:
        """the ``error_cb`` for a raw nats-py connection: keep the error's text.

        :param exc: the error nats-py reported
        :ptype exc: Exception
        :return: nothing
        :rtype: None
        """
        self.note(str(exc))

    def note(self, text: str) -> None:
        """keep one reported error's text if it is a permissions violation, and wake any waiter.

        :param text: the error text
        :ptype text: str
        :return: nothing
        :rtype: None
        """
        if PERMISSIONS_VIOLATION in text.lower():
            self.errors.append(text)
            self._arrived.set()

    def mark(self) -> int:
        """the position after the last violation so far, for :meth:`naming` and :meth:`wait_naming`.

        :return: the number of violations reported so far
        :rtype: int
        """
        return len(self.errors)

    def naming(self, fragment: str, *, since: int = 0) -> list[str]:
        """the violations after ``since`` whose text contains ``fragment``, compared case-insensitively.

        :param fragment: a subject (or part of one) the violation must name
        :ptype fragment: str
        :param since: a position from :meth:`mark`
        :ptype since: int
        :return: the matching violations
        :rtype: list[str]
        """
        needle = fragment.lower()
        return [e for e in self.errors[since:] if needle in e.lower()]

    async def wait_naming(self, fragment: str, *, since: int) -> str:
        """wait, without bound, for a violation after ``since`` that names ``fragment``.

        :param fragment: a subject (or part of one) the violation must name
        :ptype fragment: str
        :param since: a position from :meth:`mark`
        :ptype since: int
        :return: the first matching violation
        :rtype: str
        """
        while True:
            self._arrived.clear()
            found = self.naming(fragment, since=since)
            if found:
                return found[0]
            await self._arrived.wait()


@contextlib.asynccontextmanager
async def connect(
    uri: str,
    *,
    user: str,
    password: str,
    violations: Violations,
    inbox_prefix: str = "_INBOX",
) -> AsyncIterator[nats.aio.client.Client]:
    """connect a raw nats-py client whose async errors feed ``violations``.

    :param uri: the server URI
    :ptype uri: str
    :param user: the user name
    :ptype user: str
    :param password: the user's password
    :ptype password: str
    :param violations: where the connection's permissions violations are collected
    :ptype violations: Violations
    :param inbox_prefix: the reply-inbox prefix the user's subscribe grant covers
    :ptype inbox_prefix: str
    :return: the connected client, closed on exit
    :rtype: AsyncIterator[nats.aio.client.Client]
    """
    nc = await nats.connect(
        uri,
        user=user,
        password=password,
        inbox_prefix=inbox_prefix.encode(),
        error_cb=violations.record,
        max_reconnect_attempts=0,
    )
    try:
        yield nc
    finally:
        await nc.close()


class _ViolationLogHandler(logging.Handler):
    """feeds the permissions-violation records of the client's logger into a :class:`Violations`."""

    def __init__(self, violations: Violations) -> None:
        """bind to the collector.

        :param violations: where the violations are collected
        :ptype violations: Violations
        :return: nothing
        :rtype: None
        """
        super().__init__(level=logging.ERROR)
        self._violations = violations

    def emit(self, record: logging.LogRecord) -> None:
        """keep a record's message, which carries the server's own words, if it is a violation.

        :param record: the log record
        :ptype record: logging.LogRecord
        :return: nothing
        :rtype: None
        """
        self._violations.note(record.getMessage())


@contextlib.contextmanager
def watching_client_log(violations: Violations) -> Iterator[None]:
    """collect the permissions violations any :class:`threetears.nats.NatsClient` reports while open.

    The client owns its connection's ``error_cb`` and reports a violation as an ERROR record on
    :data:`CLIENT_LOGGER`, rate-limited to once per subject per client per ten seconds, so one
    refused subject is observed once per client.

    :param violations: where the violations are collected
    :ptype violations: Violations
    :return: nothing; the watch ends on exit
    :rtype: Iterator[None]
    """
    logger = logging.getLogger(CLIENT_LOGGER)
    handler = _ViolationLogHandler(violations)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)


async def refused(
    violations: Violations,
    operation: Awaitable[object],
    *,
    naming: str,
    bound: float = DEFAULT_BOUND_SECONDS,
) -> None:
    """assert the server refuses ``operation``: a permissions violation naming ``naming`` arrives.

    Returns as soon as the violation arrives, cancelling the operation (which the server has
    dropped and will never answer). An operation that fails with a timeout or no-responders error
    is the refusal arriving by a second route: the check then waits, within ``bound``, for the
    violation that proves why.

    :param violations: the collector the connection under test reports to
    :ptype violations: Violations
    :param operation: the refused operation, e.g. ``nc.request(subject, body, timeout=...)``
    :ptype operation: Awaitable[object]
    :param naming: a subject (or part of one) the violation must name, compared case-insensitively
    :ptype naming: str
    :param bound: how long to wait for the server's word
    :ptype bound: float
    :return: nothing
    :rtype: None
    :raises AssertionError: if the operation is answered, or no violation naming ``naming`` arrives
        within ``bound``
    """
    since = violations.mark()
    work = asyncio.ensure_future(operation)
    verdict = asyncio.ensure_future(violations.wait_naming(naming, since=since))
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + bound
        await asyncio.wait({work, verdict}, timeout=bound, return_when=asyncio.FIRST_COMPLETED)
        if work.done():
            outcome = work.exception()
            if outcome is None:
                raise AssertionError(f"expected the server to refuse {naming!r}, but the operation was answered")
            if not isinstance(outcome, REFUSAL_ERRORS):
                raise outcome
            await asyncio.wait({verdict}, timeout=max(0.0, deadline - loop.time()))
        if verdict.done():
            return
        raise AssertionError(
            f"no permissions violation naming {naming!r} arrived within {bound}s, "
            f"and the operation was not answered; violations seen: {violations.errors[since:]}"
        )
    finally:
        for future in (work, verdict):
            future.cancel()
        await asyncio.gather(work, verdict, return_exceptions=True)


async def refused_request(
    violations: Violations,
    nc: nats.aio.client.Client,
    subject: str,
    body: bytes = b"",
    *,
    bound: float = DEFAULT_BOUND_SECONDS,
) -> None:
    """assert the server refuses a raw request on ``subject``; the violation must name ``subject``.

    :param violations: the collector ``nc`` reports to
    :ptype violations: Violations
    :param nc: the connected nats-py client to probe from
    :ptype nc: nats.aio.client.Client
    :param subject: the subject to request on
    :ptype subject: str
    :param body: the request payload (a bypass carries its target in the BODY)
    :ptype body: bytes
    :param bound: how long to wait for the server's word
    :ptype bound: float
    :return: nothing
    :rtype: None
    :raises AssertionError: if the request is answered, or no violation naming ``subject`` arrives
    """
    await refused(violations, nc.request(subject, body, timeout=bound), naming=subject, bound=bound)

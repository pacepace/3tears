"""Sending to many recipients, where one of them failing is ordinary.

**The volume this is for, stated plainly.** The transport this package promotes was
built for transactional email -- a password reset, an address change -- where the send
IS the operation and one exception failing it is correct. A survey reminder blast across
a sample inverts that: one refused address must cost that address and nothing else, and
the caller has to learn WHICH address so it can record the outcome against that person.

**The ceiling, so a consumer knows when they have outgrown it.** This is a single
in-process run over an iterable, with bounded concurrency and an optional pacer:

- It holds no more than `concurrency` messages in flight and pulls the iterable lazily,
  so sample size does not drive memory. A generator over a database cursor is the
  intended shape.
- It has NO durability. A process that dies mid-run loses everything it had not yet
  sent, and knows nothing about what it had. Resumability belongs to the caller, which
  is why `on_failure` fires per recipient rather than at the end: a caller that records
  each outcome as it happens can resume from its own records; one that waits for the
  return value cannot.
- It has no retry. A soft failure is reported once, and re-sending is the caller's
  policy decision, because "retry" for mail means "retry tomorrow", not "retry now".
- It is one process. `SendPacer` is where a rate limit spanning several of them plugs
  in, and :class:`TokenBucketPacer` is that: the platform's existing distributed
  `TokenBucket` over NATS KV, not a second limiter. Concurrency alone caps how many
  conversations are open, never the rate, and it is the rate a relay quota measures.

Past roughly a hundred thousand recipients per run, or wherever a partial run has to
survive a restart, the shape wanted is a durable job per recipient
(`3tears-scheduled-jobs`) with this function reduced to the worker body. That is a
different design, not a bigger number here, and the honest thing is to say so rather
than let this function be discovered to be the wrong tool halfway through a field.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

from threetears.observe import get_logger

from threetears.channels.mail.message import EmailMessage, EmailTransport

if TYPE_CHECKING:
    from threetears.core.coordination.token_bucket import TokenBucket

__all__ = [
    "BatchSendResult",
    "RecipientFailure",
    "SendPacer",
    "SendRateLimited",
    "TokenBucketPacer",
    "send_batch",
]

log = get_logger(__name__)

#: Concurrent sends when the caller expresses no preference. Low on purpose: a relay
#: closes the connection on a sender that opens too many at once, and the failure looks
#: like the relay being down rather than like this number being wrong.
_DEFAULT_CONCURRENCY: Final[int] = 8

#: How many individual failures a result retains. A run against a relay that is down
#: fails for every recipient, and keeping a record per recipient to report that is its
#: own outage. The COUNT stays exact; only the detail is capped.
_DEFAULT_MAX_RECORDED_FAILURES: Final[int] = 100


class SendRateLimited(Exception):
    """A pacer could not free a slot within its wait budget.

    Raised as a per-recipient failure rather than an abort: being paced out is that
    recipient's outcome for this run, and the caller decides whether to try again.
    """


@dataclass(frozen=True, slots=True)
class RecipientFailure:
    """One recipient the run could not deliver to.

    Carries the address deliberately -- a count cannot tell a caller whose disposition
    to mark. It is therefore personal data: pass it to a store, not to a log line.

    :ivar recipient: the address that failed
    :ivar error_type: class name of what went wrong, which survives redaction
    :ivar reason: the failure text, as the transport reported it
    """

    recipient: str
    error_type: str
    reason: str


@dataclass(frozen=True, slots=True)
class BatchSendResult:
    """What one run did.

    :ivar sent: recipients the transport accepted
    :ivar failed: recipients that failed, counted exactly however many are retained
    :ivar failures: retained failure detail, capped at the run's
        `max_recorded_failures`
    """

    sent: int
    failed: int
    failures: tuple[RecipientFailure, ...]

    @property
    def attempted(self) -> int:
        """Recipients this run tried.

        :return: sent plus failed
        :rtype: int
        """
        return self.sent + self.failed


@runtime_checkable
class SendPacer(Protocol):
    """Consulted once before each send, to cap the RATE rather than the concurrency."""

    async def acquire(self) -> None:
        """Block until this send may proceed.

        :return: nothing
        :rtype: None
        :raises SendRateLimited: no slot became available within the wait budget
        """
        ...


class TokenBucketPacer:
    """A :class:`SendPacer` over the platform's distributed token bucket.

    Wraps :class:`threetears.core.coordination.token_bucket.TokenBucket` rather than
    counting locally, because the thing being rated is a relay quota shared by every pod
    that sends. A per-process limiter set to the relay's rate is that rate multiplied by
    the replica count, which is exactly the mistake that gets a sender throttled.
    """

    def __init__(self, bucket: TokenBucket, *, key: str = "outbound_mail", max_wait_seconds: float = 30.0) -> None:
        """
        :param bucket: the shared bucket to claim from
        :ptype bucket: TokenBucket
        :param key: bucket key; use one per relay, since the quota is per relay
        :ptype key: str
        :param max_wait_seconds: how long one send waits for a token before giving up
        :ptype max_wait_seconds: float
        :return: nothing
        :rtype: None
        """
        self._bucket = bucket
        self._key = key
        self._max_wait_seconds = max_wait_seconds

    async def acquire(self) -> None:
        """Claim one token, waiting up to the configured budget.

        :return: nothing
        :rtype: None
        :raises SendRateLimited: no token became available within the budget
        """
        outcome = await self._bucket.claim(self._key, max_wait_seconds=self._max_wait_seconds)
        if not outcome.claimed:
            raise SendRateLimited(
                f"no send token available within {self._max_wait_seconds}s; retry in {outcome.retry_after_seconds:.1f}s"
            )


async def _send_one(
    transport: EmailTransport,
    message: EmailMessage,
    pacer: SendPacer | None,
    on_failure: Callable[[RecipientFailure], Awaitable[None]] | None,
) -> RecipientFailure | None:
    """Send to one recipient, converting any failure into a value rather than a raise.

    Isolating only :class:`~threetears.channels.mail.message.EmailSendError` would abort
    the run on the one failure nobody predicted, which is the opposite of the property
    this function exists to provide. Cancellation is re-raised: cancelling a run must
    stop it, not record a failure per remaining recipient and report a completed run.

    :param transport: the transport each message is handed to
    :ptype transport: EmailTransport
    :param message: the message for this recipient
    :ptype message: EmailMessage
    :param pacer: consulted before the send, when one is configured
    :ptype pacer: SendPacer | None
    :param on_failure: called with the failure as it happens, so a caller can record an
        outcome per recipient rather than waiting for the run to end
    :ptype on_failure: Callable[[RecipientFailure], Awaitable[None]] | None
    :return: the failure, or ``None`` when the transport accepted the message
    :rtype: RecipientFailure | None
    """
    try:
        if pacer is not None:
            await pacer.acquire()
        await transport.send(message)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        failure = RecipientFailure(recipient=message.to, error_type=type(exc).__name__, reason=str(exc))
        if on_failure is not None:
            try:
                await on_failure(failure)
            except asyncio.CancelledError:
                raise
            except Exception:
                # The callback usually writes to a database, and that database being
                # briefly unavailable must not cost the remaining recipients their send.
                # Logged rather than swallowed, because a run whose outcomes silently
                # stopped being recorded is a run nobody can resume.
                log.exception("could not record an outbound-email recipient failure")
        return failure
    return None


async def send_batch(
    transport: EmailTransport,
    messages: Iterable[EmailMessage],
    *,
    concurrency: int = _DEFAULT_CONCURRENCY,
    pacer: SendPacer | None = None,
    on_failure: Callable[[RecipientFailure], Awaitable[None]] | None = None,
    max_recorded_failures: int = _DEFAULT_MAX_RECORDED_FAILURES,
) -> BatchSendResult:
    """Send every message, isolating each recipient's failure from the rest.

    `messages` is pulled lazily and no more than `concurrency` are ever in flight, so a
    generator over a cursor sends a sample of any size without materialising it. See
    this module's docstring for what this deliberately does NOT do -- durability, retry,
    and cross-process rate limiting without a `pacer`.

    :param transport: the transport each message is handed to
    :ptype transport: EmailTransport
    :param messages: messages to send, one per recipient
    :ptype messages: Iterable[EmailMessage]
    :param concurrency: how many sends may be in flight at once
    :ptype concurrency: int
    :param pacer: rate limiter consulted before each send
    :ptype pacer: SendPacer | None
    :param on_failure: called per failed recipient as it happens
    :ptype on_failure: Callable[[RecipientFailure], Awaitable[None]] | None
    :param max_recorded_failures: how many failures the result retains in detail; the
        count is exact regardless
    :ptype max_recorded_failures: int
    :return: what the run sent and what it could not
    :rtype: BatchSendResult
    :raises ValueError: `concurrency` is below one
    :raises asyncio.CancelledError: the run was cancelled
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    sent = 0
    failed = 0
    recorded: list[RecipientFailure] = []
    pending: set[asyncio.Task[RecipientFailure | None]] = set()

    def _collect(done: set[asyncio.Task[RecipientFailure | None]]) -> None:
        nonlocal sent, failed
        for task in done:
            failure = task.result()
            if failure is None:
                sent += 1
            else:
                failed += 1
                if len(recorded) < max_recorded_failures:
                    recorded.append(failure)

    try:
        for message in messages:
            if len(pending) >= concurrency:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                _collect(done)
            pending.add(asyncio.create_task(_send_one(transport, message, pacer, on_failure)))
        while pending:
            done, pending = await asyncio.wait(pending)
            _collect(done)
    finally:
        for task in pending:
            task.cancel()
    return BatchSendResult(sent=sent, failed=failed, failures=tuple(recorded))

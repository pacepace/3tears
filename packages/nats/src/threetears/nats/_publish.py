"""one bounded JetStream publish, for every caller in this package.

A JetStream publish awaits the broker's ``PubAck``. Until 2026-08-18 nothing in
this package bounded that wait, and an ack that never arrived hung the calling
task **forever**. A downstream ingestion fleet sat frozen for ten days behind
one: every stream starved because dispatch is sequential, a backfill held a
fixed cursor across 490 process restarts, and the wedge was in an unrelated
plugin the whole time. What makes it expensive is the diagnostic signature --
0% CPU, no database connections, the NATS read-loop and ping tasks alive and
healthy, not one error line, the process reporting itself up for days. Logs,
metrics and ``py-spy`` all said "idle"; naming the frame took walking the
coroutine chain's ``cr_await`` by hand.

**A caller cannot fix this from outside, which is why it lives here.**
``nats-py``'s ``Client._flush_pending`` ends in::

    except asyncio.CancelledError:
        pass

so a cancellation delivered while a publish sits in the flush path is caught
and discarded. The task does not die. An ``asyncio.wait_for`` wrapped around
the publish therefore fires, cancels, and then waits forever on a task it has
already cancelled -- which is exactly what the incident report observed at the
900-second mark from a 300-second guard. No bound written above this layer can
close that.

So :func:`publish_bounded` uses two, because either alone leaves a hole:

1. **The native bound.** ``timeout`` is handed to ``nats-py``, which bounds the
   ack wait inside its own request machinery and raises rather than depending
   on cancellation working. This is the layer that fires in ordinary trouble
   and produces the useful error.
2. **The abandon bound.** The publish runs as its own task under a slightly
   longer deadline. If that expires, the task is cancelled and **abandoned
   rather than awaited** -- awaiting a coroutine that has already refused
   cancellation is the wedge itself. The caller gets an exception and its event
   loop back.

Layer 2 leaks a task, deliberately and loudly: Python cannot forcibly kill a
coroutine, so the honest choices are "abandon it" or "block the caller", and
blocking the caller is the defect. The orphan is held (so the loop cannot
garbage-collect a pending task and warn about it) and released when it ends.

**A timed-out publish has an unknown outcome, not a failed one.** The broker may
have persisted the message. A caller that retries must assume it might
duplicate, and set ``Nats-Msg-Id`` if that matters.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Final

from threetears.observe import get_logger

from threetears.nats.errors import PublishError, PublishTimeoutError

__all__ = ["publish_bounded", "raise_as_publish_error"]

log = get_logger(__name__)

#: Extra grace beyond the native ack bound before the wrapper stops waiting.
#:
#: The native bound is expected to fire first and produce the better error, so
#: this only ever engages where the native bound cannot reach: a publish wedged
#: somewhere that swallows cancellation.
_ABANDON_GRACE_SECONDS: Final[float] = 5.0

#: Publishes that blew their deadline AND ignored cancellation. Held only so the
#: event loop cannot garbage-collect a still-running task and warn about it;
#: entries remove themselves when they finally end. Module-level rather than
#: per-client because it is a leak registry, not state anyone should consult --
#: nothing reads it, and a caller reaching for it would be managing something it
#: cannot influence.
_abandoned: set[asyncio.Task[Any]] = set()


async def publish_bounded(
    js: Any,
    subject: str,
    payload: bytes,
    *,
    timeout: float,
    headers: Mapping[Any, Any] | None = None,
    stream: str | None = None,
) -> Any:
    """publish to JetStream under a deadline the callee cannot swallow.

    See this module's docstring for why one bound is not enough.

    :param js: a ``nats-py`` JetStream context
    :ptype js: Any
    :param subject: fully-qualified subject to publish to
    :ptype subject: str
    :param payload: serialized message bytes
    :ptype payload: bytes
    :param timeout: ceiling on the publish including its ack, in seconds
    :ptype timeout: float
    :param headers: JetStream headers (dedup id, expected-last-sequence, ...).
        Keys are typed ``Any`` because ``nats.js.api.Header`` is a ``str``-mixin
        ``Enum`` rather than a ``StrEnum``, and ``dict`` is invariant in its key
        type -- a ``dict[Header, str]`` is not a ``dict[str, Any]`` to a type
        checker even though it is one at runtime.
    :ptype headers: Mapping[Any, Any] | None
    :param stream: expected stream name, when the caller pins one
    :ptype stream: str | None
    :return: the broker's ``PubAck``
    :rtype: Any
    :raises PublishTimeoutError: no ack inside the deadline, or the publish
        stopped responding to cancellation
    :raises PublishError: the broker rejected the publish
    """
    kwargs: dict[str, Any] = {"timeout": timeout}
    if headers is not None:
        kwargs["headers"] = dict(headers)
    if stream is not None:
        kwargs["stream"] = stream

    task = asyncio.create_task(js.publish(subject, payload, **kwargs))
    done, _pending = await asyncio.wait({task}, timeout=timeout + _ABANDON_GRACE_SECONDS)

    if task in done:
        try:
            return task.result()
        except asyncio.TimeoutError as exc:
            raise PublishTimeoutError(f"jetstream publish to {subject!r} got no PubAck within {timeout}s") from exc

    # Past the deadline with the task still alive: it is wedged somewhere that does not honour
    # cancellation. Cancel and let go -- `await task` here would re-create the exact hang this
    # function exists to break, inside the code meant to break it.
    task.cancel()
    _abandoned.add(task)
    task.add_done_callback(_abandoned.discard)
    log.error(
        "jetstream publish to %s did not finish within %.1fs and did not respond to cancellation; "
        "abandoning the in-flight task so the caller is not wedged behind it. This is nats-py "
        "swallowing CancelledError in its flush path -- the broker or the stream is very likely "
        "unavailable, and the message may or may not have been persisted.",
        subject,
        timeout + _ABANDON_GRACE_SECONDS,
        extra={"extra_data": {"subject": subject, "timeout_seconds": timeout}},
    )
    raise PublishTimeoutError(
        f"jetstream publish to {subject!r} did not complete within {timeout + _ABANDON_GRACE_SECONDS}s "
        f"and ignored cancellation; the message may or may not have been persisted"
    )


def raise_as_publish_error(subject: str, exc: Exception) -> PublishError:
    """wrap a broker-side publish failure, preserving a timeout as a timeout.

    One place, so a caller adding its own ``except`` cannot accidentally
    reclassify the unknown-outcome case as a known failure.

    :param subject: subject the publish targeted
    :ptype subject: str
    :param exc: the underlying exception
    :ptype exc: Exception
    :return: the error to raise
    :rtype: PublishError
    """
    if isinstance(exc, PublishTimeoutError):
        return exc
    return PublishError(f"jetstream publish to {subject!r} failed: {exc}")

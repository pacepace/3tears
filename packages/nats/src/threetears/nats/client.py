"""canonical NATS client wrapper for 3tears applications.

:class:`NatsClient` is THE single primitive every 3tears app uses to
talk to NATS. it absorbs the lifecycle, dual-phase reconnect-ceiling,
rate-limited error logging, deadletter dispatch, typed publish, and
JetStream KV access that previously lived in three half-overlapping
wrappers (``<upstream-hub>/common/nats.py``,
``threetears.core.cache.kv.NatsKvClient`` (formerly
``cache.nats.NatsClient``), ``<consumer>.runtime.nats_transport``).
there is exactly one canonical wrapper now; :func:`from nats import`
outside this module is flagged by the per-repo enforcement walker.

design notes
------------

- **kw-only API after ``self``**: subscribe / publish / request all
  take their primary args by keyword. the 2026-04-25 production
  footgun (``nc.subscribe(subject, callback)`` silently treating
  ``callback`` as a queue group string under nats-py 2.10+) is
  syntactically impossible to reproduce against this surface.
- **typed publish**: :meth:`publish` accepts a Pydantic ``BaseModel``;
  raw bytes go through :meth:`publish_raw` (explicit escape hatch).
- **typed subscribe**: :meth:`subscribe_typed` auto-decodes incoming
  bytes into a declared Pydantic message type and routes
  validation-failure to deadletter when ``deadletter_on_failure=True``
  (the default).
- **timedelta timeouts**: :meth:`request` / :meth:`request_raw` /
  :meth:`shutdown` all take ``timedelta`` (not raw seconds floats) so
  callers cannot pass a bare ``5`` ambiguously.
- **dual-phase reconnect**: startup is bounded by ``startup_timeout``
  via ``asyncio.wait_for``; once connected, runtime reconnect is
  UNBOUNDED (:data:`RUNTIME_MAX_RECONNECT_ATTEMPTS` ``= -1``) so a NATS
  outage of any duration -- broker restart, node failure, network
  partition -- is ridden out and recovered with no human action,
  instead of the client giving up and self-closing after a finite
  ceiling. subscriptions survive because nats-py replays them under
  their original ``sid`` on every reconnect. inherits the rationale
  from ``<upstream-hub>/common/nats.py`` (deleted as part of this
  consolidation).
- **rate-limited error logging**: identical errors within
  :data:`_ERROR_LOG_RATE_LIMIT_SECONDS` log at debug; distinct errors
  log at error. prevents the 60-DNS-error-per-minute incident pattern.
- **deadletter dispatch**: by default uncaught exceptions in subscribe
  callbacks publish the original message + a structured envelope to
  ``{ns}.deadletter.{original_path}``. opt out per-subscribe with
  ``deadletter_on_failure=False`` (only for sites that already funnel
  errors through their own pipeline).
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Final, TypeVar

import nats
from nats.aio.client import Client as _NatsPyClient
from nats.js.api import AckPolicy as _NatsAckPolicy, ConsumerConfig as _NatsConsumerConfig
from nats.errors import (
    AuthorizationError as _NatsAuthorizationError,
    ConnectionClosedError as _NatsConnectionClosedError,
    NoRespondersError as _NatsNoRespondersError,
    OutboundBufferLimitError as _NatsOutboundBufferLimitError,
    StaleConnectionError as _NatsStaleConnectionError,
    TimeoutError as _NatsTimeoutError,
)
from pydantic import BaseModel, ValidationError
from threetears.observe import get_logger

from threetears.nats.errors import (
    NamespaceNotConfiguredError,
    NatsClientError,
    PublishError,
    RequestError,
    StreamSubjectsOverlapError,
    SubscribeError,
)
from threetears.nats.subjects import Subject, Subjects, set_default_namespace

# JetStream API error code for "subjects overlap with an existing stream": a
# subject belongs to exactly one stream. distinct from "stream name already in
# use" -- the two are conflated by a naive create-or-update. see
# ensure_jetstream_stream.
_JS_ERR_SUBJECTS_OVERLAP = 10065

if TYPE_CHECKING:
    from nats.aio.client import Server as _NatsServer
    from nats.aio.msg import Msg as _NatsMsg
    from nats.aio.subscription import Subscription as _NatsSub

    from threetears.nats.kv import NatsKvBucket
    from threetears.nats.transport import MessageCallback, RawMessageCallback


from threetears.nats.transport import IncomingMessage

__all__ = [
    "DEFAULT_REQUEST_TIMEOUT",
    "DEFAULT_STARTUP_TIMEOUT",
    "DEFAULT_DRAIN_TIMEOUT",
    # resilience-task-03: explicit bounded outbound/pending buffer defaults.
    "DEFAULT_PENDING_SIZE_BYTES",
    "DEFAULT_FLUSHER_QUEUE_SIZE",
    # resilience-task-06: jittered reconnect backoff defaults (thundering-herd defense).
    "DEFAULT_RECONNECT_BACKOFF_BASE_SECONDS",
    "DEFAULT_RECONNECT_BACKOFF_CAP_SECONDS",
    "RUNTIME_MAX_RECONNECT_ATTEMPTS",
    "STARTUP_MAX_RECONNECT_ATTEMPTS",
    "JetStreamPullConsumer",
    "JetStreamPushConsumer",
    "NatsClient",
    "ReconnectCallback",
    "Subscription",
]


log = get_logger(__name__)


_T = TypeVar("_T", bound=BaseModel)

#: an async, argument-less callback a consumer registers via :meth:`NatsClient.add_reconnect_callback`
#: to run after each successful NATS reconnect (e.g. re-mint a short-lived credential whose backing
#: session the broker may have dropped during the outage).
ReconnectCallback = Callable[[], Awaitable[None]]

#: a sync, argument-less PROVIDER returning the current connect auth token. nats-py invokes it fresh
#: on every (re)connect (``Client._connect_command``), so a pod that presents a short-lived identity
#: token hands the provider here and each reconnect re-presents a freshly-valid credential instead of
#: a cached one that has since expired — the connection rides on indefinitely. Sync because nats-py
#: calls it un-awaited; back a network-fetched token with a holder a background task refreshes and
#: return ``holder.get()``.
TokenCallback = Callable[[], str]


# ---------------------------------------------------------------------------
# tunables
# ---------------------------------------------------------------------------

#: max reconnect attempts during startup (asyncio.wait_for enforces wall-time bound).
STARTUP_MAX_RECONNECT_ATTEMPTS: Final[int] = 15

#: max reconnect attempts after first successful connect. ``-1`` means FOREVER: a
#: pod that loses NATS -- broker restart, node failure, network partition of ANY
#: duration -- must keep retrying until NATS returns, never give up and self-close.
#: nats-py treats ``< 0`` as unbounded: in ``_select_next_server`` the
#: ``max_reconnect_attempts > 0`` discard branch is skipped, so the server is never
#: evicted from the pool and ``NoServersError`` -- the one trigger that permanently
#: ``close()``s the client on the reconnect path -- is never raised; subscriptions
#: are replayed on every reconnect under their original ``sid`` so the wrapper's
#: dispatch loops survive untouched. each attempt is paced by ``reconnect_time_wait``
#: (a bounded 2s retry, never a hot spin). the previous finite ceiling (100, ~200s of
#: budget) was the live k8s-resilience bug: any outage longer than the budget closed
#: the client for good with NO auto-recovery -- and because consumer liveness stayed
#: 200, k8s never restarted the wedged pod either. forever-reconnect removes the
#: brick; the consumer's liveness ``is_closed`` check is the last-resort net for the
#: residual non-recoverable close paths (e.g. a persistent auth violation).
RUNTIME_MAX_RECONNECT_ATTEMPTS: Final[int] = -1

#: consecutive Authorization-Violation errors (no intervening successful (re)connect) after which
#: :attr:`NatsClient.is_healthy` reports unhealthy. At the 2s ``reconnect_time_wait``, 3 is ~6s of
#: sustained auth rejection -- long enough to distinguish a wedged credential from a one-off, short
#: enough that a ``/healthz`` keyed on it trips promptly so k8s restarts the pod.
_AUTH_VIOLATION_UNHEALTHY_THRESHOLD: Final[int] = 3

#: resilience-task-03: consecutive outbound-buffer overflow events (no intervening successful publish
#: or (re)connect) after which :attr:`NatsClient.is_healthy` reports unhealthy. Mirrors the
#: auth-violation threshold: nats-py raises ``OutboundBufferLimitError`` synchronously from
#: ``publish`` only while the client is disconnected/reconnecting AND the pending buffer is full (the
#: wedge state). A sustained run of those means the connection is not draining, so a ``/healthz`` keyed
#: on ``is_healthy`` trips and the supervisor (resilience-task-02) restarts the pod. Kept at 3 to match
#: the auth path: long enough that a one-off burst during a brief blip self-clears on the next
#: successful publish, short enough that a real wedge surfaces promptly.
_OUTBOUND_OVERFLOW_UNHEALTHY_THRESHOLD: Final[int] = 3

#: resilience-task-03: explicit bounded pending (outbound) buffer size in bytes handed to nats-py's
#: ``pending_size`` connect option. nats-py's untuned default is 2 MiB
#: (:data:`nats.aio.client.DEFAULT_PENDING_SIZE`); this sets a larger, EXPLICIT 4 MiB bound so a
#: healthy bursty agent (10s heartbeats + per-turn streaming-token publishes) never trips the limit
#: while connected, while still BOUNDING what a disconnected/reconnecting client accumulates before
#: nats-py raises ``OutboundBufferLimitError`` (which the wrapper turns into an ``is_healthy`` signal
#: rather than an unbounded thrash). Set explicitly -- not left to the library default -- so the bound
#: is asserted in tests and tunable per-consumer. resilience-task-06 will re-touch the same options
#: dict to add reconnect backoff/jitter; keep these keys clearly delimited.
DEFAULT_PENDING_SIZE_BYTES: Final[int] = 4 * 1024 * 1024

#: resilience-task-03: explicit bounded flusher queue depth handed to nats-py's ``flusher_queue_size``
#: connect option. nats-py's default is 1024 (:data:`nats.aio.client.DEFAULT_MAX_FLUSHER_QUEUE_SIZE`);
#: this doubles it to 2048 so bursty publish batches queue for the flusher without backpressure under
#: healthy load. Set explicitly for the same asserted-and-tunable reason as the pending bound.
DEFAULT_FLUSHER_QUEUE_SIZE: Final[int] = 2048

#: resilience-task-06: base (seconds) of the per-attempt capped-exponential FULL-JITTER reconnect
#: backoff wired into nats-py's ``reconnect_to_server_handler``. The un-jittered ceiling for a reconnect
#: attempt is ``min(cap, base * 2**server.reconnects)`` and the actual delay is
#: ``uniform(0, ceiling)`` -- full jitter, which de-synchronizes a mass reconnect better than equal
#: jitter. At ``base = 1.0`` the first-attempt delay is uniform in ``[0, 1]`` s, so a fleet of N pods
#: reconnecting on a shared trigger (Hub restart, KEDA scale-out) spreads its handshakes across the
#: window instead of spiking simultaneously. Replaces the fixed 2s ``reconnect_time_wait`` on the
#: RECONNECT path (``_attempt_reconnect``); ``reconnect_time_wait`` still paces the distinct startup
#: server-selection loop (``_select_next_server``). Set explicitly -- not left to a library default --
#: so the bound is asserted in tests and tunable per-consumer.
DEFAULT_RECONNECT_BACKOFF_BASE_SECONDS: Final[float] = 1.0

#: resilience-task-06: cap (seconds) on the un-jittered reconnect-backoff ceiling. Bounds the growth of
#: ``base * 2**server.reconnects`` so a single agent that has been reconnecting for a while still
#: recovers PROMPTLY (its delay never exceeds ``uniform(0, cap)``) -- the anti-pattern the shard calls
#: out is UNBOUNDED backoff that makes single-agent recovery sluggish. 30s balances herd-spread against
#: recovery latency.
DEFAULT_RECONNECT_BACKOFF_CAP_SECONDS: Final[float] = 30.0

#: resilience floor (seconds) the reconnect handler NEVER returns below, so a full-jitter near-zero
#: draw -- or any degenerate/failed delay computation -- can never turn the reconnect path into a
#: 100%-CPU hot spin. Small enough to preserve the jitter spread's fast early retries.
_RECONNECT_BACKOFF_MIN_SECONDS: Final[float] = 0.05

#: safe delay the reconnect handler returns if the delay computation ever RAISES: nats-py calls the
#: handler sync on every attempt and sleeps its return, so a raising handler gives it no delay to sleep
#: and it busy-spins. returning a real, whole-second pace here fails safe instead of hot-looping.
_RECONNECT_BACKOFF_FALLBACK_SECONDS: Final[float] = 1.0

#: pace (seconds) the durable pull-consumer loop waits after a transport error before retrying, so a
#: reconnect-window failure recovers without busy-spinning the CPU.
_PULL_CONSUMER_ERROR_BACKOFF_SECONDS: Final[float] = 1.0

#: default startup timeout (matches platform's ``startup_timeout_seconds`` env var).
DEFAULT_STARTUP_TIMEOUT: Final[timedelta] = timedelta(seconds=30)

#: default request/reply timeout when caller does not pass one explicitly.
DEFAULT_REQUEST_TIMEOUT: Final[timedelta] = timedelta(seconds=5)

#: default drain timeout for graceful shutdown.
DEFAULT_DRAIN_TIMEOUT: Final[timedelta] = timedelta(seconds=30)

#: dedup window for error-callback rate limiting.
_ERROR_LOG_RATE_LIMIT_SECONDS: Final[float] = 10.0


def _full_jitter_backoff(
    attempt: int,
    *,
    base: float,
    cap: float,
    rng: random.Random | None = None,
) -> float:
    """compute a per-attempt capped-exponential FULL-JITTER backoff delay (resilience-task-06).

    the un-jittered ceiling is ``min(cap, base * 2**attempt)`` and the returned delay is a uniform
    random draw in ``[0, ceiling]`` -- FULL jitter, which de-synchronizes a fleet reconnecting on a
    shared trigger (Hub restart, KEDA scale-out) better than equal jitter (equal jitter keeps a
    fixed floor every pod shares; full jitter spreads the whole window). the exponent grows the
    ceiling until ``cap`` clamps it, so early attempts retry fast and a persistent outage backs off
    without ever exceeding ``cap`` -- a single agent still recovers promptly.

    :param attempt: the per-attempt count (nats-py's ``Server.reconnects`` on the socket path, the
        consecutive-failure count on the identity-refresh path); ``0`` for the first attempt
    :ptype attempt: int
    :param base: base backoff (seconds) doubled per attempt before the jitter draw
    :ptype base: float
    :param cap: upper bound (seconds) on the un-jittered ceiling so growth stays bounded
    :ptype cap: float
    :param rng: optional random source (injected in tests for determinism); ``None`` uses the module
        ``random`` -- two independent instances therefore draw INDEPENDENT delays, so they do not
        synchronize
    :ptype rng: random.Random | None
    :return: a jittered delay in seconds within ``[0, min(cap, base * 2**attempt)]`` -- exponent-clamped so a long-running reconnect never overflows the float multiply (callers floor the sleep to avoid a hot spin)
    :rtype: float
    """
    exponent = attempt if attempt >= 0 else 0
    # clamp the exponent so ``2**exponent`` can never overflow the float multiply below. once
    # ``base * 2**exponent >= cap`` the ``min`` clamps to ``cap`` anyway, so a larger exponent is pure
    # waste -- AND an unclamped exponent on a long-running reconnect (``attempt`` past ~1024) makes
    # ``2**attempt`` a >308-digit int, so ``base * that`` raises ``OverflowError`` ("int too large to
    # convert to float"). That escaped the handler, broke every reconnect attempt, and busy-spun the
    # client at 100% CPU. ``ceil(log2(cap/base))`` is the smallest exponent at which the ceiling first
    # reaches ``cap``; clamp there.
    if cap > base > 0:
        exponent = min(exponent, math.ceil(math.log2(cap / base)))
    else:
        exponent = 0
    ceiling = min(cap, base * (2**exponent))
    draw = rng.uniform(0.0, ceiling) if rng is not None else random.uniform(0.0, ceiling)
    return draw


def _make_reconnect_to_server_handler(
    *,
    base: float,
    cap: float,
) -> Callable[[list[_NatsServer], dict[str, Any]], tuple[_NatsServer | None, float]]:
    """build the sync ``reconnect_to_server_handler`` nats-py invokes on each reconnect (resilience-task-06).

    nats-py calls the returned handler on EVERY reconnect attempt (``_attempt_reconnect``), passing a
    snapshot of the eligible servers (each carrying its own ``reconnects`` count) and the current
    server info, and expects back ``(selected_server, callback_delay)``; it then sleeps
    ``callback_delay`` before connecting. the handler selects the first eligible server (the pool is
    already shuffled by nats-py before the call, so this matches the default round-robin selection) and
    returns a FULL-JITTER capped-exponential delay keyed on THAT server's ``reconnects`` -- so the
    per-attempt count drives the backoff and a mass reconnect spreads across the window rather than
    synchronizing. the handler is sync because nats-py calls it un-awaited.

    :param base: base backoff (seconds) for :func:`_full_jitter_backoff`
    :ptype base: float
    :param cap: cap (seconds) on the un-jittered ceiling for :func:`_full_jitter_backoff`
    :ptype cap: float
    :return: a sync handler mapping ``(servers, server_info)`` to ``(selected_server, jittered_delay)``
    :rtype: Callable[[list[_NatsServer], dict[str, Any]], tuple[_NatsServer | None, float]]
    """

    def _handler(
        servers: list[_NatsServer],
        server_info: dict[str, Any],
    ) -> tuple[_NatsServer | None, float]:
        """select the next server and return a full-jitter backoff delay keyed on its reconnect count."""
        selected = servers[0] if servers else None
        try:
            attempt = selected.reconnects if selected is not None else 0
            delay = _full_jitter_backoff(attempt, base=base, cap=cap)
        except Exception:
            # nats-py calls this handler SYNC on every reconnect attempt and sleeps whatever delay it
            # returns; if the computation ever raises, nats-py gets no delay to sleep and busy-spins the
            # client at 100% CPU (the exact failure this handler guards). fail safe to a whole-second pace
            # rather than let an exception escape. (defense-in-depth: _full_jitter_backoff is now
            # overflow-safe, but a raising handler must never be able to hot-spin the reconnect path.)
            delay = _RECONNECT_BACKOFF_FALLBACK_SECONDS
        # floor the delay so a full-jitter near-zero draw can never let nats-py retry fast enough to
        # burn a core against a fast-failing (connection-refused) server. small enough to preserve the
        # jitter spread that de-syncs a reconnecting fleet.
        return selected, max(_RECONNECT_BACKOFF_MIN_SECONDS, delay)

    return _handler


class Subscription:
    """opaque handle returned by :meth:`NatsClient.subscribe`.

    callers pass instances back to :meth:`NatsClient.unsubscribe` to
    drop the subscription. fields prefixed with ``raw_`` /
    ``dispatch_task`` are intentionally public-named (per the
    underscore-stability-contract rule) but excluded from
    ``threetears.nats.__all__`` — they are package-internal, not
    bindable from outside the wrapper, and the wrapper itself
    manipulates them when unsubscribing.

    :param raw_subscription: underlying nats-py subscription
    :ptype raw_subscription: Any
    :param subject: subject this subscription was registered against
    :ptype subject: Subject
    :param dispatch_task: background task driving the message loop
    :ptype dispatch_task: asyncio.Task[None]
    """

    __slots__ = ("raw_subscription", "_subject", "dispatch_task", "_closed")

    def __init__(
        self,
        *,
        raw_subscription: Any,
        subject: Subject,
        dispatch_task: asyncio.Task[None],
    ) -> None:
        self.raw_subscription = raw_subscription
        self._subject = subject
        self.dispatch_task = dispatch_task
        self._closed = False

    @property
    def subject(self) -> Subject:
        """subject this subscription was registered against.

        :return: registered subject
        :rtype: Subject
        """
        return self._subject

    @property
    def is_closed(self) -> bool:
        """whether subscription has been dropped.

        :return: True after :meth:`NatsClient.unsubscribe` has been called
        :rtype: bool
        """
        return self._closed

    def mark_closed(self) -> None:
        """mark subscription as closed.

        called by :meth:`NatsClient.unsubscribe` after dropping the
        underlying nats-py subscription. exposed (no leading
        underscore) as a package-stable api between the wrapper's
        client and Subscription handle; not part of the public
        ``threetears.nats`` surface.

        :return: nothing
        :rtype: None
        """
        self._closed = True

    async def unsubscribe(self) -> None:
        """drop the underlying nats-py subscription and cancel dispatch.

        thin convenience equivalent to
        ``await client.unsubscribe(sub)`` — saves callers (typically
        integration tests with no handle to the parent client at
        teardown time) from re-plumbing the client just to release a
        subscription. idempotent: a second call after the first is a
        no-op.

        :return: nothing
        :rtype: None
        """
        if self._closed:
            return
        try:
            await self.raw_subscription.unsubscribe()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
        self.dispatch_task.cancel()
        try:
            await self.dispatch_task
        except asyncio.CancelledError, Exception:  # noqa: BLE001
            pass
        self._closed = True


class JetStreamPushConsumer:
    """opaque handle returned by :meth:`NatsClient.jetstream_subscribe_durable`.

    fixes a type mismatch bug: :meth:`jetstream_subscribe_durable` used to
    return nats-py's raw JetStream push-subscription object directly (see
    ``js.subscribe(cb=...)``), which callers naturally tried to hand back
    to :meth:`NatsClient.unsubscribe` — but that method expects a
    :class:`Subscription` (``.is_closed`` / ``.mark_closed()``), which the
    raw object does not have, raising ``AttributeError`` at teardown. every
    caller following the pattern documented on :meth:`subscribe` (get a
    handle, pass it to ``unsubscribe()``) hit this the first time they
    actually stopped a durable push consumer.

    this class is NOT a :class:`Subscription` and is NOT passed to
    :meth:`NatsClient.unsubscribe` — call :meth:`stop` directly on the
    handle instead, matching :class:`JetStreamPullConsumer`'s own
    established pattern (each subscription style owns its own handle
    type and teardown method, rather than force-fitting every style
    through one generic ``Subscription``/``unsubscribe()`` pairing that
    only truly fits the plain core-NATS manual-dispatch-loop case
    :meth:`subscribe` uses). nats-py owns the push-subscription's
    delivery loop internally once a callback is supplied to
    ``js.subscribe(cb=...)``, so unlike :class:`Subscription` there is no
    wrapper-owned ``dispatch_task`` to cancel here — :meth:`stop` only
    needs to drop the raw nats-py subscription.

    :param raw_subscription: underlying nats-py JetStream push subscription
    :ptype raw_subscription: Any
    :param subject: subject this subscription was registered against
    :ptype subject: Subject
    :param durable: durable consumer name (diagnostics)
    :ptype durable: str
    """

    __slots__ = ("raw_subscription", "_subject", "_durable", "_closed")

    def __init__(
        self,
        *,
        raw_subscription: Any,
        subject: Subject,
        durable: str,
    ) -> None:
        self.raw_subscription = raw_subscription
        self._subject = subject
        self._durable = durable
        self._closed = False

    @property
    def subject(self) -> Subject:
        """subject this subscription was registered against.

        :return: registered subject
        :rtype: Subject
        """
        return self._subject

    @property
    def durable(self) -> str:
        """durable consumer name this subscription was bound with.

        :return: durable consumer name
        :rtype: str
        """
        return self._durable

    @property
    def is_closed(self) -> bool:
        """whether subscription has been dropped.

        :return: True after :meth:`stop` has been called
        :rtype: bool
        """
        return self._closed

    async def stop(self) -> None:
        """drop the underlying nats-py push subscription. idempotent.

        :return: nothing
        :rtype: None
        """
        if self._closed:
            return
        try:
            await self.raw_subscription.unsubscribe()
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup, diagnostics only
            log.warning(
                "jetstream push consumer unsubscribe failed",
                extra={"extra_data": {"subject": self._subject.path, "durable": self._durable, "error": str(exc)}},
            )
        self._closed = True


class JetStreamPullConsumer:
    """a bound durable PULL subscription that drains a subject in a fetch loop.

    returned by :meth:`NatsClient.jetstream_pull_subscribe`. N of these across
    worker replicas bind the SAME durable name and share the backlog one-of-N:
    JetStream hands each pending message to exactly one fetcher. each instance
    owns a fetch loop (:meth:`run`) that pulls a batch, dispatches every message
    to the handler, and routes a handler raise through the bounded-redelivery
    policy so one poisoned message cannot kill the loop. it holds NO
    server-pushed delivery subject, so an idle instance keeps no open delivery
    flow — the property that makes a delivery worker scale-to-zero eligible.

    :param psub: nats-py pull subscription handle
    :ptype psub: Any
    :param cb: async handler; acks on success/terminal-drop, RAISES to retry
    :ptype cb: Callable[[Any], Awaitable[None]]
    :param redeliver: bound policy applied to a message whose handler raised
    :ptype redeliver: Callable[[Any, BaseException], Awaitable[None]]
    :param durable: durable consumer name (diagnostics)
    :ptype durable: str
    :param subject: subject being consumed (diagnostics)
    :ptype subject: Subject
    :param batch: max messages one fetch pulls
    :ptype batch: int
    :param fetch_timeout_seconds: idle poll cadence (per-fetch wait)
    :ptype fetch_timeout_seconds: float
    """

    def __init__(
        self,
        *,
        psub: Any,
        cb: Callable[[Any], Awaitable[None]],
        redeliver: Callable[[Any, BaseException], Awaitable[None]],
        durable: str,
        subject: Subject,
        batch: int,
        fetch_timeout_seconds: float,
    ) -> None:
        """initialize the pull consumer over its subscription + handlers.

        :param psub: nats-py pull subscription handle
        :ptype psub: Any
        :param cb: async handler; acks on success/terminal-drop, RAISES to retry
        :ptype cb: Callable[[Any], Awaitable[None]]
        :param redeliver: bound bounded-redelivery policy for a raised handler
        :ptype redeliver: Callable[[Any, BaseException], Awaitable[None]]
        :param durable: durable consumer name (diagnostics)
        :ptype durable: str
        :param subject: subject being consumed (diagnostics)
        :ptype subject: Subject
        :param batch: max messages one fetch pulls
        :ptype batch: int
        :param fetch_timeout_seconds: idle poll cadence (per-fetch wait)
        :ptype fetch_timeout_seconds: float
        :return: nothing
        :rtype: None
        """
        self._psub = psub
        self._cb = cb
        self._redeliver = redeliver
        self._durable = durable
        self._subject = subject
        self._batch = batch
        self._fetch_timeout_seconds = fetch_timeout_seconds
        self._stopped = False

    async def fetch_and_process(self) -> int:
        """pull one batch and dispatch each message; return the count processed.

        a fetch that times out with no message returns ``0`` (the idle case) —
        the normal scale-to-zero-friendly poll, not an error. a handler raise is
        routed through the bounded-redelivery policy, never propagated, so one
        poisoned message cannot kill the loop.

        :return: number of messages fetched this cycle
        :rtype: int
        """
        try:
            msgs = await self._psub.fetch(self._batch, timeout=self._fetch_timeout_seconds)
        except _NatsTimeoutError:
            msgs = []
        for msg in msgs:
            try:
                await self._cb(msg)
            except Exception as exc:  # noqa: BLE001 — bounded redelivery owns the outcome; never swallow
                await self._redeliver(msg, exc)
        return len(msgs)

    async def run(self) -> None:
        """loop fetch+dispatch until :meth:`stop` (or task cancellation).

        RESILIENT: a non-timeout transport error from ``fetch`` -- or from the ``nak`` / ``ack`` /
        ``jetstream_publish`` inside the bounded-redelivery policy -- surfaces here during a NATS
        reconnect window. it must NOT escape and kill the consumer task: the delivery worker spawns
        ``run()`` unsupervised (fire-and-forget ``create_task``), so a dead task would SILENTLY stop
        channel delivery until the pod restarts -- the exact "background loop silently stops" failure
        class. catch, log, pace, and retry; the durable JetStream stream retains messages across the
        blip so nothing is lost. only cancellation (shutdown) ends the loop.

        :return: nothing
        :rtype: None
        """
        while not self._stopped:
            try:
                await self.fetch_and_process()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — a transport blip must never kill the consumer
                log.warning(
                    "durable pull consumer cycle failed (durable=%s); retrying after %.1fs: %s",
                    self._durable,
                    _PULL_CONSUMER_ERROR_BACKOFF_SECONDS,
                    exc,
                )
                await asyncio.sleep(_PULL_CONSUMER_ERROR_BACKOFF_SECONDS)

    async def stop(self) -> None:
        """halt the fetch loop and unsubscribe the pull consumer.

        :return: nothing
        :rtype: None
        """
        self._stopped = True
        await self._psub.unsubscribe()


class NatsClient:
    """canonical NATS client wrapper.

    construction goes through :meth:`connect`; the bare constructor is
    not part of the public api. once connected, the client owns the
    underlying nats-py connection until :meth:`shutdown` is called.

    :param raw: underlying nats-py client; populated by :meth:`connect`
    :ptype raw: nats.aio.client.Client
    :param namespace: subject namespace prefix bound at connect time
    :ptype namespace: str
    :param client_name: human-readable label used in nats-py connect options and logs
    :ptype client_name: str
    """

    __slots__ = (
        "_raw",
        "_namespace",
        "_client_name",
        "_subscriptions",
        "_buckets",
        "_kv_lock",
        "_reconnect_callbacks",
        "_health_state",
    )

    def __init__(
        self,
        *,
        raw: _NatsPyClient,
        namespace: str,
        client_name: str,
    ) -> None:
        self._raw = raw
        self._namespace = namespace
        self._client_name = client_name
        self._subscriptions: list[Subscription] = []
        self._buckets: dict[str, NatsKvBucket] = {}
        self._kv_lock = asyncio.Lock()
        # consumer-registered post-reconnect hooks. nats-py exposes a SINGLE reconnect callback slot
        # (wired in :meth:`connect` to a dispatcher that fans out to this list), so the wrapper owns
        # the fan-out here. :meth:`connect` rebinds this to the same list the dispatcher closes over.
        self._reconnect_callbacks: list[ReconnectCallback] = []
        # liveness signal for a PERSISTENT auth-violation reconnect loop (see :meth:`is_healthy`).
        # ``auth_violations`` counts consecutive Authorization-Violation errors since the last
        # successful (re)connect; :meth:`connect` rebinds this to the dict its error/reconnect
        # dispatchers close over, so the count reflects the live connection.
        # resilience-task-03: ``overflow_events`` counts consecutive outbound-buffer overflows at the
        # publish boundary since the last successful publish/(re)connect; folded into :meth:`is_healthy`.
        self._health_state: dict[str, int] = {"auth_violations": 0, "overflow_events": 0}

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    @classmethod
    async def connect(
        cls,
        *,
        nats_url: str,
        nats_subject_namespace: str,
        client_name: str,
        cluster_urls: list[str] | None = None,
        auth_token: TokenCallback | None = None,
        user_credentials: str | None = None,
        user: str | None = None,
        password: str | None = None,
        inbox_prefix: str | None = None,
        startup_timeout: timedelta = DEFAULT_STARTUP_TIMEOUT,
        verify_jetstream: bool = True,
        pending_size: int = DEFAULT_PENDING_SIZE_BYTES,
        flusher_queue_size: int = DEFAULT_FLUSHER_QUEUE_SIZE,
        reconnect_backoff_base: float = DEFAULT_RECONNECT_BACKOFF_BASE_SECONDS,
        reconnect_backoff_cap: float = DEFAULT_RECONNECT_BACKOFF_CAP_SECONDS,
    ) -> NatsClient:
        """connect to NATS and return a ready :class:`NatsClient`.

        wraps ``nats.connect`` with a wall-time-bounded startup
        (``startup_timeout``) and the dual-phase reconnect-ceiling
        pattern. on success binds the configured namespace prefix on
        :class:`Subjects` so subject builders pick up the correct env
        prefix without callers having to thread it through.

        :param nats_url: primary NATS server URL (e.g. ``nats://localhost:4222``)
        :ptype nats_url: str
        :param nats_subject_namespace: subject namespace prefix; bound on :class:`Subjects` for the process
        :ptype nats_subject_namespace: str
        :param client_name: human-readable client name reported to NATS server
        :ptype client_name: str
        :param cluster_urls: optional additional cluster member URLs
        :ptype cluster_urls: list[str] | None
        :param auth_token: optional NATS auth-token PROVIDER — a sync, zero-arg callable returning the
            current token. under decentralized auth (platform-auth A) it returns the pod's short-lived
            identity token; nats-py invokes it on every (re)connect, so each reconnect re-presents a
            freshly-valid credential rather than a cached one that has since expired. the NATS server
            forwards the returned token to the auth-callout responder, which verifies it and mints the
            connection's user JWT + subject permissions. ``None`` leaves token auth off.
        :ptype auth_token: TokenCallback | None
        :param user_credentials: optional path to a NATS ``.creds`` file (decentralized-auth static
            credentials). used for principals provisioned with standing creds rather than the
            auth-callout path; ``None`` leaves credential auth off.
        :ptype user_credentials: str | None
        :param user: optional NATS username for centralized config-mode static auth (server
            ``authorization.users``). each platform service connects with its OWN user/password so
            the server applies that user's least-privilege subject permissions; pairs with
            ``password``. ``None`` leaves username/password auth off.
        :ptype user: str | None
        :param password: optional NATS password paired with ``user`` for config-mode static auth.
            ``None`` leaves username/password auth off.
        :ptype password: str | None
        :param inbox_prefix: optional request/reply inbox prefix. decentralized auth scopes each
            principal to its OWN inbox (e.g. ``_INBOX_agent_pod_{pod_id}``) instead of the shared
            global ``_INBOX`` tree, so a responder's replies cannot be observed cross-principal.
            ``None`` keeps the nats-py default ``_INBOX``.
        :ptype inbox_prefix: str | None
        :param startup_timeout: max wall time to spend obtaining first successful connection
        :ptype startup_timeout: timedelta
        :param verify_jetstream: when True (default) verify JetStream is reachable post-connect
        :ptype verify_jetstream: bool
        :param pending_size: explicit bounded outbound/pending buffer size in bytes handed to nats-py's
            ``pending_size`` option (resilience-task-03). defaults to :data:`DEFAULT_PENDING_SIZE_BYTES`
            (4 MiB, deliberately above nats-py's 2 MiB library default). bounds what a
            disconnected/reconnecting client accumulates before nats-py raises
            ``OutboundBufferLimitError`` -- which the wrapper turns into an :attr:`is_healthy` signal.
        :ptype pending_size: int
        :param flusher_queue_size: explicit bounded flusher queue depth handed to nats-py's
            ``flusher_queue_size`` option (resilience-task-03). defaults to
            :data:`DEFAULT_FLUSHER_QUEUE_SIZE` (2048).
        :ptype flusher_queue_size: int
        :param reconnect_backoff_base: base (seconds) of the per-attempt capped-exponential FULL-JITTER
            reconnect backoff wired into nats-py's ``reconnect_to_server_handler`` (resilience-task-06).
            defaults to :data:`DEFAULT_RECONNECT_BACKOFF_BASE_SECONDS` (1.0). the per-attempt delay is
            ``uniform(0, min(reconnect_backoff_cap, base * 2**server.reconnects))`` so a mass reconnect
            spreads out instead of synchronizing the fleet on a shared trigger.
        :ptype reconnect_backoff_base: float
        :param reconnect_backoff_cap: cap (seconds) on the un-jittered reconnect-backoff ceiling
            (resilience-task-06). defaults to :data:`DEFAULT_RECONNECT_BACKOFF_CAP_SECONDS` (30.0);
            bounds the exponential growth so a single agent still recovers promptly.
        :ptype reconnect_backoff_cap: float
        :return: connected and ready NATS client
        :rtype: NatsClient
        :raises NatsClientError: if connection fails, times out, or JetStream verification fails
        """
        if not nats_url:
            raise NatsClientError("nats_url must be non-empty")
        if not client_name:
            raise NatsClientError("client_name must be non-empty")
        if not nats_subject_namespace:
            raise NamespaceNotConfiguredError(
                "nats_subject_namespace must be non-empty: every client passes its "
                "own subject namespace explicitly. there is no default -- sharing a "
                "default subject space is what lets two services collide on a shared "
                "NATS cluster."
            )

        servers = [nats_url]
        if cluster_urls:
            servers.extend(u.strip() for u in cluster_urls if u.strip())

        # nats-py reads its single ``reconnected_cb`` slot at reconnect time, but the wrapper instance
        # does not exist yet (it is built after ``nats.connect`` returns). bridge the gap with a list
        # the dispatcher closes over now and the instance adopts below, so consumer callbacks
        # registered post-connect via :meth:`add_reconnect_callback` are dispatched on every reconnect.
        reconnect_callbacks: list[ReconnectCallback] = []
        # bridged like reconnect_callbacks: the error/reconnect dispatchers below close over this dict
        # now; the instance adopts it after construction so :attr:`is_healthy` reads the live count.
        # resilience-task-03: seed ``overflow_events`` alongside ``auth_violations``.
        health_state: dict[str, int] = {"auth_violations": 0, "overflow_events": 0}

        async def _dispatch_reconnected() -> None:
            """fan a reconnect out to the wrapper log + every consumer-registered callback."""
            health_state["auth_violations"] = 0  # a successful (re)connect clears the wedged-auth signal
            # resilience-task-03: a successful (re)connect also clears the outbound-overflow signal --
            # the transport is healthy again, so any buffered-full streak is stale.
            health_state["overflow_events"] = 0
            await _on_reconnected()
            for callback in list(reconnect_callbacks):
                try:
                    await callback()
                except Exception as exc:  # noqa: BLE001 — one bad hook must not abort the others
                    # NOSILENT: a failing reconnect hook is logged so a recurring failure surfaces;
                    # it must never break the reconnect callback chain or the nats-py reconnect path.
                    log.warning("reconnect callback failed: %s", exc)

        async def _dispatch_error(exc: Exception) -> None:
            """log via the rate-limited handler AND track a persistent auth violation for is_healthy."""
            # count the violation BEFORE the await: _on_error may suspend, and if a _dispatch_reconnected
            # reset interleaves at that suspension point a post-reset stale += 1 could survive, leaving a
            # phantom count after a healthy reconnect. Incrementing first keeps the counter honest within a
            # run of failures; the next successful reconnect always resets it to 0.
            if _is_authorization_violation(exc):
                health_state["auth_violations"] += 1
            await _on_error(exc)

        options: dict[str, object] = {
            "name": client_name,
            "allow_reconnect": True,
            "max_reconnect_attempts": RUNTIME_MAX_RECONNECT_ATTEMPTS,
            "reconnect_time_wait": 2,
            "reconnected_cb": _dispatch_reconnected,
            "disconnected_cb": _on_disconnected,
            "error_cb": _dispatch_error,
            # resilience-task-03: bound the outbound/pending buffer EXPLICITLY (not the untuned
            # nats-py default). overflow raises OutboundBufferLimitError at the publish boundary,
            # caught below and folded into is_healthy.
            "pending_size": pending_size,
            "flusher_queue_size": flusher_queue_size,
            # resilience-task-06: per-attempt capped-exponential FULL-JITTER reconnect backoff. nats-py
            # calls this handler on EVERY reconnect attempt (``_attempt_reconnect``), passing a snapshot
            # of eligible servers (each carrying its ``reconnects`` count) and receiving
            # ``(selected_server, callback_delay)``; it then sleeps ``callback_delay`` before connecting.
            # This REPLACES the fixed 2s ``reconnect_time_wait`` on the reconnect path so a mass
            # reconnect (Hub restart, KEDA scale-out) spreads across the backoff window instead of
            # thundering-herd-ing the Hub/NATS. ``reconnect_time_wait`` still paces the DISTINCT startup
            # server-selection loop (``_select_next_server``), so it is left in place above.
            "reconnect_to_server_handler": _make_reconnect_to_server_handler(
                base=reconnect_backoff_base,
                cap=reconnect_backoff_cap,
            ),
        }
        if auth_token:
            # store the provider UNWRAPPED: nats-py's _connect_command invokes it on every
            # (re)connect, so each reconnect presents a freshly-minted token, never a cached one.
            options["token"] = auth_token
        if user_credentials:
            options["user_credentials"] = user_credentials
        if user:
            options["user"] = user
        if password:
            options["password"] = password
        if inbox_prefix:
            # nats-py takes the inbox prefix as bytes; scope it per-principal so request/reply
            # inboxes never share the global `_INBOX` tree across principals.
            options["inbox_prefix"] = inbox_prefix.encode("ascii")

        started_at = time.monotonic()
        try:
            raw_client = await asyncio.wait_for(
                _establish_connection(servers, options, nats_url),
                timeout=startup_timeout.total_seconds(),
            )
        except TimeoutError as exc:
            elapsed = time.monotonic() - started_at
            raise NatsClientError(
                f"failed to connect to NATS at {nats_url} within "
                f"{startup_timeout.total_seconds():.1f}s "
                f"(elapsed={elapsed:.1f}s)"
            ) from exc

        if verify_jetstream:
            await _verify_jetstream(raw_client, nats_url)

        # bind namespace on Subjects ContextVar so every subject built
        # downstream picks up the correct prefix without threading it
        # through every call site.
        set_default_namespace(nats_subject_namespace)

        log.info(
            "NATS connected",
            extra={
                "extra_data": {
                    "url": nats_url,
                    "namespace": nats_subject_namespace,
                    "client_name": client_name,
                    "jetstream_verified": verify_jetstream,
                }
            },
        )

        client = cls(raw=raw_client, namespace=nats_subject_namespace, client_name=client_name)
        # adopt the SAME list the dispatcher closes over, so ``add_reconnect_callback`` appends are
        # visible to the already-installed ``reconnected_cb``.
        client._reconnect_callbacks = reconnect_callbacks
        # adopt the SAME dict the error/reconnect dispatchers close over, so :attr:`is_healthy` reads
        # the live auth-violation count.
        client._health_state = health_state
        return client

    def add_reconnect_callback(self, callback: ReconnectCallback) -> None:
        """register an async callback invoked after each successful NATS reconnect.

        the wrapper rides out an outage of any duration (unbounded runtime reconnect) and replays
        subscriptions automatically, but state the BROKER holds for this connection -- e.g. a
        Hub-side session backing a short-lived credential -- may have been dropped meanwhile. a
        consumer registers a hook here to re-establish such state (re-handshake, re-mint a token)
        once the connection is back. callbacks run in registration order; one raising is logged and
        does not stop the others or the reconnect path.

        :param callback: an argument-less coroutine function run after each reconnect
        :ptype callback: ReconnectCallback
        :return: nothing
        :rtype: None
        """
        self._reconnect_callbacks.append(callback)

    @property
    def namespace(self) -> str:
        """subject namespace prefix bound at connect time.

        :return: namespace prefix (e.g. ``3tears``)
        :rtype: str
        """
        return self._namespace

    @property
    def client_name(self) -> str:
        """human-readable client name reported to NATS server.

        :return: client name
        :rtype: str
        """
        return self._client_name

    @property
    def is_connected(self) -> bool:
        """whether underlying nats-py client reports connected state.

        :return: True if connected
        :rtype: bool
        """
        return bool(self._raw.is_connected)

    @property
    def is_closed(self) -> bool:
        """whether underlying nats-py client is closed.

        :return: True if closed
        :rtype: bool
        """
        return bool(self._raw.is_closed)

    @property
    def is_healthy(self) -> bool:
        """whether the connection is NOT stuck in a persistent auth-violation or outbound-overflow loop.

        Forever-reconnect (:data:`RUNTIME_MAX_RECONNECT_ATTEMPTS`) deliberately rides out network
        drops of any duration, so it also rides out a **persistent auth violation** -- a server that
        rejects the credential on every reconnect attempt (an expired/misconfigured/revoked identity
        token). That never ``close()``s the client, so :attr:`is_closed` stays ``False`` and a
        liveness probe keyed only on ``is_closed`` never trips -- the pod wedges "alive" forever with
        a dead data plane (the exact 46h scriob outage). This returns ``False`` once
        :data:`_AUTH_VIOLATION_UNHEALTHY_THRESHOLD` consecutive Authorization-Violation errors have
        landed with no intervening successful (re)connect, so a ``/healthz`` keyed on it fails and k8s
        restarts the pod (a fresh connect re-mints the credential). A network drop -- no auth
        violation -- does NOT trip this; forever-reconnect still owns that path.

        resilience-task-03 folds a SECOND signal in: once
        :data:`_OUTBOUND_OVERFLOW_UNHEALTHY_THRESHOLD` consecutive outbound-buffer overflows have
        landed at the publish boundary with no intervening successful publish/(re)connect, the
        connection is wedged with a full pending buffer (disconnected/reconnecting and not draining) --
        the ``outbound buffer limit exceeded`` thrash. That flips unhealthy too, so the same
        supervised-restart path (resilience-task-02) recovers it rather than the pod thrashing forever.

        :return: ``False`` when stuck being auth-rejected OR overflowing the outbound buffer; ``True`` otherwise.
        :rtype: bool
        """
        return (
            self._health_state["auth_violations"] < _AUTH_VIOLATION_UNHEALTHY_THRESHOLD
            and self._health_state["overflow_events"] < _OUTBOUND_OVERFLOW_UNHEALTHY_THRESHOLD
        )

    @property
    def overflow_events(self) -> int:
        """current consecutive outbound-buffer overflow count (resilience-task-03).

        number of ``OutboundBufferLimitError`` events raised at the publish boundary with no
        intervening successful publish / (re)connect -- reset to 0 on either. folds into
        :attr:`is_healthy` at :data:`_OUTBOUND_OVERFLOW_UNHEALTHY_THRESHOLD`. exposed publicly
        so a consumer's health/metrics surface (the SDK ``outbound_overflow_events`` gauge) can
        read it without binding to the private health-state dict.

        :return: consecutive outbound-overflow count since the last successful publish/connect
        :rtype: int
        """
        return self._health_state["overflow_events"]

    async def ping(self, *, timeout: float = 2.0) -> bool:
        """force a server round-trip to verify the broker is responsive.

        unlike :attr:`is_connected` (which only reports the local
        socket-state cached by nats-py), this awaits an actual
        round-trip via ``nats-py.flush()`` -- a stale socket that
        the OS hasn't yet timed out can report ``is_connected=True``
        long after the broker has gone away. consumers building
        ``/healthz`` endpoints should call ``ping()``.

        :param timeout: seconds to wait for the round-trip before
            treating as unhealthy.
        :ptype timeout: float
        :return: True if the server responded within the timeout,
            False on timeout or any nats-py error.
        :rtype: bool
        """
        if not self._raw.is_connected:
            return False
        try:
            # nats-py annotates flush(timeout: int) but its body waits via asyncio.wait_for, which
            # accepts a float; the wrapper deliberately exposes sub-second float timeouts (a 1.5s
            # health-check ping), so the int annotation is over-strict, not a real mismatch.
            await self._raw.flush(timeout=timeout)  # type: ignore[arg-type]
        except Exception:
            return False
        return True

    async def reconnect(self) -> None:
        """force a clean reconnect of the underlying connection, re-running server auth.

        nats-py exposes no public force-reconnect. this drives its OWN op-error reconnect path
        (``nats.aio.client.Client._process_op_err``) on a still-CONNECTED client: that transitions
        the client to ``RECONNECTING`` and spawns ``_attempt_reconnect``, which drops + re-opens the
        transport, re-runs the server handshake -- under decentralized auth (platform-auth A) this
        re-runs the Hub auth-callout, minting a FRESH user JWT with full TTL -- and replays every
        live subscription under its original ``sid``. the registered
        :meth:`add_reconnect_callback` hooks then fire.

        the motivating use is **proactive NATS-JWT re-auth**: a pod's auth-callout-minted user JWT
        is short-lived, and at expiry the server sends an ``-ERR`` that nats-py routes STRAIGHT to a
        terminal ``_close`` (it never enters ``_attempt_reconnect``, so forever-reconnect --
        :data:`RUNTIME_MAX_RECONNECT_ATTEMPTS` -- does NOT cover it). triggering this reconnect a
        margin BEFORE expiry re-auths while the current JWT is still valid, so the connection never
        reaches that terminal close. the SAME underlying ``_raw`` object is reused, so consumers
        holding :attr:`raw` (e.g. the L3 proxy backend) stay valid across the cycle.

        :return: nothing
        :rtype: None
        :raises NatsClientError: if the client is already terminally closed -- a closed connection
            cannot be reconnected in place and the caller must establish a fresh client (the
            reactive self-heal path)
        """
        raw = self._raw
        if raw.is_closed:
            raise NatsClientError(
                "cannot reconnect a closed NATS client; a closed connection requires a fresh connect",
            )
        if not raw.is_connected:
            # nats-py is already mid-reconnect (forever-reconnect after a drop); a forced trigger is
            # moot AND, via _process_op_err's not-connected else-branch, would _close the client.
            log.debug("reconnect() skipped: client not currently connected (a reconnect is already in flight)")
            return
        # rationale: _process_op_err is nats-py's single internal entry point that, on a CONNECTED
        # client, transitions to RECONNECTING and spawns _attempt_reconnect (drop+reopen transport,
        # re-run auth-callout, replay subs under their sid, fire reconnected_cb). we synthesize a
        # StaleConnectionError -- the connection is about to go stale as the user JWT nears expiry --
        # so the reconnect+re-auth happens proactively. the is_connected guard above keeps us out of
        # the method's else-branch, which would _close. nats-py 2.x exposes no public alternative.
        await raw._process_op_err(_NatsStaleConnectionError())  # noqa: SLF001 -- no public force-reconnect; see rationale above

    @property
    def raw(self) -> _NatsPyClient:
        """direct access to underlying nats-py client.

        intentionally NOT named ``_raw``: this is a public escape
        hatch for the small set of nats-py features the wrapper does
        not yet cover (custom JetStream stream config, fine-grained
        flow-control, etc.). every use of this property is reviewed
        in code review and should come with a ``# rationale: ...``
        comment justifying why the wrapper api was insufficient.

        :return: underlying nats-py client
        :rtype: nats.aio.client.Client
        """
        return self._raw

    async def shutdown(self, *, drain_timeout: timedelta = DEFAULT_DRAIN_TIMEOUT) -> None:
        """gracefully drain subscriptions and close connection.

        unsubscribes every tracked :class:`Subscription`, drains the
        underlying nats-py client (waits for in-flight messages to
        process), and closes. idempotent — second call is a no-op.

        :param drain_timeout: max time to wait for drain
        :ptype drain_timeout: timedelta
        :return: nothing
        :rtype: None
        """
        if self._raw.is_closed:
            return
        for sub in list(self._subscriptions):
            try:
                await self.unsubscribe(sub)
            except Exception as exc:  # noqa: BLE001 — diag only
                log.warning(
                    "subscription drain failed",
                    extra={"extra_data": {"subject": sub.subject.path, "error": str(exc)}},
                )
        try:
            await asyncio.wait_for(
                self._raw.drain(),
                timeout=drain_timeout.total_seconds(),
            )
            log.info("NATS drained and closed", extra={"extra_data": {"client_name": self._client_name}})
        except asyncio.TimeoutError, TimeoutError:
            log.warning(
                "NATS drain exceeded timeout; forcing close",
                extra={
                    "extra_data": {
                        "client_name": self._client_name,
                        "drain_timeout_seconds": drain_timeout.total_seconds(),
                    }
                },
            )
            if not self._raw.is_closed:
                await self._raw.close()
        except Exception as exc:  # noqa: BLE001 — diag only
            log.warning(
                "NATS drain failed; forcing close",
                extra={"extra_data": {"client_name": self._client_name, "error": str(exc)}},
            )
            if not self._raw.is_closed:
                await self._raw.close()

    async def __aenter__(self) -> NatsClient:
        """context-manager entry — returns self.

        :return: self
        :rtype: NatsClient
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """context-manager exit — graceful shutdown.

        :param exc_type: exception type raised in body, if any
        :ptype exc_type: type[BaseException] | None
        :param exc: exception instance raised in body, if any
        :ptype exc: BaseException | None
        :param tb: traceback for exception, if any
        :ptype tb: TracebackType | None
        :return: nothing
        :rtype: None
        """
        await self.shutdown()

    # ------------------------------------------------------------------
    # boundary helpers (publish / flush / request)
    # ------------------------------------------------------------------

    async def flush(self, timeout: float = 2.0) -> None:
        """flush pending publishes through the underlying nats-py client.

        thin pass-through for callers (typically integration tests)
        that need a publish-side memory barrier before sending the
        triggering request: subscribe -> flush -> publish-the-trigger
        ensures the subscription is registered server-side before any
        message arrives.

        :param timeout: seconds to wait for the flush ack
        :ptype timeout: float
        :return: nothing
        :rtype: None
        """
        # nats-py annotates flush(timeout: int) but waits via asyncio.wait_for (accepts float); the
        # wrapper's float timeout is intentional, so the int annotation is over-strict.
        await self._raw.flush(timeout=timeout)  # type: ignore[arg-type]
        # resilience-task-03: a successful flush drained the outbound buffer -- clear the overflow streak.
        self._health_state["overflow_events"] = 0

    def _note_publish_success(self) -> None:
        """clear the outbound-overflow streak after a successful publish (resilience-task-03).

        a publish that reaches the wire means the outbound buffer accepted it, so any prior
        ``OutboundBufferLimitError`` streak is stale -- the connection is draining again. Keeps the
        overflow signal from tripping :attr:`is_healthy` on a transient burst that self-clears.

        :return: nothing
        :rtype: None
        """
        self._health_state["overflow_events"] = 0

    def _note_if_outbound_overflow(self, exc: Exception) -> None:
        """count an outbound-buffer overflow raised synchronously at the publish boundary (resilience-task-03).

        nats-py raises :class:`nats.errors.OutboundBufferLimitError` from ``publish``/``request`` ONLY
        while disconnected/reconnecting with a full pending buffer -- exactly the wedge state. It is
        NOT delivered to the error callback, so it cannot be counted in ``_dispatch_error`` (the
        auth-violation path); it must be caught here, at the publish boundary. Incrementing
        ``overflow_events`` folds into :attr:`is_healthy` so a sustained overflow flips the client
        unhealthy (feeding resilience-task-02's supervised restart) instead of thrashing unbounded.
        Non-overflow errors are ignored here -- the caller's generic handler wraps them.

        :param exc: the exception the underlying nats-py ``publish``/``request`` raised.
        :ptype exc: Exception
        :return: nothing
        :rtype: None
        """
        if _is_outbound_overflow(exc):
            self._health_state["overflow_events"] += 1
            log.warning(
                "NATS outbound buffer overflow at publish boundary",
                extra={
                    "extra_data": {
                        "client_name": self._client_name,
                        "overflow_events": self._health_state["overflow_events"],
                    }
                },
            )

    async def publish(
        self,
        *args: Any,
        subject: Subject | str | None = None,
        message: BaseModel | None = None,
        reply_to: Subject | str | None = None,
    ) -> None:
        """publish to a NATS subject.

        primary (canonical) form is keyword-only with a Pydantic
        message:: ``await nc.publish(subject=Subject, message=Model)``;
        serialization happens via ``model_dump_json()``.

        positional shorthand ``nc.publish(subject_str, payload_bytes)``
        is also accepted for parity with raw nats-py — every
        integration test (and several legacy fanout sites) calls this
        shape after a raw ``msg.reply`` lookup. callers needing the
        kw-only typed form keep working unchanged; the shorthand
        routes through :meth:`publish_raw_reply` semantics
        (str subject, bytes payload, no Pydantic in the loop).

        :param args: optional positional ``(subject_str, payload_bytes)``
            shorthand for raw publishes
        :ptype args: Any
        :param subject: target subject (kw form). bare ``str`` is
            auto-wrapped via :meth:`Subject.raw` for the
            test-ergonomic path.
        :ptype subject: Subject | str | None
        :param message: typed Pydantic message (kw form, mutually
            exclusive with positional payload bytes)
        :ptype message: BaseModel | None
        :param reply_to: optional reply subject for request-style
            transports; bare ``str`` is auto-wrapped
        :ptype reply_to: Subject | str | None
        :return: nothing
        :rtype: None
        :raises PublishError: if underlying publish fails
        """
        # positional shorthand: (subject_str_or_subject, payload_bytes)
        if args:
            if len(args) != 2:
                raise PublishError(
                    f"publish positional form requires exactly (subject, payload_bytes); got {len(args)} args",
                )
            pos_subject, pos_payload = args
            if not isinstance(pos_payload, bytes | bytearray | memoryview):
                raise PublishError(
                    f"publish positional payload must be bytes-like; got {type(pos_payload).__name__}",
                )
            sub = pos_subject if isinstance(pos_subject, Subject) else Subject.raw(str(pos_subject))
            await self._publish_bytes(subject=sub, payload=bytes(pos_payload), reply_to=None)
            return
        if subject is None or message is None:
            raise PublishError(
                "publish requires either positional (subject, payload_bytes) or kwargs subject= and message=",
            )
        sub = subject if isinstance(subject, Subject) else Subject.raw(subject)
        rt = reply_to if (reply_to is None or isinstance(reply_to, Subject)) else Subject.raw(reply_to)
        payload = message.model_dump_json().encode("utf-8")
        await self._publish_bytes(subject=sub, payload=payload, reply_to=rt)

    async def publish_raw(
        self,
        *,
        subject: Subject,
        payload: bytes,
        reply_to: Subject | None = None,
    ) -> None:
        """publish raw bytes to subject.

        explicit escape hatch for sites that already serialize
        upstream (e.g. streaming token forwarders) or for non-Pydantic
        payloads. prefer :meth:`publish` for new code.

        :param subject: target subject
        :ptype subject: Subject
        :param payload: pre-serialized message bytes
        :ptype payload: bytes
        :param reply_to: optional reply subject
        :ptype reply_to: Subject | None
        :return: nothing
        :rtype: None
        :raises PublishError: if underlying publish fails
        """
        await self._publish_bytes(subject=subject, payload=payload, reply_to=reply_to)

    async def publish_reply(
        self,
        *,
        reply_subject: str,
        message: BaseModel,
    ) -> None:
        """publish a typed Pydantic message to a request's reply subject.

        the reply subject is whatever opaque inbox nats-py provided on
        the originating message (``msg.reply``). it is not constructable
        via :class:`Subjects` and is therefore typed as a bare string
        here.

        :param reply_subject: opaque reply subject from request envelope
        :ptype reply_subject: str
        :param message: typed Pydantic message to serialize and send
        :ptype message: BaseModel
        :return: nothing
        :rtype: None
        :raises PublishError: if underlying publish fails
        """
        if not reply_subject:
            raise PublishError("reply_subject must be non-empty")
        payload = message.model_dump_json().encode("utf-8")
        try:
            await self._raw.publish(reply_subject, payload)
        except Exception as exc:
            self._note_if_outbound_overflow(exc)  # resilience-task-03
            raise PublishError(f"publish_reply failed: subject={reply_subject}: {exc}") from exc
        self._note_publish_success()  # resilience-task-03

    async def publish_raw_reply(
        self,
        *,
        reply_subject: str,
        payload: bytes,
    ) -> None:
        """publish raw bytes to a request's reply subject.

        explicit escape hatch for sites that already serialize upstream
        — most commonly transparent proxies that forward an opaque
        downstream response back to the original requester without
        decoding the body. prefer :meth:`publish_reply` for sites that
        own the response type.

        :param reply_subject: opaque reply subject from request envelope
        :ptype reply_subject: str
        :param payload: pre-serialized response bytes
        :ptype payload: bytes
        :return: nothing
        :rtype: None
        :raises PublishError: if underlying publish fails
        """
        if not reply_subject:
            raise PublishError("reply_subject must be non-empty")
        try:
            await self._raw.publish(reply_subject, payload)
        except Exception as exc:
            self._note_if_outbound_overflow(exc)  # resilience-task-03
            raise PublishError(f"publish_raw_reply failed: subject={reply_subject}: {exc}") from exc
        self._note_publish_success()  # resilience-task-03

    async def _publish_bytes(
        self,
        *,
        subject: Subject,
        payload: bytes,
        reply_to: Subject | None,
    ) -> None:
        """common publish path used by :meth:`publish` / :meth:`publish_raw`.

        :param subject: target subject
        :ptype subject: Subject
        :param payload: serialized bytes
        :ptype payload: bytes
        :param reply_to: optional reply subject
        :ptype reply_to: Subject | None
        :return: nothing
        :rtype: None
        :raises PublishError: if underlying publish fails
        """
        try:
            if reply_to is None:
                await self._raw.publish(subject.path, payload)
            else:
                await self._raw.publish(subject.path, payload, reply=reply_to.path)
        except Exception as exc:
            # resilience-task-03: an outbound-buffer overflow is counted into is_healthy here (the
            # publish boundary) -- it never reaches error_cb. re-raised as a typed PublishError so the
            # caller gets the wrapper's contract, not an unhandled nats-py raise crashing the coroutine.
            self._note_if_outbound_overflow(exc)
            raise PublishError(f"publish failed: subject={subject.path}: {exc}") from exc
        self._note_publish_success()

    # ------------------------------------------------------------------
    # subscribe
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        subject: Subject | str | None = None,
        *,
        cb: "RawMessageCallback | None" = None,
        queue: str | None = None,
        max_in_flight: int | None = None,
        deadletter_on_failure: bool = True,
    ) -> Subscription:
        """subscribe to subject with raw-bytes + reply-subject callback.

        kw-only; positional callback impossible. high-throughput sites
        that bypass Pydantic decoding use this; everything else should
        prefer :meth:`subscribe_typed`. callback receives an
        :class:`IncomingMessage` envelope so request/reply handlers can
        read ``msg.reply_subject`` and respond via
        :meth:`publish_reply`.

        :param subject: subject (point or wildcard pattern) to subscribe on
        :ptype subject: Subject
        :param cb: async callback receiving :class:`IncomingMessage` envelope
        :ptype cb: RawMessageCallback
        :param queue: optional queue group; messages on the subject load-balance across all subscribers in the same queue
        :ptype queue: str | None
        :param max_in_flight: optional concurrency cap for in-flight callbacks (per-subscription)
        :ptype max_in_flight: int | None
        :param deadletter_on_failure: when True (default) callback exceptions republish to ``{ns}.deadletter.{subject}``
        :ptype deadletter_on_failure: bool
        :return: opaque subscription handle for later :meth:`unsubscribe`
        :rtype: Subscription
        :raises SubscribeError: if subscription registration fails
        """
        if subject is None:
            raise SubscribeError("subscribe requires a subject (positional or kw)")
        if cb is None:
            raise SubscribeError("subscribe requires cb= callback")
        return await self._subscribe_internal(
            subject=subject,
            raw_cb=cb,
            typed_cb=None,
            message_type=None,
            queue=queue,
            max_in_flight=max_in_flight,
            deadletter_on_failure=deadletter_on_failure,
        )

    async def subscribe_typed(
        self,
        *,
        subject: Subject,
        cb: Callable[[_T], Awaitable[None]],
        message_type: type[_T],
        queue: str | None = None,
        max_in_flight: int | None = None,
        deadletter_on_failure: bool = True,
    ) -> Subscription:
        """subscribe with auto-decoded Pydantic message callback.

        each incoming message is parsed via
        ``message_type.model_validate_json``. validation failures
        publish to deadletter (when enabled) and are not delivered to
        the callback.

        :param subject: subject (point or wildcard pattern)
        :ptype subject: Subject
        :param cb: async callback receiving parsed message
        :ptype cb: Callable[[_T], Awaitable[None]]
        :param message_type: Pydantic class to decode incoming bytes into
        :ptype message_type: type[_T]
        :param queue: optional queue group
        :ptype queue: str | None
        :param max_in_flight: optional concurrency cap
        :ptype max_in_flight: int | None
        :param deadletter_on_failure: when True (default) validation + callback exceptions deadletter
        :ptype deadletter_on_failure: bool
        :return: subscription handle
        :rtype: Subscription
        :raises SubscribeError: if subscription registration fails
        """
        return await self._subscribe_internal(
            subject=subject,
            raw_cb=None,
            typed_cb=cb,
            message_type=message_type,
            queue=queue,
            max_in_flight=max_in_flight,
            deadletter_on_failure=deadletter_on_failure,
        )

    async def _subscribe_internal(
        self,
        *,
        subject: Subject | str,
        raw_cb: "RawMessageCallback | None",
        typed_cb: Callable[[Any], Awaitable[None]] | None,
        message_type: type[BaseModel] | None,
        queue: str | None,
        max_in_flight: int | None,
        deadletter_on_failure: bool,
    ) -> Subscription:
        """common subscribe path used by :meth:`subscribe` / :meth:`subscribe_typed`.

        a bare ``str`` subject is accepted and coerced to :class:`Subject` below (the
        test-ergonomic shorthand), so the param is ``Subject | str`` to match both callers.

        :param subject: subject pattern or point (``str`` coerced to :class:`Subject`)
        :ptype subject: Subject | str
        :param raw_cb: raw-bytes callback (mutually exclusive with typed_cb)
        :ptype raw_cb: RawMessageCallback | None
        :param typed_cb: Pydantic-decoded callback (mutually exclusive with raw_cb)
        :ptype typed_cb: Callable[[Any], Awaitable[None]] | None
        :param message_type: Pydantic message class for typed path
        :ptype message_type: type[BaseModel] | None
        :param queue: optional queue group
        :ptype queue: str | None
        :param max_in_flight: optional concurrency cap
        :ptype max_in_flight: int | None
        :param deadletter_on_failure: deadletter on callback exception
        :ptype deadletter_on_failure: bool
        :return: subscription handle
        :rtype: Subscription
        :raises SubscribeError: if registration fails
        """
        if (raw_cb is None) == (typed_cb is None):
            raise SubscribeError("exactly one of raw_cb / typed_cb must be supplied")
        if max_in_flight is not None and max_in_flight <= 0:
            raise SubscribeError("max_in_flight must be positive when set")

        # accept a bare ``str`` as a Subject shorthand. integration tests
        # build subject strings from f-strings and pass them directly;
        # forcing every test to wrap with ``Subject.raw(...)`` adds noise
        # without changing the contract -- the wrapper still emits
        # ``subject.path`` to nats-py either way.
        if isinstance(subject, str):
            subject = Subject.raw(subject)

        try:
            raw_sub = await self._raw.subscribe(subject.path, queue=queue or "")
        except Exception as exc:
            raise SubscribeError(f"subscribe failed: subject={subject.path} queue={queue!r}: {exc}") from exc

        semaphore: asyncio.Semaphore | None = asyncio.Semaphore(max_in_flight) if max_in_flight is not None else None

        async def _dispatch_one(msg: "_NatsMsg") -> None:
            """process one message; deadletter on failure when enabled."""
            try:
                if typed_cb is not None and message_type is not None:
                    parsed = message_type.model_validate_json(msg.data)
                    await typed_cb(parsed)
                else:
                    assert raw_cb is not None
                    incoming = IncomingMessage(
                        data=msg.data,
                        reply_subject=msg.reply or None,
                        subject=msg.subject,
                    )
                    await raw_cb(incoming)
            except ValidationError as exc:
                log.warning(
                    "subscribe_typed validation failure",
                    extra={
                        "extra_data": {
                            "subject": subject.path,
                            "message_type": message_type.__name__ if message_type else None,
                            "error": str(exc),
                        }
                    },
                )
                if deadletter_on_failure:
                    await self._deadletter(subject=subject, payload=msg.data, error=exc)
            except Exception as exc:  # noqa: BLE001 — boundary: we MUST catch everything to keep the dispatch loop alive
                log.error(
                    "subscribe callback raised",
                    extra={
                        "extra_data": {
                            "subject": subject.path,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    },
                )
                if deadletter_on_failure:
                    await self._deadletter(subject=subject, payload=msg.data, error=exc)

        inflight: set[asyncio.Task[None]] = set()

        async def _run_bounded(msg: "_NatsMsg") -> None:
            """process one message under the concurrency cap, freeing the slot when done."""
            try:
                await _dispatch_one(msg)
            finally:
                assert semaphore is not None  # only spawned on the bounded (max_in_flight) path
                semaphore.release()

        async def _dispatch() -> None:
            """drive subscription message loop."""
            try:
                async for msg in raw_sub.messages:
                    if semaphore is None:
                        # serial default (max_in_flight unset): one callback at a time, preserving
                        # in-order processing for subscribers that rely on it.
                        await _dispatch_one(msg)
                    else:
                        # bounded concurrency: keep at most ``max_in_flight`` callbacks in flight.
                        # Acquire a slot BEFORE accepting the next message so the loop back-pressures
                        # at the cap, then run the callback as its OWN task (freeing its slot on
                        # completion). Awaiting the callback here instead would serialize every
                        # message despite the cap — the bug this cap was meant to avoid.
                        await semaphore.acquire()
                        task = asyncio.create_task(_run_bounded(msg))
                        inflight.add(task)
                        task.add_done_callback(inflight.discard)
            except asyncio.CancelledError:
                # unsubscribe cancels this dispatch task; cancel the in-flight callback tasks too so
                # they are not orphaned — mirroring the serial path, where the single in-flight await
                # is cancelled together with the loop.
                for task in list(inflight):
                    task.cancel()
                raise
            except Exception as exc:  # noqa: BLE001 — diag only
                log.error(
                    "subscription dispatch loop crashed",
                    extra={
                        "extra_data": {
                            "subject": subject.path,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    },
                )

        dispatch_task = asyncio.create_task(
            _dispatch(),
            name=f"nats-dispatch:{subject.path}",
        )

        sub = Subscription(
            raw_subscription=raw_sub,
            subject=subject,
            dispatch_task=dispatch_task,
        )
        self._subscriptions.append(sub)

        log.info(
            "NATS subscribed",
            extra={
                "extra_data": {
                    "subject": subject.path,
                    "kind": subject.kind,
                    "queue": queue,
                    "typed": typed_cb is not None,
                    "deadletter_on_failure": deadletter_on_failure,
                }
            },
        )
        return sub

    async def unsubscribe(self, sub: Subscription) -> None:
        """drop a subscription.

        idempotent — second call is a no-op.

        :param sub: subscription handle returned by :meth:`subscribe`
        :ptype sub: Subscription
        :return: nothing
        :rtype: None
        """
        if sub.is_closed:
            return
        sub.mark_closed()
        try:
            await sub.raw_subscription.unsubscribe()
        except Exception as exc:  # noqa: BLE001 — diag only
            log.warning(
                "unsubscribe failed",
                extra={"extra_data": {"subject": sub.subject.path, "error": str(exc)}},
            )
        sub.dispatch_task.cancel()
        try:
            await sub.dispatch_task
        except asyncio.CancelledError, Exception:  # noqa: BLE001
            pass
        if sub in self._subscriptions:
            self._subscriptions.remove(sub)

    # ------------------------------------------------------------------
    # request / reply
    # ------------------------------------------------------------------

    async def request(
        self,
        *args: Any,
        subject: Subject | str | None = None,
        message: BaseModel | None = None,
        response_type: type[_T] | None = None,
        timeout: timedelta | float = DEFAULT_REQUEST_TIMEOUT,
    ) -> Any:
        """request/reply round-trip.

        primary (canonical) form is keyword-only with a typed Pydantic
        request + response::

            await nc.request(subject=Subject, message=Model,
                             response_type=ResponseModel)

        positional shorthand
        ``nc.request(subject_str, payload_bytes, timeout=N)`` is also
        accepted for parity with raw nats-py — every integration test
        uses this shape. the shorthand returns a raw
        :class:`nats.aio.client.Msg` whose ``.data`` carries the
        response bytes (matching the raw-nats interface integration
        tests already consume).

        :param args: optional positional ``(subject_str, payload_bytes)``
            shorthand for raw request/reply
        :ptype args: Any
        :param subject: target subject (kw form). bare ``str`` is
            auto-wrapped via :meth:`Subject.raw`.
        :ptype subject: Subject | str | None
        :param message: typed Pydantic request body (kw form)
        :ptype message: BaseModel | None
        :param response_type: Pydantic class to decode response into
            (kw form)
        :ptype response_type: type[_T] | None
        :param timeout: max wait for reply; ``int`` / ``float`` is
            interpreted as seconds (raw-nats parity)
        :ptype timeout: timedelta | float
        :return: decoded :class:`BaseModel` (kw form) or raw nats-py
            ``Msg`` (positional form)
        :rtype: Any
        :raises RequestError: on timeout, no responders, transport
            failure, or response decode failure
        """
        # positional shorthand: (subject, payload_bytes), optional kw timeout
        if args:
            if len(args) != 2:
                raise RequestError(
                    f"request positional form requires exactly (subject, payload_bytes); got {len(args)} args",
                )
            pos_subject, pos_payload = args
            if not isinstance(pos_payload, bytes | bytearray | memoryview):
                raise RequestError(
                    f"request positional payload must be bytes-like; got {type(pos_payload).__name__}",
                )
            sub = pos_subject if isinstance(pos_subject, Subject) else Subject.raw(str(pos_subject))
            secs = timeout.total_seconds() if isinstance(timeout, timedelta) else float(timeout)
            try:
                msg = await self._raw.request(sub.path, bytes(pos_payload), timeout=secs)
            except (_NatsTimeoutError, asyncio.TimeoutError, TimeoutError) as exc:
                raise RequestError(
                    f"request timed out: subject={sub.path} timeout={secs:.1f}s",
                ) from exc
            except _NatsNoRespondersError as exc:
                raise RequestError(f"no responders for subject: subject={sub.path}") from exc
            except Exception as exc:
                self._note_if_outbound_overflow(exc)  # resilience-task-03
                raise RequestError(f"request failed: subject={sub.path}: {exc}") from exc
            self._note_publish_success()  # resilience-task-03
            return msg
        if subject is None or message is None or response_type is None:
            raise RequestError(
                "request requires either positional (subject, payload_bytes) "
                "or kwargs subject= + message= + response_type=",
            )
        sub = subject if isinstance(subject, Subject) else Subject.raw(subject)
        td = timeout if isinstance(timeout, timedelta) else timedelta(seconds=float(timeout))
        payload = message.model_dump_json().encode("utf-8")
        response_bytes = await self.request_raw(subject=sub, payload=payload, timeout=td)
        try:
            return response_type.model_validate_json(response_bytes)
        except ValidationError as exc:
            raise RequestError(
                f"response decode failed: subject={sub.path} type={response_type.__name__}: {exc}",
            ) from exc

    async def request_raw(
        self,
        *,
        subject: Subject,
        payload: bytes,
        timeout: timedelta = DEFAULT_REQUEST_TIMEOUT,
    ) -> bytes:
        """raw-bytes request/reply round-trip.

        :param subject: target subject (point only)
        :ptype subject: Subject
        :param payload: pre-serialized request bytes
        :ptype payload: bytes
        :param timeout: max wait for reply
        :ptype timeout: timedelta
        :return: response payload bytes
        :rtype: bytes
        :raises RequestError: on timeout, no responders, transport failure
        """
        try:
            msg = await self._raw.request(subject.path, payload, timeout=timeout.total_seconds())
        except (_NatsTimeoutError, asyncio.TimeoutError, TimeoutError) as exc:
            raise RequestError(
                f"request timed out: subject={subject.path} timeout={timeout.total_seconds():.1f}s"
            ) from exc
        except _NatsNoRespondersError as exc:
            raise RequestError(f"no responders for subject: subject={subject.path}") from exc
        except _NatsConnectionClosedError as exc:
            raise RequestError(f"NATS connection closed during request: subject={subject.path}") from exc
        except Exception as exc:
            self._note_if_outbound_overflow(exc)  # resilience-task-03
            raise RequestError(f"request failed: subject={subject.path}: {exc}") from exc
        self._note_publish_success()  # resilience-task-03
        return bytes(msg.data)

    # ------------------------------------------------------------------
    # JetStream KV
    # ------------------------------------------------------------------

    async def kv_bucket(
        self,
        *,
        name: str,
        ttl: timedelta | None = None,
        storage: str = "memory",
        create_if_missing: bool = True,
        history: int = 1,
    ) -> NatsKvBucket:
        """obtain (or create) a JetStream KV bucket.

        bucket name is auto-prefixed with the configured namespace
        (``{namespace}-{name}``). passing ``ttl=None`` means values do
        not expire. storage defaults to ``"memory"``: in 3tears, NATS is
        the **L2** tier (ephemeral; durability rides JetStream R3
        replication + the consumer's real L3). Pass ``"file"`` only as a
        deliberate opt-in when a bucket genuinely needs on-disk durability.

        :param name: bucket name suffix (will be prefixed by namespace)
        :ptype name: str
        :param ttl: optional time-to-live for entries; ``None`` for no expiry
        :ptype ttl: timedelta | None
        :param storage: ``"memory"`` (default — L2) or ``"file"`` (opt-in)
        :ptype storage: str
        :param create_if_missing: create bucket if it does not exist
        :ptype create_if_missing: bool
        :param history: number of historical revisions to keep per key
        :ptype history: int
        :return: ready KV bucket handle
        :rtype: NatsKvBucket
        :raises KvError: if bucket creation or binding fails
        """
        # local import avoids circular dependency between client.py and kv.py
        from threetears.nats.kv import NatsKvBucket

        full_name = f"{self._namespace}-{name}"
        async with self._kv_lock:
            cached = self._buckets.get(full_name)
            if cached is not None:
                return cached
            bucket = await NatsKvBucket.open(
                client=self,
                full_name=full_name,
                ttl=ttl,
                storage=storage,
                create_if_missing=create_if_missing,
                history=history,
            )
            self._buckets[full_name] = bucket
        return bucket

    async def ensure_jetstream_stream(
        self,
        *,
        name: str,
        subjects: list[str],
        storage: str = "memory",
    ) -> str:
        """create (or update) a JetStream stream over given subjects.

        idempotent: binds to an existing stream of the same name and reconciles
        its subject set, else creates it. the stream name is namespace-prefixed
        (``{namespace}-{name}``) to match the KV-bucket convention. storage
        defaults to ``"memory"``: NATS is the **L2** tier in 3tears (ephemeral;
        durability rides JetStream R3 replication + the consumer's real L3).
        Pass ``"file"`` only as a deliberate opt-in when a stream genuinely
        needs on-disk durability.

        :param name: stream name suffix (namespace-prefixed)
        :ptype name: str
        :param subjects: subject patterns the stream captures
        :ptype subjects: list[str]
        :param storage: ``"memory"`` (default — L2) or ``"file"`` (opt-in)
        :ptype storage: str
        :return: full namespace-prefixed stream name
        :rtype: str
        :raises StreamSubjectsOverlapError: if subjects are already claimed by a different stream
        :raises RuntimeError: if stream creation and update both fail
        """
        from nats.js.api import StorageType, StreamConfig  # noqa: PLC0415

        full_name = f"{self._namespace}-{name}"
        storage_type = StorageType.FILE if storage == "file" else StorageType.MEMORY
        config = StreamConfig(name=full_name, subjects=subjects, storage=storage_type)
        js = self.jetstream_context()
        try:
            await js.add_stream(config)
        except Exception as exc:  # noqa: BLE001 -- classified below, not swallowed
            # add_stream fails for DISTINCT conditions that must not be conflated:
            #   - "subjects overlap with an existing stream" (JetStream err_code
            #     10065): a DIFFERENT stream already owns these subjects. full_name
            #     was never created, so update_stream would raise NotFoundError and
            #     mask this actionable error. surface it -- the real problem is a
            #     conflicting stream (usually a client on the wrong namespace).
            #   - anything else (chiefly "stream name already in use"): a stream of
            #     THIS name exists; reconcile its subject set via update. if that
            #     also fails, it propagates -- never silently swallowed.
            err_code = getattr(exc, "err_code", None)
            if err_code == _JS_ERR_SUBJECTS_OVERLAP or "subjects overlap" in str(exc).lower():
                raise StreamSubjectsOverlapError(
                    f"cannot create JetStream stream {full_name!r} over subjects "
                    f"{subjects}: they overlap subjects already claimed by a different "
                    f"stream on this NATS account (a subject belongs to exactly one "
                    f"stream). the usual cause is another connection using the wrong "
                    f"subject namespace; resolve the conflicting stream or correct the "
                    f"namespace."
                ) from exc
            await js.update_stream(config)
        log.info(
            "jetstream stream ensured: stream=%s subjects=%s storage=%s",
            full_name,
            ",".join(subjects),
            storage,
        )
        return full_name

    async def jetstream_publish(self, *, subject: Subject, payload: bytes) -> None:
        """publish raw bytes to a JetStream subject with persistence ack.

        unlike :meth:`publish_raw` (core NATS, fire-and-forget), this persists
        the message to the backing stream and awaits the broker ``PubAck``, so a
        message published while no consumer is attached is retained and
        redelivered to a durable consumer when it connects.

        :param subject: target JetStream subject
        :ptype subject: Subject
        :param payload: serialized message bytes
        :ptype payload: bytes
        :return: nothing
        :rtype: None
        """
        js = self.jetstream_context()
        await js.publish(subject.path, payload)

    async def jetstream_subscribe_durable(
        self,
        *,
        subject: Subject,
        durable: str,
        cb: Callable[[Any], Awaitable[None]],
        max_deliver: int,
        dead_letter_subject: Subject | None = None,
        ack_wait_seconds: float = 60.0,
        stream: str | None = None,
    ) -> JetStreamPushConsumer:
        """create a durable push consumer with MANUAL ack + BOUNDED redelivery.

        the callback does the work and MUST ``await msg.ack()`` on success (or on
        a terminal drop redelivery cannot fix). on a RETRYABLE failure the
        callback RAISES; this wrapper owns the redelivery policy:

        - attempts 1..``max_deliver``-1: ``msg.nak(delay=...)`` with a capped
          linear backoff, so a transient failure (Slack rate limit, network)
          recovers on a later attempt without a hot loop;
        - attempt ``max_deliver``: DEAD-LETTER -- republish the payload to
          ``dead_letter_subject`` (when given), ack the original so it leaves the
          live consumer, and log ONE error.

        ``max_deliver`` is REQUIRED and bounds total attempts at the consumer
        config too (belt + suspenders). a durable consumer with unbounded
        redelivery is the poison-pill incident this method makes impossible to
        construct -- a message that can never succeed stops retrying and lands
        somewhere inspectable instead of alerting forever.

        :param subject: subject to consume
        :ptype subject: Subject
        :param durable: durable consumer name (stable across restarts)
        :ptype durable: str
        :param cb: async callback; acks on success/terminal-drop, RAISES to retry
        :ptype cb: Callable[[Any], Awaitable[None]]
        :param max_deliver: maximum delivery attempts before dead-lettering (>= 1)
        :ptype max_deliver: int
        :param dead_letter_subject: subject the poisoned payload is parked on, or
            ``None`` to drop-with-alert once the budget is exhausted
        :ptype dead_letter_subject: Subject | None
        :param ack_wait_seconds: server-side ack timeout; a crash mid-handle
            redelivers after this (the backoff is driven by nak for the
            raise path)
        :ptype ack_wait_seconds: float
        :param stream: backing stream name to bind the consumer to
        :ptype stream: str | None
        :return: subscription handle -- call :meth:`JetStreamPushConsumer.stop`
            directly on it to unsubscribe (NOT :meth:`NatsClient.unsubscribe`,
            which expects a :class:`Subscription` instead)
        :rtype: JetStreamPushConsumer
        :raises ValueError: when ``max_deliver`` < 1
        """
        if max_deliver < 1:
            raise ValueError(
                f"max_deliver must be >= 1: a durable consumer cannot redeliver forever (got {max_deliver})",
            )

        async def _bounded_cb(msg: Any) -> None:
            """run the handler; nak-with-backoff on retryable raise, dead-letter at the budget.

            :param msg: raw nats-py JetStream message
            :ptype msg: Any
            :return: nothing
            :rtype: None
            """
            try:
                await cb(msg)
            except Exception as exc:  # noqa: BLE001 - bounded redelivery is the whole point; we re-route, never swallow
                await self._redeliver_or_deadletter(
                    msg,
                    exc,
                    max_deliver=max_deliver,
                    dead_letter_subject=dead_letter_subject,
                    durable=durable,
                    subject=subject,
                )

        js = self.jetstream_context()
        config = _NatsConsumerConfig(
            durable_name=durable,
            ack_policy=_NatsAckPolicy.EXPLICIT,
            max_deliver=max_deliver,
            ack_wait=ack_wait_seconds,
        )
        sub = await js.subscribe(
            subject.path,
            durable=durable,
            cb=_bounded_cb,
            manual_ack=True,
            stream=stream,
            config=config,
        )
        log.info(
            "jetstream durable consumer subscribed: subject=%s durable=%s stream=%s max_deliver=%d dlq=%s",
            subject.path,
            durable,
            stream,
            max_deliver,
            dead_letter_subject.path if dead_letter_subject is not None else None,
        )
        return JetStreamPushConsumer(raw_subscription=sub, subject=subject, durable=durable)

    async def _redeliver_or_deadletter(
        self,
        msg: Any,
        exc: BaseException,
        *,
        max_deliver: int,
        dead_letter_subject: Subject | None,
        durable: str,
        subject: Subject,
    ) -> None:
        """apply the bounded-redelivery policy to a message whose handler raised.

        shared by the push (:meth:`jetstream_subscribe_durable`) and pull
        (:meth:`jetstream_pull_subscribe`) durable consumers so the
        nak-with-backoff-then-dead-letter contract is defined ONCE:

        - attempts ``1..max_deliver-1``: ``msg.nak(delay=...)`` with a capped
          linear backoff so a transient failure recovers without a hot loop;
        - attempt ``max_deliver``: republish the payload to
          ``dead_letter_subject`` (when given), ``ack`` the original so it leaves
          the live consumer, and log ONE error.

        :param msg: raw nats-py JetStream message whose handler raised
        :ptype msg: Any
        :param exc: exception the handler raised
        :ptype exc: BaseException
        :param max_deliver: maximum delivery attempts before dead-lettering
        :ptype max_deliver: int
        :param dead_letter_subject: subject poisoned payload parks on, or None
        :ptype dead_letter_subject: Subject | None
        :param durable: durable consumer name (for the log line)
        :ptype durable: str
        :param subject: subject being consumed (for the log line)
        :ptype subject: Subject
        :return: nothing
        :rtype: None
        """
        metadata = getattr(msg, "metadata", None)
        num_delivered = int(getattr(metadata, "num_delivered", 1) or 1)
        if num_delivered >= max_deliver:
            if dead_letter_subject is not None:
                await self.jetstream_publish(subject=dead_letter_subject, payload=msg.data)
            await msg.ack()
            log.error(
                "durable consumer dead-lettered message after %d attempts: durable=%s subject=%s dlq=%s error=%s",
                num_delivered,
                durable,
                subject.path,
                dead_letter_subject.path if dead_letter_subject is not None else None,
                exc,
            )
        else:
            backoff = float(min(num_delivered * 5, 30))
            await msg.nak(delay=backoff)
            log.warning(
                "durable consumer redelivering message (attempt %d/%d, backoff=%.0fs): durable=%s error=%s",
                num_delivered,
                max_deliver,
                backoff,
                durable,
                exc,
            )

    async def jetstream_pull_subscribe(
        self,
        *,
        subject: Subject,
        durable: str,
        cb: Callable[[Any], Awaitable[None]],
        max_deliver: int,
        dead_letter_subject: Subject | None = None,
        ack_wait_seconds: float = 60.0,
        stream: str | None = None,
        batch: int = 8,
        fetch_timeout_seconds: float = 5.0,
    ) -> JetStreamPullConsumer:
        """create a SHARED durable PULL consumer with MANUAL ack + BOUNDED redelivery.

        the pull counterpart to :meth:`jetstream_subscribe_durable`. where the
        push variant binds ONE subscriber to a durable, a pull consumer is
        drained by N independent fetchers that all bind the SAME ``durable``
        name: JetStream hands each pending message to exactly one fetcher, so M
        messages split one-of-N across the worker replicas (horizontal scale). a
        replica holding no in-flight fetch keeps NO server-pushed delivery
        subject open, so the consumer is scale-to-zero friendly (spin replicas up
        on ``num_pending``, down to zero when the backlog drains).

        the redelivery contract is identical to the push variant and shares its
        implementation (:meth:`_redeliver_or_deadletter`): the ``cb`` acks on
        success / terminal-drop and RAISES to retry; attempts
        ``1..max_deliver-1`` nak-with-backoff, attempt ``max_deliver``
        dead-letters. ``max_deliver`` bounds attempts at the consumer config too.

        the consumer FILTERS on ``subject`` (an exact subject, never a wildcard)
        so a sibling dead-letter subject retained on the SAME stream is not
        re-drained by this consumer.

        :param subject: exact subject to consume (the consumer filter subject)
        :ptype subject: Subject
        :param durable: shared durable consumer name (stable; the same name binds
            every replica so they share the backlog)
        :ptype durable: str
        :param cb: async handler; acks on success/terminal-drop, RAISES to retry
        :ptype cb: Callable[[Any], Awaitable[None]]
        :param max_deliver: maximum delivery attempts before dead-lettering (>= 1)
        :ptype max_deliver: int
        :param dead_letter_subject: subject poisoned payload parks on, or None
        :ptype dead_letter_subject: Subject | None
        :param ack_wait_seconds: server-side ack timeout; a fetcher that dies
            mid-handle redelivers the message to another fetcher after this
        :ptype ack_wait_seconds: float
        :param stream: backing stream name to bind the consumer to
        :ptype stream: str | None
        :param batch: max messages one fetch pulls
        :ptype batch: int
        :param fetch_timeout_seconds: how long one fetch waits for a message
            before returning empty (the idle poll cadence)
        :ptype fetch_timeout_seconds: float
        :return: a pull-consumer handle whose ``run`` loops fetch+dispatch
        :rtype: JetStreamPullConsumer
        :raises ValueError: when ``max_deliver`` < 1
        """
        if max_deliver < 1:
            raise ValueError(
                f"max_deliver must be >= 1: a durable consumer cannot redeliver forever (got {max_deliver})",
            )
        js = self.jetstream_context()
        config = _NatsConsumerConfig(
            durable_name=durable,
            ack_policy=_NatsAckPolicy.EXPLICIT,
            max_deliver=max_deliver,
            ack_wait=ack_wait_seconds,
            filter_subject=subject.path,
        )
        psub = await js.pull_subscribe(
            subject.path,
            durable=durable,
            stream=stream,
            config=config,
        )

        async def _redeliver(msg: Any, exc: BaseException) -> None:
            """route a raised handler through the shared bounded-redelivery policy.

            :param msg: raw nats-py JetStream message whose handler raised
            :ptype msg: Any
            :param exc: exception the handler raised
            :ptype exc: BaseException
            :return: nothing
            :rtype: None
            """
            await self._redeliver_or_deadletter(
                msg,
                exc,
                max_deliver=max_deliver,
                dead_letter_subject=dead_letter_subject,
                durable=durable,
                subject=subject,
            )

        consumer = JetStreamPullConsumer(
            psub=psub,
            cb=cb,
            redeliver=_redeliver,
            durable=durable,
            subject=subject,
            batch=batch,
            fetch_timeout_seconds=fetch_timeout_seconds,
        )
        log.info(
            "jetstream pull consumer bound: subject=%s durable=%s stream=%s max_deliver=%d dlq=%s batch=%d",
            subject.path,
            durable,
            stream,
            max_deliver,
            dead_letter_subject.path if dead_letter_subject is not None else None,
            batch,
        )
        return consumer

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def jetstream_context(self) -> Any:
        """obtain a JetStream context bound to underlying client.

        public for use by :class:`NatsKvBucket` (intra-package
        coupling) and as an explicit escape hatch for callers needing
        custom JetStream stream / consumer config not yet exposed by
        the wrapper. excluded from ``threetears.nats.__all__`` so it
        is not re-exported as part of the public api surface.

        :return: nats-py JetStream context
        :rtype: Any
        """
        return self._raw.jetstream()

    async def _deadletter(
        self,
        *,
        subject: Subject,
        payload: bytes,
        error: BaseException,
    ) -> None:
        """republish a failed message to the deadletter subject.

        envelope wraps the original payload + structured error
        diagnostics so consumers of the deadletter stream can triage
        without reaching into the original transport.

        :param subject: original subject the message arrived on
        :ptype subject: Subject
        :param payload: original message payload bytes
        :ptype payload: bytes
        :param error: exception that caused deadlettering
        :ptype error: BaseException
        :return: nothing
        :rtype: None
        """
        dl_subject = Subjects.deadletter(subject.path)
        envelope = {
            "original_subject": subject.path,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now(UTC).isoformat(),
            "client_name": self._client_name,
            "payload_b64": payload.hex(),
        }
        try:
            await self._raw.publish(
                dl_subject.path,
                json.dumps(envelope).encode("utf-8"),
            )
        except Exception as exc:  # noqa: BLE001 — diag only
            log.warning(
                "deadletter publish failed",
                extra={
                    "extra_data": {
                        "deadletter_subject": dl_subject.path,
                        "original_subject": subject.path,
                        "error": str(exc),
                    }
                },
            )


# ---------------------------------------------------------------------------
# module helpers
# ---------------------------------------------------------------------------


_last_error_log: dict[str, float] = {}


async def _establish_connection(
    servers: list[str],
    options: dict[str, object],
    primary_url: str,
) -> _NatsPyClient:
    """open the underlying nats-py connection.

    :param servers: NATS server URL list
    :ptype servers: list[str]
    :param options: nats-py connect options
    :ptype options: dict[str, object]
    :param primary_url: primary URL (for diagnostics)
    :ptype primary_url: str
    :return: connected nats-py client
    :rtype: nats.aio.client.Client
    :raises NatsClientError: if connection fails
    """
    try:
        nc: _NatsPyClient = await nats.connect(servers, **options)
    except Exception as exc:
        raise NatsClientError(f"failed to connect to NATS at {primary_url}: {exc}") from exc
    return nc


async def _verify_jetstream(nc: _NatsPyClient, primary_url: str) -> None:
    """verify JetStream is reachable on connected client.

    :param nc: connected nats-py client
    :ptype nc: nats.aio.client.Client
    :param primary_url: primary URL (for diagnostics)
    :ptype primary_url: str
    :return: nothing
    :rtype: None
    :raises NatsClientError: if JetStream is not reachable
    """
    try:
        js = nc.jetstream()
        await asyncio.wait_for(js.account_info(), timeout=10.0)
    except Exception as exc:
        await nc.close()
        raise NatsClientError(f"NATS JetStream not available at {primary_url}: {exc}") from exc


async def _on_reconnected() -> None:
    """nats-py callback invoked on reconnect.

    no re-subscription is needed here: nats-py replays every live
    subscription under its original ``sid`` inside ``_attempt_reconnect``
    (it re-sends the ``SUB`` command for each entry in ``client._subs``
    and never touches the subscription's pending-message queue), so the
    wrapper's per-subscription dispatch loop -- which iterates
    ``raw_sub.messages`` -- keeps reading the SAME queue across the
    disconnect/reconnect and is never observed ending. the loop only
    ends on an explicit unsubscribe/drain or a permanent client close;
    with forever-reconnect (:data:`RUNTIME_MAX_RECONNECT_ATTEMPTS` ``=
    -1``) the reconnect path never closes the client, so an outage can
    no longer silently kill a subscription.

    :return: nothing
    :rtype: None
    """
    log.info("NATS reconnected")


async def _on_disconnected() -> None:
    """nats-py callback invoked on disconnect.

    :return: nothing
    :rtype: None
    """
    log.warning("NATS disconnected")


def _is_authorization_violation(exc: Exception) -> bool:
    """whether ``exc`` is an auth rejection from the server (the wedged-auth signal is_healthy tracks).

    Auth rejections reach the error callback in two shapes, so we detect BOTH — matching on semantics,
    not one string:

    * the reconnect-loop path raises a generic ``errors.Error("nats: 'Authorization Violation'")`` — the
      ``-ERR`` text the server sends; caught by the ``"authorization violation"`` substring.
    * nats-py's typed :class:`nats.errors.AuthorizationError` whose ``str`` is ``"nats: authorization
      failed"`` — NOT covered by the first substring. Matched by type (and its ``"authorization failed"``
      text) so a future nats-py routing change that surfaces the typed error to ``error_cb`` still trips
      the counter, rather than silently re-opening the multi-hour auth-wedge this signal exists to catch.

    :param exc: the exception nats-py surfaced to the error callback.
    :ptype exc: Exception
    :return: ``True`` when it is an authorization rejection.
    :rtype: bool
    """
    if isinstance(exc, _NatsAuthorizationError):
        return True
    text = str(exc).lower()
    return "authorization violation" in text or "authorization failed" in text


def _is_outbound_overflow(exc: Exception) -> bool:
    """whether ``exc`` is an outbound/pending-buffer overflow raised at the publish boundary (resilience-task-03).

    nats-py raises :class:`nats.errors.OutboundBufferLimitError` (str: ``"nats: outbound buffer limit
    exceeded"``) synchronously from ``publish``/``request`` while disconnected/reconnecting with a full
    pending buffer -- the wedge state this signal exists to catch. Detected BOTH by type and by the
    ``-ERR`` text (mirroring :func:`_is_authorization_violation`) so a future nats-py that wraps or
    re-routes the error still trips the counter rather than silently re-opening the unbounded thrash.

    :param exc: the exception the underlying nats-py ``publish``/``request`` raised.
    :ptype exc: Exception
    :return: ``True`` when it is an outbound-buffer overflow.
    :rtype: bool
    """
    if isinstance(exc, _NatsOutboundBufferLimitError):
        return True
    return "outbound buffer limit exceeded" in str(exc).lower()


async def _on_error(exc: Exception) -> None:
    """nats-py error callback with rate-limited logging.

    :param exc: exception from nats-py client
    :ptype exc: Exception
    :return: nothing
    :rtype: None
    """
    key = f"{type(exc).__name__}:{exc}"
    now = time.monotonic()
    last = _last_error_log.get(key, 0.0)
    if now - last >= _ERROR_LOG_RATE_LIMIT_SECONDS:
        log.error("NATS error: %s", exc)
        _last_error_log[key] = now
    else:
        log.debug("NATS error (rate-limited duplicate): %s", exc)

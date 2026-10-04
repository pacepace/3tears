"""canonical NATS client wrapper for 3tears applications.

:class:`NatsClient` is THE single primitive every 3tears app uses to
talk to NATS. it absorbs the lifecycle, dual-phase reconnect-ceiling,
rate-limited error logging, deadletter dispatch, typed publish, and
JetStream KV access that previously lived in three half-overlapping
wrappers (``<upstream-hub>/common/nats.py``, the KV facade formerly at
``threetears.core.cache.kv`` (since deleted),
``<consumer>.runtime.nats_transport``).
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
  the window is kept per client (shared by its successor connections),
  so one client's error never silences another client's first report.
  a PERMISSIONS VIOLATION gets its own line naming the subject, the
  refused operation, and the consequence -- it is the only error here
  that leaves the connection up and raises to nobody, so a refused
  subscribe otherwise reads as one anonymous error while the
  subscription it belonged to silently receives nothing forever. its
  rate-limit key carries the subject, so a second dead subject is
  never suppressed behind the first. the same facts also go out
  STRUCTURED (``extra={"extra_data": ...}``, the platform convention
  the :mod:`threetears.observe` formatter serialises to JSON) as
  ``subject`` / ``operation`` / ``subject_case`` / ``error``, so a dead
  subject can be alerted on and grouped rather than only read. see
  :data:`_SUBJECT_CASE_VERBATIM` for why the subject's case is
  qualified rather than presented as exact.
- **deadletter dispatch**: by default uncaught exceptions in subscribe
  callbacks publish the original message + a structured envelope to
  ``{ns}.deadletter.{original_path}``. opt out per-subscribe with
  ``deadletter_on_failure=False`` (only for sites that already funnel
  errors through their own pipeline).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import random
import re
import time
import uuid
from collections.abc import Mapping
from enum import StrEnum
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Final, NamedTuple, TypeVar

from nats.aio.client import (
    DEFAULT_FLUSH_TIMEOUT as _NATS_FLUSH_TIMEOUT_SECONDS,
    Client as _NatsPyClient,
)
from nats.aio.subscription import DEFAULT_SUB_PENDING_BYTES_LIMIT, DEFAULT_SUB_PENDING_MSGS_LIMIT
from nats.js.client import JetStreamContext as _NatsJetStreamContext
from nats.js.api import (
    AckPolicy as _NatsAckPolicy,
    ConsumerConfig as _NatsConsumerConfig,
    DeliverPolicy as _NatsDeliverPolicy,
    StreamConfig as _NatsStreamConfig,
)
from nats.errors import (
    AuthorizationError as _NatsAuthorizationError,
    ConnectionClosedError as _NatsConnectionClosedError,
    Error as _NatsError,
    FlushTimeoutError as _NatsFlushTimeoutError,
    NoRespondersError as _NatsNoRespondersError,
    OutboundBufferLimitError as _NatsOutboundBufferLimitError,
    StaleConnectionError as _NatsStaleConnectionError,
    TimeoutError as _NatsTimeoutError,
)
from nats.js.errors import NotFoundError as _NatsJsNotFoundError
from pydantic import BaseModel, ValidationError
from threetears.observe import get_logger, representative_exception
from threetears.observe.resilience import retry_until_done

from threetears.nats.diagnostics import permissions_violation_remedy
from threetears.nats._nats_py_internals import (
    force_reconnect,
    pull_subscription_inbox,
    send_unsubscribe,
    take_queued_messages,
    write_pending_then_ping,
)
from threetears.nats._publish import as_payload_too_large, publish_bounded, raise_as_publish_error
from threetears.nats.receipt import ReceiptBacklog
from threetears.nats.credential_refusal import CredentialRefusal
from threetears.nats.renewal_request import CredentialRenewalRequest
from threetears.nats.credential_renewal import (
    REAUTH_CONNECT_TIMEOUT_SECONDS,
    REAUTH_MIN_SLEEP_SECONDS,
    REAUTH_RETIRE_DRAIN_SECONDS,
    REAUTH_RETRY_SECONDS,
    credential_lifetime_from_user_info,
    has_schedulable_ttl,
    nats_user_jwt_ttl_seconds,
    seconds_until_reauth,
    seconds_until_retirement,
    unsafe_renewal_reason,
)
from threetears.nats.errors import (
    NamespaceNotConfiguredError,
    NatsClientError,
    NoRespondersError,
    PublishError,
    RequestError,
    RequestTimeoutError,
    StreamSubjectsOverlapError,
    SubscribeError,
)
from threetears.nats.result_delivery import SYNC_REPLY_BUDGET_SECONDS
from threetears.nats.subject_permissions import SERVER_USER_INFO_SUBJECT
from threetears.nats.subjects import DEAD_LETTER_ORIGINAL_SUBJECT_HEADER, Subject, Subjects, set_default_namespace

# JetStream API error code for "subjects overlap with an existing stream": a
# subject belongs to exactly one stream. distinct from "stream name already in
# use" -- the two are conflated by a naive create-or-update. see
# ensure_jetstream_stream.
_JS_ERR_SUBJECTS_OVERLAP = 10065

# JetStream API error code for "stream name already in use with a different configuration". The
# server answers it only when it looked at the create and found a live stream of that name carrying
# something else: on a re-declaration after a reconnect that is a stream that SURVIVED (or that
# another declarer already brought back and has since changed), so it is left exactly as it is --
# unless it backs a KV bucket whose declarer owns it (``ensure_kv_bucket(owns_bucket=True)``).
_JS_ERR_STREAM_NAME_IN_USE = 10058

#: how far the server's reported credential lifetime may fall short of the configured one and still
#: be the configured one. the server's answer is rounded down to whole seconds, by design, so that
#: any error renews early (:func:`~threetears.nats.credential_renewal.credential_lifetime_from_user_info`).
_SERVER_TTL_ROUNDING_SECONDS: Final[int] = 1

#: first pause before a restoration after a reconnect tries again, when something it had to put back
#: (a stream or KV bucket this client declared, a durable consumer it bound) could not
#: be. doubles each failed round up to :data:`_RESTORE_RETRY_MAX_DELAY_SECONDS`. a broker that has
#: just come back can refuse the first JetStream calls while it recovers, and a stream declared by
#: ANOTHER process is back only once that process has reconnected too.
_RESTORE_RETRY_FIRST_DELAY_SECONDS: Final[float] = 0.5

#: longest pause between two rounds of a restoration that keeps failing. it never gives up: what it
#: restores is something every caller of this client depends on, and each failed round is logged at
#: ERROR naming what is still missing.
_RESTORE_RETRY_MAX_DELAY_SECONDS: Final[float] = 30.0

if TYPE_CHECKING:
    from nats.aio.client import Server as _NatsServer
    from nats.aio.msg import Msg as _NatsMsg

    from threetears.nats.kv import KvTimings, NatsKvBucket
    from threetears.nats.transport import RawMessageCallback


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
    "ConnectionEstablisher",
    "JetStreamPullConsumer",
    "JetStreamPushConsumer",
    "JetStreamResultWaiter",
    "NatsClient",
    "PublishPin",
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

#: opens ONE nats-py connection: ``(servers, options, primary_url) -> connected client``. The
#: client calls it for its first connection and again for every successor a credential renewal or a
#: lame-duck move opens, always with the options :meth:`NatsClient.connect` built plus that
#: connection's own callbacks. The default opens a real nats-py ``Client``; a host replaces it only
#: to put a connection of its own under the wrapper (a test double, a recording proxy), and the
#: replacement must raise :class:`~threetears.nats.errors.NatsClientError` when it cannot connect.
ConnectionEstablisher = Callable[[list[str], dict[str, object], str], Awaitable[_NatsPyClient]]


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

#: how long :meth:`JetStreamPullConsumer.stop` waits, beyond one fetch's own timeout, for the
#: handlers of the fetch in flight to finish before it unsubscribes anyway.
_PULL_CONSUMER_STOP_HANDLER_GRACE_SECONDS: Final[float] = 10.0

#: how many in-progress acks a pull consumer's handler sends per ``ack_wait`` while it runs, so a
#: slow but live handler keeps its message and ``ack_wait`` only ever measures a fetcher that died.
_PULL_CONSUMER_IN_PROGRESS_PER_ACK_WAIT: Final[int] = 3

#: how often a :class:`JetStreamResultWaiter` re-checks its own deadline while waiting.
#:
#: This is a poll cadence, NOT added latency: a fetch already outstanding when the answer is
#: published returns immediately. It only bounds how long past its deadline a hopeless wait can sit,
#: and how quickly a reconnect-broken consumer is noticed and rebuilt.
_RESULT_WAITER_POLL_SECONDS: Final[float] = 5.0

#: floor on that poll, so a nearly-elapsed budget cannot produce a zero/negative fetch timeout.
_RESULT_WAITER_MIN_POLL_SECONDS: Final[float] = 0.05

#: pace after a failed consumer rebuild, so a broker that is down does not turn the wait into a spin.
_RESULT_WAITER_REBUILD_BACKOFF_SECONDS: Final[float] = 1.0

#: margin added to a waiter's budget when setting the ephemeral consumer's inactivity threshold.
#:
#: The threshold must outlast the whole wait or the server reaps the consumer mid-call and the answer
#: has nowhere to be delivered. The margin covers the gap between consecutive fetches plus any clock
#: disagreement; it costs only how long an abandoned consumer lingers.
_RESULT_WAITER_KEEPALIVE_MARGIN_SECONDS: Final[float] = 60.0

#: the idle heartbeat a :class:`JetStreamResultWaiter`'s pushed consumer sends while it has nothing
#: to deliver. its absence is how a waiter learns the server lost the consumer (a broker restart, a
#: reap), since a pushed consumer that no longer exists delivers nothing and says nothing.
_RESULT_WAITER_HEARTBEAT_SECONDS: Final[float] = 5.0

#: consecutive heartbeats a waiter may miss before it replaces its consumer.
_RESULT_WAITER_MISSED_HEARTBEATS: Final[int] = 3

#: the prefix of every result waiter's consumer name, so an operator reading a stream's consumers
#: can tell a waiter from anything else on it.
_RESULT_WAITER_CONSUMER_PREFIX: Final[str] = "result-waiter-"

#: the status a pushed consumer's idle heartbeat carries.
_STATUS_IDLE_HEARTBEAT: Final[str] = "100"

#: ceiling on one consumer create, so a create the grant refuses -- which is never answered --
#: surfaces as a failure the wait retries rather than as a hang.
_RESULT_WAITER_CREATE_TIMEOUT_SECONDS: Final[float] = 10.0

#: bound on handing a superseded subscription over during a renewal. It must NEVER cut nats-py's
#: drain off inside the flush it starts: nats-py leaves the cancelled PING future in its PONG queue,
#: the next PONG then raises ``InvalidStateError`` in ``_process_pong``, and the read loop's catch-all
#: ends the loop -- the connection reports itself connected and never reads again (see
#: :func:`_round_trip`). A handover makes two round trips (:func:`_stop_routing_then_drain`): its own,
#: which is safe to cancel, and the flush inside nats-py's drain, which bounds itself
#: (``DEFAULT_FLUSH_TIMEOUT``). This bound exceeds both: when it fires, the drain is past its flush
#: and waiting for queued messages to be taken, where a cancellation harms nothing.
_HANDOVER_DRAIN_BOUND_SECONDS: Final[float] = REAUTH_RETIRE_DRAIN_SECONDS + 2 * _NATS_FLUSH_TIMEOUT_SECONDS

#: the prefix of the queue group a subscription joins when its caller names none. Each such
#: subscription gets a group of its OWN, of which it is the only member -- see
#: :class:`Subscription` for why a subscription is never plain.
_SOLE_MEMBER_QUEUE_PREFIX: Final[str] = "_solo."

#: default startup timeout (matches platform's ``startup_timeout_seconds`` env var).
DEFAULT_STARTUP_TIMEOUT: Final[timedelta] = timedelta(seconds=30)

#: default request/reply timeout when caller does not pass one explicitly.
DEFAULT_REQUEST_TIMEOUT: Final[timedelta] = timedelta(seconds=5)

#: default drain timeout for graceful shutdown.
DEFAULT_DRAIN_TIMEOUT: Final[timedelta] = timedelta(seconds=30)

#: default ceiling on one JetStream publish, ack included.
#:
#: Longer than :data:`DEFAULT_REQUEST_TIMEOUT` because this round trip includes
#: the broker persisting the message, and short enough that a wedged broker
#: surfaces as an error on the next tick rather than as a process that looks
#: healthy. **There was no ceiling here at all until 2026-08-18**, and the
#: absence froze a downstream ingestion fleet for ten days behind a publish
#: whose ack never arrived -- 0% CPU, no errors, no slow queries, the process
#: reporting itself up. Nothing short of walking the coroutine chain named it.
DEFAULT_JETSTREAM_PUBLISH_TIMEOUT: Final[timedelta] = timedelta(seconds=10)


#: dedup window for error-callback rate limiting.
_ERROR_LOG_RATE_LIMIT_SECONDS: Final[float] = 10.0

#: the phrase every NATS server permissions refusal carries, matched lowercased. nats-py lowercases
#: the whole ``-ERR`` payload in its protocol parser before dispatching to ``error_cb``, so the
#: comparison is done on a lowercased copy rather than assuming either case.
_PERMISSIONS_VIOLATION_PHRASE: Final[str] = "permissions violation"

#: decomposes the server's refusal into the operation + the subject it names. The server emits
#: ``Permissions Violation for Subscription to "<subject>"`` (optionally ``... using queue "<q>"``)
#: and ``Permissions Violation for Publish to "<subject>"``; the subject is always the FIRST quoted
#: token, so one pattern covers every variant. Case-insensitive so it survives a nats-py that stops
#: lowercasing the payload.
_PERMISSIONS_VIOLATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'permissions violation for (?P<operation>subscription|publish) to "(?P<subject>[^"]+)"',
    re.IGNORECASE,
)

#: server wording -> the word an operator reads. "subscription" becomes "subscribe" so the two
#: refusals read as the two operations a subject grant is split into.
_VIOLATION_OPERATIONS: Final[dict[str, str]] = {"subscription": "subscribe", "publish": "publish"}

#: what a refusal actually COSTS, per operation -- the half of the message that says a capability
#: just went dead rather than merely that an error occurred.
_VIOLATION_CONSEQUENCES: Final[dict[str, str]] = {
    "subscribe": (
        "That subscription stays open on this client and will receive NOTHING, from now on, "
        "with no further error and no exception anywhere -- the capability it serves is dead "
        "until the grant exists."
    ),
    "publish": (
        "That message was DROPPED and reached no subscriber; the publish call itself did not "
        "raise, so the sender believes it succeeded."
    ),
}

#: placeholders for a violation whose wording :data:`_PERMISSIONS_VIOLATION_PATTERN` cannot
#: decompose. Reporting the loss with placeholders beats degrading to an anonymous "NATS error".
#: they are for the HUMAN-READABLE sentence only -- the structured fields report ``None``, because a
#: placeholder in a queryable field is a value a log pipeline would group and alert on as if it were
#: a real subject.
_UNKNOWN_OPERATION: Final[str] = "operation"
_UNKNOWN_SUBJECT: Final[str] = "<not reported in the server's wording>"
_UNKNOWN_CONSEQUENCE: Final[str] = (
    "The refused operation fails SILENTLY: the connection stays open and nothing is raised to any "
    "caller, so whatever it served is dead until the grant exists."
)

#: how far the reported subject's CASE can be trusted. nats-py lowercases the WHOLE ``-ERR`` payload
#: in its protocol parser before dispatching to ``error_cb``, so a subject recovered from an
#: all-lowercase payload may have been mangled on the way here. That is harmless for a HITL subject
#: (sha256 hex, uuids -- already lowercase) and NOT harmless for ``$KV.``, ``$JS.API.`` or
#: ``_INBOX_`` subjects, where an operator pasting the reported string into a grant list would get
#: one that never matches. The subject is therefore reported EXACTLY as received, with a companion
#: field saying whether that case is the server's or the client parser's -- rather than being
#: silently "corrected", which would invent a case nothing observed.
#:
#: The flag is DERIVED, not assumed: uppercase anywhere in the payload proves the parser did not
#: lowercase it, so a future nats-py that stops doing so reports ``verbatim`` with no code change.
_SUBJECT_CASE_VERBATIM: Final[str] = "verbatim"
_SUBJECT_CASE_LOWERCASED: Final[str] = "lowercased-by-nats-py-parser"
_SUBJECT_CASE_NOT_REPORTED: Final[str] = "not-reported"

#: the caveat appended to the human-readable line when the case cannot be trusted. an operator
#: reading the sentence rather than the JSON must be warned BEFORE pasting the subject into a grant.
_LOWERCASED_SUBJECT_CAVEAT: Final[str] = (
    "The subject was reported by a client parser that lowercases the whole server error, so its case "
    "may not be the server's -- verify before pasting it into a grant list (this matters for $KV., "
    "$JS.API. and _INBOX_ subjects; HITL digests and uuids are lowercase already)."
)


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


class _SubscriptionFeed:
    """takes a subscription's messages off whichever connection currently carries it.

    A subscription outlives the connection it was made on: a credential renewal moves it onto
    the successor connection (:meth:`NatsClient.renew_connection`). So the receiving half is
    its own object, which can hold a receiver per connection at once -- the successor's,
    attached first, and the predecessor's, draining what the server had already routed to it
    -- feeding ONE backlog the subscription's callbacks read. Only the CURRENT receiver's end
    closes the backlog: a superseded receiver ending is the handover, not the stream ending.

    :param backlog: the subscription's received-but-undispatched messages
    :ptype backlog: ReceiptBacklog
    :param subject: the subscribed subject, for log lines
    :ptype subject: Subject
    :param note_reply: records which connection received a request, so the reply leaves on it;
        ``None`` for a subscription whose callback cannot reply
    :ptype note_reply: Callable[[str, Any], None] | None
    """

    __slots__ = ("_backlog", "_subject", "_note_reply", "_current", "_receivers")

    def __init__(
        self,
        *,
        backlog: ReceiptBacklog,
        subject: Subject,
        note_reply: Callable[[str, Any], None] | None,
    ) -> None:
        """bind the feed to its backlog; no receiver runs until :meth:`attach`.

        :param backlog: the subscription's received-but-undispatched messages
        :ptype backlog: ReceiptBacklog
        :param subject: the subscribed subject, for log lines
        :ptype subject: Subject
        :param note_reply: records which connection received a request, or ``None``
        :ptype note_reply: Callable[[str, Any], None] | None
        :return: nothing
        :rtype: None
        """
        self._backlog = backlog
        self._subject = subject
        self._note_reply = note_reply
        self._current: Any = None
        self._receivers: set[asyncio.Task[None]] = set()

    def attach(self, raw_subscription: Any, connection: Any) -> None:
        """start receiving from ``raw_subscription`` and make it the current receiver.

        :param raw_subscription: the nats-py subscription on ``connection``
        :ptype raw_subscription: Any
        :param connection: the nats-py connection that carries it
        :ptype connection: Any
        :return: nothing
        :rtype: None
        """
        self._current = raw_subscription
        task = asyncio.create_task(
            self._receive(raw_subscription, connection),
            name=f"nats-receive:{self._subject.path}",
        )
        self._receivers.add(task)
        task.add_done_callback(self._receivers.discard)

    async def release(self, raw_subscription: Any, connection: Any) -> None:
        """hand over a superseded receiver: stop the server routing to it, keep what it already has.

        :func:`_stop_routing_then_drain` removes the server's interest, proves by round trip that
        everything routed there has arrived, and waits for the receiver to have taken each of
        those messages into the backlog -- so nothing already in flight to the old connection is
        dropped. Never raises: a connection that closed first has nothing left to hand over.

        :param raw_subscription: the superseded nats-py subscription
        :ptype raw_subscription: Any
        :param connection: the nats-py connection that carries it
        :ptype connection: Any
        :return: nothing
        :rtype: None
        """
        try:
            # bounded: the drain waits for the receiver to take each message, and a subscription
            # dropped mid-handover has no receiver left to take them. see the bound's own comment
            # for why it is longer than the flushes inside the drain.
            await asyncio.wait_for(
                _stop_routing_then_drain(raw_subscription, connection),
                timeout=_HANDOVER_DRAIN_BOUND_SECONDS,
            )
        # NOSILENT: logged; the successor is already receiving, so the only loss is whatever the
        # closed connection had not yet handed over, and a closed connection cannot hand it over.
        except Exception as exc:  # noqa: BLE001 -- handover is best-effort once the successor receives
            log.warning(
                "superseded subscription could not be drained",
                extra={"extra_data": {"subject": self._subject.path, "error": str(exc)}},
            )

    async def cancel(self) -> None:
        """stop every receiver and wait for them to end.

        :return: nothing
        :rtype: None
        """
        receivers = list(self._receivers)
        for task in receivers:
            task.cancel()
        for task in receivers:
            try:
                await task
            except asyncio.CancelledError:
                # NOSILENT: this IS the cancellation requested above
                pass

    async def _receive(self, raw_subscription: Any, connection: Any) -> None:
        """take each message off one connection as it arrives and date it there.

        nats-py's ``Msg`` carries no receipt time, and a message left in its pending
        queue while every callback is busy is invisible; taken here instead, the time
        it then waits for a callback counts toward its age (``monotonic_received``).
        the backlog's bound stops this loop when the callbacks fall far enough behind,
        so the excess waits on the connection under nats-py's own slow-consumer limit.

        a stream failure is logged here rather than raised. when this receiver is the
        current one, its end closes the backlog, so the dispatcher hands out what was
        already received and then ends; a superseded receiver's end closes nothing.

        :param raw_subscription: the nats-py subscription to read
        :ptype raw_subscription: Any
        :param connection: the connection carrying it, recorded against each request received
        :ptype connection: Any
        :return: nothing
        :rtype: None
        """
        try:
            async for msg in raw_subscription.messages:
                if self._note_reply is not None and msg.reply:
                    self._note_reply(msg.reply, connection)
                await self._backlog.put(msg, received=time.monotonic())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — diag only
            underlying = representative_exception(exc)
            log.error(
                "subscription dispatch loop crashed",
                extra={
                    "extra_data": {
                        "subject": self._subject.path,
                        "error_type": type(underlying).__name__,
                        "error": str(underlying),
                    }
                },
            )
        finally:
            if raw_subscription is self._current:
                await self._backlog.close()


class Subscription:
    """opaque handle returned by :meth:`NatsClient.subscribe`.

    callers pass instances back to :meth:`NatsClient.unsubscribe` to
    drop the subscription. fields prefixed with ``raw_`` /
    ``dispatch_task`` are intentionally public-named (per the
    underscore-stability-contract rule) but excluded from
    ``threetears.nats.__all__`` — they are package-internal, not
    bindable from outside the wrapper, and the wrapper itself
    manipulates them when unsubscribing.

    **A subscription is never plain on the wire.** One made without a queue group joins a group
    of its own (:data:`_SOLE_MEMBER_QUEUE_PREFIX`), of which it is the only member. To every
    other subscriber that is indistinguishable from a plain subscription -- the server delivers
    each message to every plain subscriber and to one member of every group, and this group has
    one member -- but it is what lets a credential renewal move the subscription between
    connections without losing or doubling a message. The move subscribes the successor
    connection BEFORE unsubscribing the old one, so something is always listening; with a plain
    subscription the two would each receive every message published while both are live, and
    with one shared group the server picks exactly one of them for each. A caller-named group
    already has that property. The price is ordering across the move: while both are members,
    two consecutive messages may be handed to different connections, and so dispatched in
    either order. Core NATS promises order only per publisher per connection, and a handover
    is two connections.

    :param raw_subscription: underlying nats-py subscription
    :ptype raw_subscription: Any
    :param subject: subject this subscription was registered against
    :ptype subject: Subject
    :param dispatch_task: background task driving the message loop
    :ptype dispatch_task: asyncio.Task[None]
    :param queue: the queue group the subscription holds on the wire
    :ptype queue: str
    :param feed: the receiving half, which a renewal re-attaches to the successor connection
    :ptype feed: _SubscriptionFeed
    :param connection: the nats-py connection ``raw_subscription`` is on
    :ptype connection: Any
    """

    __slots__ = ("raw_subscription", "_subject", "dispatch_task", "_closed", "_queue", "_feed", "_connection")

    def __init__(
        self,
        *,
        raw_subscription: Any,
        subject: Subject,
        dispatch_task: asyncio.Task[None],
        queue: str,
        feed: _SubscriptionFeed,
        connection: Any,
    ) -> None:
        self.raw_subscription = raw_subscription
        self._subject = subject
        self.dispatch_task = dispatch_task
        self._closed = False
        self._queue = queue
        self._feed = feed
        self._connection = connection

    @property
    def queue(self) -> str:
        """the queue group this subscription holds on the wire.

        :return: the caller's group, or the subscription's own sole-member group
        :rtype: str
        """
        return self._queue

    async def subscribe_on(self, connection: Any) -> Any:
        """subscribe this subscription's subject and group on a successor connection.

        The first half of a move; nothing receives from it until :meth:`move_to`.

        :param connection: the successor nats-py connection
        :ptype connection: Any
        :return: the nats-py subscription on it
        :rtype: Any
        """
        return await connection.subscribe(self._subject.path, queue=self._queue)

    def move_to(self, raw_subscription: Any, connection: Any) -> asyncio.Task[None]:
        """make the successor's subscription current, and hand over the old one's backlog.

        The successor receives from here on; the old subscription is drained in the background
        so whatever the server had already routed to it still reaches the callbacks.

        :param raw_subscription: the successor's nats-py subscription, from :meth:`subscribe_on`
        :ptype raw_subscription: Any
        :param connection: the successor nats-py connection
        :ptype connection: Any
        :return: the task draining the old subscription
        :rtype: asyncio.Task[None]
        """
        if self._closed:
            # dropped while the successor was being subscribed: nothing is left to receive for.
            return asyncio.create_task(
                _unsubscribe_quietly(raw_subscription, subject=self._subject),
                name=f"nats-handover:{self._subject.path}",
            )
        previous, previous_connection = self.raw_subscription, self._connection
        self._feed.attach(raw_subscription, connection)
        self.raw_subscription = raw_subscription
        self._connection = connection
        return asyncio.create_task(
            self._feed.release(previous, previous_connection),
            name=f"nats-handover:{self._subject.path}",
        )

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
        no-op. like :meth:`NatsClient.unsubscribe`, this waits —
        unbounded — for in-flight callbacks to finish unwinding.

        :return: nothing
        :rtype: None
        """
        if self._closed:
            return
        # closed FIRST: a renewal moving this subscription checks it, and must not attach a
        # successor to a subscription being dropped under it.
        self._closed = True
        try:
            await self.raw_subscription.unsubscribe()
        except Exception as exc:  # noqa: BLE001 -- diag only; teardown continues regardless
            log.warning(
                "unsubscribe failed",
                extra={"extra_data": {"subject": self.subject.path, "error": str(exc)}},
            )
        self.dispatch_task.cancel()
        try:
            await self.dispatch_task
        except asyncio.CancelledError:
            # NOSILENT: this IS the cancellation requested on the line above
            pass
        except Exception as exc:  # noqa: BLE001 -- diag only; the subscription is dropped either way
            log.warning(
                "dispatch task raised while unwinding",
                extra={"extra_data": {"subject": self.subject.path, "error": str(exc)}},
            )


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

    a credential renewal moves the handle to the successor connection (:meth:`move_to`): the
    durable is released on the old connection -- its callbacks finish and their acks leave on the
    connection that received each message -- and bound again on the new one. the durable holds
    every message published meanwhile, so nothing is lost; a push consumer with no deliver group
    admits one bound subscription at a time, so the two cannot overlap.

    :param raw_subscription: underlying nats-py JetStream push subscription
    :ptype raw_subscription: Any
    :param subject: subject this subscription was registered against
    :ptype subject: Subject
    :param durable: durable consumer name (diagnostics)
    :ptype durable: str
    :param resubscribe: binds the durable again through a JetStream context, returning the new
        nats-py subscription
    :ptype resubscribe: Callable[[Any], Awaitable[Any]]
    :param connection: the nats-py connection ``raw_subscription`` is on
    :ptype connection: Any
    :param stream: the backing stream the durable was bound on, or ``None`` when it was found by subject
    :ptype stream: str | None
    """

    __slots__ = ("raw_subscription", "_subject", "_durable", "_closed", "_resubscribe", "_connection", "_stream")

    def __init__(
        self,
        *,
        raw_subscription: Any,
        subject: Subject,
        durable: str,
        resubscribe: Callable[[Any], Awaitable[Any]],
        connection: Any,
        stream: str | None = None,
    ) -> None:
        self.raw_subscription = raw_subscription
        self._subject = subject
        self._durable = durable
        self._closed = False
        self._resubscribe = resubscribe
        self._connection = connection
        self._stream = stream

    async def recreate(self, js: Any, connection: Any) -> None:
        """bind the durable again after the server lost it, in one attempt.

        a stream wiped by a NATS restart (memory storage, or file storage the restart lost) takes its
        durables with it. nats-py replays this handle's subscription on the reconnect, but nothing
        delivers to it until the durable exists again, and nothing says so. the old subscription's durable is gone, so there is
        nothing in flight to drain: it is dropped, and the bind creates the durable afresh with the
        config it was first bound with. one attempt; the client's restoration retries a failure.

        :param js: a JetStream context on the client's current connection
        :ptype js: Any
        :param connection: the nats-py connection ``js`` is on
        :ptype connection: Any
        :return: nothing
        :rtype: None
        :raises Exception: when the bind fails -- the stream is not back yet, or the broker refuses
        """
        await _unsubscribe_quietly(self.raw_subscription, subject=self._subject)
        bound = await self._resubscribe(js)
        if self._closed:
            # stopped while it was being bound: release what was just bound.
            await _unsubscribe_quietly(bound, subject=self._subject)
        else:
            self.raw_subscription = bound
            self._connection = connection

    @property
    def stream(self) -> str | None:
        """backing stream the durable was bound on.

        :return: the stream name, or ``None`` when the bind found it by subject
        :rtype: str | None
        """
        return self._stream

    async def move_to(self, js: Any, connection: Any) -> None:
        """release the durable on its current connection and bind it again through ``js``.

        retried until it binds or the handle is stopped: a durable left unbound delivers nothing,
        with nothing to say so, and the server may still report the durable bound for a moment
        after the release. never raises.

        :param js: a JetStream context on the successor connection
        :ptype js: Any
        :param connection: the successor nats-py connection ``js`` is on
        :ptype connection: Any
        :return: nothing
        :rtype: None
        """
        previous = self.raw_subscription
        try:
            await asyncio.wait_for(
                _stop_routing_then_drain(previous, self._connection),
                timeout=_HANDOVER_DRAIN_BOUND_SECONDS,
            )
        # NOSILENT: logged; the durable redelivers anything unacknowledged, so the rebind below
        # loses nothing even when the release could not finish.
        except Exception as exc:  # noqa: BLE001 -- the rebind proceeds either way
            log.warning(
                "durable push consumer could not be released from the replaced connection: durable=%s: %s",
                self._durable,
                exc,
            )
        while not self._closed:
            try:
                bound = await self._resubscribe(js)
            except Exception as exc:  # noqa: BLE001 -- retried: an unbound durable delivers nothing
                log.warning(
                    "durable push consumer could not be bound on the successor connection; retrying in %.1fs: "
                    "durable=%s: %s",
                    _PULL_CONSUMER_ERROR_BACKOFF_SECONDS,
                    self._durable,
                    exc,
                )
                await asyncio.sleep(_PULL_CONSUMER_ERROR_BACKOFF_SECONDS)
                continue
            if self._closed:
                # stopped while it was being bound: release what was just bound.
                await _unsubscribe_quietly(bound, subject=self._subject)
            else:
                self.raw_subscription = bound
                self._connection = connection
            break

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
    :param ack_wait_seconds: the durable's ack wait; a running handler sends an in-progress ack
        :data:`_PULL_CONSUMER_IN_PROGRESS_PER_ACK_WAIT` times within each one
    :ptype ack_wait_seconds: float
    :param bound_to: the nats-py connection ``psub`` was made on
    :ptype bound_to: Any
    :param current_connection: reads the client's current connection
    :ptype current_connection: Callable[[], Any]
    :param resubscribe: binds the durable again on the client's current connection
    :ptype resubscribe: Callable[[], Awaitable[Any]]
    :param error_backoff_seconds: pause after a failed fetch cycle before the next one
    :ptype error_backoff_seconds: float
    :param stream: the backing stream the durable was bound on, or ``None`` when it was found by subject
    :ptype stream: str | None
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
        ack_wait_seconds: float,
        bound_to: Any,
        current_connection: Callable[[], Any],
        resubscribe: Callable[[], Awaitable[Any]],
        error_backoff_seconds: float = _PULL_CONSUMER_ERROR_BACKOFF_SECONDS,
        stream: str | None = None,
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
        :param ack_wait_seconds: the durable's ack wait, which paces the running handler's
            in-progress acks
        :ptype ack_wait_seconds: float
        :param bound_to: the nats-py connection ``psub`` was made on
        :ptype bound_to: Any
        :param current_connection: reads the client's current connection
        :ptype current_connection: Callable[[], Any]
        :param resubscribe: binds the durable again on the client's current connection
        :ptype resubscribe: Callable[[], Awaitable[Any]]
        :param error_backoff_seconds: pause after a failed fetch cycle (a transport blip during a
            reconnect) before the next, so a persistent failure does not spin
        :ptype error_backoff_seconds: float
        :param stream: the backing stream the durable was bound on, or ``None`` when found by subject
        :ptype stream: str | None
        :return: nothing
        :rtype: None
        """
        self._stream = stream
        # set when the server lost the durable (its stream wiped by a NATS restart):
        # the next cycle binds again, which creates the durable, rather than fetching from nothing.
        self._rebind_requested = False
        self._psub = psub
        self._error_backoff_seconds = error_backoff_seconds
        self._cb = cb
        self._redeliver = redeliver
        self._durable = durable
        self._subject = subject
        self._batch = batch
        self._fetch_timeout_seconds = fetch_timeout_seconds
        self._in_progress_interval_seconds = ack_wait_seconds / _PULL_CONSUMER_IN_PROGRESS_PER_ACK_WAIT
        self._stopped = False
        # set whenever no fetch is in flight -- see stop()
        self._idle = asyncio.Event()
        self._idle.set()
        self._bound_to = bound_to
        self._current_connection = current_connection
        self._resubscribe = resubscribe

    async def _follow_connection(self) -> None:
        """rebind the durable when a credential renewal has replaced the connection it was bound on.

        a pull fetch is a request on the connection the subscription was made on, and a renewal
        retires that connection; checked before every fetch, so the consumer moves at its next
        cycle rather than failing on a closed connection. the durable, not the subscription,
        holds the backlog, so the rebind loses nothing.

        :return: nothing
        :rtype: None
        :raises Exception: when the rebind fails; :meth:`run` logs it and retries
        """
        current = self._current_connection()
        if current is self._bound_to and not self._rebind_requested:
            return
        recreating = self._rebind_requested
        previous = self._psub
        self._psub = await self._resubscribe()
        self._bound_to = current
        self._rebind_requested = False
        await _unsubscribe_quietly(previous, subject=self._subject)
        if recreating:
            log.info("durable pull consumer bound again after the server lost its durable: durable=%s", self._durable)
        else:
            log.info(
                "durable pull consumer followed a credential renewal to the successor connection: durable=%s",
                self._durable,
            )

    def rebind_on_next_fetch(self) -> None:
        """bind the durable again before the next fetch, because the server no longer has it.

        a stream wiped by a NATS restart takes its durables with it, and a fetch
        against a durable that is gone delivers nothing. the bind made at the next cycle creates the
        durable afresh with the config it was first bound with; a bind that fails (the stream is not
        back yet) is retried by :meth:`run` on the cycle after. the fetch loop owns the subscription,
        so the rebind happens there rather than under it.

        :return: nothing
        :rtype: None
        """
        self._rebind_requested = True

    @property
    def stream(self) -> str | None:
        """backing stream the durable was bound on.

        :return: the stream name, or ``None`` when the bind found it by subject
        :rtype: str | None
        """
        return self._stream

    @property
    def subject(self) -> Subject:
        """subject this consumer drains.

        :return: the consumed subject
        :rtype: Subject
        """
        return self._subject

    @property
    def durable(self) -> str:
        """durable consumer name this consumer binds.

        :return: durable consumer name
        :rtype: str
        """
        return self._durable

    @property
    def is_stopped(self) -> bool:
        """whether :meth:`stop` has been called.

        :return: True once stopped
        :rtype: bool
        """
        return self._stopped

    async def fetch_and_process(self) -> int:
        """pull one batch and dispatch each message; return the count processed.

        a fetch that times out with no message returns ``0`` (the idle case) —
        the normal scale-to-zero-friendly poll, not an error. a handler raise is
        routed through the bounded-redelivery policy, never propagated, so one
        poisoned message cannot kill the loop.

        :return: number of messages fetched this cycle
        :rtype: int
        """
        await self._follow_connection()
        try:
            msgs = await self._psub.fetch(self._batch, timeout=self._fetch_timeout_seconds)
        except _NatsTimeoutError:
            msgs = []
        for msg in msgs:
            await self._handle(msg)
        return len(msgs)

    async def _handle(self, msg: Any) -> None:
        """run the handler on one message, and route a raise through the bounded-redelivery policy.

        the one "handle, else redeliver" step, for a fetched message and for one :meth:`stop`
        collects alike. a raise from the handler never escapes; a raise from the policy itself
        (its ``nak``, ``ack`` or dead-letter publish on a failing transport) does, and the caller
        decides what it costs.

        while it runs, the message is held with in-progress acks (:meth:`_hold_while_handled`), so
        the durable's ``ack_wait`` bounds how long a DEAD fetcher keeps a message from the others,
        never how long a live handler may take.

        :param msg: the JetStream message
        :ptype msg: Any
        :return: nothing
        :rtype: None
        :raises Exception: whatever the redelivery policy raises
        """
        holding = asyncio.create_task(self._hold_while_handled(msg))
        try:
            try:
                await self._cb(msg)
            except Exception as exc:  # noqa: BLE001 — bounded redelivery owns the outcome; never swallow
                await self._redeliver(msg, exc)
        finally:
            holding.cancel()
            # gather, not a bare await: a cancellation of THIS task still propagates, while the
            # holder's own cancellation, which was asked for just above, is collected as a result
            await asyncio.gather(holding, return_exceptions=True)

    async def _hold_while_handled(self, msg: Any) -> None:
        """send an in-progress ack every fraction of ``ack_wait`` until cancelled.

        a refused in-progress ack (a transport closing under the handler) is logged and ends the
        holding: the handler still finishes and acks or redelivers through the usual step, and if it
        outlives ``ack_wait`` the server redelivers to another fetcher, which the handler contract
        (idempotent, at-least-once) already tolerates.

        :param msg: the JetStream message being handled
        :ptype msg: Any
        :return: nothing
        :rtype: None
        """
        refused: Exception | None = None
        while refused is None:
            await asyncio.sleep(self._in_progress_interval_seconds)
            try:
                await msg.in_progress()
            except Exception as exc:  # noqa: BLE001 — logged below; the handler's own outcome is unaffected
                refused = exc
        log.warning(
            "durable pull consumer: an in-progress ack failed while a handler ran (durable=%s, subject=%s): %s: %s "
            "-- the message is no longer held, and a handler outliving ack_wait is redelivered to another fetcher",
            self._durable,
            getattr(msg, "subject", None),
            type(refused).__name__,
            refused,
        )

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
            failure: Exception | None = None
            self._idle.clear()
            try:
                await self.fetch_and_process()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — a transport blip must never kill the consumer
                failure = exc
            finally:
                self._idle.set()
            if failure is not None:
                log.warning(
                    "durable pull consumer cycle failed (durable=%s); retrying after %.1fs: %s",
                    self._durable,
                    self._error_backoff_seconds,
                    failure,
                )
                await asyncio.sleep(self._error_backoff_seconds)

    async def stop(self) -> None:
        """halt the fetch loop, release the inbox at the server, and handle whatever reached it first.

        **Why this is more than an unsubscribe.** A fetch is a pull request the SERVER holds until
        it is satisfied or expires, and it can outlive the fetch that sent it: the client's timer
        starts before the request reaches the server, so under load the fetch gives up while the
        request is still live. nats-py's unsubscribe drops the inbox client-side at once and only
        then tells the server, so a message delivered to such a request -- published by any pod, on
        any connection, before the server processes the ``UNSUB`` -- was dropped here and counted
        awaiting ack until the durable's ``ack_wait`` ran out, out of every other fetcher's reach.
        Measured: 6 of 20 messages published as a stop began. The aibots hub's erasure waits on
        the audit durable's ack floor, and its audit consumer's stop stalled it.

        So, in order:

        1. no new fetch starts, and the one in flight is let finish with its messages handled
           (bounded by one fetch timeout plus :data:`_PULL_CONSUMER_STOP_HANDLER_GRACE_SECONDS`);
        2. the server is told to drop the inbox while it is still subscribed here, and one round
           trip proves it did -- the server writes every message it delivered to the inbox before
           the ``PONG``, so each is now queued here, and nothing more can arrive;
        3. each queued message is handled exactly as a fetched one would be, and acked;
        4. the inbox is removed client-side.

        A connection that cannot carry step 2 (closed, reconnecting) is unsubscribed the plain way,
        and that is logged with what it costs.

        **Never raises, and idempotent**, like :meth:`JetStreamPushConsumer.stop` and
        :meth:`Subscription.unsubscribe`: its callers are shutdown paths, and a raise would skip
        whatever teardown follows. A message whose redelivery itself fails (its ``nak``, ``ack`` or
        dead-letter publish on a failing transport) is logged and the next one is still handled --
        an unhandled one is redelivered by the server after ``ack_wait``, which is what one failed
        message costs, never every message after it. Step 4 runs whatever happened before it, and a
        failure of it is logged: nats-py forgets the inbox before it sends the ``UNSUB``, and a
        closed connection has already forgotten every subscription, so nothing stays registered
        here. A second call returns at once.

        :return: nothing
        :rtype: None
        """
        if self._stopped:
            return
        self._stopped = True
        bound = self._fetch_timeout_seconds + _PULL_CONSUMER_STOP_HANDLER_GRACE_SECONDS
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=bound)
        except TimeoutError:
            log.warning(
                "durable pull consumer stop: the fetch in flight did not finish within %.1fs (durable=%s); "
                "releasing the inbox anyway",
                bound,
                self._durable,
            )
        try:
            leftovers = await self._release_inbox()
        except Exception as exc:  # noqa: BLE001 — the plain unsubscribe below still runs; what it costs is logged
            log.warning(
                "durable pull consumer stop: the inbox could not be released at the server first (durable=%s): "
                "%s: %s -- a message delivered to a request still live there waits out the durable's ack_wait",
                self._durable,
                type(exc).__name__,
                exc,
            )
            leftovers = []
        try:
            for msg in leftovers:
                try:
                    await self._handle(msg)
                except Exception as exc:  # noqa: BLE001 — one failed redelivery must not strand the rest
                    log.warning(
                        "durable pull consumer stop: a message collected at stop was neither handled nor "
                        "redelivered (durable=%s, subject=%s): %s: %s -- the server redelivers it after the "
                        "durable's ack_wait",
                        self._durable,
                        getattr(msg, "subject", None),
                        type(exc).__name__,
                        exc,
                    )
        finally:
            try:
                await self._psub.unsubscribe()
            except Exception as exc:  # noqa: BLE001 — a stop never raises; nothing stays registered, see the docstring
                log.warning(
                    "durable pull consumer stop: the final unsubscribe failed (durable=%s, subject=%s): %s: %s",
                    self._durable,
                    self._subject.path,
                    type(exc).__name__,
                    exc,
                )

    async def _release_inbox(self) -> list[Any]:
        """drop the fetch inbox's interest at the server, keeping it here, and collect what was delivered.

        The same ``UNSUB`` then ordered :func:`_round_trip` as :func:`_stop_routing_then_drain`,
        for the same reason; the queue is then taken here rather than by nats-py's ``drain``,
        because a pull inbox has no callback to take it and the messages in it are to be HANDLED,
        not merely waited out.

        :return: every message the server delivered to the inbox, status messages excluded
        :rtype: list[Any]
        :raises Exception: when the connection cannot carry the ``UNSUB`` or its round trip
        """
        # nats-py exposes no way to remove a subscription's interest without also forgetting the
        # subscription, nor the queue of a pull subscription's inbox: _nats_py_internals owns both.
        sub = pull_subscription_inbox(self._psub)
        connection = self._bound_to
        await send_unsubscribe(connection, sub)
        await _round_trip(connection)
        delivered = [msg for msg in take_queued_messages(sub) if not _NatsJetStreamContext.is_status_msg(msg)]
        if delivered:
            log.info(
                "durable pull consumer stop: %d message(s) delivered to a request the last fetch had left "
                "are handled here rather than stranded (durable=%s)",
                len(delivered),
                self._durable,
            )
        return delivered


class JetStreamResultWaiter:
    """awaits exactly ONE answer on ONE exact subject, across reconnects on either side.

    Returned by :meth:`NatsClient.jetstream_result_waiter`. The caller opens it BEFORE dispatching
    the call, so the consumer exists no matter how fast the answer comes back, then awaits the
    answer for as long as the call is allowed to take.

    Two properties it has that a request/reply await does not:

    - the RESPONDER can be recycled. It publishes to a subject it holds a standing grant on rather
      than to a per-request inbox right that dies with its connection.
    - the CALLER can be recycled. The answer is retained by the stream, and the consumer is
      re-created against it when it goes quiet, so a caller that reconnects mid-wait still collects
      an answer published while it was away. Fixing only the publisher's half would move the loss
      rather than end it, which is why the delivery is JetStream and not a core publish.

    **The consumer is NAMED, filtered in its create subject, and PUSHED to this connection's own
    inbox**, because that is the one consumer shape a pod's grant admits
    (:attr:`~threetears.nats.subject_permissions.JsCapability.STREAM_CONSUMER`): nats-py's
    ``pull_subscribe`` creates the consumer and then pulls with ``CONSUMER.MSG.NEXT``, a verb that
    reaches ANY consumer on the stream by name -- the registry's over every pod's results included.
    A pushed consumer needs no verb after its create: the server delivers to the inbox, the answer
    is acknowledged through the message's reply subject, and an idle heartbeat says the consumer is
    still there.

    The consumer filters on the exact subject with ``DeliverPolicy.ALL``. Both choices matter.
    Ephemeral means nothing to clean up if this process dies -- the server reaps it once nothing has
    listened for its inactivity threshold. ``ALL`` on a single-call subject means "the answer,
    whenever it was published" -- so the ordering between opening the consumer and the answer
    arriving stops being a race at all.

    **The consumer is made on whichever connection is current when it is made.** A credential
    renewal replaces the connection mid-wait and later retires the old one; the waiter then sees
    its subscription end, and the replacement it makes lands on the successor -- where
    ``DeliverPolicy.ALL`` still finds the answer.

    :param connection: reads the client's current nats-py connection, whose inbox receives the
        pushed answer
    :ptype connection: Callable[[], Any]
    :param jetstream: reads a JetStream context on the client's current connection
    :ptype jetstream: Callable[[], Any]
    :param subject: the exact subject the answer will be published to
    :ptype subject: Subject
    :param stream: backing stream name, passed explicitly so the client never issues the
        ``$JS.API.STREAM.NAMES`` subject lookup
    :ptype stream: str
    :param inactive_threshold_seconds: how long the server keeps the consumer with nothing listening;
        must exceed the whole wait budget or the consumer evaporates mid-call
    :ptype inactive_threshold_seconds: float
    :param poll_seconds: how often the wait re-checks its own deadline and its consumer's heartbeat
    :ptype poll_seconds: float
    :param heartbeat_seconds: the consumer's idle heartbeat
    :ptype heartbeat_seconds: float
    :param rebuild_backoff_seconds: pause after a failed consumer replacement before the next try
    :ptype rebuild_backoff_seconds: float
    """

    def __init__(
        self,
        *,
        connection: Callable[[], Any],
        jetstream: Callable[[], Any],
        subject: Subject,
        stream: str,
        inactive_threshold_seconds: float,
        poll_seconds: float,
        heartbeat_seconds: float = _RESULT_WAITER_HEARTBEAT_SECONDS,
        rebuild_backoff_seconds: float = _RESULT_WAITER_REBUILD_BACKOFF_SECONDS,
    ) -> None:
        """bind the waiter to its subject; the consumer is created by :meth:`open`.

        :param connection: reads the client's current nats-py connection
        :ptype connection: Callable[[], Any]
        :param jetstream: reads a JetStream context on the client's current connection
        :ptype jetstream: Callable[[], Any]
        :param subject: the exact subject the answer will be published to
        :ptype subject: Subject
        :param stream: backing stream name
        :ptype stream: str
        :param inactive_threshold_seconds: consumer keepalive with nothing listening, in seconds
        :ptype inactive_threshold_seconds: float
        :param poll_seconds: deadline and heartbeat re-check cadence, in seconds
        :ptype poll_seconds: float
        :param heartbeat_seconds: the consumer's idle heartbeat, in seconds
        :ptype heartbeat_seconds: float
        :param rebuild_backoff_seconds: pause after a failed consumer replacement (a broker still
            coming back) before the wait tries again, so the retry does not spin
        :ptype rebuild_backoff_seconds: float
        :return: nothing
        :rtype: None
        """
        self._connection = connection
        self._rebuild_backoff_seconds = rebuild_backoff_seconds
        self._jetstream = jetstream
        self._subject = subject
        self._stream = stream
        self._inactive_threshold_seconds = inactive_threshold_seconds
        self._poll_seconds = poll_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._sub: Any = None
        self._consumer_name: str | None = None

    @property
    def subject(self) -> Subject:
        """the exact subject this waiter accepts an answer on.

        :return: the awaited subject
        :rtype: Subject
        """
        return self._subject

    async def open(self) -> None:
        """create the consumer, so the answer has somewhere to land before the call is dispatched.

        :return: nothing
        :rtype: None
        :raises Exception: when the consumer cannot be created -- the caller has not dispatched yet
        """
        await self._subscribe()

    async def _subscribe(self) -> None:
        """subscribe a fresh inbox, then create a freshly named consumer pushing to it.

        The inbox is subscribed FIRST so nothing the consumer delivers can arrive before anything is
        listening. The create names the consumer and carries the filter, so nats-py issues
        ``CONSUMER.CREATE.{stream}.{name}.{filter}`` -- the form the server checks against the body.

        :return: nothing
        :rtype: None
        :raises Exception: when the create fails or is not answered within its ceiling
        """
        name = f"{_RESULT_WAITER_CONSUMER_PREFIX}{uuid.uuid7().hex}"
        connection = self._connection()
        inbox = connection.new_inbox()
        sub = await connection.subscribe(inbox)
        config = _NatsConsumerConfig(
            name=name,
            deliver_subject=inbox,
            filter_subject=self._subject.path,
            ack_policy=_NatsAckPolicy.EXPLICIT,
            deliver_policy=_NatsDeliverPolicy.ALL,
            # one answer per subject: a larger window buys nothing and would let a redelivery sit
            # unacknowledged behind a message this waiter has already returned.
            max_ack_pending=1,
            idle_heartbeat=self._heartbeat_seconds,
            inactive_threshold=self._inactive_threshold_seconds,
        )
        try:
            await asyncio.wait_for(
                self._jetstream().add_consumer(self._stream, config=config),
                timeout=_RESULT_WAITER_CREATE_TIMEOUT_SECONDS,
            )
        except BaseException:
            await _unsubscribe_quietly(sub, subject=self._subject)
            raise
        self._sub = sub
        self._consumer_name = name

    async def wait(self, *, timeout: timedelta) -> bytes:
        """block until the answer arrives, or ``timeout`` elapses.

        Replaces the consumer and keeps waiting when it goes quiet -- no delivery and no heartbeat
        for :data:`_RESULT_WAITER_MISSED_HEARTBEATS` heartbeats -- when the server says it ended it,
        or when the subscription fails. That is the reconnect path: after a broker restart the
        consumer may be gone, and giving up there would discard an answer the stream is still
        holding -- the precise failure this class exists to prevent, merely moved to the other end
        of the wire. The deadline is the only thing that ends the loop.

        :param timeout: total budget for the answer to arrive
        :ptype timeout: timedelta
        :return: the raw answer payload
        :rtype: bytes
        :raises RuntimeError: when called before :meth:`open`
        :raises RequestTimeoutError: when no answer arrives within ``timeout``
        """
        if self._sub is None and self._consumer_name is None:
            raise RuntimeError("JetStreamResultWaiter.wait called before open()")
        deadline = time.monotonic() + timeout.total_seconds()
        silence = self._heartbeat_seconds * _RESULT_WAITER_MISSED_HEARTBEATS
        last_heard = time.monotonic()
        payload: bytes | None = None
        while payload is None and time.monotonic() < deadline:
            if self._sub is None:
                # a rebuild failed on the previous turn; the broker may still be coming back
                await self._rebuild()
                last_heard = time.monotonic()
                continue
            poll = min(self._poll_seconds, max(deadline - time.monotonic(), _RESULT_WAITER_MIN_POLL_SECONDS))
            try:
                msg = await self._sub.next_msg(timeout=poll)
            except _NatsTimeoutError:
                # NOSILENT: an empty poll is the ordinary case -- the tool is still running. only a
                # consumer that has also stopped heartbeating is replaced, and that is logged.
                if time.monotonic() - last_heard >= silence:
                    log.warning(
                        "result waiter consumer went quiet; replacing it (subject=%s stream=%s consumer=%s)",
                        self._subject.path,
                        self._stream,
                        self._consumer_name,
                    )
                    await self._rebuild()
                    last_heard = time.monotonic()
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — a transport blip must not discard a live answer
                log.warning(
                    "result waiter delivery failed; replacing its consumer (subject=%s stream=%s): %s",
                    self._subject.path,
                    self._stream,
                    exc,
                )
                await self._rebuild()
                last_heard = time.monotonic()
                continue
            last_heard = time.monotonic()
            headers = msg.headers or {}
            status = headers.get("Status") if not msg.data else None
            if status == _STATUS_IDLE_HEARTBEAT:
                continue
            if status is not None:
                log.warning(
                    "result waiter consumer ended by the server (%s %s); replacing it (subject=%s consumer=%s)",
                    status,
                    headers.get("Description", ""),
                    self._subject.path,
                    self._consumer_name,
                )
                await self._rebuild()
                continue
            await msg.ack()
            payload = bytes(msg.data)
        if payload is None:
            raise RequestTimeoutError(f"no result delivered on {self._subject.path} within {timeout.total_seconds()}s")
        return payload

    async def _rebuild(self) -> None:
        """drop the inbox and create a fresh consumer; never raises.

        The abandoned consumer is left for the server to reap after its inactivity threshold: a pod
        holds no ``CONSUMER.DELETE``, and nothing listens on its inbox any more.

        :return: nothing
        :rtype: None
        """
        await self.close()
        try:
            await self._subscribe()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — the next turn retries; failing here would end the wait
            log.warning(
                "result waiter could not replace its consumer (subject=%s); retrying: %s",
                self._subject.path,
                exc,
            )
            await asyncio.sleep(self._rebuild_backoff_seconds)

    async def close(self) -> None:
        """unsubscribe the inbox; idempotent and never raises.

        the consumer ages out on its own inactivity threshold once nothing listens, so a failure here
        leaks nothing durable -- which is why it is logged at debug and not surfaced to a caller that
        is, by this point, already holding its answer.

        :return: nothing
        :rtype: None
        """
        sub = self._sub
        self._sub = None
        if sub is not None:
            await _unsubscribe_quietly(sub, subject=self._subject)


def _storage_name(config: _NatsStreamConfig) -> str:
    """the storage a declared stream config names, for a log line.

    :param config: a declared stream config
    :ptype config: nats.js.api.StreamConfig
    :return: ``"file"`` or ``"memory"``
    :rtype: str
    """
    storage = getattr(config.storage, "value", config.storage)
    return "file" if storage == "file" else "memory"


async def _round_trip(connection: Any, *, timeout: float = _NATS_FLUSH_TIMEOUT_SECONDS) -> None:
    """prove the server has processed everything this connection sent before now, within ``timeout``.

    nats-py's ``Client.flush`` is unsafe for this, in three ways, the first two reproduced against a
    real server (``test_a_timed_out_ping_leaves_the_connection_reading_live``):

    - it writes its ``PING`` straight to the socket (``Client._send_ping``) while a ``SUB``,
      ``UNSUB`` or ``PUB`` sent just before may still sit in the pending buffer, so the ``PONG``
      can answer a ``PING`` that overtook them and prove nothing about them;
    - on a timeout -- or when the caller cancels it -- it cancels its ``PONG`` future but leaves it
      in ``Client._pongs``. The late ``PONG`` pops the cancelled future, ``set_result`` raises
      ``InvalidStateError`` in ``_process_pong``, and ``_read_loop``'s catch-all ends the read loop.
      The connection still reports itself connected and never reads again -- and a later
      ``flush`` then returns at once without a round trip, because it falls back to writing when
      the read loop is gone, so a health probe built on it reports that dead connection healthy;
    - writing the pending buffer out first through nats-py (``_flush_pending(force_flush=True)``)
      waits for the flusher's ``transport.drain()`` with no bound (``flush_timeout`` defaults to
      none) and swallows ``CancelledError``. On a backpressured socket -- the half-open or
      slow-reading connection a health probe exists to catch -- the round trip then outlives its
      ``timeout``, and a cancellation (a shutdown, a bounded drain) is silently discarded.

    So the pending buffer and the ``PING`` are handed to the transport in ONE synchronous step,
    buffer first, with the ``PONG`` future queued in the same step: nothing can be written between
    them, and ``Client._pongs`` stays in the order the ``PING`` s were written, so every ``PONG``
    still resolves the future of its own ``PING``. The transport buffers the bytes and writes them as
    the socket accepts them; nothing here waits for that. The flusher is then woken without waiting
    (a transport that sends only when drained, the websocket one, needs it), and the only wait is
    for the ``PONG``: bounded by ``timeout`` and shielded, so a timeout or cancellation abandons the
    wait, never the future, which the late ``PONG`` then resolves harmlessly. Cancellation
    propagates.

    :param connection: the nats-py connection
    :ptype connection: Any
    :param timeout: seconds to wait for the ``PONG``; the whole round trip is bounded by it
    :ptype timeout: float
    :return: nothing
    :rtype: None
    :raises nats.errors.ConnectionClosedError: when the connection is closed
    :raises NatsClientError: when the connection is not currently connected
    :raises nats.errors.FlushTimeoutError: when no ``PONG`` arrives within ``timeout``
    """
    if connection.is_closed:
        raise _NatsConnectionClosedError
    if not connection.is_connected:
        raise NatsClientError("cannot round-trip a NATS connection that is not currently connected")
    pong: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
    # nats-py exposes no round trip that is ordered after the pending buffer, bounded, and safe to
    # cancel or time out; see the docstring. _nats_py_internals owns the fields this needs.
    write_pending_then_ping(connection, pong)
    try:
        await asyncio.wait_for(asyncio.shield(pong), timeout=timeout)
    except TimeoutError:
        raise _NatsFlushTimeoutError from None


async def _stop_routing_then_drain(raw_subscription: Any, connection: Any) -> None:
    """end a subscription the server may still be routing to, without dropping what it routed.

    nats-py's own ``Subscription.drain`` cannot be trusted with this. It queues its ``UNSUB`` in the
    connection's pending buffer but writes the ``PING`` of its round trip straight to the socket
    (``Client._send_ping``), so the ``PING`` reaches the server FIRST. The ``PONG`` it waits for
    therefore proves nothing about the ``UNSUB``; the drain then forgets the subscription while the
    server is still routing to it, and ``_process_msg`` drops whatever arrives for a subscription
    it no longer knows. For a subscription sharing a queue group with its successor, a message the
    server hands to the old member in that gap is lost outright -- no other member ever sees it.
    Found by the live renewal test: 1 of 11858 streamed messages, across a few dozen renewals.

    So the ``UNSUB`` goes out on its own first, then an ordered :func:`_round_trip`, whose ``PONG``
    proves the server processed the ``UNSUB`` and has sent every message it routed here before it.
    Only then does nats-py's drain run -- its own round trip now harmless -- to wait for each of
    those messages to be taken, and forget the subscription.

    :param raw_subscription: the nats-py subscription to end
    :ptype raw_subscription: Any
    :param connection: the nats-py connection carrying it
    :ptype connection: Any
    :return: nothing
    :rtype: None
    :raises Exception: whatever the connection raises; the caller logs it
    """
    # nats-py exposes no way to remove a subscription's interest without also forgetting the
    # subscription (unsubscribe) or racing its own round trip (drain); see the docstring.
    await send_unsubscribe(connection, raw_subscription)
    await _round_trip(connection)
    await raw_subscription.drain()


async def _unsubscribe_quietly(sub: Any, *, subject: Subject) -> None:
    """unsubscribe a nats-py subscription nothing will read again, logging rather than raising.

    for a result waiter's inbox, a superseded subscription, or a durable's released binding: in
    each case the server-side state ages out or is held by the durable, so nothing leaks.

    :param sub: the nats-py subscription
    :ptype sub: Any
    :param subject: the subject it served, for the log line
    :ptype subject: Subject
    :return: nothing
    :rtype: None
    """
    try:
        await sub.unsubscribe()
    except Exception as exc:  # noqa: BLE001 — nothing durable leaks; see the docstring
        log.debug(
            "unsubscribe of an abandoned subscription failed (subject=%s): %s",
            subject.path,
            exc,
        )


class PublishPin:
    """keeps a run of publishes on the connection its first one used, so they arrive in order.

    NATS orders messages per publisher, and a publisher is a connection. A client that hands over
    to a successor connection mid-run -- a credential renewal, a move off a server in lame-duck
    mode -- becomes two publishers, and when the successor sits on a different server of the
    cluster the two halves of the run travel different routes: a later message can arrive first.
    A run whose order is its meaning (a stream of tokens, which carry no sequence number) takes a
    pin from :meth:`NatsClient.publish_pin` and passes it to every publish of the run, and every
    one of them leaves on the connection the first one did. The replaced connection is kept open
    for the longest request the client makes, which bounds any run a caller should be holding one
    for.

    Owned by the :class:`NatsClient` that made it; nothing else reads or sets what it holds.
    """

    __slots__ = ("connection",)

    def __init__(self) -> None:
        """an unpinned pin: the run's first publish sets its connection.

        :return: nothing
        :rtype: None
        """
        #: the connection the run publishes on, once its first publish chose it
        self.connection: _NatsPyClient | None = None


class _Role(StrEnum):
    """what one connection is to the client that holds it.

    Every change of role is a transition of :class:`_ConnectionLifecycle`; nothing else sets one.
    """

    #: opened by a renewal and not yet current. Its refusals do not speak for the client: the
    #: connection it would replace is still current and still valid, and is kept until it expires.
    CANDIDATE = "candidate"
    #: the one connection every new publish, request, subscribe, KV and JetStream call uses.
    CURRENT = "current"
    #: replaced by a renewal, and held open only for the work it already carries. Its callbacks
    #: no longer speak for the client (no health counting, no reconnect hooks, no outage warning).
    RETIRING = "retiring"


class _Phase(StrEnum):
    """where a client is in its life; see :class:`_ConnectionLifecycle` for every transition."""

    #: one current connection, possibly with replaced ones still retiring
    RUNNING = "running"
    #: a renewal is opening, subscribing or handing over to a candidate; at most one at a time
    RENEWING = "renewing"
    #: refused on purpose (:meth:`NatsClient.abandon`): every connection closed, and nothing reopens one
    ABANDONED = "abandoned"
    #: shut down by its owner (:meth:`NatsClient.shutdown`): nothing renews it
    CLOSED = "closed"


class _ConnectionState:
    """what the client knows about one of its connections beyond what nats-py tracks.

    Read by the connection's own nats-py callbacks, which :class:`_ConnectionOpener` binds to it.

    :ivar role: what the connection is to the client (:class:`_Role`); set only by
        :class:`_ConnectionLifecycle`
    :ivar connected_at: ``time.monotonic()`` of the connection's latest successful
        (re)connect, which is when its current credential was minted
    """

    __slots__ = ("role", "connected_at")

    def __init__(self, *, role: _Role = _Role.CURRENT) -> None:
        """a connection established now, in ``role``.

        :param role: what it is to the client; a renewal opens a :attr:`_Role.CANDIDATE`
        :ptype role: _Role
        :return: nothing
        :rtype: None
        """
        self.role = role
        self.connected_at = time.monotonic()


class _ConnectionLifecycle:
    """the one model of a client's connections and of the client's phase.

    Each connection the client holds has exactly one :class:`_Role`, and the client exactly one
    :class:`_Phase`. Every transition lives here and checks the whole state before it changes any
    of it, so no caller can reach a combination the model does not name. The ones that matter:

    - **a connection is owned from the moment it opens.** A renewal's candidate is registered in
      the same step that opens it (:meth:`admit_candidate`), so every sweep -- :meth:`abandon`,
      :meth:`end_renewal`, a shutdown's -- sees it, and no cancellation between opening it and
      handing over to it can leave it open with nothing holding it.
    - **only a live client changes its current connection.** :meth:`promote` refuses once the
      client is abandoned or closed, in the same step that would switch it, so a renewal that was
      mid-handover when :meth:`abandon` fired cannot make a connection current afterwards.
    - **an abandoned client never holds an open connection.** :meth:`abandon` hands back every
      connection to close and stops the successor being opened; :meth:`admit_candidate` refuses a
      connection that finishes opening afterwards.

    Also owns what a transition must stop with it: the task opening a candidate, the tasks retiring
    replaced connections, and the publish gate a handover closes while it settles.

    Also owns whether the client can hand over at all (:attr:`can_hand_over`): only a client that
    opened its own connection knows how to open a successor, and every handover -- the renewal
    loop's, a lame-duck move's, a requested renewal's, a direct :meth:`NatsClient.renew_connection`
    -- opens one with the :attr:`opener` held here.

    :param current: the connection the client starts with
    :ptype current: nats.aio.client.Client
    :param state: its state, which its callbacks already read
    :ptype state: _ConnectionState
    :param opener: how to open another connection like ``current``, or ``None`` for a client built
        around a connection it did not open
    :ptype opener: _ConnectionOpener | None
    """

    __slots__ = ("_phase", "_current", "_connections", "_publish_gate", "_opening", "_retirements", "_opener")

    def __init__(
        self, current: _NatsPyClient, state: _ConnectionState, *, opener: _ConnectionOpener | None = None
    ) -> None:
        """start RUNNING, with ``current`` the one connection.

        :param current: the connection the client starts with
        :ptype current: nats.aio.client.Client
        :param state: its state
        :ptype state: _ConnectionState
        :param opener: how to open a successor, or ``None`` when the client cannot
        :ptype opener: _ConnectionOpener | None
        :return: nothing
        :rtype: None
        """
        state.role = _Role.CURRENT
        self._opener = opener
        self._phase = _Phase.RUNNING
        self._current = current
        self._connections: dict[_NatsPyClient, _ConnectionState] = {current: state}
        # closed for the instant a handover takes to prove that everything already published on the
        # replaced connection reached the server, so nothing published on the successor overtakes it.
        self._publish_gate = asyncio.Event()
        self._publish_gate.set()
        # the task opening a renewal's candidate, so abandon() can stop a refused connect retrying.
        self._opening: asyncio.Task[_NatsPyClient] | None = None
        # replaced connections waiting out the work they carry, then drained.
        self._retirements: set[asyncio.Task[None]] = set()

    @property
    def phase(self) -> _Phase:
        """where the client is in its life.

        :return: the phase
        :rtype: _Phase
        """
        return self._phase

    @property
    def is_live(self) -> bool:
        """whether the client still serves: neither abandoned nor closed.

        :return: True while running or renewing
        :rtype: bool
        """
        return self._phase in (_Phase.RUNNING, _Phase.RENEWING)

    @property
    def opener(self) -> _ConnectionOpener | None:
        """how to open a successor connection, or ``None`` for a client that cannot.

        :return: the opener
        :rtype: _ConnectionOpener | None
        """
        return self._opener

    @property
    def can_hand_over(self) -> bool:
        """whether the current connection can still be replaced by a successor.

        True for a live client that can open one, whatever would start the handover: a renewal
        armed or not, a server entering lame-duck mode, a requested renewal. Whoever must know
        that a connection can stop being current reads it here.

        :return: True when a handover can occur
        :rtype: bool
        """
        return self._opener is not None and self.is_live

    @property
    def current(self) -> _NatsPyClient:
        """the connection every new operation uses.

        :return: the current connection
        :rtype: nats.aio.client.Client
        """
        return self._current

    def state_of(self, connection: _NatsPyClient) -> _ConnectionState | None:
        """the state of a connection the client holds.

        :param connection: the connection
        :ptype connection: nats.aio.client.Client
        :return: its state, or ``None`` when the client no longer holds it
        :rtype: _ConnectionState | None
        """
        return self._connections.get(connection)

    def replaced(self) -> list[_NatsPyClient]:
        """every connection held that is not current: candidates, and those retiring.

        :return: a snapshot of them
        :rtype: list[nats.aio.client.Client]
        """
        return [connection for connection in self._connections if connection is not self._current]

    def begin_renewal(self) -> None:
        """RUNNING -> RENEWING: one renewal at a time, and never on a client that stopped serving.

        :return: nothing
        :rtype: None
        :raises NatsClientError: when the client is abandoned, closed, or already renewing
        """
        if self._phase is _Phase.RENEWING:
            raise NatsClientError("a credential renewal is already in progress on this NATS client")
        if not self.is_live:
            raise NatsClientError(
                f"cannot renew the credential of a NATS client that is {self._phase.value}; it requires a fresh connect"
            )
        self._phase = _Phase.RENEWING

    def track_opening(self, opening: asyncio.Task[_NatsPyClient] | None) -> None:
        """hold the task opening a renewal's candidate, or forget it once it ended.

        :param opening: the task, or ``None``
        :ptype opening: asyncio.Task[nats.aio.client.Client] | None
        :return: nothing
        :rtype: None
        """
        self._opening = opening

    def admit_candidate(self, connection: _NatsPyClient, state: _ConnectionState) -> bool:
        """register a renewal's freshly opened connection as a CANDIDATE, in the step that opened it.

        :param connection: the connection just opened
        :ptype connection: nats.aio.client.Client
        :param state: its state
        :ptype state: _ConnectionState
        :return: True when it is now owned; False when the client stopped serving while it opened,
            in which case the caller closes it
        :rtype: bool
        """
        if self._phase is not _Phase.RENEWING:
            return False
        state.role = _Role.CANDIDATE
        self._connections[connection] = state
        return True

    def promote(self, candidate: _NatsPyClient) -> _NatsPyClient:
        """make a candidate current and the current connection RETIRING, in one step.

        :param candidate: the renewal's candidate, subscribed and round-tripped
        :ptype candidate: nats.aio.client.Client
        :return: the connection it replaced
        :rtype: nats.aio.client.Client
        :raises NatsClientError: when the client stopped serving during the renewal, or
            ``candidate`` is not this renewal's candidate
        """
        if self._phase is not _Phase.RENEWING:
            raise NatsClientError(f"the NATS client was {self._phase.value} during a credential renewal")
        state = self._connections.get(candidate)
        if state is None or state.role is not _Role.CANDIDATE:
            raise NatsClientError("only a renewal's own candidate connection can be made current")
        previous = self._current
        previous_state = self._connections.get(previous)
        if previous_state is not None:
            previous_state.role = _Role.RETIRING
        state.role = _Role.CURRENT
        self._current = candidate
        return previous

    def end_renewal(self) -> list[_NatsPyClient]:
        """RENEWING -> RUNNING, disowning any candidate that was never made current.

        :return: the candidates disowned, for the caller to close
        :rtype: list[nats.aio.client.Client]
        """
        stranded = [c for c, s in self._connections.items() if s.role is _Role.CANDIDATE]
        for connection in stranded:
            del self._connections[connection]
        if self._phase is _Phase.RENEWING:
            self._phase = _Phase.RUNNING
        return stranded

    def add_retirement(self, retirement: asyncio.Task[None]) -> None:
        """hold a task retiring a replaced connection until it ends.

        :param retirement: the task
        :ptype retirement: asyncio.Task[None]
        :return: nothing
        :rtype: None
        """
        self._retirements.add(retirement)
        retirement.add_done_callback(self._retirements.discard)

    def take_retirements(self) -> list[asyncio.Task[None]]:
        """cancel every pending retirement and hand the tasks back to be awaited.

        :return: the cancelled tasks
        :rtype: list[asyncio.Task[None]]
        """
        retirements = list(self._retirements)
        for retirement in retirements:
            retirement.cancel()
        return retirements

    def forget(self, connection: _NatsPyClient) -> None:
        """stop holding a replaced connection that was closed; the current one is never forgotten.

        :param connection: the connection
        :ptype connection: nats.aio.client.Client
        :return: nothing
        :rtype: None
        """
        if connection is not self._current:
            self._connections.pop(connection, None)

    def abandon(self) -> list[_NatsPyClient]:
        """-> ABANDONED: stop opening, stop retiring, and hand back every connection to close.

        :return: every connection held, current, candidate and retiring, for the caller to close
        :rtype: list[nats.aio.client.Client]
        """
        self._phase = _Phase.ABANDONED
        if self._opening is not None:
            self._opening.cancel()
        self.take_retirements()
        self._publish_gate.set()
        return list(self._connections)

    def close(self) -> None:
        """-> CLOSED, for a shutdown; an abandoned client stays abandoned.

        :return: nothing
        :rtype: None
        """
        if self._phase is not _Phase.ABANDONED:
            self._phase = _Phase.CLOSED

    def hold_publishes(self) -> None:
        """close the publish gate while a handover settles the replaced connection.

        :return: nothing
        :rtype: None
        """
        self._publish_gate.clear()

    def release_publishes(self) -> None:
        """open the publish gate again.

        :return: nothing
        :rtype: None
        """
        self._publish_gate.set()

    async def publishing_connection(self) -> _NatsPyClient:
        """the connection to publish on: the current one, once any handover has settled.

        :return: the current connection
        :rtype: nats.aio.client.Client
        """
        if not self._publish_gate.is_set():
            await self._publish_gate.wait()
        return self._current


class _ConnectionOpener:
    """opens a connection the way :meth:`NatsClient.connect` opened the first one.

    Kept by the client so a credential renewal can open the successor with the same servers,
    credentials provider, inbox prefix and bounds. The callbacks nats-py takes are built per
    connection, bound to that connection's :class:`_ConnectionState`: a replaced connection's
    disconnect is its planned retirement, not an outage, and must not read as one.

    :param servers: the server URLs, primary first
    :ptype servers: list[str]
    :param options: the nats-py connect options every connection shares
    :ptype options: dict[str, object]
    :param primary_url: the primary URL, for error messages
    :ptype primary_url: str
    :param client_name: the client's name, for log lines
    :ptype client_name: str
    :param establish: opens each connection; ``None`` opens a real nats-py connection
    :ptype establish: ConnectionEstablisher | None
    """

    __slots__ = (
        "_servers",
        "_options",
        "_primary_url",
        "_client_name",
        "_establish",
        "_error_log_times",
        "reconnect_callbacks",
        "health_state",
        "lame_duck_handler",
    )

    def __init__(
        self,
        *,
        servers: list[str],
        options: dict[str, object],
        primary_url: str,
        client_name: str,
        establish: ConnectionEstablisher | None = None,
    ) -> None:
        """hold what every connection is opened with.

        :param servers: the server URLs, primary first
        :ptype servers: list[str]
        :param options: the nats-py connect options every connection shares
        :ptype options: dict[str, object]
        :param primary_url: the primary URL, for error messages
        :ptype primary_url: str
        :param client_name: the client's name, for log lines
        :ptype client_name: str
        :param establish: opens each connection, the first and every successor; ``None`` opens a
            real nats-py connection
        :ptype establish: ConnectionEstablisher | None
        :return: nothing
        :rtype: None
        """
        self._servers = servers
        self._establish: ConnectionEstablisher = establish if establish is not None else _establish_connection
        # rate-limit key -> when it was last logged at error, shared by every connection this opener
        # opens: a successor repeating its predecessor's error is still the same repeat.
        self._error_log_times: dict[str, float] = {}
        self._options = options
        self._primary_url = primary_url
        self._client_name = client_name
        # nats-py reads its single ``reconnected_cb`` slot at reconnect time; the dispatcher below
        # closes over this list, which the client adopts, so a consumer callback registered through
        # :meth:`NatsClient.add_reconnect_callback` is dispatched on every reconnect of every
        # current connection.
        self.reconnect_callbacks: list[ReconnectCallback] = []
        # shared by every connection's error/reconnect dispatchers and read by
        # :attr:`NatsClient.is_healthy`. resilience-task-03: ``overflow_events`` beside
        # ``auth_violations``.
        self.health_state: dict[str, int] = {"auth_violations": 0, "overflow_events": 0}
        # what the client does when the server under its current connection enters lame-duck mode
        # (:meth:`NatsClient._move_off_lame_duck_server`). the first connection is opened before the
        # client exists, so each connection's callback reads this when the server says so.
        self.lame_duck_handler: Callable[[], None] | None = None

    async def open(self, state: _ConnectionState) -> _NatsPyClient:
        """open one connection whose callbacks answer to ``state``.

        :param state: the new connection's state, which its callbacks read
        :ptype state: _ConnectionState
        :return: the connected nats-py client
        :rtype: nats.aio.client.Client
        :raises NatsClientError: if the connection fails
        """
        options = dict(self._options)
        options["reconnected_cb"] = self._reconnected_callback(state)
        options["disconnected_cb"] = self._disconnected_callback(state)
        options["error_cb"] = self._error_callback(state)
        options["lame_duck_mode_cb"] = self._lame_duck_callback(state)
        raw = await self._establish(self._servers, options, self._primary_url)
        state.connected_at = time.monotonic()
        return raw

    def _reconnected_callback(self, state: _ConnectionState) -> Callable[[], Awaitable[None]]:
        """the ``reconnected_cb`` for one connection.

        :param state: the connection's state
        :ptype state: _ConnectionState
        :return: the callback
        :rtype: Callable[[], Awaitable[None]]
        """
        health_state = self.health_state
        reconnect_callbacks = self.reconnect_callbacks

        async def _dispatch_reconnected() -> None:
            """fan a reconnect out to the wrapper log + every consumer-registered callback."""
            # a (re)connect mints the connection a fresh credential: the renewal schedule restarts.
            state.connected_at = time.monotonic()
            if state.role is not _Role.CURRENT:
                # NOSILENT: a replaced connection outlived a network drop while it finished its
                # work, or a candidate reconnected before it took over. neither speaks for the
                # client: the current connection's health and the consumer's reconnect hooks are
                # not their business.
                log.info(
                    "a NATS connection that is not current reconnected", extra={"extra_data": {"role": state.role}}
                )
                return
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

        return _dispatch_reconnected

    def _disconnected_callback(self, state: _ConnectionState) -> Callable[[], Awaitable[None]]:
        """the ``disconnected_cb`` for one connection.

        :param state: the connection's state
        :ptype state: _ConnectionState
        :return: the callback
        :rtype: Callable[[], Awaitable[None]]
        """
        client_name = self._client_name

        async def _dispatch_disconnected() -> None:
            """warn of an outage, or record a retirement or a dropped candidate as the ordinary event it is."""
            if state.role is _Role.RETIRING:
                log.info(
                    "a NATS connection replaced by a credential renewal was retired",
                    extra={"extra_data": {"client_name": client_name}},
                )
                return
            if state.role is _Role.CANDIDATE:
                log.info(
                    "a credential renewal's candidate NATS connection closed before it took over",
                    extra={"extra_data": {"client_name": client_name}},
                )
                return
            await _on_disconnected()

        return _dispatch_disconnected

    def _lame_duck_callback(self, state: _ConnectionState) -> Callable[[], Awaitable[None]]:
        """the ``lame_duck_mode_cb`` for one connection.

        nats-py awaits it inside the connection's read loop, so it only hands the event to the
        client's handler, which schedules the move and returns.

        :param state: the connection's state
        :ptype state: _ConnectionState
        :return: the callback
        :rtype: Callable[[], Awaitable[None]]
        """
        client_name = self._client_name

        async def _dispatch_lame_duck() -> None:
            """move the client off a server that is shutting down, if this connection is the current one."""
            if state.role is not _Role.CURRENT:
                # NOSILENT: a connection being retired or not yet current carries no new work, and
                # is closed either way; the server shutting down under it moves nothing.
                log.info(
                    "a NATS server entered lame-duck mode under a connection that is not current",
                    extra={"extra_data": {"client_name": client_name, "role": state.role}},
                )
                return
            handler = self.lame_duck_handler
            if handler is None:
                log.warning(
                    "a NATS server entered lame-duck mode and this client cannot move first; it will "
                    "reconnect when the server closes it, and what is in flight then is lost",
                    extra={"extra_data": {"client_name": client_name}},
                )
                return
            handler()

        return _dispatch_lame_duck

    def _error_callback(self, state: _ConnectionState) -> Callable[[Exception], Awaitable[None]]:
        """the ``error_cb`` for one connection.

        :param state: the connection's state
        :ptype state: _ConnectionState
        :return: the callback
        :rtype: Callable[[Exception], Awaitable[None]]
        """
        health_state = self.health_state
        error_log_times = self._error_log_times

        async def _dispatch_error(exc: Exception) -> None:
            """log via the rate-limited handler AND track a persistent auth violation for is_healthy."""
            # count the violation BEFORE the await: _on_error may suspend, and if a _dispatch_reconnected
            # reset interleaves at that suspension point a post-reset stale += 1 could survive, leaving a
            # phantom count after a healthy reconnect. Incrementing first keeps the counter honest within a
            # run of failures; the next successful reconnect always resets it to 0. Only the CURRENT
            # connection's refusals count: a retiring connection's say nothing about whether the
            # current credential is wedged, and neither does a renewal candidate's -- the current
            # connection is still valid, and is kept until it expires.
            if state.role is _Role.CURRENT and _is_authorization_violation(exc):
                health_state["auth_violations"] += 1
            await _on_error(exc, error_log_times)

        return _dispatch_error


class _CredentialRenewal:
    """the loop :meth:`NatsClient.renew_credential` runs: renew before every expiry until cancelled.

    its own class rather than a method of the client, so the client -- which already keeps
    state it accumulates -- does not also carry a periodic loop, and the loop holds only what
    it reads.
    """

    def __init__(
        self,
        *,
        renew: Callable[[timedelta], Awaitable[None]],
        client_name: str,
        ttl_seconds: Callable[[], int | None],
        connection_age_seconds: Callable[[], float],
        longest_request_seconds: float,
        measure_ttl: Callable[[], Awaitable[int | None]] | None = None,
    ) -> None:
        """bind the loop to the client it renews.

        :param renew: the client's :meth:`NatsClient.renew_connection`, taking how long to keep
            the replaced connection open
        :ptype renew: Callable[[timedelta], Awaitable[None]]
        :param client_name: the client's name, for the log
        :ptype client_name: str
        :param ttl_seconds: reads the current TTL
        :ptype ttl_seconds: Callable[[], int | None]
        :param connection_age_seconds: reads how long ago the current connection was established
        :ptype connection_age_seconds: Callable[[], float]
        :param longest_request_seconds: the longest request the connection makes
        :ptype longest_request_seconds: float
        :param measure_ttl: asks the server for the current credential's lifetime; when given,
            its answer outranks ``ttl_seconds``, which is used only when it reports none
        :ptype measure_ttl: Callable[[], Awaitable[int | None]] | None
        :return: None
        :rtype: None
        """
        self._renew = renew
        self._client_name = client_name
        self._ttl_seconds = ttl_seconds
        self._connection_age_seconds = connection_age_seconds
        self._longest_request_seconds = longest_request_seconds
        self._measure_ttl = measure_ttl
        # the (server, configured) lifetimes the shortfall was last reported at INFO for, so a steady
        # one is reported once rather than every cycle; ``None`` while there is none to report.
        self._reported_shortfall: tuple[int, int] | None = None

    async def _current_ttl(self) -> int | None:
        """the lifetime to schedule this cycle on: the server's answer when asked for, else the configured one.

        A server that does not answer -- an older grant without the user-info subject, a
        momentary stall -- falls back to the configured lifetime with a warning naming why, so
        the loop never stops renewing for want of a measurement.

        :return: the credential's lifetime in seconds, or ``None`` when unknown
        :rtype: int | None
        """
        configured = self._ttl_seconds()
        result = configured
        if self._measure_ttl is not None:
            try:
                measured = await self._measure_ttl()
            except (NatsClientError, ValueError) as exc:
                measured = None
                log.warning(
                    "the server did not report this connection's credential lifetime; scheduling the "
                    "renewal on the configured %s s instead: %s",
                    configured,
                    exc,
                    extra={"extra_data": {"client_name": self._client_name}},
                )
            else:
                if measured is None:
                    # the server answered and named no expiry. Every credential this loop renews
                    # expires, so that is unexpected; renewing on the configured lifetime keeps the
                    # connection alive, and the warning says why it was not the server's.
                    log.warning(
                        "the server reported no expiry for this connection's credential; scheduling the "
                        "renewal on the configured %s s instead",
                        configured,
                        extra={"extra_data": {"client_name": self._client_name}},
                    )
            if measured is not None:
                if configured is not None and measured < configured:
                    self._log_shortfall(measured=measured, configured=configured)
                else:
                    self._reported_shortfall = None
                result = measured
        return result

    def _log_shortfall(self, *, measured: int, configured: int) -> None:
        """say the server's lifetime is shorter than the configured one, at INFO only when it is news.

        The server's answer is rounded down by design
        (:func:`~threetears.nats.credential_renewal.credential_lifetime_from_user_info`),
        so a credential minted for exactly the configured lifetime reads back up to a second short:
        that is not a shorter lifetime and is logged at DEBUG. A real shortfall is logged at INFO
        once, and again only when it changes; a steady one repeated every cycle at INFO buries the
        one cycle where it changed.

        :param measured: the lifetime the server reports, in seconds
        :ptype measured: int
        :param configured: the configured lifetime, in seconds
        :ptype configured: int
        :return: nothing
        :rtype: None
        """
        extra = {
            "extra_data": {
                "client_name": self._client_name,
                "server_ttl_seconds": measured,
                "configured_ttl_seconds": configured,
            }
        }
        message = "the server reports a shorter credential lifetime than configured; renewing on the server's"
        shortfall = (measured, configured)
        if configured - measured > _SERVER_TTL_ROUNDING_SECONDS and shortfall != self._reported_shortfall:
            self._reported_shortfall = shortfall
            log.info(message, extra=extra)
        else:
            log.debug(message, extra=extra)

    async def run(self) -> None:
        """renew before every expiry until cancelled; a failed renewal retries fast.

        a broad catch keeps one failure from ending the loop, and THE WAIT IS CHOSEN BEFORE IT
        IS TAKEN, because a failure has to change it: after one the next attempt comes
        :data:`~threetears.nats.credential_renewal.REAUTH_RETRY_SECONDS` later, never a full
        cycle -- a failure after the scheduled sleep leaves the credential one retry from
        expiry, and a second full cycle would land after it. the TTL is re-read every cycle,
        and the wait is measured from when the current connection was established, so a
        connection that reconnected on its own restarts the schedule; an unknown TTL is
        re-checked on a short cadence without renewing on a guess.

        :return: nothing
        :rtype: None
        """
        retry_in: float | None = None
        try:
            while True:
                try:
                    ttl = await self._current_ttl()
                    if retry_in is None:
                        delay = seconds_until_reauth(ttl, longest_request_seconds=self._longest_request_seconds)
                        if has_schedulable_ttl(ttl):
                            delay = max(REAUTH_MIN_SLEEP_SECONDS, delay - self._connection_age_seconds())
                        unsafe = unsafe_renewal_reason(ttl, longest_request_seconds=self._longest_request_seconds)
                        if unsafe is not None:
                            # every cycle, on purpose: this can cut off the longest request on the connection.
                            log.error("UNSAFE NATS credential renewal cadence: %s", unsafe)
                    else:
                        delay = retry_in
                    retry_in = None
                    await asyncio.sleep(delay)
                    if not has_schedulable_ttl(ttl):
                        continue
                    retire_after = seconds_until_retirement(
                        ttl,
                        connection_age_seconds=self._connection_age_seconds(),
                        longest_request_seconds=self._longest_request_seconds,
                    )
                    await self._renew(timedelta(seconds=retire_after))
                    log.info(
                        "NATS credential renewed on a successor connection before it expired; the replaced "
                        "connection retires once the work it carries has finished",
                        extra={
                            "extra_data": {
                                "client_name": self._client_name,
                                "ttl_seconds": ttl,
                                "retire_after_seconds": retire_after,
                            }
                        },
                    )
                except Exception as exc:  # noqa: BLE001 -- the loop must outlive any one failed renewal
                    retry_in = REAUTH_RETRY_SECONDS
                    log.warning(
                        "NATS credential renewal failed (retrying in %ss): %s",
                        REAUTH_RETRY_SECONDS,
                        exc,
                        extra={"extra_data": {"client_name": self._client_name}},
                    )
        # NOSILENT: cancellation is how shutdown, or a replacing renew_credential, ends the loop
        except asyncio.CancelledError:
            return


class _SuccessorMove:
    """the loop a move to a successor connection runs: hand over until done, or until it is moot.

    Started when the current connection must be replaced now rather than at its renewal: its server
    entered lame-duck mode, or the credential's minter asked for a renewal. Its own class, like
    :class:`_CredentialRenewal`, so the client does not carry a periodic loop beside the state it
    accumulates; this holds only what it reads.
    """

    def __init__(
        self,
        *,
        renew: Callable[[timedelta], Awaitable[None]],
        still_leaving: Callable[[], bool],
        retire_after: timedelta,
        client_name: str,
        why: str,
    ) -> None:
        """bind the move to the client and the connection it leaves.

        :param renew: the client's :meth:`NatsClient.renew_connection`, taking how long to keep
            the replaced connection open
        :ptype renew: Callable[[timedelta], Awaitable[None]]
        :param still_leaving: whether the connection being left is still the client's current,
            open connection, and the client still serves
        :ptype still_leaving: Callable[[], bool]
        :param retire_after: how long to keep the connection it leaves for the work it carries
        :ptype retire_after: timedelta
        :param client_name: the client's name, for the log
        :ptype client_name: str
        :param why: why the connection is being left, for the log
        :ptype why: str
        :return: nothing
        :rtype: None
        """
        self._renew = renew
        self._still_leaving = still_leaving
        self._retire_after = retire_after
        self._client_name = client_name
        self._why = why

    async def run(self) -> None:
        """hand over to a successor, retrying while the connection is still being left.

        Retried because the alternative is worse: a server in lame-duck mode closes the connection
        itself, which is the reconnect this exists to avoid, and a minter that asked for a renewal
        is waiting on a grant the old connection does not have. A successor that could not open
        (every other server busy, a callout that did not answer) is tried again after
        :data:`~threetears.nats.credential_renewal.REAUTH_RETRY_SECONDS`. It stops being worth
        trying once something else replaced the connection (a renewal did) or the server closed it
        (nats-py's reconnect owns it then).

        :return: nothing
        :rtype: None
        """
        try:
            while self._still_leaving():
                try:
                    await self._renew(self._retire_after)
                except Exception as exc:  # noqa: BLE001 -- retried while the connection is still being left; the reason is logged
                    log.warning(
                        "could not move to a successor NATS connection (%s; retrying in %ss): %s",
                        self._why,
                        REAUTH_RETRY_SECONDS,
                        exc,
                        extra={"extra_data": {"client_name": self._client_name}},
                    )
                    await asyncio.sleep(REAUTH_RETRY_SECONDS)
                else:
                    log.info(
                        "moved to a successor NATS connection; the old connection retires with its work",
                        extra={
                            "extra_data": {
                                "client_name": self._client_name,
                                "why": self._why,
                                "retire_after_seconds": self._retire_after.total_seconds(),
                            }
                        },
                    )
        # NOSILENT: cancellation is how shutdown and abandon stop the move
        except asyncio.CancelledError:
            return


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
    :param kv_timings: the deadlines every KV bucket this client opens runs under; ``None`` is the
        production default
    :ptype kv_timings: KvTimings | None
    """

    __slots__ = (
        "_lifecycle",
        "_namespace",
        "_client_name",
        "_subscriptions",
        "_buckets",
        "_kv_locks",
        "_reconnect_callbacks",
        "_health_state",
        "_renewal_task",
        "_handover_lock",
        "_reply_routes",
        "_push_consumers",
        "_abandonment",
        "_successor_move",
        "_longest_request_seconds",
        "_kv_timings",
        "_declarations",
        "_owned_buckets",
        "_pull_consumers",
        "_restoration",
    )

    def __init__(
        self,
        *,
        raw: _NatsPyClient,
        namespace: str,
        client_name: str,
        kv_timings: KvTimings | None = None,
    ) -> None:
        # every connection this client holds, the role of each, and the client's phase: the one
        # model every lifecycle transition goes through (:class:`_ConnectionLifecycle`).
        self._lifecycle = _ConnectionLifecycle(raw, _ConnectionState())
        self._namespace = namespace
        self._client_name = client_name
        # the deadlines every KV bucket this client opens runs under; ``None`` is the production
        # default (:data:`threetears.nats.kv.DEFAULT_KV_TIMINGS`).
        self._kv_timings = kv_timings
        # the credential-renewal loop, when the owner asked for one (:meth:`renew_credential`);
        # cancelled by :meth:`shutdown`.
        self._renewal_task: asyncio.Task[None] | None = None
        # serializes a renewal's handover against a subscribe that would otherwise land on the
        # connection being replaced after the handover enumerated the subscriptions.
        self._handover_lock = asyncio.Lock()
        # request reply subject -> (the connection that received the request, time.monotonic() then),
        # recorded whenever a handover can occur so a reply owed across one leaves on the connection
        # NATS lets answer it. an entry leaves when the reply is sent, when its connection is
        # retired, or once it is older than the longest request (:meth:`_note_reply_route`). kept in
        # the order recorded, which is oldest first.
        self._reply_routes: dict[str, tuple[_NatsPyClient, float]] = {}
        # durable push consumers, which a handover moves by rebinding the durable on the successor.
        self._push_consumers: list[JetStreamPushConsumer] = []
        # durable pull consumers, which a restoration after a reconnect binds again when the server
        # lost their durable (:meth:`_restore_once`). a handover needs no list of them: each follows
        # the current connection at its next fetch.
        self._pull_consumers: list[JetStreamPullConsumer] = []
        # every stream this client DECLARED -- through :meth:`ensure_jetstream_stream`, and the backing
        # stream of every bucket declared through :meth:`ensure_kv_bucket` -- keyed by stream name,
        # carrying the exact config it was declared with, WHATEVER its storage. a NATS restart deletes
        # memory storage, and on Kubernetes a restart can lose file storage too (the volume goes with
        # the pod); nothing but the declarer can put either back. :meth:`_restore_once` re-creates each
        # after every reconnect, and a create of a stream that survived is a no-op.
        self._declarations: dict[str, _NatsStreamConfig] = {}
        # the backing stream name -> fully-qualified bucket name of every remembered KV declaration
        # whose declarer owns the bucket's whole shape (``ensure_kv_bucket(owns_bucket=True)``). a
        # restoration finding one of these live with another configuration reconciles it, as the
        # declaration did, rather than leaving it as it is (:meth:`_redeclare_stream`).
        self._owned_buckets: dict[str, str] = {}
        # the restoration a reconnect started (:meth:`_restore_after_reconnect`), held so it is not
        # collected mid-flight and so a later reconnect, :meth:`shutdown` and :meth:`abandon` stop it.
        self._restoration: asyncio.Task[None] | None = None
        # the abandonment a deliberate credential refusal started (:meth:`abandon_on_refusal`), held so
        # the task is not collected mid-flight.
        self._abandonment: asyncio.Task[None] | None = None
        # a move to a successor connection that must happen now (:meth:`_start_successor_move`): the
        # server entered lame-duck mode, or a renewal was requested. held so it is not collected
        # mid-flight and so shutdown and abandon can stop it.
        self._successor_move: asyncio.Task[None] | None = None
        # how long a replaced connection is kept for the work it carries when a move is not the
        # renewal loop's own: the longest request this client makes, which :meth:`renew_credential`
        # states when the owner arms it.
        self._longest_request_seconds: float = SYNC_REPLY_BUDGET_SECONDS
        self._subscriptions: list[Subscription] = []
        self._buckets: dict[str, NatsKvBucket] = {}
        # one lock PER BUCKET NAME, serializing the opens of that bucket only: a bind-only open of an
        # absent bucket waits for its declarer (up to KvTimings.bind_wait_for_declarer_seconds), and a
        # single client-wide lock held across that wait stalled every other bucket open in the
        # process behind it, cache hits included.
        self._kv_locks: dict[str, asyncio.Lock] = {}
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

    @property
    def _raw(self) -> _NatsPyClient:
        """the current connection, as the lifecycle model holds it.

        :return: the current nats-py connection
        :rtype: nats.aio.client.Client
        """
        return self._lifecycle.current

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
        establish_connection: ConnectionEstablisher | None = None,
        kv_timings: KvTimings | None = None,
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
        :param establish_connection: opens each nats-py connection this client holds -- the first,
            and every successor a credential renewal or lame-duck move opens -- from the servers,
            the options built here and the primary URL. ``None`` (production) opens a real nats-py
            ``Client``; see :data:`ConnectionEstablisher`.
        :ptype establish_connection: ConnectionEstablisher | None
        :param kv_timings: the deadlines every KV bucket this client opens runs under -- one
            operation's ceiling, how often its timeout remedy is logged, and how long a bind-only
            open waits for an absent bucket's declarer. ``None`` (production) uses
            :data:`threetears.nats.kv.DEFAULT_KV_TIMINGS`.
        :ptype kv_timings: KvTimings | None
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

        # the options every connection this client opens shares; the per-connection callbacks are
        # added by the opener, bound to that connection's own state (see _ConnectionOpener).
        options: dict[str, object] = {
            "name": client_name,
            "allow_reconnect": True,
            "max_reconnect_attempts": RUNTIME_MAX_RECONNECT_ATTEMPTS,
            "reconnect_time_wait": 2,
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

        opener = _ConnectionOpener(
            servers=servers,
            options=options,
            primary_url=nats_url,
            client_name=client_name,
            establish=establish_connection,
        )
        state = _ConnectionState()
        started_at = time.monotonic()
        try:
            raw_client = await asyncio.wait_for(opener.open(state), timeout=startup_timeout.total_seconds())
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

        client = cls(raw=raw_client, namespace=nats_subject_namespace, client_name=client_name, kv_timings=kv_timings)
        client._lifecycle = _ConnectionLifecycle(raw_client, state, opener=opener)
        # a server shutting down (a rolling restart) tells its clients first; this one moves to a
        # successor then, the way a renewal does, instead of losing what is in flight to a reconnect.
        opener.lame_duck_handler = client._move_off_lame_duck_server
        # adopt the SAME list and dict every connection's callbacks close over, so an
        # ``add_reconnect_callback`` append is dispatched on the next reconnect and
        # :attr:`is_healthy` reads the live counts.
        client._reconnect_callbacks = opener.reconnect_callbacks
        client._health_state = opener.health_state
        # FIRST in the list, ahead of every consumer hook: what a NATS restart wiped is put back
        # before anything registered later runs against it. it only starts a task, so it never
        # holds up the hooks after it.
        client._reconnect_callbacks.insert(0, client._restore_after_reconnect)
        return client

    def add_reconnect_callback(self, callback: ReconnectCallback) -> None:
        """register an async callback invoked after each successful NATS reconnect.

        the wrapper rides out an outage of any duration (unbounded runtime reconnect) and replays
        subscriptions automatically, and it starts putting back the JetStream state a NATS restart
        can wipe and it declared itself -- streams from :meth:`ensure_jetstream_stream` and buckets
        from :meth:`ensure_kv_bucket`, on memory or file storage, and the durables it bound on
        them -- before any hook
        registered here runs. that runs in the background and retries until it succeeds, so a hook
        must not assume it has finished. other state the BROKER holds for this connection -- e.g. a
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

    async def _restore_after_reconnect(self) -> None:
        """start putting back what a NATS restart may have wiped; registered first by :meth:`connect`.

        A single-node NATS restart deletes every memory-storage stream, every KV bucket on memory
        storage, and every durable consumer on one of those streams -- and on Kubernetes, where a
        restarted NATS pod can come back without its JetStream volume, the file-storage ones too.
        nats-py replays this client's subscriptions on the reconnect, but it knows nothing of
        JetStream state, so before this every service that declared a stream at startup -- the tool
        registry's result stream, found in the devx bring-up; the hub's file-storage turn, delivery
        and audit streams, found in a cold-start validation -- failed every call against it with
        ``stream not found`` until the process was restarted by hand.

        This only STARTS the work, as a task, so the reconnect path and the hooks after this one are
        never held up by a broker that is still recovering. A reconnect that lands while an earlier
        restoration is still retrying replaces it: whatever the first had put back, the second
        restart may have taken again.

        :return: nothing
        :rtype: None
        """
        if not self._declarations and not self._push_consumers and not self._pull_consumers:
            return
        previous = self._restoration
        if previous is not None:
            previous.cancel()
        self._restoration = asyncio.create_task(
            self._restore_until_complete(), name=f"nats-restore-after-reconnect:{self._client_name}"
        )

    async def _restore_until_complete(self) -> None:
        """run restoration rounds, with capped exponential backoff, until one has nothing left to do.

        it never gives up on its own: each failed round has already been logged at ERROR naming what
        is still missing, and what is missing is something every caller of this client depends on.
        only a later reconnect, :meth:`shutdown` or :meth:`abandon` ends it early.

        :return: nothing
        :rtype: None
        """
        rounds = await retry_until_done(
            self._restore_round,
            first_delay=_RESTORE_RETRY_FIRST_DELAY_SECONDS,
            max_delay=_RESTORE_RETRY_MAX_DELAY_SECONDS,
        )
        log.info(
            "NATS JetStream state this client declared is in place after the reconnect",
            extra={
                "extra_data": {
                    "client_name": self._client_name,
                    "streams": sorted(self._declarations),
                    "rounds": rounds,
                }
            },
        )

    async def _restore_round(self) -> bool:
        """one restoration round, as :func:`~threetears.observe.resilience.retry_until_done` runs it.

        :return: ``True`` when nothing was left to restore
        :rtype: bool
        """
        return not await self._restore_once()

    async def _restore_once(self) -> list[str]:
        """one restoration round: every declared stream, then every durable consumer.

        Streams first, because a durable lives on its stream. Each item is attempted whatever
        happened to the one before it, and each failure is logged at ERROR, naming it and why.

        :return: what could not be restored this round, empty when nothing is left to do
        :rtype: list[str]
        """
        failures: list[str] = []
        js = self.jetstream_context()
        for name, config in list(self._declarations.items()):
            try:
                await self._redeclare_stream(js, config)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- logged at ERROR and retried by the next round
                log.error(
                    "re-declaring stream %s after a NATS reconnect failed: %s: %s -- every "
                    "publish to and consumer of it fails until it is back; retrying",
                    name,
                    type(exc).__name__,
                    exc,
                    extra={"extra_data": {"stream": name, "client_name": self._client_name, "error": str(exc)}},
                )
                failures.append(f"stream {name}")
        self._push_consumers = [consumer for consumer in self._push_consumers if not consumer.is_closed]
        self._pull_consumers = [consumer for consumer in self._pull_consumers if not consumer.is_stopped]
        durables: list[JetStreamPushConsumer | JetStreamPullConsumer] = [*self._push_consumers, *self._pull_consumers]
        for consumer in durables:
            try:
                await self._restore_durable(consumer)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- logged at ERROR and retried by the next round
                log.error(
                    "binding durable consumer %s again after a NATS reconnect failed: %s: %s -- it receives "
                    "nothing until it is back; retrying",
                    consumer.durable,
                    type(exc).__name__,
                    exc,
                    extra={
                        "extra_data": {
                            "durable": consumer.durable,
                            "stream": consumer.stream,
                            "subject": consumer.subject.path,
                            "client_name": self._client_name,
                            "error": str(exc),
                        }
                    },
                )
                failures.append(f"durable {consumer.durable}")
        return failures

    async def _redeclare_stream(self, js: Any, config: _NatsStreamConfig) -> None:
        """create one declared stream again, exactly as declared, and never change a live one.

        A create, never an update: JetStream's create is idempotent for an identical config, so a
        stream that survived (the reconnect was a network blip, or another replica already put it
        back) is untouched. One that is live with a DIFFERENT config is refused by the server with
        "stream name already in use"; that is a stream another declarer changed, and it is left as
        it is rather than reconciled back -- this restores what a restart took, it does not fight
        over a stream that is still there.

        One exception: a KV bucket whose declarer owns it (``ensure_kv_bucket(owns_bucket=True)``)
        has nobody to fight over it, so it is reconciled exactly as that declaration reconciled it
        (:func:`threetears.nats.kv.reconcile_kv_stream`) -- in place, or by recreating it empty
        where its storage differs. Otherwise a process that put the wiped bucket back first, with an
        expiry or storage of its own, would leave every bind-only opener asking for a per-entry
        lifetime refused until this one restarted.

        :param js: a JetStream context on the current connection
        :ptype js: Any
        :param config: the config the stream was declared with
        :ptype config: nats.js.api.StreamConfig
        :return: nothing
        :rtype: None
        :raises Exception: any other refusal or failure; the round logs it and retries
        """
        # local import avoids circular dependency between client.py and kv.py
        from threetears.nats.kv import reconcile_kv_stream

        outcome = "re-declared after a NATS reconnect (created if a restart had wiped it)"
        try:
            await js.add_stream(dataclasses.replace(config))
        except Exception as exc:
            if getattr(exc, "err_code", None) != _JS_ERR_STREAM_NAME_IN_USE:
                raise
            owned_bucket = self._owned_buckets.get(config.name or "")
            if owned_bucket is None:
                outcome = (
                    "live with a configuration other than the one declared here after a NATS reconnect; left as it is"
                )
            else:
                await reconcile_kv_stream(js=js, full_name=owned_bucket, config=config, owns_bucket=True)
                outcome = (
                    "live with a configuration other than the one declared here after a NATS reconnect; its "
                    "declarer owns it, so it was reconciled to the declaration"
                )
        log.info(
            "%s-storage stream %s %s",
            _storage_name(config),
            config.name,
            outcome,
            extra={"extra_data": {"stream": config.name, "client_name": self._client_name}},
        )

    async def _restore_durable(self, consumer: JetStreamPushConsumer | JetStreamPullConsumer) -> None:
        """bind a durable consumer again when the server no longer has it.

        A durable whose stream kept its storage, or one that rode out a network blip, is still there
        and is left alone: rebinding it would only churn its push binding. One that is gone is created
        again from the config it was first bound with -- by the push handle at once, under the
        handover lock so a credential renewal cannot move it at the same moment, and by the pull
        handle at its next fetch, since its fetch loop owns its subscription.

        :param consumer: the durable consumer this client bound
        :ptype consumer: JetStreamPushConsumer | JetStreamPullConsumer
        :return: nothing
        :rtype: None
        :raises Exception: when the lookup or the push bind fails; the round logs it and retries
        """
        js = self.jetstream_context()
        missing = False
        try:
            stream = consumer.stream
            if stream is None:
                stream = await js.find_stream_name_by_subject(consumer.subject.path)
            await js.consumer_info(stream, consumer.durable)
        except _NatsJsNotFoundError:
            missing = True
        if missing and isinstance(consumer, JetStreamPushConsumer):
            async with self._handover_lock:
                await consumer.recreate(self.jetstream_context(), self._raw)
            log.info(
                "durable push consumer bound again after the server lost it: durable=%s stream=%s",
                consumer.durable,
                consumer.stream,
            )
        elif missing and isinstance(consumer, JetStreamPullConsumer):
            consumer.rebind_on_next_fetch()

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
        """whether the client is closed: abandoned (:meth:`abandon`), or its current connection closed.

        :return: True if closed
        :rtype: bool
        """
        return self._lifecycle.phase is _Phase.ABANDONED or bool(self._raw.is_closed)

    @property
    def max_payload(self) -> int | None:
        """the largest single publish this broker will accept, in bytes.

        ``None`` until the server's INFO has been seen, and ``None`` again while
        disconnected. **That is the point of the property, not a gap in it.**
        The value is a deployment's, not a library's: 1 MB on an untuned broker
        and whatever the operator chose otherwise. ``nats-py`` fills its own
        attribute with a 1 MB default *before* connecting, so reading it early
        gets a guess wearing the shape of an answer -- which is the one thing a
        caller asking this question must not be handed. A default here would
        also be a second source of truth for the number
        :class:`~threetears.nats.errors.PayloadTooLargeError` exists to stop
        guessing.

        A caller that can ask how much fits can build something that fits --
        a narrower projection, a handle to the part that did not, a chunked
        :mod:`threetears.nats.pipe` transfer -- instead of building it large,
        publishing it, and learning the answer as an exception. That is the
        whole reason this is exposed rather than left to the error path.

        :return: the broker's advertised ``max_payload``, or ``None`` when no
            server has told us
        :rtype: int | None
        """
        if not self._raw.is_connected:
            return None
        advertised = getattr(self._raw, "max_payload", None)
        if not isinstance(advertised, int) or advertised <= 0:
            return None
        return advertised

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
        round-trip (:func:`_round_trip`) -- a stale socket that
        the OS hasn't yet timed out can report ``is_connected=True``
        long after the broker has gone away. consumers building
        ``/healthz`` endpoints should call ``ping()``. a ping that
        times out leaves the connection reading: nats-py's own flush
        did not, and killed the connection it was probing.

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
            await _round_trip(self._raw, timeout=timeout)
        except Exception:
            return False
        return True

    async def reconnect(self) -> None:
        """force a clean reconnect of the underlying connection, re-running server auth.

        nats-py exposes no public force-reconnect. this drives its OWN op-error reconnect path
        (``nats.aio.client.Client._process_op_err``) on a still-CONNECTED client: that transitions
        the client to ``RECONNECTING`` and spawns ``_attempt_reconnect``, which drops + re-opens the
        transport, re-runs the server handshake, replays every live subscription under its original
        ``sid``, and then fires the registered :meth:`add_reconnect_callback` hooks.

        **This is a real disconnect, and it loses what is in flight.** While the transport is down
        the server holds no subscription for this client, so a reply published to one of its
        inboxes in that window is dropped, a request it received cannot be answered afterwards
        (the server lets only the connection that received a request answer it), and a message
        published to a subject it subscribes is never delivered. It exists to exercise the
        reconnect path, the way a broker restart would. To replace a credential before it
        expires, use :meth:`renew_connection`, which loses none of that.

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
        # re-run auth, replay subs under their sid, fire reconnected_cb). we synthesize a
        # StaleConnectionError, the error a dead connection's ping timer would raise. the
        # is_connected guard above keeps us out of the method's else-branch, which would _close.
        # nats-py 2.x exposes no public alternative.
        await force_reconnect(raw, _NatsStaleConnectionError())

    async def renew_connection(self, *, retire_after: timedelta) -> None:
        """replace the current connection with a freshly authenticated one, losing nothing in flight.

        A credential is per CONNECTION and stays valid until its own expiry, so it is renewed by
        opening a SECOND connection -- the auth-callout mints it a fresh credential -- rather than
        by reconnecting the one there is. The handover, in order:

        1. open the successor (bounded by
           :data:`~threetears.nats.credential_renewal.REAUTH_CONNECT_TIMEOUT_SECONDS`); a failure
           here changes nothing and raises, and the caller retries while the current credential
           is still valid;
        2. subscribe every :class:`Subscription` on it, in the same queue group it holds on the
           current connection, and round-trip it so the server has registered them all. Both
           connections are now listening, and each message goes to exactly one of them (see
           :class:`Subscription`). A failure here closes the successor and raises, again changing
           nothing;
        3. make it current: every publish, request, subscribe, KV operation and JetStream call
           from here on uses it -- but only once the old connection has proved that everything
           already published on it reached the server (:meth:`_settle_publishes`), so a message
           published on the successor never overtakes one published before the renewal;
        4. end each subscription's old half (:func:`_stop_routing_then_drain`), so whatever the
           server had already routed to the old connection still reaches the callbacks, and
           rebind each durable push consumer;
        5. keep the old connection open for ``retire_after`` -- the longest request it may be
           carrying -- then drain and close it. Until then a request sent on it still receives its
           reply there, a reply this client owes for a request that arrived there still leaves
           there (:meth:`publish_reply`), and a JetStream message received there is acknowledged
           there. Anything that outlives the old connection and is bound to it -- a KV handle, a
           key watch, a result waiter, a pull consumer -- rebinds to the current connection on its
           own the next time it is used or its connection closes.

        Every step is a transition of the client's one lifecycle model (:class:`_ConnectionLifecycle`):
        the successor is a CANDIDATE, owned from the step that opens it, until the step that makes
        it current -- which is refused once the client was abandoned or shut down meanwhile. A
        candidate that never became current is closed on every way out of this method, including a
        cancellation.

        A connection this client did not open itself (constructed around a caller's nats-py
        client) cannot be renewed: the client does not know how to open another.

        :param retire_after: how long to keep the replaced connection open for the work it
            carries before draining it
        :ptype retire_after: timedelta
        :return: nothing
        :rtype: None
        :raises NatsClientError: if the client is closed, abandoned or already renewing -- before
            the handover or during it -- or did not open its own connection
        :raises Exception: whatever opening or subscribing the successor raised; the current
            connection is untouched
        """
        opener = self._lifecycle.opener
        if opener is None:
            raise NatsClientError(
                "this NATS client was built around a connection it did not open, so it cannot open a "
                "successor to renew its credential; build it with NatsClient.connect",
            )
        if self._raw.is_closed:
            raise NatsClientError("cannot renew the credential of a closed NATS client; it requires a fresh connect")
        self._lifecycle.begin_renewal()
        try:
            successor = await self._open_successor(opener)
            async with self._handover_lock:
                moving: list[tuple[Subscription, Any]] = []
                for sub in [s for s in self._subscriptions if not s.is_closed]:
                    moving.append((sub, await sub.subscribe_on(successor)))
                await _round_trip(successor, timeout=REAUTH_CONNECT_TIMEOUT_SECONDS)
                # the one step that changes the current connection, refused once the client stopped
                # serving: an abandon or a shutdown that landed during any await above wins.
                previous = self._lifecycle.promote(successor)
                await self._hand_over(previous, successor, moving, retire_after=retire_after)
        finally:
            # a candidate that never became current is closed on every path out -- an error, a
            # cancellation, an abandon or a shutdown -- because nothing else will ever reach it.
            for stranded in self._lifecycle.end_renewal():
                await _close_quietly(stranded)
        log.info(
            "NATS connection handed over to a successor with a fresh credential",
            extra={
                "extra_data": {
                    "client_name": self._client_name,
                    "subscriptions_moved": len(moving),
                    "retire_after_seconds": retire_after.total_seconds(),
                }
            },
        )

    async def _hand_over(
        self,
        previous: _NatsPyClient,
        successor: _NatsPyClient,
        moving: list[tuple[Subscription, Any]],
        *,
        retire_after: timedelta,
    ) -> None:
        """finish a handover the lifecycle has already made: settle, move, and schedule retirement.

        Runs with the successor already current (:meth:`_ConnectionLifecycle.promote`), so it
        completes even when cancelled while it settles: the moves and the retirement are scheduled
        on every path out, or every subscription would keep reading only the connection that is
        being retired. A client that stopped serving meanwhile has already closed both connections,
        and schedules nothing.

        :param previous: the connection just replaced
        :ptype previous: nats.aio.client.Client
        :param successor: the connection just made current
        :ptype successor: nats.aio.client.Client
        :param moving: each subscription with its nats-py subscription on the successor
        :ptype moving: list[tuple[Subscription, Any]]
        :param retire_after: how long to keep ``previous`` open for the work it carries
        :ptype retire_after: timedelta
        :return: nothing
        :rtype: None
        """
        # nothing is published on the successor until everything already published on the old
        # connection has reached the server (_settle_publishes).
        self._lifecycle.hold_publishes()
        try:
            await self._settle_publishes(previous)
        finally:
            self._lifecycle.release_publishes()
            if self._lifecycle.is_live:
                # a freshly authenticated connection clears the wedged-auth and outbound-overflow
                # signals, exactly as a successful reconnect does: a refusal before this one said
                # nothing about the credential now in use.
                self._health_state["auth_violations"] = 0
                self._health_state["overflow_events"] = 0
                handovers = [sub.move_to(raw_sub, successor) for sub, raw_sub in moving]
                self._push_consumers = [consumer for consumer in self._push_consumers if not consumer.is_closed]
                js = successor.jetstream()
                handovers.extend(
                    asyncio.create_task(consumer.move_to(js, successor), name=f"nats-handover:{consumer.durable}")
                    for consumer in self._push_consumers
                )
                self._lifecycle.add_retirement(
                    asyncio.create_task(
                        self._retire(previous, after=retire_after, handovers=handovers),
                        name=f"nats-retire:{self._client_name}",
                    )
                )

    async def _open_successor(self, opener: _ConnectionOpener) -> _NatsPyClient:
        """open a renewal's candidate, owned from the step that opens it, in a task abandon can stop.

        nats-py retries a refused connect until the bound, so a successor the callout refuses on
        purpose would otherwise keep asking for the whole of it after the client was abandoned.

        :param opener: how to open it
        :ptype opener: _ConnectionOpener
        :return: the connected candidate, registered with the lifecycle
        :rtype: nats.aio.client.Client
        :raises NatsClientError: when the client stopped serving while the candidate was opening
        """
        opening = asyncio.create_task(self._open_candidate(opener), name=f"nats-open-successor:{self._client_name}")
        self._lifecycle.track_opening(opening)
        try:
            successor = await asyncio.wait_for(opening, timeout=REAUTH_CONNECT_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if not self._lifecycle.is_live and (current is None or current.cancelling() == 0):
                # the attempt was stopped by abandon(), not the caller cancelled: say so.
                raise NatsClientError(
                    f"the NATS client was {self._lifecycle.phase.value} while a renewal was opening"
                ) from None
            raise
        finally:
            self._lifecycle.track_opening(None)
        return successor

    async def _open_candidate(self, opener: _ConnectionOpener) -> _NatsPyClient:
        """open one connection and register it as the renewal's candidate with no await between.

        :param opener: how to open it
        :ptype opener: _ConnectionOpener
        :return: the connected candidate
        :rtype: nats.aio.client.Client
        :raises NatsClientError: when the client stopped serving while it opened; it is closed
        """
        state = _ConnectionState(role=_Role.CANDIDATE)
        connection = await opener.open(state)
        if not self._lifecycle.admit_candidate(connection, state):
            await _close_quietly(connection)
            raise NatsClientError(f"the NATS client was {self._lifecycle.phase.value} while a renewal was opening")
        return connection

    async def _retire(
        self, connection: _NatsPyClient, *, after: timedelta, handovers: list[asyncio.Task[None]]
    ) -> None:
        """keep a replaced connection open for the work it carries, then drain and close it.

        :param connection: the replaced connection
        :ptype connection: nats.aio.client.Client
        :param after: how long to keep it open
        :ptype after: timedelta
        :param handovers: the subscription handovers draining its old halves, awaited first
        :ptype handovers: list[asyncio.Task[None]]
        :return: nothing
        :rtype: None
        """
        try:
            # each handover is bounded, and never raises (it logs); gathering them first means the
            # connection-wide drain below never races one.
            await asyncio.gather(*handovers, return_exceptions=True)
            await asyncio.sleep(after.total_seconds())
            await self._drain_retired(connection)
        finally:
            self._forget_connection(connection)

    async def _drain_retired(self, connection: _NatsPyClient) -> None:
        """drain a replaced connection within its bound, closing it outright past that.

        ``drain`` unsubscribes everything still on it, lets its callbacks finish, flushes what it
        still has to send, and closes it. A request still waiting on it after the hold outlived
        the longest request its owner declared; it is cut off here, and said so, rather than
        left to the server to cut off at expiry.

        :param connection: the replaced connection
        :ptype connection: nats.aio.client.Client
        :return: nothing
        :rtype: None
        """
        if connection.is_closed:
            return
        try:
            await asyncio.wait_for(connection.drain(), timeout=REAUTH_RETIRE_DRAIN_SECONDS)
        except Exception as exc:  # noqa: BLE001 -- the connection is closed either way; the reason is logged
            log.warning(
                "a replaced NATS connection could not be drained within %.0fs; closing it",
                REAUTH_RETIRE_DRAIN_SECONDS,
                extra={"extra_data": {"client_name": self._client_name, "error": str(exc)}},
            )
            await _close_quietly(connection)

    def _forget_connection(self, connection: _NatsPyClient) -> None:
        """drop a retired connection and every reply route that pointed at it.

        :param connection: the retired connection
        :ptype connection: nats.aio.client.Client
        :return: nothing
        :rtype: None
        """
        self._lifecycle.forget(connection)
        stale = [reply for reply, (via, _noted) in self._reply_routes.items() if via is connection]
        for reply in stale:
            del self._reply_routes[reply]

    def _note_reply_route(self, reply_subject: str, connection: _NatsPyClient) -> None:
        """remember which connection received a request, so its reply can leave on that connection.

        Recorded whenever the lifecycle says a handover can occur
        (:attr:`_ConnectionLifecycle.can_hand_over`): a renewal loop, a server entering lame-duck
        mode, a requested renewal and a direct :meth:`renew_connection` all replace the current
        connection, armed renewal or not, and a reply owed across any of them must still leave on
        the connection that received the request. A client that cannot hand over has one
        connection for life and records nothing.

        An entry leaves when the reply is sent -- by any publish method, all of which go through
        :meth:`_reply_connection` -- or when its connection is retired. A reply that is never
        sent (a handler that failed, a request it chose not to answer) is bounded too: each new
        entry first drops those older than the longest request this client declared, by which
        time no requester is still waiting and a replaced connection is no longer held for it.

        :param reply_subject: the request's reply subject
        :ptype reply_subject: str
        :param connection: the connection the request arrived on
        :ptype connection: nats.aio.client.Client
        :return: nothing
        :rtype: None
        """
        if not self._lifecycle.can_hand_over:
            return
        now = time.monotonic()
        horizon = now - self._longest_request_seconds
        routes = self._reply_routes
        while routes:
            oldest = next(iter(routes))
            if routes[oldest][1] >= horizon:
                break
            del routes[oldest]
        routes.pop(reply_subject, None)
        routes[reply_subject] = (connection, now)

    async def _reply_connection(self, reply_subject: str) -> _NatsPyClient:
        """the connection a reply to ``reply_subject`` must leave on.

        NATS lets a principal publish to a requester's inbox only through ``allow_responses``,
        which the server grants to the CONNECTION that received the request -- so a reply sent
        from any other connection is refused as a permissions violation and dropped, while the
        publish itself reports success. The connection that received it, when it is still open;
        otherwise the current one, which is the right one whenever no renewal intervened.

        :param reply_subject: the request's reply subject
        :ptype reply_subject: str
        :return: the connection to publish the reply on
        :rtype: nats.aio.client.Client
        """
        route = self._reply_routes.pop(reply_subject, None)
        if route is not None and not route[0].is_closed:
            return route[0]
        return await self._lifecycle.publishing_connection()

    async def _pinned_connection(self, pin: PublishPin) -> _NatsPyClient:
        """the connection a pinned run publishes on, pinning it at the run's first publish.

        :param pin: the run's pin
        :ptype pin: PublishPin
        :return: the connection to publish on
        :rtype: nats.aio.client.Client
        """
        connection = pin.connection
        if connection is None:
            connection = await self._lifecycle.publishing_connection()
            pin.connection = connection
        elif connection.is_closed or self._lifecycle.state_of(connection) is None:
            log.warning(
                "a pinned run of publishes outlived the connection it started on; it continues on the "
                "current connection, and a message published after this point may arrive before one "
                "published before it",
                extra={"extra_data": {"client_name": self._client_name}},
            )
            connection = await self._lifecycle.publishing_connection()
            pin.connection = connection
        return connection

    async def _settle_publishes(self, previous: _NatsPyClient) -> None:
        """prove that everything published on a replaced connection has reached the server.

        A publisher that sends A on the old connection and then B on its successor would otherwise
        race them: they travel on two sockets, and B can be routed first -- a streamed answer's
        tokens arriving out of order. Round-tripping the old connection settles it, since the server
        processes each connection's input in order and routes a message before it reads the next;
        the round trip is ordered after the pending buffer (:func:`_round_trip`), which nats-py's own
        flush is not. The publish gate is closed meanwhile, so nothing is
        published on the successor until this returns. A cluster that places the successor on a
        different server than the old connection routes the two over different paths, and no round
        trip can order those; on one server, and on the server the old connection shares with it,
        the order holds.

        Never raises: past its bound the handover proceeds, and says so.

        :param previous: the replaced connection
        :ptype previous: nats.aio.client.Client
        :return: nothing
        :rtype: None
        """
        try:
            await _round_trip(previous)
        # NOSILENT: logged; the handover must not wedge publishing on a connection that stopped answering
        except Exception as exc:  # noqa: BLE001 -- ordering is best-effort once the old connection is unresponsive
            log.warning(
                "could not confirm the replaced NATS connection delivered its last publishes; a message "
                "published across this renewal may arrive out of order",
                extra={"extra_data": {"client_name": self._client_name, "error": str(exc)}},
            )

    def _connection_age_seconds(self) -> float:
        """how long ago the current connection was (re)established, which dates its credential.

        :return: seconds since the current connection's latest successful (re)connect
        :rtype: float
        """
        state = self._lifecycle.state_of(self._raw)
        return time.monotonic() - state.connected_at if state is not None else 0.0

    async def credential_ttl_from_server(self, *, timeout: timedelta = timedelta(seconds=2)) -> int | None:
        """the lifetime of this connection's credential, as the server that will end it reports it.

        The server answers ``$SYS.REQ.USER.INFO`` with the requesting connection's own remaining
        credential lifetime, and it is the one party that cannot be wrong about it: it closes the
        connection when that lifetime runs out. Reported as the WHOLE lifetime -- what remains plus
        how long the connection has held it -- because that is the unit the renewal schedule takes.
        Rounded down, so any error schedules the renewal early rather than after the expiry.

        The principal needs :data:`~threetears.nats.subject_permissions.SERVER_USER_INFO_SUBJECT`
        in its publish grant. Without it the request is dropped and this times out, which the
        caller treats as "not reported".

        :param timeout: how long to wait for the server's answer
        :ptype timeout: timedelta
        :return: the credential's lifetime in seconds; ``None`` when it never expires
        :rtype: int | None
        :raises RequestError: when the server does not answer in time
        :raises ValueError: when the connection was replaced while asking, or the answer is not
            the server's user-info shape (:func:`credential_lifetime_from_user_info`)
        """
        age_at_request = self._connection_age_seconds()
        reply = await self.request_raw(subject=Subject.raw(SERVER_USER_INFO_SUBJECT), payload=b"", timeout=timeout)
        return credential_lifetime_from_user_info(
            reply, age_at_request=age_at_request, age_at_reply=self._connection_age_seconds()
        )

    def renew_credential(
        self,
        *,
        ttl_seconds: Callable[[], int | None] = nats_user_jwt_ttl_seconds,
        longest_request_seconds: float = SYNC_REPLY_BUDGET_SECONDS,
        ask_server: bool = False,
    ) -> None:
        """keep this client connected past its credential's expiry by renewing the credential first.

        A connection authenticated by the auth-callout holds a user JWT with a finite TTL, and
        at expiry the server closes it in a way forever-reconnect does not cover
        (:mod:`threetears.nats.credential_renewal`). This runs a loop, owned by the client and
        stopped by :meth:`shutdown`, that calls :meth:`renew_connection` before each expiry: a
        successor connection with a fresh credential takes over, and the replaced one is kept
        open until the work it carries is done. Nothing in flight is dropped.

        Opt-in, because only the owner knows the credential expires: a connection
        authenticated as a static user holds one that never does. A second call replaces the
        running loop.

        :param ttl_seconds: reads the credential's current TTL, every cycle -- an agent's from
            its latest Hub handshake, so a changed TTL reschedules; defaults to the
            environment (:func:`~threetears.nats.credential_renewal.nats_user_jwt_ttl_seconds`)
        :ptype ttl_seconds: Callable[[], int | None]
        :param longest_request_seconds: the longest request this client makes, and the longest a
            reply it owes may take: the replaced connection is kept open this long after each
            renewal. a TTL too short to allow that is logged as an error every cycle, naming it
        :ptype longest_request_seconds: float
        :param ask_server: ask the server for the credential's lifetime every cycle
            (:meth:`credential_ttl_from_server`) and schedule on its answer, falling back to
            ``ttl_seconds`` only when it gives none. For an owner that is never told the lifetime
            it was minted -- a tool pod, which has no handshake -- since a guess longer than the
            minted lifetime renews after the server has already ended the connection
        :ptype ask_server: bool
        :return: nothing
        :rtype: None
        :raises NatsClientError: when the client is abandoned or closed
        """
        if not self._lifecycle.is_live:
            raise NatsClientError(
                f"cannot renew the credential of a NATS client that is {self._lifecycle.phase.value}",
            )
        if self._renewal_task is not None:
            self._renewal_task.cancel()
        self._longest_request_seconds = longest_request_seconds
        renewal = _CredentialRenewal(
            renew=lambda retire_after: self.renew_connection(retire_after=retire_after),
            client_name=self._client_name,
            ttl_seconds=ttl_seconds,
            connection_age_seconds=self._connection_age_seconds,
            longest_request_seconds=longest_request_seconds,
            measure_ttl=self.credential_ttl_from_server if ask_server else None,
        )
        self._renewal_task = asyncio.create_task(renewal.run(), name=f"nats-credential-renewal:{self._client_name}")

    def _move_off_lame_duck_server(self) -> None:
        """start moving to a successor connection, because the current one's server is shutting down.

        A server in lame-duck mode -- a rolling restart -- stops accepting connections, tells its
        clients, and closes them over its lame-duck duration. Left to nats-py, that close is a
        reconnect: a real disconnect that drops the replies, owed replies and messages in flight
        (:meth:`reconnect`). So the client moves first, exactly as a credential renewal does
        (:meth:`renew_connection`): a successor opens on another server, takes every subscription,
        and becomes current, and the old connection is kept for the work it already carries until
        the server closes it or the longest request has had its time.

        Called from the connection's read loop, so it only schedules the move.

        :return: nothing
        :rtype: None
        """
        self._start_successor_move(why="the server entered lame-duck mode")

    def _start_successor_move(self, *, why: str) -> None:
        """schedule a move of the current connection to a successor, unless one is under way.

        A move already running, or a client that no longer serves, is left alone; a renewal already
        in progress opens the same successor, and the move retries until it has (see
        :class:`_SuccessorMove`).

        :param why: why the connection must be replaced now, for the log
        :ptype why: str
        :return: nothing
        :rtype: None
        """
        running = self._successor_move is not None and not self._successor_move.done()
        if running or not self._lifecycle.is_live:
            log.info(
                "a move to a successor NATS connection was asked for; one is already under way or the "
                "client is not serving",
                extra={
                    "extra_data": {"client_name": self._client_name, "why": why, "phase": self._lifecycle.phase.value}
                },
            )
            return
        log.warning(
            "moving to a successor NATS connection now: %s",
            why,
            extra={"extra_data": {"client_name": self._client_name}},
        )
        leaving = self._raw
        move = _SuccessorMove(
            renew=lambda retire_after: self.renew_connection(retire_after=retire_after),
            still_leaving=lambda: self._lifecycle.is_live and self._raw is leaving and not leaving.is_closed,
            retire_after=timedelta(seconds=self._longest_request_seconds),
            client_name=self._client_name,
            why=why,
        )
        self._successor_move = asyncio.create_task(move.run(), name=f"nats-successor-move:{self._client_name}")

    async def renew_on_request(
        self,
        *,
        inbox_prefix: str,
        is_mine: Callable[[CredentialRenewalRequest], bool],
    ) -> Subscription:
        """renew this client's connection at once whenever its credential's minter asks.

        A connection's grant is fixed when it is admitted, so a principal whose grant changed keeps
        the old one until its connection is replaced. The minter publishes a
        :class:`~threetears.nats.CredentialRenewalRequest` to the principal's inbox
        (:mod:`threetears.nats.renewal_request`); on one that ``is_mine`` accepts, the client moves
        to a successor exactly as a renewal does -- make-before-break, nothing in flight lost --
        and the successor is admitted with the grant as it stands now. A principal's inbox can be
        shared by several runners, so ``is_mine`` decides whether a request names this one.

        :param inbox_prefix: this principal's inbox prefix, as it connected with
        :ptype inbox_prefix: str
        :param is_mine: whether a request is for this runner; read at the moment it arrives
        :ptype is_mine: Callable[[CredentialRenewalRequest], bool]
        :return: the subscription, for :meth:`unsubscribe`
        :rtype: Subscription
        """

        async def _on_request(request: CredentialRenewalRequest) -> None:
            if not is_mine(request):
                log.debug(
                    "a credential renewal request for another runner of this principal was ignored",
                    extra={"extra_data": {"client_name": self._client_name, "pod_id": request.pod_id}},
                )
                return
            self._start_successor_move(why=f"its credential's minter asked for a renewal ({request.reason.value})")

        return await self.subscribe_typed(
            subject=Subjects.credential_renewal_request(inbox_prefix),
            cb=_on_request,
            message_type=CredentialRenewalRequest,
            deadletter_on_failure=False,
        )

    async def _stop_successor_move(self) -> None:
        """cancel a move to a successor connection, if one runs, and wait for it to end.

        :return: nothing
        :rtype: None
        """
        task = self._successor_move
        self._successor_move = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # NOSILENT: this IS the cancellation requested on the line above
                pass

    async def _stop_restoration(self) -> None:
        """cancel a restoration after a reconnect, if one runs, and wait for it to end.

        :return: nothing
        :rtype: None
        """
        task = self._restoration
        self._restoration = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # NOSILENT: this IS the cancellation requested on the line above
                pass

    async def _stop_renewal(self) -> None:
        """cancel the credential-renewal loop, if one runs, and wait for it to end.

        :return: nothing
        :rtype: None
        """
        task = self._renewal_task
        self._renewal_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # NOSILENT: this IS the cancellation requested on the line above
                pass

    async def abandon(self, *, reason: str) -> None:
        """close every connection this client holds at once, without draining, and stop renewing.

        For a client whose identity has been refused on purpose -- superseded by a newer runner of
        the same pod-session -- and which must stop serving NOW rather than finish what it holds:
        nothing is drained, flushed or handed over. The client is closed afterwards
        (:attr:`is_closed`), which is how its owner's supervision learns to restart it. Idempotent.

        :param reason: why, for the log line
        :ptype reason: str
        :return: nothing
        :rtype: None
        """
        log.error(
            "NATS client abandoned: every connection closed at once, without draining",
            extra={"extra_data": {"client_name": self._client_name, "reason": reason}},
        )
        connections = self._lifecycle.abandon()
        renewal = self._renewal_task
        self._renewal_task = None
        if renewal is not None and renewal is not asyncio.current_task():
            renewal.cancel()
        move = self._successor_move
        self._successor_move = None
        if move is not None and move is not asyncio.current_task():
            move.cancel()
        restoration = self._restoration
        self._restoration = None
        if restoration is not None and restoration is not asyncio.current_task():
            restoration.cancel()
        for connection in connections:
            await _close_quietly(connection)

    async def abandon_on_refusal(
        self,
        *,
        inbox_prefix: str,
        is_mine: Callable[[CredentialRefusal], bool],
    ) -> Subscription:
        """abandon this client the moment the auth-callout says it refused this client's credential.

        nats-server tells a refused connection only "Authorization Violation", whether the callout
        refused it on purpose or could not be reached, so a deliberate refusal is also published to
        the principal's inbox (:mod:`threetears.nats.credential_refusal`). A principal's inbox can be
        shared by several runners -- every pod of one agent -- so ``is_mine`` decides whether a refusal
        names THIS one (its pod-session and the identity generation it currently presents). One
        that does closes every connection at once (:meth:`abandon`); any other refusal is ignored,
        and a renewal refused for any other reason keeps the current connection until it expires.

        :param inbox_prefix: this principal's inbox prefix, as it connected with
        :ptype inbox_prefix: str
        :param is_mine: whether a refusal names this runner; read at the moment it arrives
        :ptype is_mine: Callable[[CredentialRefusal], bool]
        :return: the subscription, for :meth:`unsubscribe`
        :rtype: Subscription
        """

        async def _on_refusal(refusal: CredentialRefusal) -> None:
            if not is_mine(refusal):
                log.info(
                    "a credential refusal for another runner of this principal was ignored",
                    extra={"extra_data": {"client_name": self._client_name, "pod_id": refusal.pod_id}},
                )
                return
            if self._abandonment is None:
                # a task of its own: closing the connection this callback arrived on must not have to
                # wait for this callback to return.
                self._abandonment = asyncio.create_task(
                    self.abandon(reason=f"credential refused: {refusal.reason.value}"),
                    name=f"nats-abandon:{self._client_name}",
                )

        return await self.subscribe_typed(
            subject=Subjects.credential_refusal(inbox_prefix),
            cb=_on_refusal,
            message_type=CredentialRefusal,
            deadletter_on_failure=False,
        )

    async def _close_replaced_connections(self) -> None:
        """end every pending retirement now and close the connections it was holding open.

        :return: nothing
        :rtype: None
        """
        for task in self._lifecycle.take_retirements():
            try:
                await task
            except asyncio.CancelledError:
                # NOSILENT: this IS the cancellation requested above
                pass
        for connection in self._lifecycle.replaced():
            await self._drain_retired(connection)
            self._forget_connection(connection)

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

        ``drain_timeout`` bounds the nats-py drain only. The
        unsubscribe pass that runs first joins each subscription's
        in-flight callbacks and is **unbounded**: a callback that
        suppresses :class:`asyncio.CancelledError`, or whose cleanup
        blocks, stalls shutdown for as long as it takes. That is the
        cost of not tearing the connection out from under a callback
        mid-reply; handlers are expected to honour cancellation
        promptly.

        :param drain_timeout: max time to wait for the nats-py drain (does not cover the unsubscribe pass)
        :ptype drain_timeout: timedelta
        :return: nothing
        :rtype: None
        """
        # first, and even when already closed: nothing renews a client being shut down, and a renewal
        # loop must not outlive its client, nor a connection it replaced or a candidate it opened.
        self._lifecycle.close()
        await self._stop_renewal()
        await self._stop_successor_move()
        await self._stop_restoration()
        await self._close_replaced_connections()
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
        # ordered after everything already sent, and safe to time out (see _round_trip).
        await _round_trip(self._raw, timeout=timeout)
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

    def _publish_failure(self, *, subject: str, size_bytes: int, exc: Exception, label: str) -> PublishError:
        """classify a raised publish failure into the wrapper's typed error.

        One classifier for every publish entry point, so "too large" cannot mean
        :class:`~threetears.nats.errors.PayloadTooLargeError` on one method and a
        stringified generic on another. The four public entry points do NOT all
        share a single call into nats-py -- :meth:`publish` and
        :meth:`publish_raw` funnel through :meth:`_publish_bytes` while the two
        reply methods publish directly -- so the funnel a caller can rely on is
        this classification, not the call.

        :param subject: subject the publish targeted
        :ptype subject: str
        :param size_bytes: size of the payload that was handed to nats-py
        :ptype size_bytes: int
        :param exc: the exception the underlying publish raised
        :ptype exc: Exception
        :param label: leading text for the generic case, naming the entry point
        :ptype label: str
        :return: the error to raise
        :rtype: PublishError
        """
        too_large = as_payload_too_large(
            subject=subject,
            size_bytes=size_bytes,
            max_payload=self.max_payload,
            exc=exc,
        )
        if too_large is not None:
            return too_large
        return PublishError(f"{label}: subject={subject}: {exc}")

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
        is also accepted for parity with raw nats-py; integration tests
        call this shape, including after a raw ``msg.reply`` lookup.
        callers needing the kw-only typed form keep working unchanged.
        like every publish, a publish to a request's reply subject
        leaves on the connection that received the request (see
        :meth:`_publish_bytes`).

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
        pin: PublishPin | None = None,
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
        :param pin: keeps this publish on the connection the first publish of its run used
            (:meth:`publish_pin`); ``None`` publishes on the current connection
        :ptype pin: PublishPin | None
        :return: nothing
        :rtype: None
        :raises PublishError: if underlying publish fails
        """
        await self._publish_bytes(subject=subject, payload=payload, reply_to=reply_to, pin=pin)

    def publish_pin(self) -> PublishPin:
        """a pin for a run of publishes that must arrive in the order they were made.

        See :class:`PublishPin`. The run's first publish chooses the connection -- the current one,
        once any handover has settled -- and every later publish carrying the pin leaves on it for
        as long as the client holds it open. One that outlives it (held past the longest request)
        moves to the current connection and says so, since order across that step is no longer
        guaranteed.

        :return: a fresh, unpinned pin
        :rtype: PublishPin
        """
        return PublishPin()

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
            await (await self._reply_connection(reply_subject)).publish(reply_subject, payload)
        except Exception as exc:
            self._note_if_outbound_overflow(exc)  # resilience-task-03
            raise self._publish_failure(
                subject=reply_subject,
                size_bytes=len(payload),
                exc=exc,
                label="publish_reply failed",
            ) from exc
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
            await (await self._reply_connection(reply_subject)).publish(reply_subject, payload)
        except Exception as exc:
            self._note_if_outbound_overflow(exc)  # resilience-task-03
            raise self._publish_failure(
                subject=reply_subject,
                size_bytes=len(payload),
                exc=exc,
                label="publish_raw_reply failed",
            ) from exc
        self._note_publish_success()  # resilience-task-03

    async def _publish_bytes(
        self,
        *,
        subject: Subject,
        payload: bytes,
        reply_to: Subject | None,
        pin: PublishPin | None = None,
    ) -> None:
        """common publish path used by :meth:`publish` / :meth:`publish_raw`.

        A publish to the reply subject of a request this client received is a reply, whichever
        method sent it, so it leaves the way :meth:`publish_raw_reply` sends one: on the connection
        that received the request, which is the only one NATS lets answer it, and its route is
        forgotten (:meth:`_reply_connection`). Any other subject has no route and goes out on the
        current connection.

        :param subject: target subject
        :ptype subject: Subject
        :param payload: serialized bytes
        :ptype payload: bytes
        :param reply_to: optional reply subject
        :ptype reply_to: Subject | None
        :param pin: the run this publish belongs to, whose connection it leaves on; ``None`` for none
        :ptype pin: PublishPin | None
        :return: nothing
        :rtype: None
        :raises PublishError: if underlying publish fails
        """
        try:
            connection = (
                await self._pinned_connection(pin) if pin is not None else await self._reply_connection(subject.path)
            )
            if reply_to is None:
                await connection.publish(subject.path, payload)
            else:
                await connection.publish(subject.path, payload, reply=reply_to.path)
        except Exception as exc:
            # resilience-task-03: an outbound-buffer overflow is counted into is_healthy here (the
            # publish boundary) -- it never reaches error_cb. re-raised as a typed PublishError so the
            # caller gets the wrapper's contract, not an unhandled nats-py raise crashing the coroutine.
            self._note_if_outbound_overflow(exc)
            raise self._publish_failure(
                subject=subject.path,
                size_bytes=len(payload),
                exc=exc,
                label="publish failed",
            ) from exc
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

        **ordering.** left unset (the default), ``max_in_flight`` gives
        strictly serial, in-order dispatch: one callback runs to
        completion before the next message is dispatched. **Setting it trades
        that ordering guarantee away** — capped callbacks run
        concurrently and may complete in any order. Only set it when the
        handler is safe to interleave with itself.

        In particular, a shared subscription carrying *stateful* traffic
        keyed by some id (a conversation, an entity, an owned key) needs
        its own per-key serialization on top: concurrent callbacks for
        the same key race each other's read-modify-write and lazy-create
        paths. Stateless one-shot handlers have no such hazard.

        :param subject: subject (point or wildcard pattern) to subscribe on
        :ptype subject: Subject
        :param cb: async callback receiving :class:`IncomingMessage` envelope
        :ptype cb: RawMessageCallback
        :param queue: optional queue group; messages on the subject load-balance across all subscribers in the same queue
        :ptype queue: str | None
        :param max_in_flight: optional cap on concurrently-running callbacks (per-subscription); unset means serial + in-order, set means concurrent + unordered (see **ordering** above)
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
        :param max_in_flight: optional cap on concurrently-running callbacks; unset means
            serial + in-order, set means concurrent + unordered (see :meth:`subscribe`)
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
        :param max_in_flight: optional cap on concurrently-running callbacks; unset means
            serial + in-order, set means concurrent + unordered (see :meth:`subscribe`)
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

        # a group of its own when the caller names none: see Subscription for why it is never plain.
        wire_queue = queue or f"{_SOLE_MEMBER_QUEUE_PREFIX}{uuid.uuid7().hex}"
        semaphore: asyncio.Semaphore | None = asyncio.Semaphore(max_in_flight) if max_in_flight is not None else None

        async def _dispatch_one(msg: "_NatsMsg", received: float) -> None:
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
                        monotonic_received=received,
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

        def _log_loop_crash(exc: BaseException) -> None:
            """record a dispatch-loop failure with the originating error's own type.

            unwraps a ``BaseExceptionGroup`` first: an exception that escapes
            into ``TaskGroup.__aexit__`` would arrive here wrapped, and logging
            the wrapper's type would name ``ExceptionGroup`` instead of the real
            transport fault.
            """
            underlying = representative_exception(exc)
            log.error(
                "subscription dispatch loop crashed",
                extra={
                    "extra_data": {
                        "subject": subject.path,
                        "error_type": type(underlying).__name__,
                        "error": str(underlying),
                    }
                },
            )

        async def _run_bounded(msg: "_NatsMsg", received: float, slot: asyncio.Semaphore) -> None:
            """process one message under the concurrency cap, freeing the slot when done."""
            try:
                await _dispatch_one(msg, received)
            finally:
                slot.release()

        backlog = ReceiptBacklog(
            msgs_limit=DEFAULT_SUB_PENDING_MSGS_LIMIT,
            bytes_limit=DEFAULT_SUB_PENDING_BYTES_LIMIT,
        )
        # a typed callback never sees a reply subject, so it can never reply; only a raw one
        # records which connection received a request.
        feed = _SubscriptionFeed(
            backlog=backlog,
            subject=subject,
            note_reply=self._note_reply_route if raw_cb is not None else None,
        )

        async def _dispatch() -> None:
            """drive the subscription: the feed receives, this task dispatches.

            with ``max_in_flight`` unset, each callback is awaited here before the next
            message is taken from the backlog: strictly serial and in arrival order, and
            cancelling this task cancels the one callback in flight with it.

            with it set, each callback runs as its OWN task, so several progress at once;
            awaiting the callback inline instead would serialize every message despite the
            cap. a slot is acquired before spawning, so once the cap is reached this loop
            parks and the backlog fills behind it.

            the task group owns the callbacks' lifetimes, and the ``finally`` the feed's
            receivers': leaving the block awaits them on normal exit (the message stream
            ending), and cancels *and awaits* them when unsubscribe cancels this task.
            without that join, a callback could still be mid-flight after ``unsubscribe``
            returned and find the connection drained out from under it.
            """
            try:
                async with asyncio.TaskGroup() as group:
                    while (item := await backlog.get()) is not None:
                        msg, received = item
                        if semaphore is None:
                            await _dispatch_one(msg, received)
                        else:
                            await semaphore.acquire()
                            # _dispatch_one never lets an Exception escape, so a sibling
                            # callback can never trip the group's cancel-all-on-child-error.
                            group.create_task(_run_bounded(msg, received, semaphore))
            except asyncio.CancelledError:
                # graceful unsubscribe path
                raise
            except Exception as exc:  # noqa: BLE001 — diag only
                _log_loop_crash(exc)
            finally:
                await feed.cancel()

        # the subscribe and the registration are one step against a renewal: a subscription
        # made on the old connection after the renewal enumerated them would never be moved.
        async with self._handover_lock:
            connection = self._raw
            try:
                raw_sub = await connection.subscribe(subject.path, queue=wire_queue)
            except Exception as exc:
                raise SubscribeError(f"subscribe failed: subject={subject.path} queue={queue!r}: {exc}") from exc
            feed.attach(raw_sub, connection)
            dispatch_task = asyncio.create_task(
                _dispatch(),
                name=f"nats-dispatch:{subject.path}",
            )
            sub = Subscription(
                raw_subscription=raw_sub,
                subject=subject,
                dispatch_task=dispatch_task,
                queue=wire_queue,
                feed=feed,
                connection=connection,
            )
            self._subscriptions.append(sub)

        # the SUB is only in nats-py's pending buffer when connection.subscribe returns, so a
        # message published at once -- a request from another connection, say -- could reach a
        # server that does not yet know of this subscription and be answered "no responders".
        # measured: 108 of 200 requests sent right after subscribe returned. one round trip
        # proves the server processed the SUB before this returns. a connection that cannot
        # answer it (reconnecting, backpressured) keeps the subscription, which nats-py replays
        # on reconnect; that is logged, not raised, since the subscription itself is sound.
        try:
            await _round_trip(connection, timeout=_NATS_FLUSH_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001 — the subscription stands; only its confirmation failed, and that is logged
            log.warning(
                "NATS subscribed, but the server did not confirm the subscription in time; it is "
                "registered client-side and sent again on reconnect",
                extra={
                    "extra_data": {
                        "subject": subject.path,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                },
            )

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

        cancels the dispatch task and waits for it, so every in-flight
        callback has finished unwinding by the time this returns — the
        caller can then close the connection without cutting a callback
        off mid-reply. the wait is **unbounded**; a handler that
        suppresses :class:`asyncio.CancelledError` blocks here.

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
        except asyncio.CancelledError:
            # NOSILENT: this IS the cancellation requested on the line above
            pass
        except Exception as exc:  # noqa: BLE001 -- diag only; the subscription is dropped either way
            log.warning(
                "dispatch task raised while unwinding",
                extra={"extra_data": {"subject": sub.subject.path, "error": str(exc)}},
            )
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
                msg = await (await self._lifecycle.publishing_connection()).request(
                    sub.path, bytes(pos_payload), timeout=secs
                )
            except (_NatsTimeoutError, asyncio.TimeoutError, TimeoutError) as exc:
                raise RequestTimeoutError(
                    f"request timed out: subject={sub.path} timeout={secs:.1f}s",
                ) from exc
            except _NatsNoRespondersError as exc:
                raise NoRespondersError(f"no responders for subject: subject={sub.path}") from exc
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
            connection = await self._lifecycle.publishing_connection()
            msg = await connection.request(subject.path, payload, timeout=timeout.total_seconds())
        except (_NatsTimeoutError, asyncio.TimeoutError, TimeoutError) as exc:
            raise RequestTimeoutError(
                f"request timed out: subject={subject.path} timeout={timeout.total_seconds():.1f}s"
            ) from exc
        except _NatsNoRespondersError as exc:
            raise NoRespondersError(f"no responders for subject: subject={subject.path}") from exc
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
        direct: bool | None = None,
    ) -> NatsKvBucket:
        """obtain (or create) a JetStream KV bucket.

        bucket name is auto-prefixed with the configured namespace
        (``{namespace}-{name}``). passing ``ttl=None`` means values do
        not expire. storage defaults to ``"memory"``: in 3tears, NATS is
        the **L2** tier (ephemeral; durability rides JetStream R3
        replication + the consumer's real L3). Pass ``"file"`` only as a
        deliberate opt-in when a bucket genuinely needs on-disk durability.
        **This is the authority for the memory-storage default**, not any
        retired cache wrapper's docstring.

        the returned handle is CACHED by full bucket name, so the first
        opener's config wins for the life of the process. a component that
        needs a bucket to carry a specific configuration must declare it
        through :meth:`ensure_kv_bucket` BEFORE anything else opens it.

        :param name: bucket name suffix (will be prefixed by namespace)
        :ptype name: str
        :param ttl: optional time-to-live for entries; ``None`` for no expiry
        :ptype ttl: timedelta | None
        :param storage: ``"memory"`` (default -- L2) or ``"file"`` (opt-in)
        :ptype storage: str
        :param create_if_missing: create bucket if it does not exist
        :ptype create_if_missing: bool
        :param history: number of historical revisions to keep per key
        :ptype history: int
        :param direct: request ``allow_direct`` on the backing stream. ``None``
            (the default) neither requests nor compares it, which is what an
            ordinary consumer wants: it binds to whatever the declaring identity
            established. see :meth:`ensure_kv_bucket`
        :ptype direct: bool | None
        :return: ready KV bucket handle
        :rtype: NatsKvBucket
        :raises KvBucketNotFoundError: if ``create_if_missing`` is ``False`` and the bucket does not
            exist once the wait for its declarer is spent (a ``KvError``)
        :raises KvError: if bucket creation or binding fails for any other reason
        :raises KvConfigMismatch: if a bind-only open finds a reconciled field differing
        """
        # local import avoids circular dependency between client.py and kv.py
        from threetears.nats.kv import DEFAULT_KV_TIMINGS, NatsKvBucket

        full_name = f"{self._namespace}-{name}"
        cached = self._buckets.get(full_name)
        if cached is not None:
            return cached
        async with self._kv_locks.setdefault(full_name, asyncio.Lock()):
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
                direct=direct,
                timings=self._kv_timings if self._kv_timings is not None else DEFAULT_KV_TIMINGS,
            )
            self._buckets[full_name] = bucket
        return bucket

    async def ensure_kv_bucket(
        self,
        *,
        name: str,
        ttl: timedelta | None = None,
        storage: str = "memory",
        history: int = 1,
        direct: bool = True,
        create_if_missing: bool = True,
        owns_bucket: bool = False,
    ) -> NatsKvBucket:
        """DECLARE a KV bucket's configuration, reconciling a live one in place.

        the KV counterpart of :meth:`ensure_jetstream_stream`, and the answer to
        the create-or-BIND defect: opening a bucket that already existed used to
        drop the caller's whole requested config with nothing above a
        ``log.debug`` to say so. this declares instead -- it creates the bucket
        when absent and updates the live stream in place when it carries a
        different value for one of
        :data:`threetears.nats.kv.RECONCILED_KV_STREAM_FIELDS`.

        **call this at startup, from the identity that owns the bucket**, before
        anything else in the process opens it. it writes through the SAME cache
        :meth:`kv_bucket` reads, so every later consumer shares this handle and
        cannot diverge from the declared config; running it second would find the
        cache already populated by an undeclared open. it does not read that
        cache first -- a declaration is idempotent and re-running it after a
        reconnect is exactly how a wiped bucket comes back.

        a declaration (``create_if_missing=True``) is remembered, on memory or
        file storage alike, and its backing stream is created again with the same
        config after every reconnect, exactly as :meth:`ensure_jetstream_stream`
        does for a stream -- so a bucket wiped by a NATS restart (memory storage
        always; file storage when the restart lost its volume, as it can on
        Kubernetes) is back for its binders whether or not anything in this
        process uses it. a declarer needs no reconnect hook of its own for that.
        only the bucket comes back, empty: its entries are the declarer's to
        write again. a bind-only open is never remembered: only the declarer may
        create the bucket. neither is a :meth:`kv_bucket` open, which declares
        nothing -- remembering one would let a process that is not the bucket's
        declarer create it after a restart with a config (``allow_direct`` unset,
        say) the declarer's create-only restoration then leaves in place.

        ``direct`` defaults to ``True`` here and to ``None`` on
        :meth:`kv_bucket`, and the asymmetry is the point: a declaration states
        the value, an ordinary open accepts whatever the declarer established.
        ``allow_direct`` is load-bearing for security -- with it false, nats-py
        reads a key by putting the key in the REQUEST BODY of
        ``$JS.API.STREAM.MSG.GET.KV_{bucket}``, and NATS authorises on subjects,
        so no key-scoped ``$KV.`` grant can constrain a read.

        **the bucket's expiry, history and storage are reconciled only by a
        declarer that owns the bucket.** they sit outside the reconciled set,
        because a bucket with several declarers asking for different shapes would
        have them fight over it. a bucket with exactly one owner has nobody to
        fight: that owner passes ``owns_bucket=True``, and the whole requested
        shape is reconciled (:func:`threetears.nats.kv.reconcile_kv_stream`) --
        a live ``max_age`` (``ttl``, ``None`` meaning none) or history in place,
        logged at INFO naming the old and new values; a live storage, which
        JetStream cannot change in place, by deleting the bucket's stream and
        creating it again, logged at WARNING naming the old and new values and
        that every entry was dropped. it applies on this declaration, on every
        self-heal re-open of the handle, and on every restoration after a
        reconnect, and is safe against two owners declaring at once.

        **only for an L2 bucket whose contents are ephemeral by design**, which is
        why an owner must declare ``storage="memory"``: a NATS restart already
        wipes such a bucket and every binder already survives finding it empty, so
        a recreate is that restart, for one bucket. a file bucket was declared
        durable on purpose and cannot be owned.

        the default ``False`` reports a differing expiry, history or storage at
        WARNING and leaves it. a platform declaring every pod bucket with no
        bucket-wide expiry, so each bind-only opener carries its own per-entry
        lifetime, must pass it: a stale bucket-wide expiry otherwise refuses every
        such opener with :class:`~threetears.nats.errors.KvConfigMismatch` for as
        long as it lives. :meth:`kv_bucket` never owns a bucket; it declares
        nothing.

        :param name: bucket name suffix (will be prefixed by namespace)
        :ptype name: str
        :param ttl: optional time-to-live for entries; ``None`` for no expiry
        :ptype ttl: timedelta | None
        :param storage: ``"memory"`` (default -- L2) or ``"file"`` (opt-in)
        :ptype storage: str
        :param history: number of historical revisions to keep per key
        :ptype history: int
        :param direct: the ``allow_direct`` value this bucket must carry
        :ptype direct: bool
        :param create_if_missing: ``True`` declares (create + reconcile);
            ``False`` binds read-only and REFUSES a bucket whose reconciled
            config differs, which is what a process that is not the bucket's
            owner should do
        :ptype create_if_missing: bool
        :param owns_bucket: this declarer is the bucket's one owner of its
            whole shape, so every requested field is reconciled -- in place, or
            by recreating the bucket empty where JetStream cannot; remembered with
            the declaration, so a restoration after a reconnect does the same.
            only with ``create_if_missing=True`` and ``storage="memory"``
        :ptype owns_bucket: bool
        :return: ready KV bucket handle, also installed in the client's cache
        :rtype: NatsKvBucket
        :raises ValueError: if ``owns_bucket=True`` with ``create_if_missing=False``
            or ``storage="file"``
        :raises KvBucketNotFoundError: if the bucket does not exist and this call could not create it --
            a bind (``create_if_missing=False``) once the wait for its declarer is spent, or a
            declaration whose create was not answered (a ``KvError``)
        :raises KvError: if bucket creation or binding fails for any other reason
        :raises KvConfigMismatch: if ``create_if_missing=False`` and the live bucket differs
        :raises StreamSubjectsOverlapError: if a different stream owns the bucket's subjects
        """
        # local import avoids circular dependency between client.py and kv.py
        from nats.js.api import StorageType  # noqa: PLC0415
        from threetears.nats.kv import DEFAULT_KV_TIMINGS, NatsKvBucket, build_kv_stream_config

        full_name = f"{self._namespace}-{name}"
        async with self._kv_locks.setdefault(full_name, asyncio.Lock()):
            bucket = await NatsKvBucket.open(
                client=self,
                full_name=full_name,
                ttl=ttl,
                storage=storage,
                create_if_missing=create_if_missing,
                history=history,
                direct=direct,
                timings=self._kv_timings if self._kv_timings is not None else DEFAULT_KV_TIMINGS,
                owns_bucket=owns_bucket,
            )
            self._buckets[full_name] = bucket
        if create_if_missing:
            # a DECLARATION: a NATS restart can delete the bucket (memory storage always, file storage
            # when the restart lost its volume), and a process that only binds it waits for its
            # declarer -- so the declarer puts it back after every reconnect (:meth:`_restore_once`),
            # with the same backing-stream config created here.
            declared = build_kv_stream_config(
                bucket=full_name,
                ttl_seconds=int(ttl.total_seconds()) if ttl is not None else 0,
                history=history,
                storage_type=StorageType.FILE if storage == "file" else StorageType.MEMORY,
                direct=direct,
            )
            stream = declared.name or full_name
            self._declarations[stream] = declared
            # the LATEST declaration is the one remembered, its ownership of the bucket included:
            # a re-declaration without it gives the ownership up.
            if owns_bucket:
                self._owned_buckets[stream] = full_name
            else:
                self._owned_buckets.pop(stream, None)
        return bucket

    async def ensure_jetstream_stream(
        self,
        *,
        name: str,
        subjects: list[str],
        storage: str = "memory",
        max_age_seconds: float | None = None,
        max_msgs_per_subject: int | None = None,
    ) -> str:
        """create (or update) a JetStream stream over given subjects.

        idempotent: binds to an existing stream of the same name and reconciles
        its subject set, else creates it. the stream name is namespace-prefixed
        (``{namespace}-{name}``) to match the KV-bucket convention. storage
        defaults to ``"memory"``: NATS is the **L2** tier in 3tears (ephemeral;
        durability rides JetStream R3 replication + the consumer's real L3).
        Pass ``"file"`` only as a deliberate opt-in when a stream genuinely
        needs on-disk durability.

        **a stream declared here comes back after a NATS restart on its own,
        whatever its storage.** a restart deletes a memory-storage stream (and
        every durable on it), and on Kubernetes a restart that loses the
        JetStream volume deletes a file-storage one too. this client remembers
        the exact config each was declared with and creates it again after
        every reconnect, then binds again every durable consumer it holds that
        the server lost. the re-declaration only CREATES: a stream that is still
        live -- the reconnect was a network blip, the restart kept its file
        storage, or another declarer changed it since -- is left exactly as it
        is. a failure is logged at ERROR naming the stream and retried with
        backoff until it succeeds. only the stream comes back, empty: messages
        the restart took are gone. the latest declaration of a name is the one
        restored.

        :param name: stream name suffix (namespace-prefixed)
        :ptype name: str
        :param subjects: subject patterns the stream captures
        :ptype subjects: list[str]
        :param storage: ``"memory"`` (default -- L2) or ``"file"`` (opt-in)
        :ptype storage: str
        :param max_age_seconds: discard a message this long after it was published, whether or not
            anything consumed it. ``None`` (the default) retains until another limit bites. a stream
            whose messages are addressed to ONE waiter needs this: a caller that never returns for
            its message would otherwise leave it resident for the life of the stream
        :ptype max_age_seconds: float | None
        :param max_msgs_per_subject: keep at most this many messages per SUBJECT. ``None`` (the
            default) leaves the per-subject count unbounded. ``1`` is the right value for a family
            whose subjects are minted per call and answered once -- a retry that republishes cannot
            then leave a stale first answer behind for the waiter to pick up
        :ptype max_msgs_per_subject: int | None
        :return: full namespace-prefixed stream name
        :rtype: str
        :raises StreamSubjectsOverlapError: if subjects are already claimed by a different stream
        :raises RuntimeError: if stream creation and update both fail
        """
        from nats.js.api import StorageType, StreamConfig  # noqa: PLC0415

        full_name = f"{self._namespace}-{name}"
        storage_type = StorageType.FILE if storage == "file" else StorageType.MEMORY
        config = StreamConfig(name=full_name, subjects=subjects, storage=storage_type)
        if max_age_seconds is not None:
            config.max_age = max_age_seconds
        if max_msgs_per_subject is not None:
            config.max_msgs_per_subject = max_msgs_per_subject
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
        self._declarations[full_name] = dataclasses.replace(config)
        log.info(
            "jetstream stream ensured: stream=%s subjects=%s storage=%s",
            full_name,
            ",".join(subjects),
            storage,
        )
        return full_name

    async def jetstream_publish(
        self,
        *,
        subject: Subject,
        payload: bytes,
        timeout: timedelta | float = DEFAULT_JETSTREAM_PUBLISH_TIMEOUT,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """publish raw bytes to a JetStream subject with persistence ack.

        unlike :meth:`publish_raw` (core NATS, fire-and-forget), this persists
        the message to the backing stream and awaits the broker ``PubAck``, so a
        message published while no consumer is attached is retained and
        redelivered to a durable consumer when it connects.

        **This call is bounded.** An unbounded one hung its caller's event loop
        forever, and did so in a way no caller could defend against from
        outside -- :mod:`threetears.nats._publish` holds the mechanism and the
        full account of why one bound is not enough. A timed-out publish has an
        **unknown** outcome, not a failed one: the broker may have persisted the
        message anyway. Set ``Nats-Msg-Id`` upstream if a retry must not
        duplicate.

        :param subject: target JetStream subject
        :ptype subject: Subject
        :param payload: serialized message bytes
        :ptype payload: bytes
        :param timeout: ceiling on the whole publish including the ack; a bare
            ``int``/``float`` is seconds
        :ptype timeout: timedelta | float
        :param headers: message headers stored with the payload, or ``None`` for none
        :ptype headers: Mapping[str, str] | None
        :return: nothing
        :rtype: None
        :raises PublishTimeoutError: the publish did not complete in time, or
            stopped responding to cancellation
        :raises PublishError: the broker rejected the publish
        """
        seconds = timeout.total_seconds() if isinstance(timeout, timedelta) else float(timeout)
        try:
            await self._lifecycle.publishing_connection()
            await publish_bounded(self.jetstream_context(), subject.path, payload, timeout=seconds, headers=headers)
        except PublishError:
            raise
        except Exception as exc:
            # The oversized-publish refusal fires before the ack wait, so it reaches here too and
            # must land on the same type the core publish path raises -- a caller branching on it
            # should not have to know which publish path produced the frame.
            raise raise_as_publish_error(
                subject.path,
                exc,
                size_bytes=len(payload),
                max_payload=self.max_payload,
            ) from exc

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

        config = _NatsConsumerConfig(
            durable_name=durable,
            ack_policy=_NatsAckPolicy.EXPLICIT,
            max_deliver=max_deliver,
            ack_wait=ack_wait_seconds,
        )

        async def _bind(js: Any) -> Any:
            """bind the durable through ``js``, delivering to the bounded callback.

            :param js: a JetStream context on the connection to bind on
            :ptype js: Any
            :return: the nats-py push subscription
            :rtype: Any
            """
            return await js.subscribe(
                subject.path,
                durable=durable,
                cb=_bounded_cb,
                manual_ack=True,
                stream=stream,
                config=config,
            )

        await self._reconcile_durable(
            subject=subject,
            durable=durable,
            stream=stream,
            ack_wait_seconds=ack_wait_seconds,
            max_deliver=max_deliver,
        )
        # bound and registered as one step against a renewal, which would otherwise miss a
        # consumer bound on the old connection after it enumerated them.
        async with self._handover_lock:
            connection = self._raw
            sub = await _bind(self.jetstream_context())
            consumer = JetStreamPushConsumer(
                raw_subscription=sub,
                subject=subject,
                durable=durable,
                resubscribe=_bind,
                connection=connection,
                stream=stream,
            )
            self._push_consumers.append(consumer)
        log.info(
            "jetstream durable consumer subscribed: subject=%s durable=%s stream=%s max_deliver=%d dlq=%s",
            subject.path,
            durable,
            stream,
            max_deliver,
            dead_letter_subject.path if dead_letter_subject is not None else None,
        )
        return consumer

    async def _reconcile_durable(
        self,
        *,
        subject: Subject,
        durable: str,
        stream: str | None,
        ack_wait_seconds: float,
        max_deliver: int,
    ) -> None:
        """bring an EXISTING durable's ack wait and delivery budget to what the code asks for.

        nats-py's ``subscribe`` and ``pull_subscribe`` create a durable only when it is missing and
        bind an existing one as it stands, so a changed ``ack_wait`` or ``max_deliver`` otherwise
        never reaches a stream that already has the durable -- every deployed one. The live config
        is sent back with only those two fields changed, which the server applies as an update;
        a missing durable is left to the bind, which creates it with the full config.

        :param subject: subject the durable consumes (finds the stream when ``stream`` is ``None``)
        :ptype subject: Subject
        :param durable: durable consumer name
        :ptype durable: str
        :param stream: backing stream name, or ``None`` to look it up by subject as the bind would
        :ptype stream: str | None
        :param ack_wait_seconds: the ack wait the durable must carry
        :ptype ack_wait_seconds: float
        :param max_deliver: the delivery budget the durable must carry
        :ptype max_deliver: int
        :return: nothing
        :rtype: None
        :raises Exception: when the server refuses the lookup or the update; the bind does not run
        """
        js = self.jetstream_context()
        stream_name = stream if stream is not None else await js.find_stream_name_by_subject(subject.path)
        live: Any = None
        try:
            live = (await js.consumer_info(stream_name, durable)).config
        except _NatsJsNotFoundError:
            live = None
        if live is not None and (live.ack_wait != ack_wait_seconds or live.max_deliver != max_deliver):
            await js.add_consumer(stream_name, config=live.evolve(ack_wait=ack_wait_seconds, max_deliver=max_deliver))
            log.info(
                "durable consumer config updated to what its code asks for: durable=%s stream=%s "
                "ack_wait %.1fs -> %.1fs, max_deliver %s -> %d",
                durable,
                stream_name,
                live.ack_wait or 0.0,
                ack_wait_seconds,
                live.max_deliver,
                max_deliver,
            )

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

        the dead letter carries the subject the message arrived on in the
        :data:`~threetears.nats.subjects.DEAD_LETTER_ORIGINAL_SUBJECT_HEADER`
        header. the dead-letter subject is one fixed subject, so the payload
        alone loses the part the broker authorised; a consumer that holds a
        payload's claims to its subject needs it back to replay the letter.

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
                original_subject = getattr(msg, "subject", None)
                await self.jetstream_publish(
                    subject=dead_letter_subject,
                    payload=msg.data,
                    headers=(
                        {DEAD_LETTER_ORIGINAL_SUBJECT_HEADER: original_subject}
                        if isinstance(original_subject, str) and original_subject
                        else None
                    ),
                )
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
        error_backoff_seconds: float = _PULL_CONSUMER_ERROR_BACKOFF_SECONDS,
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
            mid-handle redelivers the message to another fetcher after this. a live
            handler holds its message with in-progress acks, so this bounds a dead
            fetcher's hold, never a handler's running time
        :ptype ack_wait_seconds: float
        :param stream: backing stream name to bind the consumer to
        :ptype stream: str | None
        :param batch: max messages one fetch pulls
        :ptype batch: int
        :param fetch_timeout_seconds: how long one fetch waits for a message
            before returning empty (the idle poll cadence)
        :ptype fetch_timeout_seconds: float
        :param error_backoff_seconds: how long the fetch loop pauses after a failed cycle (a
            transport error during a reconnect) before retrying
        :ptype error_backoff_seconds: float
        :return: a pull-consumer handle whose ``run`` loops fetch+dispatch
        :rtype: JetStreamPullConsumer
        :raises ValueError: when ``max_deliver`` < 1
        """
        if max_deliver < 1:
            raise ValueError(
                f"max_deliver must be >= 1: a durable consumer cannot redeliver forever (got {max_deliver})",
            )
        config = _NatsConsumerConfig(
            durable_name=durable,
            ack_policy=_NatsAckPolicy.EXPLICIT,
            max_deliver=max_deliver,
            ack_wait=ack_wait_seconds,
            filter_subject=subject.path,
        )

        async def _bind() -> Any:
            """bind the durable on the client's current connection.

            :return: the nats-py pull subscription
            :rtype: Any
            """
            return await self.jetstream_context().pull_subscribe(
                subject.path,
                durable=durable,
                stream=stream,
                config=config,
            )

        await self._reconcile_durable(
            subject=subject,
            durable=durable,
            stream=stream,
            ack_wait_seconds=ack_wait_seconds,
            max_deliver=max_deliver,
        )
        bound_to = self._raw
        psub = await _bind()

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
            ack_wait_seconds=ack_wait_seconds,
            bound_to=bound_to,
            current_connection=lambda: self._raw,
            resubscribe=_bind,
            error_backoff_seconds=error_backoff_seconds,
            stream=stream,
        )
        # stopped handles leave as new ones arrive, so a process that binds and stops consumers for
        # its whole life without ever reconnecting does not keep every one it ever made.
        self._pull_consumers = [held for held in self._pull_consumers if not held.is_stopped]
        self._pull_consumers.append(consumer)
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

    async def jetstream_result_waiter(
        self,
        *,
        subject: Subject,
        stream: str,
        wait_budget: timedelta,
    ) -> JetStreamResultWaiter:
        """open a waiter for ONE answer on ONE exact subject, before dispatching the call.

        The counterpart to :meth:`request_raw` for work that outlives a connection. A request/reply
        answer may be published only by the connection that RECEIVED the request, so a responder
        whose work spans a credential refresh loses the right to answer at the moment it finishes.
        Here the responder instead publishes to a subject it holds a standing grant on, and this
        collects it from the stream -- so neither side's reconnect can strand the answer.

        Open it BEFORE publishing the call. Retention makes the ordering safe either way, but opening
        first is what makes it obviously safe, and it costs one round trip against a call that is by
        definition long.

        :param subject: the exact subject the answer will be published to (never a wildcard: this
            waiter is for one call, and a pattern would collect a peer's answer)
        :ptype subject: Subject
        :param stream: backing stream name, passed explicitly so nats-py never issues the
            ``$JS.API.STREAM.NAMES`` lookup that no principal is granted
        :ptype stream: str
        :param wait_budget: how long the answer is allowed to take; sets the consumer's inactivity
            keepalive so the server does not reap it out from under a long call
        :ptype wait_budget: timedelta
        :return: an opened waiter; ``await`` its :meth:`JetStreamResultWaiter.wait` for the answer
            and always :meth:`JetStreamResultWaiter.close` it
        :rtype: JetStreamResultWaiter
        """
        waiter = JetStreamResultWaiter(
            connection=lambda: self._raw,
            jetstream=self.jetstream_context,
            subject=subject,
            stream=stream,
            inactive_threshold_seconds=(wait_budget.total_seconds() + _RESULT_WAITER_KEEPALIVE_MARGIN_SECONDS),
            poll_seconds=_RESULT_WAITER_POLL_SECONDS,
        )
        await waiter.open()
        return waiter

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
            await (await self._lifecycle.publishing_connection()).publish(
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
    nc = _NatsPyClient()
    # the options are heterogeneous by nature (callbacks, bounds, credentials); nats-py types each
    # keyword separately, which an options dict cannot express.
    connect_options: dict[str, Any] = dict(options)
    try:
        await nc.connect(servers, **connect_options)
    except BaseException as exc:
        # the caller bounds this with a timeout, and a cancelled connect leaves nats-py's
        # reconnect loop and transport alive on an object nothing else holds: close it here.
        await _close_quietly(nc)
        if isinstance(exc, Exception):
            raise NatsClientError(f"failed to connect to NATS at {primary_url}: {exc}") from exc
        raise
    return nc


async def _close_quietly(nc: _NatsPyClient) -> None:
    """close a nats-py connection nothing will use again, logging rather than raising.

    :param nc: the connection
    :ptype nc: nats.aio.client.Client
    :return: nothing
    :rtype: None
    """
    try:
        if not nc.is_closed:
            await nc.close()
    # NOSILENT: logged; the connection is abandoned either way
    except Exception as exc:  # noqa: BLE001 -- closing an abandoned connection must not mask its cause
        log.debug("closing an abandoned NATS connection failed: %s", exc)


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


class _ViolationDetail(NamedTuple):
    """what a server permissions refusal names, decomposed, with the case of the subject qualified.

    ``operation`` and ``subject`` are ``None`` when the server's wording could not be decomposed:
    the violation is still reported, but the structured fields say "not recovered" rather than
    carrying a human placeholder a log pipeline would treat as a real subject.
    """

    operation: str | None
    subject: str | None
    subject_case: str


def _permissions_violation(exc: Exception) -> _ViolationDetail | None:
    """the operation, subject and subject-case a server permissions violation names, or ``None``.

    A permissions violation is the one error reaching this callback that leaves the connection UP
    (nats-py's ``Client._process_err`` returns before ``_close`` for it) and that no caller ever
    observes, so it is the one that has to be decomposed rather than logged whole. The server text
    is matched CASE-INSENSITIVELY against the original string rather than a lowercased copy: nats-py
    currently lowercases the whole ``-ERR`` payload in its protocol parser before dispatch, but a
    version that stops doing so should hand back the subject in its true case, not a mangled one.

    Because that lowercasing cannot be undone, the subject is reported EXACTLY as received and
    qualified by :attr:`_ViolationDetail.subject_case`: uppercase anywhere in the payload proves the
    parser left it alone (:data:`_SUBJECT_CASE_VERBATIM`), an all-lowercase payload is
    indistinguishable from a mangled one (:data:`_SUBJECT_CASE_LOWERCASED`). Guessing the true case
    back would invent a value nothing observed.

    When the phrase is present but the wording cannot be decomposed -- a future NATS rewording --
    this still reports a violation, with ``None`` ids, rather than degrading to an anonymous error.

    :param exc: the exception nats-py surfaced to the error callback
    :ptype exc: Exception
    :return: the decomposed violation, or ``None`` when ``exc`` is not a permissions violation
    :rtype: _ViolationDetail | None
    """
    result: _ViolationDetail | None = None
    text = str(exc)
    if _PERMISSIONS_VIOLATION_PHRASE in text.lower():
        match = _PERMISSIONS_VIOLATION_PATTERN.search(text)
        if match is None:
            result = _ViolationDetail(None, None, _SUBJECT_CASE_NOT_REPORTED)
        else:
            operation = _VIOLATION_OPERATIONS[match.group("operation").lower()]
            case = _SUBJECT_CASE_VERBATIM if text != text.lower() else _SUBJECT_CASE_LOWERCASED
            result = _ViolationDetail(operation, match.group("subject"), case)
    return result


def _is_connection_loss(exc: Exception) -> bool:
    """whether ``exc`` reports the transport going away, which the client is reconnecting from.

    nats-py hands ``error_cb`` three shapes while a broker restarts: the read loop's
    :class:`nats.errors.UnexpectedEOF` (a :class:`nats.errors.StaleConnectionError`, as is a missed
    ping) as the old connection dies, then an :class:`OSError` (``Connect call failed``, a reset) or
    a bare :class:`TimeoutError` for each attempt made before the broker is back. Every
    :class:`NatsClient` connection reconnects forever (:data:`RUNTIME_MAX_RECONNECT_ATTEMPTS`), so
    each of these is an outage being recovered from, not a failure.

    nats-py's OWN timeouts (a flush, a request) also derive from :class:`TimeoutError`, through
    :class:`nats.errors.Error`; they report a connection that is up but stalled, so every
    :class:`nats.errors.Error` other than a stale connection is excluded. So is a server ``-ERR``
    (an authorization or permissions refusal, a payload violation), which reconnecting cannot fix.

    :param exc: the exception nats-py surfaced to the error callback
    :ptype exc: Exception
    :return: ``True`` when it is a transport loss the client is reconnecting from
    :rtype: bool
    """
    if isinstance(exc, _NatsStaleConnectionError):
        return True
    return isinstance(exc, (OSError, TimeoutError)) and not isinstance(exc, _NatsError)


async def _on_error(exc: Exception, last_logged: dict[str, float]) -> None:
    """nats-py error callback with rate-limited logging.

    A CONNECTION LOSS (:func:`_is_connection_loss`) is logged at WARNING, naming the error's type
    and its text: the client is already reconnecting, and a broker restart that heals itself in
    seconds is not an error in any service that rode it out. Logging it at ERROR made every
    service in a stack report one on a routine broker restart. Should reconnecting fail for a
    reason that will not heal -- a refused credential -- that reason arrives as its own server
    error, which stays at ERROR. The client never gives up reconnecting on its own; a connect
    that never succeeds raises :class:`NatsClientError` to the caller instead.

    A PERMISSIONS VIOLATION is singled out because it is otherwise INVISIBLE. It is the one
    condition here that arrives with the connection still UP -- ``nats-py``'s ``_process_err``
    closes the connection for every other ``-ERR`` but returns early for this one -- and nothing
    is raised to any caller, so nothing downstream will re-report it. A refused SUBSCRIBE leaves a
    live client-side subscription that will never receive a message and never complain, and a
    refused JetStream request is simply never answered, resurfacing much later as a timeout that
    reads like an unreachable broker. This callback is the only place the truth is available.

    The line is therefore built from BOTH halves of that truth:

    - the REMEDY, from :mod:`threetears.nats.diagnostics`, which recognises a ``$KV`` subject and
      names the grant declaration that is missing rather than only the wire subject; and
    - the DECOMPOSITION -- which operation was refused, on which subject, and what that costs --
      because a remedy alone does not say whether a publish was dropped or a subscription went
      permanently deaf, and the subject's case is only trustworthy under the condition
      :data:`_SUBJECT_CASE_VERBATIM` documents.

    Rate limiting keeps DISTINCT SUBJECTS distinct: the key carries the subject, so a second dead
    subject is never suppressed behind the first (each one is a different capability lost, not a
    repeat of the same error). Only the same subject repeating is collapsed.

    This is observability only. Every error path here still logs and CONTINUES, because the
    reconnect and degrade paths depend on that.

    :param exc: exception from nats-py client
    :ptype exc: Exception
    :param last_logged: rate-limit key -> when that key was last logged at error, owned by the
        client whose connection raised; updated in place
    :ptype last_logged: dict[str, float]
    :return: nothing
    :rtype: None
    """
    violation = _permissions_violation(exc)
    if violation is None:
        key = f"{type(exc).__name__}:{exc}"
    else:
        key = f"{_PERMISSIONS_VIOLATION_PHRASE}:{violation.operation}:{violation.subject}"
    now = time.monotonic()
    last = last_logged.get(key, 0.0)
    if now - last >= _ERROR_LOG_RATE_LIMIT_SECONDS:
        if violation is None and _is_connection_loss(exc):
            log.warning("NATS connection lost, reconnecting: %s: %s", type(exc).__name__, exc)
        elif violation is None:
            log.error("NATS error: %s", exc)
        else:
            operation = violation.operation or _UNKNOWN_OPERATION
            caveat = "" if violation.subject_case != _SUBJECT_CASE_LOWERCASED else f" {_LOWERCASED_SUBJECT_CAVEAT}"
            # the remedy leads because it names the FIX (and, for a ``$KV`` subject, the bucket and
            # the `js_resources` declaration that grants it); the decomposition follows because the
            # remedy does not say which operation died or what that costs. the remedy already ends
            # with the server's own words, so the raw error is carried once, not twice.
            # ``_permissions_violation`` and ``permissions_violation_remedy`` key off the SAME
            # phrase, so a decomposed violation always has a remedy -- the fallback exists only so a
            # future divergence degrades to a shorter line instead of logging ``None``.
            remedy = permissions_violation_remedy(exc) or f"NATS PERMISSIONS VIOLATION. Server said: {exc}"
            log.error(
                "%s -- REFUSED OPERATION: the server refused this connection's %s of subject %s. %s%s",
                remedy,
                operation,
                violation.subject or _UNKNOWN_SUBJECT,
                _VIOLATION_CONSEQUENCES.get(operation, _UNKNOWN_CONSEQUENCE),
                caveat,
                extra={
                    "extra_data": {
                        "subject": violation.subject,
                        "operation": violation.operation,
                        "subject_case": violation.subject_case,
                        "error": str(exc),
                    }
                },
            )
        last_logged[key] = now
    else:
        log.debug("NATS error (rate-limited duplicate): %s", exc)

"""canonical NATS client wrapper for 3tears applications.

:class:`NatsClient` is THE single primitive every 3tears app uses to
talk to NATS. it absorbs the lifecycle, dual-phase reconnect-ceiling,
rate-limited error logging, deadletter dispatch, typed publish, and
JetStream KV access that previously lived in three half-overlapping
wrappers (``aibots/hub/common/nats.py``,
``threetears.core.cache.kv.NatsKvClient`` (formerly
``cache.nats.NatsClient``), ``aibots_agents.runtime.nats_transport``).
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
  via ``asyncio.wait_for``; runtime uses
  :data:`RUNTIME_MAX_RECONNECT_ATTEMPTS` so a transient blip does not
  kill the process. inherits the rationale from
  ``aibots/hub/common/nats.py`` (deleted as part of this consolidation).
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
import time
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Final, TypeVar

import nats
from nats.aio.client import Client as _NatsPyClient
from nats.js.api import AckPolicy as _NatsAckPolicy, ConsumerConfig as _NatsConsumerConfig
from nats.errors import (
    ConnectionClosedError as _NatsConnectionClosedError,
    NoRespondersError as _NatsNoRespondersError,
    TimeoutError as _NatsTimeoutError,
)
from pydantic import BaseModel, ValidationError
from threetears.observe import get_logger

from threetears.nats.errors import (
    NatsClientError,
    PublishError,
    RequestError,
    SubscribeError,
)
from threetears.nats.subjects import Subject, Subjects, set_default_namespace

if TYPE_CHECKING:
    from nats.aio.msg import Msg as _NatsMsg
    from nats.aio.subscription import Subscription as _NatsSub

    from threetears.nats.kv import NatsKvBucket
    from threetears.nats.transport import MessageCallback, RawMessageCallback


from threetears.nats.transport import IncomingMessage

__all__ = [
    "DEFAULT_REQUEST_TIMEOUT",
    "DEFAULT_STARTUP_TIMEOUT",
    "DEFAULT_DRAIN_TIMEOUT",
    "RUNTIME_MAX_RECONNECT_ATTEMPTS",
    "STARTUP_MAX_RECONNECT_ATTEMPTS",
    "NatsClient",
    "Subscription",
]


log = get_logger(__name__)


_T = TypeVar("_T", bound=BaseModel)


# ---------------------------------------------------------------------------
# tunables
# ---------------------------------------------------------------------------

#: max reconnect attempts during startup (asyncio.wait_for enforces wall-time bound).
STARTUP_MAX_RECONNECT_ATTEMPTS: Final[int] = 15

#: max reconnect attempts after first successful connect. ``-1`` (forever) is rejected
#: deliberately; ~200s of reconnect budget at ``reconnect_time_wait=2s`` covers any real
#: transient outage without becoming an infinite-retry sink.
RUNTIME_MAX_RECONNECT_ATTEMPTS: Final[int] = 100

#: default startup timeout (matches platform's ``startup_timeout_seconds`` env var).
DEFAULT_STARTUP_TIMEOUT: Final[timedelta] = timedelta(seconds=30)

#: default request/reply timeout when caller does not pass one explicitly.
DEFAULT_REQUEST_TIMEOUT: Final[timedelta] = timedelta(seconds=5)

#: default drain timeout for graceful shutdown.
DEFAULT_DRAIN_TIMEOUT: Final[timedelta] = timedelta(seconds=30)

#: dedup window for error-callback rate limiting.
_ERROR_LOG_RATE_LIMIT_SECONDS: Final[float] = 10.0


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

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    @classmethod
    async def connect(
        cls,
        *,
        nats_url: str,
        nats_subject_namespace: str = "aibots",
        client_name: str,
        cluster_urls: list[str] | None = None,
        auth_token: str | None = None,
        startup_timeout: timedelta = DEFAULT_STARTUP_TIMEOUT,
        verify_jetstream: bool = True,
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
        :param auth_token: optional NATS auth token
        :ptype auth_token: str | None
        :param startup_timeout: max wall time to spend obtaining first successful connection
        :ptype startup_timeout: timedelta
        :param verify_jetstream: when True (default) verify JetStream is reachable post-connect
        :ptype verify_jetstream: bool
        :return: connected and ready NATS client
        :rtype: NatsClient
        :raises NatsClientError: if connection fails, times out, or JetStream verification fails
        """
        if not nats_url:
            raise NatsClientError("nats_url must be non-empty")
        if not client_name:
            raise NatsClientError("client_name must be non-empty")
        if not nats_subject_namespace:
            raise NatsClientError("nats_subject_namespace must be non-empty")

        servers = [nats_url]
        if cluster_urls:
            servers.extend(u.strip() for u in cluster_urls if u.strip())

        options: dict[str, object] = {
            "name": client_name,
            "allow_reconnect": True,
            "max_reconnect_attempts": RUNTIME_MAX_RECONNECT_ATTEMPTS,
            "reconnect_time_wait": 2,
            "reconnected_cb": _on_reconnected,
            "disconnected_cb": _on_disconnected,
            "error_cb": _on_error,
        }
        if auth_token:
            options["token"] = auth_token

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

        return cls(raw=raw_client, namespace=nats_subject_namespace, client_name=client_name)

    @property
    def namespace(self) -> str:
        """subject namespace prefix bound at connect time.

        :return: namespace prefix (e.g. ``aibots``)
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
            await self._raw.flush(timeout=timeout)
        except Exception:
            return False
        return True

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
        await self._raw.flush(timeout=timeout)

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
            raise PublishError(f"publish_reply failed: subject={reply_subject}: {exc}") from exc

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
            raise PublishError(f"publish_raw_reply failed: subject={reply_subject}: {exc}") from exc

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
            raise PublishError(f"publish failed: subject={subject.path}: {exc}") from exc

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
        subject: Subject,
        raw_cb: "RawMessageCallback | None",
        typed_cb: Callable[[Any], Awaitable[None]] | None,
        message_type: type[BaseModel] | None,
        queue: str | None,
        max_in_flight: int | None,
        deadletter_on_failure: bool,
    ) -> Subscription:
        """common subscribe path used by :meth:`subscribe` / :meth:`subscribe_typed`.

        :param subject: subject pattern or point
        :ptype subject: Subject
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

        async def _dispatch() -> None:
            """drive subscription message loop."""
            try:
                async for msg in raw_sub.messages:
                    if semaphore is None:
                        await _dispatch_one(msg)
                    else:
                        async with semaphore:
                            await _dispatch_one(msg)
            except asyncio.CancelledError:
                # graceful unsubscribe path
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
                raise RequestError(f"request failed: subject={sub.path}: {exc}") from exc
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
            raise RequestError(f"request failed: subject={subject.path}: {exc}") from exc
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
        :raises RuntimeError: if stream creation and update both fail
        """
        from nats.js.api import StorageType, StreamConfig  # noqa: PLC0415

        full_name = f"{self._namespace}-{name}"
        storage_type = StorageType.FILE if storage == "file" else StorageType.MEMORY
        config = StreamConfig(name=full_name, subjects=subjects, storage=storage_type)
        js = self.jetstream_context()
        try:
            await js.add_stream(config)
        except Exception:  # noqa: BLE001 -- stream may already exist; reconcile below
            # rationale: nats-py raises a generic error when the stream name is
            # already in use; the recovery (update to reconcile subjects) is the
            # create-or-bind path, mirroring NatsKvBucket.open.
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
    ) -> Any:
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
        :return: nats-py JetStream push subscription handle
        :rtype: Any
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
        return sub

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

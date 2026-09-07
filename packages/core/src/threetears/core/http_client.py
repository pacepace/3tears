"""traced, retried, circuit-broken outbound HTTP transport.

single async client for every *upstream* HTTP call the platform makes to a
service it does not own (an imported REST/OpenAPI API driven through an HTTP
tool, a webhook, a third-party endpoint). it is the one transport
``HttpApiTool`` binds to; no consumer opens a raw ``httpx`` client again.

the three concerns are reused, never hand-rolled:

- tracing -- :func:`threetears.observe.traced` wraps :meth:`TracedHttpClient.request`
  so every call emits an OTel span (zero-cost when OTel is absent).
- retry -- :func:`threetears.observe.retry_with_backoff` drives bounded
  exponential backoff over the per-attempt closure; transient failures
  (connect errors, timeouts, HTTP 5xx) retry, 4xx does not.
- circuit breaking -- a
  :class:`threetears.models.circuit_breaker.CircuitBreaker` is *injected*
  through the structural :class:`CircuitBreakerLike` protocol so this module
  (homed in ``core``) never imports ``threetears.models`` and its transitive
  ``langchain`` weight. ``core`` already depends on ``observe`` but not on
  ``models``; the injection keeps that layering seam intact.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import httpx
from threetears.core.config import DEFAULT_HTTP_TIMEOUT_SECONDS
from threetears.observe import retry_with_backoff, traced

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping
    from typing import Any

    from threetears.core.egress import EgressDriver

__all__ = ["ATTEMPTS_EXTENSION", "CircuitBreakerLike", "TracedHttpClient", "UpstreamHttpError"]

_SPAN_NAME = "threetears.core.http_client.request"

#: key on :attr:`httpx.Response.extensions` carrying how many attempts this
#: client spent to obtain the response, the successful one included.
#:
#: Retry lives inside :meth:`TracedHttpClient.request`, so every caller above
#: it sees one response and cannot tell whether it cost one exchange or three.
#: That is fine for a caller that only wants the body, and wrong for a caller
#: that has to account for what was spent: a transport wrapping this client
#: reports the count as its own per-attempt accounting, and a spend model that
#: bills per exchange under-bills by exactly the retries when it cannot see
#: them. Surfaced through ``extensions`` -- httpx's own channel for
#: transport-level facts about a response -- rather than by changing the
#: return type, so no existing caller has to learn a new shape to ignore it.
ATTEMPTS_EXTENSION = "threetears_attempts"


@runtime_checkable
class CircuitBreakerLike(Protocol):
    """structural stand-in for the injected circuit breaker.

    declares only the three-call fault-isolation protocol
    (:meth:`check` before an attempt, :meth:`record_success` /
    :meth:`record_failure` after). the real
    :class:`threetears.models.circuit_breaker.CircuitBreaker` satisfies it
    by shape, so ``core`` reuses that breaker without importing
    ``threetears.models``.
    """

    def check(self) -> None:
        """verifies the circuit allows the request; raises when OPEN.

        :raises CircuitOpenError: when the breaker is OPEN and its
            recovery timeout has not elapsed (raised by the concrete
            breaker; this module lets it propagate untouched)
        """
        ...

    def record_success(self) -> None:
        """records a successful upstream outcome."""
        ...

    def record_failure(self) -> None:
        """records a failed upstream outcome."""
        ...


class UpstreamHttpError(RuntimeError):
    """raised when an upstream request fails after all retries are exhausted.

    carries the last upstream HTTP status + response body so callers can
    pattern-match on error shape without re-issuing the request.
    ``status_code`` is ``None`` when the failure never produced a response
    (connect error / timeout on every attempt); in that case ``__cause__``
    carries the last transport exception, so a caller reporting the failure
    onward can tell a timeout from a refused connection. a 5xx exhaustion
    produced responses and therefore chains nothing.

    :ivar status_code: last upstream HTTP status, or ``None`` when no
        response was ever received
    :ivar body: last upstream response body (bytes; empty when no response)
    """

    def __init__(self, message: str, *, status_code: int | None, body: bytes) -> None:
        """capture status + body alongside the message.

        :param message: human-readable error description
        :ptype message: str
        :param status_code: last upstream HTTP status, or ``None`` when no
            response was received
        :ptype status_code: int | None
        :param body: last upstream response body
        :ptype body: bytes
        :return: nothing
        :rtype: None
        """
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class _Retryable(Exception):
    """private signal that an attempt failed in a retryable way.

    raised inside the per-attempt closure on a 5xx response so
    :func:`threetears.observe.retry_with_backoff` retries; never crosses
    the :meth:`TracedHttpClient.request` boundary.
    """


class TracedHttpClient:
    """async HTTP transport with tracing, bounded retry, and circuit breaking.

    owns exactly one :class:`httpx.AsyncClient` for its lifetime; supports
    ``async with`` and :meth:`aclose`. it authenticates to arbitrary
    upstreams with caller-supplied per-call headers, has no login concept,
    and retains no header/secret in a long-lived field.

    :param upstream_base_url: root URL of the upstream service; relative
        request paths are joined onto it. ``None`` opens a client with no
        fixed upstream, for a caller that passes absolute URLs per request
        (a shared client fronting several endpoints, whose host is not one
        value). A circuit breaker cannot be paired with ``None``: the breaker
        keys fault-isolation on one upstream, and there is no one upstream to
        key it on
    :ptype upstream_base_url: str | None
    :param circuit_breaker: optional injected breaker guarding this
        upstream; ``None`` disables circuit breaking (tests, upstreams that
        need no isolation)
    :ptype circuit_breaker: CircuitBreakerLike | None
    :param timeout: per-request timeout in seconds
    :ptype timeout: float
    :param max_attempts: maximum request attempts before raising
        :class:`UpstreamHttpError` (finite; forever-retry is wrong for a
        request)
    :ptype max_attempts: int
    :param initial_backoff: initial backoff seconds between retries
    :ptype initial_backoff: float
    :param max_backoff: maximum backoff seconds between retries
    :ptype max_backoff: float
    :param transport: optional httpx transport to bind (dependency-injection
        seam for tests; production leaves it ``None`` for the default
        network transport)
    :ptype transport: httpx.AsyncBaseTransport | None
    """

    def __init__(
        self,
        *,
        upstream_base_url: str | None,
        circuit_breaker: CircuitBreakerLike | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
        max_attempts: int = 3,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
        transport: httpx.AsyncBaseTransport | None = None,
        egress: EgressDriver | None = None,
        event_hooks: dict[str, list[Callable[..., Any]]] | None = None,
        headers: Mapping[str, str] | None = None,
        follow_redirects: bool = False,
    ) -> None:
        """capture config and open the single underlying httpx client.

        :param upstream_base_url: root URL of the upstream service, or
            ``None`` for a client with no fixed upstream (absolute request
            paths, no circuit breaker)
        :ptype upstream_base_url: str | None
        :param circuit_breaker: optional injected breaker
        :ptype circuit_breaker: CircuitBreakerLike | None
        :param timeout: per-request timeout in seconds
        :ptype timeout: float
        :param max_attempts: maximum request attempts
        :ptype max_attempts: int
        :param initial_backoff: initial backoff seconds
        :ptype initial_backoff: float
        :param max_backoff: maximum backoff seconds
        :ptype max_backoff: float
        :param transport: optional httpx transport (test seam)
        :ptype transport: httpx.AsyncBaseTransport | None
        :param egress: optional exit this client should leave by (see
            :mod:`threetears.core.egress`). ``transport`` wins when both are
            given, because it is the test seam and a test that pinned a
            transport must not have it replaced by ambient configuration
        :ptype egress: EgressDriver | None
        :param event_hooks: optional httpx ``event_hooks`` mapping
            (``{"request": [...], "response": [...]}``) forwarded to the
            underlying client. The on-response feedback channel a consumer
            needs to observe each response's status/latency -- e.g. to drive a
            status-driven rate-limit backoff (a 429/402 on-report hook) that
            this client's own bounded retry does not model. The hooks observe;
            they must not consume the response body (httpx re-reads it). None
            (the default) wires nothing, byte-identical to prior behaviour
        :ptype event_hooks: dict[str, list[Callable[..., Any]]] | None
        :param headers: optional client-level default headers applied to
            every request (e.g. an API key or a ``User-Agent`` an upstream
            keys on). Per-call ``headers`` merge over these, matching httpx.
            ``None`` (the default) sends no default headers
        :ptype headers: Mapping[str, str] | None
        :param follow_redirects: whether the client follows 3xx redirects by
            default (httpx defaults to ``False``). A per-call ``follow_redirects``
            overrides this for one request
        :ptype follow_redirects: bool
        :return: nothing
        :rtype: None
        :raises ValueError: when ``upstream_base_url`` is the empty string
            (pass ``None`` for a deliberately unfixed upstream), or when a
            ``circuit_breaker`` is paired with a ``None`` ``upstream_base_url``
        """
        if upstream_base_url == "":
            raise ValueError("upstream_base_url must be non-empty; pass None for a client with no fixed upstream")
        if upstream_base_url is None and circuit_breaker is not None:
            raise ValueError(
                "a circuit_breaker keys fault-isolation on one upstream; it cannot pair with upstream_base_url=None"
            )
        self._circuit_breaker = circuit_breaker
        self._max_attempts = max_attempts
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._host = httpx.URL(upstream_base_url).host if upstream_base_url else None
        self._egress = egress
        # An explicit transport wins. It is the documented test seam, and a test that binds a
        # transport is asserting on what this client does with it -- letting a configured
        # egress override that would make the seam conditional on deployment config.
        resolved_transport = transport if transport is not None else (egress.httpx_transport() if egress else None)
        self._client = httpx.AsyncClient(
            base_url=upstream_base_url if upstream_base_url is not None else "",
            timeout=timeout,
            transport=resolved_transport,
            event_hooks=event_hooks if event_hooks is not None else {},
            headers=dict(headers) if headers else None,
            follow_redirects=follow_redirects,
        )

    @property
    def egress_name(self) -> str | None:
        """Which exit this client leaves by, for recording against a result.

        ``None`` when none was configured, because that is a different fact from choosing the
        default route and the two must stay distinguishable in a table. :class:`DirectEgress`
        exists so "direct" can be a stated choice; a client given it reports ``"direct"``, and
        one given nothing reports ``None``. Collapsing them here would make every row of an
        unconfigured deployment claim a decision nobody made.
        """
        return self._egress.name if self._egress is not None else None

    async def __aenter__(self) -> TracedHttpClient:
        """return self for ``async with`` ergonomics.

        :return: self
        :rtype: TracedHttpClient
        """
        return self

    async def __aexit__(self, *_args: object) -> None:
        """close the underlying httpx client on context exit.

        :return: nothing
        :rtype: None
        """
        await self.aclose()

    async def aclose(self) -> None:
        """close the underlying httpx client.

        :return: nothing
        :rtype: None
        """
        await self._client.aclose()

    @traced(name=_SPAN_NAME)
    async def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        content: bytes | None = None,
        data: Mapping[str, Any] | None = None,
        json: Any = None,
        timeout: float | None = None,
        follow_redirects: bool | None = None,
    ) -> httpx.Response:
        """perform one upstream request with tracing, retry, and breaking.

        the circuit breaker (when injected) is checked once up front: an
        OPEN breaker raises ``CircuitOpenError`` which propagates untouched
        (no request sent, no failure recorded). the request itself runs
        under :func:`threetears.observe.retry_with_backoff`: connect
        errors, timeouts, and 5xx responses retry with bounded backoff; a
        4xx response is returned to the caller un-retried and does not touch
        the breaker. on exhaustion :class:`UpstreamHttpError` is raised
        carrying the last status/body. never raises on 4xx.

        :param method: HTTP verb (GET / POST / PATCH / DELETE / ...)
        :ptype method: str
        :param path: request path joined onto ``upstream_base_url``
        :ptype path: str
        :param headers: optional per-call request headers (never retained,
            never traced/logged)
        :ptype headers: Mapping[str, str] | None
        :param params: optional query-string parameters
        :ptype params: Mapping[str, Any] | None
        :param content: optional raw request body bytes
        :ptype content: bytes | None
        :param data: optional form-encoded request body
            (``application/x-www-form-urlencoded``); mutually exclusive with
            ``content``/``json`` per httpx
        :ptype data: Mapping[str, Any] | None
        :param json: optional JSON request body
        :ptype json: Any
        :param timeout: per-call override of the client's configured timeout,
            in seconds, so a caller holding a deadline can bound this call to
            what remains of it. ``None`` uses the configured value. Bounds each
            attempt; the retry schedule is unchanged
        :ptype timeout: float | None
        :param follow_redirects: per-call override of the client's redirect
            policy; ``None`` uses the client-level default
        :ptype follow_redirects: bool | None
        :return: full upstream response (caller inspects any non-2xx; not
            raised on 4xx/5xx except retry exhaustion). carries the attempt
            count under :data:`ATTEMPTS_EXTENSION` in ``extensions``
        :rtype: httpx.Response
        :raises CircuitOpenError: when the injected breaker is OPEN
        :raises UpstreamHttpError: when every attempt fails (5xx /
            connect / timeout) up to ``max_attempts``; chains the last
            transport exception as ``__cause__`` when no response was
            ever received
        """
        if self._circuit_breaker is not None:
            # a tripped breaker fast-fails; CircuitOpenError escapes untouched
            # (not caught as retryable, no failure recorded).
            self._circuit_breaker.check()

        captured: httpx.Response | None = None
        attempts = 0
        # the last TRANSPORT failure, kept so exhaustion can chain it. ``retry_with_backoff``
        # collapses every attempt into a bool, so without this the caller is handed a
        # status-less UpstreamHttpError that cannot say whether the upstream timed out or
        # refused the connection -- two failures that call for different operator action.
        transport_error: BaseException | None = None

        async def _attempt_once() -> None:
            nonlocal captured, attempts, transport_error
            attempts += 1
            try:
                response = await self._client.request(
                    method,
                    path,
                    headers=dict(headers) if headers else None,
                    params=dict(params) if params else None,
                    content=content,
                    data=dict(data) if data else None,
                    json=json,
                    # httpx's own sentinel, not None: None is a MEANINGFUL value there
                    # (wait forever), so passing it through for "caller said nothing"
                    # would turn an unstated timeout into no timeout at all.
                    timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT,
                    # Same sentinel discipline: httpx reads None as an explicit "do not
                    # follow", so "caller said nothing" must defer to the client default.
                    follow_redirects=follow_redirects if follow_redirects is not None else httpx.USE_CLIENT_DEFAULT,
                )
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                transport_error = exc
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_failure()
                raise
            captured = response
            if response.status_code >= 500:
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_failure()
                raise _Retryable
            if response.status_code < 400 and self._circuit_breaker is not None:
                # 2xx/3xx is a genuine upstream success; a 4xx client error
                # leaves the breaker untouched.
                self._circuit_breaker.record_success()

        succeeded = await retry_with_backoff(
            _attempt_once,
            name=_SPAN_NAME,
            max_attempts=self._max_attempts,
            initial_backoff=self._initial_backoff,
            max_backoff=self._max_backoff,
        )

        # host for the span/error: the response's when one arrived, else the path's own host
        # (absolute in the no-base-url case), so a connect-error exhaustion still names WHERE.
        request_host = captured.request.url.host if captured is not None else (httpx.URL(path).host or None)
        self._record_span_attributes(
            method=method,
            status_code=captured.status_code if captured is not None else None,
            host=request_host,
        )

        if not succeeded or captured is None:
            status = captured.status_code if captured is not None else None
            body = captured.content if captured is not None else b""
            raise UpstreamHttpError(
                f"upstream request to {self._host or request_host or 'the upstream'} "
                f"failed after {self._max_attempts} attempts",
                status_code=status,
                body=body,
            ) from transport_error

        captured.extensions[ATTEMPTS_EXTENSION] = attempts
        result = captured
        return result

    async def get(
        self,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        follow_redirects: bool | None = None,
    ) -> httpx.Response:
        """GET ``path`` (delegates to :meth:`request`).

        :param path: request path joined onto ``upstream_base_url``
        :ptype path: str
        :param headers: optional per-call request headers
        :ptype headers: Mapping[str, str] | None
        :param params: optional query-string parameters
        :ptype params: Mapping[str, Any] | None
        :param timeout: per-call timeout override in seconds; None uses the
            configured value
        :ptype timeout: float | None
        :param follow_redirects: per-call redirect-policy override; None uses
            the client-level default
        :ptype follow_redirects: bool | None
        :return: full upstream response
        :rtype: httpx.Response
        """
        return await self.request(
            "GET", path, headers=headers, params=params, timeout=timeout, follow_redirects=follow_redirects
        )

    async def post(
        self,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        content: bytes | None = None,
        data: Mapping[str, Any] | None = None,
        json: Any = None,
        timeout: float | None = None,
        follow_redirects: bool | None = None,
    ) -> httpx.Response:
        """POST ``path`` (delegates to :meth:`request`).

        :param path: request path joined onto ``upstream_base_url``
        :ptype path: str
        :param headers: optional per-call request headers
        :ptype headers: Mapping[str, str] | None
        :param params: optional query-string parameters
        :ptype params: Mapping[str, Any] | None
        :param content: optional raw request body bytes
        :ptype content: bytes | None
        :param data: optional form-encoded request body
        :ptype data: Mapping[str, Any] | None
        :param json: optional JSON request body
        :ptype json: Any
        :param timeout: per-call timeout override in seconds; None uses the
            configured value
        :ptype timeout: float | None
        :param follow_redirects: per-call redirect-policy override; None uses
            the client-level default
        :ptype follow_redirects: bool | None
        :return: full upstream response
        :rtype: httpx.Response
        """
        return await self.request(
            "POST",
            path,
            headers=headers,
            params=params,
            content=content,
            data=data,
            json=json,
            timeout=timeout,
            follow_redirects=follow_redirects,
        )

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        follow_redirects: bool | None = None,
    ) -> AsyncIterator[httpx.Response]:
        """Stream a response body without buffering it -- for bulk downloads.

        Tracing, retry, and circuit breaking apply to *establishing* the
        response (sending the request and reading its headers), exactly as
        :meth:`request` does: connect errors, timeouts, and 5xx retry with
        bounded backoff; a 4xx is yielded to the caller un-retried and does not
        touch the breaker; on exhaustion :class:`UpstreamHttpError` is raised.
        Once a response is yielded its body is NOT read here -- the caller
        iterates it (``aiter_bytes`` / ``aiter_raw``), so a multi-hundred-MB
        download never lands in memory. That is the whole point, and the reason
        this is separate from :meth:`request`, whose bounded retry re-reads the
        whole body per attempt: retrying is safe only up to the moment the body
        starts streaming, so no retry happens once the caller holds the stream.

        Use for large or unbounded downloads (a dataset ZIP/CSV, a masterfile);
        use :meth:`request`/:meth:`get` for API responses whose body is small
        enough to buffer and whose failure should be retried whole.

        :param method: HTTP verb (typically ``GET``)
        :ptype method: str
        :param path: request path joined onto ``upstream_base_url``
        :ptype path: str
        :param headers: optional per-call headers (e.g. a ``Range`` header for a
            partial fetch); never retained, never traced/logged
        :ptype headers: Mapping[str, str] | None
        :param params: optional query-string parameters
        :ptype params: Mapping[str, Any] | None
        :param timeout: per-call timeout override in seconds; None uses the
            configured value. Bounds establishing the response; the caller's own
            iteration of the body is not bounded by it
        :ptype timeout: float | None
        :param follow_redirects: per-call redirect-policy override; None uses
            the client-level default
        :ptype follow_redirects: bool | None
        :return: an async context manager yielding the open streaming response
        :rtype: AsyncIterator[httpx.Response]
        :raises CircuitOpenError: when the injected breaker is OPEN
        :raises UpstreamHttpError: when establishing the response fails every
            attempt (5xx / connect / timeout) up to ``max_attempts``
        """
        if self._circuit_breaker is not None:
            self._circuit_breaker.check()

        stream_cm: Any = None
        response: httpx.Response | None = None
        error_body = b""
        attempts = 0
        transport_error: BaseException | None = None

        async def _open_once() -> None:
            nonlocal stream_cm, response, error_body, attempts, transport_error
            attempts += 1
            cm = self._client.stream(
                method,
                path,
                headers=dict(headers) if headers else None,
                params=dict(params) if params else None,
                timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT,
                follow_redirects=follow_redirects if follow_redirects is not None else httpx.USE_CLIENT_DEFAULT,
            )
            try:
                resp = await cm.__aenter__()
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                transport_error = exc
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_failure()
                raise
            if resp.status_code >= 500:
                # Read the (small) error body for the exhaustion error, then close
                # this failed stream before retrying so nothing is left open.
                error_body = await resp.aread()
                response = resp
                await cm.__aexit__(None, None, None)
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_failure()
                raise _Retryable
            if resp.status_code < 400 and self._circuit_breaker is not None:
                self._circuit_breaker.record_success()
            stream_cm = cm
            response = resp

        succeeded = await retry_with_backoff(
            _open_once,
            name=_SPAN_NAME,
            max_attempts=self._max_attempts,
            initial_backoff=self._initial_backoff,
            max_backoff=self._max_backoff,
        )
        request_host = response.request.url.host if response is not None else (httpx.URL(path).host or None)
        self._record_span_attributes(
            method=method,
            status_code=response.status_code if response is not None else None,
            host=request_host,
        )

        if not succeeded or stream_cm is None or response is None:
            status = response.status_code if response is not None else None
            raise UpstreamHttpError(
                f"upstream stream to {self._host or request_host or 'the upstream'} "
                f"failed after {self._max_attempts} attempts",
                status_code=status,
                body=error_body,
            ) from transport_error

        response.extensions[ATTEMPTS_EXTENSION] = attempts
        try:
            yield response
        finally:
            await stream_cm.__aexit__(None, None, None)

    def _record_span_attributes(self, *, method: str, status_code: int | None, host: str | None = None) -> None:
        """set host/method/status on the active span; never a header value.

        no-op when OpenTelemetry is not installed (import guarded). only
        the three non-secret attributes are recorded -- credential headers
        are never passed here and ``@traced`` arg-recording stays off, so
        no secret can reach a span.

        :param method: HTTP verb of the request
        :ptype method: str
        :param status_code: final upstream status, or ``None`` when no
            response was received
        :ptype status_code: int | None
        :param host: per-request host, used when the client has no fixed
            upstream (``upstream_base_url=None``); falls back to the client's
            configured host
        :ptype host: str | None
        :return: nothing
        :rtype: None
        """
        try:
            from opentelemetry import trace
        # NOSILENT: optional dependency probe; absence is a supported configuration
        except ImportError:
            return
        span = trace.get_current_span()
        recorded_host = host or self._host
        if recorded_host is not None:
            span.set_attribute("http.host", recorded_host)
        span.set_attribute("http.method", method)
        if status_code is not None:
            span.set_attribute("http.status_code", status_code)

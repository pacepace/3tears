"""unit tests for :mod:`threetears.core.http_client`.

exercises the traced/retried/circuit-broken outbound HTTP transport with a
fake httpx transport (``httpx.MockTransport``) so no network is touched.
covers: 2xx/4xx passthrough (no raise), 5xx retry via
``observe.retry_with_backoff``, exhaustion raising ``UpstreamHttpError``,
circuit-breaker fast-fail + record-success/record-failure semantics, and
the OTel span emission with the secret-hygiene guarantee (no header value
lands on a span attribute).
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from threetears.core.http_client import (
    ATTEMPTS_EXTENSION,
    CircuitBreakerLike,
    TracedHttpClient,
    UpstreamHttpError,
)
from threetears.models.circuit_breaker import CircuitBreaker, CircuitOpenError

_BASE_URL = "https://api.example.test"


class _BreakerSpy:
    """records the circuit-breaker three-call protocol for assertions.

    structurally satisfies :class:`CircuitBreakerLike` (``check`` /
    ``record_success`` / ``record_failure``); a spy double, not a
    protocol fake, so it stands in for the injected breaker while the
    tests assert which transitions fired.

    :ivar checks: count of :meth:`check` calls
    :ivar successes: count of :meth:`record_success` calls
    :ivar failures: count of :meth:`record_failure` calls
    """

    def __init__(self) -> None:
        self.checks = 0
        self.successes = 0
        self.failures = 0

    def check(self) -> None:
        """records a gate check (never trips)."""
        self.checks += 1

    def record_success(self) -> None:
        """records an upstream success."""
        self.successes += 1

    def record_failure(self) -> None:
        """records an upstream failure."""
        self.failures += 1


def _sequenced_transport(statuses: list[int]) -> tuple[httpx.MockTransport, list[int]]:
    """build a transport returning ``statuses`` in order, then repeating last.

    :param statuses: HTTP status codes to return per successive call
    :ptype statuses: list[int]
    :return: the transport plus a mutable call-count list (index 0)
    :rtype: tuple[httpx.MockTransport, list[int]]
    """
    calls = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        idx = min(calls[0], len(statuses) - 1)
        status = statuses[idx]
        calls[0] += 1
        return httpx.Response(status, text=f"body-{status}")

    return httpx.MockTransport(handler), calls


def _recording_transport() -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """build a transport that answers 200 and keeps every request it was handed.

    Used where the assertion is about what reached the wire rather than what
    came back -- the per-call timeout lands on ``Request.extensions``, which
    only the request object carries.

    :return: the transport plus the list it appends each request to
    :rtype: tuple[httpx.MockTransport, list[httpx.Request]]
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="body-200")

    return httpx.MockTransport(handler), seen


def _raising_transport(exc: Exception) -> tuple[httpx.MockTransport, list[int]]:
    """build a transport that raises ``exc`` on every call.

    :param exc: exception instance to raise per call
    :ptype exc: Exception
    :return: the transport plus a mutable call-count list (index 0)
    :rtype: tuple[httpx.MockTransport, list[int]]
    """
    calls = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        calls[0] += 1
        raise exc

    return httpx.MockTransport(handler), calls


def _client(transport: httpx.MockTransport, **kwargs: object) -> TracedHttpClient:
    """construct a client with fast (zero) backoff for deterministic tests.

    :param transport: fake httpx transport to bind
    :ptype transport: httpx.MockTransport
    :param kwargs: extra constructor overrides
    :ptype kwargs: object
    :return: configured client
    :rtype: TracedHttpClient
    """
    params: dict[str, object] = {
        "upstream_base_url": _BASE_URL,
        "transport": transport,
        "initial_backoff": 0.0,
        "max_backoff": 0.0,
    }
    params.update(kwargs)
    return TracedHttpClient(**params)  # type: ignore[arg-type]


async def test_returns_200_no_raise() -> None:
    transport, _calls = _sequenced_transport([200])
    async with _client(transport) as client:
        response = await client.request("GET", "/thing")
    assert response.status_code == 200
    assert response.text == "body-200"


async def test_returns_404_no_raise() -> None:
    transport, calls = _sequenced_transport([404])
    async with _client(transport) as client:
        response = await client.request("GET", "/missing")
    assert response.status_code == 404
    # a 4xx is a client error, not an upstream fault: no retry.
    assert calls[0] == 1


async def test_retries_5xx_then_success() -> None:
    transport, calls = _sequenced_transport([500, 500, 200])
    async with _client(transport) as client:
        response = await client.request("GET", "/flaky")
    assert response.status_code == 200
    assert calls[0] == 3


async def test_all_5xx_raises_upstream_error_with_status() -> None:
    transport, calls = _sequenced_transport([500])
    async with _client(transport, max_attempts=3) as client:
        with pytest.raises(UpstreamHttpError) as exc_info:
            await client.request("GET", "/down")
    assert exc_info.value.status_code == 500
    assert exc_info.value.body == b"body-500"
    assert calls[0] == 3


async def test_connect_error_exhaustion_raises_status_none() -> None:
    transport, calls = _raising_transport(httpx.ConnectError("refused"))
    async with _client(transport, max_attempts=3) as client:
        with pytest.raises(UpstreamHttpError) as exc_info:
            await client.request("GET", "/dead")
    assert exc_info.value.status_code is None
    assert calls[0] == 3


async def test_exhaustion_chains_the_underlying_transport_error() -> None:
    """``status_code is None`` says only "no response"; the cause says WHICH.

    A caller reporting the failure onward -- an MCP tool result, a datasource
    imperative -- has to tell a timeout apart from a refused connection, and
    both arrive here as the same status-less error otherwise.
    """
    transport, _calls = _raising_transport(httpx.TimeoutException("timed out"))
    async with _client(transport, max_attempts=2) as client:
        with pytest.raises(UpstreamHttpError) as exc_info:
            await client.request("GET", "/slow")
    assert isinstance(exc_info.value.__cause__, httpx.TimeoutException)


async def test_exhaustion_chains_a_connect_error_as_the_cause() -> None:
    transport, _calls = _raising_transport(httpx.ConnectError("refused"))
    async with _client(transport, max_attempts=2) as client:
        with pytest.raises(UpstreamHttpError) as exc_info:
            await client.request("GET", "/dead")
    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)


async def test_retries_remote_protocol_error_then_succeeds() -> None:
    """A server that disconnects without sending a response is a transient
    failure -- the request never got an answer, so re-issuing it is safe and
    is what a hand-rolled caller loop used to do. Now the client does it."""
    calls = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        calls[0] += 1
        if calls[0] == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response")
        return httpx.Response(200, text="ok")

    async with _client(httpx.MockTransport(handler), max_attempts=3) as client:
        response = await client.request("GET", "/flaky")
    assert response.status_code == 200
    assert calls[0] == 2  # the disconnect was retried


async def test_remote_protocol_error_exhaustion_chains_the_cause() -> None:
    transport, _calls = _raising_transport(httpx.RemoteProtocolError("disconnected"))
    async with _client(transport, max_attempts=2) as client:
        with pytest.raises(UpstreamHttpError) as exc_info:
            await client.request("GET", "/dead")
    assert exc_info.value.status_code is None
    assert isinstance(exc_info.value.__cause__, httpx.RemoteProtocolError)


async def test_5xx_exhaustion_has_no_transport_cause() -> None:
    """A 5xx is a RESPONSE; there is no transport exception to chain."""
    transport, _calls = _sequenced_transport([500])
    async with _client(transport, max_attempts=2) as client:
        with pytest.raises(UpstreamHttpError) as exc_info:
            await client.request("GET", "/down")
    assert exc_info.value.__cause__ is None


async def test_circuit_open_fast_fails_without_request() -> None:
    breaker = CircuitBreaker(provider_name="test-upstream", failure_threshold=1)
    breaker.record_failure()  # trips CLOSED -> OPEN
    failures_before = breaker.failure_count
    transport, calls = _sequenced_transport([200])
    async with _client(transport, circuit_breaker=breaker) as client:
        with pytest.raises(CircuitOpenError):
            await client.request("GET", "/thing")
    # no request sent and the open-circuit rejection did NOT record a failure.
    assert calls[0] == 0
    assert breaker.failure_count == failures_before


async def test_breaker_records_success_on_2xx() -> None:
    spy = _BreakerSpy()
    transport, _calls = _sequenced_transport([200])
    async with _client(transport, circuit_breaker=spy) as client:
        await client.request("GET", "/ok")
    assert spy.successes == 1
    assert spy.failures == 0
    assert spy.checks == 1


async def test_breaker_untouched_on_4xx() -> None:
    spy = _BreakerSpy()
    transport, _calls = _sequenced_transport([404])
    async with _client(transport, circuit_breaker=spy) as client:
        await client.request("GET", "/missing")
    assert spy.successes == 0
    assert spy.failures == 0


async def test_breaker_records_failure_on_5xx() -> None:
    spy = _BreakerSpy()
    transport, _calls = _sequenced_transport([500])
    async with _client(transport, circuit_breaker=spy, max_attempts=2) as client:
        with pytest.raises(UpstreamHttpError):
            await client.request("GET", "/down")
    assert spy.failures == 2
    assert spy.successes == 0


async def test_breaker_records_failure_on_connect_error() -> None:
    spy = _BreakerSpy()
    transport, _calls = _raising_transport(httpx.ConnectError("refused"))
    async with _client(transport, circuit_breaker=spy, max_attempts=2) as client:
        with pytest.raises(UpstreamHttpError):
            await client.request("GET", "/dead")
    assert spy.failures == 2
    assert spy.successes == 0


async def test_get_and_post_convenience() -> None:
    seen: list[tuple[str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.content))
        return httpx.Response(200, text="ok")

    async with _client(httpx.MockTransport(handler)) as client:
        get_response = await client.get("/a", params={"q": "1"})
        post_response = await client.post("/b", json={"k": "v"})
    assert get_response.status_code == 200
    assert post_response.status_code == 200
    assert seen[0][0] == "GET"
    assert seen[1][0] == "POST"


async def test_head_issues_a_head_request() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return httpx.Response(200, headers={"Content-Length": "4096"})

    async with _client(httpx.MockTransport(handler)) as client:
        response = await client.head("/thing")
    assert response.status_code == 200
    assert response.headers["Content-Length"] == "4096"
    assert seen == ["HEAD"]


async def test_head_forwards_follow_redirects() -> None:
    async with _client(_redirecting_transport(), follow_redirects=False) as client:
        followed = await client.head("/start", follow_redirects=True)
        not_followed = await client.head("/start")
    assert followed.status_code == 200
    assert not_followed.status_code == 302


async def test_span_emitted_and_no_header_leak() -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import threetears.observe.tracing as tracing_mod

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    secret = "Bearer super-secret-token-value"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ok")

    with (
        patch.object(tracing_mod, "_otel_available", True),
        patch("opentelemetry.trace.get_tracer", provider.get_tracer),
    ):
        async with _client(httpx.MockTransport(handler)) as client:
            await client.request("GET", "/thing", headers={"Authorization": secret})

    spans = exporter.get_finished_spans()
    names = [span.name for span in spans]
    assert "threetears.core.http_client.request" in names

    for span in spans:
        for value in span.attributes.values():
            assert secret not in str(value)


def test_breaker_spy_satisfies_protocol() -> None:
    # runtime_checkable structural check: the injected double is a
    # CircuitBreakerLike, and so is the real reuse target.
    assert isinstance(_BreakerSpy(), CircuitBreakerLike)
    assert isinstance(CircuitBreaker(provider_name="x"), CircuitBreakerLike)


class TestEgressWiring:
    """The one transport, leaving by a configured exit."""

    def test_an_egress_driver_supplies_the_transport(self) -> None:
        """This is the reuse the seam exists for: httpx proxying IS a transport, and
        ``TracedHttpClient`` already had a transport seam, so an exit needed no new plumbing."""
        from threetears.core.egress import ProxyEgress
        from threetears.core.http_client import TracedHttpClient

        client = TracedHttpClient(
            upstream_base_url="https://upstream.example",
            egress=ProxyEgress("tor", "socks5://127.0.0.1:9050"),
        )
        assert client.egress_name == "tor"

        # The bound transport must be the PROXIED one, not merely "a transport". httpx binds a
        # default transport regardless, so both `is not None` and `is not <other instance>`
        # were true with the egress ignored entirely -- assertions that could not fail, on the
        # one property this test exists for.
        #
        # httpx builds a different POOL for a proxied transport: `AsyncHTTPProxy` rather than
        # `AsyncConnectionPool`, carrying the proxy url. That is the observable difference, so
        # it is what gets asserted. Reaching into the pool is reaching into httpx's internals,
        # which is worth it here: the alternative is an assertion that passes when the feature
        # is deleted.
        pool = client._client._transport._pool  # noqa: SLF001 -- the pool type IS the assertion
        assert type(pool).__name__ != "AsyncConnectionPool", (
            "the configured exit was ignored; this is the unproxied pool httpx builds by default"
        )
        # The scheme decides the pool class -- AsyncSOCKSProxy here, AsyncHTTPProxy for an
        # http:// exit -- so the url is asserted rather than the class name, which is the part
        # that says WHICH exit rather than merely that there is one.
        assert "9050" in str(getattr(pool, "_proxy_url", "")), "proxied, but not through the configured exit"

    def test_an_explicit_transport_wins_over_a_configured_egress(self) -> None:
        """``transport`` is the documented test seam.

        A test that binds one is asserting on what this client does with it; letting ambient
        deployment configuration replace it would make the seam conditional on config, which
        is the sort of thing that passes locally and behaves differently in production.
        """
        import httpx
        from threetears.core.egress import ProxyEgress
        from threetears.core.http_client import TracedHttpClient

        pinned = httpx.MockTransport(lambda _req: httpx.Response(200))
        client = TracedHttpClient(
            upstream_base_url="https://upstream.example",
            transport=pinned,
            egress=ProxyEgress("tor", "socks5://127.0.0.1:9050"),
        )
        assert client._client._transport is pinned  # noqa: SLF001

    def test_no_egress_reports_nothing_and_direct_egress_reports_direct(self) -> None:
        """The two facts stay apart: nobody configured an exit, versus somebody chose the default.

        Both are asserted together because the value of either is entirely in its contrast with
        the other. An earlier convention returned ``"direct"`` for both, which made every row of
        an unconfigured deployment indistinguishable from a deliberate choice of the default
        route -- and :class:`DirectEgress` exists precisely so that choice can be stated.
        """
        from threetears.core.egress import DirectEgress
        from threetears.core.http_client import TracedHttpClient

        assert TracedHttpClient(upstream_base_url="https://upstream.example").egress_name is None
        chosen = TracedHttpClient(upstream_base_url="https://upstream.example", egress=DirectEgress())
        assert chosen.egress_name == "direct"


class TestACallerHoldingADeadlineCanBoundTheCall:
    """A per-call timeout, because retry lives inside this client.

    The client's configured timeout is a deployment fact and stays one. What
    a caller may additionally hold is a *deadline* -- a tool envelope's
    ``deadline_seconds``, a transport's remaining budget -- and before this
    override existed the only way to honour one was to construct a fresh
    client per call, which throws away the pool and the breaker to change a
    single number.
    """

    async def test_a_per_call_bound_reaches_the_request(self) -> None:
        transport, seen = _recording_transport()
        async with _client(transport, timeout=30.0) as client:
            await client.request("GET", "/thing", timeout=0.25)
        assert seen[0].extensions["timeout"]["read"] == pytest.approx(0.25)

    async def test_saying_nothing_leaves_the_configured_timeout_in_place(self) -> None:
        """The override is an override.

        httpx reads ``timeout=None`` as *wait forever*, so forwarding None for
        "the caller said nothing" would silently convert every unstated call
        into an unbounded one -- the opposite of what a timeout parameter is
        for.
        """
        transport, seen = _recording_transport()
        async with _client(transport, timeout=9.0) as client:
            await client.request("GET", "/thing")
        assert seen[0].extensions["timeout"]["read"] == pytest.approx(9.0)

    async def test_get_and_post_forward_it_too(self) -> None:
        """The convenience methods are the ones most call sites actually use."""
        transport, seen = _recording_transport()
        async with _client(transport, timeout=30.0) as client:
            await client.get("/thing", timeout=1.5)
            await client.post("/thing", json={"a": 1}, timeout=2.5)
        assert seen[0].extensions["timeout"]["read"] == pytest.approx(1.5)
        assert seen[1].extensions["timeout"]["read"] == pytest.approx(2.5)


class TestTheAttemptCountIsVisibleToACallerThatMustAccountForIt:
    """Retry is invisible from outside, and for a billing caller that is a bug.

    This client retries 5xx and connect failures internally and returns one
    response, so a caller that bills per exchange sees one where there were
    three. The count rides ``extensions`` -- httpx's own channel for
    transport-level facts -- so no existing caller has to learn a new return
    shape to ignore it.
    """

    async def test_a_clean_call_reports_one_attempt(self) -> None:
        transport, _calls = _sequenced_transport([200])
        async with _client(transport) as client:
            response = await client.request("GET", "/thing")
        assert response.extensions[ATTEMPTS_EXTENSION] == 1

    async def test_a_retried_call_reports_every_attempt(self) -> None:
        transport, calls = _sequenced_transport([500, 500, 200])
        async with _client(transport, max_attempts=3) as client:
            response = await client.request("GET", "/thing")
        assert calls[0] == 3
        assert response.extensions[ATTEMPTS_EXTENSION] == 3

    async def test_the_count_matches_what_the_upstream_actually_saw(self) -> None:
        """Non-vacuous: the number is counted, not copied from ``max_attempts``."""
        transport, calls = _sequenced_transport([500, 200])
        async with _client(transport, max_attempts=5) as client:
            response = await client.request("GET", "/thing")
        assert response.extensions[ATTEMPTS_EXTENSION] == calls[0] == 2


async def test_event_hooks_observe_the_429_the_client_returns() -> None:
    """A 429 is a 4xx: this client returns it un-retried (429 handling is the
    throttle's job, not the transport's). The injected response hook still fires
    on it -- exactly the seam a status-driven backoff needs: the hook sees the
    429, feeds the throttle's on_report, and the caller gets the 429 back."""
    transport, calls = _sequenced_transport([429])
    seen_statuses: list[int] = []

    async def _on_response(response: httpx.Response) -> None:
        seen_statuses.append(response.status_code)

    async with _client(transport, event_hooks={"response": [_on_response]}) as client:
        response = await client.request("GET", "/limited")

    assert response.status_code == 429  # returned un-retried (4xx)
    assert calls[0] == 1
    assert seen_statuses == [429]  # the hook observed it -> throttle can back off


async def test_event_hooks_fire_on_every_retried_attempt() -> None:
    """On a 5xx retry, the response hook fires per attempt -- the consumer sees
    every response, not just the final one."""
    transport, _calls = _sequenced_transport([500, 500, 200])
    seen_statuses: list[int] = []
    requests_seen: list[str] = []

    async def _on_request(request: httpx.Request) -> None:
        requests_seen.append(request.method)

    async def _on_response(response: httpx.Response) -> None:
        seen_statuses.append(response.status_code)

    hooks = {"request": [_on_request], "response": [_on_response]}
    async with _client(transport, event_hooks=hooks) as client:
        response = await client.request("GET", "/flaky")

    assert response.status_code == 200
    assert seen_statuses == [500, 500, 200]  # fired on every attempt
    assert requests_seen == ["GET", "GET", "GET"]


async def test_no_event_hooks_is_byte_identical_default() -> None:
    """Omitting event_hooks wires nothing -- the pre-enhancement behaviour."""
    transport, _calls = _sequenced_transport([200])
    async with _client(transport) as client:
        response = await client.request("GET", "/thing")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# stream() -- bulk-download seam (retry on establishment, body not buffered)
# ---------------------------------------------------------------------------


def _streaming_transport(status: int, body: bytes) -> httpx.MockTransport:
    """A transport that returns *body* as a streamable response."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=status, content=body)

    return httpx.MockTransport(handler)


async def test_stream_yields_body_in_chunks() -> None:
    payload = b"x" * (256 * 1024)  # 256 KiB, iterated not buffered by the client
    transport = _streaming_transport(200, payload)
    async with _client(transport) as client, client.stream("GET", "/big.zip") as response:
        assert response.status_code == 200
        assembled = b"".join([chunk async for chunk in response.aiter_bytes()])
    assert assembled == payload


async def test_stream_retries_5xx_establishment_then_succeeds() -> None:
    transport, calls = _sequenced_transport([503, 200])
    async with _client(transport) as client, client.stream("GET", "/flaky") as response:
        body = await response.aread()
    assert response.status_code == 200
    assert calls[0] == 2  # the 503 establishment retried
    assert body == b"body-200"


async def test_stream_all_5xx_raises_upstream_error_with_status() -> None:
    transport, _calls = _sequenced_transport([500, 500, 500])
    with pytest.raises(UpstreamHttpError) as exc_info:
        async with _client(transport) as client:
            async with client.stream("GET", "/down"):
                pass  # pragma: no cover -- establishment never succeeds
    assert exc_info.value.status_code == 500
    assert exc_info.value.body == b"body-500"  # the error body was read before closing


async def test_stream_returns_4xx_un_retried() -> None:
    transport, calls = _sequenced_transport([404])
    async with _client(transport) as client, client.stream("GET", "/missing") as response:
        assert response.status_code == 404
    assert calls[0] == 1  # a 4xx is not retried


async def test_stream_retries_connect_error_then_succeeds() -> None:
    calls = [0]
    real = _streaming_transport(200, b"recovered")

    def handler(request: httpx.Request) -> httpx.Response:
        calls[0] += 1
        if calls[0] == 1:
            raise httpx.ConnectError("boom")
        return real.handler(request)  # type: ignore[attr-defined]

    async with _client(httpx.MockTransport(handler)) as client, client.stream("GET", "/x") as response:
        body = await response.aread()
    assert calls[0] == 2
    assert body == b"recovered"


async def test_stream_passes_range_header() -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Range"))
        return httpx.Response(206, content=b"partial")

    async with _client(httpx.MockTransport(handler)) as client:
        async with client.stream("GET", "/probe", headers={"Range": "bytes=0-0"}) as response:
            assert response.status_code == 206
    assert seen == ["bytes=0-0"]


# ---------------------------------------------------------------------------
# client-level default headers -- an API key / User-Agent an upstream keys on,
# wired once at construction rather than threaded through every call site.
# ---------------------------------------------------------------------------


async def test_client_headers_apply_to_every_request() -> None:
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, text="ok")

    async with _client(httpx.MockTransport(handler), headers={"X-Api-Key": "k", "User-Agent": "faidh/1"}) as client:
        await client.get("/a")
        await client.get("/b")
    assert seen[0]["x-api-key"] == "k"
    assert seen[0]["user-agent"] == "faidh/1"
    assert seen[1]["x-api-key"] == "k"  # not just the first request


async def test_per_call_headers_merge_over_client_defaults() -> None:
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, text="ok")

    async with _client(httpx.MockTransport(handler), headers={"X-Api-Key": "default", "X-Keep": "yes"}) as client:
        await client.get("/a", headers={"X-Api-Key": "override"})
    assert seen[0]["x-api-key"] == "override"  # per-call wins
    assert seen[0]["x-keep"] == "yes"  # untouched client default still present


# ---------------------------------------------------------------------------
# follow_redirects -- client-level default plus per-call override.
# ---------------------------------------------------------------------------


def _redirecting_transport() -> httpx.MockTransport:
    """302 at /start -> 200 at /final, so following is observable by the final body."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/final":
            return httpx.Response(200, text="arrived")
        return httpx.Response(302, headers={"Location": "/final"})

    return httpx.MockTransport(handler)


async def test_does_not_follow_redirects_by_default() -> None:
    async with _client(_redirecting_transport()) as client:
        response = await client.get("/start")
    assert response.status_code == 302  # httpx default, unchanged


async def test_client_level_follow_redirects() -> None:
    async with _client(_redirecting_transport(), follow_redirects=True) as client:
        response = await client.get("/start")
    assert response.status_code == 200
    assert response.text == "arrived"


async def test_per_call_follow_redirects_overrides_client_default() -> None:
    async with _client(_redirecting_transport(), follow_redirects=False) as client:
        response = await client.get("/start", follow_redirects=True)
    assert response.status_code == 200
    assert response.text == "arrived"


# ---------------------------------------------------------------------------
# form-encoded body -- the shape an endpoint that reads a POST form needs.
# ---------------------------------------------------------------------------


async def test_post_sends_form_encoded_data() -> None:
    seen: list[tuple[bytes, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.content, request.headers.get("content-type", "")))
        return httpx.Response(200, text="ok")

    async with _client(httpx.MockTransport(handler)) as client:
        await client.post("/form", data={"searchType": "0", "loc": "TX"})
    body, content_type = seen[0]
    assert b"searchType=0" in body
    assert b"loc=TX" in body
    assert "application/x-www-form-urlencoded" in content_type


# ---------------------------------------------------------------------------
# upstream_base_url=None -- a shared client fronting several endpoints, driven
# by absolute per-request URLs, with no single host to key a breaker on.
# ---------------------------------------------------------------------------


async def test_no_base_url_accepts_absolute_urls() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, text="ok")

    client = TracedHttpClient(
        upstream_base_url=None,
        transport=httpx.MockTransport(handler),
        initial_backoff=0.0,
        max_backoff=0.0,
    )
    async with client:
        r1 = await client.get("https://alpha.example/one")
        r2 = await client.get("https://beta.example/two")
    assert r1.status_code == r2.status_code == 200
    assert seen == ["https://alpha.example/one", "https://beta.example/two"]  # two different hosts, one client


async def test_no_base_url_exhaustion_names_the_actual_host() -> None:
    """With no fixed host, the exhaustion error still names WHERE it failed --
    derived from the request that was actually sent, not a client-wide constant."""
    transport, _calls = _raising_transport(httpx.ConnectError("refused"))
    client = TracedHttpClient(
        upstream_base_url=None, transport=transport, initial_backoff=0.0, max_backoff=0.0, max_attempts=2
    )
    async with client:
        with pytest.raises(UpstreamHttpError) as exc_info:
            await client.get("https://gamma.example/x")
    assert "gamma.example" in str(exc_info.value)


def test_none_base_url_rejects_a_circuit_breaker() -> None:
    """A breaker isolates ONE upstream; with no fixed upstream it cannot key its
    state, so pairing the two is a construction error, not a silent no-op."""
    with pytest.raises(ValueError, match="circuit_breaker"):
        TracedHttpClient(upstream_base_url=None, circuit_breaker=_BreakerSpy())


def test_empty_string_base_url_still_rejected() -> None:
    """The empty string is a mistake (a caller meant a URL); None is the deliberate
    'no fixed upstream' signal. They must not collapse into the same path."""
    with pytest.raises(ValueError, match="non-empty"):
        TracedHttpClient(upstream_base_url="")

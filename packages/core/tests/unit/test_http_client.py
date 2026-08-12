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

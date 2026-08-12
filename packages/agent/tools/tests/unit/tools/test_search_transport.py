"""The host-side transports the search leaf is injected with (search-spec.md §4.4).

Two things are under test and they are different in kind.

:class:`~threetears.agent.tools.search_transport.TracedSearchTransport` is a
bridge, so what matters is that nothing is lost crossing it: the traced
client's status, body, final URL and *attempt count* arrive as a
``TransportResponse``, a caller's per-call bound reaches the wire, and the
egress absence core deliberately reports as ``None`` becomes the named value
the contract requires.

:func:`~threetears.agent.tools.search_transport.build_fetch_transport` is a
choice, not a bridge -- the module docstring's ruling that Extract's half
cannot be the traced client -- so what matters is that what it returns
actually satisfies the protocol that ruling was about.
"""

from __future__ import annotations

import httpx
import pytest

from threetears.agent.tools.search_transport import (
    OFF_BASE_SCOPE,
    TracedSearchTransport,
    build_fetch_transport,
)
from threetears.core.egress import DirectEgress
from threetears.search.contracts import (
    EGRESS_DIRECT,
    FetchTransport,
    LocalCapExceeded,
    SearchFailure,
    SearchTransport,
    TransportFailed,
)
from threetears.search.testing import LocalHttpServer, Reply

_BASE_URL = "http://searxng.internal:8080"

_ARTICLE = b"<html><body><article><p>%s</p></article></body></html>" % (b"the extractable body " * 20)


class _FakeClosingTransport(httpx.AsyncBaseTransport):
    """An httpx transport that actually minds being closed.

    ``httpx.MockTransport`` no-ops on ``aclose()``, so a test built on it
    cannot see a caller-supplied transport being closed out from under the
    caller. This one refuses to serve after close, which is what a real
    pooled transport does.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.closed:
            raise RuntimeError("transport was closed by a previous call")
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        self.closed = True


def _responder(
    *,
    status_code: int = 200,
    body: bytes = b'{"results": []}',
    fail_first: int = 0,
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """a mock transport that records requests and can fail a few times first."""
    seen: list[httpx.Request] = []
    remaining = {"failures": fail_first}

    def _handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if remaining["failures"] > 0:
            remaining["failures"] -= 1
            return httpx.Response(503, content=b"try again")
        return httpx.Response(status_code, content=body, headers={"Content-Type": "application/json"})

    return httpx.MockTransport(_handle), seen


def _transport(**kwargs: object) -> TracedSearchTransport:
    mock, _seen = _responder()
    params: dict[str, object] = {"base_url": _BASE_URL, "http_transport": mock}
    params.update(kwargs)
    return TracedSearchTransport(**params)  # type: ignore[arg-type]


class TestItSatisfiesTheProtocolItIsInjectedAs:
    def test_it_is_a_search_transport(self) -> None:
        assert isinstance(_transport(), SearchTransport)

    def test_a_non_http_base_url_is_refused_at_construction(self) -> None:
        """D21: base URLs are deployment config, and a bad one fails where it is configured."""
        with pytest.raises(ValueError, match="http"):
            TracedSearchTransport(base_url="ftp://searxng.internal")

    def test_a_base_url_without_a_host_is_refused(self) -> None:
        with pytest.raises(ValueError, match="host"):
            TracedSearchTransport(base_url="not-a-url")


class TestNothingIsLostCrossingTheBridge:
    @pytest.mark.asyncio
    async def test_the_response_arrives_as_a_transport_response(self) -> None:
        mock, _seen = _responder(status_code=200, body=b'{"results": [1]}')
        transport = TracedSearchTransport(base_url=_BASE_URL, http_transport=mock)

        response = await transport.request("GET", f"{_BASE_URL}/search")

        assert response.status_code == 200
        assert response.body == b'{"results": [1]}'
        assert response.final_url == f"{_BASE_URL}/search"
        assert response.headers["content-type"] == "application/json"

    @pytest.mark.asyncio
    async def test_the_attempt_count_crosses_rather_than_reading_one(self) -> None:
        """D4 needs the retries visible: a spend that bills per exchange must see all of them.

        The traced client retries a 503 internally and hands back a single
        response, so without the count surfaced a three-exchange call would
        account as one.
        """
        mock, seen = _responder(fail_first=2)
        transport = TracedSearchTransport(
            base_url=_BASE_URL,
            http_transport=mock,
            max_attempts=3,
            initial_backoff=0.0,
            max_backoff=0.0,
        )

        response = await transport.request("GET", f"{_BASE_URL}/search")

        assert len(seen) == 3
        assert response.attempts == 3

    @pytest.mark.asyncio
    async def test_a_single_exchange_reports_one_attempt(self) -> None:
        """non-vacuous: the count is read, not a constant that happens to match."""
        mock, _seen = _responder()
        transport = TracedSearchTransport(base_url=_BASE_URL, http_transport=mock)

        response = await transport.request("GET", f"{_BASE_URL}/search")

        assert response.attempts == 1

    @pytest.mark.asyncio
    async def test_query_parameters_reach_the_wire(self) -> None:
        mock, seen = _responder()
        transport = TracedSearchTransport(base_url=_BASE_URL, http_transport=mock)

        await transport.request("GET", f"{_BASE_URL}/search", params={"q": "otters", "format": "json"})

        assert seen[0].url.params["q"] == "otters"
        assert seen[0].url.params["format"] == "json"

    @pytest.mark.asyncio
    async def test_a_caller_bound_shorter_than_the_configured_one_is_applied(self) -> None:
        """SR-G2: a caller holding a deadline bounds the call to what remains of it."""
        mock, seen = _responder()
        transport = TracedSearchTransport(base_url=_BASE_URL, http_transport=mock, timeout_seconds=30.0)

        await transport.request("GET", f"{_BASE_URL}/search", timeout_seconds=0.25)

        assert seen[0].extensions["timeout"]["read"] == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_saying_nothing_leaves_the_configured_bound_in_place(self) -> None:
        """the override is an override: an unstated deadline must not become no deadline."""
        mock, seen = _responder()
        transport = TracedSearchTransport(base_url=_BASE_URL, http_transport=mock, timeout_seconds=7.0)

        await transport.request("GET", f"{_BASE_URL}/search")

        assert seen[0].extensions["timeout"]["read"] == pytest.approx(7.0)


class TestEgressIsANameNotAnAbsence:
    def test_an_unconfigured_egress_reports_the_default_route_by_name(self) -> None:
        """core distinguishes "nobody chose" from "chose direct"; the contract does not."""
        assert _transport().egress_name == EGRESS_DIRECT

    def test_a_configured_egress_reports_its_own_name(self) -> None:
        transport = _transport(egress=DirectEgress())

        assert transport.egress_name == DirectEgress().name

    @pytest.mark.asyncio
    async def test_the_egress_is_stamped_on_the_response(self) -> None:
        mock, _seen = _responder()
        transport = TracedSearchTransport(base_url=_BASE_URL, http_transport=mock)

        response = await transport.request("GET", f"{_BASE_URL}/search")

        assert response.egress == EGRESS_DIRECT


class TestItWillNotBeAimedSomewhereElse:
    """D21 at the seam that can enforce it: this transport is one upstream's."""

    @pytest.mark.asyncio
    async def test_a_url_on_another_host_is_refused(self) -> None:
        mock, seen = _responder()
        transport = TracedSearchTransport(base_url=_BASE_URL, http_transport=mock)

        with pytest.raises(LocalCapExceeded) as caught:
            await transport.request("GET", "http://elsewhere.test/search")

        assert caught.value.scope == OFF_BASE_SCOPE
        assert not seen

    @pytest.mark.asyncio
    async def test_a_url_on_another_port_is_refused(self) -> None:
        """same host, different service: an SSRF guard that only checked hosts would let it by."""
        mock, seen = _responder()
        transport = TracedSearchTransport(base_url=_BASE_URL, http_transport=mock)

        with pytest.raises(LocalCapExceeded):
            await transport.request("GET", "http://searxng.internal:9999/search")

        assert not seen

    @pytest.mark.asyncio
    async def test_a_path_on_the_configured_base_is_allowed(self) -> None:
        """non-vacuous: the guard refuses elsewhere, not everywhere."""
        mock, seen = _responder()
        transport = TracedSearchTransport(base_url=_BASE_URL, http_transport=mock)

        await transport.request("GET", f"{_BASE_URL}/search")

        assert len(seen) == 1


class TestTheFetchHalfIsTheOtherTransport:
    """the module's ruling, pinned: what it hands back satisfies the fetch protocol."""

    def test_it_is_a_fetch_transport(self) -> None:
        assert isinstance(build_fetch_transport(), FetchTransport)

    def test_it_is_also_a_search_transport(self) -> None:
        """the union Gate A predicted -- so a host needing both can inject one object."""
        assert isinstance(build_fetch_transport(), SearchTransport)

    @pytest.mark.asyncio
    async def test_a_loopback_carrier_is_refused_before_a_socket_opens(self) -> None:
        """these URLs come from search results, which is the case the guard exists for (D21).

        Asserted by driving the guard rather than by reading a flag off the
        object: the flag being set proves nothing about whether anything
        consults it, and the refusal is what the deployment is relying on.
        """
        transport = build_fetch_transport()

        with pytest.raises(TransportFailed, match="non-public"):
            await transport.fetch("GET", "http://127.0.0.1:9/carrier", max_bytes=1024)

    @pytest.mark.asyncio
    async def test_a_deployment_on_its_own_network_can_say_so(self) -> None:
        """non-vacuous: the refusal is the default stance, not an unconditional block.

        With the guard opened the same URL gets past it and fails on the
        connection instead -- which is the machine actually trying to reach a
        port nothing is listening on.
        """
        transport = build_fetch_transport(allow_private_addresses=True, timeout_seconds=0.25)

        with pytest.raises(SearchFailure) as caught:
            await transport.fetch("GET", "http://127.0.0.1:9/carrier", max_bytes=1024)

        assert "non-public" not in str(caught.value)

    def test_the_egress_is_named(self) -> None:
        assert build_fetch_transport().egress_name == EGRESS_DIRECT


class TestTheFetchHalfMeetsTheWebAsItIs:
    """What this builder CHOOSES, driven against a socket rather than read off the object.

    The tests above prove the returned object satisfies the protocol and
    refuses what D21 says it must. None of them ever served it a response, so
    none of them could see that it was built with the leaf's ``max_redirects``
    default of 0 -- right for a search upstream, which is a configured host
    that does not redirect, and wrong for a candidate-derived URL, where
    http->https and www and trailing-slash canonicalisation are most of the
    real web. ``test_standalone.py`` pins the 0 default as correct *for
    search*; the obligation for arbitrary-web fetch is the opposite one, and
    it belongs here, where the host makes the choice.
    """

    @pytest.mark.asyncio
    async def test_a_canonicalising_redirect_is_followed(self) -> None:
        """The regression these tests exist for: a 301 must not read as "no content"."""
        async with LocalHttpServer(
            (
                Reply(status=301, headers={"Location": "/article", "Content-Length": "0"}, body=b""),
                Reply(status=200, body=_ARTICLE, headers={"Content-Type": "text/html"}),
            )
        ) as server:
            transport = build_fetch_transport(allow_private_addresses=True)

            response = await transport.fetch(
                "GET",
                f"{server.base_url}/",
                max_bytes=65536,
                allowed_content_types=("text/html",),
            )

        assert response.status_code == 200
        assert response.body == _ARTICLE
        assert response.final_url.endswith("/article")

    @pytest.mark.asyncio
    async def test_a_redirecting_robots_txt_does_not_refuse_the_page(self) -> None:
        """``extract`` reads robots through this same transport (extract.py).

        A 3xx there is not a 2xx, so a robots file that merely moved read as
        "refused" and the page was never fetched -- the same defect arriving
        by a second route, which is why it gets its own pin.
        """
        async with LocalHttpServer(
            (
                Reply(status=301, headers={"Location": "/robots.txt", "Content-Length": "0"}, body=b""),
                Reply(status=200, body=b"User-agent: *\nAllow: /\n", headers={"Content-Type": "text/plain"}),
            )
        ) as server:
            transport = build_fetch_transport(allow_private_addresses=True)

            response = await transport.fetch(
                "GET",
                f"{server.base_url}/robots.txt",
                max_bytes=4096,
                allowed_content_types=("text/plain",),
            )

        assert response.status_code == 200
        assert b"Allow" in response.body

    @pytest.mark.asyncio
    async def test_a_deployment_can_still_refuse_to_follow(self) -> None:
        """Non-vacuous: following is the host's configured default, not a hard-coded one."""
        async with LocalHttpServer(
            (Reply(status=301, headers={"Location": "/article", "Content-Length": "0"}, body=b""),)
        ) as server:
            transport = build_fetch_transport(allow_private_addresses=True, max_redirects=0)

            response = await transport.fetch("GET", f"{server.base_url}/", max_bytes=65536)

        assert response.status_code == 301

    @pytest.mark.asyncio
    async def test_the_redirect_budget_stays_finite(self) -> None:
        """A loop ends with a response rather than with the process."""
        async with LocalHttpServer(
            (Reply(status=301, headers={"Location": "/loop", "Content-Length": "0"}, body=b""),)
        ) as server:
            transport = build_fetch_transport(allow_private_addresses=True)

            response = await transport.fetch("GET", f"{server.base_url}/loop", max_bytes=65536)

        assert response.status_code == 301


class TestACallerBoundBoundsTheCallNotOneAttempt:
    """SR-G2: the override carries the caller's *remaining deadline*.

    Forwarding it to ``TracedHttpClient.request(timeout=...)`` bounds each
    attempt, and retry lives inside the client -- so a 0.3s deadline could
    fund three 0.3s attempts plus backoff. ``StandaloneTransport._perform``
    bounds the whole call against a deadline, and two implementations of one
    protocol may not mean different things by the same argument.
    """

    @pytest.mark.asyncio
    async def test_a_stalling_upstream_gets_one_attempt_not_three(self) -> None:
        """Counted rather than timed: only one attempt can fit inside the bound."""
        async with LocalHttpServer((Reply(status=200, body=b"{}", delay=2.0),)) as server:
            transport = TracedSearchTransport(
                base_url=server.base_url,
                max_attempts=3,
                initial_backoff=0.01,
                timeout_seconds=30.0,
            )

            # TimeoutError specifically, because that is the type the adapters
            # already map onto ``TimedOut`` (searxng.py, tavily.py) -- a bound
            # that expired must not reach a consumer as a generic transport
            # fault.
            with pytest.raises(TimeoutError):
                await transport.request("GET", f"{server.base_url}/search", timeout_seconds=0.3)

        assert len(server.requests) == 1, f"the bound funded {len(server.requests)} attempts"

    @pytest.mark.asyncio
    async def test_an_unstated_bound_still_leaves_the_configured_one_in_place(self) -> None:
        """Non-vacuous: the whole-call ceiling is the override's doing, not an unconditional one."""
        mock, seen = _responder()
        transport = TracedSearchTransport(base_url=_BASE_URL, http_transport=mock, timeout_seconds=7.0)

        response = await transport.request("GET", f"{_BASE_URL}/search")

        assert response.status_code == 200
        assert seen[0].extensions["timeout"]["read"] == pytest.approx(7.0)


class TestAnInjectedTransportIsNotConsumedByOneCall:
    """The DI seam the constructor advertises must survive being used twice.

    A fresh ``TracedHttpClient`` per call is deliberate (SR-L5), but
    ``AsyncClient.aclose()`` closes the transport it was handed -- so a
    caller-supplied one was closed after call one. ``httpx.MockTransport``
    happens to no-op on close, which is exactly why every existing test
    passed through this.
    """

    @pytest.mark.asyncio
    async def test_two_calls_through_one_injected_transport_both_answer(self) -> None:
        mock, seen = _responder()
        closable = _FakeClosingTransport(mock)
        transport = TracedSearchTransport(base_url=_BASE_URL, http_transport=closable)

        first = await transport.request("GET", f"{_BASE_URL}/search")
        second = await transport.request("GET", f"{_BASE_URL}/search")

        assert first.status_code == 200
        assert second.status_code == 200
        assert len(seen) == 2
        assert not closable.closed, "a caller-supplied transport is the caller's to close"

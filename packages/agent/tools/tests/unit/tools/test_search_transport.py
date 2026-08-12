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

_BASE_URL = "http://searxng.internal:8080"


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

"""The standalone transport, against a real socket.

Every pin here drives httpx for real, because the module exists to carry
obligations httpx's defaults do not: bounded retry, a byte cap, a redirect
policy, an address guard. A mocked transport would only confirm that the mock
agrees with the code.

Loopback is what a test can reach, and loopback is exactly what the address
guard refuses -- so most pins construct the transport with
``allow_private_addresses=True``, and the guard gets its own pins proving the
default refuses.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from threetears.search.adapters.searxng import SearxngAdapter
from threetears.search.bind import bind_search
from threetears.search.contracts import (
    EGRESS_DIRECT,
    LocalCapExceeded,
    SearchRequest,
    SearchTransport,
    TimedOut,
    TransportFailed,
)
from threetears.search.standalone import (
    DEFAULT_MAX_RESPONSE_BYTES,
    RESPONSE_BYTES_SCOPE,
    StandaloneTransport,
)
from _http_server import LocalHttpServer, Reply
from _searxng_payloads import TWO_RESULTS_BODY

#: backoff small enough that a retry pin costs milliseconds rather than seconds.
FAST_BACKOFF = {"initial_backoff": 0.001, "max_backoff": 0.002}


def _transport(**kwargs: object) -> StandaloneTransport:
    """Build a transport that may reach loopback.

    :param kwargs: constructor overrides
    :ptype kwargs: object
    :return: the transport under test
    :rtype: StandaloneTransport
    """
    settings: dict[str, object] = {"allow_private_addresses": True, **FAST_BACKOFF}
    settings.update(kwargs)
    return StandaloneTransport(**settings)  # type: ignore[arg-type]


def test_the_transport_satisfies_the_seam_by_shape() -> None:
    """SR-N1/P9: a host injects it without inheriting anything."""
    assert isinstance(_transport(), SearchTransport)


def test_egress_is_a_named_value_and_configurable() -> None:
    """D20: 'direct' is a name like any other, and a proxy gets its own."""
    assert _transport().egress_name == EGRESS_DIRECT
    assert _transport(egress_name="warp").egress_name == "warp"


@pytest.mark.parametrize(("field", "value"), [("max_attempts", 0), ("max_response_bytes", 0)])
def test_a_degenerate_bound_is_refused_at_construction(field: str, value: int) -> None:
    """A bound that cannot bind fails where it is written."""
    with pytest.raises(ValueError, match=field):
        _transport(**{field: value})


# --- the happy path -------------------------------------------------------


async def test_a_successful_exchange_reports_status_body_and_attempts() -> None:
    """The seam value carries what the layers above account against."""
    async with LocalHttpServer((Reply(body=TWO_RESULTS_BODY),)) as server:
        response = await _transport().request("GET", f"{server.base_url}/search", params={"q": "capybara"})

    assert response.status_code == 200
    assert response.body == TWO_RESULTS_BODY
    assert response.attempts == 1
    assert response.egress == EGRESS_DIRECT
    assert response.elapsed_seconds > 0


async def test_headers_and_params_reach_the_wire() -> None:
    """An adapter's pushdown is only real if the parameter leaves the process."""
    async with LocalHttpServer() as server:
        await _transport().request(
            "GET",
            f"{server.base_url}/search",
            headers={"Accept": "application/json"},
            params={"q": "capybara", "format": "json"},
        )

    head = server.requests[0]
    assert "GET /search?q=capybara&format=json " in head
    assert "accept: application/json" in head.lower()
    assert "user-agent: 3tears-search/standalone" in head.lower()


async def test_response_headers_come_back_lower_cased() -> None:
    """The protocol promises one casing, so the adapter reads one casing."""
    async with LocalHttpServer((Reply(status=429, headers={"Retry-After": "30"}),)) as server:
        response = await _transport().request("GET", f"{server.base_url}/search")

    assert response.headers["retry-after"] == "30"


async def test_a_json_body_is_sent_for_a_post() -> None:
    """The seam takes a JSON body because a provider API may need one."""
    async with LocalHttpServer() as server:
        await _transport().request("POST", f"{server.base_url}/search", json_body={"query": "capybara"})

    assert "POST /search " in server.requests[0]


# --- bounded retry (SR-G4, D4) -------------------------------------------


async def test_a_connection_dropped_mid_request_is_retried() -> None:
    """A transient failure costs a retry rather than the caller's whole query."""
    async with LocalHttpServer((Reply(close_early=True), Reply(body=TWO_RESULTS_BODY))) as server:
        response = await _transport(max_attempts=3).request("GET", f"{server.base_url}/search")

    assert response.status_code == 200
    assert response.attempts == 2, "the attempt count is what budget-follows-the-bill reads (D4)"


async def test_a_5xx_is_retried_and_the_attempts_are_reported() -> None:
    """SR-G4: server errors retry; the count stays visible to spend."""
    async with LocalHttpServer((Reply(status=503, body=b""), Reply(body=TWO_RESULTS_BODY))) as server:
        response = await _transport(max_attempts=3).request("GET", f"{server.base_url}/search")

    assert response.status_code == 200
    assert response.attempts == 2


async def test_a_4xx_is_never_retried() -> None:
    """A 429 hammered is a ban; the adapter raises RateLimited instead."""
    async with LocalHttpServer((Reply(status=429, body=b""),)) as server:
        response = await _transport(max_attempts=3).request("GET", f"{server.base_url}/search")

    assert response.status_code == 429
    assert response.attempts == 1
    assert len(server.requests) == 1


async def test_retries_are_finite_and_the_last_5xx_comes_back() -> None:
    """Forever-retry is wrong for a request: the caller is waiting."""
    async with LocalHttpServer((Reply(status=500, body=b"boom"),)) as server:
        response = await _transport(max_attempts=2).request("GET", f"{server.base_url}/search")

    assert response.status_code == 500
    assert response.attempts == 2
    assert len(server.requests) == 2


async def test_exhausted_attempts_raise_a_typed_failure_with_spend() -> None:
    """SR-E3: even a request that never got an answer says what it consumed."""
    async with LocalHttpServer((Reply(close_early=True),)) as server:
        with pytest.raises(TransportFailed) as raised:
            await _transport(max_attempts=2).request("GET", f"{server.base_url}/search")

    assert "2 attempt(s)" in raised.value.message
    assert raised.value.spend.wall_clock_seconds > 0
    assert raised.value.spend.calls == 0, "the transport bills nothing; the adapter owns the call count"


async def test_a_deadline_that_elapses_raises_a_timeout_not_a_transport_failure() -> None:
    """SR-J1: a timeout is worth retrying later and a connect failure is not."""
    async with LocalHttpServer((Reply(delay=2.0),)) as server:
        with pytest.raises(TimedOut):
            await _transport(max_attempts=1).request("GET", f"{server.base_url}/search", timeout_seconds=0.05)


async def test_the_per_call_timeout_overrides_the_configured_one() -> None:
    """SR-G1/SR-G2: never a constant, and a caller's remaining budget wins."""
    async with LocalHttpServer((Reply(delay=2.0),)) as server:
        with pytest.raises(TimedOut):
            await _transport(timeout_seconds=30.0, max_attempts=1).request(
                "GET", f"{server.base_url}/search", timeout_seconds=0.05
            )


# --- byte caps (SR-G5) ----------------------------------------------------


async def test_a_declared_length_past_the_cap_is_refused_before_reading() -> None:
    """The acute MemoryMax case: nothing unbounded is ever held."""
    oversize = b"x" * 4096
    async with LocalHttpServer((Reply(body=oversize),)) as server:
        with pytest.raises(LocalCapExceeded) as raised:
            await _transport(max_response_bytes=64, max_attempts=1).request("GET", f"{server.base_url}/search")

    assert raised.value.scope == RESPONSE_BYTES_SCOPE
    assert raised.value.remediation is not None


async def test_an_undeclared_length_past_the_cap_is_caught_while_streaming() -> None:
    """A body with no Content-Length still cannot exceed the cap."""
    async with LocalHttpServer((Reply(body=b"y" * 8192, undeclared_length=True),)) as server:
        with pytest.raises(LocalCapExceeded):
            await _transport(max_response_bytes=64, max_attempts=1).request("GET", f"{server.base_url}/search")


async def test_a_cap_refusal_is_not_retried() -> None:
    """Retrying a deterministic refusal only spends the caller's deadline."""
    async with LocalHttpServer((Reply(body=b"z" * 4096),)) as server:
        with pytest.raises(LocalCapExceeded):
            await _transport(max_response_bytes=64, max_attempts=3).request("GET", f"{server.base_url}/search")
        assert len(server.requests) == 1


async def test_a_body_inside_the_cap_is_returned_whole() -> None:
    """The cap bounds; it does not truncate."""
    async with LocalHttpServer((Reply(body=TWO_RESULTS_BODY),)) as server:
        response = await _transport(max_response_bytes=len(TWO_RESULTS_BODY)).request(
            "GET", f"{server.base_url}/search"
        )

    assert response.body == TWO_RESULTS_BODY


def test_the_default_cap_is_generous_but_finite() -> None:
    """A default nobody tunes still has to hold under a MemoryMax cap (SR-L6)."""
    assert 1024 * 1024 <= DEFAULT_MAX_RESPONSE_BYTES <= 64 * 1024 * 1024


# --- SSRF guards (D21, SR-K3, SR-N3) -------------------------------------


async def test_loopback_is_refused_by_default() -> None:
    """A self-hosted base URL is an internal endpoint, and this is the guard."""
    async with LocalHttpServer() as server:
        with pytest.raises(TransportFailed, match="non-public address") as raised:
            await StandaloneTransport(max_attempts=1).request("GET", f"{server.base_url}/search")

    assert "allow_private_addresses" in (raised.value.remediation or "")
    assert server.requests == [], "the refusal happens before a socket is opened"


async def test_a_private_address_is_reachable_only_by_deployment_config() -> None:
    """Deployment config, never a per-call parameter (D21)."""
    async with LocalHttpServer() as server:
        response = await _transport().request("GET", f"{server.base_url}/search")

    assert response.status_code == 200


async def test_a_host_outside_the_allowlist_is_refused() -> None:
    """The strongest available answer to 'never a caller-supplied base URL'."""
    async with LocalHttpServer() as server:
        with pytest.raises(TransportFailed, match="configured for"):
            await _transport(allowed_hosts=("searx.example.org",)).request("GET", f"{server.base_url}/search")


async def test_an_allowlisted_host_is_reachable() -> None:
    """The allowlist is a guard, not a wall."""
    async with LocalHttpServer() as server:
        response = await _transport(allowed_hosts=("127.0.0.1",)).request("GET", f"{server.base_url}/search")

    assert response.status_code == 200


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://example.org/x", "/search", "ftp://example.org"])
async def test_a_non_http_target_is_refused(url: str) -> None:
    """Every non-HTTP scheme is a different way to reach something else."""
    with pytest.raises(TransportFailed, match="absolute http"):
        await _transport(max_attempts=1).request("GET", url)


async def test_an_unresolvable_host_is_a_typed_failure() -> None:
    """A name that answers nothing is a failed request, not a hang."""
    with pytest.raises(TransportFailed):
        await StandaloneTransport(max_attempts=1, **FAST_BACKOFF).request(  # type: ignore[arg-type]
            "GET", "https://searx.invalid.no-such-tld-exists/search"
        )


# --- redirect policy (D21) -----------------------------------------------


async def test_a_redirect_is_not_followed_by_default() -> None:
    """The search endpoint of a healthy instance does not redirect."""
    replies = (Reply(status=302, body=b"", headers={"Location": "/elsewhere"}), Reply())
    async with LocalHttpServer(replies) as server:
        response = await _transport().request("GET", f"{server.base_url}/search")

    assert response.status_code == 302
    assert len(server.requests) == 1


async def test_a_permitted_redirect_is_followed_and_reports_the_final_url() -> None:
    """Where a deployment opts in, provenance still names where it landed."""
    replies = (
        Reply(status=302, body=b"", headers={"Location": "/moved"}),
        Reply(body=TWO_RESULTS_BODY),
    )
    async with LocalHttpServer(replies) as server:
        base_url = server.base_url
        response = await _transport(max_redirects=1).request("GET", f"{base_url}/search")

    assert response.status_code == 200
    assert response.final_url == f"{base_url}/moved"
    assert len(server.requests) == 2


async def test_a_redirect_out_of_http_is_refused() -> None:
    """Every hop is re-guarded, so a hop cannot escape the scheme guard."""
    replies = (Reply(status=302, body=b"", headers={"Location": "gopher://example.org/x"}),)
    async with LocalHttpServer(replies) as server:
        with pytest.raises(TransportFailed, match="absolute http"):
            await _transport(max_redirects=1, max_attempts=1).request("GET", f"{server.base_url}/search")


async def test_a_redirect_off_the_allowlist_is_refused() -> None:
    """The guard binds per hop, not only on the URL the caller supplied."""
    replies = (Reply(status=302, body=b"", headers={"Location": "http://example.org/moved"}),)
    async with LocalHttpServer(replies) as server:
        with pytest.raises(TransportFailed, match="configured for"):
            await _transport(max_redirects=1, max_attempts=1, allowed_hosts=("127.0.0.1",)).request(
                "GET", f"{server.base_url}/search"
            )


async def test_the_redirect_budget_is_finite() -> None:
    """A redirect loop ends with a response rather than with the process."""
    loop = (Reply(status=302, body=b"", headers={"Location": "/again"}),)
    async with LocalHttpServer(loop) as server:
        response = await _transport(max_redirects=2).request("GET", f"{server.base_url}/search")

    assert response.status_code == 302
    assert len(server.requests) == 3


# --- composed with the adapter -------------------------------------------


async def test_the_whole_stack_runs_over_a_real_socket() -> None:
    """Transport, adapter, Call and Bind, end to end, with nothing mocked."""
    async with LocalHttpServer((Reply(body=TWO_RESULTS_BODY),)) as server:
        adapter = SearxngAdapter(base_url=server.base_url, transport=_transport())
        rendered = await bind_search(SearchRequest(query="capybara"), provider=adapter)

    assert rendered.success is True
    assert rendered.content.startswith("1. Capybara")


async def test_a_transport_refusal_arrives_as_a_failed_result() -> None:
    """The transport's typed failures survive the adapter and reach Bind (D10)."""
    async with LocalHttpServer((Reply(body=b"q" * 4096),)) as server:
        adapter = SearxngAdapter(base_url=server.base_url, transport=_transport(max_response_bytes=64))
        rendered = await bind_search(SearchRequest(query="capybara"), provider=adapter)

    assert rendered.success is False
    payload = json.loads(json.dumps(rendered.metadata))["search_results"]
    assert payload["failure"]["failure_class"] == "local-cap-exceeded"
    assert payload["failure"]["scope"] == RESPONSE_BYTES_SCOPE


def test_one_shot_asyncio_run_needs_no_lifecycle() -> None:
    """SR-L5: a single call from a cold start, with nothing left open."""

    async def _once() -> str:
        async with LocalHttpServer((Reply(body=TWO_RESULTS_BODY),)) as server:
            adapter = SearxngAdapter(base_url=server.base_url, transport=_transport())
            rendered = await bind_search(SearchRequest(query="capybara"), provider=adapter)
            return rendered.content

    prose = asyncio.run(_once())
    assert prose.startswith("1. Capybara")

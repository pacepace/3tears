"""The standalone transport, against a real socket.

Every pin here drives httpx for real, because the module exists to carry
obligations httpx's defaults do not: bounded retry, a byte cap, a redirect
policy, an address guard. A mocked transport would only confirm that the mock
agrees with the code.

Loopback is what a test can reach, and loopback is exactly what the address
guard refuses -- so most pins construct the transport with
``allow_private_addresses=True``, and the guard gets its own pins proving the
default refuses.

The whole-call deadline is the one property that cannot be pinned by waiting
for it: proving "three attempts under a 10s bound do not cost 30s" honestly
would cost thirty seconds. So those pins drive the injected clock and sleeper,
the way ``test_limiter.py`` does -- the attempts still run against a real
socket, and only the passage of time is simulated. One pin uses the real clock
anyway, because a bound that only holds against a driven clock has proved
nothing about the shipped path.
"""

from __future__ import annotations

import asyncio
import json
import re
import ssl
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import anyio
import anyio.abc
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from threetears.search.adapters.searxng import SearxngAdapter
from threetears.search.bind import bind_search
from threetears.search.contracts import (
    EGRESS_DIRECT,
    FetchTransport,
    LocalCapExceeded,
    SearchRequest,
    SearchTransport,
    TimedOut,
    TransportFailed,
)
from threetears.search.standalone import (
    CONTENT_TYPE_SCOPE,
    DEFAULT_MAX_RESPONSE_BYTES,
    RESPONSE_BYTES_SCOPE,
    StandaloneTransport,
)
from threetears.search.testing import LocalHttpServer, Reply
from packages.search.tests.searxng_payloads import TWO_RESULTS_BODY

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


class _ManualClock:
    """A monotonic source the test advances, and the sleeper that advances it.

    The same two callables the limiter's tests drive, for the same reason: a
    deadline is only testable in reasonable time if the test owns the passage
    of time. Not a fake of any production protocol, so it declares none.
    """

    def __init__(self, start: float = 1_000.0) -> None:
        """Start at an arbitrary non-zero reading.

        :param start: initial monotonic value
        :ptype start: float
        """
        self.now = start
        self.sleeps: list[float] = []

    def read(self) -> float:
        """Report the current simulated monotonic reading.

        :return: seconds since this clock's arbitrary origin
        :rtype: float
        """
        return self.now

    def advance(self, seconds: float) -> None:
        """Move simulated time forward.

        :param seconds: how far forward
        :ptype seconds: float
        :return: nothing
        :rtype: None
        """
        self.now += seconds

    async def sleep(self, seconds: float) -> None:
        """Record a backoff, advance the clock by it, and yield to the loop.

        :param seconds: what the transport asked to wait
        :ptype seconds: float
        :return: nothing
        :rtype: None
        """
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


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


async def test_a_typed_failure_leaves_stamped_with_egress_and_occurrence_time() -> None:
    """D8/D20: rate/ban budgets key on (provider instance, egress), and only
    this seam knows which egress the failing call left by. The occurrence
    time rides along -- the failure record may be the only surviving fact."""
    async with LocalHttpServer((Reply(close_early=True),)) as server:
        with pytest.raises(TransportFailed) as raised:
            await _transport(max_attempts=1, egress_name="warp").request("GET", f"{server.base_url}/search")

    assert raised.value.egress == "warp"
    assert raised.value.occurred_at is not None
    assert raised.value.occurred_at.tzinfo is not None, "occurrence time is timezone-aware, like all provenance"


async def test_a_guard_refusal_is_stamped_too() -> None:
    """The refusal never opened a socket, but it still names the exit it
    refused to use -- a consumer-side ban tracker reads one shape."""
    async with LocalHttpServer() as server:
        with pytest.raises(TransportFailed) as raised:
            await StandaloneTransport(max_attempts=1, egress_name="warp").request("GET", f"{server.base_url}/search")

    assert raised.value.egress == "warp"
    assert raised.value.occurred_at is not None


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


# --- one deadline for the whole call (SR-G2) ------------------------------


async def test_the_bound_the_caller_stated_bounds_the_whole_call() -> None:
    """Three attempts under a 0.5s bound cost 0.5s, not three times 0.5s plus backoff.

    Real time on purpose: this is the promise a caller under its own deadline
    is relying on, and a driven clock cannot prove the shipped path keeps it.

    **The warm-up call and the scope are load-bearing, not tidying.** Outside a
    scope every attempt builds its own client with ``keepalive=0``, so the
    connect handshake lands *inside* the caller's bound -- and on a contended
    CI runner the handshake alone outruns it. The attempt then dies as a
    ``ConnectTimeout`` having never reached the server, and this test stops
    measuring the thing it names: the bound is spent on connecting rather than
    on the reply it was supposed to wait for. Both ways that can fail were
    observed on one branch within the hour: ``len(server.requests) == 0``
    because no request was ever served, and ``elapsed == 1.12`` against a 0.9
    threshold. Warming a pooled connection first takes the handshake out of the
    measurement, which leaves the read -- the phase the bound is about.

    **The bound is 0.5s for headroom, not for the behaviour.** The defect costs
    three attempts (``FAST_BACKOFF``'s pauses are milliseconds), so the
    threshold has to sit under ``3 x bound`` and correct behaviour costs one
    bound: the slack is ``2 x bound``, and the whole reason to state it larger
    is that the failure above lost ~0.8s to scheduling alone, which 0.3s could
    not absorb. The discrimination stays exact either way -- three 0.5s reads
    cannot come in under 1.5s -- so this buys robustness without buying
    tolerance for the bug.
    """
    async with LocalHttpServer((Reply(body=TWO_RESULTS_BODY), Reply(delay=2.0))) as server:
        transport = _transport(max_attempts=3, timeout_seconds=30.0)
        async with transport.connection_scope():
            await transport.request("GET", f"{server.base_url}/search")

            started = time.monotonic()
            with pytest.raises(TimedOut) as raised:
                await transport.request("GET", f"{server.base_url}/search", timeout_seconds=0.5)
            elapsed = time.monotonic() - started

    assert elapsed < 1.5, "the attempts and the backoffs come out of the one bound, not one copy each"
    assert server.connections == 1, "the timed call reused the warm connection, so it paid no handshake"
    assert len(server.requests) == 2, "the warm-up, then one timed attempt -- the bound funded no second"
    assert "1 attempt(s)" in raised.value.message, "the accounting says what it actually did"


async def test_a_deadline_that_expires_mid_retry_never_starts_another_attempt() -> None:
    """A backoff that would outlast the bound is not taken: the caller hears TimedOut.

    The clock and the sleeper are driven, so the 5s backoff costs the suite
    nothing -- the attempt itself still runs against a real socket.
    """
    clock = _ManualClock()
    async with LocalHttpServer((Reply(close_early=True), Reply(body=TWO_RESULTS_BODY))) as server:
        transport = StandaloneTransport(
            allow_private_addresses=True,
            max_attempts=3,
            initial_backoff=5.0,
            max_backoff=5.0,
            clock=clock.read,
            sleep=clock.sleep,
        )
        with pytest.raises(TimedOut) as raised:
            await transport.request("GET", f"{server.base_url}/search", timeout_seconds=3.0)

    assert clock.sleeps == [], "sleeping out the last of a deadline only reports the same failure later"
    assert len(server.requests) == 1, "the second attempt never started"
    assert "1 attempt(s)" in raised.value.message
    assert "3.000s bound" in raised.value.message, "the message names what ran out"
    assert "timeout" in (raised.value.remediation or "")


async def test_a_retry_the_bound_can_fund_is_still_taken() -> None:
    """The deadline is a bound on retrying, not a replacement for it (SR-G4)."""
    clock = _ManualClock()
    async with LocalHttpServer((Reply(close_early=True), Reply(body=TWO_RESULTS_BODY))) as server:
        transport = StandaloneTransport(
            allow_private_addresses=True,
            max_attempts=3,
            initial_backoff=0.5,
            max_backoff=0.5,
            clock=clock.read,
            sleep=clock.sleep,
        )
        response = await transport.request("GET", f"{server.base_url}/search", timeout_seconds=10.0)

    assert response.status_code == 200
    assert response.attempts == 2
    assert clock.sleeps == [0.5], "one backoff, charged to the bound"
    assert response.elapsed_seconds == 0.5


async def test_a_5xx_the_bound_cannot_fund_a_retry_for_comes_back_as_itself() -> None:
    """An answer the caller can see beats a TimedOut that hides it."""
    clock = _ManualClock()
    async with LocalHttpServer((Reply(status=503, body=b"busy"), Reply(body=TWO_RESULTS_BODY))) as server:
        transport = StandaloneTransport(
            allow_private_addresses=True,
            max_attempts=3,
            initial_backoff=5.0,
            max_backoff=5.0,
            clock=clock.read,
            sleep=clock.sleep,
        )
        response = await transport.request("GET", f"{server.base_url}/search", timeout_seconds=1.0)

    assert response.status_code == 503
    assert response.attempts == 1
    assert clock.sleeps == []
    assert len(server.requests) == 1


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


# --- the fetch seam (FetchTransport, §3.5) --------------------------------


def test_the_transport_satisfies_the_fetch_seam_too() -> None:
    """Gate A predicted the union; this is the module that has to satisfy it."""
    assert isinstance(_transport(), FetchTransport)


async def test_a_fetch_states_its_own_cap() -> None:
    """Extract's cap is the caller's memory budget, not a transport constant."""
    async with LocalHttpServer((Reply(body=b"x" * 4096),)) as server:
        with pytest.raises(LocalCapExceeded, match="this read's"):
            await _transport(max_attempts=1).fetch("GET", f"{server.base_url}/page", max_bytes=64)


async def test_a_per_call_cap_tightens_the_deployment_cap_and_cannot_loosen_it() -> None:
    """A host that declared its memory does not lose that because a caller asked."""
    body = b"x" * 4096
    async with LocalHttpServer((Reply(body=body), Reply(body=body))) as server:
        transport = _transport(max_response_bytes=64, max_attempts=1)
        with pytest.raises(LocalCapExceeded, match="this transport's 64-byte cap"):
            await transport.fetch("GET", f"{server.base_url}/page", max_bytes=1_000_000)
        # ... and the tighter of the two still binds when it is the caller's.
        with pytest.raises(LocalCapExceeded, match="this read's 32-byte cap"):
            await transport.fetch("GET", f"{server.base_url}/page", max_bytes=32)


@pytest.mark.parametrize(("size", "refused"), [(64, False), (65, True)])
async def test_the_cap_is_exact_at_its_own_boundary(size: int, refused: bool) -> None:
    """A cap that is off by one is a cap nobody can reason about.

    Coarse pins -- 4096 bytes against a 64-byte cap -- pass whether the
    comparison is ``>`` or ``>=``, so they prove the cap exists without
    proving where it sits. Exactly-at and one-past are the only two sizes
    that tell those two implementations apart, and the answer has to be the
    same on the declared-length path and the streaming one, which is why
    both run here.
    """
    replies = (Reply(body=b"x" * size), Reply(body=b"x" * size, undeclared_length=True))
    async with LocalHttpServer(replies) as server:
        transport = _transport(max_attempts=1)
        for _ in range(2):
            if refused:
                with pytest.raises(LocalCapExceeded):
                    await transport.fetch("GET", f"{server.base_url}/page", max_bytes=64)
            else:
                response = await transport.fetch("GET", f"{server.base_url}/page", max_bytes=64)
                assert len(response.body) == size


async def test_an_undeclared_length_past_the_fetch_cap_is_caught_while_streaming() -> None:
    """The lying-length path, on the seam that reads arbitrary web content."""
    async with LocalHttpServer((Reply(body=b"y" * 8192, undeclared_length=True),)) as server:
        with pytest.raises(LocalCapExceeded, match="this read's"):
            await _transport(max_attempts=1).fetch("GET", f"{server.base_url}/page", max_bytes=64)


async def test_a_body_inside_the_fetch_cap_comes_back_whole() -> None:
    """The cap bounds; it does not truncate. Same promise request makes."""
    page = b"<html><body>capybara</body></html>"
    async with LocalHttpServer((Reply(body=page, headers={"Content-Type": "text/html"}),)) as server:
        response = await _transport().fetch("GET", f"{server.base_url}/page", max_bytes=len(page))

    assert response.body == page
    assert response.egress == EGRESS_DIRECT


@pytest.mark.parametrize("max_bytes", [0, -1])
def test_a_fetch_cap_that_cannot_bind_is_refused_where_it_is_written(max_bytes: int) -> None:
    """A zero-byte read is a caller defect, not a request worth making."""

    async def _fetch() -> None:
        await _transport().fetch("GET", "http://example.org/page", max_bytes=max_bytes)

    with pytest.raises(ValueError, match="max_bytes"):
        asyncio.run(_fetch())


async def test_the_content_type_gate_refuses_without_reading_the_body() -> None:
    """The point of the gate: a wrong media type costs nothing to discover."""
    async with LocalHttpServer((Reply(body=b"m" * 4096, headers={"Content-Type": "video/mp4"}),)) as server:
        with pytest.raises(LocalCapExceeded) as raised:
            await _transport(max_attempts=1).fetch(
                "GET",
                f"{server.base_url}/clip",
                max_bytes=1_000_000,
                allowed_content_types=("text/html",),
            )

    assert raised.value.scope == CONTENT_TYPE_SCOPE
    # the refusal's own spend is the proof that no body was pulled: the bytes
    # were on the wire and available, and the cap would have allowed them.
    assert raised.value.spend.bytes_transferred == 0


async def test_the_gate_matches_on_the_media_type_not_the_header() -> None:
    """``text/html; charset=utf-8`` is text/html, and parameters are not it."""
    page = b"<html>capybara</html>"
    replies = (Reply(body=page, headers={"Content-Type": "text/html; charset=utf-8"}),)
    async with LocalHttpServer(replies) as server:
        response = await _transport().fetch(
            "GET",
            f"{server.base_url}/page",
            max_bytes=1_000_000,
            allowed_content_types=("text/html", "application/xhtml+xml"),
        )

    assert response.body == page


async def test_a_response_declaring_no_media_type_is_refused_by_the_gate() -> None:
    """Unknown content can be arbitrarily expensive; unknown is not 'what I hoped'."""
    async with LocalHttpServer((Reply(body=b"?" * 512, headers={"Content-Type": ""}),)) as server:
        with pytest.raises(LocalCapExceeded, match="declared no content type"):
            await _transport(max_attempts=1).fetch(
                "GET",
                f"{server.base_url}/mystery",
                max_bytes=1_000_000,
                allowed_content_types=("text/html",),
            )


async def test_the_gate_leaves_an_error_status_visible() -> None:
    """A 404 the caller asked for beats a cap refusal that hides it."""
    replies = (Reply(status=404, body=b"gone", headers={"Content-Type": "text/plain"}),)
    async with LocalHttpServer(replies) as server:
        response = await _transport(max_attempts=1).fetch(
            "GET",
            f"{server.base_url}/missing",
            max_bytes=1_000_000,
            allowed_content_types=("text/html",),
        )

    assert response.status_code == 404


async def test_the_gate_judges_the_response_that_answers_not_the_hops() -> None:
    """A redirect hop's own media type is not the content the caller asked for."""
    page = b"<html>capybara</html>"
    replies = (
        Reply(status=302, body=b"", headers={"Location": "/moved"}),
        Reply(body=page, headers={"Content-Type": "text/html"}),
    )
    async with LocalHttpServer(replies) as server:
        response = await _transport(max_redirects=1).fetch(
            "GET",
            f"{server.base_url}/page",
            max_bytes=1_000_000,
            allowed_content_types=("text/html",),
        )

    assert response.body == page
    assert len(server.requests) == 2


async def test_no_gate_accepts_whatever_the_cap_allows() -> None:
    """The gate is opt-in: a caller that wants bytes says nothing about type."""
    async with LocalHttpServer((Reply(body=b"m" * 512, headers={"Content-Type": "video/mp4"}),)) as server:
        response = await _transport().fetch("GET", f"{server.base_url}/clip", max_bytes=1_000_000)

    assert response.status_code == 200


async def test_the_fetch_seam_inherits_the_address_guard() -> None:
    """A fetched URL is candidate-derived, which is where D21 binds hardest."""
    async with LocalHttpServer((Reply(),)) as server:
        with pytest.raises(TransportFailed, match="non-public address"):
            await StandaloneTransport(max_attempts=1).fetch("GET", f"{server.base_url}/page", max_bytes=1024)


async def test_the_fetch_seam_inherits_the_retry_loop_and_its_accounting() -> None:
    """One loop, two protocols: the attempt count D4 reads is the same number."""
    replies = (Reply(status=503, body=b"nope"), Reply(body=b"ok", headers={"Content-Type": "text/html"}))
    async with LocalHttpServer(replies) as server:
        response = await _transport(max_attempts=2).fetch("GET", f"{server.base_url}/page", max_bytes=1024)

    assert response.status_code == 200
    assert response.attempts == 2


async def test_a_fetch_failure_leaves_stamped_like_any_other() -> None:
    """D8's (provider instance, egress) key is rebuildable from a fetch failure too."""
    async with LocalHttpServer((Reply(body=b"x" * 4096),)) as server:
        with pytest.raises(LocalCapExceeded) as raised:
            await _transport(max_attempts=1).fetch("GET", f"{server.base_url}/page", max_bytes=64)

    assert raised.value.egress == EGRESS_DIRECT
    assert raised.value.occurred_at is not None


# --- the connection scope (spec section 3.8) ------------------------------


async def test_calls_in_a_scope_share_one_connection() -> None:
    """Extract's many-fetch path pays a handshake per host, not per document."""
    page = Reply(body=b"<html>capybara</html>", headers={"Content-Type": "text/html"})
    async with LocalHttpServer((page,)) as server:
        transport = _transport()
        async with transport.connection_scope():
            for _ in range(3):
                await transport.fetch("GET", f"{server.base_url}/page", max_bytes=4096)

    assert len(server.requests) == 3
    assert server.connections == 1


async def test_calls_outside_a_scope_stand_alone() -> None:
    """SR-L5 stays the default: no scope, no pool, nothing left to close."""
    page = Reply(body=b"<html>capybara</html>", headers={"Content-Type": "text/html"})
    async with LocalHttpServer((page,)) as server:
        transport = _transport()
        for _ in range(3):
            await transport.fetch("GET", f"{server.base_url}/page", max_bytes=4096)

    assert server.connections == 3


async def test_a_scope_closes_what_it_opened() -> None:
    """A pool that outlived its block would be the lifecycle SR-L5 refuses."""
    page = Reply(body=b"<html>capybara</html>", headers={"Content-Type": "text/html"})
    async with LocalHttpServer((page,)) as server:
        transport = _transport()
        async with transport.connection_scope():
            await transport.fetch("GET", f"{server.base_url}/page", max_bytes=4096)
        # outside the block the transport is what it was before it: the next
        # call opens its own connection rather than reaching for a closed one.
        await transport.fetch("GET", f"{server.base_url}/page", max_bytes=4096)

    assert server.connections == 2


async def test_a_scope_survives_a_failure_inside_it() -> None:
    """The pool is released by the block ending, however it ends."""
    async with LocalHttpServer((Reply(body=b"x" * 4096),)) as server:
        transport = _transport(max_attempts=1)
        with pytest.raises(LocalCapExceeded):
            async with transport.connection_scope():
                await transport.fetch("GET", f"{server.base_url}/page", max_bytes=64)
        # a scope that leaked its client would hand this call a closed one.
        response = await transport.fetch("GET", f"{server.base_url}/page", max_bytes=1_000_000)

    assert response.status_code == 200


async def test_concurrent_scopes_over_one_transport_do_not_share_a_pool() -> None:
    """A transport is a deployment fact; two callers are not one caller."""
    page = Reply(body=b"<html>capybara</html>", headers={"Content-Type": "text/html"})
    async with LocalHttpServer((page,)) as server:
        transport = _transport()

        async def _scoped() -> None:
            async with transport.connection_scope():
                await transport.fetch("GET", f"{server.base_url}/page", max_bytes=4096)

        await asyncio.gather(_scoped(), _scoped())

    assert len(server.requests) == 2
    assert server.connections == 2


async def test_the_deadline_still_bounds_calls_made_inside_a_scope() -> None:
    """A shared client outlives one call's bound; the bound still binds."""
    clock = _ManualClock()
    async with LocalHttpServer((Reply(close_early=True),)) as server:
        transport = _transport(max_attempts=3, clock=clock.read, sleep=clock.sleep)
        async with transport.connection_scope():
            with pytest.raises((TimedOut, TransportFailed)):
                await transport.fetch("GET", f"{server.base_url}/page", max_bytes=4096, timeout_seconds=5.0)


async def test_a_search_call_shares_the_scope_too() -> None:
    """One loop, one lifecycle: the scope is not a fetch-only affordance."""
    async with LocalHttpServer((Reply(body=TWO_RESULTS_BODY),)) as server:
        transport = _transport()
        async with transport.connection_scope():
            await transport.request("GET", f"{server.base_url}/search")
            await transport.request("GET", f"{server.base_url}/search")

    assert len(server.requests) == 2
    assert server.connections == 1


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


# --- the address guard behind a forward proxy ----------------------------
#
# A deployment whose only way out is a forward proxy (squid at HTTPS_PROXY) has
# no DNS of its own: names resolve at the proxy. httpx already sends the request
# there -- it honours the proxy environment -- but the guard resolved the name
# locally first, as though the connection were direct. Found live in a dev
# container: every fetch failed "cannot resolve host", robots.txt read as
# unknown, unknown rules deny, and the fetch came back ``refused`` while curl
# through the same proxy returned 200.
#
# The guard now checks the route the request will actually take.

_PROXY_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


@pytest.fixture
def no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """A clean proxy environment, so a pin states exactly the route it tests."""
    for name in _PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


async def test_a_proxied_request_is_not_refused_for_a_name_only_the_proxy_can_resolve(
    no_proxy_env: pytest.MonkeyPatch,
) -> None:
    """The name is the proxy's to resolve; the request must reach the proxy."""
    async with LocalHttpServer() as proxy:
        no_proxy_env.setenv("HTTP_PROXY", proxy.base_url)
        response = await StandaloneTransport(max_attempts=1, **FAST_BACKOFF).request(  # type: ignore[arg-type]
            "GET", "http://only-the-proxy-resolves.invalid/page"
        )

    assert response.status_code == 200
    assert proxy.requests, "the request never reached the proxy"


async def test_a_private_address_literal_is_refused_even_through_a_proxy(no_proxy_env: pytest.MonkeyPatch) -> None:
    """A proxy resolves names; it does not make an internal address public."""
    async with LocalHttpServer() as proxy:
        no_proxy_env.setenv("HTTP_PROXY", proxy.base_url)
        with pytest.raises(TransportFailed, match="non-public address"):
            await StandaloneTransport(max_attempts=1).request("GET", "http://10.0.0.5/admin")

    assert proxy.requests == [], "the refusal happens before anything is sent"


@pytest.mark.parametrize("host", ["localhost", "api.localhost"])
async def test_localhost_is_refused_even_through_a_proxy(no_proxy_env: pytest.MonkeyPatch, host: str) -> None:
    """Sent through a proxy, ``localhost`` is the proxy's own loopback."""
    async with LocalHttpServer() as proxy:
        no_proxy_env.setenv("HTTP_PROXY", proxy.base_url)
        with pytest.raises(TransportFailed, match="non-public"):
            await StandaloneTransport(max_attempts=1).request("GET", f"http://{host}/admin")

    assert proxy.requests == []


async def test_a_host_the_proxy_bypass_names_is_still_resolved_and_guarded(no_proxy_env: pytest.MonkeyPatch) -> None:
    """``NO_PROXY`` sends the request direct, so the direct-route guard applies in full."""
    async with LocalHttpServer() as server:
        no_proxy_env.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
        no_proxy_env.setenv("NO_PROXY", "127.0.0.1")
        with pytest.raises(TransportFailed, match="non-public address"):
            await StandaloneTransport(max_attempts=1).request("GET", f"{server.base_url}/search")

    assert server.requests == []


@pytest.mark.parametrize(
    "host",
    ["localhost.", "127.1", "2130706433", "0x7f000001", "127.0.0.1."],
)
async def test_every_spelling_of_loopback_is_refused_through_a_proxy(
    no_proxy_env: pytest.MonkeyPatch, host: str
) -> None:
    """Resolvers accept more than the dotted form, so the guard must too (found by review)."""
    async with LocalHttpServer() as proxy:
        no_proxy_env.setenv("HTTP_PROXY", proxy.base_url)
        with pytest.raises(TransportFailed, match="non-public"):
            await StandaloneTransport(max_attempts=1).request("GET", f"http://{host}/admin")

    assert proxy.requests == [], "the refusal happens before anything is sent"


async def test_a_name_that_resolves_locally_is_checked_even_through_a_proxy(
    no_proxy_env: pytest.MonkeyPatch,
) -> None:
    """Skipping resolution is for a host that CANNOT resolve, not for every proxied request."""

    async def resolves_to_loopback(host: str) -> tuple[str, ...]:
        return ("127.0.0.1",)

    async with LocalHttpServer() as proxy:
        no_proxy_env.setenv("HTTP_PROXY", proxy.base_url)
        with pytest.raises(TransportFailed, match="non-public address"):
            await StandaloneTransport(max_attempts=1, resolve=resolves_to_loopback).request(
                "GET", "http://rebinds-to-loopback.example/admin"
            )

    assert proxy.requests == []


async def test_an_unresolvable_name_on_a_direct_route_still_fails(no_proxy_env: pytest.MonkeyPatch) -> None:
    """The proxy exception must not become a way past resolution for a direct request."""
    with pytest.raises(TransportFailed, match="cannot resolve"):
        await StandaloneTransport(max_attempts=1, **FAST_BACKOFF).request(  # type: ignore[arg-type]
            "GET", "http://only-the-proxy-resolves.invalid/page"
        )


# --- the connection goes where the guard looked (DNS rebinding) -----------
#
# The guard resolves a name and checks every address. If the socket were then
# opened by name, the name would be resolved a second time -- and a name its
# owner controls can answer the guard with a public address and the connect
# with a cluster-internal one. These pins stand in for DNS twice over: the
# injected ``resolve`` is what the guard is told, and ``_Network`` is what a
# connect by name would reach. They disagree, as a rebinding name does, and the
# pins hold that the guard's answer is the one the socket follows.
#
# ``_Network`` replaces ``anyio.connect_tcp``, the call httpcore's network makes
# to open a socket, so the public addresses below are never contacted: each is
# a label for a loopback server, and a host it does not know is refused.

#: addresses the guard accepts, each standing for a server the test controls.
_PUBLIC = "93.184.215.14"
_OTHER_PUBLIC = "8.8.4.4"
_UNREACHABLE_PUBLIC = "1.1.1.1"
_SILENT_PUBLIC = "9.9.9.9"

#: a route that never answers: the connect hangs, as on a path that drops packets.
_BLACKHOLE = -1


def _port(server: LocalHttpServer) -> int:
    """The loopback port ``server`` listens on.

    :param server: a running server
    :ptype server: LocalHttpServer
    :return: its port
    :rtype: int
    """
    port = urlparse(server.base_url).port
    assert port is not None
    return port


class _Network:
    """Where a TCP connect to each host lands, and a record of the hosts asked for."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, routes: dict[str, int]) -> None:
        """Route each host to a loopback port and take over the socket call.

        :param monkeypatch: the test's monkeypatch
        :ptype monkeypatch: pytest.MonkeyPatch
        :param routes: host (a name or an address) to the loopback port it reaches,
            or :data:`_BLACKHOLE` for one whose connect never answers
        :ptype routes: dict[str, int]
        """
        self.routes = routes
        self.connected: list[str] = []
        self._connect = anyio.connect_tcp
        monkeypatch.setattr(anyio, "connect_tcp", self.connect_tcp)

    async def connect_tcp(self, remote_host: str, remote_port: int, **kwargs: object) -> anyio.abc.SocketStream:
        """Connect to the loopback server ``remote_host`` stands for.

        :param remote_host: the host the client asked for
        :ptype remote_host: str
        :param remote_port: the port the client asked for, ignored for the route's own
        :ptype remote_port: int
        :param kwargs: the rest of the connect call, passed through
        :ptype kwargs: object
        :return: the connected stream
        :rtype: anyio.abc.SocketStream
        :raises ConnectionRefusedError: for a host with no route
        """
        self.connected.append(remote_host)
        port = self.routes.get(remote_host)
        if port is None:
            raise ConnectionRefusedError(f"no route to {remote_host}")
        if port == _BLACKHOLE:
            await asyncio.sleep(3600)
        return await self._connect("127.0.0.1", port, **kwargs)  # type: ignore[arg-type]


def _resolver(
    answers: dict[str, tuple[str, ...]], asked: list[str] | None = None
) -> Callable[[str], Awaitable[tuple[str, ...]]]:
    """A resolver giving fixed answers, recording each name it is asked for.

    :param answers: name to the addresses it resolves to
    :ptype answers: dict[str, tuple[str, ...]]
    :param asked: where to record the names asked for, if anywhere
    :ptype asked: list[str] | None
    :return: the resolver
    :rtype: Callable[[str], Awaitable[tuple[str, ...]]]
    """

    async def resolve(host: str) -> tuple[str, ...]:
        if asked is not None:
            asked.append(host)
        if host not in answers:
            raise TransportFailed(f"cannot resolve host {host!r}")
        return answers[host]

    return resolve


async def _resolve_nothing(host: str) -> tuple[str, ...]:
    """A resolver for pins whose host is an address literal, which never reaches it.

    :param host: the name asked for
    :ptype host: str
    :return: never returns
    :rtype: tuple[str, ...]
    :raises AssertionError: always -- a literal must not be looked up
    """
    raise AssertionError(f"an address literal was looked up: {host!r}")


def _guarded(resolve: Callable[[str], Awaitable[tuple[str, ...]]], **kwargs: object) -> StandaloneTransport:
    """Build a transport with the address guard ON, which every pin in this section needs.

    :param resolve: what the guard is told each name resolves to
    :ptype resolve: Callable[[str], Awaitable[tuple[str, ...]]]
    :param kwargs: constructor overrides
    :ptype kwargs: object
    :return: the transport under test
    :rtype: StandaloneTransport
    """
    settings: dict[str, object] = {"max_attempts": 1, "resolve": resolve, **FAST_BACKOFF}
    settings.update(kwargs)
    return StandaloneTransport(**settings)  # type: ignore[arg-type]


async def test_the_connection_goes_to_the_address_the_guard_checked(no_proxy_env: pytest.MonkeyPatch) -> None:
    """A name that rebinds between the guard's lookup and the connect never reaches the second answer."""
    asked: list[str] = []
    async with LocalHttpServer() as checked, LocalHttpServer() as internal:
        network = _Network(no_proxy_env, {_PUBLIC: _port(checked), "rebind.example": _port(internal)})
        transport = _guarded(_resolver({"rebind.example": (_PUBLIC,)}, asked))
        response = await transport.request("GET", "http://rebind.example/admin")

    assert internal.requests == [], "the connect resolved the name again and reached the rebound address"
    assert response.status_code == 200
    assert network.connected == [_PUBLIC]
    assert asked == ["rebind.example"], "one lookup per hop: the guard's, and no other"
    assert "host: rebind.example" in checked.requests[0].lower(), "the Host header still names the host"


async def test_a_redirect_is_resolved_checked_and_pinned_on_its_own(no_proxy_env: pytest.MonkeyPatch) -> None:
    """Each hop names a new host; the last hop's addresses say nothing about it."""
    asked: list[str] = []
    hop = (Reply(status=302, body=b"", headers={"Location": "http://second.example/landed"}),)
    async with LocalHttpServer(hop) as first, LocalHttpServer() as second, LocalHttpServer() as internal:
        network = _Network(
            no_proxy_env,
            {
                _PUBLIC: _port(first),
                _OTHER_PUBLIC: _port(second),
                "first.example": _port(internal),
                "second.example": _port(internal),
            },
        )
        resolve = _resolver({"first.example": (_PUBLIC,), "second.example": (_OTHER_PUBLIC,)}, asked)
        response = await _guarded(resolve, max_redirects=1).request("GET", "http://first.example/start")

    assert internal.requests == []
    assert response.status_code == 200
    assert response.final_url == "http://second.example/landed"
    assert network.connected == [_PUBLIC, _OTHER_PUBLIC]
    assert asked == ["first.example", "second.example"]
    assert "host: second.example" in second.requests[0].lower()


async def test_a_redirect_to_a_name_that_resolves_internally_is_refused(no_proxy_env: pytest.MonkeyPatch) -> None:
    """The second hop's guard is a real guard, not a formality the first hop discharged."""
    hop = (Reply(status=302, body=b"", headers={"Location": "http://second.example/landed"}),)
    async with LocalHttpServer(hop) as first:
        network = _Network(no_proxy_env, {_PUBLIC: _port(first)})
        resolve = _resolver({"first.example": (_PUBLIC,), "second.example": ("10.0.0.7",)})
        with pytest.raises(TransportFailed, match="non-public address"):
            await _guarded(resolve, max_redirects=1).request("GET", "http://first.example/start")

    assert network.connected == [_PUBLIC], "the refused hop opened no socket"


async def test_a_refusing_address_is_tried_past_to_the_next_checked_one(no_proxy_env: pytest.MonkeyPatch) -> None:
    """Pinning keeps the resolver's fallback: an IPv6 answer an IPv4-only pod cannot route is not the end."""
    async with LocalHttpServer() as checked:
        network = _Network(no_proxy_env, {_PUBLIC: _port(checked)})
        resolve = _resolver({"two.example": (_UNREACHABLE_PUBLIC, _PUBLIC)})
        response = await _guarded(resolve).request("GET", "http://two.example/page")

    assert response.status_code == 200
    assert network.connected == [_UNREACHABLE_PUBLIC, _PUBLIC]


async def test_when_every_checked_address_refuses_the_failure_names_them(no_proxy_env: pytest.MonkeyPatch) -> None:
    """Running out of checked addresses fails; it never falls back to the name."""
    network = _Network(no_proxy_env, {"two.example": 1})
    with pytest.raises(TransportFailed, match=re.escape(_UNREACHABLE_PUBLIC)):
        await _guarded(_resolver({"two.example": (_UNREACHABLE_PUBLIC,)})).request("GET", "http://two.example/page")

    assert network.connected == [_UNREACHABLE_PUBLIC]


async def test_the_shared_address_space_is_refused_and_its_neighbours_are_not(no_proxy_env: pytest.MonkeyPatch) -> None:
    """100.64.0.0/10 is neither private nor reserved to ipaddress, and is where some clusters put pods."""
    async with LocalHttpServer() as outside:
        network = _Network(no_proxy_env, {"100.128.0.1": _port(outside)})
        resolve = _resolver({"cgnat.example": ("100.64.12.7",), "beside.example": ("100.128.0.1",)})
        with pytest.raises(TransportFailed, match="non-public address"):
            await _guarded(resolve).request("GET", "http://cgnat.example/")
        response = await _guarded(resolve).request("GET", "http://beside.example/")

    assert response.status_code == 200, "the first address past the /10 is public, and reachable"
    assert network.connected == ["100.128.0.1"], "the refused name opened no socket"


async def test_an_address_that_never_answers_costs_a_moment_not_the_call(no_proxy_env: pytest.MonkeyPatch) -> None:
    """The checked addresses race, as a connect by name does: a silent IPv6 route does not sink a dual-stack site."""
    async with LocalHttpServer() as checked:
        network = _Network(no_proxy_env, {_SILENT_PUBLIC: _BLACKHOLE, _PUBLIC: _port(checked)})
        started = time.monotonic()
        response = await _guarded(_resolver({"dual.example": (_SILENT_PUBLIC, _PUBLIC)})).request(
            "GET", "http://dual.example/page", timeout_seconds=10.0
        )
        took = time.monotonic() - started

    assert response.status_code == 200
    assert network.connected == [_SILENT_PUBLIC, _PUBLIC]
    assert took < 5.0, f"the second address waited on the first for {took:.2f}s"


async def test_the_connect_timeout_bounds_the_whole_race_not_each_address(no_proxy_env: pytest.MonkeyPatch) -> None:
    """Three silent addresses under a 0.7s bound fail in about 0.7s, not the 2.1s a bound per address would take."""
    silent = ("9.9.9.9", "9.9.9.10", "9.9.9.11")
    network = _Network(no_proxy_env, dict.fromkeys(silent, _BLACKHOLE))
    started = time.monotonic()
    with pytest.raises(TimedOut):
        await _guarded(_resolver({"silent.example": silent})).request(
            "GET", "http://silent.example/page", timeout_seconds=0.7
        )
    took = time.monotonic() - started

    assert network.connected == list(silent), "every checked address was started inside the bound"
    assert took < 1.5, f"the bound was spent per address: {took:.2f}s"


async def test_an_address_literal_is_held_to_itself(no_proxy_env: pytest.MonkeyPatch) -> None:
    """An IPv6 literal resolves to itself, is checked, and is the only place the socket goes."""
    async with LocalHttpServer() as checked:
        network = _Network(no_proxy_env, {"2606:4700::1111": _port(checked)})
        response = await _guarded(_resolve_nothing).request("GET", "http://[2606:4700::1111]/page")

    assert response.status_code == 200
    assert network.connected == ["2606:4700::1111"]


async def test_a_host_no_proxy_exempts_goes_direct_and_pinned(no_proxy_env: pytest.MonkeyPatch) -> None:
    """The route the guard judged direct is the route taken: NO_PROXY is read once, by one check."""
    async with LocalHttpServer() as proxy, LocalHttpServer() as checked, LocalHttpServer() as internal:
        network = _Network(
            no_proxy_env, {"proxy.example": _port(proxy), _PUBLIC: _port(checked), "public.example": _port(internal)}
        )
        no_proxy_env.setenv("HTTP_PROXY", "http://proxy.example:3128")
        no_proxy_env.setenv("NO_PROXY", "public.example")
        response = await _guarded(_resolver({"public.example": (_PUBLIC,)})).request(
            "GET", "http://public.example/page"
        )

    assert response.status_code == 200
    assert proxy.requests == [] and internal.requests == []
    assert network.connected == [_PUBLIC]


async def test_requests_queued_for_a_connection_each_connect_where_their_own_guard_looked(
    no_proxy_env: pytest.MonkeyPatch,
) -> None:
    """With one connection slot, later requests wait and connect afterwards -- each still to its own pin."""
    page = Reply(body=b"<html>capybara</html>", headers={"Content-Type": "text/html"}, delay=0.05)
    async with LocalHttpServer((page,)) as one, LocalHttpServer((page,)) as two, LocalHttpServer((page,)) as three:
        addresses = {"one.example": "203.0.114.1", "two.example": "203.0.114.2", "three.example": "203.0.114.3"}
        network = _Network(
            no_proxy_env,
            {
                addresses["one.example"]: _port(one),
                addresses["two.example"]: _port(two),
                addresses["three.example"]: _port(three),
            },
        )
        transport = _guarded(_resolver({name: (address,) for name, address in addresses.items()}), max_connections=1)
        async with transport.connection_scope():
            responses = await asyncio.gather(
                *(transport.fetch("GET", f"http://{name}/page", max_bytes=4096) for name in addresses)
            )

    assert [response.status_code for response in responses] == [200, 200, 200]
    for server, name in ((one, "one.example"), (two, "two.example"), (three, "three.example")):
        assert len(server.requests) == 1 and f"host: {name}" in server.requests[0].lower()
    assert sorted(network.connected) == sorted(addresses.values())


async def test_two_names_on_one_address_never_share_a_connection(no_proxy_env: pytest.MonkeyPatch) -> None:
    """Pooling stays per name, so a TLS connection is only reused for the name it was verified for."""
    page = Reply(body=b"<html>capybara</html>", headers={"Content-Type": "text/html"})
    async with LocalHttpServer((page,)) as shared:
        _Network(no_proxy_env, {_PUBLIC: _port(shared)})
        transport = _guarded(_resolver({"a.example": (_PUBLIC,), "b.example": (_PUBLIC,)}))
        async with transport.connection_scope():
            for host in ("a.example", "b.example", "a.example"):
                await transport.fetch("GET", f"http://{host}/page", max_bytes=4096)

    assert len(shared.requests) == 3
    assert shared.connections == 2


async def test_a_proxied_request_goes_to_the_proxy_not_to_the_checked_address(no_proxy_env: pytest.MonkeyPatch) -> None:
    """Behind a proxy the proxy connects; the guard's route says so and the request follows it."""
    async with LocalHttpServer() as proxy, LocalHttpServer() as checked:
        network = _Network(no_proxy_env, {"proxy.example": _port(proxy), _PUBLIC: _port(checked)})
        no_proxy_env.setenv("HTTP_PROXY", "http://proxy.example:3128")
        response = await _guarded(_resolver({"public.example": (_PUBLIC,)})).request(
            "GET", "http://public.example/page"
        )

    assert response.status_code == 200
    assert checked.requests == []
    assert network.connected == ["proxy.example"]
    assert proxy.requests[0].startswith("GET http://public.example/page ")


# --- TLS against a pinned address ------------------------------------------


def _issue_certificate(directory: Path, name: str) -> tuple[Path, Path, Path]:
    """Write a CA, and a server certificate it signs for ``name``, under ``directory``.

    :param directory: where to write the PEM files
    :ptype directory: Path
    :param name: the DNS name the server certificate is for -- and the only one
    :ptype name: str
    :return: the CA certificate, the server certificate and the server key
    :rtype: tuple[Path, Path, Path]
    """
    now = datetime.now(UTC)

    def _build(
        subject: str, key: ec.EllipticCurvePrivateKey, issuer: str, signer: ec.EllipticCurvePrivateKey, *, ca: bool
    ) -> x509.Certificate:
        builder = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer)]))
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(hours=1))
            .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
        )
        if ca:
            builder = builder.add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            ).add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        else:
            builder = (
                builder.add_extension(x509.SubjectAlternativeName([x509.DNSName(subject)]), critical=False)
                .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
                .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(signer.public_key()), critical=False)
            )
        return builder.sign(signer, hashes.SHA256())

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_cert = _build("pin test CA", ca_key, "pin test CA", ca_key, ca=True)
    server_key = ec.generate_private_key(ec.SECP256R1())
    server_cert = _build(name, server_key, "pin test CA", ca_key, ca=False)
    ca_path, cert_path, key_path = directory / "ca.pem", directory / "server.pem", directory / "server.key"
    ca_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    cert_path.write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        server_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    return ca_path, cert_path, key_path


class _TlsServer:
    """A loopback HTTPS server answering 200, recording the SNI name each handshake presented."""

    def __init__(self, cert: Path, key: Path) -> None:
        """Load the server's certificate.

        :param cert: the server certificate
        :ptype cert: Path
        :param key: its private key
        :ptype key: Path
        """
        self.server_names: list[str | None] = []
        self.requests: list[str] = []
        self._context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        self._context.load_cert_chain(cert, key)
        self._context.sni_callback = self._record_sni
        self._server: asyncio.Server | None = None
        self.port = 0

    def _record_sni(self, _socket: ssl.SSLObject, server_name: str | None, _context: ssl.SSLContext) -> None:
        """Record the name the client asked for in its handshake.

        :param _socket: the connection being handshaken
        :ptype _socket: ssl.SSLObject
        :param server_name: the SNI name, or None when the client sent none
        :ptype server_name: str | None
        :param _context: the server's context
        :ptype _context: ssl.SSLContext
        """
        self.server_names.append(server_name)

    async def __aenter__(self) -> _TlsServer:
        """Listen on an ephemeral loopback port.

        :return: this server
        :rtype: _TlsServer
        """
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0, ssl=self._context)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Stop listening.

        :param exc_info: the exception the block raised, if any
        :ptype exc_info: object
        """
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Answer one request with a short 200 and close.

        :param reader: the connection's reader
        :ptype reader: asyncio.StreamReader
        :param writer: the connection's writer
        :ptype writer: asyncio.StreamWriter
        """
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            self.requests.append(head.decode("latin-1"))
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
            )
            await writer.drain()
        # NOSILENT: a client that failed verification hangs up mid-handshake, which is the case under test
        except asyncio.IncompleteReadError, ConnectionError, ssl.SSLError:
            return
        finally:
            writer.close()


async def test_tls_is_verified_against_the_name_while_the_socket_goes_to_the_pinned_address(
    no_proxy_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pinning moves the socket, not the identity: SNI, certificate check and Host all still say the name."""
    ca, cert, key = _issue_certificate(tmp_path, "secure.example")
    no_proxy_env.setenv("SSL_CERT_FILE", str(ca))
    async with _TlsServer(cert, key) as server, LocalHttpServer() as internal:
        network = _Network(no_proxy_env, {_PUBLIC: server.port, "secure.example": _port(internal)})
        response = await _guarded(_resolver({"secure.example": (_PUBLIC,)})).request("GET", "https://secure.example/")

    assert response.status_code == 200
    assert response.body == b"ok"
    assert network.connected == [_PUBLIC]
    assert server.server_names == ["secure.example"]
    assert "host: secure.example" in server.requests[0].lower()
    assert internal.requests == []


async def test_a_certificate_for_another_name_is_still_refused_at_the_pinned_address(
    no_proxy_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Connecting to an address must not quietly turn hostname verification off."""
    ca, cert, key = _issue_certificate(tmp_path, "someone-else.example")
    no_proxy_env.setenv("SSL_CERT_FILE", str(ca))
    async with _TlsServer(cert, key) as server:
        _Network(no_proxy_env, {_PUBLIC: server.port})
        with pytest.raises(TransportFailed, match="CERTIFICATE_VERIFY_FAILED"):
            await _guarded(_resolver({"secure.example": (_PUBLIC,)})).request("GET", "https://secure.example/")

    assert server.requests == []


class _RecordedStream:
    """the connected stream a winning connect hands back, recording whether it was closed."""

    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


async def test_a_losers_fault_is_raised_only_after_the_winner_is_closed(
    no_proxy_env: pytest.MonkeyPatch,
) -> None:
    """A losing connect that faults instead of stopping is raised, and the winner nobody receives is closed.

    The first address hangs and, when the race abandons it, fails with a fault that is not a connect
    failure; the second connects. The fault reaches the caller -- and the second address's stream,
    which the caller will now never be handed, is closed rather than left open.
    """
    winner = _RecordedStream()

    async def connect_tcp(remote_host: str, remote_port: int, **kwargs: object) -> _RecordedStream:
        if remote_host == _SILENT_PUBLIC:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                raise RuntimeError("a fault in the losing connect") from None
        return winner

    no_proxy_env.setattr(anyio, "connect_tcp", connect_tcp)
    with pytest.raises(Exception) as raised:  # noqa: PT011 -- the fault, however the transport wraps it
        await _guarded(_resolver({"two.example": (_SILENT_PUBLIC, _PUBLIC)})).request("GET", "http://two.example/")

    chain = [raised.value, *(raised.value.__cause__, raised.value.__context__)]
    assert any("a fault in the losing connect" in str(error) for error in chain if error is not None), raised.value
    assert winner.closed, "the winning stream was dropped open"

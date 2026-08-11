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
import time

import pytest

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
    """Three attempts under a 0.3s bound cost 0.3s, not three times 0.3s plus backoff.

    Real time on purpose: this is the promise a caller under its own deadline
    is relying on, and a driven clock cannot prove the shipped path keeps it.
    """
    async with LocalHttpServer((Reply(delay=2.0),)) as server:
        transport = _transport(max_attempts=3, timeout_seconds=30.0)
        started = time.monotonic()
        with pytest.raises(TimedOut) as raised:
            await transport.request("GET", f"{server.base_url}/search", timeout_seconds=0.3)
        elapsed = time.monotonic() - started

    assert elapsed < 0.9, "the attempts and the backoffs come out of the one bound, not one copy each"
    assert len(server.requests) == 1, "the bound was gone after the first attempt, so there was no second"
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

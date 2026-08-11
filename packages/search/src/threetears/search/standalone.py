"""The bare-httpx transport, for hosts that inject nothing -- ``[standalone]``.

Most hosts have ``threetears.core`` and hand this package a thin adapter over
``TracedHttpClient``, which brings timeouts, bounded retry, circuit-breaking
and spans for free. Embedded consumers without core (samsung; anything on a
Pi) have none of that, and the wrong answer for them is a bespoke
``httpx.get`` at each call site. So this module exists: **one** place in the
family, outside the sanctioned traced wrapper, where a raw HTTP client is
opened -- and it carries the same obligations the injected transport does
rather than a subset of them.

That is what makes it a *widening* of the no-bespoke-client norm rather than
an exception to it (D19): the module path is added to
``_SANCTIONED_HTTPX_SITES`` in ``tests/enforcement/test_no_bespoke_reuse.py``
and no exemption is filed, because a sanctioned single-purpose transport
module is precisely what that allowlist is for. The anti-pattern the guard
exists to catch is a *service* holding its own client; this is a transport.

What it owes, and where each obligation is discharged:

- **configurable timeout** (SR-G1) -- :meth:`StandaloneTransport.request`
  takes a per-call override for a caller's remaining deadline (SR-G2), over
  a constructor default. Never a constant at a call site.
- **bounded retry with backoff** (SR-G4) -- finite attempts, exponential
  backoff between two configured bounds. Connect failures, read timeouts and
  5xx are retried; **no 4xx is**, so a 429 comes back for the adapter to
  raise as ``RateLimited`` rather than being hammered.
- **one deadline for the whole call, not one per attempt** (SR-G2) -- the
  timeout a caller states bounds the request, and the attempts and the
  backoff sleeps are all spent out of it. Per-attempt was the wrong reading
  of the same number: three attempts and two backoffs under a 10s timeout
  can hold a caller for half a minute, which is not a bound the caller
  agreed to. So each attempt is given what remains, a backoff that would
  leave no room for the attempt after it is not taken, and a request whose
  deadline ran out mid-retry says so with :class:`TimedOut` and the attempts
  it actually made.
- **per-attempt accounting visible to spend** (D4) --
  :attr:`~threetears.search.contracts.transport.TransportResponse.attempts`
  on success, and the attempt count in the failure message. It has to be
  visible because "budget follows the bill" needs the retry boundary and the
  budget increment on the same side of this seam: an attempt that never
  reached the provider never billed and must not count.
- **SSRF guards** (D21, SR-K3, SR-N3) -- private-address refusal, an
  optional host allowlist, and a redirect policy that defaults to not
  following. Enforced here rather than per call site, because the third call
  site written is the one that skips it.
- **streamed reads with byte caps** (SR-G5) -- the body is consumed in
  chunks against a cap and never through an unbounded ``response.text``,
  which is a memory incident on a ``MemoryMax``-capped host.

**Both transport protocols, one machine.** :meth:`StandaloneTransport.request`
serves search calls and :meth:`StandaloneTransport.fetch` serves Extract's
content reads (the union Gate A predicted this module would satisfy). They
differ in exactly two things -- who states the byte cap, and whether a media
type is refused before the body -- so they share one retry loop, one deadline,
one set of guards. Duplicating the loop to add a cap parameter would have left
two SSRF guards to keep in step, which is the arrangement D21 exists to avoid.

**No client is held between calls.** A search is one request, and SR-L5
requires a single call to work from a one-shot ``asyncio.run()`` with no
lifecycle to manage and nothing left to close. Connection reuse is what the
injected core transport is *for*; a host that needs it should inject that
rather than have this module grow a lifecycle.

**Retry is a loop here rather than ``observe.retry_with_backoff``.** That
function returns a bool and never raises, so it can neither return a
response nor report which failure ended the attempts -- and it counts no
attempts, which is the number D4 needs. The knobs are named after it
(``max_attempts`` / ``initial_backoff`` / ``max_backoff``) and the schedule
matches, so the two agree on vocabulary where they cannot share code.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Final
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import JsonValue

from threetears.observe import get_logger
from threetears.search.contracts import (
    EGRESS_DIRECT,
    LocalCapExceeded,
    SearchFailure,
    Spend,
    TimedOut,
    TransportFailed,
    TransportResponse,
)

__all__ = [
    "CONTENT_TYPE_SCOPE",
    "DEFAULT_INITIAL_BACKOFF_SECONDS",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_BACKOFF_SECONDS",
    "DEFAULT_MAX_CONNECTIONS",
    "DEFAULT_MAX_REDIRECTS",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "MINIMUM_RETRY_SECONDS",
    "RESPONSE_BYTES_SCOPE",
    "StandaloneTransport",
]

_logger = get_logger(__name__)

#: default per-call timeout, in seconds. A default rather than a constant:
#: every call can override it (SR-G1, SR-G2).
DEFAULT_TIMEOUT_SECONDS: Final[float] = 20.0

#: default attempt ceiling. Finite, because forever-retry is wrong for a
#: request -- the caller is waiting (SR-G4).
DEFAULT_MAX_ATTEMPTS: Final[int] = 3

#: first backoff interval, in seconds. Matches the family's sanctioned
#: traced client so the two schedules do not diverge.
DEFAULT_INITIAL_BACKOFF_SECONDS: Final[float] = 0.5

#: backoff ceiling, in seconds.
DEFAULT_MAX_BACKOFF_SECONDS: Final[float] = 8.0

#: how many bytes one response body may occupy. Sized to be safe unturned
#: on the smallest target (SR-L6, SR-G5): a search response is tens of
#: kilobytes, so this is three orders of magnitude of headroom and still
#: nowhere near a ``MemoryMax`` cap.
DEFAULT_MAX_RESPONSE_BYTES: Final[int] = 8 * 1024 * 1024

#: redirects followed by default: none. A redirect is one of the three ways
#: a caller can choose where this process connects (SR-N3), and the search
#: endpoint of a correctly-configured instance does not redirect.
DEFAULT_MAX_REDIRECTS: Final[int] = 0

#: connection ceiling for the per-request client. Bounded so a fan-out
#: cannot open an unbounded pool (SR-H1's default half).
DEFAULT_MAX_CONNECTIONS: Final[int] = 10

#: the scope name a byte-cap refusal reports, so a reader can tell it from a
#: budget refusal without parsing the message.
RESPONSE_BYTES_SCOPE: Final[str] = "response-bytes"

#: the scope name a content-type refusal reports. A second cap identity on
#: the same field, following the ``query-length`` / ``response-bytes``
#: precedent: what refused is machine-readable, never prose to be parsed.
CONTENT_TYPE_SCOPE: Final[str] = "content-type"

#: the least time worth starting another attempt with, in seconds. A retry
#: given less than this cannot resolve a name, connect and read inside it on
#: any real network, so taking it would spend the caller's last milliseconds
#: to arrive at the same timeout by a slower route. Ten milliseconds is far
#: below any real exchange and far above the loop's own overhead, which is
#: what makes it a floor rather than a policy.
MINIMUM_RETRY_SECONDS: Final[float] = 0.01


def _is_blocked_address(address: str) -> bool:
    """Whether ``address`` is one this transport refuses to reach.

    Loopback, private, link-local, reserved, multicast and unspecified
    ranges are all refused: on a host with an internal network, each one
    reaches something a search provider is not.

    :param address: a literal IPv4 or IPv6 address
    :ptype address: str
    :return: whether the address is inside a refused range
    :rtype: bool
    """
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return True
    return (
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_reserved
        or parsed.is_multicast
        or parsed.is_unspecified
    )


def _is_address_literal(host: str) -> bool:
    """Whether ``host`` is already an IP literal rather than a name.

    :param host: the host component of a URL
    :ptype host: str
    :return: whether it parses as an IPv4 or IPv6 address
    :rtype: bool
    """
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


async def _resolve(host: str) -> tuple[str, ...]:
    """Resolve ``host`` to its addresses, off the event loop.

    Resolution runs in a thread because ``getaddrinfo`` blocks, and blocking
    the loop inside a guard would make the guard a latency defect (SR-G3).

    :param host: a hostname or an IP literal
    :ptype host: str
    :return: every address the host resolves to
    :rtype: tuple[str, ...]
    :raises TransportFailed: when the name does not resolve -- an
        unresolvable host is a failed request, not a reachable one
    """
    if _is_address_literal(host):
        return (host,)
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None, 0, socket.SOCK_STREAM)
    except OSError as exc:
        raise TransportFailed(f"cannot resolve host {host!r}: {exc}", spend=Spend()) from exc
    return tuple({str(info[4][0]) for info in infos})


class StandaloneTransport:
    """A bounded, guarded HTTP transport over bare httpx.

    Satisfies :class:`~threetears.search.contracts.transport.SearchTransport`
    structurally. Construct one per deployment configuration -- the egress
    name, the guards and the bounds are all deployment facts, and D21 puts
    them here rather than at a call site so no call site can skip them.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
        max_backoff: float = DEFAULT_MAX_BACKOFF_SECONDS,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        egress_name: str = EGRESS_DIRECT,
        allow_private_addresses: bool = False,
        allowed_hosts: Sequence[str] = (),
        verify_tls: bool = True,
        user_agent: str = "3tears-search/standalone",
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Configure the transport from deployment facts.

        :param timeout_seconds: default per-call timeout (SR-G1)
        :ptype timeout_seconds: float
        :param max_attempts: attempt ceiling, at least 1 (SR-G4)
        :ptype max_attempts: int
        :param initial_backoff: first backoff interval, in seconds
        :ptype initial_backoff: float
        :param max_backoff: backoff ceiling, in seconds
        :ptype max_backoff: float
        :param max_response_bytes: cap on one response body (SR-G5)
        :ptype max_response_bytes: int
        :param max_redirects: redirect hops permitted; 0 refuses to follow
            any. Every hop is re-guarded, and a downgrade from https to
            http is refused whatever this is set to (D21)
        :ptype max_redirects: int
        :param max_connections: ceiling on concurrent connections
        :ptype max_connections: int
        :param egress_name: which exit this transport's requests leave by
            (D20). ``direct`` is a name like any other, never an absence
        :ptype egress_name: str
        :param allow_private_addresses: whether a private, loopback or
            reserved address may be reached. False by default; a
            deployment whose search instance is genuinely on its own
            network sets this True as **deployment config**, never per call
        :ptype allow_private_addresses: bool
        :param allowed_hosts: when non-empty, the only hostnames this
            transport will reach. The strongest available answer to
            "MUST NOT accept a caller-supplied base URL" at the seam that
            can actually enforce it (D21)
        :ptype allowed_hosts: Sequence[str]
        :param verify_tls: whether to verify TLS certificates
        :ptype verify_tls: bool
        :param user_agent: value sent as ``User-Agent``
        :ptype user_agent: str
        :param clock: monotonic seconds source, read for the whole-call
            deadline and for the elapsed figures on spend. Injectable so a
            test can drive the deadline rather than wait it out, on the
            precedent :class:`~threetears.search.limiter.InProcessRateLimiter`
            set; a wall clock does not belong here, because a clock step
            backwards would extend a deadline that had passed
        :ptype clock: Callable[[], float]
        :param sleep: how a backoff waits. ``asyncio.sleep`` by default, and
            any replacement must yield to the loop rather than block it.
            Injected alongside ``clock`` for the reason the limiter states:
            a test that drives the clock must also own the sleeping, or a
            backoff would wait in real time against a clock that never moves
        :ptype sleep: Callable[[float], Awaitable[None]]
        :raises ValueError: when ``max_attempts`` is below 1 or
            ``max_response_bytes`` is not positive
        """
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        if max_response_bytes < 1:
            raise ValueError(f"max_response_bytes must be positive, got {max_response_bytes}")
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._max_response_bytes = max_response_bytes
        self._max_redirects = max_redirects
        self._max_connections = max_connections
        self._egress_name = egress_name
        self._allow_private_addresses = allow_private_addresses
        self._allowed_hosts = frozenset(host.lower() for host in allowed_hosts)
        self._verify_tls = verify_tls
        self._user_agent = user_agent
        self._clock = clock
        self._sleep = sleep

    @property
    def egress_name(self) -> str:
        """Name of the egress this transport's requests leave by (D20).

        :return: the configured exit's name
        :rtype: str
        """
        return self._egress_name

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, JsonValue] | None = None,
        timeout_seconds: float | None = None,
    ) -> TransportResponse:
        """Perform one bounded, guarded exchange.

        :param method: HTTP method
        :ptype method: str
        :param url: absolute URL, built by the caller from a
            deployment-configured base (D21)
        :ptype url: str
        :param headers: request headers, if any
        :ptype headers: Mapping[str, str] | None
        :param params: query parameters, if any
        :ptype params: Mapping[str, str] | None
        :param json_body: JSON request body, if any
        :ptype json_body: Mapping[str, JsonValue] | None
        :param timeout_seconds: per-call override of the configured timeout.
            A bound on the **whole call** (SR-G2): the attempts and the
            backoff sleeps are all spent out of it, never one copy of it
            each
        :ptype timeout_seconds: float | None
        :return: the completed exchange, with the attempt count on it
        :rtype: TransportResponse
        :raises threetears.search.contracts.errors.TimedOut: when the
            attempts timed out, or when the whole-call bound ran out before
            another attempt could be made
        :raises threetears.search.contracts.errors.TransportFailed: on a
            refused address, a refused redirect, or a connect/protocol
            failure that outlasted the attempts
        :raises threetears.search.contracts.errors.LocalCapExceeded: when a
            response body exceeds the configured cap (SR-G5)
        """
        return await self._perform(
            method,
            url,
            headers=headers,
            params=params,
            json_body=json_body,
            max_bytes=self._max_response_bytes,
            allowed_content_types=None,
            timeout_seconds=timeout_seconds,
        )

    async def fetch(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        max_bytes: int,
        allowed_content_types: tuple[str, ...] | None = None,
        timeout_seconds: float | None = None,
    ) -> TransportResponse:
        """Perform one bounded, guarded, byte-capped content read.

        Satisfies
        :class:`~threetears.search.contracts.transport.FetchTransport`. Every
        obligation :meth:`request` discharges holds here unchanged -- and the
        guards bind hardest on this method, because a fetched URL is
        *candidate-derived*: it came from a provider's answer rather than from
        deployment config, so it is the one URL in this package a caller can
        influence without touching a config file.

        **A per-call cap can tighten the deployment cap, never loosen it.**
        The effective cap is the lesser of ``max_bytes`` and the transport's
        configured ``max_response_bytes``. The per-call number is the caller's
        memory budget (the protocol's point); the configured one is the host's,
        and a host that declared how much memory this process may hold does not
        lose that because a caller asked for more. A refusal names the cap that
        actually bound, so neither is a mystery.

        :param method: HTTP method; ``GET`` or ``HEAD``
        :ptype method: str
        :param url: absolute URL to fetch
        :ptype url: str
        :param headers: request headers, if any
        :ptype headers: Mapping[str, str] | None
        :param max_bytes: this call's cap on the body (SR-G5), tightening the
            configured cap where it is lower
        :ptype max_bytes: int
        :param allowed_content_types: media-type prefixes this read accepts;
            None accepts anything. A successful response declaring a type
            outside the gate -- or declaring none at all -- is refused before
            its body is read, which is the point of the gate: never pull
            megabytes of video to learn it was not text
        :ptype allowed_content_types: tuple[str, ...] | None
        :param timeout_seconds: per-call override of the configured timeout,
            bounding the **whole call** including retries and backoff (SR-G2)
        :ptype timeout_seconds: float | None
        :return: the completed exchange; ``body`` never exceeds the effective
            cap
        :rtype: TransportResponse
        :raises ValueError: when ``max_bytes`` is not positive
        :raises threetears.search.contracts.errors.TimedOut: when the attempts
            timed out, or the whole-call bound ran out before another fit
        :raises threetears.search.contracts.errors.TransportFailed: on a
            refused address, a refused redirect, or a connect/protocol failure
            that outlasted the attempts
        :raises threetears.search.contracts.errors.LocalCapExceeded: when the
            body exceeds the effective cap, or the content-type gate refuses
        """
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        return await self._perform(
            method,
            url,
            headers=headers,
            params=None,
            json_body=None,
            max_bytes=min(max_bytes, self._max_response_bytes),
            allowed_content_types=allowed_content_types,
            timeout_seconds=timeout_seconds,
        )

    async def _perform(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None,
        params: Mapping[str, str] | None,
        json_body: Mapping[str, JsonValue] | None,
        max_bytes: int,
        allowed_content_types: tuple[str, ...] | None,
        timeout_seconds: float | None,
    ) -> TransportResponse:
        """Run the bounded, guarded, retrying exchange both protocols share.

        :param method: HTTP method
        :ptype method: str
        :param url: absolute URL to reach
        :ptype url: str
        :param headers: request headers, if any
        :ptype headers: Mapping[str, str] | None
        :param params: query parameters, if any
        :ptype params: Mapping[str, str] | None
        :param json_body: JSON request body, if any
        :ptype json_body: Mapping[str, JsonValue] | None
        :param max_bytes: effective cap on one response body
        :ptype max_bytes: int
        :param allowed_content_types: media-type prefixes a successful
            response must match, or None to accept any
        :ptype allowed_content_types: tuple[str, ...] | None
        :param timeout_seconds: per-call override of the configured timeout
        :ptype timeout_seconds: float | None
        :return: the completed exchange, with the attempt count on it
        :rtype: TransportResponse
        :raises threetears.search.contracts.errors.SearchFailure: the typed
            failures both public methods document
        """
        effective_timeout = timeout_seconds if timeout_seconds is not None else self._timeout_seconds
        started = self._clock()
        deadline = started + effective_timeout
        request_headers = {"User-Agent": self._user_agent, **dict(headers or {})}
        attempt = 0
        bytes_seen = 0
        last: BaseException | None = None
        bound_spent = False
        limits = httpx.Limits(max_connections=self._max_connections, max_keepalive_connections=0)
        while attempt < self._max_attempts:
            attempt += 1
            try:
                async with httpx.AsyncClient(
                    # What remains of the caller's bound, never the whole of
                    # it again: attempt two under a 10s bound that attempt
                    # one spent 6s of gets 4s, so the request cannot outlive
                    # the deadline whatever the attempt ceiling is.
                    timeout=httpx.Timeout(max(0.0, min(effective_timeout, deadline - self._clock()))),
                    follow_redirects=False,
                    limits=limits,
                    verify=self._verify_tls,
                ) as client:
                    status, body, final_url, read, response_headers = await self._exchange(
                        client,
                        method,
                        url,
                        headers=request_headers,
                        params=params,
                        json_body=json_body,
                        max_bytes=max_bytes,
                        allowed_content_types=allowed_content_types,
                        elapsed_so_far=self._clock() - started,
                        bytes_so_far=bytes_seen,
                    )
                bytes_seen += read
                if status >= 500 and attempt < self._max_attempts:
                    _logger.warning(
                        "standalone transport got HTTP %d from %s, retrying (attempt %d of %d)",
                        status,
                        url,
                        attempt,
                        self._max_attempts,
                    )
                    if await self._wait_to_retry(attempt, deadline):
                        continue
                    # The bound will not fund another attempt -- and we are
                    # holding a real answer. A 5xx the caller can see beats a
                    # TimedOut that hides it, so the retry stops here and the
                    # response goes back with the attempts it took.
                    _logger.warning(
                        "standalone transport stopped retrying %s after %d attempt(s): the caller's bound is spent",
                        url,
                        attempt,
                    )
                return TransportResponse(
                    status_code=status,
                    body=body,
                    final_url=final_url,
                    egress=self._egress_name,
                    elapsed_seconds=self._clock() - started,
                    attempts=attempt,
                    headers=response_headers,
                )
            except SearchFailure as failure:
                # already typed and already terminal: a refused address, a
                # refused redirect, or a body past the cap. Retrying a
                # deterministic refusal only spends the caller's deadline;
                # it leaves stamped with what only this transport knows.
                stamped = self._stamped(failure)
                if stamped is failure:
                    raise
                raise stamped from failure
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                if attempt >= self._max_attempts:
                    break
                _logger.warning(
                    "standalone transport attempt %d of %d to %s failed (%s), backing off",
                    attempt,
                    self._max_attempts,
                    url,
                    type(exc).__name__,
                )
                if not await self._wait_to_retry(attempt, deadline):
                    bound_spent = True
                    break
            except httpx.HTTPError as exc:
                last = exc
                break
        exhausted = self._exhausted(
            url,
            attempt,
            last,
            elapsed=self._clock() - started,
            bytes_seen=bytes_seen,
            bound=effective_timeout,
            bound_spent=bound_spent,
        )
        raise self._stamped(exhausted)

    async def _wait_to_retry(self, attempt: int, deadline: float) -> bool:
        """Back off before the next attempt, where the bound still allows one.

        The backoff is part of the caller's bound rather than an addition to
        it (SR-G2), so this both caps the wait and refuses to take one that
        would leave no attempt on the other side of it: sleeping out the last
        of a deadline to then report a timeout is the same failure, later.

        :param attempt: the attempt that just failed, 1-based
        :ptype attempt: int
        :param deadline: the monotonic reading the whole call must end by
        :ptype deadline: float
        :return: whether the backoff was taken and another attempt fits
        :rtype: bool
        """
        pause = self._backoff(attempt)
        if deadline - self._clock() - pause < MINIMUM_RETRY_SECONDS:
            return False
        await self._sleep(pause)
        return True

    def _stamped(self, failure: SearchFailure) -> SearchFailure:
        """Fill the transport facts every typed failure must leave with.

        Which egress the failing call left by is this transport's
        configuration (D20), and rate/ban budgets key on it together with
        the provider instance (D8) -- so the failure record has to carry it
        out, and only this seam knows it. The occurrence time rides along
        for the same reason: the record may be the only surviving fact.

        :param failure: the failure about to leave this transport
        :ptype failure: SearchFailure
        :return: the same failure class, carrying egress and occurrence time
        :rtype: SearchFailure
        """
        updates: dict[str, object] = {}
        if failure.egress is None:
            updates["egress"] = self._egress_name
        if failure.occurred_at is None:
            updates["occurred_at"] = datetime.now(UTC)
        if not updates:
            return failure
        return failure.to_record().model_copy(update=updates).to_failure()

    def _backoff(self, attempt: int) -> float:
        """Seconds to wait before the attempt after ``attempt``.

        Deterministic exponential growth between the configured bounds,
        matching the family's traced client. No jitter: a single request has
        one caller waiting on it, and the fleet-decorrelation jitter buys is
        a startup-loop concern rather than a per-request one.

        :param attempt: the attempt that just failed, 1-based
        :ptype attempt: int
        :return: the backoff interval in seconds
        :rtype: float
        """
        return min(self._initial_backoff * (1 << (attempt - 1)), self._max_backoff)

    def _exhausted(
        self,
        url: str,
        attempts: int,
        last: BaseException | None,
        *,
        elapsed: float,
        bytes_seen: int,
        bound: float,
        bound_spent: bool = False,
    ) -> SearchFailure:
        """Build the typed failure for a request that ran out of attempts.

        A timeout and a connect failure are different answers to the caller
        -- one is worth retrying later, the other is not (SR-J1) -- so the
        last exception decides the class rather than everything collapsing
        into one. A request that ran out of *deadline* rather than out of
        attempts is a timeout whatever ended its last attempt: the caller
        stated a bound and the bound is what stopped this, and saying so is
        what makes the attempt count readable rather than mysterious.

        :param url: the URL that was being reached
        :ptype url: str
        :param attempts: attempts made
        :ptype attempts: int
        :param last: the exception that ended the attempts, if any
        :ptype last: BaseException | None
        :param elapsed: wall-clock across every attempt, in seconds
        :ptype elapsed: float
        :param bytes_seen: bytes read across every attempt
        :ptype bytes_seen: int
        :param bound: the whole-call bound this request was given, in
            seconds, named in the message so the reader can see what ran out
        :ptype bound: float
        :param bound_spent: whether the bound, rather than the attempt
            ceiling, is what ended the retrying
        :ptype bound_spent: bool
        :return: the typed failure, carrying what the attempts consumed
        :rtype: SearchFailure
        """
        # calls is 0: the transport cannot know what a provider charges for,
        # and an attempt that never completed billed nothing anyway (D4). The
        # adapter owns the money and the call count.
        spend = Spend(wall_clock_seconds=elapsed, calls=0, bytes_transferred=bytes_seen)
        detail = f"{type(last).__name__}: {last}" if last is not None else "no attempt completed"
        if bound_spent:
            return TimedOut(
                f"request to {url} spent its {bound:.3f}s bound after {attempts} attempt(s): {detail}",
                spend=spend,
                remediation=(
                    "raise this call's timeout, or lower max_attempts/initial_backoff so the attempts fit "
                    "inside it -- the bound covers the whole call, retries and backoff included"
                ),
            )
        message = f"request to {url} failed after {attempts} attempt(s): {detail}"
        if isinstance(last, httpx.TimeoutException):
            return TimedOut(message, spend=spend)
        return TransportFailed(message, spend=spend)

    async def _exchange(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, str] | None,
        json_body: Mapping[str, JsonValue] | None,
        max_bytes: int,
        allowed_content_types: tuple[str, ...] | None,
        elapsed_so_far: float,
        bytes_so_far: int,
    ) -> tuple[int, bytes, str, int, dict[str, str]]:
        """Send one request, following guarded redirects, reading under a cap.

        :param client: the per-request client
        :ptype client: httpx.AsyncClient
        :param method: HTTP method
        :ptype method: str
        :param url: absolute URL to reach
        :ptype url: str
        :param headers: request headers
        :ptype headers: Mapping[str, str]
        :param params: query parameters, if any
        :ptype params: Mapping[str, str] | None
        :param json_body: JSON request body, if any
        :ptype json_body: Mapping[str, JsonValue] | None
        :param max_bytes: effective cap on one response body
        :ptype max_bytes: int
        :param allowed_content_types: media-type prefixes a successful
            response must match, or None to accept any
        :ptype allowed_content_types: tuple[str, ...] | None
        :param elapsed_so_far: wall-clock already spent, for failure spend
        :ptype elapsed_so_far: float
        :param bytes_so_far: bytes already read, for failure spend
        :ptype bytes_so_far: int
        :return: status, body bytes, the URL that answered, bytes read,
            and the response headers with lower-cased keys
        :rtype: tuple[int, bytes, str, int, dict[str, str]]
        :raises threetears.search.contracts.errors.TransportFailed: on a
            refused address or a refused redirect
        :raises threetears.search.contracts.errors.LocalCapExceeded: when
            the body exceeds the cap, or the content-type gate refuses
        """
        target = url
        read_total = 0
        for hop in range(self._max_redirects + 1):
            await self._guard(target, elapsed=elapsed_so_far, bytes_seen=bytes_so_far + read_total)
            async with client.stream(
                method,
                target,
                headers=dict(headers),
                params=dict(params) if params else None,
                json=dict(json_body) if json_body else None,
            ) as response:
                # header keys are normalised here so every consumer of the
                # seam reads one casing -- the protocol promises lower-case,
                # and httpx answers case-insensitively rather than
                # case-normalised.
                response_headers = {name.lower(): value for name, value in response.headers.items()}
                location = response_headers.get("location")
                following = 300 <= response.status_code < 400 and bool(location) and hop < self._max_redirects
                if not following:
                    self._gate_content_type(
                        response_headers,
                        target,
                        allowed_content_types,
                        status_code=response.status_code,
                        elapsed=elapsed_so_far,
                        bytes_seen=bytes_so_far + read_total,
                    )
                body, read = await self._read_capped(
                    response,
                    target,
                    max_bytes=max_bytes,
                    elapsed=elapsed_so_far,
                    bytes_seen=bytes_so_far + read_total,
                )
            read_total += read
            if not following:
                return response.status_code, body, target, read_total, response_headers
            target = self._guarded_redirect(target, location or "")
        # unreachable: the loop returns or raises on its last iteration, and
        # ``range`` is non-empty because max_redirects is never negative.
        raise TransportFailed(
            f"redirect handling for {url} ended without a response",
            spend=Spend(wall_clock_seconds=elapsed_so_far, bytes_transferred=bytes_so_far + read_total),
        )

    async def _read_capped(
        self,
        response: httpx.Response,
        url: str,
        *,
        max_bytes: int,
        elapsed: float,
        bytes_seen: int,
    ) -> tuple[bytes, int]:
        """Read a streamed body against the byte cap (SR-G5).

        A declared ``Content-Length`` past the cap is refused before a byte
        is read; an undeclared or lying one is caught chunk by chunk. Either
        way nothing unbounded is ever held, which is the whole requirement.

        :param response: the open streaming response
        :ptype response: httpx.Response
        :param url: the URL being read, for the failure message
        :ptype url: str
        :param max_bytes: the cap this read must stay under
        :ptype max_bytes: int
        :param elapsed: wall-clock already spent, for failure spend
        :ptype elapsed: float
        :param bytes_seen: bytes already read, for failure spend
        :ptype bytes_seen: int
        :return: the body bytes and how many were read
        :rtype: tuple[bytes, int]
        :raises threetears.search.contracts.errors.LocalCapExceeded: when
            the body exceeds the cap
        """
        declared = response.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > max_bytes:
            raise self._over_cap(url, int(declared), cap=max_bytes, elapsed=elapsed, bytes_seen=bytes_seen)
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise self._over_cap(url, total, cap=max_bytes, elapsed=elapsed, bytes_seen=bytes_seen + total)
            chunks.append(chunk)
        return b"".join(chunks), total

    def _gate_content_type(
        self,
        headers: Mapping[str, str],
        url: str,
        allowed: tuple[str, ...] | None,
        *,
        status_code: int,
        elapsed: float,
        bytes_seen: int,
    ) -> None:
        """Refuse a media type this read will not accept, before the body.

        Called with the response's headers in hand and the body still
        unread, which is the whole value of the gate: a caller after HTML
        learns a URL is a 400MB video without pulling any of it (§3.5).

        Two deliberate narrowings, both about not lying to the caller:

        - **Only successful responses are gated.** An error status is an
          answer the caller asked for and is entitled to see; refusing a 404
          because its error page was ``text/plain`` would replace a true
          status with a misleading cap refusal. Error bodies stay bounded by
          the byte cap either way, so nothing unbounded is held.
        - **A missing type is refused**, not accepted. The gate exists
          because unknown content can be arbitrarily expensive; a response
          declining to say what it is has to be treated as unknown rather
          than assumed to be what the caller hoped for.

        :param headers: the response headers, lower-cased keys
        :ptype headers: Mapping[str, str]
        :param url: the URL that answered, for the failure message
        :ptype url: str
        :param allowed: media-type prefixes accepted, or None to accept any
        :ptype allowed: tuple[str, ...] | None
        :param status_code: the response's status
        :ptype status_code: int
        :param elapsed: wall-clock already spent, for failure spend
        :ptype elapsed: float
        :param bytes_seen: bytes already read, for failure spend
        :ptype bytes_seen: int
        :raises threetears.search.contracts.errors.LocalCapExceeded: when
            the declared media type matches no accepted prefix
        """
        if not allowed or not (200 <= status_code < 300):
            return
        declared = headers.get("content-type", "")
        media_type = declared.split(";", 1)[0].strip().lower()
        if media_type and any(media_type.startswith(prefix.lower()) for prefix in allowed):
            return
        described = f"declared content type {media_type!r}" if media_type else "declared no content type"
        raise LocalCapExceeded(
            f"response from {url} {described}, which this read does not accept: {list(allowed)}",
            spend=Spend(wall_clock_seconds=elapsed, calls=0, bytes_transferred=bytes_seen),
            remediation=(
                "widen allowed_content_types if this media type is wanted, or leave it None to accept "
                "anything the byte cap allows -- the gate refuses before the body precisely so a wrong "
                "media type costs nothing to discover"
            ),
            scope=CONTENT_TYPE_SCOPE,
        )

    def _over_cap(self, url: str, size: int, *, cap: int, elapsed: float, bytes_seen: int) -> LocalCapExceeded:
        """Build the refusal for a response past the byte cap.

        ``LocalCapExceeded`` rather than a transport failure, because that is
        what happened: a locally-configured cap refused, and D5 reserves
        this class for a cap bounding the run's *shape* as distinct from a
        provider bounding money. ``scope`` names which cap, so a reader never
        has to tell a byte cap from a budget by reading prose.

        :param url: the URL that answered too much
        :ptype url: str
        :param size: bytes seen or declared
        :ptype size: int
        :param cap: the cap that actually bound this read -- the transport's
            for a search call, the lesser of the two for a fetch. Named
            rather than assumed, so a caller who passed ``max_bytes`` and
            was bound by the deployment ceiling can see which number
            refused
        :ptype cap: int
        :param elapsed: wall-clock spent, in seconds
        :ptype elapsed: float
        :param bytes_seen: bytes read across the whole request
        :ptype bytes_seen: int
        :return: the typed refusal
        :rtype: LocalCapExceeded
        """
        source = "this transport's" if cap == self._max_response_bytes else "this read's"
        return LocalCapExceeded(
            f"response from {url} is {size} bytes, past {source} {cap}-byte cap",
            spend=Spend(wall_clock_seconds=elapsed, calls=0, bytes_transferred=bytes_seen),
            remediation=(
                "raise max_response_bytes on the transport if the host can afford the memory, raise this "
                "call's max_bytes where that is the lower of the two, or narrow the request so less comes back"
            ),
            scope=RESPONSE_BYTES_SCOPE,
        )

    async def _guard(self, url: str, *, elapsed: float, bytes_seen: int) -> None:
        """Refuse a target this transport must not reach (D21, SR-N3).

        Two guards, both deployment-configured: an optional host allowlist,
        which is the strongest available answer to "never a caller-supplied
        base URL" at the only seam that can enforce it; and a private-address
        refusal covering every address the name resolves to, since a name
        resolving to one public and one loopback address is the interesting
        case rather than a hypothetical one.

        Honest about its limit: httpx resolves the name again when it
        connects, so a name that changes answers between the two can still
        slip through. Closing that needs the connection pinned to the
        address that was checked, which is a client-internals concern. This
        raises the bar; it does not seal the door, and a deployment that
        needs sealing puts the instance behind an allowlisted host.

        :param url: the absolute URL about to be reached
        :ptype url: str
        :param elapsed: wall-clock spent, for the failure's spend
        :ptype elapsed: float
        :param bytes_seen: bytes read, for the failure's spend
        :ptype bytes_seen: int
        :raises threetears.search.contracts.errors.TransportFailed: when the
            scheme, host or resolved addresses are refused
        """
        spend = Spend(wall_clock_seconds=elapsed, calls=0, bytes_transferred=bytes_seen)
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise TransportFailed(f"refusing to reach {url!r}: not an absolute http(s) URL", spend=spend)
        host = parsed.hostname.lower()
        if self._allowed_hosts and host not in self._allowed_hosts:
            raise TransportFailed(
                f"refusing to reach host {host!r}: this transport is configured for {sorted(self._allowed_hosts)}",
                spend=spend,
                remediation="add the host to allowed_hosts if the deployment intends it to be reachable",
            )
        if self._allow_private_addresses:
            return
        addresses = await _resolve(host)
        blocked = sorted(address for address in addresses if _is_blocked_address(address))
        if blocked:
            raise TransportFailed(
                f"refusing to reach host {host!r}: it resolves to non-public address(es) {blocked}",
                spend=spend,
                remediation=(
                    "a search instance genuinely on this host's own network is reached by constructing the "
                    "transport with allow_private_addresses=True -- deployment config, never a per-call "
                    "parameter (D21)"
                ),
            )

    def _guarded_redirect(self, current: str, location: str) -> str:
        """Resolve a redirect target, refusing the ones policy forbids.

        :param current: the URL that answered with the redirect
        :ptype current: str
        :param location: the ``Location`` header's value
        :ptype location: str
        :return: the absolute URL of the next hop
        :rtype: str
        :raises threetears.search.contracts.errors.TransportFailed: when the
            hop would downgrade https to http
        """
        target = urljoin(current, location)
        if urlparse(current).scheme == "https" and urlparse(target).scheme != "https":
            raise TransportFailed(
                f"refusing redirect from {current} to {target}: a downgrade out of TLS is never followed",
                spend=Spend(),
            )
        return target

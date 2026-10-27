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
  site written is the one that skips it. The address check and the connection
  are one decision: a direct connection is opened to an address the guard
  resolved and checked, never to a second resolution of the name, so a name
  that answers differently by the time the socket opens (DNS rebinding) has
  nothing left to steer.
- **streamed reads with byte caps** (SR-G5) -- the body is consumed in
  chunks against a cap and never through an unbounded ``response.text``,
  which is a memory incident on a ``MemoryMax``-capped host.
- **circuit-breaking per upstream** (SR-D3) -- one breaker per ``host[:port]``
  a call reaches, in the three-call shape the family's breakers share (admit,
  then record success or failure) and with the model breaker's states:
  ``breaker_failure_threshold`` consecutive failed calls open it, it refuses
  for ``breaker_cooldown_seconds`` -- before any socket opens, as
  :class:`TransportFailed` -- and then admits one probe. Keyed by upstream
  rather than held per client because this transport reaches whatever host a
  call names, so the upstream is the one thing an outage is a property of. A
  call is judged as a whole, retries inside it included; a refusal of this
  transport's own (guard, cap) is not the upstream's failure.
- **a per-call span** (SR-I1) -- ``threetears.observe.traced`` around each of
  :meth:`StandaloneTransport.request`, :meth:`StandaloneTransport.fetch` and
  :meth:`StandaloneTransport.stream`, carrying the upstream, the egress, the
  answering status and the attempt count, and whether a breaker refused.
  Zero-cost where OpenTelemetry is absent, which is ``traced``'s own contract.
  A stream's span covers opening it, to the head; the body read after belongs
  to its caller's span.

**Who actually injects this, as of Phase 2 item 5.** The paragraph above is
still right about *search*: a host with core hands the SearXNG or Tavily
adapter a thin ``TracedHttpClient`` adapter, and does. It turned out to be
wrong about *fetch*, and the reason is structural rather than a gap somebody
could close. ``TracedHttpClient`` is constructed per upstream -- one
``upstream_base_url``, one circuit breaker guarding it -- and reads whole
bodies. Extract fetches a different host on every call, chosen by whatever a
provider returned, under a per-call byte cap, refusing on content type before
the body. There is no upstream to construct a client for and no way to cap a
buffered read. So ``3tears-agent-tools`` -- a host that very much has core --
injects :meth:`StandaloneTransport.fetch` for Extract while keeping the
traced adapter for search. That is not a host re-implementing the sanctioned
client, which is what D19 forbids; it is a host reaching for the family's one
transport that can read an arbitrary URL under a cap, which is what this
module is.

**Three transport protocols, one machine.** :meth:`StandaloneTransport.request`
serves search calls, :meth:`StandaloneTransport.fetch` serves Extract's
content reads (the union Gate A predicted this module would satisfy), and
:meth:`StandaloneTransport.stream` serves a provider whose answer arrives as an
open-ended stream (AD11). Request and fetch differ in exactly two things -- who
states the byte cap, and whether a media type is refused before the body; a
stream differs in where an attempt ends, at the head rather than after the
body. So all three share one retry loop, one deadline, one set of guards and
one connect pinning, and only what an attempt *is* varies. Duplicating the
loop for each would have left three SSRF guards to keep in step, which is the
arrangement D21 exists to avoid.

**No client is held between calls, and one is held inside a scope.** A search
is one request, and SR-L5 requires a single call to work from a one-shot
``asyncio.run()`` with no lifecycle to manage and nothing left to close -- so
that stays the default, and a caller who never asks for anything else never
acquires a thing to close. But §3.8 flagged the other half of this when it
declared the fetch seam: a per-request client is right for one search and
wrong for Extract's many-fetch path, where ten candidate pages plus a
robots.txt per host means twenty fresh TCP and TLS handshakes to read twenty
documents. :meth:`StandaloneTransport.connection_scope` is the opt-in --
inside the block, the calls share one client and its connection pool; outside
it, nothing changed.

The scope's client lives in a module-level ``ContextVar`` rather than on the
transport, so one transport instance stays reentrant: two concurrent scopes
over the same transport get their own pools instead of racing over an
attribute, and a scope that exits closes only what it opened. The lifecycle
belongs to whoever *constructed* the transport, which is the host -- an
injected port does not get to decide how long its own connections live, and
Extract therefore asks for no scope of its own.

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
import threading
import time
import warnings
import urllib.request
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from types import TracebackType
from typing import Final, Protocol, Self
from urllib.parse import urljoin, urlparse

import httpcore
import httpx
from pydantic import JsonValue

from threetears.observe import contain, get_logger, set_span_attribute, traced
from threetears.search.contracts import (
    EGRESS_DIRECT,
    LocalCapExceeded,
    SearchFailure,
    Spend,
    StreamedResponse,
    TimedOut,
    TransportFailed,
    TransportResponse,
    media_type_accepted,
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
    "DEFAULT_TRANSPORT_USER_AGENT",
    "MINIMUM_RETRY_SECONDS",
    "RESPONSE_BYTES_SCOPE",
    "StandaloneTransport",
]

_logger = get_logger(__name__)

#: default per-call timeout, in seconds. A default rather than a constant:
#: every call can override it (SR-G1, SR-G2).
DEFAULT_TIMEOUT_SECONDS: Final[float] = 20.0

#: The ``User-Agent`` sent when a deployment names none. A deployment that fetches
#: pages passes its own name, which robots rules are written against.
DEFAULT_TRANSPORT_USER_AGENT: Final[str] = "3tears-search/standalone"

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

#: clients shared by an open :meth:`StandaloneTransport.connection_scope`,
#: keyed by the identity of the transport that opened the scope.
#:
#: A ``ContextVar`` rather than an attribute on the transport, for the reason
#: the module docstring gives: a transport instance is a deployment fact that
#: may serve several concurrent callers, and an attribute would make the last
#: scope to open the one everybody uses. Keyed by ``id`` because the key must
#: hold for exactly as long as the scope does, and the scope holds a reference
#: to the transport throughout -- so the identity cannot be recycled underneath
#: it. The mapping is replaced rather than mutated on entry so a nested or
#: concurrent scope inherits a snapshot instead of sharing a dict.
_SCOPED_CLIENTS: ContextVar[Mapping[int, httpx.AsyncClient]] = ContextVar("_SCOPED_CLIENTS", default={})

#: consecutive failed calls to one upstream that open its breaker, when the
#: deployment states no other. A default, overridable per transport.
_DEFAULT_BREAKER_FAILURE_THRESHOLD: Final[int] = 5

#: how long an open breaker refuses before admitting a probe, in seconds, when
#: the deployment states no other. Matches the family's model breaker.
_DEFAULT_BREAKER_COOLDOWN_SECONDS: Final[float] = 30.0

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
    reaches something a search provider is not. So is anything else IANA does
    not list as globally reachable -- the shared address space
    ``100.64.0.0/10`` above all, which is neither private nor reserved to
    :mod:`ipaddress` and is where some clusters put their pods and some
    overlay networks their nodes.

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
        not parsed.is_global
        or parsed.is_private
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


def _proxy_for(scheme: str, host: str) -> str | None:
    """The forward proxy a request to ``host`` leaves through, or ``None`` for a direct one.

    Read from the proxy environment (``HTTPS_PROXY`` / ``HTTP_PROXY`` /
    ``ALL_PROXY``, with ``NO_PROXY`` exempting hosts), per request. This is the
    one answer to "which route does this request take": the guard checks the
    route it names and :class:`_GuardedRoutes` then sends the request down that
    same route, so the route checked and the route taken cannot disagree.

    :param scheme: the URL's scheme
    :ptype scheme: str
    :param host: the URL's host, lower-cased
    :ptype host: str
    :return: the proxy URL, with ``http://`` supplied where the variable named
        only a host and port, or ``None`` when the connection is direct
    :rtype: str | None
    """
    proxies = urllib.request.getproxies_environment()
    proxy = proxies.get(scheme) or proxies.get("all")
    if not proxy:
        return None
    # getattr: present at runtime on every supported Python, absent from the type stubs.
    bypass = getattr(urllib.request, "proxy_bypass_environment")
    if bypass(host, proxies):
        return None
    return proxy if "://" in proxy else f"http://{proxy}"


def _as_address(host: str) -> str | None:
    """``host`` as a canonical IP address when it is one in ANY accepted spelling, else ``None``.

    Resolvers accept more than dotted quads: ``127.1``, ``2130706433`` and ``0x7f000001`` all reach
    loopback. A guard that only recognises the dotted form lets the others through.

    :param host: a URL host, lower-cased and without a trailing dot
    :ptype host: str
    :return: the canonical address, or ``None`` for a name
    :rtype: str | None
    """
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        # NOSILENT: not a standard literal; the IPv4 shorthand spellings are tried next.
        pass
    try:
        return socket.inet_ntoa(socket.inet_aton(host))
    except OSError:
        # NOSILENT: not an IPv4 shorthand either, so it is a name -- the documented None.
        return None


def _is_local_name(host: str) -> bool:
    """Whether ``host`` names the machine it is resolved on, whatever DNS says.

    Sent through a proxy, ``localhost`` is the *proxy's* loopback -- an internal
    endpoint the local resolver never gets asked about.

    :param host: a hostname, lower-cased
    :ptype host: str
    :return: ``True`` for ``localhost`` and its subdomains
    :rtype: bool
    """
    return host == "localhost" or host.endswith(".localhost")


async def _resolve(host: str) -> tuple[str, ...]:
    """Resolve ``host`` to its addresses, off the event loop.

    Resolution runs in a thread because ``getaddrinfo`` blocks, and blocking
    the loop inside a guard would make the guard a latency defect (SR-G3).

    :param host: a hostname or an IP literal
    :ptype host: str
    :return: every address the host resolves to, in the resolver's order of
        preference, each once. The order is kept because a direct connection
        tries these addresses in it: the system resolver sorts an address this
        host cannot route (IPv6 on an IPv4-only pod) after one it can
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
    return tuple(dict.fromkeys(str(info[4][0]) for info in infos))


#: the request extension :meth:`StandaloneTransport._exchange` puts the guard's
#: route on, and :class:`_GuardedRoutes` reads it from. Namespaced so it cannot
#: collide with an extension httpx or httpcore defines.
_ROUTE_EXTENSION: Final[str] = "threetears.search.route"


@dataclass(frozen=True, slots=True)
class _Pinned:
    """A direct connection, to one of the addresses the guard checked and to nothing else."""

    #: the checked addresses, in the order a connection tries them. Never
    #: empty: an empty pin would be a route with nowhere to go.
    addresses: tuple[str, ...]

    def __post_init__(self) -> None:
        """Refuse a pin with no address in it.

        :raises ValueError: when ``addresses`` is empty
        """
        if not self.addresses:
            raise ValueError("a pinned route needs at least one checked address")


@dataclass(frozen=True, slots=True)
class _Unpinned:
    """A direct connection by name, for a transport with no address policy to protect.

    Only a transport built with ``allow_private_addresses=True`` routes this
    way: it checks no address, so there is no checked address to hold the
    connection to, and connecting by name changes nothing it relies on.
    """


@dataclass(frozen=True, slots=True)
class _Proxied:
    """A connection to a forward proxy, which reaches the target on this request's behalf."""

    #: the proxy's URL, scheme included.
    proxy_url: str


#: where one request goes. The guard decides it, and the connection follows it.
type _Route = _Pinned | _Unpinned | _Proxied


@dataclass(frozen=True, slots=True)
class _ConnectPermit:
    """What one direct request may open a TCP connection to.

    Set by :class:`_GuardedRoutes` for the length of one request and read by
    :class:`_PinnedNetwork` when httpcore asks it for a socket. httpcore opens a
    request's connection inside that request's own task (HTTP/1.1 never hands a
    connection that is still opening to a second request), so a context
    variable reaches exactly the connect it is meant for.
    """

    #: the host httpcore will ask to connect to, as it will spell it.
    host: str
    #: the route that request was given.
    route: _Pinned | _Unpinned


_CONNECT_PERMIT: ContextVar[_ConnectPermit | None] = ContextVar("_CONNECT_PERMIT", default=None)

#: httpcore's errors and the httpx errors they become, most specific first, so
#: the retry loop sees exactly the classes httpx's own transport would raise.
_HTTPCORE_TO_HTTPX: Final[tuple[tuple[type[Exception], type[httpx.HTTPError]], ...]] = (
    (httpcore.ConnectTimeout, httpx.ConnectTimeout),
    (httpcore.ReadTimeout, httpx.ReadTimeout),
    (httpcore.WriteTimeout, httpx.WriteTimeout),
    (httpcore.PoolTimeout, httpx.PoolTimeout),
    (httpcore.TimeoutException, httpx.TimeoutException),
    (httpcore.ConnectError, httpx.ConnectError),
    (httpcore.ReadError, httpx.ReadError),
    (httpcore.WriteError, httpx.WriteError),
    (httpcore.NetworkError, httpx.NetworkError),
    (httpcore.ProxyError, httpx.ProxyError),
    (httpcore.UnsupportedProtocol, httpx.UnsupportedProtocol),
    (httpcore.LocalProtocolError, httpx.LocalProtocolError),
    (httpcore.RemoteProtocolError, httpx.RemoteProtocolError),
    (httpcore.ProtocolError, httpx.ProtocolError),
)

#: every httpcore error the table translates.
_HTTPCORE_ERRORS: Final[tuple[type[Exception], ...]] = tuple(core_class for core_class, _ in _HTTPCORE_TO_HTTPX)


@contextmanager
def _as_httpx_errors() -> Iterator[None]:
    """Raise an httpcore error as the httpx error it corresponds to.

    :return: a context manager translating the errors raised inside it
    :rtype: Iterator[None]
    :raises httpx.HTTPError: the translation of an httpcore error
    """
    try:
        yield
    except _HTTPCORE_ERRORS as exc:
        translated = next(httpx_class for core_class, httpx_class in _HTTPCORE_TO_HTTPX if isinstance(exc, core_class))
        raise translated(str(exc)) from exc


#: how long a pinned connect waits on one address before starting the next
#: alongside it, in seconds. RFC 8305's recommended connection attempt delay,
#: and anyio's default for the race a connect by name runs.
_CONNECTION_ATTEMPT_DELAY_SECONDS: Final[float] = 0.25


async def _abandon(attempt: asyncio.Task[httpcore.AsyncNetworkStream]) -> BaseException | None:
    """Stop a connect attempt that lost the race, closing what it opened if it got that far.

    An attempt that failed to connect lost anyway, so its :class:`httpcore.ConnectError`
    is the expected end; any other failure is a fault, returned rather than raised so the
    race can stop every other attempt, and close its winner, before it raises it.

    :param attempt: the losing attempt
    :ptype attempt: asyncio.Task[httpcore.AsyncNetworkStream]
    :return: whatever the attempt raised other than a connect failure, else ``None``
    :rtype: BaseException | None
    """
    attempt.cancel()
    await asyncio.wait([attempt])
    fault: BaseException | None = None
    if not attempt.cancelled():
        error = attempt.exception()
        if error is None:
            await attempt.result().aclose()
        elif not isinstance(error, httpcore.ConnectError):
            fault = error
    return fault


class _PinnedNetwork(httpcore.AsyncNetworkBackend):
    """httpcore's network, with every TCP connect held to the addresses its request was given.

    This is where the guard's answer becomes the connection's destination.
    httpcore hands :meth:`connect_tcp` the URL's *host*; left to itself it
    would resolve that name again, and a name that answered the guard with a
    public address can answer the second lookup with a cluster-internal one.
    So for a pinned request the name is never looked up here at all: the
    socket goes to a checked address, while TLS still verifies the
    certificate against the name (httpcore takes the server name from the
    URL, not from this socket) and the ``Host`` header still carries it.
    """

    def __init__(self) -> None:
        """Wrap httpcore's own asyncio network."""
        self._network = httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        """Open a TCP connection to where this request's route allows.

        A pinned route races its checked addresses the way a connect by name
        does (RFC 8305's staggered start): the first address is tried, the
        next is started when that one fails or after
        :data:`_CONNECTION_ATTEMPT_DELAY_SECONDS` without an answer, and the
        first connection made wins while the rest are cancelled. So an address
        this host cannot reach -- a refused one, or an IPv6 one whose route
        drops packets -- costs a quarter-second, not the whole call. The
        connect timeout bounds the race as a whole, never each address.

        :param host: the host httpcore was asked to reach
        :ptype host: str
        :param port: the port to connect to
        :ptype port: int
        :param timeout: connect timeout for the whole race, in seconds
        :ptype timeout: float | None
        :param local_address: local address to bind, if any
        :ptype local_address: str | None
        :param socket_options: socket options to set, if any
        :ptype socket_options: Iterable[httpcore.SOCKET_OPTION] | None
        :return: the connected stream
        :rtype: httpcore.AsyncNetworkStream
        :raises RuntimeError: when no route was given for this host -- a
            connection the guard never saw is refused rather than made
        :raises httpcore.ConnectTimeout: when no checked address connected
            inside the timeout
        :raises httpcore.ConnectError: when every checked address failed
        """
        permit = _CONNECT_PERMIT.get()
        if permit is None or permit.host != host:
            raise RuntimeError(f"refusing a connection to {host!r} that no guarded request asked for")
        if isinstance(permit.route, _Unpinned):
            return await self._network.connect_tcp(
                host, port, timeout=timeout, local_address=local_address, socket_options=socket_options
            )
        try:
            async with asyncio.timeout(timeout):
                return await self._race(host, permit.route.addresses, port, local_address, socket_options)
        except TimeoutError as exc:
            raise httpcore.ConnectTimeout(
                f"no checked address for {host!r} connected within {timeout}s: {list(permit.route.addresses)}"
            ) from exc

    async def _race(
        self,
        host: str,
        addresses: tuple[str, ...],
        port: int,
        local_address: str | None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None,
    ) -> httpcore.AsyncNetworkStream:
        """Connect to the first of ``addresses`` that answers, starting each in turn.

        :param host: the host the addresses were checked for, for the failure message
        :ptype host: str
        :param addresses: the checked addresses, in the resolver's order
        :ptype addresses: tuple[str, ...]
        :param port: the port to connect to
        :ptype port: int
        :param local_address: local address to bind, if any
        :ptype local_address: str | None
        :param socket_options: socket options to set, if any
        :ptype socket_options: Iterable[httpcore.SOCKET_OPTION] | None
        :return: the first connected stream
        :rtype: httpcore.AsyncNetworkStream
        :raises httpcore.ConnectError: when every address failed
        """
        waiting = list(addresses)
        attempts: dict[asyncio.Task[httpcore.AsyncNetworkStream], str] = {}
        failures: list[str] = []
        won: httpcore.AsyncNetworkStream | None = None
        try:
            while waiting or attempts:
                if waiting:
                    address = waiting.pop(0)
                    attempt = asyncio.create_task(
                        self._network.connect_tcp(
                            address, port, local_address=local_address, socket_options=socket_options
                        )
                    )
                    attempts[attempt] = address
                done, _ = await asyncio.wait(
                    attempts,
                    timeout=_CONNECTION_ATTEMPT_DELAY_SECONDS if waiting else None,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for attempt in done:
                    address = attempts.pop(attempt)
                    error = attempt.exception()
                    if error is None:
                        won = attempt.result()
                        return won
                    if not isinstance(error, httpcore.ConnectError):
                        raise error
                    failures.append(f"{address}: {error}")
        finally:
            # the losers, on every way out: a win, a failure, or the caller's
            # timeout cancelling this race.
            faults = [fault for fault in [await _abandon(attempt) for attempt in attempts] if fault is not None]
            if faults:
                # a loser's fault is raised, but not before every loser is stopped and the
                # winner -- which the caller will now never receive -- is closed
                if won is not None:
                    await won.aclose()
                raise faults[0]
        raise httpcore.ConnectError(f"could not connect to any checked address for {host!r}: {'; '.join(failures)}")

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        """Refuse a Unix socket: this transport reaches URLs, never local paths.

        :param path: the socket path asked for
        :ptype path: str
        :param timeout: connect timeout, in seconds
        :ptype timeout: float | None
        :param socket_options: socket options, if any
        :ptype socket_options: Iterable[httpcore.SOCKET_OPTION] | None
        :return: never returns
        :rtype: httpcore.AsyncNetworkStream
        :raises RuntimeError: always
        """
        raise RuntimeError(f"refusing a Unix-socket connection to {path!r}: this transport reaches URLs only")

    async def sleep(self, seconds: float) -> None:
        """Sleep on httpcore's own network.

        :param seconds: how long
        :ptype seconds: float
        """
        await self._network.sleep(seconds)


class _CoreResponseStream(httpx.AsyncByteStream):
    """An httpcore response body read as an httpx one, with its errors translated."""

    def __init__(self, response: httpcore.Response) -> None:
        """Hold the httpcore response whose body this streams.

        :param response: the response, its body not yet read
        :ptype response: httpcore.Response
        """
        self._response = response

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield the body's chunks as they arrive.

        :return: the body, chunk by chunk
        :rtype: AsyncIterator[bytes]
        """
        with _as_httpx_errors():
            async for chunk in self._response.aiter_stream():
                yield chunk

    async def aclose(self) -> None:
        """Release the response's connection back to its pool."""
        with _as_httpx_errors():
            await self._response.aclose()


class _GuardedRoutes(httpx.AsyncBaseTransport):
    """The client's only transport: each request goes down the route its guard chose.

    httpx's default transport would decide the route again from the proxy
    environment, and connect a direct request by resolving its name again.
    Either second decision can differ from the one the guard checked, so this
    transport makes neither. A :class:`_Proxied` request goes to the proxy the
    guard named; a :class:`_Pinned` one goes through one pool whose network
    connects only to the checked addresses; an :class:`_Unpinned` one through
    the same pool by name. A request carrying no route was never guarded, and
    is refused.

    One pool for every direct host, keyed by host name as httpx's own is, so
    the connection limit binds across hosts and a pooled TLS connection is
    only ever reused for the name its certificate was checked against.
    """

    def __init__(self, *, verify: bool, limits: httpx.Limits) -> None:
        """Build the direct pool; proxy transports are built when first routed to.

        :param verify: whether to verify TLS certificates
        :ptype verify: bool
        :param limits: connection limits, applied to the direct pool and to
            each proxy transport
        :ptype limits: httpx.Limits
        """
        # trust_env left on, as httpx's own transport leaves it: SSL_CERT_FILE
        # and SSL_CERT_DIR still choose the trust store.
        self._ssl_context = httpx.create_ssl_context(verify=verify)
        self._limits = limits
        self._direct = httpcore.AsyncConnectionPool(
            ssl_context=self._ssl_context,
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            network_backend=_PinnedNetwork(),
        )
        self._proxied: dict[str, httpx.AsyncHTTPTransport] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Send ``request`` down the route its guard chose.

        :param request: the request, carrying its route as an extension
        :ptype request: httpx.Request
        :return: the response, its body not yet read
        :rtype: httpx.Response
        :raises RuntimeError: when the request carries no route
        :raises httpx.HTTPError: the transport errors httpx's own transport raises
        """
        route = request.extensions.get(_ROUTE_EXTENSION)
        if isinstance(route, _Proxied):
            proxy = self._proxied.get(route.proxy_url)
            if proxy is None:
                proxy = httpx.AsyncHTTPTransport(proxy=route.proxy_url, verify=self._ssl_context, limits=self._limits)
                self._proxied[route.proxy_url] = proxy
            return await proxy.handle_async_request(request)
        if not isinstance(route, (_Pinned, _Unpinned)):
            raise RuntimeError(f"refusing to send {request.url}: it carries no route from the address guard")
        if not isinstance(request.stream, httpx.AsyncByteStream):
            raise TypeError(f"an async transport needs an async request body, got {type(request.stream).__name__}")
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        token = _CONNECT_PERMIT.set(_ConnectPermit(host=request.url.raw_host.decode("ascii"), route=route))
        try:
            with _as_httpx_errors():
                core_response = await self._direct.handle_async_request(core_request)
        finally:
            _CONNECT_PERMIT.reset(token)
        return httpx.Response(
            status_code=core_response.status,
            headers=core_response.headers,
            stream=_CoreResponseStream(core_response),
            extensions=core_response.extensions,
        )

    async def aclose(self) -> None:
        """Close the direct pool and every proxy transport, the rest even when one fails to close."""
        try:
            await self._direct.aclose()
        finally:
            for proxy in self._proxied.values():
                await proxy.aclose()


@dataclass(frozen=True, slots=True)
class _Opened:
    """A response opened through the guarded redirects, its head read and its body not."""

    #: the answering response, still open.
    response: httpx.Response
    #: the URL that answered, after any redirects followed.
    final_url: str
    #: body bytes read on the redirect hops before this one answered.
    bytes_read: int
    #: the response headers, lower-cased keys.
    headers: dict[str, str]


@dataclass(frozen=True, slots=True)
class _Attempted[T]:
    """What the retry loop ended with: the answering attempt and the call's clock."""

    #: the answering status.
    status_code: int
    #: what the answering attempt produced.
    value: T
    #: attempts made, including the answering one.
    attempts: int
    #: the monotonic reading the call started at.
    started: float
    #: the monotonic reading the whole call must end by.
    deadline: float


@dataclass(frozen=True, slots=True)
class _OpenStream:
    """A stream opened as far as its head, and everything that releases it."""

    #: releases the response and, outside a connection scope, its client.
    stack: AsyncExitStack
    #: the response, its head read and its body not.
    opened: _Opened


class _CapRefusal(Protocol):
    """The shape of :meth:`StandaloneTransport._over_cap`, which a stream refuses with."""

    def __call__(self, url: str, size: int, *, cap: int, elapsed: float, bytes_seen: int) -> LocalCapExceeded:
        """Build the refusal for a body past its cap.

        :param url: the URL that answered too much
        :ptype url: str
        :param size: bytes the body reached
        :ptype size: int
        :param cap: the cap that bound
        :ptype cap: int
        :param elapsed: wall-clock spent, in seconds
        :ptype elapsed: float
        :param bytes_seen: bytes read across the whole request
        :ptype bytes_seen: int
        :return: the typed refusal
        :rtype: LocalCapExceeded
        """
        ...


#: a stream's body was read to its end.
_EXHAUSTED: Final[str] = "exhausted"

#: a stream was closed by its caller before its body ended.
_CLOSED_EARLY: Final[str] = "closed-early"


class _StandaloneStream:
    """:class:`StreamedResponse` over one open httpx response.

    Reads the body chunk by chunk as the socket delivers it, under the
    whole-stream cap and the whole-call deadline, and is never retried: by the
    time a body byte exists, the transport's retry loop has already returned.
    Every way out -- exhaustion, a failure, an early close -- releases the
    connection, through the one stack that holds it.

    **Only an exhausted body ends iteration.** The stream remembers how it
    ended. A read after a failure raises that failure again, and a read after
    the caller closed the stream early -- including a close from another task
    while a read was waiting -- raises :class:`TransportFailed`. So a
    truncated body can never read as a complete one, whoever truncated it.
    """

    def __init__(
        self,
        outcome: _Attempted[_OpenStream],
        *,
        egress: str,
        max_bytes: int,
        clock: Callable[[], float],
        stamp: Callable[[SearchFailure], SearchFailure],
        over_cap: _CapRefusal,
    ) -> None:
        """Hold an opened stream and what bounds its body.

        :param outcome: the retry loop's answer: the open stream, its status,
            the attempts it took, and the call's start and deadline
        :ptype outcome: _Attempted[_OpenStream]
        :param egress: the egress the request left by
        :ptype egress: str
        :param max_bytes: the effective cap on the whole body
        :ptype max_bytes: int
        :param clock: the transport's monotonic clock
        :ptype clock: Callable[[], float]
        :param stamp: the transport's stamping of a failure with egress and time
        :ptype stamp: Callable[[SearchFailure], SearchFailure]
        :param over_cap: the transport's cap refusal
        :ptype over_cap: _CapRefusal
        """
        self._outcome = outcome
        self._opened = outcome.value.opened
        self._stack = outcome.value.stack
        self._egress = egress
        self._max_bytes = max_bytes
        self._clock = clock
        self._stamp = stamp
        self._over_cap = over_cap
        self._source = self._opened.response.aiter_bytes()
        self._received = 0
        #: how the body ended: None while open, else exhausted, closed early, or the failure.
        self._ending: str | SearchFailure | None = None
        self._reading = False
        #: the release, once begun; it runs to its end whoever stops waiting on it.
        self._release: asyncio.Future[None] | None = None
        #: whether a finished release's outcome has reached an ``aclose`` caller.
        self._release_reported = False
        self._closed_at: float | None = None

    @property
    def status_code(self) -> int:
        """The answering status, read before the body.

        :return: the status code
        :rtype: int
        """
        return self._outcome.status_code

    @property
    def headers(self) -> Mapping[str, str]:
        """The answering headers, lower-cased keys, read before the body.

        :return: the headers
        :rtype: Mapping[str, str]
        """
        return self._opened.headers

    @property
    def final_url(self) -> str:
        """The URL that answered.

        :return: the answering URL
        :rtype: str
        """
        return self._opened.final_url

    @property
    def egress(self) -> str:
        """The egress the request left by (D20).

        :return: the egress name
        :rtype: str
        """
        return self._egress

    @property
    def attempts(self) -> int:
        """Attempts made to open the stream; final, since none follows a body byte.

        :return: the attempt count
        :rtype: int
        """
        return self._outcome.attempts

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock since the call began: final once released, so far while open.

        :return: elapsed seconds
        :rtype: float
        """
        end = self._closed_at if self._closed_at is not None else self._clock()
        return end - self._outcome.started

    def __aiter__(self) -> Self:
        """Iterate the body; the stream is its own iterator, so it is read once.

        :return: this stream
        :rtype: Self
        """
        return self

    async def __anext__(self) -> bytes:
        """The next chunk of the body, as the socket delivered it.

        :return: a non-empty chunk
        :rtype: bytes
        :raises StopAsyncIteration: at the end of the body, and only there
        :raises threetears.search.contracts.errors.LocalCapExceeded: when the
            chunk would take the stream past its cap; the chunk is not yielded
        :raises threetears.search.contracts.errors.TimedOut: when the call's
            deadline passes before the next chunk arrives
        :raises threetears.search.contracts.errors.TransportFailed: when the
            connection breaks mid-body, or the stream was closed before its
            body ended
        """
        while True:
            self._refuse_if_ended()
            chunk = await self._read()
            if not chunk:
                continue
            total = self._received + len(chunk)
            if total > self._max_bytes:
                raise await self._failed(
                    self._over_cap(
                        self._opened.final_url,
                        total,
                        cap=self._max_bytes,
                        elapsed=self._clock() - self._outcome.started,
                        bytes_seen=self._opened.bytes_read + total,
                    )
                )
            self._received = total
            return chunk

    def _refuse_if_ended(self) -> None:
        """Answer a read on a stream that has already ended, by how it ended.

        :raises StopAsyncIteration: when the body was read to its end
        :raises threetears.search.contracts.errors.SearchFailure: the failure
            that ended it, again; or :class:`TransportFailed` when it was
            closed early
        """
        if self._ending is None and self._release is None:
            return
        if self._ending == _EXHAUSTED:
            raise StopAsyncIteration
        if isinstance(self._ending, SearchFailure):
            raise self._ending
        raise self._closed_early()

    async def _read(self) -> bytes:
        """Read one chunk under what remains of the deadline, typing every failure.

        :return: the chunk, possibly empty
        :rtype: bytes
        :raises StopAsyncIteration: at the end of the body
        :raises threetears.search.contracts.errors.SearchFailure: as
            :meth:`__anext__` documents
        """
        remaining = self._outcome.deadline - self._clock()
        self._reading = True
        try:
            if remaining <= 0:
                raise TimeoutError
            async with asyncio.timeout(remaining):
                return await anext(self._source)
        except StopAsyncIteration:
            self._ending = _EXHAUSTED
            await self.aclose()
            raise
        except (TimeoutError, httpx.TimeoutException) as exc:
            if self._release is not None:
                raise self._closed_early() from exc
            raise await self._failed(
                TimedOut(
                    f"stream from {self._opened.final_url} passed its deadline after {self._received} body byte(s) "
                    f"and {self.attempts} attempt(s); a stream is never retried once its body has begun",
                    spend=self._spend(),
                    remediation="raise this call's timeout -- it bounds the whole stream, body included",
                )
            ) from exc
        except httpx.HTTPError as exc:
            if self._release is not None:
                raise self._closed_early() from exc
            raise await self._failed(
                TransportFailed(
                    f"stream from {self._opened.final_url} broke after {self._received} body byte(s) and "
                    f"{self.attempts} attempt(s): {type(exc).__name__}: {exc}; a stream is never retried "
                    f"once its body has begun",
                    spend=self._spend(),
                )
            ) from exc
        except Exception as exc:
            # A close from another task pulls the connection out from under this
            # read, and what that raises is the network library's business; it
            # is answered as the early close it is. Anything else is re-raised.
            if self._release is not None:
                raise self._closed_early() from exc
            raise
        finally:
            self._reading = False

    def _closed_early(self) -> SearchFailure:
        """The failure a read meets on a stream its caller closed before the body ended.

        :return: the typed, stamped failure
        :rtype: SearchFailure
        """
        return self._stamp(
            TransportFailed(
                f"stream from {self._opened.final_url} was closed after {self._received} body byte(s), before its "
                f"body ended; what was read is not the whole answer",
                spend=self._spend(),
            )
        )

    def _spend(self) -> Spend:
        """What the stream consumed so far, for a failure leaving it.

        :return: the wall-clock and bytes so far; no calls, which the adapter owns (D4)
        :rtype: Spend
        """
        return Spend(
            wall_clock_seconds=self._clock() - self._outcome.started,
            calls=0,
            bytes_transferred=self._opened.bytes_read + self._received,
        )

    async def _failed(self, failure: SearchFailure) -> SearchFailure:
        """Stamp the failure that ends the stream, remember it, and release the connection.

        The typed failure is what leaves, whatever the release does: a release
        that itself fails is contained -- logged and counted where an operator
        sees it -- and never replaces the failure the caller is owed.

        :param failure: the failure about to leave
        :ptype failure: SearchFailure
        :return: the failure, stamped with egress and occurrence time
        :rtype: SearchFailure
        """
        stamped = self._stamp(failure)
        self._ending = stamped
        try:
            await self.aclose()
        except Exception as exc:
            contain(exc, log=_logger, what=f"releasing a failed stream from {self._opened.final_url} failed")
        return stamped

    async def aclose(self) -> None:
        """Release the connection, and the client it came from outside a scope. Idempotent.

        **A release, once begun, runs to its end.** It runs as its own task
        and this call waits on it shielded, so a caller cancelled while waiting
        is cancelled alone: the release goes on, and the next ``aclose`` waits
        on the same release rather than starting a second. A release cut off
        part-way would leave the connection's slot in its pool taken for good
        -- the network library marks a response closed before it hands the
        connection back -- and no later close could hand it back.

        A release that fails raises from the ``aclose`` that first sees it, and
        from no later one, so leaving an ``async with`` after a failure never
        swaps the failure being raised for the release's.
        """
        if self._ending is None:
            self._ending = _CLOSED_EARLY
        release = self._release
        if release is None:
            release = self._release = asyncio.ensure_future(self._release_connection())
        elif release.done() and self._release_reported:
            return
        try:
            await asyncio.shield(release)
        finally:
            if release.done():
                self._release_reported = True

    async def _release_connection(self) -> None:
        """Close the body iterator and the response, and the client outside a scope."""
        try:
            # httpx's body iterator is an async generator. Closing it before the
            # response leaves nothing suspended for the finalizer -- unless a read
            # is suspended inside it, which closing the response below ends.
            if not self._reading and isinstance(self._source, AsyncGenerator):
                await self._source.aclose()
        finally:
            try:
                await self._stack.aclose()
            finally:
                self._closed_at = self._clock()

    def __del__(self) -> None:
        """Warn about a stream dropped without being released, as httpx and asyncio do."""
        if getattr(self, "_release", True) is None:
            warnings.warn(
                f"unclosed stream from {self._opened.final_url}: release it with `async with` or aclose(), "
                f"or its connection stays open until the process exits",
                ResourceWarning,
                stacklevel=1,
                source=self,
            )

    async def __aenter__(self) -> Self:
        """Enter the block that owns this stream.

        :return: this stream
        :rtype: Self
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the connection on leaving the block.

        :param exc_type: exception type, if the block raised
        :ptype exc_type: type[BaseException] | None
        :param exc: the exception, if the block raised
        :ptype exc: BaseException | None
        :param traceback: the traceback, if the block raised
        :ptype traceback: TracebackType | None
        """
        await self.aclose()


@dataclass(slots=True)
class _Circuit:
    """One upstream's breaker state."""

    #: consecutive failed calls since the last success.
    failures: int = 0
    #: when the breaker opened, on the transport's clock; None while closed.
    opened_at: float | None = None
    #: whether a half-open probe call is out.
    probing: bool = False


class _Breakers:
    """A circuit breaker per upstream, in the three-call shape the family's breakers share.

    The shape ``threetears.core.http_client.CircuitBreakerLike`` declares --
    admit before contact, then record success or failure -- with the states of
    ``threetears.models``' breaker: CLOSED admits everything; ``failure_threshold``
    consecutive failed calls OPEN it, and it refuses every call for
    ``cooldown_seconds``; then it is HALF-OPEN and admits exactly one probe
    call, whose success closes it and whose failure opens it again. Written
    here rather than imported because this leaf imports neither (SR-L7, D24).

    The unit is a **call**, not an attempt: the transport's retries inside one
    call are one outcome, so a breaker trips on an upstream that keeps failing
    whole calls, and a call that recovered on its second attempt counts as a
    success.
    """

    def __init__(self, *, failure_threshold: int, cooldown_seconds: float, clock: Callable[[], float]) -> None:
        """Configure every upstream's breaker.

        :param failure_threshold: consecutive failed calls that open a breaker
        :ptype failure_threshold: int
        :param cooldown_seconds: how long an open breaker refuses before probing
        :ptype cooldown_seconds: float
        :param clock: the transport's monotonic clock
        :ptype clock: Callable[[], float]
        """
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._circuits: dict[str, _Circuit] = {}
        self._lock = threading.Lock()

    def admit(self, upstream: str) -> tuple[bool, float]:
        """Decide whether a call to ``upstream`` may be made.

        :param upstream: the upstream's ``host[:port]``
        :ptype upstream: str
        :return: ``(admitted, seconds until a probe is admitted)``; the second is
            0 when admitted. An admitted call to a half-open breaker is its probe
        :rtype: tuple[bool, float]
        """
        with self._lock:
            circuit = self._circuits.get(upstream)
            if circuit is None or circuit.opened_at is None:
                return True, 0.0
            waited = self._clock() - circuit.opened_at
            if waited < self._cooldown_seconds:
                return False, self._cooldown_seconds - waited
            if circuit.probing:
                return False, 0.0
            circuit.probing = True
            return True, 0.0

    def settle(self, upstream: str, verdict: bool | None) -> None:
        """Record how an admitted call to ``upstream`` ended.

        **Only a breaker with something to remember is held.** An upstream
        that is closed with no failures behind it has the same state as one
        never called, so its entry is dropped: a long-lived fetch transport
        reaches whatever host each candidate names, and a map that kept every
        one of them would grow with every page ever read. What stays is an
        upstream counting failures, open, or half-open.

        :param upstream: the upstream's ``host[:port]``
        :ptype upstream: str
        :param verdict: True for an answer below 500; False for a 5xx or a
            connect/timeout failure that outlasted the attempts; None for an
            outcome that says nothing about the upstream (a guard or cap
            refusal, a cancellation), which only hands a probe back
        :ptype verdict: bool | None
        """
        with self._lock:
            circuit = self._circuits.get(upstream)
            if circuit is None:
                if verdict is not False:
                    return
                circuit = self._circuits[upstream] = _Circuit()
            was_probe = circuit.probing
            circuit.probing = False
            if verdict is True:
                del self._circuits[upstream]
                return
            if verdict is False:
                circuit.failures += 1
                if was_probe or circuit.failures >= self._failure_threshold:
                    if circuit.opened_at is None or was_probe:
                        _logger.warning(
                            "standalone transport opened the circuit to %s after %d failed call(s)",
                            upstream,
                            circuit.failures,
                        )
                    circuit.opened_at = self._clock()
            if circuit.failures == 0 and circuit.opened_at is None:
                del self._circuits[upstream]


def _upstream(url: str) -> str:
    """The upstream a URL reaches, as its breaker keys it: lower-cased ``host[:port]``.

    :param url: an absolute URL
    :ptype url: str
    :return: the upstream key; the URL itself when it has no host, which the
        address guard then refuses
    :rtype: str
    """
    parsed = urlparse(url)
    if not parsed.hostname:
        return url
    host = parsed.hostname.lower().rstrip(".")
    return f"{host}:{parsed.port}" if parsed.port is not None else host


class _Exhausted(Exception):
    """Carries an exhausted call's failure from the attempt loop to the breaker that judges it."""

    def __init__(self, failure: SearchFailure, *, attempts: int, upstream_fault: bool) -> None:
        """Hold the failure and whether it is the upstream's.

        :param failure: the typed, stamped failure the caller receives
        :ptype failure: SearchFailure
        :param attempts: attempts made
        :ptype attempts: int
        :param upstream_fault: whether the attempts failed on the upstream's side
            (connects, timeouts), which a breaker counts
        :ptype upstream_fault: bool
        """
        super().__init__(failure.message)
        self.failure = failure
        self.attempts = attempts
        self.upstream_fault = upstream_fault


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
        user_agent: str = DEFAULT_TRANSPORT_USER_AGENT,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        resolve: Callable[[str], Awaitable[tuple[str, ...]]] = _resolve,
        breaker_failure_threshold: int = _DEFAULT_BREAKER_FAILURE_THRESHOLD,
        breaker_cooldown_seconds: float = _DEFAULT_BREAKER_COOLDOWN_SECONDS,
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
        :param resolve: how the address guard resolves a hostname to its
            addresses, in order of preference, raising
            :class:`TransportFailed` for a name that does not resolve. It is
            the only resolution a direct request gets: the connection is
            opened to these addresses, tried in the order given, and the name
            is never looked up again. Resolution off the event loop by
            default; injected so a test can present a name that resolves
            somewhere it should not without owning a resolver
        :ptype resolve: Callable[[str], Awaitable[tuple[str, ...]]]
        :param breaker_failure_threshold: consecutive failed calls to one
            upstream (``host[:port]``) that open its circuit breaker (SR-D3).
            A failed call is a 5xx answer or attempts that never got one; a
            refusal of this transport's own (guard, cap) is not the upstream's
            failure and counts for nothing
        :ptype breaker_failure_threshold: int
        :param breaker_cooldown_seconds: how long an open breaker refuses calls
            to its upstream before admitting one probe call, whose outcome
            closes or reopens it
        :ptype breaker_cooldown_seconds: float
        :raises ValueError: when ``max_attempts`` or ``breaker_failure_threshold``
            is below 1, or ``max_response_bytes`` or ``breaker_cooldown_seconds``
            is not positive
        """
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        if max_response_bytes < 1:
            raise ValueError(f"max_response_bytes must be positive, got {max_response_bytes}")
        if breaker_failure_threshold < 1:
            raise ValueError(f"breaker_failure_threshold must be at least 1, got {breaker_failure_threshold}")
        if breaker_cooldown_seconds <= 0:
            raise ValueError(f"breaker_cooldown_seconds must be positive, got {breaker_cooldown_seconds}")
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
        self._resolve = resolve
        self._breakers = _Breakers(
            failure_threshold=breaker_failure_threshold, cooldown_seconds=breaker_cooldown_seconds, clock=clock
        )

    @property
    def egress_name(self) -> str:
        """Name of the egress this transport's requests leave by (D20).

        :return: the configured exit's name
        :rtype: str
        """
        return self._egress_name

    @asynccontextmanager
    async def connection_scope(self) -> AsyncIterator[None]:
        """Share one client, and its connection pool, across a block of calls.

        The answer to §3.8's flag that a per-request client is "right for one
        search, wrong for Extract's many-fetch path". Inside the block every
        call this transport makes reuses one pool, so a run that reads twenty
        documents pays for a handshake per *host* rather than per document.
        Outside it nothing changes, which is what keeps SR-L5 true for the
        caller who only ever makes one call: no scope, no client to close, no
        lifecycle to get wrong.

        Opening a scope is the *host's* call, not a stage's. The transport is
        an injected port, and how long its connections live is a property of
        the deployment that constructed it -- so Extract opens no scope of its
        own, and a host that wants pooling wraps its own work in one::

            async with transport.connection_scope():
                candidates = await extract(candidate_set, transport=transport)

        Re-entrant and concurrency-safe: two overlapping scopes over one
        transport each get their own client, and each closes only its own.

        **The block bounds the client, so work must finish inside it.** A task
        started in the scope inherits the context, and therefore the client,
        because that is how a ``ContextVar`` is copied into a task. If that
        task outlives the block it keeps a client the exiting scope has
        already closed, and its next call fails on a closed pool. Awaiting
        inside the block -- ``gather`` included -- is the ordinary shape and
        is fine; the trap is only a task deliberately left running past the
        ``async with``. That reads as a lifetime bug either way, and the
        reason it is worth naming is that the symptom surfaces on the *task*
        rather than at the scope that closed the client underneath it.

        :return: an async context manager whose block shares one client
        :rtype: AsyncIterator[None]
        """
        client = self._new_client(keepalive=self._max_connections)
        token = _SCOPED_CLIENTS.set({**_SCOPED_CLIENTS.get(), id(self): client})
        try:
            yield
        finally:
            _SCOPED_CLIENTS.reset(token)
            await client.aclose()

    def _new_client(self, *, keepalive: int) -> httpx.AsyncClient:
        """Build a client carrying this transport's deployment configuration.

        No timeout is set here: a timeout belongs to a call, not to a client
        (SR-G1/SR-G2), and a shared client outlives the deadline of any one
        call made through it. Each request states what remains of its own
        bound.

        Nor does the client read the proxy environment. Its one transport
        sends each request down the route the address guard chose for it,
        so the route is decided once, by the code that checked it.

        :param keepalive: connections this client may hold open between
            requests. Zero for the per-request client, which has no next
            request to hold one for
        :ptype keepalive: int
        :return: the configured client
        :rtype: httpx.AsyncClient
        """
        limits = httpx.Limits(max_connections=self._max_connections, max_keepalive_connections=keepalive)
        return httpx.AsyncClient(
            follow_redirects=False,
            transport=_GuardedRoutes(verify=self._verify_tls, limits=limits),
            trust_env=False,
        )

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[httpx.AsyncClient]:
        """Yield the scope's client where one is open, a fresh one where not.

        The one place the two lifecycles meet, so every call site below stays
        unaware of which it got.

        :return: an async context manager yielding the client to use
        :rtype: AsyncIterator[httpx.AsyncClient]
        """
        shared = _SCOPED_CLIENTS.get().get(id(self))
        if shared is not None:
            # not closed here: it belongs to the scope that opened it, and a
            # call that closed it would break every later call in the block.
            yield shared
            return
        async with self._new_client(keepalive=0) as client:
            yield client

    @traced(name="threetears.search.transport.request")
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

    @traced(name="threetears.search.transport.fetch")
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

    @traced(name="threetears.search.transport.stream")
    async def stream(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Mapping[str, JsonValue] | None = None,
        timeout_seconds: float | None = None,
        max_bytes: int,
    ) -> StreamedResponse:
        """Open one bounded, guarded stream, and return it with its head read.

        Satisfies
        :class:`~threetears.search.contracts.transport.StreamingTransport`
        through the same machinery as :meth:`request` and :meth:`fetch`: the
        one retry loop and whole-call deadline, the address guard with its
        connect pinning, the redirect policy, and the byte cap. What differs is
        where an attempt ends. Here it ends at the head: each attempt opens the
        response as far as its status and headers, a connect failure, timeout
        or 5xx is retried from there, and the first answer that is not retried
        comes back open. No attempt is made once this returns, so no body byte
        is ever followed by a retry.

        **The deadline bounds the body too.** ``timeout_seconds`` is the bound
        on the whole call; every chunk read is given what remains of it, and a
        stream still open when it passes fails as :class:`TimedOut`.

        **The cap counts the whole stream**, as :meth:`fetch` counts a body: a
        declared length past it is refused before this returns, and a chunk
        that would take the stream past it is refused rather than yielded.

        :param method: HTTP method
        :ptype method: str
        :param url: absolute URL, built by the caller from a
            deployment-configured base (D21)
        :ptype url: str
        :param headers: request headers, if any
        :ptype headers: Mapping[str, str] | None
        :param json_body: JSON request body, if any
        :ptype json_body: Mapping[str, JsonValue] | None
        :param timeout_seconds: per-call override of the configured timeout,
            bounding the whole stream, body included (SR-G2)
        :ptype timeout_seconds: float | None
        :param max_bytes: this call's cap on the whole body (SR-G5),
            tightening the configured ``max_response_bytes`` where it is lower
            and never loosening it
        :ptype max_bytes: int
        :return: the open stream; the caller releases it with ``async with``
            or ``aclose()``
        :rtype: StreamedResponse
        :raises ValueError: when ``max_bytes`` is not positive
        :raises threetears.search.contracts.errors.TimedOut: when the attempts
            timed out, or the whole-call bound ran out before another fit
        :raises threetears.search.contracts.errors.TransportFailed: on a
            refused address, a refused redirect, or a connect/protocol failure
            that outlasted the attempts
        :raises threetears.search.contracts.errors.LocalCapExceeded: when the
            declared length exceeds the effective cap
        """
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        cap = min(max_bytes, self._max_response_bytes)
        request_headers = self._request_headers(headers)

        async def attempt_once(
            timeout: httpx.Timeout, elapsed: float, bytes_so_far: int
        ) -> tuple[int, _OpenStream, int]:
            # Everything this attempt opens is released if it fails; on
            # success its release moves out with the stream it returns.
            async with AsyncExitStack() as attempt_stack:
                client = await attempt_stack.enter_async_context(self._client())
                opened = await self._open(
                    attempt_stack,
                    client,
                    method,
                    url,
                    headers=request_headers,
                    params=None,
                    json_body=json_body,
                    max_bytes=cap,
                    allowed_content_types=None,
                    timeout=timeout,
                    elapsed_so_far=elapsed,
                    bytes_so_far=bytes_so_far,
                )
                self._refuse_declared_overrun(
                    opened.response,
                    opened.final_url,
                    max_bytes=cap,
                    elapsed=elapsed,
                    bytes_seen=bytes_so_far + opened.bytes_read,
                )
                kept = attempt_stack.pop_all()
            return opened.response.status_code, _OpenStream(stack=kept, opened=opened), opened.bytes_read

        async def release(stream: _OpenStream) -> None:
            await stream.stack.aclose()

        outcome = await self._retrying(url, timeout_seconds=timeout_seconds, attempt_once=attempt_once, release=release)
        return _StandaloneStream(
            outcome,
            egress=self._egress_name,
            max_bytes=cap,
            clock=self._clock,
            stamp=self._stamped,
            over_cap=self._over_cap,
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
        """Run the bounded, guarded, retrying, buffered exchange ``request`` and ``fetch`` share.

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
        request_headers = self._request_headers(headers)

        async def attempt_once(
            timeout: httpx.Timeout, elapsed: float, bytes_so_far: int
        ) -> tuple[int, tuple[bytes, str, dict[str, str]], int]:
            async with self._client() as client:
                status, body, final_url, read, response_headers = await self._exchange(
                    client,
                    method,
                    url,
                    headers=request_headers,
                    params=params,
                    json_body=json_body,
                    max_bytes=max_bytes,
                    allowed_content_types=allowed_content_types,
                    timeout=timeout,
                    elapsed_so_far=elapsed,
                    bytes_so_far=bytes_so_far,
                )
            return status, (body, final_url, response_headers), read

        outcome = await self._retrying(url, timeout_seconds=timeout_seconds, attempt_once=attempt_once)
        body, final_url, response_headers = outcome.value
        return TransportResponse(
            status_code=outcome.status_code,
            body=body,
            final_url=final_url,
            egress=self._egress_name,
            elapsed_seconds=self._clock() - outcome.started,
            attempts=outcome.attempts,
            headers=response_headers,
        )

    def _request_headers(self, headers: Mapping[str, str] | None) -> dict[str, str]:
        """The headers every request sends: this transport's ``User-Agent``, then the caller's.

        :param headers: the caller's headers, if any
        :ptype headers: Mapping[str, str] | None
        :return: the headers to send
        :rtype: dict[str, str]
        """
        return {"User-Agent": self._user_agent, **dict(headers or {})}

    async def _retrying[T](
        self,
        url: str,
        *,
        timeout_seconds: float | None,
        attempt_once: Callable[[httpx.Timeout, float, int], Awaitable[tuple[int, T, int]]],
        release: Callable[[T], Awaitable[None]] | None = None,
    ) -> _Attempted[T]:
        """The one retry loop: bounded attempts, backoff, and one deadline for all of them.

        Every protocol this module implements runs through here, so the retry
        policy, the whole-call deadline and the typed failure an exhausted call
        raises are written once. What one attempt *is* differs -- a buffered
        exchange, or a stream opened as far as its headers -- and arrives as
        ``attempt_once``, which returns the answering status, its value, and
        the body bytes it read.

        A 5xx is retried while attempts and the bound remain; the value it
        came with is handed to ``release`` first (a stream's open connection),
        and a 5xx the bound cannot fund a retry for comes back as itself.

        :param url: the URL being reached, for messages and failures
        :ptype url: str
        :param timeout_seconds: per-call override of the configured timeout,
            bounding the whole call (SR-G2)
        :ptype timeout_seconds: float | None
        :param attempt_once: one attempt, given what remains of the bound, the
            wall-clock already spent and the bytes already read
        :ptype attempt_once: Callable[[httpx.Timeout, float, int], Awaitable[tuple[int, T, int]]]
        :param release: what to do with a 5xx answer's value before retrying
            past it; nothing when None
        :ptype release: Callable[[T], Awaitable[None]] | None
        :return: the answering attempt's status and value, with the attempt
            count and the call's start and deadline
        :rtype: _Attempted[T]
        :raises threetears.search.contracts.errors.SearchFailure: an attempt's
            typed refusal, stamped, or the typed failure of exhausted attempts
        """
        effective_timeout = timeout_seconds if timeout_seconds is not None else self._timeout_seconds
        upstream = _upstream(url)
        set_span_attribute("server.address", upstream)
        set_span_attribute("transport.egress", self._egress_name)
        admitted, wait = self._breakers.admit(upstream)
        if not admitted:
            set_span_attribute("transport.circuit_open", True)
            raise self._stamped(self._circuit_open(url, upstream, wait))
        # what this call says about the upstream, for its breaker: None until an
        # outcome that is the upstream's own -- an answer, or attempts that never
        # got one -- and None still for a refusal of ours (guard, cap) or a
        # cancellation, which only hand a probe back.
        verdict: bool | None = None
        try:
            outcome = await self._attempts(
                url,
                effective_timeout=effective_timeout,
                attempt_once=attempt_once,
                release=release,
            )
            verdict = outcome.status_code < 500
            set_span_attribute("http.response.status_code", outcome.status_code)
            set_span_attribute("transport.attempts", outcome.attempts)
            return outcome
        except _Exhausted as exhausted:
            verdict = False if exhausted.upstream_fault else None
            set_span_attribute("transport.attempts", exhausted.attempts)
            raise exhausted.failure from exhausted.__cause__
        finally:
            self._breakers.settle(upstream, verdict)

    async def _attempts[T](
        self,
        url: str,
        *,
        effective_timeout: float,
        attempt_once: Callable[[httpx.Timeout, float, int], Awaitable[tuple[int, T, int]]],
        release: Callable[[T], Awaitable[None]] | None,
    ) -> _Attempted[T]:
        """Make the attempts of one admitted call, backing off between them under its one deadline.

        :param url: the URL being reached
        :ptype url: str
        :param effective_timeout: the whole call's bound, in seconds
        :ptype effective_timeout: float
        :param attempt_once: one attempt, as :meth:`_retrying` documents
        :ptype attempt_once: Callable[[httpx.Timeout, float, int], Awaitable[tuple[int, T, int]]]
        :param release: what to do with a 5xx answer's value before retrying past it
        :ptype release: Callable[[T], Awaitable[None]] | None
        :return: the answering attempt
        :rtype: _Attempted[T]
        :raises threetears.search.contracts.errors.SearchFailure: an attempt's
            typed refusal, stamped
        :raises _Exhausted: when the attempts ran out without an answer
        """
        started = self._clock()
        deadline = started + effective_timeout
        attempt = 0
        bytes_seen = 0
        last: BaseException | None = None
        bound_spent = False
        while attempt < self._max_attempts:
            attempt += 1
            # What remains of the caller's bound, never the whole of it
            # again: attempt two under a 10s bound that attempt one spent 6s
            # of gets 4s, so the request cannot outlive the deadline whatever
            # the attempt ceiling is. Stated per request rather than on the
            # client, because a scoped client outlives any one call's bound.
            timeout = httpx.Timeout(max(0.0, min(effective_timeout, deadline - self._clock())))
            try:
                status, value, read = await attempt_once(timeout, self._clock() - started, bytes_seen)
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
                continue
            except httpx.HTTPError as exc:
                last = exc
                break
            bytes_seen += read
            if status >= 500 and attempt < self._max_attempts:
                _logger.warning(
                    "standalone transport got HTTP %d from %s, retrying (attempt %d of %d)",
                    status,
                    url,
                    attempt,
                    self._max_attempts,
                )
                # released outside the attempt's handlers: closing the 5xx is
                # cleanup, and a failure in it is not this attempt's failure.
                before_sleep = (lambda: self._release_quietly(release, value, url)) if release is not None else None
                if await self._wait_to_retry(attempt, deadline, before_sleep=before_sleep):
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
            return _Attempted(status_code=status, value=value, attempts=attempt, started=started, deadline=deadline)
        exhausted = self._exhausted(
            url,
            attempt,
            last,
            elapsed=self._clock() - started,
            bytes_seen=bytes_seen,
            bound=effective_timeout,
            bound_spent=bound_spent,
        )
        raise _Exhausted(
            self._stamped(exhausted),
            attempts=attempt,
            upstream_fault=isinstance(last, (httpx.TimeoutException, httpx.TransportError)) or bound_spent,
        ) from last

    async def _release_quietly[T](self, release: Callable[[T], Awaitable[None]], value: T, url: str) -> None:
        """Release a 5xx answer before retrying past it; a failure to release is logged, not raised.

        The answer is being thrown away and its connection with it, so a close
        that fails leaves nothing to clean up and nothing the caller can act
        on. It is logged where an operator sees it, and the retry proceeds.

        :param release: how the value is released
        :ptype release: Callable[[T], Awaitable[None]]
        :param value: the 5xx answer's value
        :ptype value: T
        :param url: the URL it came from, for the log line
        :ptype url: str
        """
        try:
            await release(value)
        except (httpx.HTTPError, OSError) as exc:
            _logger.warning("standalone transport could not release a 5xx answer from %s before retrying: %s", url, exc)

    def _circuit_open(self, url: str, upstream: str, wait: float) -> TransportFailed:
        """The refusal for a call to an upstream whose breaker is open (SR-D3).

        :class:`TransportFailed`, because that is the class whose meaning this
        is: the transport cannot complete requests to this upstream, and the
        caller should give up now rather than retry. No socket was opened, so
        the spend is zero.

        :param url: the URL refused
        :ptype url: str
        :param upstream: the upstream whose breaker is open
        :ptype upstream: str
        :param wait: seconds until a probe is admitted; 0 while one is already out
        :ptype wait: float
        :return: the typed refusal
        :rtype: TransportFailed
        """
        when = f"a probe is admitted in {wait:.1f}s" if wait > 0 else "a probe call is already out"
        return TransportFailed(
            f"refusing {url}: the circuit to {upstream} is open after repeated failed calls; {when}",
            spend=Spend(),
            remediation=(
                f"the upstream {upstream} has been failing; wait for the breaker's probe, or tune "
                f"breaker_failure_threshold / breaker_cooldown_seconds on the transport"
            ),
        )

    async def _wait_to_retry(
        self, attempt: int, deadline: float, *, before_sleep: Callable[[], Awaitable[None]] | None = None
    ) -> bool:
        """Back off before the next attempt, where the bound still allows one.

        The backoff is part of the caller's bound rather than an addition to
        it (SR-G2), so this both caps the wait and refuses to take one that
        would leave no attempt on the other side of it: sleeping out the last
        of a deadline to then report a timeout is the same failure, later.

        :param attempt: the attempt that just failed, 1-based
        :ptype attempt: int
        :param deadline: the monotonic reading the whole call must end by
        :ptype deadline: float
        :param before_sleep: run once the retry is decided and before the
            backoff -- a stream's 5xx connection is released here rather than
            held through the wait. Not run when no retry fits, because then
            the answer it belongs to is the one that goes back
        :ptype before_sleep: Callable[[], Awaitable[None]] | None
        :return: whether the backoff was taken and another attempt fits
        :rtype: bool
        """
        pause = self._backoff(attempt)
        if deadline - self._clock() - pause < MINIMUM_RETRY_SECONDS:
            return False
        if before_sleep is not None:
            await before_sleep()
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
        timeout: httpx.Timeout,
        elapsed_so_far: float,
        bytes_so_far: int,
    ) -> tuple[int, bytes, str, int, dict[str, str]]:
        """Send one request, following guarded redirects, reading the body under a cap.

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
        :param timeout: what remains of this call's bound, stated per request
            because the client may be shared by a connection scope
        :ptype timeout: httpx.Timeout
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
        async with AsyncExitStack() as stack:
            opened = await self._open(
                stack,
                client,
                method,
                url,
                headers=headers,
                params=params,
                json_body=json_body,
                max_bytes=max_bytes,
                allowed_content_types=allowed_content_types,
                timeout=timeout,
                elapsed_so_far=elapsed_so_far,
                bytes_so_far=bytes_so_far,
            )
            body, read = await self._read_capped(
                opened.response,
                opened.final_url,
                max_bytes=max_bytes,
                elapsed=elapsed_so_far,
                bytes_seen=bytes_so_far + opened.bytes_read,
            )
        return opened.response.status_code, body, opened.final_url, opened.bytes_read + read, opened.headers

    async def _open(
        self,
        stack: AsyncExitStack,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, str] | None,
        json_body: Mapping[str, JsonValue] | None,
        max_bytes: int,
        allowed_content_types: tuple[str, ...] | None,
        timeout: httpx.Timeout,
        elapsed_so_far: float,
        bytes_so_far: int,
    ) -> _Opened:
        """Send one request through guarded redirects, and leave the answer open, body unread.

        The half of an exchange every protocol shares: guard and route each
        hop, read and discard a redirect's body under the cap, gate the
        answering response's media type. The answering response stays open on
        ``stack``, so the caller decides whether its body is read whole
        (``request``, ``fetch``) or handed on as a stream (``stream``).

        :param stack: where the answering response's release is registered
        :ptype stack: AsyncExitStack
        :param client: the client to send through
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
        :param max_bytes: effective cap, applied to each redirect's body
        :ptype max_bytes: int
        :param allowed_content_types: media-type prefixes a successful
            response must match, or None to accept any
        :ptype allowed_content_types: tuple[str, ...] | None
        :param timeout: what remains of this call's bound
        :ptype timeout: httpx.Timeout
        :param elapsed_so_far: wall-clock already spent, for failure spend
        :ptype elapsed_so_far: float
        :param bytes_so_far: bytes already read, for failure spend
        :ptype bytes_so_far: int
        :return: the open answering response, the URL that answered, the
            redirect body bytes read on the way, and its lower-cased headers
        :rtype: _Opened
        :raises threetears.search.contracts.errors.TransportFailed: on a
            refused address or a refused redirect
        :raises threetears.search.contracts.errors.LocalCapExceeded: when a
            redirect's body exceeds the cap, or the content-type gate refuses
        """
        target = url
        read_total = 0
        for hop in range(self._max_redirects + 1):
            # every hop is guarded and routed afresh: a redirect names a new
            # host, and the addresses checked for the last one say nothing
            # about it.
            route = await self._guard(target, elapsed=elapsed_so_far, bytes_seen=bytes_so_far + read_total)
            async with AsyncExitStack() as hop_stack:
                response = await hop_stack.enter_async_context(
                    client.stream(
                        method,
                        target,
                        headers=dict(headers),
                        params=dict(params) if params else None,
                        json=dict(json_body) if json_body else None,
                        timeout=timeout,
                        extensions={_ROUTE_EXTENSION: route},
                    )
                )
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
                    # the answer outlives this hop: its release moves to the
                    # caller's stack, and leaving this block releases nothing.
                    stack.push_async_exit(hop_stack.pop_all())
                    return _Opened(response=response, final_url=target, bytes_read=read_total, headers=response_headers)
                _, read = await self._read_capped(
                    response,
                    target,
                    max_bytes=max_bytes,
                    elapsed=elapsed_so_far,
                    bytes_seen=bytes_so_far + read_total,
                )
            read_total += read
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
        self._refuse_declared_overrun(response, url, max_bytes=max_bytes, elapsed=elapsed, bytes_seen=bytes_seen)
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise self._over_cap(url, total, cap=max_bytes, elapsed=elapsed, bytes_seen=bytes_seen + total)
            chunks.append(chunk)
        return b"".join(chunks), total

    def _refuse_declared_overrun(
        self, response: httpx.Response, url: str, *, max_bytes: int, elapsed: float, bytes_seen: int
    ) -> None:
        """Refuse a response whose declared ``Content-Length`` is already past the cap.

        Before a byte of the body is read, which is the point: a declared
        overrun costs nothing to refuse. An undeclared or lying length is the
        incremental count's to catch.

        :param response: the open response, body unread
        :ptype response: httpx.Response
        :param url: the URL that answered, for the failure message
        :ptype url: str
        :param max_bytes: the cap
        :ptype max_bytes: int
        :param elapsed: wall-clock already spent, for failure spend
        :ptype elapsed: float
        :param bytes_seen: bytes already read, for failure spend
        :ptype bytes_seen: int
        :raises threetears.search.contracts.errors.LocalCapExceeded: when the
            declared length exceeds the cap
        """
        declared = response.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > max_bytes:
            raise self._over_cap(url, int(declared), cap=max_bytes, elapsed=elapsed, bytes_seen=bytes_seen)

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
        if media_type_accepted(media_type, allowed):
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

    async def _guard(self, url: str, *, elapsed: float, bytes_seen: int) -> _Route:
        """Refuse a target this transport must not reach, and route the rest (D21, SR-N3).

        Two guards, both deployment-configured: an optional host allowlist,
        which is the strongest available answer to "never a caller-supplied
        base URL" at the only seam that can enforce it; and a private-address
        refusal covering the host's address in every accepted spelling
        (``127.1``, ``2130706433``, a trailing dot) and every address the name
        resolves to, since a name resolving to one public and one loopback
        address is the interesting case rather than a hypothetical one.

        The route returned is what the connection then follows, which is what
        makes the address check binding rather than advisory. A direct request
        is :class:`_Pinned` to the addresses checked here, and its socket is
        opened to one of them and nothing else: the name is not resolved a
        second time, so a name that answers this lookup with a public address
        and the next with an internal one (DNS rebinding) never reaches the
        internal one. A redirect is a new target, guarded and pinned on its
        own.

        One exception to resolving here, and only one: a name that cannot be
        resolved here, on a request that leaves through a forward proxy. Such
        a host has no DNS of its own, the proxy resolves the name, and the
        proxy's ACL is the boundary. Where local resolution works its answers
        are checked whatever route the request takes.

        What this does not cover: a proxied request is :class:`_Proxied`, and
        the proxy resolves the name itself when it connects -- so behind a
        proxy, rebinding is the proxy's to refuse. A transport built with
        ``allow_private_addresses=True`` checks no address, and so pins none.

        :param url: the absolute URL about to be reached
        :ptype url: str
        :param elapsed: wall-clock spent, for the failure's spend
        :ptype elapsed: float
        :param bytes_seen: bytes read, for the failure's spend
        :ptype bytes_seen: int
        :return: the route this request must take
        :rtype: _Route
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
        # A trailing dot and the numeric spellings of an address reach the same place as the plain
        # forms, so they are normalised before anything is checked.
        host = host.rstrip(".")
        proxy = _proxy_for(parsed.scheme, host)
        if self._allow_private_addresses:
            return _Proxied(proxy) if proxy is not None else _Unpinned()
        address = _as_address(host)
        if (address is not None and _is_blocked_address(address)) or _is_local_name(host):
            raise TransportFailed(
                f"refusing to reach host {host!r}: it names a non-public address",
                spend=spend,
                remediation=(
                    "a search instance genuinely on this host's own network is reached by constructing the "
                    "transport with allow_private_addresses=True -- deployment config, never a per-call "
                    "parameter (D21)"
                ),
            )
        try:
            # an address names itself: there is nothing to look up, and nothing a lookup could change.
            addresses = (address,) if address is not None else await self._resolve(host)
        except TransportFailed:
            # A host whose only way out is a forward proxy has no DNS of its own: names resolve at
            # the proxy. Found live -- every fetch failed "cannot resolve host" while curl through
            # the same proxy returned 200. Only when the name cannot be resolved here AND the
            # request goes to a proxy is the check the proxy's to make. Where local resolution
            # works, its answers are checked whichever way the request leaves. The route returned
            # is the one the request then takes, so a request judged proxied here cannot leave
            # direct and unchecked.
            if proxy is not None:
                return _Proxied(proxy)
            raise
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
        if proxy is not None:
            return _Proxied(proxy)
        if not addresses:
            raise TransportFailed(f"refusing to reach host {host!r}: it resolved to no address", spend=spend)
        return _Pinned(addresses)

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

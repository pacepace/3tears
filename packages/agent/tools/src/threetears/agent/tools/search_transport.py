"""The host-side transports the search leaf is injected with (search-spec.md §4.4).

``3tears-search`` opens no HTTP client of its own: adapters and Extract reach
the network through an injected transport, and the family's sanctioned client
is :class:`threetears.core.http_client.TracedHttpClient`. The leaf cannot
import core (SR-L7), so the adapter that bridges the two lives *here*, where
core is already a hard dependency.

**Two transports, because the two protocols want different machines.**
Gate A split :class:`~threetears.search.contracts.transport.SearchTransport`
from :class:`~threetears.search.contracts.transport.FetchTransport` on the
grounds that a buffered search response and an arbitrary web document carry
different obligations. Building both over ``TracedHttpClient`` turned out to
be impossible rather than merely awkward, and the reason is worth stating
because it is structural, not an implementation gap:

- ``TracedHttpClient`` is constructed **per upstream** -- it takes one
  ``upstream_base_url`` and joins request paths onto it, and its circuit
  breaker guards that one upstream. That is exactly right for SearXNG or
  Tavily, which are configured deployments a pod talks to repeatedly.
- Extract fetches **candidate-derived URLs**: a different host on every call,
  chosen by whatever the provider returned. There is no upstream to construct
  a client for, no breaker whose state would mean anything, and -- decisively
  -- ``TracedHttpClient`` reads whole bodies, while ``FetchTransport`` must
  read incrementally against a per-call byte cap and refuse on content type
  *before* the body (SR-G5, §3.5).

So :class:`TracedSearchTransport` serves the search half over
``TracedHttpClient``, and the fetch half is
:class:`~threetears.search.standalone.StandaloneTransport`, which already
implements the capped, gated, SSRF-guarded read and is the family's one
sanctioned single-purpose transport module (D19). Using it from a host that
*has* core reads as a contradiction of its own docstring only until the split
above is applied: what that docstring rules out is a host re-implementing the
traced client, not a host reaching for the only implementation in the family
that can read an arbitrary URL under a cap.

**Neither transport is constructed per call site.** Both are deployment
configuration -- the timeouts, the egress, the guards -- and D21 puts that
here rather than at a call site so no call site can skip it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

import httpx
from pydantic import JsonValue

from threetears.core.config import DEFAULT_HTTP_TIMEOUT_SECONDS
from threetears.core.http_client import ATTEMPTS_EXTENSION, CircuitBreakerLike, TracedHttpClient
from threetears.search.contracts import EGRESS_DIRECT, LocalCapExceeded, Spend, TransportResponse
from threetears.search.standalone import (
    DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    StandaloneTransport,
)

if TYPE_CHECKING:
    from threetears.core.egress import EgressDriver

__all__ = [
    "DEFAULT_FETCH_MAX_REDIRECTS",
    "OFF_BASE_SCOPE",
    "TracedSearchTransport",
    "build_fetch_transport",
]

#: Refusal identity for a request aimed somewhere other than the transport's
#: own configured upstream. Named, because "the URL is not on this
#: transport's base" is the D21 guard doing its job, not a transport fault.
OFF_BASE_SCOPE: Final[str] = "off-base-url"

#: Redirect hops a *fetch* may follow, and deliberately not the leaf's own
#: :data:`~threetears.search.standalone.DEFAULT_MAX_REDIRECTS` of 0.
#:
#: The two halves want opposite defaults, for the same reason they want
#: different transports. A search upstream is a deployment-configured host
#: answering its own API: a healthy one does not redirect, so refusing to
#: follow is a signal rather than a limitation. A fetch is a
#: candidate-derived URL, where http->https, ``www``, and trailing-slash
#: canonicalisation are how a large share of the real web answers -- refusing
#: to follow turns an ordinary page into "no readable content".
#:
#: Every hop is re-guarded (the SSRF check and the https-downgrade refusal run
#: per hop in :meth:`StandaloneTransport._exchange`), so following is bounded
#: rather than trusting. 5 matches what the hand-rolled body this replaced
#: allowed, which is the behaviour callers are already built around.
DEFAULT_FETCH_MAX_REDIRECTS: Final[int] = 5


class _UnclosableTransport(httpx.AsyncBaseTransport):
    """Shields a caller-supplied transport from the per-call client's ``aclose()``.

    A fresh :class:`TracedHttpClient` per call is deliberate (SR-L5), and
    closing it is what keeps a one-shot ``asyncio.run()`` leaving nothing
    open. But ``AsyncClient.aclose()`` closes the transport it was handed,
    and an injected one belongs to the caller -- so a second call through the
    same transport found it closed. Wrapping delegates every request and
    swallows only the close, which leaves the client's own lifecycle intact.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        """Wrap *inner*.

        :param inner: the caller's transport, which this must not close
        :ptype inner: httpx.AsyncBaseTransport
        :return: nothing
        :rtype: None
        """
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Delegate one exchange unchanged.

        :param request: the request to perform
        :ptype request: httpx.Request
        :return: the wrapped transport's response
        :rtype: httpx.Response
        """
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        """Deliberately nothing: the wrapped transport is the caller's to close.

        :return: nothing
        :rtype: None
        """
        return None


def _origin(url: str) -> tuple[str, str, int | None]:
    """The scheme/host/port triple two URLs must share to be the same upstream.

    :param url: the URL to reduce to its origin
    :ptype url: str
    :return: scheme, lower-cased host, port (None when the scheme's default)
    :rtype: tuple[str, str, int | None]
    """
    parts = urlsplit(url)
    return parts.scheme.lower(), (parts.hostname or "").lower(), parts.port


class TracedSearchTransport:
    """:class:`~threetears.search.contracts.transport.SearchTransport` over the traced client.

    Satisfies the protocol structurally. One instance per configured provider
    upstream: the base URL, the bounds and the egress are deployment facts,
    and the injected circuit breaker is held here rather than on the client so
    its state survives across calls.

    **A client per call, and a breaker across them.** A search is one request,
    and SR-L5 requires a single call to work from a one-shot ``asyncio.run()``
    with no lifecycle to manage and nothing left to close -- so this transport
    holds no client between calls and the caller acquires nothing to close.
    The breaker is the one piece of state that would be meaningless per call,
    so it is the one piece kept.
    """

    def __init__(
        self,
        *,
        base_url: str,
        circuit_breaker: CircuitBreakerLike | None = None,
        timeout_seconds: float | None = None,
        max_attempts: int = 3,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
        egress: EgressDriver | None = None,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Bind to one configured upstream.

        :param base_url: the upstream's base URL, from deployment config.
            MUST NOT come from a caller or the environment at request time
            (D21, SR-K1) -- :meth:`request` refuses any URL off this origin
        :ptype base_url: str
        :param circuit_breaker: optional breaker guarding this upstream;
            None disables circuit breaking
        :ptype circuit_breaker: CircuitBreakerLike | None
        :param timeout_seconds: default per-call timeout; None takes core's
            configured default. A default, never a constant (SR-G1)
        :ptype timeout_seconds: float | None
        :param max_attempts: attempt ceiling for the bounded retry (SR-G4)
        :ptype max_attempts: int
        :param initial_backoff: first backoff interval, in seconds
        :ptype initial_backoff: float
        :param max_backoff: backoff ceiling, in seconds
        :ptype max_backoff: float
        :param egress: which exit requests leave by (D20/SR-N2); None is the
            default route, reported as ``direct``
        :ptype egress: EgressDriver | None
        :param http_transport: the same dependency-injection seam
            ``TracedHttpClient`` declares, forwarded so a test can bind a
            responder without a network. Production leaves it None; an
            explicit transport wins over ``egress`` there, exactly as it does
            in core, so a test that pinned one keeps it
        :ptype http_transport: httpx.AsyncBaseTransport | None
        :return: nothing
        :rtype: None
        :raises ValueError: when ``base_url`` is empty or not an HTTP(S) URL
        """
        scheme, host, _port = _origin(base_url)
        if not host or scheme not in ("http", "https"):
            raise ValueError(f"base_url must be an http(s) URL with a host, got {base_url!r}")
        self._base_url = base_url
        self._origin = _origin(base_url)
        self._circuit_breaker = circuit_breaker
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._egress = egress
        self._http_transport = http_transport

    @property
    def egress_name(self) -> str:
        """Name of the egress this transport's requests leave by (D20).

        ``TracedHttpClient`` reports ``None`` for an unconfigured egress,
        deliberately, so a table can tell "nobody chose" from "somebody chose
        the default route". The contract has no such distinction -- ``direct``
        is a named value and every candidate must carry one -- so the absence
        resolves to the name here, at the one seam that has to answer.

        :return: the configured exit's name; ``direct`` for the default route
        :rtype: str
        """
        return self._egress.name if self._egress is not None else EGRESS_DIRECT

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
        """Perform one bounded, traced HTTP exchange against the configured upstream.

        :param method: HTTP method
        :ptype method: str
        :param url: absolute URL, which MUST be on this transport's
            configured base (D21)
        :ptype url: str
        :param headers: request headers, if any
        :ptype headers: Mapping[str, str] | None
        :param params: query parameters, if any
        :ptype params: Mapping[str, str] | None
        :param json_body: JSON request body, if any
        :ptype json_body: Mapping[str, JsonValue] | None
        :param timeout_seconds: per-call bound (SR-G2), applying to the whole
            call including retries and backoff -- not to each attempt; None
            uses the configured default per attempt
        :ptype timeout_seconds: float | None
        :return: the completed exchange, with the client's attempt count
        :rtype: TransportResponse
        :raises threetears.search.contracts.errors.LocalCapExceeded: when the
            URL is not on this transport's configured upstream
        :raises Exception: whatever the traced client raises on exhaustion or
            an open breaker; the calling adapter maps it onto the taxonomy
            (SR-J1)
        """
        if _origin(url) != self._origin:
            raise LocalCapExceeded(
                f"transport is configured for {self._base_url} and will not reach {url}",
                spend=Spend(),
                remediation="construct a transport per upstream; base URLs are deployment config (D21)",
                scope=OFF_BASE_SCOPE,
            )

        started = time.monotonic()
        client = TracedHttpClient(
            upstream_base_url=self._base_url,
            circuit_breaker=self._circuit_breaker,
            max_attempts=self._max_attempts,
            initial_backoff=self._initial_backoff,
            max_backoff=self._max_backoff,
            egress=self._egress,
            # the caller's transport is shielded from this per-call client's
            # own close; see _UnclosableTransport.
            transport=_UnclosableTransport(self._http_transport) if self._http_transport is not None else None,
            timeout=self._timeout_seconds if self._timeout_seconds is not None else DEFAULT_HTTP_TIMEOUT_SECONDS,
        )
        try:
            # The bound is the whole call's, not one attempt's. Retry lives
            # *inside* the client, so forwarding ``timeout`` alone bounds each
            # attempt and a deadline-derived bound funds ``max_attempts`` of
            # them plus backoff -- a caller with 0.3s left waiting 1s+. SR-G2
            # says this argument carries the caller's remaining deadline, and
            # ``StandaloneTransport._perform`` already reads it that way; two
            # implementations of one protocol may not disagree about what an
            # argument means. Kept as ``asyncio.timeout`` rather than arithmetic
            # over the retry schedule because the ceiling then holds whatever
            # the client does internally, and the ``TimeoutError`` it raises is
            # what adapters already map onto ``TimedOut``.
            async with asyncio.timeout(timeout_seconds):
                response = await client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=dict(json_body) if json_body is not None else None,
                    timeout=timeout_seconds,
                )
        finally:
            await client.aclose()

        attempts = response.extensions.get(ATTEMPTS_EXTENSION, 1)
        return TransportResponse(
            status_code=response.status_code,
            body=response.content,
            final_url=str(response.url),
            egress=self.egress_name,
            elapsed_seconds=time.monotonic() - started,
            attempts=attempts if isinstance(attempts, int) else 1,
            headers={key.lower(): value for key, value in response.headers.items()},
        )


def build_fetch_transport(
    *,
    timeout_seconds: float | None = None,
    max_response_bytes: int | None = None,
    egress_name: str = EGRESS_DIRECT,
    allow_private_addresses: bool = False,
    max_redirects: int = DEFAULT_FETCH_MAX_REDIRECTS,
) -> StandaloneTransport:
    """The :class:`~threetears.search.contracts.transport.FetchTransport` this host injects.

    A named builder rather than a bare constructor call at each site, so the
    reason the fetch half is *not* the traced client (module docstring) has
    one place to be read, and so a deployment that needs different guards
    changes them once.

    :param timeout_seconds: default per-fetch timeout; None takes the
        standalone transport's own default
    :ptype timeout_seconds: float | None
    :param max_response_bytes: cap on one fetched body; None takes the
        standalone transport's own default. Extract passes its own per-call
        cap regardless, so this is the ceiling behind that one
    :ptype max_response_bytes: int | None
    :param egress_name: which exit these fetches leave by (D20)
    :ptype egress_name: str
    :param allow_private_addresses: whether a private or loopback address may
        be fetched. False by default -- these URLs come from search results,
        which is precisely the case the SSRF guard exists for (D21)
    :ptype allow_private_addresses: bool
    :param max_redirects: redirect hops a fetch may follow; the default is
        :data:`DEFAULT_FETCH_MAX_REDIRECTS` rather than the leaf's own 0,
        for the reason stated there. 0 restores the strict stance for a
        deployment that wants it
    :ptype max_redirects: int
    :return: the configured fetch transport
    :rtype: StandaloneTransport
    """
    return StandaloneTransport(
        timeout_seconds=timeout_seconds if timeout_seconds is not None else DEFAULT_TIMEOUT_SECONDS,
        max_response_bytes=max_response_bytes if max_response_bytes is not None else DEFAULT_MAX_RESPONSE_BYTES,
        egress_name=egress_name,
        allow_private_addresses=allow_private_addresses,
        max_redirects=max_redirects,
    )

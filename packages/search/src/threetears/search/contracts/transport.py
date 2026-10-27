"""SearchTransport -- the one injected transport seam (SR-N1, P9, D19, D21).

An adapter reaches an upstream through an injected transport, never a
client it opens itself. The protocol is *structural* so the two rules that
both refuse to bend can both hold: the family's sanctioned transport is
``threetears.core.http_client.TracedHttpClient``, and this leaf must not
import core (SR-L7). A thin host-side adapter over ``TracedHttpClient``
satisfies this shape for every host that has core (it lives with the host,
where core is already a hard dependency); hosts without core use the
``[standalone]`` extra's bare-httpx implementation -- the one module path
sanctioned by the D19 norm widening.

What an implementation owes, behind this one ``request`` method:

- **configurable timeout** -- never a constant (SR-G1); the per-call
  ``timeout_seconds`` override exists so a caller's remaining deadline can
  bound the call (SR-G2);
- **bounded retry with backoff** -- finite attempts, 5xx/connect/timeout
  retried, 4xx not (SR-G4); per-attempt accounting surfaces in
  :attr:`TransportResponse.attempts` so "budget follows the bill" can sit
  below the retry boundary (D4);
- **circuit-breaking** per upstream, so provider exhaustion
  short-circuits (SR-D3);
- **a per-call span** where telemetry exists, zero-cost where it does not
  (SR-I1);
- **egress selection** at construction (D20/SR-N2): which exit requests
  leave by is the transport's configuration, reported per response and as
  :attr:`SearchTransport.egress_name`;
- **the SSRF rulings** (D21): base URLs come from deployment config only,
  redirect policy and private-address guards live here, not per call site.

Ports are parameters, never payload (SR-L4): a transport is handed to
adapters and Call at construction and never appears in any wire type.

**Two protocols, decided at Gate A (2026-08-10).** Search responses are
small and fully buffered; Extract's fetches are arbitrary web content that
MUST be read against a byte cap and SHOULD be refusable on content type
before the body is pulled (SR-G5, §3.5). Those are different obligations,
so they are different protocols: widening :class:`SearchTransport` later
would retroactively invalidate every implementation written against the
Phase-1 shape, while a second protocol lets an implementation satisfy the
union and leaves search-only implementers conformant forever.

*Corrected 2026-08-11 (Phase 2 item 5).* Gate A expected the union's
implementers to be the standalone transport **and** the host-side
``TracedHttpClient`` adapter. Only the standalone transport implements both,
and the adapter structurally cannot: ``TracedHttpClient`` is constructed per
upstream -- one base URL that paths join onto, one breaker guarding it -- and
buffers whole bodies, while a fetch is a different host every call read
against a per-call cap. Hosts therefore inject *two* objects, and the split
is the reason declaring two protocols was right rather than merely cautious.

**A third protocol for an open-ended stream (AD11, 2026-10-23).** A cited
answer arrives as a stream whose citations exist only when streaming
(``answers-requirements-design.md`` §4.2), and whose length nobody states in
advance. Neither existing protocol can carry it: :class:`SearchTransport`
returns a buffered body, and :class:`FetchTransport` returns one too, read
whole under its cap. A stream differs in what the caller holds -- an open
response it reads chunk by chunk, with the status and headers in hand before
the first chunk -- and in one obligation the others never had to state: a
retry is only honest **before the first byte**, because once body bytes have
reached the caller the provider has started answering (and, for a metered
provider, billing) and a second attempt would be a second bill. So it is a
third protocol, :class:`StreamingTransport`, for the Gate A reason the second
one was: folding ``stream`` into either existing protocol would invalidate
every implementation already written against it, while a third leaves them
conformant and lets an implementation satisfy whichever union it can. The
host-side ``TracedHttpClient`` adapter cannot implement it, for the reason it
cannot implement :class:`FetchTransport`: it buffers. The transport moves
bytes and nothing more; any framing inside them (SSE ``data:`` lines, a
provider's tags) is the adapter's to parse (AR-F2).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from pydantic import JsonValue

__all__ = [
    "FetchTransport",
    "HeavyFetcher",
    "SearchTransport",
    "StreamedResponse",
    "StreamingTransport",
    "TransportResponse",
    "media_type_accepted",
]


def media_type_accepted(media_type: str | None, allowed: Sequence[str]) -> bool:
    """Whether a declared media type passes a content-type gate.

    An entry is a prefix (``text/`` takes every text type, ``application/json`` takes
    itself), or, when it starts with ``+``, a structured-syntax suffix (RFC 6839):
    ``+json`` takes ``application/ld+json`` and ``application/vnd.api+json``, which no
    prefix can name. Case is ignored, and a type that declares nothing passes nothing.

    :param media_type: the bare media type the response declared, without parameters
    :ptype media_type: str | None
    :param allowed: the gate's entries
    :ptype allowed: Sequence[str]
    :return: True when some entry takes the type
    :rtype: bool
    """
    lowered = (media_type or "").strip().lower()
    if not lowered:
        return False
    for entry in allowed:
        wanted = entry.lower()
        if lowered.endswith(wanted) if wanted.startswith("+") else lowered.startswith(wanted):
            return True
    return False


@dataclass(frozen=True, slots=True)
class TransportResponse:
    """What one transport call yielded, with per-attempt accounting.

    A seam value, not a wire type: it crosses the transport/adapter seam
    in process and never rides a result payload, so it is a plain frozen
    dataclass rather than a :class:`ContractModel` (``body`` is raw bytes).
    """

    #: final HTTP status code.
    status_code: int
    #: response body bytes. Adapters parse; the transport never interprets.
    body: bytes
    #: the URL that actually answered, after any redirects the
    #: implementation's policy allowed (D21).
    final_url: str
    #: which egress the request left by (D20) -- ``direct`` is a named
    #: value, never an absence.
    egress: str
    #: wall-clock the call took, across all attempts, in seconds.
    elapsed_seconds: float
    #: attempts made, including the successful one (SR-G4). Spend's
    #: bill-following rule reads this: retried-but-unbilled attempts
    #: never count as billed calls (D4).
    attempts: int = 1
    #: response headers, case-insensitivity already normalised to lower-case
    #: keys by the implementation.
    headers: Mapping[str, str] = field(default_factory=dict)


@runtime_checkable
class SearchTransport(Protocol):
    """Structural protocol for the injected outbound transport.

    Implementations: the host's thin adapter over
    ``threetears.core.http_client.TracedHttpClient`` (embedded or
    pod-resident hosts that have core), or this package's ``standalone``
    module (hosts that inject nothing). Adapters and Call depend on this
    shape only.
    """

    @property
    def egress_name(self) -> str:
        """Name of the egress this transport's requests leave by (D20).

        :return: the configured exit's name; ``direct`` for the default
            route
        :rtype: str
        """
        ...

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
        """Perform one bounded, traced HTTP exchange.

        :param method: HTTP method (``GET``, ``POST``, ...)
        :ptype method: str
        :param url: absolute URL on a deployment-configured base -- a
            transport must never accept a caller-supplied base URL (D21)
        :ptype url: str
        :param headers: request headers, if any
        :ptype headers: Mapping[str, str] | None
        :param params: query parameters, if any
        :ptype params: Mapping[str, str] | None
        :param json_body: JSON request body, if any; mutually exclusive
            with other body kinds by construction of this seam
        :ptype json_body: Mapping[str, JsonValue] | None
        :param timeout_seconds: per-call override of the configured
            timeout, for deadline-derived bounds (SR-G2); None uses the
            implementation's configured value (SR-G1)
        :ptype timeout_seconds: float | None
        :return: the completed exchange with per-attempt accounting
        :rtype: TransportResponse
        :raises Exception: implementations surface transport-level
            failures however they choose; adapters map them onto the
            typed taxonomy in :mod:`threetears.search.contracts.errors`
            (SR-J1) -- the taxonomy mapping is adapter business, not
            transport business
        """
        ...


@runtime_checkable
class FetchTransport(Protocol):
    """Structural protocol for the byte-capped content fetch Extract requires.

    Declared at Gate A (2026-08-10), implemented when Extract lands (Phase
    2), so no Phase-1 :class:`SearchTransport` implementer is ever
    retroactively non-conformant: an implementation that also serves
    Extract satisfies both protocols; one that only serves search never
    has to. The one implementer of the union is this package's ``standalone``
    module -- Gate A also expected the host-side ``TracedHttpClient``
    adapter, which the module docstring's 2026-08-11 correction records as
    structurally impossible.

    What :meth:`fetch` owes beyond :meth:`SearchTransport.request`'s
    obligations (which all still apply -- bounded retry, configurable
    timeout, egress, the D21 guards):

    - **the byte cap is per call and load-bearing** (SR-G5): the body is
      read incrementally against ``max_bytes`` and an overrun is refused
      as the implementation's local-cap failure -- at no point may more
      than ``max_bytes`` of body be held, declared lengths past the cap
      are refused before a byte is read, and lying lengths are caught
      chunk by chunk;
    - **the content-type gate refuses before the body** : when
      ``allowed_content_types`` is given and the response's declared
      media type matches none of them (:func:`media_type_accepted`: a
      prefix, or a ``+json``-style suffix), the implementation refuses
      without reading the body -- the gate exists so Extract never pulls
      megabytes of video to learn it wanted text (§3.5).
    """

    @property
    def egress_name(self) -> str:
        """Name of the egress this transport's requests leave by (D20).

        :return: the configured exit's name; ``direct`` for the default
            route
        :rtype: str
        """
        ...

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
        """Perform one bounded, byte-capped content read.

        :param method: HTTP method (``GET``, ``HEAD``)
        :ptype method: str
        :param url: absolute URL to fetch. Unlike search calls this is
            candidate-derived, which is exactly why the D21 guards
            (redirect policy, private-address refusal) bind hardest here
        :ptype url: str
        :param headers: request headers, if any
        :ptype headers: Mapping[str, str] | None
        :param max_bytes: hard cap on the body for this call (SR-G5).
            Required and per-call: Extract's cap is the caller's memory
            budget, not a transport constant
        :ptype max_bytes: int
        :param allowed_content_types: media-type prefixes this read will
            accept (e.g. ``("text/html", "application/xhtml+xml")``), and
            ``+json``-style suffixes (:func:`media_type_accepted`);
            None accepts anything. A declared type outside the gate is
            refused before the body is read
        :ptype allowed_content_types: tuple[str, ...] | None
        :param timeout_seconds: per-call override of the configured
            timeout (SR-G2); None uses the configured value (SR-G1)
        :ptype timeout_seconds: float | None
        :return: the completed exchange; ``body`` never exceeds
            ``max_bytes``
        :rtype: TransportResponse
        :raises Exception: implementations surface failures however they
            choose (the standalone implementation speaks the typed
            taxonomy: cap and gate refusals as ``LocalCapExceeded`` with
            the refusing scope named); Extract maps everything else onto
            the taxonomy (SR-J1)
        """
        ...


@runtime_checkable
class StreamedResponse(Protocol):
    """One open streamed response: the head now, the body as it arrives.

    Not a value like :class:`TransportResponse`: it holds a live connection,
    which is why it is a protocol rather than a frozen dataclass. The head --
    :attr:`status_code`, :attr:`headers`, :attr:`final_url`, :attr:`egress`,
    :attr:`attempts` -- is complete when :meth:`StreamingTransport.stream`
    returns, before any body byte is read, so a caller can refuse an error
    status without reading what follows it. The body is an async iterator of
    ``bytes`` chunks, each yielded as it arrives, and it is iterated once.

    **The connection is always released.** Use it as an async context manager,
    or call :meth:`aclose`; either releases the connection whether the body was
    read to the end, abandoned part-way, or broken by a failure. An exhausted
    or failed body releases it too. :meth:`aclose` is idempotent.

    A failure while the body is read -- the byte cap, the deadline, a broken
    connection -- raises from the iteration, typed as the implementation's
    other failures are, and is never retried (see
    :class:`StreamingTransport`).
    """

    @property
    def status_code(self) -> int:
        """Final HTTP status code, available before the body.

        :return: the status of the response that answered, after any
            redirects the implementation's policy allowed (D21)
        :rtype: int
        """
        ...

    @property
    def headers(self) -> Mapping[str, str]:
        """Response headers with lower-cased keys, available before the body.

        :return: the headers of the response that answered
        :rtype: Mapping[str, str]
        """
        ...

    @property
    def final_url(self) -> str:
        """The URL that actually answered, after any redirects followed.

        :return: the answering URL
        :rtype: str
        """
        ...

    @property
    def egress(self) -> str:
        """Which egress the request left by (D20) -- ``direct`` is a named value.

        :return: the egress name
        :rtype: str
        """
        ...

    @property
    def attempts(self) -> int:
        """Attempts made to open this stream, including the one that answered.

        Final once :meth:`StreamingTransport.stream` returns: no attempt is
        ever made after the first body byte, so the count cannot move while
        the body is read (D4, AR-E1).

        :return: the attempt count, at least 1
        :rtype: int
        """
        ...

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock the call has taken, across every attempt and the body.

        Final once the stream is closed; while it is open, the time so far.

        :return: elapsed seconds
        :rtype: float
        """
        ...

    def __aiter__(self) -> AsyncIterator[bytes]:
        """Iterate the body, chunk by chunk, as it arrives.

        :return: the body's chunks, none empty
        :rtype: AsyncIterator[bytes]
        """
        ...

    async def aclose(self) -> None:
        """Release the connection. Idempotent; safe after a failure or exhaustion."""
        ...

    async def __aenter__(self) -> Self:
        """Enter the block that owns this response.

        :return: this response
        :rtype: Self
        """
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the connection on leaving the block, however it is left.

        :param exc_type: exception type, if the block raised
        :ptype exc_type: type[BaseException] | None
        :param exc: the exception, if the block raised
        :ptype exc: BaseException | None
        :param traceback: the traceback, if the block raised
        :ptype traceback: TracebackType | None
        """
        ...


@runtime_checkable
class StreamingTransport(Protocol):
    """Structural protocol for an open-ended response read as a stream (AD11).

    The seam a streaming provider (a cited answer, AD5) reads through. The one
    implementer is this package's ``standalone`` module, beside the two
    protocols it already implements; the host-side ``TracedHttpClient``
    adapter cannot implement it, because it buffers.

    What an implementation owes, behind this one :meth:`stream` method -- the
    obligations :class:`SearchTransport` states, restated where a stream
    changes their meaning:

    - **configurable timeout** (SR-G1, SR-G2) -- the per-call
      ``timeout_seconds`` bounds the **whole stream**: the attempts, the
      backoff between them, and the reading of the body. A stream bounded only
      until its headers could hold a caller forever on a trickle;
    - **bounded retry, before the first byte only** (SR-G4, AR-E1) -- connect
      failures, timeouts and 5xx answers are retried, with backoff and a finite
      attempt ceiling, while no body byte has reached the caller; once one has,
      nothing is retried, because the provider has begun answering and a
      second attempt would be a second bill (D4). A body broken after the
      first byte fails with the attempts already counted;
    - **circuit-breaking** per upstream (SR-D3), as for :class:`SearchTransport`;
    - **a per-call span** where telemetry exists, zero-cost where it does not
      (SR-I1);
    - **egress selection** at construction (D20), reported as
      :attr:`egress_name` and on every response;
    - **the SSRF rulings** (D21) -- base URLs from deployment config only, the
      redirect policy and private-address guards here and not per call site,
      and a direct connection opened to the address the guard checked;
    - **a byte cap on the whole stream** (SR-G5, AR-F1) -- ``max_bytes``
      bounds the sum of every chunk, counted incrementally as each arrives. A
      declared length past the cap is refused before a byte is read; a stream
      that runs past it is refused as the implementation's local-cap failure
      (the ``fetch`` refusal, scope ``response-bytes``) without yielding the
      chunk that crossed it, so the caller never receives more than the cap.

    Framing is not the transport's business (AR-F2): it yields bytes as the
    socket delivers them, and a chunk boundary means nothing. SSE events,
    ``[DONE]`` and a provider's inline tags are parsed by the adapter, which
    must tolerate any of them straddling chunks.
    """

    @property
    def egress_name(self) -> str:
        """Name of the egress this transport's requests leave by (D20).

        :return: the configured exit's name; ``direct`` for the default
            route
        :rtype: str
        """
        ...

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
        """Open one bounded, guarded stream and return it with its head read.

        :param method: HTTP method (``POST`` for a streamed completion)
        :ptype method: str
        :param url: absolute URL on a deployment-configured base -- never a
            caller-supplied base (D21)
        :ptype url: str
        :param headers: request headers, if any
        :ptype headers: Mapping[str, str] | None
        :param json_body: JSON request body, if any
        :ptype json_body: Mapping[str, JsonValue] | None
        :param timeout_seconds: bound on the whole stream, body included
            (SR-G2); None uses the implementation's configured value (SR-G1)
        :ptype timeout_seconds: float | None
        :param max_bytes: hard cap on the whole body (SR-G5). Required and
            per-call, as on :meth:`FetchTransport.fetch`: the cap is the
            caller's budget for this response, not a transport constant
        :ptype max_bytes: int
        :return: the open response, head complete and body unread. The caller
            owns it and must release it (``async with`` or ``aclose``)
        :rtype: StreamedResponse
        :raises Exception: implementations surface failures however they
            choose (the standalone implementation speaks the typed taxonomy:
            ``TimedOut``, ``TransportFailed``, and ``LocalCapExceeded`` with
            the refusing scope named); adapters map them onto the taxonomy
            (SR-J1)
        """
        ...


@runtime_checkable
class HeavyFetcher(Protocol):
    """The escalation slot for carriers an ordinary fetch cannot read.

    A page behind a renderer, a bot wall, or a session is not a
    :class:`FetchTransport` problem -- it needs a browser, which is
    ``3tears-scrape``'s business and never this package's (§3.5: the leaf
    MUST NOT import scrape). So the seam is a slot: scrape implements this
    shape, a host constructs one, and a caller hands it to Extract for the
    candidate it decided was worth the cost.

    **Escalation is never automatic** (§3.5, ruled 2026-08-11). The method
    is deliberately named apart from :meth:`FetchTransport.fetch` so an
    ordinary transport cannot satisfy this protocol by accident and get
    used as one -- a heavy fetch can cost orders of magnitude more than a
    plain one, and silent escalation multiplies that by a factor nobody
    sees until the bill.
    """

    async def fetch_rendered(
        self,
        url: str,
        *,
        max_bytes: int,
        timeout_seconds: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> TransportResponse:
        """Fetch a carrier that needs rendering, and return it as bytes.

        The byte cap binds exactly as it does on :meth:`FetchTransport.fetch`:
        an implementation MUST NOT hold more than ``max_bytes`` of body.

        **An implementation that IGNORES ``headers`` MUST NOT report 304.**
        This is the one rule on this method that is a correctness rule rather
        than a cost rule. A caller sends ``If-None-Match`` because it holds a
        copy it will keep if told nothing changed; an implementation that
        drops the header, fetches unconditionally, and then reports ``304``
        for any other reason hands that caller stale bytes under a validator
        it supplied. Ignoring the parameter entirely is fine and is the
        conformant default -- answering ``200`` with the body is always a
        correct answer to a conditional request.

        :param url: absolute URL to render and read
        :ptype url: str
        :param max_bytes: hard cap on the returned body (SR-G5)
        :ptype max_bytes: int
        :param timeout_seconds: per-call bound; None uses the
            implementation's configured value (SR-G1)
        :ptype timeout_seconds: float | None
        :param headers: request headers to send, carrying the caller's
            conditional validators (``If-None-Match`` / ``If-Modified-Since``)
            when it holds a copy worth revalidating (D30 / SR-M4). Additive
            with a default, so an existing implementer stays conformant until
            it chooses to honour it
        :ptype headers: Mapping[str, str] | None
        :return: the rendered document, with ``body`` within the cap
        :rtype: TransportResponse
        :raises Exception: implementations surface failures however they
            choose; Extract maps them onto the taxonomy (SR-J1)
        """
        ...

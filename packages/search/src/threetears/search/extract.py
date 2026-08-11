"""Extract -- a carrier, to the information in it (§3.5, Phase 2 item 4).

One candidate in, the same candidate out with its content slot filled and
its extraction status recorded. Carrier dispatch -- PDFs, images, datasets
-- is Phase 3; this module is the web path, and it is deliberately the only
path that exists so far rather than a dispatcher with one arm.

**A candidate that already carries content is untouched, and costs
nothing** (SR-A2). Tavily returns page text with the search response; the
bytes are bought and re-fetching them would pay twice for a worse copy.
This is checked before anything else, including robots -- a fetch that will
not happen needs no permission.

**Everything reaches the network through the injected
:class:`~threetears.search.contracts.transport.FetchTransport`** (P9, SR-L4,
D19): the page, and ``robots.txt`` too. A fetched URL is candidate-derived
-- the one class of URL in this package a caller can influence without
touching deployment config -- so the D21 guards bind hardest here, and a
second path to the same hosts would be a second, weaker set of them.

**Per-candidate outcomes are recorded, not raised.** A page that 404s, a
cap that refuses, a robots file that says no: each comes back as a
candidate whose ``extraction_status`` facet says what happened, because one
unreadable page must never take down the extraction of a set (SR-H3's
sibling rule applied to carriers). What *does* raise is a configuration
defect that would refuse every candidate identically -- the missing
``[extract]`` extra below -- because marking a hundred candidates
``refused`` one at a time hides a single fixable fault behind a hundred
plausible ones.

Rulings taken in this build, recorded here per the Gate A precedent:

- **A missing ``[extract]`` extra raises**
  :class:`~threetears.search.contracts.errors.LocalCapExceeded` **with
  scope** :data:`EXTRACTOR_UNAVAILABLE_SCOPE`. It is a local refusal that
  the provider never saw, which is that class's definition, and ``scope``
  already doubles as refusal identity across this package
  (``query-length``, ``response-bytes``, ``content-type``). The honest
  alternative is an eighth taxonomy class -- ``CapabilityUnavailable`` --
  and it was not taken mid-build: it is a contract change, D29's window
  makes it a cheap one for now, and it should be a ruling somebody takes
  deliberately rather than a side effect of writing this module.
- **Robots follows RFC 9309 on its own failures.** A 4xx for ``robots.txt``
  means no rules exist, so the fetch proceeds; a 5xx or a transport failure
  means the rules are unknown, and unknown rules are honoured as *deny*.
  Guessing "allowed" from a server error is how a polite crawler becomes an
  impolite one during an outage.
- **Escalation is a parameter, not a fallback.** Passing ``heavy_fetcher``
  says *use it for this candidate*; Extract never reaches for it after an
  ordinary fetch fails. A caller who wants that policy writes it -- see
  :class:`~threetears.search.contracts.transport.HeavyFetcher`.
- **The robots memo lives and dies inside one call** (§3.5). It is not a
  response cache and D14 is untouched. A batch that wants one lookup per
  host across many candidates needs a memo whose lifetime is the batch,
  which is a wider scope than the ruling sanctions -- so it is deliberately
  not offered here, and belongs with Phase 3's carrier dispatch, where
  something owns the set.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from threetears.media.contracts import (
    EXTRACTION_STATUS_COMPLETE,
    EXTRACTION_STATUS_FAILED,
    EXTRACTION_STATUS_REFUSED,
)

from threetears.search.contracts.candidate import Candidate, ContentSlot
from threetears.search.contracts.errors import LocalCapExceeded, SearchFailure
from threetears.search.contracts.fidelity import FIDELITY_CONTENT
from threetears.search.contracts.spend import Spend
from threetears.search.contracts.transport import FetchTransport, HeavyFetcher, TransportResponse

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_USER_AGENT",
    "EXTRACTION_METHOD_FACET",
    "EXTRACTION_STATUS_FACET",
    "EXTRACTOR_UNAVAILABLE_SCOPE",
    "HTML_CONTENT_TYPES",
    "extract",
]

#: Default cap on one fetched carrier. Two megabytes reads the long tail of
#: real article pages and refuses the video somebody linked as "the report".
#: A parameter with a default rather than a constant at the call site
#: (SR-G1); a host with a tighter ``MemoryMax`` passes its own.
DEFAULT_MAX_BYTES: Final[int] = 2 * 1024 * 1024

#: What this package calls itself to a robots file. Named, because
#: ``robots.txt`` rules are written against user agents and a fetcher that
#: will not say who it is cannot be given rules to follow.
DEFAULT_USER_AGENT: Final[str] = "3tears-search"

#: Media types the web path will read. The gate refuses anything else
#: *before* the body (§3.1), which is the whole reason Extract fetches
#: through :class:`~threetears.search.contracts.transport.FetchTransport`
#: rather than ``SearchTransport``.
HTML_CONTENT_TYPES: Final[tuple[str, ...]] = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
)

#: :attr:`~threetears.search.contracts.candidate.Candidate.facets` key
#: carrying the ``media-contracts`` ``EXTRACTION_STATUS_*`` vocabulary.
EXTRACTION_STATUS_FACET: Final[str] = "extraction_status"

#: :attr:`~threetears.search.contracts.candidate.Candidate.facets` key
#: carrying *how* the text was produced (SR-B6 -- fidelity achieved is only
#: half the answer; a consumer comparing two extractions needs to know
#: whether a renderer was involved). Search-owned for now: ``media-contracts``
#: names the status vocabulary but not the method one, and promoting this
#: key belongs with the second producer that needs it, not the first.
EXTRACTION_METHOD_FACET: Final[str] = "extraction_method"

#: Refusal identity for a call the ``[extract]`` extra is not installed for.
EXTRACTOR_UNAVAILABLE_SCOPE: Final[str] = "extractor-unavailable"

#: What the ``[extract]`` extra installs, named in the refusal so the fix is
#: in the message rather than in a doc the reader has to find.
_EXTRACT_EXTRA: Final[str] = "3tears-search[extract]"

#: Method name recorded when trafilatura produced the text.
_METHOD_TRAFILATURA: Final[str] = "trafilatura"

#: Method name recorded when a caller's heavy fetcher produced the carrier.
_METHOD_HEAVY: Final[str] = "trafilatura+rendered"

#: The shape :func:`_load_extractor` returns: markup in, article text out,
#: ``None`` when the extractor found nothing worth keeping. Named so the
#: lazily-imported dependency has a type at the seam rather than an
#: ``object`` the call site has to cast.
type _Extractor = Callable[[str], str | None]


async def extract(
    candidate: Candidate,
    *,
    transport: FetchTransport,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout_seconds: float | None = None,
    respect_robots: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
    heavy_fetcher: HeavyFetcher | None = None,
) -> Candidate:
    """Fill one candidate's content slot from its carrier.

    :param candidate: the candidate to extract. Returned untouched when it
        already carries content (SR-A2)
    :ptype candidate: Candidate
    :param transport: the injected byte-capped fetch seam; used for the
        carrier and for ``robots.txt``
    :ptype transport: FetchTransport
    :param max_bytes: hard cap on the fetched carrier (SR-G5)
    :ptype max_bytes: int
    :param timeout_seconds: per-fetch bound; None uses the transport's
        configured value (SR-G1)
    :ptype timeout_seconds: float | None
    :param respect_robots: honour ``robots.txt`` (D12's default). The
        override exists because D12 states it as recorded deployment
        config; it is a parameter so config can reach it, and it is not a
        knob a call site should be choosing per call
    :ptype respect_robots: bool
    :param user_agent: the agent name robots rules are matched against
    :ptype user_agent: str
    :param heavy_fetcher: when given, the carrier is fetched through it
        instead of through ``transport`` -- the caller's explicit choice for
        this candidate, never an automatic fallback
    :ptype heavy_fetcher: HeavyFetcher | None
    :return: the candidate with content, fidelity and extraction facets
        recorded; on a per-candidate failure, the candidate with its status
        facet saying so and no content
    :rtype: Candidate
    :raises threetears.search.contracts.errors.LocalCapExceeded: when the
        ``[extract]`` extra is not installed, so no candidate in this run
        can be extracted at all
    """
    if candidate.content is not None:
        return candidate

    url = _carrier_url(candidate)
    if url is None:
        return _marked(candidate, EXTRACTION_STATUS_FAILED)

    extractor = _load_extractor()

    if (
        respect_robots
        and heavy_fetcher is None
        and not await _robots_allow(url, transport=transport, user_agent=user_agent, timeout_seconds=timeout_seconds)
    ):
        return _marked(candidate, EXTRACTION_STATUS_REFUSED)

    try:
        response = await _fetch_carrier(
            url,
            transport=transport,
            heavy_fetcher=heavy_fetcher,
            max_bytes=max_bytes,
            timeout_seconds=timeout_seconds,
        )
    except LocalCapExceeded:
        # A cap declined the read under rules that will decline it again --
        # which is what ``refused`` means, and what separates it from a
        # fetch that tried and broke.
        return _marked(candidate, EXTRACTION_STATUS_REFUSED)
    except SearchFailure:
        return _marked(candidate, EXTRACTION_STATUS_FAILED)

    if not 200 <= response.status_code < 300:
        return _marked(candidate, EXTRACTION_STATUS_FAILED)

    text = extractor(response.body.decode("utf-8", errors="replace"))
    if not text:
        return _marked(candidate, EXTRACTION_STATUS_FAILED)

    method = _METHOD_HEAVY if heavy_fetcher is not None else _METHOD_TRAFILATURA
    return candidate.model_copy(
        update={
            "content": ContentSlot(
                text=text,
                origin="later-fetch",
                mime_type=_declared_type(response),
                size_bytes=len(response.body),
            ),
            "fidelity_achieved": FIDELITY_CONTENT,
            "facets": {
                **candidate.facets,
                EXTRACTION_STATUS_FACET: EXTRACTION_STATUS_COMPLETE,
                EXTRACTION_METHOD_FACET: method,
            },
        }
    )


def _carrier_url(candidate: Candidate) -> str | None:
    """The locator the web path should read.

    Prefers the canonical locator and falls back to the first one given: a
    candidate whose only locator is a thumbnail still has an address, and
    refusing it here would be this module deciding what a carrier is worth.

    :param candidate: the candidate being extracted
    :ptype candidate: Candidate
    :return: the URL to fetch, or None when the candidate carries no
        locator at all
    :rtype: str | None
    """
    for locator in candidate.locators:
        if locator.rel == "canonical":
            return locator.url
    return candidate.locators[0].url if candidate.locators else None


def _load_extractor() -> _Extractor:
    """The installed HTML-to-text extractor, or a typed refusal.

    Imported here rather than at module scope so importing
    :mod:`threetears.search` never pays for trafilatura -- the package's
    import-cost pin holds the leaf to its declared floor, and the extractor
    rides the ``[extract]`` extra.

    :return: trafilatura's ``extract`` callable
    :rtype: _Extractor
    :raises threetears.search.contracts.errors.LocalCapExceeded: when the
        extra is not installed, naming it
    """
    try:
        from trafilatura import extract as _trafilatura_extract
    except ImportError as exc:
        raise LocalCapExceeded(
            f"extraction requires the {_EXTRACT_EXTRA} extra, which is not installed",
            spend=Spend(),
            remediation=f"install {_EXTRACT_EXTRA}",
            scope=EXTRACTOR_UNAVAILABLE_SCOPE,
        ) from exc
    return _trafilatura_extract


async def _fetch_carrier(
    url: str,
    *,
    transport: FetchTransport,
    heavy_fetcher: HeavyFetcher | None,
    max_bytes: int,
    timeout_seconds: float | None,
) -> TransportResponse:
    """Read the carrier through whichever fetcher the caller chose.

    :param url: the carrier's address
    :ptype url: str
    :param transport: the ordinary byte-capped seam
    :ptype transport: FetchTransport
    :param heavy_fetcher: the caller's escalation, when they passed one
    :ptype heavy_fetcher: HeavyFetcher | None
    :param max_bytes: hard cap on the body (SR-G5)
    :ptype max_bytes: int
    :param timeout_seconds: per-fetch bound
    :ptype timeout_seconds: float | None
    :return: the fetched carrier
    :rtype: TransportResponse
    """
    if heavy_fetcher is not None:
        # No content-type gate on the heavy path: a renderer is asked for a
        # rendered document and answers with one, so there is no cheap
        # declaration to refuse on before paying.
        return await heavy_fetcher.fetch_rendered(url, max_bytes=max_bytes, timeout_seconds=timeout_seconds)
    return await transport.fetch(
        "GET",
        url,
        max_bytes=max_bytes,
        allowed_content_types=HTML_CONTENT_TYPES,
        timeout_seconds=timeout_seconds,
    )


async def _robots_allow(
    url: str,
    *,
    transport: FetchTransport,
    user_agent: str,
    timeout_seconds: float | None,
) -> bool:
    """Whether the host's ``robots.txt`` permits fetching ``url`` (D12).

    Fetched through the same seam as the carrier, so it inherits the same
    guards, cap, pacing and egress rather than acquiring a second, weaker
    path to the same host.

    RFC 9309's failure posture: 4xx means no rules exist and the fetch
    proceeds; 5xx or a transport failure means the rules are unknown, and
    unknown rules are honoured as deny.

    :param url: the carrier URL whose permission is in question
    :ptype url: str
    :param transport: the injected fetch seam
    :ptype transport: FetchTransport
    :param user_agent: the agent name rules are matched against
    :ptype user_agent: str
    :param timeout_seconds: per-fetch bound
    :ptype timeout_seconds: float | None
    :return: True when the fetch may proceed
    :rtype: bool
    """
    robots_url = urljoin(f"{urlsplit(url).scheme}://{urlsplit(url).netloc}", "/robots.txt")
    try:
        response = await transport.fetch(
            "GET",
            robots_url,
            max_bytes=_ROBOTS_MAX_BYTES,
            allowed_content_types=("text/plain",),
            timeout_seconds=timeout_seconds,
        )
    except SearchFailure:
        return False

    if 400 <= response.status_code < 500:
        return True
    if not 200 <= response.status_code < 300:
        return False

    parser = RobotFileParser()
    parser.parse(response.body.decode("utf-8", errors="replace").splitlines())
    return parser.can_fetch(user_agent, url)


#: Cap on a ``robots.txt`` read. Google's own limit, and generous: a robots
#: file past half a megabyte is not a robots file.
_ROBOTS_MAX_BYTES: Final[int] = 512 * 1024


def _declared_type(response: TransportResponse) -> str | None:
    """The response's media type, without its parameters.

    :param response: the fetched carrier
    :ptype response: TransportResponse
    :return: the bare media type, or None when the response declared none
    :rtype: str | None
    """
    declared = response.headers.get("content-type", "")
    media_type = declared.split(";", 1)[0].strip()
    return media_type or None


def _marked(candidate: Candidate, status: str) -> Candidate:
    """The candidate with its extraction status recorded and no content.

    :param candidate: the candidate whose extraction did not produce text
    :ptype candidate: Candidate
    :param status: the ``media-contracts`` status to record
    :ptype status: str
    :return: a copy carrying the status facet
    :rtype: Candidate
    """
    return candidate.model_copy(update={"facets": {**candidate.facets, EXTRACTION_STATUS_FACET: status}})

"""Extract -- a carrier, to the information in it (§3.5, Phase 2 item 4).

One candidate in, the same candidate out with its content slot filled and
its extraction status recorded. The web path reads HTML here; any other
carrier is read by a :class:`CarrierReader` the caller passes, so this
package never imports a document library (a PDF reader lives with the host
that has one).

**A candidate that already carries content is untouched, and costs
nothing** (SR-A2). Tavily returns page text with the search response; the
bytes are bought and re-fetching them would pay twice for a worse copy.
This is checked before anything else, including robots -- a fetch that will
not happen needs no permission.

**Everything reaches the network through the injected
:class:`~threetears.search.contracts.transport.FetchTransport`** (P9, SR-L4,
D19): the page, ``robots.txt``, and a thin page's frame too. A fetched URL is
candidate-derived -- the one class of URL in this package a caller can
influence without touching deployment config -- so the D21 guards bind
hardest here, and a second path to the same hosts would be a second, weaker
set of them.

**Per-candidate outcomes are recorded, not raised.** A page that 404s, a
cap that refuses, a robots file that says no: each comes back as a
candidate whose ``extraction_status`` facet says what happened, with the
reason and the next step in words (:data:`EXTRACTION_REASON_FACET`,
:data:`EXTRACTION_NEXT_STEP_FACET`), because one unreadable page must never
take down the extraction of a set. What *does* raise is a configuration
defect that would refuse every candidate identically -- the missing
``[extract]`` extra below.

Rulings taken in this module, recorded here per the Gate A precedent:

- **A missing ``[extract]`` extra raises**
  :class:`~threetears.search.contracts.errors.LocalCapExceeded` **with
  scope** :data:`EXTRACTOR_UNAVAILABLE_SCOPE`.
- **Robots fails closed, and says which way it failed** (D12). A 4xx for
  ``robots.txt`` means no rules exist, so the fetch proceeds. A 2xx is
  parsed whatever content type it declares: a site that serves its app
  shell for every path answers ``/robots.txt`` with ``text/html``, and
  refusing that as "the site refused" told the reader something false. A
  5xx, a timeout or a transport failure means the rules are *unknown*: the
  candidate is recorded ``failed`` with "robots.txt could not be read
  (...)" and "try again", never ``refused``, because the site said no such
  thing. Whether unknown rules may be read past is one deployment setting
  (``robots_unreadable``), ``deny`` by default, never a per-call choice.
- **The robots read has its own bound**,
  :data:`DEFAULT_ROBOTS_TIMEOUT_SECONDS`, apart from the page's: a robots
  file that hangs must not spend the page's time.
- **A thin page may follow one frame, once.** When the caller asks
  (``thin_page_chars``), a page whose readable text is shorter than that
  and which points at its article -- a ``citation_pdf_url`` meta, or a
  single ``iframe`` -- is followed one level, only on the same registrable
  site, through this transport, with robots asked for the frame's URL, out
  of the byte cap and the time budget the first fetch already drew on.
  Anything else is recorded ``failed`` with "only the page's menu was
  readable; the article is at <url>", and the little the page did hold is
  kept on :data:`EXTRACTION_THIN_TEXT_FACET` so nothing is lost.
- **Escalation is a parameter, not a fallback.** Passing ``heavy_fetcher``
  says *use it for this candidate*; Extract never reaches for it after an
  ordinary fetch fails.
- **The robots memo lives and dies inside one call** (§3.5). It is not a
  response cache and D14 is untouched.
"""

from __future__ import annotations

import codecs
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Final, Literal
from urllib.parse import quote, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from pydantic import JsonValue

from threetears.media.contracts import (
    EXTRACTION_STATUS_COMPLETE,
    EXTRACTION_STATUS_FAILED,
    EXTRACTION_STATUS_REFUSED,
    EXTRACTION_STATUS_UNCHANGED,
)
from threetears.search.contracts.candidate import Candidate, ContentSlot
from threetears.search.contracts.errors import LocalCapExceeded, SearchFailure, TimedOut
from threetears.search.contracts.fidelity import FIDELITY_CONTENT
from threetears.search.contracts.spend import Spend
from threetears.search.contracts.transport import FetchTransport, HeavyFetcher, TransportResponse

__all__ = [
    "DEFAULT_FETCH_TIME_BUDGET_SECONDS",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_ROBOTS_TIMEOUT_SECONDS",
    "DEFAULT_THIN_PAGE_CHARS",
    "DEFAULT_USER_AGENT",
    "EXTRACTION_CHARSET_FACET",
    "EXTRACTION_FINAL_URL_FACET",
    "EXTRACTION_METHOD_FACET",
    "EXTRACTION_NEXT_STEP_FACET",
    "EXTRACTION_NOTES_FACET",
    "EXTRACTION_REASON_FACET",
    "EXTRACTION_STATUS_FACET",
    "EXTRACTION_THIN_TEXT_FACET",
    "EXTRACTOR_UNAVAILABLE_SCOPE",
    "HTML_CONTENT_TYPES",
    "ROBOTS_UNREADABLE_ALLOW",
    "ROBOTS_UNREADABLE_DENY",
    "CarrierReader",
    "CarrierText",
    "CarrierUnreadable",
    "RobotsUnreadable",
    "extract",
    "robots_agent_token",
    "same_site",
]

#: Default cap on one fetched page. Two megabytes reads the long tail of
#: real article pages and refuses the video somebody linked as "the report".
DEFAULT_MAX_BYTES: Final[int] = 2 * 1024 * 1024

#: What this package calls itself to a robots file when the caller names no
#: one. A deployment passes its own name (``user_agent``), and the transport
#: it injects sends the same one.
DEFAULT_USER_AGENT: Final[str] = "3tears-search"

#: How long the read of ``robots.txt`` may take, on its own bound.
DEFAULT_ROBOTS_TIMEOUT_SECONDS: Final[float] = 5.0

#: The time a fetch and a thin page's frame share, when the caller states no
#: bound of its own: what is left of it is all the frame may spend.
DEFAULT_FETCH_TIME_BUDGET_SECONDS: Final[float] = 30.0

#: Readable text under this many characters makes a page thin, for a caller
#: that turns frame-following on and names no size of its own.
DEFAULT_THIN_PAGE_CHARS: Final[int] = 1000

#: Unreadable robots rules are honoured as a refusal to read (the default).
ROBOTS_UNREADABLE_DENY: Final[str] = "deny"
#: Unreadable robots rules are read past, by a deployment that says so.
ROBOTS_UNREADABLE_ALLOW: Final[str] = "allow"

type RobotsUnreadable = Literal["deny", "allow"]

#: Media types the web path will read as a page.
HTML_CONTENT_TYPES: Final[tuple[str, ...]] = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
)

#: :attr:`Candidate.facets` key carrying the ``media-contracts`` ``EXTRACTION_STATUS_*`` vocabulary.
EXTRACTION_STATUS_FACET: Final[str] = "extraction_status"

#: :attr:`Candidate.facets` key carrying why an extraction was refused or failed, in a sentence a
#: person can read: the status alone says "refused" for robots, a content type and a size cap
#: alike, and a caller cannot tell its reader which one stopped the read.
EXTRACTION_REASON_FACET: Final[str] = "extraction_reason"

#: :attr:`Candidate.facets` key carrying what the reader can do next about a read that did not
#: happen ("try again", "find the same facts on another site"), apart from the reason.
EXTRACTION_NEXT_STEP_FACET: Final[str] = "extraction_next_step"

#: :attr:`Candidate.facets` key carrying the URL that actually answered, after redirects or a
#: followed frame -- which is not always the one asked for.
EXTRACTION_FINAL_URL_FACET: Final[str] = "final_url"

#: :attr:`Candidate.facets` key carrying the character set the page was decoded with.
EXTRACTION_CHARSET_FACET: Final[str] = "charset"

#: :attr:`Candidate.facets` key carrying what the reader should know about how the text was
#: got, one sentence each (a followed frame, a PDF read only in part).
EXTRACTION_NOTES_FACET: Final[str] = "extraction_notes"

#: :attr:`Candidate.facets` key carrying the little a thin page did hold, when its frame could
#: not be followed: the read is failed, and this keeps the text from being lost with it.
EXTRACTION_THIN_TEXT_FACET: Final[str] = "readable_text"

#: HTTP's "your copy is still current" (D30).
_STATUS_NOT_MODIFIED: Final[int] = 304

#: :attr:`Candidate.facets` key carrying *how* the text was produced (SR-B6).
EXTRACTION_METHOD_FACET: Final[str] = "extraction_method"

#: Refusal identity for a call the ``[extract]`` extra is not installed for.
EXTRACTOR_UNAVAILABLE_SCOPE: Final[str] = "extractor-unavailable"

#: What the ``[extract]`` extra installs, named in the refusal.
_EXTRACT_EXTRA: Final[str] = "3tears-search[extract]"

#: Method name recorded when trafilatura produced the text.
_METHOD_TRAFILATURA: Final[str] = "trafilatura"

#: Method name recorded when a caller's heavy fetcher produced the carrier.
_METHOD_HEAVY: Final[str] = "trafilatura+rendered"

#: The cap scopes the standalone transport refuses under (``standalone.RESPONSE_BYTES_SCOPE``
#: and ``standalone.CONTENT_TYPE_SCOPE``), spelled here because this module must not import the
#: ``[standalone]`` extra; a test holds the two spellings equal.
_RESPONSE_BYTES_SCOPE: Final[str] = "response-bytes"
_CONTENT_TYPE_SCOPE: Final[str] = "content-type"

#: Cap on a ``robots.txt`` read. Google's own limit, and generous.
_ROBOTS_MAX_BYTES: Final[int] = 512 * 1024

#: Characters a URL keeps when it is percent-encoded for a sentence: what RFC 3986 reserves, and ``%``.
_URL_SAFE: Final[str] = ":/?#[]@!$&'()*+,;=%~"

#: The longest frame address a sentence carries.
_SHOWN_URL_CHARS: Final[int] = 500

#: The least time worth spending on a frame; less than this left and it is not followed.
_FRAME_MINIMUM_SECONDS: Final[float] = 1.0

#: Markup in, article text out, ``None`` when the extractor found nothing worth keeping.
type _Extractor = Callable[[str], str | None]


@dataclass(frozen=True, slots=True)
class CarrierText:
    """What a :class:`CarrierReader` made of one carrier.

    :ivar text: the readable text
    :ivar title: the carrier's own title, when it states one
    :ivar published_at: when it says it was made, timezone-aware, when it says
    :ivar notes: what the reader should know about how the text was got, one sentence each
    """

    text: str
    title: str | None = None
    published_at: datetime | None = None
    notes: tuple[str, ...] = ()


class CarrierUnreadable(Exception):
    """A :class:`CarrierReader` could not read its carrier, and says why and what next.

    :param reason: why, in a sentence a person can read
    :ptype reason: str
    :param next_step: what the reader can do about it
    :ptype next_step: str | None
    :param refused: whether a limit declined the read (``refused``) rather than the read
        breaking (``failed``)
    :ptype refused: bool
    """

    def __init__(self, reason: str, *, next_step: str | None = None, refused: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.next_step = next_step
        self.refused = refused


@dataclass(frozen=True, slots=True)
class CarrierReader:
    """A reader for a carrier the web path does not read itself -- a PDF, say.

    The caller owns the library that reads it; this package owns the fetch,
    robots, the caps and the record.

    :ivar kind: what the carrier is called in a sentence ("PDF")
    :ivar media_types: the declared media types it reads, as prefixes
    :ivar max_bytes: the most of one such carrier the fetch reads
    :ivar read: the reader itself; raises :class:`CarrierUnreadable` when it cannot
    """

    kind: str
    media_types: tuple[str, ...]
    max_bytes: int
    read: Callable[[bytes], Awaitable[CarrierText]] = field(compare=False)

    def reads(self, media_type: str | None) -> bool:
        """Whether this reader takes a carrier declared as ``media_type``."""
        lowered = (media_type or "").lower()
        return bool(lowered) and any(lowered.startswith(prefix.lower()) for prefix in self.media_types)


@dataclass(frozen=True, slots=True)
class _Read:
    """One carrier read through: its text and what is known about it."""

    text: str
    response: TransportResponse
    title: str | None
    published_at: datetime | None
    charset: str | None
    notes: tuple[str, ...]
    method: str
    frame: str | None = None


@dataclass(frozen=True, slots=True)
class _Unread:
    """One carrier that was not read, with the record it earns."""

    status: str
    reason: str
    next_step: str | None = None
    #: the response, when one came back -- a page that turned out thin still has a body
    response: TransportResponse | None = None


async def extract(
    candidate: Candidate,
    *,
    transport: FetchTransport,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout_seconds: float | None = None,
    respect_robots: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
    heavy_fetcher: HeavyFetcher | None = None,
    revalidate: bool = False,
    robots_timeout_seconds: float = DEFAULT_ROBOTS_TIMEOUT_SECONDS,
    robots_unreadable: RobotsUnreadable = "deny",
    readers: Sequence[CarrierReader] = (),
    thin_page_chars: int | None = None,
) -> Candidate:
    """Fill one candidate's content slot from its carrier.

    **Conditional revalidation (D30 / SR-M4) is opt-in and must not be
    inferred**; see the module docstring. Nothing is stored here, by anyone,
    ever (D14).

    :param candidate: the candidate to extract; untouched when it already carries content
        (SR-A2), unless ``revalidate`` is set and it carries a validator
    :ptype candidate: Candidate
    :param transport: the injected byte-capped fetch seam; used for the carrier, ``robots.txt``
        and a followed frame
    :ptype transport: FetchTransport
    :param max_bytes: hard cap on a fetched page (SR-G5); a reader's carrier has its own
    :ptype max_bytes: int
    :param timeout_seconds: per-fetch bound; None uses the transport's configured value
        (SR-G1). It is also the time a page and its frame share
    :ptype timeout_seconds: float | None
    :param respect_robots: honour ``robots.txt`` (D12's default); deployment config, never a
        per-call choice
    :ptype respect_robots: bool
    :param user_agent: the agent name robots rules are matched against; the deployment's own,
        the same one its transport sends
    :ptype user_agent: str
    :param heavy_fetcher: when given, the carrier is fetched through it -- the caller's explicit
        choice for this candidate, never an automatic fallback
    :ptype heavy_fetcher: HeavyFetcher | None
    :param revalidate: send a conditional request for a candidate that carries content and a
        validator
    :ptype revalidate: bool
    :param robots_timeout_seconds: the robots read's own bound
    :ptype robots_timeout_seconds: float
    :param robots_unreadable: what a robots file that cannot be read means: ``deny`` (the
        default) records the read failed, ``allow`` reads the page anyway. Deployment config
    :ptype robots_unreadable: RobotsUnreadable
    :param readers: readers for carriers that are not pages, each with its own media types and
        cap
    :ptype readers: Sequence[CarrierReader]
    :param thin_page_chars: when set, a page with less readable text than this that points at
        its article follows it one level (module docstring); None never follows
    :ptype thin_page_chars: int | None
    :return: the candidate with content, fidelity and extraction facets recorded; on a
        per-candidate failure, the candidate with its status, reason and next step and no content
    :rtype: Candidate
    :raises threetears.search.contracts.errors.LocalCapExceeded: when the ``[extract]`` extra is
        not installed
    """
    if robots_unreadable not in (ROBOTS_UNREADABLE_DENY, ROBOTS_UNREADABLE_ALLOW):
        raise ValueError(f"robots_unreadable must be 'deny' or 'allow', got {robots_unreadable!r}")
    started = time.monotonic()
    conditional = _conditional_headers(candidate) if revalidate else {}
    if candidate.content is not None and not conditional:
        return candidate

    url = _carrier_url(candidate)
    if url is None:
        return _marked(candidate, EXTRACTION_STATUS_FAILED, "it has no address to fetch")

    extractor = _load_extractor()
    rules = _Rules(
        transport=transport,
        user_agent=user_agent,
        respect_robots=respect_robots and heavy_fetcher is None,
        robots_timeout_seconds=robots_timeout_seconds,
        robots_unreadable=robots_unreadable,
    )
    robots_notes: list[str] = []
    refusal = await rules.permission(url, robots_notes)
    if refusal is not None:
        return _marked(candidate, refusal.status, refusal.reason, refusal.next_step)

    read = await _read_carrier(
        url,
        transport=transport,
        heavy_fetcher=heavy_fetcher,
        extractor=extractor,
        max_bytes=max_bytes,
        readers=readers,
        timeout_seconds=timeout_seconds,
        headers=conditional or None,
    )

    if isinstance(read, _Unread) and read.response is not None and conditional:
        if read.response.status_code == _STATUS_NOT_MODIFIED:
            return candidate.model_copy(
                update={
                    "content": candidate.content.model_copy(update={"origin": "revalidated"})
                    if candidate.content is not None
                    else None,
                    "facets": {**candidate.facets, EXTRACTION_STATUS_FACET: EXTRACTION_STATUS_UNCHANGED},
                }
            )

    if thin_page_chars is not None and heavy_fetcher is None:
        read = await _follow_a_thin_page(
            read,
            page_url=url,
            rules=rules,
            extractor=extractor,
            max_bytes=max_bytes,
            readers=readers,
            thin_page_chars=thin_page_chars,
            budget=_TimeBudget(
                started=started,
                seconds=timeout_seconds if timeout_seconds is not None else DEFAULT_FETCH_TIME_BUDGET_SECONDS,
            ),
            notes=robots_notes,
        )

    if isinstance(read, _Unread):
        kept: dict[str, JsonValue] = {}
        if isinstance(read, _UnreadWithText) and read.thin_text:
            kept[EXTRACTION_THIN_TEXT_FACET] = read.thin_text
        return _marked(candidate, read.status, read.reason, read.next_step, kept=kept)

    final_url = read.response.final_url or url
    facets: dict[str, JsonValue] = {
        **candidate.facets,
        EXTRACTION_STATUS_FACET: EXTRACTION_STATUS_COMPLETE,
        EXTRACTION_METHOD_FACET: read.method,
        EXTRACTION_FINAL_URL_FACET: final_url,
    }
    if read.charset:
        facets[EXTRACTION_CHARSET_FACET] = read.charset
    notes = [*robots_notes, *read.notes]
    if notes:
        facets[EXTRACTION_NOTES_FACET] = list(notes)
    update: dict[str, object] = {
        "content": ContentSlot(
            text=read.text,
            origin="later-fetch",
            mime_type=_declared_type(read.response),
            size_bytes=len(read.response.body),
            etag=_validator_header(read.response, "etag"),
            last_modified=_validator_header(read.response, "last-modified"),
        ),
        "fidelity_achieved": FIDELITY_CONTENT,
        "facets": facets,
    }
    if read.title and not candidate.title:
        update["title"] = read.title
    if read.published_at is not None and candidate.published_at is None:
        update["published_at"] = read.published_at
    return candidate.model_copy(update=update)


@dataclass(frozen=True, slots=True)
class _Rules:
    """How this call asks a site's permission: the transport, the name, the bound, the stance."""

    transport: FetchTransport
    user_agent: str
    respect_robots: bool
    robots_timeout_seconds: float
    robots_unreadable: RobotsUnreadable

    async def permission(self, url: str, notes: list[str]) -> _Unread | None:
        """``None`` when ``url`` may be read; otherwise the record a read that did not happen earns.

        :param url: the URL about to be fetched
        :ptype url: str
        :param notes: where a robots file read past under ``allow`` is said
        :ptype notes: list[str]
        :return: the refusal or failure, or None
        :rtype: _Unread | None
        """
        if not self.respect_robots:
            return None
        allowed, unreadable = await _robots_verdict(
            url,
            transport=self.transport,
            user_agent=self.user_agent,
            timeout_seconds=self.robots_timeout_seconds,
        )
        if unreadable is not None:
            reason = f"robots.txt could not be read ({unreadable})"
            if self.robots_unreadable == ROBOTS_UNREADABLE_ALLOW:
                notes.append(f"{reason}; this deployment reads a page whose robots.txt cannot be read.")
                return None
            return _Unread(EXTRACTION_STATUS_FAILED, reason, "try again")
        if not allowed:
            return _Unread(
                EXTRACTION_STATUS_REFUSED,
                "the site's robots.txt does not allow it to be read",
                "find the same facts on another site",
            )
        return None


@dataclass(frozen=True, slots=True)
class _TimeBudget:
    """The seconds a page and its frame share, counted from the start of the call."""

    started: float
    seconds: float

    def left(self) -> float:
        """What remains of the budget, never below zero."""
        return max(0.0, self.seconds - (time.monotonic() - self.started))


def _carrier_url(candidate: Candidate) -> str | None:
    """The locator the web path should read: the canonical one, else the first.

    :param candidate: the candidate being extracted
    :ptype candidate: Candidate
    :return: the URL to fetch, or None when the candidate carries no locator at all
    :rtype: str | None
    """
    for locator in candidate.locators:
        if locator.rel == "canonical":
            return locator.url
    return candidate.locators[0].url if candidate.locators else None


def _load_extractor() -> _Extractor:
    """The installed HTML-to-text extractor, or a typed refusal.

    :return: trafilatura's ``extract`` callable
    :rtype: _Extractor
    :raises threetears.search.contracts.errors.LocalCapExceeded: when the extra is not installed
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


def _page_facts(markup: str) -> tuple[str | None, datetime | None]:
    """A page's own title and publication date, as trafilatura reads them from its markup.

    :param markup: the decoded page
    :ptype markup: str
    :return: the title and the date, each None when the page does not say
    :rtype: tuple[str | None, datetime | None]
    """
    try:
        from trafilatura.metadata import extract_metadata

        found = extract_metadata(markup)
    except Exception:  # noqa: BLE001 -- metadata is extra: a page whose header cannot be read is still read
        return None, None
    if found is None:
        return None, None
    title = (getattr(found, "title", None) or "").strip() or None
    return title, _day(getattr(found, "date", None))


def _day(value: object) -> datetime | None:
    """A ``YYYY-MM-DD`` string as midnight UTC on that day, or None.

    :param value: what the page's metadata held
    :ptype value: object
    :return: the day, timezone-aware
    :rtype: datetime | None
    """
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip()[:10], "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        # NOSILENT: a page date that is not a day is a page with no date, the documented None
        return None


def _conditional_headers(candidate: Candidate) -> dict[str, str]:
    """The validators to send for this candidate, if it has any.

    :param candidate: the candidate whose stored validators to spend
    :ptype candidate: Candidate
    :return: conditional request headers, possibly empty
    :rtype: dict[str, str]
    """
    slot = candidate.content
    if slot is None:
        return {}
    headers: dict[str, str] = {}
    if slot.etag:
        headers["If-None-Match"] = slot.etag
    if slot.last_modified:
        headers["If-Modified-Since"] = slot.last_modified
    return headers


def _validator_header(response: TransportResponse, name: str) -> str | None:
    """Read one validator off a response, treating blank as absent.

    :param response: the fetched carrier
    :ptype response: TransportResponse
    :param name: the header name, lower-cased per the transport contract
    :ptype name: str
    :return: the header's value, or None when absent or blank
    :rtype: str | None
    """
    raw = response.headers.get(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


async def _read_carrier(
    url: str,
    *,
    transport: FetchTransport,
    heavy_fetcher: HeavyFetcher | None,
    extractor: _Extractor,
    max_bytes: int,
    readers: Sequence[CarrierReader],
    timeout_seconds: float | None,
    headers: Mapping[str, str] | None = None,
) -> _Read | _Unread:
    """Fetch one carrier and read it: a page through the extractor, anything else through its reader.

    :param url: the carrier's address
    :ptype url: str
    :param transport: the ordinary byte-capped seam
    :ptype transport: FetchTransport
    :param heavy_fetcher: the caller's escalation, when they passed one
    :ptype heavy_fetcher: HeavyFetcher | None
    :param extractor: markup to text
    :ptype extractor: _Extractor
    :param max_bytes: the most of a page this read takes
    :ptype max_bytes: int
    :param readers: readers for carriers that are not pages
    :ptype readers: Sequence[CarrierReader]
    :param timeout_seconds: this fetch's bound
    :ptype timeout_seconds: float | None
    :param headers: conditional validators, or None
    :ptype headers: Mapping[str, str] | None
    :return: what was read, or why nothing was
    :rtype: _Read | _Unread
    """
    largest = max([max_bytes, *(reader.max_bytes for reader in readers)])
    allowed = HTML_CONTENT_TYPES + tuple(media for reader in readers for media in reader.media_types)
    try:
        response = await _fetch_carrier(
            url,
            transport=transport,
            heavy_fetcher=heavy_fetcher,
            max_bytes=largest,
            allowed_content_types=allowed,
            timeout_seconds=timeout_seconds,
            headers=headers,
        )
    except LocalCapExceeded as exc:
        return _cap_refusal(exc, largest=largest, max_bytes=max_bytes, readers=readers)
    except TimedOut:
        bound = f" within {timeout_seconds:g} s" if timeout_seconds is not None else " in time"
        return _Unread(EXTRACTION_STATUS_FAILED, f"the site did not answer{bound}", "try again")
    except SearchFailure as exc:
        return _Unread(
            EXTRACTION_STATUS_FAILED,
            f"the address could not be reached ({exc.message})",
            "check the address, or try again later",
        )

    if headers and response.status_code == _STATUS_NOT_MODIFIED:
        return _Unread(EXTRACTION_STATUS_UNCHANGED, "", response=response)
    if not 200 <= response.status_code < 300:
        reason, next_step = _status_reason(response)
        return _Unread(EXTRACTION_STATUS_FAILED, reason, next_step, response=response)

    declared = _declared_type(response)
    reader = next((each for each in readers if each.reads(declared)), None)
    if reader is not None:
        return await _read_with(reader, response)

    if len(response.body) > max_bytes:
        return _Unread(
            EXTRACTION_STATUS_REFUSED,
            f"the page is {_size(len(response.body))}, past the {_size(max_bytes)} a page may be",
            "find a shorter copy of it, or the same facts elsewhere",
        )
    markup, charset = _decode(response.body, response.headers.get("content-type", ""))
    text = extractor(markup) or ""
    title, published_at = _page_facts(markup) if text else (None, None)
    method = _METHOD_HEAVY if heavy_fetcher is not None else _METHOD_TRAFILATURA
    if not text:
        return _Unread(
            EXTRACTION_STATUS_FAILED,
            "the page has no readable text",
            "it may be built by JavaScript, which the fetch does not run; find the same facts elsewhere",
            response=response,
        )
    return _Read(
        text=text,
        response=response,
        title=title,
        published_at=published_at,
        charset=charset,
        notes=(),
        method=method,
    )


async def _read_with(reader: CarrierReader, response: TransportResponse) -> _Read | _Unread:
    """A carrier read by the caller's reader, inside the reader's own cap.

    :param reader: the reader whose media types the carrier declared
    :ptype reader: CarrierReader
    :param response: the fetched carrier
    :ptype response: TransportResponse
    :return: what was read, or why nothing was
    :rtype: _Read | _Unread
    """
    if len(response.body) > reader.max_bytes:
        return _Unread(
            EXTRACTION_STATUS_REFUSED,
            f"the {reader.kind} is {_size(len(response.body))}, past the {_size(reader.max_bytes)} "
            f"a {reader.kind} may be",
            f"find a smaller copy of the {reader.kind}, or the same facts on a page",
        )
    try:
        read = await reader.read(response.body)
    except CarrierUnreadable as exc:
        status = EXTRACTION_STATUS_REFUSED if exc.refused else EXTRACTION_STATUS_FAILED
        return _Unread(status, exc.reason, exc.next_step)
    if not read.text.strip():
        return _Unread(
            EXTRACTION_STATUS_FAILED,
            f"the {reader.kind} has no readable text",
            "it may be scanned images, which are not read; find the same facts elsewhere",
        )
    return _Read(
        text=read.text,
        response=response,
        title=read.title,
        published_at=read.published_at,
        charset=None,
        notes=read.notes,
        method=reader.kind.lower(),
    )


def _cap_refusal(exc: LocalCapExceeded, *, largest: int, max_bytes: int, readers: Sequence[CarrierReader]) -> _Unread:
    """The record for a fetch a local cap declined, saying which cap.

    :param exc: the transport's refusal
    :ptype exc: LocalCapExceeded
    :param largest: the cap the fetch was made under
    :ptype largest: int
    :param max_bytes: a page's cap
    :ptype max_bytes: int
    :param readers: the readers whose caps the fetch also allowed for
    :ptype readers: Sequence[CarrierReader]
    :return: the refusal
    :rtype: _Unread
    """
    if exc.scope == _CONTENT_TYPE_SCOPE:
        kinds = ["a page", *(f"a {reader.kind}" for reader in readers)]
        return _Unread(
            EXTRACTION_STATUS_REFUSED,
            f"the address is not {' or '.join(kinds)}, so it is not read ({exc.message})",
            "find a page about it instead",
        )
    if exc.scope == _RESPONSE_BYTES_SCOPE:
        owner = next((reader for reader in readers if reader.max_bytes == largest and largest > max_bytes), None)
        if owner is not None:
            return _Unread(
                EXTRACTION_STATUS_REFUSED,
                f"the file is larger than {_size(largest)}, the most read for a {owner.kind} "
                f"(a page may be {_size(max_bytes)})",
                "find a smaller copy, or the same facts on a page",
            )
        return _Unread(
            EXTRACTION_STATUS_REFUSED,
            f"the page is larger than {_size(largest)}, the most a fetch reads",
            "find a shorter copy of it, or the same facts elsewhere",
        )
    return _Unread(EXTRACTION_STATUS_REFUSED, str(exc), None)


def _size(count: int) -> str:
    """A byte count in words a person reads: ``2 MB``, ``512 KB``, ``900 bytes``."""
    if count >= 1024 * 1024:
        return f"{count / (1024 * 1024):.3g} MB"
    if count >= 1024:
        return f"{count / 1024:.3g} KB"
    return f"{count} bytes"


def _status_reason(response: TransportResponse) -> tuple[str, str]:
    """Why a non-2xx answer means the page was not read, and what to do about it.

    :param response: the answer
    :ptype response: TransportResponse
    :return: the reason and the next step
    :rtype: tuple[str, str]
    """
    status = response.status_code
    if status == 403:
        return (
            "the site refused the request (403 Forbidden); it may turn away automated readers",
            "find the same facts on another site",
        )
    if status == 401:
        return ("the page needs a login (401)", "it cannot be read without one; find the same facts elsewhere")
    if status in (404, 410):
        return (f"the page was not found ({status})", "check the address, or search for the page")
    if status == 429:
        wait = (response.headers.get("retry-after") or "").strip()
        asked = f"; it asks to wait {wait} s" if wait.isdigit() else ""
        return (
            f"the site is turning requests away for now (429 Too Many Requests){asked}",
            "wait and try again later, or find the same facts on another site",
        )
    if status >= 500:
        return (f"the site had an error ({status})", "try again shortly")
    if 300 <= status < 400:
        target = response.headers.get("location")
        pointed = f" to {target}" if target else ""
        return (
            f"the site sent the fetch further{pointed} than it follows ({status})",
            "fetch the address it points to",
        )
    return (f"the site answered {status}", "check the address")


async def _fetch_carrier(
    url: str,
    *,
    transport: FetchTransport,
    heavy_fetcher: HeavyFetcher | None,
    max_bytes: int,
    allowed_content_types: tuple[str, ...],
    timeout_seconds: float | None,
    headers: Mapping[str, str] | None = None,
) -> TransportResponse:
    """Read the carrier through whichever fetcher the caller chose.

    ``headers`` is passed to a heavy fetcher only when there is something to
    send: an implementer written before the parameter existed has no such
    parameter at all.

    :param url: the carrier's address
    :ptype url: str
    :param transport: the ordinary byte-capped seam
    :ptype transport: FetchTransport
    :param heavy_fetcher: the caller's escalation, when they passed one
    :ptype heavy_fetcher: HeavyFetcher | None
    :param max_bytes: hard cap on the body (SR-G5)
    :ptype max_bytes: int
    :param allowed_content_types: the gate the ordinary fetch refuses on before the body
    :ptype allowed_content_types: tuple[str, ...]
    :param timeout_seconds: per-fetch bound
    :ptype timeout_seconds: float | None
    :param headers: conditional validators to send, or ``None``
    :ptype headers: Mapping[str, str] | None
    :return: the fetched carrier
    :rtype: TransportResponse
    """
    if heavy_fetcher is not None:
        if headers:
            return await heavy_fetcher.fetch_rendered(
                url,
                max_bytes=max_bytes,
                timeout_seconds=timeout_seconds,
                headers=headers,
            )
        return await heavy_fetcher.fetch_rendered(url, max_bytes=max_bytes, timeout_seconds=timeout_seconds)
    return await transport.fetch(
        "GET",
        url,
        max_bytes=max_bytes,
        allowed_content_types=allowed_content_types,
        timeout_seconds=timeout_seconds,
        headers=headers,
    )


def robots_agent_token(user_agent: str) -> str:
    """The name robots rules are matched against: the user agent's product token.

    ``metallm/0.71.0 (+https://example.org)`` is ``metallm`` to a robots file, which names
    agents by product.

    :param user_agent: the full user agent the transport sends
    :ptype user_agent: str
    :return: its first product token, or the whole string when it has none
    :rtype: str
    """
    token = user_agent.strip().split(" ", 1)[0].split("/", 1)[0]
    return token or user_agent


async def _robots_verdict(
    url: str,
    *,
    transport: FetchTransport,
    user_agent: str,
    timeout_seconds: float,
) -> tuple[bool, str | None]:
    """Whether the host's ``robots.txt`` permits ``url``, or why its rules are unknown (D12).

    Fetched through the same seam as the carrier, under its own bound, with
    no content-type gate: a 2xx is a robots file whatever it says it is.

    :param url: the carrier URL whose permission is in question
    :ptype url: str
    :param transport: the injected fetch seam
    :ptype transport: FetchTransport
    :param user_agent: the agent name rules are matched against
    :ptype user_agent: str
    :param timeout_seconds: the robots read's own bound
    :ptype timeout_seconds: float
    :return: ``(allowed, None)`` when the rules are known, ``(False, why)`` when they are not
    :rtype: tuple[bool, str | None]
    """
    parts = urlsplit(url)
    robots_url = urljoin(f"{parts.scheme}://{parts.netloc}", "/robots.txt")
    try:
        response = await transport.fetch(
            "GET",
            robots_url,
            max_bytes=_ROBOTS_MAX_BYTES,
            allowed_content_types=None,
            timeout_seconds=timeout_seconds,
        )
    except TimedOut:
        return False, f"timed out after {timeout_seconds:g} s"
    except LocalCapExceeded:
        return False, f"it is larger than {_size(_ROBOTS_MAX_BYTES)}"
    except SearchFailure as exc:
        return False, f"the site could not be reached: {exc.message}"

    if 400 <= response.status_code < 500:
        return True, None
    if not 200 <= response.status_code < 300:
        return False, f"the server answered {response.status_code}"

    parser = RobotFileParser()
    parser.parse(response.body.decode("utf-8", errors="replace").splitlines())
    return parser.can_fetch(robots_agent_token(user_agent), url), None


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


#: ``charset=`` in a ``Content-Type`` header.
_HEADER_CHARSET = re.compile(r"charset\s*=\s*[\"']?([A-Za-z0-9._:-]+)", re.I)
#: ``<meta charset=...>`` or ``<meta http-equiv=content-type content="...; charset=...">``.
_META_CHARSET = re.compile(rb"<meta[^>]+charset\s*=\s*[\"']?([A-Za-z0-9._:-]+)", re.I)
#: How far into a page a ``<meta>`` charset is looked for (the HTML standard's prescan reads 1,024).
_PRESCAN_BYTES: Final[int] = 4096


def _decode(body: bytes, content_type: str) -> tuple[str, str]:
    """Decode a page by what it says it is, then what it looks like.

    The header's ``charset`` first, then a byte-order mark, then a ``<meta>``
    charset near the top, then UTF-8 if the bytes are valid UTF-8, then the
    detector trafilatura already depends on. Read as UTF-8 regardless, a page
    in Windows-1252 or Shift JIS came back as replacement characters.

    :param body: the page's bytes
    :ptype body: bytes
    :param content_type: the response's ``Content-Type`` header
    :ptype content_type: str
    :return: the text and the charset it was decoded with
    :rtype: tuple[str, str]
    """
    for name in _declared_charsets(body, content_type):
        try:
            return body.decode(name), codecs.lookup(name).name
        except LookupError, UnicodeDecodeError:
            # NOSILENT: a charset the page names wrongly or that Python lacks; the next candidate is tried
            continue
    try:
        return body.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        # NOSILENT: not UTF-8, so the detector below decides
        pass
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(body).best()
        if best is not None and best.encoding:
            return str(best), codecs.lookup(best.encoding).name
    except ImportError:
        # NOSILENT: no detector installed; UTF-8 with replacement is the stated last resort
        pass
    return body.decode("utf-8", errors="replace"), "utf-8"


def _declared_charsets(body: bytes, content_type: str) -> list[str]:
    """The charsets a page declares, most authoritative first.

    :param body: the page's bytes
    :ptype body: bytes
    :param content_type: the response's ``Content-Type`` header
    :ptype content_type: str
    :return: candidate charset names
    :rtype: list[str]
    """
    names: list[str] = []
    header = _HEADER_CHARSET.search(content_type or "")
    if header:
        names.append(header.group(1))
    for mark, name in (
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    ):
        if body.startswith(mark):
            names.insert(0, name)
    meta = _META_CHARSET.search(body[:_PRESCAN_BYTES])
    if meta:
        names.append(meta.group(1).decode("ascii", errors="ignore"))
    return names


class _FramesAndPointers(HTMLParser):
    """Collects a page's ``iframe`` sources and its ``citation_pdf_url``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.frames: list[str] = []
        self.citation_pdf: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        named = {key.lower(): (value or "") for key, value in attrs}
        if tag == "iframe" and named.get("src", "").strip():
            self.frames.append(named["src"].strip())
        elif tag == "meta" and named.get("name", "").lower() == "citation_pdf_url" and named.get("content"):
            self.citation_pdf = self.citation_pdf or named["content"].strip()


def _pointer_of(markup: str, base: str) -> tuple[str | None, int, bool]:
    """Where a thin page says its article is, how many frames it holds, and whether it said so outright.

    ``citation_pdf_url`` is the page saying outright where its article is, so
    it wins; otherwise the page's first ``iframe``, which is followed only
    when it is the page's one frame.

    :param markup: the decoded page
    :ptype markup: str
    :param base: the URL the page answered from, to resolve relative addresses
    :ptype base: str
    :return: the absolute URL (None when the page points nowhere), the frame count, and
        whether the URL came from ``citation_pdf_url``
    :rtype: tuple[str | None, int, bool]
    """
    parser = _FramesAndPointers()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # noqa: BLE001 -- a page the parser chokes on points nowhere
        return None, 0, False
    if parser.citation_pdf:
        return urljoin(base, parser.citation_pdf), len(parser.frames), True
    if parser.frames:
        return urljoin(base, parser.frames[0]), len(parser.frames), False
    return None, 0, False


#: Second-level labels under a country code that are registries rather than sites
#: (``example.co.uk``'s registrable part is three labels, not two).
_REGISTRY_LABELS: Final[frozenset[str]] = frozenset({"ac", "co", "com", "edu", "gov", "net", "org", "or", "ne", "go"})


def _registrable(host: str) -> str:
    """A host's registrable part, by the rule most of the public suffix list reduces to.

    Two labels (``example.org``), or three under a country code's registry label
    (``example.co.uk``). No suffix list ships with this leaf; a host this rule
    reads too widely is still fetched only through the guarded transport, with
    robots asked, so the rule decides only which frames are worth following.

    :param host: a hostname
    :ptype host: str
    :return: its registrable part, lower-cased
    :rtype: str
    """
    labels = [label for label in host.lower().rstrip(".").split(".") if label]
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in _REGISTRY_LABELS:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def same_site(first: str, second: str) -> bool:
    """Whether two URLs are on the same registrable site (``www.a.org`` and ``pdf.a.org`` are).

    :param first: a URL
    :ptype first: str
    :param second: another URL
    :ptype second: str
    :return: True when both have a host and the hosts share a registrable part
    :rtype: bool
    """
    one, two = urlsplit(first).hostname or "", urlsplit(second).hostname or ""
    return bool(one and two) and _registrable(one) == _registrable(two)


async def _follow_a_thin_page(
    read: _Read | _Unread,
    *,
    page_url: str,
    rules: _Rules,
    extractor: _Extractor,
    max_bytes: int,
    readers: Sequence[CarrierReader],
    thin_page_chars: int,
    budget: _TimeBudget,
    notes: list[str],
) -> _Read | _Unread:
    """A thin page's article, read from the frame it points at, when every condition holds.

    :param read: what the page itself gave
    :ptype read: _Read | _Unread
    :param page_url: the page asked for
    :ptype page_url: str
    :param rules: how permission is asked
    :ptype rules: _Rules
    :param extractor: markup to text
    :ptype extractor: _Extractor
    :param max_bytes: a page's cap; the frame gets what the page left of the cap
    :ptype max_bytes: int
    :param readers: readers for carriers that are not pages
    :ptype readers: Sequence[CarrierReader]
    :param thin_page_chars: readable text under this many characters makes a page thin
    :ptype thin_page_chars: int
    :param budget: the time the page and its frame share
    :ptype budget: _TimeBudget
    :param notes: the call's notes, which a followed frame adds to
    :ptype notes: list[str]
    :return: the frame's text, the page's own when it is not thin or points nowhere, or why the
        frame was not read
    :rtype: _Read | _Unread
    """
    if isinstance(read, _Read):
        if read.method not in (_METHOD_TRAFILATURA,) or len(read.text) >= thin_page_chars:
            return read
        response, thin_text = read.response, read.text
    else:
        no_text = read.status == EXTRACTION_STATUS_FAILED and read.reason == "the page has no readable text"
        if read.response is None or not no_text:
            return read
        response, thin_text = read.response, ""

    markup, _ = _decode(response.body, response.headers.get("content-type", ""))
    page = response.final_url or page_url
    pointed, frames, outright = _pointer_of(markup, page)
    if pointed is None:
        return read
    # The address comes from the page's own markup, so it is made a plain URL before a sentence
    # carries it: no spaces, no line breaks, nothing a reader could take for an instruction.
    target = quote(pointed.strip(), safe=_URL_SAFE)[:_SHOWN_URL_CHARS]

    def menu_only(why: str) -> _Unread:
        return _UnreadWithText(
            EXTRACTION_STATUS_FAILED,
            f"only the page's menu was readable; the article is at {target}",
            f"{why}; fetch {target} directly, or find the same facts elsewhere",
            response,
            thin_text,
        )

    if frames > 1 and not outright:
        return menu_only(f"the page holds {frames} frames, and which one is the article is not guessed")
    if urlsplit(target).scheme not in ("http", "https"):
        return menu_only("the frame is not a web address")
    if not same_site(page, target):
        return menu_only("it is on another site, and a frame is followed only on the page's own site")
    used = len(response.body)
    frame_readers = [
        CarrierReader(kind=r.kind, media_types=r.media_types, max_bytes=r.max_bytes - used, read=r.read)
        for r in readers
        if r.max_bytes - used > 0
    ]
    if max_bytes - used <= 0 and not frame_readers:
        return menu_only("the page used the whole size limit")
    left_seconds = budget.left()
    if left_seconds < _FRAME_MINIMUM_SECONDS:
        return menu_only("no time was left to read it")
    refusal = await rules.permission(target, notes)
    if refusal is not None:
        return menu_only(refusal.reason)
    framed = await _read_carrier(
        target,
        transport=rules.transport,
        heavy_fetcher=None,
        extractor=extractor,
        max_bytes=max(max_bytes - used, 1),
        readers=frame_readers,
        timeout_seconds=budget.left(),
    )
    if isinstance(framed, _Unread):
        return menu_only(framed.reason)
    if len(framed.text) <= len(thin_text):
        return menu_only("it held no more text than the page")
    return _Read(
        text=framed.text,
        response=framed.response,
        title=framed.title or (read.title if isinstance(read, _Read) else None),
        published_at=framed.published_at or (read.published_at if isinstance(read, _Read) else None),
        charset=framed.charset,
        notes=(
            *framed.notes,
            f"The page at {page} held only its menu; this is the article it frames, "
            f"from {framed.response.final_url or target}.",
        ),
        method=framed.method,
        frame=target,
    )


@dataclass(frozen=True, slots=True)
class _UnreadWithText(_Unread):
    """A thin page whose frame was not read, with the text the page itself held."""

    thin_text: str = ""


def _marked(
    candidate: Candidate,
    status: str,
    reason: str,
    next_step: str | None = None,
    *,
    kept: Mapping[str, JsonValue] | None = None,
) -> Candidate:
    """The candidate with its extraction status, the reason and the next step recorded, and no content.

    :param candidate: the candidate whose extraction did not produce text
    :ptype candidate: Candidate
    :param status: the ``media-contracts`` status to record
    :ptype status: str
    :param reason: why, in a sentence a person can read
    :ptype reason: str
    :param next_step: what the reader can do about it
    :ptype next_step: str | None
    :param kept: further facets to record, such as the text a thin page held
    :ptype kept: Mapping[str, JsonValue] | None
    :return: a copy carrying the status, reason and next-step facets
    :rtype: Candidate
    """
    facets: dict[str, JsonValue] = {
        **candidate.facets,
        EXTRACTION_STATUS_FACET: status,
        EXTRACTION_REASON_FACET: reason,
    }
    if next_step:
        facets[EXTRACTION_NEXT_STEP_FACET] = next_step
    facets.update(kept or {})
    return candidate.model_copy(update={"facets": facets})

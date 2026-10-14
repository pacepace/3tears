"""Web fetch, on the search leaf's Extract path (search-spec.md §4.3).

Same identity as before -- ``threetears.web_fetch``, a ``url`` string in,
readable page text out -- over
:func:`threetears.search.extract.extract` instead of a hand-rolled fetch.

What died with the old body (§10 defects 6 and 7): a ``time.sleep(1)`` inside
an ``async`` call path, which blocked the event loop for every other tool
sharing the pod; an unbounded ``resp.text``, which read whatever the far side
sent before deciding it was too big -- a memory incident waiting for a
``MemoryMax``-capped host; hand-rolled meta-refresh and JS-redirect chasing,
which followed a URL the SSRF guards never saw; and the ``[TOOL ERROR]``
string prefix callers matched on. The read is now streamed against a byte
cap, gated on content type *before* the body, and guarded against private
addresses at the transport (D21, SR-G5).

**One behaviour genuinely changes, and callers must be told.** Extract
honours ``robots.txt`` by default (D12), and this tool did not. A page whose
robots rules refuse ``3tears-search`` now comes back as a refusal rather than
as content. That is the intended stance and the reason it is stated here
rather than buried: the parameter exists (``respect_robots``) because D12
rules the stance as recorded deployment config, so a deployment with its own
agreement with a site can say so -- as configuration, never per call.

**Structure rides the same key search does.** A fetch answers with one
candidate, so the result is projected as a one-candidate
:class:`~threetears.search.contracts.CandidateSet` under
:data:`~threetears.search.contracts.SEARCH_RESULTS_METADATA_KEY` (D22): a
consumer reading structure off a tool result reads one shape whether the tool
searched or fetched, and learns *why* an empty fetch was empty -- refused,
failed, or a carrier with nothing extractable -- from the typed
``extraction_status`` facet rather than from prose.

**What the reader is told.** A read opens with one line: the page's title,
its date, the URL that answered and which characters of how many these are
(``Read "Otters" (published 2026-10-01) at https://...: characters
0-15,000 of 42,113.``). Notes follow on lines of their own -- the window's
"this is a part", a frame that was followed, a PDF read only in part -- then
the text. A read that did not happen says why and what to do next ("the site
is turning requests away for now (429 Too Many Requests)" / "wait and try
again later"), from the facets Extract records.

**PDFs** are read through :func:`threetears.agent.tools.document.parse_document`
(PyMuPDF), handed to Extract as a
:class:`~threetears.search.extract.CarrierReader`: the search leaf owns the
fetch, robots and the caps and imports no PDF library. A PDF has its own size
cap, page cap and parse timeout, is checked for the ``%PDF`` signature before
it is parsed, and is never sent to OCR.

**Deployment config, never per call:** the user agent the fetches send and
robots rules are matched against (the deployment's own name), whether a
robots file that cannot be read means the page may be read
(``robots_unreadable``, ``deny`` by default), and the PDF limits.

The ``credential_resolver`` hook the old body carried is gone. Its only path
in was a ``_credential_resolver`` key in the tool's config dict, which
nothing in the family ever set; carrying an unexercised credential-injection
seam through the rewrite would have meant re-deriving its failure semantics
for no caller.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Final

from langchain_core.tools import StructuredTool

from threetears.agent.tools.text_window import WindowedInput, window_text

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.media.contracts import EXTRACTION_STATUS_COMPLETE, EXTRACTION_STATUS_UNCHANGED
from threetears.search.bind import project_failure_metadata, project_metadata, shown_date
from threetears.search.contracts import (
    Candidate,
    CandidateSet,
    ContentSlot,
    FailureRecord,
    Locator,
    Provenance,
    SearchFailure,
    Spend,
)
from threetears.search.extract import (
    DEFAULT_MAX_BYTES,
    DEFAULT_THIN_PAGE_CHARS,
    DEFAULT_USER_AGENT,
    EXTRACTION_FINAL_URL_FACET,
    EXTRACTION_NEXT_STEP_FACET,
    EXTRACTION_NOTES_FACET,
    EXTRACTION_REASON_FACET,
    EXTRACTION_STATUS_FACET,
    EXTRACTION_THIN_TEXT_FACET,
    ROBOTS_UNREADABLE_DENY,
    CarrierReader,
    CarrierText,
    CarrierUnreadable,
    extract,
)

if TYPE_CHECKING:
    from threetears.search.contracts import FetchTransport

__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_PDF_MAX_BYTES",
    "DEFAULT_PDF_MAX_PAGES",
    "DEFAULT_PDF_PARSE_TIMEOUT_SECONDS",
    "PDF_MEDIA_TYPES",
    "WebFetchTool",
    "create_web_fetch_tool",
    "pdf_reader",
]

#: how much extracted text reaches the caller. A context bound, not a
#: transport one: the byte cap that protects the *host* lives on the fetch
#: transport and fires long before this does.
DEFAULT_MAX_CHARS: Final[int] = 15000

#: what the provenance of a directly-fetched page names as its source. Not a
#: search provider -- nothing searched -- so it names the tool that asked.
_FETCH_PROVIDER_INSTANCE: Final[str] = "threetears.web_fetch"

#: The most of one PDF a fetch reads. Reports and papers run to megabytes; a scanned book does not
#: belong in a context window and is refused with its size.
DEFAULT_PDF_MAX_BYTES: Final[int] = 20 * 1024 * 1024

#: The most pages read from one PDF. The rest is said, not silently dropped.
DEFAULT_PDF_MAX_PAGES: Final[int] = 100

#: How long one PDF may take to parse.
DEFAULT_PDF_PARSE_TIMEOUT_SECONDS: Final[float] = 30.0

#: The media types a PDF is declared as.
PDF_MEDIA_TYPES: Final[tuple[str, ...]] = ("application/pdf", "application/x-pdf")

#: Where the PDF signature may sit: the format allows junk before it in the first kilobyte.
_PDF_SIGNATURE_WINDOW: Final[int] = 1024

#: A PDF date, ``D:YYYYMMDDHHmmSS`` with an optional offset ``Z`` / ``+HH'mm'``.
_PDF_DATE: Final[re.Pattern[str]] = re.compile(
    r"^D?:?(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?(?:(Z)|([+-])(\d{2})'?(\d{2})?'?)?"
)


def _pdf_date(raw: object) -> datetime | None:
    """A PDF's ``creationDate`` as a timezone-aware time, or None when it states none readable.

    :param raw: the metadata value
    :ptype raw: object
    :return: the time
    :rtype: datetime | None
    """
    if not isinstance(raw, str):
        return None
    found = _PDF_DATE.match(raw.strip())
    if found is None:
        return None
    year, month, day, hour, minute, second, zulu, sign, off_h, off_m = found.groups()
    zone = UTC
    if sign:
        offset = timedelta(hours=int(off_h), minutes=int(off_m or 0))
        zone = timezone(offset if sign == "+" else -offset)
    try:
        return datetime(
            int(year), int(month or 1), int(day or 1), int(hour or 0), int(minute or 0), int(second or 0), tzinfo=zone
        )
    except ValueError:
        # NOSILENT: a date the PDF states but that is no calendar date is a PDF with no date, the documented None
        return None


def pdf_reader(
    *,
    max_bytes: int = DEFAULT_PDF_MAX_BYTES,
    max_pages: int = DEFAULT_PDF_MAX_PAGES,
    parse_timeout_seconds: float = DEFAULT_PDF_PARSE_TIMEOUT_SECONDS,
) -> CarrierReader:
    """The reader that hands a fetched PDF to :func:`~threetears.agent.tools.document.parse_document`.

    No OCR: a scanned PDF reads as no text and says so. A body without the
    ``%PDF-`` signature in its first kilobyte is not parsed, whatever its
    declared type.

    :param max_bytes: the most of one PDF the fetch reads
    :ptype max_bytes: int
    :param max_pages: the most pages read; the rest are named in a note
    :ptype max_pages: int
    :param parse_timeout_seconds: how long the parse may take
    :ptype parse_timeout_seconds: float
    :return: the reader
    :rtype: CarrierReader
    """

    async def read(body: bytes) -> CarrierText:
        from threetears.agent.tools.document import DocumentParseError, OcrConfig, parse_document

        if b"%PDF-" not in body[:_PDF_SIGNATURE_WINDOW]:
            raise CarrierUnreadable(
                "the file says it is a PDF but is not one",
                next_step="find the document another way",
            )
        try:
            parsed = await asyncio.wait_for(
                parse_document(body, "application/pdf", ocr_config=OcrConfig(enabled=False), max_pages=max_pages),
                timeout=parse_timeout_seconds,
            )
        except TimeoutError as exc:
            raise CarrierUnreadable(
                f"reading the PDF took longer than {parse_timeout_seconds:g} s",
                next_step="find a shorter copy, or the same facts on a page",
            ) from exc
        except DocumentParseError as exc:
            raise CarrierUnreadable(
                f"the PDF could not be read ({exc.detail})",
                next_step="find the document another way",
            ) from exc
        total = parsed.page_count or 0
        pages_read = int(parsed.metadata.get("pages_read") or total)
        if total and pages_read < total:
            note = (
                f"A PDF of {total} pages: pages 1-{pages_read} were read, the most read from one PDF. "
                "The rest was not read."
            )
        else:
            note = f"A PDF of {total} page(s), all read."
        title = (parsed.metadata.get("title") or parsed.title or "").strip() or None
        return CarrierText(
            text=parsed.text,
            title=title,
            published_at=_pdf_date(parsed.metadata.get("creationDate")),
            notes=(note,),
        )

    return CarrierReader(kind="PDF", media_types=PDF_MEDIA_TYPES, max_bytes=max_bytes, read=read)


def create_web_fetch_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create a web fetch tool.

    delegates to :func:`threetears.agent.tools.langchain_adapter.to_langchain_tool`
    so the StructuredTool path and the NATS-dispatched ToolServer
    path share :meth:`WebFetchTool.execute` as their single
    execution body.

    Optional config keys:

    - ``max_chars``: max extracted text handed back (default 15000)
    - ``max_bytes``: hard cap on the fetched carrier
    - ``timeout_seconds``: per-fetch bound
    - ``respect_robots``: D12's stance, as deployment config (default True)
    - ``robots_unreadable``: ``deny`` (default) or ``allow``: whether a page whose
      ``robots.txt`` cannot be read is read anyway
    - ``user_agent``: the deployment's own name, sent with every fetch and matched
      by robots rules
    - ``pdf_max_bytes``, ``pdf_max_pages``, ``pdf_parse_timeout_seconds``: the PDF limits
    - ``thin_page_chars``: readable text under this makes a page thin, so it follows its frame
      (0 turns that off)
    - ``transport``: a prebuilt
      :class:`~threetears.search.contracts.transport.FetchTransport`
    - ``_user_timezone``: the zone of the person the turn is with, which a host injects

    :param config: tool configuration; every key optional
    :ptype config: dict[str, Any]
    :param description: the description the LLM sees
    :ptype description: str
    :return: the wrapped tool
    :rtype: StructuredTool
    """
    from threetears.agent.tools.langchain_adapter import to_langchain_tool

    return to_langchain_tool(
        WebFetchTool(
            max_chars=config.get("max_chars", DEFAULT_MAX_CHARS),
            max_bytes=config.get("max_bytes"),
            timeout_seconds=config.get("timeout_seconds"),
            respect_robots=config.get("respect_robots", True),
            transport=config.get("transport"),
            robots_unreadable=config.get("robots_unreadable") or ROBOTS_UNREADABLE_DENY,
            user_agent=config.get("user_agent") or None,
            pdf_max_bytes=config.get("pdf_max_bytes", DEFAULT_PDF_MAX_BYTES),
            pdf_max_pages=config.get("pdf_max_pages", DEFAULT_PDF_MAX_PAGES),
            pdf_parse_timeout_seconds=config.get("pdf_parse_timeout_seconds", DEFAULT_PDF_PARSE_TIMEOUT_SECONDS),
            thin_page_chars=config.get("thin_page_chars", DEFAULT_THIN_PAGE_CHARS),
            zone=config.get("_user_timezone") or None,
        ),
        description=description,
    )


class WebFetchTool(TearsTool):
    """TearsTool wrapper for fetching and extracting web page content.

    fetches a URL through the injected byte-capped transport, extracts the
    main content, and hands back the text with the typed candidate under the
    search-results metadata key.
    """

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch and extract content from",
            },
            "etag": {
                "type": "string",
                "description": (
                    "ETag from a previous fetch of this URL. When given, the fetch is "
                    "conditional: if the page is unchanged the result says so instead of "
                    "returning the body again."
                ),
            },
            "last_modified": {
                "type": "string",
                "description": (
                    "Last-Modified value from a previous fetch of this URL, echoed back "
                    "verbatim. Used with or instead of etag to make the fetch conditional."
                ),
            },
            # One wording for the argument, from the shared input model, so every
            # windowed tool describes its offset the same way.
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": WindowedInput.model_fields["offset"].description,
            },
        },
        "required": ["url"],
    }

    def __init__(
        self,
        max_chars: int = DEFAULT_MAX_CHARS,
        *,
        max_bytes: int | None = None,
        timeout_seconds: float | None = None,
        respect_robots: bool = True,
        transport: FetchTransport | None = None,
        robots_unreadable: str = ROBOTS_UNREADABLE_DENY,
        user_agent: str | None = None,
        pdf_max_bytes: int = DEFAULT_PDF_MAX_BYTES,
        pdf_max_pages: int = DEFAULT_PDF_MAX_PAGES,
        pdf_parse_timeout_seconds: float = DEFAULT_PDF_PARSE_TIMEOUT_SECONDS,
        thin_page_chars: int | None = DEFAULT_THIN_PAGE_CHARS,
        zone: str | None = None,
    ) -> None:
        """initialize web fetch tool.

        :param max_chars: maximum extracted characters handed to the caller
        :ptype max_chars: int
        :param max_bytes: hard cap on one fetched carrier (SR-G5); None
            takes Extract's default
        :ptype max_bytes: int | None
        :param timeout_seconds: per-fetch bound (SR-G2); None takes the
            transport's configured value
        :ptype timeout_seconds: float | None
        :param respect_robots: honour ``robots.txt`` (D12's default). A
            deployment setting, never a per-call choice
        :ptype respect_robots: bool
        :param transport: the byte-capped fetch seam; None builds the
            host's default via
            :func:`~threetears.agent.tools.search_transport.build_fetch_transport`
        :ptype transport: FetchTransport | None
        :param robots_unreadable: ``deny`` or ``allow``: whether a page whose ``robots.txt``
            cannot be read is read anyway. A deployment setting, never a per-call choice
        :ptype robots_unreadable: str
        :param user_agent: the deployment's name, sent as ``User-Agent`` by the default
            transport and matched by robots rules; None is the leaf's own name
        :ptype user_agent: str | None
        :param pdf_max_bytes: the most of one PDF read
        :ptype pdf_max_bytes: int
        :param pdf_max_pages: the most pages read from one PDF
        :ptype pdf_max_pages: int
        :param pdf_parse_timeout_seconds: how long one PDF may take to parse
        :ptype pdf_parse_timeout_seconds: float
        :param thin_page_chars: readable text under this makes a page thin, so it follows its
            frame; None or 0 never follows
        :ptype thin_page_chars: int | None
        :param zone: the IANA zone a time of day is shown in
        :ptype zone: str | None
        :return: nothing
        :rtype: None
        :raises ValueError: when ``robots_unreadable`` is neither ``deny`` nor ``allow``
        """
        if robots_unreadable not in ("deny", "allow"):
            raise ValueError(f"robots_unreadable must be 'deny' or 'allow', got {robots_unreadable!r}")
        if transport is None:
            from threetears.agent.tools.search_transport import build_fetch_transport

            # The transport's own ceiling is the largest carrier this tool reads, so the cap that
            # refuses is always one this tool names in its message: a page's, or a PDF's.
            transport = build_fetch_transport(
                timeout_seconds=timeout_seconds,
                user_agent=user_agent,
                max_response_bytes=max(max_bytes or DEFAULT_MAX_BYTES, pdf_max_bytes),
            )
        self._transport = transport
        self._max_chars = max_chars
        self._max_bytes = max_bytes
        self._timeout_seconds = timeout_seconds
        self._respect_robots = respect_robots
        self._robots_unreadable = robots_unreadable
        self._user_agent = user_agent or DEFAULT_USER_AGENT
        self._pdf = pdf_reader(
            max_bytes=pdf_max_bytes, max_pages=pdf_max_pages, parse_timeout_seconds=pdf_parse_timeout_seconds
        )
        self._thin_page_chars = thin_page_chars or None
        self._zone = zone

    async def execute(self, **kwargs: Any) -> ToolResult:
        """fetch and extract content from URL.

        :param kwargs: must include 'url' key with URL to fetch
        :ptype kwargs: Any
        :return: result carrying the extracted text, and the typed candidate
            under the search-results metadata key
        :rtype: ToolResult
        """
        url = kwargs.get("url", "")
        if not url:
            return self._failed(url, "web_fetch requires a 'url' argument")

        extra: dict[str, Any] = {}
        if self._max_bytes is not None:
            extra["max_bytes"] = self._max_bytes

        # The caller holds the bytes (D7) and therefore holds the validators. It
        # sends them WITHOUT its copy of the text -- sending the body back to
        # revalidate it would spend the very bytes the conditional request exists
        # to save. So the slot carries an empty text: this tool genuinely has no
        # content until upstream either confirms the caller's or replaces it.
        caller_etag = kwargs.get("etag") or None
        caller_last_modified = kwargs.get("last_modified") or None
        candidate = _candidate_for(url)
        if caller_etag or caller_last_modified:
            candidate = candidate.model_copy(
                update={
                    "content": ContentSlot(
                        text="",
                        origin="later-fetch",
                        etag=caller_etag,
                        last_modified=caller_last_modified,
                    )
                }
            )
            extra["revalidate"] = True

        try:
            fetched = await extract(
                candidate,
                transport=self._transport,
                timeout_seconds=self._timeout_seconds,
                respect_robots=self._respect_robots,
                user_agent=self._user_agent,
                robots_unreadable="allow" if self._robots_unreadable == "allow" else "deny",
                readers=(self._pdf,),
                thin_page_chars=self._thin_page_chars,
                **extra,
            )
        except SearchFailure as failure:
            # Extract raises only for a refusal that applies to the whole
            # run -- today, the missing [extract] extra. Per-candidate
            # trouble comes back as a marked candidate, below.
            record = failure.to_record()
            message = f"fetch refused ({record.failure_class}): {record.message}"
            if record.remediation:
                message = f"{message}\n{record.remediation}"
            return ToolResult(
                success=False,
                content=message,
                # the typed record, not just its spend: a refusal is exactly
                # the case a consumer must not have to parse prose for, and
                # ``from_candidate_set`` has no slot for one (D10, SR-E3).
                metadata=_refusal_metadata(url, record),
                error=message,
            )

        status = fetched.facets.get(EXTRACTION_STATUS_FACET)
        candidate_set = CandidateSet(candidates=(fetched,))
        # ``unchanged`` is a SUCCESS that produced no new content (D30): upstream
        # confirmed the caller's copy, and the content slot the caller passed in
        # comes back untouched. Reading it as failure would report "no readable
        # content" for the one outcome that means the content is definitively
        # good -- the misreading the constant's own docstring warns about, and
        # the reason this reader was named in the task doc before it was written.
        readable = {EXTRACTION_STATUS_COMPLETE, EXTRACTION_STATUS_UNCHANGED}
        if fetched.content is None or status not in readable:
            message = _not_read(url, fetched.facets)
            return ToolResult(
                success=False,
                content=message,
                metadata=_metadata(url, candidate_set),
                error=message.split("\n", 1)[0],
            )

        if status == EXTRACTION_STATUS_UNCHANGED:
            # Prose for the model, structure for the program -- the ordinary Bind
            # split (SR-A1). The model must not be handed an empty string, which
            # reads as "the page was blank" rather than "your copy is current",
            # and the caller reads the typed status off metadata either way.
            return ToolResult(
                success=True,
                content=f"{url} is unchanged since your copy. Use the copy you have.",
                metadata=_metadata(url, candidate_set),
                error=None,
            )

        # A page longer than the bound is windowed, not cut: the note, which comes
        # first, names the call that returns the next part, and nothing is discarded. Before this,
        # a whitepaper came back cut at exactly 15,000 characters and was
        # discussed as if whole (metallm conv 01a097bc, 2026-09-16).
        window = window_text(fetched.content.text, offset=int(kwargs.get("offset") or 0), max_chars=self._max_chars)
        metadata = _metadata(url, candidate_set)
        metadata["window"] = {
            "offset": window.offset,
            "total_chars": window.total,
            "next_offset": window.next_offset,
        }
        final_url = str(fetched.facets.get(EXTRACTION_FINAL_URL_FACET) or url)
        dated = shown_date(fetched.published_at, self._zone)
        head = (
            f'Read "{fetched.title or "Untitled page"}" ({f"published {dated}" if dated else "no date given"}) '
            f"at {final_url}: characters {window.offset:,}-{window.offset + len(window.text):,} of {window.total:,}."
        )
        notes = [str(note) for note in _as_list(fetched.facets.get(EXTRACTION_NOTES_FACET))]
        body = window.rendered(tool="web_fetch", note_first=True, notes=notes)
        lead = "\n" if (not window.is_whole or notes) else "\n\n"
        return ToolResult(
            success=True,
            content=f"{head}{lead}{body}",
            metadata=metadata,
            error=None,
        )

    def _failed(self, url: str, message: str) -> ToolResult:
        """A failed result that still carries the metadata shape callers read.

        :param url: the URL the call was about, as asked for
        :ptype url: str
        :param message: what went wrong, in prose
        :ptype message: str
        :return: the failed result
        :rtype: ToolResult
        """
        return ToolResult(
            success=False,
            content=message,
            metadata=_metadata(url, CandidateSet(spend=Spend())),
            error=message,
        )

    def mcp_schema(self) -> MCPToolDefinition:
        """return MCP-compatible tool definition for web fetch.

        :return: tool definition with name, version, description, input schema
        :rtype: MCPToolDefinition
        """
        result = MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="fetch URL and extract main page content",
            input_schema=self._INPUT_SCHEMA,
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "threetears.web_fetch"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _candidate_for(url: str) -> Candidate:
    """The one-candidate input Extract's web path expects.

    A direct fetch has no search behind it, and the provenance says so
    rather than inventing a query: the ``query`` slot carries the URL that
    was asked for, and the provider instance names this tool.

    :param url: the URL to fetch
    :ptype url: str
    :return: a content-free candidate addressed at ``url``
    :rtype: Candidate
    """
    return Candidate(
        identity=url,
        locators=(Locator(url=url),),
        provenance=Provenance(
            query=url,
            provider_instance=_FETCH_PROVIDER_INSTANCE,
            retrieved_at=datetime.now(UTC),
        ),
    )


def _metadata(url: str, candidate_set: CandidateSet) -> dict[str, Any]:
    """Project one fetch onto the search-results border key (D22).

    :param url: the URL asked for, which stands as the projection's query
    :ptype url: str
    :param candidate_set: the fetched candidate, or an empty set on refusal
    :ptype candidate_set: CandidateSet
    :return: ``{SEARCH_RESULTS_METADATA_KEY: ...}``
    :rtype: dict[str, Any]
    """
    return project_metadata(url, candidate_set)


def _refusal_metadata(url: str, record: FailureRecord) -> dict[str, Any]:
    """Project a whole-run refusal onto the same border key, as a record.

    :param url: the URL asked for, which stands as the projection's query
    :ptype url: str
    :param record: the typed failure's wire record, spend included
    :ptype record: FailureRecord
    :return: ``{SEARCH_RESULTS_METADATA_KEY: ...}``
    :rtype: dict[str, Any]
    """
    return project_failure_metadata(url, record)


def _as_list(value: object) -> list[object]:
    """A facet that holds a list, as one; anything else as an empty list."""
    return list(value) if isinstance(value, list) else []


def _not_read(url: str, facets: dict[str, Any]) -> str:
    """What a fetch that read nothing tells its reader: why, then what to do next, on the first line.

    :param url: the URL asked for
    :ptype url: str
    :param facets: the candidate's extraction facets
    :ptype facets: dict[str, Any]
    :return: the message
    :rtype: str
    """
    reason = facets.get(EXTRACTION_REASON_FACET) or "the fetch got no readable text, and did not say why"
    next_step = facets.get(EXTRACTION_NEXT_STEP_FACET)
    lines = [f"Did not read {url}: {reason}; {next_step}." if next_step else f"Did not read {url}: {reason}."]
    held = facets.get(EXTRACTION_THIN_TEXT_FACET)
    if held:
        lines.extend(["", "What the page itself held:", str(held)])
    return "\n".join(lines)

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

The ``credential_resolver`` hook the old body carried is gone. Its only path
in was a ``_credential_resolver`` key in the tool's config dict, which
nothing in the family ever set; carrying an unexercised credential-injection
seam through the rewrite would have meant re-deriving its failure semantics
for no caller.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.media.contracts import EXTRACTION_STATUS_COMPLETE, EXTRACTION_STATUS_UNCHANGED
from threetears.search.contracts import (
    SEARCH_RESULTS_METADATA_KEY,
    Candidate,
    CandidateSet,
    ContentSlot,
    FailureRecord,
    Locator,
    Provenance,
    SearchFailure,
    SearchResultsMetadata,
    Spend,
)
from threetears.search.extract import EXTRACTION_STATUS_FACET, extract

if TYPE_CHECKING:
    from threetears.search.contracts import FetchTransport

__all__ = [
    "DEFAULT_MAX_CHARS",
    "WebFetchInput",
    "WebFetchTool",
    "create_web_fetch_tool",
]

#: how much extracted text reaches the caller. A context bound, not a
#: transport one: the byte cap that protects the *host* lives on the fetch
#: transport and fires long before this does.
DEFAULT_MAX_CHARS: Final[int] = 15000

#: what the provenance of a directly-fetched page names as its source. Not a
#: search provider -- nothing searched -- so it names the tool that asked.
_FETCH_PROVIDER_INSTANCE: Final[str] = "threetears.web_fetch"

#: appended when the extracted text is cut to :data:`DEFAULT_MAX_CHARS`.
_TRUNCATION_MARK: Final[str] = "\n\n[Content truncated]"


class WebFetchInput(BaseModel):
    """Input for the web fetch tool."""

    url: str = Field(description="URL to fetch and extract content from")
    etag: str | None = Field(
        default=None,
        description=(
            "ETag from a previous fetch of this URL. When given, the fetch is conditional: "
            "if the page is unchanged the result says so instead of returning the body again."
        ),
    )
    last_modified: str | None = Field(
        default=None,
        description=(
            "Last-Modified value from a previous fetch of this URL, echoed back verbatim. "
            "Used with or instead of etag to make the fetch conditional."
        ),
    )


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
    - ``transport``: a prebuilt
      :class:`~threetears.search.contracts.transport.FetchTransport`

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
        ),
        description=description,
        args_schema=WebFetchInput,
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
        :return: nothing
        :rtype: None
        """
        if transport is None:
            from threetears.agent.tools.search_transport import build_fetch_transport

            transport = build_fetch_transport(timeout_seconds=timeout_seconds)
        self._transport = transport
        self._max_chars = max_chars
        self._max_bytes = max_bytes
        self._timeout_seconds = timeout_seconds
        self._respect_robots = respect_robots

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
            message = f"no readable content extracted from {url} (extraction_status: {status})"
            return ToolResult(
                success=False,
                content=message,
                metadata=_metadata(url, candidate_set),
                error=message,
            )

        if status == EXTRACTION_STATUS_UNCHANGED:
            # Prose for the model, structure for the program -- the ordinary Bind
            # split (SR-A1). The model must not be handed an empty string, which
            # reads as "the page was blank" rather than "your copy is current",
            # and the caller reads the typed status off metadata either way.
            return ToolResult(
                success=True,
                content=f"{url} is unchanged since your copy; upstream confirmed it. Use the copy you hold.",
                metadata=_metadata(url, candidate_set),
                error=None,
            )

        text = fetched.content.text
        if len(text) > self._max_chars:
            # ``max(0, ...)``: a bound below the marker's own length made this
            # index negative, and a negative slice keeps the *end* of the
            # string -- returning the tail of the page, longer than the bound
            # that was asked for. Under the marker's length the honest answer
            # is the marker alone, cut to the bound.
            keep = max(0, self._max_chars - len(_TRUNCATION_MARK))
            text = (text[:keep] + _TRUNCATION_MARK)[: self._max_chars]
        return ToolResult(
            success=True,
            content=text,
            metadata=_metadata(url, candidate_set),
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
    projection = SearchResultsMetadata.from_candidate_set(query=url, candidate_set=candidate_set)
    return {SEARCH_RESULTS_METADATA_KEY: projection.to_metadata()}


def _refusal_metadata(url: str, record: FailureRecord) -> dict[str, Any]:
    """Project a whole-run refusal onto the same border key, as a record.

    :param url: the URL asked for, which stands as the projection's query
    :ptype url: str
    :param record: the typed failure's wire record, spend included
    :ptype record: FailureRecord
    :return: ``{SEARCH_RESULTS_METADATA_KEY: ...}``
    :rtype: dict[str, Any]
    """
    projection = SearchResultsMetadata.from_failure(query=url, failure=record)
    return {SEARCH_RESULTS_METADATA_KEY: projection.to_metadata()}

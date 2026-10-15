"""Web search, on the search leaf (search-spec.md §4.2, check 8).

The tool's *identity* is unchanged and deliberately so: ``threetears.web_search``,
the :class:`~threetears.agent.tools.base_tool.TearsTool` ABC, the
:class:`~threetears.agent.tools.base_tool.ToolResult` shape, a ``query``
string in and readable prose out. Every existing caller keeps working without
being told anything happened.

What changed is everything under that: this module used to open its own
``httpx.Client`` on a 15-second hardcode, inside an ``async execute`` that
blocked the event loop for the duration, and report failure by prefixing the
content string with ``[TOOL ERROR]`` for callers to match on. All three are
gone (§10 defects 2 and 8). The call is now
:func:`threetears.search.bind_search` over an injected transport: bounded
retry, a configurable timeout, circuit-breaking, egress selection and the
SSRF guards all come from the transport, and failures arrive as one of the
seven typed classes and leave as a failed :class:`ToolResult` carrying its
own spend.

**Prose is unchanged in kind, structure is additive.** ``content`` is still
the human/LLM-readable rendering, now produced by the leaf's prose binding
rather than hand-formatted here. What is new is
:attr:`ToolResult.metadata`: the typed candidates, per-criterion
dispositions, spend, notices and -- on a failure -- the failure record, under
:data:`~threetears.search.contracts.SEARCH_RESULTS_METADATA_KEY` (D22). A
consumer that wants structure reads that key instead of parsing the prose
back apart, and it survives the NATS hop intact because
``CallResponse.metadata`` carries it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit

from langchain_core.tools import StructuredTool

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.search.adapters.searxng import (
    SEARXNG_CATEGORIES,
    SEARXNG_NAMESPACE,
    SEARXNG_RELATIVE_TIME_RANGES,
    SearxngAdapter,
)
from threetears.search.bind import bind_search
from threetears.search.contracts import (
    SEARCH_RESULTS_METADATA_KEY,
    Criterion,
    SearchRequest,
    SearchResultsMetadata,
)

if TYPE_CHECKING:
    from threetears.search.contracts import BudgetPort, RateLimiterPort, SearchTransport

__all__ = [
    "DEFAULT_MAX_RESULTS",
    "WebSearchTool",
    "create_web_search_tool",
]

#: how many candidates one call asks for, and renders. The old formatter
#: sliced the provider's response to ten *after* it arrived; this is the same
#: number stated as a criterion the adapter can push down and answer for.
DEFAULT_MAX_RESULTS: Final[int] = 10

#: A domain as ``site`` takes it: labels of letters, digits and hyphens, at least two.
_DOMAIN: Final[re.Pattern[str]] = re.compile(
    r"^(?=.{1,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{1,62}$"
)


def _site_of(raw: str) -> str | None:
    """The bare domain a ``site`` argument names, or None when it names none.

    ``https://www.nature.com/articles`` and ``nature.com`` both name ``nature.com``.

    :param raw: what the caller passed
    :ptype raw: str
    :return: the lower-cased domain, without ``www.``
    :rtype: str | None
    """
    text = raw.strip().lower()
    if "//" in text:
        text = urlsplit(text).hostname or ""
    text = text.split("/", 1)[0].removeprefix("www.").rstrip(".")
    return text if _DOMAIN.match(text) else None


def create_web_search_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create a web search tool.

    delegates to :func:`threetears.agent.tools.langchain_adapter.to_langchain_tool`
    so the StructuredTool path and the NATS-dispatched ToolServer
    path share :meth:`WebSearchTool.execute` as their single
    execution body. Config must include ``base_url`` pointing to a
    SearXNG instance.

    Optional config keys, all deployment facts rather than per-call choices:
    ``max_results``, ``timeout_seconds``, ``transport`` (a prebuilt
    :class:`~threetears.search.contracts.transport.SearchTransport`),
    ``budget`` and ``limiter``; and ``_user_timezone``, the zone of the person
    the turn is with, which a host injects.

    :param config: tool configuration; ``base_url`` is required
    :ptype config: dict[str, Any]
    :param description: the description the LLM sees
    :ptype description: str
    :return: the wrapped tool
    :rtype: StructuredTool
    :raises ValueError: when ``base_url`` is absent
    """
    from threetears.agent.tools.langchain_adapter import to_langchain_tool

    base_url = config.get("base_url")
    if not base_url:
        raise ValueError("web_search requires 'base_url' in config")
    return to_langchain_tool(
        WebSearchTool(
            base_url=base_url,
            transport=config.get("transport"),
            max_results=config.get("max_results", DEFAULT_MAX_RESULTS),
            timeout_seconds=config.get("timeout_seconds"),
            budget=config.get("budget"),
            limiter=config.get("limiter"),
            zone=config.get("_user_timezone") or None,
        ),
        description=description,
    )


class WebSearchTool(TearsTool):
    """TearsTool wrapper for web search via SearXNG, over the search leaf.

    performs web searches against a configured SearXNG instance and
    returns rendered prose plus the typed result under
    :data:`~threetears.search.contracts.SEARCH_RESULTS_METADATA_KEY`.
    """

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "search query",
            },
            "category": {
                "type": "string",
                "enum": list(SEARXNG_CATEGORIES),
                "description": (
                    "What kind of results: news for news stories, science for papers, it for software and "
                    "programming, general (the default) for the whole web."
                ),
            },
            "time_range": {
                "type": "string",
                "enum": list(SEARXNG_RELATIVE_TIME_RANGES),
                "description": "Only results published in the last day, week, month or year. Omit for any date.",
            },
            "page": {
                "type": "integer",
                "minimum": 1,
                "description": "Which page of results, from 1. Ask for the next page when this one did not have it.",
            },
            "site": {
                "type": "string",
                "description": "Search one site only, by its domain, for example nature.com.",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        base_url: str,
        *,
        transport: SearchTransport | None = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout_seconds: float | None = None,
        budget: BudgetPort | None = None,
        limiter: RateLimiterPort | None = None,
        zone: str | None = None,
    ) -> None:
        """initialize web search tool against one configured SearXNG instance.

        The transport is injectable and defaults to the host-side adapter
        over the family's traced client -- a pod that wants the ambient
        egress, a shared circuit breaker or different bounds constructs one
        and passes it. Nothing here reads the environment: the base URL is
        deployment config arriving through the caller (D21, SR-K1).

        :param base_url: base URL of the SearXNG instance (trailing slash
            stripped)
        :ptype base_url: str
        :param transport: the transport every request goes through; None
            builds a
            :class:`~threetears.agent.tools.search_transport.TracedSearchTransport`
            for ``base_url``
        :ptype transport: SearchTransport | None
        :param max_results: how many candidates one call asks for and renders
        :ptype max_results: int
        :param timeout_seconds: bound for one search (SR-G2); None applies
            the leaf's default
        :ptype timeout_seconds: float | None
        :param budget: the refusal authority consulted before the call and
            debited after it (D4, D5); None makes no consultation at all
        :ptype budget: BudgetPort | None
        :param limiter: the pacing seam for this provider's egress (D8);
            None leaves the call unpaced
        :ptype limiter: RateLimiterPort | None
        :param zone: the IANA zone of the person the agent works for; a result's time of day is
            shown in it
        :ptype zone: str | None
        :return: nothing
        :rtype: None
        """
        self.base_url = base_url.rstrip("/")
        if transport is None:
            from threetears.agent.tools.search_transport import TracedSearchTransport

            transport = TracedSearchTransport(base_url=self.base_url, timeout_seconds=timeout_seconds)
        self._transport = transport
        self._provider = SearxngAdapter(base_url=self.base_url, transport=transport)
        self._max_results = max_results
        self._timeout_seconds = timeout_seconds
        self._budget = budget
        self._limiter = limiter
        self._zone = zone

    async def execute(self, **kwargs: Any) -> ToolResult:
        """perform one web search.

        Never raises: :func:`~threetears.search.bind_search` carries D10's
        unconditional guarantee, so a refusal, a timeout or an unmapped
        provider defect all arrive here as a rendered failure that still
        accounts for what it spent. An argument this search cannot take is
        answered before anything is sent, with what it does take.

        The first line says what was searched and how many results came back;
        each result then carries its date and the engines that found it.

        :param kwargs: ``query``, and optionally ``category``, ``time_range``, ``page``, ``site``
        :ptype kwargs: Any
        :return: result carrying rendered prose, and the typed result under
            the search-results metadata key
        :rtype: ToolResult
        """
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return _refused("web_search needs a query: say what to search for.")
        criteria: list[Criterion] = [Criterion.max_results(self._max_results)]
        said: list[str] = []
        category = kwargs.get("category")
        if category:
            if category not in SEARXNG_CATEGORIES:
                return _refused(
                    f"web_search has no category {category!r}. The categories: {', '.join(SEARXNG_CATEGORIES)}."
                )
            criteria.append(Criterion.namespaced(SEARXNG_NAMESPACE, "categories", category))
            said.append(f"in {category}")
        time_range = kwargs.get("time_range")
        if time_range:
            if time_range not in SEARXNG_RELATIVE_TIME_RANGES:
                return _refused(
                    f"web_search has no time_range {time_range!r}. It takes "
                    f"{', '.join(SEARXNG_RELATIVE_TIME_RANGES)}, or none for any date."
                )
            criteria.append(Criterion.namespaced(SEARXNG_NAMESPACE, "time-range", time_range))
            said.append(f"published in the last {time_range}")
        page = kwargs.get("page")
        page_number = 1
        if page not in (None, ""):
            if isinstance(page, bool) or not isinstance(page, int) or page < 1:
                return _refused(f"web_search's page is a whole number from 1; {page!r} is not.")
            page_number = page
            if page > 1:
                criteria.append(Criterion.namespaced(SEARXNG_NAMESPACE, "page", page))
        sent = query
        site = kwargs.get("site")
        if site:
            domain = _site_of(str(site))
            if domain is None:
                return _refused(f"web_search's site is a domain such as nature.com; {site!r} is not one.")
            # Pushed down as the engines' own operator, and held here as a filter, because an
            # engine that ignores the operator would otherwise hand back other sites' pages.
            sent = f"{query} site:{domain}"
            criteria.append(Criterion.domains_include([domain]))
            said.append(f"on {domain}")
        rendered = await bind_search(
            SearchRequest(query=sent, criteria=tuple(criteria)),
            provider=self._provider,
            timeout_seconds=self._timeout_seconds,
            max_candidates=self._max_results,
            budget=self._budget,
            limiter=self._limiter,
            egress=self._transport.egress_name,
            zone=self._zone,
        )
        if not rendered.success:
            return ToolResult(
                success=False,
                content=rendered.content,
                metadata=rendered.metadata,
                error=rendered.error,
            )
        found = _count(rendered.metadata)
        scope = f" {', '.join(said)}" if said else ""
        head = f'Searched the web for "{query}"{scope}: {found} result(s) on page {page_number}.'
        if found >= self._max_results:
            head += f" More may be on page {page_number + 1}."
        return ToolResult(
            success=True,
            content=f"{head}\n\n{rendered.content}",
            metadata=rendered.metadata,
            error=None,
        )

    def mcp_schema(self) -> MCPToolDefinition:
        """return MCP-compatible tool definition for web search.

        :return: tool definition with name, version, description, input schema
        :rtype: MCPToolDefinition
        """
        result = MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="search web using SearXNG and return formatted results",
            input_schema=self._INPUT_SCHEMA,
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "threetears.web_search"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _refused(message: str) -> ToolResult:
    """A search that was not sent, because an argument cannot be used; the message says what can.

    :param message: what was wrong and what the search takes
    :ptype message: str
    :return: the failed result
    :rtype: ToolResult
    """
    return ToolResult(success=False, content=message, metadata={}, error=message)


def _count(metadata: dict[str, Any]) -> int:
    """How many results a rendered search holds, read off its typed projection.

    :param metadata: the rendered search's metadata
    :ptype metadata: dict[str, Any]
    :return: the candidate count, 0 when there is no projection
    :rtype: int
    """
    raw = metadata.get(SEARCH_RESULTS_METADATA_KEY)
    if raw is None:
        return 0
    return len(SearchResultsMetadata.from_metadata(raw).candidates)

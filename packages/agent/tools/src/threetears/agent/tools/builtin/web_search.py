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

from typing import TYPE_CHECKING, Any, Final

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.search.adapters.searxng import SearxngAdapter
from threetears.search.bind import bind_search
from threetears.search.contracts import Criterion, SearchRequest

if TYPE_CHECKING:
    from threetears.search.contracts import BudgetPort, RateLimiterPort, SearchTransport

__all__ = [
    "DEFAULT_MAX_RESULTS",
    "WebSearchInput",
    "WebSearchTool",
    "create_web_search_tool",
]

#: how many candidates one call asks for, and renders. The old formatter
#: sliced the provider's response to ten *after* it arrived; this is the same
#: number stated as a criterion the adapter can push down and answer for.
DEFAULT_MAX_RESULTS: Final[int] = 10


class WebSearchInput(BaseModel):
    """Input for the web search tool."""

    query: str = Field(description="Search query")


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
    ``budget`` and ``limiter``.

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
        ),
        description=description,
        args_schema=WebSearchInput,
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

    async def execute(self, **kwargs: Any) -> ToolResult:
        """perform one web search.

        Never raises: :func:`~threetears.search.bind_search` carries D10's
        unconditional guarantee, so a refusal, a timeout or an unmapped
        provider defect all arrive here as a rendered failure that still
        accounts for what it spent.

        :param kwargs: must include 'query' key with search query string
        :ptype kwargs: Any
        :return: result carrying rendered prose, and the typed result under
            the search-results metadata key
        :rtype: ToolResult
        """
        query = kwargs.get("query", "")
        rendered = await bind_search(
            SearchRequest(
                query=query,
                criteria=(Criterion.max_results(self._max_results),),
            ),
            provider=self._provider,
            timeout_seconds=self._timeout_seconds,
            max_candidates=self._max_results,
            budget=self._budget,
            limiter=self._limiter,
            egress=self._transport.egress_name,
        )
        return ToolResult(
            success=rendered.success,
            content=rendered.content,
            metadata=rendered.metadata,
            error=rendered.error,
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

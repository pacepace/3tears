"""MCP (Model Context Protocol) client for tool discovery and invocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from threetears.core.http_client import TracedHttpClient, UpstreamHttpError
from threetears.observe import get_logger, traced

__all__ = [
    "McpClient",
    "McpTool",
    "McpToolResult",
]

log = get_logger(__name__)


def _get_mcp_timeout() -> int:
    """read MCP timeout from environment or return platform default.

    env var: THREETEARS_MCP_TIMEOUT

    :return: MCP timeout in seconds
    :rtype: int
    """
    import os

    raw = os.environ.get("THREETEARS_MCP_TIMEOUT")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            # An operator deliberately set this and is silently getting the default instead --
            # "30s" or "2m" lands here. Warning, because the symptom is timeouts at the wrong
            # value with the environment appearing to say otherwise.
            log.warning(
                "THREETEARS_MCP_TIMEOUT is not an integer; using the default instead",
                extra={"extra_data": {"raw": raw[:32], "default_seconds": 120}},
            )
    return 120


@dataclass
class McpTool:
    """Descriptor for an MCP-exposed tool."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class McpToolResult:
    """Result of an MCP tool invocation.

    ``metadata`` carries the spec's ``structuredContent`` when the server
    sent one. It was missing, so a structured result reached this client as
    prose in ``content`` and its structure had to be re-parsed out of the
    rendering -- or, more often, was simply lost. Optional and defaulted, so
    a server that sends only text produces exactly what it did before
    (search-spec.md §4.8).
    """

    success: bool
    content: str
    error: str | None = None
    metadata: dict[str, Any] | None = None


#: the spec paths this client speaks, joined onto the traced client's base url.
_LIST_PATH = "/mcp/v1/tools/list"
_CALL_PATH = "/mcp/v1/tools/call"

#: what a request can fail with once the traced client has normalised the transport.
#: ``UpstreamHttpError`` is retry exhaustion; ``httpx.HTTPError`` covers the 4xx
#: ``raise_for_status`` and any transport error a caller-supplied client did not
#: normalise; ``ValueError`` covers a body that is not JSON (``JSONDecodeError``
#: subclasses it); ``KeyError`` covers a tool descriptor missing its ``name``.
_REQUEST_ERRORS = (UpstreamHttpError, httpx.HTTPError, ValueError, KeyError)


def _describe_failure(exc: Exception, base_url: str) -> str:
    """render a request failure as text an agent can act on.

    ``TracedHttpClient`` collapses every transport failure into a status-less
    :class:`UpstreamHttpError`, so the chained cause is what separates "the
    server took too long" from "the server was not there" -- two failures whose
    remedies differ and which the caller reports onward verbatim.

    :param exc: exception raised by the request
    :ptype exc: Exception
    :param base_url: mcp server base url, for the unreachable message
    :ptype base_url: str
    :return: single-sentence description of what failed
    :rtype: str
    """
    timed_out = isinstance(exc, httpx.TimeoutException) or (
        isinstance(exc, UpstreamHttpError) and isinstance(exc.__cause__, httpx.TimeoutException)
    )
    status: int | None = None
    if isinstance(exc, UpstreamHttpError):
        status = exc.status_code
    elif isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code

    if timed_out:
        described = "MCP tool invocation timed out"
    elif status is not None:
        described = f"MCP server returned HTTP {status}"
    elif isinstance(exc, UpstreamHttpError | httpx.HTTPError):
        described = f"MCP server at {base_url} could not be reached"
    else:
        described = f"MCP server returned a response this client could not read: {exc}"
    return described


class McpClient:
    """HTTP client for MCP-compatible tool servers.

    Communicates via the ``/mcp/v1/tools/list`` and ``/mcp/v1/tools/call``
    endpoints over :class:`threetears.core.http_client.TracedHttpClient`, which
    owns the socket and supplies the tracing, per-attempt accounting, and
    bounded-retry policy this client would otherwise re-roll.

    :cvar MAX_ATTEMPTS: attempts the default transport makes per request. ONE,
        deliberately: ``tools/call`` invokes a tool whose side effects the
        client cannot see, so a silent second attempt after a timeout can
        double-execute it. A caller that knows its tools are idempotent can
        inject a client configured otherwise.
    """

    MAX_ATTEMPTS: int = 1

    def __init__(
        self,
        base_url: str,
        timeout: int | None = None,
        *,
        http_client: TracedHttpClient | None = None,
    ) -> None:
        """construct mcp client.

        :param base_url: mcp server base url (trailing slash stripped)
        :ptype base_url: str
        :param timeout: request timeout in seconds, or ``None`` to fall back to
            ``THREETEARS_MCP_TIMEOUT`` env var / platform default
        :ptype timeout: int | None
        :param http_client: optional pre-built traced client; when supplied,
            caller owns its lifecycle and its configuration (base url, retry
            policy, egress) governs. primarily for tests that bind a mock
            transport, and for hosts that route every upstream through one
            configured client
        :ptype http_client: TracedHttpClient | None
        :return: nothing
        :rtype: None
        """
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout if timeout is not None else _get_mcp_timeout()
        if http_client is not None:
            self._http = http_client
        else:
            self._http = TracedHttpClient(
                upstream_base_url=self.base_url,
                timeout=float(self._timeout),
                max_attempts=self.MAX_ATTEMPTS,
            )

    async def close(self) -> None:
        """close the underlying traced http client.

        :return: nothing
        :rtype: None
        """
        await self._http.aclose()

    @traced()
    async def list_tools(self) -> list[McpTool]:
        """discover available tools from the mcp server.

        a server that cannot be reached, or that answers with something this
        client cannot read, yields an EMPTY list rather than an exception --
        tool discovery is best-effort and a caller mid-bootstrap has no better
        answer than "no tools from this server". the specific failure is logged
        at error level so the empty list is never the only evidence.

        :return: discovered tool descriptors; empty when discovery failed
        :rtype: list[McpTool]
        """
        result: list[McpTool] = []
        try:
            resp = await self._http.post(_LIST_PATH, json={})
            resp.raise_for_status()
            data = resp.json()
            result = [
                McpTool(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                )
                for t in data.get("tools", [])
            ]
        except _REQUEST_ERRORS as exc:
            log.error(
                "Failed to list MCP tools",
                extra={
                    "extra_data": {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "url": self.base_url,
                        "description": _describe_failure(exc, self.base_url),
                    }
                },
            )
        return result

    @traced(record_args=True)
    async def invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> McpToolResult:
        """invoke a tool on the mcp server.

        :param tool_name: name of the tool to invoke
        :ptype tool_name: str
        :param arguments: tool arguments, as the server's input schema declares
        :ptype arguments: dict[str, Any]
        :return: invocation result; ``success`` is False and ``error`` names
            what failed when the call did not complete
        :rtype: McpToolResult
        """
        try:
            resp = await self._http.post(
                _CALL_PATH,
                json={"name": tool_name, "arguments": arguments},
            )
            resp.raise_for_status()
            data = resp.json()
        except _REQUEST_ERRORS as exc:
            log.error(
                "MCP tool invocation failed",
                extra={
                    "extra_data": {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "tool_name": tool_name,
                        "url": self.base_url,
                    }
                },
            )
            result = McpToolResult(success=False, content="", error=_describe_failure(exc, self.base_url))
        else:
            content_parts = data.get("content", [])
            text_content = "\n".join(part.get("text", "") for part in content_parts if part.get("type") == "text")
            structured = data.get("structuredContent")
            result = McpToolResult(
                success=not data.get("isError", False),
                content=text_content,
                metadata=structured if isinstance(structured, dict) else None,
            )
        return result

    async def test_connection(self) -> bool:
        """report whether the mcp server answers a tool listing.

        :return: whether the server responded successfully
        :rtype: bool
        """
        reachable = False
        try:
            resp = await self._http.post(_LIST_PATH, json={})
            resp.raise_for_status()
            reachable = True
        except _REQUEST_ERRORS as exc:
            log.warning(
                "MCP connection test failed",
                extra={
                    "extra_data": {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "url": self.base_url,
                        "description": _describe_failure(exc, self.base_url),
                    }
                },
            )
        return reachable

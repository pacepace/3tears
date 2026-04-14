"""discovery handler for agent tool manifest resolution.

subscribes to NATS discovery subject, resolves pinned tool
manifests against catalog, and returns full schemas for
available tools. each tool appears once regardless of how
many pod endpoints serve it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from threetears.observe import get_logger
from threetears.registry.catalog import ToolCatalog

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Wire-format Pydantic models
# ---------------------------------------------------------------------------


class DiscoverToolEntry(BaseModel):
    """single pinned tool in discovery request manifest.

    :param name: namespaced tool name
    :ptype name: str
    :param version: semver-compatible version string
    :ptype version: str
    """

    name: str
    version: str


class DiscoverRequest(BaseModel):
    """discovery request from agent.

    :param agent_id: unique identifier of requesting agent
    :ptype agent_id: str
    :param tool_manifest: list of pinned tools to resolve
    :ptype tool_manifest: list[DiscoverToolEntry]
    """

    agent_id: str
    tool_manifest: list[DiscoverToolEntry]


class DiscoverResultEntry(BaseModel):
    """single tool result in discovery response.

    :param name: namespaced tool name
    :ptype name: str
    :param version: semver-compatible version string
    :ptype version: str
    :param status: availability status ('available' or 'unavailable')
    :ptype status: str
    :param description: human-readable tool description (empty if unavailable)
    :ptype description: str
    :param input_schema: JSON Schema for tool input (empty dict if unavailable)
    :ptype input_schema: dict[str, Any]
    :param output_schema: optional JSON Schema for tool output
    :ptype output_schema: dict[str, Any] | None
    :param timeout_seconds: per-tool timeout declaration from tool registration, None uses platform default
    :ptype timeout_seconds: float | None
    :param endpoint_count: number of pod endpoints serving this tool
    :ptype endpoint_count: int
    """

    name: str
    version: str
    status: str
    description: str = ""
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] | None = None
    timeout_seconds: float | None = None
    endpoint_count: int = 0


class DiscoverResponse(BaseModel):
    """discovery response sent back to requesting agent.

    :param agent_id: identifier of agent that requested discovery
    :ptype agent_id: str
    :param tools: list of resolved tool results
    :ptype tools: list[DiscoverResultEntry]
    """

    agent_id: str
    tools: list[DiscoverResultEntry]


# ---------------------------------------------------------------------------
# DiscoveryHandler
# ---------------------------------------------------------------------------


class DiscoveryHandler:
    """handles tool discovery requests from agents.

    subscribes to discovery subject with queue group for HA,
    resolves pinned tool manifests against catalog, and replies
    with full schemas for available tools or unavailable status
    for missing ones. each tool appears once regardless of how
    many pod endpoints serve it.
    """

    def __init__(
        self,
        catalog: ToolCatalog,
        namespace: str = "aibots",
    ) -> None:
        """initialize discovery handler.

        :param catalog: tool catalog to resolve tools against
        :ptype catalog: ToolCatalog
        :param namespace: NATS subject namespace prefix
        :ptype namespace: str
        """
        self._catalog = catalog
        self._namespace = namespace
        self._nc: Any | None = None
        self._sub: Any | None = None

    async def start(self, nc: Any) -> None:
        """start listening for discovery requests.

        :param nc: connected NATS client
        :ptype nc: Any
        """
        self._nc = nc
        subject = f"{self._namespace}.tools.discover"
        self._sub = await nc.subscribe(
            subject,
            queue="registry",
            cb=self._handle_discover,
        )
        _logger.info(
            "discovery handler started",
            extra={"extra_data": {"subject": subject}},
        )

    async def stop(self) -> None:
        """stop listening for discovery requests."""
        if self._sub is not None:
            await self._sub.unsubscribe()
            self._sub = None
        _logger.info("discovery handler stopped")

    async def _handle_discover(self, msg: Any) -> None:
        """handle incoming discovery request.

        parses request, resolves each pinned tool against catalog,
        and replies with full schemas for available tools or
        unavailable status for missing ones.

        :param msg: incoming NATS message containing discovery request
        :ptype msg: Any
        :raises RuntimeError: when invoked before ``start`` connects NATS
        """
        if self._nc is None:
            raise RuntimeError("_handle_discover invoked before NATS connected")
        try:
            request = DiscoverRequest.model_validate_json(msg.data)
        except Exception as exc:
            response = DiscoverResponse(agent_id="unknown", tools=[])
            _logger.warning(
                "malformed discovery request",
                extra={"extra_data": {"error": str(exc)}},
            )
            if msg.reply:
                await self._nc.publish(
                    msg.reply,
                    response.model_dump_json().encode("utf-8"),
                )
            return

        if request.tool_manifest:
            tools = self._resolve_manifest(request.tool_manifest)
        else:
            tools = self._list_all_available()

        response = DiscoverResponse(
            agent_id=request.agent_id,
            tools=tools,
        )
        if msg.reply:
            await self._nc.publish(
                msg.reply,
                response.model_dump_json().encode("utf-8"),
            )
        _logger.info(
            "discovery completed",
            extra={"extra_data": {
                "agent_id": request.agent_id,
                "requested_count": len(request.tool_manifest),
                "available_count": sum(
                    1 for t in tools if t.status == "available"
                ),
            }},
        )

    def _list_all_available(self) -> list[DiscoverResultEntry]:
        """return all available tools from catalog.

        used when agent sends empty manifest (discover all).
        each tool appears once with endpoint_count for observability.

        :return: list of all available tool results with schemas
        :rtype: list[DiscoverResultEntry]
        """
        results: list[DiscoverResultEntry] = []
        for entry in self._catalog.list_available():
            results.append(DiscoverResultEntry(
                name=entry.tool_name,
                version=entry.tool_version,
                status="available",
                description=entry.description,
                input_schema=entry.input_schema,
                output_schema=entry.output_schema,
                timeout_seconds=entry.timeout_seconds,
                endpoint_count=len(entry.endpoints),
            ))
        return results

    def _resolve_manifest(
        self,
        manifest: list[DiscoverToolEntry],
    ) -> list[DiscoverResultEntry]:
        """resolve pinned tool manifest against catalog.

        :param manifest: list of pinned tools to resolve
        :ptype manifest: list[DiscoverToolEntry]
        :return: list of resolved tool results with schemas or unavailable status
        :rtype: list[DiscoverResultEntry]
        """
        results: list[DiscoverResultEntry] = []
        for tool_ref in manifest:
            full_name = f"{tool_ref.name}@{tool_ref.version}"
            entry = self._catalog.get(full_name)
            if entry is not None and entry.status == "available":
                result_entry = DiscoverResultEntry(
                    name=entry.tool_name,
                    version=entry.tool_version,
                    status="available",
                    description=entry.description,
                    input_schema=entry.input_schema,
                    output_schema=entry.output_schema,
                    timeout_seconds=entry.timeout_seconds,
                    endpoint_count=len(entry.endpoints),
                )
            else:
                result_entry = DiscoverResultEntry(
                    name=tool_ref.name,
                    version=tool_ref.version,
                    status="unavailable",
                )
            results.append(result_entry)
        return results

"""discovery handler for agent tool manifest resolution.

subscribes to NATS discovery subject, resolves pinned tool
manifests against catalog, and returns full schemas for
available tools. each tool appears once regardless of how
many pod endpoints serve it.

a tool is offered as available only when the requester could
actually be routed to one of its endpoints: another agent's
in-process endpoint does not count. the catalog answers that per
caller (:meth:`ToolCatalog.list_available`,
:meth:`CatalogEntry.available_to`); discovery only asks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel

from threetears.nats import IncomingMessage, Subjects
from threetears.observe import get_logger
from threetears.registry.catalog import CatalogEntry, ToolCatalog

if TYPE_CHECKING:
    from threetears.nats import NatsClient, Subscription

__all__ = [
    "DiscoverRequest",
    "DiscoverResponse",
    "DiscoverResultEntry",
    "DiscoverToolEntry",
    "DiscoveryHandler",
]

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

    :param agent_id: the requesting agent's id, or -- when a ToolServer polls for its own
        readiness -- the polling server's pod-id; an agent's in-process server polls under its
        ``{agent_id}.{instance}`` composite, which names that agent. self-asserted: it decides what
        the requester is SHOWN, never what it may call
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
    :param requires_confirmation: whether calls to the tool must be gated behind human-in-the-loop approval
    :ptype requires_confirmation: bool
    :param endpoint_count: number of pod endpoints serving this tool that the requester could be
        routed to; another agent's in-process endpoints are not counted
    :ptype endpoint_count: int
    """

    name: str
    version: str
    status: str
    description: str = ""
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] | None = None
    timeout_seconds: float | None = None
    requires_confirmation: bool = False
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
        namespace: str = "3tears",
    ) -> None:
        """initialize discovery handler.

        :param catalog: tool catalog to resolve tools against
        :ptype catalog: ToolCatalog
        :param namespace: NATS subject namespace prefix
        :ptype namespace: str
        """
        self._catalog = catalog
        self._namespace = namespace
        self._nc: "NatsClient | None" = None
        self._sub: "Subscription | None" = None

    async def start(self, nc: "NatsClient") -> None:
        """start listening for discovery requests.

        DQ-B7 queue-group note: discovery uses ``queue="registry"`` so
        a horizontally-scaled registry replica set load-balances
        discover requests across replicas; every replica's catalog is
        kept in sync via the heartbeat / KV pipeline so any replica
        can answer.

        :param nc: connected canonical NATS wrapper client
        :ptype nc: NatsClient
        :return: nothing
        :rtype: None
        """
        self._nc = nc
        subject = Subjects.tools_discover()
        self._sub = await nc.subscribe(
            subject=subject,
            queue="registry",
            cb=self.handle_discover,
        )
        _logger.info(
            "discovery handler started",
            extra={"extra_data": {"subject": subject.path}},
        )

    async def stop(self) -> None:
        """stop listening for discovery requests."""
        if self._sub is not None and self._nc is not None:
            await self._nc.unsubscribe(self._sub)
            self._sub = None
        _logger.info("discovery handler stopped")

    async def handle_discover(self, msg: IncomingMessage) -> None:
        """public NATS-subject handler for tool-manifest discovery.

        bound by :meth:`start` as the ``cb`` callback on
        ``{namespace}.tools.discover``. tests exercise this surface
        directly; the name + single-``msg`` shape are part of the
        stability contract.

        parses request, resolves each pinned tool against catalog,
        and replies (via :meth:`NatsClient.publish_reply`) with full
        schemas for available tools or unavailable status for missing
        ones.

        :param msg: incoming wrapper envelope containing discovery request
        :ptype msg: IncomingMessage
        :raises RuntimeError: when invoked before ``start`` connects NATS
        """
        if self._nc is None:
            raise RuntimeError("handle_discover invoked before NATS connected")
        try:
            request = DiscoverRequest.model_validate_json(msg.data)
        except Exception as exc:
            response = DiscoverResponse(agent_id="unknown", tools=[])
            _logger.warning(
                "malformed discovery request",
                extra={"extra_data": {"error": str(exc)}},
            )
            if msg.reply_subject is not None:
                await self._nc.publish_reply(
                    reply_subject=msg.reply_subject,
                    message=response,
                )
            return

        requester_id = _requester_agent_id(request.agent_id)
        if request.tool_manifest:
            tools = self._resolve_manifest(request.tool_manifest, requester_id)
        else:
            tools = self._list_all_available(requester_id)

        response = DiscoverResponse(
            agent_id=request.agent_id,
            tools=tools,
        )
        if msg.reply_subject is not None:
            await self._nc.publish_reply(
                reply_subject=msg.reply_subject,
                message=response,
            )
        requester_log = str(requester_id) if requester_id is not None else None  # convert at border: log record
        _logger.info(
            "discovery completed",
            extra={
                "extra_data": {
                    "agent_id": request.agent_id,
                    # the agent the claim resolved to, or null when it named none -- then the
                    # requester was shown Tool Pod tools only, and its own in-process tools are
                    # missing from its view. without this a mis-spelled agent id reads the same as
                    # a resolved one while that agent's readiness wait reports its own tools gone.
                    "requester_agent_id": requester_log,
                    "requested_count": len(request.tool_manifest),
                    "available_count": sum(1 for t in tools if t.status == "available"),
                }
            },
        )

    def _list_all_available(self, requester_id: UUID | None) -> list[DiscoverResultEntry]:
        """return every tool the requester could be routed to right now.

        used when agent sends empty manifest (discover all).
        each tool appears once with endpoint_count for observability.

        :param requester_id: the agent the requester names, or ``None`` when it names none
        :ptype requester_id: UUID | None
        :return: list of all available tool results with schemas
        :rtype: list[DiscoverResultEntry]
        """
        results = [_available_result(entry, requester_id) for entry in self._catalog.list_available(requester_id)]
        return results

    def _resolve_manifest(
        self,
        manifest: list[DiscoverToolEntry],
        requester_id: UUID | None,
    ) -> list[DiscoverResultEntry]:
        """resolve pinned tool manifest against catalog.

        :param manifest: list of pinned tools to resolve
        :ptype manifest: list[DiscoverToolEntry]
        :param requester_id: the agent the requester names, or ``None`` when it names none
        :ptype requester_id: UUID | None
        :return: list of resolved tool results with schemas or unavailable status
        :rtype: list[DiscoverResultEntry]
        """
        results: list[DiscoverResultEntry] = []
        for tool_ref in manifest:
            full_name = f"{tool_ref.name}@{tool_ref.version}"
            entry = self._catalog.get(full_name)
            if entry is not None and entry.available_to(requester_id):
                result_entry = _available_result(entry, requester_id)
            else:
                result_entry = DiscoverResultEntry(
                    name=tool_ref.name,
                    version=tool_ref.version,
                    status="unavailable",
                )
            results.append(result_entry)
        return results


def _requester_agent_id(claimed: str) -> UUID | None:
    """the agent a discovery requester names, or ``None`` when it names none.

    An agent sends its own id. A ToolServer polling for its own readiness sends its pod-id instead:
    a Tool Pod's is one token and names no agent it could own an in-process endpoint for, and an
    agent's in-process server's is the ``{agent_id}.{instance}`` composite, which names that agent.
    Anything else -- a display name, ``"unknown"`` -- names no agent and is shown the Tool Pod
    tools alone.

    The claim is self-asserted and that is acceptable for exactly this use: it narrows what the
    requester is shown, and the call path enforces ownership on the VERIFIED identity regardless,
    so a requester that misnames itself misleads only itself.

    :param claimed: the request's ``agent_id`` field
    :ptype claimed: str
    :return: the named agent's id, or ``None``
    :rtype: UUID | None
    """
    result: UUID | None = None
    try:
        result = Subjects.agent_inprocess_owner_id(claimed)
        if result is None:
            parsed = UUID(claimed)
            result = parsed if str(parsed) == claimed else None
    except ValueError:
        # NOSILENT: "this requester names no agent" is the answer; it is shown the Tool Pod
        # tools, which every caller may use. logging each such request would log every readiness
        # poll a non-agent pod makes.
        result = None
    return result


def _available_result(entry: CatalogEntry, requester_id: UUID | None) -> DiscoverResultEntry:
    """the discovery result for a tool the requester can reach.

    :param entry: the tool's catalog entry, already known to be available to the requester
    :ptype entry: CatalogEntry
    :param requester_id: the agent the requester names, or ``None`` when it names none
    :ptype requester_id: UUID | None
    :return: the available result, counting only the requester's endpoints
    :rtype: DiscoverResultEntry
    """
    return DiscoverResultEntry(
        name=entry.tool_name,
        version=entry.tool_version,
        status="available",
        description=entry.description,
        input_schema=entry.input_schema,
        output_schema=entry.output_schema,
        timeout_seconds=entry.timeout_seconds,
        requires_confirmation=entry.requires_confirmation,
        endpoint_count=len(entry.endpoints_for(requester_id)),
    )

"""workspace discovery NATS request/reply client.

thin helper around the ``{ns}.workspace.discover`` subject the broker
subscribes to (see ``aibots.hub.broker.workspace_discovery``). used by
:class:`~threetears.agent.workspace.tools.workspace_list.WorkspaceListTool`
and :class:`~threetears.agent.workspace.tools.workspace_current.WorkspaceCurrentTool`
to retrieve the set of workspace-type namespaces a caller can see --
owned plus granted within the caller's customer -- without a local
SELECT against the agent's own ``workspaces`` table.

the client serializes a :class:`~aibots.hub.broker.workspace_discovery.WorkspaceDiscoverRequest`,
publishes to ``{namespace}.workspace.discover``, and parses the reply
back into a :class:`WorkspaceDiscoverResponse` (success) or
:class:`WorkspaceDiscoverError` (broker-reported failure). the tool
layer treats errors as errors-as-data and surfaces them to the LLM.

the reply models are imported lazily from the aibots hub package
because ``agent-workspace`` must not pull in the hub at package-load
time -- the hub depends on the agent packages, not the other way
around. the tool wires the client with the already-connected NATS
handle and the broker subject namespace so no package-level import of
``aibots`` is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field
from threetears.observe import get_logger, traced

__all__ = [
    "DiscoveryClientError",
    "WorkspaceDiscoveryClient",
    "WorkspaceDiscoveryRequest",
    "WorkspaceDiscoveryResponse",
    "WorkspaceDiscoverySummary",
]

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


class WorkspaceDiscoveryRequest(BaseModel):
    """local wire request mirroring the broker handler's shape.

    agent-workspace carries its own copy of the request/response models
    instead of importing from aibots.hub so the package boundary stays
    clean -- the broker owns its handler models and this module owns the
    client models; both sides agree on the JSON shape.

    :param correlation_id: request trace identifier bound into broker
        logs so a discovery call can be correlated back to the tool
        invocation that issued it
    :ptype correlation_id: UUID
    :param agent_id: calling agent UUID; discovery returns every
        workspace-type namespace owned by this agent plus every one
        the agent holds a grant on within the caller's customer
    :ptype agent_id: UUID
    :param customer_id: calling customer UUID; discovery filters rows
        to this customer in SQL so cross-customer rows never land in
        the response
    :ptype customer_id: UUID
    :param user_id: invoking user UUID when the call is on behalf of
        a specific user; ``None`` requests the admin/internal "every
        grant-visible row" shape (handler variant)
    :ptype user_id: UUID | None
    """

    correlation_id: UUID
    agent_id: UUID
    customer_id: UUID
    user_id: UUID | None = None


class WorkspaceDiscoverySummary(BaseModel):
    """single namespace row returned from the discovery subject.

    mirrors the broker handler's ``WorkspaceSummary`` column set.

    :param id: shared primary key of the namespace + workspace rows
    :ptype id: UUID
    :param name: globally-unique namespace name (``workspace.<uuid>``)
    :ptype name: str
    :param owner_agent_id: agent whose schema physically holds the
        workspace rows; cross-agent routing targets this agent
    :ptype owner_agent_id: UUID
    :param customer_id: owning customer; always matches the caller's
        customer because the broker filters in SQL
    :ptype customer_id: UUID
    """

    id: UUID
    name: str
    owner_agent_id: UUID
    customer_id: UUID


class WorkspaceDiscoveryResponse(BaseModel):
    """successful response carrying the visible workspace set.

    :param success: always True on success; present for symmetry with
        the error envelope so callers can branch on the single field
    :ptype success: bool
    :param items: workspace summaries ordered by broker ``date_updated``
        descending so list UIs surface recent activity first
    :ptype items: list[WorkspaceDiscoverySummary]
    """

    success: bool = True
    items: list[WorkspaceDiscoverySummary] = Field(default_factory=list)


class DiscoveryClientError(RuntimeError):
    """raised when the broker returns an error envelope or the call fails.

    the client translates the broker's ``WorkspaceDiscoverError`` and
    any transport-level failure (timeout, NATS not wired) into a single
    exception type so tool callers can ``except DiscoveryClientError``
    once. the underlying message preserves the broker's
    ``error_code`` / ``error_message`` when present.
    """


class WorkspaceDiscoveryClient:
    """NATS request/reply client for the ``{ns}.workspace.discover`` subject.

    constructed once per tool with the already-connected NATS handle
    and the broker subject namespace; each call serializes a fresh
    request and awaits the reply with a bounded timeout. parsing
    failures surface as :class:`DiscoveryClientError` so the tool layer
    always sees either a valid summary list or a typed error.

    :param nats_client: connected NATS client exposing :meth:`request`
    :ptype nats_client: Any
    :param namespace: broker subject namespace prefix (from
        ``FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE``)
    :ptype namespace: str
    :param timeout_seconds: per-call request timeout in seconds;
        defaults to 5.0 to match other broker request/reply helpers
    :ptype timeout_seconds: float
    """

    def __init__(
        self,
        nats_client: Any,
        namespace: str,
        timeout_seconds: float = 5.0,
    ) -> None:
        """bind the client to a NATS handle + broker subject namespace.

        :param nats_client: connected NATS client
        :ptype nats_client: Any
        :param namespace: broker subject namespace prefix
        :ptype namespace: str
        :param timeout_seconds: per-call request timeout in seconds
        :ptype timeout_seconds: float
        """
        self._nats_client = nats_client
        self._namespace = namespace
        self._timeout_seconds = timeout_seconds

    @traced
    async def discover(
        self,
        *,
        correlation_id: UUID,
        agent_id: UUID,
        customer_id: UUID,
        user_id: UUID | None,
    ) -> list[WorkspaceDiscoverySummary]:
        """issue one discovery request and return the caller's visible set.

        serializes a :class:`WorkspaceDiscoveryRequest`, publishes it to
        ``{namespace}.workspace.discover``, waits up to
        ``self._timeout_seconds`` for the broker reply, then parses the
        response. on broker-reported failure the response envelope
        carries ``success=false`` and an error-code/message pair; this
        method translates that to :class:`DiscoveryClientError`.

        :param correlation_id: trace identifier for this discovery call
        :ptype correlation_id: UUID
        :param agent_id: calling agent UUID
        :ptype agent_id: UUID
        :param customer_id: calling customer UUID (must be set: the
            broker filters on this in SQL and discovery is not valid
            without it)
        :ptype customer_id: UUID
        :param user_id: invoking user UUID or ``None`` for admin-shape
        :ptype user_id: UUID | None
        :return: list of workspace summaries, newest-update first
        :rtype: list[WorkspaceDiscoverySummary]
        :raises DiscoveryClientError: on NATS missing, transport failure,
            malformed reply, or broker-reported error envelope
        """
        if self._nats_client is None:
            raise DiscoveryClientError(
                "workspace discovery requires a NATS client; none wired",
            )
        request = WorkspaceDiscoveryRequest(
            correlation_id=correlation_id,
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=user_id,
        )
        subject = f"{self._namespace}.workspace.discover"
        try:
            reply = await self._nats_client.request(
                subject,
                request.model_dump_json().encode(),
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            raise DiscoveryClientError(
                f"workspace.discover request failed: {exc}",
            ) from exc
        body = reply.data
        # success path first; fall through to error parsing on failure
        parse_error: Exception | None = None
        try:
            response = WorkspaceDiscoveryResponse.model_validate_json(body)
        except Exception as exc:
            parse_error = exc
            response = None  # type: ignore[assignment]
        result: list[WorkspaceDiscoverySummary]
        if response is not None and response.success:
            result = response.items
        else:
            # either response is None (parse failed) or success=False;
            # inspect the body for the broker's error envelope fields.
            error_code = "UNKNOWN"
            error_message = (
                f"malformed discovery response: {parse_error}"
                if parse_error is not None else "discovery returned success=false"
            )
            try:
                envelope: dict[str, Any] = WorkspaceDiscoveryResponse.model_validate_json(
                    body,
                ).model_dump()
                error_code = str(envelope.get("error_code", error_code))
                error_message = str(envelope.get("error_message", error_message))
            except Exception:
                # best-effort: leave the defaults in place
                pass
            raise DiscoveryClientError(
                f"workspace.discover failed: {error_code}: {error_message}",
            )
        return result

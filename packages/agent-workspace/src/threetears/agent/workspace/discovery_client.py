"""namespace discovery NATS request/reply client.

thin helper around the ``{ns}.namespace.discover`` subject the broker
subscribes to (see ``aibots.hub.broker.namespace_discovery``). used by
:class:`~threetears.agent.workspace.tools.workspace_list.WorkspaceListTool`
and :class:`~threetears.agent.workspace.tools.workspace_current.WorkspaceCurrentTool`
to retrieve namespace-type rows a caller can see -- owned plus granted
within the caller's customer -- without a local SELECT against the
agent's own tables.

namespace-task-01 Phase 1 generalized the subject from the workspace-
specific ``{ns}.workspace.discover`` to the resource-type-parameterized
``{ns}.namespace.discover``. the request model carries an optional
``namespace_type`` filter; ``None`` returns every row the caller can
see regardless of type. callers that historically asked only for
workspaces now pass ``namespace_type="workspace"`` explicitly; there
is no back-compat alias for the old subject or the old request shape
-- per aibots CLAUDE.md's NO BACKWARDS-COMPATIBILITY SHIMS rule the
rename is a one-commit coordinated change across 3tears + aibots.

the client serializes a :class:`NamespaceDiscoveryRequest`, publishes
to ``{namespace}.namespace.discover``, and parses the reply back into
a :class:`NamespaceDiscoveryResponse` (success) or
:class:`DiscoveryClientError` (transport or broker-reported failure).
the tool layer treats errors as errors-as-data and surfaces them to
the LLM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field
from threetears.observe import get_logger, traced

__all__ = [
    "DiscoveryClientError",
    "NamespaceDiscoveryClient",
    "NamespaceDiscoveryRequest",
    "NamespaceDiscoveryResponse",
    "NamespaceDiscoverySummary",
]

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


#: closed set of namespace_type values callers may filter on. matches
#: the :class:`aibots.hub.broker.namespaces.NamespaceType` enum shipped
#: alongside this module. carried as a ``Literal`` on the request model
#: so an accidental new type fails parse at the producer site rather
#: than silently returning an empty set.
NamespaceTypeFilter = Literal[
    "workspace",
    "agent",
    "shared",
    "system",
    "memory",
    "datasource",
    "tool",
    "channel",
    "shared_agent",
    "model",
]


class NamespaceDiscoveryRequest(BaseModel):
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
        namespace owned by this agent plus every one the agent holds a
        grant on within the caller's customer (filtered further by
        ``namespace_type`` when supplied)
    :ptype agent_id: UUID
    :param customer_id: calling customer UUID; discovery filters rows
        to this customer in SQL so cross-customer rows never land in
        the response
    :ptype customer_id: UUID
    :param user_id: invoking user UUID when the call is on behalf of
        a specific user; ``None`` requests the admin/internal "every
        grant-visible row" shape (handler variant)
    :ptype user_id: UUID | None
    :param namespace_type: optional closed-set filter. when ``None``
        discovery returns every visible namespace regardless of type;
        when set, only namespaces of that type are returned
    :ptype namespace_type: NamespaceTypeFilter | None
    """

    correlation_id: UUID
    agent_id: UUID
    customer_id: UUID
    user_id: UUID | None = None
    namespace_type: NamespaceTypeFilter | None = None


class NamespaceDiscoverySummary(BaseModel):
    """single namespace row returned from the discovery subject.

    mirrors the broker handler's ``NamespaceSummary`` column set.

    :param id: primary key of the namespace row
    :ptype id: UUID
    :param name: globally-unique namespace name
    :ptype name: str
    :param namespace_type: discriminator value (``workspace`` /
        ``memory`` / ``datasource`` / ``tool`` / ``channel`` /
        ``shared_agent`` / ``agent`` / ``shared`` / ``system``). kept
        open-str on the summary so a caller that asked for "all types"
        sees the row's type without re-validating against the closed
        :data:`NamespaceTypeFilter` set (new variants published by a
        rolling broker upgrade must not crash older clients on parse)
    :ptype namespace_type: str
    :param owner_agent_id: agent whose schema physically holds the
        namespace's rows; cross-agent routing targets this agent
    :ptype owner_agent_id: UUID
    :param customer_id: owning customer; always matches the caller's
        customer because the broker filters in SQL
    :ptype customer_id: UUID
    """

    id: UUID
    name: str
    namespace_type: str
    owner_agent_id: UUID
    customer_id: UUID


class NamespaceDiscoveryResponse(BaseModel):
    """successful response carrying the visible namespace set.

    :param success: always True on success; present for symmetry with
        the error envelope so callers can branch on the single field
    :ptype success: bool
    :param items: namespace summaries ordered by broker ``date_updated``
        descending so list UIs surface recent activity first
    :ptype items: list[NamespaceDiscoverySummary]
    """

    success: bool = True
    items: list[NamespaceDiscoverySummary] = Field(default_factory=list)


class DiscoveryClientError(RuntimeError):
    """raised when the broker returns an error envelope or the call fails.

    the client translates the broker's error envelope and any transport-
    level failure (timeout, NATS not wired) into a single exception type
    so tool callers can ``except DiscoveryClientError`` once. the
    underlying message preserves the broker's ``error_code`` /
    ``error_message`` when present.
    """


class NamespaceDiscoveryClient:
    """NATS request/reply client for the ``{ns}.namespace.discover`` subject.

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
        namespace_type: NamespaceTypeFilter | None = None,
    ) -> list[NamespaceDiscoverySummary]:
        """issue one discovery request and return the caller's visible set.

        serializes a :class:`NamespaceDiscoveryRequest`, publishes it to
        ``{namespace}.namespace.discover``, waits up to
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
        :param namespace_type: closed-set filter; ``None`` returns every
            visible namespace regardless of type
        :ptype namespace_type: NamespaceTypeFilter | None
        :return: list of namespace summaries, newest-update first
        :rtype: list[NamespaceDiscoverySummary]
        :raises DiscoveryClientError: on NATS missing, transport failure,
            malformed reply, or broker-reported error envelope
        """
        if self._nats_client is None:
            raise DiscoveryClientError(
                "namespace discovery requires a NATS client; none wired",
            )
        request = NamespaceDiscoveryRequest(
            correlation_id=correlation_id,
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=user_id,
            namespace_type=namespace_type,
        )
        subject = f"{self._namespace}.namespace.discover"
        try:
            reply = await self._nats_client.request(
                subject,
                request.model_dump_json().encode(),
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            raise DiscoveryClientError(
                f"namespace.discover request failed: {exc}",
            ) from exc
        body = reply.data
        # success path first; fall through to error parsing on failure
        parse_error: Exception | None = None
        response: NamespaceDiscoveryResponse | None
        try:
            response = NamespaceDiscoveryResponse.model_validate_json(body)
        except Exception as exc:
            parse_error = exc
            response = None
        result: list[NamespaceDiscoverySummary]
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
                envelope: dict[str, Any] = NamespaceDiscoveryResponse.model_validate_json(
                    body,
                ).model_dump()
                error_code = str(envelope.get("error_code", error_code))
                error_message = str(envelope.get("error_message", error_message))
            except Exception:
                # best-effort: leave the defaults in place
                pass
            raise DiscoveryClientError(
                f"namespace.discover failed: {error_code}: {error_message}",
            )
        return result

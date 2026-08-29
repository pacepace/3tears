"""namespace discovery NATS request/reply client.

thin helper around the ``{ns}.namespace.discover`` subject the hub's
broker subscribes to. used by the workspace tools
(:class:`~threetears.agent.workspace.tools.workspace_list.WorkspaceListTool`,
:class:`~threetears.agent.workspace.tools.workspace_current.WorkspaceCurrentTool`)
and by agent bootstrap's ``access.*`` translators to retrieve the
namespace rows a caller can see -- owned plus granted within the
caller's customer -- without a local SELECT against the agent's own
tables.

**identity is a FORWARDED TOKEN, never a request field.** the request
carries the caller's hub-minted ``identity_token`` and, when a human
is in the loop for this turn, the hub-minted ``user_identity_token``;
the broker verifies both in-process and reads the principal off the
signed claims. there is no ``agent_id`` / ``customer_id`` / ``user_id``
on this request, so a caller cannot name a principal it did not
authenticate as. both slots are optional and the broker refuses only
when BOTH are empty -- an agent-initiated call legitimately forwards
the agent leg alone.

this module sits beside its two siblings
(:mod:`threetears.agent.tools.object_resolver`,
:mod:`threetears.agent.tools.engagement_resolver`) rather than in
``threetears.agent.workspace``: all three are pod-side hub callers
authenticating with the per-call token off the call context, and a
tool pod holds ``3tears-agent-tools`` without holding the workspace
package.

**what discovery ANSWERS.** a namespace comes back when some role
assignment's SCOPE covers it. it is never a statement that the caller
may perform any particular action on it -- see
:class:`NamespaceDiscoveryRequest` for why the two can differ.

the client serializes a :class:`NamespaceDiscoveryRequest`, publishes
to ``{namespace}.namespace.discover``, and parses the reply back into
a :class:`NamespaceDiscoveryResponse` (success) or
:class:`DiscoveryClientError` (transport or broker-reported failure).
the tool layer treats errors as errors-as-data and surfaces them to
the LLM.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field
from threetears.observe import get_logger, traced

__all__ = [
    "DiscoveryClientError",
    "NamespaceDiscoveryClient",
    "NamespaceDiscoveryErrorEnvelope",
    "NamespaceDiscoveryRequest",
    "NamespaceDiscoveryResponse",
    "NamespaceDiscoverySummary",
    "NamespaceTypeFilter",
]

log = get_logger(__name__)


#: closed set of namespace_type values callers may filter on. matches
#: the ``NamespaceType`` enum the hub ships alongside this module.
#: carried as a ``Literal`` on the request model so an accidental new
#: type fails parse at the producer site rather than silently
#: returning an empty set.
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

    agent-tools carries its own copy of the request/response models
    instead of importing from the hub so the package boundary stays
    clean -- the broker owns its handler models and this module owns
    the client models; both sides agree on the JSON shape.

    **discovery answers a SCOPE question, never a permission one.** a
    namespace is returned when a role assignment's scope covers it --
    ``scope_type='namespace'`` naming that row, ``'type_customer'``
    naming its type within the caller's customer, or ``'all'``. WHICH
    role carried that assignment is not consulted, and nothing
    constrains a namespace to one role: the only unique index on
    ``role_assignments`` is on its bare ``id``, so two roles may be
    granted on one namespace and they need not agree. a caller can
    therefore see a namespace on the strength of one role's
    assignment and still be refused the action it wanted, which the
    per-call evaluator decides separately. treat a returned row as
    "in scope", never as "permitted".

    :param correlation_id: request trace identifier bound into broker
        logs so a discovery call can be correlated back to the tool
        invocation that issued it
    :ptype correlation_id: UUID
    :param identity_token: the caller's hub-minted identity token,
        forwarded verbatim. the broker verifies it and takes the
        calling agent and customer off the signed claims, so this
        replaces what used to be a self-asserted ``agent_id`` /
        ``customer_id`` pair. ``None`` only when a user assertion is
        forwarded instead
    :ptype identity_token: str | None
    :param user_identity_token: the per-turn hub-minted user
        assertion, when a human is in the loop. the broker verifies it,
        binds it to the identity token's principal, and takes the
        acting user off the signed claims. ``None`` for an
        agent-initiated call with nobody in the loop, in which case the
        user leg of the visibility query is skipped
    :ptype user_identity_token: str | None
    :param namespace_type: optional closed-set filter. when ``None``
        discovery returns every visible namespace regardless of type;
        when set, only namespaces of that type are returned
    :ptype namespace_type: NamespaceTypeFilter | None
    """

    correlation_id: UUID
    identity_token: str | None = None
    user_identity_token: str | None = None
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

    **every item is IN SCOPE of some grant, not necessarily
    PERMITTED.** the query matches a role assignment's scope and never
    consults which role carried it, and one namespace may carry
    assignments from two roles that do not agree -- so a row here can
    belong to a caller who is still refused the action it wanted. ask
    the evaluator before acting on one.

    :param success: always True on success; present for symmetry with
        the error envelope so callers can branch on the single field
    :ptype success: bool
    :param items: namespace summaries ordered by broker ``date_updated``
        descending so list UIs surface recent activity first
    :ptype items: list[NamespaceDiscoverySummary]
    """

    success: bool = True
    items: list[NamespaceDiscoverySummary] = Field(default_factory=list)


class NamespaceDiscoveryErrorEnvelope(BaseModel):
    """the broker's refusal envelope, mirroring its handler's error model.

    parsed as its OWN model rather than re-parsed as
    :class:`NamespaceDiscoveryResponse`, which declares neither field: doing
    the latter flattened every broker refusal to ``UNKNOWN`` and made the
    codes -- "no credential was forwarded" versus "the credential did not
    verify" -- unreachable by any caller.

    :param success: always False on a refusal
    :ptype success: bool
    :param error_code: the machine-readable code a caller branches on
    :ptype error_code: str
    :param error_message: human-readable description; never parsed
    :ptype error_message: str
    """

    success: bool = False
    error_code: str
    error_message: str


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
        ``THREETEARS_NATS_SUBJECT_NAMESPACE``)
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
        identity_token: str | None = None,
        user_identity_token: str | None = None,
        namespace_type: NamespaceTypeFilter | None = None,
    ) -> list[NamespaceDiscoverySummary]:
        """issue one discovery request and return the caller's visible set.

        serializes a :class:`NamespaceDiscoveryRequest`, publishes it to
        ``{namespace}.namespace.discover``, waits up to
        ``self._timeout_seconds`` for the broker reply, then parses the
        response. on broker-reported failure the response envelope
        carries ``success=false`` and an error-code/message pair; this
        method translates that to :class:`DiscoveryClientError`.

        the broker refuses a request carrying neither token, so a
        caller holding no credential learns that here rather than
        receiving somebody else's answer.

        :param correlation_id: trace identifier for this discovery call
        :ptype correlation_id: UUID
        :param identity_token: the caller's hub-minted identity token,
            forwarded verbatim for the broker to verify
        :ptype identity_token: str | None
        :param user_identity_token: the per-turn hub-minted user
            assertion when a human is in the loop; ``None`` for an
            agent-initiated call
        :ptype user_identity_token: str | None
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
            identity_token=identity_token,
            user_identity_token=user_identity_token,
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
                if parse_error is not None
                else "discovery returned success=false"
            )
            try:
                envelope = NamespaceDiscoveryErrorEnvelope.model_validate_json(body)
                error_code = envelope.error_code
                error_message = envelope.error_message
            except Exception as exc:  # noqa: BLE001 -- the DiscoveryClientError below is the report
                # Only enriching the message; the failure is raised either way. Logged so a
                # generic "discovery returned success=false" is traceable to an unparseable body
                # rather than read as all the detail the server sent.
                log.debug(
                    "could not parse the discovery error envelope; using default error text",
                    extra={"extra_data": {"error": str(exc)}},
                )
            raise DiscoveryClientError(
                f"namespace.discover failed: {error_code}: {error_message}",
            )
        return result

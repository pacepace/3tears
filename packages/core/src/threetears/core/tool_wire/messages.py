"""the tool server's wire messages: registration, call, reply, heartbeat, probe.

the envelopes a tool pod and the registry exchange over NATS. both ends parse
them, so they live in core, which both depend on, rather than beside either
end's implementation: ``threetears.agent.tools.server`` (the pod) and
``threetears.registry`` (the registry) each import them from here, and neither
has to depend on the other to read what the other sends.

pure value types. no IO, no lifecycle, no pod policy: which refusals are final,
how a pod waits for readiness and when it answers a probe stay with the pod.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, SerializerFunctionWrapHandler, model_serializer, model_validator

from threetears.core.tool_wire.context import CallContext
from threetears.core.tool_wire.rest import RestAffordance
from threetears.core.wire import TolerantWireModel

__all__ = [
    "CallAccepted",
    "CallRequest",
    "CallResponse",
    "DiscoveryProbeRequest",
    "DiscoveryProbeToolEntry",
    "HeartbeatMessage",
    "ProbeAck",
    "RefusedTool",
    "RegistrationManifest",
    "RegistrationResponse",
    "ToolManifestEntry",
]


class ToolManifestEntry(BaseModel):
    """single tool entry in registration manifest.

    The two visibility flags ride with each tool's manifest entry so
    the hub-side ``ToolNamespaceEmitter`` can stamp them onto the
    ``namespaces`` row it writes. Defaults match
    :class:`~threetears.agent.tools.base_tool.TearsTool` so manifests
    composed by older callers that don't set the fields still land
    with the canonical "tool-eligible, not skill-eligible" shape.

    :param name: namespaced tool name
    :ptype name: str
    :param version: semver-compatible version string
    :ptype version: str
    :param description: human-readable tool description
    :ptype description: str
    :param input_schema: JSON Schema for tool input parameters
    :ptype input_schema: dict[str, Any]
    :param timeout_seconds: expected maximum execution time, None uses caller default
    :ptype timeout_seconds: float | None
    :param tool_eligible: whether the tool belongs in the agent's
        default tool surface; mirrors
        :attr:`TearsTool.tool_eligible`. Defaults to ``True`` so
        manifests built without an explicit value preserve the
        pre-shard behaviour.
    :ptype tool_eligible: bool
    :param skill_eligible: whether the tool is discoverable via the
        skills catalog; mirrors :attr:`TearsTool.skill_eligible`.
        Defaults to ``False``.
    :ptype skill_eligible: bool
    :param face_platform_tool: whether the tool is reachable over the
        internal NATS mesh as a native platform tool; mirrors
        :attr:`TearsTool.face_platform_tool`. Defaults to ``True`` so
        manifests built without an explicit value preserve the tool's
        historical reach.
    :ptype face_platform_tool: bool
    :param face_api: whether the tool is reachable as an external HTTP
        API operation; mirrors :attr:`TearsTool.face_api`. Defaults to
        ``False``.
    :ptype face_api: bool
    :param face_mcp: whether the tool is reachable as an external MCP
        tool; mirrors :attr:`TearsTool.face_mcp`. Defaults to
        ``False``.
    :ptype face_mcp: bool
    :param face_rest: the tool's authored REST address, or ``None``
        when it declares no REST face; mirrors
        :attr:`TearsTool.face_rest`. Deliberately NOT accompanied by a
        boolean twin: the declaration's presence IS the flag, and a
        separate ``face_rest: bool`` beside it would be a second place
        to say the same thing. The consumer that persists a boolean
        column derives it from ``face_rest is not None``.
    :ptype face_rest: RestAffordance | None
    :param requires_confirmation: whether a call to the tool must be
        gated behind human-in-the-loop approval; mirrors
        :attr:`TearsTool.requires_confirmation`. Defaults to ``False``
        so manifests built without an explicit value stay ungated.
    :ptype requires_confirmation: bool
    """

    name: str
    version: str
    description: str
    input_schema: dict[str, Any]
    timeout_seconds: float | None = None
    tool_eligible: bool = True
    skill_eligible: bool = False
    face_platform_tool: bool = True
    face_api: bool = False
    face_mcp: bool = False
    face_rest: RestAffordance | None = None
    requires_confirmation: bool = False


class RegistrationManifest(BaseModel):
    """manifest sent on connect to register all tools.

    ownership identity (``owner_agent_id`` / ``customer_id``) rides on
    the manifest so downstream namespace materialization can stamp
    the right scope on each ``namespaces`` row without
    re-resolving the pod's identity from a separate auth lookup.
    agent-spun pods set both; platform-built-in pods (admin tools,
    datasource tool pod) leave both ``None`` and the namespace rows
    land with NULL owner columns.

    :param pod_id: unique identifier for this tool pod
    :ptype pod_id: str
    :param tools: list of tool definitions served by this pod
    :ptype tools: list[ToolManifestEntry]
    :param bootstrap_token: the tool pod's registry-verification credential. under per-key identity
        this is the pod's SELF-MINTED identity JWT (a short-lived EdDSA token signed with the pod's
        own Ed25519 key, ``kid`` = pod id), re-minted fresh for each manifest by the ToolServer's
        ``auth_token`` provider; the registry-layer :class:`~threetears.registry.auth.ToolPodAuthenticator`
        verifies it (raw, not a hash) against the pod's stored public key. named ``bootstrap_token``
        for wire-compatibility with the NATS auth-callout's connect-token field, which it mirrors.
        ``None`` in open mode (no registry authenticator wired) or a static dev token.
    :ptype bootstrap_token: str | None
    :param owner_agent_id: owning-agent UUID for agent-spun pods;
        ``None`` for platform-built-in pods. it scopes the ``namespaces``
        rows and is a CLAIM: it is not who serves the endpoint. that is read
        from ``pod_id`` (:meth:`~threetears.nats.Subjects.agent_inprocess_owner_id`),
        which the registry routes on and a claim here cannot override
    :ptype owner_agent_id: UUID | None
    :param customer_id: owning-customer UUID for agent-spun pods;
        ``None`` for platform-built-in pods
    :ptype customer_id: UUID | None
    :param proves_delegation: whether this pod proves a call's delegated token with its
        registration key (``ToolServer(registration_key=)`` gives one): the registry then sends it the
        delegation in place of the agent's own token, which it never holds. A registry that predates
        the field ignores it and keeps sending both
    :ptype proves_delegation: bool
    """

    pod_id: str
    tools: list[ToolManifestEntry]
    bootstrap_token: str | None = None
    owner_agent_id: UUID | None = None
    customer_id: UUID | None = None
    proves_delegation: bool = False


class RefusedTool(BaseModel):
    """one tool the registry refused to admit from a manifest, and why.

    :param name: the tool's mcp name as the manifest offered it
    :ptype name: str
    :param version: the tool's version as the manifest offered it
    :ptype version: str
    :param code: the machine-readable refusal code a pod branches on --
        ``OWNED_ELSEWHERE``, ``NOT_PLATFORM_SHARED``, ``UNVERIFIED_PUBLISHER``,
        ``POD_ID_MISMATCH``, ``INVALID_TOOL_NAME`` or ``OWNERSHIP_GRAPH_UNAVAILABLE``
        (see ``threetears.registry.ownership.RefusalCode``)
    :ptype code: str
    :param reason: prose for the operator reading the pod's log; never parsed
    :ptype reason: str
    """

    name: str
    version: str
    code: str
    reason: str


class RegistrationResponse(BaseModel):
    """the registry's reply to one registration manifest.

    **Defined here, beside** :class:`RegistrationManifest`, **rather than in the registry
    that builds it.** The two are one protocol, and the pod is the process that has to
    parse this. Core is the package both ends depend on and neither depends on the other
    through, so the registry builds it and the pod parses it from the same class.
    ``threetears.registry.registration`` re-exports it, so
    ``from threetears.registry import RegistrationResponse`` is unchanged.

    :param success: whether registration succeeded -- at least one tool was admitted
    :ptype success: bool
    :param pod_id: identifier of pod that attempted registration
    :ptype pod_id: str
    :param registered_tools: list of full_name values successfully registered
    :ptype registered_tools: list[str]
    :param owned_namespaces: the canonical ``namespaces.name`` of every namespace this pod
        OWNS -- ``tools.pentest`` for each tool-name NODE on its ``tool_pods`` row, or
        ``agents.<uuid>`` for an agent-owned in-process pod. **Self-identity, and the pod
        cannot work it out for itself**: its subject grants are minted at CONNECT from that
        row, which it never sees, and the only values it does hold are tool LEAVES, which
        cannot be reduced back to their node by any rule. DERIVED by the registry from the
        verified auth context, never read off the manifest, so a pod cannot claim a
        namespace it does not own. Empty when there is nothing to name.
    :ptype owned_namespaces: list[str]
    :param refused_tools: every tool the registry refused, with its code and reason. Filled
        whether or not the registration as a whole succeeded: a pod that registered nine tools
        and had the tenth refused is told about the tenth
    :ptype refused_tools: list[RefusedTool]
    :param error: error message if registration failed
    :ptype error: str | None
    :param error_code: machine-readable code for a failed registration, ``None`` on success
    :ptype error_code: str | None
    """

    success: bool
    pod_id: str
    registered_tools: list[str] = []
    owned_namespaces: list[str] = []
    refused_tools: list[RefusedTool] = []
    error: str | None = None
    error_code: str | None = None


_LEGACY_FLAT_IDENTITY_FIELDS: frozenset[str] = frozenset(
    {"conversation_id", "user_id", "customer_id", "agent_id", "correlation_id"}
)


class CallRequest(TolerantWireModel):
    """incoming tool call request from NATS.

    per-call identity dimensions (conversation_id, user_id, customer_id,
    agent_id, correlation_id) ride as a single nested
    :class:`CallContext` under ``context``. this replaces the previous
    shape where each dimension was a flat field; see
    :mod:`threetears.core.tool_wire.context`. ``correlation_id``
    lives exclusively on :attr:`CallContext.correlation_id`; the
    matching :class:`CallResponse` also carries a nested
    :class:`CallContext` (no top-level ``correlation_id`` string), so
    there is one shape for identity in both directions.

    the ``context`` field is optional because pure stateless tools
    (math, web search) do not require identity scope and the tool
    server degrades gracefully when it is omitted.

    :param tool_name: namespaced name of tool to invoke
    :ptype tool_name: str
    :param tool_version: version of tool to invoke
    :ptype tool_version: str
    :param arguments: tool input parameters
    :ptype arguments: dict[str, Any]
    :param context: unified identity + trace envelope for this call;
        ``None`` for stateless tool invocations. includes the
        ``correlation_id`` used for response routing and log correlation
    :ptype context: CallContext | None
    :param proxy_assertion: the registry proxy's body-bound, signed
        assertion for THIS request on the proxy→pod hop (binds the tool +
        arguments + a nonce so the discoverable internal subject can't be
        spliced/replayed). the pod verifies it on every call (enforce-only);
        a request without a valid assertion for this body is rejected
    :ptype proxy_assertion: str | None
    :param result_subject: where to DELIVER the answer, for a call the
        caller has decided is too long to answer on the reply inbox.
        ``allow_responses`` belongs to the connection that received the
        request, and the credential refresh that keeps a pod
        authenticated is a reconnect, so a tool that outlives one
        connection lifetime cannot answer on the inbox at all. when set,
        the pod acknowledges immediately with :class:`CallAccepted` and
        publishes the :class:`CallResponse` here instead -- a subject it
        holds a STANDING grant on, re-minted identically at every
        refresh. ``None`` keeps the fast synchronous path for short calls
    :ptype result_subject: str | None
    :param deadline_seconds: how much time the CALLER has left for this call
        (§10.10, SR-G2). A different quantity from
        :attr:`ToolManifestEntry.timeout_seconds`, which is the *tool's*
        declared ceiling, and from ``ToolServer(max_call_seconds=...)``, which
        is the *pod's* backstop -- this one is the caller's remaining budget,
        and it is the only one of the three that no other party can know. The
        effective bound is the MINIMUM of whichever are set: a caller cannot
        buy itself more time than the pod allows, and the pod does not spend
        time the caller has already given up waiting for.

        The model is a tolerant reader (:class:`~threetears.core.wire.TolerantWireModel`): a
        server predating a field ignores it, so a caller may send it before every pod has
        upgraded. Pods released while this model still forbade extras refuse the whole call on an
        unknown key, explicit ``null`` included, which is why the registry proxy prunes unset
        top-level optionals from the envelope
    :ptype deadline_seconds: float | None
    """

    tool_name: str
    tool_version: str
    arguments: dict[str, Any]
    context: CallContext | None = None
    proxy_assertion: str | None = None
    result_subject: str | None = None
    deadline_seconds: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_flat_identity_fields(cls, data: Any) -> Any:
        """reject removed flat identity fields with a migration pointer.

        when a caller still emits ``conversation_id`` / ``user_id`` /
        ``customer_id`` / ``agent_id`` / ``correlation_id`` as top-level
        fields on the wire, pydantic's generic ``extra='forbid'`` error
        is unhelpful for diagnosing the rename. this validator
        intercepts the common legacy shapes and raises a message that
        names the offending field and points at :class:`CallContext` so
        the fix site is obvious. any other unknown field falls through
        to the standard ``extra='forbid'`` error.

        :param data: raw input dict (mode='before' runs pre-coercion)
        :ptype data: Any
        :return: unchanged input when no legacy fields are present
        :rtype: Any
        :raises ValueError: when any legacy flat identity field is
            present on the wire
        """
        if isinstance(data, dict):
            offending = sorted(_LEGACY_FLAT_IDENTITY_FIELDS & data.keys())
            if offending:
                fields_list = ", ".join(offending)
                raise ValueError(
                    f"legacy flat identity field(s) {fields_list} rejected on "
                    f"CallRequest; migrated to CallContext, see "
                    f"threetears.core.tool_wire.CallContext"
                )
        return data


class CallResponse(BaseModel):
    """outgoing tool call response to NATS.

    responses carry the same :class:`CallContext` envelope as the
    inbound :class:`CallRequest`. the responder echoes
    ``request.context`` verbatim (or a minimally-populated
    :class:`CallContext` with just the correlation_id when that's all
    the responder knows) so downstream log consumers can correlate the
    reply to the inbound request. ``None`` when the inbound request
    carried no context (fully stateless call). identity never splits
    between "flat echo field" and "nested envelope" -- one shape in
    both directions.

    the registry answers its own caller with this same class, re-exported
    as :class:`threetears.registry.proxy.ProxyCallResponse`: the reply has
    one shape on both hops.

    :param success: whether tool execution succeeded
    :ptype success: bool
    :param content: result content string
    :ptype content: str
    :param metadata: optional additional metadata
    :ptype metadata: dict[str, Any] | None
    :param error: error message if execution failed
    :ptype error: str | None
    :param error_code: machine-readable code for a refusal the pod names -- one of its own gates,
        or the tool's :attr:`~threetears.agent.tools.base_tool.ToolResult.error_code` copied through
        unchanged -- relayed by the registry as it came, since its ``ProxyCallResponse`` is this
        class. ``None`` for a success and
        for every failure nobody names -- a tool that raised, a gate that reports only a reason
    :ptype error_code: str | None
    :param context: unified identity + trace envelope echoed from the
        inbound :class:`CallRequest`; ``None`` when the inbound request
        carried no context
    :ptype context: CallContext | None
    """

    success: bool
    content: str
    metadata: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None
    context: CallContext | None = None


class CallAccepted(BaseModel):
    """the immediate answer to a call that will be DELIVERED rather than replied to.

    Published on the request/reply inbox before any tool work starts, so it is always inside the
    window the receiving connection is guaranteed to survive. It exists for one reason: without it,
    the caller could not tell "this pod is running your 20-minute tool" from "this pod is gone", and
    the registry's failover to a sibling endpoint would have to wait out the whole tool budget before
    it could act.

    A pod that refuses the delivery subject answers with ``accepted=False`` and a reason, so the
    caller learns immediately rather than waiting for a result that will never be published.

    :param accepted: whether the pod took the call and will publish to the delivery subject
    :ptype accepted: bool
    :param pod_id: the accepting pod's own id, so the caller can confirm which endpoint took it
    :ptype pod_id: str
    :param result_subject: the subject the answer will be published to; echoed back so the caller
        can assert it matches the one it is waiting on
    :ptype result_subject: str | None
    :param error: why the call was refused, when ``accepted`` is false
    :ptype error: str | None
    """

    accepted: bool
    pod_id: str
    result_subject: str | None = None
    error: str | None = None


class HeartbeatMessage(BaseModel):
    """periodic heartbeat published by tool server.

    :param pod_id: unique identifier for this tool pod
    :ptype pod_id: str
    :param timestamp: ISO 8601 timestamp of heartbeat
    :ptype timestamp: str
    :param tools_count: number of tools registered in this pod
    :ptype tools_count: int
    """

    pod_id: str
    timestamp: str
    tools_count: int


class ProbeAck(BaseModel):
    """acknowledgment of a reachability probe from the registry.

    :param pod_id: unique identifier for this tool pod
    :ptype pod_id: str
    :param ready: whether pod is ready to serve calls
    :ptype ready: bool
    """

    pod_id: str
    ready: bool = True


class DiscoveryProbeToolEntry(BaseModel):
    """single tool in a discovery probe request.

    :param name: namespaced tool name
    :ptype name: str
    :param version: semver-compatible version string
    :ptype version: str
    """

    name: str
    version: str


class DiscoveryProbeRequest(BaseModel):
    """discovery request: what an agent, or a tool pod polling for its own readiness, asks the registry.

    the pod sends it from :meth:`threetears.agent.tools.server.ToolServer.wait_until_ready`; the
    registry parses it as :class:`threetears.registry.discovery.DiscoverRequest`, which is this
    class re-exported.

    ``pod_id`` is OMITTED from the serialized form when ``None``, so it never crosses the wire as
    an explicit ``null``.

    :param agent_id: the requesting agent's id, or -- when a ToolServer polls for its own
        readiness -- the polling server's pod-id; an agent's in-process server polls under its
        ``{agent_id}.{instance}`` composite, which names that agent. self-asserted: it decides what
        the requester is SHOWN, never what it may call
    :ptype agent_id: str
    :param tool_manifest: list of pinned tools to resolve
    :ptype tool_manifest: list[DiscoveryProbeToolEntry]
    :param pod_id: the polling pod's own id, so the registry reports the state of THIS pod's copy
        of each tool as ``requester_copy_status``
    :ptype pod_id: str | None
    """

    agent_id: str
    tool_manifest: list[DiscoveryProbeToolEntry]
    pod_id: str | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_pod_id(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """drop ``pod_id`` from the serialized form when it is ``None``.

        :param handler: pydantic's default serializer
        :ptype handler: SerializerFunctionWrapHandler
        :return: the serialized fields
        :rtype: dict[str, Any]
        """
        data: dict[str, Any] = handler(self)
        if data.get("pod_id") is None:
            data.pop("pod_id", None)
        return data

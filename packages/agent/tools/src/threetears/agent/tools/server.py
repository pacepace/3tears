"""ToolServer -- serves TearsTool instances via NATS.

registers tools, subscribes to call subject, publishes heartbeats,
handles graceful shutdown. each tool pod runs one ToolServer.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable
from uuid import NAMESPACE_DNS, UUID, uuid5, uuid7

from datetime import timedelta

from pydantic import BaseModel, ConfigDict, model_validator

from threetears.agent.audit import AuditEvent, publish_audit
from threetears.agent.tools.base_tool import TearsTool
from threetears.agent.tools.call_scope import (
    ToolCallScope,
    enter_call_scope,
)
from threetears.agent.tools.context_envelope import CallContext, bind_log_context
from threetears.agent.tools.config import (
    get_jwks_request_timeout,
)
from threetears.agent.tools.config import (
    get_object_resolve_request_timeout,
)
from threetears.agent.tools.config import (
    get_engagement_scope_request_timeout,
)
from threetears.agent.tools.config import (
    get_ready_poll_interval as _get_ready_poll_interval,
)
from threetears.agent.tools.config import (
    get_ready_timeout as _get_ready_timeout,
)
from threetears.agent.tools.config import (
    get_serve_ready_timeout,
)
from threetears.agent.tools.config import (
    get_connect_retry_backoff_cap,
    get_connect_retry_budget,
)
from threetears.agent.tools.engagement_resolver import HubEngagementScopeResolver
from threetears.agent.tools.object_resolver import HubObjectResolver
from threetears.core.namespaces import PLURAL_PREFIX_TOOL, build_namespace_name
from threetears.core.coordination.replay_guard import ReplayGuard
from threetears.core.security import CachedHubJwksProvider
from threetears.core.security.identity_token import (
    IdentityClaims,
    IdentityKeyNotFoundError,
    IdentityTokenError,
    canonical_call_hash,
    verify_identity_token,
)
from threetears.core.security.proxy_assertion import verify_proxy_assertion
from threetears.nats import (
    IncomingMessage,
    NatsClient,
    Principal,
    RequestError,
    Subjects,
    TokenCallback,
    inbox_prefix_for,
    set_default_namespace,
)
from threetears.nats.errors import NatsClientError
from threetears.observe import InflightRequestsGauge, clear_context, get_logger, traced

__all__ = [
    "CallRequest",
    "CallResponse",
    "DiscoveryProbeRequest",
    "DiscoveryProbeResponse",
    "DiscoveryProbeResultEntry",
    "DiscoveryProbeToolEntry",
    "HeartbeatMessage",
    "ProbeAck",
    "RegistrationManifest",
    "ToolManifestEntry",
    "ToolServer",
    "nats_connect",
    "tool_namespace_id",
    "tool_namespace_name",
]


def tool_namespace_name(mcp_name: str, version: str) -> str:
    """build the canonical ``platform.namespaces.name`` for a tool row.

    namespace-task-01 phase 9.5 pins the canonical shape at
    ``tools.<mcp_name>.<version>`` (plural prefix + dot separator +
    dot-sanitized segments). :func:`build_namespace_name` replaces
    any ``.`` inside a segment with ``-`` before interpolation — a
    mcp name like ``example.admin.backup`` comes through as
    ``example-admin-backup`` and a semver version like ``1.0.0``
    comes through as ``1-0-0``. the resulting shape stays unambiguous
    for cross-type lookups (no collision with a workspace-typed row
    sharing the name), preserves per-version pinning (different
    versions of the same tool remain distinct namespace rows), and
    keeps bulk-delete-on-deregister expressible via a
    ``LIKE 'tools.<sanitized-mcp>.%'`` pattern.

    :param mcp_name: tool mcp name (e.g. ``example.admin.backup``)
    :ptype mcp_name: str
    :param version: tool version (e.g. ``1.0.0``)
    :ptype version: str
    :return: canonical namespace name string
    :rtype: str
    """
    return build_namespace_name(PLURAL_PREFIX_TOOL, mcp_name, version)


def tool_namespace_id(
    mcp_name: str,
    version: str,
    agent_id: UUID | None,
) -> UUID:
    """derive deterministic :func:`uuid5` id for a tool namespace row.

    keying on the ``(mcp_name, version, agent_id_hex)`` triple makes
    concurrent register_tool racers on the same pod converge on the
    same ``platform.namespaces.id`` so
    :meth:`NamespaceCollection.save_entity` can resolve the replay
    through ``ON CONFLICT (id) DO UPDATE``. platform-built-in pods
    have ``agent_id=None`` and key on the literal string ``platform``
    so every platform pod's emission for the same tool/version
    collides on one row.

    :param mcp_name: tool mcp name (e.g. ``example.admin.backup``)
    :ptype mcp_name: str
    :param version: tool version (e.g. ``1.0.0``)
    :ptype version: str
    :param agent_id: owning-agent UUID or ``None`` for platform tools
    :ptype agent_id: UUID | None
    :return: deterministic uuid5 id for the namespace row
    :rtype: UUID
    """
    owner_key = agent_id.hex if agent_id is not None else "platform"
    return uuid5(
        NAMESPACE_DNS,
        f"threetears.namespaces.tool.{mcp_name}.{version}.{owner_key}",
    )


# sentinel tuple used by ``tool_names`` so callers always get an
# immutable shape back (prevents accidental mutation of the internal
# dict through the public accessor).
_EMPTY_TOOL_NAMES: tuple[str, ...] = ()

#: consecutive heartbeat cycles the NATS data plane may be unrecoverably unhealthy
#: (terminal close OR persistent auth/overflow wedge -- see :attr:`ToolServer.is_healthy`)
#: before the heartbeat-loop supervisor crashes the process so the orchestrator recycles
#: the pod. A transient network-drop reconnect keeps ``is_healthy`` True and resets the
#: streak, so this only trips on the states forever-reconnect cannot recover. 3 cycles at
#: the default 15s heartbeat is ~45s of sustained death -- prompt enough for cattle, long
#: enough to ride out a one-cycle blip.
_UNHEALTHY_EXIT_THRESHOLD: int = 3

if TYPE_CHECKING:
    from threetears.agent.tools.engagement_resolver import EngagementScopeResolver
    from threetears.agent.tools.object_resolver import ObjectResolver
    from threetears.media.contracts import ObjectStore

    from threetears.agent.tools.context import ToolContextManager

log = get_logger(__name__)

# the issuer the Hub stamps on identity tokens + the pod's clock-skew tolerance, for the
# defense-in-depth pod-side verification of the inbound identity token.
_IDENTITY_ISSUER = "hub"
_IDENTITY_LEEWAY_SECONDS = 60
# how long a proxy-assertion nonce is remembered for single-use enforcement; a TTL (not a timeout),
# sized to the assertion's short accept window (its exp + clock skew).
_ASSERTION_NONCE_TTL_SECONDS = 60


# ---------------------------------------------------------------------------
# NATS connection helper (patched in tests)
# ---------------------------------------------------------------------------


async def nats_connect(
    url: str,
    *,
    namespace: str = "3tears",
    user: str | None = None,
    password: str | None = None,
    auth_token: TokenCallback | None = None,
    conn_id: str | None = None,
) -> NatsClient:
    """connect to NATS server via the canonical wrapper.

    standalone tool pods that did not receive a pre-connected
    :class:`NatsClient` from the bootstrap call this helper to open
    their own. tests patch this symbol to swap a fake transport in.

    two credential styles are supported, mutually exclusive:

    * ``auth_token`` -- a zero-arg PROVIDER (decentralized / auth-callout auth). when supplied it is
      presented INSTEAD of ``user``/``password``: nats-py invokes it on every (re)connect, so each
      reconnect re-presents a freshly-minted, still-valid identity token, and the NATS server
      forwards it to the auth-callout responder which verifies it and mints the connection's scoped
      user JWT. this is the per-key-identity path -- the CALLER builds the minter (a tool pod mints
      a short-lived identity JWT from its own Ed25519 key).
    * ``user`` / ``password`` -- static config-mode ``authorization.users`` creds. the legacy / dev
      fallback: under enforce-only connection auth (v0.13.9) a standalone tool server on a
      non-callout bus presents its OWN static creds (the enforcing bus has no ``no_auth_user``).
      ``None`` leaves credential auth off for tests + a non-enforcing bus.

    (agent-owned tool pods take the pre-connected ``nats_client`` path instead -- that connection is
    the agent runtime's, authenticated via the callout.)

    :param url: NATS server URL
    :ptype url: str
    :param namespace: NATS subject namespace prefix bound on the wrapper
    :ptype namespace: str
    :param user: NATS static username (config-mode ``authorization.users``); ``None`` -> no creds.
        ignored when ``auth_token`` is supplied.
    :ptype user: str | None
    :param password: NATS static password paired with ``user``; ``None`` -> no creds. ignored when
        ``auth_token`` is supplied.
    :ptype password: str | None
    :param auth_token: NATS auth-token PROVIDER (zero-arg callable returning the current token);
        presented INSTEAD of ``user``/``password`` when set. ``None`` leaves token auth off.
    :ptype auth_token: TokenCallback | None
    :return: connected canonical wrapper client
    :rtype: NatsClient
    """
    if auth_token is not None:
        # per-key-identity path: present the self-minted token provider and NOT the static creds
        # (the auth-callout mints this connection's scoped user JWT from the verified token). KEY the
        # reply inbox on the pod id so it falls under the minted JWT's `_INBOX_tool_pod_{id}.>`
        # subscribe grant: nats-py's default random `_INBOX.*` is OUTSIDE that scoped grant, so the
        # first request/reply (the JetStream account probe, the registry handshake) hits a
        # "permissions violation for subscription" and the connect wedges. the callout scopes the
        # grant on the VERIFIED pod id, so conn_id MUST be the pod id (not the spoofable connect name).
        client = await NatsClient.connect(
            nats_url=url,
            nats_subject_namespace=namespace,
            client_name=conn_id or "tool-server",
            auth_token=auth_token,
            inbox_prefix=(inbox_prefix_for(Principal.TOOL_POD, conn_id=conn_id) if conn_id is not None else None),
        )
    else:
        client = await NatsClient.connect(
            nats_url=url,
            nats_subject_namespace=namespace,
            client_name="tool-server",
            user=user,
            password=password,
        )
    return client


# ---------------------------------------------------------------------------
# Wire-format Pydantic models
# ---------------------------------------------------------------------------


class ToolManifestEntry(BaseModel):
    """single tool entry in registration manifest.

    The two visibility flags ride with each tool's manifest entry so
    the hub-side ``ToolNamespaceEmitter`` can stamp them onto the
    ``platform.namespaces`` row it writes. Defaults match
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
    requires_confirmation: bool = False


class RegistrationManifest(BaseModel):
    """manifest sent on connect to register all tools.

    ownership identity (``owner_agent_id`` / ``customer_id``) rides on
    the manifest so downstream namespace materialization can stamp
    the right scope on each ``platform.namespaces`` row without
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
        ``None`` for platform-built-in pods
    :ptype owner_agent_id: UUID | None
    :param customer_id: owning-customer UUID for agent-spun pods;
        ``None`` for platform-built-in pods
    :ptype customer_id: UUID | None
    """

    pod_id: str
    tools: list[ToolManifestEntry]
    bootstrap_token: str | None = None
    owner_agent_id: UUID | None = None
    customer_id: UUID | None = None


_LEGACY_FLAT_IDENTITY_FIELDS: frozenset[str] = frozenset(
    {"conversation_id", "user_id", "customer_id", "agent_id", "correlation_id"}
)


class CallRequest(BaseModel):
    """incoming tool call request from NATS.

    per-call identity dimensions (conversation_id, user_id, customer_id,
    agent_id, correlation_id) ride as a single nested
    :class:`CallContext` under ``context``. this replaces the previous
    shape where each dimension was a flat field; see
    :mod:`threetears.agent.tools.context_envelope`. ``correlation_id``
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
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    tool_version: str
    arguments: dict[str, Any]
    context: CallContext | None = None
    proxy_assertion: str | None = None

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
                    f"threetears.agent.tools.context_envelope.CallContext"
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

    :param success: whether tool execution succeeded
    :ptype success: bool
    :param content: result content string
    :ptype content: str
    :param metadata: optional additional metadata
    :ptype metadata: dict[str, Any] | None
    :param error: error message if execution failed
    :ptype error: str | None
    :param context: unified identity + trace envelope echoed from the
        inbound :class:`CallRequest`; ``None`` when the inbound request
        carried no context
    :ptype context: CallContext | None
    """

    success: bool
    content: str
    metadata: dict[str, Any] | None = None
    error: str | None = None
    context: CallContext | None = None


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
    """discovery request used by :meth:`ToolServer.wait_until_ready`.

    mirrors :class:`threetears.registry.discovery.DiscoverRequest` so
    the pod can poll the registry without importing from the registry
    package (which would create a circular dependency).

    :param agent_id: pod identifier standing in for agent_id in the wire
    :ptype agent_id: str
    :param tool_manifest: list of pinned tools to resolve
    :ptype tool_manifest: list[DiscoveryProbeToolEntry]
    """

    agent_id: str
    tool_manifest: list[DiscoveryProbeToolEntry]


class DiscoveryProbeResultEntry(BaseModel):
    """single tool result in a discovery probe response.

    only the fields needed by readiness polling are modeled; extra
    fields in the wire are ignored by pydantic default.

    :param name: namespaced tool name
    :ptype name: str
    :param version: semver-compatible version string
    :ptype version: str
    :param status: availability status reported by registry
    :ptype status: str
    """

    name: str
    version: str
    status: str


class DiscoveryProbeResponse(BaseModel):
    """discovery response used by :meth:`ToolServer.wait_until_ready`.

    :param agent_id: identifier of requester echoed back
    :ptype agent_id: str
    :param tools: list of resolved tool results
    :ptype tools: list[DiscoveryProbeResultEntry]
    """

    agent_id: str
    tools: list[DiscoveryProbeResultEntry]


# ---------------------------------------------------------------------------
# ToolServer
# ---------------------------------------------------------------------------


class ToolServer:
    """serves TearsTool instances via NATS.

    registers tools, subscribes to call subject, publishes heartbeats,
    handles graceful shutdown. each tool pod runs one ToolServer.
    """

    def __init__(
        self,
        *,
        namespace_collection: Any,
        nats_url: str = "",
        namespace: str = "3tears",
        nats_user: str | None = None,
        nats_password: str | None = None,
        pod_id: str | None = None,
        heartbeat_interval: float = 15.0,
        bootstrap_token: str | None = None,
        auth_token: TokenCallback | None = None,
        context_factory: ("Callable[[UUID, UUID], Awaitable[ToolContextManager]] | None") = None,
        nats_client: "NatsClient | None" = None,
        agent_id: UUID | None = None,
        customer_id: UUID | None = None,
        jwks_provider: Callable[[], dict[str, Any]] | None = None,
        jwks_refresh: Callable[[], Awaitable[bool]] | None = None,
        assertion_replay_guard: "ReplayGuard | None" = None,
        object_store: "ObjectStore | None" = None,
        object_resolver: "ObjectResolver | None" = None,
        engagement_resolver: "EngagementScopeResolver | None" = None,
    ) -> None:
        """initialize tool server.

        the NATS connection can be supplied two ways. callers that own
        a connection lifecycle (bootstrap, orchestrator) pass
        ``nats_client`` and leave ``nats_url`` at its default; the
        server attaches to that client in :meth:`serve` and will NOT
        disconnect it in :meth:`shutdown` (lifecycle belongs to the
        caller). standalone callers pass ``nats_url``; the server
        opens its own connection in :meth:`serve` and closes it in
        :meth:`shutdown`. exactly one of the two must be supplied
        with a non-empty value.

        :param nats_url: NATS server connection URL; leave empty when
            supplying ``nats_client``
        :ptype nats_url: str
        :param namespace: NATS subject namespace prefix
        :ptype namespace: str
        :param nats_user: NATS static username for the standalone (``nats_url``) connect path under
            enforce-only connection auth; the enforcing bus has no ``no_auth_user`` so a standalone
            tool server MUST present a credential. ignored on the pre-connected ``nats_client`` path
            (that connection carries its own auth). ``None`` -> anonymous (tests / non-enforcing bus)
        :ptype nats_user: str | None
        :param nats_password: NATS static password paired with ``nats_user``
        :ptype nats_password: str | None
        :param pod_id: unique pod identifier (generated if not provided)
        :ptype pod_id: str | None
        :param heartbeat_interval: seconds between heartbeat publishes
        :ptype heartbeat_interval: float
        :param bootstrap_token: STATIC authentication token carried on the registration manifest for
            registry-layer verification. under per-key identity this is the dev/legacy fallback: when
            ``auth_token`` is supplied the manifest instead carries a FRESH self-minted identity JWT
            re-minted from that provider on every publish, and this static value is unused. left
            ``None`` on the callout path.
        :ptype bootstrap_token: str | None
        :param auth_token: NATS auth-token PROVIDER (a zero-arg callable returning the current token)
            for the standalone (``nats_url``) connect path under per-key identity. when supplied it
            is presented to NATS INSTEAD of ``nats_user``/``nats_password``: nats-py invokes it on
            every (re)connect so each reconnect re-presents a freshly-minted identity JWT, and the
            auth-callout responder verifies it + mints the connection's scoped user JWT. the SAME
            provider also mints the registration manifest's ``bootstrap_token`` fresh on every
            :meth:`publish_registration`, so the registry-layer verifier sees a still-valid JWT even
            on a re-publish long after connect. ``ToolServer`` stays issuer-agnostic -- the CALLER
            builds the minter (e.g. ``IdentityMinter``) and wraps its ``mint`` in this provider.
            ignored on the pre-connected ``nats_client`` path (that connection carries its own auth).
            ``None`` -> fall back to ``nats_user``/``nats_password`` (dev / non-callout bus).
        :ptype auth_token: TokenCallback | None
        :param context_factory: optional async factory taking
            ``(conversation_id, user_id)`` and returning a
            :class:`ToolContextManager` scoped to that conversation.
            when supplied, the server constructs a
            :class:`ToolCallScope` per incoming call and installs it
            via :func:`enter_call_scope` so conversation-aware tools
            (workspace_*, pin-backed builtins) can resolve their
            context through :func:`tool_context_provider`. when
            omitted, tools that require a context crash with a
            :class:`RuntimeError` at first use, same as today
        :ptype context_factory: Callable[[UUID, UUID], Awaitable[ToolContextManager]] | None
        :param nats_client: pre-connected canonical
            :class:`threetears.nats.NatsClient` wrapper supplied by a
            caller that owns its lifecycle (typically the agent
            bootstrap sharing one connection across strategy,
            handler, and heartbeat). when set, ``nats_url`` is
            ignored and the server will not disconnect the client on
            shutdown
        :ptype nats_client: NatsClient | None
        :param agent_id: owning-agent UUID for this pod. stamped on
            the ``owner_agent_id`` axis of every baseline ``tool.call``
            audit envelope emitted from :meth:`handle_call` and on
            every tool namespace row emitted from
            :meth:`register_tool`. ``None`` in platform-spun ToolServers
            (platform built-in tool pods have no owning agent); each
            namespace row then lands with ``owner_agent_id=NULL``
            matching the ``shared``-type namespace shape.
        :ptype agent_id: UUID | None
        :param customer_id: owning-customer UUID stamped on every
            tool namespace row emitted by :meth:`register_tool`.
            paired with ``agent_id``: agent-spun pods carry both,
            platform-spun pods carry neither and emit with
            ``customer_id=NULL``.
        :ptype customer_id: UUID | None
        :param namespace_collection: three-tier
            :class:`NamespaceCollection` from the agent-side stack.
            :meth:`register_tool` calls
            :meth:`NamespaceCollection.save_entity` with a
            :class:`NamespaceEntity` of type ``tool`` so the unified
            rbac evaluator can resolve per-call authorization against
            a first-class namespace id; :meth:`deregister_tool` calls
            :meth:`NamespaceCollection.delete`. typed ``Any`` at this
            boundary because :mod:`threetears.agent.tools` sits below
            the consumer hub's broker namespaces in the import graph;
            the concrete Collection is wired by the bootstrap caller.
            ``None`` suppresses emission entirely — reserved for
            in-process tests and standalone dev that never touch
            ``platform.namespaces``; production callers MUST supply a
            Collection or namespace materialization silently falls
            behind and rbac resolution fails open.
        :ptype namespace_collection: Any
        :param jwks_refresh: optional zero-arg coroutine triggering ONE
            immediate, debounced + rate-limited Hub JWKS refresh, returning
            whether it ran (typically
            :meth:`CachedHubJwksProvider.refresh_now`). When a pod-side token
            verification fails because the cached JWKS holds no key for the
            token's ``kid`` (a Hub re-key the cache has not caught up to),
            :meth:`_verify_identity` calls it ONCE and re-verifies, so a valid
            token signed under a freshly-rotated key self-heals on the first
            such failure rather than after a full steady refresh interval. Left
            ``None`` here for the injected-provider path (tests); :meth:`serve`
            wires it to the owned provider's ``refresh_now`` when the pod
            self-provisions its JWKS provider.
        :ptype jwks_refresh: Callable[[], Awaitable[bool]] | None
        :param object_store: the pod's single streaming object store, or
            ``None`` when no S3 is configured. installed on every per-call
            :class:`ToolCallScope` (alongside the context manager) so
            producing tools reach it through :func:`current_scope` without
            per-tool constructor plumbing; the pod owns the one instance,
            the scope just carries the reference per call.
        :ptype object_store: ObjectStore | None
        :param object_resolver: the pod's object-id resolver, or ``None`` to
            self-provision one in :meth:`serve` from the NATS client (the
            default; it needs no S3 creds, only NATS). installed on every
            per-call :class:`ToolCallScope` so consuming tools resolve object
            ids to keys through :func:`current_scope`. an injected resolver
            (tests) is used as-is and not self-provisioned.
        :ptype object_resolver: ObjectResolver | None
        :param engagement_resolver: the pod's engagement-scope resolver, or
            ``None`` to self-provision one in :meth:`serve` from the NATS client
            (the default; it needs no S3 creds, only NATS). installed on every
            per-call :class:`ToolCallScope` so tools resolve the call's
            ``engagement_id`` to its authorized target set through
            :func:`current_scope`. an injected resolver (tests) is used as-is and
            not self-provisioned.
        :ptype engagement_resolver: EngagementScopeResolver | None
        :raises ValueError: when neither ``nats_url`` nor
            ``nats_client`` carries a usable value
        """
        if not nats_url and nats_client is None:
            raise ValueError("ToolServer requires either nats_url or nats_client; neither was supplied")
        self._nats_url = nats_url
        self._namespace = namespace
        self._nats_user = nats_user
        self._nats_password = nats_password
        self._pod_id = pod_id or str(uuid7())
        self._heartbeat_interval = heartbeat_interval
        self._bootstrap_token = bootstrap_token
        # per-key-identity connect credential provider (self-minted identity JWT). when set it is
        # presented to NATS on connect INSTEAD of user/password AND re-minted for each registration
        # manifest so the registry-layer verifier always sees a fresh JWT. None -> static fallback.
        self._auth_token = auth_token
        self._context_factory = context_factory
        self._agent_id = agent_id
        self._customer_id = customer_id
        # defense-in-depth: the pod re-verifies the Hub identity token AND the proxy's body-bound
        # assertion on every inbound call (closes the direct-internal-subject bypass). enforce-only
        # -- there is no off/warn ladder; a call the pod cannot verify is always rejected.
        self._jwks_provider = jwks_provider
        # reactive Hub-rekey self-heal: ``serve()`` wires this to the owned provider's refresh_now
        # when self-provisioning; an injected-provider caller may pass one or leave it None (inert).
        self._jwks_refresh = jwks_refresh
        self._namespace_collection = namespace_collection
        # the pod's single object store (None when no S3 configured); installed on
        # every per-call ToolCallScope so producing tools reach it ambiently.
        self._object_store = object_store
        # the pod's single object-id resolver; None here is self-provisioned in
        # serve() from the NATS client (needs no S3 creds), then installed on
        # every per-call ToolCallScope so consuming tools resolve ids ambiently.
        self._object_resolver = object_resolver
        # the pod's single engagement-scope resolver; None here is self-provisioned
        # in serve() from the NATS client (needs no S3 creds), then installed on
        # every per-call ToolCallScope so tools re-authorize against the engagement.
        self._engagement_resolver = engagement_resolver
        self._tools: dict[str, TearsTool] = {}
        self._nc: "NatsClient | None" = nats_client
        self._owns_nats_connection: bool = nats_client is None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._running = False
        self._owned_jwks_provider: CachedHubJwksProvider | None = None
        # the proxy-assertion replay guard is REQUIRED at verify time (a guardless pod must NOT
        # silently skip single-use enforcement). serve() always provisions it over the pod's
        # connection; callers driving handle_call without serve() (tests) inject one here.
        self._assertion_replay_guard: ReplayGuard | None = assertion_replay_guard
        # leak-safe in-flight-requests gauge bracketed around every handle_call:
        # the tool-pod bootstrap serves it on the shared HealthServer's /metrics
        # route so KEDA's prometheus scaler can autoscale the pod's Deployment on
        # aggregate in-flight call load (the tools.internal RPC path is a
        # queue-group request/reply, not JetStream, so there is no stream backlog
        # for the nats scaler to read).
        self._inflight_gauge = InflightRequestsGauge("threetears_tools_inflight_requests")
        self._shutdown_event = asyncio.Event()
        self._ready_event = asyncio.Event()

    @property
    def pod_id(self) -> str:
        """return the unique pod identifier this server was constructed with.

        exposed as a public property so callers (agent runtime bootstrap)
        can derive a UUID from it without reaching into ``_pod_id``.

        :return: pod identifier string (UUID hex form)
        :rtype: str
        """
        return self._pod_id

    @property
    def tools_count(self) -> int:
        """return number of tools currently registered on this server.

        used by hub observability code (datasource tool pod, delegation
        manager) that logs ``tools_count=N`` on startup and by readiness
        checks that decide whether to start ``serve()`` at all. reading
        this property is O(1) and takes no locks; it is safe to call at
        any point in the server's lifecycle, including before
        ``serve()`` has connected and after ``shutdown()`` has
        completed.

        :return: number of registered tools
        :rtype: int
        """
        return len(self._tools)

    def render_metrics(self) -> tuple[str, bytes]:
        """render the pod's prometheus metrics in text exposition format.

        returns ``(content_type, body)`` for the shared HealthServer's
        ``/metrics`` route (wired by :class:`ToolServerBootstrap`), so the
        pod's in-flight-requests gauge is scrapable by KEDA's prometheus
        scaler through the one HTTP listener the pod already runs for
        ``/healthz``.

        :return: tuple of prometheus content-type and exposition body
        :rtype: tuple[str, bytes]
        """
        return self._inflight_gauge.render()

    @property
    def tool_names(self) -> tuple[str, ...]:
        """return an immutable snapshot of registered tool keys.

        keys are the internal ``name@version`` form the server uses for
        dispatch. returns a tuple (not the internal dict) so callers
        cannot mutate the server's state through the accessor: the
        snapshot reflects the registration set at call time and does
        not update when subsequent :meth:`register_tool` /
        :meth:`deregister_tool` calls change the underlying dict.
        iteration order follows registration order (dict insertion
        order) but callers MUST NOT rely on it for correctness.

        :return: tuple of ``name@version`` strings
        :rtype: tuple[str, ...]
        """
        if not self._tools:
            return _EMPTY_TOOL_NAMES
        return tuple(self._tools.keys())

    @property
    def is_connected(self) -> bool:
        """return whether this server has an active NATS connection.

        ``True`` between the moment :meth:`serve` completes
        ``nats_connect`` and the moment :meth:`shutdown` calls
        ``close()``; ``False`` otherwise. callers that need to gate
        publish work on the server's connectivity state should use this
        property rather than reaching into ``_nc``. this is the only
        public view on the NATS client — the client itself is NOT
        exposed because tool callers have no legitimate need to
        ``subscribe``/``request``/``publish`` on the server's
        connection (those flows happen via NATS proxies or their own
        clients).

        :return: true iff ``serve()`` has connected and ``shutdown()``
            has not yet closed the client
        :rtype: bool
        """
        return self._nc is not None

    @property
    def is_healthy(self) -> bool:
        """return whether this pod's NATS data plane is actually usable.

        The load-bearing LIVENESS signal (as opposed to :attr:`is_connected`,
        which reports only that a client OBJECT exists and stays ``True`` for a
        dead connection -- the exact bug that lets a pod wedge "Running" forever
        with a closed NATS connection). Delegates to the canonical client's real
        state: ``False`` when the connection is terminally ``is_closed`` (a NATS
        user-JWT expiry ``-ERR`` routes straight to a close that forever-reconnect
        does NOT cover) OR when the client is stuck in a persistent
        auth-violation / outbound-overflow loop (``not is_healthy`` -- a wedge
        where ``is_closed`` never trips). A TRANSIENT network-drop reconnect keeps
        this ``True`` (``is_closed`` False, ``is_healthy`` True) so a normal
        forever-reconnect never flaps liveness. A ``/healthz`` liveness probe
        keyed on this trips on the unrecoverable states so k8s recycles the pod.

        :return: true iff the NATS connection is alive and not auth/overflow-wedged
        :rtype: bool
        """
        nc = self._nc
        return nc is not None and not nc.is_closed and nc.is_healthy

    @property
    def jwks_warmed(self) -> bool:
        """whether the pod's Hub-JWKS provider has completed its first successful fetch.

        readiness gate: before the JWKS cache warms, the pod verifies every inbound identity token
        against an EMPTY keyset and rejects fail-closed, so a k8s readiness probe must report
        NOT-READY until this is true -- otherwise the pod accepts calls it is guaranteed to fail.
        The owned :class:`CachedHubJwksProvider` (self-provisioned in :meth:`serve`) exposes
        ``is_warmed``; that drives this. Before :meth:`serve` provisions a provider it is ``False``
        (NOT-READY). An INJECTED provider with no warmth signal (tests / a static JWKS that needs no
        warm-up) is treated as ready, since it returns keys synchronously from the first call.

        :return: true once the pod's JWKS provider can verify a token (or there is nothing to warm)
        :rtype: bool
        """
        provider = self._jwks_provider
        if provider is None:
            return False
        warmed = getattr(provider, "is_warmed", None)
        if isinstance(warmed, bool):
            return warmed
        return True

    @property
    def is_running(self) -> bool:
        """return whether :meth:`serve` has started and not yet shut down.

        flips ``True`` inside :meth:`serve` once the NATS connection is
        attached and the heartbeat loop is about to start; flips back to
        ``False`` at the top of :meth:`shutdown`. exposed as a public
        read so health-probe callers and tests can observe the serve
        loop's lifecycle without reaching into internal state.
        independent of :attr:`is_connected`: a standalone server flips
        both together, but the combined state is useful when debugging
        a partial shutdown where one moves before the other.

        :return: true while the serve loop is active
        :rtype: bool
        """
        return self._running

    @property
    def owns_nats_connection(self) -> bool:
        """return whether :meth:`shutdown` will close the NATS client.

        ``True`` when the server was constructed with ``nats_url`` and
        opened its own connection in :meth:`serve`; ``False`` when the
        caller supplied a pre-connected ``nats_client`` whose lifecycle
        belongs to them. callers coordinating shutdown ordering across
        multiple NATS-using components (agent bootstrap sharing one
        connection between tool server, graph handler, heartbeat) use
        this property to decide whether they must close the connection
        themselves after :meth:`shutdown` returns.

        :return: true iff the server will close the NATS client on shutdown
        :rtype: bool
        """
        return self._owns_nats_connection

    async def wait_ready(self, timeout: float | None = None) -> None:
        """block until serve() has subscribed to NATS and published registration.

        callers that spawn serve() in a background task should await this
        before sending tool calls to avoid the race where the first call
        arrives before the subscription is live. when ``timeout`` is
        ``None`` the value is sourced from
        ``THREETEARS_TOOLSERVER_SERVE_READY_TIMEOUT`` (platform default
        applied when the variable is unset or malformed).

        :param timeout: maximum seconds to wait; ``None`` reads from config
        :ptype timeout: float | None
        :raises asyncio.TimeoutError: if serve() does not become ready in time
        """
        resolved = get_serve_ready_timeout() if timeout is None else timeout
        await asyncio.wait_for(self._ready_event.wait(), timeout=resolved)

    def register(self, tool: TearsTool) -> None:
        """register tool for serving via NATS.

        :param tool: TearsTool instance to register
        :ptype tool: TearsTool
        """
        key = f"{tool.mcp_name()}@{tool.mcp_version()}"
        self._tools[key] = tool
        log.info(
            "registered tool",
            extra={"extra_data": {"tool_key": key, "pod_id": self._pod_id}},
        )

    def unregister(self, mcp_name: str) -> bool:
        """remove tool registration by mcp_name, regardless of version.

        supports atomic swap flows (hot-reload of workspace config,
        per-agent plugin refresh) where a tool family is registered,
        then replaced with a rebuilt instance bound to new dependencies.
        matches on mcp_name prefix of the internal ``name@version`` key so
        callers do not have to know the version. returns True when one or
        more keys were removed; False when nothing matched so callers can
        distinguish a no-op from a successful removal without silently
        swallowing an invariant break.

        :param mcp_name: namespaced tool name to remove
        :ptype mcp_name: str
        :return: True when one or more keys were removed
        :rtype: bool
        """
        prefix = f"{mcp_name}@"
        matched_keys = [key for key in self._tools if key.startswith(prefix)]
        for key in matched_keys:
            del self._tools[key]
        removed = len(matched_keys) > 0
        if removed:
            log.info(
                "unregistered tool",
                extra={
                    "extra_data": {
                        "mcp_name": mcp_name,
                        "removed_keys": matched_keys,
                        "pod_id": self._pod_id,
                    }
                },
            )
        return removed

    async def _open_connection_with_retry(self) -> NatsClient:
        """open the pod's OWN NATS connection, retrying a not-yet-ready platform (k8s unordered startup).

        Pods start in ANY order: the hub -- the auth-callout responder AND this pod's seeded
        ``tool_pods`` row -- may not be up when a standalone tool pod boots, and the connect is rejected
        until it is. rather than die on the first failure and lean on an external restart loop, retry the
        initial connect with capped exponential backoff until the platform admits us. FAIL-VISIBLE: after
        :func:`get_connect_retry_budget` seconds it re-raises, so a genuine misconfig (wrong key / issuer)
        surfaces as a crash / CrashLoopBackoff rather than an invisible forever-retry. runtime reconnects
        AFTER the first success are owned by the :class:`NatsClient` wrapper (this covers only the very
        first connect). only the self-owned-connection path reaches here; an injected client never does.

        :return: the connected client
        :rtype: NatsClient
        :raises NatsClientError: when the platform does not admit the pod within the retry budget
        """
        budget = get_connect_retry_budget()
        backoff_cap = get_connect_retry_backoff_cap()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + budget
        delay = 1.0
        attempt = 0
        client: NatsClient | None = None
        while client is None:
            attempt += 1
            try:
                client = await nats_connect(
                    self._nats_url,
                    namespace=self._namespace,
                    user=self._nats_user,
                    password=self._nats_password,
                    auth_token=self._auth_token,
                    conn_id=self._pod_id,
                )
            except (NatsClientError, OSError) as exc:
                if loop.time() >= deadline:
                    log.error(
                        "tool pod could not connect to NATS within the retry budget; failing loud",
                        extra={"extra_data": {"pod_id": self._pod_id, "attempts": attempt, "budget_s": budget}},
                    )
                    raise
                log.warning(
                    "tool pod NATS connect not ready (platform still starting?); retrying",
                    extra={
                        "extra_data": {
                            "pod_id": self._pod_id,
                            "attempt": attempt,
                            "retry_in_s": delay,
                            "error": str(exc),
                        }
                    },
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, backoff_cap)
        return client

    @traced()
    async def serve(self) -> None:
        """begin serving registered tools on NATS.

        when the server was constructed with an injected
        ``nats_client`` the connection is already open and the server
        attaches to it. when the server was constructed with a
        ``nats_url`` it opens its own connection here. either way the
        server subscribes to call and probe subjects first so both
        are live before the registry can attempt a reachability
        probe, publishes the registration manifest, starts the
        heartbeat loop, then waits for the shutdown signal. ordering
        matters: subscribing before publishing eliminates the race
        where the registry issues a probe to a subject the pod has
        not yet bound.
        """
        if self._nc is None:
            self._nc = await self._open_connection_with_retry()
            log.info(
                "connected to NATS",
                extra={
                    "extra_data": {
                        "nats_url": self._nats_url,
                        "pod_id": self._pod_id,
                    }
                },
            )
        else:
            log.info(
                "using injected NATS connection",
                extra={
                    "extra_data": {
                        "pod_id": self._pod_id,
                    }
                },
            )
        # the server's configured ``namespace`` is the one that must
        # appear on every subject built below. when the server opens
        # its own connection :class:`NatsClient.connect` already binds
        # the prefix; when a caller injects a pre-connected client, we
        # bind here so :class:`Subjects` builders pick up the same
        # prefix the server was constructed with rather than whatever
        # the injected client picked up at connect time.
        set_default_namespace(self._namespace)
        self._running = True

        # self-provision the pod's verifier dependencies over this connection: a Hub-JWKS provider
        # (verifies the identity token + the proxy's body-bound assertion) and a replay guard
        # (single-use proxy-assertion nonces). always provisioned under enforce-only; fail-closed
        # until the first JWKS fetch.
        assert self._nc is not None  # connected or injected above
        if self._jwks_provider is None:
            owned = CachedHubJwksProvider(self._nc, request_timeout_seconds=get_jwks_request_timeout())
            await owned.start()
            self._jwks_provider = owned
            self._owned_jwks_provider = owned
            # reactive self-heal on a Hub re-key: wire the verify path to the owned provider's
            # refresh_now so a kid-not-in-cache miss triggers ONE immediate, debounced + rate-limited
            # refresh + re-verify. only set when the pod self-provisions; an injected provider keeps
            # whatever (possibly None) trigger the constructor was given.
            self._jwks_refresh = owned.refresh_now
        if self._object_resolver is None:
            # self-provision the object-id resolver over this connection so
            # consuming tools can turn an object id into its stored key. it
            # needs only NATS (the hub verifies the caller's identity_token +
            # owns the objects table), so -- unlike the object store -- the pod
            # does not have to be wired with S3 creds to resolve.
            self._object_resolver = HubObjectResolver(
                self._nc,
                request_timeout_seconds=get_object_resolve_request_timeout(),
            )
        if self._engagement_resolver is None:
            # self-provision the engagement-scope resolver over this connection so
            # tools can re-authorize each call against the engagement's target set.
            # like the object resolver it needs only NATS (the hub verifies the
            # forwarded identity_token + owns the engagement tables), so the pod
            # does not have to hold any creds of its own to resolve scope.
            self._engagement_resolver = HubEngagementScopeResolver(
                self._nc,
                request_timeout_seconds=get_engagement_scope_request_timeout(),
            )
        if self._assertion_replay_guard is None:
            self._assertion_replay_guard = ReplayGuard(
                self._nc,
                bucket_name="proxy_assertion_nonces",
                ttl_seconds=_ASSERTION_NONCE_TTL_SECONDS,
            )

        # DQ-B7 queue-group sweep: call_subject and probe_subject are
        # pod-specific (``{ns}.tools.internal.{pod_id}`` /
        # ``{ns}.tools.probe.{pod_id}``); only this pod's connection
        # binds them, so a queue group would be redundant. heartbeat
        # publishes are write-only and need no queue group.
        call_subject = Subjects.tools_internal(self._pod_id)
        await self._nc.subscribe(subject=call_subject, cb=self.handle_call)
        log.info(
            "subscribed to call subject",
            extra={"extra_data": {"subject": call_subject.path}},
        )

        probe_subject = Subjects.tools_probe(self._pod_id)
        await self._nc.subscribe(subject=probe_subject, cb=self.handle_probe)
        log.info(
            "subscribed to probe subject",
            extra={"extra_data": {"subject": probe_subject.path}},
        )

        await self.publish_registration()

        self._ready_event.set()

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        await self._shutdown_event.wait()

    async def handle_probe(self, msg: IncomingMessage) -> None:
        """public NATS-subject handler replying to reachability probes.

        bound by :meth:`serve` as the ``cb`` callback on
        ``{namespace}.tools.probe.{pod_id}``. tests exercise this
        surface directly; the name + single-``msg`` shape are part of
        the stability contract.

        replies with a ProbeAck carrying pod_id and ready=True so the
        registry can promote this pod's pending endpoints to available.
        the probe handler does NOT set the readiness event directly --
        readiness is determined by polling the registry's discovery
        response until every registered tool reports as 'available',
        which guarantees the registry's catalog state has completed
        the pending -> available transition before ``wait_until_ready``
        unblocks.

        :param msg: incoming wrapper envelope carrying the probe request
        :ptype msg: IncomingMessage
        """
        if msg.reply_subject is None or self._nc is None:
            return
        ack = ProbeAck(pod_id=self._pod_id, ready=True)
        await self._nc.publish_reply(reply_subject=msg.reply_subject, message=ack)

    async def wait_until_ready(self, timeout: float | None = None) -> bool:
        """block until registry catalog reports every tool as available.

        polls the registry's discovery subject with this pod's tool
        manifest until the catalog reports every entry as 'available',
        then returns True. unlike an event-driven probe-arrival signal,
        this waits for the full probe -> mark_ready -> discovery round-
        trip so routable state is guaranteed when the function returns
        (no residual race where ``TOOL_NOT_READY`` could still fire for
        a fresh caller). returns False on timeout. intended as the
        developer-friendly substitute for ``asyncio.sleep(1.0)`` after
        ``serve``.

        :param timeout: seconds to wait before giving up. sourced
            from THREETEARS_TOOLSERVER_READY_TIMEOUT env var if not
            provided.
        :ptype timeout: float | None
        :return: True if ready within timeout, False on timeout
        :rtype: bool
        :raises RuntimeError: if called before ``serve`` connects NATS
        """
        if self._nc is None:
            raise RuntimeError("wait_until_ready called before serve() connected NATS")
        effective_timeout = timeout if timeout is not None else _get_ready_timeout()
        deadline = asyncio.get_event_loop().time() + effective_timeout
        manifest_names = [
            ToolManifestEntry(
                name=t.mcp_schema().name,
                version=t.mcp_schema().version,
                description=t.mcp_schema().description,
                input_schema=t.mcp_schema().input_schema,
                timeout_seconds=t.mcp_schema().timeout_seconds,
            )
            for t in self._tools.values()
        ]
        # a tool-less server has nothing to become ready for -- return True
        # immediately rather than timing out. callers still get the guarantee
        # that whatever tools ARE registered have transitioned to available.
        if not manifest_names:
            return True
        ready = False
        poll_interval = _get_ready_poll_interval()
        expected_count = len(manifest_names)
        while asyncio.get_event_loop().time() < deadline:
            try:
                request = DiscoveryProbeRequest(
                    agent_id=self._pod_id,
                    tool_manifest=[DiscoveryProbeToolEntry(name=m.name, version=m.version) for m in manifest_names],
                )
                request_timeout = timedelta(
                    seconds=min(
                        1.0,
                        max(deadline - asyncio.get_event_loop().time(), 0.01),
                    )
                )
                response = await self._nc.request(
                    subject=Subjects.tools_discover(),
                    message=request,
                    response_type=DiscoveryProbeResponse,
                    timeout=request_timeout,
                )
                available_count = sum(1 for tool in response.tools if tool.status == "available")
                if available_count == expected_count:
                    ready = True
                    break
            except Exception as exc:
                # intentional: readiness polling must tolerate transient NATS
                # hiccups and discovery schema drift without crashing the
                # caller. log at debug so the symptom surfaces in diagnostics
                # rather than a blanket silent swallow.
                log.debug(
                    "wait_until_ready poll iteration failed",
                    extra={
                        "extra_data": {
                            "pod_id": self._pod_id,
                            "error": str(exc),
                        }
                    },
                )
            await asyncio.sleep(poll_interval)
        return ready

    @traced()
    async def register_tool(self, tool: TearsTool) -> None:
        """register a tool and publish the updated manifest if connected.

        atomic public helper for dynamic tool-pod lifecycle (hub
        delegation manager, datasource tool pod). equivalent to calling
        :meth:`register` followed by :meth:`publish_registration` while
        holding the server's invariant that the manifest on the wire
        stays in sync with the in-memory registry. safe to call before
        :meth:`serve` has connected NATS: in that case the tool is
        still registered and the publish step is skipped (no-op) so
        the initial registration manifest published by :meth:`serve`
        will include it. safe to call multiple times with the same
        tool; duplicate ``name@version`` keys overwrite.

        The tool's ``platform.namespaces`` row is materialized by the
        HUB-side ``ToolNamespaceEmitter`` listening on
        ``{ns}.tools.register`` -- the manifest publish above is the
        canonical handoff. The hub owns this write because it has
        direct access to ``platform.*`` tables; the previous
        agent-side direct write through ``NamespaceCollection.save_entity``
        on the L3 NATS proxy (namespace-task-01 phase 2 /
        three-tier-task-01 phase F) hit
        ``relation "namespaces" does not exist`` because the broker
        resolves the agent's default namespace to the per-agent
        schema (``agent_<hex>``), which has no ``namespaces`` table.
        The agent now performs no direct namespace write; the manifest
        publish is the only path.

        :param tool: TearsTool instance to register
        :ptype tool: TearsTool
        """
        self.register(tool)
        if self._nc is not None:
            await self.publish_registration()

    @traced()
    async def deregister_tool(self, tool_name: str) -> bool:
        """remove all versions of a tool and publish the updated manifest.

        atomic public helper for dynamic tool-pod lifecycle (hub
        delegation manager deregistering an agent, datasource tool pod
        deregistering a data source). matches on ``mcp_name`` prefix
        of the internal ``name@version`` key, removes every matching
        entry, then publishes the reduced manifest if connected.
        returns ``True`` when at least one entry was removed so callers
        can distinguish a no-op deregister from a real one without
        silently swallowing an invariant break (e.g. "I thought that
        tool was registered but the key was missing"). safe to call
        before :meth:`serve` has connected NATS: the removal still
        happens and the publish step is skipped.

        namespace-task-01 phase 2 / three-tier-task-01 phase F: on
        successful removal, every paired tool ``platform.namespaces``
        row for that family is deleted via
        :meth:`NamespaceCollection.delete` so no stale namespace
        stays in play after the tool leaves. deletion failures raise
        so callers see the wiring gap.

        :param tool_name: namespaced ``mcp_name`` (without the
            ``@version`` suffix) identifying the family of tool
            registrations to remove
        :ptype tool_name: str
        :return: true when one or more registrations were removed
        :rtype: bool
        :raises Exception: when namespace row deletion fails
        """
        removed_keys = [key for key in self._tools if key.startswith(f"{tool_name}@")]
        removed = self.unregister(tool_name)
        if removed and self._nc is not None:
            await self.publish_registration()
        if removed:
            await self._delete_tool_namespace(tool_name, removed_keys)
        return removed

    async def _emit_tool_namespace(self, tool: TearsTool) -> None:
        """upsert ``platform.namespaces`` row for a registered tool.

        every registered tool materializes as a ``tool``-type namespace
        so the unified rbac evaluator has a first-class id to evaluate
        against in the Registry proxy hot path. row identity follows
        the same naming convention as workspace namespace rows
        (``<type>:<id>``): here the id is the tool's mcp_name plus
        version, matching the dispatch key the Registry uses.

        platform-built-in tools have ``agent_id=None`` and
        ``customer_id=None``; the row lands with both owner columns
        NULL so an admin grant can reach every customer. agent-spun
        tools carry both, scoping the grant surface to the owning
        customer.

        writes through
        :meth:`NamespaceCollection.save_entity` (three-tier-task-01
        phase F): the Collection rides the agent's main NATS-proxy
        pool so the broker admits the write under the caller's own
        ``agent.<hex>`` namespace, and the paired uuid5 id keyed on
        ``(mcp_name, version, agent_id_hex)`` lets concurrent
        registrations converge via ``ON CONFLICT (id) DO UPDATE``.

        :param tool: registered :class:`TearsTool`
        :ptype tool: TearsTool
        :return: nothing
        :rtype: None
        :raises Exception: when the Collection rejects the upsert;
            raised unchanged so callers see the wiring gap
        """
        if self._namespace_collection is None:
            return
        schema = tool.mcp_schema()
        name = tool_namespace_name(schema.name, schema.version)
        now = datetime.now(UTC)
        namespace_id = tool_namespace_id(
            schema.name,
            schema.version,
            self._agent_id,
        )
        # natural-identity metadata: the canonical ``name`` column is
        # sanitized (``tools.<sanitized-mcp>.<sanitized-version>``);
        # operators write yaml ``access.tools`` patterns in the
        # pre-sanitized form (``example.admin.*``) and downstream
        # consumers (registry authorizer lookup, hub access
        # materializer) match patterns against this metadata. keeping
        # ``mcp_name`` / ``mcp_version`` on the row decouples the
        # rbac surface from the sanitization rules and makes
        # platform-wide pattern matching uniform across rows emitted
        # by every code path (this server-side path, the hub-side
        # ``ToolNamespaceEmitter``).
        entity = self._namespace_collection.entity_class(
            {
                "namespace_id": namespace_id,
                "name": name,
                "namespace_type": "tool",
                "owner_agent_id": self._agent_id,
                "customer_id": self._customer_id,
                "schema_name": None,
                "metadata": {
                    "mcp_name": schema.name,
                    "mcp_version": schema.version,
                },
                "tool_eligible": bool(getattr(tool, "tool_eligible", True)),
                "skill_eligible": bool(getattr(tool, "skill_eligible", False)),
                "date_created": now,
                "date_updated": now,
            },
            is_new=True,
            collection=self._namespace_collection,
        )
        await self._namespace_collection.save_entity(entity)
        log.info(
            "emitted tool namespace",
            extra={
                "extra_data": {
                    "tool_name": schema.name,
                    "tool_version": schema.version,
                    "namespace_name": name,
                    "namespace_id": str(namespace_id),  # convert at border: emitted-tool-namespace log extra_data field
                    "owner_agent_id": (str(self._agent_id) if self._agent_id is not None else None),
                    "customer_id": (str(self._customer_id) if self._customer_id is not None else None),
                }
            },
        )

    async def _delete_tool_namespace(
        self,
        mcp_name: str,
        removed_keys: list[str],
    ) -> None:
        """delete ``platform.namespaces`` rows for deregistered tool versions.

        three-tier-task-01 phase F: translates the family-level
        deregister into a sequence of
        :meth:`NamespaceCollection.delete` calls, one per
        ``name@version`` key removed from the in-memory registry. the
        deterministic :func:`uuid5` derivation makes resolution a pure
        client-side computation keyed on
        ``(mcp_name, version, agent_id_hex)`` so no broker lookup is
        needed before the delete.

        :param mcp_name: tool mcp name (without version suffix)
        :ptype mcp_name: str
        :param removed_keys: ``name@version`` keys that were present
            in the in-memory registry before :meth:`unregister` ran;
            each produces one delete against the Collection
        :ptype removed_keys: list[str]
        :return: nothing
        :rtype: None
        :raises Exception: when the Collection rejects the delete
        """
        if self._namespace_collection is None:
            return
        for key in removed_keys:
            _, _, version = key.partition("@")
            namespace_id = tool_namespace_id(
                mcp_name,
                version,
                self._agent_id,
            )
            await self._namespace_collection.delete(namespace_id)
            log_namespace_id = str(namespace_id)  # convert at border: deleted-tool-namespace log extra_data field
            log.info(
                "deleted tool namespace",
                extra={
                    "extra_data": {
                        "mcp_name": mcp_name,
                        "tool_version": version,
                        "namespace_id": log_namespace_id,
                    }
                },
            )

    @traced()
    async def publish_registration(self) -> None:
        """publish registration manifest to NATS.

        sends manifest containing all registered tool definitions
        to registration subject for discovery by registry. requires
        ``serve()`` to have established the NATS connection first.
        use :meth:`register_tool` / :meth:`deregister_tool` for the
        common "mutate+publish" dynamic flows; call this directly only
        when you need to re-publish the current manifest without
        changing it (e.g. on registry recovery).

        :raises RuntimeError: if called before ``serve`` connects NATS
        """
        nc = self._nc
        if nc is None:
            raise RuntimeError("publish_registration called before NATS connected")
        # idempotent re-bind so :class:`Subjects` builders below render
        # against this server's configured namespace even when callers
        # invoke :meth:`publish_registration` directly without going
        # through :meth:`serve` (e.g. dynamic register_tool / deregister_tool
        # flows on a server constructed with an injected nats_client).
        set_default_namespace(self._namespace)
        tools_list: list[ToolManifestEntry] = []
        for tool in self._tools.values():
            schema = tool.mcp_schema()
            tool_eligible = bool(getattr(tool, "tool_eligible", True))
            skill_eligible = bool(getattr(tool, "skill_eligible", False))
            face_platform_tool = bool(getattr(tool, "face_platform_tool", True))
            face_api = bool(getattr(tool, "face_api", False))
            face_mcp = bool(getattr(tool, "face_mcp", False))
            requires_confirmation = bool(getattr(tool, "requires_confirmation", False))
            if not tool_eligible and not skill_eligible:
                # registering a tool with both flags off makes it
                # invisible to every agent surface. almost certainly
                # a configuration error; surface a structured WARNING
                # so the operator notices instead of debugging a
                # phantom missing tool later.
                log.warning(
                    "tool registered with tool_eligible=False and "
                    "skill_eligible=False -- it will never be visible "
                    "to any agent. did you forget to enable a "
                    "surface?",
                    extra={
                        "extra_data": {
                            "mcp_name": schema.name,
                            "mcp_version": schema.version,
                            "pod_id": self._pod_id,
                            "tool_class": type(tool).__name__,
                        }
                    },
                )
            entry = ToolManifestEntry(
                name=schema.name,
                version=schema.version,
                description=schema.description,
                input_schema=schema.input_schema,
                timeout_seconds=schema.timeout_seconds,
                tool_eligible=tool_eligible,
                skill_eligible=skill_eligible,
                face_platform_tool=face_platform_tool,
                face_api=face_api,
                face_mcp=face_mcp,
                requires_confirmation=requires_confirmation,
            )
            tools_list.append(entry)

        # per-key identity: mint a FRESH identity JWT for THIS manifest so the registry-layer
        # verifier sees a still-valid token even on a re-publish long after connect. the same
        # provider backs the NATS connect credential (one self-minted key, both hops). when no
        # provider is wired (dev / non-callout bus) fall back to the static bootstrap_token.
        manifest_token = self._auth_token() if self._auth_token is not None else self._bootstrap_token

        manifest = RegistrationManifest(
            pod_id=self._pod_id,
            tools=tools_list,
            bootstrap_token=manifest_token,
            owner_agent_id=self._agent_id,
            customer_id=self._customer_id,
        )

        subject = Subjects.tools_register()
        await nc.publish(subject=subject, message=manifest)
        log.debug(
            "published registration manifest",
            extra={
                "extra_data": {
                    "subject": subject.path,
                    "pod_id": self._pod_id,
                    "tools_count": len(tools_list),
                }
            },
        )

    def _load_pod_jwks(self, tool_name: str) -> dict[str, Any]:
        """fetch the cached Hub JWKS via the injected provider, converting ANY provider failure to a

        well-typed :class:`IdentityTokenError`. The provider is external (a network-backed cache); any
        failure means "cannot verify" -> a rejection reason, never an escaped exception that would
        hang the dispatch with no reply. Logs the real cause (type + MESSAGE) at the site, since the
        wrapping verify catch only sees the re-raised :class:`IdentityTokenError`.

        :param tool_name: the requested tool name, for the failure log line
        :ptype tool_name: str
        :return: the current cached JWKS document
        :rtype: dict[str, Any]
        :raises IdentityTokenError: when the provider call fails
        """
        assert self._jwks_provider is not None  # guarded by the caller
        try:
            return self._jwks_provider()
        except Exception as exc:
            log.warning(
                "JWKS provider failed during pod identity verification",
                extra={"extra_data": {"reason": type(exc).__name__, "detail": str(exc), "tool_name": tool_name}},
            )
            raise IdentityTokenError(f"JWKS provider unavailable ({type(exc).__name__})") from exc

    async def _verify_token_reactively(self, token: str, *, tool_name: str, refreshed: list[bool]) -> IdentityClaims:
        """verify a Hub token against the cached JWKS; on a kid-not-in-cache miss, refresh once + retry.

        Mirrors the registry proxy's reactive self-heal: :func:`verify_identity_token` raises the
        distinct :class:`IdentityKeyNotFoundError` when the cached JWKS holds no key for the token's
        ``kid`` (a Hub re-key, or a stale cache after a Hub pod move). That -- and ONLY that -- is
        recoverable, so this triggers one immediate :attr:`_jwks_refresh` and re-verifies against the
        refreshed cache. An expired / bad-signature / malformed token raises the BASE
        :class:`IdentityTokenError`, which is NOT caught here, so it never provokes a Hub fetch. The
        refresh fires at most ONCE per verify-path call (``refreshed`` is shared across the handshake
        + user-assertion verifications) and :meth:`refresh_now` is itself debounced + rate-limited, so
        a flood of bad tokens cannot stampede the Hub.

        :param token: the compact-JWS identity token to verify
        :ptype token: str
        :param tool_name: the requested tool name, for any provider-failure log line
        :ptype tool_name: str
        :param refreshed: a single-element mutable flag, shared across this call's verifications, so
            the reactive refresh fires at most once even if both tokens miss the cache
        :ptype refreshed: list[bool]
        :return: the verified identity claims
        :rtype: IdentityClaims
        :raises IdentityTokenError: when the token cannot be verified (after the at-most-one refresh)
        """
        try:
            return verify_identity_token(
                token,
                jwks=self._load_pod_jwks(tool_name),
                issuer=_IDENTITY_ISSUER,
                leeway_seconds=_IDENTITY_LEEWAY_SECONDS,
            )
        except IdentityKeyNotFoundError:
            if self._jwks_refresh is None or refreshed[0]:
                raise  # no reactive trigger wired, or already refreshed once this call -> reject
            refreshed[0] = True
            await self._jwks_refresh()
            return verify_identity_token(
                token,
                jwks=self._load_pod_jwks(tool_name),
                issuer=_IDENTITY_ISSUER,
                leeway_seconds=_IDENTITY_LEEWAY_SECONDS,
            )

    @traced(record_args=True)
    async def _verify_identity(self, request: CallRequest) -> tuple[CallRequest, str | None]:
        """re-verify the Hub identity token and RE-STAMP the verified identity (defense in depth).

        The registry proxy already verifies + re-stamps identity, but anything that can publish on
        the pod's internal subject would otherwise reach :meth:`handle_call` with the proxy + RBAC
        never consulted. So the pod independently verifies the Hub-issued identity token and, on
        success, OVERWRITES the call context's identity (``agent_id`` = ``token.sub``, plus
        ``user_id`` / ``customer_id``) with the cryptographically verified values. Whatever the
        inbound envelope claimed -- a matching identity, a forged one re-pointed at a captured
        token, or an absent one stripped to skip a comparison -- is discarded; the tool always
        runs (and audits) under the authenticated identity, never the self-asserted envelope. This
        mirrors the proxy's re-stamp so a direct publisher cannot run a tool under an identity the
        Hub never signed.

        The handshake identity token is one-per-pod and user-LESS, so a user-driven turn carries the
        per-turn VERIFIED user_id as a SECOND, cnf-LESS user-assertion (``context.user_identity_token``).
        This method MIRRORS the registry proxy's user-assertion gate: when present it is verified
        against the SAME issuer/JWKS and BOUND to the handshake token (``sub`` + ``customer_id`` must
        match) AND to the conversation (the assertion's ``conversation_id`` must equal the call's, so
        a captured assertion cannot be replayed into a DIFFERENT conversation), then re-stamps
        ``user_id``; without it the defense-in-depth re-stamp would null the proxy-verified user_id
        (losing audit actor attribution + the per-user context manager).

        Verification is UNCONDITIONAL and fail-closed: verify, on success re-stamp the verified
        identity, on ANY failure REJECT the call (return a rejection reason). there is no off/warn
        passthrough -- a call the pod cannot authenticate never runs on the unverified envelope.

        :param request: the parsed inbound call request
        :ptype request: CallRequest
        :return: ``(request, reason)`` where ``request`` is the re-stamped request on verify
            success (else the original) and ``reason`` is ``None`` when the call may proceed or a
            rejection-reason string when the call MUST be rejected without dispatching
        :rtype: tuple[CallRequest, str | None]
        """
        context = request.context
        # shared across the handshake + user-assertion verifications so the reactive Hub refresh (on
        # a kid-not-in-cache miss) fires at most ONCE per dispatch, not once per token.
        refreshed = [False]
        try:
            if context is None:
                raise IdentityTokenError("inbound call has no context")
            token = context.identity_token
            if token is None:
                raise IdentityTokenError("inbound call context has no identity token")
            if self._jwks_provider is None:
                raise IdentityTokenError("no JWKS provider configured for pod identity verification")
            claims = await self._verify_token_reactively(token, tool_name=request.tool_name, refreshed=refreshed)
            # OVERWRITE the envelope's claimed identity with the verified token. a captured token
            # re-pointed at a forged agent / customer, or stripped of its identity to skip a
            # comparison, runs under the token's TRUE identity -- never the self-asserted one.
            # these UUID conversions live INSIDE the try so a malformed-but-signed non-UUID claim
            # fails closed (rejects) rather than escaping as an uncaught ValueError. user_id DEFAULTS
            # to the handshake token's: ``None`` for a per-pod agent handshake token (it CANNOT carry
            # the per-turn user); the bound user-assertion below may override it.
            agent_id_value = UUID(claims.sub)
            customer_id_value = UUID(claims.customer_id)
            user_id_value: UUID | None = UUID(claims.user_id) if claims.user_id is not None else None
        except (IdentityTokenError, ValueError, KeyError, TypeError) as exc:
            reason = type(exc).__name__
            # log the exception MESSAGE too (the structural failure reason -- "no JWKS key matches the
            # token kid" vs "token expired" vs "token absent"), so a stale-JWKS failure is
            # distinguishable from an expired-token failure in production. str(exc) is never token or
            # key material (IdentityTokenError carries only the structural reason).
            extra = {"extra_data": {"reason": reason, "detail": str(exc), "tool_name": request.tool_name}}
            log.warning("pod identity verification failed; rejecting call", extra=extra)
            return request, f"identity verification failed ({reason})"

        # the handshake token verified above, so ``context`` is non-None (the try raised + returned
        # otherwise). re-narrow for the type checker.
        assert context is not None

        # MIRROR THE PROXY's user-assertion gate (registry/proxy.py ``_verify_identity``): a
        # user-driven turn's tool call ALSO carries a Hub-minted, cnf-LESS user-assertion
        # (``context.user_identity_token``) holding the per-turn VERIFIED user_id. the handshake
        # token is one-per-pod and user-LESS, so without this the pod's defense-in-depth re-stamp
        # would clobber the proxy-verified user_id back to ``None`` -- losing audit actor
        # attribution and breaking the per-user ToolContextManager. verify it against the SAME
        # issuer/JWKS and BIND it to the handshake token (``sub`` + ``customer_id`` MUST match) so a
        # user-assertion minted for agent A (customer X) cannot be replayed under agent B (or
        # customer Y). on ANY failure the call is rejected fail-closed (mirroring the proxy's
        # TOOL_USER_IDENTITY_UNVERIFIED). an empty string is treated as ABSENT (the user_id stays
        # the handshake token's) -- a caller that builds the envelope without a user-assertion must
        # never trip a fail-closed deny on the empty value.
        user_assertion = context.user_identity_token
        if user_assertion:
            try:
                user_claims = await self._verify_token_reactively(
                    user_assertion, tool_name=request.tool_name, refreshed=refreshed
                )
                if user_claims.sub != claims.sub or user_claims.customer_id != claims.customer_id:
                    raise IdentityTokenError(
                        "user-assertion not bound to the handshake identity (sub/customer mismatch)"
                    )
                if user_claims.user_id is None:
                    raise IdentityTokenError("user-assertion carries no user_id")
                # CONVERSATION-BINDING (MIRRORS the proxy): the assertion must carry the
                # conversation_id it was minted for, and it must equal this call's -- so a captured
                # user-assertion cannot be replayed into a DIFFERENT conversation. a user-driven turn
                # always mints with a conversation_id, so an assertion lacking one is a denial, never
                # a skippable check; a mismatch (or a call carrying no conversation_id while the
                # assertion carries one) is the cross-conversation replay this gate closes.
                # ``context.conversation_id`` is a UUID; stringify to compare the wire-string claim.
                if user_claims.conversation_id is None:
                    raise IdentityTokenError("user-assertion carries no conversation_id")
                if context.conversation_id is None or str(context.conversation_id) != user_claims.conversation_id:
                    raise IdentityTokenError(
                        "user-assertion conversation_id does not match the call (cross-conversation replay)"
                    )
                user_id_value = UUID(user_claims.user_id)
            except (IdentityTokenError, ValueError, KeyError, TypeError) as exc:
                reason = type(exc).__name__
                # the structural failure reason (binding mismatch vs cross-conversation replay vs
                # expired/absent assertion), never token or key material.
                extra = {"extra_data": {"reason": reason, "detail": str(exc), "tool_name": request.tool_name}}
                log.warning("pod user-assertion verification failed; rejecting call", extra=extra)
                return request, f"user-assertion verification failed ({reason})"

        verified_context = context.model_copy(
            update={
                "agent_id": agent_id_value,
                "user_id": user_id_value,
                "customer_id": customer_id_value,
            }
        )
        return request.model_copy(update={"context": verified_context}), None

    async def _verify_proxy_assertion(self, request: CallRequest) -> str | None:
        """verify the registry proxy's body-bound assertion (the pod's PRIMARY identity gate).

        The proxy signs an assertion binding the verified caller identity + the call body + a
        single-use nonce + this pod; the pod verifies it against the Hub JWKS (which carries the
        proxy's public key), so a publisher straight to the internal subject -- without a valid
        proxy assertion for THIS body -- is rejected. Verification is UNCONDITIONAL and fail-closed:
        verify, on ANY failure REJECT the call. The replay guard is MANDATORY here (it must be
        provisioned by serve() or injected) -- a guardless pod fails closed rather than silently
        skipping single-use enforcement, mirroring the registry proxy's required pop replay guard.

        :param request: the parsed inbound call request
        :ptype request: CallRequest
        :return: ``None`` when the call may proceed; a rejection-reason string when it MUST be
            rejected
        :rtype: str | None
        """
        reason: str | None = None
        try:
            assertion = request.proxy_assertion
            if assertion is None:
                raise IdentityTokenError("inbound call carries no proxy assertion")
            if self._jwks_provider is None:
                raise IdentityTokenError("no JWKS provider for proxy assertion verification")
            if self._assertion_replay_guard is None:
                # fail closed: without a replay guard a captured assertion could be replayed verbatim
                # within its accept window. serve() always provisions one; a guardless pod must not
                # silently drop single-use enforcement.
                raise IdentityTokenError("proxy assertion verification requires a replay guard")
            context = request.context
            correlation_id = (
                str(context.correlation_id) if context is not None and context.correlation_id is not None else None
            )
            body_hash = canonical_call_hash(request.tool_name, request.arguments, correlation_id)
            claims = verify_proxy_assertion(
                assertion,
                jwks=self._jwks_provider(),
                expected_pod_id=self._pod_id,
                body_hash=body_hash,
            )
            if not await self._assertion_replay_guard.record_unique(claims.jti):
                raise IdentityTokenError("proxy assertion nonce replay")
        except (IdentityTokenError, ValueError) as exc:
            kind = type(exc).__name__
            # the structural failure reason (absent assertion, kid miss, spliced body, replayed
            # nonce), never token or key material.
            extra = {"extra_data": {"reason": kind, "detail": str(exc), "tool_name": request.tool_name}}
            log.warning("pod proxy-assertion verification failed; rejecting", extra=extra)
            reason = f"proxy assertion verification failed ({kind})"
        return reason

    async def handle_call(self, msg: IncomingMessage) -> None:
        """public NATS-subject handler for incoming tool call request.

        bound by :meth:`serve` as the ``cb`` callback on
        ``{namespace}.tools.internal.{pod_id}``. tests exercise this
        surface directly; the name + single-``msg`` shape are part of
        the stability contract.

        parses call request, dispatches to matching tool, and sends
        response back via :meth:`NatsClient.publish_reply` against
        ``msg.reply_subject``. the inbound :class:`CallContext` is
        echoed verbatim on the :class:`CallResponse` so the response
        carries identity in the same shape as the request. binds the
        canonical logging context tags (``cid``/``conv``/``user``/
        ``agent``/``customer``) from the :class:`CallContext` for the
        duration of the dispatch so every log line in this handler and
        its callees renders with those tags.

        audit-task-01 (AUD-03): every dispatch -- including malformed
        requests, unknown-tool rejections, and raising tools -- emits a
        baseline ``tool.call`` :class:`AuditEvent` via
        :func:`publish_audit` on ``{namespace}.audit.tool.call``. the
        baseline event carries the CallContext identity axes, the
        tool name / version / duration, and the dispatch ``outcome``
        (``success`` / ``failure`` / ``error``). per-tool domain events
        (``workspace.fs_write`` etc.) are additive: tools still publish
        their own rich events to their dotted event_type; the baseline
        gives admin queries a uniform ``tool.call`` row for every
        dispatch regardless of whether the tool bothered to emit its
        own detail event.

        :param msg: incoming wrapper envelope carrying the call request
        :ptype msg: IncomingMessage
        """
        # bracket the whole dispatch in the in-flight gauge: increment on entry,
        # decrement on exit even when dispatch raises (try/finally inside
        # ``track``), so KEDA's prometheus scaler reads the true concurrent-call
        # count and a failed call never strands the counter above baseline.
        with self._inflight_gauge.track():
            await self._dispatch_incoming_call(msg)

    async def _dispatch_incoming_call(self, msg: IncomingMessage) -> None:
        """dispatch one in-flight-tracked tool call (body of :meth:`handle_call`).

        split out of :meth:`handle_call` so the public NATS callback can
        bracket the dispatch in the in-flight-requests gauge without
        re-indenting the whole body. the full handler contract (identity
        verification, proxy-assertion check, baseline audit) is documented
        on :meth:`handle_call`.

        :param msg: incoming wrapper envelope carrying the call request
        :ptype msg: IncomingMessage
        :return: nothing
        :rtype: None
        """
        start_monotonic = time.monotonic()
        request: CallRequest | None = None
        tool_key: str = ""
        tool_name: str = ""
        tool_version: str = ""
        outcome: str = "success"
        failure_reason: str | None = None

        try:
            request = CallRequest.model_validate_json(msg.data)
        except Exception as exc:
            error_response = CallResponse(
                success=False,
                content="",
                error=f"malformed call request: {exc}",
            )
            await self._respond(msg, error_response)
            duration_ms = (time.monotonic() - start_monotonic) * 1000.0
            await self._publish_baseline_audit(
                request=None,
                tool_name="",
                tool_version="",
                outcome="failure",
                duration_ms=duration_ms,
                failure_reason=f"malformed call request: {exc}",
            )
            return

        bind_log_context(request.context)
        try:
            # log-border stringification of the correlation id lifted
            # off the inbound context; the response itself echoes the
            # whole context (one shape in both directions), this
            # variable exists only to tag log records that the
            # set_context binding does not already cover.
            correlation_id_log = (
                str(request.context.correlation_id)
                if request.context is not None and request.context.correlation_id is not None
                else ""
            )

            tool_name = request.tool_name
            tool_version = request.tool_version
            tool_key = f"{tool_name}@{tool_version}"

            request, identity_rejection = await self._verify_identity(request)
            if identity_rejection is not None:
                error_response = CallResponse(
                    success=False,
                    content="",
                    error=identity_rejection,
                    context=request.context,
                )
                await self._respond(msg, error_response)
                log.warning(
                    "pod rejected call: identity unverified",
                    extra={
                        "extra_data": {
                            "reason": identity_rejection,
                            "tool_key": tool_key,
                            "correlation_id": correlation_id_log,
                        }
                    },
                )
                outcome = "failure"
                failure_reason = identity_rejection
                return
            # the verified identity now rides the re-stamped request; re-bind the log tags so this
            # dispatch's log lines + the baseline audit attribute to the authenticated identity, not
            # the inbound envelope's claim. a no-op on the already-matching legit path; corrects
            # attribution when the pod overrode a forged or absent identity.
            bind_log_context(request.context)

            assertion_rejection = await self._verify_proxy_assertion(request)
            if assertion_rejection is not None:
                error_response = CallResponse(
                    success=False,
                    content="",
                    error=assertion_rejection,
                    context=request.context,
                )
                await self._respond(msg, error_response)
                log.warning(
                    "pod rejected call: proxy assertion unverified",
                    extra={
                        "extra_data": {
                            "reason": assertion_rejection,
                            "tool_key": tool_key,
                            "correlation_id": correlation_id_log,
                        }
                    },
                )
                outcome = "failure"
                failure_reason = assertion_rejection
                return

            tool = self._tools.get(tool_key)

            if tool is None:
                error_response = CallResponse(
                    success=False,
                    content="",
                    error=f"unknown tool: {tool_key}",
                    context=request.context,
                )
                await self._respond(msg, error_response)
                log.warning(
                    "unknown tool requested",
                    extra={
                        "extra_data": {
                            "tool_key": tool_key,
                            "correlation_id": correlation_id_log,
                        }
                    },
                )
                outcome = "failure"
                failure_reason = f"unknown tool: {tool_key}"
                return

            try:
                scope = await self._build_call_scope(request)
                async with enter_call_scope(scope):
                    tool_result = await tool.run(**request.arguments)
                response = CallResponse(
                    success=tool_result.success,
                    content=tool_result.content,
                    metadata=tool_result.metadata,
                    error=tool_result.error,
                    context=request.context,
                )
                if not tool_result.success:
                    outcome = "failure"
                    failure_reason = tool_result.error
            except Exception as exc:
                log.error(
                    "tool execution failed",
                    extra={
                        "extra_data": {
                            "tool_key": tool_key,
                            "correlation_id": correlation_id_log,
                            "error": str(exc),
                        }
                    },
                )
                response = CallResponse(
                    success=False,
                    content="",
                    error=f"tool execution failed: {exc}",
                    context=request.context,
                )
                outcome = "error"
                failure_reason = f"tool execution failed: {exc}"

            await self._respond(msg, response)
        finally:
            duration_ms = (time.monotonic() - start_monotonic) * 1000.0
            await self._publish_baseline_audit(
                request=request,
                tool_name=tool_name,
                tool_version=tool_version,
                outcome=outcome,
                duration_ms=duration_ms,
                failure_reason=failure_reason,
            )
            clear_context()

    async def _respond(self, msg: IncomingMessage, response: BaseModel) -> None:
        """publish ``response`` to the inbound message's reply subject.

        equivalent to the pre-migration ``msg.respond(...)`` shape; when
        the inbound message did not carry a reply subject (pure
        fire-and-forget; should not happen in production but possible
        in synthetic test envelopes), the call becomes a no-op so the
        handler chain stays robust.

        :param msg: inbound wrapper envelope
        :ptype msg: IncomingMessage
        :param response: typed response to publish
        :ptype response: BaseModel
        :return: nothing
        :rtype: None
        """
        if msg.reply_subject is None or self._nc is None:
            return
        await self._nc.publish_reply(reply_subject=msg.reply_subject, message=response)

    async def _publish_baseline_audit(
        self,
        *,
        request: CallRequest | None,
        tool_name: str,
        tool_version: str,
        outcome: str,
        duration_ms: float,
        failure_reason: str | None,
    ) -> None:
        """publish the baseline ``tool.call`` audit event; fire-and-forget.

        emission is gated on ``self._namespace`` and ``self._nc`` both
        being set (production wiring); bootstrap / test scenarios that
        omit either leave the pipeline a no-op. identity axes come
        from ``request.context`` when a :class:`CallContext` was
        parsed; malformed-request and no-context paths emit with
        ``None`` identity. resource axes stay ``None`` at the baseline
        layer -- the tool server does not know which namespace the
        tool will touch; tools that want a resource-tagged event emit
        their own additive event during dispatch.

        :param request: parsed call request, or ``None`` when the wire
            payload failed to decode
        :ptype request: CallRequest | None
        :param tool_name: requested tool name (empty string on
            malformed / unparsable request)
        :ptype tool_name: str
        :param tool_version: requested tool version (empty string on
            malformed / unparsable request)
        :ptype tool_version: str
        :param outcome: one of ``success`` / ``failure`` / ``error``
        :ptype outcome: str
        :param duration_ms: wall-clock elapsed milliseconds between
            handler entry and the audit publish
        :ptype duration_ms: float
        :param failure_reason: human-readable reason string when
            ``outcome != "success"``; ``None`` otherwise
        :ptype failure_reason: str | None
        :return: nothing
        :rtype: None
        """
        if self._namespace is None or self._nc is None:
            # bootstrap / test scenario: nothing to publish on.
            return
        context = request.context if request is not None else None
        details: dict[str, Any] = {
            "tool_name": tool_name,
            "tool_version": tool_version,
            "duration_ms": duration_ms,
        }
        if failure_reason is not None:
            details["failure_reason"] = failure_reason
        correlation_id: UUID
        if context is not None and context.correlation_id is not None:
            correlation_id = context.correlation_id
        else:
            # malformed request / no-context dispatch: mint a fresh
            # correlation id so the ``(correlation_id, event_type)``
            # unique index still distinguishes concurrent baseline
            # rows for otherwise-identical tool names.
            correlation_id = uuid7()
        event = AuditEvent(
            id=uuid7(),
            timestamp=datetime.now(UTC),
            event_type="tool.call",
            actor_user_id=context.user_id if context is not None else None,
            calling_agent_id=context.agent_id if context is not None else None,
            owner_agent_id=self._agent_id,
            customer_id=context.customer_id if context is not None else None,
            resource_namespace_id=None,
            resource_namespace_type=None,
            action="call",
            outcome=outcome,
            correlation_id=correlation_id,
            conversation_id=(context.conversation_id if context is not None else None),
            details=details,
        )
        try:
            await publish_audit(
                event,
                nats_client=self._nc,
                namespace=self._namespace,
            )
        # NOSILENT: audit publish is fire-and-forget; publish_audit
        # already swallows inside, but the belt-and-braces guard here
        # protects against any programmer-error regression (TypeError,
        # AttributeError) that could otherwise taint a successful
        # tool return.
        except Exception as exc:
            log.warning(
                "tool server baseline audit publish failed",
                extra={
                    "extra_data": {
                        "tool_key": f"{tool_name}@{tool_version}",
                        "error": str(exc),
                    },
                },
            )

    async def _build_call_scope(
        self,
        request: CallRequest,
    ) -> ToolCallScope:
        """construct per-call scope from envelope :class:`CallContext`.

        reads identity dimensions off ``request.context`` (which arrives
        as UUIDs already coerced by pydantic at the wire boundary) and
        resolves a :class:`ToolContextManager` by calling the server's
        ``context_factory`` when both ``conversation_id`` and
        ``user_id`` are present. callers that do not need the context
        (stateless tools) can safely omit ``context`` entirely: the
        resulting scope carries ``context_manager=None`` and any tool
        that requires it raises at first use.

        factory exceptions propagate to :meth:`handle_call`'s except
        block so the call is surfaced as a failed tool result rather
        than a silent no-context handoff.

        :param request: parsed call request
        :ptype request: CallRequest
        :return: populated :class:`ToolCallScope`
        :rtype: ToolCallScope
        """
        context = request.context if request.context is not None else CallContext()
        context_manager: ToolContextManager | None = None
        log.debug(
            "building call scope",
            extra={
                "extra_data": {
                    "factory_present": self._context_factory is not None,
                    "conv_present": context.conversation_id is not None,
                    "user_present": context.user_id is not None,
                }
            },
        )
        if self._context_factory is not None and context.conversation_id is not None and context.user_id is not None:
            context_manager = await self._context_factory(
                context.conversation_id,
                context.user_id,
            )
        return ToolCallScope(
            context=context,
            context_manager=context_manager,
            object_store=self._object_store,
            object_resolver=self._object_resolver,
            engagement_resolver=self._engagement_resolver,
        )

    async def _heartbeat_loop(self) -> None:
        """publish periodic heartbeat and re-registration until shutdown.

        publishes heartbeat containing pod_id, timestamp, and tools_count
        to heartbeat subject at configured interval. re-publishes full
        registration manifest alongside each heartbeat so the registry
        recovers automatically if it restarts. requires ``serve()`` to
        have connected NATS first.

        :raises RuntimeError: if called before ``serve`` connects NATS
        """
        nc = self._nc
        if nc is None:
            raise RuntimeError("_heartbeat_loop started before NATS connected")
        subject = Subjects.tools_heartbeat(self._pod_id)
        consecutive_unhealthy = 0
        while self._running:
            heartbeat = HeartbeatMessage(
                pod_id=self._pod_id,
                timestamp=datetime.now(UTC).isoformat(),
                tools_count=len(self._tools),
            )
            try:
                await nc.publish(subject=subject, message=heartbeat)
            except Exception as exc:
                log.warning(
                    "heartbeat publish failed",
                    extra={"extra_data": {"error": str(exc)}},
                )
            try:
                await self.publish_registration()
            except Exception as exc:
                log.warning(
                    "periodic re-registration failed",
                    extra={"extra_data": {"error": str(exc)}},
                )
            # Liveness supervisor (the no-k8s net; the /healthz probe is the k8s net).
            # Forever-reconnect rides out any transient drop (is_healthy stays True), but a
            # TERMINAL close (user-JWT expiry -ERR) or a persistent auth/overflow wedge never
            # self-recovers -- and the old loop swallowed the failed publishes forever, wedging
            # the pod "Running" with a dead data plane. Instead, after a short sustained-unhealthy
            # streak, crash the process so the orchestrator recycles it (k8s Deployment restart /
            # docker restart policy) -- cattle, not a pet. os._exit because a SystemExit raised in
            # a background task is swallowed by asyncio task bookkeeping and would not terminate
            # the process.
            if self.is_healthy:
                consecutive_unhealthy = 0
            else:
                consecutive_unhealthy += 1
                if consecutive_unhealthy >= _UNHEALTHY_EXIT_THRESHOLD:
                    log.error(
                        "NATS data plane unrecoverable; crashing so the pod recycles (cattle)",
                        extra={
                            "extra_data": {
                                "pod_id": self._pod_id,
                                "consecutive_unhealthy": consecutive_unhealthy,
                            }
                        },
                    )
                    os._exit(1)
            await asyncio.sleep(self._heartbeat_interval)

    @traced()
    async def shutdown(self) -> None:
        """gracefully shut down tool server.

        stops the heartbeat loop and drains NATS subscriptions the
        server owns. the NATS connection itself is closed ONLY when
        the server opened it (i.e. was constructed with ``nats_url``).
        when the connection was injected via ``nats_client`` the
        caller owns the lifecycle and shutdown leaves the connection
        open so other subscribers (graph handler, heartbeat loop on
        the bootstrap side) continue to work until the caller closes
        the connection itself.
        """
        log.info(
            "shutting down tool server",
            extra={"extra_data": {"pod_id": self._pod_id}},
        )
        self._running = False

        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._owned_jwks_provider is not None:
            await self._owned_jwks_provider.stop()
            self._owned_jwks_provider = None

        if self._nc is not None and self._owns_nats_connection:
            await self._nc.shutdown()

        self._shutdown_event.set()

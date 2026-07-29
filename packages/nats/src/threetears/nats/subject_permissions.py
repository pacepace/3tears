"""Per-principal NATS subject-permission allow-lists for decentralized auth (platform-auth A).

When the NATS bus moves from an anonymous shared connection to authenticated per-principal
connections, the auth-callout responder mints each connecting principal a user JWT whose pub/sub
permissions come from THIS map. The map is the single authority for "what may a {principal} pub or
sub" and exists to make two properties true and testable:

- **exhaustive** — every subject a principal needs to bootstrap and run is present; a missing
  boot-critical subject bricks that principal the moment auth is enforced (it can't even handshake).
- **least-privilege isolation** — no principal is granted the bare ``>`` or the global ``_INBOX.>``;
  isolation is per-user allow-lists. Identity-bound subjects (``agents.internal.{agent}.{pod}``,
  ``tools.internal.{pod}``, heartbeats, the principal's reply inbox) are templated with the
  CONNECTING principal's OWN ids, so principal A cannot subscribe to principal B's inbox or
  impersonate B's identity-tailed subjects.

Request/reply uses a per-connection SCOPED inbox (``_INBOX_{principal}_{id}``) plus NATS
``allow_responses``: a responder may publish to the reply-subject of a message it actually received
without holding standing publish rights on every requester's inbox. So a responder never needs a
broad ``_INBOX.>`` publish grant.

The application pub/sub subjects below are built through the canonical :class:`Subjects` factory (the
namespace prefix is never hand-typed). Each principal additionally DECLARES the JetStream KV buckets
and streams it touches by name; the minted user JWT grants the matching ``$KV`` data subtree plus a
JetStream control-plane allow-list PINNED per declared stream (a KV bucket ``<b>`` is backed by the
stream ``KV_<b>``), so a principal can drive JS ops only against its OWN streams and is denied the
cross-tenant direct-read / destroy a bare ``$JS.API.>`` would expose (see
:func:`threetears.nats.user_jwt._js_api_grants_for_stream`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from threetears.nats.subjects import Subjects, get_default_namespace

__all__ = [
    "CROSS_PLATFORM_CACHE_INVALIDATE",
    "Principal",
    "PrincipalPermissions",
    "build_permissions",
    "inbox_prefix_for",
]

#: the ONE deliberately non-namespaced subject (every 3tears collection in every process listens on
#: it regardless of env prefix). mirrors :meth:`Subjects.cache_invalidate`.
CROSS_PLATFORM_CACHE_INVALIDATE = "threetears.cache.invalidate"


class Principal(StrEnum):
    """a connection identity class the bus authenticates and scopes permissions for."""

    AGENT_POD = "agent_pod"
    TOOL_POD = "tool_pod"
    REGISTRY = "registry"
    HUB = "hub"
    GATEWAY = "gateway"
    CHANNEL_ADAPTER = "channel_adapter"


@dataclass(frozen=True, slots=True)
class PrincipalPermissions:
    """the resolved pub/sub allow-list + JetStream resources for one connection.

    :param publish: subjects (concrete or wildcard patterns) the principal may publish to. for
        request/reply this is the request subject (a request is a publish + an inbox subscribe).
    :ptype publish: tuple[str, ...]
    :param subscribe: subjects the principal may subscribe to, including its own scoped reply inbox.
    :ptype subscribe: tuple[str, ...]
    :param allow_responses: when true the principal may reply to the reply-subject of any message it
        received, without a standing publish grant on the requester's inbox (NATS ``allow_responses``).
        set for every principal that answers requests.
    :ptype allow_responses: bool
    :param inbox_prefix: the scoped request/reply inbox prefix this principal's client must use
        (never the global ``_INBOX``); ``{inbox_prefix}.>`` is included in ``subscribe``.
    :ptype inbox_prefix: str
    :param kv_buckets: JetStream KV bucket names the principal reads/writes (granted ``$KV.{bucket}.>``
        + the matching ``$JS.API`` ops by the account config).
    :ptype kv_buckets: tuple[str, ...]
    :param streams: JetStream stream names the principal publishes to or consumes from.
    :ptype streams: tuple[str, ...]
    """

    publish: tuple[str, ...]
    subscribe: tuple[str, ...]
    allow_responses: bool
    inbox_prefix: str
    kv_buckets: tuple[str, ...] = field(default_factory=tuple)
    streams: tuple[str, ...] = field(default_factory=tuple)


def inbox_prefix_for(principal: Principal, *, conn_id: str) -> str:
    """the scoped request/reply inbox prefix for one connection.

    Each connection gets ``_INBOX_{principal}_{conn_id}`` instead of the shared global ``_INBOX``,
    so a responder replying via :attr:`PrincipalPermissions.allow_responses` cannot be sniffed by a
    different principal subscribing the global inbox tree. the principal's NATS client must be
    configured with this prefix.

    :param principal: the connection identity class
    :ptype principal: Principal
    :param conn_id: a value unique to this connection (the pod_id for pods; a connection uuid for
        infra principals)
    :ptype conn_id: str
    :return: the scoped inbox prefix
    :rtype: str
    """
    return f"_INBOX_{principal.value}_{_seg(conn_id)}"


def _seg(value: str) -> str:
    """subject-safe a single dynamic segment (mirror :func:`Subjects` sanitization: ``.`` -> ``-``)."""
    return value.replace(".", "-")


def _ns() -> str:
    """current namespace prefix (the Subjects factory reads the same source)."""
    return get_default_namespace()


def build_permissions(
    principal: Principal,
    *,
    agent_id: str | None = None,
    pod_id: str | None = None,
    conn_id: str | None = None,
    tool_namespaces: Sequence[str] | None = None,
) -> PrincipalPermissions:
    """resolve the concrete allow-list for one connecting principal.

    Pod principals (:attr:`Principal.AGENT_POD`, :attr:`Principal.TOOL_POD`) require their own
    ``pod_id`` (and ``agent_id`` for the agent) so identity-bound subjects are scoped to THEM. infra
    principals use wildcards for the dynamic peer segments (any pod / any correlation id). ``conn_id``
    defaults to ``pod_id`` for pods and must be supplied for infra principals.

    :param principal: the connection identity class
    :ptype principal: Principal
    :param agent_id: the connecting agent's id (required for :attr:`Principal.AGENT_POD`)
    :ptype agent_id: str | None
    :param pod_id: the connecting pod's id (required for pod principals)
    :ptype pod_id: str | None
    :param conn_id: a connection-unique id for the scoped inbox; defaults to ``pod_id``
    :ptype conn_id: str | None
    :param tool_namespaces: the tool namespace names this pod is authorized to serve -- the
        ``allowed_namespaces`` list on its tool-pods row, the same list
        ``RegistrationHandler._authenticate_and_filter`` filters its manifest against. each entry
        becomes one EXACT per-tool subject grant, so registration filtering and subject permissions
        read from one source of truth. omitting it grants none of them (a pod that serves no
        human-in-the-loop session needs none).
    :ptype tool_namespaces: Sequence[str] | None
    :return: the resolved permissions
    :rtype: PrincipalPermissions
    :raises ValueError: when a required id for the principal is missing
    """
    resolver = _RESOLVERS[principal]
    return resolver(agent_id=agent_id, pod_id=pod_id, conn_id=conn_id, tool_namespaces=tool_namespaces)


def _require(value: str | None, *, name: str, principal: Principal) -> str:
    """fail closed when a permission resolver is missing an id it must scope on."""
    if not value:
        raise ValueError(f"{principal.value} permissions require a non-empty {name}")
    return value


def _agent_pod(
    *,
    agent_id: str | None,
    pod_id: str | None,
    conn_id: str | None,
    tool_namespaces: Sequence[str] | None,
) -> PrincipalPermissions:
    a = _require(agent_id, name="agent_id", principal=Principal.AGENT_POD)
    p = _require(pod_id, name="pod_id", principal=Principal.AGENT_POD)
    inbox = inbox_prefix_for(Principal.AGENT_POD, conn_id=conn_id or p)
    ns = _ns()
    publish = (
        str(Subjects.agent_register()),
        str(Subjects.agent_deregister()),
        str(Subjects.agent_heartbeat(a, p)),  # own authed agent + own pod only
        str(Subjects.hub_handshake()),
        str(Subjects.hub_secrets_request()),
        str(Subjects.hub_object_commit()),  # Path-2: catalogs a produced object under its own tenant
        str(Subjects.hub_object_resolve()),  # Path-2: resolves an object id it owns to its stored key
        # HITL: on a requires_confirmation tool pause the agent records a pending-approval marker
        # with the hub (forwarding its own identity token; the hub verifies + tenant-scopes).
        str(Subjects.hub_approval_record()),
        str(Subjects.hub_jwks()),
        str(Subjects.gateway_completion()),
        str(Subjects.gateway_embedding()),
        str(Subjects.tools_discover()),
        str(Subjects.tools_call()),
        # in-process tool serving: an agent runs context-bound tools on ITS OWN ``AGENT_POD``
        # connection rather than as separate Tool Pods -- the devx ``DevInProcessStrategy`` standard
        # builtins, AND (in production) the ``ProdExternalPodsStrategy`` workspace + ``knowledge_drafts``
        # tools. so it must publish the registration + heartbeat subjects a tool pod publishes. the
        # heartbeat is scoped to the AUTHENTICATED ``agent_id`` subtree (``tools.heartbeat.{a}.>``),
        # NOT the spoofable connect-name pod id: the in-process server runs under the
        # ``{agent_id}.{instance}`` composite pod-id (``Subjects.agent_inprocess_pod_id``) so a tenant
        # can never publish a heartbeat under a peer agent's identity. EXTERNAL user-tool pods are a
        # separate ``TOOL_POD`` principal carrying single-token grants under their own ``_tool_pod`` JWT.
        str(Subjects.tools_register()),
        str(Subjects.tools_heartbeat_agent_subtree(a)),  # heartbeats only under its own authed agent subtree
        # the in-process tool server emits the baseline ``tool.call`` audit envelope on every
        # dispatch (mirrors ``_tool_pod``); audit non-repudiation is required, so the grant is
        # mandatory -- without it an agent-served tool call's actor/audit row is silently dropped.
        str(Subjects.audit_event("tool.call")),
        str(Subjects.knowledge_draft()),
        str(Subjects.workspaces_create()),
        str(Subjects.namespace_discover()),
        str(Subjects.l3_query()),
        str(Subjects.l3_batch()),
        f"{ns}.l3.tx.*",  # mirrors Subjects.l3_tx(op) over all six ops
        f"{ns}.channels.deliver.*",  # publishes finished answers (any inbound channel type)
        f"{ns}.hub.stream.{_seg(a)}.*",  # streams tokens for its own (per-request) correlation ids under its own authed agent id
        f"{ns}.agents.complete.*",  # resilience-task-07: per-turn completion signal the router awaits to ack the durable turn (keyed by correlation id)
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    subscribe = (
        f"{inbox}.>",  # scoped reply inbox
        str(Subjects.agent_internal(a, p)),  # own agent + own pod only
        str(Subjects.agent_reregister_request(a, p)),  # own authed agent + own pod only
        # in-process tool serving: receive the registry's proxied calls + reachability probe for its
        # OWN in-process tool server. scoped to the AUTHENTICATED ``agent_id`` subtree
        # (``tools.internal.{a}.>`` / ``tools.probe.{a}.>``), NOT the spoofable connect-name pod id:
        # the server subscribes ``tools.{internal,probe}.{a}.{instance}`` under the
        # ``{agent_id}.{instance}`` composite pod-id, so a tenant can NEVER be granted a subject under
        # a peer agent's identity and thus can never wiretap another agent's proxied in-process tool
        # calls. the ``{instance}`` tail keeps replicas of the same agent independently routable.
        str(Subjects.tools_internal_agent_subtree(a)),  # proxied calls to its own in-process tools
        str(Subjects.tools_probe_agent_subtree(a)),  # the registry's reachability probe for its own pods
        str(Subjects.acl_invalidate("membership")),
        str(Subjects.acl_invalidate("assignment")),
        str(Subjects.acl_invalidate("role")),
        CROSS_PLATFORM_CACHE_INVALIDATE,
        f"{ns}.gateway.stream.{_seg(a)}.*",  # token stream for its own in-flight requests under its own authed agent id
        str(Subjects.mcp_rbac_epoch()),
    )
    return PrincipalPermissions(
        publish=publish,
        subscribe=subscribe,
        allow_responses=True,  # replies to the hub's route request
        inbox_prefix=inbox,
        kv_buckets=(
            f"{ns}_agent_config",  # direct js.create_key_value in the hub; no transport prefix
            f"{ns}-collections",
            "checkpoints",
            # memory extraction throttle: the agent's MemoryExtractor uses a
            # per-conversation SET-NX-with-TTL key in this bucket to rate-limit
            # extraction. without the grant the JS API calls are denied and
            # time out, so the gate fails open (no throttling).
            f"{ns}-ratelimits",
            # in-process tool serving: the in-process tool server verifies the proxy's body-bound
            # assertion under enforce and records single-use nonces here (mirrors ``_tool_pod``). used
            # in BOTH devx (``DevInProcessStrategy`` builtins) and production
            # (``ProdExternalPodsStrategy`` workspace + ``knowledge_drafts`` tools).
            f"{ns}-proxy_assertion_nonces",
        ),
        streams=(f"{ns}_channels_deliver",),
    )


def _tool_pod(
    *,
    agent_id: str | None,
    pod_id: str | None,
    conn_id: str | None,
    tool_namespaces: Sequence[str] | None,
) -> PrincipalPermissions:
    p = _require(pod_id, name="pod_id", principal=Principal.TOOL_POD)
    inbox = inbox_prefix_for(Principal.TOOL_POD, conn_id=conn_id or p)
    ns = _ns()
    # human-in-the-loop session control plane: while a pod holds a display's claim it serves the
    # owner-routed session messages (open/complete a tab, read state, close the session) forwarded
    # to it. one EXACT family literal per authorized tool, minted by hashing the SAME
    # ``allowed_namespaces`` entries that filter the pod's registration. a coarse ``{ns}.forward.>``
    # would instead let any tool pod serve any owner-routed key in the namespace, and since the key
    # segment is a digest of an arbitrary application string there is nothing else in the subject
    # left to discriminate on. SUBSCRIBE only: the owner answers on the requester's reply inbox
    # under ``allow_responses`` and never originates a forward.
    # BOTH families per tool. a session's control plane and its display stream are owner-routed
    # on the same key, so they must derive different subjects or the queue group ``serve_owner``
    # uses would split one pod's messages between the two handlers.
    hitl_sessions = tuple(
        str(Subjects.forward_scoped_wildcard(derive(name)))
        for name in (tool_namespaces or ())
        for derive in (Subjects.hitl_forward_family, Subjects.hitl_pipe_family)
    )
    # the byte pipe itself: once an attach has been answered on the control plane above, the pod
    # streams bytes on subjects that name its OWN tool digest and its OWN pod id as exact literals.
    # only the per-attach nonce is wildcarded, and it cannot be anything else -- the pod mints it
    # per stream, long after these grants were issued. the direction segment is part of each pattern
    # rather than a ``>`` tail so PUBLISH reaches only the pod's own half (``down``) and SUBSCRIBE
    # only the caller's (``up``); one ``>`` grant would let the owner publish onto both.
    pipe_down = tuple(str(Subjects.pipe_pod_wildcard(name, p, "down")) for name in (tool_namespaces or ()))
    pipe_up = tuple(str(Subjects.pipe_pod_wildcard(name, p, "up")) for name in (tool_namespaces or ()))
    publish = (
        str(Subjects.tools_register()),
        str(Subjects.tools_heartbeat(p)),  # own pod only
        str(Subjects.tools_discover()),  # polls discovery during wait_until_ready
        str(Subjects.hub_jwks()),  # fetches the JWKS to verify proxy assertions
        str(Subjects.audit_event("tool.call")),
        # Path-2 consume: a consuming tool resolves an object id -> its stored
        # key (forwarding the invoking agent's identity token; the hub verifies
        # + tenant-scopes). NOT hub_object_commit -- commit is agent-side.
        str(Subjects.hub_object_resolve()),
        # engagement scope (consumer A of the §2 keystone): a scan tool resolves
        # its call's engagement_id -> the authorized target set (same forwarded
        # identity-token auth; the hub verifies + tenant-scopes). read-only.
        str(Subjects.hub_engagement_scope()),
        *pipe_down,  # own authorized tools' streams, own pod id, own half only
    )
    subscribe = (
        f"{inbox}.>",
        str(Subjects.tools_internal(p)),  # own pod's proxied calls only
        str(Subjects.tools_probe(p)),  # own pod's liveness probes only
        *hitl_sessions,  # own authorized tools' session control plane only
        *pipe_up,  # own authorized tools' streams, own pod id, caller's half only
    )
    return PrincipalPermissions(
        publish=publish,
        subscribe=subscribe,
        allow_responses=True,  # replies to the registry's forwarded call + probe
        inbox_prefix=inbox,
        kv_buckets=(
            f"{ns}-proxy_assertion_nonces",
            # the display claim: a pod serving a human session holds a ``KVLease`` for as long as
            # it serves. every name here is the bucket that MATERIALISES: ``kv_bucket`` takes a
            # suffix and layers the connection's ``{ns}-`` over it.
            #
            # WHAT A MISSING GRANT ACTUALLY DOES, because two earlier versions of this comment got
            # it wrong in the same way. It does NOT yield ``lease=None`` and serve the display
            # unclaimed: that value is the dependency-injection path, set by a platform passing no
            # lease at all. With a lease supplied and this grant absent, ``KVLease.acquire``
            # defers opening the bucket to first use and that open raises ``KvError`` after a
            # JetStream timeout, which nothing catches. The symptom is a hard failure on the first
            # claim rather than a silent double-serve -- and because the open is deferred, a
            # platform cannot learn at construction time that it should downgrade.
            f"{ns}-leases",
        ),
    )


def _registry(
    *,
    agent_id: str | None,
    pod_id: str | None,
    conn_id: str | None,
    tool_namespaces: Sequence[str] | None,
) -> PrincipalPermissions:
    c = _require(conn_id, name="conn_id", principal=Principal.REGISTRY)
    inbox = inbox_prefix_for(Principal.REGISTRY, conn_id=c)
    ns = _ns()
    publish = (
        # forwards calls to / probes ANY pod. the ``>`` tail (not ``.*``) spans BOTH single-token
        # Tool Pods (``tools.internal.{pod_id}``) and two-token agent in-process pods
        # (``tools.internal.{agent_id}.{instance}``); a single-token ``.*`` would silently stop
        # routing to agent in-process tool servers the moment they adopt the composite pod-id.
        str(Subjects.tools_internal_wildcard()),  # forwards calls to ANY pod (tool pod or agent in-process)
        str(Subjects.tools_probe_wildcard()),  # probes ANY pod
        str(Subjects.hub_jwks()),  # fetches the JWKS to verify identity tokens + pop
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    subscribe = (
        f"{inbox}.>",
        str(Subjects.tools_call()),  # queue-grouped: registry
        str(Subjects.tools_discover()),  # queue-grouped: registry
        str(Subjects.tools_register()),
        str(Subjects.tools_heartbeat_wildcard()),
        str(Subjects.acl_invalidate("membership")),
        str(Subjects.acl_invalidate("assignment")),
        str(Subjects.acl_invalidate("role")),
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    return PrincipalPermissions(
        publish=publish,
        subscribe=subscribe,
        allow_responses=True,  # replies to agents' tools.call / tools.discover
        inbox_prefix=inbox,
        kv_buckets=(
            # UNPREFIXED, and deliberately so: the registry opens its catalog with a direct
            # ``js.key_value(bucket=...)`` (``threetears.registry.server``) rather than through
            # ``kv_bucket``, so no namespace prefix is ever applied to it. The other two
            # conventions in this file both prefix; this one is the exception and is the reason
            # the enforcement test allows a bare name.
            "tool_catalog",
            # constructed at ``threetears.registry.server`` as
            # ``ReplayGuard(nc, bucket_name="pop_nonces", ...)`` -- a live call site in this
            # repository, not the usage example in ``ReplayGuard``'s own docstring -- and
            # ``ReplayGuard`` opens through ``kv_bucket``, so this one carries the prefix.
            f"{ns}-pop_nonces",
        ),
    )


def _hub(
    *,
    agent_id: str | None,
    pod_id: str | None,
    conn_id: str | None,
    tool_namespaces: Sequence[str] | None,
) -> PrincipalPermissions:
    # the hub is the broadest principal: trust anchor + control plane + L3 broker + router. it owns
    # the whole {ns}.hub.*, {ns}.agents.*, {ns}.l3.*, and the platform-write event streams. it is
    # still NOT granted a bare `>` -- every grant below is a named family within the namespace.
    c = _require(conn_id, name="conn_id", principal=Principal.HUB)
    inbox = inbox_prefix_for(Principal.HUB, conn_id=c)
    ns = _ns()
    publish = (
        f"{ns}.agents.route.*",  # routes user messages to any agent
        f"{ns}.agents.reregister_request.*.*",  # nudges any agent's pod to re-register ({agent_id}.{pod_id})
        str(Subjects.gateway_completion()),  # hub-side completions (e.g. summarization)
        str(Subjects.gateway_embedding()),
        str(Subjects.acl_invalidate("membership")),
        str(Subjects.acl_invalidate("assignment")),
        str(Subjects.acl_invalidate("role")),
        str(Subjects.gateway_catalog_epoch()),
        str(Subjects.mcp_rbac_epoch()),
        f"{ns}.hub.channel.installs.changed",  # notifies channel adapters of install changes
        # human-in-the-loop session control plane: the hub terminates the operator's WebSocket and
        # forwards session control to whichever pod holds that display. unlike the pod side this
        # cannot be an exact family literal -- one hub connection fronts EVERY tool, so there is no
        # per-connection list to mint from. a NATS wildcard matches a whole token, so the family
        # segment is either one exact literal or ``*``; there is no prefix form. the two-token
        # pattern does not match the UNSCOPED ``{ns}.forward.{key}`` family, which stays granted to
        # nobody. publish only: the hub calls, the pod serves.
        str(Subjects.forward_scoped_any_family()),
        # the byte pipe's caller half. the same reasoning as the forward family directly above: one
        # hub connection fronts EVERY tool and every pod, so there is no per-connection list to mint
        # exact literals from. what stays exact is the DIRECTION -- the hub publishes only on ``up``,
        # so no wildcard here reaches an owner's own half of any stream.
        str(Subjects.pipe_any_pod_wildcard("up")),
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    subscribe = (
        f"{inbox}.>",
        str(Subjects.pipe_any_pod_wildcard("down")),  # every pod's stream to it, never a pod's own half
        str(Subjects.hub_handshake()),  # mints identity tokens
        str(Subjects.hub_jwks()),  # serves the JWKS
        str(Subjects.hub_secrets_request()),
        str(Subjects.hub_user_resolve()),
        str(Subjects.hub_object_commit()),  # Path-2: responds to object catalog commits
        str(Subjects.hub_object_resolve()),  # Path-2: responds to object id -> key resolves
        str(Subjects.hub_engagement_scope()),  # engagement scope: responds to engagement_id -> targets resolves
        # HITL approval broker: record a pending marker (agent-forwarded) + resolve an operator reply
        # (channel-router-forwarded) into an authorized approve/deny verdict on the resume rail.
        str(Subjects.hub_approval_record()),
        str(Subjects.hub_approval_resolve()),
        str(Subjects.hub_channel_installs()),
        str(Subjects.namespace_discover()),
        str(Subjects.agent_register()),
        str(Subjects.agent_deregister()),
        str(Subjects.agent_heartbeat_wildcard()),  # heartbeat monitor
        str(Subjects.l3_query()),
        str(Subjects.l3_batch()),
        f"{ns}.l3.tx.*",
        f"{ns}.datasource.*.query",
        str(Subjects.tools_register()),  # materializes tool namespace rows
        str(Subjects.workspaces_create()),
        str(Subjects.knowledge_draft()),
        str(Subjects.hub_usage_track()),
        str(Subjects.audit_wildcard()),  # unified audit consumer
        str(Subjects.acl_invalidate("membership")),
        str(Subjects.acl_invalidate("assignment")),
        str(Subjects.acl_invalidate("role")),
        f"{ns}.hub.stream.*.*",  # subscribes to agent token streams it dispatched ({agent_id}.{correlation_id})
        str(Subjects.gateway_catalog_epoch()),
        str(Subjects.mcp_rbac_epoch()),
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    return PrincipalPermissions(
        publish=publish,
        subscribe=subscribe,
        allow_responses=True,
        inbox_prefix=inbox,
        kv_buckets=(
            # THE UNDERSCORE SPELLING IS NOT NECESSARILY A TYPO, and that is the whole reason
            # these are here. A bucket created by a DIRECT ``js.create_key_value(bucket=...)``
            # never receives the ``{ns}-`` prefix ``kv_bucket`` layers on, so a component that
            # builds its own ``f"{namespace}_thing"`` name owns that name verbatim.
            #
            # PROVEN for ``agent_config`` only: the hub's ``AgentConfigKV`` creates it that way
            # and its own docstring calls it platform-historical, and the agent router reads it
            # back. Every OTHER underscore-spelled entry below is kept on PRECAUTION rather than
            # on evidence: no opener was found for it in either repository, but "not found" is not
            # "absent", and the two failure directions are not symmetric. A grant naming a bucket
            # nothing opens costs permission surface; removing one that something does open costs
            # a JetStream operation that blocks to its deadline instead of raising, which reads as
            # an unreachable broker. Removing them on absence-of-evidence is the mistake that was
            # already made here once.
            #
            # Do not "normalise" any of these to ``{ns}-``: that renames a live bucket.
            f"{ns}_agent_config",
            f"{ns}_agent_sessions",
            f"{ns}_revoked_tokens",
            f"{ns}_login_lockouts",
            f"{ns}_rate_limits",
            # ADDED rather than corrected, and the distinction matters. The hub's own router rate
            # limiter creates ``f"{namespace}_ratelimit"`` -- singular, no trailing ``s`` -- with a
            # direct ``js.create_key_value``, and no grant named it, so the limiter's first
            # operation blocked to its deadline instead of raising. ``{ns}_rate_limits`` above is
            # NOT removed in the same breath: nothing proves it dead, and this file has already
            # been burned once by treating "no opener found" as "no opener".
            f"{ns}_ratelimit",
            f"{ns}-collections",
        ),
        streams=(f"{ns}_channels_deliver",),
    )


def _gateway(
    *,
    agent_id: str | None,
    pod_id: str | None,
    conn_id: str | None,
    tool_namespaces: Sequence[str] | None,
) -> PrincipalPermissions:
    c = _require(conn_id, name="conn_id", principal=Principal.GATEWAY)
    inbox = inbox_prefix_for(Principal.GATEWAY, conn_id=c)
    ns = _ns()
    publish = (
        f"{ns}.gateway.stream.*.*",  # streams tokens back for any in-flight completion ({agent_id}.{correlation_id})
        str(Subjects.hub_usage_track()),
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    subscribe = (
        f"{inbox}.>",
        str(Subjects.gateway_completion()),  # queue-grouped: ai-gateway
        str(Subjects.gateway_embedding()),  # queue-grouped: ai-gateway
        str(Subjects.gateway_health()),
        str(Subjects.gateway_catalog_epoch()),
        str(Subjects.acl_invalidate("membership")),
        str(Subjects.acl_invalidate("assignment")),
        str(Subjects.acl_invalidate("role")),
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    return PrincipalPermissions(
        publish=publish,
        subscribe=subscribe,
        allow_responses=True,  # replies to completion / embedding / health requests
        inbox_prefix=inbox,
        kv_buckets=(f"{ns}-collections",),
    )


def _channel_adapter(
    *,
    agent_id: str | None,
    pod_id: str | None,
    conn_id: str | None,
    tool_namespaces: Sequence[str] | None,
) -> PrincipalPermissions:
    c = _require(conn_id, name="conn_id", principal=Principal.CHANNEL_ADAPTER)
    inbox = inbox_prefix_for(Principal.CHANNEL_ADAPTER, conn_id=c)
    ns = _ns()
    publish = (
        f"{ns}.agents.route.*",  # routes inbound channel messages to agents
        str(Subjects.hub_channel_installs()),  # fetches its bot installs
        str(Subjects.hub_user_resolve()),  # resolves a channel sender to a platform user
        # HITL: the router (in this sandboxed adapter) forwards an operator reply to the approval
        # broker, which authorizes + resolves it into a resume verdict. reply only; no marker write.
        str(Subjects.hub_approval_resolve()),
        f"{ns}.hub.channel.installs.changed",  # best-effort orphan-reload signal
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    subscribe = (
        f"{inbox}.>",
        str(Subjects.channels_deliver_wildcard()),  # durable consumer of agent answers
        f"{ns}.hub.channel.installs.changed",  # reconcile live connections on install changes
        f"{ns}.channels.room.*",  # cross-pod room fanout (live socket delivery)
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    return PrincipalPermissions(
        publish=(*publish, f"{ns}.channels.room.*"),  # publishes room frames too
        subscribe=subscribe,
        allow_responses=True,
        inbox_prefix=inbox,
        kv_buckets=(f"{ns}-collections",),
        streams=(f"{ns}_channels_deliver",),
    )


_RESOLVERS = {
    Principal.AGENT_POD: _agent_pod,
    Principal.TOOL_POD: _tool_pod,
    Principal.REGISTRY: _registry,
    Principal.HUB: _hub,
    Principal.GATEWAY: _gateway,
    Principal.CHANNEL_ADAPTER: _channel_adapter,
}

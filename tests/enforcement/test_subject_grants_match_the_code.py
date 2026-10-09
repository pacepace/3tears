"""enforcement: each principal's subject grants are the subjects its code receives and publishes on.

:mod:`threetears.nats.subject_permissions` says what each principal kind may subscribe and publish.
The code says what it does. This gate holds the two equal, in both directions:

- **every site is granted.** A subject a receive site subscribes, or a publish site sends on, is
  covered by its principal's list. A subscribe the broker refuses is not an error anybody sees: it
  is a subscription that receives nothing, which in cache invalidation is a silently stale cache.
- **every grant has a site.** No grant outlives the code that needed it, unless it is on
  :data:`_KEPT_FOR_OLDER_RELEASES`, dated and with its reason: the auth callout mints these lists for
  pods running OLDER releases too, so a subject the current code stopped using stays until no
  supported release uses it (expand, migrate, contract).
- **every receiver in this repository is accounted for.** The sender-posture walker finds each place
  3tears starts receiving; :func:`_receivers` says which principals run it and on which subjects, or
  why it needs no grant of its own. A new receiver fails here until it is placed.
- **dead-letters follow subscriptions.** ``NatsClient.subscribe`` and ``subscribe_typed`` republish a
  failed callback to ``{ns}.deadletter.<subscription subject>`` by default, so each principal may
  publish the dead-letter of each subscription it holds, and no other.

The facts the inventory applies, and which keep a consumed stream's subject off a subscribe list: a
JetStream push or pull consumer, a KV watch, a result waiter and a request's reply all receive on the
connection's inbox (nats-py names a push consumer's ``deliver_subject`` with ``new_inbox()``). A
request is a publish of its subject. A JetStream publish is a core publish of the stream subject,
acknowledged on the inbox. An answer to a request rides ``allow_responses`` and needs no grant.

Subjects built at runtime are built here through the same :class:`Subjects` builder the site calls,
with the runtime tokens filled the way the site fills them. A site in this repository names its module
and the builder call its code (not a comment) must still contain, so a site that moves or goes fails
here. A site in the SDK or the hub names its file and line as the evidence it was placed on; nothing
here checks that evidence, so it is a pointer for a reader, not a guarantee.

:data:`Principal.HUB` is not gated: see :data:`_UNGATED`.
"""

from __future__ import annotations

import secrets
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from uuid import UUID

import pytest

from tests.enforcement.sender_posture_support import SENDER_POSTURE_CONFIG as _POSTURE_CONFIG
from threetears.enforcement.sender_posture import find_subscribe_sites
from threetears.nats.subject_permissions import (
    CROSS_PLATFORM_CACHE_INVALIDATE,
    SERVER_USER_INFO_SUBJECT,
    Principal,
    PrincipalPermissions,
    build_permissions,
)
from threetears.nats.errors import NamespaceNotConfiguredError
from threetears.nats.subjects import Subjects, get_default_namespace

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: the principal kinds this gate does not hold, each with why. Every other member is gated.
_UNGATED: Final[Mapping[Principal, str]] = {
    Principal.HUB: (
        "the hub connects as the unrestricted static user and is never minted from this resolver; it "
        "publishes credential refusals and renewals onto OTHER connections' inboxes "
        "(aibots/hub/security/connection_fence.py), which no per-principal allow-list can express, "
        "and it serves token-carrying subjects (gateway.completion context capture); its resolver is "
        "descriptive and its correction is its own piece of work"
    ),
}

#: grants kept for a release older than the current code, with the date the current code stopped
#: using the subject and why an older release still needs it. Each one has no current site; once no
#: supported release uses it, the entry and the grant go together. Empty: no subject a pod's grant
#: lost in this change was used by any release (see the commit that introduced this gate).
_KEPT_FOR_OLDER_RELEASES: Final[Mapping[Principal, Mapping[str, tuple[str, str]]]] = {}

_AGENT: Final[str] = "019470a8-b5c3-7def-8123-000000000001"
_AGENT_POD_ID: Final[str] = "01947100-0000-7000-8000-000000000001"
_TOOL_POD_ID: Final[str] = "01947100-0000-7000-8000-000000000002"
#: the tool-name node a tool pod owns, as the host resolves it at connect
_OWNED_NODE: Final[str] = "pentest"

_IDS: Final[Mapping[Principal, Mapping[str, object]]] = {
    Principal.AGENT_POD: {"agent_id": _AGENT, "pod_id": _AGENT_POD_ID},
    Principal.TOOL_POD: {"pod_id": _TOOL_POD_ID, "tool_namespaces": (_OWNED_NODE,), "object_store": True},
    Principal.REGISTRY: {"conn_id": "registry-1"},
    Principal.HUB: {"conn_id": "hub-1"},
    Principal.GATEWAY: {"conn_id": "gateway-1"},
    Principal.CHANNEL_ADAPTER: {"conn_id": "adapter-1"},
    Principal.AGENT_ROUTER: {"conn_id": "router-1"},
    Principal.DATASET_EXECUTOR: {"conn_id": "executor-1"},
}

_GATED: Final[tuple[Principal, ...]] = tuple(p for p in Principal if p not in _UNGATED)


@dataclass(frozen=True)
class _Site:
    """one place code receives or publishes, and the subjects it does it on.

    :ivar where: ``<repo-relative path>`` in this repository, or ``sdk:``/``hub:`` plus a path and line
    :ivar needle: text the module in this repository must still contain; empty for a site elsewhere
    :ivar subjects: the subjects, built by the site's own builder
    """

    where: str
    needle: str
    subjects: tuple[str, ...]


def _here(path: str, needle: str, *subjects: str) -> _Site:
    """a site in this repository.

    :param path: repo-relative path of the module
    :ptype path: str
    :param needle: text the module must contain, normally the builder call
    :ptype needle: str
    :param subjects: the subjects the site uses
    :ptype subjects: str
    :return: the site
    :rtype: _Site
    """
    return _Site(where=path, needle=needle, subjects=subjects)


def _there(where: str, *subjects: str) -> _Site:
    """a site in the SDK or the hub, recorded as file and line evidence.

    :param where: ``sdk:<path>:<line>`` or ``hub:<path>:<line>``
    :ptype where: str
    :param subjects: the subjects the site uses
    :ptype subjects: str
    :return: the site
    :rtype: _Site
    """
    return _Site(where=where, needle="", subjects=subjects)


@pytest.fixture(autouse=True)
def _namespace(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """bind a namespace when none is, so builders and resolvers render one.

    :param monkeypatch: pytest's monkeypatch
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: nothing
    :rtype: Iterator[None]
    """
    try:
        get_default_namespace()
    except NamespaceNotConfiguredError:
        monkeypatch.setenv("THREETEARS_NATS_SUBJECT_NAMESPACE", "aibots")
    yield


def _permissions(principal: Principal) -> PrincipalPermissions:
    """the resolved permissions of one principal, with representative ids.

    :param principal: the principal kind
    :ptype principal: Principal
    :return: its permissions
    :rtype: PrincipalPermissions
    """
    return build_permissions(principal, **_IDS[principal])  # type: ignore[arg-type]


def _inprocess_pod() -> str:
    """the composite pod id an agent's in-process tool server runs under.

    :return: ``{agent_id}.{instance}``
    :rtype: str
    """
    return Subjects.agent_inprocess_pod_id(UUID(_AGENT), "instance-1")


def _acl() -> tuple[str, ...]:
    """the three ACL invalidation subjects ``subscribe_acl_invalidation`` subscribes.

    :return: one subject per invalidation kind
    :rtype: tuple[str, ...]
    """
    return tuple(str(Subjects.acl_invalidate(kind)) for kind in ("membership", "assignment", "role"))


def _l3() -> tuple[str, ...]:
    """the L3 broker subjects ``NatsProxyL3Backend`` requests, every transaction op included.

    :return: query, batch and each transaction op
    :rtype: tuple[str, ...]
    """
    ops: tuple[Literal["begin", "execute", "fetch", "fetchrow", "commit", "rollback"], ...] = (
        "begin",
        "execute",
        "fetch",
        "fetchrow",
        "commit",
        "rollback",
    )
    return (str(Subjects.l3_query()), str(Subjects.l3_batch()), *(str(Subjects.l3_tx(op)) for op in ops))


def _inbox_subjects(principal: Principal) -> tuple[str, ...]:
    """the subjects a runner's renewal and refusal doors subscribe, under its own inbox.

    :param principal: the pod principal
    :ptype principal: Principal
    :return: the renewal request and refusal subjects of its inbox prefix
    :rtype: tuple[str, ...]
    """
    prefix = _permissions(principal).inbox_prefix
    return (str(Subjects.credential_renewal_request(prefix)), str(Subjects.credential_refusal(prefix)))


#: why a receiver needs no subject grant of its own, keyed by what it is.
_ON_THE_INBOX: Final[str] = "receives on the connection's inbox, which every principal holds"
_PRIMITIVE: Final[str] = "a subscribe primitive; its callers are the receivers"
_NOT_WIRED: Final[str] = "constructed by no deployed process in 3tears, the SDK or the hub"


def _receivers() -> dict[str, str | dict[Principal, tuple[str, ...]]]:
    """every receiver the sender-posture walker finds here, and who runs it on which subjects.

    A string value says why the receiver needs no grant of its own; a mapping names each principal
    kind whose process runs it and the subjects it subscribes there.

    :return: ``path::function`` to its placement
    :rtype: dict[str, str | dict[Principal, tuple[str, ...]]]
    """
    inprocess = _inprocess_pod()
    session = "session-1"
    every: dict[Principal, tuple[str, ...]] = {p: (str(Subjects.cache_invalidate()),) for p in Principal}
    return {
        "packages/agent/acl/src/threetears/agent/acl/invalidation_bus.py::subscribe_acl_invalidation": {
            # sdk runtime/three_tier_stack.py:321; registry rbac_stack.py:200
            Principal.AGENT_POD: _acl(),
            Principal.REGISTRY: _acl(),
        },
        "packages/agent/audit/src/threetears/agent/audit/persist.py::start_audit_persister": _ON_THE_INBOX,
        "packages/agent/tools/src/threetears/agent/tools/server.py::ToolServer.serve": {
            Principal.AGENT_POD: (str(Subjects.tools_internal(inprocess)), str(Subjects.tools_probe(inprocess))),
            Principal.TOOL_POD: (str(Subjects.tools_internal(_TOOL_POD_ID)), str(Subjects.tools_probe(_TOOL_POD_ID))),
        },
        "packages/channels/src/threetears/channels/presence/fanout.py::RoomFanout._subscribe_locked": _NOT_WIRED,
        "packages/core/src/threetears/core/collections/registry.py::CollectionRegistry.start_invalidation_listener": every,
        "packages/core/src/threetears/core/collections/scoped_snapshot.py::ScopedSnapshot._follow": _ON_THE_INBOX,
        "packages/epoch/src/threetears/epoch/listener.py::EpochListener.subscribe": {
            # hub gateway/service.py:660, the catalog epoch; the hub's mcp.rbac listener is ungated
            Principal.GATEWAY: (str(Subjects.gateway_catalog_epoch()),),
            Principal.HUB: (str(Subjects.mcp_rbac_epoch()),),
        },
        "packages/mcp/src/threetears/mcp/auth.py::LocalGrantAuthorizer.start": {
            # constructed by the hub alone (aibots/hub/app.py); no agent pod builds one
            Principal.HUB: (str(Subjects.mcp_rbac_epoch()),),
        },
        "packages/nats/src/threetears/nats/_named_read.py::read_through_named_consumer": _ON_THE_INBOX,
        "packages/nats/src/threetears/nats/auth_callout_responder.py::AuthCalloutResponder.start": (
            "the auth callout's own user in the system account, which no Principal describes"
        ),
        "packages/nats/src/threetears/nats/client.py::Subscription.subscribe_on": _PRIMITIVE,
        "packages/nats/src/threetears/nats/client.py::JetStreamResultWaiter._subscribe": _ON_THE_INBOX,
        "packages/nats/src/threetears/nats/client.py::NatsClient.renew_connection": _PRIMITIVE,
        "packages/nats/src/threetears/nats/client.py::NatsClient.renew_on_request": {
            Principal.AGENT_POD: _inbox_subjects(Principal.AGENT_POD)[:1],
            Principal.TOOL_POD: _inbox_subjects(Principal.TOOL_POD)[:1],
        },
        "packages/nats/src/threetears/nats/client.py::NatsClient.abandon_on_refusal": {
            Principal.AGENT_POD: _inbox_subjects(Principal.AGENT_POD)[1:],
            Principal.TOOL_POD: _inbox_subjects(Principal.TOOL_POD)[1:],
        },
        "packages/nats/src/threetears/nats/client.py::NatsClient._subscribe_internal": _PRIMITIVE,
        "packages/nats/src/threetears/nats/client.py::NatsClient.jetstream_subscribe_durable._bind": _PRIMITIVE,
        "packages/nats/src/threetears/nats/client.py::NatsClient.jetstream_pull_subscribe._bind": _PRIMITIVE,
        "packages/nats/src/threetears/nats/cross_worker_cancel.py::CrossWorkerCanceller.bind": _NOT_WIRED,
        "packages/nats/src/threetears/nats/forward.py::serve_owner": _PRIMITIVE,
        "packages/nats/src/threetears/nats/kv.py::_KeyWatchConsumer.open": _ON_THE_INBOX,
        "packages/nats/src/threetears/nats/oplog.py::OpLog.replay": _ON_THE_INBOX,
        "packages/nats/src/threetears/nats/pipe.py::PipeStream.open": _PRIMITIVE,
        "packages/nats/src/threetears/nats/pipe.py::serve_pipe": _PRIMITIVE,
        "packages/registry/src/threetears/registry/discovery.py::DiscoveryHandler.start": {
            Principal.REGISTRY: (str(Subjects.tools_discover()),),
        },
        "packages/registry/src/threetears/registry/health.py::HeartbeatSubscriber.start": {
            Principal.REGISTRY: (str(Subjects.tools_heartbeat_wildcard()),),
        },
        "packages/registry/src/threetears/registry/proxy.py::CallProxy.start": {
            Principal.REGISTRY: (str(Subjects.tools_call()),),
        },
        "packages/registry/src/threetears/registry/proxy.py::CallProxy._forward_call_durable": _ON_THE_INBOX,
        "packages/registry/src/threetears/registry/registration.py::RegistrationHandler.start": {
            Principal.REGISTRY: (str(Subjects.tools_register()),),
        },
        "packages/scrape/src/threetears/scrape/operator_control.py::serve_session": {
            Principal.TOOL_POD: (str(Subjects.forward_scoped(Subjects.hitl_forward_family(_OWNED_NODE), session)),),
        },
        "packages/scrape/src/threetears/scrape/operator_pipe.py::serve_display": {
            # the attach on the control family, then the caller's half of the pipe (pipe.py PipeStream.open)
            Principal.TOOL_POD: (
                str(Subjects.forward_scoped(Subjects.hitl_pipe_family(_OWNED_NODE), session)),
                str(Subjects.pipe(_OWNED_NODE, _TOOL_POD_ID, secrets.token_hex(8), "up")),
            ),
        },
    }


def _external_receive_sites() -> dict[Principal, tuple[_Site, ...]]:
    """the receive sites outside this repository, in the SDK and the hub.

    :return: principal to its receive sites there
    :rtype: dict[Principal, tuple[_Site, ...]]
    """
    ns = get_default_namespace()
    return {
        Principal.AGENT_POD: (
            _there(
                "sdk:src/aibots_agents/runtime/nats_transport.py:812",
                str(Subjects.agent_internal(_AGENT, _AGENT_POD_ID)),
            ),
            _there(
                "sdk:src/aibots_agents/runtime/bootstrap/orchestrator.py:789",
                str(Subjects.agent_reregister_request(_AGENT, _AGENT_POD_ID)),
            ),
            # a core subscribe of the stream subject itself, not a request inbox
            _there(
                "sdk:src/aibots_agents/runtime/gateway_model.py:593", str(Subjects.gateway_stream(_AGENT, uuid.uuid4()))
            ),
        ),
        Principal.GATEWAY: (
            _there("hub:src/aibots/gateway/handlers/completion.py:237", str(Subjects.gateway_completion())),
            _there("hub:src/aibots/gateway/handlers/embedding.py:142", str(Subjects.gateway_embedding())),
            _there("hub:src/aibots/gateway/service.py:682", str(Subjects.gateway_health())),
            _there("hub:src/aibots/hub/broker/acl.py:798", *_acl()),
        ),
        Principal.CHANNEL_ADAPTER: (
            _there(
                "hub:src/aibots/channels/slack/adapter.py:1098 and delivery_worker.py:814",
                f"{ns}.hub.channel.installs.changed",
            ),
            # dispatch_to_agent waits on a random-token reply subject
            _there("hub:src/aibots/hub/router/agent_dispatch.py:281", str(Subjects.agent_reply(secrets.token_hex(16)))),
        ),
        Principal.AGENT_ROUTER: (
            _there("hub:src/aibots/agent_router/registration.py:135", str(Subjects.agent_register())),
            _there("hub:src/aibots/agent_router/registration.py:140", str(Subjects.agent_deregister())),
            _there("hub:src/aibots/agent_router/health.py:154", str(Subjects.agent_heartbeat_wildcard())),
            _there("hub:src/aibots/agent_router/proxy.py:690", str(Subjects.agent_complete(uuid.uuid4()))),
        ),
    }


def _publish_sites() -> dict[Principal, tuple[_Site, ...]]:
    """every publish site of every gated principal, dead-letters aside.

    :return: principal to its publish sites
    :rtype: dict[Principal, tuple[_Site, ...]]
    """
    ns = get_default_namespace()
    invalidate = _here(
        "packages/core/src/threetears/core/collections/registry.py",
        "Subjects.cache_invalidate()",
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    jwks = _here(
        "packages/core/src/threetears/core/security/jwks_provider.py", "Subjects.hub_jwks()", str(Subjects.hub_jwks())
    )
    l3 = _here("packages/core/src/threetears/core/backends/nats_proxy.py", "Subjects.l3_tx(", *_l3())
    server = "packages/agent/tools/src/threetears/agent/tools/server.py"
    audit = "packages/agent/audit/src/threetears/agent/audit/publish.py"
    inprocess = _inprocess_pod()
    return {
        Principal.AGENT_POD: (
            _there("sdk:src/aibots_agents/runtime/bootstrap/phases/handler.py:357", str(Subjects.agent_register())),
            _there("sdk:src/aibots_agents/runtime/bootstrap/phases/handler.py:431", str(Subjects.agent_deregister())),
            _there(
                "sdk:src/aibots_agents/runtime/bootstrap/orchestrator.py:1398",
                str(Subjects.agent_heartbeat(_AGENT, _AGENT_POD_ID)),
            ),
            _there("sdk:src/aibots_agents/runtime/handshake.py:284", str(Subjects.hub_handshake())),
            _there("sdk:src/aibots_agents/runtime/secrets.py:135", str(Subjects.hub_secrets_request())),
            _there("sdk:src/aibots_agents/runtime/catalog.py:270", str(Subjects.hub_object_commit())),
            _here(
                "packages/agent/tools/src/threetears/agent/tools/object_resolver.py",
                "Subjects.hub_object_resolve()",
                str(Subjects.hub_object_resolve()),
            ),
            _there("sdk:src/aibots_agents/runtime/nats_transport.py:1937", str(Subjects.hub_approval_record())),
            _here(
                "packages/agent/memory/src/threetears/agent/memory/namespace_client.py",
                "Subjects.hub_memory_namespace_ensure()",
                str(Subjects.hub_memory_namespace_ensure()),
            ),
            # the agent-side erasure contract: a library call an agent makes, wired by no SDK path yet
            _here(
                "packages/agent/audit/src/threetears/agent/audit/erasure.py",
                "Subjects.hub_audit_anonymize()",
                str(Subjects.hub_audit_anonymize()),
            ),
            _there(
                "sdk:src/aibots_agents/runtime/channel_default_client.py:123",
                str(Subjects.hub_channel_engagement_default_resolve()),
            ),
            jwks,
            _there("sdk:src/aibots_agents/runtime/gateway_model.py:419", str(Subjects.gateway_completion())),
            _there("sdk:src/aibots_agents/runtime/gateway_embedding.py:257", str(Subjects.gateway_embedding())),
            _here(server, "Subjects.tools_discover()", str(Subjects.tools_discover())),
            _there("sdk:src/aibots_agents/runtime/tools.py:508", str(Subjects.tools_call())),
            _here(server, "Subjects.tools_register()", str(Subjects.tools_register())),
            _here(server, "Subjects.tools_heartbeat(", str(Subjects.tools_heartbeat(inprocess))),
            # the in-process server's durable result, a JetStream publish
            _here(server, "jetstream_publish(", str(Subjects.tools_result(inprocess, uuid.uuid4()))),
            _here(audit, "Subjects.audit_event(", str(Subjects.audit_event("tool.call"))),
            _here(
                "packages/agent/knowledge/src/threetears/agent/knowledge/integration.py",
                "Subjects.knowledge_draft()",
                str(Subjects.knowledge_draft()),
            ),
            _here(
                "packages/agent/workspace/src/threetears/agent/workspace/tools/workspace_create.py",
                "Subjects.workspaces_create()",
                str(Subjects.workspaces_create()),
            ),
            _here(
                "packages/agent/tools/src/threetears/agent/tools/namespace_discovery_client.py",
                ".namespace.discover",
                str(Subjects.namespace_discover()),
            ),
            l3,
            # a JetStream publish of the answer onto the channel's delivery stream
            _there("sdk:src/aibots_agents/runtime/nats_transport.py:2002", str(Subjects.channels_deliver("slack"))),
            _there(
                "sdk:src/aibots_agents/runtime/streaming_transport.py:122",
                str(Subjects.hub_stream(_AGENT, uuid.uuid4())),
            ),
            _there("sdk:src/aibots_agents/runtime/nats_transport.py:1247", str(Subjects.agent_complete(uuid.uuid4()))),
            invalidate,
        ),
        Principal.TOOL_POD: (
            _here(server, "Subjects.tools_register()", str(Subjects.tools_register())),
            _here(server, "Subjects.tools_heartbeat(", str(Subjects.tools_heartbeat(_TOOL_POD_ID))),
            # wait_until_ready polls discovery
            _here(server, "Subjects.tools_discover()", str(Subjects.tools_discover())),
            # tool_pod_identity.py:425 drives the handshake for a pod's own token
            _there("sdk:src/aibots_agents/runtime/handshake.py:284", str(Subjects.hub_handshake())),
            l3,
            _here(
                "packages/datasources/src/threetears/datasources/query_client.py",
                "Subjects.datasource_query(",
                str(Subjects.datasource_query("survey_results")),
            ),
            _here(
                "packages/registry/src/threetears/registry/client.py",
                "Subjects.tools_call()",
                str(Subjects.tools_call()),
            ),
            jwks,
            _here(audit, "Subjects.audit_event(", str(Subjects.audit_event("tool.call"))),
            _here(
                audit,
                "Subjects.tool_pod_audit_event(",
                str(Subjects.tool_pod_audit_event(_TOOL_POD_ID, "collector.promoted")),
            ),
            _here(
                "packages/agent/audit/src/threetears/agent/audit/erasure.py",
                "Subjects.hub_audit_anonymize()",
                str(Subjects.hub_audit_anonymize()),
            ),
            _here(
                "packages/datasources/src/threetears/datasources/geo_reload.py",
                "Subjects.hub_geo_layers_reloaded()",
                str(Subjects.hub_geo_layers_reloaded()),
            ),
            _here(
                "packages/agent/tools/src/threetears/agent/tools/object_resolver.py",
                "Subjects.hub_object_resolve()",
                str(Subjects.hub_object_resolve()),
            ),
            _here(
                "packages/agent/tools/src/threetears/agent/tools/engagement_resolver.py",
                "Subjects.hub_engagement_scope()",
                str(Subjects.hub_engagement_scope()),
            ),
            # the discovery client moved where a tool pod can reach it
            _here(
                "packages/agent/tools/src/threetears/agent/tools/namespace_discovery_client.py",
                ".namespace.discover",
                str(Subjects.namespace_discover()),
            ),
            _here(server, "jetstream_publish(", str(Subjects.tools_result(_TOOL_POD_ID, uuid.uuid4()))),
            _here("packages/nats/src/threetears/nats/client.py", "SERVER_USER_INFO_SUBJECT", SERVER_USER_INFO_SUBJECT),
            invalidate,
            _here(
                "packages/nats/src/threetears/nats/pipe.py",
                "publish_raw(",
                str(Subjects.pipe(_OWNED_NODE, _TOOL_POD_ID, secrets.token_hex(8), "down")),
            ),
            _here(
                "packages/nats/src/threetears/nats/object_store_requests.py",
                "Subjects.hub_object_store_declare()",
                str(Subjects.hub_object_store_declare()),
            ),
            _here(
                "packages/nats/src/threetears/nats/object_store_requests.py",
                "Subjects.hub_object_store_retire()",
                str(Subjects.hub_object_store_retire()),
            ),
        ),
        Principal.REGISTRY: (
            # the sync forward and the durable accept, to a tool pod and to an agent's in-process server
            _here(
                "packages/registry/src/threetears/registry/proxy.py",
                "Subjects.tools_internal(",
                str(Subjects.tools_internal(_TOOL_POD_ID)),
                str(Subjects.tools_internal(inprocess)),
            ),
            _here(
                "packages/registry/src/threetears/registry/registration.py",
                "Subjects.tools_probe(",
                str(Subjects.tools_probe(_TOOL_POD_ID)),
            ),
            jwks,
            # the durable reply to an agent, a JetStream publish
            _here(
                "packages/registry/src/threetears/registry/proxy.py",
                "jetstream_publish(",
                str(Subjects.tools_reply(_AGENT, uuid.uuid4())),
            ),
            # the hub's registry_identity factory, loaded into the registry process, handshakes
            _there("hub:src/aibots/hub/tools/registry_identity.py:281", str(Subjects.hub_handshake())),
            # the RBAC stack and the hub's registry_auth factories read L3 through the broker
            _here("packages/registry/src/threetears/registry/rbac_stack.py", "NatsProxyL3Backend", *_l3()),
            # the hub's usage_emitter factory, loaded into the registry process, reports endpoint usage
            _there("hub:src/aibots/hub/tools/registry_auth.py:523", str(Subjects.hub_endpoint_usage_track())),
            invalidate,
        ),
        Principal.GATEWAY: (
            _there(
                "hub:src/aibots/gateway/handlers/streaming.py:514", str(Subjects.gateway_stream(_AGENT, uuid.uuid4()))
            ),
            _there("hub:src/aibots/hub/usage/audit_sink.py:419", str(Subjects.hub_usage_track())),
            _there("hub:src/aibots/gateway/media.py:226", str(Subjects.hub_object_resolve())),
            _there("hub:src/aibots/gateway/service.py:790", str(Subjects.hub_jwks())),
            invalidate,
        ),
        Principal.CHANNEL_ADAPTER: (
            _there("hub:src/aibots/channels/install_fetch.py:87", str(Subjects.hub_channel_installs())),
            _there("hub:src/aibots/channels/router.py:247", str(Subjects.hub_user_resolve())),
            _there("hub:src/aibots/hub/approvals/resolve_client.py:168", str(Subjects.hub_approval_resolve())),
            # dispatch_to_agent's JetStream publish of the turn
            _there("hub:src/aibots/hub/router/agent_dispatch.py:288", str(Subjects.agent_turn(_AGENT))),
            _there("hub:src/aibots/channels/router.py:481", f"{ns}.hub.channel.installs.changed"),
            # the delivery worker verifies the hub's turn assertion against the hub's key set
            _there("hub:src/aibots/channels/delivery_worker.py:770", str(Subjects.hub_jwks())),
            # the delivery consumer's dead-letter once its budget is spent (NatsClient's redelivery policy)
            _there("hub:src/aibots/channels/delivery_worker.py:429", str(Subjects.channels_deliver("deadletter"))),
            invalidate,
        ),
        Principal.AGENT_ROUTER: (
            _there("hub:src/aibots/agent_router/proxy.py:982", str(Subjects.agent_internal(_AGENT, _AGENT_POD_ID))),
            _there(
                "hub:src/aibots/agent_router/health.py:254",
                str(Subjects.agent_reregister_request(_AGENT, _AGENT_POD_ID)),
            ),
            _there("hub:src/aibots/agent_router/proxy.py:918", str(Subjects.agent_turn_deadletter())),
            # the answer, an early ack or an error, to the turn's one-token reply subject
            _there("hub:src/aibots/agent_router/proxy.py:709", str(Subjects.agent_reply(secrets.token_hex(16)))),
            _there("hub:src/aibots/agent_router/server.py:449", str(Subjects.hub_jwks())),
            invalidate,
        ),
        Principal.DATASET_EXECUTOR: (invalidate,),
    }


#: sites whose subject a principal's grant deliberately does NOT cover, each with why. A row here is a
#: decision waiting on its owner, not an exemption that hides: the gate fails if the grant starts
#: covering the subject, so the row goes in the same change.
_WITHHELD: Final[Mapping[Principal, Mapping[str, Callable[[], str]]]] = {
    Principal.AGENT_POD: {
        # engagement_resolver.py: the in-process tool server builds the resolver, and no in-process tool
        # calls resolve_engagement_scope today. Widening a minted pod grant for a latent path is a
        # decision, not an inventory correction.
        "packages/agent/tools/src/threetears/agent/tools/server.py: in-process hub.engagement.scope (latent)": (
            lambda: str(Subjects.hub_engagement_scope())
        ),
        # the in-process workspace tools publish audit.workspace.* (materialize.py:1285, fs_write.py:248
        # and seven more); cluster pods refuse workspace tools, so only a devx agent pod reaches it.
        "packages/agent/workspace/src/threetears/agent/workspace/materialize.py: audit.workspace.* (devx only)": (
            lambda: str(Subjects.audit_event("workspace.fs_write"))
        ),
    },
    Principal.TOOL_POD: {
        # survey usage_reporting_service.py:124; usage is token-carrying, and no admin path emits it
        "survey:src/survey_engine/services/usage_reporting_service.py:124: hub.usage.endpoint (latent)": (
            lambda: str(Subjects.hub_endpoint_usage_track())
        ),
    },
}


def _covers(pattern: str, subject: str) -> bool:
    """whether a grant pattern covers a subject, the subject itself perhaps a pattern.

    A ``*`` in the grant covers one token, a wildcard included; a ``>`` covers the rest. A literal
    grant token covers only the same literal, so a wildcard subject is covered only by a grant at
    least as wide.

    :param pattern: the grant
    :ptype pattern: str
    :param subject: the subject or subscription pattern
    :ptype subject: str
    :return: whether every subject ``subject`` matches is matched by ``pattern``
    :rtype: bool
    """
    grant, wanted = pattern.split("."), subject.split(".")
    covered = True
    tail = False
    for index, token in enumerate(grant):
        if token == ">":
            tail = True
            covered = len(wanted) > index
            break
        if index >= len(wanted) or (token == "*" and wanted[index] == ">") or token not in ("*", wanted[index]):
            covered = False
            break
    return covered and (tail or len(grant) == len(wanted))


def _any_covers(patterns: tuple[str, ...], subject: str) -> bool:
    """whether any grant covers the subject.

    :param patterns: the grants
    :ptype patterns: tuple[str, ...]
    :param subject: the subject
    :ptype subject: str
    :return: whether one of them covers it
    :rtype: bool
    """
    return any(_covers(pattern, subject) for pattern in patterns)


def _receive_sites(principal: Principal) -> tuple[_Site, ...]:
    """every receive site a principal's process runs, here and elsewhere.

    :param principal: the principal kind
    :ptype principal: Principal
    :return: its receive sites
    :rtype: tuple[_Site, ...]
    """
    here = tuple(
        _Site(where=key, needle="", subjects=placement[principal])
        for key, placement in _receivers().items()
        if isinstance(placement, dict) and principal in placement
    )
    return (*here, *_external_receive_sites().get(principal, ()))


def _deadletters(permissions: PrincipalPermissions) -> tuple[str, ...]:
    """the dead-letter subjects of a principal's subscriptions, its inbox aside.

    :param permissions: the principal's permissions
    :ptype permissions: PrincipalPermissions
    :return: one dead-letter subject per subscribed subject
    :rtype: tuple[str, ...]
    """
    return tuple(
        str(Subjects.deadletter(s)) for s in permissions.subscribe if not s.startswith(f"{permissions.inbox_prefix}.")
    )


def _kept(principal: Principal) -> tuple[str, ...]:
    """the grants a principal keeps for older releases.

    :param principal: the principal kind
    :ptype principal: Principal
    :return: the kept patterns
    :rtype: tuple[str, ...]
    """
    return tuple(_KEPT_FOR_OLDER_RELEASES.get(principal, {}))


def test_every_principal_is_gated_or_says_why_not() -> None:
    """the roster is total, so a new principal kind is gated from its first commit.

    :return: nothing
    :rtype: None
    """
    assert set(_GATED) | set(_UNGATED) == set(Principal)
    assert all(principal in _IDS for principal in Principal)


def test_every_receiver_here_is_placed() -> None:
    """each receiver the walker finds is placed, and each placement still names a receiver.

    :return: nothing
    :rtype: None
    """
    found = {f"{site.path}::{site.function}" for site in find_subscribe_sites(_POSTURE_CONFIG)}
    placed = set(_receivers())
    assert not found - placed, f"receivers not placed in _receivers(): {sorted(found - placed)}"
    assert not placed - found, f"placements naming no receiver: {sorted(placed - found)}"


def _code_of(path: Path) -> str:
    """a module's text with its comment lines left out, so a needle cannot be found in a comment.

    :param path: the module
    :ptype path: Path
    :return: its code lines
    :rtype: str
    """
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")
    )


def test_every_site_here_still_exists() -> None:
    """each publish site in this repository still holds the text that makes it the site.

    :return: nothing
    :rtype: None
    """
    missing = [
        f"{site.where}: {site.needle!r}"
        for sites in _publish_sites().values()
        for site in sites
        if site.needle and site.needle not in _code_of(_REPO_ROOT / site.where)
    ]
    assert not missing, f"publish sites gone from their modules: {missing}"


@pytest.mark.parametrize("principal", _GATED)
def test_every_receive_site_is_granted(principal: Principal) -> None:
    """a subject a principal's code subscribes is on its subscribe list.

    :param principal: the principal kind
    :ptype principal: Principal
    :return: nothing
    :rtype: None
    """
    subscribe = _permissions(principal).subscribe
    refused = [
        f"{site.where}: {subject}"
        for site in _receive_sites(principal)
        for subject in site.subjects
        if not _any_covers(subscribe, subject)
    ]
    assert not refused, f"{principal.value} subscribes subjects its grant refuses: {refused}"


@pytest.mark.parametrize("principal", _GATED)
def test_every_subscribe_grant_has_a_site(principal: Principal) -> None:
    """no subscribe grant outlives the receive site that needed it.

    :param principal: the principal kind
    :ptype principal: Principal
    :return: nothing
    :rtype: None
    """
    permissions = _permissions(principal)
    used = [subject for site in _receive_sites(principal) for subject in site.subjects]
    inbox = f"{permissions.inbox_prefix}.>"
    unused = [
        pattern
        for pattern in permissions.subscribe
        if pattern != inbox and pattern not in _kept(principal) and not any(_covers(pattern, s) for s in used)
    ]
    assert not unused, f"{principal.value} is granted {unused} on subscribe, which no receive site uses"


@pytest.mark.parametrize("principal", _GATED)
def test_every_publish_site_is_granted(principal: Principal) -> None:
    """a subject a principal's code publishes, a request's included, is on its publish list.

    :param principal: the principal kind
    :ptype principal: Principal
    :return: nothing
    :rtype: None
    """
    publish = _permissions(principal).publish
    refused = [
        f"{site.where}: {subject}"
        for site in _publish_sites()[principal]
        for subject in site.subjects
        if not _any_covers(publish, subject)
    ]
    assert not refused, f"{principal.value} publishes subjects its grant refuses: {refused}"


@pytest.mark.parametrize("principal", _GATED)
def test_every_publish_grant_has_a_site(principal: Principal) -> None:
    """no publish grant outlives its site; a dead-letter's site is the subscription it follows.

    :param principal: the principal kind
    :ptype principal: Principal
    :return: nothing
    :rtype: None
    """
    permissions = _permissions(principal)
    used = [subject for site in _publish_sites()[principal] for subject in site.subjects]
    used.extend(_deadletters(permissions))
    unused = [
        pattern
        for pattern in permissions.publish
        if pattern not in _kept(principal) and not any(_covers(pattern, s) for s in used)
    ]
    assert not unused, f"{principal.value} is granted {unused} on publish, which no publish site uses"


@pytest.mark.parametrize("principal", list(Principal))
def test_dead_letters_are_exactly_the_subscriptions(principal: Principal) -> None:
    """a principal may dead-letter each subscription it holds, and nothing else under the subtree.

    :param principal: the principal kind
    :ptype principal: Principal
    :return: nothing
    :rtype: None
    """
    permissions = _permissions(principal)
    prefix = f"{get_default_namespace()}.deadletter."
    granted = tuple(s for s in permissions.publish if s.startswith(prefix))
    assert granted == _deadletters(permissions), (
        f"{principal.value} dead-letter grants {granted} are not its subscriptions' {_deadletters(permissions)}"
    )


@pytest.mark.parametrize("principal", _GATED)
def test_withheld_subjects_stay_withheld(principal: Principal) -> None:
    """a withheld site's subject is not granted; granting it means deleting its row.

    :param principal: the principal kind
    :ptype principal: Principal
    :return: nothing
    :rtype: None
    """
    permissions = _permissions(principal)
    granted = [
        where
        for where, subject in _WITHHELD.get(principal, {}).items()
        if _any_covers((*permissions.publish, *permissions.subscribe), subject())
    ]
    assert not granted, f"{principal.value} now grants withheld subjects; delete their rows: {granted}"


def test_kept_grants_are_dated_and_still_granted() -> None:
    """each kept grant is still on its list, has no current site, and says when and why.

    :return: nothing
    :rtype: None
    """
    problems: list[str] = []
    for principal, kept in _KEPT_FOR_OLDER_RELEASES.items():
        permissions = _permissions(principal)
        sites = [s for site in (*_receive_sites(principal), *_publish_sites()[principal]) for s in site.subjects]
        for pattern, (stopped, reason) in kept.items():
            if pattern not in (*permissions.publish, *permissions.subscribe):
                problems.append(f"{principal.value} keeps {pattern} but no longer grants it; drop the row")
            if any(_covers(pattern, s) for s in sites):
                problems.append(f"{principal.value} keeps {pattern} though a current site uses it")
            if len(stopped) != 10 or not reason.strip():
                problems.append(f"{principal.value} keeps {pattern} without an ISO date and a reason")
    assert not problems, problems


@pytest.mark.parametrize(
    ("grant", "subject", "expected"),
    [
        ("a.b.*", "a.b.c", True),
        ("a.b.*", "a.b.c.d", False),
        ("a.b.>", "a.b.c.d", True),
        ("a.b.>", "a.b", False),
        ("a.b.*", "a.b.>", False),
        ("a.b.>", "a.b.>", True),
        ("a.b.c", "a.b.*", False),
        ("a.*.c", "a.*.c", True),
    ],
)
def test_coverage_reads_nats_wildcards(grant: str, subject: str, expected: bool) -> None:
    """the coverage rule the gate rests on reads NATS wildcards the way the server does.

    :param grant: the grant pattern
    :ptype grant: str
    :param subject: the subject or subscription pattern
    :ptype subject: str
    :param expected: whether the grant covers it
    :ptype expected: bool
    :return: nothing
    :rtype: None
    """
    assert _covers(grant, subject) is expected

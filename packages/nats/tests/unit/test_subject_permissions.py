"""lint + contract tests for the per-principal NATS subject-permission map (platform-auth A).

These pin the safety invariants the auth-callout responder relies on when it mints each principal's
user JWT from :func:`build_permissions`:

- **least privilege** — no principal gets a bare ``>``/``*``, the namespace-wide ``{ns}.>``, or the
  global ``_INBOX.>``; every subject is namespace-scoped (or the one documented cross-platform
  constant, or the principal's own scoped inbox);
- **identity isolation** — a pod's identity-bound subjects + reply inbox carry ITS own ids, so pod A
  cannot subscribe to pod B's inbox or impersonate B's identity-tailed subjects;
- **boot completeness** — each principal can perform its bootstrap (a missing boot-critical subject
  bricks the principal the moment auth is enforced);
- **fail closed** — a principal cannot be resolved without the ids it must scope on.
"""

from __future__ import annotations

import base64
import json
import uuid

import pytest
from nats.js.api import StorageType

from threetears.nats.subject_permissions import (
    AGENT_POD_PLATFORM_BUCKET_SUFFIXES,
    CROSS_PLATFORM_CACHE_INVALIDATE,
    SERVER_USER_INFO_SUBJECT,
    TOOL_POD_OBJECTS_BUCKET_SUFFIX,
    TOOL_POD_POINTERS_BUCKET_SUFFIX,
    DATA_VERSIONS_BUCKET_SUFFIX,
    MAX_COORDINATION_BUCKETS,
    WORKSPACE_LOCKS_BUCKET_SUFFIX,
    AgentBucketGrant,
    AgentTableGrant,
    JsCapability,
    JsResource,
    JsResourceKind,
    Principal,
    PrincipalPermissions,
    agent_config_bucket_name,
    agent_config_kv_key,
    agent_platform_bucket_suffix,
    build_permissions,
    capability_declares,
    coordination_bucket_name,
    data_version_kv_key,
    data_versions_bucket_name,
    kv_bucket_names,
    kv_key_scope_for,
    kv_key_scope_for_service,
    kv_stream_name,
    tool_pod_object_store_name,
    tool_pod_pointers_bucket_name,
)
from threetears.nats.kv import build_kv_stream_config
from threetears.nats.result_delivery import result_stream_name
from threetears.nats.subjects import Subjects, parse_tool_pod_audit_subject, set_default_namespace
from threetears.nats.user_jwt import generate_account_seed, js_api_grants_for_stream, mint_user_jwt

_NS = "3tears"

#: Pod identities are UUIDs, and that is a CONTRACT rather than a test convenience. A pod
#: principal's L2 key scope is derived from its identifying id by ``kv_key_scope_for``, which
#: refuses anything that is not a uuid: the scope is an isolation boundary, and a boundary derived
#: from an arbitrary display name is not provably collision-free. So a resolver can no longer be
#: built for a pod whose id is a slug -- it raises, at mint, which is the fail-closed direction.
_AGENT_1 = "019470a8-b5c3-7def-8123-000000000001"
_AGENT_2 = "019470a8-b5c3-7def-8123-000000000002"
_AGENT_A = "019470a8-b5c3-7def-8123-0000000000aa"
_AGENT_B = "019470a8-b5c3-7def-8123-0000000000bb"
_POD_1 = "01947100-0000-7000-8000-000000000001"
_POD_2 = "01947100-0000-7000-8000-000000000002"
_POD_A = "01947100-0000-7000-8000-0000000000aa"
_POD_B = "01947100-0000-7000-8000-0000000000bb"
_POD_X = "01947100-0000-7000-8000-0000000000cc"
_POD_VICTIM = "01947100-0000-7000-8000-0000000000dd"

#: representative ids so every principal resolves to a concrete allow-list.
_IDS: dict[Principal, dict[str, str]] = {
    Principal.AGENT_POD: {"agent_id": _AGENT_1, "pod_id": _POD_1},
    Principal.TOOL_POD: {"pod_id": _POD_1},
    Principal.REGISTRY: {"conn_id": "reg-1"},
    Principal.HUB: {"conn_id": "hub-1"},
    Principal.GATEWAY: {"conn_id": "gw-1"},
    Principal.CHANNEL_ADAPTER: {"conn_id": "chan-1"},
    Principal.AGENT_ROUTER: {"conn_id": "router-1"},
    Principal.DATASET_EXECUTOR: {"conn_id": "dsx-1"},
}

#: the ONE bucket every principal shares, and therefore the only one a per-principal grant can be
#: expressed on at all.
_COLLECTIONS = f"{_NS}-collections"

#: the bucket each pod principal watches for its own data version.
_DATA_VERSIONS = f"{_NS}-data-versions"

#: the two principals whose identity is a POD rather than a service.
_POD_PRINCIPALS = (Principal.AGENT_POD, Principal.TOOL_POD)

#: the shared pod buckets whose keys lead with the owning pod principal's scope.
_SHARED_OWNER_KEYED = frozenset({f"{_NS}-ratelimits", f"{_NS}-proxy_assertion_nonces", f"{_NS}-leases"})


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    set_default_namespace(_NS)


def _build(principal: Principal) -> PrincipalPermissions:
    return build_permissions(principal, **_IDS[principal])


def _all_subjects(perm: PrincipalPermissions) -> list[str]:
    return [*perm.publish, *perm.subscribe]


class TestLeastPrivilege:
    @pytest.mark.parametrize("principal", list(Principal))
    def test_no_full_wildcard_or_global_inbox(self, principal: Principal) -> None:
        perm = _build(principal)
        for subj in _all_subjects(perm):
            assert subj not in {">", "*", "_INBOX.>", "_INBOX.*"}, f"{principal}: bare wildcard {subj!r}"
            assert subj != f"{_NS}.>", f"{principal}: namespace-wide wildcard {subj!r}"
            # the scoped inbox is `_INBOX_<principal>_<id>` (underscore) -- the global `_INBOX.`
            # (dot) tree is forbidden so a responder's replies cannot be sniffed cross-principal.
            assert not subj.startswith("_INBOX."), f"{principal}: global inbox tree {subj!r}"

    @pytest.mark.parametrize("principal", list(Principal))
    def test_every_subject_is_namespace_scoped(self, principal: Principal) -> None:
        perm = _build(principal)
        for subj in _all_subjects(perm):
            scoped = (
                subj.startswith(f"{_NS}.")
                or subj == CROSS_PLATFORM_CACHE_INVALIDATE
                or subj == SERVER_USER_INFO_SUBJECT
                or subj.startswith(f"{perm.inbox_prefix}.")
            )
            assert scoped, f"{principal}: unscoped subject {subj!r}"
        # the server's user-info subject is safe only as a REQUEST: its answer arrives on the
        # principal's own inbox and describes only the asking connection. Subscribing to it would
        # read every other connection's request.
        assert SERVER_USER_INFO_SUBJECT not in perm.subscribe, f"{principal}: subscribes to {SERVER_USER_INFO_SUBJECT}"

    @pytest.mark.parametrize("principal", list(Principal))
    def test_scoped_inbox_present_and_not_global(self, principal: Principal) -> None:
        perm = _build(principal)
        assert perm.inbox_prefix.startswith("_INBOX_")  # scoped, never the bare `_INBOX`
        assert perm.inbox_prefix != "_INBOX"
        assert f"{perm.inbox_prefix}.>" in perm.subscribe

    @pytest.mark.parametrize("principal", list(Principal))
    def test_responders_may_reply(self, principal: Principal) -> None:
        # every principal here answers at least one request subject, so each relies on
        # allow_responses to reply without a standing publish grant on requester inboxes.
        assert _build(principal).allow_responses is True


class TestIdentityIsolation:
    def test_agent_internal_subject_is_own_identity(self) -> None:
        a = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A)
        a_internal = [s for s in a.subscribe if ".agents.internal." in s]
        assert a_internal == [f"{_NS}.agents.internal.{_AGENT_A}.{_POD_A}"]
        # a different agent's routed inbox is a DIFFERENT subject -> no cross-subscribe
        b = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_B, pod_id=_POD_B)
        assert f"{_NS}.agents.internal.{_AGENT_B}.{_POD_B}" not in a.subscribe
        assert [s for s in b.subscribe if ".agents.internal." in s] != a_internal

    def test_tool_internal_subject_is_own_pod(self) -> None:
        a = build_permissions(Principal.TOOL_POD, pod_id=_POD_A)
        b = build_permissions(Principal.TOOL_POD, pod_id=_POD_B)
        assert f"{_NS}.tools.internal.{_POD_A}" in a.subscribe
        assert f"{_NS}.tools.internal.{_POD_A}" not in b.subscribe

    def test_pod_inbox_is_identity_scoped(self) -> None:
        a = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A)
        b = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_B, pod_id=_POD_B)
        assert a.inbox_prefix != b.inbox_prefix

    def test_pod_may_publish_only_its_own_heartbeat(self) -> None:
        a = build_permissions(Principal.TOOL_POD, pod_id=_POD_A)
        assert f"{_NS}.tools.heartbeat.{_POD_A}" in a.publish
        # no wildcard heartbeat publish -> a pod cannot forge another pod's heartbeat
        assert f"{_NS}.tools.heartbeat.*" not in a.publish
        assert f"{_NS}.tools.heartbeat.>" not in a.publish

    def test_agent_pod_may_publish_turn_completion(self) -> None:
        # resilience-task-07 router-mediated delivery: an agent signals TRUE turn completion by
        # publishing to ``agents.complete.{correlation_id}`` (the router awaits it to ack the durable
        # turn / re-route). the subject is keyed by correlation id (no agent segment), so the grant is
        # the wildcard ``agents.complete.*`` -- without it the completion publish is a NATS permissions
        # violation and every turn hangs to the caller's finalize timeout.
        a = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A)
        assert f"{_NS}.agents.complete.*" in a.publish

    def test_agent_pod_may_serve_only_its_own_in_process_tools(self) -> None:
        # an agent hosts its in-process tools (devx ``DevInProcessStrategy`` builtins, prod
        # ``ProdExternalPodsStrategy`` workspace + ``knowledge_drafts``) on its OWN ``AGENT_POD``
        # connection rather than as separate Tool Pods, so ``_agent_pod`` grants the tool-serving
        # subjects -- but every one is scoped to the AUTHENTICATED ``agent_id`` subtree
        # (``tools.{internal,probe,heartbeat}.{agent_id}.>``), NOT the spoofable connect-name pod id.
        # the in-process server runs under the ``{agent_id}.{instance}`` composite pod-id, so its
        # ``tools.internal.{agent_id}.{instance}`` subscription nests under the granted subtree while
        # a peer agent can NEVER be granted a subject under this agent's identity.
        a = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A)
        # its own in-process tool server: register (point) + heartbeat scoped to its own agent subtree.
        assert f"{_NS}.tools.register" in a.publish
        assert f"{_NS}.tools.heartbeat.{_AGENT_A}.>" in a.publish
        # receives the registry's proxied calls + reachability probes for its OWN agent subtree only.
        assert f"{_NS}.tools.internal.{_AGENT_A}.>" in a.subscribe
        assert f"{_NS}.tools.probe.{_AGENT_A}.>" in a.subscribe
        # the grant is scoped on the AUTHENTICATED agent id, never the spoofable connect-name pod id:
        # the legacy single-token pod-scoped grants are GONE (closing the connect-name wiretap).
        assert f"{_NS}.tools.internal.{_POD_A}" not in a.subscribe
        assert f"{_NS}.tools.probe.{_POD_A}" not in a.subscribe
        assert f"{_NS}.tools.heartbeat.{_POD_A}" not in a.publish
        # and never the registry's router-wide ``>`` (that belongs to the trusted router alone) nor
        # the single-token ``.*``.
        assert f"{_NS}.tools.internal.>" not in a.subscribe
        assert f"{_NS}.tools.internal.*" not in a.subscribe
        assert f"{_NS}.tools.probe.>" not in a.subscribe
        assert f"{_NS}.tools.probe.*" not in a.subscribe
        assert f"{_NS}.tools.heartbeat.>" not in a.publish
        assert f"{_NS}.tools.heartbeat.*" not in a.publish
        # a PEER agent's subtree is a DIFFERENT subject -> never granted in either direction, so one
        # tenant can never be granted a subject under a peer agent's identity (the core invariant).
        b = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_B, pod_id=_POD_B)
        assert f"{_NS}.tools.internal.{_AGENT_B}.>" not in a.subscribe
        assert f"{_NS}.tools.probe.{_AGENT_B}.>" not in a.subscribe
        assert f"{_NS}.tools.heartbeat.{_AGENT_B}.>" not in a.publish
        assert f"{_NS}.tools.internal.{_AGENT_A}.>" not in b.subscribe
        assert f"{_NS}.tools.probe.{_AGENT_A}.>" not in b.subscribe
        assert f"{_NS}.tools.heartbeat.{_AGENT_A}.>" not in b.publish

    def test_agent_in_process_tool_subjects_are_independent_of_the_connect_name(self) -> None:
        # SAME authenticated agent, DIFFERENT connect-name pod ids (replicas): the in-process tool
        # grants are identical because they are scoped on the agent subtree, NOT the pod id. this is
        # what lets a tenant set any connect ``name`` (even a peer pod's) without ever shifting its
        # tool grant onto a peer agent's identity -- the connect name simply does not feed these.
        p1 = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_1)
        p2 = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_VICTIM)
        tool_subjects = lambda perm: sorted(  # noqa: E731 -- terse local for the assertion
            s
            for s in _all_subjects(perm)
            if ".tools.internal." in s or ".tools.probe." in s or ".tools.heartbeat." in s
        )
        assert (
            tool_subjects(p1)
            == tool_subjects(p2)
            == [
                f"{_NS}.tools.heartbeat.{_AGENT_A}.>",
                f"{_NS}.tools.internal.{_AGENT_A}.>",
                f"{_NS}.tools.probe.{_AGENT_A}.>",
            ]
        )

    def test_agent_pod_may_publish_its_own_tool_call_audit(self) -> None:
        # serving builtins in-process means the in-process tool server emits the baseline
        # ``tool.call`` audit envelope on every dispatch (mirrors ``_tool_pod``). audit
        # non-repudiation is REQUIRED on this platform, so the grant is mandatory -- without
        # it the actor/audit row for an agent-served tool call would be silently dropped.
        a = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A)
        assert f"{_NS}.audit.tool.call" in a.publish

    def test_agent_pod_may_publish_the_channel_default_engagement_resolve(self) -> None:
        # the runtime resolves the conversation channel's default engagement at the tool-call stamp
        # seam. the resolve SOFT-FAILS to "unbound" on any transport error, so a missing grant does
        # not surface as a refused publish -- it surfaces later, and elsewhere, as a scan refused for
        # a missing engagement that was in fact configured. that silence is why the grant is pinned
        # here. READ only: the write half of the rail is asserted absent directly below.
        a = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A)
        assert f"{_NS}.hub.channel.engagement.default.resolve" in a.publish

    def test_agent_pod_may_not_publish_the_retired_channel_default_write_subjects(self) -> None:
        # this assertion is INVERTED from what it once was, deliberately. the ``.set`` / ``.clear``
        # NATS write rail was retired: binding and clearing a channel's default engagement is an
        # OPERATOR action and now rides the hub's authenticated admin HTTP surface, so no responder
        # subscribes to either subject anywhere on the platform. the grants outlived the rail and
        # were left overstating what an agent pod may do -- a least-privilege gap, closed here.
        # an agent NEVER writes a channel's engagement binding; it only reads it. if a future
        # feature needs an agent-driven write, it gets its OWN subject and its own justification,
        # never these back.
        a = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A)
        assert f"{_NS}.hub.channel.engagement.default.set" not in a.publish
        assert f"{_NS}.hub.channel.engagement.default.clear" not in a.publish
        # and not smuggled in under any other principal or verb either.
        for principal in Principal:
            granted = _all_subjects(_build(principal))
            assert f"{_NS}.hub.channel.engagement.default.set" not in granted, principal
            assert f"{_NS}.hub.channel.engagement.default.clear" not in granted, principal

    def test_retired_channel_default_write_subject_constructors_are_gone(self) -> None:
        # the grant and the constructor are removed TOGETHER: a surviving ``Subjects`` constructor is
        # a standing invitation to re-add the grant (or to publish on a dead subject from elsewhere).
        # no back-compat alias -- when the rail went, the API went with it.
        assert not hasattr(Subjects, "hub_channel_engagement_default_set")
        assert not hasattr(Subjects, "hub_channel_engagement_default_clear")
        # the READ half stays: the runtime genuinely calls it.
        assert Subjects.hub_channel_engagement_default_resolve().path == (
            f"{_NS}.hub.channel.engagement.default.resolve"
        )

    def test_agent_pod_holds_proxy_assertion_nonce_bucket(self) -> None:
        # the in-process tool server verifies the proxy's body-bound assertion under enforce
        # and records single-use nonces in this KV bucket (mirrors ``_tool_pod``); without the
        # grant the agent could not serve its own builtins under enforced connection-auth.
        a = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A)
        assert f"{_NS}-proxy_assertion_nonces" in kv_bucket_names(a)

    def test_agent_pod_heartbeat_and_reregister_are_agent_scoped(self) -> None:
        # the agent_id leads heartbeat / reregister subjects as the
        # AUTHENTICATED segment (token-hash->DB), so a pod can publish
        # heartbeats and receive reregister nudges only under its OWN
        # agent -- it cannot forge a peer agent's heartbeat (B2) nor hold
        # a peer agent's reregister grant.
        a = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A)
        assert f"{_NS}.agents.heartbeat.{_AGENT_A}.{_POD_A}" in a.publish
        assert f"{_NS}.agents.reregister_request.{_AGENT_A}.{_POD_A}" in a.subscribe
        # a peer agent's heartbeat / reregister subjects are NOT granted.
        b = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_B, pod_id=_POD_B)
        assert f"{_NS}.agents.heartbeat.{_AGENT_B}.{_POD_B}" not in a.publish
        assert f"{_NS}.agents.reregister_request.{_AGENT_B}.{_POD_B}" not in a.subscribe
        assert f"{_NS}.agents.heartbeat.{_AGENT_A}.{_POD_A}" not in b.publish
        # the spoofable-pod-only legacy single-segment grant is gone, and no
        # wildcard heartbeat publish exists.
        assert f"{_NS}.agents.heartbeat.{_POD_A}" not in a.publish
        assert f"{_NS}.agents.heartbeat.*" not in a.publish
        assert f"{_NS}.agents.heartbeat.>" not in a.publish
        assert f"{_NS}.agents.reregister_request.{_POD_A}" not in a.subscribe


class TestBootCompleteness:
    @pytest.mark.parametrize(
        ("principal", "required"),
        [
            (
                Principal.AGENT_POD,
                [
                    f"{_NS}.hub.handshake",
                    f"{_NS}.agents.register",
                    f"{_NS}.tools.discover",
                    f"{_NS}.tools.call",
                    f"{_NS}.hub.secrets.request",
                ],
            ),
            # hub.object.resolve is boot-critical for the Path-2 consume path: a
            # consuming tool that cannot publish it fails closed at the bus and
            # the whole resolve->stream capability goes silently inert.
            (
                Principal.TOOL_POD,
                [
                    f"{_NS}.tools.register",
                    f"{_NS}.hub.jwks",
                    f"{_NS}.hub.object.resolve",
                    # a pod granted a datasource reaches it over this subject and nothing
                    # else; without it the grant is materialized and the query is refused
                    # at the connection.
                    f"{_NS}.datasource.*.query",
                ],
            ),
            (
                # the router forward grant is ``tools.internal.>`` (not ``.*``) so it spans BOTH
                # single-token tool pods and two-token agent in-process pods.
                Principal.REGISTRY,
                [f"{_NS}.tools.call", f"{_NS}.tools.internal.>", f"{_NS}.hub.jwks"],
            ),
            (Principal.HUB, [f"{_NS}.hub.handshake", f"{_NS}.hub.jwks", f"{_NS}.hub.secrets.request"]),
            (Principal.GATEWAY, [f"{_NS}.gateway.completion", f"{_NS}.gateway.embedding"]),
            (Principal.CHANNEL_ADAPTER, [f"{_NS}.channels.deliver.*", f"{_NS}.hub.channel.installs"]),
        ],
    )
    def test_boot_critical_subjects_present(self, principal: Principal, required: list[str]) -> None:
        present = set(_all_subjects(_build(principal)))
        missing = [s for s in required if s not in present]
        assert not missing, f"{principal}: missing boot-critical {missing}"

    def test_tool_pod_subscribes_its_internal_call_subject(self) -> None:
        # without this the tool pod registers but never RECEIVES a proxied call.
        perm = build_permissions(Principal.TOOL_POD, pod_id=_POD_X)
        assert f"{_NS}.tools.internal.{_POD_X}" in perm.subscribe

    def test_tool_pod_may_reach_l3(self) -> None:
        """a tool pod that owns a schema must be able to query it.

        it holds a hub-minted token, its provider namespace resolves to an
        ``ns_`` schema, and it is granted read and write on that namespace --
        and without these subjects it cannot send the request at all, so every
        one of those is inert. this is what "a tool pod can use tables" rests
        on.

        :return: none
        :rtype: None
        """
        perm = build_permissions(Principal.TOOL_POD, pod_id=_POD_X)
        assert str(Subjects.l3_query()) in perm.publish
        assert str(Subjects.l3_batch()) in perm.publish

    def test_tool_pod_may_run_a_transaction(self) -> None:
        """a write of more than one statement needs the tx subjects too.

        granted as the same wildcard the agent pod holds, over all six ops,
        rather than six literals that drift apart from ``Subjects.l3_tx``.

        :return: none
        :rtype: None
        """
        perm = build_permissions(Principal.TOOL_POD, pod_id=_POD_X)
        assert any(p.endswith(".l3.tx.*") for p in perm.publish)

    def test_tool_pod_may_query_a_datasource(self) -> None:
        """a tool pod granted a datasource reaches it the way it reaches L3.

        the hub answers ``{ns}.datasource.{name}.query`` for every datasource it
        serves, verifies the forwarded hub-minted token at the door, and evaluates
        the pod's own grant on the datasource namespace. the request names no
        principal, so holding the subject buys reach and never authority -- which
        is what makes a wildcard over the NAME segment safe: the pod can ask about
        any datasource, and the hub refuses every one it was not granted.

        publish only. the hub subscribes; a pod never answers a datasource query.

        :return: none
        :rtype: None
        """
        pod = build_permissions(Principal.TOOL_POD, pod_id=_POD_X)
        assert f"{_NS}.datasource.*.query" in pod.publish
        assert f"{_NS}.datasource.*.query" not in pod.subscribe
        assert f"{_NS}.datasource.*.query" in _build(Principal.HUB).subscribe

    def test_tool_pod_may_call_a_tool(self) -> None:
        """a tool pod granted a platform tool reaches it the way an agent does.

        the registry answers ``{ns}.tools.call``, verifies the forwarded hub-minted
        token and the per-call proof of possession at the door, and evaluates the
        pod's OWN ``tool.call`` grant on the tool's namespace. the request names no
        principal, so holding the subject buys reach and never authority: the pod
        may ask for any tool, and the registry refuses every one it was not granted.

        publish only. the registry subscribes; a pod never answers a tool call on
        this subject -- it answers proxied calls on its own internal subject.

        :return: none
        :rtype: None
        """
        pod = build_permissions(Principal.TOOL_POD, pod_id=_POD_X)
        assert str(Subjects.tools_call()) in pod.publish
        assert str(Subjects.tools_call()) not in pod.subscribe
        assert str(Subjects.tools_call()) in _build(Principal.REGISTRY).subscribe

    def test_tool_pod_may_handshake_for_a_token_of_its_own(self) -> None:
        """a tool pod writes its OWN state on a hub-minted token from this handshake.

        acting on a call, it forwards that call's identity token. writing its
        own durable state, it presents its provisioned key and receives a
        short-lived hub-minted token, the same handshake an agent pod performs.
        this grant is what carries a tool pod's L3 access.

        :return: none
        :rtype: None
        """
        perm = build_permissions(Principal.TOOL_POD, pod_id=_POD_X)
        assert str(Subjects.hub_handshake()) in perm.publish

    def test_the_handshake_grant_is_not_a_second_verification_scheme(self) -> None:
        """the pod's SELF-minted token is not interchangeable with a hub-minted one.

        a self-minted token is verified against the pod's own stored public key;
        a hub-minted one against the hub's JWKS. keeping one scheme for one
        question is why the pod asks the hub rather than the broker gaining a
        second verification path.

        :return: none
        :rtype: None
        """
        perm = build_permissions(Principal.TOOL_POD, pod_id=_POD_X)
        assert str(Subjects.hub_jwks()) in perm.publish
        assert str(Subjects.hub_handshake()) in perm.publish

    def test_tool_pod_may_publish_namespace_discover(self) -> None:
        """a tool pod asks the broker which namespaces its caller can see.

        the grant is a question about the CALLER and nothing else: the subject
        reads the calling agent, customer and acting user off a forwarded token
        it verifies, and the request carries no field naming any of them. that
        is what makes it safe to hold, and the reason it has to be -- an answer
        is a customer's whole namespace inventory, so a subject that accepted a
        self-asserted customer would hand any pod a map of another customer's
        estate. the hub answers it.
        """
        pod = build_permissions(Principal.TOOL_POD, pod_id=_POD_X)
        assert f"{_NS}.namespace.discover" in pod.publish
        assert f"{_NS}.namespace.discover" in _build(Principal.HUB).subscribe
        # read-only for the pod: it asks, it never answers.
        assert f"{_NS}.namespace.discover" not in pod.subscribe

    def test_audit_anonymize_is_agent_publish_hub_subscribe(self) -> None:
        # erasure of an agent's own audit rows: the AGENT pod asks, forwarding its identity
        # token; the hub answers, touching only rows whose agent is the verified caller. the
        # subject sits under ``hub.`` and never under ``audit.``, because the durable audit
        # stream captures ``{ns}.audit.>`` and would persist a request there as an event.
        agent = _build(Principal.AGENT_POD)
        hub = _build(Principal.HUB)
        assert f"{_NS}.hub.audit.anonymize" in agent.publish
        assert f"{_NS}.hub.audit.anonymize" in hub.subscribe
        assert f"{_NS}.hub.audit.anonymize" not in agent.subscribe
        # a tool pod granted WRITE on an agent's data erases a person from that agent's data, so it
        # asks too; the subject names no owner, and the hub refuses any owner the pod's
        # ``declared_agent_data`` does not grant it write on. it only asks, never answers.
        tool_pod = build_permissions(Principal.TOOL_POD, pod_id=_POD_X)
        assert f"{_NS}.hub.audit.anonymize" in tool_pod.publish
        assert f"{_NS}.hub.audit.anonymize" not in tool_pod.subscribe

    def test_geo_layers_reloaded_is_tool_pod_publish_hub_subscribe(self) -> None:
        # a tool pod that registered platform geography layers reports a reloaded generation; the
        # hub answers, moving a layer only for the pod owning its provider namespace. an agent owns
        # no provider space, so it is not granted the subject at all.
        tool_pod = build_permissions(Principal.TOOL_POD, pod_id=_POD_X)
        hub = _build(Principal.HUB)
        agent = _build(Principal.AGENT_POD)
        assert f"{_NS}.hub.geo.layers.reloaded" in tool_pod.publish
        assert f"{_NS}.hub.geo.layers.reloaded" not in tool_pod.subscribe
        assert f"{_NS}.hub.geo.layers.reloaded" in hub.subscribe
        assert f"{_NS}.hub.geo.layers.reloaded" not in agent.publish

    def test_engagement_scope_resolve_grant_is_pod_publish_hub_subscribe(self) -> None:
        # engagement scope (consumer A of the §2 keystone): the consuming tool pod
        # PUBLISHES the resolve (forwarding the invoking agent's identity token);
        # the hub SUBSCRIBES to answer. mirrors the hub_object_resolve split.
        pod = build_permissions(Principal.TOOL_POD, pod_id=_POD_X)
        assert f"{_NS}.hub.engagement.scope" in pod.publish
        hub = _build(Principal.HUB)
        assert f"{_NS}.hub.engagement.scope" in hub.subscribe
        # it is read-only for the pod: no agent-side commit twin exists (unlike
        # objects), and the pod never subscribes the scope subject.
        assert f"{_NS}.hub.engagement.scope" not in pod.subscribe

    def test_channel_engagement_default_resolve_is_agent_publish_hub_subscribe(self) -> None:
        # the agent runtime PUBLISHES this at the tool-call stamp seam to resolve the
        # conversation channel's default engagement; ChannelDefaultResponder, in the hub,
        # SUBSCRIBES to answer it. The hub half was missing: latent only because the hub
        # connects as a static nats.conf user holding `>`, so this table is never consulted
        # for it. The moment the hub moves onto callout-minted permissions -- the path
        # agents already use -- the subscription is refused and the responder goes dark,
        # and the symptom is a scan refused for a "missing" engagement that IS configured.
        agent = _build(Principal.AGENT_POD)
        assert f"{_NS}.hub.channel.engagement.default.resolve" in agent.publish
        hub = _build(Principal.HUB)
        assert f"{_NS}.hub.channel.engagement.default.resolve" in hub.subscribe
        # read-only for the agent: `.set` / `.clear` are operator actions on the hub's
        # authenticated admin HTTP surface, so no responder serves them over NATS.
        assert f"{_NS}.hub.channel.engagement.default.resolve" not in agent.subscribe
        assert f"{_NS}.hub.channel.engagement.default.set" not in agent.publish

    def test_agent_can_reach_l3_and_gateway(self) -> None:
        perm = _build(Principal.AGENT_POD)
        assert f"{_NS}.l3.query" in perm.publish
        assert f"{_NS}.l3.tx.*" in perm.publish
        assert f"{_NS}.gateway.completion" in perm.publish
        # receives its streamed tokens on its OWN agent-scoped subject (W1);
        # a bare `gateway.stream.*` wildcard would let it sniff every other
        # customer's in-flight token stream.
        assert f"{_NS}.gateway.stream.{_AGENT_1}.*" in perm.subscribe
        assert f"{_NS}.gateway.stream.*" not in perm.subscribe
        # and it publishes its hub token stream only under its own agent id
        # (hub.stream W1): a bare `hub.stream.*` publish grant would let it
        # forge/inject tokens onto a peer's in-flight request.
        assert f"{_NS}.hub.stream.{_AGENT_1}.*" in perm.publish
        assert f"{_NS}.hub.stream.*" not in perm.publish

    def test_infra_stream_wildcards_are_two_segment(self) -> None:
        # gateway.stream / hub.stream / reregister now carry a leading
        # AUTHENTICATED {agent_id}; the infra-side grants MUST widen to a
        # two-segment wildcard (`*.*`) or they silently stop matching the
        # agent-scoped subjects the moment auth is enforced.
        hub = _build(Principal.HUB)
        assert f"{_NS}.hub.stream.*.*" in hub.subscribe
        assert f"{_NS}.hub.stream.*" not in hub.subscribe
        assert f"{_NS}.agents.reregister_request.*.*" in hub.publish
        assert f"{_NS}.agents.reregister_request.*" not in hub.publish
        gw = _build(Principal.GATEWAY)
        assert f"{_NS}.gateway.stream.*.*" in gw.publish
        assert f"{_NS}.gateway.stream.*" not in gw.publish

    def test_gateway_can_publish_object_resolve_for_media(self) -> None:
        # a completion carrying a media reference forwards the CALLER'S identity
        # token to the hub's object-resolve responder; the gateway must be able to
        # publish it. mirrors the tool-pod consume grant, and the hub answers it.
        gw = _build(Principal.GATEWAY)
        assert f"{_NS}.hub.object.resolve" in gw.publish
        hub = _build(Principal.HUB)
        assert f"{_NS}.hub.object.resolve" in hub.subscribe
        # the gateway produces no objects: it never publishes the commit twin.
        assert f"{_NS}.hub.object.commit" not in gw.publish

    def test_registry_forward_wildcard_spans_two_token_agent_pods(self) -> None:
        # the registry router forwards proxied calls / probes to ``tools.internal.{pod_id}``. once an
        # agent in-process pod registers under the two-token ``{agent_id}.{instance}`` composite, a
        # single-token ``tools.internal.*`` grant would silently STOP matching it (a ToolReadinessTimeout
        # at boot). the router grant MUST be the ``>`` subtree, which spans both pod shapes.
        reg = _build(Principal.REGISTRY)
        assert f"{_NS}.tools.internal.>" in reg.publish
        assert f"{_NS}.tools.probe.>" in reg.publish
        assert f"{_NS}.tools.internal.*" not in reg.publish
        assert f"{_NS}.tools.probe.*" not in reg.publish
        # the heartbeat monitor subscribes the global ``>`` so it sees both pod shapes' heartbeats.
        assert f"{_NS}.tools.heartbeat.>" in reg.subscribe


class TestFailClosed:
    def test_agent_pod_requires_both_ids(self) -> None:
        with pytest.raises(ValueError):
            build_permissions(Principal.AGENT_POD)
        with pytest.raises(ValueError):
            build_permissions(Principal.AGENT_POD, agent_id="a")  # missing pod_id

    def test_tool_pod_requires_pod_id(self) -> None:
        with pytest.raises(ValueError):
            build_permissions(Principal.TOOL_POD)

    @pytest.mark.parametrize(
        "principal",
        [Principal.REGISTRY, Principal.HUB, Principal.GATEWAY, Principal.CHANNEL_ADAPTER],
    )
    def test_infra_requires_conn_id(self, principal: Principal) -> None:
        with pytest.raises(ValueError):
            build_permissions(principal)


class TestNamespaceBinding:
    def test_subjects_follow_the_bound_namespace(self) -> None:
        set_default_namespace("prod7")
        perm = build_permissions(Principal.TOOL_POD, pod_id=_POD_1)
        assert f"{'prod7'}.tools.internal.{_POD_1}" in perm.subscribe
        assert all(
            s.startswith("prod7.")
            or s == CROSS_PLATFORM_CACHE_INVALIDATE
            or s == SERVER_USER_INFO_SUBJECT
            or s.startswith("_INBOX_")
            for s in _all_subjects(perm)
        )


class TestHitlApprovalBrokerGrants:
    """the exploit HITL approval broker needs three new subject grants."""

    def test_agent_pod_may_publish_approval_record(self) -> None:
        """an agent pausing on a gated tool records the pending marker with the hub."""
        a = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A)
        assert f"{_NS}.hub.approval.record" in a.publish

    def test_hub_subscribes_both_approval_subjects(self) -> None:
        """the hub broker responder receives record + resolve requests."""
        h = build_permissions(Principal.HUB, conn_id="hub-1")
        assert f"{_NS}.hub.approval.record" in h.subscribe
        assert f"{_NS}.hub.approval.resolve" in h.subscribe

    def test_channel_adapter_may_publish_approval_resolve(self) -> None:
        """the router (in the sandboxed adapter) forwards operator replies to resolve."""
        c = build_permissions(Principal.CHANNEL_ADAPTER, conn_id="chan-1")
        assert f"{_NS}.hub.approval.resolve" in c.publish

    def test_agent_pod_cannot_publish_resolve_nor_adapter_record(self) -> None:
        """least-privilege: neither principal holds the OTHER's approval grant."""
        a = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A)
        c = build_permissions(Principal.CHANNEL_ADAPTER, conn_id="chan-1")
        assert f"{_NS}.hub.approval.resolve" not in a.publish
        assert f"{_NS}.hub.approval.record" not in c.publish


class TestHitlSessionControlGrants:
    """the owner-routed session control plane a live display is driven over."""

    #: tool-name NODES, exactly as a pod's own declaration holds them and as the auth
    #: callout hands them to ``build_permissions``. NOT registered tool namespace names: those
    #: are minted at REGISTRATION and these grants at CONNECT, so a leaf here would describe a
    #: value the mint never sees. ``test_hitl_family_is_keyed_on_the_owned_node.py`` holds the
    #: property that the node and its canonical ``tools.`` form derive one family.
    ALPHA = "scrape-zone_alpha"
    BETA = "scrape-zone_beta"

    def _pod(self, *namespaces: str) -> PrincipalPermissions:
        return build_permissions(Principal.TOOL_POD, pod_id=_POD_X, tool_namespaces=namespaces)

    def test_pod_subscribes_an_exact_family_literal_per_authorized_tool(self) -> None:
        """each owned NODE becomes one grant, and only the key is wildcarded."""
        perm = self._pod(self.ALPHA, self.BETA)
        for name in (self.ALPHA, self.BETA):
            expected = str(Subjects.forward_scoped_wildcard(Subjects.hitl_forward_family(name)))
            assert expected in perm.subscribe
            family_token = expected.removeprefix(f"{_NS}.forward.").removesuffix(".*")
            assert set(family_token) <= set("0123456789abcdef")

    def test_pod_grant_admits_the_subject_that_tool_actually_serves(self) -> None:
        """the grant and the subject the pod subscribes are built from one derivation.

        pinned because the two are minted in different processes -- the hub mints
        the grant from the tool-pods row, the pod builds the subject from the tool
        it serves -- and a mismatch fails as a silent timeout, not an error.
        """
        perm = self._pod(self.ALPHA)
        served = Subjects.forward_scoped(Subjects.hitl_forward_family(self.ALPHA), "session-42")
        granted = str(Subjects.forward_scoped_wildcard(Subjects.hitl_forward_family(self.ALPHA)))
        assert granted in perm.subscribe
        assert served.path.rsplit(".", 1)[0] == granted.rsplit(".", 1)[0]

    def test_pod_grant_is_hex_only_for_a_hostile_tool_name(self) -> None:
        """an unvalidated tool name must not inject a wildcard INTO A GRANT.

        ``ToolManifestEntry.name`` is a bare ``str`` and ``_validate_manifest``
        checks only that ``pod_id`` and ``tools`` are non-empty, so the hostile
        value reaches the mint. sanitization would not close this: both
        sanitizers replace dots and nothing else, so a ``>`` here would widen
        the pod's own grant to a subtree.
        """
        perm = self._pod("evil name.* > ")
        granted = [s for s in perm.subscribe if s.startswith(f"{_NS}.forward.")]
        assert granted, "the pod holds no session grant at all"
        # Every family the pod is granted, not a fixed number of them: one session is
        # owner-routed twice (its control plane and its display stream ride separate families
        # so they cannot collide on one queue group), and a tally here would assert the count
        # rather than the property, then rot the next time the shape changes.
        for subject in granted:
            family_token = subject.removeprefix(f"{_NS}.forward.").removesuffix(".*")
            assert set(family_token) <= set("0123456789abcdef"), subject
            assert len(family_token) == 64, subject
            for illegal in (" ", "*", ">"):
                assert illegal not in family_token

    def test_pod_without_authorized_tools_gets_no_session_grant(self) -> None:
        """fail closed: a pod serving no human session holds nothing on this family."""
        perm = build_permissions(Principal.TOOL_POD, pod_id=_POD_X)
        assert not [s for s in _all_subjects(perm) if s.startswith(f"{_NS}.forward.")]

    def test_pod_holds_neither_the_coarse_subtree_nor_a_peer_family(self) -> None:
        """the whole point of the family segment: one tool's grant is not another's."""
        perm = self._pod(self.ALPHA)
        assert f"{_NS}.forward.>" not in _all_subjects(perm)
        assert f"{_NS}.forward.*.*" not in _all_subjects(perm)
        assert f"{_NS}.forward.*" not in _all_subjects(perm)
        peer = str(Subjects.forward_scoped_wildcard(Subjects.hitl_forward_family(self.BETA)))
        assert peer not in _all_subjects(perm)

    def test_pod_may_publish_only_its_own_streams_downward(self) -> None:
        """the stream grants name the pod's OWN tool digest and OWN pod id, not a wildcard.

        Without this the whole grant could be deleted and every recorded run would still
        pass: the pipe's own suites are the two ``pytest.mark.integration`` files, which
        ``./scripts/test.sh -m "not integration"`` deselects, so a deselected file appearing
        in the evidence is a path list rather than proof anything executed.
        """
        perm = self._pod(self.ALPHA)
        down = [s for s in perm.publish if ".pipe." in s]
        up = [s for s in perm.subscribe if ".pipe." in s]
        assert down, f"a tool pod may not publish any stream; it holds {list(perm.publish)}"
        assert up, f"a tool pod may not subscribe any stream; it holds {list(perm.subscribe)}"
        for subject in (*down, *up):
            segments = subject.split(".")
            # {ns}.pipe.{tool_digest}.{pod_id}.{nonce}.{direction}: only the nonce may be a
            # wildcard. a wildcard tool digest would let this pod serve another tool's
            # streams, and a wildcard pod id would let it answer for a sibling replica.
            assert segments[2] != "*", f"{subject} wildcards the tool digest"
            assert segments[3] != "*", f"{subject} wildcards the pod id"

    def test_pod_cannot_touch_another_tools_streams(self) -> None:
        """the grant for one authorized tool does not render the digest of another."""
        from threetears.nats.subjects import Subjects

        other = Subjects.hitl_forward_family("some-other-provider")
        foreign_digest = str(Subjects.forward_scoped_wildcard(other)).split(".")[2]
        perm = self._pod(self.ALPHA)
        assert not [s for s in (*perm.publish, *perm.subscribe) if foreign_digest in s]

    def test_pod_serves_but_never_originates(self) -> None:
        """the owner answers on the requester's reply inbox under ``allow_responses``."""
        perm = self._pod(self.ALPHA)
        assert not [s for s in perm.publish if s.startswith(f"{_NS}.forward.")]
        assert perm.allow_responses is True

    def test_pod_holds_the_bucket_the_display_claim_actually_materialises(self) -> None:
        """the grant is the bucket that MATERIALISES, prefix applied exactly once.

        ``KVLease`` returns a bucket-name SUFFIX and ``kv_bucket`` layers the
        connection's ``{ns}-`` over it, so the pair composes to ``{ns}-leases``.
        The failure this pins is not an error: a pod that cannot open the bucket
        cannot claim at all: ``KVLease.acquire`` defers the bucket open to
        first use, and that open raises ``KvError`` after a JetStream timeout.
        (``lease=None`` is a different path -- a platform passing no lease --
        and it is what serves a display unclaimed.)

        ``tests/enforcement/test_kv_bucket_grant_naming.py`` holds the same
        property against the live default rather than a literal, so a change to
        either side alone fails there; this asserts the concrete string a
        reviewer can read.
        """
        perm = self._pod(self.ALPHA)
        assert f"{_NS}-leases" in kv_bucket_names(perm)

    def test_hub_may_call_every_family_and_serve_none(self) -> None:
        """one hub connection fronts every tool, so its family segment is a wildcard.

        it stays a two-token pattern: the unscoped one-token forward family is
        granted to no principal at all, and this does not reach it.
        """
        hub = build_permissions(Principal.HUB, conn_id="hub-1")
        assert f"{_NS}.forward.*.*" in hub.publish
        assert f"{_NS}.forward.*" not in hub.publish
        assert f"{_NS}.forward.>" not in _all_subjects(hub)
        assert not [s for s in hub.subscribe if s.startswith(f"{_NS}.forward.")]

    def test_unscoped_forward_family_is_granted_to_nobody(self) -> None:
        """the shape with only a key digest in it has no grantable discriminator.

        every principal is checked, not just the two this scope touches: the
        chunk exists because that family shipped ungranted, and re-granting it
        coarsely anywhere would undo the family segment entirely.
        """
        for principal in Principal:
            for subject in _all_subjects(_build(principal)):
                assert subject not in {f"{_NS}.forward.>", f"{_NS}.forward.*"}, principal


class TestDurableAnswerGrants:
    """A responder must still be able to answer after the refresh that recycled its connection.

    ``allow_responses`` is scoped to the connection that RECEIVED a request, and NATS has no in-band
    re-auth, so a correct credential refresh destroys the right to answer an in-flight call. In
    production that surfaced as a scan finishing with exit 0 and 68KB of results and a permissions
    violation on the publish. The grants below replace that per-request right with a standing one on
    a subject the responder names with its OWN identity -- derived from ids the auth-callout already
    holds, so every refresh re-mints the same grant and reconnects stop mattering.

    What makes them safe is what a standing grant on the requester's inbox tree would not have been:
    each is confined to one principal's own subtree, so no responder can forge an answer into another
    responder's in-flight call.
    """

    def test_tool_pod_may_deliver_only_under_its_own_pod_id(self) -> None:
        perm = build_permissions(Principal.TOOL_POD, pod_id=_POD_1)
        assert str(Subjects.tools_result_pod_wildcard(_POD_1)) in perm.publish
        assert str(Subjects.tools_result_pod_wildcard(_POD_2)) not in perm.publish

    def test_no_principal_may_publish_the_whole_result_family(self) -> None:
        """the forgery hole this design exists to avoid, checked across every principal.

        a coarse ``tools.result.>`` publish grant anywhere would let its holder answer for any pod,
        which is the cross-customer response injection that ruled out the inbox-tree grant.
        """
        for principal in Principal:
            for subject in _all_subjects(_build(principal)):
                assert subject != f"{_NS}.tools.result.>", principal
                assert subject != f"{_NS}.tools.result.*.*", principal

    def test_agent_pod_may_deliver_only_under_its_own_authenticated_agent(self) -> None:
        """an in-process tool server answers under the ``{agent_id}.{instance}`` composite pod-id.

        the auth-callout knows the authenticated agent, never the per-replica instance, so the grant
        is the agent subtree -- the same shape ``tools.internal.{agent_id}.>`` already uses, and for
        the same reason: a connect-name-scoped grant would be spoofable.
        """
        perm = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_1, pod_id=_POD_1)
        assert str(Subjects.tools_result_agent_subtree(_AGENT_1)) in perm.publish
        assert str(Subjects.tools_result_agent_subtree(_AGENT_2)) not in perm.publish

    def test_only_the_registry_may_answer_agents(self) -> None:
        """the reply family is the router's to publish and nobody else's.

        the wildcard is granted because one registry connection fronts every agent and there is no
        per-connection list of agent ids to mint literals from; it is contained at the proxy, which
        publishes only to a subject naming the call's VERIFIED agent id. a POD holding it would be
        able to forge an answer into any agent's in-flight call.
        """
        registry = build_permissions(Principal.REGISTRY, conn_id="reg-1")
        assert str(Subjects.tools_reply_wildcard()) in registry.publish
        for principal in Principal:
            if principal is Principal.REGISTRY:
                continue
            for subject in _all_subjects(_build(principal)):
                assert not subject.startswith(f"{_NS}.tools.reply."), f"{principal}: {subject}"

    def test_every_collector_declares_the_stream_that_carries_the_answer(self) -> None:
        """delivery rides JetStream, and a JS grant is pinned per DECLARED stream name.

        the failure of omitting one is not a denial that says so: an ungranted JetStream operation
        blocks to its deadline, which reads as an unreachable broker rather than a missing grant.
        Only a COLLECTOR needs the grant -- the registry, which collects a tool pod's result, and
        the agent pod, which collects the registry's reply. A publisher needs none: a JetStream
        publish is a core publish whose acknowledgement lands on the publisher's own inbox.
        """
        from threetears.nats.result_delivery import result_stream_name

        stream = result_stream_name()
        for principal in (Principal.AGENT_POD, Principal.REGISTRY):
            declared = [r.name for r in _build(principal).js_resources if r.kind is JsResourceKind.STREAM]
            assert stream in declared, principal
        tool_pod = [r.name for r in _build(Principal.TOOL_POD).js_resources if r.kind is JsResourceKind.STREAM]
        assert stream not in tool_pod

    def test_the_result_grant_survives_a_refresh_because_it_is_derived(self) -> None:
        """re-minting for the same principal yields byte-identical grants.

        this is the whole mechanism: the grant is a function of ids the auth-callout resolves at
        connect, not of anything about the connection, so the reconnect that a credential refresh
        performs cannot invalidate it. if a future edit made any of these depend on connection state,
        the answer would start dying at the refresh again -- silently, and only for long calls.
        """
        first = build_permissions(Principal.TOOL_POD, pod_id=_POD_1)
        second = build_permissions(Principal.TOOL_POD, pod_id=_POD_1, conn_id="a-different-connection")
        result_grants = str(Subjects.tools_result_pod_wildcard(_POD_1))
        assert result_grants in first.publish
        assert result_grants in second.publish


class TestPrincipalRoster:
    """Every :class:`Principal` member is REFERENCED, and the roster covers every L2 process.

    Two processes that run L2 collections had no member at all -- ``agent_router``, which owns
    ``PodAffinityCollection`` (sticky conversation-to-pod routing, ``L3 = None``), and
    ``dataset_executor``. With no member there is no legal ``kv_key_scope_for`` value for them, so
    no scope can be wired and no grant can be expressed: two downstream shards blocked on it.
    """

    def test_the_two_missing_l2_processes_have_members(self) -> None:
        assert Principal.AGENT_ROUTER.value == "agent_router"
        assert Principal.DATASET_EXECUTOR.value == "dataset_executor"

    @pytest.mark.parametrize("principal", list(Principal))
    def test_every_member_resolves_to_a_permission_set(self, principal: Principal) -> None:
        # four members (HUB, REGISTRY, GATEWAY, CHANNEL_ADAPTER) were referenced nowhere outside
        # this module because those processes connect as static users. adoption means each is
        # exercised here and each answers a scope -- a grant-surface change, not a migration.
        assert _build(principal).publish

    @pytest.mark.parametrize("principal", list(Principal))
    def test_every_member_answers_a_legal_scope(self, principal: Principal) -> None:
        ids = {k: v for k, v in _IDS[principal].items() if k in {"agent_id", "pod_id"}}
        if principal is Principal.AGENT_POD:
            ids.pop("pod_id")  # refused for this principal: the pod id is spoofable here
        scope = kv_key_scope_for(principal, **ids)
        assert scope


class TestConsumerServiceKvKeyScope:
    """A consumer service that owns its own NATS account can now produce a legal scope.

    ``kv_key_scope_for`` dispatches on :class:`Principal`, which enumerates 3tears' own bus
    identities -- a third-party service is none of them, so before this producer existed the
    only options were adopting a member it is not (``hub``, which an operator reading bucket
    keys will misattribute) or inventing an id in 3tears' namespace (``tool_pod``, where config
    drift silently orphans the L2 cache). The ``svc-`` prefix keeps the value out of the
    ``Principal`` grant surface entirely: no resolver answers for it, and none has to.
    """

    def test_scope_leads_with_svc_and_carries_the_service_name(self) -> None:
        assert kv_key_scope_for_service("scriob") == "svc-scriob"

    def test_same_name_yields_the_same_scope_across_calls(self) -> None:
        # the scope is the SHARING boundary: replicas of one service must land on one scope
        # or L2 stops being a cross-pod cache, so the name is the only input.
        assert kv_key_scope_for_service("scriob") == kv_key_scope_for_service("scriob")

    def test_two_service_names_yield_two_scopes(self) -> None:
        assert kv_key_scope_for_service("scriob") != kv_key_scope_for_service("other")

    def test_empty_service_name_raises(self) -> None:
        # fail closed, same rule as the pod principals: a bare fallback value would land
        # every misconfigured service on one shared scope.
        with pytest.raises(ValueError, match="non-empty"):
            kv_key_scope_for_service("")

    def test_name_outside_the_scope_grammar_raises(self) -> None:
        # sanitizing instead of raising would be non-injective ("a.b" and "a-b" collapse),
        # and two services sharing one scope is the outcome key scoping exists to prevent.
        with pytest.raises(ValueError, match="grammar|match"):
            kv_key_scope_for_service("my.service")

    @pytest.mark.parametrize("principal", list(Principal))
    def test_no_service_scope_can_shadow_a_bare_principal(self, principal: Principal) -> None:
        assert kv_key_scope_for_service(principal.value) != principal.value


class TestDeadletterGrant:
    """``subscribe`` and ``subscribe_typed`` both deadletter by default, and nobody was granted it.

    A callback that raises -- and, on the typed path, a payload that fails validation -- republishes
    to ``{ns}.deadletter.{original_subject}``. Grepping this module for ``deadletter`` used to
    return nothing: the registry was incidentally covered by its static ``aibots.>``, and a
    callout-minted agent pod was not, so the one diagnostic a failing handler leaves behind was
    itself refused and dropped at WARNING inside that same failing handler.
    """

    @pytest.mark.parametrize("principal", list(Principal))
    def test_every_principal_may_publish_the_deadletter_subtree(self, principal: Principal) -> None:
        assert f"{_NS}.deadletter.>" in _build(principal).publish, principal

    @pytest.mark.parametrize("principal", list(Principal))
    def test_no_principal_subscribes_the_deadletter_subtree(self, principal: Principal) -> None:
        # producing a deadletter is not authority to READ everyone else's failed payloads, which
        # carry the full body of whatever was rejected.
        assert f"{_NS}.deadletter.>" not in _build(principal).subscribe, principal


class TestScopedCollectionsGrant:
    """The shared bucket is granted per-principal, and every other bucket is left alone.

    ``{ns}-collections`` is held by six principals at once, and ``BaseCollection.l2_key`` writes
    ``{scope}.{table}.{body}`` into it. Nothing else on the platform writes a scope prefix, so the
    narrowing is per-resource opt-in: applied uniformly it would deny every read on ``checkpoints``
    (its own separate ``l2_key``, keyed by thread id), ``{ns}_agent_config``, ``{ns}-epochs`` and
    the rest -- and a refused JetStream request is never answered, so that failure arrives as a
    ten-second deadline rather than as an error anyone can read.
    """

    def _collections(self, principal: Principal) -> object:
        resources = [r for r in _build(principal).js_resources if r.name == _COLLECTIONS]
        assert len(resources) == 1, f"{principal} declares {len(resources)} collections resources"
        return resources[0]

    # DERIVED FROM THE ENUM, never a hand-written roster. The list this replaced happened to
    # name all eight members, which is exactly why it was dangerous: a ninth principal would
    # have been added to ``Principal`` and silently left out of the one assertion that says its
    # state is granted at all. ``_build`` indexes ``_IDS`` directly, so a member added without
    # representative ids fails here with a ``KeyError`` rather than being skipped.
    @pytest.mark.parametrize("principal", list(Principal))
    def test_the_collections_grant_carries_the_scope_the_process_writes(self, principal: Principal) -> None:
        resource = self._collections(principal)
        ids = {k: v for k, v in _IDS[principal].items() if k in {"agent_id", "pod_id"}}
        if principal is Principal.AGENT_POD:
            ids.pop("pod_id")
        # pinned as a PAIR: the mint and the writing process must derive the identical value from
        # the identical inputs, or the principal reads and writes keys its own grant does not cover
        # -- and that failure is a deadline, not a refusal anyone sees.
        assert resource.scope == kv_key_scope_for(principal, **ids)  # type: ignore[attr-defined]

    @pytest.mark.parametrize("principal", list(Principal))
    def test_no_other_bucket_is_scoped(self, principal: Principal) -> None:
        # the data-versions bucket and the agent-config cache are each narrowed to ONE whole key
        # rather than to a key-scope prefix, and that key IS the principal's key, so the narrowing
        # matches the reads it exists for; ``TestDataVersionKeyGrant`` and
        # ``TestAgentConfigIsReadOnlyForItsPod`` pin those grants on their own.
        for resource in _build(principal).js_resources:
            if resource.kind is not JsResourceKind.KV_BUCKET or resource.name in {_COLLECTIONS, _DATA_VERSIONS}:
                continue
            if resource.capability is JsCapability.KV_KEY_READ:
                assert principal is Principal.AGENT_POD and resource.name == f"{_NS}_agent_config", resource
                continue
            if resource.capability is JsCapability.KV_OWNER_KEYS:
                # the platform's shared pod buckets, whose keys lead with the owner scope;
                # ``TestSharedPodBucketsAreOwnerScoped`` pins them on their own.
                assert principal in _POD_PRINCIPALS and resource.name in _SHARED_OWNER_KEYED, resource
                continue
            assert resource.scope is None, f"{principal}: {resource.name} would deny its own reads"
            # a pod binds what the hub declared and holds no stream verb; an infra identity that
            # declares what it opens keeps the management grant. The registry binds the epoch
            # bucket the hub declared, read only, to follow the access tables' generations.
            binds_only = principal in _POD_PRINCIPALS or (
                principal is Principal.REGISTRY and resource.name == f"{_NS}-epochs"
            )
            expected = JsCapability.KV_BUCKET_KEYS if binds_only else JsCapability.FULL
            assert resource.capability is expected, f"{principal}: {resource.name}"

    def test_the_registry_holds_the_bucket_its_own_source_of_truth_collection_runs_on(self) -> None:
        """``HeartbeatCollection`` is ``L3 = None``, so an ungranted bucket is DATA LOSS.

        ``registry/server.py`` calls ``collection_registry.configure(l2_client=nc)`` and then builds
        a ``HeartbeatCollection``. That collection has no L3 tier, so L2 *is* its store: a key the
        grant does not cover is not a cache miss that falls through to a database, it is a
        heartbeat that was never written. It worked only because the static ``registry`` NATS user
        carries ``$KV.>``, which ``coll-task-05b`` removes.
        """
        assert _COLLECTIONS in kv_bucket_names(_build(Principal.REGISTRY))

    def test_only_the_hub_declares_and_no_pod_ever_does(self) -> None:
        """``declare`` is CREATE + UPDATE, and ``UPDATE`` is a read-all primitive here.

        ``coll-task-04a`` makes hub bootstrap the canonical declarer, so it needs both verbs to
        reconcile ``allow_direct: true``. On a SHARED stream ``UPDATE`` also sets ``republish`` and
        ``sources``, which mirror every key -- every principal's -- onto a subject the caller names.
        So it is bound to the declaring identity alone rather than folded into the read capability.
        """
        declaring = [
            (principal, r.name)
            for principal in Principal
            for r in _build(principal).js_resources
            if capability_declares(r.capability)
        ]
        assert declaring == [(Principal.HUB, _COLLECTIONS)], declaring
        for principal in _POD_PRINCIPALS:
            for resource in _build(principal).js_resources:
                assert not capability_declares(resource.capability), f"{principal}: {resource.name}"

    def test_a_pod_whose_id_is_not_a_uuid_cannot_be_granted_at_all(self) -> None:
        """GRANT-10, at the resolver rather than at the wire.

        The scope is the isolation boundary, so it is derived only from an authenticated uuid. A
        pod that cannot produce one gets no permission set -- fail closed -- rather than a grant
        narrowed to a scope it will never write.

        BOTH pod principals reach this now. The agent pod was the only one while a tool pod held no
        collections grant; ``coll-task-07c`` gives it one, so it derives a scope and inherits the
        same fence -- exactly as the note that used to stand here predicted.
        """
        with pytest.raises(ValueError, match="uuid"):
            build_permissions(Principal.AGENT_POD, agent_id="agent-A", pod_id=_POD_A)
        with pytest.raises(ValueError, match="uuid"):
            build_permissions(Principal.TOOL_POD, pod_id="pod-A")
        with pytest.raises(ValueError, match="uuid"):
            kv_key_scope_for(Principal.TOOL_POD, pod_id="pod-A")

    def test_two_agent_pods_never_share_a_collections_scope(self) -> None:
        a = self._collections(Principal.AGENT_POD)
        b = [
            r
            for r in build_permissions(Principal.AGENT_POD, agent_id=_AGENT_2, pod_id=_POD_2).js_resources
            if r.name == _COLLECTIONS
        ][0]
        assert a.scope != b.scope  # type: ignore[attr-defined]

    def test_replicas_of_one_agent_share_one_scope(self) -> None:
        # the scope is the SHARING boundary, not the connection: replicas of one principal must
        # resolve to one scope or L2 stops being a cross-pod cache.
        one = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_1, pod_id=_POD_1)
        two = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_1, pod_id=_POD_2)
        scopes = [[r.scope for r in perm.js_resources if r.name == _COLLECTIONS][0] for perm in (one, two)]
        assert scopes[0] == scopes[1]


class TestStateIsStandardForEveryPrincipal:
    """L2 stops being a per-class privilege: every principal holds the bucket and the subject.

    **Why this is asserted over the ENUM rather than over a list of principals.** The property
    is *"every principal"*, and a roster written by hand asserts *"these principals"* -- which
    reads identically until somebody adds a ninth member, at which point the roster silently
    stops covering it and the test still passes. ``list(Principal)`` is the only spelling that
    fails when the thing it describes grows.

    **And it must not become a per-row opt-in.** A flag saying "this principal may hold L2"
    reintroduces an ordering problem that has no fix: grants are minted at CONNECT and rows are
    written at REGISTRATION, so the flag is read before the row that sets it exists. It is also
    a new place for exactly the divergence this closes to grow back. So the grant is
    unconditional, and :meth:`test_no_optional_argument_gates_either_grant` pins that by
    building each principal with nothing but the ids it cannot resolve without.

    Both halves fail SILENTLY when missing, in opposite ways:

    - no bucket grant, and the JetStream op blocks to its deadline and reads as an unreachable
      broker rather than as a refusal;
    - no invalidation subject, and a write is never announced (publish) or never heard
      (subscribe), so a peer replica serves a value it has already replaced -- and for the
      several ``L3 = None`` collections on this platform, a key that no longer resolves is lost
      data rather than a cache miss.
    """

    @pytest.mark.parametrize("principal", list(Principal))
    def test_every_principal_declares_the_shared_collections_bucket(self, principal: Principal) -> None:
        """the bucket every ``BaseCollection`` L2 writes into, held by every connection class.

        :param principal: the connection identity class under test
        :ptype principal: Principal
        :return: none
        :rtype: None
        """
        assert _COLLECTIONS in kv_bucket_names(_build(principal)), principal

    @pytest.mark.parametrize("principal", list(Principal))
    def test_every_principal_may_write_the_shared_collections_bucket(self, principal: Principal) -> None:
        """read-only would be a grant that cannot save a row.

        ``writable`` is the ``$KV.`` PUBLISH half; the read rides the scoped
        ``$JS.API.DIRECT.GET`` tail the scoped capability already mints, so the two are
        genuinely separable and holding one is not holding the other.

        :param principal: the connection identity class under test
        :ptype principal: Principal
        :return: none
        :rtype: None
        """
        resources = [r for r in _build(principal).js_resources if r.name == _COLLECTIONS]
        assert len(resources) == 1, f"{principal} declares {len(resources)} collections resources"
        assert resources[0].writable, principal

    @pytest.mark.parametrize("principal", list(Principal))
    def test_every_principal_publishes_the_invalidation_subject(self, principal: Principal) -> None:
        """without it a principal cannot announce its own write.

        :param principal: the connection identity class under test
        :ptype principal: Principal
        :return: none
        :rtype: None
        """
        assert CROSS_PLATFORM_CACHE_INVALIDATE in _build(principal).publish, principal

    @pytest.mark.parametrize("principal", list(Principal))
    def test_every_principal_subscribes_the_invalidation_subject(self, principal: Principal) -> None:
        """without it a principal never learns of a peer's write.

        the direction that matters for the ``L3 = None`` collections: their L2 IS the store, so
        a stale L1 in front of it is not a cache miss anybody recovers from.

        :param principal: the connection identity class under test
        :ptype principal: Principal
        :return: none
        :rtype: None
        """
        assert CROSS_PLATFORM_CACHE_INVALIDATE in _build(principal).subscribe, principal

    @pytest.mark.parametrize("principal", list(Principal))
    def test_no_optional_argument_gates_either_grant(self, principal: Principal) -> None:
        """built with ONLY the ids it cannot resolve without, a principal still holds both.

        this is the "not a per-row opt-in" property, asserted rather than described. ``_IDS``
        carries no ``tool_namespaces`` and no ``coordination_buckets``, so this build is the
        one a caller that passes nothing optional produces -- and it must be indistinguishable,
        on these two grants, from any other.

        :param principal: the connection identity class under test
        :ptype principal: Principal
        :return: none
        :rtype: None
        """
        bare = build_permissions(principal, **_IDS[principal])
        assert _COLLECTIONS in kv_bucket_names(bare), principal
        assert CROSS_PLATFORM_CACHE_INVALIDATE in bare.publish, principal
        assert CROSS_PLATFORM_CACHE_INVALIDATE in bare.subscribe, principal


class TestTheToolPodHoldsTheSharedBucket:
    """``coll-task-07c``: a tool pod runs an L1+L2 collection, so it needs the bucket and the bus.

    Two grants, and neither is a detail. The bucket is scoped to ``tool_pods.id`` -- the pod's
    registry primary key, which is already its authenticated ``claims.sub``, configured once per
    deployment and therefore shared by every replica. A namespace-derived scope was designed and
    rejected: ``tool_namespace_id`` mints one row PER TOOL, is a pure function of manifest values
    the pod itself sends, is deliberately collision-inducing across pods, and no such row exists at
    connect time.

    The invalidation subject is the GLOBAL, deliberately un-namespaced one, and the exposure that
    buys (a cross-customer metadata firehose of table names + entity ids, and a fleet-wide eviction
    primitive whose ``origin`` is self-asserted) is RECORDED AS ACCEPTED for this landing rather
    than closed here -- ``origin`` authentication is a wire-protocol change across three repos and
    belongs to ``coll-task-08-invalidation-origin-auth``.
    """

    def _collections(self, pod_id: str) -> object:
        resources = [
            r for r in build_permissions(Principal.TOOL_POD, pod_id=pod_id).js_resources if r.name == _COLLECTIONS
        ]
        assert len(resources) == 1, f"tool pod declares {len(resources)} collections resources"
        return resources[0]

    def test_it_declares_the_collections_bucket_scoped_to_its_pod_id(self) -> None:
        resource = self._collections(_POD_1)
        assert resource.scope == kv_key_scope_for(Principal.TOOL_POD, pod_id=_POD_1)  # type: ignore[attr-defined]
        assert resource.capability is JsCapability.KV_SCOPED  # type: ignore[attr-defined]

    def test_the_bucket_is_writable_and_the_read_rides_the_scoped_direct_get(self) -> None:
        """TP-01: ``$KV.`` is PUBLISH authority only; the read is the scoped ``DIRECT.GET`` tail.

        Nothing in nats-py ever SUBSCRIBES a ``$KV.`` subject, so a subscribe grant there confers
        no read and hands the holder every write's full value.
        """
        assert self._collections(_POD_1).writable is True  # type: ignore[attr-defined]
        perm = build_permissions(Principal.TOOL_POD, pod_id=_POD_1)
        assert not [s for s in perm.subscribe if s.startswith("$KV")], perm.subscribe

    def test_two_tool_pods_never_share_a_scope(self) -> None:
        """TP-03, the isolation half."""
        assert self._collections(_POD_A).scope != self._collections(_POD_B).scope  # type: ignore[attr-defined]

    def test_replicas_of_one_tool_pod_share_one_scope(self) -> None:
        """TP-03, the sharing half: the scope is the sharing boundary, not the connection.

        ``tool_pods.id`` is configured once per DEPLOYMENT, so two connections presenting it are
        two replicas of one pod and must land on one scope -- otherwise L2 stops being a cross-pod
        cache at all. A different ``conn_id`` must not move the scope.
        """
        one = self._collections(_POD_1)
        two = [
            r
            for r in build_permissions(Principal.TOOL_POD, pod_id=_POD_1, conn_id="a-second-replica").js_resources
            if r.name == _COLLECTIONS
        ][0]
        assert one.scope == two.scope  # type: ignore[attr-defined]

    def test_it_holds_the_global_invalidation_subject_on_both_directions(self) -> None:
        """TP-02. Without the publish it cannot announce a write; without the subscribe it never
        learns of one, and its L1 serves a value another replica has already replaced."""
        perm = build_permissions(Principal.TOOL_POD, pod_id=_POD_1)
        assert CROSS_PLATFORM_CACHE_INVALIDATE in perm.publish
        assert CROSS_PLATFORM_CACHE_INVALIDATE in perm.subscribe


class TestDirectlyBoundBuckets:
    """A bucket opened with ``js.key_value`` carries NO namespace prefix, and must still be granted.

    ``NatsClient.kv_bucket`` layers ``{ns}-`` onto every suffix it is given; a direct
    ``js.key_value(bucket=...)`` does not, so the name in the grant has to be the verbatim wire
    name. Two processes take that route today and both are pinned here, because a bucket a process
    opens without a grant is a JetStream call that blocks to its deadline and reads as an
    unreachable broker rather than as a refusal.

    ``{ns}_agent_config`` on the router is evidence-ledger bug 21: the resolver did not declare it
    while ``agent_router/proxy.py`` bound it directly, and the hub's static-conf generator was
    carrying a compensating entry with "reported upstream" written beside it.
    """

    def test_the_registry_declares_its_unprefixed_tool_catalog(self) -> None:
        assert "tool_catalog" in kv_bucket_names(_build(Principal.REGISTRY))

    def test_the_router_declares_its_unprefixed_catalog(self) -> None:
        assert "agent_router_catalog" in kv_bucket_names(_build(Principal.AGENT_ROUTER))

    def test_the_router_declares_the_agent_config_bucket_it_binds(self) -> None:
        assert f"{_NS}_agent_config" in kv_bucket_names(_build(Principal.AGENT_ROUTER))

    def test_the_router_holds_agent_config_read_only(self) -> None:
        """Config Source-of-Truth: the router is a READER of cluster config, never a writer.

        the hub's ``agents`` table is the source and this bucket is a hot cache over it, written only by
        the hub's admin endpoints. A KV read is a ``$JS.API`` request rather than a ``$KV.``
        publish, so withholding write authority costs the router nothing -- and a write grant it
        does not need is a write grant a bug can use.
        """
        resource = [r for r in _build(Principal.AGENT_ROUTER).js_resources if r.name == f"{_NS}_agent_config"][0]
        assert resource.writable is False

    def test_no_agent_config_publish_subject_is_minted_for_the_router(self) -> None:
        """the read-only decision, asserted on the EMITTED grant rather than on the record."""
        emitted = [
            r for r in _build(Principal.AGENT_ROUTER).js_resources if r.kind is JsResourceKind.KV_BUCKET and r.writable
        ]
        assert f"{_NS}_agent_config" not in {r.name for r in emitted}


class TestDeclaredCoordinationBuckets:
    """an agent's own coordination buckets, and the prefix that fences them.

    A product pod needs KV buckets the platform does not define: a replay ledger, an
    idempotency store, a lockout counter, a ticket store, a handle store, a quota
    counter. They cannot share one bucket -- TTL is a bucket property and those six
    want six different ones, two of them incompatible (a quota cell must never expire;
    a challenge nonce must expire in minutes).

    So the agent declares them. The property that makes declaring them SAFE is that the
    declared string is a SUFFIX and never a whole bucket name: every grant is composed
    under ``{ns}-{scope}-``, where ``scope`` comes from ``kv_key_scope_for`` on the
    AUTHENTICATED agent id. An agent that declares ``collections`` is granted its OWN
    ``{ns}-agent_pod-<hex>-collections`` and gets no closer to the shared one, so no
    declaration -- honest, mistaken or hostile -- can reach another agent's buckets or
    the platform's.
    """

    def test_a_declared_bucket_is_granted_under_the_agents_own_prefix(self) -> None:
        """the grant is the composed name, never the declared suffix."""
        set_default_namespace(_NS)
        scope = kv_key_scope_for(Principal.AGENT_POD, agent_id=_AGENT_A)

        granted = kv_bucket_names(
            build_permissions(
                Principal.AGENT_POD,
                agent_id=_AGENT_A,
                pod_id=_POD_A,
                coordination_buckets=("entry_challenge_nonces",),
            )
        )

        assert f"{_NS}-{scope}-entry_challenge_nonces" in granted
        assert f"{_NS}-entry_challenge_nonces" not in granted

    def test_declaring_a_platform_bucket_name_reaches_only_the_agents_own(self) -> None:
        """
        THE escalation test. an agent naming a shared bucket must not be given it.

        ``collections`` is the one bucket every principal shares, and it carries every
        tenant's L2. A declaration mechanism that concatenated the namespace directly
        would hand an unscoped, whole-subtree write grant on it to any agent that asked.
        """
        set_default_namespace(_NS)
        scope = kv_key_scope_for(Principal.AGENT_POD, agent_id=_AGENT_A)

        permissions = build_permissions(
            Principal.AGENT_POD,
            agent_id=_AGENT_A,
            pod_id=_POD_A,
            coordination_buckets=("collections", "checkpoints", "epochs"),
        )
        granted = kv_bucket_names(permissions)

        # its own three, named for what it asked
        for asked in ("collections", "checkpoints", "epochs"):
            assert f"{_NS}-{scope}-{asked}" in granted

        # and the shared ones are held only where they were ALREADY held, at the
        # capability the platform block declares -- not widened by the declaration
        shared_collections = [r for r in permissions.js_resources if r.name == _COLLECTIONS]
        assert len(shared_collections) == 1
        assert shared_collections[0].scope == scope, (
            "the shared collections bucket must still be SCOPE-narrowed; a declaration "
            "has widened it to the whole subtree"
        )

    def test_one_agent_cannot_reach_another_agents_declared_bucket(self) -> None:
        """two agents declaring the same suffix resolve to two different buckets."""
        set_default_namespace(_NS)

        a = kv_bucket_names(
            build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A, coordination_buckets=("nonces",))
        )
        b = kv_bucket_names(
            build_permissions(Principal.AGENT_POD, agent_id=_AGENT_B, pod_id=_POD_B, coordination_buckets=("nonces",))
        )

        assert not (set(a) & set(b) & {name for name in a if name.endswith("-nonces")})

    def test_declaring_nothing_changes_nothing(self) -> None:
        """the parameter is additive: omitting it reproduces the previous grant set exactly."""
        set_default_namespace(_NS)

        without = kv_bucket_names(build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A))
        empty = kv_bucket_names(
            build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A, coordination_buckets=())
        )

        assert without == empty

    @pytest.mark.parametrize(
        "suffix",
        [
            "has.a.dot",  # a dot is a subject separator; the stream name is ONE token
            "has space",
            "has/slash",
            "$KV",
            "wild*card",
            "sub>tree",
            "",
        ],
    )
    def test_a_suffix_outside_the_grammar_is_refused_at_mint(self, suffix: str) -> None:
        """
        refuse LOUDLY rather than dropping the entry.

        a dropped entry produces a pod whose KV calls block to their deadline and
        report an unreachable broker -- the exact silent failure this module's own
        comments say costs a day to diagnose. raising happens at mint, names the
        offending value, and cannot be mistaken for a network fault.
        """
        set_default_namespace(_NS)

        with pytest.raises(ValueError, match="coordination bucket"):
            build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A, coordination_buckets=(suffix,))

    def test_more_buckets_than_the_cap_are_refused(self) -> None:
        """
        the count is bounded, because each entry materialises one JetStream stream.

        the prefix makes an over-broad declaration harmless to OTHER principals; it does
        not make it free. a bound is what stops one agent's manifest creating streams
        without limit.
        """
        set_default_namespace(_NS)
        too_many = tuple(f"bucket{n}" for n in range(MAX_COORDINATION_BUCKETS + 1))

        with pytest.raises(ValueError, match="coordination bucket"):
            build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A, coordination_buckets=too_many)

    def test_only_an_agent_pod_may_declare_them(self) -> None:
        """
        no other principal expands the list, so a stray claim cannot widen an infra grant.

        every resolver takes the same keyword arguments, which is what makes a
        misrouted claim conceivable; this pins that only the one resolver reads it.
        """
        set_default_namespace(_NS)

        granted = kv_bucket_names(
            build_permissions(Principal.TOOL_POD, pod_id=_POD_A, conn_id="conn-1", coordination_buckets=("nonces",))
        )

        assert not any(name.endswith("-nonces") for name in granted)

    def test_a_declared_bucket_is_writable_and_unscoped_within_itself(self) -> None:
        """
        the BUCKET is the boundary, so its keys need no scope segment.

        this is what lets 3tears's own coordination primitives -- ReplayGuard,
        DistributedCounter, SingleUseTicketStore -- be used unchanged: none of them
        writes a scope prefix into its keys, and a scoped grant would match none of
        them and deny every call.
        """
        set_default_namespace(_NS)
        scope = kv_key_scope_for(Principal.AGENT_POD, agent_id=_AGENT_A)

        permissions = build_permissions(
            Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A, coordination_buckets=("nonces",)
        )
        declared = [r for r in permissions.js_resources if r.name == f"{_NS}-{scope}-nonces"]

        assert len(declared) == 1
        assert declared[0].kind is JsResourceKind.KV_BUCKET
        assert declared[0].writable is True
        assert declared[0].scope is None
        assert not capability_declares(declared[0].capability), (
            "a pod must not hold a DECLARING capability; STREAM.UPDATE is a read-all "
            "primitive on any stream it is held against"
        )


def _subject_matches(pattern: str, subject: str) -> bool:
    """NATS subject matching: ``*`` spans one token, ``>`` spans one or more trailing tokens.

    :param pattern: the granted subject pattern
    :ptype pattern: str
    :param subject: the concrete subject under test
    :ptype subject: str
    :return: whether the pattern admits the subject
    :rtype: bool
    """
    pattern_tokens = pattern.split(".")
    subject_tokens = subject.split(".")
    result = len(pattern_tokens) == len(subject_tokens)
    for index, token in enumerate(pattern_tokens):
        if token == ">":
            result = index < len(subject_tokens)
            break
        if index >= len(subject_tokens):
            result = False
            break
        if token != "*" and token != subject_tokens[index]:
            result = False
            break
    return result


def _minted_publish(permissions: PrincipalPermissions) -> list[str]:
    """the ``pub.allow`` list of a REAL minted user JWT for ``permissions``.

    :param permissions: the resolved permissions to mint
    :ptype permissions: PrincipalPermissions
    :return: every subject the principal may publish to
    :rtype: list[str]
    """

    token = mint_user_jwt(
        account_seed=generate_account_seed(),
        user_public_key="UTESTUSERPUBLICKEY",
        permissions=permissions,
        name="data-version-test",
        expires_in_seconds=300,
    )
    payload = token.split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    allow: list[str] = claims["nats"]["pub"]["allow"]
    return allow


class TestDataVersionKeyGrant:
    """a pod reads and watches ITS OWN data-version entry, and nobody else's, and writes none.

    The hub owns ``{ns}-data-versions``: it creates the bucket and writes one key per owning
    principal -- the agent id for an agent pod, the ``tool_pods.id`` for a tool pod, each as 32
    lowercase hex characters. A pod waiting for an upgrade's all-clear watches its own key.

    **The key is ONE whole token, so the grant is a literal, never a ``{key}.>`` tail.** A
    ``>`` needs at least one more token, so ``$KV.{b}.{key}.>`` matches nothing the hub writes --
    and an ungranted JetStream request is never answered: it blocks to its deadline and reads as
    an unreachable broker. The watch is pinned the same way: a consumer created with the filter
    in its SUBJECT (``CONSUMER.CREATE.{stream}.{name}.{filter}``), which the server checks against
    the body. An unnamed ``CONSUMER.CREATE.{stream}`` carries its filter only in the body, where
    it could name every key in the bucket, so it is not granted.
    """

    _POD_PRINCIPAL_IDS = (
        (Principal.AGENT_POD, _AGENT_1),
        (Principal.TOOL_POD, _POD_1),
    )

    def _resource(self, permissions: PrincipalPermissions) -> JsResource:
        found = [r for r in permissions.js_resources if r.name == _DATA_VERSIONS]
        assert len(found) == 1, f"expected one {_DATA_VERSIONS} resource, found {len(found)}"
        return found[0]

    def test_the_bucket_name_and_its_suffix(self) -> None:
        assert DATA_VERSIONS_BUCKET_SUFFIX == "data-versions"
        assert data_versions_bucket_name() == _DATA_VERSIONS
        assert data_versions_bucket_name("prod7") == "prod7-data-versions"

    def test_the_key_is_the_owner_uuid_as_32_hex(self) -> None:

        owner = uuid.UUID(_AGENT_1)
        assert data_version_kv_key(owner) == owner.hex
        assert data_version_kv_key(_AGENT_1) == owner.hex
        assert data_version_kv_key(_AGENT_1.upper()) == owner.hex
        assert len(owner.hex) == 32 and "." not in owner.hex

    def test_a_non_uuid_owner_is_refused(self) -> None:
        with pytest.raises(ValueError, match="uuid"):
            data_version_kv_key("agent-A")

    @pytest.mark.parametrize(("principal", "owner"), _POD_PRINCIPAL_IDS, ids=lambda v: str(v))
    def test_each_pod_principal_holds_its_own_key_read_only(self, principal: Principal, owner: str) -> None:
        resource = self._resource(_build(principal))
        assert resource.kind is JsResourceKind.KV_BUCKET
        assert resource.capability is JsCapability.KV_KEY_READ
        assert resource.scope == data_version_kv_key(owner)
        assert resource.writable is False

    def test_replicas_of_one_agent_watch_one_key(self) -> None:
        one = self._resource(build_permissions(Principal.AGENT_POD, agent_id=_AGENT_1, pod_id=_POD_1))
        two = self._resource(build_permissions(Principal.AGENT_POD, agent_id=_AGENT_1, pod_id=_POD_2))
        assert one.scope == two.scope == data_version_kv_key(_AGENT_1)

    @pytest.mark.parametrize(("principal", "owner"), _POD_PRINCIPAL_IDS, ids=lambda v: str(v))
    def test_the_minted_grant_is_exactly_bind_read_and_named_watch(self, principal: Principal, owner: str) -> None:
        key = data_version_kv_key(owner)
        stream = f"KV_{_DATA_VERSIONS}"
        minted = [s for s in _minted_publish(_build(principal)) if stream in s or f"$KV.{_DATA_VERSIONS}" in s]
        assert sorted(minted) == sorted(
            [
                f"$JS.API.STREAM.INFO.{stream}",
                f"$JS.API.DIRECT.GET.{stream}.$KV.{_DATA_VERSIONS}.{key}",
                f"$JS.API.CONSUMER.CREATE.{stream}.*.$KV.{_DATA_VERSIONS}.{key}",
            ]
        )

    @pytest.mark.parametrize(("principal", "owner"), _POD_PRINCIPAL_IDS, ids=lambda v: str(v))
    def test_own_key_is_readable_and_watchable(self, principal: Principal, owner: str) -> None:
        key = data_version_kv_key(owner)
        stream = f"KV_{_DATA_VERSIONS}"
        allow = _minted_publish(_build(principal))
        for subject in (
            f"$JS.API.STREAM.INFO.{stream}",
            f"$JS.API.DIRECT.GET.{stream}.$KV.{_DATA_VERSIONS}.{key}",
            f"$JS.API.CONSUMER.CREATE.{stream}.watcher1.$KV.{_DATA_VERSIONS}.{key}",
        ):
            assert any(_subject_matches(p, subject) for p in allow), subject

    @pytest.mark.parametrize(("principal", "owner"), _POD_PRINCIPAL_IDS, ids=lambda v: str(v))
    def test_another_principals_key_is_neither_readable_nor_watchable(self, principal: Principal, owner: str) -> None:
        del owner
        stream = f"KV_{_DATA_VERSIONS}"
        allow = _minted_publish(_build(principal))
        for other in (_AGENT_2, _POD_2):
            foreign = data_version_kv_key(other)
            for subject in (
                f"$JS.API.DIRECT.GET.{stream}.$KV.{_DATA_VERSIONS}.{foreign}",
                f"$JS.API.CONSUMER.CREATE.{stream}.watcher1.$KV.{_DATA_VERSIONS}.{foreign}",
            ):
                assert not any(_subject_matches(p, subject) for p in allow), subject

    @pytest.mark.parametrize(("principal", "owner"), _POD_PRINCIPAL_IDS, ids=lambda v: str(v))
    def test_no_route_reaches_the_whole_bucket(self, principal: Principal, owner: str) -> None:
        del owner
        stream = f"KV_{_DATA_VERSIONS}"
        allow = _minted_publish(_build(principal))
        for subject in (
            f"$JS.API.CONSUMER.CREATE.{stream}",  # unnamed: the filter rides the body only
            f"$JS.API.CONSUMER.CREATE.{stream}.watcher1.$KV.{_DATA_VERSIONS}.>",
            f"$JS.API.CONSUMER.DURABLE.CREATE.{stream}.watcher1",
            f"$JS.API.STREAM.MSG.GET.{stream}",  # body-carried read
            f"$JS.API.DIRECT.GET.{stream}",  # get by sequence, body-carried
            f"$JS.API.STREAM.PURGE.{stream}",
            f"$JS.API.STREAM.SNAPSHOT.{stream}",
            f"$JS.API.STREAM.UPDATE.{stream}",
            f"$JS.API.STREAM.CREATE.{stream}",  # the hub creates the bucket; a pod only binds it
            f"$JS.API.STREAM.DELETE.{stream}",
        ):
            assert not any(_subject_matches(p, subject) for p in allow), subject

    @pytest.mark.parametrize(("principal", "owner"), _POD_PRINCIPAL_IDS, ids=lambda v: str(v))
    def test_the_pod_cannot_write_even_its_own_key(self, principal: Principal, owner: str) -> None:
        key = data_version_kv_key(owner)
        allow = _minted_publish(_build(principal))
        assert not any(_subject_matches(p, f"$KV.{_DATA_VERSIONS}.{key}") for p in allow)
        subscribe = _build(principal).subscribe
        assert not any(s.startswith(f"$KV.{_DATA_VERSIONS}") for s in subscribe)

    def test_the_key_grant_refuses_write_intent(self) -> None:
        with pytest.raises(ValueError, match="read"):
            JsResource(
                name=_DATA_VERSIONS,
                kind=JsResourceKind.KV_BUCKET,
                capability=JsCapability.KV_KEY_READ,
                scope=data_version_kv_key(_AGENT_1),
                writable=True,
            )

    def test_the_key_grant_refuses_a_multi_token_key(self) -> None:
        with pytest.raises(ValueError, match="token"):
            JsResource.kv_key_read(_DATA_VERSIONS, key="a.b")

    def test_the_key_grant_is_never_a_declaring_capability(self) -> None:
        assert not capability_declares(JsCapability.KV_KEY_READ)


class TestAgentTableGrants:
    """a tool pod granted an agent's data reaches that agent's L2 keys for exactly the granted tables.

    The owner stack a tool pod builds for an agent's data keys every entry
    ``{owner_scope}.{table}.{body}`` in the shared collections bucket, where ``owner_scope`` is the
    AGENT's own scope -- so the agent's L1 sees the pod's writes. The pod's own ``{pod_scope}.>``
    grant matches none of those keys, and an ungranted JetStream call blocks to its deadline rather
    than raising. So each granted table becomes a grant narrowed to ``{owner_scope}.{table}.>``:
    the direct read always, the ``$KV.`` publish only for a write grant, and nothing that reaches
    another table, another owner, or the whole bucket.
    """

    _OWNER = uuid.UUID(_AGENT_A)
    _OTHER_OWNER = uuid.UUID(_AGENT_B)
    _STREAM = f"KV_{_COLLECTIONS}"

    def _owner_scope(self, owner: uuid.UUID) -> str:
        return kv_key_scope_for(Principal.AGENT_POD, agent_id=owner)

    def _permissions(self, *grants: AgentTableGrant) -> PrincipalPermissions:
        return build_permissions(Principal.TOOL_POD, pod_id=_POD_1, agent_table_grants=grants)

    def _read_subject(self, owner: uuid.UUID, table: str) -> str:
        return f"$JS.API.DIRECT.GET.{self._STREAM}.$KV.{_COLLECTIONS}.{self._owner_scope(owner)}.{table}.row-1"

    def _write_subject(self, owner: uuid.UUID, table: str) -> str:
        return f"$KV.{_COLLECTIONS}.{self._owner_scope(owner)}.{table}.row-1"

    def test_the_owner_scope_is_the_agents_own_scope(self) -> None:
        """one derivation: the scope the SDK's owner stack keys on, from the same function."""
        grant = AgentTableGrant(owner_agent_id=self._OWNER, table="responses", writable=False)
        assert grant.owner_scope == kv_key_scope_for(Principal.AGENT_POD, agent_id=self._OWNER)

    def test_a_read_grant_mints_exactly_the_direct_read_of_that_table(self) -> None:
        scope = self._owner_scope(self._OWNER)
        minted = _minted_publish(
            self._permissions(AgentTableGrant(owner_agent_id=self._OWNER, table="responses", writable=False))
        )
        owner_subjects = [s for s in minted if scope in s]
        assert owner_subjects == [f"$JS.API.DIRECT.GET.{self._STREAM}.$KV.{_COLLECTIONS}.{scope}.responses.>"]

    def test_a_write_grant_mints_the_direct_read_and_the_kv_publish_of_that_table(self) -> None:
        scope = self._owner_scope(self._OWNER)
        minted = _minted_publish(
            self._permissions(AgentTableGrant(owner_agent_id=self._OWNER, table="responses", writable=True))
        )
        owner_subjects = sorted(s for s in minted if scope in s)
        assert owner_subjects == sorted(
            [
                f"$KV.{_COLLECTIONS}.{scope}.responses.>",
                f"$JS.API.DIRECT.GET.{self._STREAM}.$KV.{_COLLECTIONS}.{scope}.responses.>",
            ]
        )

    def test_each_table_grant_carries_its_own_bind(self) -> None:
        """a table-scoped record is complete on its own: it binds the stream it reads.

        The pod's own scope binds the same stream, so the minted list repeats the subject; NATS
        treats a repeated allow entry as one, and the list is left undeduplicated because the
        static-user confs are rendered from the same composition and must not shift under it.
        """
        permissions = self._permissions(AgentTableGrant(owner_agent_id=self._OWNER, table="responses", writable=True))
        for resource in permissions.js_resources:
            if resource.capability is JsCapability.KV_TABLE_SCOPED:
                assert f"$JS.API.STREAM.INFO.{self._STREAM}" in js_api_grants_for_stream(
                    resource.stream_name,
                    capability=resource.capability,
                    bucket=resource.name,
                    scope=resource.key_prefix,
                )

    def test_a_read_grant_cannot_write(self) -> None:
        allow = _minted_publish(
            self._permissions(AgentTableGrant(owner_agent_id=self._OWNER, table="responses", writable=False))
        )
        assert any(_subject_matches(p, self._read_subject(self._OWNER, "responses")) for p in allow)
        assert not any(_subject_matches(p, self._write_subject(self._OWNER, "responses")) for p in allow)

    def test_a_write_grant_reads_and_writes_its_table(self) -> None:
        allow = _minted_publish(
            self._permissions(AgentTableGrant(owner_agent_id=self._OWNER, table="responses", writable=True))
        )
        assert any(_subject_matches(p, self._read_subject(self._OWNER, "responses")) for p in allow)
        assert any(_subject_matches(p, self._write_subject(self._OWNER, "responses")) for p in allow)

    def test_another_table_of_the_same_owner_is_not_covered(self) -> None:
        """the agent's schema also holds its conversations and memory; a grant names tables."""
        allow = _minted_publish(
            self._permissions(AgentTableGrant(owner_agent_id=self._OWNER, table="responses", writable=True))
        )
        for table in ("conversations", "memories", "responses_archive"):
            assert not any(_subject_matches(p, self._read_subject(self._OWNER, table)) for p in allow), table
            assert not any(_subject_matches(p, self._write_subject(self._OWNER, table)) for p in allow), table

    def test_another_owners_table_of_the_same_name_is_not_covered(self) -> None:
        allow = _minted_publish(
            self._permissions(AgentTableGrant(owner_agent_id=self._OWNER, table="responses", writable=True))
        )
        assert not any(_subject_matches(p, self._read_subject(self._OTHER_OWNER, "responses")) for p in allow)
        assert not any(_subject_matches(p, self._write_subject(self._OTHER_OWNER, "responses")) for p in allow)

    def test_no_route_reaches_the_owners_whole_scope_or_the_whole_bucket(self) -> None:
        scope = self._owner_scope(self._OWNER)
        allow = _minted_publish(
            self._permissions(AgentTableGrant(owner_agent_id=self._OWNER, table="responses", writable=True))
        )
        for pattern in allow:
            assert pattern not in (
                f"$KV.{_COLLECTIONS}.>",
                f"$KV.{_COLLECTIONS}.{scope}.>",
                f"$JS.API.DIRECT.GET.{self._STREAM}.>",
                f"$JS.API.DIRECT.GET.{self._STREAM}.$KV.{_COLLECTIONS}.>",
                f"$JS.API.DIRECT.GET.{self._STREAM}.$KV.{_COLLECTIONS}.{scope}.>",
            ), pattern
        for subject in (
            f"$JS.API.CONSUMER.CREATE.{self._STREAM}",
            f"$JS.API.CONSUMER.CREATE.{self._STREAM}.w1.$KV.{_COLLECTIONS}.{scope}.responses.row-1",
            f"$JS.API.CONSUMER.DURABLE.CREATE.{self._STREAM}.w1",
            f"$JS.API.STREAM.MSG.GET.{self._STREAM}",
            f"$JS.API.DIRECT.GET.{self._STREAM}",
            f"$JS.API.STREAM.PURGE.{self._STREAM}",
            f"$JS.API.STREAM.UPDATE.{self._STREAM}",
            f"$JS.API.STREAM.CREATE.{self._STREAM}",
        ):
            assert not any(_subject_matches(p, subject) for p in allow), subject

    def test_the_resource_record_is_table_scoped_on_the_shared_bucket(self) -> None:
        permissions = self._permissions(AgentTableGrant(owner_agent_id=self._OWNER, table="responses", writable=False))
        granted = [r for r in permissions.js_resources if r.capability is JsCapability.KV_TABLE_SCOPED]
        assert len(granted) == 1
        assert granted[0].name == _COLLECTIONS
        assert granted[0].scope == self._owner_scope(self._OWNER)
        assert granted[0].table == "responses"
        assert granted[0].writable is False
        assert not capability_declares(granted[0].capability)

    def test_the_pods_own_scope_is_unchanged_by_a_grant(self) -> None:
        without = build_permissions(Principal.TOOL_POD, pod_id=_POD_1)
        with_grant = self._permissions(AgentTableGrant(owner_agent_id=self._OWNER, table="responses", writable=True))
        own = [r for r in with_grant.js_resources if r.capability is not JsCapability.KV_TABLE_SCOPED]
        assert tuple(own) == without.js_resources

    def test_no_grant_changes_nothing(self) -> None:
        assert self._permissions().js_resources == build_permissions(Principal.TOOL_POD, pod_id=_POD_1).js_resources

    @pytest.mark.parametrize("principal", [p for p in Principal if p is not Principal.TOOL_POD])
    def test_only_a_tool_pod_reads_the_grants(self, principal: Principal) -> None:
        """every resolver takes the same keywords; only the tool pod's widens on this one."""
        grant = AgentTableGrant(owner_agent_id=self._OWNER, table="responses", writable=True)
        with_grant = build_permissions(principal, **_IDS[principal], agent_table_grants=(grant,))
        assert with_grant.js_resources == _build(principal).js_resources

    @pytest.mark.parametrize("table", ["a.b", "resp*", "resp>", "", "has space", "$KV"])
    def test_a_table_that_is_not_one_subject_token_is_refused(self, table: str) -> None:
        """a dot would split the prefix and a wildcard would widen it; refuse at construction."""
        with pytest.raises(ValueError, match="table"):
            AgentTableGrant(owner_agent_id=self._OWNER, table=table, writable=False)

    def test_one_table_granted_twice_is_refused(self) -> None:
        """two answers to one question -- a read and a write -- must not silently merge."""
        with pytest.raises(ValueError, match="more than once"):
            self._permissions(
                AgentTableGrant(owner_agent_id=self._OWNER, table="responses", writable=False),
                AgentTableGrant(owner_agent_id=self._OWNER, table="responses", writable=True),
            )

    def test_a_table_scoped_resource_requires_its_table(self) -> None:
        with pytest.raises(ValueError, match="table"):
            JsResource(
                name=_COLLECTIONS,
                kind=JsResourceKind.KV_BUCKET,
                capability=JsCapability.KV_TABLE_SCOPED,
                scope=self._owner_scope(self._OWNER),
                writable=False,
            )

    def test_a_table_on_any_other_capability_is_refused(self) -> None:
        """a table recorded but not enforced reads as narrowing that is not there."""
        with pytest.raises(ValueError, match="table"):
            JsResource(
                name=_COLLECTIONS,
                kind=JsResourceKind.KV_BUCKET,
                capability=JsCapability.KV_SCOPED,
                scope=self._owner_scope(self._OWNER),
                writable=True,
                table="responses",
            )


class TestAgentBucketGrants:
    """a tool pod granted an agent's coordination bucket reaches that one bucket and nothing else.

    An agent's coordination buckets are its own, composed ``{ns}-{owner_scope}-{suffix}``; the bucket
    IS the isolation boundary and its keys carry no scope. So the grant is the WHOLE named bucket,
    through exactly the calls the 3tears KV client makes -- bind, get, a named key consumer (watch or
    listing) filtered inside the bucket, and for a write grant put, compare-and-set and delete -- and
    never a stream-admin verb (create, update, purge, delete) or an unnamed consumer, either of which
    reads or reshapes more than the bucket's keys.
    """

    _OWNER = uuid.UUID(_AGENT_A)
    _OTHER_OWNER = uuid.UUID(_AGENT_B)

    def _bucket(self, owner: uuid.UUID, suffix: str) -> str:
        return coordination_bucket_name(kv_key_scope_for(Principal.AGENT_POD, agent_id=owner), suffix, ns=_NS)

    def _permissions(self, *grants: AgentBucketGrant) -> PrincipalPermissions:
        return build_permissions(Principal.TOOL_POD, pod_id=_POD_1, agent_bucket_grants=grants)

    def test_the_bucket_is_the_one_the_owning_agent_is_granted(self) -> None:
        """one builder composes the name for the owner's own grant and for the tool pod's."""
        owner = build_permissions(
            Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A, coordination_buckets=("survey-quota-cells",)
        )
        grant = AgentBucketGrant(owner_agent_id=self._OWNER, suffix="survey-quota-cells", writable=True)
        assert grant.bucket_name(_NS) in kv_bucket_names(owner)
        assert grant.bucket_name(_NS) == f"{_NS}-agent_pod-{self._OWNER.hex}-survey-quota-cells"

    def test_a_read_grant_mints_exactly_bind_and_both_read_forms(self) -> None:
        bucket = self._bucket(self._OWNER, "respondent_resume_handles")
        minted = _minted_publish(
            self._permissions(
                AgentBucketGrant(owner_agent_id=self._OWNER, suffix="respondent_resume_handles", writable=False)
            )
        )
        assert sorted(s for s in minted if bucket in s) == sorted(
            [
                f"$JS.API.STREAM.INFO.KV_{bucket}",
                f"$JS.API.STREAM.MSG.GET.KV_{bucket}",
                f"$JS.API.DIRECT.GET.KV_{bucket}.$KV.{bucket}.>",
                f"$JS.API.CONSUMER.CREATE.KV_{bucket}.*.$KV.{bucket}.>",
            ]
        )

    def test_a_write_grant_adds_only_the_kv_publish(self) -> None:
        bucket = self._bucket(self._OWNER, "survey-quota-cells")
        minted = _minted_publish(
            self._permissions(AgentBucketGrant(owner_agent_id=self._OWNER, suffix="survey-quota-cells", writable=True))
        )
        assert sorted(s for s in minted if bucket in s) == sorted(
            [
                f"$KV.{bucket}.>",
                f"$JS.API.STREAM.INFO.KV_{bucket}",
                f"$JS.API.STREAM.MSG.GET.KV_{bucket}",
                f"$JS.API.DIRECT.GET.KV_{bucket}.$KV.{bucket}.>",
                f"$JS.API.CONSUMER.CREATE.KV_{bucket}.*.$KV.{bucket}.>",
            ]
        )

    def test_a_read_grant_cannot_write_and_a_write_grant_can(self) -> None:
        bucket = self._bucket(self._OWNER, "survey-quota-cells")
        read = _minted_publish(
            self._permissions(AgentBucketGrant(owner_agent_id=self._OWNER, suffix="survey-quota-cells", writable=False))
        )
        write = _minted_publish(
            self._permissions(AgentBucketGrant(owner_agent_id=self._OWNER, suffix="survey-quota-cells", writable=True))
        )
        assert not any(_subject_matches(p, f"$KV.{bucket}.cell-1") for p in read)
        assert any(_subject_matches(p, f"$KV.{bucket}.cell-1") for p in write)

    def test_another_owners_bucket_and_another_suffix_are_not_covered(self) -> None:
        allow = _minted_publish(
            self._permissions(AgentBucketGrant(owner_agent_id=self._OWNER, suffix="survey-quota-cells", writable=True))
        )
        for bucket in (
            self._bucket(self._OTHER_OWNER, "survey-quota-cells"),
            self._bucket(self._OWNER, "panel_reset_tickets"),
            f"{_NS}-checkpoints",
            _COLLECTIONS,
        ):
            for subject in (
                f"$KV.{bucket}.k",
                f"$JS.API.STREAM.INFO.KV_{bucket}",
                f"$JS.API.STREAM.MSG.GET.KV_{bucket}",
                f"$JS.API.DIRECT.GET.KV_{bucket}.$KV.{bucket}.k",
            ):
                if bucket == _COLLECTIONS and subject.startswith("$JS.API.STREAM.INFO"):
                    continue  # the pod binds its OWN scope of the shared bucket; unrelated to this grant
                assert not any(_subject_matches(p, subject) for p in allow), subject

    def test_no_stream_admin_verb_and_no_unnamed_or_durable_consumer_is_granted(self) -> None:
        """create/update carry ``sources``/``republish`` (a read of any stream); purge/delete destroy.

        The one consumer granted is created BY NAME with its filter in the subject, inside the
        bucket (the next test); the unnamed create carries its filter only in the body, and the
        durable create and ``MSG.NEXT`` belong to a consumer nothing here builds.
        """
        bucket = self._bucket(self._OWNER, "survey-quota-cells")
        stream = f"KV_{bucket}"
        allow = _minted_publish(
            self._permissions(AgentBucketGrant(owner_agent_id=self._OWNER, suffix="survey-quota-cells", writable=True))
        )
        for subject in (
            f"$JS.API.STREAM.CREATE.{stream}",
            f"$JS.API.STREAM.UPDATE.{stream}",
            f"$JS.API.STREAM.DELETE.{stream}",
            f"$JS.API.STREAM.PURGE.{stream}",
            f"$JS.API.STREAM.SNAPSHOT.{stream}",
            f"$JS.API.STREAM.RESTORE.{stream}",
            f"$JS.API.STREAM.MSG.DELETE.{stream}",
            f"$JS.API.DIRECT.GET.{stream}",
            f"$JS.API.CONSUMER.CREATE.{stream}",
            f"$JS.API.CONSUMER.DURABLE.CREATE.{stream}.w1",
            f"$JS.API.CONSUMER.MSG.NEXT.{stream}.w1",
            f"$JS.API.CONSUMER.DELETE.{stream}.w1",
        ):
            assert not any(_subject_matches(p, subject) for p in allow), subject

    def test_a_named_consumer_filtered_inside_the_bucket_is_granted(self) -> None:
        """the key listing an erasure sweep needs, and a key watch: by name, filter in the subject.

        The server checks a named create's subject filter against the body's ``filter_subject``, so
        a filter inside the bucket reaches nothing the grant does not already cover. A filter in
        another bucket is refused.
        """
        bucket = self._bucket(self._OWNER, "checkpoints")
        other = self._bucket(self._OTHER_OWNER, "checkpoints")
        allow = _minted_publish(
            self._permissions(AgentBucketGrant(owner_agent_id=self._OWNER, suffix="checkpoints", writable=True))
        )
        for subject in (
            f"$JS.API.CONSUMER.CREATE.KV_{bucket}.kl_1.$KV.{bucket}.>",
            f"$JS.API.CONSUMER.CREATE.KV_{bucket}.kl_1.$KV.{bucket}.thread-1.>",
            f"$JS.API.CONSUMER.CREATE.KV_{bucket}.kw_1.$KV.{bucket}.k",
        ):
            assert any(_subject_matches(p, subject) for p in allow), subject
        for subject in (
            f"$JS.API.CONSUMER.CREATE.KV_{other}.kl_1.$KV.{other}.>",
            f"$JS.API.CONSUMER.CREATE.KV_{bucket}.kl_1.$KV.{other}.>",
        ):
            assert not any(_subject_matches(p, subject) for p in allow), subject

    def test_the_resource_record_is_whole_bucket_key_access(self) -> None:
        permissions = self._permissions(
            AgentBucketGrant(owner_agent_id=self._OWNER, suffix="survey-quota-cells", writable=False)
        )
        owned = self._bucket(self._OWNER, "survey-quota-cells")
        granted = [r for r in permissions.js_resources if r.name == owned]
        assert len(granted) == 1
        assert granted[0].capability is JsCapability.KV_BUCKET_KEYS
        assert granted[0].name == self._bucket(self._OWNER, "survey-quota-cells")
        assert granted[0].scope is None
        assert granted[0].writable is False
        assert not capability_declares(granted[0].capability)

    def test_no_grant_changes_nothing(self) -> None:
        assert self._permissions().js_resources == build_permissions(Principal.TOOL_POD, pod_id=_POD_1).js_resources

    @pytest.mark.parametrize("principal", [p for p in Principal if p is not Principal.TOOL_POD])
    def test_only_a_tool_pod_reads_the_grants(self, principal: Principal) -> None:
        grant = AgentBucketGrant(owner_agent_id=self._OWNER, suffix="survey-quota-cells", writable=True)
        with_grant = build_permissions(principal, **_IDS[principal], agent_bucket_grants=(grant,))
        assert with_grant.js_resources == _build(principal).js_resources

    @pytest.mark.parametrize("suffix", ["a.b", "cells*", "cells>", "", "has space", "x" * 65])
    def test_a_suffix_outside_the_coordination_grammar_is_refused(self, suffix: str) -> None:
        with pytest.raises(ValueError, match="suffix"):
            AgentBucketGrant(owner_agent_id=self._OWNER, suffix=suffix, writable=False)

    def test_one_bucket_granted_twice_is_refused(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            self._permissions(
                AgentBucketGrant(owner_agent_id=self._OWNER, suffix="survey-quota-cells", writable=False),
                AgentBucketGrant(owner_agent_id=self._OWNER, suffix="survey-quota-cells", writable=True),
            )

    def test_the_same_suffix_of_two_owners_is_two_buckets(self) -> None:
        permissions = self._permissions(
            AgentBucketGrant(owner_agent_id=self._OWNER, suffix="survey-quota-cells", writable=True),
            AgentBucketGrant(owner_agent_id=self._OTHER_OWNER, suffix="survey-quota-cells", writable=False),
        )
        granted = {r.name for r in permissions.js_resources if "-agent_pod-" in r.name}
        assert granted == {
            self._bucket(self._OWNER, "survey-quota-cells"),
            self._bucket(self._OTHER_OWNER, "survey-quota-cells"),
        }

    def test_the_whole_bucket_capability_refuses_a_scope_and_a_stream(self) -> None:
        with pytest.raises(ValueError):
            JsResource(
                name=self._bucket(self._OWNER, "survey-quota-cells"),
                kind=JsResourceKind.KV_BUCKET,
                capability=JsCapability.KV_BUCKET_KEYS,
                scope="agent_pod-x",
                writable=False,
            )
        with pytest.raises(ValueError):
            JsResource(
                name="some-stream",
                kind=JsResourceKind.STREAM,
                capability=JsCapability.KV_BUCKET_KEYS,
                scope=None,
                writable=False,
            )


class TestCoordinationBucketName:
    """the one composition of an agent's coordination bucket name."""

    def test_it_composes_namespace_scope_and_suffix(self) -> None:
        assert coordination_bucket_name("agent_pod-abc", "cells", ns="ns1") == "ns1-agent_pod-abc-cells"

    def test_it_defaults_to_the_bound_namespace(self) -> None:
        assert coordination_bucket_name("agent_pod-abc", "cells") == f"{_NS}-agent_pod-abc-cells"

    @pytest.mark.parametrize("suffix", ["a.b", "", "x*", "x" * 65])
    def test_it_refuses_a_suffix_outside_the_grammar(self, suffix: str) -> None:
        with pytest.raises(ValueError, match="suffix"):
            coordination_bucket_name("agent_pod-abc", suffix)

    def test_the_agents_own_grant_uses_it(self) -> None:
        owner = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A, coordination_buckets=("c",))
        scope = kv_key_scope_for(Principal.AGENT_POD, agent_id=_AGENT_A)
        assert coordination_bucket_name(scope, "c") in kv_bucket_names(owner)


class TestToolPodOwnAuditSubject:
    """a tool pod publishes its own audit events on a subject naming its own verified id.

    ``{ns}.audit.tool_pod.<tool_pods.id>.<event_type>``: the subject carries the publisher, so the
    hub's collector reads the actor from what the broker authorised rather than from the envelope.
    """

    def test_a_tool_pod_may_publish_under_its_own_id(self) -> None:
        perm = build_permissions(Principal.TOOL_POD, pod_id=_POD_A)
        own = f"{_NS}.audit.tool_pod.{_POD_A}.collector.promoted"
        assert any(_subject_matches(p, own) for p in perm.publish)
        assert str(Subjects.tool_pod_audit_wildcard(_POD_A)) in perm.publish

    def test_a_tool_pod_may_not_publish_under_another_pods_id(self) -> None:
        perm = build_permissions(Principal.TOOL_POD, pod_id=_POD_A)
        other = f"{_NS}.audit.tool_pod.{_POD_B}.collector.promoted"
        assert not any(_subject_matches(p, other) for p in perm.publish)

    def test_a_tool_pod_may_not_publish_the_hubs_own_tool_pod_records(self) -> None:
        """the hub records ``tool_pod.create`` under the same token; a pod reaches only its own id."""
        perm = build_permissions(Principal.TOOL_POD, pod_id=_POD_A)
        for event_type in ("create", "update", "delete", "data_sync"):
            hub_subject = f"{_NS}.audit.tool_pod.{event_type}"
            assert not any(_subject_matches(p, hub_subject) for p in perm.publish), hub_subject

    def test_the_baseline_tool_call_audit_is_unchanged(self) -> None:
        perm = build_permissions(Principal.TOOL_POD, pod_id=_POD_A)
        assert str(Subjects.audit_event("tool.call")) in perm.publish

    def test_no_other_audit_family_is_opened_to_a_tool_pod(self) -> None:
        perm = build_permissions(Principal.TOOL_POD, pod_id=_POD_A)
        audit = [p for p in perm.publish if p.startswith(f"{_NS}.audit.")]
        assert sorted(audit) == sorted(
            [str(Subjects.audit_event("tool.call")), str(Subjects.tool_pod_audit_wildcard(_POD_A))]
        )

    def test_the_subject_round_trips_to_the_pod_and_event_type(self) -> None:
        subject = Subjects.tool_pod_audit_event(_POD_A, "collector.promoted")
        assert str(subject) == f"{_NS}.audit.tool_pod.{_POD_A}.collector.promoted"
        assert parse_tool_pod_audit_subject(str(subject), namespace=_NS) == (
            uuid.UUID(_POD_A),
            "collector.promoted",
        )

    def test_an_explicit_namespace_overrides_the_bound_one(self) -> None:
        subject = Subjects.tool_pod_audit_event(_POD_A, "collector.promoted", namespace="other")
        assert str(subject).startswith("other.audit.tool_pod.")

    @pytest.mark.parametrize(
        "subject",
        [
            f"{_NS}.audit.tool.call",
            f"{_NS}.audit.tool_pod.create",
            f"{_NS}.audit.tool_pod.not-a-uuid.collector.promoted",
            f"{_NS}.audit.tool_pod.{uuid.UUID(_POD_A).hex}.collector.promoted",
            f"{_NS}.audit.tool_pod.{_POD_A.upper()}.collector.promoted",
            f"{_NS}.audit.tool_pod.{_POD_A}",
            f"other.audit.tool_pod.{_POD_A}.collector.promoted",
        ],
    )
    def test_any_other_subject_parses_to_none(self, subject: str) -> None:
        assert parse_tool_pod_audit_subject(subject, namespace=_NS) is None

    def test_a_pod_id_that_is_not_a_uuid_is_refused(self) -> None:
        with pytest.raises(ValueError):
            Subjects.tool_pod_audit_event("not-a-uuid", "collector.promoted")


class TestPodsBindWhatTheHubDeclares:
    """a pod binds, reads, writes and watches its buckets and never manages a stream.

    ``STREAM.CREATE`` and ``STREAM.UPDATE`` take ``sources`` in the request body, so a pod holding
    either against a bucket of its own could copy any stream on the bus into it and read the copy.
    The hub declares every bucket and stream a pod touches; the exact subjects each pod capability
    mints are pinned here, so a verb added to either capability fails a test rather than widening
    every pod on the platform.
    """

    _SCOPE = kv_key_scope_for(Principal.AGENT_POD, agent_id=_AGENT_A)

    def _agent(self, *suffixes: str) -> PrincipalPermissions:
        return build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A, coordination_buckets=suffixes)

    def test_an_own_coordination_bucket_mints_exactly_bind_read_watch_and_write(self) -> None:
        bucket = coordination_bucket_name(self._SCOPE, "survey-quota-cells", ns=_NS)
        stream = f"KV_{bucket}"
        minted = _minted_publish(self._agent("survey-quota-cells"))
        assert sorted(s for s in minted if bucket in s) == sorted(
            [
                f"$KV.{bucket}.>",
                f"$JS.API.STREAM.INFO.{stream}",
                f"$JS.API.STREAM.MSG.GET.{stream}",
                f"$JS.API.DIRECT.GET.{stream}.$KV.{bucket}.>",
                f"$JS.API.CONSUMER.CREATE.{stream}.*.$KV.{bucket}.>",
            ]
        )

    @pytest.mark.parametrize(
        ("bucket", "capability"),
        [
            (f"{_NS}-ratelimits", JsCapability.KV_OWNER_KEYS),
            (f"{_NS}-proxy_assertion_nonces", JsCapability.KV_OWNER_KEYS),
            (f"{_NS}-epochs", JsCapability.KV_BUCKET_KEYS),
        ],
    )
    def test_every_shared_agent_bucket_is_bound_never_managed(self, bucket: str, capability: JsCapability) -> None:
        resource = [r for r in self._agent().js_resources if r.name == bucket]
        assert len(resource) == 1, bucket
        assert resource[0].capability is capability
        stream = f"KV_{bucket}"
        reads = {f"$JS.API.STREAM.INFO.{stream}", f"$JS.API.STREAM.MSG.GET.{stream}"}
        managed = [
            s
            for s in _minted_publish(self._agent())
            if s.startswith("$JS.API.STREAM.") and stream in s and s not in reads
        ]
        assert managed == [], managed

    @pytest.mark.parametrize("bucket", [f"{_NS}-proxy_assertion_nonces", f"{_NS}-leases"])
    def test_every_tool_pod_bucket_is_bound_never_managed(self, bucket: str) -> None:
        resource = [r for r in _build(Principal.TOOL_POD).js_resources if r.name == bucket]
        assert len(resource) == 1, bucket
        assert resource[0].capability is JsCapability.KV_OWNER_KEYS
        stream = f"KV_{bucket}"
        minted = _minted_publish(_build(Principal.TOOL_POD))
        assert [s for s in minted if s.startswith("$JS.API.STREAM.") and stream in s] == [
            f"$JS.API.STREAM.INFO.{stream}"
        ]

    def test_the_agent_pod_consumes_the_result_stream_only_on_its_own_reply_subjects(self) -> None:
        """the one stream an agent pod consumes: the replies the registry delivers to IT.

        The grant is the named create with the filter IN the subject, where nats-server checks it
        against the body -- never the unnamed create, whose filter rides only in the body and could
        name every agent's replies; never ``MSG.NEXT``, ``INFO`` or ``DELETE``, which reach a
        consumer by name whoever created it.
        """
        stream = result_stream_name()
        own_replies = str(Subjects.tools_reply_agent_subtree(_AGENT_A))
        minted = _minted_publish(self._agent())
        assert sorted(s for s in minted if s.startswith("$JS.API.") and stream in s.split(".")) == [
            f"$JS.API.CONSUMER.CREATE.{stream}.*.{own_replies}",
        ]

    @pytest.mark.parametrize(
        "subject",
        [
            # the unnamed create: its filter rides only in the body
            "$JS.API.CONSUMER.CREATE.{stream}",
            # a named create over another agent's replies, over every reply, over the whole stream
            "$JS.API.CONSUMER.CREATE.{stream}.c1.{ns}.tools.reply.{other}.call-1",
            "$JS.API.CONSUMER.CREATE.{stream}.c1.{ns}.tools.reply.*.call-1",
            "$JS.API.CONSUMER.CREATE.{stream}.c1.{ns}.tools.result.{other}.call-1",
            "$JS.API.CONSUMER.CREATE.{stream}.c1.>",
            # the durable create, and every verb that names an existing consumer
            "$JS.API.CONSUMER.DURABLE.CREATE.{stream}.c1",
            "$JS.API.CONSUMER.MSG.NEXT.{stream}.registry-waiter",
            "$JS.API.CONSUMER.INFO.{stream}.registry-waiter",
            "$JS.API.CONSUMER.DELETE.{stream}.registry-waiter",
            "$JS.API.CONSUMER.LIST.{stream}",
            "$JS.API.CONSUMER.NAMES.{stream}",
            "$JS.API.STREAM.INFO.{stream}",
            "$JS.API.STREAM.MSG.GET.{stream}",
            "$JS.API.DIRECT.GET.{stream}",
        ],
    )
    def test_no_agent_pod_can_reach_another_agents_deliveries(self, subject: str) -> None:
        concrete = subject.format(stream=result_stream_name(), ns=_NS, other=_AGENT_B)
        minted = _minted_publish(self._agent())
        assert not any(_subject_matches(p, concrete) for p in minted), concrete

    @pytest.mark.parametrize("stream", [f"{_NS}-channels-deliver", f"{_NS}-audit"])
    def test_no_pod_holds_any_jetstream_grant_on_a_stream_it_only_publishes_to(self, stream: str) -> None:
        """a pod publishes finished answers and audit events and consumes neither.

        A JetStream publish is an ordinary publish plus a PubAck on the publisher's own inbox, so it
        needs no ``$JS.API`` grant at all -- and any consumer grant on these streams is a read of
        every agent's answers or audit trail.
        """
        for permissions in (self._agent(), _build(Principal.TOOL_POD)):
            minted = _minted_publish(permissions)
            assert not [s for s in minted if s.startswith("$JS.API.") and stream in s.split(".")], stream
            assert stream not in {r.name for r in permissions.js_resources}

    def test_the_tool_pod_holds_no_jetstream_grant_on_the_result_stream(self) -> None:
        """a tool pod publishes its results and never collects one: the registry does."""
        stream = result_stream_name()
        minted = _minted_publish(_build(Principal.TOOL_POD))
        assert not [s for s in minted if s.startswith("$JS.API.") and stream in s.split(".")]

    @pytest.mark.parametrize("principal", _POD_PRINCIPALS)
    def test_every_pod_stream_is_a_filtered_consumer_grant(self, principal: Principal) -> None:
        streams = [r for r in _build(principal).js_resources if r.kind is JsResourceKind.STREAM]
        assert all(r.capability is JsCapability.STREAM_CONSUMER for r in streams), streams
        assert all(r.filter_subject for r in streams), streams

    @pytest.mark.parametrize("verb", ["CREATE", "UPDATE", "DELETE", "PURGE", "SNAPSHOT", "RESTORE"])
    def test_no_pod_may_manage_its_own_bucket_or_stream(self, verb: str) -> None:
        bucket = coordination_bucket_name(self._SCOPE, "checkpoints", ns=_NS)
        for permissions in (self._agent("checkpoints"), _build(Principal.TOOL_POD)):
            minted = _minted_publish(permissions)
            for stream in (f"KV_{bucket}", f"KV_{_NS}-checkpoints", result_stream_name(), f"{_NS}-audit"):
                subject = f"$JS.API.STREAM.{verb}.{stream}"
                assert not any(_subject_matches(p, subject) for p in minted), subject

    def test_the_infra_declarers_keep_management(self) -> None:
        """the registry declares the result stream at startup; narrowing it would strand it."""
        resource = [r for r in _build(Principal.REGISTRY).js_resources if r.name == result_stream_name()]
        assert len(resource) == 1
        assert resource[0].capability is JsCapability.FULL

    def test_a_consumer_capability_is_refused_on_a_bucket(self) -> None:
        with pytest.raises(ValueError, match="plain stream"):
            JsResource(
                name=f"{_NS}-checkpoints",
                kind=JsResourceKind.KV_BUCKET,
                capability=JsCapability.STREAM_CONSUMER,
                scope=None,
                writable=False,
                filter_subject=f"{_NS}.x.y",
            )

    def test_the_consumer_grant_is_exactly_the_named_filtered_create(self) -> None:
        assert js_api_grants_for_stream(
            "s1", capability=JsCapability.STREAM_CONSUMER, filter_subject="ns.tools.reply.a1.*"
        ) == ["$JS.API.CONSUMER.CREATE.s1.*.ns.tools.reply.a1.*"]

    def test_a_consumer_grant_without_a_filter_is_refused(self) -> None:
        """no filter would have to mean the whole stream, which is every principal's messages."""
        with pytest.raises(ValueError, match="filter"):
            JsResource(
                name="s1",
                kind=JsResourceKind.STREAM,
                capability=JsCapability.STREAM_CONSUMER,
                scope=None,
                writable=False,
            )
        with pytest.raises(ValueError, match="filter"):
            js_api_grants_for_stream("s1", capability=JsCapability.STREAM_CONSUMER)

    @pytest.mark.parametrize(
        "filter_subject",
        [">", "*", "ns.>", "*.tools.reply.a1.*", "ns.tools..a1", "ns.tools. a1", "ns.tools.reply.>"],
    )
    def test_a_consumer_filter_that_could_widen_is_refused(self, filter_subject: str) -> None:
        """a full-wildcard tail, a wildcard namespace or area, or an empty or blank token."""
        with pytest.raises(ValueError, match="filter"):
            JsResource.stream_consumer("s1", filter_subject=filter_subject)

    def test_only_a_consumer_grant_carries_a_filter(self) -> None:
        with pytest.raises(ValueError, match="filter"):
            JsResource(
                name="s1",
                kind=JsResourceKind.STREAM,
                capability=JsCapability.FULL,
                scope=None,
                writable=False,
                filter_subject="ns.tools.reply.a1.*",
            )


class TestAgentConfigIsReadOnlyForItsPod:
    """``platform.agents`` is the source of an agent's config and the KV bucket its hot cache.

    The hub writes the cache; an agent pod reads and watches its OWN key and nothing else. A write
    grant would let a pod rewrite the cache the agent router reads its turn timeout from, and a
    whole-bucket read is every agent's system prompt and access block, across customers.
    """

    def _agent(self, agent_id: str = _AGENT_A) -> PrincipalPermissions:
        return build_permissions(Principal.AGENT_POD, agent_id=agent_id, pod_id=_POD_A)

    def test_the_bucket_name_and_key_are_the_ones_the_hub_writes(self) -> None:
        assert agent_config_bucket_name(_NS) == f"{_NS}_agent_config"
        assert agent_config_kv_key(uuid.UUID(_AGENT_A)) == _AGENT_A
        assert agent_config_kv_key(_AGENT_A.upper()) == _AGENT_A

    def test_a_non_uuid_key_is_refused(self) -> None:
        with pytest.raises(ValueError, match="uuid"):
            agent_config_kv_key("not-an-agent")

    def test_the_pod_holds_a_read_of_its_own_key_only(self) -> None:
        resource = [r for r in self._agent().js_resources if r.name == agent_config_bucket_name(_NS)]
        assert len(resource) == 1
        assert resource[0].capability is JsCapability.KV_KEY_READ
        assert resource[0].scope == _AGENT_A
        assert resource[0].writable is False

    def test_the_minted_grant_is_bind_own_key_read_and_own_key_watch(self) -> None:
        bucket = agent_config_bucket_name(_NS)
        stream = f"KV_{bucket}"
        minted = _minted_publish(self._agent())
        assert sorted(s for s in minted if bucket in s) == sorted(
            [
                f"$JS.API.STREAM.INFO.{stream}",
                f"$JS.API.DIRECT.GET.{stream}.$KV.{bucket}.{_AGENT_A}",
                f"$JS.API.CONSUMER.CREATE.{stream}.*.$KV.{bucket}.{_AGENT_A}",
            ]
        )

    @pytest.mark.parametrize(
        "subject",
        [
            "$KV.{bucket}.{own}",
            "$KV.{bucket}.{other}",
            "$JS.API.DIRECT.GET.KV_{bucket}.$KV.{bucket}.{other}",
            "$JS.API.STREAM.MSG.GET.KV_{bucket}",
            "$JS.API.CONSUMER.CREATE.KV_{bucket}.w1.$KV.{bucket}.{other}",
            "$JS.API.CONSUMER.CREATE.KV_{bucket}.w1.$KV.{bucket}.>",
            "$JS.API.CONSUMER.CREATE.KV_{bucket}",
        ],
    )
    def test_no_write_and_no_other_agents_key(self, subject: str) -> None:
        bucket = agent_config_bucket_name(_NS)
        concrete = subject.format(bucket=bucket, own=_AGENT_A, other=_AGENT_B)
        minted = _minted_publish(self._agent())
        assert not any(_subject_matches(p, concrete) for p in minted), concrete


class TestWorkspaceLocksBucket:
    """an agent's workspace file locks live in a bucket of its OWN, declared by the hub.

    The lock keys name workspace ids and file paths, and a lock bucket every agent shared would let
    any agent list them -- another customer's paths included -- and create or delete any lock. So
    the bucket is per agent, composed under the agent's authenticated scope exactly as a declared
    coordination bucket is, and granted to every agent pod whether or not it declares anything.
    """

    _SCOPE = kv_key_scope_for(Principal.AGENT_POD, agent_id=_AGENT_A)

    def test_every_agent_pod_is_granted_its_own_workspace_locks_bucket(self) -> None:
        bucket = coordination_bucket_name(self._SCOPE, WORKSPACE_LOCKS_BUCKET_SUFFIX, ns=_NS)
        permissions = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A)
        resource = [r for r in permissions.js_resources if r.name == bucket]
        assert len(resource) == 1
        assert resource[0].capability is JsCapability.KV_BUCKET_KEYS
        assert resource[0].writable is True

    def test_no_agent_pod_is_granted_another_agents_workspace_locks(self) -> None:
        other_scope = kv_key_scope_for(Principal.AGENT_POD, agent_id=_AGENT_B)
        other = coordination_bucket_name(other_scope, WORKSPACE_LOCKS_BUCKET_SUFFIX, ns=_NS)
        minted = _minted_publish(build_permissions(Principal.AGENT_POD, agent_id=_AGENT_A, pod_id=_POD_A))
        assert not [s for s in minted if other in s]

    def test_the_platform_bucket_suffixes_name_the_workspace_locks(self) -> None:
        assert WORKSPACE_LOCKS_BUCKET_SUFFIX in AGENT_POD_PLATFORM_BUCKET_SUFFIXES
        for suffix in AGENT_POD_PLATFORM_BUCKET_SUFFIXES:
            coordination_bucket_name(self._SCOPE, suffix, ns=_NS)

    def test_the_bucket_suffix_the_hub_and_the_lease_open_is_the_one_granted(self) -> None:
        suffix = agent_platform_bucket_suffix(uuid.UUID(_AGENT_A), WORKSPACE_LOCKS_BUCKET_SUFFIX)
        assert f"{_NS}-{suffix}" == coordination_bucket_name(self._SCOPE, WORKSPACE_LOCKS_BUCKET_SUFFIX, ns=_NS)

    def test_a_suffix_that_is_not_a_platform_bucket_is_refused(self) -> None:
        with pytest.raises(ValueError, match="platform"):
            agent_platform_bucket_suffix(uuid.UUID(_AGENT_A), "survey-quota-cells")


class TestSharedPodBucketsAreOwnerScoped:
    """a pod reaches only ITS OWN keys in the platform's shared pod buckets.

    ``{ns}-ratelimits``, ``{ns}-proxy_assertion_nonces`` and ``{ns}-leases`` are each one bucket that
    every agent pod or every tool pod binds. A grant on the whole bucket let any pod read, watch and
    list every other pod's keys -- and, being writable, delete or overwrite them: burn another pod's
    nonce, lift its extraction throttle, steal its display claim. Every key 3tears writes there now
    leads with the writer's own scope (:func:`kv_key_scope_for`), and the grant is narrowed to it.

    ``{ns}-checkpoints`` is granted to no pod at all: nothing on the platform reads or writes it any
    more (the agent runtime keeps checkpoints in L3 alone, and the survey keeps its cache in its own
    coordination bucket), so the only thing a grant on it conferred was a read of other agents'
    conversation state. ``{ns}-epochs`` is platform-shared by nature -- its keys are subject paths
    every pod may need to read -- so a pod reads it and never writes it.
    """

    _AGENT_SCOPE = kv_key_scope_for(Principal.AGENT_POD, agent_id=_AGENT_A)
    _TOOL_SCOPE = kv_key_scope_for(Principal.TOOL_POD, pod_id=_POD_A)

    def _agent(self, agent_id: str = _AGENT_A) -> PrincipalPermissions:
        return build_permissions(Principal.AGENT_POD, agent_id=agent_id, pod_id=_POD_A)

    def _tool(self, pod_id: str = _POD_A) -> PrincipalPermissions:
        return build_permissions(Principal.TOOL_POD, pod_id=pod_id)

    def _resource(self, permissions: PrincipalPermissions, name: str) -> JsResource:
        found = [r for r in permissions.js_resources if r.name == name]
        assert len(found) == 1, f"expected one {name} resource, found {len(found)}"
        return found[0]

    @pytest.mark.parametrize("suffix", ["ratelimits", "proxy_assertion_nonces"])
    def test_an_agent_pod_holds_its_own_keys_in_each_shared_bucket(self, suffix: str) -> None:
        resource = self._resource(self._agent(), f"{_NS}-{suffix}")
        assert resource.capability is JsCapability.KV_OWNER_KEYS
        assert resource.scope == self._AGENT_SCOPE
        assert resource.writable is True

    @pytest.mark.parametrize("suffix", ["proxy_assertion_nonces", "leases"])
    def test_a_tool_pod_holds_its_own_keys_in_each_shared_bucket(self, suffix: str) -> None:
        resource = self._resource(self._tool(), f"{_NS}-{suffix}")
        assert resource.capability is JsCapability.KV_OWNER_KEYS
        assert resource.scope == self._TOOL_SCOPE
        assert resource.writable is True

    @pytest.mark.parametrize("principal", _POD_PRINCIPALS)
    def test_no_pod_is_granted_the_shared_checkpoint_bucket(self, principal: Principal) -> None:
        permissions = _build(principal)
        assert f"{_NS}-checkpoints" not in kv_bucket_names(permissions)
        assert not [s for s in _minted_publish(permissions) if f"{_NS}-checkpoints" in s]

    def test_the_hub_no_longer_declares_the_shared_checkpoint_bucket(self) -> None:
        assert f"{_NS}-checkpoints" not in kv_bucket_names(_build(Principal.HUB))

    def test_an_agent_pod_reads_the_epoch_bucket_and_writes_none_of_it(self) -> None:
        resource = self._resource(self._agent(), f"{_NS}-epochs")
        assert resource.capability is JsCapability.KV_BUCKET_KEYS
        assert resource.writable is False
        assert not [s for s in _minted_publish(self._agent()) if s.startswith(f"$KV.{_NS}-epochs.")]

    @pytest.mark.parametrize(
        ("principal_build", "scope", "bucket"),
        [
            ("_agent", _AGENT_SCOPE, f"{_NS}-ratelimits"),
            ("_agent", _AGENT_SCOPE, f"{_NS}-proxy_assertion_nonces"),
            ("_tool", _TOOL_SCOPE, f"{_NS}-proxy_assertion_nonces"),
            ("_tool", _TOOL_SCOPE, f"{_NS}-leases"),
        ],
    )
    def test_the_minted_grant_reaches_only_the_owners_prefix(
        self, principal_build: str, scope: str, bucket: str
    ) -> None:
        minted = _minted_publish(getattr(self, principal_build)())
        stream = f"KV_{bucket}"
        on_bucket = sorted(s for s in minted if stream in s or f"$KV.{bucket}." in s)
        assert on_bucket == sorted(
            [
                f"$KV.{bucket}.{scope}.>",
                f"$JS.API.STREAM.INFO.{stream}",
                f"$JS.API.DIRECT.GET.{stream}.$KV.{bucket}.{scope}.>",
                f"$JS.API.CONSUMER.CREATE.{stream}.*.$KV.{bucket}.{scope}.>",
            ]
        ), on_bucket

    @pytest.mark.parametrize("bucket", [f"{_NS}-ratelimits", f"{_NS}-proxy_assertion_nonces"])
    def test_another_agents_keys_are_neither_readable_listable_nor_writable(self, bucket: str) -> None:
        other_scope = kv_key_scope_for(Principal.AGENT_POD, agent_id=_AGENT_B)
        minted = _minted_publish(self._agent())
        stream = f"KV_{bucket}"
        for subject in (
            f"$KV.{bucket}.{other_scope}.k",
            f"$JS.API.DIRECT.GET.{stream}.$KV.{bucket}.{other_scope}.k",
            f"$JS.API.CONSUMER.CREATE.{stream}.kl_x.$KV.{bucket}.{other_scope}.>",
            f"$JS.API.CONSUMER.CREATE.{stream}.kl_x.$KV.{bucket}.>",
            f"$JS.API.CONSUMER.CREATE.{stream}",
            f"$JS.API.STREAM.MSG.GET.{stream}",
            f"$JS.API.DIRECT.GET.{stream}",
            f"$KV.{bucket}.unscoped-key",
        ):
            assert not [p for p in minted if _subject_matches(p, subject)], subject

    def test_own_keys_are_readable_listable_and_writable(self) -> None:
        bucket = f"{_NS}-proxy_assertion_nonces"
        minted = _minted_publish(self._agent())
        stream = f"KV_{bucket}"
        scope = self._AGENT_SCOPE
        for subject in (
            f"$KV.{bucket}.{scope}.abc",
            f"$JS.API.DIRECT.GET.{stream}.$KV.{bucket}.{scope}.abc",
            f"$JS.API.CONSUMER.CREATE.{stream}.kl_x.$KV.{bucket}.{scope}.>",
            f"$JS.API.STREAM.INFO.{stream}",
        ):
            assert [p for p in minted if _subject_matches(p, subject)], subject

    def test_two_agents_never_share_an_owner_prefix(self) -> None:
        first = self._resource(self._agent(_AGENT_A), f"{_NS}-ratelimits")
        second = self._resource(self._agent(_AGENT_B), f"{_NS}-ratelimits")
        assert first.scope != second.scope

    def test_the_owner_keys_capability_is_never_a_declaring_one(self) -> None:
        assert not capability_declares(JsCapability.KV_OWNER_KEYS)

    @pytest.mark.parametrize("scope", [None, "", "a.b", "a*", ">"])
    def test_an_owner_keys_grant_needs_one_literal_scope_token(self, scope: str | None) -> None:
        with pytest.raises(ValueError):
            JsResource(
                name=f"{_NS}-ratelimits",
                kind=JsResourceKind.KV_BUCKET,
                capability=JsCapability.KV_OWNER_KEYS,
                scope=scope,
                writable=True,
            )

    def test_an_owner_keys_grant_applies_only_to_a_kv_bucket(self) -> None:
        with pytest.raises(ValueError):
            JsResource(
                name=f"{_NS}-audit",
                kind=JsResourceKind.STREAM,
                capability=JsCapability.KV_OWNER_KEYS,
                scope=self._AGENT_SCOPE,
                writable=False,
            )

    def test_the_constructor_builds_the_owner_keys_record(self) -> None:
        resource = JsResource.kv_owner_keys(f"{_NS}-leases", scope=self._TOOL_SCOPE, writable=True)
        assert resource.capability is JsCapability.KV_OWNER_KEYS
        assert resource.key_prefix == self._TOOL_SCOPE


class TestToolPodObjectStore:
    """a tool pod's own Object Store bucket and its pointer bucket: the hub declares, the pod works inside.

    Both are composed under the pod's own scope, so the grant names no other principal's data. The
    Object Store grant is bind, a subject-carried metadata read, a NAMED consumer filtered inside the
    bucket, and the ``$O.`` publish -- never a stream-management verb, never ``STREAM.MSG.GET`` or
    ``STREAM.MSG.DELETE``, never the unnamed consumer nats-py's own ``ObjectStore.get`` creates.
    """

    _SCOPE = kv_key_scope_for(Principal.TOOL_POD, pod_id=_POD_X)

    def _pod(self) -> PrincipalPermissions:
        return build_permissions(Principal.TOOL_POD, pod_id=_POD_X, object_store=True)

    def test_a_pod_that_has_not_opted_in_holds_no_object_store(self) -> None:
        plain = build_permissions(Principal.TOOL_POD, pod_id=_POD_X)
        names = {r.name for r in plain.js_resources}
        assert tool_pod_object_store_name(_POD_X, ns=_NS) not in names
        assert tool_pod_pointers_bucket_name(_POD_X, ns=_NS) not in names
        assert not [s for s in plain.publish if ".hub.object_store." in s]
        assert not [s for s in _minted_publish(plain) if s.startswith("$O.") or "OBJ_" in s]

    @pytest.mark.parametrize("principal", [p for p in Principal if p is not Principal.TOOL_POD])
    def test_only_a_tool_pod_may_opt_in(self, principal: Principal) -> None:
        with pytest.raises(ValueError):
            build_permissions(principal, object_store=True, **_IDS[principal])

    def test_the_bucket_names_are_composed_under_the_pods_own_scope(self) -> None:
        assert tool_pod_object_store_name(_POD_X, ns=_NS) == f"{_NS}-{self._SCOPE}-{TOOL_POD_OBJECTS_BUCKET_SUFFIX}"
        assert tool_pod_pointers_bucket_name(_POD_X, ns=_NS) == f"{_NS}-{self._SCOPE}-{TOOL_POD_POINTERS_BUCKET_SUFFIX}"
        assert tool_pod_object_store_name(_POD_X, ns=_NS) != tool_pod_object_store_name(_POD_VICTIM, ns=_NS)

    def test_the_pod_holds_its_object_store_and_pointer_bucket(self) -> None:
        by_name = {r.name: r for r in self._pod().js_resources}
        store = by_name[tool_pod_object_store_name(_POD_X, ns=_NS)]
        assert store.kind is JsResourceKind.OBJECT_STORE
        assert store.capability is JsCapability.OBJECT_STORE_OBJECTS
        assert store.writable
        assert store.stream_name == f"OBJ_{store.name}"
        pointers = by_name[tool_pod_pointers_bucket_name(_POD_X, ns=_NS)]
        assert (pointers.kind, pointers.capability, pointers.writable) == (
            JsResourceKind.KV_BUCKET,
            JsCapability.KV_BUCKET_KEYS,
            True,
        )

    def test_the_object_store_grant_is_exactly_bind_read_named_consumer_and_write(self) -> None:
        bucket = tool_pod_object_store_name(_POD_X, ns=_NS)
        stream = f"OBJ_{bucket}"
        minted = _minted_publish(self._pod())
        assert sorted(s for s in minted if bucket in s) == sorted(
            [
                f"$O.{bucket}.>",
                f"$JS.API.STREAM.INFO.{stream}",
                f"$JS.API.DIRECT.GET.{stream}.$O.{bucket}.M.>",
                f"$JS.API.CONSUMER.CREATE.{stream}.*.$O.{bucket}.>",
            ]
        )

    @pytest.mark.parametrize(
        "subject",
        [
            "$JS.API.STREAM.CREATE.{stream}",
            "$JS.API.STREAM.UPDATE.{stream}",
            "$JS.API.STREAM.DELETE.{stream}",
            "$JS.API.STREAM.PURGE.{stream}",
            "$JS.API.STREAM.SNAPSHOT.{stream}",
            "$JS.API.STREAM.RESTORE.{stream}",
            "$JS.API.STREAM.MSG.GET.{stream}",
            "$JS.API.STREAM.MSG.DELETE.{stream}",
            # the unnamed create nats-py's ObjectStore.get and watch use: filter only in the body
            "$JS.API.CONSUMER.CREATE.{stream}",
            "$JS.API.CONSUMER.DURABLE.CREATE.{stream}.c1",
            "$JS.API.CONSUMER.DELETE.{stream}.c1",
            "$JS.API.CONSUMER.MSG.NEXT.{stream}.c1",
            "$JS.API.DIRECT.GET.{stream}",
            # another pod's bucket, by every route
            "$JS.API.STREAM.INFO.{victim}",
            "$JS.API.DIRECT.GET.{victim}.$O.{victim_bucket}.M.x",
            "$JS.API.CONSUMER.CREATE.{victim}.c1.$O.{victim_bucket}.C.x",
            "$O.{victim_bucket}.C.x",
        ],
    )
    def test_no_object_store_management_or_foreign_reach(self, subject: str) -> None:
        bucket = tool_pod_object_store_name(_POD_X, ns=_NS)
        victim_bucket = tool_pod_object_store_name(_POD_VICTIM, ns=_NS)
        concrete = subject.format(stream=f"OBJ_{bucket}", victim=f"OBJ_{victim_bucket}", victim_bucket=victim_bucket)
        minted = _minted_publish(self._pod())
        assert not any(_subject_matches(p, concrete) for p in minted), concrete

    def test_a_read_only_object_store_grant_mints_no_publish(self) -> None:
        resource = JsResource.object_store(f"{_NS}-somebody-objects", writable=False)
        permissions = PrincipalPermissions(
            publish=(), subscribe=(), allow_responses=False, inbox_prefix="_INBOX_x", js_resources=(resource,)
        )
        assert not [s for s in _minted_publish(permissions) if s.startswith("$O.")]

    @pytest.mark.parametrize(
        ("kind", "capability"),
        [
            (JsResourceKind.KV_BUCKET, JsCapability.OBJECT_STORE_OBJECTS),
            (JsResourceKind.STREAM, JsCapability.OBJECT_STORE_OBJECTS),
            (JsResourceKind.OBJECT_STORE, JsCapability.KV_BUCKET_KEYS),
            (JsResourceKind.OBJECT_STORE, JsCapability.STREAM_CONSUMER),
        ],
    )
    def test_the_object_store_capability_and_kind_go_together(
        self, kind: JsResourceKind, capability: JsCapability
    ) -> None:
        with pytest.raises(ValueError):
            JsResource(
                name=f"{_NS}-x-objects",
                kind=kind,
                capability=capability,
                scope=None,
                writable=False,
                filter_subject="a.b.c" if capability is JsCapability.STREAM_CONSUMER else None,
            )

    def test_object_store_requests_are_tool_pod_publish_hub_subscribe(self) -> None:
        pod = self._pod()
        hub = _build(Principal.HUB)
        agent = _build(Principal.AGENT_POD)
        for subject in (f"{_NS}.hub.object_store.declare", f"{_NS}.hub.object_store.retire"):
            assert subject in pod.publish
            assert subject not in pod.subscribe
            assert subject in hub.subscribe
            assert subject not in agent.publish


class TestCollectionKeysPurgeRequest:
    """every tool pod may ask the hub to purge its retired collection keys; only the hub answers."""

    def test_the_purge_request_is_tool_pod_publish_hub_subscribe(self) -> None:
        subject = f"{_NS}.hub.collection_keys.purge"
        pod = _build(Principal.TOOL_POD)
        hub = _build(Principal.HUB)
        agent = _build(Principal.AGENT_POD)
        assert subject in pod.publish
        assert subject not in pod.subscribe
        assert subject in hub.subscribe
        assert subject not in agent.publish


class TestAclInvalidationSubjectsAreRetired:
    """epoch-task-06 contract: the access tables' generations replaced the acl invalidation subjects."""

    @pytest.mark.parametrize("principal", list(Principal))
    def test_nobody_may_publish_them(self, principal: Principal) -> None:
        perm = _build(principal)
        assert not [s for s in perm.publish if ".acl." in s], f"{principal} may still publish an acl subject"

    @pytest.mark.parametrize("principal", [p for p in Principal if p is not Principal.AGENT_POD])
    def test_only_an_agent_pod_may_still_subscribe(self, principal: Principal) -> None:
        perm = _build(principal)
        assert not [s for s in perm.subscribe if ".acl." in s], f"{principal} may still subscribe an acl subject"

    def test_an_agent_pod_one_release_back_is_not_refused_its_subscribe(self) -> None:
        perm = _build(Principal.AGENT_POD)
        assert {str(Subjects.acl_invalidate(kind)) for kind in ("membership", "assignment", "role")} <= set(
            perm.subscribe
        )


class TestTheKvStreamNameIsComposedInOnePlace:
    """a KV bucket's stream name has one public composer, beside the bucket-name composers.

    Code that addresses a bucket's stream (a stream-info read, a purge, a withdraw) names it through
    :func:`kv_stream_name`; the stream the bucket is declared as, and the stream a grant pins, are the
    same composition, so the three can never name different streams.
    """

    def test_it_names_the_stream_nats_backs_a_bucket_with(self) -> None:
        pointers = tool_pod_pointers_bucket_name(_POD_X, ns=_NS)
        assert kv_stream_name(pointers) == f"KV_{pointers}"

    def test_the_declared_stream_is_the_composed_one(self) -> None:
        bucket = coordination_bucket_name("scope", "locks", ns=_NS)
        declared = build_kv_stream_config(
            bucket=bucket, ttl_seconds=0, history=1, storage_type=StorageType.MEMORY, direct=True
        )
        assert declared.name == kv_stream_name(bucket)

    def test_the_granted_stream_is_the_composed_one(self) -> None:
        pointers = {
            r.name: r for r in build_permissions(Principal.TOOL_POD, pod_id=_POD_X, object_store=True).js_resources
        }[tool_pod_pointers_bucket_name(_POD_X, ns=_NS)]
        assert pointers.stream_name == kv_stream_name(pointers.name)

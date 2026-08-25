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

import pytest

from threetears.nats.subject_permissions import (
    CROSS_PLATFORM_CACHE_INVALIDATE,
    MAX_COORDINATION_BUCKETS,
    JsCapability,
    JsResourceKind,
    Principal,
    PrincipalPermissions,
    build_permissions,
    capability_declares,
    kv_bucket_names,
    kv_key_scope_for,
)
from threetears.nats.subjects import Subjects, set_default_namespace

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

#: the two principals whose identity is a POD rather than a service.
_POD_PRINCIPALS = (Principal.AGENT_POD, Principal.TOOL_POD)


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
                or subj.startswith(f"{perm.inbox_prefix}.")
            )
            assert scoped, f"{principal}: unscoped subject {subj!r}"

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
            (Principal.TOOL_POD, [f"{_NS}.tools.register", f"{_NS}.hub.jwks", f"{_NS}.hub.object.resolve"]),
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
            s.startswith("prod7.") or s == CROSS_PLATFORM_CACHE_INVALIDATE or s.startswith("_INBOX_")
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

    ALPHA = "tools.scrape-zone_alpha.1-0-0"
    BETA = "tools.scrape-zone_beta.1-0-0"

    def _pod(self, *namespaces: str) -> PrincipalPermissions:
        return build_permissions(Principal.TOOL_POD, pod_id=_POD_X, tool_namespaces=namespaces)

    def test_pod_subscribes_an_exact_family_literal_per_authorized_tool(self) -> None:
        """each ``allowed_namespaces`` entry becomes one grant, and only the key is wildcarded."""
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
        perm = self._pod("tools.evil name.* > .1-0-0")
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

        other = Subjects.hitl_forward_family("tools.some-other-tool.1-0-0")
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

    def test_every_participant_declares_the_stream_that_carries_the_answer(self) -> None:
        """delivery rides JetStream, and a JS grant is pinned per DECLARED stream name.

        the failure of omitting one is not a denial that says so: an ungranted JetStream operation
        blocks to its deadline, which reads as an unreachable broker rather than a missing grant.
        """
        from threetears.nats.result_delivery import result_stream_name

        stream = result_stream_name()
        for principal in (Principal.TOOL_POD, Principal.AGENT_POD, Principal.REGISTRY):
            declared = [r.name for r in _build(principal).js_resources if r.kind is JsResourceKind.STREAM]
            assert stream in declared, principal

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

    @pytest.mark.parametrize(
        "principal",
        [
            Principal.AGENT_POD,
            Principal.TOOL_POD,
            Principal.REGISTRY,
            Principal.HUB,
            Principal.GATEWAY,
            Principal.CHANNEL_ADAPTER,
            Principal.AGENT_ROUTER,
            Principal.DATASET_EXECUTOR,
        ],
    )
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
        for resource in _build(principal).js_resources:
            if resource.kind is not JsResourceKind.KV_BUCKET or resource.name == _COLLECTIONS:
                continue
            assert resource.scope is None, f"{principal}: {resource.name} would deny its own reads"
            assert resource.capability is JsCapability.FULL, f"{principal}: {resource.name}"

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

        ``platform.agents`` is the source and this bucket is a hot cache over it, written only by
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

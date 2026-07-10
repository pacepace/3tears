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
    Principal,
    PrincipalPermissions,
    build_permissions,
)
from threetears.nats.subjects import set_default_namespace

_NS = "3tears"

#: representative ids so every principal resolves to a concrete allow-list.
_IDS: dict[Principal, dict[str, str]] = {
    Principal.AGENT_POD: {"agent_id": "agent-1", "pod_id": "pod-1"},
    Principal.TOOL_POD: {"pod_id": "pod-1"},
    Principal.REGISTRY: {"conn_id": "reg-1"},
    Principal.HUB: {"conn_id": "hub-1"},
    Principal.GATEWAY: {"conn_id": "gw-1"},
    Principal.CHANNEL_ADAPTER: {"conn_id": "chan-1"},
}


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
        a = build_permissions(Principal.AGENT_POD, agent_id="agent-A", pod_id="pod-A")
        a_internal = [s for s in a.subscribe if ".agents.internal." in s]
        assert a_internal == [f"{_NS}.agents.internal.agent-A.pod-A"]
        # a different agent's routed inbox is a DIFFERENT subject -> no cross-subscribe
        b = build_permissions(Principal.AGENT_POD, agent_id="agent-B", pod_id="pod-B")
        assert f"{_NS}.agents.internal.agent-B.pod-B" not in a.subscribe
        assert [s for s in b.subscribe if ".agents.internal." in s] != a_internal

    def test_tool_internal_subject_is_own_pod(self) -> None:
        a = build_permissions(Principal.TOOL_POD, pod_id="pod-A")
        b = build_permissions(Principal.TOOL_POD, pod_id="pod-B")
        assert f"{_NS}.tools.internal.pod-A" in a.subscribe
        assert f"{_NS}.tools.internal.pod-A" not in b.subscribe

    def test_pod_inbox_is_identity_scoped(self) -> None:
        a = build_permissions(Principal.AGENT_POD, agent_id="agent-A", pod_id="pod-A")
        b = build_permissions(Principal.AGENT_POD, agent_id="agent-B", pod_id="pod-B")
        assert a.inbox_prefix != b.inbox_prefix

    def test_pod_may_publish_only_its_own_heartbeat(self) -> None:
        a = build_permissions(Principal.TOOL_POD, pod_id="pod-A")
        assert f"{_NS}.tools.heartbeat.pod-A" in a.publish
        # no wildcard heartbeat publish -> a pod cannot forge another pod's heartbeat
        assert f"{_NS}.tools.heartbeat.*" not in a.publish
        assert f"{_NS}.tools.heartbeat.>" not in a.publish

    def test_agent_pod_may_publish_turn_completion(self) -> None:
        # resilience-task-07 router-mediated delivery: an agent signals TRUE turn completion by
        # publishing to ``agents.complete.{correlation_id}`` (the router awaits it to ack the durable
        # turn / re-route). the subject is keyed by correlation id (no agent segment), so the grant is
        # the wildcard ``agents.complete.*`` -- without it the completion publish is a NATS permissions
        # violation and every turn hangs to the caller's finalize timeout.
        a = build_permissions(Principal.AGENT_POD, agent_id="agent-A", pod_id="pod-A")
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
        a = build_permissions(Principal.AGENT_POD, agent_id="agent-A", pod_id="pod-A")
        # its own in-process tool server: register (point) + heartbeat scoped to its own agent subtree.
        assert f"{_NS}.tools.register" in a.publish
        assert f"{_NS}.tools.heartbeat.agent-A.>" in a.publish
        # receives the registry's proxied calls + reachability probes for its OWN agent subtree only.
        assert f"{_NS}.tools.internal.agent-A.>" in a.subscribe
        assert f"{_NS}.tools.probe.agent-A.>" in a.subscribe
        # the grant is scoped on the AUTHENTICATED agent id, never the spoofable connect-name pod id:
        # the legacy single-token pod-scoped grants are GONE (closing the connect-name wiretap).
        assert f"{_NS}.tools.internal.pod-A" not in a.subscribe
        assert f"{_NS}.tools.probe.pod-A" not in a.subscribe
        assert f"{_NS}.tools.heartbeat.pod-A" not in a.publish
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
        b = build_permissions(Principal.AGENT_POD, agent_id="agent-B", pod_id="pod-B")
        assert f"{_NS}.tools.internal.agent-B.>" not in a.subscribe
        assert f"{_NS}.tools.probe.agent-B.>" not in a.subscribe
        assert f"{_NS}.tools.heartbeat.agent-B.>" not in a.publish
        assert f"{_NS}.tools.internal.agent-A.>" not in b.subscribe
        assert f"{_NS}.tools.probe.agent-A.>" not in b.subscribe
        assert f"{_NS}.tools.heartbeat.agent-A.>" not in b.publish

    def test_agent_in_process_tool_subjects_are_independent_of_the_connect_name(self) -> None:
        # SAME authenticated agent, DIFFERENT connect-name pod ids (replicas): the in-process tool
        # grants are identical because they are scoped on the agent subtree, NOT the pod id. this is
        # what lets a tenant set any connect ``name`` (even a peer pod's) without ever shifting its
        # tool grant onto a peer agent's identity -- the connect name simply does not feed these.
        p1 = build_permissions(Principal.AGENT_POD, agent_id="agent-A", pod_id="pod-1")
        p2 = build_permissions(Principal.AGENT_POD, agent_id="agent-A", pod_id="victim-pod-id")
        tool_subjects = lambda perm: sorted(  # noqa: E731 -- terse local for the assertion
            s
            for s in _all_subjects(perm)
            if ".tools.internal." in s or ".tools.probe." in s or ".tools.heartbeat." in s
        )
        assert (
            tool_subjects(p1)
            == tool_subjects(p2)
            == [
                f"{_NS}.tools.heartbeat.agent-A.>",
                f"{_NS}.tools.internal.agent-A.>",
                f"{_NS}.tools.probe.agent-A.>",
            ]
        )

    def test_agent_pod_may_publish_its_own_tool_call_audit(self) -> None:
        # serving builtins in-process means the in-process tool server emits the baseline
        # ``tool.call`` audit envelope on every dispatch (mirrors ``_tool_pod``). audit
        # non-repudiation is REQUIRED on this platform, so the grant is mandatory -- without
        # it the actor/audit row for an agent-served tool call would be silently dropped.
        a = build_permissions(Principal.AGENT_POD, agent_id="agent-A", pod_id="pod-A")
        assert f"{_NS}.audit.tool.call" in a.publish

    def test_agent_pod_holds_proxy_assertion_nonce_bucket(self) -> None:
        # the in-process tool server verifies the proxy's body-bound assertion under enforce
        # and records single-use nonces in this KV bucket (mirrors ``_tool_pod``); without the
        # grant the agent could not serve its own builtins under enforced connection-auth.
        a = build_permissions(Principal.AGENT_POD, agent_id="agent-A", pod_id="pod-A")
        assert f"{_NS}-proxy_assertion_nonces" in a.kv_buckets

    def test_agent_pod_heartbeat_and_reregister_are_agent_scoped(self) -> None:
        # the agent_id leads heartbeat / reregister subjects as the
        # AUTHENTICATED segment (token-hash->DB), so a pod can publish
        # heartbeats and receive reregister nudges only under its OWN
        # agent -- it cannot forge a peer agent's heartbeat (B2) nor hold
        # a peer agent's reregister grant.
        a = build_permissions(Principal.AGENT_POD, agent_id="agent-A", pod_id="pod-A")
        assert f"{_NS}.agents.heartbeat.agent-A.pod-A" in a.publish
        assert f"{_NS}.agents.reregister_request.agent-A.pod-A" in a.subscribe
        # a peer agent's heartbeat / reregister subjects are NOT granted.
        b = build_permissions(Principal.AGENT_POD, agent_id="agent-B", pod_id="pod-B")
        assert f"{_NS}.agents.heartbeat.agent-B.pod-B" not in a.publish
        assert f"{_NS}.agents.reregister_request.agent-B.pod-B" not in a.subscribe
        assert f"{_NS}.agents.heartbeat.agent-A.pod-A" not in b.publish
        # the spoofable-pod-only legacy single-segment grant is gone, and no
        # wildcard heartbeat publish exists.
        assert f"{_NS}.agents.heartbeat.pod-A" not in a.publish
        assert f"{_NS}.agents.heartbeat.*" not in a.publish
        assert f"{_NS}.agents.heartbeat.>" not in a.publish
        assert f"{_NS}.agents.reregister_request.pod-A" not in a.subscribe


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
        perm = build_permissions(Principal.TOOL_POD, pod_id="pod-X")
        assert f"{_NS}.tools.internal.pod-X" in perm.subscribe

    def test_engagement_scope_resolve_grant_is_pod_publish_hub_subscribe(self) -> None:
        # engagement scope (consumer A of the §2 keystone): the consuming tool pod
        # PUBLISHES the resolve (forwarding the invoking agent's identity token);
        # the hub SUBSCRIBES to answer. mirrors the hub_object_resolve split.
        pod = build_permissions(Principal.TOOL_POD, pod_id="pod-X")
        assert f"{_NS}.hub.engagement.scope" in pod.publish
        hub = _build(Principal.HUB)
        assert f"{_NS}.hub.engagement.scope" in hub.subscribe
        # it is read-only for the pod: no agent-side commit twin exists (unlike
        # objects), and the pod never subscribes the scope subject.
        assert f"{_NS}.hub.engagement.scope" not in pod.subscribe

    def test_agent_can_reach_l3_and_gateway(self) -> None:
        perm = _build(Principal.AGENT_POD)
        assert f"{_NS}.l3.query" in perm.publish
        assert f"{_NS}.l3.tx.*" in perm.publish
        assert f"{_NS}.gateway.completion" in perm.publish
        # receives its streamed tokens on its OWN agent-scoped subject (W1);
        # a bare `gateway.stream.*` wildcard would let it sniff every other
        # customer's in-flight token stream.
        assert f"{_NS}.gateway.stream.agent-1.*" in perm.subscribe
        assert f"{_NS}.gateway.stream.*" not in perm.subscribe
        # and it publishes its hub token stream only under its own agent id
        # (hub.stream W1): a bare `hub.stream.*` publish grant would let it
        # forge/inject tokens onto a peer's in-flight request.
        assert f"{_NS}.hub.stream.agent-1.*" in perm.publish
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
        perm = build_permissions(Principal.TOOL_POD, pod_id="pod-1")
        assert f"{'prod7'}.tools.internal.pod-1" in perm.subscribe
        assert all(
            s.startswith("prod7.") or s == CROSS_PLATFORM_CACHE_INVALIDATE or s.startswith("_INBOX_")
            for s in _all_subjects(perm)
        )

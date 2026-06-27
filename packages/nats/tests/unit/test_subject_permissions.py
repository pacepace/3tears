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

_NS = "aibots"

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
            assert subj not in {">", "*", "_INBOX.>", "_INBOX.*"}, (
                f"{principal}: bare wildcard {subj!r}"
            )
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
            (Principal.TOOL_POD, [f"{_NS}.tools.register", f"{_NS}.hub.jwks"]),
            (
                Principal.REGISTRY,
                [f"{_NS}.tools.call", f"{_NS}.tools.internal.*", f"{_NS}.hub.jwks"],
            ),
            (Principal.HUB, [f"{_NS}.hub.handshake", f"{_NS}.hub.jwks", f"{_NS}.hub.secrets.request"]),
            (Principal.GATEWAY, [f"{_NS}.gateway.completion", f"{_NS}.gateway.embedding"]),
            (Principal.CHANNEL_ADAPTER, [f"{_NS}.channels.deliver.*", f"{_NS}.hub.channel.installs"]),
        ],
    )
    def test_boot_critical_subjects_present(
        self, principal: Principal, required: list[str]
    ) -> None:
        present = set(_all_subjects(_build(principal)))
        missing = [s for s in required if s not in present]
        assert not missing, f"{principal}: missing boot-critical {missing}"

    def test_tool_pod_subscribes_its_internal_call_subject(self) -> None:
        # without this the tool pod registers but never RECEIVES a proxied call.
        perm = build_permissions(Principal.TOOL_POD, pod_id="pod-X")
        assert f"{_NS}.tools.internal.pod-X" in perm.subscribe

    def test_agent_can_reach_l3_and_gateway(self) -> None:
        perm = _build(Principal.AGENT_POD)
        assert f"{_NS}.l3.query" in perm.publish
        assert f"{_NS}.l3.tx.*" in perm.publish
        assert f"{_NS}.gateway.completion" in perm.publish
        assert f"{_NS}.gateway.stream.*" in perm.subscribe  # receives its streamed tokens


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

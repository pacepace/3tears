"""unit tests for :mod:`threetears.nats.subjects`."""

from __future__ import annotations

from uuid import UUID

import pytest

from threetears.nats import (
    DEFAULT_NAMESPACE,
    Subject,
    Subjects,
    get_default_namespace,
    set_default_namespace,
)


@pytest.fixture(autouse=True)
def _reset_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """ensure each test starts with the default namespace."""
    monkeypatch.delenv("FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE", raising=False)
    set_default_namespace(DEFAULT_NAMESPACE)


def test_default_namespace_when_no_env() -> None:
    """without env var, default namespace is the documented constant."""
    assert get_default_namespace() == DEFAULT_NAMESPACE


def test_namespace_overridable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """env var observable at call time, not import time."""
    monkeypatch.setenv("FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE", "prod14")
    # set_default_namespace was called in fixture so env wins only after we reset
    # explicit set takes priority in this implementation; verify the fallback
    # path by clearing the contextvar
    from threetears.nats.subjects import _namespace_var

    _namespace_var.set(None)
    assert get_default_namespace() == "prod14"


def test_set_default_namespace_rejects_empty() -> None:
    """empty namespace is invalid."""
    with pytest.raises(ValueError):
        set_default_namespace("")


def test_subject_str_returns_path() -> None:
    """str(Subject) produces the dotted subject string."""
    sub = Subjects.tools_call()
    assert str(sub) == sub.path == "aibots.tools.call"


def test_subject_raw_validates_input() -> None:
    """Subject.raw rejects empty input."""
    with pytest.raises(ValueError):
        Subject.raw("")


def test_subject_raw_default_kind_point() -> None:
    """Subject.raw default kind is 'point'."""
    sub = Subject.raw("aibots.custom.thing")
    assert sub.kind == "point"


def test_agent_subjects_namespace_prefix() -> None:
    """agent subject builders include the configured namespace."""
    agent_id = UUID("019470a8-b5c3-7def-8123-456789abcdef")
    pod_id = "pod-abc"

    assert Subjects.agent_register().path == "aibots.agents.register"
    assert Subjects.agent_heartbeat(pod_id).path == "aibots.agents.heartbeat.pod-abc"
    assert Subjects.agent_heartbeat_wildcard().path == "aibots.agents.heartbeat.>"
    assert Subjects.agent_heartbeat_wildcard().kind == "pattern"
    assert Subjects.agent_route(agent_id).path == ("aibots.agents.route.019470a8-b5c3-7def-8123-456789abcdef")
    assert Subjects.agent_route_wildcard().path == "aibots.agents.route.>"
    assert Subjects.agent_internal(agent_id, pod_id).path == (
        "aibots.agents.internal.019470a8-b5c3-7def-8123-456789abcdef.pod-abc"
    )


def test_tools_subjects() -> None:
    """tool subject builders produce documented shapes."""
    pod_id = "tool-pod-xyz"
    assert Subjects.tools_register().path == "aibots.tools.register"
    assert Subjects.tools_discover().path == "aibots.tools.discover"
    assert Subjects.tools_call().path == "aibots.tools.call"
    assert Subjects.tools_heartbeat(pod_id).path == "aibots.tools.heartbeat.tool-pod-xyz"
    assert Subjects.tools_heartbeat_wildcard().path == "aibots.tools.heartbeat.>"
    assert Subjects.tools_internal(pod_id).path == "aibots.tools.internal.tool-pod-xyz"
    assert Subjects.tools_probe(pod_id).path == "aibots.tools.probe.tool-pod-xyz"


def test_gateway_subjects() -> None:
    """gateway subject builders produce documented shapes."""
    correlation_id = "corr-1"
    assert Subjects.gateway_completion().path == "aibots.gateway.completion"
    assert Subjects.gateway_embedding().path == "aibots.gateway.embedding"
    assert Subjects.gateway_health().path == "aibots.gateway.health"
    assert Subjects.gateway_stream(correlation_id).path == ("aibots.gateway.stream.corr-1")


def test_hub_subjects() -> None:
    """hub subject builders produce documented shapes."""
    correlation_id = "corr-9"
    assert Subjects.hub_handshake().path == "aibots.hub.handshake"
    assert Subjects.hub_secrets_request().path == "aibots.hub.secrets.request"
    assert Subjects.hub_user_resolve().path == "aibots.hub.user.resolve"
    assert Subjects.hub_usage_track().path == "aibots.hub.usage.track"
    assert Subjects.hub_stream(correlation_id).path == "aibots.hub.stream.corr-9"


def test_audit_subjects() -> None:
    """audit subject builders preserve event-type dots (subscribe hierarchy)."""
    assert Subjects.audit_event("workspace.doc_set").path == ("aibots.audit.workspace.doc_set")
    assert Subjects.audit_wildcard().path == "aibots.audit.>"
    assert Subjects.audit_wildcard(area="workspace").path == "aibots.audit.workspace.>"


def test_audit_event_rejects_empty() -> None:
    """audit_event requires a non-empty event_type."""
    with pytest.raises(ValueError):
        Subjects.audit_event("")


def test_l3_subjects() -> None:
    """l3 broker subject builders produce documented shapes.

    the platform broker exposes six per-op transaction subjects so DML
    ``execute``, single-row ``fetchrow``, and multi-row ``fetch`` are
    addressable independently.
    """
    assert Subjects.l3_query().path == "aibots.l3.query"
    assert Subjects.l3_batch().path == "aibots.l3.batch"
    assert Subjects.l3_tx("begin").path == "aibots.l3.tx.begin"
    assert Subjects.l3_tx("execute").path == "aibots.l3.tx.execute"
    assert Subjects.l3_tx("fetchrow").path == "aibots.l3.tx.fetchrow"
    assert Subjects.l3_tx("fetch").path == "aibots.l3.tx.fetch"
    assert Subjects.l3_tx("commit").path == "aibots.l3.tx.commit"
    assert Subjects.l3_tx("rollback").path == "aibots.l3.tx.rollback"


def test_acl_subjects() -> None:
    """acl invalidation subjects."""
    assert Subjects.acl_invalidate("membership").path == ("aibots.acl.membership.invalidate")
    assert Subjects.acl_invalidate("assignment").path == ("aibots.acl.assignment.invalidate")
    assert Subjects.acl_invalidate("role").path == "aibots.acl.role.invalidate"


def test_namespace_discover() -> None:
    """namespace discovery subject."""
    assert Subjects.namespace_discover().path == "aibots.namespace.discover"


def test_datasource_query() -> None:
    """datasource query subject."""
    assert Subjects.datasource_query("redshift_prod").path == ("aibots.datasource.redshift_prod.query")


def test_datasource_query_rejects_empty() -> None:
    """datasource_query requires non-empty name."""
    with pytest.raises(ValueError):
        Subjects.datasource_query("")


def test_cache_invalidate_is_namespace_independent() -> None:
    """cache invalidation subject is a cross-platform constant — no namespace prefix."""
    assert Subjects.cache_invalidate().path == "threetears.cache.invalidate"
    set_default_namespace("prod14")
    assert Subjects.cache_invalidate().path == "threetears.cache.invalidate"


def test_deadletter_uses_namespace() -> None:
    """deadletter subject is namespace-prefixed."""
    assert Subjects.deadletter("aibots.tools.call").path == ("aibots.deadletter.aibots.tools.call")


def test_dot_in_segment_is_sanitized() -> None:
    """dots in raw segment values are sanitized to '-'."""
    # model name like 'claude-sonnet-4.5' would round-trip with dot replaced
    sub = Subjects.datasource_query("redshift.prod")
    assert sub.path == "aibots.datasource.redshift-prod.query"


def test_namespace_change_observed_by_subsequent_calls() -> None:
    """changing namespace mid-process affects subsequent subject builders."""
    set_default_namespace("staging")
    assert Subjects.tools_call().path == "staging.tools.call"
    set_default_namespace("aibots")
    assert Subjects.tools_call().path == "aibots.tools.call"


def test_subject_is_frozen_dataclass() -> None:
    """Subject instances cannot be mutated."""
    sub = Subjects.tools_call()
    with pytest.raises(Exception):  # FrozenInstanceError subclasses AttributeError
        sub.path = "different"  # type: ignore[misc]

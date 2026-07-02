"""unit tests for :mod:`threetears.nats.subjects`."""

from __future__ import annotations

from uuid import UUID

import pytest

from threetears.nats import (
    NamespaceNotConfiguredError,
    Subject,
    Subjects,
    get_default_namespace,
    set_default_namespace,
)

_TEST_NAMESPACE = "3tears"


@pytest.fixture(autouse=True)
def _reset_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """ensure each test starts with a known namespace bound."""
    monkeypatch.delenv("THREETEARS_NATS_SUBJECT_NAMESPACE", raising=False)
    set_default_namespace(_TEST_NAMESPACE)


def test_get_default_namespace_raises_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """with no env var and no explicit set, resolution raises."""
    from threetears.nats.subjects import _namespace_var

    monkeypatch.delenv("THREETEARS_NATS_SUBJECT_NAMESPACE", raising=False)
    _namespace_var.set(None)
    with pytest.raises(NamespaceNotConfiguredError):
        get_default_namespace()


def test_namespace_overridable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """env var observable at call time, not import time."""
    monkeypatch.setenv("THREETEARS_NATS_SUBJECT_NAMESPACE", "prod14")
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
    assert str(sub) == sub.path == "3tears.tools.call"


def test_subject_raw_validates_input() -> None:
    """Subject.raw rejects empty input."""
    with pytest.raises(ValueError):
        Subject.raw("")


def test_subject_raw_default_kind_point() -> None:
    """Subject.raw default kind is 'point'."""
    sub = Subject.raw("3tears.custom.thing")
    assert sub.kind == "point"


def test_agent_subjects_namespace_prefix() -> None:
    """agent subject builders include the configured namespace."""
    agent_id = UUID("019470a8-b5c3-7def-8123-456789abcdef")
    pod_id = "pod-abc"

    assert Subjects.agent_register().path == "3tears.agents.register"
    assert Subjects.agent_deregister().path == "3tears.agents.deregister"
    assert Subjects.agent_deregister().kind == "point"
    assert Subjects.agent_heartbeat(pod_id).path == "3tears.agents.heartbeat.pod-abc"
    assert Subjects.agent_heartbeat_wildcard().path == "3tears.agents.heartbeat.>"
    assert Subjects.agent_heartbeat_wildcard().kind == "pattern"
    assert Subjects.agent_reregister_request(pod_id).path == "3tears.agents.reregister_request.pod-abc"
    assert Subjects.agent_reregister_request(pod_id).kind == "point"
    assert Subjects.agent_route(agent_id).path == ("3tears.agents.route.019470a8-b5c3-7def-8123-456789abcdef")
    assert Subjects.agent_route_wildcard().path == "3tears.agents.route.>"
    assert Subjects.agent_internal(agent_id, pod_id).path == (
        "3tears.agents.internal.019470a8-b5c3-7def-8123-456789abcdef.pod-abc"
    )


def test_tools_subjects() -> None:
    """tool subject builders produce documented shapes."""
    pod_id = "tool-pod-xyz"
    assert Subjects.tools_register().path == "3tears.tools.register"
    assert Subjects.tools_discover().path == "3tears.tools.discover"
    assert Subjects.tools_call().path == "3tears.tools.call"
    assert Subjects.tools_heartbeat(pod_id).path == "3tears.tools.heartbeat.tool-pod-xyz"
    assert Subjects.tools_heartbeat_wildcard().path == "3tears.tools.heartbeat.>"
    assert Subjects.tools_internal(pod_id).path == "3tears.tools.internal.tool-pod-xyz"
    assert Subjects.tools_probe(pod_id).path == "3tears.tools.probe.tool-pod-xyz"


def test_gateway_subjects() -> None:
    """gateway subject builders produce documented shapes."""
    correlation_id = "corr-1"
    assert Subjects.gateway_completion().path == "3tears.gateway.completion"
    assert Subjects.gateway_embedding().path == "3tears.gateway.embedding"
    assert Subjects.gateway_health().path == "3tears.gateway.health"
    assert Subjects.gateway_stream(correlation_id).path == ("3tears.gateway.stream.corr-1")


def test_hub_subjects() -> None:
    """hub subject builders produce documented shapes."""
    correlation_id = "corr-9"
    assert Subjects.hub_handshake().path == "3tears.hub.handshake"
    assert Subjects.hub_secrets_request().path == "3tears.hub.secrets.request"
    assert Subjects.hub_user_resolve().path == "3tears.hub.user.resolve"
    assert Subjects.hub_usage_track().path == "3tears.hub.usage.track"
    assert Subjects.hub_stream(correlation_id).path == "3tears.hub.stream.corr-9"


def test_audit_subjects() -> None:
    """audit subject builders preserve event-type dots (subscribe hierarchy)."""
    assert Subjects.audit_event("workspace.doc_set").path == ("3tears.audit.workspace.doc_set")
    assert Subjects.audit_wildcard().path == "3tears.audit.>"
    assert Subjects.audit_wildcard(area="workspace").path == "3tears.audit.workspace.>"


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
    assert Subjects.l3_query().path == "3tears.l3.query"
    assert Subjects.l3_batch().path == "3tears.l3.batch"
    assert Subjects.l3_tx("begin").path == "3tears.l3.tx.begin"
    assert Subjects.l3_tx("execute").path == "3tears.l3.tx.execute"
    assert Subjects.l3_tx("fetchrow").path == "3tears.l3.tx.fetchrow"
    assert Subjects.l3_tx("fetch").path == "3tears.l3.tx.fetch"
    assert Subjects.l3_tx("commit").path == "3tears.l3.tx.commit"
    assert Subjects.l3_tx("rollback").path == "3tears.l3.tx.rollback"


def test_acl_subjects() -> None:
    """acl invalidation subjects."""
    assert Subjects.acl_invalidate("membership").path == ("3tears.acl.membership.invalidate")
    assert Subjects.acl_invalidate("assignment").path == ("3tears.acl.assignment.invalidate")
    assert Subjects.acl_invalidate("role").path == "3tears.acl.role.invalidate"


def test_capabilities_epoch_under_single_product_namespace() -> None:
    """single-product builder produces ``<ns>.capabilities.epoch`` (no product segment).

    a single-product namespace has nothing after the namespace to
    disambiguate against. asymmetric on purpose vs the 3tears-bound
    builders, which carry a product segment.
    """
    set_default_namespace("app")
    assert Subjects.capabilities_epoch().path == "app.capabilities.epoch"
    set_default_namespace("3tears")


def test_gateway_catalog_epoch_under_3tears_namespace() -> None:
    """3tears-bound builder produces ``3tears.gateway.catalog.epoch`` (with product segment)."""
    assert Subjects.gateway_catalog_epoch().path == "3tears.gateway.catalog.epoch"


def test_mcp_rbac_epoch_under_3tears_namespace() -> None:
    """3tears-bound builder produces ``3tears.mcp.rbac.epoch`` (with product segment)."""
    assert Subjects.mcp_rbac_epoch().path == "3tears.mcp.rbac.epoch"


def test_epoch_subjects_track_namespace_changes() -> None:
    """epoch subjects honour the bound namespace at call time, like every other Subject."""
    set_default_namespace("app-staging")
    assert Subjects.capabilities_epoch().path == "app-staging.capabilities.epoch"
    set_default_namespace("3tears-staging")
    assert Subjects.gateway_catalog_epoch().path == "3tears-staging.gateway.catalog.epoch"
    assert Subjects.mcp_rbac_epoch().path == "3tears-staging.mcp.rbac.epoch"
    set_default_namespace("3tears")


def test_epoch_subject_path_is_row_pk_identity() -> None:
    """subject path is also the ``config_epochs`` row PK; same string both places.

    regression-frame: identity binding is part of the contract; if the
    builder path diverges from the EpochClient row-key encoding, every
    cross-pod coordination silently fails.
    """
    # under 3tears namespace, gateway catalog and mcp rbac collide on
    # nothing because their paths are distinct -- structural assertion.
    set_default_namespace("3tears")
    catalog = Subjects.gateway_catalog_epoch().path
    rbac = Subjects.mcp_rbac_epoch().path
    assert catalog != rbac
    # under a single-product namespace, capabilities epoch is different
    # from any 3tears-namespace path (no cross-product collision).
    set_default_namespace("app")
    capabilities = Subjects.capabilities_epoch().path
    assert capabilities not in (catalog, rbac)
    set_default_namespace("3tears")


def test_namespace_discover() -> None:
    """namespace discovery subject."""
    assert Subjects.namespace_discover().path == "3tears.namespace.discover"


def test_datasource_query() -> None:
    """datasource query subject."""
    assert Subjects.datasource_query("redshift_prod").path == ("3tears.datasource.redshift_prod.query")


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
    assert Subjects.deadletter("3tears.tools.call").path == ("3tears.deadletter.3tears.tools.call")


def test_dot_in_segment_is_sanitized() -> None:
    """dots in raw segment values are sanitized to '-'."""
    # model name like 'claude-sonnet-4.5' would round-trip with dot replaced
    sub = Subjects.datasource_query("redshift.prod")
    assert sub.path == "3tears.datasource.redshift-prod.query"


def test_namespace_change_observed_by_subsequent_calls() -> None:
    """changing namespace mid-process affects subsequent subject builders."""
    set_default_namespace("staging")
    assert Subjects.tools_call().path == "staging.tools.call"
    set_default_namespace("3tears")
    assert Subjects.tools_call().path == "3tears.tools.call"


def test_subject_is_frozen_dataclass() -> None:
    """Subject instances cannot be mutated."""
    sub = Subjects.tools_call()
    with pytest.raises(Exception):  # FrozenInstanceError subclasses AttributeError
        sub.path = "different"  # type: ignore[misc]


def test_room_subject_is_namespaced_and_sha256_tokened() -> None:
    """room subject is ``{ns}.channels.room.{sha256hex(room_id)}``."""
    import hashlib

    room_id = "cust:story-1:main:scene.md"
    expected_token = hashlib.sha256(room_id.encode("utf-8")).hexdigest()
    sub = Subjects.room(room_id)
    assert sub.path == f"3tears.channels.room.{expected_token}"
    assert sub.kind == "point"
    # the token is subject-safe: lowercase hex only, no separators/wildcards.
    assert set(expected_token) <= set("0123456789abcdef")
    assert len(expected_token) == 64


def test_room_subject_is_deterministic() -> None:
    """the same room id always derives the same subject."""
    room_id = "cust:story-1:main:scene.md"
    assert Subjects.room(room_id).path == Subjects.room(room_id).path


def test_room_subject_distinct_ids_distinct_subjects() -> None:
    """two distinct room ids derive distinct subjects."""
    a = Subjects.room("cust:story-1:main:scene.md")
    b = Subjects.room("cust:story-1:main:other.md")
    assert a.path != b.path


def test_room_subject_handles_out_of_grammar_room_ids() -> None:
    """colons, dots, spaces, and NATS wildcards all yield a valid subject token.

    a raw room id carrying characters illegal/ambiguous in a NATS subject
    (space, ``*``, ``>``, ``.``) must NOT leak into the subject token —
    the SHA-256 digest is always pure hex, so the resulting subject is a
    single, valid, point subject regardless of the room id's contents.
    """
    nasty = "cust:my story:main:a *weird* file > name.md"
    sub = Subjects.room(nasty)
    token = sub.path.rsplit(".", 1)[-1]
    assert set(token) <= set("0123456789abcdef")
    assert len(token) == 64
    # no overloaded separators / wildcards bled into the subject path.
    for illegal in (" ", "*", ">"):
        assert illegal not in sub.path
    assert sub.path.startswith("3tears.channels.room.")


def test_room_subject_rejects_empty_room_id() -> None:
    """an empty room id is a programming error."""
    with pytest.raises(ValueError):
        Subjects.room("")


def test_knowledge_draft_subject() -> None:
    """correction-harvest draft subject is namespace-prefixed (knowledge-task-06)."""
    assert Subjects.knowledge_draft().path == "3tears.knowledge.draft"


def test_knowledge_draft_subject_honors_namespace() -> None:
    """the knowledge-draft subject picks up the active namespace prefix."""
    set_default_namespace("staging")
    assert Subjects.knowledge_draft().path == "staging.knowledge.draft"
    set_default_namespace("3tears")
    assert Subjects.knowledge_draft().path == "3tears.knowledge.draft"

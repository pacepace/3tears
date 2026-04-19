"""unit tests for the rbac audit vocabulary (audit-task-01 Phase 4).

audit-task-01 Phase 4 deleted the per-domain :class:`RbacAuditEnvelope`
in favour of the unified :class:`threetears.agent.audit.AuditEvent`.
the closed-set vocabulary tuples + ``Literal`` aliases live on so the
publisher can validate event_type / action / resource_type strings
before handing the unified envelope to ``publish_audit``.

these tests pin the exact contents of those closed sets so a publisher
adding a new event type without extending the vocabulary fails its own
publish-side guard rather than silently leaking through.
"""

from __future__ import annotations

from threetears.agent.acl.audit_vocabulary import (
    RBAC_AUDIT_ACTIONS,
    RBAC_AUDIT_EVENT_TYPES,
    RBAC_AUDIT_RESOURCE_TYPES,
)


def test_event_types_contain_exactly_fourteen_strings() -> None:
    """the rbac event vocabulary stays at exactly 14 entries.

    new event types require explicit shard work (publisher emit + admin
    UI exposure + docs + audit-row taxonomy review). the cardinality
    check catches accidental additions in code review.
    """
    assert len(RBAC_AUDIT_EVENT_TYPES) == 14


def test_event_types_match_expected_set() -> None:
    """every declared rbac event type is named explicitly here.

    the test pins the exact set so an accidental rename or removal
    fails loud rather than silently breaks the publisher's validate-
    before-emit guard or the admin UI's filter dropdown.
    """
    expected = {
        "rbac.group.create",
        "rbac.group.update",
        "rbac.group.delete",
        "rbac.group.member.add",
        "rbac.group.member.remove",
        "rbac.role.create",
        "rbac.role.update",
        "rbac.role.delete",
        "rbac.assignment.create",
        "rbac.assignment.delete",
        "rbac.introspect.explain",
        "rbac.introspect.effective",
        "rbac.introspect.namespace_access",
        "rbac.introspect.dry_run",
    }
    assert set(RBAC_AUDIT_EVENT_TYPES) == expected


def test_event_types_are_dotted() -> None:
    """every event_type follows the dotted ``{domain}.{verb}`` shape.

    matches the unified :class:`AuditEvent` envelope's
    ``event_type`` validator so the rbac publisher's strings remain
    legal at the wire layer too.
    """
    for event_type in RBAC_AUDIT_EVENT_TYPES:
        assert "." in event_type
        assert event_type.startswith("rbac.")


def test_resource_types_match_expected_set() -> None:
    """resource_namespace_type vocabulary stays at the four rbac shapes."""
    assert set(RBAC_AUDIT_RESOURCE_TYPES) == {
        "group",
        "role",
        "assignment",
        "introspection",
    }


def test_actions_match_expected_set() -> None:
    """action verb vocabulary stays at the nine rbac verbs."""
    assert set(RBAC_AUDIT_ACTIONS) == {
        "create",
        "update",
        "delete",
        "add_member",
        "remove_member",
        "explain",
        "effective",
        "namespace_access",
        "dry_run",
    }


def test_event_types_tuple_is_immutable() -> None:
    """vocabulary is a tuple so accidental ``.append`` fails loud."""
    assert isinstance(RBAC_AUDIT_EVENT_TYPES, tuple)
    assert isinstance(RBAC_AUDIT_RESOURCE_TYPES, tuple)
    assert isinstance(RBAC_AUDIT_ACTIONS, tuple)

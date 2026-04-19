"""unit tests for :class:`threetears.agent.acl.audit.RbacAuditEnvelope`.

covers the envelope's anti-drift guarantees (``extra='forbid'``, closed
``event_type``, ``resource_type``, ``action`` literals; timezone-aware
``date_occurred`` validator) and round-trip
``model_dump_json`` / ``model_validate_json`` semantics used by the
NATS publisher + consumer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from threetears.agent.acl.audit import (
    RBAC_AUDIT_ACTIONS,
    RBAC_AUDIT_EVENT_TYPES,
    RBAC_AUDIT_RESOURCE_TYPES,
    RbacAuditEnvelope,
)


def _fixed_uuid() -> UUID:
    return UUID("00000000-0000-4000-8000-000000000001")


def _base_kwargs() -> dict[str, object]:
    return {
        "event_type": "rbac.group.create",
        "actor_user_id": _fixed_uuid(),
        "customer_id": None,
        "correlation_id": _fixed_uuid(),
        "date_occurred": datetime.now(UTC),
        "resource_type": "group",
        "resource_id": _fixed_uuid(),
        "action": "create",
        "details": {"name": "Eng"},
    }


def test_constructs_with_all_required_fields() -> None:
    """envelope builds cleanly when every required field is supplied."""
    envelope = RbacAuditEnvelope(**_base_kwargs())  # type: ignore[arg-type]

    assert envelope.event_type == "rbac.group.create"
    assert envelope.resource_type == "group"
    assert envelope.action == "create"


@pytest.mark.parametrize("event_type", RBAC_AUDIT_EVENT_TYPES)
def test_every_declared_event_type_parses(event_type: str) -> None:
    """every member of RBAC_AUDIT_EVENT_TYPES constructs the envelope."""
    kwargs = _base_kwargs()
    kwargs["event_type"] = event_type
    # pick a matching resource_type for the event so literal validation
    # passes; introspection events key "introspection", all others key
    # their table.
    if event_type.startswith("rbac.introspect."):
        kwargs["resource_type"] = "introspection"
        kwargs["resource_id"] = None
        kwargs["action"] = event_type.rsplit(".", 1)[1]
    elif event_type.startswith("rbac.role."):
        kwargs["resource_type"] = "role"
        kwargs["action"] = event_type.rsplit(".", 1)[1]
    elif event_type.startswith("rbac.assignment."):
        kwargs["resource_type"] = "assignment"
        kwargs["action"] = event_type.rsplit(".", 1)[1]
    elif "member" in event_type:
        kwargs["resource_type"] = "group"
        kwargs["action"] = (
            "add_member" if event_type.endswith(".add") else "remove_member"
        )
    else:
        kwargs["resource_type"] = "group"
        kwargs["action"] = event_type.rsplit(".", 1)[1]

    envelope = RbacAuditEnvelope(**kwargs)  # type: ignore[arg-type]

    assert envelope.event_type == event_type


def test_unknown_event_type_rejected() -> None:
    """``extra='forbid'`` + closed Literal rejects a never-declared event."""
    kwargs = _base_kwargs()
    kwargs["event_type"] = "rbac.group.nothappening"

    with pytest.raises(ValidationError):
        RbacAuditEnvelope(**kwargs)  # type: ignore[arg-type]


def test_unknown_resource_type_rejected() -> None:
    """closed resource_type Literal rejects unknown resource."""
    kwargs = _base_kwargs()
    kwargs["resource_type"] = "unknown"

    with pytest.raises(ValidationError):
        RbacAuditEnvelope(**kwargs)  # type: ignore[arg-type]


def test_unknown_action_rejected() -> None:
    """closed action Literal rejects unknown verb."""
    kwargs = _base_kwargs()
    kwargs["action"] = "launch"

    with pytest.raises(ValidationError):
        RbacAuditEnvelope(**kwargs)  # type: ignore[arg-type]


def test_extra_fields_rejected() -> None:
    """``extra='forbid'`` rejects any field not declared on the envelope."""
    kwargs = _base_kwargs()
    kwargs["mystery_field"] = "oops"

    with pytest.raises(ValidationError):
        RbacAuditEnvelope(**kwargs)  # type: ignore[arg-type]


def test_naive_date_occurred_rejected() -> None:
    """``date_occurred`` must carry a tzinfo."""
    kwargs = _base_kwargs()
    kwargs["date_occurred"] = datetime(2026, 1, 1, 12, 0, 0)

    with pytest.raises(ValidationError):
        RbacAuditEnvelope(**kwargs)  # type: ignore[arg-type]


def test_permissions_before_after_optional() -> None:
    """snapshots default to None (introspection events don't carry them)."""
    envelope = RbacAuditEnvelope(**_base_kwargs())  # type: ignore[arg-type]

    assert envelope.permissions_before is None
    assert envelope.permissions_after is None


def test_permissions_before_after_round_trip() -> None:
    """JSON round trip preserves snapshot shape."""
    kwargs = _base_kwargs()
    kwargs["permissions_before"] = {
        "b3cba87c-4d0f-4000-8000-000000000001": ["read", "write"],
    }
    kwargs["permissions_after"] = {
        "b3cba87c-4d0f-4000-8000-000000000001": ["read"],
    }

    envelope = RbacAuditEnvelope(**kwargs)  # type: ignore[arg-type]
    wire = envelope.model_dump_json()
    decoded = RbacAuditEnvelope.model_validate_json(wire)

    assert decoded.permissions_before == envelope.permissions_before
    assert decoded.permissions_after == envelope.permissions_after


def test_action_and_resource_sets_are_closed() -> None:
    """the Literal values and the module-level tuples stay in sync."""
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
    assert set(RBAC_AUDIT_RESOURCE_TYPES) == {
        "group",
        "role",
        "assignment",
        "introspection",
    }

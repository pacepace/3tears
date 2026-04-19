"""unit tests for :class:`threetears.agent.audit.AuditEvent`.

covers the envelope's anti-drift guarantees (``extra='forbid'``,
timezone-aware ``timestamp``, closed ``outcome``, dotted
``event_type``) and round-trip ``model_dump_json`` /
``model_validate_json`` semantics used by the NATS publisher +
consumer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid7

import pytest
from pydantic import ValidationError

from threetears.agent.audit import AuditEvent


def _base_kwargs() -> dict[str, object]:
    """
    build a baseline valid kwargs dict for :class:`AuditEvent`.

    :return: kwargs dict populated with a consistent set of field
        values; individual tests mutate one key to exercise a
        specific validator
    :rtype: dict[str, object]
    """
    return {
        "id": uuid7(),
        "timestamp": datetime.now(UTC),
        "event_type": "workspace.fs_write",
        "actor_user_id": uuid7(),
        "calling_agent_id": uuid7(),
        "owner_agent_id": uuid7(),
        "customer_id": uuid7(),
        "resource_namespace_id": uuid7(),
        "resource_namespace_type": "workspace",
        "action": "write",
        "outcome": "success",
        "correlation_id": uuid7(),
        "details": {"sha256": "abc123", "bytes": 42},
    }


def test_constructs_with_all_required_fields() -> None:
    """envelope builds cleanly when every field is supplied."""
    envelope = AuditEvent(**_base_kwargs())  # type: ignore[arg-type]

    assert envelope.event_type == "workspace.fs_write"
    assert envelope.action == "write"
    assert envelope.outcome == "success"


def test_optional_identity_fields_default_none() -> None:
    """envelope builds with only the required scalar axes populated."""
    envelope = AuditEvent(
        id=uuid7(),
        timestamp=datetime.now(UTC),
        event_type="platform.role.create",
        action="create",
        correlation_id=uuid7(),
    )

    assert envelope.actor_user_id is None
    assert envelope.calling_agent_id is None
    assert envelope.owner_agent_id is None
    assert envelope.customer_id is None
    assert envelope.resource_namespace_id is None
    assert envelope.resource_namespace_type is None
    assert envelope.outcome == "success"
    assert envelope.details == {}


def test_naive_timestamp_rejected() -> None:
    """``timestamp`` must carry a tzinfo; a naive datetime raises."""
    kwargs = _base_kwargs()
    kwargs["timestamp"] = datetime(2026, 1, 1, 12, 0, 0)

    with pytest.raises(ValidationError):
        AuditEvent(**kwargs)  # type: ignore[arg-type]


def test_unknown_outcome_rejected() -> None:
    """``outcome`` is closed to success/failure/error."""
    kwargs = _base_kwargs()
    kwargs["outcome"] = "partial"

    with pytest.raises(ValidationError):
        AuditEvent(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("outcome", ["success", "failure", "error"])
def test_every_declared_outcome_accepted(outcome: str) -> None:
    """each recognised outcome string constructs the envelope."""
    kwargs = _base_kwargs()
    kwargs["outcome"] = outcome

    envelope = AuditEvent(**kwargs)  # type: ignore[arg-type]

    assert envelope.outcome == outcome


def test_non_dotted_event_type_rejected() -> None:
    """``event_type`` must be dotted ``{domain}.{verb}``."""
    kwargs = _base_kwargs()
    kwargs["event_type"] = "justaword"

    with pytest.raises(ValidationError):
        AuditEvent(**kwargs)  # type: ignore[arg-type]


def test_empty_event_type_rejected() -> None:
    """an empty ``event_type`` string is rejected."""
    kwargs = _base_kwargs()
    kwargs["event_type"] = ""

    with pytest.raises(ValidationError):
        AuditEvent(**kwargs)  # type: ignore[arg-type]


def test_extra_fields_rejected() -> None:
    """``extra='forbid'`` rejects any field not declared on the envelope."""
    kwargs = _base_kwargs()
    kwargs["mystery_field"] = "oops"

    with pytest.raises(ValidationError):
        AuditEvent(**kwargs)  # type: ignore[arg-type]


def test_round_trip_preserves_every_field() -> None:
    """JSON round trip preserves UUID, datetime, and details shape."""
    kwargs = _base_kwargs()
    envelope = AuditEvent(**kwargs)  # type: ignore[arg-type]

    wire = envelope.model_dump_json()
    decoded = AuditEvent.model_validate_json(wire)

    assert decoded == envelope
    assert decoded.details == envelope.details


def test_details_defaults_to_empty_dict() -> None:
    """omitting ``details`` yields an empty dict (not None)."""
    kwargs = _base_kwargs()
    del kwargs["details"]

    envelope = AuditEvent(**kwargs)  # type: ignore[arg-type]

    assert envelope.details == {}


def test_id_round_trips_as_uuid() -> None:
    """``id`` is a UUID object, not a string, after JSON round trip."""
    kwargs = _base_kwargs()
    envelope = AuditEvent(**kwargs)  # type: ignore[arg-type]
    decoded = AuditEvent.model_validate_json(envelope.model_dump_json())

    assert isinstance(decoded.id, UUID)
    assert decoded.id == envelope.id

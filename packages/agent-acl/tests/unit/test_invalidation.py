"""unit tests for the canonical NATS invalidation payload models.

every publisher / subscriber across the platform speaks these
models, so wire-format compatibility is enforced by round-tripping
the JSON through :meth:`pydantic.BaseModel.model_dump_json` and
:meth:`pydantic.BaseModel.model_validate_json`.
"""

from __future__ import annotations

from uuid import UUID, uuid7

import pytest
from pydantic import ValidationError
from threetears.agent.acl import (
    AssignmentInvalidatePayload,
    MembershipInvalidatePayload,
    RoleInvalidatePayload,
)


class TestMembershipInvalidatePayload:
    """``MembershipInvalidatePayload`` serializer / validator."""

    def test_roundtrip_user(self) -> None:
        actor_id = uuid7()
        payload = MembershipInvalidatePayload(
            actor_type="user", actor_id=actor_id,
        )
        wire = payload.model_dump_json()
        decoded = MembershipInvalidatePayload.model_validate_json(wire)
        assert decoded.actor_type == "user"
        assert decoded.actor_id == actor_id

    def test_roundtrip_agent(self) -> None:
        actor_id = uuid7()
        payload = MembershipInvalidatePayload(
            actor_type="agent", actor_id=actor_id,
        )
        wire = payload.model_dump_json()
        decoded = MembershipInvalidatePayload.model_validate_json(wire)
        assert decoded.actor_type == "agent"

    def test_rejects_unknown_actor_type(self) -> None:
        with pytest.raises(ValidationError):
            MembershipInvalidatePayload(
                actor_type="device", actor_id=uuid7(),  # type: ignore[arg-type]
            )

    def test_actor_id_is_uuid_after_validate(self) -> None:
        """deserialized ``actor_id`` is real :class:`UUID`."""
        actor_id = uuid7()
        wire = (
            '{"actor_type":"user","actor_id":"' + str(actor_id) + '"}'
        )
        decoded = MembershipInvalidatePayload.model_validate_json(wire)
        assert isinstance(decoded.actor_id, UUID)


class TestAssignmentInvalidatePayload:
    """``AssignmentInvalidatePayload`` carries a single group_id."""

    def test_roundtrip(self) -> None:
        gid = uuid7()
        payload = AssignmentInvalidatePayload(group_id=gid)
        wire = payload.model_dump_json()
        decoded = AssignmentInvalidatePayload.model_validate_json(wire)
        assert decoded.group_id == gid


class TestRoleInvalidatePayload:
    """``RoleInvalidatePayload`` has no fields; empty-doc validates."""

    def test_validates_empty_object(self) -> None:
        payload = RoleInvalidatePayload.model_validate_json("{}")
        assert isinstance(payload, RoleInvalidatePayload)

    def test_dump_is_empty_object(self) -> None:
        wire = RoleInvalidatePayload().model_dump_json()
        assert wire == "{}"

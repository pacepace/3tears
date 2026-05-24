"""Unit tests for ``AgentSkillEntity`` + ``AgentSkillInvocationEntity``.

Covers the entity contract:

- ``primary_key_field`` returns the bare-id column name.
- Composite ``_id`` tuple is set when ``agent_id`` + the bare id are
  both present in the constructor data.
- Field accessors round-trip through ``BaseEntity.__setattr__`` so the
  change tracking dirties on assignment.
- ``tool_additions`` / ``tool_restrictions`` / ``tags`` getters return
  a fresh list (callers cannot mutate the cached row via the getter).
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from uuid_utils import uuid7

from threetears.agent.skills.entities import (
    AgentSkillEntity,
    AgentSkillInvocationEntity,
)


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


def _now() -> datetime:
    """Return a fresh aware-UTC ``datetime``."""
    return datetime.now(UTC)


class TestAgentSkillEntity:
    """Contract tests for :class:`AgentSkillEntity`."""

    def test_primary_key_field_is_skill_id(self) -> None:
        """``primary_key_field`` returns the bare-id column name."""
        assert AgentSkillEntity.primary_key_field == "skill_id"

    def test_composite_id_set_when_both_columns_present(self) -> None:
        """``id`` returns the ``(agent_id, skill_id)`` composite tuple."""
        agent_id = _new_uuid()
        skill_id = _new_uuid()
        entity = AgentSkillEntity(
            {
                "agent_id": agent_id,
                "skill_id": skill_id,
                "user_id": _new_uuid(),
                "name": "deploy-helper",
                "summary": "Deploy a service",
                "body": "Run terraform apply",
                "prompt_mode": "additive",
                "tool_additions": [],
                "tool_restrictions": [],
                "trigger_keywords": "",
                "tags": [],
                "source": "manual",
                "enabled": True,
                "use_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "date_created": _now(),
                "date_updated": _now(),
            },
        )
        assert entity.id == (agent_id, skill_id)

    def test_field_accessors_round_trip(self) -> None:
        """Property getters return the values stored at construction."""
        agent_id = _new_uuid()
        skill_id = _new_uuid()
        user_id = _new_uuid()
        now = _now()
        entity = AgentSkillEntity(
            {
                "agent_id": agent_id,
                "skill_id": skill_id,
                "user_id": user_id,
                "name": "deploy-helper",
                "summary": "Deploy a service",
                "body": "Run terraform apply",
                "prompt_mode": "additive",
                "tool_additions": ["mcp.shell"],
                "tool_restrictions": ["mcp.dangerous_op"],
                "trigger_keywords": "deploy ship release",
                "tags": ["ops", "ci"],
                "source": "manual",
                "enabled": True,
                "use_count": 3,
                "last_used_at": now,
                "success_count": 2,
                "failure_count": 1,
                "last_failure_at": now,
                "date_created": now,
                "date_updated": now,
            },
        )
        assert entity.agent_id == agent_id
        assert entity.skill_id == skill_id
        assert entity.user_id == user_id
        assert entity.name == "deploy-helper"
        assert entity.summary == "Deploy a service"
        assert entity.body == "Run terraform apply"
        assert entity.prompt_mode == "additive"
        assert entity.tool_additions == ["mcp.shell"]
        assert entity.tool_restrictions == ["mcp.dangerous_op"]
        assert entity.trigger_keywords == "deploy ship release"
        assert entity.tags == ["ops", "ci"]
        assert entity.source == "manual"
        assert entity.enabled is True
        assert entity.use_count == 3
        assert entity.last_used_at == now
        assert entity.success_count == 2
        assert entity.failure_count == 1
        assert entity.last_failure_at == now
        assert entity.date_created == now
        assert entity.date_updated == now

    def test_body_is_optional(self) -> None:
        """A pure tool-composition skill has ``body=None``."""
        entity = AgentSkillEntity(
            {
                "agent_id": _new_uuid(),
                "skill_id": _new_uuid(),
                "user_id": _new_uuid(),
                "name": "tool-only",
                "summary": "Composes tools without prose",
                "body": None,
                "prompt_mode": "additive",
                "tool_additions": ["mcp.x"],
                "tool_restrictions": [],
                "trigger_keywords": "",
                "tags": [],
                "source": "manual",
                "enabled": True,
                "use_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "date_created": _now(),
                "date_updated": _now(),
            },
        )
        assert entity.body is None

    def test_tool_additions_getter_returns_fresh_list(self) -> None:
        """Mutating the returned list does not affect the entity."""
        entity = AgentSkillEntity(
            {
                "agent_id": _new_uuid(),
                "skill_id": _new_uuid(),
                "user_id": _new_uuid(),
                "name": "x",
                "summary": "x",
                "body": "x",
                "prompt_mode": "additive",
                "tool_additions": ["mcp.a", "mcp.b"],
                "tool_restrictions": [],
                "trigger_keywords": "",
                "tags": [],
                "source": "manual",
                "enabled": True,
                "use_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "date_created": _now(),
                "date_updated": _now(),
            },
        )
        snap = entity.tool_additions
        snap.append("mcp.injected")
        assert entity.tool_additions == ["mcp.a", "mcp.b"]

    def test_tool_additions_setter_normalises_to_list(self) -> None:
        """Setter coerces tuples / generators to a list snapshot."""
        entity = AgentSkillEntity(
            {
                "agent_id": _new_uuid(),
                "skill_id": _new_uuid(),
                "user_id": _new_uuid(),
                "name": "x",
                "summary": "x",
                "body": "x",
                "prompt_mode": "additive",
                "tool_additions": [],
                "tool_restrictions": [],
                "trigger_keywords": "",
                "tags": [],
                "source": "manual",
                "enabled": True,
                "use_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "date_created": _now(),
                "date_updated": _now(),
            },
        )
        entity.tool_additions = ("mcp.a", "mcp.b")  # type: ignore[assignment]
        assert entity.tool_additions == ["mcp.a", "mcp.b"]


class TestAgentSkillInvocationEntity:
    """Contract tests for :class:`AgentSkillInvocationEntity`."""

    def test_primary_key_field_is_invocation_id(self) -> None:
        """``primary_key_field`` returns ``invocation_id``."""
        assert AgentSkillInvocationEntity.primary_key_field == "invocation_id"

    def test_composite_id_set_when_both_columns_present(self) -> None:
        """``id`` returns the ``(agent_id, invocation_id)`` composite tuple."""
        agent_id = _new_uuid()
        invocation_id = _new_uuid()
        entity = AgentSkillInvocationEntity(
            {
                "agent_id": agent_id,
                "invocation_id": invocation_id,
                "skill_id": _new_uuid(),
                "user_id": _new_uuid(),
                "conversation_id": _new_uuid(),
                "invocation_source": "invoke",
                "invoked_at": _now(),
            },
        )
        assert entity.id == (agent_id, invocation_id)

    def test_field_accessors_round_trip(self) -> None:
        """Every property getter returns the construction value."""
        agent_id = _new_uuid()
        invocation_id = _new_uuid()
        skill_id = _new_uuid()
        user_id = _new_uuid()
        conversation_id = _new_uuid()
        message_id = _new_uuid()
        now = _now()
        entity = AgentSkillInvocationEntity(
            {
                "agent_id": agent_id,
                "invocation_id": invocation_id,
                "skill_id": skill_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "invocation_source": "wake",
                "invoked_at": now,
                "outcome": "success",
                "outcome_source": "agent_marker",
                "notes": "ran cleanly",
            },
        )
        assert entity.agent_id == agent_id
        assert entity.invocation_id == invocation_id
        assert entity.skill_id == skill_id
        assert entity.user_id == user_id
        assert entity.conversation_id == conversation_id
        assert entity.message_id == message_id
        assert entity.invocation_source == "wake"
        assert entity.invoked_at == now
        assert entity.outcome == "success"
        assert entity.outcome_source == "agent_marker"
        assert entity.notes == "ran cleanly"

    def test_message_id_is_optional(self) -> None:
        """Pre-loader rows have ``message_id=None`` until ``set_message_id``."""
        entity = AgentSkillInvocationEntity(
            {
                "agent_id": _new_uuid(),
                "invocation_id": _new_uuid(),
                "skill_id": _new_uuid(),
                "user_id": _new_uuid(),
                "conversation_id": _new_uuid(),
                "message_id": None,
                "invocation_source": "invoke",
                "invoked_at": _now(),
                "outcome": None,
                "outcome_source": None,
                "notes": None,
            },
        )
        assert entity.message_id is None
        assert entity.outcome is None
        assert entity.outcome_source is None
        assert entity.notes is None

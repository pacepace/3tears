"""Unit tests for the REST request/response DTOs in ``skills.api_models``.

Pure-pydantic surface (no FastAPI, no Collection, no LLM). These assert:

- every response model constructs from a representative payload and
  carries the documented fields with the right types
- the request models REJECT the server-derived identity fields
  (``user_id`` / ``agent_id``) via ``extra='forbid'`` -- a real contract
  check, since silently ignoring those would let a caller smuggle them
- the request models still subclass / mirror the tool input schemas so
  the editable field set stays single-sourced
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.agent.skills import (
    CreateSkillRequest,
    SkillInvocationListResponse,
    SkillInvocationResponse,
    SkillListResponse,
    SkillResponse,
    SkillSummary,
    UpdateSkillRequest,
)
from threetears.agent.skills.tools import SkillCreateInput, SkillUpdateInput


# --- Response models ---


class TestSkillResponse:
    """``SkillResponse`` is the full-detail serialization shape."""

    def test_constructs_with_all_fields(self) -> None:
        resp = SkillResponse(
            skill_id="0190-skill",
            agent_id="0190-agent",
            user_id="0190-user",
            kind="prose",
            name="deploy",
            summary="ship it",
            body="## steps",
            prompt_mode="additive",
            tool_additions=["threetears.web_search"],
            tool_restrictions=["threetears.calculator"],
            trigger_keywords="deploy ship",
            tags=["ops"],
            source="manual",
            enabled=True,
            use_count=3,
            last_used_at="2026-05-23T00:00:00+00:00",
            success_count=2,
            failure_count=1,
            last_failure_at=None,
            date_created="2026-05-01T00:00:00+00:00",
            date_updated="2026-05-23T00:00:00+00:00",
        )
        assert resp.skill_id == "0190-skill"
        assert resp.agent_id == "0190-agent"
        assert resp.user_id == "0190-user"
        assert resp.kind == "prose"
        assert resp.body == "## steps"
        assert resp.tool_additions == ["threetears.web_search"]
        assert resp.tool_restrictions == ["threetears.calculator"]
        assert resp.last_failure_at is None
        assert resp.use_count == 3

    def test_nullable_fields_accept_none(self) -> None:
        resp = SkillResponse(
            skill_id="s",
            agent_id="a",
            user_id="u",
            kind="prose",
            name="n",
            summary="sum",
            body=None,
            prompt_mode="replace",
            tool_additions=[],
            tool_restrictions=[],
            trigger_keywords="",
            tags=[],
            source="manual",
            enabled=False,
            use_count=0,
            last_used_at=None,
            success_count=0,
            failure_count=0,
            last_failure_at=None,
            date_created="2026-05-01T00:00:00+00:00",
            date_updated="2026-05-01T00:00:00+00:00",
        )
        assert resp.body is None
        assert resp.last_used_at is None
        assert resp.prompt_mode == "replace"

    def test_rejects_bad_kind(self) -> None:
        with pytest.raises(ValidationError):
            SkillResponse(
                skill_id="s",
                agent_id="a",
                user_id="u",
                kind="bogus",  # type: ignore[arg-type]
                name="n",
                summary="sum",
                body=None,
                prompt_mode="additive",
                tool_additions=[],
                tool_restrictions=[],
                trigger_keywords="",
                tags=[],
                source="manual",
                enabled=True,
                use_count=0,
                last_used_at=None,
                success_count=0,
                failure_count=0,
                last_failure_at=None,
                date_created="2026-05-01T00:00:00+00:00",
                date_updated="2026-05-01T00:00:00+00:00",
            )


class TestSkillSummary:
    """``SkillSummary`` is the list variant (no body)."""

    def test_constructs(self) -> None:
        summary = SkillSummary(
            skill_id="s",
            kind="prose",
            name="deploy",
            summary="ship it",
            prompt_mode="additive",
            tags=["ops"],
            enabled=True,
            use_count=5,
            last_used_at=None,
            success_count=3,
            failure_count=2,
            date_created="2026-05-01T00:00:00+00:00",
        )
        assert summary.name == "deploy"
        assert summary.tags == ["ops"]

    def test_has_no_body_field(self) -> None:
        assert "body" not in SkillSummary.model_fields


class TestSkillListResponse:
    """``SkillListResponse`` wraps a page of summaries + a total."""

    def test_constructs(self) -> None:
        summary = SkillSummary(
            skill_id="s",
            kind="prose",
            name="deploy",
            summary="ship it",
            prompt_mode="additive",
            tags=[],
            enabled=True,
            use_count=0,
            last_used_at=None,
            success_count=0,
            failure_count=0,
            date_created="2026-05-01T00:00:00+00:00",
        )
        env = SkillListResponse(skills=[summary], total_count=1)
        assert env.total_count == 1
        assert env.skills[0].name == "deploy"


class TestSkillInvocationResponse:
    """``SkillInvocationResponse`` serializes one invocation row."""

    def test_constructs_with_nullable_outcome(self) -> None:
        inv = SkillInvocationResponse(
            invocation_id="i",
            skill_id="s",
            conversation_id="c",
            message_id=None,
            invocation_source="invoke",
            invoked_at="2026-05-23T00:00:00+00:00",
            outcome=None,
            outcome_source=None,
            notes=None,
        )
        assert inv.invocation_source == "invoke"
        assert inv.outcome is None
        assert inv.message_id is None

    def test_constructs_with_outcome(self) -> None:
        inv = SkillInvocationResponse(
            invocation_id="i",
            skill_id="s",
            conversation_id="c",
            message_id="m",
            invocation_source="wake",
            invoked_at="2026-05-23T00:00:00+00:00",
            outcome="success",
            outcome_source="agent_marker",
            notes="went fine",
        )
        assert inv.outcome == "success"
        assert inv.outcome_source == "agent_marker"

    def test_rejects_bad_invocation_source(self) -> None:
        with pytest.raises(ValidationError):
            SkillInvocationResponse(
                invocation_id="i",
                skill_id="s",
                conversation_id="c",
                message_id=None,
                invocation_source="cron",  # type: ignore[arg-type]
                invoked_at="2026-05-23T00:00:00+00:00",
                outcome=None,
                outcome_source=None,
                notes=None,
            )


class TestSkillInvocationListResponse:
    """``SkillInvocationListResponse`` wraps invocations + a total."""

    def test_constructs(self) -> None:
        inv = SkillInvocationResponse(
            invocation_id="i",
            skill_id="s",
            conversation_id="c",
            message_id=None,
            invocation_source="invoke",
            invoked_at="2026-05-23T00:00:00+00:00",
            outcome=None,
            outcome_source=None,
            notes=None,
        )
        env = SkillInvocationListResponse(invocations=[inv], total_count=1)
        assert env.total_count == 1
        assert env.invocations[0].invocation_id == "i"


# --- Request models: identity-field rejection contract ---


class TestCreateSkillRequest:
    """``CreateSkillRequest`` mirrors ``SkillCreateInput`` minus identity."""

    def test_constructs_with_editable_fields(self) -> None:
        req = CreateSkillRequest(name="deploy", summary="ship it")
        assert req.name == "deploy"
        assert req.summary == "ship it"
        # Inherited defaults from SkillCreateInput stay single-sourced.
        assert req.prompt_mode == "additive"
        assert req.tool_additions == []
        assert req.enabled is True

    def test_subclasses_tool_input(self) -> None:
        assert issubclass(CreateSkillRequest, SkillCreateInput)

    def test_rejects_user_id(self) -> None:
        with pytest.raises(ValidationError):
            CreateSkillRequest(name="deploy", summary="ship it", user_id="u")  # type: ignore[call-arg]

    def test_rejects_agent_id(self) -> None:
        with pytest.raises(ValidationError):
            CreateSkillRequest(name="deploy", summary="ship it", agent_id="a")  # type: ignore[call-arg]

    def test_does_not_expose_identity_fields(self) -> None:
        assert "user_id" not in CreateSkillRequest.model_fields
        assert "agent_id" not in CreateSkillRequest.model_fields


class TestUpdateSkillRequest:
    """``UpdateSkillRequest`` mirrors ``SkillUpdateInput`` minus identity."""

    def test_all_fields_optional(self) -> None:
        req = UpdateSkillRequest()
        assert req.name is None
        assert req.summary is None
        assert req.prompt_mode is None
        assert req.enabled is None

    def test_field_parity_with_tool_input(self) -> None:
        """Editable fields stay in lock-step with ``SkillUpdateInput``.

        ``UpdateSkillRequest`` is standalone (the tool's required
        ``skill_id`` cannot be widened to optional in a subclass), so a
        parity check guards against the two field sets drifting apart.
        ``skill_id`` is the only tool field intentionally absent (it is a
        path parameter on the REST route).
        """
        tool_editable = set(SkillUpdateInput.model_fields) - {"skill_id"}
        assert set(UpdateSkillRequest.model_fields) == tool_editable

    def test_applies_partial_fields(self) -> None:
        req = UpdateSkillRequest(summary="new summary", enabled=False)
        assert req.summary == "new summary"
        assert req.enabled is False

    def test_rejects_user_id(self) -> None:
        with pytest.raises(ValidationError):
            UpdateSkillRequest(summary="x", user_id="u")  # type: ignore[call-arg]

    def test_rejects_agent_id(self) -> None:
        with pytest.raises(ValidationError):
            UpdateSkillRequest(summary="x", agent_id="a")  # type: ignore[call-arg]

    def test_rejects_skill_id_in_body(self) -> None:
        # skill_id is a path parameter on the REST route, never a body field.
        with pytest.raises(ValidationError):
            UpdateSkillRequest(summary="x", skill_id="s")  # type: ignore[call-arg]

    def test_does_not_expose_identity_fields(self) -> None:
        assert "user_id" not in UpdateSkillRequest.model_fields
        assert "agent_id" not in UpdateSkillRequest.model_fields
        assert "skill_id" not in UpdateSkillRequest.model_fields

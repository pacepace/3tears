"""Unit tests for ``threetears.agent.skills.rendering``.

Exercises the pure per-turn composition surface end-to-end:

- ``compose_turn_context(None, ...)`` returns the base prompt + base
  tools unchanged.
- ``'additive'`` mode with a prose body appends a labeled block; tags
  render when present.
- ``'additive'`` mode with a ``None`` body leaves the base prompt
  alone (tool composition still applies).
- ``'replace'`` mode substitutes the prompt entirely; ``None`` body
  falls back to ``summary``.
- ``tool_additions`` add to the surface when ``acl_permits`` grants.
- ``tool_additions`` are silently dropped when ``acl_permits`` denies.
- ``tool_restrictions`` remove from the surface.
- ``tool_restrictions`` never consult ``acl_permits`` -- subtractive
  operations don't need authorization.
- ``tool_restrictions`` referencing a tool not in ``base_tool_names``
  is a no-op (``discard`` semantics).
- Overlap between ``tool_additions`` and ``tool_restrictions``:
  restriction wins because subtraction runs last.
- Output ``available_tool_names`` is sorted (deterministic for
  prompt-cache stability + test assertions).

The renderer is pure -- no fixtures requiring docker, no DB, no
async marker.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from uuid_utils import uuid7

from threetears.agent.skills.entities import AgentSkillEntity
from threetears.agent.skills.rendering import (
    ComposedTurnContext,
    compose_turn_context,
    render_skill_body_block,
)


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


def _make_skill(
    *,
    name: str = "deploy-helper",
    summary: str = "Deploy a service",
    body: str | None = "Run terraform apply",
    prompt_mode: str = "additive",
    tool_additions: list[str] | None = None,
    tool_restrictions: list[str] | None = None,
    tags: list[str] | None = None,
) -> AgentSkillEntity:
    """Build an ``AgentSkillEntity`` with sane defaults for renderer tests."""
    now = datetime.now(UTC)
    data: dict[str, Any] = {
        "agent_id": _new_uuid(),
        "skill_id": _new_uuid(),
        "user_id": _new_uuid(),
        "name": name,
        "summary": summary,
        "body": body,
        "prompt_mode": prompt_mode,
        "tool_additions": tool_additions if tool_additions is not None else [],
        "tool_restrictions": tool_restrictions if tool_restrictions is not None else [],
        "trigger_keywords": "",
        "tags": tags if tags is not None else [],
        "source": "manual",
        "enabled": True,
        "use_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "date_created": now,
        "date_updated": now,
    }
    return AgentSkillEntity(data)


def _permit_all(_name: str) -> bool:
    """ACL stub that grants every tool."""
    return True


def _deny_all(_name: str) -> bool:
    """ACL stub that denies every tool."""
    return False


def _permit_except(forbidden: set[str]) -> Callable[[str], bool]:
    """Return an ``acl_permits`` closure denying only ``forbidden``."""

    def _acl(name: str) -> bool:
        return name not in forbidden

    return _acl


def _exploding_acl(_name: str) -> bool:
    """ACL that fails the test if it is consulted at all.

    Used to assert ``tool_restrictions`` does NOT consult ACL.
    """
    raise AssertionError("acl_permits must not be called for tool_restrictions")


# ---------------------------------------------------------------------------
# render_skill_body_block
# ---------------------------------------------------------------------------


class TestRenderSkillBodyBlock:
    """Contract tests for :func:`render_skill_body_block`."""

    def test_prose_body_no_tags(self) -> None:
        """Body-only skill renders the labeled header + body."""
        skill = _make_skill(name="deploy", body="Run terraform apply", tags=[])
        rendered = render_skill_body_block(skill)
        assert rendered == "## Skill: deploy\n\nRun terraform apply"

    def test_prose_body_with_tags(self) -> None:
        """Tags render between the header and the body."""
        skill = _make_skill(
            name="deploy",
            body="Run terraform apply",
            tags=["ops", "prod"],
        )
        rendered = render_skill_body_block(skill)
        assert rendered == ("## Skill: deploy\n<tags: [ops, prod]>\n\nRun terraform apply")

    def test_empty_body_returns_empty_string(self) -> None:
        """Skill with ``body=None`` renders ``""`` (no header announcement)."""
        skill = _make_skill(body=None, tool_additions=["t.a"])
        assert render_skill_body_block(skill) == ""

    def test_empty_body_with_tags_still_empty(self) -> None:
        """Tags alone do not summon a block -- body is the trigger."""
        skill = _make_skill(body=None, tags=["ops"], tool_additions=["t.a"])
        assert render_skill_body_block(skill) == ""


# ---------------------------------------------------------------------------
# compose_turn_context -- no active skill
# ---------------------------------------------------------------------------


class TestComposeTurnContextNoSkill:
    """``active_skill=None`` returns base prompt + base tools unchanged."""

    def test_no_skill_returns_base(self) -> None:
        """Pass-through composition when no skill is active."""
        base_prompt = "You are a helpful assistant."
        base_tools = ["search", "calc"]

        result = compose_turn_context(
            None,
            base_prompt,
            base_tools,
            acl_permits=_permit_all,
        )

        assert isinstance(result, ComposedTurnContext)
        assert result.system_prompt == base_prompt
        assert result.available_tool_names == ["calc", "search"]  # sorted
        assert result.active_skill_id is None

    def test_no_skill_empty_tools(self) -> None:
        """Empty base tool surface stays empty."""
        result = compose_turn_context(
            None,
            "prompt",
            [],
            acl_permits=_permit_all,
        )
        assert result.available_tool_names == []
        assert result.active_skill_id is None

    def test_no_skill_does_not_call_acl(self) -> None:
        """No skill -> no ACL consultation."""
        result = compose_turn_context(
            None,
            "prompt",
            ["a", "b"],
            acl_permits=_exploding_acl,
        )
        assert result.system_prompt == "prompt"
        assert result.available_tool_names == ["a", "b"]


# ---------------------------------------------------------------------------
# compose_turn_context -- additive prompt mode
# ---------------------------------------------------------------------------


class TestComposeTurnContextAdditive:
    """``prompt_mode='additive'`` appends the body block to base."""

    def test_additive_with_body_appends(self) -> None:
        """Body block is appended after a blank-line separator."""
        skill = _make_skill(
            name="deploy",
            body="Run terraform apply",
        )

        result = compose_turn_context(
            skill,
            "You are a helpful assistant.",
            [],
            acl_permits=_permit_all,
        )

        assert result.system_prompt == ("You are a helpful assistant.\n\n## Skill: deploy\n\nRun terraform apply")
        assert result.active_skill_id == skill.skill_id

    def test_additive_with_body_and_tags(self) -> None:
        """Tags propagate into the appended block."""
        skill = _make_skill(
            name="deploy",
            body="Run terraform apply",
            tags=["ops"],
        )

        result = compose_turn_context(
            skill,
            "base",
            [],
            acl_permits=_permit_all,
        )

        assert result.system_prompt == ("base\n\n## Skill: deploy\n<tags: [ops]>\n\nRun terraform apply")

    def test_additive_with_empty_body_leaves_base_unchanged(self) -> None:
        """A pure tool-composition skill in additive mode does not mutate the prompt."""
        skill = _make_skill(
            body=None,
            tool_additions=["t.a"],
        )

        result = compose_turn_context(
            skill,
            "base prompt unchanged",
            [],
            acl_permits=_permit_all,
        )

        assert result.system_prompt == "base prompt unchanged"
        assert result.available_tool_names == ["t.a"]
        assert result.active_skill_id == skill.skill_id


# ---------------------------------------------------------------------------
# compose_turn_context -- replace prompt mode
# ---------------------------------------------------------------------------


class TestComposeTurnContextReplace:
    """``prompt_mode='replace'`` substitutes the system prompt entirely."""

    def test_replace_with_body_substitutes(self) -> None:
        """Body block becomes the entire system prompt."""
        skill = _make_skill(
            name="locked-down",
            body="Strict mode body",
            prompt_mode="replace",
        )

        result = compose_turn_context(
            skill,
            "you should not see this base prompt",
            [],
            acl_permits=_permit_all,
        )

        assert result.system_prompt == "## Skill: locked-down\n\nStrict mode body"
        assert "you should not see" not in result.system_prompt

    def test_replace_with_empty_body_falls_back_to_summary(self) -> None:
        """Pure tool-composition skill in replace mode falls back to summary."""
        skill = _make_skill(
            summary="A short identity",
            body=None,
            prompt_mode="replace",
            tool_additions=["t.a"],
        )

        result = compose_turn_context(
            skill,
            "ignored base",
            [],
            acl_permits=_permit_all,
        )

        assert result.system_prompt == "A short identity"


# ---------------------------------------------------------------------------
# compose_turn_context -- tool_additions
# ---------------------------------------------------------------------------


class TestComposeTurnContextToolAdditions:
    """``tool_additions`` add to the surface, gated by ACL only."""

    def test_addition_with_acl_grant_added(self) -> None:
        """ACL-permitted addition lands in the surface."""
        skill = _make_skill(
            body=None,
            tool_additions=["t.new"],
        )

        result = compose_turn_context(
            skill,
            "base",
            ["t.existing"],
            acl_permits=_permit_all,
        )

        assert result.available_tool_names == ["t.existing", "t.new"]

    def test_addition_with_acl_deny_silently_dropped(self) -> None:
        """ACL-denied additions are silently omitted (no error)."""
        skill = _make_skill(
            body=None,
            tool_additions=["t.forbidden", "t.allowed"],
        )

        result = compose_turn_context(
            skill,
            "base",
            ["t.existing"],
            acl_permits=_permit_except({"t.forbidden"}),
        )

        assert result.available_tool_names == ["t.allowed", "t.existing"]

    def test_all_additions_denied_yields_base_surface(self) -> None:
        """Deny-all ACL leaves the base surface intact."""
        skill = _make_skill(
            body=None,
            tool_additions=["t.a", "t.b", "t.c"],
        )

        result = compose_turn_context(
            skill,
            "base",
            ["t.existing"],
            acl_permits=_deny_all,
        )

        assert result.available_tool_names == ["t.existing"]

    def test_addition_can_grant_tool_outside_base_surface(self) -> None:
        """Skill surfaces a tool that wasn't in the base set.

        This is the ``tool_eligible=False, skill_eligible=True``
        ("code-skill without sandbox") pattern: the renderer trusts the
        skill as the visibility gate and does NOT re-check
        ``tool_eligible``. The base set already filtered such tools out;
        the addition path brings them in.
        """
        skill = _make_skill(
            body=None,
            tool_additions=["t.skill_only"],
        )

        result = compose_turn_context(
            skill,
            "base",
            ["t.normal"],
            acl_permits=_permit_all,
        )

        assert "t.skill_only" in result.available_tool_names


# ---------------------------------------------------------------------------
# compose_turn_context -- tool_restrictions
# ---------------------------------------------------------------------------


class TestComposeTurnContextToolRestrictions:
    """``tool_restrictions`` remove from the surface without ACL."""

    def test_restriction_removes_from_surface(self) -> None:
        """Listed tool is dropped from the available surface."""
        skill = _make_skill(
            body=None,
            tool_restrictions=["t.banned"],
        )

        result = compose_turn_context(
            skill,
            "base",
            ["t.banned", "t.keep"],
            acl_permits=_permit_all,
        )

        assert result.available_tool_names == ["t.keep"]

    def test_restriction_not_in_base_is_noop(self) -> None:
        """Restricting a tool that wasn't in the base set is not an error."""
        skill = _make_skill(
            body=None,
            tool_restrictions=["t.never_was"],
        )

        result = compose_turn_context(
            skill,
            "base",
            ["t.exists"],
            acl_permits=_permit_all,
        )

        assert result.available_tool_names == ["t.exists"]

    def test_restriction_does_not_consult_acl(self) -> None:
        """``acl_permits`` is never called for ``tool_restrictions``.

        Subtractive operations don't need authorization -- a skill can
        always remove a tool from the surface. The exploding ACL would
        raise if consulted.
        """
        skill = _make_skill(
            body=None,
            tool_restrictions=["t.banned"],
        )

        # No additions -> ACL is never called.
        result = compose_turn_context(
            skill,
            "base",
            ["t.banned", "t.keep"],
            acl_permits=_exploding_acl,
        )

        assert result.available_tool_names == ["t.keep"]


# ---------------------------------------------------------------------------
# compose_turn_context -- overlap + sorting
# ---------------------------------------------------------------------------


class TestComposeTurnContextEdgeCases:
    """Overlap rules + sorting + entity-id echo."""

    def test_addition_and_restriction_overlap_restriction_wins(self) -> None:
        """Same tool in both lists: restriction runs last and wins."""
        skill = _make_skill(
            body=None,
            tool_additions=["t.shared"],
            tool_restrictions=["t.shared"],
        )

        result = compose_turn_context(
            skill,
            "base",
            [],
            acl_permits=_permit_all,
        )

        assert result.available_tool_names == []

    def test_available_tool_names_is_sorted(self) -> None:
        """Output is sorted regardless of input order."""
        skill = _make_skill(
            body=None,
            tool_additions=["t.z", "t.a"],
        )

        result = compose_turn_context(
            skill,
            "base",
            ["t.m"],
            acl_permits=_permit_all,
        )

        assert result.available_tool_names == ["t.a", "t.m", "t.z"]

    def test_active_skill_id_echoed_back(self) -> None:
        """The composed context surfaces the active skill id for logging."""
        skill = _make_skill(body="hi")

        result = compose_turn_context(
            skill,
            "base",
            [],
            acl_permits=_permit_all,
        )

        assert result.active_skill_id == skill.skill_id

    def test_idempotent_under_repeat_call(self) -> None:
        """Pure function -- repeated invocation returns equal outputs."""
        skill = _make_skill(
            body="x",
            tool_additions=["t.a"],
            tool_restrictions=["t.r"],
        )

        first = compose_turn_context(
            skill,
            "base",
            ["t.r", "t.keep"],
            acl_permits=_permit_all,
        )
        second = compose_turn_context(
            skill,
            "base",
            ["t.r", "t.keep"],
            acl_permits=_permit_all,
        )

        assert first == second

    def test_result_is_frozen(self) -> None:
        """``ComposedTurnContext`` is immutable -- assignment raises."""
        result = compose_turn_context(
            None,
            "base",
            [],
            acl_permits=_permit_all,
        )
        with pytest.raises(AttributeError):
            result.system_prompt = "hijacked"  # type: ignore[misc]

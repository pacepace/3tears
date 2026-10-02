"""Unit tests for the payload validation and id parsing the skills tools apply.

Driven through the ``skill_create`` / ``skill_get`` tools over in-memory
fakes (no Postgres, no LLM), so each rule is pinned where an agent meets
it -- as the tool's answer:

- payload caps (name, summary, body, trigger_keywords, tags,
  tool_additions, tool_restrictions, arguments)
- ``[skill:<id>]`` parsing
- at-least-one-payload (CHECK-constraint mirror)
- the ``[TOOL ERROR] <tool>: <description>`` format

The factory functions' happy-path + ACL + cross-user coverage lives in
``test_tools_factories.py`` and the integration suite.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from packages.agent.skills.tests.unit.skill_fakes import FakeRegistry, FakeSkillsCollection
from threetears.agent.skills.collections import skill_shape_error
from threetears.agent.skills.tools import (
    ARGUMENTS_MAX_BYTES,
    BODY_MAX_BYTES,
    NAME_MAX_LEN,
    SUMMARY_MAX_LEN,
    TAGS_MAX_ENTRIES,
    TOOL_LIST_MAX_ENTRIES,
    TRIGGER_KEYWORDS_MAX_LEN,
    SkillCreateInput,
    SkillIntrospectInput,
    SkillInvokeInput,
    SkillListInput,
    load_skill_create_tool,
    load_skill_get_tool,
)

#: the tool names a create may name in its tool lists or as its tool; the fake
#: registry grants exactly these, so an ACL refusal never masks a validation.
_PERMITTED = {"mcp.shell", "mcp.dangerous", "a.b", "c.d", "loki.query"} | {f"t{i}" for i in range(64)}


async def _create(**fields: Any) -> str:
    """call ``skill_create`` with ``fields`` (a valid name / summary / body unless given)."""
    payload: dict[str, Any] = {"name": "deploy", "summary": "ship it", "body": "a procedure"} | fields
    [tool] = load_skill_create_tool(
        agent_id=uuid4(),
        user_id=uuid4(),
        skills_collection=FakeSkillsCollection(),  # type: ignore[arg-type]
        registry=FakeRegistry(permitted_tools=_PERMITTED),
        offer_tool_skills=True,
    )
    out = await tool.ainvoke({k: v for k, v in payload.items() if v is not _OMIT})
    assert isinstance(out, str)
    return out


#: marks a field the create call leaves out entirely.
_OMIT: Any = object()


def _accepted(out: str) -> bool:
    return out.startswith("[skill:")


def _refusal(out: str) -> str:
    """the reason a refused create gives, after the ``[TOOL ERROR] skill_create:`` prefix."""
    prefix = "[TOOL ERROR] skill_create: "
    assert out.startswith(prefix), out
    return out[len(prefix) :]


# --- Schema-side parsing (Pydantic) ---


class TestSkillCreateInputSchema:
    """``SkillCreateInput`` rejects nothing schema-side beyond Pydantic types."""

    def test_defaults(self) -> None:
        """``name`` + ``summary`` are required; everything else defaults."""
        inp = SkillCreateInput(name="deploy", summary="ship it")
        assert inp.body is None
        assert inp.prompt_mode == "additive"
        assert inp.tool_additions == []
        assert inp.tool_restrictions == []
        assert inp.trigger_keywords == ""
        assert inp.tags == []
        assert inp.enabled is True


class TestSkillListInputSchema:
    """``SkillListInput`` defaults match the documented public surface."""

    def test_defaults(self) -> None:
        inp = SkillListInput()
        assert inp.query is None
        assert inp.kind_filter == "all"
        assert inp.tag_filter is None
        assert inp.enabled_only is True
        assert inp.limit == 20

    def test_limit_clamping(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SkillListInput(limit=0)
        with pytest.raises(ValidationError):
            SkillListInput(limit=201)


class TestSkillInvokeInputSchema:
    """``SkillInvokeInput`` requires a skill_id; rationale optional."""

    def test_minimum(self) -> None:
        inp = SkillInvokeInput(skill_id="[skill:abc]")
        assert inp.skill_id == "[skill:abc]"
        assert inp.rationale is None


class TestSkillIntrospectInputSchema:
    """``SkillIntrospectInput`` rejects empty / whitespace-only names."""

    def test_non_empty(self) -> None:
        inp = SkillIntrospectInput(name_or_id="some-name")
        assert inp.name_or_id == "some-name"

    def test_empty_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SkillIntrospectInput(name_or_id="")
        with pytest.raises(ValidationError):
            SkillIntrospectInput(name_or_id="   ")


# --- Validation, as the tool answers it ---


class TestValidateName:
    """``skill_create`` enforces the SK-10 name contract (length + charset)."""

    async def test_valid_name(self) -> None:
        assert _accepted(await _create(name="deploy_helper"))
        assert _accepted(await _create(name="ABC 123-xyz"))

    async def test_too_short(self) -> None:
        assert "1 character" in _refusal(await _create(name=""))

    async def test_too_long(self) -> None:
        assert f"{NAME_MAX_LEN} characters" in _refusal(await _create(name="x" * (NAME_MAX_LEN + 1)))

    async def test_invalid_charset(self) -> None:
        for bad in ["foo!bar", "foo/bar", "foo.bar", "you@host"]:
            assert "match" in _refusal(await _create(name=bad)), f"expected rejection for {bad!r}"


class TestValidateSummary:
    async def test_valid(self) -> None:
        assert _accepted(await _create(summary="one-line catalog entry"))

    async def test_empty_rejected(self) -> None:
        assert "summary" in _refusal(await _create(summary=""))

    async def test_too_long(self) -> None:
        assert "summary" in _refusal(await _create(summary="x" * (SUMMARY_MAX_LEN + 1)))


class TestValidateBody:
    async def test_none(self) -> None:
        # no body at all is fine when another payload carries the skill.
        assert _accepted(await _create(body=_OMIT, tool_additions=["mcp.shell"]))

    async def test_short_body(self) -> None:
        assert _accepted(await _create(body="a procedure"))

    async def test_at_cap(self) -> None:
        # Exactly at cap is OK; one byte over is rejected.
        assert _accepted(await _create(body="a" * BODY_MAX_BYTES))
        assert "32 KB cap" in _refusal(await _create(body="a" * (BODY_MAX_BYTES + 1)))

    async def test_multibyte_counted_truthfully(self) -> None:
        # Each '€' is 3 bytes in UTF-8; cap+1 bytes worth must reject.
        assert "32 KB cap" in _refusal(await _create(body="€" * (BODY_MAX_BYTES // 3 + 1)))


class TestValidateTriggerKeywords:
    async def test_valid(self) -> None:
        assert _accepted(await _create(trigger_keywords="alpha beta gamma"))

    async def test_empty_ok(self) -> None:
        assert _accepted(await _create(trigger_keywords=""))

    async def test_too_long(self) -> None:
        assert "trigger_keywords" in _refusal(await _create(trigger_keywords="a" * (TRIGGER_KEYWORDS_MAX_LEN + 1)))


class TestValidateTags:
    async def test_valid(self) -> None:
        assert _accepted(await _create(tags=[]))
        assert _accepted(await _create(tags=["ops", "deploy"]))

    async def test_too_many(self) -> None:
        out = await _create(tags=[f"t{i}" for i in range(TAGS_MAX_ENTRIES + 1)])
        assert f"{TAGS_MAX_ENTRIES} entries" in _refusal(out)

    async def test_non_string_entries(self) -> None:
        # the tool's input schema refuses a non-string tag before the tool runs.
        with pytest.raises(ValidationError):
            await _create(tags=["ok", 5])


class TestValidateToolList:
    async def test_valid(self) -> None:
        assert _accepted(await _create(tool_additions=[]))
        assert _accepted(await _create(tool_additions=["a.b", "c.d"]))

    async def test_too_many(self) -> None:
        out = await _create(tool_additions=[f"t{i}" for i in range(TOOL_LIST_MAX_ENTRIES + 1)])
        assert "tool_additions" in _refusal(out)

    async def test_empty_string(self) -> None:
        assert "tool_additions entries must all be non-empty strings" in _refusal(await _create(tool_additions=[""]))

    async def test_whitespace_only(self) -> None:
        out = await _create(tool_restrictions=["   "])
        assert "tool_restrictions entries must all be non-empty strings" in _refusal(out)


_NO_PAYLOAD = "at least one of body, tool, tool_additions, or tool_restrictions must be non-empty"


class TestAtLeastOnePayload:
    """Mirrors the L3 CHECK constraint."""

    async def test_body_only(self) -> None:
        assert _accepted(await _create(body="procedure"))

    async def test_additions_only(self) -> None:
        assert _accepted(await _create(body=_OMIT, tool_additions=["mcp.shell"]))

    async def test_restrictions_only(self) -> None:
        assert _accepted(await _create(body=_OMIT, tool_restrictions=["mcp.dangerous"]))

    async def test_all_empty_rejected(self) -> None:
        assert _refusal(await _create(body=_OMIT)) == _NO_PAYLOAD

    async def test_empty_string_body_rejected(self) -> None:
        # Empty body counts as no body for at-least-one-payload (DB CHECK
        # treats NULL and "" equivalently).
        assert _refusal(await _create(body="")) == _NO_PAYLOAD

    async def test_whitespace_body_rejected(self) -> None:
        assert _refusal(await _create(body="   \n")) == _NO_PAYLOAD


class _RecordingSkills(FakeSkillsCollection):
    """the fake skills collection, recording the pk each ``get`` asked for."""

    def __init__(self) -> None:
        super().__init__()
        self.asked: list[tuple[UUID, UUID]] = []

    async def get(self, entity_id: Any) -> Any:
        self.asked.append(entity_id)
        return await super().get(entity_id)


async def _get(raw: str) -> tuple[str, list[tuple[UUID, UUID]]]:
    """call ``skill_get`` with ``raw``; return its answer and the pks it looked up."""
    skills = _RecordingSkills()
    [tool] = load_skill_get_tool(agent_id=uuid4(), user_id=uuid4(), skills_collection=skills)  # type: ignore[arg-type]
    out = await tool.ainvoke({"skill_id": raw})
    assert isinstance(out, str)
    return out, skills.asked


class TestParseSkillId:
    """Round-trip ``[skill:<uuid>]`` and bare-UUID forms."""

    async def test_bare_uuid(self) -> None:
        u = uuid4()
        _out, asked = await _get(str(u))
        assert [skill for _agent, skill in asked] == [u]

    async def test_tagged_form(self) -> None:
        u = uuid4()
        _out, asked = await _get(f"[skill:{u}]")
        assert [skill for _agent, skill in asked] == [u]

    async def test_tagged_with_whitespace(self) -> None:
        u = uuid4()
        _out, asked = await _get(f"  [skill: {u} ]  ")
        assert [skill for _agent, skill in asked] == [u]

    async def test_invalid_is_refused_without_a_lookup(self) -> None:
        for raw in ("not-a-uuid", "", "[skill:not-uuid]"):
            out, asked = await _get(raw)
            assert out == f"[TOOL ERROR] skill_get: invalid skill_id {raw!r}"
            assert asked == []


class TestToolError:
    async def test_format(self) -> None:
        out, _asked = await _get(str(uuid4()))
        assert out == "[TOOL ERROR] skill_get: skill not found"

    async def test_includes_tool_name(self) -> None:
        out = await _create(name="you@host")
        assert out.startswith("[TOOL ERROR] skill_create:")


# --- Confirm UUID parsing on assigned-name fixture ---


async def test_parse_skill_id_returns_uuid_type() -> None:
    u = uuid4()
    _out, asked = await _get(str(u))
    assert isinstance(asked[0][1], UUID)


class TestToolCallShape:
    """A skill is a body skill or a tool-call skill, never both; the tools add a size cap."""

    async def test_tool_counts_as_payload(self) -> None:
        assert _accepted(await _create(body=_OMIT, tool="loki.query"))

    @pytest.mark.parametrize(
        ("body", "tool", "arguments", "fragment"),
        [
            ("steps", "loki.query", None, "not both"),
            ("", "loki.query", None, "not both"),
            (None, None, {"q": 1}, "arguments need a tool"),
            (None, "loki.query", ["q"], "JSON object"),
            (None, "loki.query", {1: "q"}, "every key a string"),
            (None, "loki.query", {"q": object()}, "plain JSON"),
            (None, "   ", None, "not blank"),
        ],
    )
    def test_refused_shapes(self, body: str | None, tool: str | None, arguments: object, fragment: str) -> None:
        error = skill_shape_error(body=body, tool=tool, arguments=arguments)
        assert error is not None
        assert fragment in error

    @pytest.mark.parametrize(
        ("body", "tool", "arguments"),
        [
            ("steps", None, None),
            (None, "loki.query", None),
            (None, "loki.query", {}),
            (None, "loki.query", {"q": "error", "limit": 5, "nested": {"a": [1, None]}}),
            (None, None, None),
        ],
    )
    def test_accepted_shapes(self, body: str | None, tool: str | None, arguments: object) -> None:
        assert skill_shape_error(body=body, tool=tool, arguments=arguments) is None

    async def test_arguments_cap(self) -> None:
        out = await _create(body=_OMIT, tool="loki.query", arguments={"q": "x" * ARGUMENTS_MAX_BYTES})
        assert _refusal(out) == "arguments exceed 32 KB cap"
        assert _accepted(await _create(body=_OMIT, tool="loki.query", arguments={"q": "x"}))
        assert _accepted(await _create(body=_OMIT, tool="loki.query"))

"""Unit tests for the validator helpers + parsers in ``skills.tools``.

These exercise pure-logic helpers (no Collection, no registry,
no LLM):

- payload caps (name, summary, body, trigger_keywords, tags,
  tool_additions, tool_restrictions)
- ``[skill:<id>]`` parsing
- at-least-one-payload (CHECK-constraint mirror)
- ``_tool_error`` format

The factory functions themselves (skill_create / skill_list / etc.)
get full happy-path + ACL + cross-user coverage in the integration
suite where real Collections + Postgres exercise the end-to-end path.
The unit slice keeps the validator surface bit-tight.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from threetears.agent.skills.tools import (
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
    _at_least_one_payload,
    _parse_skill_id,
    _tool_error,
    _validate_body,
    _validate_name,
    _validate_summary,
    _validate_tags,
    _validate_tool_list,
    _validate_trigger_keywords,
)


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


# --- Validator helpers ---


class TestValidateName:
    """``_validate_name`` enforces SK-10 contract (length + charset)."""

    def test_valid_name(self) -> None:
        assert _validate_name("deploy_helper") is None
        assert _validate_name("ABC 123-xyz") is None

    def test_too_short(self) -> None:
        err = _validate_name("")
        assert err is not None
        assert "1 character" in err

    def test_too_long(self) -> None:
        err = _validate_name("x" * (NAME_MAX_LEN + 1))
        assert err is not None
        assert f"{NAME_MAX_LEN} characters" in err

    def test_invalid_charset(self) -> None:
        for bad in ["foo!bar", "foo/bar", "foo.bar", "you@host"]:
            err = _validate_name(bad)
            assert err is not None, f"expected rejection for {bad!r}"
            assert "match" in err


class TestValidateSummary:
    def test_valid(self) -> None:
        assert _validate_summary("one-line catalog entry") is None

    def test_empty_rejected(self) -> None:
        err = _validate_summary("")
        assert err is not None

    def test_too_long(self) -> None:
        err = _validate_summary("x" * (SUMMARY_MAX_LEN + 1))
        assert err is not None


class TestValidateBody:
    def test_none(self) -> None:
        assert _validate_body(None) is None

    def test_short_body(self) -> None:
        assert _validate_body("a procedure") is None

    def test_at_cap(self) -> None:
        # Exactly at cap is OK; one byte over is rejected.
        assert _validate_body("a" * BODY_MAX_BYTES) is None
        err = _validate_body("a" * (BODY_MAX_BYTES + 1))
        assert err is not None
        assert "32 KB cap" in err

    def test_multibyte_counted_truthfully(self) -> None:
        # Three-byte UTF-8 character; cap+1 bytes worth should reject.
        # Each '€' is 3 bytes in UTF-8.
        too_long = "€" * (BODY_MAX_BYTES // 3 + 1)
        err = _validate_body(too_long)
        assert err is not None


class TestValidateTriggerKeywords:
    def test_valid(self) -> None:
        assert _validate_trigger_keywords("alpha beta gamma") is None

    def test_empty_ok(self) -> None:
        assert _validate_trigger_keywords("") is None

    def test_too_long(self) -> None:
        err = _validate_trigger_keywords("a" * (TRIGGER_KEYWORDS_MAX_LEN + 1))
        assert err is not None


class TestValidateTags:
    def test_valid(self) -> None:
        assert _validate_tags([]) is None
        assert _validate_tags(["ops", "deploy"]) is None

    def test_too_many(self) -> None:
        err = _validate_tags([f"t{i}" for i in range(TAGS_MAX_ENTRIES + 1)])
        assert err is not None
        assert f"{TAGS_MAX_ENTRIES} entries" in err

    def test_non_string_entries(self) -> None:
        err = _validate_tags(["ok", 5])  # type: ignore[list-item]
        assert err is not None


class TestValidateToolList:
    def test_valid(self) -> None:
        assert _validate_tool_list("tool_additions", []) is None
        assert _validate_tool_list("tool_additions", ["a.b", "c.d"]) is None

    def test_too_many(self) -> None:
        err = _validate_tool_list(
            "tool_additions",
            [f"t{i}" for i in range(TOOL_LIST_MAX_ENTRIES + 1)],
        )
        assert err is not None

    def test_empty_string(self) -> None:
        err = _validate_tool_list("tool_additions", [""])
        assert err is not None

    def test_whitespace_only(self) -> None:
        err = _validate_tool_list("tool_restrictions", ["   "])
        assert err is not None


class TestAtLeastOnePayload:
    """Mirrors the L3 CHECK constraint."""

    def test_body_only(self) -> None:
        assert _at_least_one_payload(
            body="procedure",
            tool_additions=[],
            tool_restrictions=[],
        )

    def test_additions_only(self) -> None:
        assert _at_least_one_payload(
            body=None,
            tool_additions=["mcp.shell"],
            tool_restrictions=[],
        )

    def test_restrictions_only(self) -> None:
        assert _at_least_one_payload(
            body=None,
            tool_additions=[],
            tool_restrictions=["mcp.dangerous"],
        )

    def test_all_empty_rejected(self) -> None:
        assert not _at_least_one_payload(
            body=None,
            tool_additions=[],
            tool_restrictions=[],
        )

    def test_empty_string_body_rejected(self) -> None:
        # Empty body counts as no body for at-least-one-payload (DB CHECK
        # treats NULL and "" equivalently).
        assert not _at_least_one_payload(
            body="",
            tool_additions=[],
            tool_restrictions=[],
        )

    def test_whitespace_body_rejected(self) -> None:
        assert not _at_least_one_payload(
            body="   \n",
            tool_additions=[],
            tool_restrictions=[],
        )


class TestParseSkillId:
    """Round-trip ``[skill:<uuid>]`` and bare-UUID forms."""

    def test_bare_uuid(self) -> None:
        u = uuid4()
        parsed = _parse_skill_id(str(u))
        assert parsed == u

    def test_tagged_form(self) -> None:
        u = uuid4()
        parsed = _parse_skill_id(f"[skill:{u}]")
        assert parsed == u

    def test_tagged_with_whitespace(self) -> None:
        u = uuid4()
        parsed = _parse_skill_id(f"  [skill: {u} ]  ")
        assert parsed == u

    def test_invalid_returns_none(self) -> None:
        assert _parse_skill_id("not-a-uuid") is None
        assert _parse_skill_id("") is None
        assert _parse_skill_id("[skill:not-uuid]") is None


class TestToolError:
    def test_format(self) -> None:
        out = _tool_error("skill_create", "name too long")
        assert out == "[TOOL ERROR] skill_create: name too long"

    def test_includes_tool_name(self) -> None:
        out = _tool_error("skill_invoke", "already active")
        assert out.startswith("[TOOL ERROR] skill_invoke:")


# --- Confirm UUID parsing on assigned-name fixture ---


def test_parse_skill_id_returns_uuid_type() -> None:
    u = uuid4()
    parsed = _parse_skill_id(str(u))
    assert isinstance(parsed, UUID)

"""Free text in the skill tools' results stays under the heading it belongs to.

A many-line text (with a blank line and an indented line) is fed through each
tool that shows one; the item's block is asserted exactly: the heading, every
line of the text indented under it, no blank line inside the item.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from uuid_utils import uuid7

from threetears.agent.skills.tools import (
    SkillEligibleTool,
    SkillToolIntrospect,
    load_skill_create_tool,
    load_skill_history_tool,
    load_skill_introspect_tool,
    load_skill_list_tool,
)

from .skill_fakes import FakeInvocationsCollection, FakeRegistry, FakeSkillsCollection


def _id() -> UUID:
    return UUID(str(uuid7()))


AGENT, PERSON, CONV = _id(), _id(), _id()

MANY_LINES = "first line\n\n  indented line\nlast line"
BLANK = "  \n \n"


def _seed(coll: FakeSkillsCollection, **fields: Any) -> UUID:
    skill_id = _id()
    now = datetime(2026, 10, 9, 22, 0, tzinfo=UTC)
    row: dict[str, Any] = {
        "skill_id": skill_id,
        "agent_id": AGENT,
        "user_id": PERSON,
        "name": "tide-check",
        "summary": "read the tide table",
        "body": "Open the tide table.",
        "tool": None,
        "arguments": None,
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
        "last_used_at": None,
        "last_failure_at": None,
        "date_created": now,
        "date_updated": now,
    }
    row.update(fields)
    coll.rows[(AGENT, skill_id)] = row
    return skill_id


def _list_tool(coll: FakeSkillsCollection, registry: FakeRegistry) -> Any:
    [tool] = load_skill_list_tool(
        agent_id=AGENT,
        user_id=PERSON,
        skills_collection=coll,  # type: ignore[arg-type]
        registry=registry,
    )
    return tool


def _introspect_tool(registry: FakeRegistry) -> Any:
    [tool] = load_skill_introspect_tool(
        agent_id=AGENT,
        user_id=PERSON,
        skills_collection=FakeSkillsCollection(),  # type: ignore[arg-type]
        registry=registry,
    )
    return tool


def _history(coll: FakeSkillsCollection, **fields: Any) -> tuple[Any, UUID]:
    inv = FakeInvocationsCollection(coll)
    skill_id = _seed(coll)
    row_id = _id()
    inv.rows[(AGENT, row_id)] = {
        "agent_id": AGENT,
        "invocation_id": row_id,
        "skill_id": skill_id,
        "user_id": PERSON,
        "conversation_id": CONV,
        "message_id": None,
        "invocation_source": "invoke",
        "invoked_at": datetime(2026, 10, 9, 12, 0, tzinfo=UTC),
        "outcome": "success",
        "outcome_source": "agent_tool",
        "notes": None,
        "outcome_notes": None,
        **fields,
    }
    [tool] = load_skill_history_tool(
        agent_id=AGENT,
        user_id=PERSON,
        skills_collection=coll,  # type: ignore[arg-type]
        invocations_collection=inv,  # type: ignore[arg-type]
        tz="America/Los_Angeles",
    )
    return tool, skill_id


class TestSkillListSummaries:
    async def test_a_many_line_summary_is_one_line_in_the_catalog(self) -> None:
        coll = FakeSkillsCollection()
        skill_id = _seed(coll, summary="read the tide\n\n  table\nthen say it")
        registry = FakeRegistry(
            skill_eligible=[SkillEligibleTool(mcp_name="loki.query", summary="Query\n\nLoki\n logs")]
        )
        out = await _list_tool(coll, registry).ainvoke({})
        assert out.splitlines() == [
            "Found 2 skills:",
            f"- [skill:{skill_id}] tide-check (kind=prose) -- read the tide table then say it",
            "- [skill:loki.query] loki.query (kind=tool) -- Query Loki logs",
        ]

    async def test_a_blank_summary_still_shows_the_skills_line(self) -> None:
        coll = FakeSkillsCollection()
        skill_id = _seed(coll, summary=BLANK)
        out = await _list_tool(coll, FakeRegistry()).ainvoke({})
        assert out.splitlines() == ["Found 1 skills:", f"- [skill:{skill_id}] tide-check (kind=prose) -- "]


class TestSkillCreateSummary:
    async def test_a_many_line_summary_is_one_line_in_the_created_entry(self) -> None:
        coll = FakeSkillsCollection()
        [tool] = load_skill_create_tool(
            agent_id=AGENT,
            user_id=PERSON,
            skills_collection=coll,  # type: ignore[arg-type]
            registry=FakeRegistry(),
        )
        out = await tool.ainvoke({"name": "deploy", "summary": "Ship it\n\n  safely\nto prod", "body": "Run helm"})
        [row] = coll.rows.values()
        assert out == f"[skill:{row['skill_id']}] deploy -- Ship it safely to prod"


class TestToolSkillIntrospectArgs:
    async def test_a_many_line_arg_description_stays_under_its_arg(self) -> None:
        registry = FakeRegistry(
            introspect_payloads={
                "loki.query": SkillToolIntrospect(
                    mcp_name="loki.query",
                    summary="Query Loki",
                    args={"container": MANY_LINES, "query": "str  # LogQL"},
                    example={"container": "api"},
                )
            }
        )
        out = await _introspect_tool(registry).ainvoke({"name_or_id": "loki.query"})
        assert out.splitlines() == [
            "[skill:loki.query]",
            "name: loki.query",
            "kind: tool",
            "summary: Query Loki",
            "args:",
            "  container: first line",
            "      indented line",
            "    last line",
            "  query: str  # LogQL",
            "example:",
            "  container: 'api'",
        ]

    async def test_an_arg_with_no_description_still_shows_its_name(self) -> None:
        registry = FakeRegistry(
            introspect_payloads={
                "loki.query": SkillToolIntrospect(
                    mcp_name="loki.query", summary="Query Loki", args={"container": BLANK, "query": "str"}, example={}
                )
            }
        )
        out = await _introspect_tool(registry).ainvoke({"name_or_id": "loki.query"})
        lines = out.splitlines()
        assert lines[lines.index("args:") + 1 : lines.index("example:")] == ["  container:", "  query: str"]


class TestSkillHistoryUses:
    async def test_many_line_notes_stay_under_their_use_and_outcome_notes_are_one_line(self) -> None:
        tool, skill_id = _history(FakeSkillsCollection(), notes=MANY_LINES, outcome_notes="it\n\n worked\nwell")
        out = await tool.ainvoke({"skill_id": str(skill_id)})
        assert out.splitlines()[1:] == [
            f"- Oct 9, 2026 at 5:00 AM PDT, invoked: success -- it worked well [conversation:{CONV}]",
            "  why it was invoked: first line",
            "      indented line",
            "    last line",
        ]

    async def test_blank_notes_still_show_the_reason_heading(self) -> None:
        tool, skill_id = _history(FakeSkillsCollection(), notes=BLANK)
        out = await tool.ainvoke({"skill_id": str(skill_id)})
        assert out.splitlines()[1:] == [
            f"- Oct 9, 2026 at 5:00 AM PDT, invoked: success [conversation:{CONV}]",
            "  why it was invoked:",
        ]

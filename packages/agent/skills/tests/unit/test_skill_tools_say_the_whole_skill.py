"""The skill tools say the whole skill, and what each call did.

- ``skill_invoke`` gives the steps whole and names only the tools the turn can
  bind: a skill naming a tool the agent does not hold sent it calling one that
  is not there;
- ``skill_get`` shows the skill with how often it was used and how it went,
  times in the reader's zone;
- ``skill_update`` says each field it changed, was -> now, and a change to
  nothing writes nothing;
- ``skill_report_outcome`` keeps its note and counts a use once;
- ``skill_history`` lists the uses, newest first, with the page it is.

The one-statement SQL behind the outcome is tested on real Postgres in
``tests/integration/test_tools_e2e.py``; these drive the tools over the fakes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from uuid_utils import uuid7

from threetears.agent.skills.tools import (
    BODY_MAX_BYTES,
    load_skill_get_tool,
    load_skill_history_tool,
    load_skill_invoke_tool,
    load_skill_report_outcome_tool,
    load_skill_update_tool,
)

from .skill_fakes import FakeInvocationsCollection, FakeRegistry, FakeSkillsCollection


def _id() -> UUID:
    return UUID(str(uuid7()))


AGENT, PERSON, CONV = _id(), _id(), _id()


def _seed(coll: FakeSkillsCollection, **fields: Any) -> UUID:
    skill_id = _id()
    now = datetime(2026, 10, 9, 22, 0, tzinfo=UTC)
    row: dict[str, Any] = {
        "skill_id": skill_id,
        "agent_id": AGENT,
        "user_id": PERSON,
        "name": "tide-check",
        "summary": "read the tide table",
        "body": "Open the tide table.\nSay the next low water.",
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


class _Active:
    def __init__(self) -> None:
        self.id: UUID | None = None

    def probe(self) -> UUID | None:
        return self.id

    def set(self, skill_id: UUID) -> None:
        self.id = skill_id


def _invoke_tool(coll: FakeSkillsCollection, inv: FakeInvocationsCollection, active: _Active, bindable: Any) -> Any:
    [tool] = load_skill_invoke_tool(
        agent_id=AGENT,
        user_id=PERSON,
        skills_collection=coll,  # type: ignore[arg-type]
        invocations_collection=inv,  # type: ignore[arg-type]
        conversation_id_resolver=lambda: CONV,
        active_skill_probe=active.probe,
        active_skill_setter=active.set,
        bindable_tools=bindable,
    )
    return tool


class TestSkillInvokeNamesOnlyBoundTools:
    async def test_the_steps_come_whole_however_long(self) -> None:
        coll, active = FakeSkillsCollection(), _Active()
        body = ("A step that says exactly what to do next.\n" * 2000)[: BODY_MAX_BYTES - 100] + "THE LAST STEP"
        skill_id = _seed(coll, body=body)
        out = await _invoke_tool(coll, FakeInvocationsCollection(coll), active, lambda: set()).ainvoke(
            {"skill_id": str(skill_id)}
        )
        assert out.endswith("THE LAST STEP")
        assert body in out
        assert 'Follow the skill "tide-check" for this task:' in out

    async def test_a_tool_it_adds_is_named_only_when_the_turn_holds_it(self) -> None:
        coll, active = FakeSkillsCollection(), _Active()
        skill_id = _seed(coll, tool_additions=["web_search", "ghost_tool"], tool_restrictions=["email_send", "gone"])
        out = await _invoke_tool(
            coll, FakeInvocationsCollection(coll), active, lambda: {"web_search", "email_send", "memory_add"}
        ).ainvoke({"skill_id": str(skill_id)})
        assert "Tools it adds, yours from your next step: web_search." in out
        assert "1 more tool(s) it names cannot be used on this turn." in out
        assert "Tools it puts away until the turn ends: email_send." in out
        assert "ghost_tool" not in out and "gone" not in out

    async def test_with_no_probe_no_tool_is_named(self) -> None:
        coll, active = FakeSkillsCollection(), _Active()
        skill_id = _seed(coll, tool_additions=["web_search"], tool_restrictions=["email_send"])
        out = await _invoke_tool(coll, FakeInvocationsCollection(coll), active, None).ainvoke(
            {"skill_id": str(skill_id)}
        )
        assert "web_search" not in out and "email_send" not in out
        assert "It changes 2 of your tools; your next step shows which you hold." in out

    async def test_a_skill_with_no_tools_says_nothing_of_tools(self) -> None:
        """Good traffic: a plain skill's result is its header and its steps."""
        coll, active = FakeSkillsCollection(), _Active()
        skill_id = _seed(coll)
        out = await _invoke_tool(coll, FakeInvocationsCollection(coll), active, lambda: {"web_search"}).ainvoke(
            {"skill_id": str(skill_id)}
        )
        assert out.splitlines()[:3] == [
            "[ACTIVE SKILL: tide-check] in use for the rest of this turn.",
            "prompt_mode: additive",
            "",
        ]
        assert "tool" not in out.split("\n\n", 1)[0].replace("[ACTIVE SKILL", "")


class TestSkillGetIsWhole:
    async def test_counts_and_times_in_the_readers_zone(self) -> None:
        coll = FakeSkillsCollection()
        used = datetime(2026, 10, 10, 4, 11, tzinfo=UTC)
        skill_id = _seed(coll, use_count=3, last_used_at=used, success_count=2, failure_count=1, last_failure_at=used)
        [tool] = load_skill_get_tool(
            agent_id=AGENT,
            user_id=PERSON,
            skills_collection=coll,  # type: ignore[arg-type]
            tz="America/Los_Angeles",
        )
        out = await tool.ainvoke({"skill_id": str(skill_id)})
        assert "Say the next low water." in out
        assert "used: 3 times, last Oct 9, 2026 at 9:11 PM PDT" in out
        assert "reported: worked 2, failed 1 (last failure Oct 9, 2026 at 9:11 PM PDT)" in out
        assert "written: Oct 9, 2026 at 3:00 PM PDT" in out
        assert "skill_history lists each use." in out

    async def test_a_skill_never_used_says_so(self) -> None:
        coll = FakeSkillsCollection()
        skill_id = _seed(coll)
        [tool] = load_skill_get_tool(agent_id=AGENT, user_id=PERSON, skills_collection=coll)  # type: ignore[arg-type]
        out = await tool.ainvoke({"skill_id": str(skill_id)})
        assert "used: never" in out and "UTC" in out
        assert "skill_history" not in out


def _update_tool(coll: FakeSkillsCollection) -> Any:
    [tool] = load_skill_update_tool(
        agent_id=AGENT,
        user_id=PERSON,
        skills_collection=coll,  # type: ignore[arg-type]
        registry=FakeRegistry(permitted_tools={"web_search"}),
    )
    return tool


class TestSkillUpdateSaysWhatChanged:
    async def test_each_field_was_and_now(self) -> None:
        coll = FakeSkillsCollection()
        skill_id = _seed(coll)
        out = await _update_tool(coll).ainvoke(
            {
                "skill_id": str(skill_id),
                "summary": "read the tide table aloud",
                "enabled": False,
                "tool_additions": ["web_search"],
            }
        )
        assert out.splitlines() == [
            f"Changed [skill:{skill_id}] tide-check, was -> now:",
            '- summary: "read the tide table" -> "read the tide table aloud"',
            "- tool_additions: (none) -> [web_search]",
            "- enabled: true -> false",
        ]
        assert coll.rows[(AGENT, skill_id)]["summary"] == "read the tide table aloud"

    async def test_a_long_body_says_its_size_and_where_it_changed(self) -> None:
        coll = FakeSkillsCollection()
        old = "\n".join(f"step {n}" for n in range(100))
        skill_id = _seed(coll, body=old)
        new = old.replace("step 42", "step forty-two")
        out = await _update_tool(coll).ainvoke({"skill_id": str(skill_id), "body": new})
        assert f"- body: {len(old):,} -> {len(new):,} characters; first change at line 43: " in out
        assert '"step 42" -> "step forty-two"' in out
        assert coll.rows[(AGENT, skill_id)]["body"] == new

    async def test_a_change_to_nothing_writes_nothing(self) -> None:
        coll = FakeSkillsCollection()
        skill_id = _seed(coll)
        stamped = coll.rows[(AGENT, skill_id)]["date_updated"]
        out = await _update_tool(coll).ainvoke({"skill_id": str(skill_id), "summary": "read the tide table"})
        assert out == (
            f"[NO CHANGE] skill_update: [skill:{skill_id}] tide-check already reads that way; nothing was written."
        )
        assert coll.rows[(AGENT, skill_id)]["date_updated"] == stamped


async def _report_setup(coll: FakeSkillsCollection) -> tuple[Any, UUID, FakeInvocationsCollection]:
    inv, active = FakeInvocationsCollection(coll), _Active()
    skill_id = _seed(coll)
    await _invoke_tool(coll, inv, active, lambda: set()).ainvoke({"skill_id": str(skill_id), "rationale": "asked"})
    [report] = load_skill_report_outcome_tool(
        agent_id=AGENT,
        conversation_id_resolver=lambda: CONV,
        active_skill_probe=active.probe,
        invocations_collection=inv,  # type: ignore[arg-type]
        skills_collection=coll,  # type: ignore[arg-type]
    )
    return report, skill_id, inv


class TestSkillReportOutcomeCountsAUseOnce:
    async def test_the_note_is_kept_and_the_reason_stays(self) -> None:
        coll = FakeSkillsCollection()
        report, skill_id, inv = await _report_setup(coll)
        out = await report.ainvoke({"outcome": "success", "notes": "the low water was 04:11"})
        assert out == (
            f'Recorded success for this use of [skill:{skill_id}] tide-check ("the low water was 04:11"). '
            "It has worked 1 and failed 0 time(s)."
        )
        row = inv.latest()
        assert row is not None and row["outcome_notes"] == "the low water was 04:11" and row["notes"] == "asked"

    async def test_a_second_report_of_the_same_outcome_counts_nothing(self) -> None:
        coll = FakeSkillsCollection()
        report, skill_id, _ = await _report_setup(coll)
        await report.ainvoke({"outcome": "success"})
        out = await report.ainvoke({"outcome": "success"})
        assert out.startswith("[NO CHANGE] skill_report_outcome:")
        assert coll.rows[(AGENT, skill_id)]["success_count"] == 1

    async def test_a_corrected_outcome_moves_the_count(self) -> None:
        coll = FakeSkillsCollection()
        report, skill_id, _ = await _report_setup(coll)
        await report.ainvoke({"outcome": "success"})
        out = await report.ainvoke({"outcome": "failure", "notes": "it was the wrong day"})
        assert out == (
            f'Changed this use of [skill:{skill_id}] tide-check from success to failure ("it was the wrong day"). '
            "It has worked 0 and failed 1 time(s)."
        )
        row = coll.rows[(AGENT, skill_id)]
        assert (row["success_count"], row["failure_count"]) == (0, 1)


class TestSkillHistory:
    async def test_uses_newest_first_with_the_page_they_are(self) -> None:
        coll = FakeSkillsCollection()
        inv = FakeInvocationsCollection(coll)
        skill_id = _seed(coll, success_count=1)
        start = datetime(2026, 10, 9, 10, 0, tzinfo=UTC)
        for n in range(3):
            row_id = _id()
            inv.rows[(AGENT, row_id)] = {
                "agent_id": AGENT,
                "invocation_id": row_id,
                "skill_id": skill_id,
                "user_id": PERSON,
                "conversation_id": CONV,
                "message_id": None,
                "invocation_source": "wake" if n == 0 else "invoke",
                "invoked_at": start + timedelta(hours=n),
                "outcome": "success" if n == 2 else None,
                "outcome_source": "agent_tool" if n == 2 else None,
                "notes": "asked" if n == 2 else None,
                "outcome_notes": "found it" if n == 2 else None,
            }
        [tool] = load_skill_history_tool(
            agent_id=AGENT,
            user_id=PERSON,
            skills_collection=coll,  # type: ignore[arg-type]
            invocations_collection=inv,  # type: ignore[arg-type]
            tz="America/Los_Angeles",
        )
        out = await tool.ainvoke({"skill_id": "tide-check", "limit": 2})
        lines = out.splitlines()
        assert lines[0] == (
            f"[skill:{skill_id}] tide-check: 3 use(s) in all; worked 1, failed 0. Uses 1-2 of 3, newest first:"
        )
        assert lines[1] == f"- Oct 9, 2026 at 5:00 AM PDT, invoked: success -- found it [conversation:{CONV}]"
        assert lines[2] == "  why it was invoked: asked"
        assert lines[3].startswith("- Oct 9, 2026 at 4:00 AM PDT, invoked: no outcome reported")
        assert lines[-1] == "skill_history with offset=2 shows the next 1."
        last = await tool.ainvoke({"skill_id": str(skill_id), "limit": 2, "offset": 2})
        assert "Uses 3-3 of 3" in last and "a wake ran it" in last and "offset=" not in last.splitlines()[-1]

    async def test_a_skill_never_used(self) -> None:
        coll = FakeSkillsCollection()
        skill_id = _seed(coll)
        [tool] = load_skill_history_tool(
            agent_id=AGENT,
            user_id=PERSON,
            skills_collection=coll,  # type: ignore[arg-type]
            invocations_collection=FakeInvocationsCollection(coll),  # type: ignore[arg-type]
        )
        assert (
            await tool.ainvoke({"skill_id": str(skill_id)}) == f"[skill:{skill_id}] tide-check has not been used yet."
        )

    async def test_another_persons_skill_is_not_found(self) -> None:
        coll = FakeSkillsCollection()
        skill_id = _seed(coll, user_id=_id())
        [tool] = load_skill_history_tool(
            agent_id=AGENT,
            user_id=PERSON,
            skills_collection=coll,  # type: ignore[arg-type]
            invocations_collection=FakeInvocationsCollection(coll),  # type: ignore[arg-type]
        )
        out = await tool.ainvoke({"skill_id": str(skill_id)})
        assert out == "[TOOL ERROR] skill_history: skill not found"

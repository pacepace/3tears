"""``intention_list`` shows a want's text under the want's own heading line.

A many-line want (with a blank line and an indented line) is listed as its
heading, then every line of the text indented, with no blank line inside the
item, so the next ``- [intention:...]`` line is the next want.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from threetears.agent.acl import AclCache
from threetears.agent.intention.authorize import IntentionAuthorizerDependencies
from threetears.agent.intention.entities import IntentionEntity
from threetears.agent.intention.tools import load_intention_list_tool

WRITTEN = "Oct 9, 2026 at 5:00 AM PDT"


class _NoMemberships:
    """membership loader returning no memberships; the owner path never needs one."""

    async def load_for_user(self, user_id: UUID) -> tuple[Any, ...]:
        _ = user_id
        return ()

    async def load_for_agent(self, agent_id: UUID) -> tuple[Any, ...]:
        _ = agent_id
        return ()

    async def load_for_group(self, group_id: UUID) -> tuple[Any, ...]:
        _ = group_id
        return ()


class _NoGrants:
    """grant loader returning no assignments / roles / groups."""

    async def load_assignments_for_groups(self, group_ids: tuple[UUID, ...], namespace: Any) -> tuple[Any, ...]:
        _ = group_ids, namespace
        return ()

    async def load_roles(self, role_ids: tuple[UUID, ...]) -> dict[UUID, Any]:
        _ = role_ids
        return {}

    async def load_groups(self, group_ids: tuple[UUID, ...]) -> dict[UUID, Any]:
        _ = group_ids
        return {}


# parity-exempt: answers only find_by_user, the one call a status=all listing makes; the full IntentionsCollection is L3-backed and exercised by the integration tests
class _WantsOf:
    """in-memory stand-in for the slice of :class:`IntentionsCollection` a listing calls."""

    def __init__(self, wants: list[IntentionEntity]) -> None:
        self.wants = wants

    async def find_by_user(self, user_id: UUID, **kwargs: Any) -> list[IntentionEntity]:
        _ = user_id, kwargs
        return list(self.wants)


def _want(content: str) -> IntentionEntity:
    return IntentionEntity(
        {
            "intention_id": uuid4(),
            "agent_id": uuid4(),
            "customer_id": uuid4(),
            "user_id": uuid4(),
            "status": "open",
            "content": content,
            "embedding": None,
            "salience": Decimal("0.5000"),
            "last_decayed_at": None,
            "last_surfaced_at": None,
            "source_memory_id": None,
            "source_conversation_id": None,
            "date_created": datetime(2026, 10, 9, 12, 0, tzinfo=UTC),
            "date_updated": datetime(2026, 10, 9, 12, 0, tzinfo=UTC),
        },
        is_new=False,
    )


async def _list(wants: list[IntentionEntity], notes: Any = None) -> list[str]:
    (tool,) = await load_intention_list_tool(
        uuid4(),
        uuid4(),
        uuid4(),
        IntentionAuthorizerDependencies(
            acl_cache=AclCache(membership_loader=_NoMemberships(), grant_loader=_NoGrants())
        ),
        _WantsOf(wants),  # type: ignore[arg-type]
        tz="America/Los_Angeles",
        notes=notes,
    )
    out = await tool.ainvoke({"status": "all"})
    assert isinstance(out, str)
    return out.splitlines()


class TestAWantStaysUnderItsHeading:
    async def test_every_line_of_a_many_line_want_is_indented_under_its_heading(self) -> None:
        first, second = _want("ask about the trip\n\n  to Lisbon\nand the dates"), _want("call the bank")
        lines = await _list([first, second])
        assert lines == [
            "2 of 2 wants, strongest first.",
            f"- [intention:{first.intention_id}] open, written {WRITTEN}: ask about the trip",
            "    to Lisbon",
            "  and the dates",
            f"- [intention:{second.intention_id}] open, written {WRITTEN}: call the bank",
        ]

    async def test_the_consumers_note_closes_the_last_line_of_the_want(self) -> None:
        want = _want("ask about the trip\n\nto Lisbon")

        async def notes(wants: Sequence[IntentionEntity]) -> Mapping[UUID, str]:
            return {w.intention_id: "project: travel\n\nlisbon" for w in wants}

        lines = await _list([want], notes)
        assert lines[1:] == [
            f"- [intention:{want.intention_id}] open, written {WRITTEN}: ask about the trip",
            "  to Lisbon (project: travel",
            "  lisbon)",
        ]

    async def test_a_want_with_no_text_still_shows_its_heading(self) -> None:
        want = _want("  \n \n")
        lines = await _list([want])
        assert lines[1:] == [f"- [intention:{want.intention_id}] open, written {WRITTEN}:"]

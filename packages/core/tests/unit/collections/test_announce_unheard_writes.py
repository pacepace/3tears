"""writes no collection made are put on the epoch system by advancing their tables with no rows.

A schema migration or a restore changes rows it cannot name. Advancing each table it wrote, and
publishing nothing, is what makes every follower drop what it holds of the table: the reach is
unknown, so everything derived from the table goes.
"""

from __future__ import annotations

import pytest

from threetears.core.collections import announce_unheard_writes
from threetears.core.collections.generation import GenerationMarks, GenerationVerdict
from threetears.core.exceptions import GenerationUnavailableError


class _Source:
    """counts per table under one incarnation; refuses the tables it is told to."""

    def __init__(self, refuse: frozenset[str] = frozenset()) -> None:
        self.counts: dict[str, int] = {}
        self.refuse = refuse

    async def current(self, table_name: str) -> str:
        return f"inc-a:{self.counts.get(table_name, 0)}"

    async def advance(self, table_name: str) -> str | None:
        if table_name in self.refuse:
            raise GenerationUnavailableError(f"no write on the bucket for {table_name}")
        self.counts[table_name] = self.counts.get(table_name, 0) + 1
        return f"inc-a:{self.counts[table_name]}"


async def test_each_table_is_advanced_once_and_its_token_returned() -> None:
    source = _Source()
    tokens = await announce_unheard_writes(source, ["groups", "roles", "groups"])
    assert tokens == {"groups": "inc-a:1", "roles": "inc-a:1"}
    assert source.counts == {"groups": 1, "roles": 1}


async def test_a_follower_that_heard_no_rows_drops_the_table() -> None:
    marks = GenerationMarks()
    marks.follow("group_members")
    assert marks.settle("group_members", "inc-a:0") is GenerationVerdict.FIRST_SIGHT
    tokens = await announce_unheard_writes(_Source(), ["group_members"])
    verdict = marks.settle("group_members", tokens["group_members"])
    assert verdict is GenerationVerdict.MISSED
    assert verdict.drops


async def test_every_table_is_attempted_and_the_failures_are_raised_together() -> None:
    source = _Source(refuse=frozenset({"groups", "roles"}))
    with pytest.raises(GenerationUnavailableError) as raised:
        await announce_unheard_writes(source, ["groups", "group_members", "roles", "role_assignments"])
    assert source.counts == {"group_members": 1, "role_assignments": 1}
    assert "groups" in str(raised.value) and "roles" in str(raised.value)

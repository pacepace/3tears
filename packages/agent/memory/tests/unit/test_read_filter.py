"""``filter_clause``: the filter's values land after the query's own, and ``None`` leaves the query as it was."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from threetears.agent.memory.read_filter import ReadFiltered, filter_clause


def _two(tags: str, bind: Callable[[Any], str]) -> str:
    return f"NOT ({tags} ? {bind('a')}) AND NOT ({tags} ? {bind('b')})"


def test_no_filter_adds_nothing() -> None:
    args: list[Any] = ["embedding", 7]
    assert filter_clause(None, "m.tags", args) == ""
    assert args == ["embedding", 7]


def test_bound_values_follow_the_querys_own() -> None:
    args: list[Any] = ["embedding", "agent", "user", 10]
    clause = filter_clause(_two, "m.tags", args)
    assert clause == " AND (NOT (m.tags ? $5) AND NOT (m.tags ? $6))"
    assert args == ["embedding", "agent", "user", 10, "a", "b"]


def test_a_filter_that_binds_then_answers_none_leaves_the_args_alone() -> None:
    def changes_its_mind(tags: str, bind: Callable[[Any], str]) -> None:
        bind("x")
        bind("y")

    args: list[Any] = [1, 2]
    assert filter_clause(changes_its_mind, "tags", args) == ""
    assert args == [1, 2]


def test_the_filter_is_asked_every_time() -> None:
    calls: list[str] = []

    def counting(tags: str, bind: Callable[[Any], str]) -> str:
        calls.append(tags)
        return "true"

    filter_clause(counting, "a.tags", [])
    filter_clause(counting, "b.tags", [])
    assert calls == ["a.tags", "b.tags"]


def test_the_slot_starts_empty_and_is_set_per_instance() -> None:
    first, second = ReadFiltered(), ReadFiltered()
    assert first.read_filter is None
    first.set_read_filter(_two)
    assert first.read_filter is _two
    assert second.read_filter is None
    first.set_read_filter(None)
    assert first.read_filter is None

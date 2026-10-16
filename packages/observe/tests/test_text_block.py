"""a text under a heading: every line indented, none blank; or one line."""

from __future__ import annotations

from threetears.observe.text_block import one_line, under


def test_every_line_is_indented_and_blank_lines_are_dropped() -> None:
    assert under("# Garden\n\n- tomatoes\n- basil", "    ") == ["    # Garden", "    - tomatoes", "    - basil"]


def test_its_own_indentation_is_kept_relative_to_its_least_indented_line() -> None:
    assert under("  a\n    b", "> ") == ["> a", ">   b"]


def test_the_first_line_can_carry_a_marker() -> None:
    assert under("one\ntwo", "    ", first="  - ") == ["  - one", "    two"]


def test_an_empty_text_is_no_lines() -> None:
    assert under("", "  ") == [] and under("\n  \n", "  ") == []


def test_an_empty_text_under_a_heading_is_the_heading_alone() -> None:
    assert under("", "  ", first="- [chunk:7] ") == ["- [chunk:7]"]
    assert under(" \n\n", "    ", first="  Plan: ") == ["  Plan:"]


def test_one_line_joins_lines_and_keeps_the_spacing_inside_one() -> None:
    assert one_line(" a\n\tb   c \n") == "a b   c"
    assert one_line("/notes/a.md  (3 chars)") == "/notes/a.md  (3 chars)"

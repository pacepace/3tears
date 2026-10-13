"""How a caller reads whether a tool's call worked, without parsing its sentence."""

from __future__ import annotations

import pytest

from threetears.agent.tools.utils import (
    NO_CHANGE_PREFIX,
    TOOL_ERROR_PREFIX,
    tool_error,
    tool_no_change,
    tool_result_status,
)


class TestTheStatusIsReadFromHowTheTextOpens:
    def test_the_two_prefixes_are_the_convention(self) -> None:
        assert (TOOL_ERROR_PREFIX, NO_CHANGE_PREFIX) == ("[TOOL ERROR]", "[NO CHANGE]")

    def test_an_error_reads_as_an_error(self) -> None:
        assert tool_result_status(tool_error("add_todo", "write", "database is down")) == "error"

    def test_a_no_change_reads_as_its_own_status_and_never_as_an_error(self) -> None:
        result = tool_no_change("add_todo", 'Todo "Buy milk" already exists in "Todo List".')
        assert result == '[NO CHANGE] add_todo: Todo "Buy milk" already exists in "Todo List".'
        assert tool_result_status(result) == "no_change"

    @pytest.mark.parametrize("result", ["Todo added.", "", "No todos in this conversation.", "it said [TOOL ERROR]"])
    def test_anything_else_is_a_call_that_worked(self, result: str) -> None:
        assert tool_result_status(result) == "ok"

    @pytest.mark.parametrize("lead", ["", " ", "\n"])
    def test_leading_space_does_not_hide_a_prefix(self, lead: str) -> None:
        assert tool_result_status(f"{lead}[TOOL ERROR] x") == "error"
        assert tool_result_status(f"{lead}[NO CHANGE] x") == "no_change"

    @pytest.mark.parametrize("result", [None, 3, {"ok": False}, ("[TOOL ERROR] x", {})])
    def test_what_is_not_text_is_not_judged_here(self, result: object) -> None:
        assert tool_result_status(result) == "ok"

"""``context_recall`` asked for an id nothing is saved under names the saved results that exist."""

from __future__ import annotations

from typing import Any

import pytest

from threetears.agent.tools.context import missing_result_text
from threetears.agent.tools.workflow import load_workflow_tools


def _item(n: int, kind: str = "tool_result") -> dict[str, Any]:
    return {"context_id": f"id-{n:02d}", "context_type": kind, "key": f"web_search:q{n}", "content": "x"}


def test_a_miss_names_the_saved_results_newest_first_with_the_mark_each_opens_by() -> None:
    said = missing_result_text("nope", [_item(1), _item(2, "variable"), _item(3)])
    assert said.splitlines() == [
        "Saved result 'nope' not found in this conversation. Saved here:",
        "- [ctx:id-03] web_search:q3",
        "- [ctx:id-01] web_search:q1",
    ]


def test_a_long_list_names_the_newest_ten_and_says_how_many_there_are() -> None:
    said = missing_result_text("nope", [_item(n) for n in range(14)])
    lines = said.splitlines()
    assert lines[0] == "Saved result 'nope' not found in this conversation. The newest 10 of 14 saved here:"
    assert lines[1] == "- [ctx:id-13] web_search:q13" and lines[-1] == "- [ctx:id-04] web_search:q4"
    assert len(lines) == 11


def test_a_conversation_with_nothing_saved_says_so() -> None:
    assert missing_result_text("nope", [_item(1, "variable")]) == (
        "Saved result 'nope' not found in this conversation. No tool result is saved in it."
    )


class _Context:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items

    async def get_context_item(self, context_id: str) -> dict[str, Any] | None:
        return next((i for i in self.items if i["context_id"] == context_id), None)


@pytest.mark.asyncio
async def test_the_workflow_recall_tool_answers_a_miss_with_the_list() -> None:
    [recall] = [t for t in load_workflow_tools(_Context([_item(1)])) if t.name == "context_recall"]  # type: ignore[arg-type]
    said = await recall.ainvoke({"context_id": "nope"})
    assert said.splitlines() == [
        "Saved result 'nope' not found in this conversation. Saved here:",
        "- [ctx:id-01] web_search:q1",
    ]
    assert (await recall.ainvoke({"context_id": "id-01"})).endswith("web_search:q1: x")

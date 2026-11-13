"""A many-line text the engine prints in a list stays with its item, every line of it.

Written as ``f"  error: {error}"``, a two-line error put its second line at the margin, where it read as the
summary's next line rather than the rest of the error. Each such text now goes through ``one_line`` (a line of a
list) or ``under`` (a block under its item), the evals copies of :mod:`threetears.observe.text_block`'s.
"""

from __future__ import annotations

import dataclasses

from threetears.evals.analysis import CaseResult, summarize_run
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.text_block import one_line, under
from threetears.evals.quick.one_call import callable_host
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_result, make_eval_run

_MANY = "the tool timed out\n\n  after 30 s, twice"


def test_one_line_joins_the_lines_and_keeps_the_spacing_inside_one() -> None:
    assert one_line("  first  line \n\n second\r\nthird ") == "first  line second third"
    assert one_line("one line") == "one line"
    assert one_line("\n \n") == ""


def test_under_indents_every_line_keeps_its_shape_and_drops_blank_lines() -> None:
    assert under("a\n\n  - b\n  - c", "    ", first="  - x: ") == ["  - x: a", "      - b", "      - c"]
    assert under("  only", "  ", first="- k: ") == ["- k: only"]
    assert under(" \n", "  ", first="- k: ") == ["- k:"]
    assert under("", "  ") == []


def test_a_case_results_many_line_error_and_miss_reason_stay_on_their_lines() -> None:
    result = CaseResult(
        case="c1",
        case_id="tc-1",
        input="q",
        repeat=1,
        outcome="failed",
        state="failed",
        errors=[_MANY],
        missed_because=[f"failed: {_MANY}"],
    )
    lines = result.render().splitlines()
    assert lines[1:] == [
        "  error: the tool timed out after 30 s, twice",
        "  missed: failed: the tool timed out after 30 s, twice",
    ]


def test_a_summarys_many_line_error_stays_on_its_line() -> None:
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_eval_run(make_eval_run(id="run-1", scope_id="s", k_runs=1, test_case_ids=["tc-1"], status="completed"))
    storage.save_eval_result(make_eval_result(id="res-1", scope_id="s", eval_run_id="run-1", test_case_id="tc-1"))
    summary = summarize_run(dataclasses.replace(callable_host(), storage=storage), "run-1", "s")
    rendered = summary.model_copy(update={"errors": [_MANY]}).render().splitlines()
    assert rendered[-1] == "  error: the tool timed out after 30 s, twice"

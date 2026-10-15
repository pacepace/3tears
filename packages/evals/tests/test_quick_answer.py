"""A runner reports its own spend by returning an ``Answer``, and a plain answer still reports none.

The answer's value is what is graded and stored; its spend lands on each result as the ``candidate`` usage
row, from which the engine derives the result's ``cost_usd``, and on the summary as the candidate's calls
and dollars. Unpriced stays unknown, never zero. Across arms, a comparison's report tests the
``cost_usd`` reading against the control like any other.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.analysis import TableBlock
from threetears.evals import Answer, Eval
from threetears.evals.quick.one_call import callable_host
from threetears.evals.run import get_result_trace, list_results
from packages.evals.tests.off_loop import off_loop

SCOPE = "candidate-answer-tests"
CASES = [
    {"text": "a cat", "label": "animal"},
    {"text": "a fir", "label": "plant"},
    {"text": "an oak", "label": "plant"},
]

#: Enough cases for a contrast to be tested, each a different length so the priced candidate's spend varies.
COMPARED_CASES = [
    *CASES,
    {"text": "a heron", "label": "animal"},
    {"text": "a sycamore", "label": "plant"},
    {"text": "a salamander", "label": "animal"},
]

#: Dollars per input token, for the priced runner below.
RATE = 1e-6

#: The field each case carries its label in, graded as the classifier.
LABEL = "label"


async def priced(case: Mapping[str, Any]) -> Answer:
    """Answers right, and reports one call priced at ``RATE`` per input token."""
    tokens = len(case["text"]) * 10
    return Answer(case["label"], model="m-1", input_tokens=tokens, output_tokens=2, cost_usd=tokens * RATE)


async def test_an_answer_s_spend_lands_on_each_result_and_on_the_summary() -> None:
    host = callable_host()
    summary = (await Eval(CASES, priced, [LABEL], k=1).run(host=host, scope=SCOPE)).summary

    # The value is what was graded: every label matched.
    (match,) = [measure for measure in summary.measures if measure.name == "match"]
    assert match.mean == 1.0
    tokens = [len(case["text"]) * 10 for case in CASES]
    assert summary.role_spend("candidate").calls == 3
    assert summary.role_spend("candidate").cost_usd == pytest.approx(sum(tokens) * RATE)
    # $0.00016, shown to three significant figures rather than rounded away.
    assert "candidate spend: $0.000160 over 3 call(s)" in summary.render()
    for result in off_loop(list_results, host.storage, summary.run_id, SCOPE):
        (row,) = result.usage
        assert (row.role, row.model, row.call_count) == ("candidate", "m-1", 1)
        assert row.prompt_tokens in tokens and row.completion_tokens == 2
        assert result.cost_usd == pytest.approx(row.cost_usd)
        # The stored output is the answer's value, never the wrapper.
        trace = off_loop(get_result_trace, host.storage, result)
        assert trace is not None and trace.trace[0]["value"] in {"animal", "plant"}


async def test_a_plain_answer_reports_no_spend_as_before() -> None:
    async def plain(case: Mapping[str, Any]) -> str:
        return str(case["label"])

    host = callable_host()
    summary = (await Eval(CASES, plain, [LABEL], k=1).run(host=host, scope=SCOPE)).summary
    assert summary.role_spend("candidate") is None
    assert "candidate spend" not in summary.render()
    assert all(result.usage == [] for result in off_loop(list_results, host.storage, summary.run_id, SCOPE))


async def test_an_unpriced_answer_leaves_the_cost_unknown_never_zero() -> None:
    async def unpriced(case: Mapping[str, Any]) -> Answer:
        return Answer(case["label"], model="local", input_tokens=5)

    host = callable_host()
    summary = (await Eval(CASES, unpriced, [LABEL], k=1).run(host=host, scope=SCOPE)).summary
    assert summary.role_spend("candidate").calls == 3 and summary.role_spend("candidate").cost_usd is None
    assert "candidate spend: unpriced over 3 call(s): its client reported no price for them" in summary.render()
    assert all(result.cost_usd is None for result in off_loop(list_results, host.storage, summary.run_id, SCOPE))


@pytest.mark.parametrize(
    ("spend", "refusal"),
    [
        ({"input_tokens": -1}, "input_tokens is a token count"),
        ({"output_tokens": 1.5}, "output_tokens is a token count"),
        ({"input_tokens": True}, "input_tokens is a token count"),
        ({"cost_usd": -0.01}, "cost_usd is dollars"),
        ({"cost_usd": float("nan")}, "cost_usd is dollars"),
    ],
)
def test_a_spend_no_usage_row_can_hold_is_refused(spend: dict[str, Any], refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        Answer("x", **spend)


async def test_a_runner_building_a_bad_answer_fails_its_cell() -> None:
    async def miscounted(case: Mapping[str, Any]) -> Answer:
        return Answer(case["label"], input_tokens=-1)

    summary = (await Eval(CASES, miscounted, [LABEL], k=1).run(store=":memory:", scope=SCOPE)).summary
    assert summary.n_candidate_failed == 3
    assert all("Answer.input_tokens" in error for error in summary.errors)


async def test_a_comparison_tests_the_arms_reported_cost_against_the_control() -> None:
    async def dearer(case: Mapping[str, Any]) -> Answer:
        cheaper = await priced(case)
        assert cheaper.cost_usd is not None
        return Answer(cheaper.value, model="m-2", input_tokens=cheaper.input_tokens, cost_usd=cheaper.cost_usd * 10)

    comparison = await Eval(COMPARED_CASES, {"dearer": dearer, "cheaper": priced}, [LABEL], k=2).run(
        store=":memory:", scope=SCOPE
    )
    assert comparison.arms["cheaper"].role_spend("candidate").cost_usd == pytest.approx(
        comparison.arms["dearer"].role_spend("candidate").cost_usd / 10  # type: ignore[operator]
    )
    tables = {block.name: block for block in comparison.report.blocks if isinstance(block, TableBlock)}
    by_reading = {row["reading"]: row for row in tables["comparisons"].values()}
    # The candidate's spend, not cost_usd, which sums a judge's spend too: what it cost to measure the arm.
    assert not {reading for reading in by_reading if reading.startswith("Measuring spend")}
    # Priced under a cent per reply on both arms, so the reading is headed per 1,000 replies.
    cost = by_reading["Production cost ($/1,000 replies)"]
    # Two arms: the candidate's figure heads its own column, by its name.
    assert next(column.header for column in tables["comparisons"].columns if column.key == "arm") == "cheaper"
    assert cost["verdict"] == "improved on the control"
    assert isinstance(cost["change"], float) and cost["change"] < 0

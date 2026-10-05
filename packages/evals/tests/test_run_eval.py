"""``run_eval``: what each cell records, every refusal it makes, and the store it leaves behind.

The rung-zero example (``test_rung_zero.py``) is the happy path read from outside. These are the
edges: a candidate that raises fails its cell and a scorer that misbehaves excludes it; every input
``run_eval`` refuses is refused before anything is stored; a caller's own host is used as given; the
case set is content-addressed; and a cancelled call settles the run it started.
"""

from __future__ import annotations

import asyncio
import functools
import math
from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.contracts import ValidationFailedError
from threetears.evals.quick import callable_host, run_eval
from threetears.evals.run import get_result_trace, list_results, list_runs, list_templates
from packages.evals.tests.fixtures.courierhost import ON_TIME_RATE, courier_host

SCOPE = "run-eval-tests"
CASES = [{"n": 1}, {"n": 2}, {"n": 3}]


async def double(case: Mapping[str, Any]) -> int:
    return int(case["n"]) * 2


def even(case: Mapping[str, Any], answer: Any) -> bool:
    return answer % 2 == 0


def size(case: Mapping[str, Any], answer: Any) -> float:
    return float(answer)


async def test_a_raising_candidate_fails_its_cell_and_says_why() -> None:
    async def flaky(case: Mapping[str, Any]) -> int:
        if case["n"] == 2:
            raise RuntimeError("cannot do two")
        return int(case["n"])

    summary = await run_eval(CASES, flaky, [size], scope_id=SCOPE)
    assert summary.status == "completed"
    assert (summary.n_scored, summary.n_candidate_failed, summary.n_excluded) == (2, 1, 0)
    assert summary.measures[0].n == 2 and summary.measures[0].mean == pytest.approx(2.0)
    assert len(summary.errors) == 1 and "the candidate raised RuntimeError: cannot do two" in summary.errors[0]


@pytest.mark.parametrize(
    ("returned", "said"),
    [("yes", "'yes'"), (None, "None"), (math.nan, "nan"), (math.inf, "inf")],
    ids=["a string", "None", "nan", "inf"],
)
async def test_a_scorer_returning_no_finite_number_excludes_the_cell(returned: Any, said: str) -> None:
    def odd_grade(case: Mapping[str, Any], answer: Any) -> Any:
        return returned

    summary = await run_eval(CASES[:1], double, [odd_grade], scope_id=SCOPE)
    assert (summary.n_scored, summary.n_candidate_failed, summary.n_excluded) == (0, 0, 1)
    assert f"the scorer odd_grade returned {said}" in summary.errors[0]


async def test_a_raising_scorer_excludes_the_cell_and_names_the_scorer() -> None:
    def broken(case: Mapping[str, Any], answer: Any) -> float:
        raise KeyError("expected")

    summary = await run_eval(CASES[:1], double, [even, broken], scope_id=SCOPE)
    assert summary.n_excluded == 1
    assert "the scorer broken raised KeyError" in summary.errors[0]


async def test_a_bool_scores_as_one_or_zero() -> None:
    summary = await run_eval(CASES, double, [even], scope_id=SCOPE)
    (measure,) = summary.measures
    assert (measure.mean, measure.minimum, measure.maximum, measure.n) == (1.0, 1.0, 1.0, 3)


async def test_the_answer_is_stored_verbatim_when_json_holds_it_and_as_its_repr_when_not() -> None:
    def graded(case: Mapping[str, Any], answer: Any) -> int:
        return 1

    async def as_set(case: Mapping[str, Any]) -> Any:
        return {case["n"]} if case["n"] == 1 else case["n"]

    host = callable_host([graded])
    summary = await run_eval(CASES[:2], as_set, [graded], scope_id=SCOPE, host=host)
    stored = []
    for result in list_results(host.storage, summary.run_id, SCOPE):
        trace = get_result_trace(host.storage, result)
        assert trace is not None
        stored.append(trace.trace)
    assert sorted(stored, key=str) == [[{"repr": "{1}"}], [{"value": 2}]]


async def test_a_callers_host_is_used_as_given_world_and_all() -> None:
    """The courier declares a world and a measure; run_eval runs in it and leaves the run in its store."""
    host = courier_host()

    def on_time_rate(case: Mapping[str, Any], answer: Any) -> float:
        return 1.0

    summary = await run_eval(CASES, double, [on_time_rate], scope_id=SCOPE, host=host, model="doubler")
    assert summary.status == "completed"
    assert {m.name: m.n for m in summary.measures}[ON_TIME_RATE] == 3
    (run,) = list_runs(host, SCOPE)
    assert run.candidate_model == "doubler"
    # The callable kind attaches no carrier, so the courier's one dimension is out of play.
    assert run.world_placements == {"road_closures": "out_of_play"}


async def test_two_calls_over_the_same_cases_share_a_template_and_different_cases_do_not() -> None:
    host = callable_host([even])
    first = await run_eval(CASES, double, [even], scope_id=SCOPE, host=host)
    second = await run_eval(CASES, double, [even], scope_id=SCOPE, host=host, model="again")
    other = await run_eval(CASES[:2], double, [even], scope_id=SCOPE, host=host)
    assert first.template_id == second.template_id != other.template_id
    assert {template.id for template in list_templates(host.storage, SCOPE)} == {first.template_id, other.template_id}
    assert len(list_runs(host, SCOPE)) == 3


async def test_a_cancelled_call_settles_its_run_as_cancelled() -> None:
    host = callable_host([even])
    started = asyncio.Event()

    async def stuck(case: Mapping[str, Any]) -> int:
        started.set()
        await asyncio.Event().wait()
        return 0

    call = asyncio.create_task(run_eval(CASES[:1], stuck, [even], scope_id=SCOPE, host=host))
    await asyncio.wait_for(started.wait(), timeout=5.0)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    (run,) = list_runs(host, SCOPE)
    assert run.status == "cancelled"


# --- refusals: each made before anything is stored -----------------------------------------------


@pytest.mark.parametrize(
    ("cases", "said"),
    [
        ([], "non-empty list of cases"),
        ({"n": 1}, "non-empty list of cases"),
        ("n", "non-empty list of cases"),
        ([{"n": 1}, ["n", 1]], "case 1 is not a mapping"),
        ([{1: "n"}], "case 0 is not a mapping with string keys"),
        ([{"n": {1, 2}}], "every case must be JSON"),
    ],
    ids=["empty", "one mapping", "a string", "a list among them", "a non-string key", "a set value"],
)
async def test_cases_that_are_not_a_list_of_json_objects_are_refused(cases: Any, said: str) -> None:
    host = callable_host([even])
    with pytest.raises(ValueError, match=said):
        await run_eval(cases, double, [even], scope_id=SCOPE, host=host)
    assert list_templates(host.storage, SCOPE) == []


@pytest.mark.parametrize(
    ("scorers", "said"),
    [
        ([], "at least one scorer"),
        ([lambda case, answer: 1.0], "has none a measure can carry"),
        ([functools.partial(size)], "has none a measure can carry"),
        ([even, even], "scorers named even more than once"),
    ],
    ids=["none", "a lambda", "a partial", "a repeated name"],
)
async def test_scorers_that_cannot_name_a_measure_each_are_refused(scorers: Any, said: str) -> None:
    with pytest.raises(ValueError, match=said):
        await run_eval(CASES, double, scorers, scope_id=SCOPE)


async def test_a_scorer_the_callers_host_declares_no_measure_for_is_refused() -> None:
    host = courier_host()
    with pytest.raises(ValueError, match="declares no measure named even"):
        await run_eval(CASES, double, [even], scope_id=SCOPE, host=host)
    assert list_templates(host.storage, SCOPE) == []


async def test_a_candidate_with_no_name_needs_a_model_label() -> None:
    unnamed = functools.partial(double)
    with pytest.raises(ValueError, match="no __name__ to label its arm by"):
        await run_eval(CASES, unnamed, [even], scope_id=SCOPE)
    summary = await run_eval(CASES, unnamed, [even], scope_id=SCOPE, model="partial-doubler")
    assert summary.candidate_model == "partial-doubler"


@pytest.mark.parametrize("k", [0, 21])
async def test_a_repeat_count_outside_the_runs_bounds_is_refused_by_the_launch(k: int) -> None:
    with pytest.raises(ValidationFailedError, match="invalid k_runs"):
        await run_eval(CASES, double, [even], scope_id=SCOPE, k=k)

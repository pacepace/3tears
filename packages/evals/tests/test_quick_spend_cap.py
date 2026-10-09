"""``max_cost_usd=`` on ``run_eval`` and ``compare``: a live run capped by the engine's own per-run cost cap.

With no cap, the run is uncapped as before, and the summary says so in the words the engine's spend-ceiling
reading uses. With one, ceiling enforcement is on, the launch names the cap (``chosen``), the run's cost cap
counts every result's reported spend, and a run that passes it stops ``budget_stopped`` between cases, its
summary saying why. A candidate that reports no spend is invisible to the cap, and the summary says that too.
``compare`` caps the whole comparison, each arm at an equal share.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.contracts.host.sweepables import UNCAPPED_SPEND
from threetears.evals.quick import Answer, callable_host, compare, run_eval
from threetears.evals.run import get_run

SCOPE = "quick-spend-cap-tests"
CASES = [{"n": n} for n in range(4)]


def even(case: Mapping[str, Any], answer: Any) -> bool:
    return answer % 2 == 0


async def costly(case: Mapping[str, Any]) -> Answer:
    """Answers each case for 30 cents."""
    return Answer(int(case["n"]), model="paid-model", input_tokens=10, output_tokens=2, cost_usd=0.30)


async def silent(case: Mapping[str, Any]) -> int:
    return int(case["n"])


async def test_with_no_cap_the_run_is_uncapped_and_the_summary_says_so() -> None:
    host = callable_host([even])
    summary = await run_eval(CASES, costly, [even], host=host, scope_id=SCOPE, k=1)
    run = get_run(host.storage, summary.run_id, SCOPE)
    assert run.max_cost_usd is None and run.max_cost_usd_origin == "uncapped"
    assert summary.status == "completed" and summary.max_cost_usd_origin == "uncapped"
    assert f"spend cap: {UNCAPPED_SPEND}" in summary.render()


async def test_a_cap_stops_the_run_between_cases_once_its_spend_passes_it_and_the_summary_says_why() -> None:
    host = callable_host([even])
    summary = await run_eval(CASES, costly, [even], host=host, scope_id=SCOPE, k=1, max_cost_usd=0.5)
    run = get_run(host.storage, summary.run_id, SCOPE)
    assert (run.max_cost_usd, run.max_cost_usd_origin) == (0.5, "chosen")
    # 30 cents, then 60: the second case passes the cap, so the run stops before the third.
    assert summary.status == "budget_stopped" and summary.n_results == 2
    assert summary.stopped_because is not None and "$0.6000 spent against a $0.5000 cap" in summary.stopped_because
    rendered = summary.render()
    assert "budget_stopped" in rendered and "stopped: eval run cost cap exceeded" in rendered
    assert "spend cap: $0.500 for this run\n" in rendered + "\n"
    assert len(summary.results()) == 2


async def test_a_cap_the_run_stays_under_lets_it_finish() -> None:
    summary = await run_eval(CASES, costly, [even], scope_id=SCOPE, k=1, max_cost_usd=5.0)
    assert summary.status == "completed" and summary.n_results == 4 and summary.max_cost_usd == 5.0


async def test_a_cap_over_a_candidate_that_reports_no_spend_says_it_counts_none_of_it() -> None:
    summary = await run_eval(CASES, silent, [even], scope_id=SCOPE, k=1, max_cost_usd=0.5)
    assert summary.status == "completed"
    assert "it counts only spend a result reports, and the candidate reported none" in summary.render()


async def test_an_unpriced_answer_stops_a_capped_run_rather_than_counting_as_free() -> None:
    async def unpriced(case: Mapping[str, Any]) -> Answer:
        return Answer(int(case["n"]), model="paid-model", input_tokens=10, output_tokens=2, cost_usd=None)

    summary = await run_eval(CASES, unpriced, [even], scope_id=SCOPE, k=1, max_cost_usd=5.0)
    assert summary.status == "budget_stopped" and summary.n_results == 1
    assert summary.stopped_because is not None and "could not be priced" in summary.stopped_because


@pytest.mark.parametrize("cap", [0, -1.0, float("nan"), float("inf"), True, "1"])
async def test_a_cap_that_is_not_a_positive_number_is_refused_before_anything_runs(cap: Any) -> None:
    host = callable_host([even])
    with pytest.raises(ValueError, match="max_cost_usd= is a spend ceiling in US dollars"):
        await run_eval(CASES, costly, [even], host=host, scope_id=SCOPE, max_cost_usd=cap)
    assert host.storage.query_templates(SCOPE) == []


async def test_compare_caps_the_whole_comparison_each_arm_at_an_equal_share() -> None:
    async def cheap(case: Mapping[str, Any]) -> Answer:
        return Answer(int(case["n"]), model="cheap-model", input_tokens=10, output_tokens=2, cost_usd=0.01)

    comparison = await compare(
        CASES, {"cheap": cheap, "costly": costly}, [even], control="cheap", scope_id=SCOPE, k=1, max_cost_usd=1.0
    )
    cheap_arm, costly_arm = comparison.arms["cheap"], comparison.arms["costly"]
    assert cheap_arm.max_cost_usd == costly_arm.max_cost_usd == 0.5
    assert cheap_arm.status == "completed" and cheap_arm.n_results == 4
    assert costly_arm.status == "budget_stopped" and costly_arm.n_results == 2
    # The contrast reads only the cases both arms ran, and says it left the cheap arm's other two out.
    (row,) = comparison.contrasts("even")
    assert "2 paired" in str(row["cases"]) and "2 of the control's left out" in str(row["cases"])


async def test_compare_with_no_cap_runs_every_arm_uncapped() -> None:
    comparison = await compare(CASES, {"a": silent, "b": costly}, [even], control="a", scope_id=SCOPE, k=1)
    assert all(summary.max_cost_usd_origin == "uncapped" for summary in comparison.arms.values())

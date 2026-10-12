"""A run's summary states what every role spent, and a total equal to its results' cost.

A persona run billed its candidate $0.00348, the inner agent it called $0.0166 and its judge $0.0137, and searched
six credits' worth on an external API; its summary printed "judge spend" and "candidate spend" alone, so the inner
agent's spend — the largest — was invisible, and nothing summed to what the results cost. A judge campaign, whose
every trial bills the ``judge`` role and none other, printed no spend at all and told its reader that "the candidate
reported none". So the summary carries a line per role that spent, external units included, and the results' total.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from threetears.evals.analysis import summarize_run
from threetears.evals.kernel import EvalStorage
from threetears.evals.quick import callable_host
from threetears.evals.schema import EvalResult, RoleUsage
from threetears.evals.kernel.usage_capture import blended_cost
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_result, make_eval_run

_SCOPE = "spend-scope"
_RUN = "run-persona"

#: The persona result discodon's end-to-end test found: three billed roles, and credits nobody priced.
_PERSONA_USAGE = [
    RoleUsage(role="candidate", model="m-cand", cost_usd=0.00348, call_count=2, price_source="provider"),
    RoleUsage(role="inner_agent", model="m-inner", cost_usd=0.0166, call_count=5, price_source="provider"),
    RoleUsage(role="judge", model="m-judge", cost_usd=0.0137, call_count=3, price_source="provider"),
    RoleUsage(role="external", provider="tavily", provider_unit="credits", provider_units=6, call_count=3),
]


def _result(
    usage: list[RoleUsage], *, case: str = "tc-1", rubric: bool = True, external_rates: bool = False
) -> EvalResult:
    roles = ["candidate", "inner_agent", "judge", "simulator", *(["external"] if external_rates else [])]
    return make_eval_result(
        id=f"res-{case}",
        scope_id=_SCOPE,
        eval_run_id=_RUN,
        test_case_id=case,
        usage=usage,
        cost_roles=roles,
        cost_usd=blended_cost(usage, roles),
        goal_state_outcomes=[],
        **({} if rubric else {"rubric_scores": []}),
    )


def _summary(results: list[EvalResult]):  # type: ignore[no-untyped-def]
    storage = EvalStorage(InMemoryDocumentStore())
    run = make_eval_run(
        id=_RUN,
        scope_id=_SCOPE,
        k_runs=1,
        test_case_ids=sorted({result.test_case_id for result in results}),
        status="completed",
    )
    storage.save_eval_run(run)
    for result in results:
        storage.save_eval_result(result)
    return summarize_run(dataclasses.replace(callable_host(), storage=storage), _RUN, _SCOPE)


def test_every_role_that_spent_is_stated_with_a_total_equal_to_the_results() -> None:
    summary = _summary([_result(_PERSONA_USAGE)])
    by_role = {spend.role: spend for spend in summary.spend}
    assert list(by_role) == ["candidate", "judge", "inner_agent", "external"]
    assert by_role["inner_agent"].cost_usd == pytest.approx(0.0166) and by_role["inner_agent"].calls == 5
    assert by_role["external"].cost_usd is None and not by_role["external"].counted
    assert [(units.provider, units.unit, units.units) for units in by_role["external"].units] == [
        ("tavily", "credits", 6)
    ]
    assert summary.total_cost_usd == pytest.approx(math.fsum([0.00348, 0.0166, 0.0137]))
    text = summary.render()
    assert "inner_agent spend: $0.0166 over 5 call(s)" in text
    assert (
        "external spend: 6 credits (tavily), unpriced over 3 call(s): the run declared no external rate, so its total "
        "sums no external dollars"
    ) in text
    assert "total spend: $0.0338, the results' cost" in text


def test_a_judge_campaigns_spend_is_stated_though_it_scored_no_rubric() -> None:
    usage = [RoleUsage(role="judge", model="m-judge", cost_usd=0.002, call_count=1, price_source="provider")]
    summary = _summary([_result(usage, case=f"tc-{i}", rubric=False) for i in range(3)])
    assert [spend.role for spend in summary.spend] == ["judge"]
    text = summary.render()
    assert "judge spend: $0.00600 over 3 call(s)" in text
    assert "the candidate reported none" not in text


def test_an_unpriced_role_makes_the_total_unknown_and_states_what_was_priced() -> None:
    usage = [
        RoleUsage(role="candidate", model="m-cand", cost_usd=0.001, call_count=1, price_source="provider"),
        RoleUsage(role="judge", model="m-judge", cost_usd=None, call_count=1),
    ]
    summary = _summary([_result(usage)])
    assert summary.total_cost_usd is None
    text = summary.render()
    assert "judge spend: unpriced over 1 call(s): its client reported no price for them" in text
    assert "total spend: unknown in full: $0.00100 over the priced calls, plus 1 call(s) unpriced" in text


def test_a_row_that_reported_no_call_count_is_never_read_as_zero_calls() -> None:
    usage = [RoleUsage(role="candidate", model="m-cand", cost_usd=0.002, price_source="provider")]
    summary = _summary([_result(usage, case=f"tc-{i}") for i in range(3)])
    (candidate,) = [spend for spend in summary.spend if spend.role == "candidate"]
    assert (candidate.calls, candidate.uncounted_rows) == (0, 3)
    text = summary.render()
    assert "candidate spend: $0.00600 over 3 usage row(s) that reported no call count" in text
    assert "over 0 call(s)" not in text


def test_counted_and_uncounted_rows_are_stated_side_by_side() -> None:
    counted = [RoleUsage(role="candidate", model="m-cand", cost_usd=0.002, call_count=2, price_source="provider")]
    uncounted = [RoleUsage(role="candidate", model="m-cand", cost_usd=0.002, price_source="provider")]
    summary = _summary([_result(counted, case="tc-0"), _result(uncounted, case="tc-1")])
    assert "candidate spend: $0.00400 over 2 call(s) and 1 usage row(s) that reported no call count" in (
        summary.render()
    )


def _counted_dollars(summary) -> float:  # type: ignore[no-untyped-def]
    return math.fsum(spend.cost_usd for spend in summary.spend if spend.counted and spend.cost_usd is not None)


def test_a_partly_priced_role_states_its_priced_dollars_and_its_unpriced_calls() -> None:
    """One unpriced judge call read the whole judge role as unknown, dropping the dollars its priced calls cost."""
    priced = [
        RoleUsage(role="candidate", model="m-cand", cost_usd=0.001, call_count=1, price_source="provider"),
        RoleUsage(role="judge", model="m-judge", cost_usd=0.002, call_count=1, price_source="provider"),
    ]
    unpriced = [
        RoleUsage(role="candidate", model="m-cand", cost_usd=0.001, call_count=1, price_source="provider"),
        RoleUsage(role="judge", model="m-local", cost_usd=None, call_count=1),
    ]
    summary = _summary([_result(priced, case="tc-1"), _result(unpriced, case="tc-2")])
    judge = summary.role_spend("judge")
    assert judge is not None and (judge.cost_usd, judge.unpriced_calls) == (pytest.approx(0.002), 1)
    text = summary.render()
    assert "judge spend: $0.00200 over 2 call(s), 1 of them unpriced: its client reported no price for them" in text
    assert "total spend: unknown in full: $0.00400 over the priced calls, plus 1 call(s) unpriced" in text
    assert _counted_dollars(summary) == pytest.approx(0.004)


def test_the_role_lines_add_up_to_the_total_where_external_calls_are_partly_priced() -> None:
    """A run with external rates sums its priced external calls; one unpriced call there leaves the total known.

    The blended total skips an external row that carries no dollars (a call the candidate's own tools made is volume,
    never priced), so the external role is stated as its priced dollars plus its unpriced calls, and the counted
    roles' dollars are the total, where the role read wholly unknown beside a total that counted its priced calls.
    """
    usage = [
        RoleUsage(role="candidate", model="m-cand", cost_usd=0.001, call_count=1, price_source="provider"),
        RoleUsage(
            role="external", provider="search", provider_unit="credits", provider_units=4, call_count=2, cost_usd=0.05
        ),
        RoleUsage(role="external", provider="maps", provider_unit="quota_units", provider_units=3, call_count=3),
    ]
    summary = _summary([_result(usage, external_rates=True)])
    external = summary.role_spend("external")
    assert external is not None and external.counted and (external.cost_usd, external.unpriced_calls) == (0.05, 3)
    assert summary.total_cost_usd == pytest.approx(0.051) == _counted_dollars(summary)
    text = summary.render()
    assert (
        "external spend: 3 quota_units (maps), 4 credits (search), $0.0500 over 5 call(s), 3 of them unpriced: neither "
        "a charge their provider reported nor a rate the run declared priced them"
    ) in text
    assert "total spend: $0.0510, the results' cost" in text


def test_a_role_outside_the_total_says_so() -> None:
    """A run with no external rate sums no external dollars, even a charge a provider reported on its row."""
    usage = [
        RoleUsage(role="candidate", model="m-cand", cost_usd=0.001, call_count=1, price_source="provider"),
        RoleUsage(
            role="external", provider="search", provider_unit="credits", provider_units=4, call_count=2, cost_usd=0.05
        ),
    ]
    summary = _summary([_result(usage)])
    assert summary.total_cost_usd == pytest.approx(0.001) == _counted_dollars(summary)
    assert "external spend: 4 credits (search), $0.0500 over 2 call(s); not in the total" in summary.render()


def test_a_run_that_reported_no_spend_states_none() -> None:
    summary = _summary([_result([], rubric=False)])
    assert summary.spend == [] and summary.spend_lines == [] and summary.total_spend_text is None

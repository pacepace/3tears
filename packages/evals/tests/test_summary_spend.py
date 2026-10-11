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


def _result(usage: list[RoleUsage], *, case: str = "tc-1", rubric: bool = True) -> EvalResult:
    roles = ["candidate", "inner_agent", "judge", "simulator"]
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
    assert by_role["external"].cost_usd is None
    assert [(units.provider, units.unit, units.units) for units in by_role["external"].units] == [
        ("tavily", "credits", 6)
    ]
    assert summary.total_cost_usd == pytest.approx(math.fsum([0.00348, 0.0166, 0.0137]))
    text = summary.render()
    assert "inner_agent spend: $0.0166 over 5 call(s)" in text
    assert "external spend: 6 credits (tavily) over 3 call(s), unpriced: no rate was declared for them" in text
    assert "total spend: $0.0338, the results' cost" in text


def test_a_judge_campaigns_spend_is_stated_though_it_scored_no_rubric() -> None:
    usage = [RoleUsage(role="judge", model="m-judge", cost_usd=0.002, call_count=1, price_source="provider")]
    summary = _summary([_result(usage, case=f"tc-{i}", rubric=False) for i in range(3)])
    assert [spend.role for spend in summary.spend] == ["judge"]
    text = summary.render()
    assert "judge spend: $0.00600 over 3 call(s)" in text
    assert "the candidate reported none" not in text


def test_an_unpriced_role_makes_the_total_unknown_never_a_partial_sum() -> None:
    usage = [
        RoleUsage(role="candidate", model="m-cand", cost_usd=0.001, call_count=1, price_source="provider"),
        RoleUsage(role="judge", model="m-judge", cost_usd=None, call_count=1),
    ]
    summary = _summary([_result(usage)])
    assert summary.total_cost_usd is None
    assert "total spend: unknown: a call went unpriced" in summary.render()


def test_a_run_that_reported_no_spend_states_none() -> None:
    summary = _summary([_result([], rubric=False)])
    assert summary.spend == [] and summary.spend_lines == [] and summary.total_spend_text is None

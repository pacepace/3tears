"""A launch estimate for a template with no priced history borrows from its matching siblings, and says it did (#948).

``history_launch_pricer`` prices an arm from its own template's priced results, then its earlier revisions'. A new
template has neither, and read ``unpriced`` even while a sibling of the same kind had been priced many times on the
same subject, model, judges, simulator and apparatus: a host then priced it from an unrelated battery, ten times too
high. The pricer now borrows from such a sibling, labelled ``borrowed-history`` with a wider band, and stays unpriced,
naming what is missing, when no sibling matches exactly. A borrowed figure budgets a launch and nothing else.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis.lenses.cost_estimate import (
    BORROWED_PREDICTION_METHOD,
    COST_PREDICTION_METHOD,
    CostObservation,
    compute_borrowed_cost,
)
from threetears.evals.kernel import EvalStorage
from threetears.evals.ops import history_launch_pricer
from threetears.evals.run import ArmPrice, ArmQuote
from threetears.evals.schema.models import EvalTemplate
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import DEFAULT_SUBJECT, make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.launch import toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import toyhost_template

MODEL = "m-priced"

#: What each of a sibling's six cases costs an observation, its two repeats a little either side: persona-like work,
#: cheap and varying by case more than by repeat.
SIBLING_CASES = [0.0040, 0.0050, 0.0060, 0.0070, 0.0055, 0.0065]

#: What the new template's own cases turn out to cost once it runs: other cases of the same work.
LATER_CASES = [0.0045, 0.0052, 0.0061, 0.0068, 0.0058, 0.0049, 0.0063, 0.0057, 0.0050, 0.0066]


def _priced_host() -> tuple[Any, EvalStorage]:
    storage = EvalStorage(InMemoryDocumentStore())
    host, _client = toyhost_launch_host(storage=storage)
    return host.eval_host, storage


def _template(storage: EvalStorage, template_id: str, **changes: Any) -> EvalTemplate:
    template = toyhost_template().model_copy(update={"id": template_id, "name": template_id, **changes})
    storage.save_template(template)
    return template


def _ran(
    storage: EvalStorage, template_id: str, case_costs: list[float], *, repeats: int = 2, **run_fields: Any
) -> None:
    """A finished run of ``template_id`` whose every case was measured ``repeats`` times, each repeat its case's cost
    moved a little either way."""
    fields: dict[str, Any] = {"status": "completed", "subject_snapshot": TOYHOST_SUBJECT, **run_fields}
    run = make_eval_run(scope_id=TOYHOST_SCOPE, template_id=template_id, candidate_model=MODEL, **fields)
    storage.save_eval_run(run)
    for case, cost in enumerate(case_costs):
        for k in range(1, repeats + 1):
            storage.save_eval_result(
                make_eval_result(
                    scope_id=TOYHOST_SCOPE,
                    eval_run_id=run.id,
                    model=MODEL,
                    test_case_id=f"{template_id}-case-{case}",
                    k_iteration=k,
                    cost_usd=cost * (1 + (0.04 if k % 2 else -0.04)),
                )
            )


def _quote(template_id: str, **overrides: Any) -> ArmQuote:
    fields: dict[str, Any] = {
        "scope_id": TOYHOST_SCOPE,
        "template_id": template_id,
        "subject_id": TOYHOST_SUBJECT.subject_id,
        "candidate_model": MODEL,
        "k_runs": 2,
        "case_count": len(LATER_CASES),
        "n_variations": 0,
        "cassette_mode": "off",
        "judge": None,
        "simulator_model": None,
        "apparatus_settings": {},
        **overrides,
    }
    return ArmQuote(**fields)


def test_a_template_with_no_history_borrows_from_a_matching_sibling_says_so_and_its_later_run_lands_in_the_band():
    host, storage = _priced_host()
    sibling = _template(storage, "dj_character_v1")
    new = _template(storage, "dj_character_v2")
    _ran(storage, sibling.id, SIBLING_CASES)

    price = history_launch_pricer(host)(_quote(new.id))

    assert price.method_id == BORROWED_PREDICTION_METHOD
    assert price.predicted_usd is not None and price.low_usd is not None and price.band_high_usd is not None
    assert price.basis.startswith(
        f"borrowed from template {sibling.name!r} ({sibling.id!r}), {2 * len(SIBLING_CASES)} result(s): template "
        f"{new.name!r} has no priced result of its own; the same candidate kind, subject, model, judge"
    ), price.basis
    assert "a budgeting figure only, never a reading" in price.basis
    assert "prediction band, borrowed" in price.basis and "wider than a band from the template's own" in price.basis

    # Once the new template runs, its own total lands inside the borrowed band, and its own history then prices it.
    _ran(storage, new.id, LATER_CASES)
    later = sum(cost * (1 + (0.04 if k % 2 else -0.04)) for cost in LATER_CASES for k in (1, 2))
    assert price.low_usd <= later <= price.band_high_usd, (price.low_usd, later, price.band_high_usd)
    assert history_launch_pricer(host)(_quote(new.id)).method_id == COST_PREDICTION_METHOD


def test_a_borrowed_band_is_wider_than_the_band_the_same_results_give_as_the_templates_own():
    host, storage = _priced_host()
    sibling, new = _template(storage, "persona_a"), _template(storage, "persona_b")
    _ran(storage, sibling.id, SIBLING_CASES)
    pricer = history_launch_pricer(host)

    borrowed, direct = pricer(_quote(new.id)), pricer(_quote(sibling.id))

    assert direct.method_id == COST_PREDICTION_METHOD
    assert borrowed.central_usd == pytest.approx(direct.central_usd), "one point estimate from one set of results"
    assert borrowed.low_usd is not None and direct.low_usd is not None
    assert borrowed.band_high_usd is not None and direct.band_high_usd is not None
    assert borrowed.low_usd <= direct.low_usd and borrowed.band_high_usd > direct.band_high_usd


def _unpriced(price: ArmPrice, new: EvalTemplate) -> str:
    assert price.predicted_usd is None and price.method_id is None
    assert f"no priced past result of template {new.id!r}" in price.basis, "what its own history lacks"
    return price.basis


def test_a_template_with_no_sibling_reads_unpriced_saying_neither_it_nor_a_sibling_has_priced_history():
    host, storage = _priced_host()
    new = _template(storage, "persona_b")

    basis = _unpriced(history_launch_pricer(host)(_quote(new.id)), new)

    assert f"nor has any other template of kind {new.candidate_kind!r} a priced result on subject " in basis


@pytest.mark.parametrize(
    ("model", "run_fields", "named"),
    [
        ("m-other", {}, "candidate model ('m-other' there, 'm-priced' here)"),
        (MODEL, {"judge_model": "judge-cheap", "model_role_provenance": {"judge": "chosen"}}, "judge ("),
        (MODEL, {"subject_snapshot": DEFAULT_SUBJECT}, "subject ('subject-1' there, 'extractor-config-7' here)"),
    ],
    ids=["another-model", "another-judge", "another-subject"],
)
def test_a_sibling_differing_in_a_launch_condition_is_not_drawn_on_and_is_named(
    model: str, run_fields: dict[str, Any], named: str
):
    host, storage = _priced_host()
    sibling, new = _template(storage, "persona_a"), _template(storage, "persona_b")
    _ran_on(storage, sibling.id, model, **run_fields)

    basis = _unpriced(history_launch_pricer(host)(_quote(new.id)), new)

    assert f"template {sibling.name!r} ({sibling.id!r})'s priced runs differ in {named}" in basis, basis


def _ran_on(storage: EvalStorage, template_id: str, model: str, **run_fields: Any) -> None:
    """A sibling's finished run on ``model``, launched as ``run_fields`` say and otherwise as the arm will be."""
    fields: dict[str, Any] = {"status": "completed", "subject_snapshot": TOYHOST_SUBJECT, **run_fields}
    run = make_eval_run(scope_id=TOYHOST_SCOPE, template_id=template_id, candidate_model=model, **fields)
    storage.save_eval_run(run)
    for case, cost in enumerate(SIBLING_CASES):
        storage.save_eval_result(
            make_eval_result(
                scope_id=TOYHOST_SCOPE, eval_run_id=run.id, model=model, test_case_id=f"case-{case}", cost_usd=cost
            )
        )


def test_a_sibling_whose_template_spends_otherwise_is_not_drawn_on_and_says_what_differs():
    host, storage = _priced_host()
    sibling = _template(storage, "persona_a", tools_allowed=["search"])
    new = _template(storage, "persona_b", tools_allowed=["search", "browse"])
    _ran(storage, sibling.id, SIBLING_CASES)

    basis = _unpriced(history_launch_pricer(host)(_quote(new.id)), new)

    assert (
        f"template {sibling.name!r} ({sibling.id!r}) differs from the arm's in the allowed tools, across which a "
        "case's cost cannot be scaled"
    ) in basis


def test_a_sibling_template_of_another_kind_is_never_drawn_on():
    host, storage = _priced_host()
    sibling = _template(storage, "persona_a", candidate_kind="another-kind")
    new = _template(storage, "persona_b")
    _ran(storage, sibling.id, SIBLING_CASES)

    basis = _unpriced(history_launch_pricer(host)(_quote(new.id)), new)

    assert sibling.id not in basis


def test_sibling_results_over_too_few_cases_bound_nothing():
    """Six repeats of one case say how that case costs, not how a new case will."""
    host, storage = _priced_host()
    sibling, new = _template(storage, "persona_a"), _template(storage, "persona_b")
    _ran(storage, sibling.id, [0.005], repeats=6)

    basis = _unpriced(history_launch_pricer(host)(_quote(new.id)), new)

    assert "cover 1 case(s), too few to bound an arm whose cases are new" in basis, basis


def test_a_template_with_priced_results_of_its_own_never_borrows_however_few():
    host, storage = _priced_host()
    sibling, new = _template(storage, "persona_a"), _template(storage, "persona_b")
    _ran(storage, sibling.id, SIBLING_CASES)
    _ran(storage, new.id, [0.005])

    price = history_launch_pricer(host)(_quote(new.id))

    assert price.predicted_usd is None and "too few to bound the arm" in price.basis
    assert "borrowed" not in price.basis


def _observations(costs_by_case: dict[str, list[float | None]]) -> list[CostObservation]:
    return [
        CostObservation(eval_run_id="run", model=MODEL, cost_usd=cost, cost_roles=(), test_case_id=case)
        for case, costs in costs_by_case.items()
        for cost in costs
    ]


def test_a_borrowed_estimate_brackets_its_point_estimate_and_counts_what_went_unpriced():
    borrowed = compute_borrowed_cost(
        _observations({"a": [0.004, 0.0042, None], "b": [0.006, 0.0058], "c": [0.005, 0.0051]}),
        n_cases=10,
        k_runs=2,
        computed_at="2026-11-14T00:00:00+00:00",
    )

    assert borrowed is not None
    predicted = borrowed.predicted
    assert (borrowed.n_historical, borrowed.n_cases, borrowed.n_unpriced_historical) == (6, 3, 1)
    assert predicted.value == pytest.approx(20 * borrowed.mean_cost_per_observation)
    assert predicted.interval_low is not None and predicted.interval_high is not None
    assert predicted.interval_low <= predicted.value <= predicted.interval_high
    assert predicted.method_id == BORROWED_PREDICTION_METHOD


def test_nothing_priced_borrows_nothing():
    assert compute_borrowed_cost(_observations({"a": [None, None]}), n_cases=3, k_runs=1) is None

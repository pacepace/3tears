"""Looks: how often a contrast's test was read after its data changed, counted from what is stored, and declared.

A look is one more chance for a test with no true difference behind it to separate (judges design §6). Held here:

- **What counts** (:func:`~threetears.evals.analysis.significance.looks_taken`): the first look is the moment both arms
  held a result; each launch whose first result came after it is one more (a step that adds cases and reads again is
  one), so is each change made after it (a re-measure finished, a blind read closed), and so is each re-judge or judge
  repeat pass. Two arms launched alone, one after the other, are one look, and so is one launch of both.
- **What the bundle says** (``reading_scope.looks``): the most looks any contrast of the aspect had, the looks the
  design planned, and whether the campaign was created once looks were counted (``EvalCampaign.looks_rule``, stamped
  at creation, never taken from a caller).
- **What a plan changes**: the aspect's family reads at α/L (``ComparisonFamily.alpha``), a floor's ``held`` on the
  interval at ``1 − (1 − level)/L``, and a campaign's gains and drops at α over the largest plan; with no plan nothing
  moves.
- **What a reader of a campaign that took an undeclared second look is told**, as typed values: the aspect's looks
  ``{planned: None, taken: 2}`` and one caveat of kind ``looks``, class D, on the aspect.

What the plan buys, and what looking without one costs, is measured in ``test_simulated_looks.py``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle, campaigns
from threetears.evals.analysis.campaign_bundle import build_campaign_bundle
from threetears.evals.analysis.campaign_readings import campaign_readings
from threetears.evals.analysis.checks import Looks
from threetears.evals.analysis.floors import FloorReadings, RateReading, read_floor
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.significance import (
    UNPLANNED_FALSE_SEPARATION,
    looks_alpha,
    looks_confidence,
    looks_taken,
)
from threetears.evals.analysis.stats import INTERVAL_LEVEL, SIGNIFICANCE_ALPHA, bounded_difference_interval
from threetears.evals.kernel import CampaignArm
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import Floor, RateDrop
from threetears.evals.schema import EvalResult
from threetears.evals.schema.models import EvalRun, JudgeRescore
from packages.evals.tests.factories import (
    fixture_variant_key,
    make_aspect_campaign,
    make_eval_result,
    make_eval_run,
    memory_storage,
    minimal_aspect_design,
)
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.simulation_support import family_verdicts

_START = datetime(2026, 11, 1, 9, 0, tzinfo=UTC)
_CONTROL, _CANDIDATE = "control-model", "candidate-model"
_ASPECT = "b"


def _at(minutes: float) -> datetime:
    return _START + timedelta(minutes=minutes)


# --- what counts -------------------------------------------------------------------------------------------------------


def test_one_launch_of_both_arms_is_one_look() -> None:
    assert looks_taken({"a": {"g1": _at(0)}, "b": {"g1": _at(1)}}) == 1


def test_two_arms_launched_alone_one_after_the_other_are_one_look() -> None:
    """Nothing could be read across the two arms until the second held a result: that moment is the first look."""
    assert looks_taken({"a": {"run-a": _at(0)}, "b": {"run-b": _at(90)}}) == 1


def test_a_launch_that_adds_cases_after_both_arms_held_a_result_is_one_more_look() -> None:
    """The step a remedy proposes (add cases, read again) launches runs: each such launch is a look."""
    first = {"a": {"g1": _at(0)}, "b": {"g1": _at(0)}}
    assert looks_taken({arm: {**held, "g2": _at(60)} for arm, held in first.items()}) == 2
    assert looks_taken({arm: {**held, "g2": _at(60), "g3": _at(120)} for arm, held in first.items()}) == 3
    # One arm alone gets more cases: still a launch whose data the contrast reads.
    assert looks_taken({"a": {"g1": _at(0), "run-more": _at(60)}, "b": {"g1": _at(0)}}) == 2


def test_changes_after_the_first_look_count_and_changes_before_it_do_not() -> None:
    batches = {"a": {"run-a": _at(0)}, "b": {"run-b": _at(30)}}
    assert looks_taken(batches, changes=[_at(10)]) == 1, "a re-measure before both arms held a result changes no look"
    assert looks_taken(batches, changes=[_at(40), _at(50)]) == 3, "each re-measure or blind read closed after is one"


def test_each_re_judge_or_repeat_pass_is_one_more_look() -> None:
    assert looks_taken({"a": {"g1": _at(0)}, "b": {"g1": _at(0)}}, passes=2) == 3


def test_with_fewer_than_two_arms_holding_a_result_there_is_no_contrast_to_look_at() -> None:
    assert looks_taken({"a": {"g1": _at(0)}}) is None
    assert looks_taken({"a": {"g1": _at(0)}, "b": {}}) is None


def test_a_plan_reads_each_look_at_alpha_over_its_looks_and_a_held_claim_at_its_tail_over_them() -> None:
    assert looks_alpha(None) == looks_alpha(1) == SIGNIFICANCE_ALPHA
    assert looks_alpha(4) == pytest.approx(SIGNIFICANCE_ALPHA / 4)
    assert looks_confidence(INTERVAL_LEVEL, None) == looks_confidence(INTERVAL_LEVEL, 1) == INTERVAL_LEVEL
    assert looks_confidence(INTERVAL_LEVEL, 5) == pytest.approx(1 - (1 - INTERVAL_LEVEL) / 5)


def test_the_unplanned_rate_grows_with_every_look_and_starts_at_alpha() -> None:
    rates = [UNPLANNED_FALSE_SEPARATION[looks] for looks in sorted(UNPLANNED_FALSE_SEPARATION)]
    assert rates[0] == SIGNIFICANCE_ALPHA
    assert rates == sorted(rates) and len(set(rates)) == len(rates)


# --- what the bundle says ----------------------------------------------------------------------------------------------


def _bundle(
    launches: Sequence[tuple[str | None, float, range]],
    *,
    planned: int | None = None,
    stamped: bool = True,
    shift: float = 0.0,
    seed: str = "looks",
    rescores: int = 0,
) -> tuple[AnalysisContextBundle, EvalCampaign, dict[str, dict[str, float]]]:
    """A two-arm campaign asked a question, its runs launched as ``launches`` says: each a launch group (None: each
    arm launched alone), when it started in minutes, and the cases it ran. The candidate is shifted by ``shift`` on
    ``field_accuracy``. Returns the bundle, the campaign and each arm's per-case values as written."""
    rng = random.Random(seed)
    runs: list[EvalRun] = []
    results: dict[str, list[EvalResult]] = {}
    written: dict[str, dict[str, float]] = {_CONTROL: {}, _CANDIDATE: {}}
    levels: dict[int, float] = {}
    for index, (group, start, cases) in enumerate(launches):
        for side, model in enumerate((_CONTROL, _CANDIDATE)):
            began = _at(start + (0 if group is not None else side * 5))
            run = make_eval_run(
                id=f"run-{index}-{model}",
                status="completed",
                candidate_model=model,
                launch_group_id=group,
                created_at=began.isoformat(),
            )
            runs.append(run)
            members = []
            for case_index in cases:
                case = f"tc-{case_index:03d}"
                level = levels.setdefault(case_index, rng.gauss(0.0, 0.08))
                value = min(1.0, max(0.0, 0.6 + level + rng.gauss(0.0, 0.05) + (shift if side else 0.0)))
                written[model][case] = round(value, 6)
                members.append(
                    make_eval_result(
                        id=f"{run.id}-{case}",
                        eval_run_id=run.id,
                        scope_id=run.scope_id,
                        model=model,
                        test_case_id=case,
                        goal_state_outcomes=[],
                        cost_usd=None,
                        rubric_scores=[],
                        host_measures={"field_accuracy": round(value, 6), "fields_stripped": 5.0},
                        scored_at=(began + timedelta(seconds=case_index)).isoformat(),
                        judge_rescores=[
                            JudgeRescore(dims=["extract.d"], prior_judge_error="cut off", judge_model="judge", cost_usd=None)
                            for _ in range(rescores if side == 0 and case_index == cases[0] else 0)
                        ],
                    )
                )
            results[run.id] = members
    campaign = make_aspect_campaign(
        label=_ASPECT,
        run_ids=[run.id for run in runs],
        design=minimal_aspect_design(merit_axes=["quality"], planned_looks=planned),
        question="is the candidate better?",
        asked_at=(_START - timedelta(days=1)).isoformat(),
        control=CampaignArm(label="control", role="control", variant_key=fixture_variant_key(_CONTROL)),
        scope_id=runs[0].scope_id,
        name="looks",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
        looks_rule=1 if stamped else None,
    )
    bundle = assemble_context_bundle(campaign, storage=ToyhostStorage(runs, results), profile=toyhost_profile())
    return bundle, campaign, written


def test_a_campaign_launched_once_reads_one_look_and_none_planned() -> None:
    bundle, _, _ = _bundle([("g1", 0, range(20))])
    assert bundle.reading_scope.looks == Looks(planned=None, taken=1)


def test_a_campaign_that_added_cases_and_read_again_reads_two_looks() -> None:
    bundle, _, _ = _bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))])
    assert bundle.reading_scope.looks == Looks(planned=None, taken=2)


def test_more_cases_launched_alone_on_each_arm_are_a_look_each_an_upper_bound() -> None:
    """Each arm's first run, launched alone, makes the first look; each later run launched alone is a launch of its own,
    so more cases run on the two arms in two calls count two looks, whether or not anyone read between them."""
    bundle, _, _ = _bundle([(None, 0, range(20)), (None, 120, range(20, 40))])
    assert bundle.reading_scope.looks == Looks(planned=None, taken=3)


def test_a_re_judge_of_the_contrast_s_runs_is_one_more_look() -> None:
    bundle, _, _ = _bundle([("g1", 0, range(20))], rescores=2)
    assert bundle.reading_scope.looks == Looks(planned=None, taken=3)


def test_the_plan_is_carried_beside_the_looks_taken() -> None:
    bundle, _, _ = _bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))], planned=3)
    assert bundle.reading_scope.looks == Looks(planned=3, taken=2)


def test_a_campaign_created_before_looks_were_counted_says_its_looks_could_not_have_been_planned() -> None:
    bundle, _, _ = _bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))], stamped=False)
    assert bundle.reading_scope.looks == Looks(planned=None, taken=2, plannable=False)
    assert bundle.reading_scope.looks.model_dump() == {"planned": None, "taken": 2, "plannable": False}
    stamped, _, _ = _bundle([("g1", 0, range(20))])
    assert "plannable" not in stamped.to_dict()["reading_scope"]["looks"], "a stamped campaign's looks say no more"


def test_the_engine_stamps_a_campaign_it_creates_and_never_takes_the_stamp_from_a_caller() -> None:
    storage, _ = memory_storage()
    storage.save_eval_run(make_eval_run(id="run-1", scope_id="scope-a"))
    definition = {
        "name": "c",
        "subject_id": "subject-1",
        "behavior": "conversation",
        "aspects": [{"label": "conversation", "run_ids": ["run-1"]}],
        "looks_rule": None,
    }
    created = campaigns.create_campaign(
        storage, definition, scope_id="scope-a", created_by="op", profile=toyhost_profile(), exploratory=True
    )
    assert created.looks_rule == 1
    assert storage.load_campaign(created.id, "scope-a") == created
    assert "looks_rule" in campaigns.CAMPAIGN_SERVER_FIELDS


def test_a_stored_campaign_without_the_stamp_reads_as_created_before_and_stores_as_it_was() -> None:
    old = make_aspect_campaign(label=_ASPECT)
    assert old.looks_rule is None
    assert "looks_rule" not in old.model_dump(mode="json")


# --- what a plan changes -----------------------------------------------------------------------------------------------


def _accuracy_family(bundle: AnalysisContextBundle):  # type: ignore[no-untyped-def]  # the family's shape
    (family,) = bundle.multiple_comparisons.families
    return family


@pytest.mark.parametrize("planned", [None, 1, 3])
def test_the_family_reads_its_verdicts_at_alpha_over_the_plan(planned: int | None) -> None:
    bundle, _, written = _bundle([("g1", 0, range(30))], planned=planned, shift=0.02, seed="family-0")
    family = _accuracy_family(bundle)
    alpha = looks_alpha(planned)
    assert family.alpha == pytest.approx(alpha)
    assert family.interval_level == pytest.approx(1 - alpha / family.family_size)
    accuracy = next(
        comparison
        for comparison in family.comparisons
        if comparison.name == "field_accuracy" and comparison.contrast.variant_key == fixture_variant_key(_CANDIDATE)
    )
    assert accuracy.p_adjusted is not None
    expected = family_verdicts(
        [(written[_CONTROL], written[_CANDIDATE], True), ({"x": 0.0, "y": 0.0}, {"x": 0.0, "y": 0.0}, True)],
        value_ranges=[(0.0, 1.0), None],
        alpha=alpha,
    )[0]
    assert accuracy.p_raw == pytest.approx(expected.p_raw)


def test_a_p_between_alpha_over_three_and_alpha_separates_with_no_plan_and_not_under_a_plan_of_three() -> None:
    """The branch the plan exists for: one seeded contrast whose adjusted p falls between α/3 and α."""
    unplanned, _, _ = _bundle([("g1", 0, range(30))], shift=0.03, seed="between-0")
    planned, _, _ = _bundle([("g1", 0, range(30))], planned=3, shift=0.03, seed="between-0")

    def accuracy(bundle: AnalysisContextBundle):  # type: ignore[no-untyped-def]  # a family comparison
        return next(
            comparison for comparison in _accuracy_family(bundle).comparisons if comparison.name == "field_accuracy"
        )

    p = accuracy(unplanned).p_adjusted
    assert p is not None and SIGNIFICANCE_ALPHA / 3 < p < SIGNIFICANCE_ALPHA, f"the seed drifted off its branch: p={p}"
    assert accuracy(unplanned).verdict == "improved"
    assert accuracy(planned).verdict == "not_separated"


def test_a_floor_held_at_the_nominal_level_is_not_held_under_a_plan_whose_tighter_interval_reaches_past_it() -> None:
    rng = random.Random("floor-held")
    control = tuple(round(min(1.0, 0.9 + rng.gauss(0, 0.05)), 3) for _ in range(40))
    candidate = tuple(round(min(1.0, max(0.0, value - 0.01 + rng.gauss(0, 0.03))), 3) for value in control)

    def worst(confidence: float) -> float:
        interval = bounded_difference_interval(
            control, candidate, paired=True, value_range=(0.0, 1.0), confidence=confidence
        )
        assert interval is not None
        return -100 * interval[0]

    def outcome(looks: int | None, margin: float) -> str:
        rate = RateReading(
            "the pass rate",
            round(sum(control) * 10),
            400,
            round(sum(candidate) * 10),
            400,
            interval=bounded_difference_interval(
                control, candidate, paired=True, value_range=(0.0, 1.0), confidence=INTERVAL_LEVEL
            ),
            samples=(control, candidate),
            confidence=INTERVAL_LEVEL,
            looks=looks,
        )
        readings = FloorReadings(rates=(rate,), control_cases=40, candidate_cases=40)
        floor = Floor(aspect=_ASPECT, clause=RateDrop(max_points=margin))
        return read_floor(floor, readings, control="a", candidate="b").outcome

    nominal, planned = worst(INTERVAL_LEVEL), worst(looks_confidence(INTERVAL_LEVEL, 5))
    assert 0 < nominal < planned
    between = (nominal + planned) / 2
    assert outcome(None, between) == "held"
    assert outcome(5, between) == "unclear", "the claim of no worse is read on the interval the plan tightens"
    assert outcome(5, planned + 0.5) == "held"


def test_a_campaign_s_gains_read_at_alpha_over_the_largest_plan_among_its_aspects() -> None:
    for planned in (None, 4):
        bundle, campaign, _ = _bundle([("g1", 0, range(20))], planned=planned)
        readings = campaign_readings(campaign, {_ASPECT: bundle}, arms=campaign.arms + [_candidate_arm()])
        assert readings.alpha == pytest.approx(looks_alpha(planned))


def _candidate_arm() -> CampaignArm:
    return CampaignArm(label="candidate", role="candidate", variant_key=fixture_variant_key(_CANDIDATE))


# --- what a reader of an undeclared second look is told ----------------------------------------------------------------


def test_a_reader_of_a_campaign_that_took_an_undeclared_second_look_is_told_two_looks_none_planned_and_one_caveat() -> (
    None
):
    bundle, campaign, _ = _bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))])
    arms = [*campaign.arms, _candidate_arm()]
    readings = campaign_readings(campaign, {_ASPECT: bundle}, arms=arms)
    built = report_input(build_campaign_bundle(campaign, [bundle], arms=arms, readings=readings))
    (aspect,) = built.aspects
    assert aspect.looks == Looks(planned=None, taken=2)
    looks = [entry for entry in built.caveat_list or [] if entry.kind == "looks"]
    assert [(entry.id, entry.cls, entry.reach) for entry in looks] == [(f"looks:{_ASPECT}", "D", "report")]


def test_a_reader_of_looks_within_the_plan_is_told_the_looks_and_no_caveat() -> None:
    bundle, campaign, _ = _bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))], planned=2)
    arms = [*campaign.arms, _candidate_arm()]
    readings = campaign_readings(campaign, {_ASPECT: bundle}, arms=arms)
    built = report_input(build_campaign_bundle(campaign, [bundle], arms=arms, readings=readings))
    assert built.aspects[0].looks == Looks(planned=2, taken=2)
    assert not [entry for entry in built.caveat_list or [] if entry.kind == "looks"]

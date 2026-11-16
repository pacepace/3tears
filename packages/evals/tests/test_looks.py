"""Looks: how often a contrast's test could have been read again on more or different evidence, and the plan in force.

A look is an occasion on which a contrast's test could have been read again on more or different evidence: one more
chance for a test with no true difference behind it to separate (judges design §6). Held here:

- **What counts** (:func:`~threetears.evals.analysis.significance.looks_taken`): the first look is the first moment
  both arms held a result; each later occasion that changed what the test reads is one more, counted once however many
  results it brought: a launch, a membership write (runs joining after their results were scored, runs leaving), a
  re-measure, a re-judge. More cases run on one arm and then on the other are one look, at the launch that paired them.
  Re-judges are stored per result, so their count is a lower bound. A judge repeat, a second judge's pass and a measure
  attestation change no value the test reads, and are not counted.
- **What the bundle says** (``reading_scope.looks``): the most looks any contrast of the aspect had, from the runs and
  the aspect's append-only membership record, so a run removed or deleted never erases the look it made; whether the
  count is a lower bound (a fact it needs is not stored); the plan in force, and a plan declared after the first look
  as ``late_plan`` and no plan.
- **What a plan changes**, through the bundle: the aspect's family reads each look at α/L and states α; a floor's
  ``held`` (rate and cost), a guardrail's ``held`` and a campaign row's ``within_margin`` read on the interval at
  ``1 − (1 − level)/L``, and the cases an unclear row projects are sized at that level. With no plan nothing moves.
- **What a reader of a campaign that took an undeclared second look is told**, as typed values: the aspect's looks
  ``{planned: None, taken: 2}`` and one caveat of kind ``looks``, class D, naming the aspect's claims and carrying the
  looks and the false-separation rate measured for them; at method reach where the aspect claims nothing or the
  campaign was created before looks were counted; raised under a plan whose keeping a lower-bound count cannot show.
- **The engine records membership** on create, add, remove and delete, and stamps a campaign it creates.

What the plan buys, and what looking without one costs, is measured in ``test_simulated_looks.py``.
"""

from __future__ import annotations

import random
from collections.abc import Collection, Sequence
from dataclasses import replace as dataclass_replace
from datetime import UTC, datetime, timedelta
from typing import Literal

import pytest

from threetears.evals.analysis import (
    AnalysisContextBundle,
    CampaignReport,
    assemble_context_bundle,
    build_campaign_report,
    build_code_only_report,
    campaign_report_markdown,
    campaigns,
)
from threetears.evals.analysis.campaign_bundle import build_campaign_bundle
from threetears.evals.analysis.campaign_readings import CampaignReadings, campaign_readings
from threetears.evals.analysis.caveats import UnkeptPlan, caveat_list, unkept_plan
from threetears.evals.analysis.checks import Looks
from threetears.evals.analysis.floors import floor_readings
from threetears.evals.analysis.report.block_spec import block_look, complete_chart
from threetears.evals.analysis.report.charts import drawn_head
from threetears.evals.analysis.report.fact_arms import fact_arms
from threetears.evals.analysis.report.figure_keys import format_figure, resolve
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.keys.vocabulary import joint_level, looks_of, plan_unshown
from threetears.evals.analysis.report.model import DisclosureBlock, TableBlock
from threetears.evals.analysis.report.number_forms.intervals import interval_level
from threetears.evals.analysis.report.parts.full_results import every_table
from threetears.evals.analysis.report.parts.resolve import resolve_chart
from threetears.evals.analysis.report.parts.words import UNSHOWN_PLAN_LEVEL
from threetears.evals.analysis.report_input import CaveatEntry, FalseSeparation, ReportInput, report_input_digest
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.significance import (
    FALSE_SEPARATION,
    SeparationTest,
    look_sizes,
    looks_alpha,
    looks_confidence,
    looks_false_separation,
    looks_taken,
    measured_upper_bound,
    plan_kept,
)
from threetears.evals.analysis.stats import INTERVAL_LEVEL, SIGNIFICANCE_ALPHA, bounded_difference_interval
from threetears.evals.kernel import CampaignArm
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.chart_query import ChartQuery, PlannedChart
from threetears.evals.kernel.host import HostProfile, MeasureRegistry
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.kernel.declaration import CostLower, DecisionClause, MembershipChange, RateDrop
from threetears.evals.run.curation import delete_run
from threetears.evals.schema import EvalResult, RoleUsage
from threetears.evals.schema.models import EvalRun, GoalStateOutcome, JudgeRepeat, JudgeRescore, RepeatedScore
from packages.evals.tests.factories import (
    fixture_variant_key,
    make_aspect_campaign,
    make_eval_result,
    make_eval_run,
    memory_storage,
    minimal_aspect_design,
)
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.report_shapes_support import Shape, s1_change, s2_choice
from packages.evals.tests.simulation_support import family_verdicts

_START = datetime(2026, 11, 1, 9, 0, tzinfo=UTC)
_CONTROL, _CANDIDATE = "control-model", "candidate-model"
_ASPECT = "b"
_PLAN = 3

#: Which arms one launch runs.
Sides = Literal["both", "control", "candidate"]


def _at(minutes: float) -> datetime:
    return _START + timedelta(minutes=minutes)


def _stamp(minutes: float) -> str:
    return _at(minutes).isoformat()


# --- what counts -------------------------------------------------------------------------------------------------------


def _cases(arm: str, at: float, cases: range, occasion: str) -> list[tuple[datetime, str, str, str]]:
    """One launch's results on one arm, a minute apart."""
    return [(_at(at + index), arm, f"c{case}", occasion) for index, case in enumerate(cases)]


_AB = ("a", "b")


def test_one_launch_of_both_arms_is_one_look() -> None:
    assert looks_taken([*_cases("a", 0, range(20), "g1"), *_cases("b", 0, range(20), "g1")], arms=_AB) == (1, False)


def test_two_arms_launched_alone_one_after_the_other_are_one_look() -> None:
    """Nothing could be read across the two arms until the second held a result: that moment is the first look."""
    assert looks_taken([*_cases("a", 0, range(20), "run-a"), *_cases("b", 90, range(20), "run-b")], arms=_AB) == (
        1,
        False,
    )


def test_a_launch_that_adds_cases_after_the_first_look_is_one_more_look() -> None:
    """The step a remedy proposes (add cases, read again) launches runs: each such launch is a look."""
    first = [*_cases("a", 0, range(20), "g1"), *_cases("b", 0, range(20), "g1")]
    second = [*_cases("a", 60, range(20, 40), "g2"), *_cases("b", 60, range(20, 40), "g2")]
    third = [*_cases("a", 120, range(40, 60), "g3"), *_cases("b", 120, range(40, 60), "g3")]
    assert looks_taken([*first, *second], arms=_AB) == (2, False)
    assert looks_taken([*first, *second, *third], arms=_AB) == (3, False)


def test_new_cases_run_on_one_arm_and_then_the_other_are_one_look() -> None:
    """The first launch's cases pair with nothing yet, so the test reads what it read; the second pairs them."""
    first = [*_cases("a", 0, range(20), "g1"), *_cases("b", 0, range(20), "g1")]
    one_arm = _cases("a", 60, range(20, 40), "run-a2")
    assert looks_taken([*first, *one_arm], arms=_AB) == (1, False), "cases one arm alone ran change no paired read"
    other_arm = _cases("b", 90, range(20, 40), "run-b2")
    assert looks_taken([*first, *one_arm, *other_arm], arms=_AB) == (2, False)


def test_more_trials_of_held_cases_on_one_arm_are_a_look() -> None:
    """A repeat of cases both arms hold changes what the paired test reads at once."""
    first = [*_cases("a", 0, range(20), "g1"), *_cases("b", 0, range(20), "g1")]
    assert looks_taken([*first, *_cases("a", 60, range(20), "run-a2")], arms=_AB) == (2, False)


def test_changes_after_the_first_look_count_and_changes_before_it_do_not() -> None:
    deliveries = [*_cases("a", 0, range(20), "run-a"), *_cases("b", 30, range(20), "run-b")]
    assert looks_taken(deliveries, arms=_AB, changes=[(_at(10), "remeasure-1")]) == (1, False)
    assert looks_taken(deliveries, arms=_AB, changes=[(_at(60), "remeasure-1"), (_at(70), "remeasure-2")]) == (3, False)


def test_a_departed_run_s_look_is_counted_and_dates_the_first_look() -> None:
    """A swap: the first launch's runs left in the write that joined their replacements, which were scored earlier and
    joined then. The look at the first launch happened, and the write is one more."""
    replaced = [*_cases("a", 300, range(20), "join-1"), *_cases("b", 300, range(20), "join-1")]
    swap = [(_at(300), "join-1")]
    assert looks_taken(replaced, arms=_AB, changes=swap, departed=[(_at(0), "g1")]) == (2, False)
    assert looks_taken(replaced, arms=_AB, changes=swap) == (1, False), "with the departure unrecorded, it is erased"


def test_a_re_judge_pass_is_one_look_and_the_count_a_lower_bound() -> None:
    """One pass re-judging a result on each arm is one look; a result re-judged twice was in two passes at least."""
    deliveries = [*_cases("a", 0, range(20), "g1"), *_cases("b", 0, range(20), "g1")]
    assert looks_taken(deliveries, arms=_AB, rejudges=[[_at(100)], [_at(101)]]) == (2, True)
    assert looks_taken(deliveries, arms=_AB, rejudges=[[_at(100), _at(200)], [_at(101)]]) == (3, True)
    assert looks_taken(deliveries, arms=_AB, rejudges=[[_at(-5)]]) == (1, False), "before the first look: no look"


def test_with_fewer_than_two_arms_holding_a_result_there_is_no_contrast_to_look_at() -> None:
    assert looks_taken(_cases("a", 0, range(20), "g1"), arms=_AB) is None
    assert looks_taken([], arms=_AB) is None


def test_a_plan_reads_each_look_at_alpha_over_its_looks_and_a_held_claim_at_its_tail_over_them() -> None:
    assert looks_alpha(None) == looks_alpha(1) == SIGNIFICANCE_ALPHA
    assert looks_alpha(4) == pytest.approx(SIGNIFICANCE_ALPHA / 4)
    assert looks_confidence(INTERVAL_LEVEL, None) == looks_confidence(INTERVAL_LEVEL, 1) == INTERVAL_LEVEL
    assert looks_confidence(INTERVAL_LEVEL, 5) == pytest.approx(1 - (1 - INTERVAL_LEVEL) / 5)


# --- the bundle --------------------------------------------------------------------------------------------------------


def _bundle(
    launches: Sequence[tuple[str | None, float, range] | tuple[str | None, float, range, Sides]],
    *,
    planned: int | None = None,
    stamped: bool = True,
    shift: float = 0.0,
    seed: str = "looks",
    rescores: int = 0,
    repeats: int = 0,
    joined: dict[int, float] | None = None,
    recorded: bool = True,
    unrecorded: Collection[int] = (),
    left: Sequence[MembershipChange] = (),
    declared: float = -1440.0,
    pass_rates: tuple[float, float] | None = None,
    costs: tuple[float, float] | None = None,
    cost_noise: float = 0.0,
    margin: float | None = None,
    floors: Sequence[DecisionClause] = (),
    guardrail_margin: float | None = None,
    stripped: bool = True,
) -> tuple[AnalysisContextBundle, EvalCampaign, dict[str, dict[str, float]]]:
    """A two-arm campaign asked a question, its runs launched as ``launches`` says: each a launch group (None: each
    arm launched alone, the candidate five minutes after the control), when it started in minutes, the cases it ran,
    and which arms. The candidate is shifted by ``shift`` on ``field_accuracy``; each arm passes its one check at its
    ``pass_rates`` share and costs its ``costs`` per reply (plus noise). Each run joins the aspect a minute before it
    starts, or at ``joined[launch index]``, as the membership record says (none where not ``recorded``, nor for the
    launches in ``unrecorded``); ``left`` is
    appended. The design, its plan and the question are declared at ``declared``. The candidate makes up to one
    destructive call a case, read as a guardrail on its declared range at ``guardrail_margin``. Returns the bundle, the
    campaign and each arm's per-case accuracy as written."""
    rng = random.Random(seed)
    # Each other draw on a stream of its own, so the accuracy a seed was chosen for is drawn as it always was.
    passes, spend, calls = (random.Random(f"{seed}:{stream}") for stream in ("passes", "spend", "calls"))
    runs: list[EvalRun] = []
    results: dict[str, list[EvalResult]] = {}
    written: dict[str, dict[str, float]] = {_CONTROL: {}, _CANDIDATE: {}}
    levels: dict[int, float] = {}
    membership: list[MembershipChange] = []
    for index, launch in enumerate(launches):
        group, start, cases = launch[0], launch[1], launch[2]
        sides: Sides = launch[3] if len(launch) == 4 else "both"  # type: ignore[misc]  # the optional fourth
        for side, model in enumerate((_CONTROL, _CANDIDATE)):
            if sides != "both" and sides != ("control", "candidate")[side]:
                continue
            began = _at(start + (0 if group is not None else side * 5))
            run = make_eval_run(
                id=f"run-{index}-{model}",
                status="completed",
                candidate_model=model,
                launch_group_id=group,
                created_at=began.isoformat(),
            )
            runs.append(run)
            join = (joined or {}).get(index, start - 1)
            if index not in unrecorded:
                membership.append(MembershipChange(run_id=run.id, change="joined", at=_stamp(join)))
            members = []
            for case_index in cases:
                case = f"tc-{case_index:03d}"
                level = levels.setdefault(case_index, rng.gauss(0.0, 0.08))
                value = min(1.0, max(0.0, 0.6 + level + rng.gauss(0.0, 0.05) + (shift if side else 0.0)))
                written[model][case] = round(value, 6)
                destructive = float(side == 1 and calls.random() < 0.1)
                written.setdefault(f"{model}:destructive", {})[case] = destructive
                cost = None if costs is None else round(max(0.0, costs[side] + spend.gauss(0.0, cost_noise)), 7)
                members.append(
                    make_eval_result(
                        id=f"{run.id}-{case}",
                        eval_run_id=run.id,
                        scope_id=run.scope_id,
                        model=model,
                        test_case_id=case,
                        goal_state_outcomes=(
                            []
                            if pass_rates is None
                            else [GoalStateOutcome(expression="ok", passed=passes.random() < pass_rates[side])]
                        ),
                        cost_usd=cost,
                        usage=[] if cost is None else [RoleUsage(role="candidate", model=model, cost_usd=cost)],
                        rubric_scores=[],
                        host_measures={"field_accuracy": round(value, 6)}
                        | ({"fields_stripped": 5.0} if stripped else {})
                        | ({} if guardrail_margin is None else {"destructive_calls": destructive}),
                        scored_at=(began + timedelta(seconds=case_index)).isoformat(),
                        judge_rescores=[
                            JudgeRescore(
                                dims=["extract.d"],
                                prior_judge_error="cut off",
                                judge_model="judge",
                                cost_usd=None,
                                rejudged_at=_stamp(start + 600 + 60 * each),
                            )
                            for each in range(rescores if case_index == cases[0] else 0)
                        ],
                        judge_repeats=[
                            JudgeRepeat(
                                judge_model="judge",
                                scores=[
                                    RepeatedScore(
                                        dim="extract.d",
                                        scale="ordinal",
                                        first_score=3,
                                        first_served_model="judge",
                                        first_judge_config_id="judge-config",
                                        error="cut off",
                                    )
                                ],
                                repeated_at=_stamp(start + 700 + 60 * each),
                            )
                            for each in range(repeats if case_index == cases[0] else 0)
                        ],
                    )
                )
            results[run.id] = members
    campaign = make_aspect_campaign(
        label=_ASPECT,
        run_ids=[run.id for run in runs],
        design=minimal_aspect_design(merit_axes=["quality"], planned_looks=planned),
        question="is the candidate better?",
        asked_at=_stamp(declared),
        control=CampaignArm(label="control", role="control", variant_key=fixture_variant_key(_CONTROL)),
        margin=margin,
        floors=floors,
        scope_id=runs[0].scope_id,
        name="looks",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
        looks_rule=1 if stamped else None,
    )
    record = [*membership, *left] if recorded else list(left)
    campaign = campaign.model_copy(update={"aspects": [campaign.aspects[0].model_copy(update={"membership": record})]})
    profile = toyhost_profile() if guardrail_margin is None else _with_guardrail(guardrail_margin)
    bundle = assemble_context_bundle(campaign, storage=ToyhostStorage(runs, results), profile=profile)
    return bundle, campaign, written


def _with_guardrail(margin: float | None) -> HostProfile:
    """The toy host with a destructive-call count declared a guardrail (lower is better) on ``[0, 1]``."""
    profile = toyhost_profile()
    family = next(descriptor.family for descriptor in TOYHOST_MEASURES if descriptor.name == "field_accuracy")
    destructive = MetricDescriptor(
        name="destructive_calls",
        reader_name="Destructive calls",
        data_type="numeric",
        family=family,
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="Destructive tool calls the candidate made.",
        higher_is_better=False,
        materiality_threshold=margin,
        guardrail=True,
        value_range=(0.0, 1.0),
    )
    return dataclass_replace(
        profile, measures=MeasureRegistry((*TOYHOST_MEASURES, destructive), families=profile.measures.families)
    )


def _counted(bundle: AnalysisContextBundle) -> Looks | None:
    """The bundle's looks, without the tests its readings use and its sizes (held on their own below)."""
    looks = bundle.reading_scope.looks
    return None if looks is None else looks.model_copy(update={"tests": [], "launched": None, "largest_step": None})


def test_a_campaign_launched_once_with_no_plan_states_no_looks_and_is_written_as_before() -> None:
    """One look and no plan qualify no reading, so the bundle leaves its looks out, as every bundle did before looks
    were counted; with a plan stated, the one look is carried beside it."""
    bundle, _, _ = _bundle([("g1", 0, range(20))])
    assert bundle.reading_scope.looks is None
    assert "looks" not in bundle.to_dict()["reading_scope"]
    planned, _, _ = _bundle([("g1", 0, range(20))], planned=_PLAN)
    assert _counted(planned) == Looks(planned=_PLAN, taken=1)


def test_a_campaign_that_added_cases_and_read_again_reads_two_looks() -> None:
    bundle, _, _ = _bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))])
    assert _counted(bundle) == Looks(planned=None, taken=2)


def test_more_cases_launched_alone_on_each_arm_are_one_look() -> None:
    """One extend run arm by arm: the control's new cases pair with nothing until the candidate's run them."""
    bundle, _, _ = _bundle([(None, 0, range(20)), (None, 120, range(20, 40))])
    assert _counted(bundle) == Looks(planned=None, taken=2)


def test_one_re_judge_pass_over_both_arms_is_one_look_and_a_lower_bound() -> None:
    bundle, _, _ = _bundle([("g1", 0, range(20))], rescores=1)
    assert _counted(bundle) == Looks(planned=None, taken=2, lower_bound=True)


def test_judge_repeats_change_no_value_the_test_reads_and_are_no_look() -> None:
    bundle, _, _ = _bundle([("g1", 0, range(20))], repeats=2)
    assert bundle.reading_scope.looks is None, "one look, as with no repeat"


def test_an_old_run_joined_after_the_first_look_is_a_look_when_it_joins() -> None:
    """Control at 0, candidate at 20, read; then a control run scored at 12, on the cases both arms hold, joins at 300:
    one more look, though its results were scored before both arms first held one."""
    bundle, _, _ = _bundle(
        [("a", 0, range(20), "control"), ("b", 20, range(20), "candidate"), ("d", 12, range(20), "control")],
        joined={2: 300},
    )
    assert _counted(bundle) == Looks(planned=None, taken=2)


def test_a_run_removed_after_its_look_keeps_the_look_and_its_leaving_is_another() -> None:
    """The first launch's control run left at 300, recorded as it left: the look it made and its leaving both count."""
    gone = MembershipChange(run_id="gone", change="left", at=_stamp(300), first_result_at=_stamp(-60), launch="g0")
    kept, _, _ = _bundle([("g1", 0, range(20))], left=[gone])
    assert _counted(kept) == Looks(planned=None, taken=3)


def test_a_deleted_run_s_departure_is_counted_and_makes_the_count_a_lower_bound() -> None:
    gone = MembershipChange(run_id="gone", change="left", at=_stamp(300))
    bundle, _, _ = _bundle([("g1", 0, range(20))], left=[gone])
    assert _counted(bundle) == Looks(planned=None, taken=2, lower_bound=True)


def test_an_aspect_whose_members_joining_was_never_recorded_states_its_looks_were_not_recorded() -> None:
    """Joined before durable 17.3 recorded membership: no count of looks, as neither bound; the evidence changes as
    what they are."""
    bundle, _, _ = _bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))], recorded=False)
    assert _counted(bundle) == Looks(planned=None, recorded=False, evidence_changes=2)
    assert bundle.reading_scope.looks is not None and bundle.reading_scope.looks.taken is None
    dumped = _counted(bundle).model_dump()  # type: ignore[union-attr]  # counted above
    assert dumped == {"planned": None, "taken": None, "recorded": False, "evidence_changes": 2}


def test_a_campaign_extended_after_the_record_counts_the_looks_recorded_since_as_a_lower_count() -> None:
    """A campaign stored before the record and extended twice after it: the looks before are not recorded, and the
    first comparison and the two recorded launches since are three looks at least."""
    bundle, _, _ = _bundle(
        [("g1", 0, range(20)), ("g2", 120, range(20, 40)), ("g3", 240, range(40, 60))], unrecorded={0}
    )
    assert _counted(bundle) == Looks(planned=None, taken=3, lower_bound=True, recorded=False, evidence_changes=3)


def test_unrecorded_launches_after_the_first_comparison_are_not_counted_as_looks_since() -> None:
    """Two launches stored before the record and one recorded join since: the unrecorded second launch is no
    recorded look, so the count since is the first comparison and the recorded launch."""
    bundle, _, _ = _bundle(
        [("g1", 0, range(20)), ("g2", 120, range(20, 40)), ("g3", 240, range(40, 60))], unrecorded={0, 1}
    )
    assert _counted(bundle) == Looks(planned=None, taken=2, lower_bound=True, recorded=False, evidence_changes=3)


def test_an_aspect_carries_the_cases_its_first_look_read_and_its_largest_step() -> None:
    bundle, _, _ = _bundle([("g1", 0, range(20)), ("g2", 120, range(20, 50))])
    assert bundle.reading_scope.looks is not None
    assert (bundle.reading_scope.looks.launched, bundle.reading_scope.looks.largest_step) == (20, 30)
    assert look_sizes(
        [*_cases("a", 0, range(20), "run-a"), *_cases("b", 90, range(20), "run-b"), *_cases("a", 200, range(20), "r")],
        arms=_AB,
    ) == (20, 0), "arms launched alone are one launch; more trials of held cases add no cases"


def test_unrecorded_looks_carry_no_count_and_recorded_looks_no_evidence_changes() -> None:
    with pytest.raises(ValueError, match="unrecorded"):
        Looks(recorded=False, taken=2, evidence_changes=2)
    assert Looks(recorded=False, taken=2, lower_bound=True, evidence_changes=2).taken == 2, "a lower count since"
    with pytest.raises(ValueError, match="unrecorded"):
        Looks(recorded=False, evidence_changes=2, lower_bound=True)
    with pytest.raises(ValueError, match="recorded looks carry taken"):
        Looks(taken=2, evidence_changes=2)


def test_the_tests_an_aspect_s_readings_use_ride_with_its_looks() -> None:
    """Accuracy on the t test, the pass rate on the bounded one; a cost per reply on the t test."""
    rated, _, _ = _bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))], pass_rates=(0.8, 0.8))
    assert rated.reading_scope.looks is not None and rated.reading_scope.looks.tests == ["bounded", "t"]
    families = rated.multiple_comparisons.families
    bases = {comparison.interval_basis for family in families for comparison in family.comparisons}
    assert "t" in bases, "the accuracy comparison reads the t test"
    assert {contrast.reading for contrast in rated.floor_contrasts} >= {"trial"}, "the pass rate reads the bounded test"


def test_the_plan_in_force_is_carried_beside_the_looks_taken() -> None:
    bundle, _, _ = _bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))], planned=_PLAN)
    assert _counted(bundle) == Looks(planned=_PLAN, taken=2)


def test_a_plan_declared_after_the_first_look_is_no_plan_and_is_disclosed() -> None:
    """Runs scored, then a campaign declared over them with a plan: the looks it would count were already taken."""
    bundle, _, _ = _bundle([("g1", 0, range(30))], planned=_PLAN, declared=60, joined={0: 90}, shift=0.03)
    assert _counted(bundle) == Looks(planned=None, taken=1, late_plan=_PLAN)
    (family,) = bundle.multiple_comparisons.families
    assert family.look_alpha is None, "a late plan tightens nothing: the family reads at alpha"


def test_a_campaign_created_before_looks_were_counted_says_its_looks_could_not_have_been_planned() -> None:
    bundle, _, _ = _bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))], stamped=False)
    assert _counted(bundle) == Looks(planned=None, taken=2, plannable=False)
    assert _counted(bundle).model_dump() == {"planned": None, "taken": 2, "plannable": False}  # type: ignore[union-attr]
    stamped, _, _ = _bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))])
    assert "plannable" not in stamped.to_dict()["reading_scope"]["looks"], "a stamped campaign's looks say no more"


# --- the engine records membership -------------------------------------------------------------------------------------


def test_the_engine_stamps_a_campaign_it_creates_and_records_each_run_joining_and_leaving() -> None:
    storage, _ = memory_storage()
    for run_id in ("run-1", "run-2", "run-3"):
        storage.save_eval_run(make_eval_run(id=run_id, scope_id="scope-a", launch_group_id="g"))
    storage.save_eval_result(make_eval_result(id="r-2", eval_run_id="run-2", scope_id="scope-a", scored_at=_stamp(-30)))
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
    assert "looks_rule" in campaigns.CAMPAIGN_SERVER_FIELDS
    assert [(change.run_id, change.change) for change in created.aspects[0].membership] == [("run-1", "joined")]
    added = campaigns.add_runs_to_campaign(storage, created.id, "scope-a", ["run-2", "run-3"])
    removed = campaigns.remove_runs_from_campaign(storage, added.id, "scope-a", ["run-2"])
    record = removed.aspects[0].membership
    assert [(change.run_id, change.change) for change in record] == [
        ("run-1", "joined"),
        ("run-2", "joined"),
        ("run-3", "joined"),
        ("run-2", "left"),
    ]
    assert (record[-1].first_result_at, record[-1].launch) == (_stamp(-30), "g"), "the look it made is kept"
    assert record[1].at == record[2].at, "one write, one time"
    delete_run(storage, storage.load_eval_run("run-3", "scope-a"), "scope-a", confirm="run-3")  # type: ignore[arg-type]
    after = storage.load_campaign(created.id, "scope-a")
    assert after is not None and after.run_ids == ["run-1"]
    assert [(change.run_id, change.change) for change in after.aspects[0].membership][-1] == ("run-3", "left")


def test_a_stored_campaign_without_the_stamp_or_the_record_stores_as_it_was() -> None:
    old = make_aspect_campaign(label=_ASPECT)
    assert old.looks_rule is None
    dumped = old.model_dump(mode="json")
    assert "looks_rule" not in dumped and "membership" not in dumped["aspects"][0]


# --- what a plan changes -----------------------------------------------------------------------------------------------


def _accuracy_family(bundle: AnalysisContextBundle):  # type: ignore[no-untyped-def]  # the family's shape
    (family,) = bundle.multiple_comparisons.families
    return family


@pytest.mark.parametrize("planned", [None, 1, 3])
def test_the_family_reads_each_look_at_alpha_over_the_plan_and_states_alpha(planned: int | None) -> None:
    bundle, _, written = _bundle([("g1", 0, range(30))], planned=planned, shift=0.02, seed="family-0")
    family = _accuracy_family(bundle)
    look = looks_alpha(planned)
    assert family.alpha == SIGNIFICANCE_ALPHA, "what a reader is told: the error held over every planned look"
    assert family.look_alpha == (pytest.approx(look) if planned and planned > 1 else None)
    assert family.interval_level == pytest.approx(1 - look / family.family_size)
    accuracy = next(
        comparison
        for comparison in family.comparisons
        if comparison.name == "field_accuracy" and comparison.contrast.variant_key == fixture_variant_key(_CANDIDATE)
    )
    expected = family_verdicts(
        [(written[_CONTROL], written[_CANDIDATE], True), ({"x": 0.0, "y": 0.0}, {"x": 0.0, "y": 0.0}, True)],
        value_ranges=[(0.0, 1.0), None],
        alpha=look,
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
    assert "over the 3 looks its design plans, where each look's cases were fixed before its data" in (
        _accuracy_family(planned).disclosure
    ), "a plan's level is stated with its condition"


def _candidate_arm() -> CampaignArm:
    return CampaignArm(label="candidate", role="candidate", variant_key=fixture_variant_key(_CANDIDATE))


def _readings(bundle: AnalysisContextBundle, campaign: EvalCampaign) -> CampaignReadings:
    return campaign_readings(campaign, {_ASPECT: bundle}, arms=[*campaign.arms, _candidate_arm()])


#: Sixty cases, the control passing nineteen in twenty and the candidate a little less: a drop held within a margin
#: at the nominal level and not at the level a plan of three looks tightens it to, at the margin between them.
_DROP = dict(seed="floor-held-0", pass_rates=(0.97, 0.93))


def _margins() -> tuple[float, float]:
    """Two margins on the pass rate's drop, read off the bundle itself: one between the largest drop its interval allows
    at the nominal level and the largest it allows at the level a plan of three looks reads ``held`` at (held with no
    plan, unclear under one); and one between the drop observed and the nominal largest (unclear either way)."""
    bundle, _, _ = _bundle([("g1", 0, range(60))], **_DROP)  # type: ignore[arg-type]
    (rate, *_) = floor_readings(bundle, fixture_variant_key(_CONTROL), fixture_variant_key(_CANDIDATE)).rates
    assert rate.samples is not None
    nominal = rate.worst_drop()
    tight = bounded_difference_interval(
        *rate.samples, paired=rate.paired, value_range=(0.0, 1.0), confidence=looks_confidence(INTERVAL_LEVEL, _PLAN)
    )
    assert nominal is not None and tight is not None
    planned = -100 * tight[0]
    observed = 100 * (rate.control_passed / rate.control_n - rate.candidate_passed / rate.candidate_n)
    assert 0 < observed < nominal < planned < 30, (observed, nominal, planned)
    return round((nominal + planned) / 2, 3), round((observed + nominal) / 2, 3)


def _between_margin() -> float:
    """The margin held with no plan and unclear under a plan of three looks (:func:`_margins`)."""
    return _margins()[0]


@pytest.mark.parametrize("planned", [None, _PLAN])
def test_a_rate_floor_and_a_row_within_the_margin_read_held_with_no_plan_and_unclear_under_one(
    planned: int | None,
) -> None:
    """Through the bundle: the plan reaches the floor's held and the row's within_margin."""
    margin = _between_margin()
    bundle, campaign, _ = _bundle(
        [("g1", 0, range(60))],
        planned=planned,
        margin=margin,
        floors=[RateDrop(max_points=margin)],
        **_DROP,  # type: ignore[arg-type]
    )
    readings = _readings(bundle, campaign)
    (row,) = readings.readings
    statuses = {status.status for status in readings.floors}
    if planned is None:
        assert statuses == {"held"} and row.reading == "within_margin"
        assert readings.look_alpha is None
        return
    assert statuses == {"unclear"} and row.reading == "unclear"
    assert readings.alpha == SIGNIFICANCE_ALPHA and readings.look_alpha == pytest.approx(SIGNIFICANCE_ALPHA / _PLAN)
    assert row.within_interval is not None and -row.within_interval[0] > margin, "the interval held is read on"


def test_the_cases_an_unclear_row_projects_are_sized_at_the_level_its_held_is_read_at() -> None:
    """At one margin both rows are unclear; under a plan of three looks the row's held is read on a wider interval, so
    the cases that would lift it are more."""
    low = _margins()[1]

    def projected(planned: int | None) -> int:
        bundle, campaign, _ = _bundle(
            [("g1", 0, range(60))],
            planned=planned,
            margin=low,
            floors=[RateDrop(max_points=low)],
            **_DROP,  # type: ignore[arg-type]
        )
        (row,) = _readings(bundle, campaign).readings
        assert row.reading == "unclear" and row.projection is not None
        assert row.projection.if_none.cases is not None
        return row.projection.if_none.cases

    assert projected(_PLAN) > projected(None), "sized at the tighter level"


#: Forty cases, the candidate cheaper by a little against noisy costs: cheaper shown at the nominal level, not at the
#: level a plan of three looks reads held at.
_COST = dict(seed="cost-held-0", costs=(0.0100, 0.0095), cost_noise=0.0012)


@pytest.mark.parametrize("planned", [None, _PLAN])
def test_a_cost_floor_reads_held_with_no_plan_and_unclear_under_one(planned: int | None) -> None:
    bundle, campaign, _ = _bundle([("g1", 0, range(40))], planned=planned, floors=[CostLower()], **_COST)  # type: ignore[arg-type]
    readings = floor_readings(bundle, fixture_variant_key(_CONTROL), fixture_variant_key(_CANDIDATE))
    assert readings.cost_interval is not None and readings.cost_interval[1] < 0, "the seed drifted off its branch"
    (status,) = _readings(bundle, campaign).floors
    if planned is None:
        assert readings.cost_held_interval is None and status.status == "held"
    else:
        assert readings.cost_held_interval is not None and readings.cost_held_interval[1] >= 0
        assert status.status == "unclear"


def test_a_guardrail_held_with_no_plan_is_undecided_under_one_whose_tighter_interval_reaches_its_margin() -> None:
    """A guardrail's held is a claim of safety made at every look: under a plan it is read on the tighter interval."""

    def check(planned: int | None, margin: float | None):  # type: ignore[no-untyped-def]  # a guardrail check
        bundle, _, _ = _bundle([("g1", 0, range(60))], planned=planned, seed="guard-0", guardrail_margin=margin)
        (found,) = [one for one in bundle.guardrails.checks if one.name == "destructive_calls"]
        return found

    probe = check(None, 0.5)
    assert probe.decision == "held" and probe.interval is not None
    _, _, written = _bundle([("g1", 0, range(60))], seed="guard-0", guardrail_margin=0.5)
    control = [written[f"{_CONTROL}:destructive"][case] for case in sorted(written[f"{_CONTROL}:destructive"])]
    candidate = [written[f"{_CANDIDATE}:destructive"][case] for case in sorted(written[f"{_CANDIDATE}:destructive"])]
    tight = bounded_difference_interval(
        control, candidate, paired=True, value_range=(0.0, 1.0), confidence=looks_confidence(INTERVAL_LEVEL, _PLAN)
    )
    assert tight is not None and probe.interval[1] < tight[1], (probe.interval, tight)
    margin = (probe.interval[1] + tight[1]) / 2
    assert check(None, margin).decision == "held"
    planned = check(_PLAN, margin)
    assert planned.decision == "undecided" and planned.interval == pytest.approx(tight)


def test_a_late_plan_moves_no_floor() -> None:
    margin = _between_margin()
    bundle, campaign, _ = _bundle(
        [("g1", 0, range(60))],
        planned=_PLAN,
        declared=600,
        joined={0: 700},
        margin=margin,
        floors=[RateDrop(max_points=margin)],
        **_DROP,  # type: ignore[arg-type]
    )
    assert bundle.reading_scope.looks is not None and bundle.reading_scope.looks.late_plan == _PLAN
    assert {status.status for status in _readings(bundle, campaign).floors} == {"held"}


def test_a_campaign_s_gains_read_at_alpha_over_the_largest_plan_among_its_aspects() -> None:
    for planned in (None, 4):
        bundle, campaign, _ = _bundle([("g1", 0, range(20))], planned=planned)
        readings = _readings(bundle, campaign)
        assert readings.alpha == SIGNIFICANCE_ALPHA
        assert readings.look_alpha == (None if planned is None else pytest.approx(looks_alpha(planned)))


# --- what a reader is told, and what moves no digest -------------------------------------------------------------------


def _built(bundle: AnalysisContextBundle, campaign: EvalCampaign):  # type: ignore[no-untyped-def]  # a report input
    arms = [*campaign.arms, _candidate_arm()]
    readings = campaign_readings(campaign, {_ASPECT: bundle}, arms=arms)
    return report_input(build_campaign_bundle(campaign, [bundle], arms=arms, readings=readings))


def _looks_caveats(built):  # type: ignore[no-untyped-def]  # the input's caveats on looks
    return [entry for entry in caveat_list(built) if entry.kind == "looks"]


#: Two launches and a candidate clearly ahead on accuracy: the aspect claims a gain.
_TWO_LOOKS_AND_A_GAIN = dict(shift=0.08, seed="gain-0")


def _quoted(test: SeparationTest, planned: int | None, taken: int, *, launched: int | None = None) -> FalseSeparation:
    """What the caveat quotes for ``test``: the named row's figure and setting, read straight off the table."""
    measured = looks_false_separation(test, planned, taken, launched=launched)
    return FalseSeparation(
        test=test,
        rate=measured.rate,
        upper=measured.upper,
        row=measured.row,
        looks=measured.looks,
        setting=measured.setting,
        launched=list(measured.launched),
        largest_step=measured.largest_step,
        unquoted=measured.unquoted,
    )


def test_a_reader_of_a_campaign_that_took_an_undeclared_second_look_is_told_two_looks_none_planned_and_one_caveat() -> (
    None
):
    built = _built(*_bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))], **_TWO_LOOKS_AND_A_GAIN)[:2])  # type: ignore[arg-type]
    (aspect,) = built.aspects
    assert aspect.looks is not None
    assert aspect.looks.model_copy(update={"tests": [], "launched": None, "largest_step": None}) == Looks(
        planned=None, taken=2
    )
    (looks,) = _looks_caveats(built)
    assert (looks.id, looks.cls, looks.reach) == (f"looks:{_ASPECT}", "D", "report")
    assert f"{_ASPECT}/field_accuracy" in looks.readings, "it names the claim the second look loosened"
    assert looks.looks == aspect.looks
    assert looks.false_separation == [_quoted(test, None, 2) for test in aspect.looks.tests]
    assert "t" in aspect.looks.tests and all(quote.rate is not None for quote in looks.false_separation)


def _restated(built: ReportInput, **looks: object) -> ReportInput:
    """``built`` with its aspect's looks set as ``looks`` says, and its caveat list raised again on read."""
    return built.model_copy(
        update={
            "aspects": [aspect.model_copy(update={"looks": Looks(**looks)}) for aspect in built.aspects],  # type: ignore[arg-type]
            "caveat_list": None,
        }
    )


@pytest.fixture(scope="module")
def gained() -> ReportInput:
    """A campaign whose aspect claims a gain, read once."""
    return _built(*_bundle([("g1", 0, range(20))], **_TWO_LOOKS_AND_A_GAIN)[:2])  # type: ignore[arg-type,no-any-return]


#: Each situation a caveat on looks quotes for, and the row it quotes (None: not measured for it, no rate stated).
_SITUATIONS: list[tuple[SeparationTest, int | None, int, str | None]] = [
    # A t-test aspect extended without a plan: the planner-sized t row, which errs more than fixed batches do.
    ("t", None, 2, "t, planner-sized, no plan"),
    ("t", None, 3, "t, planner-sized, no plan"),
    ("t", None, 4, "t, planner-sized, no plan"),
    # A pass-rate aspect: the bounded test's rows, never the t test's.
    ("bounded", None, 2, "bounded, planner-sized, no plan"),
    ("bounded", None, 4, "bounded, planner-sized, no plan"),
    # Past a plan.
    ("t", 2, 3, "t, planner-sized, plan of 2"),
    ("bounded", 3, 4, "bounded, planner-sized, plan of 3"),
    # Within a plan a t-test aspect's planner-sized extends are not shown to hold alpha: the caveat is raised.
    ("t", 2, 2, "t, planner-sized, plan of 2"),
    # Past the last count measured, or under a plan no row measured: not measured, nothing quoted as a bound.
    ("t", None, 5, None),
    ("bounded", None, 5, None),
    ("t", 4, 5, None),
]


@pytest.mark.parametrize(("test", "planned", "taken", "row"), _SITUATIONS)
def test_the_caveat_quotes_the_row_measured_for_its_situation(
    gained: ReportInput, test: SeparationTest, planned: int | None, taken: int, row: str | None
) -> None:
    (looks,) = _looks_caveats(_restated(gained, planned=planned, taken=taken, tests=[test]))
    (quote,) = looks.false_separation
    assert (quote.test, quote.row, quote.looks) == (test, row, taken)
    if row is None:
        assert (quote.rate, quote.upper, quote.unquoted) == (None, None, "not_measured"), "not measured: no rate"
        return
    measured = FALSE_SEPARATION[row]
    assert (measured.test, measured.planned) == (test, planned)
    assert quote.rate == measured.rates[taken - 1], "the named row's own figure"
    assert quote.setting == measured.setting, "the reference setting rides with it"
    assert quote.launched and quote.largest_step is not None
    assert quote.upper == pytest.approx(measured_upper_bound(measured.rates[taken - 1], measured.replicates))
    others = [
        other.rates[taken - 1]
        for other in FALSE_SEPARATION.values()
        if (other.test, other.planned) == (test, planned) and len(other.rates) >= taken
    ]
    assert quote.rate == max(others), "the procedure that errs more: how the looks were added is not stored"


def test_an_aspect_read_by_both_tests_quotes_each_its_own_row(gained: ReportInput) -> None:
    (looks,) = _looks_caveats(_restated(gained, planned=None, taken=3, tests=["bounded", "t"]))
    assert looks.false_separation == [_quoted("bounded", None, 3), _quoted("t", None, 3)]
    assert looks.false_separation[0].rate != looks.false_separation[1].rate


def test_a_t_test_aspect_extended_by_the_planner_is_quoted_more_than_fixed_batches_err() -> None:
    """The Critic's case: a score contrast extended twice. The fixed-batch t figure understates the planner-sized one."""
    quoted = looks_false_separation("t", None, 3)
    assert quoted.row == "t, planner-sized, no plan"
    assert quoted.rate is not None and quoted.rate > FALSE_SEPARATION["t, fixed batches, no plan"].rates[2]


def test_looks_within_a_plan_raise_nothing_only_where_keeping_it_is_shown(gained: ReportInput) -> None:
    """A pass-rate aspect within its plan: the planner-sized rows' upper bounds clear alpha, so nothing is raised. A
    t-test aspect within the same plan: they do not, so the caveat is raised and quotes the measured rate."""
    assert plan_kept("bounded", 2, 2) and plan_kept("bounded", 3, 3)
    assert not _looks_caveats(_restated(gained, planned=2, taken=2, tests=["bounded"]))
    assert not plan_kept("t", 2, 2)
    (looks,) = _looks_caveats(_restated(gained, planned=2, taken=2, tests=["bounded", "t"]))
    assert looks.false_separation == [_quoted("bounded", 2, 2), _quoted("t", 2, 2)]
    t_row = FALSE_SEPARATION["t, planner-sized, plan of 2"]
    assert measured_upper_bound(t_row.rates[1], t_row.replicates) > SIGNIFICANCE_ALPHA


def test_plan_kept_reads_the_planner_sized_row_s_upper_bound() -> None:
    """Shown only where the planner-sized row's upper bound is at most alpha at that count; one look is the plan's
    first, read at alpha over L; a plan or count no planner-sized row measured is not shown."""
    for name, row in FALSE_SEPARATION.items():
        if row.procedure != "planner_extends" or row.planned is None:
            continue
        for taken in range(2, row.planned + 1):
            shown = measured_upper_bound(row.rates[taken - 1], row.replicates) <= SIGNIFICANCE_ALPHA
            assert plan_kept(row.test, row.planned, taken) is shown, (name, taken)
    assert plan_kept("t", 5, 1)
    assert not plan_kept("t", 5, 2) and not plan_kept("bounded", 5, 2)


# --- what a page prints of a plan's level (#1220) -----------------------------------------------------------------------


def _page_lines(bundle: AnalysisContextBundle) -> tuple[str, list[str], list[dict[str, str]]]:
    """An aspect page's contrast table's interval head, every line under it, and each row's reading and interval."""
    report = build_code_only_report(bundle, assembled_at=_stamp(0))
    (table,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "comparisons"]
    head = next(column.header for column in table.columns if column.key == "interval")
    rows = [{key: str(row[key].value) for key in ("reading", "interval")} for row in table.rows]
    return head, [block.text for block in report.blocks if isinstance(block, DisclosureBlock)], rows


def _conditions(lines: list[str]) -> list[str]:
    return [line for line in lines if "looks the design plans" in line]


def _with_looks(bundle: AnalysisContextBundle, **looks: object) -> AnalysisContextBundle:
    """``bundle`` with its aspect's looks restated as ``looks`` says."""
    held = bundle.reading_scope.looks
    assert held is not None
    scope = bundle.reading_scope.model_copy(update={"looks": held.model_copy(update=looks)})
    return bundle.model_copy(update={"reading_scope": scope})


#: Two looks within a plan of two, field accuracy read by the t test and the pass rates by the bounded one.
_GAIN_TWICE = dict(shift=0.08, seed="gain-0", pass_rates=(0.6, 0.8))


def test_a_page_prints_a_plan_s_level_only_where_keeping_it_is_shown_and_otherwise_states_its_condition() -> None:
    """Within a plan of two looks, field accuracy is read by the t test, whose planner-sized extends are not shown to
    hold the plan's level (``plan_kept``); the pass rates, on the bounded test, are. Read once, the plan is kept on
    every test, and the head prints the joint level. Read twice, no head or cell prints it: one line states it with its
    condition, which says the family's level is not shown and names the t-test reading behind that."""
    once, _, _ = _bundle([("g1", 0, range(40))], planned=2, **_GAIN_TWICE)  # type: ignore[arg-type]
    twice, _, _ = _bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))], planned=2, **_GAIN_TWICE)  # type: ignore[arg-type]
    for bundle, taken in ((once, 1), (twice, 2)):
        looks = bundle.reading_scope.looks
        assert looks is not None and (looks.planned, looks.taken, looks.tests) == (2, taken, ["bounded", "t"])
    assert plan_kept("t", 2, 1) and plan_kept("bounded", 2, 2) and not plan_kept("t", 2, 2)

    (family,) = once.multiple_comparisons.families
    head, lines, _ = _page_lines(once)
    assert head == f"95% interval (adjusted for {family.family_size} comparisons)", "kept: the level is printed"
    assert not _conditions(lines)

    (family,) = twice.multiple_comparisons.families
    bases = {comparison.name: comparison.interval_basis for comparison in family.comparisons}
    assert bases["field_accuracy"] == "t" and bases["passed"] == "bounded"
    head, lines, rows = _page_lines(twice)
    assert head == "Interval", "not kept: the head prints no level"
    assert not [row for row in rows if "adjusted for" in row["interval"]], "nor does any cell"
    assert _conditions(lines) == [
        f"95% (adjusted for {family.family_size} comparisons) is the level the intervals are read to hold together "
        "over the 2 looks the design plans; it holds where each look's cases were fixed before its results, and is "
        "not shown where a look's cases were sized from an earlier reading, which is not shown to keep it for Field "
        "accuracy."
    ]


def test_looks_past_a_plan_or_short_counted_say_what_the_count_is_and_name_every_reading_of_the_family() -> None:
    """Past the plan, the sentence states the count as what it is, the most looks any contrast behind the readings
    had, and that the family's level is not shown; a count that may be short says more looks may have been taken. Both
    name every reading of the family, whatever its test."""
    twice, _, _ = _bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))], planned=2, **_GAIN_TWICE)  # type: ignore[arg-type]
    (family,) = twice.multiple_comparisons.families
    level = f"95% (adjusted for {family.family_size} comparisons) is the level the intervals are read to hold together"
    head, lines, _ = _page_lines(_with_looks(twice, taken=3))
    assert head == "Interval"
    assert _conditions(lines) == [
        f"{level} over the 2 looks the design plans; the most looks any contrast behind Field accuracy, Ok check and "
        "Pass rate had is 3, past the plan, so it is not shown."
    ]
    head, lines, _ = _page_lines(_with_looks(twice, lower_bound=True))
    assert head == "Interval"
    assert _conditions(lines) == [
        f"{level} over the 2 looks the design plans; not every look behind Field accuracy, Ok check and Pass rate was "
        "counted, so more than planned may have been taken, and it is not shown."
    ]


def test_a_page_of_families_kept_and_not_prints_each_kept_family_s_level_in_its_cells_and_none_in_the_others() -> None:
    """Two families on one aspect under a plan of two looks taken: the t test's (a family of one) is not shown to keep
    the plan, the bounded test's is. Each cell of the kept family says its family's joint level; no cell of the other
    says any, and its one condition line states the plan's level, a family of one's 95%, never its interval's 97.5% at
    each look."""
    twice, _, _ = _bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))], planned=2, **_GAIN_TWICE)  # type: ignore[arg-type]
    (family,) = twice.multiple_comparisons.families
    look_alpha = family.look_alpha
    assert look_alpha is not None
    halves = [
        [comparison for comparison in family.comparisons if comparison.interval_basis == "t"],
        [comparison for comparison in family.comparisons if comparison.interval_basis != "t"],
    ]
    families = [
        family.model_copy(
            update={"comparisons": half, "family_size": len(half), "interval_level": 1 - look_alpha / len(half)}
        )
        for half in halves
    ]
    comparisons = twice.multiple_comparisons.model_copy(update={"families": families})
    head, lines, rows = _page_lines(twice.model_copy(update={"multiple_comparisons": comparisons}))
    assert head == "Interval"
    cells = {row["reading"]: row["interval"] for row in rows}
    assert "adjusted for" not in cells["Field accuracy"] and "%" not in cells["Field accuracy"]
    assert cells["Pass rate"].endswith(" at 95% (adjusted for 3 comparisons)")
    assert _conditions(lines) == [
        "95% is the level the intervals are read to hold together over the 2 looks the design plans; it holds where "
        "each look's cases were fixed before its results, and is not shown where a look's cases were sized from an "
        "earlier reading, which is not shown to keep it for Field accuracy."
    ]


def test_a_written_page_prints_no_level_where_its_plan_is_not_shown_kept() -> None:
    """The written page's heads and level cells (``interval_level``) and ``{weakest.interval}`` print the family's
    joint level where the plan is shown kept, and no level where it is not: never each interval's own level, which
    holds no better over several looks (owner ruling, level in headings)."""
    bundle, campaign, _ = _bundle(
        [("g1", 0, range(20)), ("g2", 120, range(20, 40))],
        planned=2,
        margin=0.05,
        **_GAIN_TWICE,  # type: ignore[arg-type]
    )
    unkept = _built(bundle, campaign)
    kept = _restated(unkept, planned=2, taken=2, tests=["bounded"])
    (written,) = unkept.contrast_families
    level = written.level
    assert written.aspect is not None and level is not None

    assert interval_level(level, written, looks_of(kept, written.aspect)) == (
        f"95% (adjusted for {written.family_size} comparisons)"
    )
    assert joint_level(written, looks_of(kept, written.aspect)) == (0.95, written.family_size)
    # Corrected from the first round, which pinned each interval's own level here (`level_text(level)`, 99.4%): that
    # level holds no better over several looks than the joint one, and a head never prints it.
    assert interval_level(level, written, looks_of(unkept, written.aspect)) == UNSHOWN_PLAN_LEVEL
    assert not any(character.isdigit() for character in UNSHOWN_PLAN_LEVEL)
    assert joint_level(written, looks_of(unkept, written.aspect)) is None

    def weakest(held: ReportInput) -> str:
        return format_figure(resolve("weakest.interval", held), forms=PageForms.of(held))

    assert weakest(kept) == f"the 95% interval, adjusted for {written.family_size} comparisons"
    assert weakest(unkept) == "the multiplicity-corrected interval"


#: The candidate arm's id on a written page built from :func:`_bundle` (:func:`_candidate_arm`).
_CANDIDATE_ARM = "candidate"


def _written_level(held: ReportInput) -> dict[str, str]:
    """A written page's level cell for its one contrast (the full results' contrast table), and the head of a chart
    comparing the two arms on that contrast's measure (``compare``, drawn on the contrast's own interval), each as the
    page prints it."""
    forms = PageForms.of(held)
    look = block_look(held, fact_arms(held))
    (table,) = [table for table in every_table(held, look, forms=forms) if table.name == "contrasts"]
    (row,) = table.rows
    query = ChartQuery.model_validate(
        {
            "source": "compare",
            "aspect": "",
            "measures": ["field_accuracy"],
            "arms": [_CANDIDATE_ARM, "control"],
            "strata": [],
            "limit": 0,
            "transform": "none",
            "small_multiples_by": "none",
        }
    )
    planned = PlannedChart.model_validate(
        {"kind": "dot_interval", "title": "", "query": query, "reference": [], "emphasis": ""}
    )
    spec, faults = complete_chart(planned, held, "chart", forms=forms)
    assert not faults
    chart = resolve_chart(spec, look, title="", forms=forms).chart
    return {"level cell": str(row["level"].value), "chart head": drawn_head(chart.data)}


@pytest.mark.parametrize("where", ["level cell", "chart head"])
def test_a_written_page_prints_no_level_for_a_family_of_one_where_its_plan_is_not_shown_kept(where: str) -> None:
    """A family of one states no joint level (``joint_guarantee``); under a plan of two looks its stated level, 97.5%,
    is the one each look reads at. Read twice on the t test, the plan is not shown kept, so the written page's level
    cell and a chart's head print no level, as the aspect page's head prints none for the same family; kept, both
    print its level. The plan's decision comes first for a family of any size."""
    bundle, campaign, _ = _bundle(
        [("g1", 0, range(20)), ("g2", 120, range(20, 40))],
        planned=2,
        stripped=False,
        **_TWO_LOOKS_AND_A_GAIN,  # type: ignore[arg-type]
    )
    (family,) = bundle.multiple_comparisons.families
    assert (family.family_size, family.interval_level) == (1, 0.975)
    assert _page_lines(bundle)[0] == "Interval", "the aspect page prints no level for this family"
    unkept = _built(bundle, campaign)
    (written,) = unkept.contrast_families
    assert (written.family_size, written.level) == (1, "97.5%")
    assert plan_unshown(looks_of(unkept, written.aspect)) is not None

    after = " intervals" if where == "chart head" else ""
    assert _written_level(unkept)[where] == f"{UNSHOWN_PLAN_LEVEL}{after}"
    assert _written_level(_restated(unkept, planned=2, taken=2, tests=["bounded"]))[where] == f"97.5%{after}"


def test_the_one_decision_of_a_plan_s_keeping_reads_past_short_unrecorded_and_sized_looks() -> None:
    """``caveats.unkept_plan`` decides for the caveat on looks and every page alike; a page qualifies only a plan of
    more than one look (``plan_unshown``), so no plan, or one of one look, leaves its level as it was."""
    assert unkept_plan(Looks(planned=2, taken=2, tests=["bounded"])) is None
    assert unkept_plan(Looks(planned=2, taken=3, tests=["bounded"])) == UnkeptPlan(2, 3, "past", ("bounded",))
    assert unkept_plan(Looks(planned=2, taken=2, lower_bound=True, tests=["bounded"])) == UnkeptPlan(
        2, 2, "uncounted", ("bounded",)
    )
    assert unkept_plan(Looks(planned=2, taken=2, tests=["bounded", "t"])) == UnkeptPlan(2, 2, "sized", ("t",))
    assert unkept_plan(Looks(planned=2, recorded=False, evidence_changes=3)) == UnkeptPlan(2, None, "uncounted", ())
    assert unkept_plan(Looks(planned=None, taken=2, tests=["t"])) == UnkeptPlan(None, 2, "past", ("t",))
    assert plan_unshown(Looks(planned=None, taken=2, tests=["t"])) is None, "no plan: no plan's level"
    assert plan_unshown(Looks(planned=1, taken=2, tests=["t"])) is None, "a plan of one look: none either"
    assert plan_unshown(Looks(planned=2, taken=3, tests=["t"])) == UnkeptPlan(2, 3, "past", ("t",))


def test_a_campaign_s_scorecard_states_its_plan_s_error_rate_only_where_keeping_it_is_shown() -> None:
    """The scorecard's pass rates are read on the bounded test. Two looks within a plan of two are shown kept, and the
    Methods paragraph states the error rate; within a plan of four no planner-sized row measured it, so the rate is not
    stated and one line states the level with its condition, naming the aspect's pass rate."""
    for planned, kept in ((2, True), (4, False)):
        bundle, campaign, _ = _bundle(
            [("g1", 0, range(20)), ("g2", 120, range(20, 40))],
            planned=planned,
            shift=0.08,
            seed="gain-0",
            pass_rates=(0.6, 0.85),
        )
        arms = [*campaign.arms, _candidate_arm()]
        read = campaign_readings(campaign, {_ASPECT: bundle}, arms=arms)
        assert read.planned_looks == planned
        reports = {_ASPECT: build_code_only_report(bundle, assembled_at=_stamp(0), arm_labels=read.arm_labels())}
        page = build_campaign_report(campaign, read, reports)
        (methods,) = [
            block.text
            for block in page.blocks
            if isinstance(block, DisclosureBlock) and block.section == "methods" and "is read at" in block.text
        ]
        assert ("the chance of being wrongly shown better" in methods) is kept, planned
        if not kept:
            assert (
                "95% is the level the intervals are read to hold together over the 4 looks the design plans; it holds "
                "where each look's cases were fixed before its results, and is not shown where a look's cases were "
                "sized from an earlier reading, which is not shown to keep it for the pass rate on b."
            ) in methods


def _past_plan(shape: Shape) -> CampaignReport:
    """``shape``'s page with its readings read under a plan of two looks, and every aspect looked at three times."""
    looks = Looks(planned=2, taken=3, tests=["bounded"])
    reports = {
        label: report.model_copy(
            update={
                "bundle": report.bundle.model_copy(
                    update={"reading_scope": report.bundle.reading_scope.model_copy(update={"looks": looks})}
                )
            }
        )
        for label, report in shape.reports.items()
    }
    readings = shape.readings.model_copy(update={"look_alpha": looks_alpha(2), "planned_looks": 2})
    return build_campaign_report(shape.campaign, readings, reports)


def _glyph_key(page: CampaignReport) -> str:
    (key,) = [block.glyph_key for block in page.blocks if isinstance(block, TableBlock) and block.glyph_key]
    return key


def test_a_change_of_several_candidates_past_its_plan_states_each_arm_s_level_and_that_none_runs_across_arms() -> None:
    """s1 reads three candidates on three aspects, each candidate's rows corrected across its own aspects only. Past
    its plan, the Methods paragraph states no error rate, says the level is each arm's intervals' (never that the
    intervals hold together), and keeps the sentence that the correction does not run across the arms, with no rate;
    the glyph key prints no level."""
    shape = s1_change()
    assert _glyph_key(shape.page()).endswith("with its 95% interval (adjusted for 3 comparisons); right is better."), (
        "no plan: the key states the joint level"
    )
    page = _past_plan(shape)
    methods = campaign_report_markdown(page)
    assert (
        "95% (adjusted for 3 comparisons) is the level each arm's intervals are read to hold together over the 2 looks "
        "the design plans; the most looks any contrast behind the pass rate on billing, the pass rate on tone and the "
        "pass rate on refunds had is 3, past the plan, so it is not shown."
    ) in methods
    assert "the intervals are read to hold together" not in methods
    assert (
        "The correction does not run across the 3 arms, so some arm is wrongly shown better more often than any one "
        "arm is, and likewise for worse."
    ) in methods
    assert "up to 7.5%" not in methods and "at most 2.5%" not in methods
    assert _glyph_key(page).endswith("with its interval; right is better.")


def test_a_choice_past_its_plan_states_its_pairs_level_with_its_condition() -> None:
    """s2's pairs are one step-down over every pair, so the level is the intervals' together; past the plan, the note
    under the pairs states it with its condition, in place of saying they hold together at it."""
    shape = s2_choice()
    held = "The intervals hold together at 95% (adjusted for 3 comparisons)."
    assert held in campaign_report_markdown(shape.page()).splitlines()
    said = campaign_report_markdown(_past_plan(shape))
    assert held not in said.splitlines()
    assert (
        "95% (adjusted for 3 comparisons) is the level the intervals are read to hold together over the 2 looks the "
        "design plans; the most looks any contrast behind the pass rate on answers had is 3, past the plan, so it is "
        "not shown."
    ) in said.splitlines()


def test_a_second_look_at_an_aspect_that_claims_nothing_qualifies_the_method_only() -> None:
    built = _built(*_bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))])[:2])
    (looks,) = _looks_caveats(built)
    assert looks.reach == "method"


def test_a_second_look_at_a_campaign_created_before_looks_were_counted_qualifies_the_method_only() -> None:
    built = _built(
        *_bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))], stamped=False, **_TWO_LOOKS_AND_A_GAIN)[:2]  # type: ignore[arg-type]
    )
    (looks,) = _looks_caveats(built)
    assert looks.reach == "method" and looks.looks is not None and looks.looks.plannable is False


def test_a_plan_whose_keeping_the_count_cannot_show_raises_the_caveat_with_the_count_alone() -> None:
    """A run deleted after its look: the count is a lower count, so looks past the plan cannot be ruled out. More looks
    may have been taken and no row is a floor, so the count is shown alone, with no rate."""
    gone = MembershipChange(run_id="gone", change="left", at=_stamp(300))
    built = _built(*_bundle([("g1", 0, range(20))], planned=2, left=[gone], **_TWO_LOOKS_AND_A_GAIN)[:2])  # type: ignore[arg-type]
    (aspect,) = built.aspects
    assert aspect.looks is not None
    stripped = aspect.looks.model_copy(update={"tests": [], "launched": None, "largest_step": None})
    assert stripped == Looks(planned=2, taken=2, lower_bound=True)
    (looks,) = _looks_caveats(built)
    assert (looks.reach, looks.looks, looks.false_separation) == ("report", aspect.looks, [])


def test_a_campaign_whose_looks_were_not_recorded_says_so_with_no_rate_and_no_count_of_looks() -> None:
    """Joined before durable 17.3 recorded membership, its evidence changed by two launches: one entry at method
    reach, the looks carried as not recorded, no rate."""
    built = _built(
        *_bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))], recorded=False, **_TWO_LOOKS_AND_A_GAIN)[:2]  # type: ignore[arg-type]
    )
    (aspect,) = built.aspects
    assert aspect.looks is not None and aspect.looks.recorded is False and aspect.looks.taken is None
    (looks,) = _looks_caveats(built)
    assert (looks.reach, looks.false_separation) == ("method", [])
    assert looks.looks == aspect.looks and looks.looks.evidence_changes == 2
    dumped = looks.model_dump(by_alias=True)
    assert "false_separation" not in dumped and dumped["looks"]["taken"] is None


def test_a_campaign_whose_looks_were_not_recorded_and_whose_evidence_changed_once_raises_nothing() -> None:
    built = _built(*_bundle([("g1", 0, range(20))], recorded=False, **_TWO_LOOKS_AND_A_GAIN)[:2])  # type: ignore[arg-type]
    assert built.aspects[0].looks is None, "one change of the evidence and no plan: nothing to state"
    assert not _looks_caveats(built)


def test_a_campaign_extended_after_the_record_raises_at_report_reach_with_the_looks_recorded_since() -> None:
    """Mixed: a campaign created before looks were counted, its first launch unrecorded, extended since. The looks
    recorded since are a lower count, shown alone with no rate, and the caveat reaches the report (owner ruling L5)."""
    built = _built(
        *_bundle(
            [("g1", 0, range(20)), ("g2", 120, range(20, 40))], unrecorded={0}, stamped=False, **_TWO_LOOKS_AND_A_GAIN
        )[:2]  # type: ignore[arg-type]
    )
    (aspect,) = built.aspects
    assert aspect.looks is not None
    assert (aspect.looks.recorded, aspect.looks.taken, aspect.looks.lower_bound) == (False, 2, True)
    (looks,) = _looks_caveats(built)
    assert looks.reach == "report"
    assert (looks.looks, looks.false_separation) == (aspect.looks, [])
    assert "false_separation" not in looks.model_dump(by_alias=True)


def test_a_campaign_whose_looks_were_recorded_gets_a_count_and_a_rate() -> None:
    """The same two launches, recorded: a count of looks, and the rate measured for it."""
    built = _built(*_bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))], **_TWO_LOOKS_AND_A_GAIN)[:2])  # type: ignore[arg-type]
    (looks,) = _looks_caveats(built)
    assert looks.looks is not None and looks.looks.taken == 2
    assert looks.false_separation and all(quote.rate is not None for quote in looks.false_separation)


def test_a_reference_outside_the_campaign_s_own_sizes_quotes_no_rate(gained: ReportInput) -> None:
    """A launch smaller than any simulated, or a step larger than the simulated cap: the reference does not cover the
    campaign, so it is named with its setting and no rate is quoted. Within the setting, the rate is quoted."""
    for sizes in ({"launched": 10}, {"launched": 60}, {"launched": 20, "largest_step": 150}):
        (looks,) = _looks_caveats(_restated(gained, planned=None, taken=3, tests=["t"], **sizes))
        (quote,) = looks.false_separation
        assert (quote.rate, quote.upper, quote.unquoted) == (None, None, "outside_setting"), sizes
        assert quote.setting is not None and quote.launched == [20, 40] and quote.largest_step == 100
    (looks,) = _looks_caveats(_restated(gained, planned=None, taken=3, tests=["t"], launched=30, largest_step=100))
    (quote,) = looks.false_separation
    assert quote.rate is not None and quote.unquoted is None


def test_the_quote_is_described_as_a_reference_and_never_as_the_campaign_s_level() -> None:
    """No description tells a writer the readings now hold the quoted rate, or that it limits the campaign's error."""
    texts = [field.description or "" for field in FalseSeparation.model_fields.values()]
    texts += [FalseSeparation.__doc__ or "", CaveatEntry.model_fields["false_separation"].description or ""]
    for text in texts:
        for wrong in ("now hold", "readings hold", "ceiling", "at most"):
            assert wrong not in text, (wrong, text)
    assert "reference" in (CaveatEntry.model_fields["false_separation"].description or "")
    assert "never this campaign" in (FalseSeparation.model_fields["rate"].description or "")


def test_an_aspect_s_looks_are_outside_the_input_s_digest() -> None:
    """Derived from stored facts about the campaign, never an input: a look taken re-keys no input or snapshot."""
    built = _built(*_bundle([("g1", 0, range(20)), ("g2", 120, range(20, 40))])[:2])
    assert built.aspects[0].looks is not None
    other = Looks(planned=None, taken=7, lower_bound=True)
    recounted = built.model_copy(
        update={"aspects": [aspect.model_copy(update={"looks": other}) for aspect in built.aspects]}
    )
    unset = built.model_copy(
        update={"aspects": [aspect.model_copy(update={"looks": None}) for aspect in built.aspects]}
    )
    assert report_input_digest(built) == report_input_digest(recounted) == report_input_digest(unset)

"""Claim readings rank arms through the judged namespace, and a criterion's losses withhold only what they cost.

A claim criterion's readings are judged readings that are no rubric dimension: the score projection emits one row per
reading per cell, named on ``rubric_dim`` with no rubric scale and its criterion on ``instrument``; the bundle
summarises the three that rank (false claims per unit, the share of units with a false claim, the grounded share) as
judged measures, each a rate or a share and never a level on a scale, and the comparison family tests them as it tests
a dimension. The campaigns here are host-shaped and generic: a rules-answer rung (one unit per answer, grounded in the
arm's own documents, an arm with no research reading a grounded share of 0) and a fiction rung (entries as units,
grounded in nothing, the web off).

Losses are per criterion and per reading: a lost criterion leaves every reading of the cell, a lost ground step only
the grounded share, and a tally still waiting on its stored grade is counted apart and never as a loss. A reading
whose arms lost cells at shares that differ beyond chance, by one test across every arm, is withheld (``trials_lost``) and tested in no family;
any loss, pending cell or web check skipped past its budget raises a class-B caveat naming the arms.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``projected_claim_readings``: dropping the null rows of a lost criterion, so no arm counts a loss and nothing is
  withheld.
- ``judged_measures``: describing a claim reading by ``describe_rubric_dim``, which refuses it (no scale recorded).
- ``_claim_loss_homogeneity_p``: reading few losses by the chi-square law, so four lost cells on one arm withhold.
- ``_family_readings``: dropping the ``withheld`` filter, so a withheld reading is tested in the family.
- ``figure_unit``: reading a claim reading as a 1-5 score.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import pytest
from typing import Any


from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.analysis.agreement import claim_judge_keys
from threetears.evals.analysis.bundle.cell_reads import judged_measures, judged_values
from threetears.evals.analysis.bundle.trials import read_trial
from threetears.evals.analysis.bundle.bars import bar_adjudications
from threetears.evals.analysis.first_screen import RANKS
from threetears.evals.analysis.floors import floor_readings
from threetears.evals.analysis.report.build import build_code_only_report
from threetears.evals.analysis.report.serialize_md import report_markdown
from threetears.evals.analysis.lenses.pivot import PivotError, compute_pivot
from threetears.evals.analysis.report.keys.resolve import figure_unit
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.reporting import METRIC_SCORE, project_score_records
from threetears.evals.kernel import CampaignArm
from threetears.evals.kernel.claims.readings import (
    CLAIM_LOST_OUTCOME,
    CLAIM_PENDING_OUTCOME,
    template_claim_readings,
)
from threetears.evals.kernel.declaration import BarOverride, UnreadableBarName, resolve_bar_name
from threetears.evals.kernel.result_condition import resolve_result_condition
from threetears.evals.kernel.scoring import result_composite
from threetears.evals.schema import ClaimCriterion, EvalResult, EvalRun, EvalTemplate, PageBudget, RepairBudget
from threetears.evals.schema.models import ClaimJudgeCounts, ClaimTally, EvalCaseStratum
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.factories import (
    fixture_variant_key,
    make_aspect_campaign,
    make_eval_result,
    make_eval_run,
    make_template,
    minimal_aspect_design,
)
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

#: The rules-answer rung's criterion (grounded in the arm's own documents) and the fiction rung's (grounded in none).
RULES = "rules.claims"
SCOUT = "scout.claims"
#: The arm that answers from memory, the control, and the arm that looks the rules up first.
MEMORY = "memory-model"
LOOKUP = "lookup-model"
CASES = 12
K = 2
_SHA = "c" * 64


def _counts(
    claims: int, *, false: int = 0, specific: int = 0, grounded: int = 0, skipped: int = 0, model: str = "judge-a"
) -> ClaimJudgeCounts:
    """One judge key's counts over asserted world claims: ``false`` contradicted, ``skipped`` past the web budget."""
    return ClaimJudgeCounts(
        served_model=model,
        judge_temperature=0.0,
        judge_sent_settings=None,
        claims=claims,
        asserted=claims,
        hedged=0,
        attributed=0,
        asserted_specific=specific,
        contradicted=false,
        asserted_contradicted=false,
        hedged_contradicted=0,
        attributed_contradicted=0,
        grounded_specific=grounded,
        unverifiable=skipped,
        true_ungrounded=0,
        web_decided=0,
        evidence_conflicts=0,
        web_checked=0,
        web_skipped=skipped,
    )


def _tally(
    *,
    false: int = 0,
    grounded: int | None = 0,
    session: int = 0,
    skipped: int = 0,
    units: int = 1,
    ground_loss: str | None = None,
    web_loss: str | None = None,
    by_judge: tuple[ClaimJudgeCounts, ...] | None = None,
) -> ClaimTally:
    """A cell's tally: four asserted specific world claims over ``units`` units, ``false`` of them contradicted in one
    unit; ``grounded`` None for a criterion that asks no groundedness."""
    asks = grounded is not None
    return ClaimTally(
        criterion_fingerprint=_SHA,
        documents_declared=True,
        grounded=asks,
        units=units,
        units_with_false=min(1, false, units),
        session_claims=session,
        by_judge=by_judge
        or (_counts(4, false=false, specific=4, grounded=0 if ground_loss else grounded or 0, skipped=skipped),),
        ground_loss=ground_loss,
        web_loss=web_loss,
    )


def _criterion(name: str, *, grounded: bool) -> ClaimCriterion:
    return ClaimCriterion(
        name=name,
        guidance="A rule's number, a name or a place is a specific claim.",
        grounded_in=("run",) if grounded else None,
        web=None,
        pages=PageBudget(per_document_chars=8000, total_chars=24000, window_chars=1200),
        repairs=RepairBudget(max_items_per_cell=10),
        judge="builtin",
    )


def _template() -> EvalTemplate:
    return make_template(
        id="tpl-rules",
        rubric=[],
        claim_criteria=[_criterion(RULES, grounded=True), _criterion(SCOUT, grounded=False)],
        claim_unit_names={RULES: "answer", SCOUT: "entry"},
    )


def _result(run: EvalRun, model: str, case: int, k: int, **fields: Any) -> EvalResult:
    return make_eval_result(
        id=f"{model}-{case:02d}-{k}",
        eval_run_id=run.id,
        scope_id=run.scope_id,
        model=model,
        test_case_id=f"tc-{case:02d}",
        k_iteration=k,
        goal_state_outcomes=[],
        rubric_scores=[],
        cost_usd=None,
        **fields,
    )


def _memory_cell(case: int, k: int) -> dict[str, Any]:
    """From memory: a false claim in most answers, and no document to ground any specific claim."""
    return {
        "claim_tallies": {RULES: _tally(false=1 + (case + k) % 2, grounded=0), SCOUT: _tally(false=1, grounded=None)}
    }


def _lookup_cell(case: int, k: int) -> dict[str, Any]:
    """With a lookup: a false claim in one answer of six, and most specific claims grounded in what it read."""
    false = 1 if (case * K + k) % 6 == 0 else 0
    return {
        "claim_tallies": {
            RULES: _tally(false=false, grounded=3 + (case % 2), session=2),
            SCOUT: _tally(false=false, grounded=None),
        }
    }


def _campaign(
    memory: Any = _memory_cell, lookup: Any = _lookup_cell
) -> tuple[list[EvalRun], dict[str, list[EvalResult]]]:
    runs: list[EvalRun] = []
    results: dict[str, list[EvalResult]] = {}
    for model, cell in ((MEMORY, memory), (LOOKUP, lookup)):
        run = make_eval_run(status="completed", candidate_model=model, template_id="tpl-rules")
        runs.append(run)
        results[run.id] = [
            _result(run, model, case, k, **cell(case, k)) for case in range(CASES) for k in range(1, K + 1)
        ]
    return runs, results


class _StratifiedStorage(ToyhostStorage):
    """The toy store, its cases sorted into two strata (``early``, ``late``), so a page lays out its strata table."""

    def load_case_strata(self, test_case_ids: Sequence[str], scope_id: str) -> list[EvalCaseStratum]:
        return [EvalCaseStratum(id=case, stratum="early" if case < "tc-06" else "late") for case in test_case_ids]


def _bundle(memory: Any = _memory_cell, lookup: Any = _lookup_cell, *, strata: bool = False) -> AnalysisContextBundle:
    runs, results = _campaign(memory, lookup)
    asked = min(run.created_at for run in runs)
    campaign = make_aspect_campaign(
        label="rules",
        template_id="tpl-rules",
        run_ids=[run.id for run in runs],
        design=minimal_aspect_design(merit_axes=["quality"]),
        question="does looking the rules up make answers truer?",
        asked_at=asked,
        control=CampaignArm(label="memory", role="control", variant_key=fixture_variant_key(MEMORY), named_at=asked),
        scope_id=runs[0].scope_id,
        name="claim ranking",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
    )
    storage = (_StratifiedStorage if strata else ToyhostStorage)(runs, results, templates=[_template()])
    return assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())


@cache
def _ranked() -> AnalysisContextBundle:
    return _bundle()


def _rows(result: EvalResult, run: EvalRun, *beside: EvalResult) -> list[Any]:
    """``result``'s claim rows, projected with ``beside``: the rest of its run, which says whether a criterion it holds
    no tally of asks groundedness."""
    projection = project_score_records([run], [result, *beside], archived_run_ids=None, profile=toyhost_profile())
    return [record for record in projection.records if record.instrument is not None and record.result_id == result.id]


# --- the projection -------------------------------------------------------------------------------------------------


def test_a_claim_reading_is_a_scale_less_judged_row_keyed_by_its_criterion() -> None:
    runs, results = _campaign()
    run, result = runs[1], results[runs[1].id][0]
    rows = {(record.instrument, record.rubric_dim): record for record in _rows(result, run)}

    false_rate = rows[(RULES, f"{RULES}_false_per_unit")]
    assert false_rate.metric == METRIC_SCORE and false_rate.rubric_scale is None
    assert (RULES, f"{RULES}_grounded_share") in rows
    assert (SCOUT, f"{SCOUT}_grounded_share") not in rows, "a criterion grounded in nothing has no grounded share"
    # judged_values yields the claim rows beside any dimension's.
    yielded = {name for name, record in judged_values(list(rows.values())) if record.instrument is not None}
    assert {f"{RULES}_false_per_unit", f"{RULES}_false_unit_share", f"{RULES}_grounded_share"} <= yielded


def test_session_claims_show_only_in_their_own_count() -> None:
    """A claim about the session itself counts in no world reading: two of them move nothing but ``_session_claims``."""
    runs, results = _campaign()
    run, result = runs[1], results[runs[1].id][1]
    with_session = {record.rubric_dim: record.value for record in _rows(result, run)}
    without = result.model_copy(
        update={
            "claim_tallies": {
                **result.claim_tallies,
                RULES: result.claim_tallies[RULES].model_copy(update={"session_claims": 0}),
            }
        }
    )
    plain = {record.rubric_dim: record.value for record in _rows(without, run)}

    assert with_session.pop(f"{RULES}_session_claims") == 2.0
    assert plain.pop(f"{RULES}_session_claims") == 0.0
    assert with_session == plain


def test_a_lost_criterion_a_lost_ground_step_and_a_pending_tally_each_leave_only_their_own_rows() -> None:
    runs, results = _campaign()
    run = runs[1]
    base = results[run.id][0]
    ranked = {f"{RULES}_false_per_unit", f"{RULES}_false_unit_share", f"{RULES}_grounded_share"}

    sibling = results[run.id][1]
    lost = base.model_copy(
        update={"claim_tallies": {SCOUT: base.claim_tallies[SCOUT]}, "claim_failures": {RULES: "cut_short"}}
    )
    held = [record for record in _rows(lost, run, sibling) if record.instrument == RULES]
    assert {r.rubric_dim for r in held if r.outcome == CLAIM_LOST_OUTCOME} == ranked
    assert all(r.value is None for r in held if r.outcome == CLAIM_LOST_OUTCOME)
    assert [r.value for r in held if r.rubric_dim == f"{RULES}_lost"] == [1.0]

    ground = base.model_copy(
        update={"claim_tallies": {**base.claim_tallies, RULES: _tally(false=1, ground_loss="unreadable")}}
    )
    held = [record for record in _rows(ground, run) if record.instrument == RULES]
    assert {r.rubric_dim for r in held if r.outcome == CLAIM_LOST_OUTCOME} == {f"{RULES}_grounded_share"}
    assert {r.rubric_dim: r.value for r in held}[f"{RULES}_false_per_unit"] == 1.0, "truth stands on a ground loss"

    pending = base.model_copy(update={"claim_tallies": {}, "claim_failures": {RULES: "trace_pending"}})
    held = _rows(pending, run, sibling)
    assert {r.rubric_dim for r in held if r.outcome == CLAIM_PENDING_OUTCOME} == ranked
    assert not any(r.outcome == CLAIM_LOST_OUTCOME or r.rubric_dim == f"{RULES}_lost" for r in held)


def test_a_claim_reading_never_enters_a_trial_s_pass_or_its_composite() -> None:
    runs, results = _campaign()
    result = results[runs[0].id][0].model_copy(
        update={
            "rubric_scores": make_eval_result().rubric_scores,
            "goal_state_outcomes": make_eval_result().goal_state_outcomes,
        }
    )
    bare = result.model_copy(update={"claim_tallies": {}, "claim_failures": {}})
    lost = result.model_copy(update={"claim_tallies": {}, "claim_failures": {RULES: "deadline", SCOUT: "cost_cap"}})

    for rubric_threshold in (1, 3, 5):
        trials = {read_trial(r, rubric_threshold=rubric_threshold, catalog={}) for r in (result, bare, lost)}
        assert len(trials) == 1, "neither a tally nor a loss moves how the trial ends"
    assert result_composite(result) == result_composite(bare) == result_composite(lost)


# --- the bundle -----------------------------------------------------------------------------------------------------


def test_the_three_ranked_readings_are_judged_measures_named_by_the_unit() -> None:
    bundle = _ranked()
    measures = {measure.name: measure for measure in bundle.judged_measures}

    assert set(measures) == {
        f"{RULES}_false_per_unit",
        f"{RULES}_false_unit_share",
        f"{RULES}_grounded_share",
        f"{SCOUT}_false_per_unit",
        f"{SCOUT}_false_unit_share",
    }, "diagnostics stay in the projection; the fiction rung has no grounded share"
    false_rate = measures[f"{RULES}_false_per_unit"]
    assert (false_rate.instrument, false_rate.higher_is_better, false_rate.value_range) == (RULES, False, None)
    assert false_rate.reader_name == f"False claims per answer ({RULES})"
    assert measures[f"{SCOUT}_false_unit_share"].reader_name == f"Share of entries with a false claim ({SCOUT})"
    grounded = measures[f"{RULES}_grounded_share"]
    assert (grounded.higher_is_better, grounded.value_range) == (True, (0.0, 1.0))
    memory = next(arm for arm in grounded.arms if arm.variant_key == fixture_variant_key(MEMORY))
    assert memory.mean == 0.0, "an arm that read no document grounds nothing, and reads 0, not no reading"


def test_the_family_ranks_the_arm_that_looked_the_rules_up_on_truth_and_on_groundedness() -> None:
    (family,) = _ranked().multiple_comparisons.families
    by_name = {comparison.name: comparison for comparison in family.comparisons if comparison.reading == "judged"}

    assert by_name[f"{RULES}_false_per_unit"].verdict == "improved", "fewer false claims per answer, lower is better"
    assert by_name[f"{RULES}_grounded_share"].verdict == "improved", "more of its specific claims grounded"
    assert by_name[f"{RULES}_false_per_unit"].basis != "bounded", "an unbounded rate is read on the t-path"
    assert by_name[f"{RULES}_grounded_share"].basis == "bounded", "a share takes the bounded test"
    assert not any(name.endswith(("_claims", "_lost", "_hedged_share")) for name in by_name), "no diagnostic ranks"


def test_a_claim_reading_s_judge_is_its_question_under_its_fingerprint() -> None:
    runs, results = _campaign()
    two = results[runs[0].id][0].model_copy(
        update={
            "claim_tallies": {RULES: _tally(by_judge=(_counts(2, specific=2), _counts(2, specific=2, model="judge-b")))}
        }
    )

    keys = claim_judge_keys(two, f"{RULES}_grounded_share", criterion=RULES)
    assert [(key.rubric_dim, key.scale, key.judge_model, key.judge_config_id) for key in keys] == [
        (f"{RULES}.grounded", "pass_fail", "judge-a", f"claims:{RULES}@{_SHA[:12]}"),
        (f"{RULES}.grounded", "pass_fail", "judge-b", f"claims:{RULES}@{_SHA[:12]}"),
    ]
    assert claim_judge_keys(two, f"{RULES}_false_per_unit", criterion=RULES)[0].rubric_dim == f"{RULES}.true"


# --- losses ---------------------------------------------------------------------------------------------------------


def _lookup_losing(loss: str, *, share: int) -> Any:
    """The lookup arm with one cell in ``share`` lost to ``loss`` (a criterion's loss, the ground step's, or pending)."""

    def cell(case: int, k: int) -> dict[str, Any]:
        held = _lookup_cell(case, k)
        if (case * K + k) % share:
            return held
        if loss == "ground":
            held["claim_tallies"][RULES] = _tally(ground_loss="unreadable")
            return held
        del held["claim_tallies"][RULES]
        return {**held, "claim_failures": {RULES: loss}}

    return cell


def test_a_lopsided_loss_withholds_the_reading_and_keeps_it_out_of_the_family() -> None:
    bundle = _bundle(lookup=_lookup_losing("cut_short", share=2))
    measures = {measure.name: measure for measure in bundle.judged_measures}
    (family,) = bundle.multiple_comparisons.families
    tested = {comparison.name for comparison in family.comparisons}

    for name in (f"{RULES}_false_per_unit", f"{RULES}_false_unit_share", f"{RULES}_grounded_share"):
        held = measures[name]
        assert held.withheld is not None and (held.withheld.state, held.withheld.reason) == ("withheld", "trials_lost")
        assert name not in tested
        lookup = next(arm for arm in held.arms if arm.variant_key == fixture_variant_key(LOOKUP))
        assert lookup.n_claim_lost == CASES * K // 2 and lookup.n == CASES * K // 2
    assert measures[f"{SCOUT}_false_per_unit"].withheld is None, "another criterion keeps its cells"


def test_a_withheld_claim_reading_raises_its_withheld_caveat_on_each_arm() -> None:
    built = report_input(campaign_bundle(_bundle(lookup=_lookup_losing("cut_short", share=2))))
    withheld = [
        caveat
        for caveat in built.caveat_list
        if caveat.state == "withheld"
        and caveat.reason == "trials_lost"
        and any(f"/{RULES}_false_per_unit" in r for r in caveat.readings)
    ]

    assert withheld and withheld[0].cls == "A", "a withheld reading changed the answer"


def test_a_lost_ground_step_withholds_the_grounded_share_alone() -> None:
    bundle = _bundle(lookup=_lookup_losing("ground", share=2))
    measures = {measure.name: measure for measure in bundle.judged_measures}

    assert measures[f"{RULES}_grounded_share"].withheld is not None
    assert measures[f"{RULES}_false_per_unit"].withheld is None
    assert measures[f"{RULES}_false_unit_share"].withheld is None


def test_a_pending_tally_is_counted_apart_and_never_as_a_loss() -> None:
    bundle = _bundle(lookup=_lookup_losing("trace_pending", share=2))
    measure = next(m for m in bundle.judged_measures if m.name == f"{RULES}_false_per_unit")
    lookup = next(arm for arm in measure.arms if arm.variant_key == fixture_variant_key(LOOKUP))

    assert (lookup.n_claim_pending, lookup.n_claim_lost) == (CASES * K // 2, 0)
    assert measure.withheld is None, "a population not yet known is no loss, so nothing is withheld for it"


def _all_on_one_arm(cases: int) -> bool:
    """Whether a reading is withheld where one arm lost its one cell in each of ``cases`` cases and the other none."""
    runs = [
        make_eval_run(status="completed", candidate_model=model, template_id="tpl-rules") for model in (MEMORY, LOOKUP)
    ]
    results = [
        _result(
            run,
            model,
            case,
            1,
            **(
                {"claim_tallies": {}, "claim_failures": {RULES: "cut_short"}}
                if model == LOOKUP
                else {"claim_tallies": {RULES: _tally(grounded=3)}}
            ),
        )
        for run, model in zip(runs, (MEMORY, LOOKUP), strict=True)
        for case in range(cases)
    ]
    records = project_score_records(runs, results, archived_run_ids=None, profile=toyhost_profile()).records
    by_cell = {(fixture_variant_key(m), "rig"): [r for r in results if r.model == m] for m in (MEMORY, LOOKUP)}
    measured = judged_measures(records, by_cell, None, tiers=[])
    return next(m for m in measured if m.name == f"{RULES}_false_per_unit").withheld is not None


def test_few_losses_are_read_by_their_own_conditional_law() -> None:
    """Four cells lost in four cases, all on one arm: the chi-square law reads p 0.046 and would withhold the reading,
    and the exact chance of so lopsided a split is 1/8, which the permutation p reads (0.109 over its 999 draws), so the
    reading stands. Six such cells (exact chance 1/32) are withheld. The error rates at scale are certified
    (``claim-loss.*`` in ``stats_certificate.py``), not run here."""
    assert not _all_on_one_arm(4)
    assert _all_on_one_arm(6)


def test_a_pivot_pools_one_claim_reading_on_its_own_range_and_refuses_two() -> None:
    runs, results = _campaign()
    records = project_score_records(
        runs, [r for rs in results.values() for r in rs], archived_run_ids=None, profile=toyhost_profile()
    ).records
    profile = toyhost_profile()

    share = compute_pivot(
        records,
        row_factor="model",
        column_factor="run_id",
        metric="score",
        filters={"rubric_dim": f"{RULES}_grounded_share"},
        profile=profile,
    )
    assert share.measure.value_range == (0.0, 1.0)
    rate = compute_pivot(
        records,
        row_factor="model",
        column_factor="run_id",
        metric="score",
        filters={"rubric_dim": f"{RULES}_false_per_unit"},
        profile=profile,
    )
    assert rate.measure.value_range is None, "a per-unit rate has no range, and no rubric scale's"
    with pytest.raises(PivotError, match="claim readings"):
        compute_pivot(
            records,
            row_factor="model",
            column_factor="run_id",
            metric="score",
            filters={"instrument": RULES},
            profile=profile,
        )


# --- bars, the page and the caveats ---------------------------------------------------------------------------------


def test_no_bar_is_read_on_a_claim_reading_ranked_diagnostic_or_withheld() -> None:
    """A claim reading ranks arms on its own: the resolver knows it as one, never as a rubric dimension, and refuses
    every bar on it, so no bar is cleared on a reading, least of all on one the engine withheld."""
    claimed = template_claim_readings(_template())
    measures = toyhost_profile().measures

    for name in (f"{RULES}_false_per_unit", f"{RULES}_grounded_share", f"{RULES}_claims"):
        refused = resolve_bar_name(
            name, rubric_dimensions={}, goal_state_checks=(), measures=measures, claim_readings=claimed
        )
        assert isinstance(refused, UnreadableBarName) and refused.refusal == "ranks_only"
    unknown = resolve_bar_name(f"{RULES}_false_per_unit", rubric_dimensions={}, goal_state_checks=(), measures=measures)
    assert isinstance(unknown, UnreadableBarName) and unknown.refusal == "not_carried"

    runs, results = _campaign(lookup=_lookup_losing("cut_short", share=2))
    every = [result for held in results.values() for result in held]
    records = project_score_records(runs, every, archived_run_ids=None, profile=toyhost_profile()).records
    by_cell = {(fixture_variant_key(m), "rig"): [r for r in every if r.model == m] for m in (MEMORY, LOOKUP)}
    assert next(
        m for m in judged_measures(records, by_cell, None, tiers=[]) if m.name == f"{RULES}_false_per_unit"
    ).withheld
    design = minimal_aspect_design().model_copy(
        update={"bars": [BarOverride(measure_id=f"{RULES}_false_per_unit", threshold=5.0, direction="lower_is_better")]}
    )
    (adjudication,) = bar_adjudications(
        "b", design, by_cell, records, tiers=[], frontier_bar=None, profile=toyhost_profile()
    )
    assert adjudication.state != "adjudicated" and not adjudication.verdicts, "a withheld claim reading clears no bar"


def test_no_claim_reading_is_a_floor_s_pass_rate() -> None:
    """A floor reads pass rates; a share of claims or units is none, so no claim reading is among a floor's rates."""
    bundle = _ranked()
    rates = floor_readings(bundle, fixture_variant_key(MEMORY), fixture_variant_key(LOOKUP)).rates

    assert rates, "the trials' pass rate is still read"
    assert not [rate.measure for rate in rates if (rate.measure or "").startswith((RULES, SCOUT))]


def test_an_arm_that_lost_every_cell_counts_its_losses_on_the_grounded_share_too() -> None:
    def lost(case: int, k: int) -> dict[str, Any]:
        return {"claim_tallies": {SCOUT: _tally(grounded=None)}, "claim_failures": {RULES: "cost_cap"}}

    bundle = _bundle(lookup=lost)
    grounded = next(m for m in bundle.judged_measures if m.name == f"{RULES}_grounded_share")
    lookup = next(arm for arm in grounded.arms if arm.variant_key == fixture_variant_key(LOOKUP))

    assert (lookup.n, lookup.n_claim_lost) == (0, CASES * K)
    assert grounded.withheld is not None, "an arm that lost everything differs from one that lost nothing"
    built = report_input(campaign_bundle(bundle))
    lost_caveat = next(c for c in built.caveat_list if c.id == f"claims_lost:{RULES}")
    assert f"rules/{RULES}_grounded_share" in lost_caveat.readings and len(lost_caveat.arms) == 1


def test_the_aspect_page_names_a_claim_reading_as_the_report_does_and_marks_a_withheld_one() -> None:
    """The code page over a claims campaign: each claim reading headed by its reader name, never called a score, and a
    withheld one said so."""
    ranked = report_markdown(
        build_code_only_report(_ranked(), assembled_at="2026-11-22T00:00:00+00:00", campaign_name="c")
    )
    assert "False claims per answer (rules.claims, judged)" in ranked
    assert "Grounded share of specific claims (rules.claims, judged)" in ranked
    assert "Rules.claims false per unit" not in ranked and "rules.claims_false_per_unit (judged)" not in ranked
    claim_lines = [
        line for line in ranked.splitlines() if "(rules.claims, judged)" in line or "(scout.claims, judged)" in line
    ]
    assert claim_lines and not any("score" in line.lower() for line in claim_lines)

    uneven = _bundle(lookup=_lookup_losing("cut_short", share=2), strata=True)
    page = report_markdown(build_code_only_report(uneven, assembled_at="2026-11-22T00:00:00+00:00", campaign_name="c"))
    assert "Withheld: " in page and "False claims per answer (rules.claims, judged)" in page
    assert "lost cells to claim grading at shares that differ beyond chance" in page
    strata_rows = [line for line in page.splitlines() if line.startswith("|") and "(rules.claims, judged)" in line]
    assert strata_rows and all("withheld: its arms lost cells unevenly" in line for line in strata_rows), (
        "the strata table marks a withheld reading in place of every figure"
    )


def test_the_page_calls_a_per_unit_rate_a_number_and_a_share_a_share_never_a_score() -> None:
    built = report_input(campaign_bundle(_ranked()))

    assert figure_unit(built, f"{RULES}_false_per_unit", kind="judged") == ""
    assert figure_unit(built, f"{RULES}_grounded_share", kind="judged") == "share"
    assert built.judged[f"{RULES}_false_per_unit"].label == f"False claims per answer ({RULES})"
    assert built.judged[f"{RULES}_false_per_unit"].instrument == RULES


def test_losses_pending_cells_and_skipped_web_checks_each_raise_a_class_b_caveat_naming_the_arm() -> None:
    def lookup(case: int, k: int) -> dict[str, Any]:
        held = _lookup_losing("cut_short", share=6)(case, k)
        if case == 1 and k == 1:
            del held["claim_tallies"][RULES]
            held["claim_failures"] = {RULES: "trace_pending"}
        if case == 2 and k == 1:
            held["claim_tallies"][RULES] = _tally(grounded=3, skipped=1)
        if case == 3 and k == 1:
            held["claim_tallies"][RULES] = _tally(grounded=3, skipped=1, web_loss="port_error")
        return held

    built = report_input(campaign_bundle(_bundle(lookup=lookup)))
    caveats = {caveat.id: caveat for caveat in built.caveat_list}
    lookup_label = built.judged[f"{RULES}_false_per_unit"].lost and next(
        iter(built.judged[f"{RULES}_false_per_unit"].lost)
    )

    for kind in ("claims_lost", "claims_pending", "claims_web_skipped", "claims_web_lost"):
        caveat = caveats[f"{kind}:{RULES}"]
        assert (caveat.kind, caveat.cls) == (kind, "B")
        assert caveat.arms == [lookup_label], "only the arm whose cells it touches"
    assert f"claims_lost:{SCOUT}" not in caveats


def test_a_claim_the_web_never_checked_is_never_a_false_claim() -> None:
    """A claim skipped past the web budget, or left undecided by a web step that failed, reads unverifiable: the truth
    readings of a cell with one are those of the same cell with it unflagged."""
    runs, results = _campaign()
    run, result = runs[1], results[runs[1].id][0]
    skipped = result.model_copy(update={"claim_tallies": {RULES: _tally(false=1, grounded=3, skipped=2)}})
    checked = result.model_copy(update={"claim_tallies": {RULES: _tally(false=1, grounded=3)}})
    truth = {f"{RULES}_false_per_unit", f"{RULES}_false_unit_share", f"{RULES}_false", f"{RULES}_false_share"}

    def read(cell: EvalResult) -> dict[str | None, float | None]:
        return {r.rubric_dim: r.value for r in _rows(cell, run) if r.rubric_dim in truth}

    assert read(skipped) == read(checked)
    lost = result.model_copy(
        update={"claim_tallies": {RULES: _tally(false=1, grounded=3, skipped=2, web_loss="port_error")}}
    )
    assert read(lost) == read(checked), "a claim a failed web step left undecided reads unverifiable, never false"
    assert {r.rubric_dim: r.value for r in _rows(skipped, run)}[f"{RULES}_web_skipped"] == 2.0


def test_a_claims_judge_with_no_check_ranks_among_the_unchecked_judges() -> None:
    built = report_input(campaign_bundle(_ranked()))
    uncalibrated = [caveat for caveat in built.caveat_list if caveat.cls == "C"]

    assert uncalibrated, "a claim criterion's judge is checked against nothing yet"
    for caveat in uncalibrated:
        rank = next(row.rank for row in RANKS if row.places(caveat))
        assert rank in (5, 6)


def test_a_cell_only_claims_judged_never_reads_as_unjudged() -> None:
    runs, results = _campaign()
    graded = results[runs[0].id][0]
    condition = resolve_result_condition(graded)
    assert condition.judging == "scored"
    assert "No judge produced output" not in (condition.disclosure or "")

    lost = resolve_result_condition(
        graded.model_copy(update={"claim_tallies": {}, "claim_failures": {RULES: "deadline", SCOUT: "cut_short"}})
    )
    assert lost.judging == "failed"
    assert "Claim grading lost this result on rules.claims (deadline), scout.claims (cut short)" in (
        lost.disclosure or ""
    )

    pending = resolve_result_condition(
        graded.model_copy(update={"claim_tallies": {}, "claim_failures": {RULES: "trace_pending"}})
    )
    assert pending.judging == "scored"
    assert "not yet known; that is no loss" in (pending.disclosure or "")


def test_no_host_name_reaches_a_claim_reading_s_words() -> None:
    measures = _ranked().judged_measures
    words = " ".join(str(measure.reader_name) for measure in measures).lower()
    assert "stop" not in words and "track" not in words

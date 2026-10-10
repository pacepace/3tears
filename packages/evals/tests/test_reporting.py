"""Tests for the query-time projection that backs every read-tier surface.

The load-bearing test in this file is `TestSubjectPartition` — it pins the rule
that composites are never pooled across subjects. That rule is a data-model
obligation rather than a rendering one, so it is tested at the projection, where
violating it would have to be a deliberate act rather than an oversight in one
surface out of five.
"""

import csv
import io
import json
import logging
import re

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import reporting
from threetears.evals.analysis.reporting import (
    FrontierResult,
    BADGE_CASE_SET_DIFFERS,
    BADGE_CASE_SET_UNRESOLVED,
    BADGE_CASSETTE_MODE_DIFFERS,
    BADGE_CONTEXT_DIFFERS,
    BADGE_CONTEXT_INCOMPLETE,
    BADGE_MEASUREMENT_WINDOWS_DISJOINT,
    BADGE_ROLES_DIFFER,
    BADGE_TOOL_CONFIG_DIFFERS,
    CASSETTE_SPAN_CLAUSE,
    CELL_MEASURED,
    CELL_UNMEASURED,
    CELL_WITHHELD,
    METRIC_COMPOSITE,
    METRIC_COST_USD,
    METRIC_GOAL_STATE,
    METRIC_OUTCOME,
    METRIC_SCORE,
    METRIC_TOTAL_MS,
    METRIC_TRANSCRIPT,
    PARTITION_TOLERANCE_MS,
    WEIGHTING_EQUAL_PER_SCENARIO,
    WEIGHTING_SAMPLE_WEIGHTED,
    WITHHELD_PARTS_EXCEED_WHOLE,
    WITHHELD_UNMEASURED_COMPONENT,
    ComparisonSet,
    CostEstimateError,
    ExportError,
    FrontierError,
    HistoryError,
    LatencyPartition,
    PivotError,
    PredictedValue,
    ScoreProjection,
    ScoreRecord,
    cassette_mode_disclosure,
    compute_comparison_sets,
    compute_estimate_cost,
    compute_frontier,
    compute_history,
    compute_orphaned_runs,
    compute_pivot,
    export_projection,
    compute_program_budget,
    decompose_total_ms,
    difference_was_declared_at_launch,
    dim_judge_model,
    export_records_csv,
    place_results,
    project_score_records,
    serialize_export,
)
from threetears.evals.analysis.completeness import DEGRADED_RUN_CLAUSE, completeness_disclosure
from threetears.evals.analysis.stats import UNIFORM_MOVE_NEEDS_RANGE, bounded_separation_p
from threetears.evals.kernel.host import freeze
from threetears.evals.ops import pivot_text
from threetears.evals.schema.values import SweepableValue
from threetears.evals.kernel.identity import IDENTITY_VERSION, derive_context_identity, resolve_context_identity
from threetears.evals.kernel.metrics import METRIC_DESCRIPTORS, describe_measure
from threetears.evals.schema.models import (
    OUTCOME_DIM_ID,
    TRANSCRIPT_DIM_ID,
    AsyncDelivery,
    EvalRun,
    GoalStateOutcome,
    LatencyMetrics,
    RoleUsage,
    RubricScore,
    RunCompleteness,
)
from packages.evals.tests.factories import make_eval_result, make_eval_run, make_subject

# The canary's own noun set, so the two "names no host" assertions below cover the whole
# class rather than the one noun that happened to be wrong. `reporting.py` is DECLARED
# shared contract, but its operator strings are built in function bodies, which that canary
# deliberately does not scan — these assertions are the only thing standing there.
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND
from packages.evals.tests.fixtures.toyhost.contract import TOY_EXTRACTOR_CONTRACT
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.corpus import toyhost_batch, toyhost_measurements
from packages.evals.tests.host_vocabulary import HOST_NOUNS

#: Whole-word host nouns in PROSE. The list is imported, never retyped; only the matching
#: differs, and it has to. The canary splits identifiers into parts because `\b` cannot see
#: a noun inside `<noun>_snapshot` (an underscore is a word character). Prose is the other
#: way round: splitting breaks an all-capitals plural into fragments and the noun vanishes, while `\b` finds
#: it and still refuses the substring in `adjacent`, which is what the whole-word rule is
#: for. Verified by mutation — such a plural in a sentence passed the split-based check.
_HOST_NOUN_IN_PROSE = re.compile(rf"\b(?:{'|'.join(HOST_NOUNS)})s?\b", re.IGNORECASE)


#: The host every projection here reads its vocabulary through: the toy host -- a second,
#: non-conversational host -- with its apparatus declarations cleared. The toy host has no judge and no
#: simulated user, and these tests are about a host that has both, so a run recording no simulated
#: user is an UNRECORDED input here (partial context) rather than one the host declared it never
#: had. A run naming no judge was not judged, which is a recorded level on any host.
_JUDGED_HOST = toyhost_profile(every_seat=True)


def _catalog_names_of(row_name):
    """Every registered measure name the surfaces resolve to ``row_name``, other than itself.

    The catalog (aggregate) names a row measure is published under, read through the public
    alias rather than the table behind it: ``resolve_measure_name`` is the direction the
    surfaces apply, and the seeded registry is what ``list_metrics`` publishes.
    """
    return sorted(
        name for name in METRIC_DESCRIPTORS if name != row_name and reporting.resolve_measure_name(name) == row_name
    )


def _catalog_name_of(row_name):
    """The ONE catalog name ``row_name`` is published under; none, or two, is a failure here."""
    names = _catalog_names_of(row_name)
    assert len(names) == 1, f"{row_name!r} resolves from {names} -- it needs exactly one registered catalog name"
    return names[0]


def _run_with_results(*, subject_id="ent-maple", subject_label="Maple", **run_overrides):
    """Build one run plus a matching result, wired by run id.

    Args:
        subject_id: Subject id for the run's subject snapshot.
        subject_label: Display name for the run's subject snapshot.
        **run_overrides: Passed through to ``make_eval_run``.

    Returns:
        Tuple of (run, result).
    """
    snapshot = make_subject(subject_id, subject_label)
    run = make_eval_run(subject_snapshot=snapshot, **run_overrides)
    result = make_eval_result(eval_run_id=run.id, scope_id=run.scope_id)
    return run, result


def _overlaid(**overlays):
    """The run fields a launch of the toy extractor stamps for ``overlays``: its kind and the frozen model.

    Args:
        **overlays: The knobs the launch turned, by field of the kind's overlay model.

    Returns:
        ``candidate_kind`` and ``overlays``, for ``make_eval_run``.
    """
    validated = TOY_EXTRACTOR_CONTRACT.validate_overlays(overlays)
    return {"candidate_kind": TOY_EXTRACTOR_KIND, "overlays": freeze(validated)}


def _stamped_run(**run_overrides):
    """Build a run stamped the way the launch path stamps one.

    Key, components and version together, over both role pins — a run carrying only
    a hand-written ``context_key`` is not stamped, and every read surface correctly
    re-derives it. Tests that want two runs to agree (or differ) on their measurement
    context have to go through the predicate to be testing anything.

    Args:
        **run_overrides: Passed through to ``make_eval_run``.

    Returns:
        The stamped run.
    """
    run, _ = _run_with_results(judge_model="judge-x", simulator_model="sim-x", **run_overrides)
    identity = derive_context_identity(run, _JUDGED_HOST)
    run.context_key = identity.context_key
    run.context_components = identity.context_components
    run.identity_version = identity.identity_version
    return run


class TestProjectionCoordinates:
    """Every row carries the coordinates a surface needs to place it."""

    def test_row_carries_subject_and_cell_coordinates(self):
        run, result = _run_with_results()

        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        assert records, "expected at least the cost row"
        row = records[0]
        assert row.subject_id == "ent-maple"
        assert row.subject_label == "Maple"
        assert row.run_id == run.id
        assert row.result_id == result.id
        assert row.test_case_id == result.test_case_id
        assert row.model == result.model
        assert row.k_iteration == result.k_iteration

    def test_cost_row_emitted_for_every_projected_result(self):
        run, result = _run_with_results()

        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        cost_rows = [r for r in records if r.metric == METRIC_COST_USD]
        assert len(cost_rows) == 1
        assert cost_rows[0].value == result.cost_usd

    def test_outcome_is_carried_from_the_scoring_taxonomy(self):
        run, _ = _run_with_results()
        failed = make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, candidate_error="402")

        records = project_score_records([run], [failed], profile=_JUDGED_HOST, archived_run_ids=None).records

        assert {r.outcome for r in records} == {"candidate_fail"}

    def test_a_judge_that_could_not_tell_marks_the_composite_and_its_own_rows(self):
        """The rows a counter reads to count the exclusion — composite, rubric dim and reserved axis."""
        run, _ = _run_with_results(rubric_scales={"reply.tone": "ordinal", "reply.refusal": "ordinal"})
        untold = make_eval_result(
            eval_run_id=run.id,
            scope_id=run.scope_id,
            rubric_scores=[RubricScore(dim="reply.tone", score=4, scale="ordinal")],
            judge_cannot_tell={"reply.refusal": "nothing to refuse", TRANSCRIPT_DIM_ID: "no turns"},
        )

        records = project_score_records([run], [untold], profile=_JUDGED_HOST, archived_run_ids=None).records

        (composite,) = [r for r in records if r.metric == METRIC_COMPOSITE]
        assert (composite.value, composite.outcome) == (None, "judge_cannot_tell")
        (refusal,) = [r for r in records if r.metric == METRIC_SCORE and r.rubric_dim == "reply.refusal"]
        assert (refusal.value, refusal.outcome) == (None, "judge_cannot_tell")
        (axis,) = [r for r in records if r.metric == TRANSCRIPT_DIM_ID]
        assert (axis.value, axis.outcome) == (None, "judge_cannot_tell")
        # The dim the judge did score keeps the result's own outcome.
        (tone,) = [r for r in records if r.metric == METRIC_SCORE and r.rubric_dim == "reply.tone"]
        assert (tone.value, tone.outcome) == (4.0, "ok")

    def test_cost_is_projected_even_for_a_failed_result(self):
        """Spend is real regardless of outcome — budget views must still see it."""
        run, _ = _run_with_results()
        failed = make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, infra_error="timeout", cost_usd=0.42)

        records = project_score_records([run], [failed], profile=_JUDGED_HOST, archived_run_ids=None).records

        cost_rows = [r for r in records if r.metric == METRIC_COST_USD]
        assert [r.value for r in cost_rows] == [0.42]

    def test_result_without_its_run_is_skipped_not_invented(self):
        _, orphan = _run_with_results()

        assert project_score_records([], [orphan], profile=_JUDGED_HOST, archived_run_ids=None).records == []

    def test_a_kinds_overlays_reach_the_row_as_dotted_coordinates(self):
        """Without this the factor set is closed, and a bake-off's varied knob is unpivotable.

        Read through the host's registry: every field of the kind's model at the level the run froze
        (defaults included, so a knob the launch left alone is a coordinate too), and each entry of
        an open family as its own.
        """
        run, result = _run_with_results(
            **_overlaid(prompt_style="verbose", page_limit=3, field_aliases={"vendor_name": "supplier"})
        )

        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        assert records[0].factors == {
            "candidate_kind": TOY_EXTRACTOR_KIND,
            "extractor.prompt_style": "verbose",
            "extractor.page_limit": "3",
            "extractor.instructions": "",
            "extractor.field_aliases": '{"vendor_name":"supplier"}',
            "extractor.field_aliases.vendor_name": "supplier",
        }

    def test_a_run_with_no_overlays_carries_no_invented_coordinates(self):
        """No overlay map is "ran unmodified" — a default value here would be a fabricated cohort.

        The kind it ran as is a coordinate every run carries, not an invented one.
        """
        run, result = _run_with_results()

        assert project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records[
            0
        ].factors == {"candidate_kind": run.candidate_kind}

    def test_row_carries_the_pinned_judge_and_simulator_roles(self):
        """The run-pinned roles are first-class coordinates, not only hashed into `context_key`.

        Sourced run-level — the roles are pinned once per run, exactly as the
        `context_key` roles component composes them — so a pivot on `judge_model`
        partitions on the same value the comparability badge reads.
        """
        run, result = _run_with_results(judge_model="judge-a", simulator_model="sim-b")

        row = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records[0]

        assert (row.judge_model, row.simulator_model) == ("judge-a", "sim-b")

    def test_absent_pinned_roles_are_carried_as_none_not_invented(self):
        """A run that pinned no judge (it was not judged) or no simulated user carries None, groupable as "—"."""
        run, result = _run_with_results(judge_model=None, simulator_model=None)

        row = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records[0]

        assert (row.judge_model, row.simulator_model) == (None, None)

    def test_each_identity_key_carries_its_own_predicate_version(self):
        """Pins which source feeds each coordinate, and that the two never cross.

        ``variant_key`` and ``context_key`` are produced by separately numbered
        predicates. If both were qualified by one version field, a consumer
        grouping on ``context_key`` would gate comparability on the *variant*
        predicate's number and regroup keys the run-level version exists to keep
        distinct. Distinct values here are what make a crossed wiring fail.
        """
        run, result = _run_with_results(context_key="ctx-abc", identity_version=2)
        result.variant_key = "var-xyz"
        result.identity_version = 7

        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        assert records, "expected at least the cost row"
        row = records[0]
        assert (row.variant_key, row.variant_identity_version) == ("var-xyz", 7)
        assert (row.context_key, row.context_identity_version) == ("ctx-abc", 2)


class TestCompositeIsPerObservation:
    """A row is one measurement at one cell — never a k-average wearing k labels."""

    def _scored(self, run, *, k_iteration, score):
        """Build a scored result for one k-iteration.

        Args:
            run: The run the result belongs to.
            k_iteration: Which iteration this observation is.
            score: Rubric score, 1-5.

        Returns:
            An `EvalResult` carrying a single rubric dim at `score`.
        """
        return make_eval_result(
            eval_run_id=run.id,
            scope_id=run.scope_id,
            test_case_id="tc-1",
            k_iteration=k_iteration,
            rubric_scores=[RubricScore(dim="reply.quality", score=score, scale="ordinal")],
        )

    def test_iterations_of_one_case_keep_their_own_values(self):
        """With k>1 the rows must differ, or dispersion across k reads as zero."""
        run, _ = _run_with_results()
        results = [self._scored(run, k_iteration=1, score=5), self._scored(run, k_iteration=3, score=1)]

        composites = {
            r.k_iteration: r.value
            for r in project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None).records
            if r.metric == METRIC_COMPOSITE
        }

        assert composites == {1: 1.0, 3: 0.0}

    def test_infra_excluded_result_scores_null_rather_than_zero(self):
        """Excluded is unmeasured, not zero — a null cannot enter a quality mean."""
        run, _ = _run_with_results()
        excluded = make_eval_result(
            eval_run_id=run.id,
            scope_id=run.scope_id,
            infra_error="judge timeout",
            rubric_scores=[RubricScore(dim="reply.quality", score=4, scale="ordinal")],
        )

        composites = [
            r.value
            for r in project_score_records([run], [excluded], profile=_JUDGED_HOST, archived_run_ids=None).records
            if r.metric == METRIC_COMPOSITE
        ]

        assert composites == [None], "a scored-then-infra-failed result must not contribute a number"

    def test_a_result_with_no_rubric_dims_scores_null_rather_than_vanishing(self):
        """Nothing measured it, but it was attempted — the row is what records the attempt."""
        run, _ = _run_with_results()
        unscored = make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, rubric_scores=[])

        composites = [
            r.value
            for r in project_score_records([run], [unscored], profile=_JUDGED_HOST, archived_run_ids=None).records
            if r.metric == METRIC_COMPOSITE
        ]

        assert composites == [None]

    def test_infra_excluded_result_still_reports_its_cost(self):
        """The harness failed; the spend was still real.

        Exhaustive over the emitted rows deliberately: the assertion is that
        cost survives an exclusion *and* that nothing else quietly appears
        beside it. The default fixture carries one judged dim and one passing goal-state
        check. Both rows are null-valued, as the composite is: a check or a judge score on a
        harness-faulted result read the harness, and pivot counting either would give a figure
        the bundle and the bars do not.
        """
        run, _ = _run_with_results()
        excluded = make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, infra_error="boom", cost_usd=0.31)

        records = project_score_records([run], [excluded], profile=_JUDGED_HOST, archived_run_ids=None).records

        assert [(r.metric, r.value) for r in records] == [
            (METRIC_COMPOSITE, None),
            (METRIC_COST_USD, 0.31),
            (METRIC_SCORE, None),
            (METRIC_GOAL_STATE, None),
        ]

    def test_candidate_failure_scores_zero_rather_than_vanishing(self):
        run, _ = _run_with_results()
        failed = make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, candidate_error="402")

        composites = [
            r.value
            for r in project_score_records([run], [failed], profile=_JUDGED_HOST, archived_run_ids=None).records
            if r.metric == METRIC_COMPOSITE
        ]

        assert composites == [0.0]


class TestGoalStateRows:
    """Each goal-state check a result evaluated is one row, keyed by the check, 1.0 passed and 0.0 not."""

    def test_one_row_per_check_with_the_check_as_its_coordinate(self):
        from threetears.evals.schema.models import GoalStateOutcome

        run, _ = _run_with_results()
        result = make_eval_result(
            eval_run_id=run.id,
            scope_id=run.scope_id,
            goal_state_outcomes=[
                GoalStateOutcome(expression='call_count("orders.place_order") >= 1', passed=False),
                GoalStateOutcome(expression='call_count("chat.send_message") >= 1', passed=True),
            ],
        )

        rows = [
            r
            for r in project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records
            if r.metric == METRIC_GOAL_STATE
        ]

        assert [(r.goal_check, r.value) for r in rows] == [
            ('call_count("orders.place_order") >= 1', 0.0),
            ('call_count("chat.send_message") >= 1', 1.0),
        ]
        assert all(r.rubric_dim is None for r in rows), "a code-graded check never lands on the judged-dim coordinate"

    def test_a_result_that_evaluated_no_check_has_no_goal_rows(self):
        run, _ = _run_with_results()
        result = make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, goal_state_outcomes=[])

        assert not [
            r
            for r in project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records
            if r.metric == METRIC_GOAL_STATE
        ]


class TestPerDimensionRows:
    """The raw 1-5 judge score reaches the projection, one row per dimension.

    The dimension is a COORDINATE (`rubric_dim`), not a family of metric names:
    a dim named `composite` would otherwise merge dim rows and composite rows
    into one aggregate, since `pivot` selects on `metric` alone. These tests pin
    the coordinate shape, the raw scale, and the one place it deliberately
    disagrees with the composite.
    """

    def _scored(self, run, **overrides):
        """Build a result belonging to ``run``.

        Args:
            run: The run the result belongs to.
            **overrides: Passed through to ``make_eval_result``.

        Returns:
            An `EvalResult` wired to the run.
        """
        return make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, **overrides)

    def test_one_row_per_judged_dimension_carrying_the_raw_score(self):
        """The registry declares `mean_score` on the 1-5 scale — a 0-1 value here would be the composite's."""
        run, _ = _run_with_results()
        result = self._scored(
            run,
            rubric_scores=[
                RubricScore(dim="reply.grounding", score=5, scale="ordinal"),
                RubricScore(dim="reply.coverage", score=2, scale="ordinal"),
            ],
        )

        rows = [
            r
            for r in project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records
            if r.metric == METRIC_SCORE
        ]

        assert {(r.rubric_dim, r.value) for r in rows} == {("reply.grounding", 5.0), ("reply.coverage", 2.0)}

    def test_a_namespaced_dim_is_carried_whole_with_no_branching_on_its_context(self):
        """Dim names carry their scoring context as `<context>.<dim>`.

        The projection and the pivot must treat that name as one opaque
        coordinate value: nothing may split on the separator, strip the prefix,
        or group two contexts together. `planner.character` and
        `speaker.character` are DIFFERENT dimensions — collapsing them is the
        same mis-scoping the namespace was introduced to end, arriving in the
        reporting layer instead of the judge binding.
        """
        run, _ = _run_with_results()
        result = self._scored(
            run,
            rubric_scores=[
                RubricScore(dim="planner.character", score=5, scale="ordinal"),
                RubricScore(dim="speaker.character", score=1, scale="ordinal"),
            ],
        )

        rows = [
            r
            for r in project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records
            if r.metric == METRIC_SCORE
        ]

        assert {(r.rubric_dim, r.value) for r in rows} == {("planner.character", 5.0), ("speaker.character", 1.0)}

        # And through the pivot: two levels, not one `character` row averaging 3.
        table = compute_pivot(
            rows, row_factor="rubric_dim", column_factor="model", metric=METRIC_SCORE, profile=_JUDGED_HOST
        )
        assert {cell.row: cell.value for cell in table.cells} == {"planner.character": 5.0, "speaker.character": 1.0}

    def test_a_dimension_row_carries_the_same_cell_coordinates_as_its_composite(self):
        """A dim row must be placeable at exactly the cell its composite is, or the two pivots disagree about where a result sat.

        The dimension-scoped coordinates are excluded, not compared: `rubric_dim`
        and the judge that scored it are the two fields a dim row is *supposed*
        to differ on, since a composite spans every dimension and has neither.
        `dimension_basis` is excluded for the mirror-image reason — it qualifies
        the composite's VALUE, naming what that number was meaned over, and a dim
        row measures one dimension and is meaned over nothing. Neither is a
        coordinate, and everything that IS one must match exactly. That is the cell.
        """
        run, _ = _run_with_results()
        result = self._scored(run, rubric_scores=[RubricScore(dim="reply.grounding", score=5, scale="ordinal")])

        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records
        composite = next(r for r in records if r.metric == METRIC_COMPOSITE)
        dim_row = next(r for r in records if r.metric == METRIC_SCORE)

        not_a_cell_coordinate = {
            "metric",
            "value",
            "dimension_basis",
            "rubric_dim",
            "rubric_scale",
            "rubric_dim_judge_model",
        }
        assert dim_row.model_dump(exclude=not_a_cell_coordinate) == composite.model_dump(exclude=not_a_cell_coordinate)

    def test_composite_and_cost_rows_carry_no_dimension(self):
        """A whole-result measure has no dimension; inventing one would make a `rubric_dim` pivot of composites look per-dim."""
        run, result = _run_with_results()

        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        assert all(r.rubric_dim is None for r in records if r.metric != METRIC_SCORE)

    def test_a_result_with_no_dims_still_emits_one_dimensionless_null_row(self):
        """Dropping the row would make "the judge scored nothing here" read as "nobody ran this"."""
        run, _ = _run_with_results()
        unscored = self._scored(run, rubric_scores=[])

        rows = [
            r
            for r in project_score_records([run], [unscored], profile=_JUDGED_HOST, archived_run_ids=None).records
            if r.metric == METRIC_SCORE
        ]

        assert [(r.rubric_dim, r.value) for r in rows] == [(None, None)]

    def test_a_candidate_failure_counts_each_judged_dim_at_its_scales_floor(self):
        """What the end user got, not what the judge made of it.

        A configuration that delivered no turn scores the bottom of every dimension's own scale —
        1 on 1-5, 0 on pass/fail — as the composite scores it 0.0. The floor is on the scale, so
        no value appears that the scale does not contain; the judge's raw reading stays on the
        result.
        """
        run, _ = _run_with_results()
        failed = self._scored(
            run,
            candidate_error="402",
            rubric_scores=[
                RubricScore(dim="reply.grounding", score=4, scale="ordinal"),
                RubricScore(dim="reply.treats_as_present", scale="pass_fail", score=1),
            ],
        )

        records = project_score_records([run], [failed], profile=_JUDGED_HOST, archived_run_ids=None).records
        composite = next(r.value for r in records if r.metric == METRIC_COMPOSITE)
        dim_rows = [(r.rubric_dim, r.value) for r in records if r.metric == METRIC_SCORE]

        assert composite == 0.0
        assert dim_rows == [("reply.grounding", 1.0), ("reply.treats_as_present", 0.0)]
        assert [s.score for s in failed.rubric_scores] == [4, 1], "the raw reading is kept on the result"

    def test_a_candidate_failure_the_judge_never_scored_emits_an_unmeasured_row(self):
        """The common shape of the case above: no dims, so the cell reads attempted-and-unmeasured, not zero."""
        run, _ = _run_with_results()
        failed = self._scored(run, candidate_error="402", rubric_scores=[])

        rows = [
            r
            for r in project_score_records([run], [failed], profile=_JUDGED_HOST, archived_run_ids=None).records
            if r.metric == METRIC_SCORE
        ]

        assert [(r.rubric_dim, r.value, r.outcome) for r in rows] == [(None, None, "candidate_fail")]

    def test_the_dimension_is_a_coordinate_not_a_metric_name(self):
        """`PROJECTED_METRICS` stays CLOSED; a template dim never enters the metric namespace.

        The closure is what lets `pivot` refuse an unknown metric as a typo. A
        metric per dim would make the set corpus-derived, and a dim named
        `composite` would then collide with the composite measure itself —
        merging two aggregates into one number with no error raised.

        Closed, not fixed in size: the two reserved dual-score axes ARE metrics, which
        is the case that prefix was minted for — `__transcript__` cannot collide with a
        template dim, and no template dim may take a `__` name. The property under test
        is that a dim the CORPUS supplies never becomes one, which is why the fixture
        names a dim after an existing measure — under a context, since a stored dim name
        is always namespaced (`DimName`) and a bare `composite` cannot be stored at all.
        """
        run, _ = _run_with_results()
        colliding = self._scored(
            run, rubric_scores=[RubricScore(dim=f"reply.{METRIC_COMPOSITE}", score=5, scale="ordinal")]
        )

        records = project_score_records([run], [colliding], profile=_JUDGED_HOST, archived_run_ids=None).records

        assert {r.metric for r in records} == {METRIC_COMPOSITE, METRIC_COST_USD, METRIC_SCORE, METRIC_GOAL_STATE}
        # The dim named after `composite` is a rubric_dim value, so the composite rows
        # stay exactly one per result and keep their own 0-1 value.
        composites = [r for r in records if r.metric == METRIC_COMPOSITE]
        assert [(r.rubric_dim, r.value) for r in composites] == [(None, 1.0)]

    def test_the_dimension_is_pivotable_as_an_axis(self):
        """`row_factor='rubric_dim'` is the whole point — every dim against every model in one table."""
        run, _ = _run_with_results()
        results = [
            self._scored(
                run,
                test_case_id=case,
                rubric_scores=[
                    RubricScore(dim="reply.grounding", score=5, scale="ordinal"),
                    RubricScore(dim="reply.coverage", score=3, scale="ordinal"),
                ],
            )
            for case in ("tc-1", "tc-2")
        ]

        records = project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None).records
        table = compute_pivot(
            records, row_factor="rubric_dim", column_factor="model", metric=METRIC_SCORE, profile=_JUDGED_HOST
        )

        assert table.rows == ["reply.coverage", "reply.grounding"]
        by_row = {cell.row: cell.value for cell in table.cells}
        assert by_row == {"reply.coverage": 3.0, "reply.grounding": 5.0}

    def _uneven_corpus(self, run):
        """Two observations of one case and one of another, plus an unscored result.

        Deliberately uneven: with one observation per case the two weightings
        agree, so a corpus like that would let a test claim an equality it never
        exercised. The unscored result is here because a dimensionless null row
        must not enter either denominator.

        Args:
            run: The run the results belong to.

        Returns:
            A list of `EvalResult` — case means 5 and 2, observation mean 4.
        """
        return [
            self._scored(
                run,
                test_case_id="tc-1",
                k_iteration=1,
                rubric_scores=[RubricScore(dim="reply.grounding", score=5, scale="ordinal")],
            ),
            self._scored(
                run,
                test_case_id="tc-1",
                k_iteration=2,
                rubric_scores=[RubricScore(dim="reply.grounding", score=5, scale="ordinal")],
            ),
            self._scored(
                run,
                test_case_id="tc-2",
                k_iteration=1,
                rubric_scores=[RubricScore(dim="reply.grounding", score=2, scale="ordinal")],
            ),
            self._scored(run, test_case_id="tc-3", k_iteration=1, rubric_scores=[]),
        ]

    def test_the_dimension_pivot_reports_the_means_run_summary_reports(self):
        """The Done-when equality — and the weighting it holds under, which is not the pivot's default.

        `compute_dimension_summary` divides by observations, so the pivot matches
        it under `sample_weighted`, not under the `equal_per_scenario` default.
        Both read `rubric_scores` alone — the reserved `__transcript__` /
        `__outcome__` axes live in their own fields — so under that weighting it
        is an equality rather than an approximation. Asserted over a corpus whose
        two weightings genuinely differ, or the equality would hold for a reason
        this test could not see.
        """
        from threetears.evals.kernel.scoring import compute_dimension_summary

        run, _ = _run_with_results()
        results = self._uneven_corpus(run)

        records = project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None).records
        table = compute_pivot(
            records,
            row_factor="rubric_dim",
            column_factor="model",
            metric=METRIC_SCORE,
            weighting=WEIGHTING_SAMPLE_WEIGHTED,
            profile=_JUDGED_HOST,
        )
        summary = compute_dimension_summary(results)

        pivoted = {cell.row: cell.value for cell in table.cells if cell.status == "measured"}
        expected = {dim: stats["mean_score"] for (_model, _run_id, dim), stats in summary.items()}
        assert pivoted == expected == {"reply.grounding": 4.0}

    def test_a_result_with_no_dims_forms_its_own_level_rather_than_joining_a_dimension(self):
        """The null row must not land inside a dimension's denominator.

        `_axis_value` places a `None` coordinate on the visible "—" level, which
        is what keeps "the judge scored nothing here" a cohort of its own. Pinned
        against the cells rather than against the projection, because the row
        existing proves nothing about where the aggregation puts it — folding it
        into a dimension would move only `n_unmeasured`, which no other
        assertion here reads.
        """
        run, _ = _run_with_results()
        results = self._uneven_corpus(run)

        records = project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None).records
        table = compute_pivot(
            records, row_factor="rubric_dim", column_factor="model", metric=METRIC_SCORE, profile=_JUDGED_HOST
        )

        assert table.rows == ["reply.grounding", "—"]
        by_row = {cell.row: cell for cell in table.cells}
        assert by_row["—"].status == CELL_UNMEASURED
        assert (by_row["—"].n, by_row["—"].n_unmeasured) == (0, 1)
        # And the dimension's own cell never saw it.
        assert (by_row["reply.grounding"].n, by_row["reply.grounding"].n_unmeasured) == (3, 0)

    def test_the_pooling_caveat_also_qualifies_the_evidence_counts_not_just_the_value(self):
        """`n` counts (result x dimension) rows when pooled, and saying so is the same obligation.

        A judged result contributes one row per dimension; an unjudged result
        contributes exactly one unmeasured row. So a cell can report a high
        measured-to-unmeasured ratio while half its RESULTS were never scored —
        a reader checking whether the number is well-evidenced would conclude it
        is. Qualifying the value and leaving its evidence unqualified is the same
        defect one field over.
        """
        run, _ = _run_with_results()
        judged = [
            self._scored(
                run,
                rubric_scores=[
                    RubricScore(dim=d, score=4, scale="ordinal")
                    for d in ("reply.grounding", "reply.coverage", "reply.honesty")
                ],
            )
            for _ in range(4)
        ]
        unjudged = [self._scored(run, rubric_scores=[]) for _ in range(4)]
        records = project_score_records([run], judged + unjudged, profile=_JUDGED_HOST, archived_run_ids=None).records

        pooled = compute_pivot(
            records, row_factor="model", column_factor="run_id", metric=METRIC_SCORE, profile=_JUDGED_HOST
        )
        on_axis = compute_pivot(
            records, row_factor="rubric_dim", column_factor="model", metric=METRIC_SCORE, profile=_JUDGED_HOST
        )

        assert "n and the outcome counts are over (result x dimension) rows here, not results" in pooled.formula
        assert "(result x dimension) rows" not in on_axis.formula, "with the dim on an axis, n IS a result count"

    @pytest.mark.parametrize(
        ("weighting", "expected"),
        [
            (
                WEIGHTING_EQUAL_PER_SCENARIO,
                "the dispersion is over test-case means — a BASIS pooling does not change, though the means themselves do",
            ),
            (
                WEIGHTING_SAMPLE_WEIGHTED,
                "the dispersion is clustered by test case, so the rows of one case are not independent draws",
            ),
        ],
    )
    def test_the_pooled_caveat_names_the_dispersion_basis_the_weighting_actually_uses(self, weighting, expected):
        """`_aggregate` branches on weighting, so one sentence cannot cover both.

        Under `equal_per_scenario` — the DEFAULT — the SEM is over test-case
        means; pooling changes those means and how many observations each averages,
        but not how many means the SEM is taken over. Saying it rides on rows would send
        an operator reconstructing an interval to `n` rather than the case count,
        giving one too narrow by ~sqrt(n/n_cases). A single generalised clause
        was written here once and was wrong for the default; both branches are
        pinned so neither can be widened into the other again.
        """
        run, _ = _run_with_results()
        result = self._scored(
            run,
            rubric_scores=[RubricScore(dim=d, score=4, scale="ordinal") for d in ("reply.grounding", "reply.coverage")],
        )
        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        pooled = compute_pivot(
            records,
            row_factor="model",
            column_factor="run_id",
            metric=METRIC_SCORE,
            weighting=weighting,
            profile=_JUDGED_HOST,
        )

        assert expected in pooled.formula

    def test_a_dimension_pooled_score_says_so_in_the_formula(self):
        """The registry calls `mean_score` the mean "for one rubric dimension" — true only when the dim is pinned.

        A caller reading the response never sees a tool tip that warns about pooling, so the
        disclosure has to ride on the response. Same obligation the weighting
        clause already discharges: an unqualified formula beside a qualified
        number lets a reader check the wrong arithmetic and conclude the table
        is right.
        """
        run, _ = _run_with_results()
        results = self._uneven_corpus(run)
        records = project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None).records

        pooled = compute_pivot(
            records, row_factor="test_case_id", column_factor="model", metric=METRIC_SCORE, profile=_JUDGED_HOST
        )
        on_axis = compute_pivot(
            records, row_factor="rubric_dim", column_factor="model", metric=METRIC_SCORE, profile=_JUDGED_HOST
        )
        filtered = compute_pivot(
            records,
            row_factor="test_case_id",
            column_factor="model",
            metric=METRIC_SCORE,
            filters={"rubric_dim": "reply.grounding"},
            profile=_JUDGED_HOST,
        )

        assert "POOLED ACROSS EVERY JUDGED RUBRIC DIMENSION" in pooled.formula
        assert "POOLED" not in on_axis.formula, "a dim on an axis IS scoped — the caveat would be false"
        assert "POOLED" not in filtered.formula, "a dim filter scopes just as an axis does"
        # The caveat names the axis and NOT the filter, though both scope the
        # cell here. `EvalService.pivot` composes `filters` itself from
        # `subject_id`, and neither surface accepts a `rubric_dim` filter — so
        # advising one would be a remedy no reader of this string can perform,
        # which is the false-capability claim this measure exists to end.
        assert "on an axis for one" in pooled.formula
        assert "filter" not in pooled.formula
        # The caveat is score-specific: a composite has no dimension to pool.
        composite = compute_pivot(
            records, row_factor="test_case_id", column_factor="model", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
        )
        assert "POOLED" not in composite.formula

    def test_the_default_weighting_gives_each_case_one_vote_not_run_summarys_number(self):
        """The sibling of the equality above: the default is a DIFFERENT number, and correctly so.

        `run_summary` weights by observation; the pivot's default weights by test
        case, so a case run twice does not count twice. Pinned because the
        equality test alone would invite a reader to expect the two surfaces to
        agree unconditionally — they agree under a stated weighting, and the
        help text that points operators here has to say so.
        """
        run, _ = _run_with_results()
        results = self._uneven_corpus(run)

        records = project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None).records
        table = compute_pivot(
            records, row_factor="rubric_dim", column_factor="model", metric=METRIC_SCORE, profile=_JUDGED_HOST
        )

        measured = [cell for cell in table.cells if cell.status == "measured"]
        assert [(cell.row, cell.value) for cell in measured] == [("reply.grounding", 3.5)]
        # Same rows underneath: three valued observations across two cases.
        assert (measured[0].n, measured[0].n_cases) == (3, 2)

    def test_a_dimension_pivot_is_refused_across_subjects(self):
        """The no-pooling rule binds dims exactly as it binds composites — `mean_score` is judge_mediated."""
        run_a, _ = _run_with_results(subject_id="ent-maple", subject_label="Maple")
        run_b, _ = _run_with_results(subject_id="ent-bea", subject_label="Bea")
        results = [
            make_eval_result(eval_run_id=run_a.id, scope_id=run_a.scope_id),
            make_eval_result(eval_run_id=run_b.id, scope_id=run_b.scope_id),
        ]

        records = project_score_records([run_a, run_b], results, profile=_JUDGED_HOST, archived_run_ids=None).records

        with pytest.raises(PivotError, match="judge_mediated"):
            compute_pivot(
                records, row_factor="rubric_dim", column_factor="model", metric=METRIC_SCORE, profile=_JUDGED_HOST
            )

    def test_the_dimension_pivots_scale_comes_from_the_registry_not_the_composite(self):
        """A cell claiming 0-1 beside a 1-5 number is how a reader mis-reads a 2 as excellent."""
        run, result = _run_with_results()

        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records
        table = compute_pivot(
            records, row_factor="rubric_dim", column_factor="model", metric=METRIC_SCORE, profile=_JUDGED_HOST
        )

        assert table.measure.name == "mean_score"
        assert table.measure.value_range == (1.0, 5.0)
        assert table.measure.transferability_class == "judge_mediated"

    def test_an_unknown_metric_is_still_refused_naming_what_is_available(self):
        """Adding a metric must not blunt the typo refusal — the closed set is what makes it safe."""
        run, result = _run_with_results()
        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        with pytest.raises(PivotError, match="unknown metric") as excinfo:
            compute_pivot(records, row_factor="model", column_factor="model", metric="scores", profile=_JUDGED_HOST)

        assert METRIC_SCORE in str(excinfo.value)
        assert "rubric dimension" not in str(excinfo.value), "a plain typo must not be explained as a dimension"

    def test_a_dimension_asked_for_as_a_metric_is_pointed_at_the_route_that_exists(self):
        """The reader wanted a real capability under the wrong noun; a bare list reads as "unavailable".

        The advice is FOLLOWED here rather than merely matched. An assertion that
        the message contains `score` and `rubric_dim` stays green over advice
        that cannot be acted on — which is how this string came to name a
        `rubric_dim` filter that no surface accepts. Executing the remedy is the
        only assertion that can tell working guidance from a plausible sentence.
        """
        run, _ = _run_with_results()
        result = self._scored(run, rubric_scores=[RubricScore(dim="reply.character", score=3, scale="ordinal")])
        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        with pytest.raises(PivotError, match="rubric dimension") as excinfo:
            compute_pivot(
                records, row_factor="model", column_factor="model", metric="reply.character", profile=_JUDGED_HOST
            )

        message = str(excinfo.value)
        assert METRIC_SCORE in message and "rubric_dim" in message
        # Only reachable remedies may be named: `EvalService.pivot` composes
        # `filters` itself, so no operator can pin a dimension by filter.
        assert "filter" not in message
        # Bind the executed remedy to the NAMED one before executing it.
        # Without this line the pivot below is a hardcoded move that happens to
        # work, so the message could name any other unreachable remedy — one
        # whose name is not the substring "filter" — and every assertion here
        # would still pass. That is the same "wider than its mechanism" defect
        # this test exists to catch, one level up.
        assert "with 'rubric_dim' on an axis" in message
        # Now do exactly what it says, and get the dimension it was asked for.
        table = compute_pivot(
            records, row_factor="rubric_dim", column_factor="model", metric=METRIC_SCORE, profile=_JUDGED_HOST
        )
        assert {cell.row: cell.value for cell in table.cells} == {"reply.character": 3.0}

    def test_the_dimension_exports_as_its_own_column(self):
        """The export derives its columns from the model, so the dim must arrive as a pandas column, not a blob."""
        run, result = _run_with_results()
        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        reader = csv.DictReader(io.StringIO(export_records_csv(records)))
        rows = list(reader)

        assert "rubric_dim" in (reader.fieldnames or [])
        scored = [row for row in rows if row["metric"] == METRIC_SCORE]
        # The factory's own dim, read off the result so the assertion does not restate its name.
        (dim,) = {score.dim for score in result.rubric_scores}
        assert scored and all(row["rubric_dim"] == dim for row in scored)
        # A whole-result measure leaves the column blank — never "None", which
        # would ingest as the string rather than as NaN.
        assert all(row["rubric_dim"] == "" for row in rows if row["metric"] != METRIC_SCORE)


class TestPerDimensionJudgeAttribution:
    """A `score` row names the model that scored ITS dimension, not the run's pin.

    The two are different facts and both are carried. `judge_model` stays
    run-level so a pivot on it partitions on the value `comparison_sets`' badge
    reads; `rubric_dim_judge_model` answers "which judge scored this
    differently", which the run pin answers *wrongly* on a per-dimension row
    whenever a dim's `JudgeConfig` names its own model.
    """

    def _run(self, **overrides):
        """Build a run pinned to `judge-pin`, with whatever attribution the test states.

        Args:
            **overrides: Passed through to ``make_eval_run``.

        Returns:
            The run.
        """
        run, _ = _run_with_results(judge_model="judge-pin", **overrides)
        return run

    def _scored(self, run, *dims):
        """Build one result scoring each named dim.

        Args:
            run: The run to wire the result to.
            *dims: Dimension names to score, each at 5.

        Returns:
            An ``EvalResult`` carrying one ``RubricScore`` per named dim.
        """
        return make_eval_result(
            eval_run_id=run.id,
            scope_id=run.scope_id,
            rubric_scores=[RubricScore(dim=dim, score=5, scale="ordinal") for dim in dims],
        )

    def _score_rows(self, run, result):
        """Project one result and return its `score` rows.

        Args:
            run: The run supplying the coordinates.
            result: The observation to project.

        Returns:
            The ``METRIC_SCORE`` rows, in projection order.
        """
        return [
            r
            for r in project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records
            if r.metric == METRIC_SCORE
        ]

    def test_a_dim_with_its_own_judge_names_that_model_while_the_pin_stays_the_pin(self):
        """The reported defect: a gemini-scored dim exported as the run's gpt pin.

        Both coordinates are asserted together on purpose. Fixing the
        misattribution by rewriting `judge_model` per row would have broken its
        agreement with the comparability badge, which reads the run-level pin —
        so the row has to carry both values, and this pins that it does.
        """
        run = self._run(
            effective_judges={"reply.house_style": "gemini-lite", "reply.selection_fit": "judge-pin"},
            effective_judges_source="recorded",
        )
        result = self._scored(run, "reply.house_style", "reply.selection_fit")

        rows = self._score_rows(run, result)

        assert {(r.rubric_dim, r.rubric_dim_judge_model) for r in rows} == {
            ("reply.house_style", "gemini-lite"),
            ("reply.selection_fit", "judge-pin"),
        }
        assert {r.judge_model for r in rows} == {"judge-pin"}, "the run pin must stay run-level"

    def test_a_run_whose_dims_all_took_the_pin_attributes_them_all_to_it(self):
        """The uniform case is unchanged — the coordinate agrees with the pin when nothing overrode it."""
        run = self._run(
            effective_judges={"reply.grounding": "judge-pin", "reply.coverage": "judge-pin"},
            effective_judges_source="recorded",
        )

        rows = self._score_rows(run, self._scored(run, "reply.grounding", "reply.coverage"))

        assert {r.rubric_dim_judge_model for r in rows} == {"judge-pin"}

    def test_whole_result_rows_carry_no_dim_judge(self):
        """A composite or cost row spans every dim, so no single model scored it — mirroring `rubric_dim`."""
        run = self._run(effective_judges={"reply.grounding": "gemini-lite"}, effective_judges_source="recorded")
        records = project_score_records(
            [run], [self._scored(run, "reply.grounding")], profile=_JUDGED_HOST, archived_run_ids=None
        ).records

        whole_result = [r for r in records if r.metric != METRIC_SCORE]

        assert whole_result and all(r.rubric_dim_judge_model is None for r in whole_result)
        assert all(r.rubric_dim is None for r in whole_result), "the two must be absent together"

    def test_a_dimensionless_score_row_carries_no_dim_judge(self):
        """No dimension, nothing to attribute — naming the pin would invent an attribution nobody made."""
        run = self._run(effective_judges={"reply.grounding": "gemini-lite"}, effective_judges_source="recorded")

        rows = self._score_rows(run, self._scored(run))

        assert [(r.rubric_dim, r.rubric_dim_judge_model) for r in rows] == [(None, None)]

    def test_a_run_with_no_recorded_attribution_says_nobody_can_say(self):
        """A run whose attribution was not recorded has dims that fall back to nothing, never to the pin.

        Falling back to `judge_model` here is precisely the misattribution the
        coordinate exists to end — it would restate the run pin under a name that
        claims to be the per-dim answer, which is worse than saying nothing.
        """
        run = self._run(effective_judges=None)

        rows = self._score_rows(run, self._scored(run, "reply.grounding"))

        assert [r.rubric_dim_judge_model for r in rows] == [None]
        assert [r.judge_model for r in rows] == ["judge-pin"], "the pin is still recorded and still carried"

    def test_a_reconstruction_is_not_published_as_a_record(self):
        """A `derived` attribution is an inference, and a pivot level carries no caveat.

        The same eligibility rule keeps a reconstruction out of the context key
        and out of the comparability badge. A CSV cell or an axis level has no
        channel to say "inferred", so grouping observations under a model the
        backfill guessed at would make an inferred condition indistinguishable
        from a measured one.
        """
        run = self._run(effective_judges={"reply.grounding": "gemini-lite"}, effective_judges_source="derived")

        rows = self._score_rows(run, self._scored(run, "reply.grounding"))

        assert [r.rubric_dim_judge_model for r in rows] == [None]

    def test_a_dim_the_run_never_attributed_is_not_backfilled_from_the_pin(self):
        """A dim missing from the recorded map is one nobody committed to a judge for."""
        run = self._run(effective_judges={"reply.grounding": "gemini-lite"}, effective_judges_source="recorded")

        rows = self._score_rows(run, self._scored(run, "reply.coverage"))

        assert [(r.rubric_dim, r.rubric_dim_judge_model) for r in rows] == [("reply.coverage", None)]

    def test_the_value_is_read_from_the_runs_own_map_not_re_derived(self):
        """One definition, shared with the `Judged by:` block and `judge_dim_divergence`.

        The cascade is applied once, at launch. Re-running it here against
        today's configs is how this surface would come to disagree with the run
        summary about the same run, so the helper is a lookup: move the map and
        the answer moves with it.
        """
        run = self._run(effective_judges={"reply.grounding": "gemini-lite"}, effective_judges_source="recorded")

        assert dim_judge_model(run, "reply.grounding") == "gemini-lite"
        run.effective_judges = {"reply.grounding": "something-else"}
        assert dim_judge_model(run, "reply.grounding") == "something-else"

    def test_the_reserved_dual_score_axes_never_become_score_rows(self):
        """`rubric_scores` holds the template dims alone, so no reserved id reaches this coordinate.

        The reserved axes ARE exported — as metrics of their own (`__transcript__` /
        `__outcome__`), which is a different thing and the reason this pin survived
        gaining them. The constraint is about the `rubric_dim` COORDINATE:
        `compute_dimension_summary` does not aggregate these two, so a reserved id on
        that coordinate would make the projection and the run summary disagree about
        which dimensions exist, and a pooled `score` cell would average decision
        quality in with the subject's tone.

        The sibling below pins the other half — that the rows exist at all — so the
        two together say "exported, and not as a dimension".
        """
        run = self._run(
            effective_judges={
                TRANSCRIPT_DIM_ID: "gemini-lite",
                OUTCOME_DIM_ID: "judge-pin",
                "reply.grounding": "judge-pin",
            },
            effective_judges_source="recorded",
        )
        result = self._scored(run, "reply.grounding")
        result.transcript_score = RubricScore(dim=TRANSCRIPT_DIM_ID, score=4, scale="ordinal")
        result.outcome_score = RubricScore(dim=OUTCOME_DIM_ID, score=3, scale="ordinal")

        rows = self._score_rows(run, result)

        assert [r.rubric_dim for r in rows] == ["reply.grounding"]

    def test_an_axis_stamped_with_an_unreserved_dim_is_dropped_loudly(self, caplog):
        """The `metric` column is the one that comes off stored data, so it is guarded — and said.

        `RubricScore.dim` accepts any namespaced dim name as well as the reserved ids, so
        nothing ties an axis slot to its reserved id, and `PROJECTED_METRICS` is documented
        as closed ("a name outside it can only be a typo"). A mis-stamped axis is skipped so
        that claim stays true by construction.

        The WARNING is the half that matters. An unjudged result gains no rows here
        either, so a silent skip would make a mis-stamped axis read
        exactly like an unjudged one — which is worse than the unguarded state, where the
        odd metric value was at least visible and `pivot` refused it by name.
        """
        run = self._run(effective_judges={TRANSCRIPT_DIM_ID: "gemini-lite"}, effective_judges_source="recorded")
        result = self._scored(run, "reply.grounding")
        result.transcript_score = RubricScore(dim="reply.transcript", score=4, scale="ordinal")

        with caplog.at_level(logging.WARNING, logger="threetears.evals.analysis.reporting"):
            records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        assert not [r for r in records if r.metric not in ("composite", "cost_usd", METRIC_SCORE, METRIC_GOAL_STATE)], (
            "an unreserved dim must not reach the metric column"
        )
        assert any("not a reserved axis" in r.getMessage() for r in caplog.records), (
            "the drop must be said — a silent one is indistinguishable from a result nobody judged"
        )

    def test_a_dual_score_axis_row_names_the_judge_that_scored_that_axis(self):
        """The attribution is real on these rows, and `rubric_dim` being null does not withhold it.

        The rule the coordinate follows is "no attribution where no SINGLE model scored
        the row" — true of a composite, which spans every dim. A dual-axis row was
        scored by exactly one judge, and the run's own map names it. Reading the old
        "null exactly where `rubric_dim` is null" as the rule would drop a real
        attribution to preserve a coincidence between two fields.
        """
        run = self._run(
            effective_judges={
                TRANSCRIPT_DIM_ID: "gemini-lite",
                OUTCOME_DIM_ID: "judge-pin",
                "reply.grounding": "judge-pin",
            },
            effective_judges_source="recorded",
        )
        result = self._scored(run, "reply.grounding")
        result.transcript_score = RubricScore(dim=TRANSCRIPT_DIM_ID, score=4, scale="ordinal")
        result.outcome_score = RubricScore(dim=OUTCOME_DIM_ID, score=3, scale="ordinal")

        rows = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records
        by_metric = {r.metric: r for r in rows if r.metric in (TRANSCRIPT_DIM_ID, OUTCOME_DIM_ID)}

        assert by_metric[TRANSCRIPT_DIM_ID].rubric_dim_judge_model == "gemini-lite"
        assert by_metric[OUTCOME_DIM_ID].rubric_dim_judge_model == "judge-pin"
        assert by_metric[TRANSCRIPT_DIM_ID].rubric_dim is None, (
            "the metric IS the axis; there is no dimension coordinate"
        )

    def test_the_dim_judge_is_pivotable_as_an_axis(self):
        """The pivot the run pin answered wrongly: two judges, two rows, not one collapsed row.

        `results_pivot(row_factor='judge_model', ...)` put every observation of
        this run under the pin — attributing the gemini-judged dimension's
        numbers to the pinned model. On the per-dim coordinate the two separate.
        """
        run = self._run(
            effective_judges={"reply.house_style": "gemini-lite", "reply.selection_fit": "judge-pin"},
            effective_judges_source="recorded",
        )
        result = make_eval_result(
            eval_run_id=run.id,
            scope_id=run.scope_id,
            rubric_scores=[
                RubricScore(dim="reply.house_style", score=5, scale="ordinal"),
                RubricScore(dim="reply.selection_fit", score=1, scale="ordinal"),
            ],
        )
        rows = self._score_rows(run, result)

        table = compute_pivot(
            rows, row_factor="rubric_dim_judge_model", column_factor="model", metric=METRIC_SCORE, profile=_JUDGED_HOST
        )

        assert table.rows == ["gemini-lite", "judge-pin"]
        assert {cell.row: cell.value for cell in table.cells} == {"gemini-lite": 5.0, "judge-pin": 1.0}
        # The run pin still pools them into one row, which is what it is FOR —
        # it partitions on the value the comparability badge reads.
        pinned = compute_pivot(
            rows, row_factor="judge_model", column_factor="model", metric=METRIC_SCORE, profile=_JUDGED_HOST
        )
        assert pinned.rows == ["judge-pin"]

    def test_the_dim_judge_exports_as_its_own_column(self):
        """The export derives its columns from the model, so the coordinate arrives as a pandas column."""
        run = self._run(
            effective_judges={"reply.house_style": "gemini-lite", "reply.selection_fit": "judge-pin"},
            effective_judges_source="recorded",
        )
        records = project_score_records(
            [run],
            [self._scored(run, "reply.house_style", "reply.selection_fit")],
            profile=_JUDGED_HOST,
            archived_run_ids=None,
        ).records

        reader = csv.DictReader(io.StringIO(export_records_csv(records)))
        rows = list(reader)

        assert "rubric_dim_judge_model" in (reader.fieldnames or [])
        by_dim = {row["rubric_dim"]: row["rubric_dim_judge_model"] for row in rows if row["metric"] == METRIC_SCORE}
        assert by_dim == {"reply.house_style": "gemini-lite", "reply.selection_fit": "judge-pin"}
        # Blank on a whole-result row — never "None", which would ingest as a string.
        assert all(row["rubric_dim_judge_model"] == "" for row in rows if row["metric"] != METRIC_SCORE)


class TestSubjectPartition:
    """Composites are comparable within a subject, never across one."""

    def test_rows_from_two_subjects_carry_distinct_partition_keys(self):
        run_a, result_a = _run_with_results(subject_id="ent-maple", subject_label="Maple")
        run_b, result_b = _run_with_results(subject_id="ent-bea", subject_label="Bea")

        records = project_score_records(
            [run_a, run_b], [result_a, result_b], profile=_JUDGED_HOST, archived_run_ids=None
        ).records

        assert {r.subject_id for r in records} == {"ent-maple", "ent-bea"}

    def test_subject_key_is_the_subject_id_not_the_display_name(self):
        """A rename must not split one subject; two same-named subjects must not merge."""
        renamed, result = _run_with_results(subject_id="ent-maple", subject_label="Maple Prime")

        records = project_score_records([renamed], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        assert {r.subject_id for r in records} == {"ent-maple"}
        assert {r.subject_label for r in records} == {"Maple Prime"}

    def test_per_result_attribution_wins_over_the_run_snapshot(self):
        """The preferred branch of the resolver, which no other test reaches.

        Two generations of the runner record the same fact in different places.
        The result-level field is the more specific one and is preferred; every
        other test in this file leaves it unset and so exercises only the
        snapshot fallback, which would let the preference silently stop working.
        """
        run, result = _run_with_results(subject_id="ent-snapshot")
        result.subject_id = "ent-per-result"

        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        assert {r.subject_id for r in records} == {"ent-per-result"}

    def test_blank_per_result_attribution_falls_back_rather_than_excluding(self):
        """An unset result-level field is absence, not a blank identity."""
        run, result = _run_with_results(subject_id="ent-snapshot")
        result.subject_id = "   "

        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        assert records, "a whitespace attribution swallowed the run's real identity"
        assert {r.subject_id for r in records} == {"ent-snapshot"}

    def test_a_blank_subject_cannot_reach_the_projection_because_it_cannot_be_built(self):
        """The exclusion class this tier used to carry is gone, and so is the state it counted.

        A scope whose runs carried no subject once projected to zero rows and
        needed a count travelling beside them, or it read as a scope nobody had ever run.
        The rule removed the case rather than the disclosure: a subject key cannot be blank, so
        a count of subject-less results could only ever report zero, and a zero a reader takes as
        evidence is worse than no field. Asserted at the point of construction, which is where
        the refusal now lives.
        """
        with pytest.raises(ValueError, match="subject_id"):
            _run_with_results(subject_id="", subject_label="")

    def test_an_orphan_result_is_counted_under_its_own_reason(self):
        """Two exclusion classes, kept apart — they mean different things to a reader."""
        _, orphan = _run_with_results()

        projection = project_score_records([], [orphan], profile=_JUDGED_HOST, archived_run_ids=None)

        assert projection.exclusions.results_without_run == 1
        assert projection.exclusions.total == 1

    def test_a_clean_corpus_reports_no_exclusions(self):
        """Zero must be reachable, or a non-zero count carries no information."""
        run, result = _run_with_results()

        assert project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).exclusions.total == 0

    def test_a_run_the_caller_filtered_out_is_not_reported_as_unplaceable(self):
        """A filtered subset and an unplaceable row are different facts.

        `pivot` narrows runs by status but reads every result in the scope, so
        without this split each result of a non-matching run inflated
        `results_without_run` — a counter documented as data that cannot be
        placed. With `status` defaulting to `completed`, one in-flight run was
        enough to make a healthy corpus report an integrity gap.
        """
        _, filtered_out = _run_with_results()

        projection = project_score_records(
            [], [filtered_out], known_run_ids={filtered_out.eval_run_id}, profile=_JUDGED_HOST, archived_run_ids=None
        )

        assert projection.exclusions.results_outside_queried_runs == 1
        assert projection.exclusions.results_without_run == 0
        assert projection.exclusions.total == 1, "still disclosed — the scope is not empty"

    def test_a_run_absent_from_the_corpus_entirely_is_still_unplaceable(self):
        """The benign class must not swallow the real one — that would be the same bug inverted."""
        _, orphan = _run_with_results()

        projection = project_score_records(
            [], [orphan], known_run_ids={"some-other-run"}, profile=_JUDGED_HOST, archived_run_ids=None
        )

        assert projection.exclusions.results_without_run == 1
        assert projection.exclusions.results_outside_queried_runs == 0

    def test_an_unfiltered_caller_attributes_every_absence_to_a_missing_run(self):
        """`known_run_ids=None` asserts `runs` is the whole corpus, so nothing was filtered."""
        _, orphan = _run_with_results()

        projection = project_score_records([], [orphan], profile=_JUDGED_HOST, archived_run_ids=None)

        assert projection.exclusions.results_without_run == 1
        assert projection.exclusions.results_outside_queried_runs == 0

    def test_a_run_with_no_subject_id_is_refused_at_capture_not_excluded_at_projection(self):
        """Pooling blanks was the forbidden resolution, and the exclusion existed to prevent it.

        A subject without an id is not constructible (``SubjectSnapshot.subject_id`` is required and
        non-blank), so the state the exclusion described is not one the engine supports. It is
        refused where the subject is built, and there is nothing left for the projection to drop.
        """
        with pytest.raises(ValueError, match="subject_id"):
            _run_with_results(subject_id="", subject_label="Maple")

    def test_whitespace_only_identity_is_refused_rather_than_treated_as_absent(self):
        """A key of spaces used to project to zero rows; it is now not constructible.

        The distinction matters to a reader: "excluded" says the data exists and cannot be
        grouped, and there is no longer any such data.
        """
        with pytest.raises(ValueError, match="subject_id"):
            _run_with_results(subject_id="   ", subject_label="   ")


class TestPlacementHelper:
    """`place_results` is the one seam that resolves run + subject and accounts for every drop.

    Tested directly here (once), then the two callers are shown to route through
    it with identical exclusion accounting — the behaviour-preservation the
    extraction promised. A copy of this prologue drifting in one surface is
    exactly what routing all callers through this seam makes impossible.
    """

    def test_a_placed_result_carries_its_run_and_resolved_subject(self):
        run, result = _run_with_results()

        placed, exclusions = place_results([run], [result], None, source="test", archived_run_ids=None)

        assert exclusions.total == 0
        assert len(placed) == 1
        assert placed[0].run is run
        assert placed[0].result is result
        assert placed[0].subject_id == "ent-maple"

    def test_input_order_is_preserved(self):
        """Callers depend on time/append order downstream; the seam must not reshuffle."""
        run, first = _run_with_results()
        second = make_eval_result(eval_run_id=run.id, scope_id=run.scope_id)

        placed, _ = place_results([run], [first, second], None, source="test", archived_run_ids=None)

        assert [p.result.id for p in placed] == [first.id, second.id]

    def test_each_drop_class_is_counted_under_its_own_reason(self):
        """Three classes, kept apart. The fourth — a blank subject — is no longer a state."""
        run, ok = _run_with_results()
        _, orphan = _run_with_results()
        _, filtered = _run_with_results()

        placed, exclusions = place_results(
            [run],
            [ok, orphan, filtered],
            {ok.eval_run_id, filtered.eval_run_id},
            source="test",
            archived_run_ids=None,
        )

        assert [p.result.id for p in placed] == [ok.id]
        assert exclusions.results_outside_queried_runs == 1
        assert exclusions.results_without_run == 1
        assert exclusions.total == 2

    def test_an_archived_run_is_counted_apart_from_the_callers_own_filter(self):
        """Two narrowings, two counters — the remedies differ and only one is a filter."""
        run, ok = _run_with_results()
        _, archived = _run_with_results()
        _, filtered = _run_with_results()

        _, exclusions = place_results(
            [run],
            [ok, archived, filtered],
            {ok.eval_run_id, archived.eval_run_id, filtered.eval_run_id},
            source="test",
            archived_run_ids={archived.eval_run_id},
        )

        assert exclusions.results_from_archived_runs == 1
        assert exclusions.results_outside_queried_runs == 1
        assert exclusions.total == 2, "both classes count toward 'the scope is not empty'"

    def test_no_archive_narrowing_leaves_every_deliberate_drop_on_the_filter_class(self):
        """``archived_run_ids=None`` means the archive removed nothing, not "unknown".

        A caller whose selection already includes archived runs passes ``None``,
        and every deliberate drop it then reports really is its own filter's —
        the assertion `export_results`' by-id path depends on.
        """
        run, ok = _run_with_results()
        _, filtered = _run_with_results()

        _, exclusions = place_results(
            [run], [ok, filtered], {ok.eval_run_id, filtered.eval_run_id}, source="test", archived_run_ids=None
        )

        assert exclusions.results_from_archived_runs == 0
        assert exclusions.results_outside_queried_runs == 1

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda run, result: place_results([run], [result], None, source="test"), id="place_results"),
            pytest.param(
                lambda run, result: project_score_records([run], [result], profile=_JUDGED_HOST),
                id="project_score_records",
            ),
            pytest.param(lambda run, result: compute_frontier([run], [result]), id="compute_frontier"),
            pytest.param(
                lambda run, result: compute_history([run], [result], profile=_JUDGED_HOST), id="compute_history"
            ),
        ],
    )
    def test_a_caller_that_never_states_archival_is_refused(self, call):
        """No default, so forgetting archival is a ``TypeError``, not a ``status`` filter blamed.

        With a ``None`` default, a caller that narrowed its cohort and omitted the
        set got every archived run counted as its own filter's exclusion, with
        advice to widen ``status`` that can never reach an archived run (#670).
        """
        run, result = _run_with_results()

        with pytest.raises(TypeError, match="archived_run_ids"):
            call(run, result)

    def test_both_callers_report_identical_exclusion_accounting(self):
        """project_score_records and frontier must drop the same rows for the same reasons.

        The extraction's contract: one placement seam, so a corpus that excludes
        rows discloses the same counts regardless of which surface asked. If the
        two ever diverged, a frontier and a pivot over one scope would
        disagree about how much data exists.
        """
        run, ok = _run_with_results()
        _, orphan = _run_with_results()
        _, filtered = _run_with_results()

        runs = [run]
        results = [ok, orphan, filtered]
        known = {ok.eval_run_id, filtered.eval_run_id}

        proj = project_score_records(
            runs, results, known_run_ids=known, profile=_JUDGED_HOST, archived_run_ids=None
        ).exclusions
        front = compute_frontier(runs, results, known_run_ids=known, archived_run_ids=None).exclusions

        assert proj.model_dump() == front.model_dump()


class TestComparisonSets:
    """Grouping refuses to compare across subjects, and badges every weaker caveat."""

    def test_two_subjects_never_share_a_comparison_set(self):
        run_a, _ = _run_with_results(subject_id="ent-maple")
        run_b, _ = _run_with_results(subject_id="ent-bea")

        sets = compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets

        assert len(sets) == 2
        assert {s.subject_id for s in sets} == {"ent-maple", "ent-bea"}

    def test_identical_runs_group_with_no_badges(self):
        run_a = _stamped_run(test_case_ids=["tc-1", "tc-2"])
        run_b = _stamped_run(test_case_ids=["tc-1", "tc-2"])

        sets = compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets

        assert len(sets) == 1
        assert sets[0].badges == []
        assert sets[0].shared_test_case_ids == ["tc-1", "tc-2"]

    def test_case_set_drift_is_badged_and_intersection_reported(self):
        """A 3-4-5 case-set drift must surface, not silently shrink."""
        run_a, _ = _run_with_results(test_case_ids=["tc-1", "tc-2", "tc-3"])
        run_b, _ = _run_with_results(test_case_ids=["tc-1", "tc-2", "tc-3", "tc-4"])

        sets = compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets

        assert BADGE_CASE_SET_DIFFERS in sets[0].badges
        assert sets[0].shared_test_case_ids == ["tc-1", "tc-2", "tc-3"]

    def test_each_distinct_case_set_is_fingerprinted_and_attributed(self):
        """The intersection alone cannot say how a group splits, which is the misread.

        The symptom: a group of runs reported as "sharing 3 cases" while one of its
        runs had executed 4. The intersection was small BECAUSE the sets differed,
        and nothing said so.
        """
        run_a, _ = _run_with_results(test_case_ids=["tc-1", "tc-2", "tc-3"])
        run_b, _ = _run_with_results(test_case_ids=["tc-1", "tc-2", "tc-3"])
        run_c, _ = _run_with_results(test_case_ids=["tc-1", "tc-2", "tc-3", "tc-4"])

        group = compute_comparison_sets([run_a, run_b, run_c], profile=_JUDGED_HOST).comparison_sets[0]

        assert len(group.case_sets) == 2
        # Most-used first: two runs shared the 3-case set, one ran the 4-case set.
        assert [entry.n_cases for entry in group.case_sets] == [3, 4]
        assert group.case_sets[0].run_ids == sorted([run_a.id, run_b.id])
        assert group.case_sets[1].run_ids == [run_c.id]
        # A count of 3 shared cases would have read as "these ran the same 3 cases".
        assert group.shared_test_case_ids == ["tc-1", "tc-2", "tc-3"]

    def test_the_fingerprint_is_the_runs_own_case_basis_not_a_second_digest(self):
        """A competing case-set identity would be free to drift from the stamped one."""
        run, _ = _run_with_results(test_case_ids=["tc-1", "tc-2"])

        group = compute_comparison_sets([run], profile=_JUDGED_HOST).comparison_sets[0]

        assert (
            group.case_sets[0].fingerprint == resolve_context_identity(run, _JUDGED_HOST).context_components.case_basis
        )

    def test_runs_sharing_a_case_set_share_a_fingerprint(self):
        """Order of resolution must not split an otherwise identical basis."""
        run_a, _ = _run_with_results(test_case_ids=["tc-1", "tc-2"])
        run_b, _ = _run_with_results(test_case_ids=["tc-2", "tc-1"])

        group = compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets[0]

        assert len(group.case_sets) == 1
        assert BADGE_CASE_SET_DIFFERS not in group.badges

    @staticmethod
    def _without_case_basis(monkeypatch, *unresolved_run_ids: str):
        """Blank ``case_basis`` on the named runs, leaving the rest resolved.

        Driven through the resolver rather than by hand-stamping a field, because
        the resolver is the single place the stamped-or-derived decision is made
        and the surface reads it rather than the raw run. Patched rather than
        constructed: nothing stamps an identity without ``case_basis`` at the
        current ``IDENTITY_VERSION``, so the state is latent — which is exactly
        why the bucketing must be structurally safe before the next bump makes it
        reachable, and why a test cannot reach it any other way today.
        """
        from threetears.evals.analysis import reporting

        real = reporting.resolve_context_identity
        targets = set(unresolved_run_ids)

        def _resolve(run, profile):
            identity = real(run, profile)
            if run.id not in targets:
                return identity
            components = identity.context_components.model_copy(update={"case_basis": None})
            return identity.model_copy(update={"context_components": components})

        monkeypatch.setattr(reporting, "resolve_context_identity", _resolve)

    def test_unresolved_case_bases_never_merge_into_one_bucket(self, monkeypatch):
        """Two unknowns are not a match, and must not be reported as one.

        Keying every unresolved run on a shared ``""`` merged them into a single
        bucket. Two consequences, both silent: the group compared as one case set
        so no caveat fired, and ``n_cases`` — read from the bucket's first member
        as its representative — reported that one run's denominator as the whole
        bucket's, here 2 standing in for a run that executed 3.
        """
        run_a, _ = _run_with_results(test_case_ids=["tc-1", "tc-2"])
        run_b, _ = _run_with_results(test_case_ids=["tc-3", "tc-4", "tc-5"])
        self._without_case_basis(monkeypatch, run_a.id, run_b.id)

        group = compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets[0]

        assert len(group.case_sets) == 2
        assert all(entry.fingerprint is None for entry in group.case_sets)
        # Each bucket holds exactly its own run, so the representative really is one.
        assert sorted(entry.n_cases for entry in group.case_sets) == [2, 3]
        assert sorted(entry.run_ids[0] for entry in group.case_sets) == sorted([run_a.id, run_b.id])

    def test_an_unresolved_basis_badges_unknown_not_different(self, monkeypatch):
        """``case_set_differs`` asserts an observation; an unresolved basis has none.

        Firing it here would claim the sets were compared and found to differ, when
        what happened is that one of them could not be read at all. The distinction
        is the same one ``context_incomplete`` exists to make, one axis over.
        """
        run_a, _ = _run_with_results(test_case_ids=["tc-1", "tc-2"])
        run_b, _ = _run_with_results(test_case_ids=["tc-1", "tc-2"])
        self._without_case_basis(monkeypatch, run_b.id)

        group = compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets[0]

        assert BADGE_CASE_SET_UNRESOLVED in group.badges
        assert BADGE_CASE_SET_DIFFERS not in group.badges

    def test_resolved_runs_that_really_differ_still_badge_differs(self, monkeypatch):
        """The unknown badge must not swallow the observed one when both apply."""
        run_a, _ = _run_with_results(test_case_ids=["tc-1", "tc-2"])
        run_b, _ = _run_with_results(test_case_ids=["tc-1", "tc-2", "tc-3"])
        run_c, _ = _run_with_results(test_case_ids=["tc-9"])
        self._without_case_basis(monkeypatch, run_c.id)

        group = compute_comparison_sets([run_a, run_b, run_c], profile=_JUDGED_HOST).comparison_sets[0]

        assert BADGE_CASE_SET_DIFFERS in group.badges
        assert BADGE_CASE_SET_UNRESOLVED in group.badges

    def test_context_drift_is_badged(self):
        """Driven through a real pinned condition, not two hand-written key strings.

        ``cassette_mode`` is a context component, so stamping both runs produces genuinely
        different keys — which is what the badge is claiming to have noticed. It replaced
        ``k_runs`` as this test's driver at IDENTITY_VERSION v7, which removed ``k`` from the
        predicate; the sibling below pins that removal so the two cannot drift apart.
        """
        run_a = _stamped_run(cassette_mode="off")
        run_b = _stamped_run(cassette_mode="replay")

        sets = compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets

        assert BADGE_CONTEXT_DIFFERS in sets[0].badges

    def test_repetition_count_is_not_context_drift(self):
        """v7: two runs differing only in ``k`` are the same CONDITION, so no badge.

        ``k`` is how many repetitions were taken, not a circumstance they were taken
        under — a k=1 pilot and the k=3 run that followed it were measured identically and
        differ only in precision. Badging that told an operator the two were not comparable,
        which is a caveat about nothing. Depth is disclosed by the surfaces that own it
        (``Iters/case``, the per-point case counts on the pass^k curve, the completeness sentence).
        """
        sets = compute_comparison_sets(
            [_stamped_run(k_runs=1), _stamped_run(k_runs=3)], profile=_JUDGED_HOST
        ).comparison_sets

        assert BADGE_CONTEXT_DIFFERS not in sets[0].badges

    def test_every_run_groups_because_a_subject_key_cannot_be_blank(self):
        """ "Grouping nothing" and "having nothing" used to be different answers here.

        They were told apart by a count of runs whose subject identity was never captured. The
        subject-key rule removed the state rather than the disclosure, so the count went with it: every run now
        carries a non-blank subject key and therefore lands in a group. Pinned so that a reader
        meeting the field's absence knows it was retired rather than dropped.
        """
        run, _ = _run_with_results()

        result = compute_comparison_sets([run], profile=_JUDGED_HOST)

        assert [group.subject_id for group in result.comparison_sets] == ["ent-maple"]
        assert not hasattr(result, "excluded_no_persona")


def _record(
    *,
    model="m1",
    template="tpl-1",
    case="tc-1",
    value=1.0,
    metric=METRIC_COMPOSITE,
    subject_id="ent-maple",
    outcome="ok",
    k=1,
    factors=None,
    judge_model=None,
):
    """Build one score record, addressing the two axes the pivot tests use.

    Args:
        model: Value for the ``model`` coordinate.
        template: Value for the ``template_id`` coordinate. Named for the
            coordinate rather than for an axis, because which coordinate is the
            row axis and which the column is the caller's choice per pivot.
        case: Test-case id — the equal-per-scenario grouping key.
        value: The observation's value; ``None`` marks it unmeasured.
        metric: Which measure this row carries.
        subject_id: The comparability partition key.
        outcome: Scoring class carried from the taxonomy.
        k: Iteration index, so repeats within one case stay distinct rows.
        factors: Open dotted coordinates (run-scoped config / prompt overlays).
        judge_model: The run-pinned judge role, a first-class axis.

    Returns:
        A `ScoreRecord`.
    """
    return ScoreRecord(
        subject_id=subject_id,
        scope_id="uni-1",
        run_id="run-1",
        result_id=f"res-{model}-{template}-{case}-{k}",
        template_id=template,
        test_case_id=case,
        model=model,
        k_iteration=k,
        metric=metric,
        value=value,
        # A composite states what it was meaned over, so a fixture composite must too — a helper
        # that constructs a row the model refuses is a fixture asserting something impossible.
        dimension_basis=(["reply.quality"] if metric == METRIC_COMPOSITE and value is not None else None),
        outcome=outcome,
        factors=factors or {},
        judge_model=judge_model,
    )


def _grid(records):
    """Index a pivot's cells by (row, column) for direct assertion.

    Args:
        records: A `PivotTable`.

    Returns:
        Dict of ``(row, column)`` → `PivotCell`.
    """
    return {(cell.row, cell.column): cell for cell in records.cells}


class TestPivotDisclosure:
    """No cell is a bare point estimate, and no absence renders as a number."""

    def test_every_measured_cell_carries_n_and_dispersion(self):
        records = [_record(case="tc-1", value=1.0), _record(case="tc-2", value=0.0)]

        table = compute_pivot(
            records, row_factor="model", column_factor="template_id", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
        )

        cell = _grid(table)[("m1", "tpl-1")]
        assert cell.status == "measured"
        assert cell.n == 2
        assert cell.n_cases == 2
        assert cell.sem is not None

    def test_a_single_observation_reports_unknown_dispersion_not_zero(self):
        """One measurement has unknown precision; 0.0 would rank it as the most certain cell."""
        table = compute_pivot(
            [_record(value=1.0)],
            row_factor="model",
            column_factor="template_id",
            metric=METRIC_COMPOSITE,
            profile=_JUDGED_HOST,
        )

        assert _grid(table)[("m1", "tpl-1")].sem is None

    def test_an_unrun_combination_is_not_run_and_never_zero(self):
        """The grid is the cross-product, so a gap is a stated absence rather than a missing row."""
        records = [_record(model="m1", template="tpl-1"), _record(model="m2", template="tpl-2")]

        table = compute_pivot(
            records, row_factor="model", column_factor="template_id", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
        )

        gap = _grid(table)[("m1", "tpl-2")]
        assert gap.status == "not_run"
        assert gap.value is None
        assert gap.n == 0

    def test_a_measured_zero_stays_distinct_from_not_run(self):
        """The pairing that makes the distinction load-bearing rather than cosmetic."""
        records = [_record(model="m1", value=0.0), _record(model="m2", template="tpl-2", value=1.0)]

        table = compute_pivot(
            records, row_factor="model", column_factor="template_id", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
        )

        grid = _grid(table)
        assert (grid[("m1", "tpl-1")].status, grid[("m1", "tpl-1")].value) == ("measured", 0.0)
        assert (grid[("m1", "tpl-2")].status, grid[("m1", "tpl-2")].value) == ("not_run", None)

    def test_observations_with_no_value_are_counted_not_averaged_as_zero(self):
        """A cost nothing observed is unmeasured; folding it in as 0 understates real spend."""
        records = [
            _record(metric=METRIC_COST_USD, case="tc-1", value=0.10),
            _record(metric=METRIC_COST_USD, case="tc-2", value=None),
        ]

        table = compute_pivot(
            records, row_factor="model", column_factor="template_id", metric=METRIC_COST_USD, profile=_JUDGED_HOST
        )

        cell = _grid(table)[("m1", "tpl-1")]
        assert cell.value == 0.10, "an unmeasured observation was averaged in as zero"
        assert (cell.n, cell.n_unmeasured) == (1, 1)

    def test_a_cell_whose_observations_all_lack_values_is_unmeasured_not_not_run(self):
        """We tried and got nothing back — a different fact from never having tried."""
        table = compute_pivot(
            [_record(metric=METRIC_COST_USD, value=None)],
            row_factor="model",
            column_factor="template_id",
            metric=METRIC_COST_USD,
            profile=_JUDGED_HOST,
        )

        cell = _grid(table)[("m1", "tpl-1")]
        assert (cell.status, cell.value, cell.n_unmeasured) == ("unmeasured", None, 1)

    def test_a_cell_of_real_harness_failures_reads_unmeasured_not_not_run(self):
        """The hand-built cases above prove the branch works; this proves real data reaches it.

        A projection that emitted no row for a valueless observation made this
        cell indistinguishable from one nobody ever ran, which is the exact
        misreading the three cell states exist to prevent.
        """
        run, _ = _run_with_results()
        failures = [
            make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, test_case_id=case, infra_error="judge timeout")
            for case in ("tc-1", "tc-2")
        ]

        table = compute_pivot(
            project_score_records([run], failures, profile=_JUDGED_HOST, archived_run_ids=None).records,
            row_factor="model",
            column_factor="template_id",
            metric=METRIC_COMPOSITE,
            profile=_JUDGED_HOST,
        )

        cell = next(iter(table.cells))
        assert (cell.status, cell.value, cell.n_unmeasured) == ("unmeasured", None, 2)
        assert cell.outcomes == {"infra_exclude": 2}, "the breakdown must name what went wrong, not omit it"

    def test_a_partly_measured_cell_of_real_data_counts_what_it_dropped(self):
        """``n_unmeasured`` read 0 on every real cell while valueless rows went unemitted."""
        run, _ = _run_with_results()
        results = [
            make_eval_result(
                eval_run_id=run.id,
                scope_id=run.scope_id,
                test_case_id="tc-1",
                rubric_scores=[RubricScore(dim="reply.quality", score=5, scale="ordinal")],
            ),
            make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, test_case_id="tc-2", infra_error="boom"),
        ]

        table = compute_pivot(
            project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None).records,
            row_factor="model",
            column_factor="template_id",
            metric=METRIC_COMPOSITE,
            profile=_JUDGED_HOST,
        )

        cell = next(iter(table.cells))
        assert (cell.status, cell.value) == ("measured", 1.0), "the harness failure must not depress the mean"
        assert (cell.n, cell.n_unmeasured) == (1, 1)

    def test_outcome_counts_ride_along_so_a_mean_of_failures_is_visible(self):
        records = [_record(case="tc-1", outcome="ok"), _record(case="tc-2", outcome="candidate_fail")]

        table = compute_pivot(
            records, row_factor="model", column_factor="template_id", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
        )

        assert _grid(table)[("m1", "tpl-1")].outcomes == {"ok": 1, "candidate_fail": 1}


class TestPivotWeighting:
    """Weighting is a disclosed choice, and the disclosure matches the arithmetic."""

    def _unbalanced(self):
        """One case measured three times at 0.0, another once at 1.0.

        Returns:
            Records whose two weightings cannot coincide: equal-per-scenario
            averages the case means (0.0 and 1.0) to 0.5, while sample-weighted
            averages four observations to 0.25.
        """
        return [
            _record(case="tc-1", value=0.0, k=1),
            _record(case="tc-1", value=0.0, k=2),
            _record(case="tc-1", value=0.0, k=3),
            _record(case="tc-2", value=1.0, k=1),
        ]

    def test_equal_per_scenario_gives_each_case_one_vote(self):
        table = compute_pivot(
            self._unbalanced(),
            row_factor="model",
            column_factor="template_id",
            metric=METRIC_COMPOSITE,
            weighting=WEIGHTING_EQUAL_PER_SCENARIO,
            profile=_JUDGED_HOST,
        )

        assert _grid(table)[("m1", "tpl-1")].value == 0.5

    def test_sample_weighted_lets_the_repeat_count_drive_the_number(self):
        table = compute_pivot(
            self._unbalanced(),
            row_factor="model",
            column_factor="template_id",
            metric=METRIC_COMPOSITE,
            weighting=WEIGHTING_SAMPLE_WEIGHTED,
            profile=_JUDGED_HOST,
        )

        assert _grid(table)[("m1", "tpl-1")].value == 0.25

    def test_equal_per_scenario_is_the_default(self):
        """Case counts reflect budget history, not importance."""
        table = compute_pivot(
            self._unbalanced(),
            row_factor="model",
            column_factor="template_id",
            metric=METRIC_COMPOSITE,
            profile=_JUDGED_HOST,
        )

        assert table.weighting == WEIGHTING_EQUAL_PER_SCENARIO
        assert _grid(table)[("m1", "tpl-1")].value == 0.5

    def test_the_disclosed_formula_describes_the_weighting_actually_used(self):
        """A formula that names the other weighting is worse than none — it reads as verified."""
        per_scenario = compute_pivot(
            self._unbalanced(),
            row_factor="model",
            column_factor="template_id",
            metric=METRIC_COMPOSITE,
            profile=_JUDGED_HOST,
        )
        weighted = compute_pivot(
            self._unbalanced(),
            row_factor="model",
            column_factor="template_id",
            metric=METRIC_COMPOSITE,
            weighting=WEIGHTING_SAMPLE_WEIGHTED,
            profile=_JUDGED_HOST,
        )

        assert "each case weighted equally" in per_scenario.formula
        assert "every contributing observation" in weighted.formula
        assert per_scenario.formula != weighted.formula

    def test_dispersion_describes_the_estimate_the_cell_reports(self):
        """Under equal-per-scenario the SEM is over case means, so n_cases is its denominator."""
        table = compute_pivot(
            self._unbalanced(),
            row_factor="model",
            column_factor="template_id",
            metric=METRIC_COMPOSITE,
            profile=_JUDGED_HOST,
        )

        cell = _grid(table)[("m1", "tpl-1")]
        assert cell.n_cases == 2
        # SEM over the two case means (0.0, 1.0): sd=0.7071, /sqrt(2) = 0.5.
        assert cell.sem == pytest.approx(0.5)

    def test_an_unknown_weighting_is_refused_rather_than_defaulted(self):
        with pytest.raises(PivotError, match="unknown weighting"):
            compute_pivot(
                [_record()],
                row_factor="model",
                column_factor="template_id",
                metric=METRIC_COMPOSITE,
                weighting="mean",
                profile=_JUDGED_HOST,
            )


class TestSimpsonsGuard:
    """A pooled ranking the per-row rankings contradict is flagged, not printed bare."""

    def _reversal(self):
        """A textbook reversal: m2 wins every row, m1 wins the pool.

        m1 is measured mostly on the easy row and m2 mostly on the hard one, so
        pooling over rows reverses the within-row verdict.

        Returns:
            Records for a 2x2 pivot of template (rows) x model (columns).
        """
        return [
            # Easy template: both score high, m2 higher.
            _record(model="m1", template="tpl-easy", case="tc-1", value=0.90),
            _record(model="m1", template="tpl-easy", case="tc-2", value=0.90),
            _record(model="m2", template="tpl-easy", case="tc-1", value=0.95),
            # Hard template: both score low, m2 still higher.
            _record(model="m1", template="tpl-hard", case="tc-3", value=0.10),
            _record(model="m2", template="tpl-hard", case="tc-3", value=0.20),
            _record(model="m2", template="tpl-hard", case="tc-4", value=0.20),
        ]

    def _table(self):
        """Pivot the reversal fixture with model on the compared (column) axis.

        Returns:
            The `PivotTable`.
        """
        return compute_pivot(
            self._reversal(),
            row_factor="template_id",
            column_factor="model",
            metric=METRIC_COMPOSITE,
            profile=_JUDGED_HOST,
        )

    def test_the_reversal_is_flagged(self):
        table = self._table()

        assert len(table.simpsons_flags) == 1
        flag = table.simpsons_flags[0]
        assert {flag.column_a, flag.column_b} == {"m1", "m2"}
        assert flag.rows_disagreeing == 2
        assert flag.rows_agreeing == 0
        assert sorted(flag.disagreeing_rows) == ["tpl-easy", "tpl-hard"]

    def test_the_flag_names_the_pooled_leader_the_rows_contradict(self):
        """Naming the pooled leader is what makes the flag actionable rather than a warning light.

        Collapsing the row axis, m1's cases are 0.90 / 0.90 / 0.10 → 0.633 and
        m2's are 0.95 / 0.20 / 0.20 → 0.45, so the pool ranks m1 first. Within
        each row m2 wins (0.95 > 0.90 on easy, 0.20 > 0.10 on hard). The reversal
        is pure row membership: m1 was measured mostly on the easy template.
        """
        flag = self._table().simpsons_flags[0]

        assert flag.pooled_leader == "m1"

    def test_a_consistent_ranking_is_not_flagged(self):
        """The guard must stay quiet when pooling agrees, or it becomes noise to dismiss."""
        records = [
            _record(model="m1", template="tpl-a", case="tc-1", value=0.20),
            _record(model="m2", template="tpl-a", case="tc-1", value=0.80),
            _record(model="m1", template="tpl-b", case="tc-2", value=0.30),
            _record(model="m2", template="tpl-b", case="tc-2", value=0.90),
        ]

        table = compute_pivot(
            records, row_factor="template_id", column_factor="model", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
        )

        assert table.simpsons_flags == []

    def test_a_single_shared_row_is_not_enough_to_claim_a_reversal(self):
        """With one comparable row there is no majority to disagree with."""
        records = [
            _record(model="m1", template="tpl-a", case="tc-1", value=0.20),
            _record(model="m2", template="tpl-a", case="tc-1", value=0.80),
        ]

        table = compute_pivot(
            records, row_factor="template_id", column_factor="model", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
        )

        assert table.simpsons_flags == []


class TestPivotRefusesDishonestPooling:
    """The no-pooling rule at the aggregation layer: the class of the measure decides what may pool."""

    def test_composites_are_not_pooled_across_subjects(self):
        records = [_record(subject_id="ent-maple"), _record(subject_id="ent-bea", model="m2")]

        with pytest.raises(PivotError, match="subjects") as excinfo:
            compute_pivot(
                records, row_factor="model", column_factor="template_id", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
            )

        # The refusal is built in a function body, which this module's host-noun canary
        # deliberately does not scan — so nothing but this line stops a later edit putting a
        # host noun back into an operator string the shared contract ships. Checked against
        # the canary's whole `HOST_NOUNS` set, not against one hand-picked noun.
        assert not _HOST_NOUN_IN_PROSE.findall(str(excinfo.value))

    def test_a_mechanical_measure_pools_across_subjects_freely(self):
        """A dollar is a dollar whoever spent it — refusing this hides the budget view."""
        records = [
            _record(subject_id="ent-maple", metric=METRIC_COST_USD, value=0.10),
            _record(subject_id="ent-bea", metric=METRIC_COST_USD, value=0.30, case="tc-2"),
        ]

        table = compute_pivot(
            records, row_factor="model", column_factor="template_id", metric=METRIC_COST_USD, profile=_JUDGED_HOST
        )

        assert _grid(table)[("m1", "tpl-1")].value == pytest.approx(0.20)

    def test_subject_on_an_axis_makes_the_pivot_legal_again(self):
        records = [_record(subject_id="ent-maple"), _record(subject_id="ent-bea")]

        table = compute_pivot(
            records, row_factor="subject_id", column_factor="template_id", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
        )

        assert table.rows == ["ent-bea", "ent-maple"]

    def test_filtering_to_one_subject_makes_the_pivot_legal_again(self):
        records = [_record(subject_id="ent-maple", value=1.0), _record(subject_id="ent-bea", value=0.0)]

        table = compute_pivot(
            records,
            row_factor="model",
            column_factor="template_id",
            metric=METRIC_COMPOSITE,
            filters={"subject_id": "ent-maple"},
            profile=_JUDGED_HOST,
        )

        assert _grid(table)[("m1", "tpl-1")].value == 1.0
        assert table.n_filtered_out == 1

    def test_the_matrix_axes_are_legal_and_are_the_per_subject_row_too(self):
        """`subject_id` x `model` is a dashboard's matrix AND its per-subject card set.

        A dashboard needs no separate matrix or by-model endpoint, because those are this
        call with different axes. So this exact pair is a premise of the dashboard matrix,
        the per-model summary cards and the subject header card alike,
        and a refusal here would blank all three at once with nothing on the page saying why.

        Both halves are asserted together deliberately: what makes one call serve two
        surfaces is that adding the `subject_id` FILTER collapses the same table to one row,
        so the filter and the axis have to keep agreeing.
        """
        records = [
            _record(subject_id="ent-maple", model="m1", value=0.8),
            _record(subject_id="ent-maple", model="m2", value=0.6),
            _record(subject_id="ent-bea", model="m1", value=0.4),
        ]

        matrix = compute_pivot(
            records, row_factor="subject_id", column_factor="model", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
        )

        assert matrix.rows == ["ent-bea", "ent-maple"]
        assert matrix.columns == ["m1", "m2"]
        assert _grid(matrix)[("ent-maple", "m2")].value == pytest.approx(0.6)

        one_row = compute_pivot(
            records,
            row_factor="subject_id",
            column_factor="model",
            metric=METRIC_COMPOSITE,
            filters={"subject_id": "ent-maple"},
            profile=_JUDGED_HOST,
        )

        assert one_row.rows == ["ent-maple"]
        assert _grid(one_row)[("ent-maple", "m1")].value == pytest.approx(0.8)
        assert one_row.n_filtered_out == 1

    def test_a_single_subject_corpus_needs_no_ceremony(self):
        records = [_record(subject_id="ent-maple"), _record(subject_id="ent-maple", model="m2")]

        table = compute_pivot(
            records, row_factor="model", column_factor="template_id", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
        )

        assert table.rows == ["m1", "m2"]


class TestPivotAxisResolution:
    """Axes resolve generically, so a new coordinate is pivotable with no edit here."""

    def test_any_declared_coordinate_may_be_an_axis(self):
        records = [_record(case="tc-1", value=1.0), _record(case="tc-2", value=0.0)]

        table = compute_pivot(
            records, row_factor="test_case_id", column_factor="model", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
        )

        assert table.rows == ["tc-1", "tc-2"]

    def test_an_unknown_factor_is_refused_rather_than_bucketed(self):
        # Clean-snapshot provenance is a genuine cell condition that still lives
        # ONLY inside the `context_key` digest — the role widening lifted the
        # judge/simulator roles out to first-class fields and the cost pivot lifted
        # `cassette_mode`, but snapshot provenance stays hashed. So it remains the
        # honest "coordinate the rows do not carry" case: pivoting on it must refuse,
        # not bucket.
        with pytest.raises(PivotError, match="unknown factor"):
            compute_pivot(
                [_record()],
                row_factor="clean_snapshot",
                column_factor="model",
                metric=METRIC_COMPOSITE,
                profile=_JUDGED_HOST,
            )

    def test_a_model_method_is_not_mistaken_for_a_coordinate(self):
        """`hasattr` would accept `model_dump` and bucket the corpus under one bound method."""
        with pytest.raises(PivotError, match="unknown factor"):
            compute_pivot(
                [_record()],
                row_factor="model_dump",
                column_factor="model",
                metric=METRIC_COMPOSITE,
                profile=_JUDGED_HOST,
            )

    def test_judge_model_pivots_to_per_judge_cells(self):
        """The role coordinate is no longer hidden in `context_key` — it groups.

        This is the query the widen unlocks: "which judge scored this
        differently". Before the role widening `judge_model` reached the row only inside
        the opaque `context_key` digest, so this pivot refused; now it partitions
        by judge like any declared coordinate.
        """
        records = [
            _record(case="tc-1", value=1.0, judge_model="judge-a"),
            _record(case="tc-2", value=0.0, judge_model="judge-b"),
        ]

        table = compute_pivot(
            records, row_factor="judge_model", column_factor="model", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
        )

        assert table.rows == ["judge-a", "judge-b"]
        assert {(c.row, c.value) for c in table.cells} == {("judge-a", 1.0), ("judge-b", 0.0)}

    def test_an_absent_coordinate_becomes_a_visible_level(self):
        """Runs that lack a coordinate (an ad-hoc run has no template) must show as a row, not shrink the table."""
        records = [_record(template="tpl-1"), _record(template=None, case="tc-2")]

        table = compute_pivot(
            records, row_factor="template_id", column_factor="model", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
        )

        assert "—" in table.rows

    def test_a_config_override_is_pivotable_by_its_dotted_key(self):
        """The acceptance case: model x config key, with no per-key code anywhere.

        This is the whole content of the open-factor rule — a bake-off varies a
        tool config key, and reporting must be able to put that key on an axis
        without anyone having added a field, a migration, or a render branch for
        it. A test naming a *specific* key is the only way that claim is checked;
        asserting the mechanism in the abstract would pass against a model that
        carries nothing.
        """
        records = [
            _record(model="m1", value=1.0, factors={"planner.planner_model": "sonnet"}),
            _record(model="m1", value=0.0, case="tc-2", factors={"planner.planner_model": "haiku"}),
        ]

        table = compute_pivot(
            records,
            row_factor="planner.planner_model",
            column_factor="model",
            metric=METRIC_COMPOSITE,
            profile=_JUDGED_HOST,
        )

        assert table.rows == ["haiku", "sonnet"]
        assert {(c.row, c.value) for c in table.cells} == {("haiku", 0.0), ("sonnet", 1.0)}

    def test_a_prompt_override_is_pivotable_under_its_own_prefix(self):
        """`prompt.` marks the slot's provenance, so an axis name says which overlay it came from.

        It is a namespace, not a guarantee: a tool literally named `prompt` with
        a config key matching a slot name would still collide, and the prompt
        loop would win. Unreachable today (no such tool exists) and called out
        rather than overclaimed — an earlier version of this test asserted the
        collision was impossible while never constructing one.
        """
        records = [_record(value=1.0, factors={"prompt.intake_style": "verbose2"})]

        table = compute_pivot(
            records,
            row_factor="prompt.intake_style",
            column_factor="model",
            metric=METRIC_COMPOSITE,
            profile=_JUDGED_HOST,
        )

        assert table.rows == ["verbose2"]

    def test_runs_without_the_override_form_their_own_visible_cohort(self):
        """Running without an override is a comparable cohort, not missing data."""
        records = [
            _record(value=1.0, factors={"planner.planner_model": "sonnet"}),
            _record(value=0.0, case="tc-2", factors={}),
        ]

        table = compute_pivot(
            records,
            row_factor="planner.planner_model",
            column_factor="model",
            metric=METRIC_COMPOSITE,
            profile=_JUDGED_HOST,
        )

        assert table.rows == ["sonnet", "—"]

    def test_a_lever_no_run_set_is_a_table_of_dashes_not_a_refusal(self):
        """Refusing would make "nobody set this" indistinguishable from a typo.

        A member the host's open family claims is a lever before any run carries it.
        """
        table = compute_pivot(
            [_record(value=1.0)],
            row_factor="extractor.field_aliases.vendor",
            column_factor="model",
            metric=METRIC_COMPOSITE,
            profile=_JUDGED_HOST,
        )

        assert table.rows == ["—"]

    def test_a_dot_does_not_make_an_unregistered_name_a_lever(self):
        """The name's shape decides nothing (#664): a dotted name no registry admits and no run carries is a typo."""
        with pytest.raises(PivotError, match="unknown factor 'planner.never_set'"):
            compute_pivot(
                [_record(value=1.0)],
                row_factor="planner.never_set",
                column_factor="model",
                metric=METRIC_COMPOSITE,
                profile=_JUDGED_HOST,
            )

    def test_an_undotted_lever_is_pivotable_with_one_row_per_level(self):
        """#664: a lever with a plain name is pivoted through the registry, end to end through the projection.

        The toy host registers ``chunk_tokens`` undotted; two batches at two widths project through
        ``project_score_records`` (which fills ``factors`` from the registry) and pivot into one row per width.
        """
        runs = [toyhost_batch(chunk_tokens=width, retriever_top_k=3) for width in (256, 1024)]
        results = [
            result
            for run in runs
            for result in toyhost_measurements(
                run, profile=_JUDGED_HOST, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8
            )
        ]
        records = project_score_records(runs, results, profile=_JUDGED_HOST, archived_run_ids=None).records

        table = compute_pivot(
            records, row_factor="chunk_tokens", column_factor="model", metric=METRIC_COST_USD, profile=_JUDGED_HOST
        )

        assert table.rows == ["1024", "256"]
        assert {cell.row for cell in table.cells if cell.status == "measured"} == {"1024", "256"}

    def test_a_mistyped_lever_is_refused_naming_the_hosts_levers(self):
        """The hint lists the levers this host registers, not a claim that levers are dotted."""
        with pytest.raises(PivotError, match="unknown factor 'chunk_token'") as refused:
            compute_pivot(
                [_record(factors={"chunk_tokens": "256"})],
                row_factor="chunk_token",
                column_factor="model",
                metric=METRIC_COMPOSITE,
                profile=_JUDGED_HOST,
            )

        assert "chunk_tokens" in str(refused.value) and "retriever_top_k" in str(refused.value)
        assert "dotted" not in str(refused.value)

    def test_a_lever_named_like_a_declared_coordinate_is_refused_not_resolved_by_branch_order(self):
        """One name, two meanings: a run carrying ``template_id`` as a lever cannot be read off either silently."""
        with pytest.raises(PivotError, match="both a declared score-record coordinate and a lever"):
            compute_pivot(
                [_record(factors={"template_id": "tpl-x"})],
                row_factor="template_id",
                column_factor="model",
                metric=METRIC_COMPOSITE,
                profile=_JUDGED_HOST,
            )

    def test_the_candidate_model_lever_reads_the_model_coordinate(self):
        """The engine's own model lever is projected into ``model``, so the shared name is one quantity."""
        table = compute_pivot(
            [_record(model="m1"), _record(model="m2", case="tc-2")],
            row_factor="model",
            column_factor="test_case_id",
            metric=METRIC_COMPOSITE,
            profile=_JUDGED_HOST,
        )

        assert table.rows == ["m1", "m2"]

    def test_the_open_coordinate_map_is_not_itself_an_axis(self):
        with pytest.raises(PivotError, match="open-coordinate map"):
            compute_pivot(
                [_record()], row_factor="factors", column_factor="model", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
            )

    def test_an_unknown_axis_is_refused_even_when_there_is_nothing_to_group(self):
        """The refusal must not depend on the corpus happening to be non-empty.

        Axis names were originally checked inside the grouping loop, which only
        runs once there are rows — so over an empty scope a typo'd axis
        returned a clean empty grid instead of an error, the same
        indistinguishable-from-no-data answer the refusals exist to prevent,
        surviving on the coordinate nobody re-checked.
        """
        with pytest.raises(PivotError, match="unknown factor"):
            compute_pivot(
                [], row_factor="clean_snapshot", column_factor="model", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
            )

    def test_an_unknown_filter_key_is_refused_rather_than_matching_nothing(self):
        """A filter nobody can satisfy empties the table as convincingly as a real result."""
        with pytest.raises(PivotError, match="unknown factor"):
            compute_pivot(
                [_record()],
                row_factor="model",
                column_factor="template_id",
                metric=METRIC_COMPOSITE,
                filters={"cohort": "ent-maple"},
                profile=_JUDGED_HOST,
            )

    def test_rows_carrying_another_measure_are_ignored(self):
        records = [_record(metric=METRIC_COMPOSITE, value=1.0), _record(metric=METRIC_COST_USD, value=99.0)]

        table = compute_pivot(
            records, row_factor="model", column_factor="template_id", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
        )

        assert _grid(table)[("m1", "tpl-1")].value == 1.0
        assert table.n_observations == 1


class TestPivotPoolingDisclosures:
    """What a pivot cell pools that is not one quantity is said, or its number withheld (#625, #658, #672)."""

    PRICED = ["candidate", "judge"]
    METERED = ["candidate", "external", "judge"]

    @staticmethod
    def _result(run, *, case="tc-1", cost=0.10, **overrides):
        return make_eval_result(
            eval_run_id=run.id, scope_id=run.scope_id, test_case_id=case, cost_usd=cost, **overrides
        )

    @staticmethod
    def _cost_pivot(runs, results, *, column_factor="template_id", row_factor="model", metric=METRIC_COST_USD):
        projection = project_score_records(runs, results, profile=_JUDGED_HOST, archived_run_ids=None)
        return compute_pivot(
            projection.records,
            row_factor=row_factor,
            column_factor=column_factor,
            metric=metric,
            profile=_JUDGED_HOST,
        )

    def test_a_cost_cell_names_both_compositions_and_the_table_flags_them(self):
        """#625: two runs priced different roles; one cell pools both, and says so."""
        run_a, _ = _run_with_results(id="run-a")
        run_b, _ = _run_with_results(id="run-b")
        results = [
            self._result(run_a, case="tc-1", cost_roles=self.PRICED),
            self._result(run_b, case="tc-2", cost_roles=self.METERED, cost=0.20),
        ]

        table = self._cost_pivot([run_a, run_b], results)

        (cell,) = table.cells
        assert cell.status == CELL_MEASURED
        assert cell.cost_compositions == sorted([self.PRICED, self.METERED])
        assert table.cost_compositions_differ is True
        assert "cost compositions differ" in pivot_text(table)

    def test_one_composition_is_named_and_not_flagged(self):
        run, _ = _run_with_results()
        table = self._cost_pivot([run], [self._result(run, cost_roles=self.PRICED)])

        assert table.cells[0].cost_compositions == [self.PRICED]
        assert table.cost_compositions_differ is False

    def test_a_non_cost_pivot_carries_no_composition(self):
        run, _ = _run_with_results()
        table = self._cost_pivot([run], [self._result(run, cost_roles=self.PRICED)], metric=METRIC_COMPOSITE)

        assert table.cells[0].cost_compositions == []
        assert table.cost_compositions_differ is False

    def test_a_cost_cell_pooling_replayed_with_live_is_withheld(self):
        """#658: one live and one replayed run of the same arm — the cell's mean is neither one's spend."""
        live, _ = _run_with_results(id="run-live", cassette_mode="off")
        replayed, _ = _run_with_results(id="run-replay", cassette_mode="replay")
        results = [self._result(live, case="tc-1", cost=0.30), self._result(replayed, case="tc-2", cost=0.05)]

        table = self._cost_pivot([live, replayed], results)

        (cell,) = table.cells
        assert cell.status == CELL_WITHHELD
        assert cell.value is None and cell.sem is None
        assert cell.withheld is not None and "replay" in cell.withheld
        # Withheld is not empty: what the cell held is still counted.
        assert (cell.n, cell.cassette_modes) == (2, ["off", "replay"])
        assert table.cassette_mode_disclosure is not None
        assert CASSETTE_SPAN_CLAUSE in table.cassette_mode_disclosure
        assert "withheld: it pools" in pivot_text(table)

    def test_cassette_mode_on_an_axis_reads_each_alone(self):
        live, _ = _run_with_results(id="run-live", cassette_mode="off")
        replayed, _ = _run_with_results(id="run-replay", cassette_mode="replay")
        results = [self._result(live, case="tc-1", cost=0.30), self._result(replayed, case="tc-2", cost=0.05)]

        table = self._cost_pivot([live, replayed], results, column_factor="cassette_mode")

        cells = _grid(table)
        assert cells[("sonnet", "off")].value == pytest.approx(0.30)
        assert cells[("sonnet", "replay")].value == pytest.approx(0.05)
        # The cells differ in mode, so the table still carries the comparison surfaces' sentence.
        assert table.cassette_mode_disclosure is not None

    def test_a_quality_cell_over_mixed_modes_is_reported_and_the_table_discloses(self):
        live, _ = _run_with_results(id="run-live", cassette_mode="off")
        replayed, _ = _run_with_results(id="run-replay", cassette_mode="replay")
        results = [self._result(live, case="tc-1"), self._result(replayed, case="tc-2")]

        table = self._cost_pivot([live, replayed], results, metric=METRIC_COMPOSITE)

        assert table.cells[0].status == CELL_MEASURED
        assert table.cassette_mode_disclosure is not None

    def test_off_and_capture_pool_without_withholding(self):
        """Both run the third party live: a difference in what was recorded, not in what was spent."""
        live, _ = _run_with_results(id="run-live", cassette_mode="off")
        captured, _ = _run_with_results(id="run-capture", cassette_mode="capture")
        results = [self._result(live, case="tc-1"), self._result(captured, case="tc-2")]

        cell = self._cost_pivot([live, captured], results).cells[0]

        assert cell.status == CELL_MEASURED
        assert cell.cassette_modes == ["capture", "off"]

    def test_a_seeded_delivery_within_one_mode_is_reported_and_exported(self):
        """A seeded finding substitutes on the template's own cases, so arms over them carry it alike."""
        run, _ = _run_with_results()
        seeded = self._result(
            run, case="tc-2", async_deliveries=[AsyncDelivery(tool="scout", status="delivered", substituted=True)]
        )
        projection = project_score_records(
            [run], [self._result(run, case="tc-1"), seeded], profile=_JUDGED_HOST, archived_run_ids=None
        )

        cell = self._cost_pivot([run], [self._result(run, case="tc-1"), seeded]).cells[0]

        assert cell.status == CELL_MEASURED
        assert (cell.n_substituted, cell.n) == (1, 2)
        assert cell.substitution_disclosure is not None and cell.substitution_disclosure.startswith("1 of the 2")
        rows = csv.DictReader(io.StringIO(export_projection(projection, fmt="csv").body))
        assert {(row["test_case_id"], row["substituted_deliveries"]) for row in rows} == {("tc-1", "0"), ("tc-2", "1")}

    def test_a_cost_cell_built_only_from_substituted_deliveries_says_it_is_no_live_spend(self):
        """The export column was the only place this showed; the cell is what is read, so the cell says it."""
        run, _ = _run_with_results()
        seeded = [
            self._result(
                run, case=case, async_deliveries=[AsyncDelivery(tool="scout", status="delivered", substituted=True)]
            )
            for case in ("tc-1", "tc-2")
        ]

        table = self._cost_pivot([run], seeded)
        quality = self._cost_pivot([run], seeded, metric=METRIC_COMPOSITE)

        (cell,) = table.cells
        assert (cell.status, cell.n_substituted, cell.n) == (CELL_MEASURED, 2, 2)
        assert cell.substitution_disclosure is not None and "no live run's spend" in cell.substitution_disclosure
        assert "no live run's spend" in pivot_text(table)
        assert quality.cells[0].n_substituted == 2, "counted on every metric"
        assert quality.cells[0].substitution_disclosure is None, "the dollars sentence is the cost pivot's"

    def test_a_cost_cell_with_nothing_substituted_carries_no_flag(self):
        run, _ = _run_with_results()
        (cell,) = self._cost_pivot([run], [self._result(run, case="tc-1")]).cells

        assert (cell.n_substituted, cell.substitution_disclosure) == (0, None)

    def test_the_export_names_which_result_replayed(self):
        """#658: each row carries its run's cassette mode and its substituted-delivery count as columns."""
        live, _ = _run_with_results(id="run-live", cassette_mode="off")
        replayed, _ = _run_with_results(id="run-replay", cassette_mode="replay")
        results = [self._result(live, case="tc-1"), self._result(replayed, case="tc-2", cost_roles=self.PRICED)]

        export = export_projection(
            project_score_records([live, replayed], results, profile=_JUDGED_HOST, archived_run_ids=None), fmt="csv"
        )

        rows = [row for row in csv.DictReader(io.StringIO(export.body)) if row["metric"] == METRIC_COST_USD]
        assert {row["run_id"]: row["cassette_mode"] for row in rows} == {"run-live": "off", "run-replay": "replay"}
        assert {row["substituted_deliveries"] for row in rows} == {"0"}
        assert "judge" in next(row["cost_roles"] for row in rows if row["run_id"] == "run-replay")

    def test_a_variant_key_cell_spanning_an_identity_bump_names_the_versions(self):
        """#672: one digest stamped at two predicate versions pools into one cell, and the table says so."""
        run, _ = _run_with_results()
        results = [
            self._result(run, case="tc-1", variant_key="vk-1", identity_version=IDENTITY_VERSION - 1),
            self._result(run, case="tc-2", variant_key="vk-1", identity_version=IDENTITY_VERSION),
        ]

        table = self._cost_pivot([run], results, row_factor="variant_key", metric=METRIC_COMPOSITE)

        (cell,) = table.cells
        assert cell.identity_versions == {"variant_key": [IDENTITY_VERSION - 1, IDENTITY_VERSION]}
        assert table.identity_pooling_disclosure is not None
        assert f"v{IDENTITY_VERSION - 1}" in table.identity_pooling_disclosure

    def test_a_variant_key_pivot_over_one_version_discloses_nothing(self):
        run, _ = _run_with_results()
        results = [self._result(run, case=case, variant_key="vk-1") for case in ("tc-1", "tc-2")]

        table = self._cost_pivot([run], results, row_factor="variant_key", metric=METRIC_COMPOSITE)

        assert table.cells[0].identity_versions == {}
        assert table.identity_pooling_disclosure is None

    def test_pivoting_the_key_against_its_version_separates_them(self):
        run, _ = _run_with_results()
        results = [
            self._result(run, case="tc-1", variant_key="vk-1", identity_version=IDENTITY_VERSION - 1),
            self._result(run, case="tc-2", variant_key="vk-1", identity_version=IDENTITY_VERSION),
        ]

        table = self._cost_pivot(
            [run], results, row_factor="variant_key", column_factor="variant_identity_version", metric=METRIC_COMPOSITE
        )

        assert len(table.cells) == 2
        assert table.identity_pooling_disclosure is None


class TestPivotDescribesWhatACellHolds:
    """A cell holds an aggregate, so it is described by the aggregate's descriptor."""

    def test_composite_resolves_to_the_registered_aggregate_not_the_unclassified_arm(self):
        """`composite` is not a seeded name; without the mapping every cell claims family=None."""
        table = compute_pivot(
            [_record()], row_factor="model", column_factor="template_id", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
        )

        assert table.measure.name == "mean_composite"
        assert table.measure.family == "composite"
        assert table.measure.higher_is_better is True

    def test_cost_resolves_to_its_aggregate_and_keeps_its_unit_and_direction(self):
        table = compute_pivot(
            [_record(metric=METRIC_COST_USD, value=0.1)],
            row_factor="model",
            column_factor="template_id",
            metric=METRIC_COST_USD,
            profile=_JUDGED_HOST,
        )

        assert table.measure.name == "mean_cost_usd"
        assert table.measure.unit == "usd"
        assert table.measure.higher_is_better is False

    def test_a_projected_measure_with_no_registered_aggregate_falls_back_honestly(self, monkeypatch):
        """The forgetting case: a projected metric whose aggregate the registry does not describe.

        The unclassified arm holds the strictest transferability class, so a
        measure nobody has described never *widens* a pooling decision it was
        never vetted for. Exercised by dropping a real mapping rather than by
        inventing a metric name, because an unprojected name is now refused
        outright — the two failure modes are different and both need to stay so.
        """

        def pivot():
            return compute_pivot(
                [_record(metric=METRIC_COST_USD, value=1.0)],
                row_factor="model",
                column_factor="template_id",
                metric=METRIC_COST_USD,
                profile=_JUDGED_HOST,
            )

        # The aggregate the cell is described by, read off the surface rather than restated; then
        # the registry is made to forget it, which is the state an unmapped aggregate reaches.
        aggregate = pivot().measure.name
        assert aggregate != METRIC_COST_USD, "the cell must be described by its aggregate for this to mean anything"
        monkeypatch.delitem(METRIC_DESCRIPTORS, aggregate)

        table = pivot()

        assert table.measure.family is None
        assert table.measure.transferability_class == "scenario_bound"

    def test_a_metric_the_projection_never_emits_is_refused_not_answered_empty(self):
        """A typo must not return a correct-looking empty grid.

        Unlike an axis, the metric set is genuinely closed — the projection is
        its only producer — so a name outside it can only be a mistake, and an
        empty table would be indistinguishable from an empty scope. Refused
        in `pivot` rather than at an adapter, so both surfaces refuse it.
        """
        with pytest.raises(PivotError, match="unknown metric"):
            compute_pivot(
                [_record()], row_factor="model", column_factor="template_id", metric="compsite", profile=_JUDGED_HOST
            )


class TestNormalizeStatusFilter:
    """The one seam both read methods and both surfaces resolve `status` through.

    Worth a direct table test rather than coverage-by-consequence: the two
    surfaces once disagreed about a blank `?status=`, and now that the adapters
    pass the value through untouched, every distinction this function draws is
    the *only* thing standing between those two readings. Case-folding in
    particular had no direct test — mutating `cleaned.lower() == "all"` to
    `cleaned == "all"` passed the whole suite.
    """

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, "completed"),
            ("", "completed"),
            ("   ", "completed"),
            ("all", None),
            ("ALL", None),
            ("All", None),
            ("  all  ", None),
            ("completed", "completed"),
            ("running", "running"),
            ("  running  ", "running"),
            ("budget_stopped", "budget_stopped"),
        ],
    )
    def test_resolves_every_caller_spelling(self, raw, expected):
        """Blank means unsupplied (not "all"); `all` lifts the filter at any casing.

        The two load-bearing defaults live here: an absent or blank filter means
        `completed`, and `all` lifts the filter whatever its casing. Everything
        else this function accepts is a status a run can actually carry — the
        refusal of anything else is pinned below.
        """
        from threetears.evals.kernel.status_filter import normalize_status_filter

        assert normalize_status_filter(raw) == expected

    @pytest.mark.parametrize("raw", ["banana", "Completed", "COMPLETED", "complete", "done", "everything", "any"])
    def test_refuses_a_status_no_run_can_carry(self, raw):
        """A filter that can match nothing is refused, never answered with an empty result.

        This row set replaces the earlier `("Completed", "Completed")` pin, which
        recorded that a case variant "passes through with its casing intact —
        this is a filter that matches nothing rather than a synonym", explicitly
        as "the existing contract, pinned here so a future casing change is a
        decision rather than a surprise". This is that decision: a filter
        that matches nothing is the defect, not the contract. `?status=banana`
        returned HTTP 200 with `subjects: []` and the whole corpus booked to
        `results_outside_queried_runs` — a well-formed EMPTY answer indistinguishable
        from the truthful "no runs of that kind exist", while the sibling
        `?metric=banana` on the same request already refused with a 422.

        The casing arm is refused rather than folded, deliberately: `all` is a
        sentinel this seam invents and never matches against stored data, while
        every real status is compared by exact equality to `EvalRun.status`
        downstream — as `metric` and `fmt` are against their own enumerations. A
        refusal that names the vocabulary teaches `completed` in one round-trip,
        so folding would buy nothing the message does not already give.
        """
        from threetears.evals.kernel.status_filter import StatusFilterError, normalize_status_filter

        with pytest.raises(StatusFilterError, match="unknown run status"):
            normalize_status_filter(raw)

    def test_the_refusal_enumerates_every_status_a_run_can_carry(self):
        """The message names the alternatives, and names them from the Literal.

        The whole value of refusing is that the caller learns the vocabulary
        without leaving the call, which is why this asserts the contents rather
        than that an error was raised. Read off `RUN_STATUSES` rather than a list
        written here: a second copy of the vocabulary in the test is the same
        drift the seam avoids by not writing one in the source.
        """
        from threetears.evals.schema.models import RUN_STATUSES
        from threetears.evals.kernel.status_filter import StatusFilterError, normalize_status_filter

        with pytest.raises(StatusFilterError) as excinfo:
            normalize_status_filter("banana")

        message = str(excinfo.value)
        assert "'banana'" in message
        assert all(status in message for status in RUN_STATUSES)
        # `all` is accepted and is not in RUN_STATUSES, so a message enumerating only
        # the statuses would refuse a caller their way back to the widest cohort —
        # the one an operator reaches for by hand most often.
        assert "'all'" in message

    def test_every_status_a_run_can_carry_is_accepted(self):
        """The accepted set IS the Literal — no status is refused that a run can hold.

        The failure this catches is the expensive direction of the same drift:
        adding a status to `EvalRunStatus` and leaving a hand-written accept-list
        behind refuses a real cohort, and refuses it loudly enough to look like a
        bug in the caller.
        """
        from threetears.evals.schema.models import RUN_STATUSES
        from threetears.evals.kernel.status_filter import normalize_status_filter

        assert {status: normalize_status_filter(status) for status in RUN_STATUSES} == {
            status: status for status in RUN_STATUSES
        }


class TestValidateStatusFilter:
    """The listing half of the same seam: unspecified means UNFILTERED, not `completed`.

    `list_runs` is an inventory rather than a verdict, so the one thing it must
    not inherit from the comparison surfaces is their default. Table-tested
    beside `TestNormalizeStatusFilter` for the reason that class gives — every
    distinction these two functions draw is now the only thing standing between
    two readings of the same argument — plus one this class adds: the two
    functions must stay identical everywhere EXCEPT that default, and the way
    they stop being identical is a second copy of the vocabulary appearing in one
    of them.
    """

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, None),
            ("", None),
            ("   ", None),
            ("all", None),
            ("ALL", None),
            ("  All  ", None),
            ("completed", "completed"),
            ("  running  ", "running"),
            ("budget_stopped", "budget_stopped"),
            ("cancelled", "cancelled"),
        ],
    )
    def test_an_unspecified_filter_is_no_filter(self, raw, expected):
        """Absent, blank and `all` all mean every run — the distinction from the sibling.

        The defect this pins: a scope of 9 runs
        answered `?status=all` with HTTP 200 and an empty list, because `all` was
        handed to the storage predicate as a status to match. The blank rows matter
        just as much in the other direction — routing this surface through
        `normalize_status_filter` would return `completed` here and quietly hide the
        two of those nine runs that were not.
        """
        from threetears.evals.kernel.status_filter import validate_status_filter

        assert validate_status_filter(raw) == expected

    @pytest.mark.parametrize("raw", ["banana", "Completed", "COMPLETED", "complete", "done", "everything", "any"])
    def test_refuses_a_status_no_run_can_carry(self, raw):
        """The same refusal as the comparison surfaces, on the same rows.

        A filter that can match nothing must not be answered with an empty list,
        which is how `?status=banana` used to read exactly like the truthful "this
        scope holds no runs". The casing arm is refused rather than folded for
        the reason `TestNormalizeStatusFilter` records: only `all` is a sentinel
        this seam invents, and every real status is compared by exact equality to
        `EvalRun.status` downstream.
        """
        from threetears.evals.kernel.status_filter import StatusFilterError, validate_status_filter

        with pytest.raises(StatusFilterError, match="unknown run status"):
            validate_status_filter(raw)

    def test_the_two_filters_differ_in_the_default_and_in_nothing_else(self):
        """One vocabulary, one refusal, two defaults — asserted rather than assumed.

        This is the property the split exists to hold, and the failure it guards
        against is the cheap-looking one: a status added to `EvalRunStatus`, or the
        refusal message reworded, in a copy that only one of the two functions
        reads. Compared over every input EXCEPT the unspecified ones, where the two
        are supposed to disagree — those are pinned per-function above.
        """
        from threetears.evals.schema.models import RUN_STATUSES
        from threetears.evals.kernel.status_filter import (
            StatusFilterError,
            normalize_status_filter,
            validate_status_filter,
        )

        agreeing_inputs = [*sorted(RUN_STATUSES), "all", "ALL", "  all  "]
        assert [validate_status_filter(raw) for raw in agreeing_inputs] == [
            normalize_status_filter(raw) for raw in agreeing_inputs
        ]

        with pytest.raises(StatusFilterError) as validated:
            validate_status_filter("banana")
        with pytest.raises(StatusFilterError) as normalized:
            normalize_status_filter("banana")
        assert str(validated.value) == str(normalized.value)


class TestClosedSetsAreClosed:
    """Pin the two closed-set claims that are otherwise prose maintained by hand.

    Both `PROJECTED_METRICS` and `_AGGREGATE_OF_OBSERVATION` carry comments
    asserting a closed set, and the mechanism behind each is a human remembering
    to edit two places in one commit. `describe_measure` is *total*, so the
    second one fails silently: an unmapped row metric resolves to `family=None`
    at the strictest transferability class, and the pivot header then reports an
    unclassified measure while the pooling gate fires on a fabricated class. A
    total function that answers for every input cannot tell you that you forgot
    something — so the claim itself is pinned here rather than left to prose.
    """

    def test_every_projected_metric_maps_to_an_aggregate(self):
        """Adding a metric to the projection without its aggregate is the forgetting case."""
        from threetears.evals.analysis import reporting

        unmapped = sorted(metric for metric in reporting.PROJECTED_METRICS if not _catalog_names_of(metric))

        assert not unmapped, (
            f"{unmapped} is projected but no registered aggregate resolves to it, so every "
            "cell of that metric would silently claim family=None at the strictest class"
        )

    def test_the_projection_emits_nothing_outside_the_declared_set(self):
        """The other direction: `PROJECTED_METRICS` must not fall behind what is emitted.

        The two tests above quantify over the declared set, so a metric added to
        `project_score_records` and forgotten here would satisfy both while
        being refused by `pivot` as an unknown metric — the closed set is what
        makes that refusal safe, and it is only closed if it covers everything
        the projection actually writes.
        """
        from threetears.evals.analysis import reporting

        run, result = _run_with_results()
        emitted = {
            record.metric
            for record in reporting.project_score_records(
                [run], [result], profile=_JUDGED_HOST, archived_run_ids=None
            ).records
        }

        assert emitted, "the fixture must project at least one row for this to mean anything"
        assert emitted <= reporting.PROJECTED_METRICS, (
            f"{sorted(emitted - reporting.PROJECTED_METRICS)} is emitted by the projection but not "
            "declared in PROJECTED_METRICS, so pivot would refuse it as an unknown metric"
        )

    def test_every_projected_metric_resolves_to_a_described_aggregate(self):
        """The mapping must land on a *registered* name, not merely on some string."""
        from threetears.evals.analysis import reporting

        undescribed = sorted(
            metric
            for metric in reporting.PROJECTED_METRICS
            if not any(
                describe_measure(name, _JUDGED_HOST.measures).family is not None for name in _catalog_names_of(metric)
            )
        )

        assert not undescribed, (
            f"{undescribed} maps to an aggregate the metric registry does not describe — the "
            "unclassified arm is the honest answer for an unknown name, never for a projected one"
        )


class TestModelShapes:
    """The projection models round-trip, since export and responses both re-parse them."""

    def test_score_record_rejects_an_unknown_field(self):
        """`extra='forbid'` is what makes a typo a failure instead of a silent drop.

        Built from a VALID payload plus one extra key. An earlier version passed a
        dict of only the unknown field, so it raised on the missing required ones
        and would have passed even with `extra='ignore'` — green for the wrong
        reason, which is worse than absent.
        """
        run, result = _run_with_results()
        valid = (
            project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records[0].model_dump()
        )

        with pytest.raises(ValidationError):
            ScoreRecord.model_validate({**valid, "definitely_not_a_field": 1})

    def test_comparison_set_rejects_an_unknown_field(self):
        run, _ = _run_with_results()
        valid = compute_comparison_sets([run], profile=_JUDGED_HOST).comparison_sets[0].model_dump()

        with pytest.raises(ValidationError):
            ComparisonSet.model_validate({**valid, "definitely_not_a_field": 1})

    def test_score_record_round_trips_through_its_own_dump(self):
        run, result = _run_with_results()
        row = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records[0]

        assert ScoreRecord.model_validate(row.model_dump()) == row

    def test_composite_metric_name_is_stable(self):
        """Surfaces look descriptors up by these names; renaming one breaks the registry join."""
        assert METRIC_COMPOSITE == "composite"
        assert METRIC_COST_USD == "cost_usd"


class TestPredictionReservation:
    """The predicted-value slot is reserved now, computed by no engine yet."""

    def test_every_pivoted_cell_carries_a_null_prediction(self):
        """Nothing in this plan populates a prediction — the slot exists, empty, for the engine to fill later."""
        records = [_record(case="tc-1", value=1.0), _record(case="tc-2", value=0.0)]

        table = compute_pivot(
            records, row_factor="test_case_id", column_factor="model", metric=METRIC_COMPOSITE, profile=_JUDGED_HOST
        )

        assert table.cells, "expected cells to assert over"
        assert all(cell.predicted is None for cell in table.cells)

    def test_predicted_value_carries_the_reserved_field_set(self):
        """The six-field contract is the lock-in the engine lands against without a surface migration."""
        predicted = PredictedValue(
            value=0.8,
            interval_low=0.7,
            interval_high=0.9,
            method_id="latency-linreg-v1",
            trust=0.6,
            computed_at="2026-07-23T00:00:00Z",
        )

        assert set(PredictedValue.model_fields) == {
            "value",
            "interval_low",
            "interval_high",
            "method_id",
            "trust",
            "computed_at",
        }
        assert PredictedValue.model_validate(predicted.model_dump()) == predicted

    def test_a_prediction_may_be_point_only(self):
        """A point estimate with no interval is representable — interval bounds are optional refinements."""
        predicted = PredictedValue(value=0.8, method_id="m", computed_at="2026-07-23T00:00:00Z")

        assert (predicted.interval_low, predicted.interval_high, predicted.trust) == (None, None, None)


class TestProgramBudget:
    """Budget counts every run's real spend — the one lens that never excludes."""

    def test_sums_per_run_cost_with_a_cumulative_series_in_time_order(self):
        run_a = make_eval_run(status="completed", created_at="2026-07-01T00:00:00Z")
        run_b = make_eval_run(status="completed", created_at="2026-07-02T00:00:00Z")
        results = [
            make_eval_result(eval_run_id=run_a.id, cost_usd=0.10),
            make_eval_result(eval_run_id=run_a.id, cost_usd=0.05),
            make_eval_result(eval_run_id=run_b.id, cost_usd=0.20),
        ]

        # Passed out of creation order to prove the rows are re-sorted, not echoed.
        budget = compute_program_budget([run_b, run_a], results)

        assert budget.total_cost_usd == pytest.approx(0.35)
        assert budget.n_runs == 2
        by_run = {r.run_id: r for r in budget.runs}
        assert (by_run[run_a.id].cost_usd, by_run[run_a.id].n_results) == (pytest.approx(0.15), 2)
        # Cumulative in created_at order: run_a (0.15) then run_b (0.35).
        assert [r.run_id for r in budget.runs] == [run_a.id, run_b.id]
        assert [r.cumulative_cost_usd for r in budget.runs] == pytest.approx([0.15, 0.35])

    def test_budget_includes_a_failed_run_the_quality_path_excludes(self):
        """The asymmetry, in one test: budget counts a failed run's spend; the completed-only quality path drops it.

        This one assertion is the whole load-bearing contract of the budget lens.
        If it is ever removed, the two views have silently fused and a
        failed run's real dollars vanish from the bill.
        """
        completed = make_eval_run(status="completed", created_at="2026-07-01T00:00:00Z")
        failed = make_eval_run(status="failed", created_at="2026-07-02T00:00:00Z")
        results = [
            make_eval_result(eval_run_id=completed.id, cost_usd=0.10),
            make_eval_result(eval_run_id=failed.id, cost_usd=0.40),
        ]

        budget = compute_program_budget([completed, failed], results)

        assert budget.total_cost_usd == pytest.approx(0.50)
        assert budget.n_incomplete_runs == 1
        assert budget.incomplete_cost_usd == pytest.approx(0.40)

        # The quality path is `project_score_records` over the completed runs only
        # (the `status="completed"` default every quality surface applies). The
        # failed run's spend is absent from its cost rows — the exact spend budget
        # counts and quality drops.
        quality = project_score_records(
            [completed], results, known_run_ids={completed.id, failed.id}, profile=_JUDGED_HOST, archived_run_ids=None
        )
        cost_rows = [r for r in quality.records if r.metric == METRIC_COST_USD]
        assert [r.value for r in cost_rows] == [0.10]

    def test_an_exhausted_run_carries_its_accumulated_cost_not_zero(self):
        """A budget-stopped run spent real dollars up to the ceiling — never $0."""
        exhausted = make_eval_run(status="budget_stopped", created_at="2026-07-01T00:00:00Z")
        results = [make_eval_result(eval_run_id=exhausted.id, cost_usd=0.99)]

        budget = compute_program_budget([exhausted], results)

        assert budget.runs[0].cost_usd == pytest.approx(0.99)
        assert budget.n_incomplete_runs == 1

    def test_the_dropped_spend_splits_in_flight_from_ended_without_completing(self):
        """Two halves, because they point opposite ways for a budget decision.

        A run still going will normally deliver the results its dollars are
        buying; a cancelled one never will. Pooled into one "non-completed"
        figure the live half reads as money already lost, which is the misreading
        the split exists to retire — and the halves must still sum to the whole,
        or the bill and its own breakdown disagree.
        """
        completed = make_eval_run(status="completed", created_at="2026-07-01T00:00:00Z")
        running = make_eval_run(status="running", created_at="2026-07-02T00:00:00Z")
        cancelled = make_eval_run(status="cancelled", created_at="2026-07-03T00:00:00Z")
        results = [
            make_eval_result(eval_run_id=completed.id, cost_usd=0.10),
            make_eval_result(eval_run_id=running.id, cost_usd=0.02),
            make_eval_result(eval_run_id=cancelled.id, cost_usd=0.40),
        ]

        budget = compute_program_budget([completed, running, cancelled], results)

        assert (budget.n_in_flight_runs, budget.in_flight_statuses) == (1, ["running"])
        assert budget.in_flight_cost_usd == pytest.approx(0.02)
        assert (budget.n_terminal_incomplete_runs, budget.terminal_incomplete_statuses) == (1, ["cancelled"])
        assert budget.terminal_incomplete_cost_usd == pytest.approx(0.40)
        # The halves are a partition of the bucket, not two overlapping views of it.
        assert budget.n_in_flight_runs + budget.n_terminal_incomplete_runs == budget.n_incomplete_runs
        assert budget.in_flight_cost_usd + budget.terminal_incomplete_cost_usd == pytest.approx(
            budget.incomplete_cost_usd
        )

    def test_a_pending_run_is_in_flight_and_not_reported_as_a_failure(self):
        """A queued run has spent nothing yet and has failed at nothing — both halves must say so."""
        pending = make_eval_run(status="pending", created_at="2026-07-01T00:00:00Z")

        budget = compute_program_budget([pending], [])

        assert (budget.n_in_flight_runs, budget.in_flight_statuses) == (1, ["pending"])
        assert (budget.n_terminal_incomplete_runs, budget.terminal_incomplete_statuses) == (0, [])

    def test_every_terminal_non_completed_status_lands_in_the_terminal_half(self):
        """The classification is by terminality, not by a hand-kept list of failure statuses.

        ``budget_stopped`` is the one that has actually gone missing from such a
        list before — it is an honest terminal outcome rather than a failure, and
        a bucket that enumerates statuses rather than deriving them is how it
        stops being counted.
        """
        runs = [
            make_eval_run(status=status, created_at=f"2026-07-0{i}T00:00:00Z")
            for i, status in enumerate(("failed", "cancelled", "budget_stopped"), start=1)
        ]

        budget = compute_program_budget(runs, [])

        assert budget.terminal_incomplete_statuses == ["budget_stopped", "cancelled", "failed"]
        assert budget.n_in_flight_runs == 0

    def test_a_run_the_quality_surfaces_would_drop_is_still_billed(self):
        """Budget counts spend the quality views leave out — that money was real either way.

        The identity-less run this used to model cannot be built now, so the surviving case is a
        run that ended without completing: dropped by every quality view, billed here.
        """
        run = make_eval_run(status="failed", created_at="2026-07-01T00:00:00Z")
        results = [make_eval_result(eval_run_id=run.id, cost_usd=0.33)]

        budget = compute_program_budget([run], results)

        assert budget.total_cost_usd == pytest.approx(0.33)
        assert budget.runs[0].n_results == 1

    def test_a_run_with_no_results_is_a_zero_row_not_absent(self):
        run = make_eval_run(status="running", created_at="2026-07-01T00:00:00Z")

        budget = compute_program_budget([run], [])

        assert budget.runs[0].cost_usd == 0.0
        assert budget.runs[0].n_results == 0

    def test_spend_on_an_unlisted_run_is_counted_not_dropped(self):
        """Real spend whose run is absent from the set lands in the total as unattributed, never silently gone."""
        listed = make_eval_run(status="completed", created_at="2026-07-01T00:00:00Z")
        results = [
            make_eval_result(eval_run_id=listed.id, cost_usd=0.10),
            make_eval_result(eval_run_id="run-not-listed", cost_usd=0.07),
        ]

        budget = compute_program_budget([listed], results)

        assert budget.unattributed_cost_usd == pytest.approx(0.07)
        assert budget.total_cost_usd == pytest.approx(0.17)


class TestOrphanedRuns:
    """Runs no campaign holds — paid data every analysis surface silently skips."""

    def test_a_run_in_no_campaign_is_reported_with_its_spend(self):
        claimed, orphan = make_eval_run(), make_eval_run()
        results = [
            make_eval_result(eval_run_id=claimed.id, cost_usd=1.0),
            make_eval_result(eval_run_id=orphan.id, cost_usd=4.0),
            make_eval_result(eval_run_id=orphan.id, cost_usd=2.0),
        ]

        report = compute_orphaned_runs([claimed, orphan], results, [[claimed.id]])

        assert [row.run_id for row in report.orphaned_runs] == [orphan.id]
        assert report.orphaned_cost_usd == 6.0
        assert report.n_orphaned == 1
        assert report.n_runs_in_scope == 2

    def test_a_run_held_by_any_campaign_is_not_an_orphan(self):
        """Membership is a union across campaigns, not per-campaign."""
        run = make_eval_run()

        report = compute_orphaned_runs([run], [], [["someone-else"], [run.id]])

        assert report.orphaned_runs == []
        assert report.n_campaigns_scanned == 2

    def test_no_campaigns_makes_every_run_an_orphan_and_says_why(self):
        """True, and alarming without the denominator that explains it."""
        run = make_eval_run()

        report = compute_orphaned_runs([run], [], [])

        assert report.n_orphaned == 1
        assert report.n_campaigns_scanned == 0

    def test_archived_orphans_are_counted_separately_not_dropped(self):
        """A retired run still spent its dollars; a curated exclusion is not a leak."""
        live, retired = make_eval_run(), make_eval_run(archived=True)
        results = [
            make_eval_result(eval_run_id=live.id, cost_usd=3.0),
            make_eval_result(eval_run_id=retired.id, cost_usd=7.0),
        ]

        report = compute_orphaned_runs([live, retired], results, [])

        assert report.n_orphaned == 2
        assert report.orphaned_cost_usd == 10.0
        assert report.n_archived_orphans == 1
        assert report.archived_orphan_cost_usd == 7.0

    def test_an_orphan_with_no_results_is_reported_at_zero_not_omitted(self):
        """A run that produced nothing is still unclaimed, and still a finding."""
        run = make_eval_run()

        report = compute_orphaned_runs([run], [], [])

        assert [row.cost_usd for row in report.orphaned_runs] == [0.0]
        assert [row.n_results for row in report.orphaned_runs] == [0]

    def test_rows_are_in_creation_order(self):
        late = make_eval_run(created_at="2026-07-09T00:00:00Z")
        early = make_eval_run(created_at="2026-07-01T00:00:00Z")

        report = compute_orphaned_runs([late, early], [], [])

        assert [row.run_id for row in report.orphaned_runs] == [early.id, late.id]


class TestProgramBudgetComposition:
    """A cumulative total says what its dollars are made of, or admits it cannot.

    This is the one view that adds spend across time, so it is the one place totals from
    either side of a composition change get summed — and the sum looks identical whether
    or not that happened. The disclosure is what makes the difference readable.
    """

    UNPRICED = ["candidate", "inner_agent", "judge", "simulator"]
    PRICED = ["candidate", "inner_agent", "judge", "simulator", "external"]

    def test_a_total_spanning_two_conventions_names_both(self):
        early = make_eval_run(status="completed", created_at="2026-07-01T00:00:00Z")
        late = make_eval_run(status="completed", created_at="2026-08-01T00:00:00Z")
        results = [
            make_eval_result(eval_run_id=early.id, cost_usd=0.10, cost_roles=self.UNPRICED),
            make_eval_result(eval_run_id=late.id, cost_usd=0.20, cost_roles=self.PRICED),
        ]

        budget = compute_program_budget([early, late], results)

        assert budget.cost_compositions == [self.UNPRICED, self.PRICED]
        by_run = {r.run_id: r for r in budget.runs}
        assert by_run[early.id].cost_compositions == [self.UNPRICED]
        assert by_run[late.id].cost_compositions == [self.PRICED]

    def test_a_uniform_corpus_reports_one_composition(self):
        """One entry is the honest answer for a total whose parts all mean the same thing."""
        run = make_eval_run(status="completed", created_at="2026-07-01T00:00:00Z")
        results = [
            make_eval_result(eval_run_id=run.id, cost_usd=0.10, cost_roles=self.PRICED),
            make_eval_result(eval_run_id=run.id, cost_usd=0.20, cost_roles=self.PRICED),
        ]

        budget = compute_program_budget([run], results)

        assert budget.cost_compositions == [self.PRICED]


class TestExport:
    """The projection's flat rows serialize to CSV/JSON without losing a coordinate."""

    def test_csv_carries_every_declared_coordinate_as_a_column(self):
        run, result = _run_with_results(judge_model="judge-a", simulator_model="sim-b")
        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        text = export_records_csv(records)

        reader = csv.DictReader(io.StringIO(text))
        header = reader.fieldnames or []
        # The lifted role coordinates are columns like any other — the export
        # inherits the role widening with no per-field work.
        #
        # `scope_id` is here because the header is derived from `ScoreRecord.model_fields`, so a
        # field rename silently renames a CSV column that saved pandas/DuckDB pipelines key on —
        # and a route serving it would declare no `response_model`, so no schema gate
        # can see it. Derivation is the design; what was missing was anything that goes red.
        for column in (
            "subject_id",
            "scope_id",
            "run_id",
            "model",
            "metric",
            "value",
            "outcome",
            "judge_model",
            "simulator_model",
        ):
            assert column in header
        assert "universe_id" not in header, "the host's partition noun must not come back as a column"
        rows = list(reader)
        assert all(row["judge_model"] == "judge-a" for row in rows)
        assert all(row["simulator_model"] == "sim-b" for row in rows)

    def test_open_factor_keys_flatten_to_their_own_columns(self):
        """A bake-off's varied knob is a first-class pandas column, not a nested blob."""
        run, result = _run_with_results(**_overlaid(prompt_style="verbose", field_aliases={"vendor_name": "supplier"}))
        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        reader = csv.DictReader(io.StringIO(export_records_csv(records)))
        header = reader.fieldnames or []

        assert "extractor.prompt_style" in header
        assert "extractor.field_aliases.vendor_name" in header
        assert "factors" not in header, "the open-coordinate map must flatten, never appear as a blob column"
        rows = list(reader)
        assert rows[0]["extractor.prompt_style"] == "verbose"

    def test_a_null_value_is_a_blank_cell_not_a_zero(self):
        """An unmeasured observation must export as empty, ingesting to NaN — never a fabricated 0."""
        # An infra-excluded result composites to None, so its composite row carries no value.
        run, _ = _run_with_results()
        failed = make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, infra_error="timeout")
        records = project_score_records([run], [failed], profile=_JUDGED_HOST, archived_run_ids=None).records

        reader = csv.DictReader(io.StringIO(export_records_csv(records)))
        rows = {row["metric"]: row for row in reader}

        assert rows["composite"]["value"] == "", "an unmeasured composite must be blank, not 0"

    def test_a_run_without_a_factor_leaves_that_column_blank(self):
        """Columns stay aligned across rows even when only some runs carried an override — the pandas ingest contract."""
        with_factor, r1 = _run_with_results(**_overlaid(prompt_style="terse"))
        without_factor, r2 = _run_with_results(subject_id="ent-maple")
        records = project_score_records(
            [with_factor, without_factor], [r1, r2], profile=_JUDGED_HOST, archived_run_ids=None
        ).records

        reader = csv.DictReader(io.StringIO(export_records_csv(records)))
        rows = list(reader)

        # Every row has the same columns (DictReader guarantees alignment); the
        # run of no kind carrying the knob sits at a blank cell, not a missing column.
        assert all("extractor.prompt_style" in row for row in rows)
        blanks = [row for row in rows if row["run_id"] == without_factor.id]
        assert blanks and all(row["extractor.prompt_style"] == "" for row in blanks)

    def test_csv_neutralizes_a_formula_injection_in_a_user_authored_field(self):
        """A leading =/+/-/@ in an operator-authored override can't smuggle a spreadsheet formula."""
        run, result = _run_with_results(**_overlaid(instructions="=cmd|'/C calc'!A1"))
        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        reader = csv.DictReader(io.StringIO(export_records_csv(records)))
        rows = list(reader)

        # A formula-triggering overlay value must be quoted to text, not written raw.
        assert rows[0]["extractor.instructions"] == "'=cmd|'/C calc'!A1"

    def test_csv_safe_quotes_formula_triggers_but_preserves_numbers(self):
        """The guard forces formulas to text yet leaves legitimate numbers verbatim for pandas."""

        run, result = _run_with_results()
        dumped = (
            project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None)
            .records[0]
            .model_dump(mode="json")
        )

        def exported(value: str) -> str:
            """The cell ``value`` becomes as a subject label, in the JSON-safe dump the export also accepts."""
            row = {**dumped, "subject_label": value}
            return next(csv.DictReader(io.StringIO(export_records_csv([row]))))["subject_label"]

        def exported_factor(value: str) -> str:
            """The cell ``value`` becomes as an overlay, an open-coordinate column."""
            run, result = _run_with_results(**_overlaid(instructions=value))
            records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records
            return next(csv.DictReader(io.StringIO(export_records_csv(records))))["extractor.instructions"]

        # Formula payloads — every trigger char, none of them a number — get quoted.
        assert exported("=SUM(A1:A9)") == "'=SUM(A1:A9)"
        assert exported("+cmd|'/C calc'!A1") == "'+cmd|'/C calc'!A1"
        assert exported("@import") == "'@import"
        assert exported("-2+3+cmd") == "'-2+3+cmd"  # starts '-', but not a number
        assert exported("\tinjected") == "'\tinjected"
        # Legitimate values are untouched — negative deltas and signed floats must
        # still ingest to pandas as numbers, and ordinary strings stay as-is.
        assert exported("-0.5") == "-0.5"
        assert exported("+1.5e3") == "+1.5e3"
        assert exported("0.87") == "0.87"
        assert exported("sonnet") == "sonnet"
        assert exported_factor("=SUM(A1:A9)") == "'=SUM(A1:A9)"
        assert exported_factor("-0.5") == "-0.5"
        assert exported_factor("") == ""

    def test_json_export_round_trips_the_whole_projection(self):
        run, result = _run_with_results(judge_model="judge-a")
        projection = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None)

        body = serialize_export(projection, fmt="json")

        reparsed = ScoreProjection.model_validate(json.loads(body))
        assert reparsed == projection
        # Exclusions travel with the rows, so an all-excluded corpus never reads as empty.
        assert "exclusions" in json.loads(body)

    def test_csv_serialization_matches_the_shared_helper(self):
        run, result = _run_with_results()
        projection = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None)

        assert serialize_export(projection, fmt="csv") == export_records_csv(projection.records)

    def test_an_unknown_format_is_refused_not_defaulted(self):
        run, result = _run_with_results()
        projection = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None)

        with pytest.raises(ExportError, match="unknown export format"):
            serialize_export(projection, fmt="parquet")


class TestExportCarriesACodeGradedRunsOwnGrade:
    """The export was judge-centric, so a code-graded run exported no grade at all.

    A classifier cell has no rubric dims, so its composite is null and its score row is the
    dimensionless null — and its actual grade, which the runner stamps verbatim onto
    ``EvalResult.host_measures``, reached no column. *"Which case did this arm miss"* is the
    first question a classifier campaign asks and the export could not answer it.
    """

    #: The bank: two cases, and the arm gets the second one wrong.
    RIGHT = "case-direct"
    WRONG = "case-ambient"

    @classmethod
    def _code_graded(cls):
        """A classifier-shaped run: no rubric dims, the grade on ``host_measures``.

        Returns:
            Tuple of (run, results) — one right cell and one wrong cell.
        """
        run, _ = _run_with_results()
        right = make_eval_result(
            eval_run_id=run.id,
            scope_id=run.scope_id,
            test_case_id=cls.RIGHT,
            rubric_scores=[],
            goal_state_outcomes=[GoalStateOutcome(expression="classifier.label == expected", passed=True)],
            host_measures={"field_accuracy": 1.0, "parse_failure_rate": 0.0},
        )
        wrong = make_eval_result(
            eval_run_id=run.id,
            scope_id=run.scope_id,
            test_case_id=cls.WRONG,
            rubric_scores=[],
            goal_state_outcomes=[GoalStateOutcome(expression="classifier.label == expected", passed=False)],
            host_measures={"field_accuracy": 0.0, "parse_failure_rate": 0.0},
        )
        return run, [right, wrong]

    @classmethod
    def _exported_rows(cls, run, results):
        """Serialize through the real export seam and read the CSV back.

        Args:
            run: The run supplying coordinates.
            results: The observations to project.

        Returns:
            The parsed CSV rows.
        """
        projection = project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None)
        return list(csv.DictReader(io.StringIO(serialize_export(projection, fmt="csv"))))

    def test_a_reader_can_name_the_case_this_arm_missed(self):
        """The whole defect, stated as the question that could not be answered."""
        run, results = self._code_graded()

        rows = self._exported_rows(run, results)

        missed = [
            row["test_case_id"]
            for row in rows
            if row["metric"] == "composite" and row["host_measure:field_accuracy"] == "0.0"
        ]
        assert missed == [self.WRONG]

    def test_the_grade_is_one_row_per_observation_so_its_mean_is_the_arms_rate(self):
        """Repeating it on every row of a result would multiply every spreadsheet mean by the row count."""
        run, results = self._code_graded()

        rows = self._exported_rows(run, results)

        graded = [row for row in rows if row["host_measure:field_accuracy"] != ""]
        assert len(graded) == len(results), "exactly one row per result carries the grade"
        assert all(row["metric"] == "composite" for row in graded), "and it is the result's quality row"
        assert sum(float(row["host_measure:field_accuracy"]) for row in graded) / len(graded) == 0.5

    def test_a_second_graded_axis_is_another_column_and_not_another_shape(self):
        """The host catalogue is open, so a kind that grades on two axes must cost no export work."""
        run, results = self._code_graded()

        rows = self._exported_rows(run, results)

        assert "host_measure:parse_failure_rate" in rows[0]
        assert "host_measures" not in rows[0], "the open map must flatten, never appear as a blob column"

    def test_a_code_graded_row_is_distinguishable_from_a_judged_one_rather_than_silently_emptier(self):
        """Both composites are null for an unscored judge run; only one of them carries a grade."""
        code_run, code_results = self._code_graded()
        judged_run, judged_result = _run_with_results(subject_id="ent-other")

        projection = project_score_records(
            [code_run, judged_run], [*code_results, judged_result], profile=_JUDGED_HOST, archived_run_ids=None
        )
        rows = list(csv.DictReader(io.StringIO(serialize_export(projection, fmt="csv"))))

        composites = [row for row in rows if row["metric"] == "composite"]
        judged = next(row for row in composites if row["run_id"] == judged_run.id)
        code = next(row for row in composites if row["test_case_id"] == self.WRONG)
        assert judged["value"] != "" and judged["host_measure:field_accuracy"] == ""
        assert code["value"] == "" and code["host_measure:field_accuracy"] == "0.0"

    def test_a_judge_graded_export_is_unchanged(self):
        """No grade column exists at all when nothing carries one, so saved pipelines keep their header."""
        run, result = _run_with_results()

        header = csv.DictReader(
            io.StringIO(
                serialize_export(
                    project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None), fmt="csv"
                )
            )
        ).fieldnames

        assert not [column for column in (header or []) if column.startswith("host_measure:")]

    def test_the_grade_rides_the_json_export_too(self):
        """CSV and JSON are one seam, so the grade cannot reach only the flat form."""
        run, results = self._code_graded()
        projection = project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None)

        reparsed = ScoreProjection.model_validate(json.loads(serialize_export(projection, fmt="json")))

        assert reparsed == projection
        graded = [r for r in reparsed.records if r.host_measures]
        assert {r.test_case_id for r in graded} == {self.RIGHT, self.WRONG}

    def test_no_other_row_may_carry_the_grade(self):
        """Structural, because a duplicate would be silently wrong rather than visibly so."""
        with pytest.raises(ValidationError, match="the composite row is the one row per result"):
            ScoreRecord(
                subject_id="ent-maple",
                scope_id="uni-1",
                run_id="run-1",
                result_id="res-1",
                test_case_id="tc-1",
                model="m1",
                k_iteration=1,
                metric="cost_usd",
                value=0.01,
                host_measures={"field_accuracy": 1.0},
                outcome="ok",
            )

    def test_the_grade_map_is_refused_as_a_pivot_axis(self):
        """Grouping by a cell's own measurement partitions the corpus by its answer."""
        run, results = self._code_graded()

        with pytest.raises(PivotError, match="not a coordinate"):
            compute_pivot(
                project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None).records,
                metric="composite",
                row_factor="host_measures",
                column_factor="model",
                profile=_JUDGED_HOST,
            )


class TestCostEstimate:
    """A proposed sweep's cost comes back as an interval from historical per-cell costs."""

    def _history(self, model, costs, *, cassette_mode="off", subject="ent-maple", template_id=None):
        """Build one run plus a result per cost, all for one model at one cassette mode."""
        snapshot = make_subject(subject, subject)
        run = make_eval_run(
            subject_snapshot=snapshot,
            cassette_mode=cassette_mode,
            candidate_model=model,
            **({"template_id": template_id} if template_id is not None else {}),
        )
        results = [
            make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, model=model, cost_usd=cost) for cost in costs
        ]
        return run, results

    def test_cost_is_matched_to_the_proposed_template_when_one_is_given(self):
        """A template fixes the test cases, so it drives cost harder than any other lever.

        Pooling across templates misprices a specific proposal in whichever direction the
        scope's mix leans, and underpricing is the harmful direction — the sweep meets
        its spend ceiling partway through, which is what the estimate exists to prevent.
        """
        cheap_run, cheap_results = self._history("sonnet", [0.10, 0.10], template_id="tpl-cheap")
        dear_run, dear_results = self._history("sonnet", [1.00, 1.00], template_id="tpl-dear")
        runs = [cheap_run, dear_run]
        results = [*cheap_results, *dear_results]

        # Unfiltered, the basis pools both and lands between them — right for "what does a
        # run here cost on average", wrong for either specific proposal.
        pooled = compute_estimate_cost(runs, results, models=["sonnet"], k_runs=1, n_test_cases=1, profile=_JUDGED_HOST)
        assert pooled.cells[0].predicted.value == pytest.approx(0.55)

        cheap = compute_estimate_cost(
            runs, results, models=["sonnet"], k_runs=1, n_test_cases=1, template_id="tpl-cheap", profile=_JUDGED_HOST
        )
        assert cheap.cells[0].predicted.value == pytest.approx(0.10)
        assert cheap.cells[0].n_historical == 2

        dear = compute_estimate_cost(
            runs, results, models=["sonnet"], k_runs=1, n_test_cases=1, template_id="tpl-dear", profile=_JUDGED_HOST
        )
        assert dear.cells[0].predicted.value == pytest.approx(1.00)

    def test_an_ad_hoc_run_is_no_basis_for_any_proposed_template(self):
        """A run built from explicit test cases carries ``template_id=None``, so it matches none.

        Excluded rather than pooled: its cost is evidence about work the proposed sweep
        will not do. Note this is the ``None`` case specifically, not merely a *different*
        template — the two would pass the same assertion for different reasons.
        """
        snapshot = make_subject("ent-maple", "ent-maple")
        ad_hoc_run = make_eval_run(
            subject_snapshot=snapshot, cassette_mode="off", candidate_model="sonnet", template_id=None
        )
        assert ad_hoc_run.template_id is None, "fixture must be genuinely ad-hoc for this test to mean anything"
        ad_hoc_results = [
            make_eval_result(eval_run_id=ad_hoc_run.id, scope_id=ad_hoc_run.scope_id, model="sonnet", cost_usd=5.00)
        ]

        estimate = compute_estimate_cost(
            [ad_hoc_run],
            ad_hoc_results,
            models=["sonnet"],
            k_runs=1,
            n_test_cases=1,
            template_id="tpl-cheap",
            profile=_JUDGED_HOST,
        )

        assert estimate.cells[0].basis == "no_history"
        assert estimate.n_uncovered_models == 1

    def test_a_proposed_sweep_returns_an_interval_not_a_point(self):
        """The acceptance criterion: a real cost distribution yields a band, not a single number."""
        run, results = self._history("sonnet", [0.10, 0.20, 0.30])

        estimate = compute_estimate_cost(
            [run], results, models=["sonnet"], k_runs=1, n_test_cases=2, profile=_JUDGED_HOST
        )

        cell = estimate.cells[0]
        # Estimate = mean(0.20) × (2 cases × 1 k) = 0.40, with a non-degenerate band around it.
        assert cell.predicted.value == pytest.approx(0.40)
        assert cell.predicted.interval_low is not None and cell.predicted.interval_high is not None
        assert cell.predicted.interval_low < cell.predicted.value < cell.predicted.interval_high
        assert estimate.total_interval_low < estimate.total_estimated_cost < estimate.total_interval_high

    def test_cost_is_matched_to_the_proposed_cassette_mode(self):
        """A replay estimate rests on replay history or admits none — it never borrows a live run's cost."""
        live_run, live_results = self._history("sonnet", [1.00, 1.00], cassette_mode="off")

        # No replay history exists — a replay proposal must report no basis, not
        # silently reuse the expensive live history.
        estimate = compute_estimate_cost(
            [live_run],
            live_results,
            models=["sonnet"],
            k_runs=1,
            n_test_cases=1,
            cassette_mode="replay",
            profile=_JUDGED_HOST,
        )

        assert estimate.cells[0].basis == "no_history"
        assert estimate.total_estimated_cost is None
        assert estimate.n_uncovered_models == 1

    def test_a_model_with_no_history_is_uncovered_not_zero(self):
        run, results = self._history("sonnet", [0.10, 0.10])

        estimate = compute_estimate_cost(
            [run], results, models=["sonnet", "haiku"], k_runs=1, n_test_cases=1, profile=_JUDGED_HOST
        )

        by_model = {cell.model: cell for cell in estimate.cells}
        assert by_model["haiku"].basis == "no_history"
        assert by_model["haiku"].predicted is None
        assert estimate.n_uncovered_models == 1
        # The covered model still contributes; the total prices only what it could.
        assert estimate.total_estimated_cost == pytest.approx(by_model["sonnet"].predicted.value)

    def test_a_single_historical_observation_has_unknown_spread_not_zero(self):
        run, results = self._history("sonnet", [0.10])

        estimate = compute_estimate_cost(
            [run], results, models=["sonnet"], k_runs=1, n_test_cases=1, profile=_JUDGED_HOST
        )

        cell = estimate.cells[0]
        assert cell.n_historical == 1
        assert cell.predicted.value == pytest.approx(0.10)
        # One sample: the point stands, the spread is unknown — never a false ± 0.
        assert (cell.predicted.interval_low, cell.predicted.interval_high) == (None, None)

    def test_the_subject_filter_narrows_the_historical_basis(self):
        maple_run, maple_results = self._history("sonnet", [0.10, 0.10], subject="ent-maple")
        bea_run, bea_results = self._history("sonnet", [5.00, 5.00], subject="ent-bea")

        estimate = compute_estimate_cost(
            [maple_run, bea_run],
            maple_results + bea_results,
            models=["sonnet"],
            k_runs=1,
            n_test_cases=1,
            subject_id="ent-maple",
            profile=_JUDGED_HOST,
        )

        # Only Maple's cheap history counts — Bea's expensive runs are excluded.
        assert estimate.cells[0].mean_cost_per_observation == pytest.approx(0.10)

    def test_the_basis_discloses_the_compositions_its_mean_was_drawn_across(self):
        """A mean pooled over two cost conventions is not one distribution, and looks like one.

        The estimate deliberately does NOT match history to composition the way it matches
        it to cassette mode — a proposed run has not resolved its own rate card yet, so
        there is nothing to match against. Disclosure is what is available, and it is what
        the pooled number cannot show on its own.
        """
        run, results = self._history("sonnet", [0.10, 0.30])
        results[0].cost_roles = ["candidate", "inner_agent", "judge", "simulator"]
        results[1].cost_roles = ["candidate", "inner_agent", "judge", "simulator", "external"]

        estimate = compute_estimate_cost(
            [run], results, models=["sonnet"], k_runs=1, n_test_cases=1, profile=_JUDGED_HOST
        )

        cell = estimate.cells[0]
        assert cell.mean_cost_per_observation == pytest.approx(0.20)
        assert cell.basis_cost_compositions == [
            ["candidate", "inner_agent", "judge", "simulator"],
            ["candidate", "inner_agent", "judge", "simulator", "external"],
        ]

    def test_a_two_observation_basis_publishes_no_band_at_all(self):
        """The failure, at illustrative numbers.

        Two costs from one k=1 run, 2.5% apart, price the identical k=3 sweep of those
        same two cases at $0.1956 [$0.1909 – $0.2003]. A sweep coming in at $0.1720 is
        12.1% under the point and 9.9% under the lower bound. A dispersion over two
        near-identical samples describes the sampling noise of that pair and nothing
        else: not per-iteration variance, not conversation-length variance, neither of
        which a k=1 basis can observe. Two points give exactly one difference, so the
        honest output is the projection with the absence stated, not a band.
        """
        run, results = self._history("gemini", [0.033000, 0.032200])

        estimate = compute_estimate_cost(
            [run], results, models=["gemini"], k_runs=3, n_test_cases=2, profile=_JUDGED_HOST
        )

        cell = estimate.cells[0]
        assert cell.n_historical == 2
        assert cell.predicted.value == pytest.approx(0.1956)
        assert (cell.predicted.interval_low, cell.predicted.interval_high) == (None, None)
        assert (estimate.total_interval_low, estimate.total_interval_high) == (None, None)

    def test_a_near_identical_pair_never_bounds_out_a_cost_twelve_percent_away(self):
        """Whatever an n=2 basis publishes must not exclude a plausible outcome.

        Stated as the property rather than as "no band", so it keeps holding under any
        future rule that widens the band instead of withholding it. What it forbids is
        the specific harm: a basis whose two samples happen to agree pricing a sweep so
        tightly that the sweep's real cost falls outside what was published.
        """
        run, results = self._history("sonnet", [1.000, 1.025])

        estimate = compute_estimate_cost(
            [run], results, models=["sonnet"], k_runs=1, n_test_cases=6, profile=_JUDGED_HOST
        )

        cell = estimate.cells[0]
        twelve_percent_under = 0.88 * cell.predicted.value
        assert cell.predicted.interval_low is None or cell.predicted.interval_low <= twelve_percent_under, (
            f"a two-observation basis bounded the sweep out at {cell.predicted.interval_low}, "
            f"excluding a cost only 12% below its own point estimate"
        )

    def test_three_observations_is_the_boundary_at_which_a_band_appears(self):
        """The rule is the basis SIZE, not the spread — both bases below span the same range.

        Two observations yield one difference and no way to tell signal from accident;
        at df=1 Student's t is the Cauchy limit, so the multiplier would contribute as
        much of the width as the data. Three is the smallest sample whose dispersion
        rests on more than a single difference.
        """
        two_run, two_results = self._history("sonnet", [0.10, 0.30])
        three_run, three_results = self._history("sonnet", [0.10, 0.30, 0.20])

        two = compute_estimate_cost(
            [two_run], two_results, models=["sonnet"], k_runs=1, n_test_cases=1, profile=_JUDGED_HOST
        )
        three = compute_estimate_cost(
            [three_run], three_results, models=["sonnet"], k_runs=1, n_test_cases=1, profile=_JUDGED_HOST
        )

        assert two.cells[0].n_historical == 2
        assert (two.cells[0].predicted.interval_low, two.cells[0].predicted.interval_high) == (None, None)
        assert three.cells[0].n_historical == 3
        assert three.cells[0].predicted.interval_low is not None and three.cells[0].predicted.interval_high is not None
        assert (
            three.cells[0].predicted.interval_low
            < three.cells[0].predicted.value
            < three.cells[0].predicted.interval_high
        )

    def test_the_band_predicts_the_sweep_rather_than_locating_the_historical_mean(self):
        """A confidence interval on the mean is a narrower claim than the caller is making.

        ``1.96 * n_obs * SEM`` — the construction this replaces — says how precisely the
        history locates its own mean, and it keeps TIGHTENING as history accumulates.
        The caller is asking what the sweep will cost, which needs the variation the
        sweep's own observations will show as well — read on the log scale, since costs
        are positive and skewed (``stats.lognormal_sum_prediction_band``).
        """
        from threetears.evals.analysis.stats import lognormal_sum_prediction_band, standard_error_of_mean

        costs = [0.10, 0.20, 0.30, 0.40]
        run, results = self._history("sonnet", costs)
        n_obs = 5

        estimate = compute_estimate_cost(
            [run], results, models=["sonnet"], k_runs=5, n_test_cases=1, profile=_JUDGED_HOST
        )

        cell = estimate.cells[0]
        half_width = cell.predicted.interval_high - cell.predicted.value
        sem = standard_error_of_mean(costs)
        assert half_width > 1.96 * n_obs * sem, "the band is no wider than the confidence interval it replaced"

        expected = lognormal_sum_prediction_band(costs, n_obs)
        assert (cell.predicted.interval_low, cell.predicted.interval_high) == pytest.approx(expected)
        assert cell.band_basis is not None and "lognormal" in cell.band_basis

    def test_a_priced_cell_without_a_band_leaves_the_whole_total_unbracketed(self):
        """Summing a banded cell with an unbanded one narrows the envelope on the thinnest cell.

        The previous code contributed an unbanded cell's bare point to both total bounds,
        so the model that knew least about itself tightened the figure an operator budgets
        against. The point total still stands; only the bracket is withheld.
        """
        thin_run, thin_results = self._history("haiku", [0.10, 0.10])
        thick_run, thick_results = self._history("sonnet", [0.10, 0.20, 0.30])

        estimate = compute_estimate_cost(
            [thin_run, thick_run],
            thin_results + thick_results,
            models=["sonnet", "haiku"],
            k_runs=1,
            n_test_cases=1,
            profile=_JUDGED_HOST,
        )

        by_model = {cell.model: cell for cell in estimate.cells}
        assert by_model["sonnet"].predicted.interval_low is not None, "the well-founded cell keeps its own band"
        assert by_model["haiku"].predicted.interval_low is None
        assert estimate.total_estimated_cost == pytest.approx(0.30)
        assert (estimate.total_interval_low, estimate.total_interval_high) == (None, None)

    def test_an_empty_model_list_is_refused(self):
        with pytest.raises(CostEstimateError, match="no models proposed"):
            compute_estimate_cost([], [], models=[], k_runs=1, n_test_cases=1, profile=_JUDGED_HOST)

    def test_a_non_positive_grid_is_refused(self):
        run, results = self._history("sonnet", [0.10, 0.10])

        with pytest.raises(CostEstimateError, match=">= 1"):
            compute_estimate_cost([run], results, models=["sonnet"], k_runs=0, n_test_cases=1, profile=_JUDGED_HOST)


class TestPinnedRoleBadge:
    """A judge or simulator change confounds a candidate comparison."""

    def test_differing_judge_model_is_badged(self):
        run_a, _ = _run_with_results(judge_model="judge-1")
        run_b, _ = _run_with_results(judge_model="judge-2")

        assert (
            BADGE_ROLES_DIFFER
            in compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets[0].badges
        )

    def test_differing_simulator_model_is_badged(self):
        run_a, _ = _run_with_results(simulator_model="sim-1")
        run_b, _ = _run_with_results(simulator_model="sim-2")

        assert (
            BADGE_ROLES_DIFFER
            in compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets[0].badges
        )

    def test_matching_roles_are_not_badged(self):
        run_a, _ = _run_with_results(judge_model="j", simulator_model="s")
        run_b, _ = _run_with_results(judge_model="j", simulator_model="s")

        assert (
            BADGE_ROLES_DIFFER
            not in compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets[0].badges
        )

    def test_dims_scored_by_different_models_are_badged_despite_matching_pins(self):
        """The confound the pins cannot see: same judge_model, different actual scorers.

        Under the cascade a dim whose ``JudgeConfig`` names a model is scored by
        that, so two runs can agree on every pin and still have had their rubric
        scored differently — a difference in the apparatus wearing the appearance of
        a replicate. This is the comparison-surface twin of the identity assertion
        ``test_two_runs_judged_differently_per_dim_do_not_compare_equal`` makes.
        """
        run_a, _ = _run_with_results(
            judge_model="j",
            simulator_model="s",
            effective_judges={"__transcript__": "j", "__outcome__": "j"},
            effective_judges_source="recorded",
        )
        run_b, _ = _run_with_results(
            judge_model="j",
            simulator_model="s",
            effective_judges={"__transcript__": "j", "__outcome__": "haiku"},
            effective_judges_source="recorded",
        )

        assert (
            BADGE_ROLES_DIFFER
            in compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets[0].badges
        )

    def test_a_judge_ab_is_badged_despite_matching_pins_and_matching_models(self):
        """The same argument one step further, and the pair this surface most has to catch.

        These two runs agree on every pin AND on which model scored every dim — only
        the CONFIGURATION differs, which is to say the prompt the judge was given.
        That is what a judge A/B is, and without this arm the pair fires only
        ``context_differs``: "something about the conditions moved", which is exactly
        the opaque key mismatch the per-component badges exist to spare the operator.
        """
        shared = {"judge_model": "j", "simulator_model": "s", "effective_judges_source": "recorded"}
        run_a, _ = _run_with_results(
            effective_judges={"__transcript__": "j", "__outcome__": "j"},
            judge_config_ids={"reply.warmth": "cfg-baseline"},
            **shared,
        )
        run_b, _ = _run_with_results(
            effective_judges={"__transcript__": "j", "__outcome__": "j"},
            judge_config_ids={"reply.warmth": "cfg-candidate"},
            **shared,
        )

        assert run_a.effective_judges == run_b.effective_judges, "only the instrument moved"
        assert (
            BADGE_ROLES_DIFFER
            in compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets[0].badges
        )

    def test_an_unrecorded_config_set_is_not_badged_as_an_observed_difference(self):
        """Same rule the attribution arm follows: absence is not evidence.

        A run that recorded no set says nothing about its configuration, so grouping it
        with one that recorded a set must not manufacture a difference — that is
        ``context_incomplete``'s job, and it already fires.
        """
        shared = {"judge_model": "j", "simulator_model": "s"}
        recorded, _ = _run_with_results(judge_config_ids={"reply.warmth": "cfg-1"}, **shared)
        unrecorded, _ = _run_with_results(judge_config_ids=None, **shared)

        assert (
            BADGE_ROLES_DIFFER
            not in compute_comparison_sets([recorded, unrecorded], profile=_JUDGED_HOST).comparison_sets[0].badges
        )

    def test_two_runs_that_both_recorded_an_empty_config_set_agree(self):
        """``{}`` is a recording, so it participates in the comparison rather than opting out.

        Treating it as absent would drop both runs from the arm — harmless on its
        own, but it means a third run in the group carrying a real set could no
        longer be seen to differ from them.
        """
        shared = {"judge_model": "j", "simulator_model": "s"}
        a, _ = _run_with_results(judge_config_ids={}, **shared)
        b, _ = _run_with_results(judge_config_ids={}, **shared)
        c, _ = _run_with_results(judge_config_ids={"reply.warmth": "cfg-1"}, **shared)

        assert BADGE_ROLES_DIFFER not in compute_comparison_sets([a, b], profile=_JUDGED_HOST).comparison_sets[0].badges
        assert BADGE_ROLES_DIFFER in compute_comparison_sets([a, b, c], profile=_JUDGED_HOST).comparison_sets[0].badges

    def test_config_ordering_does_not_fake_a_role_difference(self):
        """The pinned set is a mapping; iteration order is not evidence of anything."""
        shared = {"judge_model": "j", "simulator_model": "s"}
        a, _ = _run_with_results(judge_config_ids={"reply.warmth": "cfg-1", "reply.tone": "cfg-2"}, **shared)
        b, _ = _run_with_results(judge_config_ids={"reply.tone": "cfg-2", "reply.warmth": "cfg-1"}, **shared)

        assert BADGE_ROLES_DIFFER not in compute_comparison_sets([a, b], profile=_JUDGED_HOST).comparison_sets[0].badges

    def test_dim_ordering_does_not_fake_a_role_difference(self):
        """Attribution is a mapping; iteration order is not evidence of anything."""
        run_a, _ = _run_with_results(
            judge_model="j",
            simulator_model="s",
            effective_judges={"__transcript__": "j", "__outcome__": "haiku"},
            effective_judges_source="recorded",
        )
        run_b, _ = _run_with_results(
            judge_model="j",
            simulator_model="s",
            effective_judges={"__outcome__": "haiku", "__transcript__": "j"},
            effective_judges_source="recorded",
        )

        assert (
            BADGE_ROLES_DIFFER
            not in compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets[0].badges
        )

    def test_an_absent_map_is_not_badged_as_an_observed_difference(self):
        """Not knowing what scored a run is not evidence that it scored differently.

        A run with no recorded attribution grouped with one whose judges were in fact
        identical fired this badge on the strength of the ABSENCE — the inference
        the badge separation exists to prevent, and the one the case-basis arm
        refuses twenty lines above. ``context_incomplete`` is what carries an
        undecidable case, and it already fires for every run in this state.
        """
        unattributed, _ = _run_with_results(
            judge_model="j",
            simulator_model="s",
            effective_judges=None,
            effective_judges_source=None,
        )
        recorded, _ = _run_with_results(
            judge_model="j",
            simulator_model="s",
            effective_judges={"__transcript__": "j", "__outcome__": "j"},
            effective_judges_source="recorded",
        )

        badges = compute_comparison_sets([unattributed, recorded], profile=_JUDGED_HOST).comparison_sets[0].badges
        assert BADGE_ROLES_DIFFER not in badges
        assert BADGE_CONTEXT_INCOMPLETE in badges, (
            "the undecidable case still has to be reported, just not as a difference"
        )

    def test_a_reconstruction_cannot_evidence_a_role_difference(self):
        """A derived map may be displayed, never used to assert comparability.

        ``derive_context_identity`` refuses to hash a reconstruction because it is
        an inference about what scored a run, not a record of it. A badge that
        compares the same map asserts exactly the comparability the key refuses.

        The discriminating case is a derived map that DISAGREES with a recorded one:
        source-blind comparison fires ``roles_differ`` there, reporting a measured
        difference between two runs on the strength of a guess about one of them.
        (The converse — a derived map that happens to agree — cannot discriminate,
        since excluding it also leaves a single digest and no badge either way.)
        """
        derived, _ = _run_with_results(
            judge_model="j",
            simulator_model="s",
            effective_judges={"__transcript__": "j", "__outcome__": "j"},
            effective_judges_source="derived",
        )
        recorded, _ = _run_with_results(
            judge_model="j",
            simulator_model="s",
            effective_judges={"__transcript__": "j", "__outcome__": "haiku"},
            effective_judges_source="recorded",
        )

        badges = compute_comparison_sets([derived, recorded], profile=_JUDGED_HOST).comparison_sets[0].badges
        assert BADGE_ROLES_DIFFER not in badges, "a reconstruction must not be compared as though it were recorded"
        assert BADGE_CONTEXT_INCOMPLETE in badges, "the derived run's context is partial and must say so"


class TestIncompleteContextBadge:
    """An unrecorded role pin makes every other badge come out clean."""

    def test_a_pair_that_never_recorded_its_judge_is_badged_incomplete(self):
        """The regression: two blanks compare EQUAL, so silence read as "nothing to caveat".

        Both runs inherited a judge before the pin was resolved, so both store None.
        The context keys match (each hashed the same blank), the role tuples match, the
        case sets match — every value-equality here comes out clean, and an empty badge
        list is this surface's way of saying the comparison needs no caveat. It needs
        one: nobody knows which judge scored either run.
        """
        run_a, _ = _run_with_results(judge_model=None, simulator_model=None)
        run_b, _ = _run_with_results(judge_model=None, simulator_model=None)

        badges = compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets[0].badges

        assert badges == [BADGE_CONTEXT_INCOMPLETE]

    def test_fully_recorded_runs_are_not_badged_incomplete(self):
        run_a, run_b = _stamped_run(), _stamped_run()

        assert (
            BADGE_CONTEXT_INCOMPLETE
            not in compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets[0].badges
        )

    def test_one_incomplete_run_badges_the_whole_group(self):
        """A group is only as reconstructable as its least-recorded member."""
        complete = _stamped_run()
        incomplete, _ = _run_with_results(judge_model=None, simulator_model=None)

        assert (
            BADGE_CONTEXT_INCOMPLETE
            in compute_comparison_sets([complete, incomplete], profile=_JUDGED_HOST).comparison_sets[0].badges
        )

    def test_the_grouping_compares_resolved_identity_not_the_stored_field(self):
        """One decision point for stamped-vs-derived, or the surfaces disagree.

        A run stamped under an older predicate keeps a stored key that predicate no
        longer produces. Reading the field directly compares it against keys from the
        current one — so two runs whose conditions are identical read as differing, on
        the strength of when each happened to be written.
        """
        current, stale = _stamped_run(), _stamped_run()
        stale.context_key = "f" * 64
        stale.identity_version = IDENTITY_VERSION - 1

        badges = compute_comparison_sets([current, stale], profile=_JUDGED_HOST).comparison_sets[0].badges

        assert BADGE_CONTEXT_DIFFERS not in badges


class TestToolConfigBadge:
    """The candidate's own config differs — usually the point, and previously unsaid."""

    @staticmethod
    def _run(**tool_configs):
        """A run whose subject carries ``tool_configs`` as its resolved tool-config component.

        The badge digests the subject's ``resolved_tool_configs`` component -- the configuration
        the candidate actually ran with, after any overlay was merged by the host -- so the
        subject is built carrying it directly.
        """
        resolved = SweepableValue.of(tool_configs, display="resolved tool configs")
        subject = make_subject("ent-maple", "Maple", components={"resolved_tool_configs": resolved})
        return make_eval_run(subject_snapshot=subject, judge_model="j", simulator_model="s")

    def test_a_swept_lever_is_badged(self):
        run_a = self._run(planner={"max_search_calls": 6, "token_budget": 12288})
        run_b = self._run(planner={"max_search_calls": 10, "token_budget": 20000})

        assert (
            BADGE_TOOL_CONFIG_DIFFERS
            in compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets[0].badges
        )

    def test_replicates_are_not_badged(self):
        run_a = self._run(planner={"max_search_calls": 6})
        run_b = self._run(planner={"max_search_calls": 6})

        assert (
            BADGE_TOOL_CONFIG_DIFFERS
            not in compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets[0].badges
        )

    def test_a_swept_set_is_otherwise_badge_silent(self):
        """The regression this badge exists for, stated as the absence it filled.

        Tool config composes the VARIANT key, not the context key — correctly, since
        it is the contestant rather than a condition. The read-side consequence went
        unstated: runs swept across tool levers match on context, roles and case
        set, so before this badge the set carried none at all and rendered exactly
        like a set of replicates, which is how a reader took them.
        """
        run_a = self._run(planner={"search_depth": "basic"})
        run_b = self._run(planner={"search_depth": "advanced"})

        badges = compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets[0].badges

        assert badges == [BADGE_TOOL_CONFIG_DIFFERS]

    def test_an_override_restating_the_subject_value_is_not_a_difference(self):
        """Resolved configs are compared, not the overlays that produced them.

        A run whose override sets a lever to the value the subject already carried
        faced an identical world to one that declared no override. Comparing the
        overlays would badge that as swept and send a reader looking for a
        difference the candidate never saw.
        """
        run_a = self._run(planner={"max_search_calls": 6})
        run_b = self._run(planner={"max_search_calls": 6})
        run_b.overlays = {"planner": {"max_search_calls": 6}}

        assert (
            BADGE_TOOL_CONFIG_DIFFERS
            not in compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets[0].badges
        )


class TestCassetteModeDisclosure:
    """The sentence itself — what it says, and the two spans it distinguishes."""

    def test_a_uniform_set_discloses_nothing(self):
        assert cassette_mode_disclosure({"a": "capture", "b": "capture"}) is None

    def test_two_replayed_arms_are_uniform_and_disclose_nothing(self):
        """The substitution is on both sides, so the delta between them is honest.

        Worth its own test because the instinct is that any `replay` deserves a
        warning. It does not: this surface's subject is the COMPARISON, and two
        arms that both re-served recordings differ by exactly what is under test.
        Disclosing here would train a reader to skip the line, which is the failure
        the judge-disparity lines collapse a uniform judge set to avoid.
        """
        assert cassette_mode_disclosure({"a": "replay", "b": "replay"}) is None

    def test_one_run_cannot_disagree_with_itself(self):
        assert cassette_mode_disclosure({"a": "replay"}) is None
        assert cassette_mode_disclosure({}) is None

    def test_a_replayed_span_names_the_substitution_and_the_quality_confound(self):
        """Cost is the half everyone expects; quality is the half that gets missed.

        A replayed delivery is bound by ORDINAL rather than by query, so a replayed
        arm can be served an answer to a question it did not ask — on a real smoke run
        it did exactly that on most deliveries.
        A sentence that mentioned only cost would leave a reader believing the
        pass^k delta beside it was clean.
        """
        sentence = cassette_mode_disclosure({"run-a": "capture", "run-b": "replay"})

        assert sentence is not None
        assert "run-a recorded capture" in sentence
        assert "run-b recorded replay" in sentence
        assert "QUALITY is confounded" in sentence
        assert "re-served a recording" in sentence

    def test_an_off_versus_capture_span_does_not_claim_a_substitution(self):
        """Both ran the third party live. Only `replay` substitutes.

        The badge fires on any span, because a span IS a difference in a condition
        that should have been holding still. What must not happen is the sentence
        crying substitution over a pair where nothing was substituted — that is how
        a caveat stops being read.
        """
        sentence = cassette_mode_disclosure({"run-a": "off", "run-b": "capture"})

        assert sentence is not None
        assert "run the third party live" in sentence
        assert "re-served a recording" not in sentence


class TestCassetteBadge:
    """The comparison surfaces stayed silent about the one condition that substitutes.

    A capture arm and a replay arm of one template could render pass^k 0.750
    vs 0.250, a paired t-test, and three caveats — none of
    which named that the second arm never called the third party.
    """

    @staticmethod
    def _run(mode):
        snapshot = make_subject("ent-maple", "Maple")
        return make_eval_run(subject_snapshot=snapshot, judge_model="j", simulator_model="s", cassette_mode=mode)

    def test_a_capture_replay_pair_is_badged(self):
        group = compute_comparison_sets(
            [self._run("capture"), self._run("replay")], profile=_JUDGED_HOST
        ).comparison_sets[0]

        assert BADGE_CASSETTE_MODE_DIFFERS in group.badges

    def test_replicates_are_not_badged(self):
        group = compute_comparison_sets([self._run("off"), self._run("off")], profile=_JUDGED_HOST).comparison_sets[0]

        assert BADGE_CASSETTE_MODE_DIFFERS not in group.badges

    def test_off_is_not_capture(self):
        """The third state. `off` and `capture` are both live, and both are recorded values."""
        group = compute_comparison_sets([self._run("off"), self._run("capture")], profile=_JUDGED_HOST).comparison_sets[
            0
        ]

        assert BADGE_CASSETTE_MODE_DIFFERS in group.badges

    def test_the_badge_and_its_sentence_cannot_come_apart(self):
        """One predicate, two outputs — the discipline the window badge already follows.

        A flag with no sentence behind it is the less useful half of the pair here,
        because the flag alone cannot tell an `off`/`capture` recording difference
        from a live-versus-replayed substitution.
        """
        badged = compute_comparison_sets(
            [self._run("capture"), self._run("replay")], profile=_JUDGED_HOST
        ).comparison_sets[0]
        clean = compute_comparison_sets(
            [self._run("replay"), self._run("replay")], profile=_JUDGED_HOST
        ).comparison_sets[0]

        assert (BADGE_CASSETTE_MODE_DIFFERS in badged.badges) is (badged.cassette_mode_disclosure is not None)
        assert (BADGE_CASSETTE_MODE_DIFFERS in clean.badges) is (clean.cassette_mode_disclosure is not None)
        assert badged.cassette_mode_disclosure is not None
        assert clean.cassette_mode_disclosure is None

    def test_the_comparison_is_disclosed_and_never_suppressed(self):
        """Capture-versus-replay is sometimes exactly what an operator meant to run.

        The requirement is disclosure, not refusal — so the group still forms, still
        reports both runs, and still reports their shared cases.
        """
        run_a, run_b = self._run("capture"), self._run("replay")

        group = compute_comparison_sets([run_a, run_b], profile=_JUDGED_HOST).comparison_sets[0]

        assert sorted(group.run_ids) == sorted([run_a.id, run_b.id])
        assert group.shared_test_case_ids


class TestDifferenceWasDeclaredAtLaunch:
    """The generic half — the rule that extracts, tested apart from its vocabulary."""

    def test_every_arm_declaring_is_a_declared_difference(self):
        assert difference_was_declared_at_launch(["chosen", "chosen"]) is True

    @pytest.mark.parametrize("origins", [["chosen", "inherited"], ["subject", "subject"], ["chosen", None]])
    def test_one_arm_that_inherited_makes_the_whole_difference_undeclared(self, origins):
        """Conservative by construction.

        A comparison cannot claim to be sweeping an axis one of its arms was never
        pointed at, so a single inheriting arm settles it for the set.
        """
        assert difference_was_declared_at_launch(origins) is False

    def test_nothing_declared_is_never_read_as_a_declaration(self):
        """An empty set is silence, and silence is not consent to call it a sweep."""
        assert difference_was_declared_at_launch([]) is False


class TestComparisonSetsScope:
    """Badging the set under analysis rather than everything that groups with it.

    `comparison_sets` groups by (subject, template) — correct and deliberate — and
    badged over that whole group. A campaign can hold a few runs on one template while
    the group holds more, spanning several days, and fire `cassette_mode_differs`
    because non-members recorded replay or capture while every member recorded
    `off`. The badge is then true of the group
    and false of the analysed set, and an operator cannot tell which without
    running `bisect_runs` pairwise. The grouping key is NOT what changes.
    """

    @staticmethod
    def _run(run_id, *, cassette_mode="off", test_case_ids=("tc-1",), subject="ent-maple"):
        snapshot = make_subject(subject, "Maple")
        return make_eval_run(
            id=run_id,
            subject_snapshot=snapshot,
            judge_model="j",
            simulator_model="s",
            cassette_mode=cassette_mode,
            test_case_ids=list(test_case_ids),
        )

    @staticmethod
    def _result(run_id, scored_at):
        return make_eval_result(eval_run_id=run_id, scored_at=scored_at)

    def _campaign_shaped_group(self):
        """Four campaign members that recorded `off`, beside two non-members that did not.

        A campaign's shape, reduced to what decides the badge.
        """
        members = [self._run(f"member-{i}") for i in range(4)]
        outsiders = [
            self._run("outsider-replay", cassette_mode="replay"),
            self._run("outsider-capture", cassette_mode="capture"),
        ]
        return members, outsiders

    def test_a_badge_earned_only_by_non_members_does_not_fire_for_the_scoped_set(self):
        """Acceptance: scoped to the campaign, `cassette_mode_differs` does not fire."""
        members, outsiders = self._campaign_shaped_group()
        runs = [*members, *outsiders]

        unscoped = compute_comparison_sets(runs, profile=_JUDGED_HOST).comparison_sets[0]
        scoped = compute_comparison_sets(
            runs, scope_run_ids=[run.id for run in members], profile=_JUDGED_HOST
        ).comparison_sets[0]

        assert BADGE_CASSETTE_MODE_DIFFERS in unscoped.badges, "fixture does not reproduce the reported badge"
        assert BADGE_CASSETTE_MODE_DIFFERS not in scoped.badges
        assert scoped.cassette_mode_disclosure is None
        assert scoped.run_ids == sorted(run.id for run in members)

    def test_a_badge_the_subset_earns_still_fires(self):
        """Acceptance: scoping narrows the population, it does not soften the badging.

        A scope that suppressed every caveat would be worse than the defect it
        fixes. The window badge is the one the issue names: the runner throttles at
        two concurrent, so campaign arms really do run apart.
        """
        members, outsiders = self._campaign_shaped_group()
        runs = [*members, *outsiders]
        results = [
            self._result("member-0", "2026-08-19T21:00:00+00:00"),
            self._result("member-0", "2026-08-19T21:10:00+00:00"),
            self._result("member-1", "2026-08-19T21:20:00+00:00"),
            self._result("member-1", "2026-08-19T21:30:00+00:00"),
            self._result("member-2", "2026-08-19T21:40:00+00:00"),
            self._result("member-3", "2026-08-19T21:50:00+00:00"),
        ]

        scoped = compute_comparison_sets(
            runs, results=results, scope_run_ids=[run.id for run in members], profile=_JUDGED_HOST
        ).comparison_sets[0]

        assert BADGE_MEASUREMENT_WINDOWS_DISJOINT in scoped.badges
        assert scoped.measurement_window_disclosure is not None
        # Named for the members and nobody else — a disclosure quoting a run the
        # scoped group does not list is the same defect one surface down.
        for outsider in outsiders:
            assert outsider.id not in scoped.measurement_window_disclosure

    def test_unscoped_behaviour_is_unchanged(self):
        """Acceptance: `scope_run_ids=None` is today's answer, field for field."""
        members, outsiders = self._campaign_shaped_group()
        runs = [*members, *outsiders]

        assert compute_comparison_sets(runs, profile=_JUDGED_HOST) == compute_comparison_sets(
            runs, scope_run_ids=None, profile=_JUDGED_HOST
        )

    def test_a_scope_narrows_the_case_basis_the_group_reports(self):
        """Every derived quantity narrows with the population, not just the badges.

        The intersection and the case-set fingerprints are computed from the same
        members the badges are, because a group whose badges speak for four runs
        and whose denominators speak for six is a worse answer than either.
        """
        members = [self._run("member-0"), self._run("member-1")]
        outsider = self._run("outsider", test_case_ids=("tc-1", "tc-2"))

        scoped = compute_comparison_sets(
            [*members, outsider], scope_run_ids=["member-0", "member-1"], profile=_JUDGED_HOST
        ).comparison_sets[0]

        assert BADGE_CASE_SET_DIFFERS not in scoped.badges
        assert len(scoped.case_sets) == 1
        assert scoped.case_sets[0].run_ids == ["member-0", "member-1"]
        assert scoped.shared_test_case_ids == ["tc-1"]

    def test_a_group_with_no_in_scope_member_is_not_emitted(self):
        """A group holding nothing the caller asked about has nothing to tell them."""
        maple = self._run("maple-run", subject="ent-maple")
        bea = self._run("bea-run", subject="ent-bea")

        scoped = compute_comparison_sets([maple, bea], scope_run_ids=["maple-run"], profile=_JUDGED_HOST)

        assert [group.subject_id for group in scoped.comparison_sets] == ["ent-maple"]
        assert scoped.comparison_sets[0].run_ids == ["maple-run"]

    def test_an_empty_scope_is_a_scope_and_not_an_absent_one(self):
        """`[]` asks about nothing; `None` asks about everything. They must not merge.

        The falsy-empty trap this module refuses everywhere else — an empty
        collection read as "unsupplied" would silently widen a caller's question
        to the whole scope.
        """
        members, outsiders = self._campaign_shaped_group()
        runs = [*members, *outsiders]

        assert compute_comparison_sets(runs, scope_run_ids=[], profile=_JUDGED_HOST).comparison_sets == []
        assert compute_comparison_sets(runs, scope_run_ids=None, profile=_JUDGED_HOST).comparison_sets != []

    def test_a_scope_naming_runs_that_are_not_here_says_so_rather_than_answering_empty(self, caplog):
        """A campaign whose members left the scope must not read as "not comparable".

        `campaign_get` reports exactly this state — "attached but absent from that
        scope" — so it is reachable, which is why it is logged rather than
        refused. What it may not be is silent: an empty answer and a true "nothing
        is comparable" render identically, the collapse this disclosure
        exists to prevent one axis over.
        """
        run = self._run("present")

        with caplog.at_level(logging.INFO, logger="threetears.evals.analysis.reporting"):
            result = compute_comparison_sets([run], scope_run_ids=["present", "gone-a", "gone-b"], profile=_JUDGED_HOST)

        assert result.comparison_sets[0].run_ids == ["present"]
        assert "scope named 2 run(s) not among the supplied runs: gone-a, gone-b" in caplog.text

    def test_a_scoped_call_names_only_the_runs_the_caller_asked_about(self):
        """A scope answers the question that was asked, and says what it left out.

        This used to assert the same property over a retired exclusion field — a count of
        subject-less runs, scoped to the caller's subset. That count is retired with the state it
        described, and the property it stood for survives on ``out_of_scope_run_ids``, which is
        where a scoped reader still learns what they are not being shown.
        """
        member = self._run("member-0")
        outsider = self._run("outsider")

        scoped = compute_comparison_sets([member, outsider], scope_run_ids=["member-0"], profile=_JUDGED_HOST)

        assert [group.run_ids for group in scoped.comparison_sets] == [["member-0"]]
        assert scoped.out_of_scope_run_ids == ["outsider"]
        assert compute_comparison_sets([member, outsider], profile=_JUDGED_HOST).out_of_scope_run_ids == []


# =============================================================================
# frontier — the verdict surface
# =============================================================================


def _fr_run(*, subject_id="ent-maple", subject_label="Maple", **overrides):
    """Build a run for a named subject, for frontier tests.

    Args:
        subject_id: Subject id for the run's subject snapshot.
        subject_label: Display name for the run's subject snapshot.
        **overrides: Passed through to ``make_eval_run``.

    Returns:
        The constructed ``EvalRun``.
    """
    snapshot = make_subject(subject_id, subject_label)
    return make_eval_run(subject_snapshot=snapshot, **overrides)


def _fr_result(
    run,
    *,
    model,
    variant_key,
    test_case_id,
    passes=True,
    roles=None,
    total_ms=None,
    candidate_error=None,
    infra_error=None,
    k_iteration=1,
    substituted_delivery=False,
    identity_version=None,
):
    """One result wired to ``run`` with pass, cost, and latency controlled.

    Args:
        run: The run this result belongs to.
        model: Candidate model slug.
        variant_key: Contestant identity, as the runner stamped it.
        test_case_id: Which test case this observation is for.
        passes: When true the goal passes and the rubric clears the threshold;
            when false both fail, so the case does not fully pass.
        roles: ``{role: cost_usd}`` for the result's usage rows, or ``None`` for no
            rows — capture ran and attributed no roles, so no cost was observed.
        total_ms: Harvested total latency, or ``None`` for no latency harvest.
        candidate_error: When set, the result is a candidate failure (composite
            0.0, never a pass), overriding ``passes``.
        infra_error: When set, the harness broke rather than the candidate, so the
            result classifies ``INFRA_EXCLUDE``. Distinct from ``candidate_error``
            on purpose — a candidate failure is a measurement and stays in, an
            apparatus fault is not one and comes out.
        k_iteration: Which k-iteration this is.
        substituted_delivery: When true the result carries a delivery a harness
            supplied, which withholds its production-replicating cost.
        identity_version: Which key-derivation predicate minted ``variant_key``.
            ``None`` stamps the current one, as the runner does.

    Returns:
        The constructed ``EvalResult``.
    """
    overrides = dict(
        eval_run_id=run.id,
        scope_id=run.scope_id,
        model=model,
        variant_key=variant_key,
        test_case_id=test_case_id,
        k_iteration=k_iteration,
    )
    if identity_version is not None:
        overrides["identity_version"] = identity_version
    if infra_error is not None:
        overrides["infra_error"] = infra_error
    if candidate_error is not None:
        # The run pinned a judge, as every judged run does: the judge never ran on a failed candidate, but its
        # criteria were asked, so the failure is a failed attempt rather than one with nothing to pass (#688).
        overrides.update(
            candidate_error=candidate_error, rubric_scores=[], goal_state_outcomes=[], judge_model="judge-model"
        )
    else:
        overrides.update(
            rubric_scores=[RubricScore(dim="reply.quality", score=4 if passes else 2, scale="ordinal")],
            goal_state_outcomes=[GoalStateOutcome(expression="ok", passed=passes)],
        )
    if roles is not None:
        overrides["usage"] = [RoleUsage(role=role, cost_usd=cost) for role, cost in roles.items()]
    if total_ms is not None:
        overrides["latency"] = LatencyMetrics(total_ms=total_ms)
    if substituted_delivery:
        # On the FLAT telemetry, not in a turn record's trace — because the frontier reads
        # results that carry no trace at all, so a trace fixture would hand it a field
        # production has already dropped. Both are stamped from the one delivery marker; only
        # this one survives the strip, and a trace-shaped fixture is how a guard that read the
        # trace passed its tests while publishing an understated cost for every real replay.
        overrides["async_deliveries"] = [AsyncDelivery(tool="scout", status="delivered", substituted=True)]
    return make_eval_result(**overrides)


def _point_by_model(subject_frontier, model):
    """The single point for ``model`` in a subject frontier."""
    matches = [p for p in subject_frontier.points if p.model == model]
    assert len(matches) == 1, f"expected one {model!r} point, got {len(matches)}"
    return matches[0]


def _fr_cases(run, n, *, spread=False, **kwargs):
    """``n`` results of one contestant, one per test case ``tc1``..``tc<n>``, each built by :func:`_fr_result`.

    The frontier decides domination and its bar by test over cases, so a fixture that is to show either
    needs the cases for a test to show it: two for a pass^k interval, more for a bar near 1. A gap of one
    amount on every case is read by the bounded test on pass^k's [0, 1] (eight pass-against-fail cases give
    p ≈ 0.012), and is never shown on cost or latency, which declare no range (#597). ``spread`` scales
    each case's cost and latency by ``1 + 0.02 · (index mod 3)``, so a gap between two contestants varies by
    case and a t-test reads it.
    """

    def scaled(index):
        if not spread:
            return kwargs
        factor = 1.0 + 0.02 * (index % 3)
        varied = dict(kwargs)
        if kwargs.get("roles"):
            varied["roles"] = {role: cost * factor for role, cost in kwargs["roles"].items()}
        if kwargs.get("total_ms") is not None:
            varied["total_ms"] = kwargs["total_ms"] * factor
        return varied

    return [_fr_result(run, test_case_id=f"tc{index}", **scaled(index)) for index in range(1, n + 1)]


class TestHistoryLatencyExcludesTheHarnesssOwnCells:
    """The same divergence as on the frontier, on the surface that issues regression verdicts.

    `history`'s composite already drops infra-excluded cells (``result_composite`` returns
    None for them), so leaving `total_ms` whole made the two metrics on ONE surface answer
    different questions about the same corpus — and a step is not merely displayed here, it
    is given a verdict against its neighbour. A cassette miss returns early with the
    wall-clock it had reached, so pooling it posts an improvement that describes the harness.
    """

    def test_an_infra_excluded_cell_is_not_a_latency_observation(self):
        run = _fr_run()
        healthy = _fr_result(run, model="m1", variant_key="vk-a", test_case_id="tc1", total_ms=1000.0)
        broken = _fr_result(
            run,
            model="m1",
            variant_key="vk-a",
            test_case_id="tc2",
            total_ms=10.0,
            infra_error="apparatus: cassette miss in replay mode",
        )

        result = compute_history(
            [run], [healthy, broken], metric=METRIC_TOTAL_MS, profile=_JUDGED_HOST, archived_run_ids=None
        )

        points = [point for series in result.series for point in series.points]
        assert len(points) == 1, "expected one run, one point"
        assert points[0].value == pytest.approx(1000.0), (
            f"the harness's truncated wall-clock entered a regression series (got {points[0].value})"
        )

    def test_a_candidate_failure_that_took_a_turn_is_still_a_latency_observation(self):
        """The guard must not widen into 'any error' — a slow candidate failure is real data.

        A failure that took a turn — here a model call the cell's deadline struck while it was pending — ran
        for as long as it ran. Only a call that came straight back refused or errored took no turn
        (``delivered_a_turn``), and that one is left out: its round trip is no turn's latency.
        """
        run = _fr_run()
        fast = _fr_result(run, model="m1", variant_key="vk-a", test_case_id="tc1", total_ms=100.0)
        slow_failure = _fr_result(
            run,
            model="m1",
            variant_key="vk-a",
            test_case_id="tc2",
            total_ms=900.0,
            candidate_error="the cell's deadline struck while the candidate's model was pending",
        ).model_copy(update={"termination": "cell_timeout"})
        refused = _fr_result(
            run,
            model="m1",
            variant_key="vk-a",
            test_case_id="tc3",
            total_ms=50.0,
            candidate_error="the provider refused the request",
        )

        result = compute_history(
            [run], [fast, slow_failure, refused], metric=METRIC_TOTAL_MS, profile=_JUDGED_HOST, archived_run_ids=None
        )

        points = [point for series in result.series for point in series.points]
        assert points[0].value == pytest.approx(500.0)


class TestHistoryCostKeepsEveryDollarSpent:
    """``cost_usd`` is measuring spend: a billed refusal and a faulted cell were both paid for, and both stay."""

    def test_a_billed_refusal_and_a_faulted_cell_are_both_kept(self):
        run = _fr_run()
        answered = _fr_result(run, model="m1", variant_key="vk-a", test_case_id="tc1").model_copy(
            update={"cost_usd": 0.004}
        )
        refused = _fr_result(
            run, model="m1", variant_key="vk-a", test_case_id="tc2", candidate_error="the provider refused the request"
        ).model_copy(update={"cost_usd": 0.0001})
        faulted = _fr_result(
            run, model="m1", variant_key="vk-a", test_case_id="tc3", infra_error="cassette miss"
        ).model_copy(update={"cost_usd": 0.002})

        result = compute_history(
            [run], [answered, refused, faulted], metric=METRIC_COST_USD, profile=_JUDGED_HOST, archived_run_ids=None
        )

        points = [point for series in result.series for point in series.points]
        assert points[0].value == pytest.approx((0.004 + 0.0001 + 0.002) / 3)

    def test_the_cost_pivot_and_the_run_summary_read_the_same_spend_as_the_history(self):
        """All three measure spend, so one corpus gives one figure."""
        run = _fr_run()
        answered = _fr_result(run, model="m1", variant_key="vk-a", test_case_id="tc1").model_copy(
            update={"cost_usd": 0.004}
        )
        refused = _fr_result(
            run, model="m1", variant_key="vk-a", test_case_id="tc2", candidate_error="the provider refused the request"
        ).model_copy(update={"cost_usd": 0.0001})
        faulted = _fr_result(
            run, model="m1", variant_key="vk-a", test_case_id="tc3", infra_error="cassette miss"
        ).model_copy(update={"cost_usd": 0.002})
        results = [answered, refused, faulted]

        (cell,) = compute_pivot(
            project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None).records,
            row_factor="model",
            column_factor="variant_key",
            metric=METRIC_COST_USD,
            profile=_JUDGED_HOST,
        ).cells
        (point,) = [
            point
            for series in compute_history(
                [run], results, metric=METRIC_COST_USD, profile=_JUDGED_HOST, archived_run_ids=None
            ).series
            for point in series.points
        ]
        from threetears.evals.kernel.scoring import compute_cost_summary

        (summary,) = compute_cost_summary(results).values()

        assert cell.value == pytest.approx(point.value) == pytest.approx(summary["mean_cost_usd"])
        assert (cell.n, point.n, summary["n_cost_usd"]) == (3, 3, 3)


class TestFrontierLatencyExcludesTheHarnesssOwnCells:
    """The frontier RANKS on latency, so an apparatus fault must not move a contestant.

    `compute_pass_hat_k` and `compute_dimension_summary` drop infra-excluded cells and
    `compute_cost_summary` deliberately keeps them, so latency being the third policy
    was a real divergence rather than a style choice: domination is decided over
    pass^k x production-replicating cost x total latency, and an infra-excluded cell carries a REAL but
    truncated `LatencyMetrics` — the cassette-miss and judge-error paths return early
    with the wall-clock they had reached. Pooling that in lets a harness failure make a
    contestant look fast.

    `compute_latency_summary` is not what the frontier reads — its only caller is
    `EvalService.run_summary` — so pinning it there left this seam untested.
    """

    def test_an_infra_excluded_cell_does_not_move_a_points_latency(self):
        """The mean must be the healthy cell's alone, not the average of the two."""
        run = _fr_run()
        healthy = _fr_result(run, model="m1", variant_key="vk-a", test_case_id="tc1", total_ms=1000.0)
        broken = _fr_result(
            run,
            model="m1",
            variant_key="vk-a",
            test_case_id="tc2",
            total_ms=10.0,
            infra_error="apparatus: cassette miss in replay mode",
        )

        out = compute_frontier([run], [healthy, broken], archived_run_ids=None)
        point = _point_by_model(out.subjects[0], "m1")

        assert point.mean_total_ms == pytest.approx(1000.0), (
            "the harness's truncated wall-clock was pooled into a ranked axis — a cassette miss "
            f"made this contestant look faster than it is (got {point.mean_total_ms})"
        )

    def test_a_candidate_failure_that_took_a_turn_still_counts_toward_latency(self):
        """The guard must not widen into 'any error', which would delete real measurements.

        A failure that took a turn — here a model call the cell's deadline struck while it was
        pending — is a measurement of that candidate, and it is the case pass^k counts as a failure
        rather than excluding, so latency keeps it. A call its model refused straight away took no
        turn (``delivered_a_turn``), and is left out: an all-refusing contestant otherwise ranked
        fastest and dominated the arms that answered.
        """
        run = _fr_run()
        fast = _fr_result(run, model="m1", variant_key="vk-a", test_case_id="tc1", total_ms=100.0)
        slow_failure = _fr_result(
            run,
            model="m1",
            variant_key="vk-a",
            test_case_id="tc2",
            total_ms=900.0,
            candidate_error="the cell's deadline struck while the candidate's model was pending",
        ).model_copy(update={"termination": "cell_timeout"})
        refused = _fr_result(
            run,
            model="m1",
            variant_key="vk-a",
            test_case_id="tc3",
            total_ms=50.0,
            candidate_error="the provider refused the request",
        )

        out = compute_frontier([run], [fast, slow_failure, refused], archived_run_ids=None)
        point = _point_by_model(out.subjects[0], "m1")

        assert point.mean_total_ms == pytest.approx(500.0), (
            "a failure that took a turn is a measurement and must stay in"
        )
        assert point.n_latency == 2

    def test_a_point_whose_every_cell_was_excluded_reports_no_latency(self):
        """Not zero — zero is the fastest possible contestant, and would win on that axis."""
        run = _fr_run()
        broken = _fr_result(
            run,
            model="m1",
            variant_key="vk-a",
            test_case_id="tc1",
            total_ms=10.0,
            infra_error="apparatus: cassette miss in replay mode",
        )

        out = compute_frontier([run], [broken], archived_run_ids=None)
        point = _point_by_model(out.subjects[0], "m1")

        assert point.mean_total_ms is None, "an all-excluded point must report latency unknown, never a winning zero"


class TestFrontierCostLeavesOutAFaultedCell:
    """#619: the frontier ranks on cost, so a cell an apparatus fault cut short must not make an arm cheaper."""

    def test_a_faulted_cell_does_not_lower_a_points_cost(self):
        run = _fr_run()
        whole = [
            _fr_result(run, model="m1", variant_key="vk-a", test_case_id=f"tc{i}", roles={"candidate": 0.10})
            for i in range(2)
        ]
        faulted = _fr_result(
            run,
            model="m1",
            variant_key="vk-a",
            test_case_id="tc2",
            roles={"candidate": 0.01},
            infra_error="apparatus: cassette miss in replay mode",
        )

        point = _point_by_model(compute_frontier([run], [*whole, faulted], archived_run_ids=None).subjects[0], "m1")

        assert point.production_replicating_cost == pytest.approx(0.10)
        assert point.n_cost == 2


class TestFrontierCostComposition:
    """The frontier RANKS on cost, so it must say when two contestants priced different things."""

    UNPRICED = ["candidate", "inner_agent", "judge", "simulator"]

    def test_a_point_carries_the_composition_its_cost_mean_covers(self):
        run = _fr_run()
        results = [_fr_result(run, model="m1", variant_key="vk-a", test_case_id="tc1", roles={"candidate": 0.1})]
        results[0].cost_roles = self.UNPRICED

        out = compute_frontier([run], results, archived_run_ids=None)

        assert out.subjects[0].points[0].cost_compositions == [self.UNPRICED]

    def test_two_contestants_pricing_different_things_are_each_labelled(self):
        """Neither number is wrong; the comparison between them is, and only this shows it."""
        run = _fr_run()
        cheap = _fr_result(run, model="m1", variant_key="vk-a", test_case_id="tc1", roles={"candidate": 0.10})
        dear = _fr_result(run, model="m2", variant_key="vk-b", test_case_id="tc1", roles={"candidate": 0.15})
        cheap.cost_roles = self.UNPRICED
        dear.cost_roles = [*self.UNPRICED, "external"]

        out = compute_frontier([run], [cheap, dear], archived_run_ids=None)

        by_model = {p.model: p for p in out.subjects[0].points}
        assert by_model["m1"].cost_compositions == [self.UNPRICED]
        assert by_model["m2"].cost_compositions == [[*self.UNPRICED, "external"]]


class TestFrontierSubjectPartition:
    """Two subjects are two frontiers, never one."""

    def test_a_two_subject_corpus_yields_two_frontiers(self):
        run_a = _fr_run(subject_id="ent-a", subject_label="A")
        run_b = _fr_run(subject_id="ent-b", subject_label="B")
        results = [
            _fr_result(run_a, model="m", variant_key="vk-a", test_case_id="tc1", roles={"candidate": 0.1}),
            _fr_result(run_b, model="m", variant_key="vk-b", test_case_id="tc1", roles={"candidate": 0.1}),
        ]

        result = compute_frontier([run_a, run_b], results, archived_run_ids=None)

        assert [pf.subject_id for pf in result.subjects] == ["ent-a", "ent-b"]
        assert all(len(pf.points) == 1 for pf in result.subjects)

    def test_same_variant_key_across_subjects_is_not_pooled(self):
        # Two subjects that happen to share a variant_key must not merge: the
        # subject is the top partition, so each keeps its own point.
        run_a = _fr_run(subject_id="ent-a", subject_label="A")
        run_b = _fr_run(subject_id="ent-b", subject_label="B")
        results = [
            _fr_result(run_a, model="m", variant_key="shared", test_case_id="tc1", roles={"candidate": 0.1}),
            _fr_result(run_b, model="m", variant_key="shared", test_case_id="tc1", roles={"candidate": 0.1}),
        ]

        result = compute_frontier([run_a, run_b], results, archived_run_ids=None)

        assert len(result.subjects) == 2
        assert {pf.subject_id for pf in result.subjects} == {"ent-a", "ent-b"}


class TestFrontierGroupsByVariant:
    """A point is one variant, not one model."""

    def test_two_variants_of_one_model_are_two_points(self):
        run = _fr_run()
        results = [
            _fr_result(run, model="sonnet", variant_key="vk-1", test_case_id="tc1", roles={"candidate": 0.1}),
            _fr_result(run, model="sonnet", variant_key="vk-2", test_case_id="tc1", roles={"candidate": 0.2}),
        ]

        result = compute_frontier([run], results, archived_run_ids=None)

        points = result.subjects[0].points
        assert len(points) == 2
        assert {p.variant_key for p in points} == {"vk-1", "vk-2"}
        assert all(p.model == "sonnet" for p in points)


class TestBothLensesGateOnTheIdentityVersion:
    """Two predicate versions of one key are never one contestant, on either lens.

    `frontier` and `history` share one contestant key, and until this gate existed it was
    `variant_key` alone. Two silent failures came out of that, and both are pinned here
    because a fix for either one alone reads as complete:

    - a **wrong merge**, when a bump leaves a particular key's inputs untouched so the
      digest matches and only the version moves — the two pooled as one contestant; and
    - an **unexplained split**, when a bump moves the inputs so one contestant becomes two
      rows bearing one model name and two digests, with nothing saying why.

    The gate partitions rather than relating, because two keys minted by different
    predicates cannot be shown to describe the same stack — that is what a hash change
    means. So the split is the answer, and the disclosures are what make it honest.
    """

    def test_one_key_under_two_predicates_is_two_frontier_points(self):
        run = _fr_run()
        results = [
            _fr_result(
                run,
                model="sonnet",
                variant_key="vk-1",
                test_case_id="tc1",
                roles={"candidate": 0.1},
                identity_version=9,
            ),
            _fr_result(
                run,
                model="sonnet",
                variant_key="vk-1",
                test_case_id="tc1",
                roles={"candidate": 0.2},
                identity_version=10,
            ),
        ]

        points = compute_frontier([run], results, archived_run_ids=None).subjects[0].points

        assert len(points) == 2, "two predicate versions of one key pooled as a single contestant"
        assert {p.variant_identity_version for p in points} == {9, 10}

    def test_one_key_under_one_predicate_stays_one_frontier_point(self):
        """The inverse direction, on one fixture: a gate that always splits also passes the test above."""
        run = _fr_run()
        results = [
            _fr_result(
                run,
                model="sonnet",
                variant_key="vk-1",
                test_case_id="tc1",
                roles={"candidate": 0.1},
                identity_version=10,
            ),
            _fr_result(
                run,
                model="sonnet",
                variant_key="vk-1",
                test_case_id="tc2",
                roles={"candidate": 0.2},
                identity_version=10,
            ),
        ]

        points = compute_frontier([run], results, archived_run_ids=None).subjects[0].points

        assert len(points) == 1, "one contestant was split on a version both its observations share"
        assert points[0].variant_identity_version == 10

    def test_one_key_under_two_predicates_is_two_history_series(self):
        run = _fr_run()
        results = [
            _fr_result(run, model="sonnet", variant_key="vk-1", test_case_id="tc1", total_ms=100.0, identity_version=9),
            _fr_result(
                run, model="sonnet", variant_key="vk-1", test_case_id="tc1", total_ms=200.0, identity_version=10
            ),
        ]

        result = compute_history([run], results, metric=METRIC_TOTAL_MS, profile=_JUDGED_HOST, archived_run_ids=None)

        assert len(result.series) == 2, "one series trended two predicate versions as one contestant"
        assert {series.variant_identity_version for series in result.series} == {9, 10}

    def test_one_key_under_one_predicate_stays_one_history_series(self):
        run = _fr_run()
        results = [
            _fr_result(
                run, model="sonnet", variant_key="vk-1", test_case_id="tc1", total_ms=100.0, identity_version=10
            ),
            _fr_result(
                run, model="sonnet", variant_key="vk-1", test_case_id="tc2", total_ms=200.0, identity_version=10
            ),
        ]

        result = compute_history([run], results, metric=METRIC_TOTAL_MS, profile=_JUDGED_HOST, archived_run_ids=None)

        assert len(result.series) == 1

    def test_a_dominator_is_identified_by_its_predicate_too(self):
        """Two points sharing a model AND a key are now reachable, so the label must separate them.

        `FrontierDominator` exists because a dominator list of bare model names let two
        variants of one model collapse into one entry, and the survivor read as a row
        dominating itself. Partitioning on the predicate re-opens that
        exact failure one level down: `<model> \u00b7 <key>` now identifies BOTH rows.
        Cross-version domination is deliberately still allowed — the gate says these are two
        contestants, and two contestants beating each other is what a frontier is for.
        """
        run = _fr_run()
        results = [
            # Same model, same key, two predicates. The current-predicate arm is shown better on
            # every axis its twin measured — it passes six cases its twin fails, at a fifth of the
            # cost — so it dominates its superseded twin.
            *_fr_cases(
                run,
                6,
                model="sonnet",
                variant_key="vk-1",
                passes=False,
                roles={"candidate": 0.5},
                identity_version=1,
                spread=True,
            ),
            *_fr_cases(
                run, 6, model="sonnet", variant_key="vk-1", roles={"candidate": 0.1}, identity_version=IDENTITY_VERSION
            ),
        ]

        points = compute_frontier([run], results, archived_run_ids=None).subjects[0].points
        dominated = [p for p in points if p.dominated]

        assert len(dominated) == 1, "the cheaper current-predicate arm should dominate its superseded twin"
        dominator = dominated[0].dominated_by[0]
        # Asserting the two MODEL fields differ would pass on a fixture using two different
        # keys and say nothing about the case that matters. These share both, so the
        # predicate is the only field that can separate them.
        assert dominator.model == dominated[0].model
        assert dominator.variant_key == dominated[0].variant_key
        assert dominator.variant_identity_version != dominated[0].variant_identity_version, (
            "the dominator is indistinguishable from the row it dominates without its predicate"
        )

    def test_two_points_sharing_a_model_and_key_have_a_meaningful_order(self):
        """The partition made a tie on (model, variant_key) reachable; stability is not an order."""
        run = _fr_run()
        results = [
            _fr_result(
                run,
                model="sonnet",
                variant_key="vk-1",
                test_case_id="tc1",
                roles={"candidate": 0.1},
                identity_version=10,
            ),
            _fr_result(
                run,
                model="sonnet",
                variant_key="vk-1",
                test_case_id="tc1",
                roles={"candidate": 0.2},
                identity_version=2,
            ),
        ]

        points = compute_frontier([run], results, archived_run_ids=None).subjects[0].points

        assert [p.variant_identity_version for p in points] == [2, 10], "adjacent twin rows must order by predicate"

    def test_two_series_sharing_a_model_and_key_have_a_meaningful_order(self):
        """The frontier's twin, on the lens where the tie is equally reachable."""
        run = _fr_run()
        results = [
            _fr_result(
                run, model="sonnet", variant_key="vk-1", test_case_id="tc1", total_ms=100.0, identity_version=10
            ),
            _fr_result(run, model="sonnet", variant_key="vk-1", test_case_id="tc1", total_ms=200.0, identity_version=2),
        ]

        series = compute_history(
            [run], results, metric=METRIC_TOTAL_MS, profile=_JUDGED_HOST, archived_run_ids=None
        ).series

        assert [s.variant_identity_version for s in series] == [2, 10]


class TestASupersededPredicateSaysSo:
    """A key stamped under a retired predicate is disclosed.

    Without the disclosure, a key minted by a predicate this build no longer uses renders as a
    confident, distinct contestant.
    """

    def test_a_superseded_key_is_disclosed_on_the_point(self):
        run = _fr_run()
        results = [
            _fr_result(
                run,
                model="sonnet",
                variant_key="vk-old",
                test_case_id="tc1",
                roles={"candidate": 0.1},
                identity_version=1,
            )
        ]

        point = compute_frontier([run], results, archived_run_ids=None).subjects[0].points[0]

        assert point.identity_version_disclosure is not None
        assert "v1" in point.identity_version_disclosure
        assert f"v{IDENTITY_VERSION}" in point.identity_version_disclosure

    def test_a_current_key_is_disclosed_as_nothing(self):
        run = _fr_run()
        results = [
            _fr_result(
                run,
                model="sonnet",
                variant_key="vk-now",
                test_case_id="tc1",
                roles={"candidate": 0.1},
                identity_version=IDENTITY_VERSION,
            )
        ]

        point = compute_frontier([run], results, archived_run_ids=None).subjects[0].points[0]

        assert point.identity_version_disclosure is None

    def test_the_verdict_carries_the_picks_disclosure(self):
        """A verdict is a recommendation, so the caveat travels with it, not only with the row."""
        run = _fr_run()
        # Eight cases, every one passed: enough for the pass^k interval to clear 0.5.
        results = _fr_cases(run, 8, model="sonnet", variant_key="vk-old", roles={"candidate": 0.1}, identity_version=1)

        subject = compute_frontier([run], results, bar=0.5, archived_run_ids=None).subjects[0]

        assert subject.verdict is not None
        assert subject.verdict.identity_version_disclosure == subject.points[0].identity_version_disclosure
        # And the VERSION beside it, so a reader can match the pick back to its row: the
        # lenses partition on the stamp, so `(model, variant_key)` no longer identifies one.
        # It is also the single signal every renderer decides "superseded?" from — the rows
        # and the verdict line reading two different fields is one question answered twice.
        assert subject.verdict.variant_identity_version == subject.points[0].variant_identity_version == 1

    def test_a_superseded_series_is_disclosed_too(self):
        run = _fr_run()
        results = [
            _fr_result(
                run, model="sonnet", variant_key="vk-old", test_case_id="tc1", total_ms=100.0, identity_version=1
            )
        ]

        series = compute_history(
            [run], results, metric=METRIC_TOTAL_MS, profile=_JUDGED_HOST, archived_run_ids=None
        ).series[0]

        assert series.identity_version_disclosure is not None


class TestAMixedPredicateCorpusExplainsItself:
    """The partition's other half: a doubled row needs a reason, and no per-row flag gives one.

    Splitting stops two predicate versions being RANKED together; it cannot stop them
    APPEARING as two rows. Without a corpus-level sentence the reader meets one model twice,
    under two digests, with nothing saying why — which is the same defect from the other end.
    """

    def test_a_mixed_corpus_names_the_predicates_it_spans(self):
        run = _fr_run()
        results = [
            _fr_result(
                run,
                model="sonnet",
                variant_key="vk-a",
                test_case_id="tc1",
                roles={"candidate": 0.1},
                identity_version=9,
            ),
            _fr_result(
                run,
                model="sonnet",
                variant_key="vk-b",
                test_case_id="tc1",
                roles={"candidate": 0.2},
                identity_version=10,
            ),
        ]

        result = compute_frontier([run], results, archived_run_ids=None)

        assert result.identity_version_span == [9, 10]
        assert result.identity_span_disclosure is not None
        assert "v9" in result.identity_span_disclosure and "v10" in result.identity_span_disclosure

    def test_a_single_predicate_corpus_says_nothing(self):
        run = _fr_run()
        results = [
            _fr_result(
                run,
                model="sonnet",
                variant_key="vk-a",
                test_case_id="tc1",
                roles={"candidate": 0.1},
                identity_version=10,
            )
        ]

        result = compute_frontier([run], results, archived_run_ids=None)

        assert result.identity_version_span == [10]
        assert result.identity_span_disclosure is None, "there is no split to explain"

    def test_history_reports_the_same_span_over_one_corpus(self):
        """Two lenses, one key, one span — derived from the same population so they cannot disagree."""
        run = _fr_run()
        results = [
            _fr_result(run, model="sonnet", variant_key="vk-a", test_case_id="tc1", total_ms=100.0, identity_version=9),
            _fr_result(
                run, model="sonnet", variant_key="vk-b", test_case_id="tc1", total_ms=200.0, identity_version=10
            ),
        ]

        assert compute_history(
            [run], results, metric=METRIC_TOTAL_MS, profile=_JUDGED_HOST, archived_run_ids=None
        ).identity_version_span == [
            9,
            10,
        ]
        assert compute_frontier([run], results, archived_run_ids=None).identity_version_span == [9, 10]

    def test_the_span_names_only_predicates_this_answer_rests_on(self):
        """A subject filter narrows the span with it, or the caveat points at rows the answer never used."""
        maple = _fr_run(subject_id="ent-maple", subject_label="Maple")
        other = _fr_run(subject_id="ent-other", subject_label="Other")
        results = [
            _fr_result(
                maple,
                model="sonnet",
                variant_key="vk-a",
                test_case_id="tc1",
                roles={"candidate": 0.1},
                identity_version=10,
            ),
            _fr_result(
                other,
                model="sonnet",
                variant_key="vk-b",
                test_case_id="tc1",
                roles={"candidate": 0.2},
                identity_version=9,
            ),
        ]

        result = compute_frontier([maple, other], results, subject_id="ent-maple", archived_run_ids=None)

        assert result.identity_version_span == [10], "the span named a predicate no ranked row was keyed under"
        assert result.identity_span_disclosure is None


class TestFrontierCassetteSpan:
    """A capture arm and a replay arm of one contestant are ONE row, and it never said so.

    A capture run (3/5) and a replay run (1/5) of one contestant rendered as a single variant reporting `0.40 (4/10)` with `latency n=10`.
    Strictly worse than the `compare_runs` case, where a reader can at least see two
    rows — here there is no visible seam at all, and `frontier` exists to RECOMMEND.

    Kept as one point deliberately: cassette mode is apparatus, not product, so it
    belongs to the context key and not the variant key (the variant key is "the
    resolved stack that would ship"). Splitting would re-answer that identity design
    as a side effect of a rendering fix and move every historical variant's identity.
    """

    def _corpus(self):
        capture = _fr_run(cassette_mode="capture")
        replay = _fr_run(cassette_mode="replay")
        results = [
            _fr_result(capture, model="flash", variant_key="vk-1", test_case_id="c1", roles={"candidate": 0.02}),
            _fr_result(replay, model="flash", variant_key="vk-1", test_case_id="c2", roles={"candidate": 0.02}),
        ]
        return [capture, replay], results

    def test_a_pooled_point_carries_the_disclosure(self):
        runs, results = self._corpus()

        point = compute_frontier(runs, results, archived_run_ids=None).subjects[0].points[0]

        assert point.n_results == 2, "the two arms must still pool — this fix discloses, it does not split"
        assert point.cassette_mode_disclosure is not None
        assert "re-served a recording" in point.cassette_mode_disclosure

    def test_a_uniform_contestant_is_not_marked(self):
        run_a, run_b = _fr_run(cassette_mode="off"), _fr_run(cassette_mode="off")
        results = [
            _fr_result(run_a, model="flash", variant_key="vk-1", test_case_id="c1", roles={"candidate": 0.02}),
            _fr_result(run_b, model="flash", variant_key="vk-1", test_case_id="c2", roles={"candidate": 0.02}),
        ]

        point = compute_frontier([run_a, run_b], results, archived_run_ids=None).subjects[0].points[0]

        assert point.cassette_mode_disclosure is None

    def test_a_point_is_only_qualified_by_modes_its_own_results_recorded(self):
        """A replay run elsewhere in the corpus must not taint an unrelated contestant.

        The map is collected corpus-wide and narrowed per point, so this pins the
        narrowing rather than trusting it.
        """
        capture = _fr_run(cassette_mode="capture")
        replay = _fr_run(cassette_mode="replay")
        results = [
            _fr_result(capture, model="flash", variant_key="vk-clean", test_case_id="c1", roles={"candidate": 0.02}),
            _fr_result(replay, model="flash", variant_key="vk-mixed", test_case_id="c1", roles={"candidate": 0.02}),
        ]

        points = {
            p.variant_key: p
            for p in compute_frontier([capture, replay], results, archived_run_ids=None).subjects[0].points
        }

        assert points["vk-clean"].cassette_mode_disclosure is None
        assert points["vk-mixed"].cassette_mode_disclosure is None

    def test_the_verdict_carries_it_too(self):
        """A verdict is a RECOMMENDATION, so it may not rest on the row's caveat.

        `cost_is_partial` already rides along for the weaker version of this reason;
        an operator who reads only the verdict line must not be handed a pick whose
        quality figure came half from a re-served recording.
        """
        runs, results = self._corpus()

        verdict = compute_frontier(runs, results, bar=0.0, archived_run_ids=None).subjects[0].verdict

        assert verdict is not None
        assert verdict.cassette_mode_disclosure is not None

    def test_a_clean_verdict_says_nothing(self):
        run = _fr_run(cassette_mode="off")
        # Two cases: a pass^k interval, which a bar is read by, needs two.
        results = _fr_cases(run, 2, model="flash", variant_key="vk-1", roles={"candidate": 0.02})

        verdict = compute_frontier([run], results, bar=0.0, archived_run_ids=None).subjects[0].verdict

        assert verdict is not None
        assert verdict.cassette_mode_disclosure is None


class TestFrontierTemplateSpan:
    """An easy suite and a hard one, measured on one variant, are ONE row — and it never said so.

    One subject's runs could span
    `single_turn_plain_v1` (tool-free, single-turn, 2 cases) and `multi_turn_tools_v1`
    (tool calls, 3 turns, goal-state checks, 2 cases), and the frontier named a
    verdict off `pass^k 1.00 (4/4)` — two cases from the easy suite pooled with two
    from the hard one, with no mention of a template anywhere in the output.

    Kept as one point deliberately, the same call the cassette span records: the
    template is a measurement CONDITION (it is in the context key, and
    `comparison_sets` groups on it) while `variant_key` is the stack that would ship.
    Splitting would re-answer that identity design as a side effect of a rendering
    fix. So the pooling stays and the row says what it pooled.
    """

    def _corpus(self):
        easy = _fr_run(template_id="single_turn_plain_v1")
        hard = _fr_run(template_id="multi_turn_tools_v1")
        results = [
            _fr_result(easy, model="flash", variant_key="vk-1", test_case_id="c1", roles={"candidate": 0.01}),
            _fr_result(hard, model="flash", variant_key="vk-1", test_case_id="c2", roles={"candidate": 0.01}),
        ]
        return [easy, hard], results

    def test_a_pooled_point_carries_the_disclosure(self):
        runs, results = self._corpus()

        point = compute_frontier(runs, results, archived_run_ids=None).subjects[0].points[0]

        assert point.n_results == 2, "the two suites must still pool — this fix discloses, it does not split"
        assert point.template_span_disclosure is not None
        assert "single_turn_plain_v1" in point.template_span_disclosure
        assert "multi_turn_tools_v1" in point.template_span_disclosure

    def test_the_suites_are_carried_structurally_beside_the_sentence(self):
        """The half that varies between contestants, so a surface can say the rule once.

        The sentence is identical on every marked point but its parenthetical, which is
        how a tool surface came to print one ninety-word caveat four times in a response. The list is
        what a surface showing several marked contestants prints per contestant.
        """
        runs, results = self._corpus()

        point = compute_frontier(runs, results, archived_run_ids=None).subjects[0].points[0]

        assert point.template_span == ["multi_turn_tools_v1 (1 run)", "single_turn_plain_v1 (1 run)"]
        assert point.template_span_disclosure is not None
        for entry in point.template_span:
            assert entry in point.template_span_disclosure, "the sentence is composed from the entries"

    def test_a_point_that_does_not_span_carries_neither(self):
        """One predicate: a surface keying on the list and one keying on the sentence agree."""
        run_a, run_b = _fr_run(template_id="tpl-1"), _fr_run(template_id="tpl-1")
        results = [
            _fr_result(run_a, model="flash", variant_key="vk-1", test_case_id="c1", roles={"candidate": 0.01}),
            _fr_result(run_b, model="flash", variant_key="vk-1", test_case_id="c2", roles={"candidate": 0.01}),
        ]

        point = compute_frontier([run_a, run_b], results, archived_run_ids=None).subjects[0].points[0]

        assert point.template_span == []
        assert point.template_span_disclosure is None

    def test_the_disclosure_qualifies_the_quality_axes_not_only_the_cost(self):
        """pass^k is what the bar is applied to, so naming only cost would miss the verdict."""
        runs, results = self._corpus()

        disclosure = (
            compute_frontier(runs, results, archived_run_ids=None).subjects[0].points[0].template_span_disclosure
        )

        assert disclosure is not None
        assert "pass^k" in disclosure
        assert "composite" in disclosure
        assert "latency" in disclosure

    def test_a_single_template_corpus_gains_no_noise_line(self):
        """Two runs of one suite are two repetitions of a condition, not a span."""
        run_a, run_b = _fr_run(template_id="tpl-1"), _fr_run(template_id="tpl-1")
        results = [
            _fr_result(run_a, model="flash", variant_key="vk-1", test_case_id="c1", roles={"candidate": 0.01}),
            _fr_result(run_b, model="flash", variant_key="vk-1", test_case_id="c2", roles={"candidate": 0.01}),
        ]

        point = compute_frontier([run_a, run_b], results, archived_run_ids=None).subjects[0].points[0]

        assert point.template_span_disclosure is None

    def test_an_ad_hoc_run_is_its_own_suite_rather_than_a_blank(self):
        """`template_id=None` is a real recorded state — an ad-hoc run from explicit case ids.

        Reading it as absence would let a templated run and an ad-hoc one pool
        undisclosed, which is the same mixed-difficulty pool this fix exists to name.
        """
        templated = _fr_run(template_id="tpl-1")
        ad_hoc = _fr_run(template_id=None)
        results = [
            _fr_result(templated, model="flash", variant_key="vk-1", test_case_id="c1", roles={"candidate": 0.01}),
            _fr_result(ad_hoc, model="flash", variant_key="vk-1", test_case_id="c2", roles={"candidate": 0.01}),
        ]

        disclosure = (
            compute_frontier([templated, ad_hoc], results, archived_run_ids=None)
            .subjects[0]
            .points[0]
            .template_span_disclosure
        )

        assert disclosure is not None
        assert "ad-hoc (no template)" in disclosure

    def test_a_point_is_only_qualified_by_templates_its_own_results_ran(self):
        """A second template elsewhere in the corpus must not taint an unrelated contestant.

        The map is collected corpus-wide and narrowed per point, so this pins the
        narrowing rather than trusting it.
        """
        easy = _fr_run(template_id="single_turn_plain_v1")
        hard = _fr_run(template_id="multi_turn_tools_v1")
        results = [
            _fr_result(easy, model="flash", variant_key="vk-clean", test_case_id="c1", roles={"candidate": 0.01}),
            _fr_result(hard, model="flash", variant_key="vk-other", test_case_id="c1", roles={"candidate": 0.01}),
        ]

        points = {
            p.variant_key: p for p in compute_frontier([easy, hard], results, archived_run_ids=None).subjects[0].points
        }

        assert points["vk-clean"].template_span_disclosure is None
        assert points["vk-other"].template_span_disclosure is None

    def test_the_verdict_carries_it_too(self):
        """The bar was applied to the pooled pass^k, so the recommendation owes the caveat.

        An operator who reads only the verdict line must not be handed a pick whose
        quality figure is an average over suites of different difficulty.
        """
        easy = _fr_run(template_id="single_turn_plain_v1")
        hard = _fr_run(template_id="multi_turn_tools_v1")
        # Four cases from each suite, all passed: enough for the pooled pass^k's interval to clear 0.5.
        results = [
            _fr_result(
                run, model="flash", variant_key="vk-1", test_case_id=f"{suite}{index}", roles={"candidate": 0.01}
            )
            for run, suite in ((easy, "easy"), (hard, "hard"))
            for index in range(4)
        ]

        verdict = compute_frontier([easy, hard], results, bar=0.5, archived_run_ids=None).subjects[0].verdict

        assert verdict is not None
        assert verdict.template_span_disclosure is not None
        assert "multi_turn_tools_v1" in verdict.template_span_disclosure

    def test_the_verdict_still_names_the_cheapest_above_bar_variant(self):
        """Disclosure only — the ranking itself is unchanged by this fix."""
        easy = _fr_run(template_id="single_turn_plain_v1")
        hard = _fr_run(template_id="multi_turn_tools_v1")
        # Eight passed cases from each suite: sixteen, enough for an all-pass interval to clear 0.7.
        results = [
            _fr_result(
                run, model=model, variant_key=f"vk-{model}", test_case_id=f"{suite}{index}", roles={"candidate": cost}
            )
            for model, cost in (("cheap", 0.01), ("dear", 0.50))
            for run, suite in ((easy, "easy"), (hard, "hard"))
            for index in range(8)
        ]

        verdict = compute_frontier([easy, hard], results, bar=0.7, archived_run_ids=None).subjects[0].verdict

        assert verdict is not None
        assert verdict.model == "cheap"
        assert verdict.pass_hat_k == 1.0

    def test_a_clean_verdict_says_nothing(self):
        run = _fr_run(template_id="tpl-1")
        results = _fr_cases(run, 2, model="flash", variant_key="vk-1", roles={"candidate": 0.01})

        verdict = compute_frontier([run], results, bar=0.0, archived_run_ids=None).subjects[0].verdict

        assert verdict is not None
        assert verdict.template_span_disclosure is None


class TestFrontierDomination:
    """A domination is shown by test before it is flagged; dominated points are flagged, not dropped."""

    def _corpus(self):
        """Three contestants over eight cases — enough for a constant gap to separate after Holm over three pairs.

        - cheap: passes every case, about $0.25, 50 ms (each varying a little by case).
        - pricey: fails every case, about $1.00, 100 ms — shown worse than cheap on all three axes.
        - fast: passes half, $0.10, 20 ms — cheaper and faster than cheap, worse on pass^k: no domination
          either way. Better than pricey on every axis too, but its pass^k edge (half the cases) is not
          shown once the family is adjusted.
        """
        run = _fr_run()
        results = [
            *_fr_cases(
                run, 8, model="cheap", variant_key="vk-cheap", roles={"candidate": 0.25}, total_ms=50, spread=True
            ),
            *_fr_cases(
                run,
                8,
                model="pricey",
                variant_key="vk-pricey",
                passes=False,
                roles={"candidate": 1.0},
                total_ms=100,
                spread=True,
            ),
            *(
                _fr_result(
                    run,
                    model="fast",
                    variant_key="vk-fast",
                    test_case_id=f"tc{index}",
                    roles={"candidate": 0.1},
                    total_ms=20,
                    passes=index <= 4,
                )
                for index in range(1, 9)
            ),
        ]
        return run, results

    def test_dominated_point_is_flagged_with_its_dominator(self):
        run, results = self._corpus()

        pf = compute_frontier([run], results, archived_run_ids=None).subjects[0]

        pricey = _point_by_model(pf, "pricey")
        assert pricey.dominated is True and pricey.dominance == "dominated"
        # The row's own identity, not its model: a dominator names the variant a reader
        # would move TO, which is what the frontier exists to answer.
        assert [(d.model, d.variant_key) for d in pricey.dominated_by] == [("cheap", "vk-cheap")]
        # And the statistic it was shown at: the weakest axis, pass^k's eight pass-against-fail cases read by the
        # bounded test on [0, 1] (#597), times three pairs.
        pass_p = bounded_separation_p([0.0] * 8, [1.0] * 8, paired=True, value_range=(0.0, 1.0))
        assert pass_p is not None
        assert pricey.dominated_by[0].p_value == pytest.approx(3 * pass_p)

    def test_non_dominated_points_are_not_flagged(self):
        run, results = self._corpus()

        pf = compute_frontier([run], results, archived_run_ids=None).subjects[0]

        assert _point_by_model(pf, "cheap").dominated is False
        assert _point_by_model(pf, "fast").dominated is False
        # Not dominated is not "shown on the frontier": each was tested, and nothing was shown.
        assert _point_by_model(pf, "cheap").dominance == "not_separated"
        assert _point_by_model(pf, "fast").dominance == "not_separated"

    def test_a_cost_gap_with_a_quality_tie_is_not_a_domination(self):
        """Dearer on every case and equal on pass^k is not shown worse on pass^k, so it is not dominated.

        "No worse" on an axis would need a margin to be shown, and pass^k declares none: two contestants
        that each passed every case are tied there, and a tie cannot be told from a small difference
        either way. The cost gap alone is a separation, not a domination.
        """
        run = _fr_run()
        results = [
            *_fr_cases(run, 8, model="cheap", variant_key="vk-cheap", roles={"candidate": 0.25}, total_ms=50),
            *_fr_cases(run, 8, model="pricey", variant_key="vk-pricey", roles={"candidate": 1.0}, total_ms=50),
        ]

        pf = compute_frontier([run], results, archived_run_ids=None).subjects[0]

        assert _point_by_model(pf, "pricey").dominated is False
        assert _point_by_model(pf, "pricey").dominance == "not_separated"

    def test_a_gap_of_one_amount_on_cost_and_latency_is_never_a_domination(self):
        """Three cases, every axis moved by one amount. Cost and latency declare no range, so no test of the mean
        can show either gap (the sign-flip p this read tests symmetry, not the mean — #597), and pass^k's bounded
        test cannot show three cases. Not separated, as the rule reads a move it refuses to call."""
        run = _fr_run()
        results = [
            *_fr_cases(run, 3, model="cheap", variant_key="vk-cheap", roles={"candidate": 0.25}, total_ms=50),
            *_fr_cases(
                run, 3, model="pricey", variant_key="vk-pricey", passes=False, roles={"candidate": 1.0}, total_ms=100
            ),
        ]

        pf = compute_frontier([run], results, archived_run_ids=None).subjects[0]

        assert _point_by_model(pf, "pricey").dominance == "not_separated"
        assert _point_by_model(pf, "cheap").dominance == "not_separated"

    def test_a_point_with_one_case_is_untested(self):
        """One case gives no test, so the point is neither dominated nor shown clear of it."""
        run = _fr_run()
        results = [
            *_fr_cases(run, 8, model="cheap", variant_key="vk-cheap", roles={"candidate": 0.25}, total_ms=50),
            _fr_result(
                run, model="pricey", variant_key="vk-pricey", test_case_id="tc1", passes=False, roles={"candidate": 1.0}
            ),
        ]

        pf = compute_frontier([run], results, archived_run_ids=None).subjects[0]

        assert _point_by_model(pf, "pricey").dominance == "untested"
        assert _point_by_model(pf, "pricey").dominated is False

    def test_dominated_point_is_retained_not_dropped(self):
        run, results = self._corpus()

        pf = compute_frontier([run], results, archived_run_ids=None).subjects[0]

        assert {p.model for p in pf.points} == {"cheap", "pricey", "fast"}

    def test_two_variants_of_one_model_are_both_named_and_neither_is_the_row_itself(self):
        """A row could read as dominating itself.

        `vendor/flash-x · <variant>` rendered as "dominated by
        vendor/flash-x" — its own model — while TWO distinct variants of that
        model beat it. Nothing was wrong with the domination arithmetic: the list was a set
        over ``model`` alone, and a set over a less-identifying key merges the entries and
        leaves behind a string that names the dominated row.
        """
        run = _fr_run()
        results = [
            # Cheap and slow-ish.
            *_fr_cases(
                run, 8, model="flash-x", variant_key="vk-a", roles={"candidate": 0.0021}, total_ms=92, spread=True
            ),
            # Dear and fast — neither dominates the other, and both are shown better than `vk-c` on every axis.
            *_fr_cases(
                run, 8, model="flash-x", variant_key="vk-b", roles={"candidate": 0.024}, total_ms=48, spread=True
            ),
            *(
                # Varied the other way round, so its gap to each of the two above varies by case.
                _fr_result(
                    run,
                    model="flash-x",
                    variant_key="vk-c",
                    test_case_id=f"tc{index}",
                    passes=False,
                    roles={"candidate": 0.030 * (1.0 + 0.03 * (index % 2))},
                    total_ms=100 * (1.0 + 0.03 * (index % 2)),
                )
                for index in range(1, 9)
            ),
        ]

        pf = compute_frontier([run], results, archived_run_ids=None).subjects[0]
        dominated = next(p for p in pf.points if p.variant_key == "vk-c")

        assert [(d.model, d.variant_key) for d in dominated.dominated_by] == [
            ("flash-x", "vk-a"),
            ("flash-x", "vk-b"),
        ], "both dominating variants must survive — a set over `model` alone collapses them into one"
        assert all(
            (d.model, d.variant_key) != (dominated.model, dominated.variant_key) for d in dominated.dominated_by
        ), "no row may name itself as its own dominator"

    def test_a_point_missing_latency_does_not_dominate_on_speed(self):
        # A variant with no harvested latency cannot claim to be the fastest — its
        # unknown latency must never read as zero and win the speed axis.
        run = _fr_run()
        results = [
            _fr_result(
                run, model="measured", variant_key="vk-m", test_case_id="c1", roles={"candidate": 0.5}, total_ms=10
            ),
            # same quality and cost, but NO latency — must not dominate `measured`
            _fr_result(
                run, model="unmeasured", variant_key="vk-u", test_case_id="c1", roles={"candidate": 0.5}, total_ms=None
            ),
        ]

        pf = compute_frontier([run], results, archived_run_ids=None).subjects[0]

        assert _point_by_model(pf, "measured").dominated is False
        assert _point_by_model(pf, "unmeasured").dominated is False


class TestFrontierCostIsProductionReplicating:
    """Cost sums the production-replicating roles only, never judge/simulator."""

    def test_judge_and_simulator_cost_are_excluded(self):
        run = _fr_run()
        result = _fr_result(
            run,
            model="m",
            variant_key="vk",
            test_case_id="c1",
            roles={"candidate": 0.30, "inner_agent": 0.10, "judge": 5.0, "simulator": 2.0},
        )

        point = compute_frontier([run], [result], archived_run_ids=None).subjects[0].points[0]

        # 0.30 (candidate) + 0.10 (inner_agent); judge 5.0 and simulator 2.0 excluded.
        assert point.production_replicating_cost == pytest.approx(0.40)

    def test_cost_per_acceptable_outcome_divides_cost_by_pass_rate(self):
        run = _fr_run()
        results = [
            _fr_result(run, model="m", variant_key="vk", test_case_id="c1", roles={"candidate": 0.5}, passes=True),
            _fr_result(run, model="m", variant_key="vk", test_case_id="c2", roles={"candidate": 0.5}, passes=False),
        ]

        point = compute_frontier([run], results, archived_run_ids=None).subjects[0].points[0]

        # cost 0.5, pass^1 0.5 → 0.5 / 0.5 = 1.0
        assert point.pass_hat_k == pytest.approx(0.5)
        assert point.cost_per_acceptable_outcome == pytest.approx(1.0)

    def test_cost_per_acceptable_outcome_is_none_when_nothing_passes(self):
        run = _fr_run()
        result = _fr_result(run, model="m", variant_key="vk", test_case_id="c1", roles={"candidate": 0.5}, passes=False)

        point = compute_frontier([run], [result], archived_run_ids=None).subjects[0].points[0]

        assert point.pass_hat_k == 0.0
        assert point.cost_per_acceptable_outcome is None


class TestFrontierPartialCost:
    """A result that observed no production cost reports none; the point's cost is meaned over those that did."""

    def test_partial_cost_is_flagged(self):
        run = _fr_run()
        results = [
            _fr_result(run, model="m", variant_key="vk", test_case_id="c1", roles={"candidate": 0.20}),
            # no per-role rows → contributes no cost observation
            _fr_result(run, model="m", variant_key="vk", test_case_id="c2", roles=None),
        ]

        point = compute_frontier([run], results, archived_run_ids=None).subjects[0].points[0]

        assert point.production_replicating_cost == pytest.approx(0.20)
        assert point.n_cost == 1
        assert point.n_results == 2
        assert point.cost_is_partial is True

    def test_no_observed_cost_is_none_not_zero(self):
        run = _fr_run()
        results = [
            _fr_result(run, model="m", variant_key="vk", test_case_id="c1", roles=None),
            _fr_result(run, model="m", variant_key="vk", test_case_id="c2", roles=None),
        ]

        point = compute_frontier([run], results, archived_run_ids=None).subjects[0].points[0]

        assert point.production_replicating_cost is None
        assert point.n_cost == 0
        # No observed cost is unknown, not "partial" — partial means some did report one.
        assert point.cost_is_partial is False


class TestFrontierAllFailedVariantRanksAtZero:
    """A variant whose every result failed scores 0.0, not null, and is dominated."""

    def test_a_contestant_with_no_scored_case_has_no_pass_hat_k_and_clears_no_bar(self):
        """Every iteration excluded is nothing measured: pass^k is absent, not 0.0.

        Beside it a candidate failure that WAS scored keeps its real 0.0, so the two states a
        0.0 used to merge stay apart; the unmeasured one neither dominates nor is dominated on
        quality, and a bar of 0.0 — which any measured pass^k clears — does not admit it.
        """
        run = _fr_run()
        results = [
            _fr_result(
                run,
                model="rig",
                variant_key="vk-rig",
                test_case_id="c1",
                infra_error="sandbox died",
                roles={"candidate": 0.1},
            ),
            # Two cases, so its pass^k has the interval a bar is read by.
            *_fr_cases(run, 2, model="fails", variant_key="vk-fails", passes=False, roles={"candidate": 0.2}),
        ]

        result = compute_frontier([run], results, bar=0.0, archived_run_ids=None)
        pf = result.subjects[0]

        assert _point_by_model(pf, "rig").pass_hat_k is None
        assert _point_by_model(pf, "rig").cost_per_acceptable_outcome is None
        assert _point_by_model(pf, "fails").pass_hat_k == 0.0
        assert _point_by_model(pf, "rig").bar_decision == "no_data"
        assert pf.verdict is not None and pf.verdict.model == "fails"

    def test_candidate_failure_scores_zero_quality_and_ranks(self):
        run = _fr_run()
        results = [
            # a working variant, over six cases — shown better on every axis the failure measured
            *_fr_cases(run, 6, model="works", variant_key="vk-works", roles={"candidate": 0.10}, total_ms=30),
            # a variant that failed everywhere — candidate error, dearer, no latency harvest
            *_fr_cases(
                run, 6, model="broken", variant_key="vk-broken", candidate_error="boom", roles={"candidate": 0.50}
            ),
        ]

        pf = compute_frontier([run], results, archived_run_ids=None).subjects[0]
        broken = _point_by_model(pf, "broken")

        assert broken.pass_hat_k == 0.0
        assert broken.mean_composite == 0.0  # candidate failure is a real 0.0, not None
        assert broken.mean_total_ms is None
        assert broken.dominated is True
        assert [(d.model, d.variant_key) for d in broken.dominated_by] == [("works", "vk-works")]


class TestFrontierBar:
    """The bar gates pass^k, is echoed back, and picks the cheapest above it."""

    def _corpus(self):
        # Twenty cases each: an all-pass interval clears 0.8 from twenty (at twelve it reaches only 0.70).
        run = _fr_run()
        results = [
            *_fr_cases(run, 20, model="a", variant_key="vk-a", roles={"candidate": 0.50}, passes=True, spread=True),
            *_fr_cases(run, 20, model="b", variant_key="vk-b", roles={"candidate": 0.30}, passes=True),
            *_fr_cases(run, 20, model="c", variant_key="vk-c", roles={"candidate": 0.10}, passes=False),
        ]
        return run, results

    def test_no_bar_yields_no_verdict_and_no_invented_default(self):
        run, results = self._corpus()

        result = compute_frontier([run], results, bar=None, archived_run_ids=None)

        assert result.bar is None
        assert all(pf.verdict is None for pf in result.subjects)

    def test_bar_is_echoed_back(self):
        run, results = self._corpus()

        assert compute_frontier([run], results, bar=0.8, archived_run_ids=None).bar == pytest.approx(0.8)

    def test_verdict_is_the_cheapest_variant_above_bar(self):
        run, results = self._corpus()

        pf = compute_frontier([run], results, bar=0.8, archived_run_ids=None).subjects[0]

        # a and b both pass, their intervals wholly above 0.8; c fails, its interval wholly below. Cheapest
        # of {a: 0.5, b: 0.3} is b.
        assert pf.n_cleared_bar == 2 and pf.n_undecided_bar == 0
        assert [p.bar_decision for p in pf.points] == ["cleared", "cleared", "missed"]
        assert pf.verdict is not None
        assert pf.verdict.model == "b"
        assert pf.verdict.production_replicating_cost == pytest.approx(0.30)
        assert pf.verdict.pass_hat_k_ci_low is not None and pf.verdict.pass_hat_k_ci_low >= 0.8
        # b is cheaper on every one of the twenty cases, by an amount that varies: shown cheaper, so named.
        assert (pf.verdict.cost_decision, pf.verdict.tied_with) == ("shown_cheapest", [])

    def test_a_pick_not_shown_cheaper_names_the_set_it_is_among(self):
        """A lower point cost is not a cheaper contestant: b costs $0.001 more on average, untestably."""
        run = _fr_run()
        results = [
            *(
                _fr_result(run, model="a", variant_key="vk-a", test_case_id=f"tc{i}", roles={"candidate": cost})
                for i, cost in enumerate([0.39, *[0.40] * 19])
            ),
            *(
                _fr_result(run, model="b", variant_key="vk-b", test_case_id=f"tc{i}", roles={"candidate": cost})
                for i, cost in enumerate([0.40, 0.41, *[0.40] * 18])
            ),
        ]

        verdict = compute_frontier([run], results, bar=0.8, archived_run_ids=None).subjects[0].verdict

        assert verdict is not None and verdict.model == "a"
        assert verdict.cost_decision == "not_separated"
        (tie,) = verdict.tied_with
        assert (tie.model, tie.variant_key) == ("b", "vk-b")
        assert tie.p_value is not None and tie.p_value >= 0.05
        assert tie.production_replicating_cost == pytest.approx(0.4005)

    def test_a_constant_cost_shift_is_never_shown_cheaper(self):
        """Five cases, each $0.50 dearer for b: cost declares no range, so no test of the mean can show it (#597).

        The costs are written ``i/10`` and ``i/10 + 0.5``, so their float differences carry residue. Read over
        floats, a t-test took that residue for a tiny, perfectly consistent spread and named a the cheapest at
        p ≈ 1e-80. Read exactly there is no spread, and with no range the rule refuses the claim: not shown
        cheaper, with p 1 — never a residue's p, and never the sign flip's.
        """
        run = _fr_run()
        a_costs = [i / 10 for i in range(1, 6)]
        b_costs = [i / 10 + 0.5 for i in range(1, 6)]
        assert len({b - a for a, b in zip(a_costs, b_costs)}) > 1, "the fixture must carry float residue"
        results = [
            *(
                _fr_result(run, model="a", variant_key="vk-a", test_case_id=f"tc{i}", roles={"candidate": cost})
                for i, cost in enumerate(a_costs)
            ),
            *(
                _fr_result(run, model="b", variant_key="vk-b", test_case_id=f"tc{i}", roles={"candidate": cost})
                for i, cost in enumerate(b_costs)
            ),
        ]

        verdict = compute_frontier([run], results, bar=0.3, archived_run_ids=None).subjects[0].verdict

        assert verdict is not None and verdict.model == "a"
        assert verdict.cost_decision == "not_separated"
        (tie,) = verdict.tied_with
        assert tie.p_value == 1.0

    def test_the_only_cleared_contestant_is_named_alone(self):
        run, results = self._corpus()
        results = [r for r in results if r.model != "a"]

        verdict = compute_frontier([run], results, bar=0.8, archived_run_ids=None).subjects[0].verdict

        assert verdict is not None and verdict.model == "b"
        assert (verdict.cost_decision, verdict.tied_with) == ("only_cleared", [])

    def test_an_interval_across_the_bar_is_undecided_and_never_the_pick(self):
        """The rule every campaign bar is read by: a straddle is neither a pass nor a failure.

        A perfect pass^k over four cases is 1.0 — above 0.8 as a number — but its interval reaches far below
        the bar, so it does not clear it, and the cheaper contestant cannot be named on it.
        """
        run = _fr_run()
        results = [
            *_fr_cases(run, 4, model="thin", variant_key="vk-thin", roles={"candidate": 0.05}),
            *_fr_cases(run, 20, model="deep", variant_key="vk-deep", roles={"candidate": 0.30}),
        ]

        pf = compute_frontier([run], results, bar=0.8, archived_run_ids=None).subjects[0]

        assert _point_by_model(pf, "thin").pass_hat_k == 1.0
        assert _point_by_model(pf, "thin").bar_decision == "undecided"
        assert (pf.n_cleared_bar, pf.n_undecided_bar) == (1, 1)
        assert pf.verdict is not None and pf.verdict.model == "deep"

    def test_a_bar_no_one_reaches_yields_zero_cleared(self):
        run = _fr_run()
        results = [
            _fr_result(run, model="a", variant_key="vk-a", test_case_id="c1", roles={"candidate": 0.5}, passes=False),
        ]

        pf = compute_frontier([run], results, bar=0.5, archived_run_ids=None).subjects[0]

        assert pf.n_cleared_bar == 0
        assert pf.verdict is None


class TestFrontierBarValidation:
    """A bar outside [0, 1] is refused, not clamped."""

    @pytest.mark.parametrize("bad_bar", [-0.1, 1.5, 2.0])
    def test_out_of_range_bar_raises(self, bad_bar):
        run = _fr_run()
        result = _fr_result(run, model="m", variant_key="vk", test_case_id="c1", roles={"candidate": 0.1})

        with pytest.raises(FrontierError):
            compute_frontier([run], [result], bar=bad_bar, archived_run_ids=None)


class TestFrontierPassHatKPoolsRepeatRuns:
    """pass^k pools a contestant's attempts at a case across the runs of one cell (#591).

    Summed per run, two runs of one configuration were two copies of each case at the shallow
    depth; now they are one case measured twice, while the same case under another context
    stays a case of its own beside it.
    """

    @staticmethod
    def _run(context_key, *, k_runs=1):
        return _fr_run(context_key=context_key, identity_version=IDENTITY_VERSION, k_runs=k_runs)

    def test_two_runs_of_one_cell_add_depth_not_cases(self):
        first, second = self._run("ctx-1"), self._run("ctx-1")
        results = [
            _fr_result(first, model="m", variant_key="vk", test_case_id="c1", passes=True),
            _fr_result(second, model="m", variant_key="vk", test_case_id="c1", passes=False),
        ]

        point = compute_frontier([first, second], results, archived_run_ids=None).subjects[0].points[0]

        assert (point.k, point.pass_hat_k, point.n_pass_cases) == (1, 0.5, 1)
        assert [(p["k"], p["pass_hat_k"], p["n_cases"]) for p in point.pass_hat_k_curve] == [(1, 0.5, 1), (2, 0.0, 1)]

    def test_runs_under_different_contexts_keep_one_case_as_two(self):
        first, second = self._run("ctx-1"), self._run("ctx-2")
        results = [
            _fr_result(first, model="m", variant_key="vk", test_case_id="c1", passes=True),
            _fr_result(second, model="m", variant_key="vk", test_case_id="c1", passes=False),
        ]

        point = compute_frontier([first, second], results, archived_run_ids=None).subjects[0].points[0]

        assert (point.pass_hat_k, point.n_pass_cases) == (0.5, 2)
        assert len(point.pass_hat_k_curve) == 1

    def test_every_contestant_is_ranked_at_the_subject_depth(self):
        """A k=1 run beside a k=3 run ranks the subject at pass^1 — one depth, never two."""
        deep, pilot = self._run("ctx-1", k_runs=3), self._run("ctx-2")
        # Sixteen cases: enough for each contestant's pass^1 interval to clear 0.5.
        results = [
            *(
                _fr_result(
                    deep,
                    model="deep",
                    variant_key="vk-deep",
                    test_case_id=f"c{case}",
                    passes=p,
                    k_iteration=i,
                    roles={"candidate": 0.1},
                )
                for case in range(16)
                for i, p in enumerate((True, True, False), start=1)
            ),
            *(
                _fr_result(
                    pilot,
                    model="pilot",
                    variant_key="vk-pilot",
                    test_case_id=f"c{case}",
                    passes=True,
                    roles={"candidate": 0.2},
                )
                for case in range(16)
            ),
        ]

        pf = compute_frontier([deep, pilot], results, bar=0.5, archived_run_ids=None).subjects[0]

        assert pf.k == 1
        assert _point_by_model(pf, "deep").pass_hat_k == pytest.approx(2 / 3)
        assert _point_by_model(pf, "deep").pass_hat_k_curve[-1] == {"k": 3, "pass_hat_k": 0.0, "n_cases": 16}
        # Both clear 0.5 at pass^1; the cheaper is named, and the verdict says which depth it was read at.
        assert pf.verdict is not None and (pf.verdict.model, pf.verdict.k) == ("deep", 1)

    def test_cost_per_acceptable_outcome_divides_by_one_attempts_pass_rate(self):
        """One attempt's cost over one attempt's pass probability, whatever depth is ranked."""
        run = self._run("ctx-1", k_runs=2)
        results = [
            _fr_result(
                run, model="m", variant_key="vk", test_case_id="c1", passes=p, k_iteration=i, roles={"candidate": 0.5}
            )
            for i, p in enumerate((True, False), start=1)
        ]

        point = compute_frontier([run], results, archived_run_ids=None).subjects[0].points[0]

        assert (point.k, point.pass_hat_k) == (2, 0.0)
        assert point.cost_per_acceptable_outcome == pytest.approx(0.5 / 0.5)


class TestFrontierHonestSampleSize:
    """Every point carries n and dispersion; an n=1 point is distinguishable."""

    def test_n_one_point_is_distinguishable_from_a_well_sampled_one(self):
        run = _fr_run()
        results = [
            _fr_result(run, model="thin", variant_key="vk-thin", test_case_id="c1", roles={"candidate": 0.1}),
            _fr_result(run, model="thick", variant_key="vk-thick", test_case_id="c1", roles={"candidate": 0.1}),
            _fr_result(run, model="thick", variant_key="vk-thick", test_case_id="c2", roles={"candidate": 0.1}),
            _fr_result(run, model="thick", variant_key="vk-thick", test_case_id="c3", roles={"candidate": 0.1}),
        ]

        pf = compute_frontier([run], results, archived_run_ids=None).subjects[0]

        assert _point_by_model(pf, "thin").n_results == 1
        assert _point_by_model(pf, "thin").n_pass_cases == 1
        assert _point_by_model(pf, "thick").n_results == 3
        assert _point_by_model(pf, "thick").n_cases == 3

    def test_composite_carries_a_dispersion(self):
        run = _fr_run()
        results = [
            _fr_result(run, model="m", variant_key="vk", test_case_id="c1", passes=True, roles={"candidate": 0.1}),
            _fr_result(run, model="m", variant_key="vk", test_case_id="c2", passes=False, roles={"candidate": 0.1}),
        ]

        point = compute_frontier([run], results, archived_run_ids=None).subjects[0].points[0]

        # two case-means (0.75 and 0.25) → a real SEM, not None
        assert point.mean_composite == pytest.approx(0.5)
        assert point.composite_sem is not None
        assert point.n_composite_cases == 2


_BOUNDARY_DIM = "refuse.harm"


def _with_boundary(results, score):
    """``results`` each scored ``score`` on a pass/fail boundary (guardrail) dimension, as the judge stamps it."""
    for result in results:
        result.rubric_scores = [
            *result.rubric_scores,
            RubricScore(dim=_BOUNDARY_DIM, scale="pass_fail", axis="boundary", score=score),
        ]
    return results


class TestFrontierBoundaryPillar:
    """#613: a contestant that breaches a boundary dimension is disqualified, named by it, and never the pick."""

    def _corpus(self):
        run = _fr_run()
        control = _with_boundary(_fr_cases(run, 20, model="m-ctrl", variant_key="vk-ctrl", roles={"candidate": 0.5}), 1)
        cheap = _with_boundary(_fr_cases(run, 20, model="m-cheap", variant_key="vk-cheap", roles={"candidate": 0.1}), 0)
        mid = _with_boundary(_fr_cases(run, 20, model="m-mid", variant_key="vk-mid", roles={"candidate": 0.3}), 1)
        return run, control + cheap + mid

    def test_a_contestant_failing_only_the_boundary_dim_is_disqualified_and_named(self):
        run, results = self._corpus()

        subject = compute_frontier(
            [run],
            results,
            bar=0.5,
            archived_run_ids=None,
            control_variant_key="vk-ctrl",
            guardrail_margins={_BOUNDARY_DIM: 0.2},
        ).subjects[0]

        cheap, mid = _point_by_model(subject, "m-cheap"), _point_by_model(subject, "m-mid")
        assert cheap.bar_decision == "cleared", "it clears the capability bar: only the boundary dim fails it"
        assert cheap.disqualified_by == [_BOUNDARY_DIM]
        assert [(c.dimension, c.decision) for c in cheap.boundary_checks] == [(_BOUNDARY_DIM, "breached")]
        assert [(c.dimension, c.decision) for c in mid.boundary_checks] == [(_BOUNDARY_DIM, "held")]
        assert subject.verdict is not None and subject.verdict.variant_key == "vk-mid"
        assert subject.verdict.boundary_unchecked == []
        assert (subject.boundary_dimensions, subject.n_disqualified) == ([_BOUNDARY_DIM], 1)
        assert subject.boundary_pillar is not None and subject.boundary_pillar.startswith("checked")

    def test_with_no_control_the_pillar_is_unchecked_and_the_verdict_says_so(self):
        run, results = self._corpus()

        frontier = compute_frontier([run], results, bar=0.5, archived_run_ids=None)

        subject = frontier.subjects[0]
        assert frontier.control_variant_key is None
        assert all(point.boundary_checks == [] and point.disqualified_by == [] for point in subject.points)
        assert subject.boundary_pillar is not None and subject.boundary_pillar.startswith("not checked")
        assert subject.verdict is not None and subject.verdict.boundary_unchecked == [_BOUNDARY_DIM]

    def test_a_subject_with_no_boundary_dim_states_nothing(self):
        run = _fr_run()
        results = _fr_cases(run, 3, model="m", variant_key="vk", roles={"candidate": 0.1})

        subject = compute_frontier([run], results, archived_run_ids=None, control_variant_key="vk").subjects[0]

        assert (subject.boundary_dimensions, subject.boundary_pillar) == ([], None)

    def test_a_frozen_frontier_carrying_the_retired_disclosure_still_reads(self):
        stored = compute_frontier([], [], archived_run_ids=None).to_dict()
        stored["two_pillar"] = {"boundary_pillar_available": False}

        assert FrontierResult.from_dict(stored).subjects == []
        with pytest.raises(ValidationError, match="two_pillar"):
            FrontierResult.model_validate(stored)


class TestFrontierExclusions:
    """An all-excluded corpus discloses its data cannot be placed, not that it is empty."""

    def test_a_result_without_its_run_is_counted_not_dropped(self):
        run = _fr_run()
        orphan = _fr_result(run, model="m", variant_key="vk", test_case_id="c1", roles={"candidate": 0.1})
        orphan.eval_run_id = "run-that-does-not-exist"

        result = compute_frontier([run], [orphan], archived_run_ids=None)

        assert result.subjects == []
        assert result.exclusions.results_without_run == 1
        assert result.exclusions.total == 1

    def test_a_run_with_no_subject_cannot_be_built_so_the_frontier_never_drops_one(self):
        """A frontier is per subject, so a run that could not supply one had nowhere to go.

        The refusal moved to capture. What reaches here always has a key, and the exclusion class
        that counted the alternative is gone with the state.
        """
        with pytest.raises(ValueError, match="subject_id"):
            _fr_run(subject_id="", subject_label="")

    def test_subject_filter_counts_others_as_filtered_out(self):
        run_a = _fr_run(subject_id="ent-a")
        run_b = _fr_run(subject_id="ent-b")
        results = [
            _fr_result(run_a, model="m", variant_key="vk-a", test_case_id="c1", roles={"candidate": 0.1}),
            _fr_result(run_b, model="m", variant_key="vk-b", test_case_id="c1", roles={"candidate": 0.1}),
        ]

        out = compute_frontier([run_a, run_b], results, subject_id="ent-a", archived_run_ids=None)

        assert [pf.subject_id for pf in out.subjects] == ["ent-a"]
        assert out.n_filtered_out == 1
        assert out.n_results == 1


def _hist_run(
    *,
    cases,
    score=None,
    cost=0.01,
    latency_ms=None,
    created_at,
    subject_id="ent-maple",
    subject_label="Maple",
    model="sonnet",
    template_id="tpl-1",
    variant_key=None,
    transcript=None,
    outcome=None,
):
    """Build one run and a result per case, for the history series tests.

    Args:
        cases: The run's frozen ``test_case_ids`` — a change between two runs is a
            suite-version boundary.
        score: Rubric score 1-5 for every case (composite = (score-1)/4), or None
            to leave the default so the composite lands at its fixture value.
        cost: ``cost_usd`` written on every result.
        latency_ms: ``total_ms`` written on every result, or None for no latency.
        created_at: Run timestamp — orders the series.
        subject_id: Subject id.
        subject_label: Subject display name.
        model: Candidate model, also the series label.
        template_id: Template the run scored against.
        variant_key: Resolved contestant key, or None to fall back to model.
        transcript: Raw 1-5 ``__transcript__`` axis score on every result, or None to
            leave the axis unjudged — which is what a result of an unjudged run carries.
        outcome: Raw 1-5 ``__outcome__`` axis score on every result, or None as above.

    Returns:
        ``(run, results)``.
    """
    snapshot = make_subject(subject_id, subject_label)
    run = make_eval_run(
        subject_snapshot=snapshot,
        template_id=template_id,
        test_case_ids=list(cases),
        created_at=created_at,
        candidate_model=model,
    )
    results = []
    for case_id in cases:
        overrides = dict(eval_run_id=run.id, scope_id=run.scope_id, test_case_id=case_id, model=model, cost_usd=cost)
        if score is not None:
            overrides["rubric_scores"] = [RubricScore(dim="conversation.tone", score=score, scale="ordinal")]
        if latency_ms is not None:
            overrides["latency"] = LatencyMetrics(total_ms=latency_ms)
        if variant_key is not None:
            overrides["variant_key"] = variant_key
        if transcript is not None:
            overrides["transcript_score"] = RubricScore(dim=TRANSCRIPT_DIM_ID, score=transcript, scale="ordinal")
        if outcome is not None:
            overrides["outcome_score"] = RubricScore(dim=OUTCOME_DIM_ID, score=outcome, scale="ordinal")
        results.append(make_eval_result(**overrides))
    return run, results


class TestTheReservedAxesCountLikeEveryJudgedScore:
    """The transcript and outcome axes follow `counted_score`, as the rubric dims do."""

    def test_a_candidate_failure_counts_each_axis_at_the_floor_in_the_projection_and_the_series(self):
        run, results = _hist_run(cases=["c1"], transcript=5, outcome=5, created_at="2026-07-01T00:00:00Z")
        results[0].covariates = {"truncated_rounds": 1.0}

        rows = {
            r.metric: r.value
            for r in project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None).records
            if r.metric in (METRIC_TRANSCRIPT, METRIC_OUTCOME)
        }
        assert rows == {METRIC_TRANSCRIPT: 1.0, METRIC_OUTCOME: 1.0}
        assert (
            compute_history([run], results, metric=METRIC_TRANSCRIPT, profile=_JUDGED_HOST, archived_run_ids=None)
            .series[0]
            .points[0]
            .value
            == 1.0
        )

    def test_a_harness_fault_leaves_each_axis_unmeasured(self):
        run, results = _hist_run(cases=["c1"], transcript=5, outcome=5, created_at="2026-07-01T00:00:00Z")
        results[0].infra_error = "delivery never landed"

        rows = {
            r.metric: r.value
            for r in project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None).records
            if r.metric in (METRIC_TRANSCRIPT, METRIC_OUTCOME)
        }
        assert rows == {METRIC_TRANSCRIPT: None, METRIC_OUTCOME: None}


class TestAMisStampedDualScoreAxisIsNotAnObservation:
    """`RubricScore.dim` is stored free text, and the projection already refuses a bad one.

    Without the same guard here, a mis-stamped row is meaned into a series and handed a
    regression verdict while `pivot` and `export_results` drop it and warn — the two
    surfaces disagreeing about what an axis is.
    """

    def test_a_dim_that_names_no_reserved_axis_is_excluded(self, caplog):
        run, results = _hist_run(cases=["c1", "c2"], transcript=5, created_at="2026-07-01T00:00:00Z")
        results[1].transcript_score = RubricScore(dim="conversation.tone", score=1, scale="ordinal")

        with caplog.at_level("WARNING"):
            out = compute_history([run], results, metric=METRIC_TRANSCRIPT, profile=_JUDGED_HOST, archived_run_ids=None)

        point = out.series[0].points[0]
        assert point.value == 5.0, f"a mis-stamped row entered the series (got {point.value})"
        assert point.n == 1
        assert "not that axis" in caplog.text, "a dropped defect must not read like an unjudged run"

    def test_the_other_axis_id_is_refused_too_not_only_a_template_dim(self):
        """Tighter than the projection's guard, which only asks 'is this a reserved id'."""
        run, results = _hist_run(cases=["c1", "c2"], outcome=5, created_at="2026-07-01T00:00:00Z")
        results[1].outcome_score = RubricScore(dim=TRANSCRIPT_DIM_ID, score=1, scale="ordinal")

        out = compute_history([run], results, metric=METRIC_OUTCOME, profile=_JUDGED_HOST, archived_run_ids=None)

        assert out.series[0].points[0].value == 5.0
        assert out.series[0].points[0].n == 1


class TestHistoryDualScoreAxes:
    """`history` series the two axes `export_results` and `results_pivot` already carry.

    The read the catalog says the PAIR exists for is a read over time — `__outcome__`
    falling while `__transcript__` holds is a world that changed, both falling together is
    a candidate that got worse — and until this landed it was answerable per campaign
    through `pivot` and not across runs at all.

    `score` stays refused and its reason does not reach these two: a raw judge score is
    per rubric DIMENSION, so there is no single per-run value to plot, while each axis is
    one value per result.
    """

    def test_the_transcript_axis_series_on_its_own_raw_scale(self):
        run_a, res_a = _hist_run(cases=["c1", "c2"], transcript=4, created_at="2026-07-01T00:00:00Z")
        run_b, res_b = _hist_run(cases=["c1", "c2"], transcript=2, created_at="2026-07-02T00:00:00Z")

        out = compute_history(
            [run_a, run_b], res_a + res_b, metric=METRIC_TRANSCRIPT, profile=_JUDGED_HOST, archived_run_ids=None
        )

        points = out.series[0].points
        assert [point.value for point in points] == [4.0, 2.0], "the 1-5 axis, not the 0-1 composite scale"
        assert points[1].delta_from_baseline == -2.0

    def test_the_outcome_axis_series_on_its_own_raw_scale(self):
        run_a, res_a = _hist_run(cases=["c1", "c2"], outcome=5, created_at="2026-07-01T00:00:00Z")
        run_b, res_b = _hist_run(cases=["c1", "c2"], outcome=3, created_at="2026-07-02T00:00:00Z")

        out = compute_history(
            [run_a, run_b], res_a + res_b, metric=METRIC_OUTCOME, profile=_JUDGED_HOST, archived_run_ids=None
        )

        assert [point.value for point in out.series[0].points] == [5.0, 3.0]

    def test_the_two_axes_move_independently_over_one_corpus(self):
        """The divergence IS the read: outcome falling while transcript holds is the world."""
        run_a, res_a = _hist_run(cases=["c1", "c2"], transcript=4, outcome=5, created_at="2026-07-01T00:00:00Z")
        run_b, res_b = _hist_run(cases=["c1", "c2"], transcript=4, outcome=2, created_at="2026-07-02T00:00:00Z")

        transcript = compute_history(
            [run_a, run_b], res_a + res_b, metric=METRIC_TRANSCRIPT, profile=_JUDGED_HOST, archived_run_ids=None
        )
        outcome = compute_history(
            [run_a, run_b], res_a + res_b, metric=METRIC_OUTCOME, profile=_JUDGED_HOST, archived_run_ids=None
        )

        assert [point.value for point in transcript.series[0].points] == [4.0, 4.0]
        assert [point.value for point in outcome.series[0].points] == [5.0, 2.0]

    def test_a_point_is_described_by_the_aggregate_not_the_per_observation_axis(self):
        """A point holds the MEAN of the axis, which is a different quantity from one score."""
        run, results = _hist_run(cases=["c1"], transcript=4, outcome=3, created_at="2026-07-01T00:00:00Z")

        transcript = compute_history(
            [run], results, metric=METRIC_TRANSCRIPT, profile=_JUDGED_HOST, archived_run_ids=None
        )
        outcome = compute_history([run], results, metric=METRIC_OUTCOME, profile=_JUDGED_HOST, archived_run_ids=None)

        assert transcript.measure.name == "mean_transcript_score"
        assert outcome.measure.name == "mean_outcome_score"
        # The formula a reader checks the arithmetic against names the mean over cases,
        # never the registry's per-observation sentence.
        assert "mean over test cases" in transcript.formula

    def test_an_unjudged_axis_contributes_nothing_rather_than_a_zero(self):
        """An unjudged result is unmeasured on the axis, and 1-5 has no zero."""
        judged_run, judged = _hist_run(cases=["c1"], transcript=4, created_at="2026-07-01T00:00:00Z")
        unjudged_run, unjudged = _hist_run(cases=["c1"], created_at="2026-07-02T00:00:00Z")

        out = compute_history(
            [judged_run, unjudged_run],
            judged + unjudged,
            metric=METRIC_TRANSCRIPT,
            profile=_JUDGED_HOST,
            archived_run_ids=None,
        )

        points = out.series[0].points
        assert points[0].value == 4.0
        assert points[1].value is None, "an unjudged axis is unmeasured, not a 0"
        assert points[1].n == 0

    def test_an_infra_excluded_cell_is_not_a_dual_score_observation(self):
        """A judge reading a truncated transcript is not measuring the candidate.

        The same policy `total_ms` and the composite already apply on this surface, which
        issues a regression verdict on whatever it returns — leaving these two whole would
        make the quality measures on ONE surface answer different questions.
        """
        run, results = _hist_run(cases=["c1", "c2"], transcript=5, created_at="2026-07-01T00:00:00Z")
        results[1].transcript_score = RubricScore(dim=TRANSCRIPT_DIM_ID, score=1, scale="ordinal")
        results[1].infra_error = "apparatus: cassette miss in replay mode"

        out = compute_history([run], results, metric=METRIC_TRANSCRIPT, profile=_JUDGED_HOST, archived_run_ids=None)

        point = out.series[0].points[0]
        assert point.value == 5.0, f"the harness's own cell entered a regression series (got {point.value})"
        assert point.n == 1

    def test_a_candidate_failure_the_judge_scored_is_still_an_observation(self):
        """The guard must not widen into 'any error' — a scored failure is real data."""
        run, results = _hist_run(cases=["c1", "c2"], transcript=5, created_at="2026-07-01T00:00:00Z")
        results[1].transcript_score = RubricScore(dim=TRANSCRIPT_DIM_ID, score=1, scale="ordinal")
        results[1].candidate_error = "the candidate returned nothing"

        out = compute_history([run], results, metric=METRIC_TRANSCRIPT, profile=_JUDGED_HOST, archived_run_ids=None)

        assert out.series[0].points[0].value == 3.0


class TestHistoryWithholdsAttributionOnAScenarioBoundAxis:
    """`__outcome__` drifts when externals drift, so a decline there is not the candidate's.

    The flag still FIRES — staying silent would hide a real decline — and says it does not
    attribute. Keyed on the measure's `transferability_class` rather than on its name, so
    the rule is the catalog's and not a metric list kept in step by hand.
    """

    def _declining(self, metric, **scores):
        # Ten cases: every case moves by one amount, read by the bounded test on the 1-5 range (#597), which shows a
        # three-point move from ten cases and not from six.
        cases = [f"c{index}" for index in range(1, 11)]
        run_a, res_a = _hist_run(cases=cases, created_at="2026-07-01T00:00:00Z", **{k: v[0] for k, v in scores.items()})
        run_b, res_b = _hist_run(cases=cases, created_at="2026-07-02T00:00:00Z", **{k: v[1] for k, v in scores.items()})
        return compute_history(
            [run_a, run_b],
            res_a + res_b,
            metric=metric,
            min_absolute_change=0.05,
            profile=_JUDGED_HOST,
            archived_run_ids=None,
        )

    def test_a_declining_outcome_series_flags_and_withholds_attribution(self):
        out = self._declining(METRIC_OUTCOME, outcome=(5, 2))

        flag = out.series[0].points[1].regression
        assert flag is not None
        assert flag.label == "regressed", "silence would hide a real decline"
        assert flag.attribution_withheld is True
        assert flag.delta == -3.0, "the delta is stated, not suppressed"

    def test_the_answer_says_why_once_beside_the_measure(self):
        out = self._declining(METRIC_OUTCOME, outcome=(5, 2))

        assert out.attribution_disclosure is not None
        assert "mean_outcome_score" in out.attribution_disclosure
        assert "scenario-bound" in out.attribution_disclosure

    def test_an_improvement_is_withheld_on_the_same_axis(self):
        """Attribution is a property of the measure, not of the direction it moved."""
        out = self._declining(METRIC_OUTCOME, outcome=(2, 5))

        flag = out.series[0].points[1].regression
        assert flag is not None
        assert flag.label == "improved"
        assert flag.attribution_withheld is True

    def test_the_transcript_axis_attributes(self):
        """The contrast that makes the flag informative: this axis holds when the world moves."""
        out = self._declining(METRIC_TRANSCRIPT, transcript=(5, 2))

        flag = out.series[0].points[1].regression
        assert flag is not None
        assert flag.label == "regressed"
        assert flag.attribution_withheld is False
        assert out.attribution_disclosure is None

    def test_no_measure_outside_the_scenario_bound_class_withholds(self):
        """Quantified over the set, so a measure added later cannot arrive miscategorised."""
        run, results = _hist_run(
            cases=["c1"],
            score=5,
            cost=0.02,
            latency_ms=1200.0,
            transcript=4,
            outcome=3,
            created_at="2026-07-01T00:00:00Z",
        )

        for row_name in sorted(reporting.HISTORY_METRICS):
            out = compute_history([run], results, metric=row_name, profile=_JUDGED_HOST, archived_run_ids=None)
            expected = out.measure.transferability_class == "scenario_bound"
            assert (out.attribution_disclosure is not None) is expected, (
                f"{row_name!r} resolves to {out.measure.name!r} at class "
                f"{out.measure.transferability_class!r}; the disclosure must follow the class"
            )

    def test_the_disclosure_survives_an_answer_with_no_series(self):
        """A property of the MEASURE, so an empty corpus still carries it."""
        out = compute_history([], [], metric=METRIC_OUTCOME, profile=_JUDGED_HOST, archived_run_ids=None)

        assert out.series == []
        assert out.attribution_disclosure is not None


class TestHistoryCassetteStep:
    """A capture→replay step can read as "flat (Δ -0.05) · not significant".

    Cassette mode is outside the variant key, so both runs sit under one contestant
    heading and the step between them renders as a quality time series. It is not:
    the measure moved between a measurement and a substitution. Flagged at the STEP,
    beside `crosses_epoch`, because that is where the confound lands — a
    series-level note would be true of the whole series and would not say which step
    crossed.
    """

    def _series(self, mode_a, mode_b):
        run_a, res_a = _hist_run(cases=["c1", "c2"], score=5, created_at="2026-07-01T00:00:00Z", variant_key="vk-1")
        run_b, res_b = _hist_run(cases=["c1", "c2"], score=3, created_at="2026-07-02T00:00:00Z", variant_key="vk-1")
        run_a, run_b = (
            # Rebuilt whole rather than assigned field by field: a replay names the corpus it serves,
            # and the two fields are valid only together.
            EvalRun.model_validate(
                {
                    **run.model_dump(),
                    "cassette_mode": mode,
                    "cassette_corpus_id": "run-capture" if mode == "replay" else None,
                }
            )
            for run, mode in ((run_a, mode_a), (run_b, mode_b))
        )
        return (
            compute_history(
                [run_a, run_b], res_a + res_b, metric=METRIC_COMPOSITE, profile=_JUDGED_HOST, archived_run_ids=None
            )
            .series[0]
            .points
        )

    def test_a_step_across_a_mode_change_is_flagged(self):
        points = self._series("capture", "replay")

        assert points[1].regression is not None
        assert points[1].regression.crosses_cassette_mode is True

    def test_a_step_within_one_mode_is_not(self):
        points = self._series("off", "off")

        assert points[1].regression is not None
        assert points[1].regression.crosses_cassette_mode is False

    def test_the_baseline_point_carries_no_flag_to_cross(self):
        """The first point has no predecessor, so there is no step to qualify."""
        points = self._series("capture", "replay")

        assert points[0].regression is None

    def test_every_point_names_its_own_recorded_mode(self):
        """On every measure, not only cost: a replayed run's latency and quality are substituted too."""
        points = self._series("capture", "replay")

        assert [p.cassette_mode for p in points] == ["capture", "replay"]

    def test_the_verdict_is_disclosed_and_never_suppressed(self):
        """Descriptive, exactly as `crosses_epoch` is — no label is withheld, no delta adjusted."""
        points = self._series("capture", "replay")

        assert points[1].regression.label
        assert points[1].regression.delta is not None


class TestHistory:
    """Per-measure longitudinal series with suite-version epochs and honest regression flags."""

    def test_every_point_carries_its_denominators(self):
        """A point resting on 3 cases must not read like one resting on 30."""
        run, results = _hist_run(cases=["c1", "c2", "c3"], score=5, created_at="2026-07-01T00:00:00Z")

        out = compute_history([run], results, metric=METRIC_COMPOSITE, profile=_JUDGED_HOST, archived_run_ids=None)

        point = out.series[0].points[0]
        assert point.value == 1.0
        assert point.n == 3
        assert point.n_cases == 3
        assert point.sem == 0.0  # constant sample of three — measured, not absent

    def test_baseline_pointer_and_delta_from_it(self):
        base_run, base_results = _hist_run(cases=["c1", "c2", "c3"], score=5, created_at="2026-07-01T00:00:00Z")
        next_run, next_results = _hist_run(cases=["c1", "c2", "c3"], score=1, created_at="2026-07-02T00:00:00Z")

        out = compute_history(
            [base_run, next_run],
            base_results + next_results,
            metric=METRIC_COMPOSITE,
            profile=_JUDGED_HOST,
            archived_run_ids=None,
        )

        points = out.series[0].points
        assert points[0].is_baseline is True
        assert points[0].delta_from_baseline == 0.0
        assert points[1].is_baseline is False
        assert points[1].delta_from_baseline == -1.0  # 0.0 - 1.0

    def test_an_epoch_boundary_splits_the_series(self):
        """A suite edit reads as an epoch line, not a mysterious step."""
        run_a, res_a = _hist_run(cases=["c1", "c2", "c3"], score=5, created_at="2026-07-01T00:00:00Z")
        run_b, res_b = _hist_run(cases=["c1", "c2", "c3", "c4"], score=5, created_at="2026-07-02T00:00:00Z")

        out = compute_history(
            [run_a, run_b], res_a + res_b, metric=METRIC_COMPOSITE, profile=_JUDGED_HOST, archived_run_ids=None
        )

        points = out.series[0].points
        assert points[0].epoch == 1
        assert points[0].epoch_boundary is False
        assert points[1].epoch == 2
        assert points[1].epoch_boundary is True

    def test_a_cost_point_carries_the_composition_its_dollars_summed(self):
        """A cost step caused by search spend joining the total is not a cost regression.

        The value moves exactly as a real regression would, and nothing else on the point
        can separate the two — which is the same problem ``epoch_boundary`` solves for a
        change in the case set, on the other axis.
        """
        unpriced = ["candidate", "inner_agent", "judge", "simulator"]
        run_a, res_a = _hist_run(cases=["c1", "c2"], cost=0.10, created_at="2026-07-01T00:00:00Z")
        run_b, res_b = _hist_run(cases=["c1", "c2"], cost=0.15, created_at="2026-07-02T00:00:00Z")
        for r in res_a:
            r.cost_roles = unpriced
        for r in res_b:
            r.cost_roles = [*unpriced, "external"]

        out = compute_history(
            [run_a, run_b], res_a + res_b, metric=METRIC_COST_USD, profile=_JUDGED_HOST, archived_run_ids=None
        )

        points = out.series[0].points
        assert points[0].cost_compositions == [unpriced]
        assert points[1].cost_compositions == [[*unpriced, "external"]]

    def test_a_non_cost_series_carries_no_composition(self):
        """Asking a quality point what roles its dollars covered answers a question it is not."""
        run, results = _hist_run(cases=["c1", "c2"], score=5, cost=0.10, created_at="2026-07-01T00:00:00Z")
        for r in results:
            r.cost_roles = ["candidate", "inner_agent", "judge", "simulator"]

        out = compute_history([run], results, metric=METRIC_COMPOSITE, profile=_JUDGED_HOST, archived_run_ids=None)

        assert out.series[0].points[0].cost_compositions == []

    def test_a_stable_suite_stays_in_one_epoch(self):
        """Re-running the same case set is not an epoch boundary."""
        run_a, res_a = _hist_run(cases=["c1", "c2"], score=5, created_at="2026-07-01T00:00:00Z")
        run_b, res_b = _hist_run(cases=["c1", "c2"], score=4, created_at="2026-07-02T00:00:00Z")

        out = compute_history(
            [run_a, run_b], res_a + res_b, metric=METRIC_COMPOSITE, profile=_JUDGED_HOST, archived_run_ids=None
        )

        points = out.series[0].points
        assert [p.epoch for p in points] == [1, 1]
        assert points[1].epoch_boundary is False

    def test_a_significant_decline_within_an_epoch_is_flagged_regressed(self):
        # Every case moves by one amount, read by the bounded test on the composite's range (#597): ten cases show it.
        cases = [f"c{index}" for index in range(1, 11)]
        run_a, res_a = _hist_run(cases=cases, score=5, created_at="2026-07-01T00:00:00Z")
        run_b, res_b = _hist_run(cases=cases, score=2, created_at="2026-07-02T00:00:00Z")

        out = compute_history(
            [run_a, run_b],
            res_a + res_b,
            metric=METRIC_COMPOSITE,
            min_absolute_change=0.05,
            profile=_JUDGED_HOST,
            archived_run_ids=None,
        )

        flag = out.series[0].points[1].regression
        assert flag is not None
        assert flag.label == "regressed"
        assert flag.crosses_epoch is False
        assert flag.n_pairs == len(cases)

    def test_a_change_below_threshold_is_not_flagged(self):
        """The joint gate at the surface: significant but tiny reads below_threshold, never "no change" (#592).

        ``cost_usd`` is an engine-core measure, which declares no margin and which a host cannot give
        one, so no equivalence test runs and the answer says so.
        """
        cases = ["c1", "c2", "c3", "c4", "c5", "c6"]
        run_a, res_a = _hist_run(cases=cases, cost=0.100, created_at="2026-07-01T00:00:00Z")
        run_b, res_b = _hist_run(cases=cases, cost=0.101, created_at="2026-07-02T00:00:00Z")
        # A rise of about +0.001 that varies a little by case: a t-test reads it. One amount on every case would
        # not be called at all, since cost declares no range (#597).
        for index, result in enumerate(res_b):
            result.cost_usd = 0.101 + (0.00001 if index % 2 else -0.00001)

        out = compute_history(
            [run_a, run_b],
            res_a + res_b,
            metric=METRIC_COST_USD,
            min_absolute_change=0.05,
            profile=_JUDGED_HOST,
            archived_run_ids=None,
        )

        flag = out.series[0].points[1].regression
        assert flag is not None
        assert flag.significant is True  # a consistent +0.001 move
        assert flag.exceeds_threshold is False
        assert flag.label == "below_threshold"
        assert out.equivalence_margin is None
        assert flag.equivalence_margin is None and flag.equivalence_p is None
        from threetears.evals.analysis.stats import PAIRED_TEST_NAME

        assert flag.test == PAIRED_TEST_NAME

    def test_a_uniform_move_on_a_measure_with_no_range_is_not_separated_and_says_why(self):
        """Every case $0.05 dearer: cost declares no range, so no test of the mean can call it, at any n (#597).

        The sign-flip reading called this ``regressed`` from six cases; it tests symmetry, not the mean. The flag
        reads not separated, states no p, and names the remedy.
        """
        cases = [f"c{index}" for index in range(1, 21)]
        run_a, res_a = _hist_run(cases=cases, cost=0.10, created_at="2026-07-01T00:00:00Z")
        run_b, res_b = _hist_run(cases=cases, cost=0.15, created_at="2026-07-02T00:00:00Z")

        out = compute_history(
            [run_a, run_b],
            res_a + res_b,
            metric=METRIC_COST_USD,
            min_absolute_change=0.01,
            profile=_JUDGED_HOST,
            archived_run_ids=None,
        )

        flag = out.series[0].points[1].regression
        assert flag is not None
        assert (flag.label, flag.significant, flag.p) == ("not_separated", False, None)
        assert flag.not_separated_reason == UNIFORM_MOVE_NEEDS_RANGE

    def test_flags_carry_their_test_and_thresholds(self):
        run_a, res_a = _hist_run(cases=["c1", "c2"], score=5, created_at="2026-07-01T00:00:00Z")
        run_b, res_b = _hist_run(cases=["c1", "c2"], score=2, created_at="2026-07-02T00:00:00Z")

        out = compute_history(
            [run_a, run_b],
            res_a + res_b,
            metric=METRIC_COMPOSITE,
            min_absolute_change=0.1,
            min_relative_change=0.2,
            profile=_JUDGED_HOST,
            archived_run_ids=None,
        )

        flag = out.series[0].points[1].regression
        assert flag is not None
        assert "paired" in flag.test and "t-test" in flag.test
        assert flag.min_absolute_change == 0.1
        assert flag.min_relative_change == 0.2

    def test_crossing_an_epoch_is_disclosed_on_the_flag(self):
        """Across a suite change the paired test rests only on shared cases — say so."""
        run_a, res_a = _hist_run(cases=["c1", "c2", "c3"], score=5, created_at="2026-07-01T00:00:00Z")
        run_b, res_b = _hist_run(cases=["c1", "c2", "c3", "c4"], score=2, created_at="2026-07-02T00:00:00Z")

        out = compute_history(
            [run_a, run_b],
            res_a + res_b,
            metric=METRIC_COMPOSITE,
            min_absolute_change=0.05,
            profile=_JUDGED_HOST,
            archived_run_ids=None,
        )

        flag = out.series[0].points[1].regression
        assert flag is not None
        assert flag.crosses_epoch is True
        assert flag.n_pairs == 3  # only the shared cases, not the new c4

    def test_a_single_point_series_has_no_regression_flag(self):
        run, results = _hist_run(cases=["c1", "c2"], score=5, created_at="2026-07-01T00:00:00Z")

        out = compute_history([run], results, metric=METRIC_COMPOSITE, profile=_JUDGED_HOST, archived_run_ids=None)

        points = out.series[0].points
        assert len(points) == 1
        assert points[0].regression is None

    def test_subjects_are_never_pooled(self):
        """Composite is not comparable across subjects — two subjects, two sets of series."""
        run_a, res_a = _hist_run(cases=["c1"], score=5, created_at="2026-07-01T00:00:00Z", subject_id="ent-a")
        run_b, res_b = _hist_run(cases=["c1"], score=5, created_at="2026-07-01T00:00:00Z", subject_id="ent-b")

        out = compute_history(
            [run_a, run_b], res_a + res_b, metric=METRIC_COMPOSITE, profile=_JUDGED_HOST, archived_run_ids=None
        )

        assert sorted({s.subject_id for s in out.series}) == ["ent-a", "ent-b"]

    def test_series_are_grouped_by_variant(self):
        """Two variants of one model are two series, not one pooled trend."""
        run_a, res_a = _hist_run(cases=["c1"], score=5, created_at="2026-07-01T00:00:00Z", variant_key="vk-1")
        run_b, res_b = _hist_run(cases=["c1"], score=3, created_at="2026-07-02T00:00:00Z", variant_key="vk-2")

        out = compute_history(
            [run_a, run_b], res_a + res_b, metric=METRIC_COMPOSITE, profile=_JUDGED_HOST, archived_run_ids=None
        )

        assert sorted({s.variant_key for s in out.series}) == ["vk-1", "vk-2"]

    def test_cost_series_uses_the_lower_is_better_direction(self):
        run, results = _hist_run(cases=["c1", "c2"], cost=0.05, created_at="2026-07-01T00:00:00Z")

        out = compute_history([run], results, metric=METRIC_COST_USD, profile=_JUDGED_HOST, archived_run_ids=None)

        assert out.higher_is_better is False
        assert out.measure.unit == "usd"
        assert out.series[0].points[0].value == 0.05

    def test_total_ms_series_counts_only_measured_latencies(self):
        """total_ms is an advertised metric; an unmeasured latency drops, never reads as zero.

        The exact invariant established here (an unmeasured span persists as
        null, not 0.0) has to survive into the series, or a partly-harvested run's
        latency mean would be pulled toward a phantom zero.
        """
        snapshot = make_subject("ent-maple", "Maple")
        run = make_eval_run(
            subject_snapshot=snapshot,
            template_id="tpl-1",
            test_case_ids=["c1", "c2", "c3"],
            created_at="2026-07-01T00:00:00Z",
        )
        measured = make_eval_result(
            eval_run_id=run.id, scope_id=run.scope_id, test_case_id="c1", latency=LatencyMetrics(total_ms=120.0)
        )
        null_total = make_eval_result(
            eval_run_id=run.id, scope_id=run.scope_id, test_case_id="c2", latency=LatencyMetrics(total_ms=None)
        )
        no_latency = make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, test_case_id="c3")  # latency is None

        out = compute_history(
            [run], [measured, null_total, no_latency], metric="total_ms", profile=_JUDGED_HOST, archived_run_ids=None
        )

        point = out.series[0].points[0]
        assert point.value == 120.0  # the one measured case, not diluted by the two unmeasured ones
        assert point.n == 1
        assert point.n_cases == 1
        assert out.higher_is_better is False

    def test_total_ms_point_is_unmeasured_not_instant_when_no_latency_harvested(self):
        """A point that harvested no latency reads as unknown, never as the fastest."""
        snapshot = make_subject("ent-maple", "Maple")
        run = make_eval_run(
            subject_snapshot=snapshot, template_id="tpl-1", test_case_ids=["c1"], created_at="2026-07-01T00:00:00Z"
        )
        no_latency = make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, test_case_id="c1")

        out = compute_history([run], [no_latency], metric="total_ms", profile=_JUDGED_HOST, archived_run_ids=None)

        point = out.series[0].points[0]
        assert point.value is None
        assert point.n == 0

    def test_an_unknown_metric_is_refused_not_answered_empty(self):
        run, results = _hist_run(cases=["c1"], score=5, created_at="2026-07-01T00:00:00Z")

        with pytest.raises(HistoryError, match="unknown history metric"):
            compute_history([run], results, metric="compsite", profile=_JUDGED_HOST, archived_run_ids=None)

    def test_score_is_still_refused_and_the_refusal_names_where_it_does_aggregate(self):
        """`score` is not a typo — it is a real measure this surface cannot series.

        A series carries one value per contestant per run and a raw judge score is per
        rubric DIMENSION, so the refusal stands. What must not stand is a message that
        only lists the three alternatives: `score` is accepted by the pivot, emitted by
        the export and described in the catalog, so a caller reading "expected one of
        composite, cost_usd, total_ms" is left to guess whether the measure is wrong,
        misspelled, or merely unimplemented here.
        """
        run, results = _hist_run(cases=["c1"], score=5, created_at="2026-07-01T00:00:00Z")

        with pytest.raises(HistoryError) as refusal:
            compute_history([run], results, metric=METRIC_SCORE, profile=_JUDGED_HOST, archived_run_ids=None)

        message = str(refusal.value)
        assert "DIMENSION" in message, "must say why: the measure is per-dimension, not per-run"
        assert "mean_score" in message, "must name the aggregate the pivot reports it as"
        assert "pivot" in message, "must name the surface that aggregates it"

    def test_a_goal_state_series_is_refused_naming_the_check_coordinate(self):
        """Derived from SCOPED_METRICS, so a scoped metric added later is refused the same way."""
        run, results = _hist_run(cases=["c1"], score=5, created_at="2026-07-01T00:00:00Z")

        with pytest.raises(HistoryError) as refusal:
            compute_history([run], results, metric=METRIC_GOAL_STATE, profile=_JUDGED_HOST, archived_run_ids=None)

        message = str(refusal.value)
        assert "ONE goal-state check" in message
        assert "'goal_check' on an axis" in message and "goal_state_pass_rate" in message

    def test_a_corpus_that_once_would_have_been_all_excluded_cannot_be_built(self):
        """History used to disclose a corpus it could not group; there is no such corpus now.

        Every run it can be handed carries a subject key, so an empty series can no longer mean
        "the data exists but cannot be grouped" — which is what the disclosure was for.
        """
        with pytest.raises(ValueError, match="subject_id"):
            _hist_run(cases=["c1"], score=5, created_at="2026-07-01T00:00:00Z", subject_id="", subject_label="")

    def test_subject_filter_counts_others_as_filtered_out(self):
        run_a, res_a = _hist_run(cases=["c1"], score=5, created_at="2026-07-01T00:00:00Z", subject_id="ent-a")
        run_b, res_b = _hist_run(cases=["c1"], score=5, created_at="2026-07-01T00:00:00Z", subject_id="ent-b")

        out = compute_history(
            [run_a, run_b],
            res_a + res_b,
            metric=METRIC_COMPOSITE,
            subject_id="ent-a",
            profile=_JUDGED_HOST,
            archived_run_ids=None,
        )

        assert [s.subject_id for s in out.series] == ["ent-a"]
        assert out.n_filtered_out == 1
        assert out.n_results == 1


class TestTheCatalogNameIsTheNameTheSurfacesAccept:
    """`list_metrics` publishes the catalog; a name it publishes must not be refused.

    Reproduced on a served corpus: `list_metrics` names the composite-quality measure
    `mean_composite` and nothing else, and `history(metric='mean_composite')` came
    back "unknown history metric ... expected one of composite, cost_usd, total_ms".
    An operator who read the catalog and used what it said got a validation error,
    and the name that worked appeared in no catalog. The two vocabularies are
    reconciled by ACCEPTING the catalog name (`reporting.resolve_measure_name`),
    never by renaming the row constants — a row is one observation and `composite`
    is its honest name, which is why `export_results` still emits it in the metric
    column.
    """

    def test_history_accepts_the_catalog_name_for_the_composite_series(self):
        run, results = _hist_run(cases=["c1", "c2"], score=5, created_at="2026-07-01T00:00:00Z")

        catalog = compute_history([run], results, metric="mean_composite", profile=_JUDGED_HOST, archived_run_ids=None)
        short = compute_history([run], results, metric=METRIC_COMPOSITE, profile=_JUDGED_HOST, archived_run_ids=None)

        assert catalog.model_dump() == short.model_dump()

    def test_every_history_measure_is_reachable_by_the_name_the_catalog_publishes(self):
        """Quantified over the set, so a measure added later cannot be half-aliased."""
        run, results = _hist_run(
            cases=["c1"],
            score=5,
            cost=0.02,
            latency_ms=1200.0,
            transcript=4,
            outcome=3,
            created_at="2026-07-01T00:00:00Z",
        )

        for row_name in sorted(reporting.HISTORY_METRICS):
            catalog_name = _catalog_name_of(row_name)
            assert describe_measure(catalog_name, _JUDGED_HOST.measures).family is not None, (
                f"{catalog_name!r} is offered as the catalog name of {row_name!r} but the registry does not "
                "describe it, so `list_metrics` never publishes it and the alias points at nothing"
            )

            out = compute_history([run], results, metric=catalog_name, profile=_JUDGED_HOST, archived_run_ids=None)

            assert out.metric == row_name
            assert out.measure.name == catalog_name

    def test_pivot_accepts_the_catalog_name_for_what_a_cell_holds(self):
        run, result = _run_with_results()
        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        catalog = compute_pivot(
            records, row_factor="rubric_dim", column_factor="model", metric="mean_score", profile=_JUDGED_HOST
        )
        short = compute_pivot(
            records, row_factor="rubric_dim", column_factor="model", metric=METRIC_SCORE, profile=_JUDGED_HOST
        )

        assert catalog.model_dump() == short.model_dump()

    def test_every_projected_measure_is_reachable_by_the_name_the_catalog_publishes(self):
        run, result = _run_with_results()
        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        for row_name in sorted(reporting.PROJECTED_METRICS):
            catalog_name = _catalog_name_of(row_name)

            table = compute_pivot(
                records, row_factor="rubric_dim", column_factor="subject_id", metric=catalog_name, profile=_JUDGED_HOST
            )

            assert table.metric == row_name
            assert table.measure.name == catalog_name

    def test_the_alias_table_is_one_table_read_both_ways(self):
        """Derived, not written twice — a pair added to one direction only cannot exist."""
        accepted = sorted(reporting.PROJECTED_METRICS | reporting.HISTORY_METRICS)
        catalog = {observation: _catalog_name_of(observation) for observation in accepted}

        assert len(set(catalog.values())) == len(catalog), (
            "two row measures publish the same catalog name, so the inverse silently drops one"
        )
        for observation, aggregate in catalog.items():
            assert reporting.resolve_measure_name(aggregate) == observation

    def test_a_name_outside_the_alias_table_is_passed_through_to_be_refused(self):
        """The alias must not become a guesser: an unknown name reaches its own refusal."""
        assert reporting.resolve_measure_name("compsite") == "compsite"
        assert reporting.resolve_measure_name(METRIC_COMPOSITE) == METRIC_COMPOSITE

    def test_a_catalogued_measure_this_surface_cannot_series_is_still_refused(self):
        """Accepting catalog names does not mean accepting the whole catalog.

        `pass_hat_k` is a real registry measure and a real quantity — it is simply
        not one `history` can series per run from these rows. Answering it would be
        the empty-series-as-silent-nothing that `HistoryError` exists to prevent.
        """
        run, results = _hist_run(cases=["c1"], score=5, created_at="2026-07-01T00:00:00Z")

        with pytest.raises(HistoryError, match="unknown history metric"):
            compute_history([run], results, metric="pass_hat_k", profile=_JUDGED_HOST, archived_run_ids=None)

    def test_the_refusal_names_both_vocabularies(self):
        """The refusal an operator reads must name the catalog name, or it sends them in a circle."""
        run, results = _hist_run(cases=["c1"], score=5, created_at="2026-07-01T00:00:00Z")

        with pytest.raises(HistoryError) as excinfo:
            compute_history([run], results, metric="compsite", profile=_JUDGED_HOST, archived_run_ids=None)

        message = str(excinfo.value)
        for row_name in reporting.HISTORY_METRICS:
            assert row_name in message
            assert _catalog_name_of(row_name) in message

    def test_the_pivot_refusal_names_both_vocabularies(self):
        run, result = _run_with_results()
        records = project_score_records([run], [result], profile=_JUDGED_HOST, archived_run_ids=None).records

        with pytest.raises(PivotError) as excinfo:
            compute_pivot(
                records, row_factor="model", column_factor="model", metric="mean_composit", profile=_JUDGED_HOST
            )

        message = str(excinfo.value)
        for row_name in reporting.PROJECTED_METRICS:
            assert row_name in message
            assert _catalog_name_of(row_name) in message

    def test_a_latency_series_describes_the_mean_it_holds_not_one_results_wall_clock(self):
        """A point is a mean over cases; `total_ms` describes one result and its partition.

        The descriptor drives the rendered "Measure: ..." line on both surfaces, so
        describing a series of means with the row measure told an operator the number
        partitions into llm/tool/orchestration. The mean does not.
        """
        run, results = _hist_run(cases=["c1"], score=5, latency_ms=1200.0, created_at="2026-07-01T00:00:00Z")

        out = compute_history(
            [run], results, metric=reporting.METRIC_TOTAL_MS, profile=_JUDGED_HOST, archived_run_ids=None
        )

        assert out.measure.name == "mean_total_ms"
        assert out.measure.higher_is_better is False
        assert out.measure.unit == "ms"


class TestLatencyPartition:
    """The `total_ms` partition: named parts plus a named remainder, or a sentence.

    The remainder exists so a whole-run latency movement can be PLACED. Before it, a
    total that moved while `llm_ms` and `tool_ms` stayed flat was reported as an
    unexplained gap, which reads in an analysis exactly like a measurement fault.
    """

    def test_the_remainder_closes_the_partition(self):
        partition = decompose_total_ms(LatencyMetrics(total_ms=1000.0, llm_ms=600.0, tool_ms=250.0))

        assert partition.orchestration_ms == 150.0
        assert partition.withheld is None
        assert partition.llm_ms + partition.tool_ms + partition.orchestration_ms == pytest.approx(partition.total_ms)

    def test_the_disjoint_measures_are_not_folded_into_the_remainder(self):
        """The drain wait and the judge phase fall OUTSIDE the turn roots the total sums.

        Subtracting either would describe no stretch of wall-clock — the arithmetic that
        produced a ~95-second remainder nobody could place. Both are set large here
        precisely so folding one in would be impossible to miss.
        """
        partition = decompose_total_ms(
            LatencyMetrics(total_ms=1000.0, llm_ms=600.0, tool_ms=250.0, async_wait_ms=95_000.0, judge_ms=4_000.0)
        )

        assert partition.orchestration_ms == 150.0

    def test_an_unmeasured_component_withholds_the_split_rather_than_zero_filling_it(self):
        """A cell can measure an `llm.call` while producing no turn root at all.

        Treating the missing total as zero would publish a remainder computed from a
        number nobody observed, in the same units as the ones somebody did.
        """
        partition = decompose_total_ms(LatencyMetrics(llm_ms=600.0))

        assert partition.orchestration_ms is None
        assert WITHHELD_UNMEASURED_COMPONENT in partition.withheld
        assert "total_ms" in partition.withheld
        assert "tool_ms" in partition.withheld

    def test_a_cell_that_timed_nothing_gets_a_sentence_not_a_null(self):
        partition = decompose_total_ms(None)

        assert partition.orchestration_ms is None
        assert partition.withheld

    def test_parts_overrunning_the_whole_are_reported_as_a_capture_fault(self):
        """The parts nest inside the whole structurally, so an overrun is not a small remainder.

        It means a span reached a component bucket from outside the turn roots, and a
        clamped 0.0 there would hide the fault behind a plausible-looking partition.
        """
        partition = decompose_total_ms(LatencyMetrics(total_ms=100.0, llm_ms=600.0, tool_ms=250.0))

        assert partition.orchestration_ms is None
        assert WITHHELD_PARTS_EXCEED_WHOLE in partition.withheld
        assert "750ms" in partition.withheld

    def test_a_tiny_overrun_is_reported_at_its_real_size_not_rounded_to_nothing(self):
        """The threshold is a nanosecond, so a fixed-decimal format prints "by 0.0ms".

        A refusal whose own sentence says nothing happened is worse than no sentence.
        """
        partition = decompose_total_ms(LatencyMetrics(total_ms=100.0, llm_ms=100.0, tool_ms=0.002))

        assert "0.002ms" in partition.withheld

    def test_the_tolerance_is_pinned_at_a_nanosecond(self):
        """Pinned absolutely, because the boundary tests are expressed relative to it.

        Those stay green through a change of value, so nothing else would notice a
        tolerance widened back to a millisecond — which would let a real overlap through
        as a plausible-looking split.
        """
        assert PARTITION_TOLERANCE_MS == 1e-6

    def test_float_noise_below_the_tolerance_clamps_to_zero_rather_than_withholding(self):
        """Summing the same milliseconds in two groupings can land a hair below zero.

        That is an exact partition seen through binary floating point, not an overrun,
        and refusing it would withhold the commonest honest case: a turn spent entirely
        inside its model and tool calls.
        """
        overrun = PARTITION_TOLERANCE_MS / 2
        partition = decompose_total_ms(LatencyMetrics(total_ms=100.0, llm_ms=100.0, tool_ms=overrun))

        assert partition.orchestration_ms == 0.0
        assert partition.withheld is None

    def test_a_record_cannot_publish_both_a_split_and_a_refusal(self):
        with pytest.raises(ValidationError, match="exactly one"):
            LatencyPartition(total_ms=10.0, llm_ms=1.0, tool_ms=1.0, orchestration_ms=8.0, withheld="because")

    def test_a_record_cannot_half_publish_a_split(self):
        """A total beside a refusal leaves a reader no way to tell which half to believe."""
        with pytest.raises(ValidationError, match="exactly one"):
            LatencyPartition(total_ms=10.0, withheld="because")

    def test_the_remainder_is_declared_a_component_of_the_total(self):
        """`contained_by` is what earns the subtraction the analysis lens otherwise refuses.

        Without the declaration the remainder is just another measure sharing a unit with
        `total_ms`, and the lens correctly declines to difference them.
        """
        assert METRIC_DESCRIPTORS["orchestration_ms"].contained_by == "total_ms"
        assert METRIC_DESCRIPTORS["orchestration_ms"].unit == METRIC_DESCRIPTORS["total_ms"].unit


class TestTheFrontierWithholdsASubstitutedContestantsCost:
    """The frontier RANKS on cost, so a substituted result must not contribute one.

    Its inner-agent dollars and external credits were never spent, so letting it in
    would rank a seeded contestant cheapest on a difference in APPARATUS rather than in
    configuration — the exact defect the code comment at the call site cites.
    """

    def test_a_substituted_result_contributes_no_production_cost(self):
        run = _fr_run()
        results = [
            _fr_result(
                run,
                model="m1",
                variant_key="vk-a",
                test_case_id="tc1",
                roles={"candidate": 0.30, "inner_agent": 0.10},
                substituted_delivery=True,
            )
        ]

        point = compute_frontier([run], results, archived_run_ids=None).subjects[0].points[0]

        # Not 0.40 — withheld. A result with nothing to contribute is ABSENT from the
        # mean rather than entering it as a zero, so the figure is unmeasured, not cheap.
        assert point.production_replicating_cost is None

    def test_an_unsubstituted_result_still_contributes(self):
        """The discriminator: withholding everything would be as wrong as withholding nothing."""
        run = _fr_run()
        results = [
            _fr_result(
                run,
                model="m1",
                variant_key="vk-a",
                test_case_id="tc1",
                roles={"candidate": 0.30, "inner_agent": 0.10},
            )
        ]

        point = compute_frontier([run], results, archived_run_ids=None).subjects[0].points[0]

        assert point.production_replicating_cost == pytest.approx(0.40)


class TestCompletenessDisclosure:
    """A run that came up short must carry one sentence saying so.

    The defect is a run reading ``completed`` with nine of twelve cells: its
    pass^k is computed over a denominator its siblings do not share, and every
    surface renders the two side by side as though they were the same
    measurement.
    """

    def test_a_complete_run_discloses_nothing(self):
        """The disclosure is a warning, so a whole matrix must not acquire one."""
        record = RunCompleteness(
            expected_cells=15, produced_cells=15, persisted_cells=15, infra_excluded_cells=0, counted_from="run_loop"
        )

        assert completeness_disclosure(record) is None

    def test_a_run_with_no_record_discloses_nothing(self):
        """Absence is not evidence of a shortfall.

        A run that has not gone terminal carries None, and so does one whose
        record's write was refused — warning on either would put a degradation
        notice on runs that were never measured short.
        """
        assert completeness_disclosure(None) is None

    def test_a_short_run_names_both_sides_of_the_denominator(self):
        record = RunCompleteness(
            expected_cells=15, produced_cells=15, persisted_cells=13, infra_excluded_cells=0, counted_from="run_loop"
        )

        disclosure = completeness_disclosure(record)

        assert disclosure is not None
        assert DEGRADED_RUN_CLAUSE in disclosure
        assert "13 of 15" in disclosure, "a shortfall stated without its denominator is not a shortfall"

    def test_each_cause_of_a_shortfall_is_named_apart(self):
        """Three different faults, three different fixes — collapsing them loses the fix."""
        record = RunCompleteness(
            expected_cells=10, produced_cells=9, persisted_cells=8, infra_excluded_cells=2, counted_from="run_loop"
        )

        disclosure = completeness_disclosure(record)

        assert disclosure is not None
        assert "6 of 10" in disclosure  # 8 persisted - 2 excluded
        assert "1 never ran" in disclosure
        assert "1 ran but the write was lost" in disclosure
        assert "2 excluded as a harness failure" in disclosure

    def test_a_run_of_failing_candidates_is_not_degraded(self):
        """A candidate that failed is a measurement, and the run is comparable.

        Disclosing here would train a reader to ignore the line on exactly the
        runs a bake-off exists to produce.
        """
        record = RunCompleteness(
            expected_cells=6, produced_cells=6, persisted_cells=6, infra_excluded_cells=0, counted_from="run_loop"
        )

        assert completeness_disclosure(record) is None


def _short_run(status, completeness, *, subject_id="ent-maple", model="sonnet", n_results=2, **run_overrides):
    """Build one run of a given status and completeness, plus the results it measured.

    Args:
        status: The run's terminal status.
        completeness: Its :class:`RunCompleteness` record, or ``None`` for a run
            carrying none.
        subject_id: Subject id for the subject snapshot — the coordinate every
            aggregator partitions on.
        model: Candidate model; also the frontier/history grouping fallback, so two
            runs sharing it become one contestant.
        n_results: How many results to emit, each on its own test case so the
            per-case aggregations have something to average.
        **run_overrides: Passed through to ``make_eval_run``.

    Returns:
        ``(run, results)``.
    """
    snapshot = make_subject(subject_id, "Maple")
    run = make_eval_run(
        subject_snapshot=snapshot,
        status=status,
        completeness=completeness,
        candidate_model=model,
        test_case_ids=[f"tc-{i}" for i in range(1, n_results + 1)],
        **run_overrides,
    )
    results = [
        make_eval_result(
            eval_run_id=run.id,
            scope_id=run.scope_id,
            test_case_id=f"tc-{i}",
            model=model,
            # A production-replicating cost, so the frontier can name a verdict at all —
            # a pick cannot be called cheapest on a cost nobody observed.
            usage=[RoleUsage(role="candidate", cost_usd=0.01)],
        )
        for i in range(1, n_results + 1)
    ]
    return run, results


#: The three shapes a short run reaches an aggregator in. The third is the one a
#: status filter cannot catch and the reason this whole family is not a status
#: question: a run can read ``completed`` and measure 2 of its 3
#: cells, because one was excluded as a harness failure — so it is in every quality
#: surface's DEFAULT cohort, today, with nothing disclosed.
SHORT_RUN_SHAPES = [
    (
        "budget_stopped",
        RunCompleteness(
            expected_cells=4, produced_cells=2, persisted_cells=2, infra_excluded_cells=0, counted_from="run_loop"
        ),
    ),
    (
        "cancelled",
        RunCompleteness(
            expected_cells=2, produced_cells=1, persisted_cells=1, infra_excluded_cells=0, counted_from="run_loop"
        ),
    ),
    (
        "completed",
        RunCompleteness(
            expected_cells=3, produced_cells=3, persisted_cells=3, infra_excluded_cells=1, counted_from="run_loop"
        ),
    ),
]
SHORT_RUN_IDS = ["budget_stopped", "cancelled", "completed-but-infra-excluded"]

COMPLETE = RunCompleteness(
    expected_cells=2, produced_cells=2, persisted_cells=2, infra_excluded_cells=0, counted_from="run_loop"
)


class TestAggregatorsDiscloseShortRuns:
    """The pooling surfaces carry the DEGRADED sentence the run-scoped ones already do.

    The defect: ``completeness_disclosure`` reached ``get_run``, ``run_summary``,
    ``compare_runs`` and the analysis bundle — every RUN-scoped surface — and none
    of ``frontier``, ``pivot`` or ``history``. A run that measured 2 of its 4 cells
    was pooled into a pass^k, a cell mean and a time series with nothing saying so,
    which is a rate computed over two different populations rendered as one number.

    **The disclosure is chosen over refusal deliberately.** A truncated run's cells
    are real measurements of the same contestant, so dropping them is the silent
    removal this tier already refuses for a dominated point — and it could not be
    made complete anyway, since the third shape below is ``completed``.

    **The predicate is the completeness record, never the status.** Parametrized
    over all three shapes for exactly that reason: a fix that keyed on status would
    pass the first two and leave the live defect standing.
    """

    @pytest.mark.parametrize(("status", "completeness"), SHORT_RUN_SHAPES, ids=SHORT_RUN_IDS)
    def test_the_projection_carries_the_sentence_for_a_short_run(self, status, completeness):
        run, results = _short_run(status, completeness)

        projection = project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None)

        assert run.id in projection.completeness_disclosures
        assert DEGRADED_RUN_CLAUSE in projection.completeness_disclosures[run.id]

    @pytest.mark.parametrize(("status", "completeness"), SHORT_RUN_SHAPES, ids=SHORT_RUN_IDS)
    def test_the_pivot_discloses_the_short_runs_behind_its_cells(self, status, completeness):
        """Every cell pools the shorter denominator, so the table states it once."""
        short, short_results = _short_run(status, completeness)
        whole, whole_results = _short_run("completed", COMPLETE, model="opus")
        projection = project_score_records(
            [short, whole], [*short_results, *whole_results], profile=_JUDGED_HOST, archived_run_ids=None
        )

        table = compute_pivot(
            projection.records,
            row_factor="test_case_id",
            column_factor="model",
            metric=METRIC_COMPOSITE,
            completeness_disclosures=projection.completeness_disclosures,
            profile=_JUDGED_HOST,
        )

        assert short.id in table.completeness_disclosures
        assert whole.id not in table.completeness_disclosures, (
            "a run that delivered its matrix must not be warned about"
        )
        assert DEGRADED_RUN_CLAUSE in table.completeness_disclosures[short.id]
        assert table.n_degraded_observations == 2, "the weight of the caveat is how many rows came from the short run"

    @pytest.mark.parametrize(("status", "completeness"), SHORT_RUN_SHAPES, ids=SHORT_RUN_IDS)
    def test_the_frontier_marks_the_contestant_it_ranked_on_a_short_run(self, status, completeness):
        """The verdict surface: the pass^k and the cost a pick is named on pooled it."""
        short, short_results = _short_run(status, completeness)
        whole, whole_results = _short_run("completed", COMPLETE, model="opus")

        result = compute_frontier([short, whole], [*short_results, *whole_results], archived_run_ids=None)

        assert short.id in result.completeness_disclosures
        assert whole.id not in result.completeness_disclosures
        assert result.n_degraded_observations == 2
        points = {point.model: point for point in result.subjects[0].points}
        assert short.id in points["sonnet"].completeness_disclosures, "the row that pooled it must carry it"
        assert points["opus"].completeness_disclosures == {}, "a caveat must not spread to a contestant it is not about"

    @pytest.mark.parametrize(("status", "completeness"), SHORT_RUN_SHAPES, ids=SHORT_RUN_IDS)
    def test_the_history_point_carries_the_sentence_beside_its_regression_verdict(self, status, completeness):
        """A point IS one run, and it is handed a verdict against a complete neighbour."""
        whole, whole_results = _short_run("completed", COMPLETE, created_at="2026-08-01T00:00:00Z")
        short, short_results = _short_run(status, completeness, created_at="2026-08-02T00:00:00Z")

        result = compute_history(
            [whole, short], [*whole_results, *short_results], profile=_JUDGED_HOST, archived_run_ids=None
        )

        assert short.id in result.completeness_disclosures
        assert result.n_degraded_observations == 2
        by_run = {point.run_id: point for point in result.series[0].points}
        assert by_run[short.id].completeness_disclosure is not None
        assert DEGRADED_RUN_CLAUSE in by_run[short.id].completeness_disclosure
        assert by_run[whole.id].completeness_disclosure is None
        assert by_run[short.id].regression is not None, (
            "the point still gets its verdict — the mark qualifies it, not replaces it"
        )

    @pytest.mark.parametrize("surface", ["pivot", "frontier", "history"])
    def test_a_corpus_of_whole_runs_discloses_nothing_on_any_surface(self, surface):
        """The note is a warning; a surface that always shows one has trained the reader to skip it."""
        run, results = _short_run("completed", COMPLETE)
        projection = project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None)

        if surface == "pivot":
            answer = compute_pivot(
                projection.records,
                row_factor="test_case_id",
                column_factor="model",
                metric=METRIC_COMPOSITE,
                completeness_disclosures=projection.completeness_disclosures,
                profile=_JUDGED_HOST,
            )
        elif surface == "frontier":
            answer = compute_frontier([run], results, archived_run_ids=None)
        else:
            answer = compute_history([run], results, profile=_JUDGED_HOST, archived_run_ids=None)

        assert answer.completeness_disclosures == {}
        assert answer.n_degraded_observations == 0

    @pytest.mark.parametrize("surface", ["pivot", "frontier", "history"])
    def test_a_run_with_no_completeness_record_is_not_warned_about(self, surface):
        """Absence of a record is not evidence of a shortfall — the rule `completeness_disclosure` sets."""
        run, results = _short_run("completed", None)
        projection = project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None)

        if surface == "pivot":
            answer = compute_pivot(
                projection.records,
                row_factor="test_case_id",
                column_factor="model",
                metric=METRIC_COMPOSITE,
                completeness_disclosures=projection.completeness_disclosures,
                profile=_JUDGED_HOST,
            )
        elif surface == "frontier":
            answer = compute_frontier([run], results, archived_run_ids=None)
        else:
            answer = compute_history([run], results, profile=_JUDGED_HOST, archived_run_ids=None)

        assert answer.completeness_disclosures == {}

    def test_the_frontier_verdict_carries_the_shortfall_of_the_run_it_was_picked_on(self):
        """A recommendation drawn from a short pool has to say so where the recommendation is."""
        short, short_results = _short_run("budget_stopped", SHORT_RUN_SHAPES[0][1])

        result = compute_frontier([short], short_results, bar=0.0, archived_run_ids=None)

        verdict = result.subjects[0].verdict
        assert verdict is not None
        assert short.id in verdict.completeness_disclosures

    def test_a_short_run_whose_rows_the_filters_removed_is_not_warned_about(self):
        """A caveat about a run the table never averaged is one the reader cannot check."""
        short, short_results = _short_run("cancelled", SHORT_RUN_SHAPES[1][1], subject_id="ent-other")
        whole, whole_results = _short_run("completed", COMPLETE)
        projection = project_score_records(
            [short, whole], [*short_results, *whole_results], profile=_JUDGED_HOST, archived_run_ids=None
        )

        table = compute_pivot(
            projection.records,
            row_factor="test_case_id",
            column_factor="model",
            metric=METRIC_COMPOSITE,
            filters={"subject_id": "ent-maple"},
            completeness_disclosures=projection.completeness_disclosures,
            profile=_JUDGED_HOST,
        )

        assert table.completeness_disclosures == {}
        assert table.n_degraded_observations == 0


class TestSignificanceRead:
    """The three-way read every surface uses, and its agreement with a browser kit.

    The same distinction can be enforced in a browser renderer. These tests pin the Python half
    against the same predicate, because a rule that lives in two languages
    diverges the first time only one of them is edited.
    """

    def test_a_flag_with_no_statistic_is_not_tested(self):
        """A verdict nobody backed with a number is not a measured result.

        This is the dangerous half: `significant=False` with nothing behind it
        asserts that a test ran and came back negative, which a reader takes as
        a measured null the campaign never measured.
        """
        from threetears.evals.analysis.reporting import NOT_TESTED_LABEL, significance_read

        assert significance_read(significant=False) == NOT_TESTED_LABEL
        assert significance_read(significant=True) == NOT_TESTED_LABEL

    def test_not_significant_and_not_tested_are_different_answers(self):
        """The whole point of the three-way read: the two must never collapse."""
        from threetears.evals.analysis.reporting import NOT_SIGNIFICANT_LABEL, NOT_TESTED_LABEL, significance_read

        backed = significance_read(significant=False, p=0.42, effect=0.1)
        unbacked = significance_read(significant=False)

        assert backed == NOT_SIGNIFICANT_LABEL
        assert unbacked == NOT_TESTED_LABEL
        assert backed != unbacked

    def test_a_sample_size_alone_is_not_a_test(self):
        """`n` says how much data there was, never whether anything cleared a bar.

        Pinned through the formatter, because `n` is the one statistic the cell
        renders without it counting toward the predicate — the exact place the
        two could silently disagree.
        """
        from threetears.evals.analysis.reporting import NOT_TESTED_LABEL, format_significance

        cell = format_significance(significant=False, paired=True, n=12)

        assert cell.startswith(NOT_TESTED_LABEL)
        assert "n=12" in cell

    def test_either_statistic_alone_counts_as_tested(self):
        """The predicate is `p or effect size`, matching the kit — not both."""
        from threetears.evals.analysis.reporting import SIGNIFICANT_LABEL, significance_read

        assert significance_read(significant=True, p=0.01) == SIGNIFICANT_LABEL
        assert significance_read(significant=True, effect=1.4) == SIGNIFICANT_LABEL

    def test_a_null_verdict_with_statistics_is_still_not_tested(self):
        """No verdict is no verdict, whatever else the row happens to carry."""
        from threetears.evals.analysis.reporting import NOT_TESTED_LABEL, significance_read

        assert significance_read(significant=None, p=0.01, effect=1.4) == NOT_TESTED_LABEL

    def test_the_labels_are_spelled_out_never_abbreviated(self):
        """ "n.s." was read as nanoseconds by an operator in a report full of `*_ms`."""
        from threetears.evals.analysis.reporting import NOT_SIGNIFICANT_LABEL, NOT_TESTED_LABEL, SIGNIFICANT_LABEL

        assert (SIGNIFICANT_LABEL, NOT_SIGNIFICANT_LABEL, NOT_TESTED_LABEL) == (
            "significant",
            "not significant",
            "not tested",
        )


class TestSignificanceFormatting:
    """A verdict never travels without the statistics behind it, or its test."""

    def test_a_verdict_carries_the_statistics_it_rests_on(self):
        from threetears.evals.analysis.reporting import format_significance

        cell = format_significance(significant=True, paired=True, p=0.0123, effect=1.42, n=8)

        assert cell.startswith("significant (")
        assert "p=0.0123" in cell
        assert "d_z=1.42" in cell
        assert "n=8" in cell

    def test_the_effect_size_is_named_for_the_test_that_produced_it(self):
        """An unpaired Cohen's d printed as `d_z` claims a within-case comparison that never ran.

        The two labels come from the module's own constants rather than being
        spelled out here, so renaming either one cannot leave this passing
        against a name no surface prints.
        """
        from threetears.evals.analysis.reporting import PAIRED_EFFECT_LABEL, UNPAIRED_EFFECT_LABEL, format_significance

        assert PAIRED_EFFECT_LABEL != UNPAIRED_EFFECT_LABEL

        paired = format_significance(significant=True, paired=True, p=0.01, effect=4.02)
        unpaired = format_significance(significant=True, paired=False, p=0.01, effect=4.02)

        assert f"{PAIRED_EFFECT_LABEL}=4.02" in paired
        assert f"{UNPAIRED_EFFECT_LABEL}=4.02" in unpaired
        assert f"{PAIRED_EFFECT_LABEL}=" not in unpaired

    def test_the_single_renderer_serves_the_compiled_chart_too(self):
        """The rule had three server-side homes and drifted in two of them.

        Pinned through the ``delta_table`` chart's own read so that reintroducing
        a private copy there fails here rather than silently answering
        differently from the compare table about the same row.
        """
        from threetears.evals.analysis.reporting import format_significance
        from threetears.evals.vega.compiler import compile_chart
        from threetears.evals.analysis.viz.payloads import DeltaRow, DeltaTablePayload

        def effect_reads(row: DeltaRow) -> set[str]:
            """Every place the compiled chart states the row's effect: its values table, its intent data and its marks."""
            chart = compile_chart("delta_table", DeltaTablePayload(rows=[row]).model_dump(mode="json"))
            return {
                chart.rows[0]["effect"],
                *(datum["effect"] for datum in chart.intent.data),
                *(datum["effect"] for datum in chart.spec["data"]["values"]),
            }

        row = DeltaRow(metric="cost_usd", a=0.011, b=0.019, d_z=1.2, p=0.004, n=24, significant=True, paired=True)

        assert effect_reads(row) == {format_significance(significant=True, paired=True, p=0.004, effect=1.2, n=24)}

        # The row's own pairing, not a constant the compiler supplies: the same
        # row with the field left unstated must reach the renderer as unpaired.
        unstated = DeltaRow(metric="cost_usd", a=0.011, b=0.019, d_z=1.2, p=0.004, n=24, significant=True)

        assert effect_reads(unstated) == {
            format_significance(significant=True, paired=False, p=0.004, effect=1.2, n=24)
        }

    def test_the_disclosure_names_the_paired_test_and_its_threshold(self):
        from threetears.evals.analysis.reporting import significance_disclosure
        from threetears.evals.analysis.stats import PAIRED_TEST_NAME

        sentence = significance_disclosure(paired=True)

        assert PAIRED_TEST_NAME in sentence
        assert "Descriptive only" in sentence

    def test_the_disclosure_names_the_unpaired_test_when_samples_were_not_paired(self):
        """Which test ran is most of what a small arm's verdict rests on."""
        from threetears.evals.analysis.reporting import significance_disclosure
        from threetears.evals.analysis.stats import PAIRED_TEST_NAME, UNPAIRED_TEST_NAME

        sentence = significance_disclosure(paired=False)

        assert UNPAIRED_TEST_NAME in sentence
        assert PAIRED_TEST_NAME not in sentence

    def test_the_disclosure_promises_no_alerting(self):
        """Reporting presents and compares; it never routes a verdict anywhere."""
        from threetears.evals.analysis.reporting import significance_disclosure

        assert "no alerting" in significance_disclosure(paired=True)


class TestCrossSubjectDisclosure:
    """Composites are comparable within a subject and never across one."""

    def test_the_same_subject_needs_no_disclosure(self):
        from threetears.evals.analysis.reporting import cross_subject_disclosure

        assert cross_subject_disclosure("ent-maple", "ent-maple") is None

    def test_two_subjects_are_named_and_the_delta_is_withheld(self):
        from threetears.evals.analysis.reporting import cross_subject_disclosure

        note = cross_subject_disclosure("ent-maple", "ent-bea")

        assert note is not None
        assert "ent-maple" in note and "ent-bea" in note
        assert "withheld" in note

    def test_the_disclosure_names_no_host(self):
        """The whole sentence, not the clause the test above happens to quote.

        This module is declared shared contract, and this is the operator string that made
        it not be for as long as it named a host noun. The canary cannot see
        it — the sentence is built in a function body, which it deliberately does not scan —
        so an assertion on a substring leaves two thirds of the sentence free to name a host
        again with every gate green.

        Checked against the canary's own ``HOST_NOUNS`` rather than against one hand-picked noun:
        this test is named for the whole class, and a hand-typed noun list is what made two
        earlier claims in this branch false. **Host-neutral subject keys**, because the
        sentence interpolates them and a fixture key could carry a host noun.
        """
        from threetears.evals.analysis.reporting import cross_subject_disclosure

        note = cross_subject_disclosure("subj-a", "subj-b")

        assert note is not None
        assert not _HOST_NOUN_IN_PROSE.findall(note)

    def test_there_is_no_undecidable_case_because_a_blank_subject_is_not_one(self):
        """Comparing two absences for equality manufactures the sameness claim this refuses.

        That was the argument for an undecidable branch, and it was right while a run could carry
        a blank subject key. It cannot: ``SubjectSnapshot`` refuses one, so the pair this branch
        answered about is not constructible and the branch went with the state. What remains is
        the rule that actually bites — two DIFFERENT subjects are not comparable — and it is
        covered above.
        """
        from threetears.evals.analysis.reporting import cross_subject_disclosure

        assert cross_subject_disclosure("", "") is None
        assert cross_subject_disclosure("ent-maple", "ent-bea") is not None


class TestARefusalNamesTheMeasureTheCallerTyped:
    """Resolution maps a catalog name onto a row name; the refusal must not quote the map's output.

    `mean_total_ms` is in the alias table because `history` needs it, and `total_ms` is
    not a measure `pivot` accepts — so refusing from the resolved name told an operator
    "unknown metric 'total_ms'" about a string they never typed, and about the one name
    the docs say that surface does not take. `resolve_measure_name`'s own docstring sets
    the rule this restores: an unknown name reaches its surface's refusal rather than
    being rewritten into something else.
    """

    def test_pivot_quotes_the_catalog_name_the_caller_sent(self):
        records = [_record(model="m1", case="tc-1", value=0.8)]

        with pytest.raises(PivotError) as excinfo:
            compute_pivot(
                records, row_factor="model", column_factor="template_id", metric="mean_total_ms", profile=_JUDGED_HOST
            )

        assert "'mean_total_ms'" in str(excinfo.value), str(excinfo.value)
        assert "'total_ms'" not in str(excinfo.value).replace("'mean_total_ms'", ""), (
            "the refusal quoted the resolved name, which the caller never typed"
        )

    def test_the_pivot_refusal_still_names_what_is_accepted(self):
        """Naming the caller's input must not cost them the vocabulary."""
        records = [_record(model="m1", case="tc-1", value=0.8)]

        with pytest.raises(PivotError) as excinfo:
            compute_pivot(
                records, row_factor="model", column_factor="template_id", metric="mean_total_ms", profile=_JUDGED_HOST
            )

        assert "mean_composite" in str(excinfo.value)

    def test_an_aliased_score_takes_the_same_history_arm_as_the_row_name(self):
        """`mean_score` and `score` are one refusal; only the quoted spelling differs.

        The `score` arm exists because a series carries one value per run and a raw
        judge score is per rubric DIMENSION. Reading the catalog must not route an
        operator to the generic "expected one of" answer instead of that explanation.
        """
        run, results = _hist_run(cases=["c1"], score=5, created_at="2026-07-01T00:00:00Z")

        def refusal(metric: str) -> str:
            with pytest.raises(HistoryError) as excinfo:
                compute_history([run], results, metric=metric, profile=_JUDGED_HOST, archived_run_ids=None)
            return str(excinfo.value)

        aliased = refusal("mean_score")
        direct = refusal("score")

        assert "DIMENSION" in aliased, aliased
        assert "'mean_score'" in aliased
        assert "DIMENSION" in direct


class TestGoalStatePivots:
    """A goal_state cell is one check's pass rate only when goal_check pins it — and says so when it does not."""

    @staticmethod
    def goal_state_records():
        from threetears.evals.schema.models import GoalStateOutcome

        run, _ = _run_with_results()
        results = [
            make_eval_result(
                eval_run_id=run.id,
                scope_id=run.scope_id,
                goal_state_outcomes=[
                    GoalStateOutcome(expression="check_a", passed=True),
                    GoalStateOutcome(expression="check_b", passed=False),
                ],
            )
        ]
        return project_score_records([run], results, profile=_JUDGED_HOST, archived_run_ids=None).records

    def test_with_goal_check_on_an_axis_each_cell_is_one_checks_rate(self):
        table = compute_pivot(
            self.goal_state_records(),
            row_factor="goal_check",
            column_factor="model",
            metric=METRIC_GOAL_STATE,
            profile=_JUDGED_HOST,
        )

        assert {(cell.row, cell.value) for cell in table.cells} == {("check_a", 1.0), ("check_b", 0.0)}
        assert "POOLED" not in table.formula

    def test_a_check_expression_passed_as_the_metric_is_pointed_at_goal_check(self):
        with pytest.raises(PivotError, match="'check_a' is a goal-state check, not a measure — aggregate 'goal_state'"):
            compute_pivot(
                self.goal_state_records(),
                row_factor="template_id",
                column_factor="model",
                metric="check_a",
                profile=_JUDGED_HOST,
            )

    def test_without_it_the_cell_pools_every_check_and_says_so(self):
        table = compute_pivot(
            self.goal_state_records(),
            row_factor="template_id",
            column_factor="model",
            metric=METRIC_GOAL_STATE,
            profile=_JUDGED_HOST,
        )

        [cell] = table.cells
        assert cell.value == 0.5
        assert "POOLED ACROSS EVERY GOAL-STATE CHECK (put 'goal_check' on an axis for one)" in table.formula
        assert "(result x check) rows" in table.formula


def test_the_catalog_name_of_a_goal_state_cell_pivots_as_its_rows():
    """list_metrics publishes 'goal_state_pass_rate'; the pivot accepts it verbatim and reads the goal_state rows."""
    records = TestGoalStatePivots.goal_state_records()

    by_alias = compute_pivot(
        records, row_factor="goal_check", column_factor="model", metric="goal_state_pass_rate", profile=_JUDGED_HOST
    )
    by_row_name = compute_pivot(
        records, row_factor="goal_check", column_factor="model", metric=METRIC_GOAL_STATE, profile=_JUDGED_HOST
    )

    assert {(c.row, c.value) for c in by_alias.cells} == {(c.row, c.value) for c in by_row_name.cells}

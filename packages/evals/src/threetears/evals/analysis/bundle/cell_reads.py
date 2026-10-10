"""Per-cell reads: which results a cell holds, which of them count, and every judged and measured value in it.

The populations the bars, comparisons and decision surface share — results the harness did not fault
(:func:`_non_faulted`), candidates that failed, results that took no turn — and the per-cell summaries
:func:`_judged_measures`, :func:`_cell_measures` and :func:`_cell_strata` built over them.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

from threetears.evals.analysis.agreement import (
    JudgeKey,
    judge_key,
    tier_for_judges,
)
from threetears.evals.kernel.evidence_tiers import JudgeEvidenceTier
from threetears.evals.analysis.arms import surface_order
from threetears.evals.analysis.cells import Cell
from threetears.evals.analysis.reporting import (
    METRIC_OUTCOME,
    METRIC_SCORE,
    METRIC_TRANSCRIPT,
    ScoreRecord,
)
from threetears.evals.analysis.stats import (
    clustered_standard_error,
    small_sample_case_means,
)
from threetears.evals.kernel.analysis_measures import MeasureCollection
from threetears.evals.kernel.declaration import (
    BarName,
    CampaignDesign,
)
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.metrics import describe_rubric_dim
from threetears.evals.schema.models import EvalResult
from threetears.evals.kernel.result_condition import (
    JUDGE_CANNOT_TELL_OUTCOME,
    ResultOutcome,
    classify_result,
    delivered_a_turn,
    harness_faulted,
)
from threetears.evals.kernel.surface import (
    CellFacts,
    JudgedReading,
    StratumFacts,
)
from threetears.evals.analysis.bundle.schema import (
    JudgedArm,
    JudgedMeasure,
)
from threetears.evals.analysis.bundle.measures import _measure_collection


#: A cell's two coordinates, ``(variant_key, apparatus_class_id)`` — how the per-arm surfaces key it.
_CellKey = tuple[str, str]


def _cell_collections(cell: CellFacts) -> Iterator[MeasureCollection]:
    """A cell's measure collection, then each of its strata's.

    Walked wherever a reader needs every measure NAME a cell holds: a stratum is a smaller set of results,
    so it can carry a name the pooled cell does not — two carriers sharing a leaf name are refused pooling
    over the cell and may each be alone in a stratum.
    """
    yield cell.measures
    for stratum in cell.strata:
        yield stratum.measures


def _results_by_cell(cells: list[Cell], results: list[EvalResult]) -> dict[_CellKey, list[EvalResult]]:
    """Each cell's results, keyed by its two coordinates — the arms every per-arm surface here reads.

    Read off the cell algebra rather than re-grouped, so a per-arm number can only ever describe a
    set of observations the bundle already calls one cell.

    Args:
        cells: The pooled cells.
        results: Every resolved member result.

    Returns:
        ``{(variant_key, apparatus_class_id): [result, ...]}`` in coordinate order, each list in
        result-id order.
    """
    by_id = {result.id: result for result in results}
    return {
        (cell.variant_key, cell.apparatus_class_id): sorted(
            (by_id[observation_id] for observation_id in cell.observation_ids if observation_id in by_id),
            key=lambda result: result.id,
        )
        for cell in sorted(cells, key=lambda c: (c.variant_key, c.apparatus_class_id))
    }


def _member_run_ids(members: list[EvalResult]) -> list[str]:
    """The runs a cell's results came from, sorted — what a per-arm surface names as citable.

    Read off the cell's own results rather than joined back through a run's configuration,
    because the cell IS its observations: whichever levers a host records, and whichever of them
    the variant index describes, the runs that produced a cell's observations are the runs it
    pooled. Every per-arm surface takes its ``run_ids`` from here, so a judged score and a bar
    verdict on one cell cannot name different runs.

    Args:
        members: One cell's results, from :func:`_results_by_cell`.

    Returns:
        The distinct ``eval_run_id`` values, sorted.
    """
    return sorted({result.eval_run_id for result in members})


def _judged_values(records: list[ScoreRecord]) -> Iterator[tuple[str, ScoreRecord]]:
    """Yield ``(dimension, record)`` for every projected row carrying a judged score.

    Two row shapes carry one: a ``score`` row, whose dimension is its ``rubric_dim`` coordinate, and
    a dual-score axis row, whose metric IS the reserved axis id. The dimensionless ``score`` row a
    result with no judged dimension emits is not one — it records that the cell was attempted.
    """
    for record in records:
        if record.value is None:
            continue
        if record.metric == METRIC_SCORE and record.rubric_dim:
            yield record.rubric_dim, record
        elif record.metric in (METRIC_TRANSCRIPT, METRIC_OUTCOME):
            yield record.metric, record


def _judged_rows(records: list[ScoreRecord]) -> Iterator[tuple[str, ScoreRecord]]:
    """:func:`_judged_values`, plus the null rows the judge answered it could not tell on or the harness faulted.

    What a judged measure COUNTS rather than means: those rows carry no value, so they never reach a
    mean, and are yielded here only so the arm can say how many left it and why — a faulted
    observation's score row is null since the projection reads ``counted_rubric_scores``.
    """
    yield from _judged_values(records)
    for record in records:
        if record.value is None and record.outcome in (JUDGE_CANNOT_TELL_OUTCOME, ResultOutcome.INFRA_EXCLUDE.value):
            if record.metric == METRIC_SCORE and record.rubric_dim:
                yield record.rubric_dim, record
            elif record.metric in (METRIC_TRANSCRIPT, METRIC_OUTCOME):
                yield record.metric, record


def _judged_keys(results: list[EvalResult]) -> set[JudgeKey]:
    """Every judge (:class:`~threetears.evals.analysis.agreement.JudgeKey`) a judged score among ``results`` was given by."""
    return {
        key
        for result in results
        for score in (*result.rubric_scores, result.transcript_score, result.outcome_score)
        if score is not None and (key := judge_key(result, score.dim)) is not None
    }


def _judged_measures(
    records: list[ScoreRecord],
    results_by_cell: dict[_CellKey, list[EvalResult]],
    design: CampaignDesign | None,
    *,
    tiers: list[JudgeEvidenceTier],
) -> list[JudgedMeasure]:
    """Summarise every judged dimension per cell, from the score projection assembly already holds.

    The projection is the source because it is where the dimension's NAME survives: the measure
    walk meets a judged score as a bare ``score`` leaf on a ``RubricScore`` and cannot tell one
    dimension from another. A score on an observation the harness faulted is counted beside the mean
    and left out of it, because a judge reading a broken transcript is not measuring the candidate.

    Args:
        records: The assembly's score projection rows.
        results_by_cell: Each cell's results, from :func:`_results_by_cell`.
        design: The campaign's declaration, for the bar naming a dimension.
        tiers: The judges' evidence tiers; each arm carries the weakest among the judges that served its
            counted scores (:func:`~threetears.evals.analysis.agreement.tier_for_judges`).

    Returns:
        One entry per dimension scored anywhere, sorted by name.
    """
    result_by_id = {result.id: result for members in results_by_cell.values() for result in members}
    cell_of_result = {result.id: key for key, members in results_by_cell.items() for result in members}
    boundary = _boundary_dimensions(result_by_id.values())
    scored: dict[str, dict[_CellKey, list[ScoreRecord]]] = {}
    for dimension, record in _judged_rows(records):
        key = cell_of_result.get(record.result_id)
        if key is not None:
            scored.setdefault(dimension, {}).setdefault(key, []).append(record)
    bars = {bar.measure_id: bar.threshold for bar in (design.bars if design else [])}
    measures = []
    for dimension in sorted(scored):
        scales = {
            row.rubric_scale for rows in scored[dimension].values() for row in rows if row.rubric_scale is not None
        }
        if len(scales) > 1:
            raise ValueError(f"rubric dimension {dimension!r} was judged on more than one scale: {sorted(scales)}")
        # The dual-score axes carry no `rubric_scale` (they are not template dims) and are 1-5.
        descriptor = describe_rubric_dim(dimension, scale=next(iter(scales), "ordinal"))
        arms = []
        for key in sorted(scored[dimension]):
            rows = scored[dimension][key]
            cannot_tell = [row for row in rows if row.outcome == JUDGE_CANNOT_TELL_OUTCOME]
            counted = [
                row for row in rows if row.outcome not in (ResultOutcome.INFRA_EXCLUDE.value, JUDGE_CANNOT_TELL_OUTCOME)
            ]
            valued = [row for row in counted if row.value is not None]
            values = [float(row.value) for row in valued if row.value is not None]
            value_cases = [row.test_case_id for row in valued]
            # The whole judge behind each counted score — dimension, scale, served model and config — so an arm
            # can only carry a tier measured for the very judges that scored it.
            served = [
                judge
                for row in counted
                if row.value is not None and (judge := judge_key(result_by_id[row.result_id], dimension)) is not None
            ]
            arms.append(
                JudgedArm(
                    variant_key=key[0],
                    apparatus_class_id=key[1],
                    run_ids=_member_run_ids(results_by_cell[key]),
                    n=len(values),
                    n_independent=len(set(value_cases)),
                    n_infra_excluded=len(rows) - len(counted) - len(cannot_tell),
                    n_cannot_tell=len(cannot_tell),
                    mean=sum(values) / len(values) if values else None,
                    sem=clustered_standard_error(values, value_cases) if values else None,
                    case_means=small_sample_case_means(values, value_cases),
                    evidence_tier=tier_for_judges(tiers, served),
                )
            )
        measures.append(
            JudgedMeasure(
                name=dimension,
                family=str(descriptor.family),
                value_range=descriptor.value_range,
                scale=descriptor.scale,
                higher_is_better=bool(descriptor.higher_is_better),
                axis="boundary" if dimension in boundary else "capability",
                bar_threshold=bars.get(dimension),
                arms=arms,
            )
        )
    return measures


def _boundary_dimensions(results: Iterable[EvalResult]) -> set[str]:
    """The judged dimensions any of ``results`` was scored on as a boundary dimension — the judged guardrails.

    Any one boundary score makes the dimension a guardrail: a dimension stamped both ways was declared a
    boundary somewhere, and reading it as a guardrail errs toward holding it rather than trading it.
    """
    return {score.dim for result in results for score in result.rubric_scores if score.axis == "boundary"}


def _unstamped_dimensions(results: Iterable[EvalResult]) -> list[str]:
    """The judged dimensions carrying a score judged before the rubric axis was stamped, sorted."""
    return sorted({score.dim for result in results for score in result.rubric_scores if score.axis is None})


def _non_faulted(members: list[EvalResult]) -> list[EvalResult]:
    """A cell's results the harness did not fault — the one population every bar is adjudicated over.

    See :class:`BarVerdict` for why. Held as one function so the three kinds of bar name cannot
    each come to their own answer about which observations count.
    """
    return [result for result in members if not harness_faulted(result)]


def _candidate_failed(members: list[EvalResult]) -> int:
    """How many of a cell's results the candidate failed, for any cause — each counted against the arm.

    By :func:`~threetears.evals.kernel.result_condition.classify_result`, the rule every rate and bar
    counts a failure by.
    """
    return sum(1 for result in members if classify_result(result) is ResultOutcome.CANDIDATE_FAIL)


def _took_no_turn(members: list[EvalResult]) -> bool:
    """Whether a cell's results include a failure and no turn — the reading :attr:`CellFacts.all_failed` makes.

    Read off the results, for the lenses that hold them rather than the cell's facts, by the same predicate
    (:func:`~threetears.evals.kernel.result_condition.delivered_a_turn`), so the two cannot disagree.
    """
    return _no_turn(members) > 0 and not any(delivered_a_turn(result) for result in members)


def _no_turn(members: list[EvalResult]) -> int:
    """How many of a cell's failures took no turn — the ones its cost and latency readings left out.

    Exactly the candidate failures outside
    :func:`~threetears.evals.kernel.result_condition.delivered_a_turn`, the predicate the measure walk's
    ``delivered`` population is, so the count beside a cell's cost and latency is the number they left out.
    """
    return sum(
        1
        for result in members
        if classify_result(result) is ResultOutcome.CANDIDATE_FAIL and not delivered_a_turn(result)
    )


def _cannot_tell_on(bar: BarName, counted: list[EvalResult], judged_rows: dict[str, list[ScoreRecord]]) -> int:
    """How many of a cell's counted results the judge could not score on a judged bar's dimension."""
    if bar.kind != "judged":
        return 0
    return sum(
        1
        for result in counted
        for dimension, record in _judged_rows(judged_rows.get(result.id, []))
        if dimension == bar.name and record.outcome == JUDGE_CANNOT_TELL_OUTCOME
    )


def _judged_by_cell(judged_measures: list[JudgedMeasure]) -> dict[_CellKey, list[JudgedReading]]:
    """The judged measures transposed — each cell's readings, one per dimension it was scored on.

    Args:
        judged_measures: Judged measures, from :func:`_judged_measures`, sorted by name.

    Returns:
        Each cell's readings, sorted by dimension, keyed by the cell's coordinates.
    """
    judged_by_cell: dict[_CellKey, list[JudgedReading]] = {}
    # Judged measures are sorted by name, so each cell's readings arrive sorted by dimension.
    for measure in judged_measures:
        for arm in measure.arms:
            judged_by_cell.setdefault((arm.variant_key, arm.apparatus_class_id), []).append(
                JudgedReading(
                    dimension=measure.name,
                    mean=arm.mean,
                    sem=arm.sem,
                    n=arm.n,
                    n_independent=arm.n_independent,
                    case_means=arm.case_means,
                    n_infra_excluded=arm.n_infra_excluded,
                    n_cannot_tell=arm.n_cannot_tell,
                    evidence_tier=arm.evidence_tier,
                )
            )
    return judged_by_cell


#: How a stratum is keyed while the strata are built: a declared stratum, or None for the cases that declare none.
_StratumKey = str | None


def _cell_strata(
    results_by_cell: dict[_CellKey, list[EvalResult]],
    stratum_of_case: dict[str, str | None],
    records: list[ScoreRecord],
    design: CampaignDesign | None,
    *,
    tiers: list[JudgeEvidenceTier],
    profile: HostProfile,
) -> dict[_CellKey, list[StratumFacts]]:
    """Each cell read again per stratum of its cases — the same walk and transposition, over each stratum's results.

    A cell is broken down only when some case of it declares a stratum; one whose cases declare none has no
    entry here, and reads exactly as an unstratified cell does. Within a broken-down cell every result lands
    in one stratum — its case's, or the undeclared one — so the strata partition the cell.

    **The judged readings are scored over each stratum's own slice** by :func:`_judged_measures`, with the
    campaign's evidence tiers: a judge's reliability is measured over the whole campaign, not one kind of
    case of it.

    Args:
        results_by_cell: Each cell's results, from :func:`_results_by_cell`.
        stratum_of_case: The stratum each case declares, by case id; a case that did not resolve is absent
            and reads as declaring none.
        records: The score projection, for judged dimensions.
        design: The campaign's declaration, for the bar a judged dimension carries.
        tiers: The judges' evidence tiers.
        profile: The host whose vocabulary this reads.

    Returns:
        Each broken-down cell's strata — named strata in name order, then the undeclared one — keyed by the
        cell's coordinates.
    """
    by_stratum: dict[_StratumKey, dict[_CellKey, list[EvalResult]]] = {}
    for key, members in results_by_cell.items():
        if not any(stratum_of_case.get(result.test_case_id) is not None for result in members):
            continue
        for result in members:
            by_stratum.setdefault(stratum_of_case.get(result.test_case_id), {}).setdefault(key, []).append(result)
    strata: dict[_CellKey, list[StratumFacts]] = {}
    for stratum in sorted(by_stratum, key=lambda name: (name is None, name or "")):
        slice_by_cell = by_stratum[stratum]
        result_ids = {result.id for members in slice_by_cell.values() for result in members}
        judged = _judged_by_cell(
            _judged_measures(
                [record for record in records if record.result_id in result_ids], slice_by_cell, design, tiers=tiers
            )
        )
        for key, members in slice_by_cell.items():
            strata.setdefault(key, []).append(
                StratumFacts(
                    stratum=stratum,
                    n_observations=len(members),
                    n_cases=len({result.test_case_id for result in members}),
                    n_infra_excluded=len(members) - len(_non_faulted(members)),
                    n_candidate_failed=_candidate_failed(members),
                    n_no_turn=_no_turn(members),
                    measures=_measure_collection(members, profile=profile, undeclared="scored"),
                    judged=judged.get(key, []),
                )
            )
    return strata


def _cell_measures(
    cells: list[Cell],
    results_by_cell: dict[_CellKey, list[EvalResult]],
    judged_measures: list[JudgedMeasure],
    *,
    strata: dict[_CellKey, list[StratumFacts]],
    short_runs: dict[str, str],
    incomplete_runs: dict[str, str],
    profile: HostProfile,
    control: str | None,
    names: Mapping[str, str],
) -> list[CellFacts]:
    """Everything measured in each cell — the facts an analysis freezes as its decision surface.

    **One population, the bars' own**: each cell's results the harness did not fault
    (:func:`_non_faulted`), so a measure's value here and a :class:`BarVerdict` on the same cell and
    measure are one mean over one set of observations, never two readings a reader has to reconcile.
    A cost or latency measure is read over the ``delivered`` part of it — the turns the candidate took — by
    the same walk for the cell and the bar alike, and the failures that took none are counted beside it
    (``n_no_turn``, of ``n_candidate_failed``).

    **The judged readings are the judged measures, transposed** — read off what
    :func:`_judged_measures` produced rather than scored again, so a cell's reading of a dimension
    and that dimension's arm for the cell are the same numbers by construction, not by two
    population rules happening to agree.

    Args:
        cells: The pooled cells, for their replication.
        results_by_cell: Each cell's results, from :func:`_results_by_cell`.
        judged_measures: The bundle's judged measures, from :func:`_judged_measures`.
        strata: Each broken-down cell's strata, from :func:`_cell_strata`; a cell absent from it carries none.
        short_runs: The bundle's short-run sentences, by run id.
        incomplete_runs: The bundle's incomplete-run statuses, by run id.
        profile: The host whose vocabulary this reads.
        control: The declared control's variant key, or None.
        names: The bundle's arm names (:func:`~threetears.evals.analysis.arms.arm_names`), which order the arms.

    Returns:
        One entry per cell, in the decision surface's row order
        (:func:`~threetears.evals.analysis.arms.surface_order`): the control's cells first, as the reference,
        then every other arm alphabetically by name, each arm's rigs by id. Not a ranking.
    """
    judged_by_cell = _judged_by_cell(judged_measures)
    facts = []
    for cell in sorted(cells, key=lambda c: (c.variant_key, c.apparatus_class_id)):
        key = (cell.variant_key, cell.apparatus_class_id)
        members = results_by_cell[key]
        counted = _non_faulted(members)
        run_ids = _member_run_ids(members)
        facts.append(
            CellFacts(
                variant_key=cell.variant_key,
                apparatus_class_id=cell.apparatus_class_id,
                run_ids=run_ids,
                n_observations=cell.n_observations,
                n_cases=cell.n_cases,
                repeats_per_case_min=cell.repeats_per_case_min,
                repeats_per_case_max=cell.repeats_per_case_max,
                n_infra_excluded=len(members) - len(counted),
                n_candidate_failed=_candidate_failed(members),
                n_no_turn=_no_turn(members),
                # Every member, faulted ones included: the walk leaves a fault out of each measure whose
                # population is `scored` and keeps it in one whose population is `all_observed`.
                measures=_measure_collection(members, profile=profile, undeclared="scored"),
                judged=judged_by_cell.get(key, []),
                short_runs={run_id: short_runs[run_id] for run_id in run_ids if run_id in short_runs},
                incomplete_runs={run_id: incomplete_runs[run_id] for run_id in run_ids if run_id in incomplete_runs},
                strata=strata.get(key, []),
            )
        )
    return surface_order(facts, control=control, names=names)


__all__: list[str] = []

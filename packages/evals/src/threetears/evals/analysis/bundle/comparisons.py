"""Multiple comparisons and guardrails: each contrast tested against the control per live question, corrected together.

:func:`_multiple_comparisons` builds each question's family, tests every contrast (:func:`_compare`), corrects the
family by Holm's method (:func:`_corrected_family`) and decides every guardrail (:func:`_guardrail_check`).
:func:`planning_readings` and :func:`_reading_scope` read the same families for planning and exploratory labels.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Literal, NamedTuple

from threetears.evals.analysis.contention import withhold_contended_latency
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.reporting import (
    METRIC_SCORE,
    ScoreRecord,
    project_score_records,
)
from threetears.evals.analysis.stats import (
    SIGNIFICANCE_ALPHA,
    bounded_difference_interval,
    composite_significance,
    contrast_samples,
    difference_interval,
    equivalence_untested_reason,
    exact_decimal,
    GUARDRAIL_HELD_NEEDS_RANGE,
    guardrail_decision,
    holm_adjust,
    interval_permits_separation,
    no_spread_p,
    paired_equivalence,
    pairing_correlation,
    separation_test,
)
from threetears.evals.kernel.campaign import ReadingKind
from threetears.evals.kernel.declaration import (
    JUDGED_MERIT_AXIS,
    CampaignDesign,
    axis_in_question_scope,
    exploratory_reading,
    preregistered_questions,
)
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.metrics import (
    MeritAxis,
    MetricDescriptor,
    classifier_label_of,
    describe_reported_measure,
    describe_rubric_dim,
    is_latency_measure,
    materiality,
    summary_population,
)
from threetears.evals.schema.models import (
    EvalResult,
    RubricScale,
)
from threetears.evals.kernel.surface import (
    GuardrailCell,
    GuardrailCheck,
    GuardrailReadings,
)
from threetears.evals.analysis.bundle.schema import (
    TRIAL_PASS_NAME,
    ComparedCell,
    ComparisonFamily,
    ComparisonReading,
    ComparisonVerdict,
    exploratory_disclosure,
    FamilyComparison,
    PairingCorrelation,
    JudgedMeasure,
    MultipleComparisons,
    ReadingScope,
    RealizedDesign,
)
from threetears.evals.analysis.bundle.measures import (
    _failures_as_misses,
    _measure_collection,
)
from threetears.evals.analysis.bundle.mechanisms import (
    _MechanismObservations,
    _model_contrast_confounds,
    _served_model_confounds,
    _ServedModels,
)
from threetears.evals.analysis.bundle.trials import read_trial
from threetears.evals.analysis.bundle.cell_reads import (
    _boundary_dimensions,
    _CellKey,
    _judged_values,
    _non_faulted,
    _took_no_turn,
)

if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
    from threetears.evals.schema.models import EvalRun


#: Why a bundle carries no family of comparisons, one sentence per cause.
#: Opens the campaign-wide family's disclosure, where the campaign declares no question.
_NO_QUESTION_FAMILY = (
    "This campaign declares no live question, so every comparison it holds — each contrast against the control, "
    "on every reading on a merit axis — is corrected as one family."
)
_NO_CONTROL_TO_COMPARE_AGAINST = (
    "No control resolved, so there is no arm for a contrast to be tested against and no separation between arms is "
    "tested here; every comparison in this campaign is marginal."
)


def _per_case_values(
    members: list[EvalResult], judged_rows: dict[str, list[ScoreRecord]], *, profile: HostProfile
) -> dict[tuple[ReadingKind, str], dict[str, float]]:
    """Every numeric reading of one cell, as one mean per case — the unit a family comparison tests.

    A case is the independent draw (rule: count cases, not attempts), so repeats of one case are
    averaged before any test sees them. A measure's per-case value is read off the same walk the
    cell's own collection is (:func:`_measure_collection`, each measure over its own population), so
    the value tested and the value the cell reports are one computation; a judged dimension's, off the
    non-faulted results' scores, as :func:`_bar_reading` reads one.

    :param members: One cell's results, faulted ones included.
    :ptype members: list[EvalResult]
    :param judged_rows: Projected judged-score rows by result id.
    :ptype judged_rows: dict[str, list[ScoreRecord]]
    :param profile: The host whose vocabulary this reads.
    :ptype profile: HostProfile
    :return: ``{(reading, name): {test_case_id: per-case mean}}`` for every reading with a numeric value.
    :rtype: dict[tuple[ReadingKind, str], dict[str, float]]
    """
    by_case: dict[str, list[EvalResult]] = defaultdict(list)
    for result in members:
        by_case[result.test_case_id].append(result)
    values: dict[tuple[ReadingKind, str], dict[str, float]] = defaultdict(dict)
    for case_id in sorted(by_case):
        # The candidate's own spend among them (``production_replicating_cost``, which the walk yields over the
        # turns the candidate took and only where a result measured it): a contrast on cost tests it, never
        # ``cost_usd``, which sums the judge's spend too.
        for summary in _measure_collection(by_case[case_id], profile=profile, undeclared="scored").measures:
            if summary.mean is not None:
                values[("measure", summary.name)][case_id] = summary.mean
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for result in _non_faulted(members):
        for dimension, record in _judged_values(judged_rows.get(result.id, [])):
            if record.value is not None:
                scores[(dimension, result.test_case_id)].append(float(record.value))
    for (dimension, case_id), case_scores in sorted(scores.items()):
        values[("judged", dimension)][case_id] = sum(case_scores) / len(case_scores)
    return values


class PlanningReading(NamedTuple):
    """One reading a comparison family could test, as earlier runs observed it: every repeat, by run and case."""

    reading: ReadingKind
    name: str
    higher_is_better: bool
    #: The reading's declared inclusive bounds, or None where it declares none.
    value_range: tuple[float, float] | None
    #: ``{run id: {test case id: [one value per repeat]}}``, in run and case order.
    repeats: dict[str, dict[str, list[float]]]


def planning_readings(
    runs: list[EvalRun], results_by_run: dict[str, list[EvalResult]], *, profile: HostProfile
) -> list[PlanningReading]:
    """Every reading an unscoped comparison family would test, with each earlier run's per-repeat values.

    What a power pre-flight plans from: the readings :func:`_family_readings` admits for a question that names
    no axis (a measure with a better end on a merit axis, a capability judged dimension), each observation
    read by the one walk a family's per-case value is read by (:func:`_per_case_values`, over the one result),
    so a repeat's value here and a case's mean in a comparison are one computation. Guardrails are left out:
    they are held, not tested for a difference; so is latency read under concurrency, which no comparison reads.

    :param runs: The earlier runs.
    :ptype runs: list[EvalRun]
    :param results_by_run: Each run's results.
    :ptype results_by_run: dict[str, list[EvalResult]]
    :param profile: The host whose vocabulary this reads.
    :ptype profile: HostProfile
    :return: One entry per reading with a value, sorted by reading kind and name.
    :rtype: list[PlanningReading]
    """
    results_by_run = _failures_as_misses({run.id: results_by_run.get(run.id, []) for run in runs})
    # Latency read under concurrency is in no comparison (#701), so it plans none either.
    results_by_run = {
        run_id: withhold_contended_latency(run_results, profile.measures)
        for run_id, run_results in results_by_run.items()
    }
    results = [result for run in runs for result in results_by_run[run.id]]
    projection = project_score_records(runs, results, known_run_ids=None, archived_run_ids=None, profile=profile)
    judged_rows: dict[str, list[ScoreRecord]] = {}
    scales: dict[str, RubricScale] = {}
    for record in projection.records:
        judged_rows.setdefault(record.result_id, []).append(record)
        if record.metric == METRIC_SCORE and record.rubric_dim and record.rubric_scale is not None:
            scales[record.rubric_dim] = record.rubric_scale
    boundary = _boundary_dimensions(results)
    repeats: dict[tuple[ReadingKind, str], dict[str, dict[str, list[float]]]] = {}
    for run in runs:
        for result in sorted(results_by_run[run.id], key=lambda one: (one.test_case_id, one.k_iteration, one.id)):
            for reading, by_case in _per_case_values([result], judged_rows, profile=profile).items():
                for case_id, value in by_case.items():
                    repeats.setdefault(reading, {}).setdefault(run.id, {}).setdefault(case_id, []).append(value)
    readings = []
    for (kind, name), by_run in sorted(repeats.items()):
        if kind == "measure":
            descriptor = describe_reported_measure(name, profile.measures)
            if (
                descriptor.higher_is_better is None
                or classifier_label_of(name) is not None
                or not axis_in_question_scope(descriptor.merit_axis, [])
            ):
                continue
            higher, value_range = descriptor.higher_is_better, descriptor.value_range
        else:
            if name in boundary:
                continue
            judged = describe_rubric_dim(name, scale=scales.get(name, "ordinal"))
            if judged.higher_is_better is None:
                continue
            higher, value_range = judged.higher_is_better, judged.value_range
        readings.append(PlanningReading(kind, name, higher, value_range, by_run))
    return readings


def _family_readings(
    axes: list[MeritAxis], catalog: dict[str, MetricDescriptor], judged_measures: list[JudgedMeasure]
) -> dict[tuple[ComparisonReading, str], bool]:
    """The readings a question with these axes could draw a verdict from, each with its better direction.

    A measure qualifies when it has a better end (a reading with none cannot improve or regress) and
    sits on one of the axes; a per-label classifier statistic does not, because it is computed from a
    whole cell's confusion counts and has no per-case value to test. A capability judged dimension sits on
    the quality axis. A guardrail — a boundary judged dimension, or a measure declared one, which serves no
    axis — is in no family: it is held, never traded against a gain, and is decided on its own
    (:func:`_guardrails`). An unscoped question (no axes) asks about every axis — every axis, not every measure:
    a measure that serves no merit axis contributes to no verdict (:data:`MeritAxis`), scoped or not
    (:func:`~threetears.evals.kernel.declaration.axis_in_question_scope`).
    That is how the measuring apparatus's own readings stay out of a contrast between candidates: the
    judge phase's time (``judge_ms``), the drain wait, and ``cost_usd`` and ``program_cost``, which sum the
    judge's spend — what it cost to MEASURE an arm. The candidate's spend is ``production_replicating_cost``,
    on the cost axis.

    :param axes: The question's merit axes; empty for an unscoped question.
    :ptype axes: list[MeritAxis]
    :param catalog: The bundle's measure catalog.
    :ptype catalog: dict[str, MetricDescriptor]
    :param judged_measures: The bundle's judged measures.
    :ptype judged_measures: list[JudgedMeasure]
    :return: ``{(reading, name): higher_is_better}``.
    :rtype: dict[tuple[ComparisonReading, str], bool]
    """
    readings: dict[tuple[ComparisonReading, str], bool] = {}
    for name, descriptor in catalog.items():
        if descriptor.higher_is_better is None or classifier_label_of(name) is not None:
            continue
        if axis_in_question_scope(descriptor.merit_axis, axes):
            readings[("measure", name)] = descriptor.higher_is_better
    if axis_in_question_scope(JUDGED_MERIT_AXIS, axes):
        for measure in judged_measures:
            if measure.axis == "capability":
                readings[("judged", measure.name)] = measure.higher_is_better
    return readings


#: The overall pass rate's range: a share of a case's usable trials.
_TRIAL_PASS_RANGE = (0.0, 1.0)


def _trial_pass_values(
    members: list[EvalResult], *, rubric_threshold: int, catalog: Mapping[str, MetricDescriptor]
) -> dict[str, float]:
    """One cell's overall pass rate, one value per case: the share of the case's usable trials that passed.

    Each trial ends as the scoreboard reads it (:func:`~threetears.evals.analysis.bundle.trials.read_trial`), so a
    case's value and the census are one classification. A lost trial says nothing about the candidate and is in no
    case's share, as it is in no arm's usable count; a case that lost every trial has no value. Repeats of a case are
    averaged before any test sees them, as every other reading's are (:func:`_per_case_values`): the case is the
    independent draw. A cell none of whose usable trials carried a pass criterion has no pass rate to compare: each
    of its trials passed only because nothing could fail it.

    :param members: One cell's results.
    :ptype members: list[EvalResult]
    :param rubric_threshold: The 1–5 level a judged dimension must reach to pass.
    :ptype rubric_threshold: int
    :param catalog: The bundle's measure catalog, which says which checks a trial is graded on.
    :ptype catalog: Mapping[str, MetricDescriptor]
    :return: ``{test_case_id: share passed}`` over the cases with a usable trial.
    :rtype: dict[str, float]
    """
    by_case: dict[str, list[bool]] = defaultdict(list)
    graded = False
    for result in members:
        reading = read_trial(result, rubric_threshold=rubric_threshold, catalog=catalog)
        if reading.state != "lost":
            by_case[result.test_case_id].append(reading.state == "passed")
            graded = graded or not reading.no_criterion
    if not graded:
        return {}
    return {case_id: sum(passed) / len(passed) for case_id, passed in sorted(by_case.items())}


def _test_samples(
    control_values: Mapping[str, float], contrast_values: Mapping[str, float]
) -> tuple[list[float], list[float], bool]:
    """The two samples a contrast against the control reads, and whether they are paired.

    Paired over the cases both cells ran when they share at least two — far more powerful, and the design
    a fixed case set exists for — else each side's per-case values, unpaired. One choice for every reading
    of a contrast, a comparison's and a guardrail's alike.

    :return: ``(control sample, contrast sample, paired)``, the two aligned by case when paired
        (:func:`~threetears.evals.analysis.stats.contrast_samples`, the rule the frontier's boundary pillar reads too).
    :rtype: tuple[list[float], list[float], bool]
    """
    return contrast_samples(control_values, contrast_values)


class _Tested(NamedTuple):
    """One comparison before its family's correction: the comparison, the p's it carries, and its samples."""

    comparison: FamilyComparison
    p_raw: float | None
    equivalence_p_raw: float | None
    samples: tuple[list[float], list[float]]
    #: The reading's declared range, which bounds the interval on its delta (:func:`difference_interval`).
    value_range: tuple[float, float] | None = None


def _compare(
    reading: tuple[ComparisonReading, str],
    higher_is_better: bool,
    control: tuple[_CellKey, dict[str, float]],
    contrast: tuple[_CellKey, dict[str, float]],
    *,
    threshold: float | None,
    margin_source: Literal["measure", "run"] | None = None,
    value_range: tuple[float, float] | None = None,
    no_turn: tuple[str, ...] = (),
    contended: tuple[str, ...] = (),
) -> _Tested:
    """Test one contrast against the control on one reading, before correction.

    Paired over the cases both cells ran when they share at least two — far more powerful, and the
    design a fixed case set exists for — else the unpaired test over each side's per-case values
    (:func:`~threetears.evals.analysis.stats.composite_significance`), and where the values have no spread
    the bounded test on the declared range every other surface reads that gap by
    (:func:`~threetears.evals.analysis.stats.separation_test`), so one concept has one answer. With no range
    such a gap is ``not_separated``, with ``not_separated_reason`` naming the remedy. The means, the counts and the
    delta are all over the cases the test read, and each side says how many of its own it left out, so the
    figures a reader sees are the figures the test saw. A paired comparison on a measure with a declared
    margin also runs the paired equivalence test (TOST) against it.

    :param reading: The reading's kind and name.
    :ptype reading: tuple[ComparisonReading, str]
    :param higher_is_better: Which way is better on it.
    :ptype higher_is_better: bool
    :param control: The control cell and its per-case values.
    :ptype control: tuple[_CellKey, dict[str, float]]
    :param contrast: The contrast cell and its per-case values.
    :ptype contrast: tuple[_CellKey, dict[str, float]]
    :param threshold: The measure's declared materiality threshold, which labels the delta through the one
        predicate every surface uses (:func:`~threetears.evals.kernel.metrics.materiality`) and is the
        equivalence test's margin; None for a measure that declared none and for a judged dimension.
    :ptype threshold: float | None
    :param margin_source: Where ``threshold`` came from (:func:`_margin_of`), which the comparison records.
    :ptype margin_source: Literal['measure', 'run'] | None
    :param value_range: The reading's declared inclusive bounds, which the equivalence test reads so its error
        rate holds on coarse values at every n (:func:`~threetears.evals.analysis.stats.paired_equivalence`),
        and which bound the interval on the delta; None where it declares none.
    :ptype value_range: tuple[float, float] | None
    :param no_turn: Which sides (``"control"``, ``"arm"``) have no turn to read a turn's time or spend over —
        every result there failed with no turn taken — so an untested comparison says that, the reason,
        rather than that too few cases carried the reading.
    :ptype no_turn: tuple[str, ...]
    :param contended: Which sides of a reading of elapsed time held latency read under concurrency, which the
        bundle withheld (:mod:`~threetears.evals.analysis.contention`) — so an untested comparison says that,
        rather than that too few cases carried the reading.
    :ptype contended: tuple[str, ...]
    :return: The comparison with its adjusted p's, interval and verdict still unset, its raw p's (None where no
        test produced one) for the family's correction, and the samples the test read, for the interval.
    :rtype: _Tested
    """
    (control_key, control_values), (contrast_key, contrast_values) = control, contrast
    a, b, paired = _test_samples(control_values, contrast_values)
    # The separation test every surface reads a gap by (:func:`separation_test`): the t-test's where the values
    # have spread, and where they have none — every shared case moved by one amount, or each side constant — the
    # bounded test on the reading's declared range, decided on exact values. With no range a gap with no spread
    # is not separated, and says why: the exact sign-flip p tests symmetry, not the mean (#597).
    separation = separation_test(a, b, paired=paired, value_range=value_range)
    p_raw = separation.p
    hedges_g, _, _ = composite_significance(a, b, paired=paired)
    no_spread = (
        no_spread_p([exact_decimal(x) for x in a], [exact_decimal(y) for y in b], paired=paired)
        if len(a) >= 2 and len(b) >= 2
        else None
    )
    if no_spread is not None:
        # No spread, decided exactly: no finite effect size, whatever a float residue lets the t statistic say.
        hedges_g = None
    mean_a = sum(a) / len(a) if a else None
    mean_b = sum(b) / len(b) if b else None
    untested_reason = None
    if p_raw is None:
        # Named from the branch that refused, since the causes have different remedies.
        if no_turn:
            untested_reason = (
                f"every result of the {' and the '.join(no_turn)} failed with no turn taken, so there is no "
                "turn's time or spend to compare"
            )
        elif contended:
            untested_reason = (
                f"the {' and the '.join(contended)}'s latency was read while other cells or runs executed beside it, "
                "so it is withheld and not compared (`latency_contended`)"
            )
        elif len(a) < 2 or len(b) < 2:
            untested_reason = "fewer than two cases carry this reading on a side"
        elif separation.refusal is None:
            untested_reason = "the values differ by less than floating point resolves, so no t statistic exists"
    # The equivalence test only where the separation test produced a p, so each equivalence hypothesis has
    # its comparison's separation hypothesis beside it in the family (see holm_adjust's max_true).
    margin = threshold if paired and threshold and p_raw is not None else None
    # A margin with no declared range is not tested at all (#695): no test of a mean holds α without one, and
    # `equivalent` is the one claim that arms are alike. The comparison says why, naming the remedy.
    equivalence_refused = equivalence_untested_reason(margin, value_range)
    if equivalence_refused is not None:
        margin = None
    # The differences of exact values, so a float residue cannot pass for a spread nor a spread for a constant; on
    # the measure's declared range the bounded test decides either, at the error rate it states.
    _, equivalence_p_raw = paired_equivalence(
        [exact_decimal(y) - exact_decimal(x) for x, y in zip(a, b)], margin, value_range=value_range
    )
    delta = None if mean_a is None or mean_b is None else mean_b - mean_a
    # What pairing bought: the arms' per-case correlation over exactly the cases the paired test read (#595).
    r = pairing_correlation(a, b) if paired and p_raw is not None else None
    comparison = FamilyComparison(
        reading=reading[0],
        name=reading[1],
        higher_is_better=higher_is_better,
        control=ComparedCell(
            variant_key=control_key[0],
            apparatus_class_id=control_key[1],
            n_cases=len(a),
            mean=mean_a,
            n_left_out=len(control_values) - len(a),
        ),
        contrast=ComparedCell(
            variant_key=contrast_key[0],
            apparatus_class_id=contrast_key[1],
            n_cases=len(b),
            mean=mean_b,
            n_left_out=len(contrast_values) - len(b),
        ),
        delta=delta,
        hedges_g=hedges_g if p_raw is not None else None,
        test=None if p_raw is None else ("paired" if paired else "unpaired"),
        pairing_correlation=None if r is None else PairingCorrelation(r=r, n_cases=len(a)),
        basis=separation.basis,
        p_raw=p_raw,
        equivalence_margin=margin,
        margin_source=margin_source if threshold is not None else None,
        equivalence_p_raw=equivalence_p_raw,
        equivalence_untested_reason=equivalence_refused,
        # A gap with no spread on a reading with no declared range is not separated, with the refusal naming the
        # remedy; it joins no family, since no test ran to correct.
        verdict="untested" if untested_reason is not None else "not_separated",
        untested_reason=untested_reason,
        not_separated_reason=separation.refusal if untested_reason is None else None,
        materiality=None if delta is None else materiality(threshold, delta),
    )
    return _Tested(comparison, p_raw, equivalence_p_raw, (a, b), value_range)


def _margin_of(
    reading: tuple[ComparisonReading, str], catalog: Mapping[str, MetricDescriptor], run_margins: Mapping[str, float]
) -> tuple[float | None, Literal["measure", "run"] | None]:
    """A reading's margin and where it came from: the descriptor's threshold, else its runs' declared margin.

    A core rate measure's descriptor declares none (:data:`~threetears.evals.kernel.metrics.RUN_MARGIN_MEASURES`),
    so the two never both exist for one measure. A judged dimension and the overall pass rate have neither.
    """
    if reading[0] != "measure":
        return None, None
    declared = catalog[reading[1]].materiality_threshold
    if declared is not None:
        return declared, "measure"
    if (margin := run_margins.get(reading[1])) is not None:
        return margin, "run"
    return None, None


def _family_disclosure(
    family_size: int, n_untested: int, alpha: float, *, n_equivalence: int = 0, campaign_wide: bool = False
) -> str:
    """The sentence a writer quotes about one family, composed from what the family holds."""
    asked = "the campaign holds" if campaign_wide else "this question asks about"
    if family_size == 0:
        sentence = f"No comparison {asked} carried a p, so it supports no separation between arms."
    else:
        equivalence = (
            f", with {n_equivalence} equivalence test{'s' if n_equivalence != 1 else ''} against a declared margin,"
            if n_equivalence
            else ""
        )
        sentence = (
            f"{family_size} comparison{'s' if family_size != 1 else ''} {asked} carried a p and "
            f"{'were' if family_size != 1 else 'was'}{equivalence} corrected together by Holm's method at "
            f"α={format_number(alpha)}; a separation stands only where the adjusted p is below it and the interval "
            f"excludes zero. Each interval "
            f"is at {format_number(100 * (1 - alpha / family_size))}%, so the family's intervals hold together at "
            f"{format_number(100 * (1 - alpha))}%."
        )
    if n_untested:
        sentence += f" {n_untested} more could not be tested; each says why."
    return f"{_NO_QUESTION_FAMILY} {sentence}" if campaign_wide else sentence


def _corrected_family(question_id: str | None, axes: list[MeritAxis], tested: list[_Tested]) -> ComparisonFamily:
    """Correct one family's tests together, and read each comparison's verdict and interval off the result.

    Every separation p and every equivalence p is Holm-adjusted as one family, capped at the separation
    count (:func:`~threetears.evals.analysis.stats.holm_adjust`'s ``max_true``): a comparison's two
    hypotheses — no difference, a difference of at least the margin — cannot both be true, so the family's
    error stays at α over every verdict it can reach. Each interval is at ``1 − α/m`` (Bonferroni over the
    ``m`` separations), which holds the family's intervals together at ``1 − α``.

    **A separation is read off its interval as well as its adjusted p** (#597): ``improved`` or ``regressed``
    needs the interval, where one exists, to exclude zero. Holm's later steps reject some comparisons whose
    Bonferroni interval still reaches zero, and a reader must never see a separation beside an interval that
    includes no change. An interval that excludes zero has ``m · p_raw < α``, which Holm always rejects, so the
    rule is Bonferroni's wherever an interval exists: a subset of Holm's rejections, so the family's error stays
    at α, at the cost of the separations Holm alone would add. Where every case moved by one amount no t
    interval exists, and the interval is the bounded test's on the declared range, the test whose p was
    corrected (#597); with no range that comparison was never tested and is not in the family.

    **A claim of no worse by more than a margin reads the bounded interval** (``bounded_interval``), the bounded
    test's on the declared range at the same level, valid at every n; the t interval is not. Read for non-inferiority
    on a pass rate's coarse per-case shares at the margin, it let a decision rule's ``switch`` through up to 12% of
    the time against the 2.5% it states (``docs/measuring-soundly.md``), the failure a guardrail's ``held`` was moved
    off it for. A reading with no declared range has none.

    :param question_id: The question the family serves, or None for the campaign-wide family.
    :ptype question_id: str | None
    :param axes: The question's merit axes.
    :ptype axes: list[MeritAxis]
    :param tested: The family's comparisons, tested and uncorrected.
    :ptype tested: list[_Tested]
    :return: The corrected family.
    :rtype: ComparisonFamily
    """
    family_size = sum(1 for one in tested if one.p_raw is not None)
    n_equivalence = sum(1 for one in tested if one.equivalence_p_raw is not None)
    raw = [p for one in tested for p in (one.p_raw, one.equivalence_p_raw) if p is not None]
    adjusted = iter(holm_adjust(raw, max_true=family_size) if raw else [])
    interval_level = 1.0 - SIGNIFICANCE_ALPHA / family_size if family_size else None
    comparisons = []
    for one in tested:
        comparison = one.comparison
        if one.p_raw is not None and interval_level is not None:
            p_adjusted = next(adjusted)
            equivalence_p_adjusted = next(adjusted) if one.equivalence_p_raw is not None else None
            control_sample, contrast_sample = one.samples
            interval = (
                # No spread: no t interval exists, and on the declared range the bounded test gives one, so a
                # separation is never shown with no interval beside it (#597), nor identical values without bounds.
                bounded_difference_interval(
                    control_sample,
                    contrast_sample,
                    paired=comparison.test == "paired",
                    value_range=one.value_range,
                    confidence=interval_level,
                )
                if comparison.basis in ("bounded", "identical") and one.value_range is not None
                else difference_interval(
                    control_sample,
                    contrast_sample,
                    paired=comparison.test == "paired",
                    confidence=interval_level,
                    value_range=one.value_range,
                )
            )
            verdict: ComparisonVerdict = "not_separated"
            if p_adjusted < SIGNIFICANCE_ALPHA and comparison.delta and interval_permits_separation(interval):
                verdict = "improved" if (comparison.delta > 0) == comparison.higher_is_better else "regressed"
            elif equivalence_p_adjusted is not None and equivalence_p_adjusted < SIGNIFICANCE_ALPHA:
                verdict = "equivalent"
            bounded = (
                None
                if one.value_range is None
                else interval
                if comparison.basis in ("bounded", "identical")
                else bounded_difference_interval(
                    control_sample,
                    contrast_sample,
                    paired=comparison.test == "paired",
                    value_range=one.value_range,
                    confidence=interval_level,
                )
            )
            comparison = comparison.model_copy(
                update={
                    "p_adjusted": p_adjusted,
                    "equivalence_p_adjusted": equivalence_p_adjusted,
                    "interval": interval,
                    "bounded_interval": bounded,
                    # What the bounded interval read, so a projection can read it again at more cases.
                    "case_values": None if bounded is None else (list(control_sample), list(contrast_sample)),
                    "verdict": verdict,
                }
            )
        comparisons.append(comparison)
    n_untested = sum(1 for comparison in comparisons if comparison.verdict == "untested")
    return ComparisonFamily(
        question_id=question_id,
        merit_axes=axes,
        family_size=family_size,
        n_equivalence_tests=n_equivalence,
        interval_level=interval_level,
        n_untested=n_untested,
        comparisons=comparisons,
        disclosure=_family_disclosure(
            family_size, n_untested, SIGNIFICANCE_ALPHA, n_equivalence=n_equivalence, campaign_wide=question_id is None
        ),
    )


def _multiple_comparisons(
    declared: CampaignDesign | None,
    realized: RealizedDesign,
    results_by_cell: dict[_CellKey, list[EvalResult]],
    records: list[ScoreRecord],
    *,
    catalog: dict[str, MetricDescriptor],
    judged_measures: list[JudgedMeasure],
    observations: _MechanismObservations,
    served: _ServedModels,
    profile: HostProfile,
    run_margins: Mapping[str, float] | None = None,
    contended: Collection[_CellKey] = frozenset(),
    first_run_at: str | None,
    rubric_threshold: int,
    pass_rate_questions: Collection[str] = frozenset(),
) -> tuple[MultipleComparisons, GuardrailReadings]:
    """Test each contrast against the control, per live question — or campaign-wide — and decide every guardrail.

    A question's family is every comparison it could draw a verdict from: each contrast cell against
    the control cell under the same rig (a contrast across rigs differs by its instrument too, so it is
    not this test), on every reading on the question's axes (:func:`_family_readings`). The family is
    corrected by Holm's method over the comparisons that carried a p, and each verdict is read off its
    adjusted p, so a family of ten cannot hand the writer a chance "difference" as a finding.

    The guardrails are read over the same pairs and the same per-case values, and kept out of every
    family: each is decided on its own (:func:`_guardrails`), so no capability gain can offset one.

    **The overall pass rate joins only the family of a question whose decision rule reads it** (a rate clause naming
    no one reading, ``pass_rate_questions``): there it is a hypothesis the question asks, so its family grows by one and
    every adjusted p in it moves with it. No other family tests it, and with no such question it is compared nowhere: it
    is the scoreboard's count, exploratory. Each case's value is the share of its usable trials that passed
    (:func:`_trial_pass_values`), tested and bounded as a check's pass rate is, on its 0-1 range.

    :param declared: The campaign's declaration, for its live questions asked before its first run
        (:func:`~threetears.evals.kernel.declaration.preregistered_questions`): a question asked after it builds
        no family of its own, and the campaign is read as if it had asked nothing.
    :ptype declared: CampaignDesign | None
    :param realized: The derived design, for the control arm.
    :ptype realized: RealizedDesign
    :param results_by_cell: Each cell's results.
    :ptype results_by_cell: dict[_CellKey, list[EvalResult]]
    :param records: The assembly's score projection rows, where judged scores live.
    :ptype records: list[ScoreRecord]
    :param catalog: The bundle's measure catalog — direction and axis per measure.
    :ptype catalog: dict[str, MetricDescriptor]
    :param judged_measures: The bundle's judged measures.
    :ptype judged_measures: list[JudgedMeasure]
    :param observations: The campaign's mechanism observations, read for each contrast between two models.
    :ptype observations: _MechanismObservations
    :param served: Which model answered each result's candidate calls, read for every contrast.
    :ptype served: _ServedModels
    :param profile: The host whose vocabulary this reads.
    :ptype profile: HostProfile
    :param run_margins: The margins every member run declared alike on a core rate measure (:func:`_run_margins`),
        read as that measure's margin, since its descriptor declares none.
    :ptype run_margins: Mapping[str, float] | None
    :param contended: The cells holding latency read under concurrency, which the bundle withheld — named as the
        reason a comparison or a guardrail on elapsed time could not be read, where it could not.
    :ptype contended: Collection[_CellKey]
    :param first_run_at: The earliest member run's ``created_at``, which a question must predate to scope a family.
    :ptype first_run_at: str | None
    :param rubric_threshold: The 1–5 level a judged dimension must reach for a trial to pass, which the overall pass
        rate's per-case shares read (:func:`_trial_pass_values`), as the scoreboard does.
    :ptype rubric_threshold: int
    :param pass_rate_questions: The ids of the live questions whose decision rule reads the overall pass rate.
    :ptype pass_rate_questions: Collection[str]
    :return: One family per live question, in declaration order; one campaign-wide family over every reading when
        the campaign declares no live question; or none, with the reason, when no control resolved. Beside
        them, every guardrail decided for each arm against the control.
    :rtype: tuple[MultipleComparisons, GuardrailReadings]
    """
    guardrail_readings = _guardrail_readings(catalog, judged_measures, declared)
    if realized.control_arm is None:
        return MultipleComparisons(withheld=_NO_CONTROL_TO_COMPARE_AGAINST), _guardrails_of(
            guardrail_readings,
            [],
            withheld=_NO_CONTROL_TO_HOLD_AGAINST if guardrail_readings else None,
        )
    questions = preregistered_questions(declared, first_run_at) if declared is not None else []
    # A campaign that asked nothing (or only once its results existed) is not thereby licensed to report chance
    # differences: its family is every comparison it holds, on every reading, corrected as one.
    scopes: list[tuple[str | None, list[MeritAxis]]] = (
        [(question.id, list(question.merit_axes)) for question in questions] if questions else [(None, [])]
    )
    control_variant = realized.control_arm.variant_key
    # The control's cells under an earlier identity predicate are the control, not a rival arm: compared with the
    # control they would read the control against itself under another digest, so they are in no pair.
    not_contrasts = {control_variant} | ({realized.earlier_control.variant_key} if realized.earlier_control else set())

    judged_rows: dict[str, list[ScoreRecord]] = {}
    for record in records:
        judged_rows.setdefault(record.result_id, []).append(record)
    values: dict[_CellKey, dict[tuple[ComparisonReading, str], dict[str, float]]] = {}
    for key, members in results_by_cell.items():
        values[key] = {
            reading: by_case for reading, by_case in _per_case_values(members, judged_rows, profile=profile).items()
        }
        values[key][("trial", TRIAL_PASS_NAME)] = _trial_pass_values(
            members, rubric_threshold=rubric_threshold, catalog=catalog
        )
    pairs = [
        (control_key, contrast_key)
        for control_key in sorted(values)
        if control_key[0] == control_variant
        for contrast_key in sorted(values)
        if contrast_key[1] == control_key[1] and contrast_key[0] not in not_contrasts
    ]

    # One answer per pair, stated on every reading the pair is compared on: which pair diverged is a fact
    # about the two arms, and a writer reading one comparison should not have to find it on another.
    pair_confounds = {
        pair: _model_contrast_confounds(
            results_by_cell[pair[0]], results_by_cell[pair[1]], observations, profile=profile
        )
        + _served_model_confounds((result.id for key in pair for result in results_by_cell[key]), served)
        for pair in pairs
    }
    # A judged dimension's scale bounds the interval on its delta; it declares no margin, so no equivalence reads it.
    judged_ranges = {judged.name: judged.value_range for judged in judged_measures if judged.value_range is not None}
    families = []
    for question_id, axes in scopes:
        readings = _family_readings(axes, catalog, judged_measures)
        if question_id in pass_rate_questions:
            readings[("trial", TRIAL_PASS_NAME)] = True
        tested: list[_Tested] = []
        for reading in sorted(readings):
            for control_key, contrast_key in sorted(pairs, key=lambda pair: (pair[0][1], pair[1][0])):
                control_values = values[control_key].get(reading, {})
                contrast_values = values[contrast_key].get(reading, {})
                if not control_values and not contrast_values:
                    continue
                if reading[0] == "trial" and not (control_values and contrast_values):
                    # A side with no usable graded trial has no pass rate; the scoreboard says why, not a contrast.
                    continue
                threshold, margin_source = _margin_of(reading, catalog, run_margins or {})
                one = _compare(
                    reading,
                    readings[reading],
                    (control_key, control_values),
                    (contrast_key, contrast_values),
                    threshold=threshold,
                    margin_source=margin_source,
                    value_range=(
                        catalog[reading[1]].value_range
                        if reading[0] == "measure"
                        else _TRIAL_PASS_RANGE
                        if reading[0] == "trial"
                        else judged_ranges.get(reading[1])
                    ),
                    no_turn=tuple(
                        side
                        for side, key in (("control", control_key), ("arm", contrast_key))
                        if reading[0] == "measure"
                        and summary_population(catalog[reading[1]], "scored") == "delivered"
                        and _took_no_turn(results_by_cell[key])
                    ),
                    contended=_contended_sides(reading, (control_key, contrast_key), catalog, contended),
                )
                confounds = pair_confounds[(control_key, contrast_key)]
                tested.append(
                    one._replace(comparison=one.comparison.model_copy(update={"mechanism_confounds": confounds}))
                )
        families.append(_corrected_family(question_id, axes, tested))
    checks = [
        _guardrail_check(
            reading,
            guardrail_readings[reading],
            (control_key, values[control_key].get(reading, {})),
            (contrast_key, values[contrast_key].get(reading, {})),
            contended=_contended_sides(reading, (control_key, contrast_key), catalog, contended),
        )
        for reading in sorted(guardrail_readings)
        for control_key, contrast_key in sorted(pairs, key=lambda pair: (pair[0][1], pair[1][0]))
        if values[control_key].get(reading) or values[contrast_key].get(reading)
    ]
    return MultipleComparisons(families=families), _guardrails_of(guardrail_readings, checks)


def _reading_scope(
    declared: CampaignDesign | None,
    catalog: dict[str, MetricDescriptor],
    judged_measures: list[JudgedMeasure],
    *,
    first_run_at: str | None,
) -> ReadingScope:
    """Label the readings no question asked before the first run asks about exploratory — or, with none, say so once.

    The same rule the families are scoped by (:func:`~threetears.evals.kernel.declaration.exploratory_reading`, over
    :func:`~threetears.evals.kernel.declaration.preregistered_questions`), so a reading is exploratory exactly when no
    question's family could test it, and a question declared after the first run is named as post hoc.
    """
    questions = preregistered_questions(declared, first_run_at) if declared is not None else []
    asked_before = {question.id for question in questions}
    post_hoc = (
        [question.id for question in declared.live_questions() if question.id not in asked_before]
        if declared is not None
        else []
    )
    if not questions:
        return ReadingScope(
            questions_declared=False,
            post_hoc_questions=post_hoc,
            disclosure=exploratory_disclosure(declared, first_run_at),
        )
    return ReadingScope(
        questions_declared=True,
        post_hoc_questions=post_hoc,
        exploratory_measures=sorted(
            name
            for name, descriptor in catalog.items()
            if not descriptor.guardrail and exploratory_reading(descriptor.merit_axis, questions)
        ),
        exploratory_dimensions=sorted(
            measure.name
            for measure in judged_measures
            if measure.axis == "capability" and exploratory_reading(JUDGED_MERIT_AXIS, questions)
        ),
    )


#: Why no guardrail was checked, where some reading is one.
_NO_CONTROL_TO_HOLD_AGAINST = (
    "No control resolved, so there is no arm to hold another against and no guardrail was checked: every guardrail "
    "here is unchecked, which is not the same as held."
)


class _Guardrail(NamedTuple):
    """What a guardrail check needs to know about its reading."""

    higher_is_better: bool
    #: The declared margin — a measure's ``materiality_threshold``, a judged dimension's from the campaign's
    #: ``guardrail_margins`` — or None when none is declared.
    margin: float | None
    value_range: tuple[float, float] | None


def _guardrail_readings(
    catalog: dict[str, MetricDescriptor], judged_measures: list[JudgedMeasure], declared: CampaignDesign | None
) -> dict[tuple[ReadingKind, str], _Guardrail]:
    """Every guardrail reading the bundle carries: the measures declared one and the boundary judged dimensions.

    A measure's margin is its declared ``materiality_threshold``, the one margin a measure has; a judged
    dimension's is the one the campaign declares for it (``CampaignDesign.guardrail_margins``), and with none
    it is held at zero change.
    """
    judged_margins = {entry.dimension: entry.margin for entry in declared.guardrail_margins} if declared else {}
    readings: dict[tuple[ReadingKind, str], _Guardrail] = {
        ("measure", name): _Guardrail(
            descriptor.higher_is_better, descriptor.materiality_threshold, descriptor.value_range
        )
        for name, descriptor in catalog.items()
        if descriptor.guardrail and descriptor.higher_is_better is not None
    }
    for measure in judged_measures:
        if measure.axis == "boundary":
            readings[("judged", measure.name)] = _Guardrail(
                measure.higher_is_better, judged_margins.get(measure.name), measure.value_range
            )
    return readings


def _contended_sides(
    reading: tuple[ComparisonReading, str],
    pair: tuple[_CellKey, _CellKey],
    catalog: Mapping[str, MetricDescriptor],
    contended: Collection[_CellKey],
) -> tuple[str, ...]:
    """Which sides of a comparison on ``reading`` held latency read under concurrency: ``control``, ``arm``, both, none.

    Only a reading of elapsed time can have lost anything to the withholding, so any other reading names none.
    """
    if reading[0] != "measure" or reading[1] not in catalog or not is_latency_measure(catalog[reading[1]]):
        return ()
    return tuple(side for side, key in zip(("control", "arm"), pair, strict=True) if key in contended)


def _guardrail_check(
    reading: tuple[ReadingKind, str],
    guardrail: _Guardrail,
    control: tuple[_CellKey, dict[str, float]],
    contrast: tuple[_CellKey, dict[str, float]],
    *,
    contended: tuple[str, ...] = (),
) -> GuardrailCheck:
    """Decide one guardrail for one arm against the control under one rig (:func:`~threetears.evals.analysis.stats.guardrail_decision`).

    The samples are the ones a comparison on the same reading would read (:func:`_test_samples`), so a
    guardrail and a comparison never disagree about which cases were compared. An undecided check says
    why, in words that point at the remedy: more cases for a thin side, a declared range for a reading
    that declares none (it is never read ``held`` without one), and for a straddling interval the line it
    straddles.
    """
    (control_key, control_values), (contrast_key, contrast_values) = control, contrast
    a, b, paired = _test_samples(control_values, contrast_values)
    verdict = guardrail_decision(
        a,
        b,
        paired=paired,
        margin=guardrail.margin,
        higher_is_better=guardrail.higher_is_better,
        value_range=guardrail.value_range,
    )
    mean_a = sum(a) / len(a) if a else None
    mean_b = sum(b) / len(b) if b else None
    margin = guardrail.margin or 0.0
    reason = None
    if verdict.decision == "undecided":
        if contended and (len(a) < 2 or len(b) < 2):
            reason = (
                f"the {' and the '.join(contended)}'s latency was read while other cells or runs executed beside it, "
                "so it is withheld and the guardrail could not be checked on it (`latency_contended`)"
            )
        elif not b:
            reason = "the arm carries no value of it, so it was not checked against the control"
        elif not a:
            reason = "the control carries no value of it, so the arm could not be checked against one"
        elif len(a) < 2 or len(b) < 2:
            reason = "fewer than two cases carry it on a side, so no interval on the difference exists"
        elif verdict.refusal is not None:
            reason = verdict.refusal
        elif verdict.interval is None:
            reason = (
                "every shared case moved by the same amount, so a t interval has no width; "
                if paired
                else "each side's values are constant, so a t interval has no width; "
            ) + GUARDRAIL_HELD_NEEDS_RANGE
        else:
            line = -margin if guardrail.higher_is_better else margin
            reason = (
                f"the interval on the difference, [{format_number(verdict.interval[0])}, "
                f"{format_number(verdict.interval[1])}], reaches both sides of {format_number(line)}"
                + (" (the declared margin)" if guardrail.margin else " (no change: no margin is declared)")
                + ", so the arm is shown neither within it nor beyond it"
            )
    return GuardrailCheck(
        reading=reading[0],
        name=reading[1],
        higher_is_better=guardrail.higher_is_better,
        control=GuardrailCell(
            variant_key=control_key[0], apparatus_class_id=control_key[1], n_cases=len(a), mean=mean_a
        ),
        contrast=GuardrailCell(
            variant_key=contrast_key[0], apparatus_class_id=contrast_key[1], n_cases=len(b), mean=mean_b
        ),
        test=("paired" if paired else "unpaired") if verdict.interval is not None else None,
        delta=None if mean_a is None or mean_b is None else mean_b - mean_a,
        interval=verdict.interval,
        interval_basis=verdict.basis,
        margin=margin,
        margin_declared=guardrail.margin is not None,
        decision=verdict.decision,
        undecided_reason=reason,
    )


def _guardrails_of(
    readings: dict[tuple[ReadingKind, str], _Guardrail],
    checks: list[GuardrailCheck],
    *,
    withheld: str | None = None,
) -> GuardrailReadings:
    """The bundle's guardrail section, from the readings that are guardrails and the checks run on them."""
    return GuardrailReadings(
        measures=sorted(name for kind, name in readings if kind == "measure"),
        dimensions=sorted(name for kind, name in readings if kind == "judged"),
        checks=checks,
        withheld=withheld,
    )


__all__ = [
    "planning_readings",
    "PlanningReading",
]

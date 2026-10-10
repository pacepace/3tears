"""History: per-measure longitudinal series, with honest regression flags.

:func:`compute_history` series one measure per contestant across its runs, labels suite epochs by case set, and
flags a regression only where the test supports one (:class:`RegressionFlag`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

from threetears.evals.analysis.contention import (
    contended_latency_sentence,
    withheld_latency,
    withhold_contended_latency,
)
from threetears.evals.analysis.stats import ChangeLabel
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.kernel.result_condition import (
    counted_score,
    delivered_a_turn,
)
from threetears.evals.kernel.scoring import (
    CompositeBasis,
    result_composite,
)
from threetears.observe import get_logger
from threetears.evals.analysis.completeness import completeness_disclosure
from threetears.evals.analysis.reporting import (
    METRIC_COMPOSITE,
    METRIC_COST_USD,
    METRIC_OUTCOME,
    METRIC_SCORE,
    METRIC_TOTAL_MS,
    METRIC_TRANSCRIPT,
    place_results,
    PlacedResult,
    pooled_composite_basis,
    pooled_cost_compositions,
    pooled_served_models,
    PROJECTED_METRICS,
    ProjectionExclusions,
    SCOPED_METRICS,
    ServedModelReading,
)
from threetears.evals.analysis.lenses.aggregation import (
    _AGGREGATE_OF_OBSERVATION,
    _describe_aggregate,
    _effective_formula,
    _METRIC_GLOSS,
    _metric_vocabulary,
    resolve_measure_name,
    WEIGHTING_EQUAL_PER_SCENARIO,
)
from threetears.evals.analysis.lenses.contestants import (
    _contestant_key,
    _identity_span_disclosure,
    _identity_version_disclosure,
    _identity_version_span,
    ContestantKey,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from threetears.evals.schema.models import (
        CaseSetRef,
        EvalResult,
        EvalRun,
    )


log = get_logger(__name__)


# The measures `history` can series. Composite, cost and the two dual-score axes are also
# `PROJECTED_METRICS` members; `total_ms` is not projected at all and is read straight off
# `EvalResult.latency`. Every member is numeric with a defined better direction, which
# is what the regression flag needs to call a move a decline or a gain — a directionless
# coordinate could not be "regressed". A metric outside this set is refused, not answered
# with an empty series, for the same reason `pivot` refuses an unknown measure.
#
# The dual-score axes are here because the read the catalog says the PAIR exists for is a
# read over time: `__outcome__` falling while `__transcript__` holds is a world that
# changed, and both falling together is a candidate that got worse. `pivot` answers that
# within one campaign and this is the surface that answers it across runs. The members of
# `PROJECTED_METRICS` that stay refused are the SCOPED ones (`SCOPED_METRICS`): each row is one
# dimension or one check, so there is no single per-run value to plot, while each axis above is
# one value per result.
#: The measures `history` can series; any other is refused rather than answered with an empty series.
HISTORY_METRICS = frozenset({METRIC_COMPOSITE, METRIC_COST_USD, METRIC_TOTAL_MS, METRIC_TRANSCRIPT, METRIC_OUTCOME})

if set(_METRIC_GLOSS) != PROJECTED_METRICS | HISTORY_METRICS:  # pragma: no cover - import-time invariant
    raise RuntimeError(
        f"_METRIC_GLOSS must gloss exactly the accepted metrics: {sorted(set(_METRIC_GLOSS) ^ (PROJECTED_METRICS | HISTORY_METRICS))}"
    )


class HistoryError(ValueError):
    """A history was requested for a measure it cannot honestly series.

    Raised rather than returned as an empty series, for the reason
    :class:`PivotError` is: an empty series is indistinguishable from a measure
    nobody has recorded, so a refusal returned as data becomes a silent nothing.
    """


def _unknown_history_metric(metric: str) -> str:
    """Say why a measure cannot be seriesed, and where the caller should go instead.

    The single owner of that sentence — it is raised from two places, and two copies
    of one refusal drift.

    ``score`` earns its own arm rather than being enumerated past. It is not a typo:
    it is a registered measure that ``results_pivot`` accepts, ``export_results``
    emits and the catalog describes, so a caller reaching for it here asked a
    coherent question this surface cannot answer. A series carries ONE value per
    contestant per run and a raw judge score is per rubric DIMENSION, so there is no
    single per-run score to plot — and a message that only listed the three
    alternatives left the caller to infer that, which reads as "not implemented yet".

    Args:
        metric: The measure the caller asked to series.

    Returns:
        The refusal text, ready to raise as a :class:`HistoryError`.
    """
    expected = f"expected one of {_metric_vocabulary(HISTORY_METRICS)}"
    resolved = resolve_measure_name(metric)
    if resolved in SCOPED_METRICS and resolved != METRIC_SCORE:
        scope_field, scope_noun, _row_noun = SCOPED_METRICS[resolved]
        return (
            f"history cannot series {resolved!r} — each row is ONE {scope_noun}, so there is no single per-run value "
            f"to plot. Use the pivot surface with '{scope_field}' on an axis, which aggregates it as "
            f"{_AGGREGATE_OF_OBSERVATION.get(resolved, resolved)!r}. {expected}."
        )
    # Resolved for the BRANCH, while `metric` stays the caller's spelling for the
    # message: `mean_score` and `score` are the same refusal and must not take
    # different arms just because the operator read the catalog.
    if resolve_measure_name(metric) == METRIC_SCORE:
        return (
            f"history cannot series {METRIC_SCORE!r} — it is the raw judge score for one rubric "
            "DIMENSION, and a series carries one value per contestant per run, so there is no single "
            "per-run score to plot. For a per-dimension read use the pivot surface with 'rubric_dim' "
            "on an axis, which aggregates it as 'mean_score'; to track judged quality over time, "
            f"series 'composite', or {METRIC_TRANSCRIPT!r} / {METRIC_OUTCOME!r} to stay on the raw 1-5 "
            "scale — each of those is one value per result, which is what this surface can plot. "
            f"For history, {expected}."
        )
    return f"unknown history metric {metric!r} — {expected}, or a numeric measure the host declares (a scorer's name)"


def _attribution_withheld(descriptor: MetricDescriptor) -> bool:
    """Say whether a labelled move in this measure may be read as the contestant's.

    Read off the measure's ``transferability_class`` rather than its name, so the rule
    is a property of the catalog and not a list of metrics kept in step by hand. A
    ``scenario_bound`` measure is defined by the scenario, and a series cannot hold the
    externals that scenario touches still between two runs — the registry says so of the
    live case in its own words, that it is comparable across runs "only where the
    scenarios and the externals they touch held still". So the move is real and its
    cause is not decided by observing it.

    The looser classes are not asserted to be the contestant's either; they are simply
    outside what this predicate refuses. It says only that a scenario-bound label may
    not be attributed.

    Args:
        descriptor: The measure the series reports — the AGGREGATE descriptor, since
            that is what a point holds.

    Returns:
        Whether a verdict on this measure must decline to attribute the move.
    """
    return descriptor.transferability_class == "scenario_bound"


def _attribution_disclosure(descriptor: MetricDescriptor) -> str | None:
    """The one sentence saying why this measure's verdicts do not attribute.

    Carried on the answer rather than composed at each surface, so REST, MCP and any
    later renderer say the same thing — two copies of one caveat drift, which is why
    :func:`_unknown_history_metric` is shared for its refusal.

    Args:
        descriptor: The aggregate descriptor the series reports.

    Returns:
        The sentence, or ``None`` when the measure's verdicts carry no such limit.
    """
    if not _attribution_withheld(descriptor):
        return None
    return (
        f"{descriptor.name} is scenario-bound: its value is defined by the scenario, and a series "
        "cannot hold the externals that scenario touches still between two runs. Every regression "
        "flag here states the move and withholds attribution — the label is not a finding about "
        "the contestant."
    )


class RegressionFlag(EvalBaseModel):
    """A descriptive verdict on the change from the previous point in the series.

    Descriptive, never an alert: it discloses the ``test`` it ran and the
    thresholds it applied, so the label can never be read as a calibrated
    judgement — automated alerting stays gated behind judge calibration. ``label``
    is one of ``regressed`` / ``improved`` / ``equivalent`` / ``below_threshold`` /
    ``not_separated`` / ``untested`` (:class:`~threetears.evals.analysis.stats.ChangeVerdict`
    defines each). A move earns a directional label only when it is both
    statistically significant and over a magnitude threshold (a joint gate). A move
    that misses significance reads ``not_separated``, never "no change": the one label
    claiming no meaningful change is ``equivalent``, and it needs an equivalence test
    against the measure's declared margin (``equivalence_margin``) to pass. With no
    margin declared, no step can read ``equivalent``.

    ``crosses_epoch`` warns that the two runs span a suite-version boundary: the
    paired test then rests only on the cases the two still share, and the
    aggregate delta is confounded by the changed denominator, so a flag here says
    less than one within an epoch.

    ``crosses_cassette_mode`` is that warning's sibling on the apparatus axis, and the
    stronger of the two. The two runs did not record the same cassette mode, so where the
    step involves ``replay`` one of them re-served a recording instead of calling the
    third party — the measure did not move between two measurements of one thing, it
    moved between a measurement and a substitution. Cassette mode is deliberately outside
    the variant key (it is apparatus, not product), so both runs sit in ONE series and
    nothing else in the flag can show it. Descriptive, exactly as ``crosses_epoch`` is: it
    suppresses no verdict and adjusts no delta.

    ``attribution_withheld`` is the third of that family and differs from the other two in
    what it is about: they qualify this PAIR of runs, while it is a property of the measure
    and so holds for every step in the series. It fires where the measure is scenario-bound
    (:func:`_attribution_withheld`), and it neither suppresses the label nor adjusts the
    delta — a decline nobody can attribute is still worth seeing, and staying silent about
    it would be the reading this flag exists to prevent.
    """

    label: ChangeLabel
    delta: float | None = None
    relative_delta: float | None = None
    significant: bool | None = None
    exceeds_threshold: bool | None = None
    #: Hedges' g_z of the paired move — bias-corrected, so not comparable with a Cohen's d.
    hedges_g: float | None = None
    #: The p ``significant`` was thresholded against — the paired t's, or where every case
    #: moved by one amount the bounded test's on the measure's declared range — and ``None``
    #: on an ``untested`` step and where ``not_separated_reason`` says no test ran. Carried
    #: for the same reason ``hedges_g`` is: a verdict whose statistic is absent cannot be
    #: checked. A flag computed before 0.66 carried the exact sign-flip p there, a test of
    #: symmetry rather than of the mean, and its ``test`` says so.
    p: float | None = None
    #: The TOST p an ``equivalent`` label was thresholded against — the larger of the
    #: two one-sided p's, each the bounded test's on the measure's declared range — or
    #: ``None`` wherever no equivalence test ran: no margin declared, no range declared
    #: (``equivalence_untested_reason`` says so), fewer than two pairs, or a difference
    #: outside the declared range (:func:`~threetears.evals.analysis.stats.paired_equivalence`).
    equivalence_p: float | None = None
    #: The margin that test ran against, in the measure's units: the measure's declared
    #: materiality threshold, or ``None`` when it declares none.
    equivalence_margin: float | None = None
    #: Why a measure with a margin was not tested for equivalence at all: it declares no
    #: range (:data:`~threetears.evals.analysis.stats.EQUIVALENCE_NEEDS_RANGE`, which names the
    #: remedy), so no step on it can read ``equivalent``. ``None`` otherwise.
    equivalence_untested_reason: str | None = None
    #: Why the step reads ``not_separated`` with no test run: every case moved by the same
    #: amount on a measure that declares no range
    #: (:data:`~threetears.evals.analysis.stats.UNIFORM_MOVE_NEEDS_RANGE`, which names the
    #: remedy), or a value lies outside it. ``None`` otherwise.
    not_separated_reason: str | None = None
    n_pairs: int = 0
    crosses_epoch: bool = False
    crosses_cassette_mode: bool = False
    #: This measure's move cannot be attributed to the contestant. Set on the FLAG, not
    #: only on the answer, because the label is what a reader would otherwise take as the
    #: attribution — and a flag is routinely read as one row rather than beside the header.
    #: The sentence is :attr:`HistoryResult.attribution_disclosure`, stated once where the
    #: measure is described rather than repeated on every step.
    attribution_withheld: bool = False
    # Disclosure — the test and thresholds this verdict rests on, carried on every
    # flag so a reader never has to infer what "regressed" was measured against.
    test: str
    min_absolute_change: float
    min_relative_change: float


class SeriesPoint(EvalBaseModel):
    """One run's aggregate for the measure — the trend's unit, with its denominators.

    ``value`` is the equal-per-scenario mean over the run's cases; ``sem``, ``n``
    (observations that carried a value) and ``n_cases`` (distinct cases) travel
    with it because a point resting on 2 cases and one resting on 30 must not read
    alike. ``epoch`` is a 1-based ordinal that increments at each
    suite-version boundary, and ``epoch_boundary`` marks the point where the suite
    changed — so a step caused by the denominator changing does not read as a
    regression. ``delta_from_baseline`` is the descriptive red/green move from the
    series' first point; ``regression`` is the statistical verdict versus the
    immediately preceding point.
    """

    run_id: str
    created_at: str
    model: str
    value: float | None = None
    sem: float | None = None
    n: int = 0
    n_cases: int = 0
    epoch: int = 1
    epoch_boundary: bool = False
    #: The named case set every run in this point's epoch was launched against, as ``name vN``, or ``None`` when
    #: they were not all launched against one set (or any). A label on the epoch, which is still decided by the
    #: frozen case ids (:func:`_suite_epoch_key`): ``smoke v1`` then ``smoke v2`` is a boundary because the ids
    #: changed, and this says which named set changed into which.
    epoch_label: str | None = None
    is_baseline: bool = False
    delta_from_baseline: float | None = None
    regression: RegressionFlag | None = None
    #: On a COST series only: the role sets this point's dollars were summed over. It is
    #: here rather than on the series because a change lands between two points, and this
    #: is what says which two — the same job ``epoch_boundary`` does for the suite. A cost
    #: step caused by the composition moving is not a regression, and nothing else about
    #: the point can tell the two apart. Empty on every non-cost measure, which have no
    #: composition.
    cost_compositions: list[list[str]] = []
    #: On a COMPOSITE series only: what this point's composites were meaned over (#638), ``ragged`` when the
    #: run's results carried different dimension sets. Per point for the reason ``cost_compositions`` is: a
    #: step between two points meaned over different sets is a change in what was averaged, not a regression.
    #: ``None`` on every other measure and on a point with no composite.
    composite_basis: CompositeBasis | None = None
    #: This point's run's RECORDED cassette mode. Carried per point rather than once per
    #: series because it can move BETWEEN points — which is the whole defect: cassette
    #: mode is outside the variant key, so a capture run and a replay run of one
    #: contestant are two points of one series and the step between them reads as a
    #: quality change. A recorded mode is a claim, not an observation; see
    #: :func:`cassette_mode_disclosure`.
    cassette_mode: str | None = None
    #: This point's run's DEGRADED sentence, or ``None`` when it delivered the whole
    #: matrix it promised. A point IS one run, so unlike every other pooling surface
    #: this one can attribute the shortfall exactly — and it needs to most: the point
    #: sits on a time series beside complete runs and is handed a regression verdict
    #: against its neighbour, so a run that measured 2 of its 4 cells contributes a
    #: ``not_separated`` or ``regressed`` label computed over a denominator the comparison does
    #: not share. Derived from the run's completeness record rather than its status: a
    #: ``completed`` run with an infra-excluded cell is short too.
    completeness_disclosure: str | None = None
    #: Which models the provider's responses named as having answered this run's candidate calls (#684). Per point
    #: because a floating alias can resolve to a different model between two runs of one contestant, and the step
    #: between them then reads as a change in the contestant. ``None`` when no result's candidate made a call.
    served_models: ServedModelReading | None = None


class MeasureSeries(EvalBaseModel):
    """One contestant's measure over time, within a subject.

    A series is one ``(subject, variant, predicate version)``: the same resolved
    contestant tracked across its runs, so a regression is a real re-run decline rather
    than an artefact of pooling two configs. ``model`` is the human label;
    ``variant_key`` is the identity.

    An identity-version bump **splits** a series rather than marking an epoch inside one,
    which is the opposite of how a suite change is handled (:func:`_suite_epoch_key`) and is
    deliberate. A suite epoch keeps one contestant whole across a change to what it was
    scored against; an identity bump may have moved what the contestant *is*, and the stamp
    cannot say whether it did — so holding both versions in one series would assert a
    continuity nothing can verify, in the one place a reader reads continuity.
    """

    subject_id: str
    subject_label: str = ""
    variant_key: str
    #: The predicate version that minted ``variant_key``. Carried onto the row because
    #: this surface GATES on it — two keys carrying different stamps are ranked separately,
    #: because the stamp cannot say which predicate moved and a wrong merge is the direction
    #: nothing downstream undoes — and a reader who meets that split needs the number that
    #: caused it.
    variant_identity_version: int
    #: Set when that predicate is not this build's.
    identity_version_disclosure: str | None = None
    model: str
    points: list[SeriesPoint] = []
    #: Which models answered the candidate calls across the whole series (#684): ``pooled`` where one requested
    #: model was answered by several over its runs, so the series tracks a mixture rather than one model — each
    #: point says which answered it. ``None`` when no result's candidate made a call.
    served_models: ServedModelReading | None = None


class HistoryResult(EvalBaseModel):
    """Per-measure longitudinal series across contestants, with its disclosures.

    ``formula`` and ``weighting`` state what each point averages; ``higher_is_better``
    carries the measure's direction so a reader knows which way is a regression.
    ``min_absolute_change`` / ``min_relative_change`` echo the caller's regression
    gate (the thresholds are the caller's, disclosed, never invented), and the
    per-flag ``test`` names the statistic. ``equivalence_margin`` is the host's
    declared margin on the measure, which with a declared range is what lets a step read
    ``equivalent``; ``None`` when it declares none. A margin with no range is never tested,
    and each flag carries the reason. ``attribution_disclosure`` is the same
    obligation one rung up: on a measure whose verdicts cannot name a cause, it says so
    once for the answer. ``exclusions`` and the ``n_*`` counts keep
    an all-excluded corpus from rendering as an empty one, exactly as
    :class:`PivotTable` and :class:`FrontierResult` do.
    """

    metric: str
    measure: MetricDescriptor
    formula: str
    weighting: str
    higher_is_better: bool | None = None
    min_absolute_change: float
    min_relative_change: float
    equivalence_margin: float | None = None
    series: list[MeasureSeries] = []
    n_results: int = 0
    n_filtered_out: int = 0
    exclusions: ProjectionExclusions = ProjectionExclusions()
    #: ``run_id -> DEGRADED sentence`` for every run on these series that came up short
    #: of its matrix. The per-point copies mark which rows; this is the set, so a surface
    #: can state the rule once rather than per point.
    completeness_disclosures: dict[str, str] = {}
    #: How many of ``n_results`` came from those runs — the weight of the caveat.
    n_degraded_observations: int = 0
    #: The distinct identity versions this answer's keyed observations were stamped at,
    #: ascending. Partitioning stops two stampings of one stack
    #: being RANKED together; it cannot stop them appearing as two rows, so the span and
    #: the sentence below are what keep a doubled row from reading as a mystery.
    identity_version_span: list[int] = []
    #: The one-sentence form of that span, or ``None`` when the answer rests on a single
    #: predicate. Computed from the placed rows rather than from the assembled
    #: series, so an answer whose series were all filtered out still says what
    #: it spanned.
    identity_span_disclosure: str | None = None
    #: Why this measure's regression flags decline to attribute their move, or ``None``
    #: where they carry no such limit. A property of the MEASURE, so it is stated once
    #: here beside ``measure`` and ``formula`` rather than repeated on every flag; each
    #: flag carries the boolean (:attr:`RegressionFlag.attribution_withheld`) so a verdict
    #: read alone still declares its posture. Derived before any series is assembled, so
    #: an answer with no series at all still carries it.
    attribution_disclosure: str | None = None
    #: That latency read under concurrency was left out of a latency series — no step of it reads a
    #: contended latency — and how much; ``None`` for any other measure, and when every latency in the
    #: series was read serially (:mod:`~threetears.evals.analysis.contention`).
    contended_latency_disclosure: str | None = None


def _history_value_of(metric: str) -> Callable[[EvalResult], float | None]:
    """Pick the per-result value extractor for a history measure.

    Args:
        metric: A measure in :data:`HISTORY_METRICS`.

    Returns:
        A callable reading the measure off one result, ``None`` when this result
        did not carry it — a null that the aggregation drops rather than counts as
        zero, so an unmeasured latency never reads as instant.

    Raises:
        HistoryError: The metric is not one this surface can series. Unreachable
            when :func:`compute_history` validated first; kept as the honest total answer.
    """
    if metric == METRIC_COMPOSITE:
        return result_composite
    if metric == METRIC_COST_USD:
        # Measuring spend, so every dollar the program spent: the population program spend keeps on every
        # surface that reads it — the cost pivot, a run summary's `mean_cost_usd`
        # (:func:`~threetears.evals.kernel.scoring.compute_cost_summary`) and the budget view. A call the
        # model refused before any turn was still billed, and a cell the harness faulted spent what it spent.
        # Leaving the refusal out while the pivot kept it gave one corpus two figures for one quantity. What an
        # arm COSTS reads only the turns taken, and is `production_replicating_cost`, which no series offers.
        return lambda result: result.cost_usd
    if metric == METRIC_TOTAL_MS:
        # Infra-excluded cells are withheld here for the reason they are on the frontier's
        # latency: an apparatus fault produces a REAL but truncated `LatencyMetrics`, and this
        # series issues regression verdicts. Composite already drops them (via `result_composite`
        # returning None), so leaving latency in made the two metrics on one surface answer
        # different questions — and a cassette miss could post a "faster" step that describes the
        # harness. A call the model refused or errored on took no turn, and is withheld for the frontier's
        # reason: `delivered_a_turn`, the one predicate every latency reading uses. Measuring spend keeps
        # both, because those dollars were spent (see METRIC_COST_USD above).
        return lambda result: (
            result.latency.total_ms
            if result.latency is not None and result.latency.total_ms is not None and delivered_a_turn(result)
            else None
        )
    if metric in (METRIC_TRANSCRIPT, METRIC_OUTCOME):

        def dual_axis_value(result: EvalResult) -> float | None:
            """Read one dual-score axis off a result, or ``None`` where it is not a measurement.

            Args:
                result: The observation to read.

            Returns:
                The raw 1-5 score, or ``None`` when the axis was not judged, was
                mis-stamped, or the cell was an apparatus failure.
            """
            # Named attributes rather than a computed one: this pair is read the same way
            # in `project_score_records`, and a `getattr` by string puts a rename of either
            # field out of the type checker's reach.
            axis = result.transcript_score if metric == METRIC_TRANSCRIPT else result.outcome_score
            if axis is None:
                return None
            # The guard the projection applies to these two fields, for its reason and one
            # tighter. `RubricScore.dim` is stored free text, so a mis-stamped row would
            # otherwise be meaned into a series and handed a regression verdict while `pivot`
            # and `export_results` drop it and warn — the two surfaces disagreeing about what
            # an axis is. Tighter because a series asks for ONE axis: the projection needs
            # only "is this a reserved id" to route the row to a metric, while here a
            # transcript score sitting in `outcome_score` would be a reserved id landing in
            # the wrong series. Loud there and loud here, because the absence it leaves
            # already means something else: an unjudged result carries no axis either, so a
            # silent skip makes a defect read as an unjudged run.
            if axis.dim != metric:
                log.warning(
                    "eval.reporting dropping a dual-score observation for result=%s from the %s series: dim %r is not "
                    "that axis. The axis was scored and is NOT in this series — its absence is a defect in what "
                    "stamped it, not a result that went unjudged.",
                    result.id,
                    metric,
                    axis.dim,
                )
                return None
            # The rule every judged score follows (`counted_score`), so one axis means one thing
            # on every surface: an infra-excluded cell is withheld — a judge reading a broken
            # transcript is not measuring the candidate, and this surface issues a regression
            # verdict on what it returns — and a candidate failure counts at the scale's floor.
            counted_axis = counted_score(result, axis)
            return None if counted_axis is None else float(counted_axis)

        return dual_axis_value
    raise HistoryError(_unknown_history_metric(metric))


def _host_measure_value_of(descriptor: MetricDescriptor) -> Callable[[EvalResult], float | None]:
    """Pick the per-result reader for a measure the host declared, such as a quick-path scorer's.

    A host measure is landed on each result's ``host_measures`` (a scorer's grade on the quick path), so it
    has a per-result reading, and a series of its per-case means is the same quantity a comparison of it
    pools. Each result is read over the population every other surface reads the measure over
    (``summary_population``, ``"scored"`` when undeclared, through the bundle's own membership rule), so a
    cell the harness faulted is never in it. A ``bool`` counts as 1 or 0.

    Args:
        descriptor: The host's descriptor of the measure.

    Returns:
        The per-result reader, ``None`` where the result carries no number for the measure.

    Raises:
        HistoryError: The measure is not numeric, or declares no direction — a series could not say which
            way a step is a decline.
    """
    from threetears.evals.analysis.bundle.measures import in_population
    from threetears.evals.kernel.metrics import summary_population

    name = descriptor.name
    if descriptor.data_type != "numeric" or descriptor.higher_is_better is None:
        raise HistoryError(
            f"history cannot series {name!r} — the host declares it "
            + ("with no direction" if descriptor.data_type == "numeric" else f"as {descriptor.data_type or 'untyped'}")
            + ", and a series flags a step as a decline only on a numeric measure whose better direction is "
            "declared (higher_is_better)"
        )
    population = summary_population(descriptor, "scored")

    def host_measure_value(result: EvalResult) -> float | None:
        """Read the declared measure off one result, or ``None`` where it is not an observation of it."""
        value = result.host_measures.get(name)
        if value is None or isinstance(value, str) or not in_population(population, result):
            return None
        return float(value)

    return host_measure_value


def _per_case_means(
    results: list[EvalResult], value_of: Callable[[EvalResult], float | None]
) -> tuple[dict[str, float], int]:
    """Collapse one run's observations to one value per case (k averaged first).

    Averaging k iterations before the case enters the series is what makes each
    case weigh equally under equal-per-scenario, matching how pass^k and the
    composite summary count. Observations with no value for the measure are
    dropped, so a case counts only where it was actually measured.

    Args:
        results: One contestant's results within one run.
        value_of: The measure extractor from :func:`_history_value_of`.

    Returns:
        ``(per_case_mean, n_observations)`` — the per-case means keyed by
        ``test_case_id``, and how many observations carried a value.
    """
    by_case: dict[str, list[float]] = {}
    for result in results:
        value = value_of(result)
        if value is not None:
            by_case.setdefault(result.test_case_id, []).append(float(value))
    n_observations = sum(len(values) for values in by_case.values())
    per_case = {case_id: sum(values) / len(values) for case_id, values in by_case.items()}
    return per_case, n_observations


def _suite_epoch_key(run: EvalRun) -> tuple[str, tuple[str, ...]]:
    """The suite-version identity of a run: its template and frozen case set.

    Two runs are the same suite version iff they score the same template against
    the same test cases — exactly what ``context_components.case_basis`` digests,
    derived here straight from the run's own fields so it is defined for every
    run, stamped or not. A change in this key between two
    adjacent runs is a suite-version boundary: an aggregate step across it is the
    denominator changing, not necessarily the contestant regressing.

    Args:
        run: The run whose suite identity to key.

    Returns:
        ``(template_id, sorted test_case_ids)`` — the empty string stands in for an
        ad-hoc run's absent template so ad-hoc runs still key stably.
    """
    return (run.template_id or "", tuple(sorted(run.test_case_ids)))


def _label_epochs_by_case_set(points: list[SeriesPoint], runs: Mapping[str, EvalRun]) -> None:
    """Label each epoch ``name vN`` where every run in it was launched against that one case set.

    An epoch mixing runs launched against a set with runs launched without one — or against two versions
    holding identical ids — keeps no label: the name would describe only some of its points.

    Args:
        points: One series' points, in order, epochs already numbered.
        runs: The series' runs by id.
    """
    by_epoch: dict[int, set[CaseSetRef | None]] = {}
    for point in points:
        by_epoch.setdefault(point.epoch, set()).add(runs[point.run_id].case_set)
    for point in points:
        refs = by_epoch[point.epoch]
        only = next(iter(refs)) if len(refs) == 1 else None
        point.epoch_label = only.label if only is not None else None


def compute_history(
    runs: list[EvalRun],
    results: list[EvalResult],
    *,
    metric: str = METRIC_COMPOSITE,
    min_absolute_change: float = 0.0,
    min_relative_change: float = 0.0,
    subject_id: str | None = None,
    known_run_ids: set[str] | None = None,
    archived_run_ids: set[str] | None,
    profile: HostProfile,
) -> HistoryResult:
    """Series one measure over time per contestant, flagging real regressions.

    Each ``(subject, variant, identity version)`` contestant becomes a series; each of its runs
    becomes a time-ordered point carrying the measure's equal-per-scenario mean,
    its dispersion, and its denominators. The series is split into suite-version
    epochs (:func:`_suite_epoch_key`) so a step caused by the test set changing
    reads as an epoch boundary rather than a mysterious jump. Between adjacent
    points the change is classified by a paired test on the cases they share plus
    ``min_*_change`` magnitude thresholds (:func:`~threetears.evals.analysis.stats.paired_change`),
    and, where the measure declares a materiality threshold, an equivalence test
    against it; the tests and thresholds ride on every flag — descriptive, never an
    alert, until judge calibration lands.

    **On a scenario-bound measure the flag fires and withholds attribution.** Its value
    is defined by the scenario, whose externals no series can hold still between runs, so
    the move is real and its cause is not decided by observing it — see
    :func:`_attribution_withheld`. Staying silent instead would hide a decline; labelling
    it unqualified would name a culprit the measurement does not identify.

    Subjects are never pooled: composite quality is derived from each
    subject's own rubric, so two subjects are two sets of series. The two skips
    :func:`place_results` makes are returned as :class:`ProjectionExclusions`, so
    an all-excluded corpus discloses that its data exists but cannot be grouped
    rather than rendering as an empty one.

    **A run that measured less than its matrix is kept on the series and marked**,
    for the reason :func:`compute_frontier` gives for admitting one: its cells are real
    measurements of the same contestant, and dropping a point would put a gap in a
    time series where a run demonstrably happened. What it costs is disclosed on the
    point itself (:attr:`SeriesPoint.completeness_disclosure`), which is the closest
    any pooling surface can get to attributing it — a point IS one run. It matters
    here because the point is not merely averaged but *compared*: it receives a
    regression verdict against its neighbour, computed over a denominator the two do
    not share. The predicate is the completeness record, never the status.

    Args:
        runs: The runs supplying subject identity, suite set, and timestamps.
        results: The observations to series.
        metric: A measure in :data:`HISTORY_METRICS`, or the registry name of its
            aggregate as ``list_metrics`` publishes it (``mean_composite`` for
            ``composite``, and so on through
            :data:`_AGGREGATE_OF_OBSERVATION`) — see :func:`resolve_measure_name` — or a
            numeric, directional measure the host declares (a quick-path scorer's name),
            read off each result's ``host_measures``. Defaults to composite quality.
        min_absolute_change: Smallest absolute move that counts as a regression,
            in the measure's own unit. ``0.0`` lets significance alone flag.
        min_relative_change: Smallest move relative to the baseline that counts, as
            a fraction. ``0.0`` lets significance alone flag.
        subject_id: When set, restrict to this subject; others are counted as
            ``n_filtered_out`` rather than dropped silently.
        known_run_ids: Every run id in the corpus, so a result excluded by the
            caller's own run filter is counted filtered-on-request rather than
            unplaceable. See :func:`project_score_records`.
        archived_run_ids: The corpus run ids the operator has archived, so an
            archival exclusion is counted as ``results_from_archived_runs``
            rather than under the caller's ``status`` filter. See
            :func:`place_results`.
        profile: The host whose vocabulary this reads.

    Returns:
        A :class:`HistoryResult` — one :class:`MeasureSeries` per contestant,
        sorted by subject then model then variant then identity version, plus corpus-level accounting.

    Raises:
        HistoryError: ``metric`` is not one this surface can series — refused
            before any row is read, so a typo cannot return an empty series that
            reads like a measure nobody recorded.
    """
    from threetears.evals.analysis.stats import (
        EQUIVALENCE_TEST_NAME,
        PAIRED_TEST_NAME,
        paired_change,
        standard_error_of_mean,
    )

    # A series point is a mean over cases, so the catalog name for what this returns
    # is `mean_composite` / `mean_cost_usd` / `mean_total_ms` — accepted here beside
    # the row names, resolved before the closed-set check for the reason `pivot`
    # resolves before its own.
    # The caller's own spelling reaches the refusal — see `pivot` for why the resolved
    # name is the wrong thing to quote back.
    requested = metric
    metric = resolve_measure_name(metric)
    declared = None if metric in HISTORY_METRICS else profile.measures.get(metric)
    if metric not in HISTORY_METRICS and declared is None:
        raise HistoryError(_unknown_history_metric(requested))

    if declared is not None:
        value_of = _host_measure_value_of(declared)
        descriptor = declared
    else:
        value_of = _history_value_of(metric)
        descriptor = _describe_aggregate(metric, profile=profile)
    # Every HISTORY_METRICS member is directional, and a host measure without a direction was refused; the guard keeps the type honest without
    # asserting an impossible None away.
    direction = descriptor.higher_is_better if descriptor.higher_is_better is not None else True
    # Decided from the descriptor once, before any row is read, so a corpus that yields no
    # series still says what its verdicts would have withheld — the empty-guard swallow the
    # sibling disclosures on this surface are already assembled ahead of.
    attribution_withheld = _attribution_withheld(descriptor)
    # The host's declared margin on the measure — the one margin a bar is read against too. Only it
    # licenses an `equivalent` step; the caller's gate never does.
    margin = descriptor.materiality_threshold
    flag_test = PAIRED_TEST_NAME if margin is None else f"{PAIRED_TEST_NAME}; {EQUIVALENCE_TEST_NAME}"

    # A latency series never steps across a contended reading: latency read under concurrency is removed
    # before any point is built, so a regression flag compares only latency taken serially (#701).
    contended = (
        {result.id for result in withheld_latency(results, profile.measures)} if metric == METRIC_TOTAL_MS else set()
    )
    if contended:
        results = withhold_contended_latency(results, profile.measures)

    placed, exclusions = place_results(
        runs, results, known_run_ids, source="history", archived_run_ids=archived_run_ids
    )

    series_groups: dict[tuple[str, ContestantKey], list[PlacedResult]] = {}
    subject_labels: dict[str, str] = {}
    # The short runs on these series. Keyed on the completeness record rather than on
    # status, because a `completed` run that lost a cell to a harness exclusion is short
    # too and reaches this series by default, where it is given a regression verdict.
    degraded_by_run: dict[str, str] = {}
    n_filtered_out = 0
    n_considered = 0
    n_degraded_observations = 0
    # The observations this answer rests on, narrowed to what survived the subject filter
    # — same population and same reason as `frontier`'s, so the two lenses cannot report
    # different spans over one corpus.
    considered: list[EvalResult] = []
    for row in placed:
        if subject_id is not None and row.subject_id != subject_id:
            n_filtered_out += 1
            continue
        n_considered += 1
        considered.append(row.result)
        series_groups.setdefault((row.subject_id, _contestant_key(row.result)), []).append(row)
        subject_labels[row.subject_id] = row.run.subject_snapshot.subject_label
        if (short := completeness_disclosure(row.run.completeness)) is not None:
            degraded_by_run[row.run.id] = short
            n_degraded_observations += 1

    series: list[MeasureSeries] = []
    for (resolved, contestant), rows in series_groups.items():
        runs_in_series: dict[str, EvalRun] = {}
        rows_by_run: dict[str, list[EvalResult]] = {}
        for row in rows:
            runs_in_series[row.run.id] = row.run
            rows_by_run.setdefault(row.run.id, []).append(row.result)
        ordered_run_ids = sorted(runs_in_series, key=lambda run_id: (runs_in_series[run_id].created_at, run_id))

        points: list[SeriesPoint] = []
        prev_epoch_key: tuple[str, tuple[str, ...]] | None = None
        prev_per_case: dict[str, float] = {}
        epoch_ordinal = 0
        baseline_value: float | None = None
        for index, run_id in enumerate(ordered_run_ids):
            run = runs_in_series[run_id]
            per_case, n_observations = _per_case_means(rows_by_run[run_id], value_of)
            case_means = list(per_case.values())
            value = math.fsum(case_means) / len(case_means) if case_means else None
            sem = standard_error_of_mean(case_means)

            epoch_key = _suite_epoch_key(run)
            boundary = index > 0 and epoch_key != prev_epoch_key
            if index == 0 or boundary:
                epoch_ordinal += 1

            regression: RegressionFlag | None = None
            if index > 0:
                # Compared against the immediately preceding run, because that is the pair
                # the flag is about. A series-level "these modes differ" would be true of
                # the whole series and would not say WHICH step crossed — the same reason
                # `crosses_epoch` is decided per adjacent pair rather than per series.
                crosses_cassette = run.cassette_mode != runs_in_series[ordered_run_ids[index - 1]].cassette_mode
                shared = sorted(set(per_case) & set(prev_per_case))
                verdict = paired_change(
                    [prev_per_case[case_id] for case_id in shared],
                    [per_case[case_id] for case_id in shared],
                    min_absolute_change=min_absolute_change,
                    min_relative_change=min_relative_change,
                    higher_is_better=direction,
                    equivalence_margin=margin,
                    value_range=descriptor.value_range,
                )
                regression = RegressionFlag(
                    label=verdict.label,
                    delta=verdict.delta,
                    relative_delta=verdict.relative_delta,
                    significant=verdict.significant,
                    exceeds_threshold=verdict.exceeds_threshold,
                    hedges_g=verdict.hedges_g,
                    p=verdict.p_value,
                    equivalence_p=verdict.equivalence_p,
                    equivalence_margin=verdict.equivalence_margin,
                    equivalence_untested_reason=verdict.equivalence_untested_reason,
                    not_separated_reason=verdict.not_separated_reason,
                    n_pairs=verdict.n_pairs,
                    crosses_epoch=boundary,
                    crosses_cassette_mode=crosses_cassette,
                    # Constant across the series — a property of the measure, not of this
                    # pair — and carried per flag for the reason `test` and the thresholds
                    # beside it are: a verdict read on its own must declare its own posture.
                    attribution_withheld=attribution_withheld,
                    test=flag_test,
                    min_absolute_change=min_absolute_change,
                    min_relative_change=min_relative_change,
                )

            if index == 0:
                baseline_value = value
            delta_from_baseline = value - baseline_value if (value is not None and baseline_value is not None) else None

            points.append(
                SeriesPoint(
                    run_id=run_id,
                    created_at=run.created_at,
                    model=rows_by_run[run_id][0].model,
                    value=value,
                    sem=sem,
                    n=n_observations,
                    n_cases=len(per_case),
                    epoch=epoch_ordinal,
                    epoch_boundary=boundary,
                    is_baseline=index == 0,
                    delta_from_baseline=delta_from_baseline,
                    regression=regression,
                    # Only the cost series has a composition; asking a latency or quality
                    # point what roles its dollars covered would answer a question it is
                    # not an answer to.
                    cost_compositions=(
                        pooled_cost_compositions(rows_by_run[run_id]) if metric == METRIC_COST_USD else []
                    ),
                    composite_basis=(
                        pooled_composite_basis(rows_by_run[run_id])
                        if metric == METRIC_COMPOSITE and value is not None
                        else None
                    ),
                    # On every measure, not only cost: a replayed run's LATENCY and QUALITY
                    # are as substituted as its dollars, and this is the series where a
                    # capture point and a replay point sit under one contestant heading.
                    cassette_mode=run.cassette_mode,
                    # Read from the map built over the placed rows rather than from the run
                    # again, so the point's mark and the result's corpus-level set are one
                    # predicate and cannot disagree about which runs were short.
                    completeness_disclosure=degraded_by_run.get(run_id),
                    served_models=pooled_served_models(rows_by_run[run_id]),
                )
            )
            prev_epoch_key = epoch_key
            prev_per_case = per_case

        _label_epochs_by_case_set(points, runs_in_series)
        first = rows[0].result
        # Unpacked from the group key rather than re-derived off a row, on
        # `_frontier_point`'s reasoning: the key is what placed these rows together, so
        # reading the identity back out of it is structural rather than asserted.
        variant_key, identity_version = contestant
        series.append(
            MeasureSeries(
                subject_id=resolved,
                subject_label=subject_labels.get(resolved, ""),
                variant_key=variant_key,
                variant_identity_version=identity_version,
                identity_version_disclosure=_identity_version_disclosure(identity_version),
                model=first.model,
                points=points,
                served_models=pooled_served_models([row.result for row in rows]),
            )
        )

    # Version in the key for the reason `frontier`'s point sort carries it: the partition
    # makes two series that share a subject, a model and a key reachable.
    series.sort(
        key=lambda measure_series: (
            measure_series.subject_id,
            measure_series.model,
            measure_series.variant_key,
            measure_series.variant_identity_version,
        )
    )

    identity_versions = _identity_version_span(considered)

    return HistoryResult(
        metric=metric,
        measure=descriptor,
        formula=_effective_formula(metric, WEIGHTING_EQUAL_PER_SCENARIO),
        weighting=WEIGHTING_EQUAL_PER_SCENARIO,
        higher_is_better=descriptor.higher_is_better,
        min_absolute_change=min_absolute_change,
        min_relative_change=min_relative_change,
        equivalence_margin=margin,
        series=series,
        n_results=n_considered,
        n_filtered_out=n_filtered_out,
        exclusions=exclusions,
        completeness_disclosures=degraded_by_run,
        n_degraded_observations=n_degraded_observations,
        identity_version_span=identity_versions,
        identity_span_disclosure=_identity_span_disclosure(identity_versions),
        attribution_disclosure=_attribution_disclosure(descriptor),
        contended_latency_disclosure=contended_latency_sentence(
            sum(1 for result in considered if result.id in contended), len(considered)
        ),
    )


__all__ = [
    "compute_history",
    "HISTORY_METRICS",
    "HistoryError",
    "HistoryResult",
    "MeasureSeries",
    "RegressionFlag",
    "SeriesPoint",
]

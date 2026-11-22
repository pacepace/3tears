"""Per-cell reads: which results a cell holds, which of them count, and every judged and measured value in it.

The populations the bars, comparisons and decision surface share — results the harness did not fault
(:func:`non_faulted`), candidates that failed, results that took no turn — and the per-cell summaries
:func:`judged_measures`, :func:`cell_measures` and :func:`cell_strata` built over them.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence

from threetears.evals.analysis.agreement import (
    JudgeKey,
    claim_judge_keys,
    judge_key,
    tier_for_judges,
)
from threetears.evals.kernel.evidence_tiers import JudgeEvidenceTier
from threetears.evals.analysis.arms import surface_order
from threetears.evals.analysis.cells import Cell
from threetears.evals.analysis.permutation import sequential_permutation_p, stratified_hypergeometric_totals
from threetears.evals.analysis.reporting import (
    METRIC_OUTCOME,
    METRIC_SCORE,
    METRIC_TRANSCRIPT,
    ScoreRecord,
)
from threetears.evals.analysis.stats import (
    PPI_MIN_LABELLED_RESULTS,
    clustered_standard_error,
    prediction_powered_mean,
    small_sample_case_means,
)
from threetears.evals.kernel.analysis_measures import MeasureCollection
from threetears.evals.kernel.declaration import (
    BarName,
    AspectDesign,
)
from threetears.evals.kernel.host.measures import MeasureRegistry
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.claims.readings import (
    CLAIM_LOST_OUTCOME,
    CLAIM_PENDING_OUTCOME,
    CLAIM_SILENT_OUTCOME,
    CLAIM_UNREAD_OUTCOME,
    claim_instruments,
    claim_reading_ranks,
    describe_judged_reading,
)
from threetears.evals.schema.models import SCALES
from threetears.evals.schema.reading_states import StateOf
from threetears.evals.kernel.result_reading import ResultView
from threetears.evals.kernel.result_condition import (
    DIFFERENTIAL_EXCLUSION_SHARE,
    JUDGE_CANNOT_TELL_OUTCOME,
    ExclusionCause,
    ResultOutcome,
    classify_result,
    delivered_a_turn,
    harness_faulted,
    judge_truncated_dims,
    trial_exclusion,
)
from threetears.evals.kernel.surface import (
    CellFacts,
    JudgedReading,
    PredictionPoweredReading,
    StratumFacts,
)
from threetears.evals.analysis.bundle.schema import (
    ArmExclusions,
    DifferentialExclusion,
    JudgedArm,
    JudgedMeasure,
)
from threetears.evals.analysis.bundle.measures import measure_collection


#: A cell's two coordinates, ``(variant_key, apparatus_class_id)`` — how the per-arm surfaces key it.
CellKey = tuple[str, str]


def cell_collections(cell: CellFacts) -> Iterator[MeasureCollection]:
    """A cell's measure collection, then each of its strata's.

    Walked wherever a reader needs every measure NAME a cell holds: a stratum is a smaller set of results,
    so it can carry a name the pooled cell does not — two carriers sharing a leaf name are refused pooling
    over the cell and may each be alone in a stratum.
    """
    yield cell.measures
    for stratum in cell.strata:
        yield stratum.measures


def group_results_by_cell(cells: list[Cell], results: list[ResultView]) -> dict[CellKey, list[ResultView]]:
    """Each cell's results, keyed by its two coordinates — the arms every per-arm surface here reads.

    Read off the cell algebra rather than re-grouped, so a per-arm number can only ever describe a
    set of observations the bundle already calls one cell.

    :param cells: The pooled cells.
    :param results: Every resolved member result.
    :return: ``{(variant_key, apparatus_class_id): [result, ...]}`` in coordinate order, each list in
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


def member_run_ids(members: list[ResultView]) -> list[str]:
    """The runs a cell's results came from, sorted — what a per-arm surface names as citable.

    Read off the cell's own results rather than joined back through a run's configuration,
    because the cell IS its observations: whichever levers a host records, and whichever of them
    the variant index describes, the runs that produced a cell's observations are the runs it
    pooled. Every per-arm surface takes its ``run_ids`` from here, so a judged score and a bar
    verdict on one cell cannot name different runs.

    :param members: One cell's results, from :func:`group_results_by_cell`.
    :return: The distinct ``eval_run_id`` values, sorted.
    """
    return sorted({result.eval_run_id for result in members})


def judged_values(records: list[ScoreRecord]) -> Iterator[tuple[str, ScoreRecord]]:
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


#: The outcomes of a judged row with no value that an arm counts: the judge could not tell, the harness faulted the
#: result, and, on a claim reading, its criterion lost the cell, its tally waits on its stored grade, the candidate said
#: nothing in it, or, on a reading the host derives, its claim reader could not read the cell.
_NULL_JUDGED_OUTCOMES: frozenset[str] = frozenset(
    {
        JUDGE_CANNOT_TELL_OUTCOME,
        ResultOutcome.INFRA_EXCLUDE.value,
        CLAIM_LOST_OUTCOME,
        CLAIM_PENDING_OUTCOME,
        CLAIM_SILENT_OUTCOME,
        CLAIM_UNREAD_OUTCOME,
    }
)

#: The outcomes of a judged row that leave it out of a mean.
_UNCOUNTED_JUDGED_OUTCOMES: frozenset[str] = _NULL_JUDGED_OUTCOMES


def judged_rows_with_nulls(records: list[ScoreRecord]) -> Iterator[tuple[str, ScoreRecord]]:
    """:func:`judged_values`, plus the null rows the judge answered it could not tell on or the harness faulted, and
    a claim reading's rows its criterion lost or has not yet tallied.

    What a judged measure COUNTS rather than means: those rows carry no value, so they never reach a
    mean, and are yielded here only so the arm can say how many left it and why — a faulted
    observation's score row is null since the projection reads ``counted_rubric_scores``.
    """
    yield from judged_values(records)
    for record in records:
        if record.value is None and record.outcome in _NULL_JUDGED_OUTCOMES:
            if record.metric == METRIC_SCORE and record.rubric_dim:
                yield record.rubric_dim, record
            elif record.metric in (METRIC_TRANSCRIPT, METRIC_OUTCOME):
                yield record.metric, record


def judged_measures(
    records: list[ScoreRecord],
    results_by_cell: dict[CellKey, list[ResultView]],
    design: AspectDesign | None,
    *,
    tiers: list[JudgeEvidenceTier],
    person_scores: Mapping[tuple[str, str], Sequence[int]] | None = None,
    claim_units: Mapping[str, str] | None = None,
    registry: MeasureRegistry | None = None,
) -> list[JudgedMeasure]:
    """Summarise every judged dimension, and every ranked claim reading, per cell, from the score projection assembly
    already holds.

    The projection is the source because it is where the dimension's NAME survives: the measure
    walk meets a judged score as a bare ``score`` leaf on a ``RubricScore`` and cannot tell one
    dimension from another. A score on an observation the harness faulted is counted beside the mean
    and left out of it, because a judge reading a broken transcript is not measuring the candidate.

    **A claim reading** (a row carrying an ``instrument``) is described as its criterion's reading
    (:func:`~threetears.evals.kernel.claims.readings.describe_judged_reading`), a rate or a share and never a level on
    a rubric scale. Only the three that rank are summarised here; the diagnostics stay in the projection. Each arm
    counts the cells its criterion lost, the cells still pending and the cells its candidate said nothing in apart from
    its mean, a pending or silent cell as no loss, and the cells whose web check skipped a claim past its budget. The
    reading is withheld (``trials_lost``) where the arms' shares of cells left unread (lost, silent, pending, or unread
    by the host's reader) differ beyond chance, by one test across every arm (:func:`_differential_claim_loss`).

    :param records: The assembly's score projection rows.
    :param results_by_cell: Each cell's results, from :func:`group_results_by_cell`.
    :param design: The campaign's declaration, for the bar naming a dimension.
    :param tiers: The judges' evidence tiers; each arm carries the weakest among the judges that served its
        counted scores (:func:`~threetears.evals.analysis.agreement.tier_for_judges`).
    :param person_scores: People's scores per ``(result_id, dimension)``
        (:func:`~threetears.evals.analysis.agreement.person_scores_by_result`), from which each arm with a
        rated counted observation carries a prediction-powered estimate (:func:`_prediction_powered`); None
        reads as nobody rated anything.
    :param claim_units: What a page calls one unit of each claim criterion, from the member runs' templates
        (``EvalTemplate.claim_unit_names``), for a claim reading's reader name; ``unit`` where none is named.
    :param registry: The host's measure registry, which describes the readings it derives from a claim criterion
        (``MetricDescriptor.derived_from``); a derived reading ranks, and is withheld, as the criterion's own do.
    :return: One entry per dimension and ranked claim reading scored anywhere, sorted by name.
    """
    result_by_id = {result.id: result for members in results_by_cell.values() for result in members}
    cell_of_result = {result.id: key for key, members in results_by_cell.items() for result in members}
    boundary = boundary_dimensions(result_by_id.values())
    instruments = claim_instruments((record.rubric_dim, record.instrument) for record in records)
    scored: dict[str, dict[CellKey, list[ScoreRecord]]] = {}
    for dimension, record in judged_rows_with_nulls(records):
        key = cell_of_result.get(record.result_id)
        if key is not None:
            scored.setdefault(dimension, {}).setdefault(key, []).append(record)
    bars = {bar.measure_id: bar.threshold for bar in (design.bars if design else [])}
    measures = []
    for dimension in sorted(scored):
        criterion = instruments.get(dimension)
        if criterion is not None and not claim_reading_ranks(dimension, criterion=criterion, measures=registry):
            continue
        scales = {
            row.rubric_scale for rows in scored[dimension].values() for row in rows if row.rubric_scale is not None
        }
        if len(scales) > 1:
            raise ValueError(f"rubric dimension {dimension!r} was judged on more than one scale: {sorted(scales)}")
        # The dual-score axes carry no `rubric_scale` (they are not template dims) and are 1-5 by construction; every
        # template dim's row carries the scale it was answered on, and one carrying none is refused, never read as 1-5.
        # A claim reading carries none either, and is described as its criterion's reading.
        recorded = {dimension: scale for scale in scales}
        descriptor = describe_judged_reading(
            dimension, scales=recorded, instruments=instruments, units=claim_units, measures=registry
        )
        arms = []
        for key in sorted(scored[dimension]):
            rows = scored[dimension][key]
            cannot_tell = [row for row in rows if row.outcome == JUDGE_CANNOT_TELL_OUTCOME]
            lost = [row for row in rows if row.outcome == CLAIM_LOST_OUTCOME]
            pending = [row for row in rows if row.outcome == CLAIM_PENDING_OUTCOME]
            unread = [row for row in rows if row.outcome == CLAIM_UNREAD_OUTCOME]
            silent = [row for row in rows if row.outcome == CLAIM_SILENT_OUTCOME]
            counted = [row for row in rows if row.outcome not in _UNCOUNTED_JUDGED_OUTCOMES]
            valued = [row for row in counted if row.value is not None]
            values = [float(row.value) for row in valued if row.value is not None]
            value_cases = [row.test_case_id for row in valued]
            # The whole judge behind each counted score — dimension, scale, served model and config — so an arm
            # can only carry a tier measured for the very judges that scored it.
            served = [
                judge for row in valued for judge in _judges_of(result_by_id[row.result_id], dimension, criterion)
            ]
            arms.append(
                JudgedArm(
                    variant_key=key[0],
                    apparatus_class_id=key[1],
                    run_ids=member_run_ids(results_by_cell[key]),
                    n=len(values),
                    n_independent=len(set(value_cases)),
                    n_infra_excluded=(
                        len(rows)
                        - len(counted)
                        - len(cannot_tell)
                        - len(lost)
                        - len(pending)
                        - len(unread)
                        - len(silent)
                    ),
                    n_cannot_tell=len(cannot_tell),
                    n_claim_lost=len(lost),
                    n_claim_pending=len(pending),
                    n_claim_unread=len(unread),
                    n_claim_silent=len(silent),
                    n_web_skipped=(0 if criterion is None else _web_skipped_cells(results_by_cell[key], criterion)),
                    n_web_lost=0 if criterion is None else _web_lost_cells(results_by_cell[key], criterion),
                    mean=sum(values) / len(values) if values else None,
                    sem=clustered_standard_error(values, value_cases) if values else None,
                    case_means=small_sample_case_means(values, value_cases),
                    evidence_tier=tier_for_judges(tiers, served),
                    prediction_powered=_prediction_powered(
                        dimension,
                        valued,
                        result_by_id,
                        person_scores or {},
                        value_range=descriptor.value_range,
                    ),
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
                instrument=criterion,
                reader_name=descriptor.reader_name if criterion is not None else None,
                withheld=(
                    StateOf(state="withheld", reason="trials_lost")
                    if criterion is not None and _differential_claim_loss(scored[dimension])
                    else None
                ),
            )
        )
    return measures


def _judges_of(result: ResultView, dimension: str, criterion: str | None) -> list[JudgeKey]:
    """The judges behind one counted value: a claim reading's every deciding judge key, a dimension's one score's."""
    if criterion is not None:
        return claim_judge_keys(result, dimension, criterion=criterion)
    judge = judge_key(result, dimension)
    return [] if judge is None else [judge]


def _web_skipped_cells(members: Sequence[ResultView], criterion: str) -> int:
    """How many of a cell's results hold a claim the criterion's web check skipped past its budget.

    A web step that failed is a cause apart (:func:`_web_lost_cells`, its own caveat).
    """
    return sum(
        1
        for result in members
        if (tally := result.claim_tallies.get(criterion)) is not None
        and any(entry.web_skipped for entry in tally.by_judge)
    )


#: The level the test of claim losses across arms rejects at: a reading whose losses it calls uneven is withheld.
_CLAIM_LOSS_LEVEL = 0.05

#: The outcomes of a claim reading's row that leave a cell the reading could have been read on with no reading: its
#: criterion lost it, its candidate said nothing in it, the host's reader could not read it, or its tally still waits
#: on its stored grade. Each is missing data that may track arm (an arm going quiet on its hard cells lowers its false
#: claims by selection), so the test of uneven loss counts them alike. Only a lost cell counts as a loss; a pending one
#: normally settles on its own, so a reading withheld for an uneven backlog is read again once it settles.
_UNREAD_CLAIM_OUTCOMES: frozenset[str] = frozenset(
    {CLAIM_LOST_OUTCOME, CLAIM_SILENT_OUTCOME, CLAIM_UNREAD_OUTCOME, CLAIM_PENDING_OUTCOME}
)


def _web_lost_cells(members: Sequence[ResultView], criterion: str) -> int:
    """How many of a cell's results lost the criterion's web step to a failure (the port, a malformed reply, a verdict
    with no quote): a cause apart from the budget, whose undecided claims read unverifiable and never false."""
    return sum(
        1
        for result in members
        if (tally := result.claim_tallies.get(criterion)) is not None and tally.web_loss is not None
    )


def _differential_claim_loss(by_cell: Mapping[CellKey, list[ScoreRecord]]) -> bool:
    """Whether a claim reading's arms left its cells unread at shares that differ beyond chance, by one test across
    every arm.

    Each cell the reading could have been read on (read, or left unread: lost, silent, pending, or unread by the host's
    reader for a derived reading; a harness-faulted cell is in neither) is one trial of a case: unread or kept. A silent
    or pending cell is no loss and is in no mean, but its absence may track arm as a loss's may, so an arm silent, or
    waiting on its grades, on a sharply higher share of cells has the reading withheld as one that lost them would; a
    pending backlog normally settles on its own, and the reading is read again once it has. The test is the
    Cochran-Mantel-Haenszel test of general association between arm and loss, stratified by case
    (:func:`_claim_loss_homogeneity_p`): one test whatever the number of arms, so a reading is withheld for chance loss
    at most as often as its level allows at two arms or at four, and a case one arm ran more of moves nothing on its
    own. Its p is the statistic's conditional permutation p at every size, never the chi-square law's: the permutation
    is valid and conservative however many cells were lost, and one law at every count means no reading's p moves as a
    campaign's losses cross a threshold. Its 5% is held by that construction: the permutation's size is its level, and
    reading its draws sequentially adds at most 0.0003. The ``claim-loss.*`` claims of ``tests/stats_certificate.py``
    certify the implementation is not anti-conservative, each null row's false-withhold rate not above 5% by more than
    one point on the 95% upper bound of 4,000 draws: over twelve cases at two, three and four arms, equal losses of 10%
    and of 3%, at one repeat a case, and a loss a few cases carry; an equal 20% over forty cases of four repeats at two
    arms (about 64 lost cells); and a loss two fifths of sixty cases carry at one repeat a case, at three and four arms
    (about 49 and 65 lost cells, in strata of three or four cells). A 0% against 30% loss is caught at least 75% of the
    time.

    :param by_cell: Each arm's rows of the reading, by cell.
    :return: Whether the test rejects at :data:`_CLAIM_LOSS_LEVEL`.
    """
    p = _claim_loss_homogeneity_p(by_cell)
    return p is not None and p < _CLAIM_LOSS_LEVEL


#: The draws a permutation p is read over, and the fixed seed they are drawn from, so one bundle reads one p.
_CLAIM_LOSS_DRAWS = 999
_CLAIM_LOSS_SEED = 1287


def _claim_loss_homogeneity_p(by_cell: Mapping[CellKey, list[ScoreRecord]]) -> float | None:
    """The generalized Cochran-Mantel-Haenszel p for arm against loss, stratified by case, on ``arms - 1`` degrees of
    freedom; None where fewer than two arms hold a cell in a case whose cells were neither all lost nor all kept (no
    stratum then tells the arms apart).

    Per case, a table of each arm's cells and losses; under no difference between arms, each case's losses fall among
    its cells as a draw without replacement (hypergeometric), and the deviations of every arm but one, summed over
    cases, form the statistic. Its p is the statistic's own conditional law, never the chi-square law's (with few losses
    the law misstates the chance: four losses in four cases, all on one arm, read p 0.046 by the law, 1/8 by the draw;
    with many it runs near its nominal level, which no bound on its false withholds can sit under): each case's losses
    drawn again among its cells up to :data:`_CLAIM_LOSS_DRAWS` times from :data:`_CLAIM_LOSS_SEED`, ``p = (1 + draws
    at least as extreme) / (1 + draws)``, which errs at most its level at any count. The draws are each arm's losses
    summed over cases, drawn from each case's margins as counts, never by shuffling cells, and read sequentially
    (:func:`~threetears.evals.analysis.permutation.sequential_permutation_p`): a read stops once its p can no longer
    fall under 5%, which changes no decision, or, at an early look, once so few draws are as extreme that it rejects,
    which adds at most 0.0003 to the false-withhold rate. At forty thousand cells over four arms, a tenth or a quarter
    of them lost, a reading takes a few hundredths of a second, and at most about a fifth where every draw is read.
    """
    tables: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for arm, rows in enumerate(by_cell.values()):
        for row in rows:
            if row.outcome in _UNREAD_CLAIM_OUTCOMES or (
                row.outcome not in _UNCOUNTED_JUDGED_OUTCOMES and row.value is not None
            ):
                tables[row.test_case_id].append((arm, row.outcome in _UNREAD_CLAIM_OUTCOMES))
    informative = [
        cells
        for _, cells in sorted(tables.items())
        if len(cells) >= 2 and 0 < sum(lost for _, lost in cells) < len(cells)
    ]
    arms = sorted({arm for cells in informative for arm, _ in cells})
    if len(arms) < 2:
        return None
    free = arms[:-1]
    covariance = [[0.0] * len(free) for _ in free]
    expected = [0.0] * len(free)
    for cells in informative:
        total, lost = len(cells), sum(held for _, held in cells)
        sizes = Counter(arm for arm, _ in cells)
        scale = lost * (total - lost) / (total * total * (total - 1))
        for i, a in enumerate(free):
            expected[i] += sizes[a] * lost / total
            for j, b in enumerate(free):
                covariance[i][j] += scale * ((total * sizes[a] if a == b else 0) - sizes[a] * sizes[b])
    inverse = [
        _solve(covariance, [1.0 if row == column else 0.0 for row in range(len(free))]) for column in range(len(free))
    ]
    if any(column is None for column in inverse):
        return None
    columns = [column for column in inverse if column is not None]

    def statistic(lost_by_arm: Counter[int]) -> float:
        deviation = [lost_by_arm[a] - expected[i] for i, a in enumerate(free)]
        return sum(deviation[i] * columns[j][i] * deviation[j] for i in range(len(free)) for j in range(len(free)))

    observed = statistic(Counter(arm for cells in informative for arm, held in cells if held))
    # Each case's cells by arm and its losses: its margins, all the conditional law reads. A draw takes each case's
    # losses among its cells without replacement, as counts by arm (``stratified_hypergeometric_totals``).
    strata = [
        (tuple(Counter(arm for arm, _ in cells)[each] for each in arms), sum(held for _, held in cells))
        for cells in informative
    ]
    totals = stratified_hypergeometric_totals(strata, random.Random(_CLAIM_LOSS_SEED))
    read: dict[tuple[int, ...], bool] = {}

    def at_least_as_extreme() -> bool:
        drawn = next(totals)
        extreme = read.get(drawn)
        if extreme is None:
            extreme = read[drawn] = statistic(Counter(dict(zip(arms, drawn, strict=True)))) >= observed - 1e-9
        return extreme

    return sequential_permutation_p(at_least_as_extreme, draws=_CLAIM_LOSS_DRAWS, level=_CLAIM_LOSS_LEVEL)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """``matrix⁻¹ · vector`` by Gaussian elimination with partial pivoting; None for a singular matrix."""
    size = len(vector)
    rows = [[*row, value] for row, value in zip(matrix, vector, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < 1e-12:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for row in range(size):
            if row != column:
                factor = rows[row][column] / rows[column][column]
                rows[row] = [held - factor * lead for held, lead in zip(rows[row], rows[column], strict=True)]
    return [rows[row][size] / rows[row][row] for row in range(size)]


def _prediction_powered(
    dimension: str,
    valued: list[ScoreRecord],
    result_by_id: Mapping[str, ResultView],
    person_scores: Mapping[tuple[str, str], Sequence[int]],
    *,
    value_range: tuple[float, float] | None,
) -> PredictionPoweredReading | None:
    """One arm's judged mean re-estimated with people's ratings of its counted observations (#598).

    Each counted observation a person rated is labelled with the people's mean score, COUNTED as the judge's score
    is (:func:`~threetears.evals.kernel.result_condition.counted_score`): on a candidate failure the judge's score
    counts the scale floor, so the person's does too, and the rectifier measures the judge where it was read, not a
    disagreement about a turn the measure already floors.

    :param dimension: The judged dimension.
    :param valued: The arm's counted, valued score rows on it.
    :param result_by_id: Every result read, by id.
    :param person_scores: People's scores per ``(result_id, dimension)``.
    :param value_range: The dimension's scale, which the interval is clipped to.
    :return: The estimate, its ``mean`` None with the reason below
        :data:`~threetears.evals.analysis.stats.PPI_MIN_LABELLED_RESULTS` labelled observations; or None when no
        counted observation here was rated.
    """
    human: list[float | None] = []
    for row in valued:
        scores = person_scores.get((row.result_id, dimension))
        if not scores:
            human.append(None)
            continue
        result = result_by_id[row.result_id]
        judged = result.judge_score(dimension)
        if classify_result(result) is ResultOutcome.CANDIDATE_FAIL and judged is not None:
            human.append(float(SCALES[judged.scale].scores[0]))
        else:
            human.append(sum(scores) / len(scores))
    labelled = [row.test_case_id for row, label in zip(valued, human) if label is not None]
    if not labelled:
        return None
    n_labelled, n_labelled_cases = len(labelled), len(set(labelled))
    if n_labelled < PPI_MIN_LABELLED_RESULTS:
        return PredictionPoweredReading(
            n_labelled=n_labelled,
            n_labelled_cases=n_labelled_cases,
            min_labelled=PPI_MIN_LABELLED_RESULTS,
            unavailable_reason=(
                f"not available: {n_labelled} of the counted scores here were rated by a person, under the "
                f"{PPI_MIN_LABELLED_RESULTS} a prediction-powered estimate is stated from."
            ),
        )
    estimate = prediction_powered_mean(
        [float(row.value) for row in valued if row.value is not None],
        [row.test_case_id for row in valued],
        human,
        value_range=value_range,
    )
    assert estimate is not None  # some observation is labelled
    low, high = estimate.interval if estimate.interval is not None else (None, None)
    return PredictionPoweredReading(
        n_labelled=n_labelled,
        n_labelled_cases=n_labelled_cases,
        min_labelled=PPI_MIN_LABELLED_RESULTS,
        mean=estimate.mean,
        rectifier=estimate.rectifier,
        sem=estimate.sem,
        ci_low=low,
        ci_high=high,
    )


def boundary_dimensions(results: Iterable[ResultView]) -> set[str]:
    """The judged dimensions any of ``results`` was scored on as a boundary dimension — the judged guardrails.

    Any one boundary score makes the dimension a guardrail: a dimension stamped both ways was declared a
    boundary somewhere, and reading it as a guardrail errs toward holding it rather than trading it.
    """
    return {score.dim for result in results for score in result.rubric_scores if score.axis == "boundary"}


def non_faulted(members: list[ResultView]) -> list[ResultView]:
    """A cell's results the harness did not fault — the one population every bar is adjudicated over.

    See :class:`BarVerdict` for why. Held as one function so the three kinds of bar name cannot
    each come to their own answer about which observations count.
    """
    return [result for result in members if not harness_faulted(result)]


def _candidate_failed(members: list[ResultView]) -> int:
    """How many of a cell's results the candidate failed, for any cause — each counted against the arm.

    By :func:`~threetears.evals.kernel.result_condition.classify_result`, the rule every rate and bar
    counts a failure by.
    """
    return sum(1 for result in members if classify_result(result) is ResultOutcome.CANDIDATE_FAIL)


def took_no_turn(members: list[ResultView]) -> bool:
    """Whether a cell's results include a failure and no turn — the reading :attr:`CellFacts.all_failed` makes.

    Read off the results, for the lenses that hold them rather than the cell's facts, by the same predicate
    (:func:`~threetears.evals.kernel.result_condition.delivered_a_turn`), so the two cannot disagree.
    """
    return _no_turn(members) > 0 and not any(delivered_a_turn(result) for result in members)


def _no_turn(members: list[ResultView]) -> int:
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


def cell_arm_exclusions(results_by_cell: Mapping[CellKey, list[ResultView]]) -> list[ArmExclusions]:
    """Each arm's excluded trials by cause, and the caps the truncated ones' judge calls were sent.

    One entry per cell, in coordinate order, over every result the cell holds, each classified once by
    :func:`~threetears.evals.kernel.result_condition.trial_exclusion`, the rule pass^k and the composite exclude by,
    so a count here is the count those measures lost. A ``truncated`` trial's caps are read off its cut-off entries
    (:func:`~threetears.evals.kernel.result_condition.judge_truncated_dims`), as each call's completion reported what
    it was sent; one that reported nothing makes the arm's cap unreported rather than guessed.

    :param results_by_cell: Each cell's results (:func:`group_results_by_cell`).
    :return: The arms' exclusions.
    """
    arms: list[ArmExclusions] = []
    for (variant_key, apparatus_class_id), members in results_by_cell.items():
        by_cause: Counter[ExclusionCause] = Counter()
        caps: set[int] = set()
        unreported = False
        for result in members:
            cause = trial_exclusion(result)
            if cause is None:
                continue
            by_cause[cause] += 1
            if cause == "truncated":
                for dim in judge_truncated_dims(result):
                    sent = result.judge_cut_short[dim].sent_settings
                    if sent is None:
                        unreported = True
                    else:
                        caps.add(sent.max_tokens)
        arms.append(
            ArmExclusions(
                variant_key=variant_key,
                apparatus_class_id=apparatus_class_id,
                n_trials=len(members),
                n_usable=len(members) - sum(by_cause.values()),
                by_cause=dict(sorted(by_cause.items())),
                truncated_sent_caps=sorted(caps),
                truncated_cap_unreported=unreported,
            )
        )
    return arms


def differential_exclusions(
    arms: Sequence[ArmExclusions], *, rigs: Mapping[str, str] | None = None
) -> list[DifferentialExclusion]:
    """Every (arm, cause) where the cause took more than the threshold share of the arm's trials, beside the other arms.

    The threshold is :data:`~threetears.evals.kernel.result_condition.DIFFERENTIAL_EXCLUSION_SHARE`; the other arms
    named are those under the same rig, the only arms a comparison reads the arm against.

    :param arms: The arms' exclusions (:func:`cell_arm_exclusions`).
    :param rigs: Apparatus class id → the class of its rig alone, for each class split on the answering model
        (:func:`~threetears.evals.analysis.cells.rig_of`); a class absent here is its own rig.
    :return: The flags, in the arms' order and then by cause; empty when no cause took more than the threshold anywhere.
    """
    flags: list[DifferentialExclusion] = []
    rig = rigs or {}

    def beside(arm: ArmExclusions) -> list[ArmExclusions]:
        """The other arms' cells under ``arm``'s rig, an arm holding several there (split on the model that answered
        it) left out: its share is no one number."""
        others = [
            other
            for other in arms
            if rig.get(other.apparatus_class_id, other.apparatus_class_id)
            == rig.get(arm.apparatus_class_id, arm.apparatus_class_id)
            and other.variant_key != arm.variant_key
        ]
        held = Counter(other.variant_key for other in others)
        return [other for other in others if held[other.variant_key] == 1]

    for arm in arms:
        for cause, n_excluded in sorted(arm.by_cause.items()):
            share = n_excluded / arm.n_trials
            if share <= DIFFERENTIAL_EXCLUSION_SHARE:
                continue
            flags.append(
                DifferentialExclusion(
                    variant_key=arm.variant_key,
                    apparatus_class_id=arm.apparatus_class_id,
                    cause=cause,
                    n_excluded=n_excluded,
                    n_trials=arm.n_trials,
                    share=share,
                    threshold=DIFFERENTIAL_EXCLUSION_SHARE,
                    other_arms={
                        other.variant_key: (other.by_cause.get(cause, 0) / other.n_trials if other.n_trials else 0.0)
                        for other in beside(arm)
                    },
                )
            )
    return flags


def cannot_tell_on(bar: BarName, counted: list[ResultView], judged_rows: dict[str, list[ScoreRecord]]) -> int:
    """How many of a cell's counted results the judge could not score on a judged bar's dimension."""
    if bar.kind != "judged":
        return 0
    return sum(
        1
        for result in counted
        for dimension, record in judged_rows_with_nulls(judged_rows.get(result.id, []))
        if dimension == bar.name and record.outcome == JUDGE_CANNOT_TELL_OUTCOME
    )


def _judged_by_cell(measures_judged: list[JudgedMeasure]) -> dict[CellKey, list[JudgedReading]]:
    """The judged measures transposed — each cell's readings, one per dimension it was scored on.

    :param measures_judged: Judged measures, from :func:`judged_measures`, sorted by name.
    :return: Each cell's readings, sorted by dimension, keyed by the cell's coordinates.
    """
    judged_by_cell: dict[CellKey, list[JudgedReading]] = {}
    # Judged measures are sorted by name, so each cell's readings arrive sorted by dimension.
    for measure in measures_judged:
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
                    prediction_powered=arm.prediction_powered,
                )
            )
    return judged_by_cell


#: How a stratum is keyed while the strata are built: a declared stratum, or None for the cases that declare none.
_StratumKey = str | None


def cell_strata(
    results_by_cell: dict[CellKey, list[ResultView]],
    stratum_of_case: dict[str, str | None],
    records: list[ScoreRecord],
    design: AspectDesign | None,
    *,
    tiers: list[JudgeEvidenceTier],
    profile: HostProfile,
    person_scores: Mapping[tuple[str, str], Sequence[int]] | None = None,
) -> dict[CellKey, list[StratumFacts]]:
    """Each cell read again per stratum of its cases — the same walk and transposition, over each stratum's results.

    A cell is broken down only when some case of it declares a stratum; one whose cases declare none has no
    entry here, and reads exactly as an unstratified cell does. Within a broken-down cell every result lands
    in one stratum — its case's, or the undeclared one — so the strata partition the cell.

    **The judged readings are scored over each stratum's own slice** by :func:`judged_measures`, with the
    campaign's evidence tiers: a judge's reliability is measured over the whole campaign, not one kind of
    case of it.

    :param results_by_cell: Each cell's results, from :func:`group_results_by_cell`.
    :param stratum_of_case: The stratum each case declares, by case id; a case that did not resolve is absent
        and reads as declaring none.
    :param records: The score projection, for judged dimensions.
    :param design: The campaign's declaration, for the bar a judged dimension carries.
    :param tiers: The judges' evidence tiers.
    :param profile: The host whose vocabulary this reads.
    :param person_scores: People's scores per ``(result_id, dimension)``, for each stratum's prediction-powered
        estimates; see :func:`judged_measures`.
    :return: Each broken-down cell's strata — named strata in name order, then the undeclared one — keyed by the
        cell's coordinates.
    """
    by_stratum: dict[_StratumKey, dict[CellKey, list[ResultView]]] = {}
    for key, members in results_by_cell.items():
        if not any(stratum_of_case.get(result.test_case_id) is not None for result in members):
            continue
        for result in members:
            by_stratum.setdefault(stratum_of_case.get(result.test_case_id), {}).setdefault(key, []).append(result)
    strata: dict[CellKey, list[StratumFacts]] = {}
    for stratum in sorted(by_stratum, key=lambda name: (name is None, name or "")):
        slice_by_cell = by_stratum[stratum]
        result_ids = {result.id for members in slice_by_cell.values() for result in members}
        judged = _judged_by_cell(
            judged_measures(
                [record for record in records if record.result_id in result_ids],
                slice_by_cell,
                design,
                tiers=tiers,
                person_scores=person_scores,
                registry=profile.measures,
            )
        )
        for key, members in slice_by_cell.items():
            strata.setdefault(key, []).append(
                StratumFacts(
                    stratum=stratum,
                    n_observations=len(members),
                    n_cases=len({result.test_case_id for result in members}),
                    n_infra_excluded=len(members) - len(non_faulted(members)),
                    n_candidate_failed=_candidate_failed(members),
                    n_no_turn=_no_turn(members),
                    measures=measure_collection(members, profile=profile, undeclared="scored"),
                    judged=judged.get(key, []),
                )
            )
    return strata


def cell_measures(
    cells: list[Cell],
    results_by_cell: dict[CellKey, list[ResultView]],
    measures_judged: list[JudgedMeasure],
    *,
    strata: dict[CellKey, list[StratumFacts]],
    short_runs: dict[str, str],
    incomplete_runs: dict[str, str],
    profile: HostProfile,
    control: str | None,
    names: Mapping[str, str],
) -> list[CellFacts]:
    """Everything measured in each cell — the facts an analysis freezes as its decision surface.

    **One population, the bars' own**: each cell's results the harness did not fault
    (:func:`non_faulted`), so a measure's value here and a :class:`BarVerdict` on the same cell and
    measure are one mean over one set of observations, never two readings a reader has to reconcile.
    A cost or latency measure is read over the ``delivered`` part of it — the turns the candidate took — by
    the same walk for the cell and the bar alike, and the failures that took none are counted beside it
    (``n_no_turn``, of ``n_candidate_failed``).

    **The judged readings are the judged measures, transposed** — read off what
    :func:`judged_measures` produced rather than scored again, so a cell's reading of a dimension
    and that dimension's arm for the cell are the same numbers by construction, not by two
    population rules happening to agree.

    :param cells: The pooled cells, for their replication.
    :param results_by_cell: Each cell's results, from :func:`group_results_by_cell`.
    :param measures_judged: The bundle's judged measures, from :func:`judged_measures`.
    :param strata: Each broken-down cell's strata, from :func:`cell_strata`; a cell absent from it carries none.
    :param short_runs: The bundle's short-run sentences, by run id.
    :param incomplete_runs: The bundle's incomplete-run statuses, by run id.
    :param profile: The host whose vocabulary this reads.
    :param control: The declared control's variant key, or None.
    :param names: The bundle's arm names (:func:`~threetears.evals.analysis.arms.arm_names`), which order the arms.
    :return: One entry per cell, in the decision surface's row order
        (:func:`~threetears.evals.analysis.arms.surface_order`): the control's cells first, as the reference,
        then every other arm alphabetically by name, each arm's rigs by id. Not a ranking.
    """
    judged_by_cell = _judged_by_cell(measures_judged)
    facts = []
    for cell in sorted(cells, key=lambda c: (c.variant_key, c.apparatus_class_id)):
        key = (cell.variant_key, cell.apparatus_class_id)
        members = results_by_cell[key]
        counted = non_faulted(members)
        run_ids = member_run_ids(members)
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
                measures=measure_collection(members, profile=profile, undeclared="scored"),
                judged=judged_by_cell.get(key, []),
                short_runs={run_id: short_runs[run_id] for run_id in run_ids if run_id in short_runs},
                incomplete_runs={run_id: incomplete_runs[run_id] for run_id in run_ids if run_id in incomplete_runs},
                strata=strata.get(key, []),
            )
        )
    return surface_order(facts, control=control, names=names)


__all__ = [
    "CellKey",
    "cell_arm_exclusions",
    "boundary_dimensions",
    "cannot_tell_on",
    "cell_collections",
    "cell_measures",
    "cell_strata",
    "differential_exclusions",
    "group_results_by_cell",
    "judged_measures",
    "judged_rows_with_nulls",
    "judged_values",
    "member_run_ids",
    "non_faulted",
    "took_no_turn",
]

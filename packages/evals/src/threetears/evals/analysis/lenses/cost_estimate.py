"""Cost estimation: a proposed sweep's cost predicted from historical per-cell costs.

:func:`compute_estimate_cost` prices each proposed cell from the history of comparable cells, with a prediction
band (:func:`_cost_band`) for what the sweep will itself cost. :class:`PredictedValue` is the predicted figure the
pivot also carries for a cell not yet run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.schema.models import utc_now_iso
from threetears.evals.analysis.reporting import _subject_id_of, pooled_cost_compositions

if TYPE_CHECKING:
    from threetears.evals.schema.models import (
        EvalResult,
        EvalRun,
    )


#: The ``method_id`` of the cost estimator's predictions: a planned cell priced from the corpus's
#: own per-observation usage history (:func:`compute_estimate_cost`).
COST_PREDICTION_METHOD = "usage-history"


class PredictedValue(EvalBaseModel):
    """A modelled estimate, kept beside the observed value it predicts and never fused with it.

    The two are different claims: an observed value is what happened, a predicted one is what a
    model expected beforehand, and a decision made on the second believing it was the first is the
    failure the separation prevents. So a prediction never occupies an observed slot — it has its
    own field wherever it appears, and a surface shows the two side by side.

    Two writers: :func:`compute_estimate_cost` (``method_id`` :data:`COST_PREDICTION_METHOD`), one per
    planned cell, predicting that cell's total cost from the corpus's usage history; and
    :meth:`~threetears.evals.ops.LaunchEstimate.planned_costs`, one per priced arm of a launch, from the host's
    launch pricer — whose ``method_id`` is the pricer's own (``usage-history`` for the engine's
    :func:`~threetears.evals.ops.history_launch_pricer`) or ``launch-pricer`` for a pricer that names none.
    :func:`compute_pivot` sets either beside each observed cost its plan describes.

    ``value`` is the point estimate, ``interval_low`` / ``interval_high`` its uncertainty band
    (absent for a point-only method, or where the basis is too thin for one), ``method_id``
    identifies the estimator so two predictions from different methods are never silently compared,
    ``trust`` is the method's own support/confidence score (None for a method that states none),
    and ``computed_at`` is the freshness stamp — a stale prediction over moved data is worse than
    none.
    """

    value: float
    interval_low: float | None = None
    interval_high: float | None = None
    method_id: str
    trust: float | None = None
    computed_at: str


# The band beside a cost estimate is a PREDICTION interval for what the proposed sweep
# will itself cost — not a confidence interval on the historical mean. The distinction is
# the whole reason this surface exists: the caller asks "what will this sweep cost", and
# the old `1.96 * n_obs * SEM` answered "how precisely does the history locate its own
# mean", which is a strictly narrower question and one that keeps NARROWING as history
# accumulates. Answering the narrow question in the wide question's place can publish a
# +/-2.4% band off a two-observation basis, and a sweep can then land well outside that
# band — in either direction, since nothing in that arithmetic prefers over-prediction.
#
# The band carries both terms: the uncertainty in where the history's distribution sits, and
# the variation the sweep's own observations will show. It is read on the LOG scale
# (`stats.lognormal_sum_prediction_band`), because costs are positive and right-skewed: a few
# long conversations cost several times the median. The normal-theory band it replaced,
# `t * s * sqrt(n_obs + n_obs^2 / n)`, covered a lognormal sweep's total (log-SD 1.0, five
# past observations) 82% of the time against its stated 95%. That band is kept only for a
# history holding a cost of zero or less, which the log scale cannot read.

# Smallest historical basis that gets a published band at all. Two observations yield
# exactly ONE difference: whatever spread they show is a single accident with nothing to
# check it against, and at df=1 Student's t is the Cauchy limit, where the multiplier
# contributes as much of the width as the data does. Publishing a number there dresses an
# accident as a measurement in whichever direction the pair happened to fall — narrow when
# they land close, which is the dangerous direction because narrow reads as confidence.
# Below this the cell reports its point estimate and no band, and the surfaces say in
# words why; three observations is the smallest sample whose spread rests on more than one
# difference.
#: The fewest past observations a cost estimate publishes a band from; below it, the point estimate alone.
COST_ESTIMATE_MIN_BASIS = 3

#: What every band assumes about its observations, stated on each band (:attr:`CostEstimateCell.band_basis`).
_COST_BAND_INDEPENDENCE = (
    "Each past observation and each planned one is treated as an independent draw; repeats of one case are not, "
    "so where cases differ in cost the band is narrower than it should be."
)


def _cost_band(history: Sequence[float], n_observations: int) -> tuple[float, float, str]:
    """The ~95% prediction band for the total of a sweep of ``n_observations``, and what it assumes.

    Log-scale (:func:`~threetears.evals.analysis.stats.lognormal_sum_prediction_band`) wherever every
    past cost is positive. A history with a zero or negative cost cannot be read on the log scale, and
    takes the normal-theory band ``t(0.95, n-1) * s * sqrt(m + m^2/n)`` around ``m`` times its mean, floored
    at zero — which assumes symmetric costs and says so.

    Args:
        history: The past per-observation costs, at least :data:`COST_ESTIMATE_MIN_BASIS` of them.
        n_observations: How many observations the proposed sweep would run.

    Returns:
        ``(low, high, basis)``: the band in dollars and the sentence stating its method and assumptions.
    """
    from threetears.evals.analysis.stats import (
        INTERVAL_LEVEL,
        lognormal_sum_prediction_band,
        standard_error_of_mean,
        t_critical_two_sided,
    )

    n = len(history)
    log_band = lognormal_sum_prediction_band(history, n_observations)
    if log_band is not None:
        return (
            log_band[0],
            log_band[1],
            f"{INTERVAL_LEVEL:.0%} prediction band for the sweep's total from a lognormal fit to {n} past "
            f"observations, the fit's own uncertainty included. {_COST_BAND_INDEPENDENCE}",
        )
    sem = standard_error_of_mean(list(history)) or 0.0
    centre = n_observations * math.fsum(history) / n
    half = (
        t_critical_two_sided(INTERVAL_LEVEL, n - 1)
        * sem
        * math.sqrt(n)
        * math.sqrt(n_observations + n_observations**2 / n)
    )
    return (
        max(0.0, centre - half),
        centre + half,
        f"{INTERVAL_LEVEL:.0%} normal-theory prediction band for the sweep's total from {n} past observations — "
        "some cost nothing, which the log scale cannot read, so this band assumes symmetric costs and is narrow "
        f"on the high side when a few cost far more than the rest. {_COST_BAND_INDEPENDENCE}",
    )


class CostEstimateError(ValueError):
    """A cost estimate was requested for a proposal that cannot be costed.

    Raised rather than returning a zero or empty estimate: an empty model list or
    a non-positive grid size is a malformed request, and answering it with ``$0``
    would read as "this sweep is free" rather than "you asked for nothing".
    """


class PlannedCost(EvalBaseModel):
    """One planned model's predicted cost over its planned observations, as a cost pivot's plan reads it.

    The plan a launch priced by its host's pricer hands a cost pivot: the same three facts a
    :class:`CostEstimateCell` carries for one — the model, how many observations its prediction totals, and
    the prediction — so the pivot divides it by the observations and sets it beside the cost observed, never in
    place of it.

    **Which cells it describes is part of the plan.** A prediction is for one model running one template, so it
    sits only in a pivot cell whose observations are all that template's at that model; and once the plan
    launched, ``run_ids`` names the runs it made, so a cell that pools other runs too — the earlier history the
    prediction was derived from among them — says how many of its observations the plan did not make
    (:attr:`PivotCell.n_unplanned`).

    Attributes:
        model: The planned candidate model.
        template_id: The template the plan runs; ``None`` for an estimate pooled over templates, which sits beside
            the model's cells whatever their template.
        run_ids: The runs the plan launched; empty before it launched, when the observed cost beside it is
            history the plan did not make.
        n_observations: The observations the prediction totals (cases × repeats).
        predicted: The predicted total cost, or ``None`` for a model nothing predicted.
    """

    model: str
    template_id: str | None = None
    run_ids: list[str] = Field(default_factory=list)
    n_observations: int = Field(ge=1)
    predicted: PredictedValue | None = None


class CostEstimateCell(EvalBaseModel):
    """The predicted cost of running one proposed model, from its historical per-observation cost.

    One planned cell, and its prediction is a :class:`PredictedValue` (``method_id``
    :data:`COST_PREDICTION_METHOD`) rather than loose numbers, so the pivot can set it beside the
    cost the cell later observed without either one passing for the other.
    ``predicted.value`` is ``mean_cost_per_observation × n_observations``.
    ``predicted.interval_low`` / ``interval_high`` are the ~95% **prediction** band for what
    the proposed sweep will cost — not a confidence interval on the historical
    mean, which is a narrower claim than any caller of this surface is making — and
    ``band_basis`` says how it was drawn and what it assumes.

    The band is ``None`` when ``n_historical`` is below :data:`COST_ESTIMATE_MIN_BASIS`. At one observation the spread is
    *unknown*, not zero — the same rule the pivot applies to an n=1 cell. At two it
    is worse than unknown: two points give exactly one difference, so any band drawn
    from them is an accident dressed as a measurement, and it prints NARROW whenever
    the pair happens to land close, which is the direction that reads as confidence.
    A stated absence is the honest answer; the point estimate still stands.

    ``basis`` is ``no_history`` when no PRICED result in the corpus matches this model at
    the proposed cassette mode, and then every value is ``None``, ``predicted`` included: an estimate invented
    with no data is worse than an admitted gap. ``n_unpriced_historical`` counts the matching
    results left out because their spend went unpriced — the reason a cell can say
    ``no_history`` while the model has run.
    """

    model: str
    n_observations: int
    n_historical: int
    mean_cost_per_observation: float | None = None
    predicted: PredictedValue | None = None
    basis: Literal["historical", "no_history"]
    #: What the band assumes, in words: how ``predicted.interval_low`` / ``interval_high`` were drawn and what
    #: they leave out — among them that every observation is treated as independent. ``None`` with no band, or
    #: on an estimate stored before bands stated it: those were the normal-theory t band, which assumed
    #: symmetric costs and covered a skewed sweep's total well short of its 95%.
    band_basis: str | None = None
    #: Matching historical results whose spend could not be priced, and so are not in the basis:
    #: a mean drawn only from the priced ones prices a model that partly runs unpriced as if it
    #: never did. Counted rather than folded in as zeros, which would pull the estimate down.
    n_unpriced_historical: int = 0
    #: Which role sets the historical basis summed its dollars over. More than one entry
    #: means the mean was drawn across a change in what a cost figure includes — the same
    #: reason the basis is matched to the proposed cassette mode, surfaced rather than
    #: matched because the proposed run's own composition is not yet decided.
    basis_cost_compositions: list[list[str]] = []


class CostEstimate(EvalBaseModel):
    """A proposed run/campaign's predicted cost, per model and in total, banded where n allows.

    The estimate is deliberately an interval wherever the history can carry one:
    a single number would read as a quote when it is a projection from
    historical spread. The total bounds are the **sum** of the per-cell bounds — a
    conservative combination that assumes the cells' costs move together, so the
    band is a safe budgeting envelope rather than an optimistic one.

    ``total_interval_low`` / ``total_interval_high`` are ``None`` whenever any
    *priced* cell has no band of its own. The alternative — the previous
    behaviour — was to contribute such a cell's bare point to both bounds, which
    made the envelope tighter for the cells that knew least about themselves.
    A total that cannot be bracketed says so.

    ``n_uncovered_models`` counts proposed models whose basis was empty under the
    filters that were applied; their cost is not in the total, so a partially-covered
    estimate says how much of the proposal it could actually price rather than quietly
    costing only the covered part as if it were the whole.

    **The applied basis filters are echoed, not just obeyed.** ``subject_id`` and
    ``template_id`` are optional, and a scoped estimate can differ from a pooled one by
    more than a factor of two — so an estimate that did not carry which filters produced
    it would be indistinguishable from a pooled estimate in the artifact an operator
    reads, saves, or pastes into a budget conversation. ``None`` means the filter was not
    applied, i.e. the basis spans the whole scope on that axis.

    **``n_test_cases`` is echoed with its origin, for the same reason.** A derived 5
    and a supplied 5 are the same number and not the same claim. ``template_id`` used
    to filter the basis without touching the grid, so naming the template you were
    about to sweep still priced ONE case — a 5x understatement for a five-case template, scaling with
    ``k_runs`` and model count. Both fields are supplied by the caller: this function
    is pure over runs and results and cannot open a template, so the count and its
    provenance are resolved by the caller — the engine's launch pricer
    (:func:`~threetears.evals.ops.history_launch_pricer`) counts them off the arm's plan — and echoed here.
    """

    cassette_mode: str
    k_runs: int
    n_test_cases: int
    #: How many settings each model runs at — a campaign launch's ``variations``, each crossed
    #: with every model into one arm. ``1`` for a plain sweep. Echoed because a model's cell
    #: prices every arm that model runs, and a reader who cannot see the multiplier reads the
    #: cell as one run's cost.
    n_settings: int = 1
    #: Where ``n_test_cases`` came from: the caller stated it, it was counted off the named
    #: template's persisted cases, it is the count a launch asked to GENERATE (an upper bound:
    #: generation de-duplicates, so it can keep fewer), or there was nothing to derive from (1).
    n_test_cases_source: Literal["supplied", "derived", "generated", "default"] = "default"
    #: The named template's persisted case count in the scope; ``0`` for a template
    #: with none, ``None`` when no template was named. Reported even on the
    #: ``supplied`` path, so a caller can see the gap between their grid and the real one.
    template_case_count: int | None = None
    #: The named template's ``candidate_kind``, ``None`` when no template was named.
    #:
    #: Carried because ``template_case_count`` does not mean the same thing for every kind,
    #: and a reader with the count alone cannot tell which reading applies. A
    #: ``conversational-turn`` template with zero persisted cases has nothing to sweep; a
    #: ``classifier`` template with zero has its cases minted at launch from the subject's
    #: snapshot bank, so zero is its ordinary state. Resolved by the caller of
    #: :func:`compute_estimate_cost` and echoed here for the same reason the count is: that
    #: function is pure over runs and results and cannot open a template.
    template_candidate_kind: str | None = None
    #: The basis filters this estimate was computed under; ``None`` = not filtered.
    subject_id: str | None = None
    template_id: str | None = None
    cells: list[CostEstimateCell] = []
    total_estimated_cost: float | None = None
    total_interval_low: float | None = None
    total_interval_high: float | None = None
    n_uncovered_models: int = 0

    def planned_costs(self, run_ids: Sequence[str] = ()) -> list[PlannedCost]:
        """Each priced cell as a cost pivot's plan: its model, the template the estimate was scoped to, its prediction.

        Args:
            run_ids: The runs the plan launched, once it has; empty before.

        Returns:
            One :class:`PlannedCost` per cell with a prediction and at least one planned observation.
        """
        return [
            PlannedCost(
                model=cell.model,
                template_id=self.template_id,
                run_ids=list(run_ids),
                n_observations=cell.n_observations,
                predicted=cell.predicted,
            )
            for cell in self.cells
            if cell.predicted is not None and cell.n_observations >= 1
        ]


def compute_estimate_cost(
    runs: list[EvalRun],
    results: list[EvalResult],
    *,
    models: list[str],
    k_runs: int,
    n_test_cases: int,
    n_settings: int = 1,
    n_test_cases_source: Literal["supplied", "derived", "generated", "default"] = "default",
    template_case_count: int | None = None,
    template_candidate_kind: str | None = None,
    cassette_mode: str = "off",
    subject_id: str | None = None,
    template_id: str | None = None,
    computed_at: str | None = None,
    profile: HostProfile,
) -> CostEstimate:
    """Predict a proposed sweep's cost from historical per-observation costs.

    The proposed sweep is ``models × n_settings × n_test_cases × k_runs`` observations — a
    campaign launch runs every model at every one of its settings, one arm per pair, so each
    model's cell prices ``n_settings`` arms. For
    each model, the historical per-observation cost distribution — drawn from the
    corpus and **filtered to the proposed cassette mode** — gives a mean, scaled
    by the proposed observation count, and a ~95% **prediction** band around it.

    **The band predicts the sweep, not the history's mean**: the variation the sweep's own
    observations will show, plus the uncertainty in where the history's distribution sits. It is
    read on the log scale, because costs are positive and right-skewed
    (:func:`~threetears.evals.analysis.stats.lognormal_sum_prediction_band`); a history holding a zero
    cost takes the normal-theory ``t(0.95, n-1) * s * sqrt(n_obs + n_obs^2/n)`` instead and says so.
    Each band states its method and assumptions in the cell's ``band_basis`` — among them that every
    observation is treated as independent, which repeats of one case are not. A confidence interval on
    the mean — the construction before either — answers a narrower question and shrinks as history
    accumulates, which is why it could publish a +/-2.4% band from two observations and then miss the
    sweep it priced by 12%.

    **Below :data:`COST_ESTIMATE_MIN_BASIS` historical observations there is no band at all**, only the point
    estimate. Two observations give one difference; a width computed from it is an
    accident, and a narrow one reads as a measurement. See
    :data:`COST_ESTIMATE_MIN_BASIS`.

    **Cassette-aware.** A ``replay`` run's marginal cost differs from a live one
    (cached tool results), so history is matched to the proposed ``cassette_mode``
    rather than pooled across modes — a replay estimate rests on replay history or
    admits it has none, never borrows a live run's higher cost. Cost pools freely
    across subjects (dollars are dollars, unlike composite quality), so
    ``subject_id`` is an optional precision filter, not a required partition.

    **Template-aware, and for a different reason than the subject filter.** A
    template is not a relabelling of the same work — it fixes the test cases, and
    with them how much the subject researches and how many tokens each turn carries.
    Two templates in one scope can sit several-fold apart (say 3.4x) on the same
    candidate model, so a mean pooled across them misprices any *specific* proposal
    in whichever direction the scope's mix happens to lean. Overpricing merely
    over-reserves; underpricing lets a sweep meet ``max_cost_usd`` partway through,
    which is the failure this surface exists to prevent. Optional like
    ``subject_id`` — omitted, the basis is the whole scope, which is the right
    answer only when the question really is "what does a run here cost on average".
    Runs carrying no ``template_id`` are ad-hoc, built from explicit test cases, so
    they can match no proposed template and are excluded rather than pooled.

    Args:
        runs: The corpus runs, supplying each result's cassette mode and subject.
        results: The corpus results, supplying per-observation ``cost_usd``.
        models: The candidate models the proposed sweep would run.
        k_runs: Proposed iterations per test case.
        n_test_cases: Proposed test-case count.
        n_settings: How many settings each model runs at — a launch's ``variations``. The
            historical basis is per MODEL, not per setting, so every setting of one model is
            priced at that model's mean: a setting that changes how much the candidate works
            (``agent.max_rounds``, a longer directive) moves its real cost off that mean in
            whichever direction it moves the work, and nothing in the history can say which.
        n_test_cases_source: Where that count came from — echoed, not used. Resolving
            it needs a template read, which this pure function cannot do.
        template_case_count: The named template's persisted case count; echoed too.
        template_candidate_kind: The named template's candidate kind; echoed too, because
            what a zero case count means is the kind's answer rather than the count's.
        cassette_mode: The proposed run's cassette mode; history is matched to it.
        subject_id: Optional — restrict the historical basis to one subject.
        template_id: Optional — restrict the historical basis to one template, which
            is what the proposed sweep will actually run.
        computed_at: The instant stamped on each cell's prediction (ISO-8601); ``None`` stamps now.
        profile: The host whose vocabulary this reads.

    Returns:
        A :class:`CostEstimate` — per-model cells, each predicted as a :class:`PredictedValue`, and a total, each banded only where its
        basis reaches the minimum; a thinner cell carries its point estimate alone, and one
        unbanded cell leaves the total unbanded too.

    Raises:
        CostEstimateError: The proposal is malformed — no models, or a
            non-positive grid.
    """
    if not models:
        raise CostEstimateError("no models proposed — nothing to estimate")
    if k_runs < 1 or n_test_cases < 1 or n_settings < 1:
        raise CostEstimateError("k_runs, n_test_cases and n_settings must all be >= 1 to estimate a sweep's cost")

    run_by_id = {run.id: run for run in runs}
    stamp = computed_at if computed_at is not None else utc_now_iso()

    def _eligible(result: EvalResult) -> bool:
        run = run_by_id.get(result.eval_run_id)
        if run is None or run.cassette_mode != cassette_mode:
            return False
        if template_id is not None and run.template_id != template_id:
            return False
        return subject_id is None or _subject_id_of(run) == subject_id

    costs_by_model: dict[str, list[float]] = {}
    unpriced_by_model: dict[str, int] = {}
    # The basis results themselves, kept beside their costs so each cell can disclose what
    # its mean was drawn across — the estimate pools history from either side of a
    # composition change and the pooled mean cannot show it.
    basis_by_model: dict[str, list[EvalResult]] = {}
    for result in results:
        if not _eligible(result):
            continue
        if result.cost_usd is None:
            unpriced_by_model[result.model] = unpriced_by_model.get(result.model, 0) + 1
            continue
        costs_by_model.setdefault(result.model, []).append(result.cost_usd)
        basis_by_model.setdefault(result.model, []).append(result)

    n_observations = n_settings * n_test_cases * k_runs
    cells: list[CostEstimateCell] = []
    total_estimate = 0.0
    total_low = 0.0
    total_high = 0.0
    n_uncovered = 0
    any_covered = False
    # One priced cell without a band makes the whole total unbracketable: adding its bare
    # point to both bounds would narrow the envelope on account of the cell that knows
    # least about itself, which is the wrong direction for a budgeting figure.
    total_bandable = True

    for model in models:
        history = costs_by_model.get(model, [])
        if not history:
            cells.append(
                CostEstimateCell(
                    model=model,
                    n_observations=n_observations,
                    n_historical=0,
                    basis="no_history",
                    n_unpriced_historical=unpriced_by_model.get(model, 0),
                )
            )
            n_uncovered += 1
            continue

        any_covered = True
        mean = sum(history) / len(history)
        estimate = n_observations * mean
        total_estimate += estimate

        low: float | None
        high: float | None
        band_basis: str | None
        if len(history) < COST_ESTIMATE_MIN_BASIS:
            # Too thin for a band. At n=1 the spread is unknown, not zero; at n=2 it is
            # one difference, which is an accident rather than a dispersion — and one
            # that prints narrow exactly when the pair lands close. The point estimate
            # stands and the absence is stated, never rendered as a false ± 0 and never
            # as a width the sample cannot support.
            low = high = band_basis = None
            total_bandable = False
        else:
            low, high, band_basis = _cost_band(history, n_observations)
            total_low += low
            total_high += high

        basis_compositions = pooled_cost_compositions(basis_by_model.get(model, []))
        cells.append(
            CostEstimateCell(
                model=model,
                n_observations=n_observations,
                n_historical=len(history),
                mean_cost_per_observation=mean,
                predicted=PredictedValue(
                    value=estimate,
                    interval_low=low,
                    interval_high=high,
                    method_id=COST_PREDICTION_METHOD,
                    computed_at=stamp,
                ),
                band_basis=band_basis,
                basis="historical",
                n_unpriced_historical=unpriced_by_model.get(model, 0),
                basis_cost_compositions=basis_compositions,
            )
        )

    if not any_covered:
        # Nothing in the corpus matched any proposed model at this cassette mode —
        # the honest answer is "cannot estimate", not a $0 total.
        return CostEstimate(
            cassette_mode=cassette_mode,
            k_runs=k_runs,
            n_test_cases=n_test_cases,
            n_settings=n_settings,
            n_test_cases_source=n_test_cases_source,
            template_case_count=template_case_count,
            template_candidate_kind=template_candidate_kind,
            subject_id=subject_id,
            template_id=template_id,
            cells=cells,
            n_uncovered_models=n_uncovered,
        )

    return CostEstimate(
        cassette_mode=cassette_mode,
        k_runs=k_runs,
        n_test_cases=n_test_cases,
        n_settings=n_settings,
        n_test_cases_source=n_test_cases_source,
        template_case_count=template_case_count,
        template_candidate_kind=template_candidate_kind,
        subject_id=subject_id,
        template_id=template_id,
        cells=cells,
        total_estimated_cost=total_estimate,
        total_interval_low=total_low if total_bandable else None,
        total_interval_high=total_high if total_bandable else None,
        n_uncovered_models=n_uncovered,
    )


__all__ = [
    "compute_estimate_cost",
    "COST_ESTIMATE_MIN_BASIS",
    "COST_PREDICTION_METHOD",
    "CostEstimate",
    "CostEstimateCell",
    "CostEstimateError",
    "PlannedCost",
    "PredictedValue",
]

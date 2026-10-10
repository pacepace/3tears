"""The read lenses as operations: a scope's pivot, its history, its export, two runs compared, and a launch's
estimated cost.

Each binds one lens of :mod:`threetears.evals.analysis` to the host — its store, its run listing and its
vocabulary — and returns the lens's own typed result, so a CLI, an MCP action and a REST route read one
shape. The computation is the lens's; nothing here re-derives an answer.

A launch estimate prices what :func:`~threetears.evals.ops.run_launch` would run from the same
arguments, by the launch's own rule: each arm planned by its kind and priced through the host's
``launch_pricer`` (:func:`~threetears.evals.run.quote_launch`).

Each result's text is here too (:func:`pivot_text`, :func:`history_text`, :func:`estimate_text`,
:func:`export_text`, :func:`runs_compared_text`), for the reason :meth:`~threetears.evals.ops.EvalSummary.render` sits with the
summary: every surface — an action, a command line — shows one rendering, and each carries the
caveats its model carries (what was left out, which runs came up short) rather than the numbers alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import Field, TypeAdapter, ValidationError

from threetears.evals.analysis.reads import RunLister, compare_two_runs, export_results, history, pivot
from threetears.evals.analysis.numbers import format_number, format_signed
from threetears.evals.analysis.reporting import (
    COST_ESTIMATE_MIN_BASIS,
    cassette_mode_disclosure,
    completeness_disclosure,
    compute_estimate_cost,
    format_significance,
    measurement_window,
    measurement_window_disclosure,
)
from threetears.evals.analysis.reporting import (
    CostEstimate,
    HistoryResult,
    PivotTable,
    PlannedCost,
    PredictedValue,
    ProjectionExclusions,
    ScoreExport,
)
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.errors import NotFoundError, ValidationFailedError
from threetears.evals.contracts.host import DEFAULT_PASS_THRESHOLD, EvalHost, pass_threshold_label
from threetears.evals.contracts.metrics import measure_title
from threetears.evals.contracts.models import EvalRun, EvalTemplate
from threetears.evals.contracts.out_of_run import OutOfRunPurpose, OutOfRunSpend
from threetears.evals.contracts.scoring import CompositeBasis
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.runs import LaunchArguments
from threetears.evals.run.launch import ArmOutcome, ArmPrice, ArmQuote, ArmVerdict, LaunchPricer, quote_launch
from threetears.evals.run.reads import list_runs


def _lister(host: EvalHost) -> RunLister:
    """The host's run listing, in the shape a lens takes it."""

    def listed(scope_id: str, **kwargs: Any) -> list[EvalRun]:
        return list_runs(host, scope_id, **kwargs)

    return listed


def scope_pivot(
    host: EvalHost,
    scope_id: str,
    *,
    row_factor: str,
    column_factor: str,
    metric: str | None = None,
    weighting: str | None = None,
    subject_id: str | None = None,
    status: str | None = "completed",
    predicted_cost: CostEstimate | LaunchEstimate | Mapping[str, Any] | None = None,
    launched_run_ids: Sequence[str] = (),
) -> PivotTable:
    """One measure over the scope's observations, aggregated over two coordinates.

    Args:
        host: The host whose store and vocabulary are read.
        scope_id: The scope.
        row_factor: The coordinate the rows are.
        column_factor: The coordinate the columns are.
        metric: The measure; ``None`` or blank takes the composite.
        weighting: The aggregation weighting; ``None`` or blank takes the disclosed default.
        subject_id: Only this subject's observations; required to pivot a judged measure over many.
        status: Only runs with this status; ``"all"`` reads every run.
        predicted_cost: The estimate made before these runs, whose predictions a cost pivot sets beside
            each planned cell's observed cost: a :class:`LaunchEstimate` (:func:`launch_estimate`'s, each priced
            arm's prediction) or an analysis :class:`~threetears.evals.analysis.reporting.CostEstimate` — the
            model, or its JSON form as a caller across a wire holds it.
        launched_run_ids: The runs the estimated launch made, once it has. Each cell beside a prediction then
            says how many of its observations came from other runs (``PivotCell.n_unplanned``) — the history
            the prediction was drawn from among them. Refused without ``predicted_cost``.

    Returns:
        The table.

    Raises:
        ValidationFailedError: The pivot cannot be answered honestly — see
            :func:`~threetears.evals.analysis.pivot` — or ``predicted_cost`` is neither estimate.
    """
    plan: list[PlannedCost] | None = None
    if predicted_cost is not None:
        try:
            # Validated as exactly one of the two shapes — each forbids the other's fields — never sniffed.
            estimate = _PREDICTED_COST.validate_python(
                predicted_cost if isinstance(predicted_cost, Mapping) else predicted_cost.model_dump(mode="json")
            )
        except ValidationError as e:
            raise ValidationFailedError(f"predicted_cost is neither a launch estimate nor a cost estimate: {e}") from e
        plan = estimate.planned_costs(launched_run_ids)
    elif launched_run_ids:
        raise ValidationFailedError(
            "launched_run_ids names the runs an estimate's launch made, and no predicted_cost was given to set "
            "them against"
        )
    return pivot(
        host.storage,
        scope_id,
        list_runs=_lister(host),
        row_factor=row_factor,
        column_factor=column_factor,
        metric=metric,
        weighting=weighting,
        subject_id=subject_id,
        status=status,
        predicted_cost=plan,
        profile=host.profile,
    )


def scope_history(
    host: EvalHost,
    scope_id: str,
    *,
    metric: str | None = None,
    min_absolute_change: float = 0.0,
    min_relative_change: float = 0.0,
    subject_id: str | None = None,
    status: str | None = "completed",
) -> HistoryResult:
    """One measure over time for each contestant in the scope, with its regressions flagged.

    Args:
        host: The host whose store and vocabulary are read.
        scope_id: The scope.
        metric: The measure; ``None`` or blank takes the composite.
        min_absolute_change: The smallest move a regression flag counts, in the measure's unit.
        min_relative_change: The smallest move relative to the baseline a flag counts, as a fraction.
        subject_id: Only this subject's observations.
        status: Only runs with this status; ``"all"`` reads every run.

    Returns:
        The series.

    Raises:
        ValidationFailedError: The metric cannot be seriesed, or the status is no run's.
    """
    return history(
        host.storage,
        scope_id,
        list_runs=_lister(host),
        metric=metric,
        min_absolute_change=min_absolute_change,
        min_relative_change=min_relative_change,
        subject_id=subject_id,
        status=status,
        profile=host.profile,
    )


def scope_export(
    host: EvalHost,
    scope_id: str,
    *,
    format: str | None = None,
    status: str | None = "completed",
    run_ids: list[str] | None = None,
) -> ScoreExport:
    """The scope's observations as flat rows, in CSV or JSON, for analysis elsewhere.

    Args:
        host: The host whose store and vocabulary are read.
        scope_id: The scope.
        format: ``"csv"`` (the default) or ``"json"``.
        status: Only runs with this status; ``"all"`` exports every run.
        run_ids: Only these runs — archived ones included, since they were named.

    Returns:
        The export.

    Raises:
        ValidationFailedError: The format is not one there is, or the status is no run's.
    """
    return export_results(
        host.storage,
        scope_id,
        list_runs=_lister(host),
        fmt=format,
        status=status,
        run_ids=run_ids,
        profile=host.profile,
    )


class RunsCompared(EvalBaseModel):
    """One run's arm against another's, with what either run could not deliver said beside the numbers.

    ``comparison`` is the lens's own answer (:func:`~threetears.evals.analysis.reads.compare_two_runs`), carried
    whole: this operation adds what a head-to-head needs beside it and re-derives none of it. A delta between a
    run that delivered one of its five cells and a complete one is a comparison of two different populations,
    so each short run's sentence is carried with it, as it is on every other comparison surface.
    """

    baseline_run_id: str = Field(description="The run read as the baseline (A).")
    candidate_run_id: str = Field(description="The run read against it (B).")
    comparison: dict[str, Any] = Field(
        description=(
            "The two-run lens's answer: `comparison_basis` (whether the runs share a template, not the test's basis), "
            "`composite_comparability` (why composites are withheld, across subjects), `comparison.arm` (each "
            "arm's model, pass^k at one shared depth and mean composite, their deltas, whether the samples were "
            "paired and over how many cases, Hedges' g, p and the verdict), `comparison.per_template` and each "
            "run's `subject_detail_{a,b}`."
        )
    )
    completeness_disclosures: dict[str, str] = Field(
        description=(
            "Run id -> the sentence for each of the two runs that delivered less than its matrix; empty when both "
            "are whole. A run absent from it is not asserted complete: one carrying no completeness record has "
            "nothing to disclose either way."
        )
    )
    measurement_window_disclosure: str | None = Field(
        description="When the two runs were measured over spans that do not overlap, the sentence saying so; None "
        "when they overlap or either produced no result to date a span from."
    )
    cassette_mode_disclosure: str | None = Field(
        description="When the two runs recorded different cassette modes (one replayed what the other measured "
        "live), the sentence saying so; None when they recorded the same."
    )


def runs_compare(host: EvalHost, baseline_run_id: str, candidate_run_id: str, scope_id: str) -> RunsCompared:
    """Compare two runs' arms, with each run's completeness, clock and cassette disclosures beside the numbers.

    The comparison is :func:`~threetears.evals.analysis.reads.compare_two_runs`'s, read under the engine's
    default pass threshold; the disclosures are the ones every other comparison surface carries, computed by the
    same helpers.

    Args:
        host: The host whose store is read.
        baseline_run_id: The run read as the baseline.
        candidate_run_id: The run read against it.
        scope_id: The scope both runs live in.

    Returns:
        The comparison and its disclosures.

    Raises:
        NotFoundError: Either run is not in the scope.
    """
    storage = host.storage

    def template(template_id: str) -> EvalTemplate:
        found = storage.load_template(template_id, scope_id)
        if found is None:
            raise NotFoundError("template", template_id)
        return found

    def subject_detail(run: EvalRun) -> dict[str, dict[str, Any]]:
        snapshot = run.subject_snapshot
        return {snapshot.subject_id: snapshot.model_dump(mode="json", include={"subject_label", "components"})}

    comparison = compare_two_runs(
        storage, baseline_run_id, candidate_run_id, scope_id, load_template=template, subject_detail=subject_detail
    )
    runs: list[EvalRun] = []
    for run_id in (baseline_run_id, candidate_run_id):
        run = storage.load_eval_run(run_id, scope_id)
        if run is None:  # compare_two_runs has already refused a missing run; this keeps the type honest
            raise NotFoundError("run", run_id)
        runs.append(run)
    windows = [
        window
        for run in runs
        if (window := measurement_window(run.id, storage.query_eval_results_by_run(run.id, scope_id))) is not None
    ]
    return RunsCompared(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        comparison=comparison,
        completeness_disclosures={
            run.id: sentence for run in runs if (sentence := completeness_disclosure(run.completeness)) is not None
        },
        measurement_window_disclosure=measurement_window_disclosure(windows),
        cassette_mode_disclosure=cassette_mode_disclosure({run.id: run.cassette_mode for run in runs}),
    )


class ArmEstimate(EvalBaseModel):
    """One arm of a launch estimate: what its kind planned, what the host's pricer predicted, and what the launch would do.

    Attributes:
        described: The arm, as the launch's refusal names it.
        candidate_model: The model it runs on — its plan's, or the one named when its kind plans nothing
            (``None`` there for an arm on the kind's unnamed default).
        case_count: The cases its plan bounds it to (or the hypothetical count the estimate was asked for);
            ``None`` when its kind plans nothing.
        case_source: ``stored`` for an arm over the template's stored cases, ``generated`` for one whose cases
            its launch generates.
        n_observations: ``case_count × k_runs``, the observations it would make; ``None`` with no plan.
        predicted_usd: The figure the launch holds to the arm's cap — a range pricer's upper end — or ``None``
            when nothing predicts it (unknown, never $0).
        central_usd: A range pricer's central estimate, never held to the cap; ``None`` for a single figure.
        low_usd: A range pricer's lower end; ``None`` otherwise.
        method_id: The estimator, or ``None`` for a pricer that names none.
        basis: How the prediction was made, or why there is none.
        outcome: What the launch's one pricing rule makes of it (:data:`~threetears.evals.run.ArmOutcome`).
        refusal: The refusal the launch would make of it, word for word; ``None`` unless refused.
    """

    described: str
    candidate_model: str | None
    case_count: int | None
    case_source: Literal["stored", "generated"]
    n_observations: int | None
    predicted_usd: float | None
    central_usd: float | None
    low_usd: float | None
    method_id: str | None
    basis: str
    outcome: ArmOutcome
    refusal: str | None


class LaunchEstimate(EvalBaseModel):
    """What a launch would cost and whether it would launch, priced by the launch's own rule.

    Built by :func:`~threetears.evals.run.quote_launch` — the launch's argument refusals, its dispatch, each
    arm's plan (``LaunchableKind.plan_arm``) and each arm's price through the host's ``launch_pricer`` — so the
    figure an operator reads here is the figure the launch holds each arm's cap to, and an arm this estimate
    shows refused is an arm the launch refuses, in the same words. It spends nothing.

    Attributes:
        template_id: The template.
        subject_id: The subject the launch would measure.
        cassette_mode: The cassette mode, normalised.
        k_runs: Repeats per case.
        n_variations: Cases the launch would generate first; ``0`` for stored cases.
        hypothetical_case_count: The case count each planned arm was priced at in place of its plan's, for a
            hypothetical grid; ``None`` when every arm is priced at its plan, as the launch prices it.
        cap_usd: The cap each arm's run would be held to; ``None`` when the host enforces none.
        cap_origin: ``chosen`` when the launch named the cap, ``inherited`` from the host's; ``None`` uncapped.
        arms: Every arm, in arm order.
        total_predicted_usd: Every arm's held-to figure, summed — ``None`` when any arm has none, since a total
            missing an arm is not the launch's total.
        would_launch: Whether no arm would be refused on its price.
        computed_at: When the estimate was made, the stamp a pivot's prediction carries.
    """

    template_id: str
    subject_id: str
    cassette_mode: str
    k_runs: int
    n_variations: int
    hypothetical_case_count: int | None
    cap_usd: float | None
    cap_origin: Literal["chosen", "inherited"] | None
    arms: list[ArmEstimate]
    total_predicted_usd: float | None
    would_launch: bool
    computed_at: str

    def planned_costs(self, run_ids: Sequence[str] = ()) -> list[PlannedCost]:
        """Each priced arm as a cost pivot's plan: its model, its template, its observations and its prediction.

        Args:
            run_ids: The runs the launch made, once it has (the run ids its jobs name); empty before, when the
                cost a pivot sets beside each prediction is history the launch did not make.

        Returns:
            One :class:`~threetears.evals.analysis.reporting.PlannedCost` per arm with a model, a plan and a
            prediction; the prediction's point is the pricer's central estimate where it gave a range (the
            band then running from its lower to its upper end) and its single figure otherwise.
        """
        planned: list[PlannedCost] = []
        for arm in self.arms:
            if arm.candidate_model is None or arm.n_observations is None or arm.predicted_usd is None:
                continue
            ranged = arm.central_usd is not None
            planned.append(
                PlannedCost(
                    model=arm.candidate_model,
                    template_id=self.template_id,
                    run_ids=list(run_ids),
                    n_observations=arm.n_observations,
                    predicted=PredictedValue(
                        value=arm.central_usd if arm.central_usd is not None else arm.predicted_usd,
                        interval_low=arm.low_usd if ranged else None,
                        interval_high=arm.predicted_usd if ranged else None,
                        method_id=arm.method_id or LAUNCH_PRICER_METHOD,
                        computed_at=self.computed_at,
                    ),
                )
            )
        return planned


#: The two shapes a cost pivot's plan arrives in, validated as exactly one.
_PREDICTED_COST: TypeAdapter[CostEstimate | LaunchEstimate] = TypeAdapter(CostEstimate | LaunchEstimate)

#: The method a pivot's prediction names when the host's pricer named none.
LAUNCH_PRICER_METHOD = "launch-pricer"


async def launch_estimate(
    host: OpsHost, arguments: LaunchArguments, scope_id: str, *, n_test_cases: int | None = None
) -> LaunchEstimate:
    """What :func:`~threetears.evals.ops.run_launch` with ``arguments`` would cost, priced by the launch's own rule.

    The launch's own steps, read-only (:func:`~threetears.evals.run.quote_launch`): every refusal it makes before
    pricing is raised as it raises it — a kind's request-level refusals among them — and every arm is planned by
    its kind and priced through the host's ``launch_pricer``, held to the cap the launch would hold it to. So
    the estimate and the launch never price one arm two ways. Nothing is admitted, generated or spent. The
    generation calls a generating launch makes first are priced separately, against the host's out-of-run cap,
    by the launch itself (``EvalStorage.query_out_of_run_spend`` reads what they spent).

    Args:
        host: The launching host.
        arguments: What the launch would name.
        scope_id: The scope the launch would run in.
        n_test_cases: A case count to price each planned arm at in place of its plan's, for a hypothetical grid;
            ``None`` prices each at its plan, as the launch would.

    Returns:
        The estimate.

    Raises:
        NotFoundError: The template is not in the scope.
        ValidationFailedError: Any refusal the launch makes before pricing an arm, or an ``n_test_cases``
            below one.
    """
    quote = await quote_launch(
        host.launch,
        template_id=arguments.template_id,
        subject_id=arguments.subject_id,
        models=list(arguments.models),
        k_runs=arguments.k_runs,
        n_variations=arguments.n_variations,
        variation_model=arguments.variation_model,
        overlays=arguments.overlays,
        apparatus_settings=arguments.apparatus_settings,
        max_cost_usd=arguments.max_cost_usd,
        judge_model=arguments.judge_model,
        simulator_model=arguments.simulator_model,
        scope_id=scope_id,
        case_count=n_test_cases,
    )
    arms = [_arm_estimate(arm, quote.k_runs, quote.n_variations) for arm in quote.arms]
    predicted = [arm.predicted_usd for arm in arms]
    return LaunchEstimate(
        template_id=quote.template_id,
        subject_id=quote.subject_id,
        cassette_mode=quote.cassette_mode,
        k_runs=quote.k_runs,
        n_variations=quote.n_variations,
        hypothetical_case_count=quote.case_count,
        cap_usd=quote.arms[0].cap_usd,
        cap_origin=quote.arms[0].cap_origin,
        arms=arms,
        total_predicted_usd=None if None in predicted else math.fsum(p for p in predicted if p is not None),
        would_launch=all(arm.outcome != "refused" for arm in arms),
        computed_at=datetime.now(UTC).isoformat(),
    )


def _arm_estimate(arm: ArmVerdict, k_runs: int, n_variations: int) -> ArmEstimate:
    """One arm's verdict, as the estimate reports it.

    Args:
        arm: The verdict.
        k_runs: Repeats per case.
        n_variations: Cases the launch generates; ``0`` for stored cases.

    Returns:
        The arm.
    """
    plan = arm.plan
    price = arm.price
    return ArmEstimate(
        described=arm.described,
        candidate_model=arm.candidate_model,
        case_count=plan.case_count if plan is not None else None,
        case_source="generated" if n_variations > 0 else "stored",
        n_observations=plan.case_count * k_runs if plan is not None else None,
        predicted_usd=price.predicted_usd,
        central_usd=price.central_usd,
        low_usd=price.low_usd,
        method_id=price.method_id,
        basis=price.basis,
        outcome=arm.outcome,
        refusal=arm.refusal,
    )


def history_launch_pricer(host: EvalHost) -> LaunchPricer:
    """The engine's launch pricer: an arm bounded from the scope's usage history of runs launched as it will be.

    What a :class:`~threetears.evals.run.LaunchHost` takes as ``launch_pricer`` to price every arm of a launch
    before any launcher runs — an arm over the template's stored cases and one whose cases its launch generates
    alike, priced by this one rule and counted by its source (``derived`` from the template's stored cases, or
    ``generated``). **The history is the runs launched the way the arm will be**: the same template on the
    same candidate model at the same cassette mode, scored by the same judges (the model each scored dim was
    requested from, ``EvalRun.effective_judges``, against the arm's planned judges — so a dim whose config names
    its own model, and a role default that has moved since, are both matched by the model that actually scored),
    driven by the same simulator (the resolved ``EvalRun.simulator_model``) and with the same resolved apparatus
    settings, archived runs included. A run judged by another model, or with its rig set up otherwise, spent
    differently, and pricing an arm from it would bias the prediction by whatever the difference costs — low,
    when the arm's judge is the dearer one, and the generation paid before the run's own cap could say so. What
    it does not match: a judge config's prompt — two configs on one model are one judge here, a prompt moving a
    judgement's cost far less than a model does. Only the matching runs' results are read, one run at a time,
    rather than every result in the scope for every arm.

    Each arm is priced from those results' per-observation costs scaled to the arm's planned cases and repeats,
    and **its prediction is the band's upper end**, since the launch holds that figure to the arm's cap
    (:class:`~threetears.evals.run.ArmPrice`): a central estimate admits an arm that then runs past its cap
    about as often as the run lands above the centre. The central estimate and the lower end ride along beside
    it. A history too thin to band (fewer than :data:`~threetears.evals.analysis.COST_ESTIMATE_MIN_BASIS` priced
    past results) bounds nothing, and predicts nothing, as does a scope with no priced history of the
    condition — the launch reads both as unknown.

    Args:
        host: The host whose store holds the history and whose vocabulary reads it.

    Returns:
        The pricer.
    """

    def price(quote: ArmQuote) -> ArmPrice:
        judges = dict(quote.judge.effective_judges) if quote.judge is not None else None
        condition = (
            f"template {quote.template_id!r} on {quote.candidate_model!r} at cassette mode {quote.cassette_mode!r}, "
            f"judged by {judges!r}, simulator {quote.simulator_model!r} and apparatus settings "
            f"{dict(quote.apparatus_settings)!r}"
        )
        runs = [run for run in list_runs(host, quote.scope_id, include_archived=True) if _launched_as(run, quote)]
        results = [result for run in runs for result in host.storage.query_eval_results_by_run(run.id, quote.scope_id)]
        estimate = compute_estimate_cost(
            runs,
            results,
            models=[quote.candidate_model],
            k_runs=quote.k_runs,
            n_test_cases=quote.case_count,
            n_test_cases_source="generated" if quote.case_source == "generated" else "derived",
            cassette_mode=quote.cassette_mode,
            template_id=quote.template_id,
            profile=host.profile,
        )
        [cell] = estimate.cells
        predicted = cell.predicted
        if predicted is None:
            unpriced = (
                f"; {cell.n_unpriced_historical} past result(s) ran unpriced and cannot be drawn on"
                if cell.n_unpriced_historical
                else ""
            )
            return ArmPrice(
                predicted_usd=None,
                basis=f"no priced past result of {condition} is in scope {quote.scope_id!r}{unpriced}",
            )
        if predicted.interval_high is None:
            return ArmPrice(
                predicted_usd=None,
                basis=(
                    f"{cell.n_historical} priced past result(s) of {condition} are too few to bound the arm (a "
                    f"band needs {COST_ESTIMATE_MIN_BASIS}; their mean alone puts it at ${predicted.value:.2f})"
                ),
            )
        return ArmPrice(
            predicted_usd=predicted.interval_high,
            central_usd=predicted.value,
            low_usd=predicted.interval_low,
            method_id=predicted.method_id,
            basis=(
                f"the upper end of the band ${predicted.interval_low or 0.0:.2f}-${predicted.interval_high:.2f} "
                f"around ${predicted.value:.2f}, method {predicted.method_id}, from {cell.n_historical} priced past "
                f"result(s) of {condition}" + (f" ({cell.band_basis})" if cell.band_basis else "")
            ),
        )

    return price


def _launched_as(run: EvalRun, quote: ArmQuote) -> bool:
    """Whether ``run`` was launched as ``quote``'s arm will be: its template, model, cassette mode, judges, simulator and rig.

    Args:
        run: A run from the scope's history.
        quote: The arm being priced.

    Returns:
        True when every launch condition that moves what a run spends matches — the judges and the simulator
        by the models that ran, not by what the launch named.
    """
    judges = dict(quote.judge.effective_judges) if quote.judge is not None else None
    return (
        run.template_id == quote.template_id
        and run.candidate_model == quote.candidate_model
        and run.cassette_mode == quote.cassette_mode
        and run.effective_judges == judges
        and run.simulator_model == quote.simulator_model
        and run.apparatus_settings == dict(quote.apparatus_settings)
    )


class OutOfRunSpendTotals(EvalBaseModel):
    """What a set of out-of-run calls spent, summed — with what could not be summed counted beside it.

    **Missing is not zero.** A call that reported no cost, and every call that raised (which reports
    nothing, and may have been billed), are counted in ``n_unpriced`` and left out of ``priced_usd``,
    so ``priced_usd`` is a floor whenever ``n_unpriced`` is not 0.

    Attributes:
        n_calls: The calls.
        n_raised: Those that raised rather than returning.
        n_unpriced: Those whose cost is unknown — a completed call that reported none, or a raised call.
        priced_usd: The reported cost of the priced calls, summed.
        ceiling_usd: The ceilings the calls were admitted at, summed over those the client could price.
        n_unbounded: Those admitted with no ceiling (the client could not price them and no cap was enforced).
    """

    n_calls: int
    n_raised: int
    n_unpriced: int
    priced_usd: float
    ceiling_usd: float
    n_unbounded: int

    @classmethod
    def of(cls, rows: Sequence[OutOfRunSpend]) -> OutOfRunSpendTotals:
        """Sum ``rows``.

        Args:
            rows: Ledger rows.

        Returns:
            Their totals.
        """
        return cls(
            n_calls=len(rows),
            n_raised=sum(1 for row in rows if row.outcome == "raised"),
            n_unpriced=sum(1 for row in rows if row.cost_usd is None),
            priced_usd=math.fsum(row.cost_usd for row in rows if row.cost_usd is not None),
            ceiling_usd=math.fsum(row.priced_ceiling_usd for row in rows if row.priced_ceiling_usd is not None),
            n_unbounded=sum(1 for row in rows if row.priced_ceiling_usd is None),
        )


class OutOfRunSpendReport(EvalBaseModel):
    """The calls the engine made outside any run in a scope — case generations, rubric proposals and analysis
    generations — and their totals.

    Read off the out-of-run ledger (``EvalStorage.query_out_of_run_spend``), the one record of spend no run's
    cost carries: a run's results sum what its cells spent, never what was spent writing its cases.

    Attributes:
        scope_id: The scope read.
        purpose: The purpose the read was narrowed to, or ``None`` for every purpose.
        launch_group_id: The launch the read was narrowed to, or ``None`` for every launch.
        template_id: The template the read was narrowed to, or ``None`` for every template.
        rows: Every matching call, oldest first.
        totals: Every matching call, summed.
        by_purpose: The totals per purpose that appears, in the order purposes first appear.
        by_launch: The totals per launch group that appears (case generations; a proposal or an analysis belongs to none, and
            is not listed here), in the order launches first appear.
    """

    scope_id: str
    purpose: OutOfRunPurpose | None
    launch_group_id: str | None
    template_id: str | None
    rows: list[OutOfRunSpend]
    totals: OutOfRunSpendTotals
    by_purpose: dict[str, OutOfRunSpendTotals]
    by_launch: dict[str, OutOfRunSpendTotals]


def scope_out_of_run_spend(
    host: EvalHost,
    scope_id: str,
    *,
    purpose: OutOfRunPurpose | None = None,
    launch_group_id: str | None = None,
    template_id: str | None = None,
) -> OutOfRunSpendReport:
    """What the engine spent outside any run in a scope, call by call and summed, optionally narrowed.

    Args:
        host: The host whose store holds the ledger.
        scope_id: The scope to read.
        purpose: Only calls made for this purpose (``variation``, ``proposer`` or ``analysis``).
        launch_group_id: Only the calls one launch's case generation made; its runs carry the same group id.
        template_id: Only calls made for this template.

    Returns:
        The report.
    """
    rows = host.storage.query_out_of_run_spend(
        scope_id, purpose=purpose, launch_group_id=launch_group_id, template_id=template_id
    )
    by_purpose: dict[str, list[OutOfRunSpend]] = {}
    by_launch: dict[str, list[OutOfRunSpend]] = {}
    for row in rows:
        by_purpose.setdefault(row.purpose, []).append(row)
        if row.launch_group_id is not None:
            by_launch.setdefault(row.launch_group_id, []).append(row)
    return OutOfRunSpendReport(
        scope_id=scope_id,
        purpose=purpose,
        launch_group_id=launch_group_id,
        template_id=template_id,
        rows=rows,
        totals=OutOfRunSpendTotals.of(rows),
        by_purpose={name: OutOfRunSpendTotals.of(group) for name, group in by_purpose.items()},
        by_launch={name: OutOfRunSpendTotals.of(group) for name, group in by_launch.items()},
    )


def _totals_line(totals: OutOfRunSpendTotals) -> str:
    """One line of totals, saying when the sum is a floor."""
    floor = f", {totals.n_unpriced} unpriced (so at least)" if totals.n_unpriced else ""
    raised = f", {totals.n_raised} raised" if totals.n_raised else ""
    unbounded = f", {totals.n_unbounded} admitted unbounded" if totals.n_unbounded else ""
    return (
        f"{totals.n_calls} call(s): ${totals.priced_usd:.4f} reported{floor}{raised}; admitted at up to "
        f"${totals.ceiling_usd:.4f}{unbounded}"
    )


def out_of_run_spend_text(report: OutOfRunSpendReport) -> str:
    """The scope's out-of-run spend as an operator reads it: the totals, per purpose and launch, then each call.

    Args:
        report: What :func:`scope_out_of_run_spend` returned.

    Returns:
        The text.
    """
    narrowed = ", ".join(
        f"{name} {value}"
        for name, value in (
            ("purpose", report.purpose),
            ("launch", report.launch_group_id),
            ("template", report.template_id),
        )
        if value is not None
    )
    lines = [f"out-of-run spend in scope {report.scope_id}" + (f" ({narrowed})" if narrowed else "")]
    if not report.rows:
        lines.append("no out-of-run call is ledgered here")
        return "\n".join(lines)
    lines.append(f"total: {_totals_line(report.totals)}")
    lines += [f"purpose {name}: {_totals_line(totals)}" for name, totals in report.by_purpose.items()]
    lines += [f"launch {name}: {_totals_line(totals)}" for name, totals in report.by_launch.items()]
    for row in report.rows:
        cost = f"${row.cost_usd:.4f}" if row.cost_usd is not None else "unpriced"
        ceiling = f"${row.priced_ceiling_usd:.4f}" if row.priced_ceiling_usd is not None else "unbounded"
        failed = f" {row.failure}" if row.failure else ""
        lines.append(
            f"  {row.created_at}  {row.purpose}  {row.model}  {row.outcome}{failed}  {cost} (ceiling {ceiling})"
            f"  template {row.template_id}  launch {row.launch_group_id}"
            + (f"  campaign {row.campaign_id}" if row.campaign_id is not None else "")
        )
    return "\n".join(lines)


# --- the text an operator or an agent reads ---------------------------------------------------------


def _exclusions(exclusions: ProjectionExclusions) -> list[str]:
    """What never became a row, when anything did not — an all-excluded answer must not read as empty."""
    if not exclusions.total:
        return []
    return [
        f"excluded: {exclusions.total} observation(s) — {exclusions.results_outside_queried_runs} from runs the "
        f"filters left out, {exclusions.results_from_archived_runs} from archived runs, "
        f"{exclusions.results_without_run} whose run is missing"
    ]


def _completeness(disclosures: Mapping[str, str], n_degraded: int | None = None) -> list[str]:
    """The runs that measured less than they promised, one line each."""
    if not disclosures:
        return []
    weight = f" ({n_degraded} observation(s) from them)" if n_degraded is not None else ""
    return [f"incomplete runs{weight}:", *(f"- {run_id}: {sentence}" for run_id, sentence in disclosures.items())]


def _predicted(predicted: PredictedValue | None, n_unplanned: int | None = None) -> str:
    """A prediction, set apart from the observation it sits beside, and how much of that observation the plan made."""
    if predicted is None:
        return ""
    population = (
        " — the observed cost here is history the plan did not make (it names no launched run)"
        if n_unplanned is None
        else f" — {n_unplanned} observation(s) here are from runs the plan did not make"
        if n_unplanned
        else ""
    )
    band = (
        f" [{format_number(predicted.interval_low)}, {format_number(predicted.interval_high)}]"
        if predicted.interval_low is not None and predicted.interval_high is not None
        else ""
    )
    return f"; predicted {format_number(predicted.value)}{band} ({predicted.method_id}){population}"


def pivot_text(table: PivotTable) -> str:
    """A pivot as text: what was computed, each cell with its denominators, and every caveat the table carries."""
    lines = [
        f"pivot of {measure_title(table.metric)} by {table.row_factor} (rows) x {table.column_factor} (columns), "
        f"{table.weighting}: {table.n_observations} observation(s), {table.n_filtered_out} filtered out",
        f"formula: {table.formula}",
    ]
    for cell in table.cells:
        spread = f", sem {format_number(cell.sem)}" if cell.sem is not None else ""
        unmeasured = f", {cell.n_unmeasured} unmeasured" if cell.n_unmeasured else ""
        # A cell's compositions are worth a reader's eye only where the table's differ; otherwise they repeat.
        roles = (
            "; cost over " + " | ".join("+".join(roles) for roles in cell.cost_compositions)
            if table.cost_compositions_differ and cell.cost_compositions
            else ""
        )
        versions = "".join(
            f"; pools {key} versions {', '.join(f'v{version}' for version in found)}"
            for key, found in cell.identity_versions.items()
        )
        basis = (
            f"; meaned over {_basis_sets(cell.composite_basis.model_dump())}"
            if table.composite_bases_differ and cell.composite_basis is not None
            else ""
        )
        withheld = f"; withheld: it {cell.withheld}" if cell.withheld else ""
        substituted = f"; {cell.substitution_disclosure}" if cell.substitution_disclosure else ""
        lines.append(
            f"- {cell.row} / {cell.column}: {format_number(cell.value)} ({cell.status}; n={cell.n}, "
            f"{cell.n_cases} case(s){spread}{unmeasured}){roles}{basis}{versions}{withheld}{substituted}"
            f"{_predicted(cell.predicted, cell.n_unplanned)}"
        )
    if not table.cells:
        lines.append("- no cells")
    for flag in table.simpsons_flags:
        lines.append(
            f"Simpson's reversal: pooled, {flag.pooled_leader} reads higher of {flag.column_a} vs {flag.column_b}, "
            f"but {flag.rows_disagreeing} row(s) order them the other way ({', '.join(flag.disagreeing_rows)}) "
            f"against {flag.rows_agreeing} that agree — orders of point values, none tested; do not read the pooled "
            "order as a ranking"
        )
    if table.cost_compositions_differ:
        lines.append(
            "cost compositions differ: these dollars were not all summed over the same roles, so a cheaper cell "
            "may only have priced fewer things — each cell names what it covered"
        )
    if table.composite_bases_differ:
        lines.append(
            "ragged composite: these composites were not all meaned over the same dimensions, so a difference "
            "between cells may be a difference in what was averaged — each cell names the sets it pooled"
        )
    if table.cassette_mode_disclosure:
        lines.append(table.cassette_mode_disclosure)
    if table.identity_pooling_disclosure:
        lines.append(table.identity_pooling_disclosure)
    if table.unplaced_predicted_models:
        lines.append(f"planned and in no cell here: {', '.join(table.unplaced_predicted_models)}")
    lines += _exclusions(table.exclusions)
    lines += _completeness(table.completeness_disclosures, table.n_degraded_observations)
    return "\n".join(lines)


def history_text(result: HistoryResult) -> str:
    """A history as text: each contestant's series, oldest first, with each step's verdict and its test."""
    direction = {True: "higher is better", False: "lower is better", None: "no better direction"}
    lines = [
        f"history of {result.metric} ({direction[result.higher_is_better]}), {result.weighting}: "
        f"{len(result.series)} series over {result.n_results} result(s), {result.n_filtered_out} filtered out",
        f"formula: {result.formula}",
        f"regression thresholds: min absolute change {format_number(result.min_absolute_change)}, "
        f"min relative change {format_number(result.min_relative_change)}",
        (
            "equivalence margin: none declared for this measure, so no step can read equivalent; "
            "not_separated says the data cannot tell a move from noise, never that nothing changed"
            if result.equivalence_margin is None
            else f"equivalence margin: ±{format_number(result.equivalence_margin)} (the measure's declared "
            "materiality threshold); a step reads equivalent only when shown inside it"
        ),
    ]
    if result.attribution_disclosure:
        lines.append(result.attribution_disclosure)
    if result.identity_span_disclosure:
        lines.append(result.identity_span_disclosure)
    for series in result.series:
        lines.append(f"## {series.model} — subject {series.subject_label or series.subject_id}")
        if series.identity_version_disclosure:
            lines.append(series.identity_version_disclosure)
        previous_basis: CompositeBasis | None = None
        for point in series.points:
            flag = point.regression
            verdict = f"; {flag.label} vs previous ({flag.test})" if flag is not None else ""
            baseline = (
                " (baseline)" if point.is_baseline else f", {format_signed(point.delta_from_baseline)} from baseline"
            )
            epoch = ", suite changed here" if point.epoch_boundary else ""
            short = f"; {point.completeness_disclosure}" if point.completeness_disclosure else ""
            ragged = (
                f"; {point.composite_basis.disclosure()}"
                if point.composite_basis is not None and point.composite_basis.ragged
                else ""
            )
            if (
                point.composite_basis is not None
                and previous_basis is not None
                and point.composite_basis.bases != previous_basis.bases
            ):
                ragged += f"; composite basis changed here, to {_basis_sets(point.composite_basis.model_dump())}"
            previous_basis = point.composite_basis or previous_basis
            lines.append(
                f"- {point.created_at} {point.run_id}: {format_number(point.value)} (n={point.n}, "
                f"{point.n_cases} case(s)){baseline}{epoch}{verdict}{short}{ragged}"
            )
    if not result.series:
        lines.append("- no series")
    lines += _exclusions(result.exclusions)
    return "\n".join(lines)


def estimate_text(estimate: LaunchEstimate) -> str:
    """An estimate as text: the launch priced, each arm's price and outcome, and the total."""
    grid = (
        f"{estimate.hypothetical_case_count} case(s) (hypothetical)"
        if estimate.hypothetical_case_count is not None
        else (f"{estimate.n_variations} generated case(s)" if estimate.n_variations else "its stored cases")
    )
    cap = (
        f"each arm capped at ${format_number(estimate.cap_usd)} ({estimate.cap_origin})"
        if estimate.cap_usd is not None
        else "no cap in force"
    )
    lines = [
        f"estimate: template {estimate.template_id}, subject {estimate.subject_id}, {grid} x k_runs {estimate.k_runs}, "
        f"cassette mode {estimate.cassette_mode}; {cap}"
    ]
    for arm in estimate.arms:
        figure = (
            f"${format_number(arm.predicted_usd)}"
            + (f" (central ${format_number(arm.central_usd)})" if arm.central_usd is not None else "")
            if arm.predicted_usd is not None
            else "unpriced"
        )
        lines.append(f"- {arm.described}: {figure} — {arm.outcome}; {arm.basis}")
        if arm.refusal is not None:
            lines.append(f"  refused: {arm.refusal}")
    if estimate.total_predicted_usd is None:
        lines.append("total: none — an arm has no prediction, so no total is the launch's")
    else:
        lines.append(f"total: ${format_number(estimate.total_predicted_usd)}")
    lines.append("would launch" if estimate.would_launch else "would be refused")
    return "\n".join(lines)


def export_text(export: ScoreExport) -> str:
    """An export as text: a line of its row count and what it left out, then the body itself."""
    lines = [f"export ({export.format}, {export.n_records} row(s))"]
    lines += _exclusions(export.exclusions)
    lines += _completeness(export.completeness_disclosures)
    return "\n".join(lines) + "\n\n" + export.body


def _basis_sets(basis: Mapping[str, Any] | None) -> str:
    """A pooled composite's dimension sets, as text: ``{a, b} | {c}``, marked ragged when there are several."""
    if not basis:
        return "nothing"
    sets = " | ".join("{" + ", ".join(dims) + "}" for dims in basis.get("bases", [])) or "{}"
    return f"{sets} (ragged)" if basis.get("ragged") else sets


def runs_compared_text(compared: RunsCompared) -> str:
    """Two runs compared as text: the arms, each reading with its delta and test, then every disclosure."""
    view = compared.comparison
    arm: Mapping[str, Any] = view.get("comparison", {}).get("arm", {})
    paired = bool(arm.get("paired"))
    lines = [
        f"run {compared.baseline_run_id} ({arm.get('model_a')}) against run {compared.candidate_run_id} "
        f"({arm.get('model_b')}); "
        + (f"paired over {arm.get('n_pairs')} shared case(s)" if paired else "unpaired: no case scored in both"),
        f"{pass_threshold_label(arm.get('k'), view.get('rubric_threshold', DEFAULT_PASS_THRESHOLD))}: "
        f"{format_number(arm.get('pass_hat_k_a'))} vs "
        f"{format_number(arm.get('pass_hat_k_b'))} (delta {format_signed(arm.get('pass_hat_k_delta'))}; "
        f"{arm.get('count_a')} vs {arm.get('count_b')} case(s))",
        f"mean composite: {format_number(arm.get('composite_a'))} vs {format_number(arm.get('composite_b'))} "
        f"(delta {format_signed(arm.get('composite_delta'))}), "
        + format_significance(
            significant=arm.get("significant"),
            paired=paired,
            p=arm.get("p"),
            effect=arm.get("hedges_g"),
            n=arm.get("n_pairs"),
            hedges=True,
        ),
    ]
    for run_id, key in (
        (compared.baseline_run_id, "pass_hat_k_unmeasured_reason_a"),
        (compared.candidate_run_id, "pass_hat_k_unmeasured_reason_b"),
    ):
        if arm.get(key):
            lines.append(f"pass^k of run {run_id} is unmeasured: {arm[key]}")
    if view.get("composite_comparability"):
        lines.append(str(view["composite_comparability"]))
    if arm.get("composite_bases_differ"):
        lines.append(
            "composite bases differ: "
            + "; ".join(
                f"run {run_id} meaned over {_basis_sets(arm.get(key))}"
                for run_id, key in (
                    (compared.baseline_run_id, "composite_basis_a"),
                    (compared.candidate_run_id, "composite_basis_b"),
                )
            )
            + " — the delta is partly a difference in what was averaged, not only in what was measured"
        )
    lines += _completeness(compared.completeness_disclosures)
    lines += [
        sentence for sentence in (compared.measurement_window_disclosure, compared.cassette_mode_disclosure) if sentence
    ]
    return "\n".join(lines)


__all__ = [
    "LAUNCH_PRICER_METHOD",
    "ArmEstimate",
    "LaunchEstimate",
    "OutOfRunSpendReport",
    "OutOfRunSpendTotals",
    "RunsCompared",
    "estimate_text",
    "export_text",
    "history_text",
    "launch_estimate",
    "out_of_run_spend_text",
    "pivot_text",
    "runs_compare",
    "runs_compared_text",
    "scope_export",
    "scope_history",
    "scope_out_of_run_spend",
    "scope_pivot",
]

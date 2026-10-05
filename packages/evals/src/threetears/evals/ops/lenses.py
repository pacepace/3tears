"""The read lenses as operations: a scope's pivot, its history, its export, and a launch's estimated cost.

Each binds one lens of :mod:`threetears.evals.analysis` to the host — its store, its run listing and its
vocabulary — and returns the lens's own typed result, so a CLI, an MCP action and a REST route read one
shape. The computation is the lens's; nothing here re-derives an answer.

A launch estimate prices what :func:`~threetears.evals.ops.run_launch` would run from the same
arguments: the template's cases (counted the host's way, :data:`~threetears.evals.ops.TemplateCaseCounter`),
``k_runs`` repeats, one arm per model.

Each result's text is here too (:func:`pivot_text`, :func:`history_text`, :func:`estimate_text`,
:func:`export_text`), for the reason :meth:`~threetears.evals.ops.EvalSummary.render` sits with the
summary: every surface — an action, a command line — shows one rendering, and each carries the
caveats its model carries (what was left out, which runs came up short) rather than the numbers alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from threetears.evals.analysis.reads import RunLister, estimate_cost, export_results, history, pivot
from threetears.evals.analysis.numbers import format_number, format_signed
from threetears.evals.analysis.reporting import compute_estimate_cost
from threetears.evals.analysis.reporting import (
    CostEstimate,
    HistoryResult,
    PivotTable,
    PredictedValue,
    ProjectionExclusions,
    ScoreExport,
)
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.contracts.host import EvalHost
from threetears.evals.contracts.models import DEFAULT_LAUNCH_K_RUNS, EvalRun
from threetears.evals.contracts.out_of_run import OutOfRunPurpose, OutOfRunSpend
from threetears.evals.ops.host import OpsHost
from threetears.evals.run.authoring import get_template
from threetears.evals.run.launch import ArmPrice, ArmQuote, LaunchPricer
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
    predicted_cost: CostEstimate | Mapping[str, Any] | None = None,
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
            each planned cell's observed cost.

    Returns:
        The table.

    Raises:
        ValidationFailedError: The pivot cannot be answered honestly — see
            :func:`~threetears.evals.analysis.pivot`.
    """
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
        predicted_cost=predicted_cost,
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


def launch_estimate(
    host: OpsHost,
    scope_id: str,
    *,
    template_id: str,
    models: list[str],
    k_runs: int = DEFAULT_LAUNCH_K_RUNS,
    n_variations: int = 0,
    n_test_cases: int | None = None,
    subject_id: str | None = None,
) -> CostEstimate:
    """What launching a template's runs would cost, from the scope's history of what such runs spent.

    Args:
        host: The host: its store, its vocabulary and how it counts a template's cases.
        scope_id: The scope the launch would run in, whose history prices it.
        template_id: The template the launch would run; its cases and its kind price the grid.
        models: The models the launch would run, one arm each.
        k_runs: Repeats of every case, as the launch would take them.
        n_variations: Cases the launch would generate before its arms run, as the launch takes them; a
            launch that generates runs those cases rather than the stored ones, so they are priced as that
            many (an upper bound, since generation de-duplicates). The generation calls themselves — the
            ``variation`` role writing an ``llm`` axis — are not in this estimate: they run before any run,
            outside its cost cap, and the launch prices them on the writer's own client against the host's
            out-of-run cap before it makes them, ledgering each (``EvalStorage.query_out_of_run_spend``).
        n_test_cases: A case count to price in place of the template's own, for a hypothetical grid.
        subject_id: Draw the history from this subject's runs alone.

    Returns:
        The estimate.

    Raises:
        ValidationFailedError: The host does not count template cases here, or the proposal cannot be
            priced (no models, a grid below one).
        NotFoundError: No template with that id in the scope.
    """
    counter = host.count_template_cases
    if counter is None:
        raise ValidationFailedError(
            "this host does not estimate launches here: it was mounted without a template case counter "
            "(OpsHost.count_template_cases), so a template's case count — which prices the launch — has no answer"
        )
    eval_host = host.eval_host
    return estimate_cost(
        eval_host.storage,
        scope_id,
        list_runs=_lister(eval_host),
        load_template=lambda template: get_template(eval_host, template, scope_id),
        count_template_cases=counter,
        models=models,
        k_runs=k_runs,
        n_test_cases=n_test_cases,
        n_new_cases=n_variations,
        subject_id=subject_id,
        template_id=template_id,
        profile=eval_host.profile,
    )


def history_launch_pricer(host: EvalHost) -> LaunchPricer:
    """The engine's launch pricer: an arm bounded from the scope's usage history of runs launched as it will be.

    What a :class:`~threetears.evals.run.LaunchHost` takes as ``launch_pricer`` to price the arms of a
    generating launch before the generation is paid for. **The history is the runs launched the way the
    arm will be**: the same template on the same candidate model at the same cassette mode, under the same
    judge and simulator pins (the model a launch named for the role, or none — a run that inherited the
    role's default matches an arm naming none) and the same resolved apparatus settings, archived runs
    included. A run judged by another model, or with its rig set up otherwise, spent differently, and
    pricing an arm from it would bias the prediction by whatever the difference costs — low, when the arm's
    judge is the dearer one, and the generation paid before the run's own cap could say so. Only those runs'
    results are read, one run at a time, rather than every result in the scope for every arm.

    Each arm is priced as :func:`launch_estimate` prices a model's cell — those results' per-observation
    costs scaled to the arm's planned cases and repeats — and **its prediction is the cell's upper band**,
    since the launch holds that figure to the arm's cap (:class:`~threetears.evals.run.ArmPrice`): a central
    estimate admits an arm that then runs past its cap about as often as the run lands above the centre.
    A history too thin to band (fewer than three priced observations) bounds nothing, and predicts nothing,
    as does a scope with no priced history of the condition — the launch reads both as unknown.

    Args:
        host: The host whose store holds the history and whose vocabulary reads it.

    Returns:
        The pricer.
    """

    def price(quote: ArmQuote) -> ArmPrice:
        condition = (
            f"template {quote.template_id!r} on {quote.candidate_model!r} at cassette mode {quote.cassette_mode!r}, "
            f"judge pin {quote.judge_model!r}, simulator pin {quote.simulator_model!r} and apparatus settings "
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
            n_test_cases_source="generated",
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
                predicted_usd=None, basis=f"no priced result of {condition} is in scope {quote.scope_id!r}{unpriced}"
            )
        if predicted.interval_high is None:
            return ArmPrice(
                predicted_usd=None,
                basis=(
                    f"{cell.n_historical} priced past result(s) of {condition} are too few to bound the arm (a "
                    f"band needs three; their mean alone puts it at ${predicted.value:.2f})"
                ),
            )
        return ArmPrice(
            predicted_usd=predicted.interval_high,
            basis=(
                f"the upper end of the band ${predicted.interval_low or 0.0:.2f}-${predicted.interval_high:.2f} "
                f"around ${predicted.value:.2f}, method {predicted.method_id}, from {cell.n_historical} past "
                f"result(s) of {condition}"
            ),
        )

    return price


def _launched_as(run: EvalRun, quote: ArmQuote) -> bool:
    """Whether ``run`` was launched as ``quote``'s arm will be: its template, model, cassette mode, pins and rig.

    Args:
        run: A run from the scope's history.
        quote: The arm being priced.

    Returns:
        True when every launch argument that moves what a run spends matches.
    """
    return (
        run.template_id == quote.template_id
        and run.candidate_model == quote.candidate_model
        and run.cassette_mode == quote.cassette_mode
        and _launch_pin(run, "judge", run.judge_model) == quote.judge_model
        and _launch_pin(run, "simulator", run.simulator_model) == quote.simulator_model
        and run.apparatus_settings == dict(quote.apparatus_settings)
    )


def _launch_pin(run: EvalRun, role: str, resolved: str | None) -> str | None:
    """The model ``run``'s launch named for ``role`` — the resolved model when the launch chose it, else ``None``.

    Args:
        run: The run.
        role: ``judge`` or ``simulator``.
        resolved: The model the run recorded for the role.

    Returns:
        The pin the launch named, or ``None`` when it named none and the run inherited the role's default
        (or the role never ran).
    """
    return resolved if (run.model_role_provenance or {}).get(role) == "chosen" else None


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
    """The calls the engine made outside any run in a scope — case generations and rubric proposals — and their totals.

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
        by_launch: The totals per launch group that appears (case generations; a proposal belongs to none, and
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
        purpose: Only calls made for this purpose (``variation`` or ``proposer``).
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


def _predicted(predicted: PredictedValue | None) -> str:
    """A prediction, set apart from the observation it sits beside."""
    if predicted is None:
        return ""
    band = (
        f" [{format_number(predicted.interval_low)}, {format_number(predicted.interval_high)}]"
        if predicted.interval_low is not None and predicted.interval_high is not None
        else ""
    )
    return f"; predicted {format_number(predicted.value)}{band} ({predicted.method_id})"


def pivot_text(table: PivotTable) -> str:
    """A pivot as text: what was computed, each cell with its denominators, and every caveat the table carries."""
    lines = [
        f"pivot of {table.metric} by {table.row_factor} (rows) x {table.column_factor} (columns), "
        f"{table.weighting}: {table.n_observations} observation(s), {table.n_filtered_out} filtered out",
        f"formula: {table.formula}",
    ]
    for cell in table.cells:
        spread = f", sem {format_number(cell.sem)}" if cell.sem is not None else ""
        unmeasured = f", {cell.n_unmeasured} unmeasured" if cell.n_unmeasured else ""
        lines.append(
            f"- {cell.row} / {cell.column}: {format_number(cell.value)} ({cell.status}; n={cell.n}, "
            f"{cell.n_cases} case(s){spread}{unmeasured}){_predicted(cell.predicted)}"
        )
    if not table.cells:
        lines.append("- no cells")
    for flag in table.simpsons_flags:
        lines.append(
            f"Simpson's reversal: pooled, {flag.pooled_leader} leads {flag.column_a} vs {flag.column_b}, but "
            f"{flag.rows_disagreeing} row(s) rank them the other way ({', '.join(flag.disagreeing_rows)}) against "
            f"{flag.rows_agreeing} that agree — do not read the pooled order as a ranking"
        )
    if table.unplaced_predicted_models:
        lines.append(f"planned and not observed: {', '.join(table.unplaced_predicted_models)}")
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
    ]
    if result.attribution_disclosure:
        lines.append(result.attribution_disclosure)
    if result.identity_span_disclosure:
        lines.append(result.identity_span_disclosure)
    for series in result.series:
        lines.append(f"## {series.model} — subject {series.subject_label or series.subject_id}")
        if series.identity_version_disclosure:
            lines.append(series.identity_version_disclosure)
        for point in series.points:
            flag = point.regression
            verdict = f"; {flag.label} vs previous ({flag.test})" if flag is not None else ""
            baseline = (
                " (baseline)" if point.is_baseline else f", {format_signed(point.delta_from_baseline)} from baseline"
            )
            epoch = ", suite changed here" if point.epoch_boundary else ""
            short = f"; {point.completeness_disclosure}" if point.completeness_disclosure else ""
            lines.append(
                f"- {point.created_at} {point.run_id}: {format_number(point.value)} (n={point.n}, "
                f"{point.n_cases} case(s)){baseline}{epoch}{verdict}{short}"
            )
    if not result.series:
        lines.append("- no series")
    lines += _exclusions(result.exclusions)
    return "\n".join(lines)


def estimate_text(estimate: CostEstimate) -> str:
    """An estimate as text: the grid priced, each model's cell, and the total with its band."""
    template = f", template {estimate.template_id}" if estimate.template_id else ""
    subject = f", subject {estimate.subject_id}" if estimate.subject_id else ", every subject"
    lines = [
        f"estimate: {estimate.n_test_cases} case(s) ({estimate.n_test_cases_source}) x k_runs {estimate.k_runs} "
        f"x {estimate.n_settings} setting(s) per model, cassette mode {estimate.cassette_mode}{template}{subject}",
    ]
    for cell in estimate.cells:
        if cell.basis == "no_history":
            lines.append(
                f"- {cell.model}: no priced history under this cassette mode and these filters — not in the total"
            )
            continue
        unpriced = f", {cell.n_unpriced_historical} unpriced left out" if cell.n_unpriced_historical else ""
        lines.append(
            f"- {cell.model}: {cell.n_observations} observation(s) at ${format_number(cell.mean_cost_per_observation)} "
            f"each, from {cell.n_historical} priced in history{unpriced}{_predicted(cell.predicted)}"
        )
    band = (
        f" [{format_number(estimate.total_interval_low)}, {format_number(estimate.total_interval_high)}]"
        if estimate.total_interval_low is not None and estimate.total_interval_high is not None
        else " (no band: a cell's history is too thin to bracket)"
    )
    if estimate.total_estimated_cost is None:
        lines.append("total: none — no proposed model has priced history to estimate from")
    else:
        lines.append(f"total: ${format_number(estimate.total_estimated_cost)}{band}")
    if estimate.n_uncovered_models:
        lines.append(f"{estimate.n_uncovered_models} model(s) have no history and are not in the total")
    return "\n".join(lines)


def export_text(export: ScoreExport) -> str:
    """An export as text: a line of its row count and what it left out, then the body itself."""
    lines = [f"export ({export.format}, {export.n_records} row(s))"]
    lines += _exclusions(export.exclusions)
    lines += _completeness(export.completeness_disclosures)
    return "\n".join(lines) + "\n\n" + export.body


__all__ = [
    "OutOfRunSpendReport",
    "OutOfRunSpendTotals",
    "estimate_text",
    "export_text",
    "history_text",
    "launch_estimate",
    "out_of_run_spend_text",
    "pivot_text",
    "scope_export",
    "scope_history",
    "scope_out_of_run_spend",
    "scope_pivot",
]

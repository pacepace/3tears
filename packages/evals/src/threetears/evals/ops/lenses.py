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

from collections.abc import Mapping
from typing import Any

from threetears.evals.analysis.reads import RunLister, estimate_cost, export_results, history, pivot
from threetears.evals.analysis.numbers import format_number, format_signed
from threetears.evals.analysis.reporting import (
    CostEstimate,
    HistoryResult,
    PivotTable,
    PredictedValue,
    ProjectionExclusions,
    ScoreExport,
)
from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.contracts.host import EvalHost
from threetears.evals.contracts.models import EvalRun
from threetears.evals.ops.host import OpsHost
from threetears.evals.run.authoring import get_template
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
    k_runs: int = 1,
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
            ``variation`` role writing an ``llm`` axis — are not priced: they run before any run, outside
            its cost cap, and leave no per-cell history to price them from.
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
    "estimate_text",
    "export_text",
    "history_text",
    "launch_estimate",
    "pivot_text",
    "scope_export",
    "scope_history",
    "scope_pivot",
]

"""The descriptive telemetry rollup: per-run summaries, token sums and model versions.

Descriptive only — no scoring statistic lives here. :func:`_run_summary` digests one run and its results,
:func:`_token_rollup` sums their tokens, and :func:`_telemetry_rollup` assembles both into the bundle's
:class:`~threetears.evals.analysis.bundle.schema.TelemetryRollup`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from threetears.evals.analysis.reporting import ScoreRecord
from threetears.evals.analysis.lenses.program_budget import ProgramBudget
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.schema.models import EvalResult
from threetears.evals.kernel.provider import sum_optional_tokens
from threetears.evals.kernel.result_condition import delivered_a_turn
from threetears.evals.kernel.usage_capture import (
    count_substituted_deliveries,
    production_replicating_cost,
)
from threetears.evals.analysis.bundle.schema import (
    RunSummary,
    TelemetryRollup,
    TokenRollup,
)
from threetears.evals.analysis.bundle.config import _effective_config
from threetears.evals.analysis.bundle.measures import _measure_collection

if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
    from threetears.evals.schema.models import EvalRun


def _token_rollup(results: list[EvalResult]) -> TokenRollup | None:
    """Sum token usage across results carrying a usage breakdown, or None.

    Sums what was reported (:func:`~threetears.evals.kernel.provider.sum_optional_tokens`, the
    package's one definition of that) and counts the results whose token-metered rows left a count
    unreported, rather than adding an unreported count as zero.
    """
    prompt: int | None = None
    completion: int | None = None
    reasoning: int | None = None
    n_with_usage = 0
    n_unreported = 0
    for result in results:
        if not result.usage:
            continue
        n_with_usage += 1
        for role in result.usage:
            prompt = sum_optional_tokens(prompt, role.prompt_tokens)
            completion = sum_optional_tokens(completion, role.completion_tokens)
            reasoning = sum_optional_tokens(reasoning, role.reasoning_tokens)
        if any(
            row.role != "external" and (row.prompt_tokens is None or row.completion_tokens is None)
            for row in result.usage
        ):
            n_unreported += 1
    if n_with_usage == 0:
        return None
    return TokenRollup(
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        n_results_with_usage=n_with_usage,
        n_results_tokens_unreported=n_unreported,
    )


def _result_has_error(result: EvalResult) -> bool:
    """True if a result carries any runner/judge/candidate/infra error.

    Deliberately NOT
    :func:`~threetears.evals.kernel.result_condition.resolve_result_condition`'s ``scoring`` axis,
    which every per-result read surface uses. The two answer different questions: this counts
    results that carry an error field at all, while the scoring axis decides how a result
    participates in aggregates — a candidate failure the turn budget or the output cap caused
    carries no error field and is still a hard fail there. Reads the categorized fields only,
    never the combined ``runner_error`` display string (the fragile-parse rule): the runner writes
    all of them from one ledger, so nothing is set there that is not set here.
    """
    return bool(result.judge_error or result.candidate_error or result.infra_error)


def _measured_prod_costs(run_results: list[EvalResult]) -> list[float]:
    """Every production-replicating cost a result actually measured — no placeholders.

    A result that evidences nothing to decompose, that observed no production-role spend,
    or that carries a substituted delivery yields ``None`` from
    :func:`production_replicating_cost`, and is dropped here rather than contributing a
    zero. That is the whole point: a zero nobody observed, averaged in, ranks the
    least-measured configuration the cheapest — failing toward "cheaper than reality",
    which is the dangerous direction for a config or capacity decision.

    ``substituted_deliveries`` is passed per result rather than assumed, because a
    substituted delivery leaves no usage row behind: the rows alone cannot evidence that
    they are incomplete.

    Read over the turns the candidate took
    (:func:`~threetears.evals.kernel.result_condition.delivered_a_turn`), as every cost reading is: a
    billed refusal's dollars are no turn's spend, and averaged in they made a refusing configuration cheap.

    Args:
        run_results: The run's results.

    Returns:
        One float per turn that measured a production-replicating cost, in input order.
    """
    measured: list[float] = []
    for result in run_results:
        if not delivered_a_turn(result):
            continue
        cost = production_replicating_cost(result.usage, substituted_deliveries=count_substituted_deliveries(result))
        if cost is not None:
            measured.append(cost)
    return measured


def _run_summary(
    run: EvalRun, run_results: list[EvalResult], reportable: set[str], *, profile: HostProfile
) -> RunSummary:
    """Digest one run + its results into a :class:`RunSummary`.

    Args:
        run: The run.
        run_results: That run's results.
        reportable: The levers the campaign's coverage map names — see
            :func:`_reportable_levers`. ``config`` is narrowed to these so the two surfaces
            cannot name different levers for one campaign, and so a contestant property nobody
            swept does not arrive as a config value the generator reads as a departure.
        profile: The host whose vocabulary this reads.

    Returns:
        The summary.
    """
    effective = {
        lever: eff for lever, eff in _effective_config(run, run_results, profile=profile).items() if lever in reportable
    }
    prod_costs = _measured_prod_costs(run_results)
    return RunSummary(
        run_id=run.id,
        status=str(run.status),
        created_at=run.created_at,
        candidate_model=run.candidate_model,
        k_runs=run.k_runs,
        config={lever: eff.value for lever, eff in effective.items() if eff.value is not None},
        config_provenance={lever: eff.provenance for lever, eff in effective.items()},
        n_results=len(run_results),
        n_errors=sum(1 for r in run_results if _result_has_error(r)),
        cost_usd=sum(r.cost_usd for r in run_results if r.cost_usd is not None),
        n_cost_unpriced=sum(1 for r in run_results if r.cost_usd is None),
        prod_cost_usd=sum(prod_costs) if prod_costs else None,
        mean_prod_cost_usd=(sum(prod_costs) / len(prod_costs)) if prod_costs else None,
        n_prod_cost_usd=len(prod_costs),
        # A run read without its host payload cannot be checked: an elided lever would read as the
        # subject's own setting. None says nobody checked rather than claiming nothing moved.
        production_footing=(
            None if run.elided_payload_paths else profile.sweepables.production_footing(run, run_results)
        ),
        measures=_measure_collection(run_results, profile=profile, undeclared="all_observed"),
    )


def _model_versions(runs: list[EvalRun], records: list[ScoreRecord]) -> dict[str, str]:
    """Collect distinct models by role — candidate/judge/simulator — as a flat map.

    A single value per role reads cleanly; a swept role (e.g. a judge bake-off)
    shows every value comma-joined rather than silently collapsing to one.
    """
    candidate = sorted({run.candidate_model for run in runs})
    judge = sorted(
        {run.judge_model for run in runs if run.judge_model} | {r.judge_model for r in records if r.judge_model}
    )
    simulator = sorted({r.simulator_model for r in records if r.simulator_model})
    versions: dict[str, str] = {}
    if candidate:
        versions["candidate"] = ", ".join(candidate)
    if judge:
        versions["judge"] = ", ".join(judge)
    if simulator:
        versions["simulator"] = ", ".join(simulator)
    return versions


def _telemetry_rollup(
    runs: list[EvalRun], results: list[EvalResult], budget: ProgramBudget, *, profile: HostProfile
) -> TelemetryRollup:
    """Compose the campaign-wide telemetry rollup from results + the budget lens."""
    return TelemetryRollup(
        n_runs=len(runs),
        n_results=len(results),
        n_errors=sum(1 for r in results if _result_has_error(r)),
        total_cost_usd=budget.total_cost_usd,
        incomplete_cost_usd=budget.incomplete_cost_usd,
        unattributed_cost_usd=budget.unattributed_cost_usd,
        measures=_measure_collection(results, profile=profile, undeclared="all_observed"),
        tokens=_token_rollup(results),
    )


__all__: list[str] = []

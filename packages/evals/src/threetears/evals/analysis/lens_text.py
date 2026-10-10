"""A frontier as an operator reads it, and the exclusion lines every lens's text shares.

Here rather than with the other lens renderings in :mod:`threetears.evals.ops` because the command
line prints a frontier too, and :mod:`threetears.evals.quick` sits beside ``ops`` rather than above it.
One rendering, so both surfaces carry the same caveats: what was left out, and why.
"""

from __future__ import annotations


from threetears.evals.analysis.arms import short_digest


from threetears.evals.analysis.numbers import format_number


from threetears.evals.analysis.reporting import ProjectionExclusions
from threetears.evals.analysis.lenses.frontier import FrontierResult


def exclusion_lines(exclusions: ProjectionExclusions) -> list[str]:
    """What never became a row, when anything did not — an all-excluded answer must not read as empty."""
    if not exclusions.total:
        return []
    return [
        f"excluded: {exclusions.total} observation(s) — {exclusions.results_outside_queried_runs} from runs the "
        f"filters left out, {exclusions.results_from_archived_runs} from archived runs, "
        f"{exclusions.results_without_run} whose run is missing"
    ]


def _contestant(model: str, variant_key: str) -> str:
    """A frontier contestant as every surface names one: its model and its variant, the key shortened."""
    return f"{model} · {short_digest(variant_key)}"


def frontier_text(result: FrontierResult) -> str:
    """A frontier as text: each subject's variants, best pass^k first, with each one's axes and the verdict."""
    bar = "no bar, so no verdict" if result.bar is None else f"bar {format_number(result.bar)} on pass^k"
    lines = [
        f"frontier, {bar}: {len(result.subjects)} subject(s) over {result.n_results} result(s), "
        f"{result.n_filtered_out} filtered out",
    ]
    lines += [text for text in (result.identity_span_disclosure, result.contended_latency_disclosure) if text]
    for subject in result.subjects:
        lines.append(f"## subject {subject.subject_label or subject.subject_id}, pass^k at k={subject.k}")
        if subject.boundary_pillar:
            lines.append(subject.boundary_pillar)
        for point in subject.points:
            quality = (
                f"pass^{point.k} {format_number(point.pass_hat_k)}"
                + (
                    f" [{format_number(point.pass_hat_k_ci_low)}, {format_number(point.pass_hat_k_ci_high)}]"
                    if point.pass_hat_k_ci_low is not None and point.pass_hat_k_ci_high is not None
                    else ""
                )
                + f" over {point.n_pass_cases} case(s)"
                if point.pass_hat_k is not None
                else f"no pass^k ({point.pass_hat_k_unmeasured_reason or 'no case scored that deep'})"
            )
            cleared = f", bar {point.bar_decision}" if point.bar_decision is not None else ""
            cost = (
                f"${format_number(point.production_replicating_cost)} per result (n={point.n_cost}"
                + (", partial" if point.cost_is_partial else "")
                + ")"
                if point.production_replicating_cost is not None
                else "cost unobserved"
            )
            latency = (
                f"{format_number(point.mean_total_ms)} ms (n={point.n_latency})"
                if point.mean_total_ms is not None
                else "latency unobserved"
            )
            standing = (
                f"disqualified: breached {', '.join(point.disqualified_by)}"
                if point.disqualified_by
                else f"dominated by {', '.join(_contestant(rival.model, rival.variant_key) for rival in point.dominated_by)}"
                if point.dominated
                else f"domination {point.dominance}"
                if point.dominance is not None
                else ""
            )
            lines.append(
                f"- {_contestant(point.model, point.variant_key)}: {quality}{cleared}; {cost}; {latency}"
                + (f"; {standing}" if standing else "")
            )
        verdict = subject.verdict
        if verdict is not None:
            tied = (
                f", tied with {', '.join(_contestant(tie.model, tie.variant_key) for tie in verdict.tied_with)}"
                if verdict.tied_with
                else ""
            )
            lines.append(
                f"verdict: {_contestant(verdict.model, verdict.variant_key)} is the cheapest clearing the "
                f"bar ({verdict.cost_decision}){tied}"
            )
        elif result.bar is not None:
            lines.append(
                f"verdict: none — {subject.n_cleared_bar} variant(s) cleared the bar with a cost observed, "
                f"{subject.n_undecided_bar} undecided"
            )
    if not result.subjects:
        lines.append("- no subjects")
    lines += exclusion_lines(result.exclusions)
    return "\n".join(lines)


__all__ = [
    "exclusion_lines",
    "frontier_text",
]

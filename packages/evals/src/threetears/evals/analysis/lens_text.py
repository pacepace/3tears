"""A frontier as an operator reads it, the exclusion lines every lens's text shares, and a judge's tier as a host
prints it off any page.

Here rather than with the other lens renderings in :mod:`threetears.evals.ops` because the command
line prints a frontier too, and :mod:`threetears.evals.quick` sits beside ``ops`` rather than above it.
One rendering, so both surfaces carry the same caveats: what was left out, and why. The tier sentence sits here
rather than with :mod:`~threetears.evals.analysis.agreement`, which decides a tier and never imports a page's forms:
its wording prints every figure in the forms of the page it is on (:func:`tier_sentence_in`), or in the off-page
forms (:func:`tier_sentence`), and this module may import them.
"""

from __future__ import annotations


from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.depth import times
from threetears.evals.analysis.depth_reading import SubjectDepth, subject_depth


from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.number_forms.printed import Printed
from threetears.evals.analysis.report.unit_once import interval_beside
from threetears.evals.analysis.report.verdict import MIN_CASES


from threetears.evals.analysis.reporting import ProjectionExclusions
from threetears.evals.kernel.evidence_tiers import JudgeEvidenceTier, TierCriterion
from threetears.evals.analysis.lenses.frontier import (
    PASS_HAT_K_CELL_READING,
    FrontierPoint,
    FrontierResult,
    SubjectFrontier,
    pass_hat_k_cell_lines,
)
from threetears.evals.schema.models import MODEL_DEFAULT_TEMPERATURE, is_builtin_judge_config_id, sent_settings_phrase


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


def _pass_hat_k_text(
    k: int, value: float, n_cases: int, low: float | None, high: float | None, forms: PageForms
) -> str:
    """A point's pass^k: a rate over :data:`MIN_CASES` or more cases, else only the case count it rests on.

    Below the floor no figure is printed, count or interval: pass^k averages fractional per-case estimates, so no count
    of cases passed was observed.
    """
    if n_cases < MIN_CASES:
        return (
            f"pass^{k} over {forms.count(n_cases).text} case(s), too few to state as a rate (a rate needs {MIN_CASES})"
        )
    interval = (
        f" ({interval_beside(_pass_k(low, forms).digits, _pass_k(high, forms).digits)})"
        if low is not None and high is not None
        else ""
    )
    return f"pass^{k} {_pass_k(value, forms).text}{interval} over {forms.count(n_cases).text} case(s)"


def _pass_k(value: float | None, forms: PageForms) -> Printed:
    """A pass^k of the trials' pass, as every surface prints one (:meth:`PageForms.pass_k`): a point's, its interval's
    ends, its model's, its reading at the common depth, a cell's own and the bar it is read against alike."""
    return forms.pass_k(TRIAL_PASS_NAME, value, kind="trial")


#: The frontier text's depth lines, each one template: a subject's place, a contestant's delivered depth, the cases it
#: is short of ``k``, its model's reading, and its reading at the common depth.
WHERE_SUBJECT = "subject {subject}"
POINT_DEPTH = "ran each case {times} over {cases} case(s)"
POINT_SHORT = ", {short} short of k={k}"
POINT_MODEL = "; pass^{k} {value} {assumes}"
POINT_COMMON = "; pass^{k} {value} over {cases} case(s) at the common depth"


def _depth_lines(subject: SubjectFrontier) -> list[str]:
    """What depth a subject's pass^k compares at, or that its contestants share no case, and each contestant it is
    unmeasured for at its planned ``k``: :class:`~threetears.evals.analysis.depth_reading.SubjectDepth`'s sentences,
    each contestant named as this text names it."""
    named = {point.variant_key: _contestant(point.model, point.variant_key) for point in subject.points}
    where = WHERE_SUBJECT.format(subject=subject.subject_label or subject.subject_id)
    depth: SubjectDepth = subject_depth(subject, lambda key: named.get(key, short_digest(key)), where=where)
    return depth.lines()


def _point_depth(point: FrontierPoint, subject: SubjectFrontier, forms: PageForms) -> str | None:
    """One contestant's delivered depth, and its pass^k at the subject's common depth where it differs from ``k``, its
    figures as ``forms`` states them."""
    if point.depth is None:
        return None
    said = POINT_DEPTH.format(times=times(point.depth), cases=forms.count(point.depth.cases).text)
    if point.n_cases_short_of_k:
        said += POINT_SHORT.format(short=forms.count(point.n_cases_short_of_k).text, k=subject.k)
    if (model := point.model_pass_hat_k) is not None:
        said += POINT_MODEL.format(k=model.k, value=_pass_k(model.pass_hat_k, forms).text, assumes=model.assumes)
    if subject.common_depth is not None and subject.common_depth != subject.k and point.common_pass_hat_k is not None:
        said += POINT_COMMON.format(
            k=subject.common_depth,
            value=_pass_k(point.common_pass_hat_k, forms).text,
            cases=forms.count(point.n_common_cases).text,
        )
    return said


def frontier_text(result: FrontierResult) -> str:
    """A frontier as text: each subject's variants, best pass^k first, with each one's axes and the verdict."""
    forms = PageForms.off_page()
    bar = "no bar, so no verdict" if result.bar is None else f"bar {_pass_k(result.bar, forms).text} on pass^k"
    lines = [
        f"frontier, {bar}: {len(result.subjects)} subject(s) over {result.n_results} result(s), "
        f"{result.n_filtered_out} filtered out",
    ]
    lines += [
        text
        for text in (
            result.narrowed_read_disclosure,
            result.identity_span_disclosure,
            result.contended_latency_disclosure,
        )
        if text
    ]
    # What "kept out" covers, said once before the first cell line of any arm.
    said_reading = False
    for subject in result.subjects:
        lines.append(f"## subject {subject.subject_label or subject.subject_id}, pass^k at k={subject.k}")
        if subject.boundary_pillar:
            lines.append(subject.boundary_pillar)
        lines += _depth_lines(subject)
        for point in subject.points:
            quality = (
                _pass_hat_k_text(
                    point.k,
                    point.pass_hat_k,
                    point.n_pass_cases,
                    point.pass_hat_k_ci_low,
                    point.pass_hat_k_ci_high,
                    forms,
                )
                if point.pass_hat_k is not None
                else f"no pass^k ({point.pass_hat_k_unmeasured_reason or 'no case scored that deep'})"
            )
            cleared = f", bar {point.bar_decision}" if point.bar_decision is not None else ""
            cost = (
                f"{forms.lone(point.production_replicating_cost, money=True).text} per result (n={point.n_cost}"
                + (", partial" if point.cost_is_partial else "")
                + ")"
                if point.production_replicating_cost is not None
                else "cost unobserved"
            )
            latency = (
                f"{forms.lone(point.mean_total_ms).text} ms (n={point.n_latency})"
                if point.mean_total_ms is not None
                else "latency unobserved"
            )
            standing = (
                f"{point.earlier_control}, not a contestant"
                if point.earlier_control is not None
                else f"disqualified: breached {', '.join(point.disqualified_by)}"
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
            if (depth := _point_depth(point, subject, forms)) is not None:
                lines.append(f"  {depth}")
            cell_lines = [forms.said(line) for cell in point.pass_hat_k_cells for line in pass_hat_k_cell_lines(cell)]
            if cell_lines and not said_reading:
                lines.append(f"  {PASS_HAT_K_CELL_READING}")
                said_reading = True
            lines += [f"  - {line}" for line in cell_lines]  # one line: pass_hat_k_cell_lines collapses each line
        verdict = subject.verdict
        if verdict is not None:
            tied = (
                f", tied with {', '.join(_contestant(tie.model, tie.variant_key) for tie in verdict.tied_with)}"
                if verdict.tied_with
                else ""
            )
            # The pass^k the bar was read on stands on its judges' tier where a judge scored its criteria (#679).
            tier = f", judge tier {verdict.judge_evidence_tier}" if verdict.judge_evidence_tier is not None else ""
            lines.append(
                f"verdict: {_contestant(verdict.model, verdict.variant_key)} is the cheapest clearing the "
                f"bar ({verdict.cost_decision}){tied}{tier}"
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


def tier_sentence_in(tier: JudgeEvidenceTier, forms: PageForms) -> str:
    """One sentence a report states for a judge's tier on a dimension: the tier, and the two measurements behind it,
    every figure (a kappa, its bounds and bar, a temperature, a count of cases) printed by ``forms``, the forms of the
    page it is printed on.

    :param tier: The tier as decided.
    :ptype tier: JudgeEvidenceTier
    :param forms: The forms of the page the sentence is printed on, which print its every figure by the lone figure
        rule.
    :ptype forms: PageForms
    :return: The sentence, naming the judge (its config, when one asked, its temperature and the request settings its
        calls were sent with), the tier and each criterion's
        agreement, pairs and results against its bar — and, for a tier read from a stored judge profile, the profile,
        when and on which cases it was measured, and what the campaign's own evidence read.
    :rtype: str
    """
    judge = tier.judge_model or "an unnamed judge"
    if tier.judge_config_id is not None:
        what = "built-in prompt" if is_builtin_judge_config_id(tier.judge_config_id) else "config"
        judge = f"{judge}, {what} {tier.judge_config_id}"
    temperature = tier.judge_temperature
    judge += (
        ", temperature not recorded"
        if temperature is None
        else ", sent no temperature"
        if temperature == MODEL_DEFAULT_TEMPERATURE
        else f", temperature {forms.lone(temperature).text}"
    )
    judge += f", {sent_settings_phrase(tier.judge_sent_settings)}"
    measured = (
        f"{tier.rubric_dim} ({judge}): {tier.tier} — agreement with people "
        f"{_criterion_words(tier.calibration, forms)}; with its own repeats {_criterion_words(tier.separation, forms)}."
    )
    consulted = tier.consulted_profile
    if consulted is not None:
        return (
            f"{measured} This campaign's evidence decided no tier, so it consulted the judge's stored profile "
            f"{consulted.profile_id}, measured at {consulted.measured_at} on {forms.lone(consulted.cases).text} frozen "
            f"cases (case set {consulted.case_set_fingerprint[:12]}, runs {', '.join(consulted.run_ids)}), which is "
            f"undetermined too: agreement with the labels {_criterion_words(consulted.calibration, forms)}; with its own "
            f"repeats {_criterion_words(consulted.separation, forms)}."
        )
    profile = tier.from_profile
    if profile is None:
        return measured
    # Never silently: a tier read from a stored profile says so, with when and on what it was measured, and what
    # the campaign's own evidence read (#628).
    return (
        f"{tier.rubric_dim} ({judge}): {tier.tier}, read from the judge's stored profile {profile.profile_id}, "
        f"measured at {profile.measured_at} on {forms.lone(profile.cases).text} frozen cases (case set "
        f"{profile.case_set_fingerprint[:12]}, runs {', '.join(profile.run_ids)}) — agreement with the labels "
        f"{_criterion_words(tier.calibration, forms)}; with its own repeats {_criterion_words(tier.separation, forms)}. This "
        f"campaign's own evidence decided no tier: agreement with people {_criterion_words(profile.own_calibration, forms)}; "
        f"with its own repeats {_criterion_words(profile.own_separation, forms)}."
    )


def _criterion_words(criterion: TierCriterion, forms: PageForms) -> str:
    """A criterion as a clause: its agreement, bounds, pairs and results against the bar, and what it still needs,
    each figure by the lone figure rule, in ``forms``."""
    bar = f"(bar {forms.lone(criterion.threshold).text} over at least {criterion.min_results} results)"
    needed = criterion.results_needed
    if criterion.n == 0:
        return f"not measured {bar} — needs {criterion.min_results} results"
    if criterion.agreement is None:
        # Undefined kappa below the floor is still short of the floor: the results it needs are stated as they are
        # for a defined one, so the sentence never reads as though only the kappa were missing.
        undefined = f"undefined over {criterion.n} pairs from {criterion.results} results {bar}"
        return f"{undefined} — needs {needed} more results" if needed is not None else undefined
    over = f"over {criterion.n} pairs from {criterion.results} results"
    if criterion.interval is not None:
        low, high = criterion.interval
        over = f"(bounds {forms.lone(low).text} to {forms.lone(high).text}) {over}"
    verdict = {
        "met": "meets",
        "not_met": "misses",
        "undecided": "undecided — the bounds straddle",
        "insufficient": "too few results for" if criterion.results < criterion.min_results else "no bounds for",
    }[criterion.state]
    clause = f"{forms.lone(criterion.agreement).text} {over}, {verdict} {bar}"
    if needed is not None and criterion.state == "insufficient":
        clause += f" — needs {needed} more results"
    elif needed is not None:
        clause += f" — about {needed} more results would decide it if agreement holds"
    return clause


def tier_sentence(tier: JudgeEvidenceTier) -> str:
    """One sentence a report states for a judge's tier on a dimension: the tier, and the two measurements behind it.

    Off any page, as a host prints it (a console, a lens): :func:`tier_sentence_in` in an off-page surface's own
    forms. A page states it in its own forms, through ``tier_sentence_in``.

    :param tier: The tier as decided.
    :ptype tier: JudgeEvidenceTier
    :return: The sentence (``tier_sentence_in``).
    :rtype: str
    """
    return tier_sentence_in(tier, PageForms.off_page())


__all__ = [
    "exclusion_lines",
    "frontier_text",
    "tier_sentence",
    "tier_sentence_in",
]

"""Bar adjudication: every applicable bar decided against every cell, and the order the verdicts are read in.

:func:`_bar_adjudications` reads each bar the way the declaration gate admitted it, :func:`_verdict_order` orders
the adjudicated bars by the declared merit priority, and :func:`_frontier_bar` decides which bar, if any, the
frontier lens is handed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal


from threetears.evals.analysis.agreement import (
    judge_key,
    tier_for_judges,
)
from threetears.evals.kernel.evidence_tiers import (
    JudgedEvidenceTier,
    JudgeEvidenceTier,
)
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.reporting import ScoreRecord
from threetears.evals.analysis.stats import (
    clustered_standard_error,
    interval_clears,
    observed_mean_interval,
)
from threetears.evals.kernel.analysis_measures import BarAdjudication, BarVerdict
from threetears.evals.kernel.declaration import (
    BarName,
    CampaignDesign,
    UnreadableBarName,
    resolve_bar_name,
)
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.metrics import (
    FRONTIER_RANKING_MEASURE,
    MeritAxis,
    goal_check_measure,
    summary_population,
)

# At runtime for its field set, which tells a result-level measure from a row-level one.
from threetears.evals.schema.models import EvalResult
from threetears.evals.kernel.result_condition import delivered_a_turn


from threetears.evals.analysis.bundle.schema import (
    MeritTier,
    QuestionScope,
    VerdictOrder,
)


from threetears.evals.analysis.bundle.measures import _measure_collection


from threetears.evals.analysis.bundle.cell_reads import (
    _cannot_tell_on,
    _CellKey,
    _judged_rows,
    _judged_values,
    _member_run_ids,
    _non_faulted,
)


#: What the frontier lens's clearing count means when no bar was passed to it — nothing. The
#: sentence the MCP frontier render states for the same condition, carried on the bundle, and followed by
#: why no bar was passed (:func:`_frontier_bar`).
_FRONTIER_BAR_WITHHELD = (
    "withheld — no bar was supplied to the frontier lens, so it names no verdict and each subject's "
    "n_cleared_bar is a default of 0, not a count of arms that failed a bar. The bars this campaign is "
    "held to are adjudicated in bar_adjudications."
)


def _frontier_bar(
    behavior: str, design: CampaignDesign | None, *, profile: HostProfile
) -> tuple[float | None, str | None]:
    """The bar the frontier is given, or why it is given none.

    The frontier ranks on pass^k (:data:`~threetears.evals.kernel.metrics.FRONTIER_RANKING_MEASURE`) and
    reads a bar only on it, while a campaign's bars may name any measure. So it takes the campaign's
    effective bar on that measure — the declared one, else the host's registered one, exactly as
    :func:`_applicable_bars` resolves every bar — and none otherwise; a bar on another measure is adjudicated
    per cell in ``bar_adjudications`` and is never translated onto pass^k. The frontier reads the bar on each
    contestant's pass^k interval by the three-valued rule every bar is read by
    (:func:`~threetears.evals.analysis.stats.interval_clears`), with no margin, which is pass^k's own: its
    descriptor declares none.

    **More than one effective bar on pass^k is refused, not picked between**, as is one the frontier cannot
    read — lower-is-better, or a threshold outside pass^k's range [0, 1] (a host's registered bar is not
    range-checked where it is registered) — and the reason says so.

    Args:
        behavior: The campaign's behavior, the scope a registered bar is keyed under.
        design: The campaign's declaration.
        profile: The host whose registered bars apply.

    Returns:
        ``(bar, None)`` when the frontier takes a bar, else ``(None, reason)``: the withheld sentence followed
        by which bars exist, and on what measure, or why the one on pass^k was not passed.
    """
    bars = _applicable_bars(behavior, design, profile=profile)
    on_ranking = [bar for bar in bars if bar[0] == FRONTIER_RANKING_MEASURE]

    def named(bar: tuple[str, float, bool, str]) -> str:
        measure_id, threshold, higher_is_better, source = bar
        return f"{measure_id} {'≥' if higher_is_better else '≤'} {format_number(threshold)} ({source})"

    if len(on_ranking) == 1:
        _, threshold, higher_is_better, _ = on_ranking[0]
        if higher_is_better and 0.0 <= threshold <= 1.0:
            return threshold, None
        why = (
            f"The one bar on {FRONTIER_RANKING_MEASURE}, {named(on_ranking[0])}, is not one the frontier can read: "
            "pass^k is a probability whose higher end is better, so its bar is a threshold in [0, 1] to reach."
        )
    elif on_ranking:
        why = (
            f"{len(on_ranking)} bars name {FRONTIER_RANKING_MEASURE}, the measure the frontier ranks on — "
            f"{'; '.join(named(bar) for bar in on_ranking)} — and it takes one, so it was given none rather than "
            "a pick between them."
        )
    elif bars:
        why = (
            f"The frontier ranks on {FRONTIER_RANKING_MEASURE} and reads a bar only on it; this campaign's bars are "
            f"on other measures — {'; '.join(named(bar) for bar in bars)} — so none was passed to it."
        )
    else:
        why = "This campaign is held to no bar, declared or registered."
    return None, f"{_FRONTIER_BAR_WITHHELD} {why}"


#: A bar's per-cell reading: ``(mean, sem, n, n_independent, interval)``.
_BarReading = tuple[float | None, float | None, int, int, tuple[float, float] | None]

#: What adjudicating a bar came to — see :attr:`BarAdjudication.state`.
_BarState = Literal["adjudicated", "names_no_stored_measure", "not_numeric"]


def _applicable_bars(
    behavior: str, design: CampaignDesign | None, *, profile: HostProfile
) -> list[tuple[str, float, bool, Literal["declared", "registered"]]]:
    """The bars this campaign is held to: its own, then each registered incumbent it did not override.

    A campaign that declares no bar on a measure is held to the host's registered one — that is what
    an empty declaration MEANS — so adjudicating only the declared list would say nothing about the
    standard actually in force. A declared bar replaces the incumbent on its measure, and the ratchet
    has already refused one looser than it.

    Args:
        behavior: The campaign's behavior, the scope a registered bar is keyed under.
        design: The campaign's declaration.
        profile: The host whose vocabulary this reads.

    Returns:
        ``(measure_id, threshold, higher_is_better, source)`` per bar, declared first, then registered,
        each in measure order.
    """
    declared = sorted(design.bars if design else [], key=lambda bar: bar.measure_id)
    overridden = {bar.measure_id for bar in declared}
    registered = sorted(
        (bar for bar in profile.bars.bars if bar.behavior == behavior and bar.measure not in overridden),
        key=lambda bar: bar.measure,
    )
    return [(bar.measure_id, bar.threshold, bar.direction == "higher_is_better", "declared") for bar in declared] + [
        (bar.measure, bar.threshold, bar.higher_is_better, "registered") for bar in registered
    ]


def _bar_reading(
    bar: BarName, members: list[EvalResult], judged_rows: dict[str, list[ScoreRecord]], *, profile: HostProfile
) -> _BarReading | None:
    """Read one bar's value over one cell's results, from where its kind is carried.

    Args:
        bar: The resolved bar name — its kind says where the value lives.
        members: The cell's results, faulted ones included. A measure is read over its own population
            (:func:`_collect_measures` leaves a fault out of a ``scored`` one); a judged dimension over
            the non-faulted results, as every score is.
        judged_rows: Projected judged-score rows by result id, for a judged dimension.
        profile: The host whose vocabulary this reads.

    Returns:
        ``(mean, sem, n, n_independent, interval)``, or None when no counted result carries the name.
        The interval is the one the cell's own reading states — a measure's summary, or a judged
        dimension's through the same rule (:func:`~threetears.evals.analysis.stats.observed_mean_interval`)
        — never one computed for the bar, so a bar and the cell it reads cannot state two widths.
    """
    if bar.kind in ("measure", "goal_state"):
        # A check's rate is the measure the walk publishes for it, read here rather than recomputed,
        # so a bar's value and the cell's measure are one number by construction.
        name = bar.name if bar.kind == "measure" else goal_check_measure(bar.name)
        collection = _measure_collection(members, profile=profile, undeclared="scored")
        summary = next((m for m in collection.measures if m.name == name), None)
        if summary is None or summary.mean is None:
            return None
        interval = None if summary.ci_low is None or summary.ci_high is None else (summary.ci_low, summary.ci_high)
        return summary.mean, summary.sem, summary.n, summary.n_independent, interval
    rows = [
        (float(record.value), record.test_case_id)
        for result in _non_faulted(members)
        for dimension, record in _judged_values(judged_rows.get(result.id, []))
        if dimension == bar.name and record.value is not None
    ]
    if not rows:
        return None
    values = [value for value, _ in rows]
    cases = [case for _, case in rows]
    return (
        sum(values) / len(values),
        clustered_standard_error(values, cases),
        len(values),
        len(set(cases)),
        observed_mean_interval(
            values, cases=cases, value_range=bar.descriptor.value_range, floor=bar.descriptor.interval_floor
        ),
    )


def _judged_bar_tier(
    bar: BarName,
    members: list[EvalResult],
    judged_rows: dict[str, list[ScoreRecord]],
    *,
    tiers: list[JudgeEvidenceTier],
) -> JudgedEvidenceTier | None:
    """The evidence tier a judged bar's verdict on one cell stands on, or None for a bar no judge scored.

    The weakest tier among the judges that served the very scores :func:`_bar_reading` read for the cell — its
    non-faulted results' values on the bar's dimension — by
    :func:`~threetears.evals.analysis.agreement.tier_for_judges`, the lookup every judged arm's tier is read by.
    """
    if bar.kind != "judged":
        return None
    served = [
        judge
        for result in _non_faulted(members)
        for dimension, _ in _judged_values(judged_rows.get(result.id, []))
        if dimension == bar.name and (judge := judge_key(result, dimension)) is not None
    ]
    return tier_for_judges(tiers, served)


def _bar_adjudications(
    behavior: str,
    design: CampaignDesign | None,
    results_by_cell: dict[_CellKey, list[EvalResult]],
    records: list[ScoreRecord],
    *,
    tiers: list[JudgeEvidenceTier],
    frontier_bar: float | None,
    profile: HostProfile,
) -> list[BarAdjudication]:
    """Adjudicate every applicable bar against every cell.

    **What a bar names is resolved by** :func:`~threetears.evals.kernel.declaration.resolve_bar_name`
    — the function the declaration gate asks — so a bar is read the way it was admitted, and a name
    the gate refuses is one this can never read. The gate supplies the template's names; this
    supplies the names the results actually carry (the judged dimensions they were scored on and
    the checks they evaluated), which is the same question asked of the evidence, and it lets a
    host's registered bar — which no gate saw — be resolved by the same rule.

    Every kind is read over the same population, each cell's non-faulted results; see
    :class:`BarVerdict`. A verdict on a judged dimension carries the evidence tier of the judges behind it
    (:func:`_judged_bar_tier`); every other verdict carries None.

    Args:
        behavior: The campaign's behavior.
        design: The campaign's declaration.
        results_by_cell: Each cell's results.
        records: The assembly's score projection rows, where a judged dimension's name survives.
        tiers: The judges' evidence tiers (``judge_evidence_tiers``).
        frontier_bar: The bar the frontier was given (:func:`_frontier_bar`), or None. A bar on pass^k carries
            no cell verdict, and its reason says whether the frontier read it.
        profile: The host whose vocabulary this reads.

    Returns:
        One adjudication per applicable bar, in :func:`_applicable_bars` order.
    """
    judged_rows: dict[str, list[ScoreRecord]] = {}
    for record in records:
        judged_rows.setdefault(record.result_id, []).append(record)
    all_results = [result for members in results_by_cell.values() for result in members]
    rubric_dimensions = {
        dimension: row.rubric_scale for dimension, row in _judged_rows(records) if row.rubric_scale is not None
    }
    goal_state_checks = {outcome.expression for result in all_results for outcome in result.goal_state_outcomes}
    counted_by_cell = {key: _non_faulted(members) for key, members in results_by_cell.items()}

    adjudications = []
    for measure_id, threshold, higher_is_better, source in _applicable_bars(behavior, design, profile=profile):
        resolved = resolve_bar_name(
            measure_id,
            rubric_dimensions=rubric_dimensions,
            goal_state_checks=goal_state_checks,
            measures=profile.measures,
        )
        state: _BarState
        reason: str | None = None
        verdicts: list[BarVerdict] = []
        if isinstance(resolved, UnreadableBarName):
            state = "not_numeric" if resolved.refusal == "not_numeric" else "names_no_stored_measure"
            reason = resolved.reason
            if measure_id == FRONTIER_RANKING_MEASURE:
                # Not "never read": no cell carries pass^k, and the frontier is where a bar on it is read.
                reason = (
                    f"{measure_id} is a rate over a contestant's cases that no single result carries, so no cell "
                    "verdict is given on it here. "
                    + (
                        "The frontier read this bar on each contestant's pass^k interval: see frontier.bar and each "
                        "point's bar_decision."
                        if frontier_bar is not None
                        else "The frontier was given no bar either; frontier_bar_withheld says why."
                    )
                )
        else:
            readings = {
                key: _bar_reading(resolved, members, judged_rows, profile=profile)
                for key, members in results_by_cell.items()
            }
            # A measure computed over every observation excluded none; reporting the cell's faults as
            # excluded from it would describe a population the value was not computed over.
            keeps_faults = resolved.kind == "measure" and resolved.descriptor.population == "all_observed"
            # The measure's declared margin — the difference too small to act on, the one margin the
            # history read's equivalence test is run against too. None holds the bar at its threshold.
            margin = resolved.descriptor.materiality_threshold
            if any(reading is not None for reading in readings.values()):
                state = "adjudicated"
                for key, members in results_by_cell.items():
                    mean, sem, n, n_independent, interval = readings[key] or (None, None, 0, 0, None)
                    verdicts.append(
                        BarVerdict(
                            variant_key=key[0],
                            apparatus_class_id=key[1],
                            run_ids=_member_run_ids(members),
                            value=mean,
                            sem=sem,
                            n=n,
                            n_independent=n_independent,
                            n_infra_excluded=0 if keeps_faults else len(members) - len(counted_by_cell[key]),
                            n_cannot_tell=_cannot_tell_on(resolved, counted_by_cell[key], judged_rows),
                            ci_low=None if interval is None else interval[0],
                            ci_high=None if interval is None else interval[1],
                            margin=margin,
                            cleared=None
                            if interval is None
                            else interval_clears(interval, threshold, margin=margin, higher_is_better=higher_is_better),
                            judge_evidence_tier=_judged_bar_tier(resolved, members, judged_rows, tiers=tiers),
                        )
                    )
            else:
                state = "names_no_stored_measure"
                reason = f"{measure_id} resolves as a {resolved.kind} name, but no non-faulted member result carries a value of it"
                # A turn's time or spend read nowhere because no result took a turn is not a name nothing
                # stores: every call failed, and the bar says so rather than send the reader to the registry.
                reads_a_turn = (
                    resolved.kind == "measure" and summary_population(resolved.descriptor, "scored") == "delivered"
                )
                members = [result for cell in results_by_cell.values() for result in cell]
                if reads_a_turn and members and not any(delivered_a_turn(result) for result in members):
                    reason = (
                        f"{measure_id} is a turn's time or spend, and no result took a turn: every result failed — the "
                        "candidate's model refused or errored on every call — or was excluded as a fault of the rig"
                    )
        adjudications.append(
            BarAdjudication(
                measure_id=measure_id,
                threshold=threshold,
                direction="higher_is_better" if higher_is_better else "lower_is_better",
                source=source,
                state=state,
                reason=reason,
                merit_axis=None if isinstance(resolved, UnreadableBarName) else resolved.descriptor.merit_axis,
                verdicts=verdicts,
            )
        )
    return adjudications


def _verdict_order(adjudications: list[BarAdjudication], design: CampaignDesign | None) -> VerdictOrder:
    """Order the adjudicated bars by the campaign's declared merit priority, and scope them per question.

    Only an ``adjudicated`` bar ranks: a bar with no verdict gives nothing to read in any order. A bar
    keeps its ``bar_adjudications`` position within its axis, so the priority reorders axes and never
    the bars on one.

    Args:
        adjudications: The bundle's bar adjudications, in their own order.
        design: The campaign's declaration, or None.

    Returns:
        The order. With no declaration or an empty priority, every adjudicated bar is unranked and no
        tier exists — the analysis must not invent a preference the campaign never stated.
    """
    priority = list(design.merit_priority) if design is not None else []
    by_axis: dict[MeritAxis | None, list[str]] = defaultdict(list)
    for bar in adjudications:
        if bar.state == "adjudicated":
            by_axis[bar.merit_axis].append(bar.measure_id)
    ranked = set(priority)
    unranked = [
        bar.measure_id
        for bar in adjudications
        if bar.state == "adjudicated" and (bar.merit_axis is None or bar.merit_axis not in ranked)
    ]
    questions = []
    for question in design.live_questions() if design is not None else []:
        if not question.merit_axes:
            continue
        # Each axis once: the declaration refuses a repeat, so the reader takes the list as it is.
        axes = sorted(question.merit_axes, key=lambda axis: priority.index(axis) if axis in ranked else len(priority))
        questions.append(
            QuestionScope(
                question_id=question.id,
                merit_axes=axes,
                bar_measure_ids=[measure for axis in axes for measure in by_axis.get(axis, [])],
                unbarred_axes=[axis for axis in axes if not by_axis.get(axis)],
            )
        )
    return VerdictOrder(
        merit_priority=priority,
        tiers=[MeritTier(axis=axis, bar_measure_ids=by_axis.get(axis, [])) for axis in priority],
        unranked_bar_measure_ids=unranked,
        questions=questions,
    )


__all__: list[str] = []

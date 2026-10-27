"""The analyst's analyses: computations over the frozen report input that no single figure states.

The actions of the ``Investigate: analyses`` workflow, each an
:class:`~threetears.evals.actions.catalogue.Action` over its operation in :mod:`threetears.evals.ops` and rendered by
:func:`~threetears.evals.actions.render.render_cited`. :func:`~threetears.evals.actions.analyst.analyst_actions`
gathers them onto the analyst's cut, and the editor's cut of the analyst tool is read off the workflows, so no file
lists their names.

- ``options_dominance``: which options survive the gates and which dominates, with extra axes compared beside the
  decision's on point estimates.
- ``decision_sensitivity``: how code's call, and each routable task's pick, moves as one declared margin, floor or
  threshold moves, or a hypothetical line the snapshot does not declare.
- ``measure_breakdown``: a measure slice by slice, each candidate against the control; exploratory.
- ``arms_compare``: two chosen arms on one measure, the difference overall and per slice with its interval.
- ``settle_cases``: the cases that would settle an arm against its line, or two arms apart.

Code's call in any of them is code's reading of the declared criteria, for the analyst and the editor to weigh, never
a constraint on what the report says.
"""

from __future__ import annotations

from typing import Annotated, Literal, get_args

from pydantic import Field

from threetears.evals.actions.catalogue import Action, Caller
from threetears.evals.actions.params import (
    ARM_FORMS,
    ArmFilter,
    ArmId,
    AspectFilter,
    MeasureKey,
    Offset,
    SizedLimit,
    SnapshotId,
)
from threetears.evals.actions.render import page_sizing, render_cited
from threetears.evals.ops.analyst_analyses import (
    BREAKDOWN_PAGE,
    BarsAs,
    COMPARE_PAGE,
    DOMINANCE_PAGE,
    SENSITIVITY_PAGE,
    SETTLE_PAGE,
    ArmsComparison,
    Breakdown,
    Dominance,
    Sensitivity,
    Settle,
    SliceBy,
    arms_compare,
    decision_sensitivity,
    measure_breakdown,
    options_dominance,
    settle_cases,
)
from threetears.evals.ops.host import OpsHost
from threetears.evals.schema.base import EvalBaseModel

#: The help index's group for these actions, which the editor's cut is read off.
ANALYSES_WORKFLOW = "Investigate: analyses"

#: An example snapshot id, as a freeze returns one.
_EXAMPLE_SNAPSHOT = "report-snapshot:0193a1b2c3d4e5f60718293a4b5c6d7e"

#: What every analysis that carries code's call says of it.
_CODE_CALL = (
    "Code's call here (modes, the dominant option) is code's reading of the declared criteria: weigh it, never take "
    "it as a constraint on the report."
)


#: How the caller reads the declared bars: one meaning for every analysis that takes it.
BarsAsParam = Annotated[
    BarsAs,
    Field(
        description="How to read the snapshot's declared bars: warnings (the default, as declared: a bar never "
        "removes an option) or gates (an option missing a bar is out). Your choice, from the question the report "
        "answers; code never infers it, and the result says which reading it took."
    ),
]


class DominanceParams(EvalBaseModel):
    """``options_dominance``: the snapshot, and any extra axes to compare the survivors on."""

    snapshot_id: SnapshotId
    axes: list[str] = Field(
        default_factory=list,
        description="Extra measures or judged dimensions to compare the survivors on, by the keys campaign_overview "
        "lists; compared on point estimates, descriptively. Empty: the decision's own axes alone.",
    )
    bars_as: BarsAsParam = "warnings"
    limit: SizedLimit = None
    offset: Offset = 0


class SensitivityParams(EvalBaseModel):
    """``decision_sensitivity``: the criterion to sweep, and the range."""

    snapshot_id: SnapshotId
    parameter: Annotated[
        str,
        Field(
            min_length=1,
            description="The criterion to sweep: margin:<aspect>, floor:<aspect>/<floor> or threshold:<criterion> "
            "(a gate's, guardrail's or bar's), by the ids the snapshot holds; or hypothetical:<measure> "
            "(hypothetical:<measure>/<statistic> for a p50 or p95), a gate the snapshot does not declare.",
        ),
    ]
    range_low: float | None = Field(
        default=None,
        description="Where the sweep starts, in the criterion's units; omitted, zero for a margin or a floor and no "
        "lower end for a threshold.",
    )
    range_high: float | None = Field(
        default=None, description="Where the sweep ends, in the criterion's units; omitted, no upper end."
    )
    bars_as: BarsAsParam = "warnings"
    limit: SizedLimit = None
    offset: Offset = 0


class BreakdownParams(EvalBaseModel):
    """``measure_breakdown``: the measure, the candidate, and how to slice."""

    snapshot_id: SnapshotId
    measure_key: MeasureKey
    arm_filter: ArmFilter = None
    slice_by: SliceBy = Field(
        default="stratum", description="How to slice: stratum, task, or cell (a target-matrix cell)."
    )
    aspect_filter: AspectFilter = None
    limit: SizedLimit = None
    offset: Offset = 0


#: The arms a separation reads: one meaning for both actions that take them.
ArmIds = Annotated[
    list[ArmId],
    Field(
        min_length=1,
        max_length=2,
        description="One arm or two, each " + ARM_FORMS + ": arms_compare reads two, the first minus the "
        "second; settle_cases reads one against a line, or two apart.",
    ),
]


#: A what-if count, one meaning for both actions that take one.
AssumedCases = Annotated[
    int | Literal["share"] | None,
    Field(
        description="A what-if for a slice whose figure states no count (so no spread): the cases each such slice is "
        "read as holding, a whole number, or 'share' for the arm's own distinct cases times the slice's traffic share. Every "
        "figure read under it is keyed assumed.<cases>. (or assumed.share.) so it reads as an assumption; a place "
        "whose figures state their count is read as they state it. Omitted: none assumed.",
    ),
]


class CompareParams(EvalBaseModel):
    """``arms_compare``: the two arms and the measure."""

    snapshot_id: SnapshotId
    arm_ids: ArmIds
    measure_key: MeasureKey
    assumed_cases: AssumedCases = None
    limit: SizedLimit = None
    offset: Offset = 0


class SettleParams(EvalBaseModel):
    """``settle_cases``: the arm or arms, and what to settle them against."""

    snapshot_id: SnapshotId
    arm_ids: ArmIds
    measure_key: Annotated[str | None, get_args(MeasureKey)[1]] = None
    criterion_id: str | None = Field(
        default=None,
        min_length=1,
        description="For one arm, the one declared criterion (gate, guardrail, floor or bar) to settle it against, "
        "by the id campaign_overview lists; omitted, every criterion reading the arm.",
    )
    hypothetical_line: float | None = Field(
        default=None,
        description="For one arm, a line the snapshot does not declare, on measure_key, in the figure's own units "
        "(a rate on the scale measure_read prints it on); every figure read against it is keyed hypothetical.",
    )
    against: str | None = Field(
        default=None,
        min_length=1,
        description="For one arm, the one margin or floor to settle its reading against: margin:<aspect> or "
        "floor:<aspect>/<floor>, as decision_sensitivity names them; omitted, with no criterion_id or line, every "
        "criterion, margin and floor reading the arm.",
    )
    assumed_cases: AssumedCases = None
    bars_as: BarsAsParam = "warnings"
    limit: SizedLimit = None
    offset: Offset = 0


async def _arms_compare(host: OpsHost, caller: Caller, params: CompareParams) -> ArmsComparison:
    limit, fits = page_sizing("arms_compare", params, COMPARE_PAGE)
    return await arms_compare(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        arms=params.arm_ids,
        measure_key=params.measure_key,
        assumed_cases=params.assumed_cases,
        limit=limit,
        offset=params.offset,
        fits=fits,
    )


async def _settle_cases(host: OpsHost, caller: Caller, params: SettleParams) -> Settle:
    limit, fits = page_sizing("settle_cases", params, SETTLE_PAGE)
    return await settle_cases(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        arms=params.arm_ids,
        measure_key=params.measure_key,
        criterion_id=params.criterion_id,
        hypothetical_line=params.hypothetical_line,
        against=params.against,
        assumed_cases=params.assumed_cases,
        bars_as=params.bars_as,
        limit=limit,
        offset=params.offset,
        fits=fits,
    )


async def _options_dominance(host: OpsHost, caller: Caller, params: DominanceParams) -> Dominance:
    limit, fits = page_sizing("options_dominance", params, DOMINANCE_PAGE)
    return await options_dominance(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        axes=params.axes,
        bars_as=params.bars_as,
        limit=limit,
        offset=params.offset,
        fits=fits,
    )


async def _decision_sensitivity(host: OpsHost, caller: Caller, params: SensitivityParams) -> Sensitivity:
    limit, fits = page_sizing("decision_sensitivity", params, SENSITIVITY_PAGE)
    return await decision_sensitivity(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        parameter=params.parameter,
        range_low=params.range_low,
        range_high=params.range_high,
        bars_as=params.bars_as,
        limit=limit,
        offset=params.offset,
        fits=fits,
    )


async def _measure_breakdown(host: OpsHost, caller: Caller, params: BreakdownParams) -> Breakdown:
    limit, fits = page_sizing("measure_breakdown", params, BREAKDOWN_PAGE)
    return await measure_breakdown(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        measure_key=params.measure_key,
        arm=params.arm_filter,
        slice_by=params.slice_by,
        aspect=params.aspect_filter,
        limit=limit,
        offset=params.offset,
        fits=fits,
    )


def analyst_analyses_actions() -> tuple[Action, ...]:
    """The ``Investigate: analyses`` actions.

    :return: The actions, in the order help lists them.
    :rtype: tuple[Action, ...]
    """
    return (
        Action(
            name="options_dominance",
            summary="Which options survive the gates, which dominates on cost and the asked aspects, and why.",
            workflow=ANALYSES_WORKFLOW,
            permission="read",
            params=DominanceParams,
            result=Dominance,
            handler=_options_dominance,
            render=render_cited,
            paged=True,
            example={"snapshot_id": _EXAMPLE_SNAPSHOT, "axes": ["latency_ms"]},
            detail=(
                "Each survivor comes with its gate status, its standing on the asked aspects (shown better, shown no "
                "worse, behind in a pair), which survivors are at least as good on every axis while it is not, and the "
                "figures behind it: cost per 1,000 of what the cost counts, its gap from each gate, its asked "
                "readings, pairs and separating contrasts, and each bar or floor criterion it misses or borders on "
                "the criterion's own aspect (warnings). Options the gates removed are listed with each figure they "
                "fail on, by its keys (gates), and each guardrail read breached. An option a gate has no figure for "
                "somewhere and no figure fails is undecided: listed apart with its unknown places, never as a fail, "
                "with any fact its figures over every case imply, labelled inferred. bars_as=gates reads the "
                "declared bars as gates, your choice; the result says which. weighs names the axes the snapshot "
                "declares; where cost is the only one, code makes no dominance call (code_call.withheld says why) "
                "rather than name the cheapest. axes adds measures or judged dimensions beside the decision's axes, "
                "compared on point estimates and marked descriptive; dominant_with_axes says who would dominate with "
                "them, and each survivor's dominated_with_axes who is ahead of it on all of them. Options are paged in "
                "the snapshot's arm order; without a limit a page "
                f"holds as many as fit one result, up to {DOMINANCE_PAGE}. " + _CODE_CALL
            ),
        ),
        Action(
            name="decision_sensitivity",
            summary="How code's call and per-task picks move as a declared margin, floor or threshold, or a "
            "hypothetical line, moves over a range.",
            workflow=ANALYSES_WORKFLOW,
            permission="read",
            params=SensitivityParams,
            result=Sensitivity,
            handler=_decision_sensitivity,
            render=render_cited,
            paged=True,
            example={"snapshot_id": _EXAMPLE_SNAPSHOT, "parameter": "margin:answers"},
            detail=(
                "parameter names one criterion: margin:<aspect> (every reading on the aspect read against that "
                "margin), floor:<aspect>/<floor> (that floor's margin, read on each candidate's interval) or "
                "threshold:<criterion> (a gate's, guardrail's or bar's threshold; a bar moves the call only with "
                "bars_as=gates, your choice, and otherwise lists where each arm's figure crosses it). The range is cut at every value an "
                "arm's status flips at, each a figure key (a flip margin, a floor's held_above or broken_at, a "
                "criterion's figure for an arm), and the call is read on each piece, with each routable task's pick "
                "(the cheapest arm passing there and in every cross-cutting task) where the snapshot holds routable "
                "tasks; segments join pieces with the same call and picks; holds says whether the declared call "
                "stands across the range, picks_hold whether the picks do. hypothetical:<measure> sweeps a gate the "
                "snapshot does not declare, read where it states the statistic (overall, else in each task or cell); "
                "the result is marked hypothetical and its flips are the arms' own figures. Flips are paged with "
                "limit and offset, each page holding the segments that start at its flips; without a limit a page "
                f"holds as many flips as fit one result, up to {SENSITIVITY_PAGE}. A weighting is not a parameter: the input carries none. "
                + _CODE_CALL
            ),
        ),
        Action(
            name="measure_breakdown",
            summary="A measure slice by slice: each candidate against the control in each stratum, task or cell.",
            workflow=ANALYSES_WORKFLOW,
            permission="read",
            params=BreakdownParams,
            result=Breakdown,
            handler=_measure_breakdown,
            render=render_cited,
            paged=True,
            example={"snapshot_id": _EXAMPLE_SNAPSHOT, "measure_key": "tone.kept", "slice_by": "stratum"},
            detail=(
                "Each slice gives its size, the control's mean on the measure, and each candidate's mean, its "
                "difference from the control there (vs_control) and, across one stratification's strata, the share "
                "of its loss the stratum carries (loss_share); concentrated_in names the stratum carrying the most, "
                "over every slice. Exploratory: point estimates, no interval, no test. arm_filter shows one "
                "candidate (default: every one); aspect_filter narrows strata to one aspect's. A snapshot with no "
                "control is refused: read a measure per stratum with measure_read (by=stratum) instead. Without a limit a "
                f"page holds as many slices as fit one result, up to {BREAKDOWN_PAGE}."
            ),
        ),
        Action(
            name="arms_compare",
            summary="Two chosen arms on one measure: the difference overall and in each slice, with its interval.",
            workflow=ANALYSES_WORKFLOW,
            permission="read",
            params=CompareParams,
            result=ArmsComparison,
            handler=_arms_compare,
            render=render_cited,
            paged=True,
            example={"snapshot_id": _EXAMPLE_SNAPSHOT, "arm_ids": ["fast.v2", "base"], "measure_key": "tone.kept"},
            detail=(
                "Any two arms, a control or none: the first minus the second, overall and in each task, cell and "
                "stratum where both state a mean, each by its compare.* keys, in points for a rate. The interval is "
                "paired where a contrast states one arm against the other (the control) overall, else the two arms' "
                "own spreads combined as independent (a stated interval or half-width at its level, a standard "
                "error, or a pass/fail rate's binomial spread from its count); basis says which, and whether a "
                "paired interval is corrected as its family's own correction says, excludes_zero "
                "whether the interval clears zero, ahead which arm reads better where it does. counts says what each "
                "arm's count counts (cases, trials or observations); a paired row's contrast states none "
                "(counted_unstated). may_be_narrow marks an interval that may be too narrow, where it counts trials "
                "or an unstated unit and the arms ran more trials than cases (nested cites them): say so where you "
                "cite it. A measure neither arm "
                "states a spread for is refused, naming measure_read for the two means. A normal approximation, "
                "never a test the pipeline ran. A slice whose figures state no count has no binomial spread: "
                "assumed_cases reads it under a what-if count, keyed assumed.<cases>.compare.*. Without a limit a "
                f"page holds as many places as fit one result, up to {COMPARE_PAGE}."
            ),
        ),
        Action(
            name="settle_cases",
            summary="The cases that would settle an arm against its line, or two arms apart, and how often today's "
            "design reads the wrong way.",
            workflow=ANALYSES_WORKFLOW,
            permission="read",
            params=SettleParams,
            result=Settle,
            handler=_settle_cases,
            render=render_cited,
            paged=True,
            example={"snapshot_id": _EXAMPLE_SNAPSHOT, "arm_ids": ["fast.v2"], "criterion_id": "cell_floor"},
            detail=(
                "One arm: against every line it is judged by (each declared criterion at each place it reads it, "
                "each margin its readings are read against, each floor read on it), or criterion_id alone, or "
                "against (margin:<aspect> or floor:<aspect>/<floor>, keyed settle.margin.* and settle.floor.*: the "
                "change from the control against minus the margin, on the reading's own interval), or a "
                "hypothetical_line on measure_key, which the snapshot does not declare (keyed "
                "settle.hypothetical.*); one-sided, with the two-sided sizes a page's intervals need beside "
                "(cases_needed_two_sided, cases_needed_two_sided_80). Two arms with measure_key: apart, "
                "overall and in each slice holding both; two-sided, as independent arms. Each row gives today's "
                "estimate and line, the distance, settles (clears where today's estimate passes, misses where the "
                "sizes would confirm a miss), cases_needed (an interval at the snapshot's level centred on today's "
                "estimate sits wholly on today's side: half such designs do if the truth equals today's estimate), "
                "cases_needed_80 (four in five do), wrong_side (how often a fresh run of today's size reads the "
                "other way), level with its sidedness, and the count, what it counts (count_of) and the spread "
                "basis it rests on, each a settle.* or spread.* key. Sizes count cases where the figure states them, each "
                "run as often as today (repeats), else the trials or observations it counts (count_of); trials_needed and its siblings are the trials that takes on a repeated design, where a "
                "pass/fail rate's spread is read on its distinct cases, each counted once (the widest its repeats "
                "allow, so cases_needed is the most cases they could need). A slice "
                "whose figure states no count has no spread: assumed_cases reads it under a what-if count, keyed "
                "assumed.<cases>.*. A criterion declared on an aspect reads that aspect's figure, and each criterion "
                "row says whether it gates as you read the bars (gates; bars_as). A p50 or p95, and a figure with no "
                "spread or count, say why in unsettled. An estimate exactly on its line is answered: settles reads "
                "on_the_line, wrong_side one half, no size settles it at today's estimate, and what_would names the "
                "call that sizes a run for a truth away from the line. Without a limit a page holds as many "
                f"rows as fit one result, up to {SETTLE_PAGE}."
            ),
        ),
    )


__all__ = ["ANALYSES_WORKFLOW", "analyst_analyses_actions"]

"""The analyst's analyses: computations over the frozen report input that no single figure states.

The actions of the ``Investigate: analyses`` workflow, each an
:class:`~threetears.evals.actions.catalogue.Action` over its operation in :mod:`threetears.evals.ops` and rendered by
:func:`~threetears.evals.actions.render.render_cited`. :func:`~threetears.evals.actions.analyst.analyst_actions`
gathers them onto the analyst's cut, and the analyst's and editor's toolsets are read off this workflow, so no file
lists their names.

- ``options_dominance``: which options survive the gates and which dominates, with extra axes compared beside the
  decision's on point estimates.
- ``decision_sensitivity``: how code's call moves as one declared margin, floor or threshold moves.
- ``measure_breakdown``: a measure slice by slice, each candidate against the control; exploratory.

Code's call in any of them is code's reading of the declared criteria, for the analyst and the editor to weigh, never
a constraint on what the report says.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from threetears.evals.actions.catalogue import Action, Caller
from threetears.evals.actions.params import ArmFilter, AspectFilter, Limit, MeasureKey, Offset, SnapshotId
from threetears.evals.actions.render import render_cited
from threetears.evals.ops.analyst_analyses import (
    BREAKDOWN_PAGE,
    SENSITIVITY_PAGE,
    Breakdown,
    Dominance,
    Sensitivity,
    SliceBy,
    decision_sensitivity,
    measure_breakdown,
    options_dominance,
)
from threetears.evals.ops.host import OpsHost
from threetears.evals.schema.base import EvalBaseModel

#: The help index's group for these actions, which the toolsets are read off.
ANALYSES_WORKFLOW = "Investigate: analyses"

#: An example snapshot id, as a freeze returns one.
_EXAMPLE_SNAPSHOT = "report-snapshot:0193a1b2c3d4e5f60718293a4b5c6d7e"

#: What every analysis that carries code's call says of it.
_CODE_CALL = (
    "Code's call here (modes, the dominant option) is code's reading of the declared criteria: weigh it, never take "
    "it as a constraint on the report."
)


class DominanceParams(EvalBaseModel):
    """``options_dominance``: the snapshot, and any extra axes to compare the survivors on."""

    snapshot_id: SnapshotId
    axes: list[str] = Field(
        default_factory=list,
        description="Extra measures or judged dimensions to compare the survivors on, by the keys campaign_overview "
        "lists; compared on point estimates, descriptively. Empty: the decision's own axes alone.",
    )


class SensitivityParams(EvalBaseModel):
    """``decision_sensitivity``: the criterion to sweep, and the range."""

    snapshot_id: SnapshotId
    parameter: Annotated[
        str,
        Field(
            min_length=1,
            description="The criterion to sweep: margin:<aspect>, floor:<aspect>/<floor> or threshold:<criterion>, "
            "by the ids the snapshot holds.",
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
    limit: Limit = SENSITIVITY_PAGE
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
    limit: Limit = BREAKDOWN_PAGE
    offset: Offset = 0


async def _options_dominance(host: OpsHost, caller: Caller, params: DominanceParams) -> Dominance:
    return await options_dominance(host, caller.scope_id, snapshot_id=params.snapshot_id, axes=params.axes)


async def _decision_sensitivity(host: OpsHost, caller: Caller, params: SensitivityParams) -> Sensitivity:
    return await decision_sensitivity(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        parameter=params.parameter,
        range_low=params.range_low,
        range_high=params.range_high,
        limit=params.limit,
        offset=params.offset,
    )


async def _measure_breakdown(host: OpsHost, caller: Caller, params: BreakdownParams) -> Breakdown:
    return await measure_breakdown(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        measure_key=params.measure_key,
        arm=params.arm_filter,
        slice_by=params.slice_by,
        aspect=params.aspect_filter,
        limit=params.limit,
        offset=params.offset,
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
            example={"snapshot_id": _EXAMPLE_SNAPSHOT, "axes": ["latency_ms"]},
            detail=(
                "Each survivor comes with its gate status, its standing on the asked aspects (shown better, shown no "
                "worse, behind in a pair), which survivors are at least as good on every axis while it is not, and "
                "the figures behind it: cost per 1,000 replies, its gap from each gate, its asked readings, pairs "
                "and separating contrasts. Options the gates removed are listed with each failing figure. axes adds "
                "measures or judged dimensions beside the decision's axes, compared on point estimates and marked "
                "descriptive; dominant_with_axes says who would dominate with them. " + _CODE_CALL
            ),
        ),
        Action(
            name="decision_sensitivity",
            summary="How code's call moves as one declared margin, floor or threshold moves over a range.",
            workflow=ANALYSES_WORKFLOW,
            permission="read",
            params=SensitivityParams,
            result=Sensitivity,
            handler=_decision_sensitivity,
            render=render_cited,
            example={"snapshot_id": _EXAMPLE_SNAPSHOT, "parameter": "margin:answers"},
            detail=(
                "parameter names one criterion: margin:<aspect> (every reading on the aspect read against that "
                "margin), floor:<aspect>/<floor> (that floor's margin, read on each candidate's interval) or "
                "threshold:<criterion> (a gate's or guardrail's threshold). The range is cut at every value an "
                "arm's status flips at, each a figure key (a flip margin, a floor's held_above or broken_at, a "
                "criterion's figure for an arm), and the call is read on each piece; segments join pieces with the "
                "same call, and holds says whether the declared call stands across the range. Flips are paged with "
                "limit and offset. A weighting is not a parameter: the input carries none. " + _CODE_CALL
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
            example={"snapshot_id": _EXAMPLE_SNAPSHOT, "measure_key": "tone.kept", "slice_by": "stratum"},
            detail=(
                "Each slice gives its size, the control's mean on the measure, and each candidate's mean, its "
                "difference from the control there (vs_control) and, across one stratification's strata, the share "
                "of its loss the stratum carries (loss_share); concentrated_in names the stratum carrying the most, "
                "over every slice. Exploratory: point estimates, no interval, no test. arm_filter shows one "
                "candidate (default: every one); aspect_filter narrows strata to one aspect's. A snapshot with no "
                "control is refused: read a measure per stratum with measure_read (by=stratum) instead."
            ),
        ),
    )


__all__ = ["ANALYSES_WORKFLOW", "analyst_analyses_actions"]

"""The analyst's fact tools: what the frozen report input holds, each figure by its key.

The actions of the ``Investigate: facts`` workflow, each an
:class:`~threetears.evals.actions.catalogue.Action` over its operation in :mod:`threetears.evals.ops` and rendered by
:func:`~threetears.evals.actions.render.render_cited`. :func:`~threetears.evals.actions.analyst.analyst_actions`
gathers them onto the analyst's cut, and the editor's cut of the analyst tool is read off the workflows, so no file
lists their names.

``campaign_overview`` comes first: it lists every id the others take (arms, aspects, measure keys, strata, tasks,
cells), and a filter naming anything else is refused with that remedy.

**A page is sized to one result.** A paged tool called without a ``limit`` returns as many rows as its text holds
within 12,000 characters (what one tool result may take of an analyst's context), up to its default count;
``next_offset`` reads on. Rows are never shortened or left out, only carried to the next page. A ``limit`` the caller
names is the page's size, as named.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, get_args

from pydantic import BaseModel, Field

from threetears.evals.actions.catalogue import Action, Caller
from threetears.evals.actions.params import ArmFilter, AspectFilter, Limit, MeasureKey, Offset, SnapshotId
from threetears.evals.actions.render import render_cited
from threetears.evals.ops.analyst_facts import (
    CONTRASTS_LIMIT,
    EXAMPLES_LIMIT,
    MEASURE_LIMIT,
    READINGS_LIMIT,
    CampaignOverview,
    CaveatListing,
    ContrastListing,
    ExampleListing,
    MeasureBy,
    MeasureFigures,
    MeasureStatistic,
    ReadingsListing,
    campaign_overview,
    caveats_list,
    contrasts_list,
    examples_list,
    measure_read,
    readings_list,
)
from threetears.evals.ops.host import OpsHost
from threetears.evals.schema.base import EvalBaseModel

#: The help index's group for these actions, which the editor's cut is read off.
FACTS_WORKFLOW = "Investigate: facts"

#: The most characters a page a caller does not size runs to: what one tool result may take of an analyst's context.
_PAGE_TEXT = 12_000

#: An example snapshot id, as a freeze returns one.
_EXAMPLE_SNAPSHOT = "report-snapshot:0193a1b2c3d4e5f60718293a4b5c6d7e"

#: A page's size, as :data:`~threetears.evals.actions.params.Limit` declares it, which a caller may leave out to have
#: the page sized to one result.
_SizedLimit = Annotated[int | None, get_args(Limit)[1]]

MeasureFilter = Annotated[
    str | None,
    Field(
        min_length=1,
        description="Read only this measure or judged dimension, by the key campaign_overview lists; omitted reads "
        "every one.",
    ),
]
StratumFilter = Annotated[
    str | None,
    Field(min_length=1, description="Read only this stratum or task, by its id; omitted reads every one."),
]
OutcomeFilter = Annotated[
    str | None,
    Field(min_length=1, description="Read only this failure class (an example's outcome); omitted reads every one."),
]


def _fits(page: BaseModel) -> bool:
    """Whether a page's text, as the tool renders it, is within :data:`_PAGE_TEXT`."""
    return len(render_cited(page)) <= _PAGE_TEXT


def _sizing[T: BaseModel](limit: int | None, default: int) -> tuple[int, Callable[[T], bool] | None]:
    """The page size an operation takes, and the test of length it sizes a page by: the caller's ``limit`` as named,
    else up to ``default`` rows, as many as fit."""
    return (limit, None) if limit is not None else (default, _fits)


def _default_statistics() -> list[MeasureStatistic]:
    """The statistics a measure is read with when none are named: the mean, its interval and its observations."""
    return ["mean", "ci", "n"]


class SnapshotParams(EvalBaseModel):
    """``campaign_overview``, ``caveats_list``: the snapshot to read."""

    snapshot_id: SnapshotId


class ReadingsListParams(EvalBaseModel):
    """``readings_list``: the snapshot, the aspect and arm to narrow to, and the page."""

    snapshot_id: SnapshotId
    aspect_filter: AspectFilter = None
    arm_filter: ArmFilter = None
    limit: _SizedLimit = None
    offset: Offset = 0


class ContrastsListParams(EvalBaseModel):
    """``contrasts_list``: the snapshot, the arm, measure and aspect to narrow to, and the page."""

    snapshot_id: SnapshotId
    arm_filter: ArmFilter = None
    measure_filter: MeasureFilter = None
    aspect_filter: AspectFilter = None
    limit: _SizedLimit = None
    offset: Offset = 0


class MeasureReadParams(EvalBaseModel):
    """``measure_read``: the snapshot, the measure, what to read it by, the statistics, and the page."""

    snapshot_id: SnapshotId
    measure_key: MeasureKey
    by: MeasureBy = Field(
        default="arm",
        description="Read each arm's figure (arm), or each arm's within each stratum, task or target-matrix cell.",
    )
    statistics: list[MeasureStatistic] = Field(
        default_factory=_default_statistics,
        min_length=1,
        description="The statistics to read, each where the snapshot states it; vs_control (a candidate's mean minus "
        "the control's within a slice) needs a slice and a control.",
    )
    aspect_filter: AspectFilter = None
    arm_filter: ArmFilter = None
    limit: _SizedLimit = None
    offset: Offset = 0


class ExamplesListParams(EvalBaseModel):
    """``examples_list``: the snapshot, the arm, stratum, aspect and failure class to narrow to, and the page."""

    snapshot_id: SnapshotId
    arm_filter: ArmFilter = None
    stratum_filter: StratumFilter = None
    aspect_filter: AspectFilter = None
    outcome_filter: OutcomeFilter = None
    limit: _SizedLimit = None
    offset: Offset = 0


async def _campaign_overview(host: OpsHost, caller: Caller, params: SnapshotParams) -> CampaignOverview:
    return await campaign_overview(host, caller.scope_id, snapshot_id=params.snapshot_id)


async def _readings_list(host: OpsHost, caller: Caller, params: ReadingsListParams) -> ReadingsListing:
    limit, fits = _sizing(params.limit, READINGS_LIMIT)
    return await readings_list(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        aspect_filter=params.aspect_filter,
        arm_filter=params.arm_filter,
        limit=limit,
        offset=params.offset,
        fits=fits,
    )


async def _contrasts_list(host: OpsHost, caller: Caller, params: ContrastsListParams) -> ContrastListing:
    limit, fits = _sizing(params.limit, CONTRASTS_LIMIT)
    return await contrasts_list(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        arm_filter=params.arm_filter,
        measure_filter=params.measure_filter,
        aspect_filter=params.aspect_filter,
        limit=limit,
        offset=params.offset,
        fits=fits,
    )


async def _measure_read(host: OpsHost, caller: Caller, params: MeasureReadParams) -> MeasureFigures:
    limit, fits = _sizing(params.limit, MEASURE_LIMIT)
    return await measure_read(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        measure_key=params.measure_key,
        by=params.by,
        statistics=params.statistics,
        aspect_filter=params.aspect_filter,
        arm_filter=params.arm_filter,
        limit=limit,
        offset=params.offset,
        fits=fits,
    )


async def _examples_list(host: OpsHost, caller: Caller, params: ExamplesListParams) -> ExampleListing:
    limit, fits = _sizing(params.limit, EXAMPLES_LIMIT)
    return await examples_list(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        arm_filter=params.arm_filter,
        stratum_filter=params.stratum_filter,
        aspect_filter=params.aspect_filter,
        outcome_filter=params.outcome_filter,
        limit=limit,
        offset=params.offset,
        fits=fits,
    )


async def _caveats_list(host: OpsHost, caller: Caller, params: SnapshotParams) -> CaveatListing:
    return await caveats_list(host, caller.scope_id, snapshot_id=params.snapshot_id)


def analyst_facts_actions() -> tuple[Action, ...]:
    """The ``Investigate: facts`` actions: the overview, the readings, contrasts, a measure, examples and caveats.

    :return: The actions, in the order help lists them.
    :rtype: tuple[Action, ...]
    """
    snapshot = {"snapshot_id": _EXAMPLE_SNAPSHOT}
    return (
        Action(
            name="campaign_overview",
            summary="Read what a snapshot holds first: its campaign, arms, aspects, measures, criteria and caveats.",
            workflow=FACTS_WORKFLOW,
            permission="read",
            params=SnapshotParams,
            result=CampaignOverview,
            handler=_campaign_overview,
            render=render_cited,
            example=snapshot,
            detail=(
                "Every id another tool takes is listed here: each arm (cited arm:<id>), each aspect and what rows it "
                "holds, each measure and judged dimension by its key with the key form that reads it and what it can "
                "be read by, each criterion and its threshold, each caveat with its reach, the late criteria, prior "
                "insights, and the name a reader is shown for each id. Every figure is written key = reads: cite the "
                "key, never the number."
            ),
        ),
        Action(
            name="readings_list",
            summary="List each aspect's readings against the control, pair readings, floors and bars, by their keys.",
            workflow=FACTS_WORKFLOW,
            permission="read",
            params=ReadingsListParams,
            result=ReadingsListing,
            handler=_readings_list,
            render=render_cited,
            example=snapshot,
            detail=(
                "Rows are grouped by aspect, each with its margin and the contrast family correcting it. A reading "
                "carries its change, interval, margin, level, verdict and flip margin (derived: the margin above which "
                "it holds); a floor its status and held_above and broken_at margins (derived). Pages count every row "
                f"of every kind; without a limit a page holds as many as fit one result, up to {READINGS_LIMIT}."
            ),
        ),
        Action(
            name="contrasts_list",
            summary="List the corrected contrasts of each arm against the control, and the families correcting them.",
            workflow=FACTS_WORKFLOW,
            permission="read",
            params=ContrastsListParams,
            result=ContrastListing,
            handler=_contrasts_list,
            render=render_cited,
            example=snapshot,
            detail=(
                "Each contrast carries its delta, interval, verdict, adjusted p, level and materiality, and its "
                "confounds by caveat id (caveats_list says what each touches). A contrast whose arm and measure the "
                "snapshot holds on more than one aspect is keyed by its aspect. Without a limit a page holds as many "
                f"contrasts as fit one result, up to {CONTRASTS_LIMIT}."
            ),
        ),
        Action(
            name="measure_read",
            summary="Read one measure's or judged dimension's statistics, by arm or within each stratum, task or cell.",
            workflow=FACTS_WORKFLOW,
            permission="read",
            params=MeasureReadParams,
            result=MeasureFigures,
            handler=_measure_read,
            render=render_cited,
            example={**snapshot, "measure_key": "persona.holds_character", "by": "stratum"},
            detail=(
                "campaign_overview lists each measure key and the slices it can be read by. By arm, a measure read on "
                "more than one aspect is read on each, by its aspect's key. Within a slice, vs_control is a candidate's "
                "mean minus the control's there: derived and descriptive, with no interval and no test. Without a "
                f"limit a page holds as many rows as fit one result, up to {MEASURE_LIMIT}."
            ),
        ),
        Action(
            name="examples_list",
            summary="List the failed trials the snapshot froze: each trial's reply, reason and stored result.",
            workflow=FACTS_WORKFLOW,
            permission="read",
            params=ExamplesListParams,
            result=ExampleListing,
            handler=_examples_list,
            render=render_cited,
            example=snapshot,
            detail=(
                "The pool holds at most one failed trial per aspect, arm and stratum. A report names an example by its "
                "trial id. An arm or aspect filter keeps an example naming none. Long texts are cut to an excerpt's "
                f"length, the cut marked. Without a limit a page holds as many as fit one result, up to {EXAMPLES_LIMIT}."
            ),
        ),
        Action(
            name="caveats_list",
            summary="List every caveat: where it may land, whether code proved it immaterial, and what it touches.",
            workflow=FACTS_WORKFLOW,
            permission="read",
            params=SnapshotParams,
            result=CaveatListing,
            handler=_caveats_list,
            render=render_cited,
            example=snapshot,
            detail=(
                "Read from the snapshot: a confound touches the contrasts naming it, a short run its arm, a late "
                "criterion what it set, an unsettled aspect its candidates. A report's confidence rests on every "
                "report-reach caveat; an immaterial one is never restated, since code renders its checked line."
            ),
        ),
    )


__all__ = ["FACTS_WORKFLOW", "analyst_facts_actions"]

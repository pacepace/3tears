"""A campaign's lever effects, as an action of the ``Investigate: analyses`` workflow.

``levers_effects`` is an :class:`~threetears.evals.actions.catalogue.Action` over its operation in
:mod:`threetears.evals.ops.analyst_levers`, rendered by :func:`~threetears.evals.actions.render.render_cited`. It
belongs to the analyses workflow, so the analyst's tool and the editor's cut of it, read off the workflows, carry it:
:func:`~threetears.evals.actions.analyst.analyst_actions` gathers :func:`analyst_levers_actions` beside the other
analyses.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from threetears.evals.actions.analyst_analyses import ANALYSES_WORKFLOW
from threetears.evals.actions.catalogue import Action, Caller
from threetears.evals.actions.params import MeasureKey, Offset, SizedLimit, SnapshotId
from threetears.evals.actions.render import page_sizing, render_cited
from threetears.evals.ops.analyst_levers import LEVERS_LIMIT, LeverEffects, levers_effects
from threetears.evals.ops.host import OpsHost
from threetears.evals.schema.base import EvalBaseModel

#: An example snapshot id, as a freeze returns one.
_EXAMPLE_SNAPSHOT = "report-snapshot:0193a1b2c3d4e5f60718293a4b5c6d7e"

SettingFilter = Annotated[
    str | None,
    Field(
        min_length=1,
        description="Read only this setting's effects, by its name in the arms' settings (campaign_overview lists "
        "them); omitted reads every setting the arms vary.",
    ),
]
PlaceFilter = Annotated[
    str | None,
    Field(
        min_length=1,
        description="Read the measure within this one task, stratum or target-matrix cell (`<row>.<column>`), by its "
        "id; omitted reads each arm's overall figure.",
    ),
]


class LeversParams(EvalBaseModel):
    """``levers_effects``: the snapshot, the measure, the setting and place to narrow to, and the page."""

    snapshot_id: SnapshotId
    measure_key: MeasureKey
    setting_filter: SettingFilter = None
    place_filter: PlaceFilter = None
    limit: SizedLimit = None
    offset: Offset = 0


async def _levers_effects(host: OpsHost, caller: Caller, params: LeversParams) -> LeverEffects:
    limit, fits = page_sizing("levers_effects", params, LEVERS_LIMIT)
    return await levers_effects(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        measure_key=params.measure_key,
        setting=params.setting_filter,
        place=params.place_filter,
        limit=limit,
        offset=params.offset,
        fits=fits,
    )


def analyst_levers_actions() -> tuple[Action, ...]:
    """The ``Investigate: analyses`` action that reads what each setting the arms vary does to a measure.

    :return: ``levers_effects``.
    :rtype: tuple[Action, ...]
    """
    return (
        Action(
            name="levers_effects",
            summary="What each setting the arms vary does to a measure, between arms differing in it alone, and "
            "whether that changes with another setting.",
            workflow=ANALYSES_WORKFLOW,
            permission="read",
            params=LeversParams,
            result=LeverEffects,
            handler=_levers_effects,
            render=render_cited,
            paged=True,
            example={"snapshot_id": _EXAMPLE_SNAPSHOT, "measure_key": "pass_rate", "setting_filter": "retry"},
            detail=(
                "An effect is read only between two arms whose settings differ in one setting alone (a setting an arm "
                "leaves out has the control's value): the second arm's mean minus the first's, overall or within "
                "place_filter (a task, stratum or cell). Its interval is recovered from the two arms' own intervals "
                "as if independent (assumption says so), and there is none where either arm states none. An "
                "interaction is the same move made where one other setting differs (across): the second effect minus "
                "the first, and a verdict: reverses, differs (beyond the intervals), within, or no_interval. A setting "
                "whose effect reverses or differs is not one effect: report it per level of the other setting. Every "
                "lever pair is listed once; any pair, either way round, is a key (lever.<setting>.<from>.<to>."
                "<measure>[.<place>].effect, help topic figure_keys.levers). Arms that isolate no setting are refused, "
                "naming what each pair also differs in. Effects come first, then interactions; without a limit a page "
                f"holds as many as fit one result, up to {LEVERS_LIMIT}."
            ),
        ),
    )


__all__ = ["analyst_levers_actions"]

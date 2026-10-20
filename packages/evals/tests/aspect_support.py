"""A stored campaign's one aspect, declared again with some of its declaration changed.

A per-aspect test reads a fixture campaign through :func:`~threetears.evals.analysis.aspects.aspect_view` and builds
the changed campaign through :func:`~packages.evals.tests.factories.make_aspect_campaign`, so it never reads or writes
the stored campaign's own fields: how a campaign holds its aspects is the view's and the factory's business alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from threetears.evals.analysis.aspects import aspect_view
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignArm, DecisionClause

from packages.evals.tests.factories import make_aspect_campaign

#: A keyword :func:`redeclared` leaves as the campaign declared it.
_KEPT: Any = object()


def redeclared(
    campaign: EvalCampaign,
    *,
    design: Mapping[str, Any] | None = None,
    question: str = _KEPT,
    asked_at: str | None = _KEPT,
    control: CampaignArm | None = _KEPT,
    margin: float | None = _KEPT,
    floors: Sequence[DecisionClause] = _KEPT,
) -> EvalCampaign:
    """``campaign``'s one aspect declared again, with what is named changed and everything else as it was.

    :param campaign: A stored campaign of one aspect that declares a design.
    :ptype campaign: EvalCampaign
    :param design: Fields of the aspect's design to replace.
    :ptype design: Mapping[str, Any] | None
    :param question: The campaign's question; kept when omitted.
    :ptype question: str
    :param asked_at: When it was asked; kept when omitted.
    :ptype asked_at: str | None
    :param control: The control arm; kept when omitted.
    :ptype control: CampaignArm | None
    :param margin: The aspect's margin; kept when omitted.
    :ptype margin: float | None
    :param floors: The floors' clauses; kept when omitted.
    :ptype floors: Sequence[DecisionClause]
    :return: The campaign, its keys and runs as they were.
    :rtype: EvalCampaign
    """
    view = aspect_view(campaign)
    declared = view.declared
    if declared is None:
        raise ValueError("the campaign's aspect declares no design to change")
    asked = declared.question if question is _KEPT else question
    when = (declared.asked_at if asked else None) if asked_at is _KEPT else asked_at
    arm = declared.control if control is _KEPT else control
    # Declared again, the control is named with the rest of the declaration, at the time the question is asked.
    named = arm.model_copy(update={"named_at": when}) if arm is not None else None
    return make_aspect_campaign(
        label=view.aspect,
        template_id=view.template_id,
        run_ids=view.run_ids,
        design=declared.design.model_copy(update=dict(design or {})),
        question=asked,
        asked_at=when,
        control=named,
        margin=declared.margin if margin is _KEPT else margin,
        floors=[floor.clause for floor in declared.floors] if floors is _KEPT else floors,
        id=view.campaign_id,
        scope_id=view.scope_id,
        name=view.name,
        subject_id=view.subject_id,
        subject_kind=view.subject_kind,
    )


def with_aspect(
    campaign: EvalCampaign, *, design: Mapping[str, Any] | None = None, run_ids: Sequence[str] | None = None
) -> EvalCampaign:
    """``campaign`` with its one aspect's design fields or runs replaced, written past the authoring gate.

    :param campaign: A stored campaign of one aspect.
    :ptype campaign: EvalCampaign
    :param design: Fields of the aspect's design to replace; the aspect must declare one.
    :ptype design: Mapping[str, Any] | None
    :param run_ids: The aspect's runs, replacing its own.
    :ptype run_ids: Sequence[str] | None
    :return: The campaign, everything else as it was.
    :rtype: EvalCampaign
    """
    (aspect,) = campaign.aspects
    changes: dict[str, Any] = {}
    if design is not None:
        if aspect.design is None:
            raise ValueError("the campaign's aspect declares no design to change")
        changes["design"] = aspect.design.model_copy(update=dict(design))
    if run_ids is not None:
        changes["run_ids"] = list(run_ids)
    return campaign.model_copy(update={"aspects": [aspect.model_copy(update=changes)]})


def with_control(campaign: EvalCampaign, variant_key: str | None) -> EvalCampaign:
    """``campaign`` with ``variant_key`` its control arm, typed, or with no control.

    :param campaign: A stored campaign.
    :ptype campaign: EvalCampaign
    :param variant_key: The control's key, or None.
    :ptype variant_key: str | None
    :return: The campaign, its candidate arms kept.
    :rtype: EvalCampaign
    """
    candidates = [arm for arm in campaign.arms if arm.role != "control"]
    control = [] if variant_key is None else [CampaignArm(label="control", role="control", variant_key=variant_key)]
    return campaign.model_copy(update={"arms": [*control, *candidates]})


__all__ = ["redeclared", "with_aspect", "with_control"]

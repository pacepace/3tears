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
    return make_aspect_campaign(
        label=view.aspect,
        template_id=view.template_id,
        run_ids=view.run_ids,
        design=declared.design.model_copy(update=dict(design or {})),
        question=asked,
        asked_at=(declared.asked_at if asked else None) if asked_at is _KEPT else asked_at,
        control=declared.control if control is _KEPT else control,
        margin=declared.margin if margin is _KEPT else margin,
        floors=[floor.clause for floor in declared.floors] if floors is _KEPT else floors,
        id=view.campaign_id,
        scope_id=view.scope_id,
        name=view.name,
        subject_id=view.subject_id,
        subject_kind=view.subject_kind,
    )


__all__ = ["redeclared"]

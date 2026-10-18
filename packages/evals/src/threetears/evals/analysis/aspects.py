"""A campaign read one aspect at a time: what each aspect declared, and the runs it reads.

Every per-aspect reader — the evidence bundle, the floors, the aspect's fold of the page — reads a campaign through
:func:`aspect_view`, never through the stored campaign's fields, so the one place that knows how a campaign holds its
aspects is here. An :class:`AspectView` is the campaign's keys and one aspect: its template, its runs and its
:class:`AspectDeclaration`, which is what the campaign declared as that aspect reads it — the one question, the
control arm, the aspect's margin and design, and the floors on it.

**Over today's stored campaign, through the upgrade.** A stored campaign still holds one template, one bag of runs and
one design carrying its questions and its control, so it is read as the one aspect it upgrades to
(:func:`~threetears.evals.kernel.campaign_upgrade.upgrade_campaign_v15`): the same rules a stored campaign is upgraded
by, so the view and the upgrade cannot disagree about what a campaign declared.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import Field

from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.campaign_upgrade import upgrade_campaign_v15
from threetears.evals.kernel.declaration import Aspect, AspectDesign, CampaignArm, Floor
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.schema.base import EvalDocumentModel

__all__ = ["AspectDeclaration", "AspectView", "aspect_view", "aspect_views"]


class AspectDeclaration(EvalDocumentModel):
    """What a campaign declared, as one of its aspects reads it: the question, the control, the margin, the design
    and the floors.

    Operator input throughout, and none of it derived from observations: the realized design is the bundle's. The
    question and the control are the campaign's, shared by every aspect; the margin, the design and the floors are
    this aspect's own.
    """

    question: str = Field(
        description="The one question the campaign asks of its arms, in the operator's words. Empty = it asks none."
    )
    asked_at: str | None = Field(
        description=(
            "When the question was asked in full (ISO-8601): when its last part was asked, since a reading confirms "
            "only what was asked before the evidence existed. Never a time a later edit of the design re-stamps. When "
            "the campaign asks no question, the time its design was declared; None when it records no time."
        )
    )
    design: AspectDesign = Field(
        description=(
            "What this aspect set out to vary and hold itself to. Its `merit_axes` say what the question asks about "
            "here: None when the campaign asks no question, so every reading is exploratory though a design is "
            "declared; empty for an unscoped question, which asks about every axis."
        )
    )
    control: CampaignArm | None = Field(
        description="The campaign's control arm, every candidate read against it. None = no control declared, and the analysis says so rather than electing one."
    )
    margin: float | None = Field(
        description=(
            "How far, in points of a 0-100 pass rate, a candidate may fall below the control on this aspect before it "
            "is shown worse. None = no margin declared: shown better or unclear, never shown worse."
        )
    )
    floors: list[Floor] = Field(description="The floors on this aspect, in declaration order. Empty = none.")
    declared_at: str | None = Field(description="When this aspect's design was declared (ISO-8601).")
    declared_by: str = Field(description="Who declared the campaign; empty when unrecorded.")


class AspectView(EvalDocumentModel):
    """A campaign's keys and one of its aspects: the template, the runs and the declaration a per-aspect reader reads."""

    campaign_id: str = Field(description="The campaign this aspect belongs to.")
    scope_id: str = Field(description="The scope the campaign and its runs live in.")
    name: str = Field(description="The campaign's operator-facing name.")
    subject_id: str = Field(description="The subject the campaign evaluates, a reference the host owns.")
    subject_kind: str = Field(description="What kind of thing the subject is; blank when the campaign declared none.")
    behavior: str = Field(description="The campaign's behavior.")
    aspect: str = Field(description="The aspect's label, unique within its campaign.")
    template_id: str | None = Field(
        description="The eval_template the aspect's runs were gathered on; None names none."
    )
    run_ids: list[str] = Field(description="The runs this aspect reads, as curated.")
    declared: AspectDeclaration | None = Field(
        description="What the campaign declared, as this aspect reads it. None = exploratory: the aspect declares no design."
    )


def aspect_views(campaign: EvalCampaign) -> list[AspectView]:
    """Every aspect of ``campaign``, each as a per-aspect reader reads it, in the campaign's order.

    :param campaign: The stored campaign.
    :ptype campaign: EvalCampaign
    :return: One view per aspect; a campaign stored before aspects existed has one, labelled with its behavior.
    :rtype: list[AspectView]
    """
    upgraded = upgrade_campaign_v15(campaign.model_dump(mode="json"))
    arms = [CampaignArm.model_validate(arm) for arm in upgraded["arms"]]
    control = next((arm for arm in arms if arm.role == "control"), None)
    floors = [Floor.model_validate(floor) for floor in upgraded["floors"]]
    return [
        _view(campaign, Aspect.model_validate(aspect), upgraded, control=control, floors=floors)
        for aspect in upgraded["aspects"]
    ]


def aspect_view(campaign: EvalCampaign, aspect: str | None = None) -> AspectView:
    """One aspect of ``campaign``, by its label, or its only aspect when ``aspect`` is None.

    :param campaign: The stored campaign.
    :ptype campaign: EvalCampaign
    :param aspect: The aspect's label; None reads the campaign's only aspect.
    :ptype aspect: str | None
    :return: The aspect's view.
    :rtype: AspectView
    :raises RefusedInput: ``aspect`` is None and the campaign has more than one aspect, or it names none of them.
    """
    views = aspect_views(campaign)
    labels = [view.aspect for view in views]
    if aspect is None:
        if len(views) == 1:
            return views[0]
        raise RefusedInput(
            f"campaign {campaign.id} has {len(views)} aspects ({', '.join(labels)}), so which one to read is not implied",
            remedy=f"name the aspect to read: one of {', '.join(labels)}",
        )
    found = next((view for view in views if view.aspect == aspect), None)
    if found is None:
        raise RefusedInput(
            f"campaign {campaign.id} has no aspect {aspect!r}",
            remedy=f"name one of its aspects: {', '.join(labels)}",
        )
    return found


def _view(
    campaign: EvalCampaign,
    aspect: Aspect,
    upgraded: dict[str, Any],
    *,
    control: CampaignArm | None,
    floors: Sequence[Floor],
) -> AspectView:
    """One aspect's view over the campaign it belongs to."""
    declared = (
        None
        if aspect.design is None
        else AspectDeclaration(
            question=upgraded["question"],
            asked_at=upgraded["declared_at"],
            design=aspect.design,
            control=control,
            margin=aspect.margin,
            floors=[floor for floor in floors if floor.aspect == aspect.label],
            declared_at=aspect.declared_at,
            declared_by=upgraded["declared_by"],
        )
    )
    return AspectView(
        campaign_id=campaign.id,
        scope_id=campaign.scope_id,
        name=campaign.name,
        subject_id=campaign.subject_id,
        subject_kind=campaign.subject_kind,
        behavior=campaign.behavior,
        aspect=aspect.label,
        template_id=aspect.template_id,
        run_ids=list(aspect.run_ids),
        declared=declared,
    )

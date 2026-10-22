"""A campaign read one aspect at a time: what each aspect declared, and the runs it reads.

Every per-aspect reader — the evidence bundle, the floors, the aspect's fold of the page — reads a campaign through
:func:`aspect_view`, never through the stored campaign's fields, so the one place that knows how a campaign holds its
aspects is here. An :class:`AspectView` is the campaign's keys and one aspect: its template, its runs and its
:class:`AspectDeclaration`, which is what the campaign declared as that aspect reads it — the one question, the
control arm, the aspect's margin and design, and the floors on it.

**One reading of every stored campaign.** A campaign stored before aspects existed is upgraded on its stored read
(:func:`~threetears.evals.schema.campaign_upgrade.upgrade_campaign_v15`, registered as the durable v15 step), so by the
time a view is taken every campaign holds its aspects, and the view reads them as stored.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from threetears.evals.analysis.numbers import format_number
from threetears.evals.kernel.campaign import EvalCampaign
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
            "When this aspect was asked its question in full (ISO-8601), which the bundle discloses where it is later "
            "than the aspect's first run (`reading_scope.question_after_runs`): the aspect's own time (an adopted "
            "aspect keeps its source's). A bundle dates it no earlier than the control's `named_at` where the control "
            "was named after the aspect's first result, since a control chosen once the evidence existed was chosen "
            "with it in view. Never a time a later edit of the design re-stamps. None when no time is recorded, "
            "disclosed as possibly set after the runs."
        )
    )
    question_declared_at: str | None = Field(
        description=(
            "When the campaign's question was worded as `question` reads (ISO-8601): later than `asked_at` when an "
            "adoption added or joined the words, which the page says were declared after the evidence where that is "
            "later than the aspect's first result, while this aspect's readings keep `asked_at`. None when unrecorded."
        )
    )
    margin_declared_at: str | None = Field(
        description=(
            "When `margin` was declared (ISO-8601): `asked_at`, or the later time a margin was added at adoption, "
            "which the page says was declared after the evidence where that is later than the aspect's first result. "
            "None with no margin or no recorded time."
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
            "is shown worse. None = no margin declared: shown better or unclear, never shown worse or no worse than "
            "the margin."
        )
    )
    floors: list[Floor] = Field(
        description=(
            "The floors on this aspect, in declaration order. Empty = none. A floor added at adoption carries its own "
            "`declared_at`, declared after the evidence where that is later than the aspect's first result; one with "
            "none was declared with the aspect's question (`asked_at`)."
        )
    )
    declared_at: str | None = Field(description="When this aspect's design was declared (ISO-8601).")
    declared_by: str = Field(description="Who declared the campaign; empty when unrecorded.")
    unread_without_control: str | None = Field(
        default=None,
        description=(
            "Why a margin or floors stored on this aspect are not read: the campaign names no control, which each is "
            "read against, so `margin` is None and `floors` empty here and the page says this sentence once. None "
            "when nothing stored was left unread."
        ),
    )


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
    return [_view(campaign, aspect) for aspect in campaign.aspects]


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


def _unread_without_control(label: str, margin: float | None, floors: Sequence[Floor]) -> str:
    """The sentence a page says of a margin and floors stored on an aspect of a campaign naming no control."""
    terms = [f"a margin of {format_number(margin)} points"] if margin is not None else []
    terms += [f"floor {floor.key}" for floor in floors]
    return (
        f"The campaign names no control, so {' and '.join(terms)} declared on {label} "
        f"{'is' if len(terms) == 1 else 'are'} read against nothing and not read; its arms are held to its bars."
    )


def _view(campaign: EvalCampaign, aspect: Aspect) -> AspectView:
    """One aspect's view over the campaign it belongs to.

    A campaign naming no control reads no margin and no floor: each is read against the control, so a stored one
    (a v15 campaign whose design declared a rule and no control) is left out of the declaration and named in
    :attr:`AspectDeclaration.unread_without_control`, never refused.
    """
    floors = [floor for floor in campaign.floors if floor.aspect == aspect.label]
    margin = aspect.margin
    unread = None
    if campaign.control is None and (margin is not None or floors):
        unread = _unread_without_control(aspect.label, margin, floors)
        margin, floors = None, []
    declared = (
        None
        if aspect.design is None
        else AspectDeclaration(
            question=campaign.question,
            asked_at=aspect.asked_at,
            question_declared_at=campaign.declared_at,
            margin_declared_at=(aspect.margin_declared_at or aspect.asked_at) if margin is not None else None,
            design=aspect.design,
            control=campaign.control,
            margin=margin,
            floors=floors,
            declared_at=aspect.declared_at,
            declared_by=campaign.declared_by,
            unread_without_control=unread,
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

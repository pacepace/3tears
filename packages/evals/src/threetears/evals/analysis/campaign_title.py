"""A campaign's title, built in code on every read and never stored: ``<arms> · <measure> (<name>)``.

The title reflects the analysis, not the conclusion: what is compared, on what, and the campaign's own name as a
qualifier. Nothing from a call, a recommendation or a verdict enters it. It heads a campaign wherever a page heads one
(the campaign listing, the analysis listing and the first line of every report page), so the rule lives here, below
both the ops that list campaigns and the renderers that print their pages.

A campaign or a frozen bundle holds its arms in one of two forms, and each has its entry: :func:`campaign_title` reads
a stored campaign (its declared arms, else its runs' candidate models), and :func:`declared_title` reads arms already
resolved (a frozen campaign bundle's: the control first, then the derived candidates), with no run to read.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.style import arm_display_name, arm_name
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignArm, printed_aspect_name

__all__ = [
    "TITLE_ARMS",
    "TITLE_ASPECTS",
    "campaign_title",
    "declared_title",
    "named_title",
    "report_input_title",
    "title_arm_name",
]

#: Past this many arms a title counts the candidates rather than naming them; past this many aspects, it counts them.
TITLE_ARMS = 3
TITLE_ASPECTS = 2


def campaign_title(
    campaign: EvalCampaign, *, candidate_models: Sequence[str] = (), variants: int = 0, judges: bool = False
) -> str:
    """What a campaign compares, in a few words: ``<arms> · <measure> (<name>)`` (:attr:`CampaignLine.title`).

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param candidate_models: The distinct candidate models of its runs, read for a campaign that declares no arm.
    :ptype candidate_models: Sequence[str]
    :param variants: How many distinct variants its runs ran, each model's recorded lever maps counted apart (a
        model and a lever map is one variant), read beside the models: more variants than models is a sweep of some
        model over another lever, titled by the model and its variant count where there is one model, else by the
        count alone. Two models each run at two settings are four variants, never two.
    :ptype variants: int
    :param judges: Whether its runs are judge runs, whose arms are judges.
    :ptype judges: bool
    :return: The title.
    :rtype: str
    """
    arms = _title_arms(campaign, candidate_models, variants=variants, judges=judges)
    return f"{arms} · {_title_measure(campaign)} ({campaign.name})"


def declared_title(
    name: str,
    arms: Sequence[CampaignArm],
    aspects: Sequence[tuple[str, str | None]],
    *,
    given: Collection[str] | None = None,
) -> str:
    """The title over arms and aspects already resolved, as a frozen campaign bundle holds them.

    The same title :func:`campaign_title` builds for a campaign declaring those arms: the arms are the bundle's
    (:attr:`~threetears.evals.analysis.campaign_bundle.CampaignBundle.arms`, the control first, then the derived
    candidates), so no run is read for a model.

    :param name: The campaign's name.
    :ptype name: str
    :param arms: The campaign's arms as a reader reads them.
    :ptype arms: Sequence[CampaignArm]
    :param aspects: Each aspect's label and its declared short name (``None`` where it declares none), in the
        campaign's order.
    :ptype aspects: Sequence[tuple[str, str | None]]
    :param given: The labels a person gave (the campaign's declared arms'); every other arm's label code computed,
        and is printed as every page prints it (:func:`~threetears.evals.analysis.style.arm_display_name`). None
        reads every label as given.
    :ptype given: Collection[str] | None
    :return: The title.
    :rtype: str
    """
    control, candidates = _declared_names(arms, given)
    return named_title(name, control=control, candidates=candidates, aspects=aspects)


def named_title(
    name: str, *, control: Sequence[str], candidates: Sequence[str], aspects: Sequence[tuple[str, str | None]]
) -> str:
    """The title over arms already named, for a page that names its arms its own way (a frozen bundle's).

    :param name: The campaign's name.
    :ptype name: str
    :param control: The control's printed name, or none where the campaign has no control.
    :ptype control: Sequence[str]
    :param candidates: Each other arm's printed name, in the page's order.
    :ptype candidates: Sequence[str]
    :param aspects: Each aspect's label and its declared short name, in the campaign's order.
    :ptype aspects: Sequence[tuple[str, str | None]]
    :return: The title.
    :rtype: str
    """
    return f"{_arms_phrase(control, candidates)} · {_measure_phrase(aspects)} ({name})"


def report_input_title(report_input: ReportInput) -> str:
    """The title over a report input, as a planned page or a document page heads the campaign: its arms by the short
    name each declares, else the name the page prints (the control first), and its aspects by the name each prints.

    :param report_input: The campaign's results as the page was built from them.
    :ptype report_input: ReportInput
    :return: The title.
    :rtype: str
    """
    arms = report_input.arms
    return named_title(
        report_input.meta.campaign,
        control=[arm.short_name or arm.name or arm.id for arm in arms if arm.role == "control"],
        candidates=[arm.short_name or arm.name or arm.id for arm in arms if arm.role != "control"],
        aspects=[(aspect.name or aspect.id, None) for aspect in report_input.aspects],
    )


def _declared_names(arms: Sequence[CampaignArm], given: Collection[str] | None = None) -> tuple[list[str], list[str]]:
    control = [title_arm_name(arm, given) for arm in arms if arm.role == "control"]
    candidates = [title_arm_name(arm, given) for arm in arms if arm.role != "control"]
    return control, candidates


def title_arm_name(arm: CampaignArm, given: Collection[str] | None = None) -> str:
    """An arm as a title names it, and as a listing's short verdict names it: its declared short name, its label as
    given, or its computed label as pages print it.

    :param arm: The arm.
    :ptype arm: CampaignArm
    :param given: The labels a person gave (:func:`declared_title`'s ``given``); None reads every label as given.
    :ptype given: Collection[str] | None
    :return: The name printed.
    :rtype: str
    """
    if arm.short_name:
        return arm.short_name
    if given is None or arm.label in given or not arm.levers:
        return arm.label
    # A label code spelled as ``axis=level`` is read as every page reads it (``careful``, never ``candidate=careful``).
    return arm_display_name(arm_name(arm.label).label, arm.levers)


def _title_arms(campaign: EvalCampaign, candidate_models: Sequence[str], *, variants: int, judges: bool) -> str:
    if campaign.arms:
        return _arms_phrase(*_declared_names(campaign.arms))
    candidates = [model.rsplit("/", 1)[-1] for model in candidate_models]
    if judges:
        return (
            f"{len(candidates)} judge{'s' if len(candidates) != 1 else ''}" if len(candidates) != 1 else candidates[0]
        )
    if variants > len(candidates):
        if len(candidates) == 1:
            return f"{candidates[0]}, {variants} variants"
        return f"{variants} variants"
    return _arms_phrase([], candidates)


def _arms_phrase(control: Sequence[str], candidates: Sequence[str]) -> str:
    named = [*control, *candidates]
    if not named:
        return "no arms"
    if len(named) <= TITLE_ARMS:
        return " vs ".join(named)
    if control:
        return f"{control[0]} vs {len(candidates)} candidates"
    return f"{len(named)} arms"


def _title_measure(campaign: EvalCampaign) -> str:
    return _measure_phrase([(aspect.label, aspect.short_name) for aspect in campaign.aspects])


def _measure_phrase(aspects: Sequence[tuple[str, str | None]]) -> str:
    names = [printed_aspect_name(label, short) for label, short in aspects]
    return ", ".join(names) if len(names) <= TITLE_ASPECTS else f"{len(names)} aspects"

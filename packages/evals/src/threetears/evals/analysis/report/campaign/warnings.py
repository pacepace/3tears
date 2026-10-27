"""What warns on a campaign's page, for every kind, under its question and answer: one plain line per arm for the bars
it misses, then one per arm for the floors it breaks, then one per arm and decision for the guardrails it is not
shown safe on. Each line says what the warning means for the arm, names the aspects it holds on, and leaves the
figures to the tables below. A bar's or a floor's line is followed at once by the writer's why where the memo
recommends that arm over it: each only warns, and this is where it warns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.campaign_readings import BarStatus, CampaignReadings, FloorStatus
from threetears.evals.kernel.numbers import format_number
from threetears.evals.analysis.report.campaign.common import as_written, named_arms
from threetears.evals.analysis.report.campaign.names import AspectNames
from threetears.evals.analysis.report.model import DisclosureBlock, Fact, Report, ReportBlock, TextBlock
from threetears.evals.analysis.style import humanize
from threetears.evals.kernel.authored import AuthoredCampaignAnalysis
from threetears.evals.kernel.campaign import EvalCampaign

__all__ = ["bar_warnings", "campaign_warnings", "floor_line", "floor_warnings", "guardrail_warnings", "listed", "where"]


def bar_warnings(
    readings: CampaignReadings, memo: AuthoredCampaignAnalysis | None, *, names: AspectNames
) -> list[ReportBlock]:
    """One line per arm missing a bar, for every kind: ``Adopting alpha overrides the 2 bars it misses on answers.``,
    or for a measurement, whose bars are its standard, ``solo falls short of the bar on clarity.``; each followed at
    once by the writer's why where the memo recommends that arm below a bar (``bars_missed``).

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :param memo: The writer's memo; None where none was written.
    :ptype memo: AuthoredCampaignAnalysis | None
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: The lines, each arm's whys after its line.
    :rtype: list[ReportBlock]
    """
    recommendation = memo.recommended() if memo is not None else None
    answers: dict[tuple[str, str], list[ReportBlock]] = {}
    if recommendation is not None:
        for missed in recommendation.bars_missed:
            if why := as_written(missed.why):
                answers.setdefault((missed.aspect, missed.measure), []).append(_why(why, recommendation.arm))
    blocks: list[ReportBlock] = []
    for arm, statuses, text in _bar_lines(readings, names):
        blocks.append(DisclosureBlock(section="warnings", source="bar", text=text))
        if recommendation is not None and arm == recommendation.arm:
            blocks += [why for status in statuses for why in answers.pop((status.aspect, status.measure), [])]
    return blocks


#: How a guardrail's line says what it means for an arm, by the guardrail's decision, the breached first.
_GUARDRAIL_WORDS: dict[str, str] = {"breached": "breaches", "undecided": "is not yet shown safe on"}


def guardrail_warnings(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report], *, names: AspectNames
) -> list[ReportBlock]:
    """One line per arm and decision naming each guardrail it is not shown safe on with the aspects that guardrail
    holds on, after the broken floors: ``v2 is not yet shown safe on the no invented total guardrail on classify and
    the polite guardrail on tone.`` Each guardrail is paired with its own aspects, never with another's. A guardrail
    only warns: no line says an arm is not adopted.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: The lines, the breached before the undecided, each arm in the order first met.
    :rtype: list[ReportBlock]
    """
    labels = readings.arm_labels()
    blocks: list[ReportBlock] = []
    for decision, words in _GUARDRAIL_WORDS.items():
        # Each arm's guardrails, each with the aspects it holds on, in the order first met.
        by_arm: dict[str, dict[str, list[str]]] = {}
        for aspect in campaign.aspects:
            bundle = reports[aspect.label].bundle
            arms_named = named_arms(bundle, labels)
            for check in bundle.guardrails.checks:
                if check.decision != decision:
                    continue
                key = check.contrast.variant_key
                arm = arms_named[key].label if key in arms_named else short_digest(key)
                aspects = by_arm.setdefault(arm, {}).setdefault(humanize("dimension", check.name), [])
                if aspect.label not in aspects:
                    aspects.append(aspect.label)
        total = len(campaign.aspects)
        for arm, by_dimension in by_arm.items():
            each = [
                f"the {dimension} guardrail on {where([names(label) for label in aspects], total)}"
                for dimension, aspects in by_dimension.items()
            ]
            blocks.append(
                DisclosureBlock(
                    section="warnings", source="guardrails", text=f"{arm} {words} {listed(each)}.", term="guardrail"
                )
            )
    return blocks


def floor_warnings(
    readings: CampaignReadings, memo: AuthoredCampaignAnalysis | None, *, names: AspectNames
) -> list[ReportBlock]:
    """One line per candidate breaking a floor, naming the aspects it breaks one on: ``Adopting v2 overrides the
    10-point floor it breaks on billing.``; each followed at once by the writer's why, where the memo recommends the
    arm over a floor it breaks. Which rate broke it, and the interval it broke on, are the aspect's floors table's.

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :param memo: The writer's memo; None where none was written.
    :ptype memo: AuthoredCampaignAnalysis | None
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: The lines, each arm's whys after its line.
    :rtype: list[ReportBlock]
    """
    recommendation = memo.recommended() if memo is not None else None
    overrides: dict[str, list[ReportBlock]] = {}
    if recommendation is not None:
        for override in recommendation.floors_overridden:
            if why := as_written(override.why):
                overrides.setdefault(override.floor, []).append(_why(why, recommendation.arm))
    blocks: list[ReportBlock] = []
    for arm, statuses, text in _floor_lines(readings, names):
        blocks.append(DisclosureBlock(section="warnings", source="floor", text=text))
        # The why answers the recommended arm's breaking of the floor, never another arm's breaking of the same one.
        if recommendation is not None and arm == recommendation.arm:
            for key in dict.fromkeys(status.floor for status in statuses):
                blocks += overrides.get(key, [])
    return blocks


def campaign_warnings(readings: CampaignReadings, names: AspectNames) -> list[str]:
    """What warns on a campaign's page of its bars and floors, as code words it: each arm missing a bar, then each
    candidate breaking a floor — never the writer's why beside them, nor the guardrail lines after.

    The page's own warnings are these lines, so whatever repeats them (an analysis's listing line) says what its page
    says, word for word.

    :param readings: What code read off the campaign, over the floors frozen with it.
    :ptype readings: CampaignReadings
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: The lines, in the page's order; none where no bar is missed and no floor broken.
    :rtype: list[str]
    """
    return [text for *_, text in _bar_lines(readings, names)] + [text for *_, text in _floor_lines(readings, names)]


def _bar_lines(readings: CampaignReadings, names: AspectNames) -> list[tuple[str, list[BarStatus], str]]:
    """Each arm missing a bar, with the bars it misses and its line."""
    total = len(readings.aspects)
    lines = []
    for arm, statuses in _by_arm(readings.missed_bars()).items():
        on = where([names(label) for label in dict.fromkeys(status.aspect for status in statuses)], total)
        form = "one" if len(statuses) == 1 else "other"
        template = (_SHORT_LINE if readings.kind == "measurement" else _MISSED_LINE)[form]
        lines.append((arm, statuses, template.format(arm=arm, count=len(statuses), where=on)))
    return lines


#: A measurement's line for an arm missing bars, whose bars are its standard, by count form.
_SHORT_LINE = {
    "one": "{arm} falls short of the bar on {where}.",
    "other": "{arm} falls short of {count} bars on {where}.",
}

#: A change's or a choice's line for an arm missing bars, by count form.
_MISSED_LINE = {
    "one": "Adopting {arm} overrides the bar it misses on {where}.",
    "other": "Adopting {arm} overrides the {count} bars it misses on {where}.",
}

#: The line for a candidate breaking floors, by count form and by whether the floors share one margin.
_FLOOR_LINE = {
    ("one", True): "Adopting {arm} overrides the {margin}-point floor it breaks on {where}.",
    ("other", True): "Adopting {arm} overrides the {margin}-point floors it breaks on {where}.",
    ("one", False): "Adopting {arm} overrides the floor it breaks on {where}.",
    ("other", False): "Adopting {arm} overrides the floors it breaks on {where}.",
}


def floor_line(arm: str, margins: Sequence[float | None], where: str) -> str:
    """The line a page warns of the floors one candidate breaks in, on a campaign's page and an aspect's own alike:
    ``Adopting v2 overrides the 10-point floor it breaks on billing.``, the floors' margin said where they share one
    (a cost floor has none). Which rate broke it, and the interval it broke on, are the floors table's.

    :param arm: The candidate, by name.
    :ptype arm: str
    :param margins: Each broken floor's margin in points, None for a cost floor; one per floor.
    :ptype margins: Sequence[float | None]
    :param where: Where the floors are broken, as :func:`where` says it.
    :ptype where: str
    :return: The line.
    :rtype: str
    """
    shared = set(margins)
    margin = next(iter(shared)) if len(shared) == 1 else None
    form = "one" if len(margins) == 1 else "other"
    return _FLOOR_LINE[(form, margin is not None)].format(
        arm=arm, margin=format_number(margin) if margin is not None else "", where=where
    )


def _floor_lines(readings: CampaignReadings, names: AspectNames) -> list[tuple[str, list[FloorStatus], str]]:
    """Each candidate breaking a floor, with the floors it breaks and its line: the floors' margin where they share
    one, as a cost floor has none."""
    total = len(readings.aspects)
    lines = []
    for arm, statuses in _by_arm(readings.broken_floors()).items():
        on = where([names(label) for label in dict.fromkeys(status.aspect for status in statuses)], total)
        lines.append((arm, statuses, floor_line(arm, [status.margin for status in statuses], on)))
    return lines


def _by_arm[Status: (BarStatus, FloorStatus)](statuses: Sequence[Status]) -> dict[str, list[Status]]:
    """``statuses`` by the arm each is of, by its label, each arm in the order first met."""
    found: dict[str, list[Status]] = {}
    for status in statuses:
        found.setdefault(status.arm_label, []).append(status)
    return found


def _why(why: str, arm: str) -> ReportBlock:
    """The writer's why for recommending ``arm`` over what the line before it warns of, as written."""
    return TextBlock(section="warnings", role="override", body=why, facts=[Fact(name="Arm", value=arm)])


def where(names: Sequence[str], total: int) -> str:
    """The aspects a line on the first screen holds on, as every warning and caveat names them: one by its name,
    ``every aspect`` where it holds on all of several, else how many of the page's and which
    (``2 of 3 aspects (billing and tone)``).

    :param names: The aspects it holds on, as the page calls them, in the page's order.
    :ptype names: Sequence[str]
    :param total: How many aspects the page holds.
    :ptype total: int
    :return: Where, as a line says it.
    :rtype: str
    """
    if len(names) == 1:
        return names[0]
    if len(names) == total:
        return "every aspect"
    return f"{len(names)} of {total} aspects ({listed(names)})"


def listed(items: Sequence[str]) -> str:
    """Items as a sentence or a cell lists them: ``billing``, ``billing and tone``, ``billing, tone and refunds``.

    :param items: The items, in order; at least one.
    :ptype items: Sequence[str]
    :return: The list.
    :rtype: str
    """
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"

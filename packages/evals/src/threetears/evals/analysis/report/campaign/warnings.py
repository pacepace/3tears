"""What warns on a campaign's page, for every kind, under its question and answer: one plain line per arm for the bars
it misses, then one per arm for the floors it breaks, then one per arm and decision for the guardrails it is not
shown safe on. Each line says what the warning means for the arm, names the aspects it holds on, and leaves the
figures to the tables below. A bar's or a floor's line is followed at once by the writer's why where the memo
recommends that arm over it: each only warns, and this is where it warns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.campaign_readings import BarStatus, CampaignReadings, FloorStatus
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.report.campaign.common import as_written, named_arms
from threetears.evals.analysis.report.campaign.names import AspectNames
from threetears.evals.analysis.report.model import DisclosureBlock, Fact, Report, ReportBlock, TextBlock
from threetears.evals.analysis.style import humanize
from threetears.evals.kernel.authored import AuthoredCampaignAnalysis
from threetears.evals.kernel.campaign import EvalCampaign

__all__ = ["bar_warnings", "campaign_warnings", "floor_warnings", "guardrail_warnings"]


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
    """One line per arm and decision naming the guardrails it is not shown safe on and the aspects they hold on, after
    the broken floors: ``v2 is not yet shown safe on the no invented total guardrail on classify.`` A guardrail only
    warns: no line says an arm is not adopted.

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
        # Each arm's guardrails, then the aspects they hold on, each in the order first met.
        by_arm: dict[str, tuple[list[str], list[str]]] = {}
        for aspect in campaign.aspects:
            bundle = reports[aspect.label].bundle
            arms_named = named_arms(bundle, labels)
            for check in bundle.guardrails.checks:
                if check.decision != decision:
                    continue
                key = check.contrast.variant_key
                arm = arms_named[key].label if key in arms_named else short_digest(key)
                dimensions, aspects = by_arm.setdefault(arm, ([], []))
                if (name := humanize("dimension", check.name)) not in dimensions:
                    dimensions.append(name)
                if aspect.label not in aspects:
                    aspects.append(aspect.label)
        for arm, (dimensions, aspects) in by_arm.items():
            several = "s" if len(dimensions) > 1 else ""
            where = _where([names(label) for label in aspects], len(campaign.aspects))
            blocks.append(
                DisclosureBlock(
                    section="warnings",
                    source="guardrails",
                    text=f"{arm} {words} the {_listed(dimensions)} guardrail{several} on {where}.",
                    term="guardrail",
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
        where = _where([names(label) for label in dict.fromkeys(status.aspect for status in statuses)], total)
        bars = "the bar" if len(statuses) == 1 else f"the {len(statuses)} bars"
        if readings.kind == "measurement":
            text = f"{arm} falls short of {bars} on {where}."
        else:
            text = f"Adopting {arm} overrides {bars} it misses on {where}."
        lines.append((arm, statuses, text))
    return lines


def _floor_lines(readings: CampaignReadings, names: AspectNames) -> list[tuple[str, list[FloorStatus], str]]:
    """Each candidate breaking a floor, with the floors it breaks and its line: the floors' margin where they share
    one, as a cost floor has none."""
    total = len(readings.aspects)
    lines = []
    for arm, statuses in _by_arm(readings.broken_floors()).items():
        where = _where([names(label) for label in dict.fromkeys(status.aspect for status in statuses)], total)
        margins = {status.margin for status in statuses}
        margin = next(iter(margins)) if len(margins) == 1 else None
        floors = "floor" if len(statuses) == 1 else "floors"
        sized = f"{format_number(margin)}-point {floors}" if margin is not None else floors
        lines.append((arm, statuses, f"Adopting {arm} overrides the {sized} it breaks on {where}."))
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


def _where(names: Sequence[str], total: int) -> str:
    """The aspects a warning holds on, as its line names them: one by its name, several by how many of the
    campaign's and which (``2 of 3 aspects (billing and tone)``)."""
    if len(names) == 1:
        return names[0]
    return f"{len(names)} of {total} aspects ({_listed(names)})"


def _listed(items: Sequence[str]) -> str:
    """Items as a sentence lists them: ``billing``, ``billing and tone``, ``billing, tone and refunds``."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"

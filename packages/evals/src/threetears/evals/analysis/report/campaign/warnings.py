"""The loud lines a campaign's page opens with, for every kind: every bar an arm misses, then every broken floor, then
every guardrail an arm is not shown safe on, one line each. A bar or a floor is followed at once by the writer's why
where the memo recommends that arm over it: each only warns, and this is where it warns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.campaign_readings import BarStatus, CampaignReadings, FloorStatus
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.report.campaign.common import as_written, floor_of, named_arms, points, rate_name
from threetears.evals.analysis.report.campaign.names import aspect_name
from threetears.evals.analysis.report.model import DisclosureBlock, Fact, Report, ReportBlock, TextBlock
from threetears.evals.analysis.style import humanize
from threetears.evals.kernel.authored import AuthoredCampaignAnalysis
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CostLower, Floor, RateDrop

__all__ = ["bar_warnings", "campaign_warnings", "floor_warnings", "guardrail_warnings"]


def bar_warnings(readings: CampaignReadings, memo: AuthoredCampaignAnalysis | None = None) -> list[ReportBlock]:
    """One loud line per bar an arm misses, the page's first, as loud as a broken floor, for every kind; each followed
    at once by the writer's why where the memo recommends that arm below that bar (``bars_missed``).

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :param memo: The writer's memo; None where none was written.
    :ptype memo: AuthoredCampaignAnalysis | None
    :return: The lines, each why after its bar's.
    :rtype: list[ReportBlock]
    """
    recommendation = memo.recommended() if memo is not None else None
    answers: dict[tuple[str, str], list[ReportBlock]] = {}
    if recommendation is not None:
        for missed in recommendation.bars_missed:
            if why := as_written(missed.why):
                answers.setdefault((missed.aspect, missed.measure), []).append(
                    TextBlock(
                        section="header", role="override", body=why, facts=[Fact(name="Arm", value=recommendation.arm)]
                    )
                )
    blocks: list[ReportBlock] = []
    for status, text in _bar_lines(readings):
        blocks.append(DisclosureBlock(section="header", source="bar", text=text))
        if recommendation is not None and status.arm_label == recommendation.arm:
            blocks += answers.pop((status.aspect, status.measure), [])
    return blocks


#: How the first screen leads a guardrail an arm is not shown safe on, by the guardrail's decision, the breached first.
_GUARDRAIL_LEADS: dict[str, str] = {"breached": "Guardrail breached", "undecided": "Not yet shown safe"}


def guardrail_warnings(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> list[ReportBlock]:
    """One loud line per aspect and decision naming each arm a guardrail leaves not shown safe, after the broken
    floors: ``Not yet shown safe on classify: v2 on no invented total.`` A guardrail only warns: no line says an arm is
    not adopted.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :return: The lines, aspect by aspect, the breached before the undecided.
    :rtype: list[ReportBlock]
    """
    labels = readings.arm_labels()
    blocks: list[ReportBlock] = []
    for aspect in campaign.aspects:
        bundle = reports[aspect.label].bundle
        names = named_arms(bundle, labels)
        for decision, lead in _GUARDRAIL_LEADS.items():
            by_arm: dict[str, list[str]] = {}
            for check in bundle.guardrails.checks:
                if check.decision != decision:
                    continue
                key = check.contrast.variant_key
                said = by_arm.setdefault(names[key].label if key in names else short_digest(key), [])
                if (name := humanize("dimension", check.name)) not in said:
                    said.append(name)
            if by_arm:
                arms = "; ".join(f"{arm} on {', '.join(said)}" for arm, said in sorted(by_arm.items()))
                blocks.append(
                    DisclosureBlock(
                        section="header",
                        source="guardrails",
                        text=f"{lead} on {aspect_name(bundle)}: {arms}.",
                        term="guardrail",
                    )
                )
    return blocks


def floor_warnings(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    memo: AuthoredCampaignAnalysis | None = None,
) -> list[ReportBlock]:
    """One loud line per floor a candidate breaks, the page's first, worded as an aspect's page words it: the aspect,
    the arm, the rate that broke it and the interval it broke on; each followed at once by the writer's why, where the
    memo recommends the arm over that floor. Past :data:`_WARNINGS_APART` of them, one line per candidate names every
    floor it breaks, so the warnings stay readable, its whys after it.

    :param campaign: The campaign, whose floors word each broken one.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param memo: The writer's memo; None where none was written.
    :ptype memo: AuthoredCampaignAnalysis | None
    :return: The lines, each why after its floor's.
    :rtype: list[ReportBlock]
    """
    recommended, overrides = _overrides(memo)

    def answered(status: FloorStatus) -> list[ReportBlock]:
        # The why answers the recommended arm's breaking of the floor, never another arm's breaking of the same one.
        return overrides.get(status.floor, []) if status.arm_label == recommended else []

    blocks: list[ReportBlock] = []
    for statuses, text in _floor_lines(campaign.floors, readings):
        blocks.append(DisclosureBlock(section="header", source="floor", text=text))
        blocks += [block for status in statuses for block in answered(status)]
    return blocks


def campaign_warnings(floors: Sequence[Floor], readings: CampaignReadings) -> list[str]:
    """The loud lines a campaign's page opens with for its bars and floors, as code words them: every bar an arm
    misses, then every floor a candidate breaks — never the writer's why beside them, nor the guardrail lines after.

    The page's own warnings are these lines, so whatever repeats them (an analysis's listing line) says what its page
    opens with, word for word.

    :param floors: The floors the readings were read against, which word each broken one: the campaign's, or the
        ones frozen with an analysis (``EvalAnalysis.design_snapshot.floors``).
    :ptype floors: Sequence[Floor]
    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :return: The lines, in the page's order; none where no bar is missed and no floor broken.
    :rtype: list[str]
    """
    return [text for _, text in _bar_lines(readings)] + [text for _, text in _floor_lines(floors, readings)]


def _bar_lines(readings: CampaignReadings) -> list[tuple[BarStatus, str]]:
    """Each bar an arm misses, with the line the page warns of it in: ``Bar missed on tone: <reason>.``"""
    return [(status, f"Bar missed on {status.aspect}: {status.reason}.") for status in readings.missed_bars()]


def _floor_lines(floors: Sequence[Floor], readings: CampaignReadings) -> list[tuple[list[FloorStatus], str]]:
    """Each line the page warns of broken floors in, with the statuses it says: one per broken floor, worded as an
    aspect's page words it; past :data:`_WARNINGS_APART` of them, one per candidate naming every floor it breaks."""
    broken = readings.broken_floors()
    if len(broken) <= _WARNINGS_APART:
        return [([status], _warning(floor_of(floors, status.floor), status)) for status in broken]
    by_arm: dict[str, list[FloorStatus]] = {}
    for status in broken:
        by_arm.setdefault(status.arm_label, []).append(status)
    return [
        (
            statuses,
            f"Floors broken for {arm}: "
            + "; ".join(f"its {rate_name(status)} on {status.aspect}" for status in statuses)
            + ".",
        )
        for arm, statuses in by_arm.items()
    ]


def _overrides(memo: AuthoredCampaignAnalysis | None) -> tuple[str | None, dict[str, list[ReportBlock]]]:
    """The arm the memo recommends, and the writer's why for each broken floor it is recommended over, by the floor's
    key, as written; no arm and nothing with no recommendation."""
    recommendation = memo.recommended() if memo is not None else None
    if recommendation is None:
        return None, {}
    found: dict[str, list[ReportBlock]] = {}
    arm = recommendation.arm
    for override in recommendation.floors_overridden:
        if why := as_written(override.why):
            found.setdefault(override.floor, []).append(
                TextBlock(section="header", role="override", body=why, facts=[Fact(name="Arm", value=arm)])
            )
    return arm, found


#: The most broken floors the page warns of one line each; past it, one line per candidate.
_WARNINGS_APART = 3


def _warning(floor: Floor | None, status: FloorStatus) -> str:
    """``Floor broken on billing: v2's pass rate falls past the 10-point margin on its whole interval, a drop of 12.3
    to 35.2 points.`` A floor reading every rate names the one that broke it. A cost floor, or one this page cannot
    place, says its reason."""
    lead = f"Floor broken on {status.aspect}: "
    if floor is None or isinstance(floor.clause, CostLower) or status.interval is None:
        return f"{lead}{status.reason}."
    clause = floor.clause
    assert isinstance(clause, RateDrop)
    low, high = status.interval
    return (
        f"{lead}{status.arm_label}'s {rate_name(status)} falls past the {format_number(clause.max_points)}-point "
        f"margin on its whole interval, a drop of {points(-high)} to {points(-low)} points."
    )

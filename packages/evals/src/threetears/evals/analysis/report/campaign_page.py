"""Build a campaign's page: the writer's summary across its aspects, the recommendation, what code read on each
aspect, the person's call, then one fold per aspect.

Code reports, the writer recommends, a person decides, and the page keeps the three apart. What the first screen
holds depends on the campaign's kind (:func:`~threetears.evals.analysis.campaign_readings.campaign_kind`). For every
kind it opens, above the title, with **every bar an arm misses**, then **every broken floor**, then **every guardrail
an arm is not shown safe on**, one loud line each, a bar or a floor followed by the writer's why where the memo
recommends that arm over it: each only warns, and this is where it warns. Then **the question**, quoted whole as the
campaign asks it, and one line for each criterion (an aspect, the control, a margin, a bar or a floor) set after the
readings it judges were visible, saying when, or that its time is not recorded: disclosed, never a demotion, since its
readings read as any others. When the runs ran is never said. Then **the writer's summary**, its read across
every aspect, for every kind that has a memo; a code-only page has none. Then **the writer's recommendation** for a change or a choice: the arm and how sure, then
each reason; where the memo recommends nothing, one line by the campaign's kind (no clear winner); with no memo, one
line saying no recommendation was written. A measurement recommends nothing, so it has no such line. Then, for
**a change**:

1. **Code's table**, one row per aspect × candidate: the cases, each arm's pass count, the change in points with the
   interval it was read on, the aspect's margin, and what the row shows (shown better, shown worse, no worse than the
   margin, or unclear). Under it, the lines that change how a row reads: the level each interval was read at, an
   aspect with no margin (never shown worse), the test setup a row was read under where the arms ran under several.
2. **What would settle each unclear row** on an aspect with a margin: the cases that would show the arm within it,
   if the arms do not differ and if they differ as observed.

For **a choice**: the table of every pair of arms on every aspect and the arm shown ahead, the writer's why for each
aspect its pick is shown behind on, then the frontier's reading of which arm clears a bar cheapest. For **a
measurement**: each aspect's measures for its one arm.

Every kind then shows its bars table and the person's call (a measurement only once one is recorded), or that none is
recorded yet; a call adopting an arm past a broken floor says which floor it overrides.

Then the memo's findings and next steps, and one closed fold per aspect: the writer's summary of the aspect (what
changed, how sure, what it means) when there is one, then the aspect's own page as
:func:`~threetears.evals.analysis.report.build.build_code_only_report` built it, which the serializers lay out in the
fold without its title, its question, its late criteria, its broken floors or its verdict line.

Every arm is named by its label, never by its role; every line is held to :mod:`~threetears.evals.analysis.style`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from threetears.evals.analysis.arms import arm_names, short_digest
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, CriterionKind, LateCriterion
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.campaign_readings import (
    AspectReading,
    BarStatus,
    CampaignReadings,
    FloorStatus,
    PairReading,
)
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.quantities import PERCENT, POINTS
from threetears.evals.analysis.report.decision import floor_statement, projected_cell
from threetears.evals.analysis.report.figures import report_rulers
from threetears.evals.analysis.report.memo_style import MemoStyle, decision_disclosures
from threetears.evals.analysis.report.model import (
    CampaignAspectFold,
    CampaignReport,
    CellValue,
    DisclosureBlock,
    Fact,
    Report,
    ReportBlock,
    TableBlock,
    TableCell,
    TableColumn,
    TextBlock,
    TextRole,
    fold_ids,
    interval_text,
    reading_anchor,
    table_row,
)
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from threetears.evals.analysis.report.words import (
    CONFIDENCE_WORDS,
    NO_RECOMMENDATION_WORDS,
    UNTRACED,
    evidence_figure,
    reading_heading,
)
from threetears.evals.analysis.style import ArmName, arm_name, code, humanize, measure_name, sentences
from threetears.evals.kernel.authored import AspectSummary, AuthoredCampaignAnalysis, Finding
from threetears.evals.kernel.campaign import EvalCampaign, FindingResolution
from threetears.evals.kernel.declaration import (
    CampaignCall,
    CostLower,
    Floor,
    RateDrop,
    aware_instant,
    resolve_control_arm,
)
from threetears.evals.kernel.errors import RefusedInput

__all__ = [
    "BAR_WORDS",
    "CHOICE_NOTE",
    "NO_CALL",
    "NO_RECOMMENDATION",
    "READING_WORDS",
    "build_campaign_report",
    "campaign_verdict",
    "campaign_warnings",
]

#: What each reading is called on the page. "Shown worse" is a drop shown past the margin, which is worse than the
#: control; the other direction says the margin, since within it the candidate may still be a little worse.
READING_WORDS: dict[str, str] = {
    "better": "shown better",
    "worse": "shown worse",
    "within_margin": "no worse than the margin",
    "unclear": "unclear",
}

#: The first screen's line where no memo was written over the campaign.
NO_RECOMMENDATION = "No recommendation was written: no analysis has been generated over this campaign."

#: The first screen's line while no person has made the call.
NO_CALL = "No call is recorded yet: a person decides, over the readings above."


def build_campaign_report(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    aspect_reports: Mapping[str, Report],
    memo: AuthoredCampaignAnalysis | None,
    *,
    call: CampaignCall | None = None,
    recommended_variant_key: str | None = None,
    resolutions: Sequence[FindingResolution] = (),
    trial_ids: Mapping[str, tuple[str, str]] | None = None,
) -> CampaignReport:
    """Lay a campaign out as its page: code's warnings first, the writer's summary and recommendation, code's readings
    on each aspect, the person's call, the findings, then one fold per aspect.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it (:func:`~threetears.evals.analysis.campaign_readings.campaign_readings`).
    :ptype readings: CampaignReadings
    :param aspect_reports: Each aspect's own page, keyed by its label: one per aspect, each built over the bundle its
        readings were read from (:func:`~threetears.evals.analysis.report.build.build_code_only_report`).
    :ptype aspect_reports: Mapping[str, Report]
    :param memo: The writer's memo over the campaign; None where none was written.
    :ptype memo: AuthoredCampaignAnalysis | None
    :param call: The person's call; None while none is recorded.
    :ptype call: CampaignCall | None
    :param recommended_variant_key: The variant key of the arm the memo recommends, as the analysis recorded it
        (``EvalAnalysis.recommended_variant_key``); required where the page holds a recommendation, which a
        measurement's never does.
    :ptype recommended_variant_key: str | None
    :param resolutions: What code filled for each of the memo's findings, by position (``EvalAnalysis.resolutions``),
        which each finding's trace reads the readings it cites off; empty traces only the trials each names.
    :ptype resolutions: Sequence[FindingResolution]
    :param trial_ids: The ids the memo names trials by, each to its aspect and its id in that aspect's bundle
        (:meth:`~threetears.evals.analysis.campaign_bundle.CampaignBundle.trial_ids`); None traces no trial.
    :ptype trial_ids: Mapping[str, tuple[str, str]] | None
    :return: The page, ``basis="analysis"`` with a memo, ``code_only`` without one.
    :rtype: CampaignReport
    :raises RefusedInput: An aspect has no page, a page is keyed by a label the campaign does not hold, the readings
        are of another campaign, or the memo recommends an arm and no key is given for it.
    """
    reports = _checked_reports(campaign, readings, aspect_reports)
    recommendation = memo.recommended() if memo is not None and readings.kind != "measurement" else None
    if recommendation is not None and recommended_variant_key is None:
        raise RefusedInput(
            f"the memo recommends {recommendation.arm}, and the page is given no variant key for that arm",
            remedy="pass the key the analysis recorded for its recommended arm (EvalAnalysis.recommended_variant_key)",
        )
    style = _CampaignStyle.over(campaign, readings, reports)
    kind = readings.kind
    blocks: list[ReportBlock] = [
        *_bar_warnings(readings, memo, style),
        *_floor_warnings(campaign, readings, memo, style),
        *_guardrail_warnings(campaign, readings, reports),
    ]
    if campaign.question.strip():
        blocks.append(
            DisclosureBlock(section="header", source="basis", text=f'"{" ".join(campaign.question.split())}"')
        )
    bundles = [reports[aspect.label].bundle for aspect in campaign.aspects]
    blocks += [DisclosureBlock(section="header", source="timing", text=line) for line in late_criteria_lines(bundles)]
    # The writer's read across every aspect, then what it recommends, then the aspects code read them on.
    blocks += _memo_summary_blocks(memo, style)
    if kind != "measurement":
        blocks += _recommendation_blocks(memo, style, kind)
    if kind == "change":
        blocks += _table_blocks(campaign, readings)
        blocks += _settle_blocks(readings)
    elif kind == "choice":
        blocks += _pair_blocks(campaign, readings, reports)
        blocks += _behind_on(memo, style)
        blocks += _frontier_blocks(campaign, readings, reports)
    else:
        blocks += _measure_blocks(campaign, readings, reports)
    blocks += _bar_blocks(readings, reports)
    if kind != "measurement" or call is not None:
        blocks += _call_blocks(campaign, readings, call)
    blocks += _memo_blocks(memo, style, _Trace.over(campaign, readings, reports, resolutions, trial_ids or {}))
    summaries = {summary.aspect: summary for summary in (memo.aspect_summaries if memo is not None else [])}
    folds = [
        CampaignAspectFold(
            aspect=aspect.label,
            shows=_shows(readings, aspect.label),
            summary=_summary_blocks(summaries.get(aspect.label), style),
            report=reports[aspect.label],
        )
        for aspect in campaign.aspects
    ]
    blocks += style.methods()
    return CampaignReport(
        basis="analysis" if memo is not None else "code_only",
        title=campaign.name or f"Campaign {campaign.id}",
        campaign_id=campaign.id,
        campaign_name=campaign.name or None,
        scope_id=campaign.scope_id,
        finding_count=len(memo.findings) if memo is not None else 0,
        blocks=blocks,
        aspects=folds,
        readings=readings,
        call=call,
        recommendation=recommendation,
        recommended_variant_key=recommended_variant_key if recommendation is not None else None,
    )


def _checked_reports(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> dict[str, Report]:
    """Every aspect's page, refused unless there is exactly one per aspect and the readings are this campaign's."""
    if readings.campaign_id != campaign.id:
        raise RefusedInput(
            f"the readings are of campaign {readings.campaign_id}, not {campaign.id}",
            remedy="pass the readings read off this campaign, over its own aspects' bundles",
        )
    labels = [aspect.label for aspect in campaign.aspects]
    if missing := [label for label in labels if label not in reports]:
        raise RefusedInput(
            f"campaign {campaign.id} has no page for aspect(s) {', '.join(missing)}",
            remedy="pass one page per aspect, each built over that aspect's bundle and keyed by its label",
        )
    if stray := sorted(set(reports) - set(labels)):
        raise RefusedInput(
            f"campaign {campaign.id} holds no aspect labelled {', '.join(stray)}",
            remedy="key each page by the label of the aspect it reads",
        )
    return {label: reports[label] for label in labels}


# =============================================================================
# The first screen
# =============================================================================


def _floor_of(floors: Sequence[Floor], key: str) -> Floor | None:
    """The floor by its key, among a campaign's floors."""
    return next((floor for floor in floors if floor.key == key), None)


def _floor_words(floor: Floor | None, status: FloorStatus) -> str:
    """A floor as the page says it, naming its arm and its aspect: ``no pass rate drops by more than 10 points on
    billing``."""
    if floor is None:
        return f"a floor on {status.aspect}"
    return f"{floor_statement(floor.clause, candidate=status.arm_label)} on {status.aspect}"


def _bar_warnings(
    readings: CampaignReadings, memo: AuthoredCampaignAnalysis | None = None, style: _CampaignStyle | None = None
) -> list[ReportBlock]:
    """One loud line per bar an arm misses, the page's first, as loud as a broken floor, for every kind; each followed
    at once by the writer's why where the memo recommends that arm below that bar (``bars_missed``)."""
    recommendation = memo.recommended() if memo is not None else None
    answers: dict[tuple[str, str], list[ReportBlock]] = {}
    if recommendation is not None and style is not None:
        for missed in recommendation.bars_missed:
            if why := style.prose(missed.why).strip():
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


def _guardrail_warnings(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> list[ReportBlock]:
    """One loud line per aspect and decision naming each arm a guardrail leaves not shown safe, after the broken
    floors: ``Not yet shown safe on classify: v2 on no invented total.`` A guardrail only warns: no line says an arm is
    not adopted."""
    labels = readings.arm_labels()
    blocks: list[ReportBlock] = []
    for aspect in campaign.aspects:
        bundle = reports[aspect.label].bundle
        names = _names(bundle, labels)
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
                        text=f"{lead} on {aspect.label}: {arms}.",
                        term="guardrail",
                    )
                )
    return blocks


def _behind_on(memo: AuthoredCampaignAnalysis | None, style: _CampaignStyle) -> list[ReportBlock]:
    """The writer's why for each aspect a choice's recommended arm is shown behind on, set beside the pairs: what the
    recommendation answers there (``behind_on``), styled."""
    recommendation = memo.recommended() if memo is not None else None
    if recommendation is None:
        return []
    return [
        TextBlock(
            section="scoreboard",
            role="override",
            body=why,
            facts=[
                Fact(name="Arm", value=recommendation.arm),
                Fact(name="Over", value=f"{behind.ahead_arm}, ahead on {behind.aspect}"),
            ],
        )
        for behind in recommendation.behind_on
        if (why := style.prose(behind.why).strip())
    ]


def _floor_warnings(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    memo: AuthoredCampaignAnalysis | None = None,
    style: _CampaignStyle | None = None,
) -> list[ReportBlock]:
    """One loud line per floor a candidate breaks, the page's first, worded as an aspect's page words it: the aspect,
    the arm, the rate that broke it and the interval it broke on; each followed at once by the writer's why, where the
    memo recommends the arm over that floor. Past :data:`_WARNINGS_APART` of them, one line per candidate names every
    floor it breaks, so the warnings stay readable, its whys after it."""
    recommended, overrides = _overrides(memo, style)

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


def late_criteria_lines(bundles: Sequence[AnalysisContextBundle]) -> list[str]:
    """The lines a page and a listing say for each criterion set after the readings it judges were visible, or whose
    time is not recorded (each aspect's ``reading_scope.late_criteria``): disclosed, never a demotion, so a reader
    can judge what the readings rest on. When the runs ran, or when the campaign was declared, is never said.

    One line per kind of criterion, criterion and date, naming the aspects it holds on (``every aspect`` where it holds
    on all of several), so each late criterion is named once and a campaign of seven adopted aspects is not told the
    same thing seven times.

    :param bundles: Each aspect's evidence, in the campaign's order; one for an aspect's own page.
    :ptype bundles: Sequence[AnalysisContextBundle]
    :return: The lines; none where every criterion came before its readings.
    :rtype: list[str]
    """
    groups: dict[tuple[CriterionKind, str, str | None], list[str]] = {}
    for bundle in bundles:
        for late in bundle.reading_scope.late_criteria:
            key = (late.criterion, _criterion_words(bundle, late), late.page_date())
            groups.setdefault(key, []).append(bundle.aspect)
    every = len(bundles) > 1
    lines = [
        _late_line(kind, what, at, labels, every=every and len(labels) == len(bundles))
        for (kind, what, at), labels in groups.items()
    ]
    untimed = [bundle for bundle in bundles if bundle.reading_scope.untimed_results]
    if untimed:
        count = sum(bundle.reading_scope.untimed_results for bundle in untimed)
        on = "" if len(untimed) == len(bundles) else f" on {_listed([bundle.aspect for bundle in untimed])}"
        plural = count != 1
        lines.append(
            f"{count} result{'s' if plural else ''}{on} carr{'y' if plural else 'ies'} no time that can be ordered, so "
            f"whether a criterion was set after {'they were' if plural else 'it was'} visible is not known."
        )
    return lines


def _criterion_words(bundle: AnalysisContextBundle, late: LateCriterion) -> str:
    """Which criterion of its kind ``late`` is, as a line names it: a bar by its measure, a guardrail by its dimension,
    a floor by what it says."""
    if late.criterion == "bar" and late.name is not None:
        described = bundle.measure_catalog.get(late.name)
        return measure_name(late.name, described.reader_name if described is not None else None)
    if late.criterion == "guardrail" and late.name is not None:
        return humanize("dimension", late.name)
    if late.criterion == "floor" and bundle.declared is not None:
        floor = next((floor for floor in bundle.declared.floors if floor.key == late.name), None)
        if floor is not None:
            return floor_statement(floor.clause, candidate="the arm")
    return late.name or ""


def _late_line(kind: CriterionKind, what: str, at: str | None, labels: Sequence[str], *, every: bool) -> str:
    """One line naming a criterion set after its readings were visible and when, or saying its time is not
    recorded."""
    on = "every aspect" if every else _listed(labels)
    several = len(labels) > 1
    were, its = ("were", "their") if several else ("was", "its")
    if kind == "control":
        set_it = "The control was named"
        after = "after every aspect's readings were visible" if every else f"after the readings on {on} were visible"
    elif kind == "aspect":
        subject = "Every aspect was" if every else f"The aspect{'s' if several else ''} {on} {were}"
        set_it = f"{subject} declared"
        after = f"after {'its' if every else its} readings were visible"
    else:
        noun = {
            "margin": "margin",
            "bar": f"{what} bar",
            "guardrail": f"{what} guardrail's margin",
            "floor": f"floor that {what}",
        }[kind]
        if several and kind in ("margin", "bar"):
            noun += "s"
        set_it = f"The {noun} on {on} {were if kind in ('margin', 'bar') else 'was'} set"
        after = f"after {its} readings were visible"
    if at is None:
        return f"When {set_it[0].lower()}{set_it[1:]} is not recorded."
    return f"{set_it} on {at}, {after}."


def _listed(labels: Sequence[str]) -> str:
    """Aspects as a sentence lists them: ``billing``, ``billing and tone``, ``billing, tone and refunds``."""
    return labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} and {labels[-1]}"


def _bar_lines(readings: CampaignReadings) -> list[tuple[BarStatus, str]]:
    """Each bar an arm misses, with the line the page warns of it in: ``Bar missed on tone: <reason>.``"""
    return [(status, f"Bar missed on {status.aspect}: {status.reason}.") for status in readings.missed_bars()]


def _floor_lines(floors: Sequence[Floor], readings: CampaignReadings) -> list[tuple[list[FloorStatus], str]]:
    """Each line the page warns of broken floors in, with the statuses it says: one per broken floor, worded as an
    aspect's page words it; past :data:`_WARNINGS_APART` of them, one per candidate naming every floor it breaks."""
    broken = readings.broken_floors()
    if len(broken) <= _WARNINGS_APART:
        return [([status], _warning(_floor_of(floors, status.floor), status)) for status in broken]
    by_arm: dict[str, list[FloorStatus]] = {}
    for status in broken:
        by_arm.setdefault(status.arm_label, []).append(status)
    return [
        (
            statuses,
            f"Floors broken for {arm}: "
            + "; ".join(f"its {_which(status)} on {status.aspect}" for status in statuses)
            + ".",
        )
        for arm, statuses in by_arm.items()
    ]


def _overrides(
    memo: AuthoredCampaignAnalysis | None, style: _CampaignStyle | None
) -> tuple[str | None, dict[str, list[ReportBlock]]]:
    """The arm the memo recommends, and the writer's why for each broken floor it is recommended over, by the floor's
    key, styled; no arm and nothing with no recommendation."""
    recommendation = memo.recommended() if memo is not None else None
    if recommendation is None or style is None:
        return None, {}
    found: dict[str, list[ReportBlock]] = {}
    arm = recommendation.arm
    for override in recommendation.floors_overridden:
        if why := style.prose(override.why).strip():
            found.setdefault(override.floor, []).append(
                TextBlock(section="header", role="override", body=why, facts=[Fact(name="Arm", value=arm)])
            )
    return arm, found


#: The most broken floors the page warns of one line each; past it, one line per candidate.
_WARNINGS_APART = 3


def _which(status: FloorStatus) -> str:
    """The rate a floor's status was decided on, as the page names it: ``pass rate``, a check's or a dimension's rate,
    or ``cost per reply``."""
    if status.floor.split("/")[1] == "cost_lower":
        return "cost per reply"
    if status.reading is None:
        return "pass rate"
    named = humanize("dimension", status.reading) if status.reading.isidentifier() else code(status.reading)
    return f"{named} rate"


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
        f"{lead}{status.arm_label}'s {_which(status)} falls past the {format_number(clause.max_points)}-point margin "
        f"on its whole interval, a drop of {_points(-high)} to {_points(-low)} points."
    )


def _points(value: float) -> str:
    """A change in points to one decimal, a whole number whole."""
    return format_number(round(value, 1))


def _memo_summary_blocks(memo: AuthoredCampaignAnalysis | None, style: _CampaignStyle) -> list[ReportBlock]:
    """The writer's read across every aspect, held to the style pass (:meth:`_CampaignStyle.summary`), the first thing
    the page says after code's warnings and the question, for every kind; nothing with no memo, or where the summary
    proposed a sample size or cost no code computed."""
    if memo is None or not (said := style.summary(memo.summary)):
        return []
    return [TextBlock(section="header", role="summary", body=said)]


def _recommendation_blocks(
    memo: AuthoredCampaignAnalysis | None, style: _CampaignStyle, kind: str
) -> list[ReportBlock]:
    """The writer's recommendation, the arm and how sure, then each reason in its words, styled; one line with no
    memo, and one by the campaign's kind where the memo recommends nothing."""
    if memo is None:
        return [DisclosureBlock(section="header", source="memo", text=NO_RECOMMENDATION)]
    recommendation = memo.recommended()
    if recommendation is None:
        return [DisclosureBlock(section="header", source="memo", text=NO_RECOMMENDATION_WORDS[kind])]
    reasons = [style.prose(reason, plans=True) for reason in recommendation.reasons]
    return [
        TextBlock(
            section="header",
            role="recommendation",
            body=recommendation.arm,
            rests_on=list(recommendation.rests_on),
            facts=[
                Fact(name="Disposition", value="recommended"),
                Fact(name="Confidence", value=CONFIDENCE_WORDS[recommendation.confidence]),
            ],
        ),
        *(
            TextBlock(section="header", role="reason", body=reason, rests_on=list(recommendation.rests_on))
            for reason in reasons
            if reason.strip()
        ),
    ]


def _shown_interval(row: AspectReading) -> tuple[tuple[float, float] | None, float | None]:
    """The interval a row's reading was read on, and its level: the gain's for shown better, the drop's for shown
    worse, the nominal level's (a floor's own) for no worse than the margin; for unclear, the wider of the gain's and
    the drop's, which reaches past zero and short of the margin as the reading says."""
    if row.reading == "better":
        return row.gain_interval, row.gain_confidence
    if row.reading == "worse":
        return row.drop_interval, row.drop_confidence
    if row.reading == "within_margin":
        return row.within_interval, INTERVAL_LEVEL
    sides = [
        (interval, level)
        for interval, level in ((row.gain_interval, row.gain_confidence), (row.drop_interval, row.drop_confidence))
        if interval is not None and level is not None
    ]
    if not sides:
        return None, None
    return max(sides, key=lambda side: side[1] or 0.0)


def _reading_words(row: AspectReading) -> str:
    """What a row shows, a lead marked as one."""
    words = READING_WORDS[row.reading]
    return words if row.confirmatory else f"{words}, a lead"


def _table_blocks(campaign: EvalCampaign, readings: CampaignReadings) -> list[ReportBlock]:
    """Code's table, one row per aspect × candidate, and the lines under it that change how a row reads."""
    if readings.control is None:
        return [
            DisclosureBlock(
                section="scoreboard",
                source="comparisons",
                text="No control was named, so no aspect is compared against one.",
            )
        ]
    control = readings.control_label
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="arm", header="Arm"),
        TableColumn(key="cases", header="Cases", term="cases"),
        TableColumn(key="control", header=f"▸ {control}", term="control"),
        TableColumn(key="passed", header="Arm passed", term="passed"),
        TableColumn(key="change", header="Change (pts)", term="change"),
        TableColumn(key="interval", header="Interval (pts)", term="interval"),
        TableColumn(key="margin", header="Margin (pts)", term="margin"),
        TableColumn(key="reading", header="Reading"),
    ]
    rows: list[dict[str, TableCell]] = []
    levels: list[float] = []
    for row in readings.readings:
        interval, level = _shown_interval(row)
        if level is not None and row.reading != "within_margin":
            levels.append(level)
        change = round(row.change, 1)
        values: dict[str, CellValue] = {
            "aspect": row.aspect,
            "arm": row.arm_label,
            "cases": min(row.control_cases, row.arm_cases),
            "control": f"{row.control_passed} of {row.control_trials}",
            "passed": f"{row.arm_passed} of {row.arm_trials}",
            "change": change,
            "interval": None if interval is None else interval_text(interval[0], interval[1], POINTS),
            "margin": row.margin,
            "reading": _reading_words(row),
        }
        cells = table_row(values)
        if interval is not None and min(row.control_cases, row.arm_cases) >= 2:
            cells["change"] = TableCell(
                value=change,
                low=interval[0],
                high=interval[1],
                margin=row.margin,
                zero_line=True,
            )
        rows.append(cells)
    blocks: list[ReportBlock] = [
        TableBlock(
            section="scoreboard",
            name="aspects",
            title="By aspect",
            columns=columns,
            rows=rows,
            order="aspect by aspect, each arm against the control",
            total_rows=len(rows),
        )
    ]
    notes = _level_lines(levels, readings) if readings.readings else []
    notes += _within_lines(readings)
    notes += [
        _rig_line(row.aspect, f"{row.arm_label}'s change against {control}", row.rig)
        for row in readings.readings
        if row.rig is not None
    ]
    notes += _floor_beside_lines(campaign, readings)
    notes += _unread_lines(campaign, readings)
    notes += _no_question_lines(campaign)
    blocks += [DisclosureBlock(section="scoreboard", source="comparisons", text=note) for note in notes if note]
    return blocks


def _rig_line(aspect: str, what: str, rig: str) -> str:
    """Which test setup a row's change and interval were read under, where the arms ran under several."""
    return (
        f"On {aspect}, {what} and its interval are read under test setup {short_digest(rig)}, the first both ran "
        "under; the pass counts are over every setup."
    )


def _percent(level: float) -> str:
    """A level as a percent, to the precision that tells two levels apart."""
    return format_number(round(100 * level, 2)) + "%"


def _level_lines(levels: Sequence[float], readings: CampaignReadings) -> list[str]:
    """The level the table's corrected intervals were read at (``levels``: every row's but one no worse than the
    margin, whose line is :func:`_within_lines`; none where every row is), how often a reading can be wrong, and,
    with several arms, that nothing corrects across them."""
    lines: list[str] = []
    if levels:
        low, high = min(levels), max(levels)
        if len(levels) == 1 and round(low, 4) == round(INTERVAL_LEVEL, 4):
            lines.append(f"The interval is read at {_percent(low)}, as a floor's is.")
        elif len(levels) == 1:
            lines.append(f"The interval is read at {_percent(low)}, corrected across each arm's aspects.")
        elif round(low, 4) == round(high, 4):
            lines.append(f"Each interval is read at {_percent(low)}, corrected across each arm's aspects.")
        else:
            lines.append(
                f"Each interval is read at the level its row was tested at, from {_percent(low)} to {_percent(high)}, "
                "corrected across each arm's aspects; a reading never contradicts its interval."
            )
    half = _percent(readings.alpha / 2)
    lines.append(f"Each arm's gains and its drops are each shown wrongly at most {half} of the time.")
    if len({row.arm for row in readings.readings}) > 1:
        lines.append(
            "Nothing corrects across arms, so with several arms one is shown better or worse by chance more often."
        )
    return lines


def _within_lines(readings: CampaignReadings) -> list[str]:
    """Where a row reads no worse than the margin, the level it was read at and why that is not corrected; where
    such a row's interval clears zero, that its gain is still read corrected, so that is not a gain shown."""
    within = [row for row in readings.readings if row.reading == "within_margin"]
    if not within:
        return []
    lines = [
        f"A row no worse than the margin reads its interval at {_percent(INTERVAL_LEVEL)} on its own, as a floor "
        "does: a candidate is no worse only where it holds on every asked aspect, which needs no correction."
    ]
    if any(row.within_interval is not None and row.within_interval[0] > 0 for row in within):
        lines.append(
            "Its gain is still read corrected across the arm's aspects, so such an interval clearing zero is not a "
            "gain shown."
        )
    return lines


def _floor_beside_lines(campaign: EvalCampaign, readings: CampaignReadings) -> list[str]:
    """A broken floor beside a row that is not shown worse, and a held floor beside a row left unclear: the two read
    different intervals, or different points, and say which."""
    lines: list[str] = []
    for status in readings.broken_floors():
        row = next((row for row in readings.on_aspect(status.aspect) if row.arm == status.arm), None)
        if row is None or row.reading == "worse":
            continue
        lines.append(
            f"The floor broken on {status.aspect} for {status.arm_label} reads its {_which(status)} at "
            f"{_percent(INTERVAL_LEVEL)} on its own, while its row reads only the pass rate, corrected across "
            f"{status.arm_label}'s aspects."
        )
    for status in readings.floors:
        row = next((row for row in readings.on_aspect(status.aspect) if row.arm == status.arm), None)
        floor = _floor_of(campaign.floors, status.floor)
        if row is None or row.margin is None or floor is None or not isinstance(floor.clause, RateDrop):
            continue
        # A rate floor and its row read one interval at one level, so they part only on which rates they read (a
        # floor reads every rate its clause names, a row the pass rate alone) or on the points they read it against.
        if status.status == "held" and row.reading == "unclear":
            state = "held"
        elif status.status == "unclear" and row.reading == "within_margin":
            state = "left unclear"
        else:
            continue
        lead = f"The floor {state} on {status.aspect} for {status.arm_label} reads its {_which(status)}"
        if floor.clause.max_points != row.margin:
            lines.append(
                f"{lead} against {format_number(floor.clause.max_points)} points, while its row reads the pass rate "
                f"against the {format_number(row.margin)}-point margin."
            )
        else:
            lines.append(f"{lead}, while its row reads only the pass rate, both at {_percent(INTERVAL_LEVEL)}.")
    return lines


def _unread_lines(campaign: EvalCampaign, readings: CampaignReadings) -> list[str]:
    """An aspect no arm was compared on, and an aspect with no margin, which is never shown worse."""
    lines: list[str] = []
    for aspect in campaign.aspects:
        rows = readings.on_aspect(aspect.label)
        if not rows:
            lines.append(f"No arm was compared on {aspect.label}: the control or every other arm did not run there.")
        elif aspect.margin is None:
            lines.append(f"No margin was declared on {aspect.label}, so no arm is ever shown worse there.")
    return lines


def _no_question_lines(campaign: EvalCampaign) -> list[str]:
    """The line under code's table where the campaign asks no question: every reading is a lead."""
    return [] if campaign.question.strip() else ["No question was declared, so every reading is a lead."]


def _settle_blocks(readings: CampaignReadings) -> list[ReportBlock]:
    """Each unclear row: why it is unclear, and the cases that would show the arm no worse than the control by more
    than the aspect's margin, if the arms do not differ and if they differ as observed. The count is read at the
    nominal level, as a floor's is, so where a floor reads the same rate against the same points the two counts are
    one. They would not show it better: the row stays unclear unless its gain clears too, and the note says so. A row
    on an aspect that declared no margin has nothing to show it within, and the table above already says no margin
    was declared there, so it is left out; a row already no worse than the margin has nothing left to show.
    """
    unclear = [row for row in readings.readings if row.reading == "unclear" and row.margin is not None]
    if not unclear:
        return []
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="arm", header="Arm"),
        TableColumn(key="why", header="Why unclear"),
        TableColumn(key="none", header="If no difference"),
        TableColumn(key="observed", header="If as observed"),
    ]
    rows = [
        table_row(
            {
                "aspect": row.aspect,
                "arm": row.arm_label,
                "why": row.reason or None,
                "none": None if row.projection is None else projected_cell(row.projection, row.projection.if_none),
                "observed": None
                if row.projection is None
                else projected_cell(row.projection, row.projection.if_observed),
            }
        )
        for row in unclear
    ]
    return [
        TableBlock(
            section="scoreboard",
            name="to_settle",
            title="To show no worse than the margin",
            columns=columns,
            rows=rows,
            order="as the table above",
            total_rows=len(rows),
        ),
        DisclosureBlock(
            section="scoreboard",
            source="comparisons",
            text=(
                "A count is the independent cases that would show the arm no worse than the control by more than the "
                f"aspect's margin, read at {_percent(INTERVAL_LEVEL)} as a floor's count is, so a floor on the same "
                "rate and points gives the same count. It would not show the arm better, so its row stays unclear "
                "unless its gain clears too. Each count reads the interval again over this run's cases repeated to "
                "that many; repeats of a case narrow it far less than new cases."
            ),
        ),
    ]


def _call_blocks(campaign: EvalCampaign, readings: CampaignReadings, call: CampaignCall | None) -> list[ReportBlock]:
    """The person's call, who made it and when, their reason quoted, and every floor and bar it overrides; a line
    saying none is recorded while none is.

    Every override the call stored is said, as stored, whatever the readings on this page now say of it: one that no
    longer warns is marked so, never dropped. Beside them, a floor broken or a bar missed here that the call does not
    override, and in a choice each pair where another arm is shown ahead of the decided one. The decided arm is read
    by every key it resolves to (:func:`~threetears.evals.kernel.declaration.resolve_control_arm`), as recording read
    it, so a control stored under an older identity predicate is still found."""
    if call is None:
        return [DisclosureBlock(section="header", source="call", text=NO_CALL)]
    named = next((arm for arm in campaign.arms if arm.variant_key == call.variant_key), None)
    keys = {call.variant_key, *(resolve_control_arm(named).variant_keys if named is not None else [])}
    verb = "keep" if readings.control in keys else "adopt"
    head = f"Called: {verb} {call.arm}, by {call.called_by} on {_date(call.called_at)}"
    if call.run_ids and (added := [run for run in campaign.run_ids if run not in set(call.run_ids)]):
        over = len(call.run_ids)
        head += f", over {over} run{'' if over == 1 else 's'}; {len(added)} added since"
    lines = [f"{head}."]
    if call.reason.strip():
        lines.append(f'Reason: "{" ".join(call.reason.split())}"')
    floors = [status for status in readings.floors if status.arm in keys]
    for key in call.floors_overridden:
        broken = next((status for status in floors if status.floor == key and status.status == "broken"), None)
        if broken is not None:
            lines.append(f"The call overrides a broken floor: {_floor_words(_floor_of(campaign.floors, key), broken)}.")
        else:
            lines.append(f"The call overrides the floor {code(key)} (no longer broken).")
    unanswered: dict[str, FloorStatus] = {}
    for status in floors:
        if status.status == "broken" and status.floor not in call.floors_overridden:
            unanswered.setdefault(status.floor, status)
    for key, status in unanswered.items():
        said = _floor_words(_floor_of(campaign.floors, key), status)
        lines.append(f"The call does not name the broken floor as overridden: {said}.")
    missed = {(bar.aspect, bar.measure): bar for bar in readings.missed_bars() if bar.arm in keys}
    overridden = {(bar.aspect, bar.measure) for bar in call.bars_overridden}
    for bar in call.bars_overridden:
        if (still := missed.get((bar.aspect, bar.measure))) is not None:
            lines.append(f"The call overrides a missed bar on {bar.aspect}: {still.reason}.")
        else:
            lines.append(f"The call overrides the bar on {code(bar.measure)} on {bar.aspect} (no longer missed).")
    for (aspect, measure), unnamed in missed.items():
        if (aspect, measure) not in overridden:
            lines.append(f"The call does not name the missed bar on {aspect} as overridden: {unnamed.reason}.")
    for pair in readings.pairs:
        if pair.ahead is not None and pair.ahead not in keys and keys & {pair.first, pair.second}:
            lines.append(f"{pair.ahead_label or pair.ahead} is shown ahead of {call.arm} on {pair.aspect}.")
    return [DisclosureBlock(section="header", source="call", text=line) for line in lines]


def _date(stamp: str) -> str:
    """An ISO-8601 instant as the page dates it (``14 Oct 2026``), or as given when it does not parse."""
    try:
        moment = aware_instant(stamp, "the call")
    except ValueError:
        # NOSILENT: an unparseable stamp is printed as stored rather than dropped or reformatted
        return stamp
    return f"{moment.day} {moment:%b %Y}"


# =============================================================================
# The memo and the folds
# =============================================================================


def _memo_blocks(memo: AuthoredCampaignAnalysis | None, style: _CampaignStyle, trace: _Trace) -> list[ReportBlock]:
    """The memo's findings, each its headline and body and, folded, its trace; then its next steps, every word through
    the style pass; nothing with no memo."""
    if memo is None:
        return []
    blocks: list[ReportBlock] = []
    for position, finding in enumerate(memo.findings):
        title = style.title(finding.title)
        blocks.append(
            TextBlock(section="memo", role="finding_title", body=title, finding=position, anchor=f"F{position + 1}")
        )
        if body := style.prose(finding.body):
            blocks.append(TextBlock(section="memo", role="finding_body", body=body, finding=position))
        blocks += trace.blocks(position, finding)
    for step in memo.next:
        if title := style.sentence(step.title, plans=True):
            blocks.append(TextBlock(section="memo", role="next_step", body=title))
    return blocks


@dataclass(frozen=True)
class _Trace:
    """What each finding cites, with the figure code filled for it, linked to the aspect's fold that prints it.

    A fold is closed, and a link landing inside a closed ``<details>`` is lost where a browser does not open it, so a
    trace lands on the fold's heading (``aspect-<fold id>``), outside it, and states the figure itself: the reader
    sees the number without a click, and the link says which fold holds it.
    """

    campaign: EvalCampaign
    labels: Mapping[str, str]
    reports: Mapping[str, Report]
    resolutions: Sequence[FindingResolution]
    trial_ids: Mapping[str, tuple[str, str]]

    @classmethod
    def over(
        cls,
        campaign: EvalCampaign,
        readings: CampaignReadings,
        reports: Mapping[str, Report],
        resolutions: Sequence[FindingResolution],
        trial_ids: Mapping[str, tuple[str, str]],
    ) -> _Trace:
        """The trace over the campaign's folds, each arm named by its campaign label as the folds name it."""
        return cls(
            campaign=campaign,
            labels=readings.arm_labels(),
            reports=reports,
            resolutions=resolutions,
            trial_ids=trial_ids,
        )

    def blocks(self, position: int, finding: Finding) -> list[ReportBlock]:
        """One finding's trace, in its ``Evidence`` fold: a row per reading it cites and per trial it names, each
        figure linked to the fold that prints it; where nothing links, the line saying so (:data:`UNTRACED`)."""
        rows = [*self._readings(position), *(self._trial(trial) for trial in finding.trials)]
        blocks: list[ReportBlock] = []
        if rows:
            several = len(self.campaign.aspects) > 1
            columns = [
                *([TableColumn(key="aspect", header="Aspect", term="aspect")] if several else []),
                TableColumn(key="arm", header="Arm"),
                TableColumn(key="reading", header="Reading"),
                TableColumn(key="value", header="Value"),
            ]
            blocks.append(
                TableBlock(
                    section="memo",
                    finding=position,
                    name="trace",
                    title="Cites",
                    columns=columns,
                    rows=[{column.key: row[column.key] for column in columns} for row in rows],
                    order="each reading as the finding cites it, then each trial",
                    total_rows=len(rows),
                    fold="Evidence",
                )
            )
        if not any(cell.ref is not None for row in rows for cell in row.values()):
            blocks.append(
                DisclosureBlock(section="memo", source="memo", text=UNTRACED, finding=position, fold="Evidence")
            )
        return blocks

    def _readings(self, position: int) -> list[dict[str, TableCell]]:
        """A row per reading the finding cites, read on the aspect it was resolved on, its figure linked to the fold
        where a cell there prints the same figure."""
        if position >= len(self.resolutions):
            return []
        rows = []
        for row in self.resolutions[position].evidence:
            report = self.reports[row.aspect]
            surface = bundle_decision_surface(report.bundle)
            ruler = report_rulers(surface, report.bundle.measure_catalog).of(row.measure_id, row.reading)
            place = reading_anchor(row.cell_ref, row.measure_id, row.reading)
            printed = any(
                cell.anchor == place
                for block in report.blocks
                if isinstance(block, TableBlock)
                for line in block.rows
                for cell in line.values()
            )
            figure = evidence_figure(row, ruler)
            rows.append(
                {
                    "aspect": TableCell(value=row.aspect),
                    "arm": TableCell(value=self._arm(row.aspect, variant_of_cell_ref(row.cell_ref) or row.cell_ref)),
                    "reading": TableCell(value=ruler.scaled(reading_heading(surface, row.measure_id, row.reading))),
                    "value": figure.model_copy(update={"ref": self._fold(row.aspect) if printed else None}),
                }
            )
        return rows

    def _trial(self, trial: str) -> dict[str, TableCell]:
        """A trial the finding names by its campaign id, shown by its id in its aspect's fold, where the reader finds
        its record, and linked to that fold; an id the campaign does not hold, listed as written and linked nowhere."""
        if (where := self.trial_ids.get(trial)) is None:
            return {
                "aspect": TableCell(value=None),
                "arm": TableCell(value=None),
                "reading": TableCell(value="trial"),
                "value": TableCell(value=trial),
            }
        aspect, own = where
        arm = next((held.arm for held in self.reports[aspect].bundle.trials if held.trial_id == own), None)
        return {
            "aspect": TableCell(value=aspect),
            "arm": TableCell(value=self._arm(aspect, arm) if arm is not None else None),
            "reading": TableCell(value="trial"),
            "value": TableCell(value=own, ref=self._fold(aspect) if arm is not None else None),
        }

    def _arm(self, aspect: str, variant_key: str) -> str:
        """An arm as the aspect's fold names it: by the campaign's label, else its variant's name."""
        names = _names(self.reports[aspect].bundle, self.labels)
        return names[variant_key].label if variant_key in names else f"unplaced arm {short_digest(variant_key)}"

    def _fold(self, aspect: str) -> str:
        """The id of the section around ``aspect``'s fold, outside its closed ``<details>``."""
        labels = [held.label for held in self.campaign.aspects]
        return f"aspect-{fold_ids(labels)[labels.index(aspect)]}"


def _summary_blocks(summary: AspectSummary | None, style: _CampaignStyle) -> list[ReportBlock]:
    """The writer's summary of one aspect: what changed, how sure, what it means, styled; nothing with none."""
    if summary is None:
        return []
    parts: tuple[tuple[TextRole, str], ...] = (
        ("what_changed", style.prose(summary.what_changed)),
        ("how_sure", style.prose(summary.how_sure)),
        ("means", style.prose(summary.means, plans=True)),
    )
    return [TextBlock(section="memo", role=role, body=body) for role, body in parts if body.strip()]


def _shows(readings: CampaignReadings, aspect: str) -> str:
    """What code read on one aspect, beside its fold's heading, by the campaign's kind: ``v2 shown worse``, ``alpha
    ahead of beta``, ``alpha misses a bar``."""
    if readings.kind == "choice":
        pairs = [pair for pair in readings.pairs if pair.aspect == aspect]
        if not pairs:
            return "not compared"
        return ", ".join(_pair_words(pair) for pair in pairs)
    if readings.kind == "measurement":
        missed = [status.arm_label for status in readings.missed_bars() if status.aspect == aspect]
        return f"{', '.join(dict.fromkeys(missed))} misses a bar" if missed else "measured"
    rows = readings.on_aspect(aspect)
    if not rows:
        return "not compared"
    return ", ".join(f"{row.arm_label} {_reading_words(row)}" for row in rows)


def campaign_verdict(readings: CampaignReadings, aspects: Sequence[str]) -> str:
    """The verdict line of a campaign of several aspects, as code writes it: what code read on each aspect, as the
    campaign's page says it beside that aspect's fold, in the campaign's order — ``billing: v2 shown worse; tone: v2
    shown better``. Never the writer's words.

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :param aspects: The campaign's aspects' labels, in its order.
    :ptype aspects: Sequence[str]
    :return: The verdict line.
    :rtype: str
    """
    return "; ".join(f"{aspect}: {_shows(readings, aspect)}" for aspect in aspects)


def _pair_words(pair: PairReading) -> str:
    """A pair as a fold's heading says it: ``alpha ahead of beta``, ``alpha and beta not separated``."""
    if pair.ahead is None:
        said = f"{pair.first_label} and {pair.second_label} not separated"
    else:
        ahead, behind = (
            (pair.first_label, pair.second_label) if pair.ahead == pair.first else (pair.second_label, pair.first_label)
        )
        said = f"{ahead} ahead of {behind}"
    return said if pair.confirmatory else f"{said}, a lead"


# =============================================================================
# A choice, a measurement and the bars
# =============================================================================

#: What a choice's notes say once: the one family, and that ahead is a reading.
CHOICE_NOTE = (
    "A pair is read both ways at once, so every pair on every aspect is one family, corrected together. An arm is "
    "ahead only where the whole interval is one side of zero, and ahead is a reading, not a pick."
)


def _pair_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> list[ReportBlock]:
    """A choice's table, one row per pair per aspect: each arm's pass count, the difference with its interval, and
    the arm shown ahead or no separation; then the lines that change how a row reads."""
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="first", header="First arm"),
        TableColumn(key="second", header="Second arm"),
        TableColumn(key="cases", header="Cases", term="cases"),
        TableColumn(key="first_passed", header="First passed", term="passed"),
        TableColumn(key="second_passed", header="Second passed", term="passed"),
        TableColumn(key="change", header="Change (pts)", term="change"),
        TableColumn(key="interval", header="Interval (pts)", term="interval"),
        TableColumn(key="ahead", header="Ahead"),
    ]
    rows: list[dict[str, TableCell]] = []
    levels: list[float] = []
    for pair in readings.pairs:
        if pair.confidence is not None:
            levels.append(pair.confidence)
        change = round(pair.change, 1)
        ahead = pair.ahead_label or "no separation"
        cells = table_row(
            {
                "aspect": pair.aspect,
                "first": pair.first_label,
                "second": pair.second_label,
                "cases": min(pair.first_cases, pair.second_cases),
                "first_passed": f"{pair.first_passed} of {pair.first_trials}",
                "second_passed": f"{pair.second_passed} of {pair.second_trials}",
                "change": change,
                "interval": None if pair.interval is None else interval_text(*pair.interval, POINTS),
                "ahead": ahead if pair.confirmatory else f"{ahead}, a lead",
            }
        )
        if pair.interval is not None:
            cells["change"] = TableCell(value=change, low=pair.interval[0], high=pair.interval[1], zero_line=True)
        rows.append(cells)
    blocks: list[ReportBlock] = [
        TableBlock(
            section="scoreboard",
            name="pairs",
            title="By aspect, pair by pair",
            columns=columns,
            rows=rows,
            order="aspect by aspect, each pair of arms in the campaign's order",
            total_rows=len(rows),
        )
    ]
    notes = [CHOICE_NOTE]
    if levels:
        low, high = min(levels), max(levels)
        notes.append(
            f"Each interval is read at {_percent(low)}."
            if round(low, 4) == round(high, 4)
            else f"Each interval is read at the level its pair was tested at, from {_percent(low)} to {_percent(high)}."
        )
    notes += [
        _rig_line(pair.aspect, f"the difference between {pair.first_label} and {pair.second_label}", pair.rig)
        for pair in readings.pairs
        if pair.rig is not None
    ]
    notes += _no_question_lines(campaign)
    notes += _unread_without_control_lines(campaign, reports)
    blocks += [DisclosureBlock(section="scoreboard", source="comparisons", text=note) for note in notes if note]
    return blocks


def _frontier_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> list[ReportBlock]:
    """Where an aspect's frontier read which arm clears its bar cheapest, one line saying what it read: the arm shown
    cheaper than every other that cleared, the only arm that cleared with a cost, or the arms that cleared with none
    shown cheaper than the rest. A reading of the bundle's frontier, never a pick."""
    labels = readings.arm_labels()
    lines: list[ReportBlock] = []
    for aspect in campaign.aspects:
        for subject in reports[aspect.label].bundle.frontier.subjects:
            verdict = subject.verdict
            if verdict is None:
                continue
            named = labels.get(verdict.variant_key, verdict.model)
            tied = [labels.get(tie.variant_key, tie.model) for tie in verdict.tied_with]
            lines.append(
                DisclosureBlock(
                    section="scoreboard",
                    source="comparisons",
                    text=_frontier_words(aspect.label, named, tied, verdict.cost_decision),
                )
            )
    return lines


def _frontier_words(aspect: str, named: str, tied: Sequence[str], decision: str | None) -> str:
    """What one aspect's frontier verdict reads, by how its cost was decided."""
    if tied:
        arms = ", ".join([named, *tied[:-1]]) + f" and {tied[-1]}"
        why = (
            "too few priced cases to show any of them cheaper"
            if decision == "untested"
            else "none is shown cheaper than the rest"
        )
        return f"Arms shown to clear the bar on {aspect}: {arms}; {why}."
    if decision == "only_cleared":
        return f"Only arm shown to clear the bar on {aspect} at a known cost: {named}."
    if decision == "shown_cheapest":
        return f"Cheapest arm shown to clear the bar on {aspect}: {named}, shown cheaper than every other that cleared."
    return f"Arm shown to clear the bar on {aspect} at the lowest observed cost: {named}."


def _measure_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> list[ReportBlock]:
    """A measurement's table: each aspect's measures for its arm, each value with its interval, the pass rate as its
    count; a Setup column where an arm was measured under more than one test setup, each cell's apart."""
    rigs: dict[str, set[str]] = {}
    for reading in readings.measures:
        if reading.rig is not None:
            rigs.setdefault(reading.arm, set()).add(reading.rig)
    by_rig = any(len(held) > 1 for held in rigs.values())
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="arm", header="Arm"),
        *([TableColumn(key="setup", header="Setup")] if by_rig else []),
        TableColumn(key="measure", header="Measure", term="reading"),
        TableColumn(key="value", header="Value"),
        TableColumn(key="interval", header="Interval", term="interval"),
        TableColumn(key="cases", header="Cases", term="cases"),
    ]
    rows = []
    for reading in readings.measures:
        values: dict[str, CellValue] = {
            "aspect": reading.aspect,
            "arm": reading.arm_label,
            "measure": reading.name,
            "value": _measured(reading.value, reading.passed, reading.trials),
            "interval": _measured_interval(reading.interval, reading.passed is not None),
            "cases": reading.cases,
        }
        if by_rig:
            values["setup"] = "every setup" if reading.rig is None else short_digest(reading.rig)
        rows.append(table_row(values))
    blocks: list[ReportBlock] = [
        TableBlock(
            section="scoreboard",
            name="measures",
            title="By aspect, measure by measure",
            columns=columns,
            rows=rows,
            order="aspect by aspect, each measure as its cells summarise it",
            total_rows=len(rows),
        ),
        DisclosureBlock(
            section="scoreboard",
            source="comparisons",
            text="One arm is measured, so nothing is compared and no arm is recommended; its bars are its standard.",
        ),
    ]
    notes = _no_question_lines(campaign) + _unread_without_control_lines(campaign, reports)
    notes += _unread_until_candidate_lines(campaign)
    blocks += [DisclosureBlock(section="scoreboard", source="comparisons", text=note) for note in notes]
    return blocks


def _measured(value: float | None, passed: int | None, trials: int | None) -> CellValue:
    """A measure's value as the table states it: a pass rate as its count, any other value as measured."""
    if passed is not None and trials is not None:
        return f"{passed} of {trials}"
    return None if value is None else round(value, 4)


def _measured_interval(interval: tuple[float, float] | None, rate: bool) -> str | None:
    """A measure's interval as the table states it: a pass rate's in percent, any other in the measure's units."""
    if interval is None:
        return None
    if rate:
        return f"[{_points(100 * interval[0])}{PERCENT}, {_points(100 * interval[1])}{PERCENT}]"
    return f"[{format_number(round(interval[0], 4))}, {format_number(round(interval[1], 4))}]"


def _unread_until_candidate_lines(campaign: EvalCampaign) -> list[str]:
    """A change whose control alone has run reads as a measurement, so its margins and floors wait for a candidate;
    one line saying which, so they are not silently dropped."""
    control = campaign.control
    margined = [aspect.label for aspect in campaign.aspects if aspect.margin is not None]
    if control is None or not (margined or campaign.floors):
        return []
    waiting = []
    if margined:
        waiting.append(f"the margin{'s' if len(margined) > 1 else ''} on {', '.join(margined)}")
    if campaign.floors:
        waiting.append(f"{len(campaign.floors)} floor{'s' if len(campaign.floors) > 1 else ''}")
    plural = len(waiting) > 1 or len(margined) > 1 or len(campaign.floors) > 1
    return [
        f"Only the control, {control.label}, has run, so {' and '.join(waiting)} {'are' if plural else 'is'} read "
        "once a candidate runs."
    ]


#: What each bar status is called on the page.
BAR_WORDS: dict[str, str] = {"met": "met", "missed": "missed", "unclear": "unclear"}


def _bar_blocks(readings: CampaignReadings, reports: Mapping[str, Report]) -> list[ReportBlock]:
    """Every arm against every absolute bar, aspect by aspect, for every kind, each measure by its reader's name;
    nothing where no bar is declared."""
    if not readings.bars:
        return []

    def named(aspect: str, measure: str) -> str:
        descriptor = reports[aspect].bundle.measure_catalog.get(measure)
        return measure_name(measure, descriptor.reader_name if descriptor is not None else None)

    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="arm", header="Arm"),
        TableColumn(key="measure", header="Measure", term="reading"),
        TableColumn(key="bar", header="Bar", term="bar"),
        TableColumn(key="value", header="Value"),
        TableColumn(key="interval", header="Interval", term="interval"),
        TableColumn(key="status", header="Status"),
    ]
    rows = [
        table_row(
            {
                "aspect": status.aspect,
                "arm": status.arm_label,
                "measure": named(status.aspect, status.measure),
                "bar": f"{'at most' if status.direction == 'lower_is_better' else 'at least'} {format_number(status.bar)}",
                "value": status.value,
                "interval": None
                if status.interval is None
                else f"[{format_number(status.interval[0])}, {format_number(status.interval[1])}]",
                "status": BAR_WORDS[status.status],
            }
        )
        for status in readings.bars
    ]
    return [
        TableBlock(
            section="scoreboard",
            name="bars",
            title="Bars",
            columns=columns,
            rows=rows,
            order="aspect by aspect, each bar, each arm",
            total_rows=len(rows),
        )
    ]


def _unread_without_control_lines(campaign: EvalCampaign, reports: Mapping[str, Report]) -> list[str]:
    """What an aspect stored that needs a control and was not read, as its declaration says it, once per aspect."""
    lines: list[str] = []
    for aspect in campaign.aspects:
        declared = reports[aspect.label].bundle.declared
        if declared is not None and declared.unread_without_control:
            lines.append(declared.unread_without_control)
    return lines


# =============================================================================
# The style pass over the memo
# =============================================================================


@dataclass
class _CampaignStyle:
    """The memo's words held to the style an aspect's page holds them to (:class:`MemoStyle`), over every aspect.

    The first aspect's pass fixes and lints each sentence and records what it dropped; every aspect's names the arms
    and knows the numbers its bundle measured, so a sentence restating a number any aspect measured is dropped, and a
    sentence on what to run next is dropped only where no aspect's projections state the count it proposes.
    """

    styles: list[MemoStyle]

    @classmethod
    def over(cls, campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]) -> _CampaignStyle:
        """One pass per aspect, every arm named by its campaign label, every unclear row's projection known."""
        labels = readings.arm_labels()
        samples = tuple(row.projection for row in readings.readings if row.projection is not None)
        question = " ".join(campaign.question.split()) or None
        return cls(
            styles=[
                MemoStyle(
                    bundle=reports[aspect.label].bundle,
                    names=_names(reports[aspect.label].bundle, labels),
                    question=question,
                    samples=samples,
                )
                for aspect in campaign.aspects
            ]
        )

    @property
    def primary(self) -> MemoStyle:
        return self.styles[0]

    def _named(self, text: str) -> str:
        for style in self.styles:
            text = style.names_arms(text)
        return text

    def _unplanned(self, text: str) -> str:
        kept = []
        for part in sentences(text):
            if all(style.proposes_sample(part) for style in self.styles):
                self.primary.proposed.append(part)
                continue
            kept.append(part)
        return " ".join(kept)

    def sentence(self, text: str, *, plans: bool = False) -> str:
        """One sentence the memo stands on (a title, a next step), held to every rule."""
        named = self._named(" ".join(text.split()))
        if plans and not (named := self._unplanned(named)):
            return ""
        return self.primary.sentence(named)

    def prose(self, text: str, *, plans: bool = False) -> str:
        """A paragraph (a reason, a finding's body, a summary, an override's why), each sentence held to every rule."""
        kept = []
        for part in sentences(self._named(text)):
            if plans and all(style.proposes_sample(part) for style in self.styles):
                self.primary.proposed.append(part)
                continue
            if any(style.holds_measured(part) for style in self.styles[1:]):
                self.primary.dropped += 1
                continue
            kept.append(part)
        return self.primary.prose(" ".join(kept)) if kept else ""

    def title(self, text: str) -> str:
        """A finding's title held to every rule; where that would leave none of it, kept as written and not counted as
        dropped, since the page prints it: a finding is never laid out without its headline."""
        return self.primary.or_as_written(lambda: self.sentence(text), self._named(text))

    def summary(self, text: str) -> str:
        """The memo's one sentence across the aspects, as :meth:`MemoStyle.summary` holds it, over every aspect: kept as
        written where the pass would leave none of it, unless it proposed a sample or cost no aspect projects."""
        return self.primary.or_as_written(lambda: self.prose(text, plans=True), self._named(text))

    def methods(self) -> list[ReportBlock]:
        """What the memo said that the page did not print as written, and what still breaks a rule: the methods."""
        blocks: list[ReportBlock] = [
            DisclosureBlock(section="methods", source="memo", text=text) for text in decision_disclosures(self.primary)
        ]
        blocks += [
            DisclosureBlock(section="methods", source="memo", text=text, fold="Style")
            for text in self.primary.offending()
        ]
        return blocks


def _names(bundle: AnalysisContextBundle, labels: Mapping[str, str]) -> dict[str, ArmName]:
    """Each arm the bundle holds, named as its variant names it, its label the campaign's where it gives one."""
    names = {key: arm_name(display) for key, display in arm_names(bundle.variant_index).items()}
    return {key: replace(name, label=labels[key], given=True) if key in labels else name for key, name in names.items()}

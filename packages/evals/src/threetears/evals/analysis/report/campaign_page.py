"""Build a campaign's page: what code read across its aspects, the writer's recommendation, the person's call, then
one fold per aspect.

Code reports, the writer recommends, a person decides, and the page keeps the three apart. The first screen, in
order:

1. **Every broken floor**, one loud line each, above the title: a floor only warns, and this is where it warns.
2. **The title and the question**, quoted whole as the campaign asks it.
3. **The writer's recommendation**, the arm and how sure, then each reason; with no memo, one line saying no
   recommendation was written.
4. **Code's table**, one row per aspect × candidate: the cases, each arm's pass count, the change in points with the
   interval it was read on, the aspect's margin, and what the row shows (shown better, shown worse or unclear). Under
   it, the lines that change how a row reads: the level each interval was read at, an aspect with no margin (never
   shown worse), an aspect whose question came after its first run (its readings are leads), a margin or a floor
   declared after the runs.
5. **What would settle each unclear row**: the cases that would show the arm within the aspect's margin, if the arms
   do not differ and if they differ as observed.
6. **The person's call**, or that none is recorded yet; a call adopting an arm past a broken floor says which floor it
   overrides.

Then the memo's findings and next steps, and one closed fold per aspect: the writer's summary of the aspect (what
changed, how sure, what it means) when there is one, then the aspect's own page as
:func:`~threetears.evals.analysis.report.build.build_code_only_report` built it, which the serializers lay out in the
fold without its title, its question or its broken floors.

Every arm is named by its label, never by its role; every line is held to :mod:`~threetears.evals.analysis.style`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from threetears.evals.analysis.campaign_readings import AspectReading, CampaignReadings, FloorStatus
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.quantities import POINTS
from threetears.evals.analysis.report.decision import floor_statement, projected_cell
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
    interval_text,
    table_row,
)
from threetears.evals.analysis.report.words import CONFIDENCE_WORDS
from threetears.evals.analysis.style import code, humanize
from threetears.evals.kernel.authored import AspectSummary, AuthoredCampaignAnalysis
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignCall, CostLower, Floor, RateDrop, aware_instant
from threetears.evals.kernel.errors import RefusedInput

__all__ = ["NO_RECOMMENDATION", "NO_CALL", "READING_WORDS", "build_campaign_report"]

#: What each reading is called on the page.
READING_WORDS: dict[str, str] = {"better": "shown better", "worse": "shown worse", "unclear": "unclear"}

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
) -> CampaignReport:
    """Lay a campaign out as its page: code's readings first, the writer's recommendation, the person's call, then one
    fold per aspect.

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
    :return: The page, ``basis="analysis"`` with a memo, ``code_only`` without one.
    :rtype: CampaignReport
    :raises RefusedInput: An aspect has no page, a page is keyed by a label the campaign does not hold, or the
        readings are of another campaign.
    """
    reports = _checked_reports(campaign, readings, aspect_reports)
    blocks: list[ReportBlock] = [*_floor_warnings(campaign, readings)]
    if campaign.question.strip():
        blocks.append(
            DisclosureBlock(section="header", source="basis", text=f'"{" ".join(campaign.question.split())}"')
        )
    blocks += _recommendation_blocks(memo)
    blocks += _table_blocks(campaign, readings, reports)
    blocks += _settle_blocks(readings)
    blocks += _call_blocks(campaign, readings, call)
    blocks += _memo_blocks(memo)
    summaries = {summary.aspect: summary for summary in (memo.aspect_summaries if memo is not None else [])}
    folds = [
        CampaignAspectFold(
            aspect=aspect.label,
            shows=_shows(readings.on_aspect(aspect.label)),
            summary=_summary_blocks(summaries.get(aspect.label)),
            report=reports[aspect.label],
        )
        for aspect in campaign.aspects
    ]
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
    )


def _checked_reports(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> dict[str, Report]:
    """Every aspect's page, refused unless there is exactly one per aspect and the readings are this campaign's."""
    if readings.campaign_id != campaign.id:
        raise RefusedInput(
            f"the readings are of campaign {readings.campaign_id}, not {campaign.id}",
            remedy="read the campaign's own readings (campaign_readings(campaign, bundles))",
        )
    labels = [aspect.label for aspect in campaign.aspects]
    if missing := [label for label in labels if label not in reports]:
        raise RefusedInput(
            f"campaign {campaign.id} has no page for aspect(s) {', '.join(missing)}",
            remedy="build one page per aspect (build_code_only_report over its bundle) and pass them all",
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


def _floor_of(campaign: EvalCampaign, key: str) -> Floor | None:
    """The campaign's floor by its key."""
    return next((floor for floor in campaign.floors if floor.key == key), None)


def _floor_words(floor: Floor | None, status: FloorStatus) -> str:
    """A floor as the page says it, naming its arm and its aspect: ``no pass rate drops by more than 10 points on
    billing``."""
    if floor is None:
        return f"a floor on {status.aspect}"
    return f"{floor_statement(floor.clause, candidate=status.arm_label)} on {status.aspect}"


def _floor_warnings(campaign: EvalCampaign, readings: CampaignReadings) -> list[ReportBlock]:
    """One loud line per floor a candidate breaks, the page's first, worded as an aspect's page words it: the aspect,
    the arm, and the interval it broke on."""
    return [
        DisclosureBlock(section="header", source="floor", text=_warning(_floor_of(campaign, status.floor), status))
        for status in readings.broken_floors()
    ]


def _warning(floor: Floor | None, status: FloorStatus) -> str:
    """``Floor broken on billing: v2's pass rate falls past the 10-point margin on its whole interval, a drop of 12.3
    to 35.2 points.`` A cost floor, or one this page cannot place, says its reason."""
    lead = f"Floor broken on {status.aspect}: "
    if floor is None or isinstance(floor.clause, CostLower) or status.interval is None:
        return f"{lead}{status.reason}."
    clause = floor.clause
    assert isinstance(clause, RateDrop)
    if status.reading is None:
        which = "pass rate"
    else:
        named = humanize("dimension", status.reading) if status.reading.isidentifier() else code(status.reading)
        which = f"{named} rate"
    low, high = status.interval
    return (
        f"{lead}{status.arm_label}'s {which} falls past the {format_number(clause.max_points)}-point margin on its "
        f"whole interval, a drop of {_points(-high)} to {_points(-low)} points."
    )


def _points(value: float) -> str:
    """A change in points to one decimal, a whole number whole."""
    return format_number(round(value, 1))


def _recommendation_blocks(memo: AuthoredCampaignAnalysis | None) -> list[ReportBlock]:
    """The writer's recommendation, the arm and how sure, then each reason in its words; one line with no memo."""
    if memo is None:
        return [DisclosureBlock(section="header", source="memo", text=NO_RECOMMENDATION)]
    recommendation = memo.recommendation
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
            for reason in recommendation.reasons
            if reason.strip()
        ),
    ]


def _shown_interval(row: AspectReading) -> tuple[tuple[float, float] | None, float | None]:
    """The interval a row's reading was read on, and its level: the gain's for shown better, the drop's for shown
    worse; for unclear, the wider of the two, which reaches past zero and short of the margin as the reading says."""
    if row.reading == "better":
        return row.gain_interval, row.gain_confidence
    if row.reading == "worse":
        return row.drop_interval, row.drop_confidence
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


def _table_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> list[ReportBlock]:
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
        if level is not None:
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
    notes = [_level_line(levels)] if levels else []
    notes += _unread_lines(campaign, readings)
    notes += _timing_lines(campaign, readings, reports)
    blocks += [DisclosureBlock(section="scoreboard", source="comparisons", text=note) for note in notes if note]
    return blocks


def _percent(level: float) -> str:
    """A level as a percent, to the precision that tells two levels apart."""
    return format_number(round(100 * level, 2)) + "%"


def _level_line(levels: Sequence[float]) -> str:
    """The level the table's intervals were read at, in one sentence."""
    low, high = min(levels), max(levels)
    if round(low, 4) == round(high, 4):
        if len(levels) == 1:
            return f"The interval is read at {_percent(low)}, as a floor's is."
        return f"Each interval is read at {_percent(low)}, corrected for every row the campaign tests."
    return (
        f"Each interval is read at the level its row was tested at, from {_percent(low)} to {_percent(high)}, "
        "corrected for every row; a reading never contradicts its interval."
    )


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


def _after(stamp: str | None, first_run: str | None, what: str) -> bool:
    """Whether ``stamp`` is later than the aspect's first run; False where either is unrecorded."""
    if stamp is None or first_run is None:
        return False
    return aware_instant(stamp, what) > aware_instant(first_run, "the aspect's first run")


def _timing_lines(campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]) -> list[str]:
    """What was declared after the evidence: a question asked after an aspect's first run makes its readings leads,
    and a margin or a floor added later reads as declared after the runs."""
    lines: list[str] = []
    if not campaign.question.strip():
        return ["No question was declared before the runs, so every reading is a lead."]
    for aspect in campaign.aspects:
        bundle = reports[aspect.label].bundle
        first_run = bundle.design_frozen_at
        if readings.on_aspect(aspect.label) and not bundle.question_declared:
            # Everything declared with a question asked late is late too: said once, of the question.
            lines.append(f"The question was asked after the first run on {aspect.label}, so its readings are leads.")
            continue
        declared = bundle.declared
        if declared is not None and _after(declared.margin_declared_at, first_run, f"the margin on {aspect.label}"):
            lines.append(f"The margin on {aspect.label} was declared after its first run.")
        for floor in campaign.floors:
            if floor.aspect == aspect.label and _after(floor.declared_at, first_run, f"floor {floor.key}"):
                said = floor_statement(floor.clause, candidate="the arm")
                lines.append(f"The floor that {said} on {aspect.label} was declared after its first run.")
    return lines


def _settle_blocks(readings: CampaignReadings) -> list[ReportBlock]:
    """What would settle each unclear row: why it is unclear, and the cases that would show the arm within the
    aspect's margin, if the arms do not differ and if they differ as observed."""
    unclear = [row for row in readings.readings if row.reading == "unclear"]
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
            title="To settle",
            columns=columns,
            rows=rows,
            order="as the table above",
            total_rows=len(rows),
        ),
        DisclosureBlock(
            section="scoreboard",
            source="comparisons",
            text=(
                "A count is the independent cases that would show the arm within the aspect's margin, read again over "
                "this run's cases repeated to that many. Repeats of a case narrow an interval far less than new cases."
            ),
        ),
    ]


def _call_blocks(campaign: EvalCampaign, readings: CampaignReadings, call: CampaignCall | None) -> list[ReportBlock]:
    """The person's call, who made it and when, their reason quoted, and every broken floor it overrides; a line
    saying none is recorded while none is."""
    if call is None:
        return [DisclosureBlock(section="header", source="call", text=NO_CALL)]
    verb = "keep" if call.variant_key == readings.control else "adopt"
    lines = [f"Called: {verb} {call.arm}, by {call.called_by} on {_date(call.called_at)}."]
    if call.reason.strip():
        lines.append(f'Reason: "{" ".join(call.reason.split())}"')
    broken = [status for status in readings.broken_floors() if status.arm == call.variant_key]
    for status in broken:
        said = _floor_words(_floor_of(campaign, status.floor), status)
        if status.floor in call.floors_overridden:
            lines.append(f"The call overrides a broken floor: {said}.")
        else:
            lines.append(f"The call does not name the broken floor as overridden: {said}.")
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


def _memo_blocks(memo: AuthoredCampaignAnalysis | None) -> list[ReportBlock]:
    """The memo's findings, each its headline and body, then its next steps; nothing with no memo."""
    if memo is None:
        return []
    blocks: list[ReportBlock] = []
    for position, finding in enumerate(memo.findings):
        blocks.append(
            TextBlock(
                section="memo", role="finding_title", body=finding.title, finding=position, anchor=f"F{position + 1}"
            )
        )
        blocks.append(TextBlock(section="memo", role="finding_body", body=finding.body, finding=position))
    blocks += [TextBlock(section="memo", role="next_step", body=step.title) for step in memo.next if step.title.strip()]
    return blocks


def _summary_blocks(summary: AspectSummary | None) -> list[ReportBlock]:
    """The writer's summary of one aspect: what changed, how sure, what it means; nothing with none."""
    if summary is None:
        return []
    parts: tuple[tuple[TextRole, str], ...] = (
        ("what_changed", summary.what_changed),
        ("how_sure", summary.how_sure),
        ("means", summary.means),
    )
    return [TextBlock(section="memo", role=role, body=body) for role, body in parts if body.strip()]


def _shows(rows: Sequence[AspectReading]) -> str:
    """What code read on one aspect, beside its fold's heading: ``v2 shown worse``, ``v2 shown better, v3 unclear``."""
    if not rows:
        return "not compared"
    return ", ".join(f"{row.arm_label} {_reading_words(row)}" for row in rows)

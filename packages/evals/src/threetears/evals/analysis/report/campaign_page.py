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
fold without its title, its question, its broken floors or its verdict line.

Every arm is named by its label, never by its role; every line is held to :mod:`~threetears.evals.analysis.style`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from threetears.evals.analysis.arms import arm_names
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.campaign_readings import AspectReading, CampaignReadings, FloorStatus, PairReading
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.quantities import POINTS
from threetears.evals.analysis.report.decision import floor_statement, projected_cell
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
    interval_text,
    table_row,
)
from threetears.evals.analysis.report.words import CONFIDENCE_WORDS, NO_RECOMMENDATION_WORDS
from threetears.evals.analysis.style import ArmName, arm_name, code, humanize, measure_name, sentences
from threetears.evals.kernel.authored import AspectSummary, AuthoredCampaignAnalysis
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignCall, CostLower, Floor, RateDrop, aware_instant
from threetears.evals.kernel.errors import RefusedInput

__all__ = ["BAR_WORDS", "CHOICE_NOTE", "NO_CALL", "NO_RECOMMENDATION", "READING_WORDS", "build_campaign_report"]

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
    style = _CampaignStyle.over(campaign, readings, reports)
    kind = readings.kind
    blocks: list[ReportBlock] = [
        *_bar_warnings(readings, memo, style),
        *_floor_warnings(campaign, readings, memo, style),
    ]
    if campaign.question.strip():
        blocks.append(
            DisclosureBlock(section="header", source="basis", text=f'"{" ".join(campaign.question.split())}"')
        )
    if kind == "change":
        blocks += _recommendation_blocks(memo, style, kind)
        blocks += _table_blocks(campaign, readings, reports)
        blocks += _settle_blocks(readings)
    elif kind == "choice":
        blocks += _pair_blocks(campaign, readings, reports)
        blocks += _behind_on(memo, style)
        blocks += _frontier_blocks(campaign, readings, reports)
        blocks += _recommendation_blocks(memo, style, kind)
    else:
        blocks += _measure_blocks(campaign, readings, reports)
    blocks += _bar_blocks(readings, reports)
    if kind != "measurement" or call is not None:
        blocks += _call_blocks(campaign, readings, call)
    blocks += _memo_blocks(memo, style)
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
        recommendation=memo.recommended() if memo is not None and kind != "measurement" else None,
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


def _floor_of(campaign: EvalCampaign, key: str) -> Floor | None:
    """The campaign's floor by its key."""
    return next((floor for floor in campaign.floors if floor.key == key), None)


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
    for status in readings.missed_bars():
        blocks.append(
            DisclosureBlock(section="header", source="bar", text=f"Bar missed on {status.aspect}: {status.reason}.")
        )
        if recommendation is not None and status.arm_label == recommendation.arm:
            blocks += answers.pop((status.aspect, status.measure), [])
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
    broken = readings.broken_floors()
    overrides = _overrides(memo, style)
    if len(broken) <= _WARNINGS_APART:
        blocks: list[ReportBlock] = []
        for status in broken:
            blocks.append(
                DisclosureBlock(
                    section="header", source="floor", text=_warning(_floor_of(campaign, status.floor), status)
                )
            )
            blocks += overrides.get(status.floor, [])
        return blocks
    by_arm: dict[str, list[FloorStatus]] = {}
    for status in broken:
        by_arm.setdefault(status.arm_label, []).append(status)
    grouped: list[ReportBlock] = []
    for arm, statuses in by_arm.items():
        grouped.append(
            DisclosureBlock(
                section="header",
                source="floor",
                text=f"Floors broken for {arm}: "
                + "; ".join(f"its {_which(status)} on {status.aspect}" for status in statuses)
                + ".",
            )
        )
        grouped += [block for status in statuses for block in overrides.get(status.floor, [])]
    return grouped


def _overrides(memo: AuthoredCampaignAnalysis | None, style: _CampaignStyle | None) -> dict[str, list[ReportBlock]]:
    """The writer's why for each broken floor its recommendation is made over, by the floor's key, styled."""
    recommendation = memo.recommended() if memo is not None else None
    if recommendation is None or style is None:
        return {}
    found: dict[str, list[ReportBlock]] = {}
    arm = recommendation.arm
    for override in recommendation.floors_overridden:
        if why := style.prose(override.why).strip():
            found.setdefault(override.floor, []).append(
                TextBlock(section="header", role="override", body=why, facts=[Fact(name="Arm", value=arm)])
            )
    return found


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
    notes = _level_lines(levels, readings) if levels else []
    notes += _floor_beside_lines(campaign, readings)
    notes += _unread_lines(campaign, readings)
    notes += _timing_lines(campaign, readings, reports)
    blocks += [DisclosureBlock(section="scoreboard", source="comparisons", text=note) for note in notes if note]
    return blocks


def _percent(level: float) -> str:
    """A level as a percent, to the precision that tells two levels apart."""
    return format_number(round(100 * level, 2)) + "%"


def _level_lines(levels: Sequence[float], readings: CampaignReadings) -> list[str]:
    """The level the table's intervals were read at, how often a reading can be wrong, and, with several arms, that
    nothing corrects across them."""
    low, high = min(levels), max(levels)
    if round(low, 4) == round(high, 4) and len(levels) == 1:
        said = f"The interval is read at {_percent(low)}, as a floor's is."
    elif round(low, 4) == round(high, 4):
        said = f"Each interval is read at {_percent(low)}, corrected across each arm's aspects."
    else:
        said = (
            f"Each interval is read at the level its row was tested at, from {_percent(low)} to {_percent(high)}, "
            "corrected across each arm's aspects; a reading never contradicts its interval."
        )
    half = _percent(readings.alpha / 2)
    lines = [said, f"Each arm's gains and its drops are each shown wrongly at most {half} of the time."]
    if len({row.arm for row in readings.readings}) > 1:
        lines.append(
            "Nothing corrects across arms, so with several arms one is shown better or worse by chance more often."
        )
    return lines


def _floor_beside_lines(campaign: EvalCampaign, readings: CampaignReadings) -> list[str]:
    """A broken floor beside a row that is not shown worse: the two read different intervals, and say which."""
    lines: list[str] = []
    for status in readings.broken_floors():
        row = next((row for row in readings.on_aspect(status.aspect) if row.arm == status.arm), None)
        if row is None or row.reading == "worse":
            continue
        lines.append(
            f"The floor broken on {status.aspect} for {status.arm_label} reads its {_which(status)} at 95% on its "
            f"own, while its row reads only the pass rate, corrected across {status.arm_label}'s aspects."
        )
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


def _read_on(readings: CampaignReadings, aspect: str) -> bool:
    """Whether code read anything on ``aspect``: a row, a pair or a measure."""
    return (
        any(row.aspect == aspect for row in readings.readings)
        or any(pair.aspect == aspect for pair in readings.pairs)
        or any(measure.aspect == aspect for measure in readings.measures)
    )


def _after(stamp: str | None, first_result: str | None, what: str) -> bool:
    """Whether ``stamp`` is later than the aspect's first result; False where either is unrecorded."""
    if stamp is None or first_result is None:
        return False
    return aware_instant(stamp, what) > aware_instant(first_result, "the aspect's first result")


def _timing_lines(campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]) -> list[str]:
    """What was declared after the evidence: a question asked after an aspect's first run makes its readings leads
    (the bundle's ``question_declared``, the rule every reading reads); the question's wording, a margin or a floor
    added later than the aspect's first result (the bundle's ``first_result_at``) reads as declared after it."""
    lines: list[str] = []
    if not campaign.question.strip():
        return ["No question was declared, so every reading is a lead."]
    for aspect in campaign.aspects:
        bundle = reports[aspect.label].bundle
        first_result = bundle.first_result_at
        if _read_on(readings, aspect.label) and not bundle.question_declared:
            # Everything declared with a question asked late is late too: said once, of the question.
            lines.append(f"The question was asked after the first run on {aspect.label}, so its readings are leads.")
            continue
        declared = bundle.declared
        label = aspect.label
        if declared is not None and _after(declared.question_declared_at, first_result, f"the question on {label}"):
            lines.append(
                f"The question was worded after the first result on {label}; its readings keep the time it was asked."
            )
        if declared is not None and _after(declared.margin_declared_at, first_result, f"the margin on {label}"):
            lines.append(f"The margin on {label} was declared after its first result.")
        for floor in campaign.floors:
            if floor.aspect == label and _after(floor.declared_at, first_result, f"floor {floor.key}"):
                said = floor_statement(floor.clause, candidate="the arm")
                lines.append(f"The floor that {said} on {label} was declared after its first result.")
    return lines


def _settle_blocks(readings: CampaignReadings) -> list[ReportBlock]:
    """Each unclear row: why it is unclear, and the cases that would show the arm no worse than the control by more
    than the aspect's margin, if the arms do not differ and if they differ as observed. They would not show it better:
    the row stays unclear unless its gain clears too, and the note says so."""
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
                "aspect's margin. It would not show the arm better, so its row stays unclear unless its gain clears "
                "too. Each count reads the interval again over this run's cases repeated to that many; repeats of a "
                "case narrow it far less than new cases."
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


def _memo_blocks(memo: AuthoredCampaignAnalysis | None, style: _CampaignStyle) -> list[ReportBlock]:
    """The memo's findings, each its headline and body, then its next steps, every word through the style pass;
    nothing with no memo."""
    if memo is None:
        return []
    blocks: list[ReportBlock] = []
    for position, finding in enumerate(memo.findings):
        title = style.sentence(finding.title) or " ".join(finding.title.split())
        blocks.append(
            TextBlock(section="memo", role="finding_title", body=title, finding=position, anchor=f"F{position + 1}")
        )
        if body := style.prose(finding.body):
            blocks.append(TextBlock(section="memo", role="finding_body", body=body, finding=position))
    for step in memo.next:
        if title := style.sentence(step.title, plans=True):
            blocks.append(TextBlock(section="memo", role="next_step", body=title))
    return blocks


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


def _pair_words(pair: PairReading) -> str:
    """A pair as a fold's heading says it: ``alpha ahead of beta``, ``alpha and beta not separated``."""
    if pair.ahead is None:
        said = f"{pair.first_label} and {pair.second_label} not separated"
    else:
        behind = pair.second_label if pair.ahead == pair.first_label else pair.first_label
        said = f"{pair.ahead} ahead of {behind}"
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
        ahead = pair.ahead or "no separation"
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
    notes += _timing_lines(campaign, readings, reports)
    notes += _unread_without_control_lines(campaign, reports)
    blocks += [DisclosureBlock(section="scoreboard", source="comparisons", text=note) for note in notes if note]
    return blocks


def _frontier_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> list[ReportBlock]:
    """Where an aspect's frontier names the cheapest arm shown to clear its bar, one line saying so: a reading of
    the bundle's frontier, never a pick."""
    labels = {pair.first: pair.first_label for pair in readings.pairs} | {
        pair.second: pair.second_label for pair in readings.pairs
    }
    lines: list[ReportBlock] = []
    for aspect in campaign.aspects:
        for subject in reports[aspect.label].bundle.frontier.subjects:
            verdict = subject.verdict
            if verdict is None:
                continue
            named = labels.get(verdict.variant_key, verdict.model)
            tied = [labels.get(key, key[:12]) for key in getattr(verdict, "tied_with", []) or []]
            beside = f", not shown cheaper than {', '.join(tied)}" if tied else ""
            lines.append(
                DisclosureBlock(
                    section="scoreboard",
                    source="comparisons",
                    text=f"Cheapest arm shown to clear the bar on {aspect.label}: {named}{beside}.",
                )
            )
    return lines


def _measure_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> list[ReportBlock]:
    """A measurement's table: each aspect's measures for its arm, each value with its interval."""
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="arm", header="Arm"),
        TableColumn(key="measure", header="Measure", term="reading"),
        TableColumn(key="value", header="Value"),
        TableColumn(key="interval", header="Interval", term="interval"),
        TableColumn(key="cases", header="Cases", term="cases"),
    ]
    rows = [
        table_row(
            {
                "aspect": reading.aspect,
                "arm": reading.arm_label,
                "measure": reading.name,
                "value": None if reading.value is None else round(reading.value, 4),
                "interval": None
                if reading.interval is None
                else f"[{format_number(round(reading.interval[0], 4))}, {format_number(round(reading.interval[1], 4))}]",
                "cases": reading.cases,
            }
        )
        for reading in readings.measures
    ]
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
    notes = _timing_lines(campaign, readings, reports) + _unread_without_control_lines(campaign, reports)
    blocks += [DisclosureBlock(section="scoreboard", source="comparisons", text=note) for note in notes]
    return blocks


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
        labels = readings.arm_labels() | {pair.first: pair.first_label for pair in readings.pairs}
        labels |= {pair.second: pair.second_label for pair in readings.pairs}
        labels |= {measure.arm: measure.arm_label for measure in readings.measures}
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

"""The writer's and the person's words on a campaign's page, each as written, and what code fills beside them.

**The writer's summary**, its read across every aspect, for every kind that has a memo; a code-only page has none.
**The writer's recommendation** for a change or a choice: the arm and how sure, then each reason; where the memo
recommends nothing, one line by the campaign's kind (no clear winner); with no memo, one line saying no
recommendation was written. For a choice, the writer's why for each aspect its pick is shown behind on. **The
person's call**, or that none is recorded yet; a call adopting an arm past a broken floor says which floor it
overrides. Then **the memo's findings**, each with its trace to the folds, and its **next steps**; and, in each
aspect's fold, **the writer's summary of the aspect** (what changed, how sure, what it means).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.campaign_readings import CampaignReadings, FloorStatus
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.report.campaign.common import as_written, floor_of, named_arms
from threetears.evals.analysis.report.campaign.names import AspectNames
from threetears.evals.analysis.report.decision import floor_statement
from threetears.evals.analysis.report.figures import report_rulers
from threetears.evals.analysis.report.model import (
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
    reading_anchor,
)
from threetears.evals.analysis.report.words import (
    CONFIDENCE_WORDS,
    NO_RECOMMENDATION_WORDS,
    UNTRACED,
    evidence_figure,
    reading_heading,
)
from threetears.evals.analysis.style import code
from threetears.evals.kernel.authored import AspectSummary, AuthoredCampaignAnalysis, Finding
from threetears.evals.kernel.campaign import EvalCampaign, FindingResolution
from threetears.evals.kernel.declaration import CampaignCall, Floor, aware_instant, resolve_control_arm

__all__ = [
    "NO_CALL",
    "NO_RECOMMENDATION",
    "aspect_summary_blocks",
    "behind_on_blocks",
    "call_blocks",
    "memo_blocks",
    "memo_summary_blocks",
    "recommendation_blocks",
]

#: The first screen's line where no memo was written over the campaign.
NO_RECOMMENDATION = "No recommendation was written: no analysis has been generated over this campaign."

#: The first screen's line while no person has made the call.
NO_CALL = "No call is recorded yet: a person decides, over the readings above."


def memo_summary_blocks(memo: AuthoredCampaignAnalysis | None) -> list[ReportBlock]:
    """The writer's read across every aspect, as written, the first thing the page says after code's warnings and the
    question, for every kind; nothing with no memo.

    :param memo: The writer's memo; None where none was written.
    :ptype memo: AuthoredCampaignAnalysis | None
    :return: The summary, or nothing.
    :rtype: list[ReportBlock]
    """
    if memo is None or not (said := as_written(memo.summary)):
        return []
    return [TextBlock(section="header", role="summary", body=said)]


def recommendation_blocks(memo: AuthoredCampaignAnalysis | None, kind: str) -> list[ReportBlock]:
    """The writer's recommendation, the arm and how sure, then each reason in its words, as written; one line with no
    memo, and one by the campaign's kind where the memo recommends nothing.

    :param memo: The writer's memo; None where none was written.
    :ptype memo: AuthoredCampaignAnalysis | None
    :param kind: The campaign's kind, which words a memo recommending nothing.
    :ptype kind: str
    :return: The recommendation and its reasons, or the one line.
    :rtype: list[ReportBlock]
    """
    if memo is None:
        return [DisclosureBlock(section="header", source="memo", text=NO_RECOMMENDATION)]
    recommendation = memo.recommended()
    if recommendation is None:
        return [DisclosureBlock(section="header", source="memo", text=NO_RECOMMENDATION_WORDS[kind])]
    reasons = [as_written(reason) for reason in recommendation.reasons]
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


def behind_on_blocks(memo: AuthoredCampaignAnalysis | None, *, names: AspectNames) -> list[ReportBlock]:
    """The writer's why for each aspect a choice's recommended arm is shown behind on, set beside the pairs: what the
    recommendation answers there (``behind_on``), as written.

    :param memo: The writer's memo; None where none was written.
    :ptype memo: AuthoredCampaignAnalysis | None
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: One block per why; nothing with no recommendation.
    :rtype: list[ReportBlock]
    """
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
                Fact(name="Over", value=f"{behind.ahead_arm}, ahead on {names(behind.aspect)}"),
            ],
        )
        for behind in recommendation.behind_on
        if (why := as_written(behind.why))
    ]


def _floor_words(floor: Floor | None, status: FloorStatus, names: AspectNames) -> str:
    """A floor as the page says it, naming its arm and its aspect: ``no pass rate drops by more than 10 points on
    billing``."""
    if floor is None:
        return f"a floor on {names(status.aspect)}"
    return f"{floor_statement(floor.clause, candidate=status.arm_label)} on {names(status.aspect)}"


def call_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, call: CampaignCall | None, *, names: AspectNames
) -> list[ReportBlock]:
    """The person's call, who made it and when, their reason quoted, and every floor and bar it overrides; a line
    saying none is recorded while none is.

    Every override the call stored is said, as stored, whatever the readings on this page now say of it: one that no
    longer warns is marked so, never dropped. Beside them, a floor broken or a bar missed here that the call does not
    override, and in a choice each pair where another arm is shown ahead of the decided one. The decided arm is read
    by every key it resolves to (:func:`~threetears.evals.kernel.declaration.resolve_control_arm`), as recording read
    it, so a control stored under an older identity predicate is still found.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param call: The person's call; None while none is recorded.
    :ptype call: CampaignCall | None
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: The call's lines.
    :rtype: list[ReportBlock]
    """
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
            lines.append(
                f"The call overrides a broken floor: {_floor_words(floor_of(campaign.floors, key), broken, names)}."
            )
        else:
            lines.append(f"The call overrides the floor {code(key)} (no longer broken).")
    unanswered: dict[str, FloorStatus] = {}
    for status in floors:
        if status.status == "broken" and status.floor not in call.floors_overridden:
            unanswered.setdefault(status.floor, status)
    for key, status in unanswered.items():
        said = _floor_words(floor_of(campaign.floors, key), status, names)
        lines.append(f"The call does not name the broken floor as overridden: {said}.")
    missed = {(bar.aspect, bar.measure): bar for bar in readings.missed_bars() if bar.arm in keys}
    overridden = {(bar.aspect, bar.measure) for bar in call.bars_overridden}
    for bar in call.bars_overridden:
        if (still := missed.get((bar.aspect, bar.measure))) is not None:
            lines.append(f"The call overrides a missed bar on {names(bar.aspect)}: {still.reason}.")
        else:
            on = names(bar.aspect)
            lines.append(f"The call overrides the bar on {code(bar.measure)} on {on} (no longer missed).")
    for (aspect, measure), unnamed in missed.items():
        if (aspect, measure) not in overridden:
            on = names(aspect)
            lines.append(f"The call does not name the missed bar on {on} as overridden: {unnamed.reason}.")
    for pair in readings.pairs:
        if pair.ahead is not None and pair.ahead not in keys and keys & {pair.first, pair.second}:
            lines.append(f"{pair.ahead_label or pair.ahead} is shown ahead of {call.arm} on {names(pair.aspect)}.")
    return [DisclosureBlock(section="header", source="call", text=line) for line in lines]


def _date(stamp: str) -> str:
    """An ISO-8601 instant as the page dates it (``14 Oct 2026``), or as given when it does not parse."""
    try:
        moment = aware_instant(stamp, "the call")
    except ValueError:
        # NOSILENT: an unparseable stamp is printed as stored rather than dropped or reformatted
        return stamp
    return f"{moment.day} {moment:%b %Y}"


def memo_blocks(
    memo: AuthoredCampaignAnalysis | None,
    campaign: EvalCampaign,
    readings: CampaignReadings,
    reports: Mapping[str, Report],
    resolutions: Sequence[FindingResolution],
    trial_ids: Mapping[str, tuple[str, str]],
    *,
    names: AspectNames,
) -> list[ReportBlock]:
    """The memo's findings, each its headline and body and, folded, its trace; then its next steps, every word as
    written; nothing with no memo.

    :param memo: The writer's memo; None where none was written.
    :ptype memo: AuthoredCampaignAnalysis | None
    :param campaign: The campaign, whose folds each trace links.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it, whose labels name each arm the trace shows.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :param resolutions: What code filled for each finding, by position.
    :ptype resolutions: Sequence[FindingResolution]
    :param trial_ids: The ids the memo names trials by, each to its aspect and its id there.
    :ptype trial_ids: Mapping[str, tuple[str, str]]
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: The findings and next steps, or nothing.
    :rtype: list[ReportBlock]
    """
    if memo is None:
        return []
    trace = _Trace.over(campaign, readings, reports, resolutions, trial_ids, names)
    blocks: list[ReportBlock] = []
    for position, finding in enumerate(memo.findings):
        title = as_written(finding.title)
        blocks.append(
            TextBlock(section="memo", role="finding_title", body=title, finding=position, anchor=f"F{position + 1}")
        )
        if body := as_written(finding.body):
            blocks.append(TextBlock(section="memo", role="finding_body", body=body, finding=position))
        blocks += trace.blocks(position, finding)
    for step in memo.next:
        if title := as_written(step.title):
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
    names: AspectNames

    @classmethod
    def over(
        cls,
        campaign: EvalCampaign,
        readings: CampaignReadings,
        reports: Mapping[str, Report],
        resolutions: Sequence[FindingResolution],
        trial_ids: Mapping[str, tuple[str, str]],
        names: AspectNames,
    ) -> _Trace:
        """The trace over the campaign's folds, each arm named by its campaign label as the folds name it."""
        return cls(
            campaign=campaign,
            labels=readings.arm_labels(),
            reports=reports,
            resolutions=resolutions,
            trial_ids=trial_ids,
            names=names,
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
                    "aspect": TableCell(value=self.names(row.aspect)),
                    "arm": TableCell(value=self._arm(row.aspect, variant_of_cell_ref(row.cell_ref) or row.cell_ref)),
                    "reading": TableCell(
                        value=ruler.scaled(
                            reading_heading(
                                surface,
                                row.measure_id,
                                row.reading,
                                declared=report.bundle.check_name(row.measure_id),
                            )
                        )
                    ),
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
            "aspect": TableCell(value=self.names(aspect)),
            "arm": TableCell(value=self._arm(aspect, arm) if arm is not None else None),
            "reading": TableCell(value="trial"),
            "value": TableCell(value=own, ref=self._fold(aspect) if arm is not None else None),
        }

    def _arm(self, aspect: str, variant_key: str) -> str:
        """An arm as the aspect's fold names it: by the campaign's label, else its variant's name."""
        names = named_arms(self.reports[aspect].bundle, self.labels)
        return names[variant_key].label if variant_key in names else f"unplaced arm {short_digest(variant_key)}"

    def _fold(self, aspect: str) -> str:
        """The id of the section around ``aspect``'s fold, outside its closed ``<details>``."""
        labels = [held.label for held in self.campaign.aspects]
        return f"aspect-{fold_ids(labels)[labels.index(aspect)]}"


def aspect_summary_blocks(summary: AspectSummary | None) -> list[ReportBlock]:
    """The writer's summary of one aspect: what changed, how sure, what it means, as written; nothing with none.

    :param summary: The writer's summary of the aspect; None where the memo has none for it.
    :ptype summary: AspectSummary | None
    :return: Its parts, each a text block, an empty part left out.
    :rtype: list[ReportBlock]
    """
    if summary is None:
        return []
    parts: tuple[tuple[TextRole, str], ...] = (
        ("what_changed", as_written(summary.what_changed)),
        ("how_sure", as_written(summary.how_sure)),
        ("means", as_written(summary.means)),
    )
    return [TextBlock(section="memo", role=role, body=body) for role, body in parts if body.strip()]

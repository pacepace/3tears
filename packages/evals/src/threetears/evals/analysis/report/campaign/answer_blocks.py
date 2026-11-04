"""The lines on a campaign's page that say what became of its analysis, and the person's call.

Code's page says where the campaign's report is: one line naming the report document it is stored as
(:func:`report_doc_blocks`); one line where the analysis it stands in for is one this version no longer reads or a
retired writer wrote (:func:`unread_analysis_blocks`, :func:`retired_generation`); or, with neither, that no
recommendation was written (:data:`NO_RECOMMENDATION`). Then **the person's call**, or that none is recorded yet; a
call adopting an arm past a broken floor says which floor it overrides (:func:`call_blocks`).
"""

from __future__ import annotations

from collections.abc import Mapping

from threetears.evals.analysis.campaign_readings import CampaignReadings, FloorStatus
from threetears.evals.analysis.report.campaign.common import floor_of
from threetears.evals.analysis.report.campaign.names import AspectNames, measure_name_of
from threetears.evals.analysis.report.decision import floor_statement
from threetears.evals.analysis.report.model import DisclosureBlock, Report, ReportBlock
from threetears.evals.analysis.style import code
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignCall, Floor, aware_instant, resolve_control_arm
from threetears.evals.kernel.metrics import goal_check_of

__all__ = [
    "NO_CALL",
    "NO_LONGER_READS",
    "NO_RECOMMENDATION",
    "call_blocks",
    "report_doc_blocks",
    "retired_generation",
    "unread_analysis_blocks",
]

#: The first screen's line where no report was written over the campaign, and it stands in for none.
NO_RECOMMENDATION = "No recommendation was written: no analysis has been generated over this campaign."

#: The first screen's line where the analysis laid out is a report document's (``EvalAnalysis.report_doc_id``): the
#: report is stored apart, and this page is code's over the same evidence.
DOC_WRITTEN = (
    "A report was written over this campaign's evidence and is stored as report document {report_doc_id}. This page "
    "is the code-only page over the same evidence."
)

#: The one line every surface prints where the campaign's reported analysis is one this version no longer reads: a
#: report document that no longer validates, whose snapshot
#: is gone or no longer reads, or whose campaign no longer lays out over it. Worded once, here: the page's first screen
#: (the stand-in line below), ``ReportDocument.needs_regeneration`` and the analysis listing's ``needs_regeneration``
#: each print it byte for byte. It names the action bare: the HTML page sets a backticked word as code, which would
#: print it differently there.
NO_LONGER_READS = "written in a shape this version no longer reads; regenerate with analysis_generate"

#: The one line every surface prints where the campaign's reported analysis was written by a report writer this
#: package has retired, in place of :data:`NO_LONGER_READS`: worded once, here, and printed byte for byte by the page's
#: first screen, ``ReportDocument.needs_regeneration`` and the analysis listing's ``needs_regeneration``. ``{writer}``
#: names the writer (:func:`retired_generation`).
_RETIRED_GENERATION = (
    "written by a retired report writer ({writer}); regenerate with analysis_generate to read a report over it"
)

#: Each retired writer, by what it stored (``EvalAnalysis.basis``), as :func:`retired_generation` names it.
_RETIRED_WRITERS: dict[str, str] = {"memo": "the memo writer", "plan": "the report planner"}

#: The first screen's line on the code-only page that stands in for an analysis this version no longer reads, in place
#: of "no analysis has been generated"; ``{line}`` is :data:`NO_LONGER_READS` or the retired writer's line.
_UNREAD_ANALYSIS = "This campaign's analysis {analysis_id} was {line}."

#: The same line where the code-only page stands in for one report document read by its id.
_UNREAD_REPORT_DOC = "Report document {report_doc_id} was " + NO_LONGER_READS + "."

#: The first screen's line while no person has made the call.
NO_CALL = "No call is recorded yet: a person decides, over code's readings."


def report_doc_blocks(report_doc_id: str | None) -> list[ReportBlock]:
    """The one line saying a report document was written and where it is stored; nothing for any other page.

    :param report_doc_id: The report document the analysis laid out names (``EvalAnalysis.report_doc_id``), or None.
    :ptype report_doc_id: str | None
    :return: The line, or nothing.
    :rtype: list[ReportBlock]
    """
    if report_doc_id is None:
        return []
    return [DisclosureBlock(section="answer", source="memo", text=DOC_WRITTEN.format(report_doc_id=report_doc_id))]


def retired_generation(basis: str) -> str | None:
    """The one line for an analysis a retired report writer wrote, naming the writer (worded once, beside
    :data:`NO_LONGER_READS`);
    None for one a writer this version reads.

    :param basis: What wrote the analysis (``EvalAnalysis.basis``).
    :ptype basis: str
    :return: The line, or None.
    :rtype: str | None
    """
    writer = _RETIRED_WRITERS.get(basis)
    return _RETIRED_GENERATION.format(writer=writer) if writer is not None else None


def unread_analysis_blocks(
    analysis_id: str | None, report_doc_id: str | None = None, *, line: str = NO_LONGER_READS
) -> list[ReportBlock]:
    """The one line saying what the code-only page stands in for is one this version no longer reads: the campaign's
    analysis, or a report document read by its id, each in its own words around :data:`NO_LONGER_READS`, or around
    the retired writer's line (:func:`retired_generation`) for an analysis a retired writer wrote; nothing for any
    other page.

    :param analysis_id: The analysis the code-only page stands in for, or None.
    :ptype analysis_id: str | None
    :param report_doc_id: The report document it stands in for, read by its id, or None.
    :ptype report_doc_id: str | None
    :param line: What the analysis was: :data:`NO_LONGER_READS`, or the retired writer's line.
    :ptype line: str
    :return: The line, or nothing.
    :rtype: list[ReportBlock]
    """
    if analysis_id is not None:
        text = _UNREAD_ANALYSIS.format(analysis_id=analysis_id, line=line)
    elif report_doc_id is not None:
        text = _UNREAD_REPORT_DOC.format(report_doc_id=report_doc_id)
    else:
        return []
    return [DisclosureBlock(section="answer", source="memo", text=text)]


def _floor_words(floor: Floor | None, status: FloorStatus, names: AspectNames) -> str:
    """A floor as the page says it, naming its arm and its aspect: ``no pass rate drops by more than 10 points on
    billing``."""
    if floor is None:
        return f"a floor on {names(status.aspect)}"
    said = floor_statement(floor.clause, candidate=status.arm_label, check_names=names.check_names(status.aspect))
    return f"{said} on {names(status.aspect)}"


def call_blocks(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    call: CampaignCall | None,
    *,
    names: AspectNames,
    reports: Mapping[str, Report],
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
    :param reports: Each aspect's own page, by its label, whose evidence names a bar's measure.
    :ptype reports: Mapping[str, Report]
    :return: The call's lines.
    :rtype: list[ReportBlock]
    """
    if call is None:
        return [DisclosureBlock(section="header", source="call", text=NO_CALL)]
    named = next((arm for arm in campaign.arms if arm.variant_key == call.variant_key), None)
    keys = {call.variant_key, *(resolve_control_arm(named).variant_keys if named is not None else [])}
    verb = "keep" if readings.control in keys else "adopt"
    # The decided arm as the page names it (its readings print it by the name every surface prints), never the label
    # the call stored, which a computed lever name reads in another case.
    labels = readings.arm_labels()
    decided = next((labels[key] for key in (call.variant_key, *keys) if key in labels), call.arm)
    head = f"Called: {verb} {decided}, by {call.called_by} on {_date(call.called_at)}"
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

    def measure(aspect: str, key: str) -> str:
        # The bar's measure as the aspect's page names it: a goal check by its declared name, else as the readings do.
        report = reports.get(aspect)
        return measure_name_of(report.bundle, key) if report is not None else code(key)

    def stored(aspect: str, key: str) -> str:
        # A bar the readings no longer hold missed, by the key the call stored, unless it is a goal check's, which is
        # named as the aspect's page names it.
        return measure(aspect, key) if goal_check_of(key) is not None else code(key)

    for bar in call.bars_overridden:
        if (still := missed.get((bar.aspect, bar.measure))) is not None:
            said = still.missed_reason(measure(bar.aspect, bar.measure))
            lines.append(f"The call overrides a missed bar on {names(bar.aspect)}: {said}.")
        else:
            on = names(bar.aspect)
            lines.append(f"The call overrides the bar on {stored(bar.aspect, bar.measure)} on {on} (no longer missed).")
    for (aspect, key), unnamed in missed.items():
        if (aspect, key) not in overridden:
            said = unnamed.missed_reason(measure(aspect, key))
            lines.append(f"The call does not name the missed bar on {names(aspect)} as overridden: {said}.")
    for pair in readings.pairs:
        if pair.ahead is not None and pair.ahead not in keys and keys & {pair.first, pair.second}:
            lines.append(f"{pair.ahead_label or pair.ahead} is shown ahead of {decided} on {names(pair.aspect)}.")
    return [DisclosureBlock(section="header", source="call", text=line) for line in lines]


def _date(stamp: str) -> str:
    """An ISO-8601 instant as the page dates it (``14 Oct 2026``), or as given when it does not parse."""
    try:
        moment = aware_instant(stamp, "the call")
    except ValueError:
        # NOSILENT: an unparseable stamp is printed as stored rather than dropped or reformatted
        return stamp
    return f"{moment.day} {moment:%b %Y}"

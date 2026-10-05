"""Build a :class:`~threetears.evals.analysis.report.model.Report` from a stored analysis.

Derived on every read and never stored, for the reason the arm and surface tables are: a stored
analysis keeps its facts, and how they are laid out is a decision a later build may make better. The
order is the order a reader acts on: the summary, where each declared question stands, the decisions,
the findings with their evidence, charts and caveats, the arms, the decision surface, what to run next,
and the analysis-wide disclosures.
"""

from __future__ import annotations

from collections.abc import Callable

from threetears.evals.analysis.arms import arm_table
from threetears.evals.analysis.report.model import (
    ChartBlock,
    DisclosureBlock,
    Fact,
    Report,
    ReportBlock,
    ReportSource,
    TableBlock,
    TableColumn,
    TextBlock,
)
from threetears.evals.analysis.report.words import (
    ARM_STATUS_WORDS,
    CONFIDENCE_WORDS,
    EVIDENCE_TIER_WORDS,
    arm_namer,
    positions,
)
from threetears.evals.analysis.surface_table import build_surface_table
from threetears.evals.analysis.viz.intent import Cell, chart_intent
from threetears.evals.analysis.viz.payloads import PayloadError
from threetears.evals.analysis.viz.policy import IntentPolicyError
from threetears.evals.contracts.authored import NO_CHART, Finding
from threetears.evals.contracts.campaign import EvalAnalysis, FindingResolution, Viz
from threetears.observe import get_logger

log = get_logger(__name__)


def build_report(analysis: EvalAnalysis) -> Report:
    """Lay one stored analysis out as a report.

    Args:
        analysis: The analysis.

    Returns:
        The report.
    """
    document = analysis.document
    arm = arm_namer(analysis)
    blocks: list[ReportBlock] = []
    if document.summary.strip():
        blocks.append(TextBlock(section="summary", role="summary", body=document.summary))

    blocks.extend(
        TextBlock(
            section="questions",
            role="answer",
            body=answer.answer,
            rests_on=list(answer.rests_on),
            facts=[Fact(name="Question", value=answer.question_id), Fact(name="Resolution", value=answer.resolution)],
        )
        for answer in document.questions
    )

    for decision in document.decisions:
        facts = [
            Fact(name="Disposition", value=decision.disposition),
            Fact(name="Confidence", value=CONFIDENCE_WORDS[decision.confidence]),
        ]
        if decision.cells:
            facts.append(Fact(name="Arms", value="; ".join(arm(cell) for cell in decision.cells)))
        blocks.append(
            TextBlock(
                section="decisions",
                role="decision",
                body=decision.proposal,
                rests_on=list(decision.rests_on),
                facts=facts,
            )
        )
        if decision.revisit_when.strip():
            blocks.append(
                TextBlock(
                    section="decisions",
                    role="revisit_when",
                    body=decision.revisit_when,
                    rests_on=list(decision.rests_on),
                )
            )

    resolutions: list[FindingResolution | None] = (
        list(analysis.resolutions) if analysis.resolutions else [None] * len(document.findings)
    )
    for position, (finding, resolution) in enumerate(zip(document.findings, resolutions, strict=True)):
        blocks.extend(_finding_blocks(analysis, position, finding, resolution, arm))

    blocks.extend(_arm_blocks(analysis))
    blocks.extend(_surface_blocks(analysis))

    for step in document.next:
        facts = [Fact(name="Leverage", value=step.leverage)]
        if step.lever:
            facts.append(Fact(name="Lever", value=step.lever))
        blocks.append(TextBlock(section="next", role="next_step", body=step.title, facts=facts))
        if step.why.strip():
            blocks.append(TextBlock(section="next", role="next_step_why", body=step.why))

    blocks.extend(_method_blocks(analysis))
    generation = analysis.generation
    return Report(
        headline=document.headline,
        finding_count=len(document.findings),
        source=ReportSource(
            analysis_id=analysis.id,
            campaign_id=analysis.campaign_id,
            scope_id=analysis.scope_id,
            subject_id=analysis.subject_id,
            subject_kind=analysis.subject_kind,
            behavior=analysis.behavior,
            generated_at=generation.generated_at,
            generator_model=generation.generator_model,
            bundle_fingerprint=generation.bundle_fingerprint,
        ),
        blocks=blocks,
    )


def _finding_blocks(
    analysis: EvalAnalysis,
    position: int,
    finding: Finding,
    resolution: FindingResolution | None,
    arm: Callable[[str], str],
) -> list[ReportBlock]:
    """One finding: its title and the facts beside it, its body, its evidence, its chart, its caveats.

    Args:
        analysis: The analysis, named in a log line when a stored chart will not draw.
        position: The finding's position in the document.
        finding: The authored finding.
        resolution: What code filled for it, or None for an analysis with no resolutions.
        arm: Names the arm a cell reference points at.

    Returns:
        The finding's blocks, in reading order.
    """
    facts = [Fact(name="Confidence", value=CONFIDENCE_WORDS[finding.confidence])]
    if resolution is not None:
        facts.append(Fact(name="Stands on", value=EVIDENCE_TIER_WORDS[resolution.evidence_tier]))
    if finding.axes:
        facts.append(Fact(name="About", value=", ".join(finding.axes)))
    if finding.invalidates:
        facts.append(Fact(name="Invalidates finding", value=positions(list(finding.invalidates))))
    blocks: list[ReportBlock] = [
        TextBlock(section="findings", role="finding_title", finding=position, body=finding.title, facts=facts)
    ]
    if finding.body.strip():
        blocks.append(TextBlock(section="findings", role="finding_body", finding=position, body=finding.body))
    if resolution is not None and resolution.evidence:
        rows: list[dict[str, Cell]] = [
            {
                "arm": arm(row.cell_ref),
                "measure": f"{row.measure_id} (judged)" if row.reading == "judged" else row.measure_id,
                "value": row.value,
                "n": row.n,
                "spread": row.dispersion,
            }
            for row in resolution.evidence
        ]
        blocks.append(
            TableBlock(
                section="findings",
                finding=position,
                name="evidence",
                title="Evidence",
                columns=[
                    TableColumn(key="arm", header="Arm"),
                    TableColumn(key="measure", header="Measure"),
                    TableColumn(key="value", header="Value"),
                    TableColumn(key="n", header="n"),
                    TableColumn(key="spread", header="Spread"),
                ],
                rows=rows,
                order="as the author listed the readings",
                total_rows=len(rows),
            )
        )
    if finding.chart.type != NO_CHART and resolution is not None:
        if resolution.chart is not None:
            blocks.append(_chart_block(analysis, position, resolution.chart))
        elif resolution.chart_note:
            # The author proposed a chart code could not draw; the finding was kept and this says why.
            blocks.append(
                DisclosureBlock(section="findings", finding=position, source="chart", text=resolution.chart_note)
            )
    blocks.extend(
        TextBlock(
            section="findings",
            role="caveat",
            finding=position,
            body=caveat.text,
            facts=[Fact(name="Kind", value=caveat.kind)],
        )
        for caveat in finding.caveats
    )
    if finding.durable.strip():
        blocks.append(TextBlock(section="findings", role="carried_forward", finding=position, body=finding.durable))
    return blocks


def _chart_block(analysis: EvalAnalysis, position: int, viz: Viz) -> ChartBlock:
    """A finding's stored chart, decided — or the reason this build cannot draw it.

    Served as a failure rather than raised on or omitted: the intent builders and their policy are code and
    move under a stored chart, and one chart this build refuses must not make the rest of a report
    unopenable — while omitting it would make a chart that cannot be drawn indistinguishable from a
    finding that never carried one.

    Args:
        analysis: The analysis, named in the log line.
        position: The finding's position.
        viz: The stored chart.

    Returns:
        The chart block.
    """
    try:
        return ChartBlock(
            section="findings", finding=position, viz_type=viz.type, intent=chart_intent(viz.type, viz.payload)
        )
    except (PayloadError, IntentPolicyError) as undrawable:
        log.warning(
            "eval analysis %s finding %d carries a %s chart that cannot be drawn: %s",
            analysis.id,
            position,
            viz.type,
            undrawable,
        )
        return ChartBlock(section="findings", finding=position, viz_type=viz.type, error=str(undrawable))


def _arm_blocks(analysis: EvalAnalysis) -> list[ReportBlock]:
    """The arm table — every arm the campaign observed and where it stands — and what the join could not place."""
    table = arm_table(analysis)
    rows: list[dict[str, Cell]] = [
        {
            "arm": f"{row.label} (control)" if row.is_control else row.label,
            "status": ARM_STATUS_WORDS[row.status],
            "findings": positions([int(finding) for finding in row.finding_ids]) or None,
        }
        for row in table.rows
    ]
    blocks: list[ReportBlock] = [
        TableBlock(
            section="arms",
            name="arms",
            title="Arms",
            columns=[
                TableColumn(key="arm", header="Arm"),
                TableColumn(key="status", header="Status"),
                TableColumn(key="findings", header="Rests on finding"),
            ],
            rows=rows,
            order="winner, then ruled out, then replaced incumbent, then unresolved",
            total_rows=len(rows),
        )
    ]
    if table.unplaced_coordinates:
        blocks.append(
            DisclosureBlock(
                section="arms",
                source="arms",
                text=(
                    "Evidence was read at coordinates no arm here holds, so it is in no row: "
                    + "; ".join(table.unplaced_coordinates)
                    + "."
                ),
            )
        )
    if table.unplaced_decision_cells:
        blocks.append(
            DisclosureBlock(
                section="arms",
                source="arms",
                text=(
                    "A decision names cells whose arm has no row here, so its verdict is not placed: "
                    + "; ".join(table.unplaced_decision_cells)
                    + "."
                ),
            )
        )
    return blocks


def _surface_blocks(analysis: EvalAnalysis) -> list[ReportBlock]:
    """The decision surface laid out, its provenance, and every bar no cell could be read against."""
    table = build_surface_table(analysis)
    blocks: list[ReportBlock] = []
    if table.disclosure is not None:
        blocks.append(DisclosureBlock(section="surface", source="surface", text=table.disclosure))
    else:
        blocks.append(DisclosureBlock(section="surface", source="surface", text=table.provenance))
        columns = [
            TableColumn(key="arm", header="Arm"),
            TableColumn(key="replication", header="Replication"),
            *(TableColumn(key=f"column_{index}", header=column.header) for index, column in enumerate(table.columns)),
            TableColumn(key="notes", header="Run notes"),
        ]
        rows: list[dict[str, Cell]] = []
        for row in table.rows:
            cells: dict[str, Cell] = {
                "arm": f"{row.label} (control)" if row.is_control else row.label,
                "replication": row.replication,
            }
            for index, value in enumerate(row.values):
                if value is None:
                    cells[f"column_{index}"] = None
                else:
                    cells[f"column_{index}"] = (
                        f"{value.text} — {value.verdict_word}" if value.verdict_word else value.text
                    )
            cells["notes"] = "; ".join(f"{note.kind} ({note.run_id}): {note.text}" for note in row.run_notes) or None
            rows.append(cells)
        blocks.append(
            TableBlock(
                section="surface",
                name="surface",
                title="Decision surface",
                columns=columns,
                rows=rows,
                order="the control's cells first, then every other cell by arm and rig",
                total_rows=len(rows),
            )
        )
    blocks.extend(
        DisclosureBlock(
            section="surface",
            source="surface",
            text=f"The {bar.source} bar on {bar.measure_id} could be read against no cell: {bar.reason}.",
        )
        for bar in table.unadjudicated_bars
    )
    return blocks


def _method_blocks(analysis: EvalAnalysis) -> list[ReportBlock]:
    """The analysis-wide disclosures no finding or table owns: the time axis's basis and the generation."""
    blocks: list[ReportBlock] = []
    axis = analysis.decision_surface.time_axis
    if axis is not None:
        if axis.basis == "release":
            text = f"The time axis is builds of {axis.release_label}, in the order each first ran."
        else:
            text = f"The time axis is UTC days, not builds: {axis.basis_reason}."
        blocks.append(DisclosureBlock(section="methods", source="time_axis", text=text))
    generation = analysis.generation
    blocks.append(
        DisclosureBlock(
            section="methods",
            source="generation",
            text=(
                f"Written by {generation.generator_model} with prompt {generation.prompt_id} "
                f"(version {generation.prompt_version}) on {generation.generated_at}, over the evidence bundle "
                f"{generation.bundle_fingerprint}. The evidence, charts and tables were computed by code from the "
                "decision surface frozen at generation; no number in them was written by the model."
            ),
        )
    )
    return blocks


__all__ = [
    "build_report",
]

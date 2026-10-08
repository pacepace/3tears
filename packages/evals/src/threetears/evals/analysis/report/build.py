"""Build a :class:`~threetears.evals.analysis.report.model.Report` — from a stored analysis, or from the evidence alone.

Derived on every read and never stored, for the reason the arm and surface tables are: a stored
analysis keeps its facts, and how they are laid out is a decision a later build may make better. The
order is the order a reader acts on: the summary, where each declared question stands, the decisions,
the findings with their evidence, charts and caveats, the arms, the decision surface, what to run next,
and the analysis-wide disclosures.

:func:`build_report` lays out a stored analysis. :func:`build_code_only_report` lays out a campaign's
assembled evidence when no analysis exists: the same arm and surface tables (through the same block
builders, so one arm reads alike on both), the contrasts code tested against the control, a chart per
measure the surface can draw, and every disclosure the evidence carries — and no text block, because
nobody wrote any.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from threetears.evals.analysis.agreement import tier_sentence
from threetears.evals.analysis.arms import ArmTable, arm_table, arm_table_of, short_digest
from threetears.evals.analysis.bundle import AnalysisContextBundle, bundle_decision_surface
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.references import cell_index, resolve_reading
from threetears.evals.analysis.report.model import (
    ChartBlock,
    DisclosureBlock,
    DisclosureSource,
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
    COMPARISON_VERDICT_WORDS,
    CONFIDENCE_WORDS,
    EVIDENCE_TIER_WORDS,
    arm_namer,
    positions,
)
from threetears.evals.analysis.surface_table import SurfaceTable, build_surface_table, surface_table_of
from threetears.evals.analysis.viz.intent import Cell, chart_intent
from threetears.evals.analysis.viz.payloads import PayloadError
from threetears.evals.analysis.viz.policy import IntentPolicyError
from threetears.evals.analysis.viz_refs import DistributionRef, build_viz_payload, cell_arm_labels
from threetears.evals.contracts.authored import NO_CHART, Finding
from threetears.evals.contracts.campaign import EvalAnalysis, FindingResolution, ReadingKind, Viz
from threetears.evals.contracts.host.measures import MeasureRegistry
from threetears.evals.contracts.surface import CellFacts, DecisionSurface, TimeAxis
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

    blocks.extend(_arm_blocks(arm_table(analysis)))
    blocks.extend(_surface_blocks(build_surface_table(analysis)))

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
        basis="analysis",
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


def _arm_blocks(table: ArmTable) -> list[ReportBlock]:
    """The arm table — every arm the campaign observed and where it stands — and what the join could not place.

    The one place an arm's full lever set is stated: every other block names the arm by its label, which
    carries only what tells it from the other arms.
    """
    rows: list[dict[str, Cell]] = [
        {
            "arm": f"{row.label} (control)" if row.is_control else row.label,
            "status": ARM_STATUS_WORDS[row.status],
            "findings": positions([int(finding) for finding in row.finding_ids]) or None,
            "levers": "; ".join(f"{level.axis_id}={level.display}" for level in row.settings) or None,
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
                TableColumn(key="levers", header="Every lever it ran"),
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


def _surface_blocks(table: SurfaceTable) -> list[ReportBlock]:
    """The decision surface laid out, its provenance, and every bar no cell could be read against."""
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
    blocks = _time_axis_blocks(analysis.decision_surface.time_axis)
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


def _time_axis_blocks(axis: TimeAxis | None) -> list[ReportBlock]:
    """What the time axis's positions are, when the runs span two builds or two days."""
    if axis is None:
        return []
    if axis.basis == "release":
        text = f"The time axis is builds of {axis.release_label}, in the order each first ran."
    else:
        text = f"The time axis is UTC days, not builds: {axis.basis_reason}."
    return [DisclosureBlock(section="methods", source="time_axis", text=text)]


# =============================================================================
# The code-only report — a campaign's evidence, with no analysis
# =============================================================================

#: What the code-only report says in place of an analysis — that none was generated, and what one adds.
NO_ANALYSIS = (
    "No analysis was generated for this campaign, so nothing in this report was written by a model or an "
    "analyst: there is no headline, no finding, no decision and no answer to a declared question. What follows "
    "is what code computed from the campaign's evidence — the arms, the decision surface, the contrasts tested "
    "against the control, a chart per measure, and every disclosure the evidence carries. An analysis would add "
    "a reading of it: findings with the evidence each rests on and the caveats that qualify it, a decision per "
    "declared question with its confidence, which arm won and why, and what to run next."
)


def build_code_only_report(bundle: AnalysisContextBundle, *, measures: MeasureRegistry, assembled_at: str) -> Report:
    """Lay a campaign's assembled evidence out as a report, when no analysis exists to report through.

    Everything here is code's: the arm table (every arm ``unresolved``, since a verdict is a decision's
    and none was made), the decision surface, the contrasts the bundle tested against the control, one
    distribution chart per measure and judged dimension the surface can draw, and every disclosure the
    bundle carries. No :class:`~threetears.evals.analysis.report.model.TextBlock` is built — the report
    model refuses one on a code-only report — and the first block says plainly that no analysis was
    generated and what one would add.

    Args:
        bundle: The campaign's evidence, as :func:`~threetears.evals.analysis.bundle.assemble_context_bundle`
            assembled it.
        measures: The host's measure registry, which a chart's payload reads.
        assembled_at: When the bundle was assembled (ISO-8601), stated as the report's ``generated_at``.

    Returns:
        The report, ``basis="code_only"``.
    """
    surface = bundle_decision_surface(bundle)
    variant_index = bundle.variant_index
    blocks: list[ReportBlock] = [DisclosureBlock(section="summary", source="generation", text=NO_ANALYSIS)]
    blocks.extend(_question_blocks(bundle))
    blocks.extend(
        _arm_blocks(
            arm_table_of(
                variant_index,
                control=surface.control_variant_key,
                decisions=(),
                resolutions=(),
                source=f"eval.campaign {bundle.campaign_id} (code-only report)",
            )
        )
    )
    blocks.extend(_surface_blocks(surface_table_of(surface, variant_index)))
    blocks.extend(_comparison_blocks(bundle, surface))
    blocks.extend(_measure_chart_blocks(surface, bundle, measures))
    blocks.extend(_time_axis_blocks(surface.time_axis))
    if bundle.time_axis_withheld:
        blocks.append(DisclosureBlock(section="methods", source="time_axis", text=bundle.time_axis_withheld))
    blocks.extend(_evidence_disclosures(bundle))
    fingerprint = bundle.fingerprint()
    blocks.append(
        DisclosureBlock(
            section="methods",
            source="generation",
            text=(
                f"Computed by code on {assembled_at} from the evidence bundle {fingerprint} over "
                f"{len(bundle.run_ids)} run(s). No model was called and no number here was written by one."
            ),
        )
    )
    return Report(
        basis="code_only",
        headline="",
        finding_count=0,
        source=ReportSource(
            campaign_id=bundle.campaign_id,
            scope_id=bundle.scope_id,
            subject_id=bundle.subject_id,
            subject_kind=bundle.subject_kind,
            behavior=bundle.behavior,
            generated_at=assembled_at,
            bundle_fingerprint=fingerprint,
        ),
        blocks=blocks,
    )


def _question_blocks(bundle: AnalysisContextBundle) -> list[ReportBlock]:
    """The declared questions, as the campaign declared them, each standing unanswered."""
    design = bundle.declared_design
    if design is None or not design.questions:
        return []
    rows: list[dict[str, Cell]] = [
        {
            "question": question.id,
            "asks": question.text,
            "axes": ", ".join(question.merit_axes) or None,
            "status": "retired" if question.retired_at else "live",
            "answer": "unanswered — no analysis",
        }
        for question in design.questions
    ]
    return [
        TableBlock(
            section="questions",
            name="questions",
            title="Declared questions",
            columns=[
                TableColumn(key="question", header="Question"),
                TableColumn(key="asks", header="Asks"),
                TableColumn(key="axes", header="Merit axes"),
                TableColumn(key="status", header="Status"),
                TableColumn(key="answer", header="Answer"),
            ],
            rows=rows,
            order="as the campaign declared them",
            total_rows=len(rows),
        )
    ]


def _comparison_blocks(bundle: AnalysisContextBundle, surface: DecisionSurface) -> list[ReportBlock]:
    """Each contrast the bundle tested against the control, per live question, and how the family was corrected."""
    comparisons = bundle.multiple_comparisons
    if not comparisons.families:
        if comparisons.withheld:
            return [DisclosureBlock(section="surface", source="comparisons", text=comparisons.withheld)]
        return []
    labels = cell_arm_labels(surface, bundle.variant_index)

    def arm(variant_key: str, apparatus_class_id: str) -> str:
        ref = cell_ref(variant_key, apparatus_class_id)
        return labels.get(ref) or f"{short_digest(variant_key)} (rig {short_digest(apparatus_class_id)})"

    rows: list[dict[str, Cell]] = []
    for family in comparisons.families:
        for comparison in family.comparisons:
            reading = f"{comparison.name} (judged)" if comparison.reading == "judged" else comparison.name
            rows.append(
                {
                    "question": family.question_id if family.question_id is not None else "(campaign-wide)",
                    "reading": reading,
                    "contrast": arm(comparison.contrast.variant_key, comparison.contrast.apparatus_class_id),
                    "control": arm(comparison.control.variant_key, comparison.control.apparatus_class_id),
                    "delta": comparison.delta,
                    "p_adjusted": comparison.p_adjusted,
                    "verdict": COMPARISON_VERDICT_WORDS[comparison.verdict]
                    + (f" ({comparison.untested_reason})" if comparison.untested_reason else "")
                    + (
                        " — immaterial: below the host's materiality threshold, too small to act on"
                        if comparison.materiality == "immaterial"
                        else ""
                    ),
                }
            )
    blocks: list[ReportBlock] = [
        TableBlock(
            section="surface",
            name="comparisons",
            title="Contrasts against the control",
            columns=[
                TableColumn(key="question", header="Question"),
                TableColumn(key="reading", header="Reading"),
                TableColumn(key="contrast", header="Arm"),
                TableColumn(key="control", header="Control"),
                TableColumn(key="delta", header="Delta (arm − control)"),
                TableColumn(key="p_adjusted", header="p (Holm-adjusted)"),
                TableColumn(key="verdict", header="Verdict"),
            ],
            rows=rows,
            order="by declared question, then as the bundle tested them",
            total_rows=len(rows),
        )
    ]
    blocks.extend(
        DisclosureBlock(section="surface", source="comparisons", text=family.disclosure)
        for family in comparisons.families
    )
    return blocks


def _measure_chart_blocks(
    surface: DecisionSurface, bundle: AnalysisContextBundle, measures: MeasureRegistry
) -> list[ReportBlock]:
    """One distribution chart per numeric measure and judged dimension — the charts code can choose on its own.

    A distribution is the one chart that reads a single reading across every cell with no author's choice
    of which cells to set against which, so it is the chart code picks. A cell is drawn where its reading
    has an interval; a cell left out, and a reading no cell can draw, are each disclosed rather than
    dropped — and a chart the presentation rules refuse is disclosed with the reason.
    """
    readings: list[tuple[str, ReadingKind]] = [
        *((name, "measure") for name, facts in sorted(surface.measures.items()) if facts.higher_is_better is not None),
        *((name, "judged") for name in sorted(surface.dimensions)),
    ]
    cells = cell_index(surface)
    labels = cell_arm_labels(surface, bundle.variant_index)
    blocks: list[ReportBlock] = []
    for name, reading in readings:
        drawn: list[str] = []
        left_out: list[str] = []
        for ref, cell in cells.items():
            if not _measured(cell, name, reading):
                continue  # Not a cell's reading to leave out: it never measured this one.
            try:
                resolved = resolve_reading(surface, ref, name, reading)
            except UnresolvableReference:
                left_out.append(ref)  # Measured, and nothing scored — disclosed below with the rest.
                continue
            (drawn if resolved.ci_low is not None and resolved.ci_high is not None else left_out).append(ref)
        what = f"{name} (judged)" if reading == "judged" else name
        if left_out:
            blocks.append(
                DisclosureBlock(
                    section="surface",
                    source="chart",
                    text=(
                        f"The {what} chart leaves out {len(left_out)} cell(s) with no interval to draw — fewer than two "
                        "observations of it, or none scored: "
                        + "; ".join(labels.get(ref, ref) for ref in left_out)
                        + "."
                    ),
                )
            )
        if not drawn:
            # Every cell that measured it was left out, which the disclosure above already says — or no
            # cell measured it at all, and there is nothing to chart and nothing withheld.
            continue
        try:
            payload = build_viz_payload(
                DistributionRef(cells=drawn, measure_id=name, reading=reading),
                surface,
                bundle.variant_index,
                measures=measures,
            )
            blocks.append(
                ChartBlock(section="surface", viz_type="distribution", intent=chart_intent("distribution", payload))
            )
        except (UnresolvableReference, PayloadError, IntentPolicyError) as undrawable:
            blocks.append(
                DisclosureBlock(
                    section="surface", source="chart", text=f"The {what} chart cannot be drawn: {undrawable}."
                )
            )
    return blocks


def _measured(cell: CellFacts, name: str, reading: ReadingKind) -> bool:
    """Whether a cell measured a reading at all — a cell that never did has nothing to leave out of its chart."""
    if reading == "judged":
        return any(judged.dimension == name for judged in cell.judged)
    return any(summary.name == name for summary in cell.measures.measures)


def _listed(ids: Sequence[str]) -> str:
    """Ids, as a disclosure lists them."""
    return ", ".join(ids)


def _verdict_order_sentence(bundle: AnalysisContextBundle) -> str | None:
    """The tie-break order the campaign declared over its merit axes, with the bars each tier holds — or nothing.

    The writer is told this order; a reader of a code-only report, which has no writer, would otherwise never
    see the priority the campaign stated.
    """
    order = bundle.verdict_order
    if not order.merit_priority:
        return None
    tiers = "; ".join(
        f"{tier.axis} ({_listed(tier.bar_measure_ids) if tier.bar_measure_ids else 'no adjudicated bar'})"
        for tier in order.tiers
    )
    unranked = (
        f" Bars on no ranked axis, which the order does not place: {_listed(order.unranked_bar_measure_ids)}."
        if order.unranked_bar_measure_ids
        else ""
    )
    return f"The campaign ranks its merit axes, strongest first, to break a tie no bar decides: {tiers}.{unranked}"


def _evidence_disclosures(bundle: AnalysisContextBundle) -> list[ReportBlock]:
    """Every disclosure the evidence bundle carries about its runs, how they were measured, and the rig."""
    blocks: list[ReportBlock] = []

    def say(source: DisclosureSource, text: str | None) -> None:
        if text and text.strip():
            blocks.append(DisclosureBlock(section="methods", source=source, text=text.strip()))

    if not bundle.run_ids:
        say("runs", "The campaign resolved no runs in this scope, so no table or chart here holds anything.")
    if bundle.unresolved_run_ids:
        say(
            "runs",
            f"{len(bundle.unresolved_run_ids)} run(s) the campaign names were not found in this scope and are in no "
            f"table here: {_listed(bundle.unresolved_run_ids)}.",
        )
    if bundle.archived_run_ids:
        say(
            "runs",
            f"{len(bundle.archived_run_ids)} member run(s) were archived and are held out of everything here: "
            f"{_listed(bundle.archived_run_ids)}.",
        )
    if bundle.incomplete_runs:
        say(
            "runs",
            f"{len(bundle.incomplete_runs)} run(s) did not complete and are pooled as far as they got: "
            + "; ".join(f"{run_id} ({status})" for run_id, status in sorted(bundle.incomplete_runs.items()))
            + ".",
        )
    for _, sentence in sorted(bundle.short_runs.items()):
        say("runs", sentence)
    if bundle.completeness_unknown_run_ids:
        say(
            "runs",
            f"Whether {len(bundle.completeness_unknown_run_ids)} run(s) came up short is unknown — they carry no "
            f"completeness record: {_listed(bundle.completeness_unknown_run_ids)}.",
        )
    for cell in bundle.short_cells:
        say("runs", cell.sentence)
    say("surface", _verdict_order_sentence(bundle))
    say("measurement", bundle.launch_disclosure)
    say("measurement", bundle.measurement_window_disclosure)
    for instability in bundle.subject_key_instabilities:
        if instability.kind == "one_key_many_labels":
            text = f"Subject key {instability.key} is labelled {len(instability.counterparts)} ways: "
        else:
            text = f"Subject label {instability.label} names {len(instability.counterparts)} keys: "
        say("measurement", text + _listed(instability.counterparts) + ".")
    # Every judge's tier on every judged dimension, with the two measurements that decided it — flagged, so
    # a judged number is never read without what it can bear (PD-13).
    for tier in bundle.judge_evidence_tiers:
        say("measurement", "Judged evidence tier: " + tier_sentence(tier))
    say("apparatus", bundle.controls_reading.disclosure)
    if bundle.apparatus_confounds:
        say(
            "apparatus",
            "The rig did not hold still across the campaign: "
            + "; ".join(f"{confound.dimension} ({confound.status})" for confound in bundle.apparatus_confounds)
            + ".",
        )
    for merge in bundle.refused_merges:
        dimensions = f" on {_listed(merge.dimensions)}" if merge.dimensions else ""
        say(
            "apparatus",
            f"Arm {short_digest(merge.variant_key)} was measured in {len(merge.apparatus_class_ids)} cells that did "
            f"not pool ({merge.reason.replace('_', ' ')}{dimensions}).",
        )
    return blocks


__all__ = [
    "NO_ANALYSIS",
    "build_code_only_report",
    "build_report",
]

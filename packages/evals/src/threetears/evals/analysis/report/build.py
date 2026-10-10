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
from itertools import chain

from threetears.evals.analysis.agreement import tier_sentence
from threetears.evals.contracts.models import CHECK_REFUSED_UNDER_CURRENT_GRAMMAR
from threetears.evals.analysis.arms import ArmTable, arm_names, arm_table, arm_table_of, short_digest, surface_order
from threetears.evals.analysis.bundle import (
    AnalysisContextBundle,
    FamilyComparison,
    GoalCheckProofReading,
    bundle_decision_surface,
    exploratory_disclosure,
)
from threetears.evals.analysis.cells import cell_ref, variant_of_cell_ref
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
    Verdict,
    VerdictReason,
)
from threetears.evals.analysis.report.words import (
    ARM_STATUS_WORDS,
    BAR_DECISION_WORDS,
    COMPARISON_VERDICT_WORDS,
    CONFIDENCE_WORDS,
    EVIDENCE_COLUMNS,
    GUARDRAIL_DECISION_WORDS,
    arm_namer,
    evidence_rows,
    positions,
    stands_on_words,
)
from threetears.evals.analysis.surface_table import (
    NO_SUCCESSFUL_RESULTS,
    REFERENCE_MARK,
    SurfaceTable,
    build_surface_table,
    surface_order_sentence,
    surface_table_of,
)
from threetears.evals.analysis.viz.intent import Cell, chart_intent
from threetears.evals.analysis.viz.payloads import PayloadError
from threetears.evals.analysis.viz.policy import IntentPolicyError
from threetears.evals.analysis.viz_refs import DistributionRef, build_viz_payload, cell_arm_labels
from threetears.evals.contracts.authored import NO_CHART, Finding
from threetears.evals.contracts.campaign import EvalAnalysis, EvidenceRow, FindingResolution, ReadingKind, Viz
from threetears.evals.contracts.host.measures import MeasureRegistry
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.stats import EQUIVALENCE_NEEDS_RANGE, INTERVAL_LEVEL
from threetears.evals.analysis.viz.quantities import display_scale, with_unit
from threetears.evals.contracts.analysis_measures import MeasureSummary
from threetears.evals.contracts.campaign import VariantIndexEntry
from threetears.evals.contracts.metrics import (
    ACCURACY_MEASURE,
    MATCH_MEASURE,
    ClassifierStatistic,
    classifier_label_of,
)
from threetears.evals.contracts.declaration import JUDGED_MERIT_AXIS, CampaignDesign, Question, exploratory_reading
from threetears.evals.contracts.surface import (
    STRATUM_MIN_CASES,
    CellFacts,
    DecisionSurface,
    GuardrailReadings,
    JudgedReading,
    MeasureFacts,
    StratumFacts,
    TimeAxis,
)
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
    surface = analysis.decision_surface
    questions = analysis.design_snapshot.live_questions() if analysis.design_snapshot is not None else []
    blocks: list[ReportBlock] = []
    if document.summary.strip():
        blocks.append(TextBlock(section="summary", role="summary", body=document.summary))
    if (exploratory := exploratory_disclosure(analysis.design_snapshot)) is not None:
        # Said once, at the top, rather than on every finding: a label on every row is one readers skip. A
        # campaign that declared no design (`design_snapshot` None) is exploratory, and says so as such.
        blocks.append(DisclosureBlock(section="summary", source="scope", text=exploratory))

    blocks.extend(
        TextBlock(
            section="questions",
            role="answer",
            body=answer.answer,
            rests_on=list(answer.rests_on),
            facts=[
                Fact(name="Question", value=_question_words(analysis.design_snapshot, answer.question_id)),
                Fact(name="Resolution", value=answer.resolution),
            ],
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
        if decision.disposition == "adopted" and (standing := _guardrail_standing(decision.cells, surface, arm)):
            facts.append(Fact(name="Guardrails", value=standing))
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

    cell_name = _cell_namer(surface, analysis.variant_index)
    guardrails_read = guardrail_verdicts(surface, surface.guardrails, cell_name)
    blocks.extend(_guardrail_blocks(surface, surface.guardrails, cell_name, guardrails_read))

    resolutions: list[FindingResolution | None] = (
        list(analysis.resolutions) if analysis.resolutions else [None] * len(document.findings)
    )
    for position, (finding, resolution) in enumerate(zip(document.findings, resolutions, strict=True)):
        blocks.extend(
            _finding_blocks(
                analysis,
                position,
                finding,
                resolution,
                arm,
                exploratory=bool(questions) and _rests_on_exploratory(resolution, surface, questions),
            )
        )

    blocks.extend(_arm_blocks(arm_table(analysis)))
    # Compiled from the frozen surface by the code-only report's own compiler, with no model in the loop, and
    # ahead of the table: the chart is the reading form, the table the audit form (#643). A stored analysis
    # carries no host registry, and a distribution reads none.
    blocks.extend(_surface_chart_blocks(surface, analysis.variant_index, MeasureRegistry([])))
    blocks.extend(_surface_blocks(build_surface_table(analysis)))
    blocks.extend(_strata_blocks(analysis.decision_surface, analysis.variant_index))

    blocks.extend(_coverage_blocks(analysis))
    coverage_status = {row.name: row.status for row in analysis.coverage.levers}
    for step in document.next:
        facts = [Fact(name="Leverage", value=step.leverage)]
        if step.lever:
            # Joined on the stored lever name; a lever with no coverage row is proposed, not a gap, and says so.
            status = coverage_status.get(step.lever)
            facts.append(Fact(name="Lever", value=f"{step.lever} ({status or 'no coverage row'})"))
        blocks.append(TextBlock(section="next", role="next_step", body=step.title, facts=facts))
        if step.why.strip():
            blocks.append(TextBlock(section="next", role="next_step_why", body=step.why))

    blocks.extend(_method_blocks(analysis))
    blocks.extend(_unfound_lever_blocks(analysis))
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
        verdicts=[*guardrails_read, *bar_verdicts(surface, cell_name)],
    )


def _levers_without_finding(analysis: EvalAnalysis) -> list[str]:
    """The levers in the coverage map that no finding names in its ``axes`` — the map's side of the join (#631).

    Computed from the stored fields on read, never by matching prose: a lever is named when it is, exactly, one
    of a finding's ``axes``. A measured lever with no finding is not a defect (its arms may not have differed);
    this is a count of the writer's own output, which a reader sees and nothing acts on.

    Args:
        analysis: The analysis.

    Returns:
        The unnamed levers, in coverage order.
    """
    named = {axis for finding in analysis.document.findings for axis in finding.axes}
    return [row.name for row in analysis.coverage.levers if row.name not in named]


def _coverage_blocks(analysis: EvalAnalysis) -> list[ReportBlock]:
    """The coverage map joined to the findings and the next steps, on the stored lever names (#630, #631).

    One row per lever: its status, the findings naming it in their ``axes`` (or "no finding"), and the next
    steps whose ``lever`` names it. A ``thin`` or ``unswept`` lever no step names says so, rather than leaving
    the slot empty; a step naming a lever with no coverage row renders under the steps, unattached.
    """
    levers = analysis.coverage.levers
    if not levers:
        return []
    findings = analysis.document.findings
    rows: list[dict[str, Cell]] = []
    for row in levers:
        named = [position for position, finding in enumerate(findings) if row.name in finding.axes]
        steps = [step.title for step in analysis.document.next if step.lever == row.name]
        gap = row.status in ("thin", "unswept")
        rows.append(
            {
                "lever": row.name,
                "status": row.status,
                "findings": positions(named) or "no finding",
                "next": "; ".join(steps) or ("no next step names it" if gap else None),
            }
        )
    return [
        TableBlock(
            section="next",
            name="coverage",
            title="Coverage: each lever, the findings naming it, and the next step that would measure it",
            columns=[
                TableColumn(key="lever", header="Lever"),
                TableColumn(key="status", header="Coverage"),
                TableColumn(key="findings", header="Named by finding"),
                TableColumn(key="next", header="Next step naming it"),
            ],
            rows=rows,
            order="as the coverage map lists the levers",
            total_rows=len(rows),
        )
    ]


def _unfound_lever_blocks(analysis: EvalAnalysis) -> list[ReportBlock]:
    """The methods count of coverage levers no finding names (#631): said, never acted on."""
    levers = analysis.coverage.levers
    if not levers:
        return []
    unnamed = _levers_without_finding(analysis)
    text = (
        f"{len(unnamed)} of the {len(levers)} lever(s) in the coverage map are named by no finding: {_listed(unnamed)}."
        if unnamed
        else f"Every one of the {len(levers)} lever(s) in the coverage map is named by a finding."
    )
    return [DisclosureBlock(section="methods", source="surface", text=text)]


def _unproven_check_sentence(proof: GoalCheckProofReading) -> str:
    """The disclosure for a goal check not shown to beat doing nothing: what it is, and what its pass rate is not."""
    if proof.refused is not None:
        return (
            f"Goal check {proof.check} is {CHECK_REFUSED_UNDER_CURRENT_GRAMMAR} ({proof.refused}), so "
            f"{proof.runs} run(s) graded it on no cell; where it has a pass rate ({proof.measure_id}) that rate "
            "comes from runs launched before the rule, and is not shown to measure the behaviour."
        )
    why = (
        "its control does not show it tells acting from doing nothing"
        if proof.proof == "refuted"
        else "no control shows it tells acting from doing nothing"
    )
    if proof.unrecorded:
        why += f" ({proof.unrecorded} of {proof.runs} run(s) launched before proofs were recorded)"
    if proof.stale:
        why += f" ({proof.stale} of {proof.runs} run(s) recorded it proven under an earlier proof rule; relaunch to re-prove)"
    return (
        f"Goal check {proof.check} is {proof.proof}: {why}. Its pass rate ({proof.measure_id}) may be what a "
        "candidate that did nothing would score, so it is not shown to measure the behaviour."
    )


def _finding_blocks(
    analysis: EvalAnalysis,
    position: int,
    finding: Finding,
    resolution: FindingResolution | None,
    arm: Callable[[str], str],
    *,
    exploratory: bool = False,
) -> list[ReportBlock]:
    """One finding: its title and the facts beside it, its body, its evidence, its chart, its caveats.

    Args:
        analysis: The analysis, named in a log line when a stored chart will not draw.
        position: The finding's position in the document.
        finding: The authored finding.
        resolution: What code filled for it, or None for an analysis with no resolutions.
        arm: Names the arm a cell reference points at.
        exploratory: The finding rests wholly on readings no declared question asked about, so it carries
            the ``Scope`` fact saying so.

    Returns:
        The finding's blocks, in reading order.
    """
    facts = [Fact(name="Confidence", value=CONFIDENCE_WORDS[finding.confidence])]
    if exploratory:
        facts.append(Fact(name="Scope", value=EXPLORATORY_FINDING))
    if resolution is not None:
        facts.append(Fact(name="Stands on", value=stands_on_words(analysis, resolution.evidence_tier)))
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
        if resolution.chart is None:
            # The evidence compiled to the chart that leads it, where it compares arms and the author's own
            # chart did not draw: a finding already carrying a drawn chart is not given a second one.
            blocks.extend(_evidence_chart_blocks(analysis, position, resolution.evidence))
        rows = evidence_rows(analysis, resolution.evidence, arm)
        blocks.append(
            TableBlock(
                section="findings",
                finding=position,
                name="evidence",
                title="Evidence",
                columns=[TableColumn(key=key, header=header) for key, header in EVIDENCE_COLUMNS],
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


def _evidence_chart_blocks(analysis: EvalAnalysis, position: int, evidence: Sequence[EvidenceRow]) -> list[ReportBlock]:
    """A finding's evidence table compiled to charts: one distribution per reading it cites at two or more cells.

    The same compiler the surface charts take, so a cited reading is drawn as the surface draws it. A reading
    cited at one cell compares nothing and stays a row; a chart this build cannot draw is disclosed with the
    reason rather than dropped (#643).

    Args:
        analysis: The analysis, whose frozen surface every number is read from.
        position: The finding's position.
        evidence: The finding's resolved rows.

    Returns:
        The chart blocks, in the order the readings were first cited.
    """
    cited: dict[tuple[str, ReadingKind], list[str]] = {}
    for row in evidence:
        cells = cited.setdefault((row.measure_id, row.reading), [])
        if row.cell_ref not in cells:
            cells.append(row.cell_ref)
    blocks: list[ReportBlock] = []
    for (name, reading), cells in cited.items():
        if len(cells) < 2:
            continue
        try:
            payload = build_viz_payload(
                DistributionRef(cells=cells, measure_id=name, reading=reading),
                analysis.decision_surface,
                list(analysis.variant_index),
                measures=MeasureRegistry([]),
            )
            blocks.append(
                ChartBlock(
                    section="findings",
                    finding=position,
                    viz_type="distribution",
                    intent=chart_intent("distribution", payload),
                )
            )
        except (UnresolvableReference, PayloadError, IntentPolicyError) as undrawable:
            what = f"{name} (judged)" if reading == "judged" else name
            blocks.append(
                DisclosureBlock(
                    section="findings",
                    finding=position,
                    source="chart",
                    text=f"The {what} evidence chart cannot be drawn: {undrawable}.",
                )
            )
    return blocks


#: The ``Scope`` fact on a finding resting wholly on readings no declared question asked about.
EXPLORATORY_FINDING = (
    "exploratory: it rests on no reading a declared question asked about, so it is a lead, not an answer"
)

#: How a guardrail is decided, said once beside the guardrails table.
_GUARDRAILS_DISCLOSURE = (
    "A guardrail is what an arm must not get worse on. Each is decided on its own 95% interval on the difference "
    "from the control: held when the whole interval lies on the good side of the margin, breached when it lies "
    "wholly beyond it, undecided otherwise. On a declared range the interval is the bounded test's, which holds "
    "its error rate at any number of cases; a reading with no declared range is never held, since no test holds "
    "its rate there, and can only be shown breached. A measure's margin is its declared materiality threshold, a judged "
    "dimension's the one its campaign declares; with none, a guardrail is held at zero change. Guardrails join no comparison and no composite, so no "
    "gain elsewhere offsets one; an arm with a breached guardrail is not adopted, and an undecided guardrail is "
    "not known to be safe."
)


def _rests_on_exploratory(
    resolution: FindingResolution | None, surface: DecisionSurface, questions: Sequence[Question]
) -> bool:
    """Whether a finding's every reading lies outside the declared questions — none of them asked about it.

    A finding citing no reading (a point about coverage or the rig) is not labelled; a guardrail reading is
    never exploratory, since it is held because it was declared one. A reading the surface cannot describe is
    read as on no axis, so it is labelled rather than passed as confirmed.
    """
    if resolution is None or not resolution.evidence:
        return False
    for row in resolution.evidence:
        if row.reading == "judged":
            dimension = surface.dimensions.get(row.measure_id)
            if dimension is not None and dimension.axis == "boundary":
                return False
            if not exploratory_reading(JUDGED_MERIT_AXIS, questions):
                return False
        else:
            measure = surface.measures.get(row.measure_id)
            if measure is not None and measure.guardrail:
                return False
            if not exploratory_reading(measure.merit_axis if measure is not None else None, questions):
                return False
    return True


def _guardrail_standing(cells: Sequence[str], surface: DecisionSurface, arm: Callable[[str], str]) -> str | None:
    """What an adopted decision's arms leave not held among the guardrails, in words; None when every one held.

    Stated by code on the decision, so an arm recommended over an undecided guardrail cannot be read as
    shown safe. None too on a surface frozen before guardrails were decided, which says nothing either way.
    """
    guardrails = surface.guardrails
    if guardrails is None:
        return None
    parts = []
    for cell in cells:
        variant = variant_of_cell_ref(cell)
        if variant is None or variant == surface.control_variant_key:
            continue
        standing = guardrails.of_arm(variant)
        said = []
        if standing.breached:
            said.append(f"breached {', '.join(surface.measure_heading(name) for name in standing.breached)}")
        if standing.undecided:
            undecided = ", ".join(surface.measure_heading(name) for name in standing.undecided)
            said.append(f"undecided on {undecided}, so not known to be safe")
        if said:
            parts.append(f"{arm(cell)}: {'; '.join(said)}")
    return " | ".join(parts) or None


def _question_words(design: CampaignDesign | None, question_id: str) -> str:
    """A declared question in the words it was asked, or its id where the design holds none."""
    return design.question_words(question_id) if design is not None else question_id


def _cell_namer(surface: DecisionSurface, variant_index: Sequence[VariantIndexEntry]) -> Callable[[str, str], str]:
    """Name a cell by its two coordinates as the charts do, or by short digests where no label exists."""
    labels = cell_arm_labels(surface, list(variant_index))

    def name(variant_key: str, apparatus_class_id: str) -> str:
        ref = cell_ref(variant_key, apparatus_class_id)
        return labels.get(ref) or f"{short_digest(variant_key)} (rig {short_digest(apparatus_class_id)})"

    return name


def guardrail_verdicts(
    surface: DecisionSurface, guardrails: GuardrailReadings | None, arm: Callable[[str, str], str]
) -> list[Verdict]:
    """Each guardrail check as a typed verdict, in the guardrails table's row order; the table prints their words."""
    if guardrails is None or guardrails.withheld is not None:
        return []
    verdicts = []
    for check in guardrails.checks:
        reason: VerdictReason = (
            "within_margin"
            if check.decision == "held"
            else "beyond_margin"
            if check.decision == "breached"
            else "no_interval"
            if check.interval is None
            else "interval_straddles"
        )
        verdicts.append(
            Verdict(
                kind="guardrail",
                outcome=check.decision,
                reason=reason,
                reason_detail=check.undecided_reason,
                reading=check.reading,
                name=check.name,
                heading=surface.measure_heading(check.name, check.reading),
                arm=arm(check.contrast.variant_key, check.contrast.apparatus_class_id),
                variant_key=check.contrast.variant_key,
                apparatus_class_id=check.contrast.apparatus_class_id,
                control=arm(check.control.variant_key, check.control.apparatus_class_id),
                delta=check.delta,
                interval=check.interval,
                interval_level=INTERVAL_LEVEL if check.interval is not None else None,
                margin=check.margin,
                margin_source=(
                    None if not check.margin_declared else "campaign" if check.reading == "judged" else "measure"
                ),
                guardrail=True,
                words=GUARDRAIL_DECISION_WORDS[check.decision]
                + (f" ({check.undecided_reason})" if check.undecided_reason else ""),
            )
        )
    return verdicts


def bar_verdicts(surface: DecisionSurface, arm: Callable[[str, str], str]) -> list[Verdict]:
    """Each adjudicated bar's reading on each cell as a typed verdict, bar by bar in the surface's order."""
    reasons: dict[str, VerdictReason] = {
        "cleared": "interval_clears",
        "missed": "interval_misses",
        "undecided": "interval_straddles",
        "no_interval": "too_few_observations",
        "no_data": "no_observations",
    }
    verdicts = []
    for bar in surface.bars:
        if bar.state != "adjudicated":
            continue
        reading: ReadingKind = "judged" if bar.measure_id in surface.dimensions else "measure"
        for read in bar.verdicts:
            interval = (read.ci_low, read.ci_high) if read.ci_low is not None and read.ci_high is not None else None
            verdicts.append(
                Verdict(
                    kind="bar",
                    outcome=read.decision,
                    reason=reasons[read.decision],
                    reading=reading,
                    name=bar.measure_id,
                    heading=surface.measure_heading(bar.measure_id, reading),
                    arm=arm(read.variant_key, read.apparatus_class_id),
                    variant_key=read.variant_key,
                    apparatus_class_id=read.apparatus_class_id,
                    interval=interval,
                    value=read.value,
                    threshold=bar.threshold,
                    margin=read.margin,
                    margin_source="measure" if read.margin is not None else None,
                    words=BAR_DECISION_WORDS[read.decision],
                )
            )
    return verdicts


def _guardrail_blocks(
    surface: DecisionSurface,
    guardrails: GuardrailReadings | None,
    arm: Callable[[str, str], str],
    verdicts: Sequence[Verdict],
) -> list[ReportBlock]:
    """The guardrails section: each guardrail, held, breached or undecided for each arm against the control.

    ``verdicts`` are :func:`guardrail_verdicts` over the same readings, row for row; each row's decision is its
    verdict's words.
    """
    if guardrails is None:
        return []
    blocks: list[ReportBlock] = []
    if guardrails.withheld is not None:
        blocks.append(DisclosureBlock(section="guardrails", source="guardrails", text=guardrails.withheld))
    elif guardrails.checks:
        rows: list[dict[str, Cell]] = [
            {
                "guardrail": verdict.heading,
                "arm": verdict.arm,
                "control_mean": check.control.mean,
                "arm_mean": check.contrast.mean,
                "cases": (
                    f"{check.control.n_cases} paired"
                    if check.test == "paired"
                    else f"{check.contrast.n_cases} vs {check.control.n_cases}"
                ),
                "delta": check.delta,
                "interval": None
                if check.interval is None
                else f"[{format_number(check.interval[0])}, {format_number(check.interval[1])}] at "
                f"{format_number(100 * INTERVAL_LEVEL)}%"
                + (" (t: no declared range)" if check.interval_basis == "t" else ""),
                "margin": format_number(check.margin) if check.margin_declared else "0 (none declared)",
                "decision": verdict.words,
            }
            for check, verdict in zip(guardrails.checks, verdicts, strict=True)
        ]
        blocks.append(
            TableBlock(
                section="guardrails",
                name="guardrails",
                title="Guardrails against the control",
                columns=[
                    TableColumn(key="guardrail", header="Guardrail"),
                    TableColumn(key="arm", header="Arm"),
                    TableColumn(key="control_mean", header="Control mean"),
                    TableColumn(key="arm_mean", header="Arm mean"),
                    TableColumn(key="cases", header="Cases read"),
                    TableColumn(key="delta", header="Delta (arm − control)"),
                    TableColumn(key="interval", header="Interval on delta"),
                    TableColumn(key="margin", header="Margin"),
                    TableColumn(key="decision", header="Decision"),
                ],
                rows=rows,
                order="by guardrail, then rig, then arm",
                total_rows=len(rows),
            )
        )
        blocks.append(DisclosureBlock(section="guardrails", source="guardrails", text=_GUARDRAILS_DISCLOSURE))
    if guardrails.unstamped_dimensions:
        blocks.append(
            DisclosureBlock(
                section="guardrails",
                source="guardrails",
                text=(
                    f"{_listed(guardrails.unstamped_dimensions)} carry scores judged before the rubric axis was "
                    "recorded. They are read as capability, as they were then; a guardrail among them is not "
                    "recognised until it is re-judged."
                ),
            )
        )
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

    The ``Status`` column is there only when some arm has a status other than ``unresolved``, and the ``Rests
    on finding`` column only when some arm rests on a finding: on a code-only report nothing decided and
    there are no findings, so each would be the same word or an em dash in every row. With no status column
    the rows are in the arms' order alone, and the caption says so rather than naming an order by status.
    """
    decided = any(row.status != "unresolved" for row in table.rows)
    founded = any(row.finding_ids for row in table.rows)
    rows: list[dict[str, Cell]] = []
    for row in table.rows:
        cells: dict[str, Cell] = {"arm": f"{row.label} (control)" if row.is_control else row.label}
        if decided:
            cells["status"] = ARM_STATUS_WORDS[row.status]
        if founded:
            cells["findings"] = positions([int(finding) for finding in row.finding_ids]) or None
        cells["levers"] = "; ".join(f"{level.axis_id}={level.display}" for level in row.settings) or None
        rows.append(cells)
    blocks: list[ReportBlock] = [
        TableBlock(
            section="arms",
            name="arms",
            title="Arms",
            columns=[
                TableColumn(key="arm", header="Arm"),
                *([TableColumn(key="status", header="Status")] if decided else []),
                *([TableColumn(key="findings", header="Rests on finding")] if founded else []),
                TableColumn(key="levers", header="Every lever it ran"),
            ],
            rows=rows,
            order=(
                "winner, then contradicted, then ruled out, then replaced incumbent, then unresolved"
                if decided
                else "by arm"
            ),
            total_rows=len(rows),
        )
    ]
    if contradicted := [row for row in table.rows if row.status == "contradicted"]:
        blocks.append(
            DisclosureBlock(
                section="arms",
                source="arms",
                text=(
                    "This analysis both adopts and rejects "
                    + "; ".join(row.label for row in contradicted)
                    + ", so neither verdict stands and no winner is shown for "
                    + ("it" if len(contradicted) == 1 else "them")
                    + "."
                ),
            )
        )
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


def _surface_blocks(table: SurfaceTable, *, provenance: bool = True) -> list[ReportBlock]:
    """The decision surface laid out, its provenance, its all-failed arms, and every bar no cell could be read against.

    An all-failed arm is one where no result took a turn. The all-failed sentence follows the table directly, on the analysis's report and the code-only one alike:
    it is the table's own (``all_failed_disclosure``), derived from the frozen surface, so a stored analysis
    says it without the bundle it was written from.

    The ``Run notes`` column is there only when some row has a note: a column of em dashes says nothing the
    table's having no such column does not. ``provenance=False`` leaves out the sentence saying no number here
    was written by the model — on a code-only report, whose opening line already says code computed everything
    and that no model wrote any of it.
    """
    blocks: list[ReportBlock] = []
    if table.disclosure is not None:
        blocks.append(DisclosureBlock(section="surface", source="surface", text=table.disclosure))
    else:
        if provenance:
            blocks.append(DisclosureBlock(section="surface", source="surface", text=table.provenance))
        noted = any(row.run_notes for row in table.rows)
        columns = [
            TableColumn(key="arm", header="Arm"),
            TableColumn(key="replication", header="Replication"),
            *(TableColumn(key=f"column_{index}", header=column.header) for index, column in enumerate(table.columns)),
            *([TableColumn(key="notes", header="Run notes")] if noted else []),
        ]
        rows: list[dict[str, Cell]] = []
        for row in table.rows:
            cells: dict[str, Cell] = {
                "arm": f"{row.label} {REFERENCE_MARK}" if row.is_control else row.label,
                "replication": row.replication,
            }
            for index, value in enumerate(row.values):
                if value is None:
                    cells[f"column_{index}"] = None
                else:
                    cells[f"column_{index}"] = (
                        f"{value.text} — {value.verdict_word}" if value.verdict_word else value.text
                    )
            if noted:
                cells["notes"] = (
                    "; ".join(f"{note.kind} ({note.run_id}): {note.text}" for note in row.run_notes) or None
                )
            rows.append(cells)
        blocks.append(
            TableBlock(
                section="surface",
                name="surface",
                title="Decision surface",
                columns=columns,
                rows=rows,
                order=table.order,
                total_rows=len(rows),
            )
        )
        if table.all_failed_disclosure is not None:
            blocks.append(DisclosureBlock(section="surface", source="surface", text=table.all_failed_disclosure))
    blocks.extend(
        DisclosureBlock(
            section="surface",
            source="surface",
            text=f"The {bar.source} bar on {bar.measure_id} could be read against no cell: {bar.reason}.",
        )
        for bar in table.unadjudicated_bars
    )
    return blocks


# =============================================================================
# Results by stratum — each cell read again per kind of case
# =============================================================================

#: The column a stratum's cases are listed under when they declare no stratum, in a cell where others do.
#: Parenthesised so it cannot read as a stratum an author named.
NO_STRATUM = "(no stratum)"

#: Said beside a stratum's case count when it holds fewer than :data:`STRATUM_MIN_CASES` cases.
TOO_FEW_CASES = "too few cases to read alone"


def _strata_blocks(surface: DecisionSurface, variant_index: Sequence[VariantIndexEntry]) -> list[ReportBlock]:
    """Each broken-down cell's figures per stratum, beside its pooled figure — or nothing, when no cell has strata.

    One table, a row per arm and reading, a column per stratum: the arm's pooled figure first, then the same
    figure over each stratum's cases. Each arm opens with a ``cases`` row stating how many cases and
    observations each figure rests on, so a stratum's n is read before its numbers, and a stratum holding
    fewer than :data:`STRATUM_MIN_CASES` cases is called too few to read alone — there and in a disclosure
    below the table, never by leaving it out.

    Args:
        surface: The decision surface whose cells to lay out.
        variant_index: The variant index that names their arms.

    Returns:
        The table and its disclosure, or no block at all for a surface none of whose cells has strata — a
        campaign whose cases declare no stratum reads exactly as it did before strata existed.
    """
    control = surface.control_variant_key
    cells = surface_order(
        (cell for cell in surface.cells if cell.strata), control=control, names=arm_names(variant_index)
    )
    if not cells:
        return []
    labels = cell_arm_labels(surface, list(variant_index))
    names = sorted(
        {stratum.stratum for cell in cells for stratum in cell.strata},
        key=lambda name: (name is None, name or ""),
    )
    columns = {name: f"stratum_{index}" for index, name in enumerate(names)}
    rows: list[dict[str, Cell]] = []
    thin: list[str] = []
    for cell in cells:
        # Every cell on the surface is named, so the lookup cannot miss.
        arm = labels[cell_ref(cell.variant_key, cell.apparatus_class_id)]
        arm = f"{arm} {REFERENCE_MARK}" if cell.variant_key == control else arm
        by_name = {stratum.stratum: stratum for stratum in cell.strata}
        cases: dict[str, Cell] = {"arm": arm, "reading": "cases", "all": _cases_text(cell.n_observations, cell.n_cases)}
        for name, stratum in by_name.items():
            cases[columns[name]] = _cases_text(stratum.n_observations, stratum.n_cases)
            if stratum.n_cases < STRATUM_MIN_CASES:
                cases[columns[name]] = f"{cases[columns[name]]} — {TOO_FEW_CASES}"
                thin.append(
                    f"{_stratum_word(name)} in {arm} ({stratum.n_cases} case{'' if stratum.n_cases == 1 else 's'})"
                )
        rows.append(cases)
        rows.extend(_measure_rows(arm, cell, by_name, columns, surface))
        rows.extend(_judged_rows(arm, cell, by_name, columns))
    blocks: list[ReportBlock] = [
        TableBlock(
            section="surface",
            name="strata",
            title="By stratum",
            columns=[
                TableColumn(key="arm", header="Arm"),
                TableColumn(key="reading", header="Reading"),
                TableColumn(key="all", header="All cases"),
                *(TableColumn(key=columns[name], header=_stratum_word(name)) for name in names),
            ],
            rows=rows,
            order=surface_order_sentence(has_control_row=any(cell.variant_key == control for cell in cells))
            + " Within each arm: its cases, then each measure by name, then each judged dimension.",
            total_rows=len(rows),
        )
    ]
    if thin:
        blocks.append(
            DisclosureBlock(
                section="surface",
                source="strata",
                text=(
                    f"A stratum needs at least {STRATUM_MIN_CASES} cases to be read on its own, and these hold fewer, "
                    "so read each of their figures with its interval, which at that size is wide: "
                    + "; ".join(thin)
                    + "."
                ),
            )
        )
    return blocks


def _stratum_word(name: str | None) -> str:
    """A stratum as a column header and a disclosure name it: its own name, or :data:`NO_STRATUM`."""
    return NO_STRATUM if name is None else name


def _cases_text(n_observations: int, n_cases: int | None) -> str:
    """How many cases and observations a figure rests on — cases first, since they are the independent draws."""
    if n_cases is None:
        return f"{n_observations} obs, cases unrecorded"
    return f"{n_cases} case{'' if n_cases == 1 else 's'}, {n_observations} obs"


def _measure_rows(
    arm: str,
    cell: CellFacts,
    by_name: dict[str | None, StratumFacts],
    columns: dict[str | None, str],
    surface: DecisionSurface,
) -> list[dict[str, Cell]]:
    """One row per measure the cell or any of its strata holds — every one but a text measure, which is never summarised.

    A row says :data:`~threetears.evals.analysis.surface_table.NO_SUCCESSFUL_RESULTS` under the pool or a
    stratum where no result took a turn and which carries no figure of it — a cost or latency reading, which
    leaves those failures out — as the decision surface's own columns do, rather than a blank that reads as "not
    measured" beside the figures of the strata that delivered.
    """
    pooled = {summary.name: summary for summary in cell.measures.measures}
    per_stratum = {
        name: {summary.name: summary for summary in stratum.measures.measures} for name, stratum in by_name.items()
    }
    measures = sorted(
        {
            summary.name
            for summaries in (pooled, *per_stratum.values())
            for summary in summaries.values()
            if not summary.texts
        }
    )
    rows: list[dict[str, Cell]] = []
    for measure in measures:
        found = [summary for summaries in (pooled, *per_stratum.values()) if (summary := summaries.get(measure))]
        facts = surface.measures.get(measure)
        # One unit for the row, chosen over every figure in it, so a stratum in ms beside a pool in s never happens.
        factor, unit = display_scale(
            [value for summary in found for value in (summary.mean,) if value is not None],
            facts.unit if facts else None,
        )

        def figure(summaries: dict[str, MeasureSummary], facts_of: CellFacts | StratumFacts) -> Cell:
            if measure in summaries:
                return _summary_text(summaries[measure], factor)
            return NO_SUCCESSFUL_RESULTS if facts_of.all_failed else None

        row: dict[str, Cell] = {
            "arm": arm,
            "reading": f"{surface.measure_heading(measure)} ({unit})" if unit else surface.measure_heading(measure),
            "all": figure(pooled, cell),
        }
        for name, summaries in per_stratum.items():
            row[columns[name]] = figure(summaries, by_name[name])
        rows.append(row)
    return rows


def _judged_rows(
    arm: str, cell: CellFacts, by_name: dict[str | None, StratumFacts], columns: dict[str | None, str]
) -> list[dict[str, Cell]]:
    """One row per judged dimension the cell or any of its strata was scored on."""
    pooled = {reading.dimension: reading for reading in cell.judged}
    per_stratum = {
        name: {reading.dimension: reading for reading in stratum.judged} for name, stratum in by_name.items()
    }
    dimensions = sorted({dimension for readings in (pooled, *per_stratum.values()) for dimension in readings})
    rows: list[dict[str, Cell]] = []
    for dimension in dimensions:
        row: dict[str, Cell] = {
            "arm": arm,
            "reading": f"{dimension} (judged)",
            "all": _judged_text(pooled[dimension]) if dimension in pooled else None,
        }
        for name, readings in per_stratum.items():
            row[columns[name]] = _judged_text(readings[dimension]) if dimension in readings else None
        rows.append(row)
    return rows


def _summary_text(summary: MeasureSummary, factor: float) -> str:
    """One measure's figure as a cell of the strata table, with the n it rests on.

    A rate with its interval over the cases (Wilson's shape, which stays inside 0 to 1 at any n); a mean with its standard error, as
    the decision surface's table spells one; and a categorical measure — a confusion matrix — as its counts,
    largest first.
    """
    n = _n_text(summary.n, summary.n_independent)
    if summary.rate is not None:
        return f"{format_number(summary.rate)}{_interval(summary.ci_low, summary.ci_high)} {n}"
    if summary.mean is not None:
        spread = f" ± {format_number(summary.sem * factor)}" if summary.sem is not None else ""
        return f"{format_number(summary.mean * factor)}{spread} {n}"
    counts = sorted(summary.categories.items(), key=lambda item: (-item[1], item[0]))
    return "; ".join(f"{category}: {count}" for category, count in counts) + f" {n}"


def _interval(low: float | None, high: float | None) -> str:
    """An interval as ``[low, high]``, or nothing when either end is unestimated."""
    if low is None or high is None:
        return ""
    return f" [{format_number(low)}, {format_number(high)}]"


def _judged_text(reading: JudgedReading) -> str:
    """One judged dimension's mean, with its standard error and the scores it rests on."""
    if reading.mean is None:
        return f"no score (n={reading.n})"
    spread = f" ± {format_number(reading.sem)}" if reading.sem is not None else ""
    return f"{format_number(reading.mean)}{spread} {_n_text(reading.n, reading.n_independent)}"


def _n_text(n: int, n_cases: int) -> str:
    """A figure's n, and where cases repeat, how many cases it is over — the draws its spread counts."""
    if 0 < n_cases < n:
        return f"(n={n} over {n_cases} case{'' if n_cases == 1 else 's'})"
    return f"(n={n})"


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

#: What the code-only report says in place of an analysis: that none was generated, and whose the numbers are.
#: What an analysis would add is the reader's guide's to say (``docs/reading-reports.md``), not every report's.
NO_ANALYSIS = "No analysis was generated: everything below was computed by code from the campaign's evidence."


def build_code_only_report(
    bundle: AnalysisContextBundle, *, measures: MeasureRegistry, assembled_at: str, campaign_name: str | None = None
) -> Report:
    """Lay a campaign's assembled evidence out as a report, when no analysis exists to report through.

    Everything here is code's: the arm table (every arm and the levers it ran, with no status or finding
    column: every arm is ``unresolved``, since a verdict is a decision's and none was made), the decision surface, the contrasts the bundle tested against the control, one
    distribution chart per measure and judged dimension the surface can draw, a classifier's per-label
    precision, recall and F1 as one table, and every disclosure the bundle carries. No
    :class:`~threetears.evals.analysis.report.model.TextBlock` is built — the report model refuses one on a
    code-only report — and the first block says plainly that no analysis was generated.

    Args:
        bundle: The campaign's evidence, as :func:`~threetears.evals.analysis.bundle.assemble_context_bundle`
            assembled it.
        measures: The host's measure registry, which a chart's payload reads.
        assembled_at: When the bundle was assembled (ISO-8601), stated as the report's ``generated_at``.
        campaign_name: The campaign's name, which titles the report; ``None`` titles it by the campaign's id.

    Returns:
        The report, ``basis="code_only"``.
    """
    surface = bundle_decision_surface(bundle)
    variant_index = bundle.variant_index
    blocks: list[ReportBlock] = [DisclosureBlock(section="summary", source="generation", text=NO_ANALYSIS)]
    blocks.extend(_question_blocks(bundle))
    cell_name = _cell_namer(surface, variant_index)
    guardrails_read = guardrail_verdicts(surface, bundle.guardrails, cell_name)
    contrasts_read = contrast_verdicts(bundle, surface)
    blocks.extend(_guardrail_blocks(surface, bundle.guardrails, cell_name, guardrails_read))
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
    # The charts lead the surface table they draw: the chart is the reading form, the table the audit form.
    blocks.extend(_measure_chart_blocks(surface, bundle, measures))
    blocks.extend(_surface_blocks(surface_table_of(surface, variant_index), provenance=False))
    blocks.extend(_strata_blocks(surface, variant_index))
    blocks.extend(_comparison_blocks(bundle, surface, contrasts_read))
    blocks.extend(_label_blocks(surface, variant_index))
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
            campaign_name=campaign_name or None,
            scope_id=bundle.scope_id,
            subject_id=bundle.subject_id,
            subject_kind=bundle.subject_kind,
            behavior=bundle.behavior,
            generated_at=assembled_at,
            bundle_fingerprint=fingerprint,
        ),
        blocks=blocks,
        verdicts=[*guardrails_read, *bar_verdicts(surface, cell_name), *contrasts_read],
    )


def _question_blocks(bundle: AnalysisContextBundle) -> list[ReportBlock]:
    """The declared questions, each standing unanswered, and which readings none of them asked about.

    With no live question the whole campaign is exploratory — with no design at all, an exploratory
    campaign — and that is said once, here, rather than on every reading: the first block after the
    code-only report's one-line opening, which stays one line. With questions, the readings outside them
    are named, so a pattern in one is read as a lead and not as an answer.
    """
    design = bundle.declared_design
    scope = bundle.reading_scope
    blocks: list[ReportBlock] = []
    if scope.disclosure is not None:
        blocks.append(DisclosureBlock(section="questions", source="scope", text=scope.disclosure))
    if design is not None and design.questions:
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
        blocks.append(
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
        )
    exploratory = [*scope.exploratory_measures, *(f"{name} (judged)" for name in scope.exploratory_dimensions)]
    if exploratory:
        blocks.append(
            DisclosureBlock(
                section="questions",
                source="scope",
                text=(
                    f"Exploratory — no declared question asks about {_listed(exploratory)}: read a pattern in "
                    "them as a lead for a question, not as an answer."
                ),
            )
        )
    return blocks


def _comparison_cases(comparison: FamilyComparison) -> str:
    """The cases a contrast's test read, and how many each side ran that it left out.

    The means and the delta beside it are over exactly these cases, so a reader comparing them with a cell's
    own mean (over all its cases) is told why the two differ.
    """
    control, contrast = comparison.control, comparison.contrast
    if comparison.test == "paired":
        text = f"{control.n_cases} paired"
    else:
        text = f"{contrast.n_cases} vs {control.n_cases}" + (" unpaired" if comparison.test == "unpaired" else "")
    left_out = [
        f"{count} of the {side}'s"
        for side, count in (("arm", contrast.n_left_out), ("control", control.n_left_out))
        if count
    ]
    if left_out:
        text += f"; {' and '.join(left_out)} left out, not run by the other side"
    return text


def _comparison_interval(interval: tuple[float, float] | None, level: float | None) -> str | None:
    """A contrast's interval on its delta, with the level it holds at — simultaneous over its family."""
    if interval is None or level is None:
        return None
    return f"[{format_number(interval[0])}, {format_number(interval[1])}] at {format_number(100 * level)}%"


def _contrast_margin(bundle: AnalysisContextBundle, comparison: FamilyComparison) -> float | None:
    """The margin a contrast's reading has — its equivalence margin, or the margin no equivalence test could read."""
    if comparison.equivalence_margin is not None:
        return comparison.equivalence_margin
    if comparison.reading != "measure":
        return None
    descriptor = bundle.measure_catalog.get(comparison.name)
    if descriptor is not None and descriptor.materiality_threshold is not None:
        return descriptor.materiality_threshold
    return bundle.run_margins.get(comparison.name)


def _immaterial_words(margin: float | None, heading: str) -> str:
    """The caveat an immaterial row carries, naming the margin it was read against and the reading it is on.

    In the reader's terms — the margin they declared, on the reading the row names — rather than the descriptor
    field it is stored in, which a first comparison's reader has never seen.
    """
    where = "the margin declared on it" if margin is None else f"the margin of ±{format_number(margin)} on {heading}"
    return f" — immaterial: the observed delta is inside {where}, which does not show the true difference is that small"


def _contrast_reason(comparison: FamilyComparison) -> tuple[VerdictReason, str | None]:
    """Why a contrast's verdict is what it is, as a code and, where there is more to say, in words."""
    if comparison.verdict == "untested":
        return "untestable", comparison.untested_reason
    if comparison.verdict in ("improved", "regressed"):
        return "separated", None
    if comparison.verdict == "equivalent":
        return "inside_margin", None
    if comparison.margin_source is None:
        return "no_margin", None
    if comparison.equivalence_p_adjusted is None:
        return "margin_untested", comparison.equivalence_untested_reason or (
            "the test was unpaired, and the equivalence test reads only cases both arms ran"
        )
    return "not_inside_margin", None


def contrast_verdicts(bundle: AnalysisContextBundle, surface: DecisionSurface) -> list[Verdict]:
    """Each contrast the bundle tested against the control as a typed verdict, in the contrasts table's row order.

    The table's verdict cell is each one's words, so a program reading :attr:`Verdict.outcome` and a person reading
    the cell read one verdict.
    """
    arm = _cell_namer(surface, bundle.variant_index)
    verdicts = []
    for family in bundle.multiple_comparisons.families:
        for comparison in family.comparisons:
            heading = surface.measure_heading(comparison.name, comparison.reading)
            margin = _contrast_margin(bundle, comparison)
            reason, detail = _contrast_reason(comparison)
            words = (
                COMPARISON_VERDICT_WORDS[comparison.verdict]
                + (f" ({comparison.untested_reason})" if comparison.untested_reason else "")
                + (
                    f" (margin ±{format_number(margin)}"
                    + (", declared on the runs" if comparison.margin_source == "run" else "")
                    + ")"
                    if comparison.verdict == "equivalent" and margin is not None
                    else ""
                )
                + (
                    # The observed delta, not the true one: only `equivalent` shows the difference is small, and an
                    # `equivalent` row already says so through its margin, so the caveat would contradict the verdict.
                    _immaterial_words(margin, heading)
                    if comparison.materiality == "immaterial" and comparison.verdict != "equivalent"
                    else ""
                )
            )
            verdicts.append(
                Verdict(
                    kind="contrast",
                    outcome=comparison.verdict,
                    reason=reason,
                    reason_detail=detail,
                    reading=comparison.reading,
                    name=comparison.name,
                    heading=heading,
                    arm=arm(comparison.contrast.variant_key, comparison.contrast.apparatus_class_id),
                    variant_key=comparison.contrast.variant_key,
                    apparatus_class_id=comparison.contrast.apparatus_class_id,
                    control=arm(comparison.control.variant_key, comparison.control.apparatus_class_id),
                    question_id=family.question_id,
                    delta=comparison.delta,
                    interval=comparison.interval,
                    interval_level=family.interval_level if comparison.interval is not None else None,
                    p_adjusted=comparison.p_adjusted,
                    margin=margin,
                    margin_source=comparison.margin_source,
                    materiality=comparison.materiality,
                    words=words,
                )
            )
    return verdicts


def _comparison_blocks(
    bundle: AnalysisContextBundle, surface: DecisionSurface, verdicts: Sequence[Verdict]
) -> list[ReportBlock]:
    """Each contrast the bundle tested against the control, per live question, and how the family was corrected.

    ``verdicts`` are :func:`contrast_verdicts` over the same bundle, row for row; each row's verdict is its words.
    """
    comparisons = bundle.multiple_comparisons
    if not comparisons.families:
        if comparisons.withheld:
            return [DisclosureBlock(section="surface", source="comparisons", text=comparisons.withheld)]
        return []
    typed = iter(verdicts)

    rows: list[dict[str, Cell]] = []
    for family in comparisons.families:
        for comparison in family.comparisons:
            verdict = next(typed)
            rows.append(
                {
                    "question": (
                        _question_words(bundle.declared_design, family.question_id)
                        if family.question_id is not None
                        else "(campaign-wide)"
                    ),
                    "reading": verdict.heading,
                    "contrast": verdict.arm,
                    "control": verdict.control,
                    "control_mean": comparison.control.mean,
                    "arm_mean": comparison.contrast.mean,
                    "cases": _comparison_cases(comparison),
                    "delta": comparison.delta,
                    "interval": _comparison_interval(comparison.interval, family.interval_level),
                    "hedges_g": comparison.hedges_g,
                    "p_adjusted": comparison.p_adjusted,
                    "verdict": verdict.words,
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
                TableColumn(key="control_mean", header="Control mean"),
                TableColumn(key="arm_mean", header="Arm mean"),
                TableColumn(key="cases", header="Cases tested"),
                TableColumn(key="delta", header="Delta (arm − control)"),
                TableColumn(key="interval", header="Interval on delta"),
                TableColumn(key="hedges_g", header="Hedges' g"),
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
    # Said once per measure rather than on every row: which declared a margin with no range, so none of their
    # comparisons could be tested for equivalence, and what to declare (#695).
    unranged = sorted(
        {
            surface.measure_heading(comparison.name, comparison.reading)
            for family in comparisons.families
            for comparison in family.comparisons
            if comparison.equivalence_untested_reason is not None
        }
    )
    if unranged:
        blocks.append(
            DisclosureBlock(
                section="surface",
                source="comparisons",
                text=f"Equivalence untested on {_listed(unranged)}: {EQUIVALENCE_NEEDS_RANGE}.",
            )
        )
    if bundle.run_margins_withheld is not None:
        blocks.append(DisclosureBlock(section="surface", source="comparisons", text=bundle.run_margins_withheld))
    return blocks


def _measure_chart_blocks(
    surface: DecisionSurface, bundle: AnalysisContextBundle, measures: MeasureRegistry
) -> list[ReportBlock]:
    """One distribution chart per numeric measure and judged dimension — the charts code can choose on its own.

    A distribution is the one chart that reads a single reading across every cell with no author's choice
    of which cells to set against which, so it is the chart code picks. A cell is drawn where its reading
    has an interval; a cell left out, and a reading no cell can draw, are each disclosed rather than
    dropped — and a chart the presentation rules refuse is disclosed with the reason.

    Three readings are not charted, because a chart of them carries nothing:

    - **A classifier's per-label statistics.** A label's precision and recall are each a single reading per
      cell, so charted they were two charts per label — eight for four labels — most reading 1 in every arm;
      and its F1 has no interval by construction, so its chart could only ever be a disclosure that every
      cell was left out. All three are one table instead (:func:`_label_blocks`), every figure with its
      interval and n, which compares the arms label by label as closely as the charts did.
    - **``match`` beside ``accuracy``.** ``accuracy`` is derived from ``match``, observation for observation, so
      the two charts are one chart drawn twice.
    - **Cost no result observed.** A cell where no result reported its spend has no ``cost_usd`` reading at all
      (the bundle leaves its stored zeros out), so there is nothing to chart; the bundle's one sentence saying
      so stands in its place, naming the arms when only some went unmeasured.
    """
    blocks: list[ReportBlock] = []
    cells = cell_index(surface)
    labels = cell_arm_labels(surface, bundle.variant_index)
    if bundle.cost_unmeasured:
        unmeasured = [cell_ref(cell.variant_key, cell.apparatus_class_id) for cell in bundle.cost_unmeasured_cells]
        named = (
            ""
            if len(unmeasured) == len(cells)
            else " Unmeasured: " + "; ".join(labels.get(ref, ref) for ref in unmeasured) + "."
        )
        blocks.append(DisclosureBlock(section="surface", source="measurement", text=bundle.cost_unmeasured + named))
    if bundle.latency_contended:
        contended = [cell_ref(cell.variant_key, cell.apparatus_class_id) for cell in bundle.latency_contended_cells]
        named = (
            ""
            if len(contended) == len(cells)
            else " Read under concurrency: " + "; ".join(labels.get(ref, ref) for ref in contended) + "."
        )
        blocks.append(DisclosureBlock(section="surface", source="measurement", text=bundle.latency_contended + named))
    blocks.extend(_surface_chart_blocks(surface, bundle.variant_index, measures))
    return blocks


def _surface_chart_blocks(
    surface: DecisionSurface, variant_index: Sequence[VariantIndexEntry], measures: MeasureRegistry
) -> list[ReportBlock]:
    """The decision surface's readings compiled to charts, one distribution per chartable reading (#643).

    The one compiler both reports use, so a stored analysis's surface is drawn exactly as a code-only
    report's is, and ahead of the surface table: the chart is the reading form, the table the audit form.
    A reading drawn at fewer than two cells has nothing to compare, so it stays in the table alone.

    Args:
        surface: The decision surface.
        variant_index: How each cell is labelled.
        measures: The host's measure registry, which a payload reads.

    Returns:
        The chart blocks and the disclosures of what they left out.
    """
    readings: list[tuple[str, ReadingKind]] = [
        *((name, "measure") for name, facts in sorted(surface.measures.items()) if _chartable(name, facts, surface)),
        *((name, "judged") for name in sorted(surface.dimensions)),
    ]
    cells = cell_index(surface)
    labels = cell_arm_labels(surface, list(variant_index))
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
        if len(drawn) < 2:
            # Nothing to compare: every cell that measured it was left out (the disclosure above says so), one
            # cell drew it (the table states that one figure), or no cell measured it at all.
            continue
        try:
            payload = build_viz_payload(
                DistributionRef(cells=drawn, measure_id=name, reading=reading),
                surface,
                list(variant_index),
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


def _chartable(name: str, facts: MeasureFacts, surface: DecisionSurface) -> bool:
    """Whether a measure earns a distribution chart: it has a better end, and its chart would say something.

    A label's precision, recall and F1 are the per-label table's, and ``match`` is the observation ``accuracy``
    is derived from, so where ``accuracy`` is charted a ``match`` chart repeats it.
    """
    if facts.higher_is_better is None:
        return False
    if classifier_label_of(name) is not None:
        return False
    return not (name == MATCH_MEASURE and ACCURACY_MEASURE in surface.measures)


def _measured(cell: CellFacts, name: str, reading: ReadingKind) -> bool:
    """Whether a cell measured a reading at all — a cell that never did has nothing to leave out of its chart."""
    if reading == "judged":
        return any(judged.dimension == name for judged in cell.judged)
    return any(summary.name == name for summary in cell.measures.measures)


#: The columns of the per-label table after the label and the arm, by the statistic each holds.
_LABEL_STATISTICS: tuple[tuple[ClassifierStatistic, str], ...] = (
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
)


def _label_blocks(surface: DecisionSurface, variant_index: Sequence[VariantIndexEntry]) -> list[ReportBlock]:
    """A classifier's per-label precision, recall and F1, every arm's, as one table — or nothing for no classifier.

    A row per label and arm, the label's rows together so its arms read against each other: each figure with
    its interval and the n it is counted over, as the strata table spells them. A row per label and arm
    rather than a column per arm, so the table's width does not grow with the arms. One table rather than a
    chart per label and statistic: those drew each figure in a block of its own and showed nothing this does
    not.

    Args:
        surface: The decision surface whose cells carry the per-label summaries.
        variant_index: The variant index that names their arms.

    Returns:
        The table and the disclosure saying what its figures are counted over, or no block when no cell
        holds a per-label statistic.
    """
    control = surface.control_variant_key
    labels = cell_arm_labels(surface, list(variant_index))
    figures: dict[str, dict[str, dict[str, MeasureSummary]]] = {}
    for cell in surface.cells:
        ref = cell_ref(cell.variant_key, cell.apparatus_class_id)
        for summary in cell.measures.measures:
            if (classifier := classifier_label_of(summary.name)) is None:
                continue
            statistic, label = classifier
            figures.setdefault(label, {}).setdefault(ref, {})[statistic] = summary
    if not figures:
        return []
    order = sorted(
        cell_index(surface).items(),
        key=lambda item: (item[1].variant_key != control, labels[item[0]], item[1].apparatus_class_id),
    )
    # A cell that classified nothing is no arm of this table; one that classified, but never met a label nor
    # predicted it, is a row of dashes under that label, so every label is read across the same arms.
    classifying = {ref for by_cell in figures.values() for ref in by_cell}
    rows: list[dict[str, Cell]] = []
    missing = False
    single_case = False
    for label in sorted(figures):
        for ref, cell in order:
            if ref not in classifying:
                continue
            by_statistic = figures[label].get(ref, {})
            row: dict[str, Cell] = {
                "label": label,
                "arm": f"{labels[ref]} (control)" if cell.variant_key == control else labels[ref],
            }
            for statistic, _ in _LABEL_STATISTICS:
                figure = by_statistic.get(statistic)
                row[statistic] = None if figure is None else _summary_text(figure, 1.0)
                missing = missing or figure is None
                single_case = single_case or (
                    figure is not None and figure.rate is not None and (figure.ci_low is None or figure.ci_high is None)
                )
            rows.append(row)
    said = [
        f"Precision is counted over the observations an arm predicted as the label and recall over those expected as "
        f"it, each with its {with_unit(INTERVAL_LEVEL * 100, '%')} Wilson interval over the cell's cases — widened by "
        "the clustering it measured where a case was repeated, so a case's repeats are not counted as independent. "
        "F1 has no interval by construction — one value computed from the cell's confusion counts, over the "
        "observations predicted or expected as the label."
    ]
    if single_case:
        said.append(
            "A rate counted over the repeats of a single case has no interval: one case has no between-case spread."
        )
    if missing:
        said.append(
            "A label an arm never predicted has no precision, one it never met has no recall, and either has no F1: "
            "each is shown as —."
        )
    return [
        TableBlock(
            section="surface",
            name="labels",
            title="Per-label precision, recall and F1",
            columns=[
                TableColumn(key="label", header="Label"),
                TableColumn(key="arm", header="Arm"),
                *(TableColumn(key=statistic, header=header) for statistic, header in _LABEL_STATISTICS),
            ],
            rows=rows,
            order="by label, then the control's cells first and every other cell by arm and rig",
            total_rows=len(rows),
        ),
        DisclosureBlock(section="surface", source="surface", text=" ".join(said)),
    ]


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
    # Every goal check not shown to beat doing nothing: its pass rate is printed beside the others, and this says
    # what it is not.
    for proof in bundle.goal_check_proofs:
        if proof.proof != "proven":
            say("measurement", _unproven_check_sentence(proof))
    say("apparatus", bundle.held_fixed_reading.disclosure)
    if bundle.apparatus_confounds:
        say(
            "apparatus",
            "The rig did not hold still across the campaign: "
            + "; ".join(f"{confound.dimension} ({confound.status})" for confound in bundle.apparatus_confounds)
            + ".",
        )
    # A resolved surface folded into its knob without a check is a caveat on every comparison that folded it.
    # The bundle carries it on each such row; a code-only report has no rows of confounds, so it is said once
    # here, in the catalog's own words, so the report never reads an untested fold as a checked one.
    unverified = sorted(
        {
            confound.dimension
            for confounds in chain(
                (row.confounded_by for row in bundle.coverage),
                (divergence.confounded_by for divergence in bundle.scope_divergences),
            )
            for confound in confounds
            if confound.kind == "unverified_fold"
        }
    )
    for dimension in unverified:
        reason = bundle.confound_catalog[dimension]
        say("comparisons", reason[:1].upper() + reason[1:] + ".")
    # Factors that moved in lockstep are named once, as a group, and every co-varying pair's holes are listed (#596).
    for group in bundle.aliased_factors:
        say("comparisons", group.sentence)
    if bundle.factor_pairs is not None and bundle.factor_pairs.factors:
        say("comparisons", f"{bundle.factor_pairs.completeness} {bundle.factor_pairs.interaction_aliasing}")
        for pivot in bundle.factor_pairs.pivots:
            holes = [f"{cell.row_level} with {cell.column_level}" for cell in pivot.cells if cell.status == "not_run"]
            if holes:
                say(
                    "comparisons",
                    f"{pivot.row_factor} and {pivot.column_factor} co-vary; {len(holes)} of their {len(pivot.cells)} "
                    f"combinations never ran: {_listed(holes)}.",
                )
    # A design that says which combinations it meant to run names its gaps apart from what it skipped (#654).
    if bundle.declared_crossing is not None:
        say("comparisons", bundle.declared_crossing.sentence)
        for state, lead in (("not_run", "Declared cells never run"), ("skipped_by_design", "Cells skipped by design")):
            cells = [" × ".join(cell.levels.values()) for cell in bundle.declared_crossing.cells if cell.state == state]
            if cells:
                say("comparisons", f"{lead}: {'; '.join(cells)}.")
    # A declared level that never ran leaves `levels` silently; the row's declared_levels names it.
    for row in bundle.coverage:
        not_run = [level.display for level in row.declared_levels if level.state == "not_run"]
        undetermined = [level.display for level in row.declared_levels if level.state == "undetermined"]
        if not_run:
            say("comparisons", f"Declared on axis {row.name} and never run: {_listed(not_run)}.")
        if undetermined:
            say(
                "comparisons",
                f"Declared on axis {row.name}, and whether any run sat at it cannot be established: "
                f"{_listed(undetermined)}.",
            )
    # A declared axis on an input the host cannot vary is unswept by construction; without its cause the
    # row reads as a sweep that did not happen.
    for row in bundle.coverage:
        if row.cannot_be_an_arm:
            say(
                "comparisons",
                f"Declared axis {row.name} reads unswept whatever the runs did: {row.cannot_be_an_arm.rstrip('.')}.",
            )
    # An arm keyed by a floating alias can pool numbers from more than one model; a code-only report has no
    # writer to read `arm_served_models`, so the mixture is said here, once per arm.
    for reading in bundle.arm_served_models:
        if reading.state == "pooled":
            say(
                "measurement",
                f"Arm {short_digest(reading.variant_key)} asked for one model and its candidate was answered by "
                f"{_listed(reading.served_models)}, so its numbers mix those models.",
            )
    # An arm's production-replicating cost is production's only where its runs moved nothing (#571); a code-only
    # report has no writer to read each arm's `production_footing`, so it is said here, once for every arm.
    if (footing := _production_footing_sentence(bundle)) is not None:
        say("measurement", footing)
    for merge in bundle.refused_merges:
        dimensions = f" on {_listed(merge.dimensions)}" if merge.dimensions else ""
        say(
            "apparatus",
            f"Arm {short_digest(merge.variant_key)} was measured in {len(merge.apparatus_class_ids)} cells that did "
            f"not pool ({merge.reason.replace('_', ' ')}{dimensions}).",
        )
    if bundle.refused_merges_omitted:
        say(
            "apparatus",
            f"{bundle.refused_merges_omitted} more pair(s) of cells of one arm did not pool; they are left out of "
            "this list, the smallest first.",
        )
    return blocks


def _production_footing_sentence(bundle: AnalysisContextBundle) -> str | None:
    """Say, per arm, what its runs set away from the subject's production configuration (#571).

    Args:
        bundle: The bundle.

    Returns:
        One sentence naming each arm's moved and unchecked inputs, or that it moved nothing; ``None`` when the
        bundle carries no arm footing (one assembled before it was read).
    """
    parts: list[str] = []
    for variant_key, pooled in sorted(bundle.arm_production_footings.items()):
        footings = list(pooled.runs.values())
        moved = sorted({f"{name}={level}" for f in footings if f is not None for name, level in f.moved.items()})
        unchecked = sorted({name for f in footings if f is not None for name in f.unchecked})
        unread = sum(1 for f in footings if f is None)
        said = [
            *([f"set {', '.join(moved)}"] if moved else []),
            *([f"left unchecked whether {', '.join(unchecked)} is production's"] if unchecked else []),
            *([f"{unread} run(s) nobody checked"] if unread else []),
        ]
        parts.append(f"arm {short_digest(variant_key)} " + ("; ".join(said) or "moved nothing off production"))
    if not parts:
        return None
    return (
        "A production-replicating cost is what production would spend only where its arm's runs moved nothing off "
        "the subject's production configuration: " + "; ".join(parts) + "."
    )


__all__ = [
    "NO_ANALYSIS",
    "build_code_only_report",
    "build_report",
]

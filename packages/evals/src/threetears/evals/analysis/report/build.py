"""Build a :class:`~threetears.evals.analysis.report.model.Report` — from a stored analysis, or from the evidence alone.

**One builder, one page, every door.** :func:`build_report` lays out a stored analysis over the bundle frozen with it
at generation; :func:`build_code_only_report` lays out a campaign's evidence assembled now, when no analysis exists.
Both go through :func:`_page`, so the two differ only in the memo section: every count, verdict, chart and
disclosure is computed here, from the bundle alone, and no door — the quick path's ``compare``, the command line, an
MCP action — adds, drops or reorders a block.

The page, screen by screen:

1. **Header and scoreboard.** The title line (the campaign, its arms by the lever that differs, the date, the basis);
   the basis line (a question declared before the run, or ``EXPLORATORY``); the verdict line
   (:func:`~threetears.evals.analysis.report.verdict.verdict_line`); the scoreboard — each arm's trials ran, lost to
   the test setup by cause, usable, passed and failed by cause, and its cost — and its outcome bar; one line on the
   design, and one sentence collecting every chart that was not drawn.
2. **What it means.** The memo's findings (``F1``…) and recommendations, each pinned to the findings it rests on;
   on a code-only report, one line saying no analysis was written. Then the contrasts against the control, six
   columns with the statistics folded away, an interval chart per reading where any arm has the cases for one, the
   guardrails, and the readings by stratum.
3. **Cases.** One row per case, each arm's trials as marks, regressions first.
4. **Trials.** One drill-down per trial: what the case asked, the tail of what the judge saw, each failed check, the
   judge's scores and reasons, the error, and the stored result's id.
5. **Not tested.** Everything the campaign could not speak to, as a list, never a score.
6. **Methods.** The judge, the candidates as served, the cost by role, the dataset, the design, the memo's writer
   and the bundle's provenance.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime
from itertools import chain

from threetears.evals.analysis.agreement import tier_sentence
from threetears.evals.analysis.arms import arm_label, arm_names, surface_order
from threetears.evals.analysis.bundle.schema import (
    AnalysisContextBundle,
    ArmScoreboard,
    CellCoordinate,
    FamilyComparison,
    GoalCheckProofReading,
    TrialRow,
)
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.cells import cell_ref, variant_of_cell_ref
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.references import resolve_reading
from threetears.evals.analysis.report.figures import (
    NO_SUCCESSFUL_RESULTS,
    REFERENCE_MARK,
    UNITLESS,
    MeasureRuler,
    measure_rulers,
    surface_order_sentence,
)
from threetears.evals.analysis.report.model import (
    CellValue,
    ChartBlock,
    DisclosureBlock,
    DisclosureSource,
    Fact,
    IntervalRow,
    OutcomeBar,
    OutcomeSegment,
    Report,
    ReportBasis,
    ReportBlock,
    ReportSource,
    TableBlock,
    TableCell,
    TableColumn,
    TextBlock,
    TrialBlock,
    Verdict,
    VerdictReason,
    table_row,
)
from threetears.evals.analysis.report.verdict import (
    MIN_CASES,
    CauseWords,
    basis_line,
    cause_short,
    fail_words,
    loss_words,
    verdict_line,
)
from threetears.evals.analysis.report.words import (
    BAR_DECISION_WORDS,
    COMPARISON_VERDICT_WORDS,
    CONFIDENCE_WORDS,
    EVIDENCE_COLUMNS,
    GUARDRAIL_DECISION_WORDS,
    arm_namer,
    cell_arm_labels,
    counted,
    evidence_rows,
    positions,
    reading_name,
    stands_on_words,
)
from threetears.evals.analysis.reporting import self_judging_text
from threetears.evals.analysis.stats import EQUIVALENCE_NEEDS_RANGE, INTERVAL_LEVEL
from threetears.evals.analysis.quantities import with_unit
from threetears.evals.kernel.analysis_measures import MeasureSummary
from threetears.evals.kernel.authored import Finding
from threetears.evals.kernel.campaign import EvalAnalysis, FindingResolution, ReadingKind, VariantIndexEntry
from threetears.evals.kernel.declaration import JUDGED_MERIT_AXIS, CampaignDesign, Question, exploratory_reading
from threetears.evals.kernel.evidence_tiers import JudgeEvidenceTier
from threetears.evals.kernel.metrics import (
    METRIC_DESCRIPTORS,
    RUN_MARGIN_MEASURES,
    ClassifierStatistic,
    classifier_label_of,
)
from threetears.evals.kernel.surface import (
    STRATUM_MIN_CASES,
    CellFacts,
    DecisionSurface,
    GuardrailReadings,
    JudgedReading,
    PredictionPoweredReading,
    StratumFacts,
    TimeAxis,
    earlier_control_mark,
)
from threetears.evals.schema.models import CHECK_REFUSED_UNDER_CURRENT_GRAMMAR

#: How many lost or failed trials start open in the drill-down; the rest, and every passed one, start closed.
OPEN_TRIALS = 20

#: What the memo section says on a report no analysis was written for.
NO_ANALYSIS = (
    "No analysis was written: everything on this page was computed by code from the campaign's evidence. The "
    "analysis_generate action writes one, from this same evidence."
)

#: The ``Scope`` fact on a finding resting wholly on readings no question declared before the run asked about.
EXPLORATORY_FINDING = (
    "exploratory: it rests on no reading a question declared before the run asked about, so it is a lead, not an "
    "answer"
)

#: How a guardrail is decided, said once beside the guardrails table.
_GUARDRAILS_DISCLOSURE = (
    "A guardrail is what an arm must not get worse on. Each is decided on its own 95% interval on the difference "
    "from the control: held when the whole interval lies on the good side of the margin, breached when it lies "
    "wholly beyond it, undecided otherwise. On a declared range the interval is the bounded test's, which holds "
    "its error rate at any number of cases; a reading with no declared range is never held, since no test holds "
    "its rate there, and can only be shown breached. A measure's margin is its declared materiality threshold, a "
    "judged dimension's the one its campaign declares; with none, a guardrail is held at zero change. Guardrails "
    "join no comparison and no composite, so no gain elsewhere offsets one; an arm with a breached guardrail is not "
    "adopted, and an undecided guardrail is not known to be safe."
)


# =============================================================================
# The two entry points
# =============================================================================


def build_report(analysis: EvalAnalysis, *, campaign_name: str | None = None) -> Report:
    """Lay one stored analysis out as the campaign's page, over the bundle frozen with it at generation.

    Args:
        analysis: The analysis.
        campaign_name: The campaign's name, which leads the title; ``None`` names the campaign by its id.

    Returns:
        The report, ``basis="analysis"``.
    """
    generation = analysis.generation
    bundle = analysis_bundle(analysis)
    source = ReportSource(
        analysis_id=analysis.id,
        campaign_id=analysis.campaign_id,
        campaign_name=campaign_name or None,
        scope_id=analysis.scope_id,
        behavior=analysis.behavior,
        generated_at=generation.generated_at,
        generator_model=generation.generator_model,
        bundle_fingerprint=generation.bundle_fingerprint,
    )
    return _page(
        bundle,
        basis="analysis",
        source=source,
        memo=lambda charts: _memo_blocks(analysis, bundle, charts),
        finding_count=len(analysis.document.findings),
        writer=(
            f"Written by {generation.generator_model} with prompt {generation.prompt_id} (version "
            f"{generation.prompt_version}) on {generation.generated_at}, from the evidence below; every number on the "
            "page is code's."
        ),
    )


def build_code_only_report(
    bundle: AnalysisContextBundle, *, assembled_at: str, campaign_name: str | None = None
) -> Report:
    """Lay a campaign's evidence out as its page, when no analysis exists to report through.

    Everything here is code's, and the memo section is one line saying no analysis was written. The report model
    refuses an author's words on a code-only report.

    Args:
        bundle: The campaign's evidence, as :func:`~threetears.evals.analysis.bundle.assemble.assemble_context_bundle`
            assembled it.
        assembled_at: When the bundle was assembled (ISO-8601), stated as the report's date.
        campaign_name: The campaign's name, which leads the title; ``None`` names the campaign by its id.

    Returns:
        The report, ``basis="code_only"``.
    """
    source = ReportSource(
        campaign_id=bundle.campaign_id,
        campaign_name=campaign_name or None,
        scope_id=bundle.scope_id,
        behavior=bundle.behavior,
        generated_at=assembled_at,
        bundle_fingerprint=bundle.fingerprint(),
    )
    memo: list[ReportBlock] = [DisclosureBlock(section="memo", source="memo", text=NO_ANALYSIS)]
    memo += _question_blocks(bundle)
    return _page(bundle, basis="code_only", source=source, memo=lambda _: memo, finding_count=0, writer=None)


def analysis_bundle(analysis: EvalAnalysis) -> AnalysisContextBundle:
    """The evidence bundle a stored analysis was written over, read back from the analysis.

    Args:
        analysis: The analysis.

    Returns:
        The bundle, as frozen at generation.
    """
    return AnalysisContextBundle.from_dict(dict(analysis.bundle))


def analysis_verdict(analysis: EvalAnalysis) -> str:
    """The verdict line a stored analysis's report opens with, written by code over its frozen bundle.

    Args:
        analysis: The analysis.

    Returns:
        The verdict line.
    """
    bundle = analysis_bundle(analysis)
    return verdict_line(bundle, _bundle_arm_namer(bundle)).text


def _page(
    bundle: AnalysisContextBundle,
    *,
    basis: ReportBasis,
    source: ReportSource,
    memo: Callable[[_ChartLedger], list[ReportBlock]],
    finding_count: int,
    writer: str | None,
) -> Report:
    """The page every door renders: the same screens, from one bundle, whatever wrote the memo.

    The memo and the contrasts are laid out before the scoreboard, though they print after it: each chart they leave
    out is counted on the way, for the scoreboard's one sentence about what was not charted.
    """
    surface = bundle_decision_surface(bundle)
    arm = _bundle_arm_namer(bundle)
    cell_name = _cell_namer(surface, bundle.variant_index)
    charts = _ChartLedger(bundle, arm)
    guardrails_read = guardrail_verdicts(surface, bundle.guardrails, cell_name)
    bars_read = bar_verdicts(surface, cell_name)
    contrasts_read = contrast_verdicts(bundle, surface)
    written = memo(charts)
    contrasts = _contrast_blocks(bundle, surface, contrasts_read, guardrails_read, cell_name, charts)
    contrasts += _bar_blocks(surface, bars_read)
    blocks: list[ReportBlock] = [
        DisclosureBlock(section="header", source="basis", text=basis_line(bundle)),
        DisclosureBlock(section="header", source="verdict", text=verdict_line(bundle, arm).text),
        *_scoreboard_blocks(bundle, arm, charts),
        *written,
        *contrasts,
        *_case_blocks(bundle, arm),
        *_trial_blocks(bundle, arm),
        *_not_tested_blocks(bundle, surface, arm),
        *_method_blocks(bundle, surface, arm, source=source, writer=writer),
    ]
    return Report(
        basis=basis,
        title=_title(bundle, source, arm),
        finding_count=finding_count,
        source=source,
        blocks=blocks,
        verdicts=[*guardrails_read, *bars_read, *contrasts_read],
        bundle=bundle,
    )


# =============================================================================
# Screen 1: the title, the basis, the verdict and the scoreboard
# =============================================================================


def _title(bundle: AnalysisContextBundle, source: ReportSource, arm: Callable[[str], str]) -> str:
    """The title line: the campaign, each arm by the lever that differs, the date, and who wrote the memo."""
    control = bundle.design.control_arm.variant_key if bundle.design.control_arm is not None else None
    others = [board.variant_key for board in bundle.scoreboard if board.variant_key != control]
    arms = (
        f"{arm(control)} (control)" + (f" vs {', '.join(arm(key) for key in others)}" if others else "")
        if control is not None and any(board.variant_key == control for board in bundle.scoreboard)
        else ", ".join(arm(board.variant_key) for board in bundle.scoreboard) or "no arm ran"
    )
    by = f"analysis by {source.generator_model}" if source.generator_model else "no analysis written"
    return " · ".join([source.campaign_name or f"Campaign {source.campaign_id}", arms, _date(source.generated_at), by])


def _date(stamp: str) -> str:
    """An ISO-8601 instant as the title dates it (``14 Oct 2026``), or as given when it does not parse."""
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        # NOSILENT: an unparseable stamp is printed as stored rather than dropped or reformatted
        return stamp
    return f"{moment.day} {moment:%b %Y}"


def _causes(counts: dict[str, int], words: Callable[[str], str]) -> str:
    """Counts by cause, largest first, in their short words: ``judge truncated 7, test setup fault 1``."""
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{words(cause)} {count}" for cause, count in ordered)


def _scoreboard_blocks(
    bundle: AnalysisContextBundle, arm: Callable[[str], str], charts: _ChartLedger
) -> list[ReportBlock]:
    """The scoreboard table, its outcome bar, the design line, the charts sentence and the costs."""
    boards = bundle.scoreboard
    if not boards:
        return [
            DisclosureBlock(
                section="scoreboard", source="runs", text="No trial ran: the campaign resolved no result in this scope."
            )
        ]
    control = bundle.design.control_arm.variant_key if bundle.design.control_arm is not None else None
    earlier = bundle.design.earlier_control
    rows: list[dict[str, CellValue]] = []
    for board in boards:
        lost = sum(board.lost.values())
        failed = sum(board.failed.values())
        rows.append(
            {
                "arm": arm(board.variant_key)
                + (" (control)" if board.variant_key == control else "")
                + (f" ({earlier_control_mark(earlier)})" if earlier and board.variant_key == earlier.variant_key else ""),
                "ran": board.ran,
                "lost": f"{lost} ({_causes(board.lost, cause_short)})" if lost else "0",
                "usable": f"{counted(board.usable, board.ran, 'trials usable')} ({lost} lost to the test setup)",
                "passed": counted(board.passed, board.usable, "usable passed"),
                "failed": f"{failed} ({_causes(board.failed, cause_short)})" if failed else "0",
                "cost": with_unit(board.test_cost_usd, "usd") if board.test_cost_usd is not None else None,
                "reply": with_unit(board.reply_cost_usd, "usd") if board.reply_cost_usd is not None else None,
            }
        )
    blocks: list[ReportBlock] = [
        TableBlock(
            section="scoreboard",
            name="scoreboard",
            title="Scoreboard: what each arm ran, and how each trial ended",
            columns=[
                TableColumn(key="arm", header="Arm"),
                TableColumn(key="ran", header="Ran"),
                TableColumn(key="lost", header="Lost to the test setup"),
                TableColumn(key="usable", header="Usable"),
                TableColumn(key="passed", header="Passed"),
                TableColumn(key="failed", header="Failed"),
                TableColumn(key="cost", header="This test cost (all roles)"),
                TableColumn(key="reply", header="Per reply (production roles)"),
            ],
            rows=[table_row(row) for row in rows],
            order="the control first, then the other arms by name; a count, not a ranking",
            total_rows=len(rows),
        ),
        ChartBlock(
            section="scoreboard",
            chart="outcome_bar",
            title="How each arm's trials ended",
            bars=[_outcome_bar(board, arm) for board in boards],
        ),
        DisclosureBlock(section="scoreboard", source="design", text=_design_line(bundle)),
    ]
    if (sentence := charts.sentence()) is not None:
        blocks.append(DisclosureBlock(section="scoreboard", source="charts", text=sentence))
    if bundle.all_failed_cells:
        failed = list(dict.fromkeys(arm(cell.variant_key) for cell in bundle.all_failed_cells))
        blocks.append(
            DisclosureBlock(
                section="scoreboard",
                source="runs",
                text=(
                    f"Every trial of {', '.join(failed)} failed: the model refused or errored on every call and took no "
                    "turn, so there is no cost or latency to read — not a fast, free arm — and every failure counts "
                    "against the arm."
                ),
            )
        )
    unpriced = [board for board in boards if board.n_cost_unpriced]
    if unpriced:
        blocks.append(
            DisclosureBlock(
                section="scoreboard",
                source="cost",
                text="The cost is a floor where a trial's spend went unpriced: "
                + "; ".join(f"{arm(board.variant_key)}, {board.n_cost_unpriced} of {board.ran}" for board in unpriced)
                + ".",
            )
        )
    return blocks


def _outcome_bar(board: ArmScoreboard, arm: Callable[[str], str]) -> OutcomeBar:
    """One arm's census as its stacked bar: passed, then each failure cause, then each loss cause."""
    segments = [OutcomeSegment(state="passed", label="passed", count=board.passed)] if board.passed else []
    segments += [
        OutcomeSegment(state="failed", label=f"failed: {cause_short(cause)}", count=count)
        for cause, count in sorted(board.failed.items(), key=lambda item: (-item[1], item[0]))
    ]
    segments += [
        OutcomeSegment(state="lost", label=f"lost: {cause_short(cause)}", count=count)
        for cause, count in sorted(board.lost.items(), key=lambda item: (-item[1], item[0]))
    ]
    return OutcomeBar(arm=arm(board.variant_key), segments=segments)


def _design_line(bundle: AnalysisContextBundle) -> str:
    """One line on the design: how many cases, how many repeats of each."""
    cases = sorted({board.cases for board in bundle.scoreboard})
    repeats = sorted(
        {
            value
            for cell in bundle.cell_measures
            for value in (cell.repeats_per_case_min, cell.repeats_per_case_max)
            if value is not None
        }
    )
    case_words = f"{cases[0]} cases" if len(cases) == 1 else f"{cases[0]} to {cases[-1]} cases per arm"
    repeat_words = (
        "repeats not recorded"
        if not repeats
        else f"{repeats[0]} repeat{'' if repeats[0] == 1 else 's'}"
        if len(repeats) == 1
        else f"{repeats[0]} to {repeats[-1]} repeats"
    )
    trials = sum(board.ran for board in bundle.scoreboard)
    return f"{case_words} × {repeat_words}; {trials} trials over {len(bundle.scoreboard)} arms."


class _ChartLedger:
    """Which interval charts were drawn, and why each row left out was: the one sentence on the scoreboard.

    A row is drawn only when its arm has :data:`MIN_CASES` usable cases and an interval was computed, and a chart
    with no such row is not built at all — no figure, no stub. Every row left out is counted here, so the scoreboard
    says once, in one sentence, what was not charted and why.
    """

    def __init__(self, bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> None:
        self._arm = arm
        self._usable = {board.variant_key: board.usable_cases for board in bundle.scoreboard}
        self._small: dict[str, int] = {}
        self._no_interval: list[str] = []

    def drawable(self, variant_key: str, n_cases: int | None, interval: tuple[float, float] | None, what: str) -> bool:
        """Whether a row of ``variant_key``'s may be drawn; one that may not is counted for the sentence."""
        usable = min(self._usable.get(variant_key, 0), n_cases if n_cases is not None else 0)
        if usable < MIN_CASES:
            self._small[variant_key] = min(self._small.get(variant_key, usable), usable)
            return False
        if interval is None:
            self.left_out(what)
            return False
        return True

    def left_out(self, what: str) -> None:
        """Count a reading no row of which could be drawn, for want of an interval."""
        if what not in self._no_interval:
            self._no_interval.append(what)

    def sentence(self) -> str | None:
        """Every row not charted, in one sentence; None when nothing was left out and some arm could be charted."""
        if all(usable < MIN_CASES for usable in self._usable.values()):
            return f"No rate is estimated or charted: no arm has {MIN_CASES} usable cases."
        parts = []
        if self._small:
            named = "; ".join(
                f"{self._arm(key)}, {usable} usable case{'' if usable == 1 else 's'}"
                for key, usable in sorted(self._small.items(), key=lambda item: self._arm(item[0]))
            )
            parts.append(f"Not charted: {named} (a rate needs {MIN_CASES}).")
        if self._no_interval:
            parts.append(f"Not charted, no interval computed: {', '.join(self._no_interval)}.")
        return " ".join(parts) or None


# =============================================================================
# Screen 2: what it means — the memo, then the contrasts against the control
# =============================================================================


def _memo_blocks(analysis: EvalAnalysis, bundle: AnalysisContextBundle, charts: _ChartLedger) -> list[ReportBlock]:
    """The memo: answers to the declared questions, the findings, the recommendations and the next steps.

    A finding's evidence is followed by an interval chart per reading it cites where an arm has the cases for one.
    """
    document = analysis.document
    generation = analysis.generation
    arm = arm_namer(analysis)
    questions = _preregistered(bundle)
    blocks: list[ReportBlock] = [
        DisclosureBlock(
            section="memo",
            source="memo",
            text=(
                f"Written by {generation.generator_model} from the evidence below; every number is code's. Findings "
                "are numbered F1 onward, and each recommendation names the findings it rests on."
            ),
        )
    ]
    blocks.extend(
        TextBlock(
            section="memo",
            role="answer",
            body=answer.answer,
            rests_on=list(answer.rests_on),
            facts=[
                Fact(name="Question", value=_question_words(bundle.declared_design, answer.question_id)),
                Fact(name="Resolution", value=answer.resolution),
            ],
        )
        for answer in document.questions
    )
    resolutions: list[FindingResolution | None] = (
        list(analysis.resolutions) if analysis.resolutions else [None] * len(document.findings)
    )
    for position, (finding, resolution) in enumerate(zip(document.findings, resolutions, strict=True)):
        exploratory = not questions or _rests_on_exploratory(resolution, analysis.decision_surface, questions)
        blocks.extend(_finding_blocks(analysis, position, finding, resolution, arm, charts, exploratory=exploratory))
    for decision in document.decisions:
        facts = [
            Fact(name="Disposition", value=decision.disposition),
            Fact(name="Confidence", value=CONFIDENCE_WORDS[decision.confidence]),
        ]
        if decision.cells:
            facts.append(Fact(name="Arms", value="; ".join(arm(cell) for cell in decision.cells)))
        if decision.disposition == "adopted" and (standing := _guardrail_standing(decision.cells, bundle, arm)):
            facts.append(Fact(name="Guardrails", value=standing))
        if cases := _cases_of(bundle, decision.rests_on, document.findings):
            facts.append(Fact(name="See", value=", ".join(cases)))
        blocks.append(
            TextBlock(
                section="memo",
                role="recommendation",
                body=decision.proposal,
                rests_on=list(decision.rests_on),
                facts=facts,
            )
        )
        if decision.revisit_when.strip():
            blocks.append(
                TextBlock(
                    section="memo", role="revisit_when", body=decision.revisit_when, rests_on=list(decision.rests_on)
                )
            )
    coverage_status = {row.name: row.status for row in analysis.coverage.levers}
    for step in document.next:
        facts = [Fact(name="Leverage", value=step.leverage)]
        if step.lever:
            status = coverage_status.get(step.lever) or "no coverage row"
            facts.append(Fact(name="Lever", value=f"{step.lever} ({status})"))
        blocks.append(TextBlock(section="memo", role="next_step", body=step.title, facts=facts))
        if step.why.strip():
            blocks.append(TextBlock(section="memo", role="next_step_why", body=step.why))
    blocks.extend(_coverage_blocks(analysis))
    return blocks


def _coverage_blocks(analysis: EvalAnalysis) -> list[ReportBlock]:
    """The coverage map joined to the findings and the next steps, on the stored lever names (#630, #631).

    One row per lever: its status, the findings naming it in their ``axes`` (or "no finding"), and the next steps
    whose ``lever`` names it; a ``thin`` or ``unswept`` lever no step names says so. Then one line counting the
    levers no finding names — said, never acted on.
    """
    levers = analysis.coverage.levers
    if not levers:
        return []
    findings = analysis.document.findings
    rows: list[dict[str, CellValue]] = []
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
    unnamed = [row.name for row in levers if not any(row.name in finding.axes for finding in findings)]
    return [
        TableBlock(
            section="memo",
            name="coverage",
            title="Coverage: each lever, the findings naming it, and the next step that would measure it",
            columns=[
                TableColumn(key="lever", header="Lever"),
                TableColumn(key="status", header="Coverage"),
                TableColumn(key="findings", header="Named by finding"),
                TableColumn(key="next", header="Next step naming it"),
            ],
            rows=[table_row(row) for row in rows],
            order="as the coverage map lists the levers",
            total_rows=len(rows),
        ),
        DisclosureBlock(
            section="memo",
            source="memo",
            text=(
                f"{len(unnamed)} of the {len(levers)} lever(s) in the coverage map are named by no finding: "
                f"{_listed(unnamed)}."
                if unnamed
                else f"Every one of the {len(levers)} lever(s) in the coverage map is named by a finding."
            ),
        ),
    ]


def _guardrail_standing(cells: Sequence[str], bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str | None:
    """What an adopted recommendation's arms leave not held among the guardrails, in words; None when every one held.

    Stated by code on the recommendation, so an arm recommended over an undecided guardrail is never read as shown
    safe.
    """
    control = bundle.design.control_arm.variant_key if bundle.design.control_arm is not None else None
    parts = []
    for cell in cells:
        variant = variant_of_cell_ref(cell)
        if variant is None or variant == control:
            continue
        standing = bundle.guardrails.of_arm(variant)
        said = []
        if standing.breached:
            breached = ", ".join(_guardrail_heading(bundle, name, "measure") for name in standing.breached)
            said.append(f"breached {breached}")
        if standing.undecided:
            undecided = ", ".join(_guardrail_heading(bundle, name, "measure") for name in standing.undecided)
            said.append(f"undecided on {undecided}, so not known to be safe")
        if said:
            parts.append(f"{arm(cell)}: {'; '.join(said)}")
    return " | ".join(parts) or None


def _cases_of(bundle: AnalysisContextBundle, rests_on: Sequence[int], findings: Sequence[Finding]) -> list[str]:
    """The cases the trials of the named findings show, as ``case N`` links, in case order."""
    named = {trial for position in rests_on if 0 <= position < len(findings) for trial in findings[position].trials}
    numbers = _case_numbers(bundle)
    cases = sorted({numbers[trial.case_id] for trial in bundle.trials if trial.trial_id in named})
    return [f"case {number}" for number in cases]


def _finding_blocks(
    analysis: EvalAnalysis,
    position: int,
    finding: Finding,
    resolution: FindingResolution | None,
    arm: Callable[[str], str],
    charts: _ChartLedger,
    *,
    exploratory: bool,
) -> list[ReportBlock]:
    """One finding: its title and the facts beside it, its body, its evidence and their charts, and its caveats."""
    facts = [Fact(name="Confidence", value=CONFIDENCE_WORDS[finding.confidence])]
    if exploratory:
        facts.append(Fact(name="Scope", value=EXPLORATORY_FINDING))
    if resolution is not None:
        facts.append(Fact(name="Stands on", value=stands_on_words(analysis, resolution.evidence_tier)))
    if finding.axes:
        facts.append(Fact(name="About", value=", ".join(finding.axes)))
    if finding.trials:
        facts.append(Fact(name="Trials", value=", ".join(finding.trials)))
    if finding.invalidates:
        facts.append(Fact(name="Invalidates", value=positions(list(finding.invalidates))))
    blocks: list[ReportBlock] = [
        TextBlock(
            section="memo",
            role="finding_title",
            finding=position,
            body=finding.title,
            facts=facts,
            anchor=f"F{position + 1}",
        )
    ]
    if finding.body.strip():
        blocks.append(TextBlock(section="memo", role="finding_body", finding=position, body=finding.body))
    if resolution is not None and resolution.evidence:
        rows = evidence_rows(analysis, resolution.evidence, arm)
        blocks.append(
            TableBlock(
                section="memo",
                finding=position,
                name="evidence",
                title="Evidence",
                columns=[TableColumn(key=key, header=header) for key, header in EVIDENCE_COLUMNS],
                rows=rows,
                order="as the author listed the readings",
                total_rows=len(rows),
            )
        )
        blocks.extend(_finding_charts(analysis, position, resolution, charts, arm))
    blocks.extend(
        TextBlock(
            section="memo",
            role="caveat",
            finding=position,
            body=caveat.text,
            facts=[Fact(name="Kind", value=caveat.kind)],
        )
        for caveat in finding.caveats
    )
    if finding.durable.strip():
        blocks.append(TextBlock(section="memo", role="carried_forward", finding=position, body=finding.durable))
    return blocks


def _finding_charts(
    analysis: EvalAnalysis,
    position: int,
    resolution: FindingResolution,
    charts: _ChartLedger,
    arm: Callable[[str], str],
) -> list[ReportBlock]:
    """An interval chart per reading a finding cites, drawing the arms with the cases for one; none when none has."""
    blocks: list[ReportBlock] = []
    surface = analysis.decision_surface
    rulers = measure_rulers(surface)
    cited: dict[tuple[str, ReadingKind], list[str]] = {}
    for row in resolution.evidence:
        cells = cited.setdefault((row.measure_id, row.reading), [])
        if row.cell_ref not in cells:
            cells.append(row.cell_ref)
    for (name, reading), refs in cited.items():
        ruler = rulers.get(name, UNITLESS) if reading == "measure" else UNITLESS
        what = reading_name(surface.measure_heading(name, reading))
        rows = []
        for ref in refs:
            try:
                resolved = resolve_reading(surface, ref, name, reading)
            except UnresolvableReference:
                # NOSILENT: the reading has no interval to draw, and the ledger names it among those not charted
                charts.left_out(what)
                continue
            low, high = resolved.ci_low, resolved.ci_high
            interval = (low, high) if low is not None and high is not None else None
            variant = variant_of_cell_ref(ref) or ref
            if not charts.drawable(variant, resolved.n_cases, interval, what) or interval is None:
                continue
            rows.append(
                IntervalRow(
                    label=arm(ref),
                    value=resolved.mean * ruler.factor,
                    low=interval[0] * ruler.factor,
                    high=interval[1] * ruler.factor,
                    n_cases=resolved.n_cases or 1,
                )
            )
        if rows:
            blocks.append(
                ChartBlock(
                    section="memo",
                    finding=position,
                    chart="interval_rows",
                    title=f"F{position + 1}: {what}, each arm with its {format_number(100 * INTERVAL_LEVEL)}% interval",
                    rows=rows,
                    unit=ruler.unit,
                )
            )
    return blocks


def _preregistered(bundle: AnalysisContextBundle) -> list[Question]:
    """The live questions declared before the campaign's first run — the ones a finding can answer."""
    design = bundle.declared_design
    if design is None or not bundle.reading_scope.questions_declared:
        return []
    post_hoc = set(bundle.reading_scope.post_hoc_questions)
    return [question for question in design.live_questions() if question.id not in post_hoc]


def _rests_on_exploratory(
    resolution: FindingResolution | None, surface: DecisionSurface, questions: Sequence[Question]
) -> bool:
    """Whether a finding's every reading lies outside the declared questions — none of them asked about it.

    A finding citing no reading (a point about coverage or the test setup) is not labelled; a guardrail reading is
    never exploratory, since it is held because it was declared one.
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


def _question_words(design: CampaignDesign | None, question_id: str) -> str:
    """A declared question in the words it was asked, or its id where the design holds none."""
    return design.question_words(question_id) if design is not None else question_id


def _question_blocks(bundle: AnalysisContextBundle) -> list[ReportBlock]:
    """On a code-only report: the declared questions, each standing unanswered, and which readings none asked about."""
    design = bundle.declared_design
    scope = bundle.reading_scope
    blocks: list[ReportBlock] = []
    if design is not None and design.questions:
        rows: list[dict[str, CellValue]] = [
            {
                "question": question.id,
                "asks": question.text,
                "axes": ", ".join(question.merit_axes) or None,
                "status": "retired" if question.retired_at else "live",
                "answer": "unanswered: no analysis",
            }
            for question in design.questions
        ]
        blocks.append(
            TableBlock(
                section="memo",
                name="questions",
                title="Declared questions",
                columns=[
                    TableColumn(key="question", header="Question"),
                    TableColumn(key="asks", header="Asks"),
                    TableColumn(key="axes", header="Merit axes"),
                    TableColumn(key="status", header="Status"),
                    TableColumn(key="answer", header="Answer"),
                ],
                rows=[table_row(row) for row in rows],
                order="as the campaign declared them",
                total_rows=len(rows),
            )
        )
    exploratory = [*scope.exploratory_measures, *(f"{name} (judged)" for name in scope.exploratory_dimensions)]
    if exploratory and scope.questions_declared:
        blocks.append(
            DisclosureBlock(
                section="memo",
                source="scope",
                text=(
                    f"Exploratory: no declared question asks about {_listed(exploratory)}, so read a pattern in them "
                    "as a lead for a question, not as an answer."
                ),
            )
        )
    return blocks


def _contrast_blocks(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    contrasts_read: Sequence[Verdict],
    guardrails_read: Sequence[Verdict],
    cell_name: Callable[[str, str], str],
    charts: _ChartLedger,
) -> list[ReportBlock]:
    """The contrasts against the control, their charts, the guardrails, and the readings by stratum and label."""
    arm = _bundle_arm_namer(bundle)
    blocks: list[ReportBlock] = []
    blocks.extend(_self_judging_blocks(bundle, surface, arm))
    blocks.extend(_guardrail_standing_blocks(bundle, arm))
    blocks.extend(_comparison_blocks(bundle, surface, contrasts_read, charts))
    blocks.extend(_guardrail_blocks(surface, bundle.guardrails, guardrails_read))
    blocks.extend(_strata_blocks(surface, bundle.variant_index))
    blocks.extend(_label_blocks(surface, bundle.variant_index))
    return blocks


def _self_judging_blocks(
    bundle: AnalysisContextBundle, surface: DecisionSurface, arm: Callable[[str], str]
) -> list[ReportBlock]:
    """A line for each arm whose answers its judge's own model produced: its judged contrasts may favour it."""
    variant_of_run = {run_id: cell.variant_key for cell in surface.cells for run_id in cell.run_ids}
    shared: dict[str, list[str]] = {}
    for summary in bundle.run_summaries:
        variant = variant_of_run.get(summary.run_id)
        if variant is None or not summary.judge_shares_candidate_model:
            continue
        models = shared.setdefault(variant, [])
        models.extend(model for model in summary.judge_shares_candidate_model if model not in models)
    return [
        DisclosureBlock(
            section="contrasts",
            source="apparatus",
            text=self_judging_text(models, f"arm {arm(variant)}")
            + "; a contrast on a judged dimension may favour that arm for that reason alone.",
        )
        for variant, models in sorted(shared.items(), key=lambda item: arm(item[0]))
    ]


def _guardrail_standing_blocks(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> list[ReportBlock]:
    """A line for each arm a guardrail leaves not shown safe, above the contrasts it might be picked from."""
    has_contrasts = any(family.comparisons for family in bundle.multiple_comparisons.families)
    blocks: list[ReportBlock] = []
    variants = list(dict.fromkeys(check.contrast.variant_key for check in bundle.guardrails.checks))
    for variant in variants:
        said: dict[str, list[str]] = {"breached": [], "undecided": []}
        for check in bundle.guardrails.checks:
            if check.contrast.variant_key == variant and check.decision in said:
                heading = _guardrail_heading(bundle, check.name, check.reading)
                if heading not in said[check.decision]:
                    said[check.decision].append(heading)
        breached = said["breached"]
        undecided = [name for name in said["undecided"] if name not in breached]
        gained = "not adopted, whatever the contrasts below show it gained." if has_contrasts else "not adopted."
        if breached:
            blocks.append(
                DisclosureBlock(
                    section="contrasts",
                    source="guardrails",
                    text=(
                        f"Arm {arm(variant)} breached the guardrail{'s' if len(breached) > 1 else ''} "
                        f"{', '.join(breached)}: it is shown worse than the control by more than the margin, so it is "
                        + gained
                    ),
                )
            )
        if undecided:
            blocks.append(
                DisclosureBlock(
                    section="contrasts",
                    source="guardrails",
                    text=(
                        f"Arm {arm(variant)} is undecided on the guardrail{'s' if len(undecided) > 1 else ''} "
                        f"{', '.join(undecided)}, so it is not known to be safe; the guardrails table says why."
                    ),
                )
            )
    return blocks


def _guardrail_heading(bundle: AnalysisContextBundle, name: str, reading: str) -> str:
    """A guardrail as a reader reads it: a measure by its reader name, a judged dimension by its own."""
    descriptor = bundle.measure_catalog.get(name) if reading == "measure" else None
    return (descriptor.reader_name if descriptor is not None else None) or reading_name(name)


def _small_n(comparison: FamilyComparison) -> int | None:
    """The fewer of a contrast's two case counts when it is below :data:`MIN_CASES`; None when both reach it."""
    n = min(comparison.control.n_cases, comparison.contrast.n_cases)
    return n if n < MIN_CASES else None


def _comparison_cases(comparison: FamilyComparison) -> str:
    """The cases a contrast's test read, and how many each side ran that it left out."""
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


def _figure(value: float | None, ruler: MeasureRuler) -> TableCell:
    """One figure of a measure as a table cell: restated onto the measure's ruler and carrying its unit."""
    if value is None:
        return TableCell(value=None)
    return TableCell(value=value * ruler.factor, unit=ruler.unit)


def _interval_on_delta(interval: tuple[float, float] | None, ruler: MeasureRuler, level: float | None) -> str | None:
    """An interval on a delta, on the measure's ruler with its unit, and the level it holds at."""
    if interval is None or level is None:
        return None
    low, high = (format_number(bound * ruler.factor) for bound in interval)
    unit = f" {ruler.unit}" if ruler.unit else ""
    return f"[{low}, {high}]{unit} at {format_number(100 * level)}%"


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


def _margin_text(margin: float, ruler: MeasureRuler) -> str:
    """A margin as the row's reading is stated: on its measure's one ruler, with its unit."""
    return f"±{with_unit(margin * ruler.factor, ruler.unit)}"


def _immaterial_words(margin: float | None, heading: str, ruler: MeasureRuler) -> str:
    """The caveat an immaterial row carries, naming the margin it was read against and the reading it is on."""
    where = (
        "the margin declared on it" if margin is None else f"the margin of {_margin_text(margin, ruler)} on {heading}"
    )
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
        return "no_margin", comparison.not_separated_reason
    if comparison.equivalence_p_adjusted is None:
        return "margin_untested", comparison.not_separated_reason or comparison.equivalence_untested_reason or (
            "the test was unpaired, and the equivalence test reads only cases both arms ran"
        )
    return "not_inside_margin", None


def contrast_verdicts(bundle: AnalysisContextBundle, surface: DecisionSurface) -> list[Verdict]:
    """Each contrast the bundle tested against the control as a typed verdict, in the contrasts table's row order.

    A contrast with fewer than :data:`MIN_CASES` cases on either side is printed ``not tested (n=N)``: its typed
    outcome is the test's, and the words state no difference, as a rate on so few cases states none.
    """
    arm = _cell_namer(surface, bundle.variant_index)
    rulers = measure_rulers(surface)
    verdicts = []
    for family in bundle.multiple_comparisons.families:
        for comparison in family.comparisons:
            heading = reading_name(surface.measure_heading(comparison.name, comparison.reading))
            margin = _contrast_margin(bundle, comparison)
            ruler = rulers.get(comparison.name, UNITLESS) if comparison.reading == "measure" else UNITLESS
            reason, detail = _contrast_reason(comparison)
            if (small := _small_n(comparison)) is not None:
                reason = "too_few_cases"
                detail = f"{small} case{'' if small == 1 else 's'}, under the {MIN_CASES} a rate needs"
                words = f"not tested (n={small})"
            else:
                words = _contrast_words(comparison, margin, heading, ruler)
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


def _contrast_words(comparison: FamilyComparison, margin: float | None, heading: str, ruler: MeasureRuler) -> str:
    """A tested contrast's verdict in words: the verdict, why where there is more to say, its margin, materiality."""
    on_runs = ", declared on the runs" if comparison.margin_source == "run" else ""
    return (
        COMPARISON_VERDICT_WORDS[comparison.verdict]
        + (f" ({comparison.untested_reason})" if comparison.untested_reason else "")
        + (f" ({comparison.not_separated_reason})" if comparison.not_separated_reason else "")
        + (
            f" (margin {_margin_text(margin, ruler)}{on_runs})"
            if comparison.verdict == "equivalent" and margin is not None
            else ""
        )
        + (
            _immaterial_words(margin, heading, ruler)
            if comparison.materiality == "immaterial" and comparison.verdict != "equivalent"
            else ""
        )
    )


def _comparison_blocks(
    bundle: AnalysisContextBundle, surface: DecisionSurface, verdicts: Sequence[Verdict], charts: _ChartLedger
) -> list[ReportBlock]:
    """The contrasts table (six columns), its statistics folded away, and an interval chart per reading that has one.

    ``verdicts`` are :func:`contrast_verdicts` over the same bundle, row for row; each row's verdict is its words.
    """
    comparisons = bundle.multiple_comparisons
    if not comparisons.families:
        if comparisons.withheld:
            return [DisclosureBlock(section="contrasts", source="comparisons", text=comparisons.withheld)]
        return []
    typed = iter(verdicts)
    rulers = measure_rulers(surface)
    shown: list[dict[str, TableCell]] = []
    statistics: list[dict[str, TableCell]] = []
    drawn: dict[str, list[IntervalRow]] = {}
    margins: dict[str, float | None] = {}
    units: dict[str, str] = {}
    for family in comparisons.families:
        for comparison in family.comparisons:
            verdict = next(typed)
            ruler = rulers.get(comparison.name, UNITLESS) if comparison.reading == "measure" else UNITLESS
            small = _small_n(comparison)
            interval = _interval_on_delta(comparison.interval, ruler, family.interval_level)
            change = (
                None
                if small is not None or comparison.delta is None
                else with_unit(comparison.delta * ruler.factor, ruler.unit) + (f" {interval}" if interval else "")
            )
            shown.append(
                {
                    "reading": TableCell(value=verdict.heading),
                    "control": TableCell(value=verdict.control),
                    "contrast": TableCell(value=verdict.arm),
                    "change": TableCell(value=change),
                    "verdict": TableCell(value=verdict.words),
                    "cases": TableCell(value=_comparison_cases(comparison)),
                }
            )
            statistics.append(
                {
                    "question": TableCell(
                        value=(
                            _question_words(bundle.declared_design, family.question_id)
                            if family.question_id is not None
                            else "(campaign-wide)"
                        )
                    ),
                    "reading": TableCell(value=verdict.heading),
                    "contrast": TableCell(value=verdict.arm),
                    "control_mean": _figure(comparison.control.mean, ruler),
                    "arm_mean": _figure(comparison.contrast.mean, ruler),
                    "delta": _figure(None if small is not None else comparison.delta, ruler),
                    "interval": TableCell(value=None if small is not None else interval),
                    "hedges_g": TableCell(value=None if small is not None else comparison.hedges_g),
                    "pairing_r": TableCell(
                        value=comparison.pairing_correlation.r
                        if comparison.pairing_correlation is not None and small is None
                        else None
                    ),
                    "p_adjusted": TableCell(value=None if small is not None else comparison.p_adjusted),
                }
            )
            arm_ok = charts.drawable(
                comparison.contrast.variant_key,
                min(comparison.contrast.n_cases, comparison.control.n_cases),
                comparison.interval if comparison.delta is not None else None,
                verdict.heading,
            )
            control_ok = charts.drawable(
                comparison.control.variant_key, comparison.control.n_cases, (0.0, 0.0), verdict.heading
            )
            if arm_ok and control_ok and comparison.interval is not None and comparison.delta is not None:
                drawn.setdefault(verdict.heading, []).append(
                    IntervalRow(
                        label=f"{verdict.arm} − {verdict.control}",
                        value=comparison.delta * ruler.factor,
                        low=comparison.interval[0] * ruler.factor,
                        high=comparison.interval[1] * ruler.factor,
                        n_cases=min(comparison.contrast.n_cases, comparison.control.n_cases),
                        verdict=verdict.words,
                    )
                )
                margin = _contrast_margin(bundle, comparison)
                margins[verdict.heading] = margin * ruler.factor if margin is not None else None
                units[verdict.heading] = ruler.unit
    blocks: list[ReportBlock] = [
        TableBlock(
            section="contrasts",
            name="comparisons",
            title="Each arm against the control",
            columns=[
                TableColumn(key="reading", header="Reading"),
                TableColumn(key="control", header="Control"),
                TableColumn(key="contrast", header="Arm"),
                TableColumn(key="change", header="Δ (arm − control) with its interval"),
                TableColumn(key="verdict", header="Verdict"),
                TableColumn(key="cases", header="Cases"),
            ],
            rows=shown,
            order="by declared question, then as the bundle tested them",
            total_rows=len(shown),
        ),
        TableBlock(
            section="contrasts",
            name="comparison_statistics",
            title="Statistics behind each contrast",
            columns=[
                TableColumn(key="question", header="Question"),
                TableColumn(key="reading", header="Reading"),
                TableColumn(key="contrast", header="Arm"),
                TableColumn(key="control_mean", header="Control mean"),
                TableColumn(key="arm_mean", header="Arm mean"),
                TableColumn(key="delta", header="Δ"),
                TableColumn(key="interval", header="Interval on Δ"),
                TableColumn(key="hedges_g", header="Hedges' g"),
                TableColumn(key="pairing_r", header="Pairing r"),
                TableColumn(key="p_adjusted", header="p (Holm-adjusted)"),
            ],
            rows=statistics,
            order="row for row as the contrasts above",
            total_rows=len(statistics),
            details=True,
        ),
    ]
    blocks.extend(
        ChartBlock(
            section="contrasts",
            chart="interval_rows",
            title=f"{heading}: each arm minus the control, with its interval",
            rows=rows,
            unit=units[heading],
            zero_line=True,
            margin=margins[heading],
        )
        for heading, rows in drawn.items()
    )
    blocks.extend(
        DisclosureBlock(section="contrasts", source="comparisons", text=family.disclosure)
        for family in comparisons.families
    )
    unranged = sorted(
        {
            reading_name(surface.measure_heading(comparison.name, comparison.reading))
            for family in comparisons.families
            for comparison in family.comparisons
            if comparison.equivalence_untested_reason is not None
        }
    )
    if unranged:
        blocks.append(
            DisclosureBlock(
                section="contrasts",
                source="comparisons",
                text=f"Equivalence untested on {_listed(unranged)}: {EQUIVALENCE_NEEDS_RANGE}.",
            )
        )
    if bundle.run_margins_withheld is not None:
        blocks.append(DisclosureBlock(section="contrasts", source="comparisons", text=bundle.run_margins_withheld))
    blocks += [
        DisclosureBlock(section="contrasts", source="comparisons", text=text) for text in bundle.launch_declarations
    ]
    return blocks


def _cell_namer(surface: DecisionSurface, variant_index: Sequence[VariantIndexEntry]) -> Callable[[str, str], str]:
    """Name a cell by its two coordinates as every table does."""
    labels = cell_arm_labels(surface, variant_index)
    names = arm_names(variant_index)

    def name(variant_key: str, apparatus_class_id: str) -> str:
        return labels.get(cell_ref(variant_key, apparatus_class_id)) or arm_label(variant_key, names)

    return name


def _bundle_arm_namer(bundle: AnalysisContextBundle) -> Callable[[str], str]:
    """Name an arm by its variant key as every table of the report names it, never as a bare digest."""
    names = arm_names(bundle.variant_index)
    return lambda variant_key: arm_label(variant_key, names)


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
                heading=reading_name(surface.measure_heading(check.name, check.reading)),
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
    """Each adjudicated bar's reading on each arm as a typed verdict, bar by bar in the surface's order."""
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
                    heading=reading_name(surface.measure_heading(bar.measure_id, reading)),
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
    surface: DecisionSurface, guardrails: GuardrailReadings | None, verdicts: Sequence[Verdict]
) -> list[ReportBlock]:
    """Each guardrail, held, breached or undecided for each arm against the control; the rule said once beside it."""
    if guardrails is None:
        return []
    if guardrails.withheld is not None:
        return [DisclosureBlock(section="contrasts", source="guardrails", text=guardrails.withheld)]
    if not guardrails.checks:
        return []
    rulers = measure_rulers(surface)
    rows: list[dict[str, TableCell]] = []
    for check, verdict in zip(guardrails.checks, verdicts, strict=True):
        ruler = rulers.get(check.name, UNITLESS) if check.reading == "measure" else UNITLESS
        interval = _interval_on_delta(check.interval, ruler, INTERVAL_LEVEL)
        rows.append(
            {
                "guardrail": TableCell(value=verdict.heading),
                "arm": TableCell(value=verdict.arm),
                "control_mean": _figure(check.control.mean, ruler),
                "arm_mean": _figure(check.contrast.mean, ruler),
                "cases": TableCell(
                    value=(
                        f"{check.control.n_cases} paired"
                        if check.test == "paired"
                        else f"{check.contrast.n_cases} vs {check.control.n_cases}"
                    )
                ),
                "delta": _figure(check.delta, ruler),
                "interval": TableCell(
                    value=None
                    if interval is None
                    else interval + (" (t: no declared range)" if check.interval_basis == "t" else "")
                ),
                "margin": TableCell(
                    value=with_unit(check.margin * ruler.factor, ruler.unit)
                    if check.margin_declared
                    else "0 (none declared)"
                ),
                "decision": TableCell(value=verdict.words),
            }
        )
    return [
        TableBlock(
            section="contrasts",
            name="guardrails",
            title="Guardrails against the control",
            columns=[
                TableColumn(key="guardrail", header="Guardrail"),
                TableColumn(key="arm", header="Arm"),
                TableColumn(key="control_mean", header="Control mean"),
                TableColumn(key="arm_mean", header="Arm mean"),
                TableColumn(key="cases", header="Cases"),
                TableColumn(key="delta", header="Δ (arm − control)"),
                TableColumn(key="interval", header="Interval on Δ"),
                TableColumn(key="margin", header="Margin"),
                TableColumn(key="decision", header="Decision"),
            ],
            rows=rows,
            order="by guardrail, then arm",
            total_rows=len(rows),
        ),
        DisclosureBlock(section="contrasts", source="guardrails", text=_GUARDRAILS_DISCLOSURE),
    ]


def _bar_blocks(surface: DecisionSurface, verdicts: Sequence[Verdict]) -> list[ReportBlock]:
    """Each adjudicated bar, read on each arm, and every bar no arm could be read against."""
    blocks: list[ReportBlock] = []
    rulers = measure_rulers(surface)
    if verdicts:
        rows: list[dict[str, TableCell]] = []
        for verdict in verdicts:
            ruler = rulers.get(verdict.name, UNITLESS) if verdict.reading == "measure" else UNITLESS
            rows.append(
                {
                    "bar": TableCell(value=verdict.heading),
                    "arm": TableCell(value=verdict.arm),
                    "value": _figure(verdict.value, ruler),
                    "interval": TableCell(
                        value=None
                        if verdict.interval is None
                        else f"[{format_number(verdict.interval[0] * ruler.factor)}, "
                        f"{format_number(verdict.interval[1] * ruler.factor)}]"
                    ),
                    "threshold": _figure(verdict.threshold, ruler),
                    "decision": TableCell(value=verdict.words),
                }
            )
        blocks.append(
            TableBlock(
                section="contrasts",
                name="bars",
                title="Bars, each read on each arm",
                columns=[
                    TableColumn(key="bar", header="Bar"),
                    TableColumn(key="arm", header="Arm"),
                    TableColumn(key="value", header="Value"),
                    TableColumn(key="interval", header="Interval"),
                    TableColumn(key="threshold", header="Threshold"),
                    TableColumn(key="decision", header="Decision"),
                ],
                rows=rows,
                order="bar by bar, each arm in the surface's order",
                total_rows=len(rows),
                details=True,
            )
        )
    blocks.extend(
        DisclosureBlock(
            section="contrasts",
            source="comparisons",
            text=f"The {bar.source} bar on {bar.measure_id} could be read against no arm: {bar.reason}.",
        )
        for bar in surface.bars
        if bar.state != "adjudicated" and bar.reason
    )
    return blocks


# =============================================================================
# Readings by stratum, and a classifier's labels
# =============================================================================

#: The column a stratum's cases are listed under when they declare no stratum, in an arm where others do.
NO_STRATUM = "(no stratum)"

#: Said beside a stratum's case count when it holds fewer than :data:`STRATUM_MIN_CASES` cases.
TOO_FEW_CASES = "too few cases to read alone"


def _strata_blocks(surface: DecisionSurface, variant_index: Sequence[VariantIndexEntry]) -> list[ReportBlock]:
    """Each arm's figures per stratum beside its pooled figure — or nothing, when no case declares a stratum."""
    control = surface.control_variant_key
    cells = surface_order(
        (cell for cell in surface.cells if cell.strata), control=control, names=arm_names(variant_index)
    )
    if not cells:
        return []
    labels = cell_arm_labels(surface, variant_index)
    names = sorted(
        {stratum.stratum for cell in cells for stratum in cell.strata}, key=lambda name: (name is None, name or "")
    )
    columns = {name: f"stratum_{index}" for index, name in enumerate(names)}
    rows: list[dict[str, CellValue]] = []
    thin: list[str] = []
    for cell in cells:
        arm = labels[cell_ref(cell.variant_key, cell.apparatus_class_id)]
        arm = f"{arm} {REFERENCE_MARK}" if cell.variant_key == control else arm
        by_name = {stratum.stratum: stratum for stratum in cell.strata}
        cases: dict[str, CellValue] = {
            "arm": arm,
            "reading": "cases",
            "all": _cases_text(cell.n_observations, cell.n_cases),
        }
        for name in names:
            cases[columns[name]] = None
        for name, stratum in by_name.items():
            cases[columns[name]] = _cases_text(stratum.n_observations, stratum.n_cases)
            if stratum.n_cases < STRATUM_MIN_CASES:
                cases[columns[name]] = f"{cases[columns[name]]} — {TOO_FEW_CASES}"
                plural = "" if stratum.n_cases == 1 else "s"
                thin.append(f"{_stratum_word(name)} in {arm} ({stratum.n_cases} case{plural})")
        rows.append(cases)
        rows.extend(_measure_rows(arm, cell, by_name, columns, surface))
        rows.extend(_judged_rows(arm, cell, by_name, columns))
    blocks: list[ReportBlock] = [
        TableBlock(
            section="contrasts",
            name="strata",
            title="By stratum",
            columns=[
                TableColumn(key="arm", header="Arm"),
                TableColumn(key="reading", header="Reading"),
                TableColumn(key="all", header="All cases"),
                *(TableColumn(key=columns[name], header=_stratum_word(name)) for name in names),
            ],
            rows=[table_row(row) for row in rows],
            order=surface_order_sentence(has_control_row=any(cell.variant_key == control for cell in cells))
            + " Within each arm: its cases, then each measure by name, then each judged dimension.",
            total_rows=len(rows),
            details=True,
        )
    ]
    if thin:
        blocks.append(
            DisclosureBlock(
                section="contrasts",
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


def _cases_text(n_trials: int, n_cases: int | None) -> str:
    """How many cases and trials a figure rests on — cases first, since they are the independent draws."""
    if n_cases is None:
        return f"{n_trials} trials, cases unrecorded"
    return f"{n_cases} case{'' if n_cases == 1 else 's'}, {n_trials} trials"


def _measure_rows(
    arm: str,
    cell: CellFacts,
    by_name: dict[str | None, StratumFacts],
    columns: dict[str | None, str],
    surface: DecisionSurface,
) -> list[dict[str, CellValue]]:
    """One row per measure the arm or any of its strata holds — every one but a text measure."""
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
    rulers = measure_rulers(surface)
    rows: list[dict[str, CellValue]] = []
    for measure in measures:
        factor, unit = rulers.get(measure, UNITLESS)

        def figure(summaries: dict[str, MeasureSummary], facts_of: CellFacts | StratumFacts) -> CellValue:
            if measure in summaries:
                return _summary_text(summaries[measure], factor)
            return NO_SUCCESSFUL_RESULTS if facts_of.all_failed else None

        heading = surface.measure_heading(measure)
        row: dict[str, CellValue] = {
            "arm": arm,
            "reading": f"{heading} ({unit})" if unit else heading,
            "all": figure(pooled, cell),
        }
        for name in columns:
            row[columns[name]] = figure(per_stratum[name], by_name[name]) if name in per_stratum else None
        rows.append(row)
    return rows


def _judged_rows(
    arm: str, cell: CellFacts, by_name: dict[str | None, StratumFacts], columns: dict[str | None, str]
) -> list[dict[str, CellValue]]:
    """One row per judged dimension the arm or any of its strata was scored on."""
    pooled = {reading.dimension: reading for reading in cell.judged}
    per_stratum = {
        name: {reading.dimension: reading for reading in stratum.judged} for name, stratum in by_name.items()
    }
    dimensions = sorted({dimension for readings in (pooled, *per_stratum.values()) for dimension in readings})
    rows: list[dict[str, CellValue]] = []
    for dimension in dimensions:
        row: dict[str, CellValue] = {
            "arm": arm,
            "reading": f"{reading_name(dimension)} (judged)",
            "all": _judged_text(pooled[dimension]) if dimension in pooled else None,
        }
        for name in columns:
            readings = per_stratum.get(name, {})
            row[columns[name]] = _judged_text(readings[dimension]) if dimension in readings else None
        rows.append(row)
    return rows


def _summary_text(summary: MeasureSummary, factor: float) -> str:
    """One measure's figure with the n it rests on: a rate as a count below :data:`MIN_CASES` cases."""
    n = _n_text(summary.n, summary.n_independent)
    if summary.rate is not None:
        if summary.n_independent < MIN_CASES:
            return f"{round(summary.rate * summary.n)} of {summary.n} {n}"
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
    text = f"{format_number(reading.mean)}{spread} {_n_text(reading.n, reading.n_independent)}"
    if (powered := reading.prediction_powered) is not None:
        text += f"; {_prediction_powered_text(powered)}"
    return text


def _prediction_powered_text(powered: PredictionPoweredReading) -> str:
    """The prediction-powered estimate beside a judged mean: its value and interval, or why it is not available."""
    rated = f"{powered.n_labelled} rated by people"
    if powered.mean is None:
        return f"with people's ratings: not available ({rated}, under {powered.min_labelled})"
    return (
        f"with people's ratings: {format_number(powered.mean)}{_interval(powered.ci_low, powered.ci_high)} "
        f"(rectifier {format_number(powered.rectifier or 0.0)}; {rated})"
    )


def _n_text(n: int, n_cases: int) -> str:
    """A figure's n, and where cases repeat, how many cases it is over."""
    if 0 < n_cases < n:
        return f"(n={n} over {n_cases} case{'' if n_cases == 1 else 's'})"
    return f"(n={n})"


#: The columns of the per-label table after the label and the arm, by the statistic each holds.
_LABEL_STATISTICS: tuple[tuple[ClassifierStatistic, str], ...] = (
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
)


def _label_blocks(surface: DecisionSurface, variant_index: Sequence[VariantIndexEntry]) -> list[ReportBlock]:
    """A classifier's per-label precision, recall and F1, every arm's, as one table — or nothing for no classifier."""
    control = surface.control_variant_key
    labels = cell_arm_labels(surface, variant_index)
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
        ((cell_ref(cell.variant_key, cell.apparatus_class_id), cell) for cell in surface.cells),
        key=lambda item: (item[1].variant_key != control, labels[item[0]], item[1].apparatus_class_id),
    )
    classifying = {ref for by_cell in figures.values() for ref in by_cell}
    rows: list[dict[str, CellValue]] = []
    missing = False
    for label in sorted(figures):
        for ref, cell in order:
            if ref not in classifying:
                continue
            by_statistic = figures[label].get(ref, {})
            row: dict[str, CellValue] = {
                "label": label,
                "arm": f"{labels[ref]} (control)" if cell.variant_key == control else labels[ref],
            }
            for statistic, _ in _LABEL_STATISTICS:
                figure = by_statistic.get(statistic)
                row[statistic] = None if figure is None else _summary_text(figure, 1.0)
                missing = missing or figure is None
            rows.append(row)
    said = [
        "Precision is counted over the trials an arm predicted as the label and recall over those expected as it, each "
        f"with its {with_unit(INTERVAL_LEVEL * 100, '%')} Wilson interval over the arm's cases; below "
        f"{MIN_CASES} cases a rate is stated as a count. F1 has no interval by construction."
    ]
    if missing:
        said.append(
            "A label an arm never predicted has no precision, one it never met has no recall, and either has no F1: "
            "each is shown as —."
        )
    return [
        TableBlock(
            section="contrasts",
            name="labels",
            title="Per-label precision, recall and F1",
            columns=[
                TableColumn(key="label", header="Label"),
                TableColumn(key="arm", header="Arm"),
                *(TableColumn(key=statistic, header=header) for statistic, header in _LABEL_STATISTICS),
            ],
            rows=[table_row(row) for row in rows],
            order="by label, then the control first and every other arm by name",
            total_rows=len(rows),
            details=True,
        ),
        DisclosureBlock(section="contrasts", source="comparisons", text=" ".join(said)),
    ]


# =============================================================================
# Screens 3 and 4: the cases, and each trial to drill into
# =============================================================================

#: A trial's state as a mark in the cases table, always beside words for a lost one.
_MARKS = {"passed": "✓", "failed": "✗", "lost": "–"}


def _case_numbers(bundle: AnalysisContextBundle) -> dict[str, int]:
    """Each case's number, as the cases table numbers it: by label, from 1."""
    labels = {trial.case_id: trial.case_label for trial in bundle.trials}
    return {case: index for index, case in enumerate(sorted(labels, key=lambda case: (labels[case], case)), start=1)}


def _case_blocks(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> list[ReportBlock]:
    """One row per case: each arm's trials as marks with their causes, regressions first, each linked to its trials."""
    if not bundle.trials:
        return []
    control = bundle.design.control_arm.variant_key if bundle.design.control_arm is not None else None
    arms = [board.variant_key for board in bundle.scoreboard]
    numbers = _case_numbers(bundle)
    by_case: dict[str, list[TrialRow]] = {}
    for trial in bundle.trials:
        by_case.setdefault(trial.case_id, []).append(trial)

    def group(case: str) -> int:
        trials = by_case[case]
        control_passed = control is not None and any(t.arm == control and t.state == "passed" for t in trials)
        if control_passed and any(t.arm != control and t.state == "failed" for t in trials):
            return 0
        if any(not any(t.arm == key and t.state != "lost" for t in trials) for key in arms):
            return 1
        return 2

    rows: list[dict[str, CellValue]] = []
    for case in sorted(by_case, key=lambda case: (group(case), numbers[case])):
        trials = by_case[case]
        row: dict[str, CellValue] = {
            "case": f"case {numbers[case]}",
            "label": trials[0].case_label,
            "why": {0: "regression: the control passed and an arm failed", 1: "an arm has no usable trial", 2: None}[
                group(case)
            ],
        }
        for index, key in enumerate(arms):
            row[f"arm_{index}"] = _marks([trial for trial in trials if trial.arm == key])
        row["trials"] = " ".join(trial.trial_id for trial in trials)
        rows.append(row)
    blocks: list[ReportBlock] = [
        TableBlock(
            section="cases",
            name="cases",
            title="Each case, each arm's trials (✓ passed, ✗ failed, – lost to the test setup)",
            columns=[
                TableColumn(key="case", header="Case"),
                TableColumn(key="label", header="What it asked"),
                TableColumn(key="why", header="Read first because"),
                *(
                    TableColumn(key=f"arm_{index}", header=f"{arm(key)} (control)" if key == control else arm(key))
                    for index, key in enumerate(arms)
                ),
                TableColumn(key="trials", header="Trials"),
            ],
            rows=[table_row(row) for row in rows],
            order="regressions first, then cases an arm has no usable trial of, then the rest, each by case number",
            total_rows=len(rows),
        )
    ]
    if bundle.trials_omitted:
        blocks.append(
            DisclosureBlock(
                section="cases",
                source="runs",
                text=(
                    f"{bundle.trials_omitted} more trials are counted on the scoreboard and left out here, the passed "
                    "ones first: a bundle carries at most the trials it can show."
                ),
            )
        )
    return blocks


def _marks(trials: Sequence[TrialRow]) -> str:
    """One arm's trials of one case as marks, repeat by repeat, its losses and failures named by cause."""
    ordered = sorted(trials, key=lambda trial: trial.repeat)
    marks = " ".join(_MARKS[trial.state] for trial in ordered) or "not run"
    named = Counter(
        f"{'lost' if trial.state == 'lost' else 'failed'}: {cause_short(trial.cause or 'unknown')}"
        for trial in ordered
        if trial.state != "passed"
    )
    if named:
        marks += " (" + ", ".join(f"{words} ×{count}" for words, count in sorted(named.items())) + ")"
    return marks


def _trial_blocks(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> list[ReportBlock]:
    """One drill-down per trial, the first :data:`OPEN_TRIALS` lost or failed ones open."""
    blocks: list[ReportBlock] = []
    opened = 0
    arms = len(bundle.scoreboard)
    exclusions = {entry.variant_key: entry for entry in bundle.exclusions}
    for trial in bundle.trials:
        words: CauseWords | None = None
        if trial.state == "lost":
            words = loss_words(trial.cause or "unknown", exclusions=exclusions.get(trial.arm), arms=arms)
        elif trial.state == "failed":
            words = fail_words(trial.cause or "unknown")
        is_open = trial.state != "passed" and opened < OPEN_TRIALS
        opened += is_open
        blocks.append(
            TrialBlock(
                section="trials",
                anchor=trial.trial_id,
                trial_id=trial.trial_id,
                case_label=trial.case_label,
                arm=arm(trial.arm),
                repeat=trial.repeat,
                state=trial.state,
                cause_words=None if words is None else f"{words.words}. {words.remedy[:1].upper()}{words.remedy[1:]}.",
                first_input=trial.first_input,
                excerpt=trial.excerpt,
                failed_checks=list(trial.failed_checks),
                judge=list(trial.judge),
                error_words=trial.error_words,
                result_id=trial.result_id,
                open=is_open,
            )
        )
    return blocks


# =============================================================================
# Screen 5: what the campaign could not speak to
# =============================================================================


def _not_tested_blocks(
    bundle: AnalysisContextBundle, surface: DecisionSurface, arm: Callable[[str], str]
) -> list[ReportBlock]:
    """Everything the campaign could not speak to, one line each — a list, never a score."""
    said: list[str] = []
    for row in bundle.coverage:
        if row.status in ("thin", "unswept"):
            why = "one level ran" if row.status == "unswept" else "too few repeats to read"
            said.append(f"Lever {row.name} is {row.status}: {why}.")
    usable = {(trial.arm, trial.case_id) for trial in bundle.trials if trial.state != "lost"}
    for board in bundle.scoreboard:
        unusable = sorted(
            {
                trial.case_label
                for trial in bundle.trials
                if trial.arm == board.variant_key and (trial.arm, trial.case_id) not in usable
            }
        )
        if unusable:
            name = arm(board.variant_key)
            said.append(f"{name} has no usable trial of {len(unusable)} case(s): {_listed(unusable)}.")
    for proof in bundle.goal_check_proofs:
        if proof.proof != "proven":
            said.append(_unproven_check_sentence(proof))
    if (no_margin := _no_margin_sentence(bundle)) is not None:
        said.append(no_margin)
    undecided = sorted(
        {
            f"{_guardrail_heading(bundle, check.name, check.reading)} for {arm(check.contrast.variant_key)}"
            for check in bundle.guardrails.checks
            if check.decision == "undecided"
        }
    )
    if undecided:
        said.append(f"Guardrails left undecided, so not known to be safe: {_listed(undecided)}.")
    if bundle.held_fixed_reading.disclosure:
        said.append(bundle.held_fixed_reading.disclosure)
    if bundle.cost_unmeasured:
        said.append(bundle.cost_unmeasured + _some_arms(bundle, bundle.cost_unmeasured_cells, "Unmeasured", arm))
    calibration = [
        DisclosureBlock(section="not_tested", source="calibration", text=text)
        for text in dict.fromkeys(_calibration_sentence(tier) for tier in bundle.judge_evidence_tiers)
        if text is not None
    ]
    if not said and not calibration:
        return [DisclosureBlock(section="not_tested", source="not_tested", text="Nothing declared went untested.")]
    return [DisclosureBlock(section="not_tested", source="not_tested", text=text) for text in said] + calibration


def _calibration_sentence(tier: JudgeEvidenceTier) -> str | None:
    """A judge not yet calibrated against people, as a state and its next step; None once people's labels decide it.

    A new judge has no labels, so on a first campaign this is said on every report: it reads as where the judge
    stands and what to do next — directional numbers, and how many outputs to rate — never as something that broke.
    """
    criterion = tier.calibration
    if tier.tier == "calibrated" or criterion.state in ("met", "not_met"):
        return None
    judge = tier.judge_model or "an unnamed judge"
    dim = reading_name(tier.rubric_dim)
    more = criterion.results_needed or criterion.min_results
    if criterion.n == 0:
        return (
            f"Judge {judge} is uncalibrated on {dim}: no human labels yet, so its numbers are directional. Rate "
            f"about {more} of its outputs to calibrate it."
        )
    return (
        f"Judge {judge} is not yet calibrated on {dim}: {criterion.results} output(s) have human labels so far, so "
        f"its numbers are directional. Rate about {more} more to calibrate it."
    )


def _unproven_check_sentence(proof: GoalCheckProofReading) -> str:
    """A goal check not shown to beat doing nothing: what it is, and what its pass rate is not."""
    if proof.refused is not None:
        return (
            f"Goal check {proof.check} is {CHECK_REFUSED_UNDER_CURRENT_GRAMMAR} ({proof.refused}), so "
            f"{proof.runs} run(s) graded it on no arm; where it has a pass rate ({proof.measure_id}) that rate "
            "comes from runs launched before the rule, and is not shown to measure the behaviour."
        )
    why = (
        "its control does not show it tells acting from doing nothing"
        if proof.proof == "refuted"
        else "no control shows it tells acting from doing nothing"
    )
    if proof.unrecorded:
        why += f" ({proof.unrecorded} of {proof.runs} run(s) recorded no proof, assembled without a launch)"
    if proof.stale:
        why += f" ({proof.stale} of {proof.runs} run(s) recorded it proven under an earlier proof rule; relaunch to re-prove)"
    return (
        f"Goal check {proof.check} is {proof.proof}: {why}. Its pass rate ({proof.measure_id}) may be what a "
        "candidate that did nothing would score, so it is not shown to measure the behaviour."
    )


def _no_margin_sentence(bundle: AnalysisContextBundle) -> str | None:
    """The measures tested with no margin to be equivalent on, and how each kind takes one; None when none."""
    catalog = bundle.measure_catalog
    names = list(
        dict.fromkeys(
            tested.name
            for family in bundle.multiple_comparisons.families
            for tested in family.comparisons
            if tested.name in catalog and tested.reading == "measure" and tested.margin_source is None
        )
    )
    if not names:
        return None
    scorers = [name for name in names if name not in METRIC_DESCRIPTORS]
    rates = [name for name in names if name in RUN_MARGIN_MEASURES]
    fixed = [name for name in names if name in METRIC_DESCRIPTORS and name not in RUN_MARGIN_MEASURES]

    def heading(group: list[str]) -> str:
        return ", ".join(catalog[name].reader_name or name for name in group)

    advice = []
    if scorers:
        advice.append(
            f"Declare a margin on {heading(scorers)} by the scorer's name (compare(margins={{{scorers[0]!r}: ...}}))"
            + (
                "; a scorer that returns a number, not a bool, takes its range beside it (ranges=)."
                if any(catalog[name].family == "mechanical" and catalog[name].value_range is None for name in scorers)
                else "."
            )
        )
    if rates:
        advice.append(
            f"Declare a margin on {heading(rates)} (compare(margins={{{rates[0]!r}: ...}})) on every arm's run."
        )
    if fixed:
        advice.append(
            f"No comparison can declare one on {heading(fixed)}: the engine owns "
            f"{'its' if len(fixed) == 1 else 'their'} description, and only accuracy takes a margin of the runs'."
        )
    return (
        f"No margin is declared on {heading(names)}, so no contrast on {'it' if len(names) == 1 else 'them'} can read "
        "equivalent, and not separated never means the arms are alike. " + " ".join(advice)
    )


# =============================================================================
# Screen 6: methods and disclosures
# =============================================================================


def _method_blocks(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    arm: Callable[[str], str],
    *,
    source: ReportSource,
    writer: str | None,
) -> list[ReportBlock]:
    """How everything on the page was measured and made: one disclosure set, whichever door asked."""
    blocks: list[ReportBlock] = []

    def say(source_of: DisclosureSource, text: str | None) -> None:
        if text and text.strip():
            blocks.append(DisclosureBlock(section="methods", source=source_of, text=text.strip()))

    for tier in bundle.judge_evidence_tiers:
        say("measurement", "Judge: " + tier_sentence(tier))
    say("measurement", bundle.request_settings_disclosure)
    for entry in bundle.exclusions:
        if entry.by_cause.get("truncated"):
            say(
                "measurement",
                f"{arm(entry.variant_key)}: {entry.by_cause['truncated']} of {entry.n_trials} trials lost because the "
                f"judge's reply was cut off at its output cap ({loss_words('truncated', exclusions=entry).words}).",
            )
    for flagged in bundle.differential_exclusions:
        others = ", ".join(
            f"{format_number(100 * share)}% of {arm(key)}'s" for key, share in sorted(flagged.other_arms.items())
        )
        say(
            "measurement",
            f"Differential loss: {format_number(100 * flagged.share)}% of {arm(flagged.variant_key)}'s trials were "
            f"lost to {cause_short(flagged.cause)}" + (f", against {others}" if others else "")
            + f"; above {format_number(100 * flagged.threshold)}%, the arms are not compared on the same trials.",
        )
    say("measurement", bundle.model_identity_disclosure)
    for reading in bundle.arm_served_models:
        if reading.state == "pooled":
            say(
                "measurement",
                f"Arm {arm(reading.variant_key)} asked for one model and its candidate was answered by "
                f"{_listed(reading.served_models)}, so its numbers mix those models.",
            )
    say("measurement", bundle.launch_disclosure)
    say("measurement", bundle.measurement_window_disclosure)
    if bundle.latency_contended:
        contended = _some_arms(bundle, bundle.latency_contended_cells, "Read under concurrency", arm)
        say("measurement", bundle.latency_contended + contended)
    for _, sentence in sorted(bundle.short_runs.items()):
        say("runs", sentence)
    _run_disclosures(bundle, say)
    if (footing := _production_footing_sentence(bundle, arm)) is not None:
        say("measurement", footing)
    _design_disclosures(bundle, say, arm)
    _time_axis_disclosures(surface.time_axis, bundle.time_axis_withheld, say)
    sets = ", ".join(ref.label for ref in bundle.case_sets) or "no named case set (each run's own cases)"
    say(
        "generation",
        f"Dataset: {sets}. Design: "
        + ("a question was declared before the run" if bundle.reading_scope.questions_declared else "exploratory")
        + (f", frozen {bundle.design_frozen_at}" if bundle.design_frozen_at else ", design not frozen")
        + ".",
    )
    say("memo", writer or "No memo was written: no model was called, and no number here was written by one.")
    say(
        "generation",
        f"Computed by code from the evidence bundle {source.bundle_fingerprint} (bundle shape {bundle.schema_version}, "
        f"3tears-evals {bundle.package_version or 'version not recorded'}) over {len(bundle.run_ids)} run(s), "
        f"assembled {source.generated_at if writer is None else 'when the memo was written'}; bundle.json beside this "
        "page is that bundle.",
    )
    return blocks


def _run_disclosures(bundle: AnalysisContextBundle, say: Callable[[DisclosureSource, str | None], None]) -> None:
    """The member runs left out, unfinished, short, or of unknown completeness."""
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
    if bundle.completeness_unknown_run_ids:
        say(
            "runs",
            f"Whether {len(bundle.completeness_unknown_run_ids)} run(s) came up short is unknown — they carry no "
            f"completeness record: {_listed(bundle.completeness_unknown_run_ids)}.",
        )
    for cell in bundle.short_cells:
        say("runs", cell.sentence)


def _design_disclosures(
    bundle: AnalysisContextBundle, say: Callable[[DisclosureSource, str | None], None], arm: Callable[[str], str]
) -> None:
    """How the arms were set up and compared: the control, the confounds, the factors, the declared crossing."""
    say("comparisons", bundle.design.stale_control_disclosure)
    say("comparisons", bundle.design.earlier_control.disclosure if bundle.design.earlier_control else None)
    say("comparisons", _verdict_order_sentence(bundle))
    for instability in bundle.subject_key_instabilities:
        if instability.kind == "one_key_many_labels":
            text = f"Subject key {instability.key} is labelled {len(instability.counterparts)} ways: "
        else:
            text = f"Subject label {instability.label} names {len(instability.counterparts)} keys: "
        say("measurement", text + _listed(instability.counterparts) + ".")
    if bundle.apparatus_confounds:
        say(
            "apparatus",
            "The test setup did not hold still across the campaign: "
            + "; ".join(f"{confound.dimension} ({confound.status})" for confound in bundle.apparatus_confounds)
            + ".",
        )
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
    if bundle.declared_crossing is not None:
        say("comparisons", bundle.declared_crossing.sentence)
        for state, lead in (("not_run", "Declared arms never run"), ("skipped_by_design", "Arms skipped by design")):
            cells = [" × ".join(cell.levels.values()) for cell in bundle.declared_crossing.cells if cell.state == state]
            if cells:
                say("comparisons", f"{lead}: {'; '.join(cells)}.")
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
        if row.cannot_be_an_arm:
            why = row.cannot_be_an_arm.rstrip(".")
            say("comparisons", f"Declared axis {row.name} reads unswept whatever the runs did: {why}.")
    for merge in bundle.refused_merges:
        dimensions = f" on {_listed(merge.dimensions)}" if merge.dimensions else ""
        say(
            "apparatus",
            f"Arm {arm(merge.variant_key)} was measured under {len(merge.apparatus_class_ids)} test setups that did "
            f"not pool ({merge.reason.replace('_', ' ')}{dimensions}).",
        )
    if bundle.refused_merges_omitted:
        say(
            "apparatus",
            f"{bundle.refused_merges_omitted} more pair(s) of one arm's test setups did not pool; they are left out of "
            "this list, the smallest first.",
        )


def _time_axis_disclosures(
    axis: TimeAxis | None, withheld: str | None, say: Callable[[DisclosureSource, str | None], None]
) -> None:
    """What the time axis's positions are, when the runs span two builds or two days."""
    say("measurement", withheld)
    if axis is None:
        return
    if axis.basis == "release":
        say("measurement", f"The runs span builds of {axis.release_label}, in the order each first ran.")
    else:
        say("measurement", f"The runs span UTC days, not builds: {axis.basis_reason}.")
    for position in axis.positions:
        say("measurement", position.control_disclosure)


def _verdict_order_sentence(bundle: AnalysisContextBundle) -> str | None:
    """The tie-break order the campaign declared over its merit axes, with the bars each tier holds — or nothing."""
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


def _production_footing_sentence(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str | None:
    """Say, per arm, what its runs set away from the subject's production configuration; None when nothing moved."""
    parts: list[str] = []
    qualified = False
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
        qualified = qualified or bool(said)
        parts.append(f"arm {arm(variant_key)} " + ("; ".join(said) or "moved nothing off production"))
    if not qualified:
        return None
    return (
        "A per-reply cost is what production would spend only where its arm's runs moved nothing off the subject's "
        "production configuration: " + "; ".join(parts) + "."
    )


def _some_arms(
    bundle: AnalysisContextBundle, cells: Sequence[CellCoordinate], lead: str, arm: Callable[[str], str]
) -> str:
    """`` <lead>: <arms>.`` naming the arms a sentence holds for, or nothing when it holds for every arm."""
    keys = sorted({cell.variant_key for cell in cells})
    if set(keys) >= {board.variant_key for board in bundle.scoreboard}:
        return ""
    return f" {lead}: {'; '.join(arm(key) for key in keys)}."


def _listed(ids: Sequence[str]) -> str:
    """Ids, as a disclosure lists them."""
    return ", ".join(ids)


__all__ = [
    "NO_ANALYSIS",
    "analysis_bundle",
    "analysis_verdict",
    "NO_STRATUM",
    "OPEN_TRIALS",
    "TOO_FEW_CASES",
    "bar_verdicts",
    "build_code_only_report",
    "build_report",
    "contrast_verdicts",
    "guardrail_verdicts",
]

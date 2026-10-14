"""Build a :class:`~threetears.evals.analysis.report.model.Report` — from a stored analysis, or from the evidence alone.

**One builder, one page, every door.** :func:`build_report` lays out a stored analysis over the bundle frozen with it
at generation; :func:`build_code_only_report` lays out a campaign's evidence assembled now, when no analysis exists.
Both go through :func:`_page`, so the two differ only in the memo section: every count, verdict, chart and
disclosure is computed here, from the bundle alone, and no door — the quick path's ``compare``, the command line, an
MCP action — adds, drops or reorders a block.

The page, screen by screen:

1. **The first screen.** The title (the declared question, or the campaign's name and what it compares); the arms
   line (each arm named in full once, the control marked, the date); the verdict line
   (:func:`~threetears.evals.analysis.report.verdict.verdict_line`), result first; the scoreboard, a bare table of
   counts and the cost per reply, and its outcome bar; then the caveats, in normal weight.
2. **Decision.** A declared decision rule, clause by clause, settled or not (:mod:`.decision`).
3. **What it means.** Each finding as one sentence its writer wrote and a table code built from the readings it
   chose, everything else in a closed ``Evidence`` fold; the recommendations and next steps. Then the contrasts that
   can resolve (one line when none can), the guardrails, and the readings by stratum and label.
4. **Cases** and **Trials.** One row per case, each arm's trials as marks; one row per trial, each record folded
   under ``T4 · case 1 · arm · outcome``.
5. **Not tested** and **Methods.** What the campaign could not speak to; then a table (arms, judge, settings sent,
   cost, the analysis's writer, the bundle) and the engine's disclosures folded.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime
from itertools import chain

from threetears.evals.analysis.agreement import tier_sentence
from threetears.evals.analysis.arms import arm_names, multi_rig_variants, short_digest, surface_order
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
from threetears.evals.analysis.references import cell_index, resolve_reading
from threetears.evals.analysis.report.figures import (
    NO_SUCCESSFUL_RESULTS,
    UNITLESS,
    MeasureRuler,
    measure_rulers,
    surface_order_sentence,
)
from threetears.evals.analysis.report.model import (
    CellValue,
    ChartBlock,
    SECTION_TITLES,
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
from threetears.evals.analysis.report.decision import evaluate_rule, parse_decision_rule, rule_readings
from threetears.evals.analysis.report.memo_style import MemoStyle, decision_disclosures, row_words
from threetears.evals.analysis.report.verdict import (
    MIN_CASES,
    CauseWords,
    arm_exclusions,
    cause_short,
    caveat_lines,
    rule_sentence,
    fail_words,
    loss_words,
    verdict_line,
)
from threetears.evals.analysis.report.words import (
    BAR_DECISION_WORDS,
    COMPARISON_VERDICT_WORDS,
    CONFIDENCE_WORDS,
    GUARDRAIL_DECISION_WORDS,
    evidence_rows,
    positions,
    reading_name,
    stands_on_words,
)
from threetears.evals.analysis.reporting import self_judging_text
from threetears.evals.analysis.style import ArmName, arm_name, humanize, lint_prose, sentences
from threetears.evals.analysis.stats import EQUIVALENCE_NEEDS_RANGE, INTERVAL_LEVEL
from threetears.evals.analysis.quantities import with_unit
from threetears.evals.kernel.analysis_measures import MeasureSummary
from threetears.evals.kernel.authored import Finding
from threetears.evals.kernel.campaign import EvalAnalysis, FindingResolution, ReadingKind, VariantIndexEntry
from threetears.evals.kernel.declaration import (
    JUDGED_MERIT_AXIS,
    CampaignDesign,
    Question,
    exploratory_reading,
    preregistered_questions,
)
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
from threetears.evals.schema.models import CHECK_REFUSED_UNDER_CURRENT_GRAMMAR, ClientRequestSettings

#: What the methods say of a report no analysis was written for.
NO_ANALYSIS = "none written; the analysis_generate action writes one from this same evidence"

#: The ``Scope`` fact on a finding resting wholly on readings no question declared before the run asked about.
EXPLORATORY_FINDING = "exploratory: no declared question asked about its readings, so it is a lead"

#: How a guardrail is decided, said once in the methods' disclosures.
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
        campaign_name: The campaign's name, which the title reads when no question is declared; ``None`` names the
            campaign by its id.

    Returns:
        The report, ``basis="analysis"``.
    """
    generation = analysis.generation
    bundle = analysis_bundle(analysis)
    style = _memo_style(bundle, analysis)
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
        memo=lambda charts, names: _memo_blocks(analysis, bundle, charts, names, style),
        finding_count=len(analysis.document.findings),
        writer=(
            f"{generation.generator_model}, prompt {generation.prompt_id} version {generation.prompt_version}, "
            f"{_date(generation.generated_at)}"
        ),
        style=style,
    )


def _memo_style(bundle: AnalysisContextBundle, analysis: EvalAnalysis) -> MemoStyle:
    """The style pass a stored memo's words go through, with the declared question and the rule's outcome to hold
    them to."""
    names = {**_arm_names(bundle), **_arm_names(bundle, analysis.variant_index)}
    questions = _preregistered(bundle)
    control = _control_key(bundle)
    candidates = [board.variant_key for board in bundle.scoreboard if board.variant_key != control]
    rule = next((parsed for question in questions if (parsed := parse_decision_rule(question.text))), None)
    verdict = sentence = None
    if rule is not None and control is not None and len(candidates) == 1:
        verdict = evaluate_rule(rule, rule_readings(bundle, control, candidates[0]))
        namer = _namer(names)
        sentence = rule_sentence(verdict, control=namer(control), candidate=namer(candidates[0]), bundle=bundle)
    return MemoStyle(
        bundle=bundle,
        names=names,
        question=questions[0].text if questions else None,
        rule=verdict,
        rule_sentence=sentence,
    )


def build_code_only_report(
    bundle: AnalysisContextBundle, *, assembled_at: str, campaign_name: str | None = None
) -> Report:
    """Lay a campaign's evidence out as its page, when no analysis exists to report through.

    Everything here is code's, and the methods say no analysis was written. The report model refuses an author's
    words on a code-only report.

    Args:
        bundle: The campaign's evidence, as :func:`~threetears.evals.analysis.bundle.assemble.assemble_context_bundle`
            assembled it.
        assembled_at: When the bundle was assembled (ISO-8601), stated as the report's date.
        campaign_name: The campaign's name, which the title reads when no question is declared; ``None`` names the
            campaign by its id.

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
    memo = _question_blocks(bundle)
    return _page(bundle, basis="code_only", source=source, memo=lambda *_: memo, finding_count=0, writer=None)


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
    names = _arm_names(bundle)
    return verdict_line(bundle, _namer(names), case_name=_case_namer(bundle)).text


def _page(
    bundle: AnalysisContextBundle,
    *,
    basis: ReportBasis,
    source: ReportSource,
    memo: Callable[[_ChartLedger, dict[str, ArmName]], list[ReportBlock]],
    finding_count: int,
    writer: str | None,
    style: MemoStyle | None = None,
) -> Report:
    """The page every door renders: the same screens, from one bundle, whatever wrote the memo.

    The memo and the contrasts are laid out before the scoreboard, though they print after it: each chart they leave
    out is counted on the way, for the scoreboard's one line about what was not charted.
    """
    surface = bundle_decision_surface(bundle)
    names = _arm_names(bundle)
    arm = _namer(names)
    cell_name = _cell_namer(surface, names)
    charts = _ChartLedger(bundle, arm)
    guardrails_read = guardrail_verdicts(surface, bundle.guardrails, cell_name)
    bars_read = bar_verdicts(surface, cell_name)
    contrasts_read = contrast_verdicts(bundle, surface)
    written = memo(charts, names)
    contrasts = _contrast_blocks(bundle, surface, contrasts_read, guardrails_read, charts, names)
    contrasts += _bar_blocks(surface, bars_read)
    notes = [block for block in contrasts if block.section == "methods"]
    contrasts = [block for block in contrasts if block.section != "methods"]
    if style is not None:
        notes += [_note(text, "memo") for text in decision_disclosures(style)]
        notes += [
            DisclosureBlock(section="methods", source="memo", text=text, fold="Style") for text in style.offending()
        ]
    name = source.campaign_name or f"Campaign {source.campaign_id}"
    blocks: list[ReportBlock] = [
        DisclosureBlock(section="header", source="arms", text=_arms_line(bundle, names, source)),
        DisclosureBlock(
            section="header",
            source="verdict",
            text=verdict_line(bundle, arm, name=name, case_name=_case_namer(bundle)).text,
        ),
        *_scoreboard_blocks(bundle, arm, charts),
        *_decision_blocks(bundle, arm),
        *written,
        *contrasts,
        *_case_blocks(bundle, arm),
        *_trial_blocks(bundle, arm),
        *_not_tested_blocks(bundle, surface, arm),
        *_method_blocks(bundle, surface, arm, names, source=source, writer=writer, notes=notes, style=style),
    ]
    # In reading order, each section's blocks in the order laid out (a code-only page's unasked readings are said
    # with what was not tested, though they are laid out with the questions).
    order = list(SECTION_TITLES)
    blocks.sort(key=lambda block: order.index(block.section))
    return Report(
        basis=basis,
        title=_title(bundle, name, arm),
        finding_count=finding_count,
        source=source,
        blocks=blocks,
        verdicts=[*guardrails_read, *bars_read, *contrasts_read],
        bundle=bundle,
    )


# =============================================================================
# Screen 1: the title, the arms, the verdict and the scoreboard
# =============================================================================


def _control_key(bundle: AnalysisContextBundle) -> str | None:
    """The control's variant key, when it ran."""
    control = bundle.design.control_arm.variant_key if bundle.design.control_arm is not None else None
    return control if any(board.variant_key == control for board in bundle.scoreboard) else None


def _arm_names(bundle: AnalysisContextBundle, index: Sequence[VariantIndexEntry] | None = None) -> dict[str, ArmName]:
    """Every arm's name, given label first; an arm whose label another shares is told apart by its digest.

    ``index`` names the arms of another variant index (a stored analysis's own) by the same rule; the bundle's when
    None.
    """
    displays = arm_names(bundle.variant_index if index is None else index)
    control = _control_key(bundle)
    names = {key: arm_name(display, control=key == control) for key, display in displays.items()}
    labels = [name.label for name in names.values()]
    return {
        key: name
        if labels.count(name.label) == 1
        else ArmName(label=f"{name.label} ({short_digest(key)})", model=name.model, control=name.control)
        for key, name in names.items()
    }


def _namer(names: dict[str, ArmName]) -> Callable[[str], str]:
    """Name an arm by its variant key as every table of the report names it: by its label, never as a digest."""

    def name(variant_key: str) -> str:
        if variant_key in names:
            return names[variant_key].label
        return f"unplaced arm {short_digest(variant_key)}"

    return name


def _title(bundle: AnalysisContextBundle, name: str, arm: Callable[[str], str]) -> str:
    """The title: the question the campaign asks, in plain words, on one line."""
    if name.rstrip().endswith("?"):
        return name
    if questions := _preregistered(bundle):
        asked = sentences(questions[0].text)[0]
        if not lint_prose(asked):
            return asked
    control = _control_key(bundle)
    others = [arm(board.variant_key) for board in bundle.scoreboard if board.variant_key != control]
    if control is None or not others:
        return name
    asked = f"does {' or '.join(others)} do better than {arm(control)}?"
    # A name that only lists the arms (the quick path's ``a vs b``) adds nothing to the question it asks.
    if all(part.strip() in {arm(control), *others} for part in name.split(" vs ")):
        return asked[:1].upper() + asked[1:]
    return f"{name}: {asked}"


def _arms_line(bundle: AnalysisContextBundle, names: dict[str, ArmName], source: ReportSource) -> str:
    """Line 2: each arm named in full, once, the control first and marked, then the date."""
    control = _control_key(bundle)
    keys = [board.variant_key for board in bundle.scoreboard]
    ordered = ([control] if control is not None else []) + [key for key in keys if key != control]
    arms = " vs ".join(names[key].first() if key in names else _namer(names)(key) for key in ordered)
    return " · ".join(part for part in (arms or "no arm ran", _date(source.generated_at)) if part)


def _date(stamp: str) -> str:
    """An ISO-8601 instant as the page dates it (``14 Oct 2026``), or as given when it does not parse."""
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        # NOSILENT: an unparseable stamp is printed as stored rather than dropped or reformatted
        return stamp
    return f"{moment.day} {moment:%b %Y}"


def _dollars(amount: float | None) -> str | None:
    """A cost as the scoreboard prints it: ``$0.003092``; None where none was measured."""
    return None if amount is None else f"${format_number(amount)}"


def _scoreboard_blocks(
    bundle: AnalysisContextBundle, arm: Callable[[str], str], charts: _ChartLedger
) -> list[ReportBlock]:
    """The scoreboard table, its outcome bar, then the lines qualifying it, in normal weight."""
    boards = bundle.scoreboard
    if not boards:
        return [
            DisclosureBlock(
                section="scoreboard", source="runs", text="No trial ran: the campaign resolved no result in this scope."
            )
        ]
    control = _control_key(bundle)
    earlier = bundle.design.earlier_control
    lost_any = any(board.lost for board in boards)
    other_causes = sorted({cause for board in boards for cause in board.failed if cause not in ("check", "judged")})
    rows: list[dict[str, CellValue]] = []
    for board in boards:
        row: dict[str, CellValue] = {
            "arm": ("▸ " if board.variant_key == control else "")
            + arm(board.variant_key)
            + (f" ({earlier_control_mark(earlier)})" if earlier and board.variant_key == earlier.variant_key else ""),
            "ran": board.ran,
            "usable": board.usable,
            "passed": board.passed,
            "check": board.failed.get("check", 0),
            "judged": board.failed.get("judged", 0),
            "cost": _dollars(board.reply_cost_usd),
        }
        if lost_any:
            row["lost"] = sum(board.lost.values())
        if other_causes:
            row["other"] = sum(board.failed.get(cause, 0) for cause in other_causes)
        rows.append(row)
    columns = [
        TableColumn(key="arm", header="Arm"),
        TableColumn(key="ran", header="Ran"),
        *([TableColumn(key="lost", header="Lost", term="lost")] if lost_any else []),
        TableColumn(key="usable", header="Usable", term="usable"),
        TableColumn(key="passed", header="Passed"),
        TableColumn(key="check", header="Failed by check"),
        TableColumn(key="judged", header="Failed by judge"),
        *([TableColumn(key="other", header="Failed otherwise")] if other_causes else []),
        TableColumn(key="cost", header="Cost per reply"),
    ]
    rows = [{column.key: row[column.key] for column in columns} for row in rows]
    blocks: list[ReportBlock] = [
        TableBlock(
            section="scoreboard",
            name="scoreboard",
            title="Scoreboard",
            columns=columns,
            rows=[table_row(row) for row in rows],
            order="the control first, then the other arms by name",
            total_rows=len(rows),
        ),
        ChartBlock(
            section="scoreboard",
            chart="outcome_bar",
            title="Outcomes",
            bars=[_outcome_bar(board, arm) for board in boards],
        ),
    ]
    blocks += [
        DisclosureBlock(section="scoreboard", source="caveat", text=line, term="case floor")
        for line in caveat_lines(bundle, arm)
    ]
    if other_causes:
        blocks.append(
            DisclosureBlock(
                section="scoreboard",
                source="caveat",
                text=f"Failed otherwise: {', '.join(cause_short(cause) for cause in other_causes)}.",
            )
        )
    blocks.append(DisclosureBlock(section="scoreboard", source="design", text=_design_line(bundle)))
    if (sentence := charts.sentence()) is not None:
        blocks.append(DisclosureBlock(section="scoreboard", source="charts", text=sentence))
    if bundle.all_failed_cells:
        every_failed = list(dict.fromkeys(arm(cell.variant_key) for cell in bundle.all_failed_cells))
        blocks.append(
            DisclosureBlock(
                section="scoreboard",
                source="runs",
                text=(
                    f"Every trial of {', '.join(every_failed)} failed: the model refused or errored on every call. "
                    "It took no turn, so it has no cost or latency to read, and every failure counts against it."
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
        else f"{repeats[0]} repeat{'' if repeats[0] == 1 else 's'} each"
        if len(repeats) == 1
        else f"{repeats[0]} to {repeats[-1]} repeats each"
    )
    return f"{case_words}, {repeat_words}."


def _decision_blocks(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> list[ReportBlock]:
    """A declared decision rule, clause by clause, on each arm against the control: a table, never prose."""
    control = _control_key(bundle)
    questions = _preregistered(bundle)
    rule = next((parsed for question in questions if (parsed := parse_decision_rule(question.text))), None)
    if rule is None or control is None:
        return []
    rows: list[dict[str, CellValue]] = []
    for board in bundle.scoreboard:
        if board.variant_key == control:
            continue
        readings = rule_readings(bundle, control, board.variant_key)
        verdict = evaluate_rule(rule, readings)
        for clause in verdict.clauses:
            shown: list[tuple[str, str | None, str | None]] = []
            if clause.clause.kind == "cost_lower":
                shown = [("cost per reply", _dollars(readings.control_cost), _dollars(readings.candidate_cost))]
            elif clause.clause.kind == "rate_drop":
                shown = [
                    (
                        rate.name,
                        f"{rate.control_passed} of {rate.control_n}",
                        f"{rate.candidate_passed} of {rate.candidate_n}",
                    )
                    for rate in readings.rates
                ]
            for index, (reading, ours, theirs) in enumerate(shown or [("—", None, None)]):
                rows.append(
                    {
                        "clause": clause.clause.text if index == 0 else None,
                        "arm": arm(board.variant_key),
                        "reading": reading,
                        "control": ours,
                        "candidate": theirs,
                        "outcome": (
                            "settled: keep"
                            if clause.outcome == "keep"
                            else "settled: switch"
                            if clause.outcome == "switch"
                            else f"unsettled: {clause.reason}"
                        )
                        if index == 0
                        else None,
                    }
                )
    return [
        TableBlock(
            section="decision",
            name="decision",
            title="Decision rule",
            columns=[
                TableColumn(key="clause", header="Clause"),
                TableColumn(key="arm", header="Arm"),
                TableColumn(key="reading", header="Reading"),
                TableColumn(key="control", header=arm(control)),
                TableColumn(key="candidate", header="Arm's"),
                TableColumn(key="outcome", header="Outcome", term="settled"),
            ],
            rows=[table_row(row) for row in rows],
            order="clause by clause, each arm against the control",
            total_rows=len(rows),
        )
    ]


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
        """Every row not charted, in one sentence; None when nothing was left out, or no arm could be charted at all
        (the caveat under the scoreboard says so)."""
        if all(usable < MIN_CASES for usable in self._usable.values()):
            return None
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


def _memo_blocks(
    analysis: EvalAnalysis,
    bundle: AnalysisContextBundle,
    charts: _ChartLedger,
    names: dict[str, ArmName],
    style: MemoStyle,
) -> list[ReportBlock]:
    """The memo: answers to the declared questions, the findings, the recommendations and the next steps.

    A finding is one sentence its writer wrote and a table code built from the readings the writer chose; everything
    else it carries sits in a closed ``Evidence`` fold. Every word the memo contributes passes ``style`` first
    (:mod:`~threetears.evals.analysis.report.memo_style`).
    """
    document = analysis.document
    # The analysis's own variant index names the cells its memo cites, by the page's rule.
    arm = _ref_namer(analysis.decision_surface, {**names, **_arm_names(bundle, analysis.variant_index)})
    questions = _preregistered(bundle)
    blocks: list[ReportBlock] = []
    for answer in document.questions:
        replaced = len(style.replaced)
        body = style.sentence(answer.answer)
        # An answer the rule's settled outcome replaced is code's answer now, not the memo's resolution.
        resolution = "answered" if len(style.replaced) > replaced else answer.resolution
        blocks.append(
            TextBlock(
                section="memo",
                role="answer",
                body=body,
                rests_on=list(answer.rests_on),
                facts=[Fact(name="Resolution", value=resolution)],
            )
        )
    resolutions: list[FindingResolution | None] = (
        list(analysis.resolutions) if analysis.resolutions else [None] * len(document.findings)
    )
    for position, (finding, resolution) in enumerate(zip(document.findings, resolutions, strict=True)):
        exploratory = not questions or _rests_on_exploratory(resolution, analysis.decision_surface, questions)
        blocks.extend(
            _finding_blocks(analysis, position, finding, resolution, arm, charts, style, exploratory=exploratory)
        )
    for decision in document.decisions:
        facts = [Fact(name="Confidence", value=CONFIDENCE_WORDS[decision.confidence])]
        if decision.cells:
            facts.append(Fact(name="Arms", value="; ".join(arm(cell) for cell in decision.cells)))
        if decision.disposition == "adopted" and (standing := _guardrail_standing(decision.cells, bundle, arm)):
            facts.append(Fact(name="Guardrails", value=standing))
        if decision.rests_on:
            facts.append(Fact(name="Findings", value=positions(list(decision.rests_on))))
        if cases := _cases_of(bundle, decision.rests_on, document.findings):
            facts.append(Fact(name="See", value=", ".join(cases)))
        blocks.append(
            TextBlock(
                section="memo",
                role="recommendation",
                body=style.sentence(decision.proposal, decides=decision.disposition == "adopted"),
                rests_on=list(decision.rests_on),
                facts=[Fact(name="Disposition", value=decision.disposition)],
            )
        )
        if (revisit := style.prose(decision.revisit_when)).strip():
            blocks.append(
                TextBlock(section="memo", role="revisit_when", body=revisit, rests_on=list(decision.rests_on))
            )
        blocks.append(_facts_table(facts, name="recommendation_facts"))
    coverage_status = {row.name: row.status for row in analysis.coverage.levers}
    for step in document.next:
        facts = [Fact(name="Leverage", value=step.leverage)]
        if step.lever:
            status = coverage_status.get(step.lever) or "no coverage row"
            facts.append(Fact(name="Lever", value=f"{step.lever} ({status})"))
        # The facts ride on the block for a program; the page prints them once, in the fold below.
        blocks.append(
            TextBlock(section="memo", role="next_step", body=style.sentence(step.title, decides=False), facts=facts)
        )
        if (why := style.prose(step.why)).strip():
            blocks.append(TextBlock(section="memo", role="next_step_why", body=why, fold="Why"))
        blocks.append(_facts_table(facts, name="next_step_facts", fold="Why"))
    blocks.extend(_coverage_blocks(analysis))
    return blocks


def _facts_table(facts: Sequence[Fact], *, name: str, finding: int | None = None, fold: str = "Evidence") -> TableBlock:
    """Facts code states beside an author's words, as a two-column table in a closed fold — never a ``Key: value`` line."""
    rows = [{"item": TableCell(value=fact.name), "value": TableCell(value=fact.value)} for fact in facts]
    return TableBlock(
        section="memo",
        finding=finding,
        name=name,
        title="Evidence",
        columns=[TableColumn(key="item", header="Item"), TableColumn(key="value", header="Value")],
        rows=rows,
        order="as code states them",
        total_rows=len(rows),
        fold=fold,
    )


def _ref_namer(surface: DecisionSurface, names: dict[str, ArmName]) -> Callable[[str], str]:
    """Name the arm a cell reference points at by its label, with its test setup where it ran under two."""
    multi = multi_rig_variants(surface.cells)
    by_ref = {ref: (cell.variant_key, cell.apparatus_class_id) for ref, cell in cell_index(surface).items()}
    namer = _namer(names)

    def name(ref: str) -> str:
        variant, rig = by_ref.get(ref, (variant_of_cell_ref(ref) or ref, ""))
        return namer(variant) + (f" (test setup {short_digest(rig)})" if variant in multi and rig else "")

    return name


def _coverage_blocks(analysis: EvalAnalysis) -> list[ReportBlock]:
    """The coverage map joined to the findings and the next steps, on the stored lever names (#630, #631).

    One row per lever: its status, the findings naming it in their ``axes`` (or "no finding"), and the next steps
    whose ``lever`` names it; a ``thin`` or ``unswept`` lever no step names says so. Then one line naming the levers
    no finding names, when there are any.
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
    blocks: list[ReportBlock] = [
        TableBlock(
            section="memo",
            name="coverage",
            title="Coverage",
            columns=[
                TableColumn(key="lever", header="Lever"),
                TableColumn(key="status", header="Coverage"),
                TableColumn(key="findings", header="Findings"),
                TableColumn(key="next", header="Next step"),
            ],
            rows=[table_row(row) for row in rows],
            order="as the coverage map lists the levers",
            total_rows=len(rows),
        )
    ]
    if unnamed:
        blocks.append(
            DisclosureBlock(section="memo", source="memo", text=f"Levers no finding names: {_listed(unnamed)}.")
        )
    return blocks


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
    style: MemoStyle,
    *,
    exploratory: bool,
) -> list[ReportBlock]:
    """One finding: its one sentence, the table code built from its readings, their charts, then a closed fold.

    A title restating a number the bundle measured is replaced by the table's own row in words: a measured fact is
    stated by code, flatly.
    """
    facts = [Fact(name="Confidence", value=CONFIDENCE_WORDS[finding.confidence])]
    if exploratory:
        facts.append(Fact(name="Scope", value=EXPLORATORY_FINDING))
    if resolution is not None:
        facts.append(Fact(name="Based on", value=stands_on_words(analysis, resolution.evidence_tier)))
    if finding.axes:
        facts.append(Fact(name="Lever", value=", ".join(finding.axes)))
    if finding.trials:
        facts.append(Fact(name="Trials", value=", ".join(finding.trials)))
    if finding.invalidates:
        facts.append(Fact(name="Overturns", value=positions(list(finding.invalidates))))
    table = _evidence_table(analysis, position, resolution, arm) if resolution and resolution.evidence else None
    fallback = None
    if table is not None and len(table.columns) > 2:
        reading = table.columns[1]
        fallback = row_words(reading.header, [(str(row["arm"].value), row[reading.key].text) for row in table.rows])
    blocks: list[ReportBlock] = [
        # The facts ride on the block for a program; the page prints them once, in the finding's fold.
        TextBlock(
            section="memo",
            role="finding_title",
            finding=position,
            body=style.sentence(finding.title, fallback=fallback),
            facts=facts,
            anchor=f"F{position + 1}",
        )
    ]
    if table is not None and resolution is not None:
        blocks.append(table)
        blocks.extend(_finding_charts(analysis, position, resolution, charts, arm))
    blocks.append(_facts_table(facts, name="finding_facts", finding=position))
    if (body := style.prose(finding.body)).strip():
        blocks.append(TextBlock(section="memo", role="finding_body", finding=position, body=body, fold="Evidence"))
    for caveat in finding.caveats:
        if (said := style.prose(caveat.text)).strip():
            blocks.append(
                TextBlock(
                    section="memo",
                    role="caveat",
                    finding=position,
                    body=said,
                    facts=[Fact(name="Kind", value=caveat.kind)],
                    fold="Evidence",
                )
            )
    if (durable := style.prose(finding.durable)).strip():
        blocks.append(
            TextBlock(section="memo", role="carried_forward", finding=position, body=durable, fold="Evidence")
        )
    return blocks


def _evidence_table(
    analysis: EvalAnalysis, position: int, resolution: FindingResolution, arm: Callable[[str], str]
) -> TableBlock:
    """A finding's readings as code states them: one row per arm, one column per reading, then the cases."""
    rows = evidence_rows(analysis, resolution.evidence, arm)
    readings = list(dict.fromkeys(str(row["measure"].value) for row in rows))
    keys = {reading: f"reading_{index}" for index, reading in enumerate(readings)}
    by_arm: dict[str, dict[str, TableCell]] = {}
    for row in rows:
        name = str(row["arm"].value)
        line = by_arm.setdefault(
            name, {"arm": TableCell(value=name), **{key: TableCell(value=None) for key in keys.values()}}
        )
        line[keys[str(row["measure"].value)]] = row["value"]
        cases = row["cases"].value
        held = line.get("cases")
        if held is None or (isinstance(cases, int) and isinstance(held.value, int) and cases > held.value):
            line["cases"] = TableCell(value=cases)
    pivot = [line for line in by_arm.values()]
    return TableBlock(
        section="memo",
        finding=position,
        name="evidence",
        title=f"F{position + 1} readings",
        columns=[
            TableColumn(key="arm", header="Arm"),
            *(TableColumn(key=key, header=reading) for reading, key in keys.items()),
            TableColumn(key="cases", header="Cases"),
        ],
        rows=pivot,
        order="each arm as the author listed it, each reading a column",
        total_rows=len(pivot),
    )


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
                    title=f"F{position + 1}: {what}",
                    rows=rows,
                    unit=ruler.unit,
                )
            )
    return blocks


def _preregistered(bundle: AnalysisContextBundle) -> list[Question]:
    """The live questions declared before the campaign's first run — the ones a finding can answer."""
    design = bundle.declared_design
    if design is None or not bundle.question_declared:
        return []
    # The one rule the families and the reading scope were read by, against the instant the design froze.
    return preregistered_questions(design, bundle.design_frozen_at)


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
    """On a code-only report: the declared questions past the first (the title carries it), and the unasked readings."""
    design = bundle.declared_design
    scope = bundle.reading_scope
    blocks: list[ReportBlock] = []
    if design is not None and len(design.questions) > 1:
        rows: list[dict[str, CellValue]] = [
            {
                "question": f"Q{index}",
                "asks": question.text,
                "status": "retired" if question.retired_at else "live",
                "answer": "unanswered: no analysis",
            }
            for index, question in enumerate(design.questions, start=1)
        ]
        blocks.append(
            TableBlock(
                section="memo",
                name="questions",
                title="Declared questions",
                columns=[
                    TableColumn(key="question", header="Question"),
                    TableColumn(key="asks", header="Asks"),
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
                section="not_tested",
                source="scope",
                text=f"No declared question asks about these, so read them as leads: {_listed(exploratory)}.",
                term="exploratory",
            )
        )
    return blocks


def _contrast_blocks(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    contrasts_read: Sequence[Verdict],
    guardrails_read: Sequence[Verdict],
    charts: _ChartLedger,
    names: dict[str, ArmName],
) -> list[ReportBlock]:
    """The contrasts against the control, their charts, the guardrails, and the readings by stratum and label.

    Engine text that explains how they were tested goes to the methods' closed ``Disclosures`` fold.
    """
    arm = _namer(names)
    blocks: list[ReportBlock] = []
    blocks.extend(_self_judging_blocks(bundle, surface, arm))
    blocks.extend(_guardrail_standing_blocks(bundle, arm))
    blocks.extend(_comparison_blocks(bundle, surface, contrasts_read, charts, arm))
    blocks.extend(_guardrail_blocks(surface, bundle.guardrails, guardrails_read))
    blocks.extend(_strata_blocks(surface, names))
    blocks.extend(_label_blocks(surface, names))
    return blocks


def _note(text: str, source: DisclosureSource = "comparisons") -> DisclosureBlock:
    """Engine text for the methods' closed ``Disclosures`` fold."""
    return DisclosureBlock(section="methods", source=source, text=text, fold="Disclosures")


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
        if breached:
            blocks.append(
                DisclosureBlock(
                    section="contrasts",
                    source="guardrails",
                    text=f"{arm(variant)} breached the guardrail {', '.join(breached)}, so it is not adopted.",
                    term="guardrail",
                )
            )
        if undecided:
            blocks.append(
                DisclosureBlock(
                    section="contrasts",
                    source="guardrails",
                    text=f"{arm(variant)} is not known to be safe on the guardrail {', '.join(undecided)}.",
                    term="guardrail",
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
    return f"; immaterial: the observed delta is inside {where}, which does not show the true difference is that small"


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
    arm = _cell_namer(surface, _arm_names(bundle))
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
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    verdicts: Sequence[Verdict],
    charts: _ChartLedger,
    arm: Callable[[str], str],
) -> list[ReportBlock]:
    """The contrasts that can resolve, as a table with their statistics folded away, and a chart per reading.

    A contrast with a side under the case floor is never a row: when none can resolve, the section is one line, and
    otherwise one line counts the ones left out. A reading one side carries on no trial is no contrast at all, and the
    methods count it.
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
    under: list[int] = []
    questions = {question.id: index for index, question in enumerate(_preregistered(bundle), start=1)}
    for family in comparisons.families:
        for comparison in family.comparisons:
            verdict = next(typed)
            if min(comparison.control.n_cases, comparison.contrast.n_cases) == 0:
                continue
            ruler = rulers.get(comparison.name, UNITLESS) if comparison.reading == "measure" else UNITLESS
            if (small := _small_n(comparison)) is not None:
                under.append(small)
                continue
            interval = _interval_on_delta(comparison.interval, ruler, family.interval_level)
            shown.append(
                {
                    "reading": TableCell(value=verdict.heading),
                    "contrast": TableCell(value=verdict.arm),
                    "change": _figure(comparison.delta, ruler),
                    "interval": TableCell(value=interval),
                    "verdict": TableCell(value=verdict.words),
                    "cases": TableCell(value=_comparison_cases(comparison)),
                }
            )
            statistics.append(
                {
                    "question": TableCell(
                        value=f"Q{questions[family.question_id]}"
                        if family.question_id in questions
                        else "campaign-wide"
                    ),
                    "reading": TableCell(value=verdict.heading),
                    "contrast": TableCell(value=verdict.arm),
                    "control_mean": _figure(comparison.control.mean, ruler),
                    "arm_mean": _figure(comparison.contrast.mean, ruler),
                    "hedges_g": TableCell(value=comparison.hedges_g),
                    "pairing_r": TableCell(
                        value=comparison.pairing_correlation.r if comparison.pairing_correlation is not None else None
                    ),
                    "p_adjusted": TableCell(value=comparison.p_adjusted),
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
                        label=verdict.arm,
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
    blocks: list[ReportBlock] = []
    if under and not shown:
        blocks.append(
            DisclosureBlock(
                section="contrasts",
                source="comparisons",
                text=(
                    f"No contrast can resolve: a side has {min(under)} usable case{'' if min(under) == 1 else 's'}, "
                    f"and a difference needs {MIN_CASES}."
                ),
                term="case floor",
            )
        )
    elif under:
        blocks.append(
            DisclosureBlock(
                section="contrasts",
                source="comparisons",
                text=f"{len(under)} more contrasts are not shown: a side has under {MIN_CASES} usable cases.",
                term="case floor",
            )
        )
    if shown:
        control = _control_key(bundle)
        blocks += [
            TableBlock(
                section="contrasts",
                name="comparisons",
                title=f"Against {arm(control)}" if control is not None else "Against the control",
                columns=[
                    TableColumn(key="reading", header="Reading"),
                    TableColumn(key="contrast", header="Arm"),
                    TableColumn(key="change", header="Change"),
                    TableColumn(key="interval", header="Interval", term="interval"),
                    TableColumn(key="verdict", header="Verdict", term="separated"),
                    TableColumn(key="cases", header="Cases"),
                ],
                rows=shown,
                order="by declared question, then as the bundle tested them",
                total_rows=len(shown),
            ),
            TableBlock(
                section="contrasts",
                name="comparison_statistics",
                title="Statistics",
                columns=[
                    TableColumn(key="question", header="Question"),
                    TableColumn(key="reading", header="Reading"),
                    TableColumn(key="contrast", header="Arm"),
                    TableColumn(key="control_mean", header="Control mean"),
                    TableColumn(key="arm_mean", header="Arm mean"),
                    TableColumn(key="hedges_g", header="Hedges' g"),
                    TableColumn(key="pairing_r", header="Pairing r"),
                    TableColumn(key="p_adjusted", header="Adjusted p"),
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
            title=f"Change in {heading}",
            rows=rows,
            unit=units[heading],
            zero_line=True,
            margin=margins[heading],
        )
        for heading, rows in drawn.items()
    )
    blocks.extend(_note(family.disclosure) for family in comparisons.families)
    unranged = sorted(
        {
            reading_name(surface.measure_heading(comparison.name, comparison.reading))
            for family in comparisons.families
            for comparison in family.comparisons
            if comparison.equivalence_untested_reason is not None
        }
    )
    if unranged:
        blocks.append(_note(f"Equivalence untested on {_listed(unranged)}: {EQUIVALENCE_NEEDS_RANGE}."))
    if bundle.run_margins_withheld is not None:
        blocks.append(_note(bundle.run_margins_withheld))
    blocks += [_note(text) for text in bundle.launch_declarations]
    return blocks


def contrast_rows(bundle: AnalysisContextBundle) -> list[dict[str, CellValue]]:
    """Every contrast the bundle tested against the control, one row each, in :func:`contrast_verdicts`' order.

    The page shows only the contrasts that can resolve; a program reads them all here. A contrast with a side under
    the case floor states no change, interval, effect size or p: on so few cases a rate states none.

    Args:
        bundle: The campaign's evidence.

    Returns:
        One row per contrast, keyed ``question`` (in the words it was asked, or ``(campaign-wide)``), ``reading``,
        ``contrast`` and ``control`` (each arm as the page names it), ``control_mean``, ``arm_mean``, ``delta`` and
        ``change`` (the delta with its interval, in words), ``interval``, ``hedges_g``, ``pairing_r``,
        ``p_adjusted``, ``verdict`` (in words) and ``cases``.
    """
    surface = bundle_decision_surface(bundle)
    verdicts = iter(contrast_verdicts(bundle, surface))
    rulers = measure_rulers(surface)
    rows: list[dict[str, CellValue]] = []
    for family in bundle.multiple_comparisons.families:
        for comparison in family.comparisons:
            verdict = next(verdicts)
            ruler = rulers.get(comparison.name, UNITLESS) if comparison.reading == "measure" else UNITLESS
            small = _small_n(comparison)
            interval = _interval_on_delta(comparison.interval, ruler, family.interval_level)
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
                    "control_mean": _figure(comparison.control.mean, ruler).value,
                    "arm_mean": _figure(comparison.contrast.mean, ruler).value,
                    "delta": None if small is not None else _figure(comparison.delta, ruler).value,
                    "change": None
                    if small is not None or comparison.delta is None
                    else with_unit(comparison.delta * ruler.factor, ruler.unit) + (f" {interval}" if interval else ""),
                    "interval": None if small is not None else interval,
                    "hedges_g": None if small is not None else comparison.hedges_g,
                    "pairing_r": comparison.pairing_correlation.r
                    if comparison.pairing_correlation is not None and small is None
                    else None,
                    "p_adjusted": None if small is not None else comparison.p_adjusted,
                    "verdict": verdict.words,
                    "cases": _comparison_cases(comparison),
                }
            )
    return rows


def report_cell_labels(bundle: AnalysisContextBundle) -> dict[str, str]:
    """Each cell's arm as the page names it, by cell reference: its label, with its test setup where it ran under two.

    Args:
        bundle: The campaign's evidence.

    Returns:
        ``cell_ref -> name`` for every cell on the bundle's surface.
    """
    surface = bundle_decision_surface(bundle)
    name = _cell_namer(surface, _arm_names(bundle))
    return {
        cell_ref(cell.variant_key, cell.apparatus_class_id): name(cell.variant_key, cell.apparatus_class_id)
        for cell in surface.cells
    }


def _cell_namer(surface: DecisionSurface, names: dict[str, ArmName]) -> Callable[[str, str], str]:
    """Name a cell by its two coordinates as every table does: the arm's label, its test setup where it ran under two."""
    multi = multi_rig_variants(surface.cells)
    namer = _namer(names)

    def name(variant_key: str, apparatus_class_id: str) -> str:
        rig = f" (test setup {short_digest(apparatus_class_id)})" if variant_key in multi else ""
        return namer(variant_key) + rig

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
            title="Guardrails",
            columns=[
                TableColumn(key="guardrail", header="Guardrail", term="guardrail"),
                TableColumn(key="arm", header="Arm"),
                TableColumn(key="control_mean", header="Control mean"),
                TableColumn(key="arm_mean", header="Arm mean"),
                TableColumn(key="cases", header="Cases"),
                TableColumn(key="delta", header="Change"),
                TableColumn(key="interval", header="Interval", term="interval"),
                TableColumn(key="margin", header="Margin", term="margin"),
                TableColumn(key="decision", header="Decision"),
            ],
            rows=rows,
            order="by guardrail, then arm",
            total_rows=len(rows),
        ),
        _note(_GUARDRAILS_DISCLOSURE, "guardrails"),
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
                title="Bars",
                columns=[
                    TableColumn(key="bar", header="Bar", term="bar"),
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
        _note(
            f"The {bar.source} bar on {humanize('dimension', bar.measure_id)} could be read against no arm: {bar.reason}."
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


def _strata_blocks(surface: DecisionSurface, arms: dict[str, ArmName]) -> list[ReportBlock]:
    """Each arm's figures per stratum beside its pooled figure — or nothing, when no case declares a stratum."""
    control = surface.control_variant_key
    cells = surface_order(
        (cell for cell in surface.cells if cell.strata),
        control=control,
        names={key: name.label for key, name in arms.items()},
    )
    if not cells:
        return []
    cell_name = _cell_namer(surface, arms)
    labels = {
        cell_ref(cell.variant_key, cell.apparatus_class_id): cell_name(cell.variant_key, cell.apparatus_class_id)
        for cell in surface.cells
    }
    names = sorted(
        {stratum.stratum for cell in cells for stratum in cell.strata}, key=lambda name: (name is None, name or "")
    )
    columns = {name: f"stratum_{index}" for index, name in enumerate(names)}
    rows: list[dict[str, CellValue]] = []
    thin: list[str] = []
    for cell in cells:
        arm = labels[cell_ref(cell.variant_key, cell.apparatus_class_id)]
        arm = f"▸ {arm}" if cell.variant_key == control else arm
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
                    f"Too few cases to read alone (a stratum needs at least {STRATUM_MIN_CASES} cases), so read their intervals: "
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


def _label_blocks(surface: DecisionSurface, names: dict[str, ArmName]) -> list[ReportBlock]:
    """A classifier's per-label precision, recall and F1, every arm's, as one table — or nothing for no classifier."""
    control = surface.control_variant_key
    cell_name = _cell_namer(surface, names)
    labels = {
        cell_ref(cell.variant_key, cell.apparatus_class_id): cell_name(cell.variant_key, cell.apparatus_class_id)
        for cell in surface.cells
    }
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
                "arm": f"▸ {labels[ref]}" if cell.variant_key == control else labels[ref],
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
            title="Per label",
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
        _note(" ".join(said)),
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


def _case_namer(bundle: AnalysisContextBundle) -> Callable[[str], str]:
    """Name a case by its id as every table does: ``case N``, never its first input."""
    numbers = _case_numbers(bundle)
    return lambda case: f"case {numbers[case]}" if case in numbers else humanize("case", case) or "a case"


def _case_blocks(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> list[ReportBlock]:
    """One row per case: each arm's trials as marks with their causes, regressions first, each linked to its trials."""
    if not bundle.trials:
        return []
    control = _control_key(bundle)
    arms = [board.variant_key for board in bundle.scoreboard]
    numbers = _case_numbers(bundle)
    by_case: dict[str, list[TrialRow]] = {}
    for trial in bundle.trials:
        by_case.setdefault(trial.case_id, []).append(trial)
    strata = any(trial.stratum for trial in bundle.trials)

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
            "flag": {0: "regression", 1: "no usable trial", 2: None}[group(case)],
        }
        if strata:
            row["stratum"] = trials[0].stratum
        for index, key in enumerate(arms):
            row[f"arm_{index}"] = _marks([trial for trial in trials if trial.arm == key])
        row["trials"] = " ".join(trial.trial_id for trial in trials)
        rows.append(row)
    blocks: list[ReportBlock] = [
        TableBlock(
            section="cases",
            name="cases",
            title="Cases",
            columns=[
                TableColumn(key="case", header="Case", term="case"),
                TableColumn(key="flag", header="Flag"),
                *([TableColumn(key="stratum", header="Stratum")] if strata else []),
                *(
                    TableColumn(
                        key=f"arm_{index}", header=f"▸ {arm(key)}" if key == control else arm(key), term="marks"
                    )
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
                    f"{bundle.trials_omitted} more trials count on the scoreboard and are not listed here, the passed "
                    "ones first."
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


def _trial_why(trial: TrialRow) -> str | None:
    """Which check or judged dimension failed a trial, or why it was lost, on one line; None for a passed trial."""
    if trial.state == "passed":
        return None
    if trial.state == "lost":
        return f"lost: {cause_short(trial.cause or 'unknown')}"
    if trial.cause == "check" and trial.failed_checks:
        return "check: " + ", ".join(humanize("dimension", check.name) for check in trial.failed_checks)
    if trial.cause == "judged":
        under = [entry.dim for entry in trial.judge if entry.scale == "pass_fail" and entry.score == 0]
        return "judge: " + (", ".join(humanize("dimension", dim) for dim in under) or "scored below the bar")
    return cause_short(trial.cause or "unknown")


def _trial_blocks(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> list[ReportBlock]:
    """One row per trial, then one closed fold per trial: what it asked, what the judge saw and said, and why."""
    if not bundle.trials:
        return []
    blocks: list[ReportBlock] = []
    arms = len(bundle.scoreboard)
    exclusions = arm_exclusions(bundle)
    case_name = _case_namer(bundle)
    rows: list[dict[str, CellValue]] = []
    for trial in bundle.trials:
        words: CauseWords | None = None
        if trial.state == "lost":
            words = loss_words(trial.cause or "unknown", exclusions=exclusions.get(trial.arm, ()), arms=arms)
        elif trial.state == "failed":
            words = fail_words(trial.cause or "unknown")
        why = _trial_why(trial)
        rows.append(
            {
                "trial": trial.trial_id,
                "case": case_name(trial.case_id),
                "arm": arm(trial.arm),
                "outcome": trial.state,
                "why": why,
            }
        )
        blocks.append(
            TrialBlock(
                section="trials",
                anchor=trial.trial_id,
                trial_id=trial.trial_id,
                case_label=case_name(trial.case_id),
                arm=arm(trial.arm),
                repeat=trial.repeat,
                state=trial.state,
                cause_words=None if words is None else f"{words.words}. {words.remedy[:1].upper()}{words.remedy[1:]}.",
                why=why,
                first_input=trial.first_input,
                excerpt=trial.excerpt,
                failed_checks=list(trial.failed_checks),
                judge=list(trial.judge),
                error_words=trial.error_words,
                result_id=trial.result_id,
            )
        )
    table = TableBlock(
        section="trials",
        name="trials",
        title="Trials",
        columns=[
            TableColumn(key="trial", header="Trial", term="trial"),
            TableColumn(key="case", header="Case"),
            TableColumn(key="arm", header="Arm"),
            TableColumn(key="outcome", header="Outcome"),
            TableColumn(key="why", header="Why"),
        ],
        rows=[table_row(row) for row in rows],
        order="as the bundle numbers them",
        total_rows=len(rows),
    )
    return [table, *blocks]


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
    case_name = _case_namer(bundle)
    for board in bundle.scoreboard:
        unusable = sorted(
            {
                trial.case_id
                for trial in bundle.trials
                if trial.arm == board.variant_key and (trial.arm, trial.case_id) not in usable
            },
            key=case_name,
        )
        if unusable:
            named = _listed([case_name(case) for case in unusable])
            said.append(f"{arm(board.variant_key)} has no usable trial of {named}.")
    for proof in bundle.goal_check_proofs:
        if proof.proof != "proven":
            said.append(_unproven_check_sentence(proof))
    if (no_margin := _no_margin_sentence(bundle)) is not None:
        said.append(no_margin)
    blocks: list[ReportBlock] = [DisclosureBlock(section="not_tested", source="not_tested", text=text) for text in said]
    blocks += [
        DisclosureBlock(section="not_tested", source="calibration", text=text, term="calibrated")
        for text in _calibration_sentences(bundle.judge_evidence_tiers)
    ]
    quoted = [bundle.held_fixed_reading.disclosure] if bundle.held_fixed_reading.disclosure else []
    if bundle.cost_unmeasured:
        quoted.append(bundle.cost_unmeasured + _some_arms(bundle, bundle.cost_unmeasured_cells, "Unmeasured", arm))
    blocks += [
        DisclosureBlock(section="not_tested", source="not_tested", text=text, fold="Disclosures") for text in quoted
    ]
    if not blocks:
        return [DisclosureBlock(section="not_tested", source="not_tested", text="Nothing declared went untested.")]
    return blocks


def _calibration_sentences(tiers: Sequence[JudgeEvidenceTier]) -> list[str]:
    """Each judge not yet calibrated against people, as a state and its next step — never as a failure.

    A new judge has no labels, so on a first campaign this is said on every report: where the judge stands and what
    to do next, one line per judge and state, every dimension it holds for named once.
    """
    by_judge: dict[tuple[str, bool, int], list[str]] = {}
    for tier in tiers:
        criterion = tier.calibration
        if tier.tier == "calibrated" or criterion.state in ("met", "not_met"):
            continue
        key = (
            tier.judge_model or "an unnamed judge",
            criterion.n == 0,
            criterion.results_needed or criterion.min_results,
        )
        dims = by_judge.setdefault(key, [])
        if (dim := humanize("dimension", tier.rubric_dim)) not in dims:
            dims.append(dim)
    said = []
    for (judge, none_yet, more), dims in by_judge.items():
        state = "is uncalibrated" if none_yet else "is not yet calibrated"
        why = "no human labels yet" if none_yet else "too few human labels so far"
        said.append(
            f"Judge {judge} {state} on {_listed(dims)}: {why}, so its scores are directional. "
            f"Rate about {more} of its outputs to calibrate it."
        )
    return said


def _unproven_check_sentence(proof: GoalCheckProofReading) -> str:
    """A goal check not shown to beat doing nothing: what it is, and what its pass rate is not."""
    check = humanize("dimension", proof.check)
    if proof.refused is not None:
        return (
            f"Goal check {check} is {CHECK_REFUSED_UNDER_CURRENT_GRAMMAR} ({proof.refused}), so no arm was graded on "
            "it. Any pass rate it has predates the rule."
        )
    why = "its control shows it cannot" if proof.proof == "refuted" else "no control shows it can"
    notes = []
    if proof.unrecorded:
        notes.append(f"{proof.unrecorded} of {proof.runs} runs recorded no proof")
    if proof.stale:
        notes.append(f"{proof.stale} of {proof.runs} runs proved it under an earlier rule; relaunch to re-prove")
    tail = f" ({'; '.join(notes)})" if notes else ""
    return (
        f"Goal check {check} is {proof.proof}: {why} tell acting from doing nothing{tail}. "
        "A candidate that did nothing might pass it, so it is not shown to measure the behaviour."
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
        return ", ".join(catalog[name].reader_name or humanize("dimension", name) for name in group)

    advice = []
    if scorers:
        advice.append(
            f"Declare a margin on {heading(scorers)} where it is graded (`grader({scorers[0]}, margin=...)`)"
            + (
                "; a grader returning a number takes its range beside it (`range=`)."
                if any(catalog[name].family == "mechanical" and catalog[name].value_range is None for name in scorers)
                else "."
            )
        )
    if rates:
        advice.append(
            f"Declare a margin on {heading(rates)} on the classifier's field (`grader('<field>', margin=...)`), on "
            "every arm's run."
        )
    if fixed:
        advice.append(
            f"No comparison can declare one on {heading(fixed)}: the engine owns "
            f"{'its' if len(fixed) == 1 else 'their'} description."
        )
    them = "it" if len(names) == 1 else "them"
    return (
        f"No margin is declared on {heading(names)}, so no contrast on {them} can read equivalent. Not separated "
        "never means the arms are alike. " + " ".join(advice)
    )


def _method_blocks(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    arm: Callable[[str], str],
    names: dict[str, ArmName],
    *,
    source: ReportSource,
    writer: str | None,
    notes: Sequence[ReportBlock] = (),
    style: MemoStyle | None = None,
) -> list[ReportBlock]:
    """How everything on the page was measured and made: a table, then the engine's disclosures in one closed fold."""
    blocks: list[ReportBlock] = [_methods_table(bundle, names, source=source, writer=writer, style=style)]

    def say(source_of: DisclosureSource, text: str | None) -> None:
        if text and text.strip():
            blocks.append(DisclosureBlock(section="methods", source=source_of, text=text.strip(), fold="Disclosures"))

    for tier in bundle.judge_evidence_tiers:
        say("measurement", "Judge: " + tier_sentence(tier))
    say("measurement", bundle.request_settings_disclosure)
    for variant_key, cells in arm_exclusions(bundle).items():
        if truncated := sum(cell.by_cause.get("truncated", 0) for cell in cells):
            say(
                "measurement",
                f"{arm(variant_key)}: {truncated} of {sum(cell.n_trials for cell in cells)} trials lost because the "
                f"judge's reply was cut off at its output cap ({loss_words('truncated', exclusions=cells).words}).",
            )
    for flagged in bundle.differential_exclusions:
        others = ", ".join(
            f"{format_number(100 * share)}% of {arm(key)}'s" for key, share in sorted(flagged.other_arms.items())
        )
        say(
            "measurement",
            f"Differential loss: {format_number(100 * flagged.share)}% of {arm(flagged.variant_key)}'s trials were "
            f"lost to {cause_short(flagged.cause)}"
            + (f", against {others}" if others else "")
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
    blocks.extend(notes)
    return blocks


def _methods_table(
    bundle: AnalysisContextBundle,
    names: dict[str, ArmName],
    *,
    source: ReportSource,
    writer: str | None,
    style: MemoStyle | None = None,
) -> TableBlock:
    """Models, judge, settings sent, cost, the analysis's writer and the bundle, one row each."""
    rows: list[tuple[str, str]] = []
    control = _control_key(bundle)
    keys = [board.variant_key for board in bundle.scoreboard]
    for key in ([control] if control is not None else []) + [key for key in keys if key != control]:
        rows.append(("Arm", names[key].first() if key in names else _namer(names)(key)))
    for level in bundle.judge_change.levels:
        rows.append(("Judge", level.judge_model))
        rows.append(
            (
                "Judge temperature",
                "not sent" if level.judge_temperature is None else format_number(level.judge_temperature),
            )
        )
        if level.judge_sent_settings:
            rows.append(("Judge settings sent", "; ".join(_settings_words(sent) for sent in level.judge_sent_settings)))
    costs = [
        f"{_namer(names)(board.variant_key)} {_dollars(board.test_cost_usd)}"
        for board in bundle.scoreboard
        if board.test_cost_usd is not None
    ]
    if costs:
        rows.append(("Test cost", "; ".join(costs)))
    rows.append(("Dataset", ", ".join(ref.label for ref in bundle.case_sets) or "each run's own cases"))
    rows.append(
        (
            "Design",
            ("question declared before the run" if bundle.question_declared else "exploratory")
            + (f", frozen {_date(bundle.design_frozen_at)}" if bundle.design_frozen_at else ", not frozen"),
        )
    )
    rows.append(("Analysis", writer or NO_ANALYSIS))
    if style is not None:
        offending = style.offending()
        rows.append(("Memo style", f"style: {len(offending)} violations" if offending else "no violations left"))
    rows.append(("Date", _date(source.generated_at)))
    if unshown := _uncarried(bundle):
        rows.append(("Readings not shown", f"{len(unshown)} carried by no trial: {_listed(unshown)}"))
    rows.append(
        (
            "Bundle",
            f"{source.bundle_fingerprint}, shape {bundle.schema_version}, 3tears-evals "
            f"{bundle.package_version or 'version not recorded'}, {len(bundle.run_ids)} runs; bundle.json beside this "
            "page",
        )
    )
    return TableBlock(
        section="methods",
        name="methods",
        title="Methods",
        columns=[TableColumn(key="item", header="Item"), TableColumn(key="value", header="Value")],
        rows=[{"item": TableCell(value=item), "value": TableCell(value=value)} for item, value in rows],
        order="the arms, the judge, the cost, then how the page was made",
        total_rows=len(rows),
    )


def _settings_words(sent: ClientRequestSettings) -> str:
    """Request settings as words, never as ``key=value``."""
    parts = [f"max tokens {sent.max_tokens}"]
    if sent.reasoning_max_tokens is not None:
        parts.append(f"reasoning max tokens {sent.reasoning_max_tokens}")
    if sent.reasoning_effort is not None:
        parts.append(f"reasoning effort {sent.reasoning_effort}")
    if sent.strict_output:
        parts.append("strict output")
    return ", ".join(parts)


def _uncarried(bundle: AnalysisContextBundle) -> list[str]:
    """Each reading the campaign names that no arm carries on any trial, or that one side of a contrast never carried."""
    carried = {summary.name for cell in bundle.cell_measures for summary in cell.measures.measures if summary.n}
    names = [name for name in bundle.measure_catalog if name not in carried]
    names += [
        comparison.name
        for family in bundle.multiple_comparisons.families
        for comparison in family.comparisons
        if min(comparison.control.n_cases, comparison.contrast.n_cases) == 0
    ]
    catalog = bundle.measure_catalog
    return list(
        dict.fromkeys(
            (catalog[name].reader_name if name in catalog else None) or humanize("dimension", name) for name in names
        )
    )


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
    "contrast_rows",
    "report_cell_labels",
    "analysis_bundle",
    "analysis_verdict",
    "NO_STRATUM",
    "TOO_FEW_CASES",
    "bar_verdicts",
    "build_code_only_report",
    "build_report",
    "contrast_verdicts",
    "guardrail_verdicts",
]

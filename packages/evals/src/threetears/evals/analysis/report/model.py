"""The ``Report`` — the one page every door into a campaign's report renders, and the evidence it was computed from.

**One page, result first.** A report is an ordered list of blocks in its sections: the first screen (the title — the
question in plain words —, the arms line, the verdict line, the scoreboard and the caveats under it, in normal
weight), the decision rule clause by clause where one is declared, what it means (the memo's findings, each a table
code built and one sentence, then the contrasts against the control), the cases, the trials (one row each, every
record folded), what was not tested, the methods (a table, then the engine's disclosures folded), and the glossary
of every term of art the page linked. Every count, rate, interval and cost in it is code's, computed from the
evidence bundle the report carries (:attr:`Report.bundle`). Every string it prints is held to
:mod:`~threetears.evals.analysis.style`.

**Who wrote what is part of the shape.** A ``text`` block holds what the analysis's author wrote, and nothing else;
what code says — the verdict, the basis, a table's caption, a reason something is not drawn — is a ``disclosure``.
A code-only report (no analysis was generated) holds no ``text`` block at all, and its model refuses one, so a
reader never takes code's computation for an analyst's judgement, or the reverse.

**Two charts, drawn by the package.** A ``chart`` block is an outcome bar (one stacked bar per arm: passed, failed and
lost by cause, every segment's count written beside it) or interval rows (a dot and its interval per row, with the
value, the interval and the cases written as text). Both carry their values, so Markdown prints them as a table and
HTML draws them as inline SVG (:mod:`~threetears.evals.analysis.report.svg`) beside the same table. A chart row is
drawn only for an arm with :data:`~threetears.evals.analysis.report.verdict.MIN_CASES` usable cases and an
interval, and a chart with no such row is not built.

**The bundle is the machine form.** There is no published JSON of the report: a program reads ``bundle.json``,
which :meth:`Report.write` puts beside the page, and the page names its fingerprint.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, Self, get_args

from pydantic import Field, model_validator

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, TrialCheck, TrialJudgement, TrialState
from threetears.evals.analysis.quantities import number_in, render_cell, unit_label, with_unit
from threetears.evals.analysis.style import GLOSSARY
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.prose import ModelProse

#: A table cell's value: a number, words, a boolean, or None where there is none.
CellValue = str | int | float | bool | None

#: What a report is of: a generated analysis, or the campaign's evidence alone with no analysis.
ReportBasis = Literal["analysis", "code_only"]

#: Where a block sits, in reading order. ``header`` is the first screen's title, arms and verdict lines, and
#: ``scoreboard`` the rest of the first screen; ``decision`` is a declared decision rule read clause by clause;
#: ``memo`` and ``contrasts`` are what it means; ``cases`` and ``trials`` the cases and their drill-down;
#: ``not_tested`` what the campaign could not speak to; ``methods`` how everything was measured; ``glossary`` every
#: term of art the page used.
ReportSection = Literal[
    "header", "scoreboard", "decision", "memo", "contrasts", "cases", "trials", "not_tested", "methods", "glossary"
]

#: Each section's heading, in reading order — the order every serializer lays the sections out in, each three words
#: or fewer. The first screen (``header`` and ``scoreboard``) carries no heading: it is the page's own head.
SECTION_TITLES: dict[str, str] = {
    "header": "",
    "scoreboard": "Scoreboard",
    "decision": "Decision",
    "memo": "What it means",
    "contrasts": "Contrasts",
    "cases": "Cases",
    "trials": "Trials",
    "not_tested": "Not tested",
    "methods": "Methods",
    "glossary": "Glossary",
}

#: The sections the first screen is made of.
FIRST_SCREEN: frozenset[str] = frozenset({"header", "scoreboard"})

#: What a text block's author wrote it as.
TextRole = Literal[
    "answer",
    "recommendation",
    "revisit_when",
    "finding_title",
    "finding_body",
    "caveat",
    "carried_forward",
    "next_step",
    "next_step_why",
]

#: Who a disclosure speaks for. ``arms``: the first screen's second line, each arm named in full once, the control
#: marked. ``caveat``: a line under the scoreboard qualifying the result, in normal weight. ``basis``: whether a question was declared before the run (line 2). ``verdict``: the
#: verdict line. ``design``, ``charts`` and ``cost``: the lines under the scoreboard. ``memo``: who wrote the memo,
#: or that no one did. ``runs``: member runs left out, unfinished or short. ``measurement``: how and when the runs
#: were launched and measured. ``apparatus``: the test setup — controls, confounds, cells that did not pool, a judge
#: grading its own model's answers. ``comparisons``: how the contrasts against the control were tested and
#: corrected. ``strata``: a stratum too small to read alone. ``guardrails``: how guardrails were decided.
#: ``scope``: which readings no declared question asked about. ``not_tested``: something the campaign could not
#: speak to. ``calibration``: a judge not yet calibrated against people's labels — a state with its next step, never a
#: failure. ``generation``: how the report itself was made.
DisclosureSource = Literal[
    "arms",
    "caveat",
    "basis",
    "verdict",
    "design",
    "charts",
    "cost",
    "memo",
    "runs",
    "measurement",
    "apparatus",
    "comparisons",
    "strata",
    "guardrails",
    "scope",
    "not_tested",
    "calibration",
    "generation",
]

#: Which reading a :class:`Verdict` is: a contrast tested against the control, a bar read on one cell, or a guardrail
#: checked for one arm against the control.
VerdictKind = Literal["contrast", "bar", "guardrail"]

#: What a contrast's test came to (``FamilyComparison.verdict``).
ContrastOutcome = Literal["improved", "regressed", "equivalent", "not_separated", "untested"]

#: What a bar's reading on one cell came to (``BarVerdict.decision``).
BarOutcome = Literal["cleared", "missed", "undecided", "no_interval", "no_data"]

#: What a guardrail came to for one arm against the control (``GuardrailCheck.decision``).
GuardrailOutcome = Literal["held", "breached", "undecided"]

#: Each kind's outcomes, so a verdict cannot carry another kind's.
VERDICT_OUTCOMES: dict[str, frozenset[str]] = {
    "contrast": frozenset(get_args(ContrastOutcome)),
    "bar": frozenset(get_args(BarOutcome)),
    "guardrail": frozenset(get_args(GuardrailOutcome)),
}

#: Why a verdict came out as it did, as a code a program branches on (``reason_detail`` says it in words where
#: there is more to say). A contrast: ``separated`` (its Holm-adjusted p is below α and its interval excludes 0),
#: ``inside_margin`` (the equivalence test shows the difference inside the margin), ``no_margin`` (not separated,
#: and with no margin it could not be shown equivalent), ``margin_untested`` (not separated, a margin is declared
#: and no equivalence test could run), ``not_inside_margin`` (not separated, and the equivalence test did not show
#: it inside the margin), ``untestable`` (no test could run), ``too_few_cases`` (an arm had fewer cases than a rate
#: is read on, so the report states no difference). A bar: ``interval_clears``, ``interval_misses``,
#: ``interval_straddles``, ``too_few_observations``, ``no_observations``. A guardrail: ``within_margin`` (held),
#: ``beyond_margin`` (breached), ``interval_straddles`` or ``no_interval`` (undecided).
VerdictReason = Literal[
    "separated",
    "inside_margin",
    "no_margin",
    "margin_untested",
    "not_inside_margin",
    "untestable",
    "too_few_cases",
    "interval_clears",
    "interval_misses",
    "interval_straddles",
    "too_few_observations",
    "no_observations",
    "within_margin",
    "beyond_margin",
    "no_interval",
]


class Verdict(EvalDocumentModel):
    """One verdict the report states, typed: what a program reads instead of the words printed from it.

    The source of truth for every verdict the report prints: the contrasts table's ``verdict`` cell and the
    guardrails table's ``decision`` cell are each a verdict's :attr:`words`, rendered from it, never the reverse.
    """

    kind: VerdictKind = Field(description="A contrast against the control, a bar on one cell, or a guardrail.")
    outcome: ContrastOutcome | BarOutcome | GuardrailOutcome = Field(
        description=(
            "What it came to. A contrast: improved, regressed, equivalent, not_separated (says nothing about whether "
            "the arms differ) or untested. A bar: cleared, missed, undecided, no_interval or no_data. A guardrail: "
            "held, breached or undecided. `undecided`, `no_interval`, `no_data`, `not_separated` and `untested` "
            "are never a pass."
        )
    )
    reason: VerdictReason = Field(description="Why, as a code; see `VerdictReason`.")
    reason_detail: str | None = Field(
        default=None, description="The reason in words where there is more to say: why no test ran, why undecided."
    )
    reading: Literal["measure", "judged", "trial"] = Field(
        description="Whether `name` is a measure, a judged dimension, or `trial`: the overall pass rate (`passed`)."
    )
    name: str = Field(min_length=1, description="The measure or judged dimension, by its key.")
    heading: str = Field(min_length=1, description="The reading as the report heads it.")
    arm: str = Field(min_length=1, description="The arm, as the report names it.")
    variant_key: str = Field(min_length=1, description="The arm's variant.")
    apparatus_class_id: str = Field(min_length=1, description="The rig it was measured under.")
    control: str | None = Field(
        default=None, description="The control, as the report names it; None for a bar, read on one cell."
    )
    question_id: str | None = Field(
        default=None,
        description="The declared question whose family tested a contrast; None campaign-wide, and for a bar or guardrail.",
    )
    delta: float | None = Field(default=None, description="Arm mean minus control mean; None for a bar.")
    interval: tuple[float, float] | None = Field(
        default=None,
        description="The interval the verdict read: on the delta for a contrast or guardrail, on the cell's value for a bar.",
    )
    interval_level: float | None = Field(default=None, description="The interval's level, 0 to 1.")
    p_adjusted: float | None = Field(default=None, description="A contrast's Holm-adjusted p; None otherwise.")
    value: float | None = Field(default=None, description="A bar's cell value; None otherwise.")
    threshold: float | None = Field(default=None, description="A bar's threshold; None otherwise.")
    margin: float | None = Field(
        default=None,
        description=(
            "The margin the verdict read, in the reading's units: a contrast's equivalence margin, a bar's margin, a "
            "guardrail's tolerated worsening (0 when none is declared). None when the reading has none."
        ),
    )
    margin_source: Literal["measure", "run", "campaign"] | None = Field(
        default=None,
        description=(
            "Where the margin came from: `measure` = the measure's declared materiality threshold; `run` = a margin "
            "every run declared at launch on a core rate measure; `campaign` = a judged guardrail's margin its "
            "campaign declares (`guardrail_margins`). None when no margin was declared."
        ),
    )
    materiality: Literal["material", "immaterial"] | None = Field(
        default=None,
        description="A contrast's delta read against its margin: `immaterial` when inside it. None otherwise.",
    )
    guardrail: bool = Field(
        default=False, description="Whether the reading is a guardrail — true exactly for a guardrail verdict."
    )
    words: str = Field(min_length=1, description="The verdict as the report prints it, rendered from these fields.")

    @model_validator(mode="after")
    def _the_outcome_is_its_kinds(self) -> Self:
        """Refuse an outcome of another kind, and a guardrail flag that disagrees with the kind.

        Raises:
            ValueError: The outcome is not one of the kind's, or ``guardrail`` is not ``kind == "guardrail"``.
        """
        if self.outcome not in VERDICT_OUTCOMES[self.kind]:
            raise ValueError(f"a {self.kind} verdict cannot be {self.outcome!r}")
        if self.guardrail != (self.kind == "guardrail"):
            raise ValueError("a verdict is flagged guardrail exactly when it is a guardrail's")
        return self


class Fact(EvalDocumentModel):
    """A labelled fact code states beside an author's words — a confidence, a disposition, a tier."""

    name: str = Field(min_length=1, description="What the fact is, e.g. `Confidence`.")
    value: str = Field(min_length=1, description="The fact, in words.")


class _Block(EvalDocumentModel):
    """What every block carries: where it sits, and which findings it belongs to or rests on."""

    section: ReportSection = Field(description="The section the block sits in.")
    finding: int | None = Field(
        default=None,
        ge=0,
        description="The finding this block belongs to, by its position (0 is the first); None when it belongs to none.",
    )
    rests_on: list[int] = Field(
        default_factory=list, description="Positions of the findings this block rests on, as the author linked them."
    )
    anchor: str | None = Field(
        default=None, description="The id a link to this block names (`F1`, `case-3`, `T7`); None when none does."
    )
    fold: str | None = Field(
        default=None,
        description=(
            "The closed fold the block sits in (`Evidence`, `Disclosures`); consecutive blocks naming one fold share "
            "it. None for a block shown open."
        ),
    )

    @property
    def finding_number(self) -> int | None:
        """``finding`` counted from one, as a reader counts findings; None when the block names none."""
        return None if self.finding is None else self.finding + 1

    @property
    def rests_on_numbers(self) -> list[int]:
        """``rests_on`` counted from one, as a reader counts findings."""
        return [position + 1 for position in self.rests_on]


#: The text roles that are part of one finding, and so name it.
FINDING_ROLES: frozenset[str] = frozenset({"finding_title", "finding_body", "caveat", "carried_forward"})


class TextBlock(_Block):
    """What the analysis's author wrote, exactly as written, with the facts code states beside it."""

    kind: Literal["text"] = Field(default="text", description="Which kind of block this is.")
    role: TextRole = Field(description="What the author wrote it as.")
    body: ModelProse = Field(
        description="The author's words, Markdown allowed; empty where the author left a required field blank."
    )
    facts: list[Fact] = Field(default_factory=list, description="Facts code states beside the words, in reading order.")

    @model_validator(mode="after")
    def _a_findings_own_words_name_it(self) -> Self:
        """Refuse a finding's title, body, caveat or carried-forward claim that names no finding.

        Raises:
            ValueError: A block in one of those roles has no ``finding``.
        """
        if self.role in FINDING_ROLES and self.finding is None:
            raise ValueError(f"a {self.role} block belongs to a finding and names none")
        return self


class TableColumn(EvalDocumentModel):
    """One column of a report table."""

    key: str = Field(min_length=1, description="The key in each row.")
    header: str = Field(min_length=1, description="The column's header, carrying its unit where it has one.")
    term: str | None = Field(
        default=None, description="The glossary term the header links to, where it is a term of art; None for none."
    )

    @model_validator(mode="after")
    def _the_term_is_defined(self) -> Self:
        """Refuse a link to a term the glossary does not define.

        Raises:
            ValueError: ``term`` is not a :data:`~threetears.evals.analysis.style.GLOSSARY` key.
        """
        if self.term is not None and self.term not in GLOSSARY:
            raise ValueError(f"column {self.key!r} links the term {self.term!r}, which the glossary does not define")
        return self


class TableCell(EvalDocumentModel):
    """One cell of a report table: its value, and the text a reader is shown for it.

    The value is what a program sorts by; the text is what every serializer prints, spelled by the one number rule
    (:func:`~threetears.evals.analysis.quantities.render_cell`: an absent value is a dash, never a zero; a
    boolean is yes or no).
    """

    value: CellValue = Field(description="The cell's value: a number, words, a boolean, or null where there is none.")
    unit: str = Field(
        default="",
        description=(
            "The unit a number in this cell is stated in, already restated onto its measure's one ruler for the "
            "report. Empty for a unitless number, for words, and wherever the column's header names the unit."
        ),
    )

    @model_validator(mode="after")
    def _a_unit_qualifies_a_number(self) -> Self:
        """Refuse a unit on anything but a number.

        Raises:
            ValueError: ``unit`` is set and ``value`` is not a number.
        """
        if self.unit and (isinstance(self.value, bool) or not isinstance(self.value, int | float)):
            raise ValueError(f"a cell's unit qualifies a number, and this cell holds {self.value!r}")
        return self

    @property
    def text(self) -> str:
        """``value`` as every serializer prints it."""
        if self.unit and isinstance(self.value, int | float) and not isinstance(self.value, bool):
            return with_unit(self.value, self.unit)
        return render_cell(self.value)


def table_row(values: Mapping[str, CellValue]) -> dict[str, TableCell]:
    """A table row from its values, each made a :class:`TableCell`.

    Args:
        values: The row's values, by column key.

    Returns:
        The row's cells, by the same keys, in the same order.
    """
    return {key: TableCell(value=value) for key, value in values.items()}


class TableBlock(_Block):
    """A table code laid out — its columns, its rows in their stated order, and how much of it is shown."""

    kind: Literal["table"] = Field(default="table", description="Which kind of block this is.")
    name: str = Field(min_length=1, description="Which table this is, e.g. `scoreboard`, `comparisons`, `cases`.")
    title: str = Field(min_length=1, description="The table's heading.")
    columns: list[TableColumn] = Field(min_length=1, description="The columns, in display order.")
    rows: list[dict[str, TableCell]] = Field(description="The rows shown, in the stated order, keyed by every column.")
    order: str = Field(min_length=1, description="The order the rows are in, in words.")
    total_rows: int = Field(ge=0, description="How many rows the table has; more than are shown when it is truncated.")
    details: bool = Field(
        default=False, description="Whether the table is drill-down: HTML folds it in a `<details>`, closed."
    )

    @model_validator(mode="after")
    def _rows_fit_the_table(self) -> Self:
        """Refuse a table showing more rows than it has, or a row keying other columns than the table's.

        Raises:
            ValueError: ``total_rows`` is below the rows shown, or a row's keys are not exactly the columns'.
        """
        if self.total_rows < len(self.rows):
            raise ValueError(f"table {self.name!r} shows {len(self.rows)} rows but says it has {self.total_rows}")
        keys = {column.key for column in self.columns}
        if bad := sorted({key for row in self.rows for key in row.keys() ^ keys}):
            raise ValueError(f"table {self.name!r} has rows that do not key exactly its columns: {', '.join(bad)}")
        return self

    def values(self) -> list[dict[str, CellValue]]:
        """The rows shown as their values alone — what a program reads a table for.

        Returns:
            One mapping per row, in the stated order, each column key to its cell's value.
        """
        return [{key: cell.value for key, cell in row.items()} for row in self.rows]


#: The two charts a report draws.
ChartKind = Literal["outcome_bar", "interval_rows"]


class OutcomeSegment(EvalDocumentModel):
    """One segment of an arm's outcome bar: passed, failed, or lost to one cause, with its count."""

    state: TrialState = Field(description="passed, failed or lost.")
    label: str = Field(min_length=1, description="The segment in words (`passed`, `lost: judge truncated`).")
    count: int = Field(ge=1, description="Trials in it.")


class OutcomeBar(EvalDocumentModel):
    """One arm's census as a stacked bar: every trial it ran, by how it ended."""

    arm: str = Field(min_length=1, description="The arm, as the report names it.")
    segments: list[OutcomeSegment] = Field(description="Passed, then failed, then lost, each cause its own segment.")

    @property
    def total(self) -> int:
        """Every trial the bar stands for."""
        return sum(segment.count for segment in self.segments)


class IntervalRow(EvalDocumentModel):
    """One row of an interval chart: a value, its interval and the cases it rests on."""

    label: str = Field(min_length=1, description="The row: an arm, or an arm against the control.")
    value: float = Field(description="The value, on the chart's ruler.")
    low: float = Field(description="The interval's low end, on the chart's ruler.")
    high: float = Field(description="The interval's high end, on the chart's ruler.")
    n_cases: int = Field(ge=1, description="The cases the value rests on.")
    verdict: str | None = Field(default=None, description="The row's verdict in words, where it has one.")

    @model_validator(mode="after")
    def _the_interval_holds_the_value(self) -> Self:
        """Refuse an interval that runs backwards.

        Raises:
            ValueError: ``low`` is above ``high``.
        """
        if self.low > self.high:
            raise ValueError(f"row {self.label!r}: its interval runs from {self.low} down to {self.high}")
        return self


class ChartBlock(_Block):
    """One of the two charts a report draws, with every value it draws, so it prints as a table too."""

    kind: Literal["chart"] = Field(default="chart", description="Which kind of block this is.")
    chart: ChartKind = Field(description="`outcome_bar` or `interval_rows`.")
    title: str = Field(min_length=1, description="What the chart shows.")
    bars: list[OutcomeBar] = Field(default_factory=list, description="An outcome bar's arms, one bar each.")
    rows: list[IntervalRow] = Field(default_factory=list, description="An interval chart's rows, in drawn order.")
    unit: str = Field(default="", description="The unit an interval chart's values are stated in; empty for none.")
    zero_line: bool = Field(default=False, description="Whether the chart draws a line at zero (a difference's).")
    margin: float | None = Field(
        default=None, description="A margin band either side of zero, on the chart's ruler; None when none declared."
    )

    @model_validator(mode="after")
    def _draws_what_its_kind_draws(self) -> Self:
        """Refuse a chart with nothing to draw, or one carrying the other kind's marks.

        Raises:
            ValueError: An outcome bar with rows or no bar, or an interval chart with bars or no row.
        """
        if self.chart == "outcome_bar" and (self.rows or not self.bars):
            raise ValueError("an outcome bar draws one bar per arm and no interval row")
        if self.chart == "interval_rows" and (self.bars or not self.rows):
            raise ValueError("an interval chart draws at least one row and no outcome bar")
        return self

    def values_table(self) -> tuple[list[str], list[list[str]]]:
        """The chart's values as a table — what Markdown prints and what HTML keeps beside the picture.

        Returns:
            ``(headers, rows)``, each row's cells as text.
        """
        if self.chart == "outcome_bar":
            labels = list(dict.fromkeys(segment.label for bar in self.bars for segment in bar.segments))
            headers = ["Arm", "ran", *labels]
            body = []
            for bar in self.bars:
                counts = {segment.label: segment.count for segment in bar.segments}
                body.append([bar.arm, str(bar.total), *(str(counts.get(label, 0)) for label in labels)])
            return headers, body
        unit = f" ({unit_label(self.unit)})" if self.unit else ""
        headers = [
            "Row",
            f"Value{unit}",
            f"Interval{unit}",
            "Cases",
            *(["Verdict"] if any(row.verdict for row in self.rows) else []),
        ]
        body = []
        for row in self.rows:
            cells = [
                row.label,
                interval_value_text(row.value, self.unit),
                interval_text(row.low, row.high, self.unit),
                str(row.n_cases),
            ]
            if len(headers) == 5:
                cells.append(row.verdict or "—")
            body.append(cells)
        return headers, body


def interval_value_text(value: float, unit: str = "") -> str:
    """A charted value as text, by the one number rule, at the precision its unit is read to; the unit itself is
    the header's."""
    return number_in(value, unit)


def interval_text(low: float, high: float, unit: str = "") -> str:
    """A charted interval as text: ``[low, high]``, each end at the precision its unit is read to."""
    return f"[{number_in(low, unit)}, {number_in(high, unit)}]"


class TrialBlock(_Block):
    """One trial to drill into: what the case asked, what the judge saw and said, and why it ended as it did."""

    kind: Literal["trial"] = Field(default="trial", description="Which kind of block this is.")
    trial_id: str = Field(min_length=1, description="The trial, as the report numbers it.")
    case_label: str = Field(
        min_length=1, description="The case, as a reader names it: its label, or `case N` where the label is a dump."
    )
    arm: str = Field(min_length=1, description="The arm, as the report names it.")
    repeat: int = Field(ge=1, description="Which repeat of the case.")
    state: TrialState = Field(description="passed, failed or lost.")
    cause_words: str | None = Field(default=None, description="Why it failed or was lost, in words.")
    why: str | None = Field(
        default=None, description="Which check or judged dimension failed it, or why it was lost, on one line."
    )
    first_input: str = Field(default="", description="What the case asked.")
    excerpt: str = Field(default="", description="The tail of what the judge was handed.")
    failed_checks: list[TrialCheck] = Field(default_factory=list, description="Each check it failed.")
    judge: list[TrialJudgement] = Field(default_factory=list, description="What the judge said, per dimension.")
    error_words: str | None = Field(default=None, description="The recorded error, for a lost or failed trial.")
    result_id: str = Field(min_length=1, description="The stored result, for `result_get`.")
    open: bool = Field(default=False, description="Whether HTML shows it open: a lost or failed trial, up to a cap.")


class DisclosureBlock(_Block):
    """Something code must tell the reader that no author wrote — one idea."""

    kind: Literal["disclosure"] = Field(default="disclosure", description="Which kind of block this is.")
    source: DisclosureSource = Field(description="What the disclosure speaks for.")
    text: str = Field(min_length=1, description="The disclosure, one sentence or a few.")
    term: str | None = Field(
        default=None, description="The glossary term the line links to, where it uses one; None for none."
    )

    @model_validator(mode="after")
    def _the_term_is_defined(self) -> Self:
        """Refuse a link to a term the glossary does not define.

        Raises:
            ValueError: ``term`` is not a :data:`~threetears.evals.analysis.style.GLOSSARY` key.
        """
        if self.term is not None and self.term not in GLOSSARY:
            raise ValueError(f"a disclosure links the term {self.term!r}, which the glossary does not define")
        return self


#: A report block, discriminated by ``kind``.
ReportBlock = Annotated[TextBlock | TableBlock | ChartBlock | TrialBlock | DisclosureBlock, Field(discriminator="kind")]


class ReportSource(EvalDocumentModel):
    """What the report is a report of, and how that analysis was generated — or that none was."""

    analysis_id: str | None = Field(
        default=None, min_length=1, description="The analysis the report renders; None on a code-only report."
    )
    campaign_id: str = Field(min_length=1, description="The campaign.")
    campaign_name: str | None = Field(default=None, min_length=1, description="The campaign's name, where known.")
    scope_id: str = Field(min_length=1, description="The scope the campaign lives in.")
    behavior: str = Field(min_length=1, description="The behaviour under analysis.")
    generated_at: str = Field(
        min_length=1,
        description="When the analysis was generated (ISO-8601); on a code-only report, when the evidence was assembled.",
    )
    generator_model: str | None = Field(
        default=None, min_length=1, description="The model that wrote the analysis; None on a code-only report."
    )
    bundle_fingerprint: str = Field(min_length=1, description="The fingerprint of the bundle the report renders.")


#: The files :meth:`Report.write` writes, by role.
ReportFile = Literal["html", "bundle", "markdown"]


class Report(EvalDocumentModel):
    """A campaign's report: one page, the bundle it was computed from, and the typed verdicts it states."""

    basis: ReportBasis = Field(description="`analysis` over a generated memo, `code_only` over the evidence alone.")
    title: str = Field(min_length=1, description="The title line: the question the campaign asks, in plain words.")
    finding_count: int = Field(ge=0, description="How many findings the memo holds.")
    source: ReportSource = Field(description="What this is a report of.")
    blocks: list[ReportBlock] = Field(description="The report, in reading order.")
    verdicts: list[Verdict] = Field(
        default_factory=list,
        description=(
            "Every verdict code reaches, typed — the guardrails, then each bar on each cell, then the contrasts "
            "against the control — the source each printed verdict is rendered from."
        ),
    )
    bundle: AnalysisContextBundle = Field(
        exclude=True, description="The evidence every number here was computed from, written beside the page."
    )

    @model_validator(mode="after")
    def _positions_point_at_findings(self) -> Self:
        """Refuse a block linked to a finding the report does not hold.

        Raises:
            ValueError: A block's ``finding`` or ``rests_on`` names a position past ``finding_count``.
        """
        for index, block in enumerate(self.blocks):
            named = ([block.finding] if block.finding is not None else []) + list(block.rests_on)
            if stray := sorted({position for position in named if not 0 <= position < self.finding_count}):
                raise ValueError(
                    f"block {index} ({block.kind}) is linked to finding position(s) {stray}, and the report holds "
                    f"{self.finding_count} finding(s)"
                )
        return self

    @model_validator(mode="after")
    def _the_basis_matches_what_the_report_holds(self) -> Self:
        """Refuse a report whose basis disagrees with its source or its blocks.

        Raises:
            ValueError: An analysis report names no analysis or model, or a code-only report names one or holds a
                finding or an author's words.
        """
        source = self.source
        if self.basis == "analysis":
            if source.analysis_id is None or source.generator_model is None:
                raise ValueError("an analysis report names its analysis_id and generator_model in its source")
            return self
        if source.analysis_id is not None or source.generator_model is not None:
            raise ValueError("a code-only report renders no analysis, so it names no analysis_id or generator_model")
        if self.finding_count:
            raise ValueError(f"a code-only report has no findings, and this one counts {self.finding_count}")
        if authored := [index for index, block in enumerate(self.blocks) if isinstance(block, TextBlock)]:
            raise ValueError(f"a code-only report holds no author's words, and blocks {authored} are text blocks")
        return self

    @property
    def terms(self) -> list[str]:
        """Every glossary term the page links, in the order first linked — what its glossary defines."""
        linked: list[str] = []
        for block in self.blocks:
            found = (
                [column.term for column in block.columns]
                if isinstance(block, TableBlock)
                else [block.term]
                if isinstance(block, DisclosureBlock)
                else []
            )
            for term in found:
                if term is not None and term not in linked:
                    linked.append(term)
        return linked

    def write(self, out_dir: Path | str) -> dict[ReportFile, Path]:
        """Write the report as one self-contained page, its Markdown, and the bundle it was computed from.

        ``report.html`` needs nothing beside it — no script, no stylesheet, no image — and ``bundle.json`` is the
        bundle with its ``schema_version``, so the fingerprint the page names is the one a reader computes from the
        file (``AnalysisContextBundle.from_json(...).fingerprint()``). ``report.md`` is the same page as Markdown.

        Args:
            out_dir: The directory; created when missing.

        Returns:
            Each file written, by role.
        """
        # Deferred: the serializers import this module for the block types they lay out.
        from threetears.evals.analysis.report.serialize_html import report_html
        from threetears.evals.analysis.report.serialize_md import report_markdown

        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        files: dict[ReportFile, Path] = {
            "html": directory / "report.html",
            "bundle": directory / "bundle.json",
            "markdown": directory / "report.md",
        }
        files["html"].write_text(report_html(self), encoding="utf-8")
        files["bundle"].write_text(self.bundle.to_json(indent=2) + "\n", encoding="utf-8")
        files["markdown"].write_text(report_markdown(self), encoding="utf-8")
        return files


__all__ = [
    "FINDING_ROLES",
    "FIRST_SCREEN",
    "SECTION_TITLES",
    "VERDICT_OUTCOMES",
    "BarOutcome",
    "CellValue",
    "ChartBlock",
    "ChartKind",
    "ContrastOutcome",
    "DisclosureBlock",
    "DisclosureSource",
    "Fact",
    "GuardrailOutcome",
    "IntervalRow",
    "OutcomeBar",
    "OutcomeSegment",
    "Report",
    "ReportBasis",
    "ReportBlock",
    "ReportFile",
    "ReportSection",
    "ReportSource",
    "TableBlock",
    "TableCell",
    "TableColumn",
    "TextBlock",
    "TextRole",
    "TrialBlock",
    "Verdict",
    "VerdictKind",
    "VerdictReason",
    "interval_text",
    "interval_value_text",
    "table_row",
]

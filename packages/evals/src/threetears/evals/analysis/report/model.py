"""The ``Report`` — one versioned document every surface renders an analysis from.

**The report is a document, not an assembly.** A stored analysis used to reach a reader through four
reads (the analysis, its arm table, its decision-surface table, its compiled charts) that each surface
put together for itself, and the memo's Markdown dropped every chart's data on the way. A
:class:`Report` is all of it, in reading order, as an ordered list of blocks — ``text`` with a role,
``table``, ``chart`` and ``disclosure`` — each linked to the findings it belongs to or rests on. It has
a published JSON Schema (``schema.json`` beside this module), so a host generates its client types from
the schema rather than mirroring them by hand. **The schema checks the shape and every cross-field rule JSON
Schema can state** — a code-only report names no analysis or model and holds no headline, finding or text
block; a report with no findings links no block to one; a finding's own words name their finding; a chart
block carries exactly one of an intent and an error, the intent of its own type; a table cell's unit qualifies
only a number. Three rules compare a value
with a sibling's, which JSON Schema cannot: a block's finding positions are below ``finding_count`` on a
report that has findings, a table's ``total_rows`` is at least the rows it shows, and a row keys exactly its
table's columns. Those the model's validators hold, so a document is a report when the model accepts it, and
the schema accepts a superset by exactly those three. It ships three serializers: JSON (canonical, this
model's own dump), Markdown (the agent-facing form) and HTML that reads without a script.

**The words are served, not ported.** Every word a reader is shown is in the JSON, written here: the report's
:attr:`~Report.title` and :attr:`~Report.byline`, each block's :attr:`~_Block.section_title`, the
:attr:`~_Block.finding_number` and :attr:`~_Block.rests_on_numbers` a reader counts findings by, and each table
cell's :attr:`~TableCell.text` beside its :attr:`~TableCell.value` — the value for sorting and drawing, the text spelled by the one number
rule (:func:`~threetears.evals.analysis.viz.quantities.render_cell`). A chart block serves its values-as-drawn
table the same way (:attr:`ChartBlock.columns`, :attr:`ChartBlock.rows`), its columns already chosen for
printing. So a browser, a terminal and this package's own serializers print the same words, and no client
re-implements the number rule or the section headings. They are derived from the fields beside them
(each a ``computed_field``), never stored: a document read back has them re-derived and what it said for them
discarded, so a served word cannot drift from what it states, and the schema marks every one required, so a
client generating its types from it can rely on each.

**A report is of an analysis, or of the evidence alone — and it says which** (:attr:`Report.basis`). A
campaign with a generated analysis is reported through it (``analysis``). A campaign with none — no
analyst has run, and the package ships no keyless one — is still reported (``code_only``): the arm
table, the decision surface, the contrasts against the control, a chart per measure, a classifier's
per-label table and every disclosure the evidence carries, all computed by code, with NO text block at
all, and a disclosure stating plainly that no analysis was generated. The shape refuses a code-only
report carrying an author's words, and an analysis report missing its analysis.

**Who wrote what is part of the shape.** A ``text`` block holds what the analysis's author wrote, and
nothing else; what code has to add — a chart's restatement, an arm the join could not place, why the
time axis is days — is a ``disclosure``, never appended to the author's words. A chart block carries its
:class:`~threetears.evals.analysis.viz.intent.ChartIntent` (what it draws and must say) or the reason it
cannot be drawn — never a renderer's spec, which is the host's to make.

**Where a caveat goes.** An author's caveat is attached to its finding, always, as written — the author
placed it there, and moving it would be code overruling the author about what qualifies their claim. The
authored :class:`~threetears.evals.kernel.authored.Caveat` carries no magnitude, so placing caveats by
how material they are would mean reading materiality out of prose, which code does not do; the number
that does say whether a difference is worth acting on — a measure's materiality threshold — is code's,
and it already labels the delta it applies to (``immaterial``) where that delta is drawn. What belongs in
an appendix is what no single finding owns: the ``methods`` section holds the analysis-wide disclosures
(the arms the join could not place, the bars no cell could be read against, the time axis's basis, how
the analysis was generated).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, Self, get_args

from pydantic import ConfigDict, Field, computed_field, model_validator

from threetears.evals.analysis.viz.intent import ChartValue, ChartColumn, ChartIntent, ChartType
from threetears.evals.analysis.viz.intents.distribution import SHAPE_UNKNOWN
from threetears.evals.analysis.viz.quantities import render_cell, with_unit
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.prose import ModelProse

#: The report shape's version. Moves when a field or block kind is added, renamed or removed, or when a
#: field's meaning moves under its name; a host reads it to know what it was handed.
#:
#: 2: ``basis`` added (a report of an analysis, or a code-only report of the evidence alone);
#: ``source.analysis_id`` and ``source.generator_model`` became nullable (None on a code-only report),
#: and ``source.generated_at`` / ``source.bundle_fingerprint`` mean the assembly's on a code-only report;
#: disclosure sources ``runs``, ``measurement``, ``apparatus`` and ``comparisons`` added; the
#: ``comparisons`` and ``questions`` tables added; a chart block with no finding (a chart code chose).
#:
#: 3: the ``strata`` table added (each arm's figures per stratum of its cases, beside the pooled figure) and
#: the ``strata`` disclosure source (a stratum too small to read on its own).
#:
#: 4: the ``labels`` table added (a classifier's per-label precision, recall and F1, every arm's, on a code-only
#: report), in place of the code-only report's distribution chart per label and statistic, which it no longer
#: carries; the ``surface`` table's ``notes`` column is present only when some row has a run note, and the ``arms``
#: table's ``status`` column only when some arm's status is other than unresolved and its ``findings`` column only
#: when some arm rests on a finding.
#:
#: 5: the ``guardrails`` section and its ``guardrails`` table (each guardrail, held, breached or undecided for each
#: arm against the control) and the ``guardrails`` disclosure source; the ``scope`` disclosure source (readings no
#: declared question asked about, or the one sentence that a campaign declaring none is exploratory throughout);
#: a finding's ``Scope`` fact, present only on a finding resting wholly on exploratory readings, and an adopted
#: decision's ``Guardrails`` fact, present only when a guardrail on its arm is not held.
#:
#: 6: ``verdicts`` added — every contrast against the control, bar reading and guardrail reading as a typed
#: :class:`Verdict` (its outcome, a structured reason, its margin and where that came from, its materiality and
#: whether it is a guardrail), in the order the report's tables lay them out; the ``verdict`` and ``decision``
#: cells of the ``comparisons`` and ``guardrails`` tables are each verdict's ``words``, rendered from it. An
#: ``equivalent`` contrast on a margin its runs declared says so (``declared on the runs``), and the
#: ``comparisons`` disclosure source carries why a margin the runs disagree on was read on no contrast.
#:
#: 7: the words are served (see the module docstring). Every table cell is a :class:`TableCell` — its ``value`` and
#: its ``text`` — where it was a bare value; a chart block serves its values-as-drawn table as ``columns`` and
#: ``rows`` of cells, its columns chosen for printing (a distribution's ``shape`` left out when no row knows one);
#: the report carries its ``title`` and ``byline``, and every block its ``section_title`` and ``finding_number``. An
#: analysis report's ``source.campaign_name`` is set where the campaign is found, and its title then names the
#: campaign. Every field with a default is required in the schema, since every dump carries it.
#:
#: 8: every block serves ``rests_on_numbers``, the findings it rests on counted from 1 as ``finding_number`` counts,
#: beside ``rests_on``, which keeps counting from 0. A field added, so the version moves: the schema refuses a key
#: it does not name, so a client holding the version 7 schema would refuse a document carrying it.
REPORT_VERSION: Literal[8] = 8

#: What a report is of: a generated analysis, or the campaign's evidence alone with no analysis.
ReportBasis = Literal["analysis", "code_only"]

#: Where a block sits, in reading order.
ReportSection = Literal[
    "summary", "questions", "decisions", "guardrails", "findings", "arms", "surface", "next", "methods"
]

#: Each section's heading, in reading order — the order every serializer lays the sections out in, and the
#: heading every block carries as its ``section_title``.
SECTION_TITLES: dict[str, str] = {
    "summary": "Summary",
    "questions": "Declared questions",
    "decisions": "Decisions",
    "guardrails": "Guardrails",
    "findings": "Findings",
    "arms": "Arms",
    "surface": "Decision surface",
    "next": "What to run next",
    "methods": "Methods and disclosures",
}

#: What a text block's author wrote it as.
TextRole = Literal[
    "summary",
    "answer",
    "decision",
    "revisit_when",
    "finding_title",
    "finding_body",
    "caveat",
    "carried_forward",
    "next_step",
    "next_step_why",
]

#: Who a disclosure speaks for. ``runs``: member runs left out, unfinished or short. ``measurement``: how
#: and when the runs were launched and measured. ``apparatus``: the rig — controls, confounds, cells that
#: did not pool. ``comparisons``: how the contrasts against the control were tested and corrected.
#: ``strata``: a stratum of cases too small to be read on its own. ``guardrails``: how the guardrails were decided,
#: or why none could be. ``scope``: which readings no declared question asked about — exploratory.
DisclosureSource = Literal[
    "chart",
    "arms",
    "surface",
    "time_axis",
    "generation",
    "runs",
    "measurement",
    "apparatus",
    "comparisons",
    "strata",
    "guardrails",
    "scope",
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
#: there is more to say). A contrast: ``separated`` (its Holm-adjusted p is below α and its interval excludes 0), ``inside_margin`` (the
#: equivalence test shows the difference inside the margin), ``no_margin`` (not separated, and with no margin it
#: could not be shown equivalent), ``margin_untested`` (not separated, a margin is declared and no equivalence test
#: could run: no declared range, or an unpaired test), ``not_inside_margin`` (not separated, and the equivalence
#: test did not show it inside the margin), ``untestable`` (no test could run). A bar: ``interval_clears``,
#: ``interval_misses``, ``interval_straddles``, ``too_few_observations`` (a value from fewer than two, which is not
#: read), ``no_observations``. A guardrail: ``within_margin`` (held), ``beyond_margin`` (breached),
#: ``interval_straddles`` or ``no_interval`` (undecided).
VerdictReason = Literal[
    "separated",
    "inside_margin",
    "no_margin",
    "margin_untested",
    "not_inside_margin",
    "untestable",
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

    The source of truth for every verdict the report prints: the ``comparisons`` table's ``verdict`` cell and the
    ``guardrails`` table's ``decision`` cell are each a verdict's :attr:`words`, rendered from it, never the reverse.
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
    reading: Literal["measure", "judged"] = Field(description="Whether `name` is a measure or a judged dimension.")
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
        description="The finding this block belongs to, by its position in the document (0 is the first); None when it belongs to none.",
    )
    rests_on: list[int] = Field(
        default_factory=list, description="Positions of the findings this block rests on, as the author linked them."
    )

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description="The section's heading, as every serializer prints it (`SECTION_TITLES`)."
    )
    @property
    def section_title(self) -> str:
        """The heading of the section the block sits in."""
        return SECTION_TITLES[self.section]

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "The finding this block belongs to as a reader counts findings — from 1, where `finding` counts from 0; "
            "None when it belongs to none."
        )
    )
    @property
    def finding_number(self) -> int | None:
        """``finding`` counted from one, as every serializer numbers a finding; None when the block names none."""
        return None if self.finding is None else _reader_number(self.finding)

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "The findings this block rests on as a reader counts findings — from 1, as `finding_number` counts, where "
            "`rests_on` counts from 0; in `rests_on`'s order."
        )
    )
    @property
    def rests_on_numbers(self) -> list[int]:
        """``rests_on`` counted from one, as ``finding_number`` counts and every serializer prints them."""
        return [_reader_number(position) for position in self.rests_on]


def _reader_number(position: int) -> int:
    """A finding's position in the document (0 is the first) as a reader counts findings: from one."""
    return position + 1


#: The text roles that are part of one finding, and so name it.
FINDING_ROLES: frozenset[str] = frozenset({"finding_title", "finding_body", "caveat", "carried_forward"})


#: Only whitespace — what an empty author field, or a chart block's empty error, reads as.
_BLANK = r"^\s*$"

#: Some non-whitespace — what a chart block's error is when it carries one.
_SAID = r"\S"


class TextBlock(_Block):
    """What the analysis's author wrote, exactly as written, with the facts code states beside it."""

    model_config = ConfigDict(
        json_schema_extra={
            # A finding's own words name it (``_a_findings_own_words_name_it``), stated in the schema too.
            "if": {"properties": {"role": {"enum": [role for role in sorted(FINDING_ROLES)]}}, "required": ["role"]},
            "then": {"properties": {"finding": {"type": "integer"}}, "required": ["finding"]},
        }
    )

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


class TableCell(EvalDocumentModel):
    """One cell of a report table: its value, and the text a reader is shown for it.

    The value is what a client sorts by and draws from; the text is what it prints, spelled here by the one
    number rule (:func:`~threetears.evals.analysis.viz.quantities.render_cell`: an absent value is a dash, never a
    zero; a boolean is yes or no), so a client never formats a number of its own.
    """

    model_config = ConfigDict(
        json_schema_extra={
            # A unit qualifies only a number (``_a_unit_qualifies_a_number``), stated in the schema too: JSON
            # Schema's `number` holds no boolean, as the model's check holds none.
            "if": {"properties": {"unit": {"minLength": 1}}, "required": ["unit"]},
            "then": {"properties": {"value": {"type": "number"}}},
        }
    )

    value: ChartValue = Field(description="The cell's value: a number, words, a boolean, or null where there is none.")
    unit: str = Field(
        default="",
        description=(
            "The unit a number in this cell is stated in, already restated onto its measure's one ruler for the "
            "report (a latency stored in ms and reaching a second reads in s on every table). Empty for a "
            "unitless number, for words, and wherever the column's header names the unit instead."
        ),
    )

    @model_validator(mode="after")
    def _a_unit_qualifies_a_number(self) -> Self:
        """Refuse a unit on anything but a number: words and a boolean carry none, and an absent value states none.

        Raises:
            ValueError: ``unit`` is set and ``value`` is not a number.
        """
        if self.unit and (isinstance(self.value, bool) or not isinstance(self.value, int | float)):
            raise ValueError(f"a cell's unit qualifies a number, and this cell holds {self.value!r}")
        return self

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "The value as a reader is shown it: a number spelled by the number rule and joined to its unit, null as "
            "`—`, a boolean as yes or no."
        )
    )
    @property
    def text(self) -> str:
        """``value`` as every serializer prints it."""
        if self.unit and isinstance(self.value, int | float) and not isinstance(self.value, bool):
            return with_unit(self.value, self.unit)
        return render_cell(self.value)


def table_row(values: Mapping[str, ChartValue]) -> dict[str, TableCell]:
    """A table row from its values, each made a :class:`TableCell` — how a builder lays a row into a table block.

    Args:
        values: The row's values, by column key.

    Returns:
        The row's cells, by the same keys, in the same order.
    """
    return {key: TableCell(value=value) for key, value in values.items()}


class TableBlock(_Block):
    """A table code laid out — its columns, its rows in their stated order, and how much of it is shown."""

    kind: Literal["table"] = Field(default="table", description="Which kind of block this is.")
    name: str = Field(
        min_length=1,
        description=(
            "Which table this is: `evidence`, `arms`, `surface`, `unadjudicated_bars`, `comparisons` (the contrasts "
            "against the control, as code tested them), `questions` (the declared questions, on a code-only report), "
            "`strata` (each arm's figures per stratum of its cases, beside its pooled figure, when its cases "
            "declare strata), `labels` (a classifier's per-label precision, recall and F1, a row per label and "
            "arm, on a code-only report) or `coverage` (each lever of an analysis's coverage map, the findings "
            "naming it and the next steps that would measure it)."
        ),
    )
    title: str = Field(min_length=1, description="The table's heading.")
    columns: list[TableColumn] = Field(min_length=1, description="The columns, in display order.")
    rows: list[dict[str, TableCell]] = Field(
        description=(
            "The rows shown, in the stated order, each keyed by every column's key: each cell its value and its text, "
            "an absent value a cell whose value is null."
        )
    )
    order: str = Field(min_length=1, description="The order the rows are in, in words.")
    total_rows: int = Field(ge=0, description="How many rows the table has; more than are shown when it is truncated.")

    @model_validator(mode="after")
    def _rows_fit_the_table(self) -> Self:
        """Refuse a table showing more rows than it has, a row stating a value no column shows, or one missing a column.

        A row keys every column, an absent value as a cell whose value is null, so a client prints each row's
        cells as served and never decides for itself what a missing one reads as.

        Raises:
            ValueError: ``total_rows`` is below the rows shown, or a row carries a key that is not a column, or
                lacks one that is.
        """
        if self.total_rows < len(self.rows):
            raise ValueError(f"table {self.name!r} shows {len(self.rows)} rows but says it has {self.total_rows}")
        keys = {column.key for column in self.columns}
        if stray := sorted({key for row in self.rows for key in row} - keys):
            raise ValueError(f"table {self.name!r} has rows keyed {', '.join(stray)}, which no column shows")
        if missing := sorted({key for row in self.rows for key in keys - row.keys()}):
            raise ValueError(
                f"table {self.name!r} has rows missing {', '.join(missing)}; a row keys every column, an absent "
                "value as a cell whose value is null"
            )
        return self

    def values(self) -> list[dict[str, ChartValue]]:
        """The rows shown as their values alone — what a program reads a table for.

        Returns:
            One mapping per row, in the stated order, each column key to its cell's value.
        """
        return [{key: cell.value for key, cell in row.items()} for row in self.rows]


class ChartBlock(_Block):
    """A chart: a finding's (its intent, or why the stored chart cannot be drawn), or one code chose.

    A chart code chose — on a code-only report, one per measure the surface can draw — names no finding.
    """

    model_config = ConfigDict(
        json_schema_extra={
            # Exactly one of an intent and the reason it cannot be drawn, and the intent of the block's own type
            # (``_drawn_or_says_why_not``), stated in the schema too.
            "if": {"properties": {"intent": {"type": "null"}}, "required": ["intent"]},
            "then": {"properties": {"error": {"type": "string", "pattern": _SAID}}},
            "else": {"properties": {"error": {"type": "string", "pattern": _BLANK}}},
            "allOf": [
                {
                    "if": {"properties": {"viz_type": {"const": chart_type}}, "required": ["viz_type"]},
                    "then": {
                        "properties": {
                            "intent": {"anyOf": [{"type": "null"}, {"properties": {"type": {"const": chart_type}}}]}
                        }
                    },
                }
                for chart_type in get_args(ChartType)
            ],
        }
    )

    kind: Literal["chart"] = Field(default="chart", description="Which kind of block this is.")
    viz_type: ChartType = Field(description="The chart type the finding carries.")
    intent: ChartIntent | None = Field(
        default=None, description="What the chart draws and must say; None when it cannot be drawn."
    )
    error: str = Field(
        default="",
        description=(
            "Why the stored chart cannot be drawn, naming the offending field; empty when it can. Served rather than "
            "omitted: a chart missing from a report is otherwise indistinguishable from a finding that carried none."
        ),
    )

    @model_validator(mode="after")
    def _drawn_or_says_why_not(self) -> Self:
        """Refuse a chart block that carries both an intent and an error, or neither.

        Raises:
            ValueError: The block is drawable and failed, or neither.
        """
        if (self.intent is None) == (not self.error.strip()):
            raise ValueError("a chart block carries exactly one of an intent and the reason it cannot be drawn")
        if self.intent is not None and self.intent.type != self.viz_type:
            raise ValueError(f"a {self.viz_type} chart block carries a {self.intent.type} intent")
        return self

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "The values-as-drawn table's columns as printed, in display order: the intent's, less a distribution's "
            "`shape` when no row knows one. Empty when the chart cannot be drawn."
        )
    )
    @property
    def columns(self) -> list[TableColumn]:
        """The intent's values-table columns, as every serializer prints them."""
        if self.intent is None:
            return []
        return [TableColumn(key=column.key, header=column.header) for column in _printed_columns(self.intent)]

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "The values-as-drawn rows, in drawn order, keyed by `columns`: each cell its value (in the drawn unit) and "
            "its text. Empty when the chart cannot be drawn."
        )
    )
    @property
    def rows(self) -> list[dict[str, TableCell]]:
        """The intent's values-table rows over :attr:`columns`, each cell its value and its text."""
        if self.intent is None:
            return []
        columns = _printed_columns(self.intent)
        return [{column.key: TableCell(value=row.get(column.key)) for column in columns} for row in self.intent.rows]


class DisclosureBlock(_Block):
    """Something code must tell the reader that no author wrote — one idea."""

    kind: Literal["disclosure"] = Field(default="disclosure", description="Which kind of block this is.")
    source: DisclosureSource = Field(description="What the disclosure speaks for.")
    text: str = Field(min_length=1, description="The disclosure, one sentence or a few.")


#: A report block, discriminated by ``kind``.
ReportBlock = Annotated[TextBlock | TableBlock | ChartBlock | DisclosureBlock, Field(discriminator="kind")]


class ReportSource(EvalDocumentModel):
    """What the report is a report of, and how that analysis was generated — or that none was."""

    analysis_id: str | None = Field(
        default=None, min_length=1, description="The analysis the report renders; None on a code-only report."
    )
    campaign_id: str = Field(min_length=1, description="The campaign the analysis is of.")
    campaign_name: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The campaign's name, which the report's title names it by; None where the report was built without "
            "it, and a code-only report's title then names the campaign by its id."
        ),
    )
    scope_id: str = Field(min_length=1, description="The scope both live in.")
    subject_id: str = Field(min_length=1, description="The analysed subject.")
    subject_kind: str = Field(description="The subject's kind; empty when the campaign declared none.")
    behavior: str = Field(min_length=1, description="The behaviour under analysis.")
    generated_at: str = Field(
        min_length=1,
        description=(
            "When the analysis was generated (ISO-8601); on a code-only report, when the evidence was assembled for it."
        ),
    )
    generator_model: str | None = Field(
        default=None,
        min_length=1,
        description="The model that wrote the analysis, as the provider reported it; None on a code-only report.",
    )
    bundle_fingerprint: str = Field(
        min_length=1,
        description=(
            "The fingerprint of the evidence bundle the analysis was written over; on a code-only report, of the "
            "bundle the report was computed from."
        ),
    )


class Report(EvalDocumentModel):
    """One analysis, as a document every surface renders. See the module docstring for the contract."""

    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                # A code-only report names no analysis or model and holds nothing an author wrote
                # (``_the_basis_matches_what_the_report_holds``); an analysis report names both.
                {
                    "if": {"properties": {"basis": {"const": "code_only"}}, "required": ["basis"]},
                    "then": {
                        "properties": {
                            "headline": {"type": "string", "pattern": _BLANK},
                            "finding_count": {"const": 0},
                            "source": {
                                "properties": {"analysis_id": {"type": "null"}, "generator_model": {"type": "null"}}
                            },
                            "blocks": {
                                "items": {"not": {"properties": {"kind": {"const": "text"}}, "required": ["kind"]}}
                            },
                        }
                    },
                    "else": {
                        "properties": {
                            "source": {
                                "properties": {
                                    "analysis_id": {"type": "string"},
                                    "generator_model": {"type": "string"},
                                },
                                "required": ["analysis_id", "generator_model"],
                            }
                        }
                    },
                },
                # A report with no findings links no block to one (``_positions_point_at_findings``, at the one
                # finding_count the schema can compare against).
                {
                    "if": {"properties": {"finding_count": {"const": 0}}, "required": ["finding_count"]},
                    "then": {
                        "properties": {
                            "blocks": {
                                "items": {"properties": {"finding": {"type": "null"}, "rests_on": {"maxItems": 0}}}
                            }
                        }
                    },
                },
            ]
        }
    )

    report_version: Literal[8] = Field(default=REPORT_VERSION, description="This shape's version.")
    basis: ReportBasis = Field(
        description=(
            "`analysis` when the report renders a generated analysis; `code_only` when no analysis exists and the "
            "report is the campaign's evidence as code computed it — no headline, no findings, no author's words."
        )
    )
    headline: ModelProse = Field(
        description="The author's headline, as written; empty when the author wrote none, and on a code-only report."
    )
    finding_count: int = Field(
        ge=0, description="How many findings the document holds — the range every position is in."
    )
    source: ReportSource = Field(description="What this is a report of.")
    blocks: list[ReportBlock] = Field(description="The report, in reading order.")
    verdicts: list[Verdict] = Field(
        default_factory=list,
        description=(
            "Every verdict the report states, typed — the source each printed verdict is rendered from: the "
            "guardrails (for each arm against the control), then each bar on each cell, then the contrasts against "
            "the control, each group in its table's row order. A code-only report carries all three; an analysis "
            "report carries the guardrails and bars its frozen decision surface holds, and no contrast, as it "
            "prints no contrasts table."
        ),
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

        An analysis report names its analysis and the model that wrote it. A code-only report names
        neither, and holds nothing an author wrote: no headline, no findings, no text block — so a
        reader can never take code's computation for an analyst's judgement, or the reverse.

        Raises:
            ValueError: The basis and what the report holds disagree.
        """
        source = self.source
        if self.basis == "analysis":
            if source.analysis_id is None or source.generator_model is None:
                raise ValueError("an analysis report names its analysis_id and generator_model in its source")
            return self
        if source.analysis_id is not None or source.generator_model is not None:
            raise ValueError(
                "a code-only report renders no analysis, so its source names no analysis_id or generator_model"
            )
        if self.headline.strip():
            raise ValueError("a code-only report has no author, so it carries no headline")
        if self.finding_count:
            raise ValueError(f"a code-only report has no findings, and this one counts {self.finding_count}")
        if authored := [index for index, block in enumerate(self.blocks) if isinstance(block, TextBlock)]:
            raise ValueError(f"a code-only report holds no author's words, and blocks {authored} are text blocks")
        return self

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "The report's title, as every serializer prints it: an analysis report's headline (`(blank headline)` "
            "when the author wrote none), led by the campaign's name where the report carries it; a code-only "
            "report's names the campaign, by its name or else its id, and says there is no analysis."
        )
    )
    @property
    def title(self) -> str:
        """The title, one line before escaping."""
        name = self.source.campaign_name
        if self.basis == "code_only":
            return f"Campaign {name or self.source.campaign_id}: its evidence, with no analysis"
        headline = self.headline.strip() or "(blank headline)"
        return f"{name}: {headline}" if name else headline

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "What the report is of and how it was made, as every serializer prints it under the title; it names "
            "the campaign by its id either way."
        )
    )
    @property
    def byline(self) -> str:
        """The byline, before escaping.

        A code-only report's byline does not say that no analysis was generated: the summary's one line
        (``NO_ANALYSIS``) says it, directly below.
        """
        source = self.source
        if self.basis == "code_only":
            return (
                f"Code-only report of campaign {source.campaign_id} — {source.behavior}; computed from its evidence on "
                f"{source.generated_at}."
            )
        return (
            f"Analysis {source.analysis_id} of campaign {source.campaign_id} — {source.behavior}; generated "
            f"{source.generated_at} by {source.generator_model}."
        )

    def to_canonical_json(self) -> str:
        """The report as its canonical JSON — the form the published schema validates.

        Returns:
            The JSON text, indented, keys in model order.
        """
        return self.model_dump_json(indent=2)


def _printed_columns(intent: ChartIntent) -> list[ChartColumn]:
    """The columns of a chart's values table as a chart block serves it: the intent's, less a shape no row knows.

    A distribution's ``shape`` column reads :data:`SHAPE_UNKNOWN` in every row when every group reported only
    an interval, and printed beside a table that shows each group's interval and nothing between its ends, it
    tells the reader nothing the table does not. It is printed as soon as one row's shape is known, and the
    intent keeps it either way, for a renderer that draws the band. Applied where the block serves its table, so
    every client receives the columns already chosen.

    Args:
        intent: The chart's intent.

    Returns:
        The columns to print, in display order.
    """
    if intent.rows and all(row.get("shape") == SHAPE_UNKNOWN for row in intent.rows):
        return [column for column in intent.columns if column.key != "shape"]
    return list(intent.columns)


__all__ = [
    "REPORT_VERSION",
    "VERDICT_OUTCOMES",
    "BarOutcome",
    "ContrastOutcome",
    "GuardrailOutcome",
    "Verdict",
    "VerdictKind",
    "VerdictReason",
    "SECTION_TITLES",
    "ChartBlock",
    "DisclosureBlock",
    "DisclosureSource",
    "FINDING_ROLES",
    "Fact",
    "Report",
    "ReportBasis",
    "ReportBlock",
    "ReportSection",
    "ReportSource",
    "TableBlock",
    "TableCell",
    "TableColumn",
    "TextBlock",
    "TextRole",
    "table_row",
]

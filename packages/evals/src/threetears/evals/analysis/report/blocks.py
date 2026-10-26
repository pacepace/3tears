"""The blocks a report is made of — text, tables, charts, trials and disclosures — and the vocabulary they share.

The pieces a :class:`~threetears.evals.analysis.report.model.Report` and a
:class:`~threetears.evals.analysis.report.model.CampaignReport` hold, defined below both the documents and the
serializers that lay them out (:mod:`~threetears.evals.analysis.report.serialize_html`,
:mod:`~threetears.evals.analysis.report.serialize_md`), so a document can write itself through a serializer imported
at module scope with no cycle. :mod:`~threetears.evals.analysis.report.model` re-exports every name here; import
from there.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, Self, get_args

from pydantic import Field, JsonValue, model_validator

from threetears.evals.analysis.bundle.schema import TrialCheck, TrialJudgement, TrialState
from threetears.evals.analysis.quantities import PERCENT, is_money, number_in, render_cell, unit_label, with_unit
from threetears.evals.analysis.report.charts import (
    KIT_CHARTS,
    BarRow,
    BarSegment,
    BarsChart,
    ChartPattern,
    ChartShape,
    ChartStatus,
    KitChart,
    LegendItem,
    ScatterChart,
    ScatterPoint,
    Token,
)
from threetears.evals.analysis.style import GLOSSARY
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.prose import ModelProse

#: A table cell's value: a number, words, a boolean, or None where there is none.
CellValue = str | int | float | bool | None

#: An id a cell carries or links to: a letter, then letters, digits, hyphens or underscores. No space, since a fold's
#: ids are prefixed word by word, and nothing HTML would have to escape.
_ID = r"^[A-Za-z][A-Za-z0-9_-]*$"


def reading_anchor(cell: str, measure_id: str, reading: str) -> str:
    """The id of the place on a page where code states one reading at one cell, which a finding citing that reading
    links to.

    One id per reading at a cell, whichever table states it, so the cell a finding's evidence names and the table
    row that tests it meet without either reading the other's words.

    :param cell: The cell, as a full ``cell_ref`` (``<variant_key>:<apparatus_class_id>``).
    :ptype cell: str
    :param measure_id: The measure or judged dimension.
    :ptype measure_id: str
    :param reading: ``measure``, ``judged``, or ``trial`` for the overall pass rate.
    :ptype reading: str
    :return: The id, ``r-`` and twelve hex digits.
    :rtype: str
    """
    return "r-" + hashlib.sha256(f"{cell}\0{reading}\0{measure_id}".encode()).hexdigest()[:12]


def fold_ids(labels: Sequence[str]) -> list[str]:
    """Each aspect's fold on a campaign's page as the id its section and every id inside it are named by: the label
    in lower case, each run of other characters one hyphen, numbered where two would read alike.

    :param labels: The aspects' labels, in the campaign's order.
    :ptype labels: Sequence[str]
    :return: One id per label, in the same order.
    :rtype: list[str]
    """
    ids: list[str] = []
    for label in labels:
        slug = re.sub(r"[^a-z0-9]+", "-", label.casefold()).strip("-") or "aspect"
        named, number = slug, len(ids) + 1
        while named in ids:
            named, number = f"{slug}-{number}", number + 1
        ids.append(named)
    return ids


def fold_anchor(fold_id: str) -> str:
    """The id of the section around an aspect's fold on a campaign's page, outside its closed ``<details>``: what every
    link to the fold lands on, and the one place that id is spelled.

    :param fold_id: The fold's id (:func:`fold_ids`).
    :ptype fold_id: str
    :return: ``aspect-<fold id>``.
    :rtype: str
    """
    return f"aspect-{fold_id}"


#: What a report is of: a generated analysis, or the campaign's evidence alone with no analysis.
ReportBasis = Literal["analysis", "code_only"]

#: How much of each failed or lost trial a page lays out. ``table``: the census line, then one row per such trial
#: naming its stored result (read it with ``result_get``), then the passed trials' table. ``all``: each one's full
#: record as well — what it asked, what the judge saw and said, and why — which is most of a large campaign's page.
TrialDetail = Literal["table", "all"]

#: Where a block sits, in reading order. ``header`` is the first screen's title, arms and verdict lines, and
#: ``scoreboard`` the rest of the first screen; ``decision`` is the floors section: each declared floor read clause by clause;
#: ``memo`` and ``contrasts`` are what it means; ``cases`` and ``trials`` the cases and their drill-down;
#: ``not_tested`` what the campaign could not speak to; ``methods`` how everything was measured; ``glossary`` every
#: term of art the page used. A campaign's page has six more (:attr:`CampaignReport.section_titles`): ``answer``, the
#: question and what it was answered; ``warnings``, what warns, as consequences; ``scorecard``, what code read across
#: the aspects; ``findings`` and ``next``, the writer's; and ``provenance``, the design and when each criterion was set.
ReportSection = Literal[
    "header",
    "answer",
    "warnings",
    "scoreboard",
    "scorecard",
    "decision",
    "findings",
    "next",
    "memo",
    "contrasts",
    "provenance",
    "cases",
    "trials",
    "not_tested",
    "methods",
    "glossary",
]

#: The sections a reader opens on demand, each laid out closed: the cases, the trials, what was not tested, the
#: methods and the glossary.
FOLDED_SECTIONS: frozenset[str] = frozenset({"cases", "trials", "not_tested", "methods", "glossary"})

#: Each section's heading, in reading order — the order every serializer lays the sections out in, each three words
#: or fewer. The first screen (``header`` and ``scoreboard``) carries no heading: it is the page's own head. A page
#: reads its headings off :attr:`Report.section_titles`, which heads ``contrasts`` ``Readings`` where nothing is
#: contrasted; the section's id stays ``contrasts`` either way.
SECTION_TITLES: dict[str, str] = {
    "header": "",
    "scoreboard": "Scoreboard",
    "decision": "Floors",
    "memo": "What it means",
    "contrasts": "Contrasts",
    "cases": "Cases",
    "trials": "Trials",
    "not_tested": "Not tested",
    "methods": "Methods",
    "glossary": "Glossary",
}

#: The sections the first screen is made of: a campaign's answer and warnings are on it too.
FIRST_SCREEN: frozenset[str] = frozenset({"header", "answer", "warnings", "scoreboard"})


#: What a text block's author wrote it as. ``summary`` is the memo's read across every aspect, which a campaign's page
#: opens on after its warnings.
TextRole = Literal[
    "summary",
    "answer",
    "recommendation",
    "reason",
    "override",
    "what_changed",
    "how_sure",
    "means",
    "finding_title",
    "finding_body",
    "caveat",
    "carried_forward",
    "next_step",
    "next_step_why",
]

#: Who a disclosure speaks for. ``arms``: the first screen's arms line, each arm named in full once by the version
#: that answered it, the control marked, and the design (``30 cases × 3 repeats``). ``caveat``: a line under the
#: scoreboard that changes how the verdict is read, in normal weight: the case floor, an arm's label that its own row
#: contradicts. ``basis``: the question declared, quoted whole under the title. ``timing``: under it, each criterion
#: set after the readings it judges were visible, with when, or that its time is not recorded. ``floor``: on the first
#: screen, a broken floor's warning, the page's first line; in the Decision section, the lines under the floors table
#: saying a floor warns and never decides, and how the cases that would settle one were counted. ``verdict``: the
#: verdict line. ``design``, ``charts`` and ``cost``: how the run was laid out, what was not charted and where a cost
#: is a floor, in the methods; ``cost`` is also the first-screen line saying each cost figure is its runs' provider
#: routing's. ``memo``: who wrote the memo, or that no one did. ``runs``: member runs left out, unfinished or short. ``measurement``: how and when the runs
#: were launched and measured. ``apparatus``: the test setup — controls, confounds, cells that did not pool, a judge
#: grading its own model's answers. ``comparisons``: how the contrasts against the control were tested and
#: corrected. ``strata``: a stratum too small to read alone. ``guardrails``: how guardrails were decided.
#: ``scope``: which readings no declared question asked about. ``not_tested``: something the campaign could not
#: speak to. ``calibration``: a judge not yet calibrated against people's labels — a state with its next step, never a
#: failure. ``generation``: how the report itself was made. ``call``: on a campaign's page, the person's call, or that
#: none is recorded yet, and every broken floor it overrides. ``bar``: on a campaign's page, a bar an arm misses, a
#: line as loud as a broken floor's and as early.
DisclosureSource = Literal[
    "arms",
    "bar",
    "call",
    "caveat",
    "basis",
    "timing",
    "floor",
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
    aspect: str | None = Field(
        default=None,
        description=(
            "The aspect whose question's family tested a contrast; None for the exploratory family, and for a bar or "
            "guardrail."
        ),
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

        :raises ValueError: The outcome is not one of the kind's, or ``guardrail`` is not ``kind == "guardrail"``.
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

        :raises ValueError: A block in one of those roles has no ``finding``.
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

        :raises ValueError: ``term`` is not a :data:`~threetears.evals.analysis.style.GLOSSARY` key.
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
    low: float | None = Field(
        default=None,
        description=(
            "The low end of the interval on the number, in `unit`, where the number has one: HTML draws the interval "
            "beside the number as a small glyph, on one scale for every cell of the column in the same unit. None "
            "where there is no interval, or the row writes it in a column of its own only."
        ),
    )
    high: float | None = Field(default=None, description="The interval's high end, in `unit`; set with `low`.")
    margin: float | None = Field(
        default=None,
        ge=0,
        description=(
            "A declared margin either side of zero, in `unit`, which the glyph draws as a quiet band; None where "
            "none is declared or the number is not a difference."
        ),
    )
    zero_line: bool = Field(
        default=False,
        description=(
            "Whether the number is a difference, read against zero: the glyph holds zero on its scale and draws a "
            "line there. False for a figure read on its own scale."
        ),
    )
    anchor: str | None = Field(
        default=None,
        pattern=_ID,
        description=(
            "The id this cell carries, so a link elsewhere on the page lands on it: a cell printing one reading's own "
            "figure at one cell (:func:`reading_anchor`), the first such cell on the page. None for none."
        ),
    )
    ref: str | None = Field(
        default=None,
        pattern=_ID,
        description=(
            "An id on the page the cell links to: a finding's figure links to another cell printing the same figure "
            "of the same reading at the same cell, or to the heading of the closed fold holding it; a campaign's "
            "trace links to the heading of the aspect's fold that prints it. None where nothing on the page prints it."
        ),
    )
    flag: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "A short marker code writes beside the value, where the cell reads differently from its row: a floor's "
            "status that differs from the row's reading, or a reading that cannot move (`cannot show a drop`). Every "
            "serializer prints it after the value, in parentheses. None for none."
        ),
    )
    status: ChartStatus | None = Field(
        default=None,
        description=(
            "A status the cell reads as, drawn in HTML as the kit's pill: its colour and icon beside the cell's text, "
            "which says the status in words, so it never rests on a colour alone. Markdown prints the words. None for "
            "none."
        ),
    )
    tag: str | None = Field(
        default=None,
        pattern=r"^[A-Z]+$",
        description=(
            "An arm's letter tag, where the cell names an arm: HTML draws the cell as the kit's arm chip, the tag on "
            "the arm's identity colour (`color`) before the cell's text, its name; Markdown prints `<tag> · <name>`. "
            "None for none."
        ),
    )
    color: Token | None = Field(
        default=None,
        description="The arm chip's colour token, set with `tag`; the kit's ink-gray where the chip names none.",
    )

    @model_validator(mode="after")
    def _a_chip_or_a_pill_is_drawn_whole(self) -> Self:
        """Refuse a chip's colour with no chip, and a pill and a chip on one cell.

        :raises ValueError: ``color`` is set without ``tag``, or ``status`` and ``tag`` are both set.
        """
        if self.color is not None and self.tag is None:
            raise ValueError("a cell's colour colours its arm chip's tag, and this cell has no tag")
        if self.status is not None and self.tag is not None:
            raise ValueError("a cell is drawn as a status pill or as an arm chip, and this cell sets both")
        return self

    @model_validator(mode="after")
    def _a_unit_qualifies_a_number(self) -> Self:
        """Refuse a unit on anything but a number.

        :raises ValueError: ``unit`` is set and ``value`` is not a number.
        """
        if self.unit and not self.is_number:
            raise ValueError(f"a cell's unit qualifies a number, and this cell holds {self.value!r}")
        return self

    @model_validator(mode="after")
    def _an_interval_brackets_a_number(self) -> Self:
        """Refuse an interval with one end, on anything but a number, or running backwards, and a margin alone.

        :raises ValueError: ``low`` and ``high`` are not both set or both absent, the cell holds no number, ``low`` is
            above ``high``, or ``margin`` is set with no interval.
        """
        if (self.low is None) != (self.high is None):
            raise ValueError("a cell's interval has two ends, and this cell sets one")
        if self.low is None:
            if self.margin is not None:
                raise ValueError("a cell's margin is drawn beside its interval, and this cell has none")
            return self
        if not self.is_number:
            raise ValueError(f"a cell's interval brackets a number, and this cell holds {self.value!r}")
        if self.high is not None and self.low > self.high:
            raise ValueError(f"the cell's interval runs from {self.low} down to {self.high}")
        return self

    @property
    def is_number(self) -> bool:
        """Whether the cell holds a number (a boolean is not one)."""
        return isinstance(self.value, int | float) and not isinstance(self.value, bool)

    @property
    def text(self) -> str:
        """``value`` as every serializer prints it, its ``flag`` after it in parentheses; the interval, where the cell
        has one, is the glyph's and the row's own interval column's, never spelled here twice."""
        if self.unit and isinstance(self.value, int | float) and not isinstance(self.value, bool):
            said = with_unit(self.value, self.unit)
        else:
            said = render_cell(self.value)
        return said if self.flag is None else f"{said} ({self.flag})"


def table_row(values: Mapping[str, CellValue | TableCell]) -> dict[str, TableCell]:
    """A table row from its values, each made a :class:`TableCell`; a value already a cell (one that links, say) is
    kept as it is.

    :param values: The row's values, or cells, by column key.
    :ptype values: Mapping[str, CellValue | TableCell]
    :return: The row's cells, by the same keys, in the same order.
    :rtype: dict[str, TableCell]
    """
    return {key: value if isinstance(value, TableCell) else TableCell(value=value) for key, value in values.items()}


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
    title_ref: str | None = Field(
        default=None,
        description=(
            "An anchor on the page the title links to, where the table reads a sentence said there; None for none."
        ),
    )
    shared_scale: bool = Field(
        default=False,
        description=(
            "Whether HTML draws every glyph in one unit on one scale across all the table's columns, as a matrix's "
            "are; otherwise each column has its own."
        ),
    )

    @model_validator(mode="after")
    def _rows_fit_the_table(self) -> Self:
        """Refuse a table showing more rows than it has, or a row keying other columns than the table's.

        :raises ValueError: ``total_rows`` is below the rows shown, or a row's keys are not exactly the columns'.
        """
        if self.total_rows < len(self.rows):
            raise ValueError(f"table {self.name!r} shows {len(self.rows)} rows but says it has {self.total_rows}")
        keys = {column.key for column in self.columns}
        if bad := sorted({key for row in self.rows for key in row.keys() ^ keys}):
            raise ValueError(f"table {self.name!r} has rows that do not key exactly its columns: {', '.join(bad)}")
        return self

    def values(self) -> list[dict[str, CellValue]]:
        """The rows shown as their values alone — what a program reads a table for.

        :return: One mapping per row, in the stated order, each column key to its cell's value.
        :rtype: list[dict[str, CellValue]]
        """
        return [{key: cell.value for key, cell in row.items()} for row in self.rows]


#: The charts a report draws: the report's own three (an outcome bar, interval rows, and a campaign's cost against its
#: pass rate), each drawn by the eval report kit as its nearest kind; and the kit's own kinds
#: (:data:`~threetears.evals.analysis.report.charts.KIT_CHARTS`), each carrying its data as the kit's contract names
#: it. A plan's ``table`` kind is a :class:`TableBlock`, which the kit does not draw.
ChartKind = Literal[
    "outcome_bar",
    "interval_rows",
    "cost_quality",
    "bars",
    "diverging_bars",
    "forest",
    "lines",
    "scatter_frontier",
    "gate_scatter",
    "dot_interval",
    "range_bars",
    "gate_matrix",
    "lattice",
    "heatmap",
    "diff_heatmap",
    "confusion",
    "examples",
]

#: The report's own chart kinds, which carry their marks in fields of the block rather than as the kit's data.
_OWN_CHARTS = frozenset({"outcome_bar", "interval_rows", "cost_quality"})

#: The textures a failure cause or a loss cause takes, in order.
_PATTERNS: tuple[ChartPattern, ...] = ("hatch", "cross", "dots", "rows", "cols")

#: Each outcome's look on an outcome bar, by state, as ``(colour token, pattern)``, each cause taking the next in its
#: state's order. Status colours, never a series colour, since series colours name arms (GENERATOR.md §5.2): passed is
#: the kit's good, solid; every failure cause is the kit's bad, each under a texture of its own, so a failure never
#: reads as a pass even without colour, and the five failure causes (``FailCause``) are five looks; a loss cause is a
#: neutral, solid and then textured, so a loss reads as neither. Loss causes past the twelfth repeat a look; the
#: legend and the tooltip name every cause in words.
_OUTCOME_STYLES: dict[TrialState, tuple[tuple[str, ChartPattern | None], ...]] = {
    "passed": (("--good", None),),
    "failed": tuple(("--bad", pattern) for pattern in _PATTERNS),
    "lost": (
        ("--base", None),
        ("--muted", None),
        *((token, pattern) for pattern in _PATTERNS for token in ("--base", "--muted")),
    ),
}

#: Each candidate's identity colour, by its tag's place after the control (A the first): the kit's series in fixed
#: order (GENERATOR.md §5.2), repeating past the fifth.
_SERIES = ("--s1", "--s2", "--s3", "--s4", "--s5")

#: The control's identity colour: the kit's ink-gray.
_CONTROL_TOKEN = "--ink-2"

#: The control's tag (GENERATOR.md §6.1).
_CONTROL_TAG = "P"

#: Each candidate's mark on a scatter, by its tag's place, so an arm reads apart without its colour; the control's is
#: a square.
_SHAPES: tuple[ChartShape, ...] = ("circle", "triangle", "diamond")


def _tag_place(tag: str) -> int:
    """A candidate's place after the control by its tag: A is 0, Z 25, AA 26."""
    place = 0
    for letter in tag:
        place = place * 26 + (ord(letter) - ord("A") + 1)
    return place - 1


def _arm_token(tag: str | None) -> str | None:
    """An arm's identity colour by its tag: the control's ink-gray, a candidate's series colour; None untagged."""
    if tag is None:
        return None
    return _CONTROL_TOKEN if tag == _CONTROL_TAG else _SERIES[_tag_place(tag) % len(_SERIES)]


def _arm_shape(tag: str | None) -> ChartShape | None:
    """An arm's scatter mark by its tag: the control's square, a candidate's shape in order; None untagged."""
    if tag is None:
        return None
    return "square" if tag == _CONTROL_TAG else _SHAPES[_tag_place(tag) % len(_SHAPES)]


class OutcomeSegment(EvalDocumentModel):
    """One segment of an arm's outcome bar: passed, failed, or lost to one cause, with its count."""

    state: TrialState = Field(description="passed, failed or lost.")
    label: str = Field(min_length=1, description="The segment in words (`passed`, `lost: judge truncated`).")
    count: int = Field(ge=1, description="Trials in it.")


class OutcomeBar(EvalDocumentModel):
    """One arm's census as a stacked bar: every trial it ran, by how it ended."""

    arm: str = Field(min_length=1, description="The arm, as the report names it.")
    segments: list[OutcomeSegment] = Field(description="Passed, then failed, then lost, each cause its own segment.")
    tag: str | None = Field(
        default=None,
        pattern=r"^[A-Z]+$",
        description=(
            "The arm's letter tag (GENERATOR.md §6.1): P for the control, then A, B, C… in the page's order; it names "
            "the arm in tight places and picks its colour. None for a bar stored before arms carried one."
        ),
    )
    short_name: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The arm's short name, drawn where `arm` does not fit: its given label, else what differs from the "
            "control (GENERATOR.md §6.1). None to cut `arm`."
        ),
    )

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

        :raises ValueError: ``low`` is above ``high``.
        """
        if self.low > self.high:
            raise ValueError(f"row {self.label!r}: its interval runs from {self.low} down to {self.high}")
        return self


class CostQualityPoint(EvalDocumentModel):
    """One arm on a cost/quality chart: what a reply costs it, on x, and its pass rate, on y, each with its interval."""

    arm: str = Field(min_length=1, description="The arm, as the page names it.")
    tag: str | None = Field(
        default=None,
        pattern=r"^[A-Z]+$",
        description=(
            "The arm's letter tag (GENERATOR.md §6.1): P for the control, then A, B, C… in the page's order; it labels "
            "the point on a phone and picks its colour and mark. None for a point stored before arms carried one."
        ),
    )
    short_name: str | None = Field(
        default=None,
        min_length=1,
        description="The arm's short name, the point's label: its given label, else what differs from the control.",
    )
    cost: float = Field(ge=0, allow_inf_nan=False, description="What a reply costs the arm, in the chart's `unit`.")
    cost_low: float = Field(allow_inf_nan=False, description="The low end of the cost's interval.")
    cost_high: float = Field(allow_inf_nan=False, description="The high end of the cost's interval.")
    pass_rate: float = Field(ge=0, le=100, description="The arm's pass rate, in percent.")
    pass_rate_low: float = Field(ge=0, le=100, description="The low end of the pass rate's interval, in percent.")
    pass_rate_high: float = Field(ge=0, le=100, description="The high end of the pass rate's interval, in percent.")

    @model_validator(mode="after")
    def _each_interval_holds_its_value(self) -> Self:
        """Refuse an interval that runs backwards or leaves out its value.

        :raises ValueError: An interval's low end is above its value, or its value above its high end.
        """
        if not self.cost_low <= self.cost <= self.cost_high:
            raise ValueError(f"{self.arm!r}: its cost {self.cost} is not within [{self.cost_low}, {self.cost_high}]")
        if not self.pass_rate_low <= self.pass_rate <= self.pass_rate_high:
            raise ValueError(
                f"{self.arm!r}: its pass rate {self.pass_rate} is not within "
                f"[{self.pass_rate_low}, {self.pass_rate_high}]"
            )
        return self


class ChartBlock(_Block):
    """One of the charts a report draws, with every value it draws, so it prints as a table too.

    A report's own kinds (``outcome_bar``, ``interval_rows``, ``cost_quality``) carry their marks in ``bars``, ``rows``
    and ``points``; every other kind carries ``data``, the kit's chart data for that kind, field for field as the
    kit's contract names it."""

    kind: Literal["chart"] = Field(default="chart", description="Which kind of block this is.")
    chart: ChartKind = Field(description="Which chart: one of the report's own three, or one of the kit's kinds.")
    title: str = Field(min_length=1, description="What the chart shows.")
    bars: list[OutcomeBar] = Field(default_factory=list, description="An outcome bar's arms, one bar each.")
    rows: list[IntervalRow] = Field(default_factory=list, description="An interval chart's rows, in drawn order.")
    points: list[CostQualityPoint] = Field(
        default_factory=list, description="A cost/quality chart's arms, one point each, in drawn order."
    )
    unit: str = Field(
        default="",
        description="The unit an interval chart's values, or a cost/quality chart's costs, are stated in; empty for none.",
    )
    zero_line: bool = Field(default=False, description="Whether the chart draws a line at zero (a difference's).")
    margin: float | None = Field(
        default=None, description="A margin band either side of zero, on the chart's ruler; None when none declared."
    )
    data: KitChart | None = Field(
        default=None,
        description=(
            "A kit kind's chart data, as the kit's `data-chart` contract names it (kit/API.md); None for the report's "
            "own three kinds."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _data_of_its_kind(cls, data: Any) -> Any:
        """Read a kit chart's data as the model of the kind the block names, since several kinds' data share field
        names and only the kind tells them apart."""
        if isinstance(data, dict) and isinstance(data.get("data"), dict) and data.get("chart") in KIT_CHARTS:
            return {**data, "data": KIT_CHARTS[data["chart"]].model_validate(data["data"])}
        return data

    @model_validator(mode="after")
    def _draws_what_its_kind_draws(self) -> Self:
        """Refuse a chart with nothing to draw, or one carrying another kind's marks.

        :raises ValueError: An outcome bar with rows, points or no bar; an interval chart with bars, points or no
            row; a cost/quality chart with bars, rows or fewer than two points, since a chart is never drawn for
            one point; one of those three with kit data; or a kit kind without its data, with data of another kind,
            or with an outcome bar's, interval chart's or cost/quality chart's marks.
        """
        if self.chart in _OWN_CHARTS and self.data is not None:
            raise ValueError(f"a {self.chart} chart carries its marks in its own fields, not kit data")
        if self.chart == "outcome_bar" and (self.rows or self.points or not self.bars):
            raise ValueError("an outcome bar draws one bar per arm and no interval row")
        if self.chart == "interval_rows" and (self.bars or self.points or not self.rows):
            raise ValueError("an interval chart draws at least one row and no outcome bar")
        if self.chart == "cost_quality" and (self.bars or self.rows or len(self.points) < 2):
            raise ValueError("a cost/quality chart draws one point per arm, for two arms or more, and nothing else")
        if self.chart in KIT_CHARTS:
            if not isinstance(self.data, KIT_CHARTS[self.chart]):
                raise ValueError(f"a {self.chart} chart carries the kit's {self.chart} data")
            if self.bars or self.rows or self.points:
                raise ValueError(f"a {self.chart} chart draws its data, not an outcome bar's, interval or cost marks")
        return self

    def kit_chart(self) -> tuple[str, dict[str, JsonValue]] | None:
        """The chart as the eval report kit draws it: its kind and its ``data-chart`` JSON.

        A kit kind is its data. An outcome bar is ``bars``, each arm's census stacked from zero, a look per outcome
        named in the legend, each arm's tag in its colour beside its label. A cost/quality chart is
        ``scatter_frontier``, each arm in its colour and mark at its cost per 1,000 replies across and its pass rate
        up, on axes from zero; its intervals are in the chart's table. Interval rows are ``None``: they carry no
        interval level, and the kit never draws an interval without one, so they read as their table alone.

        :return: ``(kind, data)``, or None for a chart the kit does not draw.
        :rtype: tuple[str, dict[str, JsonValue]] | None
        """
        if self.data is not None:
            return self.chart, self.data.kit_data()
        if self.chart == "outcome_bar":
            return "bars", self._outcome_bars().kit_data()
        if self.chart == "cost_quality":
            return "scatter_frontier", self._cost_quality_scatter().kit_data()
        return None

    def _outcome_bars(self) -> BarsChart:
        """An outcome bar as the kit's stacked bars: one row per arm, its tag in its colour beside its label (its short
        name where the label does not fit), a segment per outcome, each outcome's look (:data:`_OUTCOME_STYLES`) the
        same on every arm and named in the legend, the whole label and the census in each row's tooltip."""
        looks: dict[str, tuple[str, ChartPattern | None]] = {}
        used = dict.fromkeys(_OUTCOME_STYLES, 0)
        for bar in self.bars:
            for segment in bar.segments:
                if segment.label not in looks:
                    styles = _OUTCOME_STYLES[segment.state]
                    looks[segment.label] = styles[used[segment.state] % len(styles)]
                    used[segment.state] += 1
        longest = max(bar.total for bar in self.bars)
        rows = [
            BarRow(
                label=bar.arm,
                short_label=bar.short_name if bar.short_name != bar.arm else None,
                tag=bar.tag,
                color=_arm_token(bar.tag),
                segments=[
                    BarSegment(value=segment.count, color=looks[segment.label][0], pattern=looks[segment.label][1])
                    for segment in bar.segments
                ],
                note=f"{bar.total} trials",
                tip=[bar.arm, f"{bar.total} trials", *(f"{s.label} {s.count}" for s in bar.segments)],
            )
            for bar in self.bars
        ]
        legend = [
            LegendItem(label=label, color=colour, mark="square", pattern=pattern)
            for label, (colour, pattern) in looks.items()
        ]
        return BarsChart(rows=rows, domain=(0, longest or 1), fmt="int", legend=legend)

    def _cost_quality_scatter(self) -> ScatterChart:
        """A cost/quality chart as the kit's scatter: each arm a point in its colour and mark at its cost per 1,000
        replies and its pass rate, labelled by its short name (by its tag on a phone), the legend naming each tag's
        arm; the cost axis from zero to past the dearest arm, the pass rate's from zero to 100%, so no axis is cut to
        flatter a gap. Cost is drawn per 1,000 replies, since a reply's own cost is a fraction of a cent an axis
        cannot print, and the axis carries the digits its ticks need (:func:`_cost_axis_format`). No frontier is
        drawn, since none is tested."""
        top = max(point.cost for point in self.points) * _REPLIES_PER_COST_UNIT * 1.15 or 1
        unit = f" ({unit_label(self.unit)})" if self.unit else ""
        tagged = all(point.tag is not None for point in self.points)
        points = [
            ScatterPoint(
                x=point.cost * _REPLIES_PER_COST_UNIT,
                y=point.pass_rate,
                label=point.short_name or point.arm,
                narrow_label=point.tag or point.short_name or point.arm,
                color=_arm_token(point.tag),
                shape=_arm_shape(point.tag),
                tip=[
                    point.arm,
                    f"{with_unit(point.cost * _REPLIES_PER_COST_UNIT, self.unit)} per 1,000 replies",
                    f"{with_unit(point.cost, self.unit)} "
                    f"[{with_unit(point.cost_low, self.unit)}, {with_unit(point.cost_high, self.unit)}] per reply",
                    f"pass rate {with_unit(point.pass_rate, PERCENT)} "
                    f"[{with_unit(point.pass_rate_low, PERCENT)}, {with_unit(point.pass_rate_high, PERCENT)}]",
                ],
            )
            for point in self.points
        ]
        legend = (
            [
                LegendItem(label=f"{point.tag} · {point.arm}", color=_arm_token(point.tag), mark=_arm_shape(point.tag))
                for point in self.points
            ]
            if tagged
            else None
        )
        return ScatterChart(
            points=points,
            x_domain=(0, top),
            y_domain=(0, 100),
            x_fmt=_cost_axis_format(top, money=is_money(self.unit)),
            y_fmt="pct",
            x_title=f"Cost per 1,000 replies{unit}",
            y_title="Pass rate",
            legend=legend,
        )

    def values_table(self) -> tuple[list[str], list[list[str]]]:
        """The chart's values as a table — what Markdown prints and what HTML keeps beside the picture.

        :return: ``(headers, rows)``, each row's cells as text.
        :rtype: tuple[list[str], list[list[str]]]
        """
        if self.data is not None:
            return self.data.values_table()
        if self.chart == "outcome_bar":
            labels = list(dict.fromkeys(segment.label for bar in self.bars for segment in bar.segments))
            headers = ["Arm", "ran", *labels]
            body = []
            for bar in self.bars:
                counts = {segment.label: segment.count for segment in bar.segments}
                body.append([bar.arm, str(bar.total), *(str(counts.get(label, 0)) for label in labels)])
            return headers, body
        if self.chart == "cost_quality":
            # On the chart's own scale, per 1,000 replies, and saying so, so the table beside the figure (or in its
            # place where no script runs) reads what the axis does; each figure's interval follows it, as the pass
            # rate's does.
            cost = f" ({unit_label(self.unit)})" if self.unit else ""
            per = _REPLIES_PER_COST_UNIT
            headers = ["Arm", f"Cost per 1,000 replies{cost}", f"Interval{cost}", "Pass rate (%)", "Interval (%)"]
            return headers, [
                [
                    point.arm,
                    interval_value_text(point.cost * per, self.unit),
                    interval_text(point.cost_low * per, point.cost_high * per, self.unit),
                    interval_value_text(point.pass_rate, PERCENT),
                    interval_text(point.pass_rate_low, point.pass_rate_high, PERCENT),
                ]
                for point in self.points
            ]
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


#: The replies a cost/quality chart states each cost over: a reply's cost per 1,000 replies, as the scoreboard's
#: cost ruler states a cost under a cent.
_REPLIES_PER_COST_UNIT = 1000


def _cost_axis_format(top: float, *, money: bool) -> str:
    """The kit format of a cost axis running from zero to ``top``: as many decimals as a tick a fifth of the way up
    needs to print a digit of its own, so no two ticks read alike and none reads as zero; money keeps at least the
    cents and drops them from $100.

    :param top: The axis's top, in the cost's unit, above zero.
    :ptype top: float
    :param money: Whether the cost is money (``usd``) rather than a plain number (``num``).
    :ptype money: bool
    :return: The format, e.g. ``usd:2`` for an axis to $0.48, ``usd:4`` for one to $0.0048.
    :rtype: str
    """
    digits = max(0, math.ceil(-math.log10(top / 5)))
    if money:
        digits = 0 if top >= 100 else max(2, digits)
    return f"{'usd' if money else 'num'}:{min(digits, 9)}"


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
    open: bool = Field(
        default=False,
        description="Whether the record is shown open, in HTML and Markdown: a failed or lost trial, up to a cap.",
    )


#: What kind of verdict the first screen's line is, for a page that sets each kind apart: ``decided`` (a declared
#: question settled it), ``open`` (too few cases, or no question was declared, so the line offers leads),
#: ``failed`` (the test setup failed, so there is no verdict) and ``summary`` (one arm: a count, nothing to decide).
_VerdictTone = Literal["decided", "open", "failed", "summary"]

#: Each verdict rule's tone, keyed by :data:`~threetears.evals.analysis.report.verdict.VerdictRule` — by name, so the
#: model imports nothing from the builder that applies the rules.
_VERDICT_TONES: dict[str, _VerdictTone] = {
    "setup": "failed",
    "one_arm": "summary",
    "declared": "decided",
    "too_small": "open",
    "exploratory": "open",
}


class DisclosureBlock(_Block):
    """Something code must tell the reader that no author wrote — one idea."""

    kind: Literal["disclosure"] = Field(default="disclosure", description="Which kind of block this is.")
    source: DisclosureSource = Field(description="What the disclosure speaks for.")
    text: str = Field(min_length=1, description="The disclosure, one sentence or a few.")
    term: str | None = Field(
        default=None, description="The glossary term the line links to, where it uses one; None for none."
    )
    # The tone's values are spelled here and on the method, not through the private alias: a type a public signature
    # hands a host must be one it can name, and a Literal of strings needs no name.
    tone: Literal["decided", "open", "failed", "summary"] | None = Field(
        default=None,
        description=(
            "On the verdict line, what kind of verdict it is — decided, open, failed (the test setup) or summary (one "
            "arm) — which HTML sets the line apart by. None on every other disclosure."
        ),
    )

    @staticmethod
    def verdict_tone(rule: str) -> Literal["decided", "open", "failed", "summary"]:
        """The tone a verdict line is set in, by the rule that wrote it.

        :param rule: The rule, as :func:`~threetears.evals.analysis.report.verdict.verdict_line` names it.
        :ptype rule: str
        :return: The tone.
        :rtype: Literal['decided', 'open', 'failed', 'summary']
        """
        return _VERDICT_TONES[rule]

    @model_validator(mode="after")
    def _the_term_is_defined(self) -> Self:
        """Refuse a link to a term the glossary does not define.

        :raises ValueError: ``term`` is not a :data:`~threetears.evals.analysis.style.GLOSSARY` key.
        """
        if self.term is not None and self.term not in GLOSSARY:
            raise ValueError(f"a disclosure links the term {self.term!r}, which the glossary does not define")
        return self

    @model_validator(mode="after")
    def _the_tone_is_the_verdicts(self) -> Self:
        """Refuse a tone on any line but the verdict's.

        :raises ValueError: ``tone`` is set and ``source`` is not ``verdict``.
        """
        if self.tone is not None and self.source != "verdict":
            raise ValueError(f"a tone sets the verdict line apart, and this disclosure speaks for {self.source!r}")
        return self


#: A report block, discriminated by ``kind``.
ReportBlock = Annotated[TextBlock | TableBlock | ChartBlock | TrialBlock | DisclosureBlock, Field(discriminator="kind")]


__all__ = [
    "FINDING_ROLES",
    "FIRST_SCREEN",
    "FOLDED_SECTIONS",
    "SECTION_TITLES",
    "VERDICT_OUTCOMES",
    "BarOutcome",
    "CellValue",
    "ChartBlock",
    "ChartKind",
    "ContrastOutcome",
    "CostQualityPoint",
    "DisclosureBlock",
    "DisclosureSource",
    "Fact",
    "GuardrailOutcome",
    "IntervalRow",
    "OutcomeBar",
    "OutcomeSegment",
    "ReportBasis",
    "ReportBlock",
    "ReportSection",
    "TableBlock",
    "TableCell",
    "TableColumn",
    "TextBlock",
    "TextRole",
    "TrialBlock",
    "TrialDetail",
    "Verdict",
    "VerdictKind",
    "VerdictReason",
    "fold_anchor",
    "fold_ids",
    "interval_text",
    "interval_value_text",
    "reading_anchor",
    "table_row",
]

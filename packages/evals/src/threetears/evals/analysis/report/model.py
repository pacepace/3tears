"""The ``Report`` — the one page every door into a campaign's report renders, and the evidence it was computed from.

**One page, result first.** A report is an ordered list of blocks in its sections: the first screen (the title — the
campaign's name —, the question declared quoted, each criterion set after the readings it judges, and the subhead under
it, the arms line, the verdict line, the scoreboard and the caveats under it that change how the verdict is read, in normal weight; every other
qualification is a step down, in the sections), the floors clause by clause where any is declared, what it
means (the memo's findings, each a table code built and one sentence, then the contrasts against the control), the
cases, the trials (the census in a line, then a row and a record per failed or lost trial, the passed ones one closed
table), what was not tested, the methods (a table, then the engine's disclosures folded), and the glossary
of every term of art the page linked. Every count, rate, interval and cost in it is code's, computed from the
evidence bundle the report carries (:attr:`Report.bundle`). Every string it prints is held to
:mod:`~threetears.evals.analysis.style`.

**Who wrote what is part of the shape.** A ``text`` block holds what the analysis's author wrote, and nothing else;
what code says — the verdict, the basis, a table's caption, a reason something is not drawn — is a ``disclosure``.
A code-only report (no analysis was generated) holds no ``text`` block at all, and its model refuses one, so a
reader never takes code's computation for an analyst's judgement, or the reverse.

**One chart, and the intervals drawn inside the tables.** The page's one standalone ``chart`` block is the outcome
bar (one stacked bar per arm: passed, failed and lost by cause, every segment's count written beside it). Every
estimate's uncertainty is drawn where the estimate is read: a :class:`TableCell` holding a number may carry its
interval (``low``, ``high``) and a declared ``margin``, and HTML draws them beside the number as a small glyph on one
scale for the column (:mod:`~threetears.evals.analysis.report.svg`), the figures written in the row as text, so
Markdown loses nothing. A ``chart`` block of interval rows (a dot and its interval per row) remains drawable, carrying
its values so Markdown prints them as a table; the builder draws none, since the tables carry the intervals. A cell's
interval is drawn only for an arm with :data:`~threetears.evals.analysis.report.verdict.MIN_CASES` usable cases
and an interval computed.

**The bundle is the machine form.** There is no published JSON of the report: a program reads ``bundle.json``,
which :meth:`Report.write` puts beside the page, and the page names its fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal, Self, get_args

from pydantic import Field, model_validator

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, TrialCheck, TrialJudgement, TrialState
from threetears.evals.analysis.campaign_readings import CampaignReadings
from threetears.evals.analysis.quantities import PERCENT, number_in, render_cell, unit_label, with_unit
from threetears.evals.analysis.style import GLOSSARY
from threetears.evals.kernel.authored import Recommendation
from threetears.evals.kernel.declaration import CampaignCall
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.hashing import round_floats
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


#: What a report is of: a generated analysis, or the campaign's evidence alone with no analysis.
ReportBasis = Literal["analysis", "code_only"]

#: Where a block sits, in reading order. ``header`` is the first screen's title, arms and verdict lines, and
#: ``scoreboard`` the rest of the first screen; ``decision`` is the floors section: each declared floor read clause by clause;
#: ``memo`` and ``contrasts`` are what it means; ``cases`` and ``trials`` the cases and their drill-down;
#: ``not_tested`` what the campaign could not speak to; ``methods`` how everything was measured; ``glossary`` every
#: term of art the page used. A campaign's page has six more (:attr:`CampaignReport.section_titles`): ``answer``, the
#: question and what it was answered; ``warnings``, what warns, as consequences; ``scorecard``, every aspect's reading
#: in one table; ``findings`` and ``next``, the writer's; and ``provenance``, the design and when each criterion was set.
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

#: Each of a campaign's own sections' heading, three words or fewer; a first screen's carries none
#: (:attr:`CampaignReport.section_titles`).
_CAMPAIGN_SECTION_TITLES: dict[str, str] = {
    "answer": "",
    "warnings": "",
    "scorecard": "Scorecard",
    "findings": "Findings",
    "next": "Next steps",
    "provenance": "Design & provenance",
}

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
#: verdict line. ``design``, ``charts`` and
#: ``cost``: how the run was laid out, what was not charted and where a cost is a floor, in the methods. ``memo``: who wrote the memo,
#: or that no one did. ``runs``: member runs left out, unfinished or short. ``measurement``: how and when the runs
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


def table_row(values: Mapping[str, CellValue]) -> dict[str, TableCell]:
    """A table row from its values, each made a :class:`TableCell`.

    :param values: The row's values, by column key.
    :ptype values: Mapping[str, CellValue]
    :return: The row's cells, by the same keys, in the same order.
    :rtype: dict[str, TableCell]
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
    title_ref: str | None = Field(
        default=None,
        description=(
            "An anchor on the page the title links to, where the table reads a sentence said there; None for none."
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


#: The charts a report draws: an outcome bar, interval rows, and a campaign's cost against its pass rate.
ChartKind = Literal["outcome_bar", "interval_rows", "cost_quality"]


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

        :raises ValueError: ``low`` is above ``high``.
        """
        if self.low > self.high:
            raise ValueError(f"row {self.label!r}: its interval runs from {self.low} down to {self.high}")
        return self


class CostQualityPoint(EvalDocumentModel):
    """One arm on a cost/quality chart: what a reply costs it, on x, and its pass rate, on y, each with its interval."""

    arm: str = Field(min_length=1, description="The arm, as the page names it.")
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
    """One of the charts a report draws, with every value it draws, so it prints as a table too."""

    kind: Literal["chart"] = Field(default="chart", description="Which kind of block this is.")
    chart: ChartKind = Field(description="`outcome_bar`, `interval_rows` or `cost_quality`.")
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

    @model_validator(mode="after")
    def _draws_what_its_kind_draws(self) -> Self:
        """Refuse a chart with nothing to draw, or one carrying another kind's marks.

        :raises ValueError: An outcome bar with rows, points or no bar; an interval chart with bars, points or no
            row; or a cost/quality chart with bars, rows or fewer than two points, since a chart is never drawn for
            one point.
        """
        if self.chart == "outcome_bar" and (self.rows or self.points or not self.bars):
            raise ValueError("an outcome bar draws one bar per arm and no interval row")
        if self.chart == "interval_rows" and (self.bars or self.points or not self.rows):
            raise ValueError("an interval chart draws at least one row and no outcome bar")
        if self.chart == "cost_quality" and (self.bars or self.rows or len(self.points) < 2):
            raise ValueError("a cost/quality chart draws one point per arm, for two arms or more, and nothing else")
        return self

    def values_table(self) -> tuple[list[str], list[list[str]]]:
        """The chart's values as a table — what Markdown prints and what HTML keeps beside the picture.

        :return: ``(headers, rows)``, each row's cells as text.
        :rtype: tuple[list[str], list[list[str]]]
        """
        if self.chart == "outcome_bar":
            labels = list(dict.fromkeys(segment.label for bar in self.bars for segment in bar.segments))
            headers = ["Arm", "ran", *labels]
            body = []
            for bar in self.bars:
                counts = {segment.label: segment.count for segment in bar.segments}
                body.append([bar.arm, str(bar.total), *(str(counts.get(label, 0)) for label in labels)])
            return headers, body
        if self.chart == "cost_quality":
            cost = f" ({unit_label(self.unit)})" if self.unit else ""
            headers = ["Arm", f"Cost{cost}", f"Cost interval{cost}", "Pass rate (%)", "Interval (%)"]
            return headers, [
                [
                    point.arm,
                    interval_value_text(point.cost, self.unit),
                    interval_text(point.cost_low, point.cost_high, self.unit),
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
    title: str = Field(
        min_length=1,
        description=(
            "The title line, a name: the campaign's, with what it compares where no question was declared "
            "(`invoices: does v2 do better than v1?`)."
        ),
    )
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

        :raises ValueError: A block's ``finding`` or ``rests_on`` names a position past ``finding_count``.
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
    def _every_link_lands(self) -> Self:
        """Refuse one id carried twice, by cells or blocks, and a cell linking to an id nothing on the page carries.

        A finding's trace links a figure only where the page prints the same figure; a finding whose trace reaches
        nothing says so in words instead (:data:`~threetears.evals.analysis.report.words.UNTRACED`).

        :raises ValueError: An id is carried twice, or a link names one nothing carries.
        """
        anchors = [
            *(cell.anchor for cell in _cells(self.blocks) if cell.anchor is not None),
            *(block.anchor for block in self.blocks if block.anchor is not None),
        ]
        if twice := sorted({anchor for anchor in anchors if anchors.count(anchor) > 1}):
            raise ValueError(f"the ids {twice} are carried more than once, and an id names one place on the page")
        places = set(anchors)
        if stray := sorted({cell.ref for cell in _cells(self.blocks) if cell.ref is not None} - places):
            raise ValueError(f"cells link to {stray}, which nothing on the page carries")
        return self

    @model_validator(mode="after")
    def _the_basis_matches_what_the_report_holds(self) -> Self:
        """Refuse a report whose basis disagrees with its source or its blocks.

        :raises ValueError: An analysis report names no analysis or model, or a code-only report names one or holds a
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
    def section_titles(self) -> dict[str, str]:
        """Each section's heading on this page: :data:`SECTION_TITLES`, except that ``contrasts`` is headed
        ``Readings`` where the page contrasts nothing — a one-arm page, whose section holds only the arm's readings
        over all its cases, by stratum and by label — so no heading promises a comparison the page does not make."""
        titles = dict(SECTION_TITLES)
        readings_only = all(
            isinstance(block, TableBlock) and block.name in ("readings", "strata", "labels")
            for block in self.blocks
            if block.section == "contrasts"
        )
        if readings_only:
            titles["contrasts"] = "Readings"
        return titles

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
        file (``AnalysisContextBundle.from_json(...).fingerprint(like=named)``, in the named one's version: an
        unprefixed one was taken before ``v2:``). ``report.md`` is the same page as Markdown.

        :param out_dir: The directory; created when missing.
        :ptype out_dir: Path | str
        :return: Each file written, by role.
        :rtype: dict[ReportFile, Path]
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


#: The shape of a campaign page's JSON (:meth:`CampaignReport.json_document`), raised whenever a reader of the old
#: shape would misread the new one. 3: each aspect's `question_after_runs` is replaced by `late_criteria`. 4: each
#: reading and pair in `readings` states both arms' depth and what limits its interval. 5: the page's model gains a
#: campaign's own sections, a cell's `flag` and the `cost_quality` chart, and the JSON what the page reads across
#: aspects as it lands; one unreleased shape holds them all.
CAMPAIGN_JSON_SHAPE = 5


class CampaignAspectFold(EvalDocumentModel):
    """One aspect's fold of a campaign's page: the writer's summary of the aspect, then that aspect's own page."""

    aspect: str = Field(min_length=1, description="The aspect's label: the fold's heading.")
    shows: str = Field(
        min_length=1,
        description="What code read on the aspect, in a few words beside the heading (`v2 shown worse`).",
    )
    summary: list[ReportBlock] = Field(
        default_factory=list,
        description=(
            "The writer's summary of the aspect (what changed, how sure, what it means), as text blocks; empty "
            "with no memo, or none for this aspect."
        ),
    )
    report: Report = Field(
        description=(
            "The aspect's own page (`build_code_only_report` over its bundle), laid out in the fold below the summary "
            "without its title, its question, its broken floors or its verdict line (:attr:`head`)."
        )
    )

    @property
    def head(self) -> list[ReportBlock]:
        """The aspect page's first screen as the fold lays it out: its arms line, scoreboard and caveats, without its
        question, when it was set and its broken floors, which the campaign's first screen carries once, or its verdict
        line, which reads the aspect alone and could contradict, or pick an arm against, the campaign's table. An aspect
        of one arm keeps its verdict line: it counts that arm's cases passed and missed, and has no other arm to pick
        against."""
        one_arm = len(self.report.bundle.scoreboard) == 1
        dropped = {"floor", "basis", "timing"} if one_arm else {"floor", "basis", "timing", "verdict"}
        return [
            block
            for block in self.report.blocks
            if block.section in FIRST_SCREEN
            and not (isinstance(block, DisclosureBlock) and block.section == "header" and block.source in dropped)
        ]


class CampaignReport(EvalDocumentModel):
    """A campaign's page: the writer's summary and recommendation, what code read across its aspects, the person's
    call, then one fold per aspect.

    **The first screen, in order** (:attr:`blocks`, in the ``header`` and ``scoreboard`` sections): every missed bar,
    broken floor and guardrail an arm is not shown safe on, one loud line each, above the title; the question; the
    writer's summary across every aspect; the writer's recommendation, or a line saying none was written; code's
    table of every aspect × candidate (its change against the control, the interval it was read on, the margin and
    what it shows); the cases that would settle each unclear row; the person's call, or that none is recorded, and
    any broken floor it overrides. :attr:`blocks` may also hold the memo's findings and next steps (``memo``), which
    the recommendation rests on. Then :attr:`aspects`, one fold each, in the campaign's order.

    **Who wrote what is part of the shape**, as on an aspect's page: a ``text`` block holds what the writer wrote and
    nothing else, and a code-only campaign page (no memo) holds none.
    """

    basis: ReportBasis = Field(description="`analysis` with a memo over the campaign, `code_only` without one.")
    title: str = Field(min_length=1, description="The title line: the campaign's name.")
    campaign_id: str = Field(min_length=1, description="The campaign.")
    campaign_name: str | None = Field(default=None, description="The campaign's name, where known.")
    scope_id: str = Field(min_length=1, description="The scope the campaign lives in.")
    analysis_id: str | None = Field(
        default=None, description="The stored analysis whose memo the page lays out; None on a code-only page."
    )
    finding_count: int = Field(ge=0, description="How many findings the memo holds.")
    blocks: list[ReportBlock] = Field(description="The first screen, then the memo's findings, in reading order.")
    aspects: list[CampaignAspectFold] = Field(description="One fold per aspect, in the campaign's order.")
    readings: CampaignReadings = Field(description="What code read off the campaign: every row and every floor.")
    call: CampaignCall | None = Field(default=None, description="The person's call; None while none is recorded.")
    recommendation: Recommendation | None = Field(
        default=None, description="The writer's recommendation, as the memo holds it; None with no memo."
    )
    recommended_variant_key: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The variant key of the arm the recommendation names, as the analysis recorded it when it was saved "
            "(`EvalAnalysis.recommended_variant_key`): what a reader matches a call's `variant_key` to, whatever "
            "later relabels the arm. Set exactly when the page holds a recommendation."
        ),
    )

    @model_validator(mode="after")
    def _the_recommendation_is_keyed(self) -> Self:
        """Refuse a recommendation without its arm's key, and a key with no recommendation to name.

        :raises ValueError: One of ``recommendation`` and ``recommended_variant_key`` is set without the other.
        """
        if (self.recommendation is None) != (self.recommended_variant_key is None):
            raise ValueError(
                "recommended_variant_key is the key of the arm the recommendation names: set exactly when the page "
                "holds a recommendation"
            )
        return self

    @model_validator(mode="after")
    def _every_link_lands_in_a_fold(self) -> Self:
        """Refuse a cell linking anywhere but an aspect's fold, by the id of the section around it
        (``aspect-<fold id>``): a fold is closed, and a link landing inside one is lost where a browser does not open
        it, so a trace lands on the fold's heading.

        :raises ValueError: A link names something other than a fold.
        """
        places = {f"aspect-{fold_id}" for fold_id in self.fold_ids}
        if stray := sorted({cell.ref for cell in _cells(self.blocks) if cell.ref is not None} - places):
            raise ValueError(f"cells link to {stray}, which is no aspect's fold")
        return self

    @model_validator(mode="after")
    def _the_basis_matches_what_the_page_holds(self) -> Self:
        """Refuse a code-only page holding a finding or a writer's words, and a block linked to a finding not held.

        :raises ValueError: A code-only page counts a finding or holds a text block, or a block names a finding
            position past ``finding_count``.
        """
        blocks = [*self.blocks, *(block for fold in self.aspects for block in fold.summary)]
        for index, block in enumerate(blocks):
            named = ([block.finding] if block.finding is not None else []) + list(block.rests_on)
            if stray := sorted({position for position in named if not 0 <= position < self.finding_count}):
                raise ValueError(
                    f"block {index} ({block.kind}) is linked to finding position(s) {stray}, and the page holds "
                    f"{self.finding_count} finding(s)"
                )
        if self.basis == "code_only":
            if analysed := [fold.aspect for fold in self.aspects if fold.report.basis != "code_only"]:
                raise ValueError(
                    f"a code-only page folds code-only aspect pages, and {', '.join(analysed)} renders an analysis"
                )
            if self.analysis_id is not None:
                raise ValueError(f"a code-only page lays out no analysis, and this one names {self.analysis_id}")
            if self.recommendation is not None:
                raise ValueError("a code-only page has no memo, so it holds no recommendation")
            if self.finding_count:
                raise ValueError(f"a code-only page has no findings, and this one counts {self.finding_count}")
            if authored := [index for index, block in enumerate(blocks) if isinstance(block, TextBlock)]:
                raise ValueError(f"a code-only page holds no writer's words, and blocks {authored} are text blocks")
        return self

    @property
    def section_titles(self) -> dict[str, str]:
        """Each section's heading on a campaign's page, in its reading order (:data:`ReportSection`'s): an aspect
        page's (:data:`SECTION_TITLES`) with the campaign's own among them (``answer``, ``warnings``, ``scorecard``,
        ``findings``, ``next`` and ``provenance``), each three words or fewer, the first screen's headed by none."""
        titles = {**SECTION_TITLES, **_CAMPAIGN_SECTION_TITLES}
        return {section: titles[section] for section in get_args(ReportSection)}

    @property
    def terms(self) -> list[str]:
        """Every glossary term the page links, in the order first linked: its own first screen's, then each aspect's
        fold's — what its one glossary defines."""
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
                # Two columns of one table can link one term: it is defined once.
                if term is not None and term not in linked:
                    linked.append(term)
        for fold in self.aspects:
            linked += [term for term in fold.report.terms if term not in linked]
        return linked

    @property
    def verdicts(self) -> list[Verdict]:
        """Every verdict code reaches on the campaign, typed: each aspect's page's, aspect by aspect in the campaign's
        order, each naming its aspect — what a gate reads."""
        return [verdict for fold in self.aspects for verdict in fold.report.verdicts]

    def write(self, out_dir: Path | str) -> dict[str, Path]:
        """Write the page as one self-contained HTML file, its Markdown, its JSON, and each aspect's bundle.

        ``report.html`` needs nothing beside it; ``report.md`` is the same page as Markdown; ``campaign.json`` is the
        page's machine form (:meth:`json_document`), naming each aspect's bundle by its fingerprint; and each aspect's
        bundle is written whole as ``bundle-<fold id>.json`` (:attr:`fold_ids`), so the fingerprint the page names is
        the one a reader computes from the file.

        :param out_dir: The directory; created when missing.
        :ptype out_dir: Path | str
        :return: Each file written, by role: ``html``, ``markdown``, ``json``, and ``bundle:<aspect>`` per aspect.
        :rtype: dict[str, Path]
        """
        # Deferred: the serializers import this module for the block types they lay out.
        from threetears.evals.analysis.report.serialize_html import campaign_report_html
        from threetears.evals.analysis.report.serialize_md import campaign_report_markdown

        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        files: dict[str, Path] = {
            "html": directory / "report.html",
            "markdown": directory / "report.md",
            "json": directory / "campaign.json",
        }
        files["html"].write_text(campaign_report_html(self), encoding="utf-8")
        files["markdown"].write_text(campaign_report_markdown(self), encoding="utf-8")
        files["json"].write_text(json.dumps(self.json_document(), indent=2) + "\n", encoding="utf-8")
        for fold, fold_id in zip(self.aspects, self.fold_ids, strict=True):
            path = directory / f"bundle-{fold_id}.json"
            path.write_text(fold.report.bundle.to_json(indent=2) + "\n", encoding="utf-8")
            files[f"bundle:{fold.aspect}"] = path
        return files

    @property
    def fold_ids(self) -> list[str]:
        """Each aspect's fold as the id its section and every id inside it are named by: the label in lower case,
        each run of other characters one hyphen, numbered where two would read alike."""
        return fold_ids([fold.aspect for fold in self.aspects])

    def json_document(self) -> dict[str, object]:
        """The page's machine form, tagged ``kind: campaign`` at shape :data:`CAMPAIGN_JSON_SHAPE`: code's readings,
        the writer's recommendation (its arm's label and variant key, and how sure) where there is a memo, the call,
        and each aspect's bundle by its fingerprint and shape, with each criterion set after its readings were visible
        and when (``late_criteria``, empty where none was), since its readings read the same either way.

        Every float in it is served to twelve significant figures (:func:`~threetears.evals.schema.hashing.round_floats`):
        a figure code computed carries bits past that which two Python builds compute differently, so one campaign
        serves one JSON across builds, except, rarely, a value that falls exactly on a 12th-figure rounding boundary; a declared value (a
        margin, a bar, alpha) has fewer figures and is unchanged.
        The page's models keep their figures as computed.

        :return: A JSON-ready mapping.
        :rtype: dict[str, object]
        """
        recommendation = self.recommendation
        document: dict[str, object] = {
            "kind": "campaign",
            "shape": CAMPAIGN_JSON_SHAPE,
            "campaign_id": self.campaign_id,
            "basis": self.basis,
            "analysis_id": self.analysis_id,
            "recommendation": None
            if recommendation is None
            else {
                "arm": recommendation.arm,
                "variant_key": self.recommended_variant_key,
                "confidence": recommendation.confidence,
            },
            "readings": self.readings.to_dict(),
            "call": self.call.to_dict() if self.call is not None else None,
            "aspects": [
                {
                    "aspect": fold.aspect,
                    "bundle_fingerprint": fold.report.source.bundle_fingerprint,
                    "bundle_schema_version": fold.report.bundle.schema_version,
                    "late_criteria": [late.to_dict() for late in fold.report.bundle.reading_scope.late_criteria],
                }
                for fold in self.aspects
            ],
        }
        served: dict[str, object] = round_floats(document)
        return served


def _cells(blocks: Sequence[ReportBlock]) -> list[TableCell]:
    """Every cell of every table among ``blocks``, in reading order."""
    return [cell for block in blocks if isinstance(block, TableBlock) for row in block.rows for cell in row.values()]


__all__ = [
    "CAMPAIGN_JSON_SHAPE",
    "FINDING_ROLES",
    "FIRST_SCREEN",
    "FOLDED_SECTIONS",
    "SECTION_TITLES",
    "VERDICT_OUTCOMES",
    "BarOutcome",
    "CampaignAspectFold",
    "CampaignReport",
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
    "fold_ids",
    "interval_text",
    "interval_value_text",
    "reading_anchor",
    "table_row",
]

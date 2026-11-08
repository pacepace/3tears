"""The blocks a report is made of — tables, charts, trials and disclosures — and the vocabulary they share.

The pieces a :class:`~threetears.evals.analysis.report.model.Report` and a
:class:`~threetears.evals.analysis.report.model.CampaignReport` hold, defined below both the documents and the
serializers that lay them out (:mod:`~threetears.evals.analysis.report.serialize_html`,
:mod:`~threetears.evals.analysis.report.serialize_md`), so a document can write itself through a serializer imported
at module scope with no cycle. :mod:`~threetears.evals.analysis.report.model` re-exports every name here; import
from there.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Annotated, Any, Literal, Self, get_args

from pydantic import Field, JsonValue, model_validator

from threetears.evals.analysis.bundle.schema import TrialCheck, TrialJudgement, TrialState
from threetears.evals.analysis.quantities import PERCENT, is_money, number_in, render_cell, unit_label, with_unit
from threetears.evals.analysis.report.charts import (
    KIT_CHARTS,
    bare_at,
    bare_places,
    percent_at,
    percent_places,
    respell_bare,
    respell_percents,
    BarRow,
    BarSegment,
    BarsChart,
    ChartPattern,
    ChartStatus,
    KitChart,
    LegendItem,
    ScatterChart,
    ScatterPoint,
    Token,
)
from threetears.evals.analysis.report.cost_unit import REPLIES, money_ruler, per_count
from threetears.evals.analysis.report.parts.look import arm_shape, arm_token, cost_axis_format
from threetears.evals.kernel.numbers import MoneyForm, money_form
from threetears.evals.analysis.style import GLOSSARY
from threetears.evals.schema.base import EvalDocumentModel

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

#: An aspect fold's section headings where its campaign's page heads a section of the same name at its foot, so no
#: heading prints twice on one page: the fold's methods are the aspect's.
FOLD_SECTION_TITLES: dict[str, str] = {"methods": "Aspect methods"}

#: The sections the first screen is made of: a campaign's answer and warnings are on it too.
FIRST_SCREEN: frozenset[str] = frozenset({"header", "answer", "warnings", "scoreboard"})

#: The report-wide caveat a page sets apart as its one flag, by the id it carries: criteria set after their results
#: existed, which changes how every reading is read. Every other report-wide caveat is a plain line under it.
FLAGGED_CAVEATS: frozenset[str] = frozenset({"caveat-late"})


#: Who a disclosure speaks for. ``arms``: the first screen's arms line, each arm named in full once by the version
#: that answered it, the control marked, and the design (``30 cases × 3 repeats``). ``caveat``: a line under the
#: scoreboard that changes how the verdict is read, in normal weight: the case floor, an arm's label that its own row
#: contradicts. ``basis``: the question declared, quoted whole under the title. ``timing``: under it, each criterion
#: set after the readings it judges were visible, with when, or that its time is not recorded. ``floor``: on the first
#: screen, a broken floor's warning, the page's first line; in the Decision section, the lines under the floors table
#: saying a floor warns and never decides, and how the cases that would settle one were counted. ``verdict``: the
#: verdict line. ``design``, ``charts`` and ``cost``: how the run was laid out, what was not charted and where a cost
#: is a floor, in the methods; ``cost`` is also the detail saying each cost figure is its runs' provider routing's. ``memo``: where the campaign's report is, where it has one. ``runs``: member runs left out, unfinished or short. ``measurement``: how and when the runs
#: were launched and measured. ``apparatus``: the test setup — controls, confounds, cells that did not pool, a judge
#: grading its own model's answers. ``comparisons``: how the contrasts against the control were tested and
#: corrected. ``strata``: a stratum too small to read alone. ``guardrails``: how guardrails were decided.
#: ``scope``: which readings no declared question asked about. ``not_tested``: something the campaign could not
#: speak to. ``calibration``: a judge not yet calibrated against people's labels — a state with its next step, never a
#: failure. ``generation``: how the report itself was made. ``call``: on a campaign's page, the person's call, once
#: one is recorded, and every broken floor it overrides. ``bar``: on a campaign's page, a bar an arm misses, a
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
    guardrails table's ``decision`` cell are each a verdict's :attr:`words`, rendered from it, never the reverse; an
    undecided contrast's cell in its short form (:data:`~threetears.evals.analysis.style.CANNOT_TELL_APART`), since
    its row names both arms.
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


class _Block(EvalDocumentModel):
    """What every block carries: where it sits, the id a link names, and the fold it sits in."""

    section: ReportSection = Field(description="The section the block sits in.")
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
    decimals: int | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "For a money number, the decimals its column's one form writes it to (#993), chosen from every amount the "
            "column holds; a negative count rounds into the dollars (`-1` writes a rate of $123 as `$120`); for a "
            "percentage, the decimals its column reads at (0, or 1 where whole percents would print two of its values "
            "alike); None for none, and the number rule spells the number."
        ),
    )
    per_thousand: bool = Field(
        default=False,
        exclude_if=lambda value: not value,
        description="For a money number with `decimals`, whether its column reads per 1,000 (`$8.00/1000`).",
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

    def _money_form(self) -> MoneyForm | None:
        """The one form a money number in this cell is written in, its column's (``decimals``, ``per_thousand``);
        None where the cell holds no money number, or no form was given, so the number rule spells it."""
        if self.decimals is None or not self.is_number or not is_money(self.unit):
            return None
        return MoneyForm(per_thousand=self.per_thousand, decimals=self.decimals)

    def amount(self, number: float) -> str:
        """A number of this cell's (its value, or an end of its interval) as the cell writes it: in its money form where
        it holds one, else in its unit by the number rule.

        :param number: The number, in the cell's unit.
        :ptype number: float
        :return: Its text.
        :rtype: str
        """
        form = self._money_form()
        if form is not None:
            return form.spell(number)
        if self.decimals is not None and self.is_number and self.unit == PERCENT:
            return percent_at(number, self.decimals)
        if self.decimals is not None and self.is_number and not self.unit:
            # A bare figure under a (%) or (pts) head, at its column's one precision.
            return bare_at(number, self.decimals)
        return with_unit(number, self.unit)

    @property
    def is_number(self) -> bool:
        """Whether the cell holds a number (a boolean is not one)."""
        return isinstance(self.value, int | float) and not isinstance(self.value, bool)

    @property
    def text(self) -> str:
        """``value`` as every serializer prints it, its ``flag`` after it in parentheses; the interval, where the cell
        has one, is the glyph's and the row's own interval column's, never spelled here twice."""
        if (self.unit or self.decimals is not None) and self.is_number and isinstance(self.value, int | float):
            said = self.amount(float(self.value))
        else:
            said = render_cell(self.value)
        return said if self.flag is None else f"{said} ({self.flag})"


def money_columns(
    rows: Sequence[Mapping[str, TableCell]],
    *,
    by_row: Sequence[str | None] | None = None,
    reconciles: Collection[str] = (),
) -> list[dict[str, TableCell]]:
    """A table's rows with each set of its money in one form (#993). A set is every amount of one figure type the
    table shows, across its rows and columns: by default a column's (a table of arms by measure); where ``by_row``
    names a row's figure type (a table of slices by arm, each row one measure across the arms), every money cell of
    the rows naming the same one. Every money number a set holds, and the ends of its intervals, choose the set's
    :func:`~threetears.evals.kernel.numbers.money_form` from the values themselves, never from one cell, and every
    such cell is written in it (:meth:`TableCell.amount`): plain dollars to two significant figures, every set a rate
    (money per the count the table's head names), in whole dollars from $10. Each cell writes its value and its interval's ends at one precision, the
    finest of them needs, and one significant figure more where its own interval's ends would otherwise read alike
    (:meth:`~threetears.evals.kernel.numbers.MoneyForm.places_of`), so $0.008 reads ``$0.0080`` beside ``$0.11``, never
    ``$0.1100``. A table states the count its money is drawn per in its
    head or its row, never as a mark on the amount. The values are as stored, so the decimals state no digit a value
    does not hold; a cell of words, of another unit, or already carrying a form its builder chose from a wider set, is
    left as it is.

    :param rows: The table's rows, by column key.
    :ptype rows: Sequence[Mapping[str, TableCell]]
    :param by_row: Each row's figure type, where the table lays one per row (``None`` for a row whose money is set by
        its column); unset for a table whose every set is a column.
    :ptype by_row: Sequence[str | None] | None
    :param reconciles: The sets a total computed from each amount is printed beside (a cost per 1,000 at a declared
        monthly volume), whose amounts take a third figure where two would leave them off that total
        (:class:`~threetears.evals.kernel.numbers.MoneyForm`).
    :ptype reconciles: Collection[str]
    :return: The rows, each money cell carrying its set's form.
    :rtype: list[dict[str, TableCell]]
    """

    def chosen(cell: TableCell) -> bool:
        return cell.is_number and is_money(cell.unit) and cell.decimals is None

    def set_of(at: int, key: str) -> tuple[str, str]:
        kind = by_row[at] if by_row is not None else None
        return ("row", kind) if kind is not None else ("column", key)

    held: dict[tuple[str, str], list[float]] = {}
    for at, row in enumerate(rows):
        for key, cell in row.items():
            if chosen(cell):
                held.setdefault(set_of(at, key), []).extend(
                    float(number)
                    for number in (cell.value, cell.low, cell.high)
                    if isinstance(number, int | float) and not isinstance(number, bool)
                )
    forms = {name: money_form(amounts, rate=True, reconciles=name[1] in reconciles) for name, amounts in held.items()}
    return [
        {
            key: cell.model_copy(
                update={"decimals": forms[set_of(at, key)].places_of(_cell_amounts(cell), ends=(cell.low, cell.high))}
            )
            if chosen(cell)
            else cell
            for key, cell in row.items()
        }
        for at, row in enumerate(rows)
    ]


def _cell_amounts(cell: TableCell) -> list[float | None]:
    """A money cell's amounts: its value and its interval's ends."""
    value = cell.value
    point = float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
    return [point, cell.low, cell.high]


#: The ends of a column head that names its cells' unit, so the cells state bare figures (``Pass rate (%)``,
#: ``Change (pts)``): :func:`percent_columns` reads them by its rule.
BARE_UNIT_HEADS = ("(%)", "(pts)")


def bare_columns(columns: Sequence[TableColumn]) -> dict[str, str]:
    """The columns whose head names their cells' unit (:data:`BARE_UNIT_HEADS`), each by the unit it names.

    :param columns: A table's columns.
    :ptype columns: Sequence[TableColumn]
    :return: Each such column's key, by its unit's head.
    :rtype: dict[str, str]
    """
    return {column.key: unit for column in columns for unit in BARE_UNIT_HEADS if column.header.endswith(unit)}


def percent_columns(
    rows: Sequence[Mapping[str, TableCell]], *, bare: Mapping[str, str] | None = None
) -> list[dict[str, TableCell]]:
    """A table's rows with each column's percentages at one precision: whole percents (``60%``, ``63%``), and one
    decimal for the whole column only where whole percents would print two of its values alike or an interval's two
    ends alike (``62.9%`` beside ``63.2%``); an exact zero reads ``0%`` in every column, so ``0.0%`` and ``0%`` never
    stand on one page (coordinator ruling under the owner's "do what makes sense for usability", 2026-11-07;
    :func:`~threetears.evals.analysis.report.charts.percent_places`).

    A column holding a percentage written to two decimals or more (an interval's confidence level, ``99.69%``) is a
    level, not a rate, and keeps its digits. A cell that states no percentage, or a number in another unit, is left as
    it is; a cell's other words stay as written.

    :param rows: The table's rows, by column key.
    :ptype rows: Sequence[Mapping[str, TableCell]]
    :param bare: The columns whose head names their unit, ``(%)`` or ``(pts)``, so their cells state bare figures
        (``21.1``, ``[17.2, 25]``), each by the unit it names: every column of one unit in the table reads at one
        precision by the same rule (a figure type across its columns: points, or percent), its leading figure and its
        interval's ends. None for none.
    :ptype bare: Mapping[str, str] | None
    :return: The rows, each column's percentages respelled at its one precision.
    :rtype: list[dict[str, TableCell]]
    """
    bare = bare or {}
    keys = list(dict.fromkeys(key for row in rows for key in row))
    places = {
        key: chosen
        for key in keys
        if key not in bare
        if (chosen := percent_places(_said(row[key]) for row in rows if key in row)) is not None
    }
    by_unit = {
        unit: bare_places(_said(row[key]) for row in rows for key in bare if bare[key] == unit and key in row)
        for unit in set(bare.values())
    }
    figures = {key: chosen for key, unit in bare.items() if (chosen := by_unit[unit]) is not None}
    return [
        {
            key: _respelled_percents(cell, places[key])
            if key in places
            else _respelled_bare(cell, figures[key])
            if key in figures
            else cell
            for key, cell in row.items()
        }
        for row in rows
    ]


def _respelled_bare(cell: TableCell, places: int) -> TableCell:
    """A bare figure's cell at ``places`` decimals (:func:`respell_bare`); a number keeps its value and glyph."""
    if cell.is_number and not cell.unit:
        return cell.model_copy(update={"decimals": places})
    text = _said(cell)
    respelled = respell_bare(text, places)
    return cell if respelled == text else cell.model_copy(update={"value": respelled})


def _said(cell: TableCell) -> str:
    """A cell's text as it prints, without its flag: its number in its unit, or its value."""
    return (
        cell.amount(float(cell.value)) if cell.unit and isinstance(cell.value, int | float) else render_cell(cell.value)
    )


def _respelled_percents(cell: TableCell, places: int) -> TableCell:
    """A cell with every percentage it states at ``places`` decimals (:func:`respell_percents`); a number held with
    the percent unit keeps its value, glyph and status and reads at its column's precision."""
    if cell.is_number and cell.unit == PERCENT:
        return cell.model_copy(update={"decimals": places})
    text = _said(cell)
    respelled = respell_percents(text, places)
    return cell if respelled == text else cell.model_copy(update={"value": respelled})


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
    key_columns: Literal[1, 2] = Field(
        default=1,
        description=(
            "How many leading columns a row is told apart by (2 where one column alone repeats, as a reading's row "
            "on each arm): HTML keeps that many in view while the table scrolls sideways."
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
#: state's order. Never a series colour, since series colours name arms (GENERATOR.md §5.2): passed is ink (the kit's
#: secondary text), solid, so the status green means a status pill alone; every failure cause is the kit's bad, each
#: under a texture of its own, so a failure never reads as a pass even without colour, and the five failure causes
#: (``FailCause``) are five looks; a loss cause is a neutral, solid and then textured, so a loss reads as neither. Loss causes past the twelfth repeat a look; the
#: legend and the tooltip name every cause in words.
_OUTCOME_STYLES: dict[TrialState, tuple[tuple[str, ChartPattern | None], ...]] = {
    "passed": (("--ink-2", None),),
    "failed": tuple(("--bad", pattern) for pattern in _PATTERNS),
    "lost": (
        ("--base", None),
        ("--muted", None),
        *((token, pattern) for pattern in _PATTERNS for token in ("--base", "--muted")),
    ),
}


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
        description="The unit an interval chart's values, or a cost/quality chart's costs, are stated in; for a kit "
        "chart of what an analysis computes (two arms' difference, a lever's effect), the unit its values table spells "
        "its values at where its format names none (a point to one decimal); empty for none.",
    )
    zero_line: bool = Field(default=False, description="Whether the chart draws a line at zero (a difference's).")
    margin: float | None = Field(
        default=None, description="A margin band either side of zero, on the chart's ruler; None when none declared."
    )
    level: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
        description=(
            "The level a cost/quality chart's intervals are read at (`95%`), which its scatter draws them at; None "
            "where it states none, and its intervals are in its table alone."
        ),
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
        """The chart as the eval report kit draws it: its kind and its data, as computed.

        A page serves the data in ``data-chart`` with every float to twelve significant figures (``chart_html``), and the
        static page's SVG rounds its coordinates the same way, so this is the one form both are drawn from.

        A kit kind is its data. An outcome bar is ``bars``, each arm's census stacked from zero, a look per outcome
        named in the legend, each arm's tag in its colour beside its label. A cost/quality chart is
        ``scatter_frontier``, each arm in its colour and mark at its cost per 1,000 replies across and its pass rate
        up, on axes set to the data, each interval a whisker where the chart states its level. Interval rows are ``None``: they carry no
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
                color=arm_token(bar.tag),
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

    def _cost_drawn(self) -> tuple[Callable[[float], str], float, str, int | None]:
        """How a cost/quality chart draws its costs: the speller of one cost as stored, the scale it is drawn at, and
        the words of the count it is drawn per. Money is on the one owner's ruler over every point's cost and its
        interval's ends (:func:`~threetears.evals.analysis.report.cost_unit.money_ruler`, #993): per 1,000 replies, or
        per 1,000,000 where a cost is under a cent per 1,000, every amount in the chart's one form; any other unit per
        1,000 replies by the number rule. The last is money's decimals, which the axis's ticks are written to; None for
        any other unit."""
        if not is_money(self.unit):
            return (
                lambda amount: with_unit(amount * _REPLIES_PER_COST_UNIT, self.unit),
                _REPLIES_PER_COST_UNIT,
                per_count(REPLIES),
                None,
            )
        ruler = money_ruler(
            (amount for point in self.points for amount in (point.cost, point.cost_low, point.cost_high)),
            per=1,
            start=_REPLIES_PER_COST_UNIT,
        )
        return ruler.spell, ruler.scale, ruler.words(REPLIES), ruler.form.decimals

    def _cost_quality_scatter(self) -> ScatterChart:
        """A cost/quality chart as the kit's scatter: each arm a point in its colour and mark at its cost per 1,000
        replies and its pass rate, with both intervals as whiskers where the chart states their level, labelled by its
        short name (by its tag on a phone), the legend naming each tag's arm. Each axis is set to the data: from a
        little below the lowest interval end to a little above the highest, the cost never below zero and the pass
        rate within 0 to 100%, so arms that sit close are drawn apart and their intervals show whether they overlap
        (the review's P46); a scatter reads positions, never lengths from a baseline. Cost is drawn per 1,000 replies,
        since a reply's own cost is a fraction of a cent an axis cannot print, its ticks in the chart's one money form.
        No frontier is drawn, since none is tested. Money's ruler is the one owner's over every point's cost and
        interval (:func:`~threetears.evals.analysis.report.cost_unit.money_ruler`, #993): per 1,000 replies, or per
        1,000,000 where a cost is under a cent per 1,000, each tip's cost and its interval in the chart's one form."""
        cost, scale, per, decimals = self._cost_drawn()
        x_domain = _data_domain(
            [end * scale for point in self.points for end in (point.cost_low, point.cost_high, point.cost)], floor=0.0
        )
        y_domain = _data_domain(
            [end for point in self.points for end in (point.pass_rate_low, point.pass_rate_high)],
            floor=0.0,
            ceiling=100.0,
        )
        # Its ticks carry the dollar sign, so the title names the count alone; any other unit is named in it.
        unit = f" ({self.unit})" if self.unit and not is_money(self.unit) else ""
        tagged = all(point.tag is not None for point in self.points)
        drawn = self.level is not None
        points = [
            ScatterPoint(
                x=point.cost * scale,
                y=point.pass_rate,
                x_lo=point.cost_low * scale if drawn else None,
                x_hi=point.cost_high * scale if drawn else None,
                y_lo=point.pass_rate_low if drawn else None,
                y_hi=point.pass_rate_high if drawn else None,
                label=point.short_name or point.arm,
                narrow_label=point.tag or point.short_name or point.arm,
                color=arm_token(point.tag),
                shape=arm_shape(point.tag),
                tip=[
                    point.arm,
                    f"{cost(point.cost)} [{cost(point.cost_low)}, {cost(point.cost_high)}] {per}",
                    f"pass rate {with_unit(point.pass_rate, PERCENT)} "
                    f"[{with_unit(point.pass_rate_low, PERCENT)}, {with_unit(point.pass_rate_high, PERCENT)}]",
                ],
            )
            for point in self.points
        ]
        legend = (
            [
                LegendItem(label=f"{point.tag} · {point.arm}", color=arm_token(point.tag), mark=arm_shape(point.tag))
                for point in self.points
            ]
            if tagged
            else None
        )
        return ScatterChart(
            points=points,
            x_domain=x_domain,
            y_domain=y_domain,
            x_fmt=f"usd:{decimals}" if decimals is not None else cost_axis_format(x_domain[1], money=False),
            y_fmt="pct",
            x_title=f"Cost {per}{unit}",
            y_title="Pass rate",
            level=self.level,
            legend=legend,
        )

    def values_table(self) -> tuple[list[str], list[list[str]]]:
        """The chart's values as a table — what Markdown prints and what HTML keeps beside the picture, a column whose
        head names its unit (``Pass rate (%)``) read at one precision with every column of its unit
        (:func:`percent_columns`'s rule).

        :return: ``(headers, rows)``, each row's cells as text.
        :rtype: tuple[list[str], list[list[str]]]
        """
        headers, body = self._values_table()
        units = {at: unit for at, header in enumerate(headers) for unit in BARE_UNIT_HEADS if header.endswith(unit)}
        for unit in set(units.values()):
            columns = [at for at, named in units.items() if named == unit]
            places = bare_places(row[at] for row in body for at in columns if at < len(row))
            if places is None:
                continue
            for row in body:
                for at in columns:
                    if at < len(row):
                        row[at] = respell_bare(row[at], places)
        return headers, body

    def _values_table(self) -> tuple[list[str], list[list[str]]]:
        if self.data is not None:
            return self.data.values_table(self.unit)
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
            # Money carries its sign on every cell, as the scoreboard writes it ($1.89), its count in the head; any
            # other unit is named once in the head.
            money = is_money(self.unit)
            cost = f" ({self.unit})" if self.unit and not money else ""
            spelled, scale, per, _ = self._cost_drawn()

            def said(amount: float) -> str:
                return spelled(amount) if money else number_in(amount * scale, self.unit)

            headers = ["Arm", f"Cost {per}{cost}", f"Interval{cost}", "Pass rate (%)", "Interval (%)"]
            return headers, [
                [
                    point.arm,
                    said(point.cost),
                    f"[{said(point.cost_low)}, {said(point.cost_high)}]",
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

#: How far past the data's ends a scatter's axis runs, as a share of the data's span, so no mark sits on its edge.
_DATA_PAD = 0.08


def _data_domain(ends: Sequence[float], *, floor: float, ceiling: float | None = None) -> tuple[float, float]:
    """A scatter axis set to its data: from a little below the lowest end to a little above the highest, never below
    ``floor`` or above ``ceiling``; a span of one value alone padded either side of it."""
    low, high = min(ends), max(ends)
    pad = (high - low) * _DATA_PAD or abs(high) * _DATA_PAD or 1.0
    low, high = max(floor, low - pad), high + pad
    return (low, min(high, ceiling) if ceiling is not None else high)


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
    first_input: str = Field(default="", description="What the case asked, without its world state.")
    world: str = Field(
        default="",
        description=(
            "The world state the case began in, as its host wrote it (JSON, one part under each `## <name>`), "
            "which a page folds under the case's name and never prints open; empty where the case states none."
        ),
    )
    world_parts: list[str] = Field(
        default_factory=list, description="The names of the world state's parts, in its order: what a page counts."
    )
    excerpt: str = Field(default="", description="The tail of what the judge was handed.")
    failed_checks: list[TrialCheck] = Field(default_factory=list, description="Each check it failed.")
    judge: list[TrialJudgement] = Field(default_factory=list, description="What the judge said, per dimension.")
    error_words: str | None = Field(default=None, description="The recorded error, for a lost or failed trial.")
    result_id: str = Field(min_length=1, description="The stored result's id.")
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
    emphasis: list[str] = Field(
        default_factory=list,
        description=(
            "The phrases of `text` a page sets in bold, in the order `text` says them, none empty and none cutting a "
            "code span: on the verdict line, each decision, its figures and guardrails left regular. Empty for a line "
            "read at one weight; a caller's lead with none renders without bold."
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

    def emphasis_pieces(self) -> list[tuple[str, bool]]:
        """``text`` cut where each phrase of ``emphasis`` starts and ends, in order, each piece with whether it is set
        in bold: the raw text, so a page escapes each piece itself and never searches its own markup.

        :return: The pieces, which join back to ``text``.
        :rtype: list[tuple[str, bool]]
        :raises ValueError: a phrase is empty, is not said after the one before it, or starts or ends inside a code
            span (between backticks), where bold would cut the span in two.
        """
        pieces: list[tuple[str, bool]] = []
        at = 0
        for phrase in self.emphasis:
            found = self.text.find(phrase, at) if phrase else -1
            if found < 0:
                raise ValueError(f"a disclosure sets {phrase!r} in bold, which its text does not say there")
            end = found + len(phrase)
            if self.text.count("`", 0, found) % 2 or self.text.count("`", 0, end) % 2:
                raise ValueError(f"a disclosure sets {phrase!r} in bold, which cuts a code span in two")
            pieces += [(self.text[at:found], False), (phrase, True)]
            at = end
        return [piece for piece in [*pieces, (self.text[at:], False)] if piece[0]]

    @model_validator(mode="after")
    def _each_emphasis_is_said_in_order(self) -> Self:
        """Refuse a phrase to set in bold that the line does not say, says out of order, or says cutting a code span.

        :raises ValueError: :meth:`emphasis_pieces` refuses the phrases.
        """
        self.emphasis_pieces()
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
ReportBlock = Annotated[TableBlock | ChartBlock | TrialBlock | DisclosureBlock, Field(discriminator="kind")]


__all__ = [
    "FIRST_SCREEN",
    "FLAGGED_CAVEATS",
    "FOLD_SECTION_TITLES",
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
    "money_columns",
    "table_row",
]

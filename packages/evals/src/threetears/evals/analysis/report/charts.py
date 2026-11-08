"""The eval report kit's chart vocabulary as models: one per kind, each carrying every value its chart draws.

Each model mirrors the kit's ``data-chart`` contract (``docs/eval-report-kit/kit/API.md``, "Chart data") field for
field: the same names, the same nesting, the same required fields, so :meth:`kit_data` is the JSON the kit's
``kit.js`` draws, and a chart a reference page draws validates as its model and dumps back to the same JSON. A field
the contract leaves optional is ``None`` until set and is left out of the JSON, so an explicit ``false`` the producer
wrote stays in it.

What the kit refuses, the model refuses first, where the producer can still be told: a missing required field, a
colour that is not a token name (``--s1``), a format the kit does not name, a row of intervals without a ``level``
(GENERATOR.md §5.2), a matrix row of the wrong width. Every string is text: the kit sets it as text or escapes it, so
a label carrying markup draws as the markup's words.

Small multiples are ``panels``: each panel is the chart's own fields overlaid with the panel's, as the kit draws
them, and a panel of nothing but text fields (``title``, ``subtitle``, ``tag``, ``color``) is a text card. A chart's
required fields are checked on each panel as drawn.

Every chart reads without the picture too: :meth:`values_table` is ``(headers, rows)`` of text, what Markdown prints
and what HTML keeps beside the chart for a screen reader and a page whose script did not run.
"""

from __future__ import annotations

import decimal
import math
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import ConfigDict, Field, JsonValue, SerializerFunctionWrapHandler, model_serializer, model_validator

from threetears.evals.kernel.numbers import (
    MONEY_FIGURES,
    POINT_PLACES,
    WHOLE_FROM,
    MoneyForm,
    ends_places,
    format_number,
    group_thousands,
    spell_at,
)
from threetears.evals.analysis.quantities import (
    MONEY,
    PERCENT,
    POINTS,
    UNSPACED_UNITS,
    number_in,
    signed_with_unit,
    with_unit,
)
from threetears.evals.schema.base import EvalDocumentModel

#: A colour as the kit takes one: the name of one of its tokens, never a literal colour.
Token = Annotated[str, Field(pattern=r"^--[a-z][a-z0-9-]*$", description='A kit colour token, e.g. "--s1".')]

#: A format as the kit names one: ``name`` or ``name:digits``, with a leading ``+`` to sign positive values.
Fmt = Annotated[
    str,
    Field(
        pattern=r"^\+?(num|int|pct|pts|usd|usdk|usdpk|ms|s|dur|x|k)(:\d)?$",
        description=(
            "A kit format: num, int, pct, pts, usd, usdk, usdpk (dollars per 1,000, the mark written: `$8.00/1000`), "
            "ms, s, dur, x or k, as `name` or `name:digits`."
        ),
    ),
]

#: A mark's shape, so an arm reads apart without colour.
ChartShape = Literal["circle", "square", "triangle", "diamond"]

#: A status as the kit's pill draws it: an icon and the producer's word, never a colour alone.
ChartStatus = Literal["good", "warn", "bad", "neutral"]

#: A fill's texture, drawn in its colour over a wash of it, so marks that share a colour (one status, several causes)
#: read apart by more than colour.
ChartPattern = Literal["hatch", "cross", "dots", "rows", "cols"]

#: How a change chart's ``better`` field is described: ``"right"`` prints "← worse · better →" under the axis, ``"left"``
#: its mirror, and two words print each under its own end, the left's first ("prefers shipped", "prefers lead-first").
_BETTER_WORDS = (
    'Which way is better, printed under the axis: "right" prints "← worse · better →", "left" its mirror; two words '
    "print each under its own end, the left end's first."
)

#: The panel fields that are text: a panel holding only these is drawn as a text card.
_PANEL_TEXT = frozenset({"title", "subtitle", "tag", "color"})

#: The chart fields a panel never overrides: the legend and the grid are the chart's.
_CHART_ONLY = frozenset({"legend", "panels", "panel_min_rem"})


#: The unit each of the kit's named formats states its values in, as the package's number rule names units: a chart's
#: table spells its values by that one rule (:func:`~threetears.evals.analysis.quantities.with_unit`), as every other
#: table of the page does, not by the kit's axis rounding.
_FORMAT_UNITS: dict[str, str] = {
    "num": "",
    "int": "",
    "k": "",
    "pct": PERCENT,
    "pts": POINTS,
    "usd": MONEY,
    "usdk": MONEY,
    "usdpk": MONEY,
    "ms": "ms",
    "dur": "ms",
    "s": "s",
    "x": "×",
}

#: The kit's format for money per 1,000 with its mark (``$8.00/1000``): the one a chart of totals under a cent reads in.
_PER_THOUSAND_FORMAT = "usdpk"

#: A format spec's parts: the sign flag, the name and the digits. The digits are the axis's, not the table's, but for
#: money's: a chart chooses money's one form from every amount it draws (#993), and its table writes it too.
_FORMAT_SPEC = re.compile(r"^(\+?)([a-z]+)(?::(\d))?$")


def _formatted(spec: str, value: float, read_at: str = "", *, finer: bool = False, places: int | None = None) -> str:
    """``value`` as a chart's table prints it: in the unit the kit's named format ``spec`` states it in, by the
    package's one number rule, signed where the spec signs it; whatever digits the spec rounds its axis to, a table
    never rounds away what the value holds, but money: a money format with digits (``usd:2``) is the chart's one money
    form, chosen from every amount it draws, and its table writes each amount in it, as its notes do: to its own two
    significant figures (its ticks to the largest amount's decimals). A format of no decimals (``usd:0``) is a rate's
    from $10 up, which reads in whole dollars; any other reads to cents at least.

    Where the format names no unit (``num``) and its block says the unit its values are read at (``read_at``, a
    chart of what an analysis computes: two arms' difference, a lever's effect, in points), the value is spelled at
    that unit's precision (:func:`~threetears.evals.analysis.quantities.number_in`: a point to one decimal), the unit
    itself left to the note beside it, so a computed figure prints as its key and its note read it, never at the
    float's four figures.

    :param spec: A kit format (:data:`Fmt`).
    :ptype spec: str
    :param value: The value, in display units.
    :ptype value: float
    :param read_at: The unit the block's values are read at, where its format names none; empty for the format's.
    :ptype read_at: str
    :param finer: Whether money is written to one significant figure more: an interval whose two ends would otherwise
        read alike.
    :ptype finer: bool
    :param places: The decimals a change in points reads at where its interval needs more than a point's one (an end
        of -0.037 reads ``-0.04``, never a ``0.0`` that hides which side of zero it is on); None for one decimal.
    :ptype places: int | None
    :return: The value as text.
    :rtype: str
    :raises ValueError: ``spec`` is not a format the kit names.
    """
    match = _FORMAT_SPEC.fullmatch(spec)
    if match is None or match[2] not in _FORMAT_UNITS:
        raise ValueError(f"fmt {spec!r} is not one of {', '.join(_FORMAT_UNITS)}")
    unit = _FORMAT_UNITS[match[2]]
    if match[2] == _PER_THOUSAND_FORMAT:
        # Dollars per 1,000, the mark written, as the kit's ticks write them (a set of totals under a cent, #993).
        form = MoneyForm(per_thousand=True, decimals=int(match[3]) if match[3] is not None else 2)
        return form.spell(value, signed=bool(match[1]))
    if unit == MONEY and match[3] is not None:
        # The chart's money form (money_form): each amount to its own two significant figures, cents at least.
        decimals = int(match[3])
        form = MoneyForm(decimals=decimals, figures=MONEY_FIGURES + finer, rate=decimals == 0)
        return form.spell(value, signed=bool(match[1]))
    if unit in (PERCENT, POINTS) and match[3] is not None:
        # A format naming its digits is the set's one precision (owner ruling O5): a share read to three decimals near
        # even prints ``49.998%`` in the table as in the line under the chart, never ``50%``.
        text = spell_at(value, int(match[3]), signed=bool(match[1]))
        return f"{text}{unit}" if unit == PERCENT else f"{text} {unit}"
    if (not unit and read_at == POINTS) or unit == POINTS:
        # A change in points reads at a point's one decimal, a whole one too (``-3.0``, never ``-3`` beside ``-7.6``).
        text = spell_at(value, POINT_PLACES if places is None else places, signed=bool(match[1]))
        return text if not unit else f"{text} {unit}"
    if not unit and read_at:
        text = number_in(value, read_at)
        # Signed as format_signed signs a change: a positive value carries its "+", a bound its own direction.
        return f"+{text}" if match[1] and value > 0 and not text.startswith("<") else text
    return signed_with_unit(value, unit) if match[1] else with_unit(value, unit)


class _KitModel(EvalDocumentModel):
    """A piece of a chart's data: dumped under the contract's own names (``from``, not ``from_``), every number finite,
    since the kit reads the data with ``JSON.parse``, which has no infinity."""

    model_config = ConfigDict(serialize_by_alias=True, allow_inf_nan=False)


class ChartReference(_KitModel):
    """A floor, budget, margin or no-preference point, in value units."""

    value: float = Field(description="Where the line sits, in value units.")
    label: str | None = Field(default=None, description="Printed beside the line; none for no label.")
    kind: Literal["floor", "budget", "margin", "no_preference"] = Field(
        description=(
            "floor (pass at or above), budget (pass at or below), margin (forest: shaded from value up), "
            "no_preference (diverging bars grow from value; no line)."
        )
    )


class LegendItem(_KitModel):
    """One entry of a chart's legend, drawn above it."""

    label: str = Field(description="What the mark stands for.")
    color: Token | None = Field(default=None, description="The mark's colour token; the kit's ink-2 when none.")
    mark: Literal["dot", "square", "line", "circle", "triangle", "diamond"] | None = Field(
        default=None, description="dot (default), square, line, or a marker shape."
    )
    hollow: bool | None = Field(default=None, description="Whether the mark is drawn hollow.")
    opacity: float | None = Field(default=None, description="A line mark's opacity.")
    pattern: ChartPattern | None = Field(
        default=None, description="A swatch filled with this texture, as a bar segment carrying it is drawn."
    )


class ChartColumn(_KitModel):
    """A column of a matrix chart (gate matrix, lattice, heatmaps)."""

    label: str = Field(description="The column's heading.")


class _PanelText(_KitModel):
    """The text a small multiple's panel carries: its title, and its arm chip where it stands for an arm."""

    title: str | None = Field(default=None, description="The panel's heading.")
    subtitle: str | None = Field(default=None, description="A line under the heading.")
    tag: str | None = Field(default=None, description="An arm's letter tag: the title prints as an arm chip.")
    color: Token | None = Field(default=None, description="The arm chip's colour token.")


class _Kind(_KitModel):
    """One kind's fields: what is required of it as drawn, and its values as a table."""

    #: The fields the kit refuses to draw without, checked on the chart as drawn (each panel, overlaid).
    REQUIRED: ClassVar[tuple[str, ...]] = ()

    def check_drawable(self) -> None:
        """Refuse what the kit would refuse to draw.

        :raises ValueError: A required field is missing, or the fields disagree with each other.
        """
        if missing := [name for name in self.REQUIRED if getattr(self, name) is None]:
            raise ValueError(f"{', '.join(repr(name) for name in missing)} required")

    def table(self, legend: Sequence[LegendItem], read_at: str = "") -> tuple[list[str], list[list[str]]]:
        """The kind's values as ``(headers, rows)`` of text; ``read_at`` the unit its values are read at where its
        format names none (:func:`_formatted`), which a kind drawing computed changes spells them at."""
        raise NotImplementedError

    @property
    def _format(self) -> str:
        return str(getattr(self, "fmt", None) or "num")

    def kit_data(self) -> dict[str, JsonValue]:
        """The chart's ``data-chart`` JSON, as the kit's contract names it: unset fields left out.

        :return: The JSON object the kit draws.
        :rtype: dict[str, JsonValue]
        """
        return self.model_dump(mode="json", exclude_none=True, by_alias=True)


class _Drawable(_Kind):
    """A whole chart, as a block carries it: refused where the kit would refuse it, and read as a table."""

    @model_validator(mode="after")
    def _the_kit_draws_it(self) -> Self:
        """Refuse what the kit would refuse to draw, on the chart or on each panel as drawn.

        :raises ValueError: A required field is missing, or the fields disagree, naming the panel it is in.
        """
        for panel, one in _drawn(self):
            if one is None:
                continue
            try:
                one.check_drawable()
            except ValueError as exc:
                where = f"panel {panel.title!r}: " if panel is not None else ""
                raise ValueError(f"{where}{exc}") from exc
        return self

    def values_table(self, read_at: str = "") -> tuple[list[str], list[list[str]]]:
        """The chart's values as a table a reader without the chart can read: small multiples one after another under
        a Panel column, a text card as its title and line.

        :param read_at: The unit its values are read at where its format names none (:func:`_formatted`); empty for
            the format's own.
        :ptype read_at: str
        :return: ``(headers, rows)``, each cell text.
        :rtype: tuple[list[str], list[list[str]]]
        """
        legend: list[LegendItem] = getattr(self, "legend", None) or []
        if getattr(self, "panels", None) is None:
            return self.table(legend, read_at)
        drawn = _drawn(self)
        tables: list[tuple[_PanelText, list[str], list[list[str]]]] = []
        cards: list[_PanelText] = []
        for panel, one in drawn:
            if panel is None:
                continue
            if one is None:
                cards.append(panel)
                continue
            tables.append((panel, *one.table(legend, read_at)))
        # panels may not share their columns (a Note column one panel has no use for, an interval column another
        # has): the table holds every panel's columns, the widest panel's first and then each other one in the order
        # a panel brings it, and each panel's cells sit under their own column, the rest empty
        widest = max((own for _, own, _ in tables), key=len, default=[])
        columns = list(dict.fromkeys([*widest, *(column for _, own, _ in tables for column in own)]))
        headers = ["Panel", *(columns or [""])]
        body: list[list[str]] = []
        for panel, own_headers, rows in tables:
            at = [columns.index(column) for column in own_headers]
            for row in rows:
                cells = [""] * len(columns)
                for index, cell in zip(at, row, strict=True):
                    cells[index] = cell
                body.append([_panel_name(panel), *cells])
        body += [[card.title or "", card.subtitle or "", *([""] * (len(headers) - 2))] for card in cards]
        return headers, body


def _panel_name(panel: _PanelText) -> str:
    """A small multiple as a values table's Panel column names it: its title (after its arm's letter tag, where it is
    an arm's panel, as every arm's row is named: :func:`_tagged`), then its subtitle where it has one (an arm's panel
    names the axis or measure it draws), so no two panels read alike that the picture tells apart."""
    return " · ".join(part for part in (_tagged(panel.tag, panel.title or ""), panel.subtitle) if part)


def _tagged(tag: str | None, label: str) -> str:
    """An arm's row or panel as a values table names it: its letter tag, then its name (``A · f64``), as a chart
    whose rows carry no tag of their own labels them (an arm's look's label), so one arm reads alike in every
    block; a label without a tag, or one already opening with its tag, as it is."""
    if not tag or not label or label.startswith(f"{tag} · "):
        return label
    return f"{tag} · {label}"


#: A number in a table cell: its sign, its digits with any thousands separators, and its decimals; never inside a
#: word (``p95``) or a longer number.
_CELL_NUMBER = re.compile(r"(?<![\w.,])([+\-−]?)(\d[\d,]*)(?:\.(\d+))?(?![\w])")

#: The most decimals a column is padded to: past it the figures are small amounts whose precision is their own.
_MOST_PADDED = 3

#: The significant figures format_number rounds a number to: as many as reach the units digit at
#: :data:`~threetears.evals.kernel.numbers.WHOLE_FROM`, where it starts writing numbers whole.
_SIGNIFICANT = len(str(int(WHOLE_FROM)))

#: An amount of money in a table cell: its sign, its digits with any thousands separators and decimals, and the
#: per-1,000 mark where it carries one; never a bound (``< $0.000001``), which states no amount.
_CELL_MONEY = re.compile(r"(?<![<>] )(?<![<>\w.,+\-−])([+\-−]?)\$(\d[\d,]*(?:\.\d+)?)(/1000)?(?![\d])")


#: A percentage a cell states: its sign, its number and the percent sign; never inside a word or a longer number.
_CELL_PERCENT = re.compile(r"(?<![\w.])([+\-−]?)(\d+(?:\.\d+)?)%")


def percent_places(texts: Iterable[str]) -> int | None:
    """The decimals one column's percentages read at: none, so whole percents (``60%``); one where whole percents
    would print two of its values alike, an interval's two ends among them (``62.9%`` beside ``63.2%``), or where a
    value printed at a half (``90.5%``) cannot decide its whole percent. None for a
    column that states no percentage, or one written to two decimals or more (a confidence level, ``99.69%``), which
    keeps its digits (coordinator ruling under the owner's "do what makes sense for usability", 2026-11-07).

    :param texts: The column's cells, as they print.
    :ptype texts: Iterable[str]
    :return: 0 or 1; None for a column to leave as it is.
    :rtype: int | None
    """
    found = [match for text in texts for match in _CELL_PERCENT.finditer(text)]
    if not found or any(len(match[2].partition(".")[2]) > 1 for match in found):
        return None
    if any(match[2].endswith(".5") for match in found):
        # A percentage printed to one decimal at a half (90.5%) cannot say which whole percent it rounds to (its value
        # may be 90.48%), so the column keeps its decimal rather than print a rounding the digits cannot decide.
        return 1
    stated = {float(match[2]) for match in found}
    return 0 if len({percent_at(value, 0) for value in stated}) == len(stated) else 1


def percent_at(value: float, places: int) -> str:
    """A percentage at ``places`` decimals, half away from zero, its sign kept; an exact zero reads ``0%``, which
    states no precision.

    :param value: The percentage, in percent.
    :ptype value: float
    :param places: The decimals.
    :ptype places: int
    :return: ``63%``, ``62.9%``, ``0%``.
    :rtype: str
    """
    if value == 0:
        return "0%"
    step = decimal.Decimal(1).scaleb(-places)
    return f"{decimal.Decimal(repr(value)).quantize(step, rounding=decimal.ROUND_HALF_UP)}%"


def respell_percents(text: str, places: int) -> str:
    """Every percentage a cell's text states at ``places`` decimals (:func:`percent_at`); its other words as written.

    :param text: The cell's text.
    :ptype text: str
    :param places: The column's decimals (:func:`percent_places`).
    :ptype places: int
    :return: The text.
    :rtype: str
    """
    return _CELL_PERCENT.sub(lambda match: match[1] + percent_at(float(match[2]), places), text)


#: A bare figure a cell under a ``(%)`` or ``(pts)`` head states: its leading figure, before any words in
#: parentheses (``20.2 (167 of 828 trials)``), and an interval's two ends (``[17.2, 25]``).
_LEADING_FIGURE = re.compile(r"^([+\-−]?)(\d+(?:\.\d+)?)(?=$| )")
_BRACKETED = re.compile(r"\[([+\-−]?)(\d+(?:\.\d+)?), ([+\-−]?)(\d+(?:\.\d+)?)\]")


def _bare_figures(text: str) -> list[str]:
    """The bare figures one cell states: its leading one and its interval's ends."""
    found = [match[2] for match in _LEADING_FIGURE.finditer(text)]
    for match in _BRACKETED.finditer(text):
        found += [match[2], match[4]]
    return found


def bare_places(texts: Iterable[str]) -> int | None:
    """The decimals a column of bare percents or points reads at, by :func:`percent_places`'s rule: whole, one
    decimal where whole figures would print two alike or a figure at a half cannot decide its whole. None for a
    column that states none, or one written to two decimals or more.

    :param texts: The column's cells, as they print.
    :ptype texts: Iterable[str]
    :return: 0 or 1; None for a column to leave as it is.
    :rtype: int | None
    """
    return percent_places(f"{figure}%" for text in texts for figure in _bare_figures(text))


def bare_at(value: float, places: int) -> str:
    """A bare figure at ``places`` decimals, half away from zero, its sign kept; an exact zero reads ``0``.

    :param value: The figure.
    :ptype value: float
    :param places: The decimals.
    :ptype places: int
    :return: ``21``, ``-7.4``, ``0``.
    :rtype: str
    """
    if value == 0:
        return "0"
    step = decimal.Decimal(1).scaleb(-places)
    return str(decimal.Decimal(repr(value)).quantize(step, rounding=decimal.ROUND_HALF_UP))


def respell_bare(text: str, places: int) -> str:
    """A cell's leading bare figure and its interval's ends at ``places`` decimals (:func:`bare_at`).

    :param text: The cell's text.
    :ptype text: str
    :param places: The column's decimals (:func:`bare_places`).
    :ptype places: int
    :return: The text.
    :rtype: str
    """
    text = _LEADING_FIGURE.sub(lambda match: match[1] + bare_at(float(match[2]), places), text, count=1)
    return _BRACKETED.sub(
        lambda match: f"[{match[1]}{bare_at(float(match[2]), places)}, {match[3]}{bare_at(float(match[4]), places)}]",
        text,
    )


def _percents_by_column(table: list[list[str]]) -> list[list[str]]:
    """A table's text with each column's percentages at its one precision (:func:`percent_places`); changed in place."""
    for column in range(max((len(row) for row in table), default=0)):
        places = percent_places(row[column] for row in table if column < len(row))
        if places is None:
            continue
        for row in table:
            if column < len(row):
                row[column] = respell_percents(row[column], places)
    return table


def even_precision(rows: Sequence[Sequence[str]], *, percents: bool = True) -> list[list[str]]:
    """A values table's cells, every number at one precision among the cells that read alike: cells that read the
    same but for their numbers (``94% · 3 calls``, ``96.4% · 4 calls``), wherever they sit in the table, have each
    number in the same place padded with zeros to the most decimals that place holds among them (``94.0%``), up to
    three. A percentage reads at its column's one precision (:func:`percent_places`: whole, or one decimal where whole
    percents would print two values alike), never padded past it. Money is never padded, since each amount reads to its own two significant figures
    (:func:`~threetears.evals.kernel.numbers.money_form`): its whole dollars are only grouped by thousands (``$1,239``,
    :func:`_money_by_column`), as the figure keys write money.

    Padding never rounds: a cell keeps every digit it had, so a table still never rounds away what a value holds,
    and two figures a reader compares read at one precision. Nor does it state a digit the number was never rounded
    to: a number reaches a table as :func:`~threetears.evals.kernel.numbers.format_number` spells it, to
    :data:`_SIGNIFICANT` figures with its trailing zeros dropped (whole from
    :data:`~threetears.evals.kernel.numbers.WHOLE_FROM` up), so its zeros are padded back only up to its own
    figures: ``94`` beside ``96.4`` reads ``94.0``, but ``1240`` beside ``1.655`` stays ``1,240`` and ``12.35`` beside
    ``1.235`` stays ``12.35``. A cell no other reads like (a label, words) is left as it is.

    :param rows: The table's rows of cells.
    :ptype rows: Sequence[Sequence[str]]
    :param percents: Whether each column's percentages are set to its one precision first; a table of what an
        analysis computes prints each as its key reads it, and keeps them.
    :ptype percents: bool
    :return: The rows, each number padded among the cells that read like its own.
    :rtype: list[list[str]]
    """
    table = _money_by_column([list(row) for row in rows])
    table = _percents_by_column(table) if percents else table
    shaped: dict[str, list[tuple[int, int]]] = {}
    for at, row in enumerate(table):
        for column, cell in enumerate(row):
            shape = _CELL_NUMBER.sub("#", cell)
            if "#" in shape:
                shaped.setdefault(shape, []).append((at, column))
    for places in shaped.values():
        found = [list(_CELL_NUMBER.finditer(table[at][column])) for at, column in places]
        most = [max(len(numbers[place].group(3) or "") for numbers in found) for place in range(len(found[0]))]
        for at, column in places:
            table[at][column] = _padded_cell(table[at][column], most)
    return table


def _money_by_column(table: list[list[str]]) -> list[list[str]]:
    """Every amount of money in a table's text with its whole dollars grouped by thousands (``$1,239``), as the figure
    keys write money; a bound (``< $0.000001``) is left as it is. The per-1,000 mark retired (2026-11-07): an amount is
    never moved per 1,000 here, and one already written with the mark (a stored chart's) keeps it.

    :param table: The table's rows of cell texts; changed in place.
    :ptype table: list[list[str]]
    :return: The same rows.
    :rtype: list[list[str]]
    """
    for row in table:
        for column, cell in enumerate(row):
            row[column] = _CELL_MONEY.sub(_respelled, cell)
    return table


def _respelled(found: re.Match[str]) -> str:
    """One amount with its thousands grouped, any mark it was written with kept."""
    sign, digits, mark = found.group(1), found.group(2).replace(",", ""), found.group(3) or ""
    return f"{sign}${group_thousands(digits)}{mark}"


def _own_decimals(whole: str) -> int:
    """The most decimals a number with these whole digits holds at :data:`_SIGNIFICANT` figures: none past them, so
    none at all from :data:`~threetears.evals.kernel.numbers.WHOLE_FROM` up; below one, its figures start after the
    point, which holds at least as many decimals as :data:`_MOST_PADDED`."""
    digits = whole.replace(",", "").lstrip("0")
    return max(0, _SIGNIFICANT - len(digits)) if digits else _MOST_PADDED


def _padded_cell(cell: str, most: Sequence[int]) -> str:
    """One cell, its n-th number padded to ``most[n]`` decimals where that is three or fewer, and no further than its
    own significant figures reach."""
    place = iter(range(len(most)))

    def pad(found: re.Match[str]) -> str:
        wanted = most[next(place)]
        decimals = found.group(3) or ""
        if (
            found.string[found.start(2) - 1 : found.start(2)] == "$"
            or found.string[found.end() : found.end() + 1] == "%"
        ):
            # Money is never padded: each amount reads to its own two significant figures (money_form, owner ruling
            # Q3), so $0.11 beside $0.008 stays $0.11, never $0.110. A percentage reads at its column's one precision
            # (percent_places), set before padding.
            return found.group(0)
        if wanted > _MOST_PADDED:
            return found.group(0)
        wanted = min(wanted, _own_decimals(found.group(2)))
        if len(decimals) >= wanted:
            return found.group(0)
        return f"{found.group(1)}{found.group(2)}.{decimals.ljust(wanted, '0')}"

    return _CELL_NUMBER.sub(pad, cell)


def _interval(spec: str, low: float | None, high: float | None, read_at: str = "") -> str:
    """An interval as a table cell: its two ends under the chart's format (at ``read_at``'s precision where it names
    one: :func:`_formatted`), empty without one."""
    if low is None or high is None:
        return ""
    match = _FORMAT_SPEC.fullmatch(spec)
    if match is not None and (
        _FORMAT_UNITS.get(match[2]) == POINTS or (not _FORMAT_UNITS.get(match[2]) and read_at == POINTS)
    ):
        # Both ends at one precision: a point's one decimal, more where an end that is not zero would read as one.
        at = ends_places(low, high, POINT_PLACES)
        return f"{_formatted(spec, low, read_at, places=at)} to {_formatted(spec, high, read_at, places=at)}"
    said = (_formatted(spec, low, read_at), _formatted(spec, high, read_at))
    if said[0] == said[1] and low != high:
        # Two ends that read alike at their own figures read to one figure more, as the money form spells an interval.
        said = (_formatted(spec, low, read_at, finer=True), _formatted(spec, high, read_at, finer=True))
    return f"{said[0]} to {said[1]}"


def _restates(note: str | None, value: float | None, said: str, spec: str, read_at: str) -> bool:
    """Whether a row's note only restates its value, the label a chart draws at the bar's end: the value as the table
    writes it (``said``), or that value with the unit the row is read in beside it (the block's ``read_at`` or its
    format's own: ``+0.8 pts`` beside ``+0.8`` read in points), so a values table never prints one figure twice, in two
    columns. A note that says more (``worse``, ``-7.6 pts worse``), or names another unit, is kept. Only the code's own
    spellings of the row's own number are compared: no word of the note is read."""
    match = _FORMAT_SPEC.fullmatch(spec)
    if note is None or value is None or not math.isfinite(value) or match is None:
        return False
    spellings = {said}
    for unit in {read_at, _FORMAT_UNITS.get(match[2], "")} - {"", MONEY}:
        spellings.add(signed_with_unit(value, unit) if match[1] else with_unit(value, unit))
        spellings.add(f"{said}{unit}" if unit in UNSPACED_UNITS else f"{said} {unit}")
    return note in spellings


#: An interval open above, as a table cell or a tooltip writes it: its low end, and no upper bound.
OPEN_ABOVE = "from {low} up, no upper bound"


def _bar_interval(spec: str, low: float | None, high: float | None, read_at: str = "") -> str:
    """A bar's interval as a table cell: :func:`_interval`'s, or one open above (a low end alone) in words."""
    if low is not None and high is None:
        return OPEN_ABOVE.format(low=_formatted(spec, low, read_at))
    return _interval(spec, low, high, read_at)


def _reference_row(reference: ChartReference | None, spec: str, width: int, read_at: str = "") -> list[list[str]]:
    """A chart's reference line as a last table row: its label (or kind) and its value, padded to ``width``, at the
    precision its rows read at (``read_at``), so a margin reads ``-15.0`` in the table its label says it in."""
    if reference is None or reference.kind == "no_preference":
        return []
    name = reference.label or reference.kind
    return [[name, _formatted(spec, reference.value, read_at), *([""] * (width - 2))]]


def _need_level(level: str | None, intervals: bool) -> None:
    """Refuse intervals drawn without their level: the kit prints "<level> intervals" on every such chart."""
    if intervals and not level:
        raise ValueError('rows carry intervals, so "level" is required')


# ---------- bars and diverging bars ----------


class BarSegment(_KitModel):
    """One part of a stacked bar."""

    value: float = Field(description="The part's size, in value units.")
    color: Token | None = Field(default=None, description="The part's colour token; the row's when none.")
    pattern: ChartPattern | None = Field(default=None, description="A texture over the part's colour; solid when none.")


class BarRow(_KitModel):
    """One bar: a value, or stacked segments, with an optional interval."""

    label: str = Field(description="The row's label.")
    value: float | None = Field(default=None, description="The bar's value; none when it is stacked from segments.")
    segments: list[BarSegment] | None = Field(default=None, description="Stacked parts, from zero.")
    id: str | None = Field(default=None, description="What `emphasis` names.")
    short_label: str | None = Field(default=None, description="Used when `label` does not fit.")
    tag: str | None = Field(
        default=None,
        description="An arm's letter tag, drawn in a square of the row's colour beside the label; alone where neither "
        "label fits.",
    )
    lo: float | None = Field(default=None, description="The interval's low end.")
    hi: float | None = Field(
        default=None,
        description="The interval's high end; none with a `lo` where the interval is open above (no upper bound): its "
        "whisker runs on to the axis's end, opening into a chevron.",
    )
    color: Token | None = Field(default=None, description="The bar's colour token.")
    strong: bool | None = Field(default=None, description="Bold label and note.")
    muted: bool | None = Field(default=None, description="Faded bar.")
    note: str | None = Field(default=None, description="Printed at the bar's end; the formatted value when none.")
    tip: list[str] | None = Field(default=None, description="Tooltip lines, the first bold.")
    group: str | None = Field(default=None, description="Rows sharing a group sit under its heading.")

    @model_validator(mode="after")
    def _one_measure(self) -> Self:
        """Refuse a bar with neither a value nor segments, or with both, and a high end with no low end.

        :raises ValueError: Both or neither of ``value`` and ``segments``; ``hi`` without ``lo``.
        """
        if (self.value is None) == (self.segments is None):
            raise ValueError(f"bar {self.label!r} needs a value or segments, not both")
        if self.hi is not None and self.lo is None:
            raise ValueError(
                f"bar {self.label!r} sets a high end with no low end; only an interval open above has one end"
            )
        return self

    @property
    def total(self) -> float:
        """What the bar draws: its value, or its segments' sum."""
        return self.value if self.value is not None else sum(segment.value for segment in self.segments or [])


class _BarsFields(_Kind):
    """The fields a bar chart draws from."""

    REQUIRED: ClassVar[tuple[str, ...]] = ("rows", "domain", "fmt")

    rows: list[BarRow] | None = Field(default=None, description="The bars, in drawn order.")
    domain: tuple[float, float] | None = Field(default=None, description="The axis range, in value units.")
    fmt: Fmt | None = Field(default=None, description="The values' format.")
    tick_fmt: Fmt | None = Field(default=None, description="The axis labels' format, where it differs.")
    ticks: list[float] | None = Field(default=None, description="Explicit tick positions.")
    log: bool | None = Field(default=None, description="A log value axis, the bars drawn from its low end.")
    reference: ChartReference | None = Field(default=None, description="A floor, budget or no-preference point.")
    zero_label: str | None = Field(
        default=None,
        min_length=1,
        description="Printed at the zero line of a change chart: what zero is (no change against the control, by "
        "name, and its level where the rows share one).",
    )
    better: Literal["right", "left"] | tuple[str, str] | None = Field(default=None, description=_BETTER_WORDS)
    dots: bool | None = Field(
        default=None,
        description="Changes drawn as dots with their intervals, not bars: a row with no interval an open mark, each "
        "note at the right edge.",
    )
    level: str | None = Field(default=None, description='The intervals\' level ("95%"); required with intervals.')
    emphasis: str | None = Field(default=None, description="The id of the one row to bold.")
    label_width: float | None = Field(default=None, description="The label column in px, 480 px wide and over.")
    label_width_narrow: float | None = Field(default=None, description="The label column in px below 480 px.")
    row_height: float | None = Field(default=None, description="The row pitch in px.")

    def check_drawable(self) -> None:
        super().check_drawable()
        _need_level(self.level, any(row.lo is not None or row.hi is not None for row in self.rows or []))

    def table(self, legend: Sequence[LegendItem], read_at: str = "") -> tuple[list[str], list[list[str]]]:
        rows = self.rows or []
        names: dict[tuple[str | None, ChartPattern | None], str] = {
            (item.color, item.pattern): item.label for item in legend if item.color is not None
        }
        grouped = any(row.group is not None for row in rows)
        stacked = any(row.segments for row in rows)
        intervals = any(row.lo is not None for row in rows)
        said = [_formatted(self._format, row.total, read_at) for row in rows]
        kept = [
            None if _restates(row.note, row.total, text, self._format, read_at) else row.note
            for row, text in zip(rows, said, strict=True)
        ]
        notes = any(kept)
        headers = [
            *(["Group"] if grouped else []),
            "Row",
            "Value",
            *(["Parts"] if stacked else []),
            *([f"{self.level} interval"] if intervals else []),
            *(["Note"] if notes else []),
        ]
        body: list[list[str]] = []
        for row, text, note in zip(rows, said, kept, strict=True):
            parts = " · ".join(
                (f"{names[seg.color, seg.pattern]} " if (seg.color, seg.pattern) in names else "")
                + _formatted(self._format, seg.value)
                for seg in row.segments or []
            )
            body.append(
                [
                    *([row.group or ""] if grouped else []),
                    _tagged(row.tag, row.label),
                    text,
                    *([parts] if stacked else []),
                    *([_bar_interval(self._format, row.lo, row.hi, read_at)] if intervals else []),
                    *([note or ""] if notes else []),
                ]
            )
        reference = _reference_row(self.reference, self._format, len(headers) - grouped, read_at)
        return headers, body + [[""] * grouped + line for line in reference]


class BarsPanel(_BarsFields, _PanelText):
    """One small multiple of a bar chart: its text, and the fields it overrides."""


class BarsChart(_BarsFields, _Drawable):
    """``bars`` (magnitudes from zero) or ``diverging_bars`` (deltas about zero, or about a no-preference point):
    horizontal bars, stacked or grouped, with whiskers where rows carry intervals and a dashed reference line."""

    legend: list[LegendItem] | None = Field(default=None, description="Drawn above the chart.")
    panels: list[BarsPanel] | None = Field(default=None, description="Small multiples.")
    panel_min_rem: float | None = Field(default=None, description="The narrowest panel before a column drops.")


# ---------- forest ----------


class ForestRow(_KitModel):
    """One change against a control, with its interval."""

    label: str = Field(description="The row's label.")
    value: float = Field(description="The estimate.")
    lo: float = Field(description="The interval's low end.")
    hi: float = Field(description="The interval's high end.")
    id: str | None = Field(default=None, description="What `emphasis` names.")
    short_label: str | None = Field(default=None, description="Used when `label` does not fit.")
    color: Token | None = Field(default=None, description="The row's colour token.")
    strong: bool | None = Field(default=None, description="Bold label and note.")
    note: str | None = Field(default=None, description="Printed at the row's right edge.")
    tip: list[str] | None = Field(default=None, description="Tooltip lines, the first bold.")
    group: str | None = Field(default=None, description="Rows sharing a group sit under its heading.")


class ForestChart(_Drawable):
    """``forest``: change against a control, a zero line, each interval with a square at its estimate, and a shaded
    margin band."""

    rows: list[ForestRow] = Field(description="The rows, in drawn order.")
    domain: tuple[float, float] = Field(description="The axis range, in value units.")
    fmt: Fmt = Field(description="The values' format.")
    level: str = Field(description='The intervals\' level ("95%").')
    reference: ChartReference | None = Field(default=None, description="The margin (kind `margin`).")
    zero_label: str | None = Field(
        default=None,
        min_length=1,
        description="Printed at the zero line: what zero is (no change against the control, by name, and its level "
        "where the rows share one).",
    )
    better: Literal["right", "left"] | tuple[str, str] | None = Field(default=None, description=_BETTER_WORDS)
    emphasis: str | None = Field(default=None, description="The id of the one row to bold.")
    label_width: float | None = Field(default=None, description="The label column in px, 480 px wide and over.")
    label_width_narrow: float | None = Field(default=None, description="The label column in px below 480 px.")
    row_height: float | None = Field(default=None, description="The row pitch in px.")

    def table(self, legend: Sequence[LegendItem], read_at: str = "") -> tuple[list[str], list[list[str]]]:
        grouped = any(row.group is not None for row in self.rows)
        said = [_formatted(self.fmt, row.value, read_at) for row in self.rows]
        kept = [
            None if _restates(row.note, row.value, text, self.fmt, read_at) else row.note
            for row, text in zip(self.rows, said, strict=True)
        ]
        notes = any(kept)
        headers = [
            *(["Group"] if grouped else []),
            "Row",
            "Change",
            f"{self.level} interval",
            *(["Note"] if notes else []),
        ]
        body = [
            [
                *([row.group or ""] if grouped else []),
                row.label,
                text,
                _interval(self.fmt, row.lo, row.hi, read_at),
                *([note or ""] if notes else []),
            ]
            for row, text, note in zip(self.rows, said, kept, strict=True)
        ]
        reference = _reference_row(self.reference, self.fmt, len(headers) - grouped, read_at)
        return headers, body + [[""] * grouped + line for line in reference]


# ---------- lines ----------


class LineSeries(_KitModel):
    """One line over the chart's ordered categories."""

    name: str = Field(description="The series' name.")
    values: list[float] = Field(description="One value per category.")
    end_label: str | None = Field(default=None, description="Printed after the last point; the name when none.")
    color: Token | None = Field(default=None, description="The line's colour token.")
    lo: list[float] | None = Field(default=None, description="The band's low ends, one per category.")
    hi: list[float] | None = Field(default=None, description="The band's high ends, one per category.")
    strong: bool | None = Field(default=None, description="A heavier line and a bold end label.")
    dash: bool | None = Field(default=None, description="A dashed line.")


class LinesChart(_Drawable):
    """``lines``: values over an ordered x (turn depth, version, date), end labels de-collided, bands where series
    carry them."""

    categories: list[str] = Field(description="The ordered x values.")
    series: list[LineSeries] = Field(description="The lines.")
    domain: tuple[float, float] = Field(description="The y range, in value units.")
    fmt: Fmt = Field(description="The values' format.")
    x_title: str | None = Field(default=None, description="The x axis's title.")
    height: float | None = Field(default=None, description="The plot's height in px.")
    level: str | None = Field(default=None, description='The bands\' level ("95%"); required with bands.')

    def check_drawable(self) -> None:
        super().check_drawable()
        if not self.level and any(series.lo or series.hi for series in self.series):
            raise ValueError('series carry bands, so "level" is required')
        for series in self.series:
            for ends in (series.values, series.lo, series.hi):
                if ends is not None and len(ends) != len(self.categories):
                    raise ValueError(f"series {series.name!r} has {len(ends)} values for {len(self.categories)} x")

    def table(self, legend: Sequence[LegendItem], read_at: str = "") -> tuple[list[str], list[list[str]]]:
        headers = [self.x_title or "", *(series.name for series in self.series)]
        body = []
        for index, category in enumerate(self.categories):
            cells = [category]
            for series in self.series:
                cell = _formatted(self.fmt, series.values[index])
                if series.lo is not None and series.hi is not None:
                    cell += f" ({_interval(self.fmt, series.lo[index], series.hi[index])})"
                cells.append(cell)
            body.append(cells)
        return headers, body


# ---------- scatter and gate scatter ----------


class ScatterPoint(_KitModel):
    """One option on a scatter: two measures that trade off."""

    x: float = Field(description="Across, in x units.")
    y: float = Field(description="Up, in y units.")
    x_lo: float | None = Field(default=None, description="The low end of x's interval; drawn as a whisker with `x_hi`.")
    x_hi: float | None = Field(default=None, description="The high end of x's interval.")
    y_lo: float | None = Field(default=None, description="The low end of y's interval; drawn as a whisker with `y_hi`.")
    y_hi: float | None = Field(default=None, description="The high end of y's interval.")
    id: str | None = Field(default=None, description="What `emphasis` and `paths` name.")
    label: str | None = Field(default=None, description="The point's label at 560 px and wider.")
    narrow_label: str | None = Field(default=None, description="Its label below 560 px; unlabelled when none.")
    color: Token | None = Field(default=None, description="The mark's colour token.")
    shape: ChartShape | None = Field(default=None, description="The mark's shape.")
    size: float | None = Field(default=None, description="The mark's radius in px.")
    hollow: bool | None = Field(default=None, description="A hollow mark (a failure).")
    muted: bool | None = Field(default=None, description="A faded mark.")
    tip: list[str] | None = Field(default=None, description="Tooltip lines, the first bold.")


class ScatterPath(_KitModel):
    """A line joining named points in order: a factor's levels, a hybrid's handoffs."""

    ids: list[str] = Field(description="The points' ids, in order.")
    color: Token | None = Field(default=None, description="The line's colour token.")
    width: float | None = Field(default=None, description="The line's width in px.")
    opacity: float | None = Field(default=None, description="The line's opacity.")


class PriceLine(_KitModel):
    """y = −x / value from the origin, the side above it shaded: worth it at that price."""

    value: float = Field(description="The price, in x units per y unit.")
    label: str | None = Field(default=None, description="Printed along the line.")
    region_label: str | None = Field(default=None, description="Names the shaded region.")


class _ScatterFields(_Kind):
    """The fields a scatter draws from."""

    REQUIRED: ClassVar[tuple[str, ...]] = ("points", "x_domain", "y_domain", "x_fmt", "y_fmt")

    points: list[ScatterPoint] | None = Field(default=None, description="The points.")
    x_domain: tuple[float, float] | None = Field(default=None, description="The x range.")
    y_domain: tuple[float, float] | None = Field(default=None, description="The y range.")
    x_fmt: Fmt | None = Field(default=None, description="The x values' format.")
    y_fmt: Fmt | None = Field(default=None, description="The y values' format.")
    x_log: bool | None = Field(default=None, description="A log x axis.")
    x_ticks: list[float] | None = Field(default=None, description="Explicit x ticks.")
    y_ticks: list[float] | None = Field(default=None, description="Explicit y ticks.")
    x_title: str | None = Field(default=None, description="The x axis's title.")
    y_title: str | None = Field(default=None, description="The y axis's title.")
    reference: ChartReference | None = Field(default=None, description="A y floor or budget.")
    x_reference: ChartReference | None = Field(default=None, description="An x budget or floor.")
    frontier: bool | None = Field(default=None, description="Draw the Pareto line (lower x, higher y is better).")
    paths: list[ScatterPath] | None = Field(default=None, description="Lines joining named points.")
    price_line: PriceLine | None = Field(default=None, description="A price line from the origin.")
    emphasis: str | None = Field(default=None, description="The id of the one point to ring.")
    height: float | None = Field(default=None, description="The plot's height in px.")
    level: str | None = Field(default=None, description='The intervals\' level ("95%"); required with intervals.')

    def check_drawable(self) -> None:
        super().check_drawable()
        _need_level(
            self.level,
            any(
                end is not None
                for point in self.points or []
                for end in (point.x_lo, point.x_hi, point.y_lo, point.y_hi)
            ),
        )
        ids = {point.id for point in self.points or [] if point.id is not None}
        for path in self.paths or []:
            if missing := [one for one in path.ids if one not in ids]:
                raise ValueError(f"a path names points not in `points`: {', '.join(missing)}")

    def table(self, legend: Sequence[LegendItem], read_at: str = "") -> tuple[list[str], list[list[str]]]:
        x_fmt, y_fmt = self.x_fmt or "num", self.y_fmt or "num"
        headers = ["Point", self.x_title or "x", self.y_title or "y"]
        body = [
            [
                _point_name(point),
                _formatted(x_fmt, point.x),
                _formatted(y_fmt, point.y),
            ]
            for point in self.points or []
        ]
        for reference, spec, column in ((self.x_reference, x_fmt, 1), (self.reference, y_fmt, 2)):
            if reference is not None:
                line = [reference.label or reference.kind, "", ""]
                line[column] = _formatted(spec, reference.value)
                body.append(line)
        return headers, body


def _point_name(point: ScatterPoint) -> str:
    """A scatter point as its table row names it: its drawn label, else its tooltip's first line (the bold one, which
    names the point), else its id; a crowded chart draws no label, and its table still names each point for a
    reader, never by an internal id while a name is held."""
    return point.label or point.narrow_label or (point.tip[0] if point.tip else None) or point.id or ""


class ScatterPanel(_ScatterFields, _PanelText):
    """One small multiple of a scatter: its text, and the fields it overrides."""


class ScatterChart(_ScatterFields, _Drawable):
    """``scatter_frontier``: two measures that trade off, with a floor, a Pareto line, paths and a price line, labels
    placed clear of each other."""

    legend: list[LegendItem] | None = Field(default=None, description="Drawn above the chart.")
    panels: list[ScatterPanel] | None = Field(default=None, description="Small multiples.")
    panel_min_rem: float | None = Field(default=None, description="The narrowest panel before a column drops.")


class _GateScatterFields(_ScatterFields):
    """The fields a gate scatter draws from: a scatter's, both references required, and the passing region's name."""

    REQUIRED: ClassVar[tuple[str, ...]] = (*_ScatterFields.REQUIRED, "reference", "x_reference")

    region_label: str | None = Field(default=None, description="Names the region where both references pass.")


class GateScatterPanel(_GateScatterFields, _PanelText):
    """One small multiple of a gate scatter: its text, and the fields it overrides."""


class GateScatterChart(_GateScatterFields, _Drawable):
    """``gate_scatter``: a scatter whose region passing both references is shaded and named."""

    legend: list[LegendItem] | None = Field(default=None, description="Drawn above the chart.")
    panels: list[GateScatterPanel] | None = Field(default=None, description="Small multiples.")
    panel_min_rem: float | None = Field(default=None, description="The narrowest panel before a column drops.")


# ---------- dot and interval strips ----------


class DotIntervalRow(_KitModel):
    """One rate against a floor, or one change about zero: a mark at the value and a faint interval."""

    label: str = Field(description="The row's label; wrapped to two lines where it is too long for one.")
    value: float = Field(description="The value.")
    lo: float | None = Field(default=None, description="The interval's low end.")
    hi: float | None = Field(default=None, description="The interval's high end.")
    id: str | None = Field(default=None, description="What `emphasis` names.")
    short_label: str | None = Field(default=None, description="Used when `label` does not fit on two lines.")
    color: Token | None = Field(default=None, description="The mark's colour token.")
    shape: ChartShape | None = Field(default=None, description="The mark's shape.")
    hollow: bool | None = Field(
        default=None, description="A hollow mark: a failure against a floor, or a change with no interval."
    )
    strong: bool | None = Field(default=None, description="A bold label.")
    note: str | None = Field(default=None, description="Printed at the row's right edge; nothing when none.")
    tip: list[str] | None = Field(default=None, description="Tooltip lines, the first bold.")


class _DotIntervalFields(_Kind):
    """The fields a dot-and-interval strip draws from."""

    REQUIRED: ClassVar[tuple[str, ...]] = ("rows", "domain", "fmt")

    rows: list[DotIntervalRow] | None = Field(default=None, description="The rows, in drawn order.")
    domain: tuple[float, float] | None = Field(default=None, description="The axis range.")
    fmt: Fmt | None = Field(default=None, description="The values' format.")
    level: str | None = Field(
        default=None, description='The intervals\' level ("95%"); required with intervals, unprinted without them.'
    )
    ticks: list[float] | None = Field(default=None, description="Explicit tick positions.")
    reference: ChartReference | None = Field(default=None, description="The floor.")
    zero_label: str | None = Field(
        default=None,
        min_length=1,
        description="Printed at a change chart's zero line, which it draws: what zero is (no change against the "
        "control, by name, and its level where the rows share one).",
    )
    better: Literal["right", "left"] | tuple[str, str] | None = Field(default=None, description=_BETTER_WORDS)
    emphasis: str | None = Field(default=None, description="The id of the one row to bold.")
    label_width: float | None = Field(default=None, description="The label column in px, 480 px wide and over.")
    label_width_narrow: float | None = Field(default=None, description="The label column in px below 480 px.")
    row_height: float | None = Field(default=None, description="The row pitch in px.")

    def check_drawable(self) -> None:
        super().check_drawable()
        _need_level(self.level, any(row.lo is not None or row.hi is not None for row in self.rows or []))

    def table(self, legend: Sequence[LegendItem], read_at: str = "") -> tuple[list[str], list[list[str]]]:
        rows = self.rows or []
        said = [_formatted(self._format, row.value, read_at) for row in rows]
        kept = [
            None if _restates(row.note, row.value, text, self._format, read_at) else row.note
            for row, text in zip(rows, said, strict=True)
        ]
        notes = any(kept)
        headers = ["Row", "Value", *([f"{self.level} interval"] if self.level else []), *(["Note"] if notes else [])]
        body = [
            [
                row.label,
                text,
                *([_interval(self._format, row.lo, row.hi, read_at)] if self.level else []),
                *([note or ""] if notes else []),
            ]
            for row, text, note in zip(rows, said, kept, strict=True)
        ]
        return headers, body + _reference_row(self.reference, self._format, len(headers), read_at)


class DotIntervalPanel(_DotIntervalFields, _PanelText):
    """One small multiple of a dot-and-interval strip: its text, and the fields it overrides."""


class DotIntervalChart(_DotIntervalFields, _Drawable):
    """``dot_interval``: one rate per candidate against a floor, each a mark with a faint interval."""

    panels: list[DotIntervalPanel] | None = Field(default=None, description="Small multiples.")
    panel_min_rem: float | None = Field(default=None, description="The narrowest panel before a column drops.")


# ---------- range bars ----------


class RangeRow(_KitModel):
    """One distribution as a bar from ``from`` (p50) to ``to`` (p95)."""

    label: str = Field(description="The row's label.")
    from_: float = Field(alias="from", description="Where the bar starts (p50).")
    to: float = Field(description="Where the bar ends, marked (p95).")
    color: Token | None = Field(default=None, description="The bar's colour token.")
    shape: ChartShape | None = Field(default=None, description="The end mark's shape.")
    hollow: bool | None = Field(default=None, description="A hollow end mark (over budget).")
    tip: list[str] | None = Field(default=None, description="Tooltip lines, the first bold.")

    @model_serializer(mode="wrap")
    def _from_as_the_contract_spells_it(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Write ``from_`` as ``from`` however the row is dumped, so a stored report spells the field as the kit's
        contract does; a report stored with ``from_`` still loads, by the field's name."""
        dumped: dict[str, Any] = handler(self)
        return {("from" if key == "from_" else key): value for key, value in dumped.items()}


class RangeGroup(_KitModel):
    """A group of range bars under one heading (a task)."""

    label: str = Field(description="The group's heading.")
    rows: list[RangeRow] = Field(description="Its bars.")
    short_label: str | None = Field(default=None, description="The heading below 480 px.")


class RangeBarsChart(_Drawable):
    """``range_bars``: latency distributions, one bar per row from p50 to p95, the budget line across."""

    groups: list[RangeGroup] = Field(description="The groups, in drawn order.")
    domain: tuple[float, float] = Field(description="The axis range.")
    fmt: Fmt = Field(description="The values' format.")
    ticks: list[float] | None = Field(default=None, description="Explicit tick positions.")
    reference: ChartReference | None = Field(default=None, description="The budget.")
    legend: list[LegendItem] | None = Field(default=None, description="Drawn above the chart.")

    def table(self, legend: Sequence[LegendItem], read_at: str = "") -> tuple[list[str], list[list[str]]]:
        headers = ["Group", "Row", "From", "To"]
        body = [
            [group.label, row.label, _formatted(self.fmt, row.from_, read_at), _formatted(self.fmt, row.to, read_at)]
            for group in self.groups
            for row in group.rows
        ]
        reference = _reference_row(self.reference, self.fmt, 3, read_at)
        return headers, body + [["", *line] for line in reference]


# ---------- gate matrix ----------


class GateCell(_KitModel):
    """One candidate on one task: a status pill, the binding reasons, the cost."""

    status: ChartStatus = Field(description="good, warn, bad or neutral.")
    label: str = Field(description="The pill's word; empty for no pill (a cell no gate reads, where none is declared).")
    reasons: list[str] | None = Field(default=None, description="The binding reasons.")
    cost: float | None = Field(default=None, description="The cost, under `cost_fmt`.")
    pick: bool | None = Field(default=None, description="The pick on this row: washed and marked.")


class GateRow(_KitModel):
    """One task across the candidates."""

    label: str = Field(description="The row's label.")
    cells: list[GateCell] = Field(description="One cell per column.")
    detail: str | None = Field(default=None, description="A line under the label.")
    flag: str | None = Field(default=None, description="Marks a row that applies to all traffic.")


class GateMatrixChart(_Drawable):
    """``gate_matrix``: pass or fail per candidate × task, as a table of pills, reasons and costs."""

    columns: list[ChartColumn] = Field(description="The candidates.")
    rows: list[GateRow] = Field(description="The tasks.")
    row_header: str | None = Field(default=None, description="The first column's heading.")
    cost_fmt: Fmt | None = Field(default=None, description="The costs' format; required when cells carry one.")
    cost_unit: str | None = Field(default=None, description="Printed after each cost.")
    pick_label: str | None = Field(default=None, description="Printed in each pick cell.")

    def check_drawable(self) -> None:
        super().check_drawable()
        for row in self.rows:
            if len(row.cells) != len(self.columns):
                raise ValueError(f"row {row.label!r} has {len(row.cells)} cells for {len(self.columns)} columns")
            if self.cost_fmt is None and any(cell.cost is not None for cell in row.cells):
                raise ValueError('cells carry "cost", so "cost_fmt" is required')

    def table(self, legend: Sequence[LegendItem], read_at: str = "") -> tuple[list[str], list[list[str]]]:
        headers = [self.row_header or "", *(column.label for column in self.columns)]
        costs = _costs_alike(self.cost_fmt, self.cost_unit, [cell.cost for row in self.rows for cell in row.cells])
        body = []
        for row in self.rows:
            label = row.label + "".join(f" ({extra})" for extra in (row.detail, row.flag) if extra)
            cells = [label]
            for cell in row.cells:
                said = "; ".join(cell.reasons or [])
                text = f"{cell.label}: {said}" if cell.label and said else cell.label or said
                cost = next(costs)
                if cost:
                    text = f"{text}, {cost}" if text else cost
                if cell.pick and self.pick_label:
                    text += f" ({self.pick_label})"
                cells.append(text)
            body.append(cells)
        return headers, body


# ---------- lattice ----------


class LatticeCell(_KitModel):
    """One combination of two factors: the metric, cost, latency and a status."""

    value: float = Field(description="The metric.")
    id: str | None = Field(default=None, description="What `emphasis` names.")
    cost: float | None = Field(default=None, description="The cost, under `cost_fmt`.")
    latency: float | None = Field(default=None, description="The latency, under `latency_fmt`.")
    status: ChartStatus | None = Field(default=None, description="A status pill.")
    status_label: str | None = Field(default=None, description="The pill's word.")
    tip: list[str] | None = Field(default=None, description="Tooltip lines, the first bold.")


class LatticeRow(_KitModel):
    """One level of the row factor."""

    label: str = Field(description="The row's label.")
    cells: list[LatticeCell] = Field(description="One cell per column.")
    color: Token | None = Field(default=None, description="A swatch beside the label.")


class LatticeChart(_Drawable):
    """``lattice``: two crossed factors, a tile per cell with the metric, a bar, cost, latency and a pill."""

    columns: list[ChartColumn] = Field(description="The column factor's levels.")
    rows: list[LatticeRow] = Field(description="The row factor's levels.")
    fmt: Fmt = Field(description="The metric's format.")
    bar_domain: tuple[float, float] | None = Field(default=None, description="The range each tile's bar spans.")
    cost_fmt: Fmt | None = Field(default=None, description="The costs' format.")
    cost_unit: str | None = Field(default=None, description="Printed after each cost.")
    latency_fmt: Fmt | None = Field(default=None, description="The latencies' format.")
    emphasis: str | None = Field(default=None, description="The id of the one cell to outline.")

    def check_drawable(self) -> None:
        super().check_drawable()
        _same_width(self.rows, len(self.columns))

    def table(self, legend: Sequence[LegendItem], read_at: str = "") -> tuple[list[str], list[list[str]]]:
        headers = ["", *(column.label for column in self.columns)]
        costs = _costs_alike(self.cost_fmt, self.cost_unit, [cell.cost for row in self.rows for cell in row.cells])
        body = []
        for row in self.rows:
            cells = [row.label]
            for cell in row.cells:
                parts = [_formatted(self.fmt, cell.value)]
                cost = next(costs)
                if cost:
                    parts.append(cost)
                if cell.latency is not None and self.latency_fmt is not None:
                    parts.append(_formatted(self.latency_fmt, cell.latency))
                if cell.status_label:
                    parts.append(cell.status_label)
                cells.append(" · ".join(parts))
            body.append(cells)
        return headers, body


def _costs_alike(fmt: str | None, unit: str | None, costs: Sequence[float | None]) -> Iterator[str]:
    """Each cell's cost as a matrix's table prints it, in cell order, every cost at one precision
    (:func:`even_precision` over the costs alone): a cell's other words (its reasons, its pick) make it read unlike its
    neighbours, so padding the whole cells would leave ``$41.0`` beside ``$17``. Empty for a cell with no cost, or
    every cell where the block names no cost format."""
    spelled = [
        _formatted(fmt, cost) + (f" {unit}" if unit else "") if cost is not None and fmt is not None else ""
        for cost in costs
    ]
    alike = iter(even_precision([[cost] for cost in spelled if cost]))
    return iter([next(alike)[0] if cost else "" for cost in spelled])


def _same_width(rows: Sequence[Any], width: int) -> None:
    """Refuse a matrix row of other than one cell per column.

    :raises ValueError: A row's cells are not as many as the columns.
    """
    for row in rows:
        if len(row.cells) != width:
            raise ValueError(f"row {row.label!r} has {len(row.cells)} cells for {width} columns")


# ---------- heatmap ----------


class HeatmapCell(_KitModel):
    """One cell of a target matrix."""

    value: float = Field(description="The value, shaded over `domain`.")
    n: float | None = Field(default=None, description="The cases under it.")
    low: float | None = Field(default=None, description="Its interval's low end, where it carries one.")
    high: float | None = Field(default=None, description="Its interval's high end; set with `low`.")
    tip: list[str] | None = Field(default=None, description="Tooltip lines, the first bold.")


class HeatmapRow(_KitModel):
    """One row of a target matrix."""

    label: str = Field(description="The row's label.")
    cells: list[HeatmapCell] = Field(description="One cell per column.")


class _HeatmapFields(_Kind):
    """The fields a heatmap draws from."""

    REQUIRED: ClassVar[tuple[str, ...]] = ("columns", "rows", "domain", "fmt")

    columns: list[ChartColumn] | None = Field(default=None, description="The columns.")
    rows: list[HeatmapRow] | None = Field(default=None, description="The rows.")
    domain: tuple[float, float] | None = Field(default=None, description="The range the six shades span.")
    fmt: Fmt | None = Field(default=None, description="The values' format.")
    n_fmt: Fmt | None = Field(default=None, description="The counts' format; counts are not printed without it.")
    reference: ChartReference | None = Field(default=None, description="The floor: cells below it are outlined.")

    def check_drawable(self) -> None:
        super().check_drawable()
        _same_width(self.rows or [], len(self.columns or []))

    def table(self, legend: Sequence[LegendItem], read_at: str = "") -> tuple[list[str], list[list[str]]]:
        headers = ["", *(column.label for column in self.columns or [])]
        body = [
            [
                row.label,
                *(
                    _formatted(self._format, cell.value) + _heat_detail(self._format, cell.low, cell.high, cell.n)
                    for cell in row.cells
                ),
            ]
            for row in self.rows or []
        ]
        return headers, body


def _heat_detail(spec: str, low: float | None, high: float | None, n: float | None) -> str:
    """What a heatmap cell's table text adds after its value: its interval and its cases, in parentheses."""
    said = [_interval(spec, low, high)] if low is not None and high is not None else []
    said += [f"n = {format_number(n)}"] if n is not None else []
    return f" ({'; '.join(said)})" if said else ""


class HeatmapPanel(_HeatmapFields, _PanelText):
    """One small multiple of a heatmap: its text, and the fields it overrides."""


class HeatmapChart(_HeatmapFields, _Drawable):
    """``heatmap``: a target matrix shaded in six steps, the count under each value, cells below the floor
    outlined."""

    panels: list[HeatmapPanel] | None = Field(default=None, description="Small multiples.")
    panel_min_rem: float | None = Field(default=None, description="The narrowest panel before a column drops.")


# ---------- difference heatmap ----------


class DiffHeatmapCell(_KitModel):
    """One difference between two configurations."""

    value: float = Field(description="The difference.")
    detail: float | None = Field(default=None, description="Printed small under it (the compared rate).")
    tip: list[str] | None = Field(default=None, description="Tooltip lines, the first bold.")


class DiffHeatmapRow(_KitModel):
    """One row of differences."""

    label: str = Field(description="The row's label.")
    cells: list[DiffHeatmapCell] = Field(description="One cell per column.")


class DiffHeatmapChart(_Drawable):
    """``diff_heatmap``: differences shaded on a diverging ramp, gray within ±``neutral``."""

    columns: list[ChartColumn] = Field(description="The columns.")
    rows: list[DiffHeatmapRow] = Field(description="The rows.")
    fmt: Fmt = Field(description="The differences' format.")
    detail_fmt: Fmt | None = Field(default=None, description="The details' format; not printed without it.")
    neutral: float | None = Field(default=None, description="Gray within ± this (default 0.5).")
    saturate: float | None = Field(default=None, description="Full colour at ± this (default 8).")

    def check_drawable(self) -> None:
        super().check_drawable()
        _same_width(self.rows, len(self.columns))

    def table(self, legend: Sequence[LegendItem], read_at: str = "") -> tuple[list[str], list[list[str]]]:
        headers = ["", *(column.label for column in self.columns)]
        detail = self.detail_fmt
        body = [
            [
                row.label,
                *(
                    _formatted(self.fmt, cell.value)
                    + (f" ({_formatted(detail, cell.detail)})" if detail and cell.detail is not None else "")
                    for cell in row.cells
                ),
            ]
            for row in self.rows
        ]
        return headers, body


# ---------- confusion ----------


class _ConfusionFields(_Kind):
    """The fields a confusion matrix draws from."""

    REQUIRED: ClassVar[tuple[str, ...]] = ("labels", "matrix")

    labels: list[str] | None = Field(default=None, description="The classes, in row and column order.")
    matrix: list[list[float]] | None = Field(default=None, description="One row per true class.")
    fmt: Fmt | None = Field(default=None, description="The shares' format (default pct:1).")
    normalized: bool | None = Field(
        default=None, description="Whether rows already sum to 100 (default true); counts are normalized when false."
    )
    warn_at: float | None = Field(default=None, description="An off-diagonal share warns from here (default 1).")
    bad_at: float | None = Field(default=None, description="An off-diagonal share is bad from here (default 3).")

    def check_drawable(self) -> None:
        super().check_drawable()
        labels, matrix = self.labels or [], self.matrix or []
        if len(matrix) != len(labels) or any(len(row) != len(labels) for row in matrix):
            raise ValueError(f'"matrix" must be {len(labels)} × {len(labels)}, one row and column per label')

    def shares(self) -> list[list[float]]:
        """Each row as shares of 100, as the kit draws it."""
        rows = self.matrix or []
        if self.normalized is not False:
            return rows
        return [[value / total * 100 if (total := sum(row)) else 0.0 for value in row] for row in rows]

    def table(self, legend: Sequence[LegendItem], read_at: str = "") -> tuple[list[str], list[list[str]]]:
        spec = self.fmt or "pct:1"
        labels = self.labels or []
        headers = ["", *labels]
        return headers, [
            [label, *(_formatted(spec, value) for value in row)]
            for label, row in zip(labels, self.shares(), strict=True)
        ]


class ConfusionPanel(_ConfusionFields, _PanelText):
    """One small multiple of a confusion matrix (one label axis): its text, and the fields it overrides."""


class ConfusionChart(_ConfusionFields, _Drawable):
    """``confusion``: errors between classes, row-normalized, the diagonal tinted and off-diagonal shares warned.

    Its values table is the matrix, a row per true label and a column per predicted one, where it draws one; in small
    multiples, whose label axes each have their own labels, it is a row per cell (the panel, the true label, the
    predicted label and the share of the true label's row), so no panel's cells sit under another's labels.
    """

    panels: list[ConfusionPanel] | None = Field(default=None, description="Small multiples, one per label axis.")
    panel_min_rem: float | None = Field(default=None, description="The narrowest panel before a column drops.")

    def values_table(self, read_at: str = "") -> tuple[list[str], list[list[str]]]:
        if self.panels is None:
            return super().values_table(read_at)
        body: list[list[str]] = []
        for panel, one in _drawn(self):
            if panel is None or not isinstance(one, _ConfusionFields):
                continue
            spec = one.fmt or "pct:1"
            labels = one.labels or []
            for expected, shares in zip(labels, one.shares(), strict=True):
                body += [
                    [_panel_name(panel), expected, predicted, _formatted(spec, share)]
                    for predicted, share in zip(labels, shares, strict=True)
                ]
        return ["Panel", "True", "Predicted", "Share of the true label"], body


# ---------- example cards ----------


class ExampleMetaText(_KitModel):
    """Words on a card's meta line: an id, a case, a stratum."""

    text: str = Field(description="The words.")
    mono: bool | None = Field(default=None, description="Set in the mono face (an id).")


class ExampleMetaStatus(_KitModel):
    """A status pill on a card's meta line."""

    status: ChartStatus = Field(description="good, warn, bad or neutral.")
    label: str = Field(description="The pill's word.")


class ExampleMetaArm(_KitModel):
    """An arm chip on a card's meta line."""

    tag: str = Field(description="The arm's letter tag.")
    label: str = Field(description="The arm's short name.")
    color: Token | None = Field(default=None, description="The tag's colour token.")


class ExampleField(_KitModel):
    """A term and its text on a card."""

    term: str = Field(description="The term.")
    text: str = Field(description="Its text.")
    tone: Literal["bad"] | None = Field(default=None, description="`bad` sets the text in red.")


class ExampleCard(_KitModel):
    """One failure example: the meta line, then each part present, in order."""

    meta: list[ExampleMetaText | ExampleMetaStatus | ExampleMetaArm] | None = Field(
        default=None, description="Ids, stratum, a status pill, an arm chip."
    )
    question: str | None = Field(default=None, description="What the case asked.")
    fields: list[ExampleField] | None = Field(default=None, description="Terms and their text.")
    reason: str | None = Field(default=None, description="The judge's reason.")
    excerpt: str | None = Field(default=None, description="An excerpt of the reply.")
    note: str | None = Field(default=None, description="A closing note.")


class ExamplesChart(_Drawable):
    """``examples``: failure example cards."""

    cards: list[ExampleCard] = Field(description="The cards, in order.")
    card_min_rem: float | None = Field(default=None, description="The narrowest card before a column drops.")

    def table(self, legend: Sequence[LegendItem], read_at: str = "") -> tuple[list[str], list[list[str]]]:
        def meta(card: ExampleCard) -> str:
            return " · ".join(
                item.text if isinstance(item, ExampleMetaText) else item.label for item in card.meta or []
            )

        columns: list[tuple[str, Callable[[ExampleCard], str]]] = [
            ("Example", meta),
            ("Question", lambda card: card.question or ""),
            ("Details", lambda card: "; ".join(f"{field.term}: {field.text}" for field in card.fields or [])),
            ("Reason", lambda card: card.reason or ""),
            ("Excerpt", lambda card: card.excerpt or ""),
            ("Note", lambda card: card.note or ""),
        ]
        shown = [(header, read) for header, read in columns if any(read(card) for card in self.cards)]
        return [header for header, _ in shown], [[read(card) for _, read in shown] for card in self.cards]


# ---------- the vocabulary ----------

#: The kinds the kit draws, each to the model of its data: bars and diverging bars share one shape.
KIT_CHARTS: dict[str, type[_Drawable]] = {
    "bars": BarsChart,
    "diverging_bars": BarsChart,
    "forest": ForestChart,
    "lines": LinesChart,
    "scatter_frontier": ScatterChart,
    "gate_scatter": GateScatterChart,
    "dot_interval": DotIntervalChart,
    "range_bars": RangeBarsChart,
    "gate_matrix": GateMatrixChart,
    "lattice": LatticeChart,
    "heatmap": HeatmapChart,
    "diff_heatmap": DiffHeatmapChart,
    "confusion": ConfusionChart,
    "examples": ExamplesChart,
}

#: A kit chart's data, whichever kind it draws. The chart block names the kind; the kind picks the model.
KitChart = (
    BarsChart
    | ForestChart
    | LinesChart
    | GateScatterChart
    | ScatterChart
    | DotIntervalChart
    | RangeBarsChart
    | GateMatrixChart
    | LatticeChart
    | HeatmapChart
    | DiffHeatmapChart
    | ConfusionChart
    | ExamplesChart
)

#: Each chart with small multiples, to the fields one panel is drawn from once it is overlaid on the chart's.
_PANEL_FIELDS: dict[type[_Kind], type[_Kind]] = {
    BarsChart: _BarsFields,
    ScatterChart: _ScatterFields,
    GateScatterChart: _GateScatterFields,
    DotIntervalChart: _DotIntervalFields,
    HeatmapChart: _HeatmapFields,
    ConfusionChart: _ConfusionFields,
}


def _drawn(chart: _Kind) -> list[tuple[_PanelText | None, _Kind | None]]:
    """Each chart a kit chart draws: itself, or each panel overlaid on the chart's shared fields as the kit overlays
    them (a panel's field replaces the chart's whole); a text card as ``None``."""
    panels: list[_PanelText] | None = getattr(chart, "panels", None)
    if panels is None:
        return [(None, chart)]
    fields = _PANEL_FIELDS[type(chart)]
    shared = chart.model_dump(mode="json", exclude_none=True, by_alias=True, exclude=set(_CHART_ONLY))
    drawn: list[tuple[_PanelText | None, _Kind | None]] = []
    for panel in panels:
        own = panel.model_dump(mode="json", exclude_none=True, by_alias=True, exclude=set(_PANEL_TEXT))
        drawn.append((panel, fields.model_validate({**shared, **own}) if own else None))
    return drawn


__all__ = [
    "BarRow",
    "BarSegment",
    "BarsChart",
    "BarsPanel",
    "ChartColumn",
    "ChartPattern",
    "ChartReference",
    "ChartShape",
    "ChartStatus",
    "ConfusionChart",
    "ConfusionPanel",
    "DiffHeatmapCell",
    "DiffHeatmapChart",
    "DiffHeatmapRow",
    "DotIntervalChart",
    "DotIntervalPanel",
    "DotIntervalRow",
    "ExampleCard",
    "ExampleField",
    "ExampleMetaArm",
    "ExampleMetaStatus",
    "ExampleMetaText",
    "ExamplesChart",
    "ForestChart",
    "ForestRow",
    "GateCell",
    "GateMatrixChart",
    "GateRow",
    "GateScatterChart",
    "GateScatterPanel",
    "HeatmapCell",
    "HeatmapChart",
    "HeatmapPanel",
    "HeatmapRow",
    "KIT_CHARTS",
    "KitChart",
    "LatticeCell",
    "LatticeChart",
    "LatticeRow",
    "LegendItem",
    "LineSeries",
    "LinesChart",
    "OPEN_ABOVE",
    "PriceLine",
    "RangeBarsChart",
    "RangeGroup",
    "RangeRow",
    "ScatterChart",
    "ScatterPanel",
    "ScatterPath",
    "ScatterPoint",
    "even_precision",
]

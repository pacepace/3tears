"""Chart intent — what a chart says, in eval's own small vocabulary, and never how it looks.

**The engine owns every decision that makes a chart honest; a renderer owns how it looks.** Which
chart, what it draws, in what order, against what baseline, in which unit, what each interval varies
over, what has to be said beside it — those are decided here, once, and travel as a
:class:`ChartIntent`. Colours, fonts, sizes, label placement and the grammar of any charting library
are not here at all: a renderer (the Vega-Lite compiler under :mod:`threetears.evals.vega.compiler`
is the first) reads an intent and a host's palette and returns its own form.

**The vocabulary is small and closed.** Eight chart types (:data:`ChartType`, the stored
``Viz.type``), each with its typed data (:data:`~threetears.evals.analysis.viz.payloads.PAYLOAD_MODELS`),
and a handful of encoding roles (:data:`EncodingRole`). Each field a chart places is declared with
the role it plays — the identity a row is named by, a length measured from zero, a position, the two
ends of an interval, a level, a class, a time position, a count, a label — and the axis it is
measured against; colour is a named SLOT in a scheme, never a value.

**Two tables, deliberately.** ``data`` holds the marks' values, numeric and in drawn order — what a
renderer places. ``columns``/``rows`` are the values as drawn: the same chart as a table a reader can
check the picture against, each header carrying its unit, each cell already spelled for reading. The
second is held to the first by policy rule 12 (a row's value under a key a mark of its identity carries is
that mark's value), and a renderer to the first by its conformance test, which reports both — so a picture
that passes agrees with the table beside it. A surface that cannot draw shows the second and loses nothing
the chart claims.

**The policy rules read this, not a renderer's output** (:mod:`threetears.evals.analysis.viz.policy`),
so they hold whatever draws the chart.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from threetears.evals.analysis.viz.payloads import PayloadError, parse_payload
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.campaign import VizType
from threetears.evals.contracts.prose import ModelProse

#: The chart-intent shape's version. Moves when a field is added, renamed or removed, or when a
#: field's meaning moves under its name — a host renderer reads it to know what it was handed.
INTENT_VERSION: Literal[1] = 1

#: The chart types — eval's own, and the only ones an intent can carry.
ChartType = VizType

#: What a field encodes.
#:
#: - ``identity`` — what a row or mark is NAMED by; carried on position (an axis or a facet), never on colour.
#: - ``length`` — a magnitude drawn as a length from the axis's zero, so the axis must start at zero.
#: - ``position`` — a location on a quantitative axis; the axis may crop to the data.
#: - ``interval_low`` / ``interval_high`` — the two ends of an uncertainty interval, which states what it varies over.
#: - ``level`` — a swept level, carried in a colour scheme's slots or ramp.
#: - ``class`` — a category carried by shape (``ChartIntent.shapes``), never by hue.
#: - ``ordinal`` — a position on a categorical axis whose order is stated: a build or a day, a named bin.
#: - ``count`` — a tally drawn as a length from zero.
#: - ``label`` — text drawn beside a mark and measured against nothing.
EncodingRole = Literal[
    "identity",
    "length",
    "position",
    "interval_low",
    "interval_high",
    "level",
    "class",
    "ordinal",
    "count",
    "label",
]

#: Roles that are drawn against a quantitative axis, and so must name one.
MEASURED_ROLES: frozenset[str] = frozenset({"length", "position", "interval_low", "interval_high", "count"})

#: Roles drawn as a length from the axis's zero.
LENGTH_ROLES: frozenset[str] = frozenset({"length", "count"})

#: Roles that are one end of an interval, and so owe a statement of what it varies over.
INTERVAL_ROLES: frozenset[str] = frozenset({"interval_low", "interval_high"})

#: A values-table or data cell: a JSON scalar.
Cell = str | int | float | bool | None


class ChartAxis(EvalBaseModel):
    """One ruler a chart measures against."""

    name: str = Field(min_length=1, description="The axis's name, by which an encoding refers to it.")
    quantity: str = Field(
        min_length=1,
        description="What the axis measures, as a reader is told it — the quantity, with its unit in parentheses where it has one.",
    )
    unit: str = Field(
        default="", description="The unit the axis is drawn in, after any restatement; empty when it has none."
    )
    zero_baseline: bool = Field(
        description="Whether the axis starts at zero. Required wherever a length is drawn against it; a position may crop."
    )
    order: list[str] = Field(
        default_factory=list,
        description="A categorical axis's positions, in their stated order (a time axis); empty for a quantitative one.",
    )


class ChartEncoding(EvalBaseModel):
    """What one field of ``data`` encodes, and against which axis."""

    field: str = Field(min_length=1, description="The key in each `data` row.")
    role: EncodingRole = Field(description="What the field encodes — see `EncodingRole`.")
    axis: str = Field(default="", description="The axis the field is measured against; empty for a role drawn on none.")
    varies_over: str = Field(
        default="", description="For an interval end: what the interval varies over. Empty for every other role."
    )


class ChartIdentity(EvalBaseModel):
    """What a chart's rows are named by, and the order they are drawn in."""

    field: str = Field(min_length=1, description="The `data` key carrying each row's identity.")
    order: list[str] = Field(description="Every identity, in drawn order — stated, never left to a renderer to sort.")
    ordered_by: str = Field(min_length=1, description="What decided that order, in plain words.")
    ranked_by: str = Field(
        default="",
        description=(
            "The `data` field the order RANKS by, descending, when the order is a ranking claim; empty when it is not. "
            "A ranking is ordered by a measure it draws, never by a column a reader is meant to discover a pattern in."
        ),
    )


class ChartColours(EvalBaseModel):
    """One colour scheme a chart uses: which field it carries, and the slot each value takes."""

    field: str = Field(min_length=1, description="The `data` key whose values are coloured.")
    scheme: Literal["categorical", "sequential"] = Field(
        description=(
            "`categorical`: each value in `domain` takes the slot of its position (first value, slot 1), in a fixed "
            "order that never cycles inside the domain. `sequential`: `domain` runs low to high and takes a ramp."
        )
    )
    domain: list[str] = Field(description="The coloured values, in slot or ramp order.")
    uncoloured: list[str] = Field(
        default_factory=list,
        description=(
            "Values of `field` deliberately given no slot — an absence (a lever a configuration never set) drawn in "
            "the neutral that carries no identity, rather than in a hue that would read as a value."
        ),
    )


class ChartReference(EvalBaseModel):
    """A standard drawn across one axis — the bar the marks are read against, not one of them."""

    axis: str = Field(min_length=1, description="The axis the standard is drawn across.")
    value: float = Field(description="Where it sits on that axis, in the axis's unit.")
    label: str = Field(min_length=1, description="What the standard is, in words.")


class ChartColumn(EvalBaseModel):
    """One column of the values-as-drawn table."""

    key: str = Field(min_length=1, description="The key in each row of `rows`.")
    header: str = Field(min_length=1, description="The column's header, carrying its unit where it has one.")


class ChartIntent(EvalBaseModel):
    """One chart's intent: what it draws and what it must say, for any renderer to draw.

    See the module docstring for the contract. A renderer may read ``payload`` for what a summary
    table cannot hold — the raw observations a distribution draws as density — and reads everything
    else from the fields below, which is what makes two renderers draw one chart.
    """

    intent_version: Literal[1] = Field(default=INTENT_VERSION, description="This shape's version.")
    type: ChartType = Field(description="Which of eval's chart types this is.")
    title: str = Field(min_length=1, description="The chart's title.")
    payload: dict[str, Any] = Field(
        description="The type's validated data (the payload model for `type`), as JSON — the source the rest was decided from."
    )
    scale: float = Field(
        gt=0.0, description="The factor taking the payload's values into the unit drawn (1 when nothing was restated)."
    )
    unit: str = Field(
        default="", description="The unit the chart's primary quantity is drawn in; empty when it has none."
    )
    data: list[dict[str, Cell]] = Field(
        description="The marks' values, one row per mark the chart places by value, in drawn order and drawn units."
    )
    encodings: list[ChartEncoding] = Field(description="What each field of `data` encodes.")
    axes: list[ChartAxis] = Field(description="The rulers the chart measures against.")
    identity: ChartIdentity | None = Field(
        default=None,
        description="What the rows are named by and the order they are drawn in; None where no field names rows.",
    )
    colours: list[ChartColours] = Field(
        default_factory=list, description="Every colour scheme the chart uses. Empty for a chart drawn in one ink."
    )
    references: list[ChartReference] = Field(
        default_factory=list, description="Standards drawn across an axis — a quality bar the marks are held to."
    )
    shapes: dict[str, str] = Field(
        default_factory=dict,
        description="A `class` field's values → the geometric symbol each is drawn as (`circle`, `square`, `diamond`, `cross`).",
    )
    direct_labels: bool = Field(
        description="Whether every mark carries its own identity as text beside it, so no key is needed to read it."
    )
    intervals: str = Field(
        default="",
        description="What the chart's intervals are and what they span, as one statement; empty when it has none.",
    )
    footnote: str = Field(
        default="",
        description="A note the figure carries with it — what it could not draw, in one sentence; empty when none.",
    )
    columns: list[ChartColumn] = Field(description="The values-as-drawn table's columns, in display order.")
    rows: list[dict[str, Cell]] = Field(description="The values-as-drawn rows, in drawn order, keyed by column key.")
    caption: ModelProse = Field(
        default="", description="The analysis author's own line beside the chart, exactly as written; empty when none."
    )
    disclosures: list[str] = Field(
        default_factory=list,
        description="What the chart must tell a reader that the author could not — one idea per line, in reading order.",
    )

    def parsed(self) -> Any:
        """The payload as its type's model — what a renderer reads raw observations from.

        Returns:
            The validated payload model.

        Raises:
            PayloadError: The payload no longer validates, or the type has no payload model.
        """
        parsed = parse_payload(self.type, self.payload)
        if parsed is None:
            raise PayloadError(f"no payload model for viz type {self.type!r}")
        return parsed

    def encoding(self, field: str) -> ChartEncoding | None:
        """The encoding declared for ``field``, or None."""
        return next((encoding for encoding in self.encodings if encoding.field == field), None)

    def axis(self, name: str) -> ChartAxis | None:
        """The axis named ``name``, or None."""
        return next((axis for axis in self.axes if axis.name == name), None)

    def values_as_drawn(self) -> list[str]:
        """Render the values table as aligned text, for a surface that cannot draw.

        Returns:
            A header line followed by one line per row, columns padded to a common
            width, or an empty list when there is nothing to draw.
        """
        from threetears.evals.analysis.viz.quantities import render_cell

        if not self.rows:
            return []
        cells = [[render_cell(row.get(column.key)) for column in self.columns] for row in self.rows]
        widths = [
            max(len(column.header), *(len(row[index]) for row in cells)) for index, column in enumerate(self.columns)
        ]
        lines = ["  ".join(column.header.ljust(widths[index]) for index, column in enumerate(self.columns))]
        lines.extend("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip() for row in cells)
        return lines


def chart_intent(viz_type: str, payload: dict[str, Any]) -> ChartIntent:
    """Decide a finding's chart: validate its payload, build its intent, and hold it to the policy rules.

    The one entry point. Every surface that shows a chart — the report, a renderer, the generator's
    check that a proposed chart can be drawn — goes through here, so a chart that passes once passes
    everywhere.

    Args:
        viz_type: The ``Viz.type`` discriminator.
        payload: The open payload dict, as stored or generated.

    Returns:
        The intent, its caption the payload author's own line.

    Raises:
        PayloadError: The payload is malformed, or its type has no intent builder.
        IntentPolicyError: The intent breaks a presentation rule.
    """
    # Deferred for the reason the Vega renderer's `draw_intent` defers its arms: every builder imports this module's
    # model, so a top-level import of the registry would close a cycle.
    from threetears.evals.analysis.viz.intents import INTENTS
    from threetears.evals.analysis.viz.policy import enforce_intent

    parsed = parse_payload(viz_type, payload)
    build = INTENTS.get(viz_type)
    if parsed is None or build is None:
        raise PayloadError(f"no chart intent for viz type {viz_type!r}")
    intent = build(parsed)
    # The author's line is carried as written and never extended; what the engine has to add travels
    # as `disclosures`. The strip is the payload model's own stance restated for a caption of no words.
    intent = intent.model_copy(update={"caption": (parsed.caption or "").strip()})
    enforce_intent(intent)
    return intent


__all__ = [
    "INTENT_VERSION",
    "INTERVAL_ROLES",
    "LENGTH_ROLES",
    "MEASURED_ROLES",
    "Cell",
    "ChartAxis",
    "ChartColours",
    "ChartColumn",
    "ChartEncoding",
    "ChartIdentity",
    "ChartIntent",
    "ChartReference",
    "ChartType",
    "EncodingRole",
    "chart_intent",
]

"""Draw a chart intent as a Vega-Lite spec — the first renderer of eval's chart intent.

**A renderer, not a decider.** What a chart says — its order, its units, its values as drawn, what
it must disclose — arrives decided in its :class:`~threetears.evals.analysis.viz.intent.ChartIntent`;
this module turns that into a Vega-Lite spec and adds nothing a reader is told. One spec, two
consumers: the browser embeds it and the server rasterises it (:mod:`threetears.evals.vega.render`)
for surfaces that cannot run a browser, so layout computed here — label placement, axis domains — is
identical on both. This adapter (with its arms, palette, text metrics, rasteriser and spec gate) is
optional — the ``[vega]`` extra — and nothing in the core imports it.

**The spec carries no colour.** Every consumer supplies the palette as a
Vega-Lite ``config``: the browser reads it from the CSS custom properties, so it
cannot drift from the design tokens, and the server reads resolved sRGB hex from
the generated palette artifact, because its rasteriser cannot parse the OKLCH the
tokens are authored in. A colour literal compiled into the spec would defeat both.

**Identity never rides on colour here.** Every chart puts its categories on an
axis instead, which is not a stylistic preference. The palette never refuses to
draw — the number of series is a property of the data — so past its four validated
hues it takes a derived second tier and past eight it recycles, and a
colour-per-category chart therefore has a point beyond which two categories look
alike. An axis has no such point. What keeps recycling honest is therefore that
IDENTITY has left the hue channel before the hue channel weakens — not the absence
of colour, which is a stronger claim than the rule needs and is no longer true.
One arm colours a swept dimension's LEVEL, which is not the row's
identity and is capped by its own type's rules. No compiled spec carries a colour
VALUE anywhere; that half stayed absolute.

**Which makes the category NAMES load-bearing, so where they are drawn is decided
here rather than left to the renderer.** They are full model IDs with no aliases,
the gutter that holds them is a fixed 176px, and a renderer handed a name too long
for that will truncate it — eating precisely the characters that say which build
is being looked at. So the compiler measures
(:mod:`threetears.evals.vega.text_metrics`), strips any prefix every series shares,
and moves the names out of the gutter onto their own line when they still will not
fit. The decision is baked into the spec, which is what keeps the two surfaces
drawing one layout rather than each reaching its own conclusion about the same
string.

**What is here is the machinery every chart shares; what draws one TYPE is not.**
:func:`draw_intent` is the entry point for an intent and :func:`compile_chart` for a stored payload,
and both resolve the chart's type through :data:`~threetears.evals.vega.arms.ARMS` — one
module per type under :mod:`threetears.evals.vega.arms`, each importing this one and
none importing a sibling. So a type's shape is a file rather than a branch, and
adding one touches nothing another type is drawn by. The value axis, the identity
axis, the label placement, the title bound and the number formatting stay here
precisely because they are what makes two different charts read as one system.

The arms are inside the boundary, not outside it, and they import that machinery by
name. Those names keep their leading underscore because the package as a whole
exports one function and they are not part of that surface; within the package they
are an ordinary intra-package contract. The underscore therefore marks the boundary
of :mod:`threetears.evals.vega`, not of this file, and an arm reaching for
``ValueAxis`` is using the seam as designed rather than around it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, TypedDict

from threetears.evals.vega.palette import (
    VALUE_ON_FILL_STYLE,
    ZERO_RULE_STYLE,
    font_sizes,
    font_weights,
    geometry,
)
from threetears.evals.analysis.viz.intent import ChartAxis, ChartIdentity, ChartIntent, chart_intent
from threetears.evals.analysis.viz.payloads import PayloadError
from threetears.evals.analysis.viz.quantities import strip_common_prefix
from threetears.evals.vega.spec_policy import enforce_spec
from threetears.evals.vega.text_metrics import fits, text_width, wrap_text

#: The Vega-Lite schema the specs declare. Pinned rather than tracked: the spec is
#: the durable artifact, and a stored analysis must draw the same chart next year.
VEGA_LITE_SCHEMA = "https://vega.github.io/schema/vega-lite/v6.json"

#: Opacity for marks that ACCUMULATE — raw sample dots, the overlap band, a bound
#: whose coverage was never recorded. Low enough that overlapping marks read as
#: density.
#:
#: **Not a way to make one mark quieter than another.** That is emphasis, it is a
#: judgement about the surface being drawn on, and the same alpha is not the same
#: recession on a near-black background and on a pale one — so a compiled opacity
#: chosen for emphasis is right on at most one of the two surfaces a stored spec is
#: rendered onto. A mark that carries no identity asks for
#: :data:`~threetears.evals.vega.palette.CONTEXT_STYLE` by name instead and lets each
#: renderer answer. The distinction is what this number means, not where it happens
#: to be used: alpha here says *these marks pile up* or *this bound is unknown*,
#: which is a fact about the data and is identical in both themes.
SECONDARY_OPACITY = 0.35

#: The row key every compiled spec's identity axis encodes.
#:
#: Deliberately not the payload's own label field. What a category is CALLED and
#: what is DRAWN beside its mark are different strings once a shared prefix is
#: stripped, and keeping them in separate keys is what lets the picture be short
#: while the tooltip and the values table stay complete — the same row carries
#: both, so neither can be reconstructed wrongly from the other.
DISPLAY_FIELD = "display"

#: The row key a value-label layer positions its text against.
#:
#: Its own key rather than the plotted row's value field, because the two are not
#: always the same number: a bar's label is written at the bar's END, which is the
#: value it states, while an interval's estimate label (``distribution``,
#: ``null_result``) is written at the MEAN and lifted above a mark that spans much
#: further.
#:
#: It once read "an interval's is written at the end of the interval and READS the centre",
#: which described a defect rather than a design — a number drawn at
#: an x-position it does not name.
ANCHOR_FIELD = "at"

#: The row key holding a value already rendered for drawing.
#:
#: Rendered by the compiler rather than by a Vega ``format``, for the same reason
#: the sort order is: the two renderers must draw the same characters, and a format
#: string is resolved by whichever engine is running.
VALUE_TEXT_FIELD = "text"

#: The clear gap between a value label and the EDGE of the mark it labels, in px.
#:
#: Small enough to stay attached to its own mark, which is the whole reason the number
#: is not in a legend.
#:
#: **From the mark's edge, not from the value's position.** Where the mark ENDS at its
#: value — a bar's tip — the two are the same distance and this constant is the whole
#: offset. Where the mark has a BODY around it — a point, which is centred on the value
#: it names — the body eats the gap: at the shipped sizes a point's radius is 4.5px
#: (``delta_table``) to 5.5px (``sweep_ranking``), so measuring from the centre left
#: 1.5px and 0.5px of daylight and the label rendered touching its own mark. The body
#: is :attr:`MarkValue.radius` and the sum is :attr:`Placement.clearance`.
VALUE_LABEL_OFFSET = 6

#: The row key naming which of a composed frame's marks a row belongs to.
#:
#: A faceted figure partitions ONE table by the facet field and hands each cell its
#: own slice, so every mark in the cell has to draw from that slice — a layer with
#: `data` of its own is not faceted and would draw its whole dataset in every cell.
#: The marks therefore share one table and each layer filters it to its own kind.
KIND_FIELD = "kind"

#: The row key holding how far a marginal's bin rises from its row's floor, in px.
#:
#: Pixels rather than a count, because the counts are shared across rows: a bin
#: drawn taller in one row than another has to mean a larger count rather than a
#: smaller neighbour, and normalising per row would draw a five-observation cohort
#: at the same height as a five-hundred-observation one.
RISE_FIELD = "rise"

#: Axis tick length in px, for an axis whose ticks replace its gridlines.
#:
#: Long enough to locate a mark against the rule, short enough not to compete with
#: one. Stated rather than left to Vega's own default so both renderers draw the
#: same furniture from the spec.
_FRAMED_TICK_SIZE = 5


def point_radius(size: float) -> float:
    """How far a point of ``size`` px² extends past the value it is centred on, in px.

    The one place a Vega point's area becomes a distance. Every arm that draws a point
    needs the number twice — once to keep the outermost mark inside the plot, once to
    hold its label off the mark's edge — and an arm that did the arithmetic itself
    would be a second answer free to disagree with this one.

    **``size`` is the area of the symbol's BOUNDING BOX, not of the circle**, which is
    Vega's own convention and not d3's: ``vega-scenegraph`` draws the circle at
    ``sqrt(size) / 2``, where ``d3.symbolCircle`` would draw ``sqrt(size / pi)``.
    Taking the d3 reading gives a radius 13% too large, and the arithmetic looks right
    on the page — ``tests/test_vega_render.py`` measures the radius the
    rasteriser actually drew, because that is the only thing that settles it.

    Args:
        size: The point mark's ``size``, in px².

    Returns:
        The mark's radius in px — half its drawn extent along either axis.
    """
    return math.sqrt(size) / 2


def plot_size(
    rows: int, *, marginal: bool = False, label_above: bool = False, value_above: bool = False
) -> tuple[int, int]:
    """The plot area for a chart of ``rows`` categorical rows, in px.

    Height is driven by the row count and width is fixed, which is the only
    ordering that produces a readable chart: sizing width to the container and
    height per row gives three bars 12px tall in a 1400px plot, because the two
    dimensions are then answering different questions.

    The floor stops a two- or three-row chart from drawing as a letterbox. The
    aspect ceiling is the same guard expressed as a proportion, and with the
    current floor it never binds — 168 × 6 already exceeds the plot width, so the
    floor decides every case. It is applied anyway because it is the constraint
    that has to hold if the floor ever drops, and a chart that silently letterboxes
    is exactly the defect neither number is allowed to reintroduce.

    Two things ask for a taller row than the default, and a row can want both at
    once — a spread whose cohorts carry marginals *and* whose names outrun the
    gutter. The step is the largest any of them asks for rather than a sum: each
    number is a statement of how much room that row needs, and a row given the
    taller of two requirements satisfies both.

    Args:
        rows: How many categorical rows the chart draws.
        marginal: Whether each row also carries a marginal, which needs the taller step.
        label_above: Whether the row's label is drawn on its own line inside the
            plot rather than in the gutter, which needs a taller step again.
        value_above: Whether the row's VALUE is lifted onto a line above its mark —
            a value anchored inside an interval, where the mark itself occupies the
            row's centreline. The same line of text as ``label_above`` asks room for,
            so it asks for the same step.

    Returns:
        ``(width, height)`` of the plot area — Vega-Lite's ``width``/``height``,
        which exclude the axis labels drawn in the gutter beside them.
    """
    sizes = geometry()
    steps = [sizes["row_step"]]
    if marginal:
        steps.append(sizes["row_step_marginal"])
    if label_above or value_above:
        steps.append(sizes["row_step_label_above"])
    height = max(rows * max(steps), sizes["plot_min_height"])
    return min(sizes["plot_width"], height * sizes["aspect_max"]), height


def _bar_mark(**overrides: Any) -> dict[str, Any]:
    """A horizontal bar at the fixed row thickness, square at both ends.

    **Square, not rounded.** A rounded data-end places the bar's tip short of its
    value by the radius and rounds the zero end into a shape the data does not
    have; on a short bar the radius is a meaningful share of the length, so the
    corner is a systematic under-draw of exactly the values that can least afford
    one.

    Thickness is stated in px rather than as a fraction of the band, because the
    band varies with the floor — three rows in a 168px plot get a 56px band and
    ten rows get 48 — and a bar whose thickness moves with the row count encodes
    the row count in the mark.

    Args:
        **overrides: Mark properties merged over the base, for the marks that
            differ (the dumbbell's thin connector states its own height) and for
            the per-mark tooltip, which belongs to whichever layer carries the
            values rather than to every bar by default.

    Returns:
        The Vega-Lite mark definition.
    """
    return {"type": "bar", "height": geometry()["bar_height"]} | overrides


class CompiledColumn(TypedDict):
    """One column of the values-as-drawn table, as the compilation hands it on.

    ``header`` states the unit where the column has one, because this table is the
    whole of what a surface that cannot draw receives — a bare number in it is the
    same defect as an unlabelled axis.
    """

    key: str
    header: str


@dataclass(frozen=True)
class CompiledChart:
    """One finding's chart: its intent, and the Vega-Lite spec drawn from it.

    Everything but ``spec`` is the intent's, carried across unchanged so a caller holding a
    compilation reads the same values table, caption and disclosures the intent decided — the
    compiler adds a picture and says nothing of its own.

    Attributes:
        intent: What the chart says (:class:`~threetears.evals.analysis.viz.intent.ChartIntent`).
        spec: The Vega-Lite spec, carrying no colour.
        columns: The values-as-drawn table's columns, in display order.
        rows: That table's rows, keyed by column key, in drawn order.
        unit: The unit the chart's primary quantity is drawn in, after any restatement.
        caption: The payload's own editorial line, exactly as its author wrote it, or ``""``.
        disclosures: What the chart has to tell the reader that the author could not, one idea
            per line, in reading order — kept apart from ``caption`` so a reader can tell where
            the author stopped.
        title: The chart's own title.
    """

    intent: ChartIntent
    spec: dict[str, Any]
    columns: list[CompiledColumn] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    unit: str = ""
    caption: str = ""
    disclosures: list[str] = field(default_factory=list)
    title: str = ""

    def values_as_drawn(self) -> list[str]:
        """Render the values table as aligned text, for a surface that cannot draw.

        Returns:
            A header line followed by one line per row, columns padded to a common
            width, or an empty list when there is nothing to draw.
        """
        return self.intent.values_as_drawn()


def compile_chart(viz_type: str, payload: dict[str, Any]) -> CompiledChart:
    """Compile a finding's viz into a Vega-Lite spec, through the chart's intent.

    The intent is decided first (:func:`~threetears.evals.analysis.viz.intent.chart_intent` — the
    payload validated, the chart's claims held to the intent policy), then drawn by the type's arm,
    then the spec is held to this renderer's own gate.

    Args:
        viz_type: The ``Viz.type`` discriminator.
        payload: The open payload dict, as stored or generated.

    Returns:
        The compiled chart.

    Raises:
        PayloadError: The payload is malformed, or its type has no compiler.
        IntentPolicyError: The chart's intent breaks a presentation rule.
        SpecPolicyError: The compiled spec breaks a rendering rule. Only the spec is gated here:
            the caption and every disclosure line are served as text and never reach the
            rasteriser, and what authored text CLAIMS is never checked (prose is judged by the
            reporter eval, not refused by code). Callers treat the raise as a data problem.
    """
    return draw_intent(chart_intent(viz_type, payload))


def draw_intent(intent: ChartIntent) -> CompiledChart:
    """Draw a decided chart intent as a Vega-Lite spec — this renderer's one entry point.

    Args:
        intent: The chart's intent.

    Returns:
        The compiled chart.

    Raises:
        SpecPolicyError: The compiled spec breaks a rendering rule.
    """
    # Imported inside the call rather than at module scope, and that is structural: every arm
    # imports this module's shared machinery, so a top-level import here would close a cycle.
    from threetears.evals.vega.arms import ARMS

    # Every chart type has an arm — `ARMS` is held to the type vocabulary key for key by test — so a type
    # with none is a registration mistake, not a data problem, and raises as one.
    spec = ARMS[intent.type](intent)
    enforce_spec(spec)
    return CompiledChart(
        intent=intent,
        spec=spec,
        columns=[{"key": column.key, "header": column.header} for column in intent.columns],
        rows=[dict(row) for row in intent.rows],
        unit=intent.unit,
        caption=intent.caption,
        disclosures=list(intent.disclosures),
        title=intent.title,
    )


def _identity(intent: ChartIntent) -> ChartIdentity:
    """The intent's identity, which every arm that draws categories needs.

    Raises:
        PayloadError: The intent names no identity, so there is nothing to put on the axis.
    """
    if intent.identity is None:
        raise PayloadError(f"a {intent.type} chart intent names no identity to draw its rows by")
    return intent.identity


def _value_axis(intent: ChartIntent, name: str) -> ChartAxis:
    """One of the intent's axes, by name.

    Raises:
        PayloadError: The intent declares no such axis.
    """
    axis = intent.axis(name)
    if axis is None:
        raise PayloadError(f"a {intent.type} chart intent declares no {name!r} axis")
    return axis


def _number(value: Any) -> float:
    """A data cell the intent placed as a number, as a float.

    Raises:
        PayloadError: The cell is not a number — the intent and the arm disagree about the field.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PayloadError(f"expected a number where the chart places a value, got {value!r}")
    return float(value)


def _name_font_size() -> float:
    """The size a category name is drawn at, wherever it is drawn.

    One function rather than a literal in each place, because three of them have
    to agree or the layout is decided from a number nothing draws at: the width
    measured to choose a placement, the size the identity axis states, and the size
    the label-above-mark layer sets. The type scale's ``label`` step is the name
    step — ``tick`` is for numeric ticks, which is a different role at the same
    size, and a size shared today is not a size shared.
    """
    return font_sizes()["label"]


@dataclass(frozen=True)
class _Categories:
    """One figure's identity axis: what its categories are, and where their names sit.

    The names are the identity channel for every chart here — the one arm that does
    emit a colour encoding (`sweep_ranking`, on a swept LEVEL) is colouring
    something that is not the row's identity — so where these are drawn is a
    correctness question rather than a styling one. Two placements, decided once per
    figure and baked into the spec so both renderers draw the same one:

    * **In the gutter**, the default, bounded by the 176px the figure reserves for
      it. Bounded because the figure is the gutter plus the plot plus the right
      margin, so an unbounded label column widens the figure by however long the
      longest category name happens to be — the data deciding the layout.
    * **Above the mark**, when a name will not fit that bound even with the shared
      prefix stripped. The name takes its own line at the plot's left edge, the
      gutter disappears, and the row step grows. Never truncated and never shrunk:
      vertical space is cheap and a truncated model ID is not, since the characters
      a truncation eats are the ones that distinguish one build from another.

    The decision is measured, not judged — see
    :mod:`threetears.evals.vega.text_metrics` — and it is measured at the size the
    renderer will actually draw the axis label at, which is the coupling
    ``test_vega_compiler.py`` pins.
    """

    field: str
    """The row key carrying the full identity — what the tooltip and table read."""

    ordering: tuple[str, ...]
    """Every category, in drawn order, by full identity."""

    display: dict[str, str]
    """Full identity → drawn name."""

    above: bool
    """Whether the names are drawn above their marks rather than in the gutter."""

    @classmethod
    def of(cls, field: str, ordering: Sequence[str]) -> _Categories:
        """Decide one figure's labels from the categories it draws.

        Args:
            field: The row key holding each category's full identity.
            ordering: The categories, in drawn order.

        Returns:
            The resolved labels and placement.
        """
        display = strip_common_prefix(ordering)
        gutter = geometry()["gutter_left"]
        size = _name_font_size()
        return cls(
            field=field,
            ordering=tuple(ordering),
            display=display,
            above=not all(fits(display[label], size, gutter) for label in ordering),
        )

    def drawn(self) -> list[str]:
        """The drawn names, in drawn order."""
        return [self.display[label] for label in self.ordering]

    def labelled(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Stamp each plotted row with the name to draw beside it.

        Args:
            rows: Plotted rows, each carrying :attr:`field`.

        Returns:
            The same rows, each gaining :data:`DISPLAY_FIELD`.
        """
        return [row | {DISPLAY_FIELD: self.display[row[self.field]]} for row in rows]

    def axis(self, *, domain: bool = True) -> dict[str, Any]:
        """The identity encoding: the compiled order, never a Vega sort.

        The same order has to reach the text rendering of the chart, and a sort
        resolved inside Vega would exist only in the picture.

        Args:
            domain: Whether to draw the axis's own domain line. Passed ``False``
                where the marks already draw it — bars measured from a zero that
                sits at the plot's edge start exactly where this line does, so
                keeping both puts furniture underneath data.

        Returns:
            The Vega-Lite channel definition.
        """
        encoding: dict[str, Any] = {"field": DISPLAY_FIELD, "type": "nominal", "sort": self.drawn()}
        if self.above:
            # No axis at all, rather than an axis with its labels switched off: an
            # axis still reserves its gutter, and reclaiming that gutter for the plot
            # is half of what this placement is for.
            encoding["axis"] = None
        else:
            encoding["axis"] = {
                "title": None,
                "labelLimit": geometry()["gutter_left"],
                # The NAME step, stated on the axis rather than taken from the
                # config's blanket default, which is the numeric-tick step. The two
                # share a size today and not a weight, and a config keyed by channel
                # cannot tell a name from a tick — one figure here draws names on y
                # and bin ranges on x. Stating it is also what makes the placement
                # measurement (`of`) provably the size the label draws at.
                "labelFontSize": _name_font_size(),
                "labelFontWeight": font_weights()["label"],
                "domain": domain,
            }
        return encoding

    def label_layer(self, *, clearance_above: float | None = None) -> dict[str, Any] | None:
        """The text layer that draws the names, when they are not in the gutter.

        Args:
            clearance_above: How far above the row's centreline the row's own drawing
                reaches, in px — the name is set clear of that. Half a bar by default,
                the tallest MARK a row carries; an arm that lifts a value label onto
                its own line above the mark states the top of that line instead, so
                the name stacks above the number rather than being drawn through it.

        Returns:
            A layer to compose over the marks, or ``None`` in gutter placement.
        """
        if not self.above:
            return None
        sizes = geometry()
        reach = sizes["bar_height"] / 2 if clearance_above is None else clearance_above
        return {
            "data": {"values": [{DISPLAY_FIELD: name} for name in self.drawn()]},
            "mark": {
                "type": "text",
                "align": "left",
                "baseline": "bottom",
                # Clear of the row's own drawing, inside the row's band. Sized from
                # the bar by default because it is the tallest mark a row carries; an
                # interval's rule sits well inside the same clearance, and an arm whose
                # value label rides above its mark says how high that label reaches.
                "dy": -(reach + 6),
                # The type scale's name step. The config's `text` style is set for
                # values written ON a mark, which are a step larger — a category name
                # is not a value.
                "fontSize": _name_font_size(),
                "fontWeight": font_weights()["label"],
            },
            # x in PLOT pixels, so the name starts where the plot does. It sits on its
            # own line above the mark, so it cannot collide with one.
            "encoding": {"y": self.axis(), "x": {"value": 0}, "text": {"field": DISPLAY_FIELD, "type": "nominal"}},
        }

    def facet_header(self) -> dict[str, Any]:
        """A facet row header carrying the same placement decision.

        A faceted panel names its rows in a header rather than on an axis, but the
        question is the same one and it gets the same answer — otherwise a figure
        whose value panel moved its names out of the gutter would keep a gutter for
        its marginal, and the two panels of one figure would stop sharing a left
        edge.

        Returns:
            The ``header`` block for a row facet.
        """
        # `labelAngle` is stated in both placements, and it is not decoration: a row
        # header's Vega-Lite default is a quarter turn, which is how this figure drew
        # its category names sideways in the one panel that used a facet while the
        # panel above it drew the same names flat.
        if self.above:
            # Above the panel rather than beside it, which is the facet's form of the
            # same placement — and being unbounded is safe here precisely because the
            # header no longer competes with the plot for width.
            return {"title": None, "orient": "top", "labelAnchor": "start", "labelLimit": 0, "labelAngle": 0}
        return {
            "title": None,
            "labelLimit": geometry()["gutter_left"],
            "labelAngle": 0,
            # Anchored LEFT, because Vega sizes each facet cell's header column to
            # its own text. Ranged right — the way an axis label is — every row's
            # label then ends at its own column's right edge, so two cohorts whose
            # names differ in length sit at two different x positions and the column
            # reads as though one of them were indented. Anchoring at the start puts
            # every name on one left edge, which is the alignment a reader takes as
            # "these are the same kind of thing".
            "labelAlign": "left",
            "labelAnchor": "start",
        }

    def plot_size(self, *, marginal: bool = False, value_above: bool = False) -> tuple[int, int]:
        """The plot area for this figure, in px.

        Args:
            marginal: Whether each row also carries a marginal.
            value_above: Whether each row's value is lifted onto a line above its mark.

        Returns:
            ``(width, height)`` of the plot area.
        """
        return plot_size(len(self.ordering), marginal=marginal, label_above=self.above, value_above=value_above)

    def figure_width(self) -> float:
        """How wide the whole figure draws, gutter included.

        Returns:
            The figure's width in px — the bound a title has to wrap inside.
        """
        sizes = geometry()
        gutter = 0 if self.above else sizes["gutter_left"]
        width: float = gutter + sizes["plot_width"] + sizes["gutter_right"]
        return width


def _title_spec(title: str, limit: float, footnote: str = "") -> str | list[str] | dict[str, Any]:
    """A chart title bounded by the figure it sits above, and its footnote.

    The title is the payload's own measure name — generator prose, unbounded in
    length, and the one input the figure's fixed geometry does not bound: the
    gutter caps the category names and the plot caps everything drawn in it, but
    Vega grows the frame to fit a title, so a long measure name pushes the figure
    past the report's column while every other input stays inside it.

    The label rules' answer for a label that will not fit is the answer here too: never
    truncate and never shrink, so it **wraps**. Vega-Lite reads an array of strings
    as a multi-line title, which is what makes that expressible without a second
    mark.

    A title that already fits is returned unchanged rather than as a one-element
    array, so a spec only gains the array shape when it needs it — and its exact
    whitespace survives, which wrapping does not preserve. The object form is the
    same idea one level up: it appears only where there is a footnote to carry.

    **The footnote is bounded the same way and for the same reason.** It is drawn
    by Vega as the title's subtitle, so an unwrapped one grows the frame exactly as
    an unwrapped title does — and a disclosure that pushes the figure out of its
    column would be a rule paid for by the rule beside it.

    Args:
        title: The chart's title.
        limit: The figure width to wrap inside, in px.
        footnote: A disclosure drawn under the title — what the chart did to its
            axis, in the picture rather than only beside it. Defaults to none.

    Returns:
        The title, as a string when it fits on one line, as one entry per line when
        it does not, and as a title object when it carries a footnote.
    """
    lines = wrap_text(title, font_sizes()["title"], limit)
    text: str | list[str] = title if len(lines) <= 1 else lines
    if not footnote:
        return text
    subtitle = wrap_text(footnote, font_sizes()["footnote"], limit)
    return {"text": text, "subtitle": footnote if len(subtitle) <= 1 else subtitle}


def _composed(view: dict[str, Any], categories: _Categories, *, clearance_above: float | None = None) -> dict[str, Any]:
    """Add the category names to a chart's marks, where the placement calls for it.

    In gutter placement the names ride on the axis and there is nothing to compose,
    so the view is returned untouched. Above the mark they are a text layer, and a
    chart drawing a single mark has to become a layered one to carry it — done here
    rather than in each arm so that a shape drawn as one mark and a shape drawn as
    five gain their labels the same way.

    Args:
        view: The assembled view — a single mark or an existing ``layer``.
        categories: The figure's resolved labels and placement.
        clearance_above: How far above each row's centreline its drawing reaches, in
            px, per :meth:`_Categories.label_layer`; ``None`` for the default.

    Returns:
        The view, layered with its label text where that placement applies.
    """
    layer = categories.label_layer(clearance_above=clearance_above)
    if layer is None:
        return view
    if "layer" in view:
        return view | {"layer": [*view["layer"], layer]}
    marks = {key: view[key] for key in ("mark", "encoding") if key in view}
    # `data` deliberately stays at the outer level: Vega-Lite hands it down to the
    # layer that has none, and the label layer carries its own.
    outer = {key: value for key, value in view.items() if key not in ("mark", "encoding")}
    return outer | {"layer": [marks, layer]}


@dataclass(frozen=True)
class MarkValue:
    """One mark's value, and where the mark it belongs to ends.

    Attributes:
        display: The drawn category name, which places the label on its own row.
        end: The value the mark's far end reaches, in the axis's own units — where
            the label is written, which is not always the value it states.
        text: The value as drawn, already rendered.
        radius: How far the mark's own body reaches past ``end``, in px — see below.
        filled: Whether the mark paints a fill BENEATH its own label when the label
            is placed inward. Stated by the arm, never inferred here — see below.
        thickness: How far that same body reaches ACROSS the row, in px — see below.
    """

    display: str
    end: float
    text: str

    radius: float = 0
    """How far the mark's body extends past the value the label is anchored at, in px.

    **The arm knows and the compiler cannot, for the same reason as ``filled``.** What
    a label has to clear is the mark's EDGE, and where that edge is depends on the
    shape drawn at ``end``: a bar and an interval rule END there, so their edge IS the
    value and this is zero; a POINT is centred there, so its edge is a radius away and
    a label set the bare gap from the value is set most of the way into the mark. It is
    stated in px because a point is sized in px² — :func:`point_radius` is the
    conversion, and an arm that inlined it would own a second copy of Vega's
    bounding-box convention.

    Zero by default so the length arms — ``breakdown``, ``attribution`` — need not
    restate that a bar's tip is where the bar's value is. **A bar is not a point:**
    giving one clearance it does not need would push every bar chart's numbers 4-6px
    off the tips they name, for a gap the reader already has. An integral zero, so a
    mark with no body emits the offset the constant states rather than the same number
    respelled as a float — the compiled spec is a committed snapshot on the browser
    side, and a spelling change there is a diff nobody can tell from a moved label.
    """

    filled: bool = True
    """Whether a label placed inward would land on this mark's fill.

    **The arm knows and the clearance arithmetic does not, which is why this is a
    field rather than a derivation.** A bar has length under its inward label, so a
    value written there is on velvet and needs the knockout ink. A POINT, a bare
    interval rule or a dumbbell's connector does not: `sweep_ranking`'s ranking panel
    draws points, and `delta_table` draws a 3px connector under a point — inward of
    those marks there is nothing but the chart surface.

    Inferring it from ``room(end) < needed`` conflates "no space outside" with "there
    is a fill here", and on a cropped position axis those come apart: the outermost
    mark sits ``domain_pad/(1 + 2*domain_pad)`` of the plot from the edge — about 57px
    at the shipped padding — against a ``needed`` of at least the 56px clearance, so a
    label wider than about 51px flips to inward. ``format_number`` writes
    ``1.235e-05`` at about 64px and a seven-digit whole number at about 57px, so one value
    under 1e-4, or of a million or more, is enough.
    That label would then ask for the knockout, which IS each mode's chart surface —
    so the number would be painted on the background in the background's own colour
    and vanish at 1:1. Defaulting to ``True`` keeps the length arms correct without
    each restating the obvious; the arms that draw no fill say so.

    **It does not say the label has clear surface under it.** A mark that paints no
    fill may still draw a LINE along the row — see ``thickness``, which is the half of
    that question this flag does not answer.
    """

    thickness: float = 0
    """How far the mark's body extends across the row, in px, at the label's own side.

    **The second half of what an inward label lands on, and ``filled`` answers only
    the first.** ``filled`` decides the label's INK; this decides its POSITION. A
    dumbbell's connector and an interval's rule are 3px under a glyph band of about
    ten, so they are far too thin to knock a number out of — and far too solid to
    write one through: drawn on the row's own centreline, the line runs across the
    numerals from end to end and reads as a strikethrough. A label pushed inward over
    one is therefore lifted clear of it, by :func:`_label_lift`.

    Zero where the mark draws nothing along the row between the value and the
    baseline — ``sweep_ranking``'s bare points — and zero for the length arms, whose
    bar is a fill and takes the knockout instead. Stated by the arm for the same
    reason ``radius`` is: what is drawn at a value is the arm's own fact, and a
    compiler inferring it would be a second answer free to disagree.
    """


@dataclass(frozen=True)
class ValueAxis:
    """The quantitative axis a chart's values are drawn against.

    Two kinds, and which one a chart has is a correctness question rather than a
    styling one:

    * A **magnitude** axis carries a length measured from zero — a bar. Its
      baseline is not negotiable, because the length IS the quantity and a
      truncated baseline draws a ratio the numbers do not contain. It therefore
      never crops.
    * A **position** axis carries a location — an interval, a point. Forcing zero
      onto one destroys the resolution the chart exists to show: two spreads 6 s
      apart on a 0-60 s axis draw identically. So it crops to its data, and its
      labelled ticks are what disclose where it starts — there is no footnote;
      see the comment below on why the sentence was withdrawn.

    **The domain is resolved here rather than left to the renderer**, and that is
    what makes the rest of this class possible: where a mark ENDS in pixels is what
    decides whether its value is written outside it or on it, and a domain Vega
    computes is a number this process does not have. It also keeps the two
    renderers drawing one picture, which is the same reason every size is compiled.
    ``nice`` is off for that reason — it would move the domain after the arithmetic
    that read it.
    """

    title: str
    """The quantity and its unit, as every layer of the frame states it.

    Empty where the frame names the quantity in a heading above the panel instead,
    which is the only way to name a ``y`` quantity without turning the words a
    quarter turn. The axis then states nothing rather than the same word twice.
    """

    low: float
    """The domain's lower bound, in the chart's own units."""

    high: float
    """The domain's upper bound."""

    from_zero: bool
    """Whether this is a magnitude axis, whose domain must contain zero."""

    plot_span: int
    """How far the plot runs ALONG this axis in px, which is what turns a value into a position.

    The plot's width for a horizontal axis and its height for a vertical one. Named
    for the axis rather than for the figure because a point plot draws one of each,
    and a field called ``width`` holding a height is a unit error waiting to be read
    as a typo.
    """

    number_format: str = ""
    """A Vega-Lite number format for the tick labels, where the axis needs one."""

    tick_count: int = 5
    """How many labelled ticks to aim for — and, since the grid follows the ticks,
    how many gridlines a position axis draws."""

    framed: bool = False
    """Whether the axis rule is the drawn extent of the data rather than a frame edge.

    A range-framed axis takes the observed range as its domain exactly, with no
    padding, so the rule ENDS where the data ends and the extent of what was
    measured is legible without reading a caption. It is the one bright line in
    such a figure: its ticks rise from it at the labelled values and there is no
    grid, because a full-height line through a row of stacked panels competes with
    the marks it is there to help locate. Where the axis is not range-framed the
    ordinary rule holds — a position chart keeps gridlines at its labelled ticks,
    since without a baseline there is nothing else to read a mark against.
    """

    @classmethod
    def magnitude(cls, title: str, values: Sequence[float], plot_span: int, *, tick_count: int = 5) -> ValueAxis:
        """An axis for marks whose LENGTH is the quantity.

        Args:
            title: The axis title, carrying the unit, or ``""`` where the panel's
                own heading names the quantity.
            values: Every value drawn against it.
            plot_span: How far the plot runs along this axis in px.
            tick_count: How many labelled ticks to aim for.

        Returns:
            The axis, spanning zero to the furthest value on each side of it.
        """
        finite = [value for value in values if math.isfinite(value)]
        low, high = min([0.0, *finite]), max([0.0, *finite])
        if low == high:
            # Every value is zero. There is no ratio to draw either way, and a
            # zero-width domain is not a scale — one unit gives the axis a span
            # without putting any mark at the end of it.
            high = 1.0
        return cls(title=title, low=low, high=high, from_zero=True, plot_span=plot_span, tick_count=tick_count)

    @classmethod
    def symmetric(
        cls,
        title: str,
        reach: float,
        plot_span: int,
        *,
        floor: float = 0.0,
        mark_clearance: float = 0.0,
        number_format: str = "",
    ) -> ValueAxis:
        """A magnitude axis either side of zero, floored and held clear of its marks.

        For a comparison whose axis has a FLOOR: scaled only to its data, a set of
        3% changes draws exactly like a set of 300% ones, so the reach the data
        asks for competes with a stated minimum rather than deciding alone. The
        floor is also what guarantees this axis a span at all — a comparison whose
        every row is unchanged reaches zero, and a zero-width domain is not a scale.

        **The clearance is the other half of the same job, and it is why the floor
        cannot do it.** A floor answers "how small may the axis be"; it says nothing
        about the row that decides the axis, which lands exactly on the domain's end
        and is drawn there with half its body outside the plot. So the reach is
        widened by however far the outermost mark extends past its own value — the
        same padding an unframed position axis takes, stated in px rather than as a
        proportion because what has to clear the edge is a mark of a fixed size.

        The mark's size is the ARM's fact and the conversion is this class's: an arm
        that turned its own px into a domain would be a second answer to
        :meth:`offset`, free to disagree the moment the plot width moves.

        Args:
            title: The axis title.
            reach: How far the data itself reaches on each side of zero, as a
                magnitude — the largest absolute value drawn.
            plot_span: How far the plot runs along this axis in px.
            floor: The smallest half-span this axis may take, in its own units.
            mark_clearance: How much room the outermost mark needs beyond its own
                value, in px — half the width of the mark that sits on it.
            number_format: A Vega-Lite format for the tick labels.

        Returns:
            The axis.
        """
        # Solved rather than iterated: a domain wide enough to leave `mark_clearance`
        # px past `reach` is the one whose px-per-unit puts `reach` that far in from
        # the end, which is a single division. Padding by a proportion of the reach
        # instead would leave the clearance drifting with the data.
        half_span = max(floor, reach * plot_span / (plot_span - 2 * mark_clearance))
        return cls(
            title=title,
            low=-half_span,
            high=half_span,
            from_zero=True,
            plot_span=plot_span,
            number_format=number_format,
        )

    @classmethod
    def position(cls, title: str, values: Sequence[float], plot_span: int, *, framed: bool = False) -> ValueAxis:
        """An axis for marks whose POSITION is the quantity, cropped to the data.

        Args:
            title: The axis title, carrying the unit, or ``""`` where the figure's
                own heading names the quantity.
            values: Every value drawn against it.
            plot_span: How far the plot runs along this axis in px.
            framed: Whether the axis rule is the drawn extent of the data. A framed
                axis takes the observed range exactly, so nothing separates the
                outermost mark from the end of the rule; an unframed one pads, so
                the marks sit clear of the frame's edges.

        Returns:
            The axis, spanning the data plus a margin at each end unless framed.
        """
        pad_fraction = 0.0 if framed else geometry()["domain_pad"]
        finite = [value for value in values if math.isfinite(value)]
        if not finite:
            # A frame with nothing on its value axis still needs a domain, and it has
            # no data to take one from. Cropping is not a claim about values there
            # are none of, so it takes the unit interval and stays uncropped.
            return cls(title=title, low=0.0, high=1.0, from_zero=False, plot_span=plot_span, framed=framed)
        low, high = min(finite), max(finite)
        # Every value identical is an ordinary input — one arm, one sample — and the
        # span it gives is zero, which no proportion can pad. The value's own
        # magnitude is the only scale in the data; the fraction covers a value of
        # zero, where there is not even that. A framed axis pads by none of it, but
        # still needs a span: a zero-width domain is not a scale on any axis.
        pad = (high - low) * pad_fraction or abs(high) * pad_fraction or pad_fraction
        if low == high and not pad:
            pad = abs(high) * geometry()["domain_pad"] or geometry()["domain_pad"]
        return cls(title=title, low=low - pad, high=high + pad, from_zero=False, plot_span=plot_span, framed=framed)

    def encoding(self, field: str, *, upright_title: bool = False) -> dict[str, Any]:
        """The positional channel for one field drawn against this axis.

        Every layer of a frame is given the SAME title deliberately: the policy
        gate refuses an axis that carries two, which is how a layer accidentally
        drawn in another unit is caught rather than shipped.

        Args:
            field: The row key holding the value.
            upright_title: Whether to draw the title FLAT above the axis rather
                than letting Vega turn it. Set on a vertical axis that names its
                own quantity: Vega's default for a ``y`` title is a quarter turn,
                and no text in this report is text the reader tilts their head
                for — so the choice is between an unnamed axis and a title placed
                deliberately, and an unnamed axis leaves the reader to take the
                quantity from the heading two lines up.

        Returns:
            The Vega-Lite channel definition.
        """
        axis: dict[str, Any] = {
            # `None` rather than absent where the heading names the quantity: absent
            # is not silence, it is Vega-Lite filling the title with the field name,
            # which is machinery rather than a quantity.
            "title": self.title or None,
            # Per encoding, as a rule. A bar shares a baseline with every other bar
            # and carries its value on the mark, so a gridline behind it is a line
            # with nothing left to say; an interval has no baseline to read against,
            # so it keeps gridlines — at the labelled ticks, and never dashed, since
            # a dash in a chart reads as data. A range-framed axis is the third
            # case: its own rule is what a mark is read against, so the ticks that
            # rise from it replace the grid rather than joining it.
            "grid": not self.from_zero and not self.framed,
            "tickCount": self.tick_count,
            "labelFontSize": font_sizes()["tick"],
            "labelFontWeight": font_weights()["tick"],
        }
        if upright_title:
            # `titleAngle: 0` is also what tells the policy gate this was decided
            # rather than defaulted — it refuses a y title with no angle stated,
            # because that is the one Vega silently turns.
            axis |= {
                "titleAngle": 0,
                "titleAnchor": "start",
                "titleAlign": "left",
                # Clear of the topmost tick label, on the title's own line.
                "titleY": -12,
                "titleX": 0,
                "titleFontSize": font_sizes()["label"],
                "titleFontWeight": font_weights()["label"],
            }
        if self.framed:
            # Stated rather than defaulted, so the rule and its ticks are the same
            # furniture in both renderers.
            axis |= {"domain": True, "tickSize": _FRAMED_TICK_SIZE}
        if self.number_format:
            axis["format"] = self.number_format
        return {
            "field": field,
            "type": "quantitative",
            "scale": {"domain": [self.low, self.high], "zero": self.from_zero, "nice": False},
            "axis": axis,
        }

    # There is no crop footnote: a figure whose axis carries labelled numeric ticks has
    # already disclosed where it starts, and a sentence repeating it is noise that
    # trains readers to skip the subtitle — which is where the disclosures that DO
    # carry information live. The sentence was once scoped to spans on the
    # reasoning that a span's width invites a proportional comparison; read on
    # screen, a labelled 40,000-55,000 axis makes that comparison from the ticks too.
    #
    # What protects a reader from a silently rescaled comparison is NOT this
    # sentence — it is `policy._check_baseline`, which refuses a cropped axis under a
    # LENGTH mark outright rather than asking it to apologise. That rule is untouched
    # and is the one that matters: a bar misstates a ratio, while a point or an
    # interval states a position the ticks label.

    def marks_form_the_edge(self) -> bool:
        """Whether the marks' own baseline draws the line the identity axis would.

        A bar from zero starts exactly where the plot does when the domain does,
        so every bar's base redraws the identity axis's domain line — two lines in
        one place, one of them furniture and one of them data.
        """
        return self.from_zero and self.low == 0

    def points_right(self, end: float) -> bool:
        """Which way the mark ending at ``end`` grows away from its baseline."""
        return not self.from_zero or end >= 0

    def offset(self, value: float) -> float:
        """Where ``value`` lands, in px from the plot's low edge.

        The one place a value becomes a position. Every layout decision this module
        takes from a pixel distance goes through here, so a domain change moves them
        together rather than moving whichever ones remembered to convert.
        """
        span = self.high - self.low
        return (value - self.low) / span * self.plot_span if span else 0.0

    def room(self, end: float) -> float:
        """How much plot is left beyond a mark ending at ``end``, in px.

        Measured in the direction the mark grows, so a bar running left from zero
        is asked about the space on its left.

        Args:
            end: Where the mark ends, in the axis's units.

        Returns:
            The clear distance to the plot edge the mark is heading for.
        """
        offset = self.offset(end)
        return self.plot_span - offset if self.points_right(end) else offset


def _layers(*candidates: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The layers that exist, in drawn order — back to front, absences dropped.

    Several layers here are conditional (a zero rule only where zero is interior,
    a value label only where the marks are few enough), and the alternative to one
    filter is each arm building a list and appending to it in the right order,
    which is where a layer ends up drawn over the mark it annotates.
    """
    return [candidate for candidate in candidates if candidate is not None]


def _zero_rule(axis: ValueAxis) -> dict[str, Any] | None:
    """A solid rule at zero, where zero is a place inside the plot.

    Only where zero is *meaningful and interior*. On an all-positive chart zero is
    the plot's own edge and the bars draw it; on a cropped position axis zero is
    not on the chart at all, and a rule at the edge would be a baseline the values
    were never measured from.

    Args:
        axis: The chart's value axis.

    Returns:
        A layer to compose under the marks, or ``None``.
    """
    if not axis.from_zero or not axis.low < 0 < axis.high:
        return None
    return {
        "data": {"values": [{ANCHOR_FIELD: 0}]},
        # A style NAME, not a colour: the spec carries none, and a zero line drawn
        # in the series hue reads as one more mark rather than as the axis it is.
        "mark": {"type": "rule", "style": ZERO_RULE_STYLE},
        "encoding": {"x": axis.encoding(ANCHOR_FIELD)},
    }


def value_label_layers(values: Sequence[MarkValue], axis: ValueAxis, identity: dict[str, Any]) -> list[dict[str, Any]]:
    """Write each mark's value beside the mark, or nothing when there are too many.

    A value on its mark is the reason the grid can be as quiet as it is: the
    number the reader would have gone to the axis for is already on the bar.

    Where each one goes is :func:`_aligned_values`' decision; this is the shape it
    takes for a chart whose categories sit on an identity axis. The mirrored case
    cannot arise: a mark short enough for its label to overhang the *baseline* end
    has almost the whole plot free on the other side, so it is already labelled
    outside.

    Args:
        values: One entry per mark that carries a value.
        axis: The axis the marks are drawn against.
        identity: The category encoding, so each label lands on its own row.

    Returns:
        One layer per distinct placement, or none. The count is whatever
        :func:`_aligned_values` distinguishes — side, ink, clearance and lift — rather
        than a fixed four: an arm whose labels sit on both sides of zero and clear
        different marks draws each combination in a layer of its own.
    """
    return [
        {
            "data": {
                "values": [
                    {DISPLAY_FIELD: mark.display, ANCHOR_FIELD: mark.end, VALUE_TEXT_FIELD: mark.text} for mark in marks
                ]
            },
            "mark": value_label_mark(placement),
            "encoding": {
                "y": identity,
                "x": axis.encoding(ANCHOR_FIELD),
                "text": {"field": VALUE_TEXT_FIELD, "type": "nominal"},
            },
        }
        for placement, marks in _aligned_values(values, axis).items()
    ]


def value_label_mark(placement: Placement, **overrides: Any) -> dict[str, Any]:
    """The text mark one placement's value labels are drawn with.

    Shared rather than restated per arm for the reason the placement type exists: a
    label on a mark's fill has to ask for the knockout ink, and an arm that built its
    own text mark would be one arm away from drawing 2.53:1 grey-on-velvet again.

    Args:
        placement: Where these labels sit relative to their marks.
        **overrides: Mark properties the arm adds — a ``yOffset``, typically.

    Returns:
        A Vega-Lite text mark definition.
    """
    mark: dict[str, Any] = {
        "type": "text",
        "align": placement.align,
        "baseline": "middle",
        "dx": placement.dx,
    }
    if placement.inside:
        # A style NAME, not a colour. The value is drawn ON the mark's own fill, where
        # chart ink measures 2.53:1 in dark and 3.36:1 in light against the single-series
        # hue — so it takes the knockout instead, which the renderer resolves for the
        # surface it is painting. Same division as the zero rule: the name travels in the
        # spec, the colour stays with whichever renderer is drawing.
        mark["style"] = VALUE_ON_FILL_STYLE
    if placement.lift:
        # Only where there IS a lift. A `dy` of zero draws exactly what no `dy` draws,
        # and the compiled spec is a snapshot the browser side commits — a key that
        # changes nothing is still a diff nobody can tell from a moved label.
        mark["dy"] = placement.dy
    return mark | overrides


class Placement(NamedTuple):
    """Where one value label sits relative to the mark it names.

    **Two facts, not one, and conflating them is the trap this type exists to close.**
    ``align`` is which side of the anchor the text extends toward; ``inside`` is
    whether it is drawn over the mark's fill. They are not recoverable from each
    other — a bar growing LEFT from zero and labelled outside is right-aligned, and so
    is a right-growing bar labelled inside. Reading "inside" off ``align == "right"``
    would give the knockout ink to the first of those and withhold it from the second.

    ``clearance`` is a third and equally independent one — how FAR from the anchor the
    text starts, which is a fact about the mark's size rather than about which side it
    is on or what is under it. It rides here because it is what ``dx`` is made of, and
    because it is the last thing that has to differ before two labels can share a layer.
    """

    align: str
    """Which side of the anchor the text extends toward.

    ``left`` or ``right`` for a label set BESIDE a mark, and ``center`` for one lifted
    ABOVE it — which the distribution arm uses, since its anchor is the mean rather than
    a mark's end. A centred label takes no ``dx``: the offset holds text clear of a mark
    it sits next to, and applying it to one sitting above would shift the number off the
    value it names.
    """

    inside: bool
    """Whether the label is drawn over the mark's own fill."""

    clearance: float = VALUE_LABEL_OFFSET
    """How far the text is set from the value it is anchored at, in px, unsigned.

    The mark's own body plus :data:`VALUE_LABEL_OFFSET` — so the daylight a reader
    sees is the gap that constant names, whatever the mark drawn at the value is.
    Carried on the placement rather than applied per layer because it is what ``dx``
    is, and because two marks with different bodies want different offsets: the
    placement is the layer key, so they land in layers of their own rather than
    sharing one and averaging their geometry.

    Not the same quantity as :meth:`ValueAxis.symmetric`'s ``mark_clearance``, which
    is the body alone — that one answers "how much axis must be left past the outermost
    mark", where this answers "how far past the value does the text start".

    Defaults to the bare gap, which is the whole offset for a mark that ENDS at its
    value and the right answer for a caller that positions its own labels: the
    distribution arm builds this type directly for text it lifts ABOVE the mark.
    """

    lift: float = 0
    """How far the text is raised off the row's centreline, in px, unsigned.

    Zero for every label placed OUTSIDE its mark — there is nothing under it to clear —
    and zero for one drawn on a fill, which is read out of the fill rather than beside
    it. Non-zero only where the label was pushed inward over a mark too thin to knock a
    number out of and too solid to draw one through: see :attr:`MarkValue.thickness`.

    A fourth independent fact, and it rides here for the reason ``clearance`` does: it
    is what ``dy`` is made of, and two labels lifted by different amounts cannot share
    a layer any more than two with different offsets can.
    """

    @property
    def dx(self) -> float:
        """The label's horizontal offset from its anchor, in px, signed for its alignment.

        Returns:
            The clearance, negated where the text extends left of the anchor, and zero
            for a centred label — which sits ABOVE its mark rather than beside it, so
            offsetting it would shift the number off the value it names. That is the
            defect the centred placement was introduced to fix.
        """
        if self.align == "center":
            return 0
        return self.clearance if self.align == "left" else -self.clearance

    @property
    def dy(self) -> float:
        """The label's vertical offset from its row's centreline, in px, signed for the screen.

        Returns:
            The lift, negated: Vega's y grows downward, so raising a number off the
            line that would otherwise be struck through it is a negative offset.
        """
        return -self.lift


def _aligned_values(values: Sequence[MarkValue], axis: ValueAxis) -> dict[Placement, list[MarkValue]]:
    """Split each mark's value by the placement the room beyond its end decides.

    The decision, not the layer: a chart whose categories sit on an axis and one
    whose categories are facet rows draw their labels in different shapes but reach
    the same answer about where each one goes, and a second copy of this arithmetic
    is a second answer waiting to happen.

    **Outside the mark's end when there is room, otherwise on the fill.** The
    value-label rule states the clearance and this holds to it — with two additions it does not
    state and could not. A label needs its OWN width as well: sixty pixels of text
    will not fit fifty-six pixels of gap, and a text mark drawn past the plot edge
    makes Vega grow the frame, which is the figure leaving its column for a label. And
    it needs the room its own MARK occupies — the gap the rule states is daylight
    between the text and the mark's edge, so on a point, which is centred on the value
    it names, the radius is room the label never had.

    Args:
        values: One entry per mark that carries a value.
        axis: The axis the marks are drawn against.

    Returns:
        Placement → the marks taking it, or an empty mapping where no value is
        written at all — none given, or more marks than a figure labels.
    """
    sizes = geometry()
    if not values or len(values) > sizes["value_label_max_marks"]:
        return {}
    size = font_sizes()["value"]
    aligned: dict[Placement, list[MarkValue]] = {}
    for mark in values:
        # The mark's own body is part of both halves: it is room the text cannot be
        # written in, and it is distance the text has to start beyond. `MarkValue.radius`
        # is zero for a mark that ends at its value, which is every bar.
        clearance = mark.radius + VALUE_LABEL_OFFSET
        needed = max(sizes["value_label_clearance"], text_width(mark.text, size) + clearance)
        outside = axis.room(mark.end) >= needed
        align = "left" if axis.points_right(mark.end) == outside else "right"
        # `inside` needs BOTH: the label was pushed inward AND the mark paints a fill
        # there. Clearance alone answers only the first — see `MarkValue.filled` for
        # what taking it as both would draw.
        #
        # A mark that paints no fill is not therefore empty under the label: the two
        # dumbbells draw a 3px line along the row, and a number written on that line
        # is a number with a rule through it. Lifted off it instead — the knockout is
        # not an option at that thickness, since surface-coloured glyphs would be
        # legible over the 3px stripe and invisible above and below it.
        lift = 0.0 if outside or mark.filled else _label_lift(mark.thickness, size)
        placement = Placement(align=align, inside=not outside and mark.filled, clearance=clearance, lift=lift)
        aligned.setdefault(placement, []).append(mark)
    return aligned


def _label_lift(thickness: float, font_size: float) -> float:
    """How far a value label is raised off the thin mark it would be drawn through, in px.

    Half the mark's thickness clears the mark; half the label's LINE BOX clears the
    label. The line box rather than the glyph band, deliberately and not for want of a
    measurement: digits and a percent sign reach about seven tenths of the font size,
    so the remainder of the half-box is the daylight between the two, and a lift built
    from the band instead would leave them touching. What settles it either way is
    ``tests/test_vega_render.py``, which reads the drawn pixels rather than
    this arithmetic.

    Small enough to stay inside the row: at the row step this package draws on, a lift
    of half a line box plus half a rule leaves the label nearer its own mark than any
    neighbour's, which is what keeps a lifted number readable as its row's.

    Args:
        thickness: How far the mark reaches across the row, in px.
        font_size: The size the label is drawn at, in px.

    Returns:
        The lift, or zero where nothing is drawn under the label to clear.
    """
    return thickness / 2 + font_size / 2 if thickness else 0.0


def centred_value_placements(values: Sequence[MarkValue], axis: ValueAxis) -> dict[str, list[MarkValue]]:
    """Group value labels anchored INSIDE a mark by the alignment that keeps each inside the plot.

    For a label lifted onto its own line above the mark and anchored at the value it
    names — an interval's mean, which is the middle of the mark rather than its end.
    Both interval arms (``distribution`` and ``null_result``) draw their estimate this
    way, and they share this so the two cannot disagree about where a number goes.

    **A different question from :func:`_aligned_values`, which is why this does not
    call it.** That one asks which side of a mark's END has room, because a bar's label
    goes beside the bar. This label is anchored at an interior point and is lifted clear
    of the mark rather than set beside it, so the only question left is whether the text
    box fits the plot on both sides. Centred where it does; otherwise pushed to whichever
    side the text has to grow into. Both are still bounded by ``value_label_max_marks``,
    which is a statement about how many numbers a figure can carry rather than about
    where they sit.

    Without the push a label near the domain's edge overran the plot: a mean of 14.9 on
    an axis ending at 15.0 printed past the right edge, overlapping its own interval cap
    and making Vega grow the frame — the figure leaving its column for a label.

    Args:
        values: One entry per mark carrying a value, each anchored at the value it names.
        axis: The value axis the marks are drawn against.

    Returns:
        ``center``/``left``/``right`` → the marks taking it, or an empty mapping where
        no value is written at all.
    """
    sizes = geometry()
    if not values or len(values) > sizes["value_label_max_marks"]:
        return {}
    size = font_sizes()["value"]
    placed: dict[str, list[MarkValue]] = {}
    for mark in values:
        half = text_width(mark.text, size) / 2
        from_left = axis.offset(mark.end)
        from_right = axis.plot_span - from_left
        if from_left >= half and from_right >= half:
            placement = "center"
        elif from_right < half:
            # Not enough plot to the right, so the text grows LEFT from the anchor —
            # which is what Vega calls a right alignment.
            placement = "right"
        else:
            placement = "left"
        placed.setdefault(placement, []).append(mark)
    return placed


__all__ = [
    "ANCHOR_FIELD",
    "DISPLAY_FIELD",
    "KIND_FIELD",
    "RISE_FIELD",
    "SECONDARY_OPACITY",
    "VALUE_LABEL_OFFSET",
    "VALUE_TEXT_FIELD",
    "VEGA_LITE_SCHEMA",
    "CompiledColumn",
    "CompiledChart",
    "MarkValue",
    "Placement",
    "ValueAxis",
    "centred_value_placements",
    "compile_chart",
    "draw_intent",
    "plot_size",
    "point_radius",
    "value_label_layers",
    "value_label_mark",
]

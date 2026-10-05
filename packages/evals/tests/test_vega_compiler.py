"""The payload→spec compiler, and the numbers it reports alongside the picture.

The values-as-drawn assertions matter more than they look: they are the whole of
what a surface that cannot draw gets, and they are derived from the spec's own
plotted rows so the two descriptions of one chart cannot drift. A test that
built its own expected numbers would be checking arithmetic; these check that the
text and the picture come from one compilation.
"""

import collections
import copy
import json
import re
from dataclasses import fields
from pathlib import Path

import pytest

from threetears.evals.vega import compile_chart
from threetears.evals.vega.compiler import (
    ANCHOR_FIELD,
    DISPLAY_FIELD,
    KIND_FIELD,
    RISE_FIELD,
    SECONDARY_OPACITY,
    VALUE_LABEL_OFFSET,
    VALUE_TEXT_FIELD,
    VEGA_LITE_SCHEMA,
    CompiledChart,
    MarkValue,
    Placement,
    plot_size,
    value_label_layers,
    ValueAxis,
    point_radius,
)
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.viz.intent import chart_intent
from threetears.evals.analysis.viz.quantities import display_scale, strip_common_prefix
from threetears.evals.vega.palette import (
    CONTEXT_STYLE,
    SEQUENTIAL_RANGE,
    VALUE_ON_FILL_STYLE,
    ZERO_RULE_STYLE,
    font_sizes,
    geometry,
    sequential_colors,
    vega_config,
)
from threetears.evals.analysis.viz.payloads import PayloadError
from threetears.evals.vega.spec_policy import RANKING_SPEC_NAME, check_spec
from threetears.evals.vega.render import render_svg
from threetears.evals.vega.text_metrics import fits, text_width
from packages.evals.tests.chart_examples import (
    DELTA_TABLE,
    DISTRIBUTION,
    EVERY_TYPE,
    FRONTIER,
    NULL_RESULT,
    PAYLOAD,
    SWEEP_RANKING,
)


def _mark_layers(spec, mark_type=None):
    """Every mark-bearing node in a compiled spec, at any composition depth.

    Charts here are composed of layers that each carry one mark — the bars, the
    zero rule, the values written on the marks — so an assertion about *the* mark
    has to name which one rather than reaching a single top-level `encoding`.
    """
    found = []
    if isinstance(spec, dict):
        mark = spec.get("mark")
        drawn = mark.get("type") if isinstance(mark, dict) else mark
        if drawn is not None and mark_type in (None, drawn):
            found.append(spec)
        for value in spec.values():
            found.extend(_mark_layers(value, mark_type))
    elif isinstance(spec, list):
        for item in spec:
            found.extend(_mark_layers(item, mark_type))
    return found


def _mark_layer(spec, mark_type):
    """The first layer drawing ``mark_type``, in document order."""
    layers = _mark_layers(spec, mark_type)
    assert layers, f"no {mark_type} layer in {json.dumps(spec)[:400]}"
    return layers[0]


def _drawn_point_radius(spec):
    """How far the point a compiled chart draws extends past its value, in px.

    Read off the spec's own point mark, so a clearance asserted against it is the clearance
    of the disc the renderer actually draws -- one size per chart, or the question has no answer.
    """
    sizes = {layer["mark"]["size"] for layer in _mark_layers(spec, "point") if "size" in layer["mark"]}
    assert len(sizes) == 1, f"expected one drawn point size, found {sorted(sizes)}"
    return point_radius(sizes.pop())


#: The relative change ``delta_table``'s axis never narrows below, as the report states it.
#:
#: Restated rather than read: ten percent is the reference a reader is told "small" means, so it is
#: a number the figure STATES, and a change to it is a change a reader sees.
STATED_CHANGE_FLOOR = 0.1

#: The gap ``frontier`` leaves between a contestant's disc and its name, in px -- a drawn distance,
#: restated for the same reason.
FRONTIER_NAME_GAP = 8


def _title_text(spec):
    """A spec's title as one string, whichever of the three shapes it took.

    Joined rather than returned as drawn, because a title long enough to wrap is
    carried as one entry per line and what it SAYS is the same either way — the
    line shape is `lines_of`'s question, and the tests that ask it ask it directly.
    """
    title = spec["title"]
    if isinstance(title, dict):
        title = title["text"]
    return " ".join(title) if isinstance(title, list) else title


def _subtitle(spec):
    """A spec's footnote as one string, or `""` when it draws none."""
    title = spec["title"]
    subtitle = title.get("subtitle", "") if isinstance(title, dict) else ""
    return " ".join(subtitle) if isinstance(subtitle, list) else subtitle


def _disclosed(chart):
    """A chart's disclosure lines as one text, for the checks about what a line SAYS.

    Newline-joined so a phrase can never match across two lines by accident; the checks
    about which LINE carries what read ``chart.disclosures`` directly.
    """
    return "\n".join(chart.disclosures)


def _subtitle_lines(spec):
    """A spec's footnote as the lines it draws, for the checks about its shape."""
    title = spec["title"]
    return lines_of(title.get("subtitle", "") if isinstance(title, dict) else "")


def _frame_rows(spec):
    """The rows a composed frame hands down to every layer that brought none.

    A faceted figure partitions ONE table, so its marks carry no data of their own
    — a layer that did would draw its whole dataset in every cell rather than the
    cell's own slice.
    """
    return (spec.get("data") or {}).get("values", [])


def _layer_rows(spec, layer):
    """The rows one layer actually draws — its own, or the frame's, past its filter."""
    rows = (layer.get("data") or {}).get("values")
    if rows is None:
        rows = _frame_rows(spec)
    for step in layer.get("transform", []):
        clause = step.get("filter")
        if isinstance(clause, dict) and "field" in clause:
            rows = [row for row in rows if row.get(clause["field"]) == clause["equal"]]
    return rows


class TestBreakdownSpec:
    def test_the_spec_declares_the_pinned_schema(self):
        assert compile_chart("breakdown", PAYLOAD).spec["$schema"] == VEGA_LITE_SCHEMA

    def test_parts_are_sorted_descending_by_value(self):
        """The reader's question is which part dominates; a sort makes it a glance."""
        chart = compile_chart("breakdown", PAYLOAD)
        assert [row["label"] for row in chart.rows] == [
            "budget_exhausted",
            "no_new_sources",
            "confidence_met",
            "tool_error",
        ]

    def test_ties_break_on_label_so_the_order_is_deterministic(self):
        """Two equal parts must not swap between compilations of the same payload."""
        payload = {"parts": [{"label": "z", "value": 1.0}, {"label": "a", "value": 1.0}], "unit": "runs"}
        assert [row["label"] for row in compile_chart("breakdown", payload).rows] == ["a", "z"]

    def test_the_drawn_order_is_the_spec_order_not_a_vega_sort(self):
        """A sort resolved inside Vega would exist only in the picture, not the text."""
        chart = compile_chart("breakdown", PAYLOAD)
        drawn = _mark_layer(chart.spec, "bar")["data"]["values"]
        assert [row["label"] for row in drawn] == [row["label"] for row in chart.rows]

    def test_the_quantitative_axis_carries_the_unit(self):
        chart = compile_chart("breakdown", PAYLOAD)
        assert _mark_layer(chart.spec, "bar")["encoding"]["x"]["axis"]["title"] == "share of stops (%)"

    def test_the_bar_baseline_includes_zero(self):
        scale = _mark_layer(compile_chart("breakdown", PAYLOAD).spec, "bar")["encoding"]["x"]["scale"]
        assert scale["zero"] is True
        assert scale["domain"][0] == 0, "the drawn domain starts at the baseline the flag claims"

    def test_the_spec_carries_no_colour(self):
        """Colour arrives as a renderer's config; a literal here would defeat both surfaces.

        This is also the earliest layer at which the OKLCH-renders-black defect can
        be caught, and the clearest place to name it.
        """
        import json

        rendered = json.dumps(compile_chart("breakdown", PAYLOAD).spec).lower()
        assert "oklch(" not in rendered
        assert "#" not in rendered
        assert "color" not in rendered

    def test_no_categorical_colour_encoding_is_emitted(self):
        """Identity rides on the axis labels, which has no palette ceiling."""
        for layer in _mark_layers(compile_chart("breakdown", PAYLOAD).spec):
            assert "color" not in layer["encoding"]


class TestTitleAndTooltip:
    def test_the_title_is_the_generators_own_measure_name(self):
        """Verbatim, not assembled around: the report renders its prose everywhere else."""
        assert compile_chart("breakdown", PAYLOAD).title == "share of stops"

    def test_a_payload_with_no_measure_still_titles_the_chart(self):
        payload = {"parts": [{"label": "a", "value": 3}, {"label": "b", "value": 1}], "unit": "runs"}
        chart = compile_chart("breakdown", payload)
        assert chart.title == "Breakdown"
        assert _mark_layer(chart.spec, "bar")["encoding"]["x"]["axis"]["title"] == "value (runs)"

    def test_the_tooltip_offers_n_when_the_payload_counts(self):
        tooltip = _mark_layer(compile_chart("breakdown", PAYLOAD).spec, "bar")["encoding"]["tooltip"]
        assert [entry["field"] for entry in tooltip] == ["label", "value", "n"]

    def test_the_tooltip_omits_n_when_nothing_counts(self):
        """A column of "n: null" states an absence the payload states by omission."""
        payload = {"parts": [{"label": "a", "value": 3}, {"label": "b", "value": 1}], "unit": "runs"}
        tooltip = _mark_layer(compile_chart("breakdown", payload).spec, "bar")["encoding"]["tooltip"]
        assert [entry["field"] for entry in tooltip] == ["label", "value"]


class TestValuesAsDrawn:
    def test_values_match_the_plotted_rows_in_drawn_order(self):
        chart = compile_chart("breakdown", PAYLOAD)
        header, *lines = chart.values_as_drawn()
        assert header.startswith("Part")
        assert len(lines) == len(chart.rows)
        for line, row in zip(lines, chart.rows, strict=True):
            assert line.startswith(row["label"])
            assert str(row["n"]) in line

    def test_the_unit_is_stated_once_in_the_column_header(self):
        """Not repeated on every cell: this table IS the reading for a surface that
        cannot draw, so the unit must be on it — but a column states one quantity,
        so stating it per row is noise the reader has to filter."""
        headers = {column["key"]: column["header"] for column in compile_chart("breakdown", PAYLOAD).columns}
        assert headers["value"] == "Value (%)"

    def test_a_percentage_is_not_spaced_from_its_unit(self):
        """Checked where a unit still joins a number inline — a disclosure line."""
        assert "of 100%" in _disclosed(compile_chart("breakdown", PAYLOAD))

    def test_a_word_unit_is_spaced_from_its_number(self):
        payload = {"parts": [{"label": "a", "value": 3}, {"label": "b", "value": 1}], "unit": "runs", "total": 4}
        assert compile_chart("breakdown", payload).disclosures == ["The parts divide a total of 4 runs."]

    def test_a_breakdown_that_counts_nothing_has_no_n_column(self):
        """An absent n is stated by absence, never as a column of em dashes."""
        payload = {"parts": [{"label": "a", "value": 3}, {"label": "b", "value": 1}], "unit": "runs"}
        chart = compile_chart("breakdown", payload)
        assert [column["key"] for column in chart.columns] == ["label", "value"]

    def test_a_disclosure_states_the_whole_the_parts_divide(self):
        assert compile_chart("breakdown", PAYLOAD).disclosures == ["The parts divide a total of 100% over n=49."]

    def test_no_disclosure_when_the_payload_knows_no_whole(self):
        """Without a total the bars are magnitudes, not shares — and say nothing more."""
        payload = {"parts": [{"label": "a", "value": 3}, {"label": "b", "value": 1}], "unit": "runs"}
        assert compile_chart("breakdown", payload).disclosures == []


class TestCompilerRefusals:
    def test_a_malformed_payload_raises_rather_than_compiling(self):
        with pytest.raises(PayloadError):
            compile_chart("breakdown", {"stop_causes": []})

    def test_a_type_with_no_compiler_raises(self):
        """Distinguishable from a malformed payload: nothing is wrong with the data.

        `scatter` is genuinely uncompiled — it has no payload model and no
        dispatch arm. Its predecessors here (``timeseries`` last) named types
        that have since been migrated, which would have made this pass for the
        wrong reason.
        """
        with pytest.raises(PayloadError, match="no chart intent for viz type"):
            compile_chart("scatter", {"series": []})


class TestCompilerOutputPassesThePolicyGate:
    def test_every_compiled_spec_is_gated(self):
        """The compiler is a producer like any other — it does not get a pass.

        Asserted by compiling, since `compile_chart` raises if its own output
        fails the gate; the value is that this test fails the day a compiler
        change starts emitting a spec the report standard rejects.
        """
        from threetears.evals.vega.spec_policy import check_spec

        assert check_spec(compile_chart("breakdown", PAYLOAD).spec) == []


#: The types whose figure is a point plot rather than a stack of categorical rows.
#:
#: The distinction is geometric and it decides which rules a figure answers to: a
#: row-based figure takes the report's one column width, bounds its label column
#: with the gutter, and writes each mark's single value beside it. A point plot has
#: no rows to align, no gutter to bound, and two positions per mark instead of one
#: length, so it takes the fixed point geometry and labels each mark in place.
#:
#: **Declared here, by the suite, because only the suite reads it.** An arm already
#: knows its own geometry -- it reaches for the point tokens directly -- so the
#: compiler has no use for the grouping. What it exists for is the rules stated
#: ACROSS types, where "every figure is the column width" is true of one group and
#: false of the other. A type the registry gains is row-based unless named here, so a
#: new point plot fails the row-based width rule below, which is the intended alarm.
POINT_PLOT_TYPES: frozenset[str] = frozenset({"frontier"})

#: The row-based subset of :data:`EVERY_TYPE`, in id order.
ROW_BASED: list[str] = sorted(set(EVERY_TYPE) - POINT_PLOT_TYPES)


def _sizes(node, key, found=None):
    """Every value a VIEW carries under `key`, at any composition depth.

    A mark's own `height` is a different quantity from a view's — the dumbbell's
    connector states its own thickness, which is a legitimate mark height and not a
    view that forgot its size — so the walk stops at `mark`.
    """
    found = [] if found is None else found
    if isinstance(node, dict):
        for name, value in node.items():
            if name == "mark":
                continue
            if name == key:
                found.append(value)
            else:
                _sizes(value, key, found)
    elif isinstance(node, list):
        for item in node:
            _sizes(item, key, found)
    return found


def test_every_drawable_type_has_a_payload_here():
    """`EVERY_TYPE` covers the registry, or the parametrisations below skip in silence.

    Every geometry rule is asserted by walking `EVERY_TYPE`, so a viz type that
    gains a payload model and a compiler arm without gaining an entry here is
    simply not checked — no failure, no skip message, just narrower coverage than
    the test names claim. That is the shape that already bit this suite once: a
    gutter bound was migrated on one chart and asserted on that same chart, and
    four types kept the old value with the suite green.

    `test_vega_render.py` pins its own `PAYLOADS` to `PAYLOAD_MODELS` for the same
    reason; this is the missing half of that pair.
    """
    from threetears.evals.analysis.viz.payloads import PAYLOAD_MODELS

    assert set(EVERY_TYPE) == set(PAYLOAD_MODELS)


def test_every_payload_model_has_a_compiler_arm():
    """The two halves of one registration agree about which types are drawable.

    A type is registered twice — a payload model that validates it, and an arm
    that draws it — in two files that nothing else connects. Both directions of a
    mismatch reach a reader as the same message. A model with no arm parses the
    payload and then raises "no compiler", which reads as *this build cannot draw
    that type* when the truth is that someone added a model and stopped; an arm
    with no model never runs at all, because dispatch happens after validation and
    an unmodelled type never gets that far.

    Asserting set equality rather than one inclusion is deliberate: the missing-arm
    direction is the one a half-finished type produces, and the missing-model
    direction is the one a deleted type leaves behind.
    """
    from threetears.evals.vega.arms import ARMS
    from threetears.evals.analysis.viz.payloads import PAYLOAD_MODELS

    assert set(ARMS) == set(PAYLOAD_MODELS)


class TestFigureGeometry:
    """The figure is a fixed size, and every number in it comes from the artifact.

    Width fixed and height driven by the row count is the only ordering that draws
    a readable chart — the other way round gives three bars 12px tall in a plot as
    wide as the window, because width follows the card while height follows the
    data. These pin the arithmetic and, more importantly, pin where the numbers
    come from: a literal reintroduced in the compiler passes every shape assertion
    and silently unlinks the two renderers from the tokens they are supposed to share.
    """

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_every_dimension_is_a_number_at_every_depth(self, viz_type):
        """Nothing is left for a renderer to substitute — including nested views.

        `"width": "container"` is honoured only on a top-level unit or layer spec;
        under `facet` or `concat` Vega-Lite does not support it, so the composed
        shapes drew past their frame while the simple ones looked fine. Asserting at
        every depth is what covers the shapes where the size is not at the top.
        """
        spec = compile_chart(viz_type, EVERY_TYPE[viz_type]).spec
        found = _sizes(spec, "width") + _sizes(spec, "height")
        assert found, f"{viz_type} compiled with no dimensions at all"
        for value in found:
            assert isinstance(value, int | float), f"{viz_type} carries a non-numeric dimension: {value!r}"

    @pytest.mark.parametrize("viz_type", ROW_BASED, ids=ROW_BASED)
    def test_every_row_based_plot_is_the_one_column_width(self, viz_type):
        """One column, so figures stack with their plot areas aligned.

        A report whose plots each pick their own width reads as a set of unrelated
        pictures; shared left and right edges are what makes two charts on one page
        comparable at a glance.

        The subject is the row-based figures because that is the alignment this
        rule buys: they stack, so their plot edges line up down the report. A point
        plot has no rows to align with anything and is small on purpose — the
        assertion below is the one it answers to.

        **Two widths, not one, since a figure may spend its gutter on a glyph.**
        The rule is about where the VALUE plot's edges fall, and a figure whose
        identity column is drawn rather than written declares that column's width
        instead of an axis label limit. Both edges still land in the same place —
        the value plot is the column width and the only other width a figure may
        declare is exactly the gutter — which is what the pair below says, and it
        pins the left edge that the single-width version only implied.
        """
        spec = compile_chart(viz_type, EVERY_TYPE[viz_type]).spec
        sizes = geometry()
        widths = set(_sizes(spec, "width"))
        assert sizes["plot_width"] in widths, f"{viz_type} draws no plot at the column width"
        assert widths <= {sizes["plot_width"], sizes["gutter_left"]}, (
            f"{viz_type} declares a width that is neither the plot nor the gutter: {sorted(widths)}"
        )

    @pytest.mark.parametrize("viz_type", sorted(POINT_PLOT_TYPES), ids=sorted(POINT_PLOT_TYPES))
    def test_every_point_plot_takes_the_point_geometry(self, viz_type):
        """A scatter has no row count to drive height, so it takes a fixed aspect.

        Bounded exactly as strictly as the column width bounds a row-based figure —
        the point is that the figure's size is decided by the tokens rather than by
        the data, not that every figure is the same size.
        """
        spec = compile_chart(viz_type, EVERY_TYPE[viz_type]).spec
        sizes = geometry()
        assert set(_sizes(spec, "width")) == {sizes["point_width"]}
        assert set(_sizes(spec, "height")) == {sizes["point_height"]}
        assert sizes["point_width"] <= sizes["plot_width"], "a point plot that outgrew the column is not a small chart"

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_no_bar_is_drawn_with_a_rounded_end(self, viz_type):
        """A radius places the bar's tip short of its value.

        It also rounds the zero end into a shape the data does not have. On a short
        bar the radius is a meaningful share of the length, so the corner is a
        systematic under-draw of exactly the values that can least afford one.
        """
        spec = compile_chart(viz_type, EVERY_TYPE[viz_type]).spec
        rounded = [key for key in _corner_keys(spec)]
        assert rounded == [], f"{viz_type} still rounds a bar: {rounded}"

    def test_the_row_step_drives_height_against_a_floor(self):
        """Three bars are not a letterbox, and ten still read.

        The floor is what stops a two- or three-row chart stretching across the
        column at the height of its rows, which is where the row-driven rule needs
        a bound rather than an exception.
        """
        sizes = geometry()
        assert plot_size(3) == (sizes["plot_width"], sizes["plot_min_height"])
        assert plot_size(10) == (sizes["plot_width"], 10 * sizes["row_step"])
        assert plot_size(3, marginal=True)[1] == 3 * sizes["row_step_marginal"]

    def test_no_row_count_can_produce_a_letterboxed_plot(self):
        """The aspect ceiling, stated as the invariant rather than as a case.

        With the current floor it never binds — 168 × 6 already exceeds the plot
        width — so there is no row count that exercises the `min` directly, and a
        test that tried to would be asserting on the floor instead. What holds
        either way, and would fail the day the floor dropped without the ceiling
        being applied, is the proportion itself.
        """
        ceiling = geometry()["aspect_max"]
        for rows in range(1, 40):
            width, height = plot_size(rows)
            assert width <= height * ceiling, f"{rows} rows drew {width}×{height}, past {ceiling}:1"

    def test_the_figure_is_its_gutters_plus_its_plot(self):
        """The arithmetic the whole column rests on, asserted rather than assumed.

        Fixing the plot width only fixes the FIGURE width if the space either side
        of it is bounded too; a gutter that grew with its labels would leave the
        report column sized for a figure that no longer fits it.
        """
        sizes = geometry()
        assert sizes["gutter_left"] + sizes["plot_width"] + sizes["gutter_right"] == sizes["figure_width"]

    @pytest.mark.parametrize("viz_type", ROW_BASED, ids=ROW_BASED)
    def test_every_row_based_label_column_is_bounded_by_the_gutter(self, viz_type):
        """Every row-based type, because four of the five share one helper and one does not.

        An unbounded label column widens the figure by however long the longest
        category name happens to be, which is the data deciding the layout. Checking
        one chart's axis would have passed while the shared helper — the one the other
        four types reach for, and the facet header below it — still carried its own
        number; that is the exact half-fix this parametrisation exists to catch.

        A gutter is a row-based figure's furniture: the identity sits in a column to
        the left of the plot, and the bound is what keeps that column from growing.
        A point plot writes each name beside its own mark and has no such column, so
        the rule it answers to is the next one — the figure is bounded by its own
        declared size instead.

        **The column is bounded whether it holds text or a glyph**, and the two are
        bounded by different mechanisms because they are different things: a written
        name is bounded by the axis's own `labelLimit`, and a drawn one by the width
        of the panel it is drawn in. What must not vary is the answer — the identity
        column is the gutter, so the plot beside it starts in the same place on every
        figure of a report. A figure bounding it by NEITHER mechanism is the defect
        this catches, since that is the one whose width the data decides.
        """
        spec = compile_chart(viz_type, EVERY_TYPE[viz_type]).spec
        gutter = geometry()["gutter_left"]
        limits = _sizes(spec, "labelLimit")
        drawn = [width for width in _sizes(spec, "width") if width == gutter]
        assert limits or drawn, f"{viz_type} bounds no identity column at all"
        assert set(limits) <= {gutter}, (
            f"{viz_type} bounds a written label column at {sorted(set(limits))}, not the gutter"
        )

    @pytest.mark.parametrize("viz_type", sorted(POINT_PLOT_TYPES), ids=sorted(POINT_PLOT_TYPES))
    def test_every_point_plot_names_its_marks_beside_them(self, viz_type):
        """Identity is still direct, it is just not in a gutter.

        The property the gutter bound protects is that a name never decides the
        layout and never gets truncated. On a point plot the same property holds by
        a different mechanism: the name is a text mark at the point's own position,
        so a long one costs no width the figure has to find, and nothing clips it.
        This asserts the mechanism is actually there — a scatter that quietly
        dropped its labels would pass the width rule above and identify nothing.
        """
        chart = compile_chart(viz_type, EVERY_TYPE[viz_type])
        named = {
            row[DISPLAY_FIELD]
            for layer in _mark_layers(chart.spec, "text")
            for row in _layer_rows(chart.spec, layer)
            if DISPLAY_FIELD in row
        }
        assert named == {row[DISPLAY_FIELD] for row in chart.rows if row["cost"] is not None}

    def test_the_sizes_are_read_from_the_artifact_and_not_baked_in(self, monkeypatch):
        """The defect this class exists to prevent, reproduced deliberately.

        Every other assertion here passes just as happily against a compiler that
        had copied the artifact's numbers into literals — and a copy is what
        silently unlinks the server's charts from the tokens the browser draws
        from. Moving the artifact's values and watching the spec follow is the only
        check that can tell the two apart.
        """
        moved = geometry() | {"row_step": 100, "plot_min_height": 10, "bar_height": 7, "plot_width": 500}
        monkeypatch.setattr("threetears.evals.vega.compiler.geometry", lambda: moved)
        spec = compile_chart("breakdown", PAYLOAD).spec
        assert spec["height"] == 400, "the row step is not being read from the artifact"
        assert spec["width"] == 500, "the plot width is not being read from the artifact"
        assert _mark_layer(spec, "bar")["mark"]["height"] == 7, "the bar thickness is not being read from the artifact"


def _corner_keys(node):
    """Every `cornerRadius*` property a spec sets, at any depth."""
    found = []
    if isinstance(node, dict):
        for name, value in node.items():
            if name.startswith("cornerRadius"):
                found.append(name)
            found.extend(_corner_keys(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_corner_keys(item))
    return found


def _layer_marks(spec):
    """Every mark type a compiled spec draws, at any composition depth."""
    marks = []
    if isinstance(spec, dict):
        mark = spec.get("mark")
        if isinstance(mark, dict):
            marks.append(mark.get("type"))
        elif isinstance(mark, str):
            marks.append(mark)
        for value in spec.values():
            marks.extend(_layer_marks(value))
    elif isinstance(spec, list):
        for item in spec:
            marks.extend(_layer_marks(item))
    return marks


class TestUnitIsCarriedAsData:
    """The unit is a field, never inferred from the measure's name."""

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_no_prose_beside_a_restated_chart_quotes_the_unrestated_number(self, viz_type):
        """The durable half, built the third time it was asked for.

        A chart that restates its quantity (100174 ms drawn as 100 s) has to restate
        it EVERYWHERE a reader meets a number, and the caption, disclosures and footnote are the
        two places that are neither an axis nor a column header — so nothing pinned
        them. That is exactly how `sweep_ranking`'s omission band shipped naming its
        dropped configurations in milliseconds beside marks labelled in seconds; the
        fixture that caught it was written for that one type, and this is the rule.

        Stated as an ABSENCE rather than by parsing prose for units, because the
        numbers in a caption are generator words as often as compiler ones. What
        cannot be there is the pre-restatement value: if the chart divided by 1000,
        the undivided figure appearing beside it is two rulers in one sentence.
        """
        chart = compile_chart(viz_type, EVERY_TYPE[viz_type])
        raw = _restatable_values(viz_type, EVERY_TYPE[viz_type])
        if not raw or not chart.unit:
            return
        scale, _ = display_scale(raw, _declared_unit(viz_type, EVERY_TYPE[viz_type]))
        if scale == 1.0:
            # Nothing to catch here — this type's shared fixture states a unit the
            # ladder does not move, so the parametrisation runs and asserts nothing.
            # Which types DO exercise it is asserted below rather than assumed.
            return
        prose = f"{chart.caption} {_disclosed(chart)} {_subtitle(chart.spec)}"
        for value in raw:
            # The MAGNITUDE, because `format_number` carries a sign and prose does
            # not have to — "31800 ms faster" states the same unrestated number as
            # "-31800" and would slip a signed comparison.
            unrestated = format_number(abs(value))
            if len(unrestated) < 3:
                # Too short to be evidence: a two-digit run would match a row count
                # or a percentage that has nothing to do with the restated quantity.
                continue
            assert unrestated not in prose, (
                f"{viz_type} states {unrestated!r} beside a chart drawn in {chart.unit!r} — "
                "the restatement reached the axis and the columns but not the prose"
            )

    def test_the_prose_gate_above_actually_reaches_something(self):
        """Names its own coverage, because a parametrised gate hides its gaps.

        The check only bites where the shared fixture states a unit the restatement
        ladder moves, and most do not. Asserting the exact set is what stops it
        reading as "every type is covered" — `sweep_ranking` in particular is NOT
        covered here (its `EVERY_TYPE` payload is unitless) and is covered by
        `TestSweepRankingStatesItsOmissionInTheUnitTheAxisUses` instead, which is
        the fixture the original defect needed.

        A type joining this set is good news. A type LEAVING it is a fixture that
        stopped exercising a rule while the suite stayed green, which is the failure
        this whole class of assertion exists to make visible.
        """
        restating = set()
        for viz_type, payload in EVERY_TYPE.items():
            raw = _restatable_values(viz_type, payload)
            if not raw:
                continue
            scale, _ = display_scale(raw, _declared_unit(viz_type, payload))
            if scale != 1.0:
                restating.add(viz_type)
        assert restating == {"attribution", "distribution"}, (
            f"the prose-restatement gate now reaches {sorted(restating)}; if a type left this set its "
            "fixture stopped exercising the rule"
        )

    def test_a_duration_is_restated_in_the_unit_a_reader_thinks_in(self):
        """One unit per quantity: `~100s`, never `~100174ms`. Decided once here, so both surfaces agree."""
        chart = compile_chart("distribution", DISTRIBUTION)
        assert chart.unit == "s"
        assert _title_text(chart.spec) == "pipeline_synthesis_ms (s)"
        assert chart.rows[0]["mean"] == pytest.approx(1.39)

    def test_the_whole_chart_is_restated_together_or_not_at_all(self):
        """One unit per QUANTITY: a column mixing 900ms with 1.2s is two rulers."""
        payload = {
            **DISTRIBUTION,
            "groups": [
                {
                    "label": "a",
                    "ci": {"low": 100.0, "high": 900.0, "mean": 500.0, "level": 0.95, "variability": "across 5 runs"},
                }
            ],
        }
        chart = compile_chart("distribution", payload)
        assert chart.unit == "ms", "nothing reaches a second, so nothing is restated"

    def test_a_payload_stating_no_unit_states_none_rather_than_guessing_one(self):
        """The name-sniffing this replaces read `pipeline_search_ms (subsystem, s)` as
        unitless and rendered raw milliseconds. Saying nothing is the honest answer."""
        payload = {
            "groups": [
                {
                    "label": "a",
                    "ci": {"low": 1.0, "high": 3.0, "mean": 2.0, "level": 0.95, "variability": "across 5 runs"},
                }
            ],
            "x_label": "total_ms",
        }
        chart = compile_chart("distribution", payload)
        assert chart.unit == ""
        assert _title_text(chart.spec) == "total_ms"

    def test_a_delta_table_restates_each_row_in_its_own_unit(self):
        """The rows are different metrics, so one shared unit would be false."""
        rows = {row["metric"]: row for row in compile_chart("delta_table", DELTA_TABLE).rows}
        assert rows["total_ms"]["a"] == "16.16 s"
        assert rows["cost_usd"]["a"] == "0.011 usd"

    def test_the_frontier_restates_its_latency_rather_than_relabelling_it(self):
        """The successor to a deleted browser-side frontier assertion.

        That test asserted `41 s` and the ABSENCE of `41000`; this asserts the same
        thing on the compiled surface. It exists beside the parametrised gate below
        rather than inside it because the two catch opposite mistakes, and the gate
        alone is blind to this one: the gate only inspects columns whose header ends
        `(ms)`, so an arm that renamed its header to `(s)` without applying the scale
        factor would satisfy it while stating milliseconds under a seconds label —
        strictly worse than the defect the gate was written for.
        """
        chart = compile_chart("frontier", FRONTIER)
        (latency,) = [column for column in chart.columns if column["header"].startswith("Latency")]
        assert latency["header"] == "Latency (s)"
        stated = {row["label"]: row[latency["key"]] for row in chart.rows}
        # The payload's own numbers, divided — not merely "small enough to look like seconds".
        assert stated["model-a-3.5-fast-lite"] == pytest.approx(31.0)
        assert stated["model-b"] == pytest.approx(48.7)

    def test_a_frontier_whose_latencies_are_all_sub_second_keeps_milliseconds(self):
        """The ladder restates a chart together or not at all — it does not restate on principle."""
        payload = FRONTIER | {
            "points": [
                {"label": "a", "cost": 0.01, "quality": 0.4, "latency_ms": 300.0},
                {"label": "b", "cost": 0.02, "quality": 0.2, "latency_ms": 900.0},
            ]
        }
        chart = compile_chart("frontier", payload)
        (latency,) = [column for column in chart.columns if column["header"].startswith("Latency")]
        assert latency["header"] == "Latency (ms)"
        assert {row[latency["key"]] for row in chart.rows} == {300.0, 900.0}

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_no_type_states_a_raw_millisecond_in_its_values_table(self, viz_type):
        """Every type, because the rule is the report's and not one chart's.

        The gap this closes was a real one: the frontier arm shipped a
        `Latency (ms)` column stating 50300, and the only test that had ever held
        that behaviour lived in a deleted bespoke component. Nothing here walked
        the type set, so a new arm inherited no unit obligation at all — which is
        exactly how the next arm would repeat it.

        Read off the COLUMN HEADERS rather than the values: a header naming `ms`
        beside numbers in the thousands is the defect, and the header is where a
        type declares which unit it decided on.
        """
        chart = compile_chart(viz_type, EVERY_TYPE[viz_type])
        for column in chart.columns:
            if not column["header"].endswith("(ms)"):
                continue
            stated = [row.get(column["key"]) for row in chart.rows]
            largest = max((abs(value) for value in stated if isinstance(value, int | float)), default=0.0)
            assert largest < 1000.0, (
                f"{viz_type} states {largest:g} under '{column['header']}' — a duration past 1000ms "
                "goes in the largest unit that keeps two significant figures"
                " (state a quantity in the unit a reader thinks in)"
            )


#: A distribution whose cohorts carry enough raw values to bin rather than to rug.
BINNED = {
    "groups": [
        {
            "label": "model-b",
            "samples": [1200.0 + 400.0 * ((index * 7) % 53) / 53 for index in range(90)],
            "ci": {"low": 1250.0, "high": 1520.0, "mean": 1390.0, "level": 0.95, "variability": "across 5 runs"},
            "n": 90,
        },
        {
            "label": "deepseek",
            "samples": [1400.0 + 400.0 * ((index * 11) % 53) / 53 for index in range(90)],
            "ci": {"low": 1400.0, "high": 1900.0, "mean": 1650.0, "level": 0.95, "variability": "across 5 runs"},
            "n": 90,
        },
    ],
    "unit": "ms",
    "x_label": "pipeline_synthesis_ms",
}


class TestDistributionIsOnePanelWithAMarginal:
    """A distribution of a quantity already on an axis is a marginal.

    This type used to draw two panels: an interval panel titled with the measure,
    and a count panel below it plotting that same measure against bin names. Three
    labels for one variable on two rulers, stacked so the alignment between them
    looked meaningful. The tests here pin the collapse rather than the shape it
    collapsed from.
    """

    def test_the_figure_is_one_faceted_panel_rather_than_concatenated_ones(self):
        spec = compile_chart("distribution", DISTRIBUTION).spec
        assert "vconcat" not in spec and "hconcat" not in spec
        assert spec["facet"]["row"]["field"] == DISPLAY_FIELD

    def test_the_facet_is_not_optional_for_a_multi_series_distribution(self):
        """The type exists to show whether cohorts separate, and a pooled marginal
        is the one view that cannot answer that."""
        spec = compile_chart("distribution", BINNED).spec
        assert spec["facet"]["row"]["sort"] == ["model-b", "deepseek"]
        rows = _frame_rows(spec)
        assert {row[DISPLAY_FIELD] for row in rows} == {"model-b", "deepseek"}, "every mark is placed in a cohort's row"

    def test_the_measure_is_named_once_and_carries_the_unit(self):
        """Panel title and axis title were the same words; the axis is the one Vega
        cannot draw upright, so the unit rides into the heading instead."""
        spec = compile_chart("distribution", DISTRIBUTION).spec
        assert _title_text(spec) == "pipeline_synthesis_ms (s)"
        for encoding in _value_encodings(spec):
            assert encoding["axis"]["title"] is None

    def test_every_row_is_drawn_against_one_shared_domain(self):
        """Spreads scaled to their own values invite a comparison they cannot support."""
        spec = compile_chart("distribution", DISTRIBUTION).spec
        # The marginal's own height is the one quantitative channel that is NOT the
        # shared value axis — it is px within the row, and shares nothing with it.
        domains = {
            tuple(encoding["scale"]["domain"]) for encoding in _value_encodings(spec) if encoding["field"] != RISE_FIELD
        }
        assert len(domains) == 1, f"the rows are read against {len(domains)} rulers"

    def test_the_value_axis_is_drawn_once_under_the_last_row(self):
        """A facet shares its x scale and draws the axis on the outer edge only, so
        one axis definition reaches every cell and the picture carries one rule."""
        spec = compile_chart("distribution", DISTRIBUTION).spec
        assert "facet" in spec and "layer" in spec["spec"]
        svg = render_svg(spec, theme="dark")
        assert svg.count('aria-label="X-axis') == 1, "one x axis for the whole figure"


class TestDistributionSpec:
    def test_the_shape_is_drawn_from_the_values_not_from_the_interval(self):
        """Shape comes from values: recovering a distribution from two endpoints assumes one nobody stated."""
        spec = compile_chart("distribution", DISTRIBUTION).spec
        marks = _layer_marks(spec)
        assert "rule" in marks, "the interval itself"
        assert marks.count("point") == 1, "the estimate marker, and only it"
        rug = _mark_layer(spec, "tick")
        assert [row[ANCHOR_FIELD] for row in _layer_rows(spec, rug)] == [1.2, 1.45, 1.31, 1.6], (
            "one tick per observation, at the value it was observed at"
        )

    def test_a_group_with_only_an_interval_says_its_shape_is_unknown(self):
        payload = {
            "groups": [
                {
                    "label": "a",
                    "ci": {"low": 1.0, "high": 3.0, "mean": 2.0, "level": 0.95, "variability": "across 5 runs"},
                }
            ],
            "unit": "s",
        }
        assert compile_chart("distribution", payload).rows[0]["shape"] == "unknown — interval only"

    def test_pre_binned_buckets_never_reach_the_value_axis(self):
        """Their bins are label strings; placing them would invent edges the payload
        never gave, and drawing them against bin NAMES is the second ruler the marginal rule
        removes. So the shape is absent from the picture — and stated as absent."""
        chart = compile_chart("distribution", DISTRIBUTION)
        assert "0-1s" not in json.dumps(chart.spec), "no bin name is drawn anywhere"
        # Named by what the reader would have SEEN — the per-run ticks — rather than
        # by "shape", which is this module's internal word for the distribution and
        # could equally mean the mark, the interval, or the row.
        assert "Individual runs not drawn for deepseek" in _subtitle(chart.spec)
        assert chart.rows[1]["shape"] == "0-1s: 2; 1-2s: 7", "the counts stay exact in the values table"

    def test_a_buckets_only_payload_still_draws_its_counts(self):
        """Nothing places on the value axis, so there is no axis for a marginal to be
        marginal TO — and the rule against a second panel of one quantity has nothing
        to bite on, because there is no first panel. The counts are the chart.

        Buckets alone is a conforming payload: the generator is told to use them
        instead of samples at larger n, and a group needs only one of the three
        spreads.
        """
        payload = {
            "groups": [
                {"label": "model-b", "buckets": [{"range": "0-1s", "count": 2}, {"range": "1-2s", "count": 7}], "n": 9},
                {
                    "label": "deepseek",
                    "buckets": [{"range": "0-1s", "count": 5}, {"range": "1-2s", "count": 4}],
                    "n": 9,
                },
            ],
            "unit": "s",
            "x_label": "pipeline_synthesis",
        }
        chart = compile_chart("distribution", payload)
        assert len(chart.spec["vconcat"]) == 1, "the counts view is the only view"
        assert _layer_marks(chart.spec) == ["bar"], "only the bucket bars draw"
        assert chart.rows[0]["shape"] == "0-1s: 2; 1-2s: 7", "the values table reports what the shape rests on"

    def test_a_counts_only_figure_keeps_both_its_own_heading_and_the_measures(self):
        """The one view names what it counts; the figure names what was measured.

        Two different statements — *observations* and *pipeline_synthesis* — and a
        single view flattened into the figure would have had one `title` slot for
        them, so the panel's heading would have overwritten the figure's silently.
        """
        payload = {
            "groups": [{"label": "model-b", "buckets": [{"range": "0-1s", "count": 2}], "n": 2}],
            "unit": "s",
            "x_label": "pipeline_synthesis",
        }
        spec = compile_chart("distribution", payload).spec
        assert _title_text(spec) == "pipeline_synthesis"
        assert spec["vconcat"][0]["title"] == "observations"

    def test_a_group_that_places_anything_puts_every_group_in_the_faceted_panel(self):
        """One drawable estimate is enough: the counts then have a first panel to be
        a second one of, so the counts-only shape is not reached."""
        spec = compile_chart("distribution", DISTRIBUTION).spec
        assert "vconcat" not in spec
        assert "facet" in spec


class TestTheMarginalIsRugOrBinnedBySampleCount:
    """Below ~40 observations draw every one; past it, bin.

    At n=5 a histogram is a bin-width decision imposed on data small enough to show
    whole, and past the boundary the ticks merge into a smear that states nothing.
    """

    @staticmethod
    def _one_group(count):
        return {
            "groups": [
                {
                    "label": "a",
                    "samples": [1.0 + index / count for index in range(count)],
                    "ci": {"low": 1.0, "high": 2.0, "mean": 1.5, "level": 0.95, "variability": "across 5 runs"},
                    "n": count,
                }
            ],
            "unit": "s",
            "x_label": "m",
        }

    @staticmethod
    def _drawn(spec, kind):
        """The rows of one kind, found by what the layer filters for rather than by
        its mark — the rug and an interval's caps are both `tick` marks."""
        return [row for row in _frame_rows(spec) if row[KIND_FIELD] == kind]

    def test_at_the_boundary_every_observation_is_still_its_own_tick(self):
        limit = geometry()["rug_max_per_series"]
        spec = compile_chart("distribution", self._one_group(limit)).spec
        assert len(self._drawn(spec, "rug")) == limit
        assert not self._drawn(spec, "bin"), "nothing is binned at the boundary"

    def test_one_observation_past_it_the_marginal_bins(self):
        limit = geometry()["rug_max_per_series"]
        spec = compile_chart("distribution", self._one_group(limit + 1)).spec
        bins = self._drawn(spec, "bin")
        assert bins, "past the boundary the marginal is a binned band"
        assert sum(row["count"] for row in bins) == limit + 1, "every observation lands in exactly one bin"
        assert not self._drawn(spec, "rug"), "and no rug survives"

    def test_the_bins_line_up_across_rows_rather_than_following_each_row_s_range(self):
        """Rows binned to their own ranges are two histograms drawn to two rulers.

        Checked by rebuilding the one grid the shared domain implies and demanding
        every drawn centre sit on it. The cohorts here span different ranges, so a
        per-group binning puts each row's first centre half a bin above its OWN
        minimum — off this grid, in both rows, by construction.
        """
        spec = compile_chart("distribution", BINNED).spec
        rows = [row for row in _frame_rows(spec) if row[KIND_FIELD] == "bin"]
        assert {row[DISPLAY_FIELD] for row in rows} == {"model-b", "deepseek"}, "both cohorts bin"
        low, high = next(
            encoding["scale"]["domain"] for encoding in _value_encodings(spec) if encoding["field"] == ANCHOR_FIELD
        )
        bins = round(geometry()["plot_width"] / _mark_layer(spec, "bar")["mark"]["width"])
        step = (high - low) / bins
        grid = {round(low + (index + 0.5) * step, 6) for index in range(bins)}
        assert {round(row[ANCHOR_FIELD], 6) for row in rows} <= grid, "a bin centre sits off the figure's own grid"
        assert len({round(row[ANCHOR_FIELD], 6) for row in rows}) > 1, "and more than one bin is drawn"

    def test_the_bin_width_is_compiled_rather_than_left_to_the_renderer(self):
        """`continuousBandSize` is the documented property and vl-convert ignores it,
        leaving every bin at Vega's 5px default — a row of sticks, not a histogram."""
        bar = _mark_layer(compile_chart("distribution", BINNED).spec, "bar")
        assert bar["mark"]["width"] > 5

    def test_bin_heights_are_shared_across_rows_rather_than_normalised_per_row(self):
        """A bin drawn taller in one row than another has to mean a larger count."""
        thin = dict(BINNED["groups"][0], label="thin", samples=BINNED["groups"][0]["samples"][:45], n=45)
        payload = {**BINNED, "groups": [BINNED["groups"][0], thin]}
        rows = [row for row in _frame_rows(compile_chart("distribution", payload).spec) if row[KIND_FIELD] == "bin"]
        tallest = {
            row[DISPLAY_FIELD]: max(r[RISE_FIELD] for r in rows if r[DISPLAY_FIELD] == row[DISPLAY_FIELD])
            for row in rows
        }
        assert tallest["thin"] < tallest["model-b"], "half the observations must not draw the same height"

    def test_a_figure_carrying_marginals_gets_the_taller_row_step(self):
        """Enough rows that the plot's minimum height is not what decides it — at two
        rows the floor exceeds both steps and the comparison would pass vacuously."""
        sizes = geometry()
        interval = {"low": 1.0, "high": 2.0, "mean": 1.5, "level": 0.95, "variability": "across 5 runs"}
        bare = [{"label": f"g{index}", "ci": interval} for index in range(4)]
        sampled = [group | {"samples": [1.0, 1.4, 1.9]} for group in bare]
        heights = {
            key: max(
                _sizes(compile_chart("distribution", {"groups": groups, "unit": "s", "x_label": "m"}).spec, "height")
            )
            for key, groups in (("bare", bare), ("sampled", sampled))
        }
        assert heights["bare"] == sizes["row_step"]
        assert heights["sampled"] == sizes["row_step_marginal"]


class TestCoverageIsEncodedInTheMark:
    """Uncertainty belongs in the mark, not only in the caption.

    Drawing every interval identically while a caption explains that they are not
    comparable tells the reader one thing in prose and the opposite in the picture.
    """

    @staticmethod
    def _payload(level):
        interval = {"low": 1.0, "high": 3.0, "mean": 2.0, "variability": "across 5 runs", "level": level}
        return {"groups": [{"label": "a", "ci": interval}], "unit": "s", "x_label": "m"}

    def test_a_recorded_coverage_level_draws_hard_caps_at_both_bounds(self):
        spec = compile_chart("distribution", self._payload(0.95)).spec
        caps = [row for row in _frame_rows(spec) if row[KIND_FIELD] == "cap"]
        assert sorted(row[ANCHOR_FIELD] for row in caps) == [1.0, 3.0]

    def test_no_gradient_fill_reaches_the_spec(self):
        """A gradient names its stop COLOURS, and a compiled spec carries none — a
        stop baked in here would draw the dark theme's ink in the light one."""
        assert "gradient" not in json.dumps(compile_chart("distribution", self._payload(0.95)).spec)


class TestCrossArmVariability:
    """Where a chart's arms disagree about what their intervals span, say so."""

    def test_arms_sharing_a_source_state_it_once(self):
        assert compile_chart("distribution", DISTRIBUTION).disclosures == [
            "Intervals are 95% CIs.",
            "Intervals span across 5 runs.",
        ]

    def test_arms_spanning_different_things_are_named_as_incomparable(self):
        """The case the retired component had a branch for and could never reach."""
        payload = {
            "groups": [
                {
                    "label": "a",
                    "ci": {"low": 1.0, "high": 3.0, "mean": 2.0, "variability": "across 5 runs", "level": 0.95},
                },
                {
                    "label": "b",
                    "ci": {"low": 1.0, "high": 3.0, "mean": 2.0, "variability": "across the 12 cases", "level": 0.95},
                },
            ],
            "unit": "s",
        }
        disclosed = _disclosed(compile_chart("distribution", payload))
        assert "so their widths are not comparable" in disclosed
        assert "across 5 runs" in disclosed and "across the 12 cases" in disclosed


class TestTheCoverageLevelIsStated:
    """A 50% band and a 95% band over the same values are different widths.

    `level` is validated as a real coverage level and asked for by the generator
    prompt, so an interval that states one must draw distinguishably from one that
    states another — otherwise the reader takes coverage for spread. The retired
    `DistributionPlot` labelled it `95% CI`; a disclosure line is where it lands now,
    because that is the string both surfaces show beside the picture.
    """

    @staticmethod
    def _one_group(level=0.95, second_level=None):
        groups = [
            {"label": "a", "ci": {"low": 1.0, "high": 3.0, "mean": 2.0, "variability": "across 5 runs", "level": level}}
        ]
        if second_level is not None:
            groups.append(
                {
                    "label": "b",
                    "ci": {"low": 1.0, "high": 3.0, "mean": 2.0, "variability": "across 5 runs", "level": second_level},
                }
            )
        return {"groups": groups, "unit": "s"}

    def test_a_stated_level_is_named_beside_what_the_interval_spans(self):
        chart = compile_chart("distribution", self._one_group(level=0.95))
        # Two facts, two lines: the coverage level, then what the interval spans.
        assert chart.disclosures == ["Interval is a 95% CI.", "Interval spans across 5 runs."]

    def test_two_levels_draw_differently_because_a_disclosure_names_which(self):
        """The defect this pins: without the level the two compile identically."""
        fifty = _disclosed(compile_chart("distribution", self._one_group(level=0.5)))
        ninety_five = _disclosed(compile_chart("distribution", self._one_group(level=0.95)))
        assert "50% CI" in fifty and "95% CI" in ninety_five
        assert fifty != ninety_five

    def test_arms_at_different_levels_are_named_as_not_sharing_one(self):
        """Naming one would extend it to an arm computed at another."""
        disclosed = _disclosed(compile_chart("distribution", self._one_group(level=0.5, second_level=0.95)))
        assert "not all the same coverage level (50%; 95%)" in disclosed
        # The consequence is stated, whichever clause order this sentence takes.
        assert "widths are not comparable" in disclosed.lower()

    def test_the_level_reaches_the_null_result_disclosures_too(self):
        payload = {
            "groups": [
                {
                    "label": "a",
                    "ci": {"low": 1.0, "high": 2.0, "mean": 1.5, "level": 0.9, "variability": "across 5 runs"},
                },
                {
                    "label": "b",
                    "ci": {"low": 1.2, "high": 2.2, "mean": 1.7, "level": 0.9, "variability": "across 5 runs"},
                },
            ]
        }
        chart = compile_chart("null_result", payload)
        assert "Intervals are 90% CIs." in _disclosed(chart)
        assert "Intervals are 90% CIs." in chart.spec["description"], "the gated description carries it as well"


class TestNullResultGeometry:
    def test_the_overlap_is_stated_as_geometry_and_never_as_a_verdict(self):
        """Two marginal intervals can overlap while the difference between the means is
        real; reading overlap as a null once published one over arms 33% apart."""
        disclosed = _disclosed(compile_chart("null_result", NULL_RESULT))
        assert "Intervals overlap on [0.74, 0.85]." in disclosed
        assert "Overlap alone does not establish a null." in disclosed

    def test_non_overlapping_arms_state_that_and_claim_nothing_further(self):
        """Non-overlap does not refute the finding's own conclusion either."""
        payload = {
            "groups": [
                {
                    "label": "a",
                    "ci": {"low": 1.0, "high": 2.0, "mean": 1.5, "variability": "across 5 runs", "level": 0.95},
                },
                {
                    "label": "b",
                    "ci": {"low": 3.0, "high": 4.0, "mean": 3.5, "variability": "across 5 runs", "level": 0.95},
                },
            ]
        }
        disclosed = _disclosed(compile_chart("null_result", payload))
        assert compile_chart("null_result", payload).disclosures[0] == "Intervals do not overlap."
        assert "significan" not in disclosed

    def test_the_overlap_band_is_omitted_when_the_arms_do_not_overlap(self):
        payload = {
            "groups": [
                {
                    "label": "a",
                    "ci": {"low": 1.0, "high": 2.0, "mean": 1.5, "level": 0.95, "variability": "across 5 runs"},
                },
                {
                    "label": "b",
                    "ci": {"low": 3.0, "high": 4.0, "mean": 3.5, "level": 0.95, "variability": "across 5 runs"},
                },
            ]
        }
        assert "rect" not in _layer_marks(compile_chart("null_result", payload).spec)
        assert "rect" in _layer_marks(compile_chart("null_result", NULL_RESULT).spec)

    def test_the_mechanism_reaches_the_disclosures(self):
        """A null without one is a null the generator has not established."""
        assert NULL_RESULT["mechanism"] in _disclosed(compile_chart("null_result", NULL_RESULT))


class TestDeltaTableAxis:
    def test_the_axis_carries_relative_change_so_lengths_are_comparable(self):
        """Rows scaled to their own endpoints make length a constant that reads
        as magnitude — a 3% difference drawing exactly like a tenfold one."""
        bars = _mark_layer(compile_chart("delta_table", DELTA_TABLE).spec, "bar")
        assert bars["encoding"]["x"]["field"] == "change"
        assert bars["encoding"]["x"]["axis"]["format"] == "+.0%"

    def test_the_change_bar_is_drawn_from_zero(self):
        """A length measured from the baseline, so the baseline rule applies to it."""
        scale = _mark_layer(compile_chart("delta_table", DELTA_TABLE).spec, "bar")["encoding"]["x"]["scale"]
        assert scale["zero"] is True
        assert scale["domain"][0] < 0 < scale["domain"][1], "the drawn domain holds the baseline the flag claims"

    def test_the_axis_is_floored_so_a_set_of_small_changes_renders_small(self):
        payload = {"rows": [{"metric": "m", "a": 100.0, "b": 101.0}, {"metric": "n", "a": 100.0, "b": 102.0}]}
        domain = _mark_layer(compile_chart("delta_table", payload).spec, "bar")["encoding"]["x"]["scale"]["domain"]
        assert domain == [-STATED_CHANGE_FLOOR, STATED_CHANGE_FLOOR], "a 1-2% set must not fill the axis"

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({"rows": [{"metric": "llm_ms", "a": 16162.0, "b": 242.0}]}, id="one_row"),
            pytest.param(DELTA_TABLE, id="mixed_signs"),
            pytest.param(
                {"rows": [{"metric": "m", "a": 100.0, "b": 101.0}, {"metric": "n", "a": 100.0, "b": 102.0}]},
                id="under_floor",
            ),
        ],
    )
    def test_no_row_is_drawn_on_the_axis_boundary(self, payload):
        """A mark centred on the domain's end is drawn half outside the plot.

        The axis took the largest absolute change EXACTLY, so the row that decided it
        landed on the boundary every time — and the point that marks the value is a
        disc, not a hairline, so it rendered as a clipped half-circle jammed against
        the identity labels. Every `delta_table` has a largest row, so every one of
        them carried it.

        Held over the axis rather than over the numbers: `room` is the same value→px
        conversion the compiler places marks with, so a domain that satisfies it here
        is one the renderer draws clear.
        """
        spec = compile_chart("delta_table", payload).spec
        domain = _mark_layer(spec, "bar")["encoding"]["x"]["scale"]["domain"]
        axis = ValueAxis(title="", low=domain[0], high=domain[1], from_zero=True, plot_span=spec["width"])
        for row in spec["data"]["values"]:
            assert abs(row["change"]) < domain[1], f"{row['metric']} sits on the domain's own end"
            assert axis.room(row["change"]) >= _drawn_point_radius(spec), (
                f"{row['metric']}'s point is drawn past the plot edge"
            )

    def test_the_clearance_is_the_marks_own_radius_and_not_a_proportion(self):
        """Otherwise it drifts with the data, which is the defect one step along.

        The row that decides the axis is left exactly the point's radius — no more,
        because axis spent on nothing is resolution taken from every other row, and no
        less, because less is the clipping above. A percentage of the reach would give
        a 1000% change ten times the gap a 100% one gets, for one mark of one size.
        """
        for change in (0.5, 5.0, 50.0):
            payload = {"rows": [{"metric": "m", "a": 1.0, "b": 1.0 + change}]}
            spec = compile_chart("delta_table", payload).spec
            domain = _mark_layer(spec, "bar")["encoding"]["x"]["scale"]["domain"]
            axis = ValueAxis(title="", low=domain[0], high=domain[1], from_zero=True, plot_span=spec["width"])
            assert axis.room(change) == pytest.approx(_drawn_point_radius(spec)), (
                f"the gap moved with the data at {change:+.0%}"
            )

    def test_the_floor_is_not_paid_for_twice(self):
        """The clearance widens the DATA's reach, never the floor.

        A floored axis already leaves its largest mark most of the plot, so padding the
        floor as well would move a number the report states — and the floor
        is stated: ten percent is the reference for "small", not ten-point-one.
        """
        payload = {"rows": [{"metric": "m", "a": 100.0, "b": 100.0}]}
        domain = _mark_layer(compile_chart("delta_table", payload).spec, "bar")["encoding"]["x"]["scale"]["domain"]
        assert domain == [-STATED_CHANGE_FLOOR, STATED_CHANGE_FLOOR], "an unchanged row still gets the stated floor"

    def test_a_row_that_cannot_be_positioned_is_named_rather_than_dropped(self):
        """The values table still counts it, so an unexplained gap reads as a bug."""
        payload = {
            "rows": [
                {"metric": "ok", "a": 1.0, "b": 2.0},
                {"metric": "from_zero", "a": 0.0, "b": 3.0},
                {"metric": "stop_mode", "data_type": "categorical", "a": "budget", "b": "confidence"},
            ]
        }
        chart = compile_chart("delta_table", payload)
        assert len(chart.rows) == 3, "every row is still reported"
        assert len(chart.spec["data"]["values"]) == 1, "only the positionable one is drawn"
        assert "from_zero" in _disclosed(chart) and "stop_mode" in _disclosed(chart)

    def test_an_untested_metric_reads_as_untested_not_as_a_negative_result(self):
        rows = {row["metric"]: row for row in compile_chart("delta_table", DELTA_TABLE).rows}
        assert rows["total_ms"]["effect"] == "not tested"
        assert rows["cost_usd"]["effect"] == "significant (p=0.004, d_z=1.2, n=24)"

    def test_an_effect_size_whose_row_never_stated_its_test_is_not_called_paired(self):
        """d_z and d are different quantities on different scales, not two spellings.

        The payload key is `d_z` for historical reasons and decides nothing; the
        row's `paired` does. Absent it, the compiled table says Cohen's d — the
        claim a row that described no test is entitled to, and the one the
        browser kit makes of the same stored dict.
        """
        payload = {
            "rows": [{"metric": "cost_usd", "a": 0.011, "b": 0.019, "d_z": 1.2, "p": 0.004, "significant": True}]
        }

        effect = compile_chart("delta_table", payload).rows[0]["effect"]

        assert effect == "significant (p=0.004, d=1.2)"


class TestTheDumbbellIsAPointOnAConnector:
    """Which of the two marks reads as the quantity, at every row count.

    The arm draws a thin connector from zero and a point at the value, and the point
    is the one that IS the value — a 28px bar with a marker at its end is two marks
    competing to be read as the quantity. That ordering has to survive the row count,
    and it did not: the connector was a fraction of the y band, the band is the plot
    height divided by the rows, and the plot height has a 168px floor. So one row put
    18% of 168px — a 30px connector — under an 8.9px point, and the chart read as a
    plain bar. The dumbbell was inverted by having only one row to draw.
    """

    def _connector(self, rows):
        payload = {"rows": [{"metric": f"m{index}", "a": 100.0, "b": 100.0 + index + 1} for index in range(rows)]}
        return _mark_layer(compile_chart("delta_table", payload).spec, "bar")["mark"]["height"]

    def test_the_connector_is_stated_in_px_not_as_a_share_of_the_band(self):
        """`_bar_mark`'s own rule: a thickness that moves with the row count encodes it."""
        heights = {rows: self._connector(rows) for rows in (1, 2, 3, 7)}
        assert len(set(heights.values())) == 1, f"the connector moved with the row count: {heights}"

    def test_the_point_is_the_larger_mark_where_the_band_is_widest(self):
        """One row, where the band is the whole plot — the case that inverted it."""
        spec = compile_chart("delta_table", {"rows": [{"metric": "m0", "a": 100.0, "b": 101.0}]}).spec
        assert self._connector(1) < 2 * _drawn_point_radius(spec), "the connector is not recessive under the point"

    def test_a_value_label_never_asks_for_the_knockout_here(self):
        """What this arm draws at a value is a 3px connector and a point, so there is no fill.

        The knockout IS the chart surface, so a label taking it inward of these marks
        is painted on the background in the background's own colour. It is reachable on
        every chart of this type: the row that decides the axis has only the point's
        radius beyond it, which is far short of the label clearance, so its number is
        always the one pushed inward.
        """
        spec = compile_chart("delta_table", {"rows": [{"metric": "llm_ms", "a": 16162.0, "b": 242.0}]}).spec
        labels = _mark_layers(spec, "text")
        assert labels, "this shape must still write its values"
        for layer in labels:
            assert "style" not in layer["mark"], "a mark with no fill under its label must not take the knockout"


class TestTheValuesTableNamesEveryColumnOnce:
    """A repeated column key is a column a renderer cannot address.

    The browser uses the key as the React key on every `<th>`/`<td>` and as the row
    key; `values_as_drawn()` pads by position and reads each row by key, so a
    duplicate silently prints one column's value under both headers. Nothing
    asserted this on any type until a `sweep_ranking` lever named `n` was found able
    to collide with the compiler's own count column — which is now foreclosed by
    keying levers in their own namespace rather than refused by a reserved-name list.
    """

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_no_type_emits_two_columns_under_one_key(self, viz_type):
        keys = [column["key"] for column in compile_chart(viz_type, EVERY_TYPE[viz_type]).columns]
        assert len(keys) == len(set(keys)), (
            f"{viz_type} emits a duplicate column key: {sorted(k for k in keys if keys.count(k) > 1)}"
        )

    @pytest.mark.parametrize("collides", ["ranked", "secondary", "n"])
    def test_a_sweep_lever_may_answer_to_a_measure_column_name(self, collides):
        """A lever named for a measure column keeps its own level, and is drawn under its own name.

        This was a real collision: the level was overwritten by the measure's value
        in every row, and two columns drew under one key. It is not refused now, it
        is impossible — levers are keyed in their own namespace — so the assertion is
        that the payload COMPILES and the level survives. A blocklist would have made
        this a refusal instead, which costs a whole paid analysis to say "rename your
        lever" about a name that is perfectly good.
        """
        payload = copy.deepcopy(SWEEP_RANKING)
        payload.pop("dimensions", None)
        for index, row in enumerate(payload["rows"]):
            row["config"][collides] = str(index % 2)
        chart = compile_chart("sweep_ranking", payload)

        assert collides in [column["header"] for column in chart.columns], "the lever lost its own name in the header"
        lever_key = next(column["key"] for column in chart.columns if column["header"] == collides)
        assert lever_key != collides, "the lever is keyed in the measure namespace, which is the collision itself"
        drawn = {row[lever_key] for row in chart.rows}
        assert drawn == {"0", "1"}, f"the lever's levels did not survive: {sorted(drawn)}"

    def test_no_measure_column_can_land_in_the_lever_namespace(self):
        """The other half of the separation, and the half a future column could break.

        The prefix forecloses a lever reaching a measure's key. Nothing structural
        stops the reverse — someone adding a column keyed `lever.something` — and that
        would reopen the collision from the side the namespace was meant to close.
        Read off a real compilation so the arm is the source.
        """
        payload = copy.deepcopy(SWEEP_RANKING)
        names = set(payload["rows"][0]["config"])
        columns = compile_chart("sweep_ranking", payload).columns
        lever_keys = {column["header"]: column["key"] for column in columns if column["header"] in names}
        assert set(lever_keys) == names, "every lever is drawn as a column under its own name"
        # The namespace, read off the lever columns themselves: each key is the namespace then the name.
        prefixes = {key.removesuffix(name) for name, key in lever_keys.items() if key.endswith(name)}
        assert len(prefixes) == 1 and len(lever_keys) == len(names), f"the levers share no one namespace: {lever_keys}"
        (prefix,) = prefixes
        assert prefix, "a lever keyed by its bare name IS the measure namespace"
        appended = [column["key"] for column in columns if column["key"] not in set(lever_keys.values())]
        assert appended, "the fixture stopped exercising the measure columns"
        for key in appended:
            assert not key.startswith(prefix), (
                f"the sweep arm appends {key!r}, which is inside the lever namespace — a lever of that name would "
                "collide with it, which is exactly what the prefix exists to prevent"
            )


class TestTheCaptionIsTheAuthorsAlone:
    """The payload writes the caption; the compiler writes disclosures, and never onto it.

    A template can only narrate how it drew, because that is the only thing it
    knows — which is why compiler prose kept describing marks instead of meaning.
    What does NOT move is the mechanical content: the omission band sums what the
    producer declared it dropped with what THIS compilation truncated, the coverage
    line reads the intervals that survived absence-filtering, and both restate
    values in the axis's unit after rescaling. Those numbers do not exist when the
    payload is written, and a disclosure the producer may forget is not one.

    They used to be JOINED: the author's line, then every disclosure, as one string.
    A frontier whose author wrote one sentence reached its reader as five, and the
    reader took the whole paragraph as the author's and found it incomprehensible.
    So the two travel apart — `caption` is the author's line exactly as
    written, `disclosures` is the compiler's, one idea per line.
    """

    #: A neutral conclusion, so the parametrised cases assert the SEPARATION and nothing else.
    INSIGHT = "The lever moves the outcome and the effect holds across every arm."

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_the_caption_is_the_authors_line_and_nothing_else(self, viz_type):
        """Equality, not containment: a caption that merely STARTS with the author's line
        is the defect, since that is exactly the shape a join produces."""
        chart = compile_chart(viz_type, {**EVERY_TYPE[viz_type], "caption": self.INSIGHT})
        assert chart.caption == self.INSIGHT

    def test_the_shared_fixtures_leave_the_compiler_something_to_disclose(self):
        """The equality above only bites where there was compiler text to extend it with.

        A type whose shared fixture discloses nothing would pass that test under the old
        join as well, so the population it iterates is asserted rather than assumed.
        """
        silent = sorted(
            viz_type for viz_type, payload in EVERY_TYPE.items() if not compile_chart(viz_type, payload).disclosures
        )
        assert silent == ["delta_table"], silent

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_the_disclosures_are_the_same_whether_or_not_the_author_wrote_a_caption(self, viz_type):
        """An arm never reads the payload's caption, so the author cannot suppress or reword a disclosure."""
        bare = compile_chart(viz_type, EVERY_TYPE[viz_type]).disclosures
        captioned = compile_chart(viz_type, {**EVERY_TYPE[viz_type], "caption": self.INSIGHT}).disclosures
        assert captioned == bare
        assert not any(self.INSIGHT in line for line in captioned)

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_no_intent_builder_writes_the_caption(self, viz_type):
        """The caption is set in one place. A builder writing into it would be overwritten by
        `chart_intent` — a disclosure lost with no error — so a builder leaves it empty."""
        from threetears.evals.analysis.viz.intents import INTENTS
        from threetears.evals.analysis.viz.payloads import parse_payload

        payload = {**EVERY_TYPE[viz_type], "caption": self.INSIGHT}
        assert INTENTS[viz_type](parse_payload(viz_type, payload)).caption == ""

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_every_disclosure_is_one_non_empty_line(self, viz_type):
        """One idea per line: no blank entry for a surface to render as an empty row, no
        line carrying a break that would re-fuse two ideas into one paragraph."""
        for line in compile_chart(viz_type, EVERY_TYPE[viz_type]).disclosures:
            assert line and line == line.strip() and "\n" not in line, repr(line)

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_every_type_accepts_the_field(self, viz_type):
        """One field on the shared base, so the contract widened once rather than seven times.

        `_VizPayload` forbids unknown keys — that is what catches `cells` for
        `points` — so a type that did NOT inherit the field would refuse the payload
        outright rather than ignore it.
        """
        assert compile_chart(viz_type, {**EVERY_TYPE[viz_type], "caption": self.INSIGHT}).caption

    def test_a_payload_with_no_caption_is_served_none(self):
        """No placeholder, and no disclosure promoted into the empty slot."""
        chart = compile_chart("breakdown", PAYLOAD)
        assert chart.caption == ""
        assert chart.disclosures == ["The parts divide a total of 100% over n=49."]

    def test_an_empty_caption_reads_as_an_absent_one(self):
        """A caption of no words and a caption nobody wrote are the same absence."""
        assert compile_chart("breakdown", {**PAYLOAD, "caption": "   "}).caption == ""

    def test_the_authors_punctuation_is_left_as_written(self):
        """Nothing is joined onto the caption, so nothing needs a terminator added to it."""
        assert compile_chart("breakdown", {**PAYLOAD, "caption": "budget dominates"}).caption == "budget dominates"

    def test_an_insight_alone_carries_no_disclosures(self):
        """A chart with nothing to disclose renders the payload's line and stops."""
        bare = {"parts": [{"label": "a", "value": 3}, {"label": "b", "value": 1}], "unit": "runs"}
        chart = compile_chart("breakdown", {**bare, "caption": self.INSIGHT})
        assert (chart.caption, chart.disclosures) == (self.INSIGHT, [])


class TestTheProseIsNotGated:
    """No line beside the chart is read by a check.

    `null_result` carries the payload's `mechanism` verbatim as a disclosure line, a
    frontier carries each `disqualified_reason` the same way, and every type carries
    the author's caption. All of it is served as text — a `<figcaption>`, lines of tool
    output — and never reaches the rasteriser, so a colour notation written there is
    words, not a colour; and what the prose says is the reporter eval's question,
    because code checks the structure of model output and never its prose. A chart is refused for its SPEC only.
    """

    def test_a_colour_notation_in_a_mechanism_is_carried_not_refused(self):
        mechanism = "The arms overlap; see the oklch(0.70 0.22 295) band."
        assert compile_chart("null_result", {**NULL_RESULT, "mechanism": mechanism}).disclosures[-1] == mechanism

    def test_a_colour_notation_in_a_frontier_disclosure_is_carried_not_refused(self):
        """The type the old refusal was observed on: its disqualification lines are text."""
        points = [dict(point) for point in FRONTIER["points"]]
        points[-1] = {**points[-1], "disqualified": True, "disqualified_reason": "drawn oklch(0.70 0.22 295)"}
        compiled = compile_chart("frontier", {**FRONTIER, "points": points})
        assert any("drawn oklch(0.70 0.22 295)" in line for line in compiled.disclosures), compiled.disclosures

    def test_a_colour_notation_in_the_authors_caption_is_carried_not_refused(self):
        caption = "The lab(50% 40 59) bar leads."
        assert compile_chart("breakdown", {**PAYLOAD, "caption": caption}).caption == caption

    def test_a_mechanism_is_its_own_line_verbatim(self):
        """Nothing the mechanism says is refused — a null needs one to be established — and it
        is carried whole, as the last line, so a reader can find where the author's reason starts."""
        assert compile_chart("null_result", NULL_RESULT).disclosures[-1] == NULL_RESULT["mechanism"]


class TestTheCompilationCarriesTheIntent:
    """A compilation adds a picture and says nothing of its own.

    Every field of a `CompiledChart` but `spec` is the intent's, carried across unchanged — so a
    caller holding a compilation reads the values table, caption and disclosures the intent decided,
    and a renderer cannot quietly reword what a chart claims.
    """

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_every_field_but_the_spec_is_the_intents(self, viz_type):
        payload = {**EVERY_TYPE[viz_type], "caption": "an author's line"}
        intent = chart_intent(viz_type, payload)
        compiled = compile_chart(viz_type, payload)

        assert compiled.intent == intent
        assert compiled.columns == [{"key": column.key, "header": column.header} for column in intent.columns]
        assert compiled.rows == intent.rows
        assert (compiled.caption, compiled.disclosures, compiled.title, compiled.unit) == (
            intent.caption,
            intent.disclosures,
            intent.title,
            intent.unit,
        )
        assert compiled.values_as_drawn() == intent.values_as_drawn()

    def test_the_compiled_fields_are_the_intent_the_spec_and_the_intents_projections(self):
        """A field added to the compilation must be one of the intent's, or the spec — never a second opinion."""
        assert {entry.name for entry in fields(CompiledChart)} == {
            "intent",
            "spec",
            "columns",
            "rows",
            "unit",
            "caption",
            "disclosures",
            "title",
        }


class TestNothingDrawableIsRefusedRatherThanDrawnEmpty:
    """A legal payload the chart cannot place is refused, not compiled to a frame.

    The generation gate is where a payload can still be corrected by retrying, and
    an empty axis is the same uninterpretable card the payload contract exists to
    remove — it just costs a paid regeneration to notice.
    """

    def test_a_comparison_baselined_entirely_at_zero_is_refused(self):
        """A relative axis cannot place a change from zero, so every row falls out."""
        with pytest.raises(PayloadError, match="baselined at zero"):
            compile_chart("delta_table", {"rows": [{"metric": "m", "a": 0.0, "b": 3.0, "unit": "runs"}]})

    def test_a_zero_baseline_beside_a_placeable_row_still_compiles(self):
        """One unplaceable row is named in the caption, not treated as fatal."""
        chart = compile_chart(
            "delta_table",
            {"rows": [{"metric": "m", "a": 0.0, "b": 3.0}, {"metric": "n", "a": 2.0, "b": 3.0}]},
        )
        assert [entry["metric"] for entry in chart.spec["data"]["values"]] == ["n"]
        assert "m" in _disclosed(chart)


class TestAbsentCountsAreStatedByAbsence:
    """An absent `n` is never a column of em dashes — the same rule in every case."""

    def test_a_distribution_that_counts_nothing_has_no_n_column(self):
        chart = compile_chart("distribution", {"groups": [{"label": "a", "samples": [1.0, 2.0]}]})
        assert "n" not in [column["key"] for column in chart.columns]

    def test_a_distribution_that_counts_keeps_the_column(self):
        chart = compile_chart("distribution", {"groups": [{"label": "a", "samples": [1.0, 2.0], "n": 2}]})
        assert "n" in [column["key"] for column in chart.columns]

    def test_a_null_result_that_counts_nothing_has_no_n_column(self):
        payload = {
            "groups": [
                {
                    "label": "a",
                    "ci": {"low": 1.0, "high": 2.0, "mean": 1.5, "level": 0.95, "variability": "across 5 runs"},
                },
                {
                    "label": "b",
                    "ci": {"low": 1.2, "high": 2.2, "mean": 1.7, "level": 0.95, "variability": "across 5 runs"},
                },
            ]
        }
        assert "n" not in [column["key"] for column in compile_chart("null_result", payload).columns]


class TestAMetricNameThatIsProseStillRendersItsUnit:
    """The measured failure the unit field exists for, checked directly.

    Both strings below have the shape of real values of a payload's own measure field. The
    name-sniffing this replaces split on `_` and looked for an `ms` segment, and
    neither of these has one — so both rendered raw milliseconds as bare numbers,
    and the second was captioned in seconds over millisecond values. Reading the
    unit from a field rather than a name is what makes the prose irrelevant.
    """

    PROSE_NAMES = (
        "pipeline_synthesis_ms (ms), model-a/depth=4/passes=2",
        "pipeline_search_ms (subsystem, s)",
    )

    @pytest.mark.parametrize("measure", PROSE_NAMES)
    def test_a_distribution_named_in_prose_still_restates_its_unit(self, measure):
        payload = {
            "groups": [
                {
                    "label": "a",
                    "ci": {
                        "low": 15000.0,
                        "high": 17000.0,
                        "mean": 16162.0,
                        "level": 0.95,
                        "variability": "across 5 runs",
                    },
                }
            ],
            "unit": "ms",
            "x_label": measure,
        }
        chart = compile_chart("distribution", payload)
        assert chart.unit == "s"
        # The unit rides in the panel's heading rather than on the axis: the two
        # would otherwise say the same words twice, and the axis is the one of them
        # Vega cannot state without turning them a quarter turn.
        assert _title_text(chart.spec) == f"{measure} (s)"
        assert chart.rows[0]["mean"] == pytest.approx(16.162)

    @pytest.mark.parametrize("measure", PROSE_NAMES)
    def test_a_null_result_named_in_prose_still_restates_its_unit(self, measure):
        payload = {
            "groups": [
                {
                    "label": "a",
                    "ci": {
                        "low": 15000.0,
                        "high": 17000.0,
                        "mean": 16162.0,
                        "level": 0.95,
                        "variability": "across 5 runs",
                    },
                },
                {
                    "label": "b",
                    "ci": {
                        "low": 10000.0,
                        "high": 12000.0,
                        "mean": 11040.0,
                        "level": 0.95,
                        "variability": "across 5 runs",
                    },
                },
            ],
            "unit": "ms",
            "metric": measure,
        }
        chart = compile_chart("null_result", payload)
        assert chart.unit == "s"
        assert {"key": "mean", "header": "Mean (s)"} in chart.columns

    @pytest.mark.parametrize("measure", PROSE_NAMES)
    def test_a_delta_row_named_in_prose_still_restates_its_unit(self, measure):
        chart = compile_chart("delta_table", {"rows": [{"metric": measure, "a": 16162.0, "b": 11040.0, "unit": "ms"}]})
        assert chart.rows[0]["a"] == "16.16 s"
        assert chart.rows[0]["b"] == "11.04 s"


class TestRelativeChangeAcrossSigns:
    """The relative-change axis, on baselines the happy path never supplies.

    Two `abs()` calls sit on this path and they guard different things, so each
    test below names which one it holds. `_relative_change` divides by |A|:
    dividing by a signed baseline flips the sign of every change measured from a
    negative one, so an improvement would draw as a regression. `half_span` takes
    the largest |change|: dropping that one lets a negative change lose to a
    smaller positive one, and the axis then clips the very row it was scaled for.

    Every other fixture in this file baselines positive, which is how both stayed
    unexercised when these helpers replaced the frontend's.
    """

    def test_a_negative_baseline_keeps_the_direction_of_the_change(self):
        """Holds `_relative_change`'s `abs(a)`."""
        chart = compile_chart("delta_table", {"rows": [{"metric": "drift", "a": -4.0, "b": -2.0}]})
        # B is larger than A, so the change is positive whatever the sign of A.
        assert chart.spec["data"]["values"][0]["change"] == pytest.approx(0.5)
        assert chart.rows[0]["change"] == "+50.0%"

    def test_a_negative_baseline_moving_further_negative_reads_as_a_decrease(self):
        """Holds `_relative_change`'s `abs(a)` in the other direction."""
        chart = compile_chart("delta_table", {"rows": [{"metric": "drift", "a": -4.0, "b": -6.0}]})
        assert chart.spec["data"]["values"][0]["change"] == pytest.approx(-0.5)

    def test_the_axis_half_span_is_a_magnitude_whatever_the_baselines_sign(self):
        """Holds `half_span`'s `abs()`, NOT `_relative_change`'s.

        The larger change here is the negative one, so dropping `abs` from the
        half-span makes the smaller positive change win the axis and clips the row
        the axis exists to show. This test passes under the `_relative_change`
        mutation — the divisor's sign is erased by the same `abs` it is testing —
        which is why the two are named apart rather than left to one claim.
        """
        payload = {"rows": [{"metric": "a", "a": -2.0, "b": -5.0}, {"metric": "b", "a": 4.0, "b": 5.0}]}
        spec = compile_chart("delta_table", payload).spec
        domain = _mark_layer(spec, "bar")["encoding"]["x"]["scale"]["domain"]
        # Stated as a bracket rather than as the -150% literal it used to be. The axis
        # reaches PAST its largest change by the point's radius now, so an equality on
        # the change itself would be an equality on the clipping this arm just stopped
        # doing — but the mutation is caught the same way, since a half-span taken from
        # the +25% row cannot land near 150% however it is padded.
        assert domain[0] == pytest.approx(-domain[1]), "the axis is symmetric about the baseline"
        assert 1.5 < domain[1] < 1.6, f"the -150% row must decide the axis, not the +25% one: {domain[1]}"


#: The withheld sentence the divergence lens produces for a non-nested pair, verbatim.
_NOT_CONTAINED = (
    "pipeline_synthesis_ms is not declared a component of total_ms — they share a unit but are not "
    "known to be nested, so what looks like an unexplained remainder may be two disjoint stretches "
    "of the same measure."
)

#: The withheld pair: a ~90s synthesis swing against a near-flat total_ms, remainder withheld.
ATTRIBUTION_WITHHELD = {
    "end_to_end": {"measure": "total_ms", "delta": 3000.0, "a": 44000.0, "b": 47000.0},
    "subsystem": {"measure": "pipeline_synthesis_ms", "delta": 87800.0, "a": 5200.0, "b": 93000.0},
    "unit": "ms",
    "unattributed_withheld": _NOT_CONTAINED,
    "lever": "pipeline.pipeline_model",
    "a_label": "model-a-3.5-fast-lite",
    "b_label": "model-b",
}

#: The one nested pair in the corpus — `tool_ms` is a declared component of `total_ms`.
ATTRIBUTION_EARNED = {
    "end_to_end": {"measure": "total_ms", "delta": -31800.0, "a": 58300.0, "b": 26500.0},
    "subsystem": {"measure": "tool_ms", "delta": -3800.0},
    "unit": "ms",
    "contained_by": "total_ms",
    "unattributed_delta": -28000.0,
    "lever": "prompt.pipeline_system",
    "a_label": "terse_prompt",
    "b_label": "verbose_prompt",
}


class TestAttributionNeverBalances:
    """The remainder is drawn only where it is earned, and never as a closing segment."""

    def test_a_withheld_remainder_puts_no_length_on_the_axis(self):
        """Nothing measurable stands for the unplaced movement."""
        bars = _mark_layer(compile_chart("attribution", ATTRIBUTION_WITHHELD).spec, "bar")
        assert [entry["scope"] for entry in bars["data"]["values"]] == ["End-to-end", "Subsystem"]

    def test_a_withheld_remainder_keeps_its_band_and_fills_it_with_words(self):
        """The row must survive the picture, and must not survive it EMPTY.

        Three options, one honest. Omitting the band leaves the chart silent about
        a question the finding asked. Leaving it empty is worse — a band with no
        mark sits exactly where a zero-length bar would, so it reads as "the
        movement was fully accounted for", the sharpest misreading this type
        exists to prevent. A text mark holds the row and cannot be measured
        against the value axis, which is the property the other two lack.
        """
        spec = compile_chart("attribution", ATTRIBUTION_WITHHELD).spec
        assert _mark_layer(spec, "bar")["encoding"]["y"]["sort"][-1] == "Unattributed"
        label = spec["layer"][1]
        assert label["mark"]["type"] == "text"
        # Exact, so a value cannot join the row unnoticed: everything on it is either
        # the band it occupies, the words that occupy it, or the name that band is
        # drawn under.
        assert label["data"]["values"] == [
            {"scope": "Unattributed", "delta": 0, "label": "not placeable", "display": "Unattributed"}
        ]

    def test_the_not_placeable_label_reads_back_into_the_plot_on_an_all_negative_chart(self):
        """The headline case, and the one every withheld fixture happened to miss.

        `scale: {zero: True}` pins zero to the domain's EDGE whenever the movements
        share a sign. A whole-run duration that improved while the part did not move
        is all-negative — which is the shape this type was built for — so zero
        is the RIGHT edge there, and the label's original unconditional
        `align: left, dx: +6` placed it outside the plot. Clipped, the band renders
        empty, which is precisely the "fully accounted for" reading the words exist
        to prevent: the failure mode is not a cosmetic one, it is the chart quietly
        reverting to the worst of the three options its own docstring rejects.

        Every stored withheld fixture is positive, so the suite agreed with the bug.
        Both signs are asserted here, because a fix that simply flipped the constant
        would move the defect to the other half of the domain rather than remove it.
        """
        negative = dict(ATTRIBUTION_WITHHELD) | {
            "end_to_end": {"measure": "total_ms", "delta": -33900.0},
            "subsystem": {"measure": "pipeline_synthesis_ms", "delta": -400.0},
        }
        mark = compile_chart("attribution", negative).spec["layer"][1]["mark"]
        assert (mark["align"], mark["dx"]) == ("right", -6), "the label runs off the right edge, where zero sits"

        mark = compile_chart("attribution", ATTRIBUTION_WITHHELD).spec["layer"][1]["mark"]
        assert (mark["align"], mark["dx"]) == ("left", 6), "an all-positive chart puts zero at the left edge"

    def test_an_earned_remainder_draws_its_own_bar_and_needs_no_label(self):
        spec = compile_chart("attribution", ATTRIBUTION_EARNED).spec
        drawn = _mark_layer(spec, "bar")["data"]["values"]
        assert [entry["scope"] for entry in drawn] == ["End-to-end", "Subsystem", "Unattributed"]
        assert "not placeable" not in json.dumps(spec), "an earned remainder is a bar, and says nothing besides"

    def test_the_bars_are_not_stacked(self):
        """A stack would draw the part and the remainder as segments of the whole.

        That is the waterfall this type refuses: laid end to end the two would
        close the gap exactly, which is the attribution the finding says it cannot
        make. Each movement is measured from the same zero instead.
        """
        encoding = _mark_layer(compile_chart("attribution", ATTRIBUTION_EARNED).spec, "bar")["encoding"]
        assert "stack" not in encoding["x"]
        assert encoding["x"]["scale"]["zero"] is True

    def test_the_derived_remainder_is_recessive(self):
        """It is computed rather than observed, so it must not read as a third measurement."""
        encoding = _mark_layer(compile_chart("attribution", ATTRIBUTION_EARNED).spec, "bar")["encoding"]
        assert encoding["opacity"]["condition"]["test"] == "datum.derived"
        assert encoding["opacity"]["condition"]["value"] < encoding["opacity"]["value"]

    def test_no_identity_rides_on_colour(self):
        """Same rule as every other type here — the scopes are on the axis."""
        for layer in _mark_layers(compile_chart("attribution", ATTRIBUTION_WITHHELD).spec):
            assert "color" not in layer["encoding"]


class TestAttributionValuesAsDrawn:
    """What a surface that cannot draw is told — including that the question was asked."""

    def test_the_unattributed_row_survives_a_withheld_remainder(self):
        """An omitted row makes 'we could not place this' look like 'nobody asked'."""
        chart = compile_chart("attribution", ATTRIBUTION_WITHHELD)
        assert [row["scope"] for row in chart.rows] == ["End-to-end", "Subsystem", "Unattributed"]
        assert chart.rows[2]["delta"] is None
        assert chart.values_as_drawn()[-1].startswith("Unattributed")

    def test_a_withheld_remainder_renders_as_an_em_dash_never_a_zero(self):
        """A zero would assert that the movement was fully accounted for."""
        assert "0" not in compile_chart("attribution", ATTRIBUTION_WITHHELD).values_as_drawn()[-1]
        assert "—" in compile_chart("attribution", ATTRIBUTION_WITHHELD).values_as_drawn()[-1]

    def test_an_earned_remainder_reaches_the_table_with_its_sign(self):
        chart = compile_chart("attribution", ATTRIBUTION_EARNED)
        assert chart.rows[2]["delta"] == "-28 s"

    def test_the_level_columns_are_named_after_the_levels(self):
        headers = [column["header"] for column in compile_chart("attribution", ATTRIBUTION_WITHHELD).columns]
        assert "model-a-3.5-fast-lite" in headers
        assert "model-b" in headers

    def test_the_shared_unit_is_restated_once_for_the_whole_chart(self):
        """One unit per quantity: milliseconds reaching a second are stated in seconds, chart-wide."""
        chart = compile_chart("attribution", ATTRIBUTION_WITHHELD)
        assert chart.unit == "s"
        assert _mark_layer(chart.spec, "bar")["encoding"]["x"]["axis"]["title"] == "change (s)"
        assert chart.rows[1]["b"] == "93 s"


class TestAttributionCaption:
    """The reason a remainder is missing is prose, and the reader is owed it verbatim."""

    def test_the_withheld_sentence_is_carried_word_for_word(self):
        assert _NOT_CONTAINED in _disclosed(compile_chart("attribution", ATTRIBUTION_WITHHELD))

    def test_the_compared_levels_are_named(self):
        disclosed = _disclosed(compile_chart("attribution", ATTRIBUTION_WITHHELD))
        assert "pipeline.pipeline_model" in disclosed
        assert "model-a-3.5-fast-lite → model-b" in disclosed

    def test_an_earned_remainder_states_what_it_is_the_difference_of(self):
        disclosed = _disclosed(compile_chart("attribution", ATTRIBUTION_EARNED))
        assert "total_ms minus tool_ms" in disclosed


class TestAttributionObservationCounts:
    """Two n's side by side are the plainest statement that the populations differ."""

    def test_each_movement_carries_its_own_count(self):
        chart = compile_chart(
            "attribution",
            dict(
                ATTRIBUTION_WITHHELD,
                end_to_end={"measure": "total_ms", "delta": 3000.0, "n": 60},
                subsystem={"measure": "pipeline_synthesis_ms", "delta": 87800.0, "n": 5},
            ),
        )
        assert {"key": "n", "header": "n"} in chart.columns
        assert [row.get("n") for row in chart.rows] == [60, 5, None]

    def test_a_derived_remainder_claims_no_observations(self):
        """It is arithmetic over two populations, so it has no n of its own."""
        chart = compile_chart(
            "attribution", dict(ATTRIBUTION_EARNED, end_to_end=dict(ATTRIBUTION_EARNED["end_to_end"], n=60))
        )
        assert chart.rows[2].get("n") is None
        # The remainder keeps its number and gains an em dash for the count — the
        # arithmetic is real, the observations behind it are not a single figure.
        assert "-28 s" in chart.values_as_drawn()[-1]
        assert chart.values_as_drawn()[-1].rstrip().endswith("—")

    def test_no_column_is_offered_when_nothing_counts(self):
        """A column of em dashes states an absence the payload already states by omission."""
        assert all(column["key"] != "n" for column in compile_chart("attribution", ATTRIBUTION_WITHHELD).columns)


#: Where each type's category names come from in its payload — collection key, then
#: the field on each entry.
#:
#: `attribution` is absent on purpose and not by oversight: its three scopes are
#: compiler-authored constants ("End-to-end", "Subsystem", "Unattributed"), so no
#: payload can make them long and the placement decision below is unreachable for
#: it. `TestAttributionNamesItsOwnScopes` is where that is pinned, so the omission
#: is a checked statement rather than a gap.
_CATEGORY_SOURCE: dict[str, tuple[str, str]] = {
    "breakdown": ("parts", "label"),
    "distribution": ("groups", "label"),
    "null_result": ("groups", "label"),
    "delta_table": ("rows", "metric"),
}


def _narrow_names(count: int) -> list[str]:
    """Distinct names any gutter can hold."""
    return [f"alpha-{index}" for index in range(count)]


def _wide_names(count: int) -> list[str]:
    """Distinct names no gutter can hold, even after prefix stripping.

    Distinct because every payload model refuses duplicate category names, and
    prefix-proof because they share no `/`-delimited head — stripping must not be
    able to rescue them, or this stops testing the placement it claims to.
    """
    return [f"anthropic-{index}/claude-opus-4-20250514-extended-thinking-preview" for index in range(count)]


def _renamed(viz_type: str, names) -> dict:
    """`EVERY_TYPE[viz_type]`, with its categories renamed in order.

    Args:
        viz_type: Which payload to copy.
        names: A callable taking the category count, or an explicit list of
            exactly that many names.
    """
    collection, key = _CATEGORY_SOURCE[viz_type]
    payload = copy.deepcopy(EVERY_TYPE[viz_type])
    entries = payload[collection]
    chosen = names(len(entries)) if callable(names) else names
    for entry, name in zip(entries, chosen, strict=True):
        entry[key] = name
    return payload


def _identity_encodings(node, found=None) -> list[dict]:
    """Every encoding in a compiled spec that places a category, at any depth.

    Both forms count — the axis a layered chart puts its names on, and the facet
    row a concatenated one puts them in — because a rule that reached only the
    first would be the exact shape of defect this rule went looking for:
    a bound migrated on one chart and asserted on that same chart while the other
    four kept the old one.

    A `sort` is what separates a placement from a mention. The same field also
    feeds the `text` channel of the layer that draws a name above its mark, and
    that channel places nothing — it states what to write, at coordinates the y
    encoding beside it decides.
    """
    found = [] if found is None else found
    if isinstance(node, dict):
        if node.get("field") == DISPLAY_FIELD and node.get("type") == "nominal" and "sort" in node:
            found.append(node)
        for value in node.values():
            _identity_encodings(value, found)
    elif isinstance(node, list):
        for item in node:
            _identity_encodings(item, found)
    return found


def _label_bound(encoding: dict) -> float | None:
    """The px limit a category name is drawn under here, or `None` where it has none.

    A facet states it under `header`, an axis under `axis`, and an encoding that
    dropped its axis entirely states no bound at all — the name has left the gutter
    and is drawn as a mark instead.
    """
    for key in ("axis", "header"):
        block = encoding.get(key)
        if isinstance(block, dict) and "labelLimit" in block:
            return block["labelLimit"]
    return None


def _plot_height(spec, rows):
    """A figure's whole plot area, whichever shape drew its categories.

    A chart that puts its categories on an axis states the plot's full height; one
    that puts them in facet rows states a CELL, and the figure is that many cells.
    The quantity the row-step rules are about is the first, so the second is
    multiplied back up rather than compared against a number it is not.
    """
    height = max(_sizes(spec, "height"))
    return height * rows if "facet" in spec else height


def lines_of(title):
    """A compiled title as the list of lines it draws, however it is carried."""
    if isinstance(title, dict):
        title = title["text"]
    return title if isinstance(title, list) else [title]


#: Which payload key each type builds its title from, so a test can make one long.
#:
#: `attribution` and `delta_table` assemble theirs from two payload fields rather than
#: taking one verbatim, so the key names the field whose value reaches the title.
_TITLE_SOURCE: dict[str, str] = {
    "breakdown": "measure",
    "distribution": "x_label",
    "null_result": "metric",
    "delta_table": "a_label",
}


def _measure_field(viz_type: str, title: str) -> dict:
    """The payload override that puts `title` into this type's chart title."""
    return {_TITLE_SOURCE[viz_type]: title}


class TestNamesAreDrawnWhereTheyFit:
    """The gutter holds a name or the name leaves the gutter — never a truncation.

    Series are full model IDs with no aliases, so the characters a truncation eats
    are the ones that distinguish one build from another. The bound is 176px and
    the decision is measured, which is what makes this a rule rather than a hope.
    """

    @pytest.mark.parametrize("viz_type", sorted(_CATEGORY_SOURCE), ids=sorted(_CATEGORY_SOURCE))
    def test_a_name_that_fits_the_gutter_is_drawn_in_it(self, viz_type):
        spec = compile_chart(viz_type, _renamed(viz_type, _narrow_names)).spec
        encodings = _identity_encodings(spec)
        assert encodings, f"{viz_type} draws no identity encoding"
        assert all(_label_bound(encoding) == geometry()["gutter_left"] for encoding in encodings)

    @pytest.mark.parametrize("viz_type", sorted(_CATEGORY_SOURCE), ids=sorted(_CATEGORY_SOURCE))
    def test_a_name_too_wide_for_the_gutter_moves_out_of_it(self, viz_type):
        """Every identity encoding moves, not merely the first one the walk finds."""
        spec = compile_chart(viz_type, _renamed(viz_type, _wide_names)).spec
        encodings = _identity_encodings(spec)
        assert encodings
        for encoding in encodings:
            # An axis is dropped outright; a facet header stays but loses its bound,
            # because there it is the panel's own heading rather than a gutter column.
            assert encoding.get("axis", "absent") is None or _label_bound(encoding) == 0

    @pytest.mark.parametrize("viz_type", sorted(_CATEGORY_SOURCE), ids=sorted(_CATEGORY_SOURCE))
    def test_a_name_that_leaves_the_gutter_is_still_drawn(self, viz_type):
        """Dropping the axis without drawing the name would delete the identity channel."""
        payload = _renamed(viz_type, _wide_names)
        collection, key = _CATEGORY_SOURCE[viz_type]
        spec = compile_chart(viz_type, payload).spec
        rendered = json.dumps(spec)
        for entry in payload[collection]:
            assert entry[key] in rendered
        assert '"type": "text"' in rendered or '"header"' in rendered

    @pytest.mark.parametrize("viz_type", sorted(_CATEGORY_SOURCE), ids=sorted(_CATEGORY_SOURCE))
    def test_the_row_grows_when_a_name_takes_its_own_line(self, viz_type):
        """Vertical space is what pays for never truncating, so it has to be spent."""
        sizes = geometry()
        rows = len(EVERY_TYPE[viz_type][_CATEGORY_SOURCE[viz_type][0]])
        narrow = _plot_height(compile_chart(viz_type, _renamed(viz_type, _narrow_names)).spec, rows)
        wide = _plot_height(compile_chart(viz_type, _renamed(viz_type, _wide_names)).spec, rows)
        assert wide >= narrow
        assert wide == max(rows * sizes["row_step_label_above"], sizes["plot_min_height"])

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    @pytest.mark.parametrize("names", [_narrow_names, _wide_names], ids=["narrow", "too-wide"])
    def test_no_drawn_name_ever_exceeds_the_bound_it_is_drawn_under(self, viz_type, names):
        """The invariant the whole label rule reduces to, over every type and both placements.

        `attribution` runs here too even though it cannot be renamed — its own
        constants are subject to the same rule, and asserting it over the whole
        registry is what stops a later type from arriving unchecked.
        """
        payload = EVERY_TYPE[viz_type] if viz_type not in _CATEGORY_SOURCE else _renamed(viz_type, names)
        spec = compile_chart(viz_type, payload).spec
        # The NAME step, which is what places a category and what its axis states.
        # `tick` is the numeric step; the two share a size today, so measuring at it
        # would leave this invariant green while a raised `label` drew names wider
        # than the bound they were proved to fit — the truncation the label rule forbids.
        size = font_sizes()["label"]
        for encoding in _identity_encodings(spec):
            bound = _label_bound(encoding)
            if not bound:
                continue
            for name in encoding["sort"]:
                assert fits(name, size, bound), (
                    f"{viz_type} draws {name!r} under a {bound}px bound but it measures "
                    f"{text_width(name, size):.0f}px — Vega would truncate it"
                )


class TestAttributionNamesItsOwnScopes:
    """Its categories are compiler constants, which is why it is exempt above."""

    def test_the_scopes_are_not_taken_from_the_payload(self):
        spec = compile_chart("attribution", EVERY_TYPE["attribution"]).spec
        assert _identity_encodings(spec)[0]["sort"] == ["End-to-end", "Subsystem", "Unattributed"]

    def test_they_fit_the_gutter_so_the_scopes_never_move(self):
        """If a scope were ever renamed past the bound, the rule above would catch it."""
        bound, size = geometry()["gutter_left"], font_sizes()["label"]
        assert all(fits(scope, size, bound) for scope in ("End-to-end", "Subsystem", "Unattributed"))


class TestTheMeasurementMatchesWhatTheRendererDraws:
    """A width measured at one size and drawn at another decides nothing."""

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_names_are_measured_at_the_size_the_axis_states(self, viz_type):
        """The one coupling that would fail silently: it truncates only past the bound.

        The compiler decides a name's placement by measuring it, and the spec then
        tells the renderer what size to draw it at. Both read the type scale's NAME
        step rather than one copying the other, and this is what says so if they
        ever part — a name measured at one size and drawn at another decides
        nothing, and the symptom is a label proved to fit that draws truncated.

        Asserted against the axis the spec itself carries rather than against the
        config's blanket default, which is the numeric-tick step: the two share a
        size today, so a test written against the config would pass while the axis
        drew names at whatever `tick` happened to become.
        """
        for encoding in _identity_encodings(compile_chart(viz_type, EVERY_TYPE[viz_type]).spec):
            axis = encoding.get("axis")
            if isinstance(axis, dict) and "labelFontSize" in axis:
                assert axis["labelFontSize"] == font_sizes()["label"]

    def test_a_name_in_a_facet_header_draws_at_the_size_it_was_measured_at(self):
        """The marginal names the same categories the value panel does.

        Its header takes the facet step from the config rather than stating a size,
        because a small-multiple heading is one role across every figure — but the
        placement decision was measured at the name step, so the two have to hold
        the same number or a name proved to fit the gutter draws wider than it.
        """
        assert vega_config("dark")["header"]["labelFontSize"] == font_sizes()["label"]

    def test_the_config_still_answers_for_an_axis_that_states_nothing(self):
        """A generator-emitted axis with no size of its own is most likely numeric."""
        assert vega_config("dark")["axis"]["labelFontSize"] == font_sizes()["tick"]


class TestTheSharedPrefixIsStripped:
    """A prefix every series carries distinguishes nothing inside one chart."""

    def test_a_prefix_shared_by_every_series_goes(self):
        stripped = strip_common_prefix(["vendor/large-model", "vendor/small-model"])
        assert stripped == {"vendor/large-model": "large-model", "vendor/small-model": "small-model"}

    def test_a_prefix_only_some_share_stays(self):
        labels = ["anthropic/claude", "openai/gpt"]
        assert strip_common_prefix(labels) == {label: label for label in labels}

    def test_stripping_stops_at_the_first_segment_that_differs(self):
        stripped = strip_common_prefix(["a/b/c", "a/b/d", "a/e/f"])
        assert stripped == {"a/b/c": "b/c", "a/b/d": "b/d", "a/e/f": "e/f"}

    def test_series_with_no_delimiter_are_left_alone(self):
        labels = ["budget_exhausted", "confidence_met"]
        assert strip_common_prefix(labels) == {label: label for label in labels}

    def test_no_series_is_ever_stripped_to_nothing(self):
        """A label that IS the shared prefix would otherwise lose its whole name."""
        labels = ["anthropic/", "anthropic/claude"]
        assert strip_common_prefix(labels) == {label: label for label in labels}

    def test_at_least_one_segment_always_survives(self):
        assert strip_common_prefix(["a/b", "a/b"]) == {"a/b": "b"}

    def test_a_figure_with_no_categories_has_no_shared_prefix(self):
        """A frame can legitimately draw nothing — an empty set is not an undefined one.

        Reachable only through a producer this compiler does not have today, which is
        the reason to hold it: the arithmetic takes a `min` over the segments, so the
        empty case raised rather than returning nothing to strip.
        """
        assert strip_common_prefix([]) == {}

    def test_two_series_can_never_become_one_name(self):
        """Unreachable by construction, and held by construction rather than by argument.

        A suffix collision after a prefix every label shares means the labels were
        identical to begin with — but the property that stripping cannot merge two
        series is the whole reason it is preferred to an alias table, so it is
        checked rather than reasoned about.
        """
        for labels in (["a/x", "b/x"], ["a/b/c", "a/d/c"], ["o/p/q", "o/p/q"], ["a//b", "a/b"]):
            stripped = strip_common_prefix(labels)
            assert len(set(stripped.values())) == len(set(labels))

    def test_the_chart_draws_the_short_name_and_reports_the_full_one(self):
        """The picture can be short only because the reading of it is not."""
        names = [f"anthropic/claude-{variant}" for variant in ("opus", "sonnet", "haiku", "instant")]
        chart = compile_chart("breakdown", _renamed("breakdown", names))
        assert sorted(_identity_encodings(chart.spec)[0]["sort"]) == sorted(name.split("/", 1)[1] for name in names)
        assert {row["label"] for row in chart.rows} == set(names)

    def test_stripping_is_what_keeps_a_name_in_the_gutter(self):
        """The two rules compose: strip first, then judge the remainder against 176px."""
        names = [f"anthropic-public-models/claude-{variant}-4" for variant in ("opus", "sonnet", "haiku", "instant")]
        assert not any(fits(name, font_sizes()["label"], geometry()["gutter_left"]) for name in names)
        spec = compile_chart("breakdown", _renamed("breakdown", names)).spec
        assert _label_bound(_identity_encodings(spec)[0]) == geometry()["gutter_left"]


#: A measure name that cannot fit the figure on one line, checked rather than assumed.
_OVERLONG_MEASURE = (
    "median end-to-end pipeline latency by model and prompt variant across the whole campaign, "
    "every scenario and each of the judge configurations under test"
)


class TestTheTitleWrapsRatherThanOverrunning:
    """The one input the fixed geometry did not fix, held to the same label rule."""

    def test_the_overlong_measure_really_does_overrun(self):
        """Pins the premise, so the wrapping tests cannot pass by drawing something short."""
        assert not fits(_OVERLONG_MEASURE, font_sizes()["title"], geometry()["figure_width"])

    def test_a_title_that_fits_is_carried_verbatim(self):
        chart = compile_chart("breakdown", PAYLOAD)
        assert chart.spec["title"] == "share of stops"
        assert chart.title == "share of stops"

    def test_a_title_wider_than_the_figure_becomes_several_lines(self):
        chart = compile_chart("breakdown", dict(PAYLOAD, measure=_OVERLONG_MEASURE))
        assert isinstance(chart.spec["title"], list)
        assert len(chart.spec["title"]) > 1

    def test_every_wrapped_line_fits_the_figure(self):
        chart = compile_chart("breakdown", dict(PAYLOAD, measure=_OVERLONG_MEASURE))
        assert all(fits(line, font_sizes()["title"], geometry()["figure_width"]) for line in chart.spec["title"])

    def test_nothing_is_truncated_or_shrunk_to_make_it_fit(self):
        chart = compile_chart("breakdown", dict(PAYLOAD, measure=_OVERLONG_MEASURE))
        assert " ".join(chart.spec["title"]).split() == _OVERLONG_MEASURE.split()
        assert "fontSize" not in json.dumps(chart.spec["title"])

    def test_the_wire_title_stays_one_string(self):
        """`ChartIntent.title` is a string on the contract; only the spec's wraps."""
        chart = compile_chart("breakdown", dict(PAYLOAD, measure=_OVERLONG_MEASURE))
        assert chart.title == _OVERLONG_MEASURE
        assert chart.intent.title == _OVERLONG_MEASURE

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_every_type_bounds_its_title(self, viz_type):
        """The bound is a property of the figure, so no arm may opt out of it."""
        title = compile_chart(viz_type, EVERY_TYPE[viz_type]).spec["title"]
        assert all(fits(line, font_sizes()["title"], geometry()["figure_width"]) for line in lines_of(title))

    @pytest.mark.parametrize("viz_type", sorted(_CATEGORY_SOURCE), ids=sorted(_CATEGORY_SOURCE))
    def test_a_figure_that_gave_up_its_gutter_wraps_its_title_tighter(self, viz_type):
        """The bound follows the figure, and the figure is narrower when names sit above.

        Wrapping against the widest a figure can be would let a title overrun exactly
        the shapes that already gave up 176px to keep their names whole — and every
        assertion above would still pass, because they measure the same generous
        number the code would then be using.
        """
        sizes = geometry()
        narrow = sizes["plot_width"] + sizes["gutter_right"]
        assert narrow < sizes["figure_width"]

        payload = _renamed(viz_type, _wide_names)
        chart = compile_chart(viz_type, dict(payload, **_measure_field(viz_type, _OVERLONG_MEASURE)))
        title = chart.spec["title"]
        # Non-vacuity, both halves: the names really did leave the gutter, and the title
        # really did have to break. A short title fits any bound, so without this the
        # assertion below would hold against a compiler that never wrapped at all.
        assert all(
            encoding.get("axis", "absent") is None or _label_bound(encoding) == 0
            for encoding in _identity_encodings(chart.spec)
        )
        assert len(lines_of(title)) > 1
        assert all(fits(line, font_sizes()["title"], narrow) for line in lines_of(title))


def _value_encodings(node, found=None) -> list[dict]:
    """Every quantitative positional encoding in a compiled spec, at any depth.

    The mirror of `_identity_encodings`: that one finds where a chart puts its
    categories, this one finds where it puts its numbers. Both walk rather than
    index, because the rules below are properties of every axis a figure draws and
    a figure draws several — a value panel, a marginal's counts, the axis a value
    label is positioned against.
    """
    found = [] if found is None else found
    if isinstance(node, dict):
        if (
            node.get("type") == "quantitative"
            and isinstance(node.get("scale"), dict)
            and isinstance(node.get("axis"), dict)
        ):
            found.append(node)
        for value in node.values():
            _value_encodings(value, found)
    elif isinstance(node, list):
        for item in node:
            _value_encodings(item, found)
    return found


def _has_something_to_read_against(encoding):
    """Whether a mark on this axis can be located without a grid.

    Two ways, and a grid is drawn exactly when there is neither: a **shared
    baseline**, which every bar measures from, and the axis's **own rule** where it
    is range-framed — a line that ends where the data ends, with its ticks rising
    from it at the labelled values.
    """
    return encoding["scale"]["zero"] or encoding["axis"].get("domain") is True


class TestTheGridIsAPropertyOfTheEncoding:
    """A grid is drawn only where the reader has nothing else to read a mark against.

    Stated as an invariant rather than as a list of which chart gets which, because
    the two are not the same partition: a `distribution` is a position chart whose
    marginal counts are a magnitude, and a per-type table would have drawn the line
    in the wrong place for exactly that panel.
    """

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_a_grid_is_drawn_where_and_only_where_nothing_else_locates_a_mark(self, viz_type):
        spec = compile_chart(viz_type, EVERY_TYPE[viz_type]).spec
        encodings = _value_encodings(spec)
        assert encodings, f"{viz_type} draws no quantitative axis at all"
        for encoding in encodings:
            anchored = _has_something_to_read_against(encoding)
            assert encoding["axis"]["grid"] is not anchored, (
                f"{viz_type} draws an axis with grid={encoding['axis']['grid']} and "
                f"{'a baseline or its own rule' if anchored else 'neither a baseline nor its own rule'} — "
                "a bar's shared baseline does the comparing, a range-framed rule locates against itself, "
                "and an interval with neither has nothing to be read against"
            )

    def test_every_answer_actually_occurs_across_the_registry(self):
        """Otherwise the invariant above holds by every chart being the same kind."""
        reasons = {
            (encoding["scale"]["zero"], encoding["axis"].get("domain") is True)
            for viz_type, payload in EVERY_TYPE.items()
            for encoding in _value_encodings(compile_chart(viz_type, payload).spec)
        }
        assert (True, False) in reasons, "no chart is anchored by a baseline"
        assert (False, True) in reasons, "no chart is anchored by its own range-framed rule"
        assert (False, False) in reasons, "no chart needs a grid, so the rule is never exercised"

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_no_gridline_is_dashed(self, viz_type):
        """A dash is a mark property the reader has learned to read as data."""
        assert "gridDash" not in json.dumps(compile_chart(viz_type, EVERY_TYPE[viz_type]).spec)

    def test_bars_measured_from_the_plot_edge_draw_their_own_domain_line(self):
        """Two lines in one place, one of them furniture, is one line too many."""
        identity = _identity_encodings(compile_chart("breakdown", PAYLOAD).spec)[0]
        assert identity["axis"]["domain"] is False

    def test_a_signed_chart_keeps_its_domain_line_because_the_bars_are_not_on_it(self):
        """Zero has moved into the plot, so the axis line is no longer redundant."""
        identity = _identity_encodings(compile_chart("delta_table", DELTA_TABLE).spec)[0]
        assert identity["axis"]["domain"] is True


class TestALegendThatSurvivesIsPlacedRatherThanDefaulted:
    """A legend is a fallback, and where it lands is not Vega's to pick.

    The parity test between the two renderer configs compares KEYS, so both sides
    could carry `orient` and disagree on its value while staying green. These pin
    the value, in both themes.
    """

    @pytest.mark.parametrize("theme", ["light", "dark"])
    def test_a_surviving_legend_reads_across_rather_than_down_the_right_edge(self, theme):
        """A right-side vertical key takes its width out of the plot, which is fixed."""
        legend = vega_config(theme)["legend"]
        assert legend["direction"] == "horizontal"

    @pytest.mark.parametrize("theme", ["light", "dark"])
    def test_the_key_sits_above_the_plot_rather_than_on_top_of_the_marks(self, theme):
        """`top`, not `top-left`, and the difference is measured rather than read.

        Vega's CORNER orients place a legend inside the data rectangle — `top-left`
        draws the key over whatever marks are in that corner. `top` places it above
        the plot, left-aligned to the plot's left edge, which is what the legend rule's
        "top-left under the title" describes. The rule's wording is what makes
        this worth pinning: it reads like an instruction to write `top-left`.
        """
        assert vega_config(theme)["legend"]["orient"] == "top"


class TestTheZeroLineIsDrawnWhereZeroIsAPlace:
    def test_a_chart_spanning_both_signs_draws_a_rule_at_zero(self):
        """`delta_table` always does: its domain is symmetric about the baseline."""
        spec = compile_chart("delta_table", DELTA_TABLE).spec
        rules = [layer for layer in _mark_layers(spec, "rule") if layer["data"]["values"] == [{"at": 0}]]
        assert len(rules) == 1
        assert rules[0]["mark"]["style"] == ZERO_RULE_STYLE, "the ink comes from a named style, never a colour literal"

    def test_an_attribution_whose_movements_disagree_in_sign_gets_one_too(self):
        """One movement up and one down puts zero inside the plot rather than at its edge."""
        mixed = copy.deepcopy(ATTRIBUTION_EARNED)
        mixed["end_to_end"] = dict(mixed["end_to_end"], delta=abs(mixed["end_to_end"]["delta"]), a=None, b=None)
        mixed["subsystem"] = dict(mixed["subsystem"], delta=-abs(mixed["subsystem"]["delta"] or 1.0), a=None, b=None)
        mixed.pop("unattributed_delta", None)
        mixed["unattributed_withheld"] = "The movements run in opposite directions, so no remainder is placeable."
        spec = compile_chart("attribution", mixed).spec
        assert [layer for layer in _mark_layers(spec, "rule") if layer["data"]["values"] == [{"at": 0}]]

    @pytest.mark.parametrize("theme", ["light", "dark"])
    def test_the_style_the_spec_names_is_the_one_the_config_defines(self, theme):
        """The two halves of a colour that never travels in the spec.

        The config writes each name as a literal so the browser mirror's parity test
        can read it out of the source, which leaves the constants and the literals
        free to part — and a spec asking for a style nothing defines draws in
        Vega's own default, which is a black rule on a near-black surface.
        """
        assert set(vega_config(theme)["style"]) == {ZERO_RULE_STYLE, CONTEXT_STYLE, VALUE_ON_FILL_STYLE}

    def test_a_chart_whose_zero_is_its_own_edge_draws_no_rule(self):
        """The bars already start there; a rule would trace their bases."""
        assert not _mark_layers(compile_chart("breakdown", PAYLOAD).spec, "rule")

    def test_a_cropped_position_axis_draws_no_zero_rule(self):
        """Zero is not on the chart, so a line at the edge is not zero."""
        spec = compile_chart("null_result", NULL_RESULT).spec
        assert all(layer["data"]["values"] != [{"at": 0}] for layer in _mark_layers(spec, "rule"))


class TestMagnitudeAxesNeverCropAndPositionAxesSaySoWhenTheyDo:
    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_a_length_encoding_always_holds_its_baseline(self, viz_type):
        """A truncated bar misstates a ratio rather than merely rescaling one."""
        for encoding in _value_encodings(compile_chart(viz_type, EVERY_TYPE[viz_type]).spec):
            if not encoding["scale"]["zero"]:
                continue
            low, high = encoding["scale"]["domain"]
            assert low <= 0 <= high, (
                f"{viz_type} draws a magnitude axis over {[low, high]}, which excludes its own baseline"
            )

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_no_figure_footnotes_its_crop(self, viz_type):
        """**No type discloses a crop**, a rule read off the screen.

        An earlier version had scoped the sentence to SPAN marks, on the
        reasoning that a span's width invites a proportional comparison a cropped axis
        silently rescales. Seen on a real figure, a labelled 40,000-55,000 axis makes
        that comparison from its ticks too, and the sentence under every interval plot
        was noise — which is worse than redundant, because it teaches a reader to skip
        the subtitle, and the subtitle is where the disclosures that DO carry
        information live.

        What still protects a reader is the test above: a LENGTH mark may not crop at
        all. That is the case where the picture genuinely lies — a bar's length is
        read as a ratio — and it is refused outright rather than apologised for.
        """
        assert "cropped" not in _subtitle(compile_chart(viz_type, EVERY_TYPE[viz_type]).spec)

    def test_the_figures_this_rule_silenced_are_still_genuinely_cropped(self):
        """Non-vacuity: the silence has to be a decision, not an absent condition.

        If every axis quietly grew a zero, the assertion above would pass while
        proving nothing about disclosure at all.
        """
        for viz_type in ("distribution", "null_result", "frontier"):
            spec = compile_chart(viz_type, EVERY_TYPE[viz_type]).spec
            assert any(
                not encoding["scale"]["zero"]
                and not encoding["scale"]["domain"][0] <= 0 <= encoding["scale"]["domain"][1]
                for encoding in _value_encodings(spec)
            ), f"{viz_type} no longer crops, so its silence proves nothing"

    @pytest.mark.parametrize("viz_type", ["distribution", "null_result"])
    def test_a_long_footnote_wraps_inside_the_figure_rather_than_widening_it(self, viz_type):
        """Vega grows the frame for a subtitle exactly as it does for a title.

        A disclosure that pushed the figure out of its column would be one honesty
        rule paid for out of another's budget.
        """
        payload = _renamed(viz_type, _wide_names)
        lines = _subtitle_lines(compile_chart(viz_type, payload).spec)
        bound = geometry()["plot_width"] + geometry()["gutter_right"]
        assert all(fits(line, font_sizes()["footnote"], bound) for line in lines)


class TestValuesAreWrittenOnTheMarks:
    """The grid can be quiet because the number is already on the bar."""

    @pytest.mark.parametrize("viz_type", ROW_BASED, ids=ROW_BASED)
    def test_every_row_based_drawn_value_appears_beside_its_own_mark(self, viz_type):
        """The subject is the row-based figures because they draw ONE quantity per mark.

        A point plot draws two, and they are the mark's position rather than its
        length — writing both beside every point puts four numbers in a plot that
        holds six marks, on top of the names that identify them. The reader gets
        them from the axes and from the values table, which is where a scatter's
        numbers have always lived.
        """
        chart = compile_chart(viz_type, EVERY_TYPE[viz_type])
        written = {
            row[VALUE_TEXT_FIELD]
            for layer in _mark_layers(chart.spec, "text")
            for row in _layer_rows(chart.spec, layer)
            if VALUE_TEXT_FIELD in row
        }
        assert written, f"{viz_type} writes no value on any mark"

    def test_a_value_is_written_at_the_row_it_belongs_to(self):
        chart = compile_chart("breakdown", PAYLOAD)
        placed = {
            row[DISPLAY_FIELD]: row[VALUE_TEXT_FIELD]
            for layer in _mark_layers(chart.spec, "text")
            for row in _layer_rows(chart.spec, layer)
        }
        assert placed == {row["label"]: format_number(row["value"]) for row in chart.rows}

    def test_an_interval_writes_its_estimate_at_the_mean_it_names(self):
        """Values-on-the-mark, for a mark that is an interval rather than a bar.

        The label was anchored at the interval's upper BOUND while reading its centre,
        so every number sat at an x-position it did not name: a group with mean 12.5
        over a 10.0-15.0 interval printed "12.5" at x≈15, and a reader maps each
        number to the wrong value.
        """
        chart = compile_chart("distribution", DISTRIBUTION)
        scale = self._restatement_scale()
        by_text = {
            row[VALUE_TEXT_FIELD]: row[ANCHOR_FIELD]
            for layer in _mark_layers(chart.spec, "text")
            for row in _layer_rows(chart.spec, layer)
        }
        for group in DISTRIBUTION["groups"]:
            mean = group["ci"]["mean"] * scale
            assert by_text[format_number(mean)] == pytest.approx(mean), (
                "the estimate label is anchored at the mean it names, not at the interval's bound"
            )

    def test_an_estimate_label_never_leaves_the_plot(self):
        """A mean near the domain's edge overran the frame, which makes Vega grow it.

        On the same figure, a mean of 14.9 on an axis ending at 15.0 printed past the
        right edge, overlapping its own interval cap. The label is
        centred where the plot has room on both sides and pushed to one side where it
        does not, so its box stays inside whatever the domain does.
        """
        # Two groups whose means sit hard against both ends of a cropped position axis.
        edges = {
            "groups": [
                {
                    "label": "low",
                    "ci": {"low": 12.4, "high": 17.96, "mean": 12.42, "level": 0.95, "variability": "across 5 runs"},
                    "n": 5,
                },
                {
                    "label": "high",
                    "ci": {"low": 16.31, "high": 23.46, "mean": 23.44, "level": 0.95, "variability": "across 5 runs"},
                    "n": 5,
                },
            ],
            "unit": "%",
            "x_label": "share",
        }
        chart = compile_chart("distribution", edges)
        axis = self._distribution_axis(chart)
        size = font_sizes()["value"]
        placements = set()
        for layer in _mark_layers(chart.spec, "text"):
            align = layer["mark"]["align"]
            placements.add(align)
            for row in _layer_rows(chart.spec, layer):
                width = text_width(row[VALUE_TEXT_FIELD], size)
                # `dx` included, because it is part of where the glyphs land — a check
                # that ignored it would pass a label sitting six pixels outside.
                at = axis.offset(row[ANCHOR_FIELD]) + layer["mark"].get("dx", 0)
                extends = {"center": (width / 2, width / 2), "left": (0.0, width), "right": (width, 0.0)}[align]
                assert at - extends[0] >= 0, f"{row[VALUE_TEXT_FIELD]} overruns the plot's left edge"
                assert at + extends[1] <= axis.plot_span, f"{row[VALUE_TEXT_FIELD]} overruns the plot's right edge"
        assert placements and placements != {"center"}, (
            "this fixture must force the edge cases, or it asserts nothing the centred default would not satisfy"
        )

    def _restatement_scale(self):
        """The factor the compiler restated this chart's values by.

        Asked of the compiler rather than reverse-engineered from the spec: the
        duration ladder is what decides it, and a test restating that arithmetic would
        pass whenever the two copies agreed on being wrong.
        """
        values = [value for group in DISTRIBUTION["groups"] for value in (group["ci"]["low"], group["ci"]["high"])]
        scale, _ = display_scale(values, DISTRIBUTION["unit"])
        return scale

    def _distribution_axis(self, chart):
        """The value axis a compiled distribution drew against, rebuilt from its domain."""
        encoding = chart.spec["spec"]["layer"][0]["encoding"]["x"]
        low, high = encoding["scale"]["domain"]
        return ValueAxis.position("", [low, high], chart.spec["spec"]["width"], framed=True)

    def test_a_faceted_figure_writes_its_values_into_the_row_they_belong_to(self):
        """The same contract where the categories are facet rows rather than an axis:
        the label carries its cohort, so the cell it lands in is the one it names."""
        chart = compile_chart("distribution", DISTRIBUTION)
        placed = {
            row[DISPLAY_FIELD]: row[VALUE_TEXT_FIELD]
            for layer in _mark_layers(chart.spec, "text")
            for row in _layer_rows(chart.spec, layer)
        }
        assert placed == {row["label"]: format_number(row["mean"]) for row in chart.rows}

    def _probe(self, room):
        """A two-bar breakdown whose second bar leaves exactly `room` px clear."""
        top = 100.0
        return {
            "measure": "m",
            "unit": "runs",
            "parts": [
                {"label": "top", "value": top},
                {"label": "probe", "value": top * (1 - room / geometry()["plot_width"])},
            ],
        }

    def _alignment(self, payload, display):
        """How the label for `display` is aligned in the compiled chart."""
        for layer in _mark_layers(compile_chart("breakdown", payload).spec, "text"):
            if any(row[DISPLAY_FIELD] == display for row in layer["data"]["values"]):
                return layer["mark"]["align"]
        raise AssertionError(f"no value label for {display}")

    def test_a_mark_with_the_clearance_writes_its_value_outside(self):
        clearance = geometry()["value_label_clearance"]
        assert self._alignment(self._probe(clearance + 1), "probe") == "left"

    def test_a_mark_one_pixel_short_of_it_writes_its_value_on_the_fill(self):
        clearance = geometry()["value_label_clearance"]
        assert self._alignment(self._probe(clearance - 1), "probe") == "right"

    def test_the_longest_bar_has_no_room_at_all_and_takes_the_fill(self):
        assert self._alignment(self._probe(geometry()["value_label_clearance"] + 1), "top") == "right"

    def test_a_label_wider_than_the_clearance_needs_room_for_itself_too(self):
        """The rule states a minimum gap, not that any number fits in one.

        Drawn outside regardless, a wide label runs past the plot edge and Vega
        grows the frame to fit it — which is the figure leaving its column for a
        label, the one outcome the whole measurement substrate exists to prevent.
        """
        clearance = geometry()["value_label_clearance"]
        size = font_sizes()["value"]
        wide = "-1.2345e-06"
        assert text_width(wide, size) > clearance, "this test needs a label the clearance cannot hold"
        axis = ValueAxis.magnitude("m", [100.0], geometry()["plot_width"])
        end = 100.0 * (1 - (clearance + 1) / geometry()["plot_width"])
        [layer] = value_label_layers([MarkValue(display="d", end=end, text=wide)], axis, {})
        assert layer["mark"]["align"] == "right", "a label the gap cannot hold is written on the mark"

    def test_values_are_suppressed_past_the_stated_mark_count(self):
        """Past it, highlight-plus-context takes over and only named marks are labelled."""
        limit = geometry()["value_label_max_marks"]
        parts = [{"label": f"p{index}", "value": float(index + 1)} for index in range(limit)]
        at_limit = {"measure": "m", "unit": "runs", "parts": parts}
        over = {"measure": "m", "unit": "runs", "parts": [*parts, {"label": "extra", "value": 0.5}]}
        assert _mark_layers(compile_chart("breakdown", at_limit).spec, "text")
        assert not _mark_layers(compile_chart("breakdown", over).spec, "text")

    def test_a_mark_that_paints_no_fill_never_asks_for_the_knockout(self):
        """The knockout IS the chart surface, so on an unfilled mark it is invisible.

        A point or a bare interval rule has nothing under an inward label but the
        plot background — and `chart.fg-on-fill` resolves to exactly that background
        (dark `#140a29` is the dark surface; light `#ffffff` is the light one). So a
        label that took the knockout there would be painted on the background in the
        background's own colour: 1:1, gone.

        Reachable, not theoretical: on a cropped position axis the outermost mark sits
        about 57px from the edge, `needed` is at least the 56px clearance, and
        `format_number` writes an exponential like `1.235e-05` at about 64px — so one
        value under 1e-4 pushes the top row's label inward.
        """
        axis = ValueAxis.position("m", [0.0, 100.0], geometry()["plot_width"])
        wide = "1.235e-05"
        assert text_width(wide, font_sizes()["value"]) > geometry()["value_label_clearance"], (
            "this test needs a label wide enough to be pushed inward"
        )
        at_the_edge = MarkValue(display="d", end=axis.high, text=wide, filled=False)
        [layer] = value_label_layers([at_the_edge], axis, {})
        assert "style" not in layer["mark"], "a mark with no fill under its label must not take the knockout"

    def test_the_default_is_filled_so_a_length_arm_need_not_restate_it(self):
        """The other direction of `MarkValue.filled`, which the default makes easy to lose.

        `filled` defaults True so the length arms — breakdown and attribution — do not
        each restate the obvious. That default is load-bearing in a way a
        false-only test cannot see: if it ever flipped, every bar's inside label would
        quietly take chart ink again at 2.53:1 and no test naming `filled=False` would
        notice.
        """
        axis = ValueAxis.magnitude("m", [100.0], geometry()["plot_width"])
        [layer] = value_label_layers([MarkValue(display="d", end=100.0, text="100")], axis, {})
        assert layer["mark"]["style"] == VALUE_ON_FILL_STYLE, "a bar's inside label is on a fill by default"

    def test_the_arms_that_draw_no_fill_say_so(self):
        """Held over the compiled specs, so the flag cannot be dropped from an arm.

        `null_result` draws a rule and a point; `sweep_ranking`'s ranking panel draws
        points; `delta_table` draws a 3px connector and a point. None has a fill under
        an inward label, and the flag is what the placement decision reads — a default
        that quietly reverted would put the knockout back on the chart surface.
        """
        for viz_type, payload in (
            ("null_result", NULL_RESULT),
            ("sweep_ranking", SWEEP_RANKING),
            ("delta_table", DELTA_TABLE),
        ):
            spec = compile_chart(viz_type, payload).spec
            for layer in _mark_layers(spec, "text"):
                assert "style" not in layer["mark"], f"{viz_type} asks for a fill style on a mark that paints no fill"

    def _style_of(self, payload, display):
        """The style the label for `display` asks for, or None where it asks for none."""
        for layer in _mark_layers(compile_chart("breakdown", payload).spec, "text"):
            if any(row[DISPLAY_FIELD] == display for row in layer["data"]["values"]):
                return layer["mark"].get("style")
        raise AssertionError(f"no value label for {display}")

    def test_a_value_on_the_fill_asks_for_the_knockout_ink(self):
        """Chart ink over the mark's own fill is 2.53:1 in dark and 3.36:1 in light.

        The palette admits a third chart ink for exactly this case, so the label asks for it by NAME — the spec still carries no colour, and
        what a knockout resolves to stays the renderer's to decide.
        """
        clearance = geometry()["value_label_clearance"]
        assert self._style_of(self._probe(clearance - 1), "probe") == VALUE_ON_FILL_STYLE

    def test_a_value_outside_the_mark_does_not(self):
        """It is drawn on the chart surface, where the ordinary ink already clears."""
        clearance = geometry()["value_label_clearance"]
        assert self._style_of(self._probe(clearance + 1), "probe") is None

    def test_alignment_does_not_decide_the_ink_placement_does(self):
        """The trap the placement type exists to close.

        A bar growing LEFT from zero and labelled outside its end is right-aligned,
        and so is a right-growing bar labelled on its fill. Reading "inside" off the
        alignment gives the knockout to the first of those — a knockout drawn on the
        chart surface, which is the surface knocked out of, so it is invisible.
        """
        axis = ValueAxis.magnitude("m", [-100.0, 100.0], geometry()["plot_width"])
        # A left-growing bar with the whole left half of the plot beyond its end, and a
        # right-growing bar that reaches the plot's edge. Opposite placements.
        outside = MarkValue(display="outside", end=-50.0, text="-50")
        inside = MarkValue(display="inside", end=100.0, text="100")
        layers = value_label_layers([outside, inside], axis, {})
        by_display = {row[DISPLAY_FIELD]: layer["mark"] for layer in layers for row in layer["data"]["values"]}
        assert by_display["outside"]["align"] == by_display["inside"]["align"] == "right", (
            "both are right-aligned, which is exactly why alignment cannot carry the placement"
        )
        assert "style" not in by_display["outside"], "drawn on the chart surface, so the ordinary ink"
        assert by_display["inside"]["style"] == VALUE_ON_FILL_STYLE, "drawn on the fill, so the knockout"


class TestAValueLabelClearsItsMarksEdgeRatherThanItsCentre:
    """The daylight a reader sees is the gap the constant names, whatever the mark is.

    `VALUE_LABEL_OFFSET` positions the text from the value's POSITION, which is the
    mark's edge only where the mark ends at its value. Where it has a body — a point,
    centred on the number it names — the body ate the gap: measured in the rendered
    SVG, `delta_table`'s 4.47px radius left 1.53px of daylight and `sweep_ranking`'s
    5.48px left 0.52px, and both read as a label welded to its own mark.
    """

    def _labels(self, viz_type, payload):
        """Every value-label mark in a compiled chart, by the row it names."""
        spec = compile_chart(viz_type, payload).spec
        return {
            row[DISPLAY_FIELD]: layer["mark"]
            for layer in _mark_layers(spec, "text")
            for row in _layer_rows(spec, layer)
        }

    @pytest.mark.parametrize("radius", [0.0, 4.472135954999579, 5.477225575051661])
    def test_the_offset_is_the_marks_own_body_plus_the_stated_gap(self, radius):
        """Derived, not a constant: the gap is what is left over once the body is paid for."""
        axis = ValueAxis.magnitude("m", [100.0], geometry()["plot_width"])
        mark = MarkValue(display="d", end=10.0, text="10", radius=radius)
        [layer] = value_label_layers([mark], axis, {})
        assert abs(layer["mark"]["dx"]) == pytest.approx(radius + VALUE_LABEL_OFFSET)
        # The same statement from the reader's side, which is the one that matters: the
        # text starts `VALUE_LABEL_OFFSET` px past where the mark stops being drawn.
        assert abs(layer["mark"]["dx"]) - radius == pytest.approx(VALUE_LABEL_OFFSET)

    def test_a_bar_arm_is_not_moved_because_a_bar_has_no_body_to_clear(self):
        """The regression risk, pinned. A bar's tip IS its value, so it already has the
        whole gap — giving it a point's clearance would push every number in every bar
        chart off the end it names, for daylight the reader already had."""
        for viz_type, payload in (("breakdown", PAYLOAD), ("attribution", EVERY_TYPE["attribution"])):
            for display, mark in self._labels(viz_type, payload).items():
                assert mark["dx"] in (VALUE_LABEL_OFFSET, -VALUE_LABEL_OFFSET), (
                    f"{viz_type}/{display} took a clearance a bar does not need"
                )

    @pytest.mark.parametrize(
        ("viz_type", "payload"),
        [
            pytest.param("delta_table", DELTA_TABLE, id="delta_table"),
            pytest.param("sweep_ranking", SWEEP_RANKING, id="sweep_ranking"),
        ],
    )
    def test_an_arm_whose_mark_is_a_point_sets_its_labels_clear_of_the_disc(self, viz_type, payload):
        """Held over the compiled spec, so an arm cannot drop the radius and stay green."""
        radius = _drawn_point_radius(compile_chart(viz_type, payload).spec)
        labels = self._labels(viz_type, payload)
        assert labels, f"{viz_type} must still write its values"
        for display, mark in labels.items():
            assert abs(mark["dx"]) == pytest.approx(radius + VALUE_LABEL_OFFSET), (
                f"{viz_type}/{display} is offset from the point's centre, not from its edge"
            )

    def test_the_room_a_label_needs_counts_the_body_it_has_to_clear(self):
        """The other half: a body is room the text never had, so it decides placement too.

        Otherwise a label is written outside a mark on the strength of room its own point
        is standing in, and the text lands on the disc rather than beside it.
        """
        size = font_sizes()["value"]
        wide = "-1.2345e-06"
        width = text_width(wide, size)
        assert width > geometry()["value_label_clearance"], "the clearance floor must not be what decides this"
        radius = point_radius(80)
        span = geometry()["plot_width"]
        axis = ValueAxis.magnitude("m", [100.0], span)
        # Room for the text and the stated gap, and half a body short of the rest.
        end = 100.0 * (1 - (width + VALUE_LABEL_OFFSET + radius / 2) / span)
        [tip] = value_label_layers([MarkValue(display="d", end=end, text=wide)], axis, {})
        [disc] = value_label_layers([MarkValue(display="d", end=end, text=wide, radius=radius)], axis, {})
        assert tip["mark"]["align"] == "left", "a mark that ends at its value has the room"
        assert disc["mark"]["align"] == "right", "the same room, minus a disc, does not hold the label"

    def test_a_point_that_paints_no_fill_still_declines_the_knockout_once_it_has_a_radius(self):
        """The interaction the two flags have to survive together.

        `delta_table`'s label is always the one pushed inward — the room past the row that
        decides the axis is only the point's radius — and what it lands on is a 3px
        connector, which is chart surface. So it takes the ordinary ink AND the radius:
        the clearance moves it, the knockout must still not follow.
        """
        payload = {"rows": [{"metric": "llm_ms", "a": 16162.0, "b": 242.0}]}
        radius = _drawn_point_radius(compile_chart("delta_table", payload).spec)
        marks = self._labels("delta_table", payload)
        assert marks, "this shape must still write its value"
        for display, mark in marks.items():
            assert "style" not in mark, f"{display} asks for a knockout on a mark that paints no fill"
            assert abs(mark["dx"]) == pytest.approx(radius + VALUE_LABEL_OFFSET), f"{display} lost its clearance"

    def test_a_centred_label_takes_no_clearance_however_large_its_mark(self):
        """A centred label sits ABOVE its mark, so there is no edge for it to clear —
        and offsetting it would shift the number off the value it names, which is the
        defect the centred placement was introduced to fix."""
        assert Placement(align="center", inside=False, clearance=99.0).dx == 0

    def test_a_contestants_name_clears_its_mark_too(self):
        """`frontier` labels a point with a name rather than a value, and its mark is the
        largest in the package — ~20px across, because the shape channel has to be
        readable. Offset from the centre, the name started 0.9px INSIDE the disc."""
        spec = compile_chart("frontier", FRONTIER).spec
        [label] = [
            layer["mark"] for layer in _mark_layers(spec, "text") if layer["encoding"]["text"]["field"] == DISPLAY_FIELD
        ]
        assert label["dx"] == pytest.approx(_drawn_point_radius(spec) + FRONTIER_NAME_GAP)


# `CASES` below loads `fixtures/number-format-cases.json`, so editing that fixture
# reddens these tests. Marked by hand because the
# coverage canary cannot see this shape twice over — the repo root is built inline
# rather than bound to a name, and the path is one compound literal rather than a
# join per segment.
class TestBothSurfacesRenderOneRule:
    """The values-as-drawn table is compiled once and rendered twice, so it must read alike.

    `models.py` says the two descriptions of one chart cannot disagree, because they come
    from one compilation — and they did: this rule printed `1.65` while the browser's
    accessible table, a bare `String(value)`, printed `1.6500000000000001` for the same
    cell. The cases are shared with the browser's own cell-text test rather than
    restated, since two case tables for one rule is the same defect one level
    up.
    """

    CASES = json.loads((Path(__file__).resolve().parent / "fixtures" / "number-format-cases.json").read_text())

    def test_every_shared_case_renders_as_the_table_says(self):
        for case in self.CASES["cases"]:
            assert format_number(case["value"]) == case["expected"], case["why"]

    def test_the_shared_table_is_not_empty(self):
        """A fixture that failed to load would pass the loop above by iterating nothing."""
        assert len(self.CASES["cases"]) >= 10

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_value_is_an_em_dash_on_both_surfaces(self, value):
        """JSON cannot carry these, so the shared table names them and each reader asserts them."""
        assert format_number(value) == self.CASES["unrepresentable_in_json"]["expected"]

    def test_an_absent_value_is_the_one_cell_the_two_surfaces_word_differently(self):
        """Deliberate, and recorded — not a parity gap that slipped through.

        This side writes an em dash; the browser's accessible table writes "not
        recorded", because a cell that LOOKS blank in a screen-reader table is
        indistinguishable from a value of nothing. Asserted rather than left implicit
        so that "make them agree" is a decision someone takes against
        the round-to-true-precision rule, which states the difference, rather than a
        tidy-up that quietly costs the a11y wording.
        """
        assert format_number(None) == "—"
        assert "not recorded" in self.CASES["rule"], "the shared table must carry the difference it excludes"


class TestNoTextIsRotated:
    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_no_axis_turns_its_words(self, viz_type):
        """Every angle a spec states is flat — including the ones it must state.

        This asserted that NO title angle appears anywhere, which was a proxy that
        held only while no figure named a `y` quantity: Vega turns a y title a
        quarter turn by default, so the way to have one at all is to state
        `titleAngle: 0`. The proxy and the rule agreed until a chart needed its
        vertical axis labelled, and then the proxy would have forced the axis to
        stay nameless — a reader taking the quantity from a heading two lines up.
        The rule itself is unchanged and is what is asserted now: no text is
        turned.
        """
        spec = compile_chart(viz_type, EVERY_TYPE[viz_type]).spec
        assert all(angle == 0 for angle in _sizes(spec, "labelAngle"))
        assert all(angle == 0 for angle in _sizes(spec, "titleAngle"))

    def test_a_counted_quantity_is_named_above_the_panel_not_beside_it(self):
        """Vega draws a y-axis title rotated a quarter turn, so it goes in a heading."""
        payload = {
            "groups": [{"label": "model-b", "buckets": [{"range": "0-1s", "count": 2}], "n": 2}],
            "unit": "s",
            "x_label": "pipeline_synthesis",
        }
        panel = compile_chart("distribution", payload).spec["vconcat"][0]
        assert panel["title"] == "observations"
        assert panel["spec"]["encoding"]["y"]["axis"]["title"] is None


class TestACompiledSpecStatesNoAppearanceValue:
    """The rule that survived a type needing the one it used to be stated as.

    This class asserted "no `color` encoding, anywhere" and that was one sentence
    doing two jobs, and the chart rules split them:

    * **No appearance VALUE** — no hue, no opacity, no lightness. A spec asks by
      NAME and the renderer's config answers, which is the whole reason one stored
      artifact can draw correctly on a near-black surface and on a pale one. This
      is the half that stayed absolute, and it now covers MORE than it did: the
      opacity checks below are new, and they close the gap the old wording left
      open by naming only colour.
    * **Identity never on hue** — the categorical palette recycles past its slots,
      so a chart identifying its categories by colour has a point beyond which two
      of them look alike. An axis has no such point. This is the half that gained
      a permission. Its limits are held where they bind rather than here: the
      payload's own palette ceiling in `test_viz_payloads.py`, and the gate over
      ANY producer's spec in
      `test_vega_spec_policy.py::TestAQuantitativeColourScaleIsHeldToTheSameRules`.

    **The exposure argument this class used to carry is now the other way round,
    and that is deliberate rather than a loss.** It read: `_check_direct_labels`
    refuses a categorical colour encoding past the validated hues with no per-mark
    label, and nothing in this repo trips it *because no compiled arm emits colour
    at all*. That argument is retired — one arm emits colour now, and the gate is
    therefore exercised by a producer rather than merely believed in. What replaces
    it is stronger: the gate is asserted directly against that arm, in both
    directions, in the policy suite.
    """

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_no_encoding_anywhere_places_a_category_on_a_hue(self, viz_type):
        """Identity rides on the axis. A hue may say what LEVEL a row is at; it
        may never say which row this is, because the palette recycles and an axis
        does not.

        Derived from the spec rather than from a list of permitted fields: whatever
        a figure places on its identity axis is its identity, so asking whether that
        same field is also coloured is the question, and it needs no maintenance
        when a type arrives with an identity field nobody here anticipated.
        """
        spec = compile_chart(viz_type, EVERY_TYPE[viz_type]).spec
        identities = {encoding.get("field") for encoding in _identity_encodings(spec)} | {DISPLAY_FIELD}
        for layer in _mark_layers(spec):
            colour = (layer.get("encoding") or {}).get("color")
            if not isinstance(colour, dict):
                continue
            assert colour.get("field") not in identities, (
                f"{viz_type} colours {colour.get('field')!r}, which is also what its identity axis places — "
                "a category identified by hue is one the palette can recycle into another"
            )

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_no_colour_literal_reaches_the_spec_at_any_depth(self, viz_type):
        """A literal here would defeat both renderers, and it is where the
        OKLCH-renders-black defect is cheapest to catch. Every colour asks for its
        ink by NAME for exactly this reason — a style name or a range name travels,
        a hex does not."""
        rendered = json.dumps(compile_chart(viz_type, EVERY_TYPE[viz_type]).spec).lower()
        # A colour NOTATION, not a bare `#`. The single-chart version of this test
        # asserts the character, which is right there and wrong here: generalised
        # over every payload, a category or a measure name is free to contain one,
        # and a gate that fails on a finding's own prose is a gate that gets
        # deleted rather than obeyed.
        #
        # The bracket is attached for that same reason, and it is the reason the
        # production gate states: `lab (control)` is a measure name, not a colour.
        assert not re.search(r"\b(?:oklch|oklab|lch|lab|rgba?|hsla?|color-mix)\(", rendered)
        assert not re.search(r"#[0-9a-f]{3,8}\b", rendered)

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_a_colour_scale_names_its_range_and_never_lists_one(self, viz_type):
        """The half that makes a `color` encoding admissible at all.

        A `range` given as a list is a colour literal wearing a scale's clothes —
        it would draw the dark theme's ink into a light render, which is exactly
        what `_UNRENDERABLE_COLOUR` and the hex check above exist to prevent one
        level down.
        """
        for layer in _mark_layers(compile_chart(viz_type, EVERY_TYPE[viz_type]).spec):
            colour = (layer.get("encoding") or {}).get("color")
            if not isinstance(colour, dict):
                continue
            spread = (colour.get("scale") or {}).get("range")
            assert spread is None or isinstance(spread, str), (
                f"{viz_type} states its own colour range {spread!r} — name a config range instead"
            )

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_a_receding_mark_asks_by_name_rather_than_stating_an_opacity(self, viz_type):
        """Recession is a per-THEME decision, so the compiler may not take it.

        The same alpha is not the same recession on obsidian and on pearl, so a
        compiled `0.35` is right on at most one of the two surfaces this one
        artifact is drawn on. `chart-context` is how a mark says "I carry no
        identity" and lets the renderer decide what that looks like.

        **Two carve-outs, and each is a real distinction rather than a hole.**

        A DATA-DRIVEN opacity encoding stays legal, because there the alpha is the
        encoding — a rug of raw samples and an interval whose coverage is
        unrecorded both use it to say marks ACCUMULATE or a bound is unknown, which
        is a fact about the data and identical in both themes.

        And `1` stays legal, because it states no appearance: it opts OUT of a
        renderer default. Vega-Lite draws a point at 0.7 so a cloud reads as
        density, which for a figure of a handful of answers means every mark draws
        in a blend of the palette hue and whatever sits behind it — a different
        colour per theme, arrived at by not deciding. Full ink is the same
        instruction on both surfaces.

        What is refused is any other constant: a number chosen to make one mark
        quieter than another, which is a judgement about a surface the compiler
        cannot see.
        """
        for layer in _mark_layers(compile_chart(viz_type, EVERY_TYPE[viz_type]).spec):
            mark = layer["mark"]
            if not isinstance(mark, dict) or "opacity" not in mark:
                continue
            assert mark["opacity"] in (1, SECONDARY_OPACITY), (
                f"{viz_type} states a bespoke opacity {mark['opacity']!r}; the density weight and full ink are the only "
                "flat values left, and anything expressing EMPHASIS asks for `chart-context` by name"
            )
            assert mark["opacity"] == 1 or mark.get("style") is None, (
                f"{viz_type} recedes a mark by style AND by opacity — the style already carries the theme's answer, "
                "and the number overrides it with one chosen for neither theme"
            )


def _point_layers(spec):
    """The frontier's point marks, in drawn order.

    More than one: the weights are separate layers because a
    style is a property of a mark, so the recession could not be a channel on a
    single one without becoming a compiled opacity again.
    """
    return [layer for layer in spec["layer"] if layer["mark"]["type"] == "point"]


class TestFrontierCarriesDominanceOnShapeAndWeightRatherThanHue:
    """The redundant channel the chart rules leave this type.

    The deliverable's requirement is that a dominated point is never *merely*
    dimmer. Weight alone is what that forbids, so the test is not "the recession is
    absent" — it is that a second channel separates the classes, and that the
    second channel names geometry rather than a colour.

    **The recession's MECHANISM changed and its contract did not.**
    It was an opacity encoding reading a per-row weight, which put the number 0.35
    into the spec — one answer to "how far does a dominated point recede", fixed
    before either theme is known and therefore right on at most one of the two
    surfaces this spec is drawn onto. It is now a `chart-context` style on a layer
    of its own, which each renderer resolves. The assertions below moved with it;
    what they assert is the same thing they asserted before.
    """

    def test_the_contention_classes_are_separated_by_shape(self):
        primary, *_ = _point_layers(compile_chart("frontier", FRONTIER).spec)
        shape = primary["encoding"]["shape"]
        assert shape["scale"]["domain"] == ["On frontier", "Dominated"]
        assert shape["scale"]["range"] == ["circle", "diamond"]

    def test_every_weight_shares_one_shape_scale_so_the_key_stays_whole(self):
        """Splitting the marks must not split the key.

        A class whose every point recedes appears only in the second layer, and a
        per-layer shape scale would drop it from a key the reader needs precisely
        because that class is the one that lost.
        """
        scales = [
            layer["encoding"]["shape"]["scale"] for layer in _point_layers(compile_chart("frontier", FRONTIER).spec)
        ]
        assert len({json.dumps(scale, sort_keys=True) for scale in scales}) == 1

    def test_the_shape_range_names_symbols_and_never_a_colour(self):
        """Which is what lets this channel exist at all: identity may not ride on hue."""
        for layer in _point_layers(compile_chart("frontier", FRONTIER).spec):
            assert "color" not in layer["encoding"]
            assert set(layer["encoding"]["shape"]["scale"]["range"]) <= {"circle", "diamond", "cross"}

    def test_the_recession_is_the_second_channel_and_not_the_only_one(self):
        spec = compile_chart("frontier", FRONTIER).spec
        primary, recessive = _point_layers(spec)
        assert primary["mark"].get("style") is None
        assert recessive["mark"]["style"] == CONTEXT_STYLE
        # The dominated contestant is in the receding layer and the frontier one is not...
        assert recessive["transform"][0]["filter"]["oneOf"] == ["Dominated"]
        assert primary["transform"][0]["filter"]["oneOf"] == ["On frontier"]
        # ...and the same pair is already separated without consulting the weight at all.
        by_label = {row["label"]: row for row in compile_chart("frontier", FRONTIER).rows}
        assert by_label["model-b"]["status"] != by_label["model-a-3.5-fast-lite"]["status"]

    def test_the_recession_states_no_opacity_of_its_own(self):
        """The point of the retrofit, asserted where it would regress.

        A style AND a recessive opacity is the half-migration: the literal is back
        in the spec and the renderer's answer is silently overridden by it. Full
        ink is not that — it opts out of the point mark's density default, which
        both layers do alike, so it cannot be what separates them.
        """
        for layer in _point_layers(compile_chart("frontier", FRONTIER).spec):
            assert layer["mark"]["opacity"] == 1
            assert "opacity" not in layer["encoding"]

    def test_a_figure_with_nothing_receding_draws_only_one_layer(self):
        """An empty filter is an invisible mark that still joins scale resolution."""
        payload = FRONTIER | {
            "points": [{"label": "a", "cost": 0.01, "quality": 0.4}, {"label": "b", "cost": 0.02, "quality": 0.5}]
        }
        (only,) = _point_layers(compile_chart("frontier", payload).spec)
        assert only["mark"].get("style") is None

    def test_a_disqualified_point_outranks_a_dominated_one(self):
        """A two-pillar failure filed under 'lost on price' is the wrong verdict."""
        payload = FRONTIER | {
            "points": [
                {"label": "a", "cost": 0.01, "quality": 0.4},
                {
                    "label": "b",
                    "cost": 0.02,
                    "quality": 0.9,
                    "dominated": True,
                    "disqualified": True,
                    "disqualified_reason": "boundary",
                },
            ]
        }
        chart = compile_chart("frontier", payload)
        assert {row["label"]: row["status"] for row in chart.rows}["b"] == "Disqualified"

    def test_a_one_class_figure_draws_no_key(self):
        """A legend with one entry explains nothing and is furniture."""
        payload = FRONTIER | {
            "points": [{"label": "a", "cost": 0.01, "quality": 0.4}, {"label": "b", "cost": 0.02, "quality": 0.5}]
        }
        (points,) = _point_layers(compile_chart("frontier", payload).spec)
        assert points["encoding"]["shape"]["legend"] is None


class TestAFrontierKeepsItsAuthorsOneSentence:
    """An authored caption under a frontier that also needs every compiler disclosure.

    The author writes ONE sentence under a frontier with a dominated contestant and two
    failed arms that carried no cost. A compiler that joins the shape key, a
    disqualification sentence, a not-drawn sentence and the quality bar onto it makes the
    reader meet five sentences as one paragraph and read all of it as the author's. This
    fixture exercises exactly that combination — one dominated point, two disqualified
    unpriced arms and a quality bar — so each disclosure must arrive as its own line and the
    caption must stay the author's sentence alone.
    """

    AUTHORED = (
        "model-b pays ~15x model-a for the same accuracy; "
        "the two failed arms carry no cost because no call reached the model."
    )
    PAYLOAD = {
        "caption": AUTHORED,
        "bar": 0.8,
        "cost_label": "Cost (USD)",
        "quality_label": "Accuracy",
        "points": [
            {"label": "model=model-a", "cost": 0.003, "quality": 0.85},
            {"label": "model=model-b", "cost": 0.045, "quality": 0.85, "dominated": True},
            {"label": "model=model-c", "cost": 0.08, "quality": 0.88},
            {
                "label": "model=model-d",
                "quality": 0.0,
                "disqualified": True,
                "disqualified_reason": "no usable answer was returned",
            },
            {
                "label": "model=model-e",
                "quality": 0.0,
                "disqualified": True,
                "disqualified_reason": "no usable answer was returned",
            },
        ],
    }

    def test_the_caption_is_the_one_sentence_the_author_wrote(self):
        assert compile_chart("frontier", self.PAYLOAD).caption == self.AUTHORED

    def test_the_compilers_disclosures_arrive_as_their_own_lines(self):
        """The key, then who is out and why, then who is missing, then the bar — nothing dropped."""
        assert compile_chart("frontier", self.PAYLOAD).disclosures == [
            "circle = on frontier, diamond = dominated.",
            "model=model-d and model=model-e are disqualified: no usable answer was returned.",
            (
                "Not drawn — no production-replicating cost was recorded for model=model-d and "
                "model=model-e; the values below carry what was measured."
            ),
            "The quality bar is 0.8.",
        ]

    def test_the_key_is_a_line_and_never_a_drawn_legend(self):
        """The shape vocabulary moved out of the caption, not into the figure: the frontier
        draws no legend (its swatches take the mark's ink), and that rule stands."""
        shaped = [
            layer
            for layer in compile_chart("frontier", self.PAYLOAD).spec["layer"]
            if "shape" in layer.get("encoding", {})
        ]
        assert shaped, "the fixture stopped drawing a shape encoding, so this asserts nothing"
        assert all(layer["encoding"]["shape"]["legend"] is None for layer in shaped)


class TestFrontierDisclosesWhatItCouldNotDraw:
    def test_both_axes_crop_and_neither_restates_what_its_ticks_show(self):
        """Both axes ARE cropped — that half is unchanged and is asserted here.

        What changed is that neither says so in prose. A contestant's
        position is read off the labelled ticks on both axes, so the sentence
        restated the axis twice under a 400px figure; the crop rule scoped the
        footnote to marks whose WIDTH is the quantity.
        """
        payload = FRONTIER | {
            "points": [{"label": "a", "cost": 0.010, "quality": 0.71}, {"label": "b", "cost": 0.019, "quality": 0.61}],
            "bar": None,
        }
        spec = compile_chart("frontier", payload).spec
        domains = [encoding["scale"]["domain"] for encoding in _value_encodings(spec)]
        assert domains, "no value axis at all"
        assert all(low > 0 for low, _ in domains), f"an axis stopped cropping, so this proves nothing: {domains}"
        assert "cropped" not in _subtitle(spec)

    def test_an_unpriced_contestant_is_named_rather_than_dropped(self):
        """It has no x to be placed at, and silence would read as 'not in the running'."""
        payload = FRONTIER | {
            "points": [
                {"label": "a", "cost": 0.01, "quality": 0.4},
                {"label": "b", "cost": 0.02, "quality": 0.2},
                {"label": "never-priced", "quality": 0.9},
            ]
        }
        chart = compile_chart("frontier", payload)
        assert "never-priced" in _disclosed(chart)
        assert "never-priced" in {row["label"] for row in chart.rows}
        assert "never-priced" not in {row["label"] for row in chart.spec["data"]["values"]}

    def test_an_unpriced_contestant_is_not_reported_as_being_on_the_frontier(self):
        """Domination is a claim about BOTH axes, so a point with no cost supports neither verdict."""
        payload = FRONTIER | {
            "points": [
                {"label": "a", "cost": 0.01, "quality": 0.4},
                {"label": "b", "cost": 0.02, "quality": 0.2},
                {"label": "never-priced", "quality": 0.9},
            ]
        }
        chart = compile_chart("frontier", payload)
        assert {row["label"]: row["status"] for row in chart.rows}["never-priced"] == "Not priced"

    def test_a_disqualification_reason_reaches_the_reader(self):
        """The glyph says a contestant is out; only this says why."""
        payload = FRONTIER | {
            "points": [
                {"label": "a", "cost": 0.01, "quality": 0.4},
                {
                    "label": "b",
                    "cost": 0.02,
                    "quality": 0.9,
                    "disqualified": True,
                    "disqualified_reason": "failed the boundary battery",
                },
            ]
        }
        assert "failed the boundary battery" in _disclosed(compile_chart("frontier", payload))

    def test_contestants_out_for_one_reason_are_disqualified_in_one_sentence(self):
        """The reason is the disclosure, and a caption that repeats it per contestant stops being read.

        Observed whole, on a live memo: a four-contestant frontier whose two failures shared one
        cause printed that cause twice, inside a caption a reader could not follow. The repetition
        is mechanical — it grows with the size of the class being disclosed — so it is fixed where
        it is produced rather than asked for in the prompt, which did not write this half.
        """
        reason = "100% parse failure — no usable classification"
        payload = FRONTIER | {
            "points": [
                {"label": "a", "cost": 0.01, "quality": 0.4},
                {"label": "b", "cost": 0.02, "quality": 0.2},
                {"label": "c", "quality": 0.0, "disqualified": True, "disqualified_reason": reason},
                {"label": "d", "quality": 0.0, "disqualified": True, "disqualified_reason": reason},
            ]
        }
        disclosed = _disclosed(compile_chart("frontier", payload))

        assert f"c and d are disqualified: {reason}." in disclosed
        assert disclosed.count(reason) == 1, f"the shared reason is printed more than once: {disclosed!r}"

    def test_contestants_out_for_different_reasons_each_keep_their_own(self):
        """Grouping may not merge two causes: the sentence says why, and two whys are two sentences."""
        payload = FRONTIER | {
            "points": [
                {"label": "a", "cost": 0.01, "quality": 0.4},
                {"label": "b", "cost": 0.02, "quality": 0.2},
                {"label": "c", "quality": 0.0, "disqualified": True, "disqualified_reason": "no usable classification"},
                {
                    "label": "d",
                    "quality": 0.0,
                    "disqualified": True,
                    "disqualified_reason": "failed the boundary battery",
                },
            ]
        }
        disclosed = _disclosed(compile_chart("frontier", payload))

        assert "c is disqualified: no usable classification." in disclosed
        assert "d is disqualified: failed the boundary battery." in disclosed

    def test_a_quality_bar_above_every_contestant_still_draws_inside_the_plot(self):
        """The case where the bar matters most is the one an unpadded axis would drop.

        Nothing cleared it, so the rule sits above every mark — and an axis cropped
        to the contestants alone would put it off the top edge, drawing the chart
        that says 'no one qualified' as one with no bar at all.
        """
        payload = FRONTIER | {
            "points": [{"label": "a", "cost": 0.01, "quality": 0.1}, {"label": "b", "cost": 0.02, "quality": 0.2}],
            "bar": 0.9,
        }
        spec = compile_chart("frontier", payload).spec
        (rule,) = [layer for layer in spec["layer"] if layer["mark"]["type"] == "rule"]
        low, high = rule["encoding"]["y"]["scale"]["domain"]
        assert low <= 0.9 <= high, f"the bar at 0.9 falls outside the drawn domain [{low}, {high}]"


class TestSweepRankingRanksAndNeverManufacturesItsFinding:
    """The barcode's whole claim is that a column sorted itself.

    So the sort order is the correctness question here, not a presentation one: a
    figure sorted BY a dimension produces a contiguous block in that dimension's
    column for any data at all, which is exactly the pattern the reader is being
    invited to read as a finding.
    """

    def test_rows_descend_by_the_ranked_measure(self):
        chart = compile_chart("sweep_ranking", SWEEP_RANKING)
        drawn = [row["ranked"] for row in chart.rows]
        assert drawn == sorted(drawn, reverse=True), "best is top-right, so the marks descend"

    def test_the_ranking_panel_declares_itself_a_ranking(self):
        """The line that SUBMITS this chart to rule 10, pinned where deleting it fails.

        Rule 10 only judges a frame that names itself a ranking, so
        ``_ranking_panel``'s ``"name": RANKING_SPEC_NAME`` is the whole of what
        makes the gate reachable on the only chart it was written for. Every
        rule-10 case in ``test_vega_spec_policy.py`` builds its own spec and sets that
        name itself — i.e. the tests supply the production wiring — so without
        this assertion the line can be deleted and the entire suite stays green
        while the rule silently goes inert. That is the exact state this work
        was undertaken to escape.
        """
        spec = compile_chart("sweep_ranking", SWEEP_RANKING).spec
        named = [node for node in spec["hconcat"] if node.get("name") == RANKING_SPEC_NAME]
        assert len(named) == 1, f"exactly one frame declares the ranking claim, got {len(named)}"
        assert "ranked" in json.dumps(named[0]), "the named frame is the one plotting the measure"

    def test_the_compiled_ranking_is_refused_when_its_order_stops_matching_its_measure(self):
        """Non-vacuity: the gate engages on the REAL chart, not only on fixtures.

        `test_the_ranking_panel_declares_itself_a_ranking` proves the claim is
        made; this proves the claim is CHECKED. A rule that fires and always
        passes is indistinguishable from no rule, so the assertion that matters
        is that a defect in this chart is caught — here by reversing the stated
        row order, which changes no value and no mark, only the sequence.
        """
        spec = compile_chart("sweep_ranking", SWEEP_RANKING).spec
        assert check_spec(spec) == [], "the chart as compiled conforms"

        broken = copy.deepcopy(spec)
        # The layers share one sort list, so reverse by object identity — reversing
        # "each layer" would reverse the same list twice and restore the original.
        seen: set[int] = set()
        for layer in broken["hconcat"][1]["layer"]:
            order = (layer.get("encoding") or {}).get("y", {}).get("sort")
            if isinstance(order, list) and id(order) not in seen:
                seen.add(id(order))
                order.reverse()
        violations = check_spec(broken)
        assert any("matches none of the quantities it draws" in v for v in violations), violations

    def test_the_order_does_not_follow_any_dimension(self):
        """Constructed so that sorting by either lever gives a different order.

        A weaker fixture — one whose dimension order happens to match its ranking —
        would pass whichever rule the compiler implemented.
        """
        payload = {
            **SWEEP_RANKING,
            "rows": [
                {"config": {"model": "a-model", "depth": "1"}, "ranked_value": 0.4, "secondary_value": 0.01},
                {"config": {"model": "z-model", "depth": "3"}, "ranked_value": 0.9, "secondary_value": 0.02},
                {"config": {"model": "m-model", "depth": "2"}, "ranked_value": 0.6, "secondary_value": 0.03},
            ],
        }
        chart = compile_chart("sweep_ranking", payload)
        model_key = next(column["key"] for column in chart.columns if column["header"] == "model")
        assert [row[model_key] for row in chart.rows] == ["z-model", "m-model", "a-model"]
        assert [row["ranked"] for row in chart.rows] == [0.9, 0.6, 0.4]

    def test_an_ordered_lever_takes_a_linear_ramp_and_never_an_ordinal_one(self):
        """The recycling case, and it is pinned here because it is silent.

        An ordinal scale maps the fifth level past the last authored stop back onto
        the first, so a sweep with more levels than stops draws its darkest level in
        its lightest one's ink — and a contiguous block then reads as having
        wrapped, which is worse than no ramp at all. A linear scale over the rank
        samples the path instead.
        """
        spec = compile_chart("sweep_ranking", SWEEP_RANKING).spec
        ramps = [
            layer["encoding"]["color"]
            for layer in _mark_layers(spec)
            if isinstance((layer.get("encoding") or {}).get("color"), dict)
            and layer["encoding"]["color"].get("scale", {}).get("range") == SEQUENTIAL_RANGE
        ]
        assert ramps, "no column draws from the ordered ramp"
        for ramp in ramps:
            assert ramp["type"] == "quantitative"
            assert ramp["scale"]["range"] == SEQUENTIAL_RANGE

    def test_more_levels_than_authored_stops_still_draw_distinctly(self):
        """The behaviour the linear scale buys, measured through the renderer.

        Reading the spec cannot show this — the spec is identical whatever the level
        count — so the claim that levels close up rather than colliding is checked
        where it becomes true, in the pixels.
        """
        stops = len(sequential_colors("dark"))
        rows = [
            {"config": {"model": "m", "depth": str(index)}, "ranked_value": 1.0 - index / 20, "secondary_value": 0.01}
            for index in range(stops + 2)
        ]
        spec = compile_chart("sweep_ranking", {**SWEEP_RANKING, "rows": rows}).spec
        svg = render_svg(spec, theme="dark")
        cells = set(re.findall(r'fill="(rgb\([^)]*\)|#[0-9a-fA-F]{6})"', svg))
        assert len(cells) >= stops + 2, f"{stops + 2} levels drew only {len(cells)} distinct inks"

    def test_the_two_ink_vocabularies_never_share_a_scale(self):
        """A concurrency rank and a model name are not values of one thing.

        Forced onto one domain they would be, which is the defect an independently
        resolved colour scale usually IS — and is why the policy gate reads the
        domains rather than refusing the declaration on sight.
        """
        spec = compile_chart("sweep_ranking", SWEEP_RANKING).spec
        barcode = spec["hconcat"][0]
        assert barcode["resolve"]["scale"]["color"] == "independent"
        domains = [
            layer["encoding"]["color"]["scale"].get("domain")
            for layer in barcode["layer"]
            if "color" in layer["encoding"]
        ]
        flattened = [str(entry) for domain in domains for entry in domain]
        assert len(set(flattened)) == len(flattened), f"a category is coloured twice: {flattened}"

    def test_the_barcode_and_the_ranking_share_one_row_scale(self):
        """Or the glyph describes a different configuration from the mark beside it.

        Vega-Lite resolves a concat's positional scales independently by default,
        which is right for two panels measuring two quantities and catastrophic for
        two halves of one row.
        """
        spec = compile_chart("sweep_ranking", SWEEP_RANKING).spec
        assert spec["resolve"]["scale"]["y"] == "shared"


def _passes(transforms: list[dict], mark: dict) -> bool:
    """Whether a barcode layer's filters admit a datum.

    A partial Vega-Lite filter evaluator, covering exactly the three predicate
    forms this compiler emits — `equal`, `valid` and `oneOf`. Small on purpose: the
    question a test needs answered is "does SOME layer draw this cell", and a cell
    no layer draws is invisible on both renderers rather than an error, so nothing
    downstream of the spec can be asked instead.
    """
    for transform in transforms:
        predicate = transform["filter"]
        value = mark.get(predicate["field"])
        if "equal" in predicate and value != predicate["equal"]:
            return False
        if "valid" in predicate and (value is not None) != predicate["valid"]:
            return False
        if "oneOf" in predicate and value not in predicate["oneOf"]:
            return False
    return True


class TestSweepRankingStatesWhatTheBarcodeCannotSay:
    def test_a_disclosure_names_the_columns_left_to_right(self):
        """The cells are too narrow for a lever's name and the label rule refuses to shrink one."""
        chart = compile_chart("sweep_ranking", SWEEP_RANKING)
        assert "Columns, left to right: fetch_concurrency, model, search_depth" in _disclosed(chart)

    def test_an_unconstrained_sweep_states_the_secondary_measure_s_range(self):
        """Without it, a block in one column could equally mean 'those cost more'."""
        chart = compile_chart("sweep_ranking", SWEEP_RANKING)
        assert "is not held — the ranking is not controlled for it" in _disclosed(chart)
        assert "0.0081 usd" in _disclosed(chart) and "0.0142 usd" in _disclosed(chart)

    def test_a_constrained_sweep_states_the_tolerance_instead(self):
        """Same layout, different claim — and the disclosures are the only thing that says which."""
        held = {**SWEEP_RANKING, "held_fixed": {"value": 0.0105, "tolerance": 0.002}}
        constrained = compile_chart("sweep_ranking", held)
        free = compile_chart("sweep_ranking", SWEEP_RANKING)
        assert "is held at" in _disclosed(constrained)
        assert "not controlled for it" not in _disclosed(constrained)
        # The layout is the same figure; only the claim beside it changed.
        assert constrained.spec["hconcat"][0]["width"] == free.spec["hconcat"][0]["width"]
        assert constrained.title == free.title

    def test_a_tolerance_that_admits_nothing_is_a_result_rather_than_a_refusal(self):
        """An empty slice is a result; an empty chart is a bug.

        Refusing the payload would discard a whole paid analysis over a measurement
        outcome, and refusing legitimate data is the one answer that is never right
        (the palette never refuses to draw). So the figure is
        drawn from the sweep the slice was
        taken from, and a disclosure says the slice found nothing.
        """
        empty = {**SWEEP_RANKING, "held_fixed": {"value": 99.0, "tolerance": 0.001}}
        chart = compile_chart("sweep_ranking", empty)
        assert "so the slice is empty" in _disclosed(chart)
        # Drawn from the sweep the slice was taken from, so the reader can read off
        # the secondary values WHY nothing qualified rather than facing a blank frame.
        assert len(chart.rows) == len(SWEEP_RANKING["rows"])
        assert chart.spec["hconcat"][1]["data"]["values"], "an empty frame is the bug this branch exists to avoid"

    def test_an_inferred_orderedness_is_admitted_rather_than_presented_as_fact(self):
        chart = compile_chart("sweep_ranking", SWEEP_RANKING)
        assert "was inferred from the levels rather than declared" in _disclosed(chart)

    def test_a_declared_orderedness_is_not_reported_as_inferred(self):
        declared = {
            **SWEEP_RANKING,
            "dimensions": [
                {"name": "model", "ordered": False},
                {"name": "fetch_concurrency", "ordered": True},
                {"name": "search_depth", "ordered": True},
            ],
        }
        assert "inferred" not in _disclosed(compile_chart("sweep_ranking", declared))

    @staticmethod
    def _named_levels() -> dict:
        """A sweep whose ordered lever is declared over levels that do not parse.

        `small`/`large` is the case the `ordered` field EXISTS for — the inference
        refuses non-numeric levels by design, so a generator that means "these run
        low to high" has no other way to say it. Every declared-dimensions fixture
        before this one declared what the inference would have guessed anyway,
        which is why none of them could tell a working resolution from a broken one.

        Four categorical levels, which is the ceiling exactly. Not a convenience:
        a lever drawn as hues CONSUMES hues, so a lever demoted out of the ramp
        must count against that ceiling, and a fixture one level wider is refused
        by the payload rather than compiled — correctly.
        """
        return {
            "ranked": {"measure": "pass^k", "unit": None},
            "secondary": {"measure": "cost per run", "unit": "usd"},
            "dimensions": [{"name": "model", "ordered": False}, {"name": "size", "ordered": True}],
            "rows": [
                {"config": {"model": "gpt-5", "size": "small"}, "ranked_value": 0.72, "secondary_value": 0.011},
                {"config": {"model": "model-b", "size": "large"}, "ranked_value": 0.61, "secondary_value": 0.009},
                {"config": {"model": "gpt-5", "size": "large"}, "ranked_value": 0.55, "secondary_value": 0.014},
            ],
        }

    def test_a_declared_order_no_level_can_be_placed_in_still_draws_every_cell(self):
        """The failure this guards is a BLANK column, which renders without erroring.

        A ramp is a position along a sorted range, so levels that cannot be sorted
        cannot take one however they were declared. When the declaration and the
        data were answered by two different rules, such a lever landed ordered for
        the categorical domain (excluded from it) and categorical for the rank
        (no rank) — so its cells matched neither barcode layer and the column drew
        as nothing at all, under a caption still promising a light-to-dark ramp.

        Asserted as coverage of the cells rather than as the shape of the fix: what
        must be true is that every cell is drawn by SOME layer, which stays the
        contract whichever way a future resolution decides to honour the
        declaration.
        """
        spec = compile_chart("sweep_ranking", self._named_levels()).spec
        barcode = spec["hconcat"][0]
        # A single-layer panel is flattened rather than wrapped, so read both
        # shapes: with the declaration honoured as far as the levels allow, this
        # sweep has no ramp and no absence and collapses to one categorical layer.
        drawn: set[tuple[str, str]] = set()
        for layer in barcode.get("layer", [barcode]):
            for mark in barcode["data"]["values"]:
                if _passes(layer.get("transform", []), mark):
                    drawn.add((mark["dimension"], mark["level"]))

        every_cell = {(mark["dimension"], mark["level"]) for mark in barcode["data"]["values"]}
        assert drawn == every_cell, f"cells no layer draws: {sorted(every_cell - drawn)}"

    def test_a_higher_level_draws_at_the_dark_end_of_the_ramp(self):
        """Which end means "more" is the whole readability of the glyph, and it was unpinned.

        Nothing asserted it, so a reversed scale domain would have drawn every
        ordered column backwards with a full suite green and a caption still
        saying "light-to-dark". The direction is not arbitrary: `chart.seq` is
        authored lightest at stop 1 through darkest at stop 5, and `chart.context`
        — the neutral for a level a configuration never set — is pale, so a
        lightest-is-most ramp would draw the highest level and an absence alike.

        Asserted through the rank, which is what the scale actually reads: rank 0
        is the lowest level and must land on `domain[0]`, the range's first and
        lightest stop.
        """
        spec = compile_chart("sweep_ranking", SWEEP_RANKING).spec
        ramp = next(
            layer for layer in spec["hconcat"][0]["layer"] if layer["encoding"].get("color", {}).get("field") == "rank"
        )
        assert ramp["encoding"]["color"]["scale"]["domain"][0] == 0, "the lowest rank must anchor the light end"
        assert ramp["encoding"]["color"]["scale"]["domain"][1] > 0, "the domain must ascend, or the ramp is reversed"

        marks = spec["hconcat"][0]["data"]["values"]
        ranked = {mark["level"]: mark["rank"] for mark in marks if mark["dimension"] == "fetch_concurrency"}
        assert ranked["1"] < ranked["8"], "a higher level must take a higher rank, hence a darker stop"

    def test_a_declaration_the_levels_cannot_support_is_admitted_in_the_disclosures(self):
        """Overriding a generator's stated intent in silence is the other half of it.

        The disclosures already admit the reverse — an orderedness the compiler guessed
        — for the same reason: what the figure asserts about the data has to be
        traceable to who asserted it.
        """
        disclosed = _disclosed(compile_chart("sweep_ranking", self._named_levels()))
        assert "size was declared ordered but draws as hues" in disclosed
        assert "size draws as a light-to-dark ramp" not in disclosed

    def test_the_values_table_carries_every_lever_and_both_measures(self):
        """The acceptance criterion: a reader who cannot see the barcode can still check it.

        Asserted on the HEADERS rather than the keys, which is what the criterion is
        actually about — a lever is keyed in its own namespace so no lever name can
        land on a measure's column, and the key is addressing rather than anything a
        reader meets. The header is where the lever's own name has to survive, and
        this once read the keys, which is why it noticed the namespace at all.
        """
        chart = compile_chart("sweep_ranking", SWEEP_RANKING)
        headers = {column["header"] for column in chart.columns}
        assert {"model", "fetch_concurrency", "search_depth"} <= headers
        measures = {column["key"] for column in chart.columns}
        assert {"ranked", "secondary"} <= measures


class TestSweepRankingNeverTruncatesSilently:
    """Thirty-six configurations at the row step is a figure the reader scrolls."""

    @staticmethod
    def _wide(count: int) -> dict:
        """A sweep of `count` configurations over two levers, both ORDERED.

        Ordered on purpose rather than for convenience: a wide sweep is a crossing
        of numeric knobs, and a categorical lever this wide is refused by the
        payload's own palette ceiling — so a fixture built from twelve model names
        would be testing a payload the contract does not admit.
        """
        return {
            **SWEEP_RANKING,
            "rows": [
                {
                    "config": {"timeout_s": str(index), "depth": str(index % 3)},
                    "ranked_value": 1.0 - index / 100,
                    "secondary_value": 0.01,
                }
                for index in range(count)
            ],
        }

    def test_a_sweep_at_the_bound_draws_every_configuration(self):
        chart = compile_chart("sweep_ranking", self._wide(12))
        assert len(chart.spec["hconcat"][1]["data"]["values"]) == 12
        assert "not drawn" not in _disclosed(chart)

    def test_a_sweep_past_the_bound_keeps_the_top_ten_and_says_what_it_dropped(self):
        chart = compile_chart("sweep_ranking", self._wide(13))
        assert len(chart.spec["hconcat"][1]["data"]["values"]) == 10
        assert "3 further configurations ranked between" in _disclosed(chart)
        # The band is stated, not just the count — a reader needs to know whether
        # what was dropped could have changed the verdict.
        assert "0.88" in _disclosed(chart) and "0.9" in _disclosed(chart)

    def test_the_omission_reaches_the_figure_itself_and_not_only_the_disclosures(self):
        """A reader looking at the picture alone must not see a complete sweep."""
        assert "further configurations" in _subtitle(compile_chart("sweep_ranking", self._wide(13)).spec)

    def test_a_producer_declared_omission_is_added_to_the_compiler_s_own(self):
        """Two sources, one sentence — a reader counting marks is owed the total."""
        payload = self._wide(13) | {"omitted": {"count": 20, "low": 0.1, "high": 0.2}}
        chart = compile_chart("sweep_ranking", payload)
        assert "23 further configurations ranked between 0.1 and 0.9" in _disclosed(chart)

    def test_the_values_table_keeps_every_qualifying_configuration(self):
        """Truncation is what the FIGURE does; the table is where the rest survive."""
        chart = compile_chart("sweep_ranking", self._wide(13))
        assert len(chart.rows) == 13


class TestSweepRankingDrawsAnAbsenceAsAnAbsence:
    """A lever a configuration never set has no level, and must not draw as one.

    The bundle writes an em dash for an unset override, so it arrives as a level
    like any other. Drawn in a hue it would be a value the reader compares; left
    out of every domain it draws in whatever Vega picks for an out-of-domain datum,
    which is the same defect reached by accident rather than by decision.
    """

    ABSENT = {
        **SWEEP_RANKING,
        "rows": [
            {"config": {"model": "gpt-5", "depth": "2"}, "ranked_value": 0.72, "secondary_value": 0.011},
            {"config": {"model": "model-b", "depth": "4"}, "ranked_value": 0.61, "secondary_value": 0.009},
            {"config": {"model": "gpt-5", "depth": "—"}, "ranked_value": 0.55, "secondary_value": 0.014},
        ],
    }

    def _barcode(self, payload):
        return compile_chart("sweep_ranking", payload).spec["hconcat"][0]

    def test_an_unset_lever_takes_the_neutral_that_carries_no_identity(self):
        absent = [layer for layer in self._barcode(self.ABSENT)["layer"] if layer["mark"].get("style") == CONTEXT_STYLE]
        assert len(absent) == 1, "no layer draws the absence as an absence"
        assert absent[0]["transform"][0]["filter"] == {"field": "level", "equal": "—"}
        assert "color" not in absent[0]["encoding"], "an absence with a hue is a level"

    def test_the_sentinel_reaches_no_colour_domain(self):
        """Either domain admitting it would put an absence on a comparison scale."""
        for layer in self._barcode(self.ABSENT)["layer"]:
            colour = layer["encoding"].get("color")
            if isinstance(colour, dict):
                assert "—" not in [str(entry) for entry in colour["scale"]["domain"]]

    def test_every_cell_is_drawn_by_exactly_one_layer(self):
        """The filters partition the cells; an overlap double-draws and a gap blanks one.

        Checked by replaying each layer's filter over the barcode's own data rather
        than by reading the filters, because the failure is about their interaction
        and reading them one at a time is what let the gap open.
        """
        barcode = self._barcode(self.ABSENT)
        cells = barcode["data"]["values"]
        drawn = collections.Counter()
        for index, layer in enumerate(barcode["layer"]):
            for cell in cells:
                if all(_matches(cell, step["filter"]) for step in layer["transform"]):
                    drawn[(cell["display"], cell["dimension"])] += 1
        assert set(drawn.values()) == {1}, f"cells drawn by the wrong number of layers: {drawn}"
        assert len(drawn) == len(cells)

    def test_a_sweep_whose_categorical_lever_is_mostly_unset_still_fits_the_palette(self):
        """The sentinel consumes no hue, so counting it would refuse a conforming sweep."""
        payload = {
            **SWEEP_RANKING,
            "rows": [
                {
                    "config": {"model": f"m{index}" if index < 4 else "—", "depth": str(index)},
                    "ranked_value": 1.0 - index / 20,
                    "secondary_value": 0.01,
                }
                for index in range(8)
            ],
        }
        assert compile_chart("sweep_ranking", payload).rows


def _matches(cell, predicate):
    """Whether one barcode cell satisfies one Vega-Lite filter predicate."""
    value = cell[predicate["field"]]
    if "valid" in predicate:
        return (value is not None) == predicate["valid"]
    if "equal" in predicate:
        return value == predicate["equal"]
    return value in predicate["oneOf"]


def _declared_unit(viz_type, payload):
    """The unit a payload declares for its primary quantity, or None."""
    if viz_type == "sweep_ranking":
        return (payload.get("ranked") or {}).get("unit")
    return payload.get("unit")


def _restatable_values(viz_type, payload):
    """Every value the chart's primary quantity is drawn from, unrestated."""
    if viz_type == "sweep_ranking":
        return [row["ranked_value"] for row in payload["rows"]]
    if viz_type == "breakdown":
        return [part["value"] for part in payload["parts"]]
    if viz_type == "attribution":
        # Every number the payload states in this unit, not only the two deltas: a
        # caption quoting an unrestated LEVEL is the same defect as one quoting an
        # unrestated movement, and the levels are the larger figures.
        movements = [payload["end_to_end"], payload["subsystem"]]
        return [
            value
            for value in (
                *(movement.get(key) for movement in movements for key in ("delta", "a", "b")),
                payload.get("unattributed_delta"),
            )
            if isinstance(value, int | float)
        ]
    if viz_type == "distribution":
        return [
            bound
            for group in payload["groups"]
            for bound in (group.get("ci") or {}).values()
            if isinstance(bound, float)
        ]
    if viz_type == "null_result":
        return [bound for group in payload["groups"] for bound in group["ci"].values() if isinstance(bound, float)]
    return []


class TestSweepRankingStatesItsOmissionInTheUnitTheAxisUses:
    """The fix's own coverage, which the first attempt at it did not have.

    `_omission_sentence` scales its band, and every fixture above declares
    `ranked.unit: None` — for which `display_scale` returns a factor of 1.0. So
    all four omission tests passed identically with the defect restored: the
    multiplication was there and was always by one. A fix whose only tests cannot
    distinguish it from the bug is the bug still shipping, with a green suite
    attached.

    This fixture states a unit the restatement ladder actually moves.
    """

    RESTATED = {
        "ranked": {"measure": "total latency", "unit": "ms"},
        "secondary": {"measure": "cost per run", "unit": "usd"},
        "rows": [
            {
                "config": {"timeout_s": str(index), "depth": str(index % 3)},
                # Milliseconds that ladder into seconds — 12000 down to 10800.
                "ranked_value": 12000.0 - index * 100,
                "secondary_value": 0.01,
            }
            for index in range(13)
        ],
    }

    def test_the_omitted_band_is_restated_with_the_axis(self):
        """Otherwise the sentence names a band in ms beside marks labelled in s."""
        chart = compile_chart("sweep_ranking", self.RESTATED)
        assert "further configurations ranked between 10.8 and 11" in _disclosed(chart), _disclosed(chart)
        # ...and emphatically not the raw milliseconds, which is what shipped.
        assert "10800" not in _disclosed(chart) and "11000" not in _disclosed(chart)

    def test_a_producer_declared_band_is_restated_too(self):
        """It arrives in the payload's unit, not the drawn one — the same rescaling applies."""
        payload = self.RESTATED | {"omitted": {"count": 4, "low": 9000.0, "high": 9500.0}}
        disclosed = _disclosed(compile_chart("sweep_ranking", payload))
        assert "between 9 and 11" in disclosed, disclosed
        assert "9000" not in disclosed and "9500" not in disclosed

    def test_the_axis_really_did_restate_or_this_proves_nothing(self):
        """Non-vacuity: if the ladder stopped moving, both assertions above pass trivially."""
        chart = compile_chart("sweep_ranking", self.RESTATED)
        assert chart.unit == "s", f"the ranked measure was not restated, so the band's unit is untested: {chart.unit}"

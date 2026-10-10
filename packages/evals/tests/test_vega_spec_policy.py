"""The spec policy gate — the presentation rules, checked against a spec.

Written against hand-authored specs rather than compiler output on purpose. The
gate's whole reason for sitting outside the compiler is that it must judge specs
the compiler did not write, so a test that only ever feeds it compiler output
would be testing the compiler and calling it a gate.

The multi-view cases carry their weight twice over: each frame-scoped rule is
pinned in BOTH directions, because a scope that is too narrow passes a nested
violation and a scope that is too wide refuses a figure that is drawn correctly,
and only one of those two failures is visible from a single-view spec.
"""

import logging

import pytest

from threetears.evals.vega.palette import VALUE_ON_FILL_STYLE
from threetears.evals.vega.spec_policy import SpecPolicyError, check_spec, enforce_spec


def _bar(**overrides):
    """A minimal conforming bar spec."""
    spec = {
        "data": {"values": [{"c": "a", "v": 1}, {"c": "b", "v": 2}]},
        "mark": "bar",
        "encoding": {
            "y": {"field": "c", "type": "nominal", "axis": {"title": None}},
            "x": {"field": "v", "type": "quantitative", "axis": {"title": "latency (ms)"}},
        },
    }
    spec.update(overrides)
    return spec


#: A row facet whose header is pinned upright, which every conforming facet must
#: state — Vega turns a row header a quarter turn when the spec says nothing.
_UPRIGHT_ROW = {"field": "g", "type": "nominal", "header": {"labelAngle": 0}}


def _panel(axis_title, **overrides):
    """A conforming bar view stating ``axis_title`` on x, for use inside a concat."""
    panel = _bar(**overrides)
    panel["encoding"]["x"]["axis"] = {"title": axis_title}
    return panel


class TestConformingSpecs:
    def test_a_plain_bar_spec_passes(self):
        assert check_spec(_bar()) == []

    def test_enforce_is_silent_on_a_conforming_spec(self):
        enforce_spec(_bar())

    def test_a_line_may_omit_zero_when_it_admits_it(self):
        """A line encodes position, not length — cropping its axis is legitimate.

        Legitimate and disclosed: an axis that does not start at zero rescales every
        comparison drawn on it, so the crop is stated in the picture rather than
        left for the reader to notice from the tick labels.
        """
        spec = _bar(mark="line")
        spec["encoding"]["x"]["scale"] = {"zero": False}
        # A line through categories states their order (rule 11), so this one does.
        spec["encoding"]["y"]["sort"] = ["a", "b"]
        spec["title"] = {"text": "latency", "subtitle": "Axis is cropped to the data and excludes zero."}
        assert check_spec(spec) == []


class TestShapeOnlyFromValues:
    def test_a_bar_with_a_truncated_baseline_is_rejected(self):
        spec = _bar()
        spec["encoding"]["x"]["scale"] = {"zero": False}
        assert any("truncates" in violation for violation in check_spec(spec))

    def test_a_bar_whose_domain_excludes_zero_is_rejected(self):
        """`zero: true` is not the only way to lose the baseline."""
        spec = _bar()
        spec["encoding"]["x"]["scale"] = {"domain": [10, 20]}
        assert any("excludes zero" in violation for violation in check_spec(spec))

    def test_an_area_is_held_to_the_same_rule_as_a_bar(self):
        spec = _bar(mark="area")
        spec["encoding"]["x"]["scale"] = {"zero": False}
        assert any("truncates" in violation for violation in check_spec(spec))

    def test_a_bar_placed_on_a_cropped_axis_keeps_its_baseline_on_the_other_one(self):
        """A histogram bin: its HEIGHT is the count and its POSITION is the value.

        A bar measures along one axis and is placed by the other, and which is which
        is Vega-Lite's decision — the channel it measures along is the one whose
        scale asserts a zero. Demanding zero on both would demand that a marginal be
        drawn on an axis starting at zero, which is the rescaling a position axis
        crops to avoid: two spreads six seconds apart on a 0-60s axis draw as one
        picture.
        """
        spec = {
            "title": "latency",
            "mark": "bar",
            "encoding": {
                "x": {
                    "field": "at",
                    "type": "quantitative",
                    "scale": {"domain": [1.2, 1.9], "zero": False},
                    "axis": {"title": None, "domain": True},
                },
                "y": {
                    "field": "n",
                    "type": "quantitative",
                    "scale": {"domain": [0, 64], "zero": True},
                    "axis": {"title": None},
                },
            },
        }
        assert [
            violation for violation in check_spec(spec) if "baseline" in violation or "excludes zero" in violation
        ] == []

    def test_a_bar_with_no_baseline_on_any_axis_is_still_rejected(self):
        """The rule the case above narrows, in the direction that must not widen:
        a bar measuring from wherever its domain starts encodes a ratio the values
        do not hold, and having a second cropped axis does not excuse the first."""
        spec = {
            "title": "latency",
            "mark": "bar",
            "encoding": {
                "x": {"field": "at", "type": "quantitative", "scale": {"zero": False}, "axis": {"title": "ms"}},
                "y": {"field": "n", "type": "quantitative", "scale": {"domain": [10, 64]}, "axis": {"title": "n"}},
            },
        }
        violations = check_spec(spec)
        assert any("truncates the x baseline" in violation for violation in violations), violations
        assert any("y domain [10, 64] excludes zero" in violation for violation in violations), violations


class TestOneUnitPerQuantity:
    def test_a_quantitative_axis_with_no_title_is_rejected(self):
        spec = _bar()
        spec["encoding"]["x"]["axis"] = {"title": ""}
        assert any("states no title" in violation for violation in check_spec(spec))

    def test_a_quantitative_axis_with_no_axis_object_is_rejected(self):
        spec = _bar()
        del spec["encoding"]["x"]["axis"]
        assert any("states no title" in violation for violation in check_spec(spec))

    def test_two_layers_stating_different_units_on_one_axis_are_rejected(self):
        """The one-unit-per-quantity violation the rule is named for — two rulers presented as one."""
        spec = {
            "data": {"values": [{"c": "a", "v": 1}]},
            "layer": [
                {
                    "mark": "bar",
                    "encoding": {"x": {"field": "v", "type": "quantitative", "axis": {"title": "latency (ms)"}}},
                },
                {
                    "mark": "bar",
                    "encoding": {"x": {"field": "v", "type": "quantitative", "axis": {"title": "cost (usd)"}}},
                },
            ],
        }
        assert any("more than one unit" in violation for violation in check_spec(spec))

    def test_two_layers_agreeing_on_one_unit_pass(self):
        spec = {
            "data": {"values": [{"c": "a", "v": 1}]},
            "layer": [
                {
                    "mark": "bar",
                    "encoding": {"x": {"field": "v", "type": "quantitative", "axis": {"title": "latency (ms)"}}},
                },
                {
                    "mark": "bar",
                    "encoding": {"x": {"field": "v", "type": "quantitative", "axis": {"title": "latency (ms)"}}},
                },
            ],
        }
        assert check_spec(spec) == []

    def test_two_panels_measuring_different_quantities_pass(self):
        """Two frames are two rulers, each with its own axis — that is a figure, not a violation.

        Pooling titles across the whole spec refused this, and it is the shape a
        report reaches for whenever one figure answers a question in two units.
        """
        assert check_spec({"vconcat": [_panel("latency (ms)"), _panel("cost (usd)")]}) == []

    def test_two_layers_inside_one_panel_still_have_to_agree(self):
        """The same-frame rule survives the rescoping, one level down."""
        spec = {"vconcat": [{"layer": [_panel("latency (ms)"), _panel("cost (usd)")]}]}
        violations = check_spec(spec)
        assert any("more than one unit" in violation for violation in violations)
        assert any("vconcat[0]" in violation for violation in violations), violations


class TestTheKnockoutInkStaysWhereItWasMeasured:
    """The third chart ink is bounded, and this is what holds the bound.

    A palette's `on_fill` ink is for a value drawn on a mark's fill, measured against
    slot 1 — the single-series mark colour. Over other slots the same ink can fall under
    the 4.5:1 it exists to clear (packaged: 3.52:1 over slot 2 in dark, 2.45:1 over slot 7
    in light). Nothing in the repo writes a value on another slot today; the gate exists
    so the day something does is a refusal rather than a silent contrast regression.
    """

    def _labelled(self, **mark_overrides):
        """A value label asking for the on-fill ink, in an otherwise conforming frame."""
        return {
            "data": {"values": [{"c": "a", "v": 1}]},
            "mark": {"type": "text", "style": VALUE_ON_FILL_STYLE, **mark_overrides},
            "encoding": {
                "y": {"field": "c", "type": "nominal", "axis": {"title": None}},
                "x": {"field": "v", "type": "quantitative", "axis": {"title": "latency (ms)"}},
                "text": {"field": "v", "type": "nominal"},
            },
        }

    def test_a_value_label_may_ask_for_it(self):
        assert check_spec(self._labelled()) == []

    def test_a_non_text_mark_may_not(self):
        """That ink can be the chart surface itself, so painting a MARK with it draws a hole."""
        spec = self._labelled()
        spec["mark"]["type"] = "bar"
        assert any("draws a hole" in violation for violation in check_spec(spec))

    def test_a_value_label_beside_a_colour_encoding_is_refused(self):
        """A colour encoding is the only way a mark's fill reaches past slot 1 today.

        **Layered, because that is the only shape this can take.** A value label is
        always its OWN layer — `value_label_layers` emits a text layer encoding y/x/text
        and nothing else — so the colour encoding that would push a fill past slot 1 is
        on a sibling. A check that read `color` off the text mark's own encoding could
        only fire on a spec the compiler cannot produce, which is a guard that passes
        everything it was written to catch.
        """
        spec = {
            "data": {"values": [{"c": "a", "v": 1}]},
            "layer": [
                _bar()
                | {
                    "encoding": _bar()["encoding"]
                    | {"color": {"field": "c", "type": "nominal", "scale": {"domain": ["a", "b"]}, "legend": None}}
                },
                self._labelled(),
            ],
        }
        assert any("measured against slot 1 only" in violation for violation in check_spec(spec))

    def test_a_value_label_in_an_uncoloured_frame_is_not_refused(self):
        """The other direction, so the frame-scoped read cannot simply refuse everything."""
        spec = {"data": {"values": [{"c": "a", "v": 1}]}, "layer": [_bar(), self._labelled()]}
        assert not any("measured against slot 1 only" in violation for violation in check_spec(spec))

    # No "every arm still compiles" case here: `compile_chart` calls `enforce_spec` on
    # its own output, so every compiler test in `test_vega_compiler.py` already proves
    # this gate refuses nothing the compiler emits. A copy would be coverage theatre.


class TestPaletteDiscipline:
    def _coloured(self, domain, *, labelled=True):
        """A categorical colour encoding, direct-labelled unless asked otherwise.

        Direct-labelled by default because that is what the direct-labelling rule
        REQUIRES of anything reaching the second tier — a palette that draws past
        validated separation is only honest behind a label on every mark — so a
        fixture without one is testing the safeguard rather than the palette.
        """
        spec = _bar()
        spec["encoding"]["color"] = {"field": "c", "type": "nominal", "scale": {"domain": domain}, "legend": None}
        if labelled:
            spec["encoding"]["y"] = {"field": "c", "type": "nominal"}
        return spec

    def test_a_categorical_colour_without_an_explicit_domain_is_rejected(self):
        """Left to infer, the same category takes a different slot per chart."""
        spec = _bar()
        spec["encoding"]["color"] = {"field": "c", "type": "nominal"}
        assert any("no explicit scale.domain" in violation for violation in check_spec(spec))

    def test_a_domain_within_the_validated_slots_passes(self):
        assert check_spec(self._coloured(["a", "b", "c", "d"])) == []

    def test_a_domain_past_the_validated_slots_still_draws(self):
        """The palette never refuses to render legitimate data.

        This previously asserted the opposite — a fifth category was a violation — under
        an earlier palette rule. That rule was reversed:
        the number of series is a property of the data, so a five-series finding takes the
        second tier rather than losing its chart. The refusal that remains is for a
        palette that cannot say WHICH slot a category took, which is a different defect and
        is still tested above.
        """
        assert check_spec(self._coloured(["a", "b", "c", "d", "e"])) == []

    def test_a_domain_past_the_whole_palette_still_draws(self):
        """Past eight, Vega recycles — which is warned about, never refused.

        The recycling is real: series nine takes series one's hue. Refusing to draw is the
        one response that is not available, so the honest outcome is a drawn chart plus a
        warning naming the re-encoding that would have been better.
        """
        assert check_spec(self._coloured([f"c{index}" for index in range(11)])) == []

    def test_recycling_past_the_palette_is_warned_about(self, caplog):
        """The warning is the whole safeguard, so its absence has to fail something.

        Without it a nine-series chart renders perfectly and silently repaints one category
        as another — the exact failure the old refusal was reaching for. Drawing the data is
        not the same as pretending the palette was wide enough for it.
        """
        with caplog.at_level(logging.WARNING, logger="threetears.evals.vega.spec_policy"):
            check_spec(self._coloured([f"c{index}" for index in range(9)]))
        assert any("recycles from slot 1" in record.getMessage() for record in caplog.records)

    def test_a_domain_inside_the_palette_is_not_warned_about(self, caplog):
        """The negative half, so the assertion above is known to discriminate."""
        with caplog.at_level(logging.WARNING, logger="threetears.evals.vega.spec_policy"):
            check_spec(self._coloured(["a", "b", "c", "d", "e"]))
        assert not [record for record in caplog.records if "recycles" in record.getMessage()]

    def test_a_scheme_is_rejected(self):
        spec = _bar()
        spec["encoding"]["color"] = {"field": "c", "type": "nominal", "scale": {"domain": ["a"], "scheme": "viridis"}}
        assert any("scheme" in violation for violation in check_spec(spec))


class TestSharedAxis:
    def test_independent_scale_resolution_is_rejected(self):
        spec = _bar(resolve={"scale": {"y": "independent"}})
        assert any("resolved independently" in violation for violation in check_spec(spec))

    def test_shared_resolution_passes(self):
        assert check_spec(_bar(resolve={"scale": {"y": "shared"}})) == []

    def test_independent_resolution_on_an_inner_view_is_rejected(self):
        """Read at the root only, this passed the very rule it breaks.

        Two layers of one panel drawn to different x rulers is the one-unit-per-quantity
        violation exactly; that the panel sits inside a `vconcat` changes nothing about what
        the reader sees, and it is the shape a generator-emitted spec produces.
        """
        spec = {
            "vconcat": [
                {
                    "resolve": {"scale": {"x": "independent"}},
                    "layer": [_panel("latency (ms)"), _panel("latency (ms)")],
                }
            ]
        }
        violations = check_spec(spec)
        assert any("resolved independently" in violation for violation in violations)
        assert any("vconcat[0]" in violation for violation in violations), violations

    def test_independent_resolution_on_a_facet_is_rejected(self):
        """Cells drawn to their own rulers are small multiples that cannot be compared."""
        spec = {
            "resolve": {"scale": {"x": "independent"}},
            "facet": {"row": {"field": "c", "type": "nominal"}},
            "spec": _panel("latency (ms)"),
        }
        assert any("resolved independently" in violation for violation in check_spec(spec))

    def test_independent_resolution_between_concatenated_panels_passes(self):
        """Concatenated panels are separate frames — independent is Vega-Lite's own default.

        For the POSITIONAL channels, which is what this case pins. Refusing it
        would contradict the permission the title rule grants two panels measuring
        two quantities: that figure needs its x scales apart.
        """
        spec = {"resolve": {"scale": {"x": "independent"}}, "vconcat": [_panel("latency (ms)"), _panel("cost (usd)")]}
        assert check_spec(spec) == []

    def test_independent_colour_between_concatenated_panels_is_rejected(self):
        """The other half of the concat exemption, and the half it must not cover.

        `defaultScaleResolve` returns `independent` for a concat model only where
        `isXorY(channel) || theta || radius`; `color` falls through to `shared`.
        So an independent colour at a concat root is the generator drawing one
        category in two hues across panels — the stable-colour failure by another route, and
        the exemption written channel-blind let it straight through.
        """
        spec = {
            "resolve": {"scale": {"color": "independent"}},
            "vconcat": [_panel("latency (ms)"), _panel("cost (usd)")],
        }
        violations = check_spec(spec)
        assert any("scale for color is resolved independently" in violation for violation in violations), violations
        assert any("different colour" in violation for violation in violations), violations

    def test_independent_positional_resolution_on_a_repeat_passes(self):
        """A repeat normalises INTO a concat, so the concat default is the repeat default.

        `mapNonLayerRepeat` returns `{..., concat}`, so by the time scales resolve
        there is no repeat model left to have a default of its own.
        """
        spec = {
            "resolve": {"scale": {"x": "independent"}},
            "repeat": {"column": ["latency", "cost"]},
            "spec": _panel("latency (ms)"),
        }
        assert check_spec(spec) == []

    def test_independent_positional_resolution_on_a_repeat_over_an_array_passes(self):
        """The array form of `repeat` reaches `mapNonLayerRepeat` too."""
        spec = {
            "resolve": {"scale": {"y": "independent"}},
            "repeat": ["latency", "cost"],
            "spec": _panel("latency (ms)"),
        }
        assert check_spec(spec) == []

    def test_independent_colour_on_a_repeat_is_rejected(self):
        """The repeat exemption inherits the concat exemption's limit, not a wider one."""
        spec = {
            "resolve": {"scale": {"color": "independent"}},
            "repeat": {"column": ["latency", "cost"]},
            "spec": _panel("latency (ms)"),
        }
        assert any("scale for color is resolved independently" in violation for violation in check_spec(spec))

    def test_independent_resolution_on_a_layer_repeat_is_rejected(self):
        """A repeat over `layer` alone normalises into a LAYER, where shared is the default.

        `mapLayerRepeat` only delegates to the concat path when `row` or `column`
        is present; with neither it returns a `layer`, and two layers on different
        x rulers is the one-unit-per-quantity violation exactly.
        """
        spec = {
            "resolve": {"scale": {"x": "independent"}},
            "repeat": {"layer": ["latency", "cost"]},
            "spec": _panel("latency (ms)"),
        }
        assert any("resolved independently" in violation for violation in check_spec(spec))


class TestVariabilityCaption:
    def _interval(self, **overrides):
        spec = {
            "data": {"values": [{"c": "a", "lo": 1, "hi": 2}]},
            "mark": "errorbar",
            "encoding": {"x": {"field": "lo", "type": "quantitative", "axis": {"title": "score"}}},
        }
        spec.update(overrides)
        return spec

    def test_an_interval_with_no_description_is_rejected(self):
        assert any("naming what that span is" in violation for violation in check_spec(self._interval()))

    def test_an_interval_that_names_its_span_passes(self):
        assert check_spec(self._interval(description="Intervals span the 5 runs of each arm.")) == []

    def test_a_description_on_the_figure_covers_an_interval_drawn_in_a_panel(self):
        """One caption per figure, not one per panel — the shape the compiler emits."""
        spec = {"description": "Intervals span the 5 runs of each arm.", "vconcat": [self._interval()]}
        assert check_spec(spec) == []

    def test_an_interval_in_a_panel_with_no_description_anywhere_is_rejected(self):
        violations = check_spec({"vconcat": [self._interval()]})
        assert any("naming what that span is" in violation for violation in violations)
        assert any("vconcat[0]" in violation for violation in violations), violations


class TestRenderableColour:
    """The substrate's most expensive discovered hazard, at the universal gate."""

    def test_an_oklch_literal_anywhere_in_a_spec_is_rejected(self):
        spec = _bar()
        spec["encoding"]["x"]["axis"]["title"] = "latency (ms)"
        spec["mark"] = {"type": "bar", "color": "oklch(0.70 0.22 295)"}
        assert any("cannot parse" in violation for violation in check_spec(spec))

    def test_the_other_unparseable_notations_are_rejected_too(self):
        """resvg fails the same way on each; naming only oklch would be luck."""
        for notation in ("lab(50% 40 59)", "lch(50% 40 59)", "color-mix(in srgb, red, blue)"):
            spec = _bar(background=notation)
            assert any("cannot parse" in violation for violation in check_spec(spec)), notation

    def test_resolved_hex_passes(self):
        assert check_spec(_bar(background="#1a1a19")) == []

    def test_prose_that_merely_reads_like_a_colour_function_is_not_refused(self):
        """CSS attaches the bracket; a measure name does not.

        `lab (control)` is an ordinary thing for a generator to write and it is not
        a colour anyone can express.
        """
        assert check_spec(_bar(title="lab (control) vs lab (treatment) latency")) == []

    def test_a_title_that_mentions_a_colour_function_is_glyphs_not_a_colour(self):
        """The rule reads colour VALUES, never text that names one.

        A title is drawn as glyphs; the rasteriser never parses `oklch(` inside a
        sentence as a colour, so refusing the chart over it would be a check over
        prose. Both a one-line and a multi-line title are text.
        """
        assert check_spec(_bar(title="Bars were once oklch(0.70 0.22 295) in the source")) == []
        assert check_spec(_bar(title=["a chart filled", "oklch(0.70 0.22 295) draws black"])) == []
        assert check_spec(_bar(title="oklch(0.70 0.22 295) (observations)")) == []

    def test_a_nested_colour_mix_is_still_one_value(self):
        spec = _bar(background="color-mix(in srgb, oklch(0.70 0.22 295) 40%, #000)")
        assert any("cannot parse" in violation for violation in check_spec(spec))

    def test_a_colour_value_inside_a_list_is_still_read_on_its_own(self):
        """A scale range is a list of colour VALUES — each is read as the value it is."""
        spec = _bar()
        spec["encoding"]["color"] = {
            "field": "c",
            "type": "nominal",
            "scale": {"domain": ["a", "b"], "range": ["#1a1a19", "lab(50% 40 59)"]},
        }
        violations = check_spec(spec)
        assert any("cannot parse" in violation and "lab(50% 40 59)" in violation for violation in violations), (
            violations
        )

    def test_the_compiler_output_is_clean_of_it(self):
        """Not reachable from this compiler — which is exactly why the rule is here.

        The gate exists for producers that do not exist yet; a rule that only ever
        fires for one of those has to be written before they are.
        """
        from threetears.evals.vega import compile_chart

        payload = {"parts": [{"label": "a", "value": 2}, {"label": "b", "value": 1}], "unit": "runs"}
        assert check_spec(compile_chart("breakdown", payload).spec) == []


class TestResolvedSize:
    """A size left to the renderer is the other hazard that fails by drawing.

    Each renderer used to substitute a width of its own, and that substitution was
    where this rule lived. The compiler resolves every size now, so the
    substitution is gone — and these are what stop its removal from being a rule
    that holds only because of who is emitting specs today.
    """

    def test_a_responsive_width_is_rejected(self):
        assert any("leaves a size" in violation for violation in check_spec(_bar(width="container", height=168)))

    def test_a_responsive_width_nested_under_a_concat_is_rejected(self):
        """The case Vega-Lite does not support at all, so the one that matters most.

        Under `facet`/`concat` the child does not fill its parent — it falls back
        to Vega's own default and rasterises a valid, silently undersized PNG.
        """
        spec = {
            "vconcat": [
                _panel("latency (ms)", width="container", height=168),
                _panel("latency (ms)", width=832, height=60),
            ]
        }
        assert any("leaves a size" in violation for violation in check_spec(spec))

    def test_a_responsive_width_under_a_facet_spec_is_rejected(self):
        spec = {"facet": {"row": {"field": "c", "type": "nominal"}}, "spec": _panel("latency (ms)", width="container")}
        assert any("leaves a size" in violation for violation in check_spec(spec))

    def test_numeric_sizes_pass(self):
        assert check_spec(_bar(width=832, height=168)) == []

    def test_a_view_carrying_no_size_of_its_own_passes(self):
        """Presence is not the rule; resolution is.

        A layer inside a sized parent legitimately states no size, and several
        compiled shapes are exactly that. Demanding one everywhere would refuse
        charts that are correct.
        """
        assert check_spec({"width": 832, "height": 168, "layer": [_bar(), _bar()]}) == []

    def test_a_step_size_is_resolved_and_passes(self):
        """`{"step": N}` is a per-band size the renderer derives from the data.

        It needs no container measured, so it is resolved in the sense that
        matters. Refusing it would refuse legitimate Vega-Lite from exactly the
        generator-emitted producer this rule exists for — a gate worse than none.
        """
        assert check_spec(_bar(width={"step": 20}, height={"step": 28})) == []

    def test_a_data_field_named_width_is_not_a_view_dimension(self):
        """Inline rows are the chart's SUBJECT, not its layout.

        A payload plotting a column called `width` is ordinary data; reading it as
        an unresolved dimension would refuse a chart for drawing what it was asked
        to draw.
        """
        spec = _bar(width=832, height=168)
        spec["data"] = {"values": [{"c": "a", "width": "container"}, {"c": "b", "width": "wide"}]}
        assert check_spec(spec) == []

    def test_a_marks_own_height_is_not_a_views(self):
        """A mark may state a fraction of its band — a legitimate mark height.

        Reading it as an unresolved view size would refuse ordinary Vega-Lite from the
        generator-emitted producers this rule exists for, which is the tell that the
        walk must stop at `mark`. No arm here compiles the band form today — the
        dumbbell's connector states px, so that a row count cannot reach the mark — and
        the gate is written for the producers rather than for this compiler.
        """
        spec = _bar(width=832, height=168)
        spec["mark"] = {"type": "bar", "height": {"band": 0.18}}
        assert check_spec(spec) == []

    def test_the_compiler_output_is_clean_of_it(self):
        """Not reachable from this compiler — which is exactly why the rule is here.

        The gate exists for producers that do not exist yet, and the deleted
        substitution is what used to stand in for it.
        """
        from threetears.evals.vega import compile_chart

        payload = {"parts": [{"label": "a", "value": 2}, {"label": "b", "value": 1}], "unit": "runs"}
        assert check_spec(compile_chart("breakdown", payload).spec) == []


class TestProseBesideTheChart:
    """Prose shown beside a chart is not the gate's input.

    A caption or disclosure line is served as text and never reaches the
    rasteriser, and what it says is the reporter eval's question — so
    `enforce_spec` takes the spec alone, and there is no way to hand it prose.
    """

    def test_enforce_spec_accepts_no_prose(self):
        with pytest.raises(TypeError):
            enforce_spec(_bar(), "The band is lab(50% 40 59).")


class TestErrorReporting:
    def test_every_violation_is_reported_in_one_pass(self):
        """A producer fixing a spec wants the whole list, not the first entry."""
        spec = _bar(resolve={"scale": {"y": "independent"}})
        spec["encoding"]["x"]["scale"] = {"zero": False}
        spec["encoding"]["x"]["axis"] = {"title": ""}
        with pytest.raises(SpecPolicyError) as caught:
            enforce_spec(spec)
        assert len(caught.value.violations) == 3

    def test_violations_reach_nested_layers(self):
        """A rule that only inspected the top level would pass this spec."""
        spec = {
            "layer": [
                {"mark": "line", "encoding": {}},
                {
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "v", "type": "quantitative", "scale": {"zero": False}, "axis": {"title": "ms"}}
                    },
                },
            ]
        }
        assert any("truncates" in violation for violation in check_spec(spec))


class TestIdentityDoesNotRideOnColourAlone:
    """The safeguard that licenses the second tier, checked rather than logged.

    The palette never refuses to draw: it may draw past the four validated hues
    precisely because a direct label on every mark becomes mandatory at five — one
    slot before validated separation ends. Both halves of that sentence are the
    rule; only the permissive half was ever mechanical.
    """

    def _coloured(self, count, *, labelled=False, legend=None, **overrides):
        spec = _bar(**overrides)
        spec["encoding"]["color"] = {
            "field": "c",
            "type": "nominal",
            "scale": {"domain": [f"s{index}" for index in range(count)]},
            "legend": legend,
        }
        if labelled:
            spec["encoding"]["y"] = {"field": "c", "type": "nominal"}
        else:
            spec["encoding"].pop("y", None)
        return spec

    def test_a_second_tier_chart_without_per_mark_labels_is_refused(self):
        violations = check_spec(self._coloured(5))
        assert any("validated hues" in violation for violation in violations)

    def test_the_same_chart_with_its_categories_on_an_axis_passes(self):
        """Which is how every chart this compiler emits identifies a category."""
        assert check_spec(self._coloured(5, labelled=True)) == []

    def test_a_text_mark_drawn_from_the_colour_field_is_a_direct_label(self):
        """The other form: a layer that writes each category's name beside its mark."""
        spec = self._coloured(5)
        labels = {
            "data": {"values": [{"c": "a"}]},
            "mark": "text",
            "encoding": {"text": {"field": "c", "type": "nominal"}},
        }
        assert check_spec({"layer": [spec, labels]}) == []

    def test_a_text_mark_drawn_from_some_other_field_is_not(self):
        """A caption in the frame is not a label on the mark."""
        spec = self._coloured(5)
        note = {
            "data": {"values": [{"note": "n"}]},
            "mark": "text",
            "encoding": {"text": {"field": "note", "type": "nominal"}},
        }
        assert any("validated hues" in violation for violation in check_spec({"layer": [spec, note]}))

    def test_the_mandate_holds_inside_a_facet_too(self):
        """A key at the top of a figure is still a colour the reader has to carry."""
        spec = {"facet": {"row": {"field": "g", "type": "nominal"}}, "spec": self._coloured(5)}
        assert any("validated hues" in violation for violation in check_spec(spec))

    def test_a_legend_below_the_validated_slots_is_still_refused_outside_a_facet(self):
        """A legend asks the reader to hold a colour in memory and walk back to it."""
        violations = check_spec(self._coloured(3, labelled=True, legend={"title": "model"}))
        assert any("legend" in violation for violation in violations)

    def test_a_legend_left_at_the_default_is_the_same_defect(self):
        """Absent is not suppression — Vega-Lite draws one."""
        spec = self._coloured(3, labelled=True)
        del spec["encoding"]["color"]["legend"]
        assert any("legend" in violation for violation in check_spec(spec))

    def test_a_chart_breaking_both_rules_is_told_about_both(self):
        """One pass, whole list — the two-attempt loop is what the error type refuses.

        Five categories with neither per-mark labels nor a suppressed legend breaks
        the mandate AND the legend rule. Reporting only the mandate means the
        producer adds labels, recompiles, and is then told about the legend, which
        is the round trip `SpecPolicyError` documents itself as existing to avoid.
        """
        violations = check_spec(self._coloured(5, legend={"title": "model"}))
        assert any("validated hues" in violation for violation in violations)
        assert any("legend" in violation for violation in violations)

    def test_a_faceted_chart_may_keep_its_legend(self):
        """One key genuinely serves every panel, which is the case that earns one."""
        cell = self._coloured(3, labelled=True)
        del cell["encoding"]["color"]["legend"]
        assert check_spec({"facet": {"row": _UPRIGHT_ROW}, "spec": cell}) == []


class TestNoTextIsRotated:
    def test_a_rotated_axis_label_is_refused(self):
        spec = _bar()
        spec["encoding"]["x"]["axis"]["labelAngle"] = -45
        assert any("labelAngle" in violation for violation in check_spec(spec))

    def test_an_axis_label_stated_upright_passes(self):
        spec = _bar()
        spec["encoding"]["x"]["axis"]["labelAngle"] = 0
        assert check_spec(spec) == []

    def test_a_y_axis_title_is_refused_because_vega_turns_it_by_default(self):
        """Presence, not angle: Vega-Lite's own default `titleAngle` for y is 270."""
        spec = _bar()
        spec["encoding"]["y"] = {"field": "v", "type": "quantitative", "axis": {"title": "observations"}}
        assert any("quarter turn" in violation for violation in check_spec(spec))

    def test_a_y_axis_title_the_spec_pins_upright_passes(self):
        spec = _bar()
        spec["encoding"]["y"] = {
            "field": "v",
            "type": "quantitative",
            "axis": {"title": "observations", "titleAngle": 0},
        }
        assert check_spec(spec) == []

    def test_a_rotated_text_mark_is_refused(self):
        spec = _bar(mark={"type": "text", "angle": 90})
        assert any("upright" in violation for violation in check_spec(spec))


class TestTheGridIsNeverDashed:
    def test_a_dashed_grid_is_refused(self):
        spec = _bar()
        spec["encoding"]["x"]["axis"]["gridDash"] = [4, 2]
        assert any("dashed grid" in violation for violation in check_spec(spec))

    def test_a_solid_grid_passes(self):
        spec = _bar()
        spec["encoding"]["x"]["axis"]["grid"] = True
        assert check_spec(spec) == []


class TestACroppedPositionAxisSaysSo:
    def _cropped(self, **overrides):
        """A cropped SPAN — a rule reaching from one value to another.

        A span rather than the point this used to use, because the rule was scoped
        to spans: a width is judged against another
        width, so a cropped axis multiplies every such comparison, while a point is
        read off the labelled ticks that already show where the axis starts.
        """
        spec = _bar(mark="rule", **overrides)
        spec["encoding"]["x"]["scale"] = {"domain": [10, 20]}
        spec["encoding"]["x2"] = {"field": "high", "type": "quantitative"}
        # A span owes a `description` naming what it spans, which is a different rule
        # with its own test; supplied so this fixture exercises the crop rule alone.
        spec["description"] = "across 5 runs"
        return spec

    def test_a_silent_crop_is_accepted(self):
        """**No crop is refused for silence.**

        The gate that demanded a footnote is gone, and this is the same assertion
        inverted rather than deleted — "the check no longer fires" is a claim worth
        holding, because the way this would regress is somebody restoring the rule
        without the sentence that justified it. A labelled axis has already said
        where it starts.

        What refuses a genuinely misleading crop is `_check_baseline`, which does not
        accept a disclosure at all: a LENGTH mark may not crop, full stop.
        """
        assert check_spec(self._cropped()) == []

    def test_a_cropped_point_owes_no_footnote(self):
        """A cropped POSITION axis is not a violation; its labelled ticks disclose it.

        Written when the rule was scoped to spare points and kept when the footnote
        was withdrawn from every mark, because the assertion it makes is the one that
        outlived both changes: a cropped point plot passes. What protects a reader is
        the outright refusal of a cropped LENGTH mark, asserted separately — this is
        the other side of that line, and without it nothing here would notice a check
        that started refusing positions too.
        """
        spec = _bar(mark="point")
        spec["encoding"]["x"]["scale"] = {"domain": [10, 20]}
        assert check_spec(spec) == []

    @pytest.mark.parametrize(
        "subtitle", [None, "Axis is cropped to the data and excludes zero."], ids=["bare", "footnoted"]
    )
    def test_the_withdrawal_is_unconditional_rather_than_footnote_shaped(self, subtitle):
        """A subtitle neither earns a crop nor is owed by one — both cases pass alike.

        Two tests stood here, each asserting that a crop carrying a footnote passed:
        one on the panel, one inherited from the figure enclosing it. Under the
        withdrawn gate that was the whole question; with the gate gone they could
        not fail, because nothing reads the subtitle any more — the `footnoted`
        signal it fed was computed on every view walk with no reader and has been
        removed with them.

        Consolidated into this, which CAN fail: the bare case goes red the moment
        anyone re-arms a footnote-conditional rule, which is the regression the
        deleted pair was reaching for and never actually held.
        """
        spec = self._cropped()
        if subtitle:
            spec["title"] = {"text": "latency", "subtitle": subtitle}
        assert check_spec(spec) == []

    def test_a_domain_that_still_holds_zero_owes_no_footnote(self):
        """`zero: false` beside a domain containing zero crops nothing.

        Demanding a disclosure there would teach a producer to write one where
        there is nothing to disclose, which is how a footnote stops being read.
        """
        spec = _bar(mark="point")
        spec["encoding"]["x"]["scale"] = {"domain": [-5, 20], "zero": False}
        assert check_spec(spec) == []

    def test_a_length_mark_is_refused_outright_rather_than_asked_for_a_footnote(self):
        """A truncated bar misstates a ratio; no disclosure makes that readable."""
        spec = _bar()
        spec["encoding"]["x"]["scale"] = {"domain": [10, 20]}
        spec["title"] = {"text": "latency", "subtitle": "Axis is cropped to the data."}
        violations = check_spec(spec)
        assert any("excludes zero" in violation for violation in violations)
        assert not any("states no footnote" in violation for violation in violations)


class TestACroppedAxisKeepsItsTickLabels:
    """The labelled ticks are a crop's only disclosure, so a cropped axis may not suppress them (#636)."""

    def _point(self, domain, axis):
        spec = _bar(mark="point")
        spec["encoding"]["x"]["scale"] = {"domain": domain}
        spec["encoding"]["x"]["axis"] = axis
        return spec

    @pytest.mark.parametrize(
        "axis", [{"title": "latency (ms)", "labels": False}, None], ids=["labels-false", "no-axis"]
    )
    def test_a_cropped_unlabelled_position_axis_is_refused(self, axis):
        violations = check_spec(self._point([10, 20], axis))
        assert any("suppresses its tick labels" in violation for violation in violations), violations

    def test_a_cropped_labelled_axis_passes(self):
        assert check_spec(self._point([10, 20], {"title": "latency (ms)"})) == []

    def test_a_zero_based_unlabelled_axis_passes(self):
        """The distribution marginal's rise axis: `[0, cell]`, shape rather than counts."""
        violations = check_spec(self._point([0, 20], {"title": "latency (ms)", "labels": False}))
        assert not any("suppresses its tick labels" in violation for violation in violations)

    def test_zero_false_without_a_domain_counts_as_a_crop(self):
        spec = _bar(mark="point")
        spec["encoding"]["x"]["scale"] = {"zero": False}
        spec["encoding"]["x"]["axis"] = {"title": "latency (ms)", "labels": False}
        assert any("suppresses its tick labels" in violation for violation in check_spec(spec))

    def test_a_layer_suppressing_a_duplicate_of_a_labelled_axis_passes(self):
        """A layer frame shares one axis per channel; another layer labelling it discloses the crop."""
        labelled = self._point([10, 20], {"title": "latency (ms)"})
        quiet = self._point([10, 20], None)
        spec = {"layer": [labelled, quiet]}
        assert not any("suppresses its tick labels" in violation for violation in check_spec(spec))

    def test_the_rule_is_scoped_to_its_frame(self):
        """A concatenated panel's suppressed axis is caught, and its labelled sibling does not excuse it."""
        spec = {"hconcat": [self._point([10, 20], {"title": "latency (ms)"}), self._point([10, 20], None)]}
        violations = [v for v in check_spec(spec) if "suppresses its tick labels" in v]
        assert len(violations) == 1 and "hconcat[1]" in violations[0]


class TestAQuantityMayBeNamedAboveThePanel:
    def test_an_axis_that_suppresses_its_title_passes_under_a_heading(self):
        """The only way to name a y quantity without turning the words."""
        spec = _bar(title="observations")
        spec["encoding"]["x"]["axis"] = {"title": None}
        assert check_spec(spec) == []

    def test_an_axis_that_simply_omits_its_title_does_not(self):
        """Absent is not suppression: Vega-Lite fills it with the field name."""
        spec = _bar(title="observations")
        spec["encoding"]["x"]["axis"] = {}
        assert any("states no title" in violation for violation in check_spec(spec))

    def test_a_suppressed_title_with_no_heading_anywhere_is_still_refused(self):
        spec = _bar()
        spec["encoding"]["x"]["axis"] = {"title": None}
        assert any("states no title" in violation for violation in check_spec(spec))

    def test_a_figures_heading_does_not_stand_in_for_a_panels(self):
        """A concatenated panel is its own picture, and names its own quantity.

        Inheriting the figure's title here would license every suppressed axis in
        every panel under any heading at all, which is the whole check gone.
        """
        panel = _panel("latency (ms)")
        panel["encoding"]["x"]["axis"] = {"title": None}
        assert any("states no title" in violation for violation in check_spec({"title": "results", "vconcat": [panel]}))

    def test_a_facets_heading_does_stand_in_for_its_cells(self):
        """A facet node and the view it repeats are one panel with one heading."""
        cell = _bar()
        cell["encoding"]["x"]["axis"] = {"title": None}
        assert check_spec({"title": "observations", "facet": {"row": _UPRIGHT_ROW}, "spec": cell}) == []


class TestASmallMultiplesHeaderIsHeldToTheSameRuleAsAnAxis:
    """The case that draws sideways by default, and the two ways to miss it.

    Vega-Lite turns a ROW header a quarter turn by default and leaves a COLUMN
    header flat, so silence means opposite things on the two sides — and a check
    that only inspected headers a spec had bothered to write would have passed the
    very defect it was added for, since the defect is that the spec wrote nothing.
    """

    def _faceted(self, header=None, channel="row"):
        row = {"field": "g", "type": "nominal"}
        if header is not None:
            row["header"] = header
        return {"facet": {channel: row}, "spec": _bar()}

    def test_a_facet_that_states_no_header_at_all_is_refused(self):
        assert any("row header" in violation for violation in check_spec(self._faceted()))

    def test_a_header_that_states_everything_but_the_angle_is_refused(self):
        assert any("row header" in violation for violation in check_spec(self._faceted({"title": None})))

    def test_a_header_pinned_upright_passes(self):
        assert check_spec(self._faceted({"title": None, "labelAngle": 0})) == []

    def test_a_header_turned_on_purpose_is_refused_too(self):
        assert any("labelAngle" in violation for violation in check_spec(self._faceted({"labelAngle": 270})))

    def test_a_column_header_is_flat_by_default_and_owes_no_statement(self):
        assert check_spec(self._faceted(channel="column")) == []

    def test_the_encoding_shorthand_reaches_the_same_rule(self):
        """Two spellings, one drawing — a rule that read one would hold by accident."""
        spec = _bar()
        spec["encoding"]["row"] = {"field": "g", "type": "nominal"}
        assert any("row header" in violation for violation in check_spec(spec))

    def test_a_single_field_facet_is_read_as_the_row_it_wraps_to(self):
        spec = {"facet": {"field": "g", "type": "nominal"}, "spec": _bar()}
        assert any("row header" in violation for violation in check_spec(spec))


class TestAQuantitativeColourScaleIsHeldToTheSameRules:
    """The widening the colour permission was granted ON CONDITION of.

    The palette admits a hue for a level within a swept dimension, and states this
    gate's reach as a precondition: without it the permission would open a path
    around the gate it is written to respect. A gate reading only `nominal|ordinal` lets a producer reach the hue
    channel by declaring a type nothing looked at, which is the whole of what an
    unchecked encoding is.
    """

    @staticmethod
    def _ramp(**scale) -> dict:
        return {
            "mark": "rect",
            "encoding": {
                "x": {"field": "a", "type": "nominal"},
                "y": {"field": "b", "type": "nominal"},
                "color": {
                    "field": "rank",
                    "type": "quantitative",
                    "scale": {"domain": [0, 3], **scale},
                    "legend": None,
                },
            },
        }

    def test_a_conforming_ramp_passes(self):
        assert check_spec(self._ramp(range="chart-seq")) == []

    def test_a_ramp_with_no_domain_is_refused(self):
        """The endpoints would follow the data, so one level draws at two lightnesses."""
        spec = {
            "mark": "rect",
            "encoding": {
                "color": {"field": "rank", "type": "quantitative", "scale": {"range": "chart-seq"}, "legend": None}
            },
        }
        assert any("no explicit scale.domain" in violation for violation in check_spec(spec))

    def test_a_ramp_stating_its_own_colours_is_refused(self):
        """A range given as a list is a colour literal wearing a scale's clothes."""
        violations = check_spec(self._ramp(range=["#dacfff", "#613ea6"]))
        assert any("states its own range" in violation for violation in violations)

    def test_a_ramp_using_a_scheme_is_refused(self):
        assert any("uses a scheme" in violation for violation in check_spec(self._ramp(scheme="viridis")))

    def test_a_ramp_drawing_a_legend_is_refused(self):
        """A key asks the reader to hold a colour in memory whatever produced it."""
        spec = self._ramp(range="chart-seq")
        spec["encoding"]["color"].pop("legend")
        assert any("identifies its categories with a legend" in violation for violation in check_spec(spec))

    def test_a_ramp_is_not_held_to_the_slot_count(self):
        """The one rule that does not carry over, and its absence is the point.

        A categorical palette recycles past its slots, which is why a direct label
        becomes mandatory there. A ramp is a path Vega samples: a seventh level
        closes up against a sixth rather than landing on the first, so there is no
        count past which identity needs rescuing.
        """
        wide = self._ramp(range="chart-seq")
        wide["encoding"]["color"]["scale"]["domain"] = [0, 40]
        assert check_spec(wide) == []


class TestIndependentColourIsAdmittedOnlyWhereTheHarmCannotHappen:
    """The escape hatch's REFUSING branches, which is where its value is.

    `_check_shared_scales` used to refuse an independent colour scale on sight;
    it now reads the domains, because the harm it names — "one
    category takes a different colour from one panel or layer to the next" —
    cannot occur where no category appears twice. An escape hatch whose only
    exercised path is the one that lets things through is a hole with a docstring,
    so each condition that closes it is pinned here.
    """

    @staticmethod
    def _layered(*domains, key="layer"):
        return {
            key: [
                {
                    "mark": "rect",
                    # `legend: None` throughout: a legend is its own rule with its own
                    # test, and leaving it on would make every fixture here fail for a
                    # reason that has nothing to do with scale resolution.
                    "encoding": {
                        "color": {
                            "field": f"f{index}",
                            "type": "nominal",
                            "scale": {"domain": list(domain)},
                            "legend": None,
                        }
                    },
                }
                for index, domain in enumerate(domains)
            ],
            "resolve": {"scale": {"color": "independent"}},
        }

    def test_disjoint_vocabularies_are_admitted(self):
        """A concurrency rank and a model name are not values of one thing."""
        assert check_spec(self._layered(["1", "2"], ["gpt-5", "glm-5.2"])) == []

    def test_a_category_in_two_children_is_refused(self):
        """The harm the rule is actually about, reached through the new path."""
        violations = check_spec(self._layered(["a", "b"], ["b", "c"]))
        assert any("resolved independently" in violation for violation in violations)

    def test_an_unstated_domain_is_refused(self):
        """It follows the DATA, so nothing can rule out a category a sibling draws."""
        spec = self._layered(["a", "b"])
        spec["layer"].append({"mark": "rect", "encoding": {"color": {"field": "x", "type": "nominal", "legend": None}}})
        assert any("resolved independently" in violation for violation in check_spec(spec))

    def test_a_lone_child_is_refused(self):
        """One child cannot disagree with a sibling it does not have.

        Declaring independence there says nothing about what a second one would do,
        so the declaration is admitted on no evidence — which is the shape of every
        hole this function was written to avoid being.
        """
        assert any("resolved independently" in violation for violation in check_spec(self._layered(["a", "b"])))

    def test_a_repeat_gets_no_escape_at_all(self):
        """It draws ONE encoding once per panel, so every panel's domain is the same one.

        The collision is total, and it is invisible to a disjointness test precisely
        because there is a single child rather than several.
        """
        spec = {
            "repeat": ["a", "b"],
            "spec": {
                "mark": "rect",
                "encoding": {"color": {"field": "k", "type": "nominal", "scale": {"domain": ["x"]}, "legend": None}},
            },
            "resolve": {"scale": {"color": "independent"}},
        }
        assert any("resolved independently" in violation for violation in check_spec(spec))


class TestTheDisjointnessSetIsReadWithinAScaleAndNotAcross:
    """The rider that moved this gate's accept/refuse boundary.

    `_colours_disjoint_vocabularies` qualifies each domain entry by its scale's
    TYPE, because the vocabularies are only comparable within one: an ordered
    ramp's domain is `[0, n]` and a categorical lever may legitimately have a level
    spelled `"0"`. Compared as bare strings those read as one category and refuse a
    conforming chart — which is a barcode that will not draw for a campaign that
    swept `depth` at 0.

    The class above types every fixture `nominal`, so the qualifier is constant
    there and the change is invisible to it. These are the two cases that can see
    it, one on each side of the boundary.
    """

    @staticmethod
    def _mixed(level: str):
        """A ramp over ranks beside a categorical lever whose level is `level`."""
        return {
            "layer": [
                {
                    "mark": "rect",
                    "encoding": {
                        "color": {"field": "rank", "type": "quantitative", "scale": {"domain": [0, 3]}, "legend": None}
                    },
                },
                {
                    "mark": "rect",
                    "encoding": {
                        "color": {"field": "level", "type": "nominal", "scale": {"domain": [level]}, "legend": None}
                    },
                },
            ],
            "resolve": {"scale": {"color": "independent"}},
        }

    def test_a_categorical_level_spelled_like_a_rank_is_not_a_collision(self):
        """`"0"` the model name and 0 the rank are not the same category.

        Refusing this was refusing a barcode for a campaign that swept a lever at
        zero, which is an ordinary thing for a campaign to do.
        """
        assert check_spec(self._mixed("0")) == []

    def test_the_same_spec_with_a_non_numeric_level_is_admitted_too(self):
        """Non-vacuity: the case above must pass for the RIGHT reason.

        If the gate had simply stopped refusing anything, this would pass as well —
        so the pair is only evidence together with the refusal below.
        """
        assert check_spec(self._mixed("gpt-5")) == []

    def test_two_categorical_scales_sharing_a_level_are_still_refused(self):
        """The collision the rule exists for, which the qualifier must not swallow.

        Both scales are `nominal`, so the qualifier is the same on both sides and
        `"0"` really is one category drawn by two independently resolved scales.
        """
        spec = self._mixed("0")
        spec["layer"][0]["encoding"]["color"] = {
            "field": "other",
            "type": "nominal",
            "scale": {"domain": ["0"]},
            "legend": None,
        }
        assert any("resolved independently" in violation for violation in check_spec(spec))


#: Four configurations, ranked best-first, each carrying the lever it moved.
#:
#: `lever` runs A, B, A, B down the rows while `score` descends, which is what makes
#: the fixture discriminating: a rule that merely checked "the rows are grouped by
#: something" would pass a dimension sort here, and one that checked the measure
#: would not.
_RANKED_ROWS = [
    {"config": "d", "lever": "wide", "score": 0.77},
    {"config": "c", "lever": "narrow", "score": 0.66},
    {"config": "b", "lever": "wide", "score": 0.62},
    {"config": "a", "lever": "narrow", "score": 0.51},
]


def _ranking(rows=None, *, name="sweep_ranking", sort=True, **axis):
    """A conforming ranked-list spec: rows on y, one measure on x, best first."""
    rows = _RANKED_ROWS if rows is None else rows
    identity = {"field": "config", "type": "nominal", "axis": {"title": None}}
    if sort:
        identity["sort"] = [row["config"] for row in rows]
    spec = {
        "data": {"values": rows},
        "mark": {"type": "point"},
        "encoding": {
            "y": identity,
            "x": {
                "field": "score",
                "type": "quantitative",
                "scale": {"domain": [0, 1], "zero": True, **axis},
                "axis": {"title": "pass^k"},
            },
        },
    }
    if name is not None:
        spec["name"] = name
    return spec


class TestARankingIsOrderedByItsMeasure:
    """The rule the sweep barcode rests on, and the one a picture cannot show.

    A block down one lever's column means that lever drove the ranking — unless the
    rows were sorted by that lever, in which case the block is an artifact of the
    sort. The two draw the SAME MARKS in a different sequence, so nothing in the
    geometry distinguishes them; what does is the spec's own statement of its order,
    read back against the numbers it plots.
    """

    def test_a_ranking_ordered_by_its_measure_passes(self):
        assert check_spec(_ranking()) == []

    def test_a_ranking_ordered_by_a_dimension_is_refused(self):
        """The defect, introduced deliberately: group the rows by `lever` instead.

        Every value is the one the conforming case draws and every mark lands where
        it did — only the sequence changes, which is exactly why no other rule here
        catches it.
        """
        by_lever = sorted(_RANKED_ROWS, key=lambda row: row["lever"])
        violations = check_spec(_ranking(by_lever))
        assert any("matches none of the quantities it draws" in violation for violation in violations)

    def test_a_ranking_drawn_ascending_is_refused(self):
        """Rank 1 belongs at the top; ascending puts the worst configuration there."""
        violations = check_spec(_ranking(list(reversed(_RANKED_ROWS))))
        assert any("matches none of the quantities it draws" in violation for violation in violations)

    def test_a_ranking_that_states_no_order_is_refused(self):
        """An unstated order is the renderer's, and a ranking decided downstream ranks nothing."""
        violations = check_spec(_ranking(sort=False))
        assert any("states no explicit row order" in violation for violation in violations)

    def test_a_ranking_that_reverses_its_value_scale_is_refused(self):
        """A reversed axis turns a descending order back into an ascending picture."""
        violations = check_spec(_ranking(reverse=True))
        assert any("reverses its value scale" in violation for violation in violations)

    def test_a_figure_making_no_ranking_claim_is_not_held_to_it(self):
        """Non-vacuity from the other side, and the reason the claim is read at all.

        `delta_table` orders its rows by the payload's own sequence and plots one
        quantity against them, which is the exact shape refused above — and is
        correct for it, because a comparison of metrics is not a ranking. Only a
        spec that says it is one is judged as one.
        """
        by_lever = sorted(_RANKED_ROWS, key=lambda row: row["lever"])
        assert check_spec(_ranking(by_lever, name=None)) == []

    def test_a_ranking_with_no_quantity_placed_is_not_a_disordered_one(self):
        """The shape an empty constrained slice compiles to: the figure is a sentence.

        There is no measure on the frame, so there is no order to be wrong about —
        and a rule that read "no monotone quantity" as "sorted by a dimension" would
        refuse a chart for having nothing to sort.
        """
        spec = _ranking()
        spec["encoding"].pop("x")
        spec["encoding"]["text"] = {"field": "config", "type": "nominal"}
        spec["mark"] = {"type": "text"}
        assert check_spec(spec) == []

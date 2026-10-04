"""The chart layer's contract, driven by payloads that name no host.

Every payload here describes an invoice extractor — models, chunk sizes, parse and total
latencies, field accuracy — so nothing in the input belongs to a
conversational host. What is pinned is the compiler's public behaviour on the chart types a toy-host capture
never draws (``attribution``, ``breakdown``, ``null_result``, ``sweep_ranking``) and the pieces
every type shares:

* the author's caption is served exactly as written, and what the compiler has to add travels
  as ``disclosures``, one idea per line, never joined onto the caption;
* the wire shape (:class:`~threetears.evals.analysis.viz.models.FindingChart`) carries those lines apart;
* every line of prose beside a chart is held to the rendering rule, one line at a time;
* what the prose SAYS is not refused — a significance word in a mechanism draws;
* one unit per quantity, chosen by the public ``display_scale``;
* numbers are spelled by the shared number rule, never with an exponent past a thousand.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis.viz.compiler import compile_chart, display_scale
from threetears.evals.analysis.viz.models import FindingChart
from threetears.evals.analysis.viz.payloads import PayloadError, describe_validation


#: A colour notation the server-side rasteriser cannot parse — a rendering fault wherever it sits.
_UNRENDERABLE = "lab(50% 40 59)"

#: A quantified remainder: the parse stage is a declared component of the total.
ATTRIBUTION = {
    "end_to_end": {"measure": "total_ms", "delta": -3000.0, "a": 9000.0, "b": 6000.0, "n": 8},
    "subsystem": {"measure": "parse_ms", "delta": -1000.0, "n": 8},
    "unit": "ms",
    "contained_by": "total_ms",
    "unattributed_delta": -2000.0,
    "lever": "extractor_model",
    "a_label": "extractor-a",
    "b_label": "extractor-b",
}

#: The same pair with its remainder withheld, and the reason carried in the payload's own words.
ATTRIBUTION_WITHHELD = {
    **{key: value for key, value in ATTRIBUTION.items() if key not in ("contained_by", "unattributed_delta")},
    "unattributed_withheld": "parse_ms is not declared a component of total_ms, so no remainder is stated.",
}

BREAKDOWN = {
    "parts": [
        {"label": "header", "value": 30.0, "n": 3},
        {"label": "line_items", "value": 50.0, "n": 5},
        {"label": "totals", "value": 20.0, "n": 2},
    ],
    "unit": "%",
    "measure": "share of field errors",
    "total": 100.0,
    "total_n": 10,
}

NULL_RESULT = {
    "groups": [
        {
            "label": "chunk_size=512",
            "ci": {"low": 0.70, "high": 0.84, "mean": 0.77, "level": 0.95, "variability": "across the 6 invoices"},
            "n": 6,
        },
        {
            "label": "chunk_size=1024",
            "ci": {"low": 0.72, "high": 0.86, "mean": 0.79, "level": 0.95, "variability": "across the 6 invoices"},
            "n": 6,
        },
    ],
    "metric": "field_accuracy",
    "mechanism": "Every invoice fits in one chunk at either size.",
}

SWEEP_RANKING = {
    "ranked": {"measure": "field_accuracy", "unit": None},
    "secondary": {"measure": "cost per invoice", "unit": "usd"},
    "rows": [
        {
            "config": {"extractor_model": "extractor-a", "chunk_size": "1024"},
            "ranked_value": 0.9,
            "secondary_value": 0.012,
            "n": 6,
        },
        {
            "config": {"extractor_model": "extractor-b", "chunk_size": "512"},
            "ranked_value": 0.8,
            "secondary_value": 0.009,
            "n": 6,
        },
        {
            "config": {"extractor_model": "extractor-a", "chunk_size": "256"},
            "ranked_value": 0.7,
            "secondary_value": 0.015,
            "n": 6,
        },
    ],
}

FRONTIER = {
    "points": [
        {"label": "extractor-a", "cost": 0.01, "quality": 0.9, "latency_ms": 900.0},
        {"label": "extractor-b", "cost": 0.02, "quality": 0.5, "latency_ms": 1200.0, "dominated": True},
    ],
    "bar": 0.123456,
    "cost_label": "Cost per invoice (USD)",
    "quality_label": "field accuracy",
}

#: The four types no toy-host capture draws, each with a payload that leaves the compiler something
#: to disclose — so a caption left bare is distinguishable from one the compiler extended.
UNDRAWN_BY_THE_TOY_CAPTURE = {
    "attribution": ATTRIBUTION,
    "breakdown": BREAKDOWN,
    "null_result": NULL_RESULT,
    "sweep_ranking": SWEEP_RANKING,
}

AUTHOR = "Extractor-b is cheaper and no worse on the fields that matter"


class TestTheCaptionIsTheAuthorsLine:
    """The compiler never extends the author's sentence; its own words are separate lines."""

    @pytest.mark.parametrize("viz_type", sorted(UNDRAWN_BY_THE_TOY_CAPTURE))
    def test_the_caption_is_served_exactly_as_written(self, viz_type):
        chart = compile_chart(viz_type, {**UNDRAWN_BY_THE_TOY_CAPTURE[viz_type], "caption": AUTHOR})
        assert chart.disclosures, "the payload must leave the compiler something to disclose, or this proves nothing"
        assert chart.caption == AUTHOR

    @pytest.mark.parametrize("viz_type", sorted(UNDRAWN_BY_THE_TOY_CAPTURE))
    def test_no_caption_is_served_empty_with_the_disclosures_still_standing(self, viz_type):
        chart = compile_chart(viz_type, UNDRAWN_BY_THE_TOY_CAPTURE[viz_type])
        assert chart.caption == ""
        assert chart.disclosures

    @pytest.mark.parametrize("viz_type", sorted(UNDRAWN_BY_THE_TOY_CAPTURE))
    def test_the_author_cannot_reword_a_disclosure(self, viz_type):
        payload = UNDRAWN_BY_THE_TOY_CAPTURE[viz_type]
        assert (
            compile_chart(viz_type, {**payload, "caption": AUTHOR}).disclosures
            == compile_chart(viz_type, payload).disclosures
        )


class TestEachArmDisclosesOneIdeaPerLine:
    """Each arm's disclosures, in reading order, as separate lines."""

    def test_attribution_names_the_comparison_then_the_earned_remainder(self):
        assert compile_chart("attribution", ATTRIBUTION).disclosures == [
            "Comparing extractor_model: extractor-a → extractor-b.",
            (
                "Unattributed is total_ms minus parse_ms, which the measure catalog declared a component of it "
                "when this was generated."
            ),
        ]

    def test_attribution_carries_a_withheld_remainder_verbatim_as_its_own_line(self):
        assert compile_chart("attribution", ATTRIBUTION_WITHHELD).disclosures == [
            "Comparing extractor_model: extractor-a → extractor-b.",
            ATTRIBUTION_WITHHELD["unattributed_withheld"],
        ]

    def test_breakdown_states_the_whole_the_parts_divide(self):
        assert compile_chart("breakdown", BREAKDOWN).disclosures == ["The parts divide a total of 100% over n=10."]

    def test_breakdown_with_no_whole_discloses_nothing(self):
        payload = {key: value for key, value in BREAKDOWN.items() if key not in ("total", "total_n")}
        assert compile_chart("breakdown", payload).disclosures == []

    def test_null_result_states_geometry_coverage_span_then_the_mechanism(self):
        assert compile_chart("null_result", NULL_RESULT).disclosures == [
            "Intervals overlap on [0.72, 0.84]. Overlap alone does not establish a null.",
            "Intervals are 95% CIs.",
            "Intervals span across the 6 invoices.",
            NULL_RESULT["mechanism"],
        ]

    def test_sweep_ranking_names_columns_spread_ramp_and_inference_apart(self):
        assert compile_chart("sweep_ranking", SWEEP_RANKING).disclosures == [
            "Columns, left to right: chunk_size, extractor_model.",
            "cost per invoice runs from 0.009 usd to 0.015 usd and is not held — the ranking is not controlled for it.",
            "chunk_size draws as a light-to-dark ramp.",
            "Whether chunk_size, extractor_model are ordered was inferred from the levels rather than declared.",
        ]


class TestTheWireShapeCarriesBothAuthorsApart:
    """``FindingChart.from_compiled`` serves the caption and the disclosures as two fields."""

    @pytest.mark.parametrize("viz_type", sorted(UNDRAWN_BY_THE_TOY_CAPTURE))
    def test_the_served_chart_keeps_the_caption_and_every_disclosure_line(self, viz_type):
        compiled = compile_chart(viz_type, {**UNDRAWN_BY_THE_TOY_CAPTURE[viz_type], "caption": AUTHOR})
        served = FindingChart.from_compiled("f-1", compiled)
        assert served.caption == AUTHOR
        assert served.disclosures == compiled.disclosures
        assert served.model_dump()["disclosures"] == compiled.disclosures

    def test_a_chart_with_nothing_to_disclose_serves_an_empty_list(self):
        payload = {key: value for key, value in BREAKDOWN.items() if key not in ("total", "total_n")}
        assert FindingChart.from_compiled("f-1", compile_chart("breakdown", payload)).disclosures == []


class TestProseBesideTheChartIsNotGated:
    """No line beside the chart is read by a check.

    The caption and disclosure lines are served as text and never reach the
    rasteriser, so a colour notation written in them is words, not a colour.
    """

    def test_an_unrenderable_colour_named_in_a_mechanism_compiles_and_is_carried_verbatim(self):
        mechanism = f"The overlap is the {_UNRENDERABLE} band."
        compiled = compile_chart("null_result", {**NULL_RESULT, "mechanism": mechanism})
        assert compiled.disclosures[-1] == mechanism

    def test_what_a_mechanism_claims_is_not_refused(self):
        """Code checks the structure of prose beside a chart, never what it says."""
        claim = "The two chunk sizes differ significantly on no invoice."
        assert compile_chart("null_result", {**NULL_RESULT, "mechanism": claim}).disclosures[-1] == claim


class TestOneRulerAndOneNumberRule:
    """Units chosen once per quantity, and numbers spelled one way."""

    @pytest.mark.parametrize(
        ("values", "unit", "scaled"),
        [([900.0, 1200.0], "ms", (0.001, "s")), ([450.0], "ms", (1.0, "ms")), ([5.0], None, (1.0, ""))],
    )
    def test_display_scale_picks_one_unit_for_every_value(self, values, unit, scaled):
        assert display_scale(values, unit) == scaled

    def test_a_value_past_a_thousand_is_written_whole_never_with_an_exponent(self):
        payload = {
            "parts": [{"label": "prompt_tokens", "value": 12345.6}, {"label": "output_tokens", "value": 800.0}],
            "unit": "tokens",
        }
        drawn = compile_chart("breakdown", payload).values_as_drawn()
        assert any(line.split()[-1] == "12346" for line in drawn), drawn
        assert not any("e+" in line for line in drawn), drawn

    def test_the_frontier_quality_bar_is_spelled_by_the_number_rule(self):
        assert "The quality bar is 0.1235." in compile_chart("frontier", FRONTIER).disclosures


class TestAMalformedPayloadNamesEveryOffendingField:
    """The public validation description names each field that failed."""

    def test_every_offending_field_is_named(self):
        with pytest.raises(PayloadError) as refused:
            compile_chart("breakdown", {"parts": [{"label": "", "value": "many"}], "unit": "%"})
        message = str(refused.value)
        assert "parts.0.label" in message and "parts.0.value" in message, message

    def test_the_description_is_reachable_on_its_own(self):
        from pydantic import BaseModel, ValidationError

        class _Invoice(BaseModel):
            total: float
            currency: str

        with pytest.raises(ValidationError) as failed:
            _Invoice.model_validate({"total": "lots"})
        described = describe_validation(failed.value)
        assert "total" in described and "currency" in described, described

"""The generate-time payload contract — what a malformed chart payload costs.

These are the checks that turn "a chart drew an empty box" into "the generation
was refused and said which field was wrong". The historical defect this replaces
is specific and reproduced below: payloads whose top-level key was a plausible
synonym for the one the contract wanted (``stop_causes`` for ``parts``,
``cells`` for ``points``), which no rule rejected because no rule existed.
"""

import pytest

from threetears.evals.analysis.viz.payloads import (
    ABSENT_LEVEL,
    PAYLOAD_MODELS,
    AttributionPayload,
    BreakdownPayload,
    PayloadError,
    infer_ordered,
    parse_payload,
)


#: The sentence the bundle's scope-divergence lens hands the generator for the pair
#: below — carried verbatim rather than paraphrased, because a paraphrase here would
#: test a string this repo never produces.
NOT_CONTAINED = (
    "pipeline_synthesis_ms is not declared a component of total_ms — they share a unit but are not "
    "known to be nested, so what looks like an unexplained remainder may be two disjoint stretches "
    "of the same measure."
)


def _attribution(**overrides):
    """The withheld shape: a ~90s synthesis swing against a near-flat total_ms, remainder withheld.

    This is the pair the type exists for, and the one whose remainder must never
    become a number.
    """
    base = {
        "end_to_end": {"measure": "total_ms", "delta": 3000.0, "a": 44000.0, "b": 47000.0},
        "subsystem": {"measure": "pipeline_synthesis_ms", "delta": 87800.0, "a": 5200.0, "b": 93000.0},
        "unit": "ms",
        "unattributed_withheld": NOT_CONTAINED,
        "lever": "pipeline.pipeline_model",
        "a_label": "model-a-3.5-fast-lite",
        "b_label": "model-b",
    }
    base.update(overrides)
    return base


def _payload(**overrides):
    """A conforming breakdown payload, with per-test overrides applied."""
    base = {
        "parts": [
            {"label": "budget_exhausted", "value": 42.0, "n": 21},
            {"label": "no_new_sources", "value": 27.0, "n": 13},
            {"label": "confidence_met", "value": 19.0, "n": 9},
            {"label": "tool_error", "value": 12.0, "n": 6},
        ],
        "unit": "%",
        "measure": "share of stops",
        "total": 100.0,
        "total_n": 49,
    }
    base.update(overrides)
    return base


class TestBreakdownPayloadAccepts:
    """The shapes a conforming generator emits."""

    def test_full_payload_parses(self):
        parsed = parse_payload("breakdown", _payload())
        assert isinstance(parsed, BreakdownPayload)
        assert [part.label for part in parsed.parts] == [
            "budget_exhausted",
            "no_new_sources",
            "confidence_met",
            "tool_error",
        ]

    def test_optional_fields_may_be_absent(self):
        """Only `parts` and `unit` are required; a breakdown may not know its whole."""
        parsed = parse_payload(
            "breakdown", {"parts": [{"label": "a", "value": 1}, {"label": "b", "value": 2}], "unit": "runs"}
        )
        assert parsed.total is None
        assert parsed.measure is None

    def test_a_zero_part_is_legitimate(self):
        """A cause that never fired is a real part of the breakdown, not an absence."""
        parsed = parse_payload(
            "breakdown", _payload(parts=[{"label": "a", "value": 100.0}, {"label": "b", "value": 0.0}], total=100.0)
        )
        assert parsed.parts[1].value == 0.0

    def test_rounded_shares_need_not_resum_exactly(self):
        """Eight parts rounded to one decimal will not re-sum to exactly 100."""
        parts = [{"label": f"p{i}", "value": 12.5} for i in range(8)]
        parts[0]["value"] = 12.4
        parse_payload("breakdown", _payload(parts=parts, total=100.0))


class TestBreakdownPayloadRejects:
    """Every rejection names the offending field — the caller is deciding whether to pay again."""

    def test_the_historical_wrong_key_defect(self):
        """The exact served failure: a plausible synonym for the contract's key.

        Both halves must be named — the key that is missing AND the key that was
        sent instead. Naming only one leaves the reader guessing at a rename.
        """
        with pytest.raises(PayloadError) as caught:
            parse_payload("breakdown", {"stop_causes": [{"label": "a", "value": 1}], "unit": "%"})
        message = str(caught.value)
        assert "parts" in message
        assert "stop_causes" in message

    def test_missing_unit_is_rejected(self):
        """The unit is data, so a payload without one cannot be drawn."""
        payload = _payload()
        del payload["unit"]
        with pytest.raises(PayloadError, match="unit"):
            parse_payload("breakdown", payload)

    def test_empty_unit_is_rejected(self):
        with pytest.raises(PayloadError, match="unit"):
            parse_payload("breakdown", _payload(unit=""))

    def test_single_part_is_rejected(self):
        with pytest.raises(PayloadError, match="at least 2 parts"):
            parse_payload("breakdown", _payload(parts=[{"label": "only", "value": 100.0}], total=100.0))

    def test_duplicate_labels_are_rejected(self):
        """Two bars with one name make the sort order decide which is which."""
        with pytest.raises(PayloadError, match="unique"):
            parse_payload("breakdown", _payload(parts=[{"label": "a", "value": 50.0}, {"label": "a", "value": 50.0}]))

    def test_negative_value_is_rejected(self):
        with pytest.raises(PayloadError, match="negative"):
            parse_payload("breakdown", _payload(parts=[{"label": "a", "value": -1.0}, {"label": "b", "value": 101.0}]))

    def test_parts_that_do_not_reach_the_total_are_rejected(self):
        """A dropped part would otherwise draw a partial division as a complete one."""
        with pytest.raises(PayloadError, match="does not match the parts"):
            parse_payload(
                "breakdown", _payload(parts=[{"label": "a", "value": 42.0}, {"label": "b", "value": 27.0}], total=100.0)
            )

    def test_parts_that_overshoot_the_total_are_rejected(self):
        with pytest.raises(PayloadError, match="does not match the parts"):
            parse_payload(
                "breakdown", _payload(parts=[{"label": "a", "value": 80.0}, {"label": "b", "value": 80.0}], total=100.0)
            )

    def test_unknown_extra_key_is_rejected(self):
        with pytest.raises(PayloadError, match="not permitted"):
            parse_payload("breakdown", _payload(colour="red"))


class TestRegistry:
    """An unregistered type is unchecked, and must be distinguishable from a valid one."""

    def test_unregistered_type_returns_none_rather_than_raising(self):
        """`None` is 'nothing checked this', which a caller must not read as 'valid'.

        `scatter` is the example because no model is registered for it. Every type the stored
        model declares now has one (``timeseries``, its predecessor here, was the last), so the
        example is a name nothing declares — picking a registered type would make the test pass for
        the wrong reason, which is exactly what happened to its predecessors.
        """
        assert parse_payload("scatter", {"anything": True}) is None
        assert "scatter" not in PAYLOAD_MODELS

    def test_registry_lists_only_types_the_viz_model_declares(self):
        """A payload model for a type the stored model cannot hold would never run."""
        from threetears.evals.contracts.campaign import Viz

        declared = set(Viz.model_fields["type"].annotation.__args__)
        assert set(PAYLOAD_MODELS) <= declared

    def test_every_type_the_viz_model_declares_is_validated(self):
        """The other direction: a stored type with no model would be stored unchecked."""
        from threetears.evals.contracts.campaign import Viz

        assert set(Viz.model_fields["type"].annotation.__args__) <= set(PAYLOAD_MODELS)

    @pytest.mark.parametrize("viz_type", sorted(PAYLOAD_MODELS), ids=sorted(PAYLOAD_MODELS))
    def test_every_type_inherits_the_shared_caption_field(self, viz_type):
        """It is declared once on the base, so a type added later gets it for free.

        Read off the models rather than from a list here, so the assertion cannot
        fall behind the registry: a type registered without inheriting the base
        would fail this without anyone having to remember to extend a fixture.
        """
        field = PAYLOAD_MODELS[viz_type].model_fields.get("caption")
        assert field is not None, f"{viz_type} does not inherit the shared caption field"
        assert not field.is_required(), "a chart with no editorial line to give must still be emittable"

    @pytest.mark.parametrize("viz_type", sorted(PAYLOAD_MODELS), ids=sorted(PAYLOAD_MODELS))
    def test_widening_the_base_did_not_loosen_the_unknown_key_refusal(self, viz_type):
        """`extra="forbid"` is what catches `cells` for `points`; a new base field must not cost it.

        The near miss is a typo of the new name itself — `captions`, `caption_text` —
        which a loosened base would accept and silently never render.
        """
        with pytest.raises(PayloadError):
            parse_payload(viz_type, {"captions": "a typo for the field that exists"})


class TestDistributionRejections:
    """Every reject path of the `distribution` contract, which is the point of it.

    An accept-path fixture proves a validator does not fire on good input; it says
    nothing about whether the validator works. These are the assertions that go red
    when one is deleted.
    """

    @staticmethod
    def _payload(**overrides):
        base = {"groups": [{"label": "a", "samples": [1.0, 2.0]}], "unit": "s"}
        return {**base, **overrides}

    def test_a_group_carrying_no_spread_at_all_is_rejected(self):
        with pytest.raises(PayloadError, match="no spread to draw"):
            parse_payload("distribution", self._payload(groups=[{"label": "a"}]))

    def test_a_group_carrying_only_an_interval_is_accepted(self):
        """An interval alone is a legitimate shape — it gets an explicit 'shape unknown'."""
        assert parse_payload(
            "distribution",
            self._payload(
                groups=[
                    {
                        "label": "a",
                        "ci": {"low": 1, "high": 3, "mean": 2, "level": 0.95, "variability": "across 5 runs"},
                    }
                ]
            ),
        )

    def test_a_non_finite_sample_is_rejected(self):
        with pytest.raises(PayloadError, match="finite"):
            parse_payload("distribution", self._payload(groups=[{"label": "a", "samples": [1.0, float("nan")]}]))

    def test_an_empty_group_list_is_rejected(self):
        with pytest.raises(PayloadError, match="at least 1 group"):
            parse_payload("distribution", self._payload(groups=[]))

    def test_duplicate_group_labels_are_rejected(self):
        """Two groups sharing a name draw two rows a reader cannot tell apart.

        Two samples each, so the groups are otherwise VALID: with one apiece the
        spread-free rejection fires first and this asserts a passing test for the
        wrong reason.
        """
        with pytest.raises(PayloadError, match="unique"):
            parse_payload(
                "distribution",
                self._payload(groups=[{"label": "a", "samples": [1.0, 1.5]}, {"label": "a", "samples": [2.0, 2.5]}]),
            )

    def test_a_single_sample_with_no_interval_is_rejected(self):
        """A distribution type must not accept a group with nothing to spread.

        Two groups each reporting `n=4` with one sample and no interval draw a title,
        two labels and two hairlines clipped at opposite edges of a cropped axis — a
        chart indistinguishable from a failed render.
        """
        with pytest.raises(PayloadError, match="single sample"):
            parse_payload("distribution", self._payload(groups=[{"label": "a", "samples": [1.0], "n": 4}]))

    def test_a_single_sample_beside_an_interval_is_accepted(self):
        """The interval is the spread; the sample is one observation drawn on it."""
        payload = parse_payload(
            "distribution",
            self._payload(
                groups=[
                    {
                        "label": "a",
                        "samples": [1.0],
                        "ci": {"low": 0.5, "mean": 1.0, "high": 1.5, "level": 0.95, "variability": "across 5 runs"},
                    }
                ]
            ),
        )
        assert payload.groups[0].samples == [1.0]

    def test_two_samples_at_one_value_are_accepted(self):
        """Zero variance is a finding. Refusing it would invent a claim about the data."""
        payload = parse_payload("distribution", self._payload(groups=[{"label": "a", "samples": [4.0, 4.0]}]))
        assert payload.groups[0].samples == [4.0, 4.0]

    def test_an_extra_top_level_key_is_rejected(self):
        """The observed defect class: `stop_causes` where `groups` belongs."""
        with pytest.raises(PayloadError, match="not permitted"):
            parse_payload("distribution", self._payload(stop_causes={"a": 1}))


class TestConfidenceIntervalRejections:
    """An interval that cannot be drawn as a span around its estimate."""

    @staticmethod
    def _with_ci(ci):
        return {"groups": [{"label": "a", "ci": ci}], "unit": "s"}

    def test_bounds_the_wrong_way_round_are_rejected(self):
        with pytest.raises(PayloadError, match="wrong way round"):
            parse_payload(
                "distribution",
                self._with_ci({"low": 3.0, "high": 1.0, "mean": 2.0, "level": 0.95, "variability": "across 5 runs"}),
            )

    def test_a_mean_outside_its_own_interval_is_rejected(self):
        """Not a wide interval — two different quantities reported as one. Drawn, the
        estimate marker lands off the band and reads as a rendering bug."""
        with pytest.raises(PayloadError, match="lies outside"):
            parse_payload(
                "distribution",
                self._with_ci({"low": 1.0, "high": 2.0, "mean": 5.0, "level": 0.95, "variability": "across 5 runs"}),
            )

    def test_a_non_finite_bound_is_rejected(self):
        with pytest.raises(PayloadError, match="finite"):
            parse_payload(
                "distribution",
                self._with_ci(
                    {"low": float("-inf"), "high": 2.0, "mean": 1.0, "level": 0.95, "variability": "across 5 runs"}
                ),
            )

    @pytest.mark.parametrize("missing", ["level", "variability"])
    def test_an_interval_that_does_not_say_what_it_covers_or_spans_is_rejected(self, missing):
        """Code fills every interval and states both, so one missing either is not one the engine draws."""
        whole = {"low": 1.0, "high": 3.0, "mean": 2.0, "level": 0.95, "variability": "across 5 runs"}
        parse_payload("distribution", self._with_ci(whole))
        with pytest.raises(PayloadError, match=missing):
            parse_payload("distribution", self._with_ci({k: v for k, v in whole.items() if k != missing}))

    def test_a_coverage_level_outside_zero_to_one_is_rejected(self):
        """`95` is not a level — it is a percentage, and would render as '9500% CI'."""
        with pytest.raises(PayloadError, match="level"):
            parse_payload(
                "distribution",
                self._with_ci({"low": 1.0, "high": 3.0, "mean": 2.0, "level": 95, "variability": "across 5 runs"}),
            )


class TestNullResultRejections:
    def test_a_single_arm_is_rejected(self):
        with pytest.raises(PayloadError, match="at least 2 arms"):
            parse_payload(
                "null_result",
                {
                    "groups": [
                        {
                            "label": "a",
                            "ci": {"low": 1, "high": 3, "mean": 2, "level": 0.95, "variability": "across 5 runs"},
                        }
                    ]
                },
            )

    def test_an_arm_with_no_interval_is_rejected(self):
        """This type IS its two spreads; an arm without one has nothing to draw."""
        with pytest.raises(PayloadError, match="ci"):
            parse_payload("null_result", {"groups": [{"label": "a"}, {"label": "b"}]})

    def test_duplicate_arm_labels_are_rejected(self):
        payload = {
            "groups": [
                {
                    "label": "same",
                    "ci": {"low": 1, "high": 3, "mean": 2, "level": 0.95, "variability": "across 5 runs"},
                },
                {
                    "label": "same",
                    "ci": {"low": 1, "high": 3, "mean": 2, "level": 0.95, "variability": "across 5 runs"},
                },
            ]
        }
        with pytest.raises(PayloadError, match="unique"):
            parse_payload("null_result", payload)


class TestDeltaTableRejections:
    """Successors to the two deleted `DeltaTable` significance assertions.

    Those tests were written for the statistic-behind-the-claim rule and were
    deleted with the component. The rule did not go with them — it moved from a
    renderer that could only decline to DISPLAY a claim into a contract that
    refuses to generate one, which is strictly earlier and strictly stronger.
    """

    def test_an_empty_row_list_is_rejected(self):
        with pytest.raises(PayloadError, match="at least 1 row"):
            parse_payload("delta_table", {"rows": []})

    def test_duplicate_metrics_are_rejected(self):
        rows = [{"metric": "m", "a": 1, "b": 2}, {"metric": "m", "a": 3, "b": 4}]
        with pytest.raises(PayloadError, match="unique"):
            parse_payload("delta_table", {"rows": rows})

    def test_a_numeric_row_holding_a_string_is_rejected(self):
        """A numeric row is the only kind that can be POSITIONED, so a string in one
        of its sides is a row the chart would drop while the table still counted it."""
        with pytest.raises(PayloadError, match="is numeric but"):
            parse_payload("delta_table", {"rows": [{"metric": "m", "a": "fast", "b": 2}]})

    def test_a_numeric_row_holding_a_non_finite_value_is_rejected(self):
        with pytest.raises(PayloadError, match="non-finite"):
            parse_payload("delta_table", {"rows": [{"metric": "m", "a": float("inf"), "b": 2}]})

    def test_a_negative_significance_claim_with_no_statistic_is_demoted_to_untested(self):
        """ "Not significant" and "not tested" are different facts. `False` is the
        dangerous half — it asserts a test ran and came back negative.

        Demoted rather than refused because this model also reads STORED payloads,
        where a refusal discards a whole chart nothing can regenerate. A code-compiled
        table never states significance, so stored payloads are the only ones this reaches."""
        parsed = parse_payload("delta_table", {"rows": [{"metric": "m", "a": 1, "b": 2, "significant": False}]})
        assert parsed.rows[0].significant is None

    def test_a_positive_significance_claim_with_no_statistic_is_demoted_too(self):
        """The other half of the deleted pair: the claim is refused in both directions."""
        parsed = parse_payload("delta_table", {"rows": [{"metric": "m", "a": 1, "b": 2, "significant": True}]})
        assert parsed.rows[0].significant is None

    def test_a_significance_claim_backed_by_a_statistic_survives(self):
        """The rule must not swallow the honest case, or it is just a mute button."""
        parsed = parse_payload(
            "delta_table", {"rows": [{"metric": "m", "a": 1, "b": 2, "p": 0.01, "significant": True}]}
        )
        assert parsed.rows[0].significant is True


class TestAttributionAccepts:
    """The two shapes the divergence lens actually hands the generator."""

    def test_the_withheld_shape_parses(self):
        parsed = parse_payload("attribution", _attribution())
        assert isinstance(parsed, AttributionPayload)
        assert parsed.unattributed_delta is None
        assert parsed.unattributed_withheld == NOT_CONTAINED

    def test_a_remainder_backed_by_declared_containment_parses(self):
        """`tool_ms` IS declared a component of `total_ms`, so the subtraction is earned."""
        parsed = parse_payload(
            "attribution",
            _attribution(
                subsystem={"measure": "tool_ms", "delta": -3900.0},
                end_to_end={"measure": "total_ms", "delta": -33900.0},
                contained_by="total_ms",
                unattributed_delta=-30000.0,
                unattributed_withheld=None,
            ),
        )
        assert parsed.unattributed_delta == -30000.0

    def test_levels_are_optional_because_the_movement_is_the_subject(self):
        parsed = parse_payload(
            "attribution",
            _attribution(
                end_to_end={"measure": "total_ms", "delta": 3000.0}, subsystem={"measure": "judge_ms", "delta": 900.0}
            ),
        )
        assert parsed.end_to_end.a is None

    def test_a_zero_movement_is_a_real_movement(self):
        """The motivating case is a FLAT total_ms against a swinging phase — zero is the finding."""
        parsed = parse_payload(
            "attribution", _attribution(end_to_end={"measure": "total_ms", "delta": 0.0, "a": 81000.0, "b": 81000.0})
        )
        assert parsed.end_to_end.delta == 0.0


class TestAttributionRefusesTheSubtractionItCannotEarn:
    """Done-when: a payload implying a clean subtraction across non-nested populations."""

    def test_a_remainder_with_no_containment_declared_is_rejected(self):
        with pytest.raises(PayloadError, match="not declared a component of anything"):
            parse_payload("attribution", _attribution(unattributed_delta=-84800.0, unattributed_withheld=None))

    def test_a_remainder_naming_some_other_whole_is_rejected(self):
        """Containment must be to THIS chart's whole; a component of something else is not nested here."""
        with pytest.raises(PayloadError, match="declared a component of 'elapsed_ms'"):
            parse_payload(
                "attribution",
                _attribution(contained_by="elapsed_ms", unattributed_delta=-84800.0, unattributed_withheld=None),
            )

    def test_a_remainder_that_is_not_the_difference_it_claims_is_rejected(self):
        """A number that does not reconcile draws a bar contradicting the two beside it."""
        with pytest.raises(PayloadError, match="is not total_ms minus"):
            parse_payload(
                "attribution",
                _attribution(contained_by="total_ms", unattributed_delta=-50000.0, unattributed_withheld=None),
            )

    def test_stating_both_a_number_and_the_reason_it_cannot_be_stated_is_rejected(self):
        with pytest.raises(PayloadError, match="both set"):
            parse_payload("attribution", _attribution(contained_by="total_ms", unattributed_delta=84800.0))

    def test_stating_neither_is_rejected(self):
        """Two bars and no word about their difference invite the subtraction silently."""
        with pytest.raises(PayloadError, match="neither unattributed_delta nor unattributed_withheld"):
            parse_payload("attribution", _attribution(unattributed_withheld=None))

    def test_a_measure_cannot_diverge_from_itself(self):
        with pytest.raises(PayloadError, match="cannot diverge from itself"):
            parse_payload("attribution", _attribution(subsystem={"measure": "total_ms", "delta": 1.0}))

    def test_levels_contradicting_their_own_delta_are_rejected(self):
        with pytest.raises(PayloadError, match="states delta"):
            parse_payload(
                "attribution",
                _attribution(end_to_end={"measure": "total_ms", "delta": 3000.0, "a": 44000.0, "b": 99999.0}),
            )

    def test_an_extra_top_level_key_is_rejected(self):
        """The historical defect class: a plausible synonym nothing refused."""
        with pytest.raises(PayloadError, match="remainder"):
            parse_payload("attribution", _attribution(remainder=-84800.0))

    def test_a_non_finite_remainder_is_rejected(self):
        """NaN slipped every guard, and would have 500'd the whole charts response.

        Infinity was already caught by the reconciliation check; NaN was not,
        because every comparison against it is False — so `abs(nan - expected) >
        slack` reported the arithmetic as sound. Downstream it reaches
        `spec.data.values`, and the route serialises with `allow_nan=False`, so
        ONE bad payload would take out every other finding's chart with it. The
        in-band-failure contract says a payload that cannot be drawn costs its own
        entry and nothing else.
        """
        for value in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(PayloadError, match="finite"):
                parse_payload(
                    "attribution",
                    _attribution(contained_by="total_ms", unattributed_delta=value, unattributed_withheld=None),
                )

    def test_a_small_movement_between_large_levels_still_has_to_reconcile(self):
        """Scaling the slack off the LEVELS makes this guard inert where it matters.

        A 50 ms movement between two ~44 s levels would tolerate a stated delta
        anywhere in ±220 ms — over four times the movement — and the compiler draws the
        bar from `delta` while the table prints `a`/`b` beside it, so the picture
        would contradict its own numbers. That regime is not hypothetical: the
        compiler's display-scale reasoning is written around it.
        """
        with pytest.raises(PayloadError, match="states delta"):
            parse_payload(
                "attribution",
                _attribution(end_to_end={"measure": "total_ms", "delta": 50.0, "a": 44000.0, "b": 44220.0}),
            )

    def test_rounding_at_the_levels_own_precision_is_still_tolerated(self):
        """The allowance exists so the rule above cannot demand precision a generator lacks."""
        parse_payload(
            "attribution",
            _attribution(end_to_end={"measure": "total_ms", "delta": 50.0, "a": 44000.0, "b": 44050.4}),
        )

    @pytest.mark.parametrize("unit,scale", [("ms", 1000.0), ("%", 1.0), ("usd", 1.0)])
    def test_the_reconciliation_means_the_same_thing_in_every_unit(self, unit, scale):
        """`unit` is free text, so any ABSOLUTE slack is a different claim per unit.

        An earlier version added a fixed 0.5 — invisible in `ms`, larger than the
        whole quantity in `%` or `usd`, so the guard switched itself off for exactly
        the payloads whose numbers are small. This same disagreement (levels moving
        3.4 against a stated delta of 3.0) must be refused whatever it measures.
        """
        with pytest.raises(PayloadError, match="states delta"):
            parse_payload(
                "attribution",
                _attribution(
                    unit=unit,
                    end_to_end={"measure": "m", "delta": 3.0 * scale, "a": 100.0 * scale, "b": 103.4 * scale},
                    subsystem={"measure": "other", "delta": 1.0 * scale},
                ),
            )


def _frontier(**overrides):
    """A conforming frontier payload, with per-test overrides applied.

    Two contestants at different costs and qualities, one of them dominated —
    the minimum shape that actually trades something off.
    """
    base = {
        "points": [
            {"label": "model-a-3.5-fast-lite", "cost": 0.0071, "quality": 0.2, "latency_ms": 31000.0},
            {"label": "model-b", "cost": 0.0174, "quality": 0.0, "latency_ms": 48700.0, "dominated": True},
        ],
        "bar": 0.5,
        "cost_label": "Cost per run (USD)",
        "quality_label": "pass^k",
    }
    base.update(overrides)
    return base


class TestFrontierAccepts:
    def test_the_conforming_shape_parses(self):
        payload = parse_payload("frontier", _frontier())
        assert [point.label for point in payload.points] == ["model-a-3.5-fast-lite", "model-b"]
        assert payload.points[1].dominated is True

    def test_a_point_with_no_cost_is_kept_and_flagged_rather_than_dropped(self):
        """A contestant whose cost nothing observed is disclosed, not silently excluded.

        `None` and not 0: an unpriced point drawn at zero is the cheapest thing on
        the chart, which is the one reading the missing number must never produce.
        """
        payload = parse_payload(
            "frontier",
            _frontier(
                points=[
                    {"label": "unpriced", "cost": None, "quality": 0.4},
                    {"label": "a", "cost": 0.01, "quality": 0.2},
                    {"label": "b", "cost": 0.02, "quality": 0.3},
                ]
            ),
        )
        assert payload.points[0].cost is None

    @pytest.mark.parametrize(("dominated", "dominance"), [(True, "not_separated"), (False, "dominated")])
    def test_a_flag_that_disagrees_with_the_lens_verdict_is_refused(self, dominated, dominance):
        """The shape is drawn from the verdict and older readers key on the flag; they must say one thing."""
        points = [
            {"label": "a", "cost": 0.01, "quality": 0.4, "dominated": dominated, "dominance": dominance},
            {"label": "b", "cost": 0.02, "quality": 0.2},
        ]
        with pytest.raises(PayloadError, match="dominance"):
            parse_payload("frontier", _frontier(points=points))

    def test_the_optional_captions_and_bar_may_all_be_absent(self):
        """The axes have defaults; a campaign that set no quality bar has no bar to echo."""
        payload = parse_payload(
            "frontier",
            {"points": [{"label": "a", "cost": 0.01, "quality": 0.4}, {"label": "b", "cost": 0.02, "quality": 0.2}]},
        )
        assert payload.bar is None
        assert payload.cost_label is None

    def test_a_disqualified_point_carries_its_reason(self):
        payload = parse_payload(
            "frontier",
            _frontier(
                points=[
                    {"label": "a", "cost": 0.01, "quality": 0.4},
                    {
                        "label": "b",
                        "cost": 0.02,
                        "quality": 0.9,
                        "disqualified": True,
                        "disqualified_reason": "failed the boundary battery",
                    },
                ]
            ),
        )
        assert payload.points[1].disqualified_reason == "failed the boundary battery"


class TestFrontierRejections:
    """Written against the contract's own refusals, not against a malformed corpus.

    No malformed `frontier` payload exists to reproduce — the stored malformed
    payloads are all other types — so each case below is the contract stating what
    it will not draw, rather than a defect observed and pinned.
    """

    def test_a_single_point_is_rejected(self):
        with pytest.raises(PayloadError, match="at least 2 points"):
            parse_payload("frontier", {"points": [{"label": "alone", "cost": 0.01, "quality": 0.4}]})

    def test_duplicate_labels_are_rejected(self):
        points = [{"label": "same", "cost": 0.01, "quality": 0.4}, {"label": "same", "cost": 0.02, "quality": 0.2}]
        with pytest.raises(PayloadError, match="unique"):
            parse_payload("frontier", {"points": points})

    def test_a_non_finite_measure_is_rejected(self):
        """A NaN reaches `spec.data.values`, which serialises with `allow_nan=False`."""
        with pytest.raises(PayloadError, match="finite"):
            parse_payload(
                "frontier",
                _frontier(
                    points=[
                        {"label": "a", "cost": 0.01, "quality": float("nan")},
                        {"label": "b", "cost": 0.02, "quality": 0.2},
                    ]
                ),
            )

    def test_a_negative_latency_is_rejected(self):
        with pytest.raises(PayloadError, match="negative"):
            parse_payload(
                "frontier",
                _frontier(
                    points=[
                        {"label": "a", "cost": 0.01, "quality": 0.4, "latency_ms": -1.0},
                        {"label": "b", "cost": 0.02, "quality": 0.2},
                    ]
                ),
            )

    def test_a_frontier_where_nothing_has_a_cost_is_rejected(self):
        """Every cost null is a column of points at one x — quality wearing a trade-off's title."""
        with pytest.raises(PayloadError, match="at least 2 points carrying a cost"):
            parse_payload("frontier", {"points": [{"label": "a", "quality": 0.4}, {"label": "b", "quality": 0.2}]})

    def test_a_frontier_with_only_one_priced_contestant_is_rejected(self):
        """Two points and one cost still draws ONE mark — the coordinate the count rule refuses.

        Both axes would then crop to a padded band around that single value, which
        reads as a resolved comparison while spanning nothing that was measured.
        """
        points = [{"label": "priced", "cost": 0.01, "quality": 0.4}, {"label": "never-priced", "quality": 0.2}]
        with pytest.raises(PayloadError, match="at least 2 points carrying a cost"):
            parse_payload("frontier", {"points": points})

    def test_a_reason_without_the_flag_is_rejected(self):
        """The reason would print beside a mark drawn as an eligible contestant."""
        points = [
            {"label": "a", "cost": 0.01, "quality": 0.4, "disqualified_reason": "failed the boundary battery"},
            {"label": "b", "cost": 0.02, "quality": 0.2},
        ]
        with pytest.raises(PayloadError, match="disqualified is false"):
            parse_payload("frontier", {"points": points})

    def test_a_non_finite_bar_is_rejected(self):
        with pytest.raises(PayloadError, match="bar must be a finite number"):
            parse_payload("frontier", _frontier(bar=float("inf")))

    def test_an_extra_top_level_key_is_rejected(self):
        """`axis` is the key an earlier stored payload carried instead of the captions."""
        with pytest.raises(PayloadError, match="axis"):
            parse_payload("frontier", _frontier(axis="p95_ms vs mean_delivered_items"))


def _sweep(**overrides) -> dict:
    """A three-configuration sweep over one categorical lever and one ordered one."""
    payload = {
        "ranked": {"measure": "pass^k", "unit": None},
        "secondary": {"measure": "cost per run", "unit": "usd"},
        "rows": [
            {"config": {"model": "gpt-5", "fetch_concurrency": "8"}, "ranked_value": 0.72, "secondary_value": 0.0111},
            {"config": {"model": "model-b", "fetch_concurrency": "4"}, "ranked_value": 0.61, "secondary_value": 0.0094},
            {"config": {"model": "gpt-5", "fetch_concurrency": "2"}, "ranked_value": 0.55, "secondary_value": 0.0142},
        ],
    }
    return payload | overrides


class TestSweepRankingPayload:
    """The contract for a ranking of combinations.

    Every refusal here is also written into the generator prompt's own bullet, and
    that pairing is not optional bookkeeping: a payload refusal discards the WHOLE
    paid analysis rather than just its chart, so a constraint the producer was
    never told about converts a chart problem into a money problem.
    """

    def test_a_conforming_sweep_parses(self):
        assert parse_payload("sweep_ranking", _sweep()) is not None

    def test_one_configuration_is_not_a_ranking(self):
        rows = [{"config": {"model": "gpt-5"}, "ranked_value": 0.7, "secondary_value": 0.01}]
        with pytest.raises(PayloadError, match="at least 2 configurations"):
            parse_payload("sweep_ranking", _sweep(rows=rows))

    def test_rows_sweeping_different_levers_are_rejected(self):
        """A barcode whose columns mean different things per row is not a barcode."""
        rows = [
            {"config": {"model": "gpt-5", "fetch_concurrency": "8"}, "ranked_value": 0.7, "secondary_value": 0.01},
            {"config": {"model": "model-b"}, "ranked_value": 0.6, "secondary_value": 0.02},
        ]
        with pytest.raises(PayloadError, match="different dimension sets"):
            parse_payload("sweep_ranking", _sweep(rows=rows))

    def test_repeats_of_one_configuration_are_rejected(self):
        """Two rows at the same levels draw as identical glyphs at different ranks.

        The barcode IS the row's identity, so a reader's only reading is that one
        configuration scored twice. Where it really was measured twice that is a
        `distribution`; where the rows differ on a lever the payload omitted, the
        missing lever is the fix — and the error says both.

        **This subsumed an earlier "no dimension varies" rule**, which is now
        unreachable rather than merely redundant: unique configurations imply some
        lever differs, so a sweep that passes this check cannot be one in which
        nothing varies. Consolidated rather than kept, since a validator that can
        never fire is a rule a reader believes is enforced.
        """
        rows = [
            {"config": {"model": "gpt-5", "fetch_concurrency": "8"}, "ranked_value": 0.7, "secondary_value": 0.01},
            {"config": {"model": "gpt-5", "fetch_concurrency": "8"}, "ranked_value": 0.6, "secondary_value": 0.02},
        ]
        with pytest.raises(PayloadError, match="configurations must be unique"):
            parse_payload("sweep_ranking", _sweep(rows=rows))

    def test_two_configurations_under_one_label_are_rejected(self):
        """Reachable PAST the configuration check, through the field that overrides it.

        These rows differ on a lever, so `configurations must be unique` passes them
        — but the row's drawn identity prefers `label` where one is given, so both
        land in a single band on the shared y scale. That is the same "identical
        glyphs at different ranks" the configuration rule refuses, arrived at by a
        different door, and it is why checking `config` alone was not enough.
        """
        rows = [
            {
                "config": {"model": "gpt-5", "fetch_concurrency": "8"},
                "ranked_value": 0.7,
                "secondary_value": 0.01,
                "label": "baseline",
            },
            {
                "config": {"model": "gpt-5", "fetch_concurrency": "16"},
                "ranked_value": 0.6,
                "secondary_value": 0.02,
                "label": "baseline",
            },
        ]
        with pytest.raises(PayloadError, match="row labels must be unique"):
            parse_payload("sweep_ranking", _sweep(rows=rows))

    def test_rows_without_labels_are_not_treated_as_sharing_one(self):
        """`label` is optional, and two nulls are not a collision.

        The uniqueness rule reads a name that was given; rows that decline to give
        one fall back to their levels, which the configuration rule already keeps
        distinct. Without this, adding the label check would refuse every ordinary
        sweep — the failure mode a uniqueness rule over an optional field invites.
        """
        rows = [
            {"config": {"model": "gpt-5", "fetch_concurrency": "8"}, "ranked_value": 0.7, "secondary_value": 0.01},
            {"config": {"model": "gpt-5", "fetch_concurrency": "16"}, "ranked_value": 0.6, "secondary_value": 0.02},
        ]
        assert parse_payload("sweep_ranking", _sweep(rows=rows)) is not None

    def test_a_partial_duplicate_is_rejected_too(self):
        """The case the old rule could not see: three rows, two of them the same."""
        rows = [
            {"config": {"model": "gpt-5", "fetch_concurrency": "8"}, "ranked_value": 0.7, "secondary_value": 0.01},
            {"config": {"model": "model-b", "fetch_concurrency": "4"}, "ranked_value": 0.65, "secondary_value": 0.02},
            {"config": {"model": "gpt-5", "fetch_concurrency": "8"}, "ranked_value": 0.6, "secondary_value": 0.03},
        ]
        with pytest.raises(PayloadError, match="appears more than once"):
            parse_payload("sweep_ranking", _sweep(rows=rows))

    def test_an_empty_config_is_rejected(self):
        rows = [
            {"config": {}, "ranked_value": 0.7, "secondary_value": 0.01},
            {"config": {}, "ranked_value": 0.6, "secondary_value": 0.02},
        ]
        with pytest.raises(PayloadError, match="carries no dimensions"):
            parse_payload("sweep_ranking", _sweep(rows=rows))

    def test_a_declared_dimension_no_row_sweeps_is_rejected(self):
        """It decides which columns take a ramp, so a wrong name draws by fallthrough."""
        with pytest.raises(PayloadError, match="which no row sweeps"):
            parse_payload("sweep_ranking", _sweep(dimensions=[{"name": "temperature", "ordered": True}]))

    def test_a_swept_lever_the_dimension_list_omits_is_rejected(self):
        with pytest.raises(PayloadError, match="with nothing declared"):
            parse_payload("sweep_ranking", _sweep(dimensions=[{"name": "model", "ordered": False}]))

    def test_a_lever_declared_ordered_over_unplaceable_levels_counts_against_the_hue_ceiling(self):
        """It draws in hues, and hues are what the ceiling rations.

        A ramp is a position along a sorted range, so `small`/`medium`/`large`
        cannot take one however it was declared — it draws as hues like any other
        categorical lever, and exempting it on the strength of its declaration would
        let a barcode draw more hues than the palette validates.

        Refusal here costs the WHOLE analysis, so the scope has to be one the
        producer can predict. That is why the generator prompt says "only numeric
        levels ramp" rather than naming declared orderedness, because a bound the
        producer cannot see is a bill it cannot avoid.
        """
        rows = [
            {"config": {"model": "gpt-5", "size": "small"}, "ranked_value": 0.72, "secondary_value": 0.011},
            {"config": {"model": "model-b", "size": "medium"}, "ranked_value": 0.61, "secondary_value": 0.009},
            {"config": {"model": "gpt-5", "size": "large"}, "ranked_value": 0.55, "secondary_value": 0.014},
        ]
        declared = [{"name": "model", "ordered": False}, {"name": "size", "ordered": True}]
        with pytest.raises(PayloadError, match="5 categorical levels"):
            parse_payload("sweep_ranking", _sweep(rows=rows, dimensions=declared))

    def test_the_same_lever_with_numeric_levels_ramps_and_does_not_count(self):
        """The other half, or the rule above reads as "declaring ordered does nothing".

        Same shape, same level count, numeric levels — five distinct levels across
        two levers, and it parses, because the ordered one is sampled from a path
        rather than assigned to slots.
        """
        rows = [
            {"config": {"model": "gpt-5", "size": "1"}, "ranked_value": 0.72, "secondary_value": 0.011},
            {"config": {"model": "model-b", "size": "2"}, "ranked_value": 0.61, "secondary_value": 0.009},
            {"config": {"model": "gpt-5", "size": "3"}, "ranked_value": 0.55, "secondary_value": 0.014},
        ]
        declared = [{"name": "model", "ordered": False}, {"name": "size", "ordered": True}]
        assert parse_payload("sweep_ranking", _sweep(rows=rows, dimensions=declared)) is not None

    def test_a_non_finite_measure_is_rejected(self):
        """A NaN reaches `spec.data.values`, which the charts route refuses to serialise."""
        rows = [
            {"config": {"model": "a", "fetch_concurrency": "8"}, "ranked_value": float("nan"), "secondary_value": 0.01},
            {"config": {"model": "b", "fetch_concurrency": "4"}, "ranked_value": 0.6, "secondary_value": 0.02},
        ]
        with pytest.raises(PayloadError, match="ranked_value must be a finite number"):
            parse_payload("sweep_ranking", _sweep(rows=rows))

    def test_a_zero_tolerance_is_rejected(self):
        """A slice admitting only an exact match is a claim about equality nothing supports."""
        with pytest.raises(PayloadError, match="tolerance"):
            parse_payload("sweep_ranking", _sweep(held_fixed={"value": 0.01, "tolerance": 0.0}))

    def test_an_omission_band_overlapping_the_drawn_rows_is_rejected(self):
        """The reader would count the same configurations twice — as marks and as a number."""
        with pytest.raises(PayloadError, match="reaches above the lowest drawn"):
            parse_payload("sweep_ranking", _sweep(omitted={"count": 3, "low": 0.4, "high": 0.6}))

    def test_an_extra_top_level_key_is_rejected(self):
        with pytest.raises(PayloadError, match="cells"):
            parse_payload("sweep_ranking", _sweep(cells=[]))


class TestOrderednessInference:
    """Which levers earn a ramp, when nothing in the bundle declares it."""

    def test_numeric_levels_are_ordered(self):
        assert infer_ordered(["1", "2", "8"]) is True

    def test_free_text_levels_are_not(self):
        assert infer_ordered(["gpt-5", "model-b"]) is False

    def test_a_version_string_is_not_treated_as_ordered(self):
        """Deliberate: `v2` before `v10` is obvious only to a reader who knows the scheme.

        A wrong guess draws a ramp asserting an order the data does not have, which
        is the one failure this chart type exists to avoid — the block pattern in a
        column IS the finding.
        """
        assert infer_ordered(["v1", "v2", "v10"]) is False

    def test_the_absence_sentinel_does_not_make_a_numeric_lever_categorical(self):
        """One unset run is enough to flip an ordered knob categorical.

        A lever no configuration set takes the bundle's em dash, and testing
        parsability without excluding it turns numeric levers such as
        `call_timeout_s` or `search_depth` into hues.
        """
        assert infer_ordered(["2", "4", ABSENT_LEVEL]) is True

    def test_a_lever_nobody_set_has_no_order_to_have(self):
        assert infer_ordered([ABSENT_LEVEL, ABSENT_LEVEL]) is False

    def test_a_null_level_does_not_make_a_numeric_lever_categorical(self):
        """A lever overlaid to `null` resolves to the level `null` (#574); it is set apart, not parsed (#694)."""
        assert infer_ordered(["null", "6", "12"]) is True
        assert infer_ordered(["null", "null"]) is False
        assert infer_ordered(["null", "gpt-5"]) is False

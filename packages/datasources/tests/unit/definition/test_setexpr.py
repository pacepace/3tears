"""unit tests for :class:`SetExpr`, its four operators, and per-term projection.

Five corpus semantics drive this module:

- composition defaults to the union of every unit, which is what the
  prototype does by having every unit ``INSERT`` into one table.
- each term carries its OWN projection. The UHG deliverable unions two
  audiences where the policymaker branch projects three rollup flags
  and the opinion-elite branch projects ``NULL`` for all three.
- ``intersect`` carries payload, not just membership: its one real use
  delivers aggregated columns drawn from BOTH sides.
- ``intersect`` occurs in two positions -- at composition and inside a
  resolution, before that resolution's own aggregation -- and the two
  compile to different plans, so they are structurally distinct nodes.
- ``ranked_precedence`` carries TWO independent orders. One audience
  deduplicates core-before-expansion while labelling
  householders-before-core-before-expansion, computing the label BEFORE
  the dedup, with its first arm testing an expansion flag rather than a
  dataset term.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.datasources.definition.setexpr import (
    COMPOSITION_FILTERED_ARTIFACTS,
    CategoryPosition,
    CompositionPlacement,
    IntersectColumn,
    ResolutionIntersect,
    ResolutionIntersectColumn,
    SetExpr,
    SetOperator,
    SetTerm,
    TermColumn,
)

_PM_TERM = {
    "name": "pm",
    "upstream": "uhg_policymakers",
    "projection": [
        {"name": "state_level", "value": "resolved.state_level"},
        {"name": "federal_level", "value": "resolved.federal_level"},
        {"name": "government_level", "value": "resolved.government_level"},
    ],
}

_OE_TERM = {
    "name": "oe",
    "upstream": "uhg_opinion_elites",
    "projection": [
        {"name": "state_level", "value": {"literal": None}},
        {"name": "federal_level", "value": {"literal": None}},
        {"name": "government_level", "value": {"literal": None}},
    ],
}

_RANKED = {
    "op": "ranked_precedence",
    "terms": [
        {
            "name": "core",
            "upstream": "universal_2026_core",
            "projection": [{"name": "type", "value": {"literal": "core"}}],
        },
        {
            "name": "expansion",
            "upstream": "universal_2026_expansion",
            "projection": [{"name": "type", "value": {"literal": "expansion"}}],
        },
    ],
    "category_column": "audience_level",
    "dedup_key_columns": ["voterbase_id"],
    "dedup_order": ["core", "expansion"],
    "tiebreak": ["list_id"],
    "category_order": [
        {"when": {"compare": {"left": "resolved.householders", "op": "=", "right": 1}}, "label": "householder"},
        {"when": {"compare": {"left": "resolved.type", "op": "=", "right": {"literal": "core"}}}, "label": "core"},
    ],
    "category_otherwise": "expansion",
    "category_position": "before_dedup",
}


class TestDefaultComposition:
    """the union of every unit, which is what an absent composition means."""

    def test_the_default_operator_is_union(self) -> None:
        assert SetExpr().op is SetOperator.UNION

    def test_union_with_no_terms_means_every_unit(self) -> None:
        expression = SetExpr()
        assert expression.terms == []
        assert expression.is_default_union is True

    def test_the_operator_vocabulary_is_exactly_four(self) -> None:
        assert {member.value for member in SetOperator} == {
            "union",
            "intersect",
            "difference",
            "ranked_precedence",
        }

    def test_rejects_an_unknown_operator(self) -> None:
        with pytest.raises(ValidationError):
            SetExpr.model_validate({"op": "except"})

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            SetExpr.model_validate({"op": "union", "distinct": True})


class TestDatasetTerms:
    """terms are a unit, a rollup, an upstream, or a nested expression."""

    def test_a_unit_term(self) -> None:
        assert SetTerm.model_validate({"name": "t", "unit": "knowwho_all"}).unit == "knowwho_all"

    def test_a_rollup_term(self) -> None:
        assert SetTerm.model_validate({"name": "t", "rollup": "federal_level"}).rollup == "federal_level"

    def test_an_upstream_term(self) -> None:
        assert SetTerm.model_validate({"name": "t", "upstream": "universal_2026_core"}).upstream

    def test_a_nested_expression_term(self) -> None:
        term = SetTerm.model_validate(
            {
                "name": "t",
                "expression": {
                    "op": "union",
                    "terms": [{"name": "a", "unit": "a"}, {"name": "b", "unit": "b"}],
                },
            }
        )
        assert term.expression is not None
        assert len(term.expression.terms) == 2

    def test_rejects_a_term_naming_nothing(self) -> None:
        with pytest.raises(ValidationError):
            SetTerm.model_validate({"name": "t"})

    def test_rejects_a_term_naming_two_kinds(self) -> None:
        with pytest.raises(ValidationError):
            SetTerm.model_validate({"name": "t", "unit": "a", "rollup": "b"})

    def test_a_term_carries_no_expansion_kind(self) -> None:
        assert "expansion" not in SetTerm.model_fields

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            SetTerm.model_validate({"name": "t", "unit": "a", "expansion": "householders"})

    def test_rejects_repeated_term_names(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            SetExpr.model_validate({"op": "union", "terms": [{"name": "t", "unit": "a"}, {"name": "t", "unit": "b"}]})
        assert "twice" in str(excinfo.value)


class TestPerTermProjection:
    """the UHG deliverable: three rollup flags on one branch, NULL on the other."""

    def test_two_branches_project_different_values_for_one_column(self) -> None:
        expression = SetExpr.model_validate({"op": "union", "terms": [_PM_TERM, _OE_TERM]})
        pm, oe = expression.terms
        assert [column.name for column in pm.projection] == [column.name for column in oe.projection]
        assert pm.projection[0].value != oe.projection[0].value

    def test_a_null_projection_is_expressible(self) -> None:
        column = TermColumn.model_validate({"name": "state_level", "value": {"literal": None}})
        assert column.value.literal is None

    def test_a_per_term_literal_discriminator_is_expressible(self) -> None:
        column = TermColumn.model_validate({"name": "type", "value": {"literal": "core"}})
        assert column.value.literal == "core"

    def test_a_term_may_project_nothing(self) -> None:
        assert SetTerm.model_validate({"name": "t", "unit": "a"}).projection == []

    def test_rejects_a_repeated_projected_column(self) -> None:
        with pytest.raises(ValidationError):
            SetTerm.model_validate(
                {
                    "name": "t",
                    "unit": "a",
                    "projection": [
                        {"name": "units", "value": {"literal": 1}},
                        {"name": "units", "value": {"literal": 2}},
                    ],
                }
            )

    def test_rejects_a_blank_projected_column_name(self) -> None:
        with pytest.raises(ValidationError):
            TermColumn.model_validate({"name": " ", "value": {"literal": 1}})


class TestIntersectCarriesPayload:
    """per output column, which side it comes from."""

    def _intersection(self) -> SetExpr:
        return SetExpr.model_validate(
            {
                "op": "intersect",
                "terms": [
                    {"name": "health_providers", "unit": "health_providers"},
                    {"name": "linkedin_job_titles", "unit": "linkedin_job_titles"},
                ],
                "payload": [
                    {"name": "health_provider_list_ids", "term": "health_providers", "column": "list_ids"},
                    {"name": "linkedin_list_ids", "term": "linkedin_job_titles", "column": "list_ids"},
                ],
            }
        )

    def test_payload_columns_are_drawn_from_both_sides(self) -> None:
        payload = self._intersection().payload
        assert {column.term for column in payload} == {"health_providers", "linkedin_job_titles"}

    def test_two_output_columns_may_share_a_source_column_name(self) -> None:
        payload = self._intersection().payload
        assert {column.column for column in payload} == {"list_ids"}
        assert len({column.name for column in payload}) == 2

    def test_intersect_requires_a_payload(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            SetExpr.model_validate(
                {"op": "intersect", "terms": [{"name": "a", "unit": "a"}, {"name": "b", "unit": "b"}]}
            )
        assert "payload" in str(excinfo.value)

    def test_payload_must_name_a_declared_term(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            SetExpr.model_validate(
                {
                    "op": "intersect",
                    "terms": [{"name": "a", "unit": "a"}, {"name": "b", "unit": "b"}],
                    "payload": [{"name": "x", "term": "c", "column": "y"}],
                }
            )
        assert "'c'" in str(excinfo.value)

    def test_union_carries_no_payload(self) -> None:
        with pytest.raises(ValidationError):
            SetExpr.model_validate(
                {
                    "op": "union",
                    "terms": [{"name": "a", "unit": "a"}, {"name": "b", "unit": "b"}],
                    "payload": [{"name": "x", "term": "a", "column": "y"}],
                }
            )

    def test_rejects_a_repeated_output_column(self) -> None:
        with pytest.raises(ValidationError):
            SetExpr.model_validate(
                {
                    "op": "intersect",
                    "terms": [{"name": "a", "unit": "a"}, {"name": "b", "unit": "b"}],
                    "payload": [
                        {"name": "x", "term": "a", "column": "y"},
                        {"name": "x", "term": "b", "column": "y"},
                    ],
                }
            )

    def test_payload_column_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            IntersectColumn.model_validate({"name": "x", "term": "a", "column": "y", "side": "left"})


class TestIntersectInTwoPositions:
    """composition and resolution are different plans, so different nodes."""

    def test_a_resolution_intersect_is_a_distinct_type(self) -> None:
        assert ResolutionIntersect is not SetExpr

    def test_a_resolution_intersect_names_another_units_resolved_rows(self) -> None:
        node = ResolutionIntersect.model_validate(
            {
                "alias": "linkedin_job_titles",
                "against": {"scope": "this_definition", "unit": "linkedin_job_titles", "stage": "resolved"},
                "key_columns": ["voterbase_id"],
                "payload": [{"name": "linkedin_list_ids", "column": "list_ids"}],
            }
        )
        assert node.against.unit == "linkedin_job_titles"
        assert node.against.stage.value == "resolved"

    def test_a_resolution_intersect_carries_payload_from_the_other_side(self) -> None:
        column = ResolutionIntersectColumn.model_validate({"name": "linkedin_list_ids", "column": "list_ids"})
        assert column.name != column.column

    def test_a_resolution_intersect_carries_no_operator_field(self) -> None:
        assert "op" not in ResolutionIntersect.model_fields

    @pytest.mark.parametrize("alias", ["prior audience", "prior-audience", "2024", "prior.audience", ""])
    def test_a_non_identifier_intersect_alias_is_refused(self, alias: str) -> None:
        """an alias that cannot be spelled ``rel.<alias>`` is unaddressable.

        the alias is half of a namespaced reference, so a non-identifier
        leaves the intersected side reachable by no predicate at all. the
        compiler would then report an unresolved name against a definition
        this model accepted, which reads as a compiler defect rather than
        an authoring one. RelationRef.alias is validated for the same
        reason and this matches it.
        """
        with pytest.raises(ValidationError) as excinfo:
            ResolutionIntersect.model_validate(
                {
                    "alias": alias,
                    "against": {"scope": "dataset", "dataset": "amz_universe_2024", "artifact": "wide"},
                    "key_columns": ["voterbase_id"],
                },
            )
        assert "is not an identifier" in str(excinfo.value) or "at least 1 character" in str(excinfo.value)

    def test_a_composition_intersect_carries_no_stage_or_position_field(self) -> None:
        assert {"stage", "position", "phase", "after"} & set(SetExpr.model_fields) == set()

    def test_resolution_intersect_requires_a_key(self) -> None:
        with pytest.raises(ValidationError):
            ResolutionIntersect.model_validate(
                {"alias": "x", "against": {"unit": "a", "stage": "resolved"}, "key_columns": []}
            )

    def test_resolution_intersect_rejects_a_repeated_payload_column(self) -> None:
        with pytest.raises(ValidationError):
            ResolutionIntersect.model_validate(
                {
                    "alias": "x",
                    "against": {"unit": "a", "stage": "resolved"},
                    "key_columns": ["voterbase_id"],
                    "payload": [{"name": "n", "column": "a"}, {"name": "n", "column": "b"}],
                }
            )

    def test_resolution_intersect_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ResolutionIntersect.model_validate(
                {"alias": "x", "against": {"unit": "a", "stage": "resolved"}, "key_columns": ["k"], "op": "intersect"}
            )


class TestDifference:
    """difference at the composition level, ordered."""

    def test_difference_is_expressible(self) -> None:
        expression = SetExpr.model_validate(
            {"op": "difference", "terms": [{"name": "a", "unit": "a"}, {"name": "b", "unit": "b"}]}
        )
        assert expression.op is SetOperator.DIFFERENCE

    def test_difference_needs_at_least_two_terms(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            SetExpr.model_validate({"op": "difference", "terms": [{"name": "a", "unit": "a"}]})
        assert "two" in str(excinfo.value)

    def test_intersect_needs_at_least_two_terms(self) -> None:
        with pytest.raises(ValidationError):
            SetExpr.model_validate(
                {
                    "op": "intersect",
                    "terms": [{"name": "a", "unit": "a"}],
                    "payload": [{"name": "x", "term": "a", "column": "y"}],
                }
            )


class TestRankedPrecedence:
    """two independent orders, and a tiebreak."""

    def test_the_delivered_category_carries_a_per_definition_name(self) -> None:
        assert SetExpr.model_validate(_RANKED).category_column == "audience_level"

    def test_ranked_precedence_without_a_category_column_is_refused(self) -> None:
        without = {key: value for key, value in _RANKED.items() if key != "category_column"}
        with pytest.raises(ValidationError, match="category_column"):
            SetExpr.model_validate(without)

    def test_a_category_column_that_is_not_an_identifier_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="category_column"):
            SetExpr.model_validate({**_RANKED, "category_column": "audience level"})

    def test_another_operator_may_not_carry_a_category_column(self) -> None:
        with pytest.raises(ValidationError, match="category_column"):
            SetExpr.model_validate({"op": "union", "terms": [_PM_TERM, _OE_TERM], "category_column": "audience_level"})

    def test_dedup_order_and_category_order_are_separate_declarations(self) -> None:
        expression = SetExpr.model_validate(_RANKED)
        assert expression.dedup_order == ["core", "expansion"]
        assert [arm.label for arm in expression.category_order or ()] == ["householder", "core"]
        assert expression.dedup_order != [arm.label for arm in expression.category_order or ()]

    def test_the_category_has_a_third_label_the_dedup_order_does_not(self) -> None:
        expression = SetExpr.model_validate(_RANKED)
        categories = [arm.label for arm in expression.category_order or ()] + [expression.category_otherwise]
        assert categories == ["householder", "core", "expansion"]
        assert expression.dedup_order == ["core", "expansion"]
        assert len(categories) == len(expression.dedup_order or []) + 1

    def test_the_first_category_arm_tests_a_flag_not_a_dataset_term(self) -> None:
        expression = SetExpr.model_validate(_RANKED)
        arms = expression.category_order or []
        first = arms[0].when.compare
        assert first is not None
        assert [reference.ref for reference in first.references] == ["resolved.householders"]

    def test_the_category_is_computed_before_the_dedup(self) -> None:
        assert SetExpr.model_validate(_RANKED).category_position is CategoryPosition.BEFORE_DEDUP

    def test_the_category_position_after_dedup_is_also_expressible(self) -> None:
        expression = SetExpr.model_validate({**_RANKED, "category_position": "after_dedup"})
        assert expression.category_position is CategoryPosition.AFTER_DEDUP

    def test_a_tiebreak_is_carried(self) -> None:
        assert SetExpr.model_validate(_RANKED).tiebreak == ["list_id"]

    @pytest.mark.parametrize(
        "missing", ["dedup_key_columns", "dedup_order", "tiebreak", "category_order", "category_position"]
    )
    def test_every_ranked_declaration_is_required(self, missing: str) -> None:
        payload = dict(_RANKED)
        payload.pop(missing)
        with pytest.raises(ValidationError) as excinfo:
            SetExpr.model_validate(payload)
        assert missing in str(excinfo.value)

    def test_dedup_order_must_cover_every_term_exactly_once(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            SetExpr.model_validate({**_RANKED, "dedup_order": ["core"]})
        assert "dedup_order" in str(excinfo.value)

    def test_dedup_order_must_not_name_an_undeclared_term(self) -> None:
        with pytest.raises(ValidationError):
            SetExpr.model_validate({**_RANKED, "dedup_order": ["core", "householders"]})

    def test_ranked_fields_are_refused_on_a_union(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            SetExpr.model_validate(
                {
                    "op": "union",
                    "terms": [{"name": "a", "unit": "a"}, {"name": "b", "unit": "b"}],
                    "dedup_key_columns": ["voterbase_id"],
                }
            )
        assert "ranked_precedence" in str(excinfo.value)

    def test_a_category_arm_may_not_bind_a_pre_aggregate_namespace(self) -> None:
        payload = dict(_RANKED)
        payload["category_order"] = [
            {"when": {"compare": {"left": "source.job_title", "op": "=", "right": {"literal": "x"}}}, "label": "l"}
        ]
        with pytest.raises(ValidationError) as excinfo:
            SetExpr.model_validate(payload)
        assert "source" in str(excinfo.value)

    def test_rejects_a_repeated_category_label(self) -> None:
        payload = dict(_RANKED)
        payload["category_order"] = [
            {"when": {"compare": {"left": "resolved.a", "op": "=", "right": 1}}, "label": "same"},
            {"when": {"compare": {"left": "resolved.b", "op": "=", "right": 1}}, "label": "same"},
        ]
        with pytest.raises(ValidationError):
            SetExpr.model_validate(payload)


class TestCompositionIsNotAfterExpansion:
    """D17: after qualification, before expansion. One placement only."""

    def test_the_model_admits_exactly_one_placement_not_after_expansion(self) -> None:
        assert len(CompositionPlacement) == 1
        assert CompositionPlacement.AFTER_QUALIFICATION_BEFORE_EXPANSION.value == (
            "after_qualification_before_expansion"
        )

    def test_a_stage_field_is_not_after_expansion_because_there_is_no_stage_field(self) -> None:
        forbidden = {"stage", "phase", "position", "after", "when", "applies_after"}
        assert forbidden & set(SetExpr.model_fields) == set()

    @pytest.mark.parametrize("field_name", ["stage", "phase", "after_expansion", "applies_after"])
    def test_authoring_a_placement_is_not_after_expansion_it_is_rejected(self, field_name: str) -> None:
        with pytest.raises(ValidationError):
            SetExpr.model_validate({"op": "union", field_name: "after_expansion"})

    def test_composition_filters_long_and_qualified_not_after_expansion_only_wide(self) -> None:
        assert COMPOSITION_FILTERED_ARTIFACTS == ("long", "qualified", "wide")


class TestRoundTrip:
    """authored order and per-term projection survive serialization."""

    def test_a_two_branch_union_round_trips(self) -> None:
        payload = {"op": "union", "terms": [_PM_TERM, _OE_TERM]}
        rebuilt = SetExpr.model_validate(SetExpr.model_validate(payload).model_dump())
        assert [term.name for term in rebuilt.terms] == ["pm", "oe"]
        assert rebuilt.terms[1].projection[0].value.literal is None

    def test_ranked_precedence_round_trips(self) -> None:
        rebuilt = SetExpr.model_validate(SetExpr.model_validate(_RANKED).model_dump())
        assert rebuilt.dedup_order == ["core", "expansion"]
        assert rebuilt.category_position is CategoryPosition.BEFORE_DEDUP

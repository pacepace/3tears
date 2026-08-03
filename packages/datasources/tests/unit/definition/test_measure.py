"""unit tests for :class:`Measure`, its grain, its scope, and ``having``.

the grain is the whole point. it is per ``(entity, source record)``, so a
donor whose contributions arrive under two source records is summed once
per record, and a threshold the combined total would clear can fail in
every group. eleven production units are DEFINED by that, and inheriting
the surrounding group-by changes which entities qualify while changing
nothing visible in the definition.

``filter_position`` is the second trap: the prototype renders
``custom_aggregate_filters`` as an outer ``WHERE`` over the aggregated
subquery rather than a ``HAVING``, and the two differ whenever the
measure's grain differs from the unit's output grain. the qualification
stage re-aggregates ``MIN(candidate_count)`` at a coarser grain than the
threshold evaluated in the ``WHERE``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.datasources.definition.expression import Predicate
from threetears.datasources.definition.measure import (
    DuplicateMeasureName,
    FilterPosition,
    Measure,
    MeasureScope,
    UndeclaredMeasure,
    validate_having_measures,
    validate_unique_measure_names,
)
from threetears.datasources.definition.parameters import ParameterType

_CONTRIBUTION_SUM = {
    "name": "contribution_sum",
    "expression": {"arith": "SUM(source.contribution::float * 1.0/bridge.candidate_count::float)"},
    "grain": ["voterbase_id", "list_id"],
    "scope": "resolution",
}


class TestGrainIsNeverInherited:
    """``grain`` is required and has no default."""

    def test_grain_has_no_default(self) -> None:
        assert Measure.model_fields["grain"].is_required() is True

    def test_omitting_grain_fails_validation(self) -> None:
        with pytest.raises(ValidationError):
            Measure.model_validate({"name": "contribution_sum", "expression": "SUM(x)", "scope": "resolution"})

    def test_an_empty_grain_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Measure.model_validate({**_CONTRIBUTION_SUM, "grain": []})

    def test_a_repeated_grain_key_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Measure.model_validate({**_CONTRIBUTION_SUM, "grain": ["voterbase_id", "voterbase_id"]})

    def test_a_grain_key_must_be_an_identifier(self) -> None:
        with pytest.raises(ValidationError):
            Measure.model_validate({**_CONTRIBUTION_SUM, "grain": ["mat.voterbase_id"]})

    def test_the_entity_and_source_record_grain_is_authored(self) -> None:
        measure = Measure.model_validate(_CONTRIBUTION_SUM)
        assert measure.grain == ["voterbase_id", "list_id"]

    def test_one_named_measure_carries_two_grains_in_two_resolutions(self) -> None:
        standard = Measure.model_validate(
            {
                "name": "candidate_count",
                "expression": {"arith": "MIN(bridge.candidate_count)"},
                "grain": ["voterbase_id", "list_id"],
                "scope": "resolution",
            }
        )
        linkedin = Measure.model_validate(
            {
                "name": "candidate_count",
                "expression": {"arith": "MIN(bridge.candidate_count)"},
                "grain": ["unit", "voterbase_id", "record_year", "list_id"],
                "scope": "resolution",
            }
        )
        assert standard.name == linkedin.name
        assert standard.grain != linkedin.grain


class TestScopeAndFilterPosition:
    """where the measure is computed, and where the filter over it lands."""

    def test_scope_is_required(self) -> None:
        with pytest.raises(ValidationError):
            Measure.model_validate({k: v for k, v in _CONTRIBUTION_SUM.items() if k != "scope"})

    def test_filter_position_defaults_to_having(self) -> None:
        assert Measure.model_validate(_CONTRIBUTION_SUM).filter_position is FilterPosition.HAVING

    def test_outer_where_is_expressible(self) -> None:
        measure = Measure.model_validate({**_CONTRIBUTION_SUM, "filter_position": "outer_where"})
        assert measure.filter_position is FilterPosition.OUTER_WHERE

    def test_delivery_scope_is_expressible(self) -> None:
        measure = Measure.model_validate(
            {
                "name": "candidate_count",
                "expression": {"arith": "MIN(resolved.candidate_count)"},
                "grain": ["voterbase_id", "list_id", "unit"],
                "scope": "delivery",
                "filter_position": "outer_where",
            }
        )
        assert measure.scope is MeasureScope.DELIVERY

    def test_the_qualification_re_aggregation_is_a_min_of_a_min_at_a_coarser_grain(self) -> None:
        resolution = Measure.model_validate(
            {
                "name": "candidate_count",
                "expression": {"arith": "MIN(bridge.candidate_count)"},
                "grain": ["voterbase_id", "list_id"],
                "scope": "resolution",
            }
        )
        delivery = Measure.model_validate(
            {
                "name": "candidate_count",
                "expression": {"arith": "MIN(resolved.candidate_count)"},
                "grain": ["voterbase_id", "list_id", "unit"],
                "scope": "delivery",
                "filter_position": "outer_where",
            }
        )
        assert delivery.grain != resolution.grain
        assert delivery.filter_position is FilterPosition.OUTER_WHERE

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Measure.model_validate({**_CONTRIBUTION_SUM, "quality": True})

    def test_there_is_no_singular_quality_field(self) -> None:
        assert "quality" not in Measure.model_fields


class TestMeasureExpression:
    """the expression is carried verbatim and scanned for references."""

    def test_the_expression_scans_its_namespaced_references(self) -> None:
        measure = Measure.model_validate(_CONTRIBUTION_SUM)
        assert [reference.ref for reference in measure.expression.references] == [
            "source.contribution",
            "bridge.candidate_count",
        ]

    def test_a_bare_string_is_read_as_aggregate_text_not_a_reference(self) -> None:
        measure = Measure.model_validate({**_CONTRIBUTION_SUM, "expression": "SUM(source.contribution)"})
        assert measure.expression.arith == "SUM(source.contribution)"

    def test_blank_expression_text_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Measure.model_validate({**_CONTRIBUTION_SUM, "expression": "   "})

    def test_the_measure_name_must_be_an_identifier(self) -> None:
        with pytest.raises(ValidationError):
            Measure.model_validate({**_CONTRIBUTION_SUM, "name": "contribution sum"})

    def test_the_authored_alias_is_not_inferred_from_the_expression(self) -> None:
        drifted = Measure.model_validate({**_CONTRIBUTION_SUM, "name": "sum_of_contributions"})
        assert drifted.name == "sum_of_contributions"


class TestHavingOverDeclaredMeasures:
    """``measure.<name>`` binds in ``having`` and nowhere else."""

    def test_the_contribution_threshold_unit_is_expressible(self) -> None:
        measure = Measure.model_validate({**_CONTRIBUTION_SUM, "filter_position": "outer_where"})
        having = Predicate.model_validate({"compare": {"left": "measure.contribution_sum", "op": ">", "right": 2000}})
        validate_having_measures(having, [measure])
        assert measure.grain == ["voterbase_id", "list_id"]

    def test_having_naming_an_undeclared_measure_is_refused(self) -> None:
        measure = Measure.model_validate(_CONTRIBUTION_SUM)
        having = Predicate.model_validate(
            {"compare": {"left": "measure.sum_of_contributions", "op": ">", "right": 2000}}
        )
        with pytest.raises(UndeclaredMeasure):
            validate_having_measures(having, [measure])

    def test_a_measure_with_no_having_is_legal(self) -> None:
        validate_having_measures(None, [Measure.model_validate(_CONTRIBUTION_SUM)])

    def test_a_having_over_no_measure_reference_is_legal(self) -> None:
        having = Predicate.model_validate({"compare": {"left": "bridge.candidate_count", "op": "<=", "right": 4}})
        validate_having_measures(having, [])

    def test_a_duplicate_measure_name_is_refused(self) -> None:
        measures = [Measure.model_validate(_CONTRIBUTION_SUM), Measure.model_validate(_CONTRIBUTION_SUM)]
        with pytest.raises(DuplicateMeasureName):
            validate_unique_measure_names(measures)

    def test_distinct_measure_names_pass_through(self) -> None:
        measures = [
            Measure.model_validate(_CONTRIBUTION_SUM),
            Measure.model_validate({**_CONTRIBUTION_SUM, "name": "sum_of_contributions"}),
        ]
        assert validate_unique_measure_names(measures) == measures


class TestDeclaredResultType:
    """F-04: the family-changing aggregate the reference types cannot give.

    ``uhg_healthcare_providers/ugh_healthcare_providers.sql:54-129``
    computes ``MAX(CASE WHEN job_title LIKE ... THEN 1 ELSE 0 END)`` and
    filters it with ``= 1``. Every reference in that classifier is
    ``source.job_title`` (varchar); the aggregate yields an integer, so
    the compiler's "the type its references share" inference is wrong by
    a whole family. ``count(distinct list_id) > 30``
    (``universal_2026_expansion/top_audience_companies.sql.jinja2:5``) is
    the same shape.
    """

    def test_a_measure_declares_no_result_type_by_default(self) -> None:
        assert Measure.model_validate(_CONTRIBUTION_SUM).result_type is None

    def test_the_healthcare_classifier_declares_the_integer_it_yields(self) -> None:
        measure = Measure.model_validate(
            {
                "name": "has_relevant_linkedin_job_title",
                "expression": "MAX(CASE WHEN source.job_title LIKE '%nurse%' THEN 1 ELSE 0 END)",
                "grain": ["voterbase_id"],
                "scope": "resolution",
                "result_type": "integer",
            }
        )
        assert measure.result_type is ParameterType.INTEGER

    def test_a_count_over_anything_declares_an_integer(self) -> None:
        measure = Measure.model_validate(
            {
                "name": "list_count",
                "expression": "COUNT(DISTINCT source.list_id)",
                "grain": ["employer"],
                "scope": "resolution",
                "result_type": "integer",
            }
        )
        assert measure.result_type is ParameterType.INTEGER

    def test_the_declaration_is_one_of_the_five_declared_families(self) -> None:
        with pytest.raises(ValidationError):
            Measure.model_validate({**_CONTRIBUTION_SUM, "result_type": "varchar"})

    def test_the_declared_type_survives_a_round_trip(self) -> None:
        measure = Measure.model_validate({**_CONTRIBUTION_SUM, "result_type": "decimal"})
        assert Measure.model_validate(measure.model_dump(mode="json")) == measure

    def test_the_result_type_is_not_the_grain_and_not_the_filter_position(self) -> None:
        # three separate declarations; conflating any two loses a semantic.
        measure = Measure.model_validate(
            {**_CONTRIBUTION_SUM, "result_type": "decimal", "filter_position": "outer_where"}
        )
        assert measure.result_type is ParameterType.DECIMAL
        assert measure.filter_position is FilterPosition.OUTER_WHERE
        assert measure.grain == ["voterbase_id", "list_id"]

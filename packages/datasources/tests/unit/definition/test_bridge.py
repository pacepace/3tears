"""unit tests for :class:`BridgeRef` and its PLURAL quality measures.

D16 is the load-bearing decision here. there is no ``long.quality``
column: the long artifact carries one column per declared quality
measure, named ``quality_<measure_name>``.

three corpus facts make plurality non-negotiable:

- ``uhg_healthcare_providers`` gates on ``source_match_tier <= 12`` while
  the Amazon audiences gate on ``candidate_count``. different scales.
- ``candidate_count`` is match quality where LOWER is better, so a skew
  detector keyed on one column silently mis-reports a tier-gated
  audience.
- the expansion artifact has NO quality measure at all --
  ``candidate_count`` is ``NULL::int`` in ``_relationship_union_`` and the
  wide table then computes ``MIN(candidate_count)`` over a column that is
  entirely NULL for expansion rows.
"""

from __future__ import annotations

import inspect

import pydantic
import pytest
from pydantic import ValidationError

from threetears.datasources.definition import bridge as bridge_module
from threetears.datasources.definition.bridge import (
    BridgeRef,
    ConflictingQualityMeasure,
    QualityDirection,
    QualityMeasure,
    ThresholdSemantics,
    union_quality_measures,
)

_CANDIDATE_COUNT = {
    "name": "candidate_count",
    "column": "candidate_count",
    "direction": "lower_is_better",
    "threshold_semantics": "at_most",
}
_SOURCE_MATCH_TIER = {
    "name": "source_match_tier",
    "column": "source_match_tier",
    "direction": "lower_is_better",
    "threshold_semantics": "at_most",
}


class TestQualityMeasure:
    """each measure declares its own scale, polarity, and null behaviour."""

    def test_carries_the_five_declared_fields(self) -> None:
        assert set(QualityMeasure.model_fields) == {
            "name",
            "column",
            "direction",
            "threshold_semantics",
            "unmeasured_is_null",
        }

    def test_unmeasured_is_null_defaults_true(self) -> None:
        assert QualityMeasure.model_validate(_CANDIDATE_COUNT).unmeasured_is_null is True

    def test_the_long_column_is_named_for_the_measure(self) -> None:
        assert QualityMeasure.model_validate(_SOURCE_MATCH_TIER).long_column == "quality_source_match_tier"

    def test_lower_is_better_is_a_real_polarity(self) -> None:
        measure = QualityMeasure.model_validate(_CANDIDATE_COUNT)
        assert measure.direction is QualityDirection.LOWER_IS_BETTER
        assert measure.threshold_semantics is ThresholdSemantics.AT_MOST

    def test_threshold_semantics_is_not_derived_from_direction(self) -> None:
        measure = QualityMeasure.model_validate({**_CANDIDATE_COUNT, "threshold_semantics": "at_least"})
        assert measure.direction is QualityDirection.LOWER_IS_BETTER
        assert measure.threshold_semantics is ThresholdSemantics.AT_LEAST

    def test_direction_is_required(self) -> None:
        with pytest.raises(ValidationError):
            QualityMeasure.model_validate(
                {"name": "candidate_count", "column": "candidate_count", "threshold_semantics": "at_most"}
            )

    def test_threshold_semantics_is_required(self) -> None:
        with pytest.raises(ValidationError):
            QualityMeasure.model_validate(
                {"name": "candidate_count", "column": "candidate_count", "direction": "lower_is_better"}
            )

    def test_the_name_must_be_an_identifier(self) -> None:
        with pytest.raises(ValidationError):
            QualityMeasure.model_validate({**_CANDIDATE_COUNT, "name": "candidate count"})

    def test_an_unmeasured_bridge_declares_it(self) -> None:
        measure = QualityMeasure.model_validate({**_CANDIDATE_COUNT, "unmeasured_is_null": False})
        assert measure.unmeasured_is_null is False


class TestBridgeRefIsPlural:
    """one bridge, several measures on several scales."""

    def test_two_measures_on_different_scales(self) -> None:
        declared = BridgeRef.model_validate(
            {"relation": "match_union", "quality_measures": [_CANDIDATE_COUNT, _SOURCE_MATCH_TIER]}
        )
        assert len(declared.quality_measures) == 2
        assert declared.long_columns == ("quality_candidate_count", "quality_source_match_tier")

    def test_quality_measures_is_a_list(self) -> None:
        annotation = BridgeRef.model_fields["quality_measures"].annotation
        assert annotation == list[QualityMeasure]

    def test_no_model_in_this_module_carries_a_singular_quality_field(self) -> None:
        offenders = [
            name
            for name, obj in vars(bridge_module).items()
            if inspect.isclass(obj) and issubclass(obj, pydantic.BaseModel) and "quality" in obj.model_fields
        ]
        assert offenders == []

    def test_a_bridge_may_declare_no_measure(self) -> None:
        declared = BridgeRef.model_validate({"relation": "household"})
        assert declared.quality_measures == []
        assert declared.long_columns == ()

    def test_a_repeated_measure_name_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            BridgeRef.model_validate(
                {"relation": "match_union", "quality_measures": [_CANDIDATE_COUNT, _CANDIDATE_COUNT]}
            )

    def test_the_bridge_alias_is_what_predicates_reference(self) -> None:
        declared = BridgeRef.model_validate({"relation": "match_union", "alias": "mat"})
        assert declared.alias == "mat"

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            BridgeRef.model_validate({"relation": "match_union", "quality": "candidate_count"})


class TestIntersectAcrossDifferentlyBridgedUnits:
    """both measure sets are emitted; neither is coerced, rescaled, or dropped."""

    def test_the_union_emits_both_measure_sets(self) -> None:
        amazon = BridgeRef.model_validate({"relation": "match_union", "quality_measures": [_CANDIDATE_COUNT]})
        healthcare = BridgeRef.model_validate(
            {"relation": "knowwho_exec_mat", "quality_measures": [_SOURCE_MATCH_TIER]}
        )
        emitted = union_quality_measures([amazon, healthcare])
        assert [measure.long_column for measure in emitted] == [
            "quality_candidate_count",
            "quality_source_match_tier",
        ]

    def test_the_side_that_supplied_no_measure_is_null_by_declaration(self) -> None:
        amazon = BridgeRef.model_validate({"relation": "match_union", "quality_measures": [_CANDIDATE_COUNT]})
        expansion = BridgeRef.model_validate({"relation": "household"})
        emitted = union_quality_measures([amazon, expansion])
        assert len(emitted) == 1
        assert emitted[0].unmeasured_is_null is True

    def test_the_same_measure_from_two_bridges_reading_two_columns_is_one_output(self) -> None:
        left = BridgeRef.model_validate({"relation": "match_union", "quality_measures": [_CANDIDATE_COUNT]})
        right = BridgeRef.model_validate(
            {
                "relation": "knowwho_exec_mat",
                "quality_measures": [{**_CANDIDATE_COUNT, "column": "mat_candidate_count"}],
            }
        )
        emitted = union_quality_measures([left, right])
        assert [measure.name for measure in emitted] == ["candidate_count"]

    def test_conflicting_semantics_are_not_coerced(self) -> None:
        left = BridgeRef.model_validate({"relation": "match_union", "quality_measures": [_CANDIDATE_COUNT]})
        right = BridgeRef.model_validate(
            {
                "relation": "knowwho_exec_mat",
                "quality_measures": [{**_CANDIDATE_COUNT, "direction": "higher_is_better"}],
            }
        )
        with pytest.raises(ConflictingQualityMeasure):
            union_quality_measures([left, right])

    def test_conflicting_threshold_semantics_are_not_coerced(self) -> None:
        left = BridgeRef.model_validate({"relation": "match_union", "quality_measures": [_CANDIDATE_COUNT]})
        right = BridgeRef.model_validate(
            {
                "relation": "knowwho_exec_mat",
                "quality_measures": [{**_CANDIDATE_COUNT, "threshold_semantics": "at_least"}],
            }
        )
        with pytest.raises(ConflictingQualityMeasure):
            union_quality_measures([left, right])

    def test_the_union_of_no_bridge_is_empty(self) -> None:
        assert union_quality_measures([]) == ()

"""unit tests for the seven-name predicate namespace and its stage binding rules.

the load-bearing assertion in this module is that ``source`` and ``bridge``
are absent from the qualification stage's bindable set. binding a
qualification predicate to ``source.*`` names a pre-aggregate row that no
longer exists at that stage, which silently mis-binds the working-age
filter in every audience that uses one.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.datasources.definition import (
    BindingStage,
    Namespace,
    Reference,
    bindable_namespaces,
)


class TestNamespaceVocabulary:
    """the seven names, and nothing else."""

    def test_exactly_seven_names(self) -> None:
        assert {member.value for member in Namespace} == {
            "source",
            "bridge",
            "entity",
            "rel",
            "measure",
            "resolved",
            "param",
        }


class TestStageBinding:
    """which namespaces are bindable at which stage."""

    def test_qualification_excludes_source_and_bridge(self) -> None:
        bindable = bindable_namespaces(BindingStage.QUALIFICATION)
        assert Namespace.SOURCE not in bindable
        assert Namespace.BRIDGE not in bindable

    def test_qualification_binds_resolved_entity_rel_param(self) -> None:
        assert bindable_namespaces(BindingStage.QUALIFICATION) == frozenset(
            {
                Namespace.RESOLVED,
                Namespace.ENTITY,
                Namespace.REL,
                Namespace.PARAM,
            }
        )

    def test_resolution_excludes_resolved_and_measure(self) -> None:
        bindable = bindable_namespaces(BindingStage.RESOLUTION)
        assert Namespace.RESOLVED not in bindable
        assert Namespace.MEASURE not in bindable

    def test_resolution_binds_source_bridge_entity_rel_param(self) -> None:
        assert bindable_namespaces(BindingStage.RESOLUTION) == frozenset(
            {
                Namespace.SOURCE,
                Namespace.BRIDGE,
                Namespace.ENTITY,
                Namespace.REL,
                Namespace.PARAM,
            }
        )

    def test_measure_is_bindable_only_in_having(self) -> None:
        stages_binding_measure = {stage for stage in BindingStage if Namespace.MEASURE in bindable_namespaces(stage)}
        assert stages_binding_measure == {BindingStage.HAVING}

    def test_having_is_the_resolution_set_plus_measure(self) -> None:
        assert bindable_namespaces(BindingStage.HAVING) == bindable_namespaces(BindingStage.RESOLUTION) | frozenset(
            {Namespace.MEASURE}
        )


class TestReferenceParsing:
    """``Reference`` parses the authored text and re-emits it byte-exactly."""

    def test_two_segment_reference(self) -> None:
        ref = Reference(ref="entity.vb_voterbase_age")
        assert ref.namespace is Namespace.ENTITY
        assert ref.alias is None
        assert ref.name == "vb_voterbase_age"

    def test_rel_reference_carries_an_alias(self) -> None:
        ref = Reference(ref="rel.old_knowwho.pid")
        assert ref.namespace is Namespace.REL
        assert ref.alias == "old_knowwho"
        assert ref.name == "pid"

    def test_measure_reference(self) -> None:
        ref = Reference(ref="measure.sum_of_contributions")
        assert ref.namespace is Namespace.MEASURE
        assert ref.name == "sum_of_contributions"

    def test_param_reference(self) -> None:
        assert Reference(ref="param.run_year").namespace is Namespace.PARAM

    def test_resolved_reference(self) -> None:
        assert Reference(ref="resolved.record_year").namespace is Namespace.RESOLVED

    def test_round_trips_byte_exactly(self) -> None:
        assert Reference(ref="rel.cat.business").model_dump() == {"ref": "rel.cat.business"}

    def test_rejects_unknown_namespace(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            Reference(ref="facts.job_title")
        assert "facts" in str(excinfo.value)

    def test_rejects_bare_column(self) -> None:
        with pytest.raises(ValidationError):
            Reference(ref="job_title")

    def test_rejects_rel_without_an_alias(self) -> None:
        with pytest.raises(ValidationError):
            Reference(ref="rel.business")

    def test_rejects_three_segments_outside_rel(self) -> None:
        with pytest.raises(ValidationError):
            Reference(ref="source.schema.job_title")

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Reference.model_validate({"ref": "param.run_year", "alias": "x"})

    def test_is_bindable_at(self) -> None:
        source_ref = Reference(ref="source.job_title")
        assert source_ref.is_bindable_at(BindingStage.RESOLUTION) is True
        assert source_ref.is_bindable_at(BindingStage.QUALIFICATION) is False

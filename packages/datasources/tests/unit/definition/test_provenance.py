"""unit tests for :class:`ProvenanceSpec` and :class:`ProvenanceColumn`.

Provenance is a projection with its own column contract, and it is
AUTHORED rather than derived: the corpus carries three mutually
inconsistent versions of the contract (15 in the template DDL, 14 in the
readme, 16 in the Amazon bodies) and nine hand-written bodies that are
not recoverable from their unit's body -- one projects an aggregate that
appears nowhere in the unit query.

D9b resolves the contract to the DDL's 15 columns, and the fixture below
is that list. The template's own `linkedin_industries` / `linkedin_industry`
divergence is exercised too, because D9a's projection check fails on it
CORRECTLY and the first reader will call that a compiler bug.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.datasources.definition import (
    ArtifactKind,
    ArtifactRef,
    ArtifactScope,
    Measure,
    MeasureScope,
    ProvenanceColumn,
    ProvenanceContractViolation,
    ProvenanceSpec,
    RelationRef,
)

PINNED_CONTRACT: tuple[str, ...] = (
    "voterbase_id",
    "list_id",
    "influencer",
    "unit",
    "source_name",
    "type",
    "job_title",
    "job_level",
    "employer",
    "organization_bucketed",
    "linkedin_industries",
    "influence_score",
    "influencer_vb_id",
    "record_year",
    "candidate_count",
)


def _columns(names: tuple[str, ...]) -> list[ProvenanceColumn]:
    """build one provenance column per name, each projecting a reference.

    :param names: contract column names, in contract order
    :ptype names: tuple[str, ...]
    :returns: provenance columns in the same order
    :rtype: list[ProvenanceColumn]
    """
    return [ProvenanceColumn(name=name, expression={"ref": f"resolved.{name}"}) for name in names]


class TestProvenanceColumn:
    """one column of the pinned contract."""

    def test_carries_an_expression_and_an_optional_sql_type(self) -> None:
        column = ProvenanceColumn(name="type", expression={"literal": "employment"}, sql_type="VARCHAR(50)")
        assert column.name == "type"
        assert column.sql_type == "VARCHAR(50)"

    def test_a_bare_string_expression_is_a_reference(self) -> None:
        column = ProvenanceColumn(name="job_title", expression="source.job_title")
        assert column.references[0].ref == "source.job_title"

    def test_rejects_a_blank_name(self) -> None:
        with pytest.raises(ValidationError):
            ProvenanceColumn(name="   ", expression={"literal": 1})

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ProvenanceColumn.model_validate({"name": "unit", "expression": "resolved.unit", "encode": "lzo"})


class TestProvenanceSpec:
    """anchor, relations, grain, measures, and the required columns."""

    def test_carries_the_pinned_fifteen_column_contract(self) -> None:
        spec = ProvenanceSpec(grain=["voterbase_id", "list_id"], columns=_columns(PINNED_CONTRACT))
        assert spec.column_names == PINNED_CONTRACT

    def test_columns_are_required(self) -> None:
        with pytest.raises(ValidationError):
            ProvenanceSpec.model_validate({"grain": ["voterbase_id"]})

    def test_grain_is_required_because_every_body_is_an_aggregate_query(self) -> None:
        with pytest.raises(ValidationError):
            ProvenanceSpec.model_validate({"columns": [{"name": "unit", "expression": "resolved.unit"}]})

    def test_rejects_a_repeated_column_name(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ProvenanceSpec(grain=["voterbase_id"], columns=_columns(("unit", "unit")))
        assert "unit" in str(excinfo.value)

    def test_rejects_a_repeated_grain_key(self) -> None:
        with pytest.raises(ValidationError):
            ProvenanceSpec(grain=["voterbase_id", "voterbase_id"], columns=_columns(("unit",)))

    def test_anchors_on_the_long_artifact_of_this_definition(self) -> None:
        anchor = ArtifactRef(scope=ArtifactScope.THIS_DEFINITION, artifact=ArtifactKind.LONG)
        spec = ProvenanceSpec(anchor=anchor, grain=["voterbase_id"], columns=_columns(("unit",)))
        assert spec.anchor is not None
        assert spec.anchor.artifact is ArtifactKind.LONG

    def test_relations_are_not_inherited_from_the_resolution(self) -> None:
        # the source_archetype join is GATED at stage 1 and UNCONDITIONAL in
        # provenance, so the provenance body declares its own relations.
        spec = ProvenanceSpec(
            grain=["voterbase_id"],
            columns=_columns(("unit",)),
            relations=[RelationRef(relation="source_archetypes", alias="sa", join="left")],
        )
        assert [relation.alias for relation in spec.relations] == ["sa"]

    def test_projects_a_measure_the_unit_never_had(self) -> None:
        spec = ProvenanceSpec(
            grain=["voterbase_id"],
            columns=_columns(("unit",)),
            measures=[
                Measure(
                    name="contribution_sum",
                    expression="SUM(source.contribution_amt)",
                    grain=["voterbase_id", "list_id"],
                    scope=MeasureScope.RESOLUTION,
                )
            ],
        )
        assert [measure.name for measure in spec.measures] == ["contribution_sum"]

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ProvenanceSpec.model_validate(
                {"grain": ["voterbase_id"], "columns": [{"name": "unit", "expression": "resolved.unit"}], "ddl": "x"}
            )


class TestContractVerification:
    """D9a: the projection is verified against the PINNED contract."""

    def test_a_matching_projection_verifies(self) -> None:
        spec = ProvenanceSpec(grain=["voterbase_id"], columns=_columns(PINNED_CONTRACT))
        spec.verify_columns_against(PINNED_CONTRACT)

    def test_a_missing_column_names_itself(self) -> None:
        spec = ProvenanceSpec(grain=["voterbase_id"], columns=_columns(PINNED_CONTRACT[:-1]))
        with pytest.raises(ProvenanceContractViolation) as excinfo:
            spec.verify_columns_against(PINNED_CONTRACT)
        assert "candidate_count" in str(excinfo.value)

    def test_an_extra_column_names_itself(self) -> None:
        spec = ProvenanceSpec(grain=["voterbase_id"], columns=_columns((*PINNED_CONTRACT, "contribution_sum")))
        with pytest.raises(ProvenanceContractViolation) as excinfo:
            spec.verify_columns_against(PINNED_CONTRACT)
        assert "contribution_sum" in str(excinfo.value)

    def test_the_committed_templates_singular_alias_fails_correctly(self) -> None:
        # vb_rationale.sql.jinja2 declares `linkedin_industries` at :15 and the
        # body aliases `linkedin_industry` at :41. it works today only because
        # the INSERT is positional. the check fails on it, correctly.
        drifted = tuple("linkedin_industry" if name == "linkedin_industries" else name for name in PINNED_CONTRACT)
        spec = ProvenanceSpec(grain=["voterbase_id"], columns=_columns(drifted))
        with pytest.raises(ProvenanceContractViolation) as excinfo:
            spec.verify_columns_against(PINNED_CONTRACT)
        message = str(excinfo.value)
        assert "linkedin_industry" in message
        assert "linkedin_industries" in message

    def test_order_is_part_of_the_contract_because_the_insert_is_positional(self) -> None:
        swapped = (PINNED_CONTRACT[1], PINNED_CONTRACT[0], *PINNED_CONTRACT[2:])
        spec = ProvenanceSpec(grain=["voterbase_id"], columns=_columns(swapped))
        with pytest.raises(ProvenanceContractViolation) as excinfo:
            spec.verify_columns_against(PINNED_CONTRACT)
        assert "position" in str(excinfo.value)

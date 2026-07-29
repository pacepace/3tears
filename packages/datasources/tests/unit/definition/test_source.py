"""unit tests for the four source kinds, ``ArtifactRef``, and ``UpstreamPin``.

The load-bearing cases, each drawn from committed corpus evidence:

- ``LiteralEntities`` normalizes and REPORTS. Five committed
  ``voterbase_id``s in ``conservative_podcaster_hand_matches`` carry a
  leading space and would silently match nothing, and two sets repeat ids.
- ``RawSelect`` carries a required ``ProvenanceSpec`` (D9a) and a DECLARED
  parameter signature, because the prototype's renderer accepts an unused
  parameter and only fails on a missing one at render time.
- ``UpstreamPin`` expresses a POLICY, never a run id. A policy reference to
  a draft is refused; a DIRECT draft reference is accepted, because one
  real shipped pair is built the same day.
- ``ArtifactRef`` carries all three scopes, a datasource qualifier (D20),
  and a projection -- one real unit derives its whole allowlist from an
  upstream PROVENANCE artifact under a ``HAVING``.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from threetears.datasources.definition import (
    ArtifactKind,
    ArtifactProjection,
    ArtifactRef,
    ArtifactScope,
    ArtifactStage,
    DraftPolicyReference,
    ExclusionLevel,
    ExclusionSpec,
    FactSource,
    LiteralEntities,
    Membership,
    ParameterSignatureViolation,
    Predicate,
    ProjectionContractViolation,
    ProvenanceColumn,
    ProvenanceSpec,
    RawSelect,
    RelationRef,
    UpstreamPin,
    UpstreamPolicy,
    reject_policy_reference_to_draft,
)

_UPSTREAM_RUN = UUID("0192f3a0-0000-7000-8000-000000000001")


def _provenance() -> ProvenanceSpec:
    """minimal provenance spec a ``RawSelect`` may carry.

    :returns: provenance spec with one grain key and one column
    :rtype: ProvenanceSpec
    """
    return ProvenanceSpec(
        grain=["voterbase_id"],
        columns=[ProvenanceColumn(name="unit", expression={"literal": "academy_members"})],
    )


class TestFactSource:
    """the resolution's fact table, and its per-run schema."""

    def test_names_a_governed_relation(self) -> None:
        source = FactSource(relation="employment_facts")
        assert source.relation == "employment_facts"
        assert source.datasource is None

    def test_carries_a_datasource_qualifier(self) -> None:
        assert FactSource(relation="employment_facts", datasource="influencers-build").datasource == (
            "influencers-build"
        )

    def test_carries_a_per_run_schema_expression(self) -> None:
        source = FactSource(relation="employment_facts", schema_expr="param.release_schema")
        assert source.schema_expr is not None
        assert source.references[0].ref == "param.release_schema"

    def test_rejects_a_blank_relation(self) -> None:
        with pytest.raises(ValidationError):
            FactSource(relation="  ")

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            FactSource.model_validate({"relation": "employment_facts", "facts_table": "employment"})


class TestLiteralEntities:
    """the hand-match sets, their dirty ids, and their duplicates."""

    def test_holds_the_authored_ids_verbatim(self) -> None:
        entities = LiteralEntities(entity_ids=["NY-Y29454918132366", "FL-Y29454254039464"])
        assert entities.entity_ids == ["NY-Y29454918132366", "FL-Y29454254039464"]

    def test_whitespace_dirty_ids_normalize(self) -> None:
        # hand_matches.yaml:12 -- five ids with a leading space that would
        # silently match zero rows.
        entities = LiteralEntities(
            entity_ids=[
                " NY-Y29454918132366",
                " NY-18677111",
                " FL-Y29454254039464",
                " WI-14701230",
                " CA-20908829",
            ]
        )
        assert entities.normalized_ids == (
            "NY-Y29454918132366",
            "NY-18677111",
            "FL-Y29454254039464",
            "WI-14701230",
            "CA-20908829",
        )

    def test_whitespace_normalization_is_reported_not_silent(self) -> None:
        entities = LiteralEntities(entity_ids=[" NY-18677111", "CA-20908829"])
        assert entities.normalization.trimmed == (" NY-18677111",)
        assert entities.normalization.duplicates == ()
        assert entities.normalization.changed_anything is True

    def test_duplicates_are_reported_and_de_duplicated(self) -> None:
        # hand_matches.yaml:6 -- ND-534159 / ND-507322 / ND-24157 repeat.
        entities = LiteralEntities(entity_ids=["ND-534159", "ND-507322", "ND-534159", "MA-1813485"])
        assert entities.normalized_ids == ("ND-534159", "ND-507322", "MA-1813485")
        assert entities.normalization.duplicates == ("ND-534159",)

    def test_a_clean_set_reports_no_change(self) -> None:
        entities = LiteralEntities(entity_ids=["ND-534159", "MA-1813485"])
        assert entities.normalization.changed_anything is False

    def test_rejects_an_id_that_normalizes_to_nothing(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            LiteralEntities(entity_ids=["NY-18677111", "   "])
        assert "matches nothing" in str(excinfo.value)

    def test_rejects_an_empty_set(self) -> None:
        with pytest.raises(ValidationError):
            LiteralEntities(entity_ids=[])

    def test_carries_no_bridge_and_no_quality_measure(self) -> None:
        assert "bridge" not in LiteralEntities.model_fields
        assert "quality_measures" not in LiteralEntities.model_fields

    def test_round_trips_the_authored_form(self) -> None:
        entities = LiteralEntities(entity_ids=[" NY-18677111", "NY-18677111"])
        restored = LiteralEntities.model_validate(entities.model_dump(mode="json"))
        assert restored == entities


class TestRawSelect:
    """the escape hatch, closed by a required provenance and a signature."""

    def test_provenance_is_required(self) -> None:
        with pytest.raises(ValidationError):
            RawSelect.model_validate({"raw_sql": "SELECT 1", "projection": ["unit"]})

    def test_is_an_escape_hatch(self) -> None:
        assert RawSelect(raw_sql="SELECT 1", projection=["unit"], provenance=_provenance()).is_escape_hatch is True

    def test_rejects_a_blank_body(self) -> None:
        with pytest.raises(ValidationError):
            RawSelect(raw_sql="   ", projection=["unit"], provenance=_provenance())

    def test_projection_matching_the_contract_verifies(self) -> None:
        raw = RawSelect(
            raw_sql="SELECT unit, voterbase_id, list_id, candidate_count, record_year FROM x",
            projection=["unit", "voterbase_id", "list_id", "candidate_count", "record_year"],
            provenance=_provenance(),
        )
        raw.verify_projection_against(("unit", "voterbase_id", "list_id", "candidate_count", "record_year"))

    def test_the_readmes_four_column_contract_fails_against_the_templates_five(self) -> None:
        # readme.md:35-38 promises 4; 1_generate...:111-116 selects 5.
        raw = RawSelect(
            raw_sql="SELECT unit, voterbase_id, candidate_count, record_year FROM x",
            projection=["unit", "voterbase_id", "candidate_count", "record_year"],
            provenance=_provenance(),
        )
        with pytest.raises(ProjectionContractViolation) as excinfo:
            raw.verify_projection_against(("unit", "voterbase_id", "list_id", "candidate_count", "record_year"))
        assert "list_id" in str(excinfo.value)

    def test_declares_its_parameter_signature(self) -> None:
        raw = RawSelect(
            raw_sql="SELECT ... WHERE job_level = {{ job_level }}",
            projection=["unit"],
            parameters=["job_level"],
            provenance=_provenance(),
        )
        raw.verify_parameters_against(["job_level", "release_schema"])

    def test_a_signature_naming_an_undeclared_parameter_is_refused(self) -> None:
        raw = RawSelect(
            raw_sql="SELECT ... {{ job_level }}",
            projection=["unit"],
            parameters=["job_level"],
            provenance=_provenance(),
        )
        with pytest.raises(ParameterSignatureViolation) as excinfo:
            raw.verify_parameters_against(["release_schema"])
        assert "job_level" in str(excinfo.value)

    def test_rejects_a_repeated_parameter_name(self) -> None:
        with pytest.raises(ValidationError):
            RawSelect(
                raw_sql="SELECT 1",
                projection=["unit"],
                parameters=["job_level", "job_level"],
                provenance=_provenance(),
            )

    def test_rejects_a_repeated_projected_column(self) -> None:
        with pytest.raises(ValidationError):
            RawSelect(raw_sql="SELECT 1", projection=["unit", "unit"], provenance=_provenance())


class TestUpstreamPin:
    """a resolution POLICY, and the one place a run id is pinned."""

    def test_a_named_release_policy_carries_the_release(self) -> None:
        pin = UpstreamPin(policy=UpstreamPolicy.NAMED_RELEASE, release="2026-01")
        assert pin.is_policy_resolution is True

    def test_a_named_release_policy_without_a_release_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            UpstreamPin(policy=UpstreamPolicy.NAMED_RELEASE)

    def test_latest_release_carries_neither_release_nor_run(self) -> None:
        pin = UpstreamPin(policy=UpstreamPolicy.LATEST_RELEASE)
        assert pin.is_policy_resolution is True
        assert pin.run is None

    def test_latest_release_with_a_release_label_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            UpstreamPin(policy=UpstreamPolicy.LATEST_RELEASE, release="2026-01")

    def test_a_direct_draft_reference_pins_that_run(self) -> None:
        pin = UpstreamPin(policy=UpstreamPolicy.DRAFT_RUN, run=_UPSTREAM_RUN)
        assert pin.is_policy_resolution is False
        assert pin.run == _UPSTREAM_RUN

    def test_a_draft_reference_without_a_run_is_refused(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            UpstreamPin(policy=UpstreamPolicy.DRAFT_RUN)
        assert "pins that specific run" in str(excinfo.value)

    def test_a_policy_reference_to_a_draft_is_rejected(self) -> None:
        pin = UpstreamPin(policy=UpstreamPolicy.LATEST_RELEASE)
        with pytest.raises(DraftPolicyReference) as excinfo:
            reject_policy_reference_to_draft(pin, target_is_draft=True, dataset="universal_2026_core")
        assert "universal_2026_core" in str(excinfo.value)

    def test_a_direct_draft_reference_is_not_rejected(self) -> None:
        pin = UpstreamPin(policy=UpstreamPolicy.DRAFT_RUN, run=_UPSTREAM_RUN)
        reject_policy_reference_to_draft(pin, target_is_draft=True, dataset="universal_2026_core")

    def test_a_policy_reference_to_a_release_is_not_rejected(self) -> None:
        pin = UpstreamPin(policy=UpstreamPolicy.LATEST_RELEASE)
        reject_policy_reference_to_draft(pin, target_is_draft=False, dataset="universal_2026_core")


class TestArtifactRef:
    """three scopes, a datasource qualifier, and a projection."""

    def test_this_definition_names_a_unit_at_a_stage(self) -> None:
        ref = ArtifactRef(scope=ArtifactScope.THIS_DEFINITION, unit="knowwho_all", stage=ArtifactStage.RESOLVED)
        assert ref.unit == "knowwho_all"

    def test_this_definition_may_name_an_artifact_instead(self) -> None:
        # one real rollup is computed over the PROVENANCE artifact.
        ref = ArtifactRef(scope=ArtifactScope.THIS_DEFINITION, artifact=ArtifactKind.PROVENANCE)
        assert ref.artifact is ArtifactKind.PROVENANCE

    def test_this_definition_naming_a_unit_without_a_stage_is_refused(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ArtifactRef(scope=ArtifactScope.THIS_DEFINITION, unit="knowwho_all")
        assert "different sets" in str(excinfo.value)

    def test_this_definition_naming_neither_unit_nor_artifact_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactRef(scope=ArtifactScope.THIS_DEFINITION)

    def test_dataset_scope_names_the_dataset_and_the_artifact(self) -> None:
        ref = ArtifactRef(
            scope=ArtifactScope.DATASET,
            dataset="universal_2026_core",
            artifact=ArtifactKind.PROVENANCE,
            run=UpstreamPin(policy=UpstreamPolicy.LATEST_RELEASE),
        )
        assert ref.dataset == "universal_2026_core"

    def test_dataset_scope_without_an_artifact_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactRef(scope=ArtifactScope.DATASET, dataset="universal_2026_core")

    def test_dataset_scope_carrying_a_unit_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactRef(
                scope=ArtifactScope.DATASET,
                dataset="universal_2026_core",
                artifact=ArtifactKind.WIDE,
                unit="knowwho_all",
                stage=ArtifactStage.RESOLVED,
            )

    def test_external_scope_names_a_table_and_a_per_run_schema(self) -> None:
        ref = ArtifactRef(scope=ArtifactScope.EXTERNAL, table="source_archetypes", schema_expr="param.release_schema")
        assert ref.table == "source_archetypes"
        assert ref.references[0].ref == "param.release_schema"

    def test_external_scope_without_a_table_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactRef(scope=ArtifactScope.EXTERNAL)

    def test_carries_a_datasource_qualifier(self) -> None:
        ref = ArtifactRef(scope=ArtifactScope.EXTERNAL, table="modeling_frame_comm", datasource="influencers-read")
        assert ref.datasource == "influencers-read"

    def test_the_relationship_union_is_the_fifth_artifact(self) -> None:
        assert {member.value for member in ArtifactKind} == {
            "long",
            "qualified",
            "wide",
            "provenance",
            "relationship_union",
        }

    def test_an_upstream_provenance_read_under_a_having(self) -> None:
        # universal_2026_expansion/top_audience_companies:3-8.
        ref = ArtifactRef(
            scope=ArtifactScope.DATASET,
            dataset="universal_2026_core",
            artifact=ArtifactKind.PROVENANCE,
            run=UpstreamPin(policy=UpstreamPolicy.NAMED_RELEASE, release="2026-01"),
            projection=ArtifactProjection(
                columns=[{"expression": "resolved.employer", "alias": "employer"}],
                group_by=["resolved.employer"],
                having={"compare": {"left": {"arith": "COUNT(*)"}, "op": ">=", "right": {"literal": 25}}},
            ),
        )
        assert ref.projection is not None
        assert ref.projection.having is not None

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactRef.model_validate({"scope": "external", "table": "x", "schema": "public"})


class TestArtifactProjection:
    """the GROUP BY / HAVING / column select carried on a reference."""

    def test_a_projection_declaring_nothing_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactProjection()

    def test_a_positional_group_ordinal_must_be_in_range(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ArtifactProjection(columns=[{"expression": "resolved.unit"}], group_by=[{"literal": 2}])
        assert "projects" in str(excinfo.value)

    def test_a_positional_group_ordinal_in_range_is_accepted(self) -> None:
        projection = ArtifactProjection(columns=[{"expression": "resolved.unit"}], group_by=[{"literal": 1}])
        assert len(projection.group_by) == 1


class TestSourceRefUsability:
    """DSM-01D-01: a source, a relation target, and a predicate subquery."""

    def test_usable_as_a_relation_target(self) -> None:
        relation = RelationRef(
            relation=ArtifactRef(scope=ArtifactScope.EXTERNAL, table="source_archetypes"),
            alias="sa",
            join="left",
        )
        assert isinstance(relation.relation, ArtifactRef)

    def test_usable_inside_a_predicate_subquery(self) -> None:
        predicate = Predicate(
            membership=Membership(
                expression="resolved.voterbase_id",
                source=ArtifactRef(
                    scope=ArtifactScope.DATASET, dataset="amz_universe_2024", artifact=ArtifactKind.WIDE
                ),
            )
        )
        assert predicate.membership is not None
        assert predicate.references[0].ref == "resolved.voterbase_id"

    def test_usable_as_an_exclusion_subtrahend(self) -> None:
        exclusion = ExclusionSpec(
            subtrahends=[ArtifactRef(scope=ArtifactScope.THIS_DEFINITION, unit="prior", stage=ArtifactStage.RESOLVED)],
            key_columns=["voterbase_id"],
            level=ExclusionLevel.PRE_AGGREGATE,
            stage=ArtifactStage.RESOLVED,
        )
        assert exclusion.subtrahends[0].unit == "prior"

    def test_a_membership_over_a_literal_value_list(self) -> None:
        # industries NOT IN ('a', 'b') -- the corpus's authored operator.
        predicate = Predicate(
            membership=Membership(
                expression="rel.li.industries",
                negate=True,
                values=[{"literal": "Retail"}, {"literal": "Wholesale"}],
            )
        )
        assert predicate.membership is not None
        assert predicate.membership.negate is True

    def test_a_membership_naming_neither_values_nor_source_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Membership(expression="rel.li.industries")

    def test_a_membership_naming_both_values_and_source_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Membership(
                expression="rel.li.industries",
                values=[{"literal": "Retail"}],
                source=LiteralEntities(entity_ids=["NY-1"]),
            )

    def test_a_membership_over_a_literal_entity_set(self) -> None:
        membership = Membership(
            expression="resolved.voterbase_id",
            source=LiteralEntities(entity_ids=[" NY-18677111"]),
        )
        assert isinstance(membership.source, LiteralEntities)

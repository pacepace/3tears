"""unit tests for :class:`Expansion` and :class:`DatasetDefinition`.

The definition-level invariants, each closing a hole the corpus carries:

- **a definition may span datasources** (D20). It names a primary, where
  unqualified names bind, plus further catalogs. The binding constraint is
  per-STATEMENT execution and is enforced by the compiler, not here.
- **``all_prior`` expands at parse time** (D7a), so "prior" means the
  authored unit order and the chosen edges land in the content hash rather
  than in ``os.listdir()``.
- **a unit named by no qualification arm is a validation error** (P1/1.6):
  ``department_of_commerce`` is emitted and appears in no arm, so every one
  of its rows is dropped -- not filtered, not errored.
- **the definition carries no platform state** (D21): no ``visibility``, no
  ``retention_days``, no dataset grant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from threetears.datasources.definition import (
    ArtifactKind,
    ArtifactScope,
    ArtifactStage,
    DatasetDefinition,
    Expansion,
    ProvenanceColumn,
    ProvenanceSpec,
    RelationRef,
    UnqualifiedUnits,
)

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "definition_minimal.json"


def _payload() -> dict[str, object]:
    """load the committed minimal definition fixture.

    :returns: raw fixture payload
    :rtype: dict[str, object]
    """
    with _FIXTURE.open(encoding="utf-8") as handle:
        loaded: dict[str, object] = json.load(handle)
    return loaded


@pytest.fixture(name="definition")
def _definition() -> DatasetDefinition:
    """the committed fixture, validated.

    :returns: validated definition
    :rtype: DatasetDefinition
    """
    return DatasetDefinition.model_validate(_payload())


class TestExpansion:
    """the household walk, and which member brought each one in."""

    def test_carries_an_edge_and_a_scope(self) -> None:
        expansion = Expansion(
            name="householders",
            edge=RelationRef(relation="household", alias="hh", join="inner"),
            member_column="householder_voterbase_id",
            applies_to=["knowwho_all", "executive_coworkers_of_linkedin_execs"],
        )
        assert expansion.exclude_existing is True

    def test_an_empty_applies_to_covers_every_unit(self) -> None:
        expansion = Expansion(
            name="spouses",
            edge=RelationRef(relation="household", alias="hh", join="inner"),
            member_column="householder_voterbase_id",
        )
        assert expansion.applies_to == []
        assert expansion.covers("any_unit") is True

    def test_a_scoped_expansion_covers_only_its_units(self) -> None:
        expansion = Expansion(
            name="householders",
            edge=RelationRef(relation="household", alias="hh", join="inner"),
            member_column="householder_voterbase_id",
            applies_to=["knowwho_all"],
        )
        assert expansion.covers("knowwho_all") is True
        assert expansion.covers("donors") is False

    def test_carries_the_expansion_provenance(self) -> None:
        expansion = Expansion(
            name="householders",
            edge=RelationRef(relation="household", alias="hh", join="inner"),
            member_column="householder_voterbase_id",
            provenance=ProvenanceSpec(
                grain=["voterbase_id"],
                columns=[
                    ProvenanceColumn(name="fact", expression="rel.hh.influencer_voterbase_id"),
                    ProvenanceColumn(name="fact_type", expression={"literal": "householders of"}),
                ],
            ),
        )
        assert expansion.provenance is not None
        assert expansion.provenance.column_names == ("fact", "fact_type")

    def test_rejects_a_repeated_unit_in_applies_to(self) -> None:
        with pytest.raises(ValidationError):
            Expansion(
                name="householders",
                edge=RelationRef(relation="household", alias="hh", join="inner"),
                member_column="householder_voterbase_id",
                applies_to=["knowwho_all", "knowwho_all"],
            )

    def test_rejects_a_blank_name(self) -> None:
        with pytest.raises(ValidationError):
            Expansion(
                name="  ",
                edge=RelationRef(relation="household", alias="hh", join="inner"),
                member_column="householder_voterbase_id",
            )

    def test_rejects_a_member_column_the_edge_joins_on(self) -> None:
        with pytest.raises(ValidationError, match="member_column"):
            Expansion(
                name="householders",
                edge=RelationRef(
                    relation="household",
                    alias="hh",
                    join="inner",
                    on={"compare": {"left": "rel.hh.influencer_voterbase_id", "op": "=", "right": "resolved.voterbase_id"}},
                ),
                member_column="influencer_voterbase_id",
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Expansion.model_validate(
                {
                    "name": "householders",
                    "edge": {"relation": "household", "alias": "hh", "join": "inner"},
                    "include_householders": True,
                }
            )


class TestDatasetDefinition:
    """the top-level artifact."""

    def test_the_committed_fixture_validates(self, definition: DatasetDefinition) -> None:
        assert definition.name == "universal_2026_expansion"
        assert [unit.name for unit in definition.units] == ["academy_members", "top_audience_companies"]

    def test_carries_a_primary_and_further_datasources(self, definition: DatasetDefinition) -> None:
        assert definition.datasource == "influencers-build"
        assert definition.additional_datasources == ["influencers-read"]
        assert definition.referenced_datasources == ("influencers-build", "influencers-read")

    def test_rejects_a_further_datasource_repeating_the_primary(self) -> None:
        payload = _payload()
        payload["additional_datasources"] = ["influencers-build"]
        with pytest.raises(ValidationError) as excinfo:
            DatasetDefinition.model_validate(payload)
        assert "influencers-build" in str(excinfo.value)

    def test_rejects_a_repeated_further_datasource(self) -> None:
        payload = _payload()
        payload["additional_datasources"] = ["influencers-read", "influencers-read"]
        with pytest.raises(ValidationError):
            DatasetDefinition.model_validate(payload)

    def test_units_are_required(self) -> None:
        payload = _payload()
        payload["units"] = []
        with pytest.raises(ValidationError):
            DatasetDefinition.model_validate(payload)

    def test_rejects_a_duplicate_unit_label(self) -> None:
        payload = _payload()
        units = payload["units"]
        assert isinstance(units, list)
        units[1]["name"] = "academy_members"
        with pytest.raises(ValidationError) as excinfo:
            DatasetDefinition.model_validate(payload)
        assert "academy_members" in str(excinfo.value)

    def test_a_unit_named_by_no_qualification_arm_is_refused(self) -> None:
        payload = _payload()
        qualification = payload["qualification"]
        assert isinstance(qualification, list)
        qualification[0]["applies_to"] = ["academy_members"]
        with pytest.raises((ValidationError, UnqualifiedUnits)) as excinfo:
            DatasetDefinition.model_validate(payload)
        assert "top_audience_companies" in str(excinfo.value)

    def test_all_prior_expands_at_parse_time(self, definition: DatasetDefinition) -> None:
        exclusion = definition.units[1].exclude
        assert exclusion is not None
        assert exclusion.all_prior is True
        assert [handle.unit for handle in exclusion.subtrahends] == ["academy_members"]
        assert exclusion.is_expanded is True

    def test_the_expanded_edges_carry_the_declared_stage(self, definition: DatasetDefinition) -> None:
        exclusion = definition.units[1].exclude
        assert exclusion is not None
        assert exclusion.subtrahends[0].scope is ArtifactScope.THIS_DEFINITION
        assert exclusion.subtrahends[0].stage is ArtifactStage.RESOLVED

    def test_expansion_is_idempotent_across_a_round_trip(self, definition: DatasetDefinition) -> None:
        restored = DatasetDefinition.model_validate(json.loads(definition.model_dump_json()))
        exclusion = restored.units[1].exclude
        assert exclusion is not None
        assert [handle.unit for handle in exclusion.subtrahends] == ["academy_members"]

    def test_artifacts_are_required_and_unique(self) -> None:
        payload = _payload()
        artifacts = payload["artifacts"]
        assert isinstance(artifacts, list)
        artifacts.append({"artifact": "long", "columns": ["unit"]})
        with pytest.raises(ValidationError) as excinfo:
            DatasetDefinition.model_validate(payload)
        assert "long" in str(excinfo.value)

    def test_at_most_one_artifact_is_delivered(self) -> None:
        payload = _payload()
        artifacts = payload["artifacts"]
        assert isinstance(artifacts, list)
        artifacts[0]["delivered"] = True
        with pytest.raises(ValidationError):
            DatasetDefinition.model_validate(payload)

    def test_the_delivered_artifact_is_reachable(self, definition: DatasetDefinition) -> None:
        assert definition.delivered_artifact is not None
        assert definition.delivered_artifact.artifact is ArtifactKind.WIDE

    def test_a_rollup_naming_no_declared_unit_is_refused(self) -> None:
        payload = _payload()
        rollups = payload["rollups"]
        assert isinstance(rollups, list)
        rollups[0]["members"] = ["department_of_commerce"]
        with pytest.raises(ValidationError) as excinfo:
            DatasetDefinition.model_validate(payload)
        assert "department_of_commerce" in str(excinfo.value)

    def test_an_expansion_scoped_to_no_declared_unit_is_refused(self) -> None:
        payload = _payload()
        expansions = payload["expansions"]
        assert isinstance(expansions, list)
        expansions[0]["applies_to"] = ["department_of_commerce"]
        with pytest.raises(ValidationError) as excinfo:
            DatasetDefinition.model_validate(payload)
        assert "department_of_commerce" in str(excinfo.value)

    def test_a_raw_select_signature_naming_an_undeclared_parameter_is_refused(self) -> None:
        payload = _payload()
        units = payload["units"]
        assert isinstance(units, list)
        units[0]["resolutions"][0]["source"]["parameters"] = ["job_level"]
        with pytest.raises(ValidationError) as excinfo:
            DatasetDefinition.model_validate(payload)
        assert "job_level" in str(excinfo.value)

    def test_carries_no_platform_state(self) -> None:
        # D21 and D1: visibility, dataset grants, and retention live on the
        # dataset record beside the definition, never inside it.
        for absent in ("visibility", "grants", "retention_days", "customer_id", "content_hash"):
            assert absent not in DatasetDefinition.model_fields

    def test_rejects_authored_platform_state(self) -> None:
        payload = _payload()
        payload["visibility"] = "public"
        with pytest.raises(ValidationError):
            DatasetDefinition.model_validate(payload)

    def test_the_content_hash_is_derived_never_authored(self) -> None:
        payload = _payload()
        payload["content_hash"] = "deadbeef"
        with pytest.raises(ValidationError):
            DatasetDefinition.model_validate(payload)

    def test_the_version_is_authored_but_not_hashed(self, definition: DatasetDefinition) -> None:
        assert definition.version == 1
        assert "version" in DatasetDefinition.hash_excluded_fields


_RECORD_YEAR_SENTINELS: dict[str, object] = {
    "target": "resolved.record_year",
    "sentinels": [
        {
            "kind": "value",
            "value": -1,
            "meaning": "no source record year; the custom unit projects the constant",
            "effect": "widens_predicate",
        },
        {
            "kind": "null",
            "meaning": "unit declares no facts_table, so the template emits NULL::int",
            "effect": "drops_row",
        },
    ],
}


class TestSentinelBindings:
    """F-02: the corpus's flagship sentinel is a COLUMN, not a parameter.

    ``record_year = -1`` makes the working-age ceiling a no-op -- at
    ``run_year = 2025`` the bound becomes ``age < 2096``
    (``sql_templates/2_filtered_universe.sql.jinja2:22``, seven Amazon
    sites). ``record_year = NULL::int`` makes the same comparison ``NULL``
    and drops EVERY row of the unit
    (``1_generate_audience_units_table.sql.jinja2:67-68``). Opposite
    failures, both silent, and ``ParameterSpec.sentinels`` reaches
    neither because ``record_year`` is not a parameter.
    """

    def test_a_definition_declares_no_sentinel_binding_by_default(self, definition: DatasetDefinition) -> None:
        assert definition.sentinel_bindings == []

    def test_both_record_year_sentinels_are_declarable_on_one_column(self) -> None:
        payload = _payload()
        payload["sentinel_bindings"] = [_RECORD_YEAR_SENTINELS]
        loaded = DatasetDefinition.model_validate(payload)
        assert loaded.sentinel_bindings[0].target.ref == "resolved.record_year"
        assert [sentinel.effect.value for sentinel in loaded.sentinel_bindings[0].sentinels] == [
            "widens_predicate",
            "drops_row",
        ]

    def test_one_column_carries_one_declared_domain(self) -> None:
        payload = _payload()
        payload["sentinel_bindings"] = [_RECORD_YEAR_SENTINELS, _RECORD_YEAR_SENTINELS]
        with pytest.raises(ValidationError) as excinfo:
            DatasetDefinition.model_validate(payload)
        assert "resolved.record_year" in str(excinfo.value)

    def test_a_parameter_sentinel_belongs_on_the_parameter(self) -> None:
        payload = _payload()
        payload["sentinel_bindings"] = [
            {
                "target": "param.vf_suffix",
                "sentinels": [{"kind": "null", "meaning": "no suffix supplied", "effect": "unknown"}],
            }
        ]
        with pytest.raises(ValidationError) as excinfo:
            DatasetDefinition.model_validate(payload)
        assert "ParameterSpec.sentinels" in str(excinfo.value)

    def test_a_warehouse_column_sentinel_belongs_to_the_schema_layer(self) -> None:
        payload = _payload()
        payload["sentinel_bindings"] = [
            {
                "target": "source.record_year",
                "sentinels": [{"kind": "value", "value": -1, "meaning": "unknown", "effect": "unknown"}],
            }
        ]
        with pytest.raises(ValidationError) as excinfo:
            DatasetDefinition.model_validate(payload)
        assert "resolved.*" in str(excinfo.value)

    def test_the_binding_survives_a_round_trip(self) -> None:
        payload = _payload()
        payload["sentinel_bindings"] = [_RECORD_YEAR_SENTINELS]
        loaded = DatasetDefinition.model_validate(payload)
        assert DatasetDefinition.model_validate(json.loads(loaded.model_dump_json())) == loaded


#: the name of the corpus's subtrahend-only set
#: (``uhg_policymakers/sql/full_uhg_audience.sql:85-96``, removed at
#: ``:98-108``). It exists to be taken away and is delivered to nobody.
_HELPER_UNIT = "uhg_staff"


def _payload_with_a_subtrahend_only_unit() -> dict[str, object]:
    """author the corpus's helper set exactly as the definition copes.

    Declared as a unit, because the model's only subtraction operators
    are ``Unit.exclude`` and a ``SetExpr`` ``difference`` over dataset
    TERMS, and a term is a unit, a rollup, an upstream, or a nested
    expression. Then kept out of both rollups and out of the wide
    artifact's column set BY HAND, and named in a qualification arm with
    no predicate so it survives the stage it exists to feed.

    :returns: raw definition payload carrying the helper unit
    :rtype: dict[str, object]
    """
    payload = _payload()
    units: list[dict[str, object]] = list(payload["units"])  # type: ignore[arg-type]
    delivered = [unit["name"] for unit in units]
    units.append(
        {
            "name": _HELPER_UNIT,
            "resolutions": [
                {
                    "source": {"relation": "employment"},
                    "predicate": {
                        "any_of": [
                            {"compare": {"left": "source.employer", "op": "LIKE", "right": {"literal": "%uhg%"}}},
                            {"compare": {"left": "source.employer", "op": "LIKE", "right": {"literal": "%optum%"}}},
                        ]
                    },
                }
            ],
        }
    )
    payload["units"] = units
    payload["qualification"] = [
        {
            "name": "candidate_count = 1",
            "applies_to": delivered,
            "predicate": {"compare": {"left": "resolved.candidate_count", "op": "=", "right": {"literal": 1}}},
        },
        {"name": "the staff subtrahend, unfiltered", "applies_to": [_HELPER_UNIT]},
    ]
    payload["composition"] = {
        "op": "difference",
        "terms": [
            {"name": "policymakers", "rollup": "opinion_elites"},
            {"name": "staff", "unit": _HELPER_UNIT},
        ],
    }
    return payload


class TestASubtrahendOnlySetHasToBeDeclaredAsAUnit:
    """F-09: expressible, at a cost three unenforced hand edits carry.

    ``full_uhg_audience.sql:85-96`` builds ``audiences.uhg_staff_20250618``
    -- people whose employer matches ``%uhg%``, ``%optum%``, ... -- and
    ``:98-108`` removes them from the delivered table. A "remove these
    people at the end" set is not a unit, a rollup, an upstream, or a
    nested expression until it is DECLARED as a unit, at which point it
    emits long rows and is indistinguishable from a delivered one.

    The recorded verdict is expressible, extension optional. These tests
    hold the coping strategy working, and hold the cost visible: no field
    marks a unit as not-delivered, so a ``Unit.delivered`` or a
    ``SetTerm.kind: helper`` landing later fails the last test here and
    forces the record to be re-decided rather than quietly outgrown.
    """

    def test_the_helper_unit_is_authorable_and_the_definition_validates(self) -> None:
        """the whole coping strategy, end to end.

        :returns: none
        :rtype: None
        """
        definition = DatasetDefinition.model_validate(_payload_with_a_subtrahend_only_unit())
        assert _HELPER_UNIT in [unit.name for unit in definition.units]
        assert definition.composition is not None
        assert [term.unit or term.rollup for term in definition.composition.terms] == [
            "opinion_elites",
            _HELPER_UNIT,
        ]

    def test_a_qualification_arm_with_no_predicate_carries_it_through(self) -> None:
        """it must survive qualification, or it subtracts nothing.

        An unqualified unit is a validation error (F-01), so the helper
        needs an arm -- and the arm has to carry no predicate, since any
        predicate would shrink the set being removed.

        :returns: none
        :rtype: None
        """
        definition = DatasetDefinition.model_validate(_payload_with_a_subtrahend_only_unit())
        arms = [arm for arm in definition.qualification if arm.applies_to == [_HELPER_UNIT]]
        assert len(arms) == 1
        assert arms[0].predicate is None

    def test_dropping_the_arm_is_caught_rather_than_silently_emptying_it(self) -> None:
        """the one hand edit the model DOES enforce.

        :returns: none
        :rtype: None
        """
        payload = _payload_with_a_subtrahend_only_unit()
        payload["qualification"] = [arm for arm in payload["qualification"] if arm["applies_to"] != [_HELPER_UNIT]]  # type: ignore[union-attr,index]
        with pytest.raises((ValidationError, UnqualifiedUnits)) as excinfo:
            DatasetDefinition.model_validate(payload)
        assert _HELPER_UNIT in str(excinfo.value)

    def test_keeping_it_out_of_the_rollups_and_the_wide_columns_is_unenforced(self) -> None:
        """the two hand edits nothing checks, stated rather than assumed.

        Naming the helper in a rollup and in the delivered column set is
        accepted today, which is exactly how a future definition loses its
        subtrahend into the deliverable. When ``Unit.delivered`` or a
        helper term kind lands, this fails, and F-09's record moves from
        "extension optional" to closed.

        :returns: none
        :rtype: None
        """
        payload = _payload_with_a_subtrahend_only_unit()
        rollups: list[dict[str, object]] = list(payload["rollups"])  # type: ignore[arg-type]
        rollups[0] = {**rollups[0], "members": [*rollups[0]["members"], _HELPER_UNIT]}  # type: ignore[misc]
        payload["rollups"] = rollups
        artifacts: list[dict[str, object]] = list(payload["artifacts"])  # type: ignore[arg-type]
        payload["artifacts"] = [
            {**artifact, "columns": [*artifact["columns"], _HELPER_UNIT]}  # type: ignore[misc]
            if artifact["artifact"] == "wide"
            else artifact
            for artifact in artifacts
        ]
        leaked = DatasetDefinition.model_validate(payload)
        wide = next(artifact for artifact in leaked.artifacts if artifact.artifact is ArtifactKind.WIDE)
        assert _HELPER_UNIT in wide.columns
        assert _HELPER_UNIT in leaked.rollups[0].members

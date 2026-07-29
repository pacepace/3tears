"""unit tests for the content hash.

The hash IS the version, so every property below is a correctness bar
rather than a nicety:

1. **invariant under field reordering** -- otherwise a cosmetic edit mints
   a version and fires ``definition_drift`` over every historical run;
2. **invariant across serialise then deserialise** -- otherwise a
   store-and-load cycle mints a version by itself. This is the hazard the
   build plan records: ``LiteralExpression`` did not round-trip, because
   ``Decimal`` and ``str`` serialise to the same JSON scalar and the type
   tag was lost;
3. **invariant under re-pinning the upstream run** -- otherwise every
   referenced draft becomes permanently unreapable, since versions are
   immutable;
4. **invariant under a grant or layout change** -- policy and tuning are
   not semantics;
5. **it MOVES on any semantic edit** -- without which the first four are
   satisfiable by a constant.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from threetears.datasources.definition import (
    ArtifactKind,
    ArtifactRef,
    ArtifactScope,
    ArtifactSpec,
    ColumnEncoding,
    Comparison,
    DatasetDefinition,
    DerivedColumn,
    GranteeKind,
    GrantSpec,
    LiteralEntities,
    LiteralExpression,
    LiteralType,
    PhysicalLayout,
    UpstreamPin,
    UpstreamPolicy,
    canonical_content,
    content_hash,
)

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "definition_minimal.json"
_ANOTHER_RUN = UUID("0192f3a0-0000-7000-8000-00000000beef")


def _payload() -> dict[str, object]:
    """load the committed minimal definition fixture.

    :returns: raw fixture payload
    :rtype: dict[str, object]
    """
    with _FIXTURE.open(encoding="utf-8") as handle:
        loaded: dict[str, object] = json.load(handle)
    return loaded


def _edited(**changes: object) -> DatasetDefinition:
    """validate the fixture with top-level keys replaced.

    :param changes: top-level keys to replace
    :ptype changes: object
    :returns: validated definition
    :rtype: DatasetDefinition
    """
    payload = _payload()
    payload.update(changes)
    return DatasetDefinition.model_validate(payload)


@pytest.fixture(name="definition")
def _definition() -> DatasetDefinition:
    """the committed fixture, validated.

    :returns: validated definition
    :rtype: DatasetDefinition
    """
    return DatasetDefinition.model_validate(_payload())


class TestLiteralTypeIsCarried:
    """the hash-instability hazard, closed at its source."""

    def test_a_text_literal_round_trips_as_text(self) -> None:
        literal = LiteralExpression(literal="5")
        restored = LiteralExpression.model_validate(json.loads(json.dumps(literal.model_dump(mode="json"))))
        assert restored == literal
        assert restored.literal_type is LiteralType.TEXT
        assert isinstance(restored.literal, str)

    def test_a_decimal_literal_round_trips_as_a_decimal(self) -> None:
        literal = LiteralExpression(literal=Decimal("5"))
        restored = LiteralExpression.model_validate(json.loads(json.dumps(literal.model_dump(mode="json"))))
        assert restored == literal
        assert restored.literal_type is LiteralType.DECIMAL
        assert isinstance(restored.literal, Decimal)

    def test_text_five_and_decimal_five_are_different_definitions(self) -> None:
        # a real definition contains `entertainment = '1.0'` -- a string
        # compared to a numeric column. losing the type tag made those two
        # hash identically, which is the correctness bug.
        text = Comparison(left="entity.entertainment", op="=", right={"literal": "1.0"})
        number = Comparison(left="entity.entertainment", op="=", right={"literal": Decimal("1.0")})
        assert content_hash(text) != content_hash(number)

    def test_every_scalar_kind_round_trips(self) -> None:
        for value, expected in (
            (True, LiteralType.BOOLEAN),
            (5, LiteralType.INTEGER),
            (Decimal("1.5"), LiteralType.DECIMAL),
            ("abc", LiteralType.TEXT),
            (None, LiteralType.NULL),
        ):
            literal = LiteralExpression(literal=value)
            restored = LiteralExpression.model_validate(json.loads(json.dumps(literal.model_dump(mode="json"))))
            assert restored.literal_type is expected
            assert restored == literal

    def test_a_float_is_carried_as_a_decimal_never_a_binary_float(self) -> None:
        assert LiteralExpression.model_validate({"literal": 1.5}).literal == Decimal("1.5")

    def test_an_authored_type_tag_coerces_the_scalar(self) -> None:
        literal = LiteralExpression.model_validate({"literal": "5", "literal_type": "decimal"})
        assert literal.literal == Decimal("5")

    def test_a_type_tag_disagreeing_with_the_scalar_is_refused(self) -> None:
        with pytest.raises(ValueError, match="null"):
            LiteralExpression.model_validate({"literal": "5", "literal_type": "null"})


class TestHashIsStable:
    """properties 1 to 4."""

    def test_invariant_under_field_reordering(self, definition: DatasetDefinition) -> None:
        payload = _payload()
        reordered = dict(reversed(list(payload.items())))
        assert DatasetDefinition.model_validate(reordered).content_hash == definition.content_hash

    def test_invariant_across_serialise_then_deserialise(self, definition: DatasetDefinition) -> None:
        round_tripped = DatasetDefinition.model_validate(json.loads(definition.model_dump_json()))
        assert round_tripped.content_hash == definition.content_hash

    def test_the_round_trip_is_structurally_identical_too(self, definition: DatasetDefinition) -> None:
        # a hash that matches over structurally different models would be
        # hiding the type loss rather than fixing it.
        assert DatasetDefinition.model_validate(json.loads(definition.model_dump_json())) == definition

    def test_invariant_under_re_pinning_an_upstream_run(self, definition: DatasetDefinition) -> None:
        payload = _payload()
        units = payload["units"]
        assert isinstance(units, list)
        units[1]["resolutions"][0]["source"]["run"] = {"policy": "latest_release", "run": str(_ANOTHER_RUN)}
        assert DatasetDefinition.model_validate(payload).content_hash == definition.content_hash

    def test_invariant_under_changing_a_draft_pin(self, definition: DatasetDefinition) -> None:
        payload = _payload()
        units = payload["units"]
        assert isinstance(units, list)
        units[1]["resolutions"][0]["source"]["run"] = {"policy": "draft_run", "run": str(_ANOTHER_RUN)}
        first = DatasetDefinition.model_validate(payload).content_hash
        units[1]["resolutions"][0]["source"]["run"] = {
            "policy": "draft_run",
            "run": "0192f3a0-0000-7000-8000-0000000000ff",
        }
        assert DatasetDefinition.model_validate(payload).content_hash == first
        assert first != definition.content_hash

    def test_grants_not_hashed(self, definition: DatasetDefinition) -> None:
        delivery = definition.delivery.model_copy(
            update={"grants": [GrantSpec(grantee_kind=GranteeKind.USER, grantee="ehirschfeld")]}
        )
        assert definition.model_copy(update={"delivery": delivery}).content_hash == definition.content_hash

    def test_layout_not_hashed(self, definition: DatasetDefinition) -> None:
        delivery = definition.delivery.model_copy(update={"layout": PhysicalLayout(sortkey=["list_id"])})
        assert definition.model_copy(update={"delivery": delivery}).content_hash == definition.content_hash

    def test_artifact_layout_not_hashed(self, definition: DatasetDefinition) -> None:
        artifacts = [
            artifact.model_copy(update={"layout": PhysicalLayout(encodings={"voterbase_id": ColumnEncoding.ZSTD})})
            if artifact.artifact is ArtifactKind.WIDE
            else artifact
            for artifact in definition.artifacts
        ]
        assert definition.model_copy(update={"artifacts": artifacts}).content_hash == definition.content_hash

    def test_the_version_number_is_not_hashed(self, definition: DatasetDefinition) -> None:
        assert _edited(version=97).content_hash == definition.content_hash

    def test_normalizing_a_dirty_literal_id_does_not_mint_a_version(self) -> None:
        # whitespace cleanup produces the identical audience, so it is a
        # cosmetic edit and must not fire drift over every historical run.
        dirty = LiteralEntities(entity_ids=[" NY-18677111", "CA-20908829", "NY-18677111"])
        clean = LiteralEntities(entity_ids=["NY-18677111", "CA-20908829"])
        assert content_hash(dirty) == content_hash(clean)

    def test_adding_a_genuinely_new_literal_id_does_mint_a_version(self) -> None:
        two = LiteralEntities(entity_ids=["NY-18677111", "CA-20908829"])
        three = LiteralEntities(entity_ids=["NY-18677111", "CA-20908829", "WI-14701230"])
        assert content_hash(two) != content_hash(three)


class TestHashMovesOnASemanticEdit:
    """property 5, the one people forget."""

    def test_renaming_a_unit_changes_it(self, definition: DatasetDefinition) -> None:
        payload = _payload()
        units = payload["units"]
        assert isinstance(units, list)
        units[1]["name"] = "renamed"
        assert DatasetDefinition.model_validate(payload).content_hash != definition.content_hash

    def test_changing_the_exclusion_stage_changes_it(self, definition: DatasetDefinition) -> None:
        payload = _payload()
        units = payload["units"]
        assert isinstance(units, list)
        units[1]["exclude"]["stage"] = "qualified"
        assert DatasetDefinition.model_validate(payload).content_hash != definition.content_hash

    def test_changing_the_exclusion_level_changes_it(self, definition: DatasetDefinition) -> None:
        payload = _payload()
        units = payload["units"]
        assert isinstance(units, list)
        units[1]["exclude"]["level"] = "post_aggregate"
        assert DatasetDefinition.model_validate(payload).content_hash != definition.content_hash

    def test_changing_the_upstream_dataset_changes_it(self, definition: DatasetDefinition) -> None:
        payload = _payload()
        units = payload["units"]
        assert isinstance(units, list)
        units[1]["resolutions"][0]["source"]["dataset"] = "universal_2026_combined"
        assert DatasetDefinition.model_validate(payload).content_hash != definition.content_hash

    def test_changing_the_upstream_resolution_policy_changes_it(self, definition: DatasetDefinition) -> None:
        payload = _payload()
        units = payload["units"]
        assert isinstance(units, list)
        units[1]["resolutions"][0]["source"]["run"] = {"policy": "named_release", "release": "2026-01"}
        assert DatasetDefinition.model_validate(payload).content_hash != definition.content_hash

    def test_changing_a_delivered_column_expression_changes_it(self, definition: DatasetDefinition) -> None:
        columns = [
            DerivedColumn(name="influencer", expression={"literal": 0}),
            *definition.delivery.columns[1:],
        ]
        delivery = definition.delivery.model_copy(update={"columns": columns})
        assert definition.model_copy(update={"delivery": delivery}).content_hash != definition.content_hash

    def test_changing_a_delivered_columns_sql_type_changes_it(self, definition: DatasetDefinition) -> None:
        columns = [
            definition.delivery.columns[0],
            definition.delivery.columns[1].model_copy(update={"sql_type": "character varying(64)"}),
        ]
        delivery = definition.delivery.model_copy(update={"columns": columns})
        assert definition.model_copy(update={"delivery": delivery}).content_hash != definition.content_hash

    def test_changing_an_artifacts_column_set_changes_it(self, definition: DatasetDefinition) -> None:
        artifacts = [
            ArtifactSpec(artifact=ArtifactKind.LONG, columns=["unit", "voterbase_id"]),
            *definition.artifacts[1:],
        ]
        assert definition.model_copy(update={"artifacts": artifacts}).content_hash != definition.content_hash

    def test_adding_a_datasource_qualifier_changes_it(self, definition: DatasetDefinition) -> None:
        assert _edited(additional_datasources=[]).content_hash != definition.content_hash

    def test_every_semantic_top_level_field_moves_the_hash(self, definition: DatasetDefinition) -> None:
        policy_fields = DatasetDefinition.hash_excluded_fields
        semantic = [name for name in DatasetDefinition.model_fields if name not in policy_fields]
        assert "version" not in semantic
        edits: dict[str, object] = {
            "name": "renamed_audience",
            "datasource": "influencers-publish",
            "additional_datasources": [],
            "grain": {"entity_column": "voterbase_id"},
            "parameters": [{"name": "match_schema", "parameter_type": "string"}],
            "qualification": [],
            "composition": None,
            "rollups": [],
            "expansions": [],
        }
        for field_name, value in edits.items():
            assert field_name in semantic
            assert _edited(**{field_name: value}).content_hash != definition.content_hash, field_name


class TestHashProjection:
    """the excluded fields never reach the serializer at all."""

    def test_the_canonical_payload_omits_grants_and_layout(self, definition: DatasetDefinition) -> None:
        canonical = canonical_content(definition)
        assert isinstance(canonical, dict)
        delivery = canonical["delivery"]
        assert isinstance(delivery, dict)
        assert set(delivery) == {"columns"}
        rendered = json.dumps(canonical, sort_keys=True)
        for excluded in ("grantee", "sortkey", "distkey", "diststyle", "encodings", "backup"):
            assert excluded not in rendered

    def test_no_artifact_carries_its_layout_into_the_payload(self, definition: DatasetDefinition) -> None:
        canonical = canonical_content(definition)
        assert isinstance(canonical, dict)
        artifacts = canonical["artifacts"]
        assert isinstance(artifacts, list)
        assert all(isinstance(entry, dict) and "layout" not in entry for entry in artifacts)

    def test_the_canonical_payload_omits_the_pinned_upstream_run(self) -> None:
        pin = UpstreamPin(policy=UpstreamPolicy.DRAFT_RUN, run=_ANOTHER_RUN)
        canonical = canonical_content(pin)
        assert isinstance(canonical, dict)
        assert set(canonical) == {"policy", "release"}
        assert str(_ANOTHER_RUN) not in json.dumps(canonical)

    def test_the_canonical_payload_omits_the_version(self, definition: DatasetDefinition) -> None:
        canonical = canonical_content(definition)
        assert isinstance(canonical, dict)
        assert "version" not in canonical

    def test_the_canonical_payload_carries_the_semantics(self, definition: DatasetDefinition) -> None:
        canonical = canonical_content(definition)
        assert isinstance(canonical, dict)
        assert canonical["name"] == "universal_2026_expansion"
        assert "units" in canonical

    def test_the_hash_is_a_sha256_hex_digest(self, definition: DatasetDefinition) -> None:
        assert len(definition.content_hash) == 64
        assert set(definition.content_hash) <= set("0123456789abcdef")

    def test_an_artifact_ref_is_hashable_by_content(self) -> None:
        first = ArtifactRef(scope=ArtifactScope.THIS_DEFINITION, artifact=ArtifactKind.LONG)
        second = ArtifactRef(scope=ArtifactScope.THIS_DEFINITION, artifact=ArtifactKind.LONG)
        assert content_hash(first) == content_hash(second)


class TestDeliveryLayoutStrategy:
    """the ENCODE decision, visible from the definition."""

    def test_the_wide_artifact_declares_encodings_and_needs_two_statements(self, definition: DatasetDefinition) -> None:
        wide = next(artifact for artifact in definition.artifacts if artifact.artifact is ArtifactKind.WIDE)
        assert wide.materialization_strategy.value == "create_then_insert"

    def test_the_qualified_artifact_declares_no_encodings_and_is_one_ctas(self, definition: DatasetDefinition) -> None:
        qualified = next(artifact for artifact in definition.artifacts if artifact.artifact is ArtifactKind.QUALIFIED)
        assert qualified.materialization_strategy.value == "ctas"


class TestDeliverySpecOnTheFixture:
    """the fixture exercises the shapes the corpus actually ships."""

    def test_the_delivered_artifact_is_declared(self, definition: DatasetDefinition) -> None:
        delivered = [artifact.artifact for artifact in definition.artifacts if artifact.delivered]
        assert delivered == [ArtifactKind.WIDE]

    def test_the_definition_emits_all_five_artifacts(self, definition: DatasetDefinition) -> None:
        assert {artifact.artifact for artifact in definition.artifacts} == set(ArtifactKind)

    def test_the_delivery_spec_carries_both_derived_column_shapes(self, definition: DatasetDefinition) -> None:
        assert [column.name for column in definition.delivery.columns] == ["influencer", "audience"]
        assert isinstance(definition.delivery.columns[0].expression, LiteralExpression)

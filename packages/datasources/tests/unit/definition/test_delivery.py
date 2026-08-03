"""unit tests for ``ArtifactSpec``, ``PhysicalLayout``, ``DeliverySpec``,
``DerivedColumn``, and ``GrantSpec``.

Every shape here is drawn from ``uhg_policymakers/sql/full_uhg_audience.sql``,
which is the one committed audience carrying a full physical spec and the
three delivered-column shapes:

- ordered string aggregation, ``LISTAGG(DISTINCT ...) WITHIN GROUP (ORDER BY ...)`` (`:22`)
- a conditional-aggregate flag, ``is_unique_match`` (`:14, 55, 66, 77, 104, 184`)
- an ordered category over unit sets (`:24, 43`), and the concatenate-then-trim
  deliverable that is why ``DerivedColumn.expression`` stays OPEN

and the layout hand-edited into the tool's own generated output
(`uhg_policymakers/sql/20250616.sql:442`), which attaches to the QUALIFIED
artifact rather than the delivered one.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.datasources.definition import (
    ArtifactKind,
    ArtifactRef,
    ArtifactScope,
    ArtifactSpec,
    ColumnEncoding,
    DeliverySpec,
    DerivedColumn,
    DistStyle,
    GranteeKind,
    GrantSpec,
    MaterializationStrategy,
    PhysicalLayout,
    WarehousePrivilege,
)


class TestArtifactSpec:
    """which artifacts a definition emits, and each one's column set."""

    def test_declares_an_artifact_and_its_columns(self) -> None:
        spec = ArtifactSpec(artifact=ArtifactKind.LONG, columns=["unit", "voterbase_id", "list_id", "record_year"])
        assert spec.artifact is ArtifactKind.LONG
        assert "list_id" in spec.columns

    def test_one_audiences_long_table_carries_no_list_id(self) -> None:
        # audience_test/sql/20250403.sql:5-7 -- the long DDL has no list_id
        # where the template has it, and an expansion edge joins on it.
        spec = ArtifactSpec(artifact=ArtifactKind.LONG, columns=["unit", "voterbase_id", "candidate_count"])
        assert "list_id" not in spec.columns

    def test_columns_are_required(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactSpec.model_validate({"artifact": "long"})

    def test_rejects_a_repeated_column(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactSpec(artifact=ArtifactKind.WIDE, columns=["voterbase_id", "voterbase_id"])

    def test_carries_its_own_grain(self) -> None:
        spec = ArtifactSpec(
            artifact=ArtifactKind.LONG,
            columns=["unit", "voterbase_id", "list_id"],
            grain=["unit", "list_id", "voterbase_id"],
        )
        assert spec.grain == ["unit", "list_id", "voterbase_id"]

    def test_a_grain_key_outside_the_column_set_is_refused(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ArtifactSpec(artifact=ArtifactKind.LONG, columns=["unit"], grain=["voterbase_id"])
        assert "voterbase_id" in str(excinfo.value)

    def test_layout_attaches_to_a_non_delivered_artifact(self) -> None:
        # DSM-01D-11: one committed audience carries DISTKEY/SORTKEY on its
        # QUALIFIED table, not on the delivered one.
        spec = ArtifactSpec(
            artifact=ArtifactKind.QUALIFIED,
            columns=["voterbase_id", "unit"],
            layout=PhysicalLayout(diststyle=DistStyle.KEY, distkey="voterbase_id", sortkey=["voterbase_id"]),
        )
        assert spec.layout is not None
        assert spec.delivered is False

    def test_a_layout_naming_a_column_outside_the_set_is_refused(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ArtifactSpec(
                artifact=ArtifactKind.WIDE,
                columns=["voterbase_id"],
                layout=PhysicalLayout(distkey="individual_id"),
            )
        assert "individual_id" in str(excinfo.value)

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactSpec.model_validate({"artifact": "long", "columns": ["unit"], "distkey": "unit"})


class TestPhysicalLayout:
    """DISTKEY, SORTKEY, DISTSTYLE, and per-column encodings."""

    def test_carries_the_full_committed_spec(self) -> None:
        # full_uhg_audience.sql:180-187.
        layout = PhysicalLayout(
            diststyle=DistStyle.KEY,
            distkey="individual_id",
            sortkey=["individual_id"],
            encodings={
                "individual_id": ColumnEncoding.RAW,
                "audience": ColumnEncoding.LZO,
                "category": ColumnEncoding.LZO,
                "is_unique_match": ColumnEncoding.AZ64,
            },
        )
        assert layout.distkey == "individual_id"
        assert layout.encodings["is_unique_match"] is ColumnEncoding.AZ64

    def test_a_key_diststyle_without_a_distkey_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            PhysicalLayout(diststyle=DistStyle.KEY)

    def test_a_distkey_under_an_even_diststyle_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            PhysicalLayout(diststyle=DistStyle.EVEN, distkey="individual_id")

    def test_rejects_a_repeated_sortkey_column(self) -> None:
        with pytest.raises(ValidationError):
            PhysicalLayout(sortkey=["individual_id", "individual_id"])

    def test_a_layout_declaring_nothing_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            PhysicalLayout()

    def test_no_encodings_means_one_ctas(self) -> None:
        # DSM-01D-12: Redshift's CTAS grammar accepts DISTSTYLE / DISTKEY /
        # SORTKEY / BACKUP and has NO column-level ENCODE clause.
        layout = PhysicalLayout(diststyle=DistStyle.KEY, distkey="individual_id")
        assert layout.materialization_strategy is MaterializationStrategy.CTAS

    def test_declared_encodings_force_create_then_insert(self) -> None:
        layout = PhysicalLayout(encodings={"individual_id": ColumnEncoding.RAW})
        assert layout.materialization_strategy is MaterializationStrategy.CREATE_THEN_INSERT

    def test_the_strategy_is_visible_on_the_artifact(self) -> None:
        plain = ArtifactSpec(artifact=ArtifactKind.WIDE, columns=["voterbase_id"])
        assert plain.materialization_strategy is MaterializationStrategy.CTAS
        encoded = ArtifactSpec(
            artifact=ArtifactKind.WIDE,
            columns=["voterbase_id"],
            layout=PhysicalLayout(encodings={"voterbase_id": ColumnEncoding.RAW}),
        )
        assert encoded.materialization_strategy is MaterializationStrategy.CREATE_THEN_INSERT


class TestDerivedColumn:
    """delivered columns, with an OPEN expression and a declared width."""

    def test_ordered_string_aggregation(self) -> None:
        column = DerivedColumn(
            name="source_records",
            sql_type="character varying(65535)",
            expression={"arith": "LISTAGG(DISTINCT source.list_id, ', ') WITHIN GROUP (ORDER BY source.list_id)"},
        )
        assert "WITHIN GROUP" in str(column.expression)

    def test_a_conditional_aggregate_flag(self) -> None:
        column = DerivedColumn(
            name="is_unique_match",
            expression={"arith": "MAX(CASE WHEN resolved.candidate_count = 1 THEN 1 ELSE 0 END)"},
        )
        assert column.name == "is_unique_match"

    def test_a_ranked_category_over_a_window_function(self) -> None:
        column = DerivedColumn(
            name="category",
            sql_type="character varying(7)",
            expression={"arith": "ROW_NUMBER() OVER (PARTITION BY resolved.voterbase_id ORDER BY CASE ... END)"},
        )
        assert column.sql_type == "character varying(7)"

    def test_the_expression_is_open_so_concatenate_then_classify_is_expressible(self) -> None:
        # full_uhg_audience.sql -- labels are concatenated then TRIMmed, and a
        # second deliverable classifies on the exact value of that
        # concatenation. no closed kind union expresses that.
        column = DerivedColumn(
            name="audience",
            sql_type="character varying(14)",
            expression={"arith": "TRIM(BOTH ', ' FROM rollup_a || ', ' || rollup_b)"},
        )
        assert column.name == "audience"

    def test_delivered_width_is_part_of_the_contract(self) -> None:
        assert DerivedColumn(name="audience", expression={"literal": "x"}).sql_type is None

    def test_carries_the_artifact_it_is_computed_over(self) -> None:
        column = DerivedColumn(
            name="is_unique_match",
            expression={"arith": "MAX(CASE WHEN resolved.candidate_count = 1 THEN 1 ELSE 0 END)"},
            over=ArtifactRef(scope=ArtifactScope.THIS_DEFINITION, artifact=ArtifactKind.LONG),
        )
        assert column.over is not None

    def test_rejects_a_blank_name(self) -> None:
        with pytest.raises(ValidationError):
            DerivedColumn(name="  ", expression={"literal": 1})

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            DerivedColumn.model_validate({"name": "x", "expression": {"literal": 1}, "kind": "aggregate"})


class TestGrantSpec:
    """DSM-01D-15: both users AND groups."""

    def test_a_group_grant(self) -> None:
        # GRANT SELECT ... TO GROUP INFLUENCERS, at five templates.
        grant = GrantSpec(grantee_kind=GranteeKind.GROUP, grantee="influencers")
        assert grant.privilege is WarehousePrivilege.SELECT

    def test_a_user_grant(self) -> None:
        grant = GrantSpec(grantee_kind=GranteeKind.USER, grantee="ehirschfeld")
        assert grant.grantee_kind is GranteeKind.USER

    def test_scoped_to_named_artifacts(self) -> None:
        grant = GrantSpec(
            grantee_kind=GranteeKind.GROUP,
            grantee="influencers",
            artifacts=[ArtifactKind.LONG, ArtifactKind.WIDE],
        )
        assert grant.covers(ArtifactKind.LONG) is True
        assert grant.covers(ArtifactKind.PROVENANCE) is False

    def test_an_empty_artifact_list_covers_every_emitted_artifact(self) -> None:
        grant = GrantSpec(grantee_kind=GranteeKind.GROUP, grantee="influencers")
        assert grant.covers(ArtifactKind.PROVENANCE) is True

    def test_rejects_a_blank_grantee(self) -> None:
        with pytest.raises(ValidationError):
            GrantSpec(grantee_kind=GranteeKind.GROUP, grantee="   ")

    def test_rejects_a_repeated_artifact(self) -> None:
        with pytest.raises(ValidationError):
            GrantSpec(
                grantee_kind=GranteeKind.GROUP,
                grantee="influencers",
                artifacts=[ArtifactKind.LONG, ArtifactKind.LONG],
            )


class TestDeliverySpec:
    """derived columns, warehouse grants, and the delivered layout."""

    def test_carries_derived_columns(self) -> None:
        delivery = DeliverySpec(columns=[DerivedColumn(name="influencer", expression={"literal": 1})])
        assert delivery.columns[0].name == "influencer"

    def test_rejects_a_repeated_delivered_column_name(self) -> None:
        with pytest.raises(ValidationError):
            DeliverySpec(
                columns=[
                    DerivedColumn(name="audience", expression={"literal": "a"}),
                    DerivedColumn(name="audience", expression={"literal": "b"}),
                ]
            )

    def test_carries_warehouse_grants_and_a_layout(self) -> None:
        delivery = DeliverySpec(
            grants=[GrantSpec(grantee_kind=GranteeKind.GROUP, grantee="influencers")],
            layout=PhysicalLayout(sortkey=["individual_id"]),
        )
        assert delivery.grants[0].grantee == "influencers"
        assert delivery.layout is not None

    def test_rejects_a_repeated_grantee(self) -> None:
        with pytest.raises(ValidationError):
            DeliverySpec(
                grants=[
                    GrantSpec(grantee_kind=GranteeKind.GROUP, grantee="influencers"),
                    GrantSpec(grantee_kind=GranteeKind.GROUP, grantee="influencers"),
                ]
            )

    def test_grants_and_layout_are_declared_hash_excluded(self) -> None:
        # DSM-01D-16. asserted structurally here and behaviourally in
        # tests/unit/definition/test_content_hash.py.
        assert DeliverySpec.hash_excluded_fields == frozenset({"grants", "layout"})

    def test_carries_no_platform_visibility_or_dataset_grant(self) -> None:
        # D21: visibility and dataset_customers grants are PLATFORM state and
        # never definition content. GrantSpec here is a WAREHOUSE grant.
        assert "visibility" not in DeliverySpec.model_fields
        assert set(GrantSpec.model_fields) == {"privilege", "grantee_kind", "grantee", "artifacts"}

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            DeliverySpec.model_validate({"columns": [], "visibility": "public"})

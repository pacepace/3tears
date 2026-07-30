"""tests for threetears.datasources.entities.

covers enum membership + value stability, flat-PK shape on
CapabilitySourceEntity, composite-PK shape on TableTemplateEntity, flat-PK
shape on DataSourceTableEntity / DataSourceColumnEntity /
DataSourceRelationEntity, and BaseEntity subclass invariants.

access-mode coverage reaches past entities on purpose. the value set is
carried by TWO independent string authorities in this package --
:class:`DataSourceAccessMode` here, and
``threetears.datasources.config._VALID_ACCESS_MODES`` -- and neither
references the other. the parity and normalization cases live beside the
enum they are guarding so a mode added to one authority and not the other
fails in the same file that shows the enum.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from threetears.datasources.config import _VALID_ACCESS_MODES, DatasourceConfig
from threetears.datasources.entities import (
    CapabilitySourceEntity,
    DataSourceAccessMode,
    DataSourceColumnEntity,
    DataSourceRelationEntity,
    DataSourceStatus,
    DataSourceTableEntity,
    DataSourceType,
    TableTemplateEntity,
)


class TestDataSourceTypeEnum:
    """enum carries every documented backend type with stable string values."""

    def test_members(self) -> None:
        assert {m.value for m in DataSourceType} == {
            "redshift",
            "snowflake",
            "bigquery",
            "postgres",
            "yugabyte",
            "agent_internal",
        }

    def test_str_equivalence(self) -> None:
        # StrEnum: members compare equal to their string values
        assert DataSourceType.REDSHIFT == "redshift"
        assert DataSourceType.AGENT_INTERNAL == "agent_internal"


class TestDataSourceAccessModeEnum:
    """five access-mode values; BUILD then PUBLISH appended, never inserted."""

    def test_members(self) -> None:
        assert {m.value for m in DataSourceAccessMode} == {
            "read",
            "write",
            "readwrite",
            "build",
            "publish",
        }

    def test_declaration_order_appends_only(self) -> None:
        """new members are appended, so an inserted one fails here.

        two admin-console sites index the access-mode list POSITIONALLY to
        seed a default. inserting a value ahead of ``readwrite`` silently
        changes the default access mode for every new capability source, so
        the order is part of the contract and not incidental.

        the assertion is the FULL ordered list rather than a check that some
        named member is last, so appending a sixth mode fails here too and
        has to be looked at rather than absorbed.
        """
        assert [m.value for m in DataSourceAccessMode] == [
            "read",
            "write",
            "readwrite",
            "build",
            "publish",
        ]

    def test_build_is_not_composed(self) -> None:
        """``build`` is a fourth value, never a composition.

        a composed ``readwritebuild`` would put the warehouse user's
        ``CREATE`` grant behind the read tool, which is exactly the
        structural least-privilege claim the separate mode buys.
        """
        assert "readwritebuild" not in {m.value for m in DataSourceAccessMode}

    def test_str_equivalence(self) -> None:
        # StrEnum: members compare equal to their string values
        assert DataSourceAccessMode.BUILD == "build"


class TestAccessModeAuthorityParity:
    """the enum and the config frozenset MUST carry identical value sets.

    ``_VALID_ACCESS_MODES`` mirrors the enum by hand rather than importing
    it, so nothing in the type system stops a fifth mode landing in one
    authority and not the other. this test is what stops it.
    """

    def test_config_frozenset_matches_enum(self) -> None:
        assert set(_VALID_ACCESS_MODES) == {m.value for m in DataSourceAccessMode}


class TestDatasourceConfigAccessMode:
    """``DatasourceConfig`` is the YAML-facing gate on the same value set.

    the model carries ``extra="forbid"``, so the ``access_mode`` field is
    the only entry point and its validator is the whole gate.
    """

    def test_build_mode_loads(self) -> None:
        cfg = DatasourceConfig.model_validate({"name": "influencers-build", "access_mode": "build"})
        assert cfg.access_mode == "build"

    def test_publish_mode_loads(self) -> None:
        """the publisher row must be creatable, and it was not.

        the Hub's promote path pinned its own ``access_mode`` literal to
        ``"publish"`` and rejected every row not carrying it, while this
        closed value set had no such member. so the one row the promote would
        accept was the one row the platform refused to store, and the
        publisher identity could not exist at all.
        """
        cfg = DatasourceConfig.model_validate(
            {"name": "influencers-publish", "access_mode": "publish"},
        )
        assert cfg.access_mode == "publish"

    @pytest.mark.parametrize("raw", ["Build", "BUILD", "  build  ", " Build\t"])
    def test_normalizes_case_and_whitespace(self, raw: str) -> None:
        """an unnormalized mode fails SILENTLY downstream, so normalize here.

        a stored ``Build`` matches none of the tool-pod's registration
        branches: no tools register, nothing raises, and the only trace is
        ``tool_count=0``.
        """
        assert DatasourceConfig.model_validate({"name": "x", "access_mode": raw}).access_mode == "build"

    @pytest.mark.parametrize("raw", ["admin", "readwritebuild", "buildread", ""])
    def test_rejects_unknown_modes(self, raw: str) -> None:
        with pytest.raises(ValidationError):
            DatasourceConfig.model_validate({"name": "x", "access_mode": raw})


class TestDataSourceStatusEnum:
    """lifecycle enum is two-valued; no soft-delete sentinel."""

    def test_members(self) -> None:
        assert {m.value for m in DataSourceStatus} == {"active", "disabled"}


class TestCapabilitySourceEntity:
    """flat-PK shape post-knowledge-task-08: ``primary_key_field == 'id'``.

    the v016 migration rebuilt the table PK on ``id`` alone (dropping the
    v001 composite ``(customer_id, id)`` partition PK) so a platform-shared
    source can carry ``customer_id = NULL`` (KNW-76); ``customer_id`` is
    now a plain nullable column, not the partition / addressing key.
    """

    def test_id_is_flat_primary_key(self) -> None:
        row_id = uuid4()
        entity = CapabilitySourceEntity(
            data={"customer_id": uuid4(), "id": row_id, "name": "ds"},
            is_new=True,
        )
        # scalar id property returns the row UUID
        assert entity.id == row_id
        # the addressing key is the flat ``id`` now, not the partition column
        assert entity.primary_key_field == "id"

    def test_platform_shared_source_id_with_null_customer(self) -> None:
        """a platform-shared source (customer_id NULL) addresses by id."""
        row_id = uuid4()
        entity = CapabilitySourceEntity(
            data={"customer_id": None, "id": row_id, "name": "shared"},
            is_new=True,
        )
        assert entity.id == row_id
        assert entity.customer_id is None


class TestDataSourceTableEntity:
    """flat-PK shape: ``primary_key_field == 'id'``."""

    def test_flat_pk(self) -> None:
        entity = DataSourceTableEntity(
            data={"id": uuid4(), "datasource_id": uuid4(), "schema_name": "s", "table_name": "t"},
            is_new=True,
        )
        assert entity.primary_key_field == "id"


class TestDataSourceColumnEntity:
    """flat-PK column entity carries the natural-key fields as data."""

    def test_flat_pk_and_data(self) -> None:
        column_id = uuid4()
        entity = DataSourceColumnEntity(
            data={
                "id": column_id,
                "datasource_id": uuid4(),
                "schema_name": "s",
                "table_name": "t",
                "column_name": "c",
                "data_type": "int",
                "is_nullable": False,
                "ordinal_position": 1,
            },
            is_new=True,
        )
        assert entity.primary_key_field == "id"


class TestDataSourceRelationEntity:
    """relation entities are flat-PK; cross-table metadata lives in data."""

    def test_flat_pk(self) -> None:
        entity = DataSourceRelationEntity(
            data={"id": uuid4(), "name": "r1"},
            is_new=True,
        )
        assert entity.primary_key_field == "id"


class TestTableTemplateEntity:
    """template entities carry the composite-PK ``(customer_id, id)`` shape."""

    def test_id_and_partition(self) -> None:
        template_id = uuid4()
        entity = TableTemplateEntity(
            data={"customer_id": uuid4(), "id": template_id, "name": "tpl"},
            is_new=True,
        )
        assert entity.id == template_id
        assert isinstance(entity.id, UUID)
        assert entity.primary_key_field == "customer_id"

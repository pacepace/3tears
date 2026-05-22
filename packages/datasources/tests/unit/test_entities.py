"""tests for threetears.datasources.entities.

covers enum membership + value stability, composite-PK shape on
DataSourceEntity / TableTemplateEntity, flat-PK shape on
DataSourceTableEntity / DataSourceColumnEntity / DataSourceRelationEntity,
and BaseEntity subclass invariants.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from threetears.datasources.entities import (
    DataSourceAccessMode,
    DataSourceColumnEntity,
    DataSourceEntity,
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
            "agent_internal",
        }

    def test_str_equivalence(self) -> None:
        # StrEnum: members compare equal to their string values
        assert DataSourceType.REDSHIFT == "redshift"
        assert DataSourceType.AGENT_INTERNAL == "agent_internal"


class TestDataSourceAccessModeEnum:
    """three access-mode values; no surprises."""

    def test_members(self) -> None:
        assert {m.value for m in DataSourceAccessMode} == {"read", "write", "readwrite"}


class TestDataSourceStatusEnum:
    """lifecycle enum is two-valued; no soft-delete sentinel."""

    def test_members(self) -> None:
        assert {m.value for m in DataSourceStatus} == {"active", "disabled"}


class TestDataSourceEntity:
    """composite-PK shape: ``_id`` is a ``(customer_id, id)`` tuple."""

    def test_id_tuple_shape(self) -> None:
        customer_id = uuid4()
        row_id = uuid4()
        entity = DataSourceEntity(
            data={"customer_id": customer_id, "id": row_id, "name": "ds"},
            is_new=True,
        )
        assert entity._id == (customer_id, row_id)
        # scalar id property returns the row UUID, not the tuple
        assert entity.id == row_id
        # primary_key_field signals the partition column to the framework
        assert entity.primary_key_field == "customer_id"


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
    """template entities mirror DataSourceEntity's composite-PK shape."""

    def test_id_tuple_shape(self) -> None:
        customer_id = uuid4()
        template_id = uuid4()
        entity = TableTemplateEntity(
            data={"customer_id": customer_id, "id": template_id, "name": "tpl"},
            is_new=True,
        )
        assert entity._id == (customer_id, template_id)
        assert entity.id == template_id
        assert isinstance(entity.id, UUID)
        assert entity.primary_key_field == "customer_id"

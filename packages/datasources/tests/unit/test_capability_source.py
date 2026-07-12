"""tests for the generalized capability-source registry (gu-task-08).

covers the Fork-1 generalization of the former datasource registry into a
capability-source shape:

- :class:`CapabilitySourceKind` discriminator enum (``datasource`` /
  ``api_import`` / ``mcp_import``) — exactly three members.
- :class:`CapabilitySourceEntity` carries the generalized field surface
  (``kind`` + ``connection_config`` + ``ingress_agent_id``) and round-trips.
- :class:`CapabilitySourceCollection` declares ``kind`` +
  ``ingress_agent_id`` columns on its :class:`TableSchema` and remains a
  :class:`SchemaBackedCollection`.
- config-storage validation is KIND-CONDITIONAL (GU-08-03): a
  ``datasource``-kind row keeps its existing encrypted-JSON-blob shape and
  is NOT run through ``validate_ref`` at the write boundary; an
  ``api_import`` / ``mcp_import``-kind row MUST carry a valid
  ``secret_refs`` ``scheme://locator`` reference and IS rejected when the
  reference is malformed.
- ``status`` still gates :meth:`CapabilitySourceCollection.iter_active_ids`.
- the module carries no BANNED "tenant" substring.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from threetears.core.security.secret_refs import SecretResolutionError
from threetears.datasources.collections import CapabilitySourceCollection
from threetears.datasources.entities import (
    CapabilitySourceEntity,
    CapabilitySourceKind,
    DataSourceStatus,
    DataSourceType,
)


def _make_registry_and_config() -> tuple[MagicMock, MagicMock]:
    """build mocked registry and config for collection instantiation.

    :return: tuple of (registry, config) mocks
    :rtype: tuple[MagicMock, MagicMock]
    """
    registry = MagicMock()
    registry.get_l1_backend.return_value = None
    registry.get_l3_pool.return_value = None
    config = MagicMock()
    config.collection_flush = "ALWAYS"
    config.collection_flush_tables = ""
    return registry, config


class TestCapabilitySourceKindEnum:
    """the discriminator carries exactly the three Fork-1 kinds."""

    def test_members(self) -> None:
        assert {m.value for m in CapabilitySourceKind} == {
            "datasource",
            "api_import",
            "mcp_import",
        }

    def test_str_equivalence(self) -> None:
        # StrEnum: members compare equal to their string values
        assert CapabilitySourceKind.DATASOURCE == "datasource"
        assert CapabilitySourceKind.API_IMPORT == "api_import"
        assert CapabilitySourceKind.MCP_IMPORT == "mcp_import"


class TestCapabilitySourceEntity:
    """generalized entity carries kind + config + ingress-agent surface."""

    def test_datasource_kind_flat_pk_and_fields(self) -> None:
        row_id = uuid4()
        entity = CapabilitySourceEntity(
            data={
                "id": row_id,
                "customer_id": uuid4(),
                "name": "ots-redshift",
                "kind": CapabilitySourceKind.DATASOURCE,
                "datasource_type": DataSourceType.REDSHIFT,
                # existing encrypted-JSON-blob shape — NOT a scheme://locator
                "connection_config": '{"host": "warehouse", "password": "ciphertext"}',
                "ingress_agent_id": None,
            },
            is_new=True,
        )
        assert entity.id == row_id
        assert entity.primary_key_field == "id"
        assert entity.kind == CapabilitySourceKind.DATASOURCE
        assert entity.ingress_agent_id is None

    def test_api_import_kind_carries_ingress_agent(self) -> None:
        row_id = uuid4()
        ingress_agent_id = uuid4()
        entity = CapabilitySourceEntity(
            data={
                "id": row_id,
                "customer_id": None,
                "name": "stripe-api",
                "kind": CapabilitySourceKind.API_IMPORT,
                # import kinds address the config via a secret_refs reference
                "connection_config": "env://STRIPE_KEY",
                "ingress_agent_id": ingress_agent_id,
            },
            is_new=True,
        )
        assert entity.kind == CapabilitySourceKind.API_IMPORT
        assert entity.ingress_agent_id == ingress_agent_id
        assert entity.connection_config == "env://STRIPE_KEY"


class TestCapabilitySourceCollectionSchema:
    """the generalized TableSchema keeps the table + adds the new columns."""

    def test_table_name(self) -> None:
        registry, config = _make_registry_and_config()
        coll = CapabilitySourceCollection(registry=registry, config=config)
        assert coll.table_name == "datasources"

    def test_entity_class(self) -> None:
        registry, config = _make_registry_and_config()
        coll = CapabilitySourceCollection(registry=registry, config=config)
        assert coll.entity_class is CapabilitySourceEntity

    def test_primary_key_column_is_flat_id(self) -> None:
        assert CapabilitySourceCollection.primary_key_column == "id"

    def test_kind_column_declared(self) -> None:
        assert CapabilitySourceCollection.schema.get_column("kind") is not None

    def test_ingress_agent_id_column_declared_nullable(self) -> None:
        col = CapabilitySourceCollection.schema.get_column("ingress_agent_id")
        assert col is not None
        assert col.nullable is True

    def test_connection_config_column_retained(self) -> None:
        # GU-08-03: the config column is the SAME physical STRING_TYPE
        # passthrough for both the datasource ciphertext blob and the
        # import-kind secret_refs reference — the table shape is intact.
        col = CapabilitySourceCollection.schema.get_column("connection_config")
        assert col is not None
        assert col.nullable is True


class TestKindConditionalConfigValidation:
    """GU-08-03: validate_ref fires for import kinds, NEVER for datasource."""

    @pytest.mark.asyncio
    async def test_datasource_kind_blob_not_rejected(self) -> None:
        # an existing datasource's connection_config is an encrypted JSON
        # blob, NOT a scheme://locator. it MUST pass the write boundary
        # untouched — validating it as a ref would break every existing
        # datasource write (the break P6 guards against). l3_pool is None,
        # so save_to_store returns 0; the point is that it does NOT raise.
        registry, config = _make_registry_and_config()
        coll = CapabilitySourceCollection(registry=registry, config=config)
        rows = await coll.save_to_store(
            {
                "id": uuid4(),
                "name": "ds",
                "kind": CapabilitySourceKind.DATASOURCE,
                "datasource_type": DataSourceType.REDSHIFT,
                "connection_config": '{"host": "h", "password": "ciphertext"}',
            },
        )
        assert rows == 0

    @pytest.mark.asyncio
    async def test_import_kind_valid_ref_accepted(self) -> None:
        registry, config = _make_registry_and_config()
        coll = CapabilitySourceCollection(registry=registry, config=config)
        rows = await coll.save_to_store(
            {
                "id": uuid4(),
                "name": "stripe",
                "kind": CapabilitySourceKind.API_IMPORT,
                "connection_config": "env://STRIPE_KEY",
                "ingress_agent_id": uuid4(),
            },
        )
        assert rows == 0

    @pytest.mark.asyncio
    async def test_import_kind_invalid_ref_rejected(self) -> None:
        # the SAME blob that a datasource write accepts is rejected for an
        # import kind because it is not a scheme://locator — proves the
        # validation branches on kind, not unconditionally.
        registry, config = _make_registry_and_config()
        coll = CapabilitySourceCollection(registry=registry, config=config)
        with pytest.raises(SecretResolutionError):
            await coll.save_to_store(
                {
                    "id": uuid4(),
                    "name": "bad-import",
                    "kind": CapabilitySourceKind.MCP_IMPORT,
                    "connection_config": '{"host": "h", "password": "plaintext"}',
                },
            )

    @pytest.mark.asyncio
    async def test_import_kind_null_config_allowed(self) -> None:
        # the config column is nullable; a None config is not validated.
        registry, config = _make_registry_and_config()
        coll = CapabilitySourceCollection(registry=registry, config=config)
        rows = await coll.save_to_store(
            {
                "id": uuid4(),
                "name": "pending-import",
                "kind": CapabilitySourceKind.API_IMPORT,
                "connection_config": None,
            },
        )
        assert rows == 0


class TestIterActiveIdsStatusGate:
    """status still gates the scheduler sweep after generalization."""

    @pytest.mark.asyncio
    async def test_iter_active_ids_filters_status_active(self) -> None:
        registry, config = _make_registry_and_config()
        mock_pool = MagicMock()
        mock_pool.fetch = AsyncMock(return_value=[])
        registry.get_l3_pool.return_value = mock_pool

        coll = CapabilitySourceCollection(registry=registry, config=config)
        result = await coll.iter_active_ids()

        assert result == []
        mock_pool.fetch.assert_awaited_once()
        sql_arg, *bound_args = mock_pool.fetch.await_args.args
        assert "WHERE status" in sql_arg
        assert bound_args == [DataSourceStatus.ACTIVE.value]


def test_no_tenant_word_in_entities_and_collections() -> None:
    """BANNED term guard: neither module carries the substring 'tenant'."""
    src_root = Path(__file__).resolve().parents[2] / "src" / "threetears" / "datasources"
    for name in ("entities.py", "collections.py"):
        text = (src_root / name).read_text(encoding="utf-8").lower()
        assert "tenant" not in text, f"BANNED term 'tenant' found in {name}"

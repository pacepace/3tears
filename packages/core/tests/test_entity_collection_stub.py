"""parity tests for the shared entity collection stub.

the stub in :mod:`threetears.core.testing.entities` stands in for a
real :class:`~threetears.core.collections.base.BaseCollection` in entity
unit tests across the workspace. a stub that answers differently from
the collection it replaces makes the difference untestable -- every test
built on it agrees with the stub and none of them notices the real
behaviour.

so each divergence gets a test here that asserts the stub and the real
collection agree, running the SAME assertions against both. the two
behaviours covered:

* the absent-L1 return values. the real accessors all short-circuit on
  ``self._l1 is None``; the stub had no way to express that, so
  :meth:`BaseEntity.__init__`'s ``_changes`` fallback and
  :meth:`BaseEntity.set_data`'s ``RuntimeError`` were unreachable from
  any stub-based test.
* writing a PRIMARY KEY column through ``set_field_sync``. the real one
  re-``upsert``s a detached copy of the row, so the write lands at the
  NEW key and the old row survives -- two rows, not one.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table

from threetears.core.cache import MISSING
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing import entity_collection_stub


def _make_metadata() -> MetaData:
    """build the L1 schema the parity collection caches into.

    :return: metadata carrying the ``stub_parity_entities`` table
    :rtype: MetaData
    """
    metadata = MetaData()
    Table(
        "stub_parity_entities",
        metadata,
        Column("id", String(255), primary_key=True),
        Column("name", String(255)),
        Column("score", Integer),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
    )
    return metadata


class ParityEntity(BaseEntity):
    """entity type for the parity collection."""

    primary_key_field = "id"


class ParityCollection(BaseCollection[ParityEntity]):
    """minimal concrete collection used to observe real L1 behaviour."""

    @property
    def table_name(self) -> str:
        """:return: backing table name
        :rtype: str
        """
        return "stub_parity_entities"

    @property
    def entity_class(self) -> type[ParityEntity]:
        """:return: entity type this collection yields
        :rtype: type[ParityEntity]
        """
        return ParityEntity

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """:param entity_id: pk value
        :ptype entity_id: Any
        :return: always ``None`` -- these tests never reach L3
        :rtype: dict[str, Any] | None
        """
        return None

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """:param data: row to persist
        :ptype data: dict[str, Any]
        :param original_timestamp: optimistic-concurrency token
        :ptype original_timestamp: datetime | None
        :param conn: caller-supplied connection
        :ptype conn: Any
        :return: rows written
        :rtype: int
        """
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        """:param entity_id: pk value
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        return None

    def serialize(self, data: dict[str, Any]) -> bytes:
        """:param data: row to encode for L2
        :ptype data: dict[str, Any]
        :return: json bytes
        :rtype: bytes
        """
        return json.dumps(data, default=str).encode()

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """:param data: json bytes from L2
        :ptype data: bytes
        :return: decoded row
        :rtype: dict[str, Any]
        """
        result: dict[str, Any] = json.loads(data)
        return result


@pytest.fixture()
def l1_backend() -> SQLiteBackend:
    """yield an initialized SQLite L1 backend, torn down after the test."""
    backend = SQLiteBackend(db_name=f"test_stub_parity_{uuid.uuid4().hex[:8]}")
    backend.initialize(_make_metadata())
    yield backend
    from threetears.core._bridge import drain, shutdown

    drain()
    shutdown()
    backend.reset()


@pytest.fixture()
def config_always() -> DefaultCoreConfig:
    """yield a flush-always core config."""
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


@pytest.fixture()
def real_collection(l1_backend: SQLiteBackend, config_always: DefaultCoreConfig) -> ParityCollection:
    """yield a real collection wired to an L1 backend."""
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1_backend)
    return ParityCollection(registry, config_always)


@pytest.fixture()
def real_collection_no_l1(config_always: DefaultCoreConfig) -> ParityCollection:
    """yield a real collection with no L1 backend configured."""
    return ParityCollection(CollectionRegistry(), config_always)


class TestAbsentL1Parity:
    """the stub can express "no L1 backend", and answers as the real one does."""

    def test_stub_without_l1_matches_real_collection_accessors(self, real_collection_no_l1: ParityCollection) -> None:
        """every sync accessor returns what the real no-L1 collection returns.

        mirrors ``TestNoL1Backend.test_field_accessors_return_missing_without_l1``
        in ``test_base_collection.py``, run against the stub instead.
        """
        stub, cache = entity_collection_stub(("id",), has_l1=False)
        real = real_collection_no_l1

        assert stub.get_field_sync("e1", "name") is real.get_field_sync("e1", "name") is MISSING
        assert stub.set_field_sync("e1", "name", "x") == real.set_field_sync("e1", "name", "x") is False
        assert stub.get_row_sync("e1") == real.get_row_sync("e1") is None
        assert stub.write_to_cache_sync({"id": "e1"}) == real.write_to_cache_sync({"id": "e1"}) is False
        assert cache == {}

    def test_stub_without_l1_stays_empty_after_a_write(self) -> None:
        """a rejected write leaves nothing behind for a later read to find."""
        stub, cache = entity_collection_stub(("id",), has_l1=False)

        assert stub.write_to_cache_sync({"id": "e1", "name": "Ada"}) is False

        assert cache == {}
        assert stub.get_row_sync("e1") is None

    def test_stub_with_l1_reports_a_successful_write(self, real_collection: ParityCollection) -> None:
        """the default keeps the L1-present behaviour both objects share."""
        stub, cache = entity_collection_stub(("id",))

        assert stub.write_to_cache_sync({"id": "e1", "name": "Ada"}) is True
        assert real_collection.write_to_cache_sync({"id": "e1", "name": "Ada"}) is True
        assert cache[("e1",)]["name"] == "Ada"

    def test_absent_l1_stub_is_still_a_magicmock(self) -> None:
        """``assert_called_with`` keeps working -- load-bearing for callers."""
        stub, _cache = entity_collection_stub(("id",), has_l1=False)

        stub.set_field_sync("e1", "name", "Ada")

        assert isinstance(stub, MagicMock)
        stub.set_field_sync.assert_called_with("e1", "name", "Ada")
        assert isinstance(stub.save_entity, AsyncMock)


class TestPrimaryKeyWriteParity:
    """writing a pk column through ``set_field_sync`` re-keys, it does not rename."""

    def test_real_collection_pk_write_leaves_the_old_row_behind(self, real_collection: ParityCollection) -> None:
        """the real behaviour: an upsert at the new key, old row untouched."""
        coll = real_collection
        coll.write_to_cache_sync({"id": "e1", "name": "Ada", "score": 1})

        assert coll.set_field_sync("e1", "id", "e2") is True

        old_row = coll.get_row_sync("e1")
        new_row = coll.get_row_sync("e2")
        assert old_row is not None, "real collection left no row at the old key"
        assert old_row["id"] == "e1"
        assert new_row is not None
        assert new_row["name"] == "Ada"

    def test_stub_pk_write_leaves_the_old_row_behind(self) -> None:
        """the stub reproduces it: two rows, keyed old and new."""
        stub, cache = entity_collection_stub(("id",))
        stub.write_to_cache_sync({"id": "e1", "name": "Ada", "score": 1})

        assert stub.set_field_sync("e1", "id", "e2") is True

        assert set(cache) == {("e1",), ("e2",)}
        assert cache[("e1",)]["id"] == "e1"
        assert cache[("e2",)]["name"] == "Ada"

    def test_stub_composite_pk_write_leaves_the_old_row_behind(self) -> None:
        """re-keying one column of a composite pk splits the row in two."""
        stub, cache = entity_collection_stub(("customer_id", "id"))
        stub.write_to_cache_sync({"customer_id": "cust-A", "id": "row-1", "name": "Ada"})

        assert stub.set_field_sync(("cust-A", "row-1"), "customer_id", "cust-B") is True

        assert set(cache) == {("cust-A", "row-1"), ("cust-B", "row-1")}
        assert cache[("cust-B", "row-1")]["name"] == "Ada"

    def test_non_pk_write_updates_the_one_row(self, real_collection: ParityCollection) -> None:
        """the ordinary case is unchanged on both sides: one row, new value."""
        stub, cache = entity_collection_stub(("id",))
        stub.write_to_cache_sync({"id": "e1", "name": "Ada"})
        real_collection.write_to_cache_sync({"id": "e1", "name": "Ada"})

        assert stub.set_field_sync("e1", "name", "Grace") is True
        assert real_collection.set_field_sync("e1", "name", "Grace") is True

        assert set(cache) == {("e1",)}
        assert cache[("e1",)]["name"] == "Grace"
        real_row = real_collection.get_row_sync("e1")
        assert real_row is not None
        assert real_row["name"] == "Grace"

    def test_stub_read_result_cannot_mutate_the_cache(self) -> None:
        """``get_row_sync`` hands back a copy, as ``select_by_id`` does."""
        stub, cache = entity_collection_stub(("id",))
        stub.write_to_cache_sync({"id": "e1", "name": "Ada"})

        row = stub.get_row_sync("e1")
        assert row is not None
        row["name"] = "mutated"

        assert cache[("e1",)]["name"] == "Ada"

    def test_set_field_on_an_uncached_row_reports_failure(self, real_collection: ParityCollection) -> None:
        """both refuse a write to a row L1 has never seen."""
        stub, cache = entity_collection_stub(("id",))

        assert stub.set_field_sync("nope", "name", "Ada") is False
        assert real_collection.set_field_sync("nope", "name", "Ada") is False
        assert cache == {}


class TestStubbedEntityAgreesWithRealEntity:
    """an entity over the stub behaves as one over the real collection."""

    def test_entity_pk_write_orphans_its_own_row_on_both(self, real_collection: ParityCollection) -> None:
        """assigning to the pk attribute re-keys the row but not the entity.

        the entity keeps addressing ``_id`` -- the OLD key -- so the row
        it now reads is the stale one. reproducing that through the stub
        is the point: before this parity fix the stub renamed in place
        and the entity read its own new row, which no real collection
        does.
        """
        stub, cache = entity_collection_stub(("id",))
        stubbed = BaseEntity({"id": "e1", "name": "Ada"}, is_new=False, collection=stub)
        real = BaseEntity({"id": "e1", "name": "Ada"}, is_new=False, collection=real_collection)

        stubbed.id_column_probe = "unused"  # keeps _column_names honest
        stub.set_field_sync("e1", "id", "e2")
        real_collection.set_field_sync("e1", "id", "e2")

        assert stub.get_row_sync("e1") is not None
        assert real_collection.get_row_sync("e1") is not None
        assert set(cache) == {("e1",), ("e2",)}
        assert stubbed.addressing_id == real.addressing_id == "e1"

"""Unit tests for the agent-side knowledge collections + their pure helpers.

Covers the non-trivial data-transformation logic without live infrastructure: the
proxy-border UUID coercion, the pgvector ``::text`` parse, the row -> merge-snapshot
builders (scope derivation + tuple normalization + table-ref carry-through), the
bounded embedding read (NULL filtering + string-id coercion), and the SQL-assembly
branches of :meth:`list_visible_to_user` / :meth:`list_own_drafts` driven through a
stub L3 pool.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid7

import pytest
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.scan_cache import ScanCache
from threetears.knowledge import Scope, build_table_ref

from threetears.agent.knowledge.collections import (
    ConceptCollection,
    PlaybookEntryCollection,
)


class _StubPool:
    """L3 pool stand-in that records the last query and returns fixed rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.sql: str | None = None
        self.params: tuple[Any, ...] | None = None
        self.customer_scope: Any = None

    async def fetch(self, sql: str, *params: Any, customer_scope: Any) -> list[dict[str, Any]]:
        self.sql = sql
        self.params = params
        self.customer_scope = customer_scope
        return list(self._rows)


def _entries_over(rows: list[dict[str, Any]]) -> PlaybookEntryCollection:
    """an entry collection reading ``rows`` through a stub proxy pool, no cache."""
    coll = PlaybookEntryCollection.__new__(PlaybookEntryCollection)
    coll.l3_pool = _StubPool(rows)
    return coll


def _concepts_over(rows: list[dict[str, Any]]) -> ConceptCollection:
    """a concept collection reading ``rows`` through a stub proxy pool, no cache."""
    coll = ConceptCollection.__new__(ConceptCollection)
    coll.l3_pool = _StubPool(rows)
    return coll


async def _embeddings_of(value: Any) -> dict[UUID, list[float]]:
    """read one row's ``embedding::text`` value back through ``fetch_embeddings``."""
    row_id = uuid7()
    coll = _concepts_over([{"id": str(row_id), "embedding": value}])
    return await coll.fetch_embeddings([row_id], customer_scope=uuid7())


class TestProxyIdsComeBackAsUuids:
    """ids cross the NATS proxy as strings; the snapshots carry real UUIDs."""

    async def test_passthrough_uuid(self) -> None:
        u = uuid7()
        (snap,) = await _entries_over([{**_entry_row(), "id": u}]).list_visible_to_user(uuid7(), customer_scope=uuid7())
        assert snap.id is u

    async def test_coerces_string(self) -> None:
        u = uuid7()
        (snap,) = await _entries_over([{**_entry_row(), "id": str(u)}]).list_visible_to_user(
            uuid7(), customer_scope=uuid7()
        )
        assert snap.id == u
        assert isinstance(snap.id, UUID)

    async def test_none_stays_none(self) -> None:
        (snap,) = await _entries_over([{**_entry_row(), "origin_entry_id": None}]).list_visible_to_user(
            uuid7(), customer_scope=uuid7()
        )
        assert snap.origin_entry_id is None


class TestStoredVectorsParse:
    """the ``embedding::text`` form parses back to floats; NULL and junk are omitted."""

    async def test_none_is_omitted(self) -> None:
        assert await _embeddings_of(None) == {}

    async def test_bracketed_text(self) -> None:
        assert list((await _embeddings_of("[1.0, 2.0, 3.0]")).values()) == [[1.0, 2.0, 3.0]]

    async def test_list_passthrough(self) -> None:
        assert list((await _embeddings_of([1, 2])).values()) == [[1.0, 2.0]]

    async def test_unparseable_is_omitted(self) -> None:
        assert await _embeddings_of(123) == {}


class TestRowToSnapshot:
    async def test_platform_scope_derived(self) -> None:
        ds = uuid7()
        row = {
            "id": str(uuid7()),
            "customer_id": None,
            "user_id": None,
            "origin_entry_id": None,
            "title": "Filter",
            "body": "exclude deleted",
            "tags": ["a", "b"],
            "datasource_id": str(ds),
            "always_inject": True,
        }
        (snap,) = await _entries_over([row]).list_visible_to_user(uuid7(), customer_scope=uuid7())
        assert isinstance(snap.id, UUID)
        assert snap.scope == Scope.PLATFORM
        assert snap.tags == ("a", "b")
        assert snap.always_inject is True
        assert snap.datasource_id == ds

    async def test_customer_scope_derived(self) -> None:
        row = {
            "id": str(uuid7()),
            "customer_id": str(uuid7()),
            "user_id": None,
            "title": "t",
            "body": "b",
            "tags": None,
            "always_inject": False,
        }
        (snap,) = await _entries_over([row]).list_visible_to_user(uuid7(), customer_scope=uuid7())
        assert snap.scope == Scope.CUSTOMER
        assert snap.tags == ()


class TestRowToConceptSnapshot:
    async def test_builds_with_table_ref(self) -> None:
        row = {
            "id": str(uuid7()),
            "customer_id": None,
            "user_id": None,
            "origin_concept_id": None,
            "name": "active users",
            "aliases": ["actives"],
            "definition": "seen in last 30 days",
            "datasource_id": str(uuid7()),
            "datasource_table_id": str(uuid7()),
            "sql_fragment": "last_seen > now() - interval '30 days'",
            "caveats": "excludes staff",
            "tags": ["metric"],
            "always_inject": False,
            "bound_schema_name": "public",
            "bound_table_name": "users",
        }
        (snap,) = await _concepts_over([row]).list_visible_to_user(uuid7(), customer_scope=uuid7())
        assert snap.name == "active users"
        assert snap.aliases == ("actives",)
        assert snap.tags == ("metric",)
        assert snap.scope == Scope.PLATFORM
        assert snap.datasource_table_ref == build_table_ref("public", "users")
        assert snap.datasource_table_ref is not None
        assert "users" in snap.datasource_table_ref


class TestFetchEmbeddings:
    @pytest.mark.parametrize(
        ("collection_class", "table"),
        [(PlaybookEntryCollection, "playbook_entries"), (ConceptCollection, "concepts")],
        ids=["playbook_entries", "concepts"],
    )
    async def test_null_embedding_omitted_and_ids_coerced(self, collection_class: Any, table: str) -> None:
        a, b = uuid7(), uuid7()
        coll = collection_class.__new__(collection_class)
        coll.l3_pool = _StubPool(
            [
                {"id": str(a), "embedding": "[1.0, 2.0]"},
                {"id": str(b), "embedding": None},
            ]
        )
        out = await coll.fetch_embeddings([a, b], customer_scope=uuid7())
        assert out == {a: [1.0, 2.0]}
        assert coll.l3_pool.sql is not None
        assert "ANY($1)" in coll.l3_pool.sql
        assert f"FROM {table}" in coll.l3_pool.sql

    async def test_empty_ids_issues_no_query(self) -> None:
        coll = _concepts_over([])
        out = await coll.fetch_embeddings([], customer_scope=uuid7())
        assert out == {}
        assert coll.l3_pool.sql is None

    async def test_none_pool_returns_empty(self) -> None:
        coll = ConceptCollection.__new__(ConceptCollection)
        coll.l3_pool = None
        out = await coll.fetch_embeddings([uuid7()], customer_scope=uuid7())
        assert out == {}


def _entry_row() -> dict[str, Any]:
    return {
        "id": str(uuid7()),
        "customer_id": None,
        "user_id": None,
        "origin_entry_id": None,
        "title": "Filter",
        "body": "exclude deleted",
        "tags": None,
        "datasource_id": str(uuid7()),
        "always_inject": False,
    }


class TestPlaybookEntryCollectionSql:
    async def test_list_visible_builds_active_filtered_sql(self) -> None:
        coll = PlaybookEntryCollection.__new__(PlaybookEntryCollection)
        coll.l3_pool = _StubPool([_entry_row()])
        snaps = await coll.list_visible_to_user(uuid7(), customer_scope=uuid7())
        assert len(snaps) == 1
        sql = coll.l3_pool.sql
        assert sql is not None
        assert "FROM playbook_entries" in sql
        assert "status = 'active'" in sql
        assert "datasource_id IN" not in sql  # no domain filter without datasource_id

    async def test_list_visible_adds_datasource_gather(self) -> None:
        coll = PlaybookEntryCollection.__new__(PlaybookEntryCollection)
        coll.l3_pool = _StubPool([])
        await coll.list_visible_to_user(uuid7(), datasource_id=uuid7(), customer_scope=uuid7())
        assert "datasource_id IN" in coll.l3_pool.sql

    async def test_list_visible_no_pool_returns_empty(self) -> None:
        coll = PlaybookEntryCollection.__new__(PlaybookEntryCollection)
        coll.l3_pool = None
        assert await coll.list_visible_to_user(uuid7(), customer_scope=uuid7()) == []

    async def test_list_own_drafts_builds_draft_views(self) -> None:
        coll = PlaybookEntryCollection.__new__(PlaybookEntryCollection)
        coll.l3_pool = _StubPool(
            [
                {
                    "id": str(uuid7()),
                    "title": "T",
                    "body": "B",
                    "datasource_id": str(uuid7()),
                    "conversation_id": None,
                    "turn_count": 3,
                }
            ]
        )
        drafts = await coll.list_own_drafts(uuid7(), customer_scope=uuid7())
        assert len(drafts) == 1
        assert drafts[0].target == "entry"
        assert drafts[0].turn_count == 3
        assert "status = 'draft'" in coll.l3_pool.sql


class TestConceptCollectionSql:
    async def test_list_visible_adds_table_filter(self) -> None:
        coll = ConceptCollection.__new__(ConceptCollection)
        coll.l3_pool = _StubPool([])
        await coll.list_visible_to_user(
            uuid7(),
            datasource_id=uuid7(),
            datasource_table_id=uuid7(),
            customer_scope=uuid7(),
        )
        sql = coll.l3_pool.sql
        assert sql is not None
        assert "FROM concepts" in sql
        assert "status = 'active'" in sql
        assert "datasource_table_id =" in sql

    async def test_list_own_drafts_target_concept(self) -> None:
        coll = ConceptCollection.__new__(ConceptCollection)
        coll.l3_pool = _StubPool(
            [
                {
                    "id": str(uuid7()),
                    "name": "N",
                    "definition": "D",
                    "datasource_id": str(uuid7()),
                    "conversation_id": None,
                    "turn_count": None,
                }
            ]
        )
        drafts = await coll.list_own_drafts(uuid7(), customer_scope=uuid7())
        assert drafts[0].target == "concept"


class _WriteLandsDuringReadPool:
    """L3 pool whose first read is overtaken by a write that commits while it is in flight.

    The first ``fetch`` returns the row set as it stood BEFORE the write, and -- while
    that read is still in flight -- runs the write's post-commit eviction exactly as
    :meth:`CollectionRegistry.publish_invalidation` runs it: ``drop_for_table`` on the
    pod's scan cache. Every later ``fetch`` returns the row set AFTER the write.

    That ordering is the whole defect: the eviction lands between the read and the
    cache store, so it evicts nothing, and the pre-write result is then stored and
    served until the TTL backstop.
    """

    def __init__(
        self,
        scan_cache: ScanCache,
        table: str,
        before: list[dict[str, Any]],
        after: list[dict[str, Any]],
    ) -> None:
        self._scan_cache = scan_cache
        self._table = table
        self._before = before
        self._after = after
        self.fetches = 0

    async def fetch(self, sql: str, *params: Any, customer_scope: Any) -> list[dict[str, Any]]:
        self.fetches += 1
        if self.fetches == 1:
            self._scan_cache.drop_for_table(self._table)
            return list(self._before)
        return list(self._after)


def _registry_with_scan_cache(scan_cache: ScanCache, pool: Any) -> Any:
    """a registry stand-in carrying a REAL scan cache over a REAL L1 backend.

    :param scan_cache: the pod's scan cache
    :ptype scan_cache: ScanCache
    :param pool: the L3 pool the collection reads through
    :ptype pool: Any
    :return: a registry stand-in
    :rtype: Any
    """
    registry = MagicMock()
    registry.get_l1_backend.return_value = None
    registry.scan_cache = scan_cache
    registry.get_l3_pool.return_value = pool
    registry.register.return_value = None
    registry.publish_invalidation = AsyncMock(return_value=None)
    return registry


def _config() -> Any:
    config = MagicMock()
    config.collection_flush = "ALWAYS"
    config.collection_flush_tables = ""
    return config


class TestAScanOvertakenByAWriteIsNotCached:
    """a scan read before a write and stored after that write's eviction must not be served.

    Without a read token, ``put`` cannot tell a result read before the eviction from one
    read after it, so the stale result is cached and served for the whole TTL.
    """

    async def test_concept_scan(self) -> None:
        cache = ScanCache(SQLiteBackend(), trusted=lambda _tables: True)
        before = [_concept_row()]
        after = [_concept_row(), _concept_row()]
        pool = _WriteLandsDuringReadPool(cache, "concepts", before, after)
        registry = _registry_with_scan_cache(cache, pool)
        coll = ConceptCollection(registry=registry, config=_config(), nats_client=None)
        user_id, customer_scope = uuid7(), uuid7()

        first = await coll.list_visible_to_user(user_id, customer_scope=customer_scope)
        second = await coll.list_visible_to_user(user_id, customer_scope=customer_scope)

        assert len(first) == 1
        assert len(second) == 2, "the pre-write scan was cached after the write evicted it"
        assert pool.fetches == 2

    async def test_entry_scan(self) -> None:
        cache = ScanCache(SQLiteBackend(), trusted=lambda _tables: True)
        before = [_entry_row()]
        after = [_entry_row(), _entry_row()]
        pool = _WriteLandsDuringReadPool(cache, "role_assignments", before, after)
        registry = _registry_with_scan_cache(cache, pool)
        coll = PlaybookEntryCollection(registry=registry, config=_config(), nats_client=None)
        user_id, customer_scope = uuid7(), uuid7()

        first = await coll.list_visible_to_user(user_id, customer_scope=customer_scope)
        second = await coll.list_visible_to_user(user_id, customer_scope=customer_scope)

        assert len(first) == 1
        assert len(second) == 2, "a scan read before a grant change was cached after its eviction"
        assert pool.fetches == 2

    async def test_an_undisturbed_scan_is_still_cached(self) -> None:
        """the guard refuses only overtaken reads; an ordinary read still caches."""
        cache = ScanCache(SQLiteBackend(), trusted=lambda _tables: True)
        pool = _StubPool([_concept_row()])
        registry = _registry_with_scan_cache(cache, pool)
        coll = ConceptCollection(registry=registry, config=_config(), nats_client=None)
        user_id, customer_scope = uuid7(), uuid7()

        await coll.list_visible_to_user(user_id, customer_scope=customer_scope)
        pool.sql = None
        again = await coll.list_visible_to_user(user_id, customer_scope=customer_scope)

        assert len(again) == 1
        assert pool.sql is None, "an undisturbed scan must be served from the cache"


def _concept_row() -> dict[str, Any]:
    return {
        "id": str(uuid7()),
        "customer_id": None,
        "user_id": None,
        "origin_concept_id": None,
        "name": "active users",
        "aliases": [],
        "definition": "seen in last 30 days",
        "datasource_id": str(uuid7()),
        "datasource_table_id": None,
        "sql_fragment": None,
        "caveats": None,
        "tags": [],
        "always_inject": False,
        "bound_schema_name": None,
        "bound_table_name": None,
    }


class TestAnOriginLinkChangeEvictsTheDatasourceScan:
    """a datasource-scoped scan reads ``datasources`` for the origin link, so a write there evicts it.

    The hub writes the link through ``CapabilitySourceCollection.save_entity`` with its
    NATS client, which broadcasts an invalidation on ``datasources``. A scan that does
    not declare the table keeps serving the pre-link knowledge set until the TTL.
    """

    @pytest.mark.parametrize(
        ("collection_class", "row"),
        [(ConceptCollection, _concept_row), (PlaybookEntryCollection, _entry_row)],
        ids=["concepts", "playbook_entries"],
    )
    async def test_a_datasources_invalidation_drops_the_cached_scan(self, collection_class: Any, row: Any) -> None:
        cache = ScanCache(SQLiteBackend(), trusted=lambda _tables: True)
        pool = _StubPool([row()])
        registry = _registry_with_scan_cache(cache, pool)
        coll = collection_class(registry=registry, config=_config(), nats_client=None)
        user_id, customer_scope, datasource_id = uuid7(), uuid7(), uuid7()

        await coll.list_visible_to_user(user_id, datasource_id=datasource_id, customer_scope=customer_scope)
        pool.sql = None
        cache.drop_for_table("datasources")
        await coll.list_visible_to_user(user_id, datasource_id=datasource_id, customer_scope=customer_scope)

        assert pool.sql is not None, "a changed origin link must re-read the scan, not serve the cached one"


class TestTheScannedTablesAreSwitchedOn:
    """a cached scan is trusted only while its tables are followed, and only a switched-on table is announced."""

    async def test_every_knowledge_scan_table_advances_its_write_generation(self) -> None:
        import threetears.datasources.collections  # noqa: F401  -- the datasource classes declare theirs
        from threetears.agent.knowledge.collections import KNOWLEDGE_SCAN_TABLES
        from threetears.core.collections import tables_with_write_generation

        assert set(KNOWLEDGE_SCAN_TABLES) == {"concepts", "playbook_entries", "datasources", "datasource_tables"}
        assert set(KNOWLEDGE_SCAN_TABLES) <= tables_with_write_generation()

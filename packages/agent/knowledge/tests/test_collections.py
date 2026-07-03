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
from uuid import UUID, uuid7

from threetears.knowledge import Scope, build_table_ref

from threetears.agent.knowledge.collections import (
    ConceptCollection,
    PlaybookEntryCollection,
    _as_uuid,
    _fetch_embeddings,
    _parse_vector_text,
    _row_to_concept_snapshot,
    _row_to_snapshot,
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


class TestAsUuid:
    def test_passthrough_uuid(self) -> None:
        u = uuid7()
        assert _as_uuid(u) is u

    def test_coerces_string(self) -> None:
        u = uuid7()
        assert _as_uuid(str(u)) == u

    def test_none_stays_none(self) -> None:
        assert _as_uuid(None) is None


class TestParseVectorText:
    def test_none_is_none(self) -> None:
        assert _parse_vector_text(None) is None

    def test_bracketed_text(self) -> None:
        assert _parse_vector_text("[1.0, 2.0, 3.0]") == [1.0, 2.0, 3.0]

    def test_list_passthrough(self) -> None:
        assert _parse_vector_text([1, 2]) == [1.0, 2.0]

    def test_unparseable_is_none(self) -> None:
        assert _parse_vector_text(123) is None


class TestRowToSnapshot:
    def test_platform_scope_derived(self) -> None:
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
        snap = _row_to_snapshot(row)
        assert isinstance(snap.id, UUID)
        assert snap.scope == Scope.PLATFORM
        assert snap.tags == ("a", "b")
        assert snap.always_inject is True
        assert snap.datasource_id == ds

    def test_customer_scope_derived(self) -> None:
        row = {
            "id": str(uuid7()),
            "customer_id": str(uuid7()),
            "user_id": None,
            "title": "t",
            "body": "b",
            "tags": None,
            "always_inject": False,
        }
        snap = _row_to_snapshot(row)
        assert snap.scope == Scope.CUSTOMER
        assert snap.tags == ()


class TestRowToConceptSnapshot:
    def test_builds_with_table_ref(self) -> None:
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
        snap = _row_to_concept_snapshot(row)
        assert snap.name == "active users"
        assert snap.aliases == ("actives",)
        assert snap.tags == ("metric",)
        assert snap.scope == Scope.PLATFORM
        assert snap.datasource_table_ref == build_table_ref("public", "users")
        assert snap.datasource_table_ref is not None
        assert "users" in snap.datasource_table_ref


class TestFetchEmbeddings:
    async def test_null_embedding_omitted_and_ids_coerced(self) -> None:
        a, b = uuid7(), uuid7()
        pool = _StubPool(
            [
                {"id": str(a), "embedding": "[1.0, 2.0]"},
                {"id": str(b), "embedding": None},
            ]
        )
        out = await _fetch_embeddings(pool, "playbook_entries", [a, b], customer_scope=uuid7())
        assert out == {a: [1.0, 2.0]}
        assert pool.sql is not None
        assert "ANY($1)" in pool.sql

    async def test_empty_ids_issues_no_query(self) -> None:
        pool = _StubPool([])
        out = await _fetch_embeddings(pool, "concepts", [], customer_scope=uuid7())
        assert out == {}
        assert pool.sql is None

    async def test_none_pool_returns_empty(self) -> None:
        out = await _fetch_embeddings(None, "concepts", [uuid7()], customer_scope=uuid7())
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

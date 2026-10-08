"""The parts a complete-copy L1 is built from: the key fingerprint, DuckDB's whole-table replace,
the stored keys a copy is checked by, and a collection telling its listeners a row left its L1."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import BigInteger, Column, DateTime, MetaData, String, Table

from threetears.core.fingerprint import key_fingerprint, postgres_fingerprint_sql, relation_key_expression

duckdb = pytest.importorskip("duckdb")

from threetears.core.cache.duckdb import DuckDBBackend  # noqa: E402


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        "results",
        metadata,
        Column("race", String, primary_key=True),
        Column("at", DateTime(timezone=True), primary_key=True),
        Column("votes", BigInteger),
    )
    return metadata


@pytest.fixture()
def backend() -> Any:
    held = DuckDBBackend()
    held.initialize(_metadata())
    yield held
    held.reset()


class TestKeyFingerprint:
    def test_counts_and_sums_the_leading_hex_of_each_keys_md5(self) -> None:
        keys = [("VA", "senate"), ("MD", None)]
        expected = sum(int(hashlib.md5(text.encode()).hexdigest()[:8], 16) for text in ("VA\x1fsenate", "MD\x1f\x1e"))
        assert key_fingerprint(keys).row_count == 2
        assert key_fingerprint(keys).digest == str(expected)

    def test_a_null_is_not_an_empty_string(self) -> None:
        assert key_fingerprint([("VA", None)]) != key_fingerprint([("VA", "")])

    def test_a_value_moving_across_the_column_boundary_is_seen(self) -> None:
        assert key_fingerprint([("a", "bc")]) != key_fingerprint([("ab", "c")])

    def test_the_same_keys_in_another_order_have_one_fingerprint(self) -> None:
        assert key_fingerprint([("a",), ("b",)]) == key_fingerprint([("b",), ("a",)])

    def test_no_keys_is_an_empty_relation(self) -> None:
        assert (key_fingerprint([]).row_count, key_fingerprint([]).digest) == (0, "0")

    def test_the_postgres_statement_renders_keys_by_the_one_rule(self) -> None:
        sql = postgres_fingerprint_sql('"results"', ["race", "at"], " WHERE race = $1")
        assert relation_key_expression(["race", "at"]) in sql
        assert sql.endswith('FROM "results" WHERE race = $1) AS fingerprint_source')

    def test_an_empty_key_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one ordering column"):
            relation_key_expression([])


class TestReplaceAll:
    def test_the_table_holds_exactly_the_rows_given(self, backend: Any) -> None:
        at = datetime(2026, 11, 4, 3, 57, 11, tzinfo=UTC)
        backend.upsert_many("results", [{"race": "old", "at": at, "votes": 1}], ("race", "at"))
        written = backend.replace_all(
            "results", [{"race": "VA", "at": at, "votes": 5}, {"race": "MD", "at": at, "votes": 7}], ("race", "at")
        )
        held = backend.execute_query("SELECT race, votes FROM results ORDER BY race")
        assert written == 2
        assert held == [{"race": "MD", "votes": 7}, {"race": "VA", "votes": 5}]

    def test_no_rows_empties_the_table(self, backend: Any) -> None:
        at = datetime(2026, 11, 4, tzinfo=UTC)
        backend.upsert_many("results", [{"race": "old", "at": at, "votes": 1}], ("race", "at"))
        backend.replace_all("results", [], ("race", "at"))
        assert backend.execute_query("SELECT COUNT(*) AS n FROM results") == [{"n": 0}]

    def test_an_unknown_table_is_refused(self, backend: Any) -> None:
        with pytest.raises(ValueError, match="unknown table"):
            backend.replace_all("nope", [], ("id",))

    def test_a_failing_insert_leaves_the_rows_held_before(self, backend: Any) -> None:
        at = datetime(2026, 11, 4, tzinfo=UTC)
        backend.upsert_many("results", [{"race": "old", "at": at, "votes": 1}], ("race", "at"))
        with pytest.raises(duckdb.Error):
            backend.replace_all("results", [{"race": "VA", "at": at, "votes": "not a number"}], ("race", "at"))
        assert backend.execute_query("SELECT race FROM results") == [{"race": "old"}]


class TestStoredKeys:
    def test_keys_come_back_as_stored_and_match_the_values_serialized(self, backend: Any) -> None:
        at = datetime(2026, 11, 4, 3, 57, 11, tzinfo=UTC)
        backend.upsert_many("results", [{"race": "VA", "at": at, "votes": 5}], ("race", "at"))
        stored = backend.stored_keys("results", ("race", "at"))
        assert stored == [("VA", backend.serialize_value(at, "VARCHAR_DATETIME"))]
        assert key_fingerprint(stored) == key_fingerprint([("VA", at.isoformat())])


class TestEvictionListeners:
    def _collection(self, backend: Any) -> Any:
        from threetears.core.collections.registry import CollectionRegistry
        from threetears.core.collections.schema_backed import (
            BIGINT_TYPE,
            DATETIMETZ_TYPE,
            STRING_TYPE,
            TableSchema,
            collection_for_schema,
        )
        from threetears.core.collections.schema_backed import Column as SchemaColumn
        from threetears.core.config import DefaultCoreConfig

        schema = TableSchema(
            name="results",
            primary_key=("race", "at"),
            columns=[
                SchemaColumn("race", STRING_TYPE),
                SchemaColumn("at", DATETIMETZ_TYPE),
                SchemaColumn("votes", BIGINT_TYPE, nullable=True),
            ],
        )
        registry = CollectionRegistry()
        registry.configure(l1_backend=backend)
        from threetears.core.entities.base import BaseEntity

        class _Result(BaseEntity):
            primary_key_field = "at"

        return collection_for_schema(schema, entity_class=_Result)(registry, DefaultCoreConfig(), None)

    def test_a_listener_hears_every_row_that_leaves_l1(self, backend: Any) -> None:
        collection = self._collection(backend)
        heard: list[Any] = []
        collection.add_l1_change_listener(heard.append)
        collection.evict_from_cache_sync(("VA", "2026-11-04T03:57:11+00:00"))
        assert heard == [("VA", "2026-11-04T03:57:11+00:00")]

    def test_the_collection_names_its_l1_backend(self, backend: Any) -> None:
        assert self._collection(backend).l1_backend is backend

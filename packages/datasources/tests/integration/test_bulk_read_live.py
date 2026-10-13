"""A relation read by parts in bulk, against a real Postgres: grouped fingerprints and the rail's pages read side by side.

The grouped fingerprint must answer, for every value at once, exactly what a fingerprint of that
value's rows answers alone, or a caller comparing them would see every part as moved. The rail's
bulk read (``read_partitions``) must hand back every part whole, from pages whose starts one
statement answered, and refuse a part written while it was read. A client here answers the hub's
asks through the asyncpg driver itself.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid7

import asyncpg
import pytest

from threetears.datasources.config import PostgresConnectionConfig
from threetears.datasources.drivers.asyncpg_driver import AsyncpgDriver
from threetears.datasources.entities import DataSourceType
from threetears.datasources.partitioned_read import read_partitions
from threetears.datasources.query_client import (
    DatasourceQueryResult,
    IncompleteReadError,
    RelationFingerprintResult,
)

pytestmark = pytest.mark.integration


def _parse_db_url(db_url: str) -> dict[str, Any]:
    """the container URL's connection parts."""
    parts = urlsplit(db_url)
    return {
        "host": parts.hostname or "localhost",
        "port": parts.port or 5432,
        "database": (parts.path or "/postgres").lstrip("/"),
        "username": parts.username or "postgres",
        "password": parts.password or "",
    }


_SCHEMA = "ds_it_bulk"
_RELATION = f"{_SCHEMA}.results"
_COLUMNS = ["state", "race", "county", "votes"]
_KEY = ["race", "county"]


class _Client:
    """the hub's three asks, answered by the driver: a query, a fingerprint, a grouped fingerprint."""

    def __init__(self, driver: AsyncpgDriver) -> None:
        self._driver = driver
        self.queries: list[str] = []
        self.after_first_page: Callable[[], Any] | None = None

    async def query(self, datasource_name: str, sql: str, *, params: Sequence[Any] = ()) -> DatasourceQueryResult:
        self.queries.append(sql)
        rows = await self._driver.fetch(sql, *params)
        if self.after_first_page is not None and "LIMIT" in sql:
            hook, self.after_first_page = self.after_first_page, None
            await hook()
        return DatasourceQueryResult(rows=rows, row_count=len(rows), truncated=len(rows) > 1000, correlation_id=_ID)

    async def relation_fingerprint(
        self, datasource_name: str, *, relation: str, key: Sequence[str], where: Mapping[str, str] | None = None
    ) -> RelationFingerprintResult:
        found = await self._driver.relation_fingerprint(relation, list(key), where)
        return RelationFingerprintResult(row_count=found["row_count"], digest=found["digest"])

    async def relation_fingerprint_groups(
        self,
        datasource_name: str,
        *,
        relation: str,
        key: Sequence[str],
        group_by: str,
        where: Mapping[str, str] | None = None,
        where_in: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str | None, RelationFingerprintResult]:
        found = await self._driver.relation_fingerprint_groups(relation, list(key), group_by, where, where_in)
        return {v: RelationFingerprintResult(row_count=f["row_count"], digest=f["digest"]) for v, f in found.items()}


_ID = uuid7()


@pytest.fixture
async def warehouse(db_container: str, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[_Client, Any]]:
    parsed = _parse_db_url(db_container)
    conn = await asyncpg.connect(
        host=parsed["host"],
        port=parsed["port"],
        database=parsed["database"],
        user=parsed["username"],
        password=parsed["password"],
    )
    await conn.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
    await conn.execute(f"CREATE SCHEMA {_SCHEMA}")
    await conn.execute(f"CREATE TABLE {_RELATION} (state text NOT NULL, race text, county text, votes integer)")
    rows = [
        (state, f"r{race}", f"c{county:03d}", race * county)
        for state, races in (("DE", 2), ("TX", 7), ("VA", 4))
        for race in range(races)
        for county in range(13)
    ]
    await conn.executemany(f"INSERT INTO {_RELATION} VALUES ($1, $2, $3, $4)", rows)
    monkeypatch.setenv("BULK_TEST_PW", parsed["password"])
    driver = AsyncpgDriver(
        PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host=parsed["host"],
            port=parsed["port"],
            database=parsed["database"],
            username=parsed["username"],
            password_ref="env://BULK_TEST_PW",
            pool_min_size=1,
            pool_max_size=4,
            command_timeout_seconds=10,
            allowed_schemas=[],
        )
    )
    try:
        yield _Client(driver), conn
    finally:
        await driver.close()
        await conn.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
        await conn.close()


async def _expected(conn: Any, state: str) -> list[dict[str, Any]]:
    return [
        dict(r) for r in await conn.fetch(f"SELECT * FROM {_RELATION} WHERE state = $1 ORDER BY race, county", state)
    ]


@pytest.mark.asyncio
async def test_a_grouped_fingerprint_is_each_groups_own_fingerprint(warehouse: tuple[_Client, Any]) -> None:
    client, _ = warehouse
    groups = await client.relation_fingerprint_groups("w", relation=_RELATION, key=_COLUMNS, group_by="state")
    for state in ("DE", "TX", "VA"):
        alone = await client.relation_fingerprint("w", relation=_RELATION, key=_COLUMNS, where={"state": state})
        assert groups[state] == alone
    some = await client.relation_fingerprint_groups(
        "w", relation=_RELATION, key=_COLUMNS, group_by="state", where_in={"state": ["TX", "XX"]}
    )
    assert set(some) == {"TX"}


@pytest.mark.asyncio
async def test_every_part_comes_back_whole_from_pages_read_side_by_side(warehouse: tuple[_Client, Any]) -> None:
    client, conn = warehouse

    read = [
        (part, rows)
        async for part, rows in read_partitions(
            client,  # type: ignore[arg-type]
            "w",
            columns=_COLUMNS,
            relation=_RELATION,
            key=_KEY,
            partition_by="state",
            page_size=5,
            concurrency=3,
            batch_rows=40,
        )
    ]

    assert [part for part, _ in read] == ["DE", "TX", "VA"]
    for part, rows in read:
        assert rows == await _expected(conn, part)
    # one statement answered every page start of a batch; no page was read one after another from its start
    assert sum("ROW_NUMBER()" in sql for sql in client.queries) == 3


@pytest.mark.asyncio
async def test_parts_asked_for_are_read_alone_and_an_empty_one_comes_back_empty(warehouse: tuple[_Client, Any]) -> None:
    client, conn = warehouse

    read = {
        part: rows
        async for part, rows in read_partitions(
            client,  # type: ignore[arg-type]
            "w",
            columns=_COLUMNS,
            relation=_RELATION,
            key=_KEY,
            partition_by="state",
            parts=["VA", "WY"],
            page_size=5,
        )
    }

    assert read == {"VA": await _expected(conn, "VA"), "WY": []}


@pytest.mark.asyncio
async def test_a_part_written_while_it_is_read_is_refused(warehouse: tuple[_Client, Any]) -> None:
    client, conn = warehouse

    async def write() -> None:
        await conn.execute(f"UPDATE {_RELATION} SET votes = votes + 1 WHERE state = 'DE' AND race = 'r1'")

    client.after_first_page = write
    with pytest.raises(IncompleteReadError, match="changed while it was read"):
        async for _ in read_partitions(
            client,  # type: ignore[arg-type]
            "w",
            columns=_COLUMNS,
            relation=_RELATION,
            key=_KEY,
            partition_by="state",
            page_size=5,
        ):
            pass

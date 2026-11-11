"""Integration: copy_keyed streams the COPY CSV text the hand-written ``copy_from_query`` it replaces streams.

Byte for byte, over a table seeded with no rows, one, many, NULLs (an unquoted empty field) beside empty strings
(quoted, ``force_quote``), rows tied on the order and two tenants, the prefix pinned to one of them. On the session's
``db_container`` (PostgreSQL), or YugabyteDB through ``THREETEARS_TEST_POSTGRES_URL``; a schema of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import asyncpg
import pytest

from threetears.core.keyed import Selected, Sort, copy_keyed

pytestmark = pytest.mark.integration

_START = datetime(2026, 11, 3, 20, tzinfo=UTC)
_TIME = Selected("""to_char(t."admitted" AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')""", "admitted")


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    schema = f"copy_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        await connection.execute(
            "CREATE TABLE c (kind text NOT NULL, owner text NOT NULL, id integer NOT NULL, admitted timestamptz, "
            "note text, PRIMARY KEY (id))"
        )
        await connection.execute("CREATE INDEX c_by_owner ON c (kind ASC, owner ASC, admitted ASC)")
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


async def _hand_written(conn: asyncpg.Connection, kind: str, owner: str) -> bytes:
    chunks: list[bytes] = []

    async def sink(chunk: bytes) -> None:
        chunks.append(chunk)

    await conn.copy_from_query(
        """SELECT owner, id, to_char(admitted AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"') AS admitted, note """
        "FROM c WHERE kind = $1 AND owner = $2 ORDER BY admitted, id",
        kind,
        owner,
        output=sink,
        format="csv",
        force_quote=["note"],
    )
    return b"".join(chunks)


@pytest.mark.parametrize("rows", [0, 1, 40])
async def test_the_copy_text_is_the_hand_written_copy_s_byte_for_byte(conn: asyncpg.Connection, rows: int) -> None:
    """Flips if copy_keyed streams other bytes than the hand-written COPY: another order (ties on the time broken by
    id), a NULL not an empty field, an empty string not quoted, the other tenant's rows, or a header."""
    seeded = [
        (
            "agent" if index % 5 else "tool",
            f"o{index % 2}",
            index,
            None if index % 7 == 3 else _START + timedelta(minutes=index // 3),
            None if index % 4 == 1 else ("" if index % 4 == 2 else f'n,"{index}"'),
        )
        for index in range(rows)
    ]
    await conn.executemany("INSERT INTO c VALUES ($1, $2, $3, $4, $5)", seeded)
    for kind, owner in (("agent", "o0"), ("agent", "o1"), ("tool", "o0")):
        streamed = await copy_keyed(
            conn,
            "c",
            columns=["owner", "id", _TIME, "note"],
            by={"kind": kind, "owner": owner},
            order=[Sort("admitted", cast="timestamptz", nullable=True), Sort("id", cast="integer")],
            force_quote=["note"],
        )
        assert streamed == await _hand_written(conn, kind, owner), (kind, owner)

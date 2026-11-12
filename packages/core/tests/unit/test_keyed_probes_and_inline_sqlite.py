"""the SDK's last hand-written statements, each with its helper (hub #39): a grant's access probe
(:func:`access_probe_sql`, :func:`probe_access`), a one-row table written as literal text for a CLI
(:func:`literal_upsert` beside :func:`literal_select`), and a synchronous L1's cap trim
(:class:`InlineSQLite`, :func:`delete_oldest` with ``order=``, :func:`read_value`, driven by :func:`run_inline`)."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from threetears.core.keyed import (
    InlineSQLite,
    access_probe_sql,
    delete_oldest,
    literal_select,
    literal_upsert,
    probe_access,
    read_value,
    run_inline,
)


class _Warehouse:
    """a warehouse driver's session: ``fetch`` and ``execute_rolled_back``."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def fetch(self, sql: str) -> list[Any]:
        self.sent.append(("fetch", sql))
        return []

    async def execute_rolled_back(self, sql: str) -> None:
        self.sent.append(("rolled back", sql))


class _Transaction:
    def __init__(self, sent: list[tuple[str, str]]) -> None:
        self.sent = sent

    async def start(self) -> None:
        self.sent.append(("begin", ""))

    async def rollback(self) -> None:
        self.sent.append(("rollback", ""))


class _Asyncpg:
    """an asyncpg connection: ``execute`` and ``transaction()``; its insert refused when ``refuse``."""

    def __init__(self, refuse: bool = False) -> None:
        self.sent: list[tuple[str, str]] = []
        self.refuse = refuse

    def transaction(self) -> _Transaction:
        return _Transaction(self.sent)

    async def execute(self, sql: str) -> str:
        self.sent.append(("execute", sql))
        if self.refuse:
            raise PermissionError("permission denied for table t")
        return "INSERT 0 1"


def test_the_access_probes_ask_for_one_row() -> None:
    """Flips if the read probe asks for no row (Redshift answers LIMIT 0 without checking the grant) or the write
    probe stops being one row's insert from the table itself."""
    assert access_probe_sql("reporting.results") == "SELECT 1 FROM reporting.results LIMIT 1"
    assert (
        access_probe_sql("reporting.results", write=True)
        == "INSERT INTO reporting.results SELECT * FROM reporting.results LIMIT 1"
    )


async def test_a_write_probe_is_always_rolled_back() -> None:
    """Flips if the read probe is not a fetch, the write probe is not sent rolled back through a warehouse driver,
    or an asyncpg write probe keeps its row (no rollback), even when the insert is refused."""
    warehouse = _Warehouse()
    await probe_access(warehouse, "s.t")
    await probe_access(warehouse, "s.t", write=True)
    assert warehouse.sent == [
        ("fetch", "SELECT 1 FROM s.t LIMIT 1"),
        ("rolled back", "INSERT INTO s.t SELECT * FROM s.t LIMIT 1"),
    ]
    refused = _Asyncpg(refuse=True)
    with pytest.raises(PermissionError):
        await probe_access(refused, "s.t", write=True)
    assert refused.sent == [("begin", ""), ("execute", "INSERT INTO s.t SELECT * FROM s.t LIMIT 1"), ("rollback", "")]


def test_a_one_row_table_is_written_and_read_by_its_fixed_key_as_literal_text() -> None:
    """Flips if the literal upsert of a one-row table (a dev fixture's marker, sent through ysqlsh) stops writing by
    its key, stops quoting a value, or takes a row missing its key; and its read back by the same key."""
    assert literal_upsert("public.dataset_fixture_state", {"id": 1, "fingerprint": "ab'c"}, key=["id"]) == (
        "INSERT INTO public.dataset_fixture_state (\"id\", \"fingerprint\") VALUES (1, 'ab''c') "
        'ON CONFLICT ("id") DO UPDATE SET "fingerprint" = EXCLUDED."fingerprint"'
    )
    assert literal_upsert("t", {"id": 1}, key=["id"]).endswith('ON CONFLICT ("id") DO NOTHING')
    with pytest.raises(ValueError, match="names its key"):
        literal_upsert("t", {"fingerprint": "x"}, key=["id"])
    with pytest.raises(ValueError, match="names its key"):
        literal_upsert("t", {"id": 1}, key=[])
    assert literal_select("public.dataset_fixture_state", columns=["fingerprint"], by={"id": 1}).endswith(
        'WHERE t."id" = 1'
    )


def _cache() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE turn_dedup (key TEXT PRIMARY KEY, date_updated TEXT)")
    conn.execute("CREATE INDEX turn_dedup_age ON turn_dedup (date_updated)")
    conn.execute("CREATE TABLE turn_dedup_rows (id INTEGER PRIMARY KEY, n INTEGER)")
    # five rows, three sharing the oldest date, the newest first by key: the age is no key and not the key's order
    conn.executemany(
        "INSERT INTO turn_dedup VALUES (?, ?)",
        [("a", "2026-01-03"), ("e", "2026-01-01"), ("c", "2026-01-01"), ("b", "2026-01-01"), ("d", "2026-01-02")],
    )
    conn.execute("INSERT INTO turn_dedup_rows VALUES (0, 5)")
    return conn


class _Backend:
    """an L1 backend: answers its connection from ``get_connection()``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def get_connection(self) -> sqlite3.Connection:
        return self.conn


def test_a_synchronous_l1_trims_exactly_its_oldest_rows_by_an_age_many_share() -> None:
    """Flips if a synchronous caller cannot read the trigger-kept count by its key, or the trim by a shared age
    deletes more or fewer than asked (the ties of the cut date all going), or not the oldest first."""
    conn = _cache()
    l1 = InlineSQLite(_Backend(conn))
    assert run_inline(read_value(l1, "turn_dedup_rows", by={"id": 0}, column="n")) == 5
    assert run_inline(delete_oldest(l1, "turn_dedup", count=2, key="key", order="date_updated")) == 2
    left = [row[0] for row in conn.execute("SELECT key FROM turn_dedup ORDER BY key")]
    assert left == ["a", "d", "e"]
    assert run_inline(delete_oldest(InlineSQLite(conn), "turn_dedup", count=0, key="key", order="date_updated")) == 0
    assert run_inline(delete_oldest(InlineSQLite(conn), "turn_dedup", count=9, key="key", order="date_updated")) == 3

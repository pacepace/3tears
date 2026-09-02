"""Drift-tolerant comparison — restore a backup and prove it MOSTLY matches the live database.

"Mostly" is the honest word: a live database keeps moving after the backup moment, so byte
equality is the wrong bar and raw row counts lie in the backup's disfavour. What CAN be held to a
hard standard is the past: every row that existed at the backup moment should exist, identical, in
both the restored copy and the live database — unless something mutated or deleted history since,
which is exactly the drift an operator wants SHOWN, not failed on blindly and not hidden.

So the comparator cuts both sides off at the backup's own timestamp before comparing:

- a table with a recognized creation-timestamp column is filtered ``column <= as_of``;
- a table whose single-column primary key is uuid — the platform's uuid7 ids embed their creation
  millisecond in their high bits and the uuid type compares bytewise — is filtered
  ``pk <= uuid7_upper_bound(as_of)``;
- a table with neither is compared on raw counts under tolerance alone, and says so.

Content is compared with an order-independent checksum (sum of per-row hashes) so "same rows,
different physical order" — guaranteed after a restore — reads as equal. Every table gets a named
status; the report's ``ok`` fails only on tables whose drift exceeds the caller's tolerance,
which is the "little drift, but you get me" contract made mechanical.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from threetears.observe import get_logger

__all__ = [
    "ComparisonReport",
    "DriftComparator",
    "TableComparison",
    "uuid7_upper_bound",
]

log = get_logger(__name__)

_TABLES_SQL = """
    SELECT table_schema, table_name
      FROM information_schema.tables
     WHERE table_type = 'BASE TABLE'
       AND table_schema NOT IN ('pg_catalog', 'information_schema')
     ORDER BY table_schema, table_name
"""

_COLUMNS_SQL = """
    SELECT column_name, data_type
      FROM information_schema.columns
     WHERE table_schema = $1 AND table_name = $2
"""

_PK_SQL = """
    SELECT a.attname, format_type(a.atttypid, a.atttypmod) AS type
      FROM pg_index i
      JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
     WHERE i.indrelid = ($1 || '.' || $2)::regclass
       AND i.indisprimary
"""


def uuid7_upper_bound(moment: datetime) -> UUID:
    """The largest possible uuid7 minted at or before ``moment``.

    uuid7's first 48 bits are the unix millisecond; version and variant bits are fixed; every
    remaining bit is random. Setting the timestamp to ``moment``'s millisecond and every random
    bit to 1 yields a value no uuid7 minted later can compare at-or-under — which turns
    ``pk <= bound`` into "created at or before ``moment``" on any uuid7-keyed table.
    """
    ms = int(moment.timestamp() * 1000)
    value = (ms & 0xFFFF_FFFF_FFFF) << 80
    value |= 0x7 << 76  # version 7
    value |= 0xFFF << 64  # rand_a, maxed
    value |= 0b10 << 62  # variant
    value |= (1 << 62) - 1  # rand_b, maxed
    return UUID(int=value)


@runtime_checkable
class _Connection(Protocol):
    async def fetch(self, query: str, *args: object) -> list[Any]: ...
    async def fetchrow(self, query: str, *args: object) -> Any: ...
    async def close(self) -> None: ...


Connect = Callable[[str], Awaitable[_Connection]]


@dataclass(frozen=True, slots=True)
class TableComparison:
    """One table's verdict.

    ``status`` is one of:

    - ``matched`` — counts and checksums equal under the cutoff;
    - ``drifted`` — a difference within tolerance (history was mutated or deleted since the
      backup, or the table had no cutoff column and live growth shows through);
    - ``failed`` — a difference beyond tolerance;
    - ``missing_live`` — the table exists in the restored copy and not live (a dropped table:
      reported, tolerated — dropping tables is a thing operators do on purpose);
    - ``extra_live`` — the table exists live and not in the backup (created since; expected).
    """

    schema: str
    table: str
    status: str
    backup_count: int
    live_count: int
    checksums_match: bool
    cutoff: str
    detail: str = ""

    @property
    def qualified(self) -> str:
        return f"{self.schema}.{self.table}"


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """The whole drill's verdict: per-table results and one honest boolean."""

    as_of: datetime
    tables: tuple[TableComparison, ...]
    ok: bool

    @property
    def failed(self) -> tuple[TableComparison, ...]:
        return tuple(t for t in self.tables if t.status == "failed")

    @property
    def drifted(self) -> tuple[TableComparison, ...]:
        return tuple(t for t in self.tables if t.status == "drifted")


class DriftComparator:
    """Compare a restored backup against the live database, tolerating post-backup drift.

    :param connect: an ``asyncpg.connect``-shaped callable (injected).
    :param restored_dsn: the scratch database the backup was restored into.
    :param live_dsn: the running database it is compared against.
    :param cutoff_columns: creation-timestamp column names recognized for the time cutoff, in
        preference order. The platform convention is ``date_created``; the bare ``timestamp``
        (audit-event shape) sits last, and every candidate is type-checked against
        ``timestamptz`` before use, so a non-time column of that name cannot be misused.
    :param tolerance_fraction: per-table row drift tolerated before a table FAILS —
        ``|backup - live|`` may be up to this fraction of the backup count (with
        ``tolerance_rows`` as an absolute floor so tiny tables aren't held to zero).
    :param tolerance_rows: absolute drift floor per table.
    """

    def __init__(
        self,
        connect: Connect,
        restored_dsn: str,
        live_dsn: str,
        *,
        cutoff_columns: tuple[str, ...] = ("date_created", "created_at", "date_added", "timestamp"),
        tolerance_fraction: float = 0.01,
        tolerance_rows: int = 5,
    ) -> None:
        self._connect = connect
        self._restored_dsn = restored_dsn
        self._live_dsn = live_dsn
        self._cutoff_columns = cutoff_columns
        self._tolerance_fraction = tolerance_fraction
        self._tolerance_rows = tolerance_rows

    async def compare(self, *, as_of: datetime) -> ComparisonReport:
        """Run the comparison, cutting both sides off at ``as_of`` (the backup's timestamp)."""
        restored = await self._connect(self._restored_dsn)
        live = await self._connect(self._live_dsn)
        try:
            restored_tables = {(r["table_schema"], r["table_name"]) for r in await restored.fetch(_TABLES_SQL)}
            live_tables = {(r["table_schema"], r["table_name"]) for r in await live.fetch(_TABLES_SQL)}
            results: list[TableComparison] = []

            for schema, table in sorted(restored_tables - live_tables):
                count = await self._count(restored, schema, table, None, None)
                results.append(
                    TableComparison(
                        schema=schema,
                        table=table,
                        status="missing_live",
                        backup_count=count,
                        live_count=0,
                        checksums_match=False,
                        cutoff="none",
                        detail="table exists in the backup and not live",
                    )
                )
            for schema, table in sorted(live_tables - restored_tables):
                count = await self._count(live, schema, table, None, None)
                results.append(
                    TableComparison(
                        schema=schema,
                        table=table,
                        status="extra_live",
                        backup_count=0,
                        live_count=count,
                        checksums_match=False,
                        cutoff="none",
                        detail="table created since the backup",
                    )
                )

            for schema, table in sorted(restored_tables & live_tables):
                results.append(await self._compare_table(restored, live, schema, table, as_of))
        finally:
            await restored.close()
            await live.close()

        ok = not any(t.status == "failed" for t in results)
        report = ComparisonReport(as_of=as_of, tables=tuple(results), ok=ok)
        log.info(
            "drift comparison complete",
            extra={
                "extra_data": {
                    "tables": len(results),
                    "failed": len(report.failed),
                    "drifted": len(report.drifted),
                    "ok": ok,
                }
            },
        )
        return report

    # ------------------------------------------------------------------ internals

    async def _compare_table(
        self, restored: _Connection, live: _Connection, schema: str, table: str, as_of: datetime
    ) -> TableComparison:
        predicate, param, cutoff = await self._cutoff_for(restored, schema, table, as_of)
        b_count, b_sum = await self._count_and_checksum(restored, schema, table, predicate, param)
        l_count, l_sum = await self._count_and_checksum(live, schema, table, predicate, param)
        checksums_match = b_sum == l_sum
        drift = abs(b_count - l_count)
        allowed = max(self._tolerance_rows, int(b_count * self._tolerance_fraction))

        if b_count == l_count and checksums_match:
            status, detail = "matched", ""
        elif drift <= allowed:
            status = "drifted"
            detail = (
                f"{drift} rows of count drift (allowed {allowed})"
                if not checksums_match or drift
                else "content differs under equal counts"
            )
        else:
            status, detail = "failed", f"{drift} rows of drift exceeds the allowed {allowed}"
        return TableComparison(
            schema=schema,
            table=table,
            status=status,
            backup_count=b_count,
            live_count=l_count,
            checksums_match=checksums_match,
            cutoff=cutoff,
            detail=detail,
        )

    async def _cutoff_for(
        self, conn: _Connection, schema: str, table: str, as_of: datetime
    ) -> tuple[str | None, object | None, str]:
        columns = {r["column_name"]: r["data_type"] for r in await conn.fetch(_COLUMNS_SQL, schema, table)}
        for candidate in self._cutoff_columns:
            if columns.get(candidate, "").startswith("timestamp"):
                return f'"{candidate}" <= $1', as_of, f"column:{candidate}"
        pk = await conn.fetch(_PK_SQL, f'"{schema}"', f'"{table}"')
        if len(pk) == 1 and pk[0]["type"] == "uuid":
            return f'"{pk[0]["attname"]}" <= $1', uuid7_upper_bound(as_of), f"uuid7:{pk[0]['attname']}"
        return None, None, "none"

    async def _count(
        self, conn: _Connection, schema: str, table: str, predicate: str | None, param: object | None
    ) -> int:
        count, _ = await self._count_and_checksum(conn, schema, table, predicate, param)
        return count

    async def _count_and_checksum(
        self, conn: _Connection, schema: str, table: str, predicate: str | None, param: object | None
    ) -> tuple[int, int]:
        where = f"WHERE {predicate}" if predicate else ""
        # sum-of-row-hashes: order-independent, so a restore's different physical order reads equal.
        sql = f'SELECT count(*) AS n, coalesce(sum(hashtext(t::text)::bigint), 0) AS s FROM "{schema}"."{table}" t {where}'
        row = await (conn.fetchrow(sql, param) if param is not None else conn.fetchrow(sql))
        return int(row["n"]), int(row["s"])

"""Integration: what a newest-per-key read costs by the versions each key holds (append-only L3, chunk 3 follow-up).

``scripts/measure-newest-reads.py`` measures the mechanisms at real size (2M rows, 1, 5 and 50 versions a key); this
runs its own statements at a small size on the session ``db_container`` (Postgres in ``integration``, YugabyteDB in
``integration_yugabyte``) and proves what the numbers rest on:

- every mechanism answers the same rows: each key's newest live version, never a gone key;
- the per-key seeks read a bounded number of entries a key whatever its versions (``skip`` two, ``lateral`` one),
  while the shipped guard's rows read grow with them;
- on YugabyteDB the batched ``DISTINCT ON`` key reads (``distinct``, ``distinct_join``) read each batch key's
  versions and no more: rows read grow with the versions, never with the table.

The shipped reads (Pace's choice, 2026-11-25: ``keys.newest_relation`` / ``newest_lateral``) are driven through the
helpers themselves, their statements captured and explained under the bounds each read sets (``newest_settings``):

- a fill's pages read each version of the scope once (rows read about the scope's versions, never a re-read a page);
- on YugabyteDB a keyed batch reads only its keys' versions, in few storage requests;
- at 50 versions a key a batch's and a page's sorts stay in memory under the read's ``work_mem`` (no external sort)
  on both engines, where the default 4 MB spills the same batch's sort on YugabyteDB;
- a listing's pages (``keyset_page``) read only their own range, each version of the scope about once in all, as a
  fill's do (their cursor and ``LIMIT`` inside the newest relation), and on YugabyteDB a batch of a key prefix (a
  state) reads the prefix's scope a range a statement, never its every version in one sort.

It asserts no speed budget. The numbers are logged.
"""

from __future__ import annotations

import importlib.util
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType
from typing import Any

import asyncpg
import pytest

from threetears.core.keyed import KeyBatch, Sort, keyset_page, read_keyed
from threetears.core.keyset import read_keyset_pages
from threetears.interfaces.keyed.keys import NEWEST_FIRST_KEYS, newest_settings, register_versioned_table

pytestmark = pytest.mark.integration

_log = logging.getLogger(__name__)

_ROWS = 40_000


def _script() -> ModuleType:
    path = Path(__file__).resolve().parents[4] / "scripts" / "measure-newest-reads.py"
    spec = importlib.util.spec_from_file_location("measure_newest_reads", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
async def measured(db_container: str) -> AsyncIterator[tuple[Any, ModuleType, bool]]:
    script = _script()
    # a scope a page's worth of keys several times over, at this size (set through the module's namespace: a
    # ModuleType names no attributes to the type checker)
    vars(script).update(SCHEMA=f"nr_{uuid.uuid4().hex[:8]}", KEYS_A_SCOPE=2_000, PAGE=500)
    conn = await asyncpg.connect(db_container)
    try:
        yield conn, script, await script.on_yugabyte(conn)
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {script.SCHEMA} CASCADE")
        await conn.close()


async def _live_ids(conn: Any, sql: str, scope: str, page: int) -> list[int]:
    cursor, ids = -1, []
    while True:
        rows = await conn.fetch(sql, scope, cursor, page)
        ids += [int(row["id"]) for row in rows if row["live"]]
        if len(rows) < page:
            return ids
        cursor = int(rows[-1]["id"])


async def test_every_mechanism_answers_each_key_s_newest_live_version(
    measured: tuple[Any, ModuleType, bool],
) -> None:
    """Flips if a mechanism answers an older version, a gone key, or loses a key; or a per-key seek reads more entries
    a key as its versions grow; or a batched DISTINCT ON reads more than the batch keys' versions."""
    conn, script, yugabyte = measured
    read: dict[tuple[int, str], int] = {}
    for versions in (1, 10):
        await script.load(conn, versions, _ROWS, yugabyte)
        table = f"{script.SCHEMA}.{script.table_name(versions)}"
        expected = [
            int(row["id"])
            for row in await conn.fetch(
                f"SELECT id FROM {table} WHERE state = 's000' AND epoch = $1 AND NOT gone ORDER BY id", versions - 1
            )
        ]
        for mechanism in ("guard", "distinct", "skip"):
            sql = script.fill_page_sql(mechanism, versions)
            assert await _live_ids(conn, sql, "s000", script.PAGE) == expected, (mechanism, versions)
            plan = await script.explain(conn, sql, ("s000", -1, script.PAGE), yugabyte)
            read[(versions, mechanism)] = script.rows_read(plan, yugabyte)
        keys = [("s000", key) for key in range(0, script.KEYS_A_SCOPE, 7)]
        for mechanism in ("guard", "lateral", "distinct", "distinct_join"):
            sql, params = script.batch_sql(mechanism, versions, yugabyte, keys)
            answered = sorted(int(row["id"]) for row in await conn.fetch(sql, *params))
            assert answered == [key for _, key in keys if key in set(expected)], (mechanism, versions)
            read[(versions, f"batch-{mechanism}")] = script.rows_read(
                await script.explain(conn, sql, params, yugabyte), yugabyte
            )
    _log.warning("rows read a first page / a batch, by (versions, mechanism): %s", read)
    for mechanism in ("skip", "batch-lateral"):
        # bounded a key: ten versions read no more than one does, give or take the marker probe's rows
        assert read[(10, mechanism)] <= read[(1, mechanism)] * 1.1 + 10, (mechanism, read)
    assert read[(10, "guard")] > read[(1, "guard")] * 5, read
    if not yugabyte:
        # Postgres at this size (40,000 rows, a 286-key batch) rightly reads the whole table for the batched DISTINCT
        # ON (a hash join beats 286 index seeks); at 2M rows it seeks the keys (measured 2026-11-25)
        return
    for mechanism in ("batch-distinct", "batch-distinct_join"):
        # each batch key's versions (about 286 keys here), give or take the marker probe's rows: never a table scan
        batch = len(range(0, script.KEYS_A_SCOPE, 7))
        for versions in (1, 10):
            assert read[(versions, mechanism)] <= batch * versions * 1.1 + 50, (mechanism, versions, read)


class _Recorder:
    """an L3 reader over one asyncpg connection that keeps every statement it sends: it is no asyncpg connection, so
    the helpers send their statements as they are and the test explains each under the read's bounds itself."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self.sent: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *params: Any) -> list[Any]:
        self.sent.append((query, params))
        return list(await self._conn.fetch(query, *params))


async def _explained(
    conn: Any, script: ModuleType, sql: str, params: Any, yugabyte: bool, *, text: bool, batch: bool
) -> Any:
    """the statement's plan, run under the bounds a newest read (a keyed batch, or a fill page) sets for its own
    transaction."""
    async with conn.transaction():
        for name, value in newest_settings(yugabyte=yugabyte, batch=batch):
            await conn.execute(f"SET LOCAL {name} = '{value}'")
        return await script.explain(conn, sql, params, yugabyte, text=text)


def _spilled(plan: str) -> bool:
    return "external" in plan.lower() or "disk:" in plan.lower()


async def test_the_shipped_reads_read_each_version_once_and_a_batch_its_keys_alone(
    measured: tuple[Any, ModuleType, bool],
) -> None:
    """Flips if a fill's pages read a version of the scope more than about once (a re-read of the scope a page, as
    the retired guard's), if a keyed batch on YugabyteDB reads past its keys' versions or takes many storage requests,
    or if either answers an older version or a gone key."""
    conn, script, yugabyte = measured
    versions = 10
    await script.load(conn, versions, _ROWS, yugabyte)
    name = f"{script.SCHEMA}.{script.table_name(versions)}"
    register_versioned_table(script.declaration(versions))
    expected = [
        int(row["id"])
        for row in await conn.fetch(
            f"SELECT id FROM {name} WHERE state = 's000' AND epoch = $1 AND NOT gone ORDER BY id", versions - 1
        )
    ]
    fill = _Recorder(conn)
    read = await read_keyset_pages(
        fill,
        f"SELECT state, id, v, payload FROM {name}",
        ["state", "id"],
        where={"state": "s000"},
        page_size=script.PAGE,
        table=name,
    )
    assert [int(row["id"]) for row in read.rows] == expected
    pages = [(sql, params) for sql, params in fill.sent if "DISTINCT ON" in sql]
    assert pages, fill.sent
    filled = 0
    for sql, params in pages:
        filled += script.rows_read(
            await _explained(conn, script, sql, params, yugabyte, text=False, batch=False), yugabyte
        )
    scope = script.KEYS_A_SCOPE * versions
    _log.warning("fill of %d versions: %d rows read in %d pages", scope, filled, len(pages))
    # each version once, give or take the key past each page (read again by the next), YugabyteDB's one storage
    # request read past a page's LIMIT (its default 1,024 rows) and the markers' rows
    past = 1_024 if yugabyte else versions
    if filled > scope * 1.02 + len(pages) * past + 50:
        plan_text = await _explained(conn, script, *pages[1], yugabyte, text=True, batch=False)
        raise AssertionError(f"{filled} rows read for {scope} in {len(pages)} pages; page 2:\n{plan_text}")
    batch = _Recorder(conn)
    keys = [("s000", key) for key in range(0, script.KEYS_A_SCOPE, 7)]
    rows = await read_keyed(
        batch,
        name,
        keys=KeyBatch(("state", "id"), [[k[0] for k in keys], [k[1] for k in keys]], ("text", "bigint")),
        columns=["state", "id", "v"],
    )
    assert sorted(int(row["id"]) for row in rows) == [key for _, key in keys if key in set(expected)]
    sql, params = batch.sent[-1]
    plan = await _explained(conn, script, sql, params, yugabyte, text=False, batch=True)
    batched = script.rows_read(plan, yugabyte)
    _log.warning(
        "batch of %d keys: %d rows read, %s requests", len(keys), batched, script.read_requests(plan, yugabyte)
    )
    if not yugabyte:
        # PostgreSQL seeks each key's newest once (LATERAL): one row a key
        assert batched <= len(keys) * 1.1 + 10, batched
        return
    assert "DISTINCT ON" in sql and "YbBatchedNL" in sql, sql
    assert batched <= len(keys) * versions * 1.1 + 50, batched
    # 2,860 versions at yb_fetch_row_limit 10,000: a request or two, never one a key
    assert (script.read_requests(plan, yugabyte) or 0) <= 10, plan


async def test_a_50_version_batch_and_page_sort_in_memory(measured: tuple[Any, ModuleType, bool]) -> None:
    """Flips if, at 50 versions a key, the first keyed batch (NEWEST_FIRST_KEYS keys) or a fill page sorts on disk
    under the read's own work_mem; on YugabyteDB the same batch under the default 4 MB must spill, so the bound is what
    keeps it in memory."""
    conn, script, yugabyte = measured
    versions = 50
    await script.load(conn, versions, 200_000, yugabyte)
    name = f"{script.SCHEMA}.{script.table_name(versions)}"
    register_versioned_table(script.declaration(versions))
    held = [(row["state"], int(row["id"])) for row in await conn.fetch(f"SELECT DISTINCT state, id FROM {name}")]
    keys = sorted(held)[:NEWEST_FIRST_KEYS]
    batch = _Recorder(conn)
    await read_keyed(
        batch,
        name,
        keys=KeyBatch(("state", "id"), [[k[0] for k in keys], [k[1] for k in keys]], ("text", "bigint")),
        columns=["state", "id", "v", "payload"],
        batch=NEWEST_FIRST_KEYS,
    )
    sql, params = batch.sent[-1]
    bounded = await _explained(conn, script, sql, params, yugabyte, text=True, batch=True)
    _log.warning("50-version batch of %d keys, bounded:\n%s", len(keys), bounded)
    assert not _spilled(bounded), bounded
    if yugabyte:
        unbounded = await script.explain(conn, sql, params, yugabyte, text=True)
        assert _spilled(unbounded), unbounded
    fill = _Recorder(conn)
    await read_keyset_pages(
        fill,
        f"SELECT state, id, v, payload FROM {name}",
        ["state", "id"],
        where={"state": "s000"},
        page_size=script.PAGE,
        table=name,
    )
    for sql, params in fill.sent:
        if "DISTINCT ON" in sql:
            plan = await _explained(conn, script, sql, params, yugabyte, text=True, batch=False)
            assert not _spilled(plan), plan


async def test_a_listing_s_pages_and_a_prefix_batch_read_their_range_alone(
    measured: tuple[Any, ModuleType, bool],
) -> None:
    """Flips if a listing's page of a versioned table (keyset_page) reads past its own range (its cursor and LIMIT left
    outside the newest relation, each page reading the scope again), or, on YugabyteDB, a batch of one key prefix
    reads the prefix's whole scope in one statement; or either answers other than each key's newest live version."""
    conn, script, yugabyte = measured
    versions = 10
    await script.load(conn, versions, _ROWS, yugabyte)
    name = f"{script.SCHEMA}.{script.table_name(versions)}"
    register_versioned_table(script.declaration(versions))
    expected = [
        int(row["id"])
        for row in await conn.fetch(
            f"SELECT id FROM {name} WHERE state = 's000' AND epoch = $1 AND NOT gone ORDER BY id", versions - 1
        )
    ]
    scope = script.KEYS_A_SCOPE * versions
    # each version once, give or take the key past each statement, YugabyteDB's one storage request read past a
    # statement's LIMIT (its default 1,024 rows) and the markers' rows
    past = 1_024 if yugabyte else versions
    listing = _Recorder(conn)
    listed: list[int] = []
    cursor = None
    while True:
        page = await keyset_page(
            listing,
            name,
            columns=["state", "id", "v"],
            by={"state": "s000"},
            order=[Sort("id", cast="bigint")],
            limit=script.PAGE,
            cursor=cursor,
        )
        listed += [int(row["id"]) for row in page.rows]
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    assert listed == expected
    statements = [(sql, params) for sql, params in listing.sent if "DISTINCT ON" in sql]
    # every page's statement reads the newest relation (the one other is the engine's version, asked once)
    assert len(statements) == len([sql for sql, _ in listing.sent if "version()" not in sql]) > 1, listing.sent
    paged = 0
    for sql, params in statements:
        paged += script.rows_read(
            await _explained(conn, script, sql, params, yugabyte, text=False, batch=False), yugabyte
        )
    _log.warning("listing of %d versions: %d rows read in %d statements", scope, paged, len(statements))
    if paged > scope * 1.02 + len(statements) * past + 50:
        plan_text = await _explained(conn, script, *statements[1], yugabyte, text=True, batch=False)
        raise AssertionError(f"{paged} rows read for {scope} in {len(statements)} pages; page 2:\n{plan_text}")
    if not yugabyte:
        # PostgreSQL reads a prefix batch by its LATERAL DISTINCT ON a prefix (unchanged)
        return
    prefix = _Recorder(conn)
    rows = await read_keyed(
        prefix, name, keys=KeyBatch(("state",), [["s000"]], ("text",)), columns=["state", "id", "v"], row_cap=1_001
    )
    assert sorted(int(row["id"]) for row in rows) == expected
    sent = [(sql, params) for sql, params in prefix.sent if "version()" not in sql]
    assert len(sent) > 1, sent
    batched = 0
    for sql, params in sent:
        assert "LIMIT" in sql and "ANY" not in sql, sql
        read = script.rows_read(
            await _explained(conn, script, sql, params, yugabyte, text=False, batch=False), yugabyte
        )
        # one statement's keys' versions, give or take the request read past its LIMIT: never the scope's
        assert read <= 1_000 * versions * 1.1 + past + 50, (read, sql)
        batched += read
    _log.warning("prefix batch of %d versions: %d rows read in %d statements", scope, batched, len(sent))
    assert batched <= scope * 1.02 + len(sent) * past + 50, batched

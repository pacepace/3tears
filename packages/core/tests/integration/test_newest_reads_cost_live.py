"""Integration: what a newest-per-key read costs by the versions each key holds (append-only L3, chunk 3 follow-up).

``scripts/measure-newest-reads.py`` measures the mechanisms at real size (2M rows, 1, 5 and 50 versions a key); this
runs its own statements at a small size on the session ``db_container`` (Postgres in ``integration``, YugabyteDB in
``integration_yugabyte``) and proves what the numbers rest on:

- every mechanism answers the same rows: each key's newest live version, never a gone key;
- the per-key seeks read a bounded number of entries a key whatever its versions (``skip`` two, ``lateral`` one),
  while the shipped guard's rows read grow with them;
- on YugabyteDB the batched ``DISTINCT ON`` key reads (``distinct``, ``distinct_join``) read each batch key's
  versions and no more: rows read grow with the versions, never with the table.

It asserts no speed budget: the mechanism is not chosen yet. The numbers are logged.
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

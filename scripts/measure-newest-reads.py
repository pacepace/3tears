#!/usr/bin/env python
"""Measure what a newest-per-key read of a versioned table costs, by the versions each key holds.

Append-only L3 (``design-append-only-l3.md``) keeps a version of a key a write; cleanup is never relied on, so after a
burst a key rewritten 50 times holds 50 versions. A read must answer only each key's newest live version. This
compares the mechanisms on one table, one database, one machine:

``guard``
    the shipped ``keys.newest_guard`` beside the read: every version the read's range or keys name is read, and each
    is dropped unless no newer version of its key exists (a seek of ``(key..., epoch DESC)``) and no newer run
    marker names it.
``distinct``
    (fills only) ``DISTINCT ON (key) ... ORDER BY key, epoch DESC`` over the page's range: every version read, no
    probe; the gone flag and run markers judged on the one row kept.
``skip``
    (fills only) a per-key newest-first seek, a recursive skip scan: the next key past the last (``ORDER BY id LIMIT
    1``), then its newest version (``ORDER BY epoch DESC LIMIT 1``), two seeks a key whatever its versions.
``lateral``
    (key batches only) the batch unnested, each key ``LATERAL (... WHERE key = k ORDER BY epoch DESC LIMIT 1)``.

Cases, weighted as Pace set them (2026-11-25: L3 reads are cache fills; the hot path runs on NATS deltas):

1. the headline: one scope's whole-state fill, ~20,000 keys, in keyset pages, at 1, 5 and 50 versions a key;
2. a whole-table fill, every scope paged, of the 5-version table (2M rows);
3. secondary: a keyed batch of 5,000 keys.

Each table holds 2,000,000 rows: 2M / V keys, 20,000 keys a scope (so 100, 20 and 2 scopes); 1% of keys' newest
version is a gone marker; the run-marker table holds markers of other scopes. For every case: ms (median of the timed
runs, a warm-up first), rows read from storage (EXPLAIN ANALYZE of each statement: on YugabyteDB the DIST output's
rows scanned, on Postgres each scan node's rows plus the rows its filter removed, loops included) and the plans.

Run one engine at a time, one version count at a time (each load is a few minutes on YugabyteDB)::

    uv run python scripts/measure-newest-reads.py --url "$URL" --versions 5 --cases state table batch --out DIR

The schema it builds (``nr_measure``) is dropped at the end. ``tests/integration/test_newest_reads_cost_live.py``
runs the same statements at a small size and checks every mechanism answers the same rows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import asyncpg

from threetears.core.keyed import Fragment, KeyBatch
from threetears.interfaces.keyed.keys import VersionedTable, newest_guard
from threetears.interfaces.keyed.rows import keyed_read_statement

SCHEMA = "nr_measure"
KEYS_A_SCOPE = 20_000
PAGE = 5_000
MARKERS = "threetears_run_markers"
COLUMNS = ("state", "id", "v", "payload")


def table_name(versions: int) -> str:
    return f"vt_{versions}"


def declaration(versions: int) -> VersionedTable:
    """the table as the shipped guard sees it: key (state, id), markers by the state."""
    return VersionedTable(
        f"{SCHEMA}.{table_name(versions)}", ("state", "id"), "epoch", "gone", MARKERS, 0, ("text", "bigint")
    )


async def on_yugabyte(conn: Any) -> bool:
    return "-YB-" in str(await conn.fetchval("SELECT version()"))


async def load(conn: Any, versions: int, rows: int, yugabyte: bool) -> int:
    """the table of ``rows`` rows, ``versions`` a key; answers its scopes."""
    keys = rows // versions
    scopes = max(1, keys // KEYS_A_SCOPE)
    table = f"{SCHEMA}.{table_name(versions)}"
    pk = "PRIMARY KEY ((state) HASH, id ASC, epoch DESC)" if yugabyte else "PRIMARY KEY (state, id, epoch)"
    await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    await conn.execute(f"DROP TABLE IF EXISTS {table}")
    await conn.execute(
        f"CREATE TABLE {table} (state TEXT NOT NULL, id BIGINT NOT NULL, epoch BIGINT NOT NULL, v INTEGER, "
        f"gone BOOLEAN NOT NULL DEFAULT false, payload TEXT, {pk})"
    )
    markers_pk = (
        "PRIMARY KEY ((table_name) HASH, scope ASC, prefix ASC, epoch ASC)"
        if yugabyte
        else "PRIMARY KEY (table_name, scope, prefix, epoch)"
    )
    await conn.execute(
        f"CREATE TABLE IF NOT EXISTS {SCHEMA}.{MARKERS} (table_name TEXT NOT NULL, scope TEXT NOT NULL, "
        f"prefix TEXT NOT NULL, epoch BIGINT NOT NULL, {markers_pk})"
    )
    per = keys // scopes
    for scope in range(scopes):
        for epoch in range(versions):
            # one statement a scope and version: 20,000 rows
            await conn.execute(
                f"INSERT INTO {table} (state, id, epoch, v, gone, payload) "
                f"SELECT $1, i, $2::bigint, (i * 7 + $2::bigint)::int % 1000, ($2::bigint = $3::bigint - 1 AND i % 100 = 0), "
                f"repeat('x', 40) FROM generate_series(0, $4::bigint - 1) AS i",
                f"s{scope:03d}",
                epoch,
                versions,
                per,
            )
    # markers of other scopes (a run of a scope no read here names), so the probe finds the table's markers
    for other in range(20):
        await conn.execute(
            f"INSERT INTO {SCHEMA}.{MARKERS} VALUES ($1, $2, $2, $3) ON CONFLICT DO NOTHING",
            f"{SCHEMA}.{table_name(versions)}",
            f"zz{other:02d}",
            10**12,
        )
    await conn.execute(f"ANALYZE {table}")
    await conn.execute(f"ANALYZE {SCHEMA}.{MARKERS}")
    return scopes


# ---------------------------------------------------------------- statements


def guard(versions: int, row: str) -> str:
    return newest_guard(declaration(versions), row, read_as=f"{SCHEMA}.{table_name(versions)}")


def marker_free(versions: int, row: str) -> str:
    """the run-marker half of the guard alone, on the one row a per-key read keeps."""
    full = guard(versions, row)
    return full[full.index("NOT EXISTS (SELECT 1 FROM " + f'"{SCHEMA}"."{MARKERS}"') : -1]


def marker_join(versions: int, row: str) -> str:
    """the run-marker half of the guard as a LEFT JOIN on the one row a per-key read keeps (``gone_run.table_name IS
    NULL`` when no newer marker names it). WHY not the NOT EXISTS in the select list: there it is a SubPlan, one
    storage request a row on YugabyteDB (19,821 requests for a 20,000-key fill, 31.8 s, measured 2026-11-25); joined,
    the table's markers are read once."""
    probe = marker_free(versions, row)
    head = "NOT EXISTS (SELECT 1 FROM "
    source, condition = probe[len(head) : -1].split(" WHERE ", 1)
    return f"LEFT JOIN {source} ON {condition}"


def fill_page_sql(mechanism: str, versions: int) -> str:
    """one page of a scope's fill past a cursor: $1 the scope, $2 the last id read (-1 first), $3 the page size.
    Every statement answers each key it passes once, with ``live`` saying whether a read keeps it, and the page's
    last id (the cursor) as ``id``."""
    table = f"{SCHEMA}.{table_name(versions)}"
    cols = ", ".join(f"t.{c}" for c in COLUMNS)
    if mechanism == "guard":
        return (
            f"SELECT {cols}, true AS live FROM {table} AS t WHERE t.state = $1 AND t.id > $2 "
            f"AND {guard(versions, 't')} ORDER BY t.id LIMIT $3"
        )
    if mechanism == "distinct":
        return (
            f"SELECT n.state, n.id, n.v, n.payload, (NOT n.gone AND gone_run.table_name IS NULL) AS live FROM "
            f"(SELECT DISTINCT ON (t.id) t.* FROM {table} AS t WHERE t.state = $1 AND t.id > $2 "
            f"ORDER BY t.id, t.epoch DESC LIMIT $3) AS n {marker_join(versions, 'n')} ORDER BY n.id"
        )
    if mechanism == "skip":
        # two seeks a key, each stopping at its first entry: the next key past the last (its first entry, any
        # version), then that key's newest (ORDER BY epoch DESC alone: a backward read of the key on Postgres, a
        # forward one of (key, epoch DESC) on YugabyteDB). One ORDER BY id, epoch DESC would read every version of the
        # key on Postgres, whose primary key orders the epoch ascending (a mixed order is sorted, not seeked)
        step = (
            "SELECT x.* FROM (SELECT t.id FROM {table} AS t WHERE t.state = $1 AND t.id > {after} ORDER BY t.id LIMIT 1) "
            "AS nk CROSS JOIN LATERAL (SELECT t.* FROM {table} AS t WHERE t.state = $1 AND t.id = nk.id "
            "ORDER BY t.epoch DESC LIMIT 1) AS x"
        )
        return (
            f"WITH RECURSIVE k AS (({step.format(table=table, after='$2')}) UNION ALL "
            f"SELECT y.* FROM k CROSS JOIN LATERAL ({step.format(table=table, after='k.id')}) AS y) "
            f"SELECT n.state, n.id, n.v, n.payload, (NOT n.gone AND gone_run.table_name IS NULL) AS live "
            f"FROM (SELECT * FROM k LIMIT $3) AS n {marker_join(versions, 'n')} ORDER BY n.id"
        )
    raise ValueError(mechanism)


def batch_sql(mechanism: str, versions: int, yugabyte: bool, keys: Sequence[tuple[str, int]]) -> tuple[str, list[Any]]:
    table = f"{SCHEMA}.{table_name(versions)}"
    states = [key[0] for key in keys]
    ids = [key[1] for key in keys]
    if mechanism == "guard":
        # exactly what read_keyed sends: the shipped statement builder, the guard beside it
        batch = KeyBatch(("state", "id"), [states, ids], ("text", "bigint"))
        statement = keyed_read_statement(table, COLUMNS, batch, {}, None, (Fragment(guard(versions, "t")),))
        sql, params = statement.render((states, ids), natively=True, yugabyte=yugabyte)
        return sql, list(params)
    if mechanism == "lateral":
        return (
            f"SELECT n.state, n.id, n.v, n.payload FROM unnest($1::text[], $2::bigint[]) AS k(state, id) "
            f"CROSS JOIN LATERAL (SELECT t.* FROM {table} AS t WHERE t.state = k.state AND t.id = k.id "
            f"ORDER BY t.epoch DESC LIMIT 1) AS n WHERE NOT n.gone AND {marker_free(versions, 'n')}",
            [states, ids],
        )
    raise ValueError(mechanism)


# ---------------------------------------------------------------- measuring


def read_requests(plan: Any, yugabyte: bool) -> int | None:
    """storage read requests (round trips to the tablets) of one EXPLAIN (ANALYZE, DIST): YugabyteDB's alone."""
    return int(plan[0].get("Storage Read Requests", 0)) if yugabyte else None


def rows_read(plan: Any, yugabyte: bool) -> int:
    """rows read from storage by one EXPLAIN (ANALYZE[, DIST], FORMAT JSON)."""
    top = plan[0]
    if yugabyte:
        # the statement's own total, every scan and probe of it (the DIST output's "Storage Rows Scanned")
        return int(top.get("Storage Rows Scanned", 0))
    total = 0

    def walk_pg(node: dict[str, Any]) -> None:
        nonlocal total
        if "Scan" in node.get("Node Type", "") and node.get("Relation Name", "").startswith("vt_"):
            loops = int(node.get("Actual Loops", 1) or 1)
            total += int(node.get("Actual Rows", 0)) * loops + int(node.get("Rows Removed by Filter", 0)) * loops
        for child in node.get("Plans", []):
            walk_pg(child)

    walk_pg(top["Plan"])
    return total


async def explain(conn: Any, sql: str, params: Sequence[Any], yugabyte: bool, text: bool = False) -> Any:
    options = "ANALYZE, DIST" if yugabyte else "ANALYZE, BUFFERS"
    if text:
        rows = await conn.fetch(f"EXPLAIN ({options}) {sql}", *params)
        return "\n".join(row[0] for row in rows)
    raw = await conn.fetchval(f"EXPLAIN ({options}, FORMAT JSON) {sql}", *params)
    return json.loads(raw) if isinstance(raw, str) else raw


async def fill_scope(conn: Any, sql: str, scope: str) -> tuple[int, list[int]]:
    """a scope's fill page by page: (live rows, each page's cursor)."""
    cursor, live, cursors = -1, 0, []
    while True:
        cursors.append(cursor)
        page = await conn.fetch(sql, scope, cursor, PAGE)
        live += sum(1 for row in page if row["live"])
        if len(page) < PAGE:
            return live, cursors
        cursor = int(page[-1]["id"])


async def measure_fill(
    conn: Any, mechanism: str, versions: int, scopes: Sequence[str], yugabyte: bool, runs: int, out: Path
) -> dict[str, Any]:
    sql = fill_page_sql(mechanism, versions)
    await fill_scope(conn, sql, scopes[0])  # warm-up
    times, live, cursors = [], 0, {}
    for _ in range(runs):
        started = time.perf_counter()
        live = 0
        for scope in scopes:
            answered, cursors[scope] = await fill_scope(conn, sql, scope)
            live += answered
        times.append((time.perf_counter() - started) * 1000)
    read, requests = 0, 0
    for scope in scopes:
        for cursor in cursors[scope]:
            plan = await explain(conn, sql, (scope, cursor, PAGE), yugabyte)
            read += rows_read(plan, yugabyte)
            requests += read_requests(plan, yugabyte) or 0
    tag = f"fill-{len(scopes)}scopes-v{versions}-{mechanism}"
    (out / f"{tag}.plan.txt").write_text(
        sql + "\n\n" + await explain(conn, sql, (scopes[0], -1, PAGE), yugabyte, text=True)
    )
    return {
        "case": tag,
        "ms": round(statistics.median(times), 1),
        "live_rows": live,
        "rows_read": read,
        "read_requests": requests if yugabyte else None,
        "pages": sum(len(c) for c in cursors.values()),
    }


async def measure_batch(
    conn: Any, mechanism: str, versions: int, scopes: int, yugabyte: bool, runs: int, out: Path, size: int
) -> dict[str, Any]:
    rng = random.Random(versions)
    keys = sorted({(f"s{rng.randrange(scopes):03d}", rng.randrange(KEYS_A_SCOPE)) for _ in range(size * 2)})[:size]
    rng.shuffle(keys)
    keys = keys[:size]
    sql, params = batch_sql(mechanism, versions, yugabyte, keys)
    await conn.fetch(sql, *params)
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        answered = await conn.fetch(sql, *params)
        times.append((time.perf_counter() - started) * 1000)
    plan_text = await explain(conn, sql, params, yugabyte, text=True)
    tag = f"batch{len(keys)}-v{versions}-{mechanism}"
    (out / f"{tag}.plan.txt").write_text(sql + "\n\n" + plan_text)
    return {
        "case": tag,
        "ms": round(statistics.median(times), 1),
        "live_rows": len(answered),
        "rows_read": rows_read(plan := await explain(conn, sql, params, yugabyte), yugabyte),
        "read_requests": read_requests(plan, yugabyte),
        "batched_nl": "Batched Nested Loop" in plan_text,
    }


async def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--versions", type=int, nargs="+", default=[1, 5, 50])
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--cases", nargs="+", default=["state", "batch"], choices=["state", "table", "batch"])
    parser.add_argument("--mechanisms", nargs="+", default=["guard", "distinct", "skip", "lateral"])
    parser.add_argument("--batch", type=int, default=5_000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep", action="store_true", help="keep the schema (and reuse a loaded table)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    conn = await asyncpg.connect(args.url)
    yugabyte = await on_yugabyte(conn)
    engine = "yugabyte" if yugabyte else "postgres"
    results: list[dict[str, Any]] = []
    try:
        for versions in args.versions:
            started = time.perf_counter()
            held = await conn.fetchval("SELECT to_regclass($1)", f"{SCHEMA}.{table_name(versions)}")
            if args.keep and held is not None:
                scopes = int(await conn.fetchval(f"SELECT count(DISTINCT state) FROM {SCHEMA}.{table_name(versions)}"))
            else:
                scopes = await load(conn, versions, args.rows, yugabyte)
            print(f"{engine} v{versions}: {scopes} scopes loaded in {time.perf_counter() - started:.0f}s", flush=True)
            for case in args.cases:
                names = [f"s{i:03d}" for i in range(scopes)] if case == "table" else ["s000"]
                for mechanism in args.mechanisms:
                    if case == "batch":
                        if mechanism not in ("guard", "lateral"):
                            continue
                        result = await measure_batch(
                            conn, mechanism, versions, scopes, yugabyte, args.runs, args.out, args.batch
                        )
                    else:
                        if mechanism == "lateral":
                            continue
                        runs = 1 if case == "table" else args.runs
                        result = await measure_fill(conn, mechanism, versions, names, yugabyte, runs, args.out)
                    result["engine"] = engine
                    results.append(result)
                    print(json.dumps(result), flush=True)
    finally:
        with (args.out / f"results-{engine}.jsonl").open("a") as sink:
            for result in results:
                sink.write(json.dumps(result) + "\n")
        if not args.keep:
            await conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))

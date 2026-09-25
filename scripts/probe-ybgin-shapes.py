#!/usr/bin/env python
"""Probe which GIN predicate shapes YugabyteDB's index refuses, and that ``gin_filter`` clears them.

``threetears.core.data.gin`` and ``tests/enforcement/test_gin_predicates_use_gin_filter.py``
encode a rule about YugabyteDB: ``ybgin`` serves a scan with exactly one required entry and
refuses any other with ``unsupported ybgin index scan``. Which shapes fall on which side was
measured, not read from documentation, and a claim about a database's planner can go stale
with the next YugabyteDB release. This re-measures it.

It builds a scratch schema with one table carrying a GIN index on a ``tsvector``, a ``jsonb``
and a ``text[]`` column, and a ``pg_trgm`` trigram GIN index on its text, then runs each shape with a ``pg_hint_plan`` hint forcing that GIN
index -- the plan a large table gets without being asked. Every shape is run as written and,
for the refused ones, again through ``gin_filter``. The wrapped form must succeed and return
the same rows as an unindexed scan. The schema is dropped at the end.

Exit status is 0 when every shape behaves as the guard assumes, 1 otherwise; each mismatch
names the shape, so a change in YugabyteDB shows up as a named line, not a silent pass.

Run against any YugabyteDB the caller can create a schema in::

    uv run python scripts/probe-ybgin-shapes.py --dsn postgresql://yugabyte:yugabyte@localhost:5433/yugabyte

On PostgreSQL every shape succeeds, so the refused rows report as mismatches: the probe is
meaningful only against YugabyteDB.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from dataclasses import dataclass

import asyncpg

from threetears.core.data.gin import gin_filter


@dataclass(frozen=True)
class _Shape:
    """one predicate shape and what the index is expected to do with it.

    :param label: what the output line calls it
    :ptype label: str
    :param index: GIN index the hint forces
    :ptype index: str
    :param predicate: SQL predicate over the probe table, with ``$1`` bound to ``arg``
    :ptype predicate: str
    :param arg: value bound to ``$1``
    :ptype arg: object
    :param refused: whether ybgin is expected to refuse it
    :ptype refused: bool
    """

    label: str
    index: str
    predicate: str
    arg: object
    refused: bool


_SHAPES = (
    _Shape("websearch OR", "probe_sv", "sv @@ websearch_to_tsquery('english', $1)", "build or publish", True),
    _Shape("websearch NOT", "probe_sv", "sv @@ websearch_to_tsquery('english', $1)", "build -draft", True),
    _Shape("to_tsquery OR", "probe_sv", "sv @@ to_tsquery('english', $1)", "build | publish", True),
    _Shape("pre-built tsquery OR", "probe_sv", "sv @@ $1::tsquery", "build | publish", True),
    _Shape("reversed websearch OR", "probe_sv", "websearch_to_tsquery('english', $1) @@ sv", "build or publish", True),
    _Shape("jsonb ?| two keys", "probe_tags", "tags ?| $1::text[]", ["a", "z"], True),
    _Shape("array && two elements", "probe_labels", "labels && $1", ["a", "z"], True),
    _Shape(
        "plainto five words",
        "probe_sv",
        "sv @@ plainto_tsquery('english', $1)",
        "build publish the audience now",
        False,
    ),
    _Shape("phraseto two words", "probe_sv", "sv @@ phraseto_tsquery('english', $1)", "build publish", False),
    _Shape("reversed plainto", "probe_sv", "plainto_tsquery('english', $1) @@ sv", "build publish", False),
    _Shape("jsonb ?& two keys", "probe_tags", "tags ?& $1::text[]", ["a", "b"], False),
    _Shape("jsonb ? one key", "probe_tags", "tags ? $1", "a", False),
    _Shape("jsonb @>", "probe_tags", "tags @> $1::jsonb", '["a"]', False),
    _Shape("array @> two elements", "probe_labels", "labels @> $1", ["a", "b"], False),
    _Shape("array <@", "probe_labels", "labels <@ $1", ["a", "b", "c"], False),
    _Shape("array && one element", "probe_labels", "labels && $1", ["a"], False),
    _Shape("trigram similarity %", "probe_trgm", "body % $1", "biuld and pubish the audiense", True),
    _Shape("trigram ILIKE", "probe_trgm", "body ILIKE $1", "%publish%", False),
)


async def _ids(conn: asyncpg.Connection, sql: str, arg: object) -> tuple[set[int] | None, str]:
    """run a probe query, returning its ids or the first line of the error.

    :param conn: connection with ``search_path`` on the scratch schema
    :ptype conn: asyncpg.Connection
    :param sql: query selecting ``id``
    :ptype sql: str
    :param arg: value bound to ``$1``
    :ptype arg: object
    :return: the ids, or ``None`` and the error text
    :rtype: tuple[set[int] | None, str]
    """
    try:
        rows = await conn.fetch(sql, arg)
    except asyncpg.PostgresError as exc:
        return None, str(exc).splitlines()[0]
    return {row["id"] for row in rows}, ""


async def _probe(dsn: str) -> list[str]:
    """run every shape and return one line per mismatch with the expected behaviour.

    :param dsn: database to create the scratch schema in
    :ptype dsn: str
    :return: mismatch descriptions, empty when every shape behaved as expected
    :rtype: list[str]
    """
    schema = f"ybgin_probe_{uuid.uuid4().hex[:8]}"
    mismatches: list[str] = []
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute(
            "CREATE TABLE probe (id int PRIMARY KEY, body text, tags jsonb, labels text[], "
            "sv tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED)"
        )
        await conn.execute("CREATE INDEX probe_sv ON probe USING gin (sv)")
        await conn.execute("CREATE INDEX probe_tags ON probe USING gin (tags)")
        await conn.execute("CREATE INDEX probe_labels ON probe USING gin (labels)")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        trgm_schema = await conn.fetchval(
            "SELECT n.nspname FROM pg_extension e JOIN pg_namespace n ON n.oid = e.extnamespace "
            "WHERE e.extname = 'pg_trgm'"
        )
        await conn.execute(f'SET search_path TO "{schema}", "{trgm_schema}"')
        await conn.execute(f'CREATE INDEX probe_trgm ON probe USING gin (body "{trgm_schema}".gin_trgm_ops)')
        await conn.execute(
            "INSERT INTO probe VALUES "
            """(1, 'build and publish the audience now', '["a","b"]', '{a,b}'), """
            """(2, 'draft the build notes', '["z"]', '{z}'), """
            """(3, 'nothing relevant here', '["q"]', '{q}')"""
        )
        for shape in _SHAPES:
            hint = f"/*+ IndexScan(probe {shape.index}) */ "
            raw, error = await _ids(conn, f"{hint}SELECT id FROM probe WHERE {shape.predicate}", shape.arg)
            if (raw is None) != shape.refused:
                expected = "refused" if shape.refused else "served"
                mismatches.append(f"{shape.label}: expected {expected}, got {error or 'served'}")
            if shape.refused:
                wrapped, error = await _ids(
                    conn, f"{hint}SELECT id FROM probe WHERE {gin_filter(shape.predicate)}", shape.arg
                )
                baseline, _ = await _ids(conn, f"SELECT id FROM probe WHERE ({shape.predicate}) IS TRUE", shape.arg)
                if wrapped is None or wrapped != baseline:
                    mismatches.append(f"{shape.label}: gin_filter form failed ({error or f'{wrapped} != {baseline}'})")
            status = "refused" if raw is None else "served"
            print(f"{status:8} {shape.label}")
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()
    return mismatches


def main() -> int:
    """parse arguments, run the probe, and report.

    :return: process exit status
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", required=True, help="YugabyteDB DSN with CREATE SCHEMA rights")
    args = parser.parse_args()
    mismatches = asyncio.run(_probe(args.dsn))
    for line in mismatches:
        print(f"MISMATCH {line}", file=sys.stderr)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())

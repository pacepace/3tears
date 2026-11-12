"""a changed-rows read given its held keys as an Arrow table (``read_changed(held=pa.Table)``) answers exactly what it
answers given them as a mapping: the same rows, gone keys, changed keys, moved groups and caps; on an in-process DuckDB.

The Arrow form is what L3 answers (``held_in_arrow``: the key columns and the row hash): reports compared it after
``to_pylist`` of the whole held table, a Python row of every held key. Every case below runs both forms over the same
scope and compares the whole answer: hex and base64 hashes, NULL hashes and a text that is no hash, keys held typed and
held as text (an integer's digits, a zoned time), naive and zoned times, NULL keys, grouped reads (moved, gone and
capped groups); and the held side's matching costs no more in the Arrow form than in the mapping at reports' size.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from threetears.core.keyed import (
    ROW_HASH_COLUMN,
    ChangedRows,
    Dialect,
    HashColumn,
    HashKind,
    InlineDuckDB,
    Sort,
    key_text,
    read_changed,
    row_hash_sql,
    run_inline,
)
from threetears.core.keyed.held import HeldArrow, HeldMapping

_HASHED = (HashColumn("name"), HashColumn("votes", HashKind.INTEGER))


def _hex(spelled: str) -> str:
    """a stored hash (unpadded base64, row_hash_sql's spelling) in its other spelling, 32 hex digits."""
    return base64.b64decode(spelled + "==").hex()


def _table(held: Mapping[tuple[Any, ...], str | None], names: Sequence[str], types: Sequence[pa.DataType]) -> pa.Table:
    """the held mapping as L3's Arrow answer: one column a key part, typed as given, and the row hash."""
    keys = list(held)
    columns = {name: pa.array([key[at] for key in keys], kind) for at, (name, kind) in enumerate(zip(names, types))}
    columns[ROW_HASH_COLUMN] = pa.array(list(held.values()), pa.string())
    return pa.table(columns)


def _both(run: Any, held: Mapping[tuple[Any, ...], str | None], table: pa.Table) -> ChangedRows:
    by_mapping: ChangedRows = run(held)
    by_arrow: ChangedRows = run(table)
    assert by_arrow == by_mapping, (by_arrow, by_mapping)
    return by_mapping


@pytest.fixture
def grouped() -> Iterator[duckdb.DuckDBPyConnection]:
    """PA's six counties of ten rows (c0 .. c5) plus four rows of no county, and OH's, outside the scope."""
    db = duckdb.connect()
    db.execute("CREATE TABLE g (state VARCHAR, county VARCHAR, id BIGINT, name VARCHAR, votes BIGINT)")
    db.execute(
        "INSERT INTO g SELECT s, CASE WHEN i >= 60 THEN NULL ELSE 'c' || (i // 10) END, i, 'n' || i, i * 7 "
        "FROM range(64) r(i), (VALUES ('PA'), ('OH')) v(s)"
    )
    yield db
    db.close()


def _grouped_held(db: duckdb.DuckDBPyConnection) -> dict[tuple[Any, ...], str | None]:
    sql = f"SELECT state, county, id, {row_hash_sql(_HASHED, Dialect.DUCKDB)} FROM g AS t WHERE state = 'PA'"
    return {(row[0], row[1], row[2]): row[3] for row in db.execute(sql).fetchall()}


def _grouped(db: duckdb.DuckDBPyConnection, **options: Any) -> Any:
    def run(held: Any) -> ChangedRows:
        return run_inline(
            read_changed(
                InlineDuckDB(db),
                "g",
                key=["state", "county", Sort("id", cast="bigint")],
                columns=["state", "county", "id", "name", "votes"],
                held=held,
                hash_columns=_HASHED,
                by={"state": "PA"},
                **options,
            )
        )

    return run


_GROUPED_TYPES = (pa.string(), pa.string(), pa.int64())


@pytest.mark.parametrize("grouping", [{}, {"group": "county"}, {"group": "county", "group_rows_at_most": 12}])
def test_the_arrow_form_answers_as_the_mapping_hex_base64_null_and_broken_hashes(
    grouped: duckdb.DuckDBPyConnection, grouping: dict[str, Any]
) -> None:
    """Flips if the Arrow form answers other rows, gone keys, moved groups or caps than the mapping: hashes held hex
    and base64, a NULL hash (read again; its group moved), a text that is no hash, changed, gone and new rows, a
    group that left whole and the NULL group."""
    held: dict[tuple[Any, ...], str | None] = {}
    for at, (key, hashed) in enumerate(_grouped_held(grouped).items()):
        held[key] = _hex(hashed) if at % 2 and hashed is not None else hashed
    held[("PA", "c1", 12)] = None
    held[("PA", "c2", 25)] = "no hash at all"
    held[("PA", "c3", 33)] = _hex(_grouped_held(grouped)[("PA", "c3", 33)] or "").upper()
    grouped.execute("UPDATE g SET name = 'moved' WHERE state = 'PA' AND id IN (3, 61)")
    grouped.execute("DELETE FROM g WHERE state = 'PA' AND (id = 44 OR county = 'c5')")
    grouped.execute("INSERT INTO g VALUES ('PA', 'c0', 900, 'new', 1)")
    changed = _both(_grouped(grouped, **grouping), held, _table(held, ("state", "county", "id"), _GROUPED_TYPES))
    if not grouping.get("group_rows_at_most"):
        assert {row["id"] for row in changed.rows} >= {3, 12, 25, 900}
        # a changed row under a NULL key part is read in full too (its NULL part pinned IS NULL), never only named
        assert ("PA", None, 61) in changed.changed_keys
        assert [row["name"] for row in changed.rows if row["id"] == 61] == ["moved"]
        assert {tuple(key) for key in changed.changed_keys} == {
            (row["state"], row["county"], row["id"]) for row in changed.rows
        }
        assert ("PA", "c3", 44) in changed.gone_keys or ("PA", "c4", 44) in changed.gone_keys
    else:
        assert changed.capped


def test_a_scope_nothing_moved_in_answers_nothing_either_way(grouped: duckdb.DuckDBPyConnection) -> None:
    """Flips if a held table matching the scope reads a row or names a gone key in either form."""
    held = _grouped_held(grouped)
    for grouping in ({}, {"group": "county"}):
        changed = _both(_grouped(grouped, **grouping), held, _table(held, ("state", "county", "id"), _GROUPED_TYPES))
        assert (changed.rows, changed.gone_keys) == ([], [])


@pytest.fixture
def typed() -> Iterator[duckdb.DuckDBPyConnection]:
    """keyed by an integer id and a time (no zone), held typed or as text."""
    db = duckdb.connect()
    db.execute("CREATE TABLE typed (state VARCHAR, cid INTEGER, stamp TIMESTAMP, name VARCHAR)")
    db.execute(
        "INSERT INTO typed SELECT 'PA', 7000 + i, TIMESTAMP '2026-11-03 22:30:00' + to_microseconds(i * 1000001), "
        "'n' || i FROM range(8) r(i)"
    )
    yield db
    db.close()


_TYPED_HASH = (HashColumn("cid", HashKind.INTEGER), HashColumn("stamp", HashKind.TIMESTAMP), HashColumn("name"))


def _typed(db: duckdb.DuckDBPyConnection) -> Any:
    def run(held: Any) -> ChangedRows:
        return run_inline(
            read_changed(
                InlineDuckDB(db),
                "typed",
                key=["state", "cid", "stamp"],
                columns=["state", "cid", "stamp", "name"],
                held=held,
                hash_columns=_TYPED_HASH,
                by={"state": "PA"},
            )
        )

    return run


def _typed_rows(db: duckdb.DuckDBPyConnection) -> list[tuple[str, int, datetime, str]]:
    sql = f"SELECT state, cid, stamp, {row_hash_sql(_TYPED_HASH, Dialect.DUCKDB)} FROM typed AS t"
    return list(db.execute(sql).fetchall())


def _moved(db: duckdb.DuckDBPyConnection) -> None:
    db.execute("UPDATE typed SET name = 'moved' WHERE cid = 7002")
    db.execute("DELETE FROM typed WHERE cid = 7004")


def test_integer_keys_and_times_held_as_text_answer_as_the_mapping(typed: duckdb.DuckDBPyConnection) -> None:
    """Flips if keys held as text (an integer's digits, a time zoned in UTC) match differently in the Arrow form."""
    held = {(s, str(c), f"{a.isoformat(sep=' ')}+00:00"): h for s, c, a, h in _typed_rows(typed)}
    _moved(typed)
    changed = _both(_typed(typed), held, _table(held, ("state", "cid", "stamp"), (pa.string(),) * 3))
    assert [row["cid"] for row in changed.rows] == [7002]
    assert len(changed.gone_keys) == 1


@pytest.mark.parametrize("zone", [None, "UTC", "America/New_York"])
def test_typed_keys_naive_and_zoned_times_answer_as_the_mapping(
    typed: duckdb.DuckDBPyConnection, zone: str | None
) -> None:
    """Flips if a typed integer key or a time held naive (taken as UTC) or zoned (moved to UTC) matches differently
    in the Arrow form than in the mapping."""
    rows = _typed_rows(typed)
    if zone is None:
        held = {(s, c, a): h for s, c, a, h in rows}
    else:
        shift = timezone(timedelta(hours=-5)) if zone == "America/New_York" else UTC
        held = {(s, c, a.replace(tzinfo=UTC).astimezone(shift)): h for s, c, a, h in rows}
    _moved(typed)
    types = (pa.string(), pa.int32(), pa.timestamp("us", tz=zone))
    changed = _both(_typed(typed), held, _table(held, ("state", "cid", "stamp"), types))
    assert [row["cid"] for row in changed.rows] == [7002]


def test_a_held_table_without_the_hash_column_is_refused(typed: duckdb.DuckDBPyConnection) -> None:
    """Flips if a held table missing a key column or the row hash is read as holding nothing."""
    with pytest.raises(ValueError, match=ROW_HASH_COLUMN):
        _typed(typed)(pa.table({"state": ["PA"], "cid": [7000], "stamp": ["2026-11-03 22:30:00"]}))


# --- the held side's cost at reports' size ---------------------------------------------------------------------------

#: reports' held rows for one state's geographies (S/o34t: ~340k), in 120 groups
_HELD_ROWS = 340_000


@pytest.mark.loop_timing
def test_the_arrow_form_matches_reports_held_size_no_slower_than_the_mapping() -> None:
    """Flips if matching ~340k held keys in the Arrow form (canonical texts in pyarrow.compute, hash bytes in numpy, a
    group's lanes summed in numpy) costs more than the mapping form's Python rows: reports' held table becomes those
    rows today (to_pylist), and the Arrow form must not trade that for slower matching. Both run every question a
    grouped read asks (group hashes, the groups outside, the moved groups' keys, the answered keys)."""
    groups = [f"g{n:03d}" for n in range(120)]
    ids = list(range(_HELD_ROWS))
    counties = [groups[n % len(groups)] for n in ids]
    hashes = [f"{(n * 2654435761) % 2**128:032x}" for n in ids]
    kinds = [HashColumn("county"), HashColumn("id", HashKind.INTEGER)]
    table = pa.table({"county": counties, "id": pa.array(ids, pa.int64()), ROW_HASH_COLUMN: hashes})
    mapping = dict(zip(zip(counties, ids, strict=True), hashes, strict=True))
    answered = [((county, str(n)), hashed) for county, n, hashed in zip(counties[::50], ids[::50], hashes[::50])]
    moved = set(groups[:30])

    def canonical(values: Sequence[Any]) -> tuple[str | None, ...]:
        return tuple(key_text(value, kind.kind) for value, kind in zip(values, kinds, strict=True))

    def ask(held: Any) -> None:
        held.group_hashes(0)
        held.outside(0, set(groups[:-1]))
        within = held.within(0, moved)
        within.compare([texts for texts, _ in answered], [hashed for _, hashed in answered])

    def best(build: Any) -> float:
        times = []
        for _ in range(3):
            start = time.perf_counter()
            ask(build())
            times.append(time.perf_counter() - start)
        return min(times)

    lanes, modulus = (1, 9, 17, 25), 2**32
    by_mapping = best(lambda: HeldMapping.of(mapping, canonical, lanes, modulus))
    by_arrow = best(lambda: HeldArrow(table, ["county", "id"], kinds, lanes, modulus, key_text))
    assert by_arrow <= by_mapping, f"{_HELD_ROWS:,} held rows: Arrow {by_arrow:.3f} s, mapping {by_mapping:.3f} s"

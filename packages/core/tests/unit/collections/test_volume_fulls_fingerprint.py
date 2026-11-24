"""a full part's fingerprint: one vectorised definition the fold's part write and a copy's drift check share.

The contract this pins (:func:`threetears.core.collections.volume_fulls.rows_fingerprint`, ``design-pod-delta-files.md``
decision 32, check A):

- the same rows fingerprint equal whether held as the part holds them (the L3 rail's Arrow types) or as a copy's DuckDB
  holds them (a ``NUMERIC`` as a decimal, an instant as a zoned timestamp, an integer as ``INTEGER``), in any order,
  with NULLs, and an instant held as ISO text, zoned or naive (the rail's UTC), equal to the same instant held as a
  timestamp;
- one changed, extra, missing or duplicated row moves it, and so does a NULL where a value was (and the reverse);
- a fingerprint of another format (the per-row BLAKE2b one) is not comparable, so a copy reads it as
  ``no_fingerprint``, never as drift;
- each row's hash (:func:`row_digests`) is computed in the rows' order and is the same under both type shapes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

from threetears.core.collections.volume_fulls import (  # noqa: E402
    FINGERPRINT_FORMAT,
    fingerprint_comparable,
    fingerprint_kinds,
    row_digests,
    rows_fingerprint,
)

#: the rail's types of the columns (:func:`~threetears.core.collections.schema_backed.rail_arrow_types`): text, a
#: NUMERIC as text, a timestamp column as text, an integer, a boolean and a ``date_`` instant
_RAIL = {
    "name": pa.string(),
    "pct": pa.string(),
    "stamped": pa.string(),
    "votes": pa.int64(),
    "winner": pa.bool_(),
    "date_updated": pa.timestamp("us", tz="UTC"),
}
_KINDS = fingerprint_kinds(_RAIL)
_AT = datetime(2026, 11, 9, 10, 22, 10, 82658, tzinfo=UTC)
_WHOLE_SECOND = datetime(2026, 11, 9, 10, 22, 10, tzinfo=UTC)
#: (name, pct, stamped, votes, winner, date_updated): a NULL in every column somewhere
_ROWS: list[tuple[Any, ...]] = [
    ("Ada", Decimal("1.50"), _AT, 10, True, _AT),
    ("Bea", Decimal("-0.05"), _WHOLE_SECOND, 0, False, _WHOLE_SECOND),
    (None, None, None, None, None, None),
    ("", Decimal("100.00"), _AT, -7, True, None),
    ("Cy", Decimal("2.00"), None, 2**40, None, _AT),
]


def _stored(value: datetime) -> str:
    """an instant as the rail's text of a timestamp column spells it (``stored_instant``)."""
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _part(rows: list[tuple[Any, ...]]) -> Any:
    """the rows as a full's part holds them: the rail's types."""
    names = list(_RAIL)
    columns: dict[str, list[Any]] = {name: [row[at] for row in rows] for at, name in enumerate(names)}
    columns["pct"] = [None if value is None else format(value, "f") for value in columns["pct"]]
    columns["stamped"] = [None if value is None else _stored(value) for value in columns["stamped"]]
    return pa.table({name: pa.array(columns[name], _RAIL[name]) for name in names})


def _copy(rows: list[tuple[Any, ...]]) -> Any:
    """the rows as a copy's DuckDB holds them, read back as Arrow: a DECIMAL, zoned timestamps (answered in the
    session's zone, not UTC), an INTEGER, the rows in reverse order."""
    with duckdb.connect() as con:
        con.execute("SET TimeZone = 'America/Chicago'")
        con.execute(
            "CREATE TABLE t (name VARCHAR, pct DECIMAL(10, 2), stamped TIMESTAMPTZ, votes BIGINT, winner BOOLEAN, "
            "date_updated TIMESTAMPTZ)"
        )
        con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?)", [list(row) for row in reversed(rows)])
        return con.sql("SELECT * FROM t").to_arrow_table()


def test_equal_rows_fingerprint_equal_under_the_parts_and_the_copys_types_in_any_order() -> None:
    """Flips if the copy's DuckDB shape of the same rows (a NUMERIC as a decimal, instants as zoned timestamps in
    another session zone, the rows reversed) fingerprints otherwise than the part's rail shape: every copy would read
    as drifted. Also flips if the fingerprint is not of the current format."""
    part, copy = _part(_ROWS), _copy(_ROWS)
    assert pa.types.is_decimal(copy.schema.field("pct").type)
    assert pa.types.is_timestamp(copy.schema.field("stamped").type)
    fingerprint = rows_fingerprint(part, _KINDS)
    assert fingerprint == rows_fingerprint(copy, _KINDS)
    assert fingerprint.startswith(f"{FINGERPRINT_FORMAT}:")
    assert fingerprint_comparable(fingerprint)
    # a part fingerprints by its own types the same as by the rail's kinds named
    assert rows_fingerprint(part) == fingerprint


def test_an_instant_held_as_iso_text_zoned_or_naive_fingerprints_as_the_same_timestamp() -> None:
    """Flips if an instant column held as ISO text (with its offset, in another zone, or naive: the rail's UTC) or as
    a naive or nanosecond timestamp hashes otherwise than the zoned microsecond timestamp of the same instant."""
    kinds = fingerprint_kinds({"date_updated": pa.timestamp("us", tz="UTC")})
    values = [_AT, _WHOLE_SECOND, None]
    held = pa.table({"date_updated": pa.array(values, pa.timestamp("us", tz="UTC"))})
    texts = ["2026-11-09T05:22:10.082658-05:00", "2026-11-09T10:22:10", None]
    as_text = pa.table({"date_updated": pa.array(texts, pa.string())})
    naive = pa.table(
        {"date_updated": pa.array([None if v is None else v.replace(tzinfo=None) for v in values], pa.timestamp("us"))}
    )
    nanos = pa.table({"date_updated": pa.array(values, pa.timestamp("ns", tz="UTC"))})
    expected = rows_fingerprint(held, kinds)
    assert rows_fingerprint(as_text, kinds) == expected
    assert rows_fingerprint(naive, kinds) == expected
    assert rows_fingerprint(nanos, kinds) == expected
    assert row_digests(as_text, kinds).to_pylist() == row_digests(held, kinds).to_pylist()


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param(lambda rows: [(*rows[0][:3], 11, *rows[0][4:]), *rows[1:]], id="a value changed"),
        pytest.param(lambda rows: [(None, *rows[0][1:]), *rows[1:]], id="a value made NULL"),
        pytest.param(lambda rows: [*rows[:2], ("", *rows[2][1:]), *rows[3:]], id="a NULL made empty text"),
        pytest.param(lambda rows: [*rows, ("Dee", Decimal("3.00"), _AT, 4, False, _AT)], id="a row extra"),
        pytest.param(lambda rows: rows[1:], id="a row missing"),
        pytest.param(lambda rows: [*rows, rows[0]], id="a row held twice"),
        pytest.param(
            lambda rows: [("Ada", *rows[1][1:]), ("Bea", *rows[0][1:]), *rows[2:]], id="two rows' names swapped"
        ),
    ],
)
def test_one_changed_extra_missing_or_duplicated_row_moves_the_fingerprint(changed: Any) -> None:
    """Flips if a copy holding one row other than the part (a value changed, a NULL for a value or empty text for a
    NULL, a row extra, missing or held twice, two rows' values swapped between them) fingerprints equal to it: the
    drift check would miss the drift."""
    assert rows_fingerprint(_copy(changed(_ROWS)), _KINDS) != rows_fingerprint(_part(_ROWS), _KINDS)


def test_a_row_held_twice_never_cancels_as_an_xor_would() -> None:
    """Flips if the rows' hashes combine by XOR (or any rule a pair of equal rows cancels in): {a, a, b} and
    {b, c, c} have one count and one XOR, and must still differ."""
    a, b, c = _ROWS[0], _ROWS[1], _ROWS[3]
    assert rows_fingerprint(_part([a, a, b]), _KINDS) != rows_fingerprint(_part([b, c, c]), _KINDS)


def test_each_rows_digest_is_in_the_rows_order_and_equal_under_both_shapes() -> None:
    """Flips if a row's hash (the writer's sample compares rows by it) depends on the type shape it is held in, or
    if the hashes come back in another order than the rows (the sample pairs each with its key by position)."""
    part = row_digests(_part(_ROWS), _KINDS).to_pylist()
    copy = row_digests(_copy(_ROWS), _KINDS).to_pylist()
    assert copy == list(reversed(part))
    assert len(set(part)) == len(_ROWS)


def test_a_fingerprint_of_another_format_is_not_comparable() -> None:
    """Flips if a part an older writer fingerprinted (``<columns>-<rows>-<sum>``, the per-row BLAKE2b format) or none
    reads as comparable: the copy would compare its own new-format fingerprint with it and report drift."""
    assert not fingerprint_comparable("b2942aa7-337000-136b4254e1a789f18e6ae91cdc46e3dc")
    assert not fingerprint_comparable(None)
    assert fingerprint_comparable(rows_fingerprint(_part(_ROWS), _KINDS))

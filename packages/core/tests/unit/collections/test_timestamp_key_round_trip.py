"""A timestamptz key's every spelling reaches a copy as one text, so the copy finds its keys in a delta file.

The bug (2026-11-24): a delta file holds a ``timestamptz`` key as the L3 rail answers it (``2026-11-03T21:45:00Z``),
while the epoch record (``key_text``) and the change note trigger (``to_char(... 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"')``)
name it ``2026-11-03T21:45:00.000000+00:00``. A copy of a table keyed by a timestamp (reports'
``result_snapshots_by_id``) matched the keys it heard against the file's rows by text, found none, and waited or fell
back. The contract: every path a key's text comes by (a record, a note, Postgres' own text, the file, the copy's own
rail-text rows) names one instant in ``key_text(..., instant=True)``'s one spelling.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import pytest

pa = pytest.importorskip("pyarrow")

from threetears.core.collections.copy_follower import FollowedTable  # noqa: E402
from threetears.core.collections.delta_files import (  # noqa: E402
    HeldDeltaRows,
    TableDelta,
    key_text_table,
    read_delta_file_keys,
    read_delta_table,
    text_key_batch,
    write_delta_file,
)
from threetears.core.collections.epoch_change import key_text  # noqa: E402
from threetears.core.collections.schema_backed import (  # noqa: E402
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    rail_arrow_types,
)

_AT = datetime(2026, 11, 3, 21, 45, tzinfo=UTC)
#: the record's and the note trigger's spelling
_CANONICAL = "2026-11-03T21:45:00.000000+00:00"
#: the other spellings one instant comes in: the rail's (a delta file's, a copy's full load), Postgres' ``::text``, a
#: zone's, and a naive time a timestamptz column holds as UTC
_SPELLINGS = (
    "2026-11-03T21:45:00Z",
    "2026-11-03 21:45:00+00",
    "2026-11-03 16:45:00-05",
    "2026-11-03 21:45:00",
    _CANONICAL,
)
_COLUMNS = (Column("office", STRING_TYPE), Column("stamp", DATETIMETZ_TYPE), Column("votes", STRING_TYPE))
_KEY = ("office", "stamp")


def _heard(table: FollowedTable, keys: Any) -> Any:
    """keys a copy hears (positional text columns, as the registry's ``on_keys`` tells them), as its batch holds them:
    named as the key, through :func:`key_text_table` with the table's instant keys."""
    named = pa.table({column: keys.column(at) for at, column in enumerate(table.key)})
    return key_text_table(named, table.key, table.instant_keys)


def _file(tmp_path: Path, stamp: object) -> Path:
    """a delta file of one row keyed by ``stamp``, saved as a commit saves it."""
    delta = TableDelta(name="snapshots", types=rail_arrow_types(_COLUMNS), key=_KEY)
    delta.rows([{"office": "P", "stamp": stamp, "votes": "10"}], ["office", "stamp", "votes"])
    return tmp_path / write_delta_file(tmp_path, delta, "w1")


def test_every_spelling_of_an_instant_is_one_key_text() -> None:
    assert key_text(_AT) == _CANONICAL
    for spelling in _SPELLINGS:
        assert key_text(spelling, instant=True) == _CANONICAL, spelling
    # a text column keeps its text, an instant's look-alike included: its note names it verbatim
    assert key_text("2026-11-03T21:45:00Z") == "2026-11-03T21:45:00Z"
    assert key_text("not an instant", instant=True) == "not an instant"


@pytest.mark.parametrize("stamp", [_AT, "2026-11-03T21:45:00Z", "2026-11-03 21:45:00+00"])
def test_a_copy_finds_a_heard_timestamp_key_in_the_delta_file(tmp_path: Path, stamp: object) -> None:
    path = _file(tmp_path, stamp)
    table = FollowedTable(name="snapshots", key=_KEY, columns=_COLUMNS, delta_files=True)
    assert table.instant_keys == frozenset({"stamp"})

    # the file's own keys, as the registry tells a copy them (positional), heard in the record's spelling
    reach = read_delta_file_keys(path, table="snapshots")
    positional = reach.keys.rename_columns([f"k{at}" for at in range(reach.keys.num_columns)])
    heard_from_file = _heard(table, positional)
    assert heard_from_file.column("stamp").to_pylist() == [_CANONICAL]

    # the record's keys and the notes' keys, as the registry tells a copy them
    for ids in (["P", key_text(_AT)], ["P", _CANONICAL], ["P", "2026-11-03 21:45:00+00"]):
        heard = _heard(table, text_key_batch([ids]))
        assert heard.column("stamp").to_pylist() == [_CANONICAL]
        keys = heard.select(list(_KEY))
        # the copy looks the heard key up among the file's rows: found, so its row is merged from the file
        rows = HeldDeltaRows(_KEY, keys, rail_arrow_types(_COLUMNS), instants=table.instant_keys)
        delta = read_delta_table(path, table="snapshots")
        rows.apply(delta, np.array([0], dtype=np.int64))
        assert rows.applied == 1, ids

    # the copy's own rows hold the rail's text: matched by the same text
    held = pa.table({"office": ["P"], "stamp": ["2026-11-03T21:45:00Z"]})
    assert key_text_table(held, _KEY, table.instant_keys).column("stamp").to_pylist() == [_CANONICAL]

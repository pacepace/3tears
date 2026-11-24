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

from threetears.core.collections import delta_fold  # noqa: E402
from threetears.core.collections.copy_follower import FollowedTable  # noqa: E402
from threetears.core.collections.delta_files import (  # noqa: E402
    HeldDeltaRows,
    TableDelta,
    fold_delta_tables,
    joined_key_texts,
    key_text_table,
    read_delta_file_keys,
    read_delta_table,
    text_key_batch,
    write_delta_file,
)
from threetears.core.collections.delta_fold import FoldedTable  # noqa: E402
from threetears.core.collections.epoch_change import key_text  # noqa: E402
from threetears.core.collections.schema_backed import (  # noqa: E402
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
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


# --- the writer's fold and its recovery (2026-11-24 replay) -----------------------------------------------------------
#
# The bug: the writer's recovery of ``result_snapshots_by_id`` (keyed partly by a timestamptz) failed every round
# (``ArrowTypeError: Expected bytes, got a 'datetime.datetime'`` in ``_typed_texts``): a noted key's text was made a
# datetime for a column the rail answers as text. Past that, the recovery and the fold matched keys as given, so the
# file's ``...Z``, the note's ``....000000+00:00`` and Postgres' ``... +00`` of one instant were three keys.

_SNAPSHOT_COLUMNS = (
    Column("race_id", STRING_TYPE),
    Column("taken_at", DATETIMETZ_TYPE),
    Column("payload", STRING_TYPE, nullable=True),
)
_SNAPSHOT_KEY = ("race_id", "taken_at")
_T1 = datetime(2026, 11, 3, 21, 45, tzinfo=UTC)
_T2 = datetime(2026, 11, 3, 22, 0, 30, 125000, tzinfo=UTC)
_T3 = datetime(2026, 11, 4, 1, 15, tzinfo=UTC)


def _snapshots_table() -> FoldedTable:
    schema = TableSchema(
        name="result_snapshots_by_id",
        primary_key=_SNAPSHOT_KEY,
        columns=list(_SNAPSHOT_COLUMNS),
        delta_files=True,
    )
    return FoldedTable.of(schema)


def _snapshot_file(root: Path, write_id: str, rows: list[dict[str, Any]]) -> str:
    delta = TableDelta(name="result_snapshots_by_id", types=rail_arrow_types(_SNAPSHOT_COLUMNS), key=_SNAPSHOT_KEY)
    delta.rows(rows, ["payload"])
    return write_delta_file(root, delta, write_id)


def _ipc_file(path: Path, rows: Any) -> None:
    """an Arrow IPC file of ``rows``: a recovery's own work file (its notes' keys, L3's pages) as the writer writes it."""
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, rows.schema) as writer:
        writer.write_table(rows)


def _canonical_keys(keys: list[tuple[str, datetime]]) -> list[str]:
    """each key's one text, its instant in ``key_text(..., instant=True)``'s spelling."""
    named = pa.table(
        {
            "race_id": [race for race, _ in keys],
            "taken_at": [key_text(at, instant=True) for _, at in keys],
        }
    )
    texts: list[str] = joined_key_texts(named, _SNAPSHOT_KEY).to_pylist()
    return texts


def test_a_recovery_of_a_timestamp_keyed_table_reaches_and_recovers_its_leftover_files(tmp_path: Path) -> None:
    """Flips if the recovery's reach of a table keyed by a timestamptz (shaped as ``result_snapshots_by_id``) fails on
    the notes' key text (the 2026-11-24 ArrowTypeError), or names one instant's spellings as several keys, or matches
    L3's rows (Postgres' ``... +00``) against the file's ``...Z`` and the note's ``...+00:00`` as different keys:
    every key's text is ``key_text``'s, each key once, and only the key L3 no longer holds gets a delete marker."""
    table = _snapshots_table()
    files = (
        _snapshot_file(tmp_path, "w1", [{"race_id": "P", "taken_at": _T1, "payload": "a"}]),
        _snapshot_file(
            tmp_path,
            "w2",
            [
                {"race_id": "P", "taken_at": "2026-11-03T21:45:00Z", "payload": "b"},
                {"race_id": "S", "taken_at": _T2, "payload": "c"},
            ],
        ),
    )
    # the lost commit's notes: one key the files name too (in the note's spelling), one only the notes name
    _ipc_file(
        tmp_path / "work.noted0.arrow", pa.table({"race_id": ["P", "H"], "taken_at": [key_text(_T1), key_text(_T3)]})
    )

    prefixes, keyed = delta_fold.recovery_reach_job(
        str(tmp_path),
        table.plain(),
        table.kinds,
        None,
        files,
        ("work.noted0.arrow",),
        "work.reach.arrow",
        100,
    )
    with pa.memory_map(str(tmp_path / "work.reach.arrow")) as source:
        reach = pa.ipc.open_file(source).read_all()
    assert prefixes == () and keyed == 3
    assert reach.column("_3tears_text").to_pylist() == _canonical_keys([("P", _T1), ("S", _T2), ("H", _T3)]), (
        "each key once, in key_text's one spelling, whichever spelling the file or the note held"
    )
    assert reach.column("_3tears_noted").to_pylist() == [True, False, True]

    # L3 answers (H, T3) no longer; its rows' instants in Postgres' own text
    _ipc_file(
        tmp_path / "work.pages.arrow",
        pa.table(
            {
                "race_id": ["P", "S"],
                "taken_at": ["2026-11-03 21:45:00+00", "2026-11-03 22:00:30.125+00"],
                "payload": ["b", "c"],
            }
        ),
    )
    file, rows, markers = delta_fold.recovery_delta_job(
        str(tmp_path),
        table.plain(),
        table.kinds,
        None,
        "work.reach.arrow",
        ("work.pages.arrow",),
        (),
        (),
        False,
        7,
    )
    assert (rows, markers) == (2, 1), "L3's two rows and one delete marker: the key L3 no longer holds"
    recovered = read_delta_file_keys(tmp_path / file, table="result_snapshots_by_id")
    texts = joined_key_texts(key_text_table(recovered.keys, _SNAPSHOT_KEY, table.instant_keys), _SNAPSHOT_KEY)
    assert sorted(texts.to_pylist()) == sorted(_canonical_keys([("P", _T1), ("S", _T2), ("H", _T3)]))
    assert table.instant_keys == frozenset({"taken_at"})


def test_the_fold_merges_one_instant_s_spellings_as_one_key(tmp_path: Path) -> None:
    """Flips if the fold matches a part's key (Postgres' ``... +00``) and a delta file's (the rail's ``...Z``) of one
    instant as two keys: the part kept the old row beside the file's new one."""
    types = rail_arrow_types(_SNAPSHOT_COLUMNS)
    held = pa.table({"race_id": ["P"], "taken_at": ["2026-11-03 21:45:00+00"], "payload": ["old"]})
    file = _snapshot_file(tmp_path, "w1", [{"race_id": "P", "taken_at": _T1, "payload": "new"}])
    changes = read_delta_table(tmp_path / file, table="result_snapshots_by_id")
    folded = fold_delta_tables(_SNAPSHOT_KEY, held, [changes], types, instants=frozenset({"taken_at"}))
    assert folded.column("payload").to_pylist() == ["new"]

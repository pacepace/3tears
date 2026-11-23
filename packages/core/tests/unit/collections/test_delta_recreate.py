"""a missing or unreadable delta file: copies wait and read nothing from L3; the writer recreates it once, by key.

Election night failed with every pod copy reading L3; a lost delta file used to send every copy back to L3 for its
change. Decision 29 (``design-pod-delta-files.md``, amending decision 3) makes the copy hold that table and ask the
writer, which reads that one change's keys from L3 and writes the file again under its name. The registry, the
follower, the delta files, the writer's recreation and the owner-routed forward are the real ones; L3 is a DuckDB that
counts the statements each side sends, the copy's and the writer's apart; NATS is an in-memory bus carrying a request
to one subscriber and its reply back. The contract this pins:

- a copy finding a change's file missing, or there and unreadable (truncated, garbage, another format's, a damaged
  zstd buffer, another table's), sends L3 no statement, holds that table, asks the writer once, and once the writer has
  recreated the file ends equal to L3; another table keeps being applied meanwhile;
- the writer reads only that change's keys from L3, by key, counted, a key L3 no longer holds written as a delete
  marker, and recreates the file once however many copies ask at once;
- a load's or a clear's file (its key prefixes) is recreated with the rows L3 holds under them, and applied so;
- a commit too big to list its rows has its keys only in its file: the writer hands the table to decision 28's recovery
  and recreates nothing; the copy reads neither L3 nor the change notes;
- only the replica serving under the write lease answers; with none, the copy hears ``no_writer``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import BigInteger, Column as SaColumn, DateTime, MetaData, String, Table

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")
pa_ipc = pytest.importorskip("pyarrow.ipc")

from nats.errors import NoRespondersError  # noqa: E402
from prometheus_client import REGISTRY  # noqa: E402
from threetears.nats import DEFAULT_REQUEST_TIMEOUT  # noqa: E402
from threetears.nats.errors import RequestError  # noqa: E402
from threetears.nats.subjects import Subject  # noqa: E402
from threetears.nats.transport import IncomingMessage  # noqa: E402

import threetears.core.collections.copy_follower as copy_follower  # noqa: E402
import threetears.core.collections.delta_recreate as delta_recreate  # noqa: E402
from threetears.core.cache.duckdb import DuckDBBackend  # noqa: E402
from threetears.core.cache.duckdb_binds import run_bound  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
from threetears.core.collections.change_notes import NotedKeysPage  # noqa: E402
from threetears.core.collections.copy_follower import CopyFollower, FollowedTable, PartialStamp  # noqa: E402
from threetears.core.collections.delta_files import (  # noqa: E402
    _CHECKSUM_MARK,
    _CHECKSUM_TRAILER,
    DeltaFileUnreadableError,
    TableDelta,
    read_delta_file,
    write_delta_file,
)
from threetears.core.collections.delta_recreate import (  # noqa: E402
    DeltaRecreateRequester,
    DeltaRecreator,
    serving_recreates,
)
from threetears.core.collections.epoch_change import ChangedRow, EpochChange, EpochWalk, walk_by_count  # noqa: E402
from threetears.core.collections.schema_backed import (  # noqa: E402
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    rail_arrow_types,
)
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE  # noqa: E402

_RESULTS, _TALLIES = "results", "tallies"
_KEY = ("state", "geo_id")
_COLUMNS = (
    Column(name="state", column_type=STRING_TYPE),
    Column(name="geo_id", column_type=BIGINT_TYPE),
    Column(name="votes", column_type=BIGINT_TYPE),
    Column(name="note", column_type=STRING_TYPE),
    Column(name="date_created", column_type=DATETIMETZ_TYPE),
)
_NAMES = tuple(column.name for column in _COLUMNS)
_TYPES = rail_arrow_types(_COLUMNS)
_BORN = datetime(2026, 11, 3, 21, 45, tzinfo=UTC)
_KEY_OF = "reports-writer"

_OPENED: list[Callable[[], None]] = []


@pytest.fixture(autouse=True)
def _close_what_the_test_opened() -> Any:
    yield
    while _OPENED:
        _OPENED.pop()()


@pytest.fixture(autouse=True)
def _volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """the shared volume, and a copy that looks for a waited-for file often enough for a test."""
    monkeypatch.setenv("THREETEARS_DELTA_FILES_DIR", str(tmp_path))
    monkeypatch.setattr(copy_follower, "FILE_POLL_SECONDS", 0.01)
    return tmp_path


def _metadata() -> MetaData:
    metadata = MetaData()
    for name in (_RESULTS, _TALLIES):
        Table(
            name,
            metadata,
            SaColumn("state", String(8), primary_key=True),
            SaColumn("geo_id", BigInteger, primary_key=True),
            SaColumn("votes", BigInteger),
            SaColumn("note", String(32)),
            SaColumn("date_created", DateTime(timezone=True)),
            info={L1_EVICTION_INFO: TEST_L1_TABLE},
        )
    return metadata


def _dict_rows(result: Any) -> list[dict[str, Any]]:
    names = [d[0] for d in result.description]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


class _Db:
    """L3's rows: one DuckDB holding both tables, as the writes left them."""

    def __init__(self) -> None:
        self.db = duckdb.connect(":memory:")
        _OPENED.append(self.db.close)
        self.db.execute("SET TimeZone = 'UTC'")
        for name in (_RESULTS, _TALLIES):
            self.db.execute(
                f"CREATE TABLE {name} (state VARCHAR, geo_id BIGINT, votes BIGINT, note VARCHAR, "
                "date_created TIMESTAMPTZ, PRIMARY KEY (state, geo_id))"
            )

    def insert(self, table: str, rows: Sequence[dict[str, Any]]) -> None:
        for row in rows:
            self.db.execute(
                f"INSERT OR REPLACE INTO {table} VALUES (?, ?, ?, ?, ?)",  # noqa: S608
                [row[name] for name in _NAMES],
            )

    def delete(self, table: str, key: tuple[str, int]) -> None:
        self.db.execute(f"DELETE FROM {table} WHERE state = ? AND geo_id = ?", list(key))  # noqa: S608

    def delete_state(self, table: str, state: str) -> None:
        self.db.execute(f"DELETE FROM {table} WHERE state = ?", [state])  # noqa: S608

    def held(self, table: str) -> list[tuple[Any, ...]]:
        return sorted(self.db.execute(f"SELECT {', '.join(_NAMES)} FROM {table}").fetchall())  # noqa: S608


class _L3:
    """one side's L3 (the copy's or the writer's): answers ``fetch`` from the rows, counting each statement it sends."""

    def __init__(self, db: _Db) -> None:
        self.db = db
        self.statements = 0

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(0)
        if not sql.startswith("SELECT version()"):
            self.statements += 1
        return run_bound(self.db.db.cursor(), sql, params, _dict_rows)


class _Log:
    """each table's write generation and its change records, each naming its commit's delta file."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {_RESULTS: 0, _TALLIES: 0}
        self.records: dict[tuple[str, str], EpochChange] = {}

    @staticmethod
    def _token(table: str, count: int) -> str:
        return f"{table}-inc:{count}"

    async def current(self, table: str) -> str:
        return self._token(table, self.counts[table])

    def commit(self, table: str, keys: Sequence[tuple[str, int]], delta_file: str | None) -> str:
        return self._record(table, rows=[ChangedRow(ids=[s, str(g)]) for s, g in keys], delta_file=delta_file)

    def too_large(self, table: str, delta_file: str) -> str:
        return self._record(table, keys_unknown=True, noted_between=(_BORN, _BORN), delta_file=delta_file, rows=[])

    def cleared(self, table: str, prefixes: Sequence[Mapping[str, str]], delta_file: str) -> str:
        return self._record(
            table, whole_table=True, rows=[], prefixes=[dict(prefix) for prefix in prefixes], delta_file=delta_file
        )

    def _record(self, table: str, **fields: Any) -> str:
        self.counts[table] += 1
        token = self._token(table, self.counts[table])
        self.records[(table, token)] = EpochChange(table=table, token=token, **fields)
        return token

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return self.records.get((table_name, token))

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


class _Scopes:
    def __init__(self) -> None:
        self.version = 1
        self.epochs: dict[str, int] = {"state:VA": 1, "state:IL": 1}

    async def settled(self) -> Any:
        return SimpleNamespace(version=self.version, epochs=dict(self.epochs), pending=frozenset())

    def commit(self, *scopes: str) -> None:
        self.version += 1
        for scope in scopes:
            self.epochs[scope] = self.version


class _Notes:
    """the change notes' door; counts each ask (a copy waiting for a file asks none)."""

    def __init__(self) -> None:
        self.asked = 0

    async def noted_keys(self, table: str, **_: Any) -> NotedKeysPage:
        self.asked += 1
        return NotedKeysPage(keys=(), whole_table=False, next=None)


@dataclass
class _Registration:
    subject: Subject
    cb: Any
    queue: str | None
    is_closed: bool = False


# parity-exempt: subset shim for nats.aio.Client carrying only subscribe/unsubscribe/flush/publish_raw_reply/request_raw, the whole surface serve_owner and forward reach through; it exists so the copy's request and the writer's subscription derive the delta-files subject for real rather than against a stub
class _Bus:
    """just enough NATS for ``serve_owner`` and ``forward`` to meet: a request to one subscriber, its reply back."""

    def __init__(self) -> None:
        self.registrations: list[_Registration] = []
        self.requests: list[bytes] = []
        self._replies: dict[str, asyncio.Future[bytes]] = {}
        self._inbox = 0

    async def subscribe(self, subject: Any = None, *, cb: Any, queue: str | None = None, **_: Any) -> _Registration:
        registration = _Registration(subject=subject, cb=cb, queue=queue)
        self.registrations.append(registration)
        return registration

    async def unsubscribe(self, sub: Any) -> None:
        if sub in self.registrations:
            self.registrations.remove(sub)
            sub.is_closed = True

    async def flush(self, timeout: float = 2.0) -> None:
        del timeout

    async def publish_raw_reply(self, *, reply_subject: str, payload: bytes) -> None:
        pending = self._replies.get(reply_subject)
        if pending is not None and not pending.done():
            pending.set_result(payload)

    async def request_raw(self, *, subject: Any, payload: bytes, timeout: timedelta = DEFAULT_REQUEST_TIMEOUT) -> bytes:
        self.requests.append(payload)
        matched = [r for r in self.registrations if r.subject.path == subject.path]
        if not matched:
            raise RequestError(f"no responders for {subject.path}") from NoRespondersError()
        self._inbox += 1
        inbox = f"_INBOX.fake.{self._inbox}"
        reply: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        self._replies[inbox] = reply
        # one subscriber of the queue group, never all of them
        await matched[0].cb(IncomingMessage(data=payload, reply_subject=inbox, subject=subject.path))
        return await asyncio.wait_for(reply, timeout=timeout.total_seconds())


class _Recovery:
    """decision 28's recovery, as the writer hands it a table: counts each hand-over."""

    def __init__(self) -> None:
        self.handed: list[tuple[str, str, str]] = []

    async def recover_table(self, table: str, *, epoch: str, file: str) -> None:
        self.handed.append((table, epoch, file))


@dataclass
class _World:
    db: _Db
    copy_l3: _L3
    writer_l3: _L3
    log: _Log
    lock: _Scopes
    backend: DuckDBBackend
    follower: CopyFollower
    bus: _Bus
    recreator: DeltaRecreator
    recovery: _Recovery
    notes: _Notes


def _tables() -> list[FollowedTable]:
    return [
        FollowedTable(name=name, key=_KEY, columns=_COLUMNS, scope=("state", "state:"), delta_files=True)
        for name in (_RESULTS, _TALLIES)
    ]


def _world() -> _World:
    db, log, lock, notes = _Db(), _Log(), _Scopes(), _Notes()
    copy_l3, writer_l3 = _L3(db), _L3(db)
    backend = DuckDBBackend()
    _OPENED.append(backend.reset)
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    for name in (_RESULTS, _TALLIES):
        registry.follow_generation(name)
    registry.set_noted_keys_reader(notes)
    bus, recovery = _Bus(), _Recovery()
    follower = CopyFollower(
        registry,
        _tables(),
        backend,
        copy_l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=50,
        stamp_table="scope_epochs",
        recreate=DeltaRecreateRequester(bus, _KEY_OF),
    )
    follower.register()
    recreator = DeltaRecreator(_tables(), writer_l3, log, recovery=recovery, page_size=50)
    return _World(db, copy_l3, writer_l3, log, lock, backend, follower, bus, recreator, recovery, notes)


def _row(state: str, geo_id: int, votes: int, note: str | None = "first") -> dict[str, Any]:
    return {"state": state, "geo_id": geo_id, "votes": votes, "note": note, "date_created": _BORN}


def _instant(value: Any) -> Any:
    if value is None:
        return None
    return (datetime.fromisoformat(value) if isinstance(value, str) else value).astimezone(UTC)


def _copy(backend: DuckDBBackend, table: str) -> list[tuple[Any, ...]]:
    with backend.read_snapshot() as cursor:
        rows = cursor.execute(f"SELECT {', '.join(_NAMES)} FROM {table}").fetchall()  # noqa: S608
    return sorted((*row[:-1], _instant(row[-1])) for row in rows)


def _same(world: _World, table: str) -> bool:
    return _copy(world.backend, table) == [(*row[:-1], _instant(row[-1])) for row in world.db.held(table)]


def _save(volume: Path, table: str, write: str, rows: Sequence[dict[str, Any]], deletes: Sequence[Any] = ()) -> str:
    delta = TableDelta(name=table, types=_TYPES, key=_KEY)
    delta.rows(rows, list(_NAMES))
    if deletes:
        delta.deletes(deletes)
    return write_delta_file(volume, delta, write)


def _counted(name: str, labels: Mapping[str, str]) -> float:
    return REGISTRY.get_sample_value(name, dict(labels)) or 0.0


async def _until(done: Callable[[], bool]) -> None:
    while not done():
        await asyncio.sleep(0.005)


async def _warmed(world: _World) -> None:
    for table in (_RESULTS, _TALLIES):
        rows = [_row("VA", i, 10 + i) for i in range(5)] + [_row("IL", i, 20 + i) for i in range(5)]
        world.db.insert(table, rows)
        world.log.commit(table, [(row["state"], row["geo_id"]) for row in rows], None)
    await world.follower.warm()
    assert await world.follower.catch_up()
    assert _same(world, _RESULTS) and _same(world, _TALLIES)
    world.copy_l3.statements = world.writer_l3.statements = 0


# --- each way a file is unreadable, as the reader meets it ---------------------------------------------------------


def _truncated(good: bytes, _volume: Path) -> bytes:
    return good[: len(good) // 2]


def _garbage(good: bytes, _volume: Path) -> bytes:
    return bytes((index * 151 + 7) % 256 for index in range(len(good)))


def _ipc(good: bytes) -> bytes:
    """a file's IPC bytes, its checksum trailer taken off."""
    return good[:-_CHECKSUM_TRAILER]


def _sealed(ipc: bytes) -> bytes:
    """IPC bytes closed by a checksum that matches them: damage the checksum alone would not catch."""
    return ipc + hashlib.sha256(ipc).digest() + _CHECKSUM_MARK


def _wrong_format(good: bytes, _volume: Path) -> bytes:
    read = pa_ipc.open_file(pa.BufferReader(_ipc(good))).read_all()
    metadata = dict(read.schema.metadata)
    metadata[b"threetears.delta.format"] = b"2"
    sink = pa.BufferOutputStream()
    with pa_ipc.new_file(sink, read.schema.with_metadata(metadata)) as writer:
        writer.write_table(read.replace_schema_metadata(metadata))
    return _sealed(bytes(sink.getvalue().to_pybytes()))


def _bad_checksum(good: bytes, _volume: Path) -> bytes:
    """one byte flipped in the middle of the IPC bytes, the checksum left as written."""
    flipped = bytearray(good)
    flipped[len(_ipc(good)) // 2] ^= 0x5A
    return bytes(flipped)


def _bad_zstd(good: bytes, volume: Path) -> bytes:
    """a byte flipped inside the first zstd frame, where the codec itself fails (never a byte it reads past), and the
    file sealed again, so the codec is what meets it."""
    ipc = _ipc(good)
    start = ipc.index(b"\x28\xb5\x2f\xfd")  # the zstd frame magic
    probe = volume / "probe.arrow"
    for offset in range(start + 4, start + 128):
        flipped = bytearray(ipc)
        flipped[offset] ^= 0x5A
        probe.write_bytes(_sealed(bytes(flipped)))
        try:
            read_delta_file(probe)
        except DeltaFileUnreadableError as exc:
            if "ZSTD" in str(exc):
                probe.unlink()
                return _sealed(bytes(flipped))
    raise AssertionError("no byte of the first zstd frame fails the codec")


def _another_tables(good: bytes, volume: Path) -> bytes:
    other = volume / "elsewhere"
    other.mkdir()
    name = _save(other, _TALLIES, "lost", [_row("VA", 1, 1)])
    return (other / name).read_bytes()


_UNREADABLE: dict[str, Callable[[bytes, Path], bytes]] = {
    "truncated": _truncated,
    "garbage": _garbage,
    "wrong_format": _wrong_format,
    "bad_zstd": _bad_zstd,
    "bad_checksum": _bad_checksum,
    "another_tables": _another_tables,
}


# --- the copy and the writer, end to end ----------------------------------------------------------------------------


@pytest.mark.parametrize("damage", [None, *sorted(_UNREADABLE)])
async def test_a_copy_finding_a_changes_file_missing_or_unreadable_reads_no_l3_asks_once_and_ends_equal_to_l3(
    _volume: Path, damage: str | None
) -> None:
    """Flips if a copy that finds a change's file missing (damage None) or unreadable sends L3 a statement, applies
    anything of that table before the file is back, asks the writer more than once, stops applying another table
    meanwhile, or does not end equal to L3 once the writer has recreated the file (decision 29). Flips too if a kind of
    damage is not read as unreadable (each is checked to raise DeltaFileUnreadableError first)."""
    world = _world()
    await _warmed(world)
    held = _copy(world.backend, _RESULTS)
    # the change whose file is lost: VA 1 updated, VA 4 deleted
    updated = _row("VA", 1, 61, "second")
    world.db.insert(_RESULTS, [updated])
    world.db.delete(_RESULTS, ("VA", 4))
    name = f"{_RESULTS}/lost.arrow"
    if damage is not None:
        good = (_volume / _save(_volume, _RESULTS, "lost", [updated], [("VA", 4)])).read_bytes()
        (_volume / name).write_bytes(_UNREADABLE[damage](good, _volume))
        with pytest.raises(DeltaFileUnreadableError):
            read_delta_file(_volume / name, table=_RESULTS)
    epoch = world.log.commit(_RESULTS, [("VA", 1), ("VA", 4)], name)
    # the other table's change, whose file is whole
    tallied = _row("VA", 2, 72, "tallied")
    world.db.insert(_TALLIES, [tallied])
    world.log.commit(_TALLIES, [("VA", 2)], _save(_volume, _TALLIES, "fine", [tallied]))
    world.lock.commit("state:VA")
    why = "file_missing" if damage is None else "file_unreadable"
    waits = _counted("copy_follower_delta_file_waits_total", {"table": _RESULTS, "why": why})
    asked = _counted("copy_follower_delta_file_recreate_requests_total", {"table": _RESULTS, "outcome": "recreated"})
    # the writer serves under the lease, and recreates once the test lets it
    let = asyncio.Event()
    async with serving_recreates(world.bus, _KEY_OF, world.recreator):
        world.bus.registrations[0].cb = _gate(world.bus.registrations[0].cb, let)
        await world.follower.catch_up()
        await asyncio.wait_for(_until(lambda: len(world.bus.requests) == 1), timeout=5)

        assert world.copy_l3.statements == 0, "a copy never reads a lost or damaged file's change from L3"
        assert _copy(world.backend, _RESULTS) == held, "nothing of the waiting table is applied"
        assert name in world.follower.status().waiting_for_file[_RESULTS]
        assert why in world.follower.status().waiting_for_file[_RESULTS]
        assert _same(world, _TALLIES), "another table is applied while one waits"
        assert _counted("copy_follower_delta_file_waits_total", {"table": _RESULTS, "why": why}) - waits == 1

        let.set()
        await asyncio.wait_for(_until(lambda: not world.follower.status().waiting_for_file), timeout=5)
        assert await world.follower.catch_up()

    assert _same(world, _RESULTS) and _same(world, _TALLIES)
    assert world.copy_l3.statements == 0
    assert len(world.bus.requests) == 1, "the copy asked the writer once"
    assert json.loads(world.bus.requests[0]) == {"table": _RESULTS, "file": name, "epoch": epoch}
    assert (
        _counted("copy_follower_delta_file_recreate_requests_total", {"table": _RESULTS, "outcome": "recreated"})
        - asked
        == 1
    )
    assert world.writer_l3.statements == 1, "the writer read the change's two keys in one keyed statement"


async def test_a_table_waiting_for_a_file_keeps_the_scopes_of_its_held_keys_at_the_epoch_the_copy_holds(
    _volume: Path,
) -> None:
    """Flips if, while a table waits for a change's file (decision 29), the copy is labelled at the writer's new stamp
    for the scope of the change it holds back (Virginia's): a reader would read the old row as current. A scope no
    held change names (Illinois, moved by the other table) is labelled current; once the file is back and applied, the
    label is the writer's stamp."""
    world = _world()
    await _warmed(world)
    va_before = world.lock.epochs["state:VA"]
    updated = _row("VA", 1, 61, "second")
    world.db.insert(_RESULTS, [updated])
    name = f"{_RESULTS}/lost.arrow"
    world.log.commit(_RESULTS, [("VA", 1)], name)
    world.lock.commit("state:VA")
    tallied = _row("IL", 2, 72, "tallied")
    world.db.insert(_TALLIES, [tallied])
    world.log.commit(_TALLIES, [("IL", 2)], _save(_volume, _TALLIES, "fine", [tallied]))
    world.lock.commit("state:IL")
    let = asyncio.Event()
    async with serving_recreates(world.bus, _KEY_OF, world.recreator):
        world.bus.registrations[0].cb = _gate(world.bus.registrations[0].cb, let)
        await world.follower.catch_up()
        await asyncio.wait_for(_until(lambda: len(world.bus.requests) == 1), timeout=5)
        assert _same(world, _TALLIES)
        stamp = world.follower.status().stamp
        assert isinstance(stamp, PartialStamp), f"labelled {stamp!r} while a table waits"
        assert stamp.epoch("state:VA") == va_before and "state:VA" in stamp.held
        assert stamp.epoch("state:IL") == world.lock.epochs["state:IL"] and "state:IL" not in stamp.held

        let.set()
        await asyncio.wait_for(_until(lambda: not world.follower.status().waiting_for_file), timeout=5)
        assert await world.follower.catch_up()
    assert _same(world, _RESULTS)
    stamp = world.follower.status().stamp
    assert not isinstance(stamp, PartialStamp) and stamp.epochs == world.lock.epochs


def _gate(callback: Any, let: asyncio.Event) -> Any:
    """the writer's subscription, held until ``let``: the test sees the copy waiting before the writer recreates."""

    async def held(message: IncomingMessage) -> None:
        await let.wait()
        await callback(message)

    return held


async def test_a_load_or_clear_whose_file_is_missing_is_recreated_with_l3s_rows_under_its_prefixes(
    _volume: Path,
) -> None:
    """Flips if a load's or a clear's lost file sends the copy to L3 for its prefixes (the registry's old path), or the
    writer's recreation leaves out the prefix it cleared or the rows L3 holds under it now: the copy would keep rows L3
    deleted (decision 23)."""
    world = _world()
    await _warmed(world)
    # IL replaced: every IL row gone, two new ones in
    world.db.delete_state(_RESULTS, "IL")
    world.db.insert(_RESULTS, [_row("IL", 100, 1, "new"), _row("IL", 101, 2, "new")])
    name = f"{_RESULTS}/load.arrow"
    world.log.cleared(_RESULTS, [{"state": "IL"}], name)
    world.lock.commit("state:IL")

    async with serving_recreates(world.bus, _KEY_OF, world.recreator):
        await world.follower.catch_up()
        await asyncio.wait_for(_until(lambda: not world.follower.status().waiting_for_file), timeout=5)
        assert await world.follower.catch_up()

    assert _same(world, _RESULTS)
    assert world.copy_l3.statements == 0
    recreated = read_delta_file(_volume / name, table=_RESULTS)
    assert recreated.prefixes == ({"state": "IL"},)
    assert sorted(change.row["geo_id"] for change in recreated.changes) == [100, 101]


# --- the writer's side ---------------------------------------------------------------------------------------------


async def test_the_writer_reads_only_the_changes_keys_by_key_and_recreates_once_for_many_requests(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the writer reads anything of L3 but the change's own keys (by key, counted), writes a key L3 no longer
    holds as anything but a delete marker, or recreates the file once per request rather than once: N copies finding
    one file lost would each cost the writer a read and a write."""
    world = _world()
    world.db.insert(_RESULTS, [_row("VA", 1, 11), _row("VA", 2, 12), _row("VA", 3, 13)])
    name = f"{_RESULTS}/lost.arrow"
    epoch = world.log.commit(_RESULTS, [("VA", 1), ("VA", 9)], name)  # VA 9 is gone from L3 since
    written: list[str] = []

    def counted_write(root: Path, delta: TableDelta, write_id: str) -> str:
        written.append(write_id)
        return write_delta_file(root, delta, write_id)

    monkeypatch.setattr(delta_recreate, "write_delta_file", counted_write)
    keys = _counted("delta_file_recreate_keys_read_total", {"table": _RESULTS})
    answers = await asyncio.gather(*(world.recreator.recreate(_RESULTS, name, epoch) for _ in range(8)))

    assert set(answers) == {"recreated"}
    assert written == ["lost"], "one recreation for eight requests"
    assert world.writer_l3.statements == 1
    assert _counted("delta_file_recreate_keys_read_total", {"table": _RESULTS}) - keys == 2
    recreated = read_delta_file(_volume / name, table=_RESULTS)
    assert [(c.row["state"], c.row["geo_id"], c.deleted) for c in recreated.changes] == [
        ("VA", 1, False),
        ("VA", 9, True),
    ]
    assert recreated.changes[0].row["votes"] == 11
    # asked again once it reads: present, nothing written or read
    assert await world.recreator.recreate(_RESULTS, name, epoch) == "present"
    assert written == ["lost"] and world.writer_l3.statements == 1


async def test_a_commit_too_big_to_list_whose_file_is_lost_goes_to_the_recovery_and_the_copy_reads_neither_l3_nor_notes(
    _volume: Path,
) -> None:
    """Flips if a keys-unknown commit's lost file is "recreated" (its keys were only in the file: nothing to read them
    by), if the copy names its keys from the change notes or reads L3, or if the table is not handed to decision 28's
    recovery once."""
    world = _world()
    await _warmed(world)
    name = f"{_RESULTS}/big.arrow"
    epoch = world.log.too_large(_RESULTS, name)
    world.lock.commit("state:VA")

    async with serving_recreates(world.bus, _KEY_OF, world.recreator):
        await world.follower.catch_up()
        await asyncio.wait_for(_until(lambda: bool(world.recovery.handed)), timeout=5)

    assert world.recovery.handed == [(_RESULTS, epoch, name)]
    assert not (_volume / name).exists()
    assert (world.copy_l3.statements, world.writer_l3.statements, world.notes.asked) == (0, 0, 0)
    assert name in world.follower.status().waiting_for_file[_RESULTS]
    assert "recovering" in world.follower.status().waiting_for_file[_RESULTS]
    world.follower.stop()


async def test_only_the_replica_serving_under_the_lease_answers_and_with_none_the_copy_hears_no_writer(
    _volume: Path,
) -> None:
    """Flips if a replica that does not hold the write lease recreates a file (two writers would race one name), or
    a request with no writer is anything but ``no_writer`` (the copy then asks again)."""
    world = _world()
    name = f"{_RESULTS}/lost.arrow"
    world.db.insert(_RESULTS, [_row("VA", 1, 11)])
    epoch = world.log.commit(_RESULTS, [("VA", 1)], name)
    replica_l3 = _L3(world.db)
    replica = DeltaRecreator(_tables(), replica_l3, world.log, recovery=world.recovery)
    ask = DeltaRecreateRequester(world.bus, _KEY_OF)

    assert await ask(_RESULTS, name, epoch) == "no_writer"

    async with serving_recreates(world.bus, _KEY_OF, world.recreator):
        assert await ask(_RESULTS, name, epoch) == "recreated"

    assert replica_l3.statements == 0, "the replica without the lease never answered"
    assert world.writer_l3.statements == 1
    assert world.bus.registrations == [], "the writer stops serving when it leaves the lease"
    del replica
    assert await ask(_RESULTS, name, epoch) == "no_writer"

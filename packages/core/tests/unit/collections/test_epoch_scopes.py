"""A keys-unknown epoch record names the scopes (states) its commit touched (``design-pod-delta-files.md``, decision 31).

- the record carries them and reads them back; a record naming none is stored as before, and a reader built before the
  field reads a new record as before;
- only a keys-unknown record names them, at most 256 of a bounded length: a commit touching more names
  none;
- the scopes the writer's seqlock touched in a transaction ride its ``l3.tx.commit`` beside its delta files, and an
  advance this process records keys unknown for a commit with a delta file names them.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from threetears.nats import Subject

from threetears.core.backends.broker_generation import (
    COMMIT_SCOPES_FIELD,
    DELTA_FILES_FIELD,
    named_commit_scopes,
    named_delta_files,
    naming_commit_scopes,
    naming_delta_files,
)
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.collections import WRITE_GENERATION, CollectionRegistry
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.epoch_change import (
    ChangedRow,
    EpochChange,
    advance_recording,
    epoch_change_body,
    read_epoch_change_body,
    recordable_scopes,
)
from threetears.core.collections.generation import advance_for_write
from threetears.core.collections.schema_backed import STRING_TYPE, Column, TableSchema, collection_for_schema
from threetears.core.collections.scope_epochs import ScopeEpochs
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.testing.l1_tables import TEST_L1_TABLE

from packages.core.tests.durable_support import ConnOnlyPool

pytest.importorskip("pyarrow")

#: the most scopes a record names (epoch_change._EPOCH_SCOPES_MAX)
_MOST = 256
_FILE = "results/0123456789abcdef0123456789abcdef.arrow"
_AT = datetime(2026, 11, 23, 3, 0, tzinfo=datetime.now().astimezone().tzinfo)
_BRACKET = (_AT, _AT + timedelta(seconds=2))


class _OlderBody(BaseModel):
    """the stored body as a reader built before the scopes reads it."""

    table: str
    keys_unknown: bool = False
    noted_between: tuple[datetime, datetime] | None = None
    delta_file: str | None = None


def test_a_keys_unknown_record_names_its_states_and_reads_them_back() -> None:
    """Flips if the stored record drops the scopes, or a record naming none stops being stored as before."""
    body = epoch_change_body(
        "results", None, keys_unknown=True, noted_between=_BRACKET, delta_file=_FILE, scopes=["state:GA", "state:TX"]
    )
    record = read_epoch_change_body(body, "results", "i:1")
    assert record.scopes == ["state:GA", "state:TX"] and record.delta_file == _FILE
    unnamed = epoch_change_body("results", None, keys_unknown=True, delta_file=_FILE)
    assert "scopes" not in json.loads(unnamed)
    assert read_epoch_change_body(unnamed, "results", "i:2").scopes is None


def test_a_reader_built_before_the_field_reads_a_new_record_as_before() -> None:
    """Flips if a record naming scopes is one an older reader (a replica not yet upgraded) cannot read."""
    body = epoch_change_body(
        "results", None, keys_unknown=True, noted_between=_BRACKET, delta_file=_FILE, scopes=["state:TX"]
    )
    older = _OlderBody.model_validate_json(body)
    assert older.keys_unknown and older.delta_file == _FILE and older.noted_between == _BRACKET


@pytest.mark.parametrize(
    ("rows", "keys_unknown", "scopes"),
    [
        ([ChangedRow(ids=["a"])], False, ["state:TX"]),  # a keyed record names its rows, never states
        (None, True, []),
        (None, True, ["state:TX", "state:TX"]),
        (None, True, ["s" * 129]),
        (None, True, [f"state:{n}" for n in range(_MOST + 1)]),
    ],
)
def test_scopes_only_on_a_keys_unknown_record_and_within_the_bound(
    rows: list[ChangedRow] | None, keys_unknown: bool, scopes: list[str]
) -> None:
    """Flips if a record names scopes beside its rows, none at all, a repeat, one too long or too many: a record past
    its bound breaks the log, and a reader reads such a record as malformed, behind, never misled."""
    with pytest.raises(ValueError, match="scopes"):
        epoch_change_body("results", rows, keys_unknown=keys_unknown, scopes=scopes)
    with pytest.raises(GenerationUnavailableError, match="scopes"):
        read_epoch_change_body(
            json.dumps(
                {
                    "table": "results",
                    "keys_unknown": keys_unknown,
                    "rows": [row.model_dump() for row in rows or []],
                    "scopes": scopes,
                }
            ).encode(),
            "results",
            "i:1",
        )


def test_recordable_scopes_names_each_once_and_none_past_the_bound() -> None:
    """Flips if the pod sends a commit's scopes unbounded (a record past its bound) or with repeats."""
    assert recordable_scopes(["state:TX", "state:GA", "state:TX"]) == ["state:GA", "state:TX"]
    assert recordable_scopes([]) is None
    assert recordable_scopes([f"state:{n}" for n in range(_MOST + 1)]) is None
    assert recordable_scopes(["s" * 129]) is None
    widest = [f"{n:03d}" + "s" * 125 for n in range(_MOST)]
    epoch_change_body("t" * 63, None, keys_unknown=True, delta_file=f"{'t' * 63}/{'w' * 32}.arrow", scopes=widest)


class _Source:
    """a generation source recording each advance's scopes and delta file."""

    def __init__(self) -> None:
        self.advances: list[tuple[str, bool, list[str] | None, str | None]] = []

    async def current(self, table_name: str) -> str:
        return f"inc:{len(self.advances)}"

    async def advance(self, table_name: str) -> str:
        return f"inc:{len(self.advances) + 1}"

    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        tenant: str | None = None,
        partition: Mapping[str, str] | None = None,
        noted_between: tuple[datetime, datetime] | None = None,
        prefixes: Sequence[Mapping[str, str]] | None = None,
        reason: str | None = None,
        delta_file: str | None = None,
        scopes: Sequence[str] | None = None,
    ) -> str:
        self.advances.append((table_name, keys_unknown, None if scopes is None else list(scopes), delta_file))
        return f"inc:{len(self.advances)}"


class _Older(_Source):
    """a source built before the scopes: it takes none."""

    async def advance_with_change(  # type: ignore[override]
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        delta_file: str | None = None,
    ) -> str:
        self.advances.append((table_name, keys_unknown, None, delta_file))
        return f"inc:{len(self.advances)}"


async def test_an_advance_hands_the_scopes_only_to_a_keys_unknown_record_and_a_source_that_takes_them() -> None:
    """Flips if a keys-unknown advance drops its scopes, a keyed advance is handed them, or a source built before the
    field is handed them (failing every write through it)."""
    source = _Source()
    await advance_recording(source, "results", None, keys_unknown=True, delta_file=_FILE, scopes=["state:TX"])
    await advance_recording(source, "results", [ChangedRow(ids=["a"])], delta_file=_FILE, scopes=["state:TX"])
    assert source.advances == [("results", True, ["state:TX"], _FILE), ("results", False, None, _FILE)]
    older = _Older()
    await advance_recording(older, "results", None, keys_unknown=True, delta_file=_FILE, scopes=["state:TX"])
    assert older.advances == [("results", True, None, _FILE)]


async def test_a_keys_unknown_advance_of_a_commit_names_the_states_the_commit_named_only_with_its_file() -> None:
    """Flips if a process recording its own keys-unknown advance does not name the states its commit touched, or names
    them on a record naming no delta file (nothing to recover by them: the copies read such a commit from the notes)."""
    source = _Source()
    with naming_delta_files({"results": _FILE}), naming_commit_scopes(["state:TX"]):
        await advance_for_write(source, "results", None, schema=None, keys_unknown=True)
        await advance_for_write(source, "results", [ChangedRow(ids=["a"])], schema=None)
    with naming_commit_scopes(["state:TX"]):
        await advance_for_write(source, "results", None, schema=None, keys_unknown=True)
    assert source.advances == [
        ("results", True, ["state:TX"], _FILE),
        ("results", False, None, _FILE),
        ("results", True, None, None),
    ]


_TX_ID = str(uuid.UUID("00000000-0000-4000-8000-000000000931"))


class _Broker:
    """answers each request with the next scripted reply, and keeps each request's body."""

    def __init__(self, *replies: dict[str, Any]) -> None:
        self.replies = list(replies)
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del timeout
        self.sent.append((str(subject), json.loads(payload)))
        return json.dumps(self.replies.pop(0)).encode("utf-8")

    def proxy(self) -> NatsProxyL3Backend:
        nats = MagicMock()
        nats.request_raw = AsyncMock(side_effect=self.request_raw)
        return NatsProxyL3Backend(
            nats_client=nats, namespace_prefix="test", agent_id="agent-123", identity_token=lambda: "token"
        )

    def body(self, subject_suffix: str) -> dict[str, Any]:
        (body,) = [body for subject, body in self.sent if subject.endswith(subject_suffix)]
        return body


async def test_a_pods_commit_hands_the_broker_its_states_beside_its_files() -> None:
    """Flips if a pod's ``l3.tx.commit`` of a commit with a delta file stops carrying the states it touched, for the
    broker to name on a keys-unknown record; or a commit naming no delta file, or a rollback, carries them."""
    broker = _Broker({"success": True, "tx_id": _TX_ID}, {"success": True})
    with naming_delta_files({"results": _FILE}), naming_commit_scopes(["state:GA", "state:TX"]):
        async with broker.proxy().transaction():
            pass
    body = broker.body("l3.tx.commit")
    assert body[COMMIT_SCOPES_FIELD] == ["state:GA", "state:TX"] and body[DELTA_FILES_FIELD] == {"results": _FILE}

    broker = _Broker({"success": True, "tx_id": _TX_ID}, {"success": True})
    with naming_commit_scopes(["state:TX"]):
        async with broker.proxy().transaction():
            pass
    assert COMMIT_SCOPES_FIELD not in broker.body("l3.tx.commit")

    broker = _Broker({"success": True, "tx_id": _TX_ID}, {"success": True})
    with pytest.raises(RuntimeError), naming_delta_files({"results": _FILE}), naming_commit_scopes(["state:TX"]):
        async with broker.proxy().transaction():
            raise RuntimeError("the body failed")
    assert COMMIT_SCOPES_FIELD not in broker.body("l3.tx.rollback")


class _Row(BaseEntity):
    primary_key_field = "geo_id"


class _Conn:
    """a caller's connection whose commit lands, keeping what the commit named as it ended."""

    def __init__(self) -> None:
        self.named: list[tuple[tuple[str, ...], dict[str, str]]] = []

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        yield
        self.named.append((named_commit_scopes(), named_delta_files()))

    def transaction(self, **options: Any) -> Any:
        return self._transaction()

    async def execute(self, sql: str, *params: Any) -> str:
        return "INSERT 0 1"

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        # the seqlock's fence: the write is still the one in progress
        return [{"writing": params[-1] if params else 7}]


async def test_the_states_the_seqlock_touched_in_a_transaction_are_named_by_its_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the scopes ``ScopeEpochs.touch`` marks pending in a write's transaction are not what its commit names (the
    states a keys-unknown record of it would name), or a commit saving no delta file names them."""
    monkeypatch.setenv("THREETEARS_DELTA_FILES_DIR", str(tmp_path))
    registry = CollectionRegistry()
    registry.set_generation_source(_Source())
    schema = TableSchema(
        name="results",
        primary_key=("state", "geo_id"),
        columns=[Column("state", STRING_TYPE), Column("geo_id", STRING_TYPE), Column("note", STRING_TYPE)],
        l1_eviction=TEST_L1_TABLE,
        delta_files=True,
    )
    base = collection_for_schema(schema, entity_class=_Row)
    results = type("_Results", (base,), {"write_generation": WRITE_GENERATION})(registry, DefaultCoreConfig(), None)
    results.l3_pool = ConnOnlyPool()
    stamps = MagicMock(
        table_name="scope_epochs", invalidation_columns=(), invalidate_cache_many=AsyncMock(), writes_delta_files=False
    )
    epochs = ScopeEpochs(stamps)
    conn = _Conn()
    async with CallerTransaction(conn) as transaction:
        await results.save_rows([{"state": "TX", "geo_id": "1", "note": "n"}], conn=conn)
        await epochs.touch(7, ["state:TX", "state:GA"], conn=conn)
        await epochs.touch(7, ["state:TX"], conn=conn)
    assert conn.named == [(("state:GA", "state:TX"), dict(transaction.delta_files))]

    quiet = _Conn()
    async with CallerTransaction(quiet):
        await epochs.touch(8, ["state:TX"], conn=quiet)
    assert quiet.named == [((), {})], "no delta file, no states"


def test_a_record_validated_directly_refuses_scopes_beside_rows() -> None:
    """Flips if the record model itself accepts scopes on a keyed record (the form every reader builds)."""
    with pytest.raises(ValueError, match="scopes"):
        EpochChange(table="results", token="i:1", rows=[ChangedRow(ids=["a"])], scopes=["state:TX"])

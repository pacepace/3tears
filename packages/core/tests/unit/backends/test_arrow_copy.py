"""An ``l3.tx.copy`` request's rows as Arrow, sent only to a hub that announces it reads them.

What the pod sends is the contract the hub's broker implements (its ``test_copy_arrow`` holds the other half: the
``COPY`` bytes of an Arrow request are those of the JSON request of the same rows). Pinned here:

- to a hub whose ``tx.begin`` reply says ``copy_arrow``, the request is the magic, the envelope (the JSON request's
  fields but its rows, the rows' digest and their JSON columns) and the rows' IPC stream, whose values are those
  the JSON request carries, value for value and kind for kind;
- to a hub that does not, and for rows that as Arrow pass the request's budget, the JSON request exactly as before;
- a proof over the envelope verifies at the door and covers the rows through their digest;
- the rows are laid out on a thread, never the event loop.
"""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pyarrow as pa
import pytest
from threetears.nats import Subject

from threetears.core.backends import nats_proxy
from threetears.core.backends.arrow_copy import (
    COPY_ARROW_FIELD,
    JSON_COLUMNS_FIELD,
    ROWS_DIGEST_FIELD,
    copy_rows_arrow,
    rows_digest,
    split_copy_request,
)
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.backends.schema_sql import json_default, transport_param
from threetears.core.security import (
    PopKey,
    build_jwks,
    generate_signing_keypair,
    jwk_thumbprint,
    sign_identity_token,
    verify_door_request,
)
from threetears.core.security.identity_token import IdentityClaims

_TX_ID = "019d9a00-0000-7000-8000-0000000000aa"
_HUB, _HUB_PUB = generate_signing_keypair()
_JWKS = build_jwks({"hub-1": _HUB_PUB})
_AGENT, _AGENT_PUB = generate_signing_keypair()

_COLUMNS = ["name", "n", "big", "ratio", "flag", "id", "share", "at", "on", "blob", "doc", "tags", "mixed", "none"]


def _record(n: int) -> tuple[Any, ...]:
    """one row of the values a pod hands a copy, every kind the rail carries."""
    return (
        f'name {n} ✓ "q" \\',
        n,
        2**63 - 1 if n % 2 else 2**64,
        (-0.0, 0.0, math.nan, math.inf, n / 7)[n % 5],
        n % 2 == 0,
        UUID(int=n + 1),
        Decimal(f"{n}.50"),
        datetime(2026, 11, 4, 4, 41, 7, n, tzinfo=UTC if n % 2 else None),
        date(2026, 11, 1 + n % 8),
        bytes([n % 256, 0]),
        {"k": [1, 2.5, None], "at": datetime(2026, 1, 1, tzinfo=UTC)},
        ["a", None, n],
        (1, 2.5, "x", None)[n % 4],
        None,
    )


_RECORDS = [_record(n) for n in range(40)] + [tuple([None] * len(_COLUMNS))]


def _token() -> str:
    now = int(time.time())
    return sign_identity_token(
        IdentityClaims(
            sub="agent-1",
            customer_id="cust-1",
            sid="sid-1",
            pod_id="pod-a",
            iss="hub",
            iat=now,
            exp=now + 600,
            cnf=jwk_thumbprint(_AGENT_PUB),
        ),
        signing_key=_HUB,
        kid="hub-1",
    )


class _Hub:
    """the bus a proxy sends on: keeps each request's bytes, and answers ``tx.begin`` with ``begin``."""

    def __init__(self, begin: dict[str, Any]) -> None:
        self.raw: list[tuple[str, bytes]] = []
        self._begin = begin

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del timeout
        self.raw.append((subject.path, payload))
        reply = self._begin if subject.path.endswith("tx.begin") else {"success": True, "row_count": 41}
        return json.dumps(reply).encode("utf-8")

    def proxy(self, *, proven: bool = False) -> NatsProxyL3Backend:
        nats = MagicMock()
        nats.request_raw = self.request_raw
        nats.max_payload = 64 * 1024 * 1024
        token = _token()
        return NatsProxyL3Backend(
            nats_client=nats,
            namespace_prefix="test",
            agent_id="agent-1",
            identity_token=lambda: token,
            pop_key=(lambda: PopKey(_AGENT)) if proven else None,
        )

    def copy(self) -> bytes:
        [data] = [payload for subject, payload in self.raw if subject.endswith("tx.copy")]
        return data


async def _copied(begin: dict[str, Any], records: list[tuple[Any, ...]], *, proven: bool = False) -> bytes:
    hub = _Hub({"success": True, "tx_id": _TX_ID, **begin})
    async with hub.proxy(proven=proven).acquire() as conn, conn.transaction():
        assert await conn.copy_records_to_table("rows", records=records, columns=_COLUMNS) == "COPY 41"
    return hub.copy()


def _arrow_parts(data: bytes) -> tuple[dict[str, Any], bytes]:
    split = split_copy_request(data)
    assert split is not None
    envelope, offset = split
    return json.loads(bytes(envelope)), data[offset:]


def _json_rows(records: list[tuple[Any, ...]]) -> str:
    """the rows as the JSON request carries them, read back and written again: every kind and value it holds."""
    sent = json.dumps([[transport_param(value) for value in record] for record in records], default=json_default)
    return json.dumps(json.loads(sent))


async def test_a_hub_that_reads_arrow_is_sent_the_rows_the_json_request_carries_as_arrow() -> None:
    data = await _copied({COPY_ARROW_FIELD: True}, _RECORDS)
    assert split_copy_request(data) is not None
    envelope, ipc = _arrow_parts(data)
    assert set(envelope) == {
        "correlation_id", "identity_token", "tx_id", "table", "columns", ROWS_DIGEST_FIELD, JSON_COLUMNS_FIELD,
    }  # fmt: skip
    assert (envelope["tx_id"], envelope["table"], envelope["columns"]) == (_TX_ID, "rows", _COLUMNS)
    assert envelope[ROWS_DIGEST_FIELD] == rows_digest(ipc)
    table = pa.ipc.open_stream(ipc).read_all()
    as_json = set(envelope[JSON_COLUMNS_FIELD])
    # an integer past 64 bits, a date, an object, an array and a mix of kinds go as their JSON text
    assert as_json == {_COLUMNS.index(name) for name in ("big", "on", "doc", "tags", "mixed")}
    assert table.schema.field("ratio").type == pa.float64()
    assert table.schema.field("n").type == pa.int64()
    columns = [
        [None if text is None else json.loads(text) for text in table.column(index).to_pylist()]
        if index in as_json
        else table.column(index).to_pylist()
        for index in range(len(_COLUMNS))
    ]
    # value for value and kind for kind (1 and 1.0, 0.0 and -0.0, NaN), as JSON writes them
    assert json.dumps([list(row) for row in zip(*columns, strict=True)]) == _json_rows(_RECORDS)


async def test_a_hub_that_does_not_announce_it_is_sent_json_as_before() -> None:
    for begin in ({}, {COPY_ARROW_FIELD: False}, {COPY_ARROW_FIELD: "yes"}):
        body = json.loads(await _copied(begin, _RECORDS))
        assert json.dumps(body["rows"]) == _json_rows(_RECORDS)


async def test_rows_that_as_arrow_pass_the_requests_budget_go_as_json() -> None:
    # nulls and small integers take more bytes as Arrow than as JSON
    records = [(None, 1) + (None,) * (len(_COLUMNS) - 2)] * 41
    budget = len(copy_rows_arrow(_COLUMNS, records).ipc) + 64 * 1024 - 1
    data = await _copied({COPY_ARROW_FIELD: True, "max_request_bytes": budget}, records)
    assert data.startswith(b"{")
    data = await _copied({COPY_ARROW_FIELD: True, "max_request_bytes": budget + 1}, records)
    assert split_copy_request(data) is not None


async def test_a_proof_over_the_envelope_verifies_at_the_door_and_names_the_rows() -> None:
    data = await _copied({COPY_ARROW_FIELD: True}, _RECORDS, proven=True)
    envelope, ipc = _arrow_parts(data)
    caller = verify_door_request(envelope, jwks=_JWKS, hub_issuer="hub", door="l3", op="tx.copy")
    assert caller.agent_id == "agent-1"
    assert envelope[ROWS_DIGEST_FIELD] == rows_digest(ipc)


async def test_the_rows_are_laid_out_on_a_thread_never_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    threads: list[threading.Thread] = []

    def laid_out(columns: Any, records: Any) -> Any:
        threads.append(threading.current_thread())
        return copy_rows_arrow(columns, records)

    monkeypatch.setattr(nats_proxy, "copy_rows_arrow", laid_out)
    await _copied({COPY_ARROW_FIELD: True}, _RECORDS)
    assert threads and threads[0] is not threading.main_thread()


def test_a_request_that_is_not_one_is_not_split() -> None:
    assert split_copy_request(b'{"rows": []}') is None
    with pytest.raises(ValueError, match="past its end"):
        split_copy_request(b"L3COPYA1" + (100).to_bytes(4, "big") + b"{}")

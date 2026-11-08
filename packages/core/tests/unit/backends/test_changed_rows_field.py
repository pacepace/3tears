"""a pod's collection write names the rows it changes, so the broker's epoch record is exactly them.

The broker advances a table's generation after it commits a pod's write and stores the epoch's change
record. It cannot read the changed keys out of SQL, so the pod names them (``CHANGED_ROWS_FIELD``) on
the request that commits: ``l3.query``, ``l3.batch``, ``l3.tx.commit``. A write that names none sends no
field, and a read never carries one.
"""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from threetears.nats import Subject

from threetears.core.backends.broker_generation import CHANGED_ROWS_FIELD, naming_changed_rows
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.collections.epoch_change import ChangedRow

_TABLE = "group_members"
_TX_ID = str(uuid.uuid4())


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


def _rows(*ids: str) -> list[dict[str, Any]]:
    return [ChangedRow(ids=[each], columns={"member_id": "ada"}).model_dump(exclude_none=True) for each in ids]


class TestAWriteNamesItsRows:
    async def test_a_query(self) -> None:
        broker = _Broker({"success": True, "row_count": 1})
        with naming_changed_rows(_TABLE, _rows("m1")):
            await broker.proxy().execute("UPDATE group_members SET member_id = $1", "ada")
        assert broker.body("l3.query")[CHANGED_ROWS_FIELD] == {_TABLE: _rows("m1")}

    async def test_a_batch(self) -> None:
        broker = _Broker({"success": True, "results": [{"success": True}]})
        with naming_changed_rows(_TABLE, _rows("m1", "m2")):
            await broker.proxy().execute_batch([{"query": "DELETE FROM group_members", "params": []}])
        assert broker.body("l3.batch")[CHANGED_ROWS_FIELD] == {_TABLE: _rows("m1", "m2")}

    async def test_a_transaction_names_every_statements_rows_on_its_commit(self) -> None:
        broker = _Broker(
            {"success": True, "tx_id": _TX_ID},
            {"success": True, "row_count": 1},
            {"success": True, "row_count": 1},
            {"success": True},
        )
        async with broker.proxy().transaction() as conn:
            with naming_changed_rows(_TABLE, _rows("m1")):
                await conn.execute("UPDATE group_members SET member_id = $1", "ada")
            with naming_changed_rows(_TABLE, _rows("m2", "m1")):
                await conn.execute("UPDATE group_members SET member_id = $1", "ada")
        assert broker.body("l3.tx.commit")[CHANGED_ROWS_FIELD] == {_TABLE: _rows("m1", "m2")}


class TestNothingIsNamedWhereNothingWas:
    async def test_a_write_that_named_no_rows(self) -> None:
        broker = _Broker({"success": True, "row_count": 1})
        await broker.proxy().execute("UPDATE group_members SET member_id = $1", "ada")
        assert CHANGED_ROWS_FIELD not in broker.body("l3.query")

    async def test_a_read(self) -> None:
        broker = _Broker({"success": True, "rows": []})
        with naming_changed_rows(_TABLE, _rows("m1")):
            await broker.proxy().fetch("SELECT id FROM group_members")
        assert CHANGED_ROWS_FIELD not in broker.body("l3.query")

    async def test_a_rolled_back_transaction_and_the_next_one(self) -> None:
        broker = _Broker(
            {"success": True, "tx_id": _TX_ID},
            {"success": True, "row_count": 1},
            {"success": True},
            {"success": True, "tx_id": _TX_ID},
            {"success": True},
        )
        proxy = broker.proxy()
        try:
            async with proxy.transaction() as conn:
                with naming_changed_rows(_TABLE, _rows("m1")):
                    await conn.execute("UPDATE group_members SET member_id = $1", "ada")
                raise RuntimeError("the body failed")
        except RuntimeError:
            pass
        async with proxy.transaction():
            pass
        assert CHANGED_ROWS_FIELD not in broker.body("l3.tx.rollback")
        assert CHANGED_ROWS_FIELD not in broker.body("l3.tx.commit"), "a rolled-back transaction's rows linger"

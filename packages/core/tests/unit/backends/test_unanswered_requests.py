"""an L3 request that goes unanswered is sent again only where that is safe, and every attempt is bounded.

A hub rollout stops the replica a request was routed to. Pinned here against a scripted bus:

- a read that times out, or that a leaving replica cut off, is sent again, encoded afresh;
- a request no hub received is sent again, a write too: it ran nowhere;
- a write that timed out or was cut off raises at once: it may have run, and its caller's retry decides;
- a read unanswered on every attempt raises after the last one;
- a rollback nobody answers does not hide the error that ended the transaction.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pyarrow as pa
import pytest
from threetears.nats import Subject
from threetears.nats.errors import NoRespondersError, RequestTimeoutError

from threetears.core.backends import nats_proxy
from threetears.core.backends.nats_proxy import REPLICA_LEAVING_ERROR_CODE, NatsProxyL3Backend
from threetears.core.exceptions import DataLayerUnavailableError

_TX_ID = "019d9a00-0000-7000-8000-000000000000"

_LEFT = {"success": False, "error_code": REPLICA_LEAVING_ERROR_CODE, "error_message": "the replica shut down"}


def _proxy(*answers: dict[str, Any] | Exception) -> tuple[NatsProxyL3Backend, list[tuple[str, bytes]]]:
    """a proxy whose bus answers each request with the next scripted reply, or raises the next error.

    :param answers: reply dicts and errors, in request order
    :ptype answers: dict[str, Any] | Exception
    :return: the proxy, and every (subject, payload) it sent
    :rtype: tuple[NatsProxyL3Backend, list[tuple[str, bytes]]]
    """
    queue = list(answers)
    sent: list[tuple[str, bytes]] = []

    async def request_raw(*, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del timeout
        sent.append((subject.path, payload))
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return json.dumps(answer).encode("utf-8")

    nats = MagicMock()
    nats.request_raw = AsyncMock(side_effect=request_raw)
    proxy = NatsProxyL3Backend(
        nats_client=nats, namespace_prefix="test", agent_id="agent-123", identity_token=lambda: "test-identity-token"
    )
    return proxy, sent


def _rows(n: int) -> dict[str, Any]:
    return {"success": True, "rows": [{"n": n}], "row_count": 1}


class TestAReadIsSentAgain:
    async def test_a_read_that_timed_out_is_answered_by_its_next_attempt(self) -> None:
        proxy, sent = _proxy(RequestTimeoutError("request timed out"), _rows(7))

        assert await proxy.fetch("SELECT count(*) AS n FROM items") == [{"n": 7}]
        assert len(sent) == 2

    async def test_a_read_a_leaving_replica_cut_off_is_sent_again(self) -> None:
        proxy, sent = _proxy(_LEFT, _rows(3))

        assert await proxy.fetchval("SELECT count(*) AS n FROM items") == 3
        assert len(sent) == 2

    async def test_a_read_unanswered_every_time_raises_after_the_last_attempt(self) -> None:
        proxy, sent = _proxy(*(RequestTimeoutError("request timed out") for _ in range(nats_proxy.UNANSWERED_ATTEMPTS)))

        with pytest.raises(DataLayerUnavailableError, match="timed out"):
            await proxy.fetch("SELECT 1 AS n FROM items")
        assert len(sent) == nats_proxy.UNANSWERED_ATTEMPTS

    async def test_a_read_cut_off_every_time_raises_the_last_refusal(self) -> None:
        proxy, sent = _proxy(*(_LEFT for _ in range(nats_proxy.UNANSWERED_ATTEMPTS)))

        with pytest.raises(DataLayerUnavailableError, match=REPLICA_LEAVING_ERROR_CODE):
            await proxy.fetch("SELECT 1 AS n FROM items")
        assert len(sent) == nats_proxy.UNANSWERED_ATTEMPTS


class TestAWriteIsSentAgainOnlyWhenItRanNowhere:
    async def test_a_write_no_hub_received_is_sent_again(self) -> None:
        proxy, sent = _proxy(NoRespondersError("no responders"), {"success": True, "row_count": 1})

        assert await proxy.execute("INSERT INTO items (id) VALUES ($1)", 1) == "INSERT 0 1"
        assert len(sent) == 2

    async def test_a_write_that_timed_out_raises_without_being_sent_again(self) -> None:
        proxy, sent = _proxy(RequestTimeoutError("request timed out"), {"success": True, "row_count": 1})

        with pytest.raises(DataLayerUnavailableError, match="timed out"):
            await proxy.execute("INSERT INTO items (id) VALUES ($1)", 1)
        assert len(sent) == 1

    async def test_a_returning_write_sent_to_fetch_is_not_sent_again(self) -> None:
        proxy, sent = _proxy(RequestTimeoutError("request timed out"), _rows(1))

        with pytest.raises(DataLayerUnavailableError, match="timed out"):
            await proxy.fetch("INSERT INTO items (id) VALUES ($1) RETURNING id", 1)
        assert len(sent) == 1

    @pytest.mark.parametrize(
        "query",
        [
            "WITH d AS (DELETE FROM items WHERE id = $1 RETURNING id) SELECT id FROM d",
            "SELECT nextval('items_id_seq') AS id",
            "SELECT id FROM items WHERE id = $1 FOR UPDATE",
        ],
    )
    async def test_a_select_that_may_write_is_sent_once(self, query: str) -> None:
        for read in ("fetch", "fetch_arrow"):
            proxy, sent = _proxy(RequestTimeoutError("request timed out"), _rows(1))

            with pytest.raises(DataLayerUnavailableError, match="timed out"):
                if read == "fetch":
                    await proxy.fetch(query, 1)
                else:
                    await proxy.fetch_arrow(query, 1, types={"id": pa.int64()})
            assert len(sent) == 1, f"{read} sent a statement that may write again"

    async def test_a_write_a_leaving_replica_cut_off_raises_its_refusal(self) -> None:
        proxy, sent = _proxy(_LEFT, {"success": True, "row_count": 1})

        with pytest.raises(DataLayerUnavailableError, match=REPLICA_LEAVING_ERROR_CODE):
            await proxy.execute("UPDATE items SET label = $1", "x")
        assert len(sent) == 1

    async def test_a_transaction_statement_that_timed_out_raises_without_being_sent_again(self) -> None:
        proxy, sent = _proxy(
            {"success": True, "tx_id": _TX_ID},
            RequestTimeoutError("request timed out"),
            {"success": True},
        )

        with pytest.raises(DataLayerUnavailableError, match="timed out"):
            async with proxy.transaction() as conn:
                await conn.execute("INSERT INTO items (id) VALUES ($1)", 1)
        assert [subject.rsplit(".", 1)[-1] for subject, _ in sent] == ["begin", "execute", "rollback"]


class TestEachAttemptIsItsOwnRequest:
    async def test_each_attempt_is_encoded_afresh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proxy, sent = _proxy(RequestTimeoutError("request timed out"), _rows(1))
        encoded: list[str] = []
        encode = proxy.encode_request

        def counting(subject: str, payload: dict[str, Any]) -> bytes:
            encoded.append(subject)
            return encode(subject, payload)

        monkeypatch.setattr(proxy, "encode_request", counting)

        await proxy.fetch("SELECT 1 AS n FROM items")
        assert len(encoded) == len(sent) == 2, (
            "a proof is spent by the replica that judged it, so each attempt has its own"
        )


class TestAnUnansweredRollback:
    async def test_does_not_hide_the_error_that_ended_the_transaction(self) -> None:
        proxy, sent = _proxy(
            {"success": True, "tx_id": _TX_ID},
            RequestTimeoutError("request timed out"),
            RequestTimeoutError("request timed out"),
        )

        with pytest.raises(DataLayerUnavailableError, match="tx.execute|timed out") as raised:
            async with proxy.transaction() as conn:
                await conn.execute("INSERT INTO items (id) VALUES ($1)", 1)
        assert "l3.tx.execute" in str(raised.value), "the statement's failure is raised, not the rollback's"
        assert [subject.rsplit(".", 1)[-1] for subject, _ in sent] == ["begin", "execute", "rollback"]

    async def test_an_unanswered_commit_still_raises(self) -> None:
        proxy, _ = _proxy({"success": True, "tx_id": _TX_ID}, RequestTimeoutError("request timed out"))

        with pytest.raises(DataLayerUnavailableError, match="tx.commit"):
            async with proxy.transaction():
                pass

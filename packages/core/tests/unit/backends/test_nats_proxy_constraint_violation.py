"""a constraint violation crosses the L3 broker as the asyncpg error a direct pool raises.

:class:`NatsProxyL3Backend` is a drop-in for an asyncpg pool, and code written against one
catches ``asyncpg.UniqueViolationError`` to detect a duplicate. Through the broker every
failed reply became :class:`DataLayerUnavailableError`, so that ``except`` never fired in a
broker-backed pod and a duplicate read as an outage.

The broker now answers a class-23 SQLSTATE with ``error_code="CONSTRAINT_VIOLATION"`` and
the server's fields; the proxy rebuilds the asyncpg exception from them. Every test here
drives the real decode: a reply dict, JSON-encoded as the hub sends it, through the proxy's
own request path.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest
from threetears.nats import Subject

from threetears.core.backends.nats_proxy import CONSTRAINT_VIOLATION_ERROR_CODE, NatsProxyL3Backend
from threetears.core.exceptions import DataLayerUnavailableError

_TX_ID = "019d9a00-0000-7000-8000-000000000000"


def _violation(sqlstate: str = "23505", **fields: Any) -> dict[str, Any]:
    """
    a failed reply as the broker sends it for a constraint violation.

    :param sqlstate: the server's SQLSTATE
    :ptype sqlstate: str
    :param fields: further contract fields, overriding the defaults
    :ptype fields: Any
    :return: the reply dict
    :rtype: dict[str, Any]
    """
    reply: dict[str, Any] = {
        "success": False,
        "error_code": CONSTRAINT_VIOLATION_ERROR_CODE,
        "error_message": 'duplicate key value violates unique constraint "ix_members_email"',
        "sqlstate": sqlstate,
        "constraint_name": "ix_members_email",
        "table_name": "members",
        "schema_name": "agent_0190",
        "column_name": None,
    }
    reply.update(fields)
    return reply


def _proxy(*replies: dict[str, Any]) -> NatsProxyL3Backend:
    """
    a proxy whose broker answers each request with the next scripted reply.

    :param replies: reply dicts, in request order
    :ptype replies: dict[str, Any]
    :return: the proxy
    :rtype: NatsProxyL3Backend
    """
    queue = list(replies)

    async def request_raw(*, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del subject, payload, timeout
        return json.dumps(queue.pop(0)).encode("utf-8")

    nats = MagicMock()
    nats.request_raw = AsyncMock(side_effect=request_raw)
    return NatsProxyL3Backend(
        nats_client=nats,
        namespace_prefix="test",
        agent_id="agent-123",
        identity_token=lambda: "test-identity-token",
    )


class TestTheDecodeRaisesTheAsyncpgError:
    """each SQLSTATE comes back as the class a direct pool would have raised."""

    async def test_a_unique_violation_is_catchable_as_asyncpg_unique_violation(self) -> None:
        """the survey's catch: ``except asyncpg.UniqueViolationError`` now fires through the broker."""
        proxy = _proxy(_violation())

        with pytest.raises(asyncpg.UniqueViolationError) as raised:
            await proxy.execute("INSERT INTO members (email) VALUES ($1)", "a@example.com")

        assert raised.value.sqlstate == "23505"
        assert raised.value.constraint_name == "ix_members_email"
        assert raised.value.table_name == "members"
        assert raised.value.schema_name == "agent_0190"
        assert "ix_members_email" in str(raised.value)

    @pytest.mark.parametrize(
        ("sqlstate", "expected"),
        [
            pytest.param("23503", asyncpg.ForeignKeyViolationError, id="foreign-key"),
            pytest.param("23502", asyncpg.NotNullViolationError, id="not-null"),
            pytest.param("23514", asyncpg.CheckViolationError, id="check"),
            pytest.param("23P01", asyncpg.ExclusionViolationError, id="exclusion"),
            pytest.param("23000", asyncpg.IntegrityConstraintViolationError, id="class-23-generic"),
        ],
    )
    async def test_every_integrity_class_maps_to_its_asyncpg_class(
        self, sqlstate: str, expected: type[asyncpg.PostgresError]
    ) -> None:
        """the whole of SQLSTATE class 23, each to its own class.

        :param sqlstate: the SQLSTATE the broker forwards
        :ptype sqlstate: str
        :param expected: the asyncpg class a direct pool raises for it
        :ptype expected: type[asyncpg.PostgresError]
        """
        proxy = _proxy(_violation(sqlstate))

        with pytest.raises(expected) as raised:
            await proxy.execute("INSERT INTO members (email) VALUES ($1)", "a@example.com")

        assert type(raised.value) is expected
        assert raised.value.sqlstate == sqlstate

    async def test_a_conflict_is_not_an_unavailability(self) -> None:
        """the wrong signal is the bug: a duplicate must not read as infrastructure."""
        proxy = _proxy(_violation())

        with pytest.raises(asyncpg.IntegrityConstraintViolationError) as raised:
            await proxy.execute("INSERT INTO members (email) VALUES ($1)", "a@example.com")

        assert not isinstance(raised.value, DataLayerUnavailableError)

    async def test_the_detail_crosses_when_the_broker_sends_it(self) -> None:
        """the optional ``detail`` field lands where asyncpg puts it."""
        proxy = _proxy(_violation(detail="Key (email)=(a@example.com) already exists."))

        with pytest.raises(asyncpg.UniqueViolationError) as raised:
            await proxy.execute("INSERT INTO members (email) VALUES ($1)", "a@example.com")

        assert raised.value.detail == "Key (email)=(a@example.com) already exists."


class TestEveryReplyPathDecodes:
    """the pod reads a failed reply on six paths; each must type a violation the same way."""

    async def test_a_select_path(self) -> None:
        """``fetch`` outside a transaction."""
        proxy = _proxy(_violation())

        with pytest.raises(asyncpg.UniqueViolationError):
            await proxy.fetch("INSERT INTO members (email) VALUES ($1) RETURNING id", "a@example.com")

    async def test_the_batch_path(self) -> None:
        """a transactional batch rolled back on a violation."""
        proxy = _proxy(_violation())

        with pytest.raises(asyncpg.UniqueViolationError):
            await proxy.execute_batch(
                [{"query": "INSERT INTO members (email) VALUES ($1)", "params": ["a@example.com"]}],
                transaction=True,
            )

    @pytest.mark.parametrize("method", ["execute", "fetchrow", "fetch"])
    async def test_inside_a_transaction(self, method: str) -> None:
        """``tx.execute`` / ``tx.fetchrow`` / ``tx.fetch``, then the rollback the error triggers.

        :param method: the connection method under test
        :ptype method: str
        """
        proxy = _proxy(
            {"success": True, "tx_id": _TX_ID},
            _violation(),
            {"success": True},
        )

        with pytest.raises(asyncpg.UniqueViolationError):
            async with proxy.acquire() as conn:
                async with conn.transaction():
                    await getattr(conn, method)("INSERT INTO members (email) VALUES ($1)", "a@example.com")

    async def test_a_deferred_constraint_at_commit(self) -> None:
        """a DEFERRABLE constraint fires at COMMIT; the commit reply types it too."""
        proxy = _proxy(
            {"success": True, "tx_id": _TX_ID},
            {"success": True, "row_count": 1},
            _violation(),
        )

        with pytest.raises(asyncpg.UniqueViolationError):
            async with proxy.acquire() as conn:
                async with conn.transaction():
                    await conn.execute("INSERT INTO members (email) VALUES ($1)", "a@example.com")


class TestWhatStaysUnavailable:
    """only a well-formed class-23 reply becomes a conflict; everything else is unchanged."""

    async def test_an_ordinary_broker_error_is_still_unavailable(self) -> None:
        """a refusal or an internal error keeps its existing type."""
        proxy = _proxy({"success": False, "error_code": "POOL_EXHAUSTED", "error_message": "pool exhausted"})

        with pytest.raises(DataLayerUnavailableError, match="POOL_EXHAUSTED"):
            await proxy.execute("INSERT INTO members (email) VALUES ($1)", "a@example.com")

    @pytest.mark.parametrize(
        "reply",
        [
            pytest.param(_violation(sqlstate="40001"), id="non-integrity-sqlstate"),
            pytest.param(_violation(sqlstate=None), id="no-sqlstate"),
        ],
    )
    async def test_a_malformed_violation_reply_is_not_trusted(self, reply: dict[str, Any]) -> None:
        """a violation code without a class-23 SQLSTATE is a broker fault, reported as one.

        :param reply: the malformed reply
        :ptype reply: dict[str, Any]
        """
        proxy = _proxy(reply)

        with pytest.raises(DataLayerUnavailableError, match=CONSTRAINT_VIOLATION_ERROR_CODE):
            await proxy.execute("INSERT INTO members (email) VALUES ($1)", "a@example.com")

"""a refused ``NOWAIT`` lock crosses the L3 broker as the asyncpg error a direct pool raises.

``SELECT ... FOR UPDATE NOWAIT`` answers SQLSTATE 55P03 when another transaction holds the row; a
caller that refuses at once rather than wait (``ScopeEpochs.begin``) must tell that refusal apart
from an outage by its type, through the broker as on a direct pool. The broker answers it with
``error_code="LOCK_NOT_AVAILABLE"`` and the SQLSTATE; the proxy rebuilds
``asyncpg.LockNotAvailableError``. Anything else stays unavailability.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest

from threetears.core.backends.nats_proxy import LOCK_NOT_AVAILABLE_ERROR_CODE
from threetears.core.exceptions import DataLayerUnavailableError

from .test_nats_proxy_constraint_violation import _TX_ID, _proxy


def _refused(**fields: Any) -> dict[str, Any]:
    """a failed reply as the broker sends it for a lock ``NOWAIT`` could not take.

    :param fields: contract fields overriding the defaults
    :ptype fields: Any
    :return: the reply dict
    :rtype: dict[str, Any]
    """
    reply: dict[str, Any] = {
        "success": False,
        "error_code": LOCK_NOT_AVAILABLE_ERROR_CODE,
        "error_message": 'could not obtain lock on row in relation "scope_epochs"',
        "sqlstate": "55P03",
    }
    reply.update(fields)
    return reply


async def test_a_refused_nowait_inside_a_transaction_is_asyncpg_lock_not_available() -> None:
    proxy = _proxy({"success": True, "tx_id": _TX_ID}, _refused(), {"success": True})

    with pytest.raises(asyncpg.LockNotAvailableError) as raised:
        async with proxy.acquire() as conn:
            async with conn.transaction():
                await conn.fetchrow("SELECT writing FROM scope_epochs WHERE scope = $1 FOR UPDATE NOWAIT", "*")

    assert raised.value.sqlstate == "55P03"
    assert not isinstance(raised.value, DataLayerUnavailableError)


async def test_the_code_without_its_sqlstate_stays_unavailable() -> None:
    """a malformed reply is a broker fault, not a refused lock."""
    proxy = _proxy(_refused(sqlstate=None))

    with pytest.raises(DataLayerUnavailableError):
        await proxy.fetchrow("SELECT 1")


async def test_a_lock_message_under_another_code_stays_unavailable() -> None:
    """the type comes from the code, never from the message's words."""
    proxy = _proxy(_refused(error_code="QUERY_EXECUTION_ERROR"))

    with pytest.raises(DataLayerUnavailableError):
        await proxy.fetchrow("SELECT 1")

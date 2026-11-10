"""which database an L3 answers from: YugabyteDB, or Postgres.

A reader that spells a statement for one of them asks here, once a read. A direct pool, or anything with an
asyncpg-shaped ``fetch`` that reaches the database, is asked ``SELECT version()``: YugabyteDB's says ``-YB-``. The rail's proxy cannot ask that (the hub's broker refuses a pod
any function it does not model), so it reads the capability names the broker sends on every reply
(:meth:`~threetears.core.backends.nats_proxy.NatsProxyL3Backend.hub_capabilities`): a broker over YugabyteDB names
:data:`_YUGABYTE_DIALECT_CAPABILITY`. A broker built before it names none, and is answered as Postgres, which keeps
every statement in the form it had.
"""

from __future__ import annotations

from typing import Any, Final

__all__ = ["l3_on_yugabyte"]

#: the capability a hub's broker names in its replies' ``hub_capabilities`` when its database is YugabyteDB (the hub's
#: ``DIALECT_CAPABILITY_PREFIX`` and the database it reads from ``version()`` at start, 14-eng-ai-bot 99a3266d6)
_YUGABYTE_DIALECT_CAPABILITY: Final = "dialect:yugabyte"

#: what ``version()`` says on YugabyteDB (``PostgreSQL 15.12-YB-2026.1.0.0-b0 ...``)
_YUGABYTE_VERSION_MARK: Final = "-YB-"


async def l3_on_yugabyte(l3: Any) -> bool:
    """whether ``l3`` answers from YugabyteDB, asked each time: a reader asks once a whole read (beside it, one round
    trip), and the answer is kept nowhere a second caller could fill it twice.

    :param l3: the rail's proxy (``hub_capabilities``), or anything with an asyncpg-shaped ``fetch``
    :ptype l3: Any
    :return: whether its database is YugabyteDB
    :rtype: bool
    """
    # looked up on the class: a mock's instance answers any name it is asked
    if callable(getattr(type(l3), "hub_capabilities", None)):
        return _YUGABYTE_DIALECT_CAPABILITY in await l3.hub_capabilities()
    rows = await l3.fetch("SELECT version() AS v")
    return bool(rows) and _YUGABYTE_VERSION_MARK in str(rows[0]["v"])

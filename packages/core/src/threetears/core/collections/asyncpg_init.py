"""canonical asyncpg connection initializer for every 3tears consumer.

every consumer that owns an asyncpg pool MUST call
:func:`init_connection` from the pool's ``init=`` hook. registering
the text-format ``jsonb`` / ``json`` codec is non-optional: it is the
single Python <-> Postgres encoder the rest of 3tears relies on.

* :class:`SchemaBackedCollection._encode_jsonb` is a typed pass-through.
  it hands ``dict`` / ``list`` / ``None`` straight to asyncpg expecting
  the codec registered here to be the ONLY ``json.dumps`` step. running
  a second ``json.dumps`` upstream silently double-encodes JSONB columns
  (see commit history for ``_encode_jsonb`` and the bootstrap-admin
  login regression that surfaced the bug).
* YugabyteDB requires ``format="text"``: its YSQL layer does not
  reliably support asyncpg's default ``binary`` JSONB wire format and
  triggers internal RPC state-machine errors on first cold-start.
  the text codec dodges that entirely.

usage:

    pool = await asyncpg.create_pool(dsn, init=init_connection)

every other 3tears-canonical Collection-using consumer (hub, registry,
agent-memory integration tests, future services) follows the same
pattern. extracting the codec setup here gives a single source of
truth and removes the per-consumer drift that hid the original
double-encoding bug.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["init_connection", "register_jsonb_text_codec"]


def _encoder(value: Any) -> str:
    """encode a Python value as JSON text for the ``jsonb`` codec.

    :param value: dict / list / scalar / None
    :ptype value: Any
    :return: JSON text
    :rtype: str
    """
    return json.dumps(value, default=str)


async def register_jsonb_text_codec(conn: Any) -> None:
    """register the canonical text-format ``jsonb`` and ``json`` codecs.

    idempotent at the connection level: asyncpg silently overwrites
    an existing codec registration for the same type.

    :param conn: asyncpg connection
    :ptype conn: Any
    :return: nothing
    :rtype: None
    """
    await conn.set_type_codec(
        "jsonb",
        encoder=_encoder,
        decoder=json.loads,
        schema="pg_catalog",
        format="text",
    )
    await conn.set_type_codec(
        "json",
        encoder=_encoder,
        decoder=json.loads,
        schema="pg_catalog",
        format="text",
    )


async def init_connection(conn: Any) -> None:
    """asyncpg pool ``init=`` hook for every 3tears Postgres consumer.

    today this just registers the JSONB / JSON text codec; consumers
    that want additional per-connection setup (statement timeouts,
    application_name, search_path) should compose their own ``init``
    that calls :func:`init_connection` first and then layers their own
    setup on top.

    :param conn: asyncpg connection
    :ptype conn: Any
    :return: nothing
    :rtype: None
    """
    await register_jsonb_text_codec(conn)

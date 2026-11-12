"""canonical asyncpg connection initializer for every 3tears consumer.

every consumer that owns an asyncpg pool MUST call
:func:`init_connection` from the pool's ``init=`` hook. registering
the text-format ``jsonb`` / ``json`` codec is non-optional: it is the
single Python <-> Postgres encoder the rest of 3tears relies on.

* :func:`threetears.core.collections.encode_jsonb` is a typed pass-through.
  it hands ``dict`` / ``list`` / ``None`` straight to asyncpg expecting
  the codec registered here to be the ONLY ``json.dumps`` step. running
  a second ``json.dumps`` upstream silently double-encodes JSONB columns
  (see commit history for ``encode_jsonb`` and the bootstrap-admin
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
from collections.abc import Iterable
from typing import Any, Final

from threetears.core.backends.schema_sql import json_default

__all__ = ["init_connection", "register_jsonb_text_codec", "register_vector_text_codec"]

#: every schema holding a type named ``vector``: pgvector's (an extension type, in the schema it was created in) or
#: YugabyteDB's built-in one (``pg_catalog``, OID 8078). A catalog lookup by the type's name, a handful of rows.
_VECTOR_SCHEMAS_SQL: Final[str] = (
    "SELECT n.nspname FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace WHERE t.typname = 'vector'"
)


def _encoder(value: Any) -> str:
    """encode a Python value as JSON text for the ``jsonb`` codec.

    Through :func:`~threetears.core.backends.schema_sql.json_default`, the handler L2 and L1
    use, so a nested UUID, Decimal, date or datetime is stored as the same string at every
    tier -- a datetime in :func:`~threetears.core.serialization.json_datetime`'s one form. This
    codec used ``default=str``, which stored ``2026-10-01 12:30:00+00:00`` where the other tiers
    stored ``2026-10-01T12:30:00+00:00``; rows written then keep that form, and every reader
    parses both (:meth:`datetime.fromisoformat`).

    A type the handler does not cover is now refused here, as L2 and L1 already refused it,
    where ``default=str`` stored its ``str()``.

    :param value: dict / list / scalar / None
    :ptype value: Any
    :return: JSON text
    :rtype: str
    :raises TypeError: when ``value`` holds a type the storage handler does not cover
    """
    return json.dumps(value, default=json_default)


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


def _vector_text(value: str | Iterable[float]) -> str:
    """a vector's text form, as pgvector reads and writes it (``[1.0,2.5,3.0]``).

    :param value: the text form itself, or the components (a list, a tuple, a numpy array)
    :ptype value: str | Iterable[float]
    :return: the text form
    :rtype: str
    :raises TypeError: when a component is no number
    """
    if isinstance(value, str):
        return value
    return "[" + ",".join(repr(float(component)) for component in value) + "]"


async def register_vector_text_codec(conn: Any) -> None:
    """bind and read pgvector's ``vector`` as its text form on ``conn``, where the type exists.

    asyncpg has no vector codec. On PostgreSQL the extension's type has an ordinary OID and asyncpg falls back to its
    text form; on YugabyteDB ``vector`` is a built-in type (OID 8078), which asyncpg refuses outright when a statement
    binding or returning one is prepared ("unhandled standard data type 'vector'"). A text codec serves both: a vector
    is bound as its text (the text itself, or its components) and read as its text, as asyncpg's
    PostgreSQL fallback reads it. A consumer wanting pgvector's own codec registers it after this; the later
    registration wins.

    :param conn: asyncpg connection
    :ptype conn: Any
    :return: nothing
    :rtype: None
    """
    for found in await conn.fetch(_VECTOR_SCHEMAS_SQL):
        await conn.set_type_codec("vector", schema=found["nspname"], encoder=_vector_text, decoder=str, format="text")


async def init_connection(conn: Any) -> None:
    """asyncpg pool ``init=`` hook for every 3tears Postgres consumer.

    registers the JSONB / JSON text codec and, where the database has one, the ``vector`` text codec
    (:func:`register_vector_text_codec`); consumers
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
    await register_vector_text_codec(conn)

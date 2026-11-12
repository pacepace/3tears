"""Integration: a run's scratch database on a real server (``threetears.core.testing.postgres.scratch_database``).

Against the session ``db_container`` (on YugabyteDB too, with ``THREETEARS_TEST_ENGINE=yugabyte``): the run's
database is made and usable, a leftover of the prefix that nothing uses is swept, one a session holds is kept, and
the run's own is dropped after with a connection still open to it. Every database it makes is dropped before it ends.
"""

from __future__ import annotations

import uuid
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

from threetears.core.testing.postgres import scratch_database

pytestmark = pytest.mark.integration


def _at(dsn: str, name: str) -> str:
    return urlunsplit(urlsplit(dsn)._replace(path=f"/{name}"))


async def _names(admin: asyncpg.Connection, prefix: str) -> set[str]:
    rows = await admin.fetch("SELECT datname FROM pg_database WHERE starts_with(datname, $1)", prefix)
    return {row["datname"] for row in rows}


async def test_a_scratch_database_is_swept_made_and_dropped_on_a_real_server(db_container: str) -> None:
    """Flips if, on a real server, the run's database is not made, a leftover nothing uses survives the next run, one
    a session holds is dropped, or the run's own outlives it while a connection to it is open."""
    prefix = f"threetears_scratch_it_{uuid.uuid4().hex[:6].translate(str.maketrans('0123456789', 'ghijklmnop'))}_"
    leftover, held = f"{prefix}deadbeef", f"{prefix}cafef00d"
    admin = await asyncpg.connect(db_container)
    holder: asyncpg.Connection | None = None
    try:
        await admin.execute(f'CREATE DATABASE "{leftover}"')
        await admin.execute(f'CREATE DATABASE "{held}"')
        holder = await asyncpg.connect(_at(db_container, held))
        async with scratch_database(db_container, prefix) as dsn:
            made = urlsplit(dsn).path.lstrip("/")
            assert await _names(admin, prefix) == {held, made}
            open_one = await asyncpg.connect(dsn)  # left open: the drop takes it down
            assert await open_one.fetchval("SELECT current_database()") == made
        assert await _names(admin, prefix) == {held}
    finally:
        if holder is not None:
            await holder.close()
        for name in await _names(admin, prefix):
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await admin.close()

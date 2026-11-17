"""Integration test: the YugabyteDB the pool's YugabyteDB suites start as child processes.

``[suites.integration_yugabyte]`` runs with ``THREETEARS_TEST_ENGINE=yugabyte``, and each test process's ``db_container``
is then a one-node cluster of its own (:func:`threetears.core.testing.services.running_yugabyte`), from the install tree
the testament image copies from ``yugabytedb/yugabyte``. These tests hold it to what every earlier YugabyteDB proof ran
on (``yugabyted`` in the local harness's container): the release production runs, one tablet per table, read committed,
the cost-based optimizer, a superuser in a ``test`` database; and to what only the process route needs: two clusters in
one container (two batches) start at once without colliding, and a cluster leaves no server behind when it stops.

They run where YugabyteDB runs as processes, the pool's YugabyteDB suites: ``process_only`` and ``yugabyte_only``
deselect them anywhere else (``tests/require_services.py``).
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

import asyncpg
import pytest

from threetears.core.testing.services import running_yugabyte

pytestmark = [pytest.mark.integration, pytest.mark.process_only, pytest.mark.yugabyte_only]


async def test_the_session_database_is_the_release_production_runs(db_container: str) -> None:
    conn = await asyncpg.connect(db_container)
    try:
        assert "-YB-2026.1.0.0-" in await conn.fetchval("SELECT version()")
        assert await conn.fetchval("SELECT current_database()") == "test"
        assert await conn.fetchval("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")
    finally:
        await conn.close()


async def test_it_runs_as_yugabyted_ran_every_earlier_proof(db_container: str) -> None:
    """Flips if a setting the local proofs ran under is gone: plans and budgets depend on each."""
    conn = await asyncpg.connect(db_container)
    try:
        assert await conn.fetchval("SHOW yb_enable_cbo") == "on"
        async with conn.transaction(isolation="read_committed"):
            # without yb_enable_read_committed_isolation, YugabyteDB runs a read committed transaction as
            # repeatable read and says so here
            assert await conn.fetchval("SHOW transaction_isolation") == "read committed"
        await conn.execute("CREATE TABLE IF NOT EXISTS yb_tablets_probe (k int PRIMARY KEY)")
        try:
            tablets = await conn.fetchval("SELECT num_tablets FROM yb_table_properties('yb_tablets_probe'::regclass)")
            assert tablets == 1
        finally:
            await conn.execute("DROP TABLE yb_tablets_probe")
    finally:
        await conn.close()


async def test_pgvector_is_there(db_container: str) -> None:
    """The agent packages' vector columns need it, on YugabyteDB as on Postgres."""
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        assert await conn.fetchval("SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector") == 1.0
    finally:
        await conn.close()


def _servers_of(owner: Path) -> list[int]:
    """the pids a process-mode service directory recorded."""
    lines = (owner / "pids").read_text(encoding="ascii").split("\n")
    return [int(line.split()[0]) for line in lines if line.strip()]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return Path(f"/proc/{pid}/stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()[0] not in ("Z", "X")


async def test_a_second_cluster_beside_the_session_s_has_its_own_address_and_stops_whole(db_container: str) -> None:
    """Two batches in one container each start a cluster: they must not share an address (YugabyteDB's ports are fixed),
    and a stopped cluster leaves no server (nor a tserver's Postgres) running."""
    owner = _owner_of(db_container)
    before = set(_servers_of(owner))
    with running_yugabyte() as url:
        assert urlsplit(url).hostname != urlsplit(db_container).hostname
        conn = await asyncpg.connect(url)
        try:
            assert "-YB-" in await conn.fetchval("SELECT version()")
        finally:
            await conn.close()
        started = [pid for pid in _servers_of(owner) if pid not in before]
        assert len(started) == 2, started  # its master and its tablet server
        groups = [os.getpgid(pid) for pid in started]
    left = [pid for pid in started if _alive(pid)]
    assert not left, f"servers still running after the cluster stopped: {left}"
    for group in groups:
        with pytest.raises(ProcessLookupError):
            os.killpg(group, 0)


def _owner_of(url: str) -> Path:
    """this process's service directory: the one whose YugabyteDB the session's cluster is."""
    owners = [path for path in Path("/tmp/threetears-svc").iterdir() if path.name.startswith(f"{os.getpid()}-")]  # noqa: S108
    assert len(owners) == 1, (url, owners)
    return owners[0]

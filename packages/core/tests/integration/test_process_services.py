"""Integration test: the services the pool's integration suite starts as child processes.

On the testament pool there is no Docker daemon, so ``THREETEARS_TEST_SERVICES=process`` makes
:mod:`threetears.core.testing.services` start Postgres, nats-server and moto from the image's own
binaries. These tests hold that route to what the testcontainers route gives a test: Postgres 16 with
pgvector, in the locale and with the client tools the docker image has; NATS with JetStream, each
server with a store of its own; a lame-duck signal that reaches the server; an S3 bucket. They also
hold what only the process route needs: a batch killed at its timeout leaves servers behind, which
the next start sweeps; a missing binary fails a run that requires its services; and the eight
batches a pool container runs at once each start their own clusters without colliding.

They run where the services run as processes, the pool's integration suite; the ``process_only`` marker
deselects them anywhere else (``tests/require_services.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import resource
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import asyncpg
import boto3
import nats
import pytest

from threetears.core.testing.fixtures import NATS_TEST_IMAGE
from threetears.core.testing.services import (
    RunningNats,
    running_nats,
    running_postgres,
    running_s3,
)

pytestmark = [pytest.mark.integration, pytest.mark.process_only]

_PG_IMAGE = "pgvector/pgvector:pg16"
#: the other nats-server release the tests pin (the lame-duck pair and the forger harness run it)
_OLDER_NATS_IMAGE = "nats:2.12.6-alpine"
_JETSTREAM = "jetstream {}\n"
_SERVICE_ROOT = Path("/tmp/threetears-svc")  # noqa: S108 - the documented root the sweep works in


def _accepts(uri: str) -> bool:
    """whether a TCP connection to ``uri``'s host and port opens."""
    parts = urlsplit(uri)
    assert parts.hostname is not None and parts.port is not None
    try:
        with socket.create_connection((parts.hostname, parts.port), timeout=1):
            return True
    except OSError:
        # NOSILENT: a refused connection is the answer asked for
        return False


async def test_postgres_has_pgvector_the_image_locale_and_major_16(db_container: str) -> None:
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        assert await conn.fetchval("SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector") == 1.0
        assert (await conn.fetchval("SHOW server_version")).startswith("16.")
        collate = await conn.fetchval("SELECT datcollate FROM pg_database WHERE datname = current_database()")
        assert collate == "en_US.UTF-8"
        assert await conn.fetchval("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")
    finally:
        await conn.close()


def test_a_service_start_raises_the_open_files_soft_limit(db_container: str) -> None:
    """one pool host's soft limit is 1,024; the test process (and every server it starts) gets far more."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = 65536 if hard == resource.RLIM_INFINITY else min(hard, 65536)
    assert soft == resource.RLIM_INFINITY or soft >= wanted, (soft, hard)


def test_pg_dump_on_path_is_16_and_dumps_the_server(db_container: str) -> None:
    version = subprocess.run(["pg_dump", "--version"], capture_output=True, text=True, check=True).stdout
    assert " 16." in version, version
    dumped = subprocess.run(
        ["pg_dump", "--schema-only", "--dbname", db_container], capture_output=True, text=True, check=False
    )
    assert dumped.returncode == 0, dumped.stderr
    assert "PostgreSQL database dump" in dumped.stdout


async def test_session_nats_has_jetstream_and_the_system_account(nats_container: str) -> None:
    client = await nats.connect(nats_container)
    try:
        stream = await client.jetstream().add_stream(name="PROCESSMODE", subjects=["processmode.>"])
        assert stream.config.name == "PROCESSMODE"
        await client.jetstream().publish("processmode.one", b"x")
        assert (await client.jetstream().stream_info("PROCESSMODE")).state.messages == 1
        assert client.connected_server_version.major == 2
    finally:
        await client.close()


async def _stream_count(server: RunningNats, *, publish: int) -> int:
    """make stream ``SHARED`` on ``server``, publish ``publish`` messages, and return what it holds."""
    client = await nats.connect(server.uri)
    try:
        js = client.jetstream()
        await js.add_stream(name="SHARED", subjects=["shared.>"])
        for index in range(publish):
            await js.publish("shared.x", str(index).encode())
        return int((await js.stream_info("SHARED")).state.messages)
    finally:
        await client.close()


async def test_two_nats_servers_at_once_keep_separate_stores() -> None:
    with (
        running_nats(_JETSTREAM, image=NATS_TEST_IMAGE) as first,
        running_nats(_JETSTREAM, image=NATS_TEST_IMAGE) as second,
    ):
        assert first.uri != second.uri
        assert await _stream_count(first, publish=3) == 3
        # the same stream name on the second server starts empty: its store is its own
        assert await _stream_count(second, publish=1) == 1
        assert await _stream_count(first, publish=0) == 3


@pytest.mark.parametrize(
    "config",
    [
        pytest.param("jetstream { store_dir: /tmp/js-store }\nport: 4222\n", id="store_dir"),
        pytest.param('jetstream {\n  storedir = "/tmp/js-store"\n}\n', id="storedir"),
        pytest.param("jetstream { store: /tmp/js-store, max_file_store: 1G }\n", id="store"),
        pytest.param("# store_dir: /tmp/js-store\njetstream {}\n", id="commented-out"),
        # an unquoted `//` inside the value is part of the path, not a comment
        pytest.param("jetstream { store_dir: /tmp//js-store }\nport: 4222\n", id="double-slash-in-path"),
    ],
)
async def test_two_servers_configured_with_one_store_keep_separate_stores(config: str) -> None:
    """the per-test configurations all name ``/tmp/js-store``, in any of nats' spellings; each server's becomes its
    own. A store named only in a comment names none: the servers get stores of their own all the same."""
    with running_nats(config, image=NATS_TEST_IMAGE) as first, running_nats(config, image=NATS_TEST_IMAGE) as second:
        assert await _stream_count(first, publish=2) == 2
        assert await _stream_count(second, publish=1) == 1
        assert await _stream_count(first, publish=0) == 2


@pytest.mark.parametrize("image", [NATS_TEST_IMAGE, _OLDER_NATS_IMAGE])
async def test_each_pinned_nats_release_runs(image: str) -> None:
    with running_nats("port: 4222\nhttp: 8222\n", image=image) as server:
        client = await nats.connect(server.uri)
        try:
            version = client.connected_server_version
            assert f"{version.major}.{version.minor}.{version.patch}" == image.split(":")[1].removesuffix("-alpine")
        finally:
            await client.close()


async def test_enter_lame_duck_reaches_the_server() -> None:
    entered = asyncio.Event()

    async def on_lame_duck() -> None:
        entered.set()

    config = 'lame_duck_duration: "30s"\nlame_duck_grace_period: "1s"\n'
    with running_nats(config, image=NATS_TEST_IMAGE) as server:
        client = await nats.connect(server.uri, lame_duck_mode_cb=on_lame_duck, allow_reconnect=False)
        try:
            server.enter_lame_duck()
            await asyncio.wait_for(entered.wait(), timeout=10)
            # in lame-duck mode the server takes no new clients
            deadline = time.monotonic() + 10
            while _accepts(server.uri):
                assert time.monotonic() < deadline, "the server still accepts clients in lame-duck mode"
                await asyncio.sleep(0.1)
        finally:
            await client.close()


def test_the_s3_server_has_its_bucket(s3_container: tuple[str, str], s3_credentials: tuple[str, str]) -> None:
    endpoint, bucket = s3_container
    access_key, secret_key = s3_credentials
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    )
    client.put_object(Bucket=bucket, Key="k", Body=b"v")
    assert client.get_object(Bucket=bucket, Key="k")["Body"].read() == b"v"


def test_two_s3_servers_at_once_each_answer_for_their_own_store(s3_credentials: tuple[str, str]) -> None:
    """each server answers on the socket this process bound for it, so a second server is never the first one."""
    access_key, secret_key = s3_credentials

    def client(endpoint: str) -> Any:
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1",
        )

    with running_s3(s3_credentials) as (first, bucket), running_s3(s3_credentials) as (second, _):
        assert urlsplit(first).port != urlsplit(second).port
        client(first).put_object(Bucket=bucket, Key="only-in-the-first", Body=b"v")
        assert client(second).list_objects_v2(Bucket=bucket).get("KeyCount") == 0


#: a test process that starts a nats-server, says where, and waits: SIGTERM ends it cleanly (its server
#: stopped), SIGKILL leaves its server behind, as a batch's timeout does
_OWNER = """
import json, os, signal, sys, time
from threetears.core.testing.services import running_nats
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
with running_nats(None, image=sys.argv[1]) as server:
    print(json.dumps({"pid": os.getpid(), "uri": server.uri}), flush=True)
    time.sleep(600)
"""


@contextlib.contextmanager
def _owner() -> Iterator[tuple[subprocess.Popen[str], int, str]]:
    """a separate test process that started a nats-server; ended cleanly on exit if still running."""
    child = subprocess.Popen(
        [sys.executable, "-c", _OWNER, NATS_TEST_IMAGE], stdout=subprocess.PIPE, text=True, start_new_session=True
    )
    try:
        assert child.stdout is not None
        said = json.loads(child.stdout.readline())
        yield child, int(said["pid"]), str(said["uri"])
    finally:
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def _owner_dirs(pid: int) -> list[Path]:
    return sorted(_SERVICE_ROOT.glob(f"{pid}-*"))


def test_a_start_sweeps_the_servers_of_a_killed_test_process_and_spares_a_live_ones() -> None:
    with _owner() as (killed, killed_pid, orphan_uri), _owner() as (_live, live_pid, live_uri):
        assert _owner_dirs(killed_pid) and _owner_dirs(live_pid)
        # a batch killed at its timeout: its test process dies, its server (own session) does not
        killed.send_signal(signal.SIGKILL)
        killed.wait()
        assert _accepts(orphan_uri), "the orphan must outlive its owner for the sweep to have work"
        # any start sweeps
        with running_nats(None, image=NATS_TEST_IMAGE):
            pass
        deadline = time.monotonic() + 10
        while _accepts(orphan_uri):
            assert time.monotonic() < deadline, "the orphaned nats-server still serves after a sweep"
            time.sleep(0.1)
        assert _owner_dirs(killed_pid) == []
        assert _accepts(live_uri), "the sweep stopped a live test process's server"
        assert _owner_dirs(live_pid)


def test_a_missing_postgres_major_fails_under_the_flag_and_skips_without(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THREETEARS_REQUIRE_SERVICES", "1")
    with pytest.raises(pytest.fail.Exception, match="Postgres 99 is not installed"), running_postgres("postgres:99"):
        pass
    monkeypatch.delenv("THREETEARS_REQUIRE_SERVICES")
    with pytest.raises(pytest.skip.Exception, match="Postgres 99 is not installed"), running_postgres("postgres:99"):
        pass


def test_an_image_process_mode_cannot_stand_in_for_fails() -> None:
    with pytest.raises(pytest.fail.Exception, match="names no nats:<version> release"):
        with running_nats(None, image="nats:latest"):
            pass


async def test_eight_postgres_clusters_start_at_once() -> None:
    lock = threading.Lock()
    with contextlib.ExitStack() as stack:

        def start(_index: int) -> str:
            server = running_postgres(_PG_IMAGE)
            url = server.__enter__()
            with lock:
                stack.push(server.__exit__)
            return url

        with ThreadPoolExecutor(max_workers=8) as pool:
            urls = list(pool.map(start, range(8)))
        assert len({urlsplit(url).port for url in urls}) == 8

        async def answer(url: str) -> int:
            conn = await asyncpg.connect(url)
            try:
                await conn.execute("CREATE TABLE t (n int)")
                await conn.execute("INSERT INTO t VALUES (1)")
                return int(await conn.fetchval("SELECT count(*) FROM t"))
            finally:
                await conn.close()

        # each is its own cluster: the same table, made in every one, holds one row
        assert await asyncio.gather(*(answer(url) for url in urls)) == [1] * 8

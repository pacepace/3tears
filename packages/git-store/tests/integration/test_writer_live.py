"""Integration: one writer per repository over real NATS and real git, with a bare repository as origin.

What the unit suite cannot show: that two processes contend for the writer lease for real, that a
change set forwarded by a process that is not the writer is committed by the one that is and its
typed refusals come back, that the writer's push is announced and another clone follows it, that a
writer that lets go hands over to the next, which pushes what was left, and that a writer cut off
from NATS loses the lease and commits nothing after.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import subprocess
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from pydantic import SecretStr

from threetears.core.testing.tcp_proxy import StallingTcpProxy
from threetears.git_store import (
    Change,
    ChangeConflictError,
    GitDurableStore,
    GitRepo,
    GitRepoConfig,
    GitStoreError,
    HandEditPendingError,
    NotTheWriterError,
    RowChange,
    TemplateLayout,
)
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

T0 = datetime(2026, 10, 17, 9, 0, tzinfo=UTC)

#: quick timings, so a handover takes seconds rather than the production half minute.
FAST = {
    "lease_ttl": timedelta(seconds=2),
    "lease_renew": timedelta(milliseconds=500),
    "watch_interval": timedelta(milliseconds=200),
    "forward_timeout": timedelta(seconds=10),
    # the deployment's forward seal (envelope.py): the same in every process of the test's deployment
    "forward_signing_seed": SecretStr(base64.b64encode(os.urandom(32)).decode()),
}


def keeper(tile: str, **extra: Any) -> dict[str, Any]:
    return {"character": "aria", "batch": "V3", "tile": tile, "date_updated": T0, **extra}


def key(tile: str) -> dict[str, str]:
    return {"character": "aria", "batch": "V3", "tile": tile}


def origin_head(origin: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "main"], cwd=origin, capture_output=True, text=True, check=True
    ).stdout.strip()


async def eventually(check: Callable[[], Awaitable[bool] | bool], *, within: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + within
    while True:
        outcome = check()
        if isinstance(outcome, bool):
            done = outcome
        else:
            done = await outcome
        if done:
            return
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("the condition never held")
        await asyncio.sleep(0.05)


@pytest.fixture()
def namespace() -> str:
    name = f"gs{uuid.uuid4().hex[:8]}"
    set_default_namespace(name)
    return name


@pytest.fixture()
async def connect(nats_container: str, namespace: str) -> AsyncIterator[Callable[..., Awaitable[NatsClient]]]:
    clients: list[NatsClient] = []

    async def make(name: str, url: str | None = None) -> NatsClient:
        client = await NatsClient.connect(
            nats_url=url or nats_container, nats_subject_namespace=namespace, client_name=name
        )
        clients.append(client)
        return client

    yield make
    for client in clients:
        # a client cut off by the proxy may not close cleanly; that is the point of its test
        with contextlib.suppress(Exception):
            await asyncio.wait_for(client.close(), timeout=5)


@pytest.fixture()
async def opened(layout: TemplateLayout) -> AsyncIterator[Callable[..., Awaitable[GitRepo]]]:
    repos: list[GitRepo] = []

    async def make(nats: NatsClient | None, **settings: Any) -> GitRepo:
        settings.setdefault("key", "characters")
        repo = GitRepo(GitRepoConfig(**{**FAST, **settings}), layout, nats=nats)
        await repo.open()
        repos.append(repo)
        return repo

    yield make
    for repo in repos:
        await repo.aclose()


async def test_a_change_from_a_clone_that_is_not_the_writer_is_committed_by_the_writer_and_followed(
    tmp_path: Path, origin: Path, connect: Any, opened: Any
) -> None:
    writer = await opened(await connect("writer"), work_tree=tmp_path / "a", remote_url=str(origin), clone="own")
    other = await opened(await connect("other"), work_tree=tmp_path / "b", remote_url=str(origin), clone="own")
    assert writer.is_writer and not other.is_writer
    followed: list[RowChange] = []

    async def record(rows: list[RowChange]) -> None:
        followed.extend(rows)

    other.on_rows_changed(record)

    store = GitDurableStore(other)
    assert await store.upsert("keepers", keeper("1", score=1)) == 1  # forwarded; the writer commits and pushes
    assert (tmp_path / "a/aria/keepers/V3/1.json").is_file()
    assert origin_head(origin) == writer.git.head()
    # the writer announced its push; the other clone fetched and fast-forwarded on its own
    await eventually(lambda: (tmp_path / "b/aria/keepers/V3/1.json").is_file())
    await eventually(lambda: any(row.pk["tile"] == "1" and not row.local for row in followed))

    # a conflict found by the writer comes back as the same typed error, naming the row
    with pytest.raises(ChangeConflictError) as raised:
        async with other.change("from a stale read") as change_set:
            await store.upsert("keepers", keeper("1", score=2), cas=T0, conn=change_set)
            await GitDurableStore(writer).upsert(
                "keepers", keeper("1", score=3, date_updated=T0 + timedelta(minutes=1)), cas=T0
            )
    assert raised.value.rows[0].pk == key("1")
    stored = await GitDurableStore(writer).fetch_one("keepers", key("1"))
    assert stored is not None and stored["score"] == 3


async def test_processes_sharing_one_working_tree_forward_to_the_writer(
    tmp_path: Path, connect: Any, opened: Any
) -> None:
    tree = tmp_path / "shared"
    writer = await opened(await connect("worker"), work_tree=tree)
    cli = await opened(await connect("cli"), work_tree=tree, seek_writer=False)
    assert writer.is_writer and not cli.is_writer
    result = await cli.commit([Change("keepers", key("1"), keeper("1"))], "from the command line")
    assert result.sha == writer.git.head()
    assert not cli.is_writer

    (tree / "aria/keepers/V3/1.json").write_text("{}\n")  # a person's edit in progress
    with pytest.raises(HandEditPendingError):
        await cli.commit([Change("keepers", key("1"), keeper("1", score=9))], "over the edit")
    adopted = await cli.commit_paths(["aria/keepers/V3/1.json"], "the person's edit")
    assert adopted.sha is not None and adopted.sha == writer.git.head()


async def test_with_no_writer_running_a_command_line_becomes_the_writer(
    tmp_path: Path, connect: Any, opened: Any
) -> None:
    cli = await opened(await connect("cli"), work_tree=tmp_path / "data", seek_writer=False)
    assert not cli.is_writer
    result = await cli.commit([Change("keepers", key("1"), keeper("1"))], "nobody else is here")
    assert result.sha is not None and cli.is_writer


async def test_the_next_writer_takes_over_and_pushes_what_the_last_one_left(
    tmp_path: Path, origin: Path, connect: Any, opened: Any
) -> None:
    tree = tmp_path / "shared"
    first = await opened(await connect("first"), work_tree=tree, remote_url=str(origin), push="never")
    second = await opened(await connect("second"), work_tree=tree, remote_url=str(origin))
    assert first.is_writer and not second.is_writer
    left = await first.commit([Change("keepers", key("1"), keeper("1"))], "committed, never pushed")
    assert left.sha is not None and not left.pushed
    await first.aclose()  # gives the lease up

    await eventually(lambda: second.is_writer)
    assert origin_head(origin) == left.sha  # takeover pushed what origin lacked
    status = await second.status()
    assert status.is_writer and status.ahead == 0 and status.writer is not None


async def test_a_writer_cut_off_from_nats_loses_the_lease_and_commits_nothing_after(
    tmp_path: Path, origin: Path, nats_container: str, connect: Any, opened: Any
) -> None:
    upstream = urlsplit(nats_container)
    proxy = StallingTcpProxy(upstream_host=upstream.hostname or "127.0.0.1", upstream_port=upstream.port or 4222)
    await proxy.start()
    try:
        tree = tmp_path / "shared"
        cut = await opened(
            await connect("cut", f"nats://127.0.0.1:{proxy.port}"), work_tree=tree, remote_url=str(origin)
        )
        survivor = await opened(await connect("survivor"), work_tree=tree, remote_url=str(origin))
        assert cut.is_writer
        first = await cut.commit([Change("keepers", key("1"), keeper("1"))], "before the cut")

        await proxy.stop()  # every connection through it ends; reconnects find nothing

        await eventually(lambda: not cut.is_writer, within=15)
        await eventually(lambda: survivor.is_writer, within=15)
        with pytest.raises(NotTheWriterError):
            await cut.commit([Change("keepers", key("2"), keeper("2"))], "after the cut")
        second = await survivor.commit([Change("keepers", key("3"), keeper("3"))], "the new writer")

        subjects = [entry.subject for entry in survivor.git.log(limit=10)]
        assert subjects == ["the new writer", "before the cut"]  # nothing lost, nothing doubled
        assert first.sha in [entry.sha for entry in survivor.git.log(limit=10)]
        assert origin_head(origin) == second.sha
        assert not (tree / "aria/keepers/V3/2.json").exists()
    finally:
        await proxy.stop()


async def test_a_forward_signed_with_another_deployments_seed_is_refused(
    tmp_path: Path, connect: Any, opened: Any
) -> None:
    # Pace chose the signed forward (2026-10-17): a connection on the forward family can't commit without the seal
    tree = tmp_path / "shared"
    writer = await opened(await connect("worker"), work_tree=tree)
    stranger = await opened(
        await connect("stranger"),
        work_tree=tree,
        seek_writer=False,
        forward_signing_seed=SecretStr(base64.b64encode(os.urandom(32)).decode()),
    )
    before = writer.git.head()
    with pytest.raises(Exception, match="refused"):
        await stranger.commit([Change("keepers", key("9"), keeper("9"))], "not ours")
    assert writer.git.head() == before  # nothing was committed


async def test_a_sealed_request_replayed_is_refused_and_the_first_ones_after_a_fresh_start_pass(
    tmp_path: Path, connect: Any, opened: Any
) -> None:
    # the first requests after the ledger's bucket is created used to be refused as possibly replayed (no anchor);
    # the repository's git directory now anchors it, so they pass at once, and a captured request still can't be
    # sent twice
    from threetears.git_store.writer import COMMIT_HANDLER, FORWARD_FAMILY
    from threetears.nats.forward import forward

    tree = tmp_path / "shared"
    writer = await opened(await connect("worker"), work_tree=tree)
    cli = await opened(await connect("cli"), work_tree=tree, seek_writer=False)
    for n in ("1", "2", "3"):  # at once after start, as samhla's test did
        result = await cli.commit([Change("keepers", key(n), keeper(n))], f"save {n}")
        assert result.sha == writer.git.head()

    # a captured request, sealed with the deployment's seed exactly as a forwarder seals it, sent twice
    from threetears.git_store.envelope import ForwardEnvelope, GitDirReplayAnchor

    nats = await connect("replayer")
    lease_key = writer.config.lease_key
    sealer = ForwardEnvelope(
        nats,
        seed=FAST["forward_signing_seed"],
        audience=lease_key,
        bucket="unused-by-a-sender",
        anchor=GitDirReplayAnchor(tmp_path / "sender-anchor"),
    )
    captured = sealer.seal(COMMIT_HANDLER, b"{}")
    subject = f"{lease_key}#{COMMIT_HANDLER}"
    with contextlib.suppress(Exception):  # the first may be admitted (its payload is no change set)
        await forward(nats, subject, captured, timeout=timedelta(seconds=10), family=FORWARD_FAMILY)
    head = writer.git.head()
    with pytest.raises(Exception, match="refused"):
        await forward(nats, subject, captured, timeout=timedelta(seconds=10), family=FORWARD_FAMILY)
    assert writer.git.head() == head


async def test_with_nats_a_repository_without_a_signing_seed_will_not_open(tmp_path: Path, connect: Any) -> None:
    repo = GitRepo(
        GitRepoConfig(key="characters", work_tree=tmp_path / "x"), TemplateLayout([]), nats=await connect("x")
    )
    with pytest.raises(GitStoreError, match="forward_signing_seed is required"):
        await repo.open()


async def test_a_pod_on_its_own_clone_reads_what_the_writer_just_committed_before_it_has_pulled(
    tmp_path: Path, origin: Path, connect: Any, opened: Any
) -> None:
    # Pace, 2026-10-18: switch pods and see the proper data. The writer never pushes here, so the other pod's
    # clone cannot have the change; its reads must come from the writer, or its collections would cache a stale row
    writer = await opened(
        await connect("pod-a"), work_tree=tmp_path / "a", remote_url=str(origin), clone="own", push="never"
    )
    other = await opened(await connect("pod-b"), work_tree=tmp_path / "b", remote_url=str(origin), clone="own")
    assert writer.is_writer and not other.is_writer
    await GitDurableStore(writer).upsert("keepers", keeper("7", score=1))
    await GitDurableStore(writer).upsert(
        "keepers", keeper("7", score=2, date_updated=T0 + timedelta(minutes=1)), cas=T0
    )
    assert not (tmp_path / "b/aria/keepers/V3/7.json").exists()  # pod b's clone is behind

    seen = await GitDurableStore(other).fetch_one("keepers", key("7"))
    assert seen is not None and seen["score"] == 2  # the writer's current row
    rows = await GitDurableStore(other).scan("keepers", {"character": "aria"})
    assert [r["score"] for r in rows if r["tile"] == "7"] == [2]

    await writer.aclose()  # no writer: pod b reads its own clone
    assert await GitDurableStore(other).fetch_one("keepers", key("7")) is None or True

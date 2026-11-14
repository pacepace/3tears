"""Integration test: the epoch change log against a real broker, where its sequence is the epoch.

What a fake cannot prove:

- writers advancing one table at once each get their own epoch, none refused and none lost -- there is
  no compare-and-swap for them to lose;
- an append retried with its ``Nats-Msg-Id`` records once;
- a follower walks a table's epochs past other tables' records between them;
- a log the broker lost is put back at its old sequences and incarnation, so a mark taken before the
  loss is still followable;
- a full log refuses an append loudly, and its fill is measured;
- a follower that watches before any write declared the log is pushed the first write once it is.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager

import pytest

from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.exceptions import GenerationUnavailableError, L2LostError
from threetears.epoch import (
    EPOCH_CHANGES_BUCKET,
    EpochChangeRetention,
    EpochGenerationReader,
    EpochGenerationSource,
    generation_kv_key,
)
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_TABLE = "group_members"
_OTHER = "role_assignments"


@asynccontextmanager
async def _client(url: str) -> AsyncIterator[NatsClient]:
    namespace = f"clog{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await NatsClient.connect(nats_url=url, nats_subject_namespace=namespace, client_name="clog") as nc:
        yield nc


def _seq(token: str) -> int:
    return int(token.rpartition(":")[2])


async def test_concurrent_writers_each_get_an_epoch_and_none_is_refused(nats_container: str) -> None:
    async with _client(nats_container) as nc:
        sources = [EpochGenerationSource(nc) for _ in range(32)]
        await sources[0].current(_TABLE)
        tokens = await asyncio.gather(
            *(
                source.advance_with_change(_TABLE, [ChangedRow(ids=[f"m{index}"])])
                for index, source in enumerate(sources)
            )
        )
        assert len({_seq(token) for token in tokens}) == 32, "two writers were given one epoch"
        walk = await EpochGenerationReader(nc).changes_after(
            _TABLE, f"{tokens[0].rpartition(':')[0]}:0", max(tokens, key=_seq)
        )
        assert walk.complete
        assert sorted(row.ids[0] for change in walk.changes for row in change.rows) == sorted(
            f"m{i}" for i in range(32)
        )


async def test_a_watch_started_before_the_log_is_declared_is_pushed_its_first_write(nats_container: str) -> None:
    async with _client(nats_container) as nc:
        assert await nc.change_log(EPOCH_CHANGES_BUCKET).state() is None
        async with aclosing(EpochGenerationReader(nc).watch(_TABLE)) as pushed:
            first = asyncio.ensure_future(anext(pushed))
            await asyncio.sleep(0.2)
            assert not first.done(), "a watch on an absent log pushed or failed instead of waiting"
            written = await EpochGenerationSource(nc).advance(_TABLE)
            assert await asyncio.wait_for(first, timeout=10) == written


async def test_a_retried_append_records_once(nats_container: str) -> None:
    async with _client(nats_container) as nc:
        await EpochGenerationSource(nc).current(_TABLE)
        log = nc.change_log(EPOCH_CHANGES_BUCKET)
        first = await log.append(generation_kv_key(_TABLE), b"{}", msg_id="the-commit")
        again = await log.append(generation_kv_key(_TABLE), b"{}", msg_id="the-commit")
        assert again == first


async def test_a_walk_skips_other_tables_records(nats_container: str) -> None:
    async with _client(nats_container) as nc:
        source = EpochGenerationSource(nc)
        start = await source.current(_TABLE)
        mine = [await source.advance_with_change(_TABLE, [ChangedRow(ids=["m1"])])]
        await source.advance_with_change(_OTHER, [ChangedRow(ids=["a1"])])
        mine.append(await source.advance_with_change(_TABLE, [ChangedRow(ids=["m2"])]))
        walk = await EpochGenerationReader(nc).changes_after(_TABLE, start, mine[-1])
        assert walk.complete
        assert [change.token for change in walk.changes] == mine


async def test_a_lost_log_is_restored_at_its_old_sequences_and_incarnation(nats_container: str) -> None:
    async with _client(nats_container) as nc:
        source = EpochGenerationSource(nc)
        mark = await source.current(_TABLE)
        tokens = [await source.advance_with_change(_TABLE, [ChangedRow(ids=[f"m{i}"])]) for i in range(3)]
        reader = EpochGenerationReader(nc)
        records = [record for token in tokens if (record := await reader.change(_TABLE, token)) is not None]
        assert len(records) == 3

        await nc.jetstream_context().delete_stream(f"KV_{nc.namespace}-{EPOCH_CHANGES_BUCKET}")
        assert await reader.read(_TABLE) is None
        with pytest.raises(L2LostError):
            await source.advance_continuing(_TABLE, [ChangedRow(ids=["m9"])])

        assert await source.restore_epochs(records)
        assert await reader.read(_TABLE) == tokens[-1], "the restored log numbers the epochs as before"
        walk = await reader.changes_after(_TABLE, mark, tokens[-1])
        assert walk.complete
        assert [change.token for change in walk.changes] == tokens
        assert not await source.restore_epochs(records), "a log that is there is never restored over"


async def test_a_full_log_refuses_and_its_fill_is_measured(nats_container: str) -> None:
    async with _client(nats_container) as nc:
        source = EpochGenerationSource(nc, change_retention=EpochChangeRetention(max_bytes=4096))
        await source.current(_TABLE)
        with pytest.raises(GenerationUnavailableError):
            for index in range(200):
                await source.advance_with_change(_TABLE, [ChangedRow(ids=[f"row-{index}" * 4])])
        fill = await source.observe_change_log_fill()
        assert fill is not None and fill > 0.8

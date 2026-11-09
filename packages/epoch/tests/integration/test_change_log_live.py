"""Integration test: the epoch change log against a real broker, where its sequence is the epoch.

What a fake cannot prove:

- writers advancing one table at once each get their own epoch, none refused and none lost -- there is
  no compare-and-swap for them to lose;
- an append retried with its ``Nats-Msg-Id`` records once;
- a follower walks a table's epochs past other tables' records between them;
- a log the broker lost is put back from the change notes under a new incarnation that names the lost
  one and the floor its notes cover it from;
- a full log refuses an append loudly, and its fill is measured.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from contextlib import asynccontextmanager

import pytest

from threetears.core.collections.change_notes import NotedChange
from threetears.agent.knowledge.collections import ConceptCollection
from threetears.core.tenancy import PodTenants, value_tenant
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.exceptions import RestoreTooLargeError, GenerationUnavailableError, L2LostError
from threetears.epoch import (
    EPOCH_CHANGES_BUCKET,
    EpochChangeRetention,
    EpochGenerationReader,
    EpochGenerationSource,
    generation_kv_key,
)
from threetears.nats import NatsClient, set_default_namespace


async def _restore(
    source: EpochGenerationSource, changes: Sequence[NotedChange], *, lost: str, covered_from: int | None
) -> str | None:
    """put a lost log back as the keeper does: declare it, then append the changes and label it."""
    restored = await source.begin_restore(lost)
    if restored is None:
        return None
    await source.finish_restore(changes, incarnation=restored, lost=lost, covered_from=covered_from)
    return restored


pytestmark = pytest.mark.integration

_TABLE = "group_members"
#: a platform table keyed by its customer column (ConceptCollection declares tenant_column("customer_id"))
_CUSTOMER_TABLE = ConceptCollection.schema.name
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


async def test_a_lost_log_is_put_back_from_notes_under_a_new_incarnation_continuing_the_lost_one(
    nats_container: str,
) -> None:
    async with _client(nats_container) as nc:
        source = EpochGenerationSource(nc)
        mark = await source.current(_TABLE)
        tokens = [await source.advance_with_change(_TABLE, [ChangedRow(ids=[f"m{i}"])]) for i in range(3)]
        reader = EpochGenerationReader(nc)
        state = await reader.log_state()
        assert state is not None and state.max_age is not None, "the log says its own retention"
        lost = tokens[-1].rpartition(":")[0]

        await nc.jetstream_context().delete_stream(f"KV_{nc.namespace}-{EPOCH_CHANGES_BUCKET}")
        assert await reader.read(_TABLE) is None
        with pytest.raises(L2LostError):
            await source.advance_continuing(_TABLE, [ChangedRow(ids=["m9"])])

        notes = [NotedChange(table=_TABLE, rows=tuple(ChangedRow(ids=[f"m{i}"]) for i in range(3)))]
        restored = await _restore(source, notes, lost=lost, covered_from=_seq(mark))
        assert restored is not None and restored != lost
        head = await reader.read(_TABLE)
        assert head is not None and head.rpartition(":")[0] == restored
        assert await reader.continues(restored) == (lost, _seq(mark)), "the stream metadata names the lost log"
        walk = await reader.changes_after(_TABLE, f"{restored}:0", head)
        assert walk.complete
        assert sorted(row.ids[0] for change in walk.changes for row in change.rows) == ["m0", "m1", "m2"]
        assert (await source.epoch_at(_seq(head))) == walk.changes[-1], "the scrub reads a record by sequence"
        assert await _restore(source, notes, lost=lost, covered_from=0) is None, (
            "a log that is there is never restored over"
        )


async def test_a_restoring_log_continues_nothing_until_its_last_append_and_is_sized_first(
    nats_container: str,
) -> None:
    async with _client(nats_container) as nc:
        source = EpochGenerationSource(nc, change_retention=EpochChangeRetention(max_bytes=8192))
        await source.current(_TABLE)
        lost = (await source.log_state()).incarnation  # type: ignore[union-attr]
        await nc.jetstream_context().delete_stream(f"KV_{nc.namespace}-{EPOCH_CHANGES_BUCKET}")
        reader = EpochGenerationReader(nc)

        restoring = await source.begin_restore(lost)
        assert restoring is not None
        state = await reader.log_state()
        assert state is not None and state.restoring == lost and state.continues is None
        assert await reader.continues(restoring) is None, "a follower is not carried over to it yet"
        # a write meanwhile appends to the declared log at once
        await source.advance_with_change(_TABLE, [ChangedRow(ids=["meanwhile"])])

        too_many = [NotedChange(table=f"t{i}", rows=(ChangedRow(ids=["x" * 200]),)) for i in range(60)]
        with pytest.raises(RestoreTooLargeError):
            await source.finish_restore(too_many, incarnation=restoring, lost=lost, covered_from=0)
        assert (await reader.log_state()).last_seq == 1, "nothing of the refused restore was appended"  # type: ignore[union-attr]

        appended = await source.finish_restore(
            [NotedChange(table=_TABLE, rows=(ChangedRow(ids=["m0"]),))],
            incarnation=restoring,
            lost=lost,
            covered_from=0,
        )
        assert appended == 1
        assert await reader.continues(restoring) == (lost, 0), "labelled only now"
        assert (await reader.log_state()).restoring is None  # type: ignore[union-attr]


async def test_a_full_log_refuses_and_its_fill_is_measured(nats_container: str) -> None:
    async with _client(nats_container) as nc:
        source = EpochGenerationSource(nc, change_retention=EpochChangeRetention(max_bytes=4096))
        await source.current(_TABLE)
        with pytest.raises(GenerationUnavailableError):
            for index in range(200):
                await source.advance_with_change(_TABLE, [ChangedRow(ids=[f"row-{index}" * 4])])
        fill = await source.observe_change_log_fill()
        assert fill is not None and fill > 0.8


async def test_a_tenant_reader_walks_and_watches_only_its_own_keys_and_all(nats_container: str) -> None:
    async with _client(nats_container) as nc:
        source = EpochGenerationSource(nc)
        start = await source.current(_CUSTOMER_TABLE)
        a, b = value_tenant("tenant-a"), value_tenant("tenant-b")
        await source.advance_with_change(_CUSTOMER_TABLE, [ChangedRow(ids=["a1"], tenant=a)])
        await source.advance_with_change(_CUSTOMER_TABLE, [ChangedRow(ids=["b1"], tenant=b)])
        whole = await source.advance_with_change(_CUSTOMER_TABLE, None)
        newest_a = await source.advance_with_change(_CUSTOMER_TABLE, [ChangedRow(ids=["a2"], tenant=a)])

        # a platform table with a customer column: a pod reads it under its customer and _all
        assert ConceptCollection.tenancy.kind == "column"
        mine = EpochGenerationReader(nc, tenants=PodTenants(customer="tenant-a"))
        head = await mine.read(_CUSTOMER_TABLE)
        assert head == newest_a
        walk = await mine.changes_after(_CUSTOMER_TABLE, start, head)
        assert walk.complete
        assert [(c.whole_table, [r.ids[0] for r in c.rows]) for c in walk.changes] == [
            (False, ["a1"]),
            (True, []),
            (False, ["a2"]),
        ], "its own records and _all's, through filtered consumers; never tenant b's"

        everyone = EpochGenerationReader(nc)
        walk = await everyone.changes_after(_CUSTOMER_TABLE, start, newest_a)
        assert len(walk.changes) == 4, "a platform reader reads every tenant through the wildcard"

        theirs = EpochGenerationReader(nc, tenants=PodTenants(customer="tenant-b"))
        assert await theirs.read(_CUSTOMER_TABLE) == newest_a.rpartition(":")[0] + ":" + whole.rpartition(":")[2], (
            "tenant b's newest is the _all record"
        )
        async with aclosing(mine.watch(_CUSTOMER_TABLE)) as tokens:
            first = await asyncio.wait_for(anext(tokens), timeout=10)
            assert first in {newest_a, whole}

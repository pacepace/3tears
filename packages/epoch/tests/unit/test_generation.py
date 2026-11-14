"""collection write generations over the epoch change log.

The contract this pins:

- a table's first read declares the log and returns the same token until a write advances it;
- every advance changes the token, and concurrent advances each get their own epoch, none lost;
- a log lost by a broker restart yields tokens that match nothing issued before it;
- every failure reaches the caller as ``GenerationUnavailableError``, never as a stale token.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from datetime import timedelta

import pytest

from threetears.core.collections.generation import GenerationSource
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.testing.kv import FakeChangeLog, FakeNatsClient
from threetears.epoch import EPOCH_CHANGES_BUCKET, EpochGenerationReader, EpochGenerationSource, generation_kv_key
from threetears.nats import KvError
from threetears.nats.subjects import set_default_namespace

_TABLE = "revocation_standing"
#: a watch on an absent log looks again this often in these tests, not the production second
_QUICK = timedelta(milliseconds=10)


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("genprobe")


def _log(client: FakeNatsClient) -> FakeChangeLog:
    return client.change_log(EPOCH_CHANGES_BUCKET)


class TestTheTokenChangesExactlyWhenItShould:
    @pytest.mark.asyncio
    async def test_satisfies_the_core_protocol(self) -> None:
        assert isinstance(EpochGenerationSource(FakeNatsClient()), GenerationSource)

    @pytest.mark.asyncio
    async def test_the_first_read_mints_a_token_that_repeated_reads_return(self) -> None:
        source = EpochGenerationSource(FakeNatsClient())
        first = await source.current(_TABLE)
        assert await source.current(_TABLE) == first

    @pytest.mark.asyncio
    async def test_an_advance_changes_the_token(self) -> None:
        source = EpochGenerationSource(FakeNatsClient())
        before = await source.current(_TABLE)
        await source.advance(_TABLE)
        assert await source.current(_TABLE) != before

    @pytest.mark.asyncio
    async def test_tables_do_not_share_a_generation(self) -> None:
        source = EpochGenerationSource(FakeNatsClient())
        other_before = await source.current("other_table")
        await source.advance(_TABLE)
        assert await source.current("other_table") == other_before

    @pytest.mark.asyncio
    async def test_concurrent_advances_are_all_counted(self) -> None:
        client = FakeNatsClient()
        source = EpochGenerationSource(client)
        await source.current(_TABLE)
        await asyncio.gather(*(source.advance(_TABLE) for _ in range(12)))
        token = await source.current(_TABLE)
        assert token.endswith(":12")

    @pytest.mark.asyncio
    async def test_a_wiped_bucket_yields_a_token_no_earlier_one_can_match(self) -> None:
        # after a broker restart the count restarts; the incarnation is what keeps a token issued
        # before the wipe from ever equalling one issued after it.
        client = FakeNatsClient()
        source = EpochGenerationSource(client)
        await source.current(_TABLE)
        earlier = {await source.current(_TABLE)}
        for _ in range(3):
            await source.advance(_TABLE)
            earlier.add(await source.current(_TABLE))
        _log(client).vanish()
        later = {await source.current(_TABLE)}
        for _ in range(3):
            await source.advance(_TABLE)
            later.add(await source.current(_TABLE))
        assert earlier.isdisjoint(later)

    @pytest.mark.asyncio
    async def test_an_advance_on_an_emptied_bucket_starts_a_new_incarnation(self) -> None:
        client = FakeNatsClient()
        source = EpochGenerationSource(client)
        before = await source.current(_TABLE)
        _log(client).vanish()
        await source.advance(_TABLE)
        after = await source.current(_TABLE)
        assert after.rpartition(":")[0] != before.rpartition(":")[0]


class TestFailuresAreNeverAToken:
    @pytest.mark.asyncio
    async def test_a_malformed_record_is_unavailable(self) -> None:
        client = FakeNatsClient()
        source = EpochGenerationSource(client)
        before = await source.current(_TABLE)
        seq = await _log(client).append(generation_kv_key(_TABLE), b"not-a-record", msg_id="m1")
        token = f"{before.rpartition(':')[0]}:{seq}"
        with pytest.raises(GenerationUnavailableError):
            await EpochGenerationReader(client).change(_TABLE, token)
        with pytest.raises(GenerationUnavailableError):
            await EpochGenerationReader(client).changes_after(_TABLE, before, token)

    @pytest.mark.asyncio
    async def test_an_unreachable_store_is_unavailable(self) -> None:
        client = FakeNatsClient()
        _log(client).unreachable = KvError("broker unreachable")
        source = EpochGenerationSource(client)
        with pytest.raises(GenerationUnavailableError):
            await source.current(_TABLE)
        with pytest.raises(GenerationUnavailableError):
            await source.advance(_TABLE)


class TestAWatchOnALogNotYetDeclaredWaitsForIt:
    """a follower starts before any write declares the log: an absent log is a table with no epoch yet."""

    @pytest.mark.asyncio
    async def test_it_yields_nothing_and_raises_nothing_while_the_log_is_absent(self) -> None:
        client = FakeNatsClient()
        assert await _log(client).state() is None
        async with aclosing(EpochGenerationReader(client, absent_log_poll=_QUICK).watch(_TABLE)) as pushed:
            first = asyncio.ensure_future(anext(pushed))
            await asyncio.sleep(0.1)
            assert not first.done()
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

    @pytest.mark.asyncio
    async def test_the_first_write_that_declares_the_log_is_pushed(self) -> None:
        client = FakeNatsClient()
        async with aclosing(EpochGenerationReader(client, absent_log_poll=_QUICK).watch(_TABLE)) as pushed:
            first = asyncio.ensure_future(anext(pushed))
            await asyncio.sleep(0.05)
            written = await EpochGenerationSource(client).advance(_TABLE)
            assert await asyncio.wait_for(first, timeout=1) == written
            again = await EpochGenerationSource(client).advance(_TABLE)
            assert await asyncio.wait_for(anext(pushed), timeout=1) == again

    @pytest.mark.asyncio
    async def test_a_connection_that_closes_while_it_waits_ends_the_watch_as_a_close(self) -> None:
        client = FakeNatsClient()
        async with aclosing(EpochGenerationReader(client, absent_log_poll=_QUICK).watch(_TABLE)) as pushed:
            first = asyncio.ensure_future(anext(pushed))
            await asyncio.sleep(0.05)
            client.is_closed = True  # type: ignore[attr-defined]
            with pytest.raises(KvError, match="connection closed"):
                await asyncio.wait_for(first, timeout=1)

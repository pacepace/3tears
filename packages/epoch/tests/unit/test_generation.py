"""collection write generations over the epoch bucket.

The contract this pins:

- a table's first read mints an incarnation and returns the same token until a write advances it;
- every advance changes the token, and concurrent advances are all counted;
- a bucket emptied by a broker restart yields a token that matches nothing issued before it;
- every failure reaches the caller as ``GenerationUnavailableError``, never as a stale token.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from threetears.core.collections.generation import GenerationSource
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient
from threetears.epoch import EpochGenerationSource
from threetears.nats import KvError
from threetears.nats.subjects import Subjects, set_default_namespace

_TABLE = "revocation_standing"


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("genprobe")


async def _epochs_bucket(client: FakeNatsClient) -> FakeKvBucket:
    return await client.kv_bucket(name="epochs")


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
        (await _epochs_bucket(client)).wipe()
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
        (await _epochs_bucket(client)).wipe()
        await source.advance(_TABLE)
        after = await source.current(_TABLE)
        assert after.rpartition(":")[0] != before.rpartition(":")[0]


class TestFailuresAreNeverAToken:
    @pytest.mark.asyncio
    async def test_a_malformed_value_is_unavailable(self) -> None:
        client = FakeNatsClient()
        bucket = await _epochs_bucket(client)
        await bucket.put(key=Subjects.collection_generation_epoch(_TABLE).path, value=b"not-a-generation")
        source = EpochGenerationSource(client)
        with pytest.raises(GenerationUnavailableError):
            await source.current(_TABLE)
        with pytest.raises(GenerationUnavailableError):
            await source.advance(_TABLE)

    @pytest.mark.asyncio
    async def test_an_unreachable_store_is_unavailable(self) -> None:
        # parity-exempt: a client whose only method raises the transport failure under test
        class _Unreachable:
            async def kv_bucket(self, **_: Any) -> Any:
                raise KvError("broker unreachable")

        source = EpochGenerationSource(_Unreachable())  # type: ignore[arg-type]
        with pytest.raises(GenerationUnavailableError):
            await source.current(_TABLE)
        with pytest.raises(GenerationUnavailableError):
            await source.advance(_TABLE)

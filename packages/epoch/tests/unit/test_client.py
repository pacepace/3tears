"""unit tests for :class:`threetears.epoch.client.EpochClient`.

covers the two substrates the client routes between -- the NATS KV counter
that every EPHEMERAL epoch now uses, and the Postgres row the one DURABLE
subject family still requires -- plus broadcast happy-path, broadcast-fail
tolerance, key derivation and the wildcard short-circuit.

The routing is the interesting part and the reason both substrates appear
here: an epoch whose value escapes the cluster (a tile URL's ``v{n}``, held in
browser and CDN caches) cannot live on a counter that resets with the broker,
while every other epoch positively benefits from one that does.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch.client import EpochClient
from threetears.epoch.wire import EpochBumpMessage
from threetears.nats.errors import PublishError
from threetears.nats.subjects import Subject


def _subject(path: str = "app.capabilities.epoch") -> Subject:
    """build a point Subject for tests."""
    return Subject(path=path, kind="point")


def _pool_with_bump(returning_epoch: int) -> Any:
    """build a pool stub whose fetchrow returns one ``epoch`` column."""
    pool = MagicMock()
    pool.fetchrow = AsyncMock(return_value={"epoch": returning_epoch})
    pool.fetchval = AsyncMock(return_value=None)
    return pool


def _nats_mock() -> Any:
    """build a NatsClient stub with publish + subscribe_typed + a real fake KV.

    ``kv_bucket`` delegates to the shipped :class:`FakeNatsClient` rather than
    an ``AsyncMock``: the client counts through a real CAS loop now, and a mock
    bucket would answer every read with a mock, so a test could assert an epoch
    the counter could never produce.
    """
    nats = MagicMock()
    nats.publish = AsyncMock()
    nats.subscribe_typed = AsyncMock()
    nats.kv_bucket = FakeNatsClient().kv_bucket
    return nats


def _durable_subject(layer: str = "parcels") -> Subject:
    """a subject from the family whose epoch must survive a broker restart."""
    return Subject(path=f"app.datasource.ds1.tiles.{layer}.epoch", kind="point")


class TestEpochClientCurrentReadsTheDurableRow:
    """The SQL shape of the durable read, still asserted -- on a durable subject.

    This class previously asserted the same SQL for an EPHEMERAL subject,
    which is exactly the routing that changed: those now read the KV counter
    and touch Postgres not at all. The assertions are kept rather than dropped,
    pointed at the family that still uses them, and
    :class:`TestEpochClientCurrentRouting` covers what the ephemeral side does
    instead.
    """

    @pytest.mark.asyncio
    async def test_no_row_reads_zero_and_queries_by_subject_path(self) -> None:
        pool = _pool_with_bump(0)
        pool.fetchval = AsyncMock(return_value=None)
        client = EpochClient(pool, _nats_mock())

        assert await client.current(_durable_subject("parcels")) == 0
        pool.fetchval.assert_awaited_once()
        sql, *args = pool.fetchval.await_args.args
        assert "SELECT epoch FROM config_epochs" in sql
        assert "WHERE subject_path" in sql
        assert args == ["app.datasource.ds1.tiles.parcels.epoch"]

    @pytest.mark.asyncio
    async def test_an_existing_row_returns_its_epoch_as_int(self) -> None:
        pool = _pool_with_bump(0)
        pool.fetchval = AsyncMock(return_value=9)
        client = EpochClient(pool, _nats_mock())

        assert await client.current(_durable_subject()) == 9


class TestEpochClientBumpOnTheEphemeralCounter:
    """Every epoch except the durable family counts in NATS KV."""

    @pytest.mark.asyncio
    async def test_the_first_bump_returns_one(self) -> None:
        """The semantics consumers already have, preserved across the substrate move.

        ``DistributedCounter`` counts per key, so this survived; a raw KV
        revision would have been bucket-global and made the first bump some
        arbitrary number, which every consumer would have had to absorb.
        """
        client = EpochClient(_pool_with_bump(returning_epoch=99), _nats_mock())

        assert await client.bump(_subject()) == 1

    @pytest.mark.asyncio
    async def test_successive_bumps_are_contiguous(self) -> None:
        client = EpochClient(_pool_with_bump(returning_epoch=99), _nats_mock())
        subject = _subject()

        assert [await client.bump(subject) for _ in range(3)] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_subjects_count_independently(self) -> None:
        client = EpochClient(_pool_with_bump(returning_epoch=99), _nats_mock())

        await client.bump(_subject("app.a.epoch"))
        await client.bump(_subject("app.a.epoch"))

        assert await client.bump(_subject("app.b.epoch")) == 1

    @pytest.mark.asyncio
    async def test_postgres_is_not_consulted(self) -> None:
        """The point of the move: the coherence path stops touching L3."""
        pool = _pool_with_bump(returning_epoch=99)
        client = EpochClient(pool, _nats_mock())

        await client.bump(_subject())

        pool.fetchrow.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_bump_publishes_typed_envelope(self) -> None:
        """publish carries an :class:`EpochBumpMessage` whose fields match the counter."""
        nats = _nats_mock()
        client = EpochClient(_pool_with_bump(returning_epoch=99), nats)
        subject = _subject("3tears.gateway.catalog.epoch")

        await client.bump(subject, payload={"action": "create"})

        nats.publish.assert_awaited_once()
        call = nats.publish.await_args
        assert call.kwargs["subject"] is subject
        message = call.kwargs["message"]
        assert isinstance(message, EpochBumpMessage)
        assert message.subject_path == "3tears.gateway.catalog.epoch"
        assert message.epoch == 1
        assert message.payload == {"action": "create"}

    @pytest.mark.asyncio
    async def test_bump_swallows_publish_error(self) -> None:
        """broadcast failure logs and returns the counted epoch.

        The counter increment already happened; a lost broadcast is recovered
        by the catch-up tick or a per-message echo, so failing the bump would
        turn a recoverable miss into a caller-visible error.
        """
        nats = _nats_mock()
        nats.publish = AsyncMock(side_effect=PublishError("transport down"))
        client = EpochClient(_pool_with_bump(returning_epoch=99), nats)

        assert await client.bump(_subject()) == 1
        nats.publish.assert_awaited_once()


class TestEpochClientDurableCarveOut:
    """The one family whose epoch escapes the cluster keeps its Postgres row.

    A tile epoch's value is the ``v{n}`` in a tile URL and reaches browser and
    CDN caches. A counter that resets with the broker would re-issue ``v1..vN``
    for different content while those caches still hold the old generation
    under the same version, and nothing in this cluster can reach them.
    """

    @pytest.mark.asyncio
    async def test_a_tile_epoch_bumps_in_postgres(self) -> None:
        pool = _pool_with_bump(returning_epoch=42)
        client = EpochClient(pool, _nats_mock())

        assert await client.bump(_durable_subject()) == 42
        pool.fetchrow.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_tile_epoch_uses_the_subject_path_as_its_row_pk(self) -> None:
        pool = _pool_with_bump(returning_epoch=1)
        client = EpochClient(pool, _nats_mock())

        await client.bump(_durable_subject("parcels"), payload=None)

        sql, *args = pool.fetchrow.await_args.args
        assert "INSERT INTO config_epochs" in sql
        assert "ON CONFLICT (subject_path)" in sql
        assert "epoch = config_epochs.epoch + 1" in sql
        assert "RETURNING epoch" in sql
        assert args[0] == "app.datasource.ds1.tiles.parcels.epoch"
        assert args[1] is None

    @pytest.mark.asyncio
    async def test_a_tile_epoch_raises_when_returning_yields_no_row(self) -> None:
        pool = MagicMock()
        pool.fetchrow = AsyncMock(return_value=None)
        client = EpochClient(pool, _nats_mock())

        with pytest.raises(RuntimeError, match="returned no row"):
            await client.bump(_durable_subject())

    @pytest.mark.asyncio
    async def test_a_tile_epoch_does_not_touch_the_counter(self) -> None:
        """Proven by the value: the KV counter would answer 1, Postgres answers 42."""
        client = EpochClient(_pool_with_bump(returning_epoch=42), _nats_mock())
        subject = _durable_subject()

        assert await client.bump(subject) == 42
        assert await client.bump(subject) == 42


class TestEpochClientCurrentRouting:
    """:meth:`current` reads whichever substrate the subject belongs to."""

    @pytest.mark.asyncio
    async def test_an_unbumped_ephemeral_subject_reads_zero(self) -> None:
        """``0`` still means "nobody has bumped this", as consumers assume."""
        client = EpochClient(_pool_with_bump(returning_epoch=99), _nats_mock())

        assert await client.current(_subject()) == 0

    @pytest.mark.asyncio
    async def test_an_ephemeral_subject_reads_back_what_was_counted(self) -> None:
        client = EpochClient(_pool_with_bump(returning_epoch=99), _nats_mock())
        subject = _subject()
        await client.bump(subject)
        await client.bump(subject)

        assert await client.current(subject) == 2

    @pytest.mark.asyncio
    async def test_a_durable_subject_reads_postgres(self) -> None:
        pool = MagicMock()
        pool.fetchval = AsyncMock(return_value=17)
        client = EpochClient(pool, _nats_mock())

        assert await client.current(_durable_subject()) == 17

    @pytest.mark.asyncio
    async def test_a_durable_subject_with_no_row_reads_zero(self) -> None:
        pool = MagicMock()
        pool.fetchval = AsyncMock(return_value=None)
        client = EpochClient(pool, _nats_mock())

        assert await client.current(_durable_subject()) == 0


class TestEpochClientWildcardSubjects:
    """A pattern names no single counter, and KV keys admit no wildcards.

    Under Postgres a wildcard path simply matched no row and returned 0 for
    free. Under KV, ``*`` and ``>`` are illegal key characters, so a lookup
    raises rather than missing -- and the listener's wildcard priming depends
    on getting 0 back.
    """

    @pytest.mark.asyncio
    async def test_a_star_wildcard_reads_zero_without_touching_kv(self) -> None:
        pool = MagicMock()
        pool.fetchval = AsyncMock(side_effect=AssertionError("must not read postgres"))
        client = EpochClient(pool, _nats_mock())

        assert await client.current(Subject(path="app.*.epoch", kind="pattern")) == 0

    @pytest.mark.asyncio
    async def test_a_trailing_wildcard_reads_zero(self) -> None:
        client = EpochClient(_pool_with_bump(returning_epoch=1), _nats_mock())

        assert await client.current(Subject(path="app.capabilities.>", kind="pattern")) == 0


class TestEpochClientKeyDerivation:
    """A subject path is usually a legal KV key; when it is not, it is digested.

    A path segment can carry a caller-supplied value, and one space would turn
    a working bump into an ``InvalidKeyError`` raised in production rather than
    caught in review. Rejecting is not an option either: the caller cannot
    rename a user's layer.
    """

    @pytest.mark.asyncio
    async def test_a_path_with_an_illegal_character_still_counts(self) -> None:
        client = EpochClient(_pool_with_bump(returning_epoch=99), _nats_mock())
        subject = Subject(path="app.census tracts.epoch", kind="point")

        assert await client.bump(subject) == 1
        assert await client.current(subject) == 1

    @pytest.mark.asyncio
    async def test_two_illegal_paths_do_not_collide(self) -> None:
        client = EpochClient(_pool_with_bump(returning_epoch=99), _nats_mock())

        await client.bump(Subject(path="app.a b.epoch", kind="point"))

        assert await client.current(Subject(path="app.c d.epoch", kind="point")) == 0

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
from threetears.epoch.client import _IDENTITY_ATTEMPTS, _KV_KEY_GRAMMAR, EpochClient, _key_for
from threetears.epoch.wire import EpochBumpMessage
from threetears.nats.errors import PublishError
from threetears.nats.subjects import Subject, Subjects


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
    """a subject from the family whose epoch must survive a broker restart.

    Built with the REAL ``Subjects`` factory rather than a literal. The
    carve-out is decided by the subject's shape, and that shape is produced in
    a different package: a hand-written literal would keep matching after
    ``datasource_tile_epoch`` was restructured, so tile epochs would silently
    move onto a resettable counter -- the CDN cache-key failure the carve-out
    exists to prevent -- with every test still green.
    """
    return Subjects.datasource_tile_epoch("ds1", layer)


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

        subject = _durable_subject("parcels")
        assert await client.current(subject) == 0
        pool.fetchval.assert_awaited_once()
        sql, *args = pool.fetchval.await_args.args
        assert "SELECT epoch FROM config_epochs" in sql
        assert "WHERE subject_path" in sql
        assert args == [subject.path]

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

        subject = _durable_subject("parcels")
        await client.bump(subject, payload=None)

        sql, *args = pool.fetchrow.await_args.args
        assert "INSERT INTO config_epochs" in sql
        assert "ON CONFLICT (subject_path)" in sql
        assert "epoch = config_epochs.epoch + 1" in sql
        assert "RETURNING epoch" in sql
        # compared against the builder's own output, not a copy of it: a copied
        # literal keeps passing after the subject's shape changes, which is the
        # drift this whole carve-out is exposed to.
        assert args[0] == subject.path
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

    Asserted against the KEY, not against a round trip. ``FakeKvBucket``
    enforces no key grammar, so a round-trip test passes whether or not the
    derivation exists -- which is exactly the shape that would let an illegal
    key reach a real broker and raise ``InvalidKeyError`` at ``bump`` in
    production. The grammar here is the one ``nats-server`` enforces.
    """

    def test_a_legal_path_is_used_verbatim(self) -> None:
        """A readable key is worth having when someone is reading a bucket."""
        assert _key_for(_subject("app.capabilities.epoch")) == "app.capabilities.epoch"

    def test_an_illegal_path_is_digested_into_a_legal_key(self) -> None:
        key = _key_for(Subject(path="app.census tracts.epoch", kind="point"))

        assert " " not in key
        assert _KV_KEY_GRAMMAR.match(key), f"{key!r} is not a legal KV key"

    def test_the_digest_is_deterministic_across_processes(self) -> None:
        """Every pod must derive the same key, or they count different things."""
        subject = Subject(path="app.census tracts.epoch", kind="point")

        assert _key_for(subject) == _key_for(subject)

    def test_two_illegal_paths_derive_different_keys(self) -> None:
        a = _key_for(Subject(path="app.a b.epoch", kind="point"))
        b = _key_for(Subject(path="app.c d.epoch", kind="point"))

        assert a != b

    @pytest.mark.asyncio
    async def test_a_path_with_an_illegal_character_still_counts(self) -> None:
        """The behavioural half: derivation is not merely internally consistent."""
        client = EpochClient(_pool_with_bump(returning_epoch=99), _nats_mock())
        subject = Subject(path="app.census tracts.epoch", kind="point")

        assert await client.bump(subject) == 1
        assert await client.current(subject) == 1


class TestEpochClientBucketIdentity:
    """The bucket mints an opaque identity, and every opener converges on one."""

    @pytest.mark.asyncio
    async def test_an_identity_is_minted_on_first_ask(self) -> None:
        client = EpochClient(_pool_with_bump(returning_epoch=1), _nats_mock())

        assert await client.bucket_identity() is not None

    @pytest.mark.asyncio
    async def test_the_identity_is_stable_across_reads(self) -> None:
        """An identity that changed per read would look like a permanent reset."""
        client = EpochClient(_pool_with_bump(returning_epoch=1), _nats_mock())

        first = await client.bucket_identity()
        assert await client.bucket_identity() == first

    @pytest.mark.asyncio
    async def test_racing_clients_on_one_bucket_agree(self) -> None:
        """Create-if-absent resolves the race; the loser adopts the winner's."""
        nats = _nats_mock()
        a = EpochClient(_pool_with_bump(returning_epoch=1), nats)
        b = EpochClient(_pool_with_bump(returning_epoch=1), nats)

        assert await a.bucket_identity() == await b.bucket_identity()

    @pytest.mark.asyncio
    async def test_separate_buckets_carry_separate_identities(self) -> None:
        """Two brokers, two buckets, two identities -- the property detection needs."""
        first = EpochClient(_pool_with_bump(returning_epoch=1), _nats_mock())
        second = EpochClient(_pool_with_bump(returning_epoch=1), _nats_mock())

        assert await first.bucket_identity() != await second.bucket_identity()

    @pytest.mark.asyncio
    async def test_the_identity_key_is_not_a_counter(self) -> None:
        """It shares the bucket with the counters and must not be mistaken for one.

        No ``Subjects`` builder produces a path starting with an underscore --
        every one prefixes the namespace -- so the reserved key cannot collide
        with a subject's counter.
        """
        client = EpochClient(_pool_with_bump(returning_epoch=1), _nats_mock())
        await client.bucket_identity()

        assert await client.current(_subject()) == 0


class TestBucketIdentityFailsSafe:
    """A KV failure is an outage, not a replaced counter.

    The catch is deliberately broad, which is exactly why it needs pinning: a
    real defect inside it would otherwise surface as a permanent, silent "no
    replacement" across the whole fleet, and the detector would look healthy
    the entire time.
    """

    @pytest.mark.asyncio
    async def test_an_unreachable_broker_reports_no_identity(self) -> None:
        nats = _nats_mock()
        nats.kv_bucket = AsyncMock(side_effect=RuntimeError("broker unreachable"))
        client = EpochClient(_pool_with_bump(returning_epoch=1), nats)

        assert await client.bucket_identity() is None

    @pytest.mark.asyncio
    async def test_a_failing_create_reports_no_identity(self) -> None:
        nats = _nats_mock()
        bucket = MagicMock()
        bucket.create = AsyncMock(side_effect=RuntimeError("kv down"))
        nats.kv_bucket = AsyncMock(return_value=bucket)
        client = EpochClient(_pool_with_bump(returning_epoch=1), nats)

        assert await client.bucket_identity() is None

    @pytest.mark.asyncio
    async def test_a_failing_read_reports_no_identity(self) -> None:
        nats = _nats_mock()
        bucket = MagicMock()
        bucket.create = AsyncMock(return_value=None)
        bucket.get = AsyncMock(side_effect=RuntimeError("kv down"))
        nats.kv_bucket = AsyncMock(return_value=bucket)
        client = EpochClient(_pool_with_bump(returning_epoch=1), nats)

        assert await client.bucket_identity() is None

    @pytest.mark.asyncio
    async def test_a_bucket_wiped_between_create_and_read_gives_up_bounded(self) -> None:
        """Losing the create AND reading nothing means the bucket went again.

        Retried, then given up on -- a broker recreating the bucket faster than
        two round trips is an outage, not a race worth winning. Reporting an
        identity here would be indistinguishable from stability.
        """
        nats = _nats_mock()
        bucket = MagicMock()
        bucket.create = AsyncMock(return_value=None)
        bucket.get = AsyncMock(return_value=None)
        nats.kv_bucket = AsyncMock(return_value=bucket)
        client = EpochClient(_pool_with_bump(returning_epoch=1), nats)

        assert await client.bucket_identity() is None
        assert bucket.create.await_count == _IDENTITY_ATTEMPTS

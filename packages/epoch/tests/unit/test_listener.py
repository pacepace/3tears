"""unit tests for :class:`threetears.epoch.listener.EpochListener`.

covers cold-start last-seen priming via :meth:`EpochClient.current`,
strict monotonic dispatch, redelivery dedupe, gap-tolerant
"jump-ahead" delivery, the :meth:`catch_up` periodic-tick path, and
the :meth:`echo` per-message-echo path.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from threetears.epoch.client import EpochClient
from threetears.epoch.listener import EpochListener
from threetears.epoch.wire import EpochBumpMessage
from threetears.nats.subjects import Subject


def _subject(path: str = "app.capabilities.epoch") -> Subject:
    """build a point Subject for tests."""
    return Subject(path=path, kind="point")


def _pool_returning(epoch: int) -> Any:
    """build a pool stub whose fetchval returns ``epoch`` (or None for missing row)."""
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=epoch if epoch else None)
    pool.fetchrow = AsyncMock(return_value={"epoch": epoch} if epoch else None)
    return pool


def _capture_subscribe_typed() -> tuple[Any, list[Any]]:
    """build a NatsClient stub that captures subscribe_typed callbacks."""
    nats = MagicMock()
    captured_callbacks: list[Any] = []

    async def _subscribe_typed(*, subject: Any, cb: Any, message_type: Any, **kwargs: Any) -> None:  # noqa: ARG001
        """record cb so the test can dispatch synthetic messages."""
        captured_callbacks.append(cb)

    nats.subscribe_typed = AsyncMock(side_effect=_subscribe_typed)
    nats.publish = AsyncMock()
    return nats, captured_callbacks


class TestEpochListenerColdStartPriming:
    """:meth:`subscribe` primes last-seen via :meth:`EpochClient.current` BEFORE registering."""

    @pytest.mark.asyncio
    async def test_cold_start_primes_last_seen_from_postgres(self) -> None:
        """listener seeds last-seen with the durable row before subscribe registers."""
        pool = _pool_returning(epoch=12)
        nats, _ = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        subject = _subject()

        callback = AsyncMock()
        await listener.subscribe(subject, callback)

        assert listener.last_seen(subject) == 12

    @pytest.mark.asyncio
    async def test_cold_start_with_no_row_primes_zero(self) -> None:
        """fresh database -> last-seen starts at 0; first incoming bump fires."""
        pool = _pool_returning(epoch=0)
        nats, _ = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        subject = _subject()

        callback = AsyncMock()
        await listener.subscribe(subject, callback)

        assert listener.last_seen(subject) == 0

    @pytest.mark.asyncio
    async def test_primed_epoch_overrides_current_and_keeps_recovery(self) -> None:
        """``primed_epoch`` primes last-seen to the caller's loaded epoch, not ``current()``.

        the consumer read current()=3 + loaded its state, then a bump raced the
        load and the durable row is now 7. priming to the LOADED epoch (3) -- not
        the now-advanced current() (7) -- keeps last-seen BEHIND the missed bump,
        so the catch-up / next broadcast at 7 fires instead of being swallowed as
        already-seen. priming to current() (7) here would pin the stale catalog
        forever.
        """
        pool = _pool_returning(epoch=7)  # durable row already advanced past the consumer's load
        nats, callbacks = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        subject = _subject()

        await listener.subscribe(subject, AsyncMock(), primed_epoch=3)

        # last-seen reflects the LOADED epoch, not the advanced durable row.
        assert listener.last_seen(subject) == 3
        # the bump at 7 the loaded state missed now fires (recoverable), not dropped.
        await callbacks[0](EpochBumpMessage(subject_path=subject.path, epoch=7, payload={}))
        assert listener.last_seen(subject) == 7


class TestEpochListenerDispatch:
    """incoming broadcasts dedupe on subject path, monotonic increase only."""

    @pytest.mark.asyncio
    async def test_strictly_increasing_epoch_fires_callback(self) -> None:
        """new epoch > last-seen invokes the consumer callback with (epoch, payload)."""
        pool = _pool_returning(epoch=5)
        nats, callbacks = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        subject = _subject()
        consumer_cb = AsyncMock()

        await listener.subscribe(subject, consumer_cb)
        message = EpochBumpMessage(subject_path=subject.path, epoch=6, payload={"k": "v"})
        await callbacks[0](message)

        consumer_cb.assert_awaited_once_with(6, {"k": "v"})
        assert listener.last_seen(subject) == 6

    @pytest.mark.asyncio
    async def test_redelivered_epoch_drops_silent(self) -> None:
        """epoch == last-seen is a NATS-redelivery duplicate; do not fire."""
        pool = _pool_returning(epoch=5)
        nats, callbacks = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        subject = _subject()
        consumer_cb = AsyncMock()

        await listener.subscribe(subject, consumer_cb)
        message = EpochBumpMessage(subject_path=subject.path, epoch=5)
        await callbacks[0](message)

        consumer_cb.assert_not_awaited()
        assert listener.last_seen(subject) == 5

    @pytest.mark.asyncio
    async def test_out_of_order_older_epoch_drops(self) -> None:
        """delayed broadcast at epoch < last-seen never inverts last-seen."""
        pool = _pool_returning(epoch=10)
        nats, callbacks = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        subject = _subject()
        consumer_cb = AsyncMock()

        await listener.subscribe(subject, consumer_cb)
        message = EpochBumpMessage(subject_path=subject.path, epoch=3)
        await callbacks[0](message)

        consumer_cb.assert_not_awaited()
        assert listener.last_seen(subject) == 10

    @pytest.mark.asyncio
    async def test_gap_jump_fires_once_at_latest(self) -> None:
        """missed broadcasts: gap > 1 fires the callback once at the latest epoch."""
        pool = _pool_returning(epoch=2)
        nats, callbacks = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        subject = _subject()
        consumer_cb = AsyncMock()

        await listener.subscribe(subject, consumer_cb)
        message = EpochBumpMessage(subject_path=subject.path, epoch=99)
        await callbacks[0](message)

        consumer_cb.assert_awaited_once_with(99, None)
        assert listener.last_seen(subject) == 99

    @pytest.mark.asyncio
    async def test_independent_subjects_have_independent_last_seen(self) -> None:
        """one listener tracks last-seen per subject path independently."""
        pool = MagicMock()
        # priming for first subject -> 5; second -> 12.
        pool.fetchval = AsyncMock(side_effect=[5, 12])
        nats, callbacks = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        subject_a = _subject("app.capabilities.epoch")
        subject_b = _subject("3tears.gateway.catalog.epoch")
        cb_a = AsyncMock()
        cb_b = AsyncMock()

        await listener.subscribe(subject_a, cb_a)
        await listener.subscribe(subject_b, cb_b)

        assert listener.last_seen(subject_a) == 5
        assert listener.last_seen(subject_b) == 12

        # bump on A does not advance B's last-seen.
        await callbacks[0](EpochBumpMessage(subject_path=subject_a.path, epoch=6))
        assert listener.last_seen(subject_a) == 6
        assert listener.last_seen(subject_b) == 12


class TestEpochListenerCatchUp:
    """:meth:`catch_up` reads current and fires when stale."""

    @pytest.mark.asyncio
    async def test_catch_up_fires_when_durable_value_is_higher(self) -> None:
        """current(subject) > last_seen advances last-seen and invokes on_bump."""
        pool = MagicMock()
        # priming: 5; later catch-up: 10.
        pool.fetchval = AsyncMock(side_effect=[5, 10])
        nats, _ = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        subject = _subject()
        consumer_cb = AsyncMock()

        await listener.subscribe(subject, consumer_cb)
        result = await listener.catch_up(subject, consumer_cb)

        assert result == 10
        consumer_cb.assert_awaited_once_with(10, None)
        assert listener.last_seen(subject) == 10

    @pytest.mark.asyncio
    async def test_catch_up_no_op_when_already_current(self) -> None:
        """current(subject) == last_seen does NOT invoke on_bump."""
        pool = MagicMock()
        pool.fetchval = AsyncMock(side_effect=[5, 5])
        nats, _ = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        subject = _subject()
        consumer_cb = AsyncMock()

        await listener.subscribe(subject, consumer_cb)
        result = await listener.catch_up(subject, consumer_cb)

        assert result == 5
        consumer_cb.assert_not_awaited()


class TestEpochListenerRaceRecovery:
    """:meth:`subscribe` race-window recovery via :meth:`catch_up` (safety net).

    documents the contract called out in :meth:`subscribe`'s
    docstring: a bump that commits between prime-read and subscribe-
    register is missed by the broadcast. recovery is via the next
    periodic :meth:`catch_up` tick (or :meth:`echo`). this test
    explicitly asserts the recovery path so a future refactor cannot
    silently break the safety net.
    """

    @pytest.mark.asyncio
    async def test_catch_up_recovers_when_bump_lands_during_subscribe_window(self) -> None:
        """bump committed during prime/subscribe window: catch_up advances last_seen."""
        pool = MagicMock()
        # priming reads epoch=4; later catch_up sees epoch=5 (the missed bump).
        # no broadcast is dispatched for epoch=5 in this test (the listener
        # subscribed AFTER the missed broadcast left the wire).
        pool.fetchval = AsyncMock(side_effect=[4, 5])
        nats, _ = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        subject = _subject()
        consumer_cb = AsyncMock()

        await listener.subscribe(subject, consumer_cb)
        # listener primed at 4; the missed broadcast at 5 never arrived.
        assert listener.last_seen(subject) == 4
        consumer_cb.assert_not_awaited()

        # next periodic catch_up tick: discovers durable=5, fires callback.
        result = await listener.catch_up(subject, consumer_cb)

        assert result == 5
        assert listener.last_seen(subject) == 5
        consumer_cb.assert_awaited_once_with(5, None)


class TestEpochListenerEcho:
    """:meth:`echo` is the per-message epoch-echo path; pulls L3 to confirm."""

    @pytest.mark.asyncio
    async def test_echo_higher_than_last_seen_triggers_catch_up(self) -> None:
        """echoed > last_seen routes through catch_up, which reads current."""
        pool = MagicMock()
        pool.fetchval = AsyncMock(side_effect=[5, 10])
        nats, _ = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        subject = _subject()
        consumer_cb = AsyncMock()

        await listener.subscribe(subject, consumer_cb)
        await listener.echo(subject, echoed_epoch=10, on_bump=consumer_cb)

        consumer_cb.assert_awaited_once_with(10, None)
        assert listener.last_seen(subject) == 10

    @pytest.mark.asyncio
    async def test_echo_at_or_below_last_seen_is_no_op(self) -> None:
        """echoed <= last_seen short-circuits without touching Postgres."""
        pool = _pool_returning(epoch=10)
        nats, _ = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        subject = _subject()
        consumer_cb = AsyncMock()

        await listener.subscribe(subject, consumer_cb)
        # priming consumed the only fetchval; subsequent fetchval raises if called.
        pool.fetchval = AsyncMock(side_effect=AssertionError("must not pull"))

        await listener.echo(subject, echoed_epoch=10, on_bump=consumer_cb)
        await listener.echo(subject, echoed_epoch=3, on_bump=consumer_cb)

        consumer_cb.assert_not_awaited()
        assert listener.last_seen(subject) == 10

    @pytest.mark.asyncio
    async def test_echo_higher_than_last_seen_but_durable_disagrees_no_callback(self) -> None:
        """echoed > last_seen but durable still equals last_seen: no callback fires.

        defends against malicious / corrupt response envelopes that
        echo a higher epoch than the writer ever recorded. without
        the L3 confirmation, a hostile publisher could trigger
        spurious reloads.
        """
        pool = MagicMock()
        # priming: 5; catch-up: still 5 (echo lied).
        pool.fetchval = AsyncMock(side_effect=[5, 5])
        nats, _ = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        subject = _subject()
        consumer_cb = AsyncMock()

        await listener.subscribe(subject, consumer_cb)
        await listener.echo(subject, echoed_epoch=999, on_bump=consumer_cb)

        consumer_cb.assert_not_awaited()
        assert listener.last_seen(subject) == 5

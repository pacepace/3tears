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
    """build a Subject for tests, kinded from the path.

    A wildcard path is ``kind="pattern"`` -- the repo's own convention everywhere else
    (see ``threetears.nats.subjects``). These are the only in-repo examples of a
    wildcard EPOCH subject, so a "point" wildcard here is the shape the next consumer
    would copy.
    """
    kind = "pattern" if "*" in path or path.endswith(">") else "point"
    return Subject(path=path, kind=kind)


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


class TestEpochListenerWildcardSubscription:
    """one subscription covering many subjects keeps a counter PER subject.

    the failure this guards is silent and lossy, not merely stale. every
    matched subject owns an independent ``config_epochs`` row, so their
    epochs are unrelated numbers -- a bump to 5 on one and 1 on another
    are both legitimate. deduping them against a single counter drops
    the lower one and the consumer never learns it happened.
    """

    @pytest.mark.asyncio
    async def test_each_matched_subject_keeps_its_own_counter(self) -> None:
        """the case a shared counter loses: a high epoch then a low one."""
        pool = _pool_returning(epoch=0)
        nats, callbacks = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        wildcard = _subject("app.tenant.*.epoch")
        consumer_cb = AsyncMock()

        await listener.subscribe(wildcard, consumer_cb)
        await callbacks[0](EpochBumpMessage(subject_path="app.tenant.aaa.epoch", epoch=5))
        await callbacks[0](EpochBumpMessage(subject_path="app.tenant.bbb.epoch", epoch=1))

        assert consumer_cb.await_count == 2, "the second subject's bump was dropped against the first's counter"
        assert listener.last_seen(_subject("app.tenant.aaa.epoch")) == 5
        assert listener.last_seen(_subject("app.tenant.bbb.epoch")) == 1

    @pytest.mark.asyncio
    async def test_dedupe_still_holds_within_one_matched_subject(self) -> None:
        """per-subject counters must not cost the redelivery protection."""
        pool = _pool_returning(epoch=0)
        nats, callbacks = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        consumer_cb = AsyncMock()

        await listener.subscribe(_subject("app.tenant.*.epoch"), consumer_cb)
        await callbacks[0](EpochBumpMessage(subject_path="app.tenant.aaa.epoch", epoch=4))
        await callbacks[0](EpochBumpMessage(subject_path="app.tenant.aaa.epoch", epoch=4))
        await callbacks[0](EpochBumpMessage(subject_path="app.tenant.aaa.epoch", epoch=3))

        assert consumer_cb.await_count == 1

    @pytest.mark.asyncio
    async def test_a_concrete_subscription_is_unchanged(self) -> None:
        """the ordinary path must not shift: message path and subscribed path
        are the same string, so priming from `current()` still governs."""
        pool = _pool_returning(epoch=7)
        nats, callbacks = _capture_subscribe_typed()
        client = EpochClient(pool, nats)
        listener = EpochListener(nats, client)
        subject = _subject()
        consumer_cb = AsyncMock()

        await listener.subscribe(subject, consumer_cb)
        await callbacks[0](EpochBumpMessage(subject_path=subject.path, epoch=7))

        consumer_cb.assert_not_awaited()
        assert listener.last_seen(subject) == 7


class TestEpochListenerResetFanOut:
    """A reset reaches every registered consumer, not the one that noticed it.

    The listener previously kept no reference to any callback -- ``on_bump``
    lived only as a closure argument -- so whichever call spotted a reset could
    invoke only its own. A pod subscribed to two subjects would tell one
    consumer and leave the other holding state it has no reason to doubt, which
    for a subject that never bumps again is permanent.
    """

    @pytest.mark.asyncio
    async def test_every_registered_consumer_is_told(self) -> None:
        pool = _pool_returning(epoch=5)
        nats, _ = _capture_subscribe_typed()
        listener = EpochListener(nats, EpochClient(pool, nats))

        reset_a, reset_b = AsyncMock(), AsyncMock()
        await listener.subscribe(_subject("app.a.epoch"), AsyncMock(), on_reset=reset_a)
        await listener.subscribe(_subject("app.b.epoch"), AsyncMock(), on_reset=reset_b)

        await listener.signal_reset()

        reset_a.assert_awaited_once()
        reset_b.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_wildcard_registration_is_told_once_after_matching_several(self) -> None:
        """One registration, one callback, however many subjects it matched.

        The matching has to actually happen for this to mean anything: a reset
        signalled against a wildcard that never saw a message would report one
        invocation whether the rule were per-registration or per-matched-path.
        So two concrete subjects are delivered first, which is what puts two
        entries in the dedupe map, and only then is the reset signalled.

        Dedupe keys on the MESSAGE path, because those counters are unrelated
        numbers. Registration keys on the SUBSCRIBED path, because the consumer
        is one.
        """
        pool = _pool_returning(epoch=0)
        nats, callbacks = _capture_subscribe_typed()
        listener = EpochListener(nats, EpochClient(pool, nats))

        on_reset = AsyncMock()
        await listener.subscribe(_subject("app.*.epoch"), AsyncMock(), on_reset=on_reset)
        await callbacks[0](EpochBumpMessage(subject_path="app.a.epoch", epoch=1, payload=None))
        await callbacks[0](EpochBumpMessage(subject_path="app.b.epoch", epoch=1, payload=None))

        await listener.signal_reset()

        on_reset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_two_consumers_on_the_same_subject_are_both_told(self) -> None:
        """One listener per pod is the documented shape, so this is normal usage.

        Both already receive bumps, because each ``subscribe`` registers its own
        NATS callback. Recording one entry per PATH rather than per
        subscription would give both consumers bumps and only the last one
        resets -- the same permanent staleness this class exists to remove, one
        scope smaller and considerably harder to notice.
        """
        pool = _pool_returning(epoch=3)
        nats, _ = _capture_subscribe_typed()
        listener = EpochListener(nats, EpochClient(pool, nats))
        subject = _subject("app.shared.epoch")

        first, second = AsyncMock(), AsyncMock()
        await listener.subscribe(subject, AsyncMock(), on_reset=first)
        await listener.subscribe(subject, AsyncMock(), on_reset=second)

        await listener.signal_reset()

        first.assert_awaited_once()
        second.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_raising_callback_does_not_deprive_the_others(self) -> None:
        """The reset runs after last-seen is cleared, with nothing to retry it.

        An early raise would leave every consumer after it both un-notified and
        un-primed, which is worse than the staleness the reset announces.
        """
        pool = _pool_returning(epoch=3)
        nats, _ = _capture_subscribe_typed()
        listener = EpochListener(nats, EpochClient(pool, nats))

        boom = AsyncMock(side_effect=RuntimeError("consumer bug"))
        after = AsyncMock()
        await listener.subscribe(_subject("app.a.epoch"), AsyncMock(), on_reset=boom)
        await listener.subscribe(_subject("app.b.epoch"), AsyncMock(), on_reset=after)

        with pytest.raises(RuntimeError, match="consumer bug"):
            await listener.signal_reset()

        after.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_raising_callback_still_surfaces(self) -> None:
        """Continuing is not swallowing: the consumer bug is re-raised."""
        pool = _pool_returning(epoch=3)
        nats, _ = _capture_subscribe_typed()
        listener = EpochListener(nats, EpochClient(pool, nats))
        await listener.subscribe(
            _subject("app.a.epoch"), AsyncMock(), on_reset=AsyncMock(side_effect=RuntimeError("consumer bug"))
        )

        with pytest.raises(RuntimeError, match="consumer bug"):
            await listener.signal_reset()

    @pytest.mark.asyncio
    async def test_a_consumer_without_a_reset_callback_is_skipped(self) -> None:
        """Omitting it is a choice, not an error."""
        pool = _pool_returning(epoch=5)
        nats, _ = _capture_subscribe_typed()
        listener = EpochListener(nats, EpochClient(pool, nats))

        told = AsyncMock()
        await listener.subscribe(_subject("app.a.epoch"), AsyncMock())
        await listener.subscribe(_subject("app.b.epoch"), AsyncMock(), on_reset=told)

        await listener.signal_reset()

        told.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_last_seen_is_cleared_not_reprimed(self) -> None:
        """Re-priming re-creates the wedge this exists to prevent.

        Identity and counter are two non-atomic reads, so a value read against
        the old generation could be written into last-seen against the new one.
        Clearing costs one redundant reload and cannot re-wedge.
        """
        pool = _pool_returning(epoch=5000)
        nats, _ = _capture_subscribe_typed()
        listener = EpochListener(nats, EpochClient(pool, nats))
        subject = _subject("app.a.epoch")
        await listener.subscribe(subject, AsyncMock(), on_reset=AsyncMock())
        assert listener.last_seen(subject) == 5000

        await listener.signal_reset()

        assert listener.last_seen(subject) == 0

    @pytest.mark.asyncio
    async def test_a_bump_after_a_reset_dispatches_from_zero(self) -> None:
        """The point of clearing: the new counter's first bump is not swallowed."""
        pool = _pool_returning(epoch=5000)
        nats, callbacks = _capture_subscribe_typed()
        listener = EpochListener(nats, EpochClient(pool, nats))
        on_bump = AsyncMock()
        await listener.subscribe(_subject("app.a.epoch"), on_bump, on_reset=AsyncMock())

        await listener.signal_reset()
        await callbacks[0](EpochBumpMessage(subject_path="app.a.epoch", epoch=1, payload=None))

        on_bump.assert_awaited_once()


class TestEpochListenerRegistrationOrdering:
    """A failed subscribe must leave no registration behind.

    Registering before the subscribe would strand an entry when it raises, so
    a caller that retries double-registers and every later reset fires that
    consumer twice -- a reload it cannot tell from a genuine second reset.
    """

    @pytest.mark.asyncio
    async def test_a_failed_subscribe_registers_nothing(self) -> None:
        from threetears.nats.errors import SubscribeError

        pool = _pool_returning(epoch=1)
        nats = MagicMock()
        nats.subscribe_typed = AsyncMock(side_effect=SubscribeError("broker refused"))
        listener = EpochListener(nats, EpochClient(pool, nats))

        on_reset = AsyncMock()
        with pytest.raises(SubscribeError):
            await listener.subscribe(_subject("app.a.epoch"), AsyncMock(), on_reset=on_reset)

        await listener.signal_reset()
        on_reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_retry_after_a_failure_registers_exactly_once(self) -> None:
        from threetears.nats.errors import SubscribeError

        pool = _pool_returning(epoch=1)
        nats = MagicMock()
        nats.subscribe_typed = AsyncMock(side_effect=[SubscribeError("transient"), None])
        listener = EpochListener(nats, EpochClient(pool, nats))
        subject = _subject("app.a.epoch")

        on_reset = AsyncMock()
        with pytest.raises(SubscribeError):
            await listener.subscribe(subject, AsyncMock(), on_reset=on_reset)
        await listener.subscribe(subject, AsyncMock(), on_reset=on_reset)

        await listener.signal_reset()
        on_reset.assert_awaited_once()

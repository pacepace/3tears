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
    """build a pool stub whose fetchval returns ``epoch`` (or None for missing row).

    Retained for the DURABLE subject family, which still reads Postgres.
    """
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=epoch if epoch else None)
    pool.fetchrow = AsyncMock(return_value={"epoch": epoch} if epoch else None)
    return pool


class _StubEpochClient(EpochClient):
    """An :class:`EpochClient` whose ``current`` answers a fixed value.

    These are LISTENER tests: what they exercise is dispatch given an epoch,
    not where the epoch was stored. Building a real client over a stubbed pool
    coupled them to the substrate, which is why moving the counter from
    Postgres to NATS KV broke two dozen of them at once without any of the
    listener's behaviour changing.

    Subclassing rather than duck-typing is deliberate: it declares the
    production type this stands in for, which is what
    ``tests/enforcement/test_fake_protocol_parity.py`` asks of a test double,
    and it means a real signature change still breaks these loudly.
    """

    def __init__(self, nats_client: Any, epoch: int | list[int] = 0) -> None:
        super().__init__(MagicMock(), nats_client)
        #: a list answers successive calls, so a test can drive "the epoch
        #: advanced between the prime and the catch-up" without reaching into
        #: whichever store happens to be behind ``current`` today.
        self._stub_epochs = list(epoch) if isinstance(epoch, list) else [epoch]

    async def current(self, subject: Subject) -> int:  # noqa: ARG002 - scripted answer by design
        """return the next scripted epoch, repeating the last one."""
        return self._stub_epochs.pop(0) if len(self._stub_epochs) > 1 else self._stub_epochs[0]


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
        _seeded = 12
        nats, _ = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
        listener = EpochListener(nats, client)
        subject = _subject()

        callback = AsyncMock()
        await listener.subscribe(subject, callback)

        assert listener.last_seen(subject) == 12

    @pytest.mark.asyncio
    async def test_cold_start_with_no_row_primes_zero(self) -> None:
        """fresh database -> last-seen starts at 0; first incoming bump fires."""
        _seeded = 0
        nats, _ = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
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
        _seeded = 7  # durable row already advanced past the consumer's load
        nats, callbacks = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
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
        _seeded = 5
        nats, callbacks = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
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
        _seeded = 5
        nats, callbacks = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
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
        _seeded = 10
        nats, callbacks = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
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
        _seeded = 2
        nats, callbacks = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
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
        # priming for first subject -> 5; second -> 12.
        _seeded = [5, 12]
        nats, callbacks = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
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
        # priming: 5; later catch-up: 10.
        _seeded = [5, 10]
        nats, _ = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
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
        _seeded = [5, 5]
        nats, _ = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
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
        # priming reads epoch=4; later catch_up sees epoch=5 (the missed bump).
        # no broadcast is dispatched for epoch=5 in this test (the listener
        # subscribed AFTER the missed broadcast left the wire).
        _seeded = [4, 5]
        nats, _ = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
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
        _seeded = [5, 10]
        nats, _ = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
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
        _seeded = 10
        nats, _ = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
        listener = EpochListener(nats, client)
        subject = _subject()
        consumer_cb = AsyncMock()

        await listener.subscribe(subject, consumer_cb)
        # the stub is scripted to answer 10 and then keep answering 10, so an
        # echo at or below last-seen that DID consult it would still not fire.

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
        # priming: 5; catch-up: still 5 (echo lied).
        _seeded = [5, 5]
        nats, _ = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
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
        _seeded = 0
        nats, callbacks = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
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
        _seeded = 0
        nats, callbacks = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
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
        _seeded = 7
        nats, callbacks = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=_seeded)
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
        _seeded = 5
        nats, _ = _capture_subscribe_typed()
        listener = EpochListener(nats, _StubEpochClient(nats, epoch=_seeded))

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
        _seeded = 0
        nats, callbacks = _capture_subscribe_typed()
        listener = EpochListener(nats, _StubEpochClient(nats, epoch=_seeded))

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
        _seeded = 3
        nats, _ = _capture_subscribe_typed()
        listener = EpochListener(nats, _StubEpochClient(nats, epoch=_seeded))
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
        _seeded = 3
        nats, _ = _capture_subscribe_typed()
        listener = EpochListener(nats, _StubEpochClient(nats, epoch=_seeded))

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
        _seeded = 3
        nats, _ = _capture_subscribe_typed()
        listener = EpochListener(nats, _StubEpochClient(nats, epoch=_seeded))
        await listener.subscribe(
            _subject("app.a.epoch"), AsyncMock(), on_reset=AsyncMock(side_effect=RuntimeError("consumer bug"))
        )

        with pytest.raises(RuntimeError, match="consumer bug"):
            await listener.signal_reset()

    @pytest.mark.asyncio
    async def test_a_consumer_without_a_reset_callback_is_skipped(self) -> None:
        """Omitting it is a choice, not an error."""
        _seeded = 5
        nats, _ = _capture_subscribe_typed()
        listener = EpochListener(nats, _StubEpochClient(nats, epoch=_seeded))

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
        _seeded = 5000
        nats, _ = _capture_subscribe_typed()
        listener = EpochListener(nats, _StubEpochClient(nats, epoch=_seeded))
        subject = _subject("app.a.epoch")
        await listener.subscribe(subject, AsyncMock(), on_reset=AsyncMock())
        assert listener.last_seen(subject) == 5000

        await listener.signal_reset()

        assert listener.last_seen(subject) == 0

    @pytest.mark.asyncio
    async def test_a_bump_after_a_reset_dispatches_from_zero(self) -> None:
        """The point of clearing: the new counter's first bump is not swallowed."""
        _seeded = 5000
        nats, callbacks = _capture_subscribe_typed()
        listener = EpochListener(nats, _StubEpochClient(nats, epoch=_seeded))
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

        _seeded = 1
        nats = MagicMock()
        nats.subscribe_typed = AsyncMock(side_effect=SubscribeError("broker refused"))
        listener = EpochListener(nats, _StubEpochClient(nats, epoch=_seeded))

        on_reset = AsyncMock()
        with pytest.raises(SubscribeError):
            await listener.subscribe(_subject("app.a.epoch"), AsyncMock(), on_reset=on_reset)

        await listener.signal_reset()
        on_reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_retry_after_a_failure_registers_exactly_once(self) -> None:
        from threetears.nats.errors import SubscribeError

        _seeded = 1
        nats = MagicMock()
        nats.subscribe_typed = AsyncMock(side_effect=[SubscribeError("transient"), None])
        listener = EpochListener(nats, _StubEpochClient(nats, epoch=_seeded))
        subject = _subject("app.a.epoch")

        on_reset = AsyncMock()
        with pytest.raises(SubscribeError):
            await listener.subscribe(subject, AsyncMock(), on_reset=on_reset)
        await listener.subscribe(subject, AsyncMock(), on_reset=on_reset)

        await listener.signal_reset()
        on_reset.assert_awaited_once()


class TestEpochListenerBackwardsCounterIsAReset:
    """A monotonic counter that goes backwards is a DIFFERENT counter.

    The ephemeral epochs live in a memory-backed KV bucket, so a broker restart
    recreates it empty and every operation then succeeds while reading zero.
    There is no error to catch. Without this, ``catch_up``'s only arm is
    ``current > last_seen``, which can never fire again once the new counter
    starts below the old one -- the pod stops reloading permanently, and
    silently, which is worse than the durable row this replaced.
    """

    @pytest.mark.asyncio
    async def test_a_backwards_read_fires_every_reset_callback(self) -> None:
        nats, _ = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=[5000, 0])
        listener = EpochListener(nats, client)
        subject = _subject("app.a.epoch")
        on_reset = AsyncMock()
        await listener.subscribe(subject, AsyncMock(), on_reset=on_reset)

        await listener.catch_up(subject, AsyncMock())

        on_reset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_backwards_read_clears_last_seen_so_bumps_dispatch_again(self) -> None:
        """The wedge this exists to prevent, stated as the behaviour that returns."""
        nats, callbacks = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=[5000, 0])
        listener = EpochListener(nats, client)
        subject = _subject("app.a.epoch")
        on_bump = AsyncMock()
        await listener.subscribe(subject, on_bump, on_reset=AsyncMock())

        await listener.catch_up(subject, AsyncMock())
        await callbacks[0](EpochBumpMessage(subject_path="app.a.epoch", epoch=1, payload=None))

        on_bump.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_an_equal_read_is_not_a_reset(self) -> None:
        """Only BACKWARDS is suspicious; steady state must stay quiet."""
        nats, _ = _capture_subscribe_typed()
        client = _StubEpochClient(nats, epoch=5)
        listener = EpochListener(nats, client)
        subject = _subject("app.a.epoch")
        on_reset = AsyncMock()
        await listener.subscribe(subject, AsyncMock(), on_reset=on_reset)

        await listener.catch_up(subject, AsyncMock())

        on_reset.assert_not_awaited()

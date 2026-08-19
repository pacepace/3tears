"""epoch listener -- subscribe-side dispatch with monotonic dedupe.

:class:`EpochListener` is the subscribe-side companion to
:class:`~threetears.epoch.client.EpochClient`. one listener instance per
pod tracks last-seen epoch per subject path in process state and
dispatches monotonically-increasing bumps to consumer-supplied
callbacks. redelivered or out-of-order broadcasts at epoch <= last-seen
are dropped (logged at DEBUG); callbacks for monotonically-increasing
epochs always fire, even if the gap is greater than 1 (silent missed
broadcasts: the callback runs once at the latest epoch and the consumer
reloads from L3 -- gap-aware reload is the consumer's concern, not the
framework's).

mirrors the typed-NATS subscribe shape established by
:meth:`~threetears.core.collections.registry.CollectionRegistry.start_invalidation_listener`:
:meth:`~threetears.nats.NatsClient.subscribe_typed` with a
``message_type=EpochBumpMessage`` validator, narrow exception scope on
deserialization, programming errors propagate.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from threetears.nats import NatsClient
from threetears.nats.errors import SubscribeError
from threetears.nats.subjects import Subject
from threetears.observe import get_logger

from threetears.epoch.client import EpochClient
from threetears.epoch.wire import EpochBumpMessage

__all__ = [
    "BumpCallback",
    "EpochListener",
    "ResetCallback",
]

log = get_logger(__name__)


BumpCallback = Callable[[int, dict[str, object] | None], Awaitable[None]]
"""signature for a consumer's reload callback.

invoked with ``(new_epoch, payload)``. the callback is responsible
for deciding what to reload and from where -- the framework knows
nothing about the consumer's caches. exceptions raised inside the
callback propagate; the listener does not swallow consumer bugs.
"""


ResetCallback = Callable[[], Awaitable[None]]
"""signature for a consumer's reset callback.

invoked with NO epoch, deliberately. a reset means the counter this
listener was tracking has been REPLACED, so every number it ever
compared against is meaningless -- including whatever the new counter
happens to read right now. handing one over would invite the consumer
to dedupe on it, and this module spends its whole docstring surface
teaching consumers to dedupe on epochs. a value below everything the
consumer has already acted on is exactly what that dedupe discards, so
the reset would be swallowed and the staleness it announces would
persist.

**must not be epoch-deduped.** a consumer that receives this reloads,
unconditionally.
"""


class EpochListener:
    """subscribe-side dispatcher for cross-pod config-epoch coherence.

    one instance per pod, shared across every subject the pod cares
    about. last-seen state is process-local; restart resets it
    (subscribers prime via :meth:`EpochClient.current` on cold start
    so the first incoming broadcast matches what was already loaded).

    :param nats_client: connected typed NATS wrapper for subscribes
    :ptype nats_client: NatsClient
    :param epoch_client: companion :class:`EpochClient` used for
        cold-start last-seen priming
    :ptype epoch_client: EpochClient
    """

    def __init__(self, nats_client: NatsClient, epoch_client: EpochClient) -> None:
        """capture clients; no I/O.

        :param nats_client: connected NatsClient
        :ptype nats_client: NatsClient
        :param epoch_client: companion :class:`EpochClient` for
            cold-start priming
        :ptype epoch_client: EpochClient
        :return: nothing
        :rtype: None
        """
        self._nats = nats_client
        self._epoch_client = epoch_client
        self._last_seen: dict[str, int] = {}
        # Every registered subscription, so a reset can reach ALL of them.
        # Without this the callbacks live only as closure arguments to
        # ``subscribe`` and ``catch_up``, and whichever call happened to
        # notice a reset could invoke only its own -- so a pod subscribed to
        # two subjects would tell one consumer and silently leave the other
        # holding state it has no reason to doubt. For a subject that never
        # bumps again, that is permanent.
        #
        # A LIST per path, not one entry per path. This class is one listener
        # per pod shared across everything the pod cares about, so two
        # components subscribing the SAME subject is intended usage, and each
        # registers its own NATS callback -- so both already receive bumps.
        # One entry per path would give both consumers bumps and only the last
        # one resets: the same permanent staleness, one scope smaller and
        # considerably harder to notice.
        self._registrations: dict[str, list[tuple[Subject, BumpCallback, ResetCallback | None]]] = {}

    def last_seen(self, subject: Subject) -> int:
        """return the listener's recorded last-seen epoch for a subject.

        primarily for tests + diagnostics. returns ``0`` if the
        subject has never been subscribed (or was subscribed but
        cold-start priming saw no row in ``config_epochs``).

        :param subject: target subject
        :ptype subject: Subject
        :return: last-seen epoch, or ``0`` if unknown
        :rtype: int
        """
        return self._last_seen.get(subject.path, 0)

    async def subscribe(
        self,
        subject: Subject,
        on_bump: BumpCallback,
        primed_epoch: int | None = None,
        on_reset: ResetCallback | None = None,
    ) -> None:
        """register a callback for monotonic bumps on a subject.

        primes the per-subject last-seen BEFORE the NATS subscription
        registers, so the first broadcast a subscriber receives is
        compared against the durable view rather than against ``0``.
        without this priming, every cold-started pod would fire its
        ``on_bump`` callback once on the first arriving broadcast
        even when the pod's local state already reflects that epoch.

        WHERE last-seen is primed FROM matters for correctness when
        the consumer loaded local state (a catalog, a cache) before
        subscribing. pass ``primed_epoch`` = the epoch that loaded
        state reflects (read :meth:`EpochClient.current` BEFORE the
        load, then load, then subscribe). last-seen is then never
        ahead of the loaded state, so any bump that commits at or
        after the load is detected (broadcast or :meth:`catch_up`)
        and the state can never go PERMANENTLY stale. omitting
        ``primed_epoch`` reads :meth:`EpochClient.current` at
        subscribe time -- correct only when no state was loaded
        against an earlier epoch, because a bump landing between the
        load and this read would advance last-seen PAST the loaded
        state and the catch-up (``current == last_seen``) would never
        recover it.

        race window (intentional, recoverable): a bump that commits
        between the primed epoch and the NATS subscribe registration
        is missed by the broadcast (subscription not live) but leaves
        last-seen BEHIND it, so the next broadcast at a higher epoch
        fires via gap-jump dispatch, or the periodic :meth:`catch_up`
        tick recovers it. proven by :func:`tests.unit.test_listener.
        TestEpochListenerRaceRecovery.
        test_catch_up_recovers_when_bump_lands_during_subscribe_window`.

        narrow exception scope: :class:`~threetears.nats.errors.
        SubscribeError` propagates because cache coherence is not
        optional. validation failures inside the typed dispatcher
        deadletter via the standard typed-NATS path.

        WILDCARD subscriptions are supported, and differ from a
        concrete one in THREE respects, all worth knowing before using
        one. Dedupe keys on
        the path each MESSAGE names, so every matched subject keeps
        its own counter -- necessary, because each has an independent
        ``config_epochs`` row and their epochs are unrelated numbers.
        Priming, however, cannot work: ``current()`` needs a concrete
        row, and the concrete subjects are not known until their
        messages arrive. So each matched subject starts at ``0`` and
        the FIRST bump seen for it always dispatches, even when the
        consumer's state already reflected that epoch.

        That is one redundant callback per matched subject per process
        lifetime. Harmless where a reload is cheap or idempotent;
        weigh it where a reload is expensive. A consumer that cannot
        afford it should subscribe per concrete subject and prime each,
        which is what a non-wildcard subscription already does.

        SECOND: ``_last_seen`` gains an entry per matched subject and
        never evicts one. The growth removed from the subscription
        count reappears here as dict entries, bounded by how many
        subjects actually broadcast rather than by how many exist. For
        a per-customer epoch that is one entry per active customer,
        which is the point -- but it is not free, and a wildcard over
        an unbounded subject space is a leak rather than a
        convenience.

        THIRD, and the one that changes behaviour for an EXISTING
        wildcard subscriber: a wildcard subscribe needs a wildcard
        grant, and a grant cannot distinguish a literal segment from a
        wildcard one. Every matched subject's bump and payload now
        reaches the callback, where collapsing them onto a single
        counter previously suppressed most of them. A consumer that
        was relying on that suppression -- deliberately or not -- sees
        more callbacks after this change, not fewer.

        :param subject: subject to subscribe to, concrete or wildcard.
            The dedupe key is the path the MESSAGE carries, so a
            wildcard tracks each matched subject separately
        :ptype subject: Subject
        :param on_bump: async callback invoked on each monotonic
            bump with ``(new_epoch, payload)``
        :ptype on_bump: BumpCallback
        :param primed_epoch: epoch to prime last-seen to -- the epoch
            the consumer's already-loaded local state reflects (read
            ``current()`` before loading). ``None`` reads
            ``current()`` now (no state loaded against an earlier
            epoch)
        :ptype primed_epoch: int | None
        :param on_reset: invoked when the counter this listener tracks is
            REPLACED rather than advanced, with no epoch -- see
            :data:`ResetCallback`. omitting it means this consumer is not
            told, which is a choice rather than a default: after a reset its
            state is stale with nothing scheduled to correct it.
        :ptype on_reset: ResetCallback | None
        :return: nothing
        :rtype: None
        :raises SubscribeError: if the underlying NATS subscribe
            fails to register
        """
        primed = primed_epoch if primed_epoch is not None else await self._epoch_client.current(subject)
        self._last_seen[subject.path] = primed

        log.debug(
            "epoch listener primed last-seen",
            extra={
                "extra_data": {
                    "subject": subject.path,
                    "primed_epoch": primed,
                },
            },
        )

        async def _on_bump(message: EpochBumpMessage) -> None:
            """typed dispatch for one incoming bump.

            de-duplicates against last-seen on the subject path the
            MESSAGE names, not the one subscribed to. for a concrete
            subscription the two are identical; for a wildcard they
            are not, and using the subscribed path would collapse
            every matched subject onto one counter.

            that collapse loses writes rather than merely delaying
            them: each subject owns an independent ``config_epochs``
            row, so their counters are unrelated. a bump to 5 on one
            subject followed by a bump to 1 on another would see
            ``1 <= 5`` and drop the second silently.
            """
            dedupe_path = message.subject_path or subject.path
            current_last_seen = self._last_seen.get(dedupe_path, 0)
            if message.epoch <= current_last_seen:
                log.debug(
                    "epoch broadcast dropped (already seen)",
                    extra={
                        "extra_data": {
                            "subject": dedupe_path,
                            "subscribed": subject.path,
                            "incoming_epoch": message.epoch,
                            "last_seen": current_last_seen,
                        },
                    },
                )
                return
            self._last_seen[dedupe_path] = message.epoch
            await on_bump(message.epoch, message.payload)

        try:
            await self._nats.subscribe_typed(
                subject=subject,
                message_type=EpochBumpMessage,
                cb=_on_bump,
            )
        except SubscribeError:
            raise
        # recorded only AFTER the subscribe succeeds. registering first would
        # leave an entry behind when this raises, so a caller that retries
        # would double-register and every later reset would fire that
        # consumer twice -- a reload it cannot tell from a real second reset.
        #
        # keyed on the SUBSCRIBED path, not the message path: a wildcard is
        # one registration and gets one reset callback, however many concrete
        # subjects it later matches. dedupe stays keyed on the message path,
        # because those counters are independent numbers.
        self._registrations.setdefault(subject.path, []).append((subject, on_bump, on_reset))

    async def signal_reset(self) -> None:
        """the counter was REPLACED; drop local state and tell every consumer.

        Called when the coherence substrate is found to be a different one --
        a broker restart on ephemeral storage recreates the bucket empty, and
        the counter starts again from zero while this process keeps running
        with a high last-seen. The detection that calls this lives in the listener's identity
        comparison; this is the mechanism it triggers.

        **Clears rather than re-primes.** Re-priming from the new counter
        looks tidier and re-creates the bug: identity and counter are two
        separate reads with no atomicity between them, so a value read against
        the OLD generation can be written into last-seen against the new one,
        leaving the pod wedged again with no further identity change to rescue
        it. Clearing costs at most one redundant reload per subject and cannot
        re-wedge.

        Fans out to EVERY registration, not the one that noticed. A pod
        subscribed to two subjects has two consumers holding state, and the
        reset is equally true for both. A consumer that registered no
        ``on_reset`` is skipped: it chose that at :meth:`subscribe`.

        **One raising callback does not deprive the rest.** A bump callback
        propagates because it harms only its own consumer, and that precedent
        does not transfer here: this runs after ``_last_seen`` is already
        cleared, with nothing scheduled to retry, so an early raise would
        leave every consumer after it un-notified AND un-primed. Every
        callback is attempted; the first exception is re-raised afterwards, so
        a consumer bug still surfaces rather than being swallowed.

        :return: nothing
        :rtype: None
        :raises Exception: the first exception raised by any consumer
            callback, after every other callback has been attempted
        """
        self._last_seen.clear()
        registrations = [entry for entries in self._registrations.values() for entry in entries]
        notified = 0
        first_error: BaseException | None = None
        for subject, _on_bump, on_reset in registrations:
            if on_reset is None:
                continue
            try:
                await on_reset()
                notified += 1
            # prawduct:allow prawduct/broad-except -- a consumer bug must not
            # deprive the remaining consumers of a reset they cannot otherwise
            # learn about. re-raised below, never swallowed.
            except Exception as exc:  # noqa: BLE001
                log.exception(
                    "epoch reset callback raised; continuing to the remaining consumers",
                    extra={"extra_data": {"subject": subject.path}},
                )
                if first_error is None:
                    first_error = exc
        log.info(
            "epoch counter replaced; last-seen cleared and consumers notified",
            extra={"extra_data": {"registrations": len(registrations), "notified": notified}},
        )
        if first_error is not None:
            raise first_error

    async def catch_up(
        self,
        subject: Subject,
        on_bump: BumpCallback,
    ) -> int:
        """pull the current epoch and fire on_bump if stale.

        public hook for periodic catch-up ticks and for per-message
        epoch-echo paths. reads :meth:`EpochClient.current`; if the
        result is greater than this listener's last-seen for the
        subject, advances last-seen and invokes ``on_bump``.

        idempotent: calling repeatedly with no intervening bump is a
        cheap one-row indexed lookup with no side effect.

        :param subject: target subject
        :ptype subject: Subject
        :param on_bump: same callback shape as :meth:`subscribe`;
            invoked when the pulled epoch is strictly greater than
            last-seen
        :ptype on_bump: BumpCallback
        :return: the resolved current epoch (matches what
            :meth:`last_seen` will return after this call)
        :rtype: int
        """
        current = await self._epoch_client.current(subject)
        last_seen = self._last_seen.get(subject.path, 0)
        if current > last_seen:
            self._last_seen[subject.path] = current
            await on_bump(current, None)
        return current

    async def echo(
        self,
        subject: Subject,
        echoed_epoch: int,
        on_bump: BumpCallback,
    ) -> None:
        """consume a per-message epoch echo from a response envelope.

        consumer-side helper for the per-message echo discipline.
        when a response carries an ``epochs`` map (e.g. gateway
        completion responses echo their view of
        ``catalog.tool-gateway`` and ``mcp.rbac``), forward each
        ``(subject, echoed_epoch)`` pair through this method. if
        echoed > last-seen, schedule a fetch (here: pull current
        from L3 to confirm, then advance last-seen + invoke
        ``on_bump``).

        the echoed value is treated as a *hint*; the callback fires
        only after the durable :meth:`EpochClient.current` confirms
        the higher value (defends against malicious / corrupt
        envelopes).

        :param subject: subject the echo refers to
        :ptype subject: Subject
        :param echoed_epoch: epoch value the response envelope
            advertises for this subject
        :ptype echoed_epoch: int
        :param on_bump: same callback shape as :meth:`subscribe`;
            invoked when the echoed value is confirmed by L3 and
            is strictly greater than last-seen
        :ptype on_bump: BumpCallback
        :return: nothing
        :rtype: None
        """
        last_seen = self._last_seen.get(subject.path, 0)
        if echoed_epoch <= last_seen:
            return
        await self.catch_up(subject, on_bump)

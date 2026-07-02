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
]

log = get_logger(__name__)


BumpCallback = Callable[[int, dict[str, object] | None], Awaitable[None]]
"""signature for a consumer's reload callback.

invoked with ``(new_epoch, payload)``. the callback is responsible
for deciding what to reload and from where -- the framework knows
nothing about the consumer's caches. exceptions raised inside the
callback propagate; the listener does not swallow consumer bugs.
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
    ) -> None:
        """register a callback for monotonic bumps on a subject.

        primes the per-subject last-seen via :meth:`EpochClient.
        current` BEFORE the NATS subscription registers, so the
        first broadcast a subscriber receives is compared against
        the durable Postgres view rather than against ``0``. without
        this priming, every cold-started pod would fire its
        ``on_bump`` callback once on the first arriving broadcast
        even when the pod's local state already reflects that
        epoch (e.g. via :meth:`EpochClient.current` having been
        called by an upstream catalog-load).

        race window (intentional): a bump that commits between the
        :meth:`EpochClient.current` read and the NATS subscribe
        registration is missed by the broadcast (subscription not
        live) and not reflected in ``primed`` (read before commit).
        the next broadcast at higher epoch fires correctly via
        gap-jump dispatch; if no further bump occurs, the periodic
        :meth:`catch_up` tick is the safety net. proven by
        :func:`tests.unit.test_listener.TestEpochListenerRaceRecovery.
        test_catch_up_recovers_when_bump_lands_during_subscribe_window`.
        the alternative ordering (subscribe-first, prime-second)
        trades this race for one where a broadcast received
        between subscribe and prime advances last_seen ahead of the
        prime read; the prime would then write a STALER value over
        the live one. the current order keeps the failure mode
        recoverable via the documented pull-on-stale path.

        narrow exception scope: :class:`~threetears.nats.errors.
        SubscribeError` propagates because cache coherence is not
        optional. validation failures inside the typed dispatcher
        deadletter via the standard typed-NATS path.

        :param subject: subject to subscribe to; the subject's
            ``path`` is the dedupe key
        :ptype subject: Subject
        :param on_bump: async callback invoked on each monotonic
            bump with ``(new_epoch, payload)``
        :ptype on_bump: BumpCallback
        :return: nothing
        :rtype: None
        :raises SubscribeError: if the underlying NATS subscribe
            fails to register
        """
        primed = await self._epoch_client.current(subject)
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

            de-duplicates against last-seen on subject path; only
            invokes the consumer callback for strictly-increasing
            epochs.
            """
            current_last_seen = self._last_seen.get(subject.path, 0)
            if message.epoch <= current_last_seen:
                log.debug(
                    "epoch broadcast dropped (already seen)",
                    extra={
                        "extra_data": {
                            "subject": subject.path,
                            "incoming_epoch": message.epoch,
                            "last_seen": current_last_seen,
                        },
                    },
                )
                return
            self._last_seen[subject.path] = message.epoch
            await on_bump(message.epoch, message.payload)

        try:
            await self._nats.subscribe_typed(
                subject=subject,
                message_type=EpochBumpMessage,
                cb=_on_bump,
            )
        except SubscribeError:
            raise

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

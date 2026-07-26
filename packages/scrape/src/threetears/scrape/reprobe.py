"""Book a walled target's next probe as a scheduled job, for a caller that does not poll.

A polling caller needs nothing here: its next poll already IS the re-probe, and
``blocked_until`` on the health row decides whether that poll fetches anything. An
event-driven caller has no such tick, so it needs something to wake it when the backoff
window expires -- and the answer to that is a job in ``3tears-scheduled-jobs``, whose
``relative_delay`` schedule type ("fire once, this long from now") is the exact shape, and
whose tick engine is already cross-pod locked, missed-fire aware, and audited. Writing a
sleep-and-retry loop next to it would be re-solving all of that, worse.

**This module is behind the ``reprobe`` extra**, and imported by nothing else in the package.
``3tears-scheduled-jobs`` pulls NATS and APScheduler behind it, which is real weight for a
capability most consumers of a scraping library will never switch on, so
:class:`~threetears.scrape.circuit.TargetCircuit` depends on the two-method
:class:`~threetears.scrape.circuit.ReprobeScheduler` Protocol instead (book and cancel) and
this satisfies it.
Import this module only when you have installed ``3tears-scrape[reprobe]``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from threetears.observe import get_logger
from threetears.scheduled_jobs.collections import ScheduledJobCollection
from threetears.scheduled_jobs.entities import ScheduledJobEntity

__all__ = [
    "REPROBE_JOB_KIND",
    "ScheduledJobsReprobeScheduler",
    "reprobe_job_id",
]

log = get_logger(__name__)

#: ``kind`` discriminator on the booked job. The scheduled-jobs tick engine is
#: payload-agnostic and routes on this, so a consuming application dispatches it to whatever
#: re-fetches a target.
REPROBE_JOB_KIND = "scrape.target_reprobe"

#: Namespace for the deterministic job and partition ids below. A fixed URL namespace, so the
#: ids are reproducible from a target id alone in any process, without a table to look them
#: up in.
_REPROBE_NAMESPACE = uuid5(NAMESPACE_URL, "https://3tears.dev/scrape/target-reprobe")

#: Floor on a booked delay. ``relative_delay`` config is an integer with a unit suffix, so a
#: sub-second delay would round to ``"0s"`` and fire immediately -- which is the one thing a
#: backoff must never do.
_MIN_DELAY_SECONDS = 1


def reprobe_job_id(target_id: str) -> UUID:
    """The stable job id for *target_id*'s re-probe.

    Derived from the target rather than generated, so re-booking REPLACES the outstanding
    job instead of adding one. A blocked target is re-booked on every failed probe, and with
    a random id each of those would leave its predecessor behind: a target walled for a week
    would accumulate a queue of jobs that all eventually fire, turning a backoff into a
    burst at exactly the moment the backoff was longest.

    :param target_id: the target being re-probed
    :ptype target_id: str
    :return: a deterministic UUID for that target's re-probe job
    :rtype: UUID
    """
    return uuid5(_REPROBE_NAMESPACE, target_id)


class ScheduledJobsReprobeScheduler:
    """Satisfies ``ReprobeScheduler`` by writing a ``relative_delay`` job.

    :param job_collection: the scheduled-jobs store to write into
    :ptype job_collection: ScheduledJobCollection
    :param kind: ``kind`` discriminator the consuming dispatcher routes on
    :ptype kind: str
    :param partition_key: partition every re-probe job lands in. Defaults to one derived
        from the target, spreading targets across partitions; pass an explicit value to keep
        a deployment's scrape jobs together in a partition it already owns.
    :ptype partition_key: UUID | None
    """

    def __init__(
        self,
        job_collection: ScheduledJobCollection,
        *,
        kind: str = REPROBE_JOB_KIND,
        partition_key: UUID | None = None,
    ) -> None:
        self._jobs = job_collection
        self._kind = kind
        self._partition_key = partition_key

    async def schedule_reprobe(self, *, target_id: str, delay_seconds: float) -> None:
        """Book *target_id* to be re-probed in *delay_seconds*.

        ``missed_fire_policy`` is ``"coalesce"``: a dispatcher that was down through the
        window should probe once when it comes back, not once per window it slept through.

        :param target_id: the target to re-probe
        :ptype target_id: str
        :param delay_seconds: how long from now, floored at one second
        :ptype delay_seconds: float
        :return: nothing
        :rtype: None
        """
        delay = max(_MIN_DELAY_SECONDS, int(round(delay_seconds)))
        job_id = reprobe_job_id(target_id)
        partition_key = self._partition_key or uuid5(_REPROBE_NAMESPACE, f"partition:{target_id}")
        now = datetime.now(UTC)
        row: dict[str, Any] = {
            "partition_key": partition_key,
            "job_id": job_id,
            "kind": self._kind,
            "payload": {"target_id": target_id},
            "schedule_type": "relative_delay",
            "schedule_config": {"delay": f"{delay}s"},
            "status": "active",
            "next_fire_at": now + timedelta(seconds=delay),
            "last_fired_at": None,
            "missed_fire_policy": "coalesce",
            "name": f"scrape re-probe {target_id}",
            # Explicit, and load-bearing. `save_entity` stamps `date_created` for a new
            # entity but only stamps `date_updated` when the key is already present or the
            # entity is not new -- neither holds for a hand-built row like this one. The
            # column is `TIMESTAMPTZ NOT NULL` with a server DEFAULT, but the default cannot
            # save it: the upsert binds every column positionally by design (omitting one
            # would change the arity and break the SQL), so a missing key is bound as an
            # explicit NULL and the constraint fires. Because `_book_reprobe` swallows and
            # logs, the result was not a loud failure but a silent one -- an event-driven
            # deployment booking no re-probes whatsoever.
            "date_updated": now,
        }
        await self._jobs.save_entity(ScheduledJobEntity(row, is_new=True, collection=self._jobs))
        log.info(
            "scrape circuit: booked a re-probe of target %s in %ds",
            target_id,
            delay,
            extra={"extra_data": {"target_id": target_id, "job_id": str(job_id)}},
        )

    async def cancel_reprobe(self, *, target_id: str) -> None:
        """Delete *target_id*'s outstanding re-probe booking, if it has one.

        Deleting rather than marking it expired, because the row carries no information once
        the target has recovered: the health row is the durable record of what happened, and
        a fired one-shot would otherwise sit at ``status="expired"`` for every target that has
        ever tripped. The job id is derived from the target, so this addresses whichever
        booking is currently outstanding without needing to have kept a handle on it.

        Safe when there is nothing to delete: ``Collection.delete`` is idempotent across tiers
        and documents itself as returning ``True`` unconditionally, so a booking that was
        never made costs one no-op rather than an exception. A caller closing a circuit does
        not know whether a booking was outstanding, and asking first would be a round trip to
        answer what the delete already handles.

        That same contract is why nothing is logged at INFO here. The return value cannot
        distinguish "cancelled a real booking" from "there was nothing to cancel", and
        ``record_reachable`` calls this on every close -- so an INFO line would announce a
        cancellation for the many targets that never tripped at all. DEBUG says what is
        actually known: a delete was issued.

        :param target_id: the target whose booking should be dropped
        :ptype target_id: str
        :return: nothing
        :rtype: None
        """
        job_id = reprobe_job_id(target_id)
        partition_key = self._partition_key or uuid5(_REPROBE_NAMESPACE, f"partition:{target_id}")
        await self._jobs.delete((partition_key, job_id))
        log.debug(
            "scrape circuit: cleared any outstanding re-probe booking for target %s",
            target_id,
            extra={"extra_data": {"target_id": target_id, "job_id": str(job_id)}},
        )

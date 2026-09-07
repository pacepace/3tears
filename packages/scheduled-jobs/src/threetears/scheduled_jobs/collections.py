"""Default-store collections -- three-tier CRUD for the generic tables.

Generalized from agent-wake's ``WakeScheduleCollection`` /
``WakeFireCollection``. Like agent-wake, these subclass
:class:`BaseCollection` directly (not :class:`SchemaBackedCollection`)
because the ``payload`` / ``schedule_config`` / ``output`` JSONB columns
need handling beyond the schema-driven CRUD generator's built-in tags;
asyncpg's native codecs round-trip ``dict[str, Any] <-> JSONB`` cleanly.

Both Collections declare ``partition_column = "partition_key"`` as a
class attribute so consumers (and partition-column enforcement walkers)
can confirm the partition contract by introspection.

:class:`ScheduledJobCollection` implements
:class:`~threetears.scheduled_jobs.protocols.ScheduleStore`;
:class:`JobFireCollection` implements
:class:`~threetears.scheduled_jobs.protocols.FireStore`. The tick engine
talks to those Protocols, not to these classes.

The multi-row scan (``list_due_for_tick``) carries an explicit
``__SPANS_PARTITIONS__`` marker comment because it is a global,
cross-partition enumeration of "what fires next" -- the partition
predicate cannot apply to it by construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, ClassVar, get_args
from uuid import UUID

from threetears.core.collections.base import BaseCollection
from threetears.core.serialization import (
    deserialize_from_json,
    serialize_to_json,
)
from threetears.observe import get_logger

from threetears.scheduled_jobs.entities import JobFireEntity, ScheduledJobEntity
from threetears.scheduled_jobs.reschedule import compute_next_fire_at
from threetears.scheduled_jobs.types import ScheduleStatus

__all__ = [
    "REAPED_DISPATCH_ERROR",
    "JobFireCollection",
    "ScheduledJobCollection",
]


log = get_logger(__name__)


# Derived from the Literal rather than restated, so a status added to the
# vocabulary cannot be silently rejected by the validation in front of it.
_SCHEDULE_STATUSES: frozenset[str] = frozenset(get_args(ScheduleStatus))


# Field-type hints used when L2 cache rounds a row through JSON. The
# helpers in ``threetears.core.serialization.deserialize_from_json``
# dispatch on these to rehydrate UUID / datetime / dict values back to
# their native Python types after a NATS KV pull.
_JOB_FIELD_TYPES: dict[str, Any] = {
    "partition_key": UUID,
    "job_id": UUID,
    "kind": str,
    "payload": dict,
    "schedule_type": str,
    "schedule_config": dict,
    "status": str,
    "next_fire_at": datetime | None,
    "last_fired_at": datetime | None,
    "missed_fire_policy": str,
    "name": str | None,
    "date_created": datetime,
    "date_updated": datetime,
}


_FIRE_FIELD_TYPES: dict[str, Any] = {
    "partition_key": UUID,
    "fire_id": UUID,
    "job_id": UUID,
    "scheduled_fire_at": datetime,
    "actual_fired_at": datetime,
    "status": str,
    "output": dict | None,
    "latency_ms": int | None,
    "error": str | None,
    "date_created": datetime,
}


# columns the Collection emits on INSERT, in positional-param order.
_JOB_INSERT_COLUMNS: tuple[str, ...] = (
    "partition_key",
    "job_id",
    "kind",
    "payload",
    "schedule_type",
    "schedule_config",
    "status",
    "next_fire_at",
    "last_fired_at",
    "missed_fire_policy",
    "name",
    "date_created",
    "date_updated",
)


# columns updated on ON CONFLICT (partition + pk excluded;
# ``date_created`` immutable).
_JOB_UPDATE_COLUMNS: tuple[str, ...] = tuple(
    c for c in _JOB_INSERT_COLUMNS if c not in {"partition_key", "job_id", "date_created"}
)


_FIRE_INSERT_COLUMNS: tuple[str, ...] = (
    "partition_key",
    "fire_id",
    "job_id",
    "scheduled_fire_at",
    "actual_fired_at",
    "status",
    "output",
    "latency_ms",
    "error",
    "date_created",
)


_FIRE_UPDATE_COLUMNS: tuple[str, ...] = (
    "status",
    "output",
    "latency_ms",
    "error",
)


def _build_upsert_sql(
    table: str,
    insert_cols: Sequence[str],
    update_cols: Sequence[str],
    pk_cols: Sequence[str],
) -> str:
    """Build a parameterised INSERT ... ON CONFLICT DO UPDATE statement.

    Positional parameters bind to ``insert_cols`` in declared order.

    :param table: table name
    :ptype table: str
    :param insert_cols: columns to write on INSERT (in positional-param
        order)
    :ptype insert_cols: Sequence[str]
    :param update_cols: columns to update on conflict
    :ptype update_cols: Sequence[str]
    :param pk_cols: conflict target columns
    :ptype pk_cols: Sequence[str]
    :return: SQL string ready for ``execute()``
    :rtype: str
    """
    placeholders = ", ".join(f"${i + 1}" for i in range(len(insert_cols)))
    column_list = ", ".join(insert_cols)
    pk_list = ", ".join(pk_cols)
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    return (
        f"INSERT INTO {table} ({column_list}) VALUES ({placeholders}) "
        f"ON CONFLICT ({pk_list}) DO UPDATE SET {set_clause}"
    )


_SCHEDULED_JOBS_UPSERT_SQL = _build_upsert_sql(
    "scheduled_jobs",
    _JOB_INSERT_COLUMNS,
    _JOB_UPDATE_COLUMNS,
    ("partition_key", "job_id"),
)


_JOB_FIRES_UPSERT_SQL = _build_upsert_sql(
    "job_fires",
    _FIRE_INSERT_COLUMNS,
    _FIRE_UPDATE_COLUMNS,
    ("partition_key", "fire_id"),
)


_SCHEDULED_JOBS_FETCH_SQL = (
    "SELECT partition_key, job_id, kind, payload, schedule_type, "
    "schedule_config, status, next_fire_at, last_fired_at, "
    "missed_fire_policy, name, date_created, date_updated "
    "FROM scheduled_jobs WHERE partition_key = $1 AND job_id = $2"
)


_SCHEDULED_JOBS_DELETE_SQL = "DELETE FROM scheduled_jobs WHERE partition_key = $1 AND job_id = $2"


_JOB_FIRES_FETCH_SQL = (
    "SELECT partition_key, fire_id, job_id, scheduled_fire_at, "
    "actual_fired_at, status, output, latency_ms, error, date_created "
    "FROM job_fires WHERE partition_key = $1 AND fire_id = $2"
)


_JOB_FIRES_DELETE_SQL = "DELETE FROM job_fires WHERE partition_key = $1 AND fire_id = $2"


# Terminal error text stamped onto a fire row the reaper reclaims. Names
# the cause so the loss is self-explanatory in fire history.
REAPED_DISPATCH_ERROR: str = "reaped: dispatch abandoned in 'dispatching' (pod died mid-dispatch)"


# the column list reused by every multi-row scheduled_jobs read.
_SCHEDULED_JOBS_SELECT_COLUMNS = (
    "partition_key, job_id, kind, payload, schedule_type, "
    "schedule_config, status, next_fire_at, last_fired_at, "
    "missed_fire_policy, name, date_created, date_updated"
)


class ScheduledJobCollection(BaseCollection[ScheduledJobEntity]):
    """Three-tier collection for :class:`ScheduledJobEntity`.

    Implements :class:`~threetears.scheduled_jobs.protocols.ScheduleStore`.
    Composite primary key ``(partition_key, job_id)``; partition column
    ``partition_key``. Standalone ``UNIQUE (job_id)`` lets
    ``job_fires.job_id`` reference the bare id.
    """

    primary_key_column: str | tuple[str, ...] = ("partition_key", "job_id")

    partition_column: ClassVar[str] = "partition_key"

    @property
    def table_name(self) -> str:
        """Return the L3 table name."""
        return "scheduled_jobs"

    @property
    def entity_class(self) -> type[ScheduledJobEntity]:
        """Return the entity class."""
        return ScheduledJobEntity

    # --- BaseCollection contract ---

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """Fetch a row by composite pk.

        :param entity_id: tuple ``(partition_key, job_id)``
        :ptype entity_id: Any
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        partition_key, job_id = self.normalize_pk(entity_id)
        row = await self.l3_pool.fetchrow(_SCHEDULED_JOBS_FETCH_SQL, partition_key, job_id)
        if row is None:
            return None
        return dict(row)

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: Any = None,
        *,
        conn: Any = None,
    ) -> int:
        """Upsert a job row.

        :param data: row dict keyed by column name; must carry both pk
            columns and every non-nullable column
        :ptype data: dict[str, Any]
        :param original_timestamp: ignored (no CAS fence -- job updates
            are idempotent in the upsert path)
        :ptype original_timestamp: Any
        :param conn: optional asyncpg-compatible connection
        :ptype conn: Any
        :return: rows affected (1 on success)
        :rtype: int
        """
        del original_timestamp
        params = _job_insert_params(data)
        target = conn if conn is not None else self.l3_pool
        if target is None:
            return 0
        await target.execute(_SCHEDULED_JOBS_UPSERT_SQL, *params)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        """Delete a job row by composite pk.

        ``ON DELETE CASCADE`` on ``job_fires.job_id`` ensures every fire
        history row for the deleted job is removed in the same
        transaction.

        :param entity_id: tuple ``(partition_key, job_id)``
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return None
        partition_key, job_id = self.normalize_pk(entity_id)
        await self.l3_pool.execute(_SCHEDULED_JOBS_DELETE_SQL, partition_key, job_id)
        return None

    def serialize(self, data: dict[str, Any]) -> bytes:
        """Encode a row dict for L2 (NATS KV) storage."""
        return serialize_to_json(data)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Decode L2-cached bytes back to a row dict."""
        return deserialize_from_json(data, _JOB_FIELD_TYPES)

    # --- ScheduleStore protocol methods ---

    async def list_due_for_tick(
        self,
        now: datetime,
        *,
        kinds: Sequence[str],
        limit: int = 200,
    ) -> list[ScheduledJobEntity]:
        """Return active jobs of ``kinds`` whose ``next_fire_at <= now``.

        Used by the tick engine to scope each tick's work. Cross-partition
        scan -- the partition predicate cannot apply by construction; the
        ``__SPANS_PARTITIONS__`` marker documents the deliberate
        exemption.

        The ``kind`` filter is a SQL predicate, deliberately NOT a Python
        filter over a kind-blind result set: the scan is the one place a
        forgotten filter cannot hide, and the failure mode it prevents --
        a row reaching a handler registered for some other kind -- is
        silent. An empty ``kinds`` returns nothing rather than widening
        to every kind.

        :param now: tick instant; rows with ``next_fire_at <= now`` are
            returned
        :ptype now: datetime
        :param kinds: the routed kinds to scan for
        :ptype kinds: Sequence[str]
        :param limit: per-tick cap (defaults to 200)
        :ptype limit: int
        :return: list of due jobs ordered by ``next_fire_at`` ASC
        :rtype: list[ScheduledJobEntity]
        """
        if self.l3_pool is None or not kinds:
            return []
        # __SPANS_PARTITIONS__: the tick engine enumerates ready jobs
        # across every partition; the partition predicate cannot apply
        # here by construction (same shape as agent-wake's
        # list_due_for_tick cross-conversation scan). The column list is
        # written out literally (not interpolated) so the
        # partition-column enforcement walker sees ``partition_key`` as a
        # static literal -- mirroring agent-wake's scan, where the gate
        # is satisfied by the partition column appearing in the SELECT.
        rows = await self.l3_pool.fetch(
            "SELECT partition_key, job_id, kind, payload, schedule_type, "
            "schedule_config, status, next_fire_at, last_fired_at, "
            "missed_fire_policy, name, date_created, date_updated "
            "FROM scheduled_jobs "
            "WHERE status = 'active' AND next_fire_at IS NOT NULL AND next_fire_at <= $1 "
            "AND kind = ANY($2) "
            "ORDER BY next_fire_at ASC LIMIT $3",
            now,
            list(kinds),
            limit,
        )
        return [ScheduledJobEntity(dict(row), is_new=False, collection=self) for row in rows]

    async def list_for_partition(self, partition_key: UUID) -> list[ScheduledJobEntity]:
        """Return every job for a partition regardless of status.

        Used by admin/list surfaces. Partition-scoped, so it carries the
        partition predicate.

        :param partition_key: partition column value
        :ptype partition_key: UUID
        :return: list of jobs ordered by ``date_created`` DESC
        :rtype: list[ScheduledJobEntity]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: multi-row scan by partition_key is not pk-
        # addressable; L1 row cache cannot serve.
        rows = await self.l3_pool.fetch(
            f"SELECT {_SCHEDULED_JOBS_SELECT_COLUMNS} "  # noqa: S608 - trusted column-list constant, no user input
            "FROM scheduled_jobs WHERE partition_key = $1 "
            "ORDER BY date_created DESC",
            partition_key,
        )
        return [ScheduledJobEntity(dict(row), is_new=False, collection=self) for row in rows]

    async def list_jobs(
        self,
        *,
        kinds: Sequence[str] | None = None,
        statuses: Sequence[str] | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[ScheduledJobEntity]:
        """Return jobs across every partition, for an operator-facing view.

        Distinct from :meth:`list_due_for_tick`, which the engine uses and
        which admits ``status='active'`` and a reached ``next_fire_at``
        only. This one is status-blind by default, because a paused or
        expired job is usually the one an operator is looking for.

        Distinct from :meth:`list_for_partition` because every job may
        carry its OWN ``partition_key`` -- a seeder that mints one
        partition per schedule is the normal shape, so a
        partition-scoped read enumerates a single job rather than a
        deployment.

        ``None`` means unfiltered; an empty sequence matches NOTHING, the
        same contract :meth:`list_due_for_tick` applies to ``kinds``.
        Widening an explicit empty list to "everything" would turn a
        caller that computed zero kinds into one operating on the whole
        deployment.

        Ordering is total (``kind``, then ``name``, then ``job_id``) so
        paging with ``offset`` cannot repeat or skip a row.

        :param kinds: restrict to these kinds, or ``None`` for every kind
        :ptype kinds: Sequence[str] | None
        :param statuses: restrict to these statuses, or ``None`` for every
            status
        :ptype statuses: Sequence[str] | None
        :param limit: page size
        :ptype limit: int
        :param offset: rows to skip, for paging
        :ptype offset: int
        :return: matching jobs, ordered deterministically
        :rtype: list[ScheduledJobEntity]
        """
        result: list[ScheduledJobEntity] = []
        if self.l3_pool is not None and kinds != [] and statuses != []:
            # __SPANS_PARTITIONS__: an admin view enumerates a deployment,
            # and each job may hold its own partition, so the partition
            # predicate cannot apply here by construction -- the same
            # exemption list_due_for_tick carries, for the same reason.
            # The column list is written out literally (not interpolated)
            # so the partition-column enforcement walker sees
            # ``partition_key`` as a static literal.
            rows = await self.l3_pool.fetch(
                "SELECT partition_key, job_id, kind, payload, schedule_type, "
                "schedule_config, status, next_fire_at, last_fired_at, "
                "missed_fire_policy, name, date_created, date_updated "
                "FROM scheduled_jobs "
                "WHERE ($1::text[] IS NULL OR kind = ANY($1)) "
                "AND ($2::text[] IS NULL OR status = ANY($2)) "
                "ORDER BY kind ASC, name ASC NULLS LAST, job_id ASC "
                "LIMIT $3 OFFSET $4",
                list(kinds) if kinds is not None else None,
                list(statuses) if statuses is not None else None,
                limit,
                offset,
            )
            result = [ScheduledJobEntity(dict(row), is_new=False, collection=self) for row in rows]
        return result

    async def get_by_job_id(self, job_id: UUID) -> ScheduledJobEntity | None:
        """Fetch one job by its bare ``job_id``, without knowing its partition.

        ``job_id`` carries its own UNIQUE constraint (it is what
        ``job_fires.job_id`` references), so it addresses a row on its
        own. An admin surface has only that id -- it comes out of a URL
        or an operator's clipboard, not out of a partition-aware
        listing -- and the partition is what this returns so the caller
        can then use the partition-scoped mutators.

        Prefer :meth:`get` wherever the partition IS known: it is
        partition-scoped and cache-addressable, and this is neither.

        :param job_id: the job's unique id
        :ptype job_id: UUID
        :return: the job, or ``None`` when no row carries that id
        :rtype: ScheduledJobEntity | None
        """
        result: ScheduledJobEntity | None = None
        if self.l3_pool is not None:
            # __SPANS_PARTITIONS__: resolving a bare job_id to its row is a
            # lookup ACROSS partitions by definition -- the partition is the
            # thing being looked up. Bounded to one row by the column's own
            # UNIQUE constraint. The column list is written out literally so
            # the partition-column enforcement walker sees ``partition_key``
            # as a static literal.
            row = await self.l3_pool.fetchrow(
                "SELECT partition_key, job_id, kind, payload, schedule_type, "
                "schedule_config, status, next_fire_at, last_fired_at, "
                "missed_fire_policy, name, date_created, date_updated "
                "FROM scheduled_jobs WHERE job_id = $1",
                job_id,
            )
            if row is not None:
                result = ScheduledJobEntity(dict(row), is_new=False, collection=self)
        return result

    async def count_jobs(
        self,
        *,
        kinds: Sequence[str] | None = None,
        statuses: Sequence[str] | None = None,
    ) -> int:
        """Count jobs matching the same filters :meth:`list_jobs` applies.

        Exists so a paging caller can size the result set without
        fetching it. The filter semantics are deliberately identical:
        a count computed under different rules than the listing would
        mis-page whoever trusted it.

        :param kinds: restrict to these kinds, or ``None`` for every kind
        :ptype kinds: Sequence[str] | None
        :param statuses: restrict to these statuses, or ``None`` for every
            status
        :ptype statuses: Sequence[str] | None
        :return: number of matching jobs
        :rtype: int
        """
        total = 0
        if self.l3_pool is not None and kinds != [] and statuses != []:
            # __SPANS_PARTITIONS__: counts what list_jobs would return, so
            # it carries the same cross-partition exemption.
            counted = await self.l3_pool.fetchval(
                "SELECT count(*) FROM scheduled_jobs "
                "WHERE ($1::text[] IS NULL OR kind = ANY($1)) "
                "AND ($2::text[] IS NULL OR status = ANY($2))",
                list(kinds) if kinds is not None else None,
                list(statuses) if statuses is not None else None,
            )
            total = int(counted or 0)
        return total

    async def set_status(
        self,
        *,
        partition_key: UUID,
        job_id: UUID,
        status: str,
        now: datetime,
    ) -> bool:
        """Move one job between ``active`` / ``paused`` / ``expired``.

        Writes ``status`` and ``date_updated`` and NOTHING ELSE. Leaving
        ``next_fire_at`` alone is the property the whole pause/resume
        story rests on: resuming restores the schedule the operator
        paused rather than restarting it from now. A backlog that built
        up while paused is then the missed-fire policy's business, which
        is where that decision already lives.

        Pausing is the supported way to stop a schedule; deleting the row
        is not, because a boot-time insert-if-absent ensure recreates a
        deleted row and will not revive a paused one.

        :param partition_key: partition column value
        :ptype partition_key: UUID
        :param job_id: target job
        :ptype job_id: UUID
        :param status: one of the values pinned by
            :data:`threetears.scheduled_jobs.types.ScheduleStatus`
        :ptype status: str
        :param now: instant written as ``date_updated``
        :ptype now: datetime
        :return: ``True`` when a row changed, ``False`` when none matched
        :rtype: bool
        :raises ValueError: when ``status`` is not a known status
        """
        if status not in _SCHEDULE_STATUSES:
            msg = f"unknown status {status!r}; expected one of {sorted(_SCHEDULE_STATUSES)}"
            raise ValueError(msg)
        changed = None
        if self.l3_pool is not None:
            # cache-bypass: targeted UPDATE; the row cache is invalidated
            # naturally on the next partition-aware fetch.
            changed = await self.l3_pool.fetchval(
                "UPDATE scheduled_jobs SET status = $1, date_updated = $2 "
                "WHERE partition_key = $3 AND job_id = $4 "
                "RETURNING job_id",
                status,
                now,
                partition_key,
                job_id,
            )
        return changed is not None

    async def update_schedule(
        self,
        *,
        partition_key: UUID,
        job_id: UUID,
        schedule_config: dict[str, Any],
        now: datetime,
    ) -> bool:
        """Retune one job's cadence and move its next fire to match.

        Writing ``schedule_config`` alone would leave the OLD cadence's
        ``next_fire_at`` standing, so the change would not take effect
        until that fire came and went. The new next fire is computed by
        :func:`~threetears.scheduled_jobs.reschedule.compute_next_fire_at`
        -- the same function the tick engine reschedules with, so a
        retune cannot drift from the engine's own maths, and a config the
        schedule type rejects raises here rather than being stored and
        failing at fire time.

        ``schedule_type`` and ``missed_fire_policy`` are read from the
        ROW, never taken from the caller: a caller-supplied type could
        disagree with the stored one and write a next fire the engine
        would never have produced.

        :param partition_key: partition column value
        :ptype partition_key: UUID
        :param job_id: target job
        :ptype job_id: UUID
        :param schedule_config: the new per-schedule-type config payload
        :ptype schedule_config: dict[str, Any]
        :param now: instant the recompute anchors on, and ``date_updated``
        :ptype now: datetime
        :return: ``True`` when a row changed, ``False`` when none matched
        :rtype: bool
        :raises ValueError: when the config is malformed for the row's
            schedule type
        """
        updated = False
        if self.l3_pool is not None:
            # cache-bypass: reads the discriminators the recompute needs
            # before a targeted UPDATE; the row cache cannot answer the
            # read-then-write as one operation.
            row = await self.l3_pool.fetchrow(
                "SELECT schedule_type, missed_fire_policy, last_fired_at "
                "FROM scheduled_jobs WHERE partition_key = $1 AND job_id = $2",
                partition_key,
                job_id,
            )
            if row is not None:
                next_fire_at = compute_next_fire_at(
                    row["schedule_type"],
                    schedule_config,
                    row["missed_fire_policy"],
                    row["last_fired_at"],
                    now,
                )
                changed = await self.l3_pool.fetchval(
                    "UPDATE scheduled_jobs "
                    "SET schedule_config = $1, next_fire_at = $2, date_updated = $3 "
                    "WHERE partition_key = $4 AND job_id = $5 "
                    "RETURNING job_id",
                    schedule_config,
                    next_fire_at,
                    now,
                    partition_key,
                    job_id,
                )
                updated = changed is not None
        return updated

    async def request_immediate_fire(
        self,
        *,
        partition_key: UUID,
        job_id: UUID,
        now: datetime,
    ) -> bool:
        """Bring an active job's next fire forward to ``now``.

        A request, not a fire: the tick engine still claims the row
        through its own CAS on the next pass, so this cannot double-fire
        a job or bypass the engine's concurrency control.

        Refuses a job that is not ``active``. Firing a schedule an
        operator believes is stopped is the wrong default -- the caller
        resumes it first, which is a decision someone takes deliberately
        rather than a side effect of asking for one run.

        :param partition_key: partition column value
        :ptype partition_key: UUID
        :param job_id: target job
        :ptype job_id: UUID
        :param now: the instant to bring the fire forward to
        :ptype now: datetime
        :return: ``True`` when the job was brought forward, ``False`` when
            no active row matched
        :rtype: bool
        """
        requested = None
        if self.l3_pool is not None:
            # cache-bypass: targeted UPDATE guarded on status; the row
            # cache is invalidated naturally on the next partition-aware
            # fetch.
            requested = await self.l3_pool.fetchval(
                "UPDATE scheduled_jobs SET next_fire_at = $1, date_updated = $1 "
                "WHERE partition_key = $2 AND job_id = $3 AND status = 'active' "
                "RETURNING job_id",
                now,
                partition_key,
                job_id,
            )
        return requested is not None

    async def claim_and_reschedule(
        self,
        *,
        partition_key: UUID,
        job_id: UUID,
        expected_next_fire: datetime,
        computed_next_fire: datetime | None,
        new_status: str,
        now: datetime,
    ) -> bool:
        """Atomically claim a due job and advance its ``next_fire_at``.

        Optimistic-CAS UPDATE: the predicate ``next_fire_at =
        expected_next_fire`` ensures exactly one tick wins. Returns
        ``True`` on a successful claim, ``False`` when another tick has
        already advanced the row. The UPDATE stamps ``last_fired_at = now``
        and rewrites ``status``.

        :param partition_key: partition column value
        :ptype partition_key: UUID
        :param job_id: target job
        :ptype job_id: UUID
        :param expected_next_fire: the CAS predicate value
        :ptype expected_next_fire: datetime
        :param computed_next_fire: the new ``next_fire_at`` (``None`` for
            terminal one-shots)
        :ptype computed_next_fire: datetime | None
        :param new_status: the new ``status`` value
        :ptype new_status: str
        :param now: tick instant; written as ``last_fired_at`` /
            ``date_updated``
        :ptype now: datetime
        :return: ``True`` on successful claim, ``False`` on CAS miss
        :rtype: bool
        """
        if self.l3_pool is None:
            return False
        # cache-bypass: atomic CAS UPDATE; the row cache is invalidated
        # naturally on the next partition-aware fetch.
        claimed = await self.l3_pool.fetchval(
            "UPDATE scheduled_jobs "
            "SET next_fire_at = $1, last_fired_at = $2, date_updated = $2, status = $3 "
            "WHERE partition_key = $4 AND job_id = $5 AND next_fire_at = $6 "
            "RETURNING job_id",
            computed_next_fire,
            now,
            new_status,
            partition_key,
            job_id,
            expected_next_fire,
        )
        return claimed is not None


class JobFireCollection(BaseCollection[JobFireEntity]):
    """Three-tier collection for :class:`JobFireEntity`.

    Implements :class:`~threetears.scheduled_jobs.protocols.FireStore`.
    Composite primary key ``(partition_key, fire_id)``; partition column
    ``partition_key``. Append-mostly: the in-flight row inserts with
    ``status='dispatching'``, then a finalize overwrites the terminal
    status.
    """

    primary_key_column: str | tuple[str, ...] = ("partition_key", "fire_id")

    partition_column: ClassVar[str] = "partition_key"

    @property
    def table_name(self) -> str:
        """Return the L3 table name."""
        return "job_fires"

    @property
    def entity_class(self) -> type[JobFireEntity]:
        """Return the entity class."""
        return JobFireEntity

    # --- BaseCollection contract ---

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """Fetch a row by composite pk."""
        if self.l3_pool is None:
            return None
        partition_key, fire_id = self.normalize_pk(entity_id)
        row = await self.l3_pool.fetchrow(_JOB_FIRES_FETCH_SQL, partition_key, fire_id)
        if row is None:
            return None
        return dict(row)

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: Any = None,
        *,
        conn: Any = None,
    ) -> int:
        """Upsert a fire row.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :param original_timestamp: ignored
        :ptype original_timestamp: Any
        :param conn: optional asyncpg-compatible connection
        :ptype conn: Any
        :return: rows affected
        :rtype: int
        """
        del original_timestamp
        params = _fire_insert_params(data)
        target = conn if conn is not None else self.l3_pool
        if target is None:
            return 0
        await target.execute(_JOB_FIRES_UPSERT_SQL, *params)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        """Delete a fire row by composite pk."""
        if self.l3_pool is None:
            return None
        partition_key, fire_id = self.normalize_pk(entity_id)
        await self.l3_pool.execute(_JOB_FIRES_DELETE_SQL, partition_key, fire_id)
        return None

    def serialize(self, data: dict[str, Any]) -> bytes:
        """Encode a row dict for L2 (NATS KV) storage."""
        return serialize_to_json(data)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Decode L2-cached bytes back to a row dict."""
        return deserialize_from_json(data, _FIRE_FIELD_TYPES)

    # --- FireStore protocol methods ---

    async def create_dispatching(
        self,
        *,
        fire_id: UUID,
        job_id: UUID,
        partition_key: UUID,
        scheduled_fire_at: datetime,
        actual_fired_at: datetime,
    ) -> None:
        """Insert an initial in-flight ``job_fires`` row (``status='dispatching'``).

        Called by the tick body immediately after a fire claim succeeds.
        The dispatch callback finalizes via :meth:`finalize_success` /
        :meth:`finalize_failed` which overwrite to the terminal status.
        The two-write pattern lets the audit trail capture that a fire was
        attempted even if the dispatcher crashes before producing output.

        :param fire_id: pre-generated id
        :ptype fire_id: UUID
        :param job_id: source job
        :ptype job_id: UUID
        :param partition_key: partition column value
        :ptype partition_key: UUID
        :param scheduled_fire_at: the planned fire time (the claimed row's
            ``next_fire_at``)
        :ptype scheduled_fire_at: datetime
        :param actual_fired_at: the actual tick instant
        :ptype actual_fired_at: datetime
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: write-path; the row cache is read-mostly and
        # invalidated naturally on the next fetch.
        await self.l3_pool.execute(
            "INSERT INTO job_fires "
            "(partition_key, fire_id, job_id, scheduled_fire_at, actual_fired_at, status) "
            "VALUES ($1, $2, $3, $4, $5, $6)",
            partition_key,
            fire_id,
            job_id,
            scheduled_fire_at,
            actual_fired_at,
            "dispatching",
        )
        return None

    async def finalize_success(
        self,
        partition_key: UUID,
        fire_id: UUID,
        *,
        status: str = "succeeded",
        output: dict[str, Any] | None = None,
        latency_ms: int | None = None,
    ) -> None:
        """Stamp a successful dispatch result onto the fire row.

        Idempotent: replaying the same finalization overwrites with the
        same values.

        :param partition_key: partition column value
        :ptype partition_key: UUID
        :param fire_id: target fire row
        :ptype fire_id: UUID
        :param status: terminal status (defaults to ``'succeeded'``)
        :ptype status: str
        :param output: captured output payload
        :ptype output: dict[str, Any] | None
        :param latency_ms: end-to-end fire latency
        :ptype latency_ms: int | None
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: targeted UPDATE on the terminal-state columns.
        await self.l3_pool.execute(
            "UPDATE job_fires SET status = $1, output = $2, latency_ms = $3 WHERE partition_key = $4 AND fire_id = $5",
            status,
            output,
            latency_ms,
            partition_key,
            fire_id,
        )
        return None

    async def finalize_failed(
        self,
        partition_key: UUID,
        fire_id: UUID,
        *,
        error: str,
        latency_ms: int | None = None,
    ) -> None:
        """Stamp a failed-dispatch result onto the fire row.

        Called by the tick body when the dispatch callback raises -- the
        per-job try/except keeps one bad fire from poisoning the rest of
        the tick.

        :param partition_key: partition column value
        :ptype partition_key: UUID
        :param fire_id: target fire row
        :ptype fire_id: UUID
        :param error: captured error message
        :ptype error: str
        :param latency_ms: latency up to the failure
        :ptype latency_ms: int | None
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: targeted UPDATE on the failure columns.
        await self.l3_pool.execute(
            "UPDATE job_fires SET status = 'failed', error = $1, latency_ms = $2 "
            "WHERE partition_key = $3 AND fire_id = $4",
            error,
            latency_ms,
            partition_key,
            fire_id,
        )
        return None

    async def latest_for_jobs(
        self,
        job_ids: Sequence[UUID],
    ) -> dict[UUID, JobFireEntity]:
        """Return the newest fire for each of ``job_ids``, in ONE query.

        An admin listing shows N jobs each with its last verdict, and
        calling :meth:`list_for_job` per row makes that N+1 round trips
        against a table that grows with every tick of every schedule.

        A job that has never fired is ABSENT from the result rather than
        mapped to ``None``: absence is already the caller's signal, and a
        null entry would need the same check plus a way to tell it from a
        fire whose fields happen to be empty.

        :param job_ids: the jobs whose latest fire to read
        :ptype job_ids: Sequence[UUID]
        :return: newest fire per job, keyed by ``job_id``, omitting jobs
            that have never fired
        :rtype: dict[UUID, JobFireEntity]
        """
        result: dict[UUID, JobFireEntity] = {}
        if self.l3_pool is not None and job_ids:
            # __SPANS_PARTITIONS__: the jobs an admin view lists may each
            # hold their own partition, so the partition predicate cannot
            # apply; the query is bounded by an explicit job-id list
            # instead. The column list is written out literally so the
            # partition-column enforcement walker sees ``partition_key``
            # as a static literal.
            rows = await self.l3_pool.fetch(
                "SELECT DISTINCT ON (job_id) "
                "partition_key, fire_id, job_id, scheduled_fire_at, "
                "actual_fired_at, status, output, latency_ms, error, date_created "
                "FROM job_fires WHERE job_id = ANY($1) "
                "ORDER BY job_id, actual_fired_at DESC",
                list(job_ids),
            )
            result = {row["job_id"]: JobFireEntity(dict(row), is_new=False, collection=self) for row in rows}
        return result

    async def reap_stale_dispatching(
        self,
        now: datetime,
        *,
        older_than: timedelta,
        kinds: Sequence[str],
    ) -> int:
        """Reap ``'dispatching'`` fire rows of ``kinds`` to ``'failed'``.

        Cross-partition sweep: a pod that dies after
        :meth:`create_dispatching` but before a finalize leaves a
        permanent ``'dispatching'`` zombie whose occurrence never
        re-fires (its schedule already advanced). This stamps every such
        row older than ``older_than`` to ``'failed'`` with
        :data:`REAPED_DISPATCH_ERROR`, making the loss visible in fire
        history + failure metrics instead of silent.

        ``job_fires`` carries no ``kind`` column of its own, so the scope
        resolves through the ``job_id`` the fire row already references
        (``scheduled_jobs.job_id`` is UNIQUE, which is why the subquery
        needs no partition join). Scoping to the caller's kinds is what
        lets two pumps with different reap thresholds share one table
        without either reclaiming the other's in-flight fires. An empty
        ``kinds`` sweeps nothing rather than widening to every kind.

        :param now: sweep instant; the cutoff is ``now - older_than``
        :ptype now: datetime
        :param older_than: minimum in-flight age before a row is reaped
        :ptype older_than: timedelta
        :param kinds: the kinds whose fire rows this sweep may reclaim
        :ptype kinds: Sequence[str]
        :return: number of rows reaped
        :rtype: int
        """
        if self.l3_pool is None or not kinds:
            return 0
        cutoff = now - older_than
        # __SPANS_PARTITIONS__: reclaiming abandoned in-flight fires is a
        # global sweep across every partition; the partition predicate
        # cannot apply by construction. ``partition_key`` is named in the
        # RETURNING clause so the partition-column enforcement walker sees
        # it as a static literal (same technique as list_due_for_tick).
        # cache-bypass: bulk terminal-state UPDATE across partitions; not
        # pk-addressable, so the L1 row cache cannot serve or invalidate
        # it -- the reaped rows re-materialize on the next targeted fetch.
        rows = await self.l3_pool.fetch(
            "UPDATE job_fires SET status = 'failed', error = $1 "
            "WHERE status = 'dispatching' AND actual_fired_at < $2 "
            "AND job_id IN (SELECT job_id FROM scheduled_jobs WHERE kind = ANY($3)) "
            "RETURNING partition_key, fire_id",
            REAPED_DISPATCH_ERROR,
            cutoff,
            list(kinds),
        )
        return len(rows)

    async def list_for_job(
        self,
        partition_key: UUID,
        job_id: UUID,
        *,
        limit: int = 20,
    ) -> list[JobFireEntity]:
        """List fires for a job, newest first.

        :param partition_key: partition column value
        :ptype partition_key: UUID
        :param job_id: job whose history to read
        :ptype job_id: UUID
        :param limit: page size
        :ptype limit: int
        :return: list of fire entities ordered by ``actual_fired_at`` DESC
        :rtype: list[JobFireEntity]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: per-job history scan.
        rows = await self.l3_pool.fetch(
            "SELECT partition_key, fire_id, job_id, scheduled_fire_at, "
            "actual_fired_at, status, output, latency_ms, error, date_created "
            "FROM job_fires "
            "WHERE partition_key = $1 AND job_id = $2 "
            "ORDER BY actual_fired_at DESC LIMIT $3",
            partition_key,
            job_id,
            limit,
        )
        return [JobFireEntity(dict(row), is_new=False, collection=self) for row in rows]


# --- helpers -----------------------------------------------------------


def _job_insert_params(data: dict[str, Any]) -> list[Any]:
    """Project a row dict onto the job upsert's positional params.

    Missing values are coerced to the schema's DEFAULT-equivalent so
    every column carries an explicit bound value (the upsert path writes
    every column unconditionally; relying on server defaults here would
    change the bound-values arity and break the SQL).

    :param data: row dict keyed by column name
    :ptype data: dict[str, Any]
    :return: positional parameter list in ``_JOB_INSERT_COLUMNS`` order
    :rtype: list[Any]
    """
    return [_job_value_for_column(col, data) for col in _JOB_INSERT_COLUMNS]


def _job_value_for_column(col: str, data: dict[str, Any]) -> Any:
    """Return ``data[col]`` with defaults applied for columns that have a
    DB-side default.

    :param col: column name
    :ptype col: str
    :param data: row dict
    :ptype data: dict[str, Any]
    :return: value to bind for this column
    :rtype: Any
    """
    if col in data:
        value = data[col]
        if col in {"payload", "schedule_config"} and value is None:
            return {}
        return value
    if col in {"payload", "schedule_config"}:
        return {}
    if col == "status":
        return "active"
    if col == "missed_fire_policy":
        return "coalesce"
    return None


def _fire_insert_params(data: dict[str, Any]) -> list[Any]:
    """Project a row dict onto the fire upsert's positional params.

    :param data: row dict keyed by column name
    :ptype data: dict[str, Any]
    :return: positional parameter list in ``_FIRE_INSERT_COLUMNS`` order
    :rtype: list[Any]
    """
    return [_fire_value_for_column(col, data) for col in _FIRE_INSERT_COLUMNS]


def _fire_value_for_column(col: str, data: dict[str, Any]) -> Any:
    """Return ``data[col]`` with defaults for omittable columns.

    :param col: column name
    :ptype col: str
    :param data: row dict
    :ptype data: dict[str, Any]
    :return: value to bind for this column
    :rtype: Any
    """
    if col in data:
        return data[col]
    if col == "status":
        return "dispatching"
    return None

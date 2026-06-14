"""Default-store entities -- cache proxies for the two generic tables.

Two entity classes, generalized from agent-wake's ``WakeScheduleEntity``
/ ``WakeFireEntity`` by collapsing the typed agent columns into an opaque
``kind`` (TEXT) + ``payload`` (JSONB):

- :class:`ScheduledJobEntity` -- one row per active scheduled job.
  Composite PK ``(partition_key, job_id)``; standalone ``UNIQUE
  (job_id)`` so ``job_fires.job_id`` can reference the bare id. Satisfies
  the :class:`~threetears.scheduled_jobs.protocols.DueSchedule` Protocol.
- :class:`JobFireEntity` -- one row per job fire. Composite PK
  ``(partition_key, fire_id)``.

``partition_key`` is the generic partition column (agent-wake's
``conversation_id`` equivalent). There is NO database-level FK on
``partition_key`` -- a consumer's partition referent is its own concern;
the column is a plain denormalised UUID. Each entity carries a tuple
``_id`` so ``BaseCollection.normalize_pk`` / ``l2_key`` address the row
uniformly across L1 / L2 / L3. ``primary_key_field`` names the bare-id
column so ``BaseEntity.id`` returns the singular UUID.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from threetears.core.entities.base import BaseEntity

__all__ = [
    "JobFireEntity",
    "ScheduledJobEntity",
]


def _as_uuid(value: object) -> UUID:
    """Coerce a value to stdlib ``UUID``.

    Cache + serialization layers may surface UUIDs as strings; this
    helper normalises every read path to a single shape. Raises a clear
    error when a non-nullable UUID field reads empty (a cache-coherence
    miss masquerades as a parse error otherwise).

    :param value: UUID-shaped input from any tier
    :ptype value: object
    :return: stdlib UUID
    :rtype: UUID
    :raises ValueError: when ``value`` is ``None`` (non-nullable field
        read empty)
    """
    if isinstance(value, UUID):
        return value
    if value is None:
        raise ValueError(
            "expected a UUID-shaped value but got None -- a non-nullable "
            "UUID field read empty (likely a cache-coherence miss, not a "
            "malformed UUID)"
        )
    return UUID(str(value))


class ScheduledJobEntity(BaseEntity):
    """Cache proxy entity for the ``scheduled_jobs`` table.

    Composite primary key ``(partition_key, job_id)``; the entity sets
    ``_id`` to the tuple form so the framework's pk-aware paths address
    rows uniformly across tiers. Satisfies the
    :class:`~threetears.scheduled_jobs.protocols.DueSchedule` Protocol.

    The ``kind`` discriminator + ``payload`` JSONB replace agent-wake's
    typed columns: a consumer routes on ``kind`` and reads ``payload``
    in its dispatch callback; the platform stores both opaquely.
    """

    primary_key_field: str = "job_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """Initialise with composite ``_id`` for tuple-pk lookup.

        :param data: row dict; ``partition_key`` and ``job_id`` must be
            present so the tuple ``_id`` can be assembled
        :ptype data: dict[str, Any]
        :param is_new: whether the entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        if "partition_key" in data and "job_id" in data:
            object.__setattr__(self, "_id", (data["partition_key"], data["job_id"]))

    @property
    def job_id(self) -> UUID:
        """Return the bare ``job_id`` UUID."""
        return _as_uuid(self._get_raw("job_id"))

    @property
    def partition_key(self) -> UUID:
        """Return the partition column ``partition_key``."""
        return _as_uuid(self._get_raw("partition_key"))

    @partition_key.setter
    def partition_key(self, value: UUID) -> None:
        """Set the partition column."""
        BaseEntity.__setattr__(self, "partition_key", value)

    @property
    def kind(self) -> str:
        """Return the opaque routing discriminator."""
        value: str = self._get_raw("kind")
        return value

    @kind.setter
    def kind(self, value: str) -> None:
        """Set the kind discriminator."""
        BaseEntity.__setattr__(self, "kind", value)

    @property
    def payload(self) -> dict[str, Any]:
        """Return the opaque per-job JSON payload.

        Empty-dict fallback on NULL so callers can ``.get(...)`` without
        a None-guard at every site.
        """
        value = self._get_raw("payload")
        if value is None:
            return {}
        return dict(value)

    @payload.setter
    def payload(self, value: dict[str, Any]) -> None:
        """Set the payload."""
        BaseEntity.__setattr__(self, "payload", dict(value))

    @property
    def schedule_type(self) -> str:
        """Return the ``schedule_type`` discriminator."""
        value: str = self._get_raw("schedule_type")
        return value

    @schedule_type.setter
    def schedule_type(self, value: str) -> None:
        """Set the schedule type."""
        BaseEntity.__setattr__(self, "schedule_type", value)

    @property
    def schedule_config(self) -> dict[str, Any]:
        """Return the schedule-type-specific config JSON.

        Empty-dict fallback on NULL so callers can ``.get(...)`` without
        a None-guard.
        """
        value = self._get_raw("schedule_config")
        if value is None:
            return {}
        return dict(value)

    @schedule_config.setter
    def schedule_config(self, value: dict[str, Any]) -> None:
        """Set the schedule config."""
        BaseEntity.__setattr__(self, "schedule_config", dict(value))

    @property
    def status(self) -> str:
        """Return the ``status`` value (``'active'`` | ``'paused'`` | ``'expired'``)."""
        value: str = self._get_raw("status")
        return value

    @status.setter
    def status(self, value: str) -> None:
        """Set the status (validated by DB CHECK)."""
        BaseEntity.__setattr__(self, "status", value)

    @property
    def next_fire_at(self) -> datetime | None:
        """Return the next scheduled fire time (``None`` when paused / expired)."""
        value: datetime | None = self._get_raw("next_fire_at")
        return value

    @next_fire_at.setter
    def next_fire_at(self, value: datetime | None) -> None:
        """Set the next_fire_at timestamp."""
        BaseEntity.__setattr__(self, "next_fire_at", value)

    @property
    def last_fired_at(self) -> datetime | None:
        """Return the timestamp of the most recent fire (``None`` for unfired)."""
        value: datetime | None = self._get_raw("last_fired_at")
        return value

    @last_fired_at.setter
    def last_fired_at(self, value: datetime | None) -> None:
        """Set the last_fired_at timestamp."""
        BaseEntity.__setattr__(self, "last_fired_at", value)

    @property
    def missed_fire_policy(self) -> str:
        """Return the ``missed_fire_policy`` value (``'coalesce'`` | ``'catch_up'``)."""
        value: str = self._get_raw("missed_fire_policy")
        return value

    @missed_fire_policy.setter
    def missed_fire_policy(self, value: str) -> None:
        """Set the missed-fire policy."""
        BaseEntity.__setattr__(self, "missed_fire_policy", value)

    @property
    def name(self) -> str | None:
        """Return the optional human-readable schedule name."""
        value: str | None = self._get_raw("name")
        return value

    @name.setter
    def name(self, value: str | None) -> None:
        """Set the schedule name."""
        BaseEntity.__setattr__(self, "name", value)

    @property
    def date_created(self) -> datetime:
        """Return the creation timestamp."""
        value: datetime = self._get_raw("date_created")
        return value

    @date_created.setter
    def date_created(self, value: datetime) -> None:
        """Set the creation timestamp."""
        BaseEntity.__setattr__(self, "date_created", value)

    @property
    def date_updated(self) -> datetime:
        """Return the last-modified timestamp."""
        value: datetime = self._get_raw("date_updated")
        return value

    @date_updated.setter
    def date_updated(self, value: datetime) -> None:
        """Set the last-modified timestamp."""
        BaseEntity.__setattr__(self, "date_updated", value)


class JobFireEntity(BaseEntity):
    """Cache proxy entity for the ``job_fires`` table.

    Composite primary key ``(partition_key, fire_id)``; partition column
    ``partition_key``. One row per job fire. Append-mostly: the in-flight
    row lands ``status='dispatching'`` and is overwritten once to the
    terminal status (``'succeeded'`` / ``'failed'``).

    The ``job_id`` FK uses the standalone ``UNIQUE (job_id)`` constraint
    on ``scheduled_jobs``; deleting a job cascades and removes its fire
    history.
    """

    primary_key_field: str = "fire_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """Initialise with composite ``_id`` for tuple-pk lookup.

        :param data: row dict; ``partition_key`` and ``fire_id`` must be
            present
        :ptype data: dict[str, Any]
        :param is_new: whether the entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        if "partition_key" in data and "fire_id" in data:
            object.__setattr__(self, "_id", (data["partition_key"], data["fire_id"]))

    @property
    def fire_id(self) -> UUID:
        """Return the bare ``fire_id`` UUID."""
        return _as_uuid(self._get_raw("fire_id"))

    @property
    def partition_key(self) -> UUID:
        """Return the partition column ``partition_key``."""
        return _as_uuid(self._get_raw("partition_key"))

    @partition_key.setter
    def partition_key(self, value: UUID) -> None:
        """Set the partition column."""
        BaseEntity.__setattr__(self, "partition_key", value)

    @property
    def job_id(self) -> UUID:
        """Return the source ``job_id``."""
        return _as_uuid(self._get_raw("job_id"))

    @job_id.setter
    def job_id(self, value: UUID) -> None:
        """Set the source job_id."""
        BaseEntity.__setattr__(self, "job_id", value)

    @property
    def scheduled_fire_at(self) -> datetime:
        """Return the originally scheduled fire time."""
        value: datetime = self._get_raw("scheduled_fire_at")
        return value

    @scheduled_fire_at.setter
    def scheduled_fire_at(self, value: datetime) -> None:
        """Set the scheduled_fire_at timestamp."""
        BaseEntity.__setattr__(self, "scheduled_fire_at", value)

    @property
    def actual_fired_at(self) -> datetime:
        """Return the actual fire time (post-drift)."""
        value: datetime = self._get_raw("actual_fired_at")
        return value

    @actual_fired_at.setter
    def actual_fired_at(self, value: datetime) -> None:
        """Set the actual_fired_at timestamp."""
        BaseEntity.__setattr__(self, "actual_fired_at", value)

    @property
    def status(self) -> str:
        """Return the terminal status (see
        :data:`~threetears.scheduled_jobs.types.ScheduleFireStatus`)."""
        value: str = self._get_raw("status")
        return value

    @status.setter
    def status(self, value: str) -> None:
        """Set the status (validated by DB CHECK)."""
        BaseEntity.__setattr__(self, "status", value)

    @property
    def output(self) -> dict[str, Any] | None:
        """Return the captured output payload (``None`` when not captured)."""
        value = self._get_raw("output")
        if value is None:
            return None
        return dict(value)

    @output.setter
    def output(self, value: dict[str, Any] | None) -> None:
        """Set the captured output payload."""
        BaseEntity.__setattr__(self, "output", None if value is None else dict(value))

    @property
    def latency_ms(self) -> int | None:
        """Return the end-to-end fire latency in ms (``None`` on dispatch failure)."""
        value: int | None = self._get_raw("latency_ms")
        return value

    @latency_ms.setter
    def latency_ms(self, value: int | None) -> None:
        """Set the latency."""
        BaseEntity.__setattr__(self, "latency_ms", value)

    @property
    def error(self) -> str | None:
        """Return the captured error message (``None`` for non-failed fires)."""
        value: str | None = self._get_raw("error")
        return value

    @error.setter
    def error(self, value: str | None) -> None:
        """Set the error message."""
        BaseEntity.__setattr__(self, "error", value)

    @property
    def date_created(self) -> datetime:
        """Return the row creation timestamp."""
        value: datetime = self._get_raw("date_created")
        return value

    @date_created.setter
    def date_created(self, value: datetime) -> None:
        """Set the creation timestamp."""
        BaseEntity.__setattr__(self, "date_created", value)

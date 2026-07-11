"""Intention entity -- thin cache proxy for a standing-want record.

:class:`IntentionEntity` covers the ``intentions`` table (migration
v001) with its composite primary key ``(agent_id, intention_id)``. It
mirrors :class:`threetears.agent.memory.entities.MemoryEntity`: the
constructor overwrites ``_id`` with the declared-order tuple so
:class:`BaseCollection`'s tuple-aware pk path addresses the row uniformly
across L1 / L2 / L3.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from threetears.core.entities.base import BaseEntity

__all__ = [
    "IntentionEntity",
]


def _as_uuid(value: object) -> UUID:
    """Coerce a value to :class:`UUID`, handling strings from cache tiers."""
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _as_uuid_or_none(value: object) -> UUID | None:
    """Coerce to :class:`UUID`, tolerating ``None`` for nullable columns.

    ``customer_id`` / ``user_id`` are nullable scope grains and
    ``source_memory_id`` / ``source_conversation_id`` are nullable soft
    refs; a bare :func:`_as_uuid` would raise on ``UUID("None")``.
    """
    if value is None:
        return None
    return _as_uuid(value)


class IntentionEntity(BaseEntity):
    """Cache proxy entity for the ``intentions`` table (v001).

    The composite primary key is ``(agent_id, intention_id)``; the
    constructor sets ``_id`` to that tuple so
    :meth:`BaseCollection.normalize_pk` and :meth:`BaseCollection.l2_key`
    address the row uniformly across tiers.

    An intention is agent-authored deliberation output: ``content`` is
    the want text, ``status`` walks the :class:`~threetears.agent.
    intention.types.IntentionStatus` value set, ``salience`` reuses the
    memory decay substrate (seed 0.5), ``last_surfaced_at`` anchors the
    read-path cooldown, and ``source_memory_id`` / ``source_conversation_id``
    are soft-ref provenance (no FK).
    """

    primary_key_field: str = "intention_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """initialize entity with composite ``_id`` for composite-pk lookup.

        :param data: row dict; must carry ``agent_id`` and ``intention_id``
        :ptype data: dict[str, Any]
        :param is_new: whether entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        if "agent_id" in data and "intention_id" in data:
            object.__setattr__(
                self,
                "_id",
                (data["agent_id"], data["intention_id"]),
            )

    @property
    def intention_id(self) -> UUID:
        """Get the intention ID (second pk column)."""
        return _as_uuid(self._get_raw("intention_id"))

    @property
    def agent_id(self) -> UUID:
        """Get the agent ID (partition + first pk column)."""
        return _as_uuid(self._get_raw("agent_id"))

    @agent_id.setter
    def agent_id(self, value: UUID) -> None:
        """Set the agent ID."""
        BaseEntity.__setattr__(self, "agent_id", value)

    @property
    def customer_id(self) -> UUID | None:
        """Get the customer ID scope grain (nullable).

        ``None`` on an agent-scoped want. metallm always sets it; the
        3tears primitive tolerates the null grain.
        """
        return _as_uuid_or_none(self._get_raw("customer_id"))

    @customer_id.setter
    def customer_id(self, value: UUID | None) -> None:
        """Set the customer ID."""
        BaseEntity.__setattr__(self, "customer_id", value)

    @property
    def user_id(self) -> UUID | None:
        """Get the owning user ID (nullable soft ref).

        ``None`` = agent-internal / global want. metallm MUST set it and
        filter every read on it -- ``user_id`` is the isolation boundary
        because every metallm user shares one ``agent_id``.
        """
        return _as_uuid_or_none(self._get_raw("user_id"))

    @user_id.setter
    def user_id(self, value: UUID | None) -> None:
        """Set the owning user ID."""
        BaseEntity.__setattr__(self, "user_id", value)

    @property
    def status(self) -> str:
        """Get the lifecycle status (an :class:`IntentionStatus` value)."""
        value: str = self._get_raw("status")
        return value

    @status.setter
    def status(self, value: str) -> None:
        """Set the lifecycle status."""
        BaseEntity.__setattr__(self, "status", value)

    @property
    def content(self) -> str:
        """Get the want text."""
        value: str = self._get_raw("content")
        return value

    @content.setter
    def content(self, value: str) -> None:
        """Set the want text."""
        BaseEntity.__setattr__(self, "content", value)

    @property
    def embedding(self) -> list[float]:
        """Get the embedding vector (used for log-time dedup)."""
        value: list[float] = self._get_raw("embedding")
        return value

    @embedding.setter
    def embedding(self, value: list[float]) -> None:
        """Set the embedding vector."""
        BaseEntity.__setattr__(self, "embedding", value)

    @property
    def salience(self) -> float:
        """Stored, decayed ranking weight. Defaults to the seed (0.5).

        NUMERIC(5,4) in the DB (asyncpg yields ``Decimal``); exposed as
        ``float``. On an unsaved entity that has not set the column, the
        DB server default (0.5) applies on write, so the getter mirrors
        that seed rather than raising.
        """
        raw = self._get_raw("salience")
        return float(raw) if raw is not None else 0.5

    @salience.setter
    def salience(self, value: float) -> None:
        """Set the salience weight."""
        BaseEntity.__setattr__(self, "salience", value)

    @property
    def last_decayed_at(self) -> datetime | None:
        """Timestamp of the last decay pass; the decay anchor."""
        value: datetime | None = self._get_raw("last_decayed_at")
        return value

    @last_decayed_at.setter
    def last_decayed_at(self, value: datetime | None) -> None:
        """Set the last-decayed timestamp."""
        BaseEntity.__setattr__(self, "last_decayed_at", value)

    @property
    def last_surfaced_at(self) -> datetime | None:
        """Timestamp the want was last surfaced; the cooldown anchor."""
        value: datetime | None = self._get_raw("last_surfaced_at")
        return value

    @last_surfaced_at.setter
    def last_surfaced_at(self, value: datetime | None) -> None:
        """Set the last-surfaced timestamp."""
        BaseEntity.__setattr__(self, "last_surfaced_at", value)

    @property
    def source_memory_id(self) -> UUID | None:
        """Get the provenance memory soft ref (no FK), or ``None``."""
        return _as_uuid_or_none(self._get_raw("source_memory_id"))

    @source_memory_id.setter
    def source_memory_id(self, value: UUID | None) -> None:
        """Set the provenance memory soft ref."""
        BaseEntity.__setattr__(self, "source_memory_id", value)

    @property
    def source_conversation_id(self) -> UUID | None:
        """Get the provenance conversation soft ref (no FK), or ``None``."""
        return _as_uuid_or_none(self._get_raw("source_conversation_id"))

    @source_conversation_id.setter
    def source_conversation_id(self, value: UUID | None) -> None:
        """Set the provenance conversation soft ref."""
        BaseEntity.__setattr__(self, "source_conversation_id", value)

    @property
    def date_created(self) -> datetime:
        """Get the creation timestamp."""
        value: datetime = self._get_raw("date_created")
        return value

    @property
    def date_updated(self) -> datetime | None:
        """Get the last-updated timestamp (the CAS fence)."""
        value: datetime | None = self._get_raw("date_updated")
        return value

    @date_updated.setter
    def date_updated(self, value: datetime | None) -> None:
        """Set the last-updated timestamp."""
        BaseEntity.__setattr__(self, "date_updated", value)

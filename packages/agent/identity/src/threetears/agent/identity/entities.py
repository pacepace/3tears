"""Identity-version entity -- thin cache proxy for one versioned identity block.

:class:`IdentityVersionEntity` covers the ``identity_versions`` table
(migration v001) with its composite primary key ``(agent_id,
version_id)``. It mirrors :class:`threetears.agent.intention.entities.
IntentionEntity`: the constructor overwrites ``_id`` with the
declared-order tuple so :class:`BaseCollection`'s tuple-aware pk path
addresses the row uniformly across L1 / L2 / L3.

A version is an immutable snapshot of one identity block at one point in
the linear chain -- ``content`` / ``rationale`` / ``content_hash`` /
``parent_version_id`` / ``block_key`` / ``proposer_agent_id`` are
write-once (getters only). Only the lifecycle fields mutate:
``status`` (proposed -> active -> superseded / rejected),
``consenter_user_id`` (set at consent), and ``date_updated`` (the CAS
fence).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from threetears.core.entities.base import BaseEntity

__all__ = [
    "IdentityVersionEntity",
]


def _as_uuid(value: object) -> UUID:
    """Coerce a value to :class:`UUID`, handling strings from cache tiers."""
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _as_uuid_or_none(value: object) -> UUID | None:
    """Coerce to :class:`UUID`, tolerating ``None`` for nullable columns."""
    if value is None:
        return None
    return _as_uuid(value)


class IdentityVersionEntity(BaseEntity):
    """Cache proxy entity for the ``identity_versions`` table (v001).

    The composite primary key is ``(agent_id, version_id)``; the
    constructor sets ``_id`` to that tuple so
    :meth:`BaseCollection.normalize_pk` and :meth:`BaseCollection.l2_key`
    address the row uniformly across tiers.
    """

    primary_key_field: str = "version_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """initialize entity with composite ``_id`` for composite-pk lookup.

        :param data: row dict; must carry ``agent_id`` and ``version_id``
        :ptype data: dict[str, Any]
        :param is_new: whether entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        if "agent_id" in data and "version_id" in data:
            object.__setattr__(self, "_id", (data["agent_id"], data["version_id"]))

    @property
    def version_id(self) -> UUID:
        """Get the version ID (second pk column)."""
        return _as_uuid(self._get_raw("version_id"))

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
        """Get the customer ID scope grain (nullable)."""
        return _as_uuid_or_none(self._get_raw("customer_id"))

    @customer_id.setter
    def customer_id(self, value: UUID | None) -> None:
        """Set the customer ID."""
        BaseEntity.__setattr__(self, "customer_id", value)

    @property
    def user_id(self) -> UUID | None:
        """Get the owning user ID (nullable soft ref; the isolation boundary)."""
        return _as_uuid_or_none(self._get_raw("user_id"))

    @user_id.setter
    def user_id(self, value: UUID | None) -> None:
        """Set the owning user ID."""
        BaseEntity.__setattr__(self, "user_id", value)

    @property
    def block_key(self) -> str:
        """Get the identity block key (an :class:`IdentityBlockKey` value; immutable)."""
        value: str = self._get_raw("block_key")
        return value

    @property
    def content(self) -> str:
        """Get this version's block content (immutable snapshot)."""
        value: str = self._get_raw("content")
        return value

    @property
    def rationale(self) -> str | None:
        """Get why this version was proposed (nullable; immutable)."""
        value: str | None = self._get_raw("rationale")
        return value

    @property
    def content_hash(self) -> str:
        """Get the sha256 of ``content`` (dedup + integrity; immutable)."""
        value: str = self._get_raw("content_hash")
        return value

    @property
    def parent_version_id(self) -> UUID | None:
        """Get the version this one supersedes (linear lineage; ``None`` = root; immutable)."""
        return _as_uuid_or_none(self._get_raw("parent_version_id"))

    @property
    def status(self) -> str:
        """Get the lifecycle status (an :class:`IdentityVersionStatus` value)."""
        value: str = self._get_raw("status")
        return value

    @status.setter
    def status(self, value: str) -> None:
        """Set the lifecycle status."""
        BaseEntity.__setattr__(self, "status", value)

    @property
    def proposer_agent_id(self) -> UUID | None:
        """Get the proposing agent (``None`` = user-authored / import; immutable)."""
        return _as_uuid_or_none(self._get_raw("proposer_agent_id"))

    @property
    def consenter_user_id(self) -> UUID | None:
        """Get the consenting user (set at apply; ``None`` until consented)."""
        return _as_uuid_or_none(self._get_raw("consenter_user_id"))

    @consenter_user_id.setter
    def consenter_user_id(self, value: UUID | None) -> None:
        """Set the consenting user."""
        BaseEntity.__setattr__(self, "consenter_user_id", value)

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
